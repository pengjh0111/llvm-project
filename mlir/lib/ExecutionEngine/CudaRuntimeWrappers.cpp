//===- CudaRuntimeWrappers.cpp - MLIR CUDA API wrapper library ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C wrappers around the CUDA library for easy linking in ORC jit.
// Also adds some debugging helpers that are helpful when writing MLIR code to
// run on GPUs.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <mutex>
#include <cmath>
#include <unordered_map>
#include <stdio.h>
#include <memory>
#include <queue>
#include <vector>
#include <atomic>
#include <unordered_set>

#include <cudnn.h>
#include <cublas_v2.h>
#include "/data/dagongcheng/pjhtest/llvm-latest/llvm-project/mlir/lib/ExecutionEngine/SNN_kernel.h"

#include "cuda.h"
#include "cuda_bf16.h"
#include "cuda_fp16.h"

#ifdef MLIR_ENABLE_CUDA_CUSPARSE
#include "cusparse.h"
#ifdef MLIR_ENABLE_CUDA_CUSPARSELT
#include "cusparseLt.h"
#endif // MLIR_ENABLE_CUDA_CUSPARSELT
#endif // MLIR_ENABLE_CUDA_CUSPARSE

#ifdef _WIN32
#include <malloc.h>
#define MLIR_CUDA_WRAPPERS_EXPORT __declspec(dllexport)
#else
#define MLIR_CUDA_WRAPPERS_EXPORT __attribute__((visibility("default")))
#endif // _WIN32

#define CUDA_REPORT_IF_ERROR(expr)                                             \
  [](CUresult result) {                                                        \
    if (!result)                                                               \
      return;                                                                  \
    const char *name = nullptr;                                                \
    cuGetErrorName(result, &name);                                             \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
  }(expr)

#define CUSPARSE_REPORT_IF_ERROR(expr)                                         \
  {                                                                            \
    cusparseStatus_t status = (expr);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "cuSPARSE '%s' failed with '%s'\n", #expr,               \
              cusparseGetErrorString(status));                                 \
    }                                                                          \
  }

thread_local static int32_t defaultDevice = 0;

const char *kDebugEnvironmentVariable = "MLIR_CUDA_DEBUG";

/// Helper method that checks environment value for debugging.
bool isDebugEnabled() {
  static bool isInitialized = false;
  static bool isEnabled = false;
  if (!isInitialized)
    isEnabled = getenv(kDebugEnvironmentVariable) != nullptr;
  return isEnabled;
}

#define debug_print(fmt, ...)                                                  \
  do {                                                                         \
    if (isDebugEnabled())                                                      \
      fprintf(stderr, "%s:%d:%s(): " fmt, "CudaRuntimeWrappers.cpp", __LINE__, \
              __func__, __VA_ARGS__);                                          \
  } while (0)

// Returns default CUdevice
CUdevice getDefaultCuDevice() {
  CUdevice device;
  CUDA_REPORT_IF_ERROR(cuDeviceGet(&device, /*ordinal=*/defaultDevice));
  return device;
}

// Make the primary context of the current default device current for the
// duration
//  of the instance and restore the previous context on destruction.
class ScopedContext {
public:
  ScopedContext() {
    // Static reference to CUDA primary context for device ordinal
    // defaultDevice.
    static CUcontext context = [] {
      CUDA_REPORT_IF_ERROR(cuInit(/*flags=*/0));
      CUcontext ctx;
      // Note: this does not affect the current context.
      CUDA_REPORT_IF_ERROR(
          cuDevicePrimaryCtxRetain(&ctx, getDefaultCuDevice()));
      return ctx;
    }();

    CUDA_REPORT_IF_ERROR(cuCtxPushCurrent(context));
  }

  ~ScopedContext() { CUDA_REPORT_IF_ERROR(cuCtxPopCurrent(nullptr)); }
};

#ifdef MLIR_ENABLE_CUDA_CUSPARSE
// Note that (1) Nvidia confirms the safety to share handle across multiple
// instances, and streams. (2) Clients are responsible to call the @mgpu
// environment initialization/destruction in a thread-safe manner, e.g.,
// at the beginning of the program before multi-threads are created.
static cusparseHandle_t cusparse_env = nullptr;

#ifdef MLIR_ENABLE_CUDA_CUSPARSELT
// cusparseLtHandle_t is not a pointer type, so we need an additional flag to
// indicate whether it is initialized.
static cusparseLtHandle_t cusparseLt_env;
static bool cusparseLt_initiated = false;

#endif // MLIR_ENABLE_CUDA_CUSPARSELT
#endif // MLIR_ENABLE_CUDA_CUSPARSE

// extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUmodule
// mgpuModuleLoad(void *data, size_t /*gpuBlobSize*/) {
//   ScopedContext scopedContext;
//   // static CUmodule module = nullptr;  // 使用static保存模块，只在函数首次调用时执行一次
//   // if (module) return module;         // 如果已加载过，直接返回缓存的模块
//   CUmodule module = nullptr;
//   CUDA_REPORT_IF_ERROR(cuModuleLoadData(&module, data));
//   return module;
// }


//===----------------------------------------------------------------------===//
// 全局CUDA上下文管理
//===----------------------------------------------------------------------===//

// 全局CUDA上下文
static CUcontext g_global_context = nullptr;
static std::mutex g_context_mutex;

// 确保全局上下文存在并激活
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuEnsureContext() {
  std::lock_guard<std::mutex> lock(g_context_mutex);
  
  // 添加CUDA初始化
  static bool cuda_initialized = false;
  if (!cuda_initialized) {
    CUDA_REPORT_IF_ERROR(cuInit(0));
    cuda_initialized = true;
  }  

  // 检查当前上下文
  CUcontext current = nullptr;
  CUDA_REPORT_IF_ERROR(cuCtxGetCurrent(&current));
  
  // 如果没有当前上下文或与全局上下文不同，则设置全局上下文
  if (current == nullptr) {
    if (g_global_context == nullptr) {
      // 创建新的上下文
      CUDA_REPORT_IF_ERROR(cuCtxCreate(&g_global_context, 0, 0));
      // fprintf(stderr, "[CONTEXT] Created new global context: %p\n", g_global_context);
    }
    // 设置为当前上下文
    CUDA_REPORT_IF_ERROR(cuCtxSetCurrent(g_global_context));
    // fprintf(stderr, "[CONTEXT] Set global context as current: %p\n", g_global_context);
  } else if (g_global_context == nullptr) {
    // 如果有当前上下文但全局上下文为空，则使用当前上下文作为全局上下文
    g_global_context = current;
    // fprintf(stderr, "[CONTEXT] Adopted current context as global: %p\n", g_global_context);
  } else if (current != g_global_context) {
    // 如果当前上下文与全局上下文不同，则设置全局上下文为当前上下文
    CUDA_REPORT_IF_ERROR(cuCtxSetCurrent(g_global_context));
    // fprintf(stderr, "[CONTEXT] Switched from context %p to global context: %p\n", 
    //         current, g_global_context);
  }

  // 最后打印当前上下文
  CUcontext final_context = nullptr;
  cuCtxGetCurrent(&final_context);
  // fprintf(stderr, "[CONTEXT-FINAL] Current context: %p\n", final_context);
}

// 获取全局上下文
extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUcontext mgpuGetGlobalContext() {
  mgpuEnsureContext();
  return g_global_context;
}

// 清理全局上下文（在程序结束时调用）
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuCleanupContext() {
  std::lock_guard<std::mutex> lock(g_context_mutex);
  
  if (g_global_context != nullptr) {
    CUDA_REPORT_IF_ERROR(cuCtxDestroy(g_global_context));
    g_global_context = nullptr;
    // fprintf(stderr, "[CONTEXT] Destroyed global context\n");
  }
}

// 添加到错误报告宏部分
#define CUDNN_REPORT_IF_ERROR(expr)                                         \
  {                                                                          \
    cudnnStatus_t status = (expr);                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "cuDNN '%s' failed with '%s'\n", #expr,               \
              cudnnGetErrorString(status));                                 \
    }                                                                       \
  }

#define CUBLAS_REPORT_IF_ERROR(expr)                                        \
  {                                                                          \
    cublasStatus_t status = (expr);                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "cuBLAS '%s' failed with status %d\n", #expr,          \
              status);                                                       \
    }                                                                        \
  }

//===----------------------------------------------------------------------===//
// 优化的Handle管理 - 分离创建与使用
//===----------------------------------------------------------------------===//

// Handle组结构体
struct StreamHandles {
  cudnnHandle_t cudnn_handle;
  cublasHandle_t cublas_handle;
  bool valid;
  
  StreamHandles() : cudnn_handle(nullptr), cublas_handle(nullptr), valid(false) {}
};

// 全局handle映射表 - 使用更具体的名字避免冲突
static std::mutex g_mgpu_handle_registry_mutex;
static std::unordered_map<CUstream, StreamHandles> g_mgpu_stream_handle_registry;

// 为指定stream创建并绑定handle组
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuCreateHandlesForStream(CUstream stream) {
  mgpuEnsureContext();
  
  std::lock_guard<std::mutex> lock(g_mgpu_handle_registry_mutex);
  
  // 检查是否已经存在
  if (g_mgpu_stream_handle_registry.find(stream) != g_mgpu_stream_handle_registry.end()) {
    fprintf(stderr, "[HANDLE] Handles already exist for stream %p\n", stream);
    // return true; // 已存在，直接返回成功
  }
  
  StreamHandles handles;
  
  // 创建cuDNN句柄
  cudnnStatus_t cudnn_status = cudnnCreate(&handles.cudnn_handle);
  if (cudnn_status != CUDNN_STATUS_SUCCESS) {
    fprintf(stderr, "[HANDLE] Failed to create cuDNN handle: %s\n", 
            cudnnGetErrorString(cudnn_status));
    // return false;
  }
  
  // 创建cuBLAS句柄
  cublasStatus_t cublas_status = cublasCreate(&handles.cublas_handle);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "[HANDLE] Failed to create cuBLAS handle: %d\n", cublas_status);
    cudnnDestroy(handles.cudnn_handle);
    // return false;
  }
  
  // 绑定到指定的stream
  CUDNN_REPORT_IF_ERROR(cudnnSetStream(handles.cudnn_handle, stream));
  CUBLAS_REPORT_IF_ERROR(cublasSetStream(handles.cublas_handle, stream));
  
  handles.valid = true;
  
  // 存储到映射表
  g_mgpu_stream_handle_registry[stream] = handles;
  
  // fprintf(stderr, "[HANDLE] Created handle group for stream %p: cuDNN=%p, cuBLAS=%p\n", 
  //         stream, handles.cudnn_handle, handles.cublas_handle);
  
  // return true;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuDestroyHandlesForStream(CUstream stream) {
  mgpuEnsureContext();
  
  std::lock_guard<std::mutex> lock(g_mgpu_handle_registry_mutex);
  
  auto it = g_mgpu_stream_handle_registry.find(stream);
  if (it != g_mgpu_stream_handle_registry.end()) {
    if (it->second.valid) {
      if (it->second.cudnn_handle != nullptr) {
        CUDNN_REPORT_IF_ERROR(cudnnDestroy(it->second.cudnn_handle));
      }
      if (it->second.cublas_handle != nullptr) {
        CUBLAS_REPORT_IF_ERROR(cublasDestroy(it->second.cublas_handle));
      }
      
      // fprintf(stderr, "[HANDLE] Destroyed handle group for stream %p\n", stream);
    }
    g_mgpu_stream_handle_registry.erase(it);
  }
}

// static bool getHandlesForStream(CUstream stream, StreamHandles& handles) {
//   std::lock_guard<std::mutex> lock(g_mgpu_handle_registry_mutex);
  
//   auto it = g_mgpu_stream_handle_registry.find(stream);
//   if (it != g_mgpu_stream_handle_registry.end() && it->second.valid) {
//     handles = it->second;
//     return true;
//   }
  
//   fprintf(stderr, "[HANDLE] ERROR: No handles found for stream %p. "
//                   "Call mgpuCreateHandlesForStream() first!\n", stream);
//   return false;
// }

// 流到cuBLAS句柄的映射
static std::mutex g_cublas_handles_mutex;
static std::unordered_map<CUstream, cublasHandle_t> g_stream_cublas_handles;

// 为流获取或创建cuBLAS句柄
extern "C" MLIR_CUDA_WRAPPERS_EXPORT cublasHandle_t mgpuCublasGetHandle(CUstream stream) {
  
  // 首先确保我们在正确的上下文中
  mgpuEnsureContext();
  
  std::lock_guard<std::mutex> lock(g_cublas_handles_mutex);
  
  // 检查是否已经有这个流的句柄
  auto it = g_stream_cublas_handles.find(stream);
  if (it != g_stream_cublas_handles.end()) {
    return it->second;
  }
  
  // 创建新句柄并关联到此流
  cublasHandle_t handle = nullptr;
  CUBLAS_REPORT_IF_ERROR(cublasCreate(&handle));
  CUBLAS_REPORT_IF_ERROR(cublasSetStream(handle, stream));
  
  // 存储并返回新句柄
  g_stream_cublas_handles[stream] = handle;
  // fprintf(stderr, "[HANDLE] Created new cuBLAS handle: %p for stream: %p\n", 
  //   handle, stream);

  return handle;
}

// 销毁特定流的cuBLAS句柄
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuCublasDestroyHandle(CUstream stream) {
  
  // 确保在正确上下文中
  mgpuEnsureContext();
  
  std::lock_guard<std::mutex> lock(g_cublas_handles_mutex);
  
  auto it = g_stream_cublas_handles.find(stream);
  if (it != g_stream_cublas_handles.end()) {
    CUBLAS_REPORT_IF_ERROR(cublasDestroy(it->second));
    g_stream_cublas_handles.erase(it);
    // fprintf(stderr, "[HANDLE] Destroyed cuBLAS handle for stream: %p\n", stream);
  }
}

// 清理所有cuBLAS句柄
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuCublasCleanup() {
  
  // 确保在正确上下文中
  mgpuEnsureContext();

  std::lock_guard<std::mutex> lock(g_cublas_handles_mutex);
  
  for (auto& pair : g_stream_cublas_handles) {
    CUBLAS_REPORT_IF_ERROR(cublasDestroy(pair.second));
  }
  g_stream_cublas_handles.clear();
}

// 流到cuDNN句柄的映射
static std::mutex g_handles_mutex;
static std::unordered_map<CUstream, cudnnHandle_t> g_stream_handles;

// 为流获取或创建cuDNN句柄
extern "C" MLIR_CUDA_WRAPPERS_EXPORT cudnnHandle_t mgpuCudnnGetHandle(CUstream stream) {
  
  // 首先确保我们在正确的上下文中
  mgpuEnsureContext();
  // ScopedContext scopedContext;
  
  std::lock_guard<std::mutex> lock(g_handles_mutex);
  
  // 检查是否已经有这个流的句柄
  auto it = g_stream_handles.find(stream);
  if (it != g_stream_handles.end()) {
    return it->second;
  }
  
  // 创建新句柄并关联到此流
  cudnnHandle_t handle = nullptr;
  CUDNN_REPORT_IF_ERROR(cudnnCreate(&handle));
  CUDNN_REPORT_IF_ERROR(cudnnSetStream(handle, stream));
  
  // 存储并返回新句柄
  g_stream_handles[stream] = handle;
  // fprintf(stderr, "[HANDLE] Created new cuDNN handle: %p for stream: %p\n", 
  //   handle, stream);

  return handle;
}

// 销毁特定流的cuDNN句柄
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuCudnnDestroyHandle(CUstream stream) {
  
  // 确保在正确上下文中
  mgpuEnsureContext();
  
  std::lock_guard<std::mutex> lock(g_handles_mutex);
  
  auto it = g_stream_handles.find(stream);
  if (it != g_stream_handles.end()) {
    CUDNN_REPORT_IF_ERROR(cudnnDestroy(it->second));
    g_stream_handles.erase(it);
    // fprintf(stderr, "[HANDLE] Destroyed cuDNN handle for stream: %p\n", stream);
  }
}

// 清理所有cuDNN句柄
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuCudnnCleanup() {
  
  // 确保在正确上下文中
  mgpuEnsureContext();

  std::lock_guard<std::mutex> lock(g_handles_mutex);
  
  for (auto& pair : g_stream_handles) {
    CUDNN_REPORT_IF_ERROR(cudnnDestroy(pair.second));
  }
  g_stream_handles.clear();
}

//===----------------------------------------------------------------------===//
// 1. Handle池数据结构定义
//===----------------------------------------------------------------------===//

struct PooledHandle {
  cudnnHandle_t cudnn_handle;
  cublasHandle_t cublas_handle;
  bool in_use;
  int pool_index;
  
  PooledHandle() : cudnn_handle(nullptr), cublas_handle(nullptr), 
                  in_use(false), pool_index(-1) {}
};

// 全局Handle池状态
static std::vector<PooledHandle> g_handle_pool;                     // 所有handle的存储
static std::queue<int> g_available_handle_indices;                  // 可用handle索引队列
static std::unordered_map<CUstream, int> g_stream_to_handle_index;  // stream到handle的映射
static std::mutex g_handle_pool_mutex;                              // 线程安全锁
static bool g_handle_pool_initialized = false;                      // 初始化标志
static std::atomic<int> g_active_handle_count{0}; 

//===----------------------------------------------------------------------===//
// 2. Handle池初始化和销毁
//===----------------------------------------------------------------------===//

/**
 * 初始化Handle池
 * @param pool_size 池中handle的数量
 */
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuInitHandlePool(int pool_size) {
  mgpuEnsureContext();
  
  std::lock_guard<std::mutex> lock(g_handle_pool_mutex);
  
  if (g_handle_pool_initialized) {
    fprintf(stderr, "[HANDLE POOL] Already initialized with %d handles\n", 
            (int)g_handle_pool.size());
    return;
  }
  
  if (pool_size <= 0) {
    fprintf(stderr, "[HANDLE POOL] ERROR: Invalid pool size: %d\n", pool_size);
    return;
  }
  
  fprintf(stderr, "[HANDLE POOL] Initializing pool with %d handles...\n", pool_size);
  
  // 预分配存储空间
  g_handle_pool.reserve(pool_size);
  g_handle_pool.resize(pool_size);
  
  // 创建所有handles
  for (int i = 0; i < pool_size; i++) {
    PooledHandle& handle = g_handle_pool[i];
    handle.pool_index = i;
    
    // 创建cuDNN handle
    cudnnStatus_t cudnn_status = cudnnCreate(&handle.cudnn_handle);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
      fprintf(stderr, "[HANDLE POOL] FATAL: Failed to create cuDNN handle %d: %s\n", 
              i, cudnnGetErrorString(cudnn_status));
      
      // 清理已创建的handles
      for (int j = 0; j < i; j++) {
        if (g_handle_pool[j].cudnn_handle) {
          cudnnDestroy(g_handle_pool[j].cudnn_handle);
        }
        if (g_handle_pool[j].cublas_handle) {
          cublasDestroy(g_handle_pool[j].cublas_handle);
        }
      }
      g_handle_pool.clear();
      return;
    }
    
    // 创建cuBLAS handle
    cublasStatus_t cublas_status = cublasCreate(&handle.cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "[HANDLE POOL] FATAL: Failed to create cuBLAS handle %d: status=%d\n", 
              i, cublas_status);
      
      // 清理当前handle的cuDNN部分
      cudnnDestroy(handle.cudnn_handle);
      
      // 清理已创建的handles
      for (int j = 0; j < i; j++) {
        if (g_handle_pool[j].cudnn_handle) {
          cudnnDestroy(g_handle_pool[j].cudnn_handle);
        }
        if (g_handle_pool[j].cublas_handle) {
          cublasDestroy(g_handle_pool[j].cublas_handle);
        }
      }
      g_handle_pool.clear();
      return;
    }
    
    // 将索引加入可用队列
    g_available_handle_indices.push(i);
    
    // fprintf(stderr, "[HANDLE POOL] Created handle %d: cuDNN=%p, cuBLAS=%p\n", 
    //         i, handle.cudnn_handle, handle.cublas_handle);
  }
  
  g_handle_pool_initialized = true;
  g_active_handle_count = 0;
  
  fprintf(stderr, "[HANDLE POOL] Successfully initialized %d handles\n", pool_size);
}

/**
 * 销毁Handle池
 */
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuDestroyHandlePool() {
  mgpuEnsureContext();
  
  std::lock_guard<std::mutex> lock(g_handle_pool_mutex);
  
  if (!g_handle_pool_initialized) {
    fprintf(stderr, "[HANDLE POOL] Pool not initialized, nothing to destroy\n");
    return;
  }
  
  fprintf(stderr, "[HANDLE POOL] Destroying pool with %d handles...\n", 
          (int)g_handle_pool.size());
  
  // 检查是否还有活跃的handles
  if (g_active_handle_count > 0) {
    fprintf(stderr, "[HANDLE POOL] WARNING: %d handles still in use during destruction\n", 
            g_active_handle_count.load());
  }
  
  // 销毁所有handles
  int destroyed_count = 0;
  for (size_t i = 0; i < g_handle_pool.size(); i++) {
    PooledHandle& handle = g_handle_pool[i];
    
    if (handle.cublas_handle) {
      cublasStatus_t status = cublasDestroy(handle.cublas_handle);
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[HANDLE POOL] Error destroying cuBLAS handle %d: %d\n", 
                (int)i, status);
      }
      handle.cublas_handle = nullptr;
    }
    
    if (handle.cudnn_handle) {
      cudnnStatus_t status = cudnnDestroy(handle.cudnn_handle);
      if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "[HANDLE POOL] Error destroying cuDNN handle %d: %s\n", 
                (int)i, cudnnGetErrorString(status));
      }
      handle.cudnn_handle = nullptr;
    }
    
    destroyed_count++;
  }
  
  // 清理所有数据结构
  g_handle_pool.clear();
  while (!g_available_handle_indices.empty()) {
    g_available_handle_indices.pop();
  }
  g_stream_to_handle_index.clear();
  g_active_handle_count = 0;
  g_handle_pool_initialized = false;
  
  fprintf(stderr, "[HANDLE POOL] Destroyed %d handles, pool cleanup complete\n", 
          destroyed_count);
}

//===----------------------------------------------------------------------===//
// 3. Handle获取函数 (替代mgpuCreateHandlesForStream)
//===----------------------------------------------------------------------===//

/**
 * 从池中获取handle并绑定到stream
 * 这个函数替代原来的mgpuCreateHandlesForStream
 * @param stream CUDA stream
 */
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuAcquirePooledHandles(CUstream stream) {
  mgpuEnsureContext();
  
  std::lock_guard<std::mutex> lock(g_handle_pool_mutex);
  
  if (!g_handle_pool_initialized) {
    fprintf(stderr, "[HANDLE POOL] FATAL: Pool not initialized. Call mgpuInitHandlePool() first!\n");
    return;
  }
  
  // 检查stream是否已经有handle分配
  if (g_stream_to_handle_index.find(stream) != g_stream_to_handle_index.end()) {
    fprintf(stderr, "[HANDLE POOL] WARNING: Stream %p already has handle assigned\n", stream);
    return;
  }
  
  // 检查是否有可用的handle
  if (g_available_handle_indices.empty()) {
    fprintf(stderr, "[HANDLE POOL] FATAL: No available handles! Active: %d, Total: %d\n",
            g_active_handle_count.load(), (int)g_handle_pool.size());
    return;
  }
  
  // 获取一个可用的handle
  int handle_index = g_available_handle_indices.front();
  g_available_handle_indices.pop();
  
  PooledHandle& handle = g_handle_pool[handle_index];
  
  // 绑定handle到stream
  cudnnStatus_t cudnn_status = cudnnSetStream(handle.cudnn_handle, stream);
  if (cudnn_status != CUDNN_STATUS_SUCCESS) {
    fprintf(stderr, "[HANDLE POOL] ERROR: Failed to set cuDNN stream for handle %d: %s\n", 
            handle_index, cudnnGetErrorString(cudnn_status));
    // 将handle放回可用队列
    g_available_handle_indices.push(handle_index);
    return;
  }
  
  cublasStatus_t cublas_status = cublasSetStream(handle.cublas_handle, stream);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "[HANDLE POOL] ERROR: Failed to set cuBLAS stream for handle %d: %d\n", 
            handle_index, cublas_status);
    // 重置cuDNN stream并将handle放回可用队列
    cudnnSetStream(handle.cudnn_handle, nullptr);
    g_available_handle_indices.push(handle_index);
    return;
  }
  
  // 更新状态
  handle.in_use = true;
  g_stream_to_handle_index[stream] = handle_index;
  g_active_handle_count++;
  
  // fprintf(stderr, "[HANDLE POOL] Acquired handle %d for stream %p (Active: %d/%d)\n", 
  //         handle_index, stream, g_active_handle_count.load(), (int)g_handle_pool.size());
}

//===----------------------------------------------------------------------===//
// 4. Handle退还函数 (替代mgpuDestroyHandlesForStream)
//===----------------------------------------------------------------------===//

/**
 * 将handle退还到池中
 * 这个函数替代原来的mgpuDestroyHandlesForStream
 * @param stream CUDA stream
 */
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuReleasePooledHandles(CUstream stream) {
  mgpuEnsureContext();
  
  std::lock_guard<std::mutex> lock(g_handle_pool_mutex);
  
  if (!g_handle_pool_initialized) {
    fprintf(stderr, "[HANDLE POOL] WARNING: Pool not initialized\n");
    return;
  }
  
  // 查找stream对应的handle
  auto it = g_stream_to_handle_index.find(stream);
  if (it == g_stream_to_handle_index.end()) {
    fprintf(stderr, "[HANDLE POOL] WARNING: No handle found for stream %p\n", stream);
    return;
  }
  
  int handle_index = it->second;
  PooledHandle& handle = g_handle_pool[handle_index];
  
  if (!handle.in_use) {
    fprintf(stderr, "[HANDLE POOL] WARNING: Handle %d is not marked as in use\n", handle_index);
    return;
  }
  
  // 解除stream绑定
  cudnnStatus_t cudnn_status = cudnnSetStream(handle.cudnn_handle, nullptr);
  if (cudnn_status != CUDNN_STATUS_SUCCESS) {
    fprintf(stderr, "[HANDLE POOL] WARNING: Failed to reset cuDNN stream for handle %d: %s\n", 
            handle_index, cudnnGetErrorString(cudnn_status));
  }
  
  cublasStatus_t cublas_status = cublasSetStream(handle.cublas_handle, nullptr);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "[HANDLE POOL] WARNING: Failed to reset cuBLAS stream for handle %d: %d\n", 
            handle_index, cublas_status);
  }
  
  // 更新状态
  handle.in_use = false;
  g_available_handle_indices.push(handle_index);
  g_stream_to_handle_index.erase(it);
  g_active_handle_count--;
  
  // fprintf(stderr, "[HANDLE POOL] Released handle %d from stream %p (Active: %d/%d)\n", 
  //         handle_index, stream, g_active_handle_count.load(), (int)g_handle_pool.size());
}

//===----------------------------------------------------------------------===//
// 5. 获取Handle用于计算 (修改原有的getHandlesForStream函数)
//===----------------------------------------------------------------------===//

// 从池中获取handle用于计算操作
static bool getPooledHandlesForStream(CUstream stream, StreamHandles& result_handles) {
  std::lock_guard<std::mutex> lock(g_handle_pool_mutex);
  
  auto it = g_stream_to_handle_index.find(stream);
  if (it == g_stream_to_handle_index.end()) {
    fprintf(stderr, "[HANDLE POOL] ERROR: No handle assigned to stream %p. "
                    "Call mgpuAcquirePooledHandles() first!\n", stream);
    return false;
  }
  
  int handle_index = it->second;
  PooledHandle& handle = g_handle_pool[handle_index];
  
  if (!handle.in_use) {
    fprintf(stderr, "[HANDLE POOL] ERROR: Handle %d is not marked as in use\n", handle_index);
    return false;
  }
  
  // 填充返回结果
  result_handles.cudnn_handle = handle.cudnn_handle;
  result_handles.cublas_handle = handle.cublas_handle;
  result_handles.valid = true;
  
  return true;
}

// 修改原有的getHandlesForStream函数以支持池
static bool getHandlesForStream(CUstream stream, StreamHandles& handles) {
  // 优先从池中获取
  if (g_handle_pool_initialized) {
    return getPooledHandlesForStream(stream, handles);
  }
  
  // 如果池未初始化，回退到原有方式
  std::lock_guard<std::mutex> lock(g_mgpu_handle_registry_mutex);
  
  auto it = g_mgpu_stream_handle_registry.find(stream);
  if (it != g_mgpu_stream_handle_registry.end() && it->second.valid) {
    handles = it->second;
    return true;
  }
  
  fprintf(stderr, "[HANDLE] ERROR: No handles found for stream %p. "
                  "Use handle pool or call mgpuCreateHandlesForStream() first!\n", stream);
  return false;
}

//===----------------------------------------------------------------------===//
// Stream池数据结构定义
//===----------------------------------------------------------------------===//

struct PooledStream {
  CUstream stream;
  bool in_use;
  int pool_index;
  
  PooledStream() : stream(nullptr), in_use(false), pool_index(-1) {}
};

// 全局Stream池状态
static std::vector<PooledStream> g_stream_pool;                     // 所有stream的存储
static std::queue<int> g_available_stream_indices;                  // 可用stream索引队列
static std::mutex g_stream_pool_mutex;                              // 线程安全锁
static bool g_stream_pool_initialized = false;                      // 初始化标志
static std::atomic<int> g_active_stream_count{0};  

//===----------------------------------------------------------------------===//
// Stream池初始化和销毁
//===----------------------------------------------------------------------===//

/**
 * 初始化Stream池
 * @param pool_size 池中stream的数量
 */
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuInitStreamPool(int pool_size) {
  mgpuEnsureContext();
  
  std::lock_guard<std::mutex> lock(g_stream_pool_mutex);
  
  if (g_stream_pool_initialized) {
    fprintf(stderr, "[STREAM POOL] Already initialized with %d streams\n", 
            (int)g_stream_pool.size());
    return;
  }
  
  if (pool_size <= 0) {
    fprintf(stderr, "[STREAM POOL] ERROR: Invalid pool size: %d\n", pool_size);
    return;
  }
  
  fprintf(stderr, "[STREAM POOL] Initializing pool with %d streams...\n", pool_size);
  
  // 预分配存储空间
  g_stream_pool.reserve(pool_size);
  g_stream_pool.resize(pool_size);
  
  // 创建所有streams
  for (int i = 0; i < pool_size; i++) {
    PooledStream& pooled_stream = g_stream_pool[i];
    pooled_stream.pool_index = i;
    
    // 创建CUDA stream
    CUresult result = cuStreamCreate(&pooled_stream.stream, CU_STREAM_NON_BLOCKING);
    if (result != CUDA_SUCCESS) {
      const char *name = nullptr;
      cuGetErrorName(result, &name);
      if (!name) name = "<unknown>";
      
      fprintf(stderr, "[STREAM POOL] FATAL: Failed to create stream %d: %s\n", 
              i, name);
      
      // 清理已创建的streams
      for (int j = 0; j < i; j++) {
        if (g_stream_pool[j].stream) {
          cuStreamDestroy(g_stream_pool[j].stream);
        }
      }
      g_stream_pool.clear();
      return;
    }
    
    // 将索引加入可用队列
    g_available_stream_indices.push(i);
    
    // fprintf(stderr, "[STREAM POOL] Created stream %d: %p\n", 
    //         i, pooled_stream.stream);
  }
  
  g_stream_pool_initialized = true;
  g_active_stream_count = 0;
  
  fprintf(stderr, "[STREAM POOL] Successfully initialized %d streams\n", pool_size);
}

/**
 * 销毁Stream池
 */
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuDestroyStreamPool() {
  mgpuEnsureContext();
  
  std::lock_guard<std::mutex> lock(g_stream_pool_mutex);
  
  if (!g_stream_pool_initialized) {
    fprintf(stderr, "[STREAM POOL] Pool not initialized, nothing to destroy\n");
    return;
  }
  
  fprintf(stderr, "[STREAM POOL] Destroying pool with %d streams...\n", 
          (int)g_stream_pool.size());
  
  // 检查是否还有活跃的streams
  if (g_active_stream_count > 0) {
    fprintf(stderr, "[STREAM POOL] WARNING: %d streams still in use during destruction\n", 
            g_active_stream_count.load());
  }
  
  // 销毁所有streams
  int destroyed_count = 0;
  for (size_t i = 0; i < g_stream_pool.size(); i++) {
    PooledStream& pooled_stream = g_stream_pool[i];
    
    if (pooled_stream.stream) {
      // 如果stream还在使用中，先处理关联的资源
      if (pooled_stream.in_use) {
        fprintf(stderr, "[STREAM POOL] WARNING: Stream %d still in use, cleaning up...\n", (int)i);
        
        // 如果使用了handle pool，释放相关的handles
        if (g_handle_pool_initialized) {
          mgpuReleasePooledHandles(pooled_stream.stream);
        } else {
          // 否则使用原有方式清理
          mgpuDestroyHandlesForStream(pooled_stream.stream);
        }
      }
      
      CUresult result = cuStreamDestroy(pooled_stream.stream);
      if (result != CUDA_SUCCESS) {
        const char *name = nullptr;
        cuGetErrorName(result, &name);
        if (!name) name = "<unknown>";
        fprintf(stderr, "[STREAM POOL] Error destroying stream %d: %s\n", 
                (int)i, name);
      }
      pooled_stream.stream = nullptr;
    }
    
    destroyed_count++;
  }
  
  // 清理所有数据结构
  g_stream_pool.clear();
  while (!g_available_stream_indices.empty()) {
    g_available_stream_indices.pop();
  }
  g_active_stream_count = 0;
  g_stream_pool_initialized = false;
  
  fprintf(stderr, "[STREAM POOL] Destroyed %d streams, pool cleanup complete\n", 
          destroyed_count);
}

//===----------------------------------------------------------------------===//
// Stream获取和释放函数 (替代mgpuStreamCreate和mgpuStreamDestroy)
//===----------------------------------------------------------------------===//

/**
 * 从池中获取stream
 * 这个函数替代原来的mgpuStreamCreate
 * @return 获取的CUDA stream，如果失败返回nullptr
 */
extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUstream mgpuAcquirePooledStream() {
  mgpuEnsureContext();
  
  std::lock_guard<std::mutex> lock(g_stream_pool_mutex);
  
  if (!g_stream_pool_initialized) {
    fprintf(stderr, "[STREAM POOL] FATAL: Pool not initialized. Call mgpuInitStreamPool() first!\n");
    return nullptr;
  }
  
  // 检查是否有可用的stream
  if (g_available_stream_indices.empty()) {
    fprintf(stderr, "[STREAM POOL] FATAL: No available streams! Active: %d, Total: %d\n", 
            g_active_stream_count.load(), (int)g_stream_pool.size());
    return nullptr;
  }
  
  // 获取一个可用的stream
  int stream_index = g_available_stream_indices.front();
  g_available_stream_indices.pop();
  
  PooledStream& pooled_stream = g_stream_pool[stream_index];
  
  // 更新状态
  pooled_stream.in_use = true;
  g_active_stream_count++;
  
  // fprintf(stderr, "[STREAM POOL] Acquired stream %d: %p (Active: %d/%d)\n", 
  //         stream_index, pooled_stream.stream, g_active_stream_count.load(), (int)g_stream_pool.size());
  
  return pooled_stream.stream;
}

/**
 * 将stream退还到池中
 * 这个函数替代原来的mgpuStreamDestroy
 * @param stream 要释放的CUDA stream
 */
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuReleasePooledStream(CUstream stream) {
  if (stream == nullptr) {
    fprintf(stderr, "[STREAM POOL] WARNING: Attempting to release NULL stream\n");
    return;
  }
  
  mgpuEnsureContext();
  
  std::lock_guard<std::mutex> lock(g_stream_pool_mutex);
  
  if (!g_stream_pool_initialized) {
    fprintf(stderr, "[STREAM POOL] WARNING: Pool not initialized\n");
    return;
  }
  
  // 查找stream在池中的索引
  int stream_index = -1;
  for (size_t i = 0; i < g_stream_pool.size(); i++) {
    if (g_stream_pool[i].stream == stream) {
      stream_index = (int)i;
      break;
    }
  }
  
  if (stream_index == -1) {
    fprintf(stderr, "[STREAM POOL] WARNING: Stream %p not found in pool\n", stream);
    return;
  }
  
  PooledStream& pooled_stream = g_stream_pool[stream_index];
  
  if (!pooled_stream.in_use) {
    fprintf(stderr, "[STREAM POOL] WARNING: Stream %d is not marked as in use\n", stream_index);
    return;
  }
  
  // 如果使用了handle pool，释放相关的handles
  if (g_handle_pool_initialized) {
    // 检查是否有handle需要退还
    bool has_handle = false;
    {
      std::lock_guard<std::mutex> handle_lock(g_handle_pool_mutex);
      auto it = g_stream_to_handle_index.find(stream);
      has_handle = (it != g_stream_to_handle_index.end());
    }
    
    if (has_handle) {
      // fprintf(stderr, "[STREAM POOL] Stream %p has handle assigned, auto-releasing...\n", stream);
      mgpuReleasePooledHandles(stream);
    }
  } else {
    // 否则使用原有方式清理handles
    mgpuDestroyHandlesForStream(stream);
  }
  
  // 注意：不在这里同步stream，因为用户应该在释放前显式调用mgpuStreamSynchronize
  // 如果需要强制同步，可以取消下面的注释
  // CUresult sync_result = cuStreamSynchronize(stream);
  // if (sync_result != CUDA_SUCCESS) {
  //   const char *name = nullptr;
  //   cuGetErrorName(sync_result, &name);
  //   if (!name) name = "<unknown>";
  //   fprintf(stderr, "[STREAM POOL] WARNING: Failed to synchronize stream %d: %s\n", 
  //           stream_index, name);
  // }
  
  // 更新状态
  pooled_stream.in_use = false;
  g_available_stream_indices.push(stream_index);
  g_active_stream_count--;
  
  // fprintf(stderr, "[STREAM POOL] Released stream %d: %p (Active: %d/%d)\n", 
  //         stream_index, stream, g_active_stream_count.load(), (int)g_stream_pool.size());
}

// 全连接层的实现，使用cuBLAS执行矩阵乘法
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuCulibsFullyConnectedForward(
    int batch_size, int input_features,   // 输入维度
    int output_features,                  // 输出特征数
    void* input_data, void* weight_data,  // 输入和权重指针
    void* bias_data,                      // 偏置指针（可为NULL）
    void* output_data,                    // 输出指针
    CUstream stream                       // CUDA流
) {
  // 确保使用全局上下文
  mgpuEnsureContext();
  
  StreamHandles handles;
  if (!getHandlesForStream(stream, handles)) {
    return; // 错误信息已在getHandlesForStream中打印
  }
  cublasHandle_t handle = handles.cublas_handle;

  // 获取此流的cuBLAS句柄
  // cublasHandle_t handle = mgpuCublasGetHandle(stream);
  
  // 设置矩阵乘法参数
  // C = alpha * op(A) * op(B) + beta * C
  // 对于FC: output = input * weight^T + bias
  const float alpha = 1.0f;
  const float beta = 0.0f;
  
  // 计算矩阵乘法：output = input * weight^T
  // cuBLAS中的矩阵是列主序的，与MLIR中的行主序不同，需要转置操作
  // 所以实际执行的是：(output)^T = (weight)^T * (input)^T
  // 这对应于cuBLAS中 C = B * A 而不是 A * B
  // 即：output(batch_size, output_features) = input(batch_size, input_features) * weight(output_features, input_features)^T
  
  // 使用CUBLAS_OP_T表示权重矩阵需要被转置
  CUBLAS_REPORT_IF_ERROR(cublasSgemm(
      handle,
      CUBLAS_OP_T,                   // op(B)：权重矩阵需要转置
      CUBLAS_OP_N,                   // op(A)：输入矩阵不需要转置
      output_features,               // 输出特征数（m：B的行数）
      batch_size,                    // 批量大小（n：A的列数）
      input_features,                // 输入特征数（k：A的行数，B的列数）
      &alpha,                        // 标量系数alpha
      (const float*)weight_data,     // B矩阵（权重）
      input_features,                // B的主维度是input_features
      (const float*)input_data,      // A矩阵（输入）
      input_features,                // A的主维度是input_features
      &beta,                         // 标量系数beta
      (float*)output_data,           // C矩阵（输出）
      output_features                // C的主维度是output_features
  ));
  
  // 如果提供了偏置，使用cuDNN的AddTensor添加偏置
  if (bias_data != nullptr) {
    // 获取cuDNN句柄
    // cudnnHandle_t cudnnHandle = mgpuCudnnGetHandle(stream);
    cudnnHandle_t cudnnHandle = handles.cudnn_handle;
    
    // 创建张量描述符
    cudnnTensorDescriptor_t outputDesc, biasDesc;
    CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&outputDesc));
    CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&biasDesc));
    
    // 设置输出描述符，将FC输出视为NCHW格式的张量，但C=output_features，H=W=1
    CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
        outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        batch_size, output_features, 1, 1));
    
    // 设置偏置描述符，偏置是1D向量，表示为1xCx1x1
    CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
        biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        1, output_features, 1, 1));
    
    // 添加偏置到输出
    const float alpha_bias = 1.0f;
    const float beta_bias = 1.0f;  // 使用1.0f以便累加到现有输出
    
    CUDNN_REPORT_IF_ERROR(cudnnAddTensor(
        cudnnHandle, &alpha_bias, biasDesc, bias_data, 
        &beta_bias, outputDesc, output_data));
    
    // 清理描述符
    CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(outputDesc));
    CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(biasDesc));
  }
  
  // 注意：不需要清理cuBLAS和cuDNN句柄，因为它们会被缓存以便重用
}

// Fully connected layer implementation with flattening, using cuBLAS for matrix multiplication
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuCulibsFlattenFullyConnectedForward(
    int batch_size, int input_channels, int input_height, int input_width,  // Original NCHW dimensions
    int output_features,                                                    // Output features
    void* input_data, void* weight_data,                                   // Input and weight pointers
    void* bias_data,                                                       // Bias pointer (can be NULL)
    void* output_data,                                                     // Output pointer
    CUstream stream                                                        // CUDA stream
) {
  // Ensure we're using the global context
  mgpuEnsureContext();
  
  // Calculate the flattened features dimension
  int flattened_features = input_channels * input_height * input_width;
  
  StreamHandles handles;
  if (!getHandlesForStream(stream, handles)) {
    return; // 错误信息已在getHandlesForStream中打印
  }
  cublasHandle_t handle = handles.cublas_handle;

  // Get cuBLAS handle for this stream
  // cublasHandle_t handle = mgpuCublasGetHandle(stream);
  
  // Set matrix multiplication parameters
  // C = alpha * op(A) * op(B) + beta * C
  // For FC with flattening: output = input * weight^T + bias
  const float alpha = 1.0f;
  const float beta = 0.0f;
  
  // Compute matrix multiplication: output = input * weight^T
  // cuBLAS uses column-major order, while our tensors are row-major
  // So we compute (output)^T = (weight)^T * (input)^T
  
  // Use CUBLAS_OP_T to indicate that the weight matrix should be transposed
  CUBLAS_REPORT_IF_ERROR(cublasSgemm(
      handle,
      CUBLAS_OP_T,                    // op(B): transpose weight matrix
      CUBLAS_OP_N,                    // op(A): don't transpose input
      output_features,                // output features (m: B's rows)
      batch_size,                     // batch size (n: A's columns)
      flattened_features,             // flattened input features (k: A's rows, B's columns)
      &alpha,                         // alpha scalar
      (const float*)weight_data,      // B matrix (weights)
      flattened_features,             // B's leading dimension
      (const float*)input_data,       // A matrix (input)
      flattened_features,             // A's leading dimension
      &beta,                          // beta scalar
      (float*)output_data,            // C matrix (output)
      output_features                 // C's leading dimension
  ));
  
  // Add bias if provided
  if (bias_data != nullptr) {
    // Get cuDNN handle
    // cudnnHandle_t cudnnHandle = mgpuCudnnGetHandle(stream);
    cudnnHandle_t cudnnHandle = handles.cudnn_handle;
    
    // Create tensor descriptors
    cudnnTensorDescriptor_t outputDesc, biasDesc;
    CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&outputDesc));
    CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&biasDesc));
    
    // Set output descriptor as a 4D tensor with H=W=1
    CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
        outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        batch_size, output_features, 1, 1));
    
    // Set bias descriptor as 1D vector (1xCx1x1)
    CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
        biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        1, output_features, 1, 1));
    
    // Add bias to output
    const float alpha_bias = 1.0f;
    const float beta_bias = 1.0f;  // Use 1.0f to add to existing output
    
    CUDNN_REPORT_IF_ERROR(cudnnAddTensor(
        cudnnHandle, &alpha_bias, biasDesc, bias_data, 
        &beta_bias, outputDesc, output_data));
    
    // Clean up descriptors
    CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(outputDesc));
    CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(biasDesc));
  }
}

// 修改后的卷积函数，每个流使用独立的句柄
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuCudnnConv2dForward(
    int n, int c, int h, int w_in,              // 输入尺寸
    int k, int r, int s,                         // 卷积核尺寸
    int pad_h, int pad_w,                        // 填充
    int stride_h, int stride_w,                  // 步长
    int dilation_h, int dilation_w,              // 膨胀
    void* x_data, void* w_data, void* bias_data, // 输入、权重和偏置指针
    void* y_data,                               // 输出指针
    CUstream stream                             // CUDA流
    //bool createContext = true
) {

  // fprintf(stderr, "Input data: %p, weight data: %p, bias data: %p, output data: %p\n", 
  //   x_data, w_data, bias_data, y_data);
  // // 验证每个指针是否有效
  // if (x_data == nullptr || w_data == nullptr || y_data == nullptr) {
  // fprintf(stderr, "Error: Invalid pointer detected in mgpuCudnnConv2dForward\n");
  // return;
  // }

  // fprintf(stderr, "[START] mgpuCudnnConv2dForward\n");
  
  // ScopedContext scopedContext;

  // 确保使用全局上下文
  mgpuEnsureContext();

  // 获取此流的cuDNN句柄
  // fprintf(stderr, "[HANDLE] Before getting handle\n");
  // 获取预创建的handle组
  StreamHandles handles;
  if (!getHandlesForStream(stream, handles)) {
    return; // 错误信息已在getHandlesForStream中打印
  }
  cudnnHandle_t handle = handles.cudnn_handle;

  // cudnnHandle_t handle = mgpuCudnnGetHandle(stream);

  // fprintf(stderr, "[HANDLE] Got handle: %p\n", handle);
  
  // 创建描述符
  cudnnTensorDescriptor_t xDesc, yDesc, biasDesc;
  cudnnFilterDescriptor_t wDesc;
  cudnnConvolutionDescriptor_t convDesc;
  
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&xDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateFilterDescriptor(&wDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&yDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&biasDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateConvolutionDescriptor(&convDesc));
  
  // 设置输入描述符
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w_in));
  
  // 设置权重描述符
  CUDNN_REPORT_IF_ERROR(cudnnSetFilter4dDescriptor(
      wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, r, s));
  
  // 设置卷积描述符
  CUDNN_REPORT_IF_ERROR(cudnnSetConvolution2dDescriptor(
      convDesc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  
  // 添加这一行以启用Tensor Cores
  // CUDNN_REPORT_IF_ERROR(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

  // 对于Amsere及更高架构，可以考虑使用以下替代方式获得更好性能
  CUDNN_REPORT_IF_ERROR(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));


  // 获取输出尺寸
  int out_n, out_c, out_h, out_w;
  CUDNN_REPORT_IF_ERROR(cudnnGetConvolution2dForwardOutputDim(
      convDesc, xDesc, wDesc, &out_n, &out_c, &out_h, &out_w));

  // fprintf(stderr, "Output dimensions: n=%d, c=%d, h=%d, w=%d\n", 
  //       out_n, out_c, out_h, out_w);
  
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
  
  // 设置偏置描述符(1xCx1x1)
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, k, 1, 1));
  
  // // 自动选择最佳算法
  // int requestedAlgoCount = 10;
  // int returnedAlgoCount;
  // cudnnConvolutionFwdAlgoPerf_t perfResults[10];
  // CUDNN_REPORT_IF_ERROR(cudnnGetConvolutionForwardAlgorithm_v7(
  //     handle, xDesc, wDesc, convDesc, yDesc,
  //     requestedAlgoCount, &returnedAlgoCount, perfResults));
  
  // // 选择最快的且可用的算法
  // cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;
  
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM; // 或其他适合你计算的预定义算法
  // cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
  // cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;


  // 获取工作空间大小
  size_t workspaceSize = 0;
  CUDNN_REPORT_IF_ERROR(cudnnGetConvolutionForwardWorkspaceSize(
      handle, xDesc, wDesc, convDesc, yDesc, algo, &workspaceSize));
  
  // 分配工作空间
  void* workspace = nullptr;
  if (workspaceSize > 0) {
    CUdeviceptr wsPtr = 0;
    CUDA_REPORT_IF_ERROR(cuMemAlloc(&wsPtr, workspaceSize));
    workspace = reinterpret_cast<void*>(wsPtr);
  }

  // 执行卷积
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // CUDNN_REPORT_IF_ERROR(cudnnConvolutionForward(
  //     handle, &alpha, xDesc, x_data, wDesc, w_data, convDesc, algo,
  //     workspace, workspaceSize, &beta, yDesc, y_data));
  

  // // 在cudnnConvolutionForward之前打印关键参数的值
  // fprintf(stderr, "[PRE-CONV] Input ptr: %p, Filter ptr: %p, Output ptr: %p\n", 
  //   x_data, w_data, y_data);

  // // 记录使用的CUDA流和cuDNN句柄
  // fprintf(stderr, "[PRE-CONV] Using stream: %p, handle: %p\n", stream, handle);

  // // 执行卷积操作
  // fprintf(stderr, "[EXECUTING] cudnnConvolutionForward...\n");
  cudnnStatus_t status = cudnnConvolutionForward(
  handle, &alpha, xDesc, x_data, wDesc, w_data, convDesc, algo,
  workspace, workspaceSize, &beta, yDesc, y_data);

  // // 打印卷积执行的结果状态
  // fprintf(stderr, "[POST-CONV] Status: %d (%s)\n", status, 
  //   cudnnGetErrorString(status));

  // // 在cudnnConvolutionForward之后再次打印参数，检查是否有变化
  // fprintf(stderr, "[POST-CONV] Input ptr: %p, Filter ptr: %p, Output ptr: %p\n", 
  //   x_data, w_data, y_data);

  // // 检查工作空间的状态
  // fprintf(stderr, "[POST-CONV] Workspace ptr: %p, size: %zu\n", workspace, workspaceSize);

  // 报告错误（如果有）
  CUDNN_REPORT_IF_ERROR(status);


  // 添加偏置(如果提供)
  if (bias_data != nullptr) {
    const float alpha_bias = 1.0f;
    const float beta_bias = 1.0f;
    CUDNN_REPORT_IF_ERROR(cudnnAddTensor(
        handle, &alpha_bias, biasDesc, bias_data, &beta_bias, yDesc, y_data));
  }
  
  // 释放工作空间
  if (workspace != nullptr) {
    CUDA_REPORT_IF_ERROR(cuMemFree(reinterpret_cast<CUdeviceptr>(workspace)));
  }
  
  // 清理描述符
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(xDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyFilterDescriptor(wDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(yDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(biasDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyConvolutionDescriptor(convDesc));

  // fprintf(stderr, "[END] mgpuCudnnConv2dForward\n");
}

// MaxPool implementation using cuDNN
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuCudnnMaxPoolForward(
    int n, int c, int h, int w,           // 输入维度 (NCHW)
    int kernel_h, int kernel_w,           // 核维度
    int pad_h_begin, int pad_w_begin,     // 填充 (开始)
    int pad_h_end, int pad_w_end,         // 填充 (结束)
    int stride_h, int stride_w,           // 步长
    int dilation_h, int dilation_w,       // 膨胀
    void* input_data,                     // 输入张量
    void* output_data,                    // 输出张量
    CUstream stream                       // CUDA流
) {
  // 确保使用全局上下文
  mgpuEnsureContext();
  
  StreamHandles handles;
  if (!getHandlesForStream(stream, handles)) {
    return; // 错误信息已在getHandlesForStream中打印
  }
  cudnnHandle_t handle = handles.cudnn_handle;

  // 获取此流的cuDNN句柄
  // cudnnHandle_t handle = mgpuCudnnGetHandle(stream);
  
  // 创建描述符
  cudnnTensorDescriptor_t inputDesc, outputDesc;
  cudnnPoolingDescriptor_t poolDesc;
  
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&inputDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&outputDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreatePoolingDescriptor(&poolDesc));
  
  // 设置输入描述符
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  
  // 检查是否为非对称填充
  bool asymmetricPadding = (pad_h_begin != pad_h_end) || (pad_w_begin != pad_w_end);
  
  if (asymmetricPadding) {
    // 对于非对称填充，使用最大填充值
    fprintf(stderr, "Warning: Asymmetric padding in MaxPool (%d,%d,%d,%d) may not produce exact results\n",
            pad_h_begin, pad_w_begin, pad_h_end, pad_w_end);
  }
  
  // cuDNN的pooling API要求对称填充，因此我们使用最大值
  int pad_h = std::max(pad_h_begin, pad_h_end);
  int pad_w = std::max(pad_w_begin, pad_w_end);
  
  // 设置池化描述符
  CUDNN_REPORT_IF_ERROR(cudnnSetPooling2dDescriptor(
      poolDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
      kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
  
  // 计算输出维度
  int out_n, out_c, out_h, out_w;
  CUDNN_REPORT_IF_ERROR(cudnnGetPooling2dForwardOutputDim(
      poolDesc, inputDesc, &out_n, &out_c, &out_h, &out_w));
  
  // 设置输出描述符
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
  
  // 执行最大池化
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUDNN_REPORT_IF_ERROR(cudnnPoolingForward(
      handle, poolDesc, &alpha, inputDesc, input_data, &beta, outputDesc, output_data));
  
  // 清理描述符
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(inputDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(outputDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyPoolingDescriptor(poolDesc));
}

// 下面是兼容旧API的函数，使用新实现
extern "C" MLIR_CUDA_WRAPPERS_EXPORT cudnnHandle_t mgpuCudnnCreate() {
  return mgpuCudnnGetHandle(nullptr);  // 使用默认流
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuCudnnDestroy() {
  mgpuCudnnDestroyHandle(nullptr);  // 销毁默认流的句柄
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuCudnnSetStream(CUstream stream) {
  // 此操作不再必要，因为每个流都有自己的句柄
  // 为了兼容旧代码，我们确保存在一个与该流关联的句柄
  mgpuCudnnGetHandle(stream);
}


// 张量乘法操作: C = A * B
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void 
mgpuCudnnMul(void* inputA, void* inputB, void* output,
             int n, int c, int h, int w,
             CUstream stream) {
  mgpuEnsureContext();
  // ScopedContext scopedContext;
  
  StreamHandles handles;
  if (!getHandlesForStream(stream, handles)) {
    return; // 错误信息已在getHandlesForStream中打印
  }
  cudnnHandle_t handle = handles.cudnn_handle;

  // 获取此流的cuDNN句柄
  // cudnnHandle_t handle = mgpuCudnnGetHandle(stream);
  
  // 创建张量描述符
  cudnnTensorDescriptor_t aDesc, bDesc, cDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&aDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&bDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&cDesc));
  
  // 设置张量描述符
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      aDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      cDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  
  // 创建操作描述符
  cudnnOpTensorDescriptor_t opDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateOpTensorDescriptor(&opDesc));
  
  // 设置为乘法操作
  CUDNN_REPORT_IF_ERROR(cudnnSetOpTensorDescriptor(
      opDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
  
  // 设置缩放因子
  float alpha1 = 1.0f;
  float alpha2 = 1.0f;
  float beta = 0.0f;
  
  // 执行操作: C = alpha1 * A * alpha2 * B + beta * C
  CUDNN_REPORT_IF_ERROR(cudnnOpTensor(
      handle, opDesc,
      &alpha1, aDesc, inputA,
      &alpha2, bDesc, inputB,
      &beta, cDesc, output));
  
  // 清理描述符
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(aDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(bDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(cDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyOpTensorDescriptor(opDesc));
}

// 张量加法操作: C = A + B
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void 
mgpuCudnnAdd(void* inputA, void* inputB, void* output,
             int n, int c, int h, int w,
             CUstream stream) {
  mgpuEnsureContext();
  // ScopedContext scopedContext;
  

  // 获取预创建的handle组
  StreamHandles handles;
  if (!getHandlesForStream(stream, handles)) {
    return; // 错误信息已在getHandlesForStream中打印
  }
  
  cudnnHandle_t handle = handles.cudnn_handle;
  // 获取此流的cuDNN句柄
  // cudnnHandle_t handle = mgpuCudnnGetHandle(stream);
  
  // 创建张量描述符
  cudnnTensorDescriptor_t aDesc, bDesc, cDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&aDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&bDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&cDesc));
  
  // 设置张量描述符
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      aDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      cDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  
  // 创建操作描述符
  cudnnOpTensorDescriptor_t opDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateOpTensorDescriptor(&opDesc));
  
  // 设置为加法操作
  CUDNN_REPORT_IF_ERROR(cudnnSetOpTensorDescriptor(
      opDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
  
  // 设置缩放因子
  float alpha1 = 1.0f;
  float alpha2 = 1.0f;
  float beta = 0.0f;
  
  // 执行操作: C = alpha1 * A + alpha2 * B + beta * C
  CUDNN_REPORT_IF_ERROR(cudnnOpTensor(
      handle, opDesc,
      &alpha1, aDesc, inputA,
      &alpha2, bDesc, inputB,
      &beta, cDesc, output));
  
  // 清理描述符
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(aDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(bDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(cDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyOpTensorDescriptor(opDesc));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void 
mgpuCudnnSub(void* inputA, void* inputB, void* output,
             int n, int c, int h, int w,
             CUstream stream) {
  mgpuEnsureContext();
  // ScopedContext scopedContext;
  
  StreamHandles handles;
  if (!getHandlesForStream(stream, handles)) {
    return; // 错误信息已在getHandlesForStream中打印
  }
  cudnnHandle_t handle = handles.cudnn_handle;

  // 获取此流的cuDNN句柄
  // cudnnHandle_t handle = mgpuCudnnGetHandle(stream);
  
  // 创建张量描述符
  cudnnTensorDescriptor_t aDesc, bDesc, cDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&aDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&bDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&cDesc));
  
  // 设置张量描述符
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      aDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      cDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  
  // 创建操作描述符
  cudnnOpTensorDescriptor_t opDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateOpTensorDescriptor(&opDesc));
  
  // 设置为加法操作
  CUDNN_REPORT_IF_ERROR(cudnnSetOpTensorDescriptor(
      opDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
  
  // 设置缩放因子：alpha1 = 1.0 (A), alpha2 = -1.0 (-B)
  float alpha1 = 1.0f;
  float alpha2 = -1.0f;  // 关键：使用负系数来实现减法
  float beta = 0.0f;
  
  // 执行操作: C = alpha1 * A + alpha2 * B + beta * C
  CUDNN_REPORT_IF_ERROR(cudnnOpTensor(
      handle, opDesc,
      &alpha1, aDesc, inputA,
      &alpha2, bDesc, inputB,
      &beta, cDesc, output));
  
  // 清理描述符
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(aDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(bDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(cDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyOpTensorDescriptor(opDesc));
}

// 张量除法操作: C = A / B
// 通过乘法实现: C = A * (1/B)
// extern "C" MLIR_CUDA_WRAPPERS_EXPORT void 
// mgpuCudnnDiv(void* inputA, void* inputB, void* output,
//              int n, int c, int h, int w,
//              CUstream stream) {
//   ScopedContext scopedContext;
  
//   // 获取此流的cuDNN句柄
//   cudnnHandle_t handle = mgpuCudnnGetHandle(stream);
  
//   // 创建张量描述符
//   cudnnTensorDescriptor_t aDesc, bDesc, cDesc;
//   CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&aDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&bDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&cDesc));
  
//   // 设置张量描述符
//   CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
//       aDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
//   CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
//       bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
//   CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
//       cDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  
//   // 注意：cuDNN没有直接的除法操作，需要创建一个临时缓冲区来存储1/B
//   // 分配临时缓冲区
//   CUdeviceptr dTemp = 0;
//   size_t size = n * c * h * w * sizeof(float);
//   CUDA_REPORT_IF_ERROR(cuMemAlloc(&dTemp, size));
//   void* temp = reinterpret_cast<void*>(dTemp);
  
//   // 设置临时描述符
//   cudnnTensorDescriptor_t tempDesc;
//   CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&tempDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
//       tempDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  
//   // 使用自定义kernel计算1/B
//   // 注意：cuDNN没有直接的倒数操作，所以需要使用一个自定义kernel
//   // 这里假设我们有一个名为inverseTensorKernel的kernel函数
//   // 实际应用中，您需要根据您的环境实现该函数
  
//   // 临时解决方案：使用标量除法
//   // 创建一个全1张量，然后用常量除法
//   float one = 1.0f;
//   CUDA_REPORT_IF_ERROR(cuMemsetD32Async(dTemp, *(unsigned int*)&one, size/sizeof(float), stream));
  
//   // 创建倒数操作描述符
//   cudnnOpTensorDescriptor_t divDesc;
//   CUDNN_REPORT_IF_ERROR(cudnnCreateOpTensorDescriptor(&divDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnSetOpTensorDescriptor(
//       divDesc, CUDNN_OP_TENSOR_DIV, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
  
//   // 计算1/B
//   float alpha1 = 1.0f;
//   float alpha2 = 1.0f;
//   float beta = 0.0f;
//   CUDNN_REPORT_IF_ERROR(cudnnOpTensor(
//       handle, divDesc,
//       &alpha1, tempDesc, temp,  // 1.0
//       &alpha2, bDesc, inputB,   // B
//       &beta, tempDesc, temp));  // 结果存入temp (1.0/B)
  
//   // 创建乘法操作描述符
//   cudnnOpTensorDescriptor_t mulDesc;
//   CUDNN_REPORT_IF_ERROR(cudnnCreateOpTensorDescriptor(&mulDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnSetOpTensorDescriptor(
//       mulDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
  
//   // 执行操作: C = A * (1/B)
//   CUDNN_REPORT_IF_ERROR(cudnnOpTensor(
//       handle, mulDesc,
//       &alpha1, aDesc, inputA,
//       &alpha2, tempDesc, temp,
//       &beta, cDesc, output));
  
//   // 清理描述符和临时内存
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(aDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(bDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(cDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(tempDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyOpTensorDescriptor(divDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyOpTensorDescriptor(mulDesc));
//   CUDA_REPORT_IF_ERROR(cuMemFree(dTemp));
// }

// 张量取反操作: B = -A
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void 
mgpuCudnnNeg(void* input, void* output,
             int n, int c, int h, int w,
             CUstream stream) {
  mgpuEnsureContext();
  // ScopedContext scopedContext;
  
  StreamHandles handles;
  if (!getHandlesForStream(stream, handles)) {
    return; // 错误信息已在getHandlesForStream中打印
  }
  cudnnHandle_t handle = handles.cudnn_handle;

  // 获取此流的cuDNN句柄
  // cudnnHandle_t handle = mgpuCudnnGetHandle(stream);
  
  // 创建张量描述符
  cudnnTensorDescriptor_t aDesc, cDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&aDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&cDesc));
  
  // 设置张量描述符
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      aDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      cDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  
  // 为第二个操作数创建一个虚拟张量描述符（实际不会使用）
  cudnnTensorDescriptor_t dummyDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&dummyDesc));
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      dummyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1));
  
  // 创建操作描述符
  cudnnOpTensorDescriptor_t opDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateOpTensorDescriptor(&opDesc));
  
  // 使用加法操作实现取反: -A = -1.0 * A + 0 * dummy
  CUDNN_REPORT_IF_ERROR(cudnnSetOpTensorDescriptor(
      opDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
  
  // 设置缩放因子，alpha1 = -1.0表示将输入值取反
  float alpha1 = -1.0f;
  float alpha2 = 0.0f;
  float beta = 0.0f;
  
  // 创建一个虚拟输入
  float dummyValue = 0.0f;
  
  // 执行操作: B = -1.0 * A + 0.0 * dummy + 0.0 * B
  CUDNN_REPORT_IF_ERROR(cudnnOpTensor(
      handle, opDesc,
      &alpha1, aDesc, input,
      &alpha2, dummyDesc, &dummyValue,
      &beta, cDesc, output));
  
  // 清理描述符
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(aDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(cDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(dummyDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyOpTensorDescriptor(opDesc));
}

// 标量乘法操作: C = A * scalar
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void 
mgpuCudnnMulScalar(void* input, void* scalar, void* output,
                  int n, int c, int h, int w,
                  CUstream stream) {
  mgpuEnsureContext();
  
  StreamHandles handles;
  if (!getHandlesForStream(stream, handles)) {
    return; // 错误信息已在getHandlesForStream中打印
  }
  cudnnHandle_t handle = handles.cudnn_handle;
  
  // 创建张量描述符
  cudnnTensorDescriptor_t inputDesc, scalarDesc, outputDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&inputDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&scalarDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&outputDesc));
  
  // 设置张量描述符
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      scalarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1));
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  
  // 创建操作描述符
  cudnnOpTensorDescriptor_t opDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateOpTensorDescriptor(&opDesc));
  
  // 设置为乘法操作
  CUDNN_REPORT_IF_ERROR(cudnnSetOpTensorDescriptor(
      opDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
  
  // 设置缩放因子
  float alpha1 = 1.0f;  // input的系数
  float alpha2 = 1.0f;  // scalar的系数
  float beta = 0.0f;    // output的系数
  
  // 执行操作: output = alpha1 * input * alpha2 * scalar + beta * output
  CUDNN_REPORT_IF_ERROR(cudnnOpTensor(
      handle, opDesc,
      &alpha1, inputDesc, input,
      &alpha2, scalarDesc, scalar,
      &beta, outputDesc, output));
  
  // 清理描述符
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(inputDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(scalarDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(outputDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyOpTensorDescriptor(opDesc));
}

// 标量加法操作: C = A + scalar
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void 
mgpuCudnnAddScalar(void* input, void* scalar, void* output,
                  int n, int c, int h, int w,
                  CUstream stream) {
  mgpuEnsureContext();
  
  StreamHandles handles;
  if (!getHandlesForStream(stream, handles)) {
    return; // 错误信息已在getHandlesForStream中打印
  }
  cudnnHandle_t handle = handles.cudnn_handle;
  
  // 创建张量描述符
  cudnnTensorDescriptor_t inputDesc, scalarDesc, outputDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&inputDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&scalarDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&outputDesc));
  
  // 设置张量描述符
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      scalarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1));
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  
  // 创建操作描述符
  cudnnOpTensorDescriptor_t opDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateOpTensorDescriptor(&opDesc));
  
  // 设置为加法操作
  CUDNN_REPORT_IF_ERROR(cudnnSetOpTensorDescriptor(
      opDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
  
  // 设置缩放因子
  float alpha1 = 1.0f;  // input的系数
  float alpha2 = 1.0f;  // scalar的系数  
  float beta = 0.0f;    // output的系数
  
  // 执行操作: output = alpha1 * input + alpha2 * scalar + beta * output
  CUDNN_REPORT_IF_ERROR(cudnnOpTensor(
      handle, opDesc,
      &alpha1, inputDesc, input,
      &alpha2, scalarDesc, scalar,
      &beta, outputDesc, output));
  
  // 清理描述符
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(inputDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(scalarDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(outputDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyOpTensorDescriptor(opDesc));
}

// 标量减法操作: C = A - scalar
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void 
mgpuCudnnSubScalar(void* input, void* scalar, void* output,
                  int n, int c, int h, int w,
                  CUstream stream) {
  mgpuEnsureContext();
  
  StreamHandles handles;
  if (!getHandlesForStream(stream, handles)) {
    return; // 错误信息已在getHandlesForStream中打印
  }
  cudnnHandle_t handle = handles.cudnn_handle;
  
  // 创建张量描述符
  cudnnTensorDescriptor_t inputDesc, scalarDesc, outputDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&inputDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&scalarDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&outputDesc));
  
  // 设置张量描述符
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      scalarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1));
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  
  // 创建操作描述符
  cudnnOpTensorDescriptor_t opDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateOpTensorDescriptor(&opDesc));
  
  // 设置为加法操作 (通过负系数实现减法)
  CUDNN_REPORT_IF_ERROR(cudnnSetOpTensorDescriptor(
      opDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
  
  // 设置缩放因子
  float alpha1 = 1.0f;   // input的系数
  float alpha2 = -1.0f;  // scalar的系数 (负数实现减法)
  float beta = 0.0f;     // output的系数
  
  // 执行操作: output = alpha1 * input + alpha2 * scalar + beta * output
  // 即: output = input + (-1) * scalar = input - scalar
  CUDNN_REPORT_IF_ERROR(cudnnOpTensor(
      handle, opDesc,
      &alpha1, inputDesc, input,
      &alpha2, scalarDesc, scalar,
      &beta, outputDesc, output));
  
  // 清理描述符
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(inputDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(scalarDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(outputDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyOpTensorDescriptor(opDesc));
}

// 反向标量减法操作: C = scalar - A
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void 
mgpuCudnnRSubScalar(void* input, void* scalar, void* output,
                   int n, int c, int h, int w,
                   CUstream stream) {
  mgpuEnsureContext();
  
  StreamHandles handles;
  if (!getHandlesForStream(stream, handles)) {
    return; // 错误信息已在getHandlesForStream中打印
  }
  cudnnHandle_t handle = handles.cudnn_handle;
  
  // 创建张量描述符
  cudnnTensorDescriptor_t inputDesc, scalarDesc, outputDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&inputDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&scalarDesc));
  CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&outputDesc));
  
  // 设置张量描述符
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      scalarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1));
  CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
      outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  
  // 创建操作描述符
  cudnnOpTensorDescriptor_t opDesc;
  CUDNN_REPORT_IF_ERROR(cudnnCreateOpTensorDescriptor(&opDesc));
  
  // 设置为加法操作 (通过负系数实现反向减法)
  CUDNN_REPORT_IF_ERROR(cudnnSetOpTensorDescriptor(
      opDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
  
  // 设置缩放因子
  float alpha1 = -1.0f;  // input的系数 (负数)
  float alpha2 = 1.0f;   // scalar的系数
  float beta = 0.0f;     // output的系数
  
  // 执行操作: output = alpha1 * input + alpha2 * scalar + beta * output
  // 即: output = (-1) * input + scalar = scalar - input
  CUDNN_REPORT_IF_ERROR(cudnnOpTensor(
      handle, opDesc,
      &alpha1, inputDesc, input,
      &alpha2, scalarDesc, scalar,
      &beta, outputDesc, output));
  
  // 清理描述符
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(inputDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(scalarDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(outputDesc));
  CUDNN_REPORT_IF_ERROR(cudnnDestroyOpTensorDescriptor(opDesc));
}

// // 标量乘法操作: B = A * scalar
// extern "C" MLIR_CUDA_WRAPPERS_EXPORT void 
// mgpuCudnnMulScalar(void* input, void* scalar, void* output,
//                   int n, int c, int h, int w,
//                   CUstream stream) {
//   mgpuEnsureContext();
  
//   StreamHandles handles;
//   if (!getHandlesForStream(stream, handles)) {
//     return; // 错误信息已在getHandlesForStream中打印
//   }
//   cudnnHandle_t handle = handles.cudnn_handle;

//   // 获取此流的cuDNN句柄
//   // cudnnHandle_t handle = mgpuCudnnGetHandle(stream);
  
//   // 创建张量描述符
//   cudnnTensorDescriptor_t xDesc, yDesc;
//   CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&xDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&yDesc));
  
//   // 设置张量描述符
//   CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
//       xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
//   CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
//       yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  
//   // 获取标量值
//   float scalarValue = *(float*)scalar;
//   float beta = 0.0f;
  
//   // 使用cudnnTransformTensor实现标量乘法: output = scalar * input + 0 * output
//   CUDNN_REPORT_IF_ERROR(cudnnTransformTensor(
//       handle,
//       &scalarValue, xDesc, input,
//       &beta, yDesc, output));
  
//   // 清理描述符
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(xDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(yDesc));
// }

// // 标量减法操作: C = A - scalar
// extern "C" MLIR_CUDA_WRAPPERS_EXPORT void 
// mgpuCudnnSubScalar(void* input, void* scalar, void* output,
//                   int n, int c, int h, int w,
//                   CUstream stream) {
//   mgpuEnsureContext();
  
//   StreamHandles handles;
//   if (!getHandlesForStream(stream, handles)) {
//     return; // 错误信息已在getHandlesForStream中打印
//   }
//   cudnnHandle_t handle = handles.cudnn_handle;

//   // 获取此流的cuDNN句柄
//   // cudnnHandle_t handle = mgpuCudnnGetHandle(stream);
  
//   // 创建张量描述符
//   cudnnTensorDescriptor_t xDesc, yDesc;
//   CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&xDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&yDesc));
  
//   // 设置张量描述符
//   CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
//       xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
//   CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
//       yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  
//   // 获取标量值，并将其变为负值以实现减法
//   float scalarValue = *(float*)scalar;
//   float negScalarValue = -scalarValue;  // 输入 - 标量 相当于 1.0 * 输入 + (-标量)
//   float one = 1.0f;
//   float zero = 0.0f;
  
//   // 首先将输入复制到输出
//   CUDNN_REPORT_IF_ERROR(cudnnTransformTensor(
//       handle,
//       &one, xDesc, input,
//       &zero, yDesc, output));
  
//   // 然后添加负的标量值 (使用cudnnAddTensor实现)
//   // 创建一个1x1x1x1的描述符用于标量
//   cudnnTensorDescriptor_t scalarDesc;
//   CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&scalarDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
//       scalarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1));
  
//   // 在GPU上创建一个只有一个元素的缓冲区存储标量
//   float h_scalar = negScalarValue;
//   float* d_scalar;
//   cudaMalloc(&d_scalar, sizeof(float));
//   cudaMemcpy(d_scalar, &h_scalar, sizeof(float), cudaMemcpyHostToDevice);
  
//   // 添加负的标量值到所有元素: output = output + negScalarValue
//   CUDNN_REPORT_IF_ERROR(cudnnAddTensor(
//       handle,
//       &one, scalarDesc, d_scalar,
//       &one, yDesc, output));
  
//   // 清理
//   cudaFree(d_scalar);
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(xDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(yDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(scalarDesc));
// }

// // 反向标量减法操作: C = scalar - A
// extern "C" MLIR_CUDA_WRAPPERS_EXPORT void 
// mgpuCudnnRSubScalar(void* input, void* scalar, void* output,
//                    int n, int c, int h, int w,
//                    CUstream stream) {
//   mgpuEnsureContext();
  
//   StreamHandles handles;
//   if (!getHandlesForStream(stream, handles)) {
//     return; // 错误信息已在getHandlesForStream中打印
//   }
//   cudnnHandle_t handle = handles.cudnn_handle;

//   // 获取此流的cuDNN句柄
//   // cudnnHandle_t handle = mgpuCudnnGetHandle(stream);
  
//   // 创建张量描述符
//   cudnnTensorDescriptor_t xDesc, yDesc;
//   CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&xDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&yDesc));
  
//   // 设置张量描述符
//   CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
//       xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
//   CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
//       yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  
//   // 取负的输入，然后加上标量值
//   float negOne = -1.0f;
//   float zero = 0.0f;
  
//   // 首先计算 output = -input
//   CUDNN_REPORT_IF_ERROR(cudnnTransformTensor(
//       handle,
//       &negOne, xDesc, input,
//       &zero, yDesc, output));
  
//   // 然后添加标量值 output = -input + scalar
//   float scalarValue = *(float*)scalar;
  
//   // 创建一个1x1x1x1的描述符用于标量
//   cudnnTensorDescriptor_t scalarDesc;
//   CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&scalarDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
//       scalarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1));
  
//   // 在GPU上创建一个只有一个元素的缓冲区存储标量
//   float h_scalar = scalarValue;
//   float* d_scalar;
//   cudaMalloc(&d_scalar, sizeof(float));
//   cudaMemcpy(d_scalar, &h_scalar, sizeof(float), cudaMemcpyHostToDevice);
  
//   // 添加标量值到所有元素: output = output + scalarValue
//   float one = 1.0f;
//   CUDNN_REPORT_IF_ERROR(cudnnAddTensor(
//       handle,
//       &one, scalarDesc, d_scalar,
//       &one, yDesc, output));
  
//   // 清理
//   cudaFree(d_scalar);
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(xDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(yDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(scalarDesc));
// }

// // 标量加法操作: C = A + scalar
// extern "C" MLIR_CUDA_WRAPPERS_EXPORT void 
// mgpuCudnnAddScalar(void* input, void* scalar, void* output,
//                   int n, int c, int h, int w,
//                   CUstream stream) {
//   mgpuEnsureContext();
  
//   StreamHandles handles;
//   if (!getHandlesForStream(stream, handles)) {
//     return; // 错误信息已在getHandlesForStream中打印
//   }
//   cudnnHandle_t handle = handles.cudnn_handle;

//   // 获取此流的cuDNN句柄
//   // cudnnHandle_t handle = mgpuCudnnGetHandle(stream);
  
//   // 创建张量描述符
//   cudnnTensorDescriptor_t xDesc, yDesc;
//   CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&xDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&yDesc));
  
//   // 设置张量描述符
//   CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
//       xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
//   CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
//       yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  
//   // 获取标量值
//   float scalarValue = *(float*)scalar;
//   float one = 1.0f;
//   float zero = 0.0f;
  
//   // 首先复制输入到输出
//   CUDNN_REPORT_IF_ERROR(cudnnTransformTensor(
//       handle,
//       &one, xDesc, input,
//       &zero, yDesc, output));
  
//   // 创建一个1x1x1x1的描述符用于标量
//   cudnnTensorDescriptor_t scalarDesc;
//   CUDNN_REPORT_IF_ERROR(cudnnCreateTensorDescriptor(&scalarDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnSetTensor4dDescriptor(
//       scalarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1));
  
//   // 在GPU上创建一个只有一个元素的缓冲区存储标量
//   float h_scalar = scalarValue;
//   float* d_scalar;
//   cudaMalloc(&d_scalar, sizeof(float));
//   cudaMemcpy(d_scalar, &h_scalar, sizeof(float), cudaMemcpyHostToDevice);
  
//   // 添加标量值到所有元素: output = output + scalarValue
//   CUDNN_REPORT_IF_ERROR(cudnnAddTensor(
//       handle,
//       &one, scalarDesc, d_scalar,
//       &one, yDesc, output));
  
//   // 清理
//   cudaFree(d_scalar);
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(xDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(yDesc));
//   CUDNN_REPORT_IF_ERROR(cudnnDestroyTensorDescriptor(scalarDesc));
// }

// 全局缓存变量
static CUmodule cachedModule = nullptr;
static void* cachedModuleData = nullptr;
static bool moduleMarkedForUnload = false;

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUmodule
mgpuModuleLoad(void *data, size_t /*gpuBlobSize*/) {
  mgpuEnsureContext();
  // ScopedContext scopedContext;
  
  // 如果缓存有效且数据匹配，直接返回缓存的module
  if (cachedModule != nullptr && cachedModuleData == data) {
    moduleMarkedForUnload = false;  // 重置卸载标记
    return cachedModule;
  }
  
  // 如果有之前标记为卸载的module，现在真正卸载它
  if (cachedModule != nullptr && moduleMarkedForUnload) {
    CUDA_REPORT_IF_ERROR(cuModuleUnload(cachedModule));
    cachedModule = nullptr;
  }
  
  // 加载新module
  CUmodule module = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleLoadData(&module, data));
  
  // 更新缓存
  cachedModule = module;
  cachedModuleData = data;
  moduleMarkedForUnload = false;
  
  return module;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUmodule mgpuModuleLoadJIT(void *data,
                                                                int optLevel) {
  // ScopedContext scopedContext;
  mgpuEnsureContext();
  CUmodule module = nullptr;
  char jitErrorBuffer[4096] = {0};
  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                               CU_JIT_OPTIMIZATION_LEVEL};
  void *jitOptionsVals[] = {jitErrorBuffer,
                            reinterpret_cast<void *>(sizeof(jitErrorBuffer)),
                            reinterpret_cast<void *>(optLevel)};

  CUresult result =
      cuModuleLoadDataEx(&module, data, 3, jitOptions, jitOptionsVals);
  if (result) {
    fprintf(stderr, "JIT compilation failed with: '%s'\n", jitErrorBuffer);
    CUDA_REPORT_IF_ERROR(result);
  }
  return module;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuModuleUnload(CUmodule module) {
  // 只标记module为待卸载，不立即执行卸载
  if (module == cachedModule) {
    moduleMarkedForUnload = true;
  } else {
    // 如果不是我们缓存的module，则正常卸载
    CUDA_REPORT_IF_ERROR(cuModuleUnload(module));
  }
  // CUDA_REPORT_IF_ERROR(cuModuleUnload(module));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUfunction
mgpuModuleGetFunction(CUmodule module, const char *name) {
  CUfunction function = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleGetFunction(&function, module, name));
  return function;
}

// The wrapper uses intptr_t instead of CUDA's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuLaunchKernel(CUfunction function, intptr_t gridX, intptr_t gridY,
                 intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                 intptr_t blockZ, int32_t smem, CUstream stream, void **params,
                 void **extra, size_t /*paramsCount*/) {
  
  mgpuEnsureContext();
  // ScopedContext scopedContext;
  if (smem > 0) {
    // Avoid checking driver as it's more expensive than if statement
    int32_t maxShmem = 0;
    CUdevice device = getDefaultCuDevice();
    CUDA_REPORT_IF_ERROR(cuDeviceGet(&device, /*ordinal=*/defaultDevice));
    CUDA_REPORT_IF_ERROR(cuDeviceGetAttribute(
        &maxShmem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        device));
    if (maxShmem < smem) {
      fprintf(stderr,
              "Requested shared memory (%dkb) is larger than maximum allowed "
              "shared memory (%dkb) for this device\n",
              smem, maxShmem);
    }
    CUDA_REPORT_IF_ERROR(cuFuncSetAttribute(
        function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
  }
  debug_print("Launching kernel, grid=%ld,%ld,%ld, "
              "threads: %ld, %ld, %ld, "
              "smem: %dkb\n",
              gridX, gridY, gridZ, blockX, blockY, blockZ, smem);
  CUDA_REPORT_IF_ERROR(cuLaunchKernel(function, gridX, gridY, gridZ, blockX,
                                      blockY, blockZ, smem, stream, params,
                                      extra));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUstream mgpuStreamCreate() {

  // 确保我们在全局上下文中
  mgpuEnsureContext();

  // ScopedContext scopedContext;
  CUstream stream = nullptr;
  CUDA_REPORT_IF_ERROR(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  return stream;
}
// extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUstream mgpuStreamCreate() {
//   ScopedContext scopedContext;
//   CUstream stream = nullptr;
  
//   static int streamCounter = 0;
//   int currentStream = streamCounter++;
  
//   // 使用您设备上实际的优先级值
//   const int priorityHigh = -5;  // 最高优先级
//   const int priorityLow = 0;    // 最低优先级
  
//   if (currentStream == 0) {
//     // 第一个流 - 低优先级
//     CUDA_REPORT_IF_ERROR(cuStreamCreateWithPriority(&stream, CU_STREAM_NON_BLOCKING, priorityLow));
//     fprintf(stderr, "Created stream 0 with LOW priority (%d)\n", priorityLow);
//   } else {
//     // 第二个流 - 高优先级
//     CUDA_REPORT_IF_ERROR(cuStreamCreateWithPriority(&stream, CU_STREAM_NON_BLOCKING, priorityHigh));
//     fprintf(stderr, "Created stream 1 with HIGH priority (%d)\n", priorityHigh);
//   }
  
//   return stream;
// }

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuDeviceSynchronize() {
  // 确保我们在全局上下文中
  mgpuEnsureContext();
  
  // 等待所有GPU操作完成
  CUDA_REPORT_IF_ERROR(cuCtxSynchronize());
  fprintf(stderr, "[DEVICE] Device synchronized\n");
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuStreamDestroy(CUstream stream) {
  if (stream == nullptr) {
    fprintf(stderr, "[STREAM] Warning: Attempting to destroy NULL stream\n");
    return;
  }

  // 确保我们在全局上下文中
  mgpuEnsureContext();
  
  if (g_handle_pool_initialized) {
    // 检查是否有handle需要退还
    bool has_handle = false;
    {
      std::lock_guard<std::mutex> lock(g_handle_pool_mutex);
      auto it = g_stream_to_handle_index.find(stream);
      has_handle = (it != g_stream_to_handle_index.end());
    }
    
    if (has_handle) {
      // fprintf(stderr, "[STREAM] WARNING: Stream %p still has handle assigned, auto-releasing...\n", stream);
      mgpuReleasePooledHandles(stream);
    }
  } else {
    // 如果没有使用Handle Pool，使用原有的销毁方式
    mgpuDestroyHandlesForStream(stream);
  }

  // mgpuDestroyHandlesForStream(stream);

  // 先销毁与此流关联的cuDNN句柄
  // mgpuCudnnDestroyHandle(stream);

  // 销毁与此流关联的cuBLAS句柄
  // mgpuCublasDestroyHandle(stream);

  CUDA_REPORT_IF_ERROR(cuStreamDestroy(stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuStreamSynchronize(CUstream stream) {
  if (stream == nullptr) {
    fprintf(stderr, "[STREAM] Warning: Attempting to synchronize NULL stream\n");
    return;
  }

  // 确保我们在全局上下文中
  mgpuEnsureContext();

  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuStreamWaitEvent(CUstream stream,
                                                              CUevent event) {
  CUDA_REPORT_IF_ERROR(cuStreamWaitEvent(stream, event, /*flags=*/0));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUevent mgpuEventCreate() {
  mgpuEnsureContext();
  //ScopedContext scopedContext;
  CUevent event = nullptr;
  // CUDA_REPORT_IF_ERROR(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
  CUDA_REPORT_IF_ERROR(cuEventCreate(&event, CU_EVENT_DEFAULT)); // modify
  return event;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuEventDestroy(CUevent event) {
  CUDA_REPORT_IF_ERROR(cuEventDestroy(event));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuEventSynchronize(CUevent event) {
  CUDA_REPORT_IF_ERROR(cuEventSynchronize(event));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuEventRecord(CUevent event,
                                                          CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuEventRecord(event, stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT float mgpuEventElapsedTime(CUevent start, CUevent end) { // modify
  float milliseconds = 0.0f;
  CUDA_REPORT_IF_ERROR(cuEventElapsedTime(&milliseconds, start, end));
  return milliseconds;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void printFloat(float value) { // modify
  printf("Elapsed time: %.3f ms\n", value);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuMemAlloc(uint64_t sizeBytes, CUstream stream, bool isHostShared) {
  mgpuEnsureContext();
  // ScopedContext scopedContext;
  CUdeviceptr ptr = 0;
  if (sizeBytes == 0)
    return reinterpret_cast<void *>(ptr);

  if (isHostShared) {
    CUDA_REPORT_IF_ERROR(
        cuMemAllocManaged(&ptr, sizeBytes, CU_MEM_ATTACH_GLOBAL));
    return reinterpret_cast<void *>(ptr);
  }
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&ptr, sizeBytes));
  return reinterpret_cast<void *>(ptr);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuMemFree(void *ptr,
                                                      CUstream /*stream*/) {
  CUDA_REPORT_IF_ERROR(cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuMemcpy(void *dst, void *src, size_t sizeBytes, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst),
                                     reinterpret_cast<CUdeviceptr>(src),
                                     sizeBytes, stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuMemset32(void *dst, unsigned int value, size_t count, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(dst),
                                        value, count, stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuMemset16(void *dst, unsigned short value, size_t count, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuMemsetD16Async(reinterpret_cast<CUdeviceptr>(dst),
                                        value, count, stream));
}

///
/// Helper functions for writing mlir example code
///

// Allows to register byte array with the CUDA runtime. Helpful until we have
// transfer functions implemented.
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuMemHostRegister(ptr, sizeBytes, /*flags=*/0));
}

/// Registers a memref with the CUDA runtime. `descriptor` is a pointer to a
/// ranked memref descriptor struct of rank `rank`. Helpful until we have
/// transfer functions implemented.
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuMemHostRegisterMemRef(int64_t rank, StridedMemRefType<char, 1> *descriptor,
                          int64_t elementSizeBytes) {
  // Only densely packed tensors are currently supported.
#ifdef _WIN32
  int64_t *denseStrides = (int64_t *)_alloca(rank * sizeof(int64_t));
#else
  int64_t *denseStrides = (int64_t *)alloca(rank * sizeof(int64_t));
#endif // _WIN32
  int64_t *sizes = descriptor->sizes;
  for (int64_t i = rank - 1, runningStride = 1; i >= 0; i--) {
    denseStrides[i] = runningStride;
    runningStride *= sizes[i];
  }
  uint64_t sizeBytes = sizes[0] * denseStrides[0] * elementSizeBytes;
  int64_t *strides = &sizes[rank];
  (void)strides;
  for (unsigned i = 0; i < rank; ++i)
    assert(strides[i] == denseStrides[i] &&
           "Mismatch in computed dense strides");

  auto *ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostRegister(ptr, sizeBytes);
}

// Allows to unregister byte array with the CUDA runtime.
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuMemHostUnregister(void *ptr) {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuMemHostUnregister(ptr));
}

/// Unregisters a memref with the CUDA runtime. `descriptor` is a pointer to a
/// ranked memref descriptor struct of rank `rank`
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuMemHostUnregisterMemRef(int64_t rank,
                            StridedMemRefType<char, 1> *descriptor,
                            int64_t elementSizeBytes) {
  auto *ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostUnregister(ptr);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuSetDefaultDevice(int32_t device) {
  defaultDevice = device;
}

// Test wrappers
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuCudnnTest() {
  printf("Starting parallel cuDNN test with 4 streams...\n");
  
  // 创建CUDA上下文
  ScopedContext scopedContext;
  
  // 创建四个流用于并行执行
  CUstream stream1 = nullptr, stream2 = nullptr, stream3 = nullptr, stream4 = nullptr;
  CUDA_REPORT_IF_ERROR(cuStreamCreate(&stream1, CU_STREAM_NON_BLOCKING));
  CUDA_REPORT_IF_ERROR(cuStreamCreate(&stream2, CU_STREAM_NON_BLOCKING));
  CUDA_REPORT_IF_ERROR(cuStreamCreate(&stream3, CU_STREAM_NON_BLOCKING));
  CUDA_REPORT_IF_ERROR(cuStreamCreate(&stream4, CU_STREAM_NON_BLOCKING));
  
  // 创建CUDA事件用于计时
  CUevent start = nullptr, stop1 = nullptr, stop2 = nullptr, stop3 = nullptr, stop4 = nullptr, stopTotal = nullptr;
  CUDA_REPORT_IF_ERROR(cuEventCreate(&start, CU_EVENT_DEFAULT));
  CUDA_REPORT_IF_ERROR(cuEventCreate(&stop1, CU_EVENT_DEFAULT));
  CUDA_REPORT_IF_ERROR(cuEventCreate(&stop2, CU_EVENT_DEFAULT));
  CUDA_REPORT_IF_ERROR(cuEventCreate(&stop3, CU_EVENT_DEFAULT));
  CUDA_REPORT_IF_ERROR(cuEventCreate(&stop4, CU_EVENT_DEFAULT));
  CUDA_REPORT_IF_ERROR(cuEventCreate(&stopTotal, CU_EVENT_DEFAULT));
  
  // 设置卷积参数
  int n = 1, c = 3, h = 1024, w = 1024;
  int k = 1024, r = 3, s = 3;
  int pad_h = 1, pad_w = 1;
  int stride_h = 1, stride_w = 1;
  int dilation_h = 1, dilation_w = 1;
  
  // 计算输出尺寸
  int out_h = (h + 2 * pad_h - (dilation_h * (r - 1) + 1)) / stride_h + 1;
  int out_w = (w + 2 * pad_w - (dilation_w * (s - 1) + 1)) / stride_w + 1;
  
  printf("Input: %dx%dx%dx%d\n", n, c, h, w);
  printf("Filter: %dx%dx%dx%d\n", k, c, r, s);
  printf("Output: %dx%dx%dx%d\n", n, k, out_h, out_w);
  
  // 分配内存大小
  size_t input_size = n * c * h * w * sizeof(float);
  size_t filter_size = k * c * r * s * sizeof(float);
  size_t bias_size = k * sizeof(float);
  size_t output_size = n * k * out_h * out_w * sizeof(float);
  
  printf("Allocating buffers (input: %.2f MB, output: %.2f MB)...\n", 
         input_size / (1024.0 * 1024.0), output_size / (1024.0 * 1024.0));
  
  // 为四个操作分配内存
  CUdeviceptr dInput1 = 0, dFilter1 = 0, dBias1 = 0, dOutput1 = 0;
  CUdeviceptr dInput2 = 0, dFilter2 = 0, dBias2 = 0, dOutput2 = 0;
  CUdeviceptr dInput3 = 0, dFilter3 = 0, dBias3 = 0, dOutput3 = 0;
  CUdeviceptr dInput4 = 0, dFilter4 = 0, dBias4 = 0, dOutput4 = 0;
  
  // 第一个操作内存
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dInput1, input_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dFilter1, filter_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dBias1, bias_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dOutput1, output_size));
  
  // 第二个操作内存
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dInput2, input_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dFilter2, filter_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dBias2, bias_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dOutput2, output_size));
  
  // 第三个操作内存
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dInput3, input_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dFilter3, filter_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dBias3, bias_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dOutput3, output_size));
  
  // 第四个操作内存
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dInput4, input_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dFilter4, filter_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dBias4, bias_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dOutput4, output_size));
  
  // 转换为void*指针
  void* input1 = reinterpret_cast<void*>(dInput1);
  void* filter1 = reinterpret_cast<void*>(dFilter1);
  void* bias1 = reinterpret_cast<void*>(dBias1);
  void* output1 = reinterpret_cast<void*>(dOutput1);
  
  void* input2 = reinterpret_cast<void*>(dInput2);
  void* filter2 = reinterpret_cast<void*>(dFilter2);
  void* bias2 = reinterpret_cast<void*>(dBias2);
  void* output2 = reinterpret_cast<void*>(dOutput2);
  
  void* input3 = reinterpret_cast<void*>(dInput3);
  void* filter3 = reinterpret_cast<void*>(dFilter3);
  void* bias3 = reinterpret_cast<void*>(dBias3);
  void* output3 = reinterpret_cast<void*>(dOutput3);
  
  void* input4 = reinterpret_cast<void*>(dInput4);
  void* filter4 = reinterpret_cast<void*>(dFilter4);
  void* bias4 = reinterpret_cast<void*>(dBias4);
  void* output4 = reinterpret_cast<void*>(dOutput4);
  
  // 初始化内存 - 每个卷积使用不同的值以区分结果
  printf("Initializing data...\n");
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(dInput1, 0x3f800000, input_size / 4, stream1));  // 1.0f
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(dFilter1, 0x3f800000, filter_size / 4, stream1)); // 1.0f
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(dBias1, 0x3f800000, bias_size / 4, stream1));    // 1.0f
  
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(dInput2, 0x3f000000, input_size / 4, stream2));  // 0.5f
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(dFilter2, 0x3f000000, filter_size / 4, stream2)); // 0.5f
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(dBias2, 0x3f000000, bias_size / 4, stream2));    // 0.5f
  
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(dInput3, 0x3e800000, input_size / 4, stream3));  // 0.25f
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(dFilter3, 0x3e800000, filter_size / 4, stream3)); // 0.25f
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(dBias3, 0x3e800000, bias_size / 4, stream3));    // 0.25f
  
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(dInput4, 0x3f400000, input_size / 4, stream4));  // 0.75f
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(dFilter4, 0x3f400000, filter_size / 4, stream4)); // 0.75f
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(dBias4, 0x3f400000, bias_size / 4, stream4));    // 0.75f
  
  // 启动计时
  CUDA_REPORT_IF_ERROR(cuEventRecord(start, 0));
  
  printf("Starting 4 parallel convolution operations...\n");
  
  // 在不同stream上执行卷积
  mgpuCudnnConv2dForward(n, c, h, w, k, r, s, pad_h, pad_w, stride_h, stride_w,
                        dilation_h, dilation_w, input1, filter1, bias1, output1, stream1);
  CUDA_REPORT_IF_ERROR(cuEventRecord(stop1, stream1));
  
  mgpuCudnnConv2dForward(n, c, h, w, k, r, s, pad_h, pad_w, stride_h, stride_w,
                        dilation_h, dilation_w, input2, filter2, bias2, output2, stream2);
  CUDA_REPORT_IF_ERROR(cuEventRecord(stop2, stream2));
  
  mgpuCudnnConv2dForward(n, c, h, w, k, r, s, pad_h, pad_w, stride_h, stride_w,
                        dilation_h, dilation_w, input3, filter3, bias3, output3, stream3);
  CUDA_REPORT_IF_ERROR(cuEventRecord(stop3, stream3));
  
  mgpuCudnnConv2dForward(n, c, h, w, k, r, s, pad_h, pad_w, stride_h, stride_w,
                        dilation_h, dilation_w, input4, filter4, bias4, output4, stream4);
  CUDA_REPORT_IF_ERROR(cuEventRecord(stop4, stream4));
  
  // 等待所有操作完成
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream1));
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream2));
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream3));
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream4));
  CUDA_REPORT_IF_ERROR(cuEventRecord(stopTotal, 0));
  CUDA_REPORT_IF_ERROR(cuEventSynchronize(stopTotal));
  
  // 计算耗时
  float ms1 = 0.0f, ms2 = 0.0f, ms3 = 0.0f, ms4 = 0.0f, msTotal = 0.0f;
  CUDA_REPORT_IF_ERROR(cuEventElapsedTime(&ms1, start, stop1));
  CUDA_REPORT_IF_ERROR(cuEventElapsedTime(&ms2, start, stop2));
  CUDA_REPORT_IF_ERROR(cuEventElapsedTime(&ms3, start, stop3));
  CUDA_REPORT_IF_ERROR(cuEventElapsedTime(&ms4, start, stop4));
  CUDA_REPORT_IF_ERROR(cuEventElapsedTime(&msTotal, start, stopTotal));
  
  printf("Convolution 1 time: %.3f ms\n", ms1);
  printf("Convolution 2 time: %.3f ms\n", ms2);
  printf("Convolution 3 time: %.3f ms\n", ms3);
  printf("Convolution 4 time: %.3f ms\n", ms4);
  printf("Total time: %.3f ms\n", msTotal);
  printf("Parallel speedup: %.2fx\n", (ms1 + ms2 + ms3 + ms4) / msTotal);
  
  // 清理资源
  CUDA_REPORT_IF_ERROR(cuEventDestroy(start));
  CUDA_REPORT_IF_ERROR(cuEventDestroy(stop1));
  CUDA_REPORT_IF_ERROR(cuEventDestroy(stop2));
  CUDA_REPORT_IF_ERROR(cuEventDestroy(stop3));
  CUDA_REPORT_IF_ERROR(cuEventDestroy(stop4));
  CUDA_REPORT_IF_ERROR(cuEventDestroy(stopTotal));
  
  // 释放内存
  CUDA_REPORT_IF_ERROR(cuMemFree(dInput1));
  CUDA_REPORT_IF_ERROR(cuMemFree(dFilter1));
  CUDA_REPORT_IF_ERROR(cuMemFree(dBias1));
  CUDA_REPORT_IF_ERROR(cuMemFree(dOutput1));
  
  CUDA_REPORT_IF_ERROR(cuMemFree(dInput2));
  CUDA_REPORT_IF_ERROR(cuMemFree(dFilter2));
  CUDA_REPORT_IF_ERROR(cuMemFree(dBias2));
  CUDA_REPORT_IF_ERROR(cuMemFree(dOutput2));
  
  CUDA_REPORT_IF_ERROR(cuMemFree(dInput3));
  CUDA_REPORT_IF_ERROR(cuMemFree(dFilter3));
  CUDA_REPORT_IF_ERROR(cuMemFree(dBias3));
  CUDA_REPORT_IF_ERROR(cuMemFree(dOutput3));
  
  CUDA_REPORT_IF_ERROR(cuMemFree(dInput4));
  CUDA_REPORT_IF_ERROR(cuMemFree(dFilter4));
  CUDA_REPORT_IF_ERROR(cuMemFree(dBias4));
  CUDA_REPORT_IF_ERROR(cuMemFree(dOutput4));
  
  // 销毁流
  CUDA_REPORT_IF_ERROR(cuStreamDestroy(stream1));
  CUDA_REPORT_IF_ERROR(cuStreamDestroy(stream2));
  CUDA_REPORT_IF_ERROR(cuStreamDestroy(stream3));
  CUDA_REPORT_IF_ERROR(cuStreamDestroy(stream4));
  
  // 清理cuDNN句柄
  mgpuCudnnCleanup();
  
  printf("Parallel cuDNN test completed successfully\n");
}

// Add our custom LIF wrapper
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuFusedLIFUpdate(void *membranePotential, void *inputCurrent, 
                  void *threshold, void *resetValue, float leakFactor,
                  void *newMembrane, void *spikeOutput, 
                  size_t size, CUstream stream) {
  // Get the current CUDA context or create one if needed
  CUcontext current = nullptr;
  CUresult err = cuCtxGetCurrent(&current);
  if (err != CUDA_SUCCESS || current == nullptr) {
    // No context exists, create one
    CUdevice device = 0;
    err = cuDeviceGet(&device, 0);
    if (err != CUDA_SUCCESS) {
      fprintf(stderr, "ERROR: Failed to get CUDA device: %d\n", err);
      return;
    }
    
    err = cuCtxCreate(&current, 0, device);
    if (err != CUDA_SUCCESS) {
      fprintf(stderr, "ERROR: Failed to create CUDA context: %d\n", err);
      return;
    }
  }
  
  // Validate inputs
  if (membranePotential == nullptr || inputCurrent == nullptr || 
      threshold == nullptr || resetValue == nullptr ||
      newMembrane == nullptr || spikeOutput == nullptr || size == 0) {
    fprintf(stderr, "ERROR: Invalid inputs to mgpuFusedLIFUpdate\n");
    return;
  }
  
  // Convert stream if necessary
  cudaStream_t cudaStream = nullptr;
  if (stream != nullptr) {
    // Use the provided stream
    cudaStream = reinterpret_cast<cudaStream_t>(stream);
  } else {
    // Use the default stream
    cudaStream = cudaStreamPerThread;
  }
  
  // Call the CUDA kernel wrapper with error handling
  launchFusedLIFUpdate(
      reinterpret_cast<float*>(membranePotential),
      reinterpret_cast<float*>(inputCurrent),
      reinterpret_cast<float*>(threshold),
      reinterpret_cast<float*>(resetValue),
      leakFactor,
      reinterpret_cast<float*>(newMembrane),
      reinterpret_cast<bool*>(spikeOutput),
      size,
      cudaStream
  );
  
  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    fprintf(stderr, "ERROR: CUDA kernel launch failed: %s\n", 
            cudaGetErrorString(cudaErr));
  }
  
  // Note: We don't destroy the context here because it might be used by other operations
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuLifBenchmark() {
  printf("Starting LIF neuron benchmark...\n");
  
  // Declare all resources at the beginning
  CUcontext context = nullptr;
  CUstream stream = nullptr;
  CUevent start = nullptr, stop = nullptr;
  CUdeviceptr dMembrane = 0, dInput = 0, dThreshold = 0, dReset = 0, dSpike = 0, dNewMembrane = 0;
  float* host_membrane = nullptr;
  bool* host_spikes = nullptr;
  bool success = true;
  
  // Initialize CUDA
  CUresult err = cuInit(0);
  if (err != CUDA_SUCCESS) {
    printf("CUDA initialization failed with error code %d\n", err);
    return;
  }
  
  // Get a CUDA device
  CUdevice device = 0;
  err = cuDeviceGet(&device, 0);
  if (err != CUDA_SUCCESS) {
    printf("cuDeviceGet failed with error code %d\n", err);
    return;
  }
  
  // Create a CUDA context
  err = cuCtxCreate(&context, 0, device);
  if (err != CUDA_SUCCESS) {
    printf("cuCtxCreate failed with error code %d\n", err);
    return;
  }
  
  // Create a stream
  err = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
  if (err != CUDA_SUCCESS) {
    printf("cuStreamCreate failed with error code %d\n", err);
    success = false;
  }
  
  // Create CUDA events for timing
  if (success) {
    err = cuEventCreate(&start, CU_EVENT_DEFAULT);
    if (err != CUDA_SUCCESS) {
      printf("cuEventCreate failed with error code %d\n", err);
      success = false;
    }
  }
  
  if (success) {
    err = cuEventCreate(&stop, CU_EVENT_DEFAULT);
    if (err != CUDA_SUCCESS) {
      printf("cuEventCreate failed with error code %d\n", err);
      success = false;
    }
  }
  
  // Set LIF neuron parameters - use a smaller network for testing
  int batch_size = 16;
  int num_neurons = 1024;
  float threshold = 1.0f;
  float reset_val = 0.0f;
  float leak_factor = 0.1f;
  
  printf("LIF parameters: batch_size=%d, num_neurons=%d\n", batch_size, num_neurons);
  printf("Threshold=%.2f, Reset=%.2f, Leak=%.2f\n", threshold, reset_val, leak_factor);
  
  // Calculate memory sizes
  size_t membrane_size = batch_size * num_neurons * sizeof(float);
  size_t input_size = batch_size * num_neurons * sizeof(float);
  size_t threshold_size = num_neurons * sizeof(float);
  size_t reset_size = num_neurons * sizeof(float);
  size_t spike_size = batch_size * num_neurons * sizeof(bool);
  
  printf("Allocating memory...\n");
  
  // Allocate device memory
  if (success) {
    err = cuMemAlloc(&dMembrane, membrane_size);
    if (err != CUDA_SUCCESS) {
      printf("cuMemAlloc failed with error code %d\n", err);
      success = false;
    }
  }
  
  if (success) {
    err = cuMemAlloc(&dInput, input_size);
    if (err != CUDA_SUCCESS) {
      printf("cuMemAlloc failed with error code %d\n", err);
      success = false;
    }
  }
  
  if (success) {
    err = cuMemAlloc(&dThreshold, threshold_size);
    if (err != CUDA_SUCCESS) {
      printf("cuMemAlloc failed with error code %d\n", err);
      success = false;
    }
  }
  
  if (success) {
    err = cuMemAlloc(&dReset, reset_size);
    if (err != CUDA_SUCCESS) {
      printf("cuMemAlloc failed with error code %d\n", err);
      success = false;
    }
  }
  
  if (success) {
    err = cuMemAlloc(&dSpike, spike_size);
    if (err != CUDA_SUCCESS) {
      printf("cuMemAlloc failed with error code %d\n", err);
      success = false;
    }
  }
  
  if (success) {
    err = cuMemAlloc(&dNewMembrane, membrane_size);
    if (err != CUDA_SUCCESS) {
      printf("cuMemAlloc failed with error code %d\n", err);
      success = false;
    }
  }
  
  // Initialize memory with test values
  if (success) {
    printf("Initializing data...\n");
    
    err = cuMemsetD32(dMembrane, 0x3f000000, membrane_size / 4);     // 0.5f
    if (err != CUDA_SUCCESS) {
      printf("cuMemsetD32 failed with error code %d\n", err);
      success = false;
    }
  }
  
  if (success) {
    err = cuMemsetD32(dInput, 0x3f000000, input_size / 4);           // 0.5f
    if (err != CUDA_SUCCESS) {
      printf("cuMemsetD32 failed with error code %d\n", err);
      success = false;
    }
  }
  
  if (success) {
    err = cuMemsetD32(dThreshold, 0x3f800000, threshold_size / 4);   // 1.0f
    if (err != CUDA_SUCCESS) {
      printf("cuMemsetD32 failed with error code %d\n", err);
      success = false;
    }
  }
  
  if (success) {
    err = cuMemsetD32(dReset, 0x00000000, reset_size / 4);           // 0.0f
    if (err != CUDA_SUCCESS) {
      printf("cuMemsetD32 failed with error code %d\n", err);
      success = false;
    }
  }
  
  if (success) {
    err = cuMemsetD8(dSpike, 0, spike_size);                         // false
    if (err != CUDA_SUCCESS) {
      printf("cuMemsetD8 failed with error code %d\n", err);
      success = false;
    }
  }
  
  // Convert to void* pointers and run benchmark if still successful
  if (success) {
    void* membrane = reinterpret_cast<void*>(dMembrane);
    void* input = reinterpret_cast<void*>(dInput);
    void* threshold_ptr = reinterpret_cast<void*>(dThreshold);
    void* reset = reinterpret_cast<void*>(dReset);
    void* spike = reinterpret_cast<void*>(dSpike);
    void* newMembrane = reinterpret_cast<void*>(dNewMembrane);
    
    // Warm-up run
    printf("Performing warm-up run...\n");
    mgpuFusedLIFUpdate(
        membrane, input, threshold_ptr, reset, leak_factor,
        newMembrane, spike, batch_size * num_neurons, stream);
    
    err = cuStreamSynchronize(stream);
    if (err != CUDA_SUCCESS) {
      printf("cuStreamSynchronize failed with error code %d\n", err);
      success = false;
    }
    
    // Run benchmark
    if (success) {
      printf("Running benchmark...\n");
      int num_iterations = 100;
      
      err = cuEventRecord(start, stream);
      if (err != CUDA_SUCCESS) {
        printf("cuEventRecord failed with error code %d\n", err);
        success = false;
      }
      
      if (success) {
        for (int i = 0; i < num_iterations; i++) {
          mgpuFusedLIFUpdate(
              membrane, input, threshold_ptr, reset, leak_factor,
              newMembrane, spike, batch_size * num_neurons, stream);
        }
        
        err = cuEventRecord(stop, stream);
        if (err != CUDA_SUCCESS) {
          printf("cuEventRecord failed with error code %d\n", err);
          success = false;
        }
      }
      
      if (success) {
        err = cuEventSynchronize(stop);
        if (err != CUDA_SUCCESS) {
          printf("cuEventSynchronize failed with error code %d\n", err);
          success = false;
        }
      }
      
      // Calculate elapsed time
      if (success) {
        float milliseconds = 0.0f;
        err = cuEventElapsedTime(&milliseconds, start, stop);
        if (err != CUDA_SUCCESS) {
          printf("cuEventElapsedTime failed with error code %d\n", err);
          success = false;
        } else {
          printf("Benchmark results:\n");
          printf("Executed %d iterations\n", num_iterations);
          printf("Total time: %.3f ms\n", milliseconds);
          printf("Average time per iteration: %.3f ms\n", milliseconds / num_iterations);
          printf("Throughput: %.2f million neurons/s\n", 
                 (batch_size * num_neurons * num_iterations) / (milliseconds * 1000.0f));
        }
      }
      
      // Validation - copy results back to host to verify
      if (success) {
        printf("Validating results...\n");
        host_membrane = new float[batch_size * num_neurons];
        host_spikes = new bool[batch_size * num_neurons];
        
        if (!host_membrane || !host_spikes) {
          printf("Failed to allocate host memory for validation\n");
          success = false;
        } else {
          err = cuMemcpyDtoH(host_membrane, dNewMembrane, membrane_size);
          if (err != CUDA_SUCCESS) {
            printf("cuMemcpyDtoH failed with error code %d\n", err);
            success = false;
          }
          
          if (success) {
            err = cuMemcpyDtoH(host_spikes, dSpike, spike_size);
            if (err != CUDA_SUCCESS) {
              printf("cuMemcpyDtoH failed with error code %d\n", err);
              success = false;
            }
          }
          
          if (success) {
            // Simple validation check
            int spike_count = 0;
            for (int i = 0; i < batch_size * num_neurons; i++) {
              if (host_spikes[i]) {
                spike_count++;
              }
            }
            
            printf("Validation complete. Spike count: %d (%.2f%%)\n", 
                   spike_count, 100.0f * spike_count / (batch_size * num_neurons));
            
            printf("LIF benchmark completed successfully\n");
          }
        }
      }
    }
  }
  
  // Clean up resources
  if (host_membrane) delete[] host_membrane;
  if (host_spikes) delete[] host_spikes;
  if (dMembrane) CUDA_REPORT_IF_ERROR(cuMemFree(dMembrane));
  if (dInput) CUDA_REPORT_IF_ERROR(cuMemFree(dInput));
  if (dThreshold) CUDA_REPORT_IF_ERROR(cuMemFree(dThreshold));
  if (dReset) CUDA_REPORT_IF_ERROR(cuMemFree(dReset));
  if (dSpike) CUDA_REPORT_IF_ERROR(cuMemFree(dSpike));
  if (dNewMembrane) CUDA_REPORT_IF_ERROR(cuMemFree(dNewMembrane));
  if (start) CUDA_REPORT_IF_ERROR(cuEventDestroy(start));
  if (stop) CUDA_REPORT_IF_ERROR(cuEventDestroy(stop));
  if (stream) CUDA_REPORT_IF_ERROR(cuStreamDestroy(stream));
  if (context) CUDA_REPORT_IF_ERROR(cuCtxDestroy(context));
}


extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuCudnnTensorOpsTest() {
  printf("Starting cuDNN tensor operations test...\n");
  
  // 创建CUDA上下文
  ScopedContext scopedContext;
  
  // 创建流用于执行
  CUstream stream = nullptr;
  CUDA_REPORT_IF_ERROR(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  
  // 创建CUDA事件用于计时
  CUevent start = nullptr, stop = nullptr;
  CUDA_REPORT_IF_ERROR(cuEventCreate(&start, CU_EVENT_DEFAULT));
  CUDA_REPORT_IF_ERROR(cuEventCreate(&stop, CU_EVENT_DEFAULT));
  
  // 设置测试参数 - 使用更小的尺寸便于测试
  int n = 512, c = 3, h = 1024, w = 1024;
  size_t tensor_size = n * c * h * w * sizeof(float);
  
  printf("Test parameters: tensor shape = [%d, %d, %d, %d]\n", n, c, h, w);
  
  // 分配内存
  printf("Allocating device memory...\n");
  
  CUdeviceptr dA = 0, dB = 0, dResult = 0, dTemp = 0;
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dA, tensor_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dB, tensor_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dResult, tensor_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dTemp, tensor_size));
  
  void* A = reinterpret_cast<void*>(dA);
  void* B = reinterpret_cast<void*>(dB);
  void* Result = reinterpret_cast<void*>(dResult);
  void* Temp = reinterpret_cast<void*>(dTemp);
  
  // 分配主机内存用于验证
  float* host_A = new float[n * c * h * w];
  float* host_B = new float[n * c * h * w];
  float* host_Result = new float[n * c * h * w];
  float* host_Expected = new float[n * c * h * w];
  
  // 初始化测试数据
  for (int i = 0; i < n * c * h * w; i++) {
    host_A[i] = 2.0f;  // A全部初始化为2.0
    host_B[i] = 4.0f;  // B全部初始化为4.0
  }
  
  // 拷贝数据到设备
  CUDA_REPORT_IF_ERROR(cuMemcpyHtoD(dA, host_A, tensor_size));
  CUDA_REPORT_IF_ERROR(cuMemcpyHtoD(dB, host_B, tensor_size));
  
  printf("Running tensor operations tests...\n");
  
  // 1. 测试Mul操作 (A * B = 2.0 * 4.0 = 8.0)
  printf("Testing Mul operation...\n");
  
  CUDA_REPORT_IF_ERROR(cuEventRecord(start, stream));
  
  mgpuCudnnMul(A, B, Result, n, c, h, w, stream);
  
  CUDA_REPORT_IF_ERROR(cuEventRecord(stop, stream));
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream));
  
  // 计算耗时
  float ms_mul = 0.0f;
  CUDA_REPORT_IF_ERROR(cuEventElapsedTime(&ms_mul, start, stop));
  
  // 验证结果
  CUDA_REPORT_IF_ERROR(cuMemcpyDtoH(host_Result, dResult, tensor_size));
  
  bool mul_correct = true;
  for (int i = 0; i < n * c * h * w; i++) {
    host_Expected[i] = host_A[i] * host_B[i];
    if (std::abs(host_Result[i] - host_Expected[i]) > 1e-5) {
      mul_correct = false;
      printf("Mul error at index %d: %f vs %f\n", i, host_Result[i], host_Expected[i]);
      break;
    }
  }
  
  printf("Mul test %s (took %.3f ms)\n", mul_correct ? "PASSED" : "FAILED", ms_mul);
  
  // 2. 测试Add操作 (A + B = 2.0 + 4.0 = 6.0)
  printf("Testing Add operation...\n");
  
  CUDA_REPORT_IF_ERROR(cuEventRecord(start, stream));
  
  mgpuCudnnAdd(A, B, Result, n, c, h, w, stream);
  
  CUDA_REPORT_IF_ERROR(cuEventRecord(stop, stream));
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream));
  
  // 计算耗时
  float ms_add = 0.0f;
  CUDA_REPORT_IF_ERROR(cuEventElapsedTime(&ms_add, start, stop));
  
  // 验证结果
  CUDA_REPORT_IF_ERROR(cuMemcpyDtoH(host_Result, dResult, tensor_size));
  
  bool add_correct = true;
  for (int i = 0; i < n * c * h * w; i++) {
    host_Expected[i] = host_A[i] + host_B[i];
    if (std::abs(host_Result[i] - host_Expected[i]) > 1e-5) {
      add_correct = false;
      printf("Add error at index %d: %f vs %f\n", i, host_Result[i], host_Expected[i]);
      break;
    }
  }
  
  printf("Add test %s (took %.3f ms)\n", add_correct ? "PASSED" : "FAILED", ms_add);
  
  // 3. 测试Neg操作 (-A = -2.0)
  printf("Testing Neg operation...\n");
  
  CUDA_REPORT_IF_ERROR(cuEventRecord(start, stream));
  
  mgpuCudnnNeg(A, Result, n, c, h, w, stream);
  
  CUDA_REPORT_IF_ERROR(cuEventRecord(stop, stream));
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream));
  
  // 计算耗时
  float ms_neg = 0.0f;
  CUDA_REPORT_IF_ERROR(cuEventElapsedTime(&ms_neg, start, stop));
  
  // 验证结果
  CUDA_REPORT_IF_ERROR(cuMemcpyDtoH(host_Result, dResult, tensor_size));
  
  bool neg_correct = true;
  for (int i = 0; i < n * c * h * w; i++) {
    host_Expected[i] = -host_A[i];
    if (std::abs(host_Result[i] - host_Expected[i]) > 1e-5) {
      neg_correct = false;
      printf("Neg error at index %d: %f vs %f\n", i, host_Result[i], host_Expected[i]);
      break;
    }
  }
  
  printf("Neg test %s (took %.3f ms)\n", neg_correct ? "PASSED" : "FAILED", ms_neg);
  
  // 4. 组合测试 - 测试LIF算子中的操作序列
  printf("Testing LIF operation sequence...\n");
  
  // LIF后续操作的序列测试:
  // 1. Mul: spikes * const1
  // 2. Neg: -spikes
  // 3. Add: (-spikes) + const2
  // 4. Mul: ((-spikes) + const2) * voltage
  // 5. Add: (spikes * const1) + (((-spikes) + const2) * voltage)
  
  // 为简化测试，我们使用以下模拟值:
  // spikes = A (2.0), voltage = B (4.0), const1 = 0.0, const2 = 1.0
  
  // 重置结果缓冲区
  CUDA_REPORT_IF_ERROR(cuMemsetD32(dResult, 0, tensor_size / sizeof(float)));
  CUDA_REPORT_IF_ERROR(cuMemsetD32(dTemp, 0, tensor_size / sizeof(float)));
  
  // 创建常量缓冲区
  CUdeviceptr dConst1 = 0, dConst2 = 0;
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dConst1, tensor_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dConst2, tensor_size));
  
  // 初始化常量
  float const1_value = 0.0f;
  float const2_value = 1.0f;
  CUDA_REPORT_IF_ERROR(cuMemsetD32(dConst1, *(unsigned int*)&const1_value, tensor_size / sizeof(float)));
  CUDA_REPORT_IF_ERROR(cuMemsetD32(dConst2, *(unsigned int*)&const2_value, tensor_size / sizeof(float)));
  
  void* Const1 = reinterpret_cast<void*>(dConst1);
  void* Const2 = reinterpret_cast<void*>(dConst2);
  
  CUDA_REPORT_IF_ERROR(cuEventRecord(start, stream));
  
  // 1. temp1 = spikes * const1 (A * 0.0 = 0.0)
  mgpuCudnnMul(A, Const1, Temp, n, c, h, w, stream);
  
  // 2. temp2 = -spikes (-A = -2.0)
  mgpuCudnnNeg(A, Result, n, c, h, w, stream);
  
  // 3. temp3 = (-spikes) + const2 (temp2 + 1.0 = -2.0 + 1.0 = -1.0)
  mgpuCudnnAdd(Result, Const2, Result, n, c, h, w, stream);
  
  // 4. temp4 = ((-spikes) + const2) * voltage (Result * B = -1.0 * 4.0 = -4.0)
  mgpuCudnnMul(Result, B, Result, n, c, h, w, stream);
  
  // 5. final = (spikes * const1) + (((-spikes) + const2) * voltage) (Temp + Result = 0.0 + (-4.0) = -4.0)
  mgpuCudnnAdd(Temp, Result, Result, n, c, h, w, stream);
  
  CUDA_REPORT_IF_ERROR(cuEventRecord(stop, stream));
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream));
  
  // 计算耗时
  float ms_lif = 0.0f;
  CUDA_REPORT_IF_ERROR(cuEventElapsedTime(&ms_lif, start, stop));
  
  // 验证结果
  CUDA_REPORT_IF_ERROR(cuMemcpyDtoH(host_Result, dResult, tensor_size));
  
  // 手动计算预期结果
  for (int i = 0; i < n * c * h * w; i++) {
    float step1 = host_A[i] * const1_value;                 // 0.0
    float step2 = -host_A[i];                               // -2.0
    float step3 = step2 + const2_value;                     // -1.0
    float step4 = step3 * host_B[i];                        // -4.0
    host_Expected[i] = step1 + step4;                       // -4.0
  }
  
  bool lif_correct = true;
  for (int i = 0; i < n * c * h * w; i++) {
    if (std::abs(host_Result[i] - host_Expected[i]) > 1e-5) {
      lif_correct = false;
      printf("LIF sequence error at index %d: %f vs %f\n", i, host_Result[i], host_Expected[i]);
      break;
    }
  }
  
  printf("LIF sequence test %s (took %.3f ms)\n", lif_correct ? "PASSED" : "FAILED", ms_lif);
  
  // 总结测试结果
  printf("\nCuDNN Tensor Operations Test Summary:\n");
  printf("Mul test: %s (%.3f ms)\n", mul_correct ? "PASSED" : "FAILED", ms_mul);
  printf("Add test: %s (%.3f ms)\n", add_correct ? "PASSED" : "FAILED", ms_add);
  printf("Neg test: %s (%.3f ms)\n", neg_correct ? "PASSED" : "FAILED", ms_neg);
  printf("LIF sequence test: %s (%.3f ms)\n", lif_correct ? "PASSED" : "FAILED", ms_lif);
  
  // 清理资源
  delete[] host_A;
  delete[] host_B;
  delete[] host_Result;
  delete[] host_Expected;
  
  CUDA_REPORT_IF_ERROR(cuMemFree(dA));
  CUDA_REPORT_IF_ERROR(cuMemFree(dB));
  CUDA_REPORT_IF_ERROR(cuMemFree(dResult));
  CUDA_REPORT_IF_ERROR(cuMemFree(dTemp));
  CUDA_REPORT_IF_ERROR(cuMemFree(dConst1));
  CUDA_REPORT_IF_ERROR(cuMemFree(dConst2));
  
  CUDA_REPORT_IF_ERROR(cuEventDestroy(start));
  CUDA_REPORT_IF_ERROR(cuEventDestroy(stop));
  CUDA_REPORT_IF_ERROR(cuStreamDestroy(stream));
  
  // 清理cuDNN句柄
  mgpuCudnnCleanup();
  
  printf("CuDNN tensor operations test completed\n");
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuFullyConnectedTest() {
  printf("Starting fully connected layer (FC) test...\n");
  
  // 创建CUDA上下文
  ScopedContext scopedContext;
  
  // 创建流用于执行
  CUstream stream = nullptr;
  CUDA_REPORT_IF_ERROR(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  
  // 创建CUDA事件用于计时
  CUevent start = nullptr, stop = nullptr;
  CUDA_REPORT_IF_ERROR(cuEventCreate(&start, CU_EVENT_DEFAULT));
  CUDA_REPORT_IF_ERROR(cuEventCreate(&stop, CU_EVENT_DEFAULT));
  
  // 设置FC层参数
  int batch_size = 128;
  int input_features = 256;
  int output_features = 128;
  
  printf("FC parameters: batch_size=%d, input_features=%d, output_features=%d\n", 
         batch_size, input_features, output_features);
  
  // 分配内存
  printf("Allocating device memory...\n");
  
  size_t input_size = batch_size * input_features * sizeof(float);
  size_t weight_size = output_features * input_features * sizeof(float);
  size_t bias_size = output_features * sizeof(float);
  size_t output_size = batch_size * output_features * sizeof(float);
  
  CUdeviceptr dInput = 0, dWeight = 0, dBias = 0, dOutput1 = 0, dOutput2 = 0;
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dInput, input_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dWeight, weight_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dBias, bias_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dOutput1, output_size));
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dOutput2, output_size));
  
  void* input = reinterpret_cast<void*>(dInput);
  void* weight = reinterpret_cast<void*>(dWeight);
  void* bias = reinterpret_cast<void*>(dBias);
  void* output1 = reinterpret_cast<void*>(dOutput1);
  void* output2 = reinterpret_cast<void*>(dOutput2);
  
  // 分配主机内存用于初始化和验证
  float* host_input = new float[batch_size * input_features];
  float* host_weight = new float[output_features * input_features];
  float* host_bias = new float[output_features];
  float* host_output1 = new float[batch_size * output_features];
  float* host_output2 = new float[batch_size * output_features];
  float* host_expected1 = new float[batch_size * output_features];
  float* host_expected2 = new float[batch_size * output_features];
  
  // 初始化测试数据
  printf("Initializing test data...\n");
  
  // 所有输入值设为1.0
  for (int i = 0; i < batch_size * input_features; i++) {
    host_input[i] = 1.0f;
  }
  
  // 所有权重值设为0.5
  for (int i = 0; i < output_features * input_features; i++) {
    host_weight[i] = 0.5f;
  }
  
  // 所有偏置值设为2.0
  for (int i = 0; i < output_features; i++) {
    host_bias[i] = 2.0f;
  }
  
  // 拷贝数据到设备
  CUDA_REPORT_IF_ERROR(cuMemcpyHtoD(dInput, host_input, input_size));
  CUDA_REPORT_IF_ERROR(cuMemcpyHtoD(dWeight, host_weight, weight_size));
  CUDA_REPORT_IF_ERROR(cuMemcpyHtoD(dBias, host_bias, bias_size));
  
  printf("Running FC tests...\n");
  
  // 1. 测试带偏置的FC层
  printf("Testing FC with bias...\n");
  
  CUDA_REPORT_IF_ERROR(cuEventRecord(start, stream));
  
  mgpuCulibsFullyConnectedForward(
      batch_size, input_features, output_features,
      input, weight, bias, output1, stream);
  
  CUDA_REPORT_IF_ERROR(cuEventRecord(stop, stream));
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream));
  
  // 计算耗时
  float ms_fc_with_bias = 0.0f;
  CUDA_REPORT_IF_ERROR(cuEventElapsedTime(&ms_fc_with_bias, start, stop));
  
  // 验证结果
  CUDA_REPORT_IF_ERROR(cuMemcpyDtoH(host_output1, dOutput1, output_size));
  
  // 手动计算预期结果: output = input * weight^T + bias
  // 每个输出元素是input_features个1.0与input_features个0.5的点积，再加上偏置2.0
  // = input_features * 0.5 + 2.0 = input_features/2 + 2.0
  float expected_value_with_bias = input_features * 0.5f + 2.0f;
  for (int i = 0; i < batch_size * output_features; i++) {
    host_expected1[i] = expected_value_with_bias;
  }
  
  bool fc_bias_correct = true;
  for (int i = 0; i < batch_size * output_features; i++) {
    if (std::abs(host_output1[i] - host_expected1[i]) > 1e-5) {
      fc_bias_correct = false;
      printf("FC with bias error at index %d: %f vs %f\n", 
             i, host_output1[i], host_expected1[i]);
      break;
    }
  }
  
  printf("FC with bias test %s (took %.3f ms)\n", 
         fc_bias_correct ? "PASSED" : "FAILED", ms_fc_with_bias);
  
  // 2. 测试不带偏置的FC层
  printf("Testing FC without bias...\n");
  
  CUDA_REPORT_IF_ERROR(cuEventRecord(start, stream));
  
  mgpuCulibsFullyConnectedForward(
      batch_size, input_features, output_features,
      input, weight, nullptr, output2, stream);
  
  CUDA_REPORT_IF_ERROR(cuEventRecord(stop, stream));
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream));
  
  // 计算耗时
  float ms_fc_no_bias = 0.0f;
  CUDA_REPORT_IF_ERROR(cuEventElapsedTime(&ms_fc_no_bias, start, stop));
  
  // 验证结果
  CUDA_REPORT_IF_ERROR(cuMemcpyDtoH(host_output2, dOutput2, output_size));
  
  // 手动计算预期结果: output = input * weight^T
  // 每个输出元素是input_features个1.0与input_features个0.5的点积
  // = input_features * 0.5 = input_features/2
  float expected_value_no_bias = input_features * 0.5f;
  for (int i = 0; i < batch_size * output_features; i++) {
    host_expected2[i] = expected_value_no_bias;
  }
  
  bool fc_no_bias_correct = true;
  for (int i = 0; i < batch_size * output_features; i++) {
    if (std::abs(host_output2[i] - host_expected2[i]) > 1e-5) {
      fc_no_bias_correct = false;
      printf("FC without bias error at index %d: %f vs %f\n", 
             i, host_output2[i], host_expected2[i]);
      break;
    }
  }
  
  printf("FC without bias test %s (took %.3f ms)\n", 
         fc_no_bias_correct ? "PASSED" : "FAILED", ms_fc_no_bias);
  
  // 验证带偏置和不带偏置的结果差异
  bool bias_diff_correct = true;
  for (int i = 0; i < batch_size * output_features; i++) {
    float expected_diff = host_expected1[i] - host_expected2[i]; // 应该等于偏置值2.0
    float actual_diff = host_output1[i] - host_output2[i];
    if (std::abs(actual_diff - expected_diff) > 1e-5) {
      bias_diff_correct = false;
      printf("Bias difference error at index %d: %f vs %f\n", 
             i, actual_diff, expected_diff);
      break;
    }
  }
  
  printf("Bias effect verification %s\n", 
         bias_diff_correct ? "PASSED" : "FAILED");
  
  // 总结测试结果
  printf("\nFC Operations Test Summary:\n");
  printf("FC with bias test: %s (%.3f ms)\n", 
         fc_bias_correct ? "PASSED" : "FAILED", ms_fc_with_bias);
  printf("FC without bias test: %s (%.3f ms)\n", 
         fc_no_bias_correct ? "PASSED" : "FAILED", ms_fc_no_bias);
  printf("Bias effect verification: %s\n", 
         bias_diff_correct ? "PASSED" : "FAILED");
  
  // 检查性能差异
  float perf_diff = (ms_fc_with_bias / ms_fc_no_bias - 1.0f) * 100.0f;
  printf("Performance overhead of bias: %.2f%%\n", perf_diff);
  
  // 清理资源
  delete[] host_input;
  delete[] host_weight;
  delete[] host_bias;
  delete[] host_output1;
  delete[] host_output2;
  delete[] host_expected1;
  delete[] host_expected2;
  
  CUDA_REPORT_IF_ERROR(cuMemFree(dInput));
  CUDA_REPORT_IF_ERROR(cuMemFree(dWeight));
  CUDA_REPORT_IF_ERROR(cuMemFree(dBias));
  CUDA_REPORT_IF_ERROR(cuMemFree(dOutput1));
  CUDA_REPORT_IF_ERROR(cuMemFree(dOutput2));
  
  CUDA_REPORT_IF_ERROR(cuEventDestroy(start));
  CUDA_REPORT_IF_ERROR(cuEventDestroy(stop));
  CUDA_REPORT_IF_ERROR(cuStreamDestroy(stream));
  
  // 清理句柄（如果您的清理函数会清理所有句柄，包括cuBLAS和cuDNN）
  // 如果您有单独的清理函数，可以分别调用
  mgpuCudnnCleanup();
  mgpuCublasCleanup();
  
  printf("FC operations test completed\n");
}

///
/// Runtime methods using CUDA 12.0+ driver
///

#if (CUDA_VERSION >= 12000)

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuLaunchClusterKernel(
    CUfunction function, intptr_t clusterX, intptr_t clusterY,
    intptr_t clusterZ, intptr_t gridX, intptr_t gridY, intptr_t gridZ,
    intptr_t blockX, intptr_t blockY, intptr_t blockZ, int32_t smem,
    CUstream stream, void **params, void **extra, size_t /*paramsCount*/) {
  ScopedContext scopedContext;
  if (smem > 0) {
    // Avoid checking driver as it's more expensive than if statement
    int32_t maxShmem = 0;
    CUdevice device = getDefaultCuDevice();
    CUDA_REPORT_IF_ERROR(cuDeviceGet(&device, /*ordinal=*/defaultDevice));
    CUDA_REPORT_IF_ERROR(cuDeviceGetAttribute(
        &maxShmem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        device));
    if (maxShmem < smem) {
      fprintf(stderr,
              "Requested shared memory (%dkb) is larger than maximum allowed "
              "shared memory (%dkb) for this device\n",
              smem, maxShmem);
    }
    CUDA_REPORT_IF_ERROR(cuFuncSetAttribute(
        function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
  }
  CUlaunchConfig config;
  config.gridDimX = gridX;
  config.gridDimY = gridY;
  config.gridDimZ = gridZ;
  config.blockDimX = blockX;
  config.blockDimY = blockY;
  config.blockDimZ = blockZ;
  config.sharedMemBytes = smem;
  config.hStream = stream;
  CUlaunchAttribute launchAttr[2];
  launchAttr[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  launchAttr[0].value.clusterDim.x = clusterX;
  launchAttr[0].value.clusterDim.y = clusterY;
  launchAttr[0].value.clusterDim.z = clusterZ;
  launchAttr[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
  launchAttr[1].value.clusterSchedulingPolicyPreference =
      CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
  config.numAttrs = 2;
  config.attrs = launchAttr;

  debug_print("Launching kernel,"
              "cluster: %ld, %ld, %ld, "
              "grid=%ld,%ld,%ld, "
              "threads: %ld, %ld, %ld, "
              "smem: %dkb\n",
              clusterX, clusterY, clusterZ, gridX, gridY, gridZ, blockX, blockY,
              blockZ, smem);

  CUDA_REPORT_IF_ERROR(cuLaunchKernelEx(&config, function, params, extra));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuTensorMapEncodeTiled(
    CUtensorMap *tensorMap,             // Tensor map object
    CUtensorMapDataType tensorDataType, // Tensor data type
    cuuint32_t tensorRank,              // Dimensionality of tensor
    void *globalAddress,                // Starting address
    const cuuint64_t *globalDim,        // Tensor size (number of elements)
    const cuuint64_t *globalStrides,    // Stride size (in bytes)
    const cuuint32_t *boxDim,           // Traversal box (number of elments)
    const cuuint32_t *elementStrides,   // Traversal stride
    CUtensorMapInterleave interleave,   // Type of interleaved layout
    CUtensorMapSwizzle swizzle,         // Bank swizzling pattern
    CUtensorMapL2promotion l2Promotion, // L2 promotion size
    CUtensorMapFloatOOBfill oobFill     // Padding zfill or NaN fill
) {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuTensorMapEncodeTiled(
      tensorMap, tensorDataType, tensorRank, globalAddress, globalDim,
      globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion,
      oobFill));
  debug_print("Created TMA descriptor\n Addr: %p\n"
              "data type : %d\n"
              "rank : %d\n"
              "globalDim[5]: %zu, %zu, %zu, %zu, %zu\n"
              "globalStrides[5]: %zu, %zu, %zu, %zu, %zu\n"
              "boxDim[5]: %u, %u, %u, %u, %u\n"
              "elementStrides[5]: %u, %u, %u, %u, %u\n"
              "interleave: %u \n"
              "swizzle: %u \n"
              "l2Promotion: %u \n"
              "oobFill: %u \n",
              (void *)&tensorMap, tensorDataType, tensorRank, globalDim[0],
              globalDim[1], globalDim[2], globalDim[3], globalDim[4],
              globalStrides[0], globalStrides[1], globalStrides[2],
              globalStrides[3], globalStrides[4], boxDim[0], boxDim[1],
              boxDim[2], boxDim[3], boxDim[4], elementStrides[0],
              elementStrides[1], elementStrides[2], elementStrides[3],
              elementStrides[4], interleave, swizzle, l2Promotion, oobFill);
}

template <int Rank>
void mgpuGetMemRefDataAndShape(void *rawDescriptor, char **addr,
                               uint64_t *globalDim, uint64_t *globalStrides,
                               const CUtensorMapDataType tensorDataType) {
  auto descriptor =
      reinterpret_cast<StridedMemRefType<char, Rank> *>(rawDescriptor);
  *addr = descriptor->data;
  for (int i = 0; i < Rank; ++i) {
    globalDim[i] = static_cast<uint64_t>(descriptor->sizes[Rank - i - 1]);
  }
  static constexpr int elementSizeInBytes[] = {1, 2, 4, 4, 8, 8, 2,
                                               4, 8, 2, 4, 4, 4};
  for (int i = 0; i < Rank - 1; ++i) {
    globalStrides[i] = static_cast<uint64_t>(
        descriptor->strides[Rank - i - 2] * elementSizeInBytes[tensorDataType]);
  }
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *mgpuTensorMapEncodeTiledMemref(
    int64_t tensorRank,                       // Dimensionality of tensor
    void *rankedDescriptor,                   // Ranked MemRef descriptor
    const CUtensorMapDataType tensorDataType, // Stride size (in bytes)
    CUtensorMapInterleave interleave,         // Type of interleaved layout
    CUtensorMapSwizzle swizzle,               // Bank swizzling pattern
    CUtensorMapL2promotion l2Promotion,       // L2 promotion size
    CUtensorMapFloatOOBfill oobFill,          // Padding zfill or NaN fill
    int64_t *inputBoxDims // Tensor size (number of elements)
) {
  CUtensorMap tensorMap;

  uint32_t boxDim[5] = {1, 1, 1, 1, 1}, elementStrides[5] = {1, 1, 1, 1, 1};
  uint64_t globalDim[5] = {1, 1, 1, 1, 1}, globalStrides[5] = {0};
  uint32_t tensorRank32 = uint32_t(tensorRank);

  char *globalAddress = nullptr;
  switch (tensorRank) {
  case 1:
    mgpuGetMemRefDataAndShape<1>(rankedDescriptor, &globalAddress, globalDim,
                                 globalStrides, tensorDataType);
    break;
  case 2:
    mgpuGetMemRefDataAndShape<2>(rankedDescriptor, &globalAddress, globalDim,
                                 globalStrides, tensorDataType);
    break;
  case 3:
    mgpuGetMemRefDataAndShape<3>(rankedDescriptor, &globalAddress, globalDim,
                                 globalStrides, tensorDataType);
    break;
  case 4:
    mgpuGetMemRefDataAndShape<4>(rankedDescriptor, &globalAddress, globalDim,
                                 globalStrides, tensorDataType);
    break;
  case 5:
    mgpuGetMemRefDataAndShape<5>(rankedDescriptor, &globalAddress, globalDim,
                                 globalStrides, tensorDataType);
    break;
  default:
    fprintf(
        stderr,
        "'mgpuTensorMapEncodeTiledMemref' failed with 'rank is too high'\n");
    return nullptr;
  }

  for (int64_t r = 0; r < tensorRank; ++r) {
    boxDim[r] = static_cast<uint32_t>(inputBoxDims[tensorRank - r - 1]);
  }

  ScopedContext scopedContext;
  mgpuTensorMapEncodeTiled(&tensorMap, tensorDataType, tensorRank32,
                           globalAddress, globalDim, globalStrides, boxDim,
                           elementStrides, interleave, swizzle, l2Promotion,
                           oobFill);
  // Copy created tensor map to device
  CUdeviceptr dTensorMap;
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dTensorMap, sizeof(CUtensorMap)));
  CUDA_REPORT_IF_ERROR(cuMemcpy(dTensorMap,
                                reinterpret_cast<CUdeviceptr>(&tensorMap),
                                sizeof(CUtensorMap)));
  return reinterpret_cast<void *>(dTensorMap);
}
#endif

#ifdef MLIR_ENABLE_CUDA_CUSPARSE

///
/// Wrapper methods for the cuSparse library.
///

// Some macro magic to get float/double alpha and beta on host.
// TODO: add support to passing alpha and beta as arguments
#define ALPHABETA(dtp, alpha, beta)                                            \
  __nv_bfloat16(alpha##16bf) = 1.0f;                                           \
  __nv_bfloat16(beta##16bf) = 1.0f;                                            \
  __half(alpha##16f) = 1.0f;                                                   \
  __half(beta##16f) = 1.0f;                                                    \
  float(alpha##f) = 1.0f;                                                      \
  float(beta##f) = 1.0f;                                                       \
  double(alpha##d) = 1.0;                                                      \
  double(beta##d) = 1.0;                                                       \
  const void *(alpha##p) = nullptr;                                            \
  const void *(beta##p) = nullptr;                                             \
  if (dtp == CUDA_R_16BF || dtp == CUDA_C_16BF) {                              \
    (alpha##p) = reinterpret_cast<void *>(&(alpha##16bf));                     \
    (beta##p) = reinterpret_cast<void *>(&(beta##16bf));                       \
  } else if (dtp == CUDA_R_16F || dtp == CUDA_C_16F) {                         \
    (alpha##p) = reinterpret_cast<void *>(&(alpha##16f));                      \
    (beta##p) = reinterpret_cast<void *>(&(beta##16f));                        \
  } else if (dtp == CUDA_R_32F || dtp == CUDA_C_32F) {                         \
    (alpha##p) = reinterpret_cast<void *>(&(alpha##f));                        \
    (beta##p) = reinterpret_cast<void *>(&(beta##f));                          \
  } else {                                                                     \
    (alpha##p) = reinterpret_cast<void *>(&(alpha##d));                        \
    (beta##p) = reinterpret_cast<void *>(&(beta##d));                          \
  }

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuCreateSparseEnv() {
  // ScopedContext is for cuda initialization.
  ScopedContext scopedContext;
  assert(!cusparse_env && "client called mgpuCreateSparseEnv() twice");
  CUSPARSE_REPORT_IF_ERROR(cusparseCreate(&cusparse_env));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuDestroySparseEnv() {
  assert(cusparse_env && "client did not call mgpuCreateSparseEnv()");
  CUSPARSE_REPORT_IF_ERROR(cusparseDestroy(cusparse_env));
  cusparse_env = nullptr;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateDnVec(intptr_t size, void *values, int32_t dtp, CUstream /*stream*/) {
  cusparseDnVecDescr_t vec = nullptr;
  auto dTp = static_cast<cudaDataType_t>(dtp);
  CUSPARSE_REPORT_IF_ERROR(cusparseCreateDnVec(&vec, size, values, dTp))
  return reinterpret_cast<void *>(vec);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroyDnVec(void *v, CUstream /*stream*/) {
  cusparseDnVecDescr_t vec = reinterpret_cast<cusparseDnVecDescr_t>(v);
  CUSPARSE_REPORT_IF_ERROR(cusparseDestroyDnVec(vec))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateDnMat(intptr_t rows, intptr_t cols, void *values, int32_t dtp,
                CUstream /*stream*/) {
  cusparseDnMatDescr_t mat = nullptr;
  auto dTp = static_cast<cudaDataType_t>(dtp);
  CUSPARSE_REPORT_IF_ERROR(cusparseCreateDnMat(&mat, rows, cols, /*ld=*/cols,
                                               values, dTp, CUSPARSE_ORDER_ROW))
  return reinterpret_cast<void *>(mat);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroyDnMat(void *m, CUstream /*stream*/) {
  cusparseDnMatDescr_t mat = reinterpret_cast<cusparseDnMatDescr_t>(m);
  CUSPARSE_REPORT_IF_ERROR(cusparseDestroyDnMat(mat))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateCoo(intptr_t rows, intptr_t cols, intptr_t nnz, void *rowIdxs,
              void *colIdxs, void *values, int32_t itp, int32_t dtp,
              CUstream /*stream*/) {
  cusparseSpMatDescr_t mat = nullptr;
  auto iTp = static_cast<cusparseIndexType_t>(itp);
  auto dTp = static_cast<cudaDataType_t>(dtp);
  CUSPARSE_REPORT_IF_ERROR(cusparseCreateCoo(&mat, rows, cols, nnz, rowIdxs,
                                             colIdxs, values, iTp,
                                             CUSPARSE_INDEX_BASE_ZERO, dTp))
  return reinterpret_cast<void *>(mat);
}

#ifdef CUSPARSE_COO_AOS // deprecated in cuSPARSE 11.2
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateCooAoS(intptr_t rows, intptr_t cols, intptr_t nnz, void *idxs,
                 void *values, int32_t itp, int32_t dtp, CUstream /*stream*/) {
  cusparseSpMatDescr_t mat = nullptr;
  auto iTp = static_cast<cusparseIndexType_t>(itp);
  auto dTp = static_cast<cudaDataType_t>(dtp);
  CUSPARSE_REPORT_IF_ERROR(cusparseCreateCooAoS(
      &mat, rows, cols, nnz, idxs, values, iTp, CUSPARSE_INDEX_BASE_ZERO, dTp))
  return reinterpret_cast<void *>(mat);
}
#endif // CUSPARSE_COO_AOS

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateCsr(intptr_t rows, intptr_t cols, intptr_t nnz, void *rowPos,
              void *colIdxs, void *values, int32_t ptp, int32_t itp,
              int32_t dtp, CUstream /*stream*/) {
  cusparseSpMatDescr_t mat = nullptr;
  auto pTp = static_cast<cusparseIndexType_t>(ptp);
  auto iTp = static_cast<cusparseIndexType_t>(itp);
  auto dTp = static_cast<cudaDataType_t>(dtp);
  CUSPARSE_REPORT_IF_ERROR(cusparseCreateCsr(&mat, rows, cols, nnz, rowPos,
                                             colIdxs, values, pTp, iTp,
                                             CUSPARSE_INDEX_BASE_ZERO, dTp))
  return reinterpret_cast<void *>(mat);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateCsc(intptr_t rows, intptr_t cols, intptr_t nnz, void *colPos,
              void *rowIdxs, void *values, int32_t ptp, int32_t itp,
              int32_t dtp, CUstream /*stream*/) {
  cusparseSpMatDescr_t mat = nullptr;
  auto pTp = static_cast<cusparseIndexType_t>(ptp);
  auto iTp = static_cast<cusparseIndexType_t>(itp);
  auto dTp = static_cast<cudaDataType_t>(dtp);
  CUSPARSE_REPORT_IF_ERROR(cusparseCreateCsc(&mat, rows, cols, nnz, colPos,
                                             rowIdxs, values, pTp, iTp,
                                             CUSPARSE_INDEX_BASE_ZERO, dTp))
  return reinterpret_cast<void *>(mat);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateBsr(intptr_t brows, intptr_t bcols, intptr_t bnnz, intptr_t rBsz,
              intptr_t cBsz, void *rowPos, void *colIdxs, void *values,
              int32_t ptp, int32_t itp, int32_t dtp, CUstream /*stream*/) {
  cusparseSpMatDescr_t mat = nullptr;
#if CUSPARSE_VERSION >= 12100
  auto pTp = static_cast<cusparseIndexType_t>(ptp);
  auto iTp = static_cast<cusparseIndexType_t>(itp);
  auto dTp = static_cast<cudaDataType_t>(dtp);
  CUSPARSE_REPORT_IF_ERROR(cusparseCreateBsr(
      &mat, brows, bcols, bnnz, rBsz, cBsz, rowPos, colIdxs, values, pTp, iTp,
      CUSPARSE_INDEX_BASE_ZERO, dTp, CUSPARSE_ORDER_ROW))
#endif
  return reinterpret_cast<void *>(mat);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroySpMat(void *m, CUstream /*stream*/) {
  cusparseSpMatDescr_t mat = reinterpret_cast<cusparseSpMatDescr_t>(m);
  CUSPARSE_REPORT_IF_ERROR(cusparseDestroySpMat(mat))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT intptr_t mgpuSpMVBufferSize(
    int32_t ma, void *a, void *x, void *y, int32_t ctp, CUstream /*stream*/) {
  assert(cusparse_env && "client did not call mgpuCreateSparseEnv()");
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseDnVecDescr_t vecX = reinterpret_cast<cusparseDnVecDescr_t>(x);
  cusparseDnVecDescr_t vecY = reinterpret_cast<cusparseDnVecDescr_t>(y);
  cudaDataType_t cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  size_t bufferSize = 0;
  CUSPARSE_REPORT_IF_ERROR(cusparseSpMV_bufferSize(
      cusparse_env, modeA, alphap, matA, vecX, betap, vecY, cTp,
      CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize))
  return bufferSize;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuSpMV(int32_t ma, void *a, void *x,
                                                   void *y, int32_t ctp,
                                                   void *buf,
                                                   CUstream /*stream*/) {
  assert(cusparse_env && "client did not call mgpuCreateSparseEnv()");
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseDnVecDescr_t vecX = reinterpret_cast<cusparseDnVecDescr_t>(x);
  cusparseDnVecDescr_t vecY = reinterpret_cast<cusparseDnVecDescr_t>(y);
  cudaDataType_t cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  CUSPARSE_REPORT_IF_ERROR(cusparseSpMV(cusparse_env, modeA, alphap, matA, vecX,
                                        betap, vecY, cTp,
                                        CUSPARSE_SPMV_ALG_DEFAULT, buf))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT intptr_t
mgpuSpMMBufferSize(int32_t ma, int32_t mb, void *a, void *b, void *c,
                   int32_t ctp, CUstream /*stream*/) {
  assert(cusparse_env && "client did not call mgpuCreateSparseEnv()");
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseDnMatDescr_t matB = reinterpret_cast<cusparseDnMatDescr_t>(b);
  cusparseDnMatDescr_t matC = reinterpret_cast<cusparseDnMatDescr_t>(c);
  cudaDataType_t cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  size_t bufferSize = 0;
  CUSPARSE_REPORT_IF_ERROR(cusparseSpMM_bufferSize(
      cusparse_env, modeA, modeB, alphap, matA, matB, betap, matC, cTp,
      CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))
  return bufferSize;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuSpMM(int32_t ma, int32_t mb,
                                                   void *a, void *b, void *c,
                                                   int32_t ctp, void *buf,
                                                   CUstream /*stream*/) {
  assert(cusparse_env && "client did not call mgpuCreateSparseEnv()");
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseDnMatDescr_t matB = reinterpret_cast<cusparseDnMatDescr_t>(b);
  cusparseDnMatDescr_t matC = reinterpret_cast<cusparseDnMatDescr_t>(c);
  cudaDataType_t cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  CUSPARSE_REPORT_IF_ERROR(cusparseSpMM(cusparse_env, modeA, modeB, alphap,
                                        matA, matB, betap, matC, cTp,
                                        CUSPARSE_SPMM_ALG_DEFAULT, buf))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT intptr_t
mgpuSDDMMBufferSize(int32_t ma, int32_t mb, void *a, void *b, void *c,
                    int32_t ctp, CUstream /*stream*/) {
  assert(cusparse_env && "client did not call mgpuCreateSparseEnv()");
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseDnMatDescr_t matA = reinterpret_cast<cusparseDnMatDescr_t>(a);
  cusparseDnMatDescr_t matB = reinterpret_cast<cusparseDnMatDescr_t>(b);
  cusparseSpMatDescr_t matC = reinterpret_cast<cusparseSpMatDescr_t>(c);
  auto cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  size_t bufferSize = 0;
  CUSPARSE_REPORT_IF_ERROR(cusparseSDDMM_bufferSize(
      cusparse_env, modeA, modeB, alphap, matA, matB, betap, matC, cTp,
      CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize))
  return bufferSize;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuSDDMM(int32_t ma, int32_t mb,
                                                    void *a, void *b, void *c,
                                                    int32_t ctp, void *buf,
                                                    CUstream /*stream*/) {
  assert(cusparse_env && "client did not call mgpuCreateSparseEnv()");
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseDnMatDescr_t matA = reinterpret_cast<cusparseDnMatDescr_t>(a);
  cusparseDnMatDescr_t matB = reinterpret_cast<cusparseDnMatDescr_t>(b);
  cusparseSpMatDescr_t matC = reinterpret_cast<cusparseSpMatDescr_t>(c);
  auto cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  CUSPARSE_REPORT_IF_ERROR(cusparseSDDMM(cusparse_env, modeA, modeB, alphap,
                                         matA, matB, betap, matC, cTp,
                                         CUSPARSE_SDDMM_ALG_DEFAULT, buf))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuSpGEMMCreateDescr(CUstream /*stream*/) {
  cusparseSpGEMMDescr_t spgemmDesc = nullptr;
  CUSPARSE_REPORT_IF_ERROR(cusparseSpGEMM_createDescr(&spgemmDesc))
  return reinterpret_cast<void *>(spgemmDesc);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuSpGEMMDestroyDescr(void *s, CUstream /*stream*/) {
  cusparseSpGEMMDescr_t spgemmDesc = reinterpret_cast<cusparseSpGEMMDescr_t>(s);
  CUSPARSE_REPORT_IF_ERROR(cusparseSpGEMM_destroyDescr(spgemmDesc))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT intptr_t mgpuSpGEMMWorkEstimation(
    void *s, int32_t ma, int32_t mb, void *a, void *b, void *c, int32_t ctp,
    intptr_t bs, void *buf, CUstream /*stream*/) {
  cusparseSpGEMMDescr_t spgemmDesc = reinterpret_cast<cusparseSpGEMMDescr_t>(s);
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseSpMatDescr_t matB = reinterpret_cast<cusparseSpMatDescr_t>(b);
  cusparseSpMatDescr_t matC = reinterpret_cast<cusparseSpMatDescr_t>(c);
  auto cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  size_t newBufferSize = bs;
  CUSPARSE_REPORT_IF_ERROR(cusparseSpGEMM_workEstimation(
      cusparse_env, modeA, modeB, alphap, matA, matB, betap, matC, cTp,
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &newBufferSize, buf))
  return newBufferSize;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT intptr_t
mgpuSpGEMMCompute(void *s, int32_t ma, int32_t mb, void *a, void *b, void *c,
                  int32_t ctp, intptr_t bsz2, void *buf2, CUstream /*stream*/) {
  cusparseSpGEMMDescr_t spgemmDesc = reinterpret_cast<cusparseSpGEMMDescr_t>(s);
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseSpMatDescr_t matB = reinterpret_cast<cusparseSpMatDescr_t>(b);
  cusparseSpMatDescr_t matC = reinterpret_cast<cusparseSpMatDescr_t>(c);
  auto cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  size_t newBufferSize2 = bsz2;
  CUSPARSE_REPORT_IF_ERROR(cusparseSpGEMM_compute(
      cusparse_env, modeA, modeB, alphap, matA, matB, betap, matC, cTp,
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &newBufferSize2, buf2))
  return newBufferSize2;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuSpGEMMCopy(void *s, int32_t ma, int32_t mb, void *a, void *b, void *c,
               int32_t ctp, CUstream /*stream*/) {
  cusparseSpGEMMDescr_t spgemmDesc = reinterpret_cast<cusparseSpGEMMDescr_t>(s);
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseSpMatDescr_t matB = reinterpret_cast<cusparseSpMatDescr_t>(b);
  cusparseSpMatDescr_t matC = reinterpret_cast<cusparseSpMatDescr_t>(c);
  auto cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  CUSPARSE_REPORT_IF_ERROR(
      cusparseSpGEMM_copy(cusparse_env, modeA, modeB, alphap, matA, matB, betap,
                          matC, cTp, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuSpMatGetSize(void *m, void *r, void *c, void *n, CUstream /*stream*/) {
  cusparseConstSpMatDescr_t matDescr =
      reinterpret_cast<cusparseConstSpMatDescr_t>(m);
  int64_t *rows = reinterpret_cast<int64_t *>(r);
  int64_t *cols = reinterpret_cast<int64_t *>(c);
  int64_t *nnz = reinterpret_cast<int64_t *>(n);
  CUSPARSE_REPORT_IF_ERROR(cusparseSpMatGetSize(matDescr, rows, cols, nnz));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuSetCsrPointers(void *m, void *p, void *c, void *v, CUstream /*stream*/) {
  cusparseSpMatDescr_t matDescr = reinterpret_cast<cusparseSpMatDescr_t>(m);
  CUSPARSE_REPORT_IF_ERROR(cusparseCsrSetPointers(matDescr, p, c, v));
}

#ifdef MLIR_ENABLE_CUDA_CUSPARSELT

///
/// Wrapper methods for the cuSparseLt library.
///

struct cusparseLtSpMatHandleAndData {
  cusparseLtMatDescriptor_t mat;
  // TODO: the following three are associated with the SpMM operator rather than
  // the sparse matrix. Create workspace buffers and pass them to the SpMM
  // execution.
  cusparseLtMatmulAlgSelection_t alg_sel;
  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulDescriptor_t matmul;
  void *values{nullptr};
};

struct cusparseLtDnMatHandleAndData {
  cusparseLtMatDescriptor_t mat;
  void *values{nullptr};
};

static_assert(sizeof(cusparseLtHandle_t) == 11024,
              "Unexpected cusparseLt handle size");
static_assert(sizeof(cusparseLtSpMatHandleAndData) == 44104,
              "Unexpected cusparseLt sparse matrix handle size");
static_assert(sizeof(cusparseLtDnMatHandleAndData) == 11032,
              "Unexpected cusparseLt dense matrix handle size");

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuCreateSparseLtEnv() {
  // ScopedContext is for cuda initialization.
  ScopedContext scopedContext;
  assert(!cusparseLt_initiated &&
         "client called mgpuCreateSparseLtEnv() twice");
  // Note that cuSparseLt still uses cusparseStatus_t.
  CUSPARSE_REPORT_IF_ERROR(cusparseLtInit(&cusparseLt_env));
  cusparseLt_initiated = true;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuDestroySparseLtEnv() {
  assert(cusparseLt_initiated && "client did not call mgpuCreateSparseLtEnv()");
  CUSPARSE_REPORT_IF_ERROR(cusparseLtDestroy(&cusparseLt_env));
  cusparseLt_initiated = false;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuCreateCuSparseLtDnMat(void *dh, intptr_t rows, intptr_t cols, void *values,
                          int32_t dtp, CUstream /*stream*/) {
  assert(cusparseLt_initiated && "client did not call mgpuCreateSparseLtEnv()");
  auto dnmat_handle = reinterpret_cast<cusparseLtDnMatHandleAndData *>(dh);
  dnmat_handle->values = values;
  auto dTp = static_cast<cudaDataType_t>(dtp);
  // Assume row-major when deciding lda.
  const uint32_t alignment = 16;
  CUSPARSE_REPORT_IF_ERROR(cusparseLtDenseDescriptorInit(
      &cusparseLt_env, &(dnmat_handle->mat), rows, cols, /*lda=*/cols,
      alignment, dTp, CUSPARSE_ORDER_ROW))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroyCuSparseLtDnMat(void *dh, CUstream /*stream*/) {
  auto dnmat_handle = reinterpret_cast<cusparseLtDnMatHandleAndData *>(dh);
  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatDescriptorDestroy(&(dnmat_handle->mat)))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuCusparseLtCreate2To4SpMat(void *sh, intptr_t rows, intptr_t cols,
                              void *values, int32_t dtp, CUstream /*stream*/) {
  assert(cusparseLt_initiated && "client did not call mgpuCreateSparseLtEnv()");
  auto spmat_handle = reinterpret_cast<cusparseLtSpMatHandleAndData *>(sh);
  spmat_handle->values = values;
  auto dTp = static_cast<cudaDataType_t>(dtp);
  // Assume row-major when deciding lda.
  const uint32_t alignment = 16;
  CUSPARSE_REPORT_IF_ERROR(cusparseLtStructuredDescriptorInit(
      &cusparseLt_env, &(spmat_handle->mat), rows, cols, /*ld=*/cols, alignment,
      dTp, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroyCuSparseLtSpMat(void *sh, CUstream /*stream*/) {
  auto spmat_handle = reinterpret_cast<cusparseLtSpMatHandleAndData *>(sh);
  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatDescriptorDestroy(&(spmat_handle->mat)))
}

// Several things are being done in this stage, algorithm selection, planning,
// and returning workspace and compressed matrices data buffer sizes.
// The parameter prune_flag is used to indicate whether pruning and pruning
// check will happen 0 means not prune or prune check, 1 means prune, 2 means
// prune & prune check
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuCuSparseLtSpMMBufferSize(void *bs, int32_t ma, int32_t mb, void *a, void *b,
                             void *c, int32_t ctp, int32_t prune_flag,
                             CUstream stream) {
  assert(cusparseLt_initiated && "client did not call mgpuCreateSparseLtEnv()");
  // TODO: support more advanced settings, e.g., the input right operand is a
  // sparse matrix assuming matA is the sparse matrix
  auto matA = reinterpret_cast<cusparseLtSpMatHandleAndData *>(a);
  auto matB = reinterpret_cast<cusparseLtDnMatHandleAndData *>(b);
  auto matC = reinterpret_cast<cusparseLtDnMatHandleAndData *>(c);
  auto workspace_size = reinterpret_cast<size_t *>(bs);
  auto compressed_size = &(reinterpret_cast<size_t *>(bs)[1]);
  auto compressed_buffer_size = &(reinterpret_cast<size_t *>(bs)[2]);
  auto cTp = static_cast<cusparseComputeType>(ctp);

  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatmulDescriptorInit(
      &cusparseLt_env, &(matA->matmul), modeA, modeB, &(matA->mat),
      &(matB->mat), &(matC->mat), &(matC->mat), cTp))
  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatmulAlgSelectionInit(
      &cusparseLt_env, &(matA->alg_sel), &(matA->matmul),
      CUSPARSELT_MATMUL_ALG_DEFAULT))
  int alg = 0;
  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatmulAlgSetAttribute(
      &cusparseLt_env, &(matA->alg_sel), CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg,
      sizeof(alg)))

  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatmulPlanInit(
      &cusparseLt_env, &(matA->plan), &(matA->matmul), &(matA->alg_sel)))

  // Pruning step (in-place).
  if (prune_flag > 0)
    CUSPARSE_REPORT_IF_ERROR(cusparseLtSpMMAPrune(
        &cusparseLt_env, &(matA->matmul), matA->values, matA->values,
        CUSPARSELT_PRUNE_SPMMA_STRIP, stream))

  // Check structure of A.
  // Note that this adds a synchronization on the stream.
  // TODO: Do we want that?
  if (prune_flag == 2) {
    int *dvalid = (int *)mgpuMemAlloc(sizeof(int), stream, false);
    CUSPARSE_REPORT_IF_ERROR(cusparseLtSpMMAPruneCheck(
        &cusparseLt_env, &(matA->matmul), matA->values, dvalid, stream))
    int valid = 0;
    mgpuMemcpy(&valid, dvalid, sizeof(int), stream);
    mgpuStreamSynchronize(stream);
    mgpuMemFree(dvalid, stream);
    if (valid != 0)
      fprintf(stderr, "CUPARSE-LT: sparse matrix is not 2:4; computed results "
                      "will be invalid\n");
  }

  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatmulGetWorkspace(
      &cusparseLt_env, &(matA->plan), workspace_size))
  CUSPARSE_REPORT_IF_ERROR(cusparseLtSpMMACompressedSize(
      &cusparseLt_env, &(matA->plan), compressed_size, compressed_buffer_size))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuCuSparseLtSpMM(void *a, void *b, void *c, void *d_workspace,
                   void *dA_compressed, void *dA_compressedBuffer,
                   CUstream stream) {
  assert(cusparseLt_initiated && "client did not call mgpuCreateSparseLtEnv()");
  auto matA = reinterpret_cast<cusparseLtSpMatHandleAndData *>(a);
  auto matB = reinterpret_cast<cusparseLtDnMatHandleAndData *>(b);
  auto matC = reinterpret_cast<cusparseLtDnMatHandleAndData *>(c);

  ALPHABETA(CUDA_R_32F, alpha, beta)
  CUSPARSE_REPORT_IF_ERROR(
      cusparseLtSpMMACompress(&cusparseLt_env, &(matA->plan), (matA->values),
                              dA_compressed, dA_compressedBuffer, stream))

  // TODO: add support to multi-stream execution
  // Perform the matrix multiplication. D = A*B+C using C==D for now
  CUSPARSE_REPORT_IF_ERROR(
      cusparseLtMatmul(&cusparseLt_env, &(matA->plan), alphap, dA_compressed,
                       matB->values, betap, matC->values,
                       /*dD*/ matC->values, d_workspace, nullptr, 0))

  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatDescriptorDestroy(&(matA->mat)))
  // destroy the plan associated with the sparse matrix
  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatmulPlanDestroy(&(matA->plan)))
}

#endif // MLIR_ENABLE_CUDA_CUSPARSELT
#endif // MLIR_ENABLE_CUDA_CUSPARSE
