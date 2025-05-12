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

#include <cudnn.h>
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
  
  // 检查当前上下文
  CUcontext current = nullptr;
  CUDA_REPORT_IF_ERROR(cuCtxGetCurrent(&current));
  
  // 如果没有当前上下文或与全局上下文不同，则设置全局上下文
  if (current == nullptr) {
    if (g_global_context == nullptr) {
      // 创建新的上下文
      CUDA_REPORT_IF_ERROR(cuCtxCreate(&g_global_context, 0, 0));
      fprintf(stderr, "[CONTEXT] Created new global context: %p\n", g_global_context);
    }
    // 设置为当前上下文
    CUDA_REPORT_IF_ERROR(cuCtxSetCurrent(g_global_context));
    fprintf(stderr, "[CONTEXT] Set global context as current: %p\n", g_global_context);
  } else if (g_global_context == nullptr) {
    // 如果有当前上下文但全局上下文为空，则使用当前上下文作为全局上下文
    g_global_context = current;
    fprintf(stderr, "[CONTEXT] Adopted current context as global: %p\n", g_global_context);
  } else if (current != g_global_context) {
    // 如果当前上下文与全局上下文不同，则设置全局上下文为当前上下文
    CUDA_REPORT_IF_ERROR(cuCtxSetCurrent(g_global_context));
    fprintf(stderr, "[CONTEXT] Switched from context %p to global context: %p\n", 
            current, g_global_context);
  }

  // 最后打印当前上下文
  CUcontext final_context = nullptr;
  cuCtxGetCurrent(&final_context);
  fprintf(stderr, "[CONTEXT-FINAL] Current context: %p\n", final_context);
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
    fprintf(stderr, "[CONTEXT] Destroyed global context\n");
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

// 流到cuDNN句柄的映射
static std::mutex g_handles_mutex;
static std::unordered_map<CUstream, cudnnHandle_t> g_stream_handles;

// 为流获取或创建cuDNN句柄
extern "C" MLIR_CUDA_WRAPPERS_EXPORT cudnnHandle_t mgpuCudnnGetHandle(CUstream stream) {
  
  // 首先确保我们在正确的上下文中
  mgpuEnsureContext();
  
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
  fprintf(stderr, "[HANDLE] Created new cuDNN handle: %p for stream: %p\n", 
    handle, stream);

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
    fprintf(stderr, "[HANDLE] Destroyed cuDNN handle for stream: %p\n", stream);
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

  fprintf(stderr, "Input data: %p, weight data: %p, bias data: %p, output data: %p\n", 
    x_data, w_data, bias_data, y_data);
  // 验证每个指针是否有效
  if (x_data == nullptr || w_data == nullptr || y_data == nullptr) {
  fprintf(stderr, "Error: Invalid pointer detected in mgpuCudnnConv2dForward\n");
  return;
  }

  fprintf(stderr, "[START] mgpuCudnnConv2dForward\n");
  
  // ScopedContext scopedContext;

  // 确保使用全局上下文
  mgpuEnsureContext();

  // 获取此流的cuDNN句柄
  fprintf(stderr, "[HANDLE] Before getting handle\n");
  cudnnHandle_t handle = mgpuCudnnGetHandle(stream);
  fprintf(stderr, "[HANDLE] Got handle: %p\n", handle);
  
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
  
  // 获取输出尺寸
  int out_n, out_c, out_h, out_w;
  CUDNN_REPORT_IF_ERROR(cudnnGetConvolution2dForwardOutputDim(
      convDesc, xDesc, wDesc, &out_n, &out_c, &out_h, &out_w));

  fprintf(stderr, "Output dimensions: n=%d, c=%d, h=%d, w=%d\n", 
        out_n, out_c, out_h, out_w);
  
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
  

  // 在cudnnConvolutionForward之前打印关键参数的值
  fprintf(stderr, "[PRE-CONV] Input ptr: %p, Filter ptr: %p, Output ptr: %p\n", 
    x_data, w_data, y_data);

  // 记录使用的CUDA流和cuDNN句柄
  fprintf(stderr, "[PRE-CONV] Using stream: %p, handle: %p\n", stream, handle);

  // 执行卷积操作
  fprintf(stderr, "[EXECUTING] cudnnConvolutionForward...\n");
  cudnnStatus_t status = cudnnConvolutionForward(
  handle, &alpha, xDesc, x_data, wDesc, w_data, convDesc, algo,
  workspace, workspaceSize, &beta, yDesc, y_data);

  // 打印卷积执行的结果状态
  fprintf(stderr, "[POST-CONV] Status: %d (%s)\n", status, 
    cudnnGetErrorString(status));

  // 在cudnnConvolutionForward之后再次打印参数，检查是否有变化
  fprintf(stderr, "[POST-CONV] Input ptr: %p, Filter ptr: %p, Output ptr: %p\n", 
    x_data, w_data, y_data);

  // 检查工作空间的状态
  fprintf(stderr, "[POST-CONV] Workspace ptr: %p, size: %zu\n", workspace, workspaceSize);

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

  fprintf(stderr, "[END] mgpuCudnnConv2dForward\n");
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
  ScopedContext scopedContext;
  
  // 获取此流的cuDNN句柄
  cudnnHandle_t handle = mgpuCudnnGetHandle(stream);
  
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
  ScopedContext scopedContext;
  
  // 获取此流的cuDNN句柄
  cudnnHandle_t handle = mgpuCudnnGetHandle(stream);
  
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
  ScopedContext scopedContext;
  
  // 获取此流的cuDNN句柄
  cudnnHandle_t handle = mgpuCudnnGetHandle(stream);
  
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
  ScopedContext scopedContext;
  
  // 获取此流的cuDNN句柄
  cudnnHandle_t handle = mgpuCudnnGetHandle(stream);
  
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


// 全局缓存变量
static CUmodule cachedModule = nullptr;
static void* cachedModuleData = nullptr;
static bool moduleMarkedForUnload = false;

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUmodule
mgpuModuleLoad(void *data, size_t /*gpuBlobSize*/) {
  ScopedContext scopedContext;
  
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
  ScopedContext scopedContext;
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
  
  // 先销毁与此流关联的cuDNN句柄
  mgpuCudnnDestroyHandle(stream);

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
  ScopedContext scopedContext;
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
  ScopedContext scopedContext;
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
