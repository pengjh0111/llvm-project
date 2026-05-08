//===- CudnnGraphBuilder.cpp - cuDNN Graph dynamic builder implementation --===//
//
// Improved implementation with clear separation of construction and execution
//
//===----------------------------------------------------------------------===//

#include "CudnnGraphBuilder.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <unordered_set>

// 错误检查宏
#define CUDNN_CHECK_WITH_INFO(call, info) \
  do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
      fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudnnGetErrorString(status)); \
      fprintf(stderr, "Context: %s\n", info); \
      return false; \
    } \
  } while(0)

#define CUDNN_CHECK_RETURN_WITH_INFO(call, ret_val, info) \
  do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
      fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudnnGetErrorString(status)); \
      fprintf(stderr, "Context: %s\n", info); \
      return ret_val; \
    } \
  } while(0)

#define CUDNN_CHECK(call) \
  do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
      fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudnnGetErrorString(status)); \
      return false; \
    } \
  } while(0)

#define CUDNN_CHECK_RETURN(call, ret_val) \
  do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
      fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudnnGetErrorString(status)); \
      return ret_val; \
    } \
  } while(0)

// ===== CudnnGraphBuilder Implementation =====

CudnnGraphBuilder::CudnnGraphBuilder(cudnnHandle_t handle) 
    : handle_(handle), graph_desc_(nullptr), engine_desc_(nullptr), 
      engine_config_desc_(nullptr), exec_plan_desc_(nullptr), next_tensor_id_(0), next_node_id_(0),
      is_compiled_(false), is_finalized_(false), workspace_(nullptr), workspace_size_(0) {
    
    // 检查cuDNN版本
    size_t version = cudnnGetVersion();
    fprintf(stderr, "cuDNN Version: %zu\n", version);
    if (version < 8000) {
        fprintf(stderr, "WARNING: cuDNN Backend API requires version 8.0+, current: %zu\n", version);
    }
    
    // 创建操作图描述符
    cudnnStatus_t status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &graph_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to create graph descriptor: %s\n", cudnnGetErrorString(status));
        graph_desc_ = nullptr;
    } else {
        fprintf(stderr, "Graph descriptor created successfully\n");
    }
}

CudnnGraphBuilder::~CudnnGraphBuilder() {
    Reset();
    
    if (graph_desc_) {
        cudnnBackendDestroyDescriptor(graph_desc_);
    }
    if (engine_desc_) {
        cudnnBackendDestroyDescriptor(engine_desc_);
    }
    if (engine_config_desc_) {
        cudnnBackendDestroyDescriptor(engine_config_desc_);
    }
    if (exec_plan_desc_) {
        cudnnBackendDestroyDescriptor(exec_plan_desc_);
    }
    if (workspace_) {
        cudaFree(workspace_);
    }
}

void CudnnGraphBuilder::Reset() {
    // 清理所有张量描述符
    for (auto& tensor : tensors_) {
        if (tensor->desc) {
            cudnnBackendDestroyDescriptor(tensor->desc);  // 修复：使用backend API
        }
    }
    tensors_.clear();
    
    // 清理所有操作描述符
    for (auto& node : nodes_) {
        if (node->op_desc) {
            cudnnBackendDestroyDescriptor(node->op_desc);
        }
    }
    nodes_.clear();
    
    ptr_to_tensor_id_.clear();
    input_tensor_ids_.clear();
    output_tensor_ids_.clear();
    
    next_tensor_id_ = 0;
    next_node_id_ = 0;
    is_compiled_ = false;
    is_finalized_ = false;
}

// 修复：阶段1: 构建阶段 - 只定义张量结构，不需要实际数据
// int64_t CudnnGraphBuilder::AddTensor(const std::vector<int>& dims, 
//                                     cudnnDataType_t data_type, 
//                                     bool is_virtual) {
    
//     fprintf(stderr, "Creating tensor with dims: [");
//     for (size_t i = 0; i < dims.size(); i++) {
//         fprintf(stderr, "%d", dims[i]);
//         if (i < dims.size() - 1) fprintf(stderr, ", ");
//     }
//     fprintf(stderr, "], type: %d, virtual: %s\n", data_type, is_virtual ? "true" : "false");
    
//     // 验证维度合理性
//     if (dims.empty() || dims.size() > 8) {
//         fprintf(stderr, "ERROR: Invalid tensor dimensions (size: %zu)\n", dims.size());
//         return -1;
//     }
    
//     for (int dim : dims) {
//         if (dim <= 0) {
//             fprintf(stderr, "ERROR: Invalid dimension value: %d\n", dim);
//             return -1;
//         }
//     }
    
//     auto tensor = std::make_unique<GraphTensor>();
//     tensor->tensor_id = next_tensor_id_++;
//     tensor->dimensions = dims;
//     tensor->data_type = data_type;
//     tensor->data_ptr = nullptr;
    
//     // 创建张量描述符
//     CUDNN_CHECK_RETURN_WITH_INFO(
//         cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensor->desc), 
//         -1, "Creating tensor descriptor");
    
//     // 转换维度到int64_t
//     std::vector<int64_t> dim64(dims.begin(), dims.end());
    
//     // 计算strides (NCHW格式)
//     std::vector<int64_t> strides(dims.size());
//     strides[dims.size()-1] = 1;
//     for (int i = dims.size()-2; i >= 0; i--) {
//         strides[i] = strides[i+1] * dim64[i+1];
//     }
    
//     fprintf(stderr, "Tensor %ld strides: [", tensor->tensor_id);
//     for (size_t i = 0; i < strides.size(); i++) {
//         fprintf(stderr, "%ld", strides[i]);
//         if (i < strides.size() - 1) fprintf(stderr, ", ");
//     }
//     fprintf(stderr, "]\n");
    
//     // 设置张量属性 - 按推荐顺序
//     CUDNN_CHECK_RETURN_WITH_INFO(
//         cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_DATA_TYPE, 
//                                CUDNN_TYPE_DATA_TYPE, 1, &data_type), 
//         -1, "Setting tensor data type");
    
//     CUDNN_CHECK_RETURN_WITH_INFO(
//         cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_DIMENSIONS, 
//                                CUDNN_TYPE_INT64, dims.size(), dim64.data()), 
//         -1, "Setting tensor dimensions");
    
//     CUDNN_CHECK_RETURN_WITH_INFO(
//         cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_STRIDES, 
//                                CUDNN_TYPE_INT64, strides.size(), strides.data()), 
//         -1, "Setting tensor strides");
    
//     CUDNN_CHECK_RETURN_WITH_INFO(
//         cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, 
//                                CUDNN_TYPE_INT64, 1, &tensor->tensor_id), 
//         -1, "Setting tensor unique ID");
    
//     // // 设置虚拟标志（如果需要）
//     // if (is_virtual) {
//     //     int64_t virtual_flag = 1;
//     //     CUDNN_CHECK_RETURN_WITH_INFO(
//     //         cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_IS_VIRTUAL, 
//     //                                CUDNN_TYPE_BOOLEAN, 1, &virtual_flag), 
//     //         -1, "Setting tensor virtual flag");
//     // }
    
//     // Finalize张量描述符
//     CUDNN_CHECK_RETURN_WITH_INFO(
//         cudnnBackendFinalize(tensor->desc), 
//         -1, "Finalizing tensor descriptor");
    
//     int64_t tensor_id = tensor->tensor_id;
//     tensors_.push_back(std::move(tensor));
    
//     fprintf(stderr, "Successfully created tensor %ld\n", tensor_id);
//     return tensor_id;
// }

int64_t CudnnGraphBuilder::AddTensor(const std::vector<int>& dims, 
                                    cudnnDataType_t data_type, 
                                    bool is_virtual) {
    
    fprintf(stderr, "Creating tensor (safe cuDNN 8.9) with dims: [");
    for (size_t i = 0; i < dims.size(); i++) {
        fprintf(stderr, "%d", dims[i]);
        if (i < dims.size() - 1) fprintf(stderr, ", ");
    }
    fprintf(stderr, "], type: %d\n", data_type);
    
    // 基本验证
    if (dims.empty() || dims.size() > 8) {
        fprintf(stderr, "ERROR: Invalid tensor dimensions\n");
        return -1;
    }
    
    for (int dim : dims) {
        if (dim <= 0) {
            fprintf(stderr, "ERROR: Invalid dimension value: %d\n", dim);
            return -1;
        }
    }
    
    auto tensor = std::make_unique<GraphTensor>();
    tensor->tensor_id = next_tensor_id_++;
    tensor->dimensions = dims;
    tensor->data_type = data_type;
    tensor->data_ptr = nullptr;
    
    // 创建张量描述符
    cudnnStatus_t status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensor->desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: Failed to create descriptor: %s\n", cudnnGetErrorString(status));
        return -1;
    }
    
    // 准备数据
    std::vector<int64_t> dim64(dims.begin(), dims.end());
    std::vector<int64_t> strides(dims.size());
    
    // 计算strides (NCHW format)
    strides[dims.size()-1] = 1;
    for (int i = dims.size()-2; i >= 0; i--) {
        strides[i] = strides[i+1] * dim64[i+1];
    }
    
    fprintf(stderr, "Calculated strides: [");
    for (size_t i = 0; i < strides.size(); i++) {
        fprintf(stderr, "%ld", strides[i]);
        if (i < strides.size() - 1) fprintf(stderr, ", ");
    }
    fprintf(stderr, "]\n");
    
    // 使用确定存在的属性，按照成功率最高的顺序设置
    
    // 1. 数据类型 (基础且必需)
    // fprintf(stderr, "Step 1: Setting data type...\n");
    status = cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_DATA_TYPE, 
                                     CUDNN_TYPE_DATA_TYPE, 1, &data_type);
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: Data type failed: %s\n", cudnnGetErrorString(status));
        cudnnBackendDestroyDescriptor(tensor->desc);
        return -1;
    }
    // fprintf(stderr, "Step 1: SUCCESS\n");
    
    // 2. Unique ID (基础标识)
    // fprintf(stderr, "Step 2: Setting unique ID...\n");
    status = cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, 
                                     CUDNN_TYPE_INT64, 1, &tensor->tensor_id);
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: Unique ID failed: %s\n", cudnnGetErrorString(status));
        cudnnBackendDestroyDescriptor(tensor->desc);
        return -1;
    }
    // fprintf(stderr, "Step 2: SUCCESS\n");
    
    // 3. 尝试维度设置的不同方法
    // fprintf(stderr, "Step 3: Setting dimensions...\n");
    
    // 方法A：先设置维度数量，再设置维度数组
    int64_t nbDims = static_cast<int64_t>(dims.size());
    status = cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_DIMENSIONS, 
                                     CUDNN_TYPE_INT64, 1, &nbDims);
    if (status == CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "Dimension count set successfully\n");
        
        // 然后设置维度数组 (使用相同的属性名但传递数组)
        status = cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_DIMENSIONS, 
                                         CUDNN_TYPE_INT64, dims.size(), dim64.data());
        if (status != CUDNN_STATUS_SUCCESS) {
            fprintf(stderr, "Dimension array failed: %s\n", cudnnGetErrorString(status));
            cudnnBackendDestroyDescriptor(tensor->desc);
            return -1;
        }
        fprintf(stderr, "Dimension array set successfully\n");
    } else {
        // 方法B：直接设置维度数组
        fprintf(stderr, "Dimension count failed, trying direct array...\n");
        status = cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_DIMENSIONS, 
                                         CUDNN_TYPE_INT64, dims.size(), dim64.data());
        if (status != CUDNN_STATUS_SUCCESS) {
            fprintf(stderr, "ERROR: Direct dimension array also failed: %s\n", cudnnGetErrorString(status));
            cudnnBackendDestroyDescriptor(tensor->desc);
            return -1;
        }
        fprintf(stderr, "Direct dimension array set successfully\n");
    }
    // fprintf(stderr, "Step 3: SUCCESS\n");
    
    // 4. 设置strides
    // fprintf(stderr, "Step 4: Setting strides...\n");
    status = cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_STRIDES, 
                                     CUDNN_TYPE_INT64, strides.size(), strides.data());
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: Strides failed: %s\n", cudnnGetErrorString(status));
        cudnnBackendDestroyDescriptor(tensor->desc);
        return -1;
    }
    // fprintf(stderr, "Step 4: SUCCESS\n");
    
    // 5. 尝试设置可选属性 (如果失败不会阻止继续)
    
    // 尝试字节对齐
    // fprintf(stderr, "Step 5: Trying optional attributes...\n");
    int64_t byte_alignment = 16;
    status = cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, 
                                     CUDNN_TYPE_INT64, 1, &byte_alignment);
    if (status == CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "Byte alignment set successfully\n");
    } else {
        fprintf(stderr, "Byte alignment not supported (OK)\n");
    }
    
    // 尝试virtual标志 (如果需要)
    if (is_virtual) {
        // fprintf(stderr, "Trying virtual flag...\n");
        int64_t virtual_flag = 1;
        
        // 尝试BOOLEAN类型
        status = cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_IS_VIRTUAL, 
                                         CUDNN_TYPE_BOOLEAN, 1, &virtual_flag);
        if (status == CUDNN_STATUS_SUCCESS) {
            // fprintf(stderr, "Virtual flag (BOOLEAN) set successfully\n");
        } else {
            // 尝试INT64类型
            status = cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_IS_VIRTUAL, 
                                             CUDNN_TYPE_INT64, 1, &virtual_flag);
            if (status == CUDNN_STATUS_SUCCESS) {
                fprintf(stderr, "Virtual flag (INT64) set successfully\n");
            } else {
                fprintf(stderr, "Virtual flag not supported (continuing)\n");
            }
        }
    }
    
    // 6. Finalize
    // fprintf(stderr, "Step 6: Finalizing tensor...\n");
    status = cudnnBackendFinalize(tensor->desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: Finalize failed: %s\n", cudnnGetErrorString(status));
        
        // 如果finalize失败，可能是因为某些可选属性导致的冲突
        // 尝试重新创建一个只有基本属性的描述符
        fprintf(stderr, "Retrying with minimal attributes...\n");
        
        cudnnBackendDestroyDescriptor(tensor->desc);
        
        // 重新创建
        status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensor->desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            fprintf(stderr, "ERROR: Failed to recreate descriptor\n");
            return -1;
        }
        
        // 只设置最基本的属性
        status = cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_DATA_TYPE, 
                                         CUDNN_TYPE_DATA_TYPE, 1, &data_type);
        if (status != CUDNN_STATUS_SUCCESS) {
            fprintf(stderr, "ERROR: Retry data type failed\n");
            cudnnBackendDestroyDescriptor(tensor->desc);
            return -1;
        }
        
        status = cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, 
                                         CUDNN_TYPE_INT64, 1, &tensor->tensor_id);
        if (status != CUDNN_STATUS_SUCCESS) {
            fprintf(stderr, "ERROR: Retry unique ID failed\n");
            cudnnBackendDestroyDescriptor(tensor->desc);
            return -1;
        }
        
        status = cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_DIMENSIONS, 
                                         CUDNN_TYPE_INT64, dims.size(), dim64.data());
        if (status != CUDNN_STATUS_SUCCESS) {
            fprintf(stderr, "ERROR: Retry dimensions failed\n");
            cudnnBackendDestroyDescriptor(tensor->desc);
            return -1;
        }
        
        status = cudnnBackendSetAttribute(tensor->desc, CUDNN_ATTR_TENSOR_STRIDES, 
                                         CUDNN_TYPE_INT64, strides.size(), strides.data());
        if (status != CUDNN_STATUS_SUCCESS) {
            fprintf(stderr, "ERROR: Retry strides failed\n");
            cudnnBackendDestroyDescriptor(tensor->desc);
            return -1;
        }
        
        // 再次尝试finalize
        status = cudnnBackendFinalize(tensor->desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            fprintf(stderr, "ERROR: Even minimal finalize failed: %s\n", cudnnGetErrorString(status));
            cudnnBackendDestroyDescriptor(tensor->desc);
            return -1;
        }
        
        fprintf(stderr, "Minimal tensor finalized successfully\n");
    } else {
        fprintf(stderr, "Full tensor finalized successfully\n");
    }
    
    fprintf(stderr, "Step 6: SUCCESS\n");
    
    int64_t tensor_id = tensor->tensor_id;
    tensors_.push_back(std::move(tensor));
    
    fprintf(stderr, "Successfully created tensor %ld\n", tensor_id);
    return tensor_id;
}

int64_t CudnnGraphBuilder::GetOrCreateTensor(void* data_ptr, 
                                           const std::vector<int>& dims, 
                                           cudnnDataType_t data_type) {
    // 首先检查是否已经有对应数据指针的张量
    auto it = ptr_to_tensor_id_.find(data_ptr);
    if (it != ptr_to_tensor_id_.end()) {
        return it->second;
    }
    
    // 修复：创建新张量但暂时不设置数据指针
    int64_t tensor_id = AddTensor(dims, data_type, true);
    
    // 记录数据指针映射关系，稍后在执行阶段会用到
    ptr_to_tensor_id_[data_ptr] = tensor_id;
    
    return tensor_id;
}

void CudnnGraphBuilder::MarkTensorAsInput(int64_t tensor_id) {
    for (auto& tensor : tensors_) {
        if (tensor->tensor_id == tensor_id) {
            if (!tensor->is_input) {
                tensor->is_input = true;
                input_tensor_ids_.push_back(tensor_id);
            }
            break;
        }
    }
}

void CudnnGraphBuilder::MarkTensorAsOutput(int64_t tensor_id) {
    for (auto& tensor : tensors_) {
        if (tensor->tensor_id == tensor_id) {
            if (!tensor->is_output) {
                tensor->is_output = true;
                output_tensor_ids_.push_back(tensor_id);
            }
            break;
        }
    }
}

// 阶段3: 执行阶段 - 设置实际数据指针
void CudnnGraphBuilder::SetTensorData(int64_t tensor_id, void* data_ptr) {
    for (auto& tensor : tensors_) {
        if (tensor->tensor_id == tensor_id) {
            tensor->data_ptr = data_ptr;
            break;
        }
    }
}

void CudnnGraphBuilder::SetTensorDataByPointer(void* old_ptr, void* new_ptr) {
    auto it = ptr_to_tensor_id_.find(old_ptr);
    if (it != ptr_to_tensor_id_.end()) {
        SetTensorData(it->second, new_ptr);
        // 更新映射关系
        ptr_to_tensor_id_.erase(it);
        ptr_to_tensor_id_[new_ptr] = it->second;
    }
}

// 自动标记输入输出张量
void CudnnGraphBuilder::AutoMarkInputsOutputs() {
    std::unordered_set<int64_t> consumed_tensors;
    std::unordered_set<int64_t> produced_tensors;
    
    // 收集所有被消费和产生的张量
    for (const auto& node : nodes_) {
        for (int64_t input_id : node->input_tensor_ids) {
            consumed_tensors.insert(input_id);
        }
        for (int64_t output_id : node->output_tensor_ids) {
            produced_tensors.insert(output_id);
        }
    }
    
    // 输入张量：被消费但不被产生
    for (int64_t tensor_id : consumed_tensors) {
        if (produced_tensors.find(tensor_id) == produced_tensors.end()) {
            MarkTensorAsInput(tensor_id);
        }
    }
    
    // 输出张量：被产生但不被消费，或者明确标记的输出
    for (int64_t tensor_id : produced_tensors) {
        if (consumed_tensors.find(tensor_id) == consumed_tensors.end()) {
            MarkTensorAsOutput(tensor_id);
        }
    }
}

// 修复：操作节点添加 (构建阶段)
int64_t CudnnGraphBuilder::AddConvolutionNode(
    int64_t input_tensor_id, int64_t weight_tensor_id, int64_t bias_tensor_id,
    int64_t output_tensor_id, const std::vector<int>& pads, 
    const std::vector<int>& strides, const std::vector<int>& dilations) {
    
    auto node = std::make_unique<GraphNode>();
    node->node_id = next_node_id_++;
    node->op_type = "CONVOLUTION";
    
    // 创建卷积操作描述符
    CUDNN_CHECK_RETURN(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &node->op_desc), -1);
    
    // 获取张量描述符
    cudnnBackendDescriptor_t input_desc = nullptr, weight_desc = nullptr, output_desc = nullptr;
    
    for (auto& tensor : tensors_) {
        if (tensor->tensor_id == input_tensor_id) input_desc = tensor->desc;
        if (tensor->tensor_id == weight_tensor_id) weight_desc = tensor->desc;
        if (tensor->tensor_id == output_tensor_id) output_desc = tensor->desc;
    }
    
    // 验证张量描述符
    if (!input_desc || !weight_desc || !output_desc) {
        fprintf(stderr, "ERROR: Missing tensor descriptors for convolution\n");
        cudnnBackendDestroyDescriptor(node->op_desc);
        return -1;
    }
    
    // 创建卷积描述符
    cudnnBackendDescriptor_t conv_desc;
    CUDNN_CHECK_RETURN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &conv_desc), -1);
    
    // 设置卷积参数
    std::vector<int64_t> pad64(pads.begin(), pads.end());
    std::vector<int64_t> stride64(strides.begin(), strides.end());
    std::vector<int64_t> dilation64(dilations.begin(), dilations.end());
    
    int64_t spatial_dims = static_cast<int64_t>(pads.size());
    
    // 设置卷积属性 - 按照cuDNN 8.9的要求
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        conv_desc, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, 
        CUDNN_TYPE_INT64, 1, &spatial_dims), -1);
    
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        conv_desc, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, 
        CUDNN_TYPE_INT64, pads.size(), pad64.data()), -1);
    
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        conv_desc, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, 
        CUDNN_TYPE_INT64, pads.size(), pad64.data()), -1);
    
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        conv_desc, CUDNN_ATTR_CONVOLUTION_DILATIONS, 
        CUDNN_TYPE_INT64, dilations.size(), dilation64.data()), -1);
    
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        conv_desc, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, 
        CUDNN_TYPE_INT64, strides.size(), stride64.data()), -1);
    
    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        conv_desc, CUDNN_ATTR_CONVOLUTION_CONV_MODE, 
        CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode), -1);
    
    // 设置计算类型
    cudnnDataType_t compute_type = CUDNN_DATA_FLOAT;
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        conv_desc, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, 
        CUDNN_TYPE_DATA_TYPE, 1, &compute_type), -1);
        
    // Finalize convolution descriptor
    CUDNN_CHECK_RETURN(cudnnBackendFinalize(conv_desc), -1);
    
    // 设置操作属性 - 只处理input, weight, output，不处理bias
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        node->op_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, 
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &input_desc), -1);
    
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        node->op_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, 
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &weight_desc), -1);
    
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        node->op_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, 
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &output_desc), -1);
    
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        node->op_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, 
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv_desc), -1);
    
    // 设置alpha和beta
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        node->op_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, 
        CUDNN_TYPE_FLOAT, 1, &alpha), -1);
    
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        node->op_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, 
        CUDNN_TYPE_FLOAT, 1, &beta), -1);
    
    // Finalize the operation
    CUDNN_CHECK_RETURN(cudnnBackendFinalize(node->op_desc), -1);
    
    // 记录输入输出关系 - 只包含实际使用的张量
    node->input_tensor_ids.push_back(input_tensor_id);
    node->input_tensor_ids.push_back(weight_tensor_id);
    // 暂时不处理bias，因为需要单独的加法操作
    node->output_tensor_ids.push_back(output_tensor_id);
    
    int64_t node_id = node->node_id;
    nodes_.push_back(std::move(node));
    
    return node_id;
}

// 简化版本的操作节点 - 由于cuDNN backend API的复杂性，这里提供基础框架
int64_t CudnnGraphBuilder::AddElementwiseNode(
    int64_t input_a_id, int64_t input_b_id, int64_t output_id,
    cudnnBackendDescriptorType_t op_type) {
    
    auto node = std::make_unique<GraphNode>();
    node->node_id = next_node_id_++;
    node->op_type = "ELEMENTWISE";
    
    // 创建逐元素操作描述符
    CUDNN_CHECK_RETURN(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &node->op_desc), -1);
    
    // 获取张量描述符
    cudnnBackendDescriptor_t input_a_desc = nullptr, input_b_desc = nullptr, output_desc = nullptr;
    for (auto& tensor : tensors_) {
        if (tensor->tensor_id == input_a_id) input_a_desc = tensor->desc;
        if (tensor->tensor_id == input_b_id) input_b_desc = tensor->desc;
        if (tensor->tensor_id == output_id) output_desc = tensor->desc;
    }
    
    // 验证张量描述符
    if (!input_a_desc || !input_b_desc || !output_desc) {
        fprintf(stderr, "ERROR: Missing tensor descriptors for elementwise operation\n");
        cudnnBackendDestroyDescriptor(node->op_desc);
        return -1;
    }
    
    // 创建pointwise描述符
    cudnnBackendDescriptor_t pw_desc;
    CUDNN_CHECK_RETURN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR, &pw_desc), -1);
    
    // 设置pointwise模式
    cudnnPointwiseMode_t mode = CUDNN_POINTWISE_ADD;
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        pw_desc, CUDNN_ATTR_POINTWISE_MODE, 
        CUDNN_TYPE_POINTWISE_MODE, 1, &mode), -1);
    
    // 设置计算类型
    cudnnDataType_t compute_type = CUDNN_DATA_FLOAT;
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        pw_desc, CUDNN_ATTR_POINTWISE_MATH_PREC, 
        CUDNN_TYPE_DATA_TYPE, 1, &compute_type), -1);
    
    // Finalize pointwise descriptor
    CUDNN_CHECK_RETURN(cudnnBackendFinalize(pw_desc), -1);
    
    // 设置操作的pointwise描述符
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        node->op_desc, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, 
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &pw_desc), -1);
    
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        node->op_desc, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, 
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &input_a_desc), -1);
    
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        node->op_desc, CUDNN_ATTR_OPERATION_POINTWISE_BDESC, 
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &input_b_desc), -1);
    
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        node->op_desc, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, 
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &output_desc), -1);
    
    // 设置alpha
    float alpha = 1.0f, alpha2 = 1.0f;
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        node->op_desc, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, 
        CUDNN_TYPE_FLOAT, 1, &alpha), -1);
    
    CUDNN_CHECK_RETURN(cudnnBackendSetAttribute(
        node->op_desc, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2, 
        CUDNN_TYPE_FLOAT, 1, &alpha2), -1);
    
    // Finalize the operation
    CUDNN_CHECK_RETURN(cudnnBackendFinalize(node->op_desc), -1);
    
    node->input_tensor_ids.push_back(input_a_id);
    node->input_tensor_ids.push_back(input_b_id);
    node->output_tensor_ids.push_back(output_id);
    
    int64_t node_id = node->node_id;
    nodes_.push_back(std::move(node));
    
    return node_id;
}

// 添加缺失的函数实现
int64_t CudnnGraphBuilder::AddPoolingNode(
    int64_t input_tensor_id, int64_t output_tensor_id,
    cudnnPoolingMode_t mode, const std::vector<int>& window_dims,
    const std::vector<int>& pads, const std::vector<int>& strides) {
    
    auto node = std::make_unique<GraphNode>();
    node->node_id = next_node_id_++;
    node->op_type = "POOLING";
    
    node->input_tensor_ids.push_back(input_tensor_id);
    node->output_tensor_ids.push_back(output_tensor_id);
    
    int64_t node_id = node->node_id;
    nodes_.push_back(std::move(node));
    
    fprintf(stderr, "AddPoolingNode: Simplified implementation\n");
    return node_id;
}

int64_t CudnnGraphBuilder::AddMatmulNode(
    int64_t input_a_id, int64_t input_b_id, int64_t output_id) {
    
    auto node = std::make_unique<GraphNode>();
    node->node_id = next_node_id_++;
    node->op_type = "MATMUL";
    
    node->input_tensor_ids.push_back(input_a_id);
    node->input_tensor_ids.push_back(input_b_id);
    node->output_tensor_ids.push_back(output_id);
    
    int64_t node_id = node->node_id;
    nodes_.push_back(std::move(node));
    
    fprintf(stderr, "AddMatmulNode: Simplified implementation\n");
    return node_id;
}

// 修复：阶段2: 编译阶段
// bool CudnnGraphBuilder::FinalizeGraph() {
//     if (is_finalized_) return true;
    
//     // 自动标记输入输出张量（如果没有手动标记）
//     if (input_tensor_ids_.empty() && output_tensor_ids_.empty()) {
//         AutoMarkInputsOutputs();
//     }
    
//     // 收集所有操作描述符
//     std::vector<cudnnBackendDescriptor_t> ops;
//     for (auto& node : nodes_) {
//         ops.push_back(node->op_desc);
//     }
    
//     // 修复：设置图的操作 - 使用正确的属性名称
//     CUDNN_CHECK(cudnnBackendSetAttribute(
//         graph_desc_, CUDNN_ATTR_OPERATIONGRAPH_OPS, 
//         CUDNN_TYPE_BACKEND_DESCRIPTOR, ops.size(), ops.data()));
    
//     // Finalize graph
//     CUDNN_CHECK(cudnnBackendFinalize(graph_desc_));
    
//     is_finalized_ = true;
//     return true;
// }

bool CudnnGraphBuilder::FinalizeGraph() {
    if (is_finalized_) return true;
    
    fprintf(stderr, "[DEBUG] Starting FinalizeGraph...\n");
    
    // 自动标记输入输出张量
    if (input_tensor_ids_.empty() && output_tensor_ids_.empty()) {
        fprintf(stderr, "[DEBUG] Auto-marking inputs and outputs...\n");
        AutoMarkInputsOutputs();
    }
    
    fprintf(stderr, "[DEBUG] Input tensors: %zu, Output tensors: %zu\n", 
            input_tensor_ids_.size(), output_tensor_ids_.size());
    fprintf(stderr, "[DEBUG] Total operations: %zu\n", nodes_.size());
    
    // 验证至少有一些操作
    if (nodes_.empty()) {
        fprintf(stderr, "[ERROR] No operations in graph\n");
        return false;
    }
    
    // 打印所有操作的详细信息
    for (size_t i = 0; i < nodes_.size(); i++) {
        fprintf(stderr, "[DEBUG] Operation %zu: type=%s, desc=%p\n", 
                i, nodes_[i]->op_type.c_str(), nodes_[i]->op_desc);
    }
    
    // 验证所有操作都有有效的描述符
    std::vector<cudnnBackendDescriptor_t> ops;
    for (auto& node : nodes_) {
        if (node->op_desc) {
            ops.push_back(node->op_desc);
        } else {
            fprintf(stderr, "[ERROR] Found null operation descriptor\n");
            return false;
        }
    }
    
    fprintf(stderr, "[DEBUG] Setting %zu operations to graph...\n", ops.size());
    
    // 只设置操作数组 - 这是必需的核心属性
    cudnnStatus_t status = cudnnBackendSetAttribute(
        graph_desc_, CUDNN_ATTR_OPERATIONGRAPH_OPS, 
        CUDNN_TYPE_BACKEND_DESCRIPTOR, ops.size(), ops.data());
    
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "[ERROR] Setting operations failed: %s\n", 
                cudnnGetErrorString(status));
        return false;
    }
    fprintf(stderr, "[DEBUG] Operations set successfully\n");
    
    // Finalize graph
    fprintf(stderr, "[DEBUG] Finalizing operation graph...\n");
    status = cudnnBackendFinalize(graph_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "[ERROR] Graph finalize failed: %s\n", cudnnGetErrorString(status));
        
        // 提供更详细的错误信息
        if (status == CUDNN_STATUS_BAD_PARAM) {
            fprintf(stderr, "[ERROR] BAD_PARAM likely means:\n");
            fprintf(stderr, "  - Invalid operation descriptor\n");
            fprintf(stderr, "  - Tensor dimension mismatch\n");
            fprintf(stderr, "  - Missing required attributes\n");
            fprintf(stderr, "  - Incompatible data types\n");
        }
        return false;
    }
    
    fprintf(stderr, "[DEBUG] Graph finalized successfully\n");
    is_finalized_ = true;
    return true;
}

bool CudnnGraphBuilder::CompileGraph() {
    if (!is_finalized_) {
        if (!FinalizeGraph()) return false;
    }
    
    if (is_compiled_) return true;
    
    // 创建引擎描述符
    CUDNN_CHECK(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine_desc_));
    
    // 修复：设置引擎属性 - 使用正确的属性名称
    CUDNN_CHECK(cudnnBackendSetAttribute(
        engine_desc_, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, 
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph_desc_));
    
    // Finalize engine
    CUDNN_CHECK(cudnnBackendFinalize(engine_desc_));
    
    // 创建引擎配置
    CUDNN_CHECK(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engine_config_desc_));
    
    CUDNN_CHECK(cudnnBackendSetAttribute(
        engine_config_desc_, CUDNN_ATTR_ENGINECFG_ENGINE, 
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine_desc_));
    
    // Finalize engine config
    CUDNN_CHECK(cudnnBackendFinalize(engine_config_desc_));
    
    // 创建execution plan来获取workspace大小
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &exec_plan_desc_));
    
    CUDNN_CHECK(cudnnBackendSetAttribute(
        exec_plan_desc_, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine_config_desc_));
    
    CUDNN_CHECK(cudnnBackendSetAttribute(
        exec_plan_desc_, CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
        CUDNN_TYPE_HANDLE, 1, &handle_));
    
    CUDNN_CHECK(cudnnBackendFinalize(exec_plan_desc_));
    
    // 修复：获取workspace大小 - 使用execution plan
    int64_t workspace_size_int64 = 0;
    CUDNN_CHECK(cudnnBackendGetAttribute(
        exec_plan_desc_, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
        CUDNN_TYPE_INT64, 1, nullptr, &workspace_size_int64));
    
    workspace_size_ = static_cast<size_t>(workspace_size_int64);
    
    // 分配workspace
    if (workspace_size_ > 0) {
        cudaError_t cuda_status = cudaMalloc(&workspace_, workspace_size_);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Failed to allocate workspace: %s\n", 
                    cudaGetErrorString(cuda_status));
            return false;
        }
    }
    
    is_compiled_ = true;
    return true;
}

// 阶段3: 执行阶段 - 现在才需要stream
bool CudnnGraphBuilder::ExecuteGraph(CUstream stream) {
    if (!is_compiled_) {
        if (!CompileGraph()) return false;
    }
    
    // 验证所有输入输出张量都有数据指针
    for (int64_t tensor_id : input_tensor_ids_) {
        bool found = false;
        for (auto& tensor : tensors_) {
            if (tensor->tensor_id == tensor_id) {
                if (!tensor->data_ptr) {
                    fprintf(stderr, "Input tensor %ld has no data pointer set\n", tensor_id);
                    return false;
                }
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "Input tensor %ld not found\n", tensor_id);
            return false;
        }
    }
    
    for (int64_t tensor_id : output_tensor_ids_) {
        bool found = false;
        for (auto& tensor : tensors_) {
            if (tensor->tensor_id == tensor_id) {
                if (!tensor->data_ptr) {
                    fprintf(stderr, "Output tensor %ld has no data pointer set\n", tensor_id);
                    return false;
                }
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "Output tensor %ld not found\n", tensor_id);
            return false;
        }
    }
    
    // 准备变量包
    std::vector<void*> data_ptrs;
    std::vector<int64_t> uids;
    
    // 添加所有张量的数据指针
    for (auto& tensor : tensors_) {
        if (tensor->data_ptr) {
            data_ptrs.push_back(tensor->data_ptr);
            uids.push_back(tensor->tensor_id);
        }
    }
    
    // 创建变量包描述符
    cudnnBackendDescriptor_t varpack_desc;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &varpack_desc));
    
    // 修复：设置变量包属性 - 使用正确的属性名称
    CUDNN_CHECK(cudnnBackendSetAttribute(
        varpack_desc, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
        CUDNN_TYPE_VOID_PTR, data_ptrs.size(), data_ptrs.data()));
    
    CUDNN_CHECK(cudnnBackendSetAttribute(
        varpack_desc, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
        CUDNN_TYPE_INT64, uids.size(), uids.data()));
    
    CUDNN_CHECK(cudnnBackendSetAttribute(
        varpack_desc, CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
        CUDNN_TYPE_VOID_PTR, 1, &workspace_));
    
    CUDNN_CHECK(cudnnBackendFinalize(varpack_desc));
    
    // 执行图 - 使用execution plan
    CUDNN_CHECK(cudnnBackendExecute(handle_, exec_plan_desc_, varpack_desc));
    
    // 清理变量包描述符
    cudnnBackendDestroyDescriptor(varpack_desc);
    
    return true;
}

void CudnnGraphBuilder::PrintGraphInfo() const {
    printf("=== cuDNN Graph Info ===\n");
    printf("Tensors: %zu\n", tensors_.size());
    printf("Operations: %zu\n", nodes_.size());
    printf("Input tensors: %zu\n", input_tensor_ids_.size());
    printf("Output tensors: %zu\n", output_tensor_ids_.size());
    printf("Workspace size: %zu bytes\n", workspace_size_);
    printf("Finalized: %s\n", is_finalized_ ? "Yes" : "No");
    printf("Compiled: %s\n", is_compiled_ ? "Yes" : "No");
    printf("========================\n");
}

// ===== C Interface Implementation =====

extern "C" {

void* mgpuCreateGraphBuilder(cudnnHandle_t handle) {
    return new CudnnGraphBuilder(handle);
}

void mgpuDestroyGraphBuilder(void* builder) {
    delete static_cast<CudnnGraphBuilder*>(builder);
}

// 修复：C接口函数 - 移除多余参数
int64_t mgpuGraphAddTensor(void* builder, int* dims, int rank, int data_type) {
    auto* graph_builder = static_cast<CudnnGraphBuilder*>(builder);
    std::vector<int> dimensions(dims, dims + rank);
    return graph_builder->AddTensor(dimensions, static_cast<cudnnDataType_t>(data_type), true);
}

void mgpuGraphMarkInput(void* builder, int64_t tensor_id) {
    auto* graph_builder = static_cast<CudnnGraphBuilder*>(builder);
    graph_builder->MarkTensorAsInput(tensor_id);
}

void mgpuGraphMarkOutput(void* builder, int64_t tensor_id) {
    auto* graph_builder = static_cast<CudnnGraphBuilder*>(builder);
    graph_builder->MarkTensorAsOutput(tensor_id);
}

int64_t mgpuGraphAddConv2d(void* builder, 
                          int64_t input_id, int64_t weight_id, int64_t bias_id, int64_t output_id,
                          int pad_h, int pad_w, int stride_h, int stride_w, 
                          int dilation_h, int dilation_w) {
    auto* graph_builder = static_cast<CudnnGraphBuilder*>(builder);
    std::vector<int> pads = {pad_h, pad_w};
    std::vector<int> strides = {stride_h, stride_w};
    std::vector<int> dilations = {dilation_h, dilation_w};
    return graph_builder->AddConvolutionNode(input_id, weight_id, bias_id, output_id, 
                                            pads, strides, dilations);
}

int64_t mgpuGraphAddElementwise(void* builder,
                               int64_t input_a_id, int64_t input_b_id, int64_t output_id,
                               int op_type) {
    auto* graph_builder = static_cast<CudnnGraphBuilder*>(builder);
    return graph_builder->AddElementwiseNode(input_a_id, input_b_id, output_id, 
                                            CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
}

bool mgpuGraphFinalize(void* builder) {
    auto* graph_builder = static_cast<CudnnGraphBuilder*>(builder);
    return graph_builder->FinalizeGraph();
}

bool mgpuGraphCompile(void* builder) {
    auto* graph_builder = static_cast<CudnnGraphBuilder*>(builder);
    return graph_builder->CompileGraph();
}

void mgpuGraphSetTensorData(void* builder, int64_t tensor_id, void* data_ptr) {
    auto* graph_builder = static_cast<CudnnGraphBuilder*>(builder);
    graph_builder->SetTensorData(tensor_id, data_ptr);
}

bool mgpuGraphExecute(void* builder, CUstream stream) {
    auto* graph_builder = static_cast<CudnnGraphBuilder*>(builder);
    return graph_builder->ExecuteGraph(stream);
}

} // extern "C"