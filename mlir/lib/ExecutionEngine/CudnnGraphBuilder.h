//===- CudnnGraphBuilder.h - cuDNN Graph dynamic builder ----------------===//
//
// Dynamic cuDNN Graph builder for parallel operation groups
// Clearly separates graph construction, compilation, and execution phases
//
//===----------------------------------------------------------------------===//

#ifndef CUDNN_GRAPH_BUILDER_H
#define CUDNN_GRAPH_BUILDER_H

#include <cudnn.h>
#include <cudnn_backend.h>
#include "cuda.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <string>

// 版本检查
#if CUDNN_VERSION < 8000
#error "This code requires cuDNN 8.0 or higher"
#endif

// Graph中张量的描述
struct GraphTensor {
    cudnnBackendDescriptor_t desc;  // 使用backend descriptor
    void* data_ptr;
    int64_t tensor_id;
    std::vector<int> dimensions;
    cudnnDataType_t data_type;
    bool is_input;
    bool is_output;
    
    GraphTensor() : desc(nullptr), data_ptr(nullptr), tensor_id(-1), 
                   data_type(CUDNN_DATA_FLOAT), is_input(false), is_output(false) {}
};

// Graph中操作节点的描述
struct GraphNode {
    cudnnBackendDescriptor_t op_desc;
    std::vector<int64_t> input_tensor_ids;
    std::vector<int64_t> output_tensor_ids;
    std::string op_type;
    int64_t node_id;
    
    GraphNode() : op_desc(nullptr), node_id(-1) {}
};

// 动态cuDNN Graph构建器
class CudnnGraphBuilder {
private:
    cudnnHandle_t handle_;
    cudnnBackendDescriptor_t graph_desc_;
    cudnnBackendDescriptor_t engine_desc_;
    cudnnBackendDescriptor_t engine_config_desc_;
    cudnnBackendDescriptor_t exec_plan_desc_;
    
    std::vector<std::unique_ptr<GraphTensor>> tensors_;
    std::vector<std::unique_ptr<GraphNode>> nodes_;
    std::unordered_map<void*, int64_t> ptr_to_tensor_id_;
    
    int64_t next_tensor_id_;
    int64_t next_node_id_;
    bool is_compiled_;
    bool is_finalized_;
    
    // Workspace管理
    void* workspace_;
    size_t workspace_size_;
    
    // 输入输出张量管理
    std::vector<int64_t> input_tensor_ids_;
    std::vector<int64_t> output_tensor_ids_;
    
public:
    CudnnGraphBuilder(cudnnHandle_t handle);
    ~CudnnGraphBuilder();
    
    // 阶段1: 张量管理 (构建阶段 - 不需要实际数据指针)
    int64_t AddTensor(const std::vector<int>& dims, cudnnDataType_t data_type, 
                     bool is_virtual = true);  // 修复：移除data_ptr参数
    int64_t GetOrCreateTensor(void* data_ptr, const std::vector<int>& dims, 
                             cudnnDataType_t data_type);
    void MarkTensorAsInput(int64_t tensor_id);
    void MarkTensorAsOutput(int64_t tensor_id);
    
    // 阶段1: 操作节点添加 (构建阶段 - 只定义计算图结构)
    int64_t AddConvolutionNode(
        int64_t input_tensor_id, int64_t weight_tensor_id, int64_t bias_tensor_id,
        int64_t output_tensor_id, const std::vector<int>& pads, 
        const std::vector<int>& strides, const std::vector<int>& dilations);
    
    int64_t AddPoolingNode(
        int64_t input_tensor_id, int64_t output_tensor_id,
        cudnnPoolingMode_t mode, const std::vector<int>& window_dims,
        const std::vector<int>& pads, const std::vector<int>& strides);
        
    int64_t AddElementwiseNode(
        int64_t input_a_id, int64_t input_b_id, int64_t output_id,
        cudnnBackendDescriptorType_t op_type); // ADD, MUL, etc.
        
    int64_t AddMatmulNode(
        int64_t input_a_id, int64_t input_b_id, int64_t output_id);
        
    int64_t AddUnaryNode(
        int64_t input_id, int64_t output_id,
        cudnnBackendDescriptorType_t op_type); // NEG, etc.
    
    // 阶段2: Graph构建和编译 (编译阶段 - 不需要stream)
    bool FinalizeGraph();
    bool CompileGraph();
    
    // 阶段3: 数据绑定和执行 (执行阶段 - 需要stream和实际数据)
    void SetTensorData(int64_t tensor_id, void* data_ptr);
    void SetTensorDataByPointer(void* old_ptr, void* new_ptr);  // 用于更新数据指针
    bool ExecuteGraph(CUstream stream);
    
    // 辅助函数
    void Reset();
    size_t GetWorkspaceSize() const { return workspace_size_; }
    bool IsCompiled() const { return is_compiled_; }
    bool IsFinalized() const { return is_finalized_; }
    
    // 调试和信息
    void PrintGraphInfo() const;
    int GetNodeCount() const { return nodes_.size(); }
    int GetTensorCount() const { return tensors_.size(); }
    
    // 高级接口 - 自动标记输入输出
    void AutoMarkInputsOutputs();
};

// C接口包装器 - 清楚分离构建和执行阶段
extern "C" {
    // 阶段1: Graph构建管理 (不需要stream)
    void* mgpuCreateGraphBuilder(cudnnHandle_t handle);
    void mgpuDestroyGraphBuilder(void* builder);
    
    // 阶段1: 张量管理 (构建阶段) - 修复：移除data_ptr参数
    int64_t mgpuGraphAddTensor(void* builder, int* dims, int rank, int data_type);
    void mgpuGraphMarkInput(void* builder, int64_t tensor_id);
    void mgpuGraphMarkOutput(void* builder, int64_t tensor_id);
    
    // 阶段1: 操作添加 (构建阶段 - 不需要stream)
    int64_t mgpuGraphAddConv2d(void* builder, 
                              int64_t input_id, int64_t weight_id, int64_t bias_id, int64_t output_id,
                              int pad_h, int pad_w, int stride_h, int stride_w, 
                              int dilation_h, int dilation_w);
    
    int64_t mgpuGraphAddMaxPool(void* builder,
                               int64_t input_id, int64_t output_id,
                               int kernel_h, int kernel_w, int pad_h, int pad_w,
                               int stride_h, int stride_w);
                               
    int64_t mgpuGraphAddElementwise(void* builder,
                                   int64_t input_a_id, int64_t input_b_id, int64_t output_id,
                                   int op_type); // 0=ADD, 1=MUL, 2=SUB
                                   
    int64_t mgpuGraphAddMatmul(void* builder,
                              int64_t input_a_id, int64_t input_b_id, int64_t output_id);
    
    // 阶段2: Graph编译 (编译阶段 - 不需要stream)
    bool mgpuGraphFinalize(void* builder);
    bool mgpuGraphCompile(void* builder);
    
    // 阶段3: 数据绑定和执行 (执行阶段 - 需要stream)
    void mgpuGraphSetTensorData(void* builder, int64_t tensor_id, void* data_ptr);
    bool mgpuGraphExecute(void* builder, CUstream stream);
    
    // 并行组高级接口
    int mgpuCreateParallelGroupGraph(cudnnHandle_t handle);
    void mgpuDestroyParallelGroupGraph(int group_id);
    bool mgpuParallelGroupExecute(int group_id, CUstream stream);
}

#endif // CUDNN_GRAPH_BUILDER_H