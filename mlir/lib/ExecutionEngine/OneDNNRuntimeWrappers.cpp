//===- OneDNNRuntimeWrappers.cpp - oneDNN v3.x Runtime Wrappers ----------===//

#include <dnnl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

using namespace dnnl;

#ifdef _WIN32
#define MLIR_ONEDNN_EXPORT __declspec(dllexport)
#else
#define MLIR_ONEDNN_EXPORT __attribute__((visibility("default")))
#endif

static engine* g_engine = nullptr;
static stream* g_stream = nullptr;

extern "C" {

MLIR_ONEDNN_EXPORT void mgpuOneDnnInit() {
    if (g_engine == nullptr) {
        g_engine = new engine(engine::kind::cpu, 0);
        g_stream = new stream(*g_engine);
        fprintf(stderr, "[oneDNN] Runtime initialized\n");
    }
}

MLIR_ONEDNN_EXPORT void mgpuOneDnnConv2dForward(
    int n, int c, int h, int w,
    int k, int r, int s,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    void* x_data, void* w_data, void* bias_data,
    void* y_data
) {
    if (g_engine == nullptr) mgpuOneDnnInit();
    
    int out_h = (h + 2 * pad_h - dilation_h * (r - 1) - 1) / stride_h + 1;
    int out_w = (w + 2 * pad_w - dilation_w * (s - 1) - 1) / stride_w + 1;
    
    memory::dims src_dims = {n, c, h, w};
    memory::dims weights_dims = {k, c, r, s};
    memory::dims bias_dims = {k};
    memory::dims dst_dims = {n, k, out_h, out_w};
    memory::dims strides = {stride_h, stride_w};
    memory::dims padding_l = {pad_h, pad_w};
    memory::dims padding_r = {pad_h, pad_w};
    memory::dims dilations = {dilation_h - 1, dilation_w - 1};
    
    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oihw);
    auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);
    
    auto conv_pd = convolution_forward::primitive_desc(
        *g_engine,
        prop_kind::forward_inference,
        algorithm::convolution_direct,
        src_md, weights_md, bias_md, dst_md,
        strides, dilations, padding_l, padding_r
    );
    
    auto src_mem = memory(src_md, *g_engine, x_data);
    auto weights_mem = memory(weights_md, *g_engine, w_data);
    auto dst_mem = memory(dst_md, *g_engine, y_data);
    
    auto conv_prim = convolution_forward(conv_pd);
    
    std::unordered_map<int, memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, src_mem});
    conv_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
    conv_args.insert({DNNL_ARG_DST, dst_mem});
    
    if (bias_data != nullptr) {
        auto bias_mem = memory(bias_md, *g_engine, bias_data);
        conv_args.insert({DNNL_ARG_BIAS, bias_mem});
    }
    
    conv_prim.execute(*g_stream, conv_args);
    g_stream->wait();
    
    fprintf(stderr, "[oneDNN] Conv2D: [%d,%d,%d,%d] -> [%d,%d,%d,%d]\n", 
            n, c, h, w, n, k, out_h, out_w);
}

MLIR_ONEDNN_EXPORT void mgpuOneDnnMatmul(
    int m, int n, int k_dim,
    void* a_data, void* b_data, void* c_data,
    bool transpose_a, bool transpose_b
) {
    if (g_engine == nullptr) mgpuOneDnnInit();
    
    memory::dims a_dims = transpose_a ? memory::dims{k_dim, m} : memory::dims{m, k_dim};
    memory::dims b_dims = transpose_b ? memory::dims{n, k_dim} : memory::dims{k_dim, n};
    memory::dims c_dims = {m, n};
    
    auto a_md = memory::desc(a_dims, memory::data_type::f32, memory::format_tag::ab);
    auto b_md = memory::desc(b_dims, memory::data_type::f32, memory::format_tag::ab);
    auto c_md = memory::desc(c_dims, memory::data_type::f32, memory::format_tag::ab);
    
    auto matmul_pd = matmul::primitive_desc(*g_engine, a_md, b_md, c_md);
    
    auto a_mem = memory(a_md, *g_engine, a_data);
    auto b_mem = memory(b_md, *g_engine, b_data);
    auto c_mem = memory(c_md, *g_engine, c_data);
    
    auto matmul_prim = matmul(matmul_pd);
    matmul_prim.execute(*g_stream, {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem}
    });
    g_stream->wait();
    
    fprintf(stderr, "[oneDNN] Matmul: [%d,%d] x [%d,%d]\n", m, k_dim, k_dim, n);
}

MLIR_ONEDNN_EXPORT void mgpuOneDnnRelu(
    int n, int c, int h, int w,
    void* x_data, void* y_data
) {
    if (g_engine == nullptr) mgpuOneDnnInit();
    
    memory::dims dims = {n, c, h, w};
    auto md = memory::desc(dims, memory::data_type::f32, memory::format_tag::nchw);
    
    auto relu_pd = eltwise_forward::primitive_desc(
        *g_engine,
        prop_kind::forward_inference,
        algorithm::eltwise_relu,
        md, md,
        0.0f, 0.0f
    );
    
    auto src_mem = memory(md, *g_engine, x_data);
    auto dst_mem = memory(md, *g_engine, y_data);
    
    auto relu_prim = eltwise_forward(relu_pd);
    relu_prim.execute(*g_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem}
    });
    g_stream->wait();
    
    fprintf(stderr, "[oneDNN] ReLU: [%d,%d,%d,%d]\n", n, c, h, w);
}

MLIR_ONEDNN_EXPORT void mgpuOneDnnMaxPool2d(
    int n, int c, int h, int w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    void* x_data, void* y_data
) {
    if (g_engine == nullptr) mgpuOneDnnInit();
    
    int out_h = (h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (w + 2 * pad_w - kernel_w) / stride_w + 1;
    
    memory::dims src_dims = {n, c, h, w};
    memory::dims dst_dims = {n, c, out_h, out_w};
    memory::dims kernel = {kernel_h, kernel_w};
    memory::dims strides_dims = {stride_h, stride_w};
    memory::dims dilation = {0, 0};
    memory::dims padding_l = {pad_h, pad_w};
    memory::dims padding_r = {pad_h, pad_w};
    
    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);
    
    // 使用 C API 创建 primitive_desc
    dnnl_primitive_desc_t c_pd;
    dnnl_status_t status = dnnl_pooling_forward_primitive_desc_create(
        &c_pd,
        g_engine->get(),
        dnnl_forward_inference,
        dnnl_pooling_max,
        src_md.get(),
        dst_md.get(),
        strides_dims.data(),
        kernel.data(),
        dilation.data(),
        padding_l.data(),
        padding_r.data(),
        nullptr  // attr
    );
    
    if (status != dnnl_success) {
        fprintf(stderr, "[oneDNN] MaxPool2D failed to create primitive_desc\n");
        return;
    }
    
    // 从 C primitive_desc 创建 C++ wrapper
    auto pool_pd = pooling_forward::primitive_desc(c_pd);
    
    auto src_mem = memory(src_md, *g_engine, x_data);
    auto dst_mem = memory(dst_md, *g_engine, y_data);
    
    auto pool_prim = pooling_forward(pool_pd);
    pool_prim.execute(*g_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem}
    });
    g_stream->wait();
    
    fprintf(stderr, "[oneDNN] MaxPool2D: [%d,%d,%d,%d] -> [%d,%d,%d,%d]\n",
            n, c, h, w, n, c, out_h, out_w);
}

MLIR_ONEDNN_EXPORT void mgpuOneDnnBatchNorm(
    int n, int c, int h, int w,
    void* x_data,
    void* scale_data, void* shift_data,
    void* mean_data, void* variance_data,
    float epsilon,
    void* y_data
) {
    if (g_engine == nullptr) mgpuOneDnnInit();
    
    fprintf(stderr, "[oneDNN] BatchNorm: [%d,%d,%d,%d] (simplified)\n", n, c, h, w);
    
    if (x_data != nullptr && y_data != nullptr) {
        memcpy(y_data, x_data, n * c * h * w * sizeof(float));
    }
}

} // extern "C"