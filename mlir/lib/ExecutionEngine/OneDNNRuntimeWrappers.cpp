// //===- OneDNNRuntimeWrappers.cpp - oneDNN v3.x Runtime Wrappers ----------===//

// #include <dnnl.hpp>
// #include <cstdio>
// #include <cstdlib>
// #include <cstring>
// #include <unordered_map>
// #include <chrono>
// #include <iostream>

// using namespace dnnl;

// #ifdef _WIN32
// #define MLIR_ONEDNN_EXPORT __declspec(dllexport)
// #else
// #define MLIR_ONEDNN_EXPORT __attribute__((visibility("default")))
// #endif

// static engine* g_engine = nullptr;
// static stream* g_stream = nullptr;

// extern "C" {

// MLIR_ONEDNN_EXPORT double get_time() {
//     auto now = std::chrono::high_resolution_clock::now();
//     return std::chrono::duration<double>(now.time_since_epoch()).count();
//   }
  
// MLIR_ONEDNN_EXPORT void print_time(double elapsed) {
//     std::cout << "Execution time: " << elapsed << " seconds" << std::endl;
//   }


// MLIR_ONEDNN_EXPORT void mgpuOneDnnInit() {
//     if (g_engine == nullptr) {
//         g_engine = new engine(engine::kind::cpu, 0);
//         g_stream = new stream(*g_engine);
//         fprintf(stderr, "[oneDNN] Runtime initialized\n");
//     }
// }

// MLIR_ONEDNN_EXPORT void mgpuOneDnnConv2dForward(
//     int n, int c, int h, int w,
//     int k, int r, int s,
//     int pad_h, int pad_w,
//     int stride_h, int stride_w,
//     int dilation_h, int dilation_w,
//     void* x_data, void* w_data, void* bias_data,
//     void* y_data
// ) {
//     if (g_engine == nullptr) mgpuOneDnnInit();
    
//     int out_h = (h + 2 * pad_h - dilation_h * (r - 1) - 1) / stride_h + 1;
//     int out_w = (w + 2 * pad_w - dilation_w * (s - 1) - 1) / stride_w + 1;
    
//     memory::dims src_dims = {n, c, h, w};
//     memory::dims weights_dims = {k, c, r, s};
//     memory::dims bias_dims = {k};
//     memory::dims dst_dims = {n, k, out_h, out_w};
//     memory::dims strides = {stride_h, stride_w};
//     memory::dims padding_l = {pad_h, pad_w};
//     memory::dims padding_r = {pad_h, pad_w};
//     memory::dims dilations = {dilation_h - 1, dilation_w - 1};
    
//     auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
//     auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oihw);
//     auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x);
//     auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);
    
//     auto conv_pd = convolution_forward::primitive_desc(
//         *g_engine,
//         prop_kind::forward_inference,
//         algorithm::convolution_direct,
//         src_md, weights_md, bias_md, dst_md,
//         strides, dilations, padding_l, padding_r
//     );
    
//     auto src_mem = memory(src_md, *g_engine, x_data);
//     auto weights_mem = memory(weights_md, *g_engine, w_data);
//     auto dst_mem = memory(dst_md, *g_engine, y_data);
    
//     auto conv_prim = convolution_forward(conv_pd);
    
//     std::unordered_map<int, memory> conv_args;
//     conv_args.insert({DNNL_ARG_SRC, src_mem});
//     conv_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
//     conv_args.insert({DNNL_ARG_DST, dst_mem});
    
//     if (bias_data != nullptr) {
//         auto bias_mem = memory(bias_md, *g_engine, bias_data);
//         conv_args.insert({DNNL_ARG_BIAS, bias_mem});
//     }
    
//     conv_prim.execute(*g_stream, conv_args);
//     g_stream->wait();
    
//     // fprintf(stderr, "[oneDNN] Conv2D: [%d,%d,%d,%d] -> [%d,%d,%d,%d]\n", 
//     //         n, c, h, w, n, k, out_h, out_w);
// }

// MLIR_ONEDNN_EXPORT void mgpuOneDnnMatMul(
//     int batch, int m, int k, int n,
//     bool transpose_a, bool transpose_b,
//     void* a_data, void* b_data, void* c_data
// ) {
//     if (g_engine == nullptr) mgpuOneDnnInit();
    
//     // 根据是否转置确定维度
//     memory::dims a_dims = transpose_a ? 
//         (batch > 1 ? memory::dims{batch, k, m} : memory::dims{k, m}) :
//         (batch > 1 ? memory::dims{batch, m, k} : memory::dims{m, k});
    
//     memory::dims b_dims = transpose_b ? 
//         (batch > 1 ? memory::dims{batch, n, k} : memory::dims{n, k}) :
//         (batch > 1 ? memory::dims{batch, k, n} : memory::dims{k, n});
    
//     memory::dims c_dims = batch > 1 ? memory::dims{batch, m, n} : memory::dims{m, n};
    
//     // 确定格式标签
//     auto a_format = batch > 1 ? 
//         (transpose_a ? memory::format_tag::bca : memory::format_tag::abc) :
//         (transpose_a ? memory::format_tag::ba : memory::format_tag::ab);
    
//     auto b_format = batch > 1 ? 
//         (transpose_b ? memory::format_tag::bca : memory::format_tag::abc) :
//         (transpose_b ? memory::format_tag::ba : memory::format_tag::ab);
    
//     auto c_format = batch > 1 ? memory::format_tag::abc : memory::format_tag::ab;
    
//     auto a_md = memory::desc(a_dims, memory::data_type::f32, a_format);
//     auto b_md = memory::desc(b_dims, memory::data_type::f32, b_format);
//     auto c_md = memory::desc(c_dims, memory::data_type::f32, c_format);
    
//     // 创建matmul primitive descriptor
//     auto matmul_pd = matmul::primitive_desc(*g_engine, a_md, b_md, c_md);
    
//     auto a_mem = memory(a_md, *g_engine, a_data);
//     auto b_mem = memory(b_md, *g_engine, b_data);
//     auto c_mem = memory(c_md, *g_engine, c_data);
    
//     auto matmul_prim = matmul(matmul_pd);
    
//     std::unordered_map<int, memory> matmul_args;
//     matmul_args.insert({DNNL_ARG_SRC, a_mem});
//     matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
//     matmul_args.insert({DNNL_ARG_DST, c_mem});
    
//     matmul_prim.execute(*g_stream, matmul_args);
//     g_stream->wait();
    
//     // if (batch > 1) {
//     //     fprintf(stderr, "[oneDNN] BatchMatMul: [%d,%d,%d] x [%d,%d,%d] -> [%d,%d,%d] (T_A=%d, T_B=%d)\n",
//     //             batch, m, k, batch, k, n, batch, m, n, transpose_a, transpose_b);
//     // } else {
//     //     fprintf(stderr, "[oneDNN] MatMul: [%d,%d] x [%d,%d] -> [%d,%d] (T_A=%d, T_B=%d)\n",
//     //             m, k, k, n, m, n, transpose_a, transpose_b);
//     // }
// }

// MLIR_ONEDNN_EXPORT void mgpuOneDnnMaxPool2d(
//     int n, int c, int h, int w,
//     int kernel_h, int kernel_w,
//     int pad_h, int pad_w,
//     int stride_h, int stride_w,
//     int dilation_h, int dilation_w,
//     void* x_data, void* y_data
// ) {
//     if (g_engine == nullptr) mgpuOneDnnInit();
    
//     // 计算输出尺寸
//     int out_h = (h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
//     int out_w = (w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
//     memory::dims src_dims = {n, c, h, w};
//     memory::dims dst_dims = {n, c, out_h, out_w};
//     memory::dims kernel = {kernel_h, kernel_w};
//     memory::dims strides = {stride_h, stride_w};
//     memory::dims padding_l = {pad_h, pad_w};
//     memory::dims padding_r = {pad_h, pad_w};
//     memory::dims dilations = {dilation_h - 1, dilation_w - 1};
    
//     auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
//     auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);
    
//     // 创建pooling primitive descriptor
//     auto pool_pd = pooling_forward::primitive_desc(
//         *g_engine,
//         prop_kind::forward_inference,
//         algorithm::pooling_max,
//         src_md, dst_md,
//         strides, kernel, dilations,
//         padding_l, padding_r
//     );
    
//     auto src_mem = memory(src_md, *g_engine, x_data);
//     auto dst_mem = memory(dst_md, *g_engine, y_data);
    
//     auto pool_prim = pooling_forward(pool_pd);
    
//     std::unordered_map<int, memory> pool_args;
//     pool_args.insert({DNNL_ARG_SRC, src_mem});
//     pool_args.insert({DNNL_ARG_DST, dst_mem});
    
//     pool_prim.execute(*g_stream, pool_args);
//     g_stream->wait();
    
//     // fprintf(stderr, "[oneDNN] MaxPool2D: [%d,%d,%d,%d] -> [%d,%d,%d,%d] (kernel=%dx%d, stride=%dx%d)\n",
//     //         n, c, h, w, n, c, out_h, out_w, kernel_h, kernel_w, stride_h, stride_w);
// }

// MLIR_ONEDNN_EXPORT void mgpuOneDnnAvgPool2d(
//     int n, int c, int h, int w,
//     int kernel_h, int kernel_w,
//     int pad_h, int pad_w,
//     int stride_h, int stride_w,
//     int dilation_h, int dilation_w,
//     bool count_include_pad,
//     void* x_data, void* y_data
// ) {
//     if (g_engine == nullptr) mgpuOneDnnInit();
    
//     int out_h = (h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
//     int out_w = (w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
//     memory::dims src_dims = {n, c, h, w};
//     memory::dims dst_dims = {n, c, out_h, out_w};
//     memory::dims kernel = {kernel_h, kernel_w};
//     memory::dims strides = {stride_h, stride_w};
//     memory::dims padding_l = {pad_h, pad_w};
//     memory::dims padding_r = {pad_h, pad_w};
//     memory::dims dilations = {dilation_h - 1, dilation_w - 1};
    
//     auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
//     auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);
    
//     // 根据count_include_pad选择算法
//     algorithm pool_algo = count_include_pad ? 
//         algorithm::pooling_avg_include_padding : 
//         algorithm::pooling_avg_exclude_padding;
    
//     auto pool_pd = pooling_forward::primitive_desc(
//         *g_engine,
//         prop_kind::forward_inference,
//         pool_algo,
//         src_md, dst_md,
//         strides, kernel, dilations,
//         padding_l, padding_r
//     );
    
//     auto src_mem = memory(src_md, *g_engine, x_data);
//     auto dst_mem = memory(dst_md, *g_engine, y_data);
    
//     auto pool_prim = pooling_forward(pool_pd);
    
//     std::unordered_map<int, memory> pool_args;
//     pool_args.insert({DNNL_ARG_SRC, src_mem});
//     pool_args.insert({DNNL_ARG_DST, dst_mem});
    
//     pool_prim.execute(*g_stream, pool_args);
//     g_stream->wait();
    
//     // fprintf(stderr, "[oneDNN] AvgPool2D: [%d,%d,%d,%d] -> [%d,%d,%d,%d] (kernel=%dx%d, stride=%dx%d)\n",
//     //         n, c, h, w, n, c, out_h, out_w, kernel_h, kernel_w, stride_h, stride_w);
// }

// MLIR_ONEDNN_EXPORT void mgpuOneDnnReduceMean(
//     int n, int c, int h, int w,
//     int axis_h, int axis_w,  // 要 reduce 的轴（0表示该轴不reduce）
//     bool keepdims,
//     void* x_data, void* y_data
// ) {
//     if (g_engine == nullptr) mgpuOneDnnInit();
    
//     // 确定输出尺寸
//     int out_h = axis_h ? (keepdims ? 1 : 0) : h;
//     int out_w = axis_w ? (keepdims ? 1 : 0) : w;
    
//     // 对于在 H 和 W 维度上的 reduce mean，使用全局平均池化
//     if (axis_h && axis_w) {
//         memory::dims src_dims = {n, c, h, w};
//         memory::dims dst_dims = keepdims ? 
//             memory::dims{n, c, 1, 1} : 
//             memory::dims{n, c};
        
//         // 使用全局平均池化：kernel size = 输入的 H 和 W
//         memory::dims kernel = {h, w};
//         memory::dims strides = {1, 1};
//         memory::dims padding_l = {0, 0};
//         memory::dims padding_r = {0, 0};
//         memory::dims dilations = {0, 0};
        
//         auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
        
//         // 根据 keepdims 选择不同的输出格式
//         memory::format_tag dst_format = keepdims ? 
//             memory::format_tag::nchw : 
//             memory::format_tag::nc;
//         auto dst_md = memory::desc(dst_dims, memory::data_type::f32, dst_format);
        
//         // 使用 pooling_avg_exclude_padding 以获得正确的平均值
//         auto pool_pd = pooling_forward::primitive_desc(
//             *g_engine,
//             prop_kind::forward_inference,
//             algorithm::pooling_avg_exclude_padding,
//             src_md, dst_md,
//             strides, kernel, dilations,
//             padding_l, padding_r
//         );
        
//         auto src_mem = memory(src_md, *g_engine, x_data);
//         auto dst_mem = memory(dst_md, *g_engine, y_data);
        
//         auto pool_prim = pooling_forward(pool_pd);
        
//         std::unordered_map<int, memory> pool_args;
//         pool_args.insert({DNNL_ARG_SRC, src_mem});
//         pool_args.insert({DNNL_ARG_DST, dst_mem});
        
//         pool_prim.execute(*g_stream, pool_args);
//         g_stream->wait();
        
//         if (keepdims) {
//             fprintf(stderr, "[oneDNN] ReduceMean: [%d,%d,%d,%d] -> [%d,%d,1,1] (axes=[2,3], keepdims=true)\n",
//                     n, c, h, w, n, c);
//         } else {
//             fprintf(stderr, "[oneDNN] ReduceMean: [%d,%d,%d,%d] -> [%d,%d] (axes=[2,3], keepdims=false)\n",
//                     n, c, h, w, n, c);
//         }
//     } else {
//         fprintf(stderr, "[oneDNN] ReduceMean: Unsupported axis configuration\n");
//     }
// }

// } // extern "C"



//===- OneDNNRuntimeWrappers.cpp - oneDNN v3.x Runtime Wrappers ----------===//

#include <dnnl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <mutex>
#include <tuple>
#include <memory>

using namespace dnnl;

#ifdef _WIN32
#define MLIR_ONEDNN_EXPORT __declspec(dllexport)
#else
#define MLIR_ONEDNN_EXPORT __attribute__((visibility("default")))
#endif

static engine* g_engine = nullptr;
static stream* g_stream = nullptr;

// ============================================================================
// Conv2D Primitive 缓存
// ============================================================================

// 定义 Conv2D 的参数 key
struct Conv2dKey {
    int n, c, h, w;
    int k, r, s;
    int pad_h, pad_w;
    int stride_h, stride_w;
    int dilation_h, dilation_w;
    bool has_bias;
    
    bool operator==(const Conv2dKey& other) const {
        return std::tie(n, c, h, w, k, r, s, pad_h, pad_w, 
                       stride_h, stride_w, dilation_h, dilation_w, has_bias) ==
               std::tie(other.n, other.c, other.h, other.w, other.k, other.r, other.s,
                       other.pad_h, other.pad_w, other.stride_h, other.stride_w,
                       other.dilation_h, other.dilation_w, other.has_bias);
    }
};

// Hash 函数
struct Conv2dKeyHash {
    size_t operator()(const Conv2dKey& k) const {
        size_t h = 0;
        auto hash_combine = [&h](int val) {
            h ^= std::hash<int>{}(val) + 0x9e3779b9 + (h << 6) + (h >> 2);
        };
        
        hash_combine(k.n);
        hash_combine(k.c);
        hash_combine(k.h);
        hash_combine(k.w);
        hash_combine(k.k);
        hash_combine(k.r);
        hash_combine(k.s);
        hash_combine(k.pad_h);
        hash_combine(k.pad_w);
        hash_combine(k.stride_h);
        hash_combine(k.stride_w);
        hash_combine(k.dilation_h);
        hash_combine(k.dilation_w);
        hash_combine(k.has_bias ? 1 : 0);
        
        return h;
    }
};

// 缓存的 primitive 和 descriptor - 支持格式转换
struct Conv2dPrimitive {
    convolution_forward::primitive_desc pd;
    convolution_forward primitive;
    
    // 用户格式 (NCHW)
    memory::desc src_md_user;
    memory::desc dst_md_user;
    
    // oneDNN 最优格式
    memory::desc src_md_optimal;
    memory::desc weights_md_optimal;
    memory::desc dst_md_optimal;
    memory::desc bias_md;
    
    // 预转换的权重（只转换一次，重复使用）
    std::shared_ptr<memory> weights_reordered;
    bool weights_need_reorder;
    
    Conv2dPrimitive(
        const convolution_forward::primitive_desc& _pd,
        const convolution_forward& _prim,
        const memory::desc& _src_md_user,
        const memory::desc& _src_md_optimal,
        const memory::desc& _weights_md_optimal,
        const memory::desc& _dst_md_user,
        const memory::desc& _dst_md_optimal,
        const memory::desc& _bias_md,
        std::shared_ptr<memory> _weights_reordered,
        bool _weights_need_reorder
    ) : pd(_pd), primitive(_prim), 
        src_md_user(_src_md_user), dst_md_user(_dst_md_user),
        src_md_optimal(_src_md_optimal), weights_md_optimal(_weights_md_optimal),
        dst_md_optimal(_dst_md_optimal), bias_md(_bias_md),
        weights_reordered(_weights_reordered), weights_need_reorder(_weights_need_reorder) {}
};

// 全局缓存
static std::unordered_map<Conv2dKey, Conv2dPrimitive, Conv2dKeyHash> g_conv2d_cache;
static std::mutex g_conv2d_cache_mutex;

extern "C" {

MLIR_ONEDNN_EXPORT double get_time() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}
  
MLIR_ONEDNN_EXPORT void print_time(double elapsed) {
    std::cout << "Execution time: " << elapsed << " seconds" << std::endl;
}

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
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // 创建 cache key
    Conv2dKey key{n, c, h, w, k, r, s, pad_h, pad_w, 
                  stride_h, stride_w, dilation_h, dilation_w,
                  bias_data != nullptr};
    
    auto t_key_created = std::chrono::high_resolution_clock::now();
    
    // 查找或创建 primitive
    std::unique_lock<std::mutex> lock(g_conv2d_cache_mutex);
    auto it = g_conv2d_cache.find(key);
    
    auto t_cache_lookup = std::chrono::high_resolution_clock::now();
    
    bool cache_miss = (it == g_conv2d_cache.end());
    auto t_primitive_create_start = std::chrono::high_resolution_clock::now();
    
    if (cache_miss) {
        // Cache miss - 创建新的 primitive
        lock.unlock(); // 解锁，避免在创建时持有锁
        
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
        
        // 用户格式 (NCHW)
        auto src_md_user = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
        auto dst_md_user = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);
        auto weights_md_user = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oihw);
        
        //   关键：让 oneDNN 选择最优格式
        auto src_md_any = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::any);
        auto weights_md_any = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::any);
        auto dst_md_any = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::any);
        auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x);
        
        auto t_md_created = std::chrono::high_resolution_clock::now();
        
        auto conv_pd = convolution_forward::primitive_desc(
            *g_engine,
            prop_kind::forward_inference,
            algorithm::convolution_auto,  // 使用 auto 让 oneDNN 选择最优算法
            src_md_any, weights_md_any, bias_md, dst_md_any,
            strides, dilations, padding_l, padding_r
        );
        
        auto t_pd_created = std::chrono::high_resolution_clock::now();
        
        auto conv_prim = convolution_forward(conv_pd);
        
        auto t_prim_created = std::chrono::high_resolution_clock::now();
        
        //   预转换权重（只做一次，保存在缓存中）
        std::shared_ptr<memory> weights_reordered;
        bool weights_need_reorder = (conv_pd.weights_desc() != weights_md_user);
        
        if (weights_need_reorder) {
            auto weights_mem_user = memory(weights_md_user, *g_engine, w_data);
            weights_reordered = std::make_shared<memory>(conv_pd.weights_desc(), *g_engine);
            reorder(weights_mem_user, *weights_reordered)
                .execute(*g_stream, weights_mem_user, *weights_reordered);
            g_stream->wait();
        }
        
        auto t_weights_reorder = std::chrono::high_resolution_clock::now();
        
        // 插入缓存
        lock.lock();
        auto result = g_conv2d_cache.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(key),
            std::forward_as_tuple(
                conv_pd, conv_prim,
                src_md_user, conv_pd.src_desc(),
                conv_pd.weights_desc(),
                dst_md_user, conv_pd.dst_desc(),
                bias_md,
                weights_reordered,
                weights_need_reorder
            )
        );
        it = result.first;
        
        auto t_cache_insert = std::chrono::high_resolution_clock::now();
        
        // 打印 primitive 创建详细耗时
        auto md_time = std::chrono::duration<double, std::milli>(t_md_created - t_primitive_create_start).count();
        auto pd_time = std::chrono::duration<double, std::milli>(t_pd_created - t_md_created).count();
        auto prim_time = std::chrono::duration<double, std::milli>(t_prim_created - t_pd_created).count();
        auto weights_reorder_time = std::chrono::duration<double, std::milli>(t_weights_reorder - t_prim_created).count();
        auto insert_time = std::chrono::duration<double, std::milli>(t_cache_insert - t_weights_reorder).count();
        
        fprintf(stderr, "[oneDNN] Conv2D primitive created and cached: [%d,%d,%d,%d] -> [%d,%d,%d,%d]\n", 
                n, c, h, w, n, k, out_h, out_w);
        fprintf(stderr, "[Timing-Create] md_desc:%.3fms, prim_desc:%.3fms, primitive:%.3fms, weights_reorder:%.3fms, cache_insert:%.3fms\n",
                md_time, pd_time, prim_time, weights_reorder_time, insert_time);
        fprintf(stderr, "[Format] src_reorder:%s, weights_reorder:%s, dst_reorder:%s\n",
                (conv_pd.src_desc() != src_md_user) ? "YES" : "NO",
                weights_need_reorder ? "YES" : "NO",
                (conv_pd.dst_desc() != dst_md_user) ? "YES" : "NO");
    }
    
    auto t_primitive_ready = std::chrono::high_resolution_clock::now();
    
    // 使用缓存的 primitive
    const Conv2dPrimitive& cached = it->second;
    lock.unlock();
    
    auto t_lock_released = std::chrono::high_resolution_clock::now();
    
    //   Reorder 输入（如果需要）
    auto src_mem_user = memory(cached.src_md_user, *g_engine, x_data);
    memory src_mem_internal;
    
    if (cached.src_md_optimal != cached.src_md_user) {
        src_mem_internal = memory(cached.src_md_optimal, *g_engine);
        reorder(src_mem_user, src_mem_internal)
            .execute(*g_stream, src_mem_user, src_mem_internal);
    } else {
        src_mem_internal = src_mem_user;
    }
    
    auto t_src_reorder = std::chrono::high_resolution_clock::now();
    
    //   使用预转换的权重（如果有）
    memory weights_mem;
    if (cached.weights_reordered) {
        weights_mem = *cached.weights_reordered;
    } else {
        weights_mem = memory(cached.weights_md_optimal, *g_engine, w_data);
    }
    
    auto t_weights_ready = std::chrono::high_resolution_clock::now();
    
    //   准备输出内存
    auto dst_mem_user = memory(cached.dst_md_user, *g_engine, y_data);
    memory dst_mem_internal;
    
    if (cached.dst_md_optimal != cached.dst_md_user) {
        dst_mem_internal = memory(cached.dst_md_optimal, *g_engine);
    } else {
        dst_mem_internal = dst_mem_user;
    }
    
    auto t_dst_alloc = std::chrono::high_resolution_clock::now();
    
    // 准备执行参数
    std::unordered_map<int, memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, src_mem_internal});
    conv_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
    conv_args.insert({DNNL_ARG_DST, dst_mem_internal});
    
    if (bias_data != nullptr) {
        auto bias_mem = memory(cached.bias_md, *g_engine, bias_data);
        conv_args.insert({DNNL_ARG_BIAS, bias_mem});
    }
    
    auto t_args_prepared = std::chrono::high_resolution_clock::now();
    
    // 执行
    cached.primitive.execute(*g_stream, conv_args);
    
    auto t_executed = std::chrono::high_resolution_clock::now();
    
    //   Reorder 输出（如果需要）
    if (cached.dst_md_optimal != cached.dst_md_user) {
        reorder(dst_mem_internal, dst_mem_user)
            .execute(*g_stream, dst_mem_internal, dst_mem_user);
    }
    
    auto t_dst_reorder = std::chrono::high_resolution_clock::now();
    
    g_stream->wait();
    
    auto t_end = std::chrono::high_resolution_clock::now();
    
    // 计算各部分耗时
    auto key_time = std::chrono::duration<double, std::milli>(t_key_created - t_start).count();
    auto lookup_time = std::chrono::duration<double, std::milli>(t_cache_lookup - t_key_created).count();
    auto primitive_time = std::chrono::duration<double, std::milli>(t_primitive_ready - t_cache_lookup).count();
    auto unlock_time = std::chrono::duration<double, std::milli>(t_lock_released - t_primitive_ready).count();
    auto src_reorder_time = std::chrono::duration<double, std::milli>(t_src_reorder - t_lock_released).count();
    auto weights_time = std::chrono::duration<double, std::milli>(t_weights_ready - t_src_reorder).count();
    auto dst_alloc_time = std::chrono::duration<double, std::milli>(t_dst_alloc - t_weights_ready).count();
    auto args_time = std::chrono::duration<double, std::milli>(t_args_prepared - t_dst_alloc).count();
    auto exec_time = std::chrono::duration<double, std::milli>(t_executed - t_args_prepared).count();
    auto dst_reorder_time = std::chrono::duration<double, std::milli>(t_dst_reorder - t_executed).count();
    auto wait_time = std::chrono::duration<double, std::milli>(t_end - t_dst_reorder).count();
    auto total_time = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    // 打印详细耗时分解
    // fprintf(stderr, "[Timing-Conv] shape:[%d,%d,%d,%d]->[%d,%d,?,?] %s | "
    //                 "key:%.3f lookup:%.3f prim:%.3f unlock:%.3f | "
    //                 "src_reorder:%.3f weights:%.3f dst_alloc:%.3f args:%.3f | "
    //                 "exec:%.3f dst_reorder:%.3f wait:%.3f | "
    //                 "TOTAL:%.3fms\n",
    //         n, c, h, w, n, k,
    //         cache_miss ? "MISS" : "HIT ",
    //         key_time, lookup_time, primitive_time, unlock_time,
    //         src_reorder_time, weights_time, dst_alloc_time, args_time,
    //         exec_time, dst_reorder_time, wait_time,
    //         total_time);
}

// 清理缓存的辅助函数（可选）
MLIR_ONEDNN_EXPORT void mgpuOneDnnClearCache() {
    std::lock_guard<std::mutex> lock(g_conv2d_cache_mutex);
    g_conv2d_cache.clear();
    fprintf(stderr, "[oneDNN] Primitive cache cleared\n");
}

MLIR_ONEDNN_EXPORT size_t mgpuOneDnnGetCacheSize() {
    std::lock_guard<std::mutex> lock(g_conv2d_cache_mutex);
    return g_conv2d_cache.size();
}

MLIR_ONEDNN_EXPORT void mgpuOneDnnMatMul(
    int batch, int m, int k, int n,
    bool transpose_a, bool transpose_b,
    void* a_data, void* b_data, void* c_data
) {
    if (g_engine == nullptr) mgpuOneDnnInit();
    
    // 根据是否转置确定维度
    memory::dims a_dims = transpose_a ? 
        (batch > 1 ? memory::dims{batch, k, m} : memory::dims{k, m}) :
        (batch > 1 ? memory::dims{batch, m, k} : memory::dims{m, k});
    
    memory::dims b_dims = transpose_b ? 
        (batch > 1 ? memory::dims{batch, n, k} : memory::dims{n, k}) :
        (batch > 1 ? memory::dims{batch, k, n} : memory::dims{k, n});
    
    memory::dims c_dims = batch > 1 ? memory::dims{batch, m, n} : memory::dims{m, n};
    
    // 确定格式标签
    auto a_format = batch > 1 ? 
        (transpose_a ? memory::format_tag::bca : memory::format_tag::abc) :
        (transpose_a ? memory::format_tag::ba : memory::format_tag::ab);
    
    auto b_format = batch > 1 ? 
        (transpose_b ? memory::format_tag::bca : memory::format_tag::abc) :
        (transpose_b ? memory::format_tag::ba : memory::format_tag::ab);
    
    auto c_format = batch > 1 ? memory::format_tag::abc : memory::format_tag::ab;
    
    auto a_md = memory::desc(a_dims, memory::data_type::f32, a_format);
    auto b_md = memory::desc(b_dims, memory::data_type::f32, b_format);
    auto c_md = memory::desc(c_dims, memory::data_type::f32, c_format);
    
    // 创建matmul primitive descriptor
    auto matmul_pd = matmul::primitive_desc(*g_engine, a_md, b_md, c_md);
    
    auto a_mem = memory(a_md, *g_engine, a_data);
    auto b_mem = memory(b_md, *g_engine, b_data);
    auto c_mem = memory(c_md, *g_engine, c_data);
    
    auto matmul_prim = matmul(matmul_pd);
    
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, a_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
    matmul_args.insert({DNNL_ARG_DST, c_mem});
    
    matmul_prim.execute(*g_stream, matmul_args);
    g_stream->wait();
}

MLIR_ONEDNN_EXPORT void mgpuOneDnnMaxPool2d(
    int n, int c, int h, int w,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    void* x_data, void* y_data
) {
    if (g_engine == nullptr) mgpuOneDnnInit();
    
    // 计算输出尺寸
    int out_h = (h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    memory::dims src_dims = {n, c, h, w};
    memory::dims dst_dims = {n, c, out_h, out_w};
    memory::dims kernel = {kernel_h, kernel_w};
    memory::dims strides = {stride_h, stride_w};
    memory::dims padding_l = {pad_h, pad_w};
    memory::dims padding_r = {pad_h, pad_w};
    memory::dims dilations = {dilation_h - 1, dilation_w - 1};
    
    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);
    
    // 创建pooling primitive descriptor
    auto pool_pd = pooling_forward::primitive_desc(
        *g_engine,
        prop_kind::forward_inference,
        algorithm::pooling_max,
        src_md, dst_md,
        strides, kernel, dilations,
        padding_l, padding_r
    );
    
    auto src_mem = memory(src_md, *g_engine, x_data);
    auto dst_mem = memory(dst_md, *g_engine, y_data);
    
    auto pool_prim = pooling_forward(pool_pd);
    
    std::unordered_map<int, memory> pool_args;
    pool_args.insert({DNNL_ARG_SRC, src_mem});
    pool_args.insert({DNNL_ARG_DST, dst_mem});
    
    pool_prim.execute(*g_stream, pool_args);
    g_stream->wait();
}

MLIR_ONEDNN_EXPORT void mgpuOneDnnAvgPool2d(
    int n, int c, int h, int w,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    bool count_include_pad,
    void* x_data, void* y_data
) {
    if (g_engine == nullptr) mgpuOneDnnInit();
    
    int out_h = (h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    memory::dims src_dims = {n, c, h, w};
    memory::dims dst_dims = {n, c, out_h, out_w};
    memory::dims kernel = {kernel_h, kernel_w};
    memory::dims strides = {stride_h, stride_w};
    memory::dims padding_l = {pad_h, pad_w};
    memory::dims padding_r = {pad_h, pad_w};
    memory::dims dilations = {dilation_h - 1, dilation_w - 1};
    
    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);
    
    // 根据count_include_pad选择算法
    algorithm pool_algo = count_include_pad ? 
        algorithm::pooling_avg_include_padding : 
        algorithm::pooling_avg_exclude_padding;
    
    auto pool_pd = pooling_forward::primitive_desc(
        *g_engine,
        prop_kind::forward_inference,
        pool_algo,
        src_md, dst_md,
        strides, kernel, dilations,
        padding_l, padding_r
    );
    
    auto src_mem = memory(src_md, *g_engine, x_data);
    auto dst_mem = memory(dst_md, *g_engine, y_data);
    
    auto pool_prim = pooling_forward(pool_pd);
    
    std::unordered_map<int, memory> pool_args;
    pool_args.insert({DNNL_ARG_SRC, src_mem});
    pool_args.insert({DNNL_ARG_DST, dst_mem});
    
    pool_prim.execute(*g_stream, pool_args);
    g_stream->wait();
}

MLIR_ONEDNN_EXPORT void mgpuOneDnnReduceMean(
    int n, int c, int h, int w,
    int axis_h, int axis_w,
    bool keepdims,
    void* x_data, void* y_data
) {
    if (g_engine == nullptr) mgpuOneDnnInit();
    
    // 确定输出尺寸
    int out_h = axis_h ? (keepdims ? 1 : 0) : h;
    int out_w = axis_w ? (keepdims ? 1 : 0) : w;
    
    // 对于在 H 和 W 维度上的 reduce mean，使用全局平均池化
    if (axis_h && axis_w) {
        memory::dims src_dims = {n, c, h, w};
        memory::dims dst_dims = keepdims ? 
            memory::dims{n, c, 1, 1} : 
            memory::dims{n, c};
        
        // 使用全局平均池化：kernel size = 输入的 H 和 W
        memory::dims kernel = {h, w};
        memory::dims strides = {1, 1};
        memory::dims padding_l = {0, 0};
        memory::dims padding_r = {0, 0};
        memory::dims dilations = {0, 0};
        
        auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
        
        // 根据 keepdims 选择不同的输出格式
        memory::format_tag dst_format = keepdims ? 
            memory::format_tag::nchw : 
            memory::format_tag::nc;
        auto dst_md = memory::desc(dst_dims, memory::data_type::f32, dst_format);
        
        // 使用 pooling_avg_exclude_padding 以获得正确的平均值
        auto pool_pd = pooling_forward::primitive_desc(
            *g_engine,
            prop_kind::forward_inference,
            algorithm::pooling_avg_exclude_padding,
            src_md, dst_md,
            strides, kernel, dilations,
            padding_l, padding_r
        );
        
        auto src_mem = memory(src_md, *g_engine, x_data);
        auto dst_mem = memory(dst_md, *g_engine, y_data);
        
        auto pool_prim = pooling_forward(pool_pd);
        
        std::unordered_map<int, memory> pool_args;
        pool_args.insert({DNNL_ARG_SRC, src_mem});
        pool_args.insert({DNNL_ARG_DST, dst_mem});
        
        pool_prim.execute(*g_stream, pool_args);
        g_stream->wait();
        
        if (keepdims) {
            fprintf(stderr, "[oneDNN] ReduceMean: [%d,%d,%d,%d] -> [%d,%d,1,1] (axes=[2,3], keepdims=true)\n",
                    n, c, h, w, n, c);
        } else {
            fprintf(stderr, "[oneDNN] ReduceMean: [%d,%d,%d,%d] -> [%d,%d] (axes=[2,3], keepdims=false)\n",
                    n, c, h, w, n, c);
        }
    } else {
        fprintf(stderr, "[oneDNN] ReduceMean: Unsupported axis configuration\n");
    }
}

MLIR_ONEDNN_EXPORT void mgpuOneDnnBatchMatMul(
    int batch, int m, int k, int n,
    bool is_a_3d, bool is_b_3d,
    bool transpose_a, bool transpose_b,
    void* a_data, void* b_data, void* c_data
) {
    if (g_engine == nullptr) mgpuOneDnnInit();
    
    // 确定输入A的维度和格式
    memory::dims a_dims;
    memory::format_tag a_format;
    
    if (is_a_3d) {
        a_dims = transpose_a ? memory::dims{batch, k, m} : memory::dims{batch, m, k};
        a_format = transpose_a ? memory::format_tag::acb : memory::format_tag::abc;
    } else {
        // 2D -> 3D: 添加batch维度
        a_dims = transpose_a ? memory::dims{batch, k, m} : memory::dims{batch, m, k};
        a_format = transpose_a ? memory::format_tag::acb : memory::format_tag::abc;
    }
    
    // 确定输入B的维度和格式
    memory::dims b_dims;
    memory::format_tag b_format;
    
    if (is_b_3d) {
        b_dims = transpose_b ? memory::dims{batch, n, k} : memory::dims{batch, k, n};
        b_format = transpose_b ? memory::format_tag::acb : memory::format_tag::abc;
    } else {
        // 2D -> 3D: 添加batch维度
        b_dims = transpose_b ? memory::dims{batch, n, k} : memory::dims{batch, k, n};
        b_format = transpose_b ? memory::format_tag::acb : memory::format_tag::abc;
    }
    
    // 输出总是3D的（带batch维度）
    memory::dims c_dims = {batch, m, n};
    memory::format_tag c_format = memory::format_tag::abc;
    
    auto a_md = memory::desc(a_dims, memory::data_type::f32, a_format);
    auto b_md = memory::desc(b_dims, memory::data_type::f32, b_format);
    auto c_md = memory::desc(c_dims, memory::data_type::f32, c_format);
    
    // 如果输入是2D，需要创建一个view或者broadcast
    memory a_mem, b_mem;
    
    if (!is_a_3d) {
        // 创建2D的memory descriptor
        memory::dims a_dims_2d = transpose_a ? memory::dims{k, m} : memory::dims{m, k};
        memory::format_tag a_format_2d = transpose_a ? memory::format_tag::ba : memory::format_tag::ab;
        auto a_md_2d = memory::desc(a_dims_2d, memory::data_type::f32, a_format_2d);
        
        // 创建2D memory
        auto a_mem_2d = memory(a_md_2d, *g_engine, a_data);
        
        // 需要broadcast到3D - 使用submemory view或者手动处理
        // oneDNN不直接支持broadcast，所以我们使用strides来模拟
        // 对于2D [m,k] broadcast到 [batch, m, k]，batch维的stride应该是0
        
        memory::dims a_strides;
        if (transpose_a) {
            // [k, m] -> [batch, k, m]
            a_strides = {0, m, 1};  // batch维stride=0表示broadcast
        } else {
            // [m, k] -> [batch, m, k]
            a_strides = {0, k, 1};  // batch维stride=0表示broadcast
        }
        
        auto a_md_broadcast = memory::desc(a_dims, memory::data_type::f32, a_strides);
        a_mem = memory(a_md_broadcast, *g_engine, a_data);
    } else {
        a_mem = memory(a_md, *g_engine, a_data);
    }
    
    if (!is_b_3d) {
        memory::dims b_strides;
        if (transpose_b) {
            // [n, k] -> [batch, n, k]
            b_strides = {0, k, 1};
        } else {
            // [k, n] -> [batch, k, n]
            b_strides = {0, n, 1};
        }
        
        auto b_md_broadcast = memory::desc(b_dims, memory::data_type::f32, b_strides);
        b_mem = memory(b_md_broadcast, *g_engine, b_data);
    } else {
        b_mem = memory(b_md, *g_engine, b_data);
    }
    
    auto c_mem = memory(c_md, *g_engine, c_data);
    
    // 创建matmul primitive descriptor
    auto matmul_pd = matmul::primitive_desc(*g_engine, 
        is_a_3d ? a_md : a_mem.get_desc(),
        is_b_3d ? b_md : b_mem.get_desc(),
        c_md);
    
    auto matmul_prim = matmul(matmul_pd);
    
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, a_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
    matmul_args.insert({DNNL_ARG_DST, c_mem});
    
    matmul_prim.execute(*g_stream, matmul_args);
    g_stream->wait();
}

MLIR_ONEDNN_EXPORT void mgpuOneDnnReduceMeanGeneral(
    int ndims,
    const int* input_dims,
    int num_axes,
    const int* axes,
    bool keepdims,  // 注意：这个参数对 oneDNN 层面不影响，因为我们总是使用 keepdims=true
    void* x_data,
    void* y_data
) {
    if (g_engine == nullptr) mgpuOneDnnInit();
    
    // 构建输入维度
    memory::dims src_dims(input_dims, input_dims + ndims);
    
    // 计算输出维度 - 关键：总是使用 keepdims=true 方式（维度数保持一致）
    memory::dims dst_dims;
    std::vector<bool> reduce_mask(ndims, false);
    
    for (int i = 0; i < num_axes; i++) {
        reduce_mask[axes[i]] = true;
    }
    
    // 总是保持维度数一致（即使用户 keepdims=false）
    for (int i = 0; i < ndims; i++) {
        if (reduce_mask[i]) {
            dst_dims.push_back(1);  // 被 reduce 的维度设为 1
        } else {
            dst_dims.push_back(input_dims[i]);
        }
    }
    
    // 根据维度选择合适的format tag
    memory::format_tag src_format, dst_format;
    
    switch (ndims) {
        case 1: src_format = dst_format = memory::format_tag::a; break;
        case 2: src_format = dst_format = memory::format_tag::ab; break;
        case 3: src_format = dst_format = memory::format_tag::abc; break;
        case 4: src_format = dst_format = memory::format_tag::abcd; break;
        case 5: src_format = dst_format = memory::format_tag::abcde; break;
        case 6: src_format = dst_format = memory::format_tag::abcdef; break;
        default: 
            fprintf(stderr, "[oneDNN] Unsupported ndims: %d\n", ndims);
            return;
    }
    
    // 使用具体的format tag创建memory descriptor
    auto src_md = memory::desc(src_dims, memory::data_type::f32, src_format);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, dst_format);
    
    // 创建reduction primitive descriptor
    auto reduce_pd = reduction::primitive_desc(
        *g_engine,
        algorithm::reduction_mean,
        src_md, dst_md,
        0.f, 0.f
    );
    
    // 创建memory对象
    auto src_mem = memory(src_md, *g_engine, x_data);
    auto dst_mem = memory(dst_md, *g_engine, y_data);
    
    // 创建并执行primitive
    auto reduce_prim = reduction(reduce_pd);
    
    std::unordered_map<int, memory> reduce_args;
    reduce_args.insert({DNNL_ARG_SRC, src_mem});
    reduce_args.insert({DNNL_ARG_DST, dst_mem});
    
    reduce_prim.execute(*g_stream, reduce_args);
    g_stream->wait();
}

MLIR_ONEDNN_EXPORT void mgpuOneDnnBatchMatMul4D(
    int dim0, int dim1, int m, int k, int n,
    bool transpose_a, bool transpose_b,
    void* a_data, void* b_data, void* c_data
) {
    if (g_engine == nullptr) mgpuOneDnnInit();
    
    // 物理维度（内存中的实际布局）
    memory::dims a_dims = {dim0, dim1, m, k};
    memory::dims b_dims = {dim0, dim1, k, n};
    memory::dims c_dims = {dim0, dim1, m, n};
    
    // 根据transpose标志选择format（控制数据如何解释）
    auto a_format = transpose_a ? memory::format_tag::abdc : memory::format_tag::abcd;
    auto b_format = transpose_b ? memory::format_tag::abdc : memory::format_tag::abcd;
    auto c_format = memory::format_tag::abcd;
    
    auto a_md = memory::desc(a_dims, memory::data_type::f32, a_format);
    auto b_md = memory::desc(b_dims, memory::data_type::f32, b_format);
    auto c_md = memory::desc(c_dims, memory::data_type::f32, c_format);
    
    auto matmul_pd = matmul::primitive_desc(*g_engine, a_md, b_md, c_md);
    
    auto a_mem = memory(a_md, *g_engine, a_data);
    auto b_mem = memory(b_md, *g_engine, b_data);
    auto c_mem = memory(c_md, *g_engine, c_data);
    
    auto matmul_prim = matmul(matmul_pd);
    
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, a_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
    matmul_args.insert({DNNL_ARG_DST, c_mem});
    
    matmul_prim.execute(*g_stream, matmul_args);
    g_stream->wait();
}

MLIR_ONEDNN_EXPORT void mgpuOneDnnSoftmax(
    int ndims,              // tensor的维度数
    const int* dims,        // shape数组
    int axis,               // softmax的轴（已归一化为正数）
    void* x_data,
    void* y_data
) {
    if (g_engine == nullptr) mgpuOneDnnInit();
    
    // 构建维度
    memory::dims src_dims(dims, dims + ndims);
    memory::dims dst_dims = src_dims;  // softmax输入输出shape相同
    
    // 确定格式标签
    memory::format_tag format;
    switch (ndims) {
        case 1: format = memory::format_tag::a; break;
        case 2: format = memory::format_tag::ab; break;
        case 3: format = memory::format_tag::abc; break;
        case 4: format = memory::format_tag::abcd; break;
        case 5: format = memory::format_tag::abcde; break;
        case 6: format = memory::format_tag::abcdef; break;
        default: 
            fprintf(stderr, "[oneDNN] Softmax: Unsupported ndims=%d\n", ndims);
            return;
    }
    
    auto src_md = memory::desc(src_dims, memory::data_type::f32, format);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, format);
    
    // 创建softmax primitive descriptor
    // 使用 softmax_accurate 获得更高精度
    auto softmax_pd = softmax_forward::primitive_desc(
        *g_engine,
        prop_kind::forward_inference,
        algorithm::softmax_accurate,
        src_md, dst_md,
        axis  // 在指定轴上做softmax
    );
    
    auto src_mem = memory(src_md, *g_engine, x_data);
    auto dst_mem = memory(dst_md, *g_engine, y_data);
    
    auto softmax_prim = softmax_forward(softmax_pd);
    
    std::unordered_map<int, memory> softmax_args;
    softmax_args.insert({DNNL_ARG_SRC, src_mem});
    softmax_args.insert({DNNL_ARG_DST, dst_mem});
    
    softmax_prim.execute(*g_stream, softmax_args);
    g_stream->wait();
    
    // Debug输出
    // fprintf(stderr, "[oneDNN] Softmax: shape=[");
    // for (int i = 0; i < ndims; i++) {
    //     fprintf(stderr, "%d%s", dims[i], i < ndims-1 ? "," : "");
    // }
    // fprintf(stderr, "], axis=%d\n", axis);
}

MLIR_ONEDNN_EXPORT void mgpuOneDnnGemm(
    int m, int k, int n,
    bool transpose_a, bool transpose_b,
    float alpha, float beta,
    void* a_data, void* b_data, void* c_data,  // c_data是bias
    void* y_data
) {
    if (g_engine == nullptr) mgpuOneDnnInit();
    
    // 逻辑维度总是固定的，不随 transpose 改变
    memory::dims a_dims = {m, k};  // A 的逻辑形状总是 [m, k]
    memory::dims b_dims = {k, n};  // B 的逻辑形状总是 [k, n]
    memory::dims y_dims = {m, n};  // Y 的逻辑形状总是 [m, n]
    memory::dims bias_dims = {1, n};
    
    // transpose 通过 format tag 表示
    // ab 表示正常顺序，ba 表示转置
    auto a_format = transpose_a ? memory::format_tag::ba : memory::format_tag::ab;
    auto b_format = transpose_b ? memory::format_tag::ba : memory::format_tag::ab;
    auto y_format = memory::format_tag::ab;
    
    // Bias 使用 stride 实现 broadcast
    memory::dims bias_strides = {0, 1};
    
    auto a_md = memory::desc(a_dims, memory::data_type::f32, a_format);
    auto b_md = memory::desc(b_dims, memory::data_type::f32, b_format);
    auto y_md = memory::desc(y_dims, memory::data_type::f32, y_format);
    auto bias_md = memory::desc(bias_dims, memory::data_type::f32, bias_strides);
    
    // 创建matmul primitive descriptor（支持bias）
    auto matmul_pd = matmul::primitive_desc(*g_engine, a_md, b_md, bias_md, y_md);
    
    auto a_mem = memory(a_md, *g_engine, a_data);
    auto b_mem = memory(b_md, *g_engine, b_data);
    auto y_mem = memory(y_md, *g_engine, y_data);
    auto bias_mem = memory(bias_md, *g_engine, c_data);
    
    auto matmul_prim = matmul(matmul_pd);
    
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, a_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
    matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
    matmul_args.insert({DNNL_ARG_DST, y_mem});
    
    matmul_prim.execute(*g_stream, matmul_args);
    g_stream->wait();
}

MLIR_ONEDNN_EXPORT void mgpuOneDnnReduceSum(
    int ndims,
    const int* input_dims,
    int num_axes,
    const int* axes,
    bool keepdims,  // 同样，这个参数不影响 oneDNN 层面
    void* x_data,
    void* y_data
) {
    if (g_engine == nullptr) mgpuOneDnnInit();
    
    // 构建输入维度
    memory::dims src_dims(input_dims, input_dims + ndims);
    
    // 计算输出维度 - 总是使用 keepdims=true 方式
    memory::dims dst_dims;
    std::vector<bool> reduce_mask(ndims, false);
    
    for (int i = 0; i < num_axes; i++) {
        reduce_mask[axes[i]] = true;
    }
    
    // 总是保持维度数一致
    for (int i = 0; i < ndims; i++) {
        if (reduce_mask[i]) {
            dst_dims.push_back(1);  // 被 reduce 的维度设为 1
        } else {
            dst_dims.push_back(input_dims[i]);
        }
    }
    
    // 根据维度选择合适的format tag
    memory::format_tag src_format, dst_format;
    
    switch (ndims) {
        case 1: src_format = dst_format = memory::format_tag::a; break;
        case 2: src_format = dst_format = memory::format_tag::ab; break;
        case 3: src_format = dst_format = memory::format_tag::abc; break;
        case 4: src_format = dst_format = memory::format_tag::abcd; break;
        case 5: src_format = dst_format = memory::format_tag::abcde; break;
        case 6: src_format = dst_format = memory::format_tag::abcdef; break;
        default: 
            fprintf(stderr, "[oneDNN] Unsupported ndims: %d\n", ndims);
            return;
    }
    
    // 使用具体的format tag创建memory descriptor
    auto src_md = memory::desc(src_dims, memory::data_type::f32, src_format);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, dst_format);
    
    // 创建reduction primitive descriptor
    auto reduce_pd = reduction::primitive_desc(
        *g_engine,
        algorithm::reduction_sum,
        src_md, dst_md,
        0.f, 0.f
    );
    
    // 创建memory对象
    auto src_mem = memory(src_md, *g_engine, x_data);
    auto dst_mem = memory(dst_md, *g_engine, y_data);
    
    // 创建并执行primitive
    auto reduce_prim = reduction(reduce_pd);
    
    std::unordered_map<int, memory> reduce_args;
    reduce_args.insert({DNNL_ARG_SRC, src_mem});
    reduce_args.insert({DNNL_ARG_DST, dst_mem});
    
    reduce_prim.execute(*g_stream, reduce_args);
    g_stream->wait();
}

} // extern "C"