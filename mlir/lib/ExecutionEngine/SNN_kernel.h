// lif_fused_kernel.h
#ifndef LIF_FUSED_KERNEL_H
#define LIF_FUSED_KERNEL_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void launchFusedLIFUpdate(
    float* membrane_potential,
    float* input_current, 
    float* threshold,
    float* reset_value,
    float leak_factor,
    float* new_membrane,
    bool* spike_output,
    int size,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // LIF_FUSED_KERNEL_H