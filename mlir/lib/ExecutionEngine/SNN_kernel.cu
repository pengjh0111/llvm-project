// lif_fused_kernel.cu
#include <cuda_runtime.h>

__global__ void fused_lif_update_kernel(
    float* membrane_potential,
    float* input_current, 
    float* threshold,
    float* reset_value,
    float leak_factor,
    float* new_membrane,
    bool* spike_output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Read current values
        float v = membrane_potential[idx];
        bool prev_spike = spike_output[idx]; // Read previous spike state
        
        // LIF neuron update (fused_sub_sub_div_add_ge)
        v = v - leak_factor;  // Apply leak
        
        // Apply reset only if there was a spike in the previous step
        if (prev_spike) {
            v = reset_value[idx];
        }
        
        // Add input current
        v = v + input_current[idx];
        
        // Check threshold crossing
        bool spike = v >= threshold[idx];
        
        // Write results
        new_membrane[idx] = spike ? reset_value[idx] : v;  // Reset if spiked
        spike_output[idx] = spike;
    }
}

// C wrapper
extern "C" {
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
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    fused_lif_update_kernel<<<blocks, threads, 0, stream>>>(
        membrane_potential, input_current, threshold, 
        reset_value, leak_factor, new_membrane, 
        spike_output, size
    );
}
}