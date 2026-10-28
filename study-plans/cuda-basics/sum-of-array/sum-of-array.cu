#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* result, int N) {
    // Write code here
    __shared__ float partial_sum [256];
    int tid = threadIdx.x;
    partial_sum[tid] = 0.0f;
    for (int i = blockDim.x * blockIdx.x + tid; i < N;i += blockDim.x * gridDim.x){
        partial_sum[tid] += input[i];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial_sum[tid] +=partial_sum[tid+stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, partial_sum[0]);
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = 160;
    cudaMemset(result, 0, sizeof(float));
    sum_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}
