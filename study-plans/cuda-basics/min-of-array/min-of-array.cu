#include <cuda_runtime.h>
#include <float.h>

__global__ void init_result(float* result) {
    result[0] = FLT_MAX;
}
__device__ float atomicMinFloat(float* addr, float val) {
    int* a = (int*)addr;
    int old = *a, assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(a, assumed, __float_as_int(val));
        if (old == assumed) break;
    }
    return __int_as_float(old);
}

__global__ void min_kernel(const float* input, float* result, int N) {
    __shared__ float s[256];
    int tid = threadIdx.x;

    float m = FLT_MAX;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x)
        m = fminf(m, input[i]);
    s[tid] = m;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] = fminf(s[tid], s[tid + stride]);
        __syncthreads();
    }

    if (tid == 0) atomicMinFloat(result, s[0]);
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    init_result<<<1, 1>>>(result);
    min_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}
