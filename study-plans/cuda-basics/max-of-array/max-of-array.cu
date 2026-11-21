#include <cuda_runtime.h>
#include <float.h>


__device__ float atomicMaxFloat(float* addr, float val) {
    int* a = (int*)addr;
    int old = *a, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(a, assumed, __float_as_int(val));
        if (old == assumed) break;
    }
    return __int_as_float(old);
}
__global__ void max_kernel(const float* input, float* result, int N) {
    // Write code here
    __shared__ float s[512];
    int tid = threadIdx.x;
    float m = -FLT_MAX;
    for(int i = blockDim.x * blockIdx.x + tid; i < N; i += blockDim.x * gridDim.x){
        m = fmaxf(input[i],m);
    }
    s[tid] = m;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] = fmaxf(s[tid], s[tid + stride]);
        __syncthreads();
    }

    if (tid == 0) atomicMaxFloat(result, s[0]);
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 512;
    int blocks = (N + threads - 1) / threads;
    float neg_inf = -FLT_MAX;
    cudaMemcpy(result, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    max_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}