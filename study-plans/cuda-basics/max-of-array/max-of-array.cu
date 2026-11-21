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

// One thread sets the starting value. Same stream => finishes before max_kernel starts.
__global__ void init_kernel(float* result) {
    *result = -FLT_MAX;
}

__global__ void max_kernel(const float* input, float* result, int N) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    float m = -FLT_MAX;
    for (int i = blockDim.x * blockIdx.x + tid; i < N; i += blockDim.x * gridDim.x)
        m = fmaxf(input[i], m);
    s[tid] = m;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] = fmaxf(s[tid], s[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) atomicMaxFloat(result, s[0]);
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 160) blocks = 160;   // cap at one full wave
    if (blocks < 1)   blocks = 1;     // guard against N == 0

    init_kernel<<<1, 1>>>(result);
    max_kernel<<<blocks, threads>>>(input, result, N);
    // cudaDeviceSynchronize();
}