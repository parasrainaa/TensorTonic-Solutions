#include <cuda_runtime.h>
#include <float.h>

#define MAX_BLOCKS 160

__device__ float partials[MAX_BLOCKS];     // one slot per block
__device__ unsigned int counter = 0;       // starts at 0; the last block resets it

__global__ void max_kernel(const float* input, float* result, int N) {
    __shared__ float s[256];
    __shared__ bool isLast;
    int tid = threadIdx.x;

    // Phase 1: each block reduces its share of the input (unchanged)
    float m = -FLT_MAX;
    for (int i = blockDim.x * blockIdx.x + tid; i < N; i += blockDim.x * gridDim.x)
        m = fmaxf(input[i], m);
    s[tid] = m;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] = fmaxf(s[tid], s[tid + stride]);
        __syncthreads();
    }

    // Phase 2: write the partial, then take a ticket
    if (tid == 0) {
        partials[blockIdx.x] = s[0];
        __threadfence();                              // make the partial visible to all blocks first
        unsigned int ticket = atomicAdd(&counter, 1); // returns the old value
        isLast = (ticket == gridDim.x - 1);
    }
    __syncthreads();                                  // share isLast with the whole block

    // Phase 3: only the last block to finish reduces the partials
    if (isLast) {
        volatile float* p = partials;                 // force a real read, not a stale cached value
        float v = -FLT_MAX;
        for (int i = tid; i < gridDim.x; i += blockDim.x)
            v = fmaxf(v, p[i]);
        s[tid] = v;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) s[tid] = fmaxf(s[tid], s[tid + stride]);
            __syncthreads();
        }
        if (tid == 0) {
            *result = s[0];
            counter = 0;                              // ready for the next call
        }
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
    if (blocks < 1) blocks = 1;

    max_kernel<<<blocks, threads>>>(input, result, N);
    // Keep this line only if Version B failed:
    // cudaDeviceSynchronize();
}