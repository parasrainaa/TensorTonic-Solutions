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

__inline__ __device__ float warpMax(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}

__inline__ __device__ float max4(float4 v) {
    return fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
}

__global__ void max_kernel(const float* __restrict__ input, float* result, int N) {
    __shared__ float warpMaxes[32];
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int gtid   = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Phase 1a: main body, 16 bytes per load
    const float4* in4 = reinterpret_cast<const float4*>(input);
    int N4 = N >> 2;
    float m = -FLT_MAX;
    for (int i = gtid; i < N4; i += stride)
        m = fmaxf(m, max4(in4[i]));
    for (int j = (N4 << 2) + gtid; j < N; j += stride)
        m = fmaxf(m, input[j]);

    m = warpMax(m);
    if (lane == 0) warpMaxes[warp] = m;
    __syncthreads();
    if (warp == 0) {
        int numWarps = blockDim.x / 32;
        m = (lane < numWarps) ? warpMaxes[lane] : -FLT_MAX;
        m = warpMax(m);
        if (lane == 0) atomicMaxFloat(result, m);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int work = (N + 3) / 4;                   
    int blocks = (work + threads - 1) / threads;
    if (blocks > 160) blocks = 160;
    if (blocks < 1)   blocks = 1;

    float neg_inf = -FLT_MAX;
    cudaMemcpy(result, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    max_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}