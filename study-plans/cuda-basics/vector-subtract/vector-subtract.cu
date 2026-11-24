#include <cuda_runtime.h>

__global__ void vector_sub(const float* A, const float* B, float* C, int N) {
    // Store N element-wise results in C; this kernel returns no value.
    int tid = threadIdx.x;
    int i = blockDim.x*blockIdx.x + tid;
    int n = N/4;
    
    if(i < n){
        float4 a = reinterpret_cast<const float4*>(A)[i];
        float4 b = reinterpret_cast<const float4*>(B)[i];
        float4 c;
        c.x = a.x - b.x;
        c.y = a.y - b.y;
        c.z = a.z - b.z;
        c.w = a.w - b.w;
        reinterpret_cast<float4*>(C)[i] = c;
    }
    int tail = n * 4 + i;
    if (i < 3 && tail < N) {
        C[tail] = A[tail] - B[tail];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threads = 256;
    int blocks = N/4 +1;
    vector_sub<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}