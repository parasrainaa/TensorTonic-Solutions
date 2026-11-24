#include <cuda_runtime.h>

__global__ void matrix_add_vec4(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = total / 4;

    if (i < n4) {
        float4 a = reinterpret_cast<const float4*>(A)[i];
        float4 b = reinterpret_cast<const float4*>(B)[i];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<float4*>(C)[i] = c;
    }

    int tail = n4 * 4 + i;
    if (i < 3 && tail < total) {
        C[tail] = A[tail] + B[tail];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    int total = M * N;
    int n4 = total / 4;
    int threads = 256;
    int blocks = (n4 + threads - 1) / threads;
    if (blocks == 0) blocks = 1;   
    matrix_add_vec4<<<blocks, threads>>>(A, B, C, total);
    cudaDeviceSynchronize();
}