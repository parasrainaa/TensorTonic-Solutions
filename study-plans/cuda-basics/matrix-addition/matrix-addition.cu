#include <cuda_runtime.h>

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int M, int N) {
    // tid = threadIdx.x;
    float sum = 0.f;
    for(int i = 0;i < M;i++){
        for(int j = 0;j < N;j++){
            sum = A[i*N + j] + B[i*N + j];
            C[i*N + j] = sum;
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matrix_add_kernel<<<blocks, threads>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}
