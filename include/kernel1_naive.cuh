#include <stdio.h>
#include <stdlib.h>

#include "helper_fn.cuh"

// C (M, N) = A (M, K) * B (K, N)

// naive version
__global__ __launch_bounds__(1024) void mysgemm_v1(int M, int N, int K,
                                                   float alpha, float *A,
                                                   float *B, float beta,
                                                   float *C) {
  int lda = M, ldb = K, ldc = M;
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx_shift = (blockIdx.x << 5);
  int by_shift = (blockIdx.y << 5);

  A = &A[index(bx_shift, 0, lda)];
  B = &B[index(0, by_shift, ldb)];
  C = &C[index(bx_shift, by_shift, ldc)];

  float c_accum = 0.;
  for (int kdx = 0; kdx < K; kdx++) {
    c_accum += A[index(tx, kdx, lda)] * B[index(kdx, ty, ldb)];
  }
  C[index(tx, ty, ldc)] = alpha * c_accum + beta * C[index(tx, ty, ldc)];
}

void test_mysgemm_v1(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  cudaDeviceSynchronize();
  dim3 blockDim(32, 32);
  dim3 gridDim(ceilDiv(M, 32), ceilDiv(N, 32));
  mysgemm_v1<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();
}
