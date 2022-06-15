#include <stdio.h>
#include <stdlib.h>

#include "helper_macros.cuh"

#define MS 32
#define NS 32
#define KS 32

// Column-Major Order
// Row = idx & (Height-1) - Mod
// Col = idx >> log2(Height) - Division

// cache blocking version, without register-level data re-use
__global__ __launch_bounds__(1024) void mysgemm_v3(int M, int N, int K,
                                                   float alpha, float *A,
                                                   float *B, float beta,
                                                   float *C) {
  __shared__ float smem_A[MS * KS];
  __shared__ float smem_B[KS * NS];

  int lda = M, ldb = K, ldc = M;
  int tx = threadIdx.x;
  int row = tx & 31, col = tx >> 5;
  int bx_shift = (blockIdx.x << 5);
  int by_shift = (blockIdx.y << 5);

  A = &A(bx_shift, 0);
  B = &B(0, by_shift);
  C = &C(bx_shift, by_shift);

  float c_accum = 0.;
  for (int cta_k = 0; cta_k < K; cta_k += KS) {
    smemA(row, col) = A(row, col);
    smemB(col, row) = B(row, col);

    A += (lda << 5);
    B += 32;
    __syncthreads();

    for (int warp_k = 0; warp_k < KS; warp_k++) {
      c_accum += smemA(row, warp_k) * smemB(col, warp_k);
    }
    __syncthreads();
  }
  C(row, col) = alpha * c_accum + beta * C(row, col);
}

void test_mysgemm_v3(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  cudaDeviceSynchronize();
  dim3 blockDim(1024);
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  mysgemm_v3<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();
}
