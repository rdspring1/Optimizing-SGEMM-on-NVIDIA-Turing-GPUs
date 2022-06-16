#include <stdio.h>
#include <stdlib.h>

#include "helper_fn.cuh"

#define MS 32
#define NS 32
#define KS 32

// Column-Major Order
// Row = idx & (Height-1) - Mod
// Col = idx >> log2(Height) - Division

// cache blocking version, without register-level data re-use
__global__ __launch_bounds__(1024) void mysgemm_v4(int M, int N, int K,
                                                   float alpha, float *A,
                                                   float *B, float beta,
                                                   float *C) {
  __shared__ float smem_A[MS * KS];
  __shared__ float smem_B[KS * NS];

  int lda = M, ldb = K, ldc = M;
  int tx = threadIdx.x;
  int row = tx & 31;
  int col = tx >> 5;

  int bx_shift = (blockIdx.x << 5);
  int by_shift = (blockIdx.y << 5);
  A = &A[index(bx_shift, 0, lda)];
  B = &B[index(0, by_shift, ldb)];
  C = &C[index(bx_shift, by_shift, ldc)];

  float c_accum = 0.;
  for (int cta_k = 0; cta_k < K; cta_k += KS) {
    smem_A[smemFlipIndex(row, col, 5)] = A[index(row, col, lda)];
    smem_B[smemFlipIndex(col, row, 5)] = B[index(row, col, ldb)];

    A += (lda << 5);
    B += 32;
    __syncthreads();

    for (int warp_k = 0; warp_k < KS; warp_k++) {
      c_accum += smem_A[smemFlipIndex(row, warp_k, 5)] *
                 smem_B[smemFlipIndex(col, warp_k, 5)];
    }
    __syncthreads();
  }
  C[index(row, col, ldc)] = alpha * c_accum + beta * C[index(row, col, ldc)];
}

void test_mysgemm_v4(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  cudaDeviceSynchronize();
  dim3 blockDim(1024);
  dim3 gridDim(ceilDiv(M, 32), ceilDiv(N, 32));
  mysgemm_v4<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();
}
