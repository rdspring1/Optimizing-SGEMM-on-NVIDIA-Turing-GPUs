#include <stdio.h>
#include <stdlib.h>

#include "helper_fn.cuh"

// cache blocking version, without register-level data re-use
__global__ __launch_bounds__(1024) void mysgemm_v2(int M, int N, int K,
                                                   float alpha, float *A,
                                                   float *B, float beta,
                                                   float *C) {
  __shared__ float smem_A[MS * KS];
  __shared__ float smem_B[KS * NS];

  int lda = M, ldb = K, ldc = M;
  int tx = threadIdx.x, ty = threadIdx.y;

  int bx_shift = (blockIdx.x << 5);
  int by_shift = (blockIdx.y << 5);
  A = &A[index(bx_shift, 0, lda)];
  B = &B[index(0, by_shift, ldb)];
  C = &C[index(bx_shift, by_shift, ldc)];

  float c_accum = 0.;
  for (int cta_k = 0; cta_k < K; cta_k += KS) {
    smem_A[smemIndex(tx, ty, 5)] = A[index(tx, ty, lda)];
    smem_B[smemIndex(ty, tx, 5)] = B[index(tx, ty, ldb)];
    A += (lda << 5);
    B += 32;
    __syncthreads();

    for (int warp_k = 0; warp_k < KS; warp_k++) {
      c_accum +=
          smem_A[smemIndex(tx, warp_k, 5)] * smem_B[smemIndex(ty, warp_k, 5)];
    }
    __syncthreads();
  }
  C[index(tx, ty, ldc)] = alpha * c_accum + beta * C[index(tx, ty, ldc)];
}

void test_mysgemm_v2(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  cudaDeviceSynchronize();
  dim3 blockDim(32, 32);
  dim3 gridDim(ceilDiv(M, 32), ceilDiv(N, 32));
  mysgemm_v2<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();
}
