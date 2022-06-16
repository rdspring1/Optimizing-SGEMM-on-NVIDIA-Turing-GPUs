#include <stdio.h>
#include <stdlib.h>

#include "helper_fn.cuh"

#define MS 32
#define NS 32
#define KS 32

constexpr int FACTOR = 4;

//! cache blocking version, without register-level data re-use with memory
//! coelascing on shared memory.
//! 4x1 micro kernel - compute more elements of C per thread
__global__ __launch_bounds__(256) void mysgemm_v4(int M, int N, int K,
                                                  float alpha, float *A,
                                                  float *B, float beta,
                                                  float *C) {
  __shared__ float smem_A[MS * KS];
  __shared__ float smem_B[KS * NS];

  int lda = M, ldb = K, ldc = M;
  int tx = threadIdx.x;

  int row = (tx & 7) << 2;
  int col = tx >> 3;

  int bx_shift = (blockIdx.x << 5);
  int by_shift = (blockIdx.y << 5);
  A = &A[index(bx_shift, 0, lda)];
  B = &B[index(0, by_shift, ldb)];
  C = &C[index(bx_shift, by_shift, ldc)];

  float c_accum[4] = {0., 0., 0., 0.};
  float b_reg;
  for (int k_count = 0; k_count < K; k_count += KS) {
#pragma unroll FACTOR
    for (int offset = 0; offset < FACTOR; offset++) {
      smem_A[smemIndex(row + offset, col, 5)] =
          A[index(row + offset, col, lda)];
    }

#pragma unroll FACTOR
    for (int offset = 0; offset < FACTOR; offset++) {
      smem_B[smemIndex(col, row + offset, 5)] =
          B[index(row + offset, col, ldb)];
    }

    A += (lda << 5);
    B += 32;
    __syncthreads();

#pragma unroll
    for (int warp_k = 0; warp_k < KS; warp_k++) {
      b_reg = smem_B[smemIndex(col, warp_k, 5)];

#pragma unroll FACTOR
      for (int offset = 0; offset < FACTOR; offset++) {
        c_accum[offset] += smem_A[smemIndex(row + offset, warp_k, 5)] * b_reg;
      }
    }
    __syncthreads();
  }

#pragma unroll FACTOR
  for (int offset = 0; offset < FACTOR; offset++) {
    C[index(row + offset, col, ldc)] =
        alpha * c_accum[offset] + beta * C[index(row + offset, col, ldc)];
  }
}

void test_mysgemm_v4(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  cudaDeviceSynchronize();
  dim3 blockDim(256);
  dim3 gridDim(ceilDiv(M, 32), ceilDiv(N, 32));
  mysgemm_v4<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();
}
