#include <stdio.h>
#include <stdlib.h>

#include "helper_macros.cuh"

#define MS 32
#define NS 32
#define KS 32

//! cache blocking version, without register-level data re-use with memory
//! coelascing on shared memory.
//!4x1 micro kernel - compute more elements of C per thread
__global__ __launch_bounds__(256) void mysgemm_v4(int M, int N, int K,
                                                  float alpha, float *A,
                                                  float *B, float beta,
                                                  float *C) {
  __shared__ float smem_A[MS * KS];
  __shared__ float smem_B[KS * NS];

  int lda = M, ldb = K, ldc = M;
  int tx = threadIdx.x;

  int row1 = (tx & 7) << 2;
  int row2 = row1 + 1;
  int row3 = row1 + 2;
  int row4 = row1 + 3;
  int col = tx >> 3;

  int bx_shift = (blockIdx.x << 5);
  int by_shift = (blockIdx.y << 5);
  A = &A(bx_shift, 0);
  B = &B(0, by_shift);
  C = &C(bx_shift, by_shift);

  float c_accum[4] = {0., 0., 0., 0.};
  float b_reg;
  for (int k_count = 0; k_count < K; k_count += KS) {
    smemA(row1, col) = A(row1, col);
    smemA(row2, col) = A(row2, col);
    smemA(row3, col) = A(row3, col);
    smemA(row4, col) = A(row4, col);

    smemB(col, row1) = B(row1, col);
    smemB(col, row2) = B(row2, col);
    smemB(col, row3) = B(row3, col);
    smemB(col, row4) = B(row4, col);

    A += (lda << 5);
    B += 32;
    __syncthreads();

#pragma unroll
    for (int warp_k = 0; warp_k < KS; warp_k++) {
      b_reg = smemB(col, warp_k);
      c_accum[0] += smemA(row1, warp_k) * b_reg;
      c_accum[1] += smemA(row2, warp_k) * b_reg;
      c_accum[2] += smemA(row3, warp_k) * b_reg;
      c_accum[3] += smemA(row4, warp_k) * b_reg;
    }
    __syncthreads();
  }
  C(row1, col) = alpha * c_accum[0] + beta * C(row1, col);
  C(row2, col) = alpha * c_accum[1] + beta * C(row2, col);
  C(row3, col) = alpha * c_accum[2] + beta * C(row3, col);
  C(row4, col) = alpha * c_accum[3] + beta * C(row4, col);
}

void test_mysgemm_v4(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  cudaDeviceSynchronize();
  dim3 blockDim(256);
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  mysgemm_v4<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();
}
