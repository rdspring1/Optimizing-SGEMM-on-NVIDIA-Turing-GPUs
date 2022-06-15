#include <stdio.h>
#include <stdlib.h>

#include "helper_macros.cuh"

#define MS 32
#define NS 32
#define KS 32

//! cache blocking version, without register-level data re-used
//! with memory coelascing on shared memory
//! 4x1 micro kernel - compute more elements of C per thread
//! vectorize load/store
__global__ __launch_bounds__(256) void mysgemm_v5(int M, int N, int K,
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

  float4 a_vec, b_vec, c_vec, c_accum;
  c_accum.x = 0., c_accum.y = 0., c_accum.z = 0., c_accum.w = 0.;
  float b_reg;
  for (int cta_k = 0; cta_k < K; cta_k += KS) {
    a_vec = *((float4 *)(&A(row1, col)));
    b_vec = *((float4 *)(&B(row1, col)));
    ((float4 *)smem_A)[tx] = a_vec;

    smemB(col, row1) = b_vec.x;
    smemB(col, row2) = b_vec.y;
    smemB(col, row3) = b_vec.z;
    smemB(col, row4) = b_vec.w;

    A += (lda << 5);
    B += 32;
    __syncthreads();

#pragma unroll
    for (int warp_k = 0; warp_k < KS; warp_k++) {
      b_reg = smemB(col, warp_k);
      c_accum.x += smemA(warp_k, row1) * b_reg;
      c_accum.y += smemA(warp_k, row2) * b_reg;
      c_accum.z += smemA(warp_k, row3) * b_reg;
      c_accum.w += smemA(warp_k, row4) * b_reg;
    }
    __syncthreads();
  }

  c_vec = *((float4 *)(&C(row1, col)));
  c_accum.x = alpha * c_accum.x + beta * c_vec.x;
  c_accum.y = alpha * c_accum.y + beta * c_vec.y;
  c_accum.z = alpha * c_accum.z + beta * c_vec.z;
  c_accum.w = alpha * c_accum.w + beta * c_vec.w;
  *(float4 *)(&(C(row1, col))) = c_accum;
}

void test_mysgemm_v5(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  cudaDeviceSynchronize();
  dim3 blockDim(256);
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  mysgemm_v5<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();
}
