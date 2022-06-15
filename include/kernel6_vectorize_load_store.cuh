#include <stdio.h>
#include <stdlib.h>

#include "helper_fn.cuh"

#define MS 32
#define NS 32
#define KS 32

//! cache blocking version, without register-level data re-used
//! with memory coelascing on shared memory
//! 4x1 micro kernel - compute more elements of C per thread
//! vectorize load/store
__global__ __launch_bounds__(256) void mysgemm_v6(int M, int N, int K,
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

  Array a_vec;
  Array b_vec;
  Array c_vec;
  Array c_accum;
  c_accum.set(0);

  float b_reg;
  for (int cta_k = 0; cta_k < K; cta_k += KS) {
    a_vec = vectorizeLoad(&A[index(row, col, lda)]);
    b_vec = vectorizeLoad(&B[index(row, col, ldb)]);

    // smem_A[tx] write 4 float values
    reinterpret_cast<Array *>(smem_A)[tx] = a_vec;

#pragma unroll
    for (int offset = 0; offset < FACTOR; offset++) {
      smem_B[smemIndex(col, row + offset, 5)] = b_vec[offset];
    }

    A += (lda << 5);
    B += 32;
    __syncthreads();

#pragma unroll
    for (int warp_k = 0; warp_k < KS; warp_k++) {
      b_reg = smem_B[smemIndex(col, warp_k, 5)];

// smemA(warp_k, row1) read 4 float values
#pragma unroll FACTOR
      for (int offset = 0; offset < FACTOR; offset++) {
        c_accum[offset] +=
            smem_A[smemFlipIndex(row + offset, warp_k, 5)] * b_reg;
      }
    }
    __syncthreads();
  }

  c_vec = vectorizeLoad(&C[index(row, col, ldc)]);
#pragma unroll FACTOR
  for (int offset = 0; offset < FACTOR; offset++) {
    c_accum[offset] = alpha * c_accum[offset] + beta * c_vec[offset];
  }
  vectorizeStore(&C(row, col), c_accum);
}

void test_mysgemm_v6(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  cudaDeviceSynchronize();
  dim3 blockDim(256);
  dim3 gridDim(ceilDiv(M, 32), ceilDiv(N, 32));
  mysgemm_v6<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();
}
