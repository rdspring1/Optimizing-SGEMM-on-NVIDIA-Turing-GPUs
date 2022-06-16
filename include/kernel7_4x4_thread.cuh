#include "helper_fn.cuh"
#include <stdio.h>
#include <stdlib.h>

// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 4x4 micro kernel.
// adopt vetorized load/store
__global__ __launch_bounds__(256) void mysgemm_v7(int M, int N, int K,
                                                  float alpha, float *A,
                                                  float *B, float beta,
                                                  float *C) {
  __shared__ float smem_A[1024];
  __shared__ float smem_B[1024];

  int lda = M, ldb = K, ldc = M;
  int tx = threadIdx.x;

  int row_a = (tx & 15) << 2, col_a = tx >> 4;
  int row_b = (tx & 3) << 2, col_b = tx >> 2;
  int col_c = col_a << 2;

  // the TB size is 64.
  int bx_shift = (blockIdx.x << 6);
  int by_shift = (blockIdx.y << 6);
  A = &A[index(bx_shift, 0, lda)];
  B = &B[index(0, by_shift, ldb)];
  C = &C[index(bx_shift, by_shift, ldc)];

  Array a_vec;
  Array b_vec;
  Array c_vec[4];
  Array c_accum[4];

#pragma unroll
  for (int offset = 0; offset < FACTOR; offset++) {
    c_accum[offset].set(0);
  }

  for (int cta_k = 0; cta_k < K; cta_k += KSL) {
    a_vec = vectorizeLoad(&A[index(row_a, col_a, lda)]);
    b_vec = vectorizeLoad(&B[index(row_b, col_b, ldb)]);

    // smem_A[tx] write 4 float values
    reinterpret_cast<Array *>(smem_A)[tx] = a_vec;

#pragma unroll
    for (int offset = 0; offset < FACTOR; offset++) {
      smem_B[smemFlipIndex(col_b, row_b + offset, 6)] = b_vec[offset];
    }

    A += (lda << 4);
    B += 16;
    __syncthreads();

#pragma unroll
    for (int warp_k = 0; warp_k < KSL; warp_k++) {
      a_vec = vectorizeLoad(&smem_A[smemFlipIndex(row_a, warp_k, 6)]);
      b_vec = vectorizeLoad(&smem_B[smemFlipIndex(col_c, warp_k, 6)]);

#pragma unroll
      for (int offset = 0; offset < FACTOR; offset++) {
        vectorizeScale(c_accum[offset], a_vec, b_vec[offset]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int offset = 0; offset < FACTOR; offset++) {
    c_vec[offset] = vectorizeLoad(&C[index(row_a, col_c + offset, ldc)]);
    axpby(c_accum[offset], alpha, c_accum[offset], beta, c_vec[offset]);
    vectorizeStore(&C(row_a, col_c + offset), c_accum[offset]);
  }
}

void test_mysgemm_v7(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  cudaDeviceSynchronize();
  dim3 blockDim(256);
  dim3 gridDim(ceilDiv(M, 64), ceilDiv(N, 64));
  mysgemm_v7<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();
}
