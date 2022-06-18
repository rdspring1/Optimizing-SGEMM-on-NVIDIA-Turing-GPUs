#include "helper_fn.cuh"
#include <stdio.h>
#include <stdlib.h>

// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 8x8 micro kernel.
// adopt vetorized load/store
// 4x8 warp-level tiling for output matrix C
__global__ __launch_bounds__(256) void mysgemm_v9(int M, int N, int K,
                                                  float alpha, float *A,
                                                  float *B, float beta,
                                                  float *C) {
  __shared__ float smem_A[1024];
  __shared__ float smem_B[1024];

  int lda = M, ldb = K, ldc = M;
  int tx = threadIdx.x;

  int warp_id = tx >> 5;
  int lane_id = tx & 31;
  int warp_row = warp_id & 3, warp_col = warp_id >> 2;
  int row_w = lane_id & 3, col_w = lane_id >> 2;
  int row_c = (warp_row << 5) + (row_w << 3);
  int col_c = (warp_col << 6) + (col_w << 3);
  int row_a = (tx & 31) << 2, col_a = tx >> 5;
  int row_b = (tx & 1) << 2, col_b = tx >> 1;

  // The TB size is 128, so shift is 7.
  int shift = 7;
  int bx_shift = (blockIdx.x << shift);
  int by_shift = (blockIdx.y << shift);
  A = &A[index(bx_shift, 0, lda)];
  B = &B[index(0, by_shift, ldb)];
  C = &C[index(bx_shift, by_shift, ldc)];

  Array a_vec[2];
  Array b_vec[2];
  Array c_vec[16];
  Array c_accum[16];

#pragma unroll
  for (int offset = 0; offset < 16; offset++) {
    c_accum[offset].set(0);
  }

  for (int cta_k = 0; cta_k < K; cta_k += KSXL) {
    /*packing A and B into shared memory*/
    a_vec[0] = vectorizeLoad(&A[index(row_a, col_a, lda)]);
    b_vec[0] = vectorizeLoad(&B[index(row_b, col_b, ldb)]);

    // smem_A[tx] write 4 float values
    reinterpret_cast<Array *>(smem_A)[tx] = a_vec[0];

#pragma unroll
    for (int offset = 0; offset < FACTOR; offset++) {
      smem_B[smemFlipIndex(col_b, row_b + offset, shift)] = b_vec[0][offset];
    }

    A += (lda << 3);
    B += 8;
    __syncthreads();
#pragma unroll
    for (int warp_k = 0; warp_k < KSXL; warp_k++) {
      a_vec[0] = vectorizeLoad(&smem_A[smemFlipIndex(row_c, warp_k, shift)]);
      a_vec[1] =
          vectorizeLoad(&smem_A[smemFlipIndex(row_c + 4, warp_k, shift)]);
      b_vec[0] = vectorizeLoad(&smem_B[smemFlipIndex(col_c, warp_k, shift)]);
      b_vec[1] =
          vectorizeLoad(&smem_B[smemFlipIndex(col_c + 4, warp_k, shift)]);

      // each thread handles 8x8 tile of C
      // vectorization => 128-bit memop / 4 bytes (float) = 4 floats
      // 64 elements / 4 = 16 vectorized structures
      int c_idx = 0;
#pragma unroll
      for (int idx = 0; idx < 2; idx++) {
#pragma unroll
        for (int jdx = 0; jdx < FACTOR; jdx++) {
#pragma unroll
          for (int kdx = 0; kdx < 2; kdx++) {
            vectorizeScale(c_accum[c_idx], a_vec[kdx], b_vec[idx][jdx]);
            ++c_idx;
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int offset = 0; offset < 16; offset++) {
    int row_offset = (offset % 2) << 2;
    int col_offset = (offset / 2);
    c_vec[offset] =
        vectorizeLoad(&C[index(row_c + row_offset, col_c + col_offset, ldc)]);
    axpby(c_accum[offset], alpha, c_accum[offset], beta, c_vec[offset]);
    vectorizeStore(&C(row_c + row_offset, col_c + col_offset), c_accum[offset]);
  }
}

void test_mysgemm_v9(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  cudaDeviceSynchronize();
  dim3 blockDim(256);
  dim3 gridDim(ceilDiv(M, 128), ceilDiv(N, 128));
  mysgemm_v9<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();
}
