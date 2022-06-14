#include <stdio.h>
#include <stdlib.h>
#define A(i, j) A[(i) + (j)*lda]
#define B(i, j) B[(i) + (j)*ldb]
#define C(i, j) C[(i) + (j)*ldc]
#define smemA(i, j) smem_A[((i) << 5) + (j)]
#define smemB(i, j) smem_B[((i) << 5) + (j)]
#define MS 32
#define NS 32
#define KS 32
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

  A = &A(bx_shift, 0);
  B = &B(0, by_shift);
  C = &C(bx_shift, by_shift);

  float c_accum = 0.;
  for (int cta_k = 0; cta_k < K; cta_k += KS) {
    smemA(tx, ty) = A(tx, ty);
    smemB(ty, tx) = B(tx, ty);
    A += (lda << 5);
    B += 32;
    __syncthreads();
    for (int warp_k = 0; warp_k < KS; warp_k++) {
      c_accum += smemA(tx, warp_k) * smemB(ty, warp_k);
    }
    __syncthreads();
  }
  C(tx, ty) = alpha * c_accum + beta * C(tx, ty);
}
