#include <stdio.h>
#include <stdlib.h>
#define A(i, j) A[(i) + (j)*lda]
#define B(i, j) B[(i) + (j)*ldb]
#define C(i, j) C[(i) + (j)*ldc]

// naive version
__global__ __launch_bounds__(1024) void mysgemm_v1(int M, int N, int K,
                                                   float alpha, float *A,
                                                   float *B, float beta,
                                                   float *C) {
  int lda = M, ldb = K, ldc = M;
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx_shift = (blockIdx.x << 5);
  int by_shift = (blockIdx.y << 5);

  A = &A(bx_shift, 0);
  B = &B(0, by_shift);
  C = &C(bx_shift, by_shift);

  float c_accum = 0.;
  for (int kdx = 0; kdx < K; kdx++) {
    c_accum += A(tx, kdx) * B(kdx, ty);
  }
  C(tx, ty) = alpha * c_accum + beta * C(tx, ty);
}
