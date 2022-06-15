#pragma once

#define CEIL_DIV(m, n) ((m) + (n)-1) / (n)

#define A(i, j) A[(i) + (j)*lda]
#define B(i, j) B[(i) + (j)*ldb]
#define C(i, j) C[(i) + (j)*ldc]

#define smemA(i, j) smem_A[((i) << 5) + (j)]
#define smemB(i, j) smem_B[((i) << 5) + (j)]

#define vectorizeLoad(v1, addr) v1 = *((float4 *)(addr));
#define vectorizeStore(addr, v1) *((float4 *)(addr)) = v1;
