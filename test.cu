#include "utils.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define MYSGEMM mysgemm_naive // select the kernel here

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Please select a kernel (range 0 - 11).\t"
           "0 is for NVIDIA cuBLAS).\n");
    exit(-1);
  }

  int SIZE[24];
  for (int i = 0; i < 24; i++) {
    SIZE[i] = (i + 1) << 8;
  }

  int kernel_num = atoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 11) {
    printf("Please enter a valid kernel number (0-11).\n");
    exit(-2);
  }
  int m, n, k, max_size;
  int n_count, N = 10, upper_limit;
  if (kernel_num <= 4 && kernel_num != 0) {
    upper_limit = 8;
  } else {
    upper_limit = (sizeof(SIZE) / sizeof(int));
  }
  max_size = SIZE[upper_limit - 1];
  float *A = NULL, *B = NULL, *C = NULL, *C_ref = NULL;     // host matrices
  float *dA = NULL, *dB = NULL, *dC = NULL, *dC_ref = NULL; // device matrices
  float alpha = 1.0, beta = 0.; // two arbitary input parameters

  float elapsed_time;
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  A = (float *)malloc(sizeof(float) * max_size * max_size);
  B = (float *)malloc(sizeof(float) * max_size * max_size);
  C = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  copy_matrix(C, C_ref, max_size * max_size);

  CUDA_CALLER(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
  CUDA_CALLER(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
  CUDA_CALLER(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
  CUDA_CALLER(
      cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));
  CUDA_CALLER(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                         cudaMemcpyHostToDevice));
  CUDA_CALLER(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                         cudaMemcpyHostToDevice));
  CUDA_CALLER(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                         cudaMemcpyHostToDevice));
  CUDA_CALLER(cudaMemcpy(dC_ref, C_ref, sizeof(float) * max_size * max_size,
                         cudaMemcpyHostToDevice));

  for (int i_count = 0; i_count < upper_limit; i_count++) {
    m = n = k = SIZE[i_count];
    printf("\nM=N=K=%d:\n", m);
    if (kernel_num != 0) { // not cuBLAS
      cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA,
                  m, dB, k, &beta, dC_ref, m);
      test_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC);
      cudaDeviceSynchronize();
      cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      if (!verify_matrix(C_ref, C, m * n)) {
        printf("Failed to pass the correctness verification against NVIDIA "
               "cuBLAS. Exited.\n");
        exit(-3);
      }
    }

    cudaEventRecord(beg);
    if (kernel_num != 0) {
      for (n_count = 0; n_count < N; n_count++) {
        test_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC);
      }
    } else {
      for (n_count = 0; n_count < N; n_count++) {
        test_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
                    cublas_handle);
      }
    }

    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.;

    printf("Average elasped time: %f second, performance: %f GFLOPS.\n",
           elapsed_time / N, 2. * 1e-9 * N * m * n * k / elapsed_time);
    fflush(stdout);
    copy_matrix(C_ref, C,
                m * n); // sync C with cuBLAS to prepare for the next run
  }
  cudaDeviceSynchronize();

  free(A);
  free(B);
  free(C);
  free(C_ref);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  cudaDeviceSynchronize();
  return 0;
}
