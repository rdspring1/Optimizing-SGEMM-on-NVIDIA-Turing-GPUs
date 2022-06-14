#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define CUDA_CALLER(call)                                                      \
  do {                                                                         \
    cudaError_t cuda_ret = (call);                                             \
    if (cuda_ret != cudaSuccess) {                                             \
      printf("CUDA Error at line %d in file %s\n", __LINE__, __FILE__);        \
      printf("  Error message: %s\n", cudaGetErrorString(cuda_ret));           \
      printf("  In the function call %s\n", #call);                            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define CUDA_KERNEL_CALLER(...)                                                \
  do {                                                                         \
    if (cudaPeekAtLastError() != cudaSuccess) {                                \
      printf("A CUDA error occurred prior to the kernel call %s at line %d\n", \
             #__VA_ARGS__, __LINE__);                                          \
      exit(1);                                                                 \
    }                                                                          \
    __VA_ARGS__;                                                               \
    cudaError_t cuda_ret = cudaPeekAtLastError();                              \
    if (cuda_ret != cudaSuccess) {                                             \
      printf("CUDA Error at line %d in file %s\n", __LINE__, __FILE__);        \
      printf("  Error message: %s\n", cudaGetErrorString(cuda_ret));           \
      printf("  In the kernel call %s\n", #__VA_ARGS__);                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

void print_matrix(const float *A, int m, int n);
void randomize_matrix(float *mat, int N);
double get_sec();
bool verify_matrix(float *mat1, float *mat2, int n);
void copy_matrix(float *src, float *dest, int n);
void test_kernel(int kernel_num, int m, int n, int k, float alpha, float *A,
                 float *B, float beta, float *C, cublasHandle_t err = nullptr);
