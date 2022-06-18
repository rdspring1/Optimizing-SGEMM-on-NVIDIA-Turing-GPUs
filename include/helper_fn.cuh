#pragma once

constexpr int FACTOR = 4;

constexpr int MS = 32;
constexpr int NS = 32;
constexpr int KS = 32;

constexpr int MSL = 64;
constexpr int NSL = 64;
constexpr int KSL = 16;

constexpr int MSXL = 8;
constexpr int NSXL = 8;
constexpr int KSXL = 8;

int ceilDiv(int m, int n) { return (m + n - 1) / n; }

__inline__ __device__ int index(int i, int j, int stride) {
  return i + j * stride;
}

__inline__ __device__ int smemIndex(int i, int j, int shift) {
  return (i << shift) + j;
}

__inline__ __device__ int smemFlipIndex(int i, int j, int shift) {
  return (j << shift) + i;
}

// aligned register array for vectorized load/store
struct alignas(sizeof(float) * 4) Array {
  float array[4];

  __device__ void set(float v) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      array[i] = v;
    }
  }

  __device__ float &operator[](const unsigned int i) { return array[i]; }
};

__inline__ __device__ Array vectorizeLoad(float *addr) {
  return reinterpret_cast<Array *>(addr)[0];
}

__inline__ __device__ void vectorizeStore(float *addr, const Array &value) {
  reinterpret_cast<Array *>(addr)[0] = value;
}

// v1 += v2 * s3, vector scaling
__inline__ __device__ void vectorizeScale(Array &v1, Array &v2, float scalar) {
  v1[0] += v2[0] * scalar;
  v1[1] += v2[1] * scalar;
  v1[2] += v2[2] * scalar;
  v1[3] += v2[3] * scalar;
}

// v1 = alpha * v2 + beta * v3, simd fma
__inline__ __device__ void axpby(Array &v1, float alpha, Array &v2, float beta,
                                 Array &v3) {
  v1[0] = alpha * v2[0] + beta * v3[0];
  v1[1] = alpha * v2[1] + beta * v3[1];
  v1[2] = alpha * v2[2] + beta * v3[2];
  v1[3] = alpha * v2[3] + beta * v3[3];
}
