#pragma once

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

__inline__ __device__ void vectorizeStore(float *addr, Array value) {
  reinterpret_cast<Array *>(addr)[0] = value;
}
