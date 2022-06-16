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

__inline__ __device__ float4 vectorizeLoad(float *addr) {
  return *((float4 *)addr);
}

__inline__ __device__ void vectorizeLoad(float *addr, float4 value) {
  *((float4 *)(addr)) = value;
}
