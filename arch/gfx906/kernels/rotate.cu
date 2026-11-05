// gfx906 HIP activation rotation for convrot int8 decode.
//
// Replaces int8_gemv.py's Triton rotate_kernel, which reads a 256x256 fp32
// Hadamard matrix in 8 serial global tiles with only m*k/256 programs. Here
// H256 = h4 kron h4 kron h4 kron h4 / 16 with h4[a][b] = -1 iff a+b==3, so
// the transform is four radix-4 stages y[sel] = s - 2*V[sel^3] over the base-4
// digits. D=1/4/16 stay inside a 64-lane wave (shfl_xor), D=64 crosses waves
// through LDS. No Hadamard global traffic, no atomics, one launch.
//
// Contract: x [m,k] fp32/fp16 contiguous, groupsize 256, k % 256 == 0.
// out same shape/dtype as x; gmax [m, k/256] fp32 = max |rot| per group.
// Compute is fp32 throughout; for fp16 x the result is more accurate than the
// Triton path (which accumulates in fp16) and therefore not bit-identical.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define G 256
#define NT 256
#define INV16 0.0625f

__device__ __forceinline__ float wave_max(float m) {
    m = __builtin_fmaxf(m, __shfl_xor(m, 1, 64));
    m = __builtin_fmaxf(m, __shfl_xor(m, 2, 64));
    m = __builtin_fmaxf(m, __shfl_xor(m, 4, 64));
    m = __builtin_fmaxf(m, __shfl_xor(m, 8, 64));
    m = __builtin_fmaxf(m, __shfl_xor(m, 16, 64));
    m = __builtin_fmaxf(m, __shfl_xor(m, 32, 64));
    return m;
}

template <typename T>
__global__ void __launch_bounds__(NT, 1)
rotate_kernel(const T* __restrict__ x, T* __restrict__ out, float* __restrict__ gmax,
              int k, int ngroups) {
    __shared__ float stage[G];
    __shared__ float wm[NT / 64];
    const int tid = threadIdx.x;
    const int group = blockIdx.x;
    const int row = group / ngroups;
    const int base = row * k + (group % ngroups) * G;
    float v = (float)x[base + tid];

#pragma unroll
    for (int D = 1; D <= 16; D *= 4) {
        float a = __shfl_xor(v, D, 64);
        float b = __shfl_xor(v, 2 * D, 64);
        float c = __shfl_xor(v, 3 * D, 64);
        float s = v + a + b + c;
        // h4[sel][t] = -1 iff sel+t==3, so y[sel] = s - 2*V[sel^3]; the
        // xor-3 shuffle holds V[sel^3] for every sel.
        v = s - 2.0f * c;
    }
    stage[tid] = v;
    __syncthreads();
    {
        const int q = tid & 63;
        const int sel = tid >> 6;
        float v0 = stage[q], v1 = stage[q + 64], v2 = stage[q + 128], v3 = stage[q + 192];
        float s = v0 + v1 + v2 + v3;
        float other = (sel == 0) ? v3 : (sel == 1) ? v2 : (sel == 2) ? v1 : v0;
        v = s - 2.0f * other;
    }
    v *= INV16;
    out[base + tid] = (T)v;

    float m = wave_max(__builtin_fabsf(v));
    if ((tid & 63) == 0) wm[tid >> 6] = m;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
#pragma unroll
        for (int w = 0; w < NT / 64; ++w) t = __builtin_fmaxf(t, wm[w]);
        gmax[group] = t;
    }
}

extern "C" int gfx906_rotate(const void* x, void* out, void* gmax,
                             int m, int k, int groupsize, int dtype, void* stream) {
    if (groupsize != G || k % G || m <= 0) return -3;
    const int ngroups = k / G;
    hipStream_t st = (hipStream_t)stream;
    const int blocks = m * ngroups;
    if (dtype == 0)
        rotate_kernel<float><<<blocks, NT, 0, st>>>((const float*)x, (float*)out,
                                                    (float*)gmax, k, ngroups);
    else
        rotate_kernel<__half><<<blocks, NT, 0, st>>>((const __half*)x, (__half*)out,
                                                     (float*)gmax, k, ngroups);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : (int)e;
}
