// gfx906 helpers for the Qwen-Image 2.1 int8 prefix cache.
//
// The cache stores K/V with comfy_kitchen's convrot: rows rotated by the
// regular Hadamard H64 = h4 (x) h4 (x) h4 / 8 (h4[a][b] = -1 iff a + b == 3),
// then int8 with one scale per row. Q and the current step's K/V have to be
// put in the same basis before the int8 attention kernel can mix them, and the
// attention output comes back in that basis, so it is rotated once more on the
// way out (H is symmetric and orthogonal, so the same transform undoes it).
//
// Three passes, each one read + one write instead of the 5-6 torch kernels a
// per-row quantize costs:
//   gfx906_rot_quant     rows -> int8 rows + per-row scale
//   gfx906_rot_pack_v    four keys -> one packed int32 word per dim, from two
//                        sources (cached prefix + current step), so a prefix
//                        length that is not a multiple of four needs no padding
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define G 64
#define INV8 0.125f

// 256-thread blocks: four 64-lane rows per block for the row kernels, four key
// quads for the pack. One tiny block per row meant 129k blocks of 64 threads and
// ran 3-5x off the bandwidth the same bytes should take.
#define NTQ 256
#define RPB (NTQ / 64)
#define QPB 8                     // quads per block: eight 32-lane groups

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

// one radix-4 stage x3 over the 64-lane wave; matches rotate.cu's butterfly
__device__ __forceinline__ float bf3(float v) {
#pragma unroll
    for (int D = 1; D <= 16; D *= 4) {
        const float a = __shfl_xor(v, D, 64);
        const float b = __shfl_xor(v, 2 * D, 64);
        const float c = __shfl_xor(v, 3 * D, 64);
        v = (v + a + b + c) - 2.0f * c;
    }
    return v * INV8;
}

__device__ __forceinline__ float wave_max(float m) {
    m = __builtin_fmaxf(m, __shfl_xor(m, 1, 64));
    m = __builtin_fmaxf(m, __shfl_xor(m, 2, 64));
    m = __builtin_fmaxf(m, __shfl_xor(m, 4, 64));
    m = __builtin_fmaxf(m, __shfl_xor(m, 8, 64));
    m = __builtin_fmaxf(m, __shfl_xor(m, 16, 64));
    return __builtin_fmaxf(m, __shfl_xor(m, 32, 64));
}

template <typename T>
__global__ void __launch_bounds__(NTQ, 4)
rot_quant_kernel(const T* __restrict__ x, int8_t* __restrict__ out,
                 float* __restrict__ scale, int rows, int D, int ng) {
    // the butterfly is a width-64 shuffle, so a row is one sub-wave
    const int lane = threadIdx.x & 63;
    const int64_t row = (int64_t)blockIdx.x * RPB + (threadIdx.x >> 6);
    if (row >= rows) return;
    float v[4];
    const T* p = x + row * D + lane;
#pragma unroll
    for (int g = 0; g < 4; ++g) v[g] = g < ng ? (float)p[g * G] : 0.f;
#pragma unroll
    for (int g = 0; g < 4; ++g) v[g] = bf3(v[g]);
    float m = 0.f;
#pragma unroll
    for (int g = 0; g < 4; ++g) m = __builtin_fmaxf(m, __builtin_fabsf(v[g]));
    m = wave_max(m);
    const float s = __builtin_fmaxf(m * (1.f / 127.f), 1e-8f);
    const float r = 1.f / s;
#pragma unroll
    for (int g = 0; g < 4; ++g)
        if (g < ng) out[row * D + g * G + lane] = (int8_t)__builtin_rintf(v[g] * r);
    if (lane == 0) scale[row] = s;
}


// Four keys per block; key t comes from the cached prefix while t < SP, else
// from the current step. Both sources are int8 rows of one head stored as
// [b][S][H][D] inside a possibly larger tensor, so each is addressed by a row
// offset and a per-batch row pitch (the cache keeps K and V stacked on one
// axis, which makes its V slice non-contiguous in b).
__device__ __forceinline__ uint32_t pack_dims(const uint32_t* key, int m) {
    // byte m of four keys, in key order: the packed word the PV dot reads
    const int sh = 8 * m;
    return ((key[0] >> sh) & 0xffu) | (((key[1] >> sh) & 0xffu) << 8)
         | (((key[2] >> sh) & 0xffu) << 16) | (((key[3] >> sh) & 0xffu) << 24);
}

__global__ void __launch_bounds__(NTQ, 4)
rot_pack_v_kernel(const int8_t* __restrict__ pa, int64_t off_a, int64_t pitch_a,
                  const int8_t* __restrict__ pb, int64_t off_b, int64_t pitch_b,
                  uint32_t* __restrict__ out, int H, int D, int Sp, int Sb, int q_total) {
    // a 32-lane group packs one quad and reads four dims per lane, so a key row is
    // one 128 B transaction instead of two 64 B halves, and the 4x4 byte transpose
    // stays inside the lane
    const int dw = threadIdx.x & 31;
    const int q = blockIdx.y * QPB + (threadIdx.x >> 5);
    if (q >= q_total || 4 * dw >= D) return;
    const int bh = blockIdx.x;
    const int b = bh / H, h = bh - b * H;
    const int t0 = 4 * q;
    uint32_t key[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int t = t0 + r;
        const int8_t* p = nullptr;
        if (t < Sp) {
            p = pa + (off_a + (int64_t)b * pitch_a + (int64_t)t * H + h) * D;
        } else if (t < Sp + Sb) {
            p = pb + (off_b + (int64_t)b * pitch_b + (int64_t)(t - Sp) * H + h) * D;
        }
        key[r] = p ? *(const uint32_t*)(p + 4 * dw) : 0u;
    }
    uint4 w = {pack_dims(key, 0), pack_dims(key, 1), pack_dims(key, 2), pack_dims(key, 3)};
    *(uint4*)(out + ((int64_t)bh * q_total + q) * D + 4 * dw) = w;
}

extern "C" int gfx906_rot_quant(const void* x, void* out, void* scale,
                                int rows, int D, int dtype, void* stream) {
    if (rows <= 0 || D % G || D > 4 * G) return -3;
    const int ng = D / G;
    hipStream_t st = (hipStream_t)stream;
    if (dtype == 0)
        rot_quant_kernel<float><<<cdiv(rows, RPB), NTQ, 0, st>>>((const float*)x, (int8_t*)out,
                                                             (float*)scale, rows, D, ng);
    else
        rot_quant_kernel<__half><<<cdiv(rows, RPB), NTQ, 0, st>>>((const __half*)x, (int8_t*)out,
                                                              (float*)scale, rows, D, ng);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : (int)e;
}


extern "C" int gfx906_rot_pack_v(const void* pa, long long off_a, long long pitch_a,
                                 const void* pb, long long off_b, long long pitch_b,
                                 void* out, int B, int H, int D, int Sp, int Sb,
                                 int q_total, void* stream) {
    if (B <= 0 || H <= 0 || D % G || D > 4 * G || Sp < 0 || Sb < 0 || q_total <= 0) return -3;
    hipStream_t st = (hipStream_t)stream;
    rot_pack_v_kernel<<<dim3(B * H, cdiv(q_total, QPB)), NTQ, 0, st>>>(
        (const int8_t*)pa, off_a, pitch_a, (const int8_t*)pb, off_b, pitch_b,
        (uint32_t*)out, H, D, Sp, Sb, q_total);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : (int)e;
}
