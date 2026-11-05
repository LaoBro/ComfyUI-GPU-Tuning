// gfx906 fused fp16 attention forward, v4: v3 templated on the head dim.
//
// Layout and why (measured on MI50, 60 CU, 1.8 GHz):
//   v_dot2_f32_f16 with 8 independent accumulators sustains 12.0 TMAC/s, but an
//   isolated QK loop with a 1x8 per-lane tile only reached 3.24: nine 16-byte
//   LDS reads per 32 dot2. With a 4x8 per-lane tile the same loop reaches 6.84
//   (the 4-wave/CU VALU ceiling is 7.2), which is what this kernel builds on.
//   128 Q rows x 64 keys per block, 256 threads (4 waves, one block per CU).
//   warp -> 32 Q rows, lane (r = lane&7, j = lane>>3) -> rows 8t+r (t<4), keys 8j+c.
//   QK: 32 fp32 accumulators, 12 LDS reads per 128 dot2 (0.375 dword/dot2).
//   PV: 4 rows x HDT/8 dims fp32, 4+HDT/8 LDS reads per 32*HDT/8 dot2.
// LDS: Q 32K/16K, K 16K, Vt 16K/8K, and P (16K) reuses the K buffer after QK.
// The pipeline pays one extra barrier for that aliasing. Row strides are
// XOR-swizzled in 16-byte chunks so the eight rows of a 16-lane phase land on
// eight different bank groups; a padded stride cannot (rows 8 apart alias).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define NQ 128
#define NK 64
#define NT 256
#define TQ 4
#define TK 8
#define LOG2E 1.4426950408889634f
#define NEG_BIG (-1e30f)

using u4 = uint4;

__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
    float d;
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    return d;
}
__device__ __forceinline__ uint32_t pk_lo(uint32_t a, uint32_t b) { return __byte_perm(a, b, 0x5410); }
__device__ __forceinline__ uint32_t pk_hi(uint32_t a, uint32_t b) { return __byte_perm(a, b, 0x7632); }
__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }

// q/k/v/out are [B, S, H, HDT] fp16 contiguous.
template <int HDT>
__global__ void __launch_bounds__(NT, 1)
attn_fwd_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
                const __half* __restrict__ v, __half* __restrict__ out,
                int SQ, int SK, int H, float sm_scale) {
    constexpr int CPT = HDT / 8;      // 16-byte chunks per row (also QK depth steps)
    constexpr int DPT = HDT / 8;      // dims per lane in PV
    constexpr int KCH = CPT / 4;      // K chunks per thread (16 threads per row quad)
    constexpr int QCH = CPT / 2;      // Q chunks per thread (2 threads per row)
    constexpr int VSEG = HDT / 8;     // V threads per key pair row
    constexpr int VWPT = 32 * VSEG / NT;

    __shared__ __align__(16) __half Qs[NQ * HDT];
    __shared__ __align__(16) __half Ks[NQ * NK];   // K tile, then the P tile
    __shared__ __align__(16) uint32_t Vt[NK / 2][HDT];

    const int bh = blockIdx.y;
    const int b = bh / H, h = bh - b * H;
    const int sq0 = blockIdx.x * NQ;
    const int warp = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int r = lane & 7, j = lane >> 3;
    const int row[TQ] = {warp * 32 + r, warp * 32 + 8 + r, warp * 32 + 16 + r, warp * 32 + 24 + r};
    const int key0 = 8 * j;

    // Q tile: two threads per row, half the row each
    {
        const int qrow = threadIdx.x >> 1, halfsel = threadIdx.x & 1;
        const int gr = imin(sq0 + qrow, SQ - 1);
        const __half* p = q + ((int64_t)(b * SQ + gr) * H + h) * HDT + halfsel * (HDT / 2);
        const u4* src = (const u4*)p;
#pragma unroll
        for (int i = 0; i < QCH; ++i) {
            const int chunk = halfsel * (CPT / 2) + i;
            *(u4*)(Qs + qrow * HDT + ((chunk ^ (qrow & 7)) * 8)) = src[i];
        }
    }

    const int krow = threadIdx.x >> 2, kseg = threadIdx.x & 3;
    const int vseg = threadIdx.x & (VSEG - 1);
    const int vw0 = threadIdx.x >> (VSEG == 16 ? 4 : 3);

    u4 kreg[KCH];
    u4 va[VWPT], vb[VWPT];
    auto load_k = [&](int kt) {
        const int gr = imin(kt * NK + krow, SK - 1);
        const __half* p = k + ((int64_t)(b * SK + gr) * H + h) * HDT + kseg * (KCH * 8);
        const u4* src = (const u4*)p;
#pragma unroll
        for (int i = 0; i < KCH; ++i) kreg[i] = src[i];
    };
    auto load_v = [&](int kt) {
#pragma unroll
        for (int t = 0; t < VWPT; ++t) {
            const int w = vw0 + (NT / VSEG) * t;
            const int g0 = imin(kt * NK + 2 * w, SK - 1);
            const int g1 = imin(kt * NK + 2 * w + 1, SK - 1);
            va[t] = *(const u4*)(v + ((int64_t)(b * SK + g0) * H + h) * HDT + vseg * 8);
            vb[t] = *(const u4*)(v + ((int64_t)(b * SK + g1) * H + h) * HDT + vseg * 8);
        }
    };

    float accq[TQ][TK];
#pragma unroll
    for (int t = 0; t < TQ; ++t)
#pragma unroll
        for (int c = 0; c < TK; ++c) accq[t][c] = 0.f;
    float accp[TQ][DPT];
#pragma unroll
    for (int t = 0; t < TQ; ++t)
#pragma unroll
        for (int d = 0; d < DPT; ++d) accp[t][d] = 0.f;
    float m_run[TQ], l_run[TQ];
#pragma unroll
    for (int t = 0; t < TQ; ++t) { m_run[t] = NEG_BIG; l_run[t] = 0.f; }

    const int ntiles = (SK + NK - 1) / NK;
    __syncthreads();
    if (ntiles > 0) load_k(0);

    for (int kt = 0; kt < ntiles; ++kt) {
        __syncthreads();                       // previous PV is done with Ks (P) and Vt
        load_v(kt);
#pragma unroll
        for (int i = 0; i < KCH; ++i) {
            const int chunk = kseg * KCH + i;
            *(u4*)(Ks + krow * HDT + ((chunk ^ (krow & 7)) * 8)) = kreg[i];
        }
        if (kt + 1 < ntiles) load_k(kt + 1);
        {
#pragma unroll
            for (int t = 0; t < VWPT; ++t) {
                const int w = vw0 + (NT / VSEG) * t;
                uint32_t o[8];
                const uint32_t* A = (const uint32_t*)&va[t];
                const uint32_t* B = (const uint32_t*)&vb[t];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    o[2 * c] = pk_lo(A[c], B[c]);
                    o[2 * c + 1] = pk_hi(A[c], B[c]);
                }
                *(u4*)&Vt[w][vseg * 8] = *(const u4*)o;
                *(u4*)&Vt[w][vseg * 8 + 4] = *(const u4*)(o + 4);
            }
        }
        __syncthreads();                       // Ks and Vt ready

        // QK
        {
#pragma unroll
            for (int t = 0; t < TQ; ++t)
#pragma unroll
                for (int c = 0; c < TK; ++c) accq[t][c] = 0.f;
#pragma unroll 2
            for (int s = 0; s < CPT; ++s) {
                u4 qw[TQ], kw[TK];
#pragma unroll
                for (int t = 0; t < TQ; ++t)
                    qw[t] = *(const u4*)(Qs + row[t] * HDT + ((s ^ (row[t] & 7)) * 8));
#pragma unroll
                for (int c = 0; c < TK; ++c)
                    kw[c] = *(const u4*)(Ks + (key0 + c) * HDT + ((s ^ c) * 8));
#pragma unroll
                for (int t = 0; t < TQ; ++t)
#pragma unroll
                    for (int c = 0; c < TK; ++c) {
                        accq[t][c] = dot2(qw[t].x, kw[c].x, accq[t][c]);
                        accq[t][c] = dot2(qw[t].y, kw[c].y, accq[t][c]);
                        accq[t][c] = dot2(qw[t].z, kw[c].z, accq[t][c]);
                        accq[t][c] = dot2(qw[t].w, kw[c].w, accq[t][c]);
                    }
            }
        }

        // softmax over the 64-key tile. Ks is dead after QK, so the barrier
        // before overwriting it with P sits right here.
        __syncthreads();
        const int kpos = kt * NK + key0;
        float mt[TQ];
#pragma unroll
        for (int t = 0; t < TQ; ++t) {
#pragma unroll
            for (int c = 0; c < TK; ++c) {
                accq[t][c] *= sm_scale;
                if (kpos + c >= SK) accq[t][c] = NEG_BIG;
            }
            float m = accq[t][0];
#pragma unroll
            for (int c = 1; c < TK; ++c) m = __builtin_fmaxf(m, accq[t][c]);
            m = __builtin_fmaxf(m, __shfl_xor(m, 8, 64));
            m = __builtin_fmaxf(m, __shfl_xor(m, 16, 64));
            m = __builtin_fmaxf(m, __shfl_xor(m, 32, 64));
            mt[t] = m;
        }
#pragma unroll
        for (int t = 0; t < TQ; ++t) {
            const float m = mt[t];
            const float m_new = __builtin_fmaxf(m_run[t], m);
            const float a = __builtin_amdgcn_exp2f((m_run[t] - m_new) * LOG2E);
            const float beta = __builtin_amdgcn_exp2f((m - m_new) * LOG2E);
            m_run[t] = m_new;
            float ps = 0.f;
            uint32_t ph[4];
#pragma unroll
            for (int c = 0; c < TK; c += 2) {
                float p0 = (kpos + c < SK) ? __builtin_amdgcn_exp2f((accq[t][c] - m) * LOG2E) * beta : 0.f;
                float p1 = (kpos + c + 1 < SK) ? __builtin_amdgcn_exp2f((accq[t][c + 1] - m) * LOG2E) * beta : 0.f;
                ph[c / 2] = (__half_as_ushort(__float2half(p0))
                             | ((uint32_t)__half_as_ushort(__float2half(p1)) << 16));
                ps += p0 + p1;
            }
            ps += __shfl_xor(ps, 8, 64);
            ps += __shfl_xor(ps, 16, 64);
            ps += __shfl_xor(ps, 32, 64);
            l_run[t] = l_run[t] * a + ps;
#pragma unroll
            for (int d = 0; d < DPT; ++d) accp[t][d] *= a;
            __half* prow = Ks + row[t] * NK;
            *(u4*)(prow + ((j ^ (row[t] & 7)) * 8)) = *(const u4*)ph;
        }
        __syncthreads();                       // P (in Ks) visible

        // PV
        {
            const int dbase = DPT * j;
#pragma unroll 2
            for (int k4 = 0; k4 < NK / 8; ++k4) {
                uint32_t p[TQ][4];
#pragma unroll
                for (int t = 0; t < TQ; ++t) {
                    const uint4 v4 = *(const uint4*)(Ks + row[t] * NK + ((k4 ^ (row[t] & 7)) * 8));
                    p[t][0] = v4.x; p[t][1] = v4.y; p[t][2] = v4.z; p[t][3] = v4.w;
                }
#pragma unroll
                for (int q4 = 0; q4 < 4; ++q4) {
                    const int q = k4 * 4 + q4;
                    uint4 w[DPT / 4];
#pragma unroll
                    for (int i = 0; i < DPT / 4; ++i) w[i] = *(const uint4*)&Vt[q][dbase + 4 * i];
#pragma unroll
                    for (int t = 0; t < TQ; ++t) {
                        const uint32_t pd = p[t][q4];
                        const uint32_t* wd = (const uint32_t*)w;
#pragma unroll
                        for (int d = 0; d < DPT; ++d) accp[t][d] = dot2(pd, wd[d], accp[t][d]);
                    }
                }
            }
        }
    }

    // output
#pragma unroll
    for (int t = 0; t < TQ; ++t) {
        const int gq = sq0 + row[t];
        if (gq >= SQ) continue;
        const float rcp = 1.f / l_run[t];
        __half hout[DPT];
#pragma unroll
        for (int d = 0; d < DPT; ++d) hout[d] = __float2half(accp[t][d] * rcp);
        const int64_t off = (((int64_t)(b * SQ + gq) * H + h) * HDT) + DPT * j;
#pragma unroll
        for (int i = 0; i < DPT / 8; ++i) *(u4*)(out + off + 8 * i) = *(const u4*)(hout + 8 * i);
    }
}

extern "C" int gfx906_attn_fwd(const void* q, const void* k, const void* v, void* out,
                               int B, int SQ, int SK, int H, int D, float scale,
                               void* stream) {
    if (SQ <= 0 || SK <= 0 || H <= 0 || B <= 0) return -2;
    const dim3 grid((SQ + NQ - 1) / NQ, B * H);
    const __half* qh = (const __half*)q;
    const __half* kh = (const __half*)k;
    const __half* vh = (const __half*)v;
    __half* oh = (__half*)out;
    hipStream_t st = (hipStream_t)stream;
    if (D == 128) {
        attn_fwd_kernel<128><<<grid, NT, 0, st>>>(qh, kh, vh, oh, SQ, SK, H, scale);
    } else if (D == 64) {
        attn_fwd_kernel<64><<<grid, NT, 0, st>>>(qh, kh, vh, oh, SQ, SK, H, scale);
    } else {
        return -2;
    }
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : (int)e;
}
