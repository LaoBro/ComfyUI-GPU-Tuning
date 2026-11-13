// gfx906 fused INT8 attention forward.
//
// Inputs are already quantized: Q/K are int8 [B, S, H, D] with one fp32 scale
// per (token, head); V is packed as [B, H, SKP/4, D] int32, each word holding
// four consecutive keys of one head dim, with one fp32 scale per (head, dim).
// The output is fp16 [B, S, H, D]. Nothing in this kernel dequantizes a whole
// tensor: scales ride on the fp32 accumulators.
//
// Why this shape:
//   v_dot4_i32_i8 / v_dot4_u32_u8 do twice the MACs of v_dot2_f32_f16 per
//   issue. QK uses signed int8 x int8; PV uses unsigned P (0..255) x V stored
//   as uint8 = int8 ^ 0x80, and the +128 bias is removed exactly by
//   subtracting 128 * sum(P8) per row, so V keeps all 8 bits and the P scale
//   never has to be folded into V.
//   Tile is the fp16 kernel's: 128 Q rows x 64 keys, 256 threads, 1 block/CU.
//   Per-lane tile is 4 rows x 8 keys for QK and 4 rows x D/8 dims for PV.
//   With int8 the same 16-byte LDS reads feed twice the MACs, and the whole
//   block needs ~33 KB of LDS instead of 64 KB (d128).
//   PV lane j owns dims 4j + 32i, not a contiguous slice: eight lanes reading
//   the same key quad then cover banks 0,4,...,28 exactly, where a contiguous
//   16-dim slice made all eight lanes alias two bank groups.
//
// Contract: D in {64, 128}; SKP = ceil(SK / 64) * 64 and V is zero padded to
// it; padded keys are masked out of softmax, so their values never matter.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define NK 64
#define NT 256
#define TK 8
#define LOG2E 1.4426950408889634f
#define NEG_BIG (-1e30f)

using u4 = uint4;
using u2 = uint2;

__device__ __forceinline__ int dot4s(uint32_t a, uint32_t b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, 0);
}
__device__ __forceinline__ int dot4u(uint32_t a, uint32_t b, int c) {
    return __builtin_amdgcn_udot4(a, b, c, 0);
}
__device__ __forceinline__ float wave_max3(float m) {
    m = __builtin_fmaxf(m, __shfl_xor(m, 8, 64));
    m = __builtin_fmaxf(m, __shfl_xor(m, 16, 64));
    return __builtin_fmaxf(m, __shfl_xor(m, 32, 64));
}
__device__ __forceinline__ float wave_sum3(float v) {
    v += __shfl_xor(v, 8, 64);
    v += __shfl_xor(v, 16, 64);
    return v + __shfl_xor(v, 32, 64);
}
__device__ __forceinline__ int wave_sumi3(int v) {
    v += __shfl_xor(v, 8, 64);
    v += __shfl_xor(v, 16, 64);
    return v + __shfl_xor(v, 32, 64);
}
__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }

// TQ is the number of Q rows per lane and also the occupancy knob: the block
// has 32*TQ Q rows, VGPR scales with TQ, and TQ=2 is what fits two blocks per
// CU (<=128 VGPR and <=32 KB LDS). TQ=4 keeps 128 rows resident and streams
// K/V half as often.
//
// VKEY: V carries one scale per key row (the Qwen-Image 2.1 cache stores it
// that way). The scale is folded into P, so the PV dot needs no per-dim scale.
// K comes from two sources: the cached prefix (k1, rows SP apart within a
// batch of pitch1 rows; the cache keeps K and V stacked, so its K plane is not
// batch-contiguous) followed by the current step (k2, [B, ST, H, D]). That
// avoids copying the prefix K into a scratch buffer on every block.
template <int HDT, int TQ, bool VKEY>
__global__ void __launch_bounds__(NT, TQ == 2 ? 2 : 1)
attn_i8_kernel(const int8_t* __restrict__ q, const float* __restrict__ qs,
               const int8_t* __restrict__ k1, const float* __restrict__ ks1,
               const float* __restrict__ vk1, int64_t pitch1,
               const int8_t* __restrict__ k2, const float* __restrict__ ks2,
               const float* __restrict__ vk2,
               const int32_t* __restrict__ vq, const float* __restrict__ vs,
               __half* __restrict__ out,
               int SQ, int SP, int ST, int SKP, int H, int HKV, int unrotate, float sm_scale) {
    constexpr int NQ = 32 * TQ;        // Q rows per block (4 warps x 8 x TQ)
    constexpr int CPT = HDT / 16;      // 16-byte chunks per row
    constexpr int DPT = HDT / 8;       // dims per lane in PV
    constexpr int KCH = CPT / 4;       // K chunks per thread (4 threads per row)
    constexpr int TPR = 8 / TQ;        // threads per Q row
    constexpr int QCH = CPT / TPR;     // Q chunks per thread
    constexpr int NQCH = NK / 16;      // 16-key PV steps per tile
    constexpr int KB = NQ > HDT ? NQ : HDT;   // K tile (NK x HDT) or P tile (NQ x NK)
    const int SK = SP + ST;

    __shared__ __align__(16) int8_t Qs[NQ * HDT];
    __shared__ __align__(16) uint8_t Ks[KB * NK];      // K tile, then the P tile
    __shared__ __align__(16) uint32_t Vt[NK / 4][HDT];
    __shared__ float Sk[NK];
    __shared__ float Sv[HDT];
    __shared__ float Svk[VKEY ? NK : 1];

    const int bh = blockIdx.y;
    const int b = bh / H, h = bh - b * H;
    const int hk = h / (H / HKV);
    const int sq0 = blockIdx.x * NQ;
    const int warp = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int r = lane & 7, j = lane >> 3;
        int row[TQ];
#pragma unroll
    for (int t = 0; t < TQ; ++t) row[t] = warp * (NQ / 4) + 8 * t + r;
    const int key0 = 8 * j;

    // Q tile: TPR threads per row, one row slice each
    {
        const int qrow = threadIdx.x / TPR, sel = threadIdx.x % TPR;
        const int gr = imin(sq0 + qrow, SQ - 1);
        const int8_t* p = q + ((int64_t)((b * SQ + gr) * H + h) * HDT) + sel * (HDT / TPR);
#pragma unroll
        for (int i = 0; i < QCH; ++i) {
            const int chunk = sel * QCH + i;
            *(u4*)(Qs + qrow * HDT + ((chunk ^ (qrow & (CPT - 1))) * 16)) = *(const u4*)(p + i * 16);
        }
    }

    float qsr[TQ];
#pragma unroll
    for (int t = 0; t < TQ; ++t)
        qsr[t] = qs[(int64_t)((b * SQ + imin(sq0 + row[t], SQ - 1)) * H + h)] * sm_scale;
    if (!VKEY && threadIdx.x < HDT) Sv[threadIdx.x] = vs[(int64_t)(b * HKV + hk) * HDT + threadIdx.x];

    const int krow = threadIdx.x >> 2, kseg = threadIdx.x & 3;
    u4 kreg[KCH];
    // key row r: cached prefix rows sit pitch1 apart per batch, the current
    // step's K is a plain [B, ST, H, D] tensor. K has HKV heads, not H.
    auto load_k = [&](int kt) {
        const int gr = imin(kt * NK + krow, SK - 1);
        const int8_t* p = gr < SP
            ? k1 + (int64_t)((int64_t)b * pitch1 + gr * HKV + hk) * HDT
            : k2 + ((int64_t)(b * ST + (gr - SP)) * HKV + hk) * HDT;
        p += kseg * (KCH * 16);
#pragma unroll
        for (int i = 0; i < KCH; ++i) kreg[i] = *(const u4*)(p + i * 16);
    };
    const int32_t* vbase = vq + (int64_t)(b * HKV + hk) * (SKP / 4) * HDT;
    auto load_v = [&](int kt) {
        constexpr int VD = HDT / 4;                       // 4-dim chunks per key quad
        constexpr int VCH = (NK / 4) * VD;                // chunks in one tile
#pragma unroll
        for (int base = threadIdx.x; base < VCH; base += NT) {
            const int qq = base / VD, dc = base - qq * VD;
            const int gq = kt * (NK / 4) + qq;
            // no swizzle: PV lane j reads chunk j + 8i of every quad, so the
            // eight lanes already cover 32 banks (see the PV loop).
            // xor 0x80 lifts the signed int8 into its uint8 image: the PV dot is
            // unsigned x unsigned, and the 128 bias is subtracted exactly once
            // per row after the tile.
            u4 vv = *(const u4*)(vbase + (int64_t)gq * HDT + dc * 4);
            vv.x ^= 0x80808080u;
            vv.y ^= 0x80808080u;
            vv.z ^= 0x80808080u;
            vv.w ^= 0x80808080u;
            *(u4*)&Vt[qq][dc * 4] = vv;
        }
    };

    int accq[TQ][TK];
    float accp[TQ][DPT];
#pragma unroll
    for (int t = 0; t < TQ; ++t) {
#pragma unroll
        for (int d = 0; d < DPT; ++d) accp[t][d] = 0.f;
    }
    float m_run[TQ], l_run[TQ];
#pragma unroll
    for (int t = 0; t < TQ; ++t) { m_run[t] = NEG_BIG; l_run[t] = 0.f; }

    const int ntiles = (SK + NK - 1) / NK;
    __syncthreads();

    for (int kt = 0; kt < ntiles; ++kt) {
        __syncthreads();
        load_v(kt);
        if (threadIdx.x < NK) {
            const int key = imin(kt * NK + threadIdx.x, SK - 1);
            if (key < SP) {
                const int64_t si = (int64_t)b * pitch1 + key * HKV + hk;
                Sk[threadIdx.x] = ks1[si];
                if (VKEY) Svk[threadIdx.x] = vk1[si];
            } else {
                const int64_t si = (int64_t)(b * ST + (key - SP)) * HKV + hk;
                Sk[threadIdx.x] = ks2[si];
                if (VKEY) Svk[threadIdx.x] = vk2[si];
            }
        }
        load_k(kt);
#pragma unroll
        for (int i = 0; i < KCH; ++i) {
            const int chunk = kseg * KCH + i;
            *(u4*)(Ks + krow * HDT + ((chunk ^ (krow & (CPT - 1))) * 16)) = kreg[i];
        }
        __syncthreads();

        // QK: scores are int32 dot products; the two scales come after.
#pragma unroll
        for (int t = 0; t < TQ; ++t)
#pragma unroll
            for (int c = 0; c < TK; ++c) accq[t][c] = 0;
#pragma unroll(TQ == 2 ? 1 : 2)
        for (int s = 0; s < CPT; ++s) {
            u4 qw[TQ], kw[TK];
#pragma unroll
            for (int t = 0; t < TQ; ++t)
                qw[t] = *(const u4*)(Qs + row[t] * HDT + ((s ^ (row[t] & (CPT - 1))) * 16));
#pragma unroll
            for (int c = 0; c < TK; ++c) {
                const int kr = key0 + c;
                kw[c] = *(const u4*)(Ks + kr * HDT + ((s ^ (kr & (CPT - 1))) * 16));
            }
#pragma unroll
            for (int t = 0; t < TQ; ++t)
#pragma unroll
                for (int c = 0; c < TK; ++c) {
                    accq[t][c] = dot4s(qw[t].x, kw[c].x, accq[t][c]);
                    accq[t][c] = dot4s(qw[t].y, kw[c].y, accq[t][c]);
                    accq[t][c] = dot4s(qw[t].z, kw[c].z, accq[t][c]);
                    accq[t][c] = dot4s(qw[t].w, kw[c].w, accq[t][c]);
                }
        }

        __syncthreads();                       // Ks is dead: P lands there
        const int kpos = kt * NK + key0;
        float ksc[TK], vsk[TK];
#pragma unroll
        for (int c = 0; c < TK; ++c) {
            ksc[c] = Sk[key0 + c];
            if (VKEY) vsk[c] = Svk[key0 + c];
        }
        float pscale[TQ], corr[TQ];
#pragma unroll
        for (int t = 0; t < TQ; ++t) {
            float scv[TK];
            float m = NEG_BIG;
#pragma unroll
            for (int c = 0; c < TK; ++c) {
                scv[c] = (kpos + c < SK) ? (float)accq[t][c] * qsr[t] * ksc[c] : NEG_BIG;
                m = __builtin_fmaxf(m, scv[c]);
            }
            m = wave_max3(m);
            const float m_new = __builtin_fmaxf(m_run[t], m);
            const float a = __builtin_amdgcn_exp2f((m_run[t] - m_new) * LOG2E);
            const float beta = __builtin_amdgcn_exp2f((m - m_new) * LOG2E);
            m_run[t] = m_new;
            float pv[TK];
#pragma unroll
            for (int c = 0; c < TK; ++c)
                pv[c] = (kpos + c < SK) ? __builtin_amdgcn_exp2f((scv[c] - m) * LOG2E) * beta : 0.f;
            // P scale: beta bounds p; with per-key V scales it is the row max
            // of p*v instead, which needs one extra wave reduction.
            float invp, sp;
            if (VKEY) {
                float mx = 0.f;
#pragma unroll
                for (int c = 0; c < TK; ++c) mx = __builtin_fmaxf(mx, pv[c] * vsk[c]);
                mx = wave_max3(mx);
                invp = mx > 0.f ? 255.f * __builtin_amdgcn_rcpf(mx) : 0.f;
                sp = mx * (1.f / 255.f);
            } else {
                invp = (beta > 1e-30f) ? 255.f / beta : 0.f;
                sp = beta * (1.f / 255.f);
            }
            float ps = 0.f;
            int pq = 0;
            uint32_t ph[2] = {0, 0};
#pragma unroll
            for (int c = 0; c < TK; ++c) {
                const float pm = VKEY ? pv[c] * vsk[c] : pv[c];
                int q0 = (int)__builtin_rintf(pm * invp);
                q0 = q0 > 255 ? 255 : q0;
                ps += pv[c];
                pq += q0;
                ph[c / 4] |= (uint32_t)q0 << (8 * (c % 4));
            }
            ps = wave_sum3(ps);
            l_run[t] = l_run[t] * a + ps;
            pscale[t] = sp;
            corr[t] = 128 * wave_sumi3(pq);
#pragma unroll
            for (int d = 0; d < DPT; ++d) accp[t][d] *= a;
            const int seg = j >> 1;
            u2 st = {ph[0], ph[1]};
            *(u2*)(Ks + row[t] * NK + ((seg ^ (row[t] & 3)) * 16) + ((j & 1) * 8)) = st;
        }
        __syncthreads();                       // P visible

        // PV: unsigned P x (V ^ 0x80); the 128 bias is removed below.
        int acci[TQ][DPT];
#pragma unroll
        for (int t = 0; t < TQ; ++t)
#pragma unroll
            for (int d = 0; d < DPT; ++d) acci[t][d] = 0;
#pragma unroll(TQ == 2 ? 1 : 2)
        for (int k4 = 0; k4 < NQCH; ++k4) {
            u4 pw[TQ];
#pragma unroll
            for (int t = 0; t < TQ; ++t)
                pw[t] = *(const u4*)(Ks + row[t] * NK + ((k4 ^ (row[t] & 3)) * 16));
#pragma unroll
            for (int q4 = 0; q4 < 4; ++q4) {
                const int q = k4 * 4 + q4;
                u4 w[DPT / 4];
#pragma unroll
                for (int i = 0; i < DPT / 4; ++i)
                    w[i] = *(const u4*)&Vt[q][4 * j + 32 * i];
#pragma unroll
                for (int t = 0; t < TQ; ++t) {
                    const uint32_t pd = ((const uint32_t*)&pw[t])[q4];
                    const uint32_t* wd = (const uint32_t*)w;
#pragma unroll
                    for (int d = 0; d < DPT; ++d) acci[t][d] = dot4u(pd, wd[d], acci[t][d]);
                }
            }
        }
        // merge the tile: pscale * (dot - 128 * sum p) * v_scale. accp already
        // carries the running-max rescale from the softmax block.
#pragma unroll
        for (int t = 0; t < TQ; ++t) {
            const float pa = pscale[t];
            const int cf = corr[t];
#pragma unroll
            for (int i = 0; i < DPT / 4; ++i) {
                if (VKEY) {
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                        accp[t][4 * i + k] += pa * (float)(acci[t][4 * i + k] - cf);
                } else {
                    const u4 vw = *(const u4*)&Sv[4 * j + 32 * i];
                    const float* vf = (const float*)&vw;
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                        accp[t][4 * i + k] += pa * vf[k] * (float)(acci[t][4 * i + k] - cf);
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
        float o[DPT / 4][4];
#pragma unroll
        for (int i = 0; i < DPT / 4; ++i)
#pragma unroll
            for (int k = 0; k < 4; ++k) o[i][k] = accp[t][4 * i + k] * rcp;
        if (unrotate) {
            // the cache stores V in the H64 basis, so the result comes back rotated;
            // H64 = H4 (x) H4 (x) H4 over each 64-dim group, with the digits being k
            // (in-register), j (lane ^8/^16/^24) and 2*(i/2) + (j >> 2) (register i^1
            // and lane ^32). Doing it here saves a separate 33 MB round trip.
#pragma unroll
            for (int i = 0; i < DPT / 4; ++i) {
                const float a0 = o[i][0], a1 = o[i][1], a2 = o[i][2], a3 = o[i][3];
                o[i][0] = a0 + a1 + a2 - a3;
                o[i][1] = a1 + a0 + a3 - a2;
                o[i][2] = a2 + a3 + a0 - a1;
                o[i][3] = a3 + a2 + a1 - a0;
            }
#pragma unroll
            for (int i = 0; i < DPT / 4; ++i)
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float a = __shfl_xor(o[i][k], 8, 64);
                    const float bq = __shfl_xor(o[i][k], 16, 64);
                    const float c = __shfl_xor(o[i][k], 24, 64);
                    o[i][k] = o[i][k] + a + bq - c;
                }
#pragma unroll
            for (int p = 0; p < DPT / 8; ++p) {
                float* u0 = o[2 * p];
                float* u1 = o[2 * p + 1];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float q0 = __shfl_xor(u0[k], 32, 64);
                    const float q1 = __shfl_xor(u1[k], 32, 64);
                    const float n0 = u0[k] + q0 + u1[k] - q1;
                    const float n1 = u1[k] + q1 + u0[k] - q0;
                    u0[k] = n0;
                    u1[k] = n1;
                }
            }
#pragma unroll
            for (int i = 0; i < DPT / 4; ++i)
#pragma unroll
                for (int k = 0; k < 4; ++k) o[i][k] *= 0.125f;
        }
        const int64_t off = ((int64_t)(b * SQ + gq) * H + h) * HDT;
#pragma unroll
        for (int i = 0; i < DPT / 4; ++i) {
            __half ho[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) ho[k] = __float2half(o[i][k]);
            *(u2*)(out + off + 4 * j + 32 * i) = *(const u2*)ho;
        }
    }
}

extern "C" int gfx906_attn_i8_fwd(const void* q, const void* qs,
                                  const void* k1, const void* ks1, const void* vk1,
                                  long long pitch1,
                                  const void* k2, const void* ks2, const void* vk2,
                                  const void* vq, const void* vs, void* out,
                                  int B, int SQ, int SP, int ST, int SKP, int H, int HKV, int D,
                                  int tq, int unrotate, float scale, void* stream) {
    const int SK = SP + ST;
    if (B <= 0 || SQ <= 0 || SK <= 0 || H <= 0 || HKV <= 0) return -2;
    if (tq != 2 && tq != 4) return -3;
    if (SP < 0 || ST < 0 || (SP > 0 && (k1 == nullptr || ks1 == nullptr))
            || (ST > 0 && (k2 == nullptr || ks2 == nullptr))) return -2;
    if (H % HKV || SKP < ((SK + NK - 1) / NK) * NK || SKP % NK) return -3;
    const int nq = 32 * tq;
    const dim3 grid((SQ + nq - 1) / nq, B * H);
    hipStream_t st = (hipStream_t)stream;
    const bool vkey = vk1 != nullptr;
#define LAUNCH(HD, TQ_, VK)                                                             \
    attn_i8_kernel<HD, TQ_, VK><<<grid, NT, 0, st>>>(                                   \
        (const int8_t*)q, (const float*)qs, (const int8_t*)k1, (const float*)ks1,       \
        (const float*)vk1, pitch1, (const int8_t*)k2, (const float*)ks2,                \
        (const float*)vk2, (const int32_t*)vq, (const float*)vs, (__half*)out,          \
        SQ, SP, ST, SKP, H, HKV, unrotate, scale)
    if (D == 128) {
        if (tq == 2) { vkey ? LAUNCH(128, 2, true) : LAUNCH(128, 2, false); }
        else { vkey ? LAUNCH(128, 4, true) : LAUNCH(128, 4, false); }
    } else if (D == 64) {
        if (tq == 2) { vkey ? LAUNCH(64, 2, true) : LAUNCH(64, 2, false); }
        else { vkey ? LAUNCH(64, 4, true) : LAUNCH(64, 4, false); }
    } else {
        return -2;
    }
#undef LAUNCH
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : (int)e;
}
