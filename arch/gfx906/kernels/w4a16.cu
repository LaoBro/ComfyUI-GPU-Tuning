// gfx906 W4A16 decode GEMV: packed int4 weights decoded in registers, fp16
// activations, fp32 accumulate. Three entries, all reached through
// w4a8_gemv.linear_dequant: w4a16_gemv for the single-row te=default decode,
// w4a16_gemv_m for m = 2..8 with fp32 activations, and w4a16_gemv_h for any
// m = 1..8 with fp16 activations and v_dot2_f32_f16 (opt-in: it rounds the 16
// codebook levels to fp16).
//
// Layout follows the read micro-benchmark in docs/HIP_KERNELS.md: a row is split
// over KP=32 lanes so one wavefront's 64 lanes stream 8 rows x 16 bytes each, and
// the grid is capped at 240 blocks (960 waves, 16/CU) with a row-group loop, which
// is where the packed plane peaks. A lane's 32 k elements are exactly one uint4, so
// there is no inner loop to schedule; the group scale is applied when a lane's block
// ends a group and the eight lanes of one group are summed by shuffles at the end.
// Only the activation chunk and the 16 levels live in LDS: the levels are exact fp32
// in 64 bytes spread over 64 banks, so an indexed read never conflicts and equal
// codes broadcast.
//
// Built by tools/build_hip_attn.py (target w4a16.cu -> libgfx906_w4a16.so).
#include <cstdint>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define NT 256
#define KP 32                     // lanes per output row
#define BN (NT / KP)              // output rows per block
#define BK 1024                   // k elements staged per chunk
#define ROWK (BK / KP)            // k elements per lane per chunk
#define QUAD (KP * 4)             // floats per activation quad row: 4 k of each lane: one uint4
#define MAX_BLOCKS 240
#define GB 32                     // row groups per LDS accumulator batch

union V4 {
    uint4 v;
    uint32_t w[4];
};

// fp8_e4m3fn bits, decoded the way the Triton kernel does (gfx906 has no fp8 ALU):
// exponent field 1..15 is 2^(e-7) * (1 + mantissa/8), e == 0 is a denormal step.
__device__ __forceinline__ float srel_decode(uint8_t b)
{
    const int e = (b >> 3) & 0xF;
    const float man = (float)(b & 7) * 0.125f;
    return e > 0 ? __builtin_exp2f((float)(e - 7)) * (1.0f + man) : man * __builtin_exp2f(-6.0f);
}

// One weight dword = 8 nibbles, byte j holding k = 8w + 2j in its low nibble and
// k = 8w + 2j + 1 in its high one. Each dword of the group goes to its own
// accumulator: one chain per lane would serialise on fp32 add latency, which is
// what caps this kernel's issue rate long before its stream rate.
//
// The activation chunk is stored in quad rows (see the staging loop), so the eight
// values this dword needs come from two ds_read_b128 instead of eight b32 reads:
// eight lane-strided b32 reads are what capped the byte rate, and a b128 across 16
// lanes covers 256 B = all 64 banks, so it is conflict-free by construction.
template <bool CB>
__device__ __forceinline__ void accumulate(uint32_t w, const float* tab, const float* xa,
                                          float* gacc, int k0, int slot)
{
    const uint32_t even = w & 0x0F0F0F0F;
    const uint32_t odd = (w >> 4) & 0x0F0F0F0F;
    const float4 q0 = *reinterpret_cast<const float4*>(xa + k0);
    const float4 q1 = *reinterpret_cast<const float4*>(xa + k0 + QUAD);
    const float x[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float we = tab[(even >> (8 * j)) & 0xFF];
        const float wo = tab[(odd >> (8 * j)) & 0xFF];
        gacc[slot] += we * x[2 * j] + wo * x[2 * j + 1];
    }
}

template <bool IN_F16, bool CB, bool OUT_F16, bool HAS_BIAS>
__global__ void __launch_bounds__(NT, 1)
w4a16_gemv(const void* __restrict__ xsrc, const uint32_t* __restrict__ wq,
           const uint8_t* __restrict__ srel, const float* __restrict__ sch,
           const void* __restrict__ bias, void* __restrict__ out, const float* cb,
           int n, int k, int gs, int64_t srel_stride)
{
    __shared__ float xa[BK];
    __shared__ float tab[16];
    const int tid = threadIdx.x;
    const int part = tid & (KP - 1);
    const int slot = tid / KP;
    const int groups = (n + BN - 1) / BN;
    // fp32 x is the convrot path (the rotation runs in fp32 before this kernel)
    const __half* xh = (const __half*)xsrc;
    const float* xf = (const float*)xsrc;
    const __half* bias_h = (const __half*)bias;
    const float* bias_f = (const float*)bias;

    if (tid < 16) tab[tid] = CB ? cb[tid] : (float)tid - 8.0f;
    __syncthreads();

    // The activation chunk is shared by every row this block touches, so the
    // staging loop is the outer loop's outer loop: pack one chunk, then walk the
    // block's rows. With the pack inside the row-group loop it was measured at
    // 44-46% of a chunk's cycles (s_memtime), paid once per BN output rows.
    // Row partials live in LDS because a block owns more row groups than it can
    // hold accumulators for; each entry has exactly one writer (the lane with
    // part == 0 of that row), so no atomics and no extra barriers.
    __shared__ float accg[GB * BN];
    const int gi_total = (groups - blockIdx.x + gridDim.x - 1) / gridDim.x;

    for (int gb = 0; gb < gi_total; gb += GB) {
        const int nb = gi_total - gb < GB ? gi_total - gb : GB;
        __syncthreads();                      // previous batch's flush must be done
        for (int i = tid; i < nb * BN; i += NT) accg[i] = 0.0f;

        for (int ck = 0; ck < k; ck += BK) {
            __syncthreads();
            // Quad-row layout: (t/4)*QUAD + lane*4 + (t%4) holds the value at
            // k = lane*ROWK + t, so each lane's four consecutive k are one float4
            // and the 32 lanes of a quad row are adjacent in LDS.
            for (int i = tid; i < BK; i += NT) {
                const int lane = i / ROWK, t = i % ROWK;
                xa[(t >> 2) * QUAD + lane * 4 + (t & 3)] = IN_F16 ? (float)xh[ck + i] : xf[ck + i];
            }
            __syncthreads();

            const int kbase = ck + part * ROWK;
            const int64_t sbase = (int64_t)kbase / gs;
            for (int jl = 0; jl < nb; ++jl) {
                const int row = (blockIdx.x + (gb + jl) * gridDim.x) * BN + slot;
                // Lanes past n still take part in the staging and must not read
                // past the weight plane; their value is dropped at the store.
                const int rowc = row < n ? row : n - 1;
                float gacc[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) gacc[i] = 0.0f;
                V4 t;
                t.v = *reinterpret_cast<const uint4*>(wq + (int64_t)rowc * (k / 8) + kbase / 8);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    accumulate<CB>(t.w[i], tab, xa, gacc, part * 4 + (i * 2) * QUAD, i);
                }
                // ROWK is 32 and gs >= 32, so a lane's block never spans a group:
                // every lane scales its own partial with the group it belongs to
                // and the shuffle sums the group. Flushing only on group borders
                // would drop the lanes that sit inside one.
                const float sg = srel_decode(srel[(int64_t)rowc * srel_stride + sbase]);
                float local = ((gacc[0] + gacc[1]) + (gacc[2] + gacc[3])) * sg;
#pragma unroll
                for (int step = 1; step < KP; step <<= 1) local += __shfl_xor(local, step);
                if (part == 0) accg[jl * BN + slot] += local;
            }
        }

        for (int jl = 0; jl < nb; ++jl) {
            const int row = (blockIdx.x + (gb + jl) * gridDim.x) * BN + slot;
            if (part == 0 && row < n) {
                float v = accg[jl * BN + slot] * sch[row];
                if (HAS_BIAS) v += OUT_F16 ? (float)bias_h[row] : bias_f[row];
                if (OUT_F16) ((__half*)out)[row] = (__half)v;
                else ((float*)out)[row] = v;
            }
        }
    }
}

// Multi-row (m = 2, 4, 8) variant for the small-batch / MTP decode tier. Same
// column split and the same uint4 weight stream, but one decoded dword feeds M
// activation rows, so an extra row costs FMA slots instead of a second pass over
// the weight plane -- which is what the Triton path does and why its m=8 number is
// 13 GB/s. The eight levels of a dword are gathered once into lv[] and reused by
// every row; ROWK stays 32 so a lane's block is still inside one quantisation group.
//
// Packing M rows costs M times the staging work, so it has to be amortized over as
// many column groups as the single-row kernel uses: GB=32 keeps accg at GB*BN*M
// floats, which is 8 KB at M=8 and does not cost a block of occupancy (M=4 and M=8
// are already one block per CU on VGPRs).
//
// The fp32 staging above is what caps this kernel: one ds_read_b128 can only feed
// four FMAs, because one activation value takes exactly one multiply per output
// column. w4a16_gemv_h below trades that for fp16 activations.
#define WORDS (ROWK / 8)                  // weight dwords per lane per chunk

template <int M, bool CB, bool OUT_F16, bool HAS_BIAS>
__global__ void __launch_bounds__(NT, 1)
w4a16_gemv_m(const float* __restrict__ x, const uint32_t* __restrict__ wq,
             const uint8_t* __restrict__ srel, const float* __restrict__ sch,
             const void* __restrict__ bias, void* __restrict__ out, const float* cb,
             int m, int n, int k, int gs, int64_t srel_stride)
{
    __shared__ float xa[M][BK];
    __shared__ float tab[16];
    __shared__ float accg[GB * BN * M];
    const int tid = threadIdx.x;
    const int part = tid & (KP - 1);
    const int slot = tid / KP;
    const int groups = (n + BN - 1) / BN;
    const __half* bias_h = (const __half*)bias;
    const float* bias_f = (const float*)bias;

    if (tid < 16) tab[tid] = CB ? cb[tid] : (float)tid - 8.0f;
    __syncthreads();

    const int gi_total = (groups - blockIdx.x + gridDim.x - 1) / gridDim.x;
    for (int gb = 0; gb < gi_total; gb += GB) {
        const int nb = gi_total - gb < GB ? gi_total - gb : GB;
        __syncthreads();
        for (int i = tid; i < nb * BN * M; i += NT) accg[i] = 0.0f;

        for (int ck = 0; ck < k; ck += BK) {
            __syncthreads();
            for (int i = tid; i < M * BK; i += NT) {
                const int r = i / BK, kk = i - r * BK;
                const int lane = kk / ROWK, t = kk - lane * ROWK;
                xa[r][(t >> 2) * QUAD + lane * 4 + (t & 3)] =
                    r < m ? x[(int64_t)r * k + ck + kk] : 0.0f;
            }
            __syncthreads();

            const int kbase = ck + part * ROWK;
            const int64_t sbase = (int64_t)kbase / gs;
            for (int jl = 0; jl < nb; ++jl) {
                const int col = (blockIdx.x + (gb + jl) * gridDim.x) * BN + slot;
                const int colc = col < n ? col : n - 1;
                const uint32_t* wrow = wq + (int64_t)colc * (k / 8) + kbase / 8;
                float gacc[M][WORDS];
#pragma unroll
                for (int r = 0; r < M; ++r)
#pragma unroll
                    for (int i = 0; i < WORDS; ++i) gacc[r][i] = 0.0f;
                V4 t;
                t.v = *reinterpret_cast<const uint4*>(wrow);
#pragma unroll
                for (int i = 0; i < WORDS; ++i) {
                    const uint32_t even = t.w[i] & 0x0F0F0F0F;
                    const uint32_t odd = (t.w[i] >> 4) & 0x0F0F0F0F;
                    float lv[8];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        lv[2 * j] = tab[(even >> (8 * j)) & 0xFF];
                        lv[2 * j + 1] = tab[(odd >> (8 * j)) & 0xFF];
                    }
                    const int k0 = part * 4 + (i * 2) * QUAD;
#pragma unroll
                    for (int r = 0; r < M; ++r) {
                        const float4 q0 = *reinterpret_cast<const float4*>(xa[r] + k0);
                        const float4 q1 = *reinterpret_cast<const float4*>(xa[r] + k0 + QUAD);
                        const float xv[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
                        for (int j = 0; j < 8; ++j) gacc[r][i] += lv[j] * xv[j];
                    }
                }
                const float sg = srel_decode(srel[(int64_t)colc * srel_stride + sbase]);
#pragma unroll
                for (int r = 0; r < M; ++r) {
                    float s = 0.0f;
#pragma unroll
                    for (int i = 0; i < WORDS; ++i) s += gacc[r][i];
                    s *= sg;
#pragma unroll
                    for (int step = 1; step < KP; step <<= 1) s += __shfl_xor(s, step);
                    if (part == 0) accg[(jl * BN + slot) * M + r] += s;
                }
            }
        }

        for (int jl = 0; jl < nb; ++jl) {
            const int col = (blockIdx.x + (gb + jl) * gridDim.x) * BN + slot;
            if (part == 0 && col < n) {
                const float sc = sch[col];
#pragma unroll
                for (int r = 0; r < M; ++r) {
                    if (r >= m) break;
                    float v = accg[(jl * BN + slot) * M + r] * sc;
                    if (HAS_BIAS) v += OUT_F16 ? (float)bias_h[col] : bias_f[col];
                    if (OUT_F16) ((__half*)out)[(int64_t)r * n + col] = (__half)v;
                    else ((float*)out)[(int64_t)r * n + col] = v;
                }
            }
        }
    }
}

// fp16 activation variant, the default feed unless the checkpoint carries a codebook
// (COMFY_GFX906_W4A16_F16ACT=1 allows it there too). Same
// shape as w4a16_gemv_m, but the staged activations are fp16 and the math is
// v_dot2_f32_f16. That breaks the cap the fp32 variant hits: a quad row now holds
// 8 k per lane, so one ds_read_b128 feeds four dot2 = eight MACs instead of four
// FMAs, and because the level table is indexed by the *byte* (both nibbles of one
// k pair packed into a single fp16x2 operand) the gathers halve as well.
//
// The price is that the 16 codebook levels are rounded to fp16: a systematic
// ~5e-4 relative error on every code, identical wherever that code appears, so it
// does not average out over k. The uniform `code - 8` grid is exact in fp16.
// Accumulators stay fp32 and neither scale path changes.
__device__ __forceinline__ float dot2f(uint32_t a, uint32_t b, float c)
{
    float d;
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    return d;
}

#define QUAD16 (KP * 8)                   // halfs per activation quad row: 8 k of each lane

template <int M, bool CB>
__global__ void __launch_bounds__(NT, 1)
w4a16_gemv_h(const __half* __restrict__ x, const uint32_t* __restrict__ wq,
             const uint8_t* __restrict__ srel, const float* __restrict__ sch,
             const void* __restrict__ bias, void* __restrict__ out, const float* cb,
             int m, int n, int k, int gs, int64_t srel_stride, int out_f16, int has_bias)
{
    __shared__ __half xh[M][BK];
    __shared__ uint32_t tab2[256];
    __shared__ float accg[GB * BN * M];
    const int tid = threadIdx.x;
    const int part = tid & (KP - 1);
    const int slot = tid / KP;
    const int groups = (n + BN - 1) / BN;
    const __half* bias_h = (const __half*)bias;
    const float* bias_f = (const float*)bias;

    for (int b = tid; b < 256; b += NT) {
        const float lo = CB ? cb[b & 15] : (float)(b & 15) - 8.0f;
        const float hi = CB ? cb[b >> 4] : (float)(b >> 4) - 8.0f;
        tab2[b] = __half_as_ushort(__float2half_rn(lo))
                | ((uint32_t)__half_as_ushort(__float2half_rn(hi)) << 16);
    }
    __syncthreads();

    const int gi_total = (groups - blockIdx.x + gridDim.x - 1) / gridDim.x;
    for (int gb = 0; gb < gi_total; gb += GB) {
        const int nb = gi_total - gb < GB ? gi_total - gb : GB;
        __syncthreads();
        for (int i = tid; i < nb * BN * M; i += NT) accg[i] = 0.0f;

        for (int ck = 0; ck < k; ck += BK) {
            __syncthreads();
            // Eight halfs at a time: the source k = lane*ROWK + 8i is contiguous
            // and so is its destination i*QUAD16 + lane*8, so the pack is uint4
            // copies rather than BK*M scalar stores. Rows past m get zeros so the
            // phantom rows cost multiplies only.
            for (int j = tid; j < M * WORDS * KP; j += NT) {
                const int r = j / (WORDS * KP), q = j - r * (WORDS * KP);
                const int lane = q & (KP - 1), i = q / KP;
                uint4* dst = (uint4*)&xh[r][i * QUAD16 + lane * 8];
                if (r < m) *dst = *(const uint4*)(x + (int64_t)r * k + ck + lane * ROWK + i * 8);
                else *dst = make_uint4(0u, 0u, 0u, 0u);
            }
            __syncthreads();

            const int kbase = ck + part * ROWK;
            const int64_t sbase = (int64_t)kbase / gs;
            for (int jl = 0; jl < nb; ++jl) {
                const int col = (blockIdx.x + (gb + jl) * gridDim.x) * BN + slot;
                const int colc = col < n ? col : n - 1;
                float gacc[M][WORDS];
#pragma unroll
                for (int r = 0; r < M; ++r)
#pragma unroll
                    for (int i = 0; i < WORDS; ++i) gacc[r][i] = 0.0f;
                V4 t;
                t.v = *reinterpret_cast<const uint4*>(wq + (int64_t)colc * (k / 8) + kbase / 8);
#pragma unroll
                for (int i = 0; i < WORDS; ++i) {
                    // Byte j of the dword holds the codes of k = 8i + 2j and
                    // k = 8i + 2j + 1, so it pairs directly with half-dword j of
                    // the activation quad row.
                    const uint32_t w = t.w[i];
                    const uint32_t lv[4] = {tab2[(w >> 0) & 0xFF], tab2[(w >> 8) & 0xFF],
                                            tab2[(w >> 16) & 0xFF], tab2[(w >> 24) & 0xFF]};
#pragma unroll
                    for (int r = 0; r < M; ++r) {
                        const uint4 av = *reinterpret_cast<const uint4*>(&xh[r][i * QUAD16 + part * 8]);
                        gacc[r][i] = dot2f(lv[0], av.x, gacc[r][i]);
                        gacc[r][i] = dot2f(lv[1], av.y, gacc[r][i]);
                        gacc[r][i] = dot2f(lv[2], av.z, gacc[r][i]);
                        gacc[r][i] = dot2f(lv[3], av.w, gacc[r][i]);
                    }
                }
                const float sg = srel_decode(srel[(int64_t)colc * srel_stride + sbase]);
#pragma unroll
                for (int r = 0; r < M; ++r) {
                    float s = 0.0f;
#pragma unroll
                    for (int i = 0; i < WORDS; ++i) s += gacc[r][i];
                    s *= sg;
#pragma unroll
                    for (int step = 1; step < KP; step <<= 1) s += __shfl_xor(s, step);
                    if (part == 0) accg[(jl * BN + slot) * M + r] += s;
                }
            }
        }

        for (int jl = 0; jl < nb; ++jl) {
            const int col = (blockIdx.x + (gb + jl) * gridDim.x) * BN + slot;
            if (part == 0 && col < n) {
                const float sc = sch[col];
#pragma unroll
                for (int r = 0; r < M; ++r) {
                    if (r >= m) break;
                    float v = accg[(jl * BN + slot) * M + r] * sc;
                    // Output dtype and bias are runtime choices here: this kernel is
                    // instantiated over M x CB, and a store that runs once per column
                    // group cannot afford to double that for them.
                    if (has_bias) v += out_f16 ? (float)bias_h[col] : bias_f[col];
                    if (out_f16) ((__half*)out)[(int64_t)r * n + col] = (__half)v;
                    else ((float*)out)[(int64_t)r * n + col] = v;
                }
            }
        }
    }
}

extern "C" int gfx906_w4a16(const void* x, const void* wq, const void* srel, const void* sch,
                           const void* bias, void* out, const void* cb, int n, int k, int gs,
                           int in_f16, int out_f16, int has_cb, int has_bias,
                           int64_t srel_stride, void* stream)
{
    if (n < 1 || k % BK || gs < 32 || gs % 32 || BK % gs) return -3;
    const int groups = (n + BN - 1) / BN;
    const int grid = groups < MAX_BLOCKS ? groups : MAX_BLOCKS;
    hipStream_t st = (hipStream_t)stream;
    const float* cbw = (const float*)cb;
#define LAUNCH(IF, CB, OF, HB)                                                    \
    w4a16_gemv<IF, CB, OF, HB><<<grid, NT, 0, st>>>(x,                            \
        (const uint32_t*)wq, (const uint8_t*)srel, (const float*)sch, bias, out,   \
        cbw, n, k, gs, srel_stride)
#define DISPATCH(IOF, CB, HB)                                     \
    do {                                                          \
        if (in_f16) { LAUNCH(true, CB, IOF, HB); }                  \
        else { LAUNCH(false, CB, IOF, HB); }                        \
    } while (0)
#define DISPATCH_CB(IOF, HB)                                            \
    do {                                                                \
        if (has_cb) DISPATCH(IOF, true, HB); else DISPATCH(IOF, false, HB); \
    } while (0)

    if (out_f16) { if (has_bias) DISPATCH_CB(true, true); else DISPATCH_CB(true, false); }
    else { if (has_bias) DISPATCH_CB(false, true); else DISPATCH_CB(false, false); }
#undef DISPATCH_CB
#undef DISPATCH
#undef LAUNCH
    return hipGetLastError() == hipSuccess ? 0 : (int)hipGetLastError();
}

// m rows of activations against the same weight plane. M is m rounded up to 2/4/8
// so the row loop unrolls; the phantom rows stage zeros and are dropped at the store.
extern "C" int gfx906_w4a16_m(const void* x, const void* wq, const void* srel, const void* sch,
                              const void* bias, void* out, const void* cb, int m, int n, int k,
                              int gs, int out_f16, int has_cb, int has_bias,
                              int64_t srel_stride, void* stream)
{
    if (m < 2 || m > 8 || n < 1 || k % BK || gs < 32 || gs % 32 || BK % gs) return -3;
    const int M = m > 4 ? 8 : m > 2 ? 4 : 2;
    const int groups = (n + BN - 1) / BN;
    const int grid = groups < MAX_BLOCKS ? groups : MAX_BLOCKS;
    hipStream_t st = (hipStream_t)stream;
    const float* cbw = (const float*)cb;
#define LAUNCHM(MM, CB, OF, HB)                                                      \
    w4a16_gemv_m<MM, CB, OF, HB><<<grid, NT, 0, st>>>((const float*)x,               \
        (const uint32_t*)wq, (const uint8_t*)srel, (const float*)sch, bias, out,     \
        cbw, m, n, k, gs, srel_stride)
#define DISPATCHM(MM, CB, IOF)                                                       \
    do {                                                                             \
        if (has_bias) { LAUNCHM(MM, CB, IOF, true); } else { LAUNCHM(MM, CB, IOF, false); } \
    } while (0)
#define DISPATCHM_CB(MM, IOF)                                                        \
    do {                                                                             \
        if (has_cb) DISPATCHM(MM, true, IOF); else DISPATCHM(MM, false, IOF);        \
    } while (0)
#define DISPATCHM_ALL(MM)                                                            \
    do {                                                                             \
        if (out_f16) { DISPATCHM_CB(MM, true); } else { DISPATCHM_CB(MM, false); }   \
    } while (0)

    if (M == 2) DISPATCHM_ALL(2);
    else if (M == 4) DISPATCHM_ALL(4);
    else DISPATCHM_ALL(8);
#undef DISPATCHM_ALL
#undef DISPATCHM_CB
#undef DISPATCHM
#undef LAUNCHM
    return hipGetLastError() == hipSuccess ? 0 : (int)hipGetLastError();
}

// Same contract as gfx906_w4a16_m down to m = 1, but x is the *fp16* activation
// rows (no conversion happens here) and the levels resolve through the packed
// fp16 table. At m = 1 this is the single-row decode with M = 1, not a padded M.
extern "C" int gfx906_w4a16_h(const void* x, const void* wq, const void* srel, const void* sch,
                              const void* bias, void* out, const void* cb, int m, int n, int k,
                              int gs, int out_f16, int has_cb, int has_bias,
                              int64_t srel_stride, void* stream)
{
    if (m < 1 || m > 8 || n < 1 || k % BK || gs < 32 || gs % 32 || BK % gs) return -3;
    const int M = m > 4 ? 8 : m > 2 ? 4 : m > 1 ? 2 : 1;
    const int groups = (n + BN - 1) / BN;
    const int grid = groups < MAX_BLOCKS ? groups : MAX_BLOCKS;
    hipStream_t st = (hipStream_t)stream;
    const float* cbw = (const float*)cb;
#define LAUNCHH(MM, CB)                                                                \
    w4a16_gemv_h<MM, CB><<<grid, NT, 0, st>>>((const __half*)x,                        \
        (const uint32_t*)wq, (const uint8_t*)srel, (const float*)sch, bias, out,       \
        cbw, m, n, k, gs, srel_stride, out_f16, has_bias)
#define DISPATCHH(MM)                                                                  \
    do {                                                                               \
        if (has_cb) LAUNCHH(MM, true); else LAUNCHH(MM, false);                        \
    } while (0)

    if (M == 1) DISPATCHH(1);
    else if (M == 2) DISPATCHH(2);
    else if (M == 4) DISPATCHH(4);
    else DISPATCHH(8);
#undef DISPATCHH
#undef LAUNCHH
    return hipGetLastError() == hipSuccess ? 0 : (int)hipGetLastError();
}
