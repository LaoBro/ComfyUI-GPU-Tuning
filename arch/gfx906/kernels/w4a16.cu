// gfx906 W4A16 decode GEMV: packed int4 weights decoded in registers, fp16
// activations, fp32 accumulate. One row (m == 1) per call, which is the te=default
// decode path for 4-bit checkpoints (w4a8_gemv.linear_dequant); multi-row MTP
// batches stay on the Triton kernel.
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
