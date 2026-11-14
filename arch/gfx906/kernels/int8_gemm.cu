// gfx906 int8 GEMM prototype: int32 dot4, fp32 dequant epilogue.
//
// Contract (row-major):
//   x8 [M, K] int8, xs [M] f32  (per-row activation scale)
//   w8 [N, K] int8, ws [N] f32  (per-channel weight scale)
//   bias [N] f32 or null, out [M, N] fp16
//   out[m][n] = (sum_k x8[m][k] * w8[n][k]) * xs[m] * ws[n] + bias[n]
//   v2: resid [M, N] (same dtype as out) and rscale [N] f32 add
//   out[m][n] = resid[m][n] + rscale[n] * out[m][n]
//
// LDS tiles are chunk-major: [buffer][16-byte K chunk][row][16 bytes]. A phase
// of 16 lanes then reads consecutive rows (consecutive 16 B slots, all banks)
// instead of rows one LDS stride apart, which aliased badly: the first version
// with padded rows ran at 3.4 TMAC/s and this layout is what fixed it.
// Each lane owns columns tx + 16*j so the same holds for the B operand, and
// the epilogue store stays coalesced (16 lanes write 16 consecutive halves).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>

#define BK 64
#ifndef GM
#define GM 32
#endif
#define NT 256
#define TM 8
#define TN 8
#define LBPB 1
#define BM (TM * (NT / 16))
#define BN (TN * 16)
#define TSTRIDE 16
#define APT (NT / BM)                 // A threads per row
#define ABYTES (BK / APT)             // A bytes per thread
#define BPT (NT / BN)
#define BBYTES (BK / BPT)
#define NCH (BK / 16)

// store_a/store_b stage whole 16-byte chunk planes with uint4 stores.
static_assert(ABYTES % 16 == 0 && BBYTES % 16 == 0, "per-thread K bytes must be whole chunk planes");

__device__ __forceinline__ int dot4(uint32_t a, uint32_t b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, 0);
}
__host__ __device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__host__ __device__ __forceinline__ int cdiv(int a, int b) { return (a + b - 1) / b; }

// Grouped tile order (Triton's group_size_m).
static __host__ __device__ void tile_index(int pid, int tiles_m, int tiles_n, int gm,
                                           int* out_m, int* out_n) {
    const int group_sz = gm * tiles_n;
    const int group = pid / group_sz;
    const int first_m = group * gm;
    const int gsz = imin(tiles_m - first_m, gm);
    const int idx = pid - group * group_sz;
    *out_m = first_m + idx % gsz;
    *out_n = idx / gsz;
}

template <typename OutT>
__device__ __forceinline__ OutT to_out(float v);
template <>
__device__ __forceinline__ __half to_out<__half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ hip_bfloat16 to_out<hip_bfloat16>(float v) {
    return hip_bfloat16(v);
}
template <>
__device__ __forceinline__ float to_out<float>(float v) { return v; }

template <typename OutT, bool ALIGNED_K>
__global__ void __launch_bounds__(NT, LBPB)
int8_gemm_kernel(const int8_t* __restrict__ x8, const float* __restrict__ xs,
                 const int8_t* __restrict__ w8, const float* __restrict__ ws, int ws_scalar,
                 const float* __restrict__ bias, const OutT* __restrict__ resid,
                 const float* __restrict__ rscale, OutT* __restrict__ out,
                 int M, int N, int K) {
    __shared__ __align__(16) int8_t As[2][NCH][BM][16];
    __shared__ __align__(16) int8_t Bs[2][NCH][BN][16];

    const int tiles_m = cdiv(M, BM), tiles_n = cdiv(N, BN);
    int bm, bn;
    tile_index(blockIdx.x, tiles_m, tiles_n, GM, &bm, &bn);
    const int m0 = bm * BM, n0 = bn * BN;
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int r0 = ty * TM;

    const int arow = threadIdx.x / APT, aq = threadIdx.x % APT;
    const int brow = threadIdx.x / BPT, bq = threadIdx.x % BPT;
    const int aseg = aq * ABYTES, bseg = bq * BBYTES;

    // A thread's ABYTES bytes span ABYTES/16 whole chunk planes, so stage each
    // plane with one uint4 store instead of ABYTES/4 scattered 4-byte stores.
    auto store_a = [&](int buf, const uint32_t* src) {
#pragma unroll
        for (int i = 0; i < ABYTES / 16; ++i)
            *(uint4*)&As[buf][aseg / 16 + i][arow][0] = ((const uint4*)src)[i];
    };
    auto store_b = [&](int buf, const uint32_t* src) {
#pragma unroll
        for (int i = 0; i < BBYTES / 16; ++i)
            *(uint4*)&Bs[buf][bseg / 16 + i][brow][0] = ((const uint4*)src)[i];
    };

    const int nk = cdiv(K, BK);

    auto load_a = [&](int kt, uint32_t* dst) {
        const int gr = imin(m0 + arow, M - 1);
        const int8_t* p = x8 + (int64_t)gr * K + kt * BK + aq * ABYTES;
        // ALIGNED_K (K % BK == 0) drops the tail masking entirely: the tail path
        // is dead code there and costs ~6% in the hot loop otherwise.
        if (ALIGNED_K || kt * BK + aq * ABYTES + ABYTES <= K) {
#pragma unroll
            for (int i = 0; i < ABYTES / 4; ++i) dst[i] = ((const uint32_t*)p)[i];
        } else {
            int8_t tmp[ABYTES];
#pragma unroll
            for (int i = 0; i < ABYTES; ++i)
                tmp[i] = (kt * BK + aq * ABYTES + i < K) ? p[i] : 0;
#pragma unroll
            for (int i = 0; i < ABYTES / 4; ++i) dst[i] = ((const uint32_t*)tmp)[i];
        }
    };
    auto load_b = [&](int kt, uint32_t* dst) {
        const int gr = imin(n0 + brow, N - 1);
        const int8_t* p = w8 + (int64_t)gr * K + kt * BK + bq * BBYTES;
        if (ALIGNED_K || kt * BK + bq * BBYTES + BBYTES <= K) {
#pragma unroll
            for (int i = 0; i < BBYTES / 4; ++i) dst[i] = ((const uint32_t*)p)[i];
        } else {
            int8_t tmp[BBYTES];
#pragma unroll
            for (int i = 0; i < BBYTES; ++i)
                tmp[i] = (kt * BK + bq * BBYTES + i < K) ? p[i] : 0;
#pragma unroll
            for (int i = 0; i < BBYTES / 4; ++i) dst[i] = ((const uint32_t*)tmp)[i];
        }
    };

    int acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;

    uint32_t pa[ABYTES / 4], pb[BBYTES / 4];
    load_a(0, pa);
    load_b(0, pb);
    store_a(0, pa);
    store_b(0, pb);
    __syncthreads();

    for (int kt = 0; kt < nk; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < nk) {
            load_a(kt + 1, pa);
            load_b(kt + 1, pb);
        }
        // Read operands from LDS one k-dword at a time, the TN B dwords reused
        // across all TM rows, so only TN+1 operand registers are live instead of
        // a u4[TM]+u4[TN] array. With the s loop unrolled 2x this lands at 124
        // VGPR / 2 blocks/CU (8 waves); the b128-array form needed 211 VGPR and
        // ran at 4 waves. Full unroll or __launch_bounds__(.,2) both spill the 64
        // accumulators, so keep LBPB=1 and let the allocator fit naturally.
        for (int half = 0; half < NCH; ++half) {
#pragma unroll 2
            for (int s = 0; s < 4; ++s) {
                uint32_t b[TN];
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    b[j] = *(const uint32_t*)&Bs[buf][half][tx + TSTRIDE * j][s * 4];
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const uint32_t a = *(const uint32_t*)&As[buf][half][r0 + i][s * 4];
#pragma unroll
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] = dot4(a, b[j], acc[i][j]);
                }
            }
        }
        if (kt + 1 < nk) {
            store_a(buf ^ 1, pa);
            store_b(buf ^ 1, pb);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gm = m0 + r0 + i;
        if (gm >= M) continue;
        const float sa = xs[gm];
        OutT* dst = out + (int64_t)gm * N + n0 + tx;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gn = n0 + tx + TSTRIDE * j;
            if (gn >= N) continue;
            float v = (float)acc[i][j] * sa * ws[ws_scalar ? 0 : gn];
            if (bias) v += bias[gn];
            // residual + rscale[n] * out, the same form comfy_kitchen's
            // _apply_residual uses; null keeps v1 bit-identical.
            if (resid) v = (float)resid[(int64_t)gm * N + gn] + v * rscale[gn];
            dst[TSTRIDE * j] = to_out<OutT>(v);
        }
    }
}

template <typename OutT>
static void launch_int8(const void* x8, const void* xs, const void* w8, const void* ws,
                        int ws_scalar, const void* bias, void* out, int M, int N, int K,
                        int tiles, const void* resid, const void* rscale, hipStream_t st) {
    if (K % BK == 0)
        int8_gemm_kernel<OutT, true><<<tiles, NT, 0, st>>>(
            (const int8_t*)x8, (const float*)xs, (const int8_t*)w8, (const float*)ws, ws_scalar,
            (const float*)bias, (const OutT*)resid, (const float*)rscale, (OutT*)out, M, N, K);
    else
        int8_gemm_kernel<OutT, false><<<tiles, NT, 0, st>>>(
            (const int8_t*)x8, (const float*)xs, (const int8_t*)w8, (const float*)ws, ws_scalar,
            (const float*)bias, (const OutT*)resid, (const float*)rscale, (OutT*)out, M, N, K);
}

extern "C" int gfx906_int8_gemm(const void* x8, const void* xs, const void* w8,
                                const void* ws, int ws_scalar, const void* bias, void* out,
                                int M, int N, int K, int out_dtype, void* stream) {
    if (M <= 0 || N <= 0 || K <= 0) return -2;
    const int tiles = cdiv(M, BM) * cdiv(N, BN);
    hipStream_t st = (hipStream_t)stream;
    if (out_dtype == 2)
        launch_int8<float>(x8, xs, w8, ws, ws_scalar, bias, out, M, N, K, tiles, nullptr, nullptr, st);
    else if (out_dtype == 1)
        launch_int8<hip_bfloat16>(x8, xs, w8, ws, ws_scalar, bias, out, M, N, K, tiles, nullptr, nullptr, st);
    else
        launch_int8<__half>(x8, xs, w8, ws, ws_scalar, bias, out, M, N, K, tiles, nullptr, nullptr, st);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : (int)e;
}

// v2 adds the optional residual epilogue (residual + rscale[n] * out). Kept as a
// separate export so an older library on disk still serves gfx906_int8_gemm.
extern "C" int gfx906_int8_gemm_v2(const void* x8, const void* xs, const void* w8,
                                   const void* ws, int ws_scalar, const void* bias, void* out,
                                   int M, int N, int K, int out_dtype,
                                   const void* resid, const void* rscale, void* stream) {
    if (M <= 0 || N <= 0 || K <= 0) return -2;
    const int tiles = cdiv(M, BM) * cdiv(N, BN);
    hipStream_t st = (hipStream_t)stream;
    if (out_dtype == 2)
        launch_int8<float>(x8, xs, w8, ws, ws_scalar, bias, out, M, N, K, tiles, resid, rscale, st);
    else if (out_dtype == 1)
        launch_int8<hip_bfloat16>(x8, xs, w8, ws, ws_scalar, bias, out, M, N, K, tiles, resid, rscale, st);
    else
        launch_int8<__half>(x8, xs, w8, ws, ws_scalar, bias, out, M, N, K, tiles, resid, rscale, st);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : (int)e;
}
