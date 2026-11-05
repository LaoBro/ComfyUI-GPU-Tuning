// gfx906 int8 GEMM prototype: int32 dot4, fp32 dequant epilogue.
//
// Contract (row-major):
//   x8 [M, K] int8, xs [M] f32  (per-row activation scale)
//   w8 [N, K] int8, ws [N] f32  (per-channel weight scale)
//   bias [N] f32 or null, out [M, N] fp16
//   out[m][n] = (sum_k x8[m][k] * w8[n][k]) * xs[m] * ws[n] + bias[n]
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

#define BK 32
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

using u4 = uint4;
using u2 = uint2;

__device__ __forceinline__ int dot4(uint32_t a, uint32_t b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, 0);
}
__host__ __device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__host__ __device__ __forceinline__ int cdiv(int a, int b) { return (a + b - 1) / b; }
__device__ __forceinline__ uint32_t pick(const u4& v, int k) {
    return k == 0 ? v.x : k == 1 ? v.y : k == 2 ? v.z : v.w;
}

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

template <typename OutT>
__global__ void __launch_bounds__(NT, LBPB)
int8_gemm_kernel(const int8_t* __restrict__ x8, const float* __restrict__ xs,
                 const int8_t* __restrict__ w8, const float* __restrict__ ws, int ws_scalar,
                 const float* __restrict__ bias, OutT* __restrict__ out,
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

    // one row's ABYTES bytes live in ABYTES/16 different chunk planes, so the
    // store is per 4 bytes with the chunk decoded from the byte offset
    auto store_a = [&](int buf, const uint32_t* src) {
#pragma unroll
        for (int i = 0; i < ABYTES / 4; ++i) {
            const int pos = aseg + i * 4;
            *(uint32_t*)&As[buf][pos / 16][arow][pos % 16] = src[i];
        }
    };
    auto store_b = [&](int buf, const uint32_t* src) {
#pragma unroll
        for (int i = 0; i < BBYTES / 4; ++i) {
            const int pos = bseg + i * 4;
            *(uint32_t*)&Bs[buf][pos / 16][brow][pos % 16] = src[i];
        }
    };

    const int nk = cdiv(K, BK);

    auto load_a = [&](int kt, uint32_t* dst) {
        const int gr = imin(m0 + arow, M - 1);
        const int8_t* p = x8 + (int64_t)gr * K + kt * BK + aq * ABYTES;
        if (kt * BK + aq * ABYTES + ABYTES <= K) {
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
        if (kt * BK + bq * BBYTES + BBYTES <= K) {
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
#pragma unroll
        for (int half = 0; half < NCH; ++half) {
            u4 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = *(const u4*)&As[buf][half][r0 + i][0];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = *(const u4*)&Bs[buf][half][tx + TSTRIDE * j][0];
#pragma unroll
            for (int s = 0; s < 4; ++s)
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] = dot4(pick(a[i], s), pick(b[j], s), acc[i][j]);
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
            dst[TSTRIDE * j] = to_out<OutT>(v);
        }
    }
}

extern "C" int gfx906_int8_gemm(const void* x8, const void* xs, const void* w8,
                                const void* ws, int ws_scalar, const void* bias, void* out,
                                int M, int N, int K, int out_bf16, void* stream) {
    if (M <= 0 || N <= 0 || K <= 0) return -2;
    const int tiles = cdiv(M, BM) * cdiv(N, BN);
    hipStream_t st = (hipStream_t)stream;
    if (out_bf16) {
        int8_gemm_kernel<hip_bfloat16><<<tiles, NT, 0, st>>>(
            (const int8_t*)x8, (const float*)xs, (const int8_t*)w8, (const float*)ws, ws_scalar,
            (const float*)bias, (hip_bfloat16*)out, M, N, K);
    } else {
        int8_gemm_kernel<__half><<<tiles, NT, 0, st>>>(
            (const int8_t*)x8, (const float*)xs, (const int8_t*)w8, (const float*)ws, ws_scalar,
            (const float*)bias, (__half*)out, M, N, K);
    }
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : (int)e;
}
