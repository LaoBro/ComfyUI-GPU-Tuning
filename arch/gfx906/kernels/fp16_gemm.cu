// gfx906 fp16 GEMM: v_dot2_f32_f16, fp32 accumulate, fp16 out.
//
// Contract (row-major), matching nn.Linear so the weight needs no transpose:
//   x [M, K] fp16, w [N, K] fp16, bias [N] fp16 or null, out [M, N] fp16
//   out[m][n] = sum_k x[m][k] * w[n][k] + bias[n]
//
// Same register-budget design as int8_gemm.cu: 128x128 tile, 64 fp32
// accumulators, operands kept in LDS and read one k-pair at a time (the TN W
// dwords reused across all TM rows), s loop unrolled 2x. That lands under the
// 128-VGPR budget for 2 blocks/CU (8 waves); materializing operand arrays or
// forcing __launch_bounds__(.,2) spills. ALIGNED_K (K % BK == 0) drops the tail
// masking. See docs/HIP_KERNELS.md and tools/occupancy_audit.py.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
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
#define APT (NT / BM)
#define ABYTES (BK * 2 / APT)         // bytes of fp16 per thread per K tile
#define BPT (NT / BN)
#define BBYTES (BK * 2 / BPT)
#define NCH (BK / 8)                  // 16-byte chunk = 8 fp16 = 8 k

// chunk planes must be whole for the uint4 LDS stores
static_assert(ABYTES % 16 == 0 && BBYTES % 16 == 0, "per-thread K bytes must be whole chunk planes");

typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2(*(const f16x2*)&a, *(const f16x2*)&b, c, false);
}
__host__ __device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__host__ __device__ __forceinline__ int cdiv(int a, int b) { return (a + b - 1) / b; }

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

template <bool ALIGNED_K>
__global__ void __launch_bounds__(NT, LBPB)
f16_gemm_kernel(const __half* __restrict__ x, const __half* __restrict__ w,
                const __half* __restrict__ bias, __half* __restrict__ out,
                int M, int N, int K) {
    __shared__ __align__(16) int8_t As[2][NCH][BM][16];
    __shared__ __align__(16) int8_t Ws[2][NCH][BN][16];

    const int tiles_m = cdiv(M, BM), tiles_n = cdiv(N, BN);
    int bm, bn;
    tile_index(blockIdx.x, tiles_m, tiles_n, GM, &bm, &bn);
    const int m0 = bm * BM, n0 = bn * BN;
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int r0 = ty * TM;

    const int arow = threadIdx.x / APT, aq = threadIdx.x % APT;
    const int brow = threadIdx.x / BPT, bq = threadIdx.x % BPT;
    const int aseg = aq * ABYTES, bseg = bq * BBYTES;

    auto store_a = [&](int buf, const uint32_t* src) {
#pragma unroll
        for (int i = 0; i < ABYTES / 16; ++i)
            *(uint4*)&As[buf][aseg / 16 + i][arow][0] = ((const uint4*)src)[i];
    };
    auto store_b = [&](int buf, const uint32_t* src) {
#pragma unroll
        for (int i = 0; i < BBYTES / 16; ++i)
            *(uint4*)&Ws[buf][bseg / 16 + i][brow][0] = ((const uint4*)src)[i];
    };

    const int nk = cdiv(K, BK);
    const int AK = ABYTES / 2, BKb = BBYTES / 2;   // fp16 elements per thread

    auto load_a = [&](int kt, uint32_t* dst) {
        const int gr = imin(m0 + arow, M - 1);
        const int k0 = kt * BK + aq * AK;
        const __half* p = x + (int64_t)gr * K + k0;
        if (ALIGNED_K || k0 + AK <= K) {
#pragma unroll
            for (int i = 0; i < ABYTES / 4; ++i) dst[i] = ((const uint32_t*)p)[i];
        } else {
            __half tmp[ABYTES / 2];
#pragma unroll
            for (int i = 0; i < ABYTES / 2; ++i) tmp[i] = (k0 + i < K) ? p[i] : __half(0.f);
#pragma unroll
            for (int i = 0; i < ABYTES / 4; ++i) dst[i] = ((const uint32_t*)tmp)[i];
        }
    };
    auto load_b = [&](int kt, uint32_t* dst) {
        const int gr = imin(n0 + brow, N - 1);
        const int k0 = kt * BK + bq * BKb;
        const __half* p = w + (int64_t)gr * K + k0;
        if (ALIGNED_K || k0 + BKb <= K) {
#pragma unroll
            for (int i = 0; i < BBYTES / 4; ++i) dst[i] = ((const uint32_t*)p)[i];
        } else {
            __half tmp[BBYTES / 2];
#pragma unroll
            for (int i = 0; i < BBYTES / 2; ++i) tmp[i] = (k0 + i < K) ? p[i] : __half(0.f);
#pragma unroll
            for (int i = 0; i < BBYTES / 4; ++i) dst[i] = ((const uint32_t*)tmp)[i];
        }
    };

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

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
        // One k-pair (dword = 2 fp16) at a time; the TN W dwords are reused
        // across all TM rows so only TN+1 operand registers stay live.
        for (int half = 0; half < NCH; ++half) {
#pragma unroll 2
            for (int s = 0; s < 4; ++s) {
                uint32_t b[TN];
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    b[j] = *(const uint32_t*)&Ws[buf][half][tx + TSTRIDE * j][s * 4];
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const uint32_t a = *(const uint32_t*)&As[buf][half][r0 + i][s * 4];
#pragma unroll
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] = dot2(a, b[j], acc[i][j]);
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
        __half* dst = out + (int64_t)gm * N + n0 + tx;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gn = n0 + tx + TSTRIDE * j;
            if (gn >= N) continue;
            float v = acc[i][j];
            if (bias) v += __half2float(bias[gn]);
            dst[TSTRIDE * j] = __float2half(v);
        }
    }
}

static void launch_f16(const void* x, const void* w, const void* bias, void* out,
                       int M, int N, int K, int tiles, hipStream_t st) {
    if (K % BK == 0)
        f16_gemm_kernel<true><<<tiles, NT, 0, st>>>(
            (const __half*)x, (const __half*)w, (const __half*)bias, (__half*)out, M, N, K);
    else
        f16_gemm_kernel<false><<<tiles, NT, 0, st>>>(
            (const __half*)x, (const __half*)w, (const __half*)bias, (__half*)out, M, N, K);
}

extern "C" int gfx906_f16_gemm(const void* x, const void* w, const void* bias, void* out,
                               int M, int N, int K, void* stream) {
    if (M <= 0 || N <= 0 || K <= 0) return -2;
    const int tiles = cdiv(M, BM) * cdiv(N, BN);
    launch_f16(x, w, bias, out, M, N, K, tiles, (hipStream_t)stream);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : (int)e;
}