// Decode-oriented MoE GEMM kernels for gfx950 with in-kernel NVFP4 weight decode.
//
// Orientation: MFMA src A = weights (16 output features), src B = activations
// (16 tokens), so a C lane holds 4 consecutive output features of one token.
//
// Weight formats (FMT):
//   F_MXFP4 : MXFP4 control (E8M0 per 32), activations MXFP8, 1 scaled MFMA / K128.
//   F_TERMS : NVFP4 exact. E4M3 per-16 scale = 2^e * (t0 + t1 - t2) / 8 with
//             power-of-two terms; each (term, 16-half) is one scaled MFMA with the
//             term as the E8M0 weight scale and a lane-masked activation copy per
//             half. t2 goes to a separate accumulator. 6 MFMA / K128, no data decode.
//   F_F16   : NVFP4 exact. fp4 -> f16 via v_cvt_scalef32_pk_f16_fp4, then times the
//             E4M3 scale in f16 (exact: <= 6 significant bits). Activations f16.
//   F_FP8   : NVFP4 lossy. fp4 * (1 + m/8) * 2^(e - E) rounded to fp8 e4m3 via a
//             per-half 8-entry LUT, E = max exponent of the two halves that share
//             one MFMA 32-block. Activations MXFP8. 1 scaled MFMA / K128.
//   F_BF16  : NVFP4 exact. fp4 -> f32 (v_cvt_scalef32_pk_f32_fp4), times the E4M3 scale
//             in f32 (v_pk_mul_f32), -> bf16 (v_cvt_pk_bf16_f32; exact, <= 6 significant
//             bits). Activations bf16 (no cast). Stage1 only; stage1 output is f16.
//
// Weight tile layout (all formats): [E][NTILE][KT][64 lanes][16 B]; lane l = 16 g + c
// holds row 16 nt + c, k in [128 kt + 32 g, +32) (raw packed fp4 bytes; F_FP8 uses
// an offline permutation, see prep code).
#include <hip/hip_runtime.h>
#include <stdint.h>

#define DEV static __device__ __forceinline__

typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v3i __attribute__((ext_vector_type(3)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef _Float16 v2h __attribute__((ext_vector_type(2)));
typedef __bf16 v2bf __attribute__((ext_vector_type(2)));
typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
typedef float v2f __attribute__((ext_vector_type(2)));

// F_MX8 is not a config format: F_F16 kernels run it for the expert p.mx8_e (MXFP8
// shared expert, own tile buffers w8 / w8_sc): fp8 -> f16 via v_cvt_scalef32_pk_f16_fp8
// with the E8M0 block scale (pre-offset per matrix so that values stay f16-normal; the
// offset is folded into inv_gs), f16 activations, plain f16 MFMA.
enum { F_MXFP4 = 0, F_TERMS = 1, F_F16 = 2, F_FP8 = 3, F_BF16 = 4, F_MX8 = 5 };
// 16-bit activation formats (X and W decoded to f16 / bf16, plain MFMA)
#define IS16(F) ((F) == F_F16 || (F) == F_BF16 || (F) == F_MX8)

struct Params
{
    const uint8_t* a;    // activations: fp8 [rows, K] or f16 [rows, K]
    const uint8_t* a_sc; // e8m0 [rows, K/32] (fp8 activations only)
    const uint8_t* w;    // weight tiles
    const uint8_t* w_sc; // weight scales (format specific layout)
    const float* inv_gs; // stage1: [E][2] (gate, up); stage2: [E]
    const int* sorted_ids;
    const float* sorted_w;
    const int* sorted_eids;
    const int* num_valid;
    uint8_t* out;    // stage1: act2 fp8/f16 [T*topk, I]; stage2: bf16 [T, H]
    uint8_t* out_sc; // stage1 fp8: e8m0 [T*topk, I/32]
    int T, topk;
    int K;       // reduction length
    int ntiles;  // weight row tiles per expert (stage1: 2I/16, stage2: H/16)
    int n_out;   // stage1: I, stage2: H
    int mx8_e;   // expert with MXFP8 weights in w8 / w8_sc (F_F16 kernels), -1: none
    const uint8_t* w8;    // MXFP8 tiles of expert mx8_e: [NTILE][KT][2][64 lanes][16 B]
    const uint8_t* w8_sc; // its E8M0 bytes: [NTILE][KT][64 lanes]
    uint16_t* zo; // DEC_XBF1 stage1: bf16 output to zero (zn16 x 16 B, split over all WGs)
    int zn16;
};

// GEMM block coordinates. DEC_XCD: WG L runs on XCD L % 8, so remap linear ids such that
// all N groups of one M block share an XCD (its X rows then stay in that L2). Needs
// gridDim.y % 8 == 0 (host pads; padded blocks exit on num_valid).
// DEC_XBF1: F16 stage1 reads the bf16 X (cast in registers) and zeroes the stage2 output
// (p.zo), so the prep launch is the sort only (dec_sort_quant xm = 3).
#ifndef DEC_XBF1
#define DEC_XBF1 0
#endif
#ifndef DEC_XCD
#define DEC_XCD 0
#endif
#if DEC_XCD == 2
__shared__ int s_bx;
#endif
DEV int blk_x()
{
#if DEC_XCD == 2
    return s_bx;
#endif
    if constexpr(DEC_XCD)
        return ((blockIdx.x + blockIdx.y * gridDim.x) >> 3) % gridDim.x;
    return blockIdx.x;
}
// DEC_XCD 2: the M blocks of the leading (heaviest, SORT_HEAVY) expert keep the default
// order, so its N block j stays on XCD j % 8 and its weights are reused in that L2. The
// other M blocks are cut into 8 contiguous chunks, one per XCD, with all N blocks of an M
// block on that XCD (X rows and light-expert weights stay in one L2). Every thread writes
// the same s_bx and reads its own write, so no barrier is needed. Returns -1 for idle WGs;
// the host pads gridDim.y by 8 more blocks.
DEV int blk_map(const Params& p, int BM, int nvalid)
{
    if constexpr(DEC_XCD != 2)
        return blockIdx.y;
    const int gx = gridDim.x;
    const int L  = blockIdx.x + blockIdx.y * gx;
    const int V  = (nvalid + BM - 1) / BM;
#if DEC_XCD_HT
    // timing experiment: shared routing, heavy expert = T tokens
    int H = (p.T + BM - 1) / BM;
#else
    const int e0 = p.sorted_eids[0];
    int H        = 0;
#pragma unroll
    for(int i = 0; i < 64; i++)
        H += i < V && p.sorted_eids[min(i, max(V - 1, 0))] == e0;
#endif
    H = H < 2 ? 0 : H;
    int x, mb;
    if(L < gx * H)
    {
        x  = L % gx;
        mb = L / gx;
    }
    else
    {
        const int Lb  = L - gx * H;
        const int idx = Lb >> 3;
        const int c   = Lb & 7;
        const int q   = idx / gx;
        const int c0  = H + c * (V - H) / 8;
        const int c1  = H + (c + 1) * (V - H) / 8;
        x             = idx % gx;
        mb            = c0 + q < c1 ? c0 + q : -1;
    }
#if DEC_XCD == 2
    s_bx = x;
#endif
    return mb;
}
DEV int blk_y()
{
    if constexpr(DEC_XCD)
    {
        const int L = blockIdx.x + blockIdx.y * gridDim.x;
        return ((L >> 3) / gridDim.x) * 8 + (L & 7);
    }
    return blockIdx.y;
}

// ---------------------------------------------------------------- MFMA helpers
template <int OPW>
DEV v4f mfma_s(v4i w, v8i x, v4f c, uint32_t sw, uint32_t sx)
{
    v8i wv = {w[0], w[1], w[2], w[3], 0, 0, 0, 0};
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(wv, x, c, 4, 0, OPW, sw, 0, sx);
}

template <int OPW>
DEV v4f mfma_s8(v8i w, v8i x, v4f c, uint32_t sw, uint32_t sx)
{
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(w, x, c, 0, 0, OPW, sw, 0, sx);
}

DEV v4f mfma_s_rt(int op, v4i w, v8i x, v4f c, uint32_t sw, uint32_t sx)
{
    // op is a compile-time constant after unrolling.
    switch(op)
    {
    case 0: return mfma_s<0>(w, x, c, sw, sx);
    case 1: return mfma_s<1>(w, x, c, sw, sx);
    case 2: return mfma_s<2>(w, x, c, sw, sx);
    default: return mfma_s<3>(w, x, c, sw, sx);
    }
}

DEV v4f mfma_s8_rt(int op, v8i w, v8i x, v4f c, uint32_t sw, uint32_t sx)
{
    switch(op)
    {
    case 0: return mfma_s8<0>(w, x, c, sw, sx);
    case 1: return mfma_s8<1>(w, x, c, sw, sx);
    case 2: return mfma_s8<2>(w, x, c, sw, sx);
    default: return mfma_s8<3>(w, x, c, sw, sx);
    }
}

DEV v2h cvt_fp4x2_f16(uint32_t d, int byte)
{
    switch(byte)
    {
    case 0: return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(d, 1.0f, 0);
    case 1: return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(d, 1.0f, 1);
    case 2: return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(d, 1.0f, 2);
    default: return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(d, 1.0f, 3);
    }
}
// Same with the block scale folded into the conversion (DEC_FOLD).
DEV v2h cvt_fp4x2_f16s(uint32_t d, int byte, float s)
{
    switch(byte)
    {
    case 0: return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(d, s, 0);
    case 1: return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(d, s, 1);
    case 2: return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(d, s, 2);
    default: return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(d, s, 3);
    }
}
#ifndef DEC_FOLD
#define DEC_FOLD 0
#endif

DEV v2f cvt_fp4x2_f32(uint32_t d, int byte)
{
    switch(byte)
    {
    case 0: return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(d, 1.0f, 0);
    case 1: return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(d, 1.0f, 1);
    case 2: return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(d, 1.0f, 2);
    default: return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(d, 1.0f, 3);
    }
}

// E8M0 byte LUTs for the NVFP4 term split, indexed by the E4M3 mantissa m.
// byte = 117 + log2(term in units of 1/8); "no term" = 1 (adds e -> 2^-111 or less).
//   m : t0  t1  -t2      m : t0  t1  -t2
//   0 :  8   -   -       4 :  8   4   -
//   1 :  8   1   -       5 : 16   1   4
//   2 :  8   2   -       6 : 16   -   2
//   3 :  8   4   1       7 : 16   -   1
#define T0_LO 0x78787878u
#define T0_HI 0x79797978u
#define T1_LO 0x77767501u
#define T1_HI 0x01017577u
#define T2_LO 0x75010101u
#define T2_HI 0x75767701u

// ---------------------------------------------------------------- tiles
template <int FMT>
struct XTile;
template <>
struct XTile<F_MXFP4>
{
    v8i v;
    uint32_t s;
};
template <>
struct XTile<F_TERMS> : XTile<F_MXFP4>
{
};
template <>
struct XTile<F_FP8> : XTile<F_MXFP4>
{
};
template <>
struct XTile<F_F16>
{
    typedef v8h vec;
    vec v[4];
};
template <>
struct XTile<F_BF16>
{
    typedef v8bf vec;
    vec v[4];
};
template <>
struct XTile<F_MX8> : XTile<F_F16>
{
};

template <int FMT>
struct WScale
{
    uint32_t s; // per lane, 2 K-tiles
};
template <>
struct WScale<F_FP8>
{
    uint32_t s[4]; // per column: 2 K-tiles x 8 halves (e4m3)
};

template <int FMT, int MT, int NT, int KC>
struct Chunk
{
    v4i w[KC][NT];
    WScale<FMT> ws[KC / 2][NT];
    XTile<FMT> x[KC][MT];
};
// MXFP8: 32 B of weights and one E8M0 byte per lane and K-tile.
template <int MT, int NT, int KC>
struct Chunk<F_MX8, MT, NT, KC>
{
    v4i w[KC][NT], w2[KC][NT];
    uint32_t ws[KC][NT];
    XTile<F_MX8> x[KC][MT];
};

// Bytes of weight scales per (expert, ntile, K-tile pair).
template <int FMT>
constexpr int ws_pair_bytes()
{
    return FMT == F_MXFP4 ? 128 : (FMT == F_FP8 ? 256 : 256);
}

#ifndef DEC_NT1
#define DEC_NT1 1
#endif
#ifndef DEC_NT2
#define DEC_NT2 1
#endif
// Buffer-resource loads: wave-uniform byte offsets go to SGPR soffset and the per-lane
// part to a 32-bit voffset, so no 64-bit per-load addresses are kept live.
typedef __amdgpu_buffer_rsrc_t rsrc_t;
DEV rsrc_t make_rsrc(const void* p)
{
    return __builtin_amdgcn_make_buffer_rsrc((void*)p, (short)0, (int)0xffffffffu, 0x00020000);
}
// Bounded resource: loads at byte offsets >= n return 0 without a memory access.
DEV rsrc_t make_rsrc(const void* p, int n)
{
    return __builtin_amdgcn_make_buffer_rsrc((void*)p, (short)0, n, 0x00020000);
}
// Weight-scale load aux bits: nt (2) as for the weights. Plain scale loads cost 25-35 us
// of stage1 time at T = 128..512 (SG layout or not; sc0 and sc1 do not help). X loads
// stay plain: X lines are reused by the WGs of all N groups (nt X loads are much slower).
#ifndef DEC_WS_AUX
#define DEC_WS_AUX 2
#endif
// DEC_EXP_XC: timing experiment only (wrong results): per X load instruction the 4 lanes of a
// row read 64 contiguous bytes (16 g + 64 q) instead of 4 x 16 B spread over 256 B.
#ifndef DEC_EXP_XC
#define DEC_EXP_XC 0
#endif
// DEC_EXP_XDIST: timing experiment, padding columns of the generic kernel read distinct
// real X rows (1) and their data is then zeroed (2).
#ifndef DEC_EXP_XDIST
#define DEC_EXP_XDIST 0
#endif
// DEC_XOOB: padding columns of the generic kernel read out of the X buffer range (zeros,
// no memory access) instead of row 0.
#ifndef DEC_XOOB
#define DEC_XOOB 0
#endif
#ifndef DEC_EXP_NOOUT
#define DEC_EXP_NOOUT 0
#endif
// DEC_EXP_X2L: timing experiment only (wrong results), F16 X lanes g = 2, 3 read the lines
// of g = 0, 1 (half the distinct X lines, same instructions and bytes per lane).
#ifndef DEC_EXP_X2L
#define DEC_EXP_X2L 0
#endif
// X load aux bits (timing experiments only)
#ifndef DEC_X_AUX
#define DEC_X_AUX 0
#endif
// Stage2 X (stage1 output) load aux bits, generic kernel
#ifndef DEC_X_AUX2
#define DEC_X_AUX2 DEC_X_AUX
#endif
// DEC_MBAUX (bit 0 stage1, bit 1 stage2, generic kernel): an expert with more than one M
// block loads its weights with aux DEC_MBAUX_V instead of nt, so the other blocks of the
// expert can hit in L2 / MALL.
#ifndef DEC_MBAUX
#define DEC_MBAUX 0
#endif
#ifndef DEC_MBAUX_V
#define DEC_MBAUX_V 0
#endif
// Stage2 weight load aux bits when DEC_NT2 (2 = nt, 3 = sc0 | nt as AITER opus "cache_b3")
#ifndef DEC_W2_AUX
#define DEC_W2_AUX 2
#endif
// Weight load aux bits: 2 = nt. Measured best with nt in both stages (stage2 time
// depends on the stage1 hint: s1 nt + s2 plain is the slowest combination).

// Load sources of one wave. Wave-uniform parts (wo, wso) are scalar.
template <int MT, int NT>
struct Src
{
    rsrc_t rw, rws, rx, rxs;
    int wo[NT], wso[NT]; // tile byte offsets within the expert (uniform)
    int wl, wsl;         // per-lane byte offsets
    int xo[MT], xso[MT]; // per-lane row offsets of X and X scales
#if DEC_EXP_XDIST == 2
    int xz[MT]; // timing experiment: 0 for padding columns (their X data is zeroed)
#endif
};

template <int FMT, int MT, int NT, int KC, int XA = DEC_X_AUX>
DEV void load_x(Chunk<FMT, MT, NT, KC>& c, const Src<MT, NT>& s, int kt);

// NOWS: weights only (scales come from load_sg registers).
template <int FMT, int MT, int NT, int KC, int W_AUX, bool NOWS = false>
DEV void load_w(Chunk<FMT, MT, NT, KC>& c, const Src<MT, NT>& s, int kt)
{
    if constexpr(NOWS)
    {
#pragma unroll
        for(int i = 0; i < KC; i++)
#pragma unroll
            for(int j = 0; j < NT; j++)
                c.w[i][j] = __builtin_amdgcn_raw_buffer_load_b128(s.rw, s.wl, s.wo[j] + (kt + i) * 1024, W_AUX);
    }
    else if constexpr(FMT == F_MX8)
    {
#pragma unroll
        for(int i = 0; i < KC; i++)
#pragma unroll
            for(int j = 0; j < NT; j++)
            {
                c.w[i][j]  = __builtin_amdgcn_raw_buffer_load_b128(s.rw, s.wl, s.wo[j] + (kt + i) * 2048, W_AUX);
                c.w2[i][j] = __builtin_amdgcn_raw_buffer_load_b128(s.rw, s.wl, s.wo[j] + (kt + i) * 2048 + 1024,
                                                                   W_AUX);
            }
#pragma unroll
        for(int i = 0; i < KC; i++)
#pragma unroll
            for(int j = 0; j < NT; j++)
                c.ws[i][j] = __builtin_amdgcn_raw_buffer_load_b8(s.rws, s.wsl, s.wso[j] + (kt + i) * 64, DEC_WS_AUX);
    }
    else
    {
#pragma unroll
        for(int i = 0; i < KC; i++)
#pragma unroll
            for(int j = 0; j < NT; j++)
                c.w[i][j] = __builtin_amdgcn_raw_buffer_load_b128(s.rw, s.wl, s.wo[j] + (kt + i) * 1024, W_AUX);
#pragma unroll
        for(int i = 0; i < KC / 2; i++)
#pragma unroll
            for(int j = 0; j < NT; j++)
            {
#if DEC_EXP_WSHIT
                // timing experiment only: scale loads always hit the first pair (wrong results)
                const int so = s.wso[j] + i * ws_pair_bytes<FMT>();
#else
                const int so = s.wso[j] + ((kt >> 1) + i) * ws_pair_bytes<FMT>();
#endif
                if constexpr(FMT == F_MXFP4)
#if DEC_EXP_NOWS
                    c.ws[i][j].s = 0x7f7fu ^ (so & 1);
#else
                    c.ws[i][j].s = __builtin_amdgcn_raw_buffer_load_b16(s.rws, s.wsl, so, DEC_WS_AUX);
#endif
                else if constexpr(FMT == F_FP8)
                {
                    v4i t = __builtin_amdgcn_raw_buffer_load_b128(s.rws, s.wsl, so, DEC_WS_AUX);
                    c.ws[i][j].s[0] = t[0];
                    c.ws[i][j].s[1] = t[1];
                    c.ws[i][j].s[2] = t[2];
                    c.ws[i][j].s[3] = t[3];
                }
                else
#if DEC_EXP_NOWS
                    // timing experiment only: no weight scale loads (wrong results)
                    c.ws[i][j].s = 0x38383838u ^ so;
#else
                    c.ws[i][j].s = __builtin_amdgcn_raw_buffer_load_b32(s.rws, s.wsl, so, DEC_WS_AUX);
#endif
            }
    }
}

// SG layout (F16 weight scales grouped by 6 K-tiles): [E][NTILE][KT/6][64 lanes][3 dwords],
// dword q of group G = the old K-tile pair 3 G + q. One b96 per (tile, group) replaces
// three b32 loads; the per-tile offset wso is the same as in the pair layout.
template <int MT, int NT, int NG>
DEV void load_sg(uint32_t (&sg)[3 * NG][NT], const Src<MT, NT>& s, int g0)
{
    const int l = (threadIdx.x & 63) * 12;
#pragma unroll
    for(int q = 0; q < NG; q++)
#pragma unroll
        for(int j = 0; j < NT; j++)
        {
#if DEC_EXP_SGHIT
            // timing experiment only: all groups read the first group of tile 0 (wrong results)
            const v3i t = __builtin_amdgcn_raw_buffer_load_b96(s.rws, l, 0, 0);
#else
            const v3i t = __builtin_amdgcn_raw_buffer_load_b96(s.rws, l, s.wso[j] + (g0 + q) * 768, DEC_WS_AUX);
#endif
            sg[3 * q][j]     = t[0];
            sg[3 * q + 1][j] = t[1];
            sg[3 * q + 2][j] = t[2];
        }
}

template <int FMT, int MT, int NT, int KC, int NP>
DEV void fill_sg(Chunk<FMT, MT, NT, KC>& c, const uint32_t (&sg)[NP][NT], int p0)
{
#pragma unroll
    for(int i = 0; i < KC / 2; i++)
#pragma unroll
        for(int j = 0; j < NT; j++)
            c.ws[i][j].s = sg[p0 + i][j];
}

template <int FMT, int MT, int NT, int KC, int W_AUX, int XA = DEC_X_AUX>
DEV void load_chunk(Chunk<FMT, MT, NT, KC>& c, const Src<MT, NT>& s, int kt)
{
    load_w<FMT, MT, NT, KC, W_AUX>(c, s, kt);
    load_x<FMT, MT, NT, KC, XA>(c, s, kt);
}

template <int FMT, int MT, int NT, int KC, int XA>
DEV void load_x(Chunk<FMT, MT, NT, KC>& c, const Src<MT, NT>& s, int kt)
{
#pragma unroll
    for(int i = 0; i < KC; i++)
#pragma unroll
        for(int m = 0; m < MT; m++)
        {
            // Padding columns read row 0; MFMA columns are independent and padded
            // tokens are never stored.
            if constexpr(FMT == F_BF16)
            {
#pragma unroll
                for(int q = 0; q < 4; q++)
                    c.x[i][m].v[q] = __builtin_bit_cast(
                        v8bf, __builtin_amdgcn_raw_buffer_load_b128(s.rx, s.xo[m], (kt + i) * 256 + 16 * q, XA));
            }
            else if constexpr(FMT == F_F16 || FMT == F_MX8)
            {
#if DEC_EXP_X2
                // timing experiment only: half of the X bytes (wrong results)
#pragma unroll
                for(int q = 0; q < 4; q++)
                    c.x[i][m].v[q] = q < 2 ? __builtin_bit_cast(
                        v8h, __builtin_amdgcn_raw_buffer_load_b128(s.rx, s.xo[m], (kt + i) * 256 + 16 * q, XA))
                                           : c.x[i][m].v[q - 2];
#else
#pragma unroll
                for(int q = 0; q < 4; q++)
                {
                    v4i t = __builtin_amdgcn_raw_buffer_load_b128(s.rx, s.xo[m], (kt + i) * 256 + (DEC_EXP_XC ? 64 : 16) * q, XA & 255);
#if DEC_EXP_XDIST == 2
                    t &= s.xz[m];
#endif
                    if constexpr((XA & 256) != 0)
                    {
                        // XA bit 8: X is bf16, cast to f16 in registers (as cast_f16_kernel)
#pragma unroll
                        for(int k = 0; k < 4; k++)
                        {
                            const float a = __uint_as_float((uint32_t)t[k] << 16);
                            const float b = __uint_as_float((uint32_t)t[k] & 0xffff0000u);
                            const v2h h   = {(_Float16)a, (_Float16)b};
                            t[k]          = __builtin_bit_cast(int, h);
                        }
                    }
                    c.x[i][m].v[q] = __builtin_bit_cast(v8h, t);
                }
#endif
            }
            else
            {
                v4i lo = __builtin_amdgcn_raw_buffer_load_b128(s.rx, s.xo[m], (kt + i) * 128, XA);
                v4i hi = __builtin_amdgcn_raw_buffer_load_b128(s.rx, s.xo[m], (kt + i) * 128 + 64, XA);
#if DEC_EXP_XDIST == 2
                lo &= s.xz[m];
                hi &= s.xz[m];
#endif
                c.x[i][m].v = v8i{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
                c.x[i][m].s = __builtin_amdgcn_raw_buffer_load_b8(s.rxs, s.xso[m], (kt + i) * 4, XA);
            }
        }
}

// fp8 e4m3 LUT decode of one fp4 dword (8 codes) to 2 dwords (8 fp8), given the
// 8-entry magnitude LUT (lo, hi). Offline layout puts element t of the 8 in nibble
// such that low nibbles -> elements 0..3 and high nibbles -> elements 4..7.
DEV v2i fp8_lut_decode(uint32_t d, uint32_t lut_lo, uint32_t lut_hi)
{
    uint32_t lo = __builtin_amdgcn_perm(lut_hi, lut_lo, d & 0x07070707u) | ((d << 4) & 0x80808080u);
    uint32_t hi = __builtin_amdgcn_perm(lut_hi, lut_lo, (d >> 4) & 0x07070707u) | (d & 0x80808080u);
    return v2i{(int)lo, (int)hi};
}

// LUT table in constant memory: [m 0..7][shift 0..15] -> 8 fp8 bytes (code magnitudes
// 0, .5, 1, 1.5, 2, 3, 4, 6 times (1 + m/8) * 2^-shift, RNE to e4m3).
__constant__ uint32_t c_fp8_lut[8 * 16 * 2];

template <int FMT, int MT, int NT, int KC>
DEV void compute_chunk(Chunk<FMT, MT, NT, KC>& c, v4f (&acc)[NT][MT], v4f (&accn)[NT][MT],
                       bool godd, int g, const uint2* lut)
{
#pragma unroll
    for(int i = 0; i < KC; i++)
    {
        const int par = i & 1;
        if constexpr(FMT == F_MXFP4)
        {
#pragma unroll
            for(int j = 0; j < NT; j++)
#pragma unroll
                for(int m = 0; m < MT; m++)
                    acc[j][m] = mfma_s_rt(par, c.w[i][j], c.x[i][m].v, acc[j][m], c.ws[i >> 1][j].s,
                                          c.x[i][m].s);
        }
        else if constexpr(FMT == F_TERMS)
        {
            v8i xh0[MT], xh1[MT];
#pragma unroll
            for(int m = 0; m < MT; m++)
#pragma unroll
                for(int q = 0; q < 8; q++)
                {
                    xh0[m][q] = godd ? 0 : c.x[i][m].v[q];
                    xh1[m][q] = godd ? c.x[i][m].v[q] : 0;
                }
            uint32_t t0[NT], t1[NT], t2[NT];
#pragma unroll
            for(int j = 0; j < NT; j++)
            {
                uint32_t s  = c.ws[i >> 1][j].s;
                uint32_t mm = s & 0x07070707u;
                uint32_t ee = (s >> 3) & 0x0F0F0F0Fu;
                t0[j]       = __builtin_amdgcn_perm(T0_HI, T0_LO, mm) + ee;
                t1[j]       = __builtin_amdgcn_perm(T1_HI, T1_LO, mm) + ee;
                t2[j]       = __builtin_amdgcn_perm(T2_HI, T2_LO, mm) + ee;
            }
#pragma unroll
            for(int h = 0; h < 2; h++)
            {
                const int op = 2 * par + h;
#pragma unroll
                for(int j = 0; j < NT; j++)
#pragma unroll
                    for(int m = 0; m < MT; m++)
                    {
                        v8i xv = h ? xh1[m] : xh0[m];
                        acc[j][m]  = mfma_s_rt(op, c.w[i][j], xv, acc[j][m], t0[j], c.x[i][m].s);
                        acc[j][m]  = mfma_s_rt(op, c.w[i][j], xv, acc[j][m], t1[j], c.x[i][m].s);
                        accn[j][m] = mfma_s_rt(op, c.w[i][j], xv, accn[j][m], t2[j], c.x[i][m].s);
                    }
            }
        }
        else if constexpr(FMT == F_F16)
        {
#pragma unroll
            for(int j = 0; j < NT; j++)
            {
                v2h sp = par ? __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(c.ws[i >> 1][j].s, 1.0f, true)
                             : __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(c.ws[i >> 1][j].s, 1.0f, false);
                v2h s0 = {sp[0], sp[0]};
                v2h s1 = {sp[1], sp[1]};
#pragma unroll
                for(int q = 0; q < 4; q++)
                {
                    uint32_t d = c.w[i][j][q];
#if DEC_NODEC
                    // timing experiment only: no decode (wrong results)
                    v8h wv = __builtin_bit_cast(
                        v8h, v4i{(int)d, (int)d ^ __builtin_bit_cast(int, sp), (int)d, (int)d});
#elif DEC_FOLD
                    // block scale applied by the conversion (f32 scale operand)
                    const float sf = q < 2 ? (float)sp[0] : (float)sp[1];
                    v2h e0         = cvt_fp4x2_f16s(d, 0, sf);
                    v2h e1         = cvt_fp4x2_f16s(d, 1, sf);
                    v2h e2         = cvt_fp4x2_f16s(d, 2, sf);
                    v2h e3         = cvt_fp4x2_f16s(d, 3, sf);
                    v8h wv         = {e0[0], e0[1], e1[0], e1[1], e2[0], e2[1], e3[0], e3[1]};
#else
                    v2h sc     = q < 2 ? s0 : s1;
                    v2h e0     = cvt_fp4x2_f16(d, 0) * sc;
                    v2h e1     = cvt_fp4x2_f16(d, 1) * sc;
                    v2h e2     = cvt_fp4x2_f16(d, 2) * sc;
                    v2h e3     = cvt_fp4x2_f16(d, 3) * sc;
                    v8h wv     = {e0[0], e0[1], e1[0], e1[1], e2[0], e2[1], e3[0], e3[1]};
#endif
#if DEC_EXP_MF1
                    // timing experiment only: one MFMA per K-tile (wrong results)
                    if(q == 0)
                    {
                        v4i wx = __builtin_bit_cast(v4i, wv) ^ v4i{(int)c.w[i][j][1], (int)c.w[i][j][2],
                                                                   (int)c.w[i][j][3], 0};
#pragma unroll
                        for(int m = 0; m < MT; m++)
                        {
                            v4i xx = __builtin_bit_cast(v4i, c.x[i][m].v[0]) ^ __builtin_bit_cast(v4i, c.x[i][m].v[1]) ^
                                     __builtin_bit_cast(v4i, c.x[i][m].v[2]) ^ __builtin_bit_cast(v4i, c.x[i][m].v[3]);
                            acc[j][m] = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                                __builtin_bit_cast(v8h, wx), __builtin_bit_cast(v8h, xx), acc[j][m], 0, 0, 0);
                        }
                    }
#else
#pragma unroll
                    for(int m = 0; m < MT; m++)
                        acc[j][m] =
                            __builtin_amdgcn_mfma_f32_16x16x32_f16(wv, c.x[i][m].v[q], acc[j][m], 0, 0, 0);
#endif
                }
            }
        }
        else if constexpr(FMT == F_MX8)
        {
#pragma unroll
            for(int j = 0; j < NT; j++)
            {
                // the convert uses only the exponent of the f32 scale
                const float sc = __uint_as_float(c.ws[i][j] << 23);
#pragma unroll
                for(int q = 0; q < 4; q++)
                {
                    const uint32_t d0 = q < 2 ? c.w[i][j][2 * q] : c.w2[i][j][2 * q - 4];
                    const uint32_t d1 = q < 2 ? c.w[i][j][2 * q + 1] : c.w2[i][j][2 * q - 3];
                    v2h e0 = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(d0, sc, false);
                    v2h e1 = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(d0, sc, true);
                    v2h e2 = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(d1, sc, false);
                    v2h e3 = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(d1, sc, true);
                    v8h wv = {e0[0], e0[1], e1[0], e1[1], e2[0], e2[1], e3[0], e3[1]};
#pragma unroll
                    for(int m = 0; m < MT; m++)
                        acc[j][m] =
                            __builtin_amdgcn_mfma_f32_16x16x32_f16(wv, c.x[i][m].v[q], acc[j][m], 0, 0, 0);
                }
            }
        }
        else if constexpr(FMT == F_BF16)
        {
#pragma unroll
            for(int j = 0; j < NT; j++)
            {
                v2f sp = par ? __builtin_amdgcn_cvt_scalef32_pk_f32_fp8(c.ws[i >> 1][j].s, 1.0f, true)
                             : __builtin_amdgcn_cvt_scalef32_pk_f32_fp8(c.ws[i >> 1][j].s, 1.0f, false);
#pragma unroll
                for(int q = 0; q < 4; q++)
                {
                    uint32_t d = c.w[i][j][q];
                    float sc   = q < 2 ? sp[0] : sp[1];
                    v2bf e0    = __builtin_convertvector(cvt_fp4x2_f32(d, 0) * sc, v2bf);
                    v2bf e1    = __builtin_convertvector(cvt_fp4x2_f32(d, 1) * sc, v2bf);
                    v2bf e2    = __builtin_convertvector(cvt_fp4x2_f32(d, 2) * sc, v2bf);
                    v2bf e3    = __builtin_convertvector(cvt_fp4x2_f32(d, 3) * sc, v2bf);
                    v8bf wv    = {e0[0], e0[1], e1[0], e1[1], e2[0], e2[1], e3[0], e3[1]};
#pragma unroll
                    for(int m = 0; m < MT; m++)
                        acc[j][m] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(wv, c.x[i][m].v[q], acc[j][m], 0, 0, 0);
                }
            }
        }
        else // F_FP8
        {
            // Lane group g holds halves g and 4 + g of this K-tile (fp8 operand
            // layout); half hb belongs to 32-block hb / 2 whose E8M0 is
            // 2^(max(e_2b, e_2b+1) - 7). Data is rescaled by 2^(e_hb - E_b) in the LUT.
#pragma unroll
            for(int j = 0; j < NT; j++)
            {
                const uint32_t sA = c.ws[i >> 1][j].s[2 * par];     // halves 0..3
                const uint32_t sB = c.ws[i >> 1][j].s[2 * par + 1]; // halves 4..7
                const uint32_t sh = 8 * g;
                const uint32_t a0 = (sA >> sh) & 0xffu;             // half g
                const uint32_t b0 = (sB >> sh) & 0xffu;             // half 4 + g
                const uint32_t a1 = (sA >> (sh ^ 8)) & 0xffu;       // half g ^ 1
                const uint32_t b1 = (sB >> (sh ^ 8)) & 0xffu;       // half (4 + g) ^ 1
                const uint32_t ea = a0 >> 3, eb = b0 >> 3;
                const uint32_t Ea = max(ea, a1 >> 3), Eb = max(eb, b1 >> 3);
                const uint32_t ia = ((a0 & 7) << 4) | min(Ea - ea, 15u);
                const uint32_t ib = ((b0 & 7) << 4) | min(Eb - eb, 15u);
                const uint2 La    = lut[ia];
                const uint2 Lb    = lut[ib];
                const v4i d       = c.w[i][j];
                const v2i x0      = fp8_lut_decode(d[0], La.x, La.y);
                const v2i x1      = fp8_lut_decode(d[1], La.x, La.y);
                const v2i y0      = fp8_lut_decode(d[2], Lb.x, Lb.y);
                const v2i y1      = fp8_lut_decode(d[3], Lb.x, Lb.y);
                const v8i wv      = {x0[0], x0[1], x1[0], x1[1], y0[0], y0[1], y1[0], y1[1]};
                // this lane's scale slot is block g = halves (2g, 2g + 1)
                const uint32_t sg = (g < 2 ? sA : sB) >> (16 * (g & 1));
                const uint32_t Eg = max((sg >> 3) & 0xfu, (sg >> 11) & 0xfu);
#pragma unroll
                for(int m = 0; m < MT; m++)
                    acc[j][m] = mfma_s8<0>(wv, c.x[i][m].v, acc[j][m], Eg + 120, c.x[i][m].s);
            }
        }
    }
}

DEV float swiglu_oai(float g, float u)
{
    g = fminf(g, 7.0f);
    u = fminf(fmaxf(u, -7.0f), 7.0f);
    return g / (1.0f + __expf(-1.702f * g)) * (u + 1.0f);
}

// DEC_S2F32: stage2 accumulates with f32 atomics into an f32 [T, H] output.
#ifndef DEC_S2F32
#define DEC_S2F32 0
#endif
#ifndef DEC_ATOM_LDS
#define DEC_ATOM_LDS 1
#endif
// Stage1: swigluoai, act2 store (f16 or MXFP8). Stage2: routing weight, bf16 atomics.
// XB: stage2 transposes through the caller's LDS buffer xb instead of its own.
template <int FMT, bool S1, int MT, int NT, int WN, int ML, bool XB = false, bool CO = false>
DEV void epilogue(const Params& p, v4f (&acc)[NT][MT], int e, int blk, const int (&tok)[MT],
                  const int (&slot)[MT], const bool (&xv)[MT], int wn, int g, int col, int lane,
                  float* xb = nullptr, const float* fwo = nullptr)
{
    constexpr int BM = 16 * MT;
    if constexpr(S1)
    {
        constexpr int P = NT / 2;
        const float ig  = p.inv_gs[2 * e];
        const float iu  = p.inv_gs[2 * e + 1];
        const int I     = p.n_out;
#pragma unroll
        for(int m = 0; m < MT; m++)
        {
            const size_t row2 = (size_t)tok[m] * p.topk + slot[m];
            float o[P][4];
            float amax = 0.0f;
#pragma unroll
            for(int pp = 0; pp < P; pp++)
#pragma unroll
                for(int v = 0; v < 4; v++)
                {
                    o[pp][v] = swiglu_oai(acc[pp][m][v] * ig, acc[P + pp][m][v] * iu);
                    amax     = fmaxf(amax, fabsf(o[pp][v]));
                }
            const int c0 = (blk_x() * WN + wn) * 16 * P;
            if constexpr(IS16(FMT))
            {
                if(xv[m])
                {
#pragma unroll
                    for(int pp = 0; pp < P; pp++)
                    {
                        v2h a0 = {(_Float16)o[pp][0], (_Float16)o[pp][1]};
                        v2h a1 = {(_Float16)o[pp][2], (_Float16)o[pp][3]};
                        v2i st = {__builtin_bit_cast(int, a0), __builtin_bit_cast(int, a1)};
                        v2i* dst = (v2i*)(p.out + (row2 * I + c0 + 16 * pp + 4 * g) * 2);
                        if constexpr(CO) // coherent (sc1) store for the fused kernel
                            __hip_atomic_store((uint64_t*)dst, __builtin_bit_cast(uint64_t, st), __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
                        else
                            *dst = st;
                    }
                }
            }
            else
            {
                static_assert(S1 && (IS16(FMT) || P == 2), "MXFP8 output needs 32 cols per wave");
                amax = fmaxf(amax, __shfl_xor(amax, 16));
                amax = fmaxf(amax, __shfl_xor(amax, 32));
                // e8m0 = ceil(log2(amax / 448)), RoundUp as AITER's MX quant.
                uint32_t qb = __float_as_uint(amax / 448.0f);
                uint32_t eb = (qb >> 23) + ((qb & 0x7fffffu) != 0);
                eb          = eb > 254 ? 254 : eb;
                float mult  = __uint_as_float((254u - eb) << 23);
                if(xv[m])
                {
#pragma unroll
                    for(int pp = 0; pp < P; pp++)
                    {
                        int pk = __builtin_amdgcn_cvt_pk_fp8_f32(o[pp][0] * mult, o[pp][1] * mult, 0, false);
                        pk = __builtin_amdgcn_cvt_pk_fp8_f32(o[pp][2] * mult, o[pp][3] * mult, pk, true);
                        *(int*)(p.out + row2 * I + c0 + 16 * pp + 4 * g) = pk;
                    }
                    if(g == 0)
                        p.out_sc[row2 * (I / 32) + c0 / 32] = (uint8_t)eb;
                }
            }
        }
    }
    else
    {
        const float igs = p.inv_gs[e];
#if DEC_ATOM_LDS
        // Transpose through LDS so that one atomic instruction covers 64 consecutive
        // output columns of 64 / (8 * NT) token rows instead of 16 rows x 16 B.
        constexpr int LD = 16 * NT + 4;
        constexpr int CP  = 8 * NT;            // column pairs per row
        constexpr int CPW = CP < 64 ? CP : 64; // column pairs per instruction
        constexpr int RPI = 64 / CPW;          // rows per instruction
        __shared__ float ep[(S1 || XB) ? 1 : WN * BM * LD];
        float* E = (XB ? xb : ep) + wn * BM * LD;
        int tv[MT];
#pragma unroll
        for(int m = 0; m < MT; m++)
        {
            const float wgt = (fwo ? fwo[m] : p.sorted_w[blk * BM + m * 16 + col]) * igs;
            tv[m]           = xv[m] ? tok[m] : -1;
#pragma unroll
            for(int j = 0; j < NT; j++)
#pragma unroll
                for(int v = 0; v < 4; v++)
                    E[(16 * m + col) * LD + 16 * j + 4 * g + v] = acc[j][m][v] * wgt;
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        const int cp = lane % CPW;
#pragma unroll
        for(int r = 0; r < BM / RPI; r++)
        {
            const int t  = r * RPI + lane / CPW;
            const int tk = __shfl(tv[(r * RPI) / 16], t % 16);
#pragma unroll
            for(int h = 0; h < CP / CPW; h++)
            {
                const int n0   = (blk_x() * WN + wn) * NT * 16 + 2 * (cp + h * CPW);
                const float2 v = *(const float2*)(E + t * LD + 2 * (cp + h * CPW));
                // DEC_EXP_NOOUT: timing experiment only, (almost) no output stores
                if(tk >= 0 && (!DEC_EXP_NOOUT || v.x == 1.2345e-30f))
                {
                    v2bf a = {(__bf16)v.x, (__bf16)v.y};
#if DEC_NOATOM
                    *(v2bf*)(p.out + ((size_t)tk * p.n_out + n0) * 2) = a;
#else
#if DEC_S2F32
                    float* fo = (float*)(p.out + ((size_t)tk * p.n_out + n0) * 4);
                    __builtin_amdgcn_global_atomic_fadd_f32(fo, v.x);
                    __builtin_amdgcn_global_atomic_fadd_f32(fo + 1, v.y);
#else
                    __builtin_amdgcn_global_atomic_fadd_v2bf16(
                        (v2bf*)(p.out + ((size_t)tk * p.n_out + n0) * 2), a);
#endif
#endif
                }
            }
        }
        if constexpr(ML > 1)
        {
            // the next block rewrites E
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        }
#else
#pragma unroll
        for(int m = 0; m < MT; m++)
        {
            if(!xv[m])
                continue;
            const float wgt = p.sorted_w[blk * BM + m * 16 + col] * igs;
            v2bf* op        = (v2bf*)(p.out + ((size_t)tok[m] * p.n_out) * 2);
#pragma unroll
            for(int j = 0; j < NT; j++)
            {
                const int n0 = ((blk_x() * WN + wn) * NT + j) * 16 + 4 * g;
                v2bf a0      = {(__bf16)(acc[j][m][0] * wgt), (__bf16)(acc[j][m][1] * wgt)};
                v2bf a1      = {(__bf16)(acc[j][m][2] * wgt), (__bf16)(acc[j][m][3] * wgt)};
#if DEC_S2F32
                float* fo = (float*)(p.out + ((size_t)tok[m] * p.n_out + n0) * 4);
#pragma unroll
                for(int v = 0; v < 4; v++)
                    __builtin_amdgcn_global_atomic_fadd_f32(fo + v, acc[j][m][v] * wgt);
#elif DEC_NOATOM
                op[n0 / 2]     = a0;
                op[n0 / 2 + 1] = a1;
#else
                __builtin_amdgcn_global_atomic_fadd_v2bf16(op + n0 / 2, a0);
                __builtin_amdgcn_global_atomic_fadd_v2bf16(op + n0 / 2 + 1, a1);
#endif
            }
        }
#endif
    }
}

// ---------------------------------------------------------------- kernel
// Grid: x = N group, y = M block. Block: WK x WN waves; wave (wk, wn) reduces K-tiles
// [wk * KTW, (wk + 1) * KTW) for NT weight tiles. Stage1: per wave NT/2 gate tiles and
// the matching NT/2 up tiles. K partial sums of the WK waves are reduced in LDS.
// KCF = KC | 256 (single-buffered chunks) | ML << 12. ML > 1: the WG of block y with
// y % ML == 0 (or the first block of its expert) also runs the following blocks of the
// same expert up to the next multiple of ML; when the whole K slice of a wave is one
// chunk the weights stay in registers and only X is reloaded per block.
#ifndef DEC_WPE
#define DEC_WPE 1
#endif
// KCF flag WP2: at least 2 waves per SIMD (keeps a kernel that fits 256 VGPRs with NVFP4
// only at 256 when the MXFP8 shared-expert path is compiled in).
#define WP2 (1 << 9)
// KCF flag SGF: F16 weight scales in the SG layout (see load_sg); needs KTW = 12.
#define SGF (1 << 10)
// KCF flag SGL (with SGF): load scale group 1 late (after chunk 2 of 12 / KC), so only one group
// (3 NT registers) is live instead of two.
#define SGL (1 << 11)
// XBF: X is bf16 (cast to f16 in registers). fsid: routing entries of the MT column
// tiles of this lane (fused kernel) instead of p.sorted_ids.
template <int FMT, bool S1, int MT, int NT, int WK, int WN, int KC, bool DB, int ML, int SGR = 0, int WA = -1,
          bool XBF = false>
DEV void gemm_body(const Params& p, int mblk, int e, int nb, const int* fsid = nullptr)
{
    constexpr int BM   = 16 * MT;
    constexpr int WAUX = WA >= 0 ? WA : S1 ? (DEC_NT1 ? 2 : 0) : (DEC_NT2 ? DEC_W2_AUX : 0);
    constexpr int XAUX = (S1 ? DEC_X_AUX : DEC_X_AUX2) | (XBF ? 256 : 0);
    constexpr bool M8  = FMT == F_MX8;
    const int lane = threadIdx.x & 63;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int wk   = wave % WK;
    const int wn   = wave / WK;
    const int g    = lane >> 4;
    const int col  = lane & 15;
    const int KT   = p.K / 128;
    const int KTW  = KT / WK;
    const int kt0  = wk * KTW;

    // weight tiles
    int nt[NT];
#pragma unroll
    for(int j = 0; j < NT; j++)
    {
        if constexpr(S1)
        {
            constexpr int P = NT / 2;
            int pi          = (blk_x() * WN + wn) * P + (j % P);
            nt[j]           = j < P ? pi : p.n_out / 16 + pi;
        }
        else
            nt[j] = (blk_x() * WN + wn) * NT + j;
    }
    Src<MT, NT> src;
    // MXFP8: one expert in w8 / w8_sc, 2 KB of weights and 64 scale bytes per tile and K-tile
    src.rw  = make_rsrc(M8 ? p.w8 : p.w + (size_t)e * p.ntiles * KT * 1024);
    src.rws = make_rsrc(M8 ? p.w8_sc : p.w_sc + (size_t)e * p.ntiles * (KT / 2) * ws_pair_bytes<FMT>());
#if DEC_XOOB
    // padding columns load from row xrows (out of range: zeros, no memory access)
    const int xrows = S1 ? p.T : p.T * p.topk;
    src.rx          = make_rsrc(p.a, xrows * p.K * (IS16(FMT) ? 2 : 1));
    src.rxs         = make_rsrc(p.a_sc, xrows * (p.K / 32));
#else
    src.rx  = make_rsrc(p.a);
    src.rxs = make_rsrc(p.a_sc);
#endif
    src.wl  = lane * 16;
    src.wsl = FMT == F_MXFP4 ? lane * 2 : (FMT == F_FP8 ? col * 16 : (M8 ? lane : lane * 4));
#pragma unroll
    for(int j = 0; j < NT; j++)
    {
        src.wo[j]  = nt[j] * KT * (M8 ? 2048 : 1024);
        src.wso[j] = M8 ? nt[j] * KT * 64 : nt[j] * (KT / 2) * ws_pair_bytes<FMT>();
    }
    // SGR: KTW = 12, two scale groups per wave, both loaded up front (1) or group 1 late (2)
    uint32_t sg[SGR == 1 ? 6 : 3][NT];
    if constexpr(SGR == 1)
        load_sg<MT, NT, 2>(sg, src, kt0 / 6);
    else if constexpr(SGR == 2)
        load_sg<MT, NT, 1>(sg, src, kt0 / 6);

    // routing of all blocks of this WG
    int sid[ML][MT];
#pragma unroll
    for(int b = 0; b < ML; b++)
#pragma unroll
        for(int m = 0; m < MT; m++)
            sid[b][m] = fsid ? fsid[m] : b < nb ? p.sorted_ids[(mblk + b) * BM + m * 16 + col] : 0;

    __shared__ uint2 s_lut[FMT == F_FP8 ? 128 : 1];
    if constexpr(FMT == F_FP8)
    {
        for(int t = threadIdx.x; t < 128; t += blockDim.x)
            s_lut[t] = uint2{c_fp8_lut[2 * t], c_fp8_lut[2 * t + 1]};
        __syncthreads();
    }
    const bool godd = g & 1;
    const int nch   = KTW / KC;
    Chunk<FMT, MT, NT, KC> cur;

#pragma unroll 1
    for(int b = 0; b < nb; b++)
    {
        const int blk = mblk + b;
        int tok[MT], slot[MT];
        bool xv[MT];
#pragma unroll
        for(int m = 0; m < MT; m++)
        {
            int s   = 0;
#pragma unroll
            for(int q = 0; q < ML; q++)
                s = q == b ? sid[q][m] : s;
            tok[m]  = s & 0xffffff;
            slot[m] = s >> 24;
            xv[m]   = tok[m] < p.T;
            // Padding columns read row 0; MFMA columns are independent and padded
            // tokens are never stored.
            int row = S1 ? tok[m] : tok[m] * p.topk + slot[m];
            row     = xv[m] ? row : 0;
#if DEC_XOOB
            row = xv[m] ? row : xrows;
#endif
#if DEC_EXP_XDIST
            // timing experiment only: padding columns read distinct real rows
            row = xv[m] ? row : (blk * BM + m * 16 + col) % (S1 ? p.T : p.T * p.topk);
#endif
#if DEC_XROW0
            row = 0; // timing experiment only: all X reads hit row 0 (wrong results)
#endif
            src.xo[m]  = IS16(FMT) ? (row * p.K + (DEC_EXP_XC ? 8 : 32) * (DEC_EXP_X2L ? (g & 1) : g)) * 2 : row * p.K + 16 * g;
            src.xso[m] = row * (p.K / 32) + g;
#if DEC_EXP_XDIST == 2
            src.xz[m] = xv[m] ? -1 : 0;
#endif
        }

        v4f acc[NT][MT], accn[NT][MT];
#pragma unroll
        for(int j = 0; j < NT; j++)
#pragma unroll
            for(int m = 0; m < MT; m++)
            {
                acc[j][m]  = v4f{0, 0, 0, 0};
                accn[j][m] = v4f{0, 0, 0, 0};
            }

        if(!DB && b > 0 && nch == 1)
        {
            // weights of the single chunk are still in registers
            load_x<FMT, MT, NT, KC, XAUX>(cur, src, kt0);
            compute_chunk<FMT, MT, NT, KC>(cur, acc, accn, godd, g, s_lut);
        }
        else if constexpr(SGR == 2)
        {
            // one scale group live: rolled loop over the two groups, chunks unrolled
            static_assert(!DB && ML == 1 && 6 % KC == 0 && KC % 2 == 0, "bad SGL config");
#pragma unroll 1
            for(int gr = 0; gr < 2; gr++)
            {
                if(gr > 0)
                    load_sg<MT, NT, 1>(sg, src, kt0 / 6 + gr);
#pragma unroll
                for(int ch = 0; ch < 6 / KC; ch++)
                {
                    const int kt = kt0 + gr * 6 + ch * KC;
                    load_w<FMT, MT, NT, KC, WAUX, true>(cur, src, kt);
                    load_x<FMT, MT, NT, KC, XAUX>(cur, src, kt);
                    fill_sg(cur, sg, ch * KC / 2);
                    __builtin_amdgcn_sched_barrier(0);
                    compute_chunk<FMT, MT, NT, KC>(cur, acc, accn, godd, g, s_lut);
                }
            }
        }
        else if constexpr(SGR)
        {
            static_assert(!DB && ML == 1 && 12 % KC == 0 && KC % 2 == 0, "bad SG config");
#pragma unroll
            for(int ch = 0; ch < 12 / KC; ch++)
            {
                load_w<FMT, MT, NT, KC, WAUX, true>(cur, src, kt0 + ch * KC);
                load_x<FMT, MT, NT, KC, XAUX>(cur, src, kt0 + ch * KC);
                fill_sg(cur, sg, ch * KC / 2);
                __builtin_amdgcn_sched_barrier(0);
                compute_chunk<FMT, MT, NT, KC>(cur, acc, accn, godd, g, s_lut);
            }
        }
        else
        {
            load_chunk<FMT, MT, NT, KC, WAUX, XAUX>(cur, src, kt0);
            __builtin_amdgcn_sched_barrier(0); // keep all loads of the chunk ahead of compute
            if constexpr(DB)
            {
                for(int ch = 0; ch < nch; ch++)
                {
                    Chunk<FMT, MT, NT, KC> nxt;
                    if(ch + 1 < nch)
                        load_chunk<FMT, MT, NT, KC, WAUX, XAUX>(nxt, src, kt0 + (ch + 1) * KC);
                    compute_chunk<FMT, MT, NT, KC>(cur, acc, accn, godd, g, s_lut);
                    cur = nxt;
                }
            }
            else
            {
                compute_chunk<FMT, MT, NT, KC>(cur, acc, accn, godd, g, s_lut);
                for(int ch = 1; ch < nch; ch++)
                {
                    load_chunk<FMT, MT, NT, KC, WAUX, XAUX>(cur, src, kt0 + ch * KC);
                    __builtin_amdgcn_sched_barrier(0);
                    compute_chunk<FMT, MT, NT, KC>(cur, acc, accn, godd, g, s_lut);
                }
            }
        }
        if constexpr(FMT == F_TERMS)
        {
#pragma unroll
            for(int j = 0; j < NT; j++)
#pragma unroll
                for(int m = 0; m < MT; m++)
                    acc[j][m] -= accn[j][m];
        }

        if constexpr(WK > 1)
        {
            extern __shared__ v4f red[];
            if(b > 0)
                __syncthreads(); // wave wk == 0 is done reading the previous block
            if(wk > 0)
            {
#pragma unroll
                for(int j = 0; j < NT; j++)
#pragma unroll
                    for(int m = 0; m < MT; m++)
                        red[((((wk - 1) * WN + wn) * NT + j) * MT + m) * 64 + lane] = acc[j][m];
            }
            __syncthreads();
            if(wk > 0)
            {
                if constexpr(ML == 1)
                    return;
                else
                    continue;
            }
            for(int w = 1; w < WK; w++)
#pragma unroll
                for(int j = 0; j < NT; j++)
#pragma unroll
                    for(int m = 0; m < MT; m++)
                        acc[j][m] += red[((((w - 1) * WN + wn) * NT + j) * MT + m) * 64 + lane];
        }

        // epilogue
        // F_MX8 shares the F_F16 epilogue (and its LDS buffer)
        epilogue<M8 ? F_F16 : FMT, S1, MT, NT, WN, ML, false, XBF>(p, acc, e, blk, tok, slot, xv, wn, g, col, lane);
    }
}

template <int FMT, bool S1, int MT, int NT, int WK, int WN, int KCF>
__global__ __launch_bounds__(64 * WK * WN)
    __attribute__((amdgpu_waves_per_eu((KCF & WP2) ? 2 : DEC_WPE))) void moe_dec_gemm(Params p)
{
    constexpr int KC  = KCF & 255;
    constexpr bool DB = !(KCF & 256);
    constexpr int ML  = (KCF >> 12) > 0 ? (KCF >> 12) : 1;
    constexpr bool XBF1 = DEC_XBF1 && S1 && FMT == F_F16; // X is bf16, cast in registers
    constexpr int BM  = 16 * MT;
#if DEC_XCD == 2
    const int nvalid = p.num_valid[0];
    const int mblk   = blk_map(p, BM, nvalid);
    if(mblk < 0)
        return;
#else
    const int mblk    = blk_y();
    // Issue the routing loads before the early-exit branch (grid y never exceeds the
    // sorted buffers) to save one dependent memory latency.
    const int nvalid = p.num_valid[0];
#endif
#if DEC_XBF1
    if constexpr(S1)
    {
        const int nw = gridDim.x * gridDim.y;
        const int L  = blockIdx.x + blockIdx.y * gridDim.x;
        const int z0 = (int)((long long)L * p.zn16 / nw), z1 = (int)((long long)(L + 1) * p.zn16 / nw);
        for(int i = z0 + (int)threadIdx.x; i < z1; i += blockDim.x)
            ((v4i*)p.zo)[i] = v4i{0, 0, 0, 0};
    }
#endif
    const int e      = p.sorted_eids[mblk];
    int en[ML];
    en[0] = (ML > 1 && mblk % ML != 0) ? p.sorted_eids[mblk - 1] : -1;
#pragma unroll
    for(int i = 1; i < ML; i++)
        en[i] = p.sorted_eids[min(mblk + i, (int)gridDim.y - 1)];
    asm volatile("" ::"s"(e));
    if(mblk * BM >= nvalid || en[0] == e)
        return;
    int nb = 1;
#pragma unroll
    for(int i = 1; i < ML; i++)
        nb += (nb == i && (mblk + i) % ML != 0 && mblk + i < (int)gridDim.y && en[i] == e &&
               (mblk + i) * BM < nvalid);
    if constexpr((DEC_MBAUX >> (S1 ? 0 : 1)) & 1)
    {
        // sorted blocks are expert-contiguous: a neighbor block of the same expert
        const int nx = mblk + nb;
        const bool multi = (mblk > 0 && p.sorted_eids[mblk - 1] == e) ||
                           (nx < (int)gridDim.y && nx * BM < nvalid && p.sorted_eids[nx] == e);
        if(multi)
        {
            if constexpr(FMT == F_F16)
            {
                if(e == p.mx8_e)
                {
                    gemm_body<F_MX8, S1, MT, NT, WK, WN, (KC > 1 ? KC / 2 : 1), DB, ML, 0, DEC_MBAUX_V>(
                        p, mblk, e, nb);
                    return;
                }
            }
            gemm_body<FMT, S1, MT, NT, WK, WN, KC, DB, ML,
                      FMT == F_F16 && (KCF & SGF) ? ((KCF & SGL) ? 2 : 1) : 0, DEC_MBAUX_V, XBF1>(p, mblk, e, nb);
            return;
        }
    }
    if constexpr(FMT == F_F16)
    {
        if(e == p.mx8_e)
        {
            // same weight register bytes per chunk: half the K-tiles
            gemm_body<F_MX8, S1, MT, NT, WK, WN, (KC > 1 ? KC / 2 : 1), DB, ML>(p, mblk, e, nb);
            return;
        }
    }
    gemm_body<FMT, S1, MT, NT, WK, WN, KC, DB, ML,
              FMT == F_F16 && (KCF & SGF) ? ((KCF & SGL) ? 2 : 1) : 0, -1, XBF1>(p, mblk, e, nb);
}

// ---------------------------------------------------------------- X-in-LDS kernel
// One WG = WN waves split N; every wave runs its NT weight tiles over the whole K (no K
// reduction). The X block (BM rows, 16-bit) goes through LDS once per WG and is shared
// by all waves: a ring of NS stages of KC K-tiles. X (global -> VGPR -> LDS) and W
// (global -> VGPR) are loaded D chunks ahead; one barrier per chunk.
// LDS unit (i, m, q) = 64 lanes x 16 B; lane l holds X[row(m, l % 16)][128 kt + 32 (l / 16)
// + 8 q, +8), the MFMA B operand layout. Wave wn loads q = wn % QW + QW a of the K-tiles
// i = wn / QW + WI t of every chunk.
#define LX_LDS_BAR() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")

template <int FMT, bool S1, int MT, int NT, int WN, int KC, int D, int NS>
__global__ __launch_bounds__(64 * WN) void moe_lx_gemm(Params p)
{
    static_assert(IS16(FMT), "LX kernel: 16-bit activations only");
    constexpr int BM   = 16 * MT;
    constexpr int QW   = WN < 4 ? WN : 4; // waves sharing one K-tile
    constexpr int QA   = 4 / QW;          // q units per wave and K-tile
    constexpr int WI   = WN / QW;         // waves along the K-tiles of a chunk
    constexpr int KI   = KC / WI;         // K-tiles per wave and chunk
    constexpr int U    = KC * MT * 4;     // LDS units per stage
    constexpr int WAUX = S1 ? (DEC_NT1 ? 2 : 0) : (DEC_NT2 ? DEC_W2_AUX : 0);
    static_assert(KC % 2 == 0 && KC % WI == 0, "bad KC");
    using vec = typename XTile<FMT>::vec;

    const int mblk   = blk_y();
    const int nvalid = p.num_valid[0];
    const int e      = p.sorted_eids[mblk];
    asm volatile("" ::"s"(e));
    if(mblk * BM >= nvalid)
        return;
    if constexpr(FMT == F_F16)
    {
        if(e == p.mx8_e)
        {
            // MXFP8 shared expert: register kernel body (X from global), same WG shape
            gemm_body<F_MX8, S1, MT, NT, 1, WN, (KC > 1 ? KC / 2 : 1), false, 1>(p, mblk, e, 1);
            return;
        }
    }

    const int lane = threadIdx.x & 63;
    const int wn   = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int g    = lane >> 4;
    const int col  = lane & 15;
    const int KT   = p.K / 128;
    const int nch  = KT / KC; // multiple of D (host side)

    Src<MT, NT> src;
    src.rw  = make_rsrc(p.w + (size_t)e * p.ntiles * KT * 1024);
    src.rws = make_rsrc(p.w_sc + (size_t)e * p.ntiles * (KT / 2) * ws_pair_bytes<FMT>());
    src.rx  = make_rsrc(p.a);
    src.rxs = src.rx;
    src.wl  = lane * 16;
    src.wsl = lane * 4;
#pragma unroll
    for(int j = 0; j < NT; j++)
    {
        int nt;
        if constexpr(S1)
        {
            constexpr int P = NT / 2;
            const int pi    = (blk_x() * WN + wn) * P + (j % P);
            nt              = j < P ? pi : p.n_out / 16 + pi;
        }
        else
            nt = (blk_x() * WN + wn) * NT + j;
        src.wo[j]  = nt * KT * 1024;
        src.wso[j] = nt * (KT / 2) * ws_pair_bytes<FMT>();
    }
    int tok[MT], slot[MT];
    bool xv[MT];
#pragma unroll
    for(int m = 0; m < MT; m++)
    {
        const int s = p.sorted_ids[mblk * BM + m * 16 + col];
        tok[m]      = s & 0xffffff;
        slot[m]     = s >> 24;
        xv[m]       = tok[m] < p.T;
        int row     = S1 ? tok[m] : tok[m] * p.topk + slot[m];
        row         = xv[m] ? row : 0;
        src.xo[m]   = (row * p.K + 32 * g) * 2;
        src.xso[m]  = 0;
    }

    __shared__ v4i xl[NS * U * 64];
    const int qx = wn % QW;
    const int ix = wn / QW;
    v4i xr[D][KI][MT][QA];
    Chunk<FMT, MT, NT, KC> wb[D];

    auto gload_x = [&](v4i(&r)[KI][MT][QA], int c) {
#pragma unroll
        for(int t = 0; t < KI; t++)
#pragma unroll
            for(int m = 0; m < MT; m++)
#pragma unroll
                for(int a = 0; a < QA; a++)
                    r[t][m][a] = __builtin_amdgcn_raw_buffer_load_b128(
                        src.rx, src.xo[m], (c * KC + ix + WI * t) * 256 + 16 * (qx + QW * a), 0);
    };
    auto lds_x = [&](const v4i(&r)[KI][MT][QA], int c) {
        const int st = NS > 1 ? c % NS : 0;
#pragma unroll
        for(int t = 0; t < KI; t++)
#pragma unroll
            for(int m = 0; m < MT; m++)
#pragma unroll
                for(int a = 0; a < QA; a++)
                    xl[(st * U + ((ix + WI * t) * MT + m) * 4 + qx + QW * a) * 64 + lane] = r[t][m][a];
    };

#pragma unroll
    for(int d = 0; d < D; d++)
        if(d < nch)
        {
            gload_x(xr[d], d);
            load_w<FMT, MT, NT, KC, WAUX>(wb[d], src, d * KC);
        }
    lds_x(xr[0], 0);
    LX_LDS_BAR();

    v4f acc[NT][MT], accn[NT][MT];
#pragma unroll
    for(int j = 0; j < NT; j++)
#pragma unroll
        for(int m = 0; m < MT; m++)
        {
            acc[j][m]  = v4f{0, 0, 0, 0};
            accn[j][m] = v4f{0, 0, 0, 0};
        }

#pragma unroll 1
    for(int c0 = 0; c0 < nch; c0 += D)
    {
#pragma unroll
        for(int d = 0; d < D; d++)
        {
            const int c = c0 + d;
            if(c + D < nch)
                gload_x(xr[d], c + D);
            Chunk<FMT, MT, NT, KC> cur = wb[d];
            const int st               = NS > 1 ? c % NS : 0;
#pragma unroll
            for(int i = 0; i < KC; i++)
#pragma unroll
                for(int m = 0; m < MT; m++)
#pragma unroll
                    for(int q = 0; q < 4; q++)
                        cur.x[i][m].v[q] = __builtin_bit_cast(vec, xl[(st * U + (i * MT + m) * 4 + q) * 64 + lane]);
            compute_chunk<FMT, MT, NT, KC>(cur, acc, accn, false, g, nullptr);
            if(c + D < nch)
                load_w<FMT, MT, NT, KC, WAUX>(wb[d], src, (c + D) * KC);
            if(c + 1 < nch)
            {
                lds_x(xr[(d + 1) % D], c + 1);
                LX_LDS_BAR();
            }
        }
    }
    epilogue<FMT, S1, MT, NT, WN, 1>(p, acc, e, mblk, tok, slot, xv, wn, g, col, lane);
}

// ---------------------------------------------------------------- stage2 X-in-LDS kernel
// AITER gemm2-like tile: one WG = WN waves split N (NT weight tiles each, BN = 16 NT WN);
// the whole X block (BM rows x KT K-tiles) goes to LDS once and is shared by all waves.
// Weights go global -> VGPR in a ring of D chunks of KC K-tiles, issued before the X
// loads complete (D * KC >= KT: all K in flight). The epilogue reuses the X LDS.
// FL bits: 1 = s_setprio(1) around the MFMA chunks, 2 = at least 2 waves per SIMD.
template <int FMT, int MT, int NT, int WN, int KT>
constexpr int s2x_lds_v4()
{
    constexpr int UX = IS16(FMT) ? 4 : 2;
    constexpr int X  = KT * MT * UX * 64;
    constexpr int EP = (WN * 16 * MT * (16 * NT + 4) + 3) / 4;
    return X > EP ? X : EP;
}

// Fused small-T kernel sync: stage1 WG (x, y) stores the call epoch to flags[y * FU_FS + x]
// after its act2 stores (slot y = 0 WGs also zero the output first).
#define FU_FS 256
struct FuWait
{
    const int* flags;
    int y, gx1, ep;
};
// Wave 0 spins until all stage1 WGs of slot y and of slot 0 (output zeroing) are done.
// No acquire fence (it emits an L2 invalidate): the producers use sc1 stores and the
// consumer reads act2 with sc1 loads.
#ifndef FU_SLEEP
#define FU_SLEEP 2
#endif
// Timing-only (wrong results): FU_NOWAIT skips the wait; FU_MODE 1 = stage1 only, 2 = stage2 only.
#ifndef FU_NOWAIT
#define FU_NOWAIT 0
#endif
#ifndef FU_MODE
#define FU_MODE 0
#endif
DEV void fu_wait(const FuWait& w)
{
    const int lane = threadIdx.x & 63;
    if(!FU_NOWAIT && threadIdx.x < 64)
        for(int x0 = 0; x0 < w.gx1; x0 += 64)
        {
            const int x = x0 + lane;
            while(true)
            {
                const int a = x < w.gx1 ? __hip_atomic_load(w.flags + w.y * FU_FS + x, __ATOMIC_RELAXED,
                                                             __HIP_MEMORY_SCOPE_AGENT)
                                        : w.ep;
                const int b = x < w.gx1 ? __hip_atomic_load(w.flags + x, __ATOMIC_RELAXED,
                                                             __HIP_MEMORY_SCOPE_AGENT)
                                        : w.ep;
                if(__all(a == w.ep && b == w.ep))
                    break;
                __builtin_amdgcn_s_sleep(FU_SLEEP);
            }
        }
    __syncthreads();
}

// FU: fused kernel (routing from fsid / fwo, weights issued before fu_wait, X after it).
template <int FMT, int MT, int NT, int WN, int KC, int D, int KT, int FL, bool FU = false>
DEV void s2x_body(const Params& p, int mblk, int e, v4i* xl, const int* fsid = nullptr,
                  const float* fwo = nullptr, const FuWait* fw = nullptr)
{
    constexpr int BM   = 16 * MT;
    constexpr bool M8  = FMT == F_MX8;
    constexpr bool X16 = IS16(FMT);
    constexpr int UX   = X16 ? 4 : 2;      // LDS units (64 lanes x 16 B) per K-tile and m
    constexpr int QW   = WN < UX ? WN : UX; // waves sharing the units of one (K-tile, m)
    constexpr int WI   = WN / QW;
    constexpr int NR   = (KT * MT + WI - 1) / WI;
    constexpr int NCH  = KT / KC;
    constexpr int WAUX = DEC_NT2 ? DEC_W2_AUX : 0;
    constexpr bool SGR = FMT == F_F16 && (FL & 4); // SG scale layout, KT = 6: one group
    static_assert(KT % KC == 0 && (M8 || KC % 2 == 0) && UX % QW == 0 && WN % QW == 0, "bad s2x");
    static_assert(!SGR || KT == 6, "SG needs KT = 6");
    const int lane = threadIdx.x & 63;
    const int wn   = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int g    = lane >> 4;
    const int col  = lane & 15;

    Src<MT, NT> src;
    src.rw  = make_rsrc(M8 ? p.w8 : p.w + (size_t)e * p.ntiles * KT * 1024);
    src.rws = make_rsrc(M8 ? p.w8_sc : p.w_sc + (size_t)e * p.ntiles * (KT / 2) * ws_pair_bytes<FMT>());
    src.rx  = make_rsrc(p.a);
    src.rxs = make_rsrc(p.a_sc);
    src.wl  = lane * 16;
    src.wsl = FMT == F_MXFP4 ? lane * 2 : (M8 ? lane : lane * 4);
#pragma unroll
    for(int j = 0; j < NT; j++)
    {
        const int nt = (blk_x() * WN + wn) * NT + j;
        src.wo[j]    = nt * KT * (M8 ? 2048 : 1024);
        src.wso[j]   = M8 ? nt * KT * 64 : nt * (KT / 2) * ws_pair_bytes<FMT>();
    }
    uint32_t sg[SGR ? 3 : 1][NT];
    if constexpr(SGR)
        load_sg<MT, NT, 1>(sg, src, 0);
    int tok[MT], slot[MT];
    bool xv[MT];
#pragma unroll
    for(int m = 0; m < MT; m++)
    {
        const int s = FU ? fsid[m] : p.sorted_ids[mblk * BM + m * 16 + col];
        tok[m]      = s & 0xffffff;
        slot[m]     = s >> 24;
        xv[m]       = tok[m] < p.T;
        int row     = tok[m] * p.topk + slot[m];
        row         = xv[m] ? row : 0;
        src.xo[m]   = X16 ? (row * p.K + 32 * g) * 2 : row * p.K + 16 * g;
        src.xso[m]  = row * (p.K / 32) + g;
    }

    // X: wave wn loads units qx + QW a of the (K-tile, m) rows r = ix + WI t
    const int qx = wn % QW;
    const int ix = wn / QW;
    v4i xr[NR][UX / QW];
    uint32_t xsr[NR];
    Chunk<FMT, MT, NT, KC> wb[D];
    if constexpr(FU)
    {
#pragma unroll
        for(int d = 0; d < D; d++)
            if(d < NCH)
                load_w<FMT, MT, NT, KC, WAUX, SGR>(wb[d], src, d * KC);
        fu_wait(*fw);
    }
#pragma unroll
    for(int t = 0; t < NR; t++)
    {
        const int r = ix + WI * t;
        if(r < KT * MT)
        {
            const int i = r / MT;
            int xo = src.xo[0], xso = src.xso[0];
#pragma unroll
            for(int m = 1; m < MT; m++)
            {
                xo  = r % MT == m ? src.xo[m] : xo;
                xso = r % MT == m ? src.xso[m] : xso;
            }
#pragma unroll
            for(int a = 0; a < UX / QW; a++)
                xr[t][a] = __builtin_amdgcn_raw_buffer_load_b128(src.rx, xo, i * (X16 ? 256 : 128) +
                                                                                 (X16 ? 16 : 64) * (qx + QW * a), FU ? 16 : 0);
            if constexpr(!X16)
                xsr[t] = qx == 0 ? __builtin_amdgcn_raw_buffer_load_b8(src.rxs, xso, i * 4, 0) : 0;
        }
    }
    if constexpr(!FU)
    {
#pragma unroll
        for(int d = 0; d < D; d++)
            if(d < NCH)
                load_w<FMT, MT, NT, KC, WAUX, SGR>(wb[d], src, d * KC);
    }
    __shared__ uint32_t xs[X16 ? 1 : KT * MT * 64];
#pragma unroll
    for(int t = 0; t < NR; t++)
    {
        const int r = ix + WI * t;
        if(r < KT * MT)
        {
#pragma unroll
            for(int a = 0; a < UX / QW; a++)
                xl[(r * UX + qx + QW * a) * 64 + lane] = xr[t][a];
            if constexpr(!X16)
                if(qx == 0)
                    xs[r * 64 + lane] = xsr[t];
        }
    }
    LX_LDS_BAR();

    v4f acc[NT][MT], accn[NT][MT];
#pragma unroll
    for(int j = 0; j < NT; j++)
#pragma unroll
        for(int m = 0; m < MT; m++)
        {
            acc[j][m]  = v4f{0, 0, 0, 0};
            accn[j][m] = v4f{0, 0, 0, 0};
        }
#pragma unroll
    for(int c = 0; c < NCH; c++)
    {
        Chunk<FMT, MT, NT, KC>& cur = wb[c % D];
#pragma unroll
        for(int i = 0; i < KC; i++)
#pragma unroll
            for(int m = 0; m < MT; m++)
            {
                const int r = (c * KC + i) * MT + m;
                if constexpr(X16)
                {
#pragma unroll
                    for(int q = 0; q < 4; q++)
                        cur.x[i][m].v[q] = __builtin_bit_cast(v8h, xl[(r * 4 + q) * 64 + lane]);
                }
                else
                {
                    const v4i lo   = xl[(r * 2) * 64 + lane];
                    const v4i hi   = xl[(r * 2 + 1) * 64 + lane];
                    cur.x[i][m].v = v8i{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
                    cur.x[i][m].s = xs[r * 64 + lane];
                }
            }
        if constexpr(SGR)
            fill_sg(cur, sg, c * KC / 2);
        if constexpr(FL & 1)
            __builtin_amdgcn_s_setprio(1);
        compute_chunk<FMT, MT, NT, KC>(cur, acc, accn, false, g, nullptr);
        if constexpr(FL & 1)
            __builtin_amdgcn_s_setprio(0);
        if(c + D < NCH)
            load_w<FMT, MT, NT, KC, WAUX, SGR>(cur, src, (c + D) * KC);
    }
    __syncthreads(); // X LDS is reused by the epilogue
    epilogue<M8 ? F_F16 : FMT, false, MT, NT, WN, 1, true>(p, acc, e, mblk, tok, slot, xv, wn, g, col, lane,
                                                          (float*)xl, fwo);
}

template <int FMT, int MT, int NT, int WN, int KC, int D, int KT, int FL>
__global__ __launch_bounds__(64 * WN) __attribute__((amdgpu_waves_per_eu((FL & 2) ? 2 : 1))) void moe_s2x_gemm(
    Params p)
{
    constexpr int BM = 16 * MT;
    const int nvalid = p.num_valid[0];
#if DEC_XCD == 2
    const int mblk = blk_map(p, BM, nvalid);
    if(mblk < 0)
        return;
#else
    const int mblk = blk_y();
#endif
    const int e = p.sorted_eids[mblk];
    asm volatile("" ::"s"(e));
    if(mblk * BM >= nvalid)
        return;
    constexpr int L16 = s2x_lds_v4<F_F16, MT, NT, WN, KT>();
    constexpr int LF  = s2x_lds_v4<FMT, MT, NT, WN, KT>();
    __shared__ v4i xl[LF > L16 || FMT != F_F16 ? LF : L16];
    if constexpr(FMT == F_F16)
    {
        if(e == p.mx8_e)
        {
            // same weight register bytes per chunk: half the K-tiles (twice the chunks
            // in flight for NT <= 2)
            constexpr int KC8 = KC > 1 ? KC / 2 : 1;
            constexpr int D8  = NT <= 2 && 2 * D * KC8 <= KT ? 2 * D : D;
            s2x_body<F_MX8, MT, NT, WN, KC8, D8, KT, FL>(p, mblk, e, xl);
            return;
        }
    }
    s2x_body<FMT, MT, NT, WN, KC, D, KT, FL>(p, mblk, e, xl);
}

// ---------------------------------------------------------------- fused small-T kernel
// One launch for T * topk <= 64 (F16 path, MT = 1): no sort and no cast kernel. Grid
// (max(gx1, gx2), 2 n), n = T * topk: rows y < n are stage1 WGs of slot y, rows n + y the
// stage2 WGs of slot y. Slot y is active iff topk_ids[y] has no earlier occurrence; its
// expert then has at most T <= 16 tokens (one 16-column tile). Every WG derives its routing
// from topk_ids. Stage1 reads bf16 X and casts in registers; slot-0 stage1 WGs zero the
// output first. Stage2 WGs issue their weight loads, then wait on the epoch flags of their
// slot and slot 0. Stage2 WGs have larger linear ids than all stage1 WGs, so (in-order WG
// dispatch) a waiting WG never blocks a stage1 WG it depends on.
struct FusedParams
{
    Params p1, p2;
    const int* ids;  // [T * topk]
    const float* tw; // [T * topk]
    int* flags;      // [64 * FU_FS], epochs
    int n, gx1, gx2, ep;
};

template <int NT1, int WK1, int KCF1, int NT2, int WN2, int KC2, int D2, int KT2, int FL2>
__global__ __launch_bounds__(64 * WK1) void moe_fused_small(FusedParams f)
{
    static_assert(WK1 == WN2, "stage1 and stage2 WG sizes differ");
    constexpr int KC1  = KCF1 & 255;
    constexpr int SGR1 = (KCF1 & SGF) ? ((KCF1 & SGL) ? 2 : 1) : 0;
    const int lane     = threadIdx.x & 63;
    const int n        = f.n;
    const bool st2     = (int)blockIdx.y >= n;
    if((FU_MODE == 1 && st2) || (FU_MODE == 2 && !st2) || FU_MODE == 3)
        return;
    const int y        = st2 ? blockIdx.y - n : blockIdx.y;
    const int bx       = blockIdx.x;
    if(bx >= (st2 ? f.gx2 : f.gx1))
        return;
    const Params& P = st2 ? f.p2 : f.p1;
    const int topk  = P.topk;
    const int id    = lane < n ? f.ids[lane] : -1;
    const int e     = __builtin_amdgcn_readfirstlane(__shfl(id, y));
    const uint64_t m = __ballot(id == e);
    if(m & ((1ull << y) - 1))
        return; // not the first occurrence of e
    // column c of the tile = c-th set bit of m (m is wave-uniform)
    const int col = lane & 15;
    int kk        = -1;
    uint64_t mm   = m;
#pragma unroll
    for(int c = 0; c < 16; c++)
    {
        const int b = mm ? (int)__builtin_ctzll(mm) : -1;
        kk          = col == c ? b : kk;
        mm &= mm - 1;
    }
    const int fsid[1] = {kk >= 0 ? ((kk % topk) << 24) | (kk / topk) : P.T};
    if(!st2)
    {
        if(y == 0)
        {
            // zero this WG's share of the bf16 output
            const int n16 = P.T * f.p2.n_out * 2 / 16;
            const int a   = bx * n16 / f.gx1;
            const int b   = (bx + 1) * n16 / f.gx1;
            for(int i = 2 * a + (int)threadIdx.x; i < 2 * b; i += blockDim.x)
                __hip_atomic_store((uint64_t*)f.p2.out + i, (uint64_t)0, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
        }
        gemm_body<F_F16, true, 1, NT1, WK1, 1, KC1, false, 1, SGR1, -1, true>(P, y, e, 1, fsid);
        // this wave's sc1 stores are complete (gfx9 counts stores in vmcnt), then the WG's
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
        if(threadIdx.x == 0)
            __hip_atomic_store(f.flags + y * FU_FS + bx, f.ep, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    else
    {
        const float fw[1] = {kk >= 0 ? f.tw[kk] : 0.0f};
        const FuWait w    = {f.flags, y, f.gx1, f.ep};
        __shared__ v4i xl[s2x_lds_v4<F_F16, 1, NT2, WN2, KT2>()];
        s2x_body<F_F16, 1, NT2, WN2, KC2, D2, KT2, FL2, true>(P, y, e, xl, fsid, fw, &w);
    }
}

#define SB(kc) (256 + (kc))
// FU(id, NT1, WK1, KCF1, NT2, WN2, KC2, D2, KT2, FL2): stage1 = generic F16 config
// (MT 1, WN 1), stage2 = SX F16 config (MT 1).
#define FU_LIST(U)                                                                    \
    U(900, 2, 4, SB(6), 2, 4, 2, 3, 6, 4)                                              \
    U(901, 4, 4, SB(4) | SGF, 2, 4, 2, 3, 6, 4)                                        \
    U(902, 2, 4, SB(6), 4, 4, 2, 3, 6, 4)                                              \
    U(903, 4, 4, SB(4) | SGF, 4, 4, 2, 3, 6, 4)                                        \
    U(904, 2, 8, SB(6), 2, 8, 2, 3, 6, 4)                                              \
    U(905, 4, 4, SB(2) | SGF | SGL, 2, 4, 2, 3, 6, 4)                                  \
    U(910, 2, 4, SB(6), 2, 4, 2, 3, 12, 0)                                             \
    U(911, 4, 4, SB(4) | SGF, 2, 4, 2, 3, 12, 0)                                       \
    U(912, 4, 4, SB(4) | SGF, 4, 4, 2, 3, 12, 0)                                       \
    U(920, 2, 4, SB(6), 2, 4, 2, 3, 24, 0)                                             \
    U(921, 4, 4, SB(4) | SGF, 2, 4, 2, 3, 24, 0)                                       \
    U(922, 4, 4, SB(4) | SGF, 4, 4, 2, 3, 24, 0)

// info: {id, nt1, wk1, kcf1, nt2, wn2, kt2, fl2}
extern "C" int dec_fu_info(int id, int* out)
{
#define U_INFO(ID, N1, W1, C1, N2, W2, KC2, D2, KT2, FL2)                                  \
    if(id == ID)                                                                       \
    {                                                                                  \
        int v[8] = {ID, N1, W1, C1, N2, W2, KT2, FL2};                                 \
        for(int k = 0; k < 8; k++)                                                     \
            out[k] = v[k];                                                             \
        return 0;                                                                      \
    }
    FU_LIST(U_INFO)
    return -1;
}

extern "C" int dec_fu_launch(int id, const FusedParams* hf, hipStream_t stream)
{
    const FusedParams f = *hf;
    if(f.n > 64 || f.n < 1 || f.gx1 > FU_FS)
        return -2;
    const int gx = f.gx1 > f.gx2 ? f.gx1 : f.gx2;
#define U_LAUNCH(ID, N1, W1, C1, N2, W2, KC2, D2, KT2, FL2)                                \
    if(id == ID)                                                                       \
    {                                                                                  \
        if(f.p2.K != 128 * (KT2) || (((C1) & SGF) && f.p1.K != 128 * 12 * (W1)))       \
            return -3;                                                                 \
        size_t lds = (size_t)((W1) - 1) * (N1) * 64 * sizeof(v4f);                      \
        hipLaunchKernelGGL((moe_fused_small<N1, W1, C1, N2, W2, KC2, D2, KT2, FL2>),     \
                           dim3(gx, 2 * f.n), dim3(64 * (W1)), lds, stream, f);         \
        return (int)hipGetLastError();                                                 \
    }
    FU_LIST(U_LAUNCH)
    return -1;
}

// ---------------------------------------------------------------- activation prep
// bf16 [rows, K] -> fp8 e4m3 [rows, K] + e8m0 [rows, K/32] (RoundUp scale as AITER).
// One thread per 32-element block.
__global__ void quant_mxfp8_kernel(const uint16_t* x, uint8_t* q, uint8_t* sc, int nblk)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= nblk)
        return;
    const v4i* src = (const v4i*)(x + (size_t)b * 32);
    float f[32];
    float amax = 0.0f;
#pragma unroll
    for(int i = 0; i < 4; i++)
    {
        v4i d = src[i];
#pragma unroll
        for(int k = 0; k < 4; k++)
        {
            f[8 * i + 2 * k]     = __uint_as_float((uint32_t)d[k] << 16);
            f[8 * i + 2 * k + 1] = __uint_as_float((uint32_t)d[k] & 0xffff0000u);
            amax = fmaxf(amax, fmaxf(fabsf(f[8 * i + 2 * k]), fabsf(f[8 * i + 2 * k + 1])));
        }
    }
    uint32_t qb = __float_as_uint(amax / 448.0f);
    uint32_t eb = (qb >> 23) + ((qb & 0x7fffffu) != 0);
    eb          = eb > 254 ? 254 : eb;
    float mult  = __uint_as_float((254u - eb) << 23);
    int o[8];
#pragma unroll
    for(int i = 0; i < 8; i++)
    {
        int pk = __builtin_amdgcn_cvt_pk_fp8_f32(f[4 * i] * mult, f[4 * i + 1] * mult, 0, false);
        o[i]   = __builtin_amdgcn_cvt_pk_fp8_f32(f[4 * i + 2] * mult, f[4 * i + 3] * mult, pk, true);
    }
    v4i* dst = (v4i*)(q + (size_t)b * 32);
    dst[0]   = v4i{o[0], o[1], o[2], o[3]};
    dst[1]   = v4i{o[4], o[5], o[6], o[7]};
    sc[b]    = (uint8_t)eb;
}

// bf16 -> f16 (exact for |x| < 65504), 8 elements per thread.
__global__ void cast_f16_kernel(const uint16_t* x, uint16_t* y, int n8)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n8)
        return;
    v4i d = ((const v4i*)x)[i];
    v4i o;
#pragma unroll
    for(int k = 0; k < 4; k++)
    {
        float a = __uint_as_float((uint32_t)d[k] << 16);
        float b = __uint_as_float((uint32_t)d[k] & 0xffff0000u);
        v2h h   = {(_Float16)a, (_Float16)b};
        o[k]    = __builtin_bit_cast(int, h);
    }
    ((v4i*)y)[i] = o;
}

// ---------------------------------------------------------------- fused prep
// One launch replaces moe_sorting + activation quant: block 0 sorts the routing (one
// workgroup), blocks 1.. quantize X (MXFP8, or the f16 cast) and zero the bf16 output
// for the stage2 atomics. Sorted layout matches the gemm kernels: per expert,
// ceil(count / BM) blocks of BM entries (token | slot << 24, padding token = T).
constexpr int SORT_THREADS = 1024;
constexpr int SORT_PT      = 4; // T * topk <= 4096
constexpr int SORT_MAXE    = 512;
constexpr int SORT_MAXB    = SORT_THREADS * SORT_PT / 16 + SORT_MAXE; // BM >= 16
constexpr int SORT_QB      = 256;
#ifndef SORT_HEAVY
#define SORT_HEAVY 1
#endif

struct PrepParams
{
    const uint16_t* x;
    uint8_t* q;
    uint8_t* sc;
    uint16_t* out;
    const int* ids;
    const float* w;
    int* sorted_ids;
    float* sorted_w;
    int* sorted_eids;
    int* num_valid;
    int T, H, topk, E, BM, nblk;
};

// XM: activation prep 0 = MXFP8 quant, 1 = f16 cast, 2 = none (bf16 used as is).
template <int XM>
__global__ __launch_bounds__(SORT_THREADS) void sort_quant_kernel(PrepParams p)
{
    const int tid = threadIdx.x;
    if(blockIdx.x > 0)
    {
#ifdef SQ_NOQUANT
        return;
#endif
        // SORT_QB 32-element blocks per workgroup: more workgroups for small T
        const int b = (blockIdx.x - 1) * SORT_QB + tid;
        if(tid >= SORT_QB || b >= p.T * p.H / 32)
            return;
        const v4i* src = (const v4i*)(p.x + (size_t)b * 32);
        v4i d[4];
#pragma unroll
        for(int i = 0; i < 4; i++)
            d[i] = src[i];
        v4i* zo = (v4i*)(p.out + (size_t)b * 32 * (DEC_S2F32 ? 2 : 1));
#pragma unroll
        for(int i = 0; i < (DEC_S2F32 ? 8 : 4); i++)
            zo[i] = v4i{0, 0, 0, 0};
        if constexpr(XM == 2)
            ;
        else if constexpr(XM == 1)
        {
            v4i* dst = (v4i*)((uint16_t*)p.q + (size_t)b * 32);
#pragma unroll
            for(int i = 0; i < 4; i++)
            {
                v4i o;
#pragma unroll
                for(int k = 0; k < 4; k++)
                {
                    float a = __uint_as_float((uint32_t)d[i][k] << 16);
                    float c = __uint_as_float((uint32_t)d[i][k] & 0xffff0000u);
                    v2h h   = {(_Float16)a, (_Float16)c};
                    o[k]    = __builtin_bit_cast(int, h);
                }
                dst[i] = o;
            }
        }
        else
        {
            float f[32];
            float amax = 0.0f;
#pragma unroll
            for(int i = 0; i < 4; i++)
#pragma unroll
                for(int k = 0; k < 4; k++)
                {
                    f[8 * i + 2 * k]     = __uint_as_float((uint32_t)d[i][k] << 16);
                    f[8 * i + 2 * k + 1] = __uint_as_float((uint32_t)d[i][k] & 0xffff0000u);
                    amax = fmaxf(amax, fmaxf(fabsf(f[8 * i + 2 * k]), fabsf(f[8 * i + 2 * k + 1])));
                }
            uint32_t qb = __float_as_uint(amax / 448.0f);
            uint32_t eb = (qb >> 23) + ((qb & 0x7fffffu) != 0);
            eb          = eb > 254 ? 254 : eb;
            float mult  = __uint_as_float((254u - eb) << 23);
            int o[8];
#pragma unroll
            for(int i = 0; i < 8; i++)
            {
                int pk = __builtin_amdgcn_cvt_pk_fp8_f32(f[4 * i] * mult, f[4 * i + 1] * mult, 0, false);
                o[i]   = __builtin_amdgcn_cvt_pk_fp8_f32(f[4 * i + 2] * mult, f[4 * i + 3] * mult, pk, true);
            }
            v4i* dst = (v4i*)(p.q + (size_t)b * 32);
            dst[0]   = v4i{o[0], o[1], o[2], o[3]};
            dst[1]   = v4i{o[4], o[5], o[6], o[7]};
            p.sc[b]  = (uint8_t)eb;
        }
        return;
    }

    __shared__ int cnt[SORT_MAXE];
    __shared__ int off[SORT_MAXE + 1];   // first block of each expert
    __shared__ int blk_e[SORT_MAXB];     // expert of each block
    const int n  = p.T * p.topk;
    const int BM = p.BM;
    // all global loads first: one memory latency instead of SORT_PT dependent ones
    int ee[SORT_PT], rr[SORT_PT];
    float ww[SORT_PT];
#pragma unroll
    for(int i = 0; i < SORT_PT; i++)
    {
        const int idx = tid + SORT_THREADS * i;
        ee[i]         = idx < n ? p.ids[idx] : 0;
        ww[i]         = idx < n ? p.w[idx] : 0.0f;
    }
    for(int e = tid; e < p.E; e += SORT_THREADS)
        cnt[e] = 0;
    __syncthreads();
#pragma unroll
    for(int i = 0; i < SORT_PT; i++)
        rr[i] = tid + SORT_THREADS * i < n ? atomicAdd(&cnt[ee[i]], 1) : 0;
    __syncthreads();
    if(tid < 64)
    {
        // SORT_HEAVY: experts with more than one block go first (stable), so the long
        // multi-block experts (e.g. a fused shared expert) do not form the grid tail.
        const int per = (p.E + 63) / 64;
        const int e0  = tid * per;
        int sh = 0, sl = 0;
        for(int k = 0; k < per; k++)
            if(e0 + k < p.E)
            {
                const int nb = (cnt[e0 + k] + BM - 1) / BM;
                if(SORT_HEAVY && nb > 1)
                    sh += nb;
                else
                    sl += nb;
            }
        int ih = sh, il = sl;
#pragma unroll
        for(int d = 1; d < 64; d <<= 1)
        {
            const int vh = __shfl_up(ih, d);
            const int vl = __shfl_up(il, d);
            ih += tid >= d ? vh : 0;
            il += tid >= d ? vl : 0;
        }
        const int th = __shfl(ih, 63);
        int xh       = ih - sh;
        int xl       = th + il - sl;
        for(int k = 0; k < per; k++)
            if(e0 + k < p.E)
            {
                const int nb = (cnt[e0 + k] + BM - 1) / BM;
                if(SORT_HEAVY && nb > 1)
                {
                    off[e0 + k] = xh;
                    xh += nb;
                }
                else
                {
                    off[e0 + k] = xl;
                    xl += nb;
                }
            }
        if(tid == 63)
        {
            off[p.E]       = th + il;
            p.num_valid[0] = (th + il) * BM;
        }
    }
    __syncthreads();
    for(int e = tid; e < p.E; e += SORT_THREADS)
        for(int b = off[e]; b < off[e] + (cnt[e] + BM - 1) / BM; b++)
            blk_e[b] = e;
    __syncthreads();
    const int nv = off[p.E];
    for(int b = tid; b < p.nblk; b += SORT_THREADS)
        p.sorted_eids[b] = b < nv ? blk_e[b] : 0;
    for(int pos = tid; pos < nv * BM; pos += SORT_THREADS)
    {
        const int e = blk_e[pos / BM];
        if(pos - off[e] * BM >= cnt[e])
        {
            p.sorted_ids[pos] = p.T;
            p.sorted_w[pos]   = 0.0f;
        }
    }
#pragma unroll
    for(int i = 0; i < SORT_PT; i++)
    {
        const int idx = tid + SORT_THREADS * i;
        if(idx < n)
        {
            const int pos     = off[ee[i]] * BM + rr[i];
            p.sorted_ids[pos] = ((idx % p.topk) << 24) | (idx / p.topk);
            p.sorted_w[pos]   = ww[i];
        }
    }
}

extern "C" int dec_sort_quant(const PrepParams* hp, int xm, hipStream_t stream)
{
    const PrepParams p = *hp;
    if(p.T * p.topk > SORT_THREADS * SORT_PT || p.E > SORT_MAXE)
        return -2;
    const int nq = (p.T * p.H / 32 + SORT_QB - 1) / SORT_QB;
    if(xm == 3) // sort only
        hipLaunchKernelGGL(sort_quant_kernel<2>, dim3(1), dim3(SORT_THREADS), 0, stream, p);
    else     if(xm == 2)
        hipLaunchKernelGGL(sort_quant_kernel<2>, dim3(1 + nq), dim3(SORT_THREADS), 0, stream, p);
    else if(xm == 1)
        hipLaunchKernelGGL(sort_quant_kernel<1>, dim3(1 + nq), dim3(SORT_THREADS), 0, stream, p);
    else
        hipLaunchKernelGGL(sort_quant_kernel<0>, dim3(1 + nq), dim3(SORT_THREADS), 0, stream, p);
    return (int)hipGetLastError();
}

extern "C" int dec_quant_mxfp8(const void* x, void* q, void* sc, int rows, int K, hipStream_t stream)
{
    int nblk = rows * K / 32;
    hipLaunchKernelGGL(quant_mxfp8_kernel, dim3((nblk + 255) / 256), dim3(256), 0, stream,
                       (const uint16_t*)x, (uint8_t*)q, (uint8_t*)sc, nblk);
    return (int)hipGetLastError();
}

extern "C" int dec_cast_f16(const void* x, void* y, int n, hipStream_t stream)
{
    int n8 = n / 8;
    hipLaunchKernelGGL(cast_f16_kernel, dim3((n8 + 255) / 256), dim3(256), 0, stream,
                       (const uint16_t*)x, (uint16_t*)y, n8);
    return (int)hipGetLastError();
}

// ---------------------------------------------------------------- host side
// X(id, FMT, S1, MT, NT, WK, WN, KCF). KCF = KC (double-buffered chunks of KC K-tiles)
// or SB(KC) (single-buffered: load chunk, compute chunk). KTW = K / 128 / WK must be a
// multiple of KC (KC even). Stage1 and stage2 configs of one run must share MT.
#define SB(kc) (256 + (kc))
#define MLB(n) ((n) << 12)
#define FMT_CFGS(X, F, B)                                                             \
    X(B + 0, F, true, 1, 4, 8, 1, 2)                                                   \
    X(B + 1, F, true, 1, 4, 8, 1, SB(6))                                                \
    X(B + 2, F, true, 1, 4, 12, 1, SB(4))                                               \
    X(B + 3, F, true, 1, 4, 4, 1, SB(4))                                                \
    X(B + 4, F, true, 1, 4, 6, 1, 2)                                                   \
    X(B + 10, F, true, 2, 4, 8, 1, SB(2))                                               \
    X(B + 11, F, true, 2, 4, 4, 1, SB(4))                                               \
    X(B + 12, F, true, 2, 4, 4, 1, 2)                                                  \
    X(B + 13, F, true, 2, 4, 12, 1, SB(4))                                              \
    X(B + 14, F, true, 2, 4, 6, 1, SB(4))                                               \
    X(B + 20, F, true, 4, 4, 4, 1, 2)                                                  \
    X(B + 21, F, true, 4, 4, 4, 1, SB(2))                                               \
    X(B + 30, F, true, 1, 4, 8, 1, SB(6) | MLB(2))                                      \
    X(B + 31, F, true, 1, 4, 8, 1, SB(6) | MLB(4))                                      \
    X(B + 32, F, true, 1, 4, 8, 1, SB(6) | MLB(8))                                      \
    X(B + 33, F, true, 1, 4, 12, 1, SB(4) | MLB(4))                                     \
    X(B + 34, F, true, 2, 4, 8, 1, SB(6) | MLB(4))                                      \
    X(B + 35, F, true, 2, 4, 12, 1, SB(4) | MLB(4))                                     \
    X(B + 50, F, false, 1, 4, 1, 1, 2)                                                 \
    X(B + 51, F, false, 1, 4, 1, 1, SB(6))                                              \
    X(B + 52, F, false, 1, 2, 1, 1, SB(6))                                              \
    X(B + 53, F, false, 1, 1, 1, 1, SB(6))                                              \
    X(B + 54, F, false, 1, 2, 3, 1, SB(2))                                              \
    X(B + 55, F, false, 1, 4, 3, 1, SB(2))                                              \
    X(B + 56, F, false, 1, 1, 3, 1, SB(2))                                              \
    X(B + 60, F, false, 2, 4, 1, 1, 2)                                                 \
    X(B + 61, F, false, 2, 4, 3, 1, SB(2))                                              \
    X(B + 62, F, false, 2, 2, 3, 1, SB(2))                                              \
    X(B + 63, F, false, 2, 2, 1, 1, SB(6))                                              \
    X(B + 70, F, false, 4, 4, 1, 1, 2)                                                 \
    X(B + 71, F, false, 4, 4, 3, 1, SB(2))                                              \
    X(B + 72, F, false, 4, 2, 3, 1, SB(2))                                              \
    X(B + 80, F, false, 1, 4, 1, 1, SB(6) | MLB(4))                                     \
    X(B + 81, F, false, 1, 4, 3, 1, SB(2) | MLB(4))                                     \
    X(B + 82, F, false, 1, 2, 3, 1, SB(2) | MLB(4))                                     \
    X(B + 83, F, false, 1, 4, 3, 1, SB(2) | MLB(8))                                     \
    X(B + 84, F, false, 2, 4, 3, 1, SB(2) | MLB(4))                                     \
    X(B + 85, F, false, 2, 4, 1, 1, SB(6) | MLB(4))

// f16 stage1 output has no 32-column scale block, so one gate/up tile pair per wave works.
#define F16_EXTRA(X, F, B)                                                            \
    X(B + 1000, F, true, 3, 8, 2, 1, SB(2))                                            \
    X(B + 1001, F, false, 3, 8, 3, 1, SB(2))                                           \
    X(B + 1002, F, false, 3, 4, 3, 1, SB(2))                                           \
    X(B + 1003, F, true, 3, 8, 3, 1, SB(2))                                            \
    X(B + 1004, F, true, 3, 4, 2, 1, SB(2))                                            \
    X(B + 1005, F, true, 3, 8, 2, 1, SB(4))                                            \
    X(B + 1006, F, true, 3, 4, 2, 2, SB(2))                                            \
    X(B + 1008, F, false, 3, 8, 2, 1, SB(2))                                           \
    X(B + 1009, F, false, 3, 4, 2, 1, SB(2))                                           \
    X(B + 8, F, true, 3, 8, 4, 1, SB(2) | WP2 | SGF | SGL)                             \
    X(B + 9, F, true, 3, 8, 4, 1, SB(2) | SGF | SGL)                                   \
    X(B + 29, F, true, 3, 4, 4, 1, SB(2) | SGF | SGL)                                  \
    X(B + 36, F, true, 3, 4, 4, 1, SB(2) | WP2 | SGF)                                  \
    X(B + 5, F, true, 1, 2, 8, 1, SB(6))                                                \
    X(B + 6, F, true, 1, 2, 12, 1, SB(4))                                               \
    X(B + 7, F, true, 1, 2, 4, 1, SB(6))                                                \
    X(B + 15, F, true, 2, 8, 4, 1, SB(2) | WP2)                                         \
    X(B + 16, F, true, 2, 8, 2, 1, SB(2))                                               \
    X(B + 17, F, true, 1, 8, 4, 1, SB(2))                                               \
    X(B + 18, F, true, 2, 8, 4, 1, 2)                                                  \
    X(B + 19, F, true, 2, 8, 8, 1, SB(2))                                               \
    X(B + 65, F, false, 2, 8, 3, 1, SB(2))                                              \
    X(B + 66, F, false, 2, 8, 1, 1, SB(6))                                              \
    X(B + 67, F, false, 1, 8, 3, 1, SB(2))                                              \
    X(B + 68, F, false, 2, 8, 3, 1, 2)                                                  \
    X(B + 22, F, true, 2, 4, 2, 2, SB(2))                                               \
    X(B + 23, F, true, 2, 4, 1, 4, SB(2))                                               \
    X(B + 24, F, true, 1, 4, 2, 2, SB(4))                                               \
    X(B + 25, F, true, 1, 4, 1, 4, SB(4))                                               \
    X(B + 26, F, true, 2, 4, 4, 2, SB(2))                                               \
    X(B + 74, F, false, 2, 4, 3, 2, SB(2))                                              \
    X(B + 75, F, false, 2, 4, 1, 4, SB(6))                                              \
    X(B + 76, F, false, 1, 4, 3, 2, SB(2))                                              \
    X(B + 77, F, false, 1, 4, 1, 4, SB(6))                                              \
    X(B + 40, F, true, 1, 4, 4, 1, SB(4) | SGF)                                         \
    X(B + 41, F, true, 2, 8, 4, 1, SB(2) | WP2 | SGF)                                   \
    X(B + 42, F, true, 1, 4, 4, 1, SB(2) | SGF)                                         \
    X(B + 43, F, true, 1, 4, 4, 1, SB(6) | SGF)                                         \
    X(B + 44, F, true, 2, 4, 4, 1, SB(4) | SGF)                                         \
    X(B + 45, F, true, 1, 8, 4, 1, SB(2) | SGF)                                         \
    X(B + 46, F, true, 1, 2, 4, 1, SB(6) | SGF)                                         \
    X(B + 47, F, true, 2, 4, 4, 1, SB(2) | SGF)                                         \
    X(B + 48, F, true, 2, 8, 4, 1, SB(2) | SGF)                                         \
    X(B + 49, F, true, 2, 8, 4, 1, SB(2) | WP2 | SGF | SGL)                             \
    X(B + 39, F, true, 2, 8, 4, 1, SB(2) | SGF | SGL)                                   \
    X(B + 38, F, true, 2, 4, 4, 1, SB(2) | SGF | SGL)                                   \
    X(B + 37, F, true, 1, 4, 4, 1, SB(2) | SGF | SGL)                                   \
    X(B + 86, F, false, 1, 4, 1, 1, SB(2) | SGF)                                        \
    X(B + 87, F, false, 1, 8, 1, 1, SB(2) | SGF)                                        \
    X(B + 88, F, false, 2, 4, 1, 1, SB(2) | SGF)                                        \
    X(B + 89, F, false, 2, 8, 1, 1, SB(2) | SGF)                                        \
    X(B + 90, F, false, 1, 4, 2, 1, SB(2) | SGF)                                        \
    X(B + 91, F, false, 1, 8, 2, 1, SB(2) | SGF)                                        \
    X(B + 92, F, false, 2, 4, 2, 1, SB(2) | SGF)                                        \
    X(B + 93, F, false, 2, 8, 2, 1, SB(2) | SGF)                                        \
    X(B + 94, F, false, 2, 8, 1, 1, SB(2) | SGF | SGL)                                  \
    X(B + 95, F, false, 2, 8, 2, 1, SB(2) | SGF | SGL)                                  \
    X(B + 96, F, false, 2, 4, 2, 2, SB(2) | SGF)                                        \
    X(B + 97, F, false, 2, 4, 1, 2, SB(2) | SGF)                                        \
    X(B + 98, F, false, 1, 4, 1, 1, SB(4) | SGF)                                        \
    X(B + 99, F, false, 1, 4, 2, 1, SB(4) | SGF)                                        \
    X(B + 57, F, false, 2, 16, 3, 1, SB(2))                                             \
    X(B + 58, F, false, 1, 16, 3, 1, SB(2))                                             \
    X(B + 59, F, false, 2, 8, 6, 1, SB(2))                                              \
    X(B + 64, F, false, 2, 4, 6, 1, SB(2))                                              \
    X(B + 69, F, false, 4, 8, 3, 1, SB(2))                                              \
    X(B + 73, F, false, 4, 4, 3, 1, SB(2) | WP2)                                        \
    X(B + 27, F, true, 4, 8, 4, 1, SB(2))                                               \
    X(B + 28, F, true, 4, 4, 4, 1, SB(2) | WP2)

// MXFP4-only generic configs (NT8: 128 output columns per WG as the AITER opus stage2)
#define MX_EXTRA(X, F, B)                                                             \
    X(B + 64, F, false, 2, 8, 3, 1, SB(2))                                              \
    X(B + 65, F, false, 2, 8, 3, 1, SB(2) | WP2)

#define CFG_LIST(X)                                                                   \
    FMT_CFGS(X, F_MXFP4, 0)                                                            \
    MX_EXTRA(X, F_MXFP4, 0)                                                            \
    FMT_CFGS(X, F_TERMS, 100)                                                          \
    FMT_CFGS(X, F_F16, 200)                                                            \
    F16_EXTRA(X, F_F16, 200)                                                           \
    FMT_CFGS(X, F_FP8, 300)                                                            \
    BF16_S1(X, F_BF16, 600)

// old kernel, bf16 stage1 (stage2 stays f16)
#define BF16_S1(X, F, B)                                                              \
    X(B + 0, F, true, 1, 4, 8, 1, 2)                                                   \
    X(B + 3, F, true, 1, 4, 4, 1, SB(4))                                                \
    X(B + 5, F, true, 1, 2, 8, 1, SB(6))                                                \
    X(B + 11, F, true, 2, 4, 4, 1, SB(4))                                               \
    X(B + 12, F, true, 2, 4, 4, 1, 2)                                                  \
    X(B + 15, F, true, 2, 8, 4, 1, SB(2))

// LX kernel: Y(id, FMT, S1, MT, NT, WN, KC, D, NS). nch = K / 128 / KC must be a multiple
// of D; NS = 1 needs nch = 1.
#define LX_S1(Y, F, B)                                                                \
    Y(B + 0, F, true, 1, 4, 4, 2, 2, 2)                                                \
    Y(B + 1, F, true, 2, 4, 4, 2, 2, 2)                                                \
    Y(B + 2, F, true, 2, 4, 4, 2, 3, 2)                                                \
    Y(B + 3, F, true, 2, 2, 8, 2, 2, 2)                                                \
    Y(B + 4, F, true, 1, 4, 8, 2, 2, 2)                                                \
    Y(B + 5, F, true, 4, 4, 4, 2, 2, 2)                                                \
    Y(B + 6, F, true, 2, 4, 4, 4, 2, 2)                                                \
    Y(B + 7, F, true, 2, 4, 4, 2, 4, 2)                                                \
    Y(B + 8, F, true, 1, 8, 4, 2, 2, 2)
#define LX_S2(Y, F, B)                                                                \
    Y(B + 50, F, false, 1, 4, 4, 6, 1, 1)                                              \
    Y(B + 51, F, false, 2, 4, 4, 6, 1, 1)                                              \
    Y(B + 52, F, false, 2, 4, 8, 6, 1, 1)                                              \
    Y(B + 53, F, false, 1, 4, 8, 6, 1, 1)                                              \
    Y(B + 54, F, false, 2, 2, 8, 6, 1, 1)                                              \
    Y(B + 55, F, false, 4, 2, 4, 6, 1, 1)                                              \
    Y(B + 56, F, false, 2, 8, 4, 6, 1, 1)                                              \
    Y(B + 57, F, false, 1, 8, 4, 6, 1, 1)                                              \
    Y(B + 58, F, false, 2, 4, 4, 2, 2, 2)                                              \
    Y(B + 59, F, false, 2, 4, 4, 2, 3, 2)                                              \
    Y(B + 60, F, false, 2, 4, 8, 2, 3, 2)                                              \
    Y(B + 61, F, false, 1, 4, 4, 2, 3, 2)                                              \
    Y(B + 62, F, false, 2, 8, 4, 2, 3, 2)                                              \
    Y(B + 63, F, false, 2, 2, 8, 2, 3, 2)                                              \
    Y(B + 64, F, false, 2, 4, 4, 4, 3, 2)                                              \
    Y(B + 65, F, false, 1, 8, 4, 2, 3, 2)                                              \
    Y(B + 66, F, false, 2, 8, 2, 2, 2, 2)                                              \
    Y(B + 67, F, false, 2, 4, 2, 2, 2, 2)                                             
#define LX_LIST(Y)                                                                    \
    LX_S1(Y, F_F16, 400)                                                               \
    LX_S2(Y, F_F16, 400)                                                               \
    LX_S1(Y, F_BF16, 500)

// Stage2 X-in-LDS kernel: Z(id, FMT, MT, NT, WN, KC, D, KT, FL); K must be 128 KT.
// FX = 4 (F16 only): weight scales in the SG layout.
#define SX_S2(Z, F, B, FX)                                                                \
    Z(B + 0, F, 1, 2, 4, 2, 3, 6, (0) | FX)                                                   \
    Z(B + 1, F, 1, 4, 4, 2, 3, 6, (0) | FX)                                                   \
    Z(B + 2, F, 2, 2, 4, 2, 3, 6, (0) | FX)                                                   \
    Z(B + 3, F, 2, 4, 4, 2, 3, 6, (0) | FX)                                                   \
    Z(B + 4, F, 2, 2, 4, 2, 3, 6, (1) | FX)                                                   \
    Z(B + 5, F, 1, 2, 4, 2, 3, 6, (1) | FX)                                                   \
    Z(B + 6, F, 2, 2, 4, 6, 1, 6, (0) | FX)                                                   \
    Z(B + 7, F, 2, 2, 4, 2, 2, 6, (0) | FX)                                                   \
    Z(B + 8, F, 1, 2, 8, 2, 3, 6, (0) | FX)                                                   \
    Z(B + 9, F, 2, 1, 8, 2, 3, 6, (0) | FX)                                                   \
    Z(B + 10, F, 2, 4, 2, 2, 3, 6, (0) | FX)                                                  \
    Z(B + 11, F, 2, 2, 4, 2, 3, 6, (2) | FX)                                                  \
    Z(B + 12, F, 4, 2, 4, 2, 3, 6, (0) | FX)                                                  \
    Z(B + 13, F, 4, 1, 4, 2, 3, 6, (0) | FX)                                                  \
    Z(B + 14, F, 2, 4, 8, 2, 3, 6, (0) | FX)                                                  \
    Z(B + 15, F, 2, 2, 8, 2, 3, 6, (0) | FX)                                                  \
    Z(B + 16, F, 2, 8, 2, 2, 2, 6, (0) | FX)                                                  \
    Z(B + 17, F, 2, 4, 4, 2, 3, 6, (1) | FX)
// TP2 (I = 1536, KT = 12) and TP1 (I = 3072, KT = 24; MT = 2 would exceed the LDS)
#define SX_S2K(Z, F, B)                                                                   \
    Z(B + 0, F, 1, 2, 4, 2, 3, 12, 0)                                                     \
    Z(B + 1, F, 1, 4, 4, 2, 3, 12, 0)                                                     \
    Z(B + 2, F, 2, 2, 4, 2, 3, 12, 0)                                                     \
    Z(B + 3, F, 2, 4, 4, 2, 3, 12, 0)                                                     \
    Z(B + 4, F, 1, 2, 4, 2, 3, 24, 0)                                                     \
    Z(B + 5, F, 1, 4, 4, 2, 3, 24, 0)                                                     \
    Z(B + 6, F, 1, 4, 8, 2, 3, 24, 0)                                                     \
    Z(B + 7, F, 2, 4, 8, 2, 3, 12, 0)
// R8: MT = 3 (48-row blocks: about one block per expert at T = 1024, E128k4)
#define SX_R8(Z, F)                                                                       \
    Z(738, F, 3, 4, 4, 2, 3, 6, 4)                                                       \
    Z(739, F, 3, 2, 4, 2, 3, 6, 4)                                                       \
    Z(748, F, 3, 2, 4, 2, 3, 12, 0)                                                      \
    Z(749, F, 3, 4, 4, 2, 3, 12, 0)
#define SX_LIST(Z)                                                                    \
    SX_S2(Z, F_F16, 700, 0)                                                            \
    SX_S2(Z, F_F16, 720, 4)                                                            \
    SX_S2(Z, F_MXFP4, 800, 0)                                                         \
    SX_S2K(Z, F_F16, 740)                                                              \
    SX_R8(Z, F_F16)                                                                   \
    SX_S2K(Z, F_MXFP4, 840)

struct CfgInfo
{
    int id, fmt, s1, mt, nt, wk, wn, kc;
};

#define X_INFO(ID, F, S, M, N, K, W, C) {ID, F, S, M, N, K, W, C},
// LX configs: kc = KC | 1 << 16 | D << 20 | NS << 24
#define Y_INFO(ID, F, S, M, N, W, KC, D, NS) {ID, F, S, M, N, 1, W, (KC) | (1 << 16) | ((D) << 20) | ((NS) << 24)},
// SX configs: wk = KT, kc = KC | 2 << 16 | D << 20 | FL << 24
#define Z_INFO(ID, F, M, N, W, KC, D, KT, FL) {ID, F, 0, M, N, KT, W, (KC) | (2 << 16) | ((D) << 20) | ((FL) << 24)},
static const CfgInfo g_cfgs[] = {CFG_LIST(X_INFO) LX_LIST(Y_INFO) SX_LIST(Z_INFO)};

extern "C" int dec_num_cfgs() { return sizeof(g_cfgs) / sizeof(g_cfgs[0]); }
extern "C" void dec_cfg_info(int i, int* out)
{
    const CfgInfo& c = g_cfgs[i];
    int v[8]         = {c.id, c.fmt, c.s1, c.mt, c.nt, c.wk, c.wn, c.kc};
    for(int k = 0; k < 8; k++)
        out[k] = v[k];
}

extern "C" int dec_set_fp8_lut(const uint32_t* host_lut)
{
    return (int)hipMemcpyToSymbol(HIP_SYMBOL(c_fp8_lut), host_lut, sizeof(uint32_t) * 8 * 16 * 2);
}

extern "C" int dec_launch(int cfg, const Params* hp, int grid_x, int grid_y, hipStream_t stream)
{
    Params p = *hp;
    const int ktot = p.K; // X_LAUNCH has a macro parameter K
#define X_LAUNCH(ID, F, S, M, N, K, W, C)                                                   \
    if(cfg == ID)                                                                         \
    {                                                                                     \
        if(((C) & SGF) && ktot != 128 * 12 * (K))                                          \
            return -3;                                                                    \
        size_t lds = (K > 1) ? (size_t)(K - 1) * W * N * M * 64 * sizeof(v4f) : 0;        \
        hipLaunchKernelGGL((moe_dec_gemm<F, S, M, N, K, W, C>), dim3(grid_x, grid_y),    \
                           dim3(64 * K * W), lds, stream, p);                             \
        return (int)hipGetLastError();                                                    \
    }
    CFG_LIST(X_LAUNCH)
#define Y_LAUNCH(ID, F, S, M, N, W, KC, D, NS)                                              \
    if(cfg == ID)                                                                         \
    {                                                                                     \
        hipLaunchKernelGGL((moe_lx_gemm<F, S, M, N, W, KC, D, NS>), dim3(grid_x, grid_y), \
                           dim3(64 * W), 0, stream, p);                                   \
        return (int)hipGetLastError();                                                    \
    }
    LX_LIST(Y_LAUNCH)
#define Z_LAUNCH(ID, F, M, N, W, KC, D, KT, FL)                                             \
    if(cfg == ID)                                                                         \
    {                                                                                     \
        if(p.K != 128 * (KT))                                                             \
            return -3;                                                                    \
        hipLaunchKernelGGL((moe_s2x_gemm<F, M, N, W, KC, D, KT, FL>), dim3(grid_x, grid_y), \
                           dim3(64 * W), 0, stream, p);                                   \
        return (int)hipGetLastError();                                                    \
    }
    SX_LIST(Z_LAUNCH)
    return -1;
}
