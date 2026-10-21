// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Horizontally fused Q/K RMSNorm + partial NeoX RoPE, with optional
// sparse-layer KV/index-cache insert. This ports the AME/vLLM operator into
// aiter's pybind/JIT extension style.

#include "fused_qknorm_idxrqknorm.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "aiter_dispatch.h"
#include "aiter_stream.h"
#include "opus/opus.hpp"
#include "quant_utils.cuh"

namespace aiter {
namespace fused_qknorm_idxrqknorm_ops {

constexpr int kHeadDim = 128;
constexpr int kNumLanes = 32;
constexpr int kElemsPerLane = kHeadDim / kNumLanes;

__device__ __forceinline__ float warpReduceSum(float val)
{
#pragma unroll
    for(int mask = 16; mask > 0; mask >>= 1)
    {
        val += __shfl_xor(val, mask, 32);
    }
    return val;
}

__device__ __forceinline__ float warpReduceMax(float val)
{
#pragma unroll
    for(int mask = 16; mask > 0; mask >>= 1)
    {
        val = fmaxf(val, __shfl_xor(val, mask, 32));
    }
    return val;
}

// Per-token dynamic quant scale = amax / fp8_max so the largest |value| maps to the
// fp8 max. The fp8 max is arch-dependent: OCP e4m3fn (448) on gfx950+, e4m3fnuz (240)
// on gfx942. opus::finfo<cache_t>::max() resolves this at compile time per arch and
// matches the opus::cast<fp8> hardware clamp range, so do NOT hardcode 448 here.
template <typename cache_t>
__device__ __forceinline__ float fp8Max()
{
    return static_cast<float>(opus::finfo<cache_t>::max());
}

template <typename scalar_t>
__device__ __forceinline__ void loadElems(const scalar_t* __restrict__ src,
                                          float (&elems)[kElemsPerLane])
{
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        elems[i] = static_cast<float>(src[i]);
    }
}

template <typename scalar_t>
__device__ __forceinline__ void storeElems(scalar_t* __restrict__ dst,
                                           const float (&elems)[kElemsPerLane])
{
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        dst[i] = static_cast<scalar_t>(elems[i]);
    }
}

template <typename scalar_t>
__device__ __forceinline__ void roundElems(float (&elems)[kElemsPerLane])
{
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        elems[i] = static_cast<float>(static_cast<scalar_t>(elems[i]));
    }
}

template <typename scalar_t>
__device__ __forceinline__ void copyRawElems(const scalar_t* __restrict__ src,
                                             scalar_t* __restrict__ dst)
{
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(src);
}

template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__device__ __forceinline__ void storeCacheElems(cache_t* __restrict__ dst,
                                                const float (&elems)[kElemsPerLane],
                                                float scale)
{
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
        {
            dst[i] = static_cast<cache_t>(elems[i]);
        }
        else
        {
            const scalar_t rounded = static_cast<scalar_t>(elems[i]);
            dst[i] = opus::cast<cache_t>(static_cast<float>(rounded) / scale);
        }
    }
}

// index_q gather: qkv dtype, or unit-scale e4m3 (4787's q_idx contract; no scale tensor).
// When neither gather buffer is provided, write the packed qkv index-Q row (inplace).
template <typename scalar_t>
__device__ __forceinline__ void storeIndexQElems(scalar_t* __restrict__ index_q_out,
                                                 opus::fp8_t* __restrict__ index_q_fp8_out,
                                                 scalar_t* __restrict__ fallback_ptr,
                                                 const float (&elems)[kElemsPerLane],
                                                 int64_t token_idx,
                                                 int niq,
                                                 int iq_head,
                                                 int dim_base)
{
    const int64_t off = token_idx * static_cast<int64_t>(niq) * kHeadDim +
                        static_cast<int64_t>(iq_head) * kHeadDim + dim_base;
    if(index_q_fp8_out != nullptr)
    {
        storeCacheElems<scalar_t, opus::fp8_t, vllm::Fp8KVCacheDataType::kFp8E4M3>(
            index_q_fp8_out + off, elems, 1.0f);
    }
    else if(index_q_out != nullptr)
    {
        storeElems(index_q_out + off, elems);
    }
    else if(fallback_ptr != nullptr)
    {
        storeElems(fallback_ptr, elems);
    }
}

template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__device__ __forceinline__ void copyOrQuantizeCacheElems(const scalar_t* __restrict__ src,
                                                         cache_t* __restrict__ dst,
                                                         float scale)
{
    if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
    {
        copyRawElems(src, dst);
    }
    else
    {
        float elems[kElemsPerLane];
        loadElems(src, elems);
        storeCacheElems<scalar_t, cache_t, kv_dt>(dst, elems, scale);
    }
}

// Strided V store for the SHUFFLE (asm) layout: the kElemsPerLane elements owned
// by a lane are `stride` apart in the destination cache (V layout interleaves the
// intra-page position x-remainder between consecutive head-dim elements). Source
// is the raw (no norm/rope) qkv row, contiguous.
template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__device__ __forceinline__ void storeCacheElemsStrided(const scalar_t* __restrict__ src,
                                                       cache_t* __restrict__ dst,
                                                       int64_t stride,
                                                       float scale)
{
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
        {
            dst[i * stride] = static_cast<cache_t>(src[i]);
        }
        else
        {
            dst[i * stride] = opus::cast<cache_t>(static_cast<float>(src[i]) / scale);
        }
    }
}

// NVFP4 K/V cache: per (page, head) chunk of a page-16 cache, the codes as
// [head_dim / 16 blocks, 16 tokens, 8 B], then on the K side the E4M3 block
// scales as [16 tokens, 16 B] with byte (s % 4) * 4 + s / 4 holding K block s
// and that + 2 V block s. Compact pages put the V codes of a page after its
// K codes and scales instead of in a separate V page.
constexpr int kNvfp4KvPageSize = 16;
constexpr int kNvfp4KvCodeBytes = kHeadDim / 2 * kNvfp4KvPageSize;
constexpr int kNvfp4KvScaleBytes = kHeadDim / 16 * 2 * kNvfp4KvPageSize;
constexpr int64_t kNvfp4KvCompactPageBytes = 2 * kNvfp4KvCodeBytes + kNvfp4KvScaleBytes;

// E4M3 (OCP) bits of 0 <= x <= 448, rounded to nearest even.
__device__ __forceinline__ uint32_t e4m3RnePositive(float x)
{
    if(x < 0.015625f)
    {
        return static_cast<uint32_t>(__builtin_rintf(x * 512.0f));
    }
    const uint32_t bits = __float_as_uint(x);
    return ((bits + 0x7FFFFu + ((bits >> 20) & 1u)) >> 20) - (120u << 3);
}

__device__ __forceinline__ float e4m3ToFloat(uint32_t code)
{
    return code < 8 ? static_cast<float>(code) * (1.0f / 512.0f)
                    : __uint_as_float((code + (120u << 3)) << 20);
}

// One 128-wide K or V head of a token as NVFP4 into the page-16 cache, matching
// vLLM's minimax_m3_insert_kv_nvfp4: x / scale, block scale
// E4M3(max(min(E4M3(max(amax, 1e-12) / 6), 448), 1 / 512)), codes RNE onto the
// E2M1 grid. Word g of block s holds elements A = s * 16 + g * 8 + 0..3 (even
// lane) and B = the next 4 (odd lane); its byte j is |A_j| | sign(A_j) << 7 |
// |B_j| << 4 | sign(B_{j-1 mod 4}) << 3.
__device__ __forceinline__ void storeKvNvfp4(const float (&elems)[kElemsPerLane],
                                             int lane_id,
                                             int64_t slot,
                                             int head,
                                             uint8_t* __restrict__ cache,
                                             uint8_t* __restrict__ k_cache,
                                             int64_t s_page,
                                             int64_t s_head,
                                             bool has_scale,
                                             float scale,
                                             bool is_v)
{
    const int64_t page = slot / kNvfp4KvPageSize;
    const int tok = static_cast<int>(slot % kNvfp4KvPageSize);
    float a[kElemsPerLane];
    float amax = 0.0f;
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        a[i] = has_scale ? __fdiv_rn(elems[i], scale) : elems[i];
        amax = fmaxf(amax, fabsf(a[i]));
    }
    amax = fmaxf(amax, __shfl_xor(amax, 1, 32));
    amax = fmaxf(amax, __shfl_xor(amax, 2, 32));
    const float sf_raw = fminf(__fdiv_rn(fmaxf(amax, 1e-12f), 6.0f), 448.0f);
    const uint32_t sf8 = e4m3RnePositive(fmaxf(e4m3ToFloat(e4m3RnePositive(sf_raw)),
                                               1.0f / 512.0f));
    const float sf = e4m3ToFloat(sf8);

    // t * sf is exact and a / sf rounded never lands on a threshold t unless
    // a / sf == t, so comparing |a| to t * sf equals comparing RN(a / sf) to t.
    const bool is_b = lane_id & 1;
    uint32_t part = 0;
    uint32_t b_sign = 0;
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        const float y = fminf(fabsf(a[i]), 6.0f * sf);
        const uint32_t mag = (y > 0.25f * sf) + (y >= 0.75f * sf) + (y > 1.25f * sf) +
                             (y >= 1.75f * sf) + (y > 2.5f * sf) + (y >= 3.5f * sf) +
                             (y > 5.0f * sf);
        const uint32_t neg = __float_as_uint(a[i]) >> 31;
        part |= (is_b ? mag << 4 : mag | neg << 7) << (8 * i);
        b_sign |= (neg << 3) << (8 * i);
    }
    if(is_b)
    {
        part |= (b_sign << 8) | (b_sign >> 24);
    }
    part |= __shfl_xor(part, 1, 32);

    const int s = lane_id / 4;
    if(!is_b)
    {
        const int g = (lane_id / 2) % 2;
        uint8_t* chunk = cache + page * s_page + head * s_head;
        reinterpret_cast<uint32_t*>(chunk)[s * 2 * kNvfp4KvPageSize + tok * 2 + g] = part;
    }
    if(lane_id % 4 == 0)
    {
        uint8_t* k_chunk = k_cache + page * s_page + head * s_head;
        k_chunk[kNvfp4KvCodeBytes + tok * 16 + (s % 4) * 4 + s / 4 + (is_v ? 2 : 0)] =
            static_cast<uint8_t>(sf8);
    }
}

// One index key as AITER MSA NVFP4 into its page: E2M1 pairs at
// tok * head_dim / 2, then an E4M3 scale per 16 dims at
// block * head_dim / 2 + tok * head_dim / 16. Matches AITER's
// msa_index_cache_insert(kv_fmt=nvfp4): scale E4M3(clamp(amax / 6, 2^-9, 448)),
// each magnitude the E2M1 one nearest |x| / scale with ties to the smaller.
__device__ __forceinline__ void storeIndexNvfp4(const float (&elems)[kElemsPerLane],
                                                int lane_id,
                                                int64_t slot,
                                                uint8_t* __restrict__ cache,
                                                int64_t s_page,
                                                int block)
{
    const int64_t page = slot / block;
    const int tok = static_cast<int>(slot - page * block);
    float amax = 0.0f;
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        amax = fmaxf(amax, fabsf(elems[i]));
    }
    amax = fmaxf(amax, __shfl_xor(amax, 1, 32));
    amax = fmaxf(amax, __shfl_xor(amax, 2, 32));
    const uint32_t s8 =
        e4m3RnePositive(fminf(fmaxf(__fdiv_rn(amax, 6.0f), 1.0f / 512.0f), 448.0f));
    const float s = e4m3ToFloat(s8);
    uint32_t nib[kElemsPerLane];
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        const float y = fabsf(elems[i]);
        nib[i] = (y > 0.25f * s) + (y > 0.75f * s) + (y > 1.25f * s) + (y > 1.75f * s) +
                 (y > 2.5f * s) + (y > 3.5f * s) + (y > 5.0f * s);
        nib[i] |= static_cast<uint32_t>(elems[i] < 0.0f) << 3;
    }
    uint8_t* base = cache + page * s_page;
    uint8_t* codes = base + tok * (kHeadDim / 2) + lane_id * 2;
    codes[0] = static_cast<uint8_t>(nib[0] | nib[1] << 4);
    codes[1] = static_cast<uint8_t>(nib[2] | nib[3] << 4);
    if(lane_id % 4 == 0)
    {
        base[block * (kHeadDim / 2) + tok * (kHeadDim / 16) + lane_id / 4] =
            static_cast<uint8_t>(s8);
    }
}

template <typename scalar_t>
__device__ __forceinline__ void normAndRope(
    float (&elems)[kElemsPerLane],
    int lane_id,
    float eps,
    const scalar_t* __restrict__ weight,
    bool do_rope,
    int rotary_dim,
    const scalar_t* __restrict__ cos_ptr,
    bool apply_norm)
{
    if(apply_norm)
    {
        float sumsq = 0.0f;
#pragma unroll
        for(int i = 0; i < kElemsPerLane; ++i)
        {
            sumsq += elems[i] * elems[i];
        }
        sumsq = warpReduceSum(sumsq);
        const float rms_rcp = rsqrtf(sumsq / static_cast<float>(kHeadDim) + eps);
#pragma unroll
        for(int i = 0; i < kElemsPerLane; ++i)
        {
            const int dim = lane_id * kElemsPerLane + i;
            elems[i] = elems[i] * rms_rcp * (1.0f + static_cast<float>(weight[dim]));
        }
    }

    if(do_rope)
    {
        const int half = rotary_dim / 2;
        const int dim0 = lane_id * kElemsPerLane;
        const bool in_rope = dim0 < rotary_dim;

        if(in_rope)
        {
            const bool first_half = dim0 < half;
            const int i_base = first_half ? dim0 : dim0 - half;
            const int partner_lane = first_half ? (dim0 + half) / kElemsPerLane
                                                : (dim0 - half) / kElemsPerLane;
            const scalar_t* sin_ptr = cos_ptr + half;
#pragma unroll
            for(int i = 0; i < kElemsPerLane; ++i)
            {
                const float c = static_cast<float>(cos_ptr[i_base + i]);
                const float s = static_cast<float>(sin_ptr[i_base + i]);
                const float partner = __shfl(elems[i], partner_lane, 32);
                elems[i] = first_half ? elems[i] * c - partner * s
                                      : elems[i] * c + partner * s;
            }
        }
    }
}

template <typename scalar_t,
          typename cache_t,
          typename index_cache_t,
          vllm::Fp8KVCacheDataType kv_dt,
          vllm::Fp8KVCacheDataType index_dt,
          bool kPerTokenKVScale,
          bool kStaticKVScale,
          bool kIsSparse,
          bool kProcessIndex,
          bool kInsertKV,
          bool kAsmLayout,
          bool kKvNvfp4,
          bool kIdxNvfp4>
__global__ void fusedQKNormIdxrQKNormKernel(
    scalar_t* __restrict__ qkv,
    scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ index_q_out,
    opus::fp8_t* __restrict__ index_q_fp8_out,
    const scalar_t* __restrict__ q_norm_w,
    const scalar_t* __restrict__ k_norm_w,
    const scalar_t* __restrict__ iq_norm_w,
    const scalar_t* __restrict__ ik_norm_w,
    const scalar_t* __restrict__ cos_sin_cache,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ index_slot_mapping,
    cache_t* __restrict__ kv_cache_k,
    cache_t* __restrict__ kv_cache_v,
    index_cache_t* __restrict__ index_cache,
    // Per-token dynamic-quant OUTPUT dequant scales (fp8 per-token path only).
    // Null for the SGLang unit-scale fp8 contract. Layout mirrors
    // reshape_and_cache_with_pertoken_quant:
    //   asm_layout : [num_blocks, num_kv_heads, block_size]
    //     idx = block_idx*(nkv*block_size) + head*block_size + block_offset
    //   page-128   : [num_kv_heads, max_kv_tokens]
    //     idx = head*max_kv_tokens + slot
    float* __restrict__ k_scale,
    float* __restrict__ v_scale,
    float eps,
    int rotary_dim,
    int num_tokens,
    int nq,
    int nkv,
    int niq,
    int block_size,
    int x,
    int max_kv_tokens,
    // page-128 (asm_layout=false) strides for the separate K/V caches
    // (each [num_blocks, block_size, num_kv_heads, head_dim]); unused when kAsmLayout.
    // NVFP4 K/V: byte strides of a page-16 page (block) and of a head.
    int64_t k_s_block,
    int64_t k_s_token,
    int64_t k_s_head,
    int64_t v_s_block,
    int64_t v_s_token,
    int64_t v_s_head,
    // NVFP4 index cache: tokens per page and page stride in bytes.
    int idx_block,
    int64_t idx_s_page)
{
    const int warps_per_block = blockDim.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int global_warp_idx = blockIdx.x * warps_per_block + (threadIdx.x / 32);

    const int v_slots = kInsertKV ? nkv : 0;
    const int idx_slots = kProcessIndex ? niq + 1 : 0;
    const int slots_per_token = nq + nkv + v_slots + idx_slots;
    const int token_idx = global_warp_idx / slots_per_token;
    const int slot = global_warp_idx % slots_per_token;
    if(token_idx >= num_tokens)
    {
        return;
    }

    const int k_begin = nq;
    const int v_begin = nq + nkv;
    const int iq_begin = nq + nkv + v_slots;
    const int ik_slot = iq_begin + niq;
    const bool is_q = slot < k_begin;
    const bool is_k = slot >= k_begin && slot < v_begin;
    bool is_v = false;
    if constexpr(kInsertKV)
    {
        is_v = slot >= v_begin && slot < v_begin + nkv;
    }
    bool is_iq = false;
    bool is_ik = false;
    if constexpr(kProcessIndex)
    {
        is_iq = slot >= iq_begin && slot < ik_slot;
        is_ik = slot == ik_slot;
    }

    // NVFP4 K/V scale, read before the dependent row loads.
    bool has_kv_scale = false;
    float kv_scale_val = 1.0f;
    if constexpr(kKvNvfp4)
    {
        has_kv_scale = k_scale != nullptr;
        if(has_kv_scale && (is_k || is_v))
        {
            kv_scale_val = is_v ? *v_scale : *k_scale;
        }
    }

    const int dim_base = lane_id * kElemsPerLane;
    const int qkv_row = (nq + 2 * nkv + (kIsSparse ? (niq + 1) : 0)) * kHeadDim;

    scalar_t* row_ptr = nullptr;
    const scalar_t* norm_w = nullptr;
    bool do_rope = true;
    int head = 0;

    if(is_q)
    {
        row_ptr = qkv + static_cast<int64_t>(token_idx) * qkv_row + slot * kHeadDim;
        norm_w = q_norm_w;
    }
    else if(is_k)
    {
        head = slot - k_begin;
        row_ptr = qkv + static_cast<int64_t>(token_idx) * qkv_row + slot * kHeadDim;
        norm_w = k_norm_w;
    }
    else if(is_v)
    {
        head = slot - v_begin;
        row_ptr = qkv + static_cast<int64_t>(token_idx) * qkv_row + slot * kHeadDim;
        do_rope = false;
    }
    else if constexpr(kProcessIndex)
    {
        if(is_iq)
        {
            const int ih = slot - iq_begin;
            row_ptr =
                qkv + static_cast<int64_t>(token_idx) * qkv_row +
                (nq + 2 * nkv + ih) * kHeadDim;
            norm_w = iq_norm_w;
        }
        else
        {
            row_ptr =
                qkv + static_cast<int64_t>(token_idx) * qkv_row +
                (nq + 2 * nkv + niq) * kHeadDim;
            norm_w = ik_norm_w;
        }
    }

    scalar_t* store_ptr = row_ptr;
    if(is_q && q_out != nullptr)
    {
        store_ptr = q_out + static_cast<int64_t>(token_idx) * nq * kHeadDim + slot * kHeadDim;
    }

    int64_t mapped_slot = -1;
    if constexpr(kInsertKV)
    {
        if constexpr(kProcessIndex)
        {
            mapped_slot = is_ik ? index_slot_mapping[token_idx]
                                : ((is_k || is_v) ? slot_mapping[token_idx] : -1);
        }
        else
        {
            mapped_slot = (is_k || is_v) ? slot_mapping[token_idx] : -1;
        }
    }

    if constexpr(kInsertKV)
    {
        if(is_v)
        {
            if constexpr(kKvNvfp4)
            {
                // Stored with K below: one store site keeps the scale byte
                // store from waiting on the code store.
            }
            else if(mapped_slot >= 0)
            {
                // Per-token dynamic quant: amax over the lane's raw V elems, then a
                // full-warp max -> the (token, head) head-dim amax. scale = amax/FP8_MAX.
                float v_scale_val = 1.0f;
                if constexpr(kv_dt != vllm::Fp8KVCacheDataType::kAuto &&
                             kPerTokenKVScale)
                {
                    float local_amax = 0.0f;
#pragma unroll
                    for(int i = 0; i < kElemsPerLane; ++i)
                    {
                        local_amax = fmaxf(local_amax,
                                           fabsf(static_cast<float>(row_ptr[dim_base + i])));
                    }
                    const float amax = warpReduceMax(local_amax);
                    v_scale_val = (amax > 0.0f) ? amax / fp8Max<cache_t>() : 1.0f;
                    if(lane_id == 0)
                    {
                        const int64_t b = mapped_slot / block_size;
                        const int64_t t = mapped_slot % block_size;
                        const int64_t scale_idx =
                            kAsmLayout
                                ? (b * (static_cast<int64_t>(nkv) * block_size) +
                                   static_cast<int64_t>(head) * block_size + t)
                                : (static_cast<int64_t>(head) * max_kv_tokens + mapped_slot);
                        v_scale[scale_idx] = v_scale_val;
                    }
                }
                else if constexpr(kStaticKVScale)
                {
                    v_scale_val = *v_scale;
                }
                if constexpr(kAsmLayout)
                {
                    // V SHUFFLE: [num_blocks, num_kv_heads, block_size/x, head_dim, x]
                    //   off(d) = b*(nkv*hd*bs) + h*(hd*bs) + (t/x)*(hd*x) + d*x + (t%x)
                    // The kElemsPerLane elements of a lane (consecutive d) are x apart.
                    const int64_t b = mapped_slot / block_size;
                    const int64_t t = mapped_slot % block_size;
                    const int64_t v_base =
                        b * (static_cast<int64_t>(nkv) * kHeadDim * block_size) +
                        static_cast<int64_t>(head) * kHeadDim * block_size +
                        (t / x) * (static_cast<int64_t>(kHeadDim) * x) +
                        static_cast<int64_t>(dim_base) * x + (t % x);
                    storeCacheElemsStrided<scalar_t, cache_t, kv_dt>(
                        row_ptr + dim_base, kv_cache_v + v_base, x, v_scale_val);
                }
                else
                {
                    // page-128 V cache [num_blocks, block_size, num_kv_heads, head_dim]
                    const int64_t b = mapped_slot / block_size;
                    const int64_t t = mapped_slot % block_size;
                    const int64_t off =
                        b * v_s_block + t * v_s_token + head * v_s_head;
                    copyOrQuantizeCacheElems<scalar_t, cache_t, kv_dt>(
                        row_ptr + dim_base, kv_cache_v + off + dim_base, v_scale_val);
                }
            }
            if constexpr(!kKvNvfp4)
            {
                return;
            }
        }
    }

    float elems[kElemsPerLane];
    loadElems(row_ptr + dim_base, elems);
    const int64_t pos = positions[token_idx];
    const scalar_t* cos_ptr = cos_sin_cache + pos * rotary_dim;
    normAndRope(elems, lane_id, eps, norm_w, do_rope, rotary_dim, cos_ptr, norm_w != nullptr);

    if constexpr(kInsertKV)
    {
        if(is_q)
        {
            storeElems(store_ptr + dim_base, elems);
        }
        if constexpr(kProcessIndex)
        {
            if(is_iq)
            {
                storeIndexQElems(index_q_out,
                                 index_q_fp8_out,
                                 store_ptr + dim_base,
                                 elems,
                                 token_idx,
                                 niq,
                                 slot - iq_begin,
                                 dim_base);
            }
        }

        if(mapped_slot >= 0)
        {
            if constexpr(kProcessIndex)
            {
                if(is_ik)
                {
                    if constexpr(kIdxNvfp4)
                    {
                        roundElems<scalar_t>(elems);
                        storeIndexNvfp4(elems,
                                        lane_id,
                                        mapped_slot,
                                        reinterpret_cast<uint8_t*>(index_cache),
                                        idx_s_page,
                                        idx_block);
                    }
                    else
                    {
                        storeCacheElems<scalar_t, index_cache_t, index_dt>(
                            index_cache + mapped_slot * kHeadDim + dim_base, elems, 1.0f);
                    }
                }
            }
            if constexpr(kKvNvfp4)
            {
                if(is_k || is_v)
                {
                    roundElems<scalar_t>(elems);
                    storeKvNvfp4(elems,
                                 lane_id,
                                 mapped_slot,
                                 head,
                                 reinterpret_cast<uint8_t*>(is_v ? kv_cache_v : kv_cache_k),
                                 reinterpret_cast<uint8_t*>(kv_cache_k),
                                 k_s_block,
                                 k_s_head,
                                 has_kv_scale,
                                 kv_scale_val,
                                 is_v);
                }
            }
            else if(is_k)
            {
                // Per-token dynamic quant: amax over the lane's post-norm/rope K elems,
                // then a full-warp max -> the (token, head) head-dim amax.
                float k_scale_val = 1.0f;
                if constexpr(kv_dt != vllm::Fp8KVCacheDataType::kAuto &&
                             kPerTokenKVScale)
                {
                    float local_amax = 0.0f;
#pragma unroll
                    for(int i = 0; i < kElemsPerLane; ++i)
                    {
                        local_amax = fmaxf(local_amax, fabsf(elems[i]));
                    }
                    const float amax = warpReduceMax(local_amax);
                    k_scale_val = (amax > 0.0f) ? amax / fp8Max<cache_t>() : 1.0f;
                    if(lane_id == 0)
                    {
                        const int64_t b = mapped_slot / block_size;
                        const int64_t t = mapped_slot % block_size;
                        const int64_t scale_idx =
                            kAsmLayout
                                ? (b * (static_cast<int64_t>(nkv) * block_size) +
                                   static_cast<int64_t>(head) * block_size + t)
                                : (static_cast<int64_t>(head) * max_kv_tokens + mapped_slot);
                        k_scale[scale_idx] = k_scale_val;
                    }
                }
                else if constexpr(kStaticKVScale)
                {
                    k_scale_val = *k_scale;
                }
                if constexpr(kAsmLayout)
                {
                    // K SHUFFLE: [num_blocks, num_kv_heads, head_dim/x, block_size, x]
                    //   off(d) = b*(nkv*(hd/x)*bs*x) + h*((hd/x)*bs*x)
                    //            + (d/x)*(bs*x) + t*x + (d%x)
                    // kElemsPerLane <= x, so a lane's elements share d/x and (d%x) runs
                    // contiguously -> a contiguous run of kElemsPerLane in the cache.
                    const int64_t b = mapped_slot / block_size;
                    const int64_t t = mapped_slot % block_size;
                    const int d0 = dim_base;  // = lane_id * kElemsPerLane
                    const int64_t k_off =
                        b * (static_cast<int64_t>(nkv) * (kHeadDim / x) * block_size * x) +
                        static_cast<int64_t>(head) * ((kHeadDim / x) * block_size * x) +
                        static_cast<int64_t>(d0 / x) * (static_cast<int64_t>(block_size) * x) +
                        t * x + (d0 % x);
                    storeCacheElems<scalar_t, cache_t, kv_dt>(
                        kv_cache_k + k_off, elems, k_scale_val);
                }
                else
                {
                    // page-128 K cache [num_blocks, block_size, num_kv_heads, head_dim]
                    const int64_t b = mapped_slot / block_size;
                    const int64_t t = mapped_slot % block_size;
                    const int64_t off = b * k_s_block + t * k_s_token + head * k_s_head;
                    storeCacheElems<scalar_t, cache_t, kv_dt>(
                        kv_cache_k + off + dim_base, elems, k_scale_val);
                }
            }
        }
    }
    else
    {
        if constexpr(kProcessIndex)
        {
            if(is_iq)
            {
                storeIndexQElems(index_q_out,
                                 index_q_fp8_out,
                                 store_ptr + dim_base,
                                 elems,
                                 token_idx,
                                 niq,
                                 slot - iq_begin,
                                 dim_base);
                return;
            }
        }
        storeElems(store_ptr + dim_base, elems);
    }
}

template <typename scalar_t,
          typename cache_t,
          typename index_cache_t,
          vllm::Fp8KVCacheDataType kv_dt,
          vllm::Fp8KVCacheDataType index_dt,
          bool kKvNvfp4,
          bool kIdxNvfp4>
void launchFusedQKNormIdxrQKNorm(
    scalar_t* qkv,
    scalar_t* q_out,
    scalar_t* index_q_out,
    opus::fp8_t* index_q_fp8_out,
    const scalar_t* q_norm_w,
    const scalar_t* k_norm_w,
    const scalar_t* iq_norm_w,
    const scalar_t* ik_norm_w,
    const scalar_t* cos_sin_cache,
    const int64_t* positions,
    const int64_t* slot_mapping,
    const int64_t* index_slot_mapping,
    cache_t* kv_cache_k,
    cache_t* kv_cache_v,
    index_cache_t* index_cache,
    float* k_scale,
    float* v_scale,
    float eps,
    int rotary_dim,
    int num_tokens,
    int nq,
    int nkv,
    int niq,
    int block_size,
    int x,
    int max_kv_tokens,
    int64_t k_s_block,
    int64_t k_s_token,
    int64_t k_s_head,
    int64_t v_s_block,
    int64_t v_s_token,
    int64_t v_s_head,
    int idx_block,
    int64_t idx_s_page,
    bool has_index,
    bool process_index,
    bool insert_kv,
    bool per_token_kv_scale,
    bool static_kv_scale,
    bool asm_layout,
    hipStream_t stream)
{
    const int v_slots = insert_kv ? nkv : 0;
    const int idx_slots = process_index ? niq + 1 : 0;
    const int slots_per_token = nq + nkv + v_slots + idx_slots;
    constexpr int kBlockSize = 256;
    constexpr int kWarpsPerBlock = kBlockSize / 32;
    const int64_t total_warps = static_cast<int64_t>(num_tokens) * slots_per_token;
    const int grid = static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
    if(grid == 0)
    {
        return;
    }

#define LAUNCH(IS_SPARSE, PROCESS_INDEX, INSERT, ASM, PER_TOKEN, STATIC_SCALE)               \
    fusedQKNormIdxrQKNormKernel<scalar_t,                                                    \
                                cache_t,                                                     \
                                index_cache_t,                                               \
                                kv_dt,                                                       \
                                index_dt,                                                    \
                                PER_TOKEN,                                                   \
                                STATIC_SCALE,                                                \
                                IS_SPARSE,                                                   \
                                PROCESS_INDEX,                                               \
                                INSERT,                                                      \
                                ASM,                                                         \
                                kKvNvfp4,                                                    \
                                kIdxNvfp4>                                                   \
        <<<grid, kBlockSize, 0, stream>>>(qkv,                                               \
                                          q_out,                                             \
                                          index_q_out,                                       \
                                          index_q_fp8_out,                                   \
                                          q_norm_w,                                          \
                                          k_norm_w,                                          \
                                          iq_norm_w,                                         \
                                          ik_norm_w,                                         \
                                          cos_sin_cache,                                     \
                                          positions,                                         \
                                          slot_mapping,                                      \
                                          index_slot_mapping,                                \
                                          kv_cache_k,                                        \
                                          kv_cache_v,                                        \
                                          index_cache,                                       \
                                          k_scale,                                           \
                                          v_scale,                                           \
                                          eps,                                               \
                                          rotary_dim,                                        \
                                          num_tokens,                                        \
                                          nq,                                                \
                                          nkv,                                               \
                                          niq,                                               \
                                          block_size,                                        \
                                          x,                                                 \
                                          max_kv_tokens,                                     \
                                          k_s_block,                                         \
                                          k_s_token,                                         \
                                          k_s_head,                                          \
                                          v_s_block,                                         \
                                          v_s_token,                                         \
                                          v_s_head,                                          \
                                          idx_block,                                         \
                                          idx_s_page)

#define LAUNCH_SCALE(IS_SPARSE, PROCESS_INDEX, INSERT, ASM)     \
    do                                                          \
    {                                                           \
        if(per_token_kv_scale)                                  \
        {                                                       \
            LAUNCH(IS_SPARSE, PROCESS_INDEX, INSERT, ASM, true, false); \
        }                                                       \
        else if(static_kv_scale)                                 \
        {                                                       \
            LAUNCH(IS_SPARSE, PROCESS_INDEX, INSERT, ASM, false, true); \
        }                                                       \
        else                                                    \
        {                                                       \
            LAUNCH(IS_SPARSE, PROCESS_INDEX, INSERT, ASM, false, false); \
        }                                                       \
    } while(0)

#define LAUNCH_ASM(IS_SPARSE, PROCESS_INDEX, INSERT)             \
    do                                                          \
    {                                                           \
        if(asm_layout)                                          \
        {                                                       \
            LAUNCH_SCALE(IS_SPARSE, PROCESS_INDEX, INSERT, true); \
        }                                                       \
        else                                                    \
        {                                                       \
            LAUNCH_SCALE(IS_SPARSE, PROCESS_INDEX, INSERT, false); \
        }                                                       \
    } while(0)

    // NVFP4 caches are only written in insert mode, NVFP4 K/V at a fixed layout
    // with optional single-element scales (null when absent).
    if constexpr(kKvNvfp4)
    {
        if(kIdxNvfp4 || process_index)
        {
            LAUNCH(true, true, true, false, false, false);
        }
        else if constexpr(!kIdxNvfp4)
        {
            if(has_index)
            {
                LAUNCH(true, false, true, false, false, false);
            }
            else
            {
                LAUNCH(false, false, true, false, false, false);
            }
        }
    }
    else if constexpr(kIdxNvfp4)
    {
        LAUNCH_ASM(true, true, true);
    }
    else if(has_index)
    {
        if(insert_kv)
        {
            if(process_index)
            {
                LAUNCH_ASM(true, true, true);
            }
            else
            {
                LAUNCH_ASM(true, false, true);
            }
        }
        else
        {
            if(process_index)
            {
                LAUNCH_ASM(true, true, false);
            }
            else
            {
                LAUNCH_ASM(true, false, false);
            }
        }
    }
    else
    {
        if(insert_kv)
        {
            LAUNCH_ASM(false, false, true);
        }
        else
        {
            LAUNCH_ASM(false, false, false);
        }
    }
#undef LAUNCH_ASM
#undef LAUNCH_SCALE
#undef LAUNCH
}

// Cache formats for the launcher: element type for a given qkv type, quant mode and
// whether the cache holds NVFP4 bytes.
struct AutoCacheFmt
{
    template <typename T>
    using type = T;
    static constexpr vllm::Fp8KVCacheDataType kDt = vllm::Fp8KVCacheDataType::kAuto;
    static constexpr bool kNvfp4 = false;
};

struct Fp8CacheFmt
{
    template <typename T>
    using type = opus::fp8_t;
    static constexpr vllm::Fp8KVCacheDataType kDt = vllm::Fp8KVCacheDataType::kFp8E4M3;
    static constexpr bool kNvfp4 = false;
};

struct Nvfp4CacheFmt
{
    template <typename T>
    using type = uint8_t;
    static constexpr vllm::Fp8KVCacheDataType kDt = vllm::Fp8KVCacheDataType::kAuto;
    static constexpr bool kNvfp4 = true;
};

} // namespace fused_qknorm_idxrqknorm_ops

static void fused_qknorm_idxrqknorm_impl(
    aiter_tensor_t& qkv,
    const aiter_tensor_t& q_norm_weight,
    const aiter_tensor_t& k_norm_weight,
    const aiter_tensor_t& cos_sin_cache,
    const aiter_tensor_t& positions,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t rotary_dim,
    double eps,
    std::optional<aiter_tensor_t> index_q_norm_weight,
    std::optional<aiter_tensor_t> index_k_norm_weight,
    int64_t num_index_heads,
    std::optional<aiter_tensor_t> slot_mapping,
    std::optional<aiter_tensor_t> kv_cache_k,
    std::optional<aiter_tensor_t> kv_cache_v,
    std::optional<aiter_tensor_t> index_cache,
    int64_t block_size,
    std::optional<aiter_tensor_t> q_out,
    std::optional<aiter_tensor_t> index_q_out,
    std::optional<aiter_tensor_t> index_slot_mapping,
    const std::string& kv_cache_dtype,
    const std::string& index_cache_dtype,
    std::optional<aiter_tensor_t> k_scale,
    std::optional<aiter_tensor_t> v_scale,
    bool asm_layout,
    bool skip_index_branch)
{
    using namespace fused_qknorm_idxrqknorm_ops;

    AITER_CHECK(qkv.is_gpu() && qkv.is_contiguous(), "qkv must be contiguous CUDA");
    AITER_CHECK(positions.is_gpu() && positions.is_contiguous() &&
                    positions.dtype() == AITER_DTYPE_i64,
                "positions must be contiguous int64 CUDA");
    AITER_CHECK(cos_sin_cache.is_gpu() && cos_sin_cache.is_contiguous(),
                "cos_sin_cache must be contiguous CUDA");
    AITER_CHECK(cos_sin_cache.dtype() == qkv.dtype(),
                "cos_sin_cache dtype must match qkv");
    AITER_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == rotary_dim,
                "cos_sin_cache shape must be [max_pos, rotary_dim]");
    AITER_CHECK(q_norm_weight.is_gpu() && k_norm_weight.is_gpu() &&
                    q_norm_weight.is_contiguous() && k_norm_weight.is_contiguous(),
                "q/k norm weights must be contiguous CUDA tensors");
    AITER_CHECK(q_norm_weight.dtype() == qkv.dtype() &&
                    k_norm_weight.dtype() == qkv.dtype(),
                "q/k norm weight dtype must match qkv");
    AITER_CHECK(q_norm_weight.numel() == kHeadDim && k_norm_weight.numel() == kHeadDim,
                "q/k norm weight must have 128 elements");
    AITER_CHECK(rotary_dim > 0 && rotary_dim % 8 == 0 && rotary_dim <= kHeadDim,
                "rotary_dim must be a positive multiple of 8 and <= 128");

    const int num_tokens = static_cast<int>(qkv.size(0));
    const int nq = static_cast<int>(num_heads);
    const int nkv = static_cast<int>(num_kv_heads);
    const int niq = static_cast<int>(num_index_heads);
    AITER_CHECK(nq > 0 && nkv > 0 && niq >= 0, "num_heads/num_kv_heads must be positive");
    const bool has_index = niq > 0;
    AITER_CHECK(!skip_index_branch || has_index,
                "skip_index_branch requires num_index_heads > 0");
    const bool process_index = has_index && !skip_index_branch;
    // Both layouts use the separate kv_cache_k / kv_cache_v caches; asm_layout only
    // selects the cache element addressing (page-16 SHUFFLE vs plain page-128).
    AITER_CHECK(kv_cache_k.has_value() == kv_cache_v.has_value(),
                "kv_cache_k and kv_cache_v must be provided together");
    const bool insert_kv = kv_cache_k.has_value();
    const bool fp8_kv_cache =
        insert_kv && kv_cache_dtype.rfind("fp8", 0) == 0;
    const bool fp8_kv_per_token =
        fp8_kv_cache &&
        (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3");
    const bool fp8_kv_static =
        fp8_kv_cache && kv_cache_dtype == "fp8_e4m3_static";
    const bool fp8_index_cache =
        insert_kv && process_index && index_cache_dtype.rfind("fp8", 0) == 0;
    const bool nvfp4_kv_cache = insert_kv && kv_cache_dtype == "nvfp4";
    const bool nvfp4_index_cache =
        insert_kv && process_index && index_cache_dtype == "nvfp4";
    const int expected_row = (nq + 2 * nkv + (has_index ? niq + 1 : 0)) * kHeadDim;
    AITER_CHECK(positions.dim() == 1 && positions.size(0) >= num_tokens,
                "positions must be 1D with at least num_tokens elements");
    AITER_CHECK(qkv.dim() == 2 && qkv.size(1) == expected_row,
                "qkv must be [num_tokens, (num_heads + 2*num_kv_heads"
                " + optional index heads) * 128]");
    if(process_index)
    {
        AITER_CHECK(index_q_norm_weight.has_value() && index_k_norm_weight.has_value(),
                    "index branch requires both index norm weights");
        AITER_CHECK(index_q_norm_weight->is_gpu() && index_k_norm_weight->is_gpu() &&
                        index_q_norm_weight->is_contiguous() &&
                        index_k_norm_weight->is_contiguous(),
                    "index norm weights must be contiguous CUDA tensors");
        AITER_CHECK(index_q_norm_weight->dtype() == qkv.dtype() &&
                        index_k_norm_weight->dtype() == qkv.dtype(),
                    "index norm weights dtype must match qkv");
        AITER_CHECK(index_q_norm_weight->numel() == kHeadDim &&
                        index_k_norm_weight->numel() == kHeadDim,
                    "index norm weights must have 128 elements");
    }

    // page-128 K/V cache strides ([num_blocks, block_size, num_kv_heads, head_dim]);
    // unused in asm_layout (kernel computes SHUFFLE offsets directly).
    int64_t k_s_block = 0;
    int64_t k_s_token = 0;
    int64_t k_s_head = 0;
    int64_t v_s_block = 0;
    int64_t v_s_token = 0;
    int64_t v_s_head = 0;
    int x = 0;  // SHUFFLE interleave factor = 16 / cache_itemsize (asm_layout only)
    int max_kv_tokens = 0;  // page-128 per-token scale stride (scales: [nkv, max_kv_tokens])
    int idx_block = 0;      // NVFP4 index cache tokens per page
    int64_t idx_s_page = 0; // NVFP4 index cache page stride in bytes
    if(insert_kv)
    {
        AITER_CHECK(block_size > 0, "block_size must be positive in insert mode");
        AITER_CHECK(slot_mapping.has_value() && slot_mapping->is_gpu() &&
                        slot_mapping->is_contiguous() &&
                        slot_mapping->dtype() == AITER_DTYPE_i64,
                    "insert mode requires contiguous int64 CUDA slot_mapping");
        AITER_CHECK(slot_mapping->dim() == 1 && slot_mapping->size(0) >= num_tokens,
                    "slot_mapping must be 1D with at least num_tokens elements");
        if(process_index && !index_slot_mapping.has_value())
        {
            index_slot_mapping = slot_mapping;
        }
        if(process_index)
        {
            AITER_CHECK(index_slot_mapping.has_value() && index_slot_mapping->is_gpu() &&
                            index_slot_mapping->is_contiguous() &&
                            index_slot_mapping->dtype() == AITER_DTYPE_i64,
                        "sparse insert mode requires contiguous int64 CUDA index_slot_mapping");
            AITER_CHECK(index_slot_mapping->dim() == 1 &&
                            index_slot_mapping->size(0) >= num_tokens,
                        "index_slot_mapping must be 1D with at least num_tokens elements");
            AITER_CHECK(index_cache.has_value() && index_cache->is_gpu() &&
                            index_cache->is_contiguous(),
                        "sparse insert mode requires contiguous CUDA index_cache");
            AITER_CHECK(index_cache_dtype == "auto" || nvfp4_index_cache ||
                            index_cache_dtype.rfind("fp8", 0) == 0,
                        "sparse insert mode expects index_cache_dtype='auto', fp8 or "
                        "nvfp4");
            if(nvfp4_index_cache)
            {
                // AITER MSA NVFP4 index pages: [pages, >= page bytes] at 128 tokens a
                // page, or [pages, block, 128] with each NVFP4 page at its head.
                const int64_t page_bytes_per_token = kHeadDim / 2 + kHeadDim / 16;
                AITER_CHECK(index_cache->element_size() == 1,
                            "nvfp4 index_cache must be a byte tensor");
                if(index_cache->dim() == 2)
                {
                    idx_block = 128;
                    AITER_CHECK(index_cache->size(1) >= idx_block * page_bytes_per_token,
                                "nvfp4 index_cache pages are smaller than one nvfp4 page");
                }
                else
                {
                    AITER_CHECK(index_cache->dim() == 3 && index_cache->size(2) == kHeadDim,
                                "nvfp4 index_cache must be [num_pages, block_size, 128] "
                                "or [num_pages, page bytes]");
                    idx_block = static_cast<int>(index_cache->size(1));
                }
                idx_s_page = index_cache->stride(0);
            }
            else if(fp8_index_cache)
            {
                AITER_CHECK(index_cache->dtype() == AITER_DTYPE_fp8 ||
                                index_cache->dtype() == AITER_DTYPE_u8,
                            "fp8 sparse insert mode requires fp8 or uint8 index_cache");
            }
            else
            {
                AITER_CHECK(index_cache->dtype() == qkv.dtype(),
                            "non-fp8 sparse insert mode requires matching index_cache");
            }
        }

        // K/V cache dtype + scale checks (common to both layouts). kv_cache_k is the
        // dtype reference for both the SHUFFLE and page-128 caches.
        const aiter_tensor_t& kc = *kv_cache_k;
        AITER_CHECK(kc.is_gpu(), "kv cache must be CUDA");
        if(fp8_kv_cache)
        {
            AITER_CHECK(kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3" ||
                            kv_cache_dtype == "fp8_e4m3_unit" ||
                            kv_cache_dtype == "fp8_e4m3_static",
                        "fused_qknorm_idxrqknorm fp8 cache insert supports "
                        "fp8/fp8_e4m3 per-token, fp8_e4m3_unit, or "
                        "fp8_e4m3_static only");
            AITER_CHECK(kc.dtype() == AITER_DTYPE_fp8 || kc.dtype() == AITER_DTYPE_u8,
                        "fp8 kv_cache must use float8_e4m3 or uint8 storage");
            AITER_CHECK(k_scale.has_value() == v_scale.has_value(),
                        "k_scale and v_scale must be provided together");
            if(fp8_kv_per_token)
            {
                // k_scale/v_scale are per-token dynamic-quant OUTPUT tensors (fp32).
                // Their exact shape is validated in the layout branches below.
                AITER_CHECK(k_scale.has_value() && k_scale->is_gpu() && v_scale->is_gpu() &&
                                k_scale->is_contiguous() && v_scale->is_contiguous() &&
                                k_scale->dtype() == AITER_DTYPE_fp32 &&
                                v_scale->dtype() == AITER_DTYPE_fp32,
                            "fp8 per-token insert requires contiguous float32 CUDA "
                            "k_scale/v_scale output tensors");
            }
            else if(fp8_kv_static)
            {
                AITER_CHECK(k_scale.has_value() && v_scale.has_value() &&
                                k_scale->is_gpu() && v_scale->is_gpu() &&
                                k_scale->is_contiguous() && v_scale->is_contiguous() &&
                                k_scale->dtype() == AITER_DTYPE_fp32 &&
                                v_scale->dtype() == AITER_DTYPE_fp32 &&
                                k_scale->numel() == 1 && v_scale->numel() == 1,
                            "fp8_e4m3_static requires single-element contiguous "
                            "float32 CUDA k_scale/v_scale input tensors");
            }
            else
            {
                AITER_CHECK(!k_scale.has_value(),
                            "fp8_e4m3_unit does not accept k_scale/v_scale tensors");
            }
        }
        else if(nvfp4_kv_cache)
        {
            AITER_CHECK(kc.dtype() == AITER_DTYPE_fp8 || kc.dtype() == AITER_DTYPE_u8,
                        "nvfp4 kv_cache must use float8_e4m3 or uint8 storage");
            AITER_CHECK(k_scale.has_value() == v_scale.has_value(),
                        "k_scale and v_scale must be provided together");
            if(k_scale.has_value())
            {
                AITER_CHECK(k_scale->is_gpu() && v_scale->is_gpu() &&
                                k_scale->dtype() == AITER_DTYPE_fp32 &&
                                v_scale->dtype() == AITER_DTYPE_fp32 &&
                                k_scale->numel() == 1 && v_scale->numel() == 1,
                            "nvfp4 k_scale/v_scale must be single-element float32 CUDA "
                            "tensors");
            }
        }
        else
        {
            AITER_CHECK(kv_cache_dtype == "auto",
                        "non-fp8 fused_qknorm_idxrqknorm expects kv_cache_dtype='auto' "
                        "or 'nvfp4'");
            AITER_CHECK(kc.dtype() == qkv.dtype(),
                        "kv_cache dtype must match qkv for non-fp8 cache");
        }

        if(nvfp4_kv_cache)
        {
            // NVFP4 inside page-16 fp8 pages: [pages, num_kv_heads, ...] with K/V
            // views of equal page and head strides. A head chunk is 2048 B, or a
            // whole 2304 B compact page (K codes, scales, V codes) for one KV head.
            AITER_CHECK(block_size == kNvfp4KvPageSize, "nvfp4 kv cache needs block_size 16");
            AITER_CHECK(kv_cache_v->dtype() == kc.dtype(),
                        "nvfp4 kv_cache_k/kv_cache_v dtype must match");
            AITER_CHECK(kc.dim() >= 2 && kv_cache_v->dim() >= 2 && kc.size(1) == nkv &&
                            kv_cache_v->size(1) == nkv,
                        "nvfp4 kv caches must be [pages, num_kv_heads, ...]");
            AITER_CHECK(kc.stride(0) == kv_cache_v->stride(0) &&
                            kc.stride(1) == kv_cache_v->stride(1),
                        "nvfp4 kv_cache_k/kv_cache_v page and head strides must match");
            AITER_CHECK(kc.stride(1) == kNvfp4KvCodeBytes * 2 ||
                            (nkv == 1 && kc.stride(1) == kNvfp4KvCompactPageBytes),
                        "nvfp4 kv cache needs 2048 B head chunks or compact pages");
            AITER_CHECK(kc.stride(0) >= nkv * kc.stride(1),
                        "nvfp4 kv cache pages must hold all heads");
            if(process_index && !nvfp4_index_cache)
            {
                AITER_CHECK(index_cache->numel() % kHeadDim == 0,
                            "index_cache must contain contiguous 128-element rows");
            }
            k_s_block = kc.stride(0);
            k_s_head = kc.stride(1);
            v_s_block = kv_cache_v->stride(0);
            v_s_head = kv_cache_v->stride(1);
        }
        else if(asm_layout)
        {
            // SHUFFLE (asm_layout) caches, matching reshape_and_cache(asm_layout=true):
            //   K [num_blocks, num_kv_heads, head_dim/x, block_size, x]
            //   V [num_blocks, num_kv_heads, block_size/x, head_dim, x]
            // with x = 16 / cache_itemsize and block_size the (physical) ASM page.
            const int itemsize = static_cast<int>(kv_cache_k->element_size());
            AITER_CHECK(itemsize > 0 && (16 % itemsize) == 0,
                        "asm_layout kv cache element size must divide 16");
            x = 16 / itemsize;
            AITER_CHECK(kHeadDim % x == 0 && block_size % x == 0,
                        "asm_layout requires x | head_dim and x | block_size");
            AITER_CHECK(kv_cache_k->is_gpu() && kv_cache_k->is_contiguous() &&
                            kv_cache_v->is_gpu() && kv_cache_v->is_contiguous(),
                        "asm_layout kv_cache_k/kv_cache_v must be contiguous CUDA");
            AITER_CHECK(kv_cache_k->dtype() == kv_cache_v->dtype(),
                        "asm_layout kv_cache_k/kv_cache_v dtype must match");
            AITER_CHECK(kv_cache_k->dim() == 5 && kv_cache_k->size(1) == nkv &&
                            kv_cache_k->size(2) == kHeadDim / x &&
                            kv_cache_k->size(3) == block_size && kv_cache_k->size(4) == x,
                        "kv_cache_k must be [num_blocks, num_kv_heads, head_dim/x, "
                        "block_size, x]");
            // The page spans are equal for separately allocated K/V caches. Packed
            // LBHNC storage instead exposes K from the beginning of each packed
            // block and V from an offset base pointer, so their contiguous tensor
            // spans intentionally have different size(0). Rebased slot mappings
            // still address the same logical page domain in both views.
            AITER_CHECK(kv_cache_v->dim() == 5 && kv_cache_v->size(1) == nkv &&
                            kv_cache_v->size(2) == block_size / x &&
                            kv_cache_v->size(3) == kHeadDim && kv_cache_v->size(4) == x,
                        "kv_cache_v must be [page_span, num_kv_heads, block_size/x, "
                        "head_dim, x]");
            if(process_index && !nvfp4_index_cache)
            {
                // Main-cache size(0) is a physical storage span in packed LBHNC
                // mode and includes gaps belonging to the opposite K/V side. It
                // therefore cannot define the independently mapped index-cache
                // capacity. As with slot_mapping, callers own bounds validity.
                AITER_CHECK(index_cache->numel() % kHeadDim == 0,
                            "index_cache must contain contiguous 128-element rows");
            }
            // strides unused in asm_layout (kernel computes SHUFFLE offsets directly).
            if(fp8_kv_per_token)
            {
                // Per-token dequant scales: [num_blocks, num_kv_heads, block_size]
                // (mirrors reshape_and_cache_with_pertoken_quant asm_layout).
                // For packed views, the shorter span is the common page-id
                // domain reachable from both offset base pointers.
                const int64_t scale_page_span =
                    kv_cache_k->size(0) < kv_cache_v->size(0)
                        ? kv_cache_k->size(0)
                        : kv_cache_v->size(0);
                AITER_CHECK(k_scale->numel() >= scale_page_span * nkv * block_size &&
                                v_scale->numel() >= scale_page_span * nkv * block_size,
                            "asm_layout fp8 per-token k_scale/v_scale must have at least "
                            "common_page_span * num_kv_heads * block_size elements");
            }
        }
        else
        {
            // page-128 caches as separate K and V tensors
            //   [num_blocks, block_size, num_kv_heads, head_dim] (head_dim contiguous).
            // These are typically the key/value slices of a fused
            // [num_blocks, 2, block_size, num_kv_heads, head_dim] cache.
            AITER_CHECK(kv_cache_k->is_gpu() && kv_cache_v->is_gpu(),
                        "kv_cache_k/kv_cache_v must be CUDA");
            AITER_CHECK(kv_cache_k->dtype() == kv_cache_v->dtype(),
                        "kv_cache_k/kv_cache_v dtype must match");
            AITER_CHECK(kv_cache_k->dim() == 4 && kv_cache_k->size(1) == block_size &&
                            kv_cache_k->size(2) == nkv && kv_cache_k->size(3) == kHeadDim &&
                            kv_cache_k->stride(3) == 1,
                        "page-128 kv_cache_k must be [num_blocks, block_size, "
                        "num_kv_heads, 128] with contiguous head_dim");
            AITER_CHECK(kv_cache_v->dim() == 4 && kv_cache_v->size(0) == kv_cache_k->size(0) &&
                            kv_cache_v->size(1) == block_size && kv_cache_v->size(2) == nkv &&
                            kv_cache_v->size(3) == kHeadDim && kv_cache_v->stride(3) == 1,
                        "page-128 kv_cache_v must be [num_blocks, block_size, "
                        "num_kv_heads, 128] with contiguous head_dim");
            if(process_index && !nvfp4_index_cache)
            {
                AITER_CHECK(index_cache->numel() >= kv_cache_k->size(0) * block_size * kHeadDim,
                            "index_cache must contain at least num_blocks * block_size * 128 elements");
            }
            k_s_block = kv_cache_k->stride(0);
            k_s_token = kv_cache_k->stride(1);
            k_s_head = kv_cache_k->stride(2);
            v_s_block = kv_cache_v->stride(0);
            v_s_token = kv_cache_v->stride(1);
            v_s_head = kv_cache_v->stride(2);
            if(fp8_kv_per_token)
            {
                // Per-token dequant scales: [num_kv_heads, max_kv_tokens] with
                // idx = head * max_kv_tokens + slot (mirrors
                // reshape_and_cache_with_pertoken_quant page-128). max_kv_tokens is the
                // scale tensor's last-dim size (caller-sized to cover all slots).
                AITER_CHECK(k_scale->dim() == 2 && v_scale->dim() == 2 &&
                                k_scale->size(0) == nkv && v_scale->size(0) == nkv &&
                                k_scale->size(1) == v_scale->size(1),
                            "page-128 fp8 per-token k_scale/v_scale must be "
                            "[num_kv_heads, max_kv_tokens]");
                max_kv_tokens = static_cast<int>(k_scale->size(1));
                const int64_t num_blocks = kv_cache_k->size(0);
                AITER_CHECK(max_kv_tokens >= num_blocks * block_size,
                            "page-128 fp8 per-token scale max_kv_tokens must be >= "
                            "num_blocks * block_size");
            }
        }
    }

    if(q_out.has_value())
    {
        AITER_CHECK(q_out->is_gpu() && q_out->is_contiguous() &&
                        q_out->dtype() == qkv.dtype(),
                    "q_out must be contiguous CUDA and match qkv dtype");
        AITER_CHECK(q_out->numel() == static_cast<int64_t>(num_tokens) * nq * kHeadDim,
                    "q_out must have num_tokens * num_heads * 128 elements");
    }
    if(index_q_out.has_value())
    {
        AITER_CHECK(has_index, "index_q_out requires num_index_heads > 0");
        const bool index_q_is_fp8 = index_q_out->dtype() == AITER_DTYPE_fp8 ||
                                    index_q_out->dtype() == AITER_DTYPE_u8;
        AITER_CHECK(index_q_out->is_gpu() && index_q_out->is_contiguous() &&
                        (index_q_out->dtype() == qkv.dtype() || index_q_is_fp8),
                    "index_q_out must be contiguous CUDA and match qkv dtype or fp8 e4m3");
        AITER_CHECK(index_q_out->numel() == static_cast<int64_t>(num_tokens) * niq * kHeadDim,
                    "index_q_out must have num_tokens * num_index_heads * 128 elements");
    }

    HipDeviceGuard device_guard(qkv.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(qkv.dtype(), "fused_qknorm_idxrqknorm", [&] {
        using T = scalar_t;
        const bool index_q_is_fp8 =
            index_q_out.has_value() && (index_q_out->dtype() == AITER_DTYPE_fp8 ||
                                        index_q_out->dtype() == AITER_DTYPE_u8);
        T* index_q_ptr = (!index_q_is_fp8 && index_q_out.has_value())
                             ? reinterpret_cast<T*>(index_q_out->data_ptr())
                             : nullptr;
        opus::fp8_t* index_q_fp8_ptr =
            index_q_is_fp8 ? reinterpret_cast<opus::fp8_t*>(index_q_out->data_ptr())
                           : nullptr;
        const bool has_kv_scale = fp8_kv_per_token || fp8_kv_static ||
                                  (nvfp4_kv_cache && k_scale.has_value());
        auto launch = [&](auto kv_fmt, auto index_fmt) {
            using KvFmt = decltype(kv_fmt);
            using IndexFmt = decltype(index_fmt);
            using cache_t = typename KvFmt::template type<T>;
            using index_cache_t = typename IndexFmt::template type<T>;
            launchFusedQKNormIdxrQKNorm<T,
                                        cache_t,
                                        index_cache_t,
                                        KvFmt::kDt,
                                        IndexFmt::kDt,
                                        KvFmt::kNvfp4,
                                        IndexFmt::kNvfp4>(
                reinterpret_cast<T*>(qkv.data_ptr()),
                q_out.has_value() ? reinterpret_cast<T*>(q_out->data_ptr()) : nullptr,
                index_q_ptr,
                index_q_fp8_ptr,
                reinterpret_cast<const T*>(q_norm_weight.data_ptr()),
                reinterpret_cast<const T*>(k_norm_weight.data_ptr()),
                process_index ? reinterpret_cast<const T*>(index_q_norm_weight->data_ptr())
                              : nullptr,
                process_index ? reinterpret_cast<const T*>(index_k_norm_weight->data_ptr())
                              : nullptr,
                reinterpret_cast<const T*>(cos_sin_cache.data_ptr()),
                reinterpret_cast<int64_t*>(positions.data_ptr()),
                insert_kv ? reinterpret_cast<int64_t*>(slot_mapping->data_ptr()) : nullptr,
                (insert_kv && process_index)
                    ? reinterpret_cast<int64_t*>(index_slot_mapping->data_ptr())
                    : nullptr,
                insert_kv ? reinterpret_cast<cache_t*>(kv_cache_k->data_ptr()) : nullptr,
                insert_kv ? reinterpret_cast<cache_t*>(kv_cache_v->data_ptr()) : nullptr,
                (insert_kv && process_index)
                    ? reinterpret_cast<index_cache_t*>(index_cache->data_ptr())
                    : nullptr,
                has_kv_scale ? reinterpret_cast<float*>(k_scale->data_ptr()) : nullptr,
                has_kv_scale ? reinterpret_cast<float*>(v_scale->data_ptr()) : nullptr,
                static_cast<float>(eps),
                static_cast<int>(rotary_dim),
                num_tokens,
                nq,
                nkv,
                niq,
                static_cast<int>(block_size),
                x,
                max_kv_tokens,
                k_s_block,
                k_s_token,
                k_s_head,
                v_s_block,
                v_s_token,
                v_s_head,
                idx_block,
                idx_s_page,
                has_index,
                process_index,
                insert_kv,
                fp8_kv_per_token,
                fp8_kv_static,
                asm_layout,
                stream);
        };
        auto launch_index = [&](auto kv_fmt) {
            if(nvfp4_index_cache)
            {
                launch(kv_fmt, Nvfp4CacheFmt{});
            }
            else if(fp8_index_cache)
            {
                launch(kv_fmt, Fp8CacheFmt{});
            }
            else
            {
                launch(kv_fmt, AutoCacheFmt{});
            }
        };
        if(nvfp4_kv_cache)
        {
            launch_index(Nvfp4CacheFmt{});
        }
        else if(fp8_kv_cache)
        {
            launch_index(Fp8CacheFmt{});
        }
        else
        {
            launch_index(AutoCacheFmt{});
        }
    });
}

void fused_qknorm_idxrqknorm(
    aiter_tensor_t& qkv,
    const aiter_tensor_t& q_norm_weight,
    const aiter_tensor_t& k_norm_weight,
    const aiter_tensor_t& cos_sin_cache,
    const aiter_tensor_t& positions,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t rotary_dim,
    double eps,
    std::optional<aiter_tensor_t> index_q_norm_weight,
    std::optional<aiter_tensor_t> index_k_norm_weight,
    int64_t num_index_heads,
    std::optional<aiter_tensor_t> slot_mapping,
    std::optional<aiter_tensor_t> kv_cache_k,
    std::optional<aiter_tensor_t> kv_cache_v,
    std::optional<aiter_tensor_t> index_cache,
    int64_t block_size,
    std::optional<aiter_tensor_t> q_out,
    std::optional<aiter_tensor_t> index_q_out,
    std::optional<aiter_tensor_t> index_slot_mapping,
    const std::string& kv_cache_dtype,
    const std::string& index_cache_dtype,
    std::optional<aiter_tensor_t> k_scale,
    std::optional<aiter_tensor_t> v_scale,
    bool asm_layout,
    bool skip_index_branch)
{
    fused_qknorm_idxrqknorm_impl(qkv,
                                                q_norm_weight,
                                                k_norm_weight,
                                                cos_sin_cache,
                                                positions,
                                                num_heads,
                                                num_kv_heads,
                                                rotary_dim,
                                                eps,
                                                index_q_norm_weight,
                                                index_k_norm_weight,
                                                num_index_heads,
                                                slot_mapping,
                                                kv_cache_k,
                                                kv_cache_v,
                                                index_cache,
                                                block_size,
                                                q_out,
                                                index_q_out,
                                                index_slot_mapping,
                                                kv_cache_dtype,
                                                index_cache_dtype,
                                                k_scale,
                                                v_scale,
                                                asm_layout,
                                                skip_index_branch);
}

} // namespace aiter
