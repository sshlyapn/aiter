# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _msa_index_cache_insert_fp8_kernel(
    k_ptr,  # [num_tokens, head_dim]
    cache_ptr,  # [num_pages, block_size, head_dim]
    slot_ptr,  # [num_tokens]
    stride_k_t,
    stride_c_p,
    stride_c_t,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = tl.load(slot_ptr + pid)
    if slot < 0:
        return
    page = slot // BLOCK_SIZE
    tok = slot - page * BLOCK_SIZE
    offs = tl.arange(0, HEAD_DIM)
    x = tl.load(k_ptr + pid * stride_k_t + offs)
    dst = cache_ptr + page.to(tl.int64) * stride_c_p + tok * stride_c_t + offs
    tl.store(dst, x.to(cache_ptr.dtype.element_ty))


@triton.jit
def _msa_index_cache_insert_nvfp4_kernel(
    k_ptr,  # [num_tokens, head_dim]
    cache_ptr,  # uint8, num_pages pages of page_stride bytes
    slot_ptr,  # [num_tokens]
    stride_k_t,
    page_stride,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """One token into its page: e2m1 pairs at ``tok * HEAD_DIM / 2``, then one
    e4m3 scale per 16 dims at ``BLOCK_SIZE * HEAD_DIM / 2 + tok * HEAD_DIM / 16``.

    scale = E4M3(clamp(amax / 6, 2^-9, 448)); each element takes the e2m1
    magnitude nearest |x| / scale, ties to the smaller one. No global scale:
    the scoring passes feed E4M3(scale * magnitude) straight to an fp8 MFMA.
    """
    GROUPS: tl.constexpr = HEAD_DIM // 16
    pid = tl.program_id(0)
    slot = tl.load(slot_ptr + pid)
    if slot < 0:
        return
    page = slot // BLOCK_SIZE
    tok = slot - page * BLOCK_SIZE

    offs = tl.arange(0, HEAD_DIM)
    x = tl.load(k_ptr + pid * stride_k_t + offs).to(tl.float32)
    g = tl.reshape(x, [GROUPS, 16])
    amax = tl.max(tl.abs(g), axis=1)
    s8 = tl.clamp(tl.div_rn(amax, 6.0), 2.0**-9, 448.0).to(
        tl.float8e4nv, fp_downcast_rounding="rtne"
    )
    # Compared against mid * scale, which is exact for an e4m3 scale, rather
    # than dividing, so the codes do not depend on the division's rounding.
    y = tl.abs(g)
    s = s8.to(tl.float32)[:, None]
    code = (
        (y > 0.25 * s).to(tl.uint8)
        + (y > 0.75 * s).to(tl.uint8)
        + (y > 1.25 * s).to(tl.uint8)
        + (y > 1.75 * s).to(tl.uint8)
        + (y > 2.5 * s).to(tl.uint8)
        + (y > 3.5 * s).to(tl.uint8)
        + (y > 5.0 * s).to(tl.uint8)
    )
    nib = code | ((g < 0).to(tl.uint8) << 3)
    lo, hi = tl.split(tl.reshape(nib, [HEAD_DIM // 2, 2]))
    packed = lo | (hi << 4)

    base = cache_ptr + page.to(tl.int64) * page_stride
    tl.store(base + tok * (HEAD_DIM // 2) + tl.arange(0, HEAD_DIM // 2), packed)
    sbase = base + BLOCK_SIZE * (HEAD_DIM // 2) + tok * GROUPS
    tl.store(sbase + tl.arange(0, GROUPS), s8.to(tl.uint8, bitcast=True))
