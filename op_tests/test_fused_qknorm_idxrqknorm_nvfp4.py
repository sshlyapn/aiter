# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Byte-exact checks of fused_qknorm_idxrqknorm's NVFP4 cache outputs.

References: q / index_q and any non-NVFP4 cache come from the same op in its
existing bf16 / fp8 modes; K / V NVFP4 pages from a CPU port of vLLM's
MiniMax-M3 minimax_m3_insert_kv_nvfp4; NVFP4 index pages from
msa_index_cache_insert. K and index_k inputs are the op's own in-place
(model dtype) outputs.
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.ops.msa_block_select import KV_FMT_NVFP4, NVFP4_BLOCK_SIZE
from aiter.ops.triton.msa_index_cache import msa_index_cache_insert

HEAD_DIM = 128
ROT = 64
EPS = 1e-6
KV_BLOCK = 128  # main KV block of the fp8 pages, as in vLLM
NUM_KV_BLOCKS = 40
IDX_BLOCK = 64
IDX_PAGES = 96
CODE_BYTES = 1024
COMPACT_PAGE = 2304
COMPACT_V_OFFSET = 1280
FP8 = dtypes.fp8


def make_slots(n: int, capacity: int, gen: torch.Generator) -> torch.Tensor:
    s = torch.randperm(capacity, generator=gen)[:n]
    s[torch.rand(n, generator=gen) < 0.2] = -1
    return s.cuda()


def make_kv_cache(kv_fmt: str, nkv: int, dtype: torch.dtype):
    """Byte buffer plus K / V views and page-16 slots for a KV format."""
    if kv_fmt == "nvfp4_compact":
        pages = NUM_KV_BLOCKS * KV_BLOCK // 16
        buf = torch.randint(0, 256, (pages * COMPACT_PAGE,), dtype=torch.uint8)
        buf = buf.cuda()
        shape, stride = (pages, 1, 8, 16, 8), (COMPACT_PAGE, COMPACT_PAGE, 128, 8, 1)
        k = buf.view(FP8).as_strided(shape, stride, 0)
        v = buf.view(FP8).as_strided(shape, stride, COMPACT_V_OFFSET)
        return buf, k, v, False
    # Model layout: blocks of KV_BLOCK K tokens then KV_BLOCK V tokens.
    itemsize = 2 if kv_fmt == "auto" else 1
    x = 16 // itemsize
    page_elems = nkv * HEAD_DIM * 16
    pages = NUM_KV_BLOCKS * 2 * KV_BLOCK // 16
    buf = torch.randint(0, 256, (pages * page_elems * itemsize,), dtype=torch.uint8)
    buf = buf.cuda()
    cache = buf.view(dtype if itemsize == 2 else FP8)
    k = cache.view(pages, nkv, HEAD_DIM // x, 16, x)
    v = cache.view(pages, nkv, 16 // x, HEAD_DIM, x)[KV_BLOCK // 16 :]
    return buf, k, v, True


def make_index_cache(index_fmt: str, dtype: torch.dtype):
    """Index cache and its token capacity."""
    if index_fmt == "nvfp4_2d":
        page = NVFP4_BLOCK_SIZE * (HEAD_DIM // 2 + HEAD_DIM // 16) + 64
        c = torch.randint(0, 256, (IDX_PAGES // 2, page), dtype=torch.uint8)
        return c.cuda().view(FP8), IDX_PAGES // 2 * NVFP4_BLOCK_SIZE
    c = torch.randint(0, 256, (IDX_PAGES, IDX_BLOCK, HEAD_DIM), dtype=torch.uint8)
    c = c.cuda()
    if index_fmt == "auto":
        c = torch.randn(IDX_PAGES, IDX_BLOCK, HEAD_DIM, device="cuda")
        return c.to(dtype), IDX_PAGES * IDX_BLOCK
    return c.view(FP8), IDX_PAGES * IDX_BLOCK


def nvfp4_kv_ref(
    buf: torch.Tensor,
    x: torch.Tensor,
    slots: torch.Tensor,
    view: torch.Tensor,
    k_view: torch.Tensor,
    scale: torch.Tensor | None,
    is_v: bool,
) -> None:
    """CPU port of vLLM's _insert_kv_nvfp4_kernel into the byte buffer."""
    keep = slots >= 0
    x = x[keep].float()
    slots = slots[keep]
    n, nkv, _ = x.shape
    a = x / scale.cpu() if scale is not None else x
    blk = a.view(n, nkv, 8, 16)
    amax = blk.abs().amax(-1)
    sf = (amax.clamp(min=1e-12) / 6.0).clamp(max=448.0).to(FP8).float()
    sf = sf.clamp(min=1.0 / 512).to(FP8)
    m = (blk / sf.float()[..., None]).abs().clamp(max=6.0)
    mag = sum(
        (m > t).to(torch.int32) if strict else (m >= t).to(torch.int32)
        for t, strict in (
            (0.25, True),
            (0.75, False),
            (1.25, True),
            (1.75, False),
            (2.5, True),
            (3.5, False),
            (5.0, True),
        )
    )
    neg = torch.signbit(blk).to(torch.int32)
    mag = mag.view(n, nkv, 8, 2, 2, 4)
    neg = neg.view(n, nkv, 8, 2, 2, 4)
    code = (
        mag[..., 0, :]
        | (neg[..., 0, :] << 7)
        | (mag[..., 1, :] << 4)
        | (torch.roll(neg[..., 1, :], 1, dims=-1) << 3)
    ).to(torch.uint8)

    page = (slots // 16).view(n, 1, 1, 1, 1)
    tok = (slots % 16).view(n, 1, 1, 1, 1)
    h = torch.arange(nkv).view(1, nkv, 1, 1, 1)
    s = torch.arange(8).view(1, 1, 8, 1, 1)
    g = torch.arange(2).view(1, 1, 1, 2, 1)
    j = torch.arange(4).view(1, 1, 1, 1, 4)
    chunk = page * view.stride(0) + h * view.stride(1)
    off = view.storage_offset() + chunk + s * 128 + tok * 8 + g * 4 + j
    buf[off.reshape(-1)] = code.reshape(-1)
    s = s.view(1, 1, 8)
    k_chunk = k_view.storage_offset() + chunk.view(n, nkv, 1)
    sf_off = k_chunk + CODE_BYTES + tok.view(n, 1, 1) * 16 + (s % 4) * 4 + s // 4
    buf[(sf_off + (2 if is_v else 0)).reshape(-1)] = sf.view(torch.uint8).reshape(-1)


def call_op(case, qkv, kv, index, q_out, iq_out, kv_dtype, index_dtype, scales):
    k_cache, v_cache = kv
    index_cache, index_slots = index
    skip = index_dtype is None
    nh, nkv, niq = case["heads"]
    aiter.fused_qknorm_idxrqknorm(
        qkv,
        case["q_w"],
        case["k_w"],
        case["cos_sin"],
        case["positions"],
        nh,
        nkv,
        ROT,
        EPS,
        case["iq_w"],
        case["ik_w"],
        niq,
        case["slots"],
        k_cache,
        v_cache,
        None if skip else index_cache,
        16,
        q_out,
        None if skip else iq_out,
        None if skip else index_slots,
        kv_cache_dtype=kv_dtype,
        index_cache_dtype="auto" if skip else index_dtype,
        k_scale=scales[0],
        v_scale=scales[1],
        asm_layout=True,
        skip_index_branch=skip,
    )


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int16)


def check_equal(name, actual, expected, info):
    mism = int((bits(actual) != bits(expected)).sum())
    assert mism == 0, f"{info}: {mism} differing {name} elements"


@torch.inference_mode()
def run_case(
    dtype: torch.dtype,
    num_tokens: int,
    heads: tuple[int, int, int],
    kv_fmt: str,
    index_fmt: str,
    scales: str,
    seed: int,
):
    nh, nkv, niq = heads
    info = (
        f"dtype={dtype} T={num_tokens} heads={heads} kv={kv_fmt} "
        f"index={index_fmt} scales={scales}"
    )
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    row = (nh + 2 * nkv + niq + 1) * HEAD_DIM
    qkv = (torch.randn(num_tokens, row, device="cuda") * 3).to(dtype)
    qkv[::5] *= 300
    qkv[1::7] *= 1e-4
    weights = [(torch.randn(HEAD_DIM, device="cuda") * 0.5).to(dtype) for _ in range(4)]
    case = {
        "heads": heads,
        "q_w": weights[0],
        "k_w": weights[1],
        "iq_w": weights[2],
        "ik_w": weights[3],
        "cos_sin": torch.randn(8192, ROT, device="cuda").to(dtype),
        "positions": torch.randint(0, 8192, (num_tokens,), device="cuda"),
    }
    kv_nvfp4 = kv_fmt.startswith("nvfp4")
    skip = index_fmt == "skip"
    index_nvfp4 = index_fmt.startswith("nvfp4")

    k_scale = v_scale = None
    if scales == "custom" or kv_fmt == "fp8":
        k_scale = torch.tensor([0.37 if scales == "custom" else 1.0], device="cuda")
        v_scale = torch.tensor([1.9 if scales == "custom" else 1.0], device="cuda")

    buf, k_cache, v_cache, rebase = make_kv_cache(kv_fmt, nkv, dtype)
    slots = make_slots(num_tokens, NUM_KV_BLOCKS * KV_BLOCK, gen)
    if rebase:
        slots = torch.where(slots >= 0, slots + slots // KV_BLOCK * KV_BLOCK, slots)
    case["slots"] = slots
    if skip:
        index_cache, index_slots = None, None
    else:
        index_cache, capacity = make_index_cache(index_fmt, dtype)
        index_slots = make_slots(num_tokens, capacity, gen)
    iq_dtype = dtype if index_fmt == "auto" else FP8

    # Old modes: bf16 K/V scratch for NVFP4 K/V, fp8 index scratch for NVFP4
    # index keys.
    if kv_nvfp4:
        old_buf, old_k, old_v, _ = make_kv_cache("auto", nkv, dtype)
        old_kv_dtype, old_scales = "auto", (None, None)
    else:
        old_buf = buf.clone()
        old_k, old_v = (
            old_buf.view(t.dtype).as_strided(t.shape, t.stride(), t.storage_offset())
            for t in (k_cache, v_cache)
        )
        old_kv_dtype = "fp8_e4m3_static" if kv_fmt == "fp8" else "auto"
        old_scales = (k_scale, v_scale)
    if skip:
        old_index, old_index_dtype = None, None
    elif index_nvfp4:
        old_index = torch.zeros(capacity, HEAD_DIM, dtype=FP8, device="cuda")
        old_index_dtype = "fp8"
    else:
        old_index = index_cache.clone()
        old_index_dtype = "fp8" if index_fmt == "fp8" else "auto"
    q_old = torch.empty(num_tokens, nh * HEAD_DIM, dtype=dtype, device="cuda")
    iq_old = torch.empty(num_tokens, niq * HEAD_DIM, dtype=iq_dtype, device="cuda")
    call_op(
        case,
        qkv.clone(),
        (old_k, old_v),
        (old_index, index_slots),
        q_old,
        iq_old,
        old_kv_dtype,
        old_index_dtype,
        old_scales,
    )

    qkv_ref = qkv.clone()
    aiter.fused_qknorm_idxrqknorm(
        qkv_ref,
        case["q_w"],
        case["k_w"],
        case["cos_sin"],
        case["positions"],
        nh,
        nkv,
        ROT,
        EPS,
        case["iq_w"],
        case["ik_w"],
        niq,
    )
    k_ref = qkv_ref[:, nh * HEAD_DIM : (nh + nkv) * HEAD_DIM]
    v_ref = qkv_ref[:, (nh + nkv) * HEAD_DIM : (nh + 2 * nkv) * HEAD_DIM]
    ik_ref = qkv_ref[:, (nh + 2 * nkv + niq) * HEAD_DIM :]
    check_equal("q (in-place vs q_out)", qkv_ref[:, : nh * HEAD_DIM], q_old, info)

    ref_buf = buf.cpu() if kv_nvfp4 else old_buf.cpu()
    if kv_nvfp4:
        slots_cpu = slots.cpu()
        for x, view, scale, is_v in (
            (k_ref, k_cache, k_scale, False),
            (v_ref, v_cache, v_scale, True),
        ):
            nvfp4_kv_ref(
                ref_buf,
                x.cpu().view(num_tokens, nkv, HEAD_DIM),
                slots_cpu,
                view,
                k_cache,
                scale,
                is_v,
            )
    ref_index = None
    if index_nvfp4:
        ref_index = index_cache.clone()
        msa_index_cache_insert(ik_ref, ref_index, index_slots, KV_FMT_NVFP4)
    elif not skip:
        ref_index = old_index

    q_out = torch.empty_like(q_old)
    iq_out = torch.empty_like(iq_old)
    qkv_in = qkv.clone()
    call_op(
        case,
        qkv,
        (k_cache, v_cache),
        (index_cache, index_slots),
        q_out,
        iq_out,
        "nvfp4" if kv_nvfp4 else old_kv_dtype,
        None if skip else ("nvfp4" if index_nvfp4 else old_index_dtype),
        (k_scale, v_scale),
    )
    torch.cuda.synchronize()
    check_equal("qkv", qkv, qkv_in, info)
    check_equal("q_out", q_out, q_old, info)
    check_equal("kv cache bytes", buf.cpu(), ref_buf, info)
    if not skip:
        check_equal("index_q_out", iq_out, iq_old, info)
        check_equal("index cache", index_cache, ref_index, info)
    return {
        "dtype": str(dtype),
        "T": num_tokens,
        "heads": heads,
        "kv": kv_fmt,
        "index": index_fmt,
        "scales": scales,
        "status": "passed",
    }


KV_FMTS = ["nvfp4", "nvfp4_compact", "fp8", "auto"]
INDEX_FMTS = ["nvfp4_3d", "nvfp4_2d", "fp8", "auto", "skip"]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test fused_qknorm_idxrqknorm NVFP4 cache outputs",
)
parser.add_argument(
    "-d", "--dtype", type=str, choices=["bf16", "fp16"], nargs="*", default=None
)
parser.add_argument(
    "--num_tokens", type=int, nargs="*", default=[1, 7, 32, 129, 1031, 4099]
)
parser.add_argument("--kv", type=str, choices=KV_FMTS, nargs="*", default=KV_FMTS)
parser.add_argument(
    "--index", type=str, choices=INDEX_FMTS, nargs="*", default=INDEX_FMTS
)
args = parser.parse_args()

# MiniMax-M3 TP4 heads, plus a multi-KV-head layout (no compact pages there).
l_heads = [(16, 1, 1), (8, 2, 2)]
l_dtype = ["bf16", "fp16"] if args.dtype is None else args.dtype

df = []
for i, (dtype_name, num_tokens, heads, kv_fmt, index_fmt, scales) in enumerate(
    itertools.product(
        l_dtype, args.num_tokens, l_heads, args.kv, args.index, ["none", "custom"]
    )
):
    if not (kv_fmt.startswith("nvfp4") or index_fmt.startswith("nvfp4")):
        continue
    if kv_fmt == "nvfp4_compact" and heads[1] != 1:
        continue
    if scales == "custom" and kv_fmt == "auto":
        continue
    if dtype_name == "fp16" and (heads != l_heads[0] or num_tokens > 129):
        continue
    df.append(
        run_case(
            dtypes.d_dtypes[dtype_name],
            num_tokens,
            heads,
            kv_fmt,
            index_fmt,
            scales,
            seed=1000 + i,
        )
    )

df = pd.DataFrame(df)
aiter.logger.info(
    "fused_qknorm_idxrqknorm NVFP4: %d cases passed (byte-exact)\n%s",
    len(df),
    df.groupby(["kv", "index"]).size().to_markdown(),
)
