# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness of the experimental MSA index-key knobs.

- msa_index_cache_insert writes the nvfp4 page byte for byte as the torch
  quantizer below does, and the fp8 cache as a plain cast.
- With AITER_MSA_INDEX_KV_FMT=nvfp4 both scoring passes match a reference
  that dequantizes the page the way the kernels do: E4M3(scale * magnitude).
- AITER_MSA_PREFILL_CFG / AITER_MSA_PREFILL_CHUNK_BLOCKS change only the
  launch, so every setting reproduces the default scores bit for bit.
"""

import argparse
import os
import sys

import torch

import aiter
from aiter.ops.msa_block_select import (
    ENV_KV_FMT,
    ENV_PREFILL_CFG,
    ENV_PREFILL_CHUNK_BLOCKS,
    KV_FMT_FP8,
    KV_FMT_NVFP4,
    SCORE_PREFILL_CFGS,
    pa_sparse_block_score_decode,
    pa_sparse_block_score_prefill,
)
from aiter.ops.triton.msa_index_cache import msa_index_cache_insert
from aiter.test_common import checkAllclose

BLOCK_SIZE = 128
HEAD_DIM = 128
WAVE = 64
FP8 = torch.float8_e4m3fn
DEV = "cuda"
NEG = -1e4
MAGS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
MIDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
DATA_BYTES = BLOCK_SIZE * HEAD_DIM // 2


class _env:
    """Set environment variables for the duration of a with-block."""

    def __init__(self, **kv):
        self.kv = kv

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kv}
        for k, v in self.kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def ref_nvfp4_token(x):
    """[N, HEAD_DIM] -> (e2m1 pair bytes [N, HEAD_DIM/2], scale bytes [N, HEAD_DIM/16])."""
    g = x.float().view(x.shape[0], HEAD_DIM // 16, 16)
    amax = g.abs().amax(-1, keepdim=True)
    s8 = (amax / 6.0).clamp(min=2.0**-9, max=448.0).to(FP8)
    s = s8.float()
    mids = torch.tensor(MIDS, device=x.device)
    code = (g.abs()[..., None] > mids * s[..., None]).sum(-1).to(torch.uint8)
    nib = (code | ((g < 0).to(torch.uint8) << 3)).view(x.shape[0], HEAD_DIM)
    data = nib[:, 0::2] | (nib[:, 1::2] << 4)
    return data, s8.view(torch.uint8).view(x.shape[0], -1)


def ref_nvfp4_dequant(cache):
    """Byte pages [P, >= page bytes] -> [P, BLOCK_SIZE, HEAD_DIM] fp32, as the
    kernels see them: E4M3(min(scale * magnitude, 448)) with the element's sign."""
    p = cache.shape[0]
    data = cache[:, :DATA_BYTES].reshape(p, BLOCK_SIZE, HEAD_DIM // 2)
    nib = torch.stack([data & 15, data >> 4], -1).reshape(p, BLOCK_SIZE, HEAD_DIM)
    sb = cache[:, DATA_BYTES : DATA_BYTES + BLOCK_SIZE * HEAD_DIM // 16]
    s = sb.view(FP8).float().reshape(p, BLOCK_SIZE, HEAD_DIM // 16, 1)
    mags = torch.tensor(MAGS, device=cache.device)
    m = mags[(nib & 7).long()].view(p, BLOCK_SIZE, HEAD_DIM // 16, 16)
    v = (s * m).clamp(max=448.0).to(FP8).float()
    v = torch.where((nib >= 8).view_as(v), -v, v)
    return v.view(p, BLOCK_SIZE, HEAD_DIM)


def ref_block_scores(q, k, block_table, seq_lens, qlens, num_slots):
    """score[h, n, b] = max over the causal tokens of block b of q[n,h,:] . k[b,t,:]."""
    total_q, heads, _ = q.shape
    ref = torch.full((heads, total_q, num_slots), -float("inf"), device=DEV)
    qf = q.float()
    row = 0
    for r, qlen in enumerate(qlens):
        seq_len = int(seq_lens[r])
        nblk = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        pages = block_table[r, :nblk].long()
        prod = torch.einsum("btd,qhd->bqht", k[pages].float(), qf[row : row + qlen])
        kv_len = seq_len - qlen + torch.arange(qlen, device=DEV) + 1
        tok = (
            torch.arange(nblk, device=DEV)[:, None] * BLOCK_SIZE
            + torch.arange(BLOCK_SIZE, device=DEV)[None, :]
        )
        visible = tok[:, None, :] < kv_len[None, :, None]
        prod = prod.masked_fill(~visible[:, :, None, :], -float("inf"))
        ref[:, row : row + qlen, :nblk] = prod.max(dim=-1).values.permute(2, 1, 0)
        row += qlen
    return ref


def _finite(t):
    return torch.nan_to_num(t, neginf=NEG)


def _setup(seq_lens, qlens, num_idx_heads, kv_fmt):
    """q, the index cache in kv_fmt filled through msa_index_cache_insert, the
    dequantized keys the reference scores against, and the launch tensors."""
    num_reqs = len(seq_lens)
    max_blk = max((s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in seq_lens)
    num_pages = num_reqs * max_blk + 2
    torch.manual_seed(0)
    q = (torch.randn(sum(qlens), num_idx_heads, HEAD_DIM, device=DEV) / 4).to(FP8)
    keys = torch.randn(num_pages * BLOCK_SIZE, HEAD_DIM, device=DEV).to(torch.bfloat16)
    # Some heavy outliers, so groups land on very different scales.
    keys[::37] *= 64
    # The cache tensor is [P, BLOCK_SIZE, HEAD_DIM] fp8 either way; nvfp4 uses
    # the head of every page.
    cache = torch.zeros(num_pages, BLOCK_SIZE, HEAD_DIM, device=DEV, dtype=FP8)
    slots = torch.randperm(num_pages * BLOCK_SIZE, device=DEV)
    msa_index_cache_insert(keys[slots], cache, slots, kv_fmt=kv_fmt)
    if kv_fmt == KV_FMT_NVFP4:
        kref = ref_nvfp4_dequant(cache.view(torch.uint8).view(num_pages, -1))
    else:
        kref = cache.float()
    block_table = torch.arange(num_reqs * max_blk, device=DEV, dtype=torch.int32).view(
        num_reqs, max_blk
    )
    seq = torch.tensor(seq_lens, device=DEV, dtype=torch.int32)
    slots_w = 1 << (-(-max_blk // WAVE) - 1).bit_length()
    score = torch.full(
        (num_idx_heads, sum(qlens), slots_w * WAVE), -float("inf"), device=DEV
    )
    return q, cache, kref, block_table, seq, score, max_blk


def test_insert(num_tokens: int):
    torch.manual_seed(1)
    num_pages = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 3
    x = torch.randn(num_tokens, HEAD_DIM, device=DEV).to(torch.bfloat16)
    x[::5] *= 100
    x[1::7] /= 1000
    x[2] = 0
    slots = torch.randperm(num_pages * BLOCK_SIZE, device=DEV)[:num_tokens]
    slots[::11] = -1

    cache = torch.full((num_pages, BLOCK_SIZE, HEAD_DIM), 0x5A, dtype=torch.uint8)
    cache = cache.to(DEV).view(FP8)
    msa_index_cache_insert(x, cache, slots, kv_fmt=KV_FMT_NVFP4)
    raw = cache.view(torch.uint8).view(num_pages, -1)
    data, sb = ref_nvfp4_token(x)
    live = slots >= 0
    page, tok = slots[live] // BLOCK_SIZE, slots[live] % BLOCK_SIZE
    cols = torch.arange(HEAD_DIM // 2, device=DEV)
    got_d = raw[page[:, None], tok[:, None] * (HEAD_DIM // 2) + cols]
    scols = torch.arange(HEAD_DIM // 16, device=DEV)
    got_s = raw[page[:, None], DATA_BYTES + tok[:, None] * (HEAD_DIM // 16) + scols]
    bad_d = (got_d != data[live]).sum().item()
    bad_s = (got_s != sb[live]).sum().item()
    # Skipped slots and the bytes past the nvfp4 page are left alone.
    expect = torch.full_like(raw, 0x5A)
    expect[page[:, None], tok[:, None] * (HEAD_DIM // 2) + cols] = data[live]
    expect[page[:, None], DATA_BYTES + tok[:, None] * (HEAD_DIM // 16) + scols] = sb[
        live
    ]
    bad_other = (raw != expect).sum().item()

    cache8 = torch.zeros(num_pages, BLOCK_SIZE, HEAD_DIM, device=DEV, dtype=FP8)
    msa_index_cache_insert(x, cache8, slots, kv_fmt=KV_FMT_FP8)
    want8 = torch.zeros(num_pages * BLOCK_SIZE, HEAD_DIM, device=DEV, dtype=torch.uint8)
    want8[slots[live].long()] = x[live].to(FP8).view(torch.uint8)
    bad8 = (cache8.view(torch.uint8).view(-1, HEAD_DIM) != want8).sum().item()

    ok = bad_d == 0 and bad_s == 0 and bad_other == 0 and bad8 == 0
    aiter.logger.info(
        f"insert N={num_tokens}: nvfp4 data mismatches {bad_d}, scale {bad_s}, "
        f"stray writes {bad_other}; fp8 mismatches {bad8} -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def _check(name, ref, score, max_blk):
    err = checkAllclose(
        _finite(ref[:, :, :max_blk]),
        _finite(score[:, :, :max_blk]),
        msg=name,
        rtol=1e-2,
        atol=1e-2,
    )
    return err == 0


def test_score(num_idx_heads, batch, ctx, query_len, kv_fmt):
    fmt = "nvfp4" if kv_fmt == KV_FMT_NVFP4 else "fp8"
    seq_lens = [ctx - (i % 4) * BLOCK_SIZE for i in range(batch)]
    ok = True

    qlens = [query_len] * batch
    q, cache, kref, bt, seq, score, max_blk = _setup(
        seq_lens, qlens, num_idx_heads, kv_fmt
    )
    with _env(**{ENV_KV_FMT: fmt}):
        pa_sparse_block_score_decode(
            q, cache, score, bt, seq, query_len=query_len, max_seq_len=max(seq_lens)
        )
    ref = ref_block_scores(q, kref, bt, seq, qlens, score.size(2))
    info = f"H:{num_idx_heads} batch:{batch} ctx:{ctx}"
    ok &= _check(f"[{fmt}] decode {info} qlen:{query_len}", ref, score, max_blk)

    qlens = [1 + (i * 37) % 300 for i in range(batch - 1)] + [300]
    q, cache, kref, bt, seq, score, max_blk = _setup(
        seq_lens, qlens, num_idx_heads, kv_fmt
    )
    cu = torch.tensor([0] + qlens, device=DEV).cumsum(0).to(torch.int32)
    ref = ref_block_scores(q, kref, bt, seq, qlens, score.size(2))
    base = None
    cfgs = [None] + [
        f"{qt},{w},{e}"
        for (w, e) in SCORE_PREFILL_CFGS
        for qt in (1, 2, 4)
        if not (e == 6 and qt > 2)
    ]
    for cfg in cfgs:
        for chunk in (None, "1", "8"):
            score.fill_(-float("inf"))
            env = {ENV_KV_FMT: fmt, ENV_PREFILL_CFG: cfg}
            env[ENV_PREFILL_CHUNK_BLOCKS] = chunk
            with _env(**env):
                pa_sparse_block_score_prefill(
                    q,
                    cache,
                    score,
                    bt,
                    cu,
                    seq,
                    max_query_len=max(qlens),
                    max_seq_len=max(seq_lens),
                )
            if base is None:
                base = score.clone()
                ok &= _check(f"[{fmt}] prefill {info}", ref, score, max_blk)
            elif not torch.equal(base, score):
                diff = (_finite(base) - _finite(score)).abs().max().item()
                aiter.logger.info(
                    f"[{fmt}] prefill cfg={cfg} chunk={chunk} differs from the "
                    f"default by up to {diff} -> FAIL"
                )
                ok = False
    aiter.logger.info(
        f"[{fmt}] prefill {len(cfgs) * 3 - 1} cfg/chunk overrides vs default: "
        f"{'bit-exact' if ok else 'see above'}"
    )
    return ok


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("-H", "--num_idx_heads", type=int, nargs="*", default=[1, 2])
parser.add_argument("-b", "--batch", type=int, default=4)
parser.add_argument("-c", "--ctx", type=int, default=4096)
parser.add_argument("-q", "--query_len", type=int, default=4)
args = parser.parse_args()

current_gfx = aiter.get_gfx()
if current_gfx != "gfx950":
    print(f"Skipping {__file__}: requires gfx950, got {current_gfx}")
    sys.exit(0)

results = [test_insert(1000)]
for h in args.num_idx_heads:
    for f in (KV_FMT_FP8, KV_FMT_NVFP4):
        results.append(test_score(h, args.batch, args.ctx, args.query_len, f))
print("ALL PASS" if all(results) else "FAILURES")
sys.exit(0 if all(results) else 1)
