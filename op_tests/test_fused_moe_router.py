# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf for the fused MoE routing preamble.

``fused_moe_router_impl`` replaces the 4-kernel decode preamble in one launch:

    biased_grouped_topk -> moe_sorting -> fused_dynamic_mx_quant_moe_sort  (MXFP4)
    biased_grouped_topk -> moe_sorting -> per_token_quant_hip              (FP8)

The reference is that stock sequence, not a hand-written torch model, so a
mismatch means the fusion diverged from the path it is meant to replace.

topk ids and sorted rows are compared as multisets: any permutation of a given
expert's rows is a valid sort, and the fused kernel's ballot rank need not
match moe_sorting's atomic cursor. fp4 bytes and e8m0 scales must be exact, as
must fp8 bytes and the per-token fp32 scales.

This test can be run two ways:

1. pytest (correctness only):
   pytest op_tests/test_fused_moe_router.py -v

2. command line (correctness + perf summary table):
   python op_tests/test_fused_moe_router.py -m 1,64,128 -ek 320,8
"""

import argparse
import dataclasses
import functools
import itertools
import os
import sys

import pandas as pd
import pytest
import torch

import aiter
from aiter import QuantType, dtypes, get_hip_quant
from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.fused_moe_router import (
    fused_moe_router_impl,
    get_fused_moe_router_workspace,
)
from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort
from aiter.ops.topk import biased_grouped_topk
from aiter.test_common import benchmark, run_perftest
from aiter.utility.dtypes import str2tuple

torch.set_default_device("cuda")

# The kernel is gfx950 only; cols must land on a phase-1 quant geometry the
# entry covers (fmr_pick_geom in csrc/kernels/fused_moe_router.cu).
HIDDEN_DIMS = (2048, 4096)
GROUP_SIZE = 32
# Activation quant modes, i.e. the kernel's (quant_type, out_q dtype) pairs.
MXFP4, FP8 = "mxfp4", "fp8"
MODES = (MXFP4, FP8)
# Every output buffer of one call, in the order the entry takes them.
KEYS = ("ti", "tw", "sids", "sw", "seids", "nv", "a1", "a1s")
# Workspace sizing hint. Larger -m still works, it just allocates a
# second, bigger buffer.
WORKSPACE_MAX_TOKENS = 128
SUPPORTED = get_gfx() == "gfx950"

# Weights are renormalized by a sum reduced in a different order than the stock
# path, so they agree to fp32 rounding rather than bit-exactly.
W_TOL = 2e-6


def _skip_msg():
    return f"fused_moe_router requires gfx950, got {get_gfx()}"


def _quant_type(mode):
    if mode == FP8:
        return QuantType.per_Token.value
    elif mode == MXFP4:
        return QuantType.per_1x32.value
    else:
        raise ValueError(f"unknown quant mode {mode!r}")


def _expert_slots(E, n_shared, ep):
    """Expert-id slots the routing map spans; mirrors the C++ expert_slots."""
    return E + n_shared + (1 if (ep and n_shared) else 0)


def _make_mask(E, ep_rank, ep_size, vllm_shape=False, n_shared=0):
    """Linear expert shard, matching vLLM's determine_expert_map.

    vllm_shape reproduces the real buffer, which is E+1 long with an
    always-masked sentinel in the trailing slot (expert_map_manager.py).
    With fused shared experts it is E+n_shared+1: the shared weights are
    replicated so every rank owns those slots, and the sentinel that
    non-owning ranks park their shared row on stays masked.
    """
    n = E + n_shared + 1 if (vllm_shape or n_shared) else E
    m = torch.zeros(n, dtype=torch.int32)
    per = E // ep_size
    m[ep_rank * per : (ep_rank + 1) * per] = 1
    m[E : E + n_shared] = 1
    return m


def _append_shared(ti, tw, M, E, n_shared, shared_w, ep_rank, ep_size):
    """Reference for the kernel's shared tail.

    The owner of a token emits ``E+s`` for each shared expert; every other
    rank emits the masked sentinel, so summing the ranks' outputs yields
    exactly one copy. ``ep_size == 1`` makes every rank the owner, i.e. the
    non-EP case, where the sentinel is never used.
    """
    owner = (torch.arange(M, device=ti.device) % ep_size) == ep_rank
    ids = torch.where(
        owner.view(-1, 1),
        (E + torch.arange(n_shared, device=ti.device)).view(1, -1).expand(M, -1),
        torch.full((M, n_shared), E + n_shared, device=ti.device),
    ).to(torch.int32)
    w = torch.full((M, n_shared), shared_w, dtype=dtypes.fp32, device=tw.device)
    return torch.cat([ti, ids], dim=1), torch.cat([tw, w], dim=1)


def _deswizzle(osc, nrows, scale_n):
    """Invert mx_scale_shuffle_idx so scale rows can be compared by token."""
    scalen_pad = ((scale_n + 7) // 8) * 8
    x = torch.arange(nrows).view(-1, 1)
    y = torch.arange(scale_n).view(1, -1)
    idx = (
        (x // 32 * scalen_pad) * 32
        + (y // 8) * 256
        + (y % 4) * 64
        + (x % 16) * 4
        + (y % 8) // 4 * 2
        + (x % 32) // 16
    )
    return osc.reshape(-1).view(torch.uint8)[idx.view(-1)].view(nrows, -1)


def _inputs(M, E, bias_dtype, seed, cols=4096):
    torch.manual_seed(seed)
    g = torch.randn(M, E, dtype=dtypes.bf16)
    h = torch.randn(M, cols, dtype=dtypes.bf16)
    # Draw the bias in bf16 and widen, so an fp32 bias holds bf16-representable
    # values. The stock wrapper only takes it in the gating dtype, so anything
    # needing the extra mantissa would make the two paths disagree on ties for
    # a reason that is not a kernel bug. What is under test here is that the
    # fp32 dispatch reads the wider layout correctly.
    b = torch.randn(E, dtype=dtypes.bf16).to(bias_dtype)
    return g, b, h


def _run_stock(
    g,
    b,
    h,
    M,
    E,
    topk,
    unit_size,
    need_renorm,
    rsf,
    mask,
    n_shared=0,
    shared_w=1.0,
    ep_rank=0,
    ep_size=1,
    mode=MXFP4,
):
    tw = torch.empty(M, topk, dtype=dtypes.fp32)
    ti = torch.empty(M, topk, dtype=torch.int32)
    # The stock wrapper coerces the bias to the gating dtype; the fused kernel
    # reads it as-is, so feed the reference the same values it would see.
    biased_grouped_topk(
        g,
        b.to(g.dtype),
        tw,
        ti,
        num_expert_group=1,
        topk_group=1,
        need_renorm=need_renorm,
        routed_scaling_factor=rsf,
    )
    if n_shared:
        # Appended after the renorm the stock path already did, matching where
        # the kernel writes them: the renorm must not see the shared weights.
        ti, tw = _append_shared(ti, tw, M, E, n_shared, shared_w, ep_rank, ep_size)
    topk_total = topk + n_shared
    E_tot = _expert_slots(E, n_shared, mask is not None)
    sids, sw, seids, nv, moe_buf = moe_sorting(
        ti, tw, E_tot, h.shape[1], dtypes.bf16, block_size=unit_size, expert_mask=mask
    )
    if mode == FP8:
        a1, a1s = get_hip_quant(QuantType.per_Token)(h, quant_dtype=dtypes.fp8)
    elif mode == MXFP4:
        a1, a1s = fused_dynamic_mx_quant_moe_sort(
            h,
            sids,
            nv,
            token_num=M,
            topk=topk_total,
            block_size=unit_size,
            quant_dtype=dtypes.fp4x2,
            sorted_weights=sw,
        )
    else:
        raise ValueError(f"unknown quant mode {mode!r}")
    return {
        "ti": ti,
        "tw": tw,
        "sids": sids,
        "sw": sw,
        "seids": seids,
        "nv": nv,
        "a1": a1,
        "a1s": a1s,
        "moe_buf": moe_buf,
    }


def _alloc(ref, M, topk, mode=MXFP4):
    """Output buffers, poisoned so an unwritten slot cannot pass by luck."""
    if mode == FP8:
        # 0x7f7f7f7f is a scale no bf16 row can produce (above bf16 max / 448).
        a1 = _fill_pat(torch.empty_like(ref["a1"]), 0)
        a1s = _fill_pat(torch.empty_like(ref["a1s"]), 0)
    elif mode == MXFP4:
        a1 = torch.zeros(ref["a1"].shape, dtype=torch.uint8).view(ref["a1"].dtype)
        a1s = torch.zeros_like(ref["a1s"])
    else:
        raise ValueError(f"unknown quant mode {mode!r}")
    return {
        "ti": torch.full((M, topk), -1, dtype=torch.int32),
        "tw": torch.zeros(M, topk, dtype=dtypes.fp32),
        "sids": torch.full_like(ref["sids"], -1),
        "sw": torch.zeros_like(ref["sw"]),
        "seids": torch.full_like(ref["seids"], -1),
        "nv": torch.zeros_like(ref["nv"]),
        "a1": a1,
        "a1s": a1s,
    }


def _call_fused(
    g,
    b,
    h,
    a,
    E,
    topk,
    unit_size,
    need_renorm,
    rsf,
    mask,
    moe_buf,
    n_shared=0,
    shared_w=1.0,
    ep_rank=0,
    ep_size=1,
    mode=MXFP4,
):
    fused_moe_router_impl(
        g,
        b,
        h,
        a["ti"],
        a["tw"],
        a["sids"],
        a["sw"],
        a["seids"],
        a["nv"],
        a["a1"],
        a["a1s"],
        E,
        topk,
        unit_size,
        GROUP_SIZE,
        need_renorm,
        rsf,
        get_fused_moe_router_workspace(g.device, max(g.shape[0], WORKSPACE_MAX_TOKENS)),
        expert_mask=mask,
        moe_buf=moe_buf,
        num_fused_shared_experts=n_shared,
        shared_expert_weight=shared_w,
        ep_rank=ep_rank,
        ep_size=ep_size,
        quant_type=_quant_type(mode),
    )


def _quant_errs(ref, got, M, mode):
    if mode == FP8:
        # Token-indexed and written for every token whatever it routes to, so
        # every row is defined, under EP too.
        r8, g8 = ref["a1"].view(torch.uint8), got["a1"].view(torch.uint8)
        rs, gs = ref["a1s"].view(torch.int32), got["a1s"].view(torch.int32)
        return {
            "fp8_err": int((r8 != g8).sum().item()),
            "scale_err": int((rs != gs).sum().item()),
        }
    elif mode == MXFP4:
        nv_r = int(ref["nv"][0])
        if nv_r != int(got["nv"][0]):
            return {"fp4_err": -1, "scale_err": -1}
        if nv_r == 0:
            return {"fp4_err": 0, "scale_err": 0}
        # Only rows the sort references are defined: under EP most tokens route
        # to no local expert and neither path writes their out_fp4 row.
        r4 = ref["a1"].view(torch.uint8).view(M, -1)
        g4 = got["a1"].view(torch.uint8).view(M, -1)
        live = torch.zeros(M, dtype=torch.bool)
        tok = ref["sids"][:nv_r] & 0xFFFFFF
        live[tok[tok < M].long()] = True
        errs = {"fp4_err": int((r4[live] != g4[live]).sum().item())}

        # The swizzled scale buffer is only defined below num_valid, and which
        # token owns a row is permutation-dependent. Deswizzle both, then
        # compare each side's row against the scale its own sorted_ids says it
        # should carry.
        scale_n = r4.shape[1] * 2 // GROUP_SIZE
        sr = _deswizzle(ref["a1s"], nv_r, scale_n)
        sg = _deswizzle(got["a1s"], nv_r, scale_n)
        tok_r = (ref["sids"][:nv_r] & 0xFFFFFF).clamp(max=M)
        tok_g = (got["sids"][:nv_r] & 0xFFFFFF).clamp(max=M)
        want = torch.zeros(M + 1, scale_n, dtype=torch.uint8)
        real = tok_r < M
        want[tok_r[real].long()] = sr[real]
        errs["scale_err"] = int((sg != want[tok_g.long()]).sum().item())
        return errs
    else:
        raise ValueError(f"unknown quant mode {mode!r}")


def _compare(ref, got, M, topk, unit_size, mode=MXFP4):
    """Return {metric: error count or magnitude}; all zero means pass."""
    errs = {}

    # topk: per-token set of (expert -> weight). Selection order within a token
    # is free, the weight attached to each expert is not.
    rset = {
        (r, int(e)): float(w)
        for r in range(M)
        for e, w in zip(ref["ti"][r].tolist(), ref["tw"][r].tolist())
    }
    gset = {
        (r, int(e)): float(w)
        for r in range(M)
        for e, w in zip(got["ti"][r].tolist(), got["tw"][r].tolist())
    }
    errs["topk_id_err"] = len(set(rset) ^ set(gset))
    errs["topk_w_err"] = (
        max((abs(rset[k] - gset[k]) for k in rset.keys() & gset.keys()), default=0.0)
        if rset
        else 0.0
    )

    nv_r, nv_g = int(ref["nv"][0]), int(got["nv"][0])
    errs["num_valid_err"] = int(nv_r != nv_g) + int(
        int(ref["nv"][1]) != int(got["nv"][1])
    )
    errs.update(_quant_errs(ref, got, M, mode))
    if nv_r != nv_g:
        # Everything downstream is indexed by num_valid; comparing past a
        # disagreement reports noise, so stop here.
        errs.update(sorted_id_err=-1, sorted_w_err=-1)
        return errs

    if nv_r == 0:
        # Under EP a rank can own no expert any token routed to. Nothing is
        # written, and both sides agreeing on that is the whole check.
        errs.update(expert_id_err=0, sorted_id_err=0, sorted_w_err=0.0)
        return errs

    nblk = nv_r // unit_size
    errs["expert_id_err"] = int(
        not torch.equal(ref["seids"][:nblk], got["seids"][:nblk])
    )

    rs, gs = ref["sids"][:nv_r].tolist(), got["sids"][:nv_r].tolist()
    rw, gw = ref["sw"][:nv_r].tolist(), got["sw"][:nv_r].tolist()
    bad_id, dw = 0, 0.0
    for blk in range(nblk):
        s = slice(blk * unit_size, (blk + 1) * unit_size)
        a, c = sorted(zip(rs[s], rw[s])), sorted(zip(gs[s], gw[s]))
        if [x[0] for x in a] != [x[0] for x in c]:
            bad_id += 1
        else:
            dw = max(dw, max((abs(x[1] - y[1]) for x, y in zip(a, c)), default=0.0))
    errs["sorted_id_err"] = bad_id
    errs["sorted_w_err"] = dw
    return errs


def _run_case(
    M,
    E,
    topk,
    unit_size,
    bias_dtype,
    need_renorm,
    rsf,
    ep,
    use_moe_buf,
    seed=None,
    n_shared=0,
    shared_w=1.0,
    mode=MXFP4,
    cols=4096,
):
    """One config, correctness only. Returns the error dict."""
    mask = None
    ep_rank, ep_size = 0, 1
    if ep is not None:
        ep_rank, ep_size = ep[0], ep[1]
        mask = _make_mask(E, ep_rank, ep_size, vllm_shape=ep[2], n_shared=n_shared)
    g, b, h = _inputs(M, E, bias_dtype, M if seed is None else seed, cols)
    shared = {
        "n_shared": n_shared,
        "shared_w": shared_w,
        "ep_rank": ep_rank,
        "ep_size": ep_size,
        "mode": mode,
    }
    ref = _run_stock(g, b, h, M, E, topk, unit_size, need_renorm, rsf, mask, **shared)
    topk_total = topk + n_shared
    got = _alloc(ref, M, topk_total, mode)
    moe_buf = None
    if use_moe_buf:
        # Poisoned: the kernel is supposed to zero it while routing runs.
        moe_buf = torch.full_like(ref["moe_buf"], 7.0)
    _call_fused(
        g,
        b,
        h,
        got,
        E,
        topk,
        unit_size,
        need_renorm,
        rsf,
        mask,
        moe_buf,
        **shared,
    )
    torch.cuda.synchronize()
    errs = _compare(ref, got, M, topk_total, unit_size, mode)
    errs["moe_buf_err"] = int((moe_buf != 0).sum().item()) if use_moe_buf else 0
    return errs, (g, b, h, ref, got, mask)


def _failed(errs):
    return any((v > W_TOL if k.endswith("_w_err") else v != 0) for k, v in errs.items())


def _pin_path(fused):
    """Pin the launch path. The host picks by token count otherwise, so a sweep
    over M would silently stop exercising the barrier above kSplitMinTokens.

    The env var is the crossover itself, so pinning means moving it out of
    reach on either side.
    """
    os.environ["AITER_MOE_ROUTING_SPLIT"] = "2147483647" if fused else "0"


def _unpin_path():
    os.environ.pop("AITER_MOE_ROUTING_SPLIT", None)


def _fill_pat(t, tag):
    """Fill with a pattern that is wrong in every interpretation.

    Zero and -1 are both plausible values for these buffers, so poisoning with
    them lets an unwritten slot pass by luck. 0x7f is a large positive int32, a
    huge float, and a nonzero e8m0 exponent byte.
    """
    t.view(torch.uint8).fill_(0x7F if tag % 2 == 0 else 0xA5)
    return t


def _alloc_poisoned(ref, M, topk, tag=0, mode=MXFP4):
    """Output buffers filled with allocator-garbage rather than clean values."""
    a = _alloc(ref, M, topk, mode)
    for k in KEYS:
        _fill_pat(a[k], tag)
    # num_valid is read as a count; garbage is not a meaningful start state and
    # the kernel unconditionally overwrites it.
    a["nv"].zero_()
    return a


def _tail_errs(got, M, topk, unit_size, ctx):
    """Check the region past num_valid against the kernel's own contract.

    NOT against stock: stock leaves that region untouched, so it holds whatever
    the allocator handed it. The fused kernel deliberately does more, because
    the downstream stage-1 GEMM reads an id per block before it consults
    num_valid_ids. Contract: sorted_ids = pack_id(M, topk),
    sorted_weights = 0, sorted_expert_ids = 0.

    The boundary is the kernel's OWN num_valid, not stock's: under a gating tie
    the two elect different experts, which shifts per-expert padding and makes
    the two values legitimately differ. num_valid agreement is _compare's job.
    """
    fails = []
    nv = int(got["nv"][0])
    sentinel = (M & 0xFFFFFF) | ((topk & 0xFF) << 24)
    checks = (
        ("sorted_ids", got["sids"][nv:], sentinel),
        ("sorted_weights", got["sw"][nv:], 0.0),
        ("sorted_expert_ids", got["seids"][nv // unit_size :], 0),
    )
    for name, buf, want in checks:
        if buf.numel() == 0:
            continue
        n = int((buf != want).sum())
        if n:
            fails.append(
                f"{ctx}: {name} tail has {n}/{buf.numel()} slots "
                f"!= {want} (first: {buf[buf != want][0].item()})"
            )
    return fails


def _tie_free_bias(E):
    """A bias with E distinct values, so the top-k boundary is never a tie.

    With a zero or coarse bias the boundary score genuinely ties for some
    tokens, and the two paths then pick different experts of equal weight --
    both correct. Measured: 4 of 64 tokens tie at E=320. Breaking ties is what
    makes the rest of the input space assertable; ties get their own test.
    """
    return (torch.arange(E, dtype=torch.float32) * 1e-3).to(dtypes.bf16)


@benchmark()
def bench_fused_moe_router(
    M, E, topk, unit_size, bias_dtype, need_renorm, rsf, ep_label, use_moe_buf
):
    if not SUPPORTED:
        return {}
    ep = {
        "noEP": None,
        "EP_r0": (0, 4, False),
        "EP_r2": (2, 4, False),
        "EP_vllm": (2, 4, True),
    }[ep_label]
    errs, (g, b, h, _ref, got, mask) = _run_case(
        M, E, topk, unit_size, bias_dtype, need_renorm, rsf, ep, use_moe_buf
    )

    _, fused_us = run_perftest(
        _call_fused,
        g,
        b,
        h,
        got,
        E,
        topk,
        unit_size,
        need_renorm,
        rsf,
        mask,
        None,
        num_iters=10,
        num_warmup=2,
    )
    _, stock_us = run_perftest(
        _run_stock,
        g,
        b,
        h,
        M,
        E,
        topk,
        unit_size,
        need_renorm,
        rsf,
        mask,
        num_iters=10,
        num_warmup=2,
    )
    return {
        "fused_us": fused_us,
        "stock_us": stock_us,
        "uplift": stock_us / fused_us if fused_us else 0.0,
        **errs,
    }


@benchmark()
def bench_barrier_stress(M_list, E, topk, unit_size, iters, mode=MXFP4):
    """Hammer the self-resetting grid barrier.

    The semaphore is a process-wide static that is zeroed once and then relies
    on the sense flip to rearm (see the workspace layout note in the entry). A
    flip bug does not show up in a single launch -- it poisons the *next* one. So: force the
    fused path at every M, queue many launches back to back with no host sync,
    and vary M between them so the barrier's participant count changes while
    the semaphore still carries the previous launch's state.
    """
    if not SUPPORTED:
        return {}
    prev = os.environ.get("AITER_MOE_ROUTING_SPLIT")
    # Crossover out of reach: force the barrier at every M.
    os.environ["AITER_MOE_ROUTING_SPLIT"] = "2147483647"
    try:
        # Precompute references, one per M, while nothing else is in flight.
        refs = {}
        for M in M_list:
            g, b, h = _inputs(M, E, dtypes.bf16, M)
            refs[M] = (
                g,
                b,
                h,
                _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None, mode=mode),
            )
        torch.cuda.synchronize()

        # Queue everything with no sync in between, each into its own buffers so
        # a late writer from a previous launch is visible rather than overwritten.
        outs = []
        for i in range(iters):
            M = M_list[i % len(M_list)]
            g, b, h, ref = refs[M]
            got = _alloc(ref, M, topk, mode)
            _call_fused(
                g, b, h, got, E, topk, unit_size, True, 1.0, None, None, mode=mode
            )
            outs.append((M, ref, got))
        torch.cuda.synchronize()

        bad = 0
        for M, ref, got in outs:
            if _failed(_compare(ref, got, M, topk, unit_size, mode)):
                bad += 1
        return {"launches": len(outs), "stress_err": bad}
    finally:
        if prev is None:
            os.environ.pop("AITER_MOE_ROUTING_SPLIT", None)
        else:
            os.environ["AITER_MOE_ROUTING_SPLIT"] = prev


def _expect_raises(fn, what):
    try:
        fn()
    except (RuntimeError, AssertionError):
        return 0
    print(f"ERROR: {what} was accepted, expected a check to fire")
    return 1


def check_rejects_bad_shapes(E=320, topk=8, unit_size=16, mode=MXFP4):
    """The entry must reject what it cannot serve instead of mis-routing."""
    if not SUPPORTED:
        return 0
    M = 16
    g, b, h = _inputs(M, E, dtypes.bf16, 0)
    cols = h.shape[1]
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None, mode=mode)
    got = _alloc(ref, M, topk, mode)
    bad = 0

    # Widths no TD covers; the output buffers are sized for cols, so only the
    # width check can fire.
    for bad_cols in (1024, 3072, 8192):
        h_bad = torch.randn(M, bad_cols, dtype=dtypes.bf16)
        bad += _expect_raises(
            lambda h_bad=h_bad: _call_fused(
                g, b, h_bad, got, E, topk, unit_size, True, 1.0, None, None, mode=mode
            ),
            f"cols={bad_cols}",
        )
    g_wide = torch.randn(M, 1024, dtype=dtypes.bf16)
    b_wide = torch.randn(1024, dtype=dtypes.bf16)
    bad += _expect_raises(
        lambda: _call_fused(
            g_wide,
            b_wide,
            h,
            got,
            1024,
            topk,
            unit_size,
            True,
            1.0,
            None,
            None,
            mode=mode,
        ),
        "num_experts=1024",
    )
    bad += _expect_raises(
        lambda: _call_fused(
            g, b, h, got, E, topk, 24, True, 1.0, None, None, mode=mode
        ),
        "unit_size=24 (not a power of two)",
    )
    bad += _expect_raises(
        lambda: _call_fused(
            g,
            b.to(torch.float16),
            h,
            got,
            E,
            topk,
            unit_size,
            True,
            1.0,
            None,
            None,
            mode=mode,
        ),
        "fp16 correction bias",
    )

    # Everything the entry reinterpret_casts: a wrong dtype or a strided view
    # would otherwise be read as bf16 at the wrong offsets.
    for name, gg, hh in (
        ("fp16 gating", g.to(torch.float16), h),
        ("fp16 hidden", g, h.to(torch.float16)),
        # [:, :cols] of a 2*cols-wide tensor: right shape, wrong row stride.
        ("non-contiguous gating", torch.randn(M, 2 * E, dtype=dtypes.bf16)[:, :E], h),
        (
            "non-contiguous hidden",
            g,
            torch.randn(M, 2 * cols, dtype=dtypes.bf16)[:, :cols],
        ),
    ):
        bad += _expect_raises(
            lambda gg=gg, hh=hh: _call_fused(
                gg, b, hh, got, E, topk, unit_size, True, 1.0, None, None, mode=mode
            ),
            name,
        )
    return bad


# ---------------------------------------------------------------------------
# pytest entry points (correctness only; the CLI below adds perf + sweeps)
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(not SUPPORTED, reason=_skip_msg())


def _assert_ok(errs, ctx):
    bad = {
        k: v for k, v in errs.items() if (v > W_TOL if k.endswith("_w_err") else v != 0)
    }
    assert not bad, f"{ctx}: {bad}"


# 103/104/105 straddle kSplitMinTokens, where the host switches from the fused
# barrier to a two-launch split. 23 is the phase-3 case where a truncated
# rows-per-block would leave a tail of rows owned by no block.
@pytest.mark.parametrize("M", [1, 8, 23, 32, 33, 64, 100, 103, 104, 105, 128])
@pytest.mark.parametrize("ep", [None, (0, 4, False), (2, 4, False), (2, 4, True)])
@pytest.mark.parametrize("mode", MODES)
def test_tokens_and_ep(mode, M, ep):
    errs, _ = _run_case(M, 320, 8, 16, dtypes.bf16, True, 1.0, ep, False, mode=mode)
    _assert_ok(errs, f"{mode} M={M} ep={ep}")


@pytest.mark.parametrize("mode", MODES)
def test_small_m_all_experts_masked(mode):
    """M=1 with every routed expert masked out.

    The small-M kernel has no expert that owns the token, so a routing-derived
    owner would leave the quantized row unwritten. num_valid is 0 here, so
    phase 3 emits nothing and only the quant prologue covers the row.
    """
    M, E, topk, unit_size = 1, 320, 8, 16
    mask = torch.zeros(E, dtype=torch.int32)  # this rank owns nothing
    g, b, h = _inputs(M, E, dtypes.bf16, M)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, mask, mode=mode)
    got = _alloc(ref, M, topk, mode)
    _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, mask, None, mode=mode)
    torch.cuda.synchronize()
    assert int(got["nv"][0].item()) == 0, "no owned expert should route no rows"
    _assert_ok(_compare(ref, got, M, topk, unit_size, mode), f"{mode} M=1 all-masked")


@pytest.mark.parametrize("unit_size", [16, 32, 64, 128])
@pytest.mark.parametrize("M", [1, 64, 128])
@pytest.mark.parametrize("mode", MODES)
def test_unit_size(mode, M, unit_size):
    errs, _ = _run_case(
        M, 320, 8, unit_size, dtypes.bf16, True, 1.0, None, False, mode=mode
    )
    _assert_ok(errs, f"{mode} M={M} unit_size={unit_size}")


@pytest.mark.parametrize("E,topk", [(256, 1), (256, 2), (320, 8), (512, 8)])
@pytest.mark.parametrize("M", [1, 64, 128])
@pytest.mark.parametrize("mode", MODES)
def test_experts_and_topk(mode, M, E, topk):
    errs, _ = _run_case(M, E, topk, 16, dtypes.bf16, True, 1.0, None, False, mode=mode)
    _assert_ok(errs, f"{mode} M={M} E={E} topk={topk}")


@pytest.mark.parametrize("need_renorm", [True, False])
@pytest.mark.parametrize("rsf", [1.0, 2.5])
@pytest.mark.parametrize("mode", MODES)
def test_renorm_and_scaling(mode, need_renorm, rsf):
    errs, _ = _run_case(
        64, 320, 8, 16, dtypes.bf16, need_renorm, rsf, None, False, mode=mode
    )
    _assert_ok(errs, f"{mode} need_renorm={need_renorm} rsf={rsf}")


# The entry dispatches on the real bias dtype rather than coercing it, because
# reading fp32 through the bf16 layout would silently mis-route.
@pytest.mark.parametrize("bias_dtype", [dtypes.bf16, dtypes.fp32])
@pytest.mark.parametrize("mode", MODES)
def test_bias_dtype(mode, bias_dtype):
    errs, _ = _run_case(64, 320, 8, 16, bias_dtype, True, 1.0, None, False, mode=mode)
    _assert_ok(errs, f"{mode} bias_dtype={bias_dtype}")


# moe_buf reaches the entry unzeroed, so correctness rests on the kernel
# writing every element. _run_case poisons it and _compare requires it clear,
# over the shapes that change the block mapping: shared slots widen topk, EP
# parks non-owners on the sentinel.
@pytest.mark.parametrize("M", [1, 64, 128])
@pytest.mark.parametrize(
    "ep,n_shared", [(None, 0), (None, 1), ((2, 4, True), 0), ((2, 4, True), 1)]
)
@pytest.mark.parametrize("mode", MODES)
def test_moe_buf_zero_fill(mode, M, ep, n_shared):
    errs, _ = _run_case(
        M, 320, 8, 16, dtypes.bf16, True, 1.0, ep, True, n_shared=n_shared, mode=mode
    )
    _assert_ok(errs, f"{mode} M={M} ep={ep} n_shared={n_shared} moe_buf")


@pytest.mark.parametrize("mode", MODES)
def test_barrier_rearms(mode):
    r = bench_barrier_stress([1, 33, 64, 100, 128], 320, 8, 16, 64, mode=mode)
    assert r.get("stress_err", 0) == 0, r


@pytest.mark.parametrize("mode", MODES)
def test_bad_shapes_rejected(mode):
    assert check_rejects_bad_shapes(mode=mode) == 0


# The override is a token count, so it must reject a value that only looks
# boolean: atoi("true") == 0 would silently mean "split at every M".
def test_split_override_rejects_non_numeric():
    M, E, topk, unit_size = 16, 320, 8, 16
    g, b, h = _inputs(M, E, dtypes.bf16, 0)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None)
    got = _alloc(ref, M, topk)
    prev = os.environ.get("AITER_MOE_ROUTING_SPLIT")
    os.environ["AITER_MOE_ROUTING_SPLIT"] = "true"
    try:
        with pytest.raises(RuntimeError, match="AITER_MOE_ROUTING_SPLIT"):
            _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, None, None)
    finally:
        if prev is None:
            os.environ.pop("AITER_MOE_ROUTING_SPLIT", None)
        else:
            os.environ["AITER_MOE_ROUTING_SPLIT"] = prev


# An explicit out dtype overrides hidden_states.dtype, so a bf16 input can
# still resolve to fp16 and reach the bf16-only moe_buf clear.
def test_fp16_out_dtype_unsupported():
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe_router_supported

    M, E, topk, cols = 16, 320, 8, 4096
    h = torch.randn(M, cols, dtype=dtypes.bf16)
    w1 = torch.empty(E, 512, cols // 2, dtype=dtypes.fp4x2)
    w2 = torch.empty(E, cols, 128, dtype=dtypes.fp4x2)
    assert not fused_moe_router_supported(
        h,
        w1,
        w2,
        topk,
        quant_type=QuantType.per_1x32.value,
        activation=ActivationType.Silu.value,
        dtype=dtypes.fp16,
    )


# ---------------------------------------------------------------------------
# Path-pinned tests. The host picks fused vs split by token count, so these
# force both: they share device code and differ only in where the barrier is.
# ---------------------------------------------------------------------------


@pytest.fixture(params=[True, False], ids=["fused", "split"])
def path(request):
    _pin_path(request.param)
    yield "fused" if request.param else "split"
    _unpin_path()


# Both widths the TD tiling admits, at the Solar-35B routing shape (E=128,
# topk=4, block_m 32, fp32 bias). M spans the 1/2-stage heuristic split at 16/17
# and the phase-3 tail case at 23; the path fixture covers the crossover.
@pytest.mark.parametrize("M", [1, 8, 16, 17, 23, 64, 104, 128])
@pytest.mark.parametrize("cols", HIDDEN_DIMS)
@pytest.mark.parametrize("mode", MODES)
def test_hidden_dims(path, mode, cols, M):
    E, topk, unit_size = 128, 4, 32
    g, b, h = _inputs(M, E, dtypes.fp32, M + cols, cols)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None, mode=mode)
    got = _alloc_poisoned(ref, M, topk, M, mode)
    _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, None, None, mode=mode)
    torch.cuda.synchronize()
    ctx = f"{path} {mode} cols={cols} M={M}"
    _assert_ok(_compare(ref, got, M, topk, unit_size, mode), ctx)
    assert not _tail_errs(got, M, topk, unit_size, ctx)


# The op test's main sweep stops at num_valid. The downstream stage-1 GEMM
# launches over the whole buffer and reads an id per block before it checks
# num_valid_ids, so the tail is live: under vLLM's caching allocator that
# memory is a previous graph's activations, and a missed slot is a float bit
# pattern used as a token index. unit_size moves where the boundary lands
# relative to the grid stride; M moves how much tail there is.
@pytest.mark.parametrize("unit_size", [16, 32, 128])
@pytest.mark.parametrize("M", [1, 7, 23, 33, 64, 103, 128])
def test_tail_fill_exact(path, M, unit_size):
    _, (_g, _b, _h, _ref, got, _m) = _run_case(
        M, 320, 8, unit_size, dtypes.bf16, True, 1.0, None, False
    )
    fails = _tail_errs(got, M, 8, unit_size, f"{path} M={M} u={unit_size}")
    assert not fails, fails


# An all-zero mask makes num_valid 0 and every phase-3 slice empty; an all-ones
# mask makes the unit table maximal. Both are reachable under EP, and both are
# where an off-by-one in the local-id scan shows up rather than in the balanced
# shard test_tokens_and_ep covers.
def _extreme_masks(E):
    m = {
        "none_owned": torch.zeros(E, dtype=torch.int32),
        "all_owned": torch.ones(E, dtype=torch.int32),
    }
    for name, sl in (
        ("one_owned", slice(E // 2, E // 2 + 1)),
        ("alternating", slice(None, None, 2)),
        ("last_only", slice(E - 1, E)),
        ("first_only", slice(0, 1)),
    ):
        t = torch.zeros(E, dtype=torch.int32)
        t[sl] = 1
        m[name] = t
    # vLLM's real buffer is E+1 with an always-masked sentinel slot.
    sent = torch.zeros(E + 1, dtype=torch.int32)
    sent[: E // 4] = 1
    m["vllm_sentinel"] = sent
    return m


@pytest.mark.parametrize("mask_name", list(_extreme_masks(320)))
@pytest.mark.parametrize("M", [1, 33, 128])
@pytest.mark.parametrize("mode", MODES)
def test_ep_mask_extremes(path, mode, M, mask_name):
    E, topk, unit_size = 320, 8, 16
    mask = _extreme_masks(E)[mask_name]
    g, b, h = _inputs(M, E, dtypes.bf16, M)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, mask, mode=mode)
    got = _alloc_poisoned(ref, M, topk, M, mode)
    _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, mask, None, mode=mode)
    torch.cuda.synchronize()
    ctx = f"{path} {mode} {mask_name} M={M}"
    _assert_ok(_compare(ref, got, M, topk, unit_size, mode), ctx)
    assert not _tail_errs(got, M, topk, unit_size, ctx)


def _numerics_inputs(M, E, cols=4096):
    """Inputs on the quantiser's decision boundaries, not sampled from randn.

    All-zero hidden rows drive the e8m0 abs-max reduction to its zero case, and
    huge/tiny magnitudes drive it to the exponent clamps -- the places a scale
    byte is computed rather than copied.
    """
    torch.manual_seed(0)
    hn = torch.randn(M, cols, dtype=dtypes.bf16)
    gr = torch.randn(M, E, dtype=dtypes.bf16)
    gz = torch.zeros(M, E, dtype=dtypes.bf16)
    bd = _tie_free_bias(E)

    # Every token routes to the same experts: one expert's count is M, the rest
    # are empty. Maximal skew for the phase-2 histogram.
    gs = torch.zeros(M, E, dtype=dtypes.bf16)
    gs[:, :8] = 10.0
    # One zero row among live rows: the zero-scale case must not contaminate
    # its neighbours in the swizzled scale tile.
    hm = torch.randn(M, cols, dtype=dtypes.bf16)
    hm[::2] = 0
    # Per-group magnitude swing: adjacent groups land on far apart exponents,
    # which is what the shuffle has to keep straight.
    hg = torch.randn(M, cols, dtype=dtypes.bf16) * (
        2.0
        ** torch.randint(-30, 30, (1, cols // GROUP_SIZE))
        .repeat_interleave(GROUP_SIZE, 1)
        .to(dtypes.bf16)
    )
    # One group huge: the abs-max must not leak across the group boundary.
    hs = torch.randn(M, cols, dtype=dtypes.bf16)
    hs[:, GROUP_SIZE : 2 * GROUP_SIZE] = 50000.0
    return {
        "gating_single_hot": (gs, bd, hn),
        # Gating carries no information, so selection is decided entirely by
        # the bias -- the value the fp32/bf16 dispatch reads differently.
        "bias_decides": (gz, bd, hn),
        "bias_fp32": (gz, bd.to(torch.float32), hn),
        "hidden_zero": (gr, bd, torch.zeros(M, cols, dtype=dtypes.bf16)),
        "hidden_half_zero": (gr, bd, hm),
        "hidden_huge": (gr, bd, torch.full((M, cols), 60000.0, dtype=dtypes.bf16)),
        "hidden_tiny": (gr, bd, torch.full((M, cols), 1e-38, dtype=dtypes.bf16)),
        "hidden_group_swing": (gr, bd, hg),
        "hidden_one_group_huge": (gr, bd, hs),
    }


@pytest.mark.parametrize("case", list(_numerics_inputs(1, 320)))
@pytest.mark.parametrize("mode", MODES)
def test_adversarial_numerics(path, mode, case):
    M, E, topk, unit_size = 64, 320, 8, 16
    g, b, h = _numerics_inputs(M, E)[case]
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None, mode=mode)
    got = _alloc_poisoned(ref, M, topk, 3, mode)
    _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, None, None, mode=mode)
    torch.cuda.synchronize()
    _assert_ok(_compare(ref, got, M, topk, unit_size, mode), f"{path} {mode} {case}")


def _fp8_edge_rows(M, cols):
    """Hidden rows on per-token FP8's decision points, one kind per case.

    The row abs-max sets the scale and every element is divided by it, so the
    cases probe the reduction (zero, outlier, sign), the fp32 reciprocal at the
    bf16 range ends, and the e4m3 conversion's rounding and saturation.
    """
    torch.manual_seed(0)
    r = torch.randn(M, cols, dtype=dtypes.bf16)
    bf16 = torch.finfo(dtypes.bf16)
    outlier = r.clone()
    outlier[:, cols // 3] = 3.0e4
    alt = torch.ones(M, cols, dtype=dtypes.bf16)
    alt[:, 1::2] = -1
    alt *= torch.arange(1, M + 1, dtype=dtypes.bf16).view(-1, 1)
    # A row max of 448 puts the scale at ~1, so 1.0625 and 1.1875 land near
    # the midpoints between the e4m3 values 1, 1.125 and 1.25.
    mid = torch.tensor([448.0, 1.0625, 1.1875, 3.25, -17.0, 0.0078125])
    mid = mid.to(dtypes.bf16).repeat(cols // mid.numel() + 1)[:cols]
    one = torch.zeros(M, cols, dtype=dtypes.bf16)
    one[:, 7] = 1.0
    # Row-wise mixes, so one call covers every kind next to its neighbours.
    mixed = r.clone()
    mixed[0::4] = 0
    mixed[1::4] *= 1e-30
    mixed[2::4] = -mixed[2::4] * 1e30
    return {
        "zero": torch.zeros(M, cols, dtype=dtypes.bf16),
        "neg_zero": torch.full((M, cols), -0.0, dtype=dtypes.bf16),
        "one_nonzero": one,
        "tiny": r * 1e-30,
        "huge": (r.float().sign() * 3.0e38).to(dtypes.bf16),
        "bf16_max": torch.full((M, cols), -bf16.max, dtype=dtypes.bf16),
        "subnormal": torch.full((M, cols), bf16.smallest_normal / 4, dtype=dtypes.bf16),
        "outlier": outlier,
        "alternating_sign": alt,
        "rounding_midpoints": mid.view(1, -1).expand(M, -1).contiguous(),
        "mixed_rows": mixed,
    }


# Per-token FP8 against per_token_quant_hip, exact bytes and scale bits. On an
# all-zero row stock emits scale 0 and every byte 0xfe (-448, what the
# saturating convert makes of 0 * rcp(0)), not zeros; pinned for the router.
@pytest.mark.parametrize("case", list(_fp8_edge_rows(4, 2048)))
@pytest.mark.parametrize("cols", HIDDEN_DIMS)
def test_fp8_edge_rows(cols, case):
    M, E, topk, unit_size = 16, 128, 4, 32
    h = _fp8_edge_rows(M, cols)[case]
    g, b, _ = _inputs(M, E, dtypes.fp32, 0, cols)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None, mode=FP8)
    got = _alloc_poisoned(ref, M, topk, 1, FP8)
    _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, None, None, mode=FP8)
    torch.cuda.synchronize()
    _assert_ok(_compare(ref, got, M, topk, unit_size, FP8), f"cols={cols} {case}")
    if case in ("zero", "neg_zero"):
        assert bool((got["a1s"].view(torch.int32) == 0).all()), got["a1s"]
        assert bool((got["a1"].view(torch.uint8) == 0xFE).all())


def _tie_gatings(M, E, topk):
    # Exactly topk+1 experts tie for topk slots.
    gb = torch.full((M, E), -10.0, dtype=dtypes.bf16)
    gb[:, : topk + 1] = 3.0
    torch.manual_seed(0)
    return {
        # Every expert scores identically: maximal ambiguity.
        "all_equal": torch.zeros(M, E, dtype=dtypes.bf16),
        "all_ones": torch.ones(M, E, dtype=dtypes.bf16),
        "boundary_tie": gb,
        "randn": torch.randn(M, E, dtype=dtypes.bf16),
    }


# Deliberate top-k ties: which expert wins is a free choice and the two paths
# may differ. What is NOT free is the multiset of weights per token, the total
# routed row count, and the tail -- a tie must not drop or duplicate a row.
@pytest.mark.parametrize("case", ["all_equal", "all_ones", "boundary_tie", "randn"])
def test_gating_ties(path, case):
    M, E, topk, unit_size = 64, 320, 8, 16
    g = _tie_gatings(M, E, topk)[case]
    b = torch.zeros(E, dtype=dtypes.bf16)
    torch.manual_seed(0)
    h = torch.randn(M, 4096, dtype=dtypes.bf16)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None)
    got = _alloc_poisoned(ref, M, topk, 5)
    _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, None, None)
    torch.cuda.synchronize()

    ctx = f"{path} {case}"
    nv = int(got["nv"][0])
    # num_valid itself is NOT invariant under ties: a different winning expert
    # changes that expert's row count and so how much its last unit is padded.
    # What is invariant is the number of REAL rows -- every (token, slot) pair
    # routes somewhere exactly once.
    sentinel = (M & 0xFFFFFF) | ((topk & 0xFF) << 24)
    live = got["sids"][:nv][got["sids"][:nv] != sentinel]
    assert live.numel() == M * topk, f"{ctx}: {live.numel()} real rows, nv={nv}"
    assert len(set(live.tolist())) == live.numel(), f"{ctx}: duplicate routed rows"

    rw = torch.sort(ref["tw"], dim=1).values
    gw = torch.sort(got["tw"], dim=1).values
    d = float((rw - gw).abs().max())
    assert d <= W_TOL, f"{ctx}: per-token weight multiset differs by {d:.2e}"

    ti = got["ti"]
    assert not int(((ti < 0) | (ti >= E)).sum()), f"{ctx}: topk_ids out of range"
    dup = [r for r in range(M) if len(set(ti[r].tolist())) != topk]
    assert not dup, f"{ctx}: tokens {dup[:4]} elected a duplicate expert"
    assert not _tail_errs(got, M, topk, unit_size, ctx)


# num_valid landing exactly on max_tokens leaves no tail to fill; valid_blocks
# landing exactly on gridDim leaves the last phase-3 slice empty. M spans every
# grid inflection: the GRID floor (16), num_cu, and the split crossover (104).
@pytest.mark.parametrize("unit_size,topk", [(16, 8), (128, 8), (16, 1), (64, 2)])
@pytest.mark.parametrize(
    "M", [1, 2, 15, 16, 17, 63, 65, 102, 104, 105, 127, 129, 160, 200, 255, 256]
)
def test_shape_boundaries(path, M, unit_size, topk):
    E = 320
    g, b, h = _inputs(M, E, dtypes.bf16, M)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None)
    got = _alloc_poisoned(ref, M, topk, M)
    try:
        _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, None, None)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        # The LDS budget check is a documented refusal, not a failure.
        if "shared memory" in str(exc) or "resident" in str(exc):
            pytest.skip(str(exc))
        raise
    ctx = f"{path} M={M} u={unit_size} topk={topk}"
    _assert_ok(_compare(ref, got, M, topk, unit_size), ctx)
    assert not _tail_errs(got, M, topk, unit_size, ctx)


# The two launch paths share device code and differ only in where the barrier
# is, so any divergence localises the bug to the barrier, not to the routing.
@pytest.mark.parametrize("M", [1, 16, 33, 64, 100, 103, 104, 128, 200])
@pytest.mark.parametrize("mode", MODES)
def test_fused_split_agree(mode, M):
    E, topk, unit_size = 320, 8, 16
    g, b, h = _inputs(M, E, dtypes.bf16, M)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None, mode=mode)
    outs = {}
    try:
        for fused in (True, False):
            _pin_path(fused)
            got = _alloc_poisoned(ref, M, topk, M, mode)
            _call_fused(
                g, b, h, got, E, topk, unit_size, True, 1.0, None, None, mode=mode
            )
            torch.cuda.synchronize()
            outs[fused] = {k: got[k].clone() for k in KEYS}
    finally:
        _unpin_path()
    diff = {
        k: int((outs[True][k] != outs[False][k]).sum())
        for k in KEYS
        if not torch.equal(outs[True][k], outs[False][k])
    }
    assert not diff, f"{mode} M={M}: fused and split differ in {diff}"


# The workspace is allocated by host code that runs only at capture time, so
# its pointer is baked into the replayed launch. get_fused_moe_router_workspace
# retains the buffer for the life of the process; a caller that allocated its
# own and dropped it would leave the graph replaying through freed memory.
# Capture small then large on ONE stream (the order and the key vLLM uses) and
# require the small graph to keep replaying correctly.
def test_graph_replay_survives_workspace_growth():
    E, topk, unit_size = 320, 8, 16
    stream = torch.cuda.Stream()
    # One pool shared by every capture, as vLLM does: a private per-graph pool
    # is never handed out, so a freed block could not be reused and the test
    # would pass vacuously.
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for M in (16, 128):
        g, b, h = _inputs(M, E, dtypes.bf16, M)
        ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None)
        got = _alloc_poisoned(ref, M, topk, M)
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, None, None)
        torch.cuda.current_stream().wait_stream(warm)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, None, None)
        torch.cuda.synchronize()
        # Retain the inputs: a replay reads them, so letting them be freed
        # hands the replay recycled memory and makes it nondeterministic.
        graphs[M] = (graph, ref, got, (g, b, h))

    for M, (graph, ref, got, _keep) in graphs.items():
        for r in range(8):
            for k in KEYS:
                _fill_pat(got[k], r)
            got["nv"].zero_()
            graph.replay()
            torch.cuda.synchronize()
            ctx = f"M={M} replay {r} after growing to 128"
            _assert_ok(_compare(ref, got, M, topk, unit_size), ctx)
            assert not _tail_errs(got, M, topk, unit_size, ctx)


# ---------------------------------------------------------------------------
# Fused shared experts. The full matrix is {non-EP, EP} x {n_shared 0, 1}:
# without EP every token owns its shared rows; with EP ownership round-robins
# so the post-MoE all-reduce sees exactly one copy.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_shared", [1])
@pytest.mark.parametrize("M", [1, 8, 33, 64, 103, 104, 128])
@pytest.mark.parametrize("mode", MODES)
def test_shared_non_ep(mode, M, n_shared):
    errs, _ = _run_case(
        M,
        320,
        8,
        16,
        dtypes.bf16,
        True,
        1.0,
        None,
        False,
        n_shared=n_shared,
        mode=mode,
    )
    _assert_ok(errs, f"{mode} M={M} n_shared={n_shared}")


@pytest.mark.parametrize("n_shared", [1])
@pytest.mark.parametrize("ep", [(0, 4, True), (2, 4, True), (3, 4, True), (0, 1, True)])
@pytest.mark.parametrize("M", [1, 33, 128])
@pytest.mark.parametrize("cols", HIDDEN_DIMS)
@pytest.mark.parametrize("mode", MODES)
def test_shared_ep(path, mode, cols, M, ep, n_shared):
    errs, _ = _run_case(
        M,
        320,
        8,
        16,
        dtypes.bf16,
        True,
        1.0,
        ep,
        False,
        n_shared=n_shared,
        mode=mode,
        cols=cols,
    )
    _assert_ok(errs, f"{path} {mode} cols={cols} M={M} ep={ep} n_shared={n_shared}")


# The shared weight is not renormalized and not scaled by rsf: the kernel
# writes it after the renorm block, so a routed-weight change must not move it.
@pytest.mark.parametrize("need_renorm,rsf", [(True, 1.0), (True, 2.5), (False, 2.5)])
@pytest.mark.parametrize("shared_w", [1.0, 0.4])
@pytest.mark.parametrize("mode", MODES)
def test_shared_weight_untouched_by_renorm(mode, need_renorm, rsf, shared_w):
    M, E, topk, n_shared = 64, 320, 8, 1
    errs, (_g, _b, _h, _ref, got, _m) = _run_case(
        M,
        E,
        topk,
        16,
        dtypes.bf16,
        need_renorm,
        rsf,
        None,
        False,
        n_shared=n_shared,
        shared_w=shared_w,
        mode=mode,
    )
    _assert_ok(errs, f"{mode} renorm={need_renorm} rsf={rsf} w={shared_w}")
    tail = got["tw"][:, topk:]
    assert torch.allclose(tail, torch.full_like(tail, shared_w)), tail


# The property the round-robin exists for: summed over every rank, each token
# contributes exactly one row per shared expert. Anything else means the
# post-MoE all-reduce double-counts (or drops) the shared output.
# M=3 leaves the high ranks owning nothing, M=33 leaves a ragged tail; M=128
# is divisible by every ep_size here and so exercises neither.
@pytest.mark.parametrize("M", [3, 33, 128])
@pytest.mark.parametrize("ep_size", [1, 2, 4, 8])
@pytest.mark.parametrize("n_shared", [1])
@pytest.mark.parametrize("mode", MODES)
def test_shared_no_double_count(mode, ep_size, n_shared, M):
    E, topk, unit_size = 320, 8, 16
    g, b, h = _inputs(M, E, dtypes.bf16, 0)
    counts = torch.zeros(M, n_shared, dtype=torch.int64, device="cpu")
    for ep_rank in range(ep_size):
        mask = _make_mask(E, ep_rank, ep_size, n_shared=n_shared)
        ref = _run_stock(
            g,
            b,
            h,
            M,
            E,
            topk,
            unit_size,
            True,
            1.0,
            mask,
            n_shared,
            1.0,
            ep_rank,
            ep_size,
            mode,
        )
        got = _alloc_poisoned(ref, M, topk + n_shared, ep_rank, mode)
        _call_fused(
            g,
            b,
            h,
            got,
            E,
            topk,
            unit_size,
            True,
            1.0,
            mask,
            None,
            n_shared,
            1.0,
            ep_rank,
            ep_size,
            mode,
        )
        torch.cuda.synchronize()
        tail = got["ti"][:, topk:]
        # Every emitted id is either this rank's owned shared slot or the
        # sentinel; nothing else is a legal value for the shared lanes.
        want = E + torch.arange(n_shared, device=tail.device).view(1, -1)
        assert bool(((tail == want) | (tail == E + n_shared)).all()), tail
        counts += (tail == want).to(torch.int64).cpu()
    assert bool((counts == 1).all()), (
        f"{mode} ep_size={ep_size} n_shared={n_shared}: shared rows per token "
        f"min={int(counts.min())} max={int(counts.max())}, expected exactly 1"
    )


# n_shared == 0 must reach the same instantiation as before the feature
# existed: the shared lanes fold out and nothing about the output moves.
@pytest.mark.parametrize("ep", [None, (2, 4, True)])
@pytest.mark.parametrize("M", [1, 64, 128])
@pytest.mark.parametrize("mode", MODES)
def test_nshared_zero_unchanged(mode, M, ep):
    E, topk, unit_size = 320, 8, 16
    mask = None if ep is None else _make_mask(E, ep[0], ep[1], vllm_shape=ep[2])
    g, b, h = _inputs(M, E, dtypes.bf16, M)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, mask, mode=mode)
    outs = []
    # Defaulted vs explicitly-zero shared args, and under ep_size 1 either way.
    for kwargs in ({}, {"n_shared": 0, "shared_w": 1.0, "ep_rank": 0, "ep_size": 1}):
        got = _alloc_poisoned(ref, M, topk, M, mode)
        _call_fused(
            g, b, h, got, E, topk, unit_size, True, 1.0, mask, None, **kwargs, mode=mode
        )
        torch.cuda.synchronize()
        outs.append({k: got[k].clone() for k in KEYS})
    diff = [k for k in KEYS if not torch.equal(outs[0][k], outs[1][k])]
    assert not diff, f"{mode} M={M} ep={ep}: explicit zero-shared args changed {diff}"
    _assert_ok(
        _compare(ref, outs[0], M, topk, unit_size, mode), f"{mode} M={M} ep={ep}"
    )


@pytest.mark.parametrize("unit_size", [16, 32, 128])
@pytest.mark.parametrize("M", [1, 23, 64, 128])
@pytest.mark.parametrize("mode", MODES)
def test_shared_tail_fill_exact(path, mode, M, unit_size):
    topk, n_shared = 8, 1
    _, (_g, _b, _h, _ref, got, _m) = _run_case(
        M,
        320,
        topk,
        unit_size,
        dtypes.bf16,
        True,
        1.0,
        None,
        False,
        n_shared=n_shared,
        mode=mode,
    )
    ctx = f"{path} {mode} M={M} u={unit_size} n_shared={n_shared}"
    assert not _tail_errs(got, M, topk + n_shared, unit_size, ctx)


# Phase 1 places the shared rows on the lanes just past topk, inside the one
# wave the selection runs in, so topk + n_shared must fit in 64.
@pytest.mark.parametrize("topk,n_shared,ok", [(63, 1, True), (64, 1, False)])
@pytest.mark.parametrize("mode", MODES)
def test_shared_topk_total_boundary(mode, topk, n_shared, ok):
    M, E, unit_size = 8, 320, 16
    g, b, h = _inputs(M, E, dtypes.bf16, 0)
    ref = _run_stock(
        g, b, h, M, E, topk, unit_size, True, 1.0, None, n_shared, 1.0, 0, 1, mode
    )
    got = _alloc(ref, M, topk + n_shared, mode)
    call = lambda: _call_fused(
        g,
        b,
        h,
        got,
        E,
        topk,
        unit_size,
        True,
        1.0,
        None,
        None,
        n_shared,
        1.0,
        0,
        1,
        mode,
    )
    if ok:
        call()
        torch.cuda.synchronize()
        _assert_ok(
            _compare(ref, got, M, topk + n_shared, unit_size, mode),
            f"{mode} topk={topk}",
        )
    else:
        with pytest.raises(RuntimeError):
            call()


# The 512 cap is on routed experts alone. The pair scan reaches 2*BlockSize
# slots; the fused shared tail past that is filled serially, so a full
# 512-expert model runs with shared experts fused, under EP too.
@pytest.mark.parametrize(
    "E,n_shared,ep,ok",
    [
        (512, 0, None, True),
        (512, 0, (0, 4, False), True),
        (511, 1, None, True),
        (512, 1, None, True),
        (512, 1, (0, 4, False), True),
        (513, 0, None, False),
        (513, 1, None, False),
    ],
)
@pytest.mark.parametrize("mode", MODES)
def test_expert_slot_cap(mode, E, n_shared, ep, ok):
    M, topk, unit_size = 8, 8, 16
    ep_rank, ep_size = (0, 1) if ep is None else (ep[0], ep[1])
    mask = None if ep is None else _make_mask(E, ep_rank, ep_size, n_shared=n_shared)
    g, b, h = _inputs(M, E, dtypes.bf16, 0)
    shared = {
        "n_shared": n_shared,
        "shared_w": 1.0,
        "ep_rank": ep_rank,
        "ep_size": ep_size,
        "mode": mode,
    }
    if not ok:
        ref = _run_stock(g, b, h, M, 320, topk, unit_size, True, 1.0, None, mode=mode)
        got = _alloc(ref, M, topk + n_shared, mode)
        with pytest.raises(RuntimeError):
            _call_fused(
                g, b, h, got, E, topk, unit_size, True, 1.0, mask, None, **shared
            )
        return
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, mask, **shared)
    got = _alloc(ref, M, topk + n_shared, mode)
    _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, mask, None, **shared)
    torch.cuda.synchronize()
    _assert_ok(
        _compare(ref, got, M, topk + n_shared, unit_size, mode), f"{mode} E={E}"
    )


@pytest.mark.parametrize("mode", MODES)
def test_shared_bad_args_rejected(mode):
    """Configurations the entry cannot serve must be refused, not mis-routed."""
    M, E, topk, unit_size, n_shared = 16, 320, 8, 16, 1
    g, b, h = _inputs(M, E, dtypes.bf16, 0)
    mask = _make_mask(E, 0, 4, n_shared=n_shared)
    ref = _run_stock(
        g, b, h, M, E, topk, unit_size, True, 1.0, mask, n_shared, 1.0, 0, 4, mode
    )
    got = _alloc(ref, M, topk + n_shared, mode)

    def call(**kw):
        kw = {"n_shared": n_shared, "shared_w": 1.0, "ep_rank": 0, "ep_size": 4, **kw}
        a = kw.pop("outs", got)
        m = kw.pop("mask", mask)
        _call_fused(g, b, h, a, E, topk, unit_size, True, 1.0, m, None, **kw, mode=mode)
        torch.cuda.synchronize()

    # More shared experts than the kernel instantiates.
    with pytest.raises(RuntimeError):
        call(n_shared=2)
    # A rank index outside the world it is round-robining over.
    with pytest.raises(RuntimeError):
        call(ep_rank=4)
    with pytest.raises(RuntimeError):
        call(ep_rank=-1)
    # ep_rank/ep_size are only read under shared fusion, and EP callers pass
    # them whether or not shared experts are fused, so this must be accepted.
    call(n_shared=0)
    # ep_size > 1 with no mask has no sentinel slot for a non-owner to park on.
    with pytest.raises(RuntimeError):
        call(mask=None)
    # A mask sized for the routed experts alone leaves the shared slots and the
    # sentinel unmapped.
    with pytest.raises(RuntimeError):
        call(mask=torch.ones(E, dtype=torch.int32))
    with pytest.raises(RuntimeError):
        call(mask=torch.ones(E + n_shared, dtype=torch.int32))
    # topk_ids/topk_weights sized [M, topk] would have every token past the
    # first write past its row.
    narrow = dict(got)
    narrow["ti"] = torch.full((M, topk), -1, dtype=torch.int32)
    narrow["tw"] = torch.zeros(M, topk, dtype=dtypes.fp32)
    with pytest.raises(RuntimeError):
        call(outs=narrow)


def test_config_supported_scalar_gate():
    """Limits knowable without tensors must decline at `config_supported`.

    Backend selection calls it once; anything only `fused_moe_router_supported`
    catches is re-checked every forward.
    """
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe_router_config_supported
    from aiter.ops.flydsl.moe_common import GateMode

    ask = lambda **kw: fused_moe_router_config_supported(
        **{
            "hidden_dim": 4096,
            "hidden_dtype": dtypes.bf16,
            "w1_dtype": dtypes.fp4x2,
            "quant_type": QuantType.per_1x32.value,
            "activation": ActivationType.Silu.value,
            "gate_mode": GateMode.SEPARATED.value,
            **kw,
        }
    )
    assert ask()
    assert ask(num_fused_shared_experts=1)
    # One wave of lanes past topk: a second shared row has nowhere to go, and
    # the entry TORCH_CHECKs rather than declining.
    assert not ask(num_fused_shared_experts=2)
    assert not ask(num_fused_shared_experts=-1)
    assert ask(hidden_dim=2048)
    assert not ask(hidden_dim=3072)
    assert not ask(hidden_dtype=dtypes.fp16)
    assert not ask(activation=ActivationType.Gelu.value)
    assert not ask(gate_mode=GateMode.INTERLEAVE.value)


def test_global_num_experts_derivations():
    """The three ways `fused_moe_router_supported` learns the gating width.

    All must land on what `fused_moe_router` reads off gating_output, or the
    `topk <= global_E` and cap checks guard the wrong quantity. Sized so only
    global_E can trip them.
    """
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import FUSED_MOE_ROUTER_MAX_EXPERTS
    from aiter.fused_moe import fused_moe_router_supported as ask

    M, E, topk, n_shared, cols = 16, 320, 8, 1, 4096
    h = torch.randn(M, cols, dtype=dtypes.bf16)
    base = {
        "quant_type": QuantType.per_1x32.value,
        "activation": ActivationType.Silu.value,
    }

    def w(local_E):
        return (
            torch.empty(local_E, 512, cols // 2, dtype=dtypes.fp4x2),
            torch.empty(local_E, cols, 128, dtype=dtypes.fp4x2),
        )

    # Explicit: taken as given, whatever the shapes imply.
    w1, w2 = w(E)
    assert ask(h, w1, w2, topk, global_num_experts=E, **base)
    assert not ask(h, w1, w2, topk, global_num_experts=topk - 1, **base)
    assert not ask(
        h, w1, w2, topk, global_num_experts=FUSED_MOE_ROUTER_MAX_EXPERTS + 1, **base
    )

    # From the mask: spans routed + shared + sentinel, so both come off.
    over = FUSED_MOE_ROUTER_MAX_EXPERTS + 1
    w1, w2 = w(E // 4 + n_shared)
    assert ask(
        h,
        w1,
        w2,
        topk,
        expert_mask=_make_mask(E, 0, 4, n_shared=n_shared),
        num_fused_shared_experts=n_shared,
        **base,
    )
    assert not ask(
        h,
        w1,
        w2,
        topk,
        expert_mask=_make_mask(over, 0, 4, n_shared=n_shared),
        num_fused_shared_experts=n_shared,
        **base,
    )

    # No mask: no EP, so w1 holds routed plus shared.
    w1, w2 = w(E + n_shared)
    assert ask(h, w1, w2, topk, num_fused_shared_experts=n_shared, **base)
    w1, w2 = w(topk - 1 + n_shared)
    assert not ask(h, w1, w2, topk, num_fused_shared_experts=n_shared, **base)


def test_shared_supported_gate():
    """`fused_moe_router_supported` must agree with the entry's limits."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe_router_supported

    M, E, topk, cols = 16, 320, 8, 4096
    h = torch.randn(M, cols, dtype=dtypes.bf16)
    w1 = torch.empty(E, 512, cols // 2, dtype=dtypes.fp4x2)
    w2 = torch.empty(E, cols, 128, dtype=dtypes.fp4x2)
    ask = lambda tk, ns: fused_moe_router_supported(
        h,
        w1,
        w2,
        tk,
        quant_type=QuantType.per_1x32.value,
        activation=ActivationType.Silu.value,
        num_fused_shared_experts=ns,
    )
    assert ask(topk, 0)
    assert ask(topk, 1)
    assert not ask(topk, 2)
    assert not ask(64, 1)


@pytest.mark.parametrize("n_shared", [0, 1])
@pytest.mark.parametrize("ep", [False, True])
def test_cfg_topk_matches_stock(ep, n_shared):
    """The tuned-config key must be the one the unfused path would build.

    Stock takes topk from `topk_ids.shape[1]`, which the caller widens by the
    shared slots and, under EP + shared fusion, one fake slot. This path builds
    `topk_ids` itself, so `_cfg_topk` has to reproduce that width.
    """
    from aiter.fused_moe import _cfg_topk

    topk = 8
    mask = torch.empty(1, dtype=dtypes.i32) if ep else None
    # What vLLM allocates for the stock path (init_aiter_topK_meta_data).
    stock_width = topk + n_shared + (1 if (ep and n_shared) else 0)
    assert _cfg_topk(topk, n_shared, mask) == stock_width


# MXFP4 is fused on the 2-stage path only; a config that picks 1-stage must
# decline even though FP8 per-token now fuses its 1-stage path.
def test_mxfp4_1stage_refused(monkeypatch):
    from aiter import ActivationType
    import aiter.fused_moe as fm

    M, E, topk, cols = 16, 320, 8, 4096
    h = torch.randn(M, cols, dtype=dtypes.bf16)
    w1 = torch.empty(E, 512, cols // 2, dtype=dtypes.fp4x2)
    w2 = torch.empty(E, cols, 128, dtype=dtypes.fp4x2)
    ask = lambda: fm.fused_moe_router_supported(
        h,
        w1,
        w2,
        topk,
        quant_type=QuantType.per_1x32.value,
        activation=ActivationType.Silu.value,
    )
    assert ask()
    real = fm.get_2stage_cfgs
    monkeypatch.setattr(
        fm,
        "get_2stage_cfgs",
        lambda *a, **k: dataclasses.replace(real(*a, **k), run_1stage=True),
    )
    assert not ask()


# FP8 per-token gates at the Solar-35B MoE shape. The default heuristic runs
# 2-stage up to M=16 and 1-stage fmoe_g1u1 from M=17 on; both are fused, FLAT
# is not. The stage1_* cases force a 1-stage config that fused_moe_1stage would
# not hand to fmoe_g1u1 as-is.
_FP8_GATES = {
    "M1": (1, {}, True),
    "M16": (16, {}, True),
    "n_shared1": (16, dict(num_fused_shared_experts=1), True),
    "M17_1stage": (17, {}, True),
    "M64_1stage": (64, {}, True),
    "M128_1stage": (128, {}, True),
    "M64_1stage_n_shared1": (64, dict(num_fused_shared_experts=1), True),
    "M8_forced_1stage": (8, dict(stage1="1stage"), True),
    "stage1_xbf16": (64, dict(stage1="xbf16"), False),
    "stage1_not_fused_moe_1stage": (64, dict(stage1="other"), False),
    "stage1_flat": (64, dict(flat=True), False),
    "g1u0_1stage": (64, dict(g1u0=True, stage1="1stage"), False),
    "M129_token_cap": (129, {}, False),
    "n_shared2": (16, dict(num_fused_shared_experts=2), False),
    "per_1x128": (16, dict(quant_type=QuantType.per_1x128.value), False),
    "per_Tensor": (16, dict(quant_type=QuantType.per_Tensor.value), False),
    "per_1x32_fp8_weights": (16, dict(quant_type=QuantType.per_1x32.value), False),
    "doweight_stage1": (16, dict(doweight_stage1=True), False),
    "static_a1_scale": (16, dict(a1_scale=1.0), False),
    "fp16_out": (16, dict(dtype=dtypes.fp16), False),
    "flat_cfg": (16, dict(flat=True), False),
}


@pytest.mark.parametrize("case", list(_FP8_GATES))
def test_fp8_supported_gate(case, monkeypatch):
    from aiter import ActivationType
    import aiter.fused_moe as fm

    M, kw, want = _FP8_GATES[case]
    kw = dict(kw)
    E, dim, inter = 128, 2048, 1024
    ns = kw.get("num_fused_shared_experts", 0)
    # w1/w2 hold routed plus shared experts, as the caller would pass them.
    gu = 1 if kw.pop("g1u0", False) else 2
    w1 = torch.empty(E + ns, gu * inter, dim, dtype=dtypes.fp8)
    w2 = torch.empty(E + ns, dim, inter, dtype=dtypes.fp8)
    w1.is_shuffled = w2.is_shuffled = True
    if "a1_scale" in kw:
        kw["a1_scale"] = torch.full((1,), kw["a1_scale"], dtype=dtypes.fp32)
    real = fm.get_2stage_cfgs
    one = real(32, dim, inter, E, 4, dtypes.bf16, dtypes.fp8, dtypes.fp8,
               QuantType.per_Token, True, ActivationType.Silu, False, 0, 0, True,
               "separated", is_ep=False)
    assert one.run_1stage and one.stage1.func is fm.fused_moe_1stage
    stage1 = {
        "1stage": one.stage1,
        "xbf16": functools.partial(one.stage1, xbf16=True),
        "other": functools.partial(fm.asm_stage1, **one.stage1.keywords),
    }.get(kw.pop("stage1", None))
    if kw.pop("flat", False):
        # No tuned config selects FLAT at this shape; force the flag instead.
        monkeypatch.setattr(
            fm,
            "get_2stage_cfgs",
            lambda *a, **k: dataclasses.replace(real(*a, **k), flat=True),
        )
    elif stage1 is not None:
        forced = dataclasses.replace(one, stage1=stage1)
        monkeypatch.setattr(fm, "get_2stage_cfgs", lambda *a, **k: forced)
    kw.setdefault("quant_type", QuantType.per_Token.value)
    h = torch.empty(M, dim, dtype=dtypes.bf16)
    got = fm.fused_moe_router_supported(
        h, w1, w2, 4, activation=ActivationType.Silu.value, **kw
    )
    assert got == want, f"{case}: supported={got}, expected {want}"


@pytest.mark.parametrize("n_shared,want", [(0, True), (1, True), (2, False)])
def test_fp8_config_gate(n_shared, want):
    from aiter import ActivationType
    from aiter.fused_moe import fused_moe_router_config_supported

    for dim in HIDDEN_DIMS:
        got = fused_moe_router_config_supported(
            dim,
            dtypes.bf16,
            dtypes.fp8,
            quant_type=QuantType.per_Token.value,
            activation=ActivationType.Silu.value,
            num_fused_shared_experts=n_shared,
        )
        assert got == want, f"dim={dim} n_shared={n_shared}: {got}"


def _fp8_bad_args(M, cols):
    """Overrides the entry must refuse in FP8 mode: buffer dtypes and layouts,
    quant_type / out_q dtype pairs that name no instantiated mode, and the
    group_size the entry divides by in every mode."""
    return {
        "fp4x2_out_q": dict(a1=torch.empty(M, cols // 2, dtype=dtypes.fp4x2)),
        "uint8_out_q": dict(a1=torch.empty(M, cols, dtype=torch.uint8)),
        "e4m3fnuz_out_q": dict(a1=torch.empty(M, cols, dtype=torch.float8_e4m3fnuz)),
        "bf16_out_scale": dict(a1s=torch.empty(M, 1, dtype=dtypes.bf16)),
        "undersized_out_q": dict(a1=torch.empty(M, cols // 2, dtype=dtypes.fp8)),
        "undersized_out_scale": dict(a1s=torch.empty(M - 1, 1, dtype=dtypes.fp32)),
        "strided_out_q": dict(a1=torch.empty(M, 2 * cols, dtype=dtypes.fp8)[:, :cols]),
        "strided_out_scale": dict(a1s=torch.empty(M, 2, dtype=dtypes.fp32)[:, :1]),
        # MXFP8 is per_1x32 with fp8 out, not instantiated yet.
        "per_1x32_fp8_out": dict(quant_type=QuantType.per_1x32.value),
        "per_1x128": dict(quant_type=QuantType.per_1x128.value),
        "per_Tensor": dict(quant_type=QuantType.per_Tensor.value),
        "quant_type_99": dict(quant_type=99),
        "group_size_0": dict(group_size=0),
    }


@pytest.mark.parametrize("case", list(_fp8_bad_args(4, 2048)))
def test_fp8_bad_args_rejected(case):
    M, E, topk, unit_size, cols = 16, 128, 4, 32, 2048
    g, b, h = _inputs(M, E, dtypes.fp32, 0, cols)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None, mode=FP8)
    over = _fp8_bad_args(M, cols)[case]
    a = _alloc(ref, M, topk, FP8)
    a.update({k: v for k, v in over.items() if k in a})
    quant_type = over.get("quant_type", QuantType.per_Token.value)
    group_size = over.get("group_size", GROUP_SIZE)
    # The torch-free binding has no AiterDtype for e4m3fnuz on gfx950, so that
    # buffer is refused while converting, before the entry runs.
    exc = AssertionError if case == "e4m3fnuz_out_q" else RuntimeError
    with pytest.raises(exc):
        fused_moe_router_impl(
            g,
            b,
            h,
            a["ti"],
            a["tw"],
            a["sids"],
            a["sw"],
            a["seids"],
            a["nv"],
            a["a1"],
            a["a1s"],
            E,
            topk,
            unit_size,
            group_size,
            True,
            1.0,
            get_fused_moe_router_workspace(g.device, WORKSPACE_MAX_TOKENS),
            quant_type=quant_type,
        )
        torch.cuda.synchronize()


# An empty batch has nothing to route; the entry must refuse it cleanly rather
# than launch a zero-sized grid.
@pytest.mark.parametrize("mode", MODES)
def test_zero_tokens_rejected(mode):
    E, topk, unit_size, cols = 128, 4, 32, 2048
    g, b, h = _inputs(1, E, dtypes.fp32, 0, cols)
    ref = _run_stock(g, b, h, 1, E, topk, unit_size, True, 1.0, None, mode=mode)
    got = _alloc(ref, 1, topk, mode)
    with pytest.raises(RuntimeError, match="positive"):
        _call_fused(
            g[:0], b, h[:0], got, E, topk, unit_size, True, 1.0, None, None, mode=mode
        )
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Whole MoE, FP8 per-token: fused_moe_router vs biased_grouped_topk ->
# fused_moe_ at the Solar-35B shape. From M=17 the default heuristic runs
# 1-stage, where the router's a1 and token-indexed scale go straight to
# fmoe_g1u1. Stage 2 and fmoe_g1u1 accumulate with atomics, so outputs are
# held to the stock-vs-stock spread; the fmoe_g1u1 inputs must match exactly.
# ---------------------------------------------------------------------------

MOE_E, MOE_TOPK, MOE_DIM, MOE_INTER = 128, 4, 2048, 1024


@functools.lru_cache(maxsize=1)
def _fp8_moe_weights():
    from aiter.ops.shuffle import shuffle_weight

    torch.manual_seed(0)
    ws = []
    for shape in ((MOE_E, 2 * MOE_INTER, MOE_DIM), (MOE_E, MOE_DIM, MOE_INTER)):
        w = torch.randn(shape, dtype=dtypes.bf16) / 16
        wq, wscale = aiter.pertoken_quant(w, quant_dtype=dtypes.fp8)
        wq = shuffle_weight(wq, layout=(16, 16))
        wq.is_shuffled = True
        ws += [wq, wscale]
    w1, w1s, w2, w2s = ws
    return w1, w2, w1s, w2s


def _moe_inputs(M, seed, zero_rows=()):
    torch.manual_seed(seed)
    h = torch.randn(M, MOE_DIM, dtype=dtypes.bf16)
    h[list(zero_rows)] = 0.0
    g = torch.randn(M, MOE_E, dtype=dtypes.bf16)
    # bf16-representable, so the stock path's bf16 cast of the bias is exact.
    b = torch.randn(MOE_E, dtype=dtypes.bf16).to(dtypes.fp32)
    return h, g, b


def _moe_stock(h, g, b, w1, w2, w1s, w2s, n_shared=0):
    from aiter import ActivationType
    from aiter.fused_moe import fused_moe_

    M = h.shape[0]
    tw = torch.empty(M, MOE_TOPK, dtype=dtypes.fp32)
    ti = torch.empty(M, MOE_TOPK, dtype=torch.int32)
    biased_grouped_topk(
        g,
        b.to(g.dtype),
        tw,
        ti,
        num_expert_group=1,
        topk_group=1,
        need_renorm=True,
        routed_scaling_factor=1.0,
    )
    if n_shared:
        ti = torch.cat([ti, torch.full((M, 1), MOE_E, dtype=torch.int32)], 1)
        tw = torch.cat([tw, torch.ones(M, 1, dtype=dtypes.fp32)], 1)
    return fused_moe_(
        h,
        w1,
        w2,
        tw,
        ti,
        activation=ActivationType.Silu.value,
        quant_type=QuantType.per_Token.value,
        w1_scale=w1s,
        w2_scale=w2s,
    )


def _moe_router(h, g, b, w1, w2, w1s, w2s, n_shared=0):
    from aiter import ActivationType
    from aiter.fused_moe import fused_moe_router, fused_moe_router_supported

    kw = dict(
        activation=ActivationType.Silu.value,
        quant_type=QuantType.per_Token.value,
        num_fused_shared_experts=n_shared,
    )
    assert fused_moe_router_supported(h, w1, w2, MOE_TOPK, **kw)
    return fused_moe_router(
        h, g, b, w1, w2, MOE_TOPK, w1_scale=w1s, w2_scale=w2s, **kw
    )


def _moe_is_1stage(M):
    import aiter.fused_moe as fm
    from aiter import ActivationType

    return fm.get_2stage_cfgs(
        fm.get_padded_M(M), MOE_DIM, MOE_INTER, MOE_E, MOE_TOPK, dtypes.bf16,
        dtypes.fp8, dtypes.fp8, QuantType.per_Token, True, ActivationType.Silu,
        False, 0, 0, True, "separated", is_ep=False,
    ).run_1stage


def _assert_near_stock(out, h, g, b, W, ctx):
    ref0, ref1 = _moe_stock(h, g, b, *W), _moe_stock(h, g, b, *W)
    noise = (ref0.float() - ref1.float()).abs().max().item()
    scale = ref0.float().abs().max().item()
    diff = (out.float() - ref0.float()).abs().max().item()
    assert torch.isfinite(out).all(), f"{ctx}: non-finite output"
    assert scale > 0, f"{ctx}: stock output is all zero"
    assert diff <= max(2 * noise, 1e-2 * scale), (
        f"{ctx}: |router - stock| = {diff}, stock-vs-stock {noise}, |max| {scale}"
    )


@pytest.mark.parametrize("M", [1, 16, 17, 32, 64, 104, 128])
def test_fp8_moe_matches_stock(M):
    W = _fp8_moe_weights()
    h, g, b = _moe_inputs(M, M)
    _assert_near_stock(
        _moe_router(h, g, b, *W), h, g, b, W, f"M={M} 1stage={_moe_is_1stage(M)}"
    )


# Padded CUDA-graph tokens are all-zero rows: scale 0 and fp8 bytes 0xfe
# (see test_fp8_edge_rows). Their MoE output must be exactly zero, not NaN.
@pytest.mark.parametrize("M", [8, 64])
def test_fp8_moe_zero_rows(M):
    W = _fp8_moe_weights()
    zero = (0, M // 2, M - 1)
    h, g, b = _moe_inputs(M, 7, zero)
    h[0] = -0.0
    out = _moe_router(h, g, b, *W)
    assert out[list(zero)].count_nonzero() == 0, f"M={M}: zero rows not zero"
    _assert_near_stock(out, h, g, b, W, f"M={M} with zero rows")


class _FmoeSpy:
    """Stand-in for aiter.fmoe_g1u1, which fused_moe_1stage looks up per call."""

    NAMES = ("moe_buf", "a1", "w1", "w2", "sorted_ids", "sorted_weights",
             "sorted_expert_ids", "num_valid_ids", "topk", "a1_scale",
             "w1_scale", "w2_scale", "kernelName")

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kw):
        rec = dict(zip(self.NAMES, args), **kw)
        self.calls.append(
            {k: v.clone() if torch.is_tensor(v) else v for k, v in rec.items()}
        )


def _block_rows(rec, i, bm):
    """(sorted_id, weight) pairs of one sorted block, ordered by id."""
    ids = rec["sorted_ids"][i : i + bm].tolist()
    return sorted(zip(ids, rec["sorted_weights"][i : i + bm].tolist()))


def _fmoe_input_errs(r, s, bm):
    buffers = ("sorted_ids", "sorted_weights", "sorted_expert_ids", "moe_buf")
    errs = [
        f"{k}: {tuple(r[k].shape)} {r[k].dtype} vs {tuple(s[k].shape)} {s[k].dtype}"
        for k in (*buffers, "a1", "a1_scale")
        if r[k].shape != s[k].shape or r[k].dtype != s[k].dtype
    ]
    errs += [
        f"{k}: {r[k]!r} vs {s[k]!r}" for k in ("topk", "kernelName") if r[k] != s[k]
    ]
    if errs:
        return errs
    if not torch.equal(r["num_valid_ids"], s["num_valid_ids"]):
        return [f"num_valid_ids {r['num_valid_ids']} vs {s['num_valid_ids']}"]
    nv = int(s["num_valid_ids"][0])
    blocks = nv // bm
    r_eids, s_eids = r["sorted_expert_ids"][:blocks], s["sorted_expert_ids"][:blocks]
    if not torch.equal(r_eids, s_eids):
        errs.append("sorted_expert_ids")
    for i in range(0, nv, bm):
        rb, sb = _block_rows(r, i, bm), _block_rows(s, i, bm)
        if [x for x, _ in rb] != [x for x, _ in sb] or any(
            abs(x - y) > W_TOL for (_, x), (_, y) in zip(rb, sb)
        ):
            errs.append(f"sorted rows of block {i // bm}")
            break
    for k, view in (("a1", torch.uint8), ("a1_scale", torch.int32)):
        if not torch.equal(r[k].view(view), s[k].view(view)):
            errs.append(f"{k} bits")
    errs += [
        f"{n} moe_buf not zeroed"
        for n, x in (("router", r), ("stock", s))
        if x["moe_buf"].count_nonzero()
    ]
    return errs


# fmoe_g1u1 picks its asm kernel from sorted_expert_ids.size(0), so the router's
# sorted buffers must be sized exactly as moe_sorting sizes them, shared slot
# included. The kernel is stubbed, so the weights are never read.
@pytest.mark.parametrize("n_shared", [0, 1])
@pytest.mark.parametrize("M", [17, 64, 128])
def test_fp8_1stage_fmoe_inputs(M, n_shared, monkeypatch):
    from aiter.fused_moe import BLOCK_SIZE_M

    assert _moe_is_1stage(M)
    E = MOE_E + n_shared
    w1 = torch.empty(E, 2 * MOE_INTER, MOE_DIM, dtype=dtypes.fp8)
    w2 = torch.empty(E, MOE_DIM, MOE_INTER, dtype=dtypes.fp8)
    w1.is_shuffled = w2.is_shuffled = True
    w1s = torch.ones(E, 2 * MOE_INTER, 1, dtype=dtypes.fp32)
    w2s = torch.ones(E, MOE_DIM, 1, dtype=dtypes.fp32)
    W = (w1, w2, w1s, w2s)
    h, g, b = _moe_inputs(M, M)
    spies = {}
    for side, run in (("router", _moe_router), ("stock", _moe_stock)):
        spies[side] = _FmoeSpy()
        monkeypatch.setattr(aiter, "fmoe_g1u1", spies[side])
        run(h, g, b, *W, n_shared=n_shared)
    torch.cuda.synchronize()
    assert len(spies["router"].calls) == 1 and len(spies["stock"].calls) == 1
    r, s = spies["router"].calls[0], spies["stock"].calls[0]
    errs = _fmoe_input_errs(r, s, BLOCK_SIZE_M)
    assert not errs, f"M={M} n_shared={n_shared}: {errs}"


# The tuned config can move the 1-stage/2-stage switch; the router must follow
# whatever get_2stage_cfgs picks. Remap the lookup key so M=8 gets M=32's
# 1-stage config and M=64 gets M=16's 2-stage one, on both paths.
@pytest.mark.parametrize("M,want_1stage", [(8, True), (64, False)])
def test_fp8_moe_forced_path(M, want_1stage, monkeypatch):
    import aiter.fused_moe as fm

    real, remap = fm.get_2stage_cfgs, {8: 32, 64: 16}
    monkeypatch.setattr(
        fm, "get_2stage_cfgs", lambda t, *a, **k: real(remap.get(t, t), *a, **k)
    )
    assert _moe_is_1stage(M) == want_1stage
    W = _fp8_moe_weights()
    h, g, b = _moe_inputs(M, 50 + M)
    spy = _FmoeSpy()
    real_fmoe = aiter.fmoe_g1u1

    def count(*a, **k):
        spy(*a, **k)
        return real_fmoe(*a, **k)

    monkeypatch.setattr(aiter, "fmoe_g1u1", count)
    out = _moe_router(h, g, b, *W)
    torch.cuda.synchronize()
    assert len(spy.calls) == int(want_1stage)
    monkeypatch.setattr(aiter, "fmoe_g1u1", real_fmoe)
    _assert_near_stock(out, h, g, b, W, f"M={M} forced 1stage={want_1stage}")


@pytest.mark.parametrize("M", [1, 32, 128])
def test_fp8_moe_graph_replay(M):
    W = _fp8_moe_weights()
    hs, gs, bs = _moe_inputs(M, 200 + M)
    call = lambda: _moe_router(hs, gs, bs, *W)
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        call()
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    for seed in (1, 2, 1):
        h, g, b = _moe_inputs(M, 300 + seed)
        hs.copy_(h), gs.copy_(g), bs.copy_(b)
        graph.replay()
        torch.cuda.synchronize()
        _assert_near_stock(out.clone(), h, g, b, W, f"M={M} replay seed {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-m",
        type=str2tuple,
        default=[1, 8, 23, 32, 33, 64, 100, 103, 104, 105, 128],
        help="""Token counts. 103/104/105 straddle the fused/split crossover.
    e.g.: -m 1,64,128""",
    )
    parser.add_argument(
        "-ek",
        "--expert_topk",
        type=str2tuple,
        nargs="*",
        default=[[256, 1], [256, 2], [320, 8], [512, 8]],
        help="""Expert count and topk pairs.
    e.g.: -ek 320,8""",
    )
    parser.add_argument(
        "-u",
        "--unit_size",
        type=str2tuple,
        default=[16, 32, 64, 128],
        help="""Sort block sizes (block_size_M), powers of two.
    e.g.: -u 16,32""",
    )
    parser.add_argument(
        "-s",
        "--stress_iters",
        type=int,
        default=128,
        help="""Back-to-back launches in the grid-barrier stress test.
    e.g.: -s 512""",
    )
    args = parser.parse_args()
    # str2tuple collapses a comma-less argument to a bare int.
    as_list = lambda v: list(v) if isinstance(v, (list, tuple)) else [v]
    args.m = as_list(args.m)
    args.unit_size = as_list(args.unit_size)
    args.expert_topk = [tuple(as_list(x)) for x in args.expert_topk]

    if not SUPPORTED:
        print(f"skip test_fused_moe_router: {_skip_msg()}")
        sys.exit(0)

    base_E, base_topk, base_u = 320, 8, 16
    sections = {}

    # Tokens x EP placement, at the base config.
    df = [
        bench_fused_moe_router(
            m, base_E, base_topk, base_u, dtypes.bf16, True, 1.0, ep, False
        )
        for ep, m in itertools.product(["noEP", "EP_r0", "EP_r2", "EP_vllm"], args.m)
    ]
    sections["tokens_ep"] = pd.DataFrame(df)

    # One axis at a time off the base config: the full product is far more
    # configs than the shard budget allows and adds little coverage.
    probe_m = [m for m in (1, 64, 128) if m in args.m] or args.m[:1]
    df = [
        bench_fused_moe_router(
            m, base_E, base_topk, u, dtypes.bf16, True, 1.0, "noEP", False
        )
        for u, m in itertools.product(args.unit_size, probe_m)
    ]
    sections["unit_size"] = pd.DataFrame(df)

    df = [
        bench_fused_moe_router(
            m, E, topk, base_u, dtypes.bf16, True, 1.0, "noEP", False
        )
        for (E, topk), m in itertools.product(args.expert_topk, probe_m)
    ]
    sections["experts_topk"] = pd.DataFrame(df)

    df = [
        bench_fused_moe_router(
            64, base_E, base_topk, base_u, bias_dtype, renorm, rsf, "noEP", moe_buf
        )
        for bias_dtype, renorm, rsf, moe_buf in [
            (dtypes.bf16, True, 1.0, True),
            (dtypes.fp32, True, 1.0, False),
            (dtypes.bf16, False, 1.0, False),
            (dtypes.bf16, True, 2.5, False),
        ]
    ]
    sections["options"] = pd.DataFrame(df)

    failed = []
    for name, frame in sections.items():
        aiter.logger.info(
            "fused_moe_router %s summary (markdown):\n%s",
            name,
            frame.to_markdown(index=False),
        )
        err_cols = [c for c in frame.columns if c.endswith("_err")]
        bad = frame[
            frame[err_cols]
            .apply(lambda c: c.abs() > (W_TOL if c.name.endswith("_w_err") else 0))
            .any(axis=1)
        ]
        if len(bad):
            print(f"\nERROR: {len(bad)} failing config(s) in {name}:")
            print(bad.to_string(index=False))
            failed.append(name)

    stress = bench_barrier_stress(
        [1, 33, 64, 100, 128], base_E, base_topk, base_u, args.stress_iters
    )
    aiter.logger.info(
        "fused_moe_router barrier stress: %d launches, %d bad",
        stress["launches"],
        stress["stress_err"],
    )
    if stress["stress_err"]:
        failed.append("barrier_stress")

    if check_rejects_bad_shapes():
        failed.append("bad_shapes")

    if failed:
        print(
            f"FAIL: section(s) with regressions: {', '.join(failed)}", file=sys.stderr
        )
        sys.exit(1)
    print("All fused_moe_router tests passed!")
