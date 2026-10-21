# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""NVFP4 routed-expert MoE decode for gfx950 (MiniMax-M3).

The kernels (csrc/kernels/moe_nvfp4_dec/moe_dec_r8.hip) decode NVFP4 weights
to f16 in registers (exact), apply the E4M3 block scale and the per-expert
global multiplier, and run f16 MFMA on bf16 -> f16 activations (no activation
quant). Three launches per MoE: fused sort + cast + output zeroing, stage 1
(gate/up + SwiGLU-OAI), stage 2 (down, weighted atomic reduce into the bf16
output, so results are not bitwise deterministic).

Usage:
    w = NvFp4DecodeWeights(w13, sc13, w2, sc2, g13, g2)  # once per layer
    y = nvfp4_moe_decode(w, x, topk_ids, topk_w)         # per step

nvfp4_moe_decode launches on the current stream, allocates through the torch
allocator and does no host sync, so it is CUDA-graph capturable. The one-time
setup (JIT build of module_moe_nvfp4_dec, fp8 LUT upload, config table) runs
in NvFp4DecodeWeights, i.e. at weight-processing time.
"""

import os
import threading

import torch
from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_moe_nvfp4_dec"
F_F16 = 2
# Largest T * topk the fused sort kernel handles (SORT_THREADS * SORT_PT).
MAX_SORT_ROWS = 4096

# Measured TP4 (E=128, I=768) best (stage1, stage2) configs by T. A T between
# two rows uses the next larger row; a T above the last row uses the last row.
TP4_DISPATCH = (
    (1, 207, 720),
    (2, 240, 720),
    (4, 207, 720),
    (64, 240, 720),
    (256, 240, 721),
    (512, 249, 723),
    (1024, 1200, 739),
)


@compile_ops(MD_NAME, develop=True)
def moe_nvfp4_dec_num_cfgs() -> int: ...


@compile_ops(MD_NAME, develop=True)
def moe_nvfp4_dec_cfg_field(i: int, k: int) -> int: ...


@compile_ops(MD_NAME, develop=True)
def moe_nvfp4_dec_set_lut(lut: Tensor) -> None: ...


@compile_ops(MD_NAME, develop=True)
def moe_nvfp4_dec_sort(
    x: Tensor,
    xq: Tensor,
    out: Tensor,
    topk_ids: Tensor,
    topk_w: Tensor,
    sorted_ids: Tensor,
    sorted_w: Tensor,
    sorted_eids: Tensor,
    num_valid: Tensor,
    num_experts: int,
    block_m: int,
    nblk: int,
) -> None: ...


@compile_ops(MD_NAME, develop=True)
def moe_nvfp4_dec_gemm(
    cfg: int,
    a: Tensor,
    w: Tensor,
    w_sc: Tensor,
    inv_gs: Tensor,
    sorted_ids: Tensor,
    sorted_w: Tensor,
    sorted_eids: Tensor,
    num_valid: Tensor,
    out: Tensor,
    T: int,
    topk: int,
    K: int,
    ntiles: int,
    n_out: int,
    grid_x: int,
    grid_y: int,
) -> None: ...


_LOCK = threading.Lock()
_CFGS: dict[int, dict] | None = None
_LUT_DEVICES: set[int] = set()
# (pybind module, torch -> aiter_tensor_t converter, raw current stream getter)
# for the hot path: the same calls the develop=True wrappers above make,
# minus their per-call custom-op dispatch (about 11 us of host time each).
_FAST = None


def _fp8_lut(device) -> Tensor:
    """[m 0..7][shift 0..15] -> 8 e4m3 bytes = |E2M1| * (1 + m/8) * 2^-shift.

    Only the fp8-LUT kernel family reads it; the module requires it set.
    """
    mags = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
    m = torch.arange(8, dtype=torch.float64).view(8, 1, 1)
    sh = torch.arange(16, dtype=torch.float64).view(1, 16, 1)
    v = (mags.view(1, 1, 8) * (1 + m / 8) * torch.exp2(-sh)).float()
    b = v.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return b.view(torch.int32).view(-1).contiguous().to(device)


def _build_cfgs() -> dict[int, dict]:
    # The kernels are gfx950-only: pin a multi-arch GPU_ARCHS to gfx950 while
    # the first call JIT-builds the module.
    archs = os.environ.get("GPU_ARCHS")
    pin = archs is not None and archs not in ("native", "gfx950")
    if pin:
        os.environ["GPU_ARCHS"] = "gfx950"
    try:
        n = moe_nvfp4_dec_num_cfgs()
    finally:
        if pin:
            os.environ["GPU_ARCHS"] = archs
    out = {}
    for i in range(n):
        cid, fmt, s1, mt, nt, wk, wn, kc = (
            moe_nvfp4_dec_cfg_field(i, k) for k in range(8)
        )
        lx = bool(kc >> 16 & 1)
        sx = bool(kc >> 16 & 2)
        out[cid] = dict(
            fmt=fmt,
            s1=bool(s1),
            mt=mt,
            nt=nt,
            wk=wk,
            wn=wn,
            kc=kc & 255,
            lx=lx,
            sx=sx,
            d=kc >> 20 & 15,
            ns=kc >> 24 & 15,
            sg=bool(kc >> 24 & 4) if sx else (not lx and bool(kc & 1024)),
        )
    return out


def prepare(device: torch.device | int | None = None) -> dict[int, dict]:
    """Build/load the module, upload the fp8 LUT to ``device`` and return the
    kernel config table. Synchronizes; call outside graph capture."""
    global _CFGS, _FAST
    if isinstance(device, torch.device):
        device = device.index
    if device is None:
        device = torch.cuda.current_device()
    dev = torch.device("cuda", device)
    with _LOCK:
        if _CFGS is None or dev.index not in _LUT_DEVICES:
            arch = torch.cuda.get_device_properties(dev).gcnArchName.split(":")[0]
            if arch != "gfx950":
                raise RuntimeError(f"NVFP4 MoE decode kernels need gfx950, got {arch}")
        if _CFGS is None:
            with torch.cuda.device(dev):
                _CFGS = _build_cfgs()
        if dev.index not in _LUT_DEVICES:
            lut = _fp8_lut(dev)
            with torch.cuda.device(dev):
                moe_nvfp4_dec_set_lut(lut)
            torch.cuda.synchronize(dev)
            _LUT_DEVICES.add(dev.index)
        if _FAST is None:
            from ..jit.core import _pybind_develop_hooks, get_module

            convert, _, raw_stream, _ = _pybind_develop_hooks()
            _FAST = (get_module(MD_NAME), convert, raw_stream)
    return _CFGS


def pick_cfg(T: int) -> tuple[int, int]:
    for t, c1, c2 in TP4_DISPATCH:
        if T <= t:
            return c1, c2
    return TP4_DISPATCH[-1][1], TP4_DISPATCH[-1][2]


# ------------------------------------------------------------------ layouts
def tile_w(packed: Tensor) -> Tensor:
    E, N, K2 = packed.shape
    return (
        packed.view(E, N // 16, 16, K2 // 64, 4, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
    )


def tile_sc_nv(sc: Tensor) -> Tensor:
    E, N, KS = sc.shape
    return (
        sc.view(E, N // 16, 16, KS // 16, 2, 4, 2)
        .permute(0, 1, 3, 5, 2, 4, 6)
        .contiguous()
    )


def sg_sc(ts: Tensor) -> Tensor:
    """Pair layout [E][NT][KT/2][64][4 B] -> SG layout [E][NT][KT/6][64][12 B]."""
    E, NT = ts.shape[:2]
    return ts.reshape(E, NT, -1, 3, 64, 4).permute(0, 1, 2, 4, 3, 5).contiguous()


class NvFp4DecodeWeights:
    """One MoE layer in the F16 kernel layout.

    Args:
        w13: packed fp4 uint8 [E, 2I, H/2] (gate rows then up rows).
        sc13: E4M3 block scales [E, 2I, H/16] (any 1-byte dtype).
        w2: packed fp4 uint8 [E, H, I/2].
        sc2: E4M3 block scales [E, H, I/16].
        g13: fp32 [E, 2] multipliers (1 / weight_global_scale of gate, up).
        g2: fp32 [E] multipliers (1 / weight_global_scale of down).
    """

    def __init__(self, w13, sc13, w2, sc2, g13, g2):
        prepare(w13.device)
        self.E, N13, H2 = w13.shape
        self.I, self.H = N13 // 2, H2 * 2
        self.g13 = g13.float().contiguous()
        self.g2 = g2.float().contiguous()
        sc13 = sc13.view(torch.uint8)
        sc2 = sc2.view(torch.uint8)
        self.w13, self.w2 = tile_w(w13), tile_w(w2)
        self.s13, self.s2 = tile_sc_nv(sc13), tile_sc_nv(sc2)
        self.s13g, self.s2g = sg_sc(self.s13), sg_sc(self.s2)
        # aiter_tensor_t views of the persistent tensors for the hot path
        cv = _FAST[1]
        self._a = {
            k: cv(getattr(self, k))
            for k in ("w13", "w2", "s13", "s2", "s13g", "s2g", "g13", "g2")
        }


def _grid(cfg, T, topk, n_out, E):
    BM = 16 * cfg["mt"]
    gy = min(E, T * topk) + (T * topk + BM - 1) // BM
    gy = (gy + 7) // 8 * 8
    if cfg["s1"]:
        gx = (n_out // 16) // (cfg["nt"] // 2 * cfg["wn"])
    else:
        gx = (n_out // 16) // (cfg["nt"] * cfg["wn"])
    return gx, gy


def nvfp4_moe_decode(
    W: NvFp4DecodeWeights,
    x: Tensor,
    topk_ids: Tensor,
    topk_w: Tensor,
) -> Tensor:
    """Routed-expert sum (no routed scaling).

    Args:
        W: layer weights from NvFp4DecodeWeights.
        x: bf16 [T, H] activations.
        topk_ids: [T, topk] expert ids (any integer dtype).
        topk_w: [T, topk] routing weights (any float dtype).

    Returns:
        bf16 [T, H].
    """
    T, H = x.shape
    topk = topk_ids.shape[1]
    if T * topk > MAX_SORT_ROWS:
        raise ValueError(f"NVFP4 MoE decode supports T * topk <= {MAX_SORT_ROWS}")
    c1, c2 = pick_cfg(T)
    cf1, cf2 = _CFGS[c1], _CFGS[c2]
    mod, cv, raw_stream = _FAST
    BM = 16 * cf1["mt"]
    dev = x.device
    x = x.contiguous()
    n = T * topk
    nblk = min(W.E, n) + (n + BM - 1) // BM
    sids = torch.empty(nblk * BM, dtype=torch.int32, device=dev)
    sw = torch.empty(nblk * BM, dtype=torch.float32, device=dev)
    seids = torch.empty((nblk + 7) // 8 * 8, dtype=torch.int32, device=dev)
    nv = torch.empty(1, dtype=torch.int32, device=dev)
    out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    xq = torch.empty(T, H, dtype=torch.float16, device=dev)
    a2 = torch.empty(n, W.I, dtype=torch.float16, device=dev)
    ids = topk_ids.to(torch.int32).contiguous()
    tw = topk_w.float().contiguous()
    sort = (cv(sids), cv(sw), cv(seids), cv(nv))
    xq_, out_, a2_ = cv(xq), cv(out), cv(a2)
    wa = W._a
    mod._set_current_hip_stream(raw_stream(dev.index))
    mod.moe_nvfp4_dec_sort(cv(x), xq_, out_, cv(ids), cv(tw), *sort, W.E, BM, nblk)
    gx, gy = _grid(cf1, T, topk, W.I, W.E)
    s13 = wa["s13g"] if cf1["sg"] else wa["s13"]
    mod.moe_nvfp4_dec_gemm(c1, xq_, wa["w13"], s13, wa["g13"], *sort, a2_, T,
                           topk, W.H, 2 * W.I // 16, W.I, gx, gy)  # fmt: skip
    gx, gy = _grid(cf2, T, topk, W.H, W.E)
    s2 = wa["s2g"] if cf2["sg"] else wa["s2"]
    mod.moe_nvfp4_dec_gemm(c2, a2_, wa["w2"], s2, wa["g2"], *sort, out_, T,
                           topk, W.I, W.H // 16, W.H, gx, gy)  # fmt: skip
    return out
