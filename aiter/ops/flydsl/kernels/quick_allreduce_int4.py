# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 TP∈{2,4,8} INT4 two-shot all-reduce.

INT4 nibble: [-8,+7], −1/8, 4 B/thread, 1152 B rank-tile. Scale is
group-16 signed E4M3 in the 128 B region. Super-tile ST∈{1,8}; host
uses ST=1 when ``num_tiles ≤`` the occupancy-clamped persistent grid.
Payload HBM is bf16; in-kernel math is packed fp16. Each rank owns
``atoms / world_size`` atoms of a tile (8 GPUs → 1, 4 → 2, 2 → 4); LDS
stays ``atoms * 1152``.

Workgroup geometry is a build parameter; every choice keeps the wire
format and the per-element math, so results are bit-identical:

* ``tile_atoms``: 16 B atoms per lane of a tile, 8 (32 KiB tile) or 4
  (16 KiB tile, twice the workgroups for the same payload).
* ``threads``: 256, or 512 for TP2/TP4. At 512, threads ``[256h, 256h+256)``
  load, quantize, reduce and store atoms ``[h * atoms/2, (h+1) * atoms/2)``
  of the same tile, so each wave runs half of the 256-thread chain.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, Stream, T, as_ir_value

from . import buffer_ops

WORLD = 8
SUPPORTED_WORLDS = (2, 4, 8)
SUPER_TILES = (1, 8)
SUPPORTED_THREADS = (256, 512)
SUPPORTED_TILE_ATOMS = (4, 8)
DEFAULT_GRID_CAP = 304 * 4

# Shared by host tile math, LDS PackStorage, and the kernel factory.
BLOCK = 256
ATOMS = 8
_FULL_TILE_ATOMS = ATOMS
WAVE = 64
GROUP = 8
TILE_BYTES = BLOCK * ATOMS * 16
RANK_TILE_BYTES = 1152
PACK_I32 = ATOMS * (RANK_TILE_BYTES // 4)

# (world_size, super_tile) → VGPR-limited workgroups per CU.
_RESIDENT_WGS_PER_CU = {
    (2, 1): 3,
    (2, 8): 4,
    (4, 1): 4,
    (4, 8): 5,
    (8, 1): 4,
    (8, 8): 6,
}

# Cache-policy bits of the gfx94x/gfx950 buffer intrinsics (LLVM CPol).
_CPOL_SC0 = 1
_CPOL_NT = 2
_CPOL_SC1 = 16


def clamp_grid_cap(
    requested: int,
    *,
    arch: str,
    world_size: int,
    super_tile: int,
    cu_count: int,
    resident: int | None = None,
) -> int:
    if requested < 1 or cu_count < 1:
        raise ValueError("grid_cap and cu_count must be positive")
    if arch not in ("gfx942", "gfx950"):
        raise ValueError(
            f"quick_allreduce_int4 has no residency measurement for {arch!r}"
        )
    if resident is None:
        try:
            resident = _RESIDENT_WGS_PER_CU[(int(world_size), int(super_tile))]
        except KeyError:
            raise ValueError(
                "quick_allreduce_int4 has no residency measurement for "
                f"{(world_size, super_tile)}"
            ) from None
    if resident < 1:
        raise ValueError(f"resident must be positive, got {resident}")
    return min(int(requested), int(resident) * int(cu_count))


def _f16x2(packed):
    return fx.Vector.from_elements([packed], fx.Int32).bitcast(fx.Float16)


def _i32(vec):
    return vec.bitcast(fx.Int32)[0]


def _splat_f16x2(x):
    return fx.Vector.filled(2, x, fx.Float16)


def _clamp_fp16_overflow():
    """Saturate packed fp16 overflow to ±65504 instead of Inf.

    Packed add/mul/FMA follow MODE bit 23 (FP16_OVFL). Unset, overflow
    becomes Inf and every later FMA in that tile is Inf. Set, it saturates
    to the max finite fp16. INT4 is already a saturating codec, so a rare
    overflow should not poison the all-reduce.

    FlyDSL has no MODE helper; ``llvm.amdgcn.s.setreg`` is the same
    intrinsic ``rocdl.disable_xdl_arb_stall`` uses for a different bit.
    """
    # hwreg(HW_REG_MODE, offset=23, size=2): id | (off<<6) | ((size-1)<<11)
    imm = as_ir_value(fx.Int32(0xDC1))
    val = as_ir_value(fx.Int32(1))
    llvm.call_intrinsic(None, "llvm.amdgcn.s.setreg", [imm, val], [], [])


def _shuffle_f16x2(vec, xor_off):
    return _f16x2(fx.Int32(gpu.shuffle_xor(_i32(vec), xor_off, WAVE)))


def _pair_signed_ext_f16(atom):
    """Signed extremum of 16 fp16 (this thread's 8 + xor-1 neighbor)."""
    p0, p1, p2, p3 = (
        _f16x2(atom[0]),
        _f16x2(atom[1]),
        _f16x2(atom[2]),
        _f16x2(atom[3]),
    )
    wmax = fx.maxnumf(fx.maxnumf(p0, p1), fx.maxnumf(p2, p3))
    wmin = fx.min(fx.min(p0, p1), fx.min(p2, p3))
    wmax = fx.maxnumf(wmax, _shuffle_f16x2(wmax, 1))
    wmin = fx.min(wmin, _shuffle_f16x2(wmin, 1))
    pk = (abs(wmax) > abs(wmin)).select(wmax, wmin)
    lo, hi = pk[0], pk[1]
    return fx.Float32((abs(lo) > abs(hi)).select(lo, hi))


def _atom_bf16_to_f16(atom):
    return fx.Vector(atom).bitcast(fx.BFloat16).to(fx.Float16).bitcast(fx.Int32)


def _atom_f16_to_bf16(atom):
    return fx.Vector(atom).bitcast(fx.Float16).to(fx.BFloat16).bitcast(fx.Int32)


def _f32_to_e4m3(x):
    """Pack f32 to a signed E4M3 byte: 1 sign, 4-bit exp (bias 7), 3-bit mantissa.

    Decode is ``±(1 + m/8) * 2**(e - 7)`` for every ``e``, including 0
    (no OCP denorms). ``0x7F`` / ``0xFF`` are max finite, not NaN.

    ``0x00`` is +0 only. Any other value that would land there (positive
    underflow, or exactly ``2**-7``) is stored as ``0x01``. Negative
    underflow keeps ``0x80`` (``-2**-7``). Overflow clamps both exponent
    and mantissa; saturating the exponent alone would map ``512`` to
    ``256``.
    """
    is_z = x == fx.Float32(0.0)
    sign = (x < fx.Float32(0.0)).select(fx.Int32(0x80), fx.Int32(0))
    bits = abs(x).bitcast(fx.Int32)
    e = (bits.shrui(fx.Int32(23)) & fx.Int32(255)) - fx.Int32(127)
    mant = bits & fx.Int32(0x7FFFFF)
    m3 = (mant + fx.Int32(1 << 19)).shrui(fx.Int32(20))
    carry = m3 == fx.Int32(8)
    e = e + carry.select(fx.Int32(1), fx.Int32(0))
    m3 = carry.select(fx.Int32(0), m3)
    e4 = e + fx.Int32(7)
    overflow = e4 > fx.Int32(15)
    m3 = overflow.select(fx.Int32(7), m3)
    e4 = (e4 < fx.Int32(0)).select(fx.Int32(0), overflow.select(fx.Int32(15), e4))
    byte = sign | (e4 << fx.Int32(3)) | (m3 & fx.Int32(7))
    return is_z.select(fx.Int32(0), (byte == fx.Int32(0)).select(fx.Int32(1), byte))


def _e4m3_to_f32(b):
    is_z = b == fx.Int32(0)
    sign = (b & fx.Int32(0x80)) != fx.Int32(0)
    e4 = b.shrui(fx.Int32(3)) & fx.Int32(15)
    m3 = b & fx.Int32(7)
    mag_bits = ((e4 + fx.Int32(120)) << fx.Int32(23)) | (m3 << fx.Int32(20))
    mag = mag_bits.bitcast(fx.Float32)
    signed = sign.select(-mag, mag)
    return is_z.select(fx.Float32(0.0), signed)


def _pack_e4m3_word(e, lane):
    """Four pair-E4M3 bytes into the i32 scale slot (lanes 0,2,4,6 of GROUP)."""
    base = (lane // GROUP) * GROUP
    e0 = fx.Int32(gpu.shuffle_idx(e, base, WAVE))
    e1 = fx.Int32(gpu.shuffle_idx(e, base + fx.Int32(2), WAVE))
    e2 = fx.Int32(gpu.shuffle_idx(e, base + fx.Int32(4), WAVE))
    e3 = fx.Int32(gpu.shuffle_idx(e, base + fx.Int32(6), WAVE))
    b = fx.Int32(0xFF)
    return (
        (e0 & b)
        | ((e1 & b) << fx.Int32(8))
        | ((e2 & b) << fx.Int32(16))
        | ((e3 & b) << fx.Int32(24))
    )


def _e4m3_decoding_scale(e):
    return _splat_f16x2(_e4m3_to_f32(e) * fx.Float32(-0.125))


def _quant_atom_fp16(atom, enc_pk):
    """Clamp to [-8, 7], round half to even, bias by 8: one nibble per fp16.

    In [1024, 2048) the fp16 ulp is 1, so ``w + 1032`` rounds ``w`` to the
    nearest integer, ties to even (1032 is even), and leaves
    ``roundeven(w) + 8`` in the low mantissa bits of each half.
    """
    q = []
    lo = _splat_f16x2(fx.Float16(-8.0))
    hi = _splat_f16x2(fx.Float16(7.0))
    magic = _splat_f16x2(fx.Float16(1032.0))
    nibbles = fx.Int32(0x000F000F)
    for i in range_constexpr(4):
        w = fx.min(fx.maxnumf(_f16x2(atom[i]) * enc_pk, lo), hi)
        q.append(_i32(w + magic) & nibbles)
    return q[0] | (q[1] << fx.Int32(4)) | (q[2] << fx.Int32(8)) | (q[3] << fx.Int32(12))


def _encode_scale(e):
    """fp16 ``1 / (d + 1e-7)`` of the E4M3 group scale byte *e*.

    ``v_rcp_f32`` instead of the IEEE divide: over all 256 bytes the two
    round to the same fp16 (``op_tests/test_flydsl_quick_allreduce_int4_codec.py``).
    """
    d = _e4m3_to_f32(e) * fx.Float32(-0.125)
    return _splat_f16x2(fx.Float32(rocdl.rcp(T.f32, d + fx.Float32(1e-7))))


def _codec_quant(atom, lane, tid):
    ext = _pair_signed_ext_f16(atom)
    e = _f32_to_e4m3(ext)
    packed = _quant_atom_fp16(atom, _encode_scale(e))
    is_leader = (tid % GROUP) == 0
    return packed, _pack_e4m3_word(e, lane), is_leader


def _codec_dequant(packed, scale, acc=None):
    """Unpack four INT4 nibbles to f16x2, scale, optionally FMA into *acc*.

    ``a * b + c`` does not contract to ``v_pk_fma_f16``; ``fx.fma`` does.
    Two fp16 lanes are independent channels, not a dot into f32.
    """
    out = []
    # nibble | 0x6400 then + (-1032.0) f16x2 reconstructs (q-8).
    mask = fx.Int32(0x000F000F)
    bias_hi = fx.Int32(0x64006400)
    bias_lo = _f16x2(fx.Int32(0xE408E408))
    for i in range_constexpr(4):
        q4 = (packed.shrui(fx.Int32(i * 4)) & mask) | bias_hi
        dq = _f16x2(q4) + bias_lo
        if acc is None:
            out.append(_i32(dq * scale))
        else:
            out.append(_i32(fx.fma(dq, scale, _f16x2(acc[i]))))
    return fx.Vector.from_elements(out, fx.Int32)


def _i32_to_bytes(i32_off):
    return fx.Int64(i32_off) * fx.Int64(4)


def _store_v4i32_nt_global(addr_i64, data):
    """NT-store 16 B through a per-lane global address.

    One instruction here sends 16 B to a different GPU in each lane of a
    4-wide group. A buffer-descriptor store wants the descriptor in scalar
    registers, so LLVM would serialize those lanes (one destination at a
    time). A flat global store takes the address from a vector register,
    so all destinations issue together. ``nontemporal`` skips L2; this is
    payload, not a flag. ``fx.ptr_store`` has no NT flag.
    """
    ptr_ty = ir.Type.parse("!llvm.ptr<1>")
    ptr = llvm.IntToPtrOp(ptr_ty, as_ir_value(addr_i64)).result
    llvm.StoreOp(as_ir_value(data), ptr, alignment=16, nontemporal=True)


def _load_i32_nt(rsrc, elem_off):
    """NT load: misses the CU cache.

    A persistent block re-reads the same inbox slot every tile, so a load
    that may hit the CU cache can return the previous tile's packet.
    """
    return fx.Int32(
        buffer_ops.buffer_load(
            rsrc, elem_off, vec_width=1, dtype=T.i32, cache_modifier=_CPOL_NT
        )
    )


def _load_i32_system(rsrc, elem_off):
    """System-scope load (``sc0 sc1``): misses the CU cache and bypasses L2."""
    val = buffer_ops.buffer_load(
        rsrc,
        elem_off,
        vec_width=1,
        dtype=T.i32,
        cache_modifier=_CPOL_SC0 | _CPOL_SC1,
    )
    rocdl.s_waitcnt(vmcnt=0)
    return fx.Int32(val)


@fx.struct
class PackStorage:
    pack: fx.Array[fx.Int32, PACK_I32, 16]


@fx.struct
class PackStorageHalfTile:
    pack: fx.Array[fx.Int32, PACK_I32 // 2, 16]


def make_quick_allreduce_int4_kernel(
    *,
    world_size: int = WORLD,
    super_tile: int = 1,
    grid: int,
    threads: int = BLOCK,
    tile_atoms: int = ATOMS,
    resident_wgs_per_cu: int | None = None,
):
    """Build the INT4 all-reduce kernel.

    ``threads`` and ``tile_atoms`` select the workgroup geometry (module
    docstring). ``resident_wgs_per_cu`` pins the VGPR budget so that many
    workgroups fit on one CU; the host sizes the persistent grid from it,
    and every workgroup spins on its peers, so the grid must be resident.
    """
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    if threads not in SUPPORTED_THREADS:
        raise ValueError(f"threads must be one of {SUPPORTED_THREADS}, got {threads}")
    if tile_atoms not in SUPPORTED_TILE_ATOMS:
        raise ValueError(
            f"tile_atoms must be one of {SUPPORTED_TILE_ATOMS}, got {tile_atoms}"
        )
    ATOMS = int(tile_atoms)
    TILE_BYTES = BLOCK * ATOMS * 16
    if ATOMS % world_size != 0:
        raise ValueError(f"ATOMS={ATOMS} is not divisible by world_size={world_size}")
    if super_tile not in SUPER_TILES:
        raise ValueError(f"super_tile must be one of {SUPER_TILES}, got {super_tile!r}")
    if grid < 1:
        raise ValueError(f"grid must be positive, got {grid}")
    if resident_wgs_per_cu is not None and resident_wgs_per_cu < 1:
        raise ValueError(
            f"resident_wgs_per_cu must be positive, got {resident_wgs_per_cu}"
        )
    PHASES = 2
    PHASE_REDUCE_SCATTER = 0
    PHASE_ALL_GATHER = 1
    RANK_TILE_I32 = RANK_TILE_BYTES // 4
    SCALE_I32_OFF = 256
    PAIR = 2
    NHALF = threads // BLOCK
    WAVES = threads // WAVE
    QUAD_LANES = 4
    QUADS_PER_WAVE = WAVE // QUAD_LANES
    QUADS_PER_HALF = BLOCK // QUAD_LANES
    N_SECTORS = RANK_TILE_BYTES // 64
    TILE_I32 = TILE_BYTES // 4
    TILE_FP16 = TILE_BYTES // 2
    LDS_BYTES = ATOMS * RANK_TILE_BYTES
    # Each rank owns this many 16-byte atoms of a 32 KiB tile
    # (8 GPUs → 1, 4 → 2, 2 → 4). LDS still holds all ATOMS atoms.
    rank_atoms = ATOMS // world_size
    # Tile atoms one half loads and stores, and reduced atoms it owns.
    HALF_ATOMS = ATOMS // NHALF
    HALF_K = rank_atoms // NHALF
    if NHALF > 1 and (rank_atoms % NHALF != 0 or HALF_ATOMS % rank_atoms != 0):
        raise ValueError(
            f"threads={threads} needs an even number of owned atoms per rank, "
            f"got {rank_atoms} at world_size={world_size}, tile_atoms={ATOMS}"
        )
    storage_cls = PackStorageHalfTile if ATOMS == 4 else PackStorage
    # Last-sector pad is ST * rank_atoms * RANK_TILE_I32 after the ST tiles.
    rank_payload_i32 = rank_atoms * RANK_TILE_I32
    release_i32_off = super_tile * rank_payload_i32
    wire_tile_i32 = release_i32_off + 16
    wire_tile_bytes = wire_tile_i32 * 4

    # flags_i32 is also the i32 offset of the wire area, so the flag prefix has
    # to be a whole number of 64 B sectors (16 i32s). At a smaller multiple every
    # rank-tile and release sector straddles two hardware sectors, so the 64 B
    # fanout stores and the last-sector release stop being one sector wide.
    grid_multiple = 16 // (PHASES * world_size)
    if grid % grid_multiple != 0:
        raise ValueError(
            f"grid must be a multiple of {grid_multiple} at "
            f"world_size={world_size} to keep the wire area 64 B aligned, got "
            f"{grid}"
        )
    flags_i32 = PHASES * grid * world_size

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def quick_allreduce_int4(
        rank: Int32,
        nbytes: Int64,
        num_tiles: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        colors_ptr: Int64,
        peer0: Int64,
        peer1: Int64,
        peer2: Int64,
        peer3: Int64,
        peer4: Int64,
        peer5: Int64,
        peer6: Int64,
        peer7: Int64,
    ):
        _clamp_fp16_overflow()
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        half = fx.Int32(0)
        alane = tid
        if NHALF > 1:
            half = tid // fx.Int32(BLOCK)
            alane = tid % fx.Int32(BLOCK)

        thread_layout = fx.make_layout((WAVES, WAVE), (WAVE, 1))
        wave, lane = fx.idx2crd(tid, thread_layout).unpack()
        quad_layout = fx.make_layout((QUADS_PER_WAVE, QUAD_LANES), (QUAD_LANES, 1))
        quad, lane_in_quad = fx.idx2crd(lane, quad_layout).unpack()
        quad_id = wave * fx.Int32(QUADS_PER_WAVE) + quad
        quad_in_half = quad_id % fx.Int32(QUADS_PER_HALF) if NHALF > 1 else quad_id

        pack_layout = fx.make_layout((ATOMS, RANK_TILE_I32), (RANK_TILE_I32, 1))
        # 64 B NT sectors of one 1152 B rank-tile: (sector, lane-in-quad)
        # -> i32 start of the dwordx4. Isolated NT store stays explicit.
        nt_own_layout = fx.make_layout((N_SECTORS, QUAD_LANES), (16, 4))
        # Remote NT fanout stays a vector-addressed nontemporal store.
        hbm_layout = fx.make_layout(
            (num_tiles, ATOMS, BLOCK * 4),
            (TILE_I32, BLOCK * 4, 1),
        )
        hbm_row_layout = fx.make_layout((1, BLOCK * 4), (BLOCK * 4, 1))
        hbm_copy_atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.Int32)
        hbm_copy = fx.make_tiled_copy_tv(
            hbm_copy_atom,
            fx.make_layout((1, BLOCK), (1, 1)),
            fx.make_layout((1, 4), (1, 1)),
        ).get_slice(alane)
        # Four group-16 E4M3 bytes share the i32 slot eight threads already own.
        scale_own_layout = fx.make_layout(
            (BLOCK // GROUP, GROUP // PAIR, PAIR), (GROUP, PAIR, 1)
        )
        scale_slot, pair_in_slot, _lane_in_pair = fx.idx2crd(
            alane, scale_own_layout
        ).unpack()
        # Rank-tile = 18 × 64 B Infinity Fabric sectors: 16 INT4 then 2 E4M3.
        # A workgroup has 64 quads. Lockstep: consecutive quads target
        # consecutive peers of one sector so one NT store hits every GPU.
        # A stripe of 8 sectors needs world_size*8 quads (64 at 8 GPUs);
        # leftover quads sit idle (always on the 2-sector scale tail, and
        # on the INT4 stripes when world_size < 8). Cover 18 as 8+8+2.
        fanout_int4_stripe = fx.make_layout((world_size, 8), (1, world_size))
        fanout_scale_stripe = fx.make_layout((world_size, 2), (1, world_size))
        color_layout = fx.make_layout((grid,), (1,))
        wire_slot_layout = fx.make_layout(
            (PHASES, grid, world_size, super_tile),
            (
                grid * world_size * wire_tile_i32,
                world_size * wire_tile_i32,
                wire_tile_i32,
                rank_payload_i32,
            ),
        )

        lds = fx.SharedAllocator().allocate(storage_cls).peek()
        pack = lds.pack.view(pack_layout)
        smem_ptr = lds.pack.ptr

        peers = [peer0, peer1, peer2, peer3, peer4, peer5, peer6, peer7]
        peer_vec = fx.Vector.from_elements(peers[:world_size], dtype=fx.Int64)
        self_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(rocdl.readfirstlane(T.i64, as_ir_value(peer_vec[rank])))
        )
        # inp/out are a 3-D i32 tensor consumed by TiledCopy (BufferCopy128b).
        # That API needs a FlyDSL buffer-backed tensor (layout + descriptor),
        # not a raw descriptor. create_buffer_resource_from_addr is the
        # scalar-offset buffer_load/store path used for the IPC inbox and
        # color flags. num_records_bytes is the live tensor size so a
        # partial last tile is out-of-range safe.
        hbm_i32_ptr = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=16
        )

        def _payload_tensor(ptr):
            view = fx.make_view(fx.inttoptr(hbm_i32_ptr, ptr), hbm_layout)
            return rocdl.make_buffer_tensor(
                view, max_size=False, num_records_bytes=nbytes
            )

        in_buf = _payload_tensor(inp_ptr)
        out_buf = _payload_tensor(out_ptr)
        color_rsrc = buffer_ops.create_buffer_resource_from_addr(colors_ptr)

        def _pack_off(peer, i32_idx):
            return fx.get_scalar(fx.crd2idx((peer, i32_idx), pack_layout))

        def _sub_tile_i32(phase, src, sub):
            slot = fx.get_scalar(
                fx.crd2idx((fx.Int32(phase), bid, src, sub), wire_slot_layout)
            )
            return fx.Int32(flags_i32) + slot

        def _hbm_atom_row(buf, tile, atom):
            return fx.make_view(
                fx.get_iter(fx.slice(buf, (tile, atom, None))),
                hbm_row_layout,
            )

        def _load_color():
            off = fx.get_scalar(fx.crd2idx((bid,), color_layout))
            return fx.Int32(
                buffer_ops.buffer_load(color_rsrc, off, vec_width=1, dtype=T.i32)
            )

        def _store_color(color):
            off = fx.get_scalar(fx.crd2idx((bid,), color_layout))
            buffer_ops.buffer_store(color, color_rsrc, off)

        def _my_atom(j):
            return half * fx.Int32(HALF_ATOMS) + fx.Int32(j) if NHALF > 1 else j

        def _my_k(kk):
            return half * fx.Int32(HALF_K) + fx.Int32(kk) if NHALF > 1 else kk

        def _load_tile_atoms(tile):
            atoms = []
            for j in range_constexpr(HALF_ATOMS):
                src = hbm_copy.partition_S(_hbm_atom_row(in_buf, tile, _my_atom(j)))
                frag = fx.make_fragment_like(src)
                fx.copy(hbm_copy_atom, src, frag)
                atoms.append(_atom_bf16_to_f16(fx.Vector(frag.load())))
            return atoms

        def _store_tile_atoms(tile, atoms):
            for j in range_constexpr(HALF_ATOMS):
                packed = _atom_f16_to_bf16(atoms[j])
                dst = hbm_copy.partition_D(_hbm_atom_row(out_buf, tile, _my_atom(j)))
                frag = fx.make_fragment_like(dst)
                frag.store(packed)
                fx.copy(hbm_copy_atom, frag, dst)

        def _lds_write_packet(slot, packed, scale, is_leader):
            fx.memref_store(packed, pack, (slot, alane))
            if is_leader:
                fx.memref_store(
                    scale, pack, (slot, fx.Int32(SCALE_I32_OFF) + scale_slot)
                )

        def _pack_reduce_scatter(atoms):
            """Quantize each destination's slice of this tile into LDS.

            A 32 KiB tile is 8 atoms; destination *d* owns
            ``atoms[d * rank_atoms : (d+1) * rank_atoms]``. Those packets
            are later NT-stored into *d*'s reduce-scatter inbox.
            """
            for j in range_constexpr(HALF_ATOMS):
                packed, scale, is_leader = _codec_quant(atoms[j], lane, alane)
                _lds_write_packet(fx.Int32(_my_atom(j)), packed, scale, is_leader)

        def _pack_all_gather(accs):
            """Quantize the reduced slice and replicate it for every peer.

            After reduce-scatter this rank holds ``rank_atoms`` reduced
            atoms. Copy the same packets into every destination slot so the
            NT fanout can push them into every peer's all-gather inbox.
            """
            for kk in range_constexpr(HALF_K):
                packed, scale, is_leader = _codec_quant(accs[kk], lane, alane)
                k = _my_k(kk)
                for dest in range_constexpr(world_size):
                    _lds_write_packet(
                        fx.Int32(dest * rank_atoms) + fx.Int32(k),
                        packed,
                        scale,
                        is_leader,
                    )

        def _fanout_nt(phase, inbox_src, sub):
            """NT-store one rank-tile from LDS to every peer's inbox.

            Three lockstep stripes cover the 18 sectors: INT4 [0, 8), INT4
            [8, 16), E4M3 [16, 18). ``stripe * 8`` is the first sector of
            each stripe (16 for the scale tail).
            """
            for kk in range_constexpr(HALF_K):
                k = _my_k(kk)
                for stripe in range_constexpr(3):
                    is_scale_tail = stripe == 2
                    n_sectors = 2 if is_scale_tail else 8
                    fanout = (
                        fanout_scale_stripe if is_scale_tail else fanout_int4_stripe
                    )
                    n_quads = fx.Int32(world_size * n_sectors)
                    safe = (quad_in_half < n_quads).select(quad_in_half, fx.Int32(0))
                    peer, sector_in_stripe = fx.idx2crd(safe, fanout).unpack()
                    sector = fx.Int32(stripe * 8) + sector_in_stripe
                    if quad_in_half < n_quads:
                        vec_idx = fx.get_scalar(
                            fx.crd2idx((sector, lane_in_quad), nt_own_layout)
                        )
                        pack_peer = peer
                        wire_idx = vec_idx
                        if rank_atoms != 1:
                            pack_peer = peer * fx.Int32(rank_atoms) + fx.Int32(k)
                            wire_idx = vec_idx + fx.Int32(k) * fx.Int32(RANK_TILE_I32)
                        # 4xi32 NT vector cannot go through the i32 pack view.
                        v4 = fx.ptr_load(
                            smem_ptr + _pack_off(pack_peer, vec_idx),
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                        dest = peer_vec[peer]
                        byte_off = _i32_to_bytes(
                            _sub_tile_i32(phase, inbox_src, sub) + wire_idx
                        )
                        _store_v4i32_nt_global(dest + byte_off, v4)

        def _publish(phase, inbox_src, color):
            """Drain payload NT stores, then write *color* into every peer inbox.

            Last 64 B of this rank's slot (after the ST rank-tiles) is the
            handshake: 16 i32s all equal to *color*. Peers spin on that
            sector in their copy of our slot; seeing *color* means our
            payload is visible.

            ``vmcnt(0)``: this 64-lane wave's NT payload stores are done.
            The workgroup barrier: the other 64-lane waves issued payload
            too; ``vmcnt`` is per-wave, so without the join a wave-0
            handshake could race stores still in flight. Neither can move
            after the color store, and neither can be dropped.
            """
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            limit = fx.Int32(world_size)
            safe = (quad_id < limit).select(quad_id, fx.Int32(0))
            if quad_id < limit:
                vec_idx = fx.Int32(release_i32_off) + lane_in_quad * fx.Int32(4)
                v4 = fx.Vector.from_elements([color, color, color, color], fx.Int32)
                dest = peer_vec[safe]
                byte_off = _i32_to_bytes(
                    _sub_tile_i32(phase, inbox_src, fx.Int32(0)) + vec_idx
                )
                _store_v4i32_nt_global(dest + byte_off, v4)

        def _wait_release(phase, color):
            if tid < world_size:
                elem = _sub_tile_i32(phase, tid, fx.Int32(0)) + fx.Int32(
                    release_i32_off
                )
                current = _load_i32_system(self_rsrc, elem)
                while current != color:
                    current = _load_i32_system(self_rsrc, elem)
            gpu.barrier()

        def _recv_quantized(phase, src, sub, k=0):
            # Packed dword is at base+lane; scale dword is 1024 B later at a
            # group slot. They are not adjacent, so they cannot share one
            # vector load.
            base = _sub_tile_i32(phase, src, sub)
            if not isinstance(k, int) or k:
                base = base + fx.Int32(k) * fx.Int32(RANK_TILE_I32)
            packed = _load_i32_nt(self_rsrc, base + alane)
            word = _load_i32_nt(self_rsrc, base + fx.Int32(SCALE_I32_OFF) + scale_slot)
            e = word.shrui(pair_in_slot * fx.Int32(8)) & fx.Int32(0xFF)
            return packed, _e4m3_decoding_scale(e)

        def _reduce_scattered(sub):
            """Dequant-accumulate every peer's reduce-scatter packet for *sub*."""
            accs = [None] * HALF_K
            for src in range_constexpr(world_size):
                for kk in range_constexpr(HALF_K):
                    packed, scale = _recv_quantized(
                        PHASE_REDUCE_SCATTER, fx.Int32(src), sub, _my_k(kk)
                    )
                    if accs[kk] is None:
                        accs[kk] = _codec_dequant(packed, scale)
                    else:
                        accs[kk] = _codec_dequant(packed, scale, accs[kk])
            return accs

        def _recv_all_gather(sub):
            """Dequantize every peer's all-gather packet back into full-tile atoms."""
            gathered = []
            per_src = HALF_ATOMS // rank_atoms
            for j in range_constexpr(HALF_ATOMS):
                src = fx.Int32(j // rank_atoms)
                if NHALF > 1:
                    src = half * fx.Int32(per_src) + src
                packed, scale = _recv_quantized(
                    PHASE_ALL_GATHER, src, sub, j % rank_atoms
                )
                gathered.append(_codec_dequant(packed, scale))
            return gathered

        n_block_tiles = (num_tiles - bid + fx.Int32(grid - 1)) // fx.Int32(grid)
        color = _load_color()
        if super_tile == 1:
            for i in range(fx.Int32(0), n_block_tiles, fx.Int32(1)):
                tile = bid + i * fx.Int32(grid)
                atoms = _load_tile_atoms(tile)
                _pack_reduce_scatter(atoms)
                gpu.barrier()
                _fanout_nt(PHASE_REDUCE_SCATTER, rank, fx.Int32(0))
                _publish(PHASE_REDUCE_SCATTER, rank, color)

                _wait_release(PHASE_REDUCE_SCATTER, color)
                acc = _reduce_scattered(fx.Int32(0))

                _pack_all_gather(acc)
                gpu.barrier()
                _fanout_nt(PHASE_ALL_GATHER, rank, fx.Int32(0))
                _publish(PHASE_ALL_GATHER, rank, color)

                _wait_release(PHASE_ALL_GATHER, color)
                gathered = _recv_all_gather(fx.Int32(0))
                _store_tile_atoms(tile, gathered)

                color = color + fx.Int32(1)
                if color == fx.Int32(0):  # 0 is unset sentinel
                    color = fx.Int32(1)
        else:
            st_i = fx.Int32(super_tile)
            for i in range(fx.Int32(0), n_block_tiles, st_i):
                remain = n_block_tiles - i
                n_this = (remain < st_i).select(remain, st_i)

                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * fx.Int32(grid)
                    atoms = _load_tile_atoms(tile)
                    _pack_reduce_scatter(atoms)
                    gpu.barrier()
                    _fanout_nt(PHASE_REDUCE_SCATTER, rank, s)
                    if (s + fx.Int32(1)) < n_this:
                        # Drain this wave's LDS loads, then join the WG.
                        # world_size<8 leaves waves idle in fanout; without the
                        # barrier they pack the next sub-tile into LDS while
                        # a busy wave still ptr_loads it. lgkmcnt only: NT
                        # payload stays in flight until _publish.
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()

                _publish(PHASE_REDUCE_SCATTER, rank, color)
                _wait_release(PHASE_REDUCE_SCATTER, color)

                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    acc = _reduce_scattered(s)
                    _pack_all_gather(acc)
                    gpu.barrier()
                    _fanout_nt(PHASE_ALL_GATHER, rank, s)
                    if (s + fx.Int32(1)) < n_this:
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()

                _publish(PHASE_ALL_GATHER, rank, color)
                _wait_release(PHASE_ALL_GATHER, color)

                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    gathered = _recv_all_gather(s)
                    tile = bid + (i + s) * fx.Int32(grid)
                    _store_tile_atoms(tile, gathered)

                color = color + fx.Int32(1)
                if color == fx.Int32(0):  # 0 is unset sentinel
                    color = fx.Int32(1)
        if tid == 0:
            _store_color(color)
        gpu.barrier()

    flat_wg = f"{threads},{threads}"
    value_attrs = {"rocdl.flat_work_group_size": flat_wg}
    if resident_wgs_per_cu is not None:
        # A workgroup puts threads / 256 waves on each SIMD.
        value_attrs["rocdl.waves_per_eu"] = int(resident_wgs_per_cu) * NHALF

    @flyc.jit
    def launch_quick_allreduce_int4(
        rank: Int32,
        nbytes: Int64,
        num_tiles: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        colors_ptr: Int64,
        peer0: Int64,
        peer1: Int64,
        peer2: Int64,
        peer3: Int64,
        peer4: Int64,
        peer5: Int64,
        peer6: Int64,
        peer7: Int64,
        grid_x: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        quick_allreduce_int4(
            rank,
            nbytes,
            num_tiles,
            inp_ptr,
            out_ptr,
            colors_ptr,
            peer0,
            peer1,
            peer2,
            peer3,
            peer4,
            peer5,
            peer6,
            peer7,
            value_attrs=value_attrs,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    geom_tag = ""
    if (threads, ATOMS) != (BLOCK, _FULL_TILE_ATOMS) or resident_wgs_per_cu is not None:
        geom_tag = f"_t{threads}_a{ATOMS}_r{resident_wgs_per_cu or 0}"
    launch_quick_allreduce_int4.func.__name__ = (
        f"launch_quick_allreduce_int4_ws{world_size}_st{super_tile}{geom_tag}"
    )
    return {
        "launch": launch_quick_allreduce_int4,
        "flags_bytes": flags_i32 * 4,
        "data_bytes": PHASES * grid * world_size * wire_tile_bytes,
        "lds_bytes": LDS_BYTES,
        "tile_bytes": TILE_BYTES,
        "tile_fp16": TILE_FP16,
        "rank_tile_bytes": RANK_TILE_BYTES,
        "wire_tile_bytes": wire_tile_bytes,
        "super_tile": super_tile,
        "world_size": world_size,
        "rank_atoms": rank_atoms,
        "grid": grid,
        "block": threads,
        "tile_atoms": ATOMS,
    }
