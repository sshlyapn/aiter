# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import torch

from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.msa_block_select import (
    KV_FMT_FP8,
    KV_FMT_NVFP4,
    NVFP4_BLOCK_SIZE,
    index_kv_fmt,
    nvfp4_page_bytes,
)
from aiter.ops.triton._triton_kernels.msa_index_cache import (
    _msa_index_cache_insert_fp8_kernel,
    _msa_index_cache_insert_nvfp4_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def msa_index_cache_insert_fake_tensor(
    index_k: torch.Tensor,
    index_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_fmt: int | None = None,
) -> None:
    return None


@torch_compile_guard(gen_fake=msa_index_cache_insert_fake_tensor)
def msa_index_cache_insert(
    index_k: torch.Tensor,
    index_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_fmt: int | None = None,
) -> None:
    """Scatter index keys into the MSA index key cache, in the format the
    scoring passes read.

    Args:
        index_k: ``[num_tokens, head_dim]``, any float dtype.
        index_cache: fp8 -- ``[num_pages, block_size, head_dim]`` e4m3. nvfp4 --
            any 1-byte tensor, either ``[num_pages, block_size, head_dim]``
            whose pages each hold an nvfp4 page at their head, or
            ``[num_pages, >= page bytes]`` at block size NVFP4_BLOCK_SIZE.
        slot_mapping: ``[num_tokens]`` int, ``page * block_size + token``;
            negative slots are skipped.
        kv_fmt: KV_FMT_FP8 or KV_FMT_NVFP4; None reads AITER_MSA_INDEX_KV_FMT.
    """
    if kv_fmt is None:
        kv_fmt = index_kv_fmt()
    num_tokens = index_k.shape[0]
    if num_tokens == 0:
        return
    head_dim = index_k.shape[1]
    _LOGGER.info(
        f"MSA_INDEX_CACHE_INSERT: index_k={tuple(index_k.shape)} "
        f"index_cache={tuple(index_cache.shape)} kv_fmt={kv_fmt}"
    )
    if index_k.dim() != 2 or index_k.stride(1) != 1:
        raise ValueError("index_k must be [num_tokens, head_dim], contiguous rows")
    if slot_mapping.dim() != 1 or slot_mapping.shape[0] != num_tokens:
        raise ValueError("slot_mapping must be [num_tokens]")
    if slot_mapping.stride(0) != 1:
        raise ValueError("slot_mapping must be contiguous")
    if head_dim & (head_dim - 1) or head_dim < 32:
        raise ValueError(f"head_dim must be a power of two >= 32, got {head_dim}")

    if kv_fmt == KV_FMT_FP8:
        if index_cache.dim() != 3 or index_cache.shape[2] != head_dim:
            raise ValueError(
                "fp8 index_cache must be [num_pages, block_size, head_dim]"
            )
        if index_cache.stride(2) != 1:
            raise ValueError("index_cache must be contiguous along head_dim")
        _msa_index_cache_insert_fp8_kernel[(num_tokens,)](
            index_k,
            index_cache,
            slot_mapping,
            index_k.stride(0),
            index_cache.stride(0),
            index_cache.stride(1),
            BLOCK_SIZE=index_cache.shape[1],
            HEAD_DIM=head_dim,
            num_warps=1,
        )
        return
    if kv_fmt != KV_FMT_NVFP4:
        raise ValueError(f"unknown kv_fmt {kv_fmt}")

    if index_cache.element_size() != 1 or not index_cache.is_contiguous():
        raise ValueError("nvfp4 index_cache must be a contiguous byte tensor")
    if index_cache.dim() == 2:
        block_size = NVFP4_BLOCK_SIZE
        if index_cache.shape[1] < nvfp4_page_bytes(block_size, head_dim):
            raise ValueError("nvfp4 index_cache pages are smaller than one nvfp4 page")
    elif index_cache.dim() == 3 and index_cache.shape[2] == head_dim:
        block_size = index_cache.shape[1]
    else:
        raise ValueError(
            "nvfp4 index_cache must be [num_pages, block_size, head_dim] or "
            "[num_pages, page bytes]"
        )
    _msa_index_cache_insert_nvfp4_kernel[(num_tokens,)](
        index_k,
        index_cache.view(torch.uint8),
        slot_mapping,
        index_k.stride(0),
        index_cache.stride(0),
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        num_warps=1,
    )
