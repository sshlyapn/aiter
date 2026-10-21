// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>

namespace aiter {

// NVFP4 routed-expert MoE decode kernels for gfx950 (MiniMax-M3), see
// csrc/kernels/moe_nvfp4_dec/moe_dec_r8.hip and aiter/ops/moe_nvfp4_decode.py.

// Number of compiled kernel configs, and field k (0..7: id, fmt, s1, mt, nt,
// wk, wn, kc) of config i.
int64_t moe_nvfp4_dec_num_cfgs();
int64_t moe_nvfp4_dec_cfg_field(int64_t i, int64_t k);

// Copies the 256 x u32 fp8 decode LUT (device tensor) into the kernels'
// constant memory of the current device. Synchronous; not graph safe.
void moe_nvfp4_dec_set_lut(const aiter_tensor_t& lut);

// Fused routing sort + bf16 -> f16 activation cast + output zeroing.
//   x [T, H] bf16, xq [T, H] f16, out [T, H] bf16 (zeroed),
//   topk_ids [T, topk] i32, topk_w [T, topk] f32,
//   sorted_ids / sorted_w [nblk * BM] i32 / f32, sorted_eids [>= nblk] i32,
//   num_valid [1] i32.
void moe_nvfp4_dec_sort(const aiter_tensor_t& x,
                        aiter_tensor_t& xq,
                        aiter_tensor_t& out,
                        const aiter_tensor_t& topk_ids,
                        const aiter_tensor_t& topk_w,
                        aiter_tensor_t& sorted_ids,
                        aiter_tensor_t& sorted_w,
                        aiter_tensor_t& sorted_eids,
                        aiter_tensor_t& num_valid,
                        int64_t num_experts,
                        int64_t block_m,
                        int64_t nblk);

// One GEMM stage of kernel config cfg on the sorted rows.
void moe_nvfp4_dec_gemm(int64_t cfg,
                        const aiter_tensor_t& a,
                        const aiter_tensor_t& w,
                        const aiter_tensor_t& w_sc,
                        const aiter_tensor_t& inv_gs,
                        const aiter_tensor_t& sorted_ids,
                        const aiter_tensor_t& sorted_w,
                        const aiter_tensor_t& sorted_eids,
                        const aiter_tensor_t& num_valid,
                        aiter_tensor_t& out,
                        int64_t T,
                        int64_t topk,
                        int64_t K,
                        int64_t ntiles,
                        int64_t n_out,
                        int64_t grid_x,
                        int64_t grid_y);

} // namespace aiter
