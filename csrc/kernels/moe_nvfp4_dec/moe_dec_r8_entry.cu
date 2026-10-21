// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Compiles the MiniMax-M3 NVFP4 MoE decode kernels (moe_dec_r8.hip, kept
// verbatim) and wraps their struct-based launchers in pointer/scalar entry
// points, so the aiter_tensor_t layer (moe_nvfp4_dec.cu) never needs the
// kernel file's Params / PrepParams layouts. This TU includes no aiter or
// ck_tile header on purpose: the kernel file defines short macros (DEV, SB,
// ...) and vector typedefs that are not meant to share a TU with them.
#include "moe_dec_r8.hip"

#include "moe_nvfp4_dec_entry.h"

int moe_nvfp4_dec_entry_num_cfgs() { return dec_num_cfgs(); }

void moe_nvfp4_dec_entry_cfg_info(int i, int* out) { dec_cfg_info(i, out); }

int moe_nvfp4_dec_entry_set_lut(const void* lut) { return dec_set_fp8_lut((const uint32_t*)lut); }

int moe_nvfp4_dec_entry_sort(const void* x,
                             void* xq,
                             void* out,
                             const void* ids,
                             const void* w,
                             void* sorted_ids,
                             void* sorted_w,
                             void* sorted_eids,
                             void* num_valid,
                             int T,
                             int H,
                             int topk,
                             int E,
                             int BM,
                             int nblk,
                             int xm,
                             hipStream_t stream)
{
    PrepParams p{};
    p.x           = (const uint16_t*)x;
    p.q           = (uint8_t*)xq;
    p.sc          = nullptr;
    p.out         = (uint16_t*)out;
    p.ids         = (const int*)ids;
    p.w           = (const float*)w;
    p.sorted_ids  = (int*)sorted_ids;
    p.sorted_w    = (float*)sorted_w;
    p.sorted_eids = (int*)sorted_eids;
    p.num_valid   = (int*)num_valid;
    p.T           = T;
    p.H           = H;
    p.topk        = topk;
    p.E           = E;
    p.BM          = BM;
    p.nblk        = nblk;
    return dec_sort_quant(&p, xm, stream);
}

int moe_nvfp4_dec_entry_gemm(int cfg,
                             const void* a,
                             const void* w,
                             const void* w_sc,
                             const void* inv_gs,
                             const void* sorted_ids,
                             const void* sorted_w,
                             const void* sorted_eids,
                             const void* num_valid,
                             void* out,
                             int T,
                             int topk,
                             int K,
                             int ntiles,
                             int n_out,
                             int grid_x,
                             int grid_y,
                             hipStream_t stream)
{
    // Same field values as the reference ctypes host code: no activation
    // scales, no stage1 output scales, no MXFP8 shared expert, no zeroing.
    Params p{};
    p.a           = (const uint8_t*)a;
    p.a_sc        = nullptr;
    p.w           = (const uint8_t*)w;
    p.w_sc        = (const uint8_t*)w_sc;
    p.inv_gs      = (const float*)inv_gs;
    p.sorted_ids  = (const int*)sorted_ids;
    p.sorted_w    = (const float*)sorted_w;
    p.sorted_eids = (const int*)sorted_eids;
    p.num_valid   = (const int*)num_valid;
    p.out         = (uint8_t*)out;
    p.out_sc      = nullptr;
    p.T           = T;
    p.topk        = topk;
    p.K           = K;
    p.ntiles      = ntiles;
    p.n_out       = n_out;
    p.mx8_e       = -1;
    p.w8          = nullptr;
    p.w8_sc       = nullptr;
    p.zo          = nullptr;
    p.zn16        = 0;
    return dec_launch(cfg, &p, grid_x, grid_y, stream);
}
