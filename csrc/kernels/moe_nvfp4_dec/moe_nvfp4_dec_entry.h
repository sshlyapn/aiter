// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// Pointer/scalar entry points of the NVFP4 MoE decode kernels, defined in
// moe_dec_r8_entry.cu. Return values are hipError_t codes, or the kernel
// file's own codes: -1 unknown cfg, -2 sort shape over its limits, -3 cfg
// built for a different K.

#include <hip/hip_runtime.h>

int moe_nvfp4_dec_entry_num_cfgs();
void moe_nvfp4_dec_entry_cfg_info(int i, int* out);
int moe_nvfp4_dec_entry_set_lut(const void* lut);
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
                             hipStream_t stream);
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
                             hipStream_t stream);
