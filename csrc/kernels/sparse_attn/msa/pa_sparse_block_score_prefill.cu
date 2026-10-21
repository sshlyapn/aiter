// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// fp8 index keys; nvfp4 is in pa_sparse_block_score_prefill_nvfp4.cu so the two
// compile in parallel.
#include "pa_sparse_block_select.hpp"

namespace aiter {
namespace sparse_attn {
SPARSE_PREFILL_TABLE_FMT(SPARSE_PREFILL_DEFINE, 0)
} // namespace sparse_attn
} // namespace aiter
