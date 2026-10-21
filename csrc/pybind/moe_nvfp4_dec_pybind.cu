// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "moe_nvfp4_dec.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    m.def("moe_nvfp4_dec_num_cfgs", &aiter::moe_nvfp4_dec_num_cfgs);
    m.def("moe_nvfp4_dec_cfg_field", &aiter::moe_nvfp4_dec_cfg_field, py::arg("i"), py::arg("k"));
    m.def("moe_nvfp4_dec_set_lut", &aiter::moe_nvfp4_dec_set_lut, py::arg("lut"));
    m.def("moe_nvfp4_dec_sort",
          &aiter::moe_nvfp4_dec_sort,
          py::arg("x"),
          py::arg("xq"),
          py::arg("out"),
          py::arg("topk_ids"),
          py::arg("topk_w"),
          py::arg("sorted_ids"),
          py::arg("sorted_w"),
          py::arg("sorted_eids"),
          py::arg("num_valid"),
          py::arg("num_experts"),
          py::arg("block_m"),
          py::arg("nblk"));
    m.def("moe_nvfp4_dec_gemm",
          &aiter::moe_nvfp4_dec_gemm,
          py::arg("cfg"),
          py::arg("a"),
          py::arg("w"),
          py::arg("w_sc"),
          py::arg("inv_gs"),
          py::arg("sorted_ids"),
          py::arg("sorted_w"),
          py::arg("sorted_eids"),
          py::arg("num_valid"),
          py::arg("out"),
          py::arg("T"),
          py::arg("topk"),
          py::arg("K"),
          py::arg("ntiles"),
          py::arg("n_out"),
          py::arg("grid_x"),
          py::arg("grid_y"));
}
