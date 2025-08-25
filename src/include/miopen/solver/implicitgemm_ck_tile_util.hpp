/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/batched_transpose_sol.hpp>
#include <miopen/buffer_info.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/miopen_internal.h>

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
// Include CK tile headers for convolution operations
#include <ck_tile/ops/grouped_convolution.hpp>
#include <ck_tile/ops/elementwise.hpp> // For PassThrough
#include <ck_tile/ops/common/utils.hpp> // For gemm_prec_str
#include <ck_tile/host/stream_config.hpp> // For stream_config
// Include headers for potentially used types
#include <ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp> // For TileGemmTraits if needed elsewhere
#include <ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp> // Might contain relevant utils/traits

namespace miopen {
namespace solver {
namespace conv_ck_tile { // Use a distinct namespace for CK Tile utilities

// Common type aliases for CK Tile operations
// Layouts are chosen to be compatible with CK Tile's built-in transformations.
// Solvers using Channel-Last (NDHWC) data may need to transpose to these layouts.
using InLayout    = ck_tile::tensor_layout::convolution::NDHWGC;
using WeiLayout   = ck_tile::tensor_layout::convolution::GKZYXC;
using OutLayout   = ck_tile::tensor_layout::convolution::NDHWGK;
using PassThrough = ck_tile::element_wise::PassThrough; // Fixed namespace

// If other element-wise ops are needed, define them here:
// using Bilinear = ck_tile::element_wise::Bilinear;
// using Scale = ck_tile::element_wise::Scale;

// Alias for TileGemmTraits if it's used in solvers or needed for consistency
// using TileGemmTraits = ck_tile::TileGemmTraits; // This might not be necessary as a direct alias

// Type alias for stream_config to match CK Tile's type
using StreamConfig = ck_tile::stream_config;

// Helper function or alias for gemm_prec_str if needed directly
// template <typename A, typename B>
// std::string GetGemmPrecStr() { return ck_tile::gemm_prec_str<A, B>(); }

} // namespace conv_ck_tile
} // namespace solver
} // namespace miopen

#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL