//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-tile-xmx.hpp"

// Host entry point for the native-q8 DPAS (XMX) flash-attention kernel. Mirrors
// ggml_sycl_flash_attn_ext_tile (fattn-tile.cpp:11) but only the D==128 case is
// instantiated in v1 -- every other head size is excluded by the dispatch gate in
// fattn.cpp and can never reach here.
void ggml_sycl_flash_attn_ext_tile_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    switch (K->ne[0]) {
        case 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_xmx_case<128, 128>(ctx, dst);
        } break;
        case 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_xmx_case<256, 256>(ctx, dst);
        } break;
        default: {
            // The gate (ggml_sycl_get_best_fattn_kernel) guarantees D in {128,256} for XMX-Q.
            GGML_ABORT("XMX-Q flash-attention: unsupported head size (supported: D==128, D==256)");
        } break;
    }
}
