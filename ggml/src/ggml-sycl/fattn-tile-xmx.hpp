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

// =====================================================================================
// Native-q8 DPAS (XMX) Flash-Attention kernel for Intel Battlemage (Xe2 / Arc Pro B70).
//
// This is "Approach A": the K/V cache stays q8_0 in VRAM (never a full-tensor f16 copy),
// and only a tile's worth of K/V is stream-dequantised to f16 in SLM for two fp16xfp16->fp32
// DPAS GEMMs (QK^T and P*V). It is a sibling of the scalar TILE kernel (fattn-tile.hpp) and
// reuses its algorithm (online softmax, causal-mask add, alibi-slope hook, parallel-blocks
// write-back with dst_meta) but replaces the two inner scalar MAC loops with joint_matrix DPAS.
//
// SCOPE v1 (see the dispatch gate in fattn.cpp): q8_0 K==V, D==128, prefill (Q->ne[1] large),
// single sequence (Q->ne[3]==1), causal mask present, no sinks / alibi / softcap. Everything
// else falls through to the scalar TILE kernel -> zero regression. Runtime env
// GGML_SYCL_FA_XMX_Q gates the whole thing and DEFAULTS OFF for v1.
//
// #1 STRUCTURAL DIFFERENCE FROM THE SCALAR TILE KERNEL (read before touching anything):
//   The scalar TILE launcher hardcodes `warp_size = WARP_32_SIZE` ("can't support WARP_16_SIZE",
//   fattn-tile.hpp:1077). B70 joint_matrix DPAS REQUIRES sub-group size 16 (proven:
//   xmx_i4_verify.c:13, xmx_peak.c:38 -> intel_reqd_sub_group_size(16)). Therefore this kernel
//   is instantiated with warp_size == 16 and launch_fattn is told warp_size=16, which stamps
//   [[sycl::reqd_sub_group_size(16)]] on the kernel (fattn-common.hpp:883). Because the whole
//   kernel is templated on warp_size and every thread mapping is expressed in terms of it, the
//   SLM geometry and cooperative loaders adapt; but the config numbers (Bc, ncols, nwarps) are
//   chosen fresh here for SG=16 rather than taken from ggml_sycl_fattn_tile_get_config (which is
//   tuned for SG=32). This is the single biggest bring-up risk -- see AUTHOR_NOTES.md.
//
// SUB-GROUP -> WORK MAPPING (the mental model for the whole file):
//   * Sub-group size          = 16 lanes                              (== warp_size, == DPAS M/N/K)
//   * One workgroup           = nwarps sub-groups                     (block_dim = (16, nwarps, 1))
//   * ncols  (query cols/blk) = ncols1*ncols2, a multiple of 16       (v1: ncols2==1)
//   * nwarps                  = ncols / 16
//   * Sub-group `w`           OWNS query rows [w*16 .. w*16+15]        (exactly one DPAS M-tile)
//   * Lane `r` within a sub-group owns query row (w*16 + r) END-TO-END for the softmax:
//       it scans the full Bc-wide row of scores itself -> NO cross-lane softmax reduction,
//       which is the big simplification versus the CUDA-MMA reference (no __shfl_xor for max/sum).
//   * KV tile of width Bc is loaded cooperatively by the WHOLE workgroup and shared by all
//       sub-groups (each sub-group multiplies its own 16 query rows against the shared KV tile).
// =====================================================================================

#ifndef GGML_SYCL_FATTN_TILE_XMX_HPP
#define GGML_SYCL_FATTN_TILE_XMX_HPP

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>   // joint_matrix DPAS
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "sycl_hw.hpp"                          // gpu_arch (Battlemage arch gate)

#include <cmath>
#include <cstring>
#include <float.h>

namespace syclex = sycl::ext::oneapi::experimental;
namespace syclmx = sycl::ext::oneapi::experimental::matrix;

// ------------------------------------------------------------------------------------------------
// Compile-time knobs. All are BRINGUP tunables -- they do NOT affect correctness of the q8_0/D=128
// path, only occupancy / SLM footprint / DPAS pipelining. Sweep them on hardware (see hardware_todos).
// ------------------------------------------------------------------------------------------------

// DPAS fragment shape actually available on B70: fp16 A x fp16 B -> fp32 C, M16 N16 K16
// (b70_joint_matrix_combos.txt line 12). Do NOT change without re-probing the device.
#define XMX_MMA_M 16
#define XMX_MMA_N 16
#define XMX_MMA_K 16

// KV tile height (number of key/value tokens streamed into SLM per main-loop iteration).
// Must be a multiple of XMX_MMA_N (QK^T key N-tiles) and XMX_MMA_K (P*V key K-steps) -> multiple of 16.
// 64 => 4 key tiles for QK^T and 4 key K-steps for P*V. // BRINGUP: try 32 / 96 / 128.
#define XMX_NBATCH_FA 64

// Elements per dequantize_V_q8_0 call along the D axis. Must be even and must not straddle a q8_0
// block (QK8_0==32) -> d0 aligned to XMX_DEQ_NE and (d0 % 32) + XMX_DEQ_NE <= 32. 8 is always safe.
#define XMX_DEQ_NE 8

// SLM padding (in elements) added to each row stride to dodge bank conflicts on joint_matrix_load.
// D and Bc are already multiples of 16 so no MMA padding is needed -- this is purely conflict padding.
#define XMX_PAD_D 8   // BRINGUP: 0 / 8 / 16
#define XMX_PAD_C 8   // BRINGUP: 0 / 8 / 16

// Prefill dispatch threshold (Q->ne[1] minimum). Below this the DPAS pipeline is latency-bound and
// the scalar VEC/TILE path is preferable. Referenced by the gate in fattn.cpp.
#define FATTN_XMX_Q_MIN_COLS 32

// O-accumulator rescale strategy. DEFAULT (undefined) = SLM round-trip (layout-agnostic, provably
// correct, matches scalar semantics). Define GGML_SYCL_FA_XMX_O_INFRAGMENT ONLY after the
// (row,col) coordinate mapping of the accumulator fragment has been verified on B70 -- see AUTHOR_NOTES.
// #define GGML_SYCL_FA_XMX_O_INFRAGMENT 1

// ================================================================================================
// Eligibility predicate -- the SINGLE SOURCE OF TRUTH for the XMX-Q dispatch tier.
//
// Structured like ggml_sycl_flash_attn_ext_dnnl_supported (#25222 / fattn-onednn.cpp): a standalone
// _supported() gate that the dispatch seam in fattn.cpp calls, and that a test can reference. It is
// deliberately co-located with the kernel so the gate and the kernel's availability share ONE
// condition: the whole body is compiled only when defined(SYCL_FLASH_ATTN) && defined(GGML_SYCL_F16)
// (the kernel needs the fp16 branch of dequantize_V_q8_0 and fp32 DPAS accumulation); otherwise it
// returns false unconditionally. This makes it IMPOSSIBLE for the gate to route to a kernel that was
// compiled out -- so an F16-OFF build (default upstream) stays green: the XMX kernel instantiates as
// an unreachable stub and is never selected.
//
// Additive-only: returns true ONLY when the runtime env GGML_SYCL_FA_XMX_Q is on (default OFF, v1)
// AND every v1 clause holds. On any miss the caller falls through to the unchanged vec/tile logic.
// ================================================================================================
inline bool ggml_sycl_flash_attn_ext_xmx_q_supported(const int device, const ggml_tensor * dst) {
#if defined(SYCL_FLASH_ATTN) && defined(GGML_SYCL_F16)
    if (!g_ggml_sycl_fa_xmx_q) {                 // GGML_SYCL_FA_XMX_Q default 0 -> prod path untouched
        return false;
    }

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float max_bias = 0.0f;
    std::memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
    float logit_softcap = 0.0f;
    std::memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    // Battlemage-only (B70 == g31; include g21 so any Battlemage part a maintainer tests qualifies).
    const gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
    const bool is_battlemage =
        arch == gpu_arch::intel_gpu_bmg_g31 || arch == gpu_arch::intel_gpu_bmg_g21;

    return
        is_battlemage &&
        K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 &&   // native q8, no f16 pre-convert
        K->ne[0] == V->ne[0] &&
        (K->ne[0] == 128 || K->ne[0] == 256) &&                     // D in {128, 256} (256 = Qwen3.6 prod)
        mask != nullptr &&                                          // causal mask present
        Q->ne[3] == 1 &&                                            // single sequence
        Q->ne[1] >= FATTN_XMX_Q_MIN_COLS &&                         // prefill only (>= 32)
        sinks == nullptr &&                                         // no attention sinks
        max_bias == 0.0f &&                                         // no ALiBi
        logit_softcap == 0.0f &&                                    // no softcap
        (Q->ne[2] % K->ne[2]) == 0 &&                               // GQA divides evenly
        (K->ne[1] % FATTN_KQ_STRIDE) == 0;                          // whole KV tiles
#else
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;   // kernel is not compiled in (no SYCL_FLASH_ATTN / no GGML_SYCL_F16) -> never select it
#endif
}

// ================================================================================================
// Cooperative q8_0 -> f16 streaming loader.
//
// Loads a Bc x D tile of q8_0 K (or V) rows from VRAM, dequantising on the fly into a row-major f16
// SLM tile [Bc][ld_kv]. Only a tile's worth of f16 is ever materialised; the KV cache in VRAM stays
// q8_0. OOB rows (key index >= i_sup, the causal/length tail) are zero-filled, mirroring the scalar
// loader's `i < i_sup ? real : zero` semantics (fattn-tile.hpp:234).
//
// `kv_base` points at this (sequence, head/gqa) K or V sub-tensor; `row_stride_bytes` is the raw
// q8_0 row byte stride (K->nb[1] / V->nb[1] == 136 B for D=128, i.e. 4 * sizeof(block_q8_0)). It is
// the ORIGINAL q8_0 stride because the launcher passes need_f16=false (no f16 pre-convert), so it is
// NOT rescaled by sizeof(half) the way launch_fattn rewrites strides for the f16 path.
// ================================================================================================
template <int warp_size, int nwarps, int Bc, int D, int ld_kv, bool oob_check>
static __dpct_inline__ void flash_attn_tile_xmx_load_kv_q8_0(const char * __restrict__ kv_base,
                                                             const int64_t             row_stride_bytes,
                                                             const int                 k0,
                                                             const int                 i_sup,
                                                             sycl::half * __restrict__ tile_kv) {
    auto      item     = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int nthreads = nwarps * warp_size;
    const int flat     = item.get_local_id(1) * warp_size + item.get_local_id(2);

    constexpr int chunks_per_row = D / XMX_DEQ_NE;   // 128/8 = 16
    constexpr int total_chunks   = Bc * chunks_per_row;
    static_assert(D % XMX_DEQ_NE == 0, "D must be a multiple of XMX_DEQ_NE");
    static_assert((32 % XMX_DEQ_NE) == 0, "XMX_DEQ_NE must divide the q8_0 block (QK8_0==32)");

#pragma unroll
    for (int c = flat; c < total_chunks; c += nthreads) {
        const int r  = c / chunks_per_row;           // key/value token within this tile (0..Bc-1)
        const int d0 = (c % chunks_per_row) * XMX_DEQ_NE;

        __dpct_align__(16) sycl::half tmp[XMX_DEQ_NE];

        if (!oob_check || (k0 + r) < i_sup) {
            // Row base = first block_q8_0 of this token's D-length row. dequantize_V_q8_0 computes
            // ib = d0/32, iqs = d0%32 internally, so we pass the ROW base and the element offset d0.
            const char * row_ptr = kv_base + (int64_t)(k0 + r) * row_stride_bytes;
            dequantize_V_q8_0<sycl::half, XMX_DEQ_NE>(row_ptr, tmp, d0);
        } else {
#pragma unroll
            for (int l = 0; l < XMX_DEQ_NE; ++l) {
                tmp[l] = sycl::half(0.0f);
            }
        }

        ggml_sycl_memcpy_1<XMX_DEQ_NE * sizeof(sycl::half)>(&tile_kv[r * ld_kv + d0], tmp);
    }
}

// ================================================================================================
// Device kernel. Signature is IDENTICAL to flash_attn_tile (fattn-tile.hpp:661) so that launch_fattn
// / lauch_kernel can dispatch it with the same argument pack. K/V arrive as raw char* q8_0 with the
// original nb11/nb21 byte strides (need_f16_K==need_f16_V==false in the launcher below).
// ================================================================================================
template <int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, int warp_size>
/*
The total declared local variable size in device function flash_attn_tile_xmx may cause high register
pressure (joint_matrix fragments live in registers). Budget: S_frag[Bc/16] (transient) + O_frag[DV/16]
(persistent) + a/b fragments. Validate occupancy on B70; if it spills, reduce ncols (fewer M-tiles per
WG) or Bc. See AUTHOR_NOTES "register pressure".
*/
static void flash_attn_tile_xmx(const char *      Q,
                                const char *      K,
                                const char *      V,
                                const char *      mask,
                                const char *      sinks,
                                const int *       KV_max,
                                float *           dst,
                                sycl::float2 *    dst_meta,
                                const float       scale,
                                const float       max_bias,
                                const float       m0,
                                const float       m1,
                                const uint32_t    n_head_log2,
                                const float       logit_softcap,
                                const int32_t     ne00,
                                const sycl::uint3 ne01,
                                const int32_t     ne02,
                                const int32_t     ne03,
                                const int32_t     nb01,
                                const int32_t     nb02,
                                const int32_t     nb03,
                                const int32_t     ne10,
                                const int32_t     ne11,
                                const int32_t     ne12,
                                const int32_t     ne13,
                                const int32_t     nb11,
                                const int32_t     nb12,
                                const int64_t     nb13,
                                const int32_t     nb21,
                                const int32_t     nb22,
                                const int64_t     nb23,
                                const int32_t     ne31,
                                const int32_t     ne32,
                                const int32_t     ne33,
                                const int32_t     nb31,
                                const int32_t     nb32,
                                const int64_t     nb33) {
// v1 only compiles the real body under GGML_SYCL_F16 so dequantize_V_q8_0 takes its sycl::half branch
// (fattn-common.hpp:555) and DPAS accumulates fp32. In an F16-OFF build (default upstream) the body
// below is compiled out to an unreachable no-op stub; the dispatch gate
// (ggml_sycl_flash_attn_ext_xmx_q_supported) returns false in that build, so this kernel -- although
// still instantiated by the GLOB'd template-instance file -- is NEVER selected. This replaces the old
// hard `#error`, which broke every SYCL_FLASH_ATTN build that did not also define GGML_SYCL_F16.
#if defined(SYCL_FLASH_ATTN) && defined(GGML_SYCL_F16)
    static_assert(warp_size == 16, "XMX DPAS requires sub-group size 16 on Battlemage");
    static_assert(ncols2 == 1,     "v1: ncols2 must be 1 (GQA handled by the block->head mapping)");
    static_assert((DKQ == 128 || DKQ == 256) && DKQ == DV, "supported D in {128, 256}");
    static_assert(!use_logit_softcap, "v1: softcap gated off in the dispatch gate");

    using syclmx::joint_matrix;
    using syclmx::use;
    using syclmx::layout;
    using syclmx::joint_matrix_load;
    using syclmx::joint_matrix_store;
    using syclmx::joint_matrix_mad;
    using syclmx::joint_matrix_fill;

    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    sycl::sub_group sg = item.get_sub_group();

    constexpr int D   = DKQ;                 // == DV in v1
    constexpr int Bc  = XMX_NBATCH_FA;       // KV tile height
    constexpr int ncols  = ncols1 * ncols2;  // query columns per block
    constexpr int nwarps = ncols / XMX_MMA_M;
    static_assert(ncols % XMX_MMA_M == 0, "ncols must be a multiple of 16");
    static_assert(Bc % XMX_MMA_N == 0 && Bc % XMX_MMA_K == 0, "Bc must be a multiple of 16");
    static_assert(D  % XMX_MMA_K == 0 && D  % XMX_MMA_N == 0, "D must be a multiple of 16");

    constexpr int n_key_tiles = Bc / XMX_MMA_N;   // QK^T output N-tiles           (4)
    constexpr int n_kd_steps  = D  / XMX_MMA_K;   // QK^T contraction K-steps      (8)
    constexpr int n_dv_tiles  = DV / XMX_MMA_N;   // P*V   output N-tiles (over D) (8)
    constexpr int n_kc_steps  = Bc / XMX_MMA_K;   // P*V   contraction K-steps     (4)

    constexpr int ld_q  = D  + XMX_PAD_D;  // Q_slm  row stride (half)
    constexpr int ld_kv = D  + XMX_PAD_D;  // kv_slm row stride (half)
    constexpr int ld_s  = Bc + XMX_PAD_C;  // S_slm  row stride (float)
    constexpr int ld_p  = Bc + XMX_PAD_C;  // P_slm  row stride (half)
    constexpr int ld_o  = D  + XMX_PAD_D;  // O_slm  row stride (float)

    // ---- SLM carve. Distinct named buffers (kept separate for clarity; kv_slm is reused K->V within
    //      a tile, and S_slm could alias O_slm since they are live in disjoint phases -- see notes). ----
    syclex::work_group_static<sycl::half [ncols * ld_q ]> Q_slm;   // Q for the whole block (persistent)
    syclex::work_group_static<sycl::half [Bc    * ld_kv]> kv_slm;  // one K tile, then reused for V tile
    syclex::work_group_static<float      [ncols * ld_s ]> S_slm;   // QK^T scores (fp32, transient/tile)
    syclex::work_group_static<sycl::half [ncols * ld_p ]> P_slm;   // softmax probabilities (fp16, tile)
    syclex::work_group_static<float      [ncols * ld_o ]> O_slm;   // O rescale round-trip + write-back
    syclex::work_group_static<float      [ncols        ]> scale_row; // per-query-row exp(m_old-m_new)

    // work_group_static<T[N]> exposes the storage via operator T&() (array ref, decays to pointer);
    // there is no .get() member in this oneAPI (2026.0) -- take the implicit array->pointer decay.
    sycl::half * Q_s  = Q_slm;
    sycl::half * kv_s = kv_slm;
    float *      S_s  = S_slm;
    sycl::half * P_s  = P_slm;
    float *      O_s  = O_slm;
    float *      arow = scale_row;

    // ---- Block / head coordinates (mirrors flash_attn_tile:724-740) ----
    const int col_Q_0  = item.get_group(2) * ncols1;          // first query column of this block
    const int sequence = item.get_group(0) / (ne02 / ncols2); // v1 ncols2==1 -> group(0) / ne02
    const int head0    = item.get_group(0) * ncols2 - sequence * ne02;
    const int gqa_ratio = ne02 / ne12;

    const char * K_c = K + (int64_t) nb13 * sequence + (int64_t) nb12 * (head0 / gqa_ratio);
    const char * V_c = V + (int64_t) nb23 * sequence + (int64_t) nb22 * (head0 / gqa_ratio);
    const sycl::half * maskh = mask ? (const sycl::half *) (mask + nb33 * (sequence % ne33)) : nullptr;
    const int stride_mask = nb31 / sizeof(sycl::half);

    // alibi slope: max_bias==0 gate => 1.0; kept for parity with the scalar kernel.
    const float slope = get_alibi_slope(max_bias, head0, n_head_log2, m0, m1);

    const int lane = item.get_local_id(2);           // 0..15, == query row within this sub-group
    const int sgid = item.get_local_id(1);           // 0..nwarps-1, sub-group index == M-tile index
    const int nthreads = nwarps * warp_size;
    const int flat     = sgid * warp_size + lane;

    // ---- Persistent per-query-row running softmax state (register, one per lane's owned row) ----
    float m_i = -FLT_MAX / 2.0f;   // running max  (init NOT -inf, keeps m_old-m_new finite)
    float l_i = 0.0f;              // running denom
    const int   row_global = sgid * XMX_MMA_M + lane;             // this lane's query row in the block
    const int   q_col      = col_Q_0 + row_global;               // absolute query column
    const int   q_row_eff  = fastmodulo(q_col, ne01);           // wrapped for partial last tile (Q + mask)

    // ---- Persistent O accumulator: n_dv_tiles fp32 DPAS fragments (replaces the scalar VKQ[] regs) ----
    joint_matrix<sycl::sub_group, float, use::accumulator, XMX_MMA_M, XMX_MMA_N> O_frag[n_dv_tiles];
#pragma unroll
    for (int n = 0; n < n_dv_tiles; ++n) {
        joint_matrix_fill(sg, O_frag[n], 0.0f);
    }

    // ============================================================================================
    // Load Q for the whole block into Q_slm as scaled fp16. GGML_SYCL_F16 is ON so we do NOT apply
    // the fattn-tile.hpp:841-845 0.25 pre-scale hack (that exists only for the !GGML_SYCL_F16 path to
    // avoid fp16 KQ overflow; DPAS accumulates in fp32 so it is unnecessary and would be a 4x error).
    // ============================================================================================
    const float * Qf = (const float *) (Q + (int64_t) nb03 * sequence + (int64_t) nb02 * head0);
    const int q_row_stride = nb01 / sizeof(float);
#pragma unroll
    for (int e = flat; e < ncols * D; e += nthreads) {
        const int jc = e / D;                       // query column within block
        const int d  = e % D;
        const int qr = fastmodulo(col_Q_0 + jc, ne01);
        Q_s[jc * ld_q + d] = sycl::half(Qf[qr * q_row_stride + d] * scale);
    }
    item.barrier(sycl::access::fence_space::local_space);

    // ============================================================================================
    // Main loop over the KV cache. Parallel-blocks decomposition is identical to the scalar kernel:
    // block (group(1)) handles KV tiles starting at group(1)*Bc, strided by group_range(1)*Bc, and
    // (when parallel_blocks>1) writes an unnormalised O plus dst_meta = (m_i, l_i) for a later combine.
    // ============================================================================================
    const int k_VKQ_max = KV_max ? KV_max[sequence * item.get_group_range(2) + item.get_group(2)] : ne11;

    for (int k_VKQ_0 = item.get_group(1) * Bc; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += item.get_group_range(1) * Bc) {

        const int k_sup = k_VKQ_max - k_VKQ_0;               // valid keys in this tile (tail guard)
        const bool tail = k_sup < Bc;

        // ----- (1) cooperative K tile: q8_0 -> f16 into kv_slm[Bc][ld_kv] -----
        if (tail) {
            flash_attn_tile_xmx_load_kv_q8_0<warp_size, nwarps, Bc, D, ld_kv, /*oob_check=*/true>(
                K_c, (int64_t) nb11, k_VKQ_0, k_VKQ_max, kv_s);
        } else {
            flash_attn_tile_xmx_load_kv_q8_0<warp_size, nwarps, Bc, D, ld_kv, /*oob_check=*/false>(
                K_c, (int64_t) nb11, k_VKQ_0, k_VKQ_max, kv_s);
        }
        item.barrier(sycl::access::fence_space::local_space);

        // ----- (2) GEMM1  S = Q * K^T  (this sub-group's 16 query rows x Bc keys) -----
        // A = Q  use::a row_major   : Q_s[row][d]                    (M = query rows, K = d)
        // B = K  use::b col_major   : kv_s[key][d] read as B[d][key] (the QK^T transpose, for free)
        // C = S  use::accumulator fp32
        joint_matrix<sycl::sub_group, float, use::accumulator, XMX_MMA_M, XMX_MMA_N> S_frag[n_key_tiles];
#pragma unroll
        for (int n = 0; n < n_key_tiles; ++n) {
            joint_matrix_fill(sg, S_frag[n], 0.0f);
        }
#pragma unroll
        for (int kd = 0; kd < n_kd_steps; ++kd) {
            joint_matrix<sycl::sub_group, sycl::half, use::a, XMX_MMA_M, XMX_MMA_K, layout::row_major> Q_frag;
            // Q_frag = Q_s[(sgid*16)..][kd*16..], row-major, ld = ld_q
            joint_matrix_load(sg, Q_frag,
                sycl::local_ptr<const sycl::half>(&Q_s[(sgid * XMX_MMA_M) * ld_q + kd * XMX_MMA_K]), ld_q);
#pragma unroll
            for (int n = 0; n < n_key_tiles; ++n) {
                joint_matrix<sycl::sub_group, sycl::half, use::b, XMX_MMA_K, XMX_MMA_N, layout::col_major> K_frag;
                // col_major over kv_s[key][d]: element(row=d, col=key) = kv_s[(n*16+col)*ld_kv + (kd*16+row)]
                // == K[key][d]  -> exactly B[d][key], the transpose. base = &kv_s[(n*16)*ld_kv + kd*16].
                // BRINGUP (fragment-correctness gate #1): Intel DPAS wants the B operand VNNI/packed;
                // the implicit col_major->VNNI repack is not reliable across all icpx/DPC++ versions.
                // Verify a single 16x16 QK^T tile against the xmx_i4_verify.c B-packing oracle BEFORE
                // trusting this. If col_major B is unsupported/mis-lowers, transpose K into kv_s at
                // dequant time (so K and V share a natural row_major B load) or pre-VNNI-pack the K tile
                // and load with layout::ext_intel_packed. This is the single highest-leverage correctness gate.
                joint_matrix_load(sg, K_frag,
                    sycl::local_ptr<const sycl::half>(&kv_s[(n * XMX_MMA_N) * ld_kv + kd * XMX_MMA_K]), ld_kv);
                joint_matrix_mad(sg, S_frag[n], Q_frag, K_frag, S_frag[n]);
            }
        }
        // Store S (fp32) to S_slm[this sub-group's rows][Bc] for the scalar online-softmax.
#pragma unroll
        for (int n = 0; n < n_key_tiles; ++n) {
            joint_matrix_store(sg, S_frag[n],
                sycl::local_ptr<float>(&S_s[(sgid * XMX_MMA_M) * ld_s + n * XMX_MMA_N]), ld_s, layout::row_major);
        }
        // QK^T finished reading kv_slm; safe to overwrite kv_slm with V after this barrier.
        item.barrier(sycl::access::fence_space::local_space);

        // ----- (3) online softmax, per query row, entirely within lane `r`. -----
        // This is the scalar TILE softmax (fattn-tile.hpp:472-563) transplanted to read a canonical
        // row-major S_slm instead of the warp-distributed KQ_acc[] registers. Formulas are identical:
        //   mask add (+ softcap, gated off), running-max with +FATTN_KQ_MAX_OFFSET, exp, rowsum,
        //   alpha = exp(m_old - m_new), l = alpha*l + rowsum. No cross-lane reduction (one lane == one row).
        {
            float * Srow = &S_s[row_global * ld_s];
            sycl::half * Prow = &P_s[row_global * ld_p];

            const float m_old = m_i;
            float m_new = m_old;
#pragma unroll
            for (int c = 0; c < Bc; ++c) {
                float s;
                // Tail keys (index >= valid length) are forced out of the softmax FIRST, WITHOUT
                // reading K or the mask. This mirrors the scalar oracle's guard
                // `if (!oob_check || i_KQ < k_VKQ_sup)` (fattn-tile.hpp:492) and -- critically --
                // avoids reading maskh[k_VKQ_0 + c] past the GGML_KQ_MASK_PAD-padded mask width when
                // XMX_NBATCH_FA is swept to a value not aligned to the pad (e.g. Bc=96). With the
                // mask read here the score is otherwise identical; only the latent OOB is removed.
                if (tail && c >= k_sup) {
                    s = -INFINITY;
                } else {
                    s = Srow[c];
                    // causal / padding mask add (slope==1 in v1). Masked-out => mask holds -INF => s -> -INF.
                    if (mask) {
                        s += slope * (float) maskh[q_row_eff * stride_mask + k_VKQ_0 + c];
                    }
                }
                Srow[c] = s;
                m_new = sycl::fmax(m_new, s + FATTN_KQ_MAX_OFFSET);
            }

            const float alpha = sycl::native::exp(m_old - m_new);   // matches scalar (no explicit FTZ)
            float l_add = 0.0f;
#pragma unroll
            for (int c = 0; c < Bc; ++c) {
                const float p = (tail && c >= k_sup) ? 0.0f : sycl::native::exp(Srow[c] - m_new);
                l_add += p;
                Prow[c] = sycl::half(p);
            }
            l_i = l_i * alpha + l_add;
            m_i = m_new;
            arow[row_global] = alpha;
        }
        item.barrier(sycl::access::fence_space::local_space);

        // ----- (4) rescale the persistent O accumulator by alpha BEFORE adding this tile's P*V. -----
#ifdef GGML_SYCL_FA_XMX_O_INFRAGMENT
        // PRIMARY (in-fragment) path. Multiplies each accumulator element by the alpha of the query
        // row it belongs to, using the fragment's element->coordinate map. HARDWARE-VERIFY the (row,col)
        // orientation on B70 before trusting this -- a wrong axis silently scales the wrong rows.
        {
            const int row_base = sgid * XMX_MMA_M;
#pragma unroll
            for (int n = 0; n < n_dv_tiles; ++n) {
                auto wi = syclmx::get_wi_data(sg, O_frag[n]);
                for (int i = 0; i < wi.length(); ++i) {
                    auto coord = wi[i].get_coord();          // {row, col} within the 16x16 tile
                    const size_t r = coord[0];
                    wi[i] *= arow[row_base + r];
                }
            }
        }
#else
        // FALLBACK (SLM round-trip) path -- the SAFE DEFAULT. Store O to SLM, scale each row by its
        // alpha in scalar code (lane r owns row r), reload. Layout-agnostic and provably correct.
#pragma unroll
        for (int n = 0; n < n_dv_tiles; ++n) {
            joint_matrix_store(sg, O_frag[n],
                sycl::local_ptr<float>(&O_s[(sgid * XMX_MMA_M) * ld_o + n * XMX_MMA_N]), ld_o, layout::row_major);
        }
        item.barrier(sycl::access::fence_space::local_space);
        {
            const float a = arow[row_global];
            float * Orow = &O_s[row_global * ld_o];
#pragma unroll
            for (int d = 0; d < DV; ++d) {
                Orow[d] *= a;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int n = 0; n < n_dv_tiles; ++n) {
            joint_matrix_load(sg, O_frag[n],
                sycl::local_ptr<const float>(&O_s[(sgid * XMX_MMA_M) * ld_o + n * XMX_MMA_N]), ld_o, layout::row_major);
        }
#endif // GGML_SYCL_FA_XMX_O_INFRAGMENT

        // ----- (5) cooperative V tile: q8_0 -> f16 into kv_slm (reused). -----
        if (tail) {
            flash_attn_tile_xmx_load_kv_q8_0<warp_size, nwarps, Bc, D, ld_kv, /*oob_check=*/true>(
                V_c, (int64_t) nb21, k_VKQ_0, k_VKQ_max, kv_s);
        } else {
            flash_attn_tile_xmx_load_kv_q8_0<warp_size, nwarps, Bc, D, ld_kv, /*oob_check=*/false>(
                V_c, (int64_t) nb21, k_VKQ_0, k_VKQ_max, kv_s);
        }
        item.barrier(sycl::access::fence_space::local_space);

        // ----- (6) GEMM2  O += P * V  into the persistent fp32 O_frag. -----
        // A = P  use::a row_major : P_s[row][key]     (M = query rows, K = key)
        // B = V  use::b row_major : kv_s[key][d]      (natural, K = key, N = d) -- NOT transposed
        //   (this K/V asymmetry -- K col_major, V row_major -- is the classic flash-attn layout trap).
#pragma unroll
        for (int kc = 0; kc < n_kc_steps; ++kc) {
            joint_matrix<sycl::sub_group, sycl::half, use::a, XMX_MMA_M, XMX_MMA_K, layout::row_major> P_frag;
            joint_matrix_load(sg, P_frag,
                sycl::local_ptr<const sycl::half>(&P_s[(sgid * XMX_MMA_M) * ld_p + kc * XMX_MMA_K]), ld_p);
#pragma unroll
            for (int n = 0; n < n_dv_tiles; ++n) {
                joint_matrix<sycl::sub_group, sycl::half, use::b, XMX_MMA_K, XMX_MMA_N, layout::row_major> V_frag;
                // row_major over kv_s[key][d]: element(row=key, col=d) = kv_s[(kc*16+row)*ld_kv + (n*16+col)]
                joint_matrix_load(sg, V_frag,
                    sycl::local_ptr<const sycl::half>(&kv_s[(kc * XMX_MMA_K) * ld_kv + n * XMX_MMA_N]), ld_kv);
                joint_matrix_mad(sg, O_frag[n], P_frag, V_frag, O_frag[n]);
            }
        }
        // Barrier before the next iteration overwrites kv_slm / S_slm / P_slm.
        item.barrier(sycl::access::fence_space::local_space);
    }

    // ============================================================================================
    // Attention-sink adjustment. GATED OFF in v1 (gate requires sinks==nullptr) but kept for parity;
    // per-lane trivial, mirrors fattn-tile.hpp:972-1000. Only the first parallel block folds sinks.
    // ============================================================================================
    if (sinks && item.get_group(1) == 0) {
        const float sink = ((const float *) sinks)[head0];
        const float m_new = sycl::fmax(m_i, sink);
        const float a     = sycl::native::exp(m_i - m_new);
        l_i = l_i * a + sycl::native::exp(sink - m_new);
        m_i = m_new;
        // scale O by `a`: cheapest via the SLM round-trip (reuse O_s).
#pragma unroll
        for (int n = 0; n < n_dv_tiles; ++n) {
            joint_matrix_store(sg, O_frag[n],
                sycl::local_ptr<float>(&O_s[(sgid * XMX_MMA_M) * ld_o + n * XMX_MMA_N]), ld_o, layout::row_major);
        }
        item.barrier(sycl::access::fence_space::local_space);
        {
            float * Orow = &O_s[row_global * ld_o];
#pragma unroll
            for (int d = 0; d < DV; ++d) { Orow[d] *= a; }
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int n = 0; n < n_dv_tiles; ++n) {
            joint_matrix_load(sg, O_frag[n],
                sycl::local_ptr<const float>(&O_s[(sgid * XMX_MMA_M) * ld_o + n * XMX_MMA_N]), ld_o, layout::row_major);
        }
    }

    // ============================================================================================
    // Write back. Store O_frag -> O_slm, then each lane normalises its own row by 1/l_i and writes
    // DV floats to dst. When parallel_blocks>1 the normalisation is deferred to the combine kernel
    // (scale = 1) and we emit dst_meta = (m_i, l_i). Mirrors fattn-tile.hpp:1002-1057.
    // ============================================================================================
#pragma unroll
    for (int n = 0; n < n_dv_tiles; ++n) {
        joint_matrix_store(sg, O_frag[n],
            sycl::local_ptr<float>(&O_s[(sgid * XMX_MMA_M) * ld_o + n * XMX_MMA_N]), ld_o, layout::row_major);
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Partial-tile guard: discard rows that ran off the end of the real query length.
    if (ncols1 > 1 && q_col >= (int) ne01.z()) {
        return;
    }

    const float norm = item.get_group_range(1) == 1 ? 1.0f / l_i : 1.0f;
    const int j_dst_unrolled =
        ((sequence * (int) ne01.z() + q_col) * ne02 + head0) * item.get_group_range(1) + item.get_group(1);

    float * Orow = &O_s[row_global * ld_o];
#pragma unroll
    for (int d = 0; d < DV; ++d) {
        dst[(int64_t) j_dst_unrolled * DV + d] = Orow[d] * norm;
    }
    if (item.get_group_range(1) != 1) {
        dst_meta[j_dst_unrolled] = make_float2(m_i, l_i);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13,
        nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
#endif // SYCL_FLASH_ATTN && GGML_SYCL_F16
}

// ================================================================================================
// Launcher chain. This is a q8_0-in-VRAM clone of launch_fattn_tile_switch_ncols1 / _ncols2
// (fattn-tile.hpp:1071-1212). The ONLY two functional changes versus the scalar tile launcher are:
//   (A) launch_fattn is called with need_f16_K = need_f16_V = false, so the ~17 GiB full-tensor
//       to_fp16 transient (fattn-common.hpp:943-1003) is SKIPPED and K/V stay q8_0 in VRAM.
//   (B) warp_size is 16 (not WARP_32_SIZE), stamping reqd_sub_group_size(16) for DPAS.
// v1 fixes ncols2 == 1 (GQA is handled by the block->head mapping, not ncols2 packing).
// ================================================================================================
template <int DKQ, int DV, bool use_logit_softcap>
static void launch_fattn_tile_xmx_switch_ncols1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    constexpr int warp_size = 16;                 // DPAS sub-group size (NOT WARP_32_SIZE)
    constexpr int ncols2    = 1;
    constexpr size_t nbytes_shared = 0;
    constexpr int nbatch_fa = XMX_NBATCH_FA;

    // Choose the query tile (cols_per_block). Must be a multiple of 16. // BRINGUP: sweep 16/32/48/64.
    // Larger => more DPAS M-tiles per workgroup (better pipelining) but more SLM + registers.
    if (Q->ne[1] >= 32) {
        constexpr int cols_per_block = 32;
        const int nwarps = cols_per_block / warp_size;   // == cols_per_block / 16
        launch_fattn<DV, cols_per_block, ncols2,
            flash_attn_tile_xmx<DKQ, DV, cols_per_block, ncols2, use_logit_softcap, warp_size>, warp_size>(
            ctx, dst, nwarps, nbytes_shared, nbatch_fa, /*need_f16_K=*/false, /*need_f16_V=*/false, /*stream_k=*/false);
        return;
    }
    {
        // The gate requires Q->ne[1] >= FATTN_XMX_Q_MIN_COLS (== 32), so this arm is only a
        // belt-and-braces fallback for a 16-wide tile; kept for completeness.
        constexpr int cols_per_block = 16;
        const int nwarps = cols_per_block / warp_size;
        launch_fattn<DV, cols_per_block, ncols2,
            flash_attn_tile_xmx<DKQ, DV, cols_per_block, ncols2, use_logit_softcap, warp_size>, warp_size>(
            ctx, dst, nwarps, nbytes_shared, nbatch_fa, /*need_f16_K=*/false, /*need_f16_V=*/false, /*stream_k=*/false);
        return;
    }
}

template <int DKQ, int DV>
void ggml_sycl_flash_attn_ext_tile_xmx_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
    // v1 gate guarantees logit_softcap == 0; the template branch is kept to match the scalar structure.
    GGML_ASSERT(logit_softcap == 0.0f && "XMX-Q v1 does not support logit_softcap");
    constexpr bool use_logit_softcap = false;
    launch_fattn_tile_xmx_switch_ncols1<DKQ, DV, use_logit_softcap>(ctx, dst);
}

// Host entry point, dispatched from fattn.cpp. v1: only the D==128 case is instantiated.
void ggml_sycl_flash_attn_ext_tile_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#define DECL_FATTN_TILE_XMX_CASE(DKQ, DV)                              \
    template void ggml_sycl_flash_attn_ext_tile_xmx_case               \
    <DKQ, DV>(ggml_backend_sycl_context & ctx, ggml_tensor * dst)      \

extern DECL_FATTN_TILE_XMX_CASE(128, 128);
extern DECL_FATTN_TILE_XMX_CASE(256, 256);   // Qwen3.6-27B production head dim (D=256)

#endif // GGML_SYCL_FATTN_TILE_XMX_HPP
