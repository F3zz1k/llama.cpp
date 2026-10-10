// fork-only: see llama-kv-cache-state-ext.h.

#include "llama-kv-cache-state-ext.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-kv-cells.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

llama_kv_cache::~llama_kv_cache() {
    // a deferred capture still pointing at these tensors takes its bytes now (a cache viewing another
    // cache's cells owns no tensors: the source's destructor does this)
    if (!other) {
        llama_state_deferred_before_free_owner(deferred_owner());
    }
}

bool llama_kv_cache::shift_shares_cells(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    const auto & cells = v_cells[seq_to_stream[seq_id]];
    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && cells.pos_in(i, p0, p1) && cells.seq_has(i, seq_id) && cells.seq_count(i) > 1) {
            return true;
        }
    }
    return false;
}

bool llama_kv_cache::state_positional(llama_seq_id seq_id) const {
    // with a sliding window find_slot reuses masked cells of the same sequence, so the window is side-state
    return seq_id >= 0 && positional_ok && other == nullptr && swa_type == LLAMA_SWA_TYPE_NONE;
}

void llama_kv_cache::state_write_kv(llama_io_write_i & io, const cell_ranges_t & cr, ggml_tensor * t, size_t offset, size_t size) const {
    if (cr.positional) {
        io.write_tensor_positional(deferred_owner(), cr.seq_id, cr.pos_min, cr.pos_max, t, offset, size);
    } else {
        io.write_tensor(t, offset, size);
    }
}

void llama_kv_cache::state_skip(llama_io_read_i & io) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return; // a cache that shares another one's cells writes nothing
    }

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));
        if (cell_count == 0) {
            continue;
        }
        if (cell_count > v_cells[s].size()) {
            throw std::runtime_error("failed to skip kv cache state: more cells than the cache holds");
        }

        // cell metadata, as state_write_meta wrote it
        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;
            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));
            if (has_cell_ext()) {
                io.skip(sizeof(llama_kv_cell_ext));
            }
            if (n_seq_id > n_seq_max) {
                throw std::runtime_error("failed to skip kv cache state: invalid seq_id count");
            }
            io.skip((size_t) n_seq_id * sizeof(llama_seq_id));
        }

        // the data, as state_write_data wrote it; the shape checks mirror state_read_data
        uint32_t v_trans_ref;
        uint32_t n_layer_ref;
        io.read(&v_trans_ref, sizeof(v_trans_ref));
        io.read(&n_layer_ref, sizeof(n_layer_ref));
        if (n_layer_ref != layers.size() || (bool) v_trans_ref != v_trans) {
            throw std::runtime_error("failed to skip kv cache state: layer count or V layout mismatch");
        }
        uint32_t n_rot_k_ref;
        uint32_t n_rot_v_ref;
        io.read(&n_rot_k_ref, sizeof(n_rot_k_ref));
        io.read(&n_rot_v_ref, sizeof(n_rot_v_ref));
        if (n_rot_k_ref != n_rot_k || n_rot_v_ref != n_rot_v) {
            throw std::runtime_error("failed to skip kv cache state: key or value rotation mismatch");
        }

        for (const auto & layer : layers) {
            const auto * k = layer.k_stream[s];
            int32_t  type_ref;
            uint64_t row_ref;
            io.read(&type_ref, sizeof(type_ref));
            io.read(&row_ref,  sizeof(row_ref));
            if (type_ref != (int32_t) k->type || row_ref != ggml_row_size(k->type, hparams.n_embd_k_gqa(layer.il))) {
                throw std::runtime_error("failed to skip kv cache state: key type or row size mismatch");
            }
            io.skip((size_t) cell_count * row_ref);
        }

        for (const auto & layer : layers) {
            const auto * v = layer.v_stream[s];
            if (!v) {
                continue;
            }
            int32_t type_ref;
            io.read(&type_ref, sizeof(type_ref));
            if (type_ref != (int32_t) v->type) {
                throw std::runtime_error("failed to skip kv cache state: value type mismatch");
            }
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(layer.il);
            if (!v_trans) {
                uint64_t row_ref;
                io.read(&row_ref, sizeof(row_ref));
                if (row_ref != ggml_row_size(v->type, n_embd_v_gqa)) {
                    throw std::runtime_error("failed to skip kv cache state: value row size mismatch");
                }
                io.skip((size_t) cell_count * row_ref);
            } else {
                uint32_t el_ref;
                uint32_t n_embd_ref;
                io.read(&el_ref,     sizeof(el_ref));
                io.read(&n_embd_ref, sizeof(n_embd_ref));
                if (el_ref != ggml_type_size(v->type) || n_embd_ref != n_embd_v_gqa) {
                    throw std::runtime_error("failed to skip kv cache state: value element size mismatch");
                }
                io.skip((size_t) n_embd_v_gqa * cell_count * el_ref);
            }
        }
    }
}

bool llama_kv_cache_append_continues(const llama_kv_cells & cells, llama_seq_id dest_seq_id, const llama_ubatch & ubatch,
                                     uint32_t cell_count, llama_state_seq_flags flags, const char * func) {
    // an appended delta must lie strictly after what the sequence already holds; an overlap means the
    //   chain does not continue this base (the server checks chain contiguity, this makes the cache
    //   refuse it on its own). gaps are legal: M-RoPE media spends fewer positions than cells
    if ((flags & LLAMA_STATE_SEQ_FLAGS_NO_CLEAR) && cell_count > 0) {
        const llama_pos pos_max_base = cells.seq_pos_max(dest_seq_id);
        if (pos_max_base >= 0) {
            llama_pos pos_min_in = ubatch.pos[0];
            for (uint32_t i = 1; i < cell_count; ++i) {
                pos_min_in = std::min(pos_min_in, ubatch.pos[i]);
            }
            if (pos_min_in <= pos_max_base) {
                LLAMA_LOG_ERROR("%s: appended cells start at pos %d, the sequence already reaches %d\n",
                        func, (int) pos_min_in, (int) pos_max_base);
                return false;
            }
        }
    }
    return true;
}
