// fork-only: the disk-cache side of the memory modules other than llama_kv_cache itself (incremental
// state_write_range overrides, reading past a recurrent state, the SKIP_* loads of hybrid memory), kept out
// of their upstream files. Declared in each class's own header.

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <stdexcept>

void llama_kv_cache_iswa::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        // global (full-attention) layers are append-only: write only the [p0, p1) delta
        kv_base->state_write_range(io, seq_id, p0, p1, flags);
    }

    // sliding-window layers keep only the last W positions (a bounded, sliding buffer), so write the
    // whole current window like a recurrent state - a [p0, p1) delta would miss window cells before p0.
    kv_swa->state_write(io, seq_id, flags);
}

void llama_kv_cache_dsa::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) const {
    // both the MLA cache and the lightning-indexer key cache hold one append-only cell per position,
    // so each takes the same [p0, p1) filter and a delta composes onto its base with NO_CLEAR
    kv_mla->state_write_range(io, seq_id, p0, p1, flags);
    kv_lid->state_write_range(io, seq_id, p0, p1, flags);
}

void llama_kv_cache_dsa_iswa::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        // the DSA (full-attention + indexer) part is append-only: write only the [p0, p1) delta
        kv_dsa->state_write_range(io, seq_id, p0, p1, flags);
    }

    // the sliding-window cache is window-sized, so it is written whole at every node, exactly as in
    // llama_kv_cache_iswa: a [p0, p1) delta would miss the window cells before p0
    kv_swa->state_write(io, seq_id, flags);
}

void llama_kv_cache_msa::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) const {
    // the base cache and the indexer key cache are both full-size, one cell per position, so each
    // takes the same [p0, p1) filter and a delta composes onto its base with NO_CLEAR
    kv_base->state_write_range(io, seq_id, p0, p1, flags);
    kv_idx ->state_write_range(io, seq_id, p0, p1, flags);
}

void llama_kv_cache_dsv4::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) const {
    if (p1 >= 0 || seq_id < 0) {
        // a bounded [p0, p1) range or a whole-context save cannot be expressed: write whole
        state_write_impl(io, seq_id, -1, flags);
        return;
    }

    state_write_impl(io, seq_id, std::max<llama_pos>(p0, 0), flags);
}

void llama_memory_hybrid::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        // the attention KV is append-only: write only the [p0, p1) delta
        mem_attn->state_write_range(io, seq_id, p0, p1, flags);
    }
    // the recurrent state is a fixed-size fold of the whole prefix - always written whole
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid_iswa::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) const {
    // iswa attention: global layers as a [p0, p1) delta, sliding-window layers whole
    mem_attn->state_write_range(io, seq_id, p0, p1, flags);
    // recurrent state is a fixed-size fold of the whole prefix - always written whole
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid_idx::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) const {
    if (!mem_idx) {
        // no indexer section to lose, so the base class's attn-delta + whole-recurrent split is correct
        llama_memory_hybrid::state_write_range(io, seq_id, p0, p1, flags);

        return;
    }

    // [TAG_HYBRID_IDX_STATE] same section order as state_write: the attention delta and the whole
    // recurrent state (llama_memory_hybrid::state_write_range), then the indexer section last. The
    // indexer holds one key per attention cell, so it takes the same [p0, p1) filter. On restore the
    // indexer delta adopts the layout the attention delta was just given ([TAG_HYBRID_IDX_SINFO]),
    // which state_read_meta supports under NO_CLEAR, so base + deltas compose like a plain hybrid.
    llama_memory_hybrid::state_write_range(io, seq_id, p0, p1, flags);

    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_idx->state_write_range(io, seq_id, p0, p1, flags);
    }
}

bool llama_memory_hybrid::state_read_skip(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (flags & LLAMA_STATE_SEQ_FLAGS_SKIP_POSITIONAL) {
        // the attention part is positional: read past it and leave the sequence's own cells as they are
        GGML_ASSERT(seq_id >= 0 && "skipping the positional state needs a single sequence");
        GGML_ASSERT((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0 && "a partial state holds no positional part");
        mem_attn->state_skip(io);
        mem_recr->state_read(io, seq_id, flags & ~LLAMA_STATE_SEQ_FLAGS_SKIP_POSITIONAL);
        return true;
    }

    if (flags & LLAMA_STATE_SEQ_FLAGS_SKIP_SIDE) {
        // apply the attention part and read past the recurrent part: the sequence's recurrent state stays as it
        // was until a later load without this flag (the last node of a chain) replaces it
        GGML_ASSERT(seq_id >= 0 && "skipping the side-state needs a single sequence");
        GGML_ASSERT((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0 && "a partial state holds only side-state");
        const llama_state_seq_flags fa = flags & ~LLAMA_STATE_SEQ_FLAGS_SKIP_SIDE;
        mem_attn->state_read(io, seq_id, fa);
        try {
            mem_recr->state_skip(io);
        } catch (...) {
            mem_attn->state_clear(seq_id);
            throw;
        }
        return true;
    }

    return false;
}

void llama_memory_recurrent::state_skip(llama_io_read_i & io) const {
    uint32_t cell_count;
    io.read(&cell_count, sizeof(cell_count));
    if (cell_count > size) {
        throw std::runtime_error("failed to skip recurrent state: more cells than the cache holds");
    }

    // cell metadata, as state_write_meta wrote it for a single sequence (no seq ids)
    for (uint32_t i = 0; i < cell_count; ++i) {
        llama_pos pos;
        uint32_t  n_seq_id;
        io.read(&pos,      sizeof(pos));
        io.read(&n_seq_id, sizeof(n_seq_id));
        if (n_seq_id != 0) {
            throw std::runtime_error("failed to skip recurrent state: not a single-sequence state");
        }
    }

    // the data, as state_write_data wrote it; the shape checks mirror state_read_data
    uint32_t s_trans;
    uint32_t n_layer;
    io.read(&s_trans, sizeof(s_trans));
    io.read(&n_layer, sizeof(n_layer));
    if (n_layer != hparams.n_layer() || s_trans != 0) {
        throw std::runtime_error("failed to skip recurrent state: layer count or s layout mismatch");
    }

    for (uint32_t il = 0; il < n_layer; ++il) {
        if (r_l[il] == nullptr) {
            continue;
        }
        int32_t  type_ref;
        uint64_t row_ref;
        io.read(&type_ref, sizeof(type_ref));
        io.read(&row_ref,  sizeof(row_ref));
        if (type_ref != (int32_t) r_l[il]->type || row_ref != ggml_row_size(r_l[il]->type, hparams.n_embd_r())) {
            throw std::runtime_error("failed to skip recurrent state: r type or row size mismatch");
        }
        io.skip((size_t) cell_count * row_ref);

        if (p_l[il] != nullptr) {
            uint64_t p_row_ref;
            io.read(&p_row_ref, sizeof(p_row_ref));
            if (p_row_ref != ggml_row_size(p_l[il]->type, hparams.ple_conv_state())) {
                throw std::runtime_error("failed to skip recurrent state: ple row size mismatch");
            }
            io.skip((size_t) cell_count * p_row_ref);
        }
    }

    for (uint32_t il = 0; il < n_layer; ++il) {
        if (s_l[il] == nullptr) {
            continue;
        }
        int32_t  type_ref;
        uint64_t row_ref;
        io.read(&type_ref, sizeof(type_ref));
        io.read(&row_ref,  sizeof(row_ref));
        if (type_ref != (int32_t) s_l[il]->type || row_ref != ggml_row_size(s_l[il]->type, hparams.n_embd_s())) {
            throw std::runtime_error("failed to skip recurrent state: s type or row size mismatch");
        }
        io.skip((size_t) cell_count * row_ref);
    }
}
