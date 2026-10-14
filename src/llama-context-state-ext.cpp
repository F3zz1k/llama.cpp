// fork-only: see llama-context-state-ext.h.

#include "llama-context-state-ext.h"

#include "llama-context.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-state-deferred.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

// Streams into a caller-provided sink: every byte, plain or tensor, goes into memory the sink hands
// out, in pieces no larger than it offers, so a tensor slice is copied from the backend straight into
// its final staging place with no intermediate buffer (a tensor split across devices is the exception, see
// llama_state_tensor_read_whole).
class llama_io_write_sink : public llama_io_write_i {
public:
    llama_io_write_sink(const llama_state_sink & sink) : sink(sink) {}

    void write(const void * src, size_t size) override {
        const uint8_t * p = (const uint8_t *) src;
        while (size > 0) {
            size_t avail = 0;
            void * dst = reserve(size, avail);
            memcpy(dst, p, avail);
            sink.commit(sink.user_data, avail);
            p            += avail;
            size         -= avail;
            size_written += avail;
        }
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (llama_state_tensor_read_whole(tensor)) {
            // a split tensor is fetched whole and the host copy is cut at the sink's boundaries
            bounce.resize(size);
            ggml_backend_tensor_get(tensor, bounce.data(), offset, size);
            write(bounce.data(), size);
            return;
        }
        while (size > 0) {
            size_t avail = 0;
            void * dst = reserve(size, avail);
            ggml_backend_tensor_get(tensor, dst, offset, avail);
            sink.commit(sink.user_data, avail);
            offset       += avail;
            size         -= avail;
            size_written += avail;
        }
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    void * reserve(size_t n, size_t & avail) {
        avail = 0;
        void * dst = sink.reserve(sink.user_data, n, &avail);
        if (dst == nullptr || avail == 0) {
            throw std::runtime_error("the state sink refused the write");
        }
        avail = std::min(avail, n);
        return dst;
    }

    const llama_state_sink & sink;
    size_t size_written = 0;
    std::vector<uint8_t> bounce; // a split tensor's slice on its way to the sink
};

// Counts like llama_io_write_dummy and keeps the first n_head bytes of the stream, tensor data as zeros.
class llama_io_write_head : public llama_io_write_i {
public:
    llama_io_write_head(uint8_t * head, size_t n_head) : head(head), n_head(n_head) {
        if (head != nullptr && n_head > 0) {
            memset(head, 0, n_head);
        }
    }

    void write(const void * src, size_t size) override {
        if (head != nullptr && size_written < n_head) {
            memcpy(head + size_written, src, std::min(size, n_head - size_written));
        }
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * head;
    size_t    n_head;
    size_t    size_written = 0;
};

bool llama_context::kv_v_trans() const {
    return mem_v_trans;
}

uint32_t llama_context::kv_n_stream() const {
    // as every llama_kv_cache derives it (llama-kv-cache.cpp constructor)
    return cparams.kv_unified ? 1 : cparams.n_seq_max;
}

size_t llama_context::state_seq_save_sink(llama_seq_id seq_id, llama_pos p0, llama_pos p1, const llama_token * tokens, size_t n_token_count, const llama_state_sink & sink) {
    llama_io_write_sink io(sink);

    // the same bytes, in the same order, as state_seq_save_file / state_seq_save_file_range
    const uint32_t hdr[3] = { LLAMA_STATE_SEQ_MAGIC, LLAMA_STATE_SEQ_VERSION, (uint32_t) n_token_count };
    io.write(hdr, sizeof(hdr));
    io.write(tokens, sizeof(llama_token) * n_token_count);

    if (p0 < 0 && p1 < 0) {
        state_seq_write_data(io, seq_id, 0);
    } else if (memory) {
        memory->state_write_range(io, seq_id, p0, p1, 0);
    }

    return io.n_bytes();
}

void llama_context::state_seq_write_for_capture(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, const llama_token * tokens, size_t n_token_count) {
    // the same bytes, in the same order, as state_seq_save_sink
    const uint32_t hdr[3] = { LLAMA_STATE_SEQ_MAGIC, LLAMA_STATE_SEQ_VERSION, (uint32_t) n_token_count };
    io.write(hdr, sizeof(hdr));
    io.write(tokens, sizeof(llama_token) * n_token_count);

    if (p0 < 0 && p1 < 0) {
        state_seq_write_data(io, seq_id, 0);
    } else if (memory) {
        memory->state_write_range(io, seq_id, p0, p1, 0);
    }
}

size_t llama_context::state_seq_get_size_deferred(llama_seq_id seq_id, llama_pos p0, llama_pos p1, size_t n_token_count, size_t * n_deferred) {
    // the token values do not change any size: a zero array of the right length stands in for them
    std::vector<llama_token> toks(n_token_count, 0);
    llama_io_write_deferred io(nullptr, true);
    state_seq_write_for_capture(io, seq_id, p0, p1, toks.data(), toks.size());
    if (n_deferred) {
        *n_deferred = io.n_positional;
    }
    return io.n_bytes();
}

llama_state_deferred * llama_context::state_seq_save_deferred(llama_seq_id seq_id, llama_pos p0, llama_pos p1, const llama_token * tokens, size_t n_token_count) {
    // pass 1 sizes the host buffer for the immediate bytes, pass 2 copies them and records the references
    llama_io_write_deferred plan(nullptr, true);
    state_seq_write_for_capture(plan, seq_id, p0, p1, tokens, n_token_count);

    std::unique_ptr<llama_state_deferred> d(new llama_state_deferred());
    d->ctx    = this;
    d->n_host = plan.n_immediate;
    d->host.reset(new uint8_t[std::max<size_t>(1, plan.n_immediate)]);

    llama_io_write_deferred io(d.get(), true);
    state_seq_write_for_capture(io, seq_id, p0, p1, tokens, n_token_count);
    if (io.n_immediate != plan.n_immediate || io.n_bytes() != plan.n_bytes()) {
        throw std::runtime_error("deferred capture: the state changed between its two passes");
    }
    d->n_total = io.n_bytes();

    llama_state_deferred_register(d.get());
    return d.release();
}

size_t llama_context::state_seq_get_size_range(llama_seq_id seq_id, llama_pos p0, llama_pos p1, uint8_t * head, size_t n_head) {
    llama_io_write_head io(head, n_head);

    if (p0 < 0 && p1 < 0) {
        state_seq_write_data(io, seq_id, 0);
    } else if (memory) {
        memory->state_write_range(io, seq_id, p0, p1, 0);
    }

    return io.n_bytes();
}

bool llama_state_seq_flags_supported(const llama_memory_i * memory, llama_state_seq_flags flags, const char * func) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_SKIP_POSITIONAL) && (flags & LLAMA_STATE_SEQ_FLAGS_SKIP_SIDE)) {
        LLAMA_LOG_ERROR("%s: SKIP_POSITIONAL and SKIP_SIDE together would load nothing\n", func);
        return false;
    }
    if ((flags & LLAMA_STATE_SEQ_FLAGS_SKIP_SIDE) && (!memory || !memory->state_can_skip_side())) {
        // refused before anything is read, so the sequence is untouched
        LLAMA_LOG_ERROR("%s: this memory type cannot load the positional state alone\n", func);
        return false;
    }
    if ((flags & LLAMA_STATE_SEQ_FLAGS_SKIP_POSITIONAL) && (!memory || !memory->state_can_skip_positional())) {
        // refused before anything is read, so the sequence is untouched
        LLAMA_LOG_ERROR("%s: this memory type cannot load the side-state alone\n", func);
        return false;
    }
    return true;
}

//
// C API (include/llama-fork.h)
//

bool llama_kv_v_trans(const llama_context * ctx) {
    return ctx->kv_v_trans();
}

uint32_t llama_kv_n_stream(const llama_context * ctx) {
    return ctx->kv_n_stream();
}

bool llama_memory_can_skip_positional(llama_memory_t mem) {
    return mem != nullptr && mem->state_can_skip_positional();
}

bool llama_memory_can_skip_side(llama_memory_t mem) {
    return mem != nullptr && mem->state_can_skip_side();
}

size_t llama_state_seq_save_file_range(llama_context * ctx, const char * filepath, llama_seq_id seq_id, llama_pos p0, llama_pos p1, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file_range(seq_id, filepath, p0, p1, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_save_sink(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, const llama_token * tokens, size_t n_token_count, const llama_state_sink * sink) {
    if (sink == nullptr || sink->reserve == nullptr || sink->commit == nullptr) {
        return 0;
    }

    ctx->synchronize();

    try {
        return ctx->state_seq_save_sink(seq_id, p0, p1, tokens, n_token_count, *sink);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_get_size_range(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, uint8_t * head, size_t n_head) {
    ctx->synchronize();

    try {
        return ctx->state_seq_get_size_range(seq_id, p0, p1, head, n_head);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting sequence state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_get_size_deferred(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, size_t n_token_count, size_t * n_deferred) {
    try {
        return ctx->state_seq_get_size_deferred(seq_id, p0, p1, n_token_count, n_deferred);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error sizing a deferred sequence state capture: %s\n", __func__, err.what());
        return 0;
    }
}

llama_state_deferred * llama_state_seq_save_deferred(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_deferred(seq_id, p0, p1, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error capturing sequence state: %s\n", __func__, err.what());
        return nullptr;
    }
}

size_t llama_state_seq_load_file_ext(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_state_seq_flags flags, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}
