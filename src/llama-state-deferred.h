#pragma once

// Deferred sequence-state capture (llama_state_seq_save_deferred, include/llama.h).
//
// A capture walks the memory module's range writer exactly as llama_state_seq_save_sink does, but splits
// what it writes in two:
//   - side-state and metadata (every plain write, and every tensor slice a module writes through
//     write_tensor) is copied into host memory at once, on the calling thread;
//   - positional K/V (write_tensor_positional: cells of an append-only cache) is only REFERENCED: the
//     tensor, offset and size, plus the (cache, sequence, position range) it depends on.
// The referenced bytes are copied later, in order, by llama_state_deferred_emit (into a sink, in chunks)
// or, when a cache is about to change those cells, by the flush-on-mutate hook below. The bytes produced
// are exactly those of llama_state_seq_save_sink at capture time: the format does not change.
//
// Every cache operation that can change a cell's data or free it (clear, seq_rm, a cross-stream seq_cp,
// seq_keep, seq_add, seq_div, a state load that replaces the sequence, the cache's destruction) calls
// llama_state_deferred_before_mutate first. It forces every capture whose references intersect the
// mutation: through the capture's flush callback when one is set (the caller then emits the bytes into
// its own sink), else into host memory owned by the capture. Only ggml_backend_tensor_get is used to
// read the device, so this works the same on every backend; on CPU the deferred copy is a memcpy.

#include "llama.h"
#include "llama-io.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct llama_context;

// a cache is about to change cells of seq_id (-1: of every sequence) at positions [p0, p1) (negative: open)
void llama_state_deferred_before_mutate(const void * owner, llama_seq_id seq_id, llama_pos p0, llama_pos p1);

// a cache is being destroyed: every capture referencing it takes its bytes into host memory (no callback)
void llama_state_deferred_before_free_owner(const void * owner);

// a context is being freed: every capture made on it is forced, callbacks included
void llama_state_deferred_before_free_ctx(const llama_context * ctx);

// a finished capture that still references cells joins the registry the hooks consult
void llama_state_deferred_register(llama_state_deferred * d);

struct llama_state_deferred {
    struct seg {
        size_t          size   = 0;
        const uint8_t * host   = nullptr; // bytes in host memory (immediate, or resolved)
        ggml_tensor *   tensor = nullptr; // otherwise: still in the cache
        size_t          t_off  = 0;
    };

    struct watch {
        const void * owner;
        llama_seq_id seq_id;
        llama_pos    pos_min;
        llama_pos    pos_max;
    };

    llama_context * ctx = nullptr;

    std::unique_ptr<uint8_t[]>              host;     // immediate bytes, one allocation (uninitialised)
    size_t                                  n_host = 0;
    std::vector<std::unique_ptr<uint8_t[]>> resolved; // referenced bytes the hook copied
    std::vector<seg>                        segs;
    std::vector<watch>                      watches;

    size_t n_total    = 0;
    size_t n_deferred = 0; // bytes captured as references
    size_t n_pending  = 0; // referenced bytes neither emitted nor resolved yet

    // emit cursor
    size_t i_seg     = 0;
    size_t seg_off   = 0;
    size_t n_emitted = 0;

    bool registered = false;
    bool failed     = false;

    llama_state_deferred_flush_cb flush_cb = nullptr;
    void *                        flush_ud = nullptr;

    uint32_t n_forced   = 0; // times the hook forced this capture
    uint32_t n_resolved = 0; // of which the bytes went into host memory owned by the capture

    // copies every referenced, not yet emitted byte into host memory owned by the capture; `sync` waits
    // for the context's backends first (not while the context is being torn down)
    void resolve(bool sync);
};

// Pass 1 (out == nullptr) only counts; pass 2 fills `out`, whose host buffer pass 1 sized. `defer` false
// copies positional bytes at once too (the capture is then an ordinary immediate one).
class llama_io_write_deferred : public llama_io_write_i {
public:
    llama_io_write_deferred(llama_state_deferred * out, bool defer) : out(out), defer(defer) {}

    void write(const void * src, size_t size) override;
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override;
    void write_tensor_positional(const void * owner, int32_t seq_id, int32_t pos_min, int32_t pos_max,
                                 ggml_tensor * tensor, size_t offset, size_t size) override;

    size_t n_bytes() override { return size_written; }

    size_t n_immediate = 0;
    size_t n_positional = 0;

private:
    uint8_t * host_reserve(size_t size);

    llama_state_deferred * out;
    const bool defer;
    size_t size_written = 0;
};
