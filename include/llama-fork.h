// fork-only additions to the public llama API, kept out of llama.h so upstream merges do not touch them.
// Included at the end of llama.h; not meant to be included on its own.

#ifndef LLAMA_FORK_H
#define LLAMA_FORK_H

#ifndef LLAMA_H
#error "include llama.h instead"
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // fork: the attention-KV layout the context's memory was created with. A sequence state saved
    // under one layout is refused by a context with another (llama_kv_cache::state_read_data), so a
    // store of state files keyed on identity needs both. Neither depends on n_ctx.
    LLAMA_API bool     llama_kv_v_trans (const struct llama_context * ctx); // V rows stored transposed (Flash Attention off at memory creation)
    LLAMA_API uint32_t llama_kv_n_stream(const struct llama_context * ctx); // KV streams: 1 when unified, else n_seq_max

    // fork: the context length above which a LongRoPE model switches from its short to its long RoPE
    // factors (hparams.n_ctx_orig_yarn, the threshold llama_model::get_rope_factors compares
    // n_ctx_seq against; --override-kv applies, --yarn-orig-ctx does not). 0 when no layer's RoPE
    // factors depend on the context size, i.e. every layer has rope_freqs or rope_long == rope_short.
    LLAMA_API uint32_t llama_model_n_ctx_orig_longrope(const struct llama_model * model);

    // Whether a sequence load with LLAMA_STATE_SEQ_FLAGS_SKIP_POSITIONAL is supported: hybrid (attention
    // plus recurrent) and recurrent-only memory
    LLAMA_API bool llama_memory_can_skip_positional(llama_memory_t mem);

    // Whether a sequence load with LLAMA_STATE_SEQ_FLAGS_SKIP_SIDE is supported: hybrid (attention plus
    // recurrent) memory
    LLAMA_API bool llama_memory_can_skip_side(llama_memory_t mem);

// Do not clear the destination sequence before loading; append the loaded cells to the existing
// sequence. Used to compose a base snapshot with incremental deltas loaded in position order.
#define LLAMA_STATE_SEQ_FLAGS_NO_CLEAR 4

// Read a sequence state that holds everything, but apply only its side-state (recurrent state): the
// positional K/V of the append-only attention cache is read past, and the destination sequence keeps its
// own positional cells untouched. For a sequence that already holds the same tokens' attention, so a node
// further back restores without moving the attention back onto the device. Only memory types for which
// llama_memory_can_skip_positional() is true accept it; the others fail the load without touching the
// sequence.
#define LLAMA_STATE_SEQ_FLAGS_SKIP_POSITIONAL 8

// The complement of SKIP_POSITIONAL: apply only the positional K/V of the attention cache and read past the
// side-state (recurrent state), leaving the destination sequence's side-state as it was. For the inner nodes
// of a composed chain (a root, then deltas loaded with NO_CLEAR): each node carries the whole side-state at
// its own end, and only the last node's is wanted, so the earlier ones are read past instead of being moved
// onto the device and then overwritten. The last node must then be loaded without this flag. Only memory
// types for which llama_memory_can_skip_side() is true accept it; the others fail the load without touching
// the sequence. Cannot be combined with SKIP_POSITIONAL.
#define LLAMA_STATE_SEQ_FLAGS_SKIP_SIDE 16

    // Save only the cells whose position is in [p0, p1) of the sequence (half-open range).
    // p0 < 0 means from the start; p1 < 0 means to the end (both < 0 saves the whole sequence).
    // Bounded state (recurrent / sliding-window) is always written whole regardless of the range.
    // The resulting file has the same format as llama_state_seq_save_file and is loadable by
    // llama_state_seq_load_file[_ext]; compose a base + deltas by loading them in position order
    // with LLAMA_STATE_SEQ_FLAGS_NO_CLEAR.
    LLAMA_API size_t llama_state_seq_save_file_range(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1,
               const llama_token * tokens,
                          size_t   n_token_count);

    // Destination for llama_state_seq_save_sink. reserve() returns writable memory for at least 1 and at
    // most n bytes and stores its size in *n_avail, or returns NULL to abort the save; commit() reports
    // that the first n bytes of the last reservation hold data. Tensor data is copied from the backend
    // straight into the reserved memory, so the sink decides where the bytes land (a staging buffer, a
    // pinned ring, a file) and in what chunk sizes.
    typedef struct llama_state_sink {
        void * (*reserve)(void * user_data, size_t n, size_t * n_avail);
        void   (*commit) (void * user_data, size_t n);
        void * user_data;
    } llama_state_sink;

    // Produces exactly the bytes that llama_state_seq_save_file (p0 < 0 and p1 < 0) or
    // llama_state_seq_save_file_range ([p0, p1), either end open when negative) would write to a file,
    // header and token array included, but hands them to `sink` instead. Returns the number of bytes
    // produced, or 0 on failure (an exception, or a sink that returned NULL).
    LLAMA_API size_t llama_state_seq_save_sink(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1,
               const llama_token * tokens,
                          size_t   n_token_count,
          const llama_state_sink * sink);

    // Size in bytes of the memory payload llama_state_seq_save_sink produces for the same arguments
    // after its 12-byte header and token array, computed without copying any tensor data. When `head`
    // is not NULL, the first n_head bytes of that payload are copied into it, with tensor data reported
    // as zero bytes: only the leading plain fields (stream and cell counts) are meaningful there.
    // Returns 0 on failure.
    LLAMA_API size_t llama_state_seq_get_size_range(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1,
                         uint8_t * head,
                          size_t   n_head);

    // Deferred capture: the bytes llama_state_seq_save_sink would produce for the same arguments, with the
    // positional K/V (cells of an append-only cache, the part a range writer filters by position) left in
    // the cache and copied later. Side-state (recurrent folds, sliding windows, compressor and indexer
    // blobs, cell metadata) is copied at once. Any cache operation that could change a referenced cell
    // (clear, seq_rm, seq_keep, a cross-stream seq_cp, seq_add, seq_div, a state load replacing the
    // sequence, freeing the context) first forces the capture: through its flush callback when one is set,
    // else into host memory the capture owns. Only ggml_backend_tensor_get reads the device, on every
    // backend. The bytes are those of llama_state_seq_save_sink at capture time: the format is unchanged.
    typedef struct llama_state_deferred llama_state_deferred;

    // Called by the flush-on-mutate hook before a referenced cell changes; the callback should emit the
    // rest of the capture (llama_state_deferred_emit). Whatever it leaves referenced is then copied into
    // host memory owned by the capture. Runs on the thread that called the mutating function.
    typedef bool (*llama_state_deferred_flush_cb)(void * user_data, llama_state_deferred * d);

    // Total bytes (header and token array included) and, in *n_deferred, how many of them a deferred
    // capture would leave in the cache. No tensor data is copied. Returns 0 on failure.
    LLAMA_API size_t llama_state_seq_get_size_deferred(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1,
                          size_t   n_token_count,
                          size_t * n_deferred);

    // Returns NULL on failure. Free with llama_state_deferred_free (after the context is freed is fine).
    LLAMA_API llama_state_deferred * llama_state_seq_save_deferred(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1,
               const llama_token * tokens,
                          size_t   n_token_count);

    LLAMA_API size_t   llama_state_deferred_size      (const llama_state_deferred * d); // total bytes
    LLAMA_API size_t   llama_state_deferred_n_deferred(const llama_state_deferred * d); // bytes captured as references
    LLAMA_API size_t   llama_state_deferred_n_pending (const llama_state_deferred * d); // references not copied yet
    LLAMA_API uint32_t llama_state_deferred_n_forced  (const llama_state_deferred * d); // times the hook forced it

    LLAMA_API void llama_state_deferred_set_flush_cb(llama_state_deferred * d, llama_state_deferred_flush_cb cb, void * user_data);

    // Hands the next bytes, in order, to `sink`: at most max_bytes (0: no limit). Stops early when the
    // sink's reserve() returns NULL, which here means "no room now" rather than an error: a later call
    // continues where this one stopped. *done is set once every byte has been emitted. Returns the bytes
    // emitted by this call.
    LLAMA_API size_t llama_state_deferred_emit(
            llama_state_deferred * d,
          const llama_state_sink * sink,
                          size_t   max_bytes,
                            bool * done);

    LLAMA_API void llama_state_deferred_free(llama_state_deferred * d);

    // Testing only: with false the flush-on-mutate hook does nothing, so a mutation corrupts a pending
    // capture (the positive control for the hook's own tests). Default true.
    LLAMA_API void llama_state_deferred_set_hook_enabled(bool enabled);

    // Like llama_state_seq_load_file but takes state-seq flags (e.g. NO_CLEAR to append a delta).
    LLAMA_API size_t llama_state_seq_load_file_ext(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   dest_seq_id,
           llama_state_seq_flags   flags,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_FORK_H
