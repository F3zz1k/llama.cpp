#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // Tensor bytes of POSITIONAL cells: cells of an append-only cache (no sliding-window reuse) that hold
    // positions [pos_min, pos_max] of sequence seq_id in the cache `owner`. Nothing but a mutation of that
    // cache can change them (and every mutation calls llama_state_deferred_before_mutate first), so a
    // writer may copy them later (llama_io_write_deferred). Everything else a memory module writes is
    // side-state: a recurrent fold, a sliding window, a compressor or indexer blob, cell metadata. The
    // default copies now, so every other writer produces exactly what write_tensor would.
    virtual void write_tensor_positional(const void * owner, int32_t seq_id, int32_t pos_min, int32_t pos_max,
                                         ggml_tensor * tensor, size_t offset, size_t size) {
        (void) owner;
        (void) seq_id;
        (void) pos_min;
        (void) pos_max;
        write_tensor(tensor, offset, size);
    }

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // drop tensor data that has been read but not yet applied (e.g. when a restore fails)
    virtual void discard() {}

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
