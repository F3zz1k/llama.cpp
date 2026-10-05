#include "llama-state-deferred.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>

//
// registry of captures that still reference cache cells
//

static std::mutex                          g_mtx;
static std::vector<llama_state_deferred *> g_live;
static std::atomic<bool>                   g_hook_enabled{true};

static void registry_add(llama_state_deferred * d) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_live.push_back(d);
    d->registered = true;
}

static void registry_remove(llama_state_deferred * d) {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (!d->registered) {
        return;
    }
    g_live.erase(std::remove(g_live.begin(), g_live.end(), d), g_live.end());
    d->registered = false;
}

static bool registry_has(const llama_state_deferred * d) {
    std::lock_guard<std::mutex> lk(g_mtx);
    return std::find(g_live.begin(), g_live.end(), d) != g_live.end();
}

// the first live capture `match` selects, or nullptr
template <typename F>
static llama_state_deferred * registry_find(F match) {
    std::lock_guard<std::mutex> lk(g_mtx);
    for (auto * d : g_live) {
        if (match(d)) {
            return d;
        }
    }
    return nullptr;
}

// Forces a capture: its flush callback first (the owner emits the bytes into its own sink), then whatever
// is still referenced goes into host memory owned by the capture. On return the capture no longer
// references any cell. The callback may free the capture, so it is only touched again while registered.
static void force(llama_state_deferred * d, bool use_cb, bool sync) {
    d->n_forced++;
    if (use_cb && d->flush_cb) {
        d->flush_cb(d->flush_ud, d);
        if (!registry_has(d)) {
            return;
        }
    }
    d->resolve(sync);
}

void llama_state_deferred_before_mutate(const void * owner, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!g_hook_enabled.load(std::memory_order_relaxed)) {
        return;
    }
    const llama_pos lo = p0 < 0 ? 0 : p0;
    const llama_pos hi = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
    if (lo >= hi) {
        return;
    }
    while (auto * d = registry_find([&](const llama_state_deferred * x) {
                for (const auto & w : x->watches) {
                    if (w.owner == owner && (seq_id < 0 || w.seq_id == seq_id) && w.pos_max >= lo && w.pos_min < hi) {
                        return true;
                    }
                }
                return false;
            })) {
        force(d, /*use_cb=*/true, /*sync=*/true);
    }
}

void llama_state_deferred_before_free_owner(const void * owner) {
    while (auto * d = registry_find([&](const llama_state_deferred * x) {
                for (const auto & w : x->watches) {
                    if (w.owner == owner) {
                        return true;
                    }
                }
                return false;
            })) {
        force(d, /*use_cb=*/false, /*sync=*/false);
    }
}

void llama_state_deferred_before_free_ctx(const llama_context * ctx) {
    while (auto * d = registry_find([&](const llama_state_deferred * x) { return x->ctx == ctx; })) {
        force(d, /*use_cb=*/true, /*sync=*/false);
    }
}

void llama_state_deferred::resolve(bool sync) {
    if (n_pending > 0) {
        if (sync && ctx) {
            llama_synchronize(ctx);
        }
        for (size_t i = i_seg; i < segs.size(); ++i) {
            seg & s = segs[i];
            if (s.host != nullptr) {
                continue;
            }
            // a partly emitted segment keeps only its remainder: it is cut at the cursor, so exactly the
            // pending bytes are copied (and n_pending is what the capture's owner charges for them)
            if (i == i_seg && seg_off > 0) {
                s.t_off += seg_off;
                s.size  -= seg_off;
                seg_off  = 0;
            }
            std::unique_ptr<uint8_t[]> buf(new uint8_t[s.size]);
            ggml_backend_tensor_get(s.tensor, buf.get(), s.t_off, s.size);
            n_pending -= s.size;
            s.host   = buf.get();
            s.tensor = nullptr;
            resolved.push_back(std::move(buf));
        }
        n_resolved++;
    }
    GGML_ASSERT(n_pending == 0);
    registry_remove(this);
}

//
// capture writer
//

uint8_t * llama_io_write_deferred::host_reserve(size_t size) {
    if (out == nullptr) {
        n_immediate += size;
        return nullptr;
    }
    if (n_immediate + size > out->n_host) {
        throw std::runtime_error("deferred capture: the state grew between its two passes");
    }
    uint8_t * dst = out->host.get() + n_immediate;
    n_immediate += size;
    // extend the previous host segment when it ends right here
    if (!out->segs.empty() && out->segs.back().host != nullptr &&
            out->segs.back().host + out->segs.back().size == dst) {
        out->segs.back().size += size;
    } else {
        llama_state_deferred::seg s;
        s.size = size;
        s.host = dst;
        out->segs.push_back(s);
    }
    return dst;
}

void llama_io_write_deferred::write(const void * src, size_t size) {
    if (size == 0) {
        return;
    }
    uint8_t * dst = host_reserve(size);
    if (dst) {
        memcpy(dst, src, size);
    }
    size_written += size;
}

void llama_io_write_deferred::write_tensor(ggml_tensor * tensor, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    uint8_t * dst = host_reserve(size);
    if (dst) {
        ggml_backend_tensor_get(tensor, dst, offset, size);
    }
    size_written += size;
}

void llama_io_write_deferred::write_tensor_positional(const void * owner, int32_t seq_id, int32_t pos_min, int32_t pos_max,
                                                      ggml_tensor * tensor, size_t offset, size_t size) {
    if (!defer || seq_id < 0 || pos_min < 0) {
        write_tensor(tensor, offset, size);
        return;
    }
    if (size == 0) {
        return;
    }
    n_positional += size;
    size_written += size;
    if (out == nullptr) {
        return;
    }
    auto & segs = out->segs;
    if (!segs.empty() && segs.back().host == nullptr && segs.back().tensor == tensor &&
            segs.back().t_off + segs.back().size == offset) {
        segs.back().size += size;
    } else {
        llama_state_deferred::seg s;
        s.size   = size;
        s.tensor = tensor;
        s.t_off  = offset;
        segs.push_back(s);
    }
    out->n_deferred += size;
    out->n_pending  += size;
    for (auto & w : out->watches) {
        if (w.owner == owner && w.seq_id == seq_id) {
            w.pos_min = std::min(w.pos_min, (llama_pos) pos_min);
            w.pos_max = std::max(w.pos_max, (llama_pos) pos_max);
            return;
        }
    }
    out->watches.push_back({ owner, seq_id, pos_min, pos_max });
}

//
// API
//

void llama_state_deferred_register(llama_state_deferred * d) {
    if (d->n_pending > 0) {
        registry_add(d);
    }
}

size_t llama_state_deferred_size(const llama_state_deferred * d) {
    return d ? d->n_total : 0;
}

size_t llama_state_deferred_n_deferred(const llama_state_deferred * d) {
    return d ? d->n_deferred : 0;
}

size_t llama_state_deferred_n_pending(const llama_state_deferred * d) {
    return d ? d->n_pending : 0;
}

uint32_t llama_state_deferred_n_forced(const llama_state_deferred * d) {
    return d ? d->n_forced : 0;
}

void llama_state_deferred_set_flush_cb(llama_state_deferred * d, llama_state_deferred_flush_cb cb, void * user_data) {
    if (d) {
        d->flush_cb = cb;
        d->flush_ud = user_data;
    }
}

size_t llama_state_deferred_emit(llama_state_deferred * d, const llama_state_sink * sink, size_t max_bytes, bool * done) {
    if (done) {
        *done = false;
    }
    if (d == nullptr || sink == nullptr || sink->reserve == nullptr || sink->commit == nullptr) {
        return 0;
    }
    size_t n = 0;
    bool synced = false;
    while (d->i_seg < d->segs.size() && (max_bytes == 0 || n < max_bytes)) {
        auto & s = d->segs[d->i_seg];
        size_t want = s.size - d->seg_off;
        if (max_bytes != 0) {
            want = std::min(want, max_bytes - n);
        }
        size_t avail = 0;
        void * dst = sink->reserve(sink->user_data, want, &avail);
        if (dst == nullptr || avail == 0) {
            break; // the sink has no room now (or gave up): the caller decides which
        }
        avail = std::min(avail, want);
        if (s.host != nullptr) {
            memcpy(dst, s.host + d->seg_off, avail);
        } else {
            if (!synced && d->ctx) {
                // the cells were complete when captured; this only orders the read after queued work
                llama_synchronize(d->ctx);
                synced = true;
            }
            ggml_backend_tensor_get(s.tensor, dst, s.t_off + d->seg_off, avail);
            d->n_pending -= avail;
        }
        sink->commit(sink->user_data, avail);
        d->seg_off   += avail;
        d->n_emitted += avail;
        n            += avail;
        if (d->seg_off == s.size) {
            d->i_seg++;
            d->seg_off = 0;
        }
    }
    if (d->n_pending == 0) {
        registry_remove(d);
    }
    if (done) {
        *done = d->i_seg == d->segs.size();
    }
    return n;
}

void llama_state_deferred_free(llama_state_deferred * d) {
    if (d == nullptr) {
        return;
    }
    registry_remove(d);
    delete d;
}

void llama_state_deferred_set_hook_enabled(bool enabled) {
    g_hook_enabled.store(enabled, std::memory_order_relaxed);
}
