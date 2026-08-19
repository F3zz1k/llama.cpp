#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "gguf.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <exception>
#include <fstream>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <fstream>

#ifndef _WIN32
#include <unistd.h> // getpid() for per-writer-unique temp filenames (cross-process atomicity)
#else
#include <process.h> // _getpid()
#define getpid _getpid
#endif

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

struct server_batch {
    llama_batch batch;
    bool batch_rendered = false;

    struct token {
        int32_t id_slot;
        llama_token token;
        llama_pos pos;
        bool output;
        bool is_prompt; // for stats tracking
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // in embd mode, we temporarily swap out the tokens arr and restore it on clear()
    bool has_embd = false;
    llama_token * tokens_ptr = nullptr;
    std::vector<float> embd;

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    server_batch() {
        batch.pos = nullptr; // sentinel: uninitialized batch
    }

    ~server_batch() {
        if (batch.pos != nullptr) {
            clear();
            llama_batch_free(batch);
        }
    }

    void init(int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        batch = llama_batch_init(n_tokens_alloc, 0, 1);
        tokens_ptr = batch.token;
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(!has_embd); // cannot mix tokens + embd in same batch
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, pos, output, is_prompt });
        return true;
    }

    bool add(int32_t id_slot, const std::vector<float> & embd_in, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, pos, output, is_prompt });
        has_embd = true;
        embd.insert(embd.end(), embd_in.begin(), embd_in.end());
        return true;
    }

    void clear() {
        tokens.clear();
        embd.clear();
        common_batch_clear(batch);
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
        batch_rendered    = false;
        has_embd          = false;
        if (batch.token == nullptr) {
            batch.token = tokens_ptr;
            batch.embd  = nullptr;
        }
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void render() {
        GGML_ASSERT(!batch_rendered);
        GGML_ASSERT(batch.pos != nullptr);
        common_batch_clear(batch);
        for (int32_t i = 0; i < size(); i++) {
            const auto & t = tokens[i];
            common_batch_add(batch, t.token, t.pos, { t.id_slot }, t.output);
        }
        if (has_embd) {
            batch.token = nullptr; // will be restored on clear()
            batch.embd  = embd.data();
        }
        batch_rendered = true;
    }

    llama_batch get_view(int32_t off, int32_t n_tokens) const {
        GGML_ASSERT(batch.pos != nullptr);
        GGML_ASSERT(batch_rendered);
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        auto * token = batch.token ? batch.token + off          : nullptr;
        auto * embd  = batch.embd  ? batch.embd  + off * n_embd : nullptr;

        llama_batch view = {
            n_tokens,
            token,
            embd,
            batch.pos      + off,
            batch.n_seq_id + off,
            batch.seq_id   + off,
            batch.logits   + off,
        };

        return view;
    }
};

// --- KV restore-reuse: logits sidecar -------------------------------------------
// When a slot's full state is saved to disk (SLOT_SAVE) for a recurrent/hybrid (FULL) model,
// we additionally persist the last decoded token's full-vocab logits in a small sidecar file
// (<state>.logits). On SLOT_RESTORE of an exact-prompt "regenerate" request, those logits let
// the server emit the first token WITHOUT re-decoding into the (un-rewindable) restored
// recurrent state — which would otherwise crash. The sidecar is independent of libllama's
// state-file format (so that format is left untouched) and is purely best-effort: any
// missing/corrupt/vocab-mismatched sidecar degrades gracefully to the existing behavior.
static constexpr uint32_t SLOT_LOGITS_MAGIC   = 0x474C4B4Cu; // "LKLG" (llama kv logits), LE
static constexpr uint32_t SLOT_LOGITS_VERSION = 1u;

static std::string slot_logits_sidecar_path(const std::string & state_filepath) {
    return state_filepath + ".logits";
}

// Best-effort "touch": bump the mtime of an auto-cache snapshot's 3-file unit (state + .logits +
// .meta) to now, so a snapshot that is REUSED (read/restored) but never rewritten is treated as
// recently-used by the mtime LRU. Without this, the LRU is least-recently-WRITTEN, which would
// evict a hot base snapshot that N forked requests keep restoring from (it never gets rewritten).
// Never throws and never errors out the caller: every failure is swallowed via error_code (the
// file may have been concurrently evicted by another process; that is harmless here). This only
// runs on a successful restore, off the generation hot path.
static void auto_touch_unit(const std::string & state_filepath) {
    const auto now = std::filesystem::file_time_type::clock::now();
    std::error_code ec;
    std::filesystem::last_write_time(state_filepath, now, ec);
    std::filesystem::last_write_time(slot_logits_sidecar_path(state_filepath), now, ec);
    std::filesystem::last_write_time(state_filepath + ".meta", now, ec);
}

// Best-effort write of the logits sidecar. Returns the number of bytes written (0 on failure or
// when there is nothing valid to write). Never throws. The file is written to a temp path and
// atomically renamed so a partial/interrupted write can never leave a corrupt sidecar in place.
// Fields are serialized byte-by-byte little-endian (not a raw struct fwrite) for portability;
// the float payload is documented LE-only, matching llama.cpp's native-LE state-file contract.
static size_t slot_logits_write(const std::string & state_filepath,
                                const std::vector<float> & logits,
                                int32_t n_vocab,
                                uint32_t n_tokens) {
    if (logits.empty() || (int32_t) logits.size() != n_vocab || n_vocab <= 0) {
        return 0;
    }
    const std::string sidecar = slot_logits_sidecar_path(state_filepath);
    const std::string tmp     = sidecar + ".tmp";

    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) {
        return 0;
    }
    auto put_u32 = [&](uint32_t v) {
        const unsigned char b[4] = {
            (unsigned char)( v        & 0xFF),
            (unsigned char)((v >> 8)  & 0xFF),
            (unsigned char)((v >> 16) & 0xFF),
            (unsigned char)((v >> 24) & 0xFF),
        };
        f.write((const char *) b, 4);
    };
    put_u32(SLOT_LOGITS_MAGIC);
    put_u32(SLOT_LOGITS_VERSION);
    put_u32((uint32_t) n_vocab);
    put_u32(n_tokens);
    f.write((const char *) logits.data(), (std::streamsize) logits.size() * sizeof(float));
    f.flush();
    if (!f.good()) {
        f.close();
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        return 0;
    }
    f.close();
    std::error_code ec;
    std::filesystem::rename(tmp, sidecar, ec); // atomic replace
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return 0;
    }
    return 16 + logits.size() * sizeof(float);
}

// Read a logits sidecar. Returns true and fills `out` (size n_vocab) iff a valid sidecar exists
// whose vocab matches `expect_n_vocab` AND whose recorded token count matches `expect_n_tokens`
// (the count of the state just restored). The token-count check is AUTHORITATIVE: it binds the
// sidecar to the exact state it was saved against, so a sidecar that was somehow written for a
// different state length can never be reused. Any mismatch / short read / missing file => false
// with `out` cleared, so the caller falls back to existing behavior. Never throws.
static bool slot_logits_read(const std::string & state_filepath,
                             int32_t expect_n_vocab,
                             uint32_t expect_n_tokens,
                             std::vector<float> & out) {
    out.clear();
    if (expect_n_vocab <= 0) {
        return false;
    }
    const std::string sidecar = slot_logits_sidecar_path(state_filepath);
    std::ifstream f(sidecar, std::ios::binary);
    if (!f) {
        return false;
    }
    auto get_u32 = [&](uint32_t & v) -> bool {
        unsigned char b[4];
        f.read((char *) b, 4);
        if (f.gcount() != 4) {
            return false;
        }
        v = (uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24);
        return true;
    };
    uint32_t magic = 0, version = 0, n_vocab = 0, n_tokens = 0;
    if (!get_u32(magic) || !get_u32(version) || !get_u32(n_vocab) || !get_u32(n_tokens)) {
        return false;
    }
    if (magic != SLOT_LOGITS_MAGIC || version != SLOT_LOGITS_VERSION ||
        (int32_t) n_vocab != expect_n_vocab || n_tokens != expect_n_tokens) {
        return false;
    }
    out.resize(n_vocab);
    const std::streamsize want = (std::streamsize) n_vocab * (std::streamsize) sizeof(float);
    f.read((char *) out.data(), want);
    if (f.gcount() != want) {
        out.clear();
        return false;
    }
    return true;
}

// --- KV restore-reuse: bounded slot-save store ----------------------------------
// One slot-save snapshot = a state file plus its optional <name>.logits sidecar and (for auto-cache
// snapshots) its <name>.meta sidecar; all are always evicted together as a single unit, keyed by
// the state file's mtime (LRU).
struct slot_save_unit {
    std::string state_path;
    std::string sidecar_path; // "<state>.logits", "" if none
    std::string meta_path;    // "<state>.meta",   "" if none (auto disk cache)
    uintmax_t   bytes = 0;
    std::filesystem::file_time_type mtime;
    // Tree-aware eviction (U5): a checkpoint node's identity, derived entirely from disk. A node is
    // identified by the PAIR (`node_id`, `n_tokens`) — both parsed from the auto filename
    // "auto-<fp>-<chain_hash>-<n_tokens>.bin": `node_id` is the middle-hex chain-hash and `n_tokens` is
    // the trailing token count. This pair is what a child delta's parent link (`parent_id`, `range_lo`)
    // resolves to — exactly the pair U4 restore feeds to auto_state_filename(parent_id, range_lo). Keying
    // on the pair (not chain_hash alone) is essential: a continuation that does NOT cross a whole-block
    // boundary shares its parent's chain-hash, so node_id ALONE would make the child collide with — and
    // even self-reference — its parent. `node_id`/`n_tokens` are 0 for any file that is not an auto-*
    // snapshot. `parent_id`/`range_lo`/`range_hi` come from the .meta (a v3 delta has parent_id != 0; a v1
    // whole snapshot or a foreign/manual file is a parentless root). When every file is a v1 root the tree
    // logic collapses to flat mtime LRU.
    uint64_t    node_id   = 0;
    uint32_t    n_tokens  = 0;
    uint64_t    parent_id = 0;
    uint32_t    range_lo  = 0;
    uint32_t    range_hi  = 0;
    bool        is_node   = false; // true => v3 delta node (parent_id != 0)
    // PINNED (a sibling "<state>.pin" marker): never evicted and excluded from the count/byte caps,
    // BUT still carried in the unit set so it counts as a live child of its parent — a pinned delta
    // node's whole ancestor chain (each of which then has a live child) is protected from eviction,
    // and its own .pin is never deleted. A pinned v1 root has no parent/children, so it behaves
    // exactly as before: reserved, uncounted, unevictable.
    bool        pinned    = false;
};

// Read only the delta-node section of a state file's .meta (parent_id + [range_lo, range_hi)). Thin
// wrapper over slot_meta_read (defined further down, after model_fp) so slot_save_enforce_limits — which
// lives above the fingerprint/meta machinery — can resolve the checkpoint tree without pulling model_fp
// into scope. Returns false (and leaves the outputs as a parentless root) for a v1/foreign/absent meta.
static bool slot_node_meta_probe(const std::string & state_filepath,
                                 uint64_t & parent_id, uint32_t & range_lo, uint32_t & range_hi);

// Parse a node's identity PAIR (chain_hash, n_tokens) from an auto-cache state filename
// ("auto-<fp_model>-<chain_hash>-<n_tokens>.bin"): the middle 16-hex group is the chain-hash a child
// delta stores as its parent_id, and the trailing decimal group is the token count a child delta stores
// as its parent link's range_lo. The pair is the node's unique key on disk. Returns false (and leaves
// both outputs 0) for any name that is not an auto snapshot — a foreign/manual file is then treated as
// its own parentless root and always ages by plain mtime. n_tokens is clamped to uint32_t; the save side
// bounds token counts far below 2^32 (slot_meta_read rejects tok_count > 2^28).
static bool slot_save_parse_node_id(const std::string & state_path, uint64_t & node_id, uint32_t & n_tokens) {
    node_id  = 0;
    n_tokens = 0;
    const size_t slash = state_path.find_last_of("/\\");
    const std::string name = (slash == std::string::npos) ? state_path : state_path.substr(slash + 1);
    // layout: "auto-" (5) + 16 hex fp + "-" + 16 hex chain_hash + "-" + digits + ".bin"
    static const char pfx[] = "auto-";
    const size_t pfx_len = 5, hex_len = 16;
    if (name.size() < pfx_len + hex_len + 1 + hex_len + 1 ||
        name.compare(0, pfx_len, pfx) != 0 ||
        name[pfx_len + hex_len] != '-' ||
        name[pfx_len + hex_len + 1 + hex_len] != '-') {
        return false;
    }
    uint64_t id = 0;
    for (size_t k = pfx_len + hex_len + 1; k < pfx_len + hex_len + 1 + hex_len; ++k) {
        const char c = name[k];
        uint64_t d;
        if      (c >= '0' && c <= '9') { d = (uint64_t) (c - '0'); }
        else if (c >= 'a' && c <= 'f') { d = (uint64_t) (c - 'a' + 10); }
        else if (c >= 'A' && c <= 'F') { d = (uint64_t) (c - 'A' + 10); }
        else { return false; }
        id = (id << 4) | d;
    }
    // trailing decimal token count, terminated by '.' (the ".bin" extension). At least one digit.
    size_t k = pfx_len + hex_len + 1 + hex_len + 1;
    if (k >= name.size() || name[k] < '0' || name[k] > '9') {
        return false;
    }
    uint64_t nt = 0;
    for (; k < name.size() && name[k] >= '0' && name[k] <= '9'; ++k) {
        nt = nt * 10 + (uint64_t) (name[k] - '0');
        if (nt > 0xffffffffull) {
            return false; // token count out of range for a real snapshot
        }
    }
    node_id  = id;
    n_tokens = (uint32_t) nt;
    return true;
}

// Enforce --slot-save-max-count / --slot-save-max-bytes over `dir` using LRU-by-mtime eviction.
// `just_written` is the state path that was just saved: it is never evicted, but if it ALONE
// exceeds the byte cap it is deleted (with its sidecar) and `oversized` is set so the caller can
// reject the save rather than evict everything else. Operates strictly within `dir`; uses only
// the error_code std::filesystem overloads so it never throws across the server loop.
//
// IMPORTANT: this runs ONLY for the opt-in --slot-save-auto cache (callers gate on
// auto_cache_enabled()); a plain manual --slot-save-path save never reaches it and never deletes
// anything. When a cap is set on the auto cache, --slot-save-path is treated as a server-owned
// store — any regular file in it (other than recognized "<X>.logits" sidecars and "*.tmp"
// temporaries) is an eviction candidate. Point --slot-save-max-count/-mb at a DEDICATED directory;
// do not mix unrelated files into the auto-cache directory. (With no caps set — the default —
// nothing is ever deleted and the directory is left exactly as before.)
// `just_written` is the exact filepath string the server built as `slot_save_path + filename`;
// directory_iterator(dir) over that same `slot_save_path` yields identically-spelled path strings
// on POSIX (the production target), so raw string equality correctly identifies the just-saved
// unit. (Not used on Windows in practice; if ever needed there, switch to filename comparison.)
static void slot_save_enforce_limits(const std::string & dir,
                                     int32_t max_count, int64_t max_bytes,
                                     const std::string & just_written,
                                     bool & oversized) {
    oversized = false;
    if (max_count <= 0 && max_bytes <= 0) {
        return; // both unlimited
    }

    std::error_code ec;
    std::vector<slot_save_unit> units;
    uintmax_t this_unit_bytes = 0;

    // First pass: enumerate every regular file once and record the full set of paths so we can
    // tell a real sidecar (sibling of a state file we wrote) from a state file a client happened
    // to name "foo.logits". We must NOT blindly skip every "*.logits" — fs_validate_filename
    // allows that suffix, so a state file literally named "foo.logits" would otherwise escape both
    // caps entirely. Only "<X>.logits" where "<X>" also exists is treated as a sidecar.
    std::vector<std::string> all_files;
    {
        std::set<std::string> present;
        for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code fec;
            if (!it->is_regular_file(fec) || fec) {
                continue;
            }
            all_files.push_back(it->path().string());
            present.insert(all_files.back());
        }

        for (const std::string & p : all_files) {
            std::error_code fec;
            // in-flight temp files are never counted or evicted (a concurrent save owns them). A
            // save streams its state to "<fname>.<pid>.<nonce>.tmp", then renames the ".logits"/
            // ".meta" sidecar temps to their final names AFTER renaming that state temp to <fname>;
            // in that publish window the sidecar temps ("…tmp.logits"/"…tmp.meta") have no base
            // file, so they must be matched here rather than reaped as orphaned sidecars below.
            if ((p.size() >= 4  && p.compare(p.size() - 4,  4,  ".tmp")        == 0) ||
                (p.size() >= 11 && p.compare(p.size() - 11, 11, ".tmp.logits") == 0) ||
                (p.size() >= 9  && p.compare(p.size() - 9,  9,  ".tmp.meta")   == 0)) {
                continue;
            }
            // a "<X>.logits" file is a sidecar ONLY when its state file "<X>" is also present;
            // accounted together with that state file below, so skip it here.
            if (p.size() >= 7 && p.compare(p.size() - 7, 7, ".logits") == 0 &&
                present.count(p.substr(0, p.size() - 7))) {
                continue;
            }
            // a "<X>.meta" file is the auto disk cache's tokens+fingerprint sidecar; treat it
            // exactly like ".logits" — accounted with its state file below, reaped if orphaned.
            if (p.size() >= 5 && p.compare(p.size() - 5, 5, ".meta") == 0 &&
                present.count(p.substr(0, p.size() - 5))) {
                continue;
            }
            // a "<X>.pin" file marks "<X>" as PINNED (never evicted, excluded from caps). Like the
            // other sidecars: skip it here when its state file is present; reaped below if orphaned.
            if (p.size() >= 4 && p.compare(p.size() - 4, 4, ".pin") == 0 &&
                present.count(p.substr(0, p.size() - 4))) {
                continue;
            }
            // reap an ORPHANED sidecar (its state file was evicted/lost): otherwise these silently
            // accumulate (we never count them) and eat real on-disk space forever.
            if (p.size() >= 7 && p.compare(p.size() - 7, 7, ".logits") == 0 &&
                !present.count(p.substr(0, p.size() - 7))) {
                std::filesystem::remove(p, fec);
                continue;
            }
            if (p.size() >= 5 && p.compare(p.size() - 5, 5, ".meta") == 0 &&
                !present.count(p.substr(0, p.size() - 5))) {
                std::filesystem::remove(p, fec);
                continue;
            }
            if (p.size() >= 4 && p.compare(p.size() - 4, 4, ".pin") == 0 &&
                !present.count(p.substr(0, p.size() - 4))) {
                std::filesystem::remove(p, fec);
                continue;
            }

            slot_save_unit u;
            u.state_path = p;
            // PINNED snapshots (a sibling "<state>.pin" marker) are never evicted and are excluded
            // from the count/byte caps entirely — a reserved, persistent entry (e.g. a permanent
            // doc / system-prompt prefix) that coexists with the normal LRU pool. Pin with
            // `touch <snapshot>.pin`; unpin by removing it. The index/restore path is unchanged:
            // a pinned snapshot is a normal auto-*.bin, still discovered and restored like any other.
            // We DON'T skip it here (as the flat-LRU version did): it is carried into the unit set so
            // the tree refcount below counts it as a live child of its parent, which protects a pinned
            // v3 delta node's whole ancestor chain from eviction (a base whose only child is pinned
            // must not be reaped). It is then excluded from the caps and never picked for eviction.
            u.pinned = (present.count(p + ".pin") > 0);
            u.bytes = std::filesystem::file_size(p, fec);
            if (fec) {
                continue;
            }
            const std::string side = p + ".logits";
            if (present.count(side)) {
                const auto sb = std::filesystem::file_size(side, fec);
                if (!fec) {
                    u.sidecar_path = side;
                    u.bytes += sb;
                }
            }
            const std::string meta = p + ".meta";
            if (present.count(meta)) {
                const auto mb = std::filesystem::file_size(meta, fec);
                if (!fec) {
                    u.meta_path = meta;
                    u.bytes += mb;
                }
            }
            u.mtime = std::filesystem::last_write_time(p, fec);
            if (fec) {
                continue;
            }

            // Tree identity for U5 eviction: the node's own key (chain_hash, n_tokens) from the filename,
            // its parent link + range from the .meta. A file with no delta meta (v1 whole snapshot or a
            // foreign file) stays a parentless root (parent_id 0), so the eviction below reduces to today's
            // flat mtime LRU for it.
            slot_save_parse_node_id(p, u.node_id, u.n_tokens);
            slot_node_meta_probe(p, u.parent_id, u.range_lo, u.range_hi);
            u.is_node = (u.parent_id != 0);

            if (p == just_written) {
                this_unit_bytes = u.bytes;
            }
            units.push_back(std::move(u));
        }
    }

    // a single snapshot larger than the byte cap is rejected: delete only the just-written unit,
    // do NOT cascade-evict every other (valid) snapshot to make room for something that can't fit.
    // NOTE: intentionally a no-op when max_bytes == 0 (byte cap disabled); in count-only mode an
    // individual snapshot's size is never bounded — only --slot-save-max-mb bounds per-snapshot size.
    if (max_bytes > 0 && this_unit_bytes > (uintmax_t) max_bytes) {
        for (const auto & u : units) {
            if (u.state_path == just_written) {
                std::filesystem::remove(u.state_path, ec);
                if (!u.sidecar_path.empty()) {
                    std::filesystem::remove(u.sidecar_path, ec);
                }
                if (!u.meta_path.empty()) {
                    std::filesystem::remove(u.meta_path, ec);
                }
                break;
            }
        }
        oversized = true;
        return;
    }

    // ---- Tree-aware eviction (U5) --------------------------------------------------------------------
    // The auto cache is a FOREST of checkpoint trees, reconstructed here purely from what is on disk (no
    // in-RAM tree map). A v1 whole snapshot — or any foreign/manual file — is a parentless root; a v3
    // delta node points at the parent whose KV prefix it extends (parent_id == the parent file's
    // chain-hash == its filename's middle hex). Two rules make eviction tree-correct:
    //   (a) NEVER evict a node that still has a child on disk — its delta .bin is meaningless without its
    //       base — so we only ever evict LEAVES (child_count[node_id] == 0), oldest leaf first, which
    //       peels a lineage tip-to-root.
    //   (b) Age every candidate by its OWN mtime. A restore touches EVERY node on the chain it used
    //       (see auto_touch_unit at the restore site), so a live lineage's shared base already has a
    //       fresh mtime of its own and is additionally un-evictable by (a) while any child survives.
    //       An untouched sibling therefore ages out on its own, which is the whole point.
    //
    // NOTE: this deliberately REPLACES a previous `tree_recency` rule that aged whole TREES by their
    // most-recent node. That rule was redundant with (a) + chain-propagating touch for its stated goal
    // ("a hot lineage keeps its cold shared base"), and it actively broke the fan-out case it was
    // supposed to help: with N subagent forks off one shared prefix, a single active fork refreshed
    // the whole tree's recency and made all N-1 dead forks immortal, so eviction pressure fell on
    // OTHER trees instead — a wide fan-out could evict the entire rest of the store. It also let a
    // single synthetic warm-on-spawn restore (which touches the chain) immunise a lineage that had
    // seen no real traffic for days. Measured on the live store: 21 sibling tips off one root, median
    // leaf age 82.5 h, shielded by a tree_recency of 2.0 h that came from the boot-time warm read.
    // Recomputed from disk each pass => cross-process correct.

    auto remove_unit_files = [&](const slot_save_unit & u) {
        std::filesystem::remove(u.state_path, ec);
        if (!u.sidecar_path.empty()) {
            std::filesystem::remove(u.sidecar_path, ec);
        }
        if (!u.meta_path.empty()) {
            std::filesystem::remove(u.meta_path, ec);
        }
    };

    // A node's on-disk identity is the PAIR (chain_hash, n_tokens): a continuation that does not cross a
    // whole-block boundary shares its parent's chain_hash, so chain_hash ALONE would make the child
    // collide with — indeed parent_id == node_id, self-reference — its parent, pinning the true tip as
    // unevictable and defeating the caps. n_tokens (from the filename) disambiguates. The parent link is
    // the pair (parent_id, range_lo) — exactly the pair U4 restore feeds to auto_state_filename.
    using node_key = std::pair<uint64_t, uint32_t>; // (chain_hash, n_tokens)
    const auto self_key   = [](const slot_save_unit & u) -> node_key { return { u.node_id,   u.n_tokens }; };
    const auto parent_key = [](const slot_save_unit & u) -> node_key { return { u.parent_id, u.range_lo  }; };

    // (chain_hash, n_tokens) -> index, for parent-link resolution. Auto filenames are unique per pair; a
    // non-auto file has node_id 0 and is never a parent target.
    std::map<node_key, size_t> node_by_key;
    for (size_t i = 0; i < units.size(); ++i) {
        if (units[i].node_id != 0) {
            node_by_key[self_key(units[i])] = i;
        }
    }

    std::vector<char> alive(units.size(), 1);

    // Reap orphan deltas up front: a delta node whose base file is gone can never be restored (a base-less
    // delta .bin would corrupt a compose-load), so it is dead weight — delete it regardless of the caps.
    // Evicting one orphan can orphan its own children, so iterate to a fixed point. just_written is never
    // touched (a freshly saved node had its parent verified present at save time).
    for (bool changed = true; changed; ) {
        changed = false;
        for (size_t i = 0; i < units.size(); ++i) {
            // never reap a pinned node's files (its .pin is authoritative "keep"); it stays alive so
            // its own children are not treated as orphaned either.
            if (!alive[i] || units[i].parent_id == 0 || units[i].pinned ||
                units[i].state_path == just_written) {
                continue;
            }
            const auto it = node_by_key.find(parent_key(units[i]));
            if (it == node_by_key.end() || !alive[it->second]) {
                remove_unit_files(units[i]);
                alive[i] = 0;
                if (units[i].node_id != 0) {
                    const auto self = node_by_key.find(self_key(units[i]));
                    if (self != node_by_key.end() && self->second == i) {
                        node_by_key.erase(self);
                    }
                }
                changed = true;
            }
        }
    }

    // child_count[(chain_hash, n_tokens)] = live children of that node (a node is a LEAF iff
    // child_count[self_key] == 0).
    std::map<node_key, size_t> child_count;
    for (size_t i = 0; i < units.size(); ++i) {
        if (alive[i] && units[i].parent_id != 0) {
            child_count[parent_key(units[i])]++;
        }
    }

    // Pinned units are excluded from the caps entirely — they occupy the tree only so their ancestors
    // stay refcount-protected — matching the flat-LRU pin semantics (a pinned unit was uncounted there).
    size_t    count = 0;
    uintmax_t total = 0;
    for (size_t i = 0; i < units.size(); ++i) {
        if (alive[i] && !units[i].pinned) {
            count++;
            total += units[i].bytes;
        }
    }

    // Pick the least-recently-used evictable leaf, by its OWN mtime. Returns units.size() when nothing
    // is evictable (every remaining node has a live child, or all that is left is just_written).
    auto pick_leaf = [&]() -> size_t {
        size_t best = units.size();
        for (size_t i = 0; i < units.size(); ++i) {
            if (!alive[i] || units[i].pinned || units[i].state_path == just_written) {
                continue; // pinned units are never evicted
            }
            if (units[i].node_id != 0) {
                const auto cc = child_count.find(self_key(units[i]));
                if (cc != child_count.end() && cc->second > 0) {
                    continue; // not a leaf: a delta still depends on it
                }
            }
            if (best == units.size()) {
                best = i;
                continue;
            }
            if (units[i].mtime < units[best].mtime) {
                best = i;
            }
        }
        return best;
    };

    auto evict_leaf = [&]() -> bool {
        const size_t i = pick_leaf();
        if (i == units.size()) {
            return false;
        }
        remove_unit_files(units[i]);
        alive[i] = 0;
        total -= std::min(total, (uintmax_t) units[i].bytes);
        count = (count > 0) ? count - 1 : 0;
        if (units[i].parent_id != 0) { // evicting a leaf may expose its parent as a new leaf
            const auto it = child_count.find(parent_key(units[i]));
            if (it != child_count.end() && it->second > 0) {
                it->second--;
            }
        }
        return true;
    };

    if (max_count > 0) {
        while (count > (size_t) max_count) {
            if (!evict_leaf()) {
                SRV_WRN("%s", "slot-save cache is over --slot-save-max-count but every remaining snapshot "
                              "has a live child delta; leaving it above the limit\n");
                break;
            }
        }
    }
    if (max_bytes > 0) {
        while (total > (uintmax_t) max_bytes) {
            if (!evict_leaf()) {
                SRV_WRN("%s", "slot-save cache is over --slot-save-max-bytes but every remaining snapshot "
                              "has a live child delta; leaving it above the limit\n");
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// --- Auto disk prompt/KV cache (opt-in: --slot-save-auto) ---
//
// Persists per-slot KV snapshots to disk (state file + '.meta' [tokens + fingerprint]
// + '.logits' sidecar) and indexes them by a chained hash over token IDs, so a cold
// process can reuse a warm process's KV with no client/router involvement.
//
// Design invariants (all must hold; comments below reference them by number):
//   1. Off by default: every hook's FIRST statement is auto_cache_enabled(); when
//      false there is no scan, index, hashing, or allocation — behavior is unchanged.
//   2. Never restore on hash alone: the snapshot's token-ID array must byte-compare
//      equal to the request prefix before any restore (collision-safe).
//   3. Model identity: each snapshot carries a fingerprint (model/vocab/ctx/rope/
//      KV-type/FULL-vs-attention/LoRA); a mismatch refuses the restore.
//   4. Fallback totality: any failure (corrupt file, fp/vocab mismatch, IO error,
//      no match) falls back to a normal prefill — never crash, never wrong output.
//   5. Hot-path purity: the multi-GB save runs only on slot release/reassign, never
//      during generation; restore happens once before prefill.
//
// Concurrency: all slot work runs on the single server-loop thread, so the index is
// single-threaded and the mutex below is uncontended today; it becomes load-bearing
// only if the save I/O is later moved to a worker thread (do not make save async
// without keeping the mutex honest). Independent of legacy --prompt-cache and the
// in-memory prefix-reuse path; auto-restore fires only when in-memory reuse is poor.
// ---------------------------------------------------------------------------

// The .meta sidecar format layer (model_fp, SLOT_META_* constants incl. the v3 delta-node
// version, slot_meta_write/slot_meta_read) and the block chain hashing layer (auto_hash_mix/
// auto_block_hashes — index keys and filenames, media-aware) live in server-common.{h,cpp} so the
// parser of untrusted on-disk bytes and the hash algorithm link into standalone
// unit tests (tests/test-slot-meta.cpp, tests/test-auto-hash.cpp). Everything
// below is the cache logic proper and stays private to this translation unit.
// Block boundaries are the only resumable prefix lengths (vLLM-APC / SGLang-radix
// granularity); we NEVER trust a hash alone (invariant 2) — the caller byte-verifies
// tokens before any restore.

// Identity hash of an mmproj GGUF file from its header only (no tensor data read,
// ~ms even for a multi-GB file): FNV-1a/splitmix chain over every KV pair (key +
// typed value bytes), every tensor's name/shape/type, plus the file size. Computed
// once whenever an mmproj is loaded — it feeds model_fp.fp_mmproj and is exposed in
// /props for operators. Fills `out` and returns true; returns false if the header
// cannot be parsed (the caller treats that as a load failure — mtmd just loaded the
// same file, so a parse failure here means it changed under us).
static bool mmproj_header_fingerprint(const std::string & path, uint64_t & out) {
    out = 0;
    gguf_init_params iparams = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), iparams);
    if (gctx == nullptr) {
        return false;
    }
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix_bytes = [&h](const void * p, size_t n) {
        const unsigned char * b = (const unsigned char *) p;
        for (size_t i = 0; i < n; ++i) {
            h ^= (uint64_t) b[i]; h *= 0x100000001b3ULL;
        }
    };
    // every variable-length field folds its length in first so adjacent fields can
    // never alias ("ab"+"c" vs "a"+"bc"); counts/types fold in via auto_hash_mix.
    auto mix_str = [&](const char * s) {
        const size_t n = strlen(s);
        h = auto_hash_mix(h, (int32_t) n);
        mix_bytes(s, n);
    };
    auto mix_u64 = [&](uint64_t v) {
        h = auto_hash_mix(h, (int32_t) (v & 0xFFFFFFFFu));
        h = auto_hash_mix(h, (int32_t) (v >> 32));
    };
    // element size of a scalar gguf type; 0 for string/array (handled separately;
    // bool arrays are stored as int8, matching gguf_get_arr_data's contract).
    auto scalar_size = [](gguf_type t) -> size_t {
        switch (t) {
            case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL:    return 1;
            case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:                         return 2;
            case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
            case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
            default:                                                             return 0;
        }
    };
    const int64_t n_kv = gguf_get_n_kv(gctx);
    h = auto_hash_mix(h, (int32_t) n_kv);
    for (int64_t i = 0; i < n_kv; ++i) {
        mix_str(gguf_get_key(gctx, i));
        const gguf_type t = gguf_get_kv_type(gctx, i);
        h = auto_hash_mix(h, (int32_t) t);
        if (t == GGUF_TYPE_STRING) {
            mix_str(gguf_get_val_str(gctx, i));
        } else if (t == GGUF_TYPE_ARRAY) {
            const gguf_type at = gguf_get_arr_type(gctx, i);
            const size_t    an = gguf_get_arr_n(gctx, i);
            h = auto_hash_mix(h, (int32_t) at);
            mix_u64((uint64_t) an);
            if (at == GGUF_TYPE_STRING) {
                for (size_t j = 0; j < an; ++j) {
                    mix_str(gguf_get_arr_str(gctx, i, j));
                }
            } else if (an > 0 && scalar_size(at) > 0) {
                mix_bytes(gguf_get_arr_data(gctx, i), an * scalar_size(at));
            }
        } else if (scalar_size(t) > 0) {
            mix_bytes(gguf_get_val_data(gctx, i), scalar_size(t));
        }
    }
    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    h = auto_hash_mix(h, (int32_t) n_tensors);
    for (int64_t i = 0; i < n_tensors; ++i) {
        mix_str(gguf_get_tensor_name(gctx, i));
        h = auto_hash_mix(h, (int32_t) gguf_get_tensor_type(gctx, i));
        const int64_t * ne = gguf_get_tensor_ne(gctx, i);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            mix_u64((uint64_t) ne[d]);
        }
    }
    gguf_free(gctx);
    std::error_code ec;
    const uintmax_t fsize = std::filesystem::file_size(path, ec);
    if (ec) {
        return false;
    }
    mix_u64((uint64_t) fsize);
    out = h;
    return true;
}

// One in-memory index entry: the longest snapshot that reaches a given block
// boundary. Mirrors slot_save_unit's mtime LRU semantics for reconciliation.
struct auto_cache_entry {
    std::string state_path; // full state file path (sidecars derived via *_path helpers)
    uint32_t    n_tokens = 0;
    model_fp    fp;         // snapshot's fingerprint (must equal the live one to be used)
    // PINNED: a sibling "<state_path>.pin" marker exists. A pinned entry is NEVER dropped from its
    // by_boundary bucket when the per-boundary cap (AUTO_MAX_CANDIDATES_PER_BOUNDARY) is exceeded, so a
    // pinned base — the SHORTEST entry in a bucket shared with many longer divergent siblings, yet the
    // only real strict-prefix parent the incremental save parent-find can use — stays findable past 32
    // warm siblings. Set from disk during the scan; re-stat'd authoritatively when a drop is forced (a
    // .pin touched AFTER the entry was first indexed — the normal deploy order — is still honoured).
    bool        pinned = false;
};

// boundary-hash -> best (longest) entry covering that prefix length. Touched only
// from the single server-loop thread in v1 (mtx documented above). `scanned`
// guards the one-time startup scan; `dir_mtime`/`last_refresh` drive the cheap
// cross-process refresh (see auto_index_refresh): a peer process that writes a new
// snapshot bumps the slot-save directory's mtime, which the next lookup notices and
// re-scans — so a freshly-created cache becomes visible to OTHER processes without a
// restart (no inotify/no background thread; one stat per gated check).
struct auto_cache_index {
    std::mutex mtx;
    // boundary-hash -> snapshots reaching that prefix length, longest first. Kept multi-valued so a
    // longer (superset) snapshot never shadows a shorter exact-length one: a FULL/recurrent/hybrid/SWA
    // model can only restore a snapshot that is a WHOLE prefix of the request, so when the request ends
    // before the longer snapshot the shorter one is the ONLY usable candidate (auto_index_lookup picks
    // model-appropriately). Incremental saving makes overlapping supersets the common case.
    std::unordered_map<uint64_t, std::vector<auto_cache_entry>> by_boundary;
    std::unordered_set<std::string> indexed_files;    // state paths already scanned (incremental refresh)
    // state paths whose .meta failed to parse (corrupt, torn, or a version this binary does not
    // know). Units are immutable once atomically renamed, so a rejected file can never become
    // parseable — remembering it means a rescan never re-opens it, and a FUTURE meta version
    // bump costs each old reader one read total instead of one per scan. Reconciled together
    // with indexed_files when files disappear (auto_index_drop_missing_locked), so a peer
    // evicting a rejected unit lets a later same-name re-create be examined afresh.
    std::unordered_set<std::string> rejected_files;
    bool scanned = false;
    std::filesystem::file_time_type dir_mtime{};      // dir mtime as of the last scan
    std::chrono::steady_clock::time_point last_refresh{}; // throttle: skip stat storms in a burst
};

// Cross-process refresh throttle: at most one dir-mtime stat per this interval on the hot lookup
// path (a forced refresh on a lookup miss bypasses it). Sub-second so a peer's new snapshot is
// visible within ~1 prefill of being written — effectively immediate from the user's view.
static constexpr int AUTO_REFRESH_MIN_MS = 1000;

// Multi-candidate index bounds. A boundary may be reached by several snapshots of different lengths
// (incremental saving makes every save a superset of the previous). They are kept longest-first so a
// longer snapshot never shadows a shorter exact-length one that a FULL model needs; the per-boundary
// list is capped (dropping the shortest UNPINNED entry — a pinned base is never dropped). A RESTORE
// lookup returns at most AUTO_MAX_RESTORE_ATTEMPTS candidates for the caller to try in order (each
// attempt is an expensive .meta read + byte-compare); the incremental SAVE parent-find instead passes
// SIZE_MAX so the deepest strict-prefix parent (the SHORTEST entry, e.g. a pinned base) is never hidden
// by the cap behind longer divergent siblings sharing the same boundary bucket.
static constexpr size_t AUTO_MAX_CANDIDATES_PER_BOUNDARY = 32;
static constexpr size_t AUTO_MAX_RESTORE_ATTEMPTS        = 4;

// Forward-declared above slot_save_enforce_limits: expose only the delta-node fields so eviction can
// resolve the tree without model_fp in scope. Leaves the outputs as a parentless root ([0,0) with no
// parent) for a v1/v2 whole snapshot, a foreign file, or any read failure (invariant 4 — never crash).
// cur_fp_mmproj is irrelevant to the node fields (it only backfills a text-only fp_mmproj), so 0 is
// passed; a throwaway media_out absorbs any v2 records.
static bool slot_node_meta_probe(const std::string & state_filepath,
                                 uint64_t & parent_id, uint32_t & range_lo, uint32_t & range_hi) {
    parent_id = 0;
    range_lo  = 0;
    range_hi  = 0;
    model_fp     fp;
    llama_tokens toks;
    std::vector<server_media_record> media;
    return slot_meta_read(state_filepath, /*cur_fp_mmproj=*/0, fp, toks, media,
                          &parent_id, &range_lo, &range_hi);
}

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // idle-delay flush: cleared each time the slot is released (a fresh idle period begins), set
    // once its warm KV has been flushed to the auto disk cache, so the timed idle wake attempts the
    // flush at most once per idle period (auto_save_slot_if_useful's dedup covers any re-attempt).
    bool auto_idle_flushed = false;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;
    bool just_restored  = false; // set on disk slot-restore; one-shot, gates restored-slot KV reuse

    // --- mid-prefill shared-context base save (Option A) ---
    // When > 0, this cold-prefilling slot is ARMED to whole-save the leading shared preamble
    // [0, ctx_save_pos) as a deduplicated base: ctx_save_pos is the block-aligned first-user
    // boundary B_ctx. The prefill loop CLAMPS the batch exactly there (never crossing it), and the
    // post-decode hook in update_slots() whole-saves [0, B_ctx) at the instant the slot is resident
    // at exactly B_ctx — the true whole state there, so the save is recurrent-/SWA-/dense-correct.
    // One-shot: armed once per task at prompt start, cleared (-> -1) after the save so the slot then
    // prefills [B_ctx, N) normally. -1 = not armed (no boundary / warm restore / small preamble).
    int32_t ctx_save_pos = -1;

    // --- KV restore-reuse (logits sidecar) ---
    // Full-vocab logits of this slot's most recently sampled token, captured at sample time
    // (only populated for FULL/recurrent models when --slot-save-path is set). Serialized to the
    // <state>.logits sidecar on SLOT_SAVE so a later exact-prompt "regenerate" can emit the first
    // token WITHOUT re-decoding into the (un-rewindable) restored recurrent state.
    std::vector<float> logits_last;     // size n_vocab when valid, else empty
    // Token count of the prompt-state that `logits_last` corresponds to (i.e. the slot's KV/token
    // length at the moment of capture). Used to BIND the captured distribution to a specific state:
    // a sidecar is only written when this equals the saved snapshot's token_count, so a stale
    // distribution (e.g. left over from a prior task, or skipped on a spec-decode step) can never
    // be serialized against a mismatched state. -1 = no valid capture.
    int32_t logits_last_n_tokens = -1;
    // Logits loaded from a sidecar at SLOT_RESTORE, consumed once by the restore-continue path.
    std::vector<float> restored_logits; // size n_vocab when a valid sidecar was loaded, else empty

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);

        prompt.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    std::vector<float> inp_embd;

    server_slot_stats stats;

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        spec_is_replay = false;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        n_accepted_per_pos.clear();

        n_predict_max = -1;

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();

        // one-shot; never carry restored sidecar logits into a non-restore request.
        // NOTE: logits_last is deliberately NOT cleared here — it is the slot's running
        // "last sampled distribution" and must survive into the idle state so a subsequent
        // SLOT_SAVE can serialize it.
        restored_logits.clear();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                add_ok &= batch.add(id, inp_embd, prompt.tokens.pos_next(), true, false);
            } else {
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true, false);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true, false);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used = ggml_time_us();

            state             = SLOT_STATE_IDLE;
            auto_idle_flushed = false; // fresh idle period: eligible for a timed idle flush again

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear();
            }

            callback_on_reset(*this);

            reset();

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = {
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            };

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.i_batch = i_batch;

        other.stats = stats;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    // stub tripwire: a placeholder chunk (identity-only, no pixels/samples, e.g. one rehydrated
    // by a manual media restore) can never be encoded. No request-driven path puts one here
    // (request chunks always carry real data), so reaching this means the slot's state is
    // inconsistent. Refuse rather than fail mid-decode with the sequence half-written; the
    // caller turns a non-zero return into send_error() + slot.release().
    if (mtmd_input_chunk_is_placeholder(chunk.get())) {
        SLT_ERR(slot, "refusing to encode a placeholder media chunk, idx = %zu\n", idx);
        return -1;
    }


    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                void * cb_data = slot.spec;
                static auto cb = [](llama_batch batch, void * user_data) {
                    common_speculative * spec = static_cast<common_speculative *>(user_data);
                    if (!common_speculative_process(spec, batch)) {
                        return 1;
                    }
                    return 0;
                };

                llama_pos new_n_past; // unused for now
                res = mtmd_helper_decode_image_chunk(
                    mctx,
                    slot.ctx_tgt,
                    chunk.get(),
                    embd,
                    slot.prompt.tokens.pos_next(),
                    slot.id,
                    llama_n_batch(slot.ctx_tgt),
                    &new_n_past,
                    cb,
                    cb_data
                );
                if (res != 0) {
                    SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                    return -1;
                }
                n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                return 0; // success
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    mbatch.reset(mtmd_batch_init(mctx));
    res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return try_decode();
}

//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    // gguf-header hash of the loaded mmproj file, computed once at load whenever
    // --mmproj is set (0 on a text-only server); see mmproj_header_fingerprint.
    uint64_t fp_mmproj = 0;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // P0.1 (anchored-resume): does THIS model's memory class actually honour a position range in
    // state_write_range? The base-class default (src/llama-memory.h:132-136) IGNORES [p0,p1) and
    // writes the WHOLE sequence, and only 4 of the 7+ memory implementations override it. On a
    // non-overriding class a "delta node" silently contains [0,N); composing root [0,lo) + that node
    // under NO_CLEAR yields DUPLICATE cells per position => wrong attention, silently.
    // We cannot ask the engine (no API reports what it serialised) and we cannot sniff the blob
    // (composite classes concatenate independently-narrowing sections). So we PROBE, once, on the
    // first delta write, against real resident state: write the range AND the whole sequence and
    // compare byte counts. Probe-don't-declare mirrors common_context_can_seq_rm, which likewise
    // decodes and tries the operation rather than trusting a class predicate.
    // Fails CLOSED: unknown/no => never write deltas, only whole roots at anchors.
    enum class delta_cap : uint8_t { unknown, yes, no };
    delta_cap delta_capable = delta_cap::unknown;

    // The value the ENGINE masks with (llama_model_n_swa), NEVER zeroed by --swa-full. Every disk
    // save/restore soundness decision must use THIS, not n_swa: --swa-full enlarges the SWA cache but
    // does not disable masking, so saves stay windowed even when n_swa above has been zeroed.
    int32_t n_swa_mem = 0;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;
    int slots_debug = 0;
    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    int64_t t_last_load_progress_ms = 0;

    // --- auto disk prompt/KV cache (opt-in: --slot-save-auto) ---
    // Default-constructed: empty and untouched when the feature is OFF (invariant 1).
    // `cur_fp` is the live model fingerprint, computed once at load (only when enabled).
    auto_cache_index auto_idx;
    model_fp         cur_fp;

    // The ONE gate for the entire auto disk cache. When false, NO hook below does
    // any work (no scan, no hash, no alloc). This is invariant 1 — the first
    // statement of every auto_* hook is `if (!auto_cache_enabled()) return;`.
    bool auto_cache_enabled() const {
        return params_base.slot_save_auto && !params_base.slot_save_path.empty();
    }

    // Can a loaded snapshot of length L only ever be reused as a WHOLE prefix, never trimmed back
    // to a shorter verified prefix v < L? Reuse lengths are block-aligned, so any v < L is at least
    // one whole block short, and that is outside what these classes can do:
    //   FULL - llama_memory_seq_rm refuses ANY partial range, so the legal resume points are
    //          {0, L} and nothing in between.
    //   RS   - partial seq_rm is legal only within llama_n_rs_seq(ctx) positions of the tail
    //          (llama-memory-recurrent.cpp: the rollback must be in [1, n_rs_seq], typically 2-5;
    //          outside it seq_rm returns false and common_context_seq_rm turns that into a
    //          GGML_ABORT). A block-aligned v can never land inside that window, so we do not try
    //          to exploit it.
    //   SWA  - llama_kv_cache::state_write DROPS every SWA-masked cell at save time, so a snapshot
    //          of length L persists only the window [L - n_swa_mem, L): a shorter prefix has no
    //          cells behind it and re-decoding from there would attend over a hole.
    // The one remaining class is PART with n_swa_mem == 0 (plain attention), which rewinds per
    // token. Uses n_swa_mem, never n_swa: --swa-full zeroes n_swa for the batch planner but the
    // engine keeps masking on save.
    bool restore_is_whole_prefix_only() const {
        return ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
               ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS   ||
               n_swa_mem > 0;
    }

    void destroy() {
        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
        fp_mmproj = 0;
    }

    // ----- auto disk cache: fingerprint, index, restore, save (all gated by auto_cache_enabled()) -----

    // hash of the active LoRA set (ids/scales) for the fingerprint; 0 when no adapter is active.
    static uint64_t auto_lora_hash(const std::vector<common_adapter_lora_info> & lora) {
        uint64_t h = 0xcbf29ce484222325ULL;
        bool any = false;
        for (const auto & a : lora) {
            if (a.scale == 0.0f) {
                continue; // disabled adapter does not affect inference identity
            }
            any = true;
            for (char c : a.path) {
                h ^= (uint64_t) (unsigned char) c; h *= 0x100000001b3ULL;
            }
            uint32_t sc; std::memcpy(&sc, &a.scale, sizeof(sc));
            h = auto_hash_mix(h, (int32_t) sc);
        }
        return any ? h : 0;
    }

    // Compute the live model fingerprint once at load (invariant 3). Pure-CPU; only called from
    // an auto_cache_enabled() branch so it costs nothing when OFF.
    // See README "Automatic disk prompt cache" for which flags invalidate the cache.
    model_fp auto_compute_fingerprint() const {
        model_fp fp;
        // model identity string (arch + params + quant), hardened with size/n_params/n_embd/n_layer.
        char desc[256] = {0};
        llama_model_desc(model_tgt, desc, sizeof(desc));
        uint64_t h = 0xcbf29ce484222325ULL;
        for (const char * p = desc; *p; ++p) {
            h ^= (uint64_t) (unsigned char) *p; h *= 0x100000001b3ULL;
        }
        const uint64_t sz = llama_model_size(model_tgt);
        const uint64_t np = llama_model_n_params(model_tgt);
        h = auto_hash_mix(h, (int32_t) (sz & 0xFFFFFFFFu)); h = auto_hash_mix(h, (int32_t) (sz >> 32));
        h = auto_hash_mix(h, (int32_t) (np & 0xFFFFFFFFu)); h = auto_hash_mix(h, (int32_t) (np >> 32));

        fp.fp_model       = h;
        fp.fp_n_vocab     = (uint32_t) llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
        fp.fp_n_ctx_train = (uint32_t) llama_model_n_ctx_train(model_tgt);
        fp.fp_n_embd      = (uint32_t) llama_model_n_embd(model_tgt);
        fp.fp_n_layer     = (uint32_t) llama_model_n_layer(model_tgt);
        fp.fp_rope_type   = (uint32_t) llama_model_rope_type(model_tgt);
        // K/V cache type has no live-ctx getter — capture from the server's own params (the value
        // used to construct ctx_tgt). Blob-layout-critical: a Q4_0-KV blob into an F16 ctx corrupts.
        fp.fp_cache_k     = (uint32_t) params_base.cache_type_k;
        fp.fp_cache_v     = (uint32_t) params_base.cache_type_v;
        fp.fp_n_ctx       = (uint32_t) llama_n_ctx_seq(ctx_tgt);
        fp.fp_kv_full     = (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) ? 1u : 0u;
        fp.fp_block       = (uint32_t) params_base.slot_save_block;
        // effective rope scale (positions are baked into the saved state). rope_freq_scale==0 means
        // "use the model's trained value", so fall back to that for a stable comparison.
        float rs = params_base.rope_freq_scale != 0.0f
                       ? params_base.rope_freq_scale
                       : llama_model_rope_freq_scale_train(model_tgt);
        uint32_t rsb; std::memcpy(&rsb, &rs, sizeof(rsb));
        fp.fp_rope_scale  = (uint64_t) rsb;
        // rope_freq_base + the five YaRN params also bake positions into the saved
        // KV, so they MUST be part of identity. There is no public getter for the model's trained
        // rope base, so we normalize "use-model-default" to a single canonical 0 sentinel: when the
        // operator left the knob at its default (rope_freq_base==0; YaRN floats<0, i.e. -1.0 "auto";
        // yarn_orig_ctx<=0), we store 0. Two runs that both rely on the model default thus match;
        // any explicit override (or two different overrides) yields a different fp and refuses
        // (conservative — a needless miss is safe, a wrong restore is not). yarn_orig_ctx is an int.
        auto bitcast_f = [](float v) -> uint32_t { uint32_t u; std::memcpy(&u, &v, sizeof(u)); return u; };
        auto norm_yarn = [&](float v) -> uint32_t { return v < 0.0f ? 0u : bitcast_f(v); }; // <0 == model default
        const float rb = params_base.rope_freq_base > 0.0f ? params_base.rope_freq_base : 0.0f; // 0 == model default
        uint32_t rbb; std::memcpy(&rbb, &rb, sizeof(rbb));
        fp.fp_rope_base      = (uint64_t) rbb;
        fp.fp_yarn_ext       = norm_yarn(params_base.yarn_ext_factor);
        fp.fp_yarn_attn      = norm_yarn(params_base.yarn_attn_factor);
        fp.fp_yarn_beta_fast = norm_yarn(params_base.yarn_beta_fast);
        fp.fp_yarn_beta_slow = norm_yarn(params_base.yarn_beta_slow);
        fp.fp_yarn_orig_ctx  = params_base.yarn_orig_ctx > 0 ? (uint32_t) params_base.yarn_orig_ctx : 0u;
        // LoRA: fingerprint the global adapter set so a snapshot under adapter A never restores
        // under B. (Per-request adapter overrides additionally gate at the restore hook.)
        fp.fp_lora        = auto_lora_hash(params_base.lora_adapters);
        // deployment-shape bit (invariant 3): text-only server vs --mmproj server get
        // disjoint stores (mmproj-aware rope/projector wiring can change the text KV layout).
        fp.fp_mmproj_loaded = (mctx != nullptr) ? 1u : 0u;
        // projector identity, computed at mmproj load (0 on a text-only server).
        fp.fp_mmproj        = fp_mmproj;
        return fp;
    }

    // Auto-snapshot filename: a full-identity prefix (every operator== field, not just fp_model)
    // so that peers sharing one dir who agree on the token prefix but differ in any geometry field
    // get DISJOINT names instead of atomically renaming over each other; chain-hash + token-count
    // make it deterministic across same-config processes (a same-prefix save from another process
    // yields the same name -> atomic-rename-idempotent). The prefix is naming/collision-avoidance
    // only: the scan still verifies fp == cur_fp after reading the sidecar (it never parses the
    // prefix), and the block chain keeps its fp_model salt, so foreign v1 units written under the
    // old fp_model-prefixed name still index and restore unchanged.
    std::string auto_state_filename(uint64_t chain_hash, size_t n_tokens) const {
        char buf[96]; // "auto-" + 16 hex id + "-" + 16 hex hash + "-" + up to 20-digit count + ".bin" < 96
        snprintf(buf, sizeof(buf), "auto-%016" PRIx64 "-%016" PRIx64 "-%zu.bin",
                 cur_fp.identity_hash(), chain_hash, n_tokens);
        return params_base.slot_save_path + std::string(buf);
    }

    // Insert a snapshot at a boundary it reaches. Multiple snapshots are RETAINED per boundary
    // (longest first, deduped by length) so a longer superset does NOT shadow a shorter exact-length
    // snapshot — the shorter one is the only candidate a FULL model can restore when the request ends
    // before the longer one (see auto_index_lookup). The list is capped; the shortest is dropped first
    // (deep-context snapshots cost the most to lose and reconstruct).
    void auto_index_insert_locked(uint64_t boundary, const auto_cache_entry & e) {
        auto & v = auto_idx.by_boundary[boundary];
        for (auto & c : v) {
            if (c.n_tokens == e.n_tokens) {
                c.state_path = e.state_path; // same length/prefix: keep the newest file for this length
                c.fp         = e.fp;
                c.pinned     = e.pinned;
                return;
            }
        }
        // keep the vector sorted by descending n_tokens
        auto pos = std::lower_bound(v.begin(), v.end(), e,
            [](const auto_cache_entry & a, const auto_cache_entry & b) { return a.n_tokens > b.n_tokens; });
        v.insert(pos, e);
        if (v.size() > AUTO_MAX_CANDIDATES_PER_BOUNDARY) {
            // Over the cap: NEVER drop the SHORTEST entry, and never a pinned one.
            //
            // The shortest entry at a boundary is the snapshot that ends closest to it — i.e. the
            // SHARED BASE for this prefix. It is the only real strict-prefix parent the incremental
            // save parent-find can use, and the only candidate a *different* conversation sharing this
            // prefix can restore from. Every longer entry is a divergent sibling: losing one costs that
            // one lineage its deep reuse; losing the base costs EVERY lineage the shared prefix and
            // forces each later sibling to fall back to a whole (v1) snapshot.
            //
            // This used to drop the shortest unpinned entry, protecting the base only via a MANUAL
            // `<state>.pin` marker. That is backwards: bases created automatically (the mid-prefill
            // context base, and any root a chat happens to establish) are never pinned, so the one
            // entry the bucket exists to serve was the first thing evicted. A boundary shared by more
            // than the cap — one long conversation with >32 saved nodes, or >32 chats behind a common
            // system prompt — silently un-indexed the shared prefix while its file stayed on disk, and
            // every new chat then cold-prefilled it. `.pin` stays honoured, but is no longer
            // load-bearing for correctness of the common case.
            //
            // Scan shortest-first among the REST (skip the last element, the base); the .pin marker is
            // re-stat'd for authority (a pin touched after the entry was indexed — the normal deploy
            // order — is honoured and the cached flag refreshed). If every droppable entry is pinned,
            // leave the bucket one over the cap rather than evict a base or a pinned entry.
            for (auto rit = std::next(v.rbegin()); rit != v.rend(); ++rit) {
                std::error_code pec;
                rit->pinned = std::filesystem::exists(rit->state_path + ".pin", pec) && !pec;
                if (!rit->pinned) {
                    v.erase(std::next(rit).base());
                    break;
                }
            }
        }
    }

    // True iff an entry of EXACTLY n_tokens length is already indexed at `boundary`. The
    // shared-context checkpoint uses this for its redundant-write dedup: the 2nd..Nth chat sharing
    // the same [0,B) prefix finds the base already published at bhs[B/block-1] and writes nothing.
    // Unlike the whole-save path's equal-or-longer dedup, the base must match on EXACT length — a
    // LONGER snapshot at the same boundary is a divergent sibling prefix, not this base, so it must
    // not suppress the base write. CALLER MUST HOLD auto_idx.mtx.
    bool auto_index_has_exact_locked(uint64_t boundary, uint32_t n_tokens) const {
        auto it = auto_idx.by_boundary.find(boundary);
        if (it == auto_idx.by_boundary.end()) {
            return false;
        }
        for (const auto_cache_entry & c : it->second) {
            if (c.n_tokens == n_tokens) {
                return true;
            }
        }
        return false;
    }

    // Scan the slot-save dir and (re)build index entries from .meta sidecars: header-only reads
    // (never the multi-GB state). Each bad/foreign file is skipped individually (invariant 4);
    // foreign-model files are left on disk (a sibling model may own them). Idempotent: re-running it
    // only ever keep-longer-inserts the same/new entries (auto_index_insert_locked), so it is safe to
    // call repeatedly for the cross-process refresh. Records the dir mtime so a refresh can cheaply
    // tell whether anything changed. CALLER MUST HOLD auto_idx.mtx.
    void auto_index_scan_locked() {
        std::error_code mec;
        const auto dmt = std::filesystem::last_write_time(params_base.slot_save_path, mec);
        if (!mec) {
            auto_idx.dir_mtime = dmt; // snapshot the dir mtime we are scanning at
        }
        std::error_code ec;
        for (std::filesystem::directory_iterator it(params_base.slot_save_path, ec), end;
             !ec && it != end; it.increment(ec)) {
            std::error_code fec;
            if (!it->is_regular_file(fec) || fec) {
                continue;
            }
            const std::string p = it->path().string();
            // only our own state files: basename "auto-*.bin" (sidecars and temps skipped). Requiring
            // the "auto-" basename prefix rejects foreign/manual .bin files BY NAME before we open any
            // sidecar — the stated scan optimization.
            const std::string base = it->path().filename().string();
            if (base.rfind("auto-", 0) != 0) {
                continue; // not one of ours
            }
            if (p.size() < 4 || p.compare(p.size() - 4, 4, ".bin") != 0) {
                continue;
            }
            // already indexed — or already parse-rejected — by a prior scan? cheap skip so a
            // refresh only opens NEW files (units are immutable after their atomic rename).
            if (auto_idx.indexed_files.count(p) || auto_idx.rejected_files.count(p)) {
                continue;
            }
            // The publish sequence renames the .bin first and the .meta LAST, so a scan can
            // legitimately list a final-named .bin whose sidecar has not landed yet (a peer
            // mid-publish, or a writer that crashed in between — a later identical-prefix
            // save completes the unit under the same deterministic name). A MISSING sidecar
            // is therefore transient, not a verdict: skip WITHOUT caching so the next scan
            // retries. Only a sidecar that already EXISTS is immutable published state, so
            // only its parse failures may be remembered as permanent rejections. The
            // existence check runs BEFORE the read: a .meta that lands in between is simply
            // parsed (or retried next scan), never mis-cached.
            std::error_code sec;
            const bool meta_present = std::filesystem::exists(slot_meta_sidecar_path(p), sec) && !sec;
            model_fp fp;
            llama_tokens toks;
            std::vector<server_media_record> media;
            if (!slot_meta_read(p, cur_fp.fp_mmproj, fp, toks, media)) {
                if (meta_present) {
                    // short/corrupt meta, or an unknown version -> not indexable (invariant 4);
                    // remember the rejection so rescans never re-open the file.
                    auto_idx.rejected_files.insert(p);
                }
                continue;
            }
            if (!(fp == cur_fp)) {
                continue; // foreign model / requant / different ctx geometry (invariant 3)
            }
            // rehash from the sidecar's cells + media records (media empty on v1 => the
            // text-only chain, bit-identical to what the writer keyed the file with). Only
            // chunk-safe boundaries are emitted — including a media unit's pure-text
            // pre-image boundaries, so a text request can reuse a media snapshot's prefix.
            const auto bhs = auto_block_hashes(toks, media, params_base.slot_save_block,
                                               cur_fp.fp_model, cur_fp.fp_mmproj);
            // pin-awareness: a sibling "<state>.pin" marks this snapshot pinned so the per-boundary cap
            // never drops it (protects a pinned base that shares a bucket with >32 divergent siblings).
            std::error_code pec;
            const bool pinned = std::filesystem::exists(p + ".pin", pec) && !pec;
            auto_cache_entry e{ p, (uint32_t) toks.size(), fp, pinned };
            for (uint64_t bh : bhs) {
                auto_index_insert_locked(bh, e);
            }
            auto_idx.indexed_files.insert(p);
        }
    }

    // One-time startup scan: builds the initial index. Invariant 1: only ever called from an
    // auto_cache_enabled() branch.
    void auto_index_scan() {
        std::lock_guard<std::mutex> lk(auto_idx.mtx);
        if (auto_idx.scanned) {
            return;
        }
        auto_idx.scanned = true;
        auto_idx.last_refresh = std::chrono::steady_clock::now();
        auto_index_scan_locked();
    }

    // Cross-process refresh: make snapshots that OTHER processes created visible here WITHOUT a
    // restart. Cheap by design: throttled to at most once per AUTO_REFRESH_MIN_MS, and even then it
    // only does one stat of the dir mtime — a full re-scan happens ONLY when the dir actually changed
    // (a peer create/rename/delete bumps the dir mtime) or when `force` is set (a lookup miss, where
    // we are about to pay a cold prefill anyway so the scan is free in comparison). On a change we
    // also drop entries whose files a peer evicted. CALLER MUST HOLD auto_idx.mtx.
    void auto_index_refresh_locked(bool force) {
        const auto now = std::chrono::steady_clock::now();
        if (!force &&
            now - auto_idx.last_refresh < std::chrono::milliseconds(AUTO_REFRESH_MIN_MS)) {
            return; // throttled: avoid a stat storm during a burst of lookups
        }
        auto_idx.last_refresh = now;
        std::error_code ec;
        const auto dmt = std::filesystem::last_write_time(params_base.slot_save_path, ec);
        if (!ec && dmt == auto_idx.dir_mtime && !force) {
            return; // nothing changed on disk since the last scan
        }
        // a peer changed the dir (or forced): re-scan for NEW files, then reconcile deletions.
        auto_index_scan_locked();
        auto_index_drop_missing_locked();
    }

    // Longest-prefix lookup over the request's cell-aligned tokens. Returns candidate snapshots to
    // try, BEST FIRST (deepest boundary first; within a boundary, longest first). For every class
    // restore_is_whole_prefix_only() covers (FULL, RS and SWA), snapshots longer than the request are
    // filtered out here: the whole snapshot must be a prefix of the request, so a longer one can never
    // restore. Plain-attention PART can rewind, so all lengths are kept. That predicate is the single
    // definition of the class set, named here so the prose and the filter cannot drift apart again
    // (RS was missing from the filter while this comment already said "recurrent").
    // The caller tries each in order until one restores (each
    // rejected candidate costs only a small .meta read + byte-compare; the multi-GB state loads only
    // once a candidate passes its gates) — this fall-through is what stops a longer superset from
    // shadowing a shorter usable one at the same boundary. Verification (byte-compare of the
    // candidate's persisted cells + per-record media identity) is mandatory and done by the caller
    // (invariant 2). O(#blocks).
    //
    // `max_attempts` caps how many candidates are returned. The RESTORE path keeps the default (each
    // rejected restore costs a full .meta read + byte-compare, and a restore-side shadow was already
    // cured by the multi-candidate index) — but the incremental SAVE parent-find passes SIZE_MAX to
    // scan the FULL per-boundary candidate vectors: the deepest strict-prefix parent (the base) is the
    // SHORTEST entry in its bucket, so with many divergent siblings it falls past the top few; a 4-cap
    // there hides it and forces a whole (v1) snapshot instead of a small delta. Ordering is unchanged
    // (deepest boundary first, longest first within a boundary), so the caller's first strict-prefix
    // match is still the deepest parent.
    std::vector<auto_cache_entry> auto_index_lookup(const server_tokens & req,
                                                    size_t max_attempts = AUTO_MAX_RESTORE_ATTEMPTS) {
        std::vector<auto_cache_entry> out;
        if (!auto_cache_enabled()) {
            return out; // off by default
        }
        // request-side identity records (empty for a text-only request, whose chain is then
        // bit-identical to the pre-media one). An identity-less chunk (e.g. a placeholder
        // bitmap) can never be verified against any snapshot, so such a request does not
        // look up at all — mirrors the save-side refusal.
        std::vector<server_media_record> media;
        try {
            media = req.extract_media_records();
        } catch (const std::exception & e) {
            SRV_WRN("auto-restore: lookup refused, %s\n", e.what());
            return out;
        }
        const auto bhs = auto_block_hashes(req.get_cell_tokens(), media, params_base.slot_save_block,
                                           cur_fp.fp_model, cur_fp.fp_mmproj);
        std::lock_guard<std::mutex> lk(auto_idx.mtx);
        // Cross-process visibility: cheaply pick up snapshots a peer process created since our last
        // scan (throttled dir-mtime check). Then search; on a MISS, force a re-scan and search again
        // — the force is justified because a miss means we are about to cold-prefill, so the scan
        // cost is negligible against it, and a peer's snapshot written <1s ago (within the throttle
        // window) is still found on this first request rather than only the next one.
        auto_index_refresh_locked(/*force=*/false);
        for (int attempt = 0; attempt < 2; ++attempt) {
            out.clear();
            std::unordered_set<std::string> seen;
            for (size_t k = bhs.size(); k-- > 0; ) { // longest boundary first
                auto it = auto_idx.by_boundary.find(bhs[k]);
                if (it == auto_idx.by_boundary.end()) {
                    continue;
                }
                for (const auto_cache_entry & c : it->second) { // longest first within the boundary
                    if (!(c.fp == cur_fp)) {
                        continue; // invariant 3
                    }
                    if (restore_is_whole_prefix_only() && c.n_tokens > req.size()) {
                        // A snapshot longer than the request is never a whole prefix of it, and every
                        // class in restore_is_whole_prefix_only() (FULL, RS and SWA alike) can restore
                        // only a whole prefix. For SWA specifically, its persisted window is anchored at
                        // its own end, so a longer snapshot can only ever be refused by the restore gate,
                        // and because the store is dominated by release-time (prompt+generated)
                        // snapshots, such siblings are the COMMON case. Left unfiltered they exhaust
                        // AUTO_MAX_RESTORE_ATTEMPTS and starve the shorter, usable mid-prefill base
                        // (which sorts last).
                        continue;
                    }
                    if (!seen.insert(c.state_path).second) {
                        continue; // the same snapshot reaches several boundaries
                    }
                    out.push_back(c);
                    if (out.size() >= max_attempts) {
                        return out;
                    }
                }
            }
            if (!out.empty()) {
                return out;
            }
            if (attempt == 0) {
                auto_index_refresh_locked(/*force=*/true); // miss -> rescan once before giving up
            }
        }
        return out;
    }

    // After an LRU eviction (which deletes files silently — ours OR a peer process's), drop index
    // boundaries pointing at files that no longer exist, and forget them in indexed_files so a future
    // re-create can be re-indexed. Cheap stat per unique path; keeps index <-> disk consistent (invariant 4).
    // A lookup that races an eviction and finds a now-deleted file simply fails the load -> prefill.
    // CALLER MUST HOLD auto_idx.mtx.
    void auto_index_drop_missing_locked() {
        std::unordered_set<std::string> gone;
        for (auto it = auto_idx.by_boundary.begin(); it != auto_idx.by_boundary.end(); ) {
            auto & vec = it->second;
            for (auto vit = vec.begin(); vit != vec.end(); ) {
                std::error_code ec;
                if (!std::filesystem::exists(vit->state_path, ec) || ec) {
                    gone.insert(vit->state_path);
                    vit = vec.erase(vit);
                } else {
                    ++vit;
                }
            }
            if (vec.empty()) {
                it = auto_idx.by_boundary.erase(it);
            } else {
                ++it;
            }
        }
        for (const auto & p : gone) {
            auto_idx.indexed_files.erase(p);
        }
        // also forget parse-rejected units whose files a peer evicted, so a later
        // re-create under the same deterministic name is examined afresh.
        for (auto it = auto_idx.rejected_files.begin(); it != auto_idx.rejected_files.end(); ) {
            std::error_code ec;
            if (!std::filesystem::exists(*it, ec) || ec) {
                it = auto_idx.rejected_files.erase(it);
            } else {
                ++it;
            }
        }
    }

    void auto_index_drop_missing() {
        std::lock_guard<std::mutex> lk(auto_idx.mtx);
        auto_index_drop_missing_locked();
    }

    // Restore a disk snapshot INTO `slot`, mirroring the SLOT_RESTORE handler body. Returns true on
    // success (slot.prompt.tokens / n_past-equivalent + just_restored + restored_logits are set as
    // for a manual restore). On ANY failure (load <=0, capacity exceeded) the slot seq is left
    // cleared and false is returned so the caller falls through to a normal prefill (invariant 4).
    bool do_slot_restore(server_slot & slot, const std::vector<std::string> & node_paths,
                         size_t * out_token_count = nullptr, size_t * out_nread = nullptr) {
        if (node_paths.empty()) {
            slot.prompt.tokens.clear();
            if (out_nread)       { *out_nread = 0; }
            if (out_token_count) { *out_token_count = 0; }
            return false;
        }
        // A snapshot is a ctx_tgt state file: llama_state_seq_load_file_ext below writes ctx_tgt only,
        // and nothing draft-side is ever persisted. So whatever ctx_dft holds for this seq belongs to
        // the PREVIOUS prompt, and after the restore the MTP/EAGLE3 head would draft while attending
        // over another conversation's cells (the post-restore prefill only decodes the SUFFIX, and the
        // seq_rm at [p0, -1) that precedes it removes nothing below p0). Reset it: a COLD draft is
        // correct, only unaccelerated. This does NOT reconstruct draft state for [0, L), no snapshot
        // carries it, so drafting stays cold until generation refills ctx_dft.
        // ORDER IS LOAD-BEARING: this must run BEFORE the load, not after. On a shared-memory draft
        // context (common/speculative.cpp is_mem_shared, gemma4-class: llama_get_ctx_other(ctx_dft) ==
        // ctx_tgt) a clear here is harmless because node [0] clears the destination seq anyway, whereas
        // the same clear after the load could discard the state we just read.
        // Deliberately a direct llama_memory_seq_rm on ctx_dft, not slot.mem.seq_rm: the wrapper mirrors
        // onto both contexts and cannot express "load into one, reset the other". (-1, -1) is the rm_all
        // path, legal on every memory class, so the result is not checked (same style as
        // auto_restore_drop below).
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
        }
        // Load the chain in position order: node [0] clears the destination seq (a whole base/root
        // snapshot), nodes [1..] append their delta cells with NO_CLEAR so base + deltas compose.
        // A 1-element chain is exactly the previous single clearing load — byte-for-byte the same.
        llama_tokens tokens;
        tokens.resize(slot.n_ctx);
        size_t token_count = 0;
        size_t total_nread = 0;
        for (size_t i = 0; i < node_paths.size(); ++i) {
            const llama_state_seq_flags flags = (i == 0) ? 0 : LLAMA_STATE_SEQ_FLAGS_NO_CLEAR;
            size_t node_token_count = 0;
            const size_t nread = llama_state_seq_load_file_ext(
                ctx_tgt, node_paths[i].c_str(), slot.id, flags,
                tokens.data(), tokens.size(), &node_token_count);
            if (nread == 0) {
                slot.prompt.tokens.clear(); // KV may already have been invalidated by the partial load
                if (out_nread)       { *out_nread = total_nread; }
                if (out_token_count) { *out_token_count = 0; }
                return false;
            }
            total_nread += nread;
            token_count = node_token_count; // the tip (last) node header carries the full [0, hi) list
        }
        if (out_nread)       { *out_nread = total_nread; }
        if (out_token_count) { *out_token_count = token_count; }
        tokens.resize(token_count);
        slot.prompt.tokens.clear();
        slot.prompt.tokens.insert(tokens);
        slot.just_restored = true;

        // FULL only. Reconstruct a context checkpoint at the restored position so a FULL model, which
        // cannot partially rewind, can reuse this state for the suffix.
        // RS is deliberately NOT included, and this is not an oversight, do not add it "for symmetry".
        // With the whole-prefix rule in auto_restore_into_slot, an RS restore leaves n_past == pos_next
        // == L while a recurrent pos_min is the TAIL (L-1), so: when the request EXTENDS,
        // pos_min_thold == L and the consumer gate `pos_min >= pos_min_thold` is false, the checkpoint
        // list is never iterated; on an exact resend, pos_min_thold == L-1 and the acceptance test
        // `cur.pos_min < pos_min_thold` is L-1 < L-1, false. The checkpoint would be unreachable in one
        // case and rejected in the other, while create_checkpoint pays a whole-sequence PARTIAL_ONLY
        // state copy per restore. Adding it only becomes live if someone relaxes the consumer to accept
        // cur.pos_min == pos_min_thold for tail memories, and that is exactly the GGML_ABORT the NOTE at
        // the acceptance predicate documents. The same reasoning forbids extending the SWA synth in
        // auto_restore_into_slot to RS.
        // SWA gets its own synth there (it is a real window start, not a tail).
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            const auto ckpt_pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
            const auto ckpt_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
            if (ckpt_pos_min >= 0) {
                slot.prompt.checkpoints.clear();
                create_checkpoint(slot, 0, ckpt_pos_min, ckpt_pos_max);
            }
        }

        // The restored state's running distribution is unknown; invalidate any stale capture so a
        // later SLOT_SAVE cannot persist a mismatched sidecar.
        slot.logits_last.clear();
        slot.logits_last_n_tokens = -1;

        // Load the regenerate logits sidecar (FULL only) so an exact-prompt regenerate can emit the
        // first token without re-decoding into the restored recurrent state.
        slot.restored_logits.clear();
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
            if (slot_logits_read(node_paths.back(), nv, (uint32_t) token_count, slot.restored_logits)) {
                SLT_INF(slot, "loaded logits sidecar (%d vocab, %zu tokens) — regenerate fast-path armed\n", nv, token_count);
            }
        }
        return true;
    }

    // Drop a just-restored snapshot entirely: empty the slot's KV seq and prompt/restore state so
    // the caller falls back to a clean cold prefill (invariant 4). Canonical post-restore bail-out
    // for auto_restore_into_slot — every abort after do_slot_restore succeeded goes through here.
    void auto_restore_drop(server_slot & slot) {
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, -1, -1);
        slot.prompt.tokens.clear();
        slot.prompt.checkpoints.clear();
        slot.just_restored = false;
        slot.restored_logits.clear();
    }

    // MANUAL-RESTORE rehydration: a media snapshot's state file persists LLAMA_TOKEN_NULL
    // cells, so after do_slot_restore the slot's prompt holds NULL cells with no live chunks
    // behind them — a shape no downstream consumer can traverse. Unlike the auto path there is
    // no request to rebuild from, so the v2 .meta sidecar's identity records are rehydrated
    // into STUB chunks (mtmd_input_chunk_init_stub: id + geometry, placeholder data). Stubs
    // verify and count positions exactly like live chunks; the persisted embeddings are already
    // inside the loaded KV state, so their pixels are never needed — any path that would
    // re-encode one refuses and clears (see process_mtmd_chunk). Returns false with `err` set
    // when the snapshot cannot be rehydrated (missing/invalid sidecar, fingerprint mismatch,
    // sidecar/state divergence, irreproducible chunk geometry); the caller must then drop the
    // restored state. A text snapshot (no NULL cells) returns true untouched.
    bool manual_restore_rehydrate_media(server_slot & slot, const std::string & filepath, std::string & err) {
        const llama_tokens & cells = slot.prompt.tokens.get_cell_tokens();
        if (std::find(cells.begin(), cells.end(), LLAMA_TOKEN_NULL) == cells.end()) {
            return true; // text snapshot: nothing to rehydrate
        }
        if (!mctx) {
            err = "state file contains media cells but the server has no multimodal projector loaded";
            return false;
        }
        model_fp disk_fp;
        llama_tokens disk_toks;
        std::vector<server_media_record> disk_media;
        if (!slot_meta_read(filepath, cur_fp.fp_mmproj, disk_fp, disk_toks, disk_media)) {
            err = "state file contains media cells but has no valid .meta sidecar to rebuild them from";
            return false;
        }
        if (disk_media.empty()) {
            // unreachable via our own writers (a v1 sidecar with NULL cells is rejected by
            // slot_meta_read), kept as an explicit guard against future format drift
            err = ".meta sidecar carries no media records for a media state file";
            return false;
        }
        if (!(disk_fp == cur_fp)) {
            err = "snapshot fingerprint mismatch (model, projector or context geometry changed)";
            return false;
        }
        if (disk_toks != cells) {
            err = ".meta sidecar does not describe this state file";
            return false;
        }
        // rebuild the prompt: text cells verbatim, each record as a stub chunk. slot_meta_read
        // guarantees the records tile the NULL cells exactly, so this walk covers every cell
        // (push_back(llama_token) throws on a NULL cell outside a record — impossible here, but
        // the catch keeps a corrupt sidecar from unwinding through the server loop).
        server_tokens rebuilt;
        rebuilt.has_mtmd = true;
        try {
            size_t r = 0;
            for (size_t i = 0; i < cells.size(); ) {
                if (r < disk_media.size() && i == (size_t) disk_media[r].start_idx) {
                    const auto & rec = disk_media[r];
                    mtmd::input_chunk_ptr stub(mtmd_input_chunk_init_stub(
                            mctx, rec.is_audio != 0, rec.id.c_str(),
                            rec.n_tokens, (llama_pos) rec.n_pos, rec.nx, rec.ny));
                    if (!stub) {
                        err = "cannot rebuild a media chunk with the snapshot's geometry on this model";
                        return false;
                    }
                    rebuilt.push_back(stub.get()); // copies
                    i += rec.n_tokens;
                    r++;
                } else {
                    rebuilt.push_back(cells[i]);
                    i++;
                }
            }
        } catch (const std::exception & e) {
            err = std::string("rehydration failed: ") + e.what();
            return false;
        }
        if (rebuilt.get_cell_tokens() != cells || !rebuilt.validate(ctx_tgt)) {
            err = "rehydrated prompt failed validation";
            return false;
        }
        slot.prompt.tokens = std::move(rebuilt);
        return true;
    }

    // Peek the number of KV cells a range-save .bin actually serialized, WITHOUT loading the
    // multi-GB state into a seq. The .bin layout is: magic(u32) version(u32) n_token_count(u32)
    // tokens[n_token_count] then the memory state — and for every media-relevant family the state
    // opens with n_stream(u32) followed by, per stream, cell_count(u32) [+ meta + data when the
    // count is non-zero, nothing when zero]. A range save writes the delta sub-cache FIRST (plain
    // FULL attention writes the single cache; iSWA writes kv_base's [p0,p1) delta before the whole
    // kv_swa; hybrid writes the attention delta before the whole recurrent state), and a single-seq
    // slot save populates exactly ONE stream, so the first non-zero cell_count is the delta's cell
    // count. Returns true and sets `cells_out` on a well-formed header whose n_token_count matches
    // `n_tokens` (a guard that this is the file we just wrote); false on any short read / mismatch,
    // which the caller treats as a failed verify. Best-effort, never throws.
    bool delta_bin_cell_count(const std::string & bin_path, size_t n_tokens, uint32_t & cells_out) {
        std::ifstream f(bin_path, std::ios::binary);
        if (!f) {
            return false;
        }
        auto rd_u32 = [&](uint32_t & v) -> bool {
            unsigned char b[4];
            f.read((char *) b, 4);
            if (f.gcount() != 4) {
                return false;
            }
            v = (uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24);
            return true;
        };
        uint32_t magic = 0, version = 0, tok_count = 0, n_stream = 0;
        if (!rd_u32(magic) || !rd_u32(version) || !rd_u32(tok_count)) {
            return false;
        }
        if ((size_t) tok_count != n_tokens) {
            return false; // header token count disagrees -> not the delta we just wrote
        }
        f.seekg((std::streamoff) tok_count * (std::streamoff) sizeof(llama_token), std::ios::cur);
        if (!f || !rd_u32(n_stream)) {
            return false;
        }
        // exactly one stream holds this seq's cells; empty streams write only cell_count == 0 with no
        // meta/data following, so scan cell_counts until the first non-zero one (the delta count).
        for (uint32_t s = 0; s < n_stream; ++s) {
            uint32_t cell_count = 0;
            if (!rd_u32(cell_count)) {
                return false;
            }
            if (cell_count != 0) {
                cells_out = cell_count;
                return true;
            }
        }
        cells_out = 0; // every stream empty: a delta that serialized nothing (a mismatch upstream)
        return true;
    }

    // Build the root->tip chain of node .bin paths for a (possibly delta) tip by walking parent
    // links ON DISK (the tree is DERIVED FROM DISK — no in-RAM map) and verifying every hop.
    // `tip_path` is the tip's .bin; `tip_toks` is its authoritative full [0, range_hi) cell-token
    // list (the meta is WHOLE even for a v3/v4 delta, so this is the byte-authority every parent
    // prefix is checked against); (parent_id, range_lo) are the tip's node-tail fields — (0, 0)
    // means a whole snapshot whose chain is just itself (a single clearing load == the pre-delta
    // behaviour). On success `chain` is root..tip in position order and returns true; on ANY
    // inconsistency (missing/corrupt/fingerprint-drift/non-contiguous/byte-mismatch parent, or a
    // pathological depth) returns false so the caller cold-prefills (invariant 4). SHARED by both
    // the auto-restore path and the manual /slots restore path so a v4 media delta tip composes
    // base + deltas (NO_CLEAR) identically on both — never mis-loading a partial .bin (decision 2).
    bool auto_build_restore_chain(const std::string & tip_path, const llama_tokens & tip_toks,
                                  uint64_t parent_id, uint32_t range_lo,
                                  std::vector<std::string> & chain) {
        chain.clear();
        chain.push_back(tip_path);
        // A v1/v2 whole snapshot is its own root: a single-element chain == the previous single
        // clearing load. A v3/v4 delta only stores its tail cells, so walk parent links to the
        // root and load base + deltas in position order (NO_CLEAR) to recompose the full prefix.
        if (parent_id == 0 && range_lo == 0) {
            return true;
        }
        uint64_t cur_parent_id = parent_id;
        uint32_t cur_range_lo  = range_lo;
        const size_t MAX_CHAIN_DEPTH = 4096; // bounded walk: a corrupt/looping link never hangs.
        while (true) {
            if (chain.size() > MAX_CHAIN_DEPTH) {
                return false; // pathological depth -> cold prefill (invariant 4)
            }
            const std::string parent_path = auto_state_filename(cur_parent_id, cur_range_lo);
            model_fp     parent_fp;
            llama_tokens parent_toks;
            std::vector<server_media_record> parent_media;
            uint64_t     parent_parent_id = 0;
            uint32_t     parent_lo        = 0;
            uint32_t     parent_hi        = 0;
            if (!slot_meta_read(parent_path, cur_fp.fp_mmproj, parent_fp, parent_toks, parent_media,
                                &parent_parent_id, &parent_lo, &parent_hi)) {
                return false; // parent meta missing/corrupt -> cold prefill
            }
            if (!(parent_fp == cur_fp)) {
                return false; // fingerprint drift on the parent -> cold prefill
            }
            // contiguity: the parent must end exactly where its child begins.
            if (parent_hi != cur_range_lo) {
                return false;
            }
            // IDENTITY: hash + range-contiguity alone do NOT prove this .bin holds the tip's actual
            // prefix — a parent_id/n_tokens filename collision (two distinct prefixes of equal length
            // whose block-boundary hash coincides) or a base rewritten for a different prefix could
            // land on the same deterministic name and compose the WRONG KV for [0, parent_hi) under
            // NO_CLEAR, silently. `tip_toks` is the authoritative full [0, range_hi) token record, so
            // byte-verify the parent's recorded tokens against that tip prefix. Media cells are
            // LLAMA_TOKEN_NULL on both sides (a media parent of a media delta), so this compare is
            // NULL==NULL for the shared media prefix — the per-record identity backstop that closes
            // that gap is enforced save-side (parent-find) and restore-side (tip full tiling verify).
            if (parent_toks.size() != (size_t) parent_hi ||
                (size_t) parent_hi > tip_toks.size() ||
                !std::equal(parent_toks.begin(), parent_toks.end(), tip_toks.begin())) {
                return false; // parent KV does not correspond to this prefix -> cold prefill
            }
            // the parent .bin must exist (meta is published last, but an orphan-reap can race).
            { std::ifstream pf(parent_path, std::ios::binary); if (!pf) { return false; } }
            chain.push_back(parent_path);
            if (parent_parent_id == 0 && parent_lo == 0) {
                break; // reached the root covering [0, hi)
            }
            cur_parent_id = parent_parent_id;
            cur_range_lo  = parent_lo;
        }
        // chain is tip..root; reverse to root..tip (position order) for the compose load.
        std::reverse(chain.begin(), chain.end());
        return true;
    }

    // AUTO-RESTORE wrapper: byte-verify the candidate's persisted cells against the request prefix
    // and its media records against the request's live chunks (invariant 2), confirm the
    // fingerprint (invariant 3), then restore. Returns the verified prefix length actually
    // restored, or 0 if nothing was restored (caller keeps the in-memory prefill path).
    // `req` is the full request; `n_keep_mem` is the in-memory match to beat.
    int auto_restore_into_slot(server_slot & slot, const auto_cache_entry & cand,
                               const server_tokens & req, int n_keep_mem) {
        // read the small .meta sidecar (tokens + fp + media records) — never opens the multi-GB
        // state file (invariant 5).
        model_fp disk_fp;
        llama_tokens disk_toks;
        std::vector<server_media_record> disk_media;
        uint64_t disk_parent_id = 0;
        uint32_t disk_range_lo  = 0;
        uint32_t disk_range_hi  = 0;
        if (!slot_meta_read(cand.state_path, cur_fp.fp_mmproj, disk_fp, disk_toks, disk_media,
                            &disk_parent_id, &disk_range_lo, &disk_range_hi)) {
            return 0; // invariant 4
        }
        if (!(disk_fp == cur_fp)) {
            return 0; // invariant 3
        }
        // request-side identity: cell-aligned tokens plus media records (empty on a text-only
        // request). The extraction throws on an identity-less chunk (e.g. a placeholder
        // bitmap) — unverifiable, so the request simply does not restore (invariant 4).
        const llama_tokens & req_cells = req.get_cell_tokens();
        std::vector<server_media_record> req_media;
        try {
            req_media = req.extract_media_records();
        } catch (const std::exception & e) {
            SLT_WRN(slot, "auto-restore: refused, %s\n", e.what());
            return 0;
        }
        // byte-verify: longest common prefix of the persisted cells and the request cells
        // (invariant 2). Media cells are LLAMA_TOKEN_NULL on both sides and pass this compare
        // blindly — their content identity is verified per record next.
        const size_t lim = std::min(disk_toks.size(), req_cells.size());
        size_t v = 0;
        while (v < lim && disk_toks[v] == req_cells[v]) {
            ++v;
        }
        // per-record verification: every disk record starting inside the LCP must match a
        // request record at EXACTLY its start index — id, shape and type. Iterating the
        // records (never "the next chunk after") verifies each image of an adjacent pair
        // separately; a missing request record or an unequal field is a hard mismatch that
        // truncates the LCP to that record's start — same text + a different image reuses
        // exactly the pre-image prefix. Both sides are ordered by start_idx, and verified
        // records align the chunk boundaries inside the LCP, so the truncated v can only
        // fall on a text cell or a chunk start (chunk-safe by construction).
        for (const auto & rec : disk_media) {
            if ((size_t) rec.start_idx >= v) {
                break; // this and all later records start outside the verified prefix
            }
            const auto it = std::lower_bound(req_media.begin(), req_media.end(), rec.start_idx,
                [](const server_media_record & r, uint32_t s) { return r.start_idx < s; });
            const bool match = it != req_media.end()      &&
                               it->start_idx == rec.start_idx &&
                               it->id        == rec.id        &&
                               it->n_tokens  == rec.n_tokens  &&
                               it->n_pos     == rec.n_pos     &&
                               it->nx        == rec.nx        &&
                               it->ny        == rec.ny        &&
                               it->is_audio  == rec.is_audio;
            if (!match) {
                v = rec.start_idx;
                break;
            }
        }
        // Only WHOLE-block prefixes are valid reuse lengths (hash boundaries).
        const int B = params_base.slot_save_block;
        int n_keep_disk;
        if (restore_is_whole_prefix_only()) {
            // FULL, RS and SWA can only ever resume a snapshot as a WHOLE prefix (see
            // restore_is_whole_prefix_only): do_slot_restore loads the ENTIRE L-token snapshot, and a
            // later keep_first(n_past < L) would issue a partial common_context_seq_rm that either
            // GGML_ABORTs (FULL refuses any partial range; RS refuses a rollback beyond n_rs_seq) or
            // attends over cells that were never persisted (SWA keeps only [L - n_swa_mem, L)).
            // So auto-restore this snapshot ONLY when the request diverges at or beyond its end
            // (v == disk_toks.size(), i.e. the request strictly extends it, or matches it exactly).
            // No block-boundary clamp here: the whole snapshot is the only legal restore length.
            // Refuse BEFORE the multi-GB read so the candidate loop falls through to a SHORTER snapshot
            // that IS a whole prefix of this request (e.g. the mid-prefill context base).
            if (v != disk_toks.size()) {
                SLT_DBG(slot, "auto-restore: snapshot is not a whole prefix of the request "
                              "(verified %zu of %zu snapshot tokens; request %zu, seq_rm_type = %d, n_swa = %d) - skipping %s\n",
                        v, disk_toks.size(), req.size(), (int) ctx_tgt_seq_rm_type, n_swa_mem, cand.state_path.c_str());
                return 0;
            }
            n_keep_disk = (int) disk_toks.size();
        } else {
            // Reached only for PART with n_swa_mem == 0 (plain attention), which supports per-token
            // partial seq_rm, so a mid-snapshot divergence is fine: claim the verified prefix clamped
            // down to the last whole block boundary <= v. That is now exactly what the branch above
            // lets through. An exact full-snapshot match keeps the whole snapshot length.
            if (v == disk_toks.size()) {
                n_keep_disk = (int) disk_toks.size();
            } else {
                n_keep_disk = (int) (v - (v % (size_t) B));
                // shared chunk-boundary rule (same predicate the hash sites emit boundaries
                // with): a reuse length may not split a media chunk, so step down block by
                // block until the cut is chunk-safe.
                while (n_keep_disk > 0 &&
                       !boundary_is_chunk_safe(disk_toks, disk_media, (size_t) n_keep_disk)) {
                    n_keep_disk -= B;
                }
            }
        }
        if (n_keep_disk <= 0) {
            return 0;
        }
        // MARGIN gate (invariant 5): only pay a multi-GB load if disk strictly beats the
        // in-memory match by at least one block — never thrash a reload to save a few tokens.
        if (n_keep_disk < n_keep_mem + B) {
            return 0;
        }
        // ABSOLUTE restore floor (--slot-restore-min-tokens, default 0 = off): skip the multi-GB
        // disk load and reprocess instead when the byte-verified, block-aligned matched prefix is
        // below the floor — for a near-cold slot reprocessing a tiny prefix beats paying the NVMe
        // read + H2D copy. Gated on n_keep_disk (the ACTUAL verified match), NOT cand.n_tokens (the
        // snapshot length, which can far exceed the match). Placed after the relative MARGIN gate
        // (which handles "is disk worth more than the resident match") and before the only multi-GB
        // read: return 0 falls through to the caller recomputing n_past + a cold prefill (invariant 4).
        if (n_keep_disk < params_base.slot_restore_min_tokens) {
            return 0;
        }
        // Build the root->tip chain of node .bin paths via the shared walker. Done BEFORE touching
        // the slot so any inconsistency (missing/corrupt/non-contiguous node) simply returns 0 for a
        // cold prefill, never disturbing the resident KV (invariant 4). `disk_toks` (the tip's full
        // [0, range_hi) record) is the byte-authority each parent prefix is verified against; it is
        // already byte-verified against the request up to `v`, and the compare keeps the legitimate
        // PART mid-parent divergence case restorable. Identical composition for v3 text and v4 media
        // tips, and shared with the manual /slots restore path.
        std::vector<std::string> chain;
        if (!auto_build_restore_chain(cand.state_path, disk_toks, disk_parent_id, disk_range_lo, chain)) {
            return 0;
        }
        // Clear the slot's resident KV before loading the snapshot (mirror the restore-continue safe
        // fallback): seq removal + token/checkpoint clear so the restore writes into an empty seq.
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, -1, -1);
        slot.prompt.tokens.clear();
        slot.prompt.checkpoints.clear();

        if (!do_slot_restore(slot, chain)) {
            // restore failed -> slot seq already cleared by do_slot_restore; caller reprefills (invariant 4).
            return 0;
        }
        if (!disk_media.empty()) {
            // do_slot_restore filled prompt.tokens with the RAW disk cells: for a media unit
            // those contain LLAMA_TOKEN_NULL cells with no live chunks behind them, which no
            // downstream consumer can traverse. Rebuild the prompt from the REQUEST instead:
            // the verified prefix [0, v) is cell-identical to the snapshot's, and the
            // request's clone carries the live chunks (with pixels) for exactly those cells
            // — the persisted embeddings are already inside the loaded KV state, so no
            // pixel ever needs to come from disk. v is chunk-safe (see the per-record
            // verification above), so keep_first cannot cut mid-image.
            bool rebuilt_ok = false;
            try {
                server_tokens rebuilt = req.clone();
                rebuilt.keep_first(v);
                slot.prompt.tokens = std::move(rebuilt);
                rebuilt_ok = slot.prompt.tokens.validate(ctx_tgt);
            } catch (const std::exception & e) {
                SLT_WRN(slot, "auto-restore: prompt rebuild failed, %s\n", e.what());
            }
            if (!rebuilt_ok) {
                // safety-clear tripwire: a NULL cell without a live chunk (or an unsafe cut)
                // is impossible by construction here — if it happens anyway, drop the
                // restored state entirely and fall back to a clean cold prefill rather than
                // let a later find_chunk() throw mid-decode (invariant 4).
                SLT_WRN(slot, "%s", "auto-restore: rebuilt prompt failed validation; clearing restored state\n");
                auto_restore_drop(slot);
                return 0;
            }
        }
        // SWA models (PART seq_rm, n_swa > 0): the downstream checkpoint search refuses any
        // checkpoint whose pos_max exceeds the request's pos_next, and a fresh process has no
        // checkpoints at all — without one it forces a full re-process, silently discarding the
        // restore. Trim the restored seq to the verified prefix NOW (the snapshot may extend past
        // the request, e.g. by generated tokens) and reconstruct a checkpoint at that boundary,
        // mirroring the FULL branch inside do_slot_restore (which never needs the trim: FULL only
        // restores whole-snapshot extend-matches). Non-SWA attention models skip the checkpoint
        // machinery entirely and need none of this.
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART && n_swa_mem > 0) {
            // (this trim is dead since the whole-prefix gate above forces v == disk_toks.size();
            // kept as a belt-and-braces guard. It was already dead before the gate was merged, the
            // old SWA branch forced the same equality.)
            if (v < disk_toks.size()) {
                if (disk_media.empty()) {
                    slot.prompt.tokens.keep_first(v); // media prompts were already rebuilt to exactly v cells
                }
                const llama_pos p0 = slot.prompt.tokens.pos_next();
                llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, p0, -1);
            }
            const auto ckpt_pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
            const auto ckpt_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
            if (ckpt_pos_min < 0) {
                // the trim emptied the SWA cache: the request diverges more than the SWA window
                // before the snapshot end, so the loaded window holds no position at or below the
                // verified prefix (for an iswa memory, seq_pos_min reports the SWA cache). Keeping
                // the restore would report n_past = v > 0 over an empty window and trip the
                // downstream pos_min==-1 GGML_ABORT — drop it and cold-prefill instead (invariant 4).
                SLT_WRN(slot, "%s", "auto-restore: verified prefix is outside the snapshot's SWA window; clearing restored state\n");
                auto_restore_drop(slot);
                return 0;
            }
            slot.prompt.checkpoints.clear();
            create_checkpoint(slot, 0, ckpt_pos_min, ckpt_pos_max);
        }
        // do_slot_restore loaded the snapshot. For every restore_is_whole_prefix_only() class (FULL, RS
        // and SWA) n_keep_disk == snapshot length (gated above), so the existing regenerate /
        // suffix-reuse path takes over with no partial rewind. For plain-attention PART the request may
        // diverge inside the snapshot; keep_first(n_past) + a PARTIAL seq_rm then reprefills the
        // divergent tail (supported for PART). The verified prefix is what we claim as reused.
        // Bump every node on the chain's mtime so the LRU treats a reused-but-not-rewritten base (and
        // each shared delta) as recently-used (true LRU, not least-recently-written) — critical for
        // the fan-out case where many requests restore one hot base prefix, and so eviction keeps the
        // whole live chain warm. Best-effort; never errors the restore (invariant 4).
        for (const std::string & node_path : chain) {
            auto_touch_unit(node_path);
        }
        SLT_INF(slot, "auto-restore: reused %d tokens from disk (in-memory match was %d), file=%s\n",
                n_keep_disk, n_keep_mem, cand.state_path.c_str());
        return n_keep_disk;
    }

    // AUTO-SAVE: persist a slot's KV before it is discarded, keyed by its token-prefix block hash.
    // Text-only prompts publish v1 .meta sidecars byte-identical to the pre-media format; media
    // prompts publish v2 sidecars carrying per-chunk identity records (the KV state file already
    // holds the embeddings, so identity metadata is all the disk side needs).
    // Skips redundant writes (an equal-or-longer snapshot already covers this prefix), writes the
    // state + .logits + .meta as a 3-file unit (atomically, .meta LAST so a torn write is never
    // indexed), enforces the bounded LRU, then reconciles the index. Invariant 1: first statement
    // is the gate; invariant 5: only called on slot release/reassign, never during generation.
    // SHARED atomic-publish tail, factored out of auto_save_slot_if_useful so the temp->fsync->
    // rename (meta last) publish invariant, the capacity pre-flight and the per-boundary index
    // insert live in ONE place. Persists KV cells [lo, hi) of slot.id's sequence as one disk unit
    // named auto_state_filename(hash, hi):
    //   - a ROOT snapshot when lo == 0 (v1 text / v2 media, `media` selecting which); either the
    //     WHOLE prompt (hi == toks.size(), byte-identical to the pre-refactor save_file path) or a
    //     PARTIAL [0, hi) root — the shared-context checkpoint — persisted via the range API;
    //   - a v3 DELTA node when lo > 0 (lo == parent_hi): cells [lo, hi=N) parented on `parent_id`.
    // `hash` is the chain hash that names the file and commits the prefix; the index is populated at
    // boundaries bhs[0..kb] inclusive. Behaviour-preserving for the whole-save and delta callers.
    void auto_publish_snapshot(server_slot & slot,
                               llama_context * ctx,
                               const llama_tokens & toks,
                               int32_t lo,
                               int32_t hi,
                               uint64_t hash,
                               const std::vector<uint64_t> & bhs,
                               size_t kb,
                               const model_fp & fp,
                               const std::vector<server_media_record> & media = {},
                               uint64_t parent_id = 0) {
        bool     is_node   = lo > 0;      // lo > 0 <=> a delta parented at parent_hi == lo
        uint32_t parent_hi = (uint32_t) lo; // both cleared below if the U6 delta cell-count check fails
        // the snapshot's own token prefix [0, hi): equals `toks` for a whole/delta save (hi == N),
        // a strict prefix for a partial-root checkpoint. Avoid the copy in the common hi == N path.
        const llama_tokens   snap_owned = ((size_t) hi == toks.size())
                                          ? llama_tokens{}
                                          : llama_tokens(toks.begin(), toks.begin() + hi);
        const llama_tokens & snap_toks  = ((size_t) hi == toks.size()) ? toks : snap_owned;

        // capacity pre-flight (statvfs via std::filesystem::space): refuse to START a multi-GB
        // write the filesystem cannot hold — on btrfs an ENOSPC mid-write can flip the whole
        // filesystem read-only, a far worse failure than a skipped opportunistic save. Exact
        // state size + the token array, with 10% slack covering the file header and the
        // .logits/.meta sidecars. An unanswerable space query skips too (conservative;
        // invariant 4: a skipped save never affects generation).
        {
            const size_t sz_state = llama_state_seq_get_size(ctx, slot.id);
            const size_t sz_need  = sz_state + snap_toks.size() * sizeof(llama_token);
            std::error_code sec;
            const auto sinfo = std::filesystem::space(params_base.slot_save_path, sec);
            if (sec || sinfo.available < sz_need + sz_need / 10) {
                SLT_DBG(slot, "auto-save: skipped, insufficient free space (need %zu bytes + 10%% slack, available %zu)\n",
                        sz_need, sec ? 0 : (size_t) sinfo.available);
                return;
            }
        }

        const std::string fname = auto_state_filename(hash, snap_toks.size());
        // cross-process atomicity: the temp path MUST be unique per writer. The final
        // name (fname) is deterministic (fp + chain hash + tok count), so two processes sharing one
        // --slot-save-path would otherwise both stream a multi-GB state into the SAME "<fname>.tmp"
        // and interleave -> a corrupt temp gets renamed over a good final file. We disambiguate the
        // temp with pid + a per-process monotonic counter, so each writer owns its own complete temp
        // and the deterministic-name rename is the ONLY shared, atomic step (idempotent: identical
        // content). The sidecar temps derive from this same unique base so they are unique too.
        // (nonce is atomic so it stays correct if save I/O is later threaded.)
        static std::atomic<uint64_t> s_tmp_nonce{0};
        const uint64_t nonce = s_tmp_nonce.fetch_add(1, std::memory_order_relaxed);
        const std::string tmp = fname + "." + std::to_string((long) getpid()) + "." +
                                std::to_string(nonce) + ".tmp";

        // 1) write the state to a per-writer-unique temp path (atomic via rename below). NOTE:
        //    llama_state_seq_save_file writes in place, so we write to the unique temp then rename — a
        //    crash mid-write never leaves a corrupt state file the index would trust.
        //    A DELTA node (lo > 0) writes only cells [lo, N) via the range save; a PARTIAL root
        //    (checkpoint, hi < N) writes cells [0, hi) via the range save; the WHOLE root takes the
        //    byte-identical save_file path.
        size_t nwrite;
        // P0.1: a class that does not honour ranges must never publish a delta. Probe once, here,
        // where real resident state exists and the range save is about to happen anyway.
        if (is_node && delta_capable == delta_cap::no) {
            return; // fail closed: this instance only writes whole roots
        }
        if (is_node) {
            // U5 (mm-delta decision 1): the range save filters by POSITION, so the boundary is
            // slot.prompt.tokens.pos_next(parent_hi) — the same function that assigned the cell
            // positions (mtmd decode seeds on pos_next). For text pos_next == parent_hi (byte-identical
            // delta); for media the two differ and pos_next is the correct boundary.
            nwrite = llama_state_seq_save_file_range(ctx, tmp.c_str(), slot.id,
                                                     slot.prompt.tokens.pos_next((llama_pos) lo), -1,
                                                     snap_toks.data(), snap_toks.size());
        } else if ((size_t) hi == toks.size()) {
            nwrite = llama_state_seq_save_file(ctx, tmp.c_str(), slot.id, snap_toks.data(), snap_toks.size());
        } else {
            nwrite = llama_state_seq_save_file_range(ctx, tmp.c_str(), slot.id,
                                                     0, (llama_pos) hi, snap_toks.data(), snap_toks.size());
        }
        if (nwrite == 0) {
            std::error_code ec; std::filesystem::remove(tmp, ec);
            return; // invariant 4: disk full / IO error -> generation unaffected
        }

        // P0.1: resolve the probe on the FIRST delta write. Cost: one extra whole-sequence write,
        // once per instance. If the range was ignored the two byte counts match (the "delta" is a
        // whole save wearing a delta's .meta) and composing it would duplicate cells -> refuse, and
        // never attempt a delta again on this instance.
        if (is_node && delta_capable == delta_cap::unknown) {
            const std::string probe = tmp + ".probe";
            const size_t nwhole = llama_state_seq_save_file(ctx, probe.c_str(), slot.id,
                                                            snap_toks.data(), snap_toks.size());
            std::error_code pec; std::filesystem::remove(probe, pec);
            if (nwhole == 0) {
                std::error_code ec; std::filesystem::remove(tmp, ec);
                return; // could not probe; try again on the next save rather than guess
            }
            delta_capable = (nwrite < nwhole) ? delta_cap::yes : delta_cap::no;
            SRV_INF("auto disk cache: delta capability probed = %s (range %zu B vs whole %zu B)\n",
                    delta_capable == delta_cap::yes ? "YES" : "NO (whole roots only)", nwrite, nwhole);
            if (delta_capable == delta_cap::no) {
                std::error_code ec; std::filesystem::remove(tmp, ec);
                return;
            }
        }

        // P0.4: refuse to publish a snapshot with NO memory payload. llama_kv_cache::state_write and
        // state_read are SILENT NO-OPS when the cache is a shared view (`if (other) return;`,
        // src/llama-kv-cache.cpp:2125-2127 / :2204-2206) — as on LLM_ARCH_GEMMA4_ASSISTANT, whose
        // sub-caches both carry mem_other. llama_state_seq_save_file still returns nwrite > 0 (file
        // header + token array), so the existing nwrite==0 gate passes, the .meta claims N tokens,
        // and a later restore loads NOTHING while reporting success. Class-agnostic guard: a real
        // snapshot must be substantially larger than its own header + token array. (The early return
        // in the engine is CORRECT for a pure view sharing v_cells_impl — we refuse the empty
        // SNAPSHOT, never the model.)
        {
            const size_t hdr_and_toks = 256 + snap_toks.size() * sizeof(llama_token);
            if (nwrite <= hdr_and_toks) {
                SRV_WRN("auto disk cache: refusing to publish a snapshot with no memory payload "
                        "(%zu B for %zu tokens) - the memory type serialised nothing\n",
                        nwrite, snap_toks.size());
                std::error_code ec; std::filesystem::remove(tmp, ec);
                return;
            }
        }
        // 1b) U6 (mm-delta decision 4): a range-save delta selects suffix cells by POSITION, so a
        //     mid-chunk anomaly could leave the boundary value right yet silently drop/duplicate suffix
        //     cells — and restore's byte-verify is NULL-blind. Peek the delta .bin's serialized cell
        //     count (no multi-GB load) and require it to equal N - parent_hi. On any mismatch do NOT
        //     persist a corrupt delta: discard and fall back to a WHOLE save of this exact prefix
        //     (is_node cleared => the meta below is v1/v2 by `media`, byte-identical to a no-parent save).
        if (is_node) {
            uint32_t written_cells = 0;
            const bool ok = delta_bin_cell_count(tmp, snap_toks.size(), written_cells) &&
                            (size_t) written_cells == snap_toks.size() - parent_hi;
            if (!ok) {
                SLT_WRN(slot, "auto-save: delta cell-count check failed (expected %zu, got %u); "
                              "falling back to a whole snapshot\n",
                        snap_toks.size() - parent_hi, written_cells);
                std::error_code ec; std::filesystem::remove(tmp, ec);
                is_node   = false;
                parent_id = 0;
                parent_hi = 0;
                nwrite = llama_state_seq_save_file(ctx, tmp.c_str(), slot.id, snap_toks.data(), snap_toks.size());
                if (nwrite == 0) {
                    std::filesystem::remove(tmp, ec);
                    return; // invariant 4
                }
            }
        }
        // 2) regenerate logits sidecar on the temp path (FULL only, and only when the captured
        //    distribution provably belongs to this exact state — the same stamp check SLOT_SAVE uses).
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL &&
            slot.logits_last_n_tokens == (int32_t) snap_toks.size() && !slot.logits_last.empty()) {
            const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
            slot_logits_write(tmp, slot.logits_last, nv, (uint32_t) snap_toks.size());
        }
        // 3) meta sidecar on the temp path. Written but renamed LAST. A whole/partial ROOT writes the
        //    v1 meta (text-only, `media` empty => byte-identical) or the v2 meta (media identity
        //    records). A DELTA (is_node) passes the FULL `media` records so slot_meta_write's
        //    (is_node, media-empty) dispatch selects v3 for a text delta (byte-identical) and v4 for a
        //    media delta; the meta stays WHOLE ([0,N) tokens + full tiling) while the .bin holds only
        //    cells [lo, N), so restore's byte-verify is the exact v2 path and the .bin composes via NO_CLEAR.
        const bool meta_ok = is_node
            ? slot_meta_write(tmp, fp, snap_toks, hash, /*media=*/media, /*is_node=*/true,
                              parent_id, parent_hi, (uint32_t) snap_toks.size())
            : slot_meta_write(tmp, fp, snap_toks, hash, media);
        if (!meta_ok) {
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            std::filesystem::remove(slot_logits_sidecar_path(tmp), ec);
            std::filesystem::remove(slot_meta_sidecar_path(tmp), ec);
            return; // invariant 4
        }
        // 4) atomic publish: rename state first, then sidecars to their final names. .meta is the
        //    last to appear, so the startup scan (which keys on .meta) never sees a half-written unit.
        std::error_code ec;
        std::filesystem::rename(tmp, fname, ec);
        if (ec) {
            std::filesystem::remove(tmp, ec);
            std::filesystem::remove(slot_logits_sidecar_path(tmp), ec);
            std::filesystem::remove(slot_meta_sidecar_path(tmp), ec);
            return; // invariant 4
        }
        std::filesystem::rename(slot_logits_sidecar_path(tmp), slot_logits_sidecar_path(fname), ec);
        ec.clear();
        std::filesystem::rename(slot_meta_sidecar_path(tmp), slot_meta_sidecar_path(fname), ec);
        // the .meta is the scan key — a unit whose .meta never landed must NOT be
        // published. If the meta rename failed, the .bin is already in place but unindexable, so we
        // unlink the orphan .bin (and any leftover temps) and DO NOT insert into the in-memory index.
        // Leaving the .bin would waste disk and a restart scan would skip it anyway (no .meta).
        if (ec) {
            std::error_code rec;
            std::filesystem::remove(fname, rec);
            std::filesystem::remove(slot_logits_sidecar_path(fname), rec);
            std::filesystem::remove(slot_logits_sidecar_path(tmp), rec);
            std::filesystem::remove(slot_meta_sidecar_path(tmp), rec);
            return; // invariant 4: don't index a unit whose .meta (the scan key) never published
        }

        if (media.empty()) {
            SLT_INF(slot, "auto-save: persisted %zu tokens to %s\n", snap_toks.size(), fname.c_str());
        } else {
            SLT_INF(slot, "auto-save: persisted %zu cells incl. %zu media chunks to %s\n",
                    snap_toks.size(), media.size(), fname.c_str());
        }

        // index insert (bhs[0..kb] -> this snapshot), then bounded-LRU + reconcile.
        {
            std::lock_guard<std::mutex> lk(auto_idx.mtx);
            auto_cache_entry e{ fname, (uint32_t) snap_toks.size(), fp };
            for (size_t i = 0; i <= kb && i < bhs.size(); ++i) {
                auto_index_insert_locked(bhs[i], e);
            }
            auto_idx.indexed_files.insert(fname); // remember our own write so a refresh won't re-open it
        }
        // Eviction is opt-in and scoped to the auto cache. This hook already runs only under
        // auto_cache_enabled(), but the gate is stated at the call site too so the invariant
        // (a bounded, self-reaping store belongs to --slot-save-auto, never plain --slot-save-path)
        // is local to every enforce_limits caller.
        if (auto_cache_enabled() &&
            (params_base.slot_save_max_count > 0 || params_base.slot_save_max_bytes > 0)) {
            bool oversized = false;
            slot_save_enforce_limits(params_base.slot_save_path,
                                     params_base.slot_save_max_count,
                                     params_base.slot_save_max_bytes,
                                     fname, oversized);
        }
        // Reconcile index with what the LRU kept (ours or a peer's) AND adopt the post-write dir
        // mtime as our scan baseline — both under ONE lock. Re-baselining here means OUR OWN
        // save+evict does not make the next lookup think a PEER changed the dir (which would force a
        // redundant full re-scan); a real peer write afterwards bumps the mtime again -> still
        // detected. CALLER holds no lock here.
        {
            std::lock_guard<std::mutex> lk(auto_idx.mtx);
            auto_index_drop_missing_locked();
            std::error_code mec;
            const auto dmt = std::filesystem::last_write_time(params_base.slot_save_path, mec);
            if (!mec) {
                auto_idx.dir_mtime = dmt;
            }
        }
    }

    // MID-PREFILL SHARED-CONTEXT BASE (Option A): persist the leading shared preamble [0, B_ctx) ONCE
    // as a deduplicated WHOLE-state v1 ROOT, so N chats sharing that prefix each restore this base via
    // the existing longest-prefix restore instead of re-prefilling it (and, with --slot-save-incremental,
    // collapse their own save to a small [B_ctx, N) delta parented on it — the incremental parent-find
    // discovers it for free). Called from update_slots() at the exact instant a COLD-prefilling slot has
    // decoded EXACTLY cells [0, B_ctx) and NOTHING beyond (armed via slot.ctx_save_pos at prompt start,
    // clamped there by the prefill loop). Because the resident sequence IS the true whole state at
    // B_ctx, the whole-save (auto_publish_snapshot lo==0, hi==toks.size() -> llama_state_seq_save_file
    // whole-root branch) serialises the correct recurrent/attention state — SOUND for dense, SWA AND
    // recurrent/hybrid. This is why there is NO model-class gate here (the old idle-flush [0,B) sub-range
    // checkpoint, taken with the slot sitting at N, mislabelled the state-after-N as a B-length prefix
    // and had to be hard-gated to PART && n_swa == 0; the mid-prefill whole-save removes that unsoundness
    // for every class). Every early return is a clean no-op (invariant 4). Remaining gates:
    //   - TEXT-ONLY: media makes the block-hash array sparse (auto_block_hashes only emits at chunk-safe
    //     boundaries), so bhs[B_ctx/block-1] would index the wrong prefix length. Arming already excludes
    //     media requests; re-checked here for safety.
    //   - LoRA-equal: the fingerprint captures the global LoRA set; refuse a base taken under a per-request
    //     adapter override (invariant 3), same guard as the whole-prefix save path.
    void auto_save_context_base(server_slot & slot) {
        if (!auto_cache_enabled()) {
            return; // off by default
        }
        if (slot.prompt.tokens.has_media()) {
            return; // text-only keeps bhs dense so the positional hash lookup below is exact
        }
        if (!are_lora_equal(slot.lora, params_base.lora_adapters)) {
            return; // fp captures the global LoRA set (invariant 3)
        }
        // B_ctx = the armed target. The slot is resident at EXACTLY B_ctx here (clamped there and this
        // batch decoded), so get_text_tokens() has length B_ctx and a whole-save serialises the whole
        // state at B_ctx. The two equalities below are guaranteed by the arm gates + clamp; re-checked
        // defensively so a spurious call can only no-op, never write a mislabelled prefix.
        const int32_t B_ctx = slot.ctx_save_pos;
        if (B_ctx <= 0) {
            return;
        }
        const llama_tokens toks = slot.prompt.tokens.get_text_tokens();
        if ((int32_t) toks.size() != B_ctx) {
            return; // defensive: the hook must fire with the slot resident at exactly B_ctx
        }
        const int B = params_base.slot_save_block;
        if (B <= 0 || (B_ctx % B) != 0) {
            return; // defensive: B_ctx is block-aligned by construction (arm uses boundary - boundary % B)
        }
        // dense (text) block-hash chain over [0, B_ctx); the boundary hash names the base.
        const auto bhs = auto_block_hashes(toks, {}, B, cur_fp.fp_model, cur_fp.fp_mmproj);
        const size_t kb = (size_t) (B_ctx / B) - 1;                  // bhs dense (text) => positional index OK
        if (kb >= bhs.size()) {
            return;                                                  // defensive: never index past the chain
        }
        const uint64_t ckpt_hash = bhs[kb];                         // NO re-hash: the boundary hash is in bhs
        // EXACT-LENGTH redundant-write dedup (not equal-or-longer): the 2nd..Nth chat sharing this
        // [0, B_ctx) preamble finds the base already published at this boundary and writes nothing.
        {
            std::lock_guard<std::mutex> lk(auto_idx.mtx);
            if (auto_index_has_exact_locked(ckpt_hash, (uint32_t) B_ctx)) {
                return;
            }
        }
        // Write cells [0, B_ctx) as a WHOLE-state v1 ROOT: lo == 0 and hi == B_ctx == toks.size() so
        // auto_publish_snapshot takes the llama_state_seq_save_file (whole-root) branch — NOT the range
        // branch — which serialises the true whole recurrent/attention state at B_ctx (text-only =>
        // media empty, parent_id 0). Same capacity pre-flight, pid+nonce temp, three-file temp->rename
        // (meta last) publish + per-boundary index insert as every other save.
        auto_publish_snapshot(slot, ctx_tgt, toks, /*lo=*/0, /*hi=*/B_ctx,
                              ckpt_hash, bhs, /*kb=*/kb, cur_fp);
    }

    void auto_save_slot_if_useful(server_slot & slot) {
        if (!auto_cache_enabled()) {
            return; // off by default
        }
        // exclusions reuse the existing guards. NOTE: an idle slot has already been reset(), so
        // `slot.task` is null here — the just-finished task survives as `slot.task_prev`. Use it for
        // the generative check (COMPLETION/INFILL only).
        const auto & wtask = slot.task ? slot.task : slot.task_prev;
        if (!wtask || !wtask->need_sampling()) {
            return;
        }
        // The fingerprint captures the GLOBAL LoRA set; refuse to persist a snapshot taken under a
        // per-request adapter override that differs from it (invariant 3). (Conservative: a future version
        // could fold the slot's adapters into the snapshot fingerprint instead.)
        if (!are_lora_equal(slot.lora, params_base.lora_adapters)) {
            return;
        }
        // media branch, gated on the PER-REQUEST has_media() (not the server-wide has_mtmd/mctx)
        // so an --mmproj server still persists its text-only turns as byte-identical v1 units.
        // A media prompt saves on exactly the same terms as a text one — including a
        // prompt+generation snapshot on a FULL-seq-rm model. A later conversation turn re-renders
        // the assistant history to byte-identical tokens (qwencode_prefix_stability, verified) and
        // so extends the WHOLE snapshot, which is the FULL restore condition — the same way a
        // text turn does. (An earlier gate refused such media snapshots on the belief they were
        // unrestorable; that was wrong and silently denied multi-turn image conversations any disk
        // cache, so it is gone: media is aligned with text.)
        const bool prompt_has_media = slot.prompt.tokens.has_media();
        std::vector<server_media_record> media;
        if (prompt_has_media) {
            try {
                media = slot.prompt.tokens.extract_media_records();
            } catch (const std::exception & e) {
                // identity-less chunk (e.g. a placeholder bitmap): it can never be re-verified
                // against a future request, so the snapshot must not be persisted (invariant 4)
                SLT_WRN(slot, "auto-save: skipped, %s\n", e.what());
                return;
            }
        }
        // Text prompts: get_text_tokens() (not get_tokens()) — media-safe accessor that never
        // trips the get_tokens() GGML_ASSERT(!has_mtmd) under mmproj. For a no-media prompt it
        // equals the full token-id prefix, so the persisted token stream and the block-hash key
        // are byte-identical to what a text-only server would write.
        // Media prompts: get_cell_tokens() — the cell-aligned list (media cells
        // LLAMA_TOKEN_NULL) whose length equals the KV cell count llama_state_seq_save_file
        // persists; `media` tiles its NULL cells exactly.
        const llama_tokens toks = prompt_has_media ? slot.prompt.tokens.get_cell_tokens()
                                                   : slot.prompt.tokens.get_text_tokens();
        if (!media.empty()) {
            // chunk-completeness tripwire: process_mtmd_chunk() only ever appends whole chunks
            // to prompt.tokens (and keep_first() refuses mid-chunk cuts), so a prompt cannot
            // end inside a media chunk. The records must tile the cell list they are stored
            // with — slot_meta_read rejects the unit otherwise.
            GGML_ASSERT((size_t) media.back().start_idx + media.back().n_tokens <= toks.size());
        }
        // effective floor = max(block, min-tokens): a snapshot must cover >= 1 block AND clear the
        // configured minimum-size threshold. A trivially small prefix saves little prefill against
        // the state-file write + later restore, so it is skipped. Applies to text and media alike.
        const int save_floor = std::max(params_base.slot_save_block, params_base.slot_save_min_tokens);
        if ((int) toks.size() < save_floor) {
            return; // below the floor: not worth a multi-GB write
        }
        // media-aware chain: text cells contribute their token ids (bit-identical to the
        // pre-media chain for a text-only prompt), media cells their record identity; only
        // chunk-safe block boundaries are emitted, so a media prompt with no safe boundary
        // yields an empty chain and is skipped below.
        const auto bhs = auto_block_hashes(toks, media, params_base.slot_save_block,
                                           cur_fp.fp_model, cur_fp.fp_mmproj);
        if (bhs.empty()) {
            return;
        }
        // NOTE: the shared-context preamble base is NO LONGER written here at idle-flush. It is now
        // whole-saved MID-PREFILL (Option A: auto_save_context_base, fired from update_slots when a cold
        // slot is resident at exactly the block-aligned first-user boundary B_ctx), which is sound for
        // dense, SWA AND recurrent/hybrid. The old idle-flush [0,B_ctx) sub-range checkpoint — unsound
        // for FULL/RS and SWA and therefore a no-op on the production qwen3.6 (hybrid) — is gone.
        const uint64_t full_hash = bhs.back(); // commits the whole whole-block prefix
        {
            std::lock_guard<std::mutex> lk(auto_idx.mtx);
            auto it = auto_idx.by_boundary.find(full_hash);
            if (it != auto_idx.by_boundary.end()) {
                for (const auto_cache_entry & c : it->second) {
                    // Whole-prefix classes (FULL, RS, SWA): only an EXACT-length snapshot substitutes,
                    // a longer one can never be rewound to this prefix and a shorter one is a different
                    // resume point, otherwise this save would be suppressed by a snapshot the model can
                    // never restore. PART with n_swa_mem == 0: an equal-or-longer snapshot already
                    // covers this prefix because it can rewind to it.
                    if (restore_is_whole_prefix_only() ? (c.n_tokens == toks.size()) : (c.n_tokens >= toks.size())) {
                        return; // a usable snapshot for this exact prefix already exists
                    }
                }
            }
        }

        // INCREMENTAL SAVE: when --slot-save-incremental, write only the KV cells added since the
        // deepest already-saved snapshot on this branch (a v3 text / v4 media delta node) instead of
        // re-D2H'ing and re-writing the whole prefix. Find the deepest candidate whose persisted cells
        // are a STRICT prefix of this prompt under the same fingerprint; the delta .bin then holds
        // cells [parent_hi, N). The parent-find scans the FULL per-boundary candidate vectors (SIZE_MAX,
        // not the RESTORE 4-cap): the real parent is the SHORTEST snapshot at the deepest SHARED
        // boundary (e.g. a pinned base whose bucket also holds N longer divergent siblings), so a 4-cap
        // hides it and forces a whole (~1.77GB) save instead of a small delta. auto_index_lookup orders
        // candidates deepest-boundary-first / longest-first, so the FIRST strict-prefix match is the
        // deepest parent. Flag off (or no parent found) => the EXACT whole-save path below (byte-
        // identical). Everything after this (nonce temp, logits sidecar, temp+rename publish, index
        // insert, LRU) is SHARED between both modes.
        //
        // TEXT AND MEDIA UNIFY here (decision 1/3): a media prompt now takes exactly the same parent-
        // find as a text one — the cell-token byte compare, the per-record parent verify and the
        // media-aware parent hash all reduce to the pre-media text behaviour when `media` is empty. A
        // text prompt therefore emits a byte-identical v3 delta; a media prompt emits a v4 delta. The
        // parent may itself be text (v1) OR media (v2/v4): a text base can parent a media delta (an
        // image added in the new turn) and vice-versa — the media-aware hash keys the exact parent
        // file for either kind (decision 6).
        bool     have_parent = false;
        uint64_t parent_id   = 0;
        uint32_t parent_hi   = 0;
        if (params_base.slot_save_incremental) {
            // `toks`/`media` are this prompt's cell tokens + records; look up against the live prompt.
            for (const auto_cache_entry & cand : auto_index_lookup(slot.prompt.tokens, /*max_attempts=*/SIZE_MAX)) {
                model_fp     disk_fp;
                llama_tokens disk_toks;
                std::vector<server_media_record> disk_media;
                if (!slot_meta_read(cand.state_path, cur_fp.fp_mmproj, disk_fp, disk_toks, disk_media)) {
                    continue; // unreadable meta -> not a usable parent (invariant 4)
                }
                if (!(disk_fp == cur_fp)) {
                    continue; // invariant 3
                }
                // STRICT prefix of the CELL tokens (media cells LLAMA_TOKEN_NULL on both sides):
                // disk_toks == toks[0:disk_toks.size()] AND disk_toks.size() < toks.size() (a delta
                // must add at least one cell; an equal/longer snapshot is not a parent here). A media
                // parent's NULL cells never equal a text prompt's real tokens, so a text prompt can
                // never pick a media parent — the text path is unchanged.
                if (disk_toks.size() >= toks.size() ||
                    !std::equal(disk_toks.begin(), disk_toks.end(), toks.begin())) {
                    continue;
                }
                const size_t parent_hi_sz = disk_toks.size();
                // the delta .bin covers cells [parent_hi, N); the boundary may not split a media chunk
                // (a chunk's cells + M-RoPE positions must all live in one .bin), so refuse a candidate
                // whose end cuts one. Always true on the text path (no media chunks in `media`).
                if (!boundary_is_chunk_safe(toks, media, parent_hi_sz)) {
                    continue;
                }
                // per-record parent verify over the shared prefix [0, parent_hi): every candidate
                // record starting inside the parent must equal THIS prompt's record at that exact start
                // index (id, shape, type). Media cells are NULL==NULL so the byte compare above is
                // identity-blind; this restores the byte backstop a text delta gets for free — without
                // it a ~2^-64 filename-hash collision between same-text/different-image prefixes could
                // compose the WRONG parent KV. Reuses the restore-path match block; a no-op for a text
                // delta (disk_media empty) (decision 5).
                bool records_ok = true;
                for (const auto & rec : disk_media) {
                    if ((size_t) rec.start_idx >= parent_hi_sz) {
                        break; // this and all later records start outside the shared prefix
                    }
                    const auto it = std::lower_bound(media.begin(), media.end(), rec.start_idx,
                        [](const server_media_record & r, uint32_t s) { return r.start_idx < s; });
                    const bool match = it != media.end()             &&
                                       it->start_idx == rec.start_idx &&
                                       it->id        == rec.id        &&
                                       it->n_tokens  == rec.n_tokens  &&
                                       it->n_pos     == rec.n_pos     &&
                                       it->nx        == rec.nx        &&
                                       it->ny        == rec.ny        &&
                                       it->is_audio  == rec.is_audio;
                    if (!match) {
                        records_ok = false;
                        break;
                    }
                }
                if (!records_ok) {
                    continue;
                }
                // parent_id = the parent node's chain_hash = the last whole-block boundary hash of its
                // cell prefix under the MEDIA-AWARE hash (prefix cells + the records fully inside
                // [0, parent_hi)). Since disk_toks == toks[0:parent_hi], this reproduces the parent's
                // own full_hash, so auto_state_filename(parent_id, parent_hi) is exactly the parent's
                // file (the deterministic link the restore walk resolves). For a text delta prefix_media
                // is empty and this is bit-identical to the pre-media parent hash. The parent cleared
                // save_floor >= block, so its prefix has at least one boundary; guard defensively.
                const llama_tokens prefix(toks.begin(), toks.begin() + parent_hi_sz);
                std::vector<server_media_record> prefix_media;
                for (const auto & rec : media) {
                    if ((size_t) rec.start_idx + rec.n_tokens <= parent_hi_sz) {
                        prefix_media.push_back(rec);
                    }
                }
                const auto pbhs = auto_block_hashes(prefix, prefix_media, params_base.slot_save_block,
                                                    cur_fp.fp_model, cur_fp.fp_mmproj);
                if (pbhs.empty()) {
                    continue;
                }
                parent_hi   = (uint32_t) parent_hi_sz;
                parent_id   = pbhs.back();
                have_parent = true;
                break; // deepest (longest-first) strict-prefix parent
            }
        }

        // Publish through the SHARED atomic-publish helper: capacity pre-flight, pid+nonce temp,
        // three-file temp->rename (meta last) with orphan cleanup, per-boundary index insert + LRU.
        // Whole root: lo=0, hi=N => byte-identical save_file path. Delta (have_parent): lo=parent_hi>0
        // => v3 node cells [parent_hi, N). Both index every boundary (kb = bhs.size()-1).
        auto_publish_snapshot(slot, ctx_tgt, toks,
                              /*lo=*/ have_parent ? (int32_t) parent_hi : 0,
                              /*hi=*/ (int32_t) toks.size(),
                              full_hash, bhs, /*kb=*/ bhs.size() - 1, cur_fp,
                              /*media=*/ media, /*parent_id=*/ have_parent ? parent_id : 0);
    }

    // AUTO-SAVE (shutdown): persist every slot's warm KV on graceful terminate — the third
    // call site of auto_save_slot_if_useful, symmetric with the idle-flush and
    // get_available_slot sites. Called exactly once, from server_context::start_loop()
    // after the queue loop has exited: same (main) thread that owns all slot mutation,
    // update_slots quiescent, strictly before llama_backend_free() in the caller's
    // clean_up(). Invariant 1: gate first.
    //
    // Processing slots ARE saved (deliberately, unlike invariant 5's release-site caveat): the
    // queue loop has fully exited, so we sit on an update_slots() boundary where prompt.tokens ==
    // KV length for both prefill and generation (handle_last_sampled_token pushes the sampled
    // token and llama_decode writes its KV within one update_slots; the next, unprocessed token
    // is staged in slot.sampled, outside both prompt.tokens and the KV). A mid-prefill snapshot of
    // a long prompt is thus coherent AND the highest-value thing to keep — a re-run extends it and
    // skips the reprefill (both attention and recurrent). auto_save_slot_if_useful's own dedup +
    // LRU bound the only residual (a recurrent mid-generation snapshot a shorter re-request can't
    // rewind into — a safe, evictable write).
    //
    // Deadline-boxed: each slot's flush is a potentially multi-GB write, and the process is
    // typically inside a supervisor's stop window (systemd SIGKILLs at TimeoutStopSec) — an
    // unbounded flush loop trades a clean exit for evictable cache units. The deadline is
    // checked BETWEEN slots (an in-flight write is never aborted), so the worst case is
    // deadline + one write; README documents sizing TimeoutStopSec against
    // slot_save_max_mb x n_slots.
    static constexpr int64_t AUTO_SAVE_SHUTDOWN_DEADLINE_MS = 60 * 1000;
    void auto_save_slots_at_shutdown() {
        if (!auto_cache_enabled()) {
            return; // off by default
        }
        if (sleeping) {
            return; // sleep entry destroy()'d ctx_tgt; the warm KV is already gone
        }
        const int64_t t_deadline_ms = ggml_time_ms() + AUTO_SAVE_SHUTDOWN_DEADLINE_MS;
        for (size_t i = 0; i < slots.size(); i++) {
            if (ggml_time_ms() >= t_deadline_ms) {
                SRV_WRN("auto-save: shutdown flush deadline (%" PRId64 " ms) exceeded, skipping %zu remaining slots\n",
                        AUTO_SAVE_SHUTDOWN_DEADLINE_MS, slots.size() - i);
                break;
            }
            auto_save_slot_if_useful(slots[i]);
        }
    }

    // AUTO-SAVE (idle-delay): the two other save sites (get_available_slot reclaim, shutdown) only
    // fire on next-task-arrival or terminate, so a lone request's warm KV stays crash-volatile and
    // invisible to peer instances until more traffic lands. This closes that window to <=N seconds:
    // once a slot has been idle for slot_save_idle_seconds it is flushed like any other site (v2 for
    // media), on the main-loop thread, off the request hot path. Driven by the queue's timed idle
    // wait (auto_idle_next_deadline bounds the wait; auto_idle_flush runs the due flushes), so there
    // is no busy poll and it works with sleeping disabled.
    bool auto_idle_flush_enabled() const {
        return auto_cache_enabled() && params_base.slot_save_idle_seconds >= 0;
    }

    // Absolute ggml_time_ms() deadline of the earliest slot due for an idle flush, or -1 if none is
    // pending. Called by the queue loop to bound its idle wait. A slot with no prior task (never ran)
    // or already flushed for its current idle period contributes nothing.
    int64_t auto_idle_next_deadline() {
        if (!auto_idle_flush_enabled()) {
            return -1;
        }
        const int64_t idle_ms = (int64_t) params_base.slot_save_idle_seconds * 1000;
        int64_t earliest = -1;
        for (const auto & slot : slots) {
            if (slot.is_processing() || slot.auto_idle_flushed || !slot.task_prev || slot.t_last_used < 0) {
                continue;
            }
            const int64_t deadline = slot.t_last_used / 1000 + idle_ms; // t_last_used is microseconds
            if (earliest < 0 || deadline < earliest) {
                earliest = deadline;
            }
        }
        return earliest;
    }

    // Flush ONE idle slot whose idle deadline has passed, then return. Runs on the main-loop thread
    // from the queue's idle wait (no task in flight, so all slots are quiescent). Flushing a single
    // slot per wakeup keeps a request that lands mid-flush from stalling behind more than one
    // (potentially multi-GB) write: the queue loop re-checks for queued tasks between wakeups, and
    // any remaining due slots are flushed on immediately-following wakeups (their deadline has already
    // passed, so the wait returns at once). auto_save_slot_if_useful carries every correctness gate
    // and its dedup makes a repeat flush free.
    void auto_idle_flush() {
        if (!auto_idle_flush_enabled()) {
            return;
        }
        const int64_t now_ms  = ggml_time_ms();
        const int64_t idle_ms = (int64_t) params_base.slot_save_idle_seconds * 1000;
        for (auto & slot : slots) {
            if (slot.is_processing() || slot.auto_idle_flushed || !slot.task_prev || slot.t_last_used < 0) {
                continue;
            }
            if (now_ms - slot.t_last_used / 1000 < idle_ms) {
                continue; // not idle long enough yet
            }
            auto_save_slot_if_useful(slot);
            slot.auto_idle_flushed = true; // one attempt per idle period; a reclaim/next task re-arms it
            return;                        // at most one flush per wakeup, then back to service tasks
        }
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        params_base = params;
        const auto output_limits = server_output_limits(params_base);
        params_base.n_outputs_max = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // optionally get the memory usage of mmproj
        if (has_mmproj && params_base.fit_params) {
            int64_t t_start = ggml_time_us();
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            int64_t t_elapsed = ggml_time_us() - t_start;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // optionally reserve VRAM for the draft / MTP context before fitting the target model
        if (params_base.fit_params) {
            if (has_spec) {
                // MTP draft context lives on the target model, only context+compute are new
                bool measure_model_bytes = has_draft;

                common_params params_dft = common_base_params_to_speculative(params_base);

                auto mparams_dft = common_model_params_to_llama(params_dft);
                auto cparams_dft = common_context_params_to_llama(params_dft);
                if (spec_mtp) {
                    cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                }
                cparams_dft.n_rs_seq = 0;

                std::vector<ggml_backend_dev_t> devs;
                uint32_t hp_ngl = 0;
                uint32_t hp_nct = 0;
                uint32_t hp_nex = 0;
                try {
                    auto dmd = common_get_device_memory_data(
                        params_dft.model.path.c_str(), &mparams_dft, &cparams_dft,
                        devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);

                    GGML_ASSERT(!params_base.fit_params_target.empty());
                    size_t total = 0;

                    std::vector<ggml_backend_dev_t> tgt_devices = params.devices;

                    if (tgt_devices.empty()) {
                        for(size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                           tgt_devices.push_back(ggml_backend_dev_get(i));
                        }
                    }

                    for (size_t j = 0; j < devs.size(); ++j) {
                        const size_t bytes = (measure_model_bytes ? dmd[j].model : 0) + dmd[j].context + dmd[j].compute;
                        total += bytes;
                        for (size_t i = 0; i < tgt_devices.size(); i++) {
                            if (tgt_devices[i] == devs[j]) {
                                SRV_DBG("[spec] adding %.2f MiB to fit_params_target for device %s\n",
                                        bytes / (1024.0 * 1024.0), ggml_backend_dev_name(devs[j]));
                                params_base.fit_params_target[i] += bytes;
                                break;
                            }
                        }
                    }
                    SRV_TRC("[spec] estimated memory usage of %s is %.2f MiB\n",
                            has_draft ? "draft model" : "MTP context",
                            total / (1024.0 * 1024.0));
                } catch (const std::exception & e) {
                    SRV_WRN("[spec] failed to measure %s memory: %s\n",
                            has_draft ? "draft model" : "MTP context", e.what());
                }
            }
        }

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft;
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            // projector identity fingerprint (header-only read, ~ms): guards media KV
            // snapshots against projector swap/requant/dimension changes and is exposed
            // in /props. Computed whenever an mmproj is loaded — NOT gated on the auto
            // disk cache — precisely because /props reports it.
            if (!mmproj_header_fingerprint(mmproj_path, fp_mmproj)) {
                SRV_ERR("failed to fingerprint multimodal projector, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("multimodal projector fingerprint: %016" PRIx64 "\n", fp_mmproj);

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        // NOTE: --swa-full only enlarges the SWA cache; it does NOT disable masking.
        // llama_kv_cache_iswa still constructs kv_swa with hparams.n_swa/swa_type, and
        // llama_kv_cache::state_write masks on those members unconditionally — so saves stay
        // WINDOWED even under --swa-full. Zeroing n_swa here would disarm the disk-cache SWA
        // guards while the engine keeps writing windowed blobs, letting a restore claim a prefix
        // over a hole (silent wrong attention). Keep the scheduling value for the batch planner,
        // but keep a model-derived value for every save/restore soundness decision.
        n_swa      = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);
        n_swa_mem  = llama_model_n_swa(model_tgt);   // what the ENGINE masks with — never 0 for an SWA model

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        int n_ctx_slot = llama_n_ctx_seq(ctx_tgt);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot, n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot, params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.callback_on_reset = [this](const server_slot & slot) {
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(std::max(n_batch, params_base.n_parallel), n_embd);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        // AUTO disk prompt/KV cache (invariant 1): compute the model fingerprint and build the
        // longest-prefix index ONCE, header-only — but ONLY when the feature is enabled. When OFF
        // this is a single boolean test and nothing else (no fingerprint, no scan, no allocation).
        // The fingerprint alone is also needed whenever --slot-save-path is set: manual media
        // snapshots persist it in their v2 .meta sidecar and manual restores verify against it.
        if (auto_cache_enabled() || !params_base.slot_save_path.empty()) {
            cur_fp = auto_compute_fingerprint();
        }
        if (auto_cache_enabled()) {
            auto_index_scan();
            SRV_INF("auto disk prompt cache enabled: indexed %zu prefix boundaries from %s (block=%d)\n",
                    auto_idx.by_boundary.size(), params_base.slot_save_path.c_str(), params_base.slot_save_block);
        }

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });
        queue_tasks.on_idle_flush(
            [this]() { return auto_idle_next_deadline(); },
            [this]() { auto_idle_flush(); });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool enabled = it != params_base.default_template_kwargs.end();
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = float(lcp_len) / task.tokens.size();

                SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            f_sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            // Second auto-save site for when cache_idle_slots is OFF (the idle-flush path that calls
            // idle-flush loop where the primary auto_save runs). Here get_available_slot just picked `ret`
            // for a new task and `update_cache` signals its prior KV is about to be discarded, so we
            // persist it before the prompt_save/prompt_load below overwrites it. Mutually exclusive
            // with the primary site via !cache_idle_slots, so no double-save. Reads `update_cache`
            // BEFORE the `&& prompt_cache` narrowing so disk save works without --cache-ram. The
            // callee carries all correctness gates; `ret` is idle so this never stalls generation.
            if (auto_cache_enabled() && !params_base.cache_idle_slots && update_cache) {
                auto_save_slot_if_useful(*ret);
            }

            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear();
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear();

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // A new task is being assigned to this slot: its prompt may differ from whatever produced
        // the slot's current `logits_last` (even at the same token count). Invalidate the capture so
        // a SLOT_SAVE issued on the new prompt can never serialize a stale, mismatched distribution.
        // The stamp is re-established only by a real decode of the new prompt (the capture point).
        slot.logits_last.clear();
        slot.logits_last_n_tokens = -1;

        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool use_backend_sampling = task.params.sampling.backend_sampling;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());
        } else {
            slot.smpl.reset();
        }

        // stash the first-user-message boundary B for the [0,B) shared-context checkpoint onto the
        // persistent prompt, computed once here at task launch. The save site also runs at idle-flush
        // and shutdown, when the transient task may be gone, so it must never read the task for this —
        // it reads slot.prompt.ctx_boundary instead. -1 (no user span) makes the checkpoint no-op.
        slot.prompt.ctx_boundary = task.params.message_spans.first_user_message_pos();
        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        // slot.task is null when create_checkpoint is called from do_slot_restore (a restore has
        // no active task); use -1 so the restored checkpoint is simply not tied to a current task.
        const int id_task = slot.task ? slot.task->id : -1;

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
            if (it->id_task != id_task && last >= 0 && it->n_tokens <= last + params_base.checkpoint_min_step) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        it->pos_min, it->pos_max, it->n_tokens, (float) it->size() / 1024 / 1024);

                it = slot.prompt.checkpoints.erase(it);
                continue;
            }

            last = it->n_tokens;
            ++it;
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(spec.get(), slot.id, cur.data_spec);

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, an encode / decode is running and only accessing metrics is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS) {
            SRV_DBG("decoding, decline task, id_task = %d\n", task.id);
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                // Auto-save (write path): persist this idle slot's KV to disk before
                                // the in-memory cache drops it. Gated so the whole call frame is elided
                                // when the feature is OFF; the callee re-checks all correctness guards
                                // (fingerprint, need_sampling, LoRA, media identity, >=1 block).
                                if (auto_cache_enabled()) {
                                    auto_save_slot_if_useful(slot);
                                }
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json(slots_debug == 0);

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->metrics             = metrics;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    // per-slot media gate (a server-wide check_no_mtmd used to 501 every manual
                    // save under --mmproj, even for text-only slots): a text slot persists its
                    // plain token-id list, byte-identical to the pre-media format; a media slot
                    // persists its cell-aligned list (media cells LLAMA_TOKEN_NULL) plus a v2
                    // .meta sidecar carrying the per-chunk identity records — the same identity
                    // layer the auto disk cache uses.
                    const bool slot_has_media = slot->prompt.tokens.has_media();
                    std::vector<server_media_record> media;
                    if (slot_has_media) {
                        try {
                            media = slot->prompt.tokens.extract_media_records();
                        } catch (const std::exception & e) {
                            // identity-less chunk (e.g. a placeholder bitmap): it could never be
                            // re-verified or rehydrated, so the snapshot must not be persisted
                            send_error(task, std::string("cannot save slot: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
                            break;
                        }
                    }
                    const llama_tokens tokens = slot_has_media ? slot->prompt.tokens.get_cell_tokens()
                                                               : slot->prompt.tokens.get_text_tokens();
                    const size_t token_count = tokens.size();
                    const size_t nwrite = llama_state_seq_save_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), token_count);

                    // persist this slot's last-token logits as a sidecar (FULL/recurrent
                    // only). Best-effort — a missing/failed sidecar simply disables the regenerate
                    // fast-path for this snapshot. NOT folded into res->n_bytes (that contract stays
                    // "state-file bytes only").
                    //
                    // CRITICAL consistency guard: only write the sidecar when the captured logits
                    // provably belong to the EXACT state being saved, i.e. logits_last_n_tokens ==
                    // token_count. This blocks every stale-logits path (restore-then-save with no
                    // intervening decode; a spec-decode step that skipped the capture; a distribution
                    // left over from a prior task on this slot object) from persisting a sidecar that
                    // does not match the saved state — which would otherwise emit a wrong first token
                    // on a later regenerate with nothing to catch it.
                    if (nwrite > 0 && ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                        if (slot->logits_last_n_tokens == (int32_t) token_count && !slot->logits_last.empty()) {
                            const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
                            const size_t nwrite_logits =
                                slot_logits_write(filepath, slot->logits_last, nv, (uint32_t) token_count);
                            if (nwrite_logits == 0) {
                                SLT_WRN(*slot, "%s", "failed to write logits sidecar; regenerate fast-path disabled for this snapshot\n");
                            }
                        } else {
                            SLT_DBG(*slot, "no matching captured logits for this state (stamp=%d, token_count=%zu); sidecar omitted\n",
                                    slot->logits_last_n_tokens, token_count);
                        }
                    }

                    // media snapshots publish their identity sidecar (v2 .meta) next to the state
                    // file: a manual restore rebuilds the prompt's media chunks from it, since —
                    // unlike the auto-restore path — there is no request to rebuild from. Text
                    // snapshots keep the base on-disk shape (state file + optional .logits, no
                    // .meta). A media state file without its sidecar is unrestorable, so a failed
                    // sidecar write withdraws the whole unit and errors the save (never publish a
                    // unit that can only be half-loaded). chain_hash is 0: manual units carry
                    // user-chosen filenames and are never indexed by the auto cache — restore
                    // authority is the byte-compared tokens.
                    if (nwrite > 0 && slot_has_media) {
                        if (!slot_meta_write(filepath, cur_fp, tokens, 0, media)) {
                            std::error_code ec;
                            std::filesystem::remove(filepath, ec);
                            std::filesystem::remove(slot_logits_sidecar_path(filepath), ec);
                            send_error(task, "failed to write the .meta sidecar for a media slot snapshot", ERROR_TYPE_SERVER);
                            break;
                        }
                    }

                    // enforce the bounded slot-save store (LRU by mtime). If this single
                    // snapshot exceeds the byte cap, reject the save instead of evicting everything.
                    // Eviction is opt-in: it runs ONLY when --slot-save-auto owns this directory as a
                    // server-managed cache. A plain manual --slot-save-path save never deletes files.
                    if (nwrite > 0 && auto_cache_enabled() &&
                        (params_base.slot_save_max_count > 0 || params_base.slot_save_max_bytes > 0)) {
                        bool oversized = false;
                        slot_save_enforce_limits(params_base.slot_save_path,
                                                 params_base.slot_save_max_count,
                                                 params_base.slot_save_max_bytes,
                                                 filepath, oversized);
                        if (oversized) {
                            send_error(task,
                                       "slot snapshot exceeds --slot-save-max-mb; save rejected",
                                       ERROR_TYPE_INVALID_REQUEST);
                            break;
                        }
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    // Compose the restore chain (U9 / decision 2). Manual SAVE only ever writes a
                    // self-contained WHOLE unit (a v1/v2 snapshot, parent 0) — the guaranteed floor —
                    // so a unit this endpoint produced restores as a single clearing load exactly as
                    // before. But the same directory may hold an auto-cache v3/v4 DELTA tip whose .bin
                    // carries only cells [parent_hi, N); pointing a manual restore at one MUST NOT be
                    // refused (decision 2). Read its node tail and, when it is a delta, walk parent
                    // links via the SHARED auto_build_restore_chain so base + deltas compose to the
                    // full cell set before loading — the identical path the auto restore uses, so a v4
                    // media delta tip restores correctly here too. A missing .meta / whole snapshot
                    // (parent 0) keeps the single-file chain; only a genuinely broken delta chain
                    // (a parent file gone or corrupt) errors, the same way a corrupt whole unit would.
                    std::vector<std::string> restore_chain = { filepath };
                    {
                        model_fp     tip_fp;
                        llama_tokens tip_toks;
                        std::vector<server_media_record> tip_media;
                        uint64_t tip_parent_id = 0;
                        uint32_t tip_range_lo  = 0;
                        uint32_t tip_range_hi  = 0;
                        if (slot_meta_read(filepath, cur_fp.fp_mmproj, tip_fp, tip_toks, tip_media,
                                           &tip_parent_id, &tip_range_lo, &tip_range_hi) &&
                            (tip_parent_id != 0 || tip_range_lo != 0)) {
                            if (!auto_build_restore_chain(filepath, tip_toks, tip_parent_id, tip_range_lo,
                                                          restore_chain)) {
                                send_error(task, "Unable to restore slot: the delta snapshot's parent "
                                                 "chain is missing or inconsistent", ERROR_TYPE_INVALID_REQUEST);
                                break;
                            }
                        }
                    }

                    // Shared restore body (also used by the transparent auto-restore path): loads the
                    // composed chain into seq slot->id, sets just_restored + restored_logits, rebuilds
                    // the FULL-model checkpoint. On a load failure the slot seq is cleared and we error.
                    size_t token_count = 0;
                    size_t nread = 0;
                    if (!do_slot_restore(*slot, restore_chain, &token_count, &nread)) {
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    // media snapshots: rebuild the prompt's media chunks as stubs from the v2
                    // .meta sidecar — without them the restored NULL cells are untraversable.
                    // Any rehydration failure drops the loaded state entirely (a half-rehydrated
                    // slot must never survive to serve requests) and errors with the reason.
                    {
                        std::string err;
                        if (!manual_restore_rehydrate_media(*slot, filepath, err)) {
                            auto_restore_drop(*slot);
                            send_error(task, "cannot restore media snapshot: " + err, ERROR_TYPE_INVALID_REQUEST);
                            break;
                        }
                    }

                    // SWA models (PART seq_rm, n_swa > 0): reconstruct a checkpoint at the restored
                    // position, mirroring auto_restore_into_slot — the downstream checkpoint search
                    // finds none in a fresh process and would force a full re-process on the next
                    // request, silently discarding the restore. (FULL models get theirs inside
                    // do_slot_restore; non-SWA attention models skip the checkpoint machinery.)
                    if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART && n_swa_mem > 0) {
                        const auto ckpt_pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot->id);
                        const auto ckpt_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot->id);
                        if (ckpt_pos_min >= 0) {
                            slot->prompt.checkpoints.clear();
                            create_checkpoint(*slot, 0, ckpt_pos_min, ckpt_pos_max);
                        }
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    void update_slots() {
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");

                metrics_flush_idle();

                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
            batch.render();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        llama_batch batch_view;
        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch_view = batch.get_view(off, n_tokens);
                bool ok = decode(n_batch, off, batch_view);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off, batch_view);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }

        // ===== MID-PREFILL SHARED-CONTEXT BASE whole-save (Option A) =====================
        // Runs AFTER the decode loop, so every batch chunk has been llama_decode'd and the KV
        // cells for the tokens added in pre_decode() actually exist. A slot armed at B_ctx and now
        // resident at EXACTLY [0, B_ctx) (clamped by the prefill loop, still SLOT_STATE_PROCESSING_
        // PROMPT because B_ctx < N is a strict prefix) holds the true whole state at B_ctx: whole-
        // save it ONCE, then disarm so the slot prefills [B_ctx, N) normally. post_decode() is NOT a
        // viable hook — a still-prefilling slot has i_batch == -1 and is skipped by its
        // is_inside_view() early-return. Cost when nothing is armed: one cheap slot scan.
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT &&
                slot.ctx_save_pos > 0 &&
                slot.prompt.n_tokens() == slot.ctx_save_pos) {
                auto_save_context_base(slot);
                slot.ctx_save_pos = -1; // one-shot: neither re-clamp nor re-save this task
            }
        }
        // ===== end MID-PREFILL BASE =====================================================
    }

    void pre_decode() {
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .n_past   = */ slot.prompt.n_tokens(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        {
            common_speculative_draft(spec.get());
        }

        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.stats.n_draft_tokens += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
                    GGML_ABORT("failed to remove sequence %d\n", slot.id);
                }
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        });

        // update the batch with the sampled/drafted tokens
        iterate(generating, [&](server_slot & slot) {
            slot.handle_last_sampled_token(batch);
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing()) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.stats.update_prompt_start();

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // ===== AUTO-RESTORE: cold/cross-process KV reuse from disk (opt-in) ==========
                                // If the in-memory match (n_past) is POOR and the disk index holds a snapshot
                                // whose persisted tokens are a verified, fingerprint-matching, longer prefix of
                                // this request, restore it INTO the slot and RECOMPUTE n_past so all downstream
                                // machinery runs unchanged — agnostic to HOW the tokens arrived. Media requests
                                // look up exactly like text ones: the chain hashes fold each chunk's identity
                                // record, and auto_restore_into_slot byte-verifies the cells PLUS every media
                                // record (id/shape/type) before any multi-GB load, falling back to a normal
                                // prefill on any mismatch/failure (invariants 2/3/4). (An earlier revision
                                // gated this block on !slot.prompt.tokens.has_media() — which tested the
                                // slot's STALE PREVIOUS prompt, not the request; no request property replaces
                                // it, media requests are simply first-class now.)
                                if (auto_cache_enabled()
                                        && slot.task->need_sampling()        // generative only (not embed/rerank)
                                        && slot.alora_invocation_start <= 0      // aLoRA caching bound (mirror below)
                                        && are_lora_equal(slot.lora, params_base.lora_adapters)) { // fp captures global LoRA (invariant 3)
                                    // Try candidates best-first (longest usable snapshot first). Each
                                    // rejected candidate costs only a small .meta read + byte-compare; the
                                    // multi-GB state loads only once a candidate passes its gates. This
                                    // fall-through is what stops a longer superset snapshot (unusable by a
                                    // FULL model, which needs a whole-prefix match) from shadowing a shorter
                                    // usable one at the same boundary. Media requests look up first-class:
                                    // auto_index_lookup folds each chunk's identity into the boundary hashes.
                                    for (const auto & cand : auto_index_lookup(input_tokens)) {
                                        // auto_restore_into_slot may CLEAR the slot
                                        // (KV seq + prompt.tokens) and then have do_slot_restore FAIL
                                        // (corrupt/short .bin, KV-capacity exceeded, racing LRU eviction
                                        // deleting the file mid-read). In that case the slot tokens are now
                                        // empty. We therefore RECOMPUTE n_past UNCONDITIONALLY after any
                                        // attempt — not only on success — so a cleared-but-failed restore
                                        // falls back to n_past=0 (clean cold prefill) instead of carrying a
                                        // stale n_keep_mem>0 into keep_first() on an empty token vector
                                        // (which would GGML_ASSERT/abort). The recompute is harmless on the
                                        // early-return-before-clear paths (margin/fp/verify rejects): those
                                        // leave prompt.tokens untouched, so the LCP is identical to before.
                                        const int restored = auto_restore_into_slot(slot, cand, input_tokens, (int) n_past);
                                        n_past = slot.prompt.tokens.get_common_prefix(input_tokens);
                                        if (restored > 0) {
                                            break; // restored; stop trying shorter candidates
                                        }
                                    }
                                }
                                // ===== end AUTO-RESTORE =====================================================

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the largest pos_min required for a checkpoint to be useful
                            // llama_memory_hybrid[_iswa]::seq_pos_min() returns max(attn_min, recr_min), and the
                            // recurrent side keeps a single live cell whose pos is the LAST decoded position. So for
                            // any memory with a recurrent component pos_min is a TAIL, not an oldest-available
                            // position, and backing it off by n_swa makes the gate below fire even for a strict
                            // extension that needs no rewind at all. Only apply the back-off when pos_min is a real
                            // window start (PART, i.e. pure-SWA attention).
                            const bool pos_min_is_tail = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                                                         ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS;
                            const auto pos_min_thold = std::max(0, pos_next - (pos_min_is_tail ? 0 : n_swa) - (has_new_tokens ? 0 : 1));

                            // ===== restore-continue (regenerate fast-path) ==============================
                            // A just-restored FULL/recurrent slot receiving the EXACT restored tokens (no
                            // suffix) cannot rewind its memory: the normal path would re-decode into the
                            // already-occupied sequence and crash (the pure-recurrent gate at the relaxed
                            // checkpoint predicate below is FALSE for this case, so just_restored would never
                            // be consumed and [TAG_PROMPT_LOGITS] would decrement n_past and re-decode into
                            // the occupied sequence). Detect that case here, BEFORE the n_past>0 guard, and:
                            //   - if we have the saved next-token logits: emit the first token with NO decode,
                            //     keeping the full restored state, then continue normal autoregression;
                            //   - otherwise: fall back to a SAFE clear-then-reprefill (never crash).
                            // Gated so it is unreachable for non-recurrent models, with-suffix requests
                            // and non-generative slots. Media prompts DO reach it: the auto path never
                            // had a server-wide mtmd gate (and the manual /slots endpoints now gate
                            // per-slot), so a byte-identical media resend on a FULL-seq_rm model
                            // restores its whole snapshot and lands exactly here.
                            // The body is media-safe by construction:
                            // init_sampler skips LLAMA_TOKEN_NULL cells, and the speculative begin feeds
                            // get_text_tokens() (the media-safe accessor), never get_tokens(). With-suffix
                            // restore reuse is left entirely to the unchanged relaxed-predicate path below.
                            // NOTE: cache_prompt==false sets n_past=0 above, so this gate (which
                            // requires n_past == task->n_tokens()) is naturally not entered for a
                            // no-cache request; that case safely takes the normal full-clear +
                            // reprefill path (common_context_seq_rm at [p0,-1) empties the restored
                            // sequence first), so regenerate degrades to a cold reprefill, never a crash.
                            // No `n_past < n_ctx` clause: the fast path emits with NO decode so it
                            // needs no free context slot; a full-n_ctx no-suffix restore is handled
                            // here rather than falling through to a zero-token-added crash window.
                            if (slot.just_restored &&
                                ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL &&
                                slot.task->need_sampling() &&
                                slot.alora_invocation_start <= 0 &&
                                n_past == slot.task->n_tokens() &&
                                n_past == (int) slot.prompt.n_tokens()) {

                                slot.just_restored = false; // one-shot consume (this path owns it)

                                if (!slot.restored_logits.empty()) {
                                    // --- fast path: emit first token from saved logits, no decode ---
                                    slot.stats.n_prompt_cached    = n_past; // entire prompt "reused"
                                    slot.stats.n_prompt_processed = 0;      // prompt_n = 0 => observable reuse signal

                                    // prime the sampler over the full restored prompt (penalties/grammar
                                    // history), exactly as the normal DONE_PROMPT transition (init_sampler) would.
                                    slot.stats.n_gen = 0;
                                    slot.init_sampler();

                                    slot.state   = SLOT_STATE_GENERATING;
                                    slot.i_batch = -1;

                                    // rebuild the (cold, unsaved) draft context for the restored prompt,
                                    // mirroring the normal prompt-done transition.
                                    if (slot.can_speculate()) {
                                        common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                                    }

                                    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
                                    const llama_token id = common_sampler_sample_from_logits(
                                            slot.smpl.get(), slot.restored_logits.data(), nv, /*grammar_first=*/false);
                                    slot.restored_logits.clear(); // consumed

                                    common_sampler_accept(slot.smpl.get(), id, true);

                                    // mirror the generation accounting from the normal sample path
                                    // (upstream: stats.n_gen, then update_prompt_last() on the first
                                    // token, then update_gen_last()). No prompt metrics are queued:
                                    // this path processes zero prompt tokens by construction.
                                    const int64_t t_current = ggml_time_us();
                                    slot.stats.n_gen += 1;
                                    if (slot.stats.n_gen == 1) {
                                        slot.stats.update_prompt_last();
                                        slot.t_print_last = t_current;
                                        slot.n_gen_last   = 0;
                                    }
                                    slot.stats.update_gen_last();

                                    if (slot.task->params.stream) {
                                        // mirror the normal prompt-start streaming signal exactly so a
                                        // return_progress client still gets its initial 0% progress event
                                        if (slot.task->params.return_progress) {
                                            send_partial_response(slot, {}, true);
                                        } else {
                                            // signal HTTP to send the headers (200 status)
                                            send_partial_response(slot, {}, false, true);
                                        }
                                    }

                                    completion_token_output result;
                                    result.tok  = id;
                                    // inline of the accept_special_token lambda (defined later in this method,
                                    // out of scope here): keep special tokens iff the server allows specials
                                    // or the request explicitly preserves this token.
                                    const bool keep_special =
                                        params_base.special ||
                                        slot.task->params.sampling.preserved_tokens.find(result.tok) !=
                                            slot.task->params.sampling.preserved_tokens.end();
                                    result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, keep_special);
                                    result.prob         = 1.0f;

                                    // First-token logprobs (n_probs>0): the post-sampling variant reads the
                                    // candidate set (cur_p), which common_sampler_sample_from_logits leaves
                                    // populated — so we can serve it exactly as the normal path does. The
                                    // pre-sampling variant reads raw ctx logits at a decode index we bypass
                                    // here; idx=-1 is passed but populate_token_probs() only uses idx in that
                                    // branch, so we restrict the call to post_sampling to stay correct.
                                    if (slot.task->params.sampling.n_probs > 0 && slot.task->params.post_sampling_probs) {
                                        populate_token_probs(slot, result, /*post_sampling=*/true, params_base.special, /*idx=*/-1);
                                    }

                                    if (!process_token(result, slot)) {
                                        slot.print_timings();
                                        send_final_response(slot);
                                        metrics_on_prediction(slot);
                                        slot.release();
                                    }

                                    SLT_INF(slot, "%s", "restore-continue: emitted first token from saved logits (prompt_n=0)\n");
                                    return; // skip the rest of prompt-batch building for this slot; iterate() proceeds to the next
                                }

                                // --- safe fallback: no valid sidecar -> clear restored seq, then reprefill ---
                                // FULL models support full-sequence removal; clearing first guarantees the
                                // subsequent reprefill writes into an EMPTY sequence instead of re-decoding
                                // into the already-occupied restored state (which is the crash being fixed).
                                SLT_WRN(slot, "%s", "restore-continue: no saved logits; clearing restored state and re-prefilling\n");
                                llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, -1, -1);
                                slot.prompt.tokens.clear();
                                slot.prompt.checkpoints.clear();
                                n_past   = 0;
                                pos_next = 0; // mirror the do_reset path; keep the stale full-length value from leaking into the checkpoint-erase loop below
                                // fall through to the normal guard below with an empty sequence (safe)
                            }
                            // ===== end restore-continue =================================================

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                // Consume just_restored UNCONDITIONALLY, before the gate: the tail-aware threshold
                                // makes the gate false for the restore-plus-suffix case, and this flag is never
                                // cleared in server_slot::reset(), so a consume inside the gate leaks for the life
                                // of the slot.
                                const bool slot_was_restored = slot.just_restored; slot.just_restored = false;

                                // A just-restored WINDOW memory (PART, in practice SWA) that is being
                                // strictly EXTENDED needs no rewind at all, so skip the search below
                                // entirely. llama_kv_cache::state_write drops exactly the cells a live
                                // slot would have evicted, so a restored slot whose KV ends at n_past is
                                // equivalent to a live slot that has just decoded [0, n_past), and a live
                                // slot extending forward never consults a checkpoint. The search can only
                                // do harm here: the ONLY checkpoint such a slot has is the one
                                // auto_restore_into_slot synthesised from the LIVE window (after clearing
                                // the list), whose pos_min IS the window start L - n_swa, the same
                                // expression as pos_min_thold when has_new_tokens, so
                                // `cur.pos_min < pos_min_thold` is false BY CONSTRUCTION, do_reset fires
                                // and every SWA disk restore is loaded and then thrown away.
                                // The four terms, exactly: (1) this task consumed a restore; (2) the
                                // request strictly extends it, so no rewind is needed; (3) pos_min is a
                                // real window start, NOT a tail, so FULL and RS keep the unchanged path,
                                // because for them the search is what keeps the NOTE's GGML_ABORT
                                // unreachable; (4) the restored KV ends exactly at n_past, so nothing
                                // above it needs removing (a diverging manual /slots restore has
                                // n_past < prompt.n_tokens() and still takes the search).
                                // Inert for PART with n_swa == 0 (no checkpoints are ever created for it,
                                // and pos_min == 0 keeps the gate closed) and under --swa-full (n_swa == 0
                                // makes pos_min_thold == pos_next, gate closed).
                                // KNOWN GAP, deliberately not covered: the SWA EXACT-resend case
                                // (has_new_tokens false) keeps today's behaviour, checkpoint rejected by
                                // one position, cold reprefill. Covering it would need either an
                                // off-by-one relaxation of the acceptance test or widening this skip to a
                                // path a warm slot does not take, which is the class of change that
                                // produced this file's incident history. Extension is the case with
                                // production evidence.
                                const bool restored_extend_only = slot_was_restored && has_new_tokens &&
                                                                  !pos_min_is_tail &&
                                                                  n_past == slot.prompt.n_tokens();

                                if (pos_min >= pos_min_thold && !restored_extend_only) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            // NOTE: no `cur.pos_min == pos_min_thold` clause. With the tail-aware
                                            // threshold, pos_min_thold == pos_next for tail memories, so such a
                                            // checkpoint restores to p0 == cell.pos and the following partial seq_rm
                                            // on a recurrent cache with n_rs_seq == 0 returns false -> GGML_ABORT.
                                            return cur.pos_min == 0 || cur.pos_min < pos_min_thold;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        it->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        it->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        // restore the draft's speculative state
                                        common_speculative_set_state(spec.get(), slot.id, it->data_spec);

                                        pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                        SLT_TRC(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            // if the prompt ends with a media chunk, the decrement lands strictly inside
                            // it and keep_first(n_past) below would throw (latent upstream crash: reachable
                            // whenever a fully cached prompt ends with an image — a whole-snapshot disk
                            // restore hits it routinely). Clamp down to the enclosing chunk's start so the
                            // chunk is re-decoded whole, which also satisfies the >=1-token requirement.
                            while (n_past > 0 && !slot.prompt.tokens.boundary_is_chunk_safe((size_t) n_past)) {
                                n_past--;
                            }
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;

                        metrics.add_prompt_cached(n_past);

                        slot.prompt.tokens.keep_first(n_past);

                        // ===== ARM the mid-prefill shared-context base save (Option A) ==============
                        // Reset here first (STARTED runs exactly once per task) so a target left over
                        // from a released/aborted task never leaks into slot reuse. Then arm iff this is
                        // a COLD prefill of a text-only generative request that has a block-aligned
                        // first-user boundary B_ctx clearing the floor and strictly inside the prompt.
                        // The prefill loop clamps the batch at B_ctx; the post-decode hook whole-saves
                        // [0, B_ctx) when the slot is resident at exactly B_ctx. Because the resident
                        // sequence IS the true whole state at B_ctx, the whole-save is sound for dense,
                        // SWA AND recurrent/hybrid — there is NO model-class gate (unlike the old
                        // idle-flush [0,B) sub-range checkpoint this replaces).
                        slot.ctx_save_pos = -1;
                        if (auto_cache_enabled() &&
                            slot.task->need_sampling() &&               // generative only (not embed/rerank; keeps the can_split path)
                            slot.alora_invocation_start <= 0 &&         // aLoRA caching bound (mirror the auto-restore gate)
                            are_lora_equal(slot.lora, params_base.lora_adapters) && // fp captures the global LoRA set (invariant 3)
                            !input_tokens.has_media()) {                // text-only keeps the block-hash array dense
                            const int     B        = params_base.slot_save_block;
                            const int     floor    = std::max(B, params_base.slot_save_context_min_tokens);
                            const int32_t boundary = slot.prompt.ctx_boundary; // first_user_message_pos, stashed at task launch
                            // block-align DOWN: base stays within the shared preamble
                            int32_t B_ctx = boundary > 0 ? boundary - (boundary % B) : -1;
                            // SWA FALLBACK: on an SWA model a release-time snapshot (prompt + generated
                            // tail) is restorable ONLY by a request that STRICTLY EXTENDS it - its state
                            // file carries just the window [L - n_swa, L), so it can never be rewound to
                            // a shorter prefix (see the SWA gate in auto_restore_into_slot). A repeat /
                            // regenerate of the SAME prompt therefore gets ZERO reuse unless the store
                            // also holds a whole-state root STRICTLY INSIDE the prompt.
                            // CORRECTION (2026-08-19): this comment used to add "and, on a reasoning
                            // model, EVERY follow-up turn, since the generated thinking tokens are not
                            // replayed". That is NOT a property of reasoning models, it is a property of
                            // the individual chat template, and it is false for current ones. Measured
                            // via /apply-template with reasoning_content set on a history message:
                            // Qwen3.6 family DROPS historical reasoning (a follow-up turn is then not an
                            // extension of the release snapshot), Qwen3.8 RENDERS it as a full <think>
                            // block (a follow-up turn CAN extend). Normal chat replays thinking; the
                            // server even logs that a template supports it and suggests
                            // --reasoning-preserve. Do not reason about reuse from "is it a reasoning
                            // model", check what the deployed template actually renders.
                            // When the shared-context boundary does not arm one (absent, or
                            // below the floor - the common single-user-message case), anchor it at the
                            // deepest block boundary below the prompt end instead. The mid-prefill
                            // whole-save is sound for SWA precisely because the resident sequence IS the
                            // true whole state at B_ctx, so its persisted window is anchored at B_ctx.
                            // Only when this request got essentially no reuse (n_past < floor), so a warm
                            // or restored slot arms nothing and pays nothing (the arm below additionally
                            // requires n_past < B_ctx).
                            // `n_swa_mem > 0` here is a COST gate, NOT a soundness gate. Do not read the
                            // sentence it replaced ("untouched by construction") as an argument that
                            // FULL/RS are unsound here: they are not, the whole-save is sound for dense,
                            // SWA and recurrent/hybrid alike, for the same reason the outer arm above
                            // states, because the resident sequence IS the true whole state at B_ctx.
                            // What FULL and RS would pay is one extra whole-state write per COLD prefill
                            // that got no reuse (multi-GB at f16 KV on a deep prompt) against an
                            // LRU-bounded store, which is a measured perf/capacity decision, not a
                            // correctness one.
                            // The excuse that used to be offered for excluding FULL, "the regenerate
                            // fast-path covers it", does NOT hold: that path (see the restore-continue
                            // gate) requires n_past == task->n_tokens() AND n_past == prompt.n_tokens(),
                            // i.e. a snapshot of EXACTLY the request's length, so it serves an exact
                            // resend of prompt+generation, never a regenerate of the prompt alone. No
                            // auto-save site routinely produces such a snapshot (release/idle/shutdown
                            // save prompt+generated; the mid-prefill base is a strict prefix). Under RS
                            // neither mechanism exists at all: that gate is FULL-only and the logits
                            // sidecar is not even written under RS. After the whole-prefix restore rule,
                            // the RS miss degrades to a correct cold reprefill, so this is a reuse-rate
                            // question to revisit with measurements taken AFTER that rule lands (it
                            // changes the very reuse distribution the decision depends on), not a defect
                            // to widen blind.
                            if (n_swa_mem > 0 && n_past < floor &&
                                !(B_ctx >= floor && B_ctx < slot.task->n_tokens())) {
                                const int32_t e = slot.task->n_tokens() - 1; // -1 keeps B_ctx a STRICT prefix
                                B_ctx = e > 0 ? e - (e % B) : -1;
                            }
                            // COLD only: n_past < B_ctx means the [0, B_ctx) region was NOT reused/
                            // restored, so there is genuinely-new state to persist. A warm/restored
                            // slot (n_past >= B_ctx) arms nothing (zero change to normal prefill).
                            if (B_ctx >= floor && B_ctx < slot.task->n_tokens() && n_past < B_ctx) {
                                slot.ctx_save_pos = B_ctx;
                            }
                        }
                        // ===== end ARM =============================================================

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    // note: the prompt timing is advanced in post_decode(), so it does not cover
                    //       the tokens added to the batch below
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    slot.mem.seq_rm(slot.id, p0, -1);

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    bool has_mtmd = false;

                    // check if we should process the mtmd chunk
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the mtmd chunk
                        // note: it submits its own decode, potentially be async
                        //       so the timing is queued and flushed on the next sync
                        metrics_pre_decode();

                        // encode on the worker thread, so we can still handle metrics tasks
                        size_t n_tokens_out = 0;
                        int32_t res = 0;
                        queue_tasks.yield_to_queue([&]() {
                            res = process_mtmd_chunk(slot, slot.mbatch, cur_token_idx, n_tokens_out);
                        });

                        if (res != 0) {
                            SLT_ERR(slot, "failed to process mtmd chunk, res = %d\n", res);
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            return; // the slot is done, skip it entirely
                        }

                        metrics_queue_prompt(n_tokens_out);
                        slot.stats.n_prompt_processed += n_tokens_out;
                        slot.stats.update_prompt_last();

                        // add the mtmd chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            slot.prompt.tokens.push_back(chunk.get()); // copy
                        }

                        has_mtmd = true;
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.id,
                            cur_tok,
                            /* pos       = */ slot.prompt.tokens.pos_next(),
                            /* output    = */ slot.need_embd(),
                            /* is_prompt = */ true);
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.stats.n_prompt_processed++;

                        // mid-prefill shared-context base (Option A): stop this batch EXACTLY at the
                        // block-aligned first-user boundary B_ctx, never crossing it, so that once this
                        // batch is decoded the slot's resident sequence is precisely [0, B_ctx) — the
                        // true whole state there for the post-decode whole-save. One-shot: the hook
                        // disarms ctx_save_pos, after which the slot prefills [B_ctx, N) unclamped. If
                        // B_ctx exceeds n_batch the loop's own batch.size() < n_batch cap stops it first
                        // and a later update_slots() iteration re-enters here to reach B_ctx.
                        if (slot.ctx_save_pos > 0 && slot.prompt.n_tokens() == slot.ctx_save_pos) {
                            break;
                        }

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back().n_tokens + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.stats.n_gen = 0;
                        slot.i_batch     = batch.size() - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off, llama_batch & batch_view) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        metrics_pre_decode();

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        bool has_output = false;
        for (int i = off; i < off + batch_view.n_tokens; ++i) {
            has_output |= batch.tokens[i].output;
        }

        // decode on the worker thread, so we can still handle metrics tasks while waiting
        // note: the sync is done here too, so that the wait also happens off the main thread
        int ret = 0;
        queue_tasks.yield_to_queue([&]() {
            ret = llama_decode(ctx_tgt, batch_view);
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });

        if (ret != 0) {
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        } else {
            // success, apply batch metrics
            metrics_post_decode(off, batch_view.n_tokens, has_output);
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (!common_speculative_process(spec.get(), batch_view)) {
            SRV_ERR("%s", "failed to process speculative batch\n");

            // TODO: handle error
            throw std::runtime_error("failed to process speculative batch");
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off, llama_batch & batch_view) {
        // for checking if a given batch index is inside batch_view
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                if (slot.can_speculate()) {
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            // Feature A: capture this slot's last-token full-vocab logits for a possible disk
            // save. Cost: one ~n_vocab*4-byte copy per decoded token, incurred ONLY on
            // FULL/recurrent models AND only when slot saving is enabled (--slot-save-path set);
            // attention models and servers without slot-save pay nothing at all. The copy is
            // unavoidable for correctness: ctx logits are overwritten by the next slot's decode,
            // so a lazy read at SLOT_SAVE would be wrong under --parallel>1. Captured per-slot
            // from this slot's own tok_idx so it is correct for any N/interleave (never read
            // from the shared ctx at save time). common_sampler_sample already synchronized the
            // context above, so llama_get_logits_ith is valid here.
            if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL && !params_base.slot_save_path.empty()) {
                const float * lg = llama_get_logits_ith(slot.ctx_tgt, tok_idx);
                if (lg) {
                    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
                    slot.logits_last.assign(lg, lg + nv);
                    // Stamp the capture with the token count of the state it corresponds to.
                    // At this point process_token() has NOT yet appended the just-sampled token,
                    // so prompt.tokens.size() is exactly the length whose final token produced
                    // these logits — i.e. it matches the token_count a SLOT_SAVE would record.
                    slot.logits_last_n_tokens = (int32_t) slot.prompt.tokens.size();
                } else {
                    // capture failed -> invalidate so a later save never serializes stale logits
                    slot.logits_last.clear();
                    slot.logits_last_n_tokens = -1;
                }
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.stats.n_gen += 1;

            if (slot.stats.n_gen == 1) {
                slot.stats.update_prompt_last();
                slot.t_print_last = t_now;
                slot.n_gen_last = 0;
            }

            slot.stats.update_gen_last();

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                slot.release();

                return;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() ||
                    slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            // verify and try to accept the draft
            {
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                slot.spec_i_batch.clear();

                GGML_ASSERT(accepted.size() >= 1);

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                        if (slot.ctx_dft) {
                            ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                slot.spec_draft = std::move(accepted);
            }

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            slot.stats.update_gen_last();

            // update how many tokens out of those tested were accepted
            slot.stats.n_draft_accepted += n_accepted;
            slot.stats.n_draft_verif_steps += 1;

            auto & n_accepted_per_pos = slot.n_accepted_per_pos;
            if (n_accepted_per_pos.empty()) {
                n_accepted_per_pos.resize(common_speculative_n_max(&params_base.speculative), 0);
            }
            for (size_t i = 0; i < n_accepted && i < n_accepted_per_pos.size(); ++i) {
                n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // TODO: set result.probs

                slot.stats.n_gen += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    //
    // metrics helpers
    //

    // call before submitting a decode, so that the queued prompt stats can be timed
    void metrics_pre_decode() {
        t_decode_start = ggml_time_us();
    }

    // the batch is submitted, but its compute may not be done yet
    void metrics_queue_prompt(uint64_t n_tokens) {
        if (n_tokens == 0) {
            return;
        }
        if (n_prompt_queued == 0) {
            t_prompt_start = t_decode_start;
        }
        n_prompt_queued += n_tokens;
    }

    // call only after the context is synchronized, otherwise the time is meaningless
    void metrics_flush_prompt() {
        if (n_prompt_queued == 0) {
            return;
        }
        metrics.add_prompt(n_prompt_queued, ggml_time_us() - t_prompt_start);
        n_prompt_queued = 0;
    }

    // has_output is computed by the caller, which also already synchronized the context if it is set
    void metrics_post_decode(int32_t off, int32_t n_tokens, bool has_output) {
        metrics.n_decode++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        uint64_t n_prompt_tokens = 0;

        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens++;

            auto & slot = slots[t.id_slot];
            if (slot.stats.is_set()) {
                slot.stats.n_prompt_processed++;
            }
        }

        metrics_queue_prompt(n_prompt_tokens);

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            metrics_flush_prompt();
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const int64_t t_now = ggml_time_us();
        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];
            auto & slot = slots[t.id_slot];
            if (t.is_prompt && slot.stats.is_set()) {
                slot.stats.set_prompt_last(t_now);
            }
        }
    }

    // flush any queued prompt metrics if all slots are now idle
    void metrics_flush_idle() {
        if (n_prompt_queued == 0) {
            return;
        }

        llama_synchronize(ctx_tgt);
        metrics_flush_prompt();
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;

        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
    // graceful shutdown: the queue loop has exited (no update_slots running) but the
    // backend is still alive — llama_backend_free() runs later, in the caller's
    // clean_up(). Last safe point to persist slots' warm KV to the auto disk cache.
    impl->auto_save_slots_at_shutdown();
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* fp_mmproj              */ impl->fp_mmproj,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
        /* model_ftype            */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
    impl->queue_tasks.on_sleeping_state([this](bool sleeping) {
        if (sleeping) {
            impl->callback_state(SERVER_STATE_SLEEPING, {});
        }
        // for sleeping == false, event is emitted by load_model()
    });
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        // message delimiters for checkpointing
        auto delimiters = common_chat_msg_delimiters_parse(json_value(data, "message_delimiters", json::array()));
        delimiters.tokenize(ctx_server.vocab);

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.params.message_spans = task.tokens.find_message_spans(delimiters);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            // the gauges are averaged over the window between two scrapes
            task.metrics_reset_bucket = true;
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = res_task->to_metrics();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "model_alias",                 meta->model_name },
            { "model_ftype",                 meta->model_ftype },
            { "model_path",                  meta->model_path },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"video",  meta->has_inp_video},
                {"audio",  meta->has_inp_audio},
            } },
            { "media_marker",                get_media_marker() },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            { "ui",                          params.ui },
            { "ui_settings",                 meta->json_ui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
            { "cors_proxy_enabled",          params.ui_mcp_proxy },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        if (meta->has_mtmd) {
            // projector identity fingerprint (hex, matches the startup log line) so
            // operators can tell which mmproj a KV snapshot store belongs to.
            char fp_hex[17];
            snprintf(fp_hex, sizeof(fp_hex), "%016" PRIx64, meta->fp_mmproj);
            props["fp_mmproj"] = fp_hex;
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                get_model_info(),
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

json server_routes::get_model_info() const {
    return json {
        {"id",       meta->model_name},
        {"aliases",  meta->model_aliases},
        {"tags",     meta->model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta->model_vocab_type},
            {"n_vocab",     meta->model_vocab_n_tokens},
            {"n_ctx",       meta->slot_n_ctx},
            {"n_ctx_train", meta->model_n_ctx_train},
            {"n_embd",      meta->model_n_embd_inp},
            {"n_params",    meta->model_n_params},
            {"size",        meta->model_size},
            {"ftype",       meta->model_ftype},
        }},
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, true).size();
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}
