// fork-only: the server-side helpers of the automatic disk KV cache (--slot-save-auto): the store lock,
// the sidecar files next to a state file (.logits, .dft, .fp), the store's LRU eviction, the projector
// fingerprint, and the in-memory index of the store. The cache itself is the server_context_impl part in
// server-context-disk-cache.inc; the format layer it shares with the tests is server-disk-cache-common.h.
#pragma once

#include "server-common.h"

#include "llama.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef _WIN32
#include <unistd.h> // getpid() for per-writer-unique temp filenames (cross-process atomicity)
#include <fcntl.h>
#include <sys/file.h> // flock() for the store lock (auto_store_lock)
#include <signal.h>   // kill(pid, 0): is the writer of a leftover temp still alive
#else
#include <process.h> // _getpid()
#define getpid _getpid
#endif

// Cross-process lock on a slot-save directory that pairs every unit's .bin with its .meta (fork).
//
// A unit is published as separate renames (.bin, .logits, .dft, then .meta) under a deterministic
// name, and peers sharing the directory can publish DIFFERENT content under one name: a whole root
// and a delta for the same tokens, chosen because their indexes saw different parents. Without a
// lock a reader can read one writer's .meta and then open the other writer's .bin. The dangerous
// interleave is a root .meta with a delta .bin: the bin header carries the full [0, N) token list in
// both formats, so the single clearing load of cells [L, N) is accepted and the slot holds a silent
// wrong state. (The opposite pair is caught by the NO_CLEAR overlap check in the kv reader, and a
// post-load pos_min check cannot close it generically: iSWA, recurrent, hybrid and DSV4 legitimately
// report pos_min > 0 after a correct whole restore.)
//
// Publishers hold the lock EXCLUSIVE for the rename sequence only (milliseconds). Restores hold it
// SHARED from re-reading the chain's .meta files until the state files are loaded, so what they load
// is exactly what those .meta files describe. Deletes (LRU, orphan reaping) take no lock: a reader
// that loses a file fails its load cleanly, and a re-publish under the name waits for the reader.
// The lock is flock(2) on the directory itself, so it creates no file and needs no cleanup. Waiting
// is bounded: on timeout a publish is skipped and a restore falls back to a cold prefill, both
// ordinary cache misses (invariant 4). A filesystem that does not support flock fails OPEN with one
// warning, i.e. the behaviour before this lock existed. On Windows it is a no-op for the same reason.
class auto_store_lock {
public:
    // lock scopes open on the calling thread (held or not): a deferred capture's flush must never wait for
    // the writer inside one, the writer may need the exclusive lock to get there (see aw_def_flush_cb)
    static int & depth() {
        static thread_local int d = 0;
        return d;
    }

    auto_store_lock(const std::string & dir, bool exclusive, int timeout_ms) {
        depth()++;
#ifndef _WIN32
        fd = ::open(dir.empty() ? "." : dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (fd < 0) {
            return; // no directory: nothing to pair, and nothing can be published or restored there
        }
        const int op = (exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (true) {
            if (::flock(fd, op) == 0) {
                ok = true;
                return;
            }
            if (errno != EWOULDBLOCK && errno != EINTR) {
                static std::atomic<bool> warned{false};
                if (!warned.exchange(true)) {
                    SRV_WRN("slot-save store lock unsupported on %s (errno %d); peers sharing it are unpaired\n",
                            dir.c_str(), errno);
                }
                ::close(fd);
                fd = -1;
                ok = true; // fail open: the pre-lock behaviour
                return;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
#else
        (void) dir; (void) exclusive; (void) timeout_ms;
        ok = true;
#endif
    }
    ~auto_store_lock() {
        release();
        depth()--;
    }
    auto_store_lock(const auto_store_lock &) = delete;
    auto_store_lock & operator=(const auto_store_lock &) = delete;

    bool held() const { return ok; }

    void release() {
#ifndef _WIN32
        if (fd >= 0) {
            ::flock(fd, LOCK_UN);
            ::close(fd);
            fd = -1;
        }
#endif
        ok = false;
    }

    // a publish waits out restores in progress (seconds for a multi-GB chain); a restore waits out a
    // rename sequence (milliseconds)
    static constexpr int PUBLISH_TIMEOUT_MS = 30000;
    static constexpr int RESTORE_TIMEOUT_MS = 10000;

private:
    int  fd = -1;
    bool ok = false;
};

struct slot_fp_record {
    uint64_t fp_model         = 0;
    uint32_t fp_mmproj_loaded = 0;
    uint64_t fp_mmproj        = 0;
};

// The draft's per-sequence carry-over (common_speculative_get_state: MTP's pending h of the unit's
// last cell) rides at the END of the .dft, after the llama_state_seq payload, which the state loader
// never reads past: [blob][i32 tail_pos][u32 blob_size][u32 SLOT_DRAFT_TRAILER_MAGIC]. tail_pos is
// the target position the blob belongs to, so a restore only applies it to a tail that ends there.
static constexpr uint32_t SLOT_DRAFT_TRAILER_MAGIC = 0x44465431; // "DFT1"

// the auto disk cache's state of a server_slot (a base of it)
struct server_slot_disk_cache {
    // idle-delay flush: cleared each time the slot is released (a fresh idle period begins), set
    // once its warm KV has been flushed to the auto disk cache, so the timed idle wake attempts the
    // flush at most once per idle period (auto_save_slot_if_useful's dedup covers any re-attempt).
    bool auto_idle_flushed = false;
    // the idle flush found the staging full and the writer busy: it is retried (at auto_idle_retry_ms, once
    // the writer is idle) instead of being given up for the idle period. Cleared when the slot is released.
    bool    auto_idle_deferred = false;
    int64_t auto_idle_retry_ms = -1;
    // a new task whose prefix the background writer still holds waits for its publish without blocking the
    // other slots: the task stays STARTED, and this is when its wait began (us), -1 when not waiting
    int64_t aw_pending_t0_us = -1;

    bool just_restored  = false; // set on disk slot-restore; one-shot, gates restored-slot KV reuse
    // tokens the auto disk cache restored for the current task, not yet accounted as hit or discard
    // (auto_restore_account consumes it where n_prompt_cached is final)
    int32_t n_auto_restored_pending = 0;
    // prompt-prefix tokens the RAM prompt cache (--cache-ram) brought into this slot for the current
    // task, beyond what the slot already held; accounted with the disk ones in auto_restore_account
    int32_t n_ram_loaded_pending = 0;
    // the unit behind n_auto_restored_pending: its length, the files on its chain, and 1 = whole state loaded,
    // 2 = side-state only (the slot's positional cells kept)
    uint32_t disk_unit_pending  = 0;
    uint32_t disk_nodes_pending = 0;
    uint8_t  disk_mode_pending  = 0;

    // --- mid-prefill shared-context base save (Option A) ---
    // When > 0, this cold-prefilling slot is ARMED to whole-save the leading shared preamble
    // [0, ctx_save_pos) as a deduplicated base: ctx_save_pos is the block-aligned first-user
    // boundary B_ctx. The prefill loop CLAMPS the batch exactly there (never crossing it), and the
    // post-decode hook in update_slots() whole-saves [0, B_ctx) at the instant the slot is resident
    // at exactly B_ctx: the true whole state there, so the save is recurrent-/SWA-/dense-correct.
    // One-shot: armed once per task at prompt start, cleared (-> -1) after the save so the slot then
    // prefills [B_ctx, N) normally. -1 = not armed (no boundary / warm restore / small preamble).
    int32_t ctx_save_pos = -1;
    // --- mid-prefill prompt node (--slot-save-node-prompt) ---
    // Same mechanism at a second position: the end of the last user message, exactly (without a user
    // span, n_tokens - 1 block-aligned down). Saved through auto_save_slot_if_useful, so it is a delta on
    // the deepest saved node under --slot-save-incremental. -1 = not armed.
    int32_t prompt_save_pos = -1;
    // --- periodic prefill checkpoints (--slot-save-prefill-interval) ---
    // The positions (ascending, each a multiple of the interval, moved down to a chunk start when it would
    // split a media chunk) at which this prefill stops its batch and publishes a node, armed at prompt start;
    // ckpt_save_idx is the next one to reach. Each node is saved through auto_save_slot_if_useful, so it is a
    // delta on the previous one under --slot-save-incremental and the true whole state there for every class.
    std::vector<int32_t> ckpt_save_pos;
    size_t               ckpt_save_idx = 0;
    // the last periodic checkpoint of this prefill that was published WHOLE (a class without deltas, or
    // --slot-save-incremental off): the next whole checkpoint replaces it, so one prefill keeps one whole checkpoint
    // (its deepest) instead of N, 2N, 3N, ... whose bytes would grow with the square of the prompt
    std::string          ckpt_prev_whole;
    // a prefill that counted as cold for the prompt node has not reached the end of its prompt yet: a resend
    // that continues it on this slot (a client disconnect, a timeout) still counts as cold
    bool                 cold_prefill_open = false;

    // --- KV restore-reuse (logits sidecar) ---
    // Full-vocab logits of this slot's most recently sampled token, captured at sample time
    // (only populated for the FULL and RS classes when --slot-save-path is set, see
    // logits_sidecar_class(); plain attention rewinds by one token instead). Serialized to the
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
    // the token count those logits follow: a restore that is then trimmed (a plain-attention or whole-window
    // SWA unit rewound to a shorter request) must not emit from the unit's logits
    int32_t restored_logits_n = -1;

    // the cache's part of server_slot::reset()
    void reset_disk_cache() {
        // one-shot; never carry restored sidecar logits into a non-restore request.
        // NOTE: logits_last is deliberately NOT cleared here: it is the slot's running
        // "last sampled distribution" and must survive into the idle state so a subsequent
        // SLOT_SAVE can serialize it.
        restored_logits.clear();

        // Same one-shot rule for the restore flag, so it can outlive at most ONE task. Its consumers
        // sit behind guards a task can miss entirely (cache_prompt == false forces n_past = 0, which
        // skips the consume), and reset() runs from release(), i.e. AFTER the consumers of the task
        // that owns the restore. A manual /slots restore is not a processing task and never reaches
        // release(), so the flag still survives from that endpoint into the next request, which is
        // what the regenerate fast-path relies on.
        just_restored = false;
        n_auto_restored_pending = 0;
        n_ram_loaded_pending    = 0;
    }

    // the cache's part of server_slot::release(), when the slot goes idle
    void release_disk_cache() {
        auto_idle_flushed  = false; // fresh idle period: eligible for a timed idle flush again
        auto_idle_deferred = false;
        auto_idle_retry_ms = -1;
        aw_pending_t0_us   = -1;
    }
};

// One in-memory index entry: the longest snapshot that reaches a given block
// boundary. Mirrors slot_save_unit's mtime LRU semantics for reconciliation.
struct auto_cache_entry {
    std::string state_path; // full state file path (sidecars derived via *_path helpers)
    uint32_t    n_tokens = 0;
    model_fp    fp;         // snapshot's fingerprint (must equal the live one to be used)
    // PINNED: a sibling "<state_path>.pin" marker exists. A pinned entry is NEVER dropped from its
    // by_boundary bucket when the per-boundary cap (AUTO_MAX_CANDIDATES_PER_BOUNDARY) is exceeded, so a
    // pinned base (the SHORTEST entry in a bucket shared with many longer divergent siblings, yet the
    // only real strict-prefix parent the incremental save parent-find can use) stays findable past 32
    // warm siblings. Set from disk during the scan; re-stat'd authoritatively when a drop is forced (a
    // .pin touched AFTER the entry was first indexed, the normal deploy order, is still honoured).
    bool        pinned = false;
    // the unit's identity over EVERY cell (auto_block_hashes' `full`): the chain value after its last
    // cell, so it also commits the trailing partial block that no boundary key covers. Two units of
    // equal length at one boundary are the same unit only when this matches too.
    uint64_t    id     = 0;
    // the unit's DEEPEST boundary key (the last entry of its block-hash chain). A request reaches a
    // unit at that key only when it agrees with the unit through its last indexed boundary, so a unit
    // met at a shallower key is left before its end. 0 = unknown (treated as possibly whole).
    uint64_t    deepest = 0;
    // the unit holds media cells (a v2 or v4 .meta): the projector is identity for it (model_fp::fp_mmproj_loaded)
    bool        has_media = false;
};

// boundary-hash -> best (longest) entry covering that prefix length. Touched by the
// server-loop thread and the background writer, always under mtx (see Concurrency above). `scanned`
// guards the one-time startup scan; `dir_mtime`/`last_refresh` drive the cheap
// cross-process refresh (see auto_index_refresh): a peer process that writes a new
// snapshot bumps the slot-save directory's mtime, which the next lookup notices and
// re-scans, so a freshly-created cache becomes visible to OTHER processes without a
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
    // parseable: remembering it means a rescan never re-opens it, and a FUTURE meta version
    // bump costs each old reader one read total instead of one per scan. Reconciled together
    // with indexed_files when files disappear (auto_index_drop_missing_locked), so a peer
    // evicting a rejected unit lets a later same-name re-create be examined afresh.
    std::unordered_set<std::string> rejected_files;
    // Units of THIS model file that fail restore_compatible (another rung's RoPE/YaRN settings, other cache
    // types, another mmproj, LoRA or block size). Never restored; kept only so a miss they would have
    // served is reported as an identity miss (llamacpp:auto_cache_restore_miss_identity_total). Their
    // boundary keys equal the live ones for the same tokens, since the chain is salted with the model and
    // the mmproj only. Read once per file, like indexed_files.
    struct foreign_unit {
        std::vector<uint64_t> bhs;
        std::string           why; // the identity fields that differ
    };
    std::unordered_map<std::string, foreign_unit> foreign_files;
    std::unordered_map<uint64_t, uint32_t>        foreign_by_boundary; // boundary -> number of foreign units
    bool scanned = false;
    std::filesystem::file_time_type dir_mtime{};      // dir mtime as of the last scan
    std::chrono::steady_clock::time_point last_refresh{}; // throttle: skip stat storms in a burst
};

// Cross-process refresh throttle: at most one dir-mtime stat per this interval on the hot lookup
// path (a forced refresh on a lookup miss bypasses it). Sub-second so a peer's new snapshot is
// visible within ~1 prefill of being written, effectively immediate from the user's view.
static constexpr int AUTO_REFRESH_MIN_MS = 1000;

// Multi-candidate index bounds. A boundary may be reached by several snapshots of different lengths
// (incremental saving makes every save a superset of the previous). They are kept longest-first so a
// longer snapshot never shadows a shorter exact-length one that a FULL model needs; the per-boundary
// list is capped (dropping the shortest UNPINNED entry: a pinned base is never dropped). A RESTORE
// lookup returns at most AUTO_MAX_RESTORE_ATTEMPTS candidates for the caller to try in order (each
// attempt is an expensive .meta read + byte-compare); the incremental SAVE parent-find instead passes
// SIZE_MAX so the deepest strict-prefix parent (the SHORTEST entry, e.g. a pinned base) is never hidden
// by the cap behind longer divergent siblings sharing the same boundary bucket.
static constexpr size_t AUTO_MAX_CANDIDATES_PER_BOUNDARY = 32;
static constexpr size_t AUTO_MAX_RESTORE_ATTEMPTS        = 4;
// --slot-save-prefill-interval: above this many checkpoints for a prompt that fills the context, warn at load
static constexpr int32_t AUTO_CKPT_NODES_RECOMMENDED     = 32;

std::string slot_logits_sidecar_path(const std::string & state_filepath);
std::string slot_draft_sidecar_path(const std::string & state_filepath);
std::string slot_fp_sidecar_path(const std::string & state_filepath);
bool slot_fp_write(const std::string & state_filepath, const slot_fp_record & r);
int slot_fp_read(const std::string & state_filepath, slot_fp_record & r);
bool slot_draft_trailer_read(const std::string & path, llama_pos & tail_pos, std::vector<uint8_t> & blob);
void auto_touch_unit(const std::string & state_filepath);
size_t slot_logits_write(const std::string & state_filepath,
                         const std::vector<float> & logits,
                         int32_t n_vocab,
                         uint32_t n_tokens);
bool slot_logits_read(const std::string & state_filepath,
                      int32_t expect_n_vocab,
                      uint32_t expect_n_tokens,
                      std::vector<float> & out);
bool slot_save_parse_node_id(const std::string & state_path, uint64_t & node_id, uint32_t & n_tokens);
void slot_save_enforce_limits(const std::string & dir,
                              int32_t max_count, int64_t max_bytes,
                              const std::string & just_written,
                              bool & oversized,
                              uint64_t * n_evicted = nullptr,
                              uint64_t * n_bound_exceeded = nullptr,
                              bool * chain_rejected = nullptr);
bool mmproj_header_fingerprint(const std::string & path, uint64_t & out);

// the cache's parts of upstream's HTTP handlers (server-context.cpp, server_routes)
struct server_context_meta;
struct server_task;
void server_props_disk_cache(const server_context_meta & meta, json & props);
void server_task_preamble_end(server_task & task, const server_context_meta & meta, const llama_vocab * vocab,
                              const json & data, const json & prompt, size_t n_inputs);
