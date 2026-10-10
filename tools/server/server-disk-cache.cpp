// fork-only: the server-side helpers of the automatic disk KV cache, see server-disk-cache.h.

#include "server-disk-cache.h"
#include "server-context.h"
#include "server-task.h"

#include "gguf.h"

#include <cstring>
#include <fstream>

// --- KV restore-reuse: logits sidecar -------------------------------------------
// When a slot's full state is saved to disk (SLOT_SAVE) for a recurrent/hybrid (FULL) model,
// we additionally persist the last decoded token's full-vocab logits in a small sidecar file
// (<state>.logits). On SLOT_RESTORE of an exact-prompt "regenerate" request, those logits let
// the server emit the first token WITHOUT re-decoding into the (un-rewindable) restored
// recurrent state, which would otherwise crash. The sidecar is independent of libllama's
// state-file format (so that format is left untouched) and is purely best-effort: any
// missing/corrupt/vocab-mismatched sidecar degrades gracefully to the existing behavior.
static constexpr uint32_t SLOT_LOGITS_MAGIC   = 0x474C4B4Cu; // "LKLG" (llama kv logits), LE
static constexpr uint32_t SLOT_LOGITS_VERSION = 1u;

std::string slot_logits_sidecar_path(const std::string & state_filepath) {
    return state_filepath + ".logits";
}

// The auto disk cache's DRAFT sidecar: the speculative draft context's (MTP / EAGLE3 / draft model)
// KV for the same cells as "<state>", in the ordinary llama_state_seq file format, written with the
// same [lo, hi) split as the unit itself so a chain of .dft files composes exactly like the chain of
// state files. Optional and best-effort: a unit without one restores with a cold draft, and draft KV
// can only change how many drafted tokens are accepted, never the output (the target verifies every
// token), so a stale or foreign sidecar costs speed, never correctness.
std::string slot_draft_sidecar_path(const std::string & state_filepath) {
    return state_filepath + ".dft";
}

// A manual /slots save's FINGERPRINT sidecar. The .bin is upstream's format, which records no model
// identity: its restore only checks that every token id is below n_vocab, so a save restored under a
// different model of the same vocabulary size, or a media save under a different --mmproj, would
// otherwise be accepted and its cells reused as if valid. This small file beside the .bin carries the
// identity the fork checked before; a restore refuses when it is present and does not match. Absent
// (a file saved by upstream or elsewhere) means unchecked, as upstream behaves.
static constexpr uint32_t SLOT_FP_MAGIC   = 0x50464C53u; // "SLFP"
static constexpr uint32_t SLOT_FP_VERSION = 1u;

std::string slot_fp_sidecar_path(const std::string & state_filepath) {
    return state_filepath + ".fp";
}

// written to a temp name and renamed, so a torn write never leaves a partial sidecar in place
bool slot_fp_write(const std::string & state_filepath, const slot_fp_record & r) {
    const std::string side = slot_fp_sidecar_path(state_filepath);
    const std::string tmp  = side + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            return false;
        }
        f.write((const char *) &SLOT_FP_MAGIC,      sizeof(SLOT_FP_MAGIC));
        f.write((const char *) &SLOT_FP_VERSION,    sizeof(SLOT_FP_VERSION));
        f.write((const char *) &r.fp_model,         sizeof(r.fp_model));
        f.write((const char *) &r.fp_mmproj_loaded, sizeof(r.fp_mmproj_loaded));
        f.write((const char *) &r.fp_mmproj,        sizeof(r.fp_mmproj));
        f.flush();
        if (!f.good()) {
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, side, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}

// 0: no sidecar; 1: read into `r`; -1: present but unreadable (a restore must refuse)
int slot_fp_read(const std::string & state_filepath, slot_fp_record & r) {
    const std::string side = slot_fp_sidecar_path(state_filepath);
    std::error_code ec;
    if (!std::filesystem::exists(side, ec) || ec) {
        return 0;
    }
    std::ifstream f(side, std::ios::binary);
    uint32_t magic = 0, version = 0;
    f.read((char *) &magic,              sizeof(magic));
    f.read((char *) &version,            sizeof(version));
    f.read((char *) &r.fp_model,         sizeof(r.fp_model));
    f.read((char *) &r.fp_mmproj_loaded, sizeof(r.fp_mmproj_loaded));
    f.read((char *) &r.fp_mmproj,        sizeof(r.fp_mmproj));
    if (!f || magic != SLOT_FP_MAGIC || version != SLOT_FP_VERSION) {
        return -1;
    }
    return 1;
}

bool slot_draft_trailer_read(const std::string & path, llama_pos & tail_pos, std::vector<uint8_t> & blob) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return false;
    }
    const std::streamoff sz = f.tellg();
    const std::streamoff ft = (std::streamoff) (sizeof(llama_pos) + 2 * sizeof(uint32_t));
    if (sz < ft) {
        return false;
    }
    uint32_t n = 0;
    uint32_t magic = 0;
    f.seekg(sz - ft);
    f.read((char *) &tail_pos, sizeof(tail_pos));
    f.read((char *) &n, sizeof(n));
    f.read((char *) &magic, sizeof(magic));
    if (!f || magic != SLOT_DRAFT_TRAILER_MAGIC || (std::streamoff) n > sz - ft) {
        return false;
    }
    blob.resize(n);
    f.seekg(sz - ft - (std::streamoff) n);
    f.read((char *) blob.data(), n);
    return (bool) f;
}

// Best-effort "touch": bump the mtime of an auto-cache snapshot's 3-file unit (state + .logits +
// .meta) to now, so a snapshot that is REUSED (read/restored) but never rewritten is treated as
// recently-used by the mtime LRU. Without this, the LRU is least-recently-WRITTEN, which would
// evict a hot base snapshot that N forked requests keep restoring from (it never gets rewritten).
// Never throws and never errors out the caller: every failure is swallowed via error_code (the
// file may have been concurrently evicted by another process; that is harmless here). This only
// runs on a successful restore, off the generation hot path.
void auto_touch_unit(const std::string & state_filepath) {
    const auto now = std::filesystem::file_time_type::clock::now();
    std::error_code ec;
    std::filesystem::last_write_time(state_filepath, now, ec);
    std::filesystem::last_write_time(slot_logits_sidecar_path(state_filepath), now, ec);
    std::filesystem::last_write_time(state_filepath + ".meta", now, ec);
    std::filesystem::last_write_time(slot_draft_sidecar_path(state_filepath), now, ec);
}

// Best-effort write of the logits sidecar. Returns the number of bytes written (0 on failure or
// when there is nothing valid to write). Never throws. The file is written to a temp path and
// atomically renamed so a partial/interrupted write can never leave a corrupt sidecar in place.
// Fields are serialized byte-by-byte little-endian (not a raw struct fwrite) for portability;
// the float payload is documented LE-only, matching llama.cpp's native-LE state-file contract.
size_t slot_logits_write(const std::string & state_filepath,
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
bool slot_logits_read(const std::string & state_filepath,
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
    std::string dft_path;     // "<state>.dft",    "" if none (auto disk cache draft sidecar)
    std::string fp_path;      // "<state>.fp",     "" if none (manual /slots fingerprint sidecar)
    uintmax_t   bytes = 0;
    std::filesystem::file_time_type mtime;
    // Tree-aware eviction (U5): a checkpoint node's identity, derived entirely from disk. A node is
    // identified by the PAIR (`node_id`, `n_tokens`), both parsed from the auto filename
    // "auto-<fp>-<chain_hash>-<n_tokens>.bin": `node_id` is the middle-hex chain-hash and `n_tokens` is
    // the trailing token count. This pair is what a child delta's parent link (`parent_id`, `range_lo`)
    // resolves to: exactly the pair U4 restore feeds to auto_state_filename(parent_id, range_lo). Keying
    // on the pair (not chain_hash alone) is essential: a continuation that does NOT cross a whole-block
    // boundary shares its parent's chain-hash, so node_id ALONE would make the child collide with (and
    // even self-reference) its parent. `node_id`/`n_tokens` are 0 for any file that is not an auto-*
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
    // BUT still carried in the unit set so it counts as a live child of its parent: a pinned delta
    // node's whole ancestor chain (each of which then has a live child) is protected from eviction,
    // and its own .pin is never deleted. A pinned v1 root has no parent/children, so it behaves
    // exactly as before: reserved, uncounted, unevictable.
    bool        pinned    = false;
};

// Read only the delta-node section of a state file's .meta (parent_id + [range_lo, range_hi)). Thin
// wrapper over slot_meta_read (defined further down, after model_fp) so slot_save_enforce_limits (which
// lives above the fingerprint/meta machinery) can resolve the checkpoint tree without pulling model_fp
// into scope. Returns false (and leaves the outputs as a parentless root) for a v1/foreign/absent meta.
static bool slot_node_meta_probe(const std::string & state_filepath,
                                 uint64_t & parent_id, uint32_t & range_lo, uint32_t & range_hi);

// Parse a node's identity PAIR (chain_hash, n_tokens) from an auto-cache state filename
// ("auto-<fp_model>-<chain_hash>-<n_tokens>.bin"): the middle 16-hex group is the chain-hash a child
// delta stores as its parent_id, and the trailing decimal group is the token count a child delta stores
// as its parent link's range_lo. The pair is the node's unique key on disk. Returns false (and leaves
// both outputs 0) for any name that is not an auto snapshot: a foreign/manual file is then treated as
// its own parentless root and always ages by plain mtime. n_tokens is clamped to uint32_t; the save side
// bounds token counts far below 2^32 (slot_meta_read rejects tok_count > 2^28).
bool slot_save_parse_node_id(const std::string & state_path, uint64_t & node_id, uint32_t & n_tokens) {
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
// store: any regular file in it (other than recognized "<X>.logits" sidecars and "*.tmp"
// temporaries) is an eviction candidate. Point --slot-save-max-count/-mb at a DEDICATED directory;
// do not mix unrelated files into the auto-cache directory. (With no caps set, the default,
// nothing is ever deleted and the directory is left exactly as before.)
// `just_written` is the exact filepath string the server built as `slot_save_path + filename`;
// directory_iterator(dir) over that same `slot_save_path` yields identically-spelled path strings
// on POSIX (the production target), so raw string equality correctly identifies the just-saved
// unit. (Not used on Windows in practice; if ever needed there, switch to filename comparison.)
void slot_save_enforce_limits(const std::string & dir,
                              int32_t max_count, int64_t max_bytes,
                              const std::string & just_written,
                              bool & oversized,
                              uint64_t * n_evicted,
                              uint64_t * n_bound_exceeded,
                              bool * chain_rejected) {
    oversized = false;
    if (chain_rejected) {
        *chain_rejected = false;
    }
    if (max_count <= 0 && max_bytes <= 0) {
        return; // both unlimited
    }

    std::error_code ec;
    std::vector<slot_save_unit> units;
    uintmax_t this_unit_bytes = 0;

    // First pass: enumerate every regular file once and record the full set of paths so we can
    // tell a real sidecar (sibling of a state file we wrote) from a state file a client happened
    // to name "foo.logits". We must NOT blindly skip every "*.logits": fs_validate_filename
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
                (p.size() >= 9  && p.compare(p.size() - 9,  9,  ".tmp.meta")   == 0) ||
                (p.size() >= 8  && p.compare(p.size() - 8,  8,  ".tmp.dft")    == 0)) {
                continue;
            }
            // a "<X>.logits" file is a sidecar ONLY when its state file "<X>" is also present;
            // accounted together with that state file below, so skip it here.
            if (p.size() >= 7 && p.compare(p.size() - 7, 7, ".logits") == 0 &&
                present.count(p.substr(0, p.size() - 7))) {
                continue;
            }
            // a "<X>.meta" file is the auto disk cache's tokens+fingerprint sidecar; treat it
            // exactly like ".logits": accounted with its state file below, reaped if orphaned.
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
            // a "<X>.dft" file is the draft-context sidecar: accounted with its state file, reaped if orphaned
            if (p.size() >= 4 && p.compare(p.size() - 4, 4, ".dft") == 0 &&
                present.count(p.substr(0, p.size() - 4))) {
                continue;
            }
            // a "<X>.fp" file is a manual save's fingerprint sidecar: the same
            if (p.size() >= 3 && p.compare(p.size() - 3, 3, ".fp") == 0 &&
                present.count(p.substr(0, p.size() - 3))) {
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
            if (p.size() >= 4 && p.compare(p.size() - 4, 4, ".dft") == 0 &&
                !present.count(p.substr(0, p.size() - 4))) {
                std::filesystem::remove(p, fec);
                continue;
            }
            if (p.size() >= 3 && p.compare(p.size() - 3, 3, ".fp") == 0 &&
                !present.count(p.substr(0, p.size() - 3))) {
                std::filesystem::remove(p, fec);
                continue;
            }

            slot_save_unit u;
            u.state_path = p;
            // PINNED snapshots (a sibling "<state>.pin" marker) are never evicted and are excluded
            // from the count/byte caps entirely: a reserved, persistent entry (e.g. a permanent
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
            const std::string dft = slot_draft_sidecar_path(p);
            if (present.count(dft)) {
                const auto db = std::filesystem::file_size(dft, fec);
                if (!fec) {
                    u.dft_path = dft;
                    u.bytes += db;
                }
            }
            const std::string fps = slot_fp_sidecar_path(p);
            if (present.count(fps)) {
                const auto fb = std::filesystem::file_size(fps, fec);
                if (!fec) {
                    u.fp_path = fps;
                    u.bytes += fb;
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
    // individual snapshot's size is never bounded; only --slot-save-max-mb bounds per-snapshot size.
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
                if (!u.dft_path.empty()) {
                    std::filesystem::remove(u.dft_path, ec);
                }
                if (!u.fp_path.empty()) {
                    std::filesystem::remove(u.fp_path, ec);
                }
                break;
            }
        }
        oversized = true;
        return;
    }

    // ---- Tree-aware eviction (U5) --------------------------------------------------------------------
    // The auto cache is a FOREST of checkpoint trees, reconstructed here purely from what is on disk (no
    // in-RAM tree map). A v1 whole snapshot (or any foreign/manual file) is a parentless root; a v3
    // delta node points at the parent whose KV prefix it extends (parent_id == the parent file's
    // chain-hash == its filename's middle hex). Two rules make eviction tree-correct:
    //   (a) NEVER evict a node that still has a child on disk: its delta .bin is meaningless without its
    //       base, so we only ever evict LEAVES (child_count[node_id] == 0), oldest leaf first, which
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
    // OTHER trees instead: a wide fan-out could evict the entire rest of the store. It also let a
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
        if (!u.dft_path.empty()) {
            std::filesystem::remove(u.dft_path, ec);
        }
        if (!u.fp_path.empty()) {
            std::filesystem::remove(u.fp_path, ec);
        }
    };

    // A node's on-disk identity is the PAIR (chain_hash, n_tokens): a continuation that does not cross a
    // whole-block boundary shares its parent's chain_hash, so chain_hash ALONE would make the child
    // collide with (indeed parent_id == node_id, self-reference) its parent, pinning the true tip as
    // unevictable and defeating the caps. n_tokens (from the filename) disambiguates. The parent link is
    // the pair (parent_id, range_lo): exactly the pair U4 restore feeds to auto_state_filename.
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
    // delta .bin would corrupt a compose-load), so it is dead weight: delete it regardless of the caps.
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

    // Pinned units are excluded from the caps entirely: they occupy the tree only so their ancestors
    // stay refcount-protected, matching the flat-LRU pin semantics (a pinned unit was uncounted there).
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
        if (n_evicted) {
            (*n_evicted)++;
        }
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

    // Nothing but the just-written unit is evictable. When it is a DELTA, every other counted unit is an ancestor
    // of it (or of a pinned node): every other lineage still has an evictable tip. The cap then cannot hold this
    // conversation's chain together with what must stay, which is the chain's version of a single unit larger than
    // the byte cap, and it is answered the same way: the just-written delta is deleted and the save rejected, so
    // the store stays within its caps (a long run of periodic prefill checkpoints under --slot-save-incremental is
    // the case that reaches this; its chain then ends at the deepest node the cap holds). A whole root keeps the
    // old behaviour: it is published and the store stays above the limit, with a WRN.
    auto reject_just_written_delta = [&]() -> bool {
        for (size_t i = 0; i < units.size(); ++i) {
            if (!alive[i] || units[i].state_path != just_written) {
                continue;
            }
            if (units[i].parent_id == 0 || units[i].pinned) {
                return false;
            }
            remove_unit_files(units[i]);
            alive[i] = 0;
            total -= std::min(total, (uintmax_t) units[i].bytes);
            count = (count > 0) ? count - 1 : 0;
            const auto it = child_count.find(parent_key(units[i]));
            if (it != child_count.end() && it->second > 0) {
                it->second--;
            }
            if (chain_rejected) {
                *chain_rejected = true;
            }
            return true;
        }
        return false;
    };

    if (max_count > 0) {
        while (count > (size_t) max_count) {
            if (!evict_leaf()) {
                if (reject_just_written_delta()) {
                    SRV_WRN("%s", "slot-save cache: --slot-save-max-count cannot hold this conversation's delta chain; "
                                  "the new delta was not kept\n");
                    if (n_bound_exceeded) {
                        (*n_bound_exceeded)++;
                    }
                    continue;
                }
                SRV_WRN("%s", "slot-save cache is over --slot-save-max-count but every remaining snapshot "
                              "has a live child delta; leaving it above the limit\n");
                if (n_bound_exceeded) {
                    (*n_bound_exceeded)++;
                }
                break;
            }
        }
    }
    if (max_bytes > 0) {
        while (total > (uintmax_t) max_bytes) {
            if (!evict_leaf()) {
                if (reject_just_written_delta()) {
                    SRV_WRN("%s", "slot-save cache: --slot-save-max-mb cannot hold this conversation's delta chain; "
                                  "the new delta was not kept\n");
                    if (n_bound_exceeded) {
                        (*n_bound_exceeded)++;
                    }
                    continue;
                }
                SRV_WRN("%s", "slot-save cache is over --slot-save-max-bytes but every remaining snapshot "
                              "has a live child delta; leaving it above the limit\n");
                if (n_bound_exceeded) {
                    (*n_bound_exceeded)++;
                }
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
//      false there is no scan, index, hashing, or allocation: behavior is unchanged.
//   2. Never restore on hash alone: the snapshot's token-ID array must byte-compare
//      equal to the request prefix before any restore (collision-safe).
//   3. Model identity: each snapshot carries a fingerprint (model/vocab/ctx/rope/
//      KV-type/FULL-vs-attention/LoRA); a mismatch refuses the restore.
//   4. Fallback totality: any failure (corrupt file, fp/vocab mismatch, IO error,
//      no match) falls back to a normal prefill: never crash, never wrong output.
//   5. Hot-path purity: no save runs during generation (token decode); restore happens
//      once before prefill. Saves are decided on slot release/reassign (idle flush, reclaim,
//      shutdown) AND, for the two mid-prefill nodes (system node, prompt node), inside a
//      cold prefill at the node position. The server-loop thread pays only the CAPTURE: the
//      gates and the copy of the state off the device into host memory. Writing, syncing and
//      publishing run on the background writer (see Concurrency); the copy itself still
//      delays a node's first token and stalls other slots for as long.
//
// Concurrency: all slot work runs on the single server-loop thread. The background writer
// (aw_*, one thread per instance) publishes captured units in FIFO order and owns no llama
// object; it shares the index (auto_idx.mtx), the writer state (aw.mtx, never held together
// with auto_idx.mtx) and atomic counters with the server thread. Restores read only published
// units; the save-side dedup and parent choice also see the writer's queue. Independent of
// legacy --prompt-cache and the in-memory prefix-reuse path; auto-restore fires only when
// in-memory reuse is poor.
// ---------------------------------------------------------------------------

// The .meta sidecar format layer (model_fp, SLOT_META_* constants incl. the v3 delta-node
// version, slot_meta_write/slot_meta_read) and the block chain hashing layer (auto_hash_mix/
// auto_block_hashes: index keys and filenames, media-aware) live in server-disk-cache-common.{h,cpp}
// so the parser of untrusted on-disk bytes and the hash algorithm link into standalone
// unit tests (tests/test-slot-meta.cpp, tests/test-auto-hash.cpp). The cache logic
// proper is server_context_impl's part in server-context-disk-cache.inc.
// Block boundaries are the only resumable prefix lengths (vLLM-APC / SGLang-radix
// granularity); we NEVER trust a hash alone (invariant 2): the caller byte-verifies
// tokens before any restore.

// Identity hash of an mmproj GGUF file from its header only (no tensor data read,
// ~ms even for a multi-GB file): FNV-1a/splitmix chain over every KV pair (key +
// typed value bytes), every tensor's name/shape/type, plus the file size. Computed
// once whenever an mmproj is loaded: it feeds model_fp.fp_mmproj and is exposed in
// /props for operators. Fills `out` and returns true; returns false if the header
// cannot be parsed (the caller treats that as a load failure: mtmd just loaded the
// same file, so a parse failure here means it changed under us).
bool mmproj_header_fingerprint(const std::string & path, uint64_t & out) {
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

// Forward-declared above slot_save_enforce_limits: expose only the delta-node fields so eviction can
// resolve the tree without model_fp in scope. Leaves the outputs as a parentless root ([0,0) with no
// parent) for a v1/v2 whole snapshot, a foreign file, or any read failure (invariant 4: never crash).
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

//
// the cache's parts of upstream's HTTP handlers (server_routes)
//

// get_res_props(): the cache's capabilities and the projector fingerprint
void server_props_disk_cache(const server_context_meta & meta, json & props) {
    if (!meta.auto_cache_caps.is_null()) {
        // the memory class as the auto disk cache sees it (docs/disk-cache.md, "Seeing hits and misses")
        props["auto_cache"] = meta.auto_cache_caps;
    }

    if (meta.has_mtmd) {
        // projector identity fingerprint (hex, matches the startup log line) so
        // operators can tell which mmproj a KV snapshot store belongs to.
        // note: this belongs here rather than in the get_props handler, so that it is also
        //       captured into cached_props and keeps being reported while the server sleeps.
        char fp_hex[17];
        snprintf(fp_hex, sizeof(fp_hex), "%016" PRIx64, meta.fp_mmproj);
        props["fp_mmproj"] = fp_hex;
    }
}

// handle_completions_impl(): the system node's position for a single string prompt
void server_task_preamble_end(server_task & task, const server_context_meta & meta, const llama_vocab * vocab,
                              const json & data, const json & prompt, size_t n_inputs) {
    // the system node's position, from the chat path's preamble end (characters of the
    // rendered prompt); a /completion client with a string prompt may send it too. A media
    // prompt takes the same path: tokens() compares against the cell tokens, where a media
    // cell never equals a text token, so a preamble containing media simply gets no node.
    if (meta.chat_params.preamble_cache && n_inputs == 1 && prompt.is_string()) {
        const int32_t n_chars = json_value(data, "preamble_end_chars", -1);
        if (n_chars > 0) {
            task.params.preamble_end = meta.chat_params.preamble_cache->tokens(
                vocab, prompt.get<std::string>(), n_chars, task.tokens.get_cell_tokens());
        }
    }
}
