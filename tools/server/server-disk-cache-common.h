// fork-only: the shared layer of the automatic disk KV cache (--slot-save-auto, server-context-disk-cache.inc).
// The media identity records, the .meta sidecar format, the model fingerprint, the block chain hashing,
// the system-prompt node boundary and the cache's fields of server_slot_stats and server_metrics. Included
// by server-common.h, so every server translation unit and the standalone tests see it.
#pragma once

#include "common.h"
#include "chat.h"
#include "llama.h"

#include <atomic>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// identity metadata for one media (image/audio) chunk of a prompt. This is what the
// auto disk cache persists per chunk: enough to re-verify a chunk against a future
// request's live chunks, without storing any pixel/sample data (the KV state file
// already holds the embeddings; the request itself carries the pixels).
struct server_media_record {
    uint32_t    start_idx = 0; // index of the chunk's first cell in the token list
    uint32_t    n_tokens  = 0; // number of cells (LLAMA_TOKEN_NULL entries) the chunk occupies
    uint32_t    n_pos     = 0; // number of positions the chunk occupies (M-RoPE: != n_tokens)
    uint32_t    nx        = 0; // token grid width (image); n_tokens for audio
    uint32_t    ny        = 0; // token grid height (image); 1 for audio
    uint32_t    is_audio  = 0; // 1 if the chunk is audio, 0 if image
    std::string id;            // mtmd bitmap id: FNV-1a of the raw uploaded bytes (never empty)
};

// same predicate as server_tokens::boundary_is_chunk_safe, for the scan-time shape of
// the data: a cell-aligned token list plus media records read back from a snapshot's
// .meta sidecar (no live chunks exist there). records must be ordered by start_idx,
// as extract_media_records produces them.
bool boundary_is_chunk_safe(const llama_tokens & cells, const std::vector<server_media_record> & records, size_t idx);


//
// auto disk cache .meta sidecar (fingerprint + tokens + media identity records)
//
// The format layer of the automatic disk prompt/KV cache (--slot-save-auto; the
// cache logic itself lives in server-context.cpp). It is defined here so the
// parser (which reads untrusted on-disk bytes) links into a standalone
// fuzz/unit test (tests/test-slot-meta.cpp).
//

static constexpr uint32_t SLOT_META_MAGIC             = 0x544D4B4Cu; // "LKMT" (llama kv meta), LE
static constexpr uint32_t SLOT_META_VERSION           = 1u;         // text-only whole snapshot (layout byte-frozen)
static constexpr uint32_t SLOT_META_VERSION_MEDIA     = 2u;         // v1 layout + appended media-record section
static constexpr uint32_t SLOT_META_VERSION_NODE      = 3u;         // v1 text layout + appended delta-node section
                                                                    // (parent_id + [range_lo, range_hi)); TEXT-ONLY,
                                                                    // no media records.
static constexpr uint32_t SLOT_META_VERSION_MEDIA_NODE = 4u;        // v1 header + v2 media tail + v3 node tail
                                                                    // (media-then-node): a media DELTA node whose
                                                                    // meta carries the WHOLE [0,N) record tiling +
                                                                    // cell-token array while its .bin holds only
                                                                    // cells [range_lo, range_hi). The ONLY format
                                                                    // with both NULL media cells AND a delta range.
// Version map (the 2x2 of {whole|delta-node} x {text|media}): v1 whole text, v2 whole media,
// v3 text delta node, v4 media delta node. The mapping between a (is_node, has_media) pair and
// the version byte lives in exactly ONE place per direction, slot_meta_version_for (write) and
// slot_meta_features_for (read) below, so the writer and reader can never disagree, and a future
// format is a contained edit to those two helpers rather than scattered version compares. See
// docs/kv-cache on the longer-term migration to capability flags / sections.
static constexpr uint32_t SLOT_META_MEDIA_MAX     = 4096u;       // cap: media records per snapshot
static constexpr uint32_t SLOT_META_ID_MAX        = 256u;        // cap: bytes per media-record id (0 invalid)

// The on-disk feature set a version byte selects: an optional v2-style media tail (fp_mmproj +
// records, with real LLAMA_TOKEN_NULL cells) and/or an optional v3-style delta-node tail
// (parent_id + range_lo + range_hi), always written media-then-node. `valid` is false for an
// unknown version byte. has_media_tail also governs the text/media split the reader keys on:
// media formats read fp_mmproj for real and permit NULL cells; text formats backfill fp_mmproj
// from the live value and reject any NULL cell.
struct slot_meta_features {
    bool has_media_tail = false;
    bool has_node_tail  = false;
    bool valid          = false;
};

// (is_node, has_media) -> version byte. The single write-side mapping point (used by
// slot_meta_write). Keep in lock-step with slot_meta_features_for.
static inline uint32_t slot_meta_version_for(bool is_node, bool has_media) {
    return is_node ? (has_media ? SLOT_META_VERSION_MEDIA_NODE : SLOT_META_VERSION_NODE)
                   : (has_media ? SLOT_META_VERSION_MEDIA       : SLOT_META_VERSION);
}

// version byte -> feature set. The single read-side mapping point (used by slot_meta_read to
// both whitelist the version and drive which tails/validation apply). Keep in lock-step with
// slot_meta_version_for.
static inline slot_meta_features slot_meta_features_for(uint32_t version) {
    switch (version) {
        case SLOT_META_VERSION:            return { /*media*/false, /*node*/false, /*valid*/true };
        case SLOT_META_VERSION_MEDIA:      return { /*media*/true,  /*node*/false, /*valid*/true };
        case SLOT_META_VERSION_NODE:       return { /*media*/false, /*node*/true,  /*valid*/true };
        case SLOT_META_VERSION_MEDIA_NODE: return { /*media*/true,  /*node*/true,  /*valid*/true };
        default:                           return { /*media*/false, /*node*/false, /*valid*/false };
    }
}

// Model/quant/context fingerprint that MUST match for a restore to be sound. All
// fields are stable inference-affecting identity captured once at model load and
// compared by exact equality (pure-CPU int compares). The blob produced by
// llama_state_seq_save_file is only safe to load into a context with identical KV
// geometry: a Q4_0-KV blob loaded into an F16 ctx, or a different rope/yarn scale
// (positions are baked into the saved state), silently corrupts, so cache_type_k/v
// and rope_scale are NOT optional.
struct model_fp {
    uint64_t fp_model      = 0; // hash of llama_model_desc + size + n_params (+ n_embd/n_layer)
    uint32_t fp_n_vocab    = 0;
    uint32_t fp_n_ctx_train= 0;
    uint32_t fp_n_embd     = 0;
    uint32_t fp_n_layer    = 0;
    uint32_t fp_rope_type  = 0;
    uint32_t fp_cache_k    = 0; // ggml_type of K cache (enum int)
    uint32_t fp_cache_v    = 0; // ggml_type of V cache (enum int)
    // Effective per-seq n_ctx of the WRITER. Persisted in the .meta, but NOT an identity field: it is
    // a capacity record. A unit restores into any context whose n_ctx holds its cells (fits_ctx), in
    // either direction, because no memory class serialises anything that depends on kv_size. Audited
    // per class (src/, merge-upstream-20261003):
    //   kv           llama_kv_cache::state_write_range picks cells by sequence, position range and
    //                position-relative SWA mask, and writes pos/seq/ext plus their K/V rows; kv_size is
    //                only the transposed-V source stride (llama-kv-cache.cpp state_write_data).
    //   iswa         base + SWA llama_kv_cache; the SWA cache's size follows n_ubatch/--swa-full, but
    //                its bytes are the unmasked window, chosen by position (llama-kv-cache-iswa.cpp).
    //   recurrent    mem_size = max(1, n_seq_max) (llama-model.cpp create_memory), no n_ctx.
    //   hybrid, hybrid-iswa, hybrid_idx, dsa, dsa-iswa, msa: compositions of the two above, each part
    //                written by the same writers (the hybrid_idx indexer and the dsa/msa indexer are
    //                plain llama_kv_cache instances sized like the attention cache).
    //   dsv4         raw part is iSWA; compressed K rows are min(comp_size, (pos_max+1)/ratio), which
    //                is (pos_max+1)/ratio whenever the unit fits; the compressor states are a fixed
    //                2*ratio rows (llama-kv-cache-dsv4.cpp state_write_impl). Only the legacy
    //                version-1 K-cache section required n_rows == kv_size; it is refused, not misread.
    // The read side enforces capacity again (state_read_meta / dsv4_state_read_k_cache throw), so a
    // unit that does not fit fails cleanly even if a caller skipped fits_ctx.
    // The ONE way n_ctx reaches the state is the model, not the memory class: LongRoPE models pick
    // rope_long over rope_short when n_ctx_seq > n_ctx_orig_yarn (llama_model::get_rope_factors),
    // which bakes different K positions. That threshold is the identity field, see ctx_long_regime.
    uint32_t fp_n_ctx      = 0;
    // Packed KV layout, every part of it refused on mismatch by the state reader or the restore rules:
    //   bit 0      1 if the seq_rm class is FULL or RS, else 0 (auto_compute_fingerprint says why the
    //              two share a value). It does NOT encode the 4-valued class, so PART and NO share 0.
    //   bit 1      1 if the attention KV stores V transposed (llama_kv_v_trans: Flash Attention was off
    //              when the memory was created). llama_kv_cache::state_read_data refuses a v_trans
    //              mismatch, so without this bit -fa on and -fa off peers of one model would name units
    //              alike and spend a whole state read on every restore of the other's unit, and the
    //              save-side parent-find could chain a delta onto a parent of the other layout.
    //   bits 8-23  KV stream count - 1 (llama_kv_n_stream: 1 when unified, else n_seq_max), which the
    //              same reader refuses on mismatch ("n_stream mismatch").
    // The common case (Flash Attention on, one stream) leaves bits 1-23 zero, so its .meta bytes are
    // what they were when this field held only bit 0.
    uint32_t fp_kv_layout  = 0;
    uint32_t fp_block      = 0; // slot_save_block this snapshot was hashed with
    uint64_t fp_rope_scale = 0; // bit-pattern of effective rope_freq_scale (position-critical)
    // rope_freq_base and ALL YaRN params also bake positions into the saved KV state exactly as
    // rope_freq_scale does: a same-model run differing only in --rope-freq-base or any --yarn-*
    // flag would otherwise pass the fingerprint and silently restore positionally-corrupt state.
    // All are bit-cast (float->u32) into identity; yarn_orig_ctx is an int. "0/negative = use
    // model-trained value" is normalized in auto_compute_fingerprint so equal effective configs match.
    uint64_t fp_rope_base       = 0; // bit-pattern of effective rope_freq_base
    uint32_t fp_yarn_ext        = 0; // bit-pattern of yarn_ext_factor
    uint32_t fp_yarn_attn       = 0; // bit-pattern of yarn_attn_factor
    uint32_t fp_yarn_beta_fast  = 0; // bit-pattern of yarn_beta_fast
    uint32_t fp_yarn_beta_slow  = 0; // bit-pattern of yarn_beta_slow
    uint32_t fp_yarn_orig_ctx   = 0; // yarn_orig_ctx (int)
    uint64_t fp_lora       = 0; // hash of active LoRA-set ids+scales (0 if none)
    // 1 if the server was launched with --mmproj (mctx != nullptr), else 0. Identity for a unit that
    // holds MEDIA cells only (restore_compatible's has_media): a text-only unit's K/V does not depend on
    // whether a projector is loaded. Text cells are decoded by the language model alone, their positions
    // come from the token index (pos[0], replicated over every M-RoPE section by the batch allocator,
    // src/llama-batch.cpp, the same with or without a projector), and the projector only produces the
    // embeddings of media chunks. A text unit therefore restores across a rung with and a rung without
    // --mmproj (A18, 2026-10-05; tests/unit/test_slot_save_mmproj_identity.py checks it on gemma3 both
    // ways against a cold prefill). The bit stays in the .meta and in identity_hash, so the file names
    // are unchanged and each rung keeps writing under its own prefix.
    uint32_t fp_mmproj_loaded   = 0;
    // gguf-header hash of the loaded --mmproj file (0 on a text-only server); see
    // mmproj_header_fingerprint. Catches projector swap, requantization and dimension
    // changes that the fp_mmproj_loaded 0/1 bit cannot. Persisted only in v2 (media)
    // sidecars: text KV is projector-independent, so slot_meta_read backfills it from
    // the live value on v1 (v1 => text-only) and the compare below only ever bites
    // for media snapshots: existing v1 snapshots on an --mmproj server keep matching.
    uint64_t fp_mmproj          = 0;

    // Exact equality of every PERSISTED field (fp_n_ctx included, the live-only fp_n_ctx_orig
    // excluded): what a .meta round trip must preserve (tests/test-slot-meta.cpp). It is not the
    // restore gate; that is restore_compatible.
    bool operator==(const model_fp & o) const {
        return fp_model == o.fp_model && fp_n_vocab == o.fp_n_vocab &&
               fp_n_ctx_train == o.fp_n_ctx_train && fp_n_embd == o.fp_n_embd &&
               fp_n_layer == o.fp_n_layer && fp_rope_type == o.fp_rope_type &&
               fp_cache_k == o.fp_cache_k && fp_cache_v == o.fp_cache_v &&
               fp_n_ctx == o.fp_n_ctx && fp_kv_layout == o.fp_kv_layout &&
               fp_block == o.fp_block && fp_rope_scale == o.fp_rope_scale &&
               fp_rope_base == o.fp_rope_base && fp_yarn_ext == o.fp_yarn_ext &&
               fp_yarn_attn == o.fp_yarn_attn && fp_yarn_beta_fast == o.fp_yarn_beta_fast &&
               fp_yarn_beta_slow == o.fp_yarn_beta_slow && fp_yarn_orig_ctx == o.fp_yarn_orig_ctx &&
               fp_lora == o.fp_lora && fp_mmproj_loaded == o.fp_mmproj_loaded &&
               fp_mmproj == o.fp_mmproj;
    }

    // The LIVE model's LongRoPE threshold, llama_model_n_ctx_orig_longrope: hparams.n_ctx_orig_yarn
    // (the GGUF rope.scaling.original_context_length with --override-kv applied, else the load-time
    // n_ctx_train) on a model whose RoPE factors depend on the context size, 0 on every other model.
    // It is the exact value llama_model::get_rope_factors compares n_ctx_seq against, read from the
    // model rather than rebuilt from CLI params: --yarn-orig-ctx sets only cparams.n_ctx_orig_yarn,
    // which get_rope_factors never reads, and llama_model_n_ctx_train is rewritten under custom YaRN
    // after the context is created. Set only on the live fingerprint by auto_compute_fingerprint and
    // never persisted: a disk fingerprint that passes every other field has the same model file,
    // hence the same threshold, so the live value decides the regime of both sides. (--override-kv is
    // not fingerprinted, as before: peers overriding that key differently must not share a store.)
    uint32_t fp_n_ctx_orig      = 0;

    // Whether a context of fp_n_ctx cells runs LongRoPE's long factors, given the live threshold.
    // Two contexts on opposite sides of it bake different K rotations into the same tokens, so the
    // regime is identity even though n_ctx itself is not. On every model without context-dependent
    // RoPE factors the threshold is 0 and the bit is constantly false, so rungs share whatever
    // rope.scaling.original_context_length the GGUF carries (YaRN baked into the GGUF, as on the
    // DeepSeek and Laguna conversions, splits nothing here; its parameters are identity already).
    bool ctx_long_regime(const model_fp & live) const {
        return live.fp_n_ctx_orig > 0 && fp_n_ctx > live.fp_n_ctx_orig;
    }

    // Whether a unit carrying THIS fingerprint (read from disk) may be restored into the live
    // context `live`: every identity field equal, n_ctx compared only through ctx_long_regime. Any
    // difference REFUSES the restore (invariant 3). The second half of the gate is capacity,
    // fits_ctx, which needs the unit's cell count and is checked by the caller before any state-file
    // I/O. fp_block is identity on purpose: a unit hashed with a different block size cannot be
    // longest-prefix-matched against the current index.
    //
    // One gate for every use: the startup scan, the lookup, the restore, every parent of a delta
    // chain, the incremental-save parent-find and the manual /slots media restore. A delta may
    // therefore be written onto a unit another rung saved; the chain stays resolvable because both
    // rungs name units by the same identity_hash.
    // `has_media`: the unit holds media cells. Only then do the projector fields (fp_mmproj_loaded,
    // fp_mmproj) take part, see fp_mmproj_loaded.
    bool restore_compatible(const model_fp & live, bool has_media) const {
        return fp_model == live.fp_model && fp_n_vocab == live.fp_n_vocab &&
               fp_n_ctx_train == live.fp_n_ctx_train && fp_n_embd == live.fp_n_embd &&
               fp_n_layer == live.fp_n_layer && fp_rope_type == live.fp_rope_type &&
               fp_cache_k == live.fp_cache_k && fp_cache_v == live.fp_cache_v &&
               ctx_long_regime(live) == live.ctx_long_regime(live) && fp_kv_layout == live.fp_kv_layout &&
               fp_block == live.fp_block && fp_rope_scale == live.fp_rope_scale &&
               fp_rope_base == live.fp_rope_base && fp_yarn_ext == live.fp_yarn_ext &&
               fp_yarn_attn == live.fp_yarn_attn && fp_yarn_beta_fast == live.fp_yarn_beta_fast &&
               fp_yarn_beta_slow == live.fp_yarn_beta_slow && fp_yarn_orig_ctx == live.fp_yarn_orig_ctx &&
               fp_lora == live.fp_lora &&
               (!has_media || (fp_mmproj_loaded == live.fp_mmproj_loaded && fp_mmproj == live.fp_mmproj));
    }

    // Capacity half of a restore: a unit of `n_cells` cells fits the live context. A unit from a
    // larger rung that is longer than this context is skipped before its state file is opened.
    static bool fits_ctx(size_t n_cells, const model_fp & live) {
        return n_cells <= (size_t) live.fp_n_ctx;
    }

    // 64-bit digest of every identity field restore_compatible compares, n_ctx entering only as
    // ctx_long_regime. Used only to name the auto-snapshot files (auto_state_filename) and only on
    // the LIVE fingerprint (it reads fp_n_ctx_orig): two peers sharing one --slot-save-path that
    // agree on fp_model but differ in any identity field (cache-type, block, rope/YaRN, LoRA,
    // mmproj, ...) would otherwise mint the same filename for the same token prefix and atomically
    // rename over each other. Peers that differ only in n_ctx (context rungs, a lower-ctx vision
    // rung) mint the SAME name for the same tokens: the unit is interchangeable, so that is a dedup,
    // and their delta chains resolve by name across rungs. It is NOT the index key or a restore
    // gate: the block-chain hash keeps its fp_model salt (text-only .meta bytes stay byte-frozen)
    // and every restore is still guarded by restore_compatible.
    uint64_t identity_hash() const;
};

// the .meta sidecar path for an auto snapshot's state file. The sidecar is the tiny
// tokens+fingerprint(+media) twin this feature adds so the startup scan / pre-restore
// verify reads only a small file, never the multi-GB state.
std::string slot_meta_sidecar_path(const std::string & state_filepath);

// Best-effort atomic write of the .meta sidecar (LE, temp+rename, the exact idiom
// of slot_logits_write). v1 (media empty, layout byte-frozen): magic/version,
// fingerprint fields, tok_count, chain_hash, then int32 tokens[tok_count]. v2 (media
// records present): the full v1 layout, then fp_mmproj, n_media and the records
// (start_idx/n_tokens/n_pos/nx/ny/is_audio/id_len/id each). For v2 `toks` must be
// the cell-aligned list (media cells LLAMA_TOKEN_NULL, see get_cell_tokens) and the
// records must tile its NULL cells exactly, as extract_media_records produces them:
// slot_meta_read rejects anything else. v3 (incremental delta node, `is_node`, no
// media): the full v1 text layout followed by parent_id + range_lo + range_hi (the KV
// .bin holds only cells [range_lo, range_hi); parent_id chains to the snapshot it
// extends, 0 = root). v4 (media delta node, `is_node` WITH non-empty `media`): the v2
// media tail and the v3 node tail concatenated media-then-node: the meta carries the
// WHOLE [0,N) tiling + cell-token array (so restore's byte-verify is the v2 path) while
// the .bin holds only cells [range_lo, range_hi). The (is_node, media-empty) pair
// selects v1/v2/v3/v4 via slot_meta_version_for, the single write-side mapping point.
// Whole snapshots (is_node false) still write v1/v2 BYTE-IDENTICALLY, and a v3 text
// delta is byte-identical to before. Returns true on success. Never throws.
bool slot_meta_write(const std::string & state_filepath,
                     const model_fp & fp,
                     const llama_tokens & toks,
                     uint64_t chain_hash,
                     const std::vector<server_media_record> & media = {},
                     bool     is_node   = false, // true => v3 delta-node meta (with parent + range)
                     uint64_t parent_id = 0,     // parent node's chain_hash (0 = root)
                     uint32_t range_lo  = 0,     // this node's .bin holds KV cells [range_lo,
                     uint32_t range_hi  = 0);    // range_hi); ignored for a whole snapshot

// Read a .meta sidecar (version-aware: v1, v2, v3 and v4). Returns true and fills the
// outputs iff a valid sidecar exists; any short read / bad magic / unknown version /
// cap or media-tiling violation => false with outputs cleared. Never throws. The version
// byte is decoded once via slot_meta_features_for (the single read-side mapping point).
// On the TEXT formats (v1, v3: no media tail), fp_out.fp_mmproj is backfilled from
// `cur_fp_mmproj` (sound: text KV is projector-independent), so the fingerprint compare
// cannot refuse text-only snapshots on an --mmproj server. To keep that backfill sound, a
// text sidecar containing any LLAMA_TOKEN_NULL cell, or any bytes past its defined layout,
// is rejected: no text writer ever emits either, so both can only be a corrupt or
// relabelled media sidecar trying to bypass fp_mmproj. The MEDIA formats (v2, v4: media
// tail present) read fp_mmproj for real (no backfill), permit NULL cells and enforce the
// v2 record-tiling invariants over [0, tok_count). The delta-node fields are exposed via
// the optional out-ptrs; the NODE formats (v3, v4) fill them from the node tail, while a
// v1/v2 whole snapshot defaults them to a parentless root (parent_id 0, range
// [0, tok_count)). Note: `chain_hash` is recorded for debuggability but the authority for
// reuse is always the byte-compared tokens.
bool slot_meta_read(const std::string & state_filepath,
                    uint64_t cur_fp_mmproj,
                    model_fp & fp_out,
                    llama_tokens & toks_out,
                    std::vector<server_media_record> & media_out,
                    uint64_t * parent_out   = nullptr,
                    uint32_t * range_lo_out = nullptr,
                    uint32_t * range_hi_out = nullptr);

//
// auto disk cache block chain hashing
//
// The identity layer of the automatic disk prompt/KV cache: chain hashes are the
// index keys AND the on-disk filenames, so (like the .meta parser above) the
// algorithm is defined here and locked by a standalone unit test
// (tests/test-auto-hash.cpp). Collision resistance is only a candidate-narrowing
// accelerator: consumers NEVER trust a hash alone: tokens (and media records) are
// byte-verified before any restore.
//

// 64-bit chained hash primitive: fold a 64-bit value via the FNV-1a prime, then a
// splitmix avalanche. Every cache identity (block chain, fingerprints) folds
// through this one primitive.
static inline uint64_t auto_hash_mix64(uint64_t h, uint64_t v) {
    h ^= v;
    h *= 0x100000001b3ULL;                                  // FNV-1a 64-bit prime
    h ^= h >> 29; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 32; // splitmix64 finalize
    return h;
}

// token-ID convenience overload (zero-extended: bit-identical to the pre-media chain)
static inline uint64_t auto_hash_mix(uint64_t h, int32_t tok) {
    return auto_hash_mix64(h, (uint64_t) (uint32_t) tok);
}

// Returns the cumulative chain hash at every CHUNK-SAFE block boundary of a
// cell-aligned prompt. Each cell folds into the chain in order: a text token as
// its ID, a media (LLAMA_TOKEN_NULL) cell as a per-cell contribution derived from
// its covering record: splitmix64(fnv64(id) ^ (i - start_idx) ^
// mix(n_tokens, n_pos, is_audio) ^ fp_mmproj). Folding the chunk shape/type means
// an audio chunk can never impersonate an image chunk with the same id; folding
// fp_mmproj means a projector swap changes media boundary hashes without touching
// text boundaries; the per-cell offset disambiguates llava-uhd slices sharing one
// bitmap id. The chain is salted with `salt` (the model fingerprint hash, for ALL
// prompts, so a media prompt's pure-text prefix boundaries hash identically to a
// text-only prompt's and text<->media prefix reuse works both ways).
//
// A key is emitted at every block-aligned position AND at the end of every media
// chunk, in both cases only where boundary_is_chunk_safe holds (not strictly
// inside a chunk); this is the SINGLE site enforcing the boundary rule, so
// save-time insert, scan rehash and lookup cannot drift. A media prompt therefore
// always has at least one key (its last chunk end), and its keys are NOT
// positional: out[k] need not commit to (k+1)*B cells. A trailing partial text
// block is never a boundary. For a text-only prompt (media empty) out[k] commits
// to tokens [0, (k+1)*B) and nothing else is emitted, bit-identical to the
// pre-media algorithm (same filenames, same index keys); a text prompt shorter
// than one block yields no key. `media` must be ordered by start_idx and tile the NULL cells
// exactly, as extract_media_records / slot_meta_read produce them.
//
// `full`, when given, receives the chain value after the LAST cell: the unit's identity over every
// cell, including a trailing partial block that no key covers. It equals out.back() exactly when the
// last cell ends a key boundary. Unit filenames and every exact-length dedup use it, so two prompts
// of equal length that differ only after their last whole block are two units, not one.
//
// `chain`, when given, receives the chain value after EVERY cell: (*chain)[n - 1] is the `full` value
// of the first n cells, so a lookup can tell whether a unit of n cells is a whole prefix of the
// request by comparing it with the unit's identity, without reading the unit's sidecar.
std::vector<uint64_t> auto_block_hashes(const llama_tokens & cells,
                                        const std::vector<server_media_record> & media,
                                        int B,
                                        uint64_t salt,
                                        uint64_t fp_mmproj,
                                        uint64_t * full = nullptr,
                                        std::vector<uint64_t> * chain = nullptr);

// Where the disk cache's system-prompt node goes (the preamble end), template-generic. Two memoised
// steps, safe to call from any HTTP thread:
//  - chars():  the preamble end in characters of a rendered chat prompt (common_chat_preamble_end),
//              cached on everything that decides the preamble, so a conversation's later turns and
//              other conversations with the same system prompt and tools skip the probe renders;
//  - tokens(): the token position for that preamble. A pure function of the preamble text, so every
//              request sharing it gets the same node: the preamble's own tokens, minus any trailing
//              tokens that some continuation could merge with (BPE seams at plain-text role headers).
//              It must also be a token prefix of the request; when it is not, there is no node.
// A wrong boundary can only cost hit rate, never correctness: every restore checks the saved tokens
// against the request.
class server_preamble_cache {
public:
    int32_t chars (const common_chat_templates * tmpls, const common_chat_templates_inputs & inputs, const std::string & prompt);
    int32_t tokens(const llama_vocab * vocab, const std::string & prompt, int32_t n_chars, const llama_tokens & prompt_tokens);

    std::atomic<uint64_t> n_probed       {0}; // chat requests whose preamble was looked up
    std::atomic<uint64_t> n_probe_renders{0}; // requests that had to render the probes (cache miss)
    std::atomic<uint64_t> n_probe_failed {0}; // requests where no probe rendered (falls back to the message delimiters)
    std::atomic<uint64_t> n_seam_mismatch{0}; // preambles whose tokens were not a prefix of the request (no node)
    std::atomic<uint64_t> n_probe_short  {0}; // prompts too short in bytes to reach min_tokens: not probed

    // the system node's floor in tokens. A prompt of fewer bytes (plus a few for BOS and the like) than
    // this cannot hold that many tokens, since no token is shorter than a byte of text, so it is not
    // probed at all: the node could never be written.
    int32_t min_tokens = 0;

private:
    struct chars_entry {
        size_t      key;
        int32_t     n_chars;
        std::string preamble; // prompt[0, n_chars) the entry was computed from
    };
    struct tokens_entry {
        std::string  preamble;
        llama_tokens toks;     // the node's tokens, [0, X)
    };
    static constexpr size_t N_ENTRIES = 64;
    std::mutex               mtx;
    std::vector<chars_entry> by_inputs;   // most recent last
    std::vector<tokens_entry> by_text;    // most recent last
};

// auto disk cache: the site that asked for a save (llamacpp:auto_cache_save_site_*_total{site=...})
enum auto_save_site : uint8_t {
    AUTO_SAVE_SITE_RECLAIM,       // a new task takes the slot over
    AUTO_SAVE_SITE_IDLE,          // --slot-save-idle-seconds
    AUTO_SAVE_SITE_SHUTDOWN,      // graceful terminate
    AUTO_SAVE_SITE_CACHE_IDLE,    // --cache-idle-slots, at a task launch
    AUTO_SAVE_SITE_SYSTEM_NODE,   // mid-prefill, at the end of the system prompt
    AUTO_SAVE_SITE_PROMPT_NODE,   // mid-prefill, at the end of the user message
    AUTO_SAVE_SITE_RESPONSE_NODE, // --slot-save-node-response / --slot-save-node-tool
    AUTO_SAVE_SITE_PREFILL_CKPT,  // --slot-save-prefill-interval, mid-prefill every N tokens
    AUTO_SAVE_SITE_COUNT,
};

inline const char * auto_save_site_name(int site) {
    static const char * const names[AUTO_SAVE_SITE_COUNT] = {
        "reclaim", "idle", "shutdown", "cache_idle", "system_node", "prompt_node", "response_node",
        "prefill_checkpoint",
    };
    return site >= 0 && site < AUTO_SAVE_SITE_COUNT ? names[site] : "unknown";
}

// the auto disk cache's part of server_slot_stats (a base of it)
struct server_slot_stats_disk_cache {
    uint64_t n_prompt_cached_disk = 0; // part of n_prompt_cached restored from the auto disk cache
    uint64_t n_prompt_cached_ram  = 0; // part of n_prompt_cached loaded from the RAM prompt cache (--cache-ram)
    // the disk unit behind n_prompt_cached_disk: its length in cells, how many files its chain spans (a whole
    // root is 1, each delta adds one) and how it was loaded (1 = the whole state, 2 = the side-state only,
    // the slot's own positional cells kept)
    uint32_t cache_disk_unit_n    = 0;
    uint32_t cache_disk_nodes     = 0;
    uint8_t  cache_disk_mode      = 0;

    // fills the cache fields of server_slot_stats::to_json() (n_prompt_cached: the upstream field of the same name)
    void to_json_disk_cache(common_json & base, uint64_t n_prompt_cached) const;
};

// the auto disk cache's part of server_metrics (a base of it)
struct server_metrics_disk_cache {
    // auto disk KV cache: opportunities that were skipped or degraded (never silent, each also logs a WRN)
    uint64_t n_auto_save_whole_fallback    = 0; // parented saves published as whole roots because the class cannot write deltas
    uint64_t n_auto_save_failed            = 0; // saves dropped with nothing published (no space, IO error, refused)
    uint64_t n_auto_save_root              = 0; // whole roots published (includes whole fallbacks)
    uint64_t n_auto_save_delta             = 0; // delta nodes published
    uint64_t n_auto_save_bytes             = 0; // state bytes written by published saves
    uint64_t n_auto_cache_evicted          = 0; // units this instance evicted from the shared store (LRU)
    uint64_t n_auto_restore_hit            = 0; // requests that restored a prefix from disk
    uint64_t n_auto_restore_miss           = 0; // eligible requests with >= 1 block beyond the in-memory match and no restore
    uint64_t n_auto_restore_failed         = 0; // restores that failed after clearing the slot
    uint64_t n_auto_restore_not_prefix     = 0; // misses where a snapshot of the prefix existed but the class could not rewind into it
    uint64_t n_auto_restore_miss_identity  = 0; // misses where a unit of the same model covered the prefix under another identity
    uint64_t n_auto_restore_side_only      = 0; // restores that loaded only the side-state and kept the slot's positional cells
    uint64_t n_auto_skipped_shared         = 0; // tasks with a shared prompt prefix (n_tokens_shared) the cache neither restores nor saves
    uint64_t n_auto_cache_bound_exceeded   = 0; // eviction passes that left the store above a cap (nothing evictable)
    uint64_t n_auto_ckpt_superseded        = 0; // whole prefill checkpoints removed because a deeper one of the same prefill replaced them
    int32_t  auto_delta_capable            = 0; // the delta probe: 0 not probed yet, 1 deltas, 2 whole roots only
    uint64_t n_auto_restore_tokens         = 0; // tokens restored from disk and kept by the prompt
    uint64_t n_auto_restore_discarded      = 0; // restores whose tokens a later n_past clamp threw away (counted as misses)
    uint64_t n_auto_save_draft             = 0; // draft-context (.dft) sidecars written next to a unit
    uint64_t n_auto_save_draft_skipped     = 0; // units published without one (no cells, or the range was ignored)
    // background writer (filled from the shared counters when the metrics are read)
    uint64_t n_auto_save_queued             = 0; // saves handed to the writer
    uint64_t n_auto_save_streamed           = 0; // of those, larger than the free staging: streamed while the writer was idle
    uint64_t n_auto_save_dropped_staging    = 0; // saves dropped because the staging was full and the writer busy
    uint64_t n_auto_save_orphan_dropped     = 0; // queued deltas dropped because their parent failed or vanished
    uint64_t n_auto_save_shutdown_abandoned = 0; // queued saves abandoned at the shutdown deadline
    uint64_t n_auto_save_staging_bytes      = 0; // gauge: host bytes held by saves not yet written
    uint64_t n_auto_save_queue_depth        = 0; // gauge: saves queued or being written
    uint64_t n_auto_save_admission_waits    = 0; // saves whose capture waited for the writer (staging full, writer busy)
    uint64_t n_auto_save_admission_wait_us  = 0; // total time those captures waited
    uint64_t n_auto_save_deferred           = 0; // saves captured deferred (stage 2): positional K/V copied after the capture
    uint64_t n_auto_save_deferred_forced    = 0; // times a cache mutation forced a pending deferred copy
    uint64_t n_auto_save_deferred_bytes     = 0; // positional bytes those captures left in the cache
    uint64_t n_auto_save_deferred_pending   = 0; // gauge: deferred captures still owing positional bytes
    uint64_t n_auto_save_deferred_host      = 0; // gauge: host bytes (side-state) held by those captures
    uint64_t n_auto_save_site_requested[AUTO_SAVE_SITE_COUNT] = {}; // units a save site decided to write
    uint64_t n_auto_save_site_published[AUTO_SAVE_SITE_COUNT] = {}; // of those, published (requested - published = lost)
    uint64_t n_auto_restore_draft_warm     = 0; // disk restores that brought the draft back from sidecars
    uint64_t n_auto_restore_draft_cold     = 0; // disk restores with a draft context left cold (a node lacked a sidecar)
    uint64_t n_auto_node_media_skipped     = 0; // mid-prefill nodes not armed: no text-after-text cut above the floor before a media chunk
    // system-node boundary (server_preamble_cache), copied from the HTTP side when metrics are read
    uint64_t n_sysnode_probed              = 0; // chat requests whose preamble end was looked up
    uint64_t n_sysnode_probe_renders       = 0; // of those, rendered (not cached)
    uint64_t n_sysnode_probe_failed        = 0; // no probe rendered: the message delimiters decide the node
    uint64_t n_sysnode_seam_mismatch       = 0; // preamble tokens not a prefix of the request: no node
    uint64_t n_sysnode_probe_short         = 0; // prompts too short in bytes to reach the node floor: not probed
};

// the auto disk cache's lines of server_task_result_metrics::to_metrics(): its counters after the upstream
// counters, its gauges after the upstream gauges and its labelled counters after both
void server_metrics_disk_cache_counters(std::ostream & prometheus, const server_metrics_disk_cache & metrics);
void server_metrics_disk_cache_gauges  (std::ostream & prometheus, const server_metrics_disk_cache & metrics);
void server_metrics_disk_cache_labelled(std::ostream & prometheus, const server_metrics_disk_cache & metrics);

// the same for a full-vocab logits row held outside a context (n_vocab entries)
std::vector<llama_token_data> get_token_probabilities_from_logits(const float * logits, int n_vocab, size_t n_top);

