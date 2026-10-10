// fork-only: the shared layer of the automatic disk KV cache, see server-disk-cache-common.h.

#include "server-common.h"
#include "server-task.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ostream>
#include <stdexcept>

//
// server_slot_stats
//

void server_slot_stats_disk_cache::to_json_disk_cache(common_json & base, uint64_t n_prompt_cached) const {
    if (n_prompt_cached_disk > 0) {
        // the part of cache_n that came from the auto disk cache rather than the resident slot
        base["cache_disk_n"] = n_prompt_cached_disk;
    }
    if (n_prompt_cached_ram > 0) {
        // the part of cache_n that came from the RAM prompt cache; cache_n minus both is the warm slot
        base["cache_ram_n"] = n_prompt_cached_ram;
    }
    // where the reused prompt came from, in one field: "disk" and "ram" when either supplied it (a disk
    // restore replaces the slot, so the two never mix), "warm" when only the resident slot did, "cold" when
    // nothing was reused
    base["cache_source"] = n_prompt_cached_disk > 0 ? "disk"
                         : n_prompt_cached_ram  > 0 ? "ram"
                         : n_prompt_cached      > 0 ? "warm" : "cold";
    if (n_prompt_cached_disk > 0 && cache_disk_unit_n > 0) {
        base["cache_disk_unit_n"] = cache_disk_unit_n; // the restored unit's length (cache_disk_n may trim it)
        base["cache_disk_nodes"]  = cache_disk_nodes;  // files on its chain: 1 for a whole root
        base["cache_disk_mode"]   = cache_disk_mode == 2 ? "side" : "whole";
    }
}

//
// server_tokens
//

const llama_tokens & server_tokens::get_cell_tokens() const {
    return tokens;
}

std::vector<server_media_record> server_tokens::extract_media_records() const {
    std::vector<server_media_record> records;
    records.reserve(map_idx_to_media.size());
    for (const auto & it : map_idx_to_media) {
        const auto * chunk = it.second.get();
        const auto   type  = mtmd_input_chunk_get_type(chunk);
        GGML_ASSERT(type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO);
        const char * id = mtmd_input_chunk_get_id(chunk);
        if (id == nullptr || id[0] == '\0') {
            // identity-less chunks (e.g. placeholder bitmaps) can never be re-verified
            throw std::runtime_error("media chunk has an empty id");
        }
        server_media_record rec;
        rec.start_idx = (uint32_t) it.first;
        rec.n_tokens  = (uint32_t) mtmd_input_chunk_get_n_tokens(chunk);
        rec.n_pos     = (uint32_t) mtmd_input_chunk_get_n_pos(chunk);
        rec.id        = id;
        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const auto * img = mtmd_input_chunk_get_tokens_image(chunk);
            // the raw token-grid shape is wanted here as an extra identity factor and to let
            // the restore path rebuild the exact geometry; get_grid returns it directly, unlike
            // mtmd_image_tokens_get_decoder_pos() which reinterprets the grid as a position
            mtmd_image_tokens_get_grid(img, &rec.nx, &rec.ny);
        } else {
            // audio has no 2D token grid; mirror the non-M-RoPE image convention
            rec.nx       = rec.n_tokens;
            rec.ny       = 1;
            rec.is_audio = 1;
        }
        records.push_back(std::move(rec));
    }
    return records;
}

bool server_tokens::boundary_is_chunk_safe(size_t idx) const {
    GGML_ASSERT(idx <= tokens.size());
    if (idx == tokens.size() || tokens[idx] != LLAMA_TOKEN_NULL) {
        return true; // one-past-the-end, or the cell at the split is a text token
    }
    // idx is a media cell: the split is safe only if a chunk starts exactly here.
    // The preceding cell's chunk membership is not evidence: adjacent chunks make
    // "previous cell is NULL" compatible with both safe and unsafe splits.
    return map_idx_to_media.find(idx) != map_idx_to_media.end();
}

bool boundary_is_chunk_safe(const llama_tokens & cells, const std::vector<server_media_record> & records, size_t idx) {
    GGML_ASSERT(idx <= cells.size());
    if (idx == cells.size() || cells[idx] != LLAMA_TOKEN_NULL) {
        return true; // one-past-the-end, or the cell at the split is a text token
    }
    // idx is a media cell: the split is safe only if a record starts exactly here
    // (records are ordered by start_idx, so binary-search)
    const auto it = std::lower_bound(records.begin(), records.end(), idx,
        [](const server_media_record & r, size_t v) { return r.start_idx < v; });
    return it != records.end() && it->start_idx == idx;
}

//
// auto disk cache .meta sidecar
//

std::string slot_meta_sidecar_path(const std::string & state_filepath) {
    return state_filepath + ".meta";
}

bool slot_meta_write(const std::string & state_filepath,
                     const model_fp & fp,
                     const llama_tokens & toks,
                     uint64_t chain_hash,
                     const std::vector<server_media_record> & media,
                     bool     is_node,
                     uint64_t parent_id,
                     uint32_t range_lo,
                     uint32_t range_hi) {
    // (is_node, media.empty()) selects v1/v2/v3/v4; the combination is_node && !media.empty()
    // is a v4 media delta node (media tail + node tail), no longer refused.
    // caps mirror the reader's: an over-cap or identity-less record would produce a
    // sidecar our own slot_meta_read rejects, so refuse to write it in the first place.
    if (media.size() > SLOT_META_MEDIA_MAX) {
        return false;
    }
    for (const auto & rec : media) {
        if (rec.id.empty() || rec.id.size() > SLOT_META_ID_MAX) {
            return false;
        }
    }
    const std::string sidecar = slot_meta_sidecar_path(state_filepath);
    const std::string tmp     = sidecar + ".tmp";

    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) {
        return false;
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
    auto put_u64 = [&](uint64_t v) {
        put_u32((uint32_t)(v & 0xFFFFFFFFu));
        put_u32((uint32_t)(v >> 32));
    };
    put_u32(SLOT_META_MAGIC);
    // single write-side version mapping point (the 2x2 of {whole|node} x {text|media}).
    put_u32(slot_meta_version_for(is_node, !media.empty()));
    put_u64(fp.fp_model);
    put_u32(fp.fp_n_vocab);
    put_u32(fp.fp_n_ctx_train);
    put_u32(fp.fp_n_embd);
    put_u32(fp.fp_n_layer);
    put_u32(fp.fp_rope_type);
    put_u32(fp.fp_cache_k);
    put_u32(fp.fp_cache_v);
    put_u32(fp.fp_n_ctx);
    put_u32(fp.fp_kv_layout);
    put_u32(fp.fp_block);
    put_u64(fp.fp_rope_scale);
    // rope_freq_base + YaRN fingerprint fields
    put_u64(fp.fp_rope_base);
    put_u32(fp.fp_yarn_ext);
    put_u32(fp.fp_yarn_attn);
    put_u32(fp.fp_yarn_beta_fast);
    put_u32(fp.fp_yarn_beta_slow);
    put_u32(fp.fp_yarn_orig_ctx);
    put_u64(fp.fp_lora);
    // mmproj deployment-shape bit: it refuses cross-shape restores.
    put_u32(fp.fp_mmproj_loaded);
    put_u32((uint32_t) toks.size());
    put_u64(chain_hash);
    // token IDs as raw LE int32 (llama_token == int32_t; llama.cpp's on-disk
    // contract is native-LE, matching slot_logits_write's float payload).
    f.write((const char *) toks.data(), (std::streamsize) toks.size() * sizeof(int32_t));
    // Tails are written media-then-node so the four versions are prefix-nested:
    //   v1 = header only; v2 = header + media tail; v3 = header + node tail;
    //   v4 = header + media tail + node tail. A whole text snapshot writes neither tail,
    // keeping v1 bytes byte-identical; v2 (media only) and v3 (node only) are unchanged.
    if (!media.empty()) {
    // v2/v4: appended media identity section (absent from text-only sidecars, which
    // stay byte-identical to v1).
        put_u64(fp.fp_mmproj);
        put_u32((uint32_t) media.size());
        for (const auto & rec : media) {
            put_u32(rec.start_idx);
            put_u32(rec.n_tokens);
            put_u32(rec.n_pos);
            put_u32(rec.nx);
            put_u32(rec.ny);
            put_u32(rec.is_audio);
            put_u32((uint32_t) rec.id.size());
            f.write(rec.id.data(), (std::streamsize) rec.id.size());
        }
    }
    if (is_node) {
    // v3/v4: appended delta-node section (parent link + covered cell range). A whole
    // snapshot never emits it.
        put_u64(parent_id);
        put_u32(range_lo);
        put_u32(range_hi);
    }
    f.flush();
    if (!f.good()) {
        f.close();
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        return false;
    }
    f.close();
    std::error_code ec;
    std::filesystem::rename(tmp, sidecar, ec); // atomic replace
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}

bool slot_meta_read(const std::string & state_filepath,
                    uint64_t cur_fp_mmproj,
                    model_fp & fp_out,
                    llama_tokens & toks_out,
                    std::vector<server_media_record> & media_out,
                    uint64_t * parent_out,
                    uint32_t * range_lo_out,
                    uint32_t * range_hi_out) {
    fp_out = model_fp{};
    toks_out.clear();
    media_out.clear();
    const std::string sidecar = slot_meta_sidecar_path(state_filepath);
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
    auto get_u64 = [&](uint64_t & v) -> bool {
        uint32_t lo = 0, hi = 0;
        if (!get_u32(lo) || !get_u32(hi)) {
            return false;
        }
        v = (uint64_t) lo | ((uint64_t) hi << 32);
        return true;
    };
    uint32_t magic = 0, version = 0;
    if (!get_u32(magic) || !get_u32(version)) {
        return false;
    }
    // single read-side version mapping point: whitelist + decode the feature set in one place.
    const slot_meta_features feat = slot_meta_features_for(version);
    if (magic != SLOT_META_MAGIC || !feat.valid) {
        return false;
    }
    model_fp fp;
    uint32_t tok_count = 0;
    uint64_t chain_hash = 0;
    if (!get_u64(fp.fp_model)         || !get_u32(fp.fp_n_vocab)       || !get_u32(fp.fp_n_ctx_train) ||
        !get_u32(fp.fp_n_embd)        || !get_u32(fp.fp_n_layer)       || !get_u32(fp.fp_rope_type)   ||
        !get_u32(fp.fp_cache_k)       || !get_u32(fp.fp_cache_v)       || !get_u32(fp.fp_n_ctx)       ||
        !get_u32(fp.fp_kv_layout)     || !get_u32(fp.fp_block)         || !get_u64(fp.fp_rope_scale)  ||
        // rope_freq_base + YaRN: must be read in the same order slot_meta_write emits.
        !get_u64(fp.fp_rope_base)     || !get_u32(fp.fp_yarn_ext)      || !get_u32(fp.fp_yarn_attn)   ||
        !get_u32(fp.fp_yarn_beta_fast)|| !get_u32(fp.fp_yarn_beta_slow)|| !get_u32(fp.fp_yarn_orig_ctx)||
        // mmproj deployment-shape bit: read in the same order slot_meta_write emits.
        !get_u64(fp.fp_lora)          || !get_u32(fp.fp_mmproj_loaded) ||
        !get_u32(tok_count)           || !get_u64(chain_hash)) {
        return false;
    }
    (void) chain_hash;
    // sanity-bound the count so a corrupt header cannot make us allocate gigabytes.
    if (tok_count > (1u << 28)) {
        return false;
    }
    toks_out.resize(tok_count);
    const std::streamsize want = (std::streamsize) tok_count * (std::streamsize) sizeof(int32_t);
    f.read((char *) toks_out.data(), want);
    if (f.gcount() != want) {
        toks_out.clear();
        return false;
    }
    // delta-node defaults: a v1/v2 whole snapshot is its own parentless root covering
    // [0, tok_count). The node formats (v3, v4) overwrite these from their node tail below.
    if (parent_out)   { *parent_out   = 0; }
    if (range_lo_out) { *range_lo_out = 0; }
    if (range_hi_out) { *range_hi_out = tok_count; }

    // Every violation below rejects the whole file (invariant 4: fall back to a normal
    // prefill, never trust a corrupt unit).
    auto fail = [&]() {
        toks_out.clear();
        media_out.clear();
        return false;
    };

    // The tails are read in the same media-then-node order the writer emits them, driven by
    // the decoded feature set. The media/text split (feat.has_media_tail) is what governs
    // NULL-cell acceptance and the fp_mmproj backfill.
    std::vector<server_media_record> media;
    if (feat.has_media_tail) {
        // v2/v4: appended media identity section. fp_mmproj is authoritative for media KV
        // (projector-dependent), read for real, never backfilled, and NULL (media) cells
        // are legitimate and MUST tile the records exactly.
        if (!get_u64(fp.fp_mmproj)) {
            return fail();
        }
        uint32_t n_media = 0;
        if (!get_u32(n_media) || n_media == 0 || n_media > SLOT_META_MEDIA_MAX) {
            return fail(); // a media sidecar with no records is never written
        }
        media.reserve(n_media);
        uint64_t next_free = 0; // first cell index not claimed by a previous record
        uint64_t n_covered = 0; // total cells claimed by records
        for (uint32_t r = 0; r < n_media; ++r) {
            server_media_record rec;
            uint32_t id_len = 0;
            if (!get_u32(rec.start_idx) || !get_u32(rec.n_tokens) || !get_u32(rec.n_pos) ||
                !get_u32(rec.nx)        || !get_u32(rec.ny)       || !get_u32(rec.is_audio) ||
                !get_u32(id_len)) {
                return fail();
            }
            // empty ids are refused at save time (an identity-less chunk can never be
            // re-verified), so they are equally invalid here.
            if (id_len == 0 || id_len > SLOT_META_ID_MAX) {
                return fail();
            }
            rec.id.resize(id_len);
            f.read(&rec.id[0], (std::streamsize) id_len);
            if (f.gcount() != (std::streamsize) id_len) {
                return fail();
            }
            // records must be non-empty, ordered by start_idx, disjoint (adjacent is
            // fine) and in-bounds; 64-bit arithmetic so start_idx + n_tokens cannot wrap.
            if (rec.n_tokens == 0 ||
                (uint64_t) rec.start_idx < next_free ||
                (uint64_t) rec.start_idx + rec.n_tokens > tok_count) {
                return fail();
            }
            // tiling invariant, half 1: every record cell is a media (NULL) cell.
            for (uint32_t i = rec.start_idx; i < rec.start_idx + rec.n_tokens; ++i) {
                if (toks_out[i] != LLAMA_TOKEN_NULL) {
                    return fail();
                }
            }
            next_free  = (uint64_t) rec.start_idx + rec.n_tokens;
            n_covered += rec.n_tokens;
            media.push_back(std::move(rec));
        }
        // tiling invariant, half 2: every NULL cell is covered by exactly one record.
        // Records are disjoint and cover only NULL cells (checked above), so covering
        // ALL of them is equivalent to the counts matching.
        uint64_t n_null = 0;
        for (const llama_token tok : toks_out) {
            if (tok == LLAMA_TOKEN_NULL) {
                ++n_null;
            }
        }
        if (n_covered != n_null) {
            return fail();
        }
    } else {
        // v1/v3: text-only token layout BY CONSTRUCTION: no text writer ever emits a NULL
        // (media) cell, so any NULL here means a corrupt or relabelled media sidecar (e.g. a
        // v2 file whose version byte flipped). Enforce that premise: the fp_mmproj backfill
        // below is only sound for genuinely text-only KV, and accepting NULL cells here would
        // silently drop the media identity records they stand for.
        for (const llama_token tok : toks_out) {
            if (tok == LLAMA_TOKEN_NULL) {
                return fail();
            }
        }
        // v1/v3 text KV is projector-independent: backfill the live value so the fingerprint
        // compare cannot refuse a text-only snapshot on an --mmproj server.
        fp.fp_mmproj = cur_fp_mmproj;
    }

    if (feat.has_node_tail) {
        // v3/v4 delta node: parent link + covered cell range, appended after any media tail.
        uint64_t node_parent = 0;
        uint32_t node_lo     = 0;
        uint32_t node_hi     = 0;
        if (!get_u64(node_parent) || !get_u32(node_lo) || !get_u32(node_hi)) {
            return fail();
        }
        if (parent_out)   { *parent_out   = node_parent; }
        if (range_lo_out) { *range_lo_out = node_lo; }
        if (range_hi_out) { *range_hi_out = node_hi; }
    }

    // every known layout ends exactly here: trailing bytes mean a relabelled/corrupt file
    // (and, for the text formats, the fp_mmproj backfill above must never apply to one).
    if (f.peek() != std::char_traits<char>::eof()) {
        return fail();
    }
    fp_out    = fp;
    media_out = std::move(media);
    return true;
}

//
// auto disk cache block chain hashing
//

// splitmix64 finalizer: avalanche a media cell's contribution before it enters the
// chain, so structured inputs (small slice offsets, similar ids) spread over all
// 64 bits and cannot resemble a plain token-ID fold.
static inline uint64_t auto_hash_splitmix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// Fold every restore_compatible field through the shared mix primitive. Order is fixed (any change
// re-names files, i.e. a cache miss, never corruption). n_ctx enters only as the LongRoPE regime
// bit, in the slot fp_n_ctx used to take, so rungs of one model share names (model_fp::fp_n_ctx).
// See the header for why this partitions the on-disk names but deliberately does NOT touch the
// fp_model-salted block chain.
uint64_t model_fp::identity_hash() const {
    uint64_t h = 0;
    h = auto_hash_mix64(h, fp_model);
    h = auto_hash_mix64(h, fp_n_vocab);
    h = auto_hash_mix64(h, fp_n_ctx_train);
    h = auto_hash_mix64(h, fp_n_embd);
    h = auto_hash_mix64(h, fp_n_layer);
    h = auto_hash_mix64(h, fp_rope_type);
    h = auto_hash_mix64(h, fp_cache_k);
    h = auto_hash_mix64(h, fp_cache_v);
    h = auto_hash_mix64(h, ctx_long_regime(*this) ? 1u : 0u); // live only: reads fp_n_ctx_orig
    h = auto_hash_mix64(h, fp_kv_layout);
    h = auto_hash_mix64(h, fp_block);
    h = auto_hash_mix64(h, fp_rope_scale);
    h = auto_hash_mix64(h, fp_rope_base);
    h = auto_hash_mix64(h, fp_yarn_ext);
    h = auto_hash_mix64(h, fp_yarn_attn);
    h = auto_hash_mix64(h, fp_yarn_beta_fast);
    h = auto_hash_mix64(h, fp_yarn_beta_slow);
    h = auto_hash_mix64(h, fp_yarn_orig_ctx);
    h = auto_hash_mix64(h, fp_lora);
    h = auto_hash_mix64(h, fp_mmproj_loaded);
    h = auto_hash_mix64(h, fp_mmproj);
    return h;
}

std::vector<uint64_t> auto_block_hashes(const llama_tokens & cells,
                                        const std::vector<server_media_record> & media,
                                        int B,
                                        uint64_t salt,
                                        uint64_t fp_mmproj,
                                        uint64_t * full,
                                        std::vector<uint64_t> * chain) {
    std::vector<uint64_t> out;
    if (full) {
        *full = 0;
    }
    if (chain) {
        chain->clear();
    }
    if (B <= 0) {
        return out;
    }
    out.reserve(cells.size() / (size_t) B);
    if (chain) {
        chain->reserve(cells.size());
    }
    size_t   ri        = 0; // index of the record covering the current NULL run
    bool     ri_seeded = false;
    uint64_t rec_seed  = 0; // cell-independent part of the record's contribution
    uint64_t h = 0xcbf29ce484222325ULL ^ salt; // FNV offset basis, fingerprint-salted
    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i] == LLAMA_TOKEN_NULL) {
            // advance to the covering record (records are ordered and tile the NULL cells)
            while (ri < media.size() && i >= (size_t) media[ri].start_idx + media[ri].n_tokens) {
                ++ri;
                ri_seeded = false;
            }
            GGML_ASSERT(ri < media.size() && i >= media[ri].start_idx && "NULL cell not covered by a media record");
            const server_media_record & rec = media[ri];
            if (!ri_seeded) {
                // fnv64 over the id bytes, then fold in the chunk shape/type and the
                // projector identity (see the header comment for why each factor is there)
                uint64_t id_h = 0xcbf29ce484222325ULL;
                for (const unsigned char c : rec.id) {
                    id_h ^= (uint64_t) c;
                    id_h *= 0x100000001b3ULL;
                }
                uint64_t shape = 0;
                shape = auto_hash_mix(shape, (int32_t) rec.n_tokens);
                shape = auto_hash_mix(shape, (int32_t) rec.n_pos);
                shape = auto_hash_mix(shape, (int32_t) rec.is_audio);
                rec_seed  = id_h ^ shape ^ fp_mmproj;
                ri_seeded = true;
            }
            h = auto_hash_mix64(h, auto_hash_splitmix64(rec_seed ^ (uint64_t) (i - rec.start_idx)));
        } else {
            h = auto_hash_mix(h, cells[i]);
        }
        if (chain) {
            chain->push_back(h);
        }
        // Emit an index key at a block-aligned position OR at the END of a media chunk. The
        // chunk-boundary emission is essential for media: one image chunk can span every
        // block-aligned position in a short prompt, so without it the snapshot has NO index key and
        // the media conversation gets no disk cache (bhs empty -> save skipped). A chunk end cuts no
        // chunk (always safe) and is a position any repeat/continuation recomputes identically, so it
        // is a valid restore key. Text prompts have no NULL cells, so at_chunk_end is never set and
        // text keys are byte-identical to before.
        bool at_chunk_end = false;
        if (cells[i] == LLAMA_TOKEN_NULL && ri < media.size()) {
            at_chunk_end = (i + 1 == (size_t) media[ri].start_idx + (size_t) media[ri].n_tokens);
        }
        if (((i + 1) % (size_t) B == 0 || at_chunk_end) && boundary_is_chunk_safe(cells, media, i + 1)) {
            out.push_back(h);
        }
    }
    if (full) {
        *full = h;
    }
    // No fallback key: a prompt with at least one media record always gets one at its last chunk end
    // (the cell after it is text, another record's start, or the end, so that boundary is chunk-safe),
    // and a text prompt shorter than one block gets none, exactly as in the pre-media chain. Such a
    // prompt is below the save floor (max(block, min-tokens)), so no snapshot keyed by it can exist.
    return out;
}

//
// system-prompt node boundary (server_preamble_cache)
//

int32_t server_preamble_cache::chars(const common_chat_templates * tmpls, const common_chat_templates_inputs & inputs, const std::string & prompt) {
    // a sound bound, not an estimate: a token covers at least one byte of the rendered text (BOS and
    // other added tokens aside, hence the slack), so a shorter prompt can never reach the floor
    if (prompt.size() + 8 < (size_t) std::max(0, min_tokens)) {
        n_probe_short++;
        return -1;
    }
    n_probed++;

    // everything that renders into the preamble: the leading system / developer messages, the tools,
    // the json_schema and the flags and kwargs the template reads (not the clock: the stored preamble text is
    // compared instead, which also catches a template that renders the date)
    size_t k = 0;
    while (k < inputs.messages.size() &&
           (inputs.messages[k].role == "system" || inputs.messages[k].role == "developer")) {
        k++;
    }
    std::string key = std::to_string((uintptr_t) tmpls);
    key += common_chat_msgs_to_json_oaicompat(std::vector<common_chat_msg>(inputs.messages.begin(), inputs.messages.begin() + k)).dump();
    key += common_chat_tools_to_json_oaicompat(inputs.tools).dump();
    key += "|" + inputs.json_schema; // rendered into the probes: some templates write it into the system turn
    key += string_format("|%d|%d|%d|%d|%d|%d|%d|%d", (int) inputs.tool_choice, (int) inputs.parallel_tool_calls,
                         (int) inputs.reasoning_format, (int) inputs.enable_thinking, (int) inputs.use_jinja,
                         (int) inputs.add_bos, (int) inputs.add_eos, (int) inputs.force_pure_content);
    for (const auto & kv : inputs.chat_template_kwargs) {
        key += "|" + kv.first + "=" + kv.second;
    }
    const size_t h = std::hash<std::string>{}(key);

    {
        std::lock_guard<std::mutex> lock(mtx);
        for (size_t i = by_inputs.size(); i-- > 0; ) {
            const auto & e = by_inputs[i];
            if (e.key == h && prompt.size() >= (size_t) e.n_chars && prompt.compare(0, e.n_chars, e.preamble) == 0) {
                const int32_t n = e.n_chars;
                std::rotate(by_inputs.begin() + i, by_inputs.begin() + i + 1, by_inputs.end());
                return n;
            }
        }
    }

    n_probe_renders++;
    const int32_t n = common_chat_preamble_end(tmpls, inputs, prompt);
    if (n < 0) {
        n_probe_failed++;
        return -1;
    }

    std::lock_guard<std::mutex> lock(mtx);
    for (auto it = by_inputs.begin(); it != by_inputs.end(); ++it) {
        if (it->key == h) {
            by_inputs.erase(it);
            break;
        }
    }
    if (by_inputs.size() >= N_ENTRIES) {
        by_inputs.erase(by_inputs.begin());
    }
    by_inputs.push_back({ h, n, prompt.substr(0, n) });
    return n;
}

// tokenize like the completion path does (BOS added, special tokens parsed), without a trailing EOS:
// the preamble is a prefix, not a whole prompt
static llama_tokens preamble_tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    llama_tokens t = common_tokenize(vocab, text, add_special, true);
    if (add_special && llama_vocab_get_add_eos(vocab) && !t.empty() && t.back() == llama_vocab_eos(vocab)) {
        t.pop_back();
    }
    return t;
}

static size_t tokens_lcp(const llama_tokens & a, size_t a0, const llama_tokens & b) {
    size_t n = 0;
    while (a0 + n < a.size() && n < b.size() && a[a0 + n] == b[n]) {
        n++;
    }
    return n;
}

// The longest prefix of `pre` (the preamble's tokens) that no continuation of the text can change.
// A special token ends a fragment of its own, so nothing merges into one. A plain-text tail (a role
// header that is not a special token, or a newline after one) can merge with what follows, e.g. a
// trailing "\n" with a message starting "\n", or "<user>" with "Hello" into "<", "user", ">Hello":
// the tail is tokenized again followed by one character of each pre-tokenizer class, and the result
// is cut back to the shortest common prefix. Depends on the preamble text only.
static size_t preamble_seam(const llama_vocab * vocab, const std::string & preamble, const llama_tokens & pre) {
    const size_t n = pre.size();
    const auto is_special = [&](llama_token t) {
        return (llama_vocab_get_attr(vocab, t) & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) != 0;
    };
    if (n == 0 || is_special(pre[n - 1])) {
        return n;
    }

    // the plain-text fragment after the last special token is tokenized on its own, so it can be
    // re-tokenized alone; checked, and the whole preamble is used when it does not reproduce
    size_t j = n - 1;
    while (j > 0 && !is_special(pre[j - 1])) {
        j--;
    }
    std::string tail;
    for (size_t i = j; i < n; i++) {
        tail += common_token_to_piece(vocab, pre[i], true);
    }
    const bool local = j > 0 && preamble.size() >= tail.size() &&
        preamble.compare(preamble.size() - tail.size(), tail.size(), tail) == 0 &&
        preamble_tokenize(vocab, tail, false) == llama_tokens(pre.begin() + j, pre.end());

    static const std::vector<std::string> conts_local = {
        "A", "a", "1", " ", " A", "\n", "\n\n", "\t", "<", ">", "#", "(", "\"", "'", "{", "-", "*", ".",
        "\xe4\xb8\x80" /* CJK */, "\xf0\x9f\x98\x80" /* emoji */,
    };
    static const std::vector<std::string> conts_full = { "A", "1", " ", "\n", "<" };

    size_t x = n;
    if (local) {
        for (const auto & c : conts_local) {
            x = std::min(x, j + tokens_lcp(pre, j, preamble_tokenize(vocab, tail + c, false)));
        }
    } else {
        for (const auto & c : conts_full) {
            x = std::min(x, tokens_lcp(pre, 0, preamble_tokenize(vocab, preamble + c, true)));
        }
    }
    return x;
}

int32_t server_preamble_cache::tokens(const llama_vocab * vocab, const std::string & prompt, int32_t n_chars, const llama_tokens & prompt_tokens) {
    if (n_chars <= 0 || (size_t) n_chars > prompt.size()) {
        return -1;
    }

    llama_tokens toks;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (size_t i = by_text.size(); i-- > 0; ) {
            const auto & e = by_text[i];
            if (e.preamble.size() == (size_t) n_chars && prompt.compare(0, n_chars, e.preamble) == 0) {
                toks = e.toks;
                found = true;
                std::rotate(by_text.begin() + i, by_text.begin() + i + 1, by_text.end());
                break;
            }
        }
    }
    if (!found) {
        const std::string preamble = prompt.substr(0, n_chars);
        llama_tokens pre = preamble_tokenize(vocab, preamble, true);
        pre.resize(preamble_seam(vocab, preamble, pre));
        toks = pre;
        std::lock_guard<std::mutex> lock(mtx);
        if (by_text.size() >= N_ENTRIES) {
            by_text.erase(by_text.begin());
        }
        by_text.push_back({ preamble, std::move(pre) });
    }

    if (toks.empty()) {
        return -1;
    }
    if (prompt_tokens.size() < toks.size() ||
        !std::equal(toks.begin(), toks.end(), prompt_tokens.begin())) {
        n_seam_mismatch++;
        return -1;
    }
    return (int32_t) toks.size();
}

//
// other utils
//

static void token_probabilities_finish(std::vector<llama_token_data> & cur, size_t n_top);

std::vector<llama_token_data> get_token_probabilities_from_logits(const float * logits, int n_vocab, size_t n_top) {
    std::vector<llama_token_data> cur((size_t) std::max(0, n_vocab));
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }
    token_probabilities_finish(cur, n_top);
    return cur;
}

// sort by logit (only the leading `n_top` need ordering) and turn the logits into probabilities
static void token_probabilities_finish(std::vector<llama_token_data> & cur, size_t n_top) {
    // sort tokens by logits (partial: only the leading `n_top` need ordering)
    if (n_top > cur.size()) {
        n_top = cur.size();
    }
    if (n_top > 0) {
        std::partial_sort(cur.begin(), cur.begin() + n_top, cur.end(),
            [](const llama_token_data & a, const llama_token_data & b) {
                return a.logit > b.logit;
            });
    }

    // apply softmax
    float max_l = -std::numeric_limits<float>::infinity();
    if (n_top > 0) {
        max_l = cur[0].logit; // partial_sort guarantees the absolute maximum is at index 0
    } else {
        for (const auto & t : cur) {
            max_l = std::max(max_l, t.logit);
        }
    }
    float cum_sum = 0.0f;
    for (auto & t : cur) {
        float p = expf(t.logit - max_l);
        t.p = p;
        cum_sum += p;
    }
    for (auto & t : cur) {
        t.p /= cum_sum;
    }
}

//
// /metrics: the auto disk cache's items of server_task_result_metrics::to_metrics()
//

using metric_item = server_task_result_metrics::metric_item;

// the add_items lambda of to_metrics()
static void add_items(std::ostream & prometheus, const char * type, const std::vector<metric_item> & items) {
    for (const auto & item : items) {
        prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
                   << "# TYPE llamacpp:" << item.name << " " << type             << "\n"
                   << "llamacpp:"        << item.name << " " << item.value       << "\n";
    }
}

void server_metrics_disk_cache_counters(std::ostream & prometheus, const server_metrics_disk_cache & metrics) {
    const std::vector<metric_item> counters = {
        {
            "auto_cache_save_whole_fallback_total",
            "Auto disk KV cache: parented saves published as whole roots because this memory type cannot write deltas",
            (double) metrics.n_auto_save_whole_fallback
        },
        {
            "auto_cache_save_failed_total",
            "Auto disk KV cache: saves dropped with nothing published (no space, IO error, refused); each logs a rate-limited WRN with the reason",
            (double) metrics.n_auto_save_failed
        },
        {
            "auto_cache_save_root_total",
            "Auto disk KV cache: whole roots published (includes whole fallbacks)",
            (double) metrics.n_auto_save_root
        },
        {
            "auto_cache_save_delta_total",
            "Auto disk KV cache: delta nodes published",
            (double) metrics.n_auto_save_delta
        },
        {
            "auto_cache_save_bytes_total",
            "Auto disk KV cache: state bytes written by published saves",
            (double) metrics.n_auto_save_bytes
        },
        {
            "auto_cache_evicted_total",
            "Auto disk KV cache: units this instance evicted from the shared store to stay under the caps",
            (double) metrics.n_auto_cache_evicted
        },
        {
            "auto_cache_restore_hit_total",
            "Auto disk KV cache: requests that restored a prefix from disk",
            (double) metrics.n_auto_restore_hit
        },
        {
            "auto_cache_restore_miss_total",
            "Auto disk KV cache: requests with at least one whole block beyond the in-memory match that restored nothing (includes new prompts)",
            (double) metrics.n_auto_restore_miss
        },
        {
            "auto_cache_restore_failed_total",
            "Auto disk KV cache: restores that failed after clearing the slot (fell back to a shorter snapshot or a cold prefill)",
            (double) metrics.n_auto_restore_failed
        },
        {
            "auto_cache_restore_not_prefix_total",
            "Auto disk KV cache: misses where a snapshot of the prefix existed but the request diverged inside it and the memory class cannot rewind (needs a node at or before the divergence)",
            (double) metrics.n_auto_restore_not_prefix
        },
        {
            "auto_cache_restore_miss_identity_total",
            "Auto disk KV cache: misses where a unit of the same model held the prefix under another identity (a rung with other RoPE/YaRN settings, cache types, mmproj, LoRA or block size)",
            (double) metrics.n_auto_restore_miss_identity
        },
        {
            "auto_cache_restore_side_only_total",
            "Auto disk KV cache: restores that loaded only a unit's side-state and kept the slot's own positional cells",
            (double) metrics.n_auto_restore_side_only
        },
        {
            "auto_cache_skipped_shared_total",
            "Auto disk KV cache: tasks with a shared prompt prefix (n_tokens_shared, decision tasks) that the cache neither restores nor saves",
            (double) metrics.n_auto_skipped_shared
        },
        {
            "auto_cache_prefill_checkpoint_superseded_total",
            "Auto disk KV cache: whole periodic prefill checkpoints removed because a deeper one of the same prefill replaced them",
            (double) metrics.n_auto_ckpt_superseded
        },
        {
            "auto_cache_evict_bound_exceeded_total",
            "Auto disk KV cache: eviction passes that left the store above a cap because every remaining unit has a live child or is pinned",
            (double) metrics.n_auto_cache_bound_exceeded
        },
        {
            "auto_cache_restore_tokens_total",
            "Auto disk KV cache: prompt tokens restored from disk",
            (double) metrics.n_auto_restore_tokens
        },
        {
            "auto_cache_restore_discarded_total",
            "Auto disk KV cache: restores whose tokens were re-prefilled before use (also counted as misses)",
            (double) metrics.n_auto_restore_discarded
        },
        {
            "auto_cache_save_draft_total",
            "Auto disk KV cache: draft-context (.dft) sidecars written next to a unit",
            (double) metrics.n_auto_save_draft
        },
        {
            "auto_cache_save_draft_skipped_total",
            "Auto disk KV cache: units published without a draft sidecar although a draft context exists",
            (double) metrics.n_auto_save_draft_skipped
        },
        {
            "auto_cache_restore_draft_warm_total",
            "Disk KV restores that brought the speculative draft back warm from .dft sidecars",
            (double) metrics.n_auto_restore_draft_warm
        },
        {
            "auto_cache_restore_draft_cold_total",
            "Disk KV restores that left the speculative draft cold (some node had no .dft sidecar)",
            (double) metrics.n_auto_restore_draft_cold
        },
        {
            "auto_cache_node_media_skipped_total",
            "Auto disk KV cache: mid-prefill nodes (system or prompt) not armed because media left no cut above the floor whose previous cell is text",
            (double) metrics.n_auto_node_media_skipped
        },
        {
            "auto_cache_sysnode_probed_total",
            "Auto disk KV cache: chat requests whose system-prompt end was looked up for the system node",
            (double) metrics.n_sysnode_probed
        },
        {
            "auto_cache_sysnode_probe_renders_total",
            "Auto disk KV cache: of those, requests that rendered the boundary probes (not cached)",
            (double) metrics.n_sysnode_probe_renders
        },
        {
            "auto_cache_sysnode_probe_failed_total",
            "Auto disk KV cache: requests where no boundary probe rendered (the message delimiters place the system node)",
            (double) metrics.n_sysnode_probe_failed
        },
        {
            "auto_cache_sysnode_seam_mismatch_total",
            "Auto disk KV cache: requests whose system-prompt tokens were not a prefix of the prompt (no system node)",
            (double) metrics.n_sysnode_seam_mismatch
        },
        {
            "auto_cache_sysnode_probe_short_total",
            "Auto disk KV cache: chat prompts too short in bytes to reach the system node's floor, so not probed",
            (double) metrics.n_sysnode_probe_short
        },
        {
            "auto_cache_save_queued_total",
            "Auto disk KV cache: saves handed to the background writer after their copy off the device",
            (double) metrics.n_auto_save_queued
        },
        {
            "auto_cache_save_streamed_total",
            "Auto disk KV cache: of those, saves larger than the free staging, streamed through two chunks while the writer was idle",
            (double) metrics.n_auto_save_streamed
        },
        {
            "auto_cache_save_dropped_staging_total",
            "Auto disk KV cache: saves dropped because the background writer made no progress for 60 s while the capture waited for staging; each logs a WRN",
            (double) metrics.n_auto_save_dropped_staging
        },
        {
            "auto_cache_save_admission_waits_total",
            "Auto disk KV cache: captures that waited for the background writer because the staging (--slot-save-staging-mb) was full and the writer busy",
            (double) metrics.n_auto_save_admission_waits
        },
        {
            "auto_cache_save_admission_wait_seconds_total",
            "Auto disk KV cache: total time those captures waited",
            (double) metrics.n_auto_save_admission_wait_us / 1e6
        },
        {
            "auto_cache_save_deferred_total",
            "Auto disk KV cache: saves whose positional K/V was copied after the capture (stage 2); only the side-state was copied at the capture",
            (double) metrics.n_auto_save_deferred
        },
        {
            "auto_cache_save_deferred_forced_total",
            "Auto disk KV cache: times a cache mutation (rewind, clear, shift, load, slot reuse, sleep) forced a pending deferred copy first",
            (double) metrics.n_auto_save_deferred_forced
        },
        {
            "auto_cache_save_deferred_bytes_total",
            "Auto disk KV cache: positional bytes deferred captures left in the cache to copy later",
            (double) metrics.n_auto_save_deferred_bytes
        },
        {
            "auto_cache_save_orphan_dropped_total",
            "Auto disk KV cache: queued deltas dropped because their parent failed to publish or left the store; each logs a WRN",
            (double) metrics.n_auto_save_orphan_dropped
        },
        {
            "auto_cache_save_shutdown_abandoned_total",
            "Auto disk KV cache: queued saves abandoned at the shutdown deadline (temps removed, nothing published); each logs a WRN",
            (double) metrics.n_auto_save_shutdown_abandoned
        },
    };

    add_items(prometheus, "counter", counters);
}

void server_metrics_disk_cache_gauges(std::ostream & prometheus, const server_metrics_disk_cache & metrics) {
    const std::vector<metric_item> gauges = {
        {
            "auto_cache_save_staging_bytes",
            "Auto disk KV cache: host bytes held by saves copied off the device and not yet written",
            (double) metrics.n_auto_save_staging_bytes
        }, {
            "auto_cache_save_queue_depth",
            "Auto disk KV cache: saves queued for the background writer or being written",
            (double) metrics.n_auto_save_queue_depth
        }, {
            "auto_cache_save_deferred_pending",
            "Auto disk KV cache: deferred captures still owing positional bytes",
            (double) metrics.n_auto_save_deferred_pending
        }, {
            "auto_cache_delta_capable",
            "Auto disk KV cache: the delta probe, 0 not probed yet (no save so far), 1 deltas, 2 whole roots only",
            (double) metrics.auto_delta_capable
        }, {
            "auto_cache_save_deferred_host_bytes",
            "Auto disk KV cache: host bytes (side-state) held by deferred captures not yet emitted",
            (double) metrics.n_auto_save_deferred_host
        },
    };

    add_items(prometheus, "gauge", gauges);
}

void server_metrics_disk_cache_labelled(std::ostream & prometheus, const server_metrics_disk_cache & metrics) {
    // labeled counters: units each save site asked for, and how many of them were published. A unit that
    // was requested and never published was lost (failed, orphaned, abandoned or dropped).
    {
        const struct {
            const char * name;
            const char * help;
            const uint64_t * v;
        } site_items[] = {
            { "auto_cache_save_site_requested_total", "Auto disk KV cache: units a save site decided to write",
              metrics.n_auto_save_site_requested },
            { "auto_cache_save_site_published_total", "Auto disk KV cache: of those, units published",
              metrics.n_auto_save_site_published },
        };
        for (const auto & it : site_items) {
            prometheus << "# HELP llamacpp:" << it.name << " " << it.help << "\n"
                       << "# TYPE llamacpp:" << it.name << " counter\n";
            for (int i = 0; i < AUTO_SAVE_SITE_COUNT; i++) {
                prometheus << "llamacpp:" << it.name << "{site=\"" << auto_save_site_name(i) << "\"} " << it.v[i] << "\n";
            }
        }
    }
}
