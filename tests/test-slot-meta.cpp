// tests for the auto disk cache's .meta sidecar format (slot_meta_write/slot_meta_read):
// v1 byte-layout freeze, v1 fp_mmproj backfill, v2 media-record round-trip, structured
// rejection cases (caps, ordering, tiling invariant, truncation, trailing bytes) and a
// seeded mutation fuzzer over the parser (which reads untrusted on-disk bytes).

#include "server-common.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <process.h>
#else
#    include <unistd.h>
#endif

namespace fs = std::filesystem;

// fixed v1 layout size: 2 u32 (magic/version) + fingerprint fields + tok_count + chain_hash
static constexpr size_t V1_FIXED = 116;

static fs::path g_dir;

static std::string state_path(const char * name) {
    return (g_dir / name).string();
}

static std::vector<char> read_meta_bytes(const std::string & state_filepath) {
    std::ifstream f(slot_meta_sidecar_path(state_filepath), std::ios::binary);
    GGML_ASSERT(f);
    return std::vector<char>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

static void write_meta_bytes(const std::string & state_filepath, const std::vector<char> & bytes) {
    std::ofstream f(slot_meta_sidecar_path(state_filepath), std::ios::binary | std::ios::trunc);
    GGML_ASSERT(f);
    f.write(bytes.data(), (std::streamsize) bytes.size());
    GGML_ASSERT(f.good());
}

static void patch_u32(std::vector<char> & bytes, size_t off, uint32_t v) {
    GGML_ASSERT(off + 4 <= bytes.size());
    bytes[off + 0] = (char) ( v        & 0xFF);
    bytes[off + 1] = (char) ((v >> 8)  & 0xFF);
    bytes[off + 2] = (char) ((v >> 16) & 0xFF);
    bytes[off + 3] = (char) ((v >> 24) & 0xFF);
}

static model_fp make_fp() {
    model_fp fp;
    fp.fp_model          = 0x1122334455667788ULL;
    fp.fp_n_vocab        = 32000;
    fp.fp_n_ctx_train    = 2048;
    fp.fp_n_embd         = 64;
    fp.fp_n_layer        = 5;
    fp.fp_rope_type      = 2;
    fp.fp_cache_k        = 1;
    fp.fp_cache_v        = 1;
    fp.fp_n_ctx          = 512;
    fp.fp_kv_full        = 0;
    fp.fp_block          = 256;
    fp.fp_rope_scale     = 0x3F800000ULL;
    fp.fp_rope_base      = 0x461C4000ULL;
    fp.fp_yarn_ext       = 1;
    fp.fp_yarn_attn      = 2;
    fp.fp_yarn_beta_fast = 3;
    fp.fp_yarn_beta_slow = 4;
    fp.fp_yarn_orig_ctx  = 5;
    fp.fp_lora           = 0xA5A5A5A5A5A5A5A5ULL;
    fp.fp_mmproj_loaded  = 1;
    fp.fp_mmproj         = 0xDEADBEEFCAFEF00DULL;
    return fp;
}

// re-check everything slot_meta_read promises about its outputs; used to validate any
// ACCEPTED fuzz input (a parser that accepts is fine — but only if the invariants hold).
static void check_read_invariants(const llama_tokens & toks, const std::vector<server_media_record> & media) {
    uint64_t next_free = 0;
    uint64_t n_covered = 0;
    for (const auto & rec : media) {
        GGML_ASSERT(!rec.id.empty() && rec.id.size() <= SLOT_META_ID_MAX);
        GGML_ASSERT(rec.n_tokens >= 1);
        GGML_ASSERT((uint64_t) rec.start_idx >= next_free);
        GGML_ASSERT((uint64_t) rec.start_idx + rec.n_tokens <= toks.size());
        for (uint32_t i = rec.start_idx; i < rec.start_idx + rec.n_tokens; ++i) {
            GGML_ASSERT(toks[i] == LLAMA_TOKEN_NULL);
        }
        next_free  = (uint64_t) rec.start_idx + rec.n_tokens;
        n_covered += rec.n_tokens;
    }
    uint64_t n_null = 0;
    for (const llama_token tok : toks) {
        if (tok == LLAMA_TOKEN_NULL) {
            ++n_null;
        }
    }
    GGML_ASSERT(n_covered == n_null);
    GGML_ASSERT(media.size() <= SLOT_META_MEDIA_MAX);
}

int main(void) {
    g_dir = fs::temp_directory_path() / ("test-slot-meta-" + std::to_string((long) getpid()));
    fs::create_directories(g_dir);

    const model_fp fp = make_fp();

    // --- v1 round-trip + byte-layout freeze -------------------------------------------------
    const llama_tokens text_toks = { 11, 22, 33, 44, 55, 66, 77 };
    const std::string  v1_path   = state_path("v1.bin");
    {
        GGML_ASSERT(slot_meta_write(v1_path, fp, text_toks, 0x0123456789ABCDEFULL));
        // layout freeze: the v1 sidecar is byte-frozen at V1_FIXED + 4 bytes/token — any
        // growth here would break invariant 0 (text-only output byte-identical).
        GGML_ASSERT(read_meta_bytes(v1_path).size() == V1_FIXED + 4 * text_toks.size());

        model_fp rfp;
        llama_tokens rtoks;
        std::vector<server_media_record> rmedia;
        GGML_ASSERT(slot_meta_read(v1_path, /*cur_fp_mmproj=*/0x77ULL, rfp, rtoks, rmedia));
        GGML_ASSERT(rtoks == text_toks);
        GGML_ASSERT(rmedia.empty());
        // v1 carries no fp_mmproj: it must be BACKFILLED from the live value, so the full
        // fingerprint equality (which now includes fp_mmproj) can only pass when the rest
        // of the fingerprint matches.
        GGML_ASSERT(rfp.fp_mmproj == 0x77ULL);
        model_fp expect = fp;
        expect.fp_mmproj = 0x77ULL;
        GGML_ASSERT(rfp == expect);
        GGML_ASSERT(!(rfp == fp)); // differing fp_mmproj now refuses
    }

    // --- v2 round-trip (adjacent image + audio records) --------------------------------------
    // cells: [3 text][8 NULL: image "img_a"][4 NULL: audio "aud_b"][2 text]
    llama_tokens media_toks = { 100, 101, 102 };
    media_toks.insert(media_toks.end(), 12, LLAMA_TOKEN_NULL);
    media_toks.push_back(103);
    media_toks.push_back(104);
    std::vector<server_media_record> media(2);
    media[0] = { /*start_idx=*/3,  /*n_tokens=*/8, /*n_pos=*/5, /*nx=*/4, /*ny=*/2, /*is_audio=*/0, "img_a" };
    media[1] = { /*start_idx=*/11, /*n_tokens=*/4, /*n_pos=*/4, /*nx=*/4, /*ny=*/1, /*is_audio=*/1, "aud_b" };
    const std::string v2_path = state_path("v2.bin");
    {
        GGML_ASSERT(slot_meta_write(v2_path, fp, media_toks, 0xFEDCBA9876543210ULL, media));

        model_fp rfp;
        llama_tokens rtoks;
        std::vector<server_media_record> rmedia;
        GGML_ASSERT(slot_meta_read(v2_path, /*cur_fp_mmproj=*/0x77ULL, rfp, rtoks, rmedia));
        GGML_ASSERT(rtoks == media_toks);
        GGML_ASSERT(rfp.fp_mmproj == fp.fp_mmproj); // v2 carries the real value — no backfill
        GGML_ASSERT(rfp == fp);
        GGML_ASSERT(rmedia.size() == 2);
        for (size_t i = 0; i < 2; ++i) {
            GGML_ASSERT(rmedia[i].start_idx == media[i].start_idx);
            GGML_ASSERT(rmedia[i].n_tokens  == media[i].n_tokens);
            GGML_ASSERT(rmedia[i].n_pos     == media[i].n_pos);
            GGML_ASSERT(rmedia[i].nx        == media[i].nx);
            GGML_ASSERT(rmedia[i].ny        == media[i].ny);
            GGML_ASSERT(rmedia[i].is_audio  == media[i].is_audio);
            GGML_ASSERT(rmedia[i].id        == media[i].id);
        }
        check_read_invariants(rtoks, rmedia);
    }

    // --- v4 round-trip (media delta node = v2 media tail + v3 node tail) ----------------------
    // Same cells + records as v2, now emitted as an incremental MEDIA delta node: the meta carries
    // the WHOLE [0,N) tiling and cell-token array (exactly like v2) while its .bin (not written by
    // this format test) would hold only the delta cells [range_lo, range_hi). Exercises the media-
    // then-node tail order, the real (non-backfilled) fp_mmproj alongside NULL cells, the node-field
    // out-ptrs and the exact-EOF requirement — the ONLY format carrying both a media tail and a
    // node tail. A whole v2 snapshot must still default the node fields to a parentless root.
    const std::string v4_path   = state_path("v4.bin");
    const uint64_t     v4_parent = 0xABCDEF0123456789ULL;
    const uint32_t     v4_lo     = 3;                            // delta from the first media chunk on
    const uint32_t     v4_hi     = (uint32_t) media_toks.size(); // .. to the end
    const size_t       v2_meta_size = read_meta_bytes(v2_path).size();
    {
        GGML_ASSERT(slot_meta_write(v4_path, fp, media_toks, 0x0011223344556677ULL, media,
                                    /*is_node=*/true, v4_parent, v4_lo, v4_hi));
        // byte layout: exactly the v2 file (v1 header + tokens + media tail) followed by the 16-byte
        // node tail (parent_id u64 + range_lo u32 + range_hi u32) — media-then-node, prefix-nested.
        GGML_ASSERT(read_meta_bytes(v4_path).size() == v2_meta_size + 16);

        model_fp rfp;
        llama_tokens rtoks;
        std::vector<server_media_record> rmedia;
        uint64_t rparent = 123; uint32_t rlo = 123, rhi = 123;
        GGML_ASSERT(slot_meta_read(v4_path, /*cur_fp_mmproj=*/0x77ULL, rfp, rtoks, rmedia,
                                   &rparent, &rlo, &rhi));
        GGML_ASSERT(rtoks == media_toks);
        GGML_ASSERT(rfp.fp_mmproj == fp.fp_mmproj); // v4 carries the real value — no backfill
        GGML_ASSERT(rfp == fp);
        GGML_ASSERT(rmedia.size() == 2);
        for (size_t i = 0; i < 2; ++i) {
            GGML_ASSERT(rmedia[i].start_idx == media[i].start_idx);
            GGML_ASSERT(rmedia[i].n_tokens  == media[i].n_tokens);
            GGML_ASSERT(rmedia[i].n_pos     == media[i].n_pos);
            GGML_ASSERT(rmedia[i].nx        == media[i].nx);
            GGML_ASSERT(rmedia[i].ny        == media[i].ny);
            GGML_ASSERT(rmedia[i].is_audio  == media[i].is_audio);
            GGML_ASSERT(rmedia[i].id        == media[i].id);
        }
        GGML_ASSERT(rparent == v4_parent && rlo == v4_lo && rhi == v4_hi);
        check_read_invariants(rtoks, rmedia);

        // a WHOLE v2 snapshot read with the node out-ptrs defaults them to a parentless root
        // covering [0, tok_count) — the whole/delta distinction is carried by the version byte.
        model_fp wfp; llama_tokens wtoks; std::vector<server_media_record> wmedia;
        uint64_t wparent = 7; uint32_t wlo = 7, whi = 7;
        GGML_ASSERT(slot_meta_read(v2_path, 0x77ULL, wfp, wtoks, wmedia, &wparent, &wlo, &whi));
        GGML_ASSERT(wparent == 0 && wlo == 0 && whi == (uint32_t) media_toks.size());
    }

    // --- write-side refusals ------------------------------------------------------------------
    {
        const std::string p = state_path("refuse.bin");
        auto bad = media;
        bad[0].id.clear(); // empty id: unverifiable -> never persisted
        GGML_ASSERT(!slot_meta_write(p, fp, media_toks, 0, bad));
        bad = media;
        bad[0].id.assign(SLOT_META_ID_MAX + 1, 'x'); // over the id cap
        GGML_ASSERT(!slot_meta_write(p, fp, media_toks, 0, bad));
        std::vector<server_media_record> too_many(SLOT_META_MEDIA_MAX + 1, media[0]); // over the record cap
        GGML_ASSERT(!slot_meta_write(p, fp, media_toks, 0, too_many));
        GGML_ASSERT(!fs::exists(slot_meta_sidecar_path(p))); // refused before any byte hit disk
    }

    // --- structured read rejections -----------------------------------------------------------
    const std::vector<char> v1_bytes = read_meta_bytes(v1_path);
    const std::vector<char> v2_bytes = read_meta_bytes(v2_path);
    const size_t v1_end   = V1_FIXED + 4 * media_toks.size(); // end of the v1 part of the v2 file
    const size_t off_nm   = v1_end + 8;                       // n_media (after fp_mmproj)
    const size_t off_rec0 = off_nm + 4;                       // record 0: start_idx
    const size_t off_rec1 = off_rec0 + 28 + media[0].id.size(); // record 1: start_idx
    GGML_ASSERT(v2_bytes.size() == off_rec1 + 28 + media[1].id.size());

    const std::string mut_path = state_path("mut.bin");
    auto expect_reject = [&](std::vector<char> bytes) {
        write_meta_bytes(mut_path, bytes);
        model_fp rfp;
        llama_tokens rtoks;
        std::vector<server_media_record> rmedia;
        GGML_ASSERT(!slot_meta_read(mut_path, 0, rfp, rtoks, rmedia));
        GGML_ASSERT(rtoks.empty() && rmedia.empty()); // outputs cleared on rejection
    };
    {
        auto b = v2_bytes; patch_u32(b, 0, 0x12345678u);              expect_reject(b); // bad magic
        b = v2_bytes; patch_u32(b, 4, 3u);                            expect_reject(b); // unknown version
        b = v2_bytes; patch_u32(b, 4, 99u);                           expect_reject(b); // unknown version
        b = v2_bytes; patch_u32(b, off_nm, 0u);                       expect_reject(b); // n_media == 0
        b = v2_bytes; patch_u32(b, off_nm, SLOT_META_MEDIA_MAX + 1);  expect_reject(b); // n_media over cap
        b = v2_bytes; patch_u32(b, off_rec0 + 24, 0u);                expect_reject(b); // id_len == 0
        b = v2_bytes; patch_u32(b, off_rec0 + 24, SLOT_META_ID_MAX + 1); expect_reject(b); // id_len over cap
        b = v2_bytes; patch_u32(b, off_rec1, 10u);                    expect_reject(b); // overlap: rec1 inside rec0
        b = v2_bytes; patch_u32(b, off_rec1, 12u);                    expect_reject(b); // gap: NULL cell 11 uncovered
        b = v2_bytes; patch_u32(b, off_rec1 + 4, 0u);                 expect_reject(b); // n_tokens == 0
        b = v2_bytes; patch_u32(b, off_rec1 + 4, 100u);               expect_reject(b); // out of bounds
        b = v2_bytes; patch_u32(b, off_rec0, 2u);                     expect_reject(b); // record over a text cell
        b = v2_bytes; patch_u32(b, V1_FIXED, (uint32_t) LLAMA_TOKEN_NULL); expect_reject(b); // extra NULL cell uncovered
        b = v2_bytes; b.push_back('\0');                              expect_reject(b); // trailing byte
        // a v2 media file relabelled v1 by a flipped version byte must NOT be accepted as
        // text-only (that would bypass the fp_mmproj check via the v1 backfill and drop
        // every media identity record): v1 rejects any NULL cell.
        b = v2_bytes; patch_u32(b, 4, 1u);                            expect_reject(b);
        // same premise on a genuine v1 file: a NULL token can only be corruption
        b = v1_bytes; patch_u32(b, V1_FIXED, (uint32_t) LLAMA_TOKEN_NULL); expect_reject(b);
        // v1 must also end exactly at the token array: trailing bytes are a
        // relabelled/corrupt file (e.g. a media section the version byte disowned)
        b = v1_bytes; b.push_back('\0');                              expect_reject(b);
        // a v2 file relabelled v1 with its tok_count grown to swallow the media section
        // as extra "tokens" still fails: the section's NULL cells and/or trailing bytes
        b = v2_bytes; patch_u32(b, 4, 1u); patch_u32(b, V1_FIXED - 12, (uint32_t) media_toks.size() + 2); expect_reject(b);
        // swapped records (unordered): swap the two start_idx values
        b = v2_bytes; patch_u32(b, off_rec0, 11u); patch_u32(b, off_rec1, 3u); expect_reject(b);
        // every truncation of v1 and v2 files must be rejected
        for (size_t len = 0; len < v1_bytes.size(); ++len) {
            expect_reject(std::vector<char>(v1_bytes.begin(), v1_bytes.begin() + len));
        }
        for (size_t len = 0; len < v2_bytes.size(); ++len) {
            expect_reject(std::vector<char>(v2_bytes.begin(), v2_bytes.begin() + len));
        }
        // untruncated files still parse after all that
        write_meta_bytes(mut_path, v2_bytes);
        model_fp rfp;
        llama_tokens rtoks;
        std::vector<server_media_record> rmedia;
        GGML_ASSERT(slot_meta_read(mut_path, 0, rfp, rtoks, rmedia));
    }

    // --- v4 structured read rejections (media delta node) -------------------------------------
    const std::vector<char> v4_bytes = read_meta_bytes(v4_path);
    GGML_ASSERT(v4_bytes.size() == v2_meta_size + 16); // v2 layout + node tail
    {
        // a trailing byte past the node tail is a corrupt/relabelled file
        auto b = v4_bytes; b.push_back('\0');                         expect_reject(b);
        // a v4 relabelled v2 (drop the node tail via the version byte): the reader consumes the
        // media tail then sees the 16-byte node tail as trailing bytes -> reject (a delta must not
        // masquerade as a whole media snapshot, whose .bin would be the full cell set)
        b = v4_bytes; patch_u32(b, 4, 2u);                            expect_reject(b);
        // a v4 relabelled v3 (text delta): v3 is a text format that rejects the NULL media cells
        b = v4_bytes; patch_u32(b, 4, 3u);                            expect_reject(b);
        // the media tail's tiling invariants are enforced for v4 exactly as for v2 (n_media == 0)
        b = v4_bytes; patch_u32(b, off_nm, 0u);                       expect_reject(b);
        // every truncation of the v4 file must be rejected (short header, tokens, media or node tail)
        for (size_t len = 0; len < v4_bytes.size(); ++len) {
            expect_reject(std::vector<char>(v4_bytes.begin(), v4_bytes.begin() + len));
        }
        // untruncated v4 still parses, node fields intact
        write_meta_bytes(mut_path, v4_bytes);
        model_fp rfp; llama_tokens rtoks; std::vector<server_media_record> rmedia;
        uint64_t rp = 0; uint32_t rl = 0, rh = 0;
        GGML_ASSERT(slot_meta_read(mut_path, 0, rfp, rtoks, rmedia, &rp, &rl, &rh));
        GGML_ASSERT(rp == v4_parent && rl == v4_lo && rh == v4_hi);
    }

    // --- mutation fuzz over the parser ---------------------------------------------------------
    // Seeded (reproducible) random mutations of valid v1/v2 sidecars plus pure-random buffers.
    // Pass criterion: no crash/UB, and every ACCEPTED input satisfies the format invariants
    // (check_read_invariants holds for ANY accepted input by construction — v1 rejects NULL
    // cells, v2 validates the tiling — so this must stay seed-independent; multiple seeds
    // guard against a single lucky trajectory).
    for (const uint32_t seed : { 0xC3u, 0x5EEDu, 0xBADC0DEu }) {
        std::mt19937 rng(seed);
        size_t n_accepted = 0;
        const size_t n_iter = 2500;
        for (size_t iter = 0; iter < n_iter; ++iter) {
            std::vector<char> b;
            const uint32_t kind = rng() % 6;
            if (kind == 0) {
                // pure-random buffer, random length
                b.resize(rng() % 600);
                for (auto & c : b) {
                    c = (char) (rng() & 0xFF);
                }
            } else {
                // seed corpus spans all three tail shapes: v1 (no tail), v2 (media tail),
                // v4 (media + node tail). check_read_invariants holds for any accepted mutant of
                // each (v1/v3 reject NULL cells, v2/v4 validate the media tiling).
                b = (kind == 1) ? v1_bytes : (kind == 2) ? v2_bytes : v4_bytes;
                const uint32_t n_mut = 1 + rng() % 8;
                for (uint32_t m = 0; m < n_mut; ++m) {
                    if (!b.empty()) {
                        b[rng() % b.size()] = (char) (rng() & 0xFF);
                    }
                }
                if (rng() % 5 == 0 && !b.empty()) {
                    b.resize(rng() % b.size()); // random truncation
                } else if (rng() % 5 == 0) {
                    const uint32_t extra = rng() % 64;
                    for (uint32_t e = 0; e < extra; ++e) {
                        b.push_back((char) (rng() & 0xFF)); // random extension
                    }
                }
            }
            write_meta_bytes(mut_path, b);
            model_fp rfp;
            llama_tokens rtoks;
            std::vector<server_media_record> rmedia;
            if (slot_meta_read(mut_path, 0x77ULL, rfp, rtoks, rmedia)) {
                check_read_invariants(rtoks, rmedia);
                ++n_accepted;
            }
        }
        printf("fuzz(seed=0x%X): %zu inputs, %zu accepted, 0 crashes\n", seed, n_iter, n_accepted);
    }

    fs::remove_all(g_dir);
    printf("test-slot-meta: all tests passed\n");
    return 0;
}
