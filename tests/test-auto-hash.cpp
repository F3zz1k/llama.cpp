// tests for the auto disk cache's block chain hashing (auto_block_hashes):
// text-only bit-identity with the pre-media algorithm (index keys and filenames are
// derived from these hashes, so the text chain is byte-frozen), media determinism,
// text-prefix boundary sharing between text and media prompts, chunk-safe boundary
// emission, and the identity factors (id, chunk shape/type, fp_mmproj, salt).

#include "server-common.h"
#include "mtmd.h"

#include <cstdint>
#include <cstdio>
#include <vector>

// independent reimplementation of the PRE-MEDIA chain (the shipped v1 algorithm) —
// the text-only path of auto_block_hashes must match it bit for bit forever.
static std::vector<uint64_t> ref_text_block_hashes(const llama_tokens & toks, int B, uint64_t salt) {
    std::vector<uint64_t> out;
    if (B <= 0) {
        return out;
    }
    uint64_t h = 0xcbf29ce484222325ULL ^ salt;
    for (size_t i = 0; i < toks.size(); ++i) {
        h ^= (uint64_t) (uint32_t) toks[i];
        h *= 0x100000001b3ULL;
        h ^= h >> 29; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 32;
        if ((i + 1) % (size_t) B == 0) {
            out.push_back(h);
        }
    }
    return out;
}

static server_media_record make_rec(uint32_t start_idx, uint32_t n_tokens, const std::string & id,
                                    uint32_t n_pos = 0, uint32_t is_audio = 0) {
    server_media_record rec;
    rec.start_idx = start_idx;
    rec.n_tokens  = n_tokens;
    rec.n_pos     = n_pos ? n_pos : n_tokens;
    rec.nx        = n_tokens;
    rec.ny        = 1;
    rec.is_audio  = is_audio;
    rec.id        = id;
    return rec;
}

// cells: n_text real tokens (1..n_text), then for each record its NULL run
static llama_tokens make_cells(size_t n_text, const std::vector<server_media_record> & recs,
                               size_t n_tail_text = 0) {
    llama_tokens cells;
    for (size_t i = 0; i < n_text; ++i) {
        cells.push_back((llama_token) (i + 1));
    }
    for (const auto & r : recs) {
        GGML_ASSERT((size_t) r.start_idx == cells.size()); // records must tile in order
        cells.insert(cells.end(), r.n_tokens, LLAMA_TOKEN_NULL);
    }
    for (size_t i = 0; i < n_tail_text; ++i) {
        cells.push_back((llama_token) (1000 + i));
    }
    return cells;
}

int main(void) {
    const uint64_t salt = 0x1122334455667788ULL;
    const uint64_t fpmm = 0xa5a5a5a55a5a5a5aULL;

    // text-only: bit-identical to the pre-media algorithm for every geometry,
    // and independent of fp_mmproj (it must only ever reach media contributions)
    {
        llama_tokens toks;
        for (int n = 0; n <= 40; ++n) {
            for (int B : { -1, 0, 1, 3, 8, 64 }) {
                const auto ref = ref_text_block_hashes(toks, B, salt);
                GGML_ASSERT(auto_block_hashes(toks, {}, B, salt, 0)    == ref);
                GGML_ASSERT(auto_block_hashes(toks, {}, B, salt, fpmm) == ref);
            }
            toks.push_back((llama_token) (n * 977 + 3));
        }
        // a different salt changes every boundary hash
        const auto a = auto_block_hashes(toks, {}, 8, salt,     0);
        const auto b = auto_block_hashes(toks, {}, 8, salt ^ 1, 0);
        GGML_ASSERT(a.size() == b.size());
        for (size_t k = 0; k < a.size(); ++k) {
            GGML_ASSERT(a[k] != b[k]);
        }
    }

    // same text +- image: identical pre-image boundary hashes, different afterwards.
    // 8 text tokens, then a 6-cell image, then 2 tail text tokens; B=4.
    const int B = 4;
    const std::vector<server_media_record> img = { make_rec(8, 6, "image_1", /*n_pos*/ 3) };
    const llama_tokens media_cells = make_cells(8, img, 2); // 16 cells
    const auto media_bhs = auto_block_hashes(media_cells, img, B, salt, fpmm);
    {
        llama_tokens text_cells = make_cells(16, {}); // same first 8 tokens, then text
        const auto text_bhs = auto_block_hashes(text_cells, {}, B, salt, fpmm);
        GGML_ASSERT(text_bhs.size() == 4); // 4, 8, 12, 16
        // media boundaries: 4 (text), 8 (chunk start), 12 (inside chunk -> skipped), 16 (end)
        GGML_ASSERT(media_bhs.size() == 3);
        GGML_ASSERT(media_bhs[0] == text_bhs[0]); // shared text prefix [0,4)
        GGML_ASSERT(media_bhs[1] == text_bhs[1]); // shared text prefix [0,8)
        GGML_ASSERT(media_bhs[2] != text_bhs[3]); // full prompts differ
        // determinism: same inputs, same chain
        GGML_ASSERT(auto_block_hashes(media_cells, img, B, salt, fpmm) == media_bhs);
    }

    // a different image id keeps the pre-image boundaries and changes the rest
    {
        const std::vector<server_media_record> other = { make_rec(8, 6, "image_2", 3) };
        const auto bhs = auto_block_hashes(media_cells, other, B, salt, fpmm);
        GGML_ASSERT(bhs.size() == media_bhs.size());
        GGML_ASSERT(bhs[0] == media_bhs[0] && bhs[1] == media_bhs[1]);
        GGML_ASSERT(bhs[2] != media_bhs[2]);
    }

    // an audio chunk can never impersonate an image chunk with the same id/shape
    {
        const std::vector<server_media_record> audio = { make_rec(8, 6, "image_1", 3, /*is_audio*/ 1) };
        const auto bhs = auto_block_hashes(media_cells, audio, B, salt, fpmm);
        GGML_ASSERT(bhs[2] != media_bhs[2]);
    }

    // n_pos is an identity factor (M-RoPE layout changes must change the hash)
    {
        const std::vector<server_media_record> other = { make_rec(8, 6, "image_1", /*n_pos*/ 6) };
        const auto bhs = auto_block_hashes(media_cells, other, B, salt, fpmm);
        GGML_ASSERT(bhs[2] != media_bhs[2]);
    }

    // a projector swap changes media boundaries WITHOUT touching text boundaries
    {
        const auto bhs = auto_block_hashes(media_cells, img, B, salt, fpmm ^ 42);
        GGML_ASSERT(bhs[0] == media_bhs[0] && bhs[1] == media_bhs[1]);
        GGML_ASSERT(bhs[2] != media_bhs[2]);
    }

    // two slices sharing one bitmap id are not the same as one double-length chunk
    // (per-cell offset restarts per record and n_tokens folds into each contribution)
    {
        const std::vector<server_media_record> split =
            { make_rec(8, 3, "image_1", 3), make_rec(11, 3, "image_1", 3) };
        const auto bhs = auto_block_hashes(media_cells, split, B, salt, fpmm);
        GGML_ASSERT(bhs.size() == media_bhs.size());
        GGML_ASSERT(bhs[2] != media_bhs[2]);
    }

    // chunk-safe emission on a chunk-start block boundary: 4 text + 6-cell image + 2 text,
    // B=4 -> boundaries 4 (chunk start, safe), 8 (inside, skipped), 12 (end, safe)
    {
        const std::vector<server_media_record> rec = { make_rec(4, 6, "image_1", 3) };
        const llama_tokens cells = make_cells(4, rec, 2);
        const auto bhs = auto_block_hashes(cells, rec, B, salt, fpmm);
        GGML_ASSERT(bhs.size() == 2);
        const auto text_bhs = auto_block_hashes(llama_tokens(cells.begin(), cells.begin() + 4), {}, B, salt, fpmm);
        GGML_ASSERT(bhs[0] == text_bhs[0]); // the [0,4) text boundary is shared
    }

    // real server_tokens shape: cells + records from the mtmd test chunks must hash
    // without tripping the record-coverage assert, and for every block size the
    // emitted boundary count must equal the member predicate's chunk-safe count
    {
        mtmd::input_chunks chunks(mtmd_test_create_input_chunks());
        GGML_ASSERT(chunks.ptr);
        server_tokens toks(chunks, /* has_mtmd */ true);
        const auto recs = toks.extract_media_records();
        GGML_ASSERT(recs.size() == 1);
        const size_t n = toks.size();
        for (size_t Bi = 1; Bi <= n; ++Bi) {
            const auto bhs = auto_block_hashes(toks.get_cell_tokens(), recs, (int) Bi, salt, fpmm);
            size_t n_safe = 0;
            for (size_t idx = Bi; idx <= n; idx += Bi) {
                n_safe += toks.boundary_is_chunk_safe(idx) ? 1 : 0;
            }
            GGML_ASSERT(bhs.size() == n_safe);
            if (Bi == 1) {
                // B=1 visits every cell: the chunk's interior boundaries must have been skipped
                GGML_ASSERT(n_safe == n - (recs[0].n_tokens - 1));
            }
        }
    }

    printf("test-auto-hash: all tests passed\n");
    return 0;
}
