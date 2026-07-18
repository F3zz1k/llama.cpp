// tests for the server_tokens media-safe accessors used by the auto disk cache:
// get_cell_tokens(), extract_media_records(), boundary_is_chunk_safe()

#include "server-common.h"
#include "mtmd.h"

#include <cstdio>

int main(void) {
    // mtmd_test_create_input_chunks: one text chunk {1, 2, 3, 4, 5} followed by one
    // 4x4 image chunk with id "image_1" (no model required)
    mtmd::input_chunks chunks(mtmd_test_create_input_chunks());
    GGML_ASSERT(chunks.ptr);
    GGML_ASSERT(chunks.size() == 2);

    const size_t n_text = 5;
    const size_t n_img  = mtmd_input_chunk_get_n_tokens(chunks[1]);
    const size_t n_pos  = (size_t) mtmd_input_chunk_get_n_pos(chunks[1]);
    GGML_ASSERT(n_img > 0);

    server_tokens toks(chunks, /* has_mtmd */ true);
    GGML_ASSERT(toks.size() == n_text + n_img);

    // get_cell_tokens: cell-aligned, media cells are LLAMA_TOKEN_NULL
    {
        const llama_tokens & cells = toks.get_cell_tokens();
        GGML_ASSERT(cells.size() == n_text + n_img);
        for (size_t i = 0; i < n_text; ++i) {
            GGML_ASSERT(cells[i] == (llama_token) (i + 1));
        }
        for (size_t i = n_text; i < cells.size(); ++i) {
            GGML_ASSERT(cells[i] == LLAMA_TOKEN_NULL);
        }
    }

    // get_cell_tokens on a text-only prompt: no media, no NULLs, no mtmd assert
    {
        const llama_tokens text = { 10, 20, 30 };
        server_tokens text_toks(text, /* has_mtmd */ false);
        GGML_ASSERT(text_toks.get_cell_tokens() == text);
        GGML_ASSERT(text_toks.extract_media_records().empty());
        for (size_t i = 0; i <= text.size(); ++i) {
            GGML_ASSERT(text_toks.boundary_is_chunk_safe(i));
        }
    }

    // extract_media_records: one record describing the image chunk
    const std::vector<server_media_record> records = toks.extract_media_records();
    {
        GGML_ASSERT(records.size() == 1);
        const server_media_record & rec = records[0];
        GGML_ASSERT(rec.start_idx == n_text);
        GGML_ASSERT(rec.n_tokens  == n_img);
        GGML_ASSERT(rec.n_pos     == n_pos);
        GGML_ASSERT(rec.nx        == 4);
        GGML_ASSERT(rec.ny        == 4);
        GGML_ASSERT(rec.is_audio  == 0);
        GGML_ASSERT(rec.id        == "image_1");
    }

    // boundary_is_chunk_safe: safe at text tokens, the chunk start and one-past-the-end;
    // unsafe strictly inside the chunk
    {
        for (size_t i = 0; i <= n_text; ++i) {
            GGML_ASSERT(toks.boundary_is_chunk_safe(i)); // text tokens + chunk start
        }
        for (size_t i = n_text + 1; i < n_text + n_img; ++i) {
            GGML_ASSERT(!toks.boundary_is_chunk_safe(i)); // strictly inside the image
        }
        GGML_ASSERT(toks.boundary_is_chunk_safe(n_text + n_img)); // one-past-the-end
    }

    // the record-based overload (scan-time shape) must agree with the member at every idx
    {
        const llama_tokens & cells = toks.get_cell_tokens();
        for (size_t i = 0; i <= cells.size(); ++i) {
            GGML_ASSERT(boundary_is_chunk_safe(cells, records, i) == toks.boundary_is_chunk_safe(i));
        }
    }

    printf("test-server-tokens: all tests passed\n");
    return 0;
}
