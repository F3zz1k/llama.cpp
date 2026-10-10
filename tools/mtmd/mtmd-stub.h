// fork-only additions to the public mtmd API (media stub chunks, so the server's disk cache can rebuild the
// tracking state of an already-computed image or audio segment), kept out of mtmd.h. Included at the end of
// mtmd.h; not meant to be included on its own.

#ifndef MTMD_STUB_H
#define MTMD_STUB_H

#ifndef MTMD_H
#error "include mtmd.h instead"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// true if the chunk carries no encodable media payload (its image/audio data is a
// placeholder); such a chunk can be counted and tracked, but encoding it fails
// (see mtmd_encode_chunk)
MTMD_API bool mtmd_input_chunk_is_placeholder(const mtmd_input_chunk * chunk);

// construct a media chunk carrying ONLY identity metadata (id + token/position geometry)
// with placeholder pixel/sample data: enough to rebuild KV-cache tracking state for an
// already-computed image/audio segment without the original media bytes
// the id must be non-empty; for images, nx/ny are the token-grid dimensions (as returned
// by mtmd_image_tokens_get_grid); nx/ny are ignored for audio
// returns nullptr if the requested geometry cannot be reproduced faithfully for the
// current model (the caller must treat that as a hard failure)
// encoding a stub chunk fails; use mtmd_input_chunk_is_placeholder to detect one
// free with mtmd_input_chunk_free
MTMD_API mtmd_input_chunk * mtmd_input_chunk_init_stub(mtmd_context * ctx,
                                                       bool          is_audio,
                                                       const char *  id,
                                                       uint32_t      n_tokens,
                                                       llama_pos     n_pos,
                                                       uint32_t      nx,
                                                       uint32_t      ny);

// raw token-grid dimensions (nx = tokens in x, ny = tokens in y; ny == 1 and nx == n_tokens
// for single-row layouts). the non-deprecated replacement for mtmd_image_tokens_get_nx/ny
// when the grid shape itself is needed (e.g. to persist or rebuild image geometry) rather
// than a decoder position from mtmd_image_tokens_get_decoder_pos()
MTMD_API void mtmd_image_tokens_get_grid(const mtmd_image_tokens * image_tokens, uint32_t * nx, uint32_t * ny);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MTMD_STUB_H
