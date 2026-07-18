# PR 3 — Design Notes: multimodal snapshots for the auto disk cache

**Branch:** `auto-disk-kvcache-mm` (stacked on `auto-disk-kvcache`)
**Files:** `tools/server/server-common.h/.cpp` (identity layer, `.meta` v2, media-aware hashing), `tools/server/server-context.cpp` (save/restore/scan/manual endpoints, mmproj fingerprint, `/props`), `tools/mtmd/mtmd.h/.cpp` (chunk id accessor + stub chunks), `tests/` (3 unit-test binaries), `tools/server/tests/unit/test_slot_save_auto.py`

---

## 1. Why this change exists

PR 2's auto disk cache skipped any prompt containing media: image cells in `server_tokens` are `LLAMA_TOKEN_NULL` placeholders, so identical token IDs do not identify image content, and caching them blind would violate invariant 2 ("never restore on hash alone"). An `--mmproj` server therefore cached only its text-only turns, and the manual `/slots` endpoints 501'd server-wide.

The keystone observation that unlocks full multimodal caching cheaply: **the image embeddings are already inside the state file.** `llama_state_seq_save_file` serializes the KV cells that the vision encoder produced, and since upstream #20132/#20273 the M-RoPE 2D position data round-trips with them. And at restore time the *request itself* carries live chunks with pixels. So the disk side needs neither pixels nor embeddings beyond the state file — only **identity**: enough metadata per media chunk to verify, before reuse, that the request's chunk is byte-for-byte the same media the snapshot was encoded from. That is ~100 bytes per chunk, and it fits in a versioned extension of the existing `.meta` sidecar. No new file, no change to the torn-write matrix, no untrusted pixel deserializer.

Credit where due: the bitmap-id-as-identity idea (FNV-1a over the raw uploaded bytes, already computed by `mtmd_helper_bitmap_init_from_buf`) and chunk-id comparison at restore come from Lissanro's prefill fork; the metadata-only persistence design here is a from-scratch replacement for its persisted-chunk sidecar.

---

## 2. The identity model

- **Media identity = the mtmd bitmap id**: FNV-1a over the raw uploaded bytes, computed before the image/audio type branch — so both types get one. Byte-identical re-upload is required for reuse (normal multi-turn clients resend attachments unchanged). A chunk with an **empty id** (e.g. a placeholder bitmap) can never be re-verified: saves refuse it, lookups skip it, restores hard-mismatch it.
- **Per-chunk record** (`server_media_record`, extracted by `server_tokens::extract_media_records()`): `start_idx` (cell index), `n_tokens`, `n_pos` (cell count ≠ position count under M-RoPE), `nx`/`ny` (token-grid geometry; for audio `n_tokens`×1), `is_audio`, and the id string. `n_tokens` is stored explicitly because mtmd derives it from pixel data we do not store.
- **Second factor**: `n_tokens`/`n_pos`/`nx`/`ny`/type are compared alongside the id at restore. A 64-bit content hash has a residual collision probability; it is documented and accepted — the same trust model upstream's in-memory prompt cache places in token IDs — and the shape factors mean a colliding file must also match geometry exactly. Text cells are still byte-compared, always.
- **Audio is first-class** by construction: identity is computed before the type branch, records carry `is_audio`, and the hash folds the type in so an audio chunk can never impersonate an image chunk with the same id.

## 3. `.meta` v2 and the tiling invariant

Version 1 (text-only) is **byte-frozen**: same layout, same bytes, written for every snapshot containing no media — old and new binaries interoperate on text units indefinitely (the frozen golden fixture under `tools/server/tests/fixtures/golden-v1/` locks this in CI). Version 2 = the full v1 layout, then `fp_mmproj` (u64), `n_media` (u32, cap 4096), and the records (id cap 256 bytes). The token array in v2 is **cell-aligned**: `LLAMA_TOKEN_NULL` at media positions, so `tok_count` equals the KV cell count the state file persists.

`slot_meta_read` (fuzzed in `tests/test-slot-meta.cpp`) enforces, rejecting the whole file on any violation:

- magic/version, all caps, exact-EOF after the last field;
- records non-empty, ordered by `start_idx`, disjoint, in-bounds (64-bit arithmetic, no overflow);
- **the tiling invariant**: every record cell is NULL *and* every NULL cell is covered by exactly one record;
- **v1 must be text-only**: any NULL cell, or trailing bytes, in a version-1 file rejects it — otherwise flipping a v2 file's version byte would shed its media records and produce an image-blind restorable unit;
- version-aware fingerprint compare: v1 predates `fp_mmproj`, and v1 ⇒ text-only ⇒ projector-independent KV, so the reader backfills the live value (pre-existing prod snapshots keep restoring); v2 compares it for real.

Parse-rejected sidecars are **remembered by filename** (`rejected_files`) — sound because units are immutable after their atomic rename — so rescans never re-open them; any future version bump costs one read total, not one per scan. A *missing* sidecar is transient (a peer mid-publish renames `.meta` last) and is retried, never cached as rejected.

## 4. Media-aware block chain hashing

`auto_block_hashes` keeps the salt at `fp_model` for **all** prompts. A text cell contributes its token id exactly as before — so pure-text prompts hash bit-identically to PR 2 (same filenames, same index keys) and the *text prefix boundaries of a media prompt equal a text-only prompt's*, which buys the dominant reuse pattern both ways: a media request reuses a plain text snapshot of its prefix, and a text request reuses the pre-image boundaries of a media snapshot. A NULL cell at index `i` inside a chunk starting at `s` contributes:

```
splitmix64( fnv64(id) ^ (i - s) ^ mix(n_tokens, n_pos, is_audio) ^ fp_mmproj )
```

Each factor earns its place: the per-slice offset `(i-s)` disambiguates llava-uhd slices sharing one bitmap id; the shape/type mix stops cross-type impersonation; `fp_mmproj` makes a projector swap change media boundary hashes without touching text boundaries. Different images in identical text now produce different chain hashes ⇒ different deterministic filenames ⇒ concurrent writers of different-image variants can never collide on one unit.

**Boundary discipline:** a single shared predicate `boundary_is_chunk_safe` (block-aligned AND not strictly inside a chunk) is used by all sites that emit or clamp reuse lengths — save-time insert, scan rehash, lookup, restore clamp — so they cannot drift. Only chunk-safe boundaries become index keys.

## 5. Save path

The `has_media()` early-return became a branch. Text prompts run **textually unchanged** code (v1 sidecar, `get_text_tokens()`). Media prompts use the NULL-preserving `server_tokens::get_cell_tokens()`, refuse empty-id chunks, and assert chunk-completeness (guaranteed by the `process_mtmd_chunk` contract — no destructive clamp). One media-specific gate: for FULL-seq-rm models, a snapshot containing generated tokens past the prompt is refused — FULL snapshots only restore on whole-snapshot extend-matches, and no re-request extends a prompt+generation artifact (chat templates re-render history and diverge at or before the prior prompt end), so the write would be a guaranteed-dead multi-GB unit. Prompt-prefix states (mid-prefill, shutdown flush) still save; they are exactly the restorable class.

Two generic robustness measures were added for all prompts (justified deviations from "text path untouched", both fail toward *skipping* an opportunistic write): a **capacity pre-flight** (`llama_state_seq_get_size` + 10% slack vs `std::filesystem::space` — an ENOSPC mid-write can flip btrfs read-only, far worse than a skipped save), and a **deadline-boxed shutdown flush** (60 s, checked between slots, in-flight writes never aborted; size `TimeoutStopSec` accordingly). Eviction is untouched: one LRU policy for all units, no media carve-out.

## 6. Restore path — verification order (cheap → expensive)

All identity work happens on the few-KB sidecar **before the multi-GB state file is ever opened**:

1. `.meta` parse (magic/version/caps/tiling as above).
2. Fingerprint equality, version-aware.
3. Byte-LCP of persisted cells vs request cells (NULL==NULL passes here — content comes next).
4. **Per-record verification**: iterate the *disk* records by `start_idx` within the LCP; the request must have a record at exactly that index with equal id, `n_tokens`, `n_pos`, `nx`, `ny`, type. Iterating records (never "the next chunk after") verifies each image of an adjacent pair independently. A mismatch truncates the LCP to that record's start — same text + a different image reuses exactly the pre-image prefix.
5. Clamp: FULL models require the whole snapshot to be a verified prefix of the request (no partial rewind exists); PART models clamp to a block boundary that is chunk-safe. Then the one-block margin gate as before.
6. Only now `do_slot_restore`. For media units the prompt is then **rebuilt from the request**: `req.clone()` + `keep_first(v)` — the request's live chunks (with pixels) back the NULL cells; the persisted embeddings are already in the loaded KV. A failed rebuild trips a safety clear (`seq_rm(-1,-1)` + full prompt clear → cold prefill), kept as a tripwire even though it is unreachable by construction.
7. SWA (PART, `n_swa > 0`): trim the restored seq to the verified prefix and reconstruct a checkpoint there; if the trim empties the SWA window (divergence deeper than the window), drop the restore and prefill cold rather than trip the downstream `pos_min == -1` abort.

Two latent-upstream-bug fixes ride along: the lookup gate that tested `slot.prompt.tokens.has_media()` — the slot's **stale previous** prompt, not the request — is deleted; and the `[TAG_PROMPT_LOGITS]` `n_past--` re-decode step now clamps to the enclosing chunk's start when the decrement lands inside a media chunk (otherwise `keep_first` throws — reachable whenever a fully-cached prompt ends with an image, which whole-snapshot disk restores hit routinely).

## 7. The mmproj fingerprint

`fp_mmproj`: an FNV/splitmix chain over the mmproj GGUF *header* — every KV pair (key + typed value bytes, lengths folded first so fields cannot alias), every tensor's name/type/shape, plus the file size. Header-only (`no_alloc`), ~ms even for multi-GB files; catches projector swap, requantization, and dimension changes that the v1 `fp_mmproj_loaded` 0/1 deployment-shape bit cannot. Computed whenever an mmproj loads (not gated on the cache), logged at startup, exposed in `/props` so operators can tell which projector a snapshot store belongs to. It participates in v2 fingerprints and per-cell hash contributions only — text snapshots stay projector-independent.

## 8. Manual `/slots` endpoints

The server-wide `check_no_mtmd` 501 became a **per-slot** gate. A text slot on an `--mmproj` server saves/restores exactly as on a text-only server (no `.meta` emitted — manual units carry user-chosen filenames and are never indexed by the auto cache; restore authority is the byte-compared token array). A media slot's save writes the v2 sidecar next to the state file; a failed sidecar write withdraws the whole unit (a media state file without identity is unrestorable — never publish a half-loadable unit).

Manual media **restore** has no request to rebuild from, so it rehydrates the sidecar's records into **stub chunks** (`mtmd_input_chunk_init_stub`: id + geometry, placeholder data). Stubs verify and count positions exactly like live chunks; the embeddings are already in the loaded state, so pixels are never needed — and any path that would *re-encode* a stub (e.g. a follow-up that diverges before the image) refuses and clears instead of emitting garbage. Rehydration failures (missing/invalid sidecar, fingerprint mismatch, sidecar/state divergence) drop the loaded state and return an explicit error.

## 9. Tests

- `tests/test-slot-meta.cpp` — round-trip + adversarial `.meta` parsing, including a seeded fuzz mode (mutated/truncated/bit-flipped inputs; the reader must reject or parse, never crash, and accepted inputs must satisfy the tiling invariant).
- `tests/test-auto-hash.cpp` — hash determinism and identity factors: same text ± image ⇒ identical pre-image boundary hashes + different full hashes; id/shape/type/projector/slice-offset each move the hash; text chains bit-identical to an independent pre-media reference.
- `tests/test-server-tokens.cpp` — cell accessors, record extraction, chunk-safe boundary predicate.
- `tools/server/tests/unit/test_slot_save_auto.py` — end-to-end on tinygemma3/tinyllama2: golden byte-identity for text units, v1 restorability, vision full reuse across restarts, different-image prefix truncation, adjacent-image independence, corrupt/torn/unknown-version fallback, cross-process sharing, `--parallel 2`, and the manual `/slots` suite.

tinygemma3 uses normal positions, so **M-RoPE coverage is on-rig only** (validated end-to-end on Qwen3.6 + mmproj); a tiny qwen2-vl GGUF is the noted CI follow-up. Audio has no runtime fixture either — the audio path is structurally verified (line-by-line walkthrough of save gate → record extraction → hash → verify → rebuild) and `is_audio` permutations are fuzzed, but no audio-input model was runtime-tested.

## 10. What could have been done better

- **Stub-chunk tripwires are code-reviewed-only.** The refuse-and-clear guard for re-encoding a stub is exercised end-to-end by pytest (stub-never-encoded), but the innermost `process_mtmd_chunk` placeholder tripwire is unreachable by any request-driven path and is verified by review, not by a test.
- **`nx`/`ny` for audio are a convention** (`n_tokens`×1) rather than a meaningful geometry; harmless, but a discriminated record layout would be more honest than reusing image fields.
- **The FULL mid-generation save gate encodes template behaviour.** "No re-request extends a prompt+generation artifact" is true for every chat template we know (they re-render history), but it is an empirical claim about clients, not a structural guarantee; the cost of being wrong is only a skipped save.
- **64-bit media identity** is a deliberate, documented residual (see §2); a future format could widen the id to 128 bits for effectively-zero collision odds at negligible cost.
