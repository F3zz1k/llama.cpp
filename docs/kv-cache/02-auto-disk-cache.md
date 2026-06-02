# PR 2 — Design Notes: opt-in automatic disk prompt/KV cache (`--slot-save-auto`)

**Branch:** `auto-disk-kvcache` (commit `bd58751`, stacked on PR 1)
**Files:** `common/arg.cpp` (+24), `common/common.h` (+5), `tools/server/server-common.h` (+4), `tools/server/server-context.cpp` (+1029/-54)

---

## 1. Why this change exists

PR 1 made the *manual* `/slots save|restore` primitive actually reuse KV for recurrent models. But a plain `/v1/chat/completions` client (OpenWebUI, qwen-code, etc.) never calls `/slots` — so it gets no disk reuse at all. A cold server (new process, restart, or a different instance in a pool) re-prefills a large prompt from scratch every time. For deep contexts that's minutes per request (#17107, #18244). Separately, when `--mmproj` is loaded, slot save/restore is blocked even for text-only turns (#21133).

This PR turns the manual primitive into a **transparent, opt-in** automatic cache: with `--slot-save-auto`, ordinary chat-completions clients get cross-request *and* cross-process KV reuse with no `/slots` calls and no client changes — modeled on vLLM Automatic-Prefix-Caching / SGLang RadixAttention, but persisted to disk.

It is **off by default**, and when off it is provably inert.

---

## 2. The five design invariants

The feature is written around five invariants (numbered in the source banner and referenced by every hook):

1. **Off by default / zero overhead when off.** Every hook's first statement is `auto_cache_enabled()`. When false: no dir scan, no index, no hashing, no allocation — bit-identical to without the patch.
2. **Never restore on hash alone.** A snapshot's persisted token-IDs must byte-compare equal to the request prefix before any restore (collision-safe).
3. **Model identity.** Each snapshot carries a fingerprint; a mismatch refuses the restore.
4. **Fallback totality.** Any failure → normal prefill. Never crash, never wrong output.
5. **Hot-path purity.** The multi-GB save runs only on slot release/reassign, never during generation.

These are the lens for everything below.

---

## 3. The index: block-hashing token IDs

`auto_hash_mix` / `auto_block_hashes` compute a **chained** 64-bit hash over token IDs in fixed-size blocks (`--slot-save-block`, default 256). The hash at block boundary *k* commits to the entire prefix `[0, (k+1)*B)` — so a single map `by_boundary: hash → entry` supports **longest-prefix lookup** in O(#blocks): hash the request block-by-block, probe boundaries longest-first, take the deepest hit.

- The chain is **salted with the model fingerprint hash**, so two different models can never collide on identical tokens.
- Only **whole-block** prefixes are index keys; a trailing partial block is not a boundary. This is the vLLM-APC/SGLang reuse granularity — sub-block precision comes from the mandatory byte-verify (invariant 2), not the hash.
- `auto_index_insert_locked` keeps the **longest** snapshot per boundary, so a long snapshot also satisfies shorter shared-prefix requests.

### Why hashing at all, if we byte-verify anyway?
The hash is purely a candidate-*narrowing* accelerator so we don't scan every snapshot per request. Correctness never depends on it: the byte-compare in `auto_restore_into_slot` is authoritative.

---

## 4. The fingerprint (`model_fp`)

Computed once at load (`auto_compute_fingerprint`), compared by exact equality. Fields: model-desc hash (+ size/n_params/n_embd/n_layer), n_vocab, n_ctx_train, rope_type, **cache_type_k/v**, n_ctx, FULL-vs-attention, block size, **rope_freq_scale**, **rope_freq_base + all five YaRN params**, LoRA-set hash, and an **mmproj-loaded** bit.

### Why so many fields?
`llama_state_seq_save_file` serializes the raw KV blob. Loading it into a context with *different KV geometry* silently corrupts:
- a Q4_0-KV blob into an F16 ctx → garbage (hence `cache_type_k/v`);
- a different rope scale/base or YaRN setting → positions are baked into the saved state, so the restored state is positionally wrong (hence all rope/YaRN fields, bit-cast into identity);
- mmproj-aware rope (M-RoPE) / projector wiring can change the text KV layout, so a text-only-server snapshot and an mmproj-server snapshot get **disjoint stores** (the `fp_mmproj_loaded` bit) — conservative until someone proves the layouts are identical.

The "use model default" cases (rope_freq_base==0, YaRN floats <0, yarn_orig_ctx<=0) are normalized to a single `0` sentinel so two runs both relying on defaults match, while any explicit override refuses. **Conservative by design:** a needless miss is safe; a wrong restore is not.

---

## 5. On-disk format & atomicity

Each snapshot is a 3-file unit sharing a base name `auto-<fp>-<chainhash>-<ntokens>.bin`:
- `.bin` — the libllama state (unchanged format);
- `.logits` — PR 1's regenerate sidecar;
- `.meta` — **new**: header + full fingerprint + `tok_count` + `chain_hash` + the raw int32 token IDs.

The `.meta` is what makes the startup scan and pre-restore verify cheap: they read only this tiny file, never the multi-GB `.bin`.

### Atomic publish (and the cross-process subtlety)
All three are written to a **per-writer-unique** temp path — `<fname>.<pid>.<atomic-nonce>.tmp` — then renamed to their final names with **`.meta` renamed LAST**. Two reasons:
- The final name is deterministic (fp + hash + count), so two processes sharing one `--slot-save-path` would otherwise both stream a multi-GB state into the *same* `<fname>.tmp` and interleave → corruption. Per-writer-unique temps mean each owns its own complete temp; the deterministic-name rename is the only shared step, and it's idempotent (identical content).
- `.meta`-last means the startup scan (which keys on `.meta`) never indexes a half-written unit. If the `.meta` rename fails, we unlink the orphan `.bin` and do **not** index it.

---

## 6. Auto-restore (read path)

Hooked in `update_slots`, immediately **after** the in-memory prefix match (`get_common_prefix`) computes `n_past`. If the feature is on, the slot is generative + no-media, and adapters match:
```cpp
const llama_tokens req = input_tokens.get_text_tokens();
if (auto cand = auto_index_lookup(req)) {
    auto_restore_into_slot(slot, *cand, req, (int) n_past);
    n_past = slot.prompt.tokens.get_common_prefix(input_tokens);  // recompute, unconditionally
}
```
`auto_restore_into_slot` byte-verifies the candidate's persisted tokens against the request (invariant 2), checks the fingerprint (invariant 3), applies a **one-block margin** (only pay a multi-GB load if disk beats memory by ≥1 block), then calls `do_slot_restore` and lets PR 1's reuse/regenerate path take over for the suffix.

### Three things worth calling out
- **Recompute `n_past` unconditionally** after any restore *attempt*. `auto_restore_into_slot` clears the slot before loading; if the load then fails, recomputing yields `n_past=0` (clean cold prefill) instead of carrying a stale `n_keep_mem>0` into `keep_first()` on an empty vector — which would assert/abort. On the early-return-before-clear paths the recompute is harmless (tokens untouched). *(This was a real bug caught in review.)*
- **FULL models: whole-snapshot only.** A recurrent state can't be partially rewound, so `auto_restore_into_slot` restores a FULL snapshot only when the request **extends** it (`v == disk_toks.size()`). Attention models can take a mid-snapshot divergence (partial seq_rm), clamped to the last block boundary. *(This is why reuse requires the request to extend the snapshot — the normal multi-turn pattern.)*
- **`get_text_tokens()` not `get_tokens()`.** Under `--mmproj`, `get_tokens()` asserts (`!has_mtmd`). For a no-media prompt, `get_text_tokens()` equals the full token-id prefix and doesn't assert — this is what lets one mmproj instance cache its text-only turns.

---

## 7. Auto-save (write path) + the `cache_idle_slots` gap

`auto_save_slot_if_useful` persists a slot's state when its KV is about to be discarded: deduped (skip if an equal-or-longer snapshot exists), `< 1 block` skipped, written off the generation hot path. It is hooked at **two** sites:
1. `slot_save_and_clear` (the idle-flush path, under `cache_idle_slots`);
2. `get_available_slot` when `update_cache` is set **and** `!cache_idle_slots`.

### Why two hooks?
`cache_idle_slots` requires `--kv-unified` + `--cache-ram`; without those it's force-disabled and the idle-flush never runs. Hooking only #1 meant auto-save silently did nothing with just `--slot-save-auto`. The second hook (gated `!cache_idle_slots` so the two are mutually exclusive → no double-save) makes the feature work with **only** `--slot-save-auto`.

---

## 8. Cross-process visibility without restart

Each process builds its index by a startup scan, and only knows its *own* saves. So a snapshot written by process A was invisible to process B until B restarted — a broken UX in a multi-instance pool. Fixed in `auto_index_refresh_locked`, called from `auto_index_lookup`:
- a **throttled** (≤ 1/s) `stat` of the slot-save dir's mtime; a peer's create/rename bumps that mtime → trigger an **incremental** re-scan (only opens `.meta` files not already in `indexed_files`);
- on a lookup **miss**, **force** a rescan once (bypassing the throttle) before giving up — a miss means a cold prefill is imminent, so the scan is free by comparison, and a peer's <1s-old snapshot is still found on *this* request.
- a process re-baselines `dir_mtime` after its own save so it doesn't needlessly rescan its own writes.

This keeps the single-loop-thread model (no inotify, no background thread) while making fresh caches cross-process-visible within ~1 prefill.

### True LRU (`auto_touch_unit`)
The bounded store evicts by mtime. A hot base snapshot reused by many forked requests is never *rewritten*, so a write-time-only LRU would evict it as "stale." On every successful restore we `touch` the unit's 3 files → the LRU becomes least-recently-**used**, protecting the fan-out case (one prefill restored by N parallel branches).

---

## 9. Multimodal handling (Option A)

`server_tokens::has_media()` (new, in `server-common.h`) reports whether *this* prompt contains media chunks (`!map_idx_to_media.empty()`) — the per-request signal, vs the server-wide `has_mtmd`. The auto hooks gate on `!has_media()`, so an `--mmproj` server caches all its **text-only** turns and skips any turn carrying an image. Image-bearing prefixes are *not* cached because identical token-IDs don't identify image content (image cells are `LLAMA_TOKEN_NULL` placeholders) — caching them would violate invariant 2. The manual `/slots` endpoints keep their existing `check_no_mtmd` guard (smaller blast radius; the auto path controls the full lifecycle). Full image-prefix caching is left as documented future work.

---

## 10. What could have been done better

- **`server-context.cpp` is now very large.** ~1000 lines of auto-cache logic live in one already-huge file. A `server-prompt-cache-disk.{h,cpp}` translation unit would be cleaner; it was kept inline to match where the existing slot/prompt-cache code lives and to avoid a build-system change, but a maintainer may reasonably ask for extraction.
- **The mutex is currently inert.** All index access is single-threaded (server loop), so `auto_idx.mtx` is uncontended today; it only matters if save I/O later moves to a worker thread. It's documented, but a reviewer could argue it's premature (the existing prompt-cache code is lock-free) — defensible either way.
- **All-miss workload cost.** On a stream of cache *misses*, the force-on-miss rescan does an O(#files) `directory_iterator` per request. Bounded by `--slot-save-max-count` (fine at 64), but at a very large cap + high QPS it's worth a smarter "nothing new since last scan" short-circuit or an inotify tier.
- **Write amplification on fan-out.** When N forked slots diverge and each evicts, each writes its own snapshot. Dedup collapses identical continuations, but a tighter LRU under wide fan-out can thrash (write N, evict N−k). Sizing guidance is in the docs; an adaptive policy would be better.
- **`do_slot_restore` out-params.** It returns two values via optional pointer args (`size_t* out_token_count, size_t* out_nread`) for the one caller that needs them; a small result struct would be more idiomatic.
- **`get_text_tokens()` equivalence is assumed.** For a no-media prompt it equals `get_tokens()`; that's true today but couples us to that internal behavior. A explicit `assert(!has_media())` before the call (there is a gate, but not an assert at the accessor) would harden it.
- **Block size is global + baked into the fingerprint.** Changing `--slot-save-block` invalidates the whole on-disk cache for that model (different boundary hashes). That's correct but a sharp edge for operators; documented, but a self-describing per-snapshot block size (already in `.meta` as `fp_block`) could allow mixed-block stores in future.
- **No automated tests**, same as PR 1 — validation was end-to-end (cold-process restore, cross-process visibility, attention regression, LRU). The hashing, fingerprint, and file-format helpers are pure and should get unit tests for upstream.
