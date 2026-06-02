# PR 1 — Design Notes: reuse disk-restored KV for recurrent/hybrid models

**Branch:** `kv-restore-reuse` (commits `ea1113e`, `ee749c8` on base `aa46bda`)
**Files:** `common/arg.cpp` (+20), `common/common.h` (+4), `common/sampling.cpp` (+84), `common/sampling.h` (+15), `tools/server/server-context.cpp` (+546/-3)

---

## 1. Why this change exists

llama-server can save a slot's KV state to disk (`POST /slots/{id}?action=save`) and load it back (`action=restore`). For ordinary **attention** models this is enough to resume a conversation. For **recurrent / hybrid** models (Mamba, Falcon-H, Jamba, Qwen3-Next, Qwen3.6 GDN, …) it was effectively broken in two ways:

1. **No reuse after restore.** After a restore, the next request re-processed the *entire* prompt instead of continuing from the restored state. The prompt-cache reuse path keys off a *context checkpoint*; a disk-loaded slot never had one reconstructed, so the matcher always fell back to a full re-prefill. (Reported in #21831; the fix approach is what discussion #19264 proposed.)

2. **Regenerate crashes the server.** Resending the *exact* saved prompt with **no new suffix** (a "regenerate") hit `GGML_ABORT`. A recurrent state cannot be partially rewound, so there is no token to (re)decode for logits; the code decremented `n_past` (`[TAG_PROMPT_LOGITS]`) and re-decoded into the already-occupied sequence, which the memory backend rejects.

There was also no bound on the slot-save directory — it grew without limit.

The fix has three pieces, gated so attention models and all existing behavior are untouched.

---

## 2. Piece 1 — reuse after restore (commit `ea1113e`)

### What changed
In the `SLOT_RESTORE` handler (`server-context.cpp`), after `llama_state_seq_load_file` populates `slot->prompt.tokens`, we now:
- set a new one-shot flag `slot->just_restored = true`;
- for FULL (recurrent/hybrid/SWA) models, **reconstruct a context checkpoint at the restored tail** via `create_checkpoint(*slot, 0, ckpt_pos_min, ckpt_pos_max)`.

And the checkpoint-search predicate in the prompt-processing loop was relaxed by one clause:
```cpp
// before:
return cur.pos_min < pos_min_thold || cur.pos_min == 0;
// after:
return cur.pos_min == 0 || cur.pos_min < pos_min_thold
    || (slot_was_restored && has_new_suffix && cur.pos_min == pos_min_thold);
```
`slot_was_restored` reads-and-clears `just_restored` (one-shot); `has_new_suffix` is `task->n_tokens() > n_past`.

### How it works
A recurrent state only physically exists at its tail position `L`. The normal reuse predicate requires a checkpoint strictly *below* the threshold (`pos_min < pos_min_thold`), which guarantees ≥1 token gets reprocessed to produce logits. But a restored recurrent checkpoint sits exactly *at* `L` (= `pos_min_thold + 1` for a full-length match). The new clause says: if this is the first request after a restore **and** the request carries genuinely new suffix tokens (which will supply the required logits), accept the tail checkpoint at `pos_min == pos_min_thold`. The suffix is then the only thing prefilled.

### Why this design
- **One-shot gating (`just_restored`)** means the relaxed clause can *only* fire on the first request after a disk restore. Every in-process request takes the unchanged path → in-process serving is byte-identical to upstream (zero regression).
- **`has_new_suffix` requirement** preserves the "≥1 token reprocessed for logits" invariant the strict `<` was protecting — the suffix provides those logits. (The no-suffix case is Piece 2.)
- Reconstructing the checkpoint *at restore time* (rather than lazily) keeps all the reuse logic in the existing matcher; we just give it something to match.

---

## 3. Piece 2 — regenerate from saved logits + the crash fix (commit `ee749c8`)

This is the bulk of the PR. Three sub-parts: a logits sidecar (A), the restore-continue fast path (B), and the new sampler helper.

### 3a. The logits sidecar (`slot_logits_write` / `slot_logits_read`)
On `SLOT_SAVE` for a FULL model, we additionally persist the **last decoded token's full-vocab logits** to `<state>.logits`:
- **Format:** little-endian header `magic / version / n_vocab / n_tokens`, then `n_vocab` f32 logits. Serialized byte-by-byte (not `fwrite` of a struct) for portability; written to `<sidecar>.tmp` and atomically `rename`d so a partial write can never leave a corrupt sidecar.
- **It does NOT touch libllama's state-file format.** That was a deliberate choice — the sidecar is an independent, optional file, so the change is additive and the state format stays frozen.
- **Binding to the exact state.** `slot_logits_read` only returns the logits if the sidecar's `n_vocab` matches the live model **and** its recorded `n_tokens` matches the restored state's token count. This token-count check is the authoritative guard preventing a stale distribution from ever being reused against a mismatched state.

The logits are captured at sample time into `slot.logits_last` (sized `n_vocab`), stamped with `slot.logits_last_n_tokens` = the prompt length that produced them. The capture is gated `ctx_tgt_seq_rm_type == FULL && !slot_save_path.empty()` — attention models and servers without slot-save pay **nothing**.

#### Why capture at sample time, and why the stamp?
`llama_get_logits_ith(ctx)` is overwritten by the next slot's decode, so under `--parallel>1` a lazy read at save time would return the wrong slot's logits. Capturing per-slot from its own `tok_idx` immediately after `common_sampler_sample` is the only correct point. The stamp (`logits_last_n_tokens`) closes every stale-logits hole: the sidecar is written **only** when `stamp == token_count`, so a distribution left over from a prior task, a restore with no intervening decode, or a skipped spec-decode step can never be serialized against a state it doesn't belong to. `launch_slot_with_task` and `SLOT_RESTORE` both invalidate the stamp as belt-and-suspenders.

### 3b. `common_sampler_sample_from_logits` (`common/sampling.cpp/.h`)
A new **public** sampler entry point that samples from a caller-provided raw logits buffer instead of from a `llama_context`. It mirrors `common_sampler_sample`'s CPU full-logits path *exactly*: build candidates from the raw logits (`common_sampler_set_logits_raw`), then apply reasoning-budget → [grammar] → chain, including grammar rejection-resampling.

#### Why a new function instead of reusing `common_sampler_sample`?
`common_sampler_sample` reads logits from `ctx` via the private `set_logits(ctx, idx)`. We have logits from disk, not a context. Rather than fake a context, the helper swaps only the candidate-building step and keeps the apply sequence identical, so the selected token is bit-for-bit what the live sampler would have produced. It deliberately omits `llama_synchronize` (no pending context op) and the backend-sampler short-circuit (a replayed distribution carries no backend-sampled token, and backend sampling is incompatible with grammar/reasoning-budget anyway). The header documents the key caveat: the result still depends on the sampler's **accumulated state** (penalties/grammar/RNG), so the caller must have advanced `gsmpl` to the right step — exactly as `common_sampler_sample` requires.

### 3c. The restore-continue fast path (Feature B, in `update_slots`)
When a just-restored FULL slot receives the **exact** restored tokens (no suffix), gated by:
```cpp
slot.just_restored && ctx_tgt_seq_rm_type == FULL && slot.task->need_sampling()
  && slot.alora_invocation_start <= 0
  && n_past == slot.task->n_tokens() && n_past == slot.prompt.n_tokens()
```
two outcomes:
- **Saved logits present:** emit the first token directly from them — no decode. Set `n_prompt_tokens_processed = 0` (the observable `prompt_n=0` reuse signal), prime the sampler over the restored prompt, transition to `GENERATING`, sample via `common_sampler_sample_from_logits`, account/stream/`process_token` exactly as the normal first-token path, then `continue` (skip all prompt-batch building). Subsequent tokens decode normally.
- **No saved logits (fallback):** clear the sequence (`llama_memory_seq_rm(... -1, -1)`), clear tokens/checkpoints, `n_past = 0`, fall through to a normal cold reprefill. **This is the crash fix** — clearing first guarantees the reprefill writes into an empty sequence instead of re-decoding into the occupied restored state.

#### Why place it *before* the `n_past > 0` guard?
That guard + `[TAG_PROMPT_LOGITS]` is exactly the path that crashed (decrement `n_past`, re-decode into the occupied seq). Intercepting earlier is what lets us either emit-from-logits or do the safe clear. The gate conditions make it unreachable for non-recurrent models, with-suffix requests (those use Piece 1), non-generative slots, multimodal, and `cache_prompt=false`.

### 3d. The one non-additive change
`server_slot::get_timings()` — 2 lines — now guards against `n_prompt_tokens_processed == 0` (the `prompt_n=0` fast path) so the per-token-ms / per-second divisions don't emit `inf`/`NaN` that would serialize as invalid JSON. Everything else in the PR is purely additive.

---

## 4. Piece 3 — bounded slot-save store (`slot_save_enforce_limits`)

New flags `--slot-save-max-count` (default **64**) and `--slot-save-max-mb` (default **32768** = 32 GiB); `0` = unlimited, negatives rejected at parse.

### How it works
After a save, enumerate the slot-save dir, pair each state file with its `.logits` sidecar into one `slot_save_unit` (evicted together), sort by mtime, and evict oldest until under both caps. The just-written unit is never evicted. A **single** snapshot larger than the byte cap is rejected (the save returns an error) rather than cascade-evicting every other valid snapshot to make room for something that can't fit.

### Design subtleties worth noting
- **`.logits` classification.** `fs_validate_filename` permits a state file literally named `foo.logits`, so we can't blindly skip `*.logits`. A `<X>.logits` is treated as a sidecar **only if `<X>` also exists**; otherwise it's a real file (counted) or an **orphan sidecar** (reaped — otherwise orphans accumulate forever since we never count them).
- **`.tmp` files** (in-flight saves owned by a concurrent writer) are never counted or evicted.
- **Dedicated-directory contract.** With a cap set, `--slot-save-path` is treated as server-owned; this is documented in the flag help and the function header. With no caps (the default unless explicitly enabled) nothing is ever deleted.

---

## 5. What could have been done better

- **The reuse-predicate clause is subtle.** The `(slot_was_restored && has_new_suffix && cur.pos_min == pos_min_thold)` addition encodes a real recurrent-memory constraint in one boolean line inside a hot matcher. It works and is gated, but a maintainer may prefer this refactored into a named helper with the recurrent reasoning attached, rather than a third OR-clause.
- **Logits sidecar size.** We persist the *full* vocab (~150k × 4 B ≈ 600 KB) for one token. For the regenerate case that's the correct, lossless choice (a high-temperature sample can pick deep in the tail), but it's notable next to a multi-GB state file; a top-k variant was considered and rejected for correctness.
- **`get_timings` divide-by-zero** was a latent bug the `prompt_n=0` path exposed; arguably it should be fixed independently regardless of this feature.
- **`do_slot_restore` factoring.** In this PR the SLOT_RESTORE body is still inline; PR 2 factors it into `do_slot_restore` so the auto-restore path can share it. Doing that factor here would have made commit 2 cleaner.
- **Block/threshold coupling.** The fast path and the with-suffix path share state (`just_restored`) consumed in two different places; a small state machine would be clearer than a one-shot bool read in two spots.
- **No unit tests.** Validation was end-to-end on real models (Qwen3.6-27B recurrent + Llama-3.2-1B attention). Upstream may want a `test-` harness exercising save→restore→regenerate and the LRU eviction; the file I/O helpers are pure and unit-testable in isolation.
