# Disk KV cache — what this fork adds

This fork extends llama.cpp's slot state save/restore into a **transparent, cross-process disk
KV cache** for `llama-server`: prompt prefixes are automatically persisted to a shared directory
and restored on a later request — even from a *different* server process sharing the directory —
so a warm prefix survives VRAM eviction and is reused across a pool of instances.

Everything is **opt-in** and **off by default**; with the cache disabled, on-disk snapshot bytes
are byte-identical to upstream. Nothing here changes the core engine (`src/` / `include/`); it all
lives in `common/` and `tools/server/`.

## Feature status

| Feature | How to enable | Status | What it does |
|---|---|---|---|
| Auto disk cache | `--slot-save-auto` | **stable** | transparent save on idle + longest-prefix restore, cross-process |
| Incremental (delta) save | `--slot-save-incremental` | **stable** | a growing conversation saves only the new `[parent, N)` tail as a delta node instead of re-writing the whole prefix |
| Pin a snapshot | touch `<state>.pin` | **stable** | exempt one snapshot from eviction (e.g. a shared assistant/system base) |
| Multimodal delta | (uses `--slot-save-incremental`) | **branch `auto-disk-kvcache-mm-delta`** | image/audio conversations also save deltas (a new additive v4 sidecar), not a whole snapshot every turn |
| Shared-context base | `--slot-save-context-min-tokens` | **branch `auto-disk-kvcache-context-ckpt`** | the shared system+tools+RAG prefix of many chats is whole-saved **once** MID-PREFILL as a deduplicated base (sound for every model class incl. recurrent/hybrid); later chats restore it instead of re-prefilling, and with `--slot-save-incremental` chain a small delta off it |
| Restore floor | `--slot-restore-min-tokens` | **branch `auto-disk-kvcache-context-ckpt`** | skip a disk restore when the matched prefix is short enough that re-prefilling is cheaper |

"Branch" features are implemented and CPU-tested on their own feature branches, pending on-hardware
validation before they merge to the integration branch. See each branch's commit and the per-feature
docs below.

## Model-class support

The cache is model-general, but the **sub-range** features depend on how a model's KV can be sliced.
`llama-server` classifies each model's memory as `PART` (rewindable attention), `FULL`/`RS`
(recurrent/hybrid), or windowed (`n_swa > 0`, SWA/iSWA).

| Model class | Example | Incremental delta | Multimodal delta | Shared-context base |
|---|---|---|---|---|
| Dense / full attention (`PART`, `n_swa == 0`) | Qwen3.x dense | ✅ full (save **and** restore reuse) | ✅ write-side win¹ | ✅ |
| SWA / iSWA (`PART`, `n_swa > 0`) | Gemma | ✅ (attention delta + window saved whole) | ✅ (+ restore reuse) | ✅² |
| Recurrent / hybrid (`FULL`/`RS`) | Mamba / GDN (a3b-class) | ✅ attention delta + recurrent state whole; restore is extend-only | ✅ | ✅² |
| Sparse-attention hybrid **with an indexer cache** | `qwen4exp` (QSA) with indexer tensors | ❌ whole roots only, by design³ | ❌ (same reason) | ✅² |

¹ On dense models a **multimodal delta** cuts the per-turn write (no more re-writing the whole KV),
but restore still needs the whole verified prefix — the restore-reuse upside is on SWA.
² The shared-context base is now written MID-PREFILL as a **whole** state save, taken at the exact
instant a cold prefill has decoded precisely `[0, B_ctx)` (the block-aligned first-user boundary) and
nothing beyond — so the resident sequence *is* the true whole state at `B_ctx`. That is sound for
every class: dense attention holds exactly cells `[0, B_ctx)`, an SWA window has evicted nothing yet
(`N == B_ctx`), and a recurrent/hybrid fold is the correct fold over `[0, B_ctx)`. (The earlier design
saved a `[0, B)` **sub-range** with the slot sitting at `N`, which mislabelled the state-after-`N` as a
`B`-length prefix and was therefore hard-gated to dense-only; the mid-prefill whole-save removes that
gate — the production qwen3.6-27b, a `qwen35` hybrid, now writes and reuses a base.) Later chats sharing
the preamble RESTORE this base (longest-prefix restore) instead of re-prefilling it.
³ `llama_memory_hybrid_idx` (upstream's block-sparse-attention memory, built for `LLM_ARCH_QWEN4EXP`
when `hparams.indexer_head_size > 0`) appends a **third** indexer section to `state_write` that its
`state_read` unconditionally reads back. A position-range write cannot carry that section, and even if
it could the composed restore would be rejected: the indexer restore adopts the attention cache's slot
layout, which `llama_kv_cache::state_read_meta` refuses under `LLAMA_STATE_SEQ_FLAGS_NO_CLEAR`. So
`llama_memory_hybrid_idx::state_write_range` **fails closed**: whenever an indexer cache is present it
ignores `[p0, p1)` and writes the whole sequence, exactly as the `llama_memory_i` base default does.
The server's one-shot delta-capability probe then measures `nwrite == nwhole` and latches
`delta_cap::no`, so such an instance only ever publishes whole roots: correct, just not incremental.
A `qwen4exp` GGUF carrying **no** indexer tensors has a null `mem_idx`, falls through to
`llama_memory_hybrid::state_write_range`, and gets normal deltas. (The other indexer-bearing memory
classes, `llama_kv_cache_msa` for `minimax_m3` and `llama_kv_cache_dsa` / `llama_kv_cache_dsa_iswa`
for `glm_dsa` and `deepseek32`, derive straight from `llama_memory_i` and never overrode
`state_write_range` at all, so they have always inherited the same whole-write default and are
likewise safe.)

## Quick start (the common case: a dense chat model)

Everything you need for a single dense model, or a pool of identical instances sharing one directory:

```sh
llama-server -m your-model.gguf -c 32768 \
    --slot-save-auto \
    --slot-save-path /var/kvcache/shared \
    --slot-save-incremental \
    --slot-save-block 256 \
    --slot-save-idle-seconds 30 \
    --slot-save-max-count 200 \
    --slot-save-max-mb 56000
```

| Flag | Meaning |
|---|---|
| `--slot-save-auto` | turn the disk cache on (save on idle + restore on match) |
| `--slot-save-path DIR` | the shared cache directory (point every pool instance at the same one) |
| `--slot-save-incremental` | save deltas for growing conversations instead of whole snapshots |
| `--slot-save-block N` | prefix-hash block size (restore/delta boundaries land on multiples of this) |
| `--slot-save-idle-seconds N` | flush a slot's KV to disk after it has been idle this long |
| `--slot-save-max-count N` / `--slot-save-max-mb N` | LRU eviction caps (by file count and by total MB) |

With the **`auto-disk-kvcache-context-ckpt`** branch you additionally get, for dense models with a
large shared system prompt (agents, RAG, tool definitions):

```sh
    --slot-save-context-min-tokens 4096 \   # whole-save the shared [0,B_ctx) preamble once as a base (min length)
    --slot-restore-min-tokens 0             # 0 = always restore; raise to skip loads shorter than this
```

`--slot-save-context-min-tokens` collapses N chats that share the same ~8–12k-token preamble from N
whole snapshots to **one base** that every later chat restores (and, with `--slot-save-incremental`,
**N small deltas** chained off it). Unlike the earlier dense-only checkpoint, the base is written mid-
prefill as a whole state save and so works for **every model class**, including the recurrent/hybrid
qwen3.6-27b. `--slot-restore-min-tokens` must be `<=` the save floors (validated at startup); the
default `0` changes nothing until you measure your own restore-vs-reprocess crossover and raise it.

## Two rules for a shared directory

1. **Identical fingerprints.** Every instance writing to one `--slot-save-path` must agree on the
   fingerprint-affecting flags — `-c` / `--ctx-size`, `--cache-type-k` / `--cache-type-v`,
   `--slot-save-block`, the model, the mmproj, and RoPE settings. A snapshot from a mismatched
   instance is rejected (it is never restored into an incompatible context), so a mismatch silently
   disables cross-instance reuse rather than corrupting anything.
2. **Keep the prefix bit-stable.** Restore and delta-chaining need turn *N*'s tokens to be an exact
   prefix of turn *N+1*. A per-turn timestamp/date injected into the system prompt changes the prefix
   every turn and defeats the cache — keep volatile tokens out of the cached prefix (or after the
   first user message).

## Engine state-file format (`.bin`) and upstream version bumps

The `.bin` half of every snapshot unit is **libllama's** own sequence-state format, not ours. We
never version it; upstream does, through `LLAMA_STATE_SEQ_VERSION` in `include/llama.h`. The loader
compares it for **exact equality** (`src/llama-context.cpp`, `magic != LLAMA_STATE_SEQ_MAGIC ||
version != LLAMA_STATE_SEQ_VERSION`), so an upstream bump makes every previously written `.bin`
unreadable in both directions. This has now happened once.

### The 2026-08-29 bump: `LLAMA_STATE_SEQ_VERSION` 2 -> 3

Upstream commit `925e11799` (PR #27762, 2026-08-26, "llama: add token ID tracking to KV cell"),
absorbed by the merge of upstream `d7bd3bfca` into `main-patched`, changed the per-cell metadata:

* `llama_kv_cell_ext` grew a third field, `llama_token tok`, so the record went from `{x, y}`
  (8 bytes) to `{x, y, tok}` (12 bytes).
* The condition under which that record is written at all widened from `hparams.n_pos_per_embd() > 1`
  to the new `llama_kv_cache::has_cell_ext()`, which is
  `hparams.n_pos_per_embd() > 1 || hparams.ple_n_heads > 0`.

**Which models change bytes.** `n_pos_per_embd() > 1` holds exactly for M-RoPE and iM-RoPE models:
`qwen2vl`, `paddleocr`, `qwen3vl`, `qwen3vlmoe`, `qwen35`, `qwen35moe`, `qwen4exp`, `qwen3tts`,
`glm4` / `glm4moe` / `hunyuan_vl` when the GGUF sets M-RoPE, and `dflash` drafts that carry rope
sections. For these the per-cell record grows by 4 bytes. `ple_n_heads > 0` is set only by `qwen4exp`
with PLE tensors, which is iM-RoPE anyway, so it adds no new architecture in practice. Note that
`qwen35` is the arch of the deployed Qwen3.6 / Qwen3.8 27B builds, so the rig's main models are in
this group.

**Which models are byte-identical.** Everything else, i.e. plain NORM / NEOX rope with no PLE heads
(Llama, Mistral, Gemma, dense and MoE Qwen3, and so on), writes no per-cell extension record at all,
before or after. For those a freshly written `.bin` differs from a pre-merge one in exactly one place:
the 4-byte version word at file offset 4.

**That distinction does not buy compatibility.** The version word is checked before anything else, so
a v2 snapshot is refused by this build regardless of whether its cell records would have parsed. All
pre-merge units are dead on this build, and units written by this build are dead on pre-merge builds.

### What an operator has to do at deploy

The disk cache does **not** detect this itself, and that is by design: neither the `.meta` sidecar nor
the `model_fp` fingerprint (`identity_hash`, whose fields are listed in
[`02-auto-disk-cache.md`](02-auto-disk-cache.md)) carries an engine state-file version. A stale unit is
therefore still indexed, still selected as the longest verified prefix, and only then refused when
`llama_state_seq_load_file_ext` reads its header.

The consequence is a **graceful miss, not a failure**: `do_slot_restore` sees `nread == 0`, clears
`slot.prompt.tokens`, returns false, and the request cold-prefills (invariants 4 and 5). The wasted
work is one 8-byte header read per attempt. Nothing is corrupted and nothing crashes.

So the operator's options at deploy are:

1. **Purge each `--slot-save-path` directory** when rolling the new binary out. Cleanest: no stale
   units are ever selected, and the store starts at its true size.
2. **Leave the stores alone and let them heal.** Snapshot filenames are deterministic
   (`auto-<identity_hash>-<chain_hash>-<n_tokens>.bin`) and the fingerprint did not change, so a
   re-warmed prompt overwrites its own stale unit in place. Everything else ages out under the normal
   LRU caps. Until it does, it occupies quota it can no longer earn back.

Do **not** run a mixed fleet across the bump against a shared directory: both halves would keep
selecting and refusing each other's units, and neither would ever restore from the other.

### If it happens again

Two follow-on jobs come with any future `LLAMA_STATE_SEQ_VERSION` change:

* **Recapture the golden fixture.** `tools/server/tests/fixtures/golden-v1/*` is a bit-frozen `.bin` +
  `.meta` pair; it locks the *fork's* format, not upstream's, so a version bump invalidates it.
  `tools/server/tests/unit/test_slot_save_auto.py` now asserts the fixture's version word explicitly
  and names the recapture steps in the failure message, so this surfaces as one clear error instead of
  an opaque sha mismatch.
* **Re-read this section.** Whether the per-cell layout also moved decides which models are merely
  version-locked and which have genuinely different bytes.

Folding the engine version into the snapshot fingerprint was considered and **not** done: it would
only convert a cheap header-read miss into a filename miss, and making it actually bite would require
a new `.meta` field, hence a new sidecar version family and a compatibility path, for no correctness
gain.

## Detailed design docs

- [`01-primitives-recurrent-restore.md`](01-primitives-recurrent-restore.md) — the libllama
  range-save / `NO_CLEAR` compose primitives and how recurrent/SWA restore is handled.
- [`02-auto-disk-cache.md`](02-auto-disk-cache.md) — the auto cache: the on-disk index, fingerprints,
  atomic publish, eviction.
- [`incremental-disk-cache.md`](incremental-disk-cache.md) — delta nodes, the `.meta` version map
  (v1 whole / v2 media / v3 delta / v4 media-delta), restore chain-walk, and the format-evolution note.
- [`03-multimodal-cache.md`](03-multimodal-cache.md) — how media snapshots and records work.
