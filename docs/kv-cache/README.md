# Disk KV cache — what this fork adds

For the copy-paste quick start (command lines, every flag with its default, the checkpoint triggers,
hits and misses), see [`../disk-cache.md`](../disk-cache.md). This page is the design overview.

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
| Sparse attention with an indexer or compressed KV | `qwen4exp`, `glm5-next`, `glm-dsa`, `deepseek32`, `minimax-m3`, `deepseek4` | ✅ every section takes the range³ | ✅ | ✅² |
| Pure recurrent | Mamba / Mamba-2 / RWKV | whole roots only (the state is a fold of the whole prefix) | n/a | ✅² |

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
³ Since 2026-10-03. Before that, `llama_memory_hybrid_idx` with an indexer (`qwen4exp`, `glm5-next`)
deliberately wrote whole, and the DSA, MSA and DSV4 caches had no `state_write_range` at all, so every
save on those models was a full multi-GB root. The per-class detail and the test that checks each
composition against an uncached run are in
[`incremental-disk-cache.md`](incremental-disk-cache.md#which-memory-classes-honour-a-range-and-what-happens-when-they-do-not).

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
   fingerprint-affecting flags: `--cache-type-k` / `--cache-type-v`, `--slot-save-block`, the model,
   the mmproj, and RoPE settings. `-c` / `--ctx-size` is not one of them (see context rungs below). A snapshot from a mismatched
   instance is rejected (it is never restored into an incompatible context), so a mismatch silently
   disables cross-instance reuse rather than corrupting anything.
2. **Keep the prefix bit-stable.** Restore and delta-chaining need turn *N*'s tokens to be an exact
   prefix of turn *N+1*. A per-turn timestamp/date injected into the system prompt changes the prefix
   every turn and defeats the cache — keep volatile tokens out of the cached prefix (or after the
   first user message).

## Recommended configurations

These are starting points; every flag is explained in the table under Quick start.

**A pool of identical instances of one model, sharing one store** (the usual production shape):

```sh
llama-server -m model.gguf -c 131072 -ngl 999 -fa on --parallel 1 \
    --slot-save-path /mnt/nvme/kvcache/shared \
    --slot-save-auto --slot-save-incremental \
    --slot-save-block 256 \
    --slot-save-idle-seconds 30 \
    --slot-save-context-min-tokens 4096 \
    --slot-save-max-mb 100000 \
    --metrics
```

* One directory and **one cap for every pool**. `--slot-save-max-mb` is enforced over the whole
  directory by whichever instance saves, so a pool with a tighter cap would govern eviction for
  everyone. Raise the cap only when the miss counters below show evictions costing hits.
* Put the store on NVMe. A restore reads the whole chain, and a spinning disk turns a sub-second
  restore into minutes.
* `--metrics` exposes the counters described under "Observing the cache".

**The same, with speculative decoding.** Nothing changes on the cache side; add the draft as usual:

```sh
    --spec-type draft-mtp --spec-draft-n-max 3            # MTP head inside the model GGUF
    --model-draft draft.gguf --spec-draft-n-max 4          # or a separate draft model
```

Each unit then also gets a `.dft` draft sidecar, so a restored conversation drafts warm (see
"Speculative decoding and the cache"). Snapshots stay interchangeable between instances with and
without speculation: the target `.bin` is the same either way, and an instance without a draft
simply ignores the sidecars.

**Context rungs (the same model at several context sizes, e.g. 1 GPU at 131072 and 2 GPUs at
262144, or a lower-context vision variant beside the text one).** `-c` is not part of the
fingerprint: no memory class writes anything that depends on the cache size (the audit is on
`model_fp::fp_n_ctx` in `tools/server/server-common.h`). Rungs therefore name units alike, never keep
two copies of one prefix, continue each other's delta chains, and restore each other's units in both
directions whenever the unit fits the reader's context (a longer one is skipped before its file is
opened). They must use identical fingerprint fields otherwise: same model file, same
`--cache-type-k/v`, same `--slot-save-block`, and the **same RoPE/YaRN settings**
(`--rope-scaling`, `--rope-scale`, `--yarn-orig-ctx`, ...). The GPU count and tensor split are not
part of the fingerprint. The one way `-c` reaches the KV is LongRoPE (`rope_factors_long` /
`rope_factors_short`, chosen by `n_ctx_seq > n_ctx_orig_yarn`), so which side of that threshold a rung
sits on is part of the fingerprint.

## Speculative decoding and the cache

With a separate draft context (`draft-mtp`, `draft-eagle3`, `draft-simple` with `--model-draft`),
every auto-save also writes `<unit>.dft`: the draft context's cells for the same `[lo, N)` split as
the unit. A restore loads the chain's `.dft` files in the same order and with the same `NO_CLEAR`
composition as the target, all or nothing. Without them the draft would attend over a hole `[0, L)`
for the rest of the restored conversation, silently lowering acceptance.

Draft KV only changes how many drafted tokens are accepted, never the output (the target verifies
every token), so a stale or missing sidecar costs speed, never correctness. A shared-cells draft
(Gemma-4 class) serialises nothing and needs none. `.dft` files are accounted, evicted and reaped
with their unit. Check `llamacpp:auto_cache_restore_draft_warm_total` against
`auto_cache_restore_draft_cold_total` to see how often a restore came back warm.

## Observing the cache (hits, misses, saves)

With `--metrics`, `GET /metrics` carries these cumulative counters (prefix `llamacpp:`):

| Counter | Meaning |
|---|---|
| `auto_cache_restore_hit_total` | requests that restored a prefix from disk |
| `auto_cache_restore_miss_total` | requests with at least one whole block beyond the in-memory match that restored nothing |
| `auto_cache_restore_not_prefix_total` | misses where a saved unit shared the prefix but the memory class cannot rewind into it (needs a node at or before the divergence, see `--slot-save-node-prompt`) |
| `auto_cache_restore_discarded_total` | restores whose tokens a later clamp threw away before use (counted as misses) |
| `auto_cache_restore_failed_total` | restores whose load failed after clearing the slot (fell back to a shorter snapshot or cold) |
| `auto_cache_restore_tokens_total` | prompt tokens restored from disk |
| `auto_cache_save_root_total` / `auto_cache_save_delta_total` | whole roots / delta nodes published |
| `auto_cache_save_bytes_total` | state bytes published |
| `auto_cache_save_whole_fallback_total` | parented saves published whole because the memory type cannot write deltas |
| `auto_cache_save_failed_total` | saves dropped with nothing published (each logs a rate-limited WRN with the reason) |
| `auto_cache_evicted_total` | units this instance evicted to stay under the caps |
| `auto_cache_save_draft_total`, `auto_cache_restore_draft_{warm,cold}_total` | draft sidecars, see above |

A miss includes prompts no cache could have held (a brand-new conversation), so read it next to
`auto_cache_evicted_total`: misses that climb together with evictions are the sign the store is too
small. Per request, `timings.cache_disk_n` (present only when non-zero) is the part of `cache_n`
that came from disk rather than from the resident slot, and `timings.cache_ram_n` the part loaded
from the RAM prompt cache; a router can forward both to its clients:

```sh
curl -s localhost:8080/metrics | grep auto_cache_
curl -s localhost:8080/completion -d '{"prompt":"...","n_predict":16,"cache_prompt":true}' | jq .timings
```

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
