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
| Shared-context checkpoint | `--slot-save-context-min-tokens` | **branch `auto-disk-kvcache-context-ckpt`** | the shared system+tools+RAG prefix of many chats is saved **once** as a deduplicated base; each chat chains a small delta off it |
| Restore floor | `--slot-restore-min-tokens` | **branch `auto-disk-kvcache-context-ckpt`** | skip a disk restore when the matched prefix is short enough that re-prefilling is cheaper |

"Branch" features are implemented and CPU-tested on their own feature branches, pending on-hardware
validation before they merge to the integration branch. See each branch's commit and the per-feature
docs below.

## Model-class support

The cache is model-general, but the **sub-range** features depend on how a model's KV can be sliced.
`llama-server` classifies each model's memory as `PART` (rewindable attention), `FULL`/`RS`
(recurrent/hybrid), or windowed (`n_swa > 0`, SWA/iSWA).

| Model class | Example | Incremental delta | Multimodal delta | Shared-context checkpoint |
|---|---|---|---|---|
| Dense / full attention (`PART`, `n_swa == 0`) | Qwen3.x dense | ✅ full (save **and** restore reuse) | ✅ write-side win¹ | ✅ |
| SWA / iSWA (`PART`, `n_swa > 0`) | Gemma | ✅ (attention delta + window saved whole) | ✅ (+ restore reuse) | ⛔ gated² → whole-snapshot caching |
| Recurrent / hybrid (`FULL`/`RS`) | Mamba / GDN (a3b-class) | ✅ attention delta + recurrent state whole; restore is extend-only | ✅ | ⛔ gated² → whole-snapshot caching |

¹ On dense models a **multimodal delta** cuts the per-turn write (no more re-writing the whole KV),
but restore still needs the whole verified prefix — the restore-reuse upside is on SWA.
² The shared-context checkpoint saves cells `[0, B)` of a *longer* prompt; that is **unsound** for
recurrent/hybrid (no positional KV to slice at `B`) and SWA (the window has already evicted `[0, B)`),
so it is **hard-gated off** for those classes — they keep normal whole-snapshot caching and never
produce a wrong result. This is a correctness boundary, not a bug.

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
    --slot-save-context-min-tokens 4096 \   # save the shared [0,B) prefix once as a base (min length)
    --slot-restore-min-tokens 0             # 0 = always restore; raise to skip loads shorter than this
```

`--slot-save-context-min-tokens` collapses N chats that share the same ~8–12k-token preamble from N
whole snapshots to **one base + N small deltas**. `--slot-restore-min-tokens` must be `<=` the save
floors (validated at startup); the default `0` changes nothing until you measure your own
restore-vs-reprocess crossover and raise it.

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

## Detailed design docs

- [`01-primitives-recurrent-restore.md`](01-primitives-recurrent-restore.md) — the libllama
  range-save / `NO_CLEAR` compose primitives and how recurrent/SWA restore is handled.
- [`02-auto-disk-cache.md`](02-auto-disk-cache.md) — the auto cache: the on-disk index, fingerprints,
  atomic publish, eviction.
- [`incremental-disk-cache.md`](incremental-disk-cache.md) — delta nodes, the `.meta` version map
  (v1 whole / v2 media / v3 delta / v4 media-delta), restore chain-walk, and the format-evolution note.
- [`03-multimodal-cache.md`](03-multimodal-cache.md) — how media snapshots and records work.
