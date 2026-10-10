# Automatic disk prompt cache (fork)

Fork-only documentation of the automatic disk prompt cache (`--slot-save-auto`) and of how it extends the
`/slots` save and restore endpoints. The server README links here; the flags themselves are listed in its
parameter table.

## Automatic disk prompt cache (`--slot-save-auto`)

With `--slot-save-path DIR --slot-save-auto`, the server transparently persists each slot's prompt KV state to disk (when the slot is reassigned, flushed while idle, or on graceful shutdown) and restores the longest verified matching snapshot when a later request shares its prefix: across requests, restarts, and multiple server processes sharing one directory. Clients need no changes; reuse shows up as a collapsed `prompt_n` in `/completion` timings. The feature is off by default and, when off, changes nothing.

Each snapshot is a file unit named `auto-<fingerprint>-<chainhash>-<ntokens>`: the `.bin` llama state file, an optional `.logits` sidecar, and a small `.meta` sidecar carrying the identity data used for verification (fingerprint, token IDs, and, for multimodal snapshots, per-media-chunk identity records). Files are written to per-writer-unique temp names and atomically renamed, `.meta` last, so a torn or half-published unit is never indexed. Usage examples live in [`README-FORK.md`](../../README-FORK.md); design write-ups in [`docs/kv-cache/`](../../docs/kv-cache/).

### Multimodal prompts (images and audio)

On a server started with `--mmproj`, the cache is fully multimodal:

- **Text-only prompts are cached exactly as on a text-only server.** Their snapshots use the version-1 `.meta` format, byte-for-byte identical to the pre-multimodal format, so old and new builds interoperate on text snapshots indefinitely (a snapshot written by an older build still indexes and restores: the scan verifies identity from the `.meta`, not the file name). The snapshot *file name* carries a fingerprint of the full runtime identity (model, plus cache type, context size, rope/YaRN and projector settings) so instances differing in any of those never write over each other's units in a shared cache directory.
- **Prompts containing images or audio are cached too, storing metadata only.** The media embeddings are already inside the `.bin` state file; the version-2 `.meta` sidecar adds ~100 bytes of *identity* per media chunk: a 64-bit FNV-1a hash of the raw uploaded bytes, the chunk's token and position counts, its token-grid geometry, and its type (image/audio). No pixels, samples, or embeddings are stored beyond the state file itself.
- **Reuse requires a byte-identical re-upload** of each media file (typical multi-turn clients resend attachments unchanged). A request with the same text but a different image reuses exactly the prefix before that image; with adjacent images, each is verified independently, so a mismatch in the second still reuses through the end of the first.
- Media chunks without an identity (e.g. placeholder bitmaps) are never saved and never matched.
- The projector file gets its own fingerprint (`fp_mmproj`, a header-only GGUF hash computed at load, logged at startup and reported in `/props`). Swapping or requantizing the mmproj refuses media restores while text snapshots keep working.

### Verification invariants

The cache is written around invariants that hold on every path:

1. **Never trust a hash or a file name.** The block-hash index only narrows candidates; before any restore, the snapshot's persisted token IDs are byte-compared against the request, and every media record inside the matched prefix is compared against the request's live media chunks (identity hash, token/position counts, geometry, type). Any mismatch truncates or refuses the reuse.
2. **All identity verification happens before any multi-GB I/O.** The `.meta` sidecar (a few KB) is fully parsed and verified (magic, version, fingerprint equality, token byte-compare, media-record checks) before the state file is opened at all.
3. **Media records must exactly tile the media cells.** In a version-2 `.meta`, the records must cover exactly the `LLAMA_TOKEN_NULL` cells of the persisted token array: ordered, disjoint, in-bounds, nothing uncovered and nothing over text. Any violation (including a version-1 file containing media cells) rejects the file, and the server falls back to a normal prefill. Unknown or unparsable `.meta` versions are skipped and remembered by file name, so rescans never re-read them.
4. **64-bit media identity is a documented residual risk, deliberately accepted.** Two different images whose raw bytes collide under FNV-1a *and* match in token count, position count, and grid geometry could impersonate each other. This is the same trust model the in-memory prompt cache applies to token IDs; text cells are still byte-compared exactly.
5. **Failure means prefill, never a wrong answer.** Corrupt units, torn writes, fingerprint mismatches, out-of-space, or any restore failure degrade to normal prompt processing.

### Restore semantics: whole-snapshot vs partial prefix

What "matching prefix" means depends on the model's memory type:

- **Recurrent and hybrid models** (Mamba-class, gated-delta hybrids such as Qwen3.6, Jamba: anything whose memory refuses partial sequence removal): a state cannot be partially rewound, so a snapshot restores **only when the request extends the whole snapshot**: a byte-for-byte prefix of the incoming request. Mid-prefill and shutdown/idle-flush states are reliably restorable this way, and so is a completed turn (a prompt+generation snapshot) whenever the client re-sends the conversation as a stable extending prefix: agentic tools that re-render history to byte-identical tokens are the multi-turn image case. A media prompt is saved on exactly the same terms as a text one, so a template that re-renders prior turns differently (e.g. dropping an empty reasoning block) simply diverges before the snapshot's end and cold-prefills; the mismatch is a cache miss, never a wrong restore.
- **Plain attention models**: a request may diverge inside a snapshot; the verified prefix is reused, clamped down to a whole hash block that does not split a media chunk, and the divergent tail is re-processed.
- **SWA attention models** (sliding-window, e.g. Gemma 3 class): partial-prefix reuse works, but the restored cache only holds roughly the last `n_swa` positions of the snapshot. A request that diverges further before the snapshot's end than the window covers drops the restore and prefills cold (logged, never a crash).

The restored KV state round-trips faithfully, but a *fully* restored prompt (one the request re-sends byte-for-byte so that nothing is re-prefilled) is not guaranteed to continue bit-identically to an uninterrupted run. With no re-prefill the first new token is sampled from a single decode into the loaded state, whereas a cold run computes that position inside a prefill batch; on some backends (notably flash-attention on SYCL) those two paths reduce in a different order and can pick a different token at a near-tie, after which generation stays coherent. This matches upstream `/slots`, which likewise makes no bit-identity promise. Any request that re-prefills even a few tokens past the reused prefix (the common case, since real reuse *extends* a prefix) is deterministic and bit-identical.

### Manual `/slots` save/restore with multimodal

The manual `/slots/{id}?action=save|restore` endpoints use upstream's implementation and file format: the state file carries the slot's tokens and, for a media slot, each media chunk's id and geometry (no pixels; the embeddings come from the state file). A restored media chunk is a placeholder that is never re-encoded: any follow-up that would require re-processing it (e.g. a divergence before it) drops it and re-processes from the request's own data. Upstream's format records no model or `--mmproj` fingerprint, so the fork writes a `.fp` sidecar next to each save and refuses a restore whose `.fp` names another model or, for a media snapshot, another projector; a file without one (saved by upstream) is restored unchecked. A fork-format file that has a `.meta` sidecar (an auto-cache unit, or a media save made by an older fork build) is restored the fork's way: a delta unit with its parent chain, media rebuilt from the sidecar.

### Operational notes

- **Disk capacity:** before each save the server queries free space and skips the write unless the exact state size plus 10% slack fits: on filesystems such as btrfs an ENOSPC mid-write can be much worse than a skipped cache write. Size `--slot-save-max-mb` to your budget; the store directory is treated as dedicated.
- **Idle-delay flush (`--slot-save-idle-seconds`, default 60):** the reassign and shutdown flushes only fire when the *next* task arrives or the process exits, so without this a lone request's KV stays crash-volatile and invisible to peer instances until more traffic lands. With it, a slot idle for that many seconds is flushed on the main loop thread, off the request hot path, via a single timed wakeup bound to the earliest slot's deadline (no busy poll, works with sleeping disabled). At most one slot is flushed per wakeup, and the loop re-checks for queued work between flushes, so an incoming request never waits behind more than one write. The flush also deduplicates against what is already on disk, so it is at most one write per slot per idle period. When both this and `--sleep-idle-seconds` are set, a pending flush is completed before the server sleeps (sleeping unloads the KV cache), even if the sleep threshold is the shorter of the two. Set `-1` for the legacy write-on-reassign/shutdown-only behaviour.
- **Shutdown flush and `TimeoutStopSec`:** on graceful shutdown every slot's warm KV is flushed to disk, bounded by a 60-second deadline checked *between* slots: an in-flight write is never aborted, so the worst case is the deadline plus one full snapshot write. Under a service supervisor, size the stop timeout (systemd `TimeoutStopSec`) above 60 s plus the time to write your largest snapshot (roughly `--slot-save-max-mb` at your disk's sequential write speed).
- **Migration and mixed fleets (`.meta` sidecar):** version-1 text snapshots written by older builds keep their file names, index keys, and `.meta`-level restorability unchanged, and new text snapshots remain version-1: old and new binaries interoperate on the sidecar indefinitely. Media (version-2) units are skipped by older binaries and evicted normally by their LRU; nothing breaks, but older binaries re-read every version-2 `.meta` on each rescan, so when several instances share one cache directory, **roll the whole fleet in one window**. This guarantee covers the `.meta` only; the `.bin` beside it is versioned separately, see the next bullet.
- **The `.bin` state file follows upstream's `LLAMA_STATE_SEQ_VERSION`, not ours.** Its header is checked for exact version equality before anything else is read, so an upstream bump makes every previously written snapshot unreadable in both directions, whatever its `.meta` says. **This build writes version 3** (upstream raised it from 2 in `925e11799` / PR #27762, absorbed 2026-08-29, which grew the per-cell record for M-RoPE and PLE models). Snapshots from any earlier build of this fork are refused here, and this build's snapshots are refused by them, gracefully in both cases: the load fails at the header, the slot is cleared, and the request cold-prefills. Nothing in the file name or the `.meta` fingerprint encodes the state version, so a stale unit is still indexed and selected, then refused; it costs one header read per attempt and heals when the same prompt is re-warmed and overwrites it. Either purge the cache directories when rolling across a bump, or accept the misses while the store turns over. Do **not** share one directory across the bump: both sides would keep refusing each other's units. Details, and the list of which architectures actually changed bytes, are in [`docs/kv-cache/README.md`](../../docs/kv-cache/README.md#engine-state-file-format-bin-and-upstream-version-bumps).
- **Older builds predating the M-RoPE per-cell records** (upstream #20132/#20273, before March 2026) misparse rather than reject such files, because those records landed without a version bump. That hazard is now historical: from version 3 onward the version word discriminates and an incompatible build refuses the file outright.

## The `/slots` endpoints and `--slot-save-auto`

The three endpoints above save and restore a slot on demand. `--slot-save-auto` turns
`--slot-save-path` into a transparent, self-managing disk prompt cache instead: the server
automatically persists each slot's warm KV (on slot reclaim, after `--slot-save-idle-seconds` of
idleness, and on graceful shutdown) and automatically restores the deepest matching prefix on a
later request, across restarts and across peer instances sharing the directory. Snapshots are
keyed by a rolling block hash of the prompt's token ids (`--slot-save-block`), guarded by a model
fingerprint so a snapshot is never reused under a different model/geometry, and bounded by
`--slot-save-max-count` / `--slot-save-max-mb` (least-recently-used eviction, scoped to the auto
cache: plain `--slot-save-path` never deletes files). Each snapshot is a three-file unit published
atomically: `<state>.bin` (the KV blob), an optional `<state>.bin.logits` sidecar, and a
`<state>.bin.meta` sidecar carrying the token ids and fingerprint.

`--slot-save-incremental` changes only how a *continuation* of an already-saved prompt is written.
Without it, every turn re-saves the whole KV blob, so the prefix already on disk is re-copied off
the device and rewritten each turn: expensive for a long, growing conversation. With it, a
continuation is written as a **delta node** that stores only the KV cells appended since its
parent snapshot, plus a small pointer to that parent; the shared prefix is written once and reused
by every continuation and fork. Hybrid (recurrent) and sliding-window (SWA) models still write
their bounded recurrent/window state whole in each node: only the unbounded, append-only
full-attention prefix is delta'd. On restore the server walks the parent chain and loads the base
plus each delta in position order to reconstruct the full prefix; the reconstructed continuation is
token-identical to an uncached run. The first (parentless) save of a lineage stays a whole
snapshot, and a context shift (which rewrites already-saved token positions) transparently rebases
to a fresh whole snapshot. `--slot-save-incremental` requires `--slot-save-auto`. The on-disk
format is documented in [`docs/kv-cache/incremental-disk-cache.md`](../../docs/kv-cache/incremental-disk-cache.md).

The flag is a request, not a promise: on the first delta write the server probes whether this model's
memory implementation actually honours a position range, and falls back to whole snapshots for the
rest of the process if it does not. That is what happens on block-sparse-attention models that carry
an indexer cache (`qwen4exp` with indexer tensors): their state cannot be split into composable
deltas, so the flag is accepted and silently degrades to whole roots rather than writing a delta that
could not be restored. Everything else about the cache is unaffected.

`--slot-save-context-min-tokens` adds a **shared-context checkpoint**. When many chats share the
same leading context (a common system prompt, developer/tool preamble, or a large retrieved (RAG)
block, everything before the first user turn), each one would otherwise persist a near-identical
whole prefix, a fan-out of large files. Instead the server saves that shared prefix `[0, B)` **once**
as a deduplicated base (B = the end of the system preamble, found by rendering the request's system
messages and tools through the chat template followed by placeholder conversations; gated to
`>= max(--slot-save-block, --slot-save-context-min-tokens)`), and every later same-context chat writes
nothing for the base. Combined with `--slot-save-incremental`, each chat's own save then collapses to
a small `[B, N)` delta parented on that one base: one base plus N deltas instead of N whole prefixes.
The base is a normal v1 root and the children normal v3 deltas: **no new on-disk format**. Without
`--slot-save-incremental` the children still whole-save, so the base only accelerates *restore*.

The checkpoint is written **only for pure-attention text models** (`PART` sequence removal,
`n_swa == 0`, non-media prompts). It is deliberately suppressed for recurrent/hybrid (e.g. GDN) and
sliding-window (SWA) models, where a `[0, B)` sub-range of the KV is not a sound prefix state and
would silently produce wrong output; those classes fall back to the existing whole-prefix save with
no change. Templates that move the system prompt into a later turn (Mistral-Nemo puts it in the last
user message) have no stable system prefix, so the boundary falls before it and the node is not
written.

`--slot-restore-min-tokens` is an absolute floor on the restore side: when the byte-verified matched
prefix is shorter than N tokens the server reprocesses the prompt rather than paying the multi-GB disk
read + host copy, which for a near-cold slot is faster. It gates **all** auto-restores, so the default
is `0` (never skip: no behaviour change for existing users); raise it after measuring your own
disk-read-vs-reprocess crossover. It must not exceed `--slot-save-context-min-tokens` (or
`--slot-save-min-tokens`), else a snapshot could be written and then always skipped on restore.
