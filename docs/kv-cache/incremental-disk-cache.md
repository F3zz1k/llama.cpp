# Incremental disk KV cache (`--slot-save-incremental`)

This document describes the on-disk format and the save / restore / evict flow of the server's
auto disk prompt cache when incremental saving is enabled. It complements the user-facing prose in
[`tools/server/README.md`](../../tools/server/README.md). The implementation lives in
`tools/server/server-context.cpp` (cache logic) and `src/llama-*.cpp` / `include/llama.h` (the
libllama primitives).

## Motivation

Under `--slot-save-auto` alone, every save writes the **whole** KV blob for the slot's current
prompt. In a growing multi-turn conversation each turn re-copies the entire already-persisted
prefix off the device and rewrites it to disk — O(N) bytes per turn, O(N²) over a conversation,
multiplied across every instance sharing the cache directory.

`--slot-save-incremental` writes each continuation as a **delta node** that stores only the KV
cells appended since its parent snapshot. The shared prefix is written once and reused by every
continuation and fork of that lineage.

### Decomposition principle

A transformer slot's persisted state splits into two parts:

* **Append-only, unbounded full-attention KV** — grows by exactly the new tokens each turn and is
  never rewritten for old positions. This is the redundancy incremental saving eliminates: a delta
  node stores only cells `[parent_hi, N)`.
* **Bounded, fixed-size state** — the recurrent conv/SSM state of hybrid models and the
  sliding-window (SWA) attention window. This is small and *genuinely* changes every turn, so each
  node re-saves it **whole**. It is never the redundancy we are killing.

So: *delta the append-only full-attention prefix; whole-save the bounded state at every node.* This
is why the scheme works uniformly for dense, hybrid/recurrent, and SWA models. The split is
implemented in the range-save path (see "libllama primitives" below); the server does not special-
case model families.

## `.meta` sidecar version map

Every snapshot unit is three files published atomically (state last-to-appear is the `.meta`, the
scan key): `<state>.bin`, an optional `<state>.bin.logits`, and `<state>.bin.meta`. The `.meta`
carries the token ids, a model fingerprint, and (for delta nodes) the parent link. Its second u32
is a version:

The four versions are the 2×2 of {whole | delta-node} × {text | media}:

| version | name | meaning | trailing section(s) |
|---------|------|---------|---------------------|
| **1** | `SLOT_META_VERSION` | whole **text** snapshot (a lineage **root**, covering `[0, N)`) | none |
| **2** | `SLOT_META_VERSION_MEDIA` | whole **media** snapshot (root with NULL media cells) | media |
| **3** | `SLOT_META_VERSION_NODE` | **text delta node** (a continuation covering `[range_lo, range_hi)`) | node |
| **4** | `SLOT_META_VERSION_MEDIA_NODE` | **media delta node** (continuation with NULL media cells) | media, then node |

Trailing sections are always written **media-then-node**, so the four layouts are prefix-nested
(v1 ⊂ v2, v1 ⊂ v3, and v2+node = v4). The `(is_node, has_media)` → version mapping lives in exactly
one place per direction — `slot_meta_version_for` (write) and `slot_meta_features_for` (read) — so
the writer and reader can never disagree.

A **root stays v1/v2** — a whole snapshot's bytes are byte-identical whether or not
`--slot-save-incremental` is set (a golden lock enforced by the tests). Only continuations with a
found parent are written as a delta node (v3 text, v4 media).

### `.meta` byte layout

All integers are little-endian. The header is common to all versions:

```
u32   magic          = 0x544D4B4C  ("LKMT")
u32   version        = 1 (whole text) | 2 (whole media) | 3 (text delta) | 4 (media delta)
...   model fingerprint  (fp_model u64, geometry, cache types, rope/yarn, lora, mmproj bit)
u32   tok_count
u64   chain_hash                       # rolling block hash of the whole token prefix
i32[tok_count]  tokens                 # the full [0, range_hi) cell-token list (media cells = NULL)
```

The **media tail** (v2, v4) appends, after the tokens — describing the WHOLE `[0, N)` tiling even
when the `.bin` is a delta:

```
u64   fp_mmproj                        # projector fingerprint (authoritative; not backfilled)
u32   n_media
per record: u32 start_idx, n_tokens, n_pos, nx, ny, is_audio, id_len, then id_len id bytes
```

The **node tail** (v3, v4) appends, after any media tail:

```
u64   parent_id      # the parent node's chain_hash (0 = root)
u32   range_lo       # first KV position this node's .bin holds  (== parent's token count)
u32   range_hi       # one past the last KV position             (== this node's tok_count)
```

A v1 meta has no trailing section, so its bytes are unchanged from the non-incremental format; a
reader treats a whole snapshot (v1/v2) as an implicit root with `parent_id = 0`, `range_lo = 0`,
`range_hi = tok_count`. For **v4 the meta is WHOLE while the `.bin` is a DELTA**: the token array
and media tiling cover `[0, N)` (so restore's byte-verify is the same v2 path), while `range_lo/hi`
describe only the `[parent_hi, N)` cells the `.bin` actually holds.

### On format evolution (why version-per-feature, and the long-term path)

Today each new capability gets its own version byte and the 2×2 above is enumerated explicitly in
`slot_meta_version_for` / `slot_meta_features_for`. This is deliberate for a small, additive feature
set: v1–v3 stay byte-frozen, the golden-lock tests are trivial, and a corrupt/relabelled file is
rejected by the exact-EOF check. It does **not** scale — an Nth independent, composable feature would
demand up to `2^N` version bytes. The intended long-term migration, when a *third* orthogonal tail
appears, is to move to **capability flags + self-describing sections**: replace the version byte with
a feature-flags word plus a sequence of `(section-id, length)`-prefixed tails, so the reader skips
unknown sections and features compose without a combinatorial version table. Because the version↔
feature dispatch is already funnelled through the two helpers above (not scattered across the writer
and reader), that migration is a contained change to those helpers plus one new reader loop — it is
structured for now and does not need to be built until the third tail exists.

### Filename scheme is the tree

Auto snapshots are named deterministically:

```
auto-<fp_model:016x>-<chain_hash:016x>-<n_tokens>.bin
```

The `(chain_hash, n_tokens)` pair is a node's unique on-disk key. A delta's `(parent_id, range_lo)`
is exactly the `(chain_hash, n_tokens)` of its parent, so a parent's filename is reproduced by
`auto_state_filename(parent_id, range_lo)` with no separate index. **The tree is derived from disk:**
there is no in-RAM tree map — parents are resolved by name, and eviction recomputes the child
refcount by reading the on-disk metas each pass (which is cross-process correct with multiple
writers sharing the directory).

## Save flow (find parent → emit delta node)

On a save (`auto_save_slot_if_useful`), after the usual floor/dedup gates, when
`--slot-save-incremental` is set:

1. Look up candidate snapshots for the current prompt (`auto_index_lookup`, longest-first).
2. The deepest candidate whose persisted tokens are a **strict prefix** of the current prompt under
   the same fingerprint (`disk_toks == toks[0:len]`, `len < toks.size()`, `fp == cur_fp`) is the
   parent. `parent_hi = len`; `parent_id` = the block-hash of `toks[0:parent_hi]` (which equals the
   parent's own `chain_hash`).
3. Write the state with `llama_state_seq_save_file_range(ctx, tmp, slot, /*p0=*/parent_hi, /*p1=*/-1,
   tokens, N)` — only cells `[parent_hi, N)` for full-attention, plus bounded state whole — and a v3
   `.meta` carrying `parent_id`, `range_lo = parent_hi`, `range_hi = N`.
4. If no parent is found (or the flag is off), take the **exact whole-save path** — a v1 root,
   byte-identical to the non-incremental format.

Everything after the parent decision (unique temp path, optional logits sidecar, atomic temp+rename
publish with `.meta` last, index insert, LRU enforcement) is **shared** by both modes.

A **context shift** rewrites already-saved token positions, so the dirtied parent fails the strict
byte-verify in step 2 → no parent is found → a whole (root) save. That is the automatic rebase; it
needs no extra flag.

## Restore walk (root → tip, compose with NO_CLEAR)

On restore (`auto_restore_into_slot`), the candidate (tip) meta is read with the node fields and
its token prefix is byte-verified against the request as usual. Then the **root→tip chain of node
`.bin` paths** is built by walking parent links (all *before* touching the slot, so any
inconsistency simply yields a cold prefill and never disturbs resident KV):

* If the tip is a root (`parent_id == 0 && range_lo == 0`) the chain is just `{tip}` — a single
  clearing load, bit-identical to the previous non-incremental restore.
* Otherwise resolve `parent_path = auto_state_filename(parent_id, range_lo)`, read its meta, and
  repeat to the root. Each hop validates: fingerprint match, **contiguity** (`parent.range_hi ==
  child.range_lo`), the parent's tokens byte-equal the tip's prefix (defends against a filename hash
  collision), the parent `.bin` exists, and a bounded depth. Any break → return 0 → cold prefill.

The chain is then loaded in position order (`do_slot_restore`): element `[0]` (the root) with a
normal clearing load, and each subsequent delta with `LLAMA_STATE_SEQ_FLAGS_NO_CLEAR` so it appends
its cells instead of wiping the sequence. For an iSWA model the tip's sliding-window sub-read is
loaded **with** clear (mask `NO_CLEAR` off `kv_swa`) so the window is overwritten by the tip's
bounded blob (tip wins) while the global-attention `kv_base` composes across the chain. Every node
on the chain has its mtime touched so a hot shared base stays warm under LRU.

## Eviction (tree-aware LRU, never orphan a live child)

`slot_save_enforce_limits` reads every auto `.bin`'s `.meta` and reconstructs the forest:

* **Orphan reap** up front: a delta whose base file is gone can never be restored (a base-less delta
  would corrupt a compose-load), so it is deleted regardless of the caps; iterate to a fixed point
  since reaping one delta can orphan its children.
* Build `child_count[(chain_hash, n_tokens)]`; a node is a **leaf** iff its child count is 0.
* Group nodes into trees by walking parent links to the root; `tree_recency` = max mtime over the
  tree.
* While over `--slot-save-max-count` / `--slot-save-max-mb`: evict the **evictable leaf** with the
  smallest `(tree_recency[root], mtime)` — the oldest tip of the least-recently-used tree. Evicting
  a leaf may expose its parent as a new leaf. **A node with a live child is never evicted**, and the
  just-written unit is never evicted; if every over-cap node still has a live child, the store is
  left (correctly) above the cap rather than orphaning a base out from under its child.

When the store contains only v1 roots (whole-snapshot mode) every node is its own leaf, so this
reduces exactly to the pre-existing flat mtime LRU (golden-safe).

## libllama primitives

Two C API additions in `include/llama.h` back the scheme (backend-general; no new kernels):

* **P1 — range-filtered save:** `llama_state_seq_save_file_range(ctx, path, seq, p0, p1, toks, n)`
  emits only cells with position in `[p0, p1)` for full-attention memory, while bounded memory
  (recurrent, sliding-window) is always serialized whole. The file format is identical to
  `llama_state_seq_save_file`, so a delta is loadable by the normal load path. Hybrid memory splits
  the write: attention delta via the range save + recurrent state whole.
* **P2 — no-clear load:** `LLAMA_STATE_SEQ_FLAGS_NO_CLEAR` (value 4) and
  `llama_state_seq_load_file_ext(...)`. Without the flag a load first does `seq_rm(dest, -1, -1)`
  (a full wipe); with it, the wipe is skipped so a delta appends into an existing sequence and
  base + deltas compose when loaded in position order.

## Correctness invariants

* Whole-snapshot (v1 root) bytes are byte-identical with the flag on vs off (golden lock).
* Any missing / corrupt / non-contiguous node on the restore walk → cold prefill, never a partial or
  wrong-KV restore.
* A base with a live delta child is never evicted; orphan deltas are reaped early.
* In-place mutation of already-saved positions (context shift, eviction) rebases to a fresh root.
* Reconstructed continuations are token-identical to an uncached run (verified for dense and iSWA,
  flash-attention on and off, in `tools/server/tests/unit/test_slot_save_incr.py`).
