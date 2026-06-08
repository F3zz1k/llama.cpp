# Fork notes — `auto-disk-kvcache`

This branch adds an **opt-in automatic disk prompt/KV cache** to `llama-server`
(on top of upstream), so a slot KV state can be persisted to disk and restored
across requests, processes, and instances — including for hybrid/recurrent models.

Key commits:
- `server : opt-in automatic disk prompt/KV cache (--slot-save-auto)`
- `server : regenerate (no-suffix restore-continue) from saved logits + bounded slot-save store`
- `server: reuse disk-restored KV slot for hybrid/recurrent models (gated on just_restored)`

Relevant flags: `--slot-save-path <dir> --slot-save-auto --slot-save-block <N>
--slot-save-max-count <N> --slot-save-max-mb <MB>`.

## ⚠️ Operational rule: the disk-cache fingerprint is the cache IDENTITY

A snapshot is only restorable by a server whose **fingerprint** matches
(`model_fp::operator==` in `tools/server/server-context.cpp`). That fingerprint
deliberately includes **`n_ctx`, `mmproj_loaded`, `cache-type-k/v`,
`slot-save-block`, and rope/yarn** — not just the model.

**Therefore every instance that shares a `--slot-save-path` (a primary "seed",
its scale-out replicas, and any refresh/pin helper) MUST be launched with
IDENTICAL:**

- `-c` (context size)
- `--mmproj` (loaded or not)
- `--cache-type-k` / `--cache-type-v`
- `--slot-save-block`
- rope / yarn settings

A mismatch **silently** splits the store into incompatible fingerprint
namespaces: each side skips the other-fingerprint `.bin`s during indexing, so a
cross-instance or cold-start lookup never matches and you get a **full cold
prefill instead of a warm restore — with no error, just slow.**

Prod example (2026-06-07): a `qwen35b-a3b` seed launched with `-c 524288` could
not restore pins written by its pool variant / refresh helper at `-c 262144`
(both were `--mmproj`). The ~120k ambassador prefix cold-prefilled (~6 min) on
every cold start. Fix: align all instances to the same `-c` (262144). A cold
seed then restored the 120017-token pin from disk in ~15 s.
