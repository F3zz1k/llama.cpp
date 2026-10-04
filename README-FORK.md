# llama.cpp — recurrent-model & mmproj KV caching fork

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The
`main-patched` branch is **upstream `master` plus a small set of llama-server patches**
that make disk KV caching work for model types where it was previously broken or blocked.

**Upstream baseline:** `main-patched` currently carries upstream `master` at `d7bd3bfca`
(2026-08-28), absorbed by the merge of 2026-08-29. The previous baseline was `9d57ce456`
(2026-08-14). That merge crossed an upstream **state-file format bump**
(`LLAMA_STATE_SEQ_VERSION` 2 -> 3), so KV snapshots written by any earlier build of this
fork are refused by this one; see
[`docs/kv-cache/README.md`](docs/kv-cache/README.md#engine-state-file-format-bin-and-upstream-version-bumps)
for who is affected and what an operator has to do at deploy time.

Everything else is stock llama.cpp — see the upstream [README.md](README.md) to build and
run normally. This document only covers what the fork adds.

**Using the disk KV cache:** [`docs/disk-cache.md`](docs/disk-cache.md) is the quick start:
copy-paste `llama-server` command lines (minimal, a pool sharing one store, with MTP or a draft
model, recurrent and hybrid models, a shared system prompt, context rungs), every flag with its
default, when a node is written, how to read hits and misses in the response, and the known limits.

---

## What this fork is for

Out of the box, llama-server can save a conversation's KV cache to disk and load it back,
so you don't re-process a long prompt every time. But two important cases didn't work:

1. **Recurrent / hybrid models** (Mamba-style and "gated-delta" hybrids). Saving and
   restoring "worked" but the model then re-processed the whole prompt anyway, and asking
   it to regenerate the *same* prompt could **crash the server**.
2. **Multimodal servers** (started with `--mmproj` for image input). KV caching was turned
   off entirely — even for plain **text** turns that contain no images.

This fork fixes both, and adds an **opt-in automatic disk cache** so a plain chat client
gets cross-request and cross-process reuse with no extra work: when a long prompt arrives
on a "cold" server (a fresh start, or a different instance), it is restored from disk in a
fraction of a second instead of being re-processed for minutes. The automatic cache is
**fully multimodal**: prompts containing images (or audio) are saved and restored too, with
each media chunk verified by a content hash of its raw uploaded bytes before any reuse.

### Why it matters (the practical payoff)

Re-processing a large prompt ("prefill") is the slow part. On a deep context (100k+ tokens)
it can take **minutes** every time a server starts cold or a request lands on a different
instance. With this fork that becomes a **sub-second disk restore**. The bigger your
context and the more instances you run, the bigger the win.

---

## Models this enables

The recurrent/hybrid fixes apply to any model llama.cpp treats as "FULL" memory
(recurrent, hybrid, or SWA). Concretely, this fork makes disk KV caching usable for, e.g.:

- **Qwen3.6** family (e.g. Qwen3.6-27B, Qwen3.6-35B-A3B) — gated-delta hybrid
- **Qwen3-Next** — hybrid
- **Mamba / Mamba-2** based models
- **Falcon-H**, **Jamba**, and other hybrid SSM/attention models
- Any model that previously logged "cache reuse is not supported" or forced full
  re-processing on every turn

Plain **attention** models (Llama, Mistral, Qwen2.5, Gemma, etc.) already worked with disk
caching upstream; this fork doesn't change their behavior, and the auto-cache works for
them too.

**Multimodal:** a server started with `--mmproj` now caches **everything** — text-only
turns exactly as a text-only server would (byte-identical snapshot format), and turns
containing images or audio as well. Media snapshots store only ~100 bytes of *identity*
per media chunk (a content hash of the raw uploaded bytes plus the chunk's shape); the
embeddings are already inside the saved KV state, and at restore time the request itself
carries the live media. Reuse requires the client to re-upload each media file
byte-identically — which is what normal multi-turn chat clients do. Same text with a
*different* image reuses exactly the prefix before that image. See
`docs/kv-cache/03-multimodal-cache.md`.

---

## TODO / not done yet

- **Code cleanup.** The auto-cache logic lives inline in the large `server-context.cpp`;
  it should be extracted into its own translation unit. The internal index mutex is
  currently uncontended (single-threaded) and only matters if saving is later threaded.
- **A CI-sized M-RoPE vision fixture.** The pytest suite covers every cache code path with
  tinygemma3 (a normal-position vision model); M-RoPE models (Qwen-VL class) are validated
  on real hardware only. A tiny qwen2-vl GGUF would close that gap in CI.

Full design write-ups (what changed, how it works, and why): see
[`docs/kv-cache/`](docs/kv-cache/).

---

## The new command-line flags

All flags are **off by default**. With none of them set, llama-server behaves exactly like
upstream.

| Flag | Default | What it does |
|------|---------|--------------|
| `--slot-save-path PATH` | (off) | Directory to store KV snapshots. *(Upstream flag — required by everything below.)* |
| `--slot-save-auto` | off | Turn on the **automatic** disk cache: the server saves/restores KV by itself, transparently, for every request. Requires `--slot-save-path`. |
| `--slot-save-block N` | 256 | Reuse granularity, in tokens. A prompt can be reused up to the nearest multiple of `N`. Smaller = finer reuse but more index entries. Leave at default unless you know you need otherwise. |
| `--slot-save-min-tokens N` | 1024 | Don't cache a prefix shorter than this — a tiny snapshot isn't worth its write and restore cost. The effective floor is `max(--slot-save-block, N)`. No effect without `--slot-save-auto`. |
| `--slot-save-idle-seconds N` | 60 | Also flush an idle slot to disk after `N` seconds of inactivity, not only when the slot is reused — so a single request survives a restart or is picked up by another instance without waiting for more traffic. `-1` disables. Requires `--slot-save-auto`. |
| `--slot-save-max-count N` | 0 (unlimited) | Bound the **`--slot-save-auto` cache** to at most `N` snapshots; oldest are deleted first. `0` = unlimited. No effect without `--slot-save-auto`. |
| `--slot-save-max-mb N` | 0 (unlimited) | Bound the **`--slot-save-auto` cache** to `N` MiB total; oldest deleted first. `0` = unlimited. A single snapshot larger than this is refused (not allowed to wipe the rest). No effect without `--slot-save-auto`. |
| `--slot-save-incremental` | off | Save a growing conversation as a small delta node chained to its previous snapshot instead of a whole snapshot each time. Every memory type except pure recurrent supports it (see `docs/kv-cache/incremental-disk-cache.md`). Requires `--slot-save-auto`. |
| `--slot-save-context-min-tokens N` | 4096 | Whole-save a shared preamble (system prompt, tools, RAG) once, mid-prefill, as a base that later chats restore. The floor is `max(--slot-save-block, N)`. |
| `--slot-restore-min-tokens N` | 0 | Skip a disk restore whose verified prefix is shorter than `N` tokens and re-prefill instead. `0` always restores. |
| `--slot-save-node-system` | on | Node at the end of the system prompt (with tools and the template's own preamble, found by rendering the chat template, for every template) while a cold prompt prefills; a request carrying only a system prompt caches all of it. `--no-slot-save-node-system` turns it off. |
| `--slot-save-node-prompt off\|cold\|on` | cold | Node at the end of the last user message while the prompt prefills, so a resend, regenerate or edit restores it on models that cannot rewind (recurrent, hybrid, sliding window past one window). `cold`: only for prompts with no reuse; `on`: whenever a block of new prompt precedes it; `off`: never. |
| `--slot-save-node-response` | off | Save the conversation as soon as each response completes, not only on idle, reclaim and shutdown. |
| `--slot-save-node-tool` | off | Same, only for responses that end in tool calls. |
| `--slot-save-on-reclaim` | on | Save a slot's conversation before a request from a different conversation takes the slot. `--no-slot-save-on-reclaim` turns it off. |

> **Eviction is opt-in.** Plain `--slot-save-path` (manual `/slots` save, upstream behaviour)
> never deletes anything. The bounded LRU store only runs when `--slot-save-auto` owns the
> directory as its cache.
>
> **Disk note:** one deep snapshot can be several GB (a 158k-token snapshot ≈ 8 GB). Point
> `--slot-save-path` at a **dedicated directory on a roomy disk**, and size
> `--slot-save-max-mb` to your budget. With `--slot-save-auto` + a cap set, the server treats
> that directory as its own — don't put other files there.

---

## How to use it

### Simplest: automatic cache, single server

```bash
mkdir -p ~/kvcache/mymodel

./build/bin/llama-server \
  -m /path/to/Qwen3.6-27B-Q5_K_M.gguf \
  -c 262144 -ngl 999 -fa on \
  --slot-save-path ~/kvcache/mymodel \
  --slot-save-auto \
  --slot-save-max-mb 40960          # 40 GiB budget for this model's snapshots
```

That's it. Clients talk to the normal OpenAI `/v1/chat/completions` endpoint. The first
time a long prompt is seen it's processed normally and saved; later, the same prompt (or a
longer conversation that starts with it) is restored from disk instead of re-processed —
even after you restart the server.

On startup you'll see a log line confirming it's on:

```
auto disk prompt cache enabled: indexed 3 prefix boundaries from /home/you/kvcache/mymodel/ (block=256)
```

### Multimodal (image-capable) server

Just add the auto-cache flags to your normal `--mmproj` command line — nothing special:

```bash
./build/bin/llama-server \
  -m /path/to/Qwen3.6-27B-Q5_K_M.gguf \
  --mmproj /path/to/mmproj-F16.gguf \
  -c 262144 -ngl 999 -fa on \
  --slot-save-path ~/kvcache/mymodel \
  --slot-save-auto
```

Text-only turns are cached in the exact same snapshot format a text-only server writes.
Turns containing images (or audio) are cached too: on a resend of the same conversation
(same text, byte-identical media re-upload) the whole prompt restores from disk; a resend
with a *different* image reuses the prefix before that image and re-processes the rest.
Swapping the `--mmproj` file invalidates media snapshots (each records a fingerprint of
the projector it was encoded with, shown in `/props` as `fp_mmproj`) while text snapshots
keep working.

### Manual save/restore (advanced, no `--slot-save-auto`)

The original `/slots` endpoints still work and now behave correctly for recurrent models.
With just `--slot-save-path` set (no `--slot-save-auto`):

```bash
# save slot 0 to <slot-save-path>/snap1.bin
curl http://localhost:8080/slots/0?action=save  -d '{"filename":"snap1.bin"}'
# restore it later (e.g. after a restart)
curl http://localhost:8080/slots/0?action=restore -d '{"filename":"snap1.bin"}'
```

They use upstream's implementation and file format, media slots included (upstream packs each
image's id and geometry into the state file). The fork adds a `.logits` sidecar on recurrent and
hybrid models, so resending exactly the saved prompt after a restore does not re-process it,
refuses saves to `auto-*` names (reserved for the automatic cache), and restores any file that has a
`.meta` sidecar (an automatic cache unit, or a media save from a fork build before 2026-10-03) the
fork's way. Upstream's format records no model or projector identity, so the fork writes a small `.fp`
sidecar beside each save, and a restore refuses a file whose `.fp` names another model, or, for a
media save, another `--mmproj` (a file without one, saved by upstream, is restored unchecked). A save is
written to a temp name and renamed over the old file, so a failed save leaves the previous one intact.
After a manual restore the speculative draft (MTP) starts cold: only the automatic cache saves draft
state. Details: `docs/kv-cache/03-multimodal-cache.md`.

### Pinning a snapshot (permanent, never-evicted cache)

*(From the `auto-disk-kvcache-pin` branch; included in `main-patched`.)*

When a cap is set, the auto-cache evicts least-recently-used snapshots once `--slot-save-max-count` /
`--slot-save-max-mb` are exceeded. To keep one snapshot **forever** — e.g. a large fixed
documentation / system-prompt prefix that every request should reuse — drop a `.pin` marker next
to its state file:

```bash
# pin: this snapshot is now never evicted and no longer counts against the caps
touch <slot-save-path>/auto-<fingerprint>-<hash>-<n>.bin.pin
# unpin: it rejoins the normal LRU pool
rm    <slot-save-path>/auto-<fingerprint>-<hash>-<n>.bin.pin
```

A pinned snapshot is otherwise a normal snapshot — still discovered and restored exactly like any
other (including across processes), so a fresh/cold instance still warms it from disk in a fraction
of a second. This lets a permanent prefix live **inside the shared cache pool** without dedicating
an instance to it. (The marker is a plain file; no flag or restart needed.)

---

## Branches

- **`master`** — mirror of upstream `master`; every patch is cut from here.
- **`kv-restore-reuse`** — recurrent/hybrid restore primitives (regenerate-from-logits + reusing a disk-restored slot).
- **`auto-disk-kvcache`** — the above + the opt-in automatic cross-process disk cache (`--slot-save-auto`).
- **`auto-disk-kvcache-pin`** — `auto-disk-kvcache` + the `.pin` eviction-exempt marker.
- **`auto-disk-kvcache-mm`** — `auto-disk-kvcache` + full multimodal snapshots (image/audio prompts cached and verified by media identity records) + the per-slot manual `/slots` gate + the test suite.
- **`l0-fattn-alloc`** — independent SYCL fix: route the flash-attention KV buffer through the Level-Zero device allocator so it isn't mirrored into host RAM under multi-GPU / P2P.
- **`main-patched`** — the deployed integration: `master` + all the above merged.

Each feature branch is a clean single-purpose delta, meant to be submittable upstream as its own PR (`kv-restore-reuse` and `l0-fattn-alloc` sit directly on `master`; `auto-disk-kvcache` stacks on `kv-restore-reuse`, and the `-pin` / `-mm` branches stack on `auto-disk-kvcache`).

## How to keep the fork up to date with upstream

Bump `master` to the new upstream, refresh each feature branch onto it, then rebuild `main-patched` as their merge:

```bash
git remote add upstream https://github.com/ggml-org/llama.cpp.git   # one-time
git fetch upstream master && git branch -f master upstream/master
# rebase each feature branch onto the new master (resolving conflicts — see below), then:
git checkout -B main-patched master
git merge --no-ff auto-disk-kvcache-pin
git merge --no-ff auto-disk-kvcache-mm
git merge --no-ff l0-fattn-alloc
git push --force-with-lease origin master main-patched kv-restore-reuse auto-disk-kvcache auto-disk-kvcache-pin auto-disk-kvcache-mm l0-fattn-alloc
```

Merge the branches **sequentially, in that order** — never as one multi-branch (octopus)
merge, which cannot resolve any conflict. The `-pin` -> `-mm` merge always conflicts
(**add/add**) on `README-FORK.md` and `docs/kv-cache/02-auto-disk-cache.md`: both branches
carry these files with different content. Take the `-mm` copies, which document the `.pin`
feature too — `git checkout --theirs README-FORK.md docs/kv-cache/02-auto-disk-cache.md`,
then `git add` both and `git commit` to conclude the merge.
(`docs/kv-cache/01-primitives-recurrent-restore.md` is identical on both branches and
resolves itself, and the two branches' `tools/server/server-context.cpp` edits touch
different regions and auto-merge.)

Conflicts against **upstream** land almost entirely in `tools/server/server-context.cpp`
and are **not** trivial: upstream's server refactors relocate code, and a 3-way merge can
silently mis-place a small hunk into the wrong decode loop (this has caused a segfault).
Always build **and** exercise the disk save->restart->restore path afterward.
`docs/kv-cache/` explains what each commit touches.

---

## How to build and test

### Build (same as upstream)

```bash
cmake -B build -DGGML_NATIVE=ON          # add your backend, e.g. -DGGML_CUDA=ON / -DGGML_SYCL=ON
cmake --build build --target llama-server -j
```

### Quick built-in checks

The fork adds unit tests for its pure helpers — the `.meta` sidecar parser (including a
fuzz mode over mutated inputs), the media-aware block chain hashing, and the
`server_tokens` cell accessors — which run as part of the standard suite:

```bash
ctest --test-dir build --output-on-failure     # unit tests incl. test-slot-meta / test-auto-hash / test-server-tokens
./build/bin/llama-server --help | grep slot-save   # confirms the new flags are present
```

The server-level behaviour (save/restore across restarts and processes, multimodal reuse
and mismatch truncation, torn/corrupt-unit fallback, manual `/slots`) is covered by
pytest — see `tools/server/tests`:

```bash
cd tools/server/tests
./tests.sh unit/test_slot_save_auto.py
```

### End-to-end test: 2 live instances (save on one, restore on the other)

This proves the headline feature — a snapshot written by one server is restored by a
**second, cold** server sharing the same directory. Use a recurrent model (e.g. Qwen3.6).

```bash
DIR=~/kvcache/test ; mkdir -p $DIR

# 1) Start instance A on port 8081
./build/bin/llama-server -m MODEL.gguf -ngl 999 -c 32768 -fa on \
  --slot-save-path $DIR --slot-save-auto --port 8081 &

# 2) Send a long-ish prompt to A (over --slot-save-min-tokens, default 1024, so it's worth
#    caching), then a different
#    prompt so A's slot is reused and the first one gets saved to disk.
curl -s http://localhost:8081/completion \
  -d '{"prompt":"<a few hundred tokens of context here ...>","n_predict":1}' >/dev/null
curl -s http://localhost:8081/completion \
  -d '{"prompt":"unrelated short prompt","n_predict":1}' >/dev/null

# 3) Confirm a snapshot was written
ls -lh $DIR/auto-*.bin          # expect at least one .bin (+ .meta, +.logits)

# 4) Start instance B on port 8082 — COLD, but pointed at the SAME directory
./build/bin/llama-server -m MODEL.gguf -ngl 999 -c 32768 -fa on \
  --slot-save-path $DIR --slot-save-auto --port 8082 &

# 5) Send the SAME long prompt + a new question to B. It should restore from disk
#    instead of re-processing the whole prompt.
curl -s http://localhost:8082/completion \
  -d '{"prompt":"<the same context ...> Now answer this new question.","n_predict":20}'
```

**What success looks like:** in B's log you'll see
`auto-restore: reused N tokens from disk ...`, and the response's `timings.prompt_n` (the
number of tokens actually processed) is small — only the new part of the prompt — instead
of the full length. On instance A the same prompt would have shown `prompt_n` equal to the
whole prompt.

> Recurrent models reuse a snapshot only when the new request **extends** it (the snapshot's
> tokens are the start of the new prompt). This is exactly how a normal multi-turn chat grows,
> so it "just works" for conversations; it's only a limitation if you send a *shorter* prompt
> than what was saved.

---

## Where to read more

- [`docs/kv-cache/01-primitives-recurrent-restore.md`](docs/kv-cache/01-primitives-recurrent-restore.md) — the recurrent-model restore/regenerate fixes
- [`docs/kv-cache/02-auto-disk-cache.md`](docs/kv-cache/02-auto-disk-cache.md) — the automatic disk cache (indexing, fingerprinting, cross-process)
- [`docs/kv-cache/03-multimodal-cache.md`](docs/kv-cache/03-multimodal-cache.md) — multimodal snapshots (media identity records, the v2 `.meta` format, verification order, manual `/slots` rehydration)
- [`docs/kv-cache/README.md`](docs/kv-cache/README.md): **recommended configurations** (pool, speculative decoding, context rungs), the `.dft` draft sidecar, and the `/metrics` counters for cache hits, misses, saves and evictions
- the "Automatic disk prompt cache" section of [`tools/server/README.md`](tools/server/README.md) — user-facing invariants, restore semantics and operational notes

---

## Retired carries (dropped in the 2026-10 upstream merge)

Two SYCL commits used to be carried on top of upstream. Both were dropped when upstream
`836d57176` was merged, and `ggml/src/ggml-sycl/ggml-sycl.cpp` is now upstream's file. Neither
replacement has been run on GPU yet, so each has a check to pass before it is trusted.

### SYCL tensor-parallel VMM hang (`cd1dd9d76`)

The carried fix moved the two `--split-mode tensor` all-reduce scratch buffers off the VMM pool
(`ggml_sycl_malloc_device` instead of `ggml_sycl_pool_alloc`), because the BF16 large path
peer-copied from a VMM pool buffer on the other device, and once that buffer spanned more than two
2 MiB pages the peer copy never signalled and decode hung (2x Arc Pro B70: `-ub 200` ok, `-ub 208`
hung; upstream issues #26409 and #25711).

Upstream has **not** changed the allocation: `ggml_backend_sycl_comm_init` still builds `buf0` /
`buf1` from `sctx0->pool()` / `sctx1->pool()`, so the removal test this section used to state does
not fire. What changed is the copy. Upstream `2a53ace3b` (#29604) stages every all-reduce exchange
through pinned host memory (`ggml_sycl_comm_exchange`: device to host `memcpy` on each queue, a host
`memcpy` between the two, host to device back), so nothing peer-reads VMM pool memory any more. The
carried commit's own measurements cover that combination:

```
VMM pool + peer copy        -> hangs
VMM pool + host-staged copy -> works  (GGML_SYCL_DEV2DEV_MEMCPY=2)
```

That makes the drop sound in principle, but the device to host copies still read from those
multi-page VMM pool buffers, and that exact upstream code has not run on our hardware. **Before
using `-sm tensor` again**, run on a drained pair of cards, with
`UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` so a lost context raises an error instead of hanging:

```sh
llama-server -m Qwen3.8-27B.gguf -ngl 999 -fa on -sm tensor --device SYCL1,SYCL2 \
    -c 16384 -ub 256 --parallel 1
# send a prompt of a few thousand tokens and generate; -ub 256 makes nelem*4 = 5 MiB on n_embd 5120,
# past the old 4 MiB cut-off
```

If it hangs, re-carry `cd1dd9d76`'s dedicated `ggml_sycl_malloc_device` buffers on top of
upstream's host exchange. Nothing in the registry uses `-sm tensor` today, so this is dormant.

### SYCL `GGML_OP_TOP_K` for k > 32 (`9dc0e2f12`)

The carried commit let ggml-sycl run `TOP_K` on the GPU for k > 32 (needed by the Qwen3.8-Flash-Next
sparse-attention indexer). Upstream `21f6b0d22` (#28670, radix select) and `370cb12e8` (#27847, row
splitting) replace it, and upstream's `supports_op` now accepts any `k <= ne0`, including a single
long row. That moves deep-context single-row decode of the indexer from the CPU fallback onto the GPU.
**Before promoting a build**, A/B prefill and decode on `qwen3.8-flash-next` against the current
release, and gate on `test-backend-ops -o TOP_K`.
