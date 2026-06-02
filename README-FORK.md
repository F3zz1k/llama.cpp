# llama.cpp — recurrent-model & mmproj KV caching fork

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The
`main-patched` branch is **upstream `master` plus a small set of llama-server patches**
that make disk KV caching work for model types where it was previously broken or blocked.

Everything else is stock llama.cpp — see the upstream [README.md](README.md) to build and
run normally. This document only covers what the fork adds.

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
fraction of a second instead of being re-processed for minutes.

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

**Multimodal:** a server started with `--mmproj` now caches the **text-only** turns of a
session. A turn that includes an image is skipped (and so is the rest of that session),
because image content can't be safely identified by tokens alone — see the TODO below.

---

## TODO / not done yet

- **Full image-token caching.** Today an image in the prompt disables caching for that
  turn onward. Supporting image prefixes needs the image content folded into the cache key
  (see `docs/kv-cache/02-auto-disk-cache.md`, "Multimodal").
- **Code cleanup.** The auto-cache logic lives inline in the large `server-context.cpp`;
  it should be extracted into its own translation unit. The internal index mutex is
  currently uncontended (single-threaded) and only matters if saving is later threaded.
- **Automated tests.** Validation so far is end-to-end on real models; the pure helpers
  (hashing, fingerprint, file format) should get unit tests.

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
| `--slot-save-max-count N` | 64 | Keep at most `N` snapshots in the directory; oldest are deleted first. `0` = unlimited. |
| `--slot-save-max-mb N` | 32768 | Keep the snapshot directory under `N` MiB total; oldest deleted first. `0` = unlimited. A single snapshot larger than this is refused (not allowed to wipe the rest). |

> **Disk note:** one deep snapshot can be several GB (a 158k-token snapshot ≈ 8 GB). Point
> `--slot-save-path` at a **dedicated directory on a roomy disk**, and size
> `--slot-save-max-mb` to your budget. With a cap set, the server treats that directory as
> its own — don't put other files there.

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

### Multimodal (image-capable) server, caching text turns

Just add the auto-cache flags to your normal `--mmproj` command line — nothing special:

```bash
./build/bin/llama-server \
  -m /path/to/Qwen3.6-27B-Q5_K_M.gguf \
  --mmproj /path/to/mmproj-F16.gguf \
  -c 262144 -ngl 999 -fa on \
  --slot-save-path ~/kvcache/mymodel \
  --slot-save-auto
```

Text-only turns are cached; turns containing an image are skipped automatically.

### Manual save/restore (advanced, no `--slot-save-auto`)

The original `/slots` endpoints still work and now behave correctly for recurrent models.
With just `--slot-save-path` set (no `--slot-save-auto`):

```bash
# save slot 0 to <slot-save-path>/snap1.bin
curl http://localhost:8080/slots/0?action=save  -d '{"filename":"snap1.bin"}'
# restore it later (e.g. after a restart)
curl http://localhost:8080/slots/0?action=restore -d '{"filename":"snap1.bin"}'
```

---

## How to keep the fork up to date with upstream

`main-patched` = upstream `master` + 3 patch commits. To pull in new upstream changes:

```bash
# one-time: add the upstream remote
git remote add upstream https://github.com/ggml-org/llama.cpp.git

# update
git fetch upstream master
git checkout main-patched
git rebase upstream/master        # replays our 3 commits onto the latest upstream
# resolve any conflicts (usually only in tools/server/server-context.cpp), then:
git push --force-with-lease origin main-patched
```

If a rebase conflict looks scary, the 3 commits are small and self-contained — the design
docs in `docs/kv-cache/` explain exactly what each one touches.

---

## How to build and test

### Build (same as upstream)

```bash
cmake -B build -DGGML_NATIVE=ON          # add your backend, e.g. -DGGML_CUDA=ON / -DGGML_SYCL=ON
cmake --build build --target llama-server -j
```

### Quick built-in checks

The patches don't add a custom test target, but the standard suite should pass and is the
fastest way to confirm nothing regressed:

```bash
ctest --test-dir build --output-on-failure     # runs llama.cpp's unit tests
./build/bin/llama-server --help | grep slot-save   # confirms the new flags are present
```

### End-to-end test: 2 live instances (save on one, restore on the other)

This proves the headline feature — a snapshot written by one server is restored by a
**second, cold** server sharing the same directory. Use a recurrent model (e.g. Qwen3.6).

```bash
DIR=~/kvcache/test ; mkdir -p $DIR

# 1) Start instance A on port 8081
./build/bin/llama-server -m MODEL.gguf -ngl 999 -c 32768 -fa on \
  --slot-save-path $DIR --slot-save-auto --port 8081 &

# 2) Send a long-ish prompt to A (>256 tokens so it's worth caching), then a different
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
- [`docs/kv-cache/02-auto-disk-cache.md`](docs/kv-cache/02-auto-disk-cache.md) — the automatic disk cache (indexing, fingerprinting, cross-process, multimodal)
