# Disk KV cache: quick start

`llama-server` in this fork can keep conversations on disk and restore them later, in the same process
after the slot was reused, after a restart, or in another instance pointed at the same directory. A
restore replaces a prefill of the whole conversation with a read of its saved state.

Everything is off by default. Turn it on with two flags:

```sh
llama-server -m model.gguf -c 32768 \
    --slot-save-path /var/kvcache/shared \
    --slot-save-auto
```

That is the minimal setup. Add `--metrics` to see the cache's hit, miss, save and eviction counters on
`GET /metrics` (see [Seeing hits and misses](#seeing-hits-and-misses)). The sections below give the recommended command lines, what each flag does,
when the server writes a node, how to see hits and misses, and the known limits. The design notes are
in [`docs/kv-cache/`](kv-cache/README.md).

## Recommended command lines

**One model, or a pool of identical instances sharing one store** (dense, sliding-window, MoE):

```sh
llama-server -m model.gguf -c 131072 -ngl 999 -fa on --parallel 1 \
    --slot-save-path /mnt/nvme/kvcache/shared \
    --slot-save-auto --slot-save-incremental \
    --slot-save-max-mb 100000 \
    --metrics
```

**With MTP or a draft model.** Nothing changes on the cache side. Each unit also gets a `.dft` draft
sidecar so a restored conversation drafts warm, and units are interchangeable between instances with
and without speculation:

```sh
    --spec-type draft-mtp --spec-draft-n-max 3       # MTP head inside the model GGUF
    --model-draft draft.gguf --spec-draft-n-max 4    # or a separate draft model
```

**Recurrent and hybrid models** (Qwen3.6 / Qwen3.8 gated delta net, Mamba hybrids, Jamba), and models
with an indexer or compressed KV (DeepSeek-V4, GLM-5 next, Qwen3.8-Flash-Next). These cannot rewind a
saved conversation to an earlier position, so a regenerate, an edit of the last answer, or a follow-up
whose history is not re-rendered token for token can only restore a node that sits before the change.
The prompt node at the end of the last user message covers that, and it is on by default (`cold`: it is
written while a prompt that got no reuse prefills). To also write it for prompts that restored part of
their prefix, for example on every turn of a long agent loop:

```sh
    --slot-save-node-prompt on
```

Sliding-window models (Gemma 3/4, Laguna) behave the same way once a conversation is longer than the
window.

**A shared system prompt (agents, RAG, tool definitions).** On by default: the first time a cold prompt
prefills past its system prompt, that prefix is saved once and every other conversation using it
restores it. Send a request that carries only the system prompt to cache it ahead of time (a
pre-cache): the whole system prompt is saved.

```sh
    --slot-save-context-min-tokens 4096    # the smallest system prefix worth its own node (default)
```

**Context rungs** (the same model at several `-c`, e.g. 1 GPU at 131072 and 2 GPUs at 262144, or a
lower-context vision variant beside the text one): point them at the same store. `-c` is not part of a
unit's identity, so the rungs name units alike, deduplicate them and continue each other's delta chains,
and a unit restores into any rung whose context holds it, in either direction, for every model class
(sliding-window and recurrent included). They must still agree on the model file, KV cache types,
Flash Attention (`-fa` on or off changes the V layout), the KV stream count (`--parallel` without
`--kv-unified`), `--slot-save-block`, the mmproj and the RoPE/YaRN settings. The one exception is a
LongRoPE model (Phi-3 style `rope_factors_long`/`rope_factors_short`): rungs on opposite sides of its
original context use different factors, so they do not share. That threshold is the model's own
(`rope.scaling.original_context_length`, else `context_length`); `--yarn-orig-ctx` does not move it, and
a model without LongRoPE factors never splits on it.

Peers sharing a store take a lock on the directory (`flock`) while they publish a unit and while they
restore one, so a restore always loads the `.bin` its `.meta` describes even when two peers publish
different content (a whole unit and a delta) under one name.

## Flags

| Flag | Default | What it does |
|---|---|---|
| `--slot-save-path DIR` | (none) | where units are stored; point every instance of a pool at the same directory |
| `--slot-save-auto` | off | turn the automatic cache on |
| `--slot-save-incremental` | off | save a growing conversation as a delta on its previous node instead of a whole snapshot |
| `--slot-save-block N` | 256 | lookup granularity; a node is found at every multiple of `N` it covers |
| `--slot-save-min-tokens N` | 1024 | smallest conversation worth saving (effective floor `max(block, N)`); shorter prompts are never saved or looked up |
| `--slot-save-context-min-tokens N` | 4096 | smallest system prefix worth a node of its own |
| `--slot-save-max-mb N` / `--slot-save-max-count N` | 0 (unlimited) | least-recently-used eviction caps for the whole directory |
| `--slot-restore-min-tokens N` | 0 | skip a restore shorter than `N` tokens and prefill instead |
| `--slot-save-staging-mb N` | 1024 | host memory a save may hold between its copy off the device and its write by the background writer; `0` writes every save on the server thread (see below) |

### When a node is written (checkpoint triggers)

| Flag | Default | Node |
|---|---|---|
| `--slot-save-node-system` | on | at the end of the system prompt, while a cold prompt prefills; a request with only a system prompt caches all of it. The end is found by rendering the request's system messages and tools through the chat template again, followed by placeholder conversations, so it works for every template, including a default system prompt the template inserts itself |
| `--slot-save-node-prompt off\|cold\|on` | cold | at the end of the last user message, while the prompt prefills. `cold`: for prompts that got essentially no reuse, and, on a model that cannot rewind, for prompts whose reuse ended inside the previous conversation (the client did not re-render the previous response token for token, for example it dropped the reasoning) once at least one block of new prompt precedes the node. `on`: whenever at least one block of new prompt precedes it (a delta under `--slot-save-incremental`). `off`: never. When the system node is written in the same prefill, the prompt node must lie at least one block past it, or it is skipped |
| `--slot-save-node-response` | off | the conversation, as soon as each response completes |
| `--slot-save-node-tool` | off | the conversation, when a response ends in tool calls |
| `--slot-save-idle-seconds N` | 60 | the conversation, once its slot has been idle `N` seconds (`-1` disables) |
| `--slot-save-on-reclaim` | on | the conversation, before a request that does not extend it takes its slot (it diverges at least one block before the slot's end, so a different conversation sharing only a system prompt counts) |
| (shutdown) | always | every slot's conversation, on a graceful stop |

The defaults save the conversation when that is useful (idle, reclaim, shutdown) and never on every
turn: a user who keeps talking to the same instance gains nothing from a save after each response, and
the idle and reclaim saves also cover tool loops. Turn on `--slot-save-node-response` when instances
die without a graceful stop. The default prompt node is written only while a prompt that got no reuse
prefills, so at most once per conversation start, because every model class that cannot rewind needs it
for a resend, a regenerate or an edit of the last answer.

### What a resend of the same request gets

| Model class | Same request again, after its conversation was saved | Needs |
|---|---|---|
| Plain attention | restores the longer unit and trims it to the request (re-prefills about one token) | nothing |
| Sliding window, conversation within one window | same as plain attention | nothing |
| Recurrent, hybrid, indexer, compressed KV, sliding window past one window | restores the prompt node and prefills the tail after it (at most one block plus the generation prompt) | the prompt node (`--slot-save-node-prompt`, `cold` by default; `off` turns this into a reported miss) |
| any | a request that **extends** the saved conversation (the previous answer included) restores all of it | nothing |

Without a usable node a miss is reported, never silent: a WRN line and the
`auto_cache_restore_not_prefix_total` counter (below).

## What runs in the background

A save has two halves. The **capture** runs on the server thread at the moment the save is decided: the
checks (floor, dedup, parent choice, the delta probe), and the copy of the state off the device into
host memory. Every request on the instance waits for that copy, as it waited for the whole save before.
The **publish** runs on one writer thread per instance, in the order the saves were captured: it writes
the `.bin`, `.logits`, `.dft` and `.meta` temp files, syncs each one to disk (`fdatasync`), renames them
into place under the store lock with the `.meta` last, syncs the directory, inserts the index entry and
runs the LRU. So a node written while a prompt prefills, or a conversation saved when another request
takes its slot, costs that request the copy only, not the write.

- **Staging budget.** A save whose bytes fit in the free part of `--slot-save-staging-mb` is copied whole
  and the request continues at once. One that does not fit is streamed through two chunks of at most
  64 MiB when the writer is idle, so the request pays about the larger of the copy and the write; no
  save is narrowed by the budget. When the writer is busy and the save does not fit, the capture
  **waits** for the writer (logged as `auto-save: waited ... for the writer to take a ...`, counted in
  `auto_cache_save_admission_waits_total` and `_wait_seconds_total`) and then stages or streams it, so
  no save is dropped for want of staging. That wait holds the server loop, as the synchronous save it
  replaces did, and is never longer than that save would have taken: it waits only for writes the
  synchronous code would already have made on the same thread. Two exceptions. The **idle flush** is
  deferred instead (`auto-save: idle flush deferred`): nothing is queued and it is retried once the
  writer is idle, so a request that arrives meanwhile is not held. A writer that shows no progress for
  60 s (a hung disk) has the waiting save dropped with a `WRN` (`made no progress`) and
  `auto_cache_save_dropped_staging_total`, rather than the server loop hung for good. Staging is plain
  pageable host memory, allocated per save and freed as the writer drains it, so an idle instance
  holds none; at most the budget plus one 128 MiB ring is held at once, on top of `--cache-ram` and the
  GPU's GTT. Budget the worst case (`N` instances times that) against host RAM, and read
  `auto_cache_save_staging_bytes` when recording a bench's host-RAM peak.
- **Visibility.** A unit is visible to restores only once its `.meta` is in place; restores never read
  the queue. A new request whose prefix is still queued in the same instance waits for that publish
  (at most 30 s, logged as `auto-restore: waited ... for the queued unit(s) that prefix this request`)
  rather than prefilling it again. Only that request waits: it stays queued in its slot while the
  other slots keep prefilling and decoding. A peer instance cannot see the queue and prefills.
- **Deltas on queued parents.** The save-side dedup and parent choice see the queue, so a delta can be
  written against a parent that is still queued (a prompt node, then a reclaim save seconds later in a
  tool loop). The writer is FIFO, so the parent is published first. If the parent fails (disk full, a
  failed rename, a lock timeout), its queued children are dropped, because their bytes are a suffix
  only, and counted in `auto_cache_save_orphan_dropped_total`; a child whose parent left the store
  while it was queued is dropped the same way. Neither loses the conversation while a slot still holds
  it: an orphan's slot gets another idle flush (logged as `re-arming the idle flush`), which writes a
  root or a delta on a published parent, and so does every slot whose save was skipped because the
  failed unit already covered it. A reclaim (the slot is about to be overwritten) that finds its
  conversation covered by a queued delta waits for that delta's outcome and writes the conversation
  itself if it did not publish. A reclaim covered by a queued root does not wait (what fails a root
  fails a second write the same way); if that root fails, the reclaim save is logged as lost and
  shows as `requested` without `published` (below).
- **Shutdown.** A graceful stop captures every slot (waiting for staging room instead of dropping),
  then the writer drains the queue, oldest first. Whatever is still queued 90 s after the stop began is
  abandoned between chunks, its temps removed and nothing renamed, which leaves 30 s for the rest of the
  shutdown inside a 120 s `TimeoutStopSec`. The writer starts each chunk's writeback as it writes it and
  waits for the previous chunk's, so at most two chunks of a file are dirty and the final `fdatasync`
  stays short; a file whose write ends past the deadline is abandoned before that sync.
- **Crashes.** A unit whose writer died before its `.meta` rename is never published. Its temps
  (`<name>.<pid>.<n>.tmp*`) are removed by the next instance that starts on the store, or by any writer
  after a publish (at most once a minute), once that pid is gone and the files are 10 minutes old.
- **Synchronous mode.** `--slot-save-staging-mb 0` runs the publish on the server thread, as builds
  before the writer did, now with the same `fdatasync` calls.

Every published save logs one line with its timings, for example:

```
auto-save: persisted 65280 tokens to .../auto-...-65280.bin (root, 268431436 B: capture d2h 163.9 ms
  [tgt 163.9 / dft 0.0 / logits 0.0], queue-wait 0.0 ms, write 237.6 ms, fdatasync 139.4 ms,
  publish 8.8 ms, evict 0.6 ms, mode staged)
```

`capture d2h` is what the request waited for (the target state, the draft state, the logits copy);
`queue-wait` is how long the unit sat behind earlier saves; `write`, `fdatasync`, `publish` (the
renames under the store lock, and the directory sync) and `evict` (the LRU and the reconcile) ran on
the writer. `mode` is `staged`, `streamed` or `sync`.

## Seeing hits and misses

Per request, `timings` in the response says where the prompt came from:

| Field | Meaning |
|---|---|
| `cache_n` | prompt tokens not prefilled |
| `cache_disk_n` | of those, restored from disk (absent when zero) |
| `cache_ram_n` | of those, loaded from the RAM prompt cache (`--cache-ram`, absent when zero) |
| `prompt_n` | prompt tokens prefilled |

`usage.prompt_tokens_details.cached_tokens` in the OpenAI-style responses equals `cache_n`. With
`--metrics`, `GET /metrics` carries cumulative counters (prefix `llamacpp:`):

| Counter | Meaning |
|---|---|
| `auto_cache_restore_hit_total` | requests that restored a prefix from disk and kept it |
| `auto_cache_restore_miss_total` | requests with at least one block beyond the in-memory match that restored nothing |
| `auto_cache_restore_not_prefix_total` | misses where a saved unit shared the prefix but the model cannot rewind into it (see the table above) |
| `auto_cache_restore_discarded_total` | restores whose tokens were thrown away before use (counted as misses) |
| `auto_cache_restore_failed_total` | restores whose load failed (fell back to a shorter unit or a cold prefill) |
| `auto_cache_restore_tokens_total` | prompt tokens restored from disk |
| `auto_cache_save_root_total` / `auto_cache_save_delta_total` | whole snapshots / delta nodes written |
| `auto_cache_save_whole_fallback_total` | saves that would have been deltas, written whole because the memory type cannot write deltas (included in the root count) |
| `auto_cache_save_bytes_total` | state bytes written by published saves |
| `auto_cache_save_failed_total` | saves dropped with nothing written (each logs a WRN with the reason) |
| `auto_cache_evicted_total` | units this instance evicted to stay under the caps |
| `auto_cache_save_draft_total` / `auto_cache_save_draft_skipped_total` | `.dft` draft sidecars written / units published without one although a draft context exists |
| `auto_cache_restore_draft_{warm,cold}_total` | restores whose draft came back warm / cold |
| `auto_cache_sysnode_probed_total` / `auto_cache_sysnode_probe_renders_total` | chat requests whose system-prompt end was looked up / of those, not already cached |
| `auto_cache_sysnode_probe_failed_total` | requests whose template rendered none of the boundary probes (the message delimiters place the node) |
| `auto_cache_sysnode_seam_mismatch_total` | requests whose system-prompt tokens were not a prefix of the prompt (no system node) |
| `auto_cache_sysnode_probe_short_total` | chat prompts too short in bytes to reach the system node's floor, so not probed at all |
| `auto_cache_node_media_skipped_total` | system or prompt nodes not written because media left no cut above the floor (see Known limits) |
| `auto_cache_save_queued_total` | saves handed to the background writer |
| `auto_cache_save_streamed_total` | of those, saves larger than the free staging, streamed while the writer was idle |
| `auto_cache_save_admission_waits_total` / `auto_cache_save_admission_wait_seconds_total` | captures that waited for a busy writer because the staging was full / total time waited |
| `auto_cache_save_dropped_staging_total` | saves dropped because the writer made no progress for 60 s while the capture waited (each logs a WRN); 0 in normal operation |
| `auto_cache_save_orphan_dropped_total` | queued deltas dropped because their parent failed to publish or left the store |
| `auto_cache_save_shutdown_abandoned_total` | queued saves abandoned at the shutdown deadline |
| `auto_cache_save_staging_bytes` (gauge) | host bytes held by saves copied and not yet written |
| `auto_cache_save_queue_depth` (gauge) | saves queued or being written |
| `auto_cache_save_site_requested_total{site=...}` / `auto_cache_save_site_published_total{site=...}` | units each save site decided to write / of those, published. `site` is `reclaim`, `idle`, `shutdown`, `cache_idle`, `system_node`, `prompt_node` or `response_node`. Requested minus published is what that site lost (failed, orphaned, abandoned or dropped); a deferred idle flush is counted once, when it is taken |

A miss includes conversations no cache could have held, so read it next to `auto_cache_evicted_total`:
misses that climb with evictions mean the store is too small.

```sh
curl -s localhost:8080/metrics | grep auto_cache_
curl -s localhost:8080/completion -d '{"prompt":"...","n_predict":16,"cache_prompt":true}' | jq .timings
```

## Known limits

- Prompts shorter than `max(--slot-save-block, --slot-save-min-tokens)` are never saved or looked up.
- The system and prompt nodes are written for prompts with images or audio too. A node is cut where the
  previous cell is text, so no media chunk is split: a position inside or right after a chunk moves down
  to the chunk's start. When that falls below the floor the node is skipped and counted in
  `auto_cache_node_media_skipped_total`.
- Every unit is named and deduplicated by its identity over all of its tokens, so two system prompts of
  the same length that differ only near the end (a date, a user name) get a node each. Index lookups
  still go by whole blocks, and every restore byte-compares the tokens.
- Both nodes are captured while a cold prompt prefills, on the server-loop thread: each delays that
  request's first token by the copy of its state off the device (a whole snapshot, or a delta under
  `--slot-save-incremental`) and holds up other slots meanwhile; the write itself runs on the background
  writer, unless the staging is full and the writer busy, when the capture also waits for the writer
  (see "Staging budget"). The default `cold` prompt node adds at most one capture per conversation start, plus one
  per turn on a model that cannot rewind when the client does not re-render the previous response token
  for token (each such turn's after-response unit cannot serve the next turn, so the node is the only
  restore point that survives a reclaim or a restart).
- The copy off the device stays on the request's critical path. It lands in ordinary (pageable) host
  memory; staging in pinned memory may copy faster on a GPU and is not measured yet. Moving the copy
  itself off the first token needs the cells to be copied lazily after the node, which is not done.
- The system node is placed from the chat template for every template (the boundary is checked over
  all of `models/templates` by `test-chat-preamble`). The first chat request with a new system prompt
  or tool set pays for the template renders that find it: about four times one render of the request,
  measured 5 ms with no tools and 55-260 ms with 50 tools on a Threadripper 1950X (600 ms on Inkling's
  template); later requests with the same system prompt and tools reuse the result. A prompt with fewer
  bytes than the node floor in tokens cannot reach it and is not probed. A system prompt that changes on
  every request (a time stamp) pays the renders every time and never gets a reusable node. A raw
  `/completion` prompt has no messages: it gets a system node only when the client sends
  `message_delimiters` (the first user message) or `preamble_end_chars` (the preamble's length in
  characters, for a string prompt).
- The prompt node still depends on the chat template's message delimiters (the end of the last user
  message). Where they do not match, as on Laguna, whose `<user>` header is plain text that byte-level
  BPE merges into the message, it falls back to one block before the end of the prompt.
- A system prompt shorter than `max(--slot-save-block, --slot-save-context-min-tokens)` (4096 tokens
  by default) gets no system node; a typical 1-2k token chat-UI system prompt is below it.
- A template that moves the system prompt into a later turn (Mistral-Nemo puts it in the last user
  message) has no stable system prefix; no system node is written for it.
- The prompt node's floor is `max(--slot-save-block, --slot-save-min-tokens)`. Builds before
  merge-upstream-20261003 armed a node near the end of a cold prompt for sliding-window models only,
  with the floor `max(--slot-save-block, --slot-save-context-min-tokens)`; the default `cold` prompt
  node now does this for every class that cannot rewind, at the lower floor.
- On a rollback to a release older than the `.dft` draft sidecars, purge `*.dft` from the store first;
  an older binary counts them as units, evicts them and can delete a newer binary's `.tmp.dft` temps.
  Never run the two on one store at the same time.
- A unit whose length is not a whole number of blocks is named by its identity over every token since
  merge-upstream-20261003; older builds named it by its last whole block. Restores still work across
  the change (the index is rebuilt from the `.meta` files), but an older binary that saves a delta onto
  a newer unit names a parent that does not exist, so that delta is dead weight. Purge the store when
  moving between the two in either direction.
- Every instance writing to one directory must agree on the model, `--cache-type-k/v`,
  `--slot-save-block`, the mmproj and the RoPE settings (`-c` may differ, see the rung rule above); a
  unit from a mismatched instance is refused, never restored into the wrong context.
- Since `-c` left the unit identity, every unit name has a new prefix. Units written before the change
  still index and restore, chains included (a delta's parent is resolved under the tip's own prefix),
  but no new delta is linked onto them, so they age out. Purging the store at deploy is the clean
  option, as for any naming change.
- Restore needs the prefix to be token-identical. A date or a counter rendered into the system prompt
  changes it every turn and defeats the cache.
- One store, one cap: `--slot-save-max-mb` is enforced over the whole directory by whichever instance
  saves, so give every pool sharing it the same value.
