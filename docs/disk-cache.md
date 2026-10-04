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

That is the minimal setup. The sections below give the recommended command lines, what each flag does,
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

**Context rungs** (the same model at several `-c`, e.g. 1 GPU at 131072 and 2 GPUs at 262144): point
them at the same store. A conversation restores from a smaller rung into a larger one when every rung
has the same model file, KV cache types, `--slot-save-block` and RoPE/YaRN settings. A larger snapshot
never restores into a smaller context, and sliding-window models only restore at an identical `-c`.

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

### When a node is written (checkpoint triggers)

| Flag | Default | Node |
|---|---|---|
| `--slot-save-node-system` | on | at the end of the system prompt, while a cold prompt prefills; a request with only a system prompt caches all of it. The end is found by rendering the request's system messages and tools through the chat template again, followed by placeholder conversations, so it works for every template, including a default system prompt the template inserts itself |
| `--slot-save-node-prompt off\|cold\|on` | cold | at the end of the last user message, while the prompt prefills. `cold`: only for prompts that got essentially no reuse. `on`: whenever at least one block of new prompt precedes it (a delta under `--slot-save-incremental`). `off`: never |
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
| `auto_cache_save_failed_total` | saves dropped with nothing written (each logs a WRN with the reason) |
| `auto_cache_evicted_total` | units this instance evicted to stay under the caps |
| `auto_cache_restore_draft_{warm,cold}_total` | restores whose draft came back warm / cold |
| `auto_cache_sysnode_probed_total` / `auto_cache_sysnode_probe_renders_total` | chat requests whose system-prompt end was looked up / of those, not already cached |
| `auto_cache_sysnode_probe_failed_total` | requests whose template rendered none of the boundary probes (the message delimiters place the node) |
| `auto_cache_sysnode_seam_mismatch_total` | requests whose system-prompt tokens were not a prefix of the prompt (no system node) |

A miss includes conversations no cache could have held, so read it next to `auto_cache_evicted_total`:
misses that climb with evictions mean the store is too small.

```sh
curl -s localhost:8080/metrics | grep auto_cache_
curl -s localhost:8080/completion -d '{"prompt":"...","n_predict":16,"cache_prompt":true}' | jq .timings
```

## Known limits

- Prompts shorter than `max(--slot-save-block, --slot-save-min-tokens)` are never saved or looked up.
- The system and prompt nodes are written for text prompts only; conversations with images or audio are
  saved on idle, reclaim, shutdown and after responses.
- The system node is placed from the chat template for every template (the boundary is checked over
  all of `models/templates` by `test-chat-preamble`). The first chat request with a new system prompt
  or tool set pays for the template renders that find it: about four times one render of the request,
  measured 5 ms with no tools and 55-260 ms with 50 tools on a Threadripper 1950X (600 ms on Inkling's
  template); later requests with the same system prompt and tools reuse the result. A raw
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
- The prompt node's floor is `max(--slot-save-block, --slot-save-min-tokens)`. main-patched armed a
  node near the end of a cold prompt for sliding-window models only, with the floor
  `max(--slot-save-block, --slot-save-context-min-tokens)`; the default `cold` prompt node now does
  this for every class that cannot rewind, at the lower floor.
- On a rollback to a release older than the `.dft` draft sidecars, purge `*.dft` from the store first;
  an older binary counts them as units, evicts them and can delete a newer binary's `.tmp.dft` temps.
  Never run the two on one store at the same time.
- Every instance writing to one directory must agree on the model, `-c` (except for the rung rule above),
  `--cache-type-k/v`, `--slot-save-block`, the mmproj and the RoPE settings; a unit from a mismatched
  instance is refused, never restored into the wrong context.
- Restore needs the prefix to be token-identical. A date or a counter rendered into the system prompt
  changes it every turn and defeats the cache.
- One store, one cap: `--slot-save-max-mb` is enforced over the whole directory by whichever instance
  saves, so give every pool sharing it the same value.
