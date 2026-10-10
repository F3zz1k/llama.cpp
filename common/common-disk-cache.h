// fork-only: the parameters of the auto disk prompt/KV cache (--slot-save-*), kept out of common.h.
// common_params inherits them, so they are used as params.slot_save_* like any other field.

#pragma once

#include <cstdint>

// when the auto disk cache publishes a node at the end of the last user message (--slot-save-node-prompt)
enum common_slot_save_node_prompt {
    COMMON_SLOT_SAVE_NODE_PROMPT_OFF  = 0, // never
    COMMON_SLOT_SAVE_NODE_PROMPT_COLD = 1, // only while prefilling a prompt that got essentially no reuse
    COMMON_SLOT_SAVE_NODE_PROMPT_ON   = 2, // whenever at least one block of new prompt precedes it
};

struct common_params_disk_cache {
    // bounded slot-save store (LRU eviction by mtime; a state file + its .logits sidecar are
    // evicted together as one unit). Default 0 = unlimited: plain --slot-save-path never deletes
    // anything (upstream behaviour). Eviction is opt-in and only runs when --slot-save-auto is set,
    // i.e. when --slot-save-path is a server-owned auto-cache directory (see auto_cache_enabled()).
    int32_t slot_save_max_count = 0;  // max snapshots in slot_save_path (0 = unlimited)
    int64_t slot_save_max_bytes = 0;  // total byte cap of the store (0 = unlimited)
    // --- auto disk prompt/KV cache (opt-in, default OFF; see tools/server "auto disk cache") ---
    // master switch for the transparent cross-process prompt/KV cache. Requires slot_save_path.
    // When false the whole feature is inert: no startup dir scan, no index, no per-request hashing.
    bool    slot_save_auto  = false;  // master switch; requires slot_save_path to be set
    bool    slot_save_incremental = false; // save only the KV delta since the last checkpoint instead
                                           // of a complete snapshot each time; requires slot_save_auto
    int32_t slot_save_block = 256;    // token-ID hash block size (vLLM-APC / SGLang-radix style)
    // minimum snapshot size (in cells/tokens) worth persisting: a trivially small prefix saves
    // little prefill against the state-file write + later restore, so it is skipped. The effective
    // floor is max(slot_save_block, slot_save_min_tokens): a snapshot must always cover >= 1 block.
    int32_t slot_save_min_tokens = 1024;
    // save-side floor for the [0,B) shared-context base checkpoint only (the sub-range save of the
    // leading context shared by N chats). Effective floor is max(slot_save_block, this); a base
    // below a few k tokens is not worth a distinct file. Larger than slot_save_min_tokens on purpose.
    int32_t slot_save_context_min_tokens = 4096;
    // restore-side absolute floor: skip the disk load (reprocess instead) when the verified,
    // block-aligned matched prefix (n_keep_disk) is below this. Default 0 = opt-in / no behaviour
    // change; a nonzero value silently suppresses short-match restores that ship today. Must be
    // <= slot_save_min_tokens and <= slot_save_context_min_tokens (else a snapshot could be written
    // and then always skipped on restore).
    int32_t slot_restore_min_tokens = 0;
    // idle-delay flush: persist a slot's warm KV after this many seconds of idleness, so a lone
    // request's state survives a crash and becomes visible to peer instances without waiting for
    // the next task to arrive. -1 disables it (legacy write-on-reuse/shutdown only).
    int32_t slot_save_idle_seconds = 60;
    bool    slot_save_idle_seconds_set = false; // true once --slot-save-idle-seconds is given explicitly
    // background writer of the auto disk cache: host memory (MiB) a save may hold between its copy off the
    // device and its write; a save that does not fit streams through two chunks when the writer is idle and
    // is dropped (and counted) when it is busy. 0 writes every save on the server thread instead.
    int32_t slot_save_staging_mb = 1024;
    bool    slot_save_defer      = true;  // stage 2: copy positional K/V after the capture (side-state at once)
    // checkpoint triggers of the auto disk cache: which moments publish a node (docs/disk-cache.md).
    // The conversation itself is saved on idle (slot_save_idle_seconds), on reclaim and at shutdown.
    bool    slot_save_node_system   = true;  // node at the end of the leading system context (chat template preamble end
                                             // message; a system-only request caches the whole system prompt)
    common_slot_save_node_prompt slot_save_node_prompt = COMMON_SLOT_SAVE_NODE_PROMPT_COLD; // node at the end of the last user message
                                             // (cold by default: every class that cannot rewind needs it)
    bool    slot_save_node_response = false; // save the conversation as soon as each response completes
    bool    slot_save_node_tool     = false; // same, but only for responses that end in tool calls
    bool    slot_save_on_reclaim    = true;  // save a slot's conversation before a different one takes the slot
    // periodic prefill checkpoint: while a prompt prefills, publish a node every N tokens (at each multiple of N),
    // so an interrupted long prefill resumes from the last one and a request that diverges inside a long message
    // restores to within N tokens of the divergence. 0 = off (default)
    int32_t slot_save_prefill_interval = 0;
    // selective restore loads (default on): a restore loads only the part of a unit it needs, the side-state alone
    // when the slot already holds the unit's positional cells (restore mode 2) and only the positional cells of a
    // composed chain's inner nodes (their side-states are overwritten by the tip's). Off = every unit loads whole.
    bool    slot_restore_selective = true;
};
