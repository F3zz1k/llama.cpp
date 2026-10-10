// fork-only: the command-line options of the auto disk prompt/KV cache, see arg-disk-cache.h.

#include "arg-disk-cache.h"

#include "common.h"

#include <stdexcept>
#include <string>

void common_params_add_disk_cache_args(const std::function<void(common_arg)> & add_opt, common_params & params) {
    add_opt(common_arg(
        {"--slot-save-max-count"}, "N",
        string_format("max number of snapshots kept in the --slot-save-auto cache (treated as a dedicated dir); oldest are evicted; no effect without --slot-save-auto (default: %d, 0 = unlimited)", params.slot_save_max_count),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("--slot-save-max-count must be >= 0 (0 = unlimited)");
            }
            params.slot_save_max_count = value;
        }
    ).set_env("LLAMA_ARG_SLOT_SAVE_MAX_COUNT").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-save-max-mb"}, "N",
        string_format("max total size (MiB) of the --slot-save-auto cache store; oldest snapshots are evicted; no effect without --slot-save-auto (default: %d, 0 = unlimited)", (int) (params.slot_save_max_bytes / (1024 * 1024))),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("--slot-save-max-mb must be >= 0 (0 = unlimited)");
            }
            params.slot_save_max_bytes = (int64_t) value * 1024 * 1024;
        }
    ).set_env("LLAMA_ARG_SLOT_SAVE_MAX_MB").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-save-auto"},
        "automatically restore/save prompt KV to/from --slot-save-path across requests and restarts (transparent disk prompt cache); requires --slot-save-path (default: disabled)",
        [](common_params & params) {
            params.slot_save_auto = true;
        }
    ).set_env("LLAMA_ARG_SLOT_SAVE_AUTO").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-save-incremental"},
        "for the auto disk cache, save only the KV delta added since the last checkpoint instead of a "
        "complete snapshot each time (much less disk/PCIe write for a growing conversation); hybrid/SWA "
        "models still write their recurrent/sliding-window state whole. Requires --slot-save-auto "
        "(default: complete snapshots)",
        [](common_params & params) {
            params.slot_save_incremental = true;
        }
    ).set_env("LLAMA_ARG_SLOT_SAVE_INCREMENTAL").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-save-block"}, "N",
        string_format("token-ID hash block size for the auto disk cache index; reuse granularity "
                      "is one block (default: %d)", params.slot_save_block),
        [](common_params & params, int value) {
            params.slot_save_block = value;
        }
    ).set_env("LLAMA_ARG_SLOT_SAVE_BLOCK").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-save-min-tokens"}, "N",
        string_format("minimum snapshot size (tokens) worth persisting to the auto disk cache; "
                      "smaller prefixes are skipped as not worth the write. The effective floor is "
                      "max(--slot-save-block, this) (default: %d)", params.slot_save_min_tokens),
        [](common_params & params, int value) {
            params.slot_save_min_tokens = value;
        }
    ).set_env("LLAMA_ARG_SLOT_SAVE_MIN_TOKENS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-save-context-min-tokens"}, "N",
        string_format("minimum size (tokens) for the [0,B) shared-context base checkpoint written to "
                      "the auto disk cache; a smaller shared prefix is not worth a distinct base file. "
                      "The effective floor is max(--slot-save-block, this) (default: %d)", params.slot_save_context_min_tokens),
        [](common_params & params, int value) {
            params.slot_save_context_min_tokens = value;
        }
    ).set_env("LLAMA_ARG_SLOT_SAVE_CONTEXT_MIN_TOKENS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-restore-min-tokens"}, "N",
        string_format("skip an auto disk cache restore (reprocess the prompt instead) when the verified, "
                      "block-aligned matched prefix is below this many tokens; the reprocess beats a "
                      "multi-GB disk load. Default 0 = opt-in, no behaviour change (default: %d)", params.slot_restore_min_tokens),
        [](common_params & params, int value) {
            params.slot_restore_min_tokens = value;
        }
    ).set_env("LLAMA_ARG_SLOT_RESTORE_MIN_TOKENS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-restore-selective"},
        {"--no-slot-restore-selective"},
        "auto disk cache: let a restore load only the part of a unit it needs on hybrid (attention plus recurrent) "
        "models: the recurrent state alone when the slot already holds the unit's attention cells, and only the "
        "attention cells of a delta chain's inner nodes, whose recurrent states the last node replaces. Disable to "
        "load every unit whole (default: enabled)",
        [](common_params & params, bool value) {
            params.slot_restore_selective = value;
        }
    ).set_env("LLAMA_ARG_SLOT_RESTORE_SELECTIVE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-save-idle-seconds"}, "N",
        string_format("flush a slot's warm KV to the auto disk cache after N seconds of idleness, "
                      "so a lone request survives a crash and is visible to peer instances without "
                      "further traffic; requires --slot-save-auto (default: %d, -1 = disabled)", params.slot_save_idle_seconds),
        [](common_params & params, int value) {
            params.slot_save_idle_seconds = value;
            params.slot_save_idle_seconds_set = true;
        }
    ).set_env("LLAMA_ARG_SLOT_SAVE_IDLE_SECONDS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-save-staging-mb"}, "N",
        string_format("auto disk cache: host memory in MiB that saves may hold between their copy off the device "
                      "and their write by the background writer, so the copy is all a request waits for. A save "
                      "that does not fit streams through two chunks when the writer is idle and is dropped (counted "
                      "in llamacpp:auto_cache_save_dropped_staging_total) when it is busy. Allocated per save and "
                      "freed as it is written. 0 writes every save on the server thread (default: %d)",
                      params.slot_save_staging_mb),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("--slot-save-staging-mb must be >= 0");
            }
            params.slot_save_staging_mb = value;
        }
    ).set_env("LLAMA_ARG_SLOT_SAVE_STAGING_MB").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-save-defer"},
        {"--no-slot-save-defer"},
        "auto disk cache: at a save, copy only the side-state (recurrent state, sliding window, indexer and "
        "compressor state) off the device at once and leave the positional K/V of append-only caches in "
        "place, copied later while the server is idle, or at once before anything could change those cells. "
        "Takes the device copy off the request's time to first token; the saved units are byte-identical. "
        "Needs the background writer (--slot-save-staging-mb > 0); classes with nothing positional copy "
        "everything at the save as before (default: enabled)",
        [](common_params & params, bool value) {
            params.slot_save_defer = value;
        }
    ).set_env("LLAMA_ARG_SLOT_SAVE_DEFER").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-save-node-system"},
        {"--no-slot-save-node-system"},
        "auto disk cache: publish a node at the end of the leading system context (system and developer "
        "messages, tools, the template's own preamble; found by rendering the chat template, for every "
        "template) while a cold prompt prefills, so other conversations sharing that system prompt restore "
        "it; a request with only a system prompt caches all of it (default: enabled)",
        [](common_params & params, bool value) {
            params.slot_save_node_system = value;
        }
    ).set_env("LLAMA_ARG_SLOT_SAVE_NODE_SYSTEM").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-save-node-prompt"}, "{off,cold,on}",
        "auto disk cache: publish a node at the end of the last user message while the prompt prefills, so "
        "a resend, regenerate or edit of the response restores it even on models that cannot rewind a "
        "longer snapshot (recurrent, hybrid, sliding-window past one window); 'cold' only for prompts that "
        "got essentially no reuse, 'on' whenever a block of new prompt precedes it (use it for clients that "
        "drop the previous reasoning or rewrite answers, and for regenerate or edit after a restart), 'off' "
        "never. Takes effect with --slot-save-auto (default: cold)",
        [](common_params & params, const std::string & value) {
            if (value == "off") {
                params.slot_save_node_prompt = COMMON_SLOT_SAVE_NODE_PROMPT_OFF;
            } else if (value == "cold") {
                params.slot_save_node_prompt = COMMON_SLOT_SAVE_NODE_PROMPT_COLD;
            } else if (value == "on") {
                params.slot_save_node_prompt = COMMON_SLOT_SAVE_NODE_PROMPT_ON;
            } else {
                throw std::invalid_argument("--slot-save-node-prompt must be off, cold or on");
            }
        }
    ).set_env("LLAMA_ARG_SLOT_SAVE_NODE_PROMPT").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-save-node-response"},
        {"--no-slot-save-node-response"},
        "auto disk cache: save the conversation as soon as each response completes, instead of only on idle, "
        "on reclaim and at shutdown (default: disabled)",
        [](common_params & params, bool value) {
            params.slot_save_node_response = value;
        }
    ).set_env("LLAMA_ARG_SLOT_SAVE_NODE_RESPONSE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-save-node-tool"},
        {"--no-slot-save-node-tool"},
        "auto disk cache: save the conversation when a response ends in tool calls (default: disabled)",
        [](common_params & params, bool value) {
            params.slot_save_node_tool = value;
        }
    ).set_env("LLAMA_ARG_SLOT_SAVE_NODE_TOOL").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-save-on-reclaim"},
        {"--no-slot-save-on-reclaim"},
        "auto disk cache: save a slot's conversation before a request from a different conversation takes "
        "the slot (default: enabled)",
        [](common_params & params, bool value) {
            params.slot_save_on_reclaim = value;
        }
    ).set_env("LLAMA_ARG_SLOT_SAVE_ON_RECLAIM").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-save-prefill-interval"}, "N",
        string_format("auto disk cache: while a prompt prefills, publish a node every N tokens (at each multiple of N "
                      "past the save floor), so a long prefill that is interrupted (a timeout, a disconnect, a crash) "
                      "resumes from the last node, and a request that changes the end of a long message restores to "
                      "within N tokens of the change. Each node is a delta on the previous one under "
                      "--slot-save-incremental (a whole snapshot without it, or on a model that cannot write deltas). "
                      "Must be 0 or at least --slot-save-block (default: %d, 0 = off)", params.slot_save_prefill_interval),
        [](common_params & params, int value) {
            params.slot_save_prefill_interval = value;
        }
    ).set_env("LLAMA_ARG_SLOT_SAVE_PREFILL_INTERVAL").set_examples({LLAMA_EXAMPLE_SERVER}));
}

void common_params_validate_disk_cache(const common_params & params) {
    // the auto disk prompt/KV cache needs a directory to live in; fail fast & loud rather than
    // silently doing nothing if --slot-save-auto is set without --slot-save-path.
    if (params.slot_save_auto && params.slot_save_path.empty()) {
        throw std::invalid_argument("--slot-save-auto requires --slot-save-path to be set");
    }
    // incremental (delta) saving only makes sense for the auto disk cache; reject it without the
    // master switch rather than silently ignoring it.
    if (params.slot_save_incremental && !params.slot_save_auto) {
        throw std::invalid_argument("--slot-save-incremental requires --slot-save-auto");
    }
    // --slot-save-block sizes the auto-cache index but is validated unconditionally: a non-positive
    // block size is never meaningful, so reject it even when the master switch is off.
    if (params.slot_save_block <= 0) {
        throw std::invalid_argument("--slot-save-block must be > 0");
    }
    // the minimum-snapshot-size floor is validated unconditionally: a negative value is never
    // meaningful (0 falls back to the block-size floor), so reject it even with the master switch off.
    if (params.slot_save_min_tokens < 0) {
        throw std::invalid_argument("--slot-save-min-tokens must be >= 0");
    }
    // the shared-context base floor and the restore floor are validated unconditionally (negatives
    // are never meaningful), even with the master switch off.
    if (params.slot_save_context_min_tokens < 0) {
        throw std::invalid_argument("--slot-save-context-min-tokens must be >= 0");
    }
    if (params.slot_restore_min_tokens < 0) {
        throw std::invalid_argument("--slot-restore-min-tokens must be >= 0");
    }
    // a written snapshot must remain restorable: a snapshot written at length L can never yield a
    // matched prefix longer than L, so if the restore floor exceeds a save floor the snapshot would
    // be written and then always skipped on restore: pure waste. Make that unrepresentable.
    if (params.slot_restore_min_tokens > params.slot_save_min_tokens) {
        throw std::invalid_argument("--slot-restore-min-tokens must be <= --slot-save-min-tokens");
    }
    if (params.slot_restore_min_tokens > params.slot_save_context_min_tokens) {
        throw std::invalid_argument("--slot-restore-min-tokens must be <= --slot-save-context-min-tokens");
    }
    // a checkpoint every N tokens: N below one block would publish units the index cannot even key apart
    if (params.slot_save_prefill_interval < 0) {
        throw std::invalid_argument("--slot-save-prefill-interval must be >= 0");
    }
    if (params.slot_save_prefill_interval > 0 && params.slot_save_prefill_interval < params.slot_save_block) {
        throw std::invalid_argument("--slot-save-prefill-interval must be 0 or >= --slot-save-block");
    }
    // idle-delay flush is inert without the master switch (auto_idle_flush_enabled() gates on
    // auto_cache_enabled()); reject an explicit --slot-save-idle-seconds without --slot-save-auto
    // rather than silently ignoring it. The default (unset) is left alone so plain servers still run.
    if (params.slot_save_idle_seconds_set && !params.slot_save_auto) {
        throw std::invalid_argument("--slot-save-idle-seconds requires --slot-save-auto");
    }
}
