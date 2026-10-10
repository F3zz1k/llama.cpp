#pragma once

// fork-only: the disk-cache side of llama_context (range, sink and deferred state saves, the KV layout
// getters and their C API in include/llama-fork.h), kept out of llama-context.cpp. The member functions
// are declared in llama-context.h and defined in llama-context-state-ext.cpp.

#include "llama.h"
#include "llama-state-deferred.h"

struct llama_memory_i;

// state_seq_read_data's check of the LLAMA_STATE_SEQ_FLAGS_SKIP_* flags against what `memory` can load;
// logs the refusal under `func` and returns false when the load must not start
bool llama_state_seq_flags_supported(const llama_memory_i * memory, llama_state_seq_flags flags, const char * func);
