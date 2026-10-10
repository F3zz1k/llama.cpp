#pragma once

// fork-only: the disk-cache side of llama_kv_cache (state_skip, positional K/V for deferred captures, the
// destructor's capture hook, and the NO_CLEAR append check), kept out of llama-kv-cache.cpp. The member
// functions are declared in llama-kv-cache.h and defined in llama-kv-cache-state-ext.cpp.

#include "llama.h"
#include "llama-state-deferred.h"

#include <cstdint>

class llama_kv_cells;
struct llama_ubatch;

// state_read_meta's check that the cells of a delta appended with LLAMA_STATE_SEQ_FLAGS_NO_CLEAR continue
// what dest_seq_id already holds; logs the refusal under `func` and returns false when the restore must fail
bool llama_kv_cache_append_continues(const llama_kv_cells & cells, llama_seq_id dest_seq_id, const llama_ubatch & ubatch,
                                     uint32_t cell_count, llama_state_seq_flags flags, const char * func);
