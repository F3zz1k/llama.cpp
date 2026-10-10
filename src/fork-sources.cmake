# fork-only sources of libllama, kept out of src/CMakeLists.txt's own list (they build outside the unity batch
# like every other LLAMA_CORE_SOURCES entry)
set(LLAMA_CORE_FORK_SOURCES
    llama-context-state-ext.cpp
    llama-kv-cache-state-ext.cpp
    llama-memory-state-ext.cpp
    llama-state-deferred.cpp
)
