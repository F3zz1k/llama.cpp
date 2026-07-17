#pragma once

#include "common.h"
#include "log.h"
#include "llama.h"
#include "chat.h"
#include "mtmd.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <string>
#include <vector>
#include <cinttypes>

using json = nlohmann::ordered_json;

#define SLT_DBG(slot, fmt, ...) LOG_DBG("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_TRC(slot, fmt, ...) LOG_TRC("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_INF(slot, fmt, ...) LOG_INF("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...) LOG_WRN("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...) LOG_ERR("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_CNT(slot, fmt, ...) LOG_CNT(""                                 fmt,                                                                __VA_ARGS__)

#define SRV_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_TRC(fmt, ...) LOG_TRC("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

using raw_buffer = std::vector<uint8_t>;

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const & err) {
            LOG_WRN("Wrong type supplied for parameter '%s'. Expected '%s', using default value: %s\n", key.c_str(), json(default_value).type_name(), err.what());
            return default_value;
        }
    } else {
        return default_value;
    }
}

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
    ERROR_TYPE_EXCEED_CONTEXT_SIZE, // custom error
};

// thin wrapper around common_grammar_trigger with (de)serialization functions
struct server_grammar_trigger {
    common_grammar_trigger value;

    server_grammar_trigger() = default;
    server_grammar_trigger(const common_grammar_trigger & value) : value(value) {}
    server_grammar_trigger(const json & in) {
        value.type = (common_grammar_trigger_type) in.at("type").get<int>();
        value.value = in.at("value").get<std::string>();
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            value.token = (llama_token) in.at("token").get<int>();
        }
    }

    json to_json() const {
        json out {
            {"type", (int) value.type},
            {"value", value.value},
        };
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            out["token"] = (int) value.token;
        }
        return out;
    }
};

json format_error_response(const std::string & message, const enum error_type type);

//
// random string / id
//

std::string random_string();
std::string gen_chatcmplid();
std::string gen_tool_call_id();

// get a random marker; note: each time the server restarts, the marker will be different
const char * get_media_marker();

//
// lora utils
//

// check whether the given lora set has only aloras activated (empty => false)
bool lora_all_alora(const std::vector<common_adapter_lora_info> & loras);

// if the two sets of loras are different, they require a cache clear unless the
// change is only from aloras to aloras.
bool lora_should_clear_cache(
        const std::vector<common_adapter_lora_info> & current,
        const std::vector<common_adapter_lora_info> & next);

std::map<int, float> parse_lora_request(const json & data);

bool are_lora_equal(
        const std::vector<common_adapter_lora_info> & l1,
        const std::vector<common_adapter_lora_info> & l2);

// get the ids of all enabled loras
std::vector<size_t> lora_get_enabled_ids(const std::vector<common_adapter_lora_info> & loras);

//
// server_tokens
//

// identity metadata for one media (image/audio) chunk of a prompt. This is what the
// auto disk cache persists per chunk: enough to re-verify a chunk against a future
// request's live chunks, without storing any pixel/sample data (the KV state file
// already holds the embeddings; the request itself carries the pixels).
struct server_media_record {
    uint32_t    start_idx = 0; // index of the chunk's first cell in the token list
    uint32_t    n_tokens  = 0; // number of cells (LLAMA_TOKEN_NULL entries) the chunk occupies
    uint32_t    n_pos     = 0; // number of positions the chunk occupies (M-RoPE: != n_tokens)
    uint32_t    nx        = 0; // token grid width (image); n_tokens for audio
    uint32_t    ny        = 0; // token grid height (image); 1 for audio
    uint32_t    is_audio  = 0; // 1 if the chunk is audio, 0 if image
    std::string id;            // mtmd bitmap id: FNV-1a of the raw uploaded bytes (never empty)
};

/**
 * server_tokens is a helper to manage the input tokens and image for the server.
 * it is made this way to simplify the logic of KV cache management.
 */
struct server_tokens {
    bool has_mtmd = false;

private: // disallow accessing these members directly, risking out-of-sync

    // map a **start** index in tokens to the image chunk
    // note: the order need to be in-sync with tokens
    std::map<size_t, mtmd::input_chunk_ptr> map_idx_to_media;

    // list of tokens
    //   if the token is LLAMA_TOKEN_NULL, it indicates that this position is occupied by media chunk
    //   otherwise, it is a normal text token
    // note: a non-text chunk can occupy multiple tokens (aka memory cells) in the token list
    // note(2): for M-RoPE, an image can occupy different number of pos; do not assume 1-to-1 mapping tokens <-> pos
    llama_tokens tokens;

    // for ex. with input of 5 text tokens and 2 images (each image occupies 3 tokens and 2 pos):
    //      [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1] [img1]
    // idx  0   1   2   3   4   5      6      7      8      9      10
    // pos  0   1   2   3   4   5      5      5      7      7      7
    // map_idx_to_media will contain: {5, img0}, {8, img1}

public:
    server_tokens() = default;
    ~server_tokens() = default;

    // Prevent copying
    // TODO: server_tokens should be copyable - remove this:
    server_tokens(const server_tokens&) = delete;
    server_tokens& operator=(const server_tokens&) = delete;

    // Allow moving (usually implicitly generated if members are movable)
    server_tokens(server_tokens&&) = default;
    server_tokens& operator=(server_tokens&&) = default;

    // Allow accessing elements using [] operator
    llama_token operator[](size_t index) { return tokens[index]; }
    const llama_token& operator[](size_t index) const { return tokens[index]; }

    server_tokens(mtmd::input_chunks & mtmd_chunks, bool has_mtmd);
    server_tokens(const llama_tokens & tokens, bool has_mtmd);

    // for debugging
    std::string str() const;

    // the next position after n_tokens. if n_tokens < 0, return the next position after all tokens.
    llama_pos pos_next(int64_t n_tokens = -1) const;

    // number of tokens with position < max_pos
    size_t size_up_to_pos(llama_pos max_pos) const;

    const mtmd::input_chunk_ptr & find_chunk(size_t idx) const;

    // find next media chunk after idx
    // returns a pair of pointer to the chunk (nullptr if not found) and its start index in tokens
    std::pair<const mtmd::input_chunk_ptr *, size_t> find_next_media_chunk(size_t idx) const;

    void push_back(llama_token tok);

    // will create a copy of the chunk if it contains non-text data
    void push_back(const mtmd_input_chunk * chunk);

    // appends server tokens, updates the media map. copies media chunks.
    void push_back(server_tokens & tokens);

    // for compatibility with context shift and prompt truncation
    void insert(const llama_tokens & inp_tokens);

    // for compatibility with speculative decoding, ctx shift, slot save/load
    const llama_tokens & get_tokens() const;

    // cell-aligned token list where every media cell is LLAMA_TOKEN_NULL; unlike
    // get_tokens() this is valid for any prompt (per-request, no server-wide mtmd
    // assert). The list's length equals the prompt's KV cell count, which is what
    // llama_state_seq_save_file persists — used by the auto disk cache.
    const llama_tokens & get_cell_tokens() const;

    // identity records for every media chunk, ordered by start_idx (see
    // server_media_record). Throws if any chunk has an empty id: identity-less
    // chunks (e.g. placeholder bitmaps) can never be re-verified, so they must
    // not be persisted or matched by the auto disk cache.
    std::vector<server_media_record> extract_media_records() const;

    // true if splitting the token list at idx does not fall strictly inside a media
    // chunk: a text token, a chunk start, or one-past-the-end are all safe. Shared
    // by the auto disk cache's boundary checks (save-time index insert, scan rehash
    // via the server_media_record overload below, lookup clamp) so they cannot drift.
    bool boundary_is_chunk_safe(size_t idx) const;

    llama_tokens get_text_tokens() const;

    // per-request media signal (has_mtmd is server-wide and wrong here): true if this
    // prompt contains ANY media chunk (image/audio); a text-only prompt has an empty media map.
    bool has_media() const { return !map_idx_to_media.empty(); }

    // for compatibility with speculative decoding
    void set_token(llama_pos pos, llama_token id);

    size_t size() const { return tokens.size(); }

    bool empty() const { return tokens.empty(); }

    void clear() {
        map_idx_to_media.clear();
        tokens.clear();
    }

    void keep_first(size_t n);

    std::string detokenize(const llama_context * ctx, bool special) const;

    size_t get_common_prefix(const server_tokens & b) const;

    // split the tokens into message spans, skipping over media chunks
    common_chat_msg_spans find_message_spans(const common_chat_msg_delimiters & delims) const;

    // make sure all text tokens are within the vocab range
    bool validate(const struct llama_context * ctx) const;

    server_tokens clone() const;
};

// same predicate as server_tokens::boundary_is_chunk_safe, for the scan-time shape of
// the data: a cell-aligned token list plus media records read back from a snapshot's
// .meta sidecar (no live chunks exist there). records must be ordered by start_idx,
// as extract_media_records produces them.
bool boundary_is_chunk_safe(const llama_tokens & cells, const std::vector<server_media_record> & records, size_t idx);


//
// auto disk cache .meta sidecar (fingerprint + tokens + media identity records)
//
// The format layer of the automatic disk prompt/KV cache (--slot-save-auto; the
// cache logic itself lives in server-context.cpp). It is defined here so the
// parser — which reads untrusted on-disk bytes — links into a standalone
// fuzz/unit test (tests/test-slot-meta.cpp).
//

static constexpr uint32_t SLOT_META_MAGIC         = 0x544D4B4Cu; // "LKMT" (llama kv meta), LE
static constexpr uint32_t SLOT_META_VERSION       = 1u;          // text-only snapshot (layout byte-frozen)
static constexpr uint32_t SLOT_META_VERSION_MEDIA = 2u;          // v1 layout + appended media-record section
static constexpr uint32_t SLOT_META_MEDIA_MAX     = 4096u;       // cap: media records per snapshot
static constexpr uint32_t SLOT_META_ID_MAX        = 256u;        // cap: bytes per media-record id (0 invalid)

// Model/quant/context fingerprint that MUST match for a restore to be sound. All
// fields are stable inference-affecting identity captured once at model load and
// compared by exact equality (pure-CPU int compares). The blob produced by
// llama_state_seq_save_file is only safe to load into a context with identical KV
// geometry — a Q4_0-KV blob loaded into an F16 ctx, or a different rope/yarn scale
// (positions are baked into the saved state), silently corrupts — so cache_type_k/v
// and rope_scale are NOT optional.
struct model_fp {
    uint64_t fp_model      = 0; // hash of llama_model_desc + size + n_params (+ n_embd/n_layer)
    uint32_t fp_n_vocab    = 0;
    uint32_t fp_n_ctx_train= 0;
    uint32_t fp_n_embd     = 0;
    uint32_t fp_n_layer    = 0;
    uint32_t fp_rope_type  = 0;
    uint32_t fp_cache_k    = 0; // ggml_type of K cache (enum int)
    uint32_t fp_cache_v    = 0; // ggml_type of V cache (enum int)
    uint32_t fp_n_ctx      = 0; // effective per-seq n_ctx
    uint32_t fp_kv_full    = 0; // 1 if COMMON_CONTEXT_SEQ_RM_TYPE_FULL else 0
    uint32_t fp_block      = 0; // slot_save_block this snapshot was hashed with
    uint64_t fp_rope_scale = 0; // bit-pattern of effective rope_freq_scale (position-critical)
    // rope_freq_base and ALL YaRN params also bake positions into the saved KV state exactly as
    // rope_freq_scale does — a same-model run differing only in --rope-freq-base or any --yarn-*
    // flag would otherwise pass the fingerprint and silently restore positionally-corrupt state.
    // All are bit-cast (float->u32) into identity; yarn_orig_ctx is an int. "0/negative = use
    // model-trained value" is normalized in auto_compute_fingerprint so equal effective configs match.
    uint64_t fp_rope_base       = 0; // bit-pattern of effective rope_freq_base
    uint32_t fp_yarn_ext        = 0; // bit-pattern of yarn_ext_factor
    uint32_t fp_yarn_attn       = 0; // bit-pattern of yarn_attn_factor
    uint32_t fp_yarn_beta_fast  = 0; // bit-pattern of yarn_beta_fast
    uint32_t fp_yarn_beta_slow  = 0; // bit-pattern of yarn_beta_slow
    uint32_t fp_yarn_orig_ctx   = 0; // yarn_orig_ctx (int)
    uint64_t fp_lora       = 0; // hash of active LoRA-set ids+scales (0 if none)
    // refuse cross-shape restores: 1 if the server was launched with --mmproj (mctx != nullptr),
    // else 0. The auto-cache only ever persists text-only prefixes, but mmproj-aware rope (M-RoPE)
    // and projector wiring CAN alter the text KV layout, so we conservatively REFUSE to cross-load
    // a text-only-server snapshot into an mmproj server (or vice-versa) — they get disjoint stores.
    // Removing this bit later would require proving the text KV layout is identical across the two
    // deployment shapes.
    uint32_t fp_mmproj_loaded   = 0;
    // gguf-header hash of the loaded --mmproj file (0 on a text-only server); see
    // mmproj_header_fingerprint. Catches projector swap, requantization and dimension
    // changes that the fp_mmproj_loaded 0/1 bit cannot. Persisted only in v2 (media)
    // sidecars: text KV is projector-independent, so slot_meta_read backfills it from
    // the live value on v1 (v1 => text-only) and the compare below only ever bites
    // for media snapshots — existing v1 snapshots on an --mmproj server keep matching.
    uint64_t fp_mmproj          = 0;

    // exact field-by-field equality (C++17: no defaulted operator==). Any difference REFUSES the
    // restore (invariant 3). Note: fp_block is intentionally part of identity — a snapshot hashed
    // with a different block size cannot be longest-prefix-matched against the current index.
    bool operator==(const model_fp & o) const {
        return fp_model == o.fp_model && fp_n_vocab == o.fp_n_vocab &&
               fp_n_ctx_train == o.fp_n_ctx_train && fp_n_embd == o.fp_n_embd &&
               fp_n_layer == o.fp_n_layer && fp_rope_type == o.fp_rope_type &&
               fp_cache_k == o.fp_cache_k && fp_cache_v == o.fp_cache_v &&
               fp_n_ctx == o.fp_n_ctx && fp_kv_full == o.fp_kv_full &&
               fp_block == o.fp_block && fp_rope_scale == o.fp_rope_scale &&
               fp_rope_base == o.fp_rope_base && fp_yarn_ext == o.fp_yarn_ext &&
               fp_yarn_attn == o.fp_yarn_attn && fp_yarn_beta_fast == o.fp_yarn_beta_fast &&
               fp_yarn_beta_slow == o.fp_yarn_beta_slow && fp_yarn_orig_ctx == o.fp_yarn_orig_ctx &&
               fp_lora == o.fp_lora && fp_mmproj_loaded == o.fp_mmproj_loaded &&
               fp_mmproj == o.fp_mmproj;
    }
};

// the .meta sidecar path for an auto snapshot's state file. The sidecar is the tiny
// tokens+fingerprint(+media) twin this feature adds so the startup scan / pre-restore
// verify reads only a small file, never the multi-GB state.
std::string slot_meta_sidecar_path(const std::string & state_filepath);

// Best-effort atomic write of the .meta sidecar (LE, temp+rename — the exact idiom
// of slot_logits_write). v1 (media empty, layout byte-frozen): magic/version,
// fingerprint fields, tok_count, chain_hash, then int32 tokens[tok_count]. v2 (media
// records present): the full v1 layout, then fp_mmproj, n_media and the records
// (start_idx/n_tokens/n_pos/nx/ny/is_audio/id_len/id each). For v2 `toks` must be
// the cell-aligned list (media cells LLAMA_TOKEN_NULL, see get_cell_tokens) and the
// records must tile its NULL cells exactly, as extract_media_records produces them —
// slot_meta_read rejects anything else. Returns true on success. Never throws.
bool slot_meta_write(const std::string & state_filepath,
                     const model_fp & fp,
                     const llama_tokens & toks,
                     uint64_t chain_hash,
                     const std::vector<server_media_record> & media = {});

// Read a .meta sidecar (version-aware: v1 and v2). Returns true and fills the
// outputs iff a valid sidecar exists; any short read / bad magic / unknown version /
// cap or media-tiling violation => false with outputs cleared. Never throws. On v1,
// fp_out.fp_mmproj is backfilled from `cur_fp_mmproj` (sound: v1 => text-only =>
// projector-independent KV), so the fingerprint compare cannot refuse pre-v2
// snapshots on an --mmproj server. To keep that backfill sound, a v1 sidecar
// containing any LLAMA_TOKEN_NULL cell — or any bytes past the token array — is
// rejected: no v1 writer ever emits either, so both can only be a corrupt or
// relabelled media sidecar trying to bypass fp_mmproj. Note: `chain_hash` is
// recorded for debuggability but the authority for reuse is always the
// byte-compared tokens.
bool slot_meta_read(const std::string & state_filepath,
                    uint64_t cur_fp_mmproj,
                    model_fp & fp_out,
                    llama_tokens & toks_out,
                    std::vector<server_media_record> & media_out);


//
// tokenizer and input processing utils
//

bool json_is_array_of_numbers(const json & data);

// is array having BOTH numbers & strings?
bool json_is_array_of_mixed_numbers_strings(const json & data);

// does array have any individual integers/tokens?
bool json_is_array_and_contains_numbers(const json & data);

// get value by path(key1 / key2)
json json_get_nested_values(const std::vector<std::string> & paths, const json & js);

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special);

// return the last index of character that can form a valid string
// if the last character is potentially cut in half, return the index before the cut
// if validate_utf8(text) == text.size(), then the whole text is valid utf8
size_t validate_utf8(const std::string& text);

// process mtmd prompt, return the server_tokens containing both text tokens and media chunks
// if is_placeholder is true, the media chunk will be treated as placeholder for counting tokens; the output tokens are not usable for actual inference (e.g. for submitting a task to server_queue)
server_tokens process_mtmd_prompt(mtmd_context * mctx, const std::string & prompt, const std::vector<raw_buffer> & files, bool is_placeholder = false);

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, 56], [78, 90, 12]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56], { "prompt_string": "string", "multimodal_data": [ "base64" ]}]
 */
std::vector<server_tokens> tokenize_input_prompts(
                                        const llama_vocab * vocab,
                                        mtmd_context * mctx,
                                        const json & json_prompt,
                                        bool add_special,
                                        bool parse_special);

//
// OAI utils
//

// global server parameters for chat formatting / parsing
struct server_chat_params {
    bool use_jinja;
    bool prefill_assistant;
    common_reasoning_format reasoning_format;
    std::map<std::string, std::string> chat_template_kwargs; // mapping key --> json value
    common_chat_templates_ptr tmpls;
    bool allow_image;
    bool allow_audio;
    bool allow_video;
    bool enable_thinking = true;
    int  reasoning_budget = -1;
    std::string reasoning_budget_message;
    std::string media_path;
    bool force_pure_content = false;
};

// used by /completions endpoint
json oaicompat_completion_params_parse(const json & body);

// used by /chat/completions endpoint
json oaicompat_chat_params_parse(
    json & body, /* openai api json semantics */
    const server_chat_params & opt,
    std::vector<raw_buffer> & out_files);

// TODO: move it to server-task.cpp
json format_embeddings_response_oaicompat(
    const json & request,
    const std::string & model_name,
    const json & embeddings,
    bool use_base64 = false);

// TODO: move it to server-task.cpp
json format_response_rerank(
        const json & request,
        const std::string & model_name,
        const json & ranks,
        bool is_tei_format,
        std::vector<std::string> & texts,
        int top_n);

//
// other utils
//

std::vector<llama_token_data> get_token_probabilities(llama_context * ctx, int idx, size_t n_top);

std::string safe_json_to_str(const json & data);

std::string tokens_to_str(llama_context * ctx, const llama_tokens & tokens);
std::string tokens_to_str(const llama_vocab * vocab, const llama_tokens & tokens);

// format incomplete utf-8 multibyte character for output
std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token);

// format server-sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string format_oai_sse(const json & data);

std::string format_oai_resp_sse(const json & data);

// format Anthropic-style SSE with event types
std::string format_anthropic_sse(const json & data);

bool is_valid_utf8(const std::string & str);

//
// formatting output responses
// TODO: move these to server-task.cpp
//

llama_tokens format_prompt_infill(
        const llama_vocab * vocab,
        const json & input_prefix,
        const json & input_suffix,
        const json & input_extra,
        const int n_batch,
        const int n_predict,
        const int n_ctx,
        const bool spm_infill,
        const llama_tokens & tokens_prompt);

// format rerank task: [BOS]query[EOS][SEP]doc[EOS].
server_tokens format_prompt_rerank(
        const struct llama_model * model,
        const struct llama_vocab * vocab,
        mtmd_context * mctx,
        const std::string & query,
        const std::string & doc);
