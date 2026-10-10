// fork-only: Test 10 of the save/load suite as it ran in test-save-load-state.cpp until the fork moved it
// here, so the upstream file stays as upstream has it. Incremental delta compose (the disk KV cache's base
// + delta chain), run after the same baseline as there. The harness below (status, logits comparison,
// generation, baseline and main) is copied from test-save-load-state.cpp and keeps its arguments and defaults
// (only the baseline's dump file is renamed, so the two tests can run in parallel).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <string>
#include <utility>
#include <vector>

constexpr double NMSE_THRESHOLD = 1e-5;

enum class test_status {
    PASS,
    FAIL,
    SKIP,
    WHOLE, // delta test only: the memory type ignores position ranges, so it can only publish whole roots
};

static const char * test_status_str(test_status status) {
    switch (status) {
        case test_status::PASS: return "\033[1;32mPASS\033[0m";
        case test_status::FAIL: return "\033[1;31mFAIL\033[0m";
        case test_status::SKIP: return "\033[1;33mSKIP\033[0m";
        case test_status::WHOLE: return "\033[1;36mWHOL\033[0m";
    }
    return "";
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        const float a_i = a[i];
        const float b_i = b[i];

        mse_a_b += (double) (a_i - b_i) * (a_i - b_i);
        mse_a_0 += (double) a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

struct generation_result {
    llama_tokens tokens;
    std::vector<std::vector<float>> logits;

    bool empty() const { return tokens.empty(); }
};

static bool get_current_logits(llama_context * ctx, std::vector<float> & out) {
    const auto * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        return false;
    }
    out.assign(logits, logits + n_vocab);
    return true;
}

static generation_result generate_tokens(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id) {
    generation_result result;
    common_batch batch(ctx);

    for (int i = 0; i < n_predict; i++) {
        std::vector<float> logits;
        if (!get_current_logits(ctx, logits)) {
            LOG_ERR("\n%s: failed to get logits\n", __func__);
            return {};
        }

        auto next_token = llama_sampler_sample(smpl, ctx, -1);

        // LOG_LEVEL_INFO gate: visible in single-model mode, silenced in --models table mode
        LOGV(LOG_LEVEL_INFO, "%d ", next_token);
        result.tokens.push_back(next_token);
        result.logits.push_back(std::move(logits));

        batch.clear();
        batch.add(next_token, n_past, seq_id, true);

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return {};
        }
        n_past++;
    }

    return result;
}

static bool generate_tokens_compare(
        llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id,
        const generation_result & expected) {
    if (expected.tokens.size() != expected.logits.size() || expected.tokens.size() < (size_t) n_predict) {
        LOG_ERR("\n%s: invalid expected generation\n", __func__);
        return false;
    }

    common_batch batch(ctx);

    for (int i = 0; i < n_predict; i++) {
        std::vector<float> logits;
        if (!get_current_logits(ctx, logits)) {
            LOG_ERR("\n%s: failed to get logits\n", __func__);
            return false;
        }
        if (logits.size() != expected.logits[i].size()) {
            LOG_ERR("\n%s: logits size mismatch at step %d: %zu != %zu\n", __func__, i, logits.size(), expected.logits[i].size());
            return false;
        }

        const double nmse_val = nmse(expected.logits[i], logits);
        LOG_TRC("%s: step %d nmse = %.6e\n", __func__, i, nmse_val);
        if (nmse_val > NMSE_THRESHOLD) {
            LOG_ERR("\n%s: error: NMSE at step %d is %.6e (threshold %.1e)\n", __func__, i, nmse_val, NMSE_THRESHOLD);
            return false;
        }

        const auto next_token = llama_sampler_sample(smpl, ctx, -1);
        const auto expected_token = expected.tokens[i];

        LOGV(LOG_LEVEL_INFO, "%d ", next_token);
        if (next_token != expected_token) {
            LOG_TRC("%s: sampled token %d differs from expected %d, using expected token\n", __func__, next_token, expected_token);
        }

        batch.clear();
        batch.add(expected_token, n_past, seq_id, true);

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return false;
        }
        n_past++;
    }

    return true;
}

// Test 1: baseline
// - decode all but the last token
// - save state to disk
// - decode the last token
// - generate n_predict tokens
static generation_result test_baseline(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    auto n_past = 0;
    if (!common_prompt_batch_decode(ctx.get(), tokens, (int)tokens.size(), n_past, params.n_batch, params.out_file, true)) {
        LOG_ERR("%s: failed to decode prompt\n", __func__);
        return {};
    }

    LOGV(LOG_LEVEL_INFO, "\n=== Test 1: baseline ===\n");

    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (result.empty()) {
        return {};
    }

    LOGV(LOG_LEVEL_INFO, "\n");

    return result;
}


// Test 10: incremental delta compose (the disk KV cache's base + delta chain)
// - decode the first half of the prompt (all but the last token) and save it whole (the root)
// - decode the rest and save only positions [L, -1) with the range API (the delta)
// - a delta that is not smaller than a whole save of the same sequence means the memory type ignores
//   the range: report WHOLE (correct, whole roots only) and never compose it, as the server does
// - otherwise load root (clearing) + delta (NO_CLEAR) into a fresh context, replay the last prompt
//   token and compare the generation against the baseline
static test_status test_state_delta_compose(struct llama_model * model, const struct common_params & params,
                                            const llama_tokens & tokens, const generation_result & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;

    LOGV(LOG_LEVEL_INFO, "\n=== Test 10: incremental delta compose ===\n");

    if (tokens.size() < 4) {
        LOG_ERR("%s: prompt too short\n", __func__);
        return test_status::FAIL;
    }

    const size_t n_prefix = tokens.size() - 1;      // the state the baseline saves: all but the last token
    const size_t n_root   = n_prefix / 2;
    const llama_tokens tok_root (tokens.begin(),          tokens.begin() + n_root);
    const llama_tokens tok_delta(tokens.begin() + n_root, tokens.begin() + n_prefix);

    const std::string base = "state-delta." + std::filesystem::path(params.model.path).filename().string();
    const std::string path_root  = base + ".root.tmp.bin";
    const std::string path_delta = base + ".delta.tmp.bin";
    const std::string path_whole = base + ".whole.tmp.bin";
    struct cleanup_t {
        std::vector<std::string> paths;
        ~cleanup_t() { std::error_code ec; for (const auto & p : paths) { std::filesystem::remove(p, ec); } }
    } cleanup { { path_root, path_delta, path_whole } };

    size_t n_delta = 0;
    size_t n_whole = 0;
    {
        auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
        if (!ctx) {
            LOG_ERR("%s: failed to create context\n", __func__);
            return test_status::FAIL;
        }
        if (llama_get_memory(ctx.get()) == nullptr) {
            LOGV(LOG_LEVEL_INFO, "PASS (model has no memory)\n");
            return test_status::PASS;
        }

        common_batch batch_root = common_batch_get_one(ctx.get(), tok_root);
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch_root.get())) {
            LOG_ERR("%s: failed to decode the root prefix\n", __func__);
            return test_status::FAIL;
        }
        if (llama_state_seq_save_file(ctx.get(), path_root.c_str(), 0, tokens.data(), n_root) == 0) {
            LOG_ERR("%s: failed to save the root\n", __func__);
            return test_status::FAIL;
        }

        common_batch batch_delta = common_batch_get_one(ctx.get(), tok_delta);
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch_delta.get())) {
            LOG_ERR("%s: failed to decode the delta suffix\n", __func__);
            return test_status::FAIL;
        }
        n_delta = llama_state_seq_save_file_range(ctx.get(), path_delta.c_str(), 0, (llama_pos) n_root, -1, tokens.data(), n_prefix);
        n_whole = llama_state_seq_save_file(ctx.get(), path_whole.c_str(), 0, tokens.data(), n_prefix);
        if (n_delta == 0 || n_whole == 0) {
            LOG_ERR("%s: failed to save the delta (%zu B) or the whole sequence (%zu B)\n", __func__, n_delta, n_whole);
            return test_status::FAIL;
        }
    }

    if (n_delta > n_whole) {
        LOG_ERR("%s: the delta (%zu B) is larger than a whole save (%zu B)\n", __func__, n_delta, n_whole);
        return test_status::FAIL;
    }
    if (n_delta == n_whole) {
        LOGV(LOG_LEVEL_INFO, "WHOLE (range ignored: delta %zu B == whole %zu B)\n", n_delta, n_whole);
        return test_status::WHOLE;
    }

    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return test_status::FAIL;
    }

    llama_tokens tok_out(tokens.size());
    size_t n_out = 0;
    if (llama_state_seq_load_file_ext(ctx.get(), path_root.c_str(), 0, 0, tok_out.data(), tok_out.size(), &n_out) == 0 || n_out != n_root) {
        LOG_ERR("%s: failed to load the root (%zu tokens)\n", __func__, n_out);
        return test_status::FAIL;
    }
    if (llama_state_seq_load_file_ext(ctx.get(), path_delta.c_str(), 0, LLAMA_STATE_SEQ_FLAGS_NO_CLEAR,
                                      tok_out.data(), tok_out.size(), &n_out) == 0 || n_out != n_prefix) {
        LOG_ERR("%s: failed to compose the delta (%zu tokens)\n", __func__, n_out);
        return test_status::FAIL;
    }

    const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0);
    if (pos_max != (llama_pos) n_prefix - 1) {
        LOG_ERR("%s: composed sequence ends at %d, expected %zu\n", __func__, pos_max, n_prefix - 1);
        return test_status::FAIL;
    }

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    int n_past = (int) n_prefix;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return test_status::FAIL;
    }
    n_past++;

    if (!generate_tokens_compare(ctx.get(), smpl.get(), n_past, params.n_predict, 0, expected_result)) {
        return test_status::FAIL;
    }

    LOGV(LOG_LEVEL_INFO, "\nPASS (delta %zu B vs whole %zu B)\n", n_delta, n_whole);
    return test_status::PASS;
}

// column headers for the --models table, one per test, in the order they are run
static const std::vector<const char *> test_names = {
    "baseline", "delta",
};

struct test_suite {
    std::vector<test_status> results;

    bool all_passed() const {
        // WHOLE is a classification, not a failure: that memory type is correct, just not incremental
        return std::all_of(results.begin(), results.end(), [](test_status s) { return s == test_status::PASS || s == test_status::WHOLE; });
    }
};

static test_suite run_delta_tests_for_model(const std::string & model_path, const struct common_params & base_params) {
    test_suite suite;

    struct common_params params = base_params;
    params.model.path = model_path;

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init model '%s'\n", __func__, model_path.c_str());
        suite.results.assign(test_names.size(), test_status::SKIP);
        return suite;
    }

    GGML_ASSERT(llama_init->context() == nullptr);

    // Tokenize prompt or generate random tokens
    llama_tokens tokens;
    if (params.prompt.empty()) {
        const int n_prompt = params.n_batch;

        // this path is useful for model files that do not have a tokenizer
        LOG_INF("%s: no prompt provided, generating %d (n_batch) random tokens\n", __func__, n_prompt);

        const auto * vocab = llama_model_get_vocab(model);
        const auto n_vocab = llama_vocab_n_tokens(vocab);

        std::mt19937 rng(params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int i = 0; i < n_prompt; i++) {
            tokens.push_back(dist(rng));
        }
    } else {
        LOG_INF("%s: tokenizing prompt '%s'\n", __func__, params.prompt.c_str());

        auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
        tokens = common_tokenize(ctx.get(), params.prompt, true);
    }

    LOG_INF("%s: the input prompt is %d tokens\n", __func__, (int)tokens.size());

    // Test 1: baseline (saves state to disk)
    auto result_baseline = test_baseline(model, params, tokens);
    suite.results.push_back(result_baseline.empty() ? test_status::FAIL : test_status::PASS);

    // Test 10: incremental delta compose (depends on the baseline generation)
    suite.results.push_back(result_baseline.empty() ? test_status::SKIP
                                                    : test_state_delta_compose(model, params, tokens, result_baseline));

    return suite;
}

static void print_usage(int /* argc */, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n  %s -m your_model.gguf\n", argv[0]);
    LOG("\n  %s --models tests/test-models\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "";
    params.n_batch = 100;
    params.out_file = "dump_state_delta.bin";
    params.sampling.seed = 1234;

    common_init();

    // extract our own --models DIR option before handing the rest to the common arg parser
    std::string models_dir;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --models requires a directory argument\n", __func__);
                return 1;
            }
            models_dir = argv[i + 1];
            i++;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int)filtered_argv.size() - 1;

    if (!models_dir.empty()) {
        params.model.path = models_dir;
    }

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    if (params.n_parallel == 1) {
        params.kv_unified = true;
    }

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    llama_backend_init();

    if (models_dir.empty()) {
        const test_suite suite = run_delta_tests_for_model(params.model.path, params);
        return suite.all_passed() ? 0 : 1;
    }

    if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
        LOG_ERR("%s: models directory '%s' does not exist\n", __func__, models_dir.c_str());
        return 1;
    }

    std::vector<std::string> models;
    for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
            models.push_back(entry.path().string());
        }
    }
    std::sort(models.begin(), models.end());

    if (models.empty()) {
        LOG_ERR("%s: no .gguf models found in '%s'\n", __func__, models_dir.c_str());
        return 1;
    }

    auto col_width = [](const char * name) { return (int) std::max(strlen(name), (size_t) 4); };

    size_t name_width = 5; // "Model"
    for (const auto & model_path : models) {
        name_width = std::max(name_width, std::filesystem::path(model_path).filename().string().size());
    }

    // silence everything but the table itself (LOG has verbosity LOG_LEVEL_OUTPUT = 0)
    common_log_set_verbosity_thold(0);

    LOG("%-*s", (int) name_width, "Model");
    for (const auto & name : test_names) {
        LOG("  %-*s", col_width(name), name);
    }
    LOG("\n");
    common_log_flush(common_log_main());

    size_t n_pass = 0;
    size_t n_fail = 0;
    for (const auto & model_path : models) {
        const auto name = std::filesystem::path(model_path).filename().string();

        LOG("%-*s", (int) name_width, name.c_str());
        common_log_flush(common_log_main());

        const test_suite suite = run_delta_tests_for_model(model_path, params);

        for (size_t i = 0; i < suite.results.size(); i++) {
            LOG("  %s%*s", test_status_str(suite.results[i]), col_width(test_names[i]) - 4, "");
        }
        LOG("\n");
        common_log_flush(common_log_main());

        if (suite.all_passed()) {
            n_pass++;
        } else {
            n_fail++;
        }
    }

    common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);
    common_log_flush(common_log_main());

    LOG_INF("%s: summary: %zu passed, %zu failed (of %zu)\n", __func__, n_pass, n_fail, models.size());

    return n_fail == 0 ? 0 : 1;
}
