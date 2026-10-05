// test-state-skip-positional: LLAMA_STATE_SEQ_FLAGS_SKIP_POSITIONAL (the server's restore mode 2) loads only the
// side-state of a saved sequence and keeps the destination's own positional cells.
//
// For one model (one memory class):
//   supported  llama_memory_can_skip_positional() matches --expect-supported.
//   whole      a sequence decoded to P is saved whole, decoded on to L, then the saved file is loaded with the
//              flag and the sequence trimmed to P: its state is then byte-identical to the state saved at P,
//              and the next tokens decode to the same logits as a context that only ever decoded to P.
//   delta      the same with a delta node (the cells [LO, P) plus the side-state at P) as the file: the skip
//              reads past the delta's cells just as well.
//   control    trimming to P WITHOUT the load leaves a different state (or fails), so the equality above is
//              not vacuous.
//   refused    on a memory type that does not support it, the load fails and the sequence is untouched.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); printf("FAIL: " __VA_ARGS__); printf("\n"); g_fail++; } } while (0)

static std::string g_model;
static std::string g_tmp = ".";
static int g_expect = -1;

static constexpr int P  = 96;   // the saved point
static constexpr int LO = 40;   // the delta node starts here
static constexpr int L  = 160;  // the slot decodes on to here before the restore
static constexpr int NX = 8;    // tokens decoded after the restore, for the logits check

static llama_context * make_ctx(llama_model * model) {
    auto cp = llama_context_default_params();
    cp.n_ctx           = 1024;
    cp.n_batch         = 1024;
    cp.n_ubatch        = 64;
    cp.n_seq_max       = 1;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create a context\n");
        exit(2);
    }
    return ctx;
}

static void decode(llama_context * ctx, const std::vector<llama_token> & toks, int p_begin, int p_end) {
    const int n = p_end - p_begin;
    if (n <= 0) {
        return;
    }
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i]     = toks[p_begin + i];
        b.pos[i]       = p_begin + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = true;
    }
    b.n_tokens = n;
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    if (rc != 0) {
        fprintf(stderr, "decode [%d, %d) failed rc=%d\n", p_begin, p_end, rc);
        exit(3);
    }
}

static std::vector<uint8_t> seq_state(llama_context * ctx) {
    std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, 0));
    const size_t n = llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0);
    buf.resize(n);
    return buf;
}

static std::vector<float> logits_after(llama_context * ctx, const std::vector<llama_token> & toks) {
    decode(ctx, toks, P, P + NX);
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * lg = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(lg, lg + nv);
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) {
            g_model = argv[++i];
        } else if (a == "--tmp" && i + 1 < argc) {
            g_tmp = argv[++i];
        } else if (a == "--expect-supported" && i + 1 < argc) {
            g_expect = atoi(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s -m model.gguf [--tmp dir] [--expect-supported 0|1]\n", argv[0]);
            return 1;
        }
    }
    if (g_model.empty()) {
        fprintf(stderr, "no model\n");
        return 1;
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(g_model.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", g_model.c_str());
        return 2;
    }

    std::mt19937 rng(1234);
    std::vector<llama_token> toks(L + NX);
    for (auto & t : toks) {
        t = 10 + (llama_token) (rng() % 90);
    }

    // the reference: a context that decoded exactly [0, P), its state and the logits of the next tokens
    std::vector<uint8_t> ref_state;
    std::vector<float>   ref_logits;
    {
        llama_context * R = make_ctx(model);
        decode(R, toks, 0, P);
        ref_state  = seq_state(R);
        ref_logits = logits_after(R, toks);
        llama_free(R);
    }

    const std::string f_whole = g_tmp + "/skip-whole.bin";
    const std::string f_delta = g_tmp + "/skip-delta.bin";

    llama_context * ctx = make_ctx(model);
    const bool supported = llama_memory_can_skip_positional(llama_get_memory(ctx));
    printf("%s: skip-positional %s\n", g_model.c_str(), supported ? "supported" : "not supported");
    if (g_expect >= 0) {
        CHECK(supported == (g_expect == 1), "llama_memory_can_skip_positional() = %d, expected %d", supported, g_expect);
    }

    decode(ctx, toks, 0, P);
    CHECK(llama_state_seq_save_file(ctx, f_whole.c_str(), 0, toks.data(), P) > 0, "whole save failed");
    CHECK(llama_state_seq_save_file_range(ctx, f_delta.c_str(), 0, LO, -1, toks.data(), P) > 0, "delta save failed");
    llama_free(ctx);

    for (const auto & [label, file] : { std::make_pair("whole", f_whole), std::make_pair("delta", f_delta) }) {
        // a prefill to P, then more tokens (a response), as a slot does: [0, P) is computed exactly as the
        // reference computed it, so its cells are bit-identical to the reference's
        llama_context * X = make_ctx(model);
        decode(X, toks, 0, P);
        decode(X, toks, P, L);
        const auto before = seq_state(X);
        std::vector<llama_token> got(P);
        size_t n_tok = 0;
        const size_t nread = llama_state_seq_load_file_ext(X, file.c_str(), 0, LLAMA_STATE_SEQ_FLAGS_SKIP_POSITIONAL,
                                                           got.data(), got.size(), &n_tok);
        if (!supported) {
            CHECK(nread == 0, "%s: an unsupported memory type accepted the load", label);
            CHECK(seq_state(X) == before, "%s: a refused load changed the sequence", label);
            printf("  %-6s refused, sequence untouched\n", label);
            llama_free(X);
            continue;
        }
        CHECK(nread > 0 && n_tok == (size_t) P, "%s: load failed (nread %zu, %zu tokens)", label, nread, n_tok);
        CHECK(std::equal(got.begin(), got.begin() + P, toks.begin()), "%s: wrong tokens", label);
        CHECK(llama_memory_seq_rm(llama_get_memory(X), 0, P, -1), "%s: trimming to P failed", label);
        const auto after = seq_state(X);
        CHECK(after == ref_state, "%s: the state differs from one that stopped at P (%zu vs %zu bytes)", label,
              after.size(), ref_state.size());
        const auto lg = logits_after(X, toks);
        float worst = 0.0f;
        for (size_t i = 0; i < lg.size(); ++i) {
            worst = std::max(worst, std::fabs(lg[i] - ref_logits[i]));
        }
        CHECK(worst < 1e-3f, "%s: next-token logits differ from the reference by %g", label, worst);
        printf("  %-6s state %s, logits max diff %g\n", label, after == ref_state ? "EQUAL" : "DIFF", worst);
        llama_free(X);
    }

    if (supported) {
        // positive control: the trim alone cannot reach the state at P (the side-state stays at L)
        llama_context * X = make_ctx(model);
        decode(X, toks, 0, P);
        decode(X, toks, P, L);
        const bool trimmed = llama_memory_seq_rm(llama_get_memory(X), 0, P, -1);
        const bool equal   = trimmed && seq_state(X) == ref_state;
        printf("  control: trim without the load %s\n", !trimmed ? "refused" : equal ? "EQUAL" : "DIFF (the load is load-bearing)");
        CHECK(!equal, "the trim alone already gives the state at P, so this test cannot see a missing side-state");
        llama_free(X);
    }

    std::remove(f_whole.c_str());
    std::remove(f_delta.c_str());
    llama_model_free(model);
    printf("checks: %d\n", g_checks);
    CHECK(g_checks >= 5, "only %d checks ran", g_checks);
    printf("RESULT %s: %s (%d failures)\n", g_model.c_str(), g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
