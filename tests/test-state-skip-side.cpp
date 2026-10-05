// test-state-skip-side: LLAMA_STATE_SEQ_FLAGS_SKIP_SIDE (the inner nodes of a composed chain) loads only the
// positional K/V of a saved sequence and reads past its side-state.
//
// For one model (one memory class):
//   supported  llama_memory_can_skip_side() matches --expect-supported.
//   chain      a sequence is decoded to A, B and P in three steps and saved as a whole root at A and two deltas
//              ([A, B) and [B, P)). A context that holds another conversation composes the chain with SKIP_SIDE
//              on the root and the first delta and a plain load of the last: its state is byte-identical to the
//              saving context's state at P, and the next tokens decode to the same logits.
//   plain      the same chain composed without the flag gives the same state (the reference for the skip).
//   control    with SKIP_SIDE on the last node too, the side-state is never loaded and the state differs, so
//              the equality above is not vacuous: the last node's side-state is what the slot ends up with.
//   both       SKIP_SIDE together with SKIP_POSITIONAL is refused and leaves the sequence untouched.
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

static constexpr int A  = 32;   // the root ends here
static constexpr int B  = 64;   // the first delta ends here
static constexpr int P  = 96;   // the second delta (the tip) ends here
static constexpr int L  = 160;  // the other conversation the destination holds before the compose
static constexpr int NX = 8;    // tokens decoded after the compose, for the logits check

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

// a context that holds another conversation (its own attention cells and recurrent state) up to L
static llama_context * occupied_ctx(llama_model * model, const std::vector<llama_token> & other) {
    llama_context * X = make_ctx(model);
    decode(X, other, 0, L);
    return X;
}

static bool compose(llama_context * X, const std::vector<std::string> & files, const std::vector<llama_state_seq_flags> & flags,
                    const std::vector<llama_token> & toks) {
    std::vector<llama_token> got(P);
    for (size_t i = 0; i < files.size(); ++i) {
        size_t n_tok = 0;
        const size_t nread = llama_state_seq_load_file_ext(X, files[i].c_str(), 0, flags[i], got.data(), got.size(), &n_tok);
        if (nread == 0) {
            return false;
        }
        if (i + 1 == files.size() && (n_tok != (size_t) P || !std::equal(got.begin(), got.begin() + P, toks.begin()))) {
            return false;
        }
    }
    return true;
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

    std::mt19937 rng(4321);
    std::vector<llama_token> toks(P + NX);
    for (auto & t : toks) {
        t = 10 + (llama_token) (rng() % 90);
    }
    std::vector<llama_token> other(L);
    for (auto & t : other) {
        t = 10 + (llama_token) (rng() % 90);
    }

    const std::string f0 = g_tmp + "/side-root.bin";
    const std::string f1 = g_tmp + "/side-d1.bin";
    const std::string f2 = g_tmp + "/side-d2.bin";

    // the saving context is the reference: it decoded [0, A), [A, B), [B, P) in exactly these steps
    std::vector<uint8_t> ref_state;
    std::vector<float>   ref_logits;
    bool supported = false;
    {
        llama_context * S = make_ctx(model);
        supported = llama_memory_can_skip_side(llama_get_memory(S));
        decode(S, toks, 0, A);
        CHECK(llama_state_seq_save_file(S, f0.c_str(), 0, toks.data(), A) > 0, "root save failed");
        decode(S, toks, A, B);
        CHECK(llama_state_seq_save_file_range(S, f1.c_str(), 0, A, -1, toks.data(), B) > 0, "first delta save failed");
        decode(S, toks, B, P);
        CHECK(llama_state_seq_save_file_range(S, f2.c_str(), 0, B, -1, toks.data(), P) > 0, "second delta save failed");
        ref_state  = seq_state(S);
        ref_logits = logits_after(S, toks);
        llama_free(S);
    }
    printf("%s: skip-side %s\n", g_model.c_str(), supported ? "supported" : "not supported");
    if (g_expect >= 0) {
        CHECK(supported == (g_expect == 1), "llama_memory_can_skip_side() = %d, expected %d", supported, g_expect);
    }

    const llama_state_seq_flags NC   = LLAMA_STATE_SEQ_FLAGS_NO_CLEAR;
    const llama_state_seq_flags SKIP = LLAMA_STATE_SEQ_FLAGS_SKIP_SIDE;

    {
        // SKIP_SIDE with SKIP_POSITIONAL loads nothing: refused on every class, the sequence untouched
        llama_context * X = occupied_ctx(model, other);
        const auto before = seq_state(X);
        std::vector<llama_token> got(P);
        size_t n_tok = 0;
        const size_t nread = llama_state_seq_load_file_ext(X, f0.c_str(), 0, SKIP | LLAMA_STATE_SEQ_FLAGS_SKIP_POSITIONAL,
                                                           got.data(), got.size(), &n_tok);
        CHECK(nread == 0, "SKIP_SIDE with SKIP_POSITIONAL was accepted");
        CHECK(seq_state(X) == before, "a refused SKIP_SIDE | SKIP_POSITIONAL load changed the sequence");
        llama_free(X);
    }

    if (!supported) {
        llama_context * X = occupied_ctx(model, other);
        const auto before = seq_state(X);
        std::vector<llama_token> got(P);
        size_t n_tok = 0;
        const size_t nread = llama_state_seq_load_file_ext(X, f0.c_str(), 0, SKIP, got.data(), got.size(), &n_tok);
        CHECK(nread == 0, "an unsupported memory type accepted the load");
        CHECK(seq_state(X) == before, "a refused load changed the sequence");
        printf("  refused, sequence untouched\n");
        llama_free(X);
    } else {
        // plain compose: the reference for the skip
        {
            llama_context * X = occupied_ctx(model, other);
            CHECK(compose(X, { f0, f1, f2 }, { 0, NC, NC }, toks), "plain compose failed");
            const auto st = seq_state(X);
            CHECK(st == ref_state, "plain compose differs from the saving context (%zu vs %zu bytes)", st.size(), ref_state.size());
            printf("  plain  state %s\n", st == ref_state ? "EQUAL" : "DIFF");
            llama_free(X);
        }
        // inner nodes positional only, the tip whole
        {
            llama_context * X = occupied_ctx(model, other);
            CHECK(compose(X, { f0, f1, f2 }, { SKIP, NC | SKIP, NC }, toks), "skip compose failed");
            const auto st = seq_state(X);
            CHECK(st == ref_state, "skip compose differs from the saving context (%zu vs %zu bytes)", st.size(), ref_state.size());
            const auto lg = logits_after(X, toks);
            float worst = 0.0f;
            for (size_t i = 0; i < lg.size(); ++i) {
                worst = std::max(worst, std::fabs(lg[i] - ref_logits[i]));
            }
            CHECK(worst < 1e-3f, "skip compose: next-token logits differ from the reference by %g", worst);
            printf("  skip   state %s, logits max diff %g\n", st == ref_state ? "EQUAL" : "DIFF", worst);
            llama_free(X);
        }
        // positive control: no node loads the side-state, so the other conversation's stays
        {
            llama_context * X = occupied_ctx(model, other);
            const bool ok    = compose(X, { f0, f1, f2 }, { SKIP, NC | SKIP, NC | SKIP }, toks);
            const bool equal = ok && seq_state(X) == ref_state;
            printf("  control: every node positional only %s\n", !ok ? "failed" : equal ? "EQUAL" : "DIFF (the tip's side-state is load-bearing)");
            CHECK(ok && !equal, "skipping every side-state still gives the state at P, so this test cannot see a missing side-state");
            llama_free(X);
        }
    }

    std::remove(f0.c_str());
    std::remove(f1.c_str());
    std::remove(f2.c_str());
    llama_model_free(model);
    printf("checks: %d\n", g_checks);
    CHECK(g_checks >= 7, "only %d checks ran", g_checks);
    printf("RESULT %s: %s (%d failures)\n", g_model.c_str(), g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
