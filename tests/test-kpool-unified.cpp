// test-kpool-unified: a sequence's output must not depend on another sequence sharing its cache.
//
// Ported as a behaviour test from two fixes on the retired glm5next branch, whose k-pool code the upstream
// glm5-next implementation replaced:
//   625fbcbad  the k-pools of one sequence picked up another sequence's cells on a unified cache (the default
//              llama-server layout with auto slots), so pools collided across sequences;
//   525a329bb  an incomplete trailing pool was padded with cell 0, which a query could then attend; with
//              another sequence at the start of a unified cache that is a foreign token.
// Both show up as the same symptom, which this test checks on every k-pool memory class: sequence A decoded
// next to sequence B (B first, so B owns the low cells, cell 0 included) on a unified cache must give the
// logits A gives alone, at every step of a prefill whose length leaves a trailing pool incomplete and of a
// token-by-token decode after it. The same holds on separate streams, the control.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); printf("FAIL: " __VA_ARGS__); printf("\n"); g_fail++; } } while (0)

static constexpr int N_A   = 45;  // not a multiple of any pool size, so A's trailing pool is incomplete
static constexpr int N_B   = 37;
static constexpr int N_GEN = 6;   // single-token steps after the prefill

static llama_context * make_ctx(llama_model * model, uint32_t n_seq, bool unified) {
    auto cp = llama_context_default_params();
    cp.n_ctx           = 512;
    cp.n_batch         = 512;
    cp.n_ubatch        = 32;  // the prefill of A spans two ubatches
    cp.n_seq_max       = n_seq;
    cp.kv_unified      = unified;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create a context\n");
        exit(2);
    }
    return ctx;
}

static std::vector<float> decode(llama_context * ctx, const std::vector<llama_token> & toks, int p0, int p1, int seq) {
    const int n = p1 - p0;
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i]     = toks[p0 + i];
        b.pos[i]       = p0 + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = seq;
        b.logits[i]    = i == n - 1;
    }
    b.n_tokens = n;
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    if (rc != 0) {
        fprintf(stderr, "decode [%d, %d) seq %d failed rc=%d\n", p0, p1, seq, rc);
        exit(3);
    }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * lg = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(lg, lg + nv);
}

// the logits of A after the prefill and after each generated step
static std::vector<std::vector<float>> run_a(llama_context * ctx, const std::vector<llama_token> & a, int seq) {
    std::vector<std::vector<float>> out;
    out.push_back(decode(ctx, a, 0, N_A, seq));
    for (int i = 0; i < N_GEN; ++i) {
        out.push_back(decode(ctx, a, N_A + i, N_A + i + 1, seq));
    }
    return out;
}

static float max_diff(const std::vector<std::vector<float>> & x, const std::vector<std::vector<float>> & y, int & at) {
    float worst = 0.0f;
    at = -1;
    for (size_t s = 0; s < x.size(); ++s) {
        for (size_t i = 0; i < x[s].size(); ++i) {
            const float d = std::fabs(x[s][i] - y[s][i]);
            if (d > worst) {
                worst = d;
                at = (int) s;
            }
        }
    }
    return worst;
}

int main(int argc, char ** argv) {
    std::string path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) {
            path = argv[++i];
        } else {
            fprintf(stderr, "usage: %s -m model.gguf\n", argv[0]);
            return 1;
        }
    }
    if (path.empty()) {
        fprintf(stderr, "no model\n");
        return 1;
    }

    llama_backend_init();
    llama_model * model = llama_model_load_from_file(path.c_str(), llama_model_default_params());
    if (!model) {
        fprintf(stderr, "failed to load %s\n", path.c_str());
        return 2;
    }

    std::mt19937 rng(42);
    std::vector<llama_token> a(N_A + N_GEN), b(N_B);
    for (auto & t : a) { t = 10 + (llama_token) (rng() % 90); }
    for (auto & t : b) { t = 10 + (llama_token) (rng() % 90); }

    // A alone
    llama_context * ref_ctx = make_ctx(model, 1, false);
    const auto ref = run_a(ref_ctx, a, 0);
    llama_free(ref_ctx);

    // instrument control first: A with its first token changed. The dummies' weights are tiny, so a token of
    // difference moves the logits by little; a foreign token reaching A's attention moves them by about as much.
    // The tolerance is a tenth of the control, which must itself be non-zero.
    float tol = 0.0f;
    {
        auto a2 = a;
        a2[0] = a[0] == 10 ? 11 : 10;
        llama_context * ctx = make_ctx(model, 1, false);
        const auto got = run_a(ctx, a2, 0);
        llama_free(ctx);
        int at = -1;
        const float worst = max_diff(ref, got, at);
        tol = worst / 10.0f;
        printf("%s: control, A with its first token changed: max logit diff %g (tolerance %g)\n", path.c_str(), worst, tol);
        CHECK(worst > 0.0f, "control: changing a token of A moved no logit at all");
    }

    for (const bool unified : { false, true }) {
        llama_context * ctx = make_ctx(model, 2, unified);
        decode(ctx, b, 0, N_B, 1);          // B first: on a unified cache it owns the low cells, cell 0 included
        const auto got = run_a(ctx, a, 0);
        llama_free(ctx);
        int at = -1;
        const float worst = max_diff(ref, got, at);
        printf("%s: A next to B, %s: max logit diff %g (step %d)\n", path.c_str(), unified ? "unified cache" : "separate streams", worst, at);
        CHECK(worst <= tol, "%s: A's logits depend on B (max diff %g at step %d)", unified ? "unified" : "separate", worst, at);
    }

    llama_model_free(model);
    printf("RESULT %s: %s (%d failures, %d checks)\n", path.c_str(), g_fail ? "FAIL" : "PASS", g_fail, g_checks);
    return g_fail ? 1 : 0;
}
