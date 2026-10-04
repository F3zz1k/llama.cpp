// test-delta-bytes: byte-level check that a disk-cache chain (whole root + range deltas composed with
// LLAMA_STATE_SEQ_FLAGS_NO_CLEAR) restores exactly the state a whole save restores.
//
// For one model:
//   A  (n_ctx = nctx_a) decodes the prompt in segments; after segment 0 it saves a whole root, after
//      every later segment a [prev_end, -1) range delta, and at the end a whole file
//   W  (nctx_a)  loads the whole file                               -> control
//   C  (nctx_a)  composes root + deltas                             -> must equal W byte for byte
//   D  (nctx_b)  composes root + deltas at another context size     -> cross-rung restore, must equal W
//   F  (nctx_b)  restores the root, extends it and writes its own deltas (a chain written on another rung)
//   G  (nctx_a)  composes F's chain                                 -> must equal F byte for byte
//   N  (nctx_a)  composes A's chain while a second sequence decodes between the nodes (interleaved,
//                fragmented cells; --noise, needs --seqmax 2)        -> must equal W byte for byte
//   NC (nctx_a)  control for N: DECODES the prompt segments under the same interleaving and holes, so
//                its logits carry the same cell-placement noise floor; N is judged against NC, not W
// Then every context decodes the same forced continuation; logits and final state blobs are compared.
// Controls: A2 repeats A exactly (determinism), A1 decodes the prompt in one call (order noise floor).
// W must equal A (blob and logits) when both use one sequence id, and G's logits must equal F's. A live
// context of a sliding-window model also holds cells the window already masks, which a save does not
// persist ([L - n_swa, L) only): there a restored context attends over fewer cells than the live one, the
// reduction order differs, and live-vs-restored logits agree to LIVE_SWA_TOL instead of bitwise (measured
// 2026-10-03: 0 for every class while the prompt fits the window, 1e-5..5e-4 past it, gemma3, dots3note,
// deepseek4; deepseek4's own decode order moves its logits by as much with no restore at all, A1 vs A up
// to 1.7e-4). Restored-vs-restored comparisons (C, D, N against W or NC) stay bitwise.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static int g_fail = 0;
static constexpr double LIVE_SWA_TOL = 2e-3;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); printf("FAIL: " __VA_ARGS__); printf("\n"); g_fail++; } } while (0)

struct opts {
    std::string model;
    std::string tmpdir = ".";
    uint32_t nctx_a = 1024;
    uint32_t nctx_b = 2048;
    uint32_t n_seq_max = 1;
    bool     unified = true;
    uint32_t n_ubatch = 512;
    std::vector<int> splits = {100, 200, 300};
    int n_gen = 8;
    uint32_t seed = 1234;
    int seq_a = 0; // the sequence A writes
    int seq_b = 0; // the sequence every restore loads into
    uint32_t n_rs = 0;   // n_rs_seq (rollback snapshots, what MTP serving sets)
    int rollback = 0;    // after the last segment, decode this many draft tokens and roll them back
    int noise = 0;       // tokens a second sequence decodes between chain nodes in context N
    bool holes = false;  // also free some of the other sequence's oldest cells so nodes land in holes
};
static opts g_o;

static std::vector<int> parse_ints(const char * s) {
    std::vector<int> r;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        r.push_back(std::atoi(tok.c_str()));
    }
    return r;
}

static llama_context * make_ctx(llama_model * model, const opts & o, uint32_t n_ctx) {
    auto cp = llama_context_default_params();
    cp.n_ctx       = n_ctx;
    cp.n_batch     = std::max<uint32_t>(o.n_ubatch, 2048);
    cp.n_ubatch    = o.n_ubatch;
    cp.n_seq_max   = o.n_seq_max;
    cp.kv_unified  = o.unified;
    cp.n_rs_seq    = o.n_rs;
    cp.n_threads   = 4;
    cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create context n_ctx=%u\n", n_ctx);
        exit(2);
    }
    return ctx;
}

static void decode(llama_context * ctx, const std::vector<llama_token> & toks, int p_begin, int p_end, int seq, int pos_off = 0) {
    const int n = p_end - p_begin;
    if (n <= 0) {
        return;
    }
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = toks[p_begin + i];
        b.pos[i] = pos_off + p_begin + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = seq;
        b.logits[i] = i == n - 1;
    }
    b.n_tokens = n;
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    if (rc != 0) {
        fprintf(stderr, "decode [%d, %d) seq %d failed rc=%d\n", p_begin, p_end, seq, rc);
        exit(3);
    }
}

static std::vector<uint8_t> blob(llama_context * ctx, int seq) {
    std::vector<uint8_t> out(llama_state_seq_get_size(ctx, seq));
    const size_t n = llama_state_seq_get_data(ctx, out.data(), out.size(), seq);
    out.resize(n);
    return out;
}

static void dump(const opts & o, const std::string & name, const std::vector<uint8_t> & b) {
    std::ofstream f(o.tmpdir + "/" + name + ".blob", std::ios::binary);
    f.write((const char *) b.data(), b.size());
}

// returns true when equal
static bool cmp_blob(const opts & o, const char * what, const std::vector<uint8_t> & ref, const std::vector<uint8_t> & got,
        const std::string & tag_ref, const std::string & tag_got) {
    size_t first = SIZE_MAX, ndiff = 0;
    const size_t n = std::min(ref.size(), got.size());
    for (size_t i = 0; i < n; ++i) {
        if (ref[i] != got[i]) {
            ndiff++;
            if (first == SIZE_MAX) {
                first = i;
            }
        }
    }
    const bool eq = ref.size() == got.size() && ndiff == 0;
    printf("  blob %-34s %s  (ref %zu B, got %zu B, %zu differing bytes, first at %lld)\n", what, eq ? "EQUAL" : "DIFF ",
            ref.size(), got.size(), ndiff, first == SIZE_MAX ? -1LL : (long long) first);
    if (!eq) {
        dump(o, tag_ref, ref);
        dump(o, tag_got, got);
    }
    return eq;
}

static std::vector<std::vector<float>> gen_forced(llama_context * ctx, const std::vector<llama_token> & cont, int n_past, int seq) {
    const llama_model * model = llama_get_model(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<std::vector<float>> res;
    for (size_t i = 0; i < cont.size(); ++i) {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0] = cont[i];
        b.pos[0] = n_past + (int) i;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = seq;
        b.logits[0] = 1;
        b.n_tokens = 1;
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "forced decode failed at %zu\n", i);
            exit(4);
        }
        llama_batch_free(b);
        const float * l = llama_get_logits_ith(ctx, -1);
        res.emplace_back(l, l + n_vocab);
    }
    return res;
}

static double max_abs_diff(const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[i].size(); ++j) {
            m = std::max(m, (double) std::fabs(a[i][j] - b[i][j]));
        }
    }
    return m;
}

static std::string per_token(const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b) {
    std::string s;
    char buf[32];
    for (size_t i = 0; i < a.size(); ++i) {
        double m = 0;
        for (size_t j = 0; j < a[i].size(); ++j) {
            m = std::max(m, (double) std::fabs(a[i][j] - b[i][j]));
        }
        snprintf(buf, sizeof(buf), " %.2g", m);
        s += buf;
    }
    return s;
}

// between(i) runs before node i is loaded (i >= 1) and once after the last node (i == files.size())
static void load_chain(llama_context * ctx, const std::vector<std::string> & files, int n_expected_last,
        const std::function<void(size_t)> & between = nullptr) {
    std::vector<llama_token> tok(65536);
    for (size_t i = 0; i < files.size(); ++i) {
        if (i > 0 && between) {
            between(i);
        }
        size_t n_out = 0;
        const llama_state_seq_flags fl = i == 0 ? 0 : LLAMA_STATE_SEQ_FLAGS_NO_CLEAR;
        const size_t r = llama_state_seq_load_file_ext(ctx, files[i].c_str(), g_o.seq_b, fl, tok.data(), tok.size(), &n_out);
        if (r == 0) {
            CHECK(false, "loading %s (flags %u) failed", files[i].c_str(), fl);
            return;
        }
    }
    if (between) {
        between(files.size());
    }
    const llama_pos pmax = llama_memory_seq_pos_max(llama_get_memory(ctx), g_o.seq_b);
    CHECK(pmax == n_expected_last - 1, "composed seq ends at %d, expected %d", pmax, n_expected_last - 1);
}

int main(int argc, char ** argv) {
    opts o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return argv[++i]; };
        if (a == "-m") o.model = next();
        else if (a == "--tmp") o.tmpdir = next();
        else if (a == "--nctx-a") o.nctx_a = std::atoi(next());
        else if (a == "--nctx-b") o.nctx_b = std::atoi(next());
        else if (a == "--seqmax") o.n_seq_max = std::atoi(next());
        else if (a == "--no-unified") o.unified = false;
        else if (a == "--ubatch") o.n_ubatch = std::atoi(next());
        else if (a == "--splits") o.splits = parse_ints(next());
        else if (a == "--gen") o.n_gen = std::atoi(next());
        else if (a == "--seed") o.seed = std::atoi(next());
        else if (a == "--seq-a") o.seq_a = std::atoi(next());
        else if (a == "--seq-b") o.seq_b = std::atoi(next());
        else if (a == "--rs") o.n_rs = std::atoi(next());
        else if (a == "--rollback") o.rollback = std::atoi(next());
        else if (a == "--noise") o.noise = std::atoi(next());
        else if (a == "--holes") o.holes = true;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (o.noise > 0 && o.n_seq_max < 2) {
        fprintf(stderr, "--noise needs --seqmax 2\n");
        return 1;
    }

    g_o = o;
    llama_log_set([](ggml_log_level lvl, const char * txt, void *) {
        if (lvl >= GGML_LOG_LEVEL_WARN) fputs(txt, stderr);
    }, nullptr);
    llama_backend_init();

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(o.model.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 2; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    const int n_prompt = o.splits.back();
    // the engine's window; llama_model_n_swa() reports 0 for deepseek4 on purpose (its SWA cannot serve
    // as a rollback), but its raw cache still masks and prunes, so read the hparam itself there
    int n_swa = llama_model_n_swa(model);
    if (n_swa == 0) {
        char arch[64] = {0};
        char key[128];
        char val[64] = {0};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        snprintf(key, sizeof(key), "%s.attention.sliding_window", arch);
        if (llama_model_meta_val_str(model, key, val, sizeof(val)) > 0) {
            n_swa = std::atoi(val);
        }
    }
    // live and restored contexts hold the same cells only when the window masked nothing
    const bool   swa_masked = n_swa > 0 && n_prompt > n_swa;
    const double tol_live   = swa_masked ? LIVE_SWA_TOL : 0.0;
    std::mt19937 rng(o.seed);
    std::uniform_int_distribution<int> dis(0, n_vocab - 1);
    std::vector<llama_token> toks(n_prompt + o.n_gen);
    for (auto & t : toks) t = dis(rng);
    const std::vector<llama_token> cont(toks.begin() + n_prompt, toks.end());
    std::vector<llama_token> noise_toks(4096);
    for (auto & t : noise_toks) t = dis(rng);

    printf("model %s  splits", o.model.c_str());
    for (int s : o.splits) printf(" %d", s);
    printf("  nctx_a %u nctx_b %u seqmax %u unified %d ubatch %u seq_a %d seq_b %d rs %u rollback %d noise %d holes %d n_swa %d%s\n",
            o.nctx_a, o.nctx_b, o.n_seq_max, o.unified, o.n_ubatch, o.seq_a, o.seq_b, o.n_rs, o.rollback, o.noise, (int) o.holes,
            n_swa, swa_masked ? " (prompt past the window: live vs restored within LIVE_SWA_TOL)" : "");

    const std::string pfx = o.tmpdir + "/chain";
    std::vector<std::string> files_a, files_g;

    // A's prompt schedule, reused verbatim by the A2 determinism control
    auto run_prompt = [&](llama_context * X, bool save) {
        int prev = 0;
        for (size_t s = 0; s < o.splits.size(); ++s) {
            if (o.rollback > 0 && s + 1 == o.splits.size()) {
                // an MTP verify step: one accepted token plus `rollback` drafts in one ubatch, all drafts
                // rejected and rolled back (the accepted token keeps the rollback plane inside the ubatch)
                decode(X, toks, prev, o.splits[s] - 1, o.seq_a);
                decode(X, toks, o.splits[s] - 1, o.splits[s] + o.rollback, o.seq_a);
                const bool ok = llama_memory_seq_rm(llama_get_memory(X), o.seq_a, o.splits[s], -1);
                if (save) printf("  rollback of %d draft tokens at %d: %s\n", o.rollback, o.splits[s], ok ? "ok" : "REFUSED");
                CHECK(ok, "rollback refused");
            } else {
                decode(X, toks, prev, o.splits[s], o.seq_a);
            }
            if (save) {
                const std::string f = pfx + ".a" + std::to_string(s) + ".bin";
                size_t n;
                if (s == 0) {
                    n = llama_state_seq_save_file(X, f.c_str(), o.seq_a, toks.data(), o.splits[s]);
                } else {
                    n = llama_state_seq_save_file_range(X, f.c_str(), o.seq_a, prev, -1, toks.data(), o.splits[s]);
                    const size_t n_whole = llama_state_seq_get_size(X, o.seq_a);
                    printf("  node %zu [%d,%d): delta file %zu B vs whole state %zu B (%.0f%%)%s\n", s, prev, o.splits[s], n, n_whole,
                            100.0*n/n_whole, n + 64 >= n_whole ? "   <-- NOT INCREMENTAL" : "");
                }
                CHECK(n > 0, "save node %zu", s);
                files_a.push_back(f);
            }
            prev = o.splits[s];
        }
    };

    // ---- A: reference, writes root + deltas + whole
    llama_context * A = make_ctx(model, o, o.nctx_a);
    run_prompt(A, true);
    const std::string fw = pfx + ".whole.bin";
    CHECK(llama_state_seq_save_file(A, fw.c_str(), o.seq_a, toks.data(), n_prompt) > 0, "save whole");
    const auto BA = blob(A, o.seq_a);

    // ---- W: whole-file control
    llama_context * W = make_ctx(model, o, o.nctx_a);
    load_chain(W, {fw}, n_prompt);
    const auto BW = blob(W, o.seq_b);
    if (o.seq_a == o.seq_b) {
        // a defect shared by the whole restore and the deltas would otherwise be invisible
        if (!cmp_blob(o, "W(whole restore) vs A", BA, BW, "A", "W")) g_fail++;
    }

    // ---- C: compose at the same n_ctx
    llama_context * C = make_ctx(model, o, o.nctx_a);
    load_chain(C, files_a, n_prompt);
    if (!cmp_blob(o, "C(compose) vs W", BW, blob(C, o.seq_b), "W", "C")) g_fail++;

    // ---- D: compose at another n_ctx
    llama_context * D = make_ctx(model, o, o.nctx_b);
    load_chain(D, files_a, n_prompt);
    if (!cmp_blob(o, "D(compose, nctx_b) vs W", BW, blob(D, o.seq_b), "W", "D")) g_fail++;

    // ---- R: a node that does not continue the sequence it is appended to must be refused by the cache
    //         itself, not only by the server's chain-contiguity check
    if (files_a.size() >= 3 && o.rollback == 0) {
        std::vector<llama_token> tok(65536);
        size_t n_out = 0;
        auto load = [&](llama_context * X, const std::string & f, llama_state_seq_flags fl) {
            return llama_state_seq_load_file_ext(X, f.c_str(), o.seq_b, fl, tok.data(), tok.size(), &n_out);
        };
        // the same delta twice: its cells overlap what the sequence already holds
        llama_context * R = make_ctx(model, o, o.nctx_a);
        CHECK(load(R, files_a[0], 0) > 0, "R: root");
        CHECK(load(R, files_a[1], LLAMA_STATE_SEQ_FLAGS_NO_CLEAR) > 0, "R: delta 1");
        const size_t r_dup = load(R, files_a[1], LLAMA_STATE_SEQ_FLAGS_NO_CLEAR);
        printf("  R: delta 1 appended twice                 %s\n", r_dup == 0 ? "REFUSED (ok)" : "ACCEPTED");
        CHECK(r_dup == 0, "a delta overlapping its base was accepted");
        llama_free(R);

        // a skipped node: the plain caches accept the gap (M-RoPE media legitimately spends fewer positions
        // than cells), the DeepSeek-V4 compressed sections must not, their rows would no longer line up
        char arch[64] = {0};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        if (std::string(arch) == "deepseek4") {
            llama_context * S = make_ctx(model, o, o.nctx_a);
            CHECK(load(S, files_a[0], 0) > 0, "S: root");
            const size_t r_skip = load(S, files_a[2], LLAMA_STATE_SEQ_FLAGS_NO_CLEAR);
            printf("  S: root + delta 2 (delta 1 skipped)      %s\n", r_skip == 0 ? "REFUSED (ok)" : "ACCEPTED");
            CHECK(r_skip == 0, "a DSV4 delta that does not continue its base was accepted");
            llama_free(S);
        }
    }

    // ---- N: compose while another sequence decodes between the nodes
    llama_context * N = nullptr;
    const int seq_noise = o.seq_b == 0 ? 1 : 0;
    int noise_pos = 0;
    if (o.noise > 0) {
        N = make_ctx(model, o, o.nctx_a);
        decode(N, noise_toks, 0, o.noise, seq_noise);
        noise_pos = o.noise;
        load_chain(N, files_a, n_prompt, [&](size_t i) {
            // grow the other sequence, then punch a hole in it so the next node lands in fragmented cells
            decode(N, noise_toks, noise_pos, noise_pos + o.noise, seq_noise);
            noise_pos += o.noise;
            if (o.holes && i < files_a.size()) {
                // free a few of its oldest cells: rm [0, k) is legal on attention caches; recurrent/DSV4
                // caches may refuse a non-tail removal, which only means less fragmentation
                llama_memory_seq_rm(llama_get_memory(N), seq_noise, 0, std::min(o.noise/2, 8));
            }
        });
        // with holes the sequence's cells are no longer in position order, so its blob lists the same
        // cells in another order: only the logits below are a valid criterion then
        if (!cmp_blob(o, "N(compose, interleaved) vs W", BW, blob(N, o.seq_b), "W", "N") && !o.holes) g_fail++;
    }
    // ---- NC: N's control, the same interleaving and holes around DECODED prompt segments
    llama_context * NC = nullptr;
    if (o.noise > 0 && o.rollback == 0) {
        NC = make_ctx(model, o, o.nctx_a);
        decode(NC, noise_toks, 0, o.noise, seq_noise);
        int nc_noise_pos = o.noise;
        int prev = 0;
        for (size_t s = 0; s < o.splits.size(); ++s) {
            decode(NC, toks, prev, o.splits[s], o.seq_b);
            prev = o.splits[s];
            decode(NC, noise_toks, nc_noise_pos, nc_noise_pos + o.noise, seq_noise);
            nc_noise_pos += o.noise;
            if (o.holes && s + 1 < o.splits.size()) {
                llama_memory_seq_rm(llama_get_memory(NC), seq_noise, 0, std::min(o.noise/2, 8));
            }
        }
    }

    // ---- rung chain: an nctx_b instance restores the root, extends it and writes the deltas, then an
    //      nctx_a instance composes that chain
    std::vector<uint8_t> BF;
    std::vector<std::vector<float>> LF;
    {
        llama_context * F = make_ctx(model, o, o.nctx_b);
        load_chain(F, {files_a[0]}, o.splits[0]);
        files_g.push_back(files_a[0]);
        int prev = o.splits[0];
        for (size_t s = 1; s < o.splits.size(); ++s) {
            if (o.rollback > 0 && s + 1 == o.splits.size()) {
                decode(F, toks, prev, o.splits[s] - 1, o.seq_b);
                decode(F, toks, o.splits[s] - 1, o.splits[s], o.seq_b);
            } else {
                decode(F, toks, prev, o.splits[s], o.seq_b);
            }
            const std::string f = pfx + ".f" + std::to_string(s) + ".bin";
            CHECK(llama_state_seq_save_file_range(F, f.c_str(), o.seq_b, prev, -1, toks.data(), o.splits[s]) > 0, "F save %zu", s);
            files_g.push_back(f);
            prev = o.splits[s];
        }
        BF = blob(F, o.seq_b);
        LF = gen_forced(F, cont, n_prompt, o.seq_b);
        llama_free(F);
    }
    llama_context * G = make_ctx(model, o, o.nctx_a);
    load_chain(G, files_g, n_prompt);
    const auto BG = blob(G, o.seq_b);
    if (!cmp_blob(o, "G(chain written @nctx_b) vs F", BF, BG, "F", "G")) {
        g_fail++;
        dump(o, "W", BW);
        cmp_blob(o, "F vs W (numeric only, F recomputed)", BW, BF, "W", "F");
        cmp_blob(o, "G vs W", BW, BG, "W", "G");
    }

    // ---- controls
    llama_context * A2 = make_ctx(model, o, o.nctx_a);
    run_prompt(A2, false);
    llama_context * A1 = nullptr;
    if (o.rollback == 0 && n_prompt <= (int) o.n_ubatch * 4) {
        A1 = make_ctx(model, o, o.nctx_a);
        decode(A1, toks, 0, n_prompt, o.seq_a);
    }

    // ---- forced continuation on every context
    const auto LA  = gen_forced(A,  cont, n_prompt, o.seq_a);
    const auto LA2 = gen_forced(A2, cont, n_prompt, o.seq_a);
    const auto LW  = gen_forced(W,  cont, n_prompt, o.seq_b);
    printf("  logits A2 vs A (determinism):      max %.3g  per token:%s\n", max_abs_diff(LA, LA2), per_token(LA, LA2).c_str());
    CHECK(max_abs_diff(LA, LA2) == 0, "the run is not deterministic");
    if (A1) {
        const auto LA1 = gen_forced(A1, cont, n_prompt, o.seq_a);
        printf("  logits A1 vs A (one-shot prompt):  max %.3g  per token:%s\n", max_abs_diff(LA, LA1), per_token(LA, LA1).c_str());
    }
    printf("  logits W vs A (whole restore):     max %.3g  per token:%s\n", max_abs_diff(LA, LW), per_token(LA, LW).c_str());
    if (o.seq_a == o.seq_b) {
        CHECK(max_abs_diff(LA, LW) <= tol_live, "a whole restore's logits differ from the live context by %.3g (tolerance %.3g)",
              max_abs_diff(LA, LW), tol_live);
    }
    // N under --noise is judged against NC (same interleaving, decoded): the placement of its cells
    // changes the reduction order, so bitwise equality with the unfragmented W is not expected
    struct { const char * name; llama_context * ctx; bool must; } others[] = { {"C", C, true}, {"D", D, true}, {"G", G, false}, {"N", N, NC == nullptr} };
    std::vector<std::vector<float>> LN;
    for (auto & x : others) {
        if (!x.ctx) continue;
        const auto L = gen_forced(x.ctx, cont, n_prompt, o.seq_b);
        const double d = max_abs_diff(LW, L);
        printf("  logits %s vs W:                     max %.3g%s  per token:%s\n", x.name, d, d == 0 ? " (bitwise)" : "", per_token(LW, L).c_str());
        if (x.must) {
            CHECK(d == 0, "%s logits differ from a whole restore by %.3g", x.name, d);
        }
        if (x.ctx == G) {
            // G composes the chain F wrote: it must continue exactly as F does
            const double dg = max_abs_diff(LF, L);
            printf("  logits G vs F (its writer):         max %.3g%s\n", dg, dg == 0 ? " (bitwise)" : "");
            CHECK(dg <= tol_live, "G logits differ from F, the live context that wrote its chain, by %.3g (tolerance %.3g)", dg, tol_live);
        }
        if (x.ctx == N) {
            LN = L;
        }
    }
    if (NC) {
        const auto LNC = gen_forced(NC, cont, n_prompt, o.seq_b);
        const double floor = max_abs_diff(LW, LNC);
        const double dn    = max_abs_diff(LNC, LN);
        printf("  logits NC vs W (interleaving floor): max %.3g\n", floor);
        printf("  logits N vs NC (compose vs decode):  max %.3g%s\n", dn, dn == 0 ? " (bitwise)" : "");
        CHECK(dn == 0 || (dn <= 2*floor && max_abs_diff(LW, LN) <= 2*floor),
              "N logits differ from the decoded control NC by %.3g (floor %.3g)", dn, floor);
    }
    const auto BW2 = blob(W, o.seq_b);
    if (o.seq_a == o.seq_b) {
        // after generation the masked-cell difference reaches the new cells' K/V too
        if (!cmp_blob(o, "after-gen W vs A", blob(A, o.seq_a), BW2, "A2", "W2") && !swa_masked) g_fail++;
    }
    if (!cmp_blob(o, "after-gen C vs W", BW2, blob(C, o.seq_b), "W2", "C2")) g_fail++;
    if (!cmp_blob(o, "after-gen D vs W", BW2, blob(D, o.seq_b), "W2", "D2")) g_fail++;
    // N generated over interleaved cells: its new K/V carry the interleaving's reduction order, so after
    // generation it is judged by the logits against NC above; without NC it must still equal W
    if (N && !cmp_blob(o, "after-gen N vs W", BW2, blob(N, o.seq_b), "W2", "N2") && !o.holes && !NC) g_fail++;

    for (auto * c : {A, A2, A1, W, C, D, G, N, NC}) if (c) llama_free(c);
    llama_model_free(model);

    printf("RESULT %s: %s (%d failures)\n", o.model.c_str(), g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
