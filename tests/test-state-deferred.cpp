// test-state-deferred: a deferred sequence-state capture (llama_state_seq_save_deferred) must produce, byte
// for byte, the unit an immediate capture (llama_state_seq_save_sink, the stage-1 writer's path) produced at
// the same instant, however much later its positional bytes are copied, and whatever the cache does in the
// meantime.
//
// For one model (one memory class):
//   append   the context keeps decoding after the capture (new cells only), then the capture is emitted in
//            odd-sized pieces: equal to the immediate unit, and the hook never fired. Root and delta.
//   above    a seq_rm of positions past the captured range: the hook must NOT fire (it is range-precise).
//   for each mutation that can change or free captured cells (rm-tail, rm-all, clear, keep, add, div,
//   shift-other, add-shared, load-seq, load-file, load-whole, cp-stream, free, callback): capture, mutate,
//   overwrite what the mutation freed, emit: equal to the immediate unit, and the hook fired first.
//   shift-other shifts a DIFFERENT sequence: the K-shift graph still ropes the captured cells (by 0, which
//   is not the identity under YaRN, see --yarn), and add-shared shifts a sequence that shares the captured
//   cells through a same-stream seq_cp.
//   positive control: the same with the hook disabled (llama_state_deferred_set_hook_enabled(false)) must
//   give a DIFFERENT unit for every mutation that overwrites captured data, so the equality above is not
//   vacuous. It only applies when the class defers anything (a class with no positional cells copies all
//   of its state at the capture, so the hook has nothing to protect).
// The memory class decides what is positional, never the model: --expect-deferred 1|0 states what this
// class must do, so a class that silently stopped deferring (or started deferring side-state) fails.

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); printf("FAIL: " __VA_ARGS__); printf("\n"); g_fail++; } } while (0)

struct opts {
    std::string model;
    std::string tmpdir = ".";
    int n_node  = 96;   // the capture point (cells [0, n_node) are in the cache)
    int n_lo    = 40;   // the delta capture starts here
    int n_more  = 48;   // tokens decoded after the capture
    int expect_deferred = -1; // -1: do not check
    uint32_t n_ubatch = 64;
    bool yarn = false;        // YaRN rope scaling: a rope by 0 then scales K (mscale), so it is not the identity
};
static opts g_o;

static llama_context * make_ctx(llama_model * model, uint32_t n_seq_max, bool unified) {
    auto cp = llama_context_default_params();
    // room for the largest scenario: keep grows seq 0 to 2N next to the captured N, plus the decodes after
    const uint32_t n_need = 2u * (3u * g_o.n_node + g_o.n_more + 128u);
    cp.n_ctx           = std::max<uint32_t>(1024, (n_need + 255u) / 256u * 256u);
    cp.n_batch         = cp.n_ctx;
    cp.n_ubatch        = g_o.n_ubatch;
    cp.n_seq_max       = n_seq_max;
    cp.kv_unified      = unified;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    if (g_o.yarn) {
        cp.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
        cp.rope_freq_scale   = 0.25f;
        cp.yarn_ext_factor   = 1.0f;
        cp.yarn_orig_ctx     = 256;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create a context\n");
        exit(2);
    }
    return ctx;
}

static void decode(llama_context * ctx, const std::vector<llama_token> & toks, int p_begin, int p_end, int seq) {
    const int n = p_end - p_begin;
    if (n <= 0) {
        return;
    }
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i]     = toks[p_begin + i];
        b.pos[i]       = p_begin + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = seq;
        b.logits[i]    = i == n - 1;
    }
    b.n_tokens = n;
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    if (rc != 0) {
        fprintf(stderr, "decode [%d, %d) seq %d failed rc=%d\n", p_begin, p_end, seq, rc);
        exit(3);
    }
}

// a sink handing out odd-sized pieces, so plain bytes and tensor slices are split at arbitrary offsets
struct piece_sink {
    std::vector<uint8_t> out;
    size_t   cap = 0;
    uint32_t rng = 7;
};

static void * piece_reserve(void * ud, size_t n, size_t * n_avail) {
    auto * ps = (piece_sink *) ud;
    ps->rng = ps->rng * 1103515245u + 12345u;
    const size_t piece = std::min(n, (size_t) 1 + (ps->rng >> 8) % 4099);
    ps->cap = ps->out.size();
    ps->out.resize(ps->cap + piece);
    *n_avail = piece;
    return ps->out.data() + ps->cap;
}

static void piece_commit(void * ud, size_t n) {
    auto * ps = (piece_sink *) ud;
    ps->out.resize(ps->cap + n);
}

// the stage-1 capture
static std::vector<uint8_t> immediate(llama_context * ctx, int seq, int p0, const std::vector<llama_token> & toks, int n_tok) {
    piece_sink ps;
    const llama_state_sink sink = { piece_reserve, piece_commit, &ps };
    const size_t n = llama_state_seq_save_sink(ctx, seq, p0, -1, toks.data(), n_tok, &sink);
    CHECK(n == ps.out.size() && n > 0, "immediate capture failed (%zu)", n);
    return ps.out;
}

// emits in two halves with a pause between them (a sink that refuses = "no room now"), then the rest
static std::vector<uint8_t> emit_all(llama_state_deferred * d) {
    piece_sink ps;
    const llama_state_sink sink = { piece_reserve, piece_commit, &ps };
    bool done = false;
    const size_t half = llama_state_deferred_size(d) / 2;
    size_t n = llama_state_deferred_emit(d, &sink, std::max<size_t>(1, half), &done);
    // a refusing sink pauses emission without losing a byte
    const llama_state_sink refuse = { [](void *, size_t, size_t * a) -> void * { *a = 0; return nullptr; },
                                      [](void *, size_t) {}, nullptr };
    n += llama_state_deferred_emit(d, &refuse, 0, &done);
    while (!done) {
        const size_t k = llama_state_deferred_emit(d, &sink, 0, &done);
        if (k == 0 && !done) {
            CHECK(false, "emission made no progress");
            break;
        }
        n += k;
    }
    CHECK(n == ps.out.size() && n == llama_state_deferred_size(d), "emitted %zu, sink holds %zu, size %zu", n, ps.out.size(),
          llama_state_deferred_size(d));
    CHECK(llama_state_deferred_n_pending(d) == 0, "references remain after a full emission");
    return ps.out;
}

static size_t ndiff(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    size_t n = a.size() > b.size() ? a.size() - b.size() : b.size() - a.size();
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        n += a[i] != b[i];
    }
    return n;
}

struct mutation {
    const char * name;
    bool needs_two_seqs;
    bool non_unified;
    // applies the mutation to `ctx` after a capture of `seq`, then overwrites whatever it freed;
    // returns false when the class does not support it (skipped, reported)
    std::function<bool(llama_context * ctx, int seq)> apply;
    // the positive control must see a difference: the mutation frees or overwrites captured cells. A shift
    // (add, div) only rotates keys in place where the class re-ropes its cache, so its control is reported
    // but not required (measured: glm5-next's div leaves the captured bytes as they were)
    bool overwrites;
    // the hook must fire; false where the mutation only changes the captured bytes in some configurations
    // (shift-other ropes them by 0: a change under YaRN, a no-op that needs no force for plain f16 K, and
    // nothing at all for a class whose K-shift does not rope, as glm5-next measured)
    bool must_force = true;
};

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return argv[++i]; };
        if      (a == "-m")                g_o.model = next();
        else if (a == "--tmp")             g_o.tmpdir = next();
        else if (a == "--node")            g_o.n_node = std::atoi(next());
        else if (a == "--lo")              g_o.n_lo = std::atoi(next());
        else if (a == "--more")            g_o.n_more = std::atoi(next());
        else if (a == "--ubatch")          g_o.n_ubatch = std::atoi(next());
        else if (a == "--expect-deferred") g_o.expect_deferred = std::atoi(next());
        else if (a == "--yarn")            g_o.yarn = true;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    llama_log_set([](ggml_log_level lvl, const char * txt, void *) {
        if (lvl >= GGML_LOG_LEVEL_WARN) fputs(txt, stderr);
    }, nullptr);
    llama_backend_init();

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(g_o.model.c_str(), mp);
    if (!model) {
        fprintf(stderr, "model load failed\n");
        return 2;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> dis(0, n_vocab - 1);
    const int n_all = 2 * g_o.n_node + g_o.n_more + 64; // keep grows seq 0 to 2N
    std::vector<llama_token> toks(n_all), other(n_all);
    for (auto & t : toks)  t = dis(rng);
    for (auto & t : other) t = dis(rng);

    const int N  = g_o.n_node;
    const int LO = g_o.n_lo;
    printf("model %s  node %d  delta lo %d  more %d  ubatch %u%s\n", g_o.model.c_str(), N, LO, g_o.n_more, g_o.n_ubatch,
           g_o.yarn ? "  yarn" : "");

    // ---- append: root and delta captures, the context keeps decoding, then emission
    size_t n_def_root = 0, n_total_root = 0;
    for (const int p0 : { -1, LO }) {
        llama_context * X = make_ctx(model, 1, true);
        decode(X, toks, 0, LO, 0);
        decode(X, toks, LO, N, 0);
        const auto ref = immediate(X, 0, p0, toks, N);
        size_t n_def_q = 0;
        const size_t n_tot_q = llama_state_seq_get_size_deferred(X, 0, p0, -1, N, &n_def_q);
        llama_state_deferred * d = llama_state_seq_save_deferred(X, 0, p0, -1, toks.data(), N);
        CHECK(d != nullptr, "deferred capture failed");
        if (!d) {
            llama_free(X);
            continue;
        }
        CHECK(n_tot_q == ref.size() && llama_state_deferred_size(d) == ref.size(),
              "%s: sizes differ (query %zu, capture %zu, immediate %zu)", p0 < 0 ? "root" : "delta", n_tot_q,
              llama_state_deferred_size(d), ref.size());
        CHECK(n_def_q == llama_state_deferred_n_deferred(d), "deferred size query %zu vs capture %zu", n_def_q,
              llama_state_deferred_n_deferred(d));
        if (p0 < 0) {
            n_def_root   = llama_state_deferred_n_deferred(d);
            n_total_root = llama_state_deferred_size(d);
        }
        // the context keeps going: new cells only
        decode(X, toks, N, N + 8, 0);
        const auto got = emit_all(d);
        const size_t nd = ndiff(ref, got);
        printf("  append  %-5s %8zu B, %8zu deferred (%5.1f%%), forced %u: %s\n", p0 < 0 ? "root" : "delta",
               ref.size(), llama_state_deferred_n_deferred(d), 100.0 * llama_state_deferred_n_deferred(d) / ref.size(),
               llama_state_deferred_n_forced(d), nd == 0 ? "EQUAL" : "DIFF");
        CHECK(nd == 0, "append %s: %zu bytes differ from the immediate capture", p0 < 0 ? "root" : "delta", nd);
        CHECK(llama_state_deferred_n_forced(d) == 0, "append: the hook fired although only new cells were written");
        llama_state_deferred_free(d);
        llama_free(X);
    }
    // the test above captured, THEN decoded: redo it with the decode between capture and emission
    bool deferring = false;
    {
        llama_context * X = make_ctx(model, 1, true);
        decode(X, toks, 0, N, 0);
        const auto ref = immediate(X, 0, LO, toks, N);
        llama_state_deferred * d = llama_state_seq_save_deferred(X, 0, LO, -1, toks.data(), N);
        deferring = llama_state_deferred_n_pending(d) > 0;
        decode(X, toks, N, N + g_o.n_more, 0);
        // positions past the captured range: not the captured cells, so the hook must stay quiet (a class that
        // cannot rewind refuses the removal; it then just keeps decoding)
        if (llama_memory_seq_rm(llama_get_memory(X), 0, N, -1)) {
            decode(X, other, N, N + g_o.n_more, 0);
        } else {
            decode(X, other, N + g_o.n_more, N + g_o.n_more + 8, 0);
        }
        const auto got = emit_all(d);
        const size_t nd = ndiff(ref, got);
        printf("  later   delta %8zu B emitted after %d more tokens and an rm above the range, forced %u: %s\n", ref.size(),
               g_o.n_more, llama_state_deferred_n_forced(d), nd == 0 ? "EQUAL" : "DIFF");
        CHECK(nd == 0, "later: %zu bytes differ", nd);
        CHECK(llama_state_deferred_n_forced(d) == 0, "the hook fired for an rm above the captured range");
        llama_state_deferred_free(d);
        llama_free(X);
    }
    printf("  class defers %zu of %zu root bytes\n", n_def_root, n_total_root);
    if (g_o.expect_deferred >= 0) {
        CHECK((n_def_root > 0) == (g_o.expect_deferred > 0), "expected this class %s positional bytes, it deferred %zu",
              g_o.expect_deferred ? "to defer" : "to defer no", n_def_root);
    }

    // a saved state of the same sequence with other tokens, for the load mutations
    std::vector<uint8_t> other_seq, other_whole;
    const std::string other_file = g_o.tmpdir + "/other.bin";
    {
        llama_context * Y = make_ctx(model, 1, true);
        decode(Y, other, 0, N, 0);
        other_seq.resize(llama_state_seq_get_size(Y, 0));
        other_seq.resize(llama_state_seq_get_data(Y, other_seq.data(), other_seq.size(), 0));
        other_whole.resize(llama_state_get_size(Y));
        other_whole.resize(llama_state_get_data(Y, other_whole.data(), other_whole.size()));
        CHECK(llama_state_seq_save_file(Y, other_file.c_str(), 0, other.data(), N) > 0, "saving the other state");
        llama_free(Y);
    }

    auto mem = [](llama_context * c) { return llama_get_memory(c); };
    // rewrite [p0, N) of `seq` with other tokens (the cells the mutation freed)
    auto redecode = [&](llama_context * c, int seq, int p0) {
        decode(c, other, p0, N, seq);
    };

    std::vector<mutation> muts = {
        { "rm-tail", false, false, [&](llama_context * c, int seq) {
            const int cut = (LO + N) / 2;
            if (!llama_memory_seq_rm(mem(c), seq, cut, -1)) {
                return false; // a class that cannot rewind (the hook still ran, the bytes are checked)
            }
            redecode(c, seq, cut);
            return true; }, true },
        { "rm-all", false, false, [&](llama_context * c, int seq) {
            llama_memory_seq_rm(mem(c), seq, -1, -1);
            redecode(c, seq, 0);
            return true; }, true },
        { "clear", false, false, [&](llama_context * c, int seq) {
            llama_memory_clear(mem(c), true);
            redecode(c, seq, 0);
            return true; }, true },
        { "keep", true, false, [&](llama_context * c, int seq) {
            // the capture is on `seq` (1); keeping only seq 0 frees it, and seq 0 then grows into its cells
            llama_memory_seq_keep(mem(c), 0);
            decode(c, other, N, N + N, 0);
            (void) seq;
            return true; }, true },
        { "add", false, false, [&](llama_context * c, int seq) {
            if (!llama_memory_can_shift(mem(c))) {
                return false;
            }
            llama_memory_seq_add(mem(c), seq, LO, -1, 5);
            decode(c, other, N + 5, N + 6, seq); // the next decode applies the K-shift in place
            return true; }, false },
        { "div", false, false, [&](llama_context * c, int seq) {
            if (!llama_memory_can_shift(mem(c))) {
                return false;
            }
            llama_memory_seq_div(mem(c), seq, LO, -1, 2);
            const int p = llama_memory_seq_pos_max(mem(c), seq) + 1; // the next decode applies the shift
            decode(c, other, p, p + 1, seq);
            return true; }, false },
        { "shift-other", true, false, [&](llama_context * c, int seq) {
            // a context shift of the OTHER sequence (seq 0), as the server does on another slot: the K-shift
            // graph ropes the captured cells too (by 0)
            const int a = N / 4, b = N / 2;
            if (!llama_memory_can_shift(mem(c)) || b - a < 1) {
                return false;
            }
            (void) seq;
            if (!llama_memory_seq_rm(mem(c), 0, a, b)) {
                return false; // a class that cannot rewind cannot context-shift either
            }
            llama_memory_seq_add(mem(c), 0, b, -1, -(b - a));
            decode(c, other, N - (b - a), N - (b - a) + 1, 0); // applies the K-shift
            return true; }, g_o.yarn, g_o.yarn },
        { "add-shared", true, false, [&](llama_context * c, int seq) {
            // seq 0 takes the captured cells by a same-stream copy, then shifts: the shared cells move and
            // are re-roped for both sequences
            if (!llama_memory_can_shift(mem(c))) {
                return false;
            }
            llama_memory_seq_rm(mem(c), 0, -1, -1);
            llama_memory_seq_cp(mem(c), seq, 0, -1, -1);
            llama_memory_seq_add(mem(c), 0, LO, -1, 5);
            decode(c, other, N + 5, N + 6, 0); // applies the K-shift
            return true; }, false },
        { "load-seq", false, false, [&](llama_context * c, int seq) {
            return llama_state_seq_set_data(c, other_seq.data(), other_seq.size(), seq) > 0; }, true },
        { "load-file", false, false, [&](llama_context * c, int seq) {
            std::vector<llama_token> t(N + 16);
            size_t n_out = 0;
            return llama_state_seq_load_file(c, other_file.c_str(), seq, t.data(), t.size(), &n_out) > 0; }, true },
        { "load-whole", false, false, [&](llama_context * c, int seq) {
            (void) seq;
            return llama_state_set_data(c, other_whole.data(), other_whole.size()) > 0; }, true },
        { "cp-stream", true, true, [&](llama_context * c, int seq) {
            // per-sequence streams: copying seq 0 over the captured seq 1 overwrites its stream at the next update
            llama_memory_seq_cp(mem(c), 0, seq, -1, -1);
            decode(c, other, N, N + 1, 0);
            return true; }, true },
    };

    auto run = [&](const mutation & m, bool hook) -> int {
        // returns 1 equal, 0 differ, -1 skipped
        const int seq = m.needs_two_seqs ? 1 : 0;
        llama_context * X = make_ctx(model, m.needs_two_seqs ? 2 : 1, !m.non_unified);
        if (m.needs_two_seqs) {
            decode(X, other, 0, N, 0);
        }
        decode(X, toks, 0, N, seq);
        const auto ref = immediate(X, seq, LO, toks, N);
        llama_state_deferred * d = llama_state_seq_save_deferred(X, seq, LO, -1, toks.data(), N);
        llama_state_deferred_set_hook_enabled(hook);
        const bool ok = m.apply(X, seq);
        const uint32_t forced = llama_state_deferred_n_forced(d);
        llama_state_deferred_set_hook_enabled(true);
        int res = -1;
        if (ok) {
            const auto got = emit_all(d);
            const size_t nd = ndiff(ref, got);
            res = nd == 0 ? 1 : 0;
            if (hook) {
                printf("  %-10s forced %u: %s (%zu differing bytes)\n", m.name, forced, nd == 0 ? "EQUAL" : "DIFF ", nd);
                CHECK(nd == 0, "%s: the deferred unit differs from the immediate one by %zu bytes", m.name, nd);
                if (deferring && m.must_force) {
                    CHECK(forced >= 1, "%s: the hook did not force the pending capture", m.name);
                }
            }
        } else if (hook) {
            printf("  %-10s not supported by this class (skipped)\n", m.name);
        }
        llama_state_deferred_free(d);
        llama_free(X);
        return res;
    };

    for (const auto & m : muts) {
        const int r = run(m, true);
        if (r < 0 || !deferring) {
            continue;
        }
        const int c = run(m, false);
        printf("  %-10s positive control (hook disabled): %s%s\n", m.name,
               c == 0 ? "DIFF (the hook is load-bearing)" : c == 1 ? "EQUAL" : "skipped", m.overwrites ? "" : " (informational)");
        if (m.overwrites) {
            CHECK(c == 0, "%s: with the hook disabled the unit is still equal, so this test cannot see a missing flush", m.name);
        }
    }

    // ---- free: the context goes away with a capture pending, which then still emits the right bytes
    {
        llama_context * X = make_ctx(model, 1, true);
        decode(X, toks, 0, N, 0);
        const auto ref = immediate(X, 0, LO, toks, N);
        llama_state_deferred * d = llama_state_seq_save_deferred(X, 0, LO, -1, toks.data(), N);
        llama_free(X);
        const uint32_t forced = llama_state_deferred_n_forced(d);
        const auto got = emit_all(d);
        const size_t nd = ndiff(ref, got);
        printf("  %-10s forced %u: %s\n", "free", forced, nd == 0 ? "EQUAL" : "DIFF");
        CHECK(nd == 0, "free: %zu bytes differ", nd);
        if (deferring) {
            CHECK(forced >= 1, "free: freeing the context did not force the capture");
        }
        llama_state_deferred_free(d);
    }

    // ---- callback: the hook hands the capture to its owner, which emits it into its own sink first
    {
        llama_context * X = make_ctx(model, 1, true);
        decode(X, toks, 0, N, 0);
        const auto ref = immediate(X, 0, LO, toks, N);
        llama_state_deferred * d = llama_state_seq_save_deferred(X, 0, LO, -1, toks.data(), N);
        struct cb_state { piece_sink ps; int calls = 0; size_t pending_at_call = 0; } cs;
        llama_state_deferred_set_flush_cb(d, [](void * ud, llama_state_deferred * dd) -> bool {
            auto * c = (cb_state *) ud;
            c->calls++;
            c->pending_at_call = llama_state_deferred_n_pending(dd);
            const llama_state_sink sink = { piece_reserve, piece_commit, &c->ps };
            bool done = false;
            while (!done) {
                if (llama_state_deferred_emit(dd, &sink, 0, &done) == 0 && !done) {
                    return false;
                }
            }
            return true;
        }, &cs);
        llama_memory_seq_rm(llama_get_memory(X), 0, -1, -1);
        redecode(X, 0, 0);
        const size_t nd = ndiff(ref, cs.ps.out);
        printf("  %-10s calls %d (pending %zu at the call): %s\n", "callback", cs.calls, cs.pending_at_call, nd == 0 ? "EQUAL" : "DIFF");
        if (deferring) {
            CHECK(cs.calls == 1 && cs.pending_at_call > 0, "callback: called %d times with %zu pending", cs.calls, cs.pending_at_call);
            CHECK(nd == 0, "callback: %zu bytes differ", nd);
        }
        llama_state_deferred_free(d);
        llama_free(X);
    }

    std::remove(other_file.c_str());
    llama_model_free(model);
    printf("checks: %d\n", g_checks);
    CHECK(g_checks >= 20, "only %d checks ran", g_checks);
    printf("RESULT %s: %s (%d failures)\n", g_o.model.c_str(), g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
