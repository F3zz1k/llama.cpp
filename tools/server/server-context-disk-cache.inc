// fork-only: the automatic disk KV cache's members of server_context_impl (--slot-save-auto).
//
// Not a header: server-context.cpp includes this file once, inside the body of server_context_impl (private
// section), so everything here is a member of that struct and sees its other members. The free helpers and
// types it uses are in server-disk-cache.h and server-disk-cache-common.h.

    // gguf-header hash of the loaded mmproj file, computed once at load whenever
    // --mmproj is set (0 on a text-only server); see mmproj_header_fingerprint.
    uint64_t fp_mmproj = 0;

    // ~server_context_impl(): the writer touches the index and the counters; whatever it still holds is abandoned
    void disk_cache_destruct() {
        aw_stop(ggml_time_ms());
        aw_def_abandon_all(); // before destroy() frees the contexts the captures reference
    }

    // P0.1 (anchored-resume): does THIS model's memory class actually honour a position range in
    // state_write_range? The base-class default in src/llama-memory.h IGNORES [p0,p1) and writes the
    // WHOLE sequence. Every memory class that can compose a delta overrides it (the plain, iSWA, hybrid,
    // hybrid-iSWA, hybrid_idx, DSA, DSA-iSWA, MSA and DSV4 caches as of this writing), but a class added
    // later may not, and the probe is what keeps such a class on whole roots. On a
    // non-overriding class a "delta node" silently contains [0,N); composing root [0,lo) + that node
    // under NO_CLEAR yields DUPLICATE cells per position => wrong attention, silently.
    // We cannot ask the engine (no API reports what it serialised) and we cannot sniff the blob
    // (composite classes concatenate independently-narrowing sections). So we PROBE, once, on the
    // first delta write, against real resident state: write the range AND the whole sequence and
    // compare byte counts. Probe-don't-declare mirrors common_context_can_seq_rm, which likewise
    // decodes and tries the operation rather than trusting a class predicate.
    // Fails CLOSED: unknown/no => never write deltas, only whole roots at anchors.
    enum class delta_cap : uint8_t { unknown, yes, no };
    delta_cap delta_capable = delta_cap::unknown;

    // F1: on a class whose probe said NO, every parented save is published as a whole root instead of
    // a delta. That is a degradation (more bytes per node), so it is never silent: one WRN per
    // instance plus a cumulative counter (llamacpp:auto_cache_save_whole_fallback_total).
    bool auto_save_whole_fallback_warned = false;
    void auto_save_note_whole_fallback(const server_slot & slot, size_t n_tokens, uint32_t parent_hi) {
        metrics.n_auto_save_whole_fallback++;
        if (!auto_save_whole_fallback_warned) {
            auto_save_whole_fallback_warned = true;
            SLT_WRN(slot, "auto-save: this memory type cannot write deltas (probe said NO); publishing "
                          "parented saves as WHOLE roots instead (this one: %zu tokens, parent at %u). "
                          "Logged once; see llamacpp:auto_cache_save_whole_fallback_total\n",
                    n_tokens, parent_hi);
        }
    }

    // Auto-save counters the background writer updates too. They live outside server_metrics, which is
    // copied as a plain struct into every metrics result, and are folded into that copy when it is taken
    // (auto_metrics_fill).
    struct auto_save_counters {
        std::atomic<uint64_t> failed{0}, root{0}, delta{0}, bytes{0}, evicted{0}, draft{0}, draft_skipped{0};
        std::atomic<uint64_t> bound_exceeded{0}, ckpt_superseded{0};
        std::atomic<uint64_t> queued{0}, streamed{0}, dropped_staging{0}, orphan_dropped{0}, shutdown_abandoned{0};
        std::atomic<uint64_t> admission_waits{0}, admission_wait_us{0};
        std::atomic<uint64_t> deferred{0}, deferred_forced{0}, deferred_bytes{0};
        std::atomic<uint64_t> site_requested[AUTO_SAVE_SITE_COUNT] = {};
        std::atomic<uint64_t> site_published[AUTO_SAVE_SITE_COUNT] = {};
    } aw_cnt;

    // A save that is DROPPED (nothing published) is never silent either: a WRN naming the reason,
    // rate-limited to one line per minute so a full disk cannot flood the journal, plus a cumulative
    // counter (llamacpp:auto_cache_save_failed_total). Generation is unaffected (invariant 4). Called from
    // the server thread and from the writer.
    std::atomic<int64_t> auto_save_fail_last_wrn_ms{-1};
    void auto_save_note_failure(int slot_id, const char * reason, size_t n_tokens) {
        const uint64_t n_failed = ++aw_cnt.failed;
        const int64_t  now_ms   = ggml_time_ms();
        int64_t last = auto_save_fail_last_wrn_ms.load();
        if ((last < 0 || now_ms - last >= 60 * 1000) && auto_save_fail_last_wrn_ms.compare_exchange_strong(last, now_ms)) {
            SRV_WRN("slot %d: auto-save: dropped a %zu-token save: %s (dropped so far = %" PRIu64 "; "
                    "this line is rate-limited to one per minute, see llamacpp:auto_cache_save_failed_total)\n",
                    slot_id, n_tokens, reason, n_failed);
        }
    }
    void auto_save_note_failure(const server_slot & slot, const char * reason, size_t n_tokens) {
        auto_save_note_failure(slot.id, reason, n_tokens);
    }

    // the metrics result: the plain struct plus the counters the writer shares
    void auto_metrics_fill(server_metrics & m) {
        m.n_auto_save_failed             = aw_cnt.failed.load();
        m.n_auto_save_root               = aw_cnt.root.load();
        m.n_auto_save_delta              = aw_cnt.delta.load();
        m.n_auto_save_bytes              = aw_cnt.bytes.load();
        m.n_auto_cache_evicted          += aw_cnt.evicted.load();
        m.n_auto_cache_bound_exceeded    = aw_cnt.bound_exceeded.load();
        m.n_auto_ckpt_superseded         = aw_cnt.ckpt_superseded.load();
        m.auto_delta_capable             = delta_capable == delta_cap::yes ? 1 : delta_capable == delta_cap::no ? 2 : 0;
        m.n_auto_save_draft              = aw_cnt.draft.load();
        m.n_auto_save_draft_skipped      = aw_cnt.draft_skipped.load();
        m.n_auto_save_queued             = aw_cnt.queued.load();
        m.n_auto_save_streamed           = aw_cnt.streamed.load();
        m.n_auto_save_dropped_staging    = aw_cnt.dropped_staging.load();
        m.n_auto_save_orphan_dropped     = aw_cnt.orphan_dropped.load();
        m.n_auto_save_shutdown_abandoned = aw_cnt.shutdown_abandoned.load();
        m.n_auto_save_admission_waits    = aw_cnt.admission_waits.load();
        m.n_auto_save_admission_wait_us  = aw_cnt.admission_wait_us.load();
        for (int i = 0; i < AUTO_SAVE_SITE_COUNT; i++) {
            m.n_auto_save_site_requested[i] = aw_cnt.site_requested[i].load();
            m.n_auto_save_site_published[i] = aw_cnt.site_published[i].load();
        }
        m.n_auto_save_deferred           = aw_cnt.deferred.load();
        m.n_auto_save_deferred_forced    = aw_cnt.deferred_forced.load();
        m.n_auto_save_deferred_bytes     = aw_cnt.deferred_bytes.load();
        std::lock_guard<std::mutex> lk(aw.mtx);
        m.n_auto_save_staging_bytes      = aw.staged;
        m.n_auto_save_queue_depth        = aw.jobs.size();
        m.n_auto_save_deferred_host      = aw.deferred_host;
        m.n_auto_save_deferred_pending   = aw.def_count;
    }

    // Disk-restore accounting, at the point where the request's cached prefix is final. A restore whose
    // tokens the prompt did not keep (a later n_past clamp re-prefilled them) is a MISS, not a hit:
    // llamacpp:auto_cache_restore_discarded_total counts those, each with a WRN.
    void auto_restore_account(server_slot & slot, int32_t n_past) {
        const int32_t n     = slot.n_auto_restored_pending;
        const int32_t n_ram = slot.n_ram_loaded_pending;
        slot.n_auto_restored_pending = 0;
        slot.n_ram_loaded_pending    = 0;
        // a disk restore replaces whatever the RAM cache loaded, so only one of the two can be the source
        slot.stats.n_prompt_cached_ram = n <= 0 && n_ram > 0 ? (uint64_t) std::max(0, std::min(n_ram, n_past)) : 0;
        if (n <= 0) {
            return;
        }
        const int32_t kept = std::max(0, std::min(n, n_past));
        slot.stats.n_prompt_cached_disk = (uint64_t) kept;
        if (kept > 0) {
            slot.stats.cache_disk_unit_n = slot.disk_unit_pending;
            slot.stats.cache_disk_nodes  = slot.disk_nodes_pending;
            slot.stats.cache_disk_mode   = slot.disk_mode_pending;
            if (slot.disk_mode_pending == 2) {
                metrics.n_auto_restore_side_only++;
            }
        }
        if (kept > 0) {
            metrics.n_auto_restore_hit++;
            metrics.n_auto_restore_tokens += (uint64_t) kept;
        } else {
            metrics.n_auto_restore_miss++;
            metrics.n_auto_restore_discarded++;
            SLT_WRN(slot, "auto-restore: the %d restored tokens were discarded before use (n_past = %d), "
                          "the prompt is re-prefilled\n", n, n_past);
        }
    }

    // DRAFT SIDECAR (see slot_draft_sidecar_path). Plans the draft context's cells for [cell_lo, N) of this
    // slot, bounded by POSITION on both ends: the draft may hold drafted-but-rejected cells past the
    // target's end, and those must never reach a sidecar or a later delta would duplicate them. Kept only if
    // it holds at least one cell and no more cells than the target range it shadows (a memory type that
    // ignores the range would write every cell, which a NO_CLEAR compose would duplicate), checked from the
    // size query before anything is copied. Returns the range, the file size and the carry-over trailer.
    // Best-effort: a skipped sidecar only means that unit restores with a cold draft.
    bool auto_draft_sidecar_plan(const server_slot & slot, const llama_tokens & snap_toks, uint32_t cell_lo, size_t hdr,
                                 llama_pos & p0, llama_pos & p1, size_t & n_bytes, std::vector<uint8_t> & trailer) {
        const size_t n_cells = snap_toks.size();
        p0 = cell_lo > 0 ? slot.prompt.tokens.pos_next((int64_t) cell_lo) : -1;
        p1 = slot.prompt.tokens.pos_next((int64_t) n_cells);
        // the draft must cover the unit up to its last cell (a save cut while the draft still lags the
        // target, e.g. mid-prefill, would otherwise publish a sidecar that ends short of its unit)
        if (llama_memory_seq_pos_max(llama_get_memory(ctx_dft), slot.id) < p1 - 1) {
            aw_cnt.draft_skipped++;
            SLT_DBG(slot, "auto-save: no draft sidecar for [%u, %zu), the draft has not reached pos %d\n",
                    cell_lo, n_cells, (int) (p1 - 1));
            return false;
        }
        std::vector<uint8_t> head(4096);
        const size_t np = llama_state_seq_get_size_range(ctx_dft, slot.id, p0, p1, head.data(), head.size());
        uint32_t cells = 0;
        const bool ok = np > 0 &&
                        state_payload_cell_count(head.data(), std::min(head.size(), np), cells) &&
                        cells > 0 && (size_t) cells <= n_cells - cell_lo;
        if (!ok) {
            aw_cnt.draft_skipped++;
            SLT_DBG(slot, "auto-save: no draft sidecar for [%u, %zu) (%zu B, %u cells)\n", cell_lo, n_cells, np, cells);
            return false;
        }
        n_bytes = hdr + np;
        // the draft's carry-over for the unit's last cell (MTP: h of that cell, which the first cell
        // decoded after a restore pairs with); without it that cell is built from a stale h.
        // The trailer is stamped with the position the blob ITSELF records, and written only when that
        // is the unit's last cell: a carry-over that lags the target tail would otherwise be labelled as
        // matching it and applied on restore (the restore compares the stamp with the target tail, and
        // set_state adopts the blob's pos unchecked). Without a trailer the draft KV still restores; only
        // the first suffix cell is built without the carry-over. Layout (slot_draft_trailer_read):
        // [blob][i32 tail_pos][u32 blob_size][u32 SLOT_DRAFT_TRAILER_MAGIC].
        trailer.clear();
        std::vector<uint8_t> st;
        llama_pos st_pos = -1;
        if (common_speculative_get_state_at(spec.get(), slot.id, st, st_pos)) {
            if (st_pos != p1 - 1) {
                SLT_DBG(slot, "auto-save: draft carry-over is at pos %d, not the unit tail %d; sidecar written without it\n",
                        (int) st_pos, (int) (p1 - 1));
            } else {
                const uint32_t n = (uint32_t) st.size();
                trailer = st;
                const auto put = [&](const void * v, size_t k) {
                    trailer.insert(trailer.end(), (const uint8_t *) v, (const uint8_t *) v + k);
                };
                put(&st_pos, sizeof(st_pos));
                put(&n, sizeof(n));
                put(&SLOT_DRAFT_TRAILER_MAGIC, sizeof(SLOT_DRAFT_TRAILER_MAGIC));
            }
        }
        return true;
    }

    // Loads the draft sidecars of a restored chain into ctx_dft, in the same order and with the same
    // NO_CLEAR composition as the target. All or nothing: a chain with any node missing its sidecar, or
    // any load failing, leaves the draft empty (cold), which is exactly what every restore did before
    // sidecars existed. Called after the target chain has loaded.
    void auto_restore_draft_chain(server_slot & slot, const std::vector<std::string> & node_paths) {
        llama_memory_t mem_dft = llama_get_memory(ctx_dft);
        bool warm = !node_paths.empty();
        for (const std::string & node : node_paths) {
            std::error_code ec;
            if (!std::filesystem::exists(slot_draft_sidecar_path(node), ec)) {
                warm = false;
                break;
            }
        }
        if (warm) {
            llama_tokens scratch(slot.n_ctx);
            for (size_t i = 0; i < node_paths.size(); ++i) {
                size_t n_out = 0;
                const llama_state_seq_flags flags = (i == 0) ? 0 : LLAMA_STATE_SEQ_FLAGS_NO_CLEAR;
                if (llama_state_seq_load_file_ext(ctx_dft, slot_draft_sidecar_path(node_paths[i]).c_str(), slot.id,
                                                  flags, scratch.data(), scratch.size(), &n_out) == 0) {
                    warm = false;
                    break;
                }
            }
        }
        if (warm) {
            // the draft must never run ahead of the target it drafts for
            const llama_pos tgt_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
            if (llama_memory_seq_pos_max(mem_dft, slot.id) > tgt_max &&
                !llama_memory_seq_rm(mem_dft, slot.id, tgt_max + 1, -1)) {
                warm = false;
            }
            // nor end short of it: a draft missing its last cells would draft over a hole that the
            // speculative begin() checks cannot see (they look at pos_max only)
            if (warm && llama_memory_seq_pos_max(mem_dft, slot.id) != tgt_max) {
                warm = false;
            }
        }
        if (warm) {
            metrics.n_auto_restore_draft_warm++;
        } else {
            llama_memory_seq_rm(mem_dft, slot.id, -1, -1);
            metrics.n_auto_restore_draft_cold++;
        }
        // The carry-over of the tip's last cell, independent of the draft KV: a cold draft still pairs
        // the first suffix cell with it. Applied only when the tip's tail is the restored target tail;
        // a draft type or width that does not match ignores it (set_state checks the blob).
        llama_pos tail = -1;
        std::vector<uint8_t> st;
        if (spec && !node_paths.empty() &&
            slot_draft_trailer_read(slot_draft_sidecar_path(node_paths.back()), tail, st) &&
            tail == llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id)) {
            common_speculative_set_state(spec.get(), slot.id, st);
        }
    }

    // The value the ENGINE masks with (llama_model_n_swa), NEVER zeroed by --swa-full. Every disk
    // save/restore soundness decision must use THIS, not n_swa: --swa-full enlarges the SWA cache but
    // does not disable masking, so saves stay windowed even when n_swa above has been zeroed.
    int32_t n_swa_mem = 0;
    // the memory type can load a unit's side-state alone (LLAMA_STATE_SEQ_FLAGS_SKIP_POSITIONAL): restore mode 2
    bool side_only_supported = false;
    // the memory type can load a unit's positional cells alone (LLAMA_STATE_SEQ_FLAGS_SKIP_SIDE): a chain's inner
    // nodes read past their side-states, which the tip's replaces
    bool side_skip_supported = false;

    // Candidates the current restore attempt passed over because the request diverges inside them
    // and this memory class cannot rewind into a snapshot (the "must extend" rule). Reset by the
    // restore site before its lookup, bumped by auto_index_lookup's filter and by the whole-prefix
    // gate in auto_restore_into_slot, read back to classify a miss.
    size_t auto_not_prefix_skips = 0;
    int64_t auto_shared_last_wrn_ms = -1; // rate limit of the shared-prefix skip WRN

    // --- auto disk prompt/KV cache (opt-in: --slot-save-auto) ---
    // Default-constructed: empty and untouched when the feature is OFF (invariant 1).
    // `cur_fp` is the live model fingerprint, computed once at load (only when enabled).
    auto_cache_index auto_idx;
    model_fp         cur_fp;

    // The ONE gate for the entire auto disk cache. When false, NO hook below does
    // any work (no scan, no hash, no alloc). This is invariant 1: the first
    // statement of every auto_* hook is `if (!auto_cache_enabled()) return;`.
    bool auto_cache_enabled() const {
        return params_base.slot_save_auto && !params_base.slot_save_path.empty();
    }

    // Can a loaded snapshot of length L only ever be reused as a WHOLE prefix, never trimmed back
    // to a shorter verified prefix v < L? Reuse lengths are block-aligned, so any v < L is at least
    // one whole block short, and that is outside what these classes can do:
    //   FULL - llama_memory_seq_rm refuses ANY partial range, so the legal resume points are
    //          {0, L} and nothing in between.
    //   RS   - partial seq_rm is legal only within llama_n_rs_seq(ctx) positions of the tail
    //          (llama-memory-recurrent.cpp: the rollback must be in [1, n_rs_seq], typically 2-5;
    //          outside it seq_rm returns false and common_context_seq_rm turns that into a
    //          GGML_ABORT). That window is NOT merely un-exploited here, it is UNSOUND after a
    //          restore, and this is a correctness requirement rather than conservatism: state_write
    //          serializes only ONE row per cell (llama-memory-recurrent.cpp, cell_id = rs_idx_cur *
    //          size + src) and state_read unconditionally resets rs_idx to 0, so the (1 + n_rs_seq)
    //          rollback history rows a restored sequence would rewind INTO are NOT RESTORED. They are
    //          not zeroed either: they hold whatever that memory last had, i.e. plausible-looking
    //          state from a previous conversation, which is why the failure is silent rather than
    //          obviously broken.
    //          seq_rm's rollback branch only range-checks `1 <= rollback <= n_rs_seq`; it does not
    //          check the row exists, so such a rewind returns TRUE and reads an unwritten row:
    //          silent garbage, not a GGML_ABORT that would announce itself. Do NOT "optimise" this
    //          by bounding --slot-save-block below n_rs_seq: the block size is not what makes it
    //          unsafe. RS is whole-prefix-only unconditionally.
    //   SWA  - llama_kv_cache::state_write DROPS every SWA-masked cell at save time, so a snapshot
    //          of length L persists only the window [L - n_swa_mem, L): a shorter prefix has no
    //          cells behind it and re-decoding from there would attend over a hole.
    //   NO   - the capability probe itself failed, so per-token rewind is NOT known to work. This is
    //          a LIVE serving state, not a dead one: common_context_can_seq_rm returns NO both for a
    //          context with no memory module and for one whose 2-token probe decode merely failed
    //          (common/common.cpp), and the server only logs "speculative decoding not supported by
    //          this context" and carries on loading. Before this was listed, NO with n_swa_mem == 0
    //          fell into the partial-rewind else-branch below and took block-aligned mid-snapshot
    //          reuse. Which way that went depended on what the model ACTUALLY was, which is exactly
    //          what a failed probe leaves unknown:
    //            - not per-token rewindable (would have probed FULL): the partial seq_rm GGML_ABORTs
    //              the process.
    //            - genuinely per-token rewindable: it simply worked.
    //          So the old behaviour was a bet on "probably attention" that pays a crash when wrong.
    //          Fail CLOSED instead. The price is real and is NOT one cold prefill: every request
    //          that would have partially matched now gets whole-prefix-only reuse, for as long as
    //          the context stays in the NO state. That is the correct trade against aborting the
    //          server, but do not record it as free.
    //          Only NO with n_swa_mem == 0 changes here: n_swa_mem comes from llama_model_n_swa(),
    //          which is model-derived and independent of the probe, so an SWA model whose probe
    //          failed was already taking this arm via the n_swa_mem term.
    // The only class this returns false for is PART with n_swa_mem == 0 (plain attention), which
    // rewinds per token. Uses n_swa_mem, never n_swa: --swa-full
    // zeroes n_swa for the batch planner but the engine keeps masking on save.
    bool restore_is_whole_prefix_only() const {
        return ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
               ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS   ||
               ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO   ||
               n_swa_mem > 0;
    }

    // The memory classes whose disk units carry a regenerate logits sidecar: the ones that cannot rewind
    // by one token (FULL refuses any partial seq_rm; RS keeps rollback rows only in memory, never in a
    // saved state, so a restored RS sequence cannot rewind either). For them an exact resend of a saved
    // unit would otherwise re-decode the whole prompt to obtain the last token's logits. RS is what a
    // hybrid model becomes as soon as a draft (MTP) asks for rollback rows, so without it here the same
    // request costs nothing with MTP off and a full prefill with MTP on.
    //
    // Every class now (Q5 (a)): NO (DeepSeek-V4's compressed caches) cannot rewind at all and needs it as much
    // as FULL and RS do; a sliding window past one window and plain attention could rewind by one token and
    // re-decode it, but with the sidecar an exact resend of a saved unit emits its first token with no decode
    // on every class, so a resend costs the same whatever the model. The cost is one n_vocab float row per
    // unit on disk and its copy at each sampled token while --slot-save-path is set.
    bool logits_sidecar_class() const {
        return true;
    }

    // The manual /slots save keeps upstream's behaviour for the classes that rewind by one token (an exact
    // resend after a manual restore re-decodes that token, as upstream does) and writes the sidecar only where
    // a resend could not continue without it: FULL, RS and NO.
    bool logits_sidecar_manual() const {
        return ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
               ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS   ||
               ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    }

    // An SWA snapshot whose cells all fit inside one window was persisted WHOLE. state_write drops
    // only the cells masked against the snapshot's OWN pos_max, and while every position is below
    // n_swa_mem nothing is masked (STANDARD keeps p1 - p0 < n_swa, CHUNKED keeps the first chunk).
    // Such a snapshot is a complete attention state, so it rewinds per token exactly like plain
    // attention and the whole-prefix rule does not apply to it. Cells bound positions from above
    // (M-RoPE media spends fewer positions than cells), so a cell count is a conservative test.
    // auto_restore_into_slot re-checks the LOADED window (pos_min == 0) before keeping a rewound
    // restore, which also covers a mask type where n_cells <= n_swa is not sufficient (SYMMETRIC).
    bool swa_snapshot_is_whole_window(size_t n_cells) const {
        return ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART &&
               n_swa_mem > 0 && n_cells <= (size_t) n_swa_mem;
    }

    // Per-snapshot form of restore_is_whole_prefix_only(): the class rule, minus the SWA snapshots
    // that were persisted whole (see swa_snapshot_is_whole_window).
    bool restore_needs_whole_prefix(size_t n_cells) const {
        return restore_is_whole_prefix_only() && !swa_snapshot_is_whole_window(n_cells);
    }

    // ----- auto disk cache: fingerprint, index, restore, save (all gated by auto_cache_enabled()) -----

    // hash of the active LoRA set (ids/scales) for the fingerprint; 0 when no adapter is active.
    static uint64_t auto_lora_hash(const std::vector<common_adapter_lora_info> & lora) {
        uint64_t h = 0xcbf29ce484222325ULL;
        bool any = false;
        for (const auto & a : lora) {
            if (a.scale == 0.0f) {
                continue; // disabled adapter does not affect inference identity
            }
            any = true;
            for (char c : a.path) {
                h ^= (uint64_t) (unsigned char) c; h *= 0x100000001b3ULL;
            }
            uint32_t sc; std::memcpy(&sc, &a.scale, sizeof(sc));
            h = auto_hash_mix(h, (int32_t) sc);
        }
        return any ? h : 0;
    }

    // Compute the live model fingerprint once at load (invariant 3). Pure-CPU; only called from
    // an auto_cache_enabled() branch so it costs nothing when OFF.
    // See README "Automatic disk prompt cache" for which flags invalidate the cache.
    model_fp auto_compute_fingerprint() const {
        model_fp fp;
        // model identity string (arch + params + quant), hardened with size/n_params/n_embd/n_layer.
        char desc[256] = {0};
        llama_model_desc(model_tgt, desc, sizeof(desc));
        uint64_t h = 0xcbf29ce484222325ULL;
        for (const char * p = desc; *p; ++p) {
            h ^= (uint64_t) (unsigned char) *p; h *= 0x100000001b3ULL;
        }
        const uint64_t sz = llama_model_size(model_tgt);
        const uint64_t np = llama_model_n_params(model_tgt);
        h = auto_hash_mix(h, (int32_t) (sz & 0xFFFFFFFFu)); h = auto_hash_mix(h, (int32_t) (sz >> 32));
        h = auto_hash_mix(h, (int32_t) (np & 0xFFFFFFFFu)); h = auto_hash_mix(h, (int32_t) (np >> 32));

        fp.fp_model       = h;
        fp.fp_n_vocab     = (uint32_t) llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
        fp.fp_n_ctx_train = (uint32_t) llama_model_n_ctx_train(model_tgt);
        fp.fp_n_embd      = (uint32_t) llama_model_n_embd(model_tgt);
        fp.fp_n_layer     = (uint32_t) llama_model_n_layer(model_tgt);
        fp.fp_rope_type   = (uint32_t) llama_model_rope_type(model_tgt);
        // K/V cache type has no live-ctx getter: capture from the server's own params (the value
        // used to construct ctx_tgt). Blob-layout-critical: a Q4_0-KV blob into an F16 ctx corrupts.
        fp.fp_cache_k     = (uint32_t) params_base.cache_type_k;
        fp.fp_cache_v     = (uint32_t) params_base.cache_type_v;
        fp.fp_n_ctx       = (uint32_t) llama_n_ctx_seq(ctx_tgt);
        // the model's own LongRoPE threshold (hparams.n_ctx_orig_yarn, what get_rope_factors compares
        // n_ctx_seq against), 0 on a model whose RoPE factors do not depend on the context size. Never
        // rebuilt from CLI params: --yarn-orig-ctx reaches cparams only, not that threshold, and
        // llama_model_n_ctx_train can be rewritten under custom YaRN after the context is created.
        fp.fp_n_ctx_orig  = llama_model_n_ctx_orig_longrope(model_tgt);
        // FULL and RS share a value: on a recurrent/hybrid model the class is RS only because MTP (or
        // any draft) asked for n_rs_seq > 0 rollback rows, which are never serialised (state_write
        // writes the current row, state_read resets rs_idx to 0), so the blob is the same and turning
        // speculation on or off must not split the store. Restore semantics follow the LIVE class.
        // The other two parts are the attention-KV layout the reader refuses on mismatch, read from
        // the context (after -fa auto and kv_unified resolve), never from params_base. On a class with
        // no attention KV (pure recurrent) they still split peers, conservatively: such pools do not
        // mix -fa or --parallel in practice, and a needless miss is safe where a wrong restore is not.
        const uint32_t kv_n_stream = std::max<uint32_t>(1u, llama_kv_n_stream(ctx_tgt));
        fp.fp_kv_layout   = ((ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                              ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS) ? 1u : 0u)
                          | (llama_kv_v_trans(ctx_tgt) ? 2u : 0u)
                          | (std::min<uint32_t>(kv_n_stream - 1u, 0xFFFFu) << 8);
        fp.fp_block       = (uint32_t) params_base.slot_save_block;
        // effective rope scale (positions are baked into the saved state). rope_freq_scale==0 means
        // "use the model's trained value", so fall back to that for a stable comparison.
        float rs = params_base.rope_freq_scale != 0.0f
                       ? params_base.rope_freq_scale
                       : llama_model_rope_freq_scale_train(model_tgt);
        uint32_t rsb; std::memcpy(&rsb, &rs, sizeof(rsb));
        fp.fp_rope_scale  = (uint64_t) rsb;
        // rope_freq_base + the five YaRN params also bake positions into the saved
        // KV, so they MUST be part of identity. There is no public getter for the model's trained
        // rope base, so we normalize "use-model-default" to a single canonical 0 sentinel: when the
        // operator left the knob at its default (rope_freq_base==0; YaRN floats<0, i.e. -1.0 "auto";
        // yarn_orig_ctx<=0), we store 0. Two runs that both rely on the model default thus match;
        // any explicit override (or two different overrides) yields a different fp and refuses
        // (conservative: a needless miss is safe, a wrong restore is not). yarn_orig_ctx is an int.
        auto bitcast_f = [](float v) -> uint32_t { uint32_t u; std::memcpy(&u, &v, sizeof(u)); return u; };
        auto norm_yarn = [&](float v) -> uint32_t { return v < 0.0f ? 0u : bitcast_f(v); }; // <0 == model default
        const float rb = params_base.rope_freq_base > 0.0f ? params_base.rope_freq_base : 0.0f; // 0 == model default
        uint32_t rbb; std::memcpy(&rbb, &rb, sizeof(rbb));
        fp.fp_rope_base      = (uint64_t) rbb;
        fp.fp_yarn_ext       = norm_yarn(params_base.yarn_ext_factor);
        fp.fp_yarn_attn      = norm_yarn(params_base.yarn_attn_factor);
        fp.fp_yarn_beta_fast = norm_yarn(params_base.yarn_beta_fast);
        fp.fp_yarn_beta_slow = norm_yarn(params_base.yarn_beta_slow);
        fp.fp_yarn_orig_ctx  = params_base.yarn_orig_ctx > 0 ? (uint32_t) params_base.yarn_orig_ctx : 0u;
        // LoRA: fingerprint the global adapter set so a snapshot under adapter A never restores
        // under B. (Per-request adapter overrides additionally gate at the restore hook.)
        fp.fp_lora        = auto_lora_hash(params_base.lora_adapters);
        // deployment-shape bit (invariant 3): text-only server vs --mmproj server get
        // disjoint stores (mmproj-aware rope/projector wiring can change the text KV layout).
        fp.fp_mmproj_loaded = (mctx != nullptr) ? 1u : 0u;
        // projector identity, computed at mmproj load (0 on a text-only server).
        fp.fp_mmproj        = fp_mmproj;
        return fp;
    }

    // Auto-snapshot filename: a full-identity prefix (every restore_compatible field, not just fp_model)
    // so that peers sharing one dir who agree on the token prefix but differ in any geometry field
    // get DISJOINT names instead of atomically renaming over each other; chain-hash + token-count
    // make it deterministic across same-config processes (a same-prefix save from another process
    // yields the same name -> atomic-rename-idempotent). The prefix is naming/collision-avoidance
    // only: the scan still runs restore_compatible after reading the sidecar (it never parses the
    // prefix), and the block chain keeps its fp_model salt, so foreign v1 units written under the
    // old fp_model-prefixed name still index and restore unchanged.
    std::string auto_state_filename(uint64_t chain_hash, size_t n_tokens) const {
        char buf[96]; // "auto-" + 16 hex id + "-" + 16 hex hash + "-" + up to 20-digit count + ".bin" < 96
        snprintf(buf, sizeof(buf), "auto-%016" PRIx64 "-%016" PRIx64 "-%zu.bin",
                 cur_fp.identity_hash(), chain_hash, n_tokens);
        return params_base.slot_save_path + std::string(buf);
    }

    // A delta's parent: the tip's own directory and "auto-<identity>-" prefix, with the parent's
    // (id, n_tokens) pair. A chain is written under one identity prefix (the save-side parent-find
    // only links a parent named under the writer's), so resolving under the tip's prefix is exact,
    // and it keeps a chain readable when the identity_hash rule changes under it (a store written
    // before n_ctx left the identity) or a peer named it. Falls back to the live identity for a tip
    // whose name is not an auto snapshot (a renamed file restored through /slots).
    std::string auto_parent_state_filename(const std::string & tip_path, uint64_t parent_id, size_t n_tokens) const {
        uint64_t tip_id = 0;
        uint32_t tip_n  = 0;
        if (!slot_save_parse_node_id(tip_path, tip_id, tip_n)) {
            return auto_state_filename(parent_id, n_tokens);
        }
        const size_t slash = tip_path.find_last_of("/\\");
        const size_t base  = (slash == std::string::npos) ? 0 : slash + 1;
        // "auto-" (5) + 16 hex identity + "-" (1); slot_save_parse_node_id validated the layout
        char tail[64];
        snprintf(tail, sizeof(tail), "%016" PRIx64 "-%zu.bin", parent_id, n_tokens);
        return tip_path.substr(0, base + 5 + 16 + 1) + std::string(tail);
    }

    // Insert a snapshot at a boundary it reaches. Multiple snapshots are RETAINED per boundary
    // (longest first, deduped by length) so a longer superset does NOT shadow a shorter exact-length
    // snapshot: the shorter one is the only candidate a FULL model can restore when the request ends
    // before the longer one (see auto_index_lookup). The list is capped; the shortest is dropped first
    // (deep-context snapshots cost the most to lose and reconstruct).
    void auto_index_insert_locked(uint64_t boundary, const auto_cache_entry & e) {
        auto & v = auto_idx.by_boundary[boundary];
        for (auto & c : v) {
            if (c.n_tokens == e.n_tokens && c.id == e.id) {
                c.state_path = e.state_path; // the same unit: keep the newest file for it
                c.fp         = e.fp;
                c.pinned     = e.pinned;
                c.deepest    = e.deepest;
                return;
            }
        }
        // keep the vector sorted by descending n_tokens
        auto pos = std::lower_bound(v.begin(), v.end(), e,
            [](const auto_cache_entry & a, const auto_cache_entry & b) { return a.n_tokens > b.n_tokens; });
        v.insert(pos, e);
        if (v.size() > AUTO_MAX_CANDIDATES_PER_BOUNDARY) {
            // Over the cap: NEVER drop the SHORTEST entry, and never a pinned one.
            //
            // The shortest entry at a boundary is the snapshot that ends closest to it, i.e. the
            // SHARED BASE for this prefix. It is the only real strict-prefix parent the incremental
            // save parent-find can use, and the only candidate a *different* conversation sharing this
            // prefix can restore from. Every longer entry is a divergent sibling: losing one costs that
            // one lineage its deep reuse; losing the base costs EVERY lineage the shared prefix and
            // forces each later sibling to fall back to a whole (v1) snapshot.
            //
            // This used to drop the shortest unpinned entry, protecting the base only via a MANUAL
            // `<state>.pin` marker. That is backwards: bases created automatically (the mid-prefill
            // context base, and any root a chat happens to establish) are never pinned, so the one
            // entry the bucket exists to serve was the first thing evicted. A boundary shared by more
            // than the cap (one long conversation with >32 saved nodes, or >32 chats behind a common
            // system prompt) silently un-indexed the shared prefix while its file stayed on disk, and
            // every new chat then cold-prefilled it. `.pin` stays honoured, but is no longer
            // load-bearing for correctness of the common case.
            //
            // Scan shortest-first among the REST (skip the last element, the base); the .pin marker is
            // re-stat'd for authority (a pin touched after the entry was indexed, the normal deploy
            // order, is honoured and the cached flag refreshed). If every droppable entry is pinned,
            // leave the bucket one over the cap rather than evict a base or a pinned entry.
            for (auto rit = std::next(v.rbegin()); rit != v.rend(); ++rit) {
                std::error_code pec;
                rit->pinned = std::filesystem::exists(rit->state_path + ".pin", pec) && !pec;
                if (!rit->pinned) {
                    v.erase(std::next(rit).base());
                    break;
                }
            }
        }
    }

    // True iff this exact unit (EXACTLY n_tokens cells with identity `id`) is already indexed at
    // `boundary`. The shared-context checkpoint uses this for its redundant-write dedup: the 2nd..Nth
    // chat sharing the same [0,B) prefix finds the base already published and writes nothing. A
    // LONGER snapshot at the same boundary is a divergent sibling prefix, not this base, so it must
    // not suppress the base write; nor may an equal-length one whose cells after the last whole block
    // differ (a system prompt ending in a date: same length every day, a different tail), which the
    // boundary key alone cannot tell apart. CALLER MUST HOLD auto_idx.mtx.
    bool auto_index_has_exact_locked(uint64_t boundary, uint32_t n_tokens, uint64_t id) const {
        auto it = auto_idx.by_boundary.find(boundary);
        if (it == auto_idx.by_boundary.end()) {
            return false;
        }
        for (const auto_cache_entry & c : it->second) {
            if (c.n_tokens == n_tokens && c.id == id) {
                return true;
            }
        }
        return false;
    }

    // Scan the slot-save dir and (re)build index entries from .meta sidecars: header-only reads
    // (never the multi-GB state). Each bad/foreign file is skipped individually (invariant 4);
    // foreign-model files are left on disk (a sibling model may own them). Idempotent: re-running it
    // only ever keep-longer-inserts the same/new entries (auto_index_insert_locked), so it is safe to
    // call repeatedly for the cross-process refresh. Records the dir mtime so a refresh can cheaply
    // tell whether anything changed. CALLER MUST HOLD auto_idx.mtx.
    // The identity fields in which a unit's fingerprint differs from the live one (restore_compatible's
    // fields, grouped as an operator would name them), for the identity-miss WRN.
    static std::string auto_fp_diff(const model_fp & d, const model_fp & live) {
        std::string out;
        const auto add = [&](bool differs, const char * what) {
            if (differs) {
                out += out.empty() ? what : std::string(", ") + what;
            }
        };
        add(d.fp_n_vocab != live.fp_n_vocab || d.fp_n_ctx_train != live.fp_n_ctx_train || d.fp_n_embd != live.fp_n_embd ||
            d.fp_n_layer != live.fp_n_layer || d.fp_rope_type != live.fp_rope_type, "model shape");
        add(d.fp_cache_k != live.fp_cache_k || d.fp_cache_v != live.fp_cache_v, "KV cache types");
        add(d.ctx_long_regime(live) != live.ctx_long_regime(live), "LongRoPE regime (-c on the other side of the original context)");
        add(d.fp_kv_layout != live.fp_kv_layout, "KV layout (-fa, KV streams)");
        add(d.fp_block != live.fp_block, "--slot-save-block");
        add(d.fp_rope_scale != live.fp_rope_scale || d.fp_rope_base != live.fp_rope_base || d.fp_yarn_ext != live.fp_yarn_ext ||
            d.fp_yarn_attn != live.fp_yarn_attn || d.fp_yarn_beta_fast != live.fp_yarn_beta_fast ||
            d.fp_yarn_beta_slow != live.fp_yarn_beta_slow || d.fp_yarn_orig_ctx != live.fp_yarn_orig_ctx, "RoPE/YaRN settings");
        add(d.fp_lora != live.fp_lora, "LoRA set");
        add(d.fp_mmproj_loaded != live.fp_mmproj_loaded || d.fp_mmproj != live.fp_mmproj, "mmproj");
        return out.empty() ? "unknown" : out;
    }

    // How many leading tokens of a text request a foreign-identity unit of this model covers (its deepest
    // shared whole block), with the differing fields of one such unit. 0 when none does or the request
    // carries media. CALLER MUST HOLD auto_idx.mtx.
    int32_t auto_foreign_cover_locked(const server_tokens & req, std::string & why) {
        if (auto_idx.foreign_by_boundary.empty() || req.has_media()) {
            return 0;
        }
        const int B = params_base.slot_save_block;
        const auto bhs = auto_block_hashes(req.get_cell_tokens(), {}, B, cur_fp.fp_model, cur_fp.fp_mmproj, nullptr);
        for (size_t k = bhs.size(); k-- > 0; ) {
            if (auto_idx.foreign_by_boundary.count(bhs[k])) {
                for (const auto & [path, fu] : auto_idx.foreign_files) {
                    if (std::find(fu.bhs.begin(), fu.bhs.end(), bhs[k]) != fu.bhs.end()) {
                        why = fu.why;
                        break;
                    }
                }
                return (int32_t) (k + 1) * B;
            }
        }
        return 0;
    }

    void auto_index_scan_locked() {
        std::error_code mec;
        const auto dmt = std::filesystem::last_write_time(params_base.slot_save_path, mec);
        if (!mec) {
            auto_idx.dir_mtime = dmt; // snapshot the dir mtime we are scanning at
        }
        std::error_code ec;
        for (std::filesystem::directory_iterator it(params_base.slot_save_path, ec), end;
             !ec && it != end; it.increment(ec)) {
            std::error_code fec;
            if (!it->is_regular_file(fec) || fec) {
                continue;
            }
            const std::string p = it->path().string();
            // only our own state files: basename "auto-*.bin" (sidecars and temps skipped). Requiring
            // the "auto-" basename prefix rejects foreign/manual .bin files BY NAME before we open any
            // sidecar: the stated scan optimization.
            const std::string base = it->path().filename().string();
            if (base.rfind("auto-", 0) != 0) {
                continue; // not one of ours
            }
            if (p.size() < 4 || p.compare(p.size() - 4, 4, ".bin") != 0) {
                continue;
            }
            // already indexed, or already parse-rejected, by a prior scan? cheap skip so a
            // refresh only opens NEW files (units are immutable after their atomic rename).
            if (auto_idx.indexed_files.count(p) || auto_idx.rejected_files.count(p)) {
                continue;
            }
            // The publish sequence renames the .bin first and the .meta LAST, so a scan can
            // legitimately list a final-named .bin whose sidecar has not landed yet (a peer
            // mid-publish, or a writer that crashed in between: a later identical-prefix
            // save completes the unit under the same deterministic name). A MISSING sidecar
            // is therefore transient, not a verdict: skip WITHOUT caching so the next scan
            // retries. Only a sidecar that already EXISTS is immutable published state, so
            // only its parse failures may be remembered as permanent rejections. The
            // existence check runs BEFORE the read: a .meta that lands in between is simply
            // parsed (or retried next scan), never mis-cached.
            std::error_code sec;
            const bool meta_present = std::filesystem::exists(slot_meta_sidecar_path(p), sec) && !sec;
            model_fp fp;
            llama_tokens toks;
            std::vector<server_media_record> media;
            if (!slot_meta_read(p, cur_fp.fp_mmproj, fp, toks, media)) {
                if (meta_present) {
                    // short/corrupt meta, or an unknown version -> not indexable (invariant 4);
                    // remember the rejection so rescans never re-open the file.
                    auto_idx.rejected_files.insert(p);
                }
                continue;
            }
            if (!fp.restore_compatible(cur_fp, !media.empty())) {
                // foreign model / requant / different KV geometry (invariant 3). A unit of the same model
                // under another identity is remembered for the identity-miss counter.
                if (fp.fp_model == cur_fp.fp_model && !auto_idx.foreign_files.count(p)) {
                    auto & fu = auto_idx.foreign_files[p];
                    fu.bhs = auto_block_hashes(toks, media, params_base.slot_save_block, cur_fp.fp_model, cur_fp.fp_mmproj, nullptr);
                    fu.why = auto_fp_diff(fp, cur_fp);
                    for (uint64_t bh : fu.bhs) {
                        auto_idx.foreign_by_boundary[bh]++;
                    }
                }
                continue;
            }
            // rehash from the sidecar's cells + media records (media empty on v1 => the
            // text-only chain, bit-identical to what the writer keyed the file with). Only
            // chunk-safe boundaries are emitted, including a media unit's pure-text
            // pre-image boundaries, so a text request can reuse a media snapshot's prefix.
            uint64_t id = 0;
            const auto bhs = auto_block_hashes(toks, media, params_base.slot_save_block,
                                               cur_fp.fp_model, cur_fp.fp_mmproj, &id);
            // pin-awareness: a sibling "<state>.pin" marks this snapshot pinned so the per-boundary cap
            // never drops it (protects a pinned base that shares a bucket with >32 divergent siblings).
            std::error_code pec;
            const bool pinned = std::filesystem::exists(p + ".pin", pec) && !pec;
            auto_cache_entry e{ p, (uint32_t) toks.size(), fp, pinned, id, bhs.empty() ? 0 : bhs.back(), !media.empty() };
            for (uint64_t bh : bhs) {
                auto_index_insert_locked(bh, e);
            }
            auto_idx.indexed_files.insert(p);
        }
    }

    // One-time startup scan: builds the initial index. Invariant 1: only ever called from an
    // auto_cache_enabled() branch.
    void auto_index_scan() {
        std::lock_guard<std::mutex> lk(auto_idx.mtx);
        if (auto_idx.scanned) {
            return;
        }
        auto_idx.scanned = true;
        auto_idx.last_refresh = std::chrono::steady_clock::now();
        auto_index_scan_locked();
    }

    // Cross-process refresh: make snapshots that OTHER processes created visible here WITHOUT a
    // restart. Cheap by design: throttled to at most once per AUTO_REFRESH_MIN_MS, and even then it
    // only does one stat of the dir mtime: a full re-scan happens ONLY when the dir actually changed
    // (a peer create/rename/delete bumps the dir mtime) or when `force` is set (a lookup miss, where
    // we are about to pay a cold prefill anyway so the scan is free in comparison). On a change we
    // also drop entries whose files a peer evicted. CALLER MUST HOLD auto_idx.mtx.
    void auto_index_refresh_locked(bool force) {
        const auto now = std::chrono::steady_clock::now();
        if (!force &&
            now - auto_idx.last_refresh < std::chrono::milliseconds(AUTO_REFRESH_MIN_MS)) {
            return; // throttled: avoid a stat storm during a burst of lookups
        }
        auto_idx.last_refresh = now;
        std::error_code ec;
        const auto dmt = std::filesystem::last_write_time(params_base.slot_save_path, ec);
        if (!ec && dmt == auto_idx.dir_mtime && !force) {
            return; // nothing changed on disk since the last scan
        }
        // a peer changed the dir (or forced): re-scan for NEW files, then reconcile deletions.
        auto_index_scan_locked();
        auto_index_drop_missing_locked();
    }

    // Longest-prefix lookup over the request's cell-aligned tokens. Returns candidate snapshots to
    // try, BEST FIRST (deepest boundary first; within a boundary, longest first). For every class
    // restore_is_whole_prefix_only() covers (FULL, RS and SWA), snapshots longer than the request are
    // filtered out here: the whole snapshot must be a prefix of the request, so a longer one can never
    // restore. Plain-attention PART can rewind, so all lengths are kept. That predicate is the single
    // definition of the class set, named here so the prose and the filter cannot drift apart again
    // (RS was missing from the filter while this comment already said "recurrent").
    // The caller tries each in order until one restores (each
    // rejected candidate costs only a small .meta read + byte-compare; the multi-GB state loads only
    // once a candidate passes its gates): this fall-through is what stops a longer superset from
    // shadowing a shorter usable one at the same boundary. Verification (byte-compare of the
    // candidate's persisted cells + per-record media identity) is mandatory and done by the caller
    // (invariant 2). O(#blocks).
    //
    // `max_attempts` caps how many candidates are returned. The RESTORE path keeps the default (each
    // rejected restore costs a full .meta read + byte-compare, and a restore-side shadow was already
    // cured by the multi-candidate index); but the incremental SAVE parent-find passes SIZE_MAX to
    // scan the FULL per-boundary candidate vectors: the deepest strict-prefix parent (the base) is the
    // SHORTEST entry in its bucket, so with many divergent siblings it falls past the top few; a 4-cap
    // there hides it and forces a whole (v1) snapshot instead of a small delta. Ordering is unchanged
    // (deepest boundary first, longest first within a boundary), so the caller's first strict-prefix
    // match is still the deepest parent.
    std::vector<auto_cache_entry> auto_index_lookup(const server_tokens & req,
                                                    size_t max_attempts = AUTO_MAX_RESTORE_ATTEMPTS) {
        std::vector<auto_cache_entry> out;
        if (!auto_cache_enabled()) {
            return out; // off by default
        }
        // request-side identity records (empty for a text-only request, whose chain is then
        // bit-identical to the pre-media one). An identity-less chunk (e.g. a placeholder
        // bitmap) can never be verified against any snapshot, so such a request does not
        // look up at all: mirrors the save-side refusal.
        std::vector<server_media_record> media;
        try {
            media = req.extract_media_records();
        } catch (const std::exception & e) {
            SRV_WRN("auto-restore: lookup refused, %s\n", e.what());
            return out;
        }
        // For the classes that restore only a whole prefix, also keep the request's chain value after
        // every cell: (*chain)[n - 1] equals the identity of a unit of n cells exactly when that unit is a
        // whole prefix of the request (see the identity check in the loop below).
        std::vector<uint64_t> req_chain;
        const auto bhs = auto_block_hashes(req.get_cell_tokens(), media, params_base.slot_save_block,
                                           cur_fp.fp_model, cur_fp.fp_mmproj, nullptr,
                                           restore_is_whole_prefix_only() ? &req_chain : nullptr);
        std::lock_guard<std::mutex> lk(auto_idx.mtx);
        // Cross-process visibility: cheaply pick up snapshots a peer process created since our last
        // scan (throttled dir-mtime check). Then search; on a MISS, force a re-scan and search again
        // and the force is justified because a miss means we are about to cold-prefill, so the scan
        // cost is negligible against it, and a peer's snapshot written <1s ago (within the throttle
        // window) is still found on this first request rather than only the next one.
        auto_index_refresh_locked(/*force=*/false);
        std::unordered_set<std::string> seen_not_prefix;
        for (int attempt = 0; attempt < 2; ++attempt) {
            out.clear();
            std::unordered_set<std::string> seen;
            for (size_t k = bhs.size(); k-- > 0; ) { // longest boundary first
                auto it = auto_idx.by_boundary.find(bhs[k]);
                if (it == auto_idx.by_boundary.end()) {
                    continue;
                }
                for (const auto_cache_entry & c : it->second) { // longest first within the boundary
                    if (!c.fp.restore_compatible(cur_fp, c.has_media)) {
                        continue; // invariant 3
                    }
                    if (!model_fp::fits_ctx(c.n_tokens, cur_fp)) {
                        continue; // a larger rung's unit longer than this whole context: capacity
                    }
                    if (restore_needs_whole_prefix(c.n_tokens) && c.n_tokens > req.size()) {
                        // A snapshot longer than the request is never a whole prefix of it, and every
                        // class in restore_is_whole_prefix_only() (FULL, RS and SWA alike) can restore
                        // only a whole prefix. For SWA specifically, its persisted window is anchored at
                        // its own end, so a longer snapshot can only ever be refused by the restore gate,
                        // and because the store is dominated by release-time (prompt+generated)
                        // snapshots, such siblings are the COMMON case. Left unfiltered they exhaust
                        // AUTO_MAX_RESTORE_ATTEMPTS and starve the shorter, usable mid-prefill base
                        // (which sorts last).
                        // Counted, not silent: the restore site reports a miss that had such a
                        // candidate as a must-extend miss (WRN + auto_cache_restore_not_prefix_total).
                        if (seen_not_prefix.insert(c.state_path).second) {
                            auto_not_prefix_skips++;
                        }
                        continue;
                    }
                    if (restore_needs_whole_prefix(c.n_tokens) && c.deepest != 0 && c.deepest != bhs[k] &&
                        !seen.count(c.state_path)) {
                        // Met at a boundary below its own deepest one: the request leaves this unit before
                        // its last whole block, so it is not a whole prefix and these classes cannot rewind
                        // into it. Skip it here, before it takes one of the max_attempts places: in a
                        // multi-turn chat whose client does not re-render the previous response token for
                        // token, every earlier turn's unit sits at these shared boundaries, deeper than the
                        // prompt node that can restore, and after four turns they used to exhaust
                        // AUTO_MAX_RESTORE_ATTEMPTS and turn a hit into a cold prefill.
                        if (seen_not_prefix.insert(c.state_path).second) {
                            auto_not_prefix_skips++;
                        }
                        continue;
                    }
                    if (restore_needs_whole_prefix(c.n_tokens) && c.id != 0 && c.n_tokens > 0 &&
                        c.n_tokens <= req_chain.size() && req_chain[c.n_tokens - 1] != c.id &&
                        !seen.count(c.state_path)) {
                        // Met at its deepest boundary, but the request leaves it inside its trailing
                        // partial block, which no boundary key covers: a previous turn whose response
                        // ended before the next block boundary and which the client re-rendered
                        // differently. The unit's identity commits every cell, so it differs from the
                        // request's chain at the same length exactly when the unit is not a whole prefix.
                        // Skip it here for the same reason as above: left in, the .meta byte-compare
                        // rejects it only after it has taken one of the max_attempts places, and from the
                        // fifth such turn the earlier node that can restore never gets tried.
                        if (seen_not_prefix.insert(c.state_path).second) {
                            auto_not_prefix_skips++;
                        }
                        continue;
                    }
                    if (!seen.insert(c.state_path).second) {
                        continue; // the same snapshot reaches several boundaries
                    }
                    out.push_back(c);
                    if (out.size() >= max_attempts) {
                        return out;
                    }
                }
            }
            if (!out.empty()) {
                return out;
            }
            if (attempt == 0) {
                auto_index_refresh_locked(/*force=*/true); // miss -> rescan once before giving up
            }
        }
        return out;
    }

    // After an LRU eviction (which deletes files silently, ours OR a peer process's), drop index
    // boundaries pointing at files that no longer exist, and forget them in indexed_files so a future
    // re-create can be re-indexed. Cheap stat per unique path; keeps index <-> disk consistent (invariant 4).
    // A lookup that races an eviction and finds a now-deleted file simply fails the load -> prefill.
    // CALLER MUST HOLD auto_idx.mtx.
    void auto_index_drop_missing_locked() {
        std::unordered_set<std::string> gone;
        for (auto it = auto_idx.by_boundary.begin(); it != auto_idx.by_boundary.end(); ) {
            auto & vec = it->second;
            for (auto vit = vec.begin(); vit != vec.end(); ) {
                std::error_code ec;
                if (!std::filesystem::exists(vit->state_path, ec) || ec) {
                    gone.insert(vit->state_path);
                    vit = vec.erase(vit);
                } else {
                    ++vit;
                }
            }
            if (vec.empty()) {
                it = auto_idx.by_boundary.erase(it);
            } else {
                ++it;
            }
        }
        for (const auto & p : gone) {
            auto_idx.indexed_files.erase(p);
        }
        // foreign-identity units a peer evicted no longer explain a miss
        for (auto it = auto_idx.foreign_files.begin(); it != auto_idx.foreign_files.end(); ) {
            std::error_code ec;
            if (!std::filesystem::exists(it->first, ec) || ec) {
                for (uint64_t bh : it->second.bhs) {
                    auto f = auto_idx.foreign_by_boundary.find(bh);
                    if (f != auto_idx.foreign_by_boundary.end() && --f->second == 0) {
                        auto_idx.foreign_by_boundary.erase(f);
                    }
                }
                it = auto_idx.foreign_files.erase(it);
            } else {
                ++it;
            }
        }
        // also forget parse-rejected units whose files a peer evicted, so a later
        // re-create under the same deterministic name is examined afresh.
        for (auto it = auto_idx.rejected_files.begin(); it != auto_idx.rejected_files.end(); ) {
            std::error_code ec;
            if (!std::filesystem::exists(*it, ec) || ec) {
                it = auto_idx.rejected_files.erase(it);
            } else {
                ++it;
            }
        }
    }

    void auto_index_drop_missing() {
        std::lock_guard<std::mutex> lk(auto_idx.mtx);
        auto_index_drop_missing_locked();
    }

    // Restore a disk snapshot INTO `slot`, mirroring the SLOT_RESTORE handler body. Returns true on
    // success (slot.prompt.tokens / n_past-equivalent + just_restored + restored_logits are set as
    // for a manual restore). On ANY failure (load <=0, capacity exceeded) the slot seq is left
    // cleared and false is returned so the caller falls through to a normal prefill (invariant 4).
    bool do_slot_restore(server_slot & slot, const std::vector<std::string> & node_paths,
                         size_t * out_token_count = nullptr, size_t * out_nread = nullptr) {
        if (node_paths.empty()) {
            slot.prompt.tokens.clear();
            if (out_nread)       { *out_nread = 0; }
            if (out_token_count) { *out_token_count = 0; }
            return false;
        }
        // Reset the draft first. Whatever ctx_dft holds for this seq belongs to the PREVIOUS prompt, and
        // the suffix prefill that follows a restore only removes cells at or above its own start, so a
        // leftover draft would attend over another conversation's cells. After the target chain loads,
        // auto_restore_draft_chain repopulates the draft from the chain's .dft sidecars when every node
        // has one (auto-saves write them whenever a separate draft context exists); otherwise the draft
        // stays empty and that conversation drafts cold for [0, L), which is SILENT to the speculative
        // begin() checks (they test pos_max, which cannot see a hole below a populated tail). Count it:
        // llamacpp:auto_cache_restore_draft_{warm,cold}_total.
        // ctx_dft and ctx_tgt always own DISTINCT memory objects (llama_context builds its own via
        // model.create_memory, passing the other context's memory as mem_other), so the clear never
        // touches target cells. On a shared-CELLS draft (gemma4-class, llama_get_ctx_other(ctx_dft) ==
        // ctx_tgt) this is a no-op by construction, no sidecar is ever written (it would serialise
        // nothing), and the draft reads the target's restored cells directly.
        // Deliberately a direct llama_memory_seq_rm on ctx_dft, not slot.mem.seq_rm: the wrapper mirrors
        // onto both contexts and cannot express "load into one, reset the other". (-1, -1) is the rm_all
        // path, legal on every memory class, so the result is not checked.
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
        }
        // Load the chain in position order: node [0] clears the destination seq (a whole base/root
        // snapshot), nodes [1..] append their delta cells with NO_CLEAR so base + deltas compose.
        // A 1-element chain is exactly the previous single clearing load: byte-for-byte the same.
        llama_tokens tokens;
        tokens.resize(slot.n_ctx);
        size_t token_count = 0;
        size_t total_nread = 0;
        for (size_t i = 0; i < node_paths.size(); ++i) {
            llama_state_seq_flags flags = (i == 0) ? 0 : LLAMA_STATE_SEQ_FLAGS_NO_CLEAR;
            // every node carries the whole side-state at its own end and the next node's replaces it, so an inner
            // node needs only its positional cells: its side-state is read past instead of moved to the device.
            // The tip loads whole. A periodic-checkpoint chain of K nodes reads one side-state instead of K.
            if (side_skip_supported && i + 1 < node_paths.size()) {
                flags |= LLAMA_STATE_SEQ_FLAGS_SKIP_SIDE;
            }
            size_t node_token_count = 0;
            const size_t nread = llama_state_seq_load_file_ext(
                ctx_tgt, node_paths[i].c_str(), slot.id, flags,
                tokens.data(), tokens.size(), &node_token_count);
            if (nread == 0) {
                // a failed node may leave cells behind: node [0] clears the seq first, but a NO_CLEAR node that
                // fails keeps everything the earlier nodes appended (and a composite cache may hold them in one
                // of its halves only), so drop the seq on both contexts rather than only forgetting the tokens
                auto_restore_drop(slot);
                if (out_nread)       { *out_nread = total_nread; }
                if (out_token_count) { *out_token_count = 0; }
                return false;
            }
            total_nread += nread;
            token_count = node_token_count; // the tip (last) node header carries the full [0, hi) list
        }
        if (out_nread)       { *out_nread = total_nread; }
        if (out_token_count) { *out_token_count = token_count; }
        tokens.resize(token_count);
        slot.prompt.tokens.clear();
        slot.prompt.tokens.insert(tokens);

        // the target is in place: bring the draft back warm from the chain's .dft sidecars when every
        // node has one, before any checkpoint below snapshots the draft state
        if (ctx_dft) {
            auto_restore_draft_chain(slot, node_paths);
        }

        restore_finish(slot, &node_paths.back(), token_count);
        return true;
    }

    // Bookkeeping after a KV state has been loaded into `slot` (prompt tokens already set), shared by
    // the auto-cache restore and both manual /slots formats: marks the slot just-restored, drops the
    // previous task's checkpoints, synthesises the FULL-class checkpoint and re-arms the regenerate
    // logits from `sidecar_path` when one is given. `synth_ckpt` false: the caller restores the slot's own
    // checkpoints instead (a manual /slots file that carries them), so no FULL-class checkpoint is made.
    void restore_finish(server_slot & slot, const std::string * sidecar_path, size_t token_count, bool synth_ckpt = true) {
        slot.just_restored = true;

        // Drop the previous task's checkpoints UNCONDITIONALLY, for every class, before any synth
        // below. They snapshot a DIFFERENT prompt's ctx_tgt state and the sequence they described no
        // longer exists. The auto path already clears the list before calling in; the manual /slots
        // restore clears it only inside its own SWA branch, so without this an RS
        // (or plain-attention) manual restore carried the previous task's checkpoints forward: the next
        // request diverging at d < L gets pos_min_thold == d against a recurrent tail pos_min of L-1,
        // opens the consumer gate, and a surviving checkpoint with cur.pos_min == 0 is accepted and
        // loaded, i.e. a foreign conversation's state.
        slot.prompt.checkpoints.clear();

        // FULL only. Reconstruct a context checkpoint at the restored position so a FULL model, which
        // cannot partially rewind, can reuse this state for the suffix.
        // RS is deliberately NOT included, and this is not an oversight, do not add it "for symmetry".
        // ON THE AUTO PATH, where the whole-prefix rule in auto_restore_into_slot holds, an RS restore
        // leaves n_past == pos_next == L while a recurrent pos_min is the TAIL (L-1), so: when the
        // request EXTENDS, pos_min_thold == L and the consumer gate `pos_min >= pos_min_thold` is
        // false, the checkpoint list is never iterated; on an exact resend, pos_min_thold == L-1 and
        // the acceptance test `cur.pos_min < pos_min_thold` is L-1 < L-1, false. The checkpoint would
        // be unreachable in one case and rejected in the other, while create_checkpoint pays a
        // whole-sequence PARTIAL_ONLY state copy per restore. On the MANUAL path no whole-prefix rule
        // applies and the next request may diverge at any d < L, which would make such a checkpoint
        // reachable, so the clear above is what keeps an RS slot from having one at all rather than
        // this predicate. Adding an RS synth only becomes live if someone relaxes the consumer to
        // accept cur.pos_min == pos_min_thold for tail memories, and that is exactly the GGML_ABORT the
        // NOTE at the acceptance predicate documents. The same reasoning forbids extending the SWA
        // synth in auto_restore_into_slot to RS.
        // SWA gets its own synth there (it is a real window start, not a tail).
        if (synth_ckpt && ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            const auto ckpt_pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
            const auto ckpt_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
            if (ckpt_pos_min >= 0) {
                create_checkpoint(slot, 0, ckpt_pos_min, ckpt_pos_max);
            }
        }

        // The restored state's running distribution is unknown; invalidate any stale capture so a
        // later SLOT_SAVE cannot persist a mismatched sidecar.
        slot.logits_last.clear();
        slot.logits_last_n_tokens = -1;

        // Load the regenerate logits sidecar (FULL and RS) so an exact-prompt regenerate can emit the
        // first token without re-decoding into the restored recurrent state.
        slot.restored_logits.clear();
        if (sidecar_path != nullptr && logits_sidecar_class()) {
            const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
            if (slot_logits_read(*sidecar_path, nv, (uint32_t) token_count, slot.restored_logits)) {
                slot.restored_logits_n = (int32_t) token_count;
                SLT_INF(slot, "loaded logits sidecar (%d vocab, %zu tokens), regenerate fast-path armed\n", nv, token_count);
            }
        }
    }

    // SWA models (PART seq_rm, n_swa > 0) after a manual /slots restore: reconstruct a checkpoint at
    // the restored position, mirroring auto_restore_into_slot. The downstream checkpoint search finds
    // none in a fresh process and would force a full re-process on the next request, silently
    // discarding the restore. (FULL models get theirs in restore_finish; non-SWA attention models skip
    // the checkpoint machinery.)
    void restore_swa_checkpoint(server_slot & slot) {
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART && n_swa_mem > 0) {
            const auto ckpt_pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
            const auto ckpt_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
            if (ckpt_pos_min >= 0) {
                slot.prompt.checkpoints.clear();
                create_checkpoint(slot, 0, ckpt_pos_min, ckpt_pos_max);
            }
        }
    }

    // Drop a just-restored snapshot entirely: empty the slot's KV seq and prompt/restore state so
    // the caller falls back to a clean cold prefill (invariant 4). Canonical post-restore bail-out
    // for auto_restore_into_slot: every abort after do_slot_restore succeeded goes through here.
    void auto_restore_drop(server_slot & slot) {
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, -1, -1);
        if (ctx_dft) {
            // the restore may have brought the draft back from sidecars; it must not outlive the target
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
        }
        slot.prompt.tokens.clear();
        slot.prompt.checkpoints.clear();
        slot.just_restored = false;
        slot.restored_logits.clear();
    }

    // MANUAL-RESTORE rehydration: a media snapshot's state file persists LLAMA_TOKEN_NULL
    // cells, so after do_slot_restore the slot's prompt holds NULL cells with no live chunks
    // behind them: a shape no downstream consumer can traverse. Unlike the auto path there is
    // no request to rebuild from, so the v2 .meta sidecar's identity records are rehydrated
    // into STUB chunks (mtmd_input_chunk_init_stub: id + geometry, placeholder data). Stubs
    // verify and count positions exactly like live chunks; the persisted embeddings are already
    // inside the loaded KV state, so their pixels are never needed: any path that would
    // re-encode one refuses and clears (see process_mtmd_chunk). Returns false with `err` set
    // when the snapshot cannot be rehydrated (missing/invalid sidecar, fingerprint mismatch,
    // sidecar/state divergence, irreproducible chunk geometry); the caller must then drop the
    // restored state. A text snapshot (no NULL cells) returns true untouched.
    bool manual_restore_rehydrate_media(server_slot & slot, const std::string & filepath, std::string & err) {
        const llama_tokens & cells = slot.prompt.tokens.get_cell_tokens();
        if (std::find(cells.begin(), cells.end(), LLAMA_TOKEN_NULL) == cells.end()) {
            return true; // text snapshot: nothing to rehydrate
        }
        if (!mctx) {
            err = "Cannot restore media tokens without an mmproj (the state file contains media cells)";
            return false;
        }
        model_fp disk_fp;
        llama_tokens disk_toks;
        std::vector<server_media_record> disk_media;
        if (!slot_meta_read(filepath, cur_fp.fp_mmproj, disk_fp, disk_toks, disk_media)) {
            err = "state file contains media cells but has no valid .meta sidecar to rebuild them from";
            return false;
        }
        if (disk_media.empty()) {
            // unreachable via our own writers (a v1 sidecar with NULL cells is rejected by
            // slot_meta_read), kept as an explicit guard against future format drift
            err = ".meta sidecar carries no media records for a media state file";
            return false;
        }
        // The same identity gate as the auto restore. Capacity needs no check here: the
        // state is already loaded, and llama_state_seq refused it if it did not fit.
        if (!disk_fp.restore_compatible(cur_fp, /*has_media=*/true)) {
            err = "snapshot fingerprint mismatch (model, projector or KV geometry changed)";
            return false;
        }
        if (disk_toks != cells) {
            err = ".meta sidecar does not describe this state file";
            return false;
        }
        // rebuild the prompt: text cells verbatim, each record as a stub chunk. slot_meta_read
        // guarantees the records tile the NULL cells exactly, so this walk covers every cell
        // (push_back(llama_token) throws on a NULL cell outside a record: impossible here, but
        // the catch keeps a corrupt sidecar from unwinding through the server loop).
        server_tokens rebuilt;
        rebuilt.has_mtmd = true;
        try {
            size_t r = 0;
            for (size_t i = 0; i < cells.size(); ) {
                if (r < disk_media.size() && i == (size_t) disk_media[r].start_idx) {
                    const auto & rec = disk_media[r];
                    mtmd::input_chunk_ptr stub(mtmd_input_chunk_init_stub(
                            mctx, rec.is_audio != 0, rec.id.c_str(),
                            rec.n_tokens, (llama_pos) rec.n_pos, rec.nx, rec.ny));
                    if (!stub) {
                        err = "cannot rebuild a media chunk with the snapshot's geometry on this model";
                        return false;
                    }
                    rebuilt.push_back(stub.get()); // copies
                    i += rec.n_tokens;
                    r++;
                } else {
                    rebuilt.push_back(cells[i]);
                    i++;
                }
            }
        } catch (const std::exception & e) {
            err = std::string("rehydration failed: ") + e.what();
            return false;
        }
        if (rebuilt.get_cell_tokens() != cells || !rebuilt.validate(ctx_tgt)) {
            err = "rehydrated prompt failed validation";
            return false;
        }
        slot.prompt.tokens = std::move(rebuilt);
        return true;
    }

    // Build the root->tip chain of node .bin paths for a (possibly delta) tip by walking parent
    // links ON DISK (the tree is DERIVED FROM DISK, no in-RAM map) and verifying every hop.
    // `tip_path` is the tip's .bin; `tip_toks` is its authoritative full [0, range_hi) cell-token
    // list (the meta is WHOLE even for a v3/v4 delta, so this is the byte-authority every parent
    // prefix is checked against); (parent_id, range_lo) are the tip's node-tail fields: (0, 0)
    // means a whole snapshot whose chain is just itself (a single clearing load == the pre-delta
    // behaviour). On success `chain` is root..tip in position order and returns true; on ANY
    // inconsistency (missing/corrupt/fingerprint-drift/non-contiguous/byte-mismatch parent, or a
    // pathological depth) returns false so the caller cold-prefills (invariant 4). SHARED by both
    // the auto-restore path and the manual /slots restore path so a v4 media delta tip composes
    // base + deltas (NO_CLEAR) identically on both, never mis-loading a partial .bin (decision 2).
    bool auto_build_restore_chain(const std::string & tip_path, const llama_tokens & tip_toks,
                                  uint64_t parent_id, uint32_t range_lo,
                                  std::vector<std::string> & chain) {
        chain.clear();
        chain.push_back(tip_path);
        // A v1/v2 whole snapshot is its own root: a single-element chain == the previous single
        // clearing load. A v3/v4 delta only stores its tail cells, so walk parent links to the
        // root and load base + deltas in position order (NO_CLEAR) to recompose the full prefix.
        if (parent_id == 0 && range_lo == 0) {
            return true;
        }
        uint64_t cur_parent_id = parent_id;
        uint32_t cur_range_lo  = range_lo;
        const size_t MAX_CHAIN_DEPTH = 4096; // bounded walk: a corrupt/looping link never hangs.
        while (true) {
            if (chain.size() > MAX_CHAIN_DEPTH) {
                return false; // pathological depth -> cold prefill (invariant 4)
            }
            const std::string parent_path = auto_parent_state_filename(tip_path, cur_parent_id, cur_range_lo);
            model_fp     parent_fp;
            llama_tokens parent_toks;
            std::vector<server_media_record> parent_media;
            uint64_t     parent_parent_id = 0;
            uint32_t     parent_lo        = 0;
            uint32_t     parent_hi        = 0;
            if (!slot_meta_read(parent_path, cur_fp.fp_mmproj, parent_fp, parent_toks, parent_media,
                                &parent_parent_id, &parent_lo, &parent_hi)) {
                return false; // parent meta missing/corrupt -> cold prefill
            }
            // Every hop is checked on its own against the live context, so a chain can never
            // mix identities. Capacity follows from the tip's (a parent is a strict prefix of it).
            if (!parent_fp.restore_compatible(cur_fp, !parent_media.empty())) {
                return false; // fingerprint drift on the parent -> cold prefill
            }
            // contiguity: the parent must end exactly where its child begins.
            if (parent_hi != cur_range_lo) {
                return false;
            }
            // IDENTITY: hash + range-contiguity alone do NOT prove this .bin holds the tip's actual
            // prefix: a parent_id/n_tokens filename collision (two distinct prefixes of equal length
            // whose block-boundary hash coincides) or a base rewritten for a different prefix could
            // land on the same deterministic name and compose the WRONG KV for [0, parent_hi) under
            // NO_CLEAR, silently. `tip_toks` is the authoritative full [0, range_hi) token record, so
            // byte-verify the parent's recorded tokens against that tip prefix. Media cells are
            // LLAMA_TOKEN_NULL on both sides (a media parent of a media delta), so this compare is
            // NULL==NULL for the shared media prefix: the per-record identity backstop that closes
            // that gap is enforced save-side (parent-find) and restore-side (tip full tiling verify).
            if (parent_toks.size() != (size_t) parent_hi ||
                (size_t) parent_hi > tip_toks.size() ||
                !std::equal(parent_toks.begin(), parent_toks.end(), tip_toks.begin())) {
                return false; // parent KV does not correspond to this prefix -> cold prefill
            }
            // the parent .bin must exist (meta is published last, but an orphan-reap can race).
            { std::ifstream pf(parent_path, std::ios::binary); if (!pf) { return false; } }
            chain.push_back(parent_path);
            if (parent_parent_id == 0 && parent_lo == 0) {
                break; // reached the root covering [0, hi)
            }
            cur_parent_id = parent_parent_id;
            cur_range_lo  = parent_lo;
        }
        // chain is tip..root; reverse to root..tip (position order) for the compose load.
        std::reverse(chain.begin(), chain.end());
        return true;
    }

    // Mode 2 (A2.3): load only the side-state of `cand` (its recurrent state at n_keep cells) into a slot that
    // already holds the same n_keep tokens' positional K/V, then trim the slot to n_keep. The draft keeps its own
    // positional cells too and gets the unit's carry-over when its .dft sidecar has one. Returns n_keep, or 0
    // after dropping the slot when the load failed.
    int auto_restore_side_only(server_slot & slot, const auto_cache_entry & cand, int n_keep) {
        llama_tokens toks((size_t) n_keep);
        size_t n_tok = 0;
        const size_t nread = llama_state_seq_load_file_ext(ctx_tgt, cand.state_path.c_str(), slot.id,
                                                           LLAMA_STATE_SEQ_FLAGS_SKIP_POSITIONAL,
                                                           toks.data(), toks.size(), &n_tok);
        const llama_pos p_keep = slot.prompt.tokens.pos_next((int64_t) n_keep);
        bool ok = nread > 0 && n_tok == (size_t) n_keep;
        // the attention cells past the unit, and the recurrent tail (now at n_keep - 1, so nothing to remove)
        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, p_keep, -1);
        if (!ok) {
            auto_restore_drop(slot);
            metrics.n_auto_restore_failed++;
            SLT_WRN(slot, "auto-restore: loading the side-state of %s failed; the slot was cleared, prefilling instead\n",
                    cand.state_path.c_str());
            return 0;
        }
        if (ctx_dft && !llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, p_keep, -1)) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1); // a draft that cannot trim drafts cold
        }
        slot.prompt.tokens.keep_first(n_keep);
        restore_finish(slot, &cand.state_path, (size_t) n_keep);
        llama_pos tail = -1;
        std::vector<uint8_t> st;
        if (spec && ctx_dft && slot_draft_trailer_read(slot_draft_sidecar_path(cand.state_path), tail, st) &&
            tail == llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id)) {
            common_speculative_set_state(spec.get(), slot.id, st);
        }
        return n_keep;
    }

    // The memory class as the auto disk cache handles it, for /props (null when the cache is off). The delta
    // probe runs at the first save, so its result is the gauge llamacpp:auto_cache_delta_capable instead.
    json auto_cache_caps() const {
        if (!auto_cache_enabled()) {
            return json();
        }
        const char * seq_rm = "no";
        switch (ctx_tgt_seq_rm_type) {
            case COMMON_CONTEXT_SEQ_RM_TYPE_PART: seq_rm = "part"; break;
            case COMMON_CONTEXT_SEQ_RM_TYPE_FULL: seq_rm = "full"; break;
            case COMMON_CONTEXT_SEQ_RM_TYPE_RS:   seq_rm = "rs";   break;
            default: break;
        }
        const char * node_prompt = params_base.slot_save_node_prompt == COMMON_SLOT_SAVE_NODE_PROMPT_ON  ? "on"
                                 : params_base.slot_save_node_prompt == COMMON_SLOT_SAVE_NODE_PROMPT_OFF ? "off" : "cold";
        return json {
            { "seq_rm",              seq_rm },                           // how the memory rewinds: part, full, rs or no
            { "n_swa",               n_swa_mem },                        // sliding window the engine masks with
            { "rewinds",             !restore_is_whole_prefix_only() },  // a longer unit restores trimmed to the request
            { "side_only_restore",   side_only_supported },              // restore mode 2: side-state alone
            { "inner_side_skipped",  side_skip_supported },              // a chain's inner nodes load positional cells only
            { "logits_sidecar",      logits_sidecar_class() },           // exact resends emit with no decode
            { "draft_sidecar",       ctx_dft != nullptr },
            { "block",               params_base.slot_save_block },
            { "incremental",         params_base.slot_save_incremental },
            { "deferred",            params_base.slot_save_defer && params_base.slot_save_staging_mb > 0 },
            { "node_prompt",         node_prompt },
            { "prefill_interval",    params_base.slot_save_prefill_interval },
        };
    }

    // How much of a raw in-memory match `n_past` the slot can actually keep, by the rule the prefill path
    // applies after the restore site: a memory class that cannot rewind keeps a strict extension of its cells
    // whole, and otherwise only what its window still covers or an in-memory context checkpoint at or below
    // the divergence restores. The restore margin compares a disk unit against this, not the raw match
    // (F3/A7): a hybrid slot that diverged inside its conversation keeps nothing past its last checkpoint, so
    // a disk node just below the divergence beats it.
    int32_t auto_usable_warm_reuse(const server_slot & slot, int32_t n_past, int32_t n_req) const {
        if (n_past <= 0 || n_past >= (int32_t) slot.prompt.n_tokens() || !restore_is_whole_prefix_only()) {
            return n_past;
        }
        const llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);
        const bool      tail     = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                                   ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS;
        const bool      has_new  = n_past < n_req;
        const llama_pos thold    = std::max<llama_pos>(0, pos_next - (tail ? 0 : n_swa) - (has_new ? 0 : 1));
        const llama_pos pos_min  = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
        if (pos_min < thold) {
            return n_past; // the window still covers the divergence (pos_min < 0: nothing held, nothing to lose)
        }
        int32_t best = 0;
        for (const auto & cp : slot.prompt.checkpoints) {
            if (cp.pos_max <= pos_next && (cp.pos_min == 0 || cp.pos_min < thold)) {
                best = std::max<int32_t>(best, std::min<int32_t>(n_past, (int32_t) cp.n_tokens));
            }
        }
        return best;
    }

    // AUTO-RESTORE wrapper: byte-verify the candidate's persisted cells against the request prefix
    // and its media records against the request's live chunks (invariant 2), confirm the
    // fingerprint (invariant 3), then restore. Returns the verified prefix length actually
    // restored, or 0 if nothing was restored (caller keeps the in-memory prefill path).
    // `req` is the full request; `n_keep_mem` is the in-memory match to beat.
    //
    // `n_match` is how many leading cells the slot ALREADY holds for this request (the raw LCP, before any
    // usable-reuse correction). When it reaches the whole unit, the unit is a whole prefix of the request
    // and the memory type can load a side-state alone, the unit's side-state is loaded over the slot's own
    // positional cells (mode 2, auto_restore_side_only) instead of the whole unit.
    int auto_restore_into_slot(server_slot & slot, const auto_cache_entry & cand,
                               const server_tokens & req, int n_keep_mem, int n_match = 0) {
        // read the small .meta sidecar (tokens + fp + media records), never opens the multi-GB
        // state file (invariant 5).
        // the restore replaces this slot's cells: its deferred copies are emitted first, outside the lock
        aw_def_force_slot(slot.id, "forced");
        // Shared store lock from this .meta read until the chain is loaded: every .meta read below
        // (the tip here, each parent in auto_build_restore_chain) then describes the .bin loaded for it.
        auto_store_lock rd_lock(std::filesystem::path(cand.state_path).parent_path().string(),
                                /*exclusive=*/false, auto_store_lock::RESTORE_TIMEOUT_MS);
        if (!rd_lock.held()) {
            SLT_WRN(slot, "auto-restore: store lock busy, skipping %s\n", cand.state_path.c_str());
            return 0; // invariant 4
        }
        model_fp disk_fp;
        llama_tokens disk_toks;
        std::vector<server_media_record> disk_media;
        uint64_t disk_parent_id = 0;
        uint32_t disk_range_lo  = 0;
        uint32_t disk_range_hi  = 0;
        if (!slot_meta_read(cand.state_path, cur_fp.fp_mmproj, disk_fp, disk_toks, disk_media,
                            &disk_parent_id, &disk_range_lo, &disk_range_hi)) {
            return 0; // invariant 4
        }
        if (!disk_fp.restore_compatible(cur_fp, !disk_media.empty())) {
            return 0; // invariant 3
        }
        if (!model_fp::fits_ctx(disk_toks.size(), cur_fp)) {
            return 0; // capacity: checked before any state-file I/O (the lookup filters it too)
        }
        if (disk_fp.fp_n_ctx != cur_fp.fp_n_ctx) {
            // Cross-rung restore. Logged unconditionally: if n_ctx ever turns out to
            // reach the state bytes the symptom is a CONFIDENT WRONG ANSWER, and this
            // line is the only forensic trail.
            SRV_INF("auto restore: cross-ctx reuse, snapshot n_ctx=%u into live n_ctx=%u\n",
                    (unsigned) disk_fp.fp_n_ctx, (unsigned) cur_fp.fp_n_ctx);
        }
        // request-side identity: cell-aligned tokens plus media records (empty on a text-only
        // request). The extraction throws on an identity-less chunk (e.g. a placeholder
        // bitmap): unverifiable, so the request simply does not restore (invariant 4).
        const llama_tokens & req_cells = req.get_cell_tokens();
        std::vector<server_media_record> req_media;
        try {
            req_media = req.extract_media_records();
        } catch (const std::exception & e) {
            SLT_WRN(slot, "auto-restore: refused, %s\n", e.what());
            return 0;
        }
        // byte-verify: longest common prefix of the persisted cells and the request cells
        // (invariant 2). Media cells are LLAMA_TOKEN_NULL on both sides and pass this compare
        // blindly: their content identity is verified per record next.
        const size_t lim = std::min(disk_toks.size(), req_cells.size());
        size_t v = 0;
        while (v < lim && disk_toks[v] == req_cells[v]) {
            ++v;
        }
        // per-record verification: every disk record starting inside the LCP must match a
        // request record at EXACTLY its start index: id, shape and type. Iterating the
        // records (never "the next chunk after") verifies each image of an adjacent pair
        // separately; a missing request record or an unequal field is a hard mismatch that
        // truncates the LCP to that record's start: same text + a different image reuses
        // exactly the pre-image prefix. Both sides are ordered by start_idx, and verified
        // records align the chunk boundaries inside the LCP, so the truncated v can only
        // fall on a text cell or a chunk start (chunk-safe by construction).
        for (const auto & rec : disk_media) {
            if ((size_t) rec.start_idx >= v) {
                break; // this and all later records start outside the verified prefix
            }
            const auto it = std::lower_bound(req_media.begin(), req_media.end(), rec.start_idx,
                [](const server_media_record & r, uint32_t s) { return r.start_idx < s; });
            const bool match = it != req_media.end()      &&
                               it->start_idx == rec.start_idx &&
                               it->id        == rec.id        &&
                               it->n_tokens  == rec.n_tokens  &&
                               it->n_pos     == rec.n_pos     &&
                               it->nx        == rec.nx        &&
                               it->ny        == rec.ny        &&
                               it->is_audio  == rec.is_audio;
            if (!match) {
                v = rec.start_idx;
                break;
            }
        }
        // Only WHOLE-block prefixes are valid reuse lengths (hash boundaries).
        const int B = params_base.slot_save_block;
        int n_keep_disk;
        if (restore_needs_whole_prefix(disk_toks.size())) {
            // FULL, RS and SWA can only ever resume a snapshot as a WHOLE prefix (see
            // restore_is_whole_prefix_only): do_slot_restore loads the ENTIRE L-token snapshot, and a
            // later keep_first(n_past < L) would issue a partial common_context_seq_rm that either
            // GGML_ABORTs (FULL refuses any partial range; RS refuses a rollback beyond n_rs_seq) or
            // attends over cells that were never persisted (SWA keeps only [L - n_swa_mem, L)).
            // So auto-restore this snapshot ONLY when the request diverges at or beyond its end
            // (v == disk_toks.size(), i.e. the request strictly extends it, or matches it exactly).
            // No block-boundary clamp here: the whole snapshot is the only legal restore length.
            // Refuse BEFORE the multi-GB read so the candidate loop falls through to a SHORTER snapshot
            // that IS a whole prefix of this request (e.g. the mid-prefill context base).
            if (v != disk_toks.size()) {
                auto_not_prefix_skips++;
                SLT_DBG(slot, "auto-restore: snapshot is not a whole prefix of the request "
                              "(verified %zu of %zu snapshot tokens; request %zu, seq_rm_type = %d, n_swa = %d) - skipping %s\n",
                        v, disk_toks.size(), req.size(), (int) ctx_tgt_seq_rm_type, n_swa_mem, cand.state_path.c_str());
                return 0;
            }
            n_keep_disk = (int) disk_toks.size();
        } else {
            // Reached for the ONE class restore_is_whole_prefix_only() rejects: PART with
            // n_swa_mem == 0 (plain attention), which supports per-token partial seq_rm. NO used to
            // land here too, which was a latent GGML_ABORT on a context whose capability probe had
            // merely failed; it is now folded into the helper and fails closed. See the helper.
            // A mid-snapshot divergence is fine here: claim the verified prefix clamped down to the
            // last whole block boundary <= v. Every class that CANNOT do that is now diverted by the
            // branch above, which admits only v == disk_toks.size() and never clamps. An exact
            // full-snapshot match keeps the whole snapshot length.
            if (v == disk_toks.size()) {
                n_keep_disk = (int) disk_toks.size();
            } else {
                n_keep_disk = (int) (v - (v % (size_t) B));
                // shared chunk-boundary rule (same predicate the hash sites emit boundaries
                // with): a reuse length may not split a media chunk, so step down block by
                // block until the cut is chunk-safe.
                while (n_keep_disk > 0 &&
                       !boundary_is_chunk_safe(disk_toks, disk_media, (size_t) n_keep_disk)) {
                    n_keep_disk -= B;
                }
            }
        }
        if (n_keep_disk <= 0) {
            return 0;
        }
        // MARGIN gate (invariant 5): only pay a multi-GB load if disk strictly beats the
        // in-memory match by at least one block: never thrash a reload to save a few tokens.
        if (n_keep_disk < n_keep_mem + B) {
            return 0;
        }
        // ABSOLUTE restore floor (--slot-restore-min-tokens, default 0 = off): skip the multi-GB
        // disk load and reprocess instead when the byte-verified, block-aligned matched prefix is
        // below the floor: for a near-cold slot reprocessing a tiny prefix beats paying the NVMe
        // read + H2D copy. Gated on n_keep_disk (the ACTUAL verified match), NOT cand.n_tokens (the
        // snapshot length, which can far exceed the match). Placed after the relative MARGIN gate
        // (which handles "is disk worth more than the resident match") and before the only multi-GB
        // read: return 0 falls through to the caller recomputing n_past + a cold prefill (invariant 4).
        if (n_keep_disk < params_base.slot_restore_min_tokens) {
            return 0;
        }
        // MODE 2: the slot already holds this unit's tokens (it diverged from the request only after them), so
        // its positional K/V for [0, n_keep_disk) is the unit's own and only the side-state at that point is
        // missing (a class that cannot rewind keeps its fold at the slot's end). Load just that, read past the
        // unit's positional bytes, and trim the slot. Text units only, whole-prefix units only (the class rule
        // already demands n_keep_disk == the unit's length), and a memory type that supports it.
        if (side_only_supported && n_match >= n_keep_disk && n_keep_disk == (int) disk_toks.size() &&
            restore_needs_whole_prefix(disk_toks.size()) && disk_media.empty() && !slot.prompt.tokens.has_media() &&
            (int) slot.prompt.tokens.size() > n_keep_disk) {
            const int r = auto_restore_side_only(slot, cand, n_keep_disk);
            rd_lock.release();
            if (r > 0) {
                auto_touch_unit(cand.state_path);
                slot.disk_unit_pending  = (uint32_t) disk_toks.size();
                slot.disk_nodes_pending = 1; // only the tip's side-state is read
                slot.disk_mode_pending  = 2;
                SLT_INF(slot, "auto-restore: reused %d tokens from disk, side-state only, the slot's own %d positional cells kept (in-memory match was %d, usable %d), file=%s\n",
                        n_keep_disk, n_keep_disk, n_match, n_keep_mem, cand.state_path.c_str());
                return n_keep_disk;
            }
            return 0; // the slot was dropped on failure: the caller prefills (or tries a shorter candidate)
        }
        // Build the root->tip chain of node .bin paths via the shared walker. Done BEFORE touching
        // the slot so any inconsistency (missing/corrupt/non-contiguous node) simply returns 0 for a
        // cold prefill, never disturbing the resident KV (invariant 4). `disk_toks` (the tip's full
        // [0, range_hi) record) is the byte-authority each parent prefix is verified against; it is
        // already byte-verified against the request up to `v`, and the compare keeps the legitimate
        // PART mid-parent divergence case restorable. Identical composition for v3 text and v4 media
        // tips, and shared with the manual /slots restore path.
        std::vector<std::string> chain;
        if (!auto_build_restore_chain(cand.state_path, disk_toks, disk_parent_id, disk_range_lo, chain)) {
            return 0;
        }
        // Clear the slot's resident KV before loading the snapshot (mirror the restore-continue safe
        // fallback): seq removal + token/checkpoint clear so the restore writes into an empty seq.
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, -1, -1);
        slot.prompt.tokens.clear();
        slot.prompt.checkpoints.clear();

        const bool restored_ok = do_slot_restore(slot, chain);
        rd_lock.release(); // the state files are loaded (or the load failed); nothing below reads the store
        if (!restored_ok) {
            // restore failed -> do_slot_restore dropped the slot's seq on both contexts; caller reprefills (invariant 4).
            metrics.n_auto_restore_failed++;
            SLT_WRN(slot, "auto-restore: loading %s failed after the slot was cleared; trying a shorter "
                          "candidate or prefilling cold (failed so far = %" PRIu64 ")\n",
                    cand.state_path.c_str(), metrics.n_auto_restore_failed);
            return 0;
        }
        if (!disk_media.empty()) {
            // do_slot_restore filled prompt.tokens with the RAW disk cells: for a media unit
            // those contain LLAMA_TOKEN_NULL cells with no live chunks behind them, which no
            // downstream consumer can traverse. Rebuild the prompt from the REQUEST instead:
            // the verified prefix [0, v) is cell-identical to the snapshot's, and the
            // request's clone carries the live chunks (with pixels) for exactly those cells
            // and the persisted embeddings are already inside the loaded KV state, so no
            // pixel ever needs to come from disk. v is chunk-safe (see the per-record
            // verification above), so keep_first cannot cut mid-image.
            bool rebuilt_ok = false;
            try {
                server_tokens rebuilt = req.clone();
                rebuilt.keep_first(v);
                slot.prompt.tokens = std::move(rebuilt);
                rebuilt_ok = slot.prompt.tokens.validate(ctx_tgt);
            } catch (const std::exception & e) {
                SLT_WRN(slot, "auto-restore: prompt rebuild failed, %s\n", e.what());
            }
            if (!rebuilt_ok) {
                // safety-clear tripwire: a NULL cell without a live chunk (or an unsafe cut)
                // is impossible by construction here: if it happens anyway, drop the
                // restored state entirely and fall back to a clean cold prefill rather than
                // let a later find_chunk() throw mid-decode (invariant 4).
                SLT_WRN(slot, "%s", "auto-restore: rebuilt prompt failed validation; clearing restored state\n");
                auto_restore_drop(slot);
                metrics.n_auto_restore_failed++;
                return 0;
            }
        }
        // SWA models (PART seq_rm, n_swa > 0): the downstream checkpoint search refuses any
        // checkpoint whose pos_max exceeds the request's pos_next, and a fresh process has no
        // checkpoints at all: without one it forces a full re-process, silently discarding the
        // restore. Trim the restored seq to the verified prefix NOW (the snapshot may extend past
        // the request, e.g. by generated tokens) and reconstruct a checkpoint at that boundary,
        // mirroring the FULL branch inside do_slot_restore (which never needs the trim: FULL only
        // restores whole-snapshot extend-matches). Non-SWA attention models skip the checkpoint
        // machinery entirely and need none of this.
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART && n_swa_mem > 0) {
            // The trim runs only for a snapshot that swa_snapshot_is_whole_window() admitted past the
            // whole-prefix gate above; every other SWA snapshot reaches here with v == its length.
            const bool rewound = v < disk_toks.size();
            if (rewound) {
                if (disk_media.empty()) {
                    slot.prompt.tokens.keep_first(v); // media prompts were already rebuilt to exactly v cells
                }
                const llama_pos p0 = slot.prompt.tokens.pos_next();
                llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, p0, -1);
            }
            const auto ckpt_pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
            const auto ckpt_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
            if (rewound && ckpt_pos_min != 0) {
                // The cell-count test admitted this snapshot, but the loaded window does not start at
                // position 0, so the save DID mask cells (a mask type the cell count does not bound)
                // and the rewound prefix would attend over a hole. Drop it and fall through to a
                // shorter candidate or a cold prefill (invariant 4).
                SLT_WRN(slot, "auto-restore: snapshot window starts at pos %d, not 0; a rewind to %zu is unsound, clearing restored state\n",
                        (int) ckpt_pos_min, v);
                auto_restore_drop(slot);
                metrics.n_auto_restore_failed++;
                return 0;
            }
            if (ckpt_pos_min < 0) {
                // the trim emptied the SWA cache: the request diverges more than the SWA window
                // before the snapshot end, so the loaded window holds no position at or below the
                // verified prefix (for an iswa memory, seq_pos_min reports the SWA cache). Keeping
                // the restore would report n_past = v > 0 over an empty window and trip the
                // downstream pos_min==-1 GGML_ABORT: drop it and cold-prefill instead (invariant 4).
                SLT_WRN(slot, "%s", "auto-restore: verified prefix is outside the snapshot's SWA window; clearing restored state\n");
                auto_restore_drop(slot);
                metrics.n_auto_restore_failed++;
                return 0;
            }
            slot.prompt.checkpoints.clear();
            create_checkpoint(slot, 0, ckpt_pos_min, ckpt_pos_max);
        }
        // do_slot_restore loaded the snapshot. For every restore_needs_whole_prefix() snapshot (FULL, RS
        // and SWA past one window) n_keep_disk == snapshot length (gated above), so the existing regenerate /
        // suffix-reuse path takes over with no partial rewind. For plain-attention PART the request may
        // diverge inside the snapshot; keep_first(n_past) + a PARTIAL seq_rm then reprefills the
        // divergent tail (supported for PART). The verified prefix is what we claim as reused.
        // Bump every node on the chain's mtime so the LRU treats a reused-but-not-rewritten base (and
        // each shared delta) as recently-used (true LRU, not least-recently-written): critical for
        // the fan-out case where many requests restore one hot base prefix, and so eviction keeps the
        // whole live chain warm. Best-effort; never errors the restore (invariant 4).
        for (const std::string & node_path : chain) {
            auto_touch_unit(node_path);
        }
        slot.disk_unit_pending  = (uint32_t) disk_toks.size();
        slot.disk_nodes_pending = (uint32_t) chain.size();
        slot.disk_mode_pending  = 1;
        SLT_INF(slot, "auto-restore: reused %d tokens from disk (in-memory match was %d), file=%s\n",
                n_keep_disk, n_keep_mem, cand.state_path.c_str());
        return n_keep_disk;
    }

    // AUTO-SAVE: persist a slot's KV before it is discarded, keyed by its token-prefix block hash.
    // Text-only prompts publish v1 .meta sidecars byte-identical to the pre-media format; media
    // prompts publish v2 sidecars carrying per-chunk identity records (the KV state file already
    // holds the embeddings, so identity metadata is all the disk side needs).
    // The callers skip redundant writes (an equal-or-longer snapshot already covers this prefix). This is
    // the CAPTURE half shared by every save site: the capacity pre-flight, the delta probe and both
    // fallbacks, the copy of the state into host memory, then the hand-off to the background writer
    // (aw_publish), which writes the state + .logits + .dft + .meta temps, syncs them, renames them with
    // .meta LAST so a torn write is never indexed, inserts the index entry, enforces the bounded LRU and
    // reconciles. Invariant 1: first statement is the gate; invariant 5: never during generation
    // (release/reassign, or a mid-prefill node). Persists KV cells [lo, hi) of slot.id's sequence as one
    // disk unit named auto_state_filename(hash, hi):
    //   - a ROOT snapshot when lo == 0 (v1 text / v2 media, `media` selecting which) of the WHOLE
    //     prompt, byte-identical to llama_state_seq_save_file. hi must equal toks.size():
    //     partial [0, hi < N) roots are refused (see the guard at the top of the body);
    //   - a v3 DELTA node when lo > 0 (lo == parent_hi): cells [lo, hi=N) parented on `parent_id`.
    // `hash` is the unit's identity over every cell (auto_block_hashes' `full`), which names the file
    // and is the entry id the exact-unit dedup compares; the index is populated at boundaries
    // bhs[0..kb] inclusive.
    // `site` is the save site that asked (counted per site, requested vs published). Returns true only when
    // the idle flush was DEFERRED (staging full, writer busy): nothing was queued and the caller retries.
    bool auto_publish_snapshot(server_slot & slot,
                               auto_save_site site,
                               llama_context * ctx,
                               const llama_tokens & toks,
                               int32_t lo,
                               int32_t hi,
                               uint64_t hash,
                               const std::vector<uint64_t> & bhs,
                               size_t kb,
                               const model_fp & fp,
                               const std::vector<server_media_record> & media = {},
                               uint64_t parent_id = 0) {
        aw_cnt.site_requested[site]++;
        bool     is_node   = lo > 0;      // lo > 0 <=> a delta parented at parent_hi == lo
        uint32_t parent_hi = (uint32_t) lo; // both cleared below if the U6 delta cell-count check fails
        // Only whole-prefix units exist: hi must equal toks.size(). A PARTIAL [0, hi < N) root would
        // go through the range save, which every class that ignores [p0, p1) (hybrid_idx with an
        // indexer, dsa, dsa_iswa, dsv4, msa, and any recurrent state, which is always "the state
        // after N") serialises as all N cells under a header claiming hi tokens: a mislabelled
        // snapshot that neither the delta probe (is_node only) nor the cell-count check would catch.
        // No caller asks for one (both pass hi == toks.size()), so refuse it outright, loudly.
        if ((size_t) hi != toks.size()) {
            auto_save_note_failure(slot, "partial [0, hi < N) roots are not supported", (size_t) hi);
            return false;
        }
        const llama_tokens & snap_toks = toks;
        const size_t hdr = 3 * sizeof(uint32_t) + snap_toks.size() * sizeof(llama_token);

        // capacity pre-flight (statvfs via std::filesystem::space): refuse to START a multi-GB
        // write the filesystem cannot hold: on btrfs an ENOSPC mid-write can flip the whole
        // filesystem read-only, a far worse failure than a skipped opportunistic save. Exact
        // state size + the token array, with 10% slack covering the file header and the
        // .logits/.meta sidecars, plus what the writer still has queued. An unanswerable space
        // query skips too (conservative; invariant 4: a skipped save never affects generation).
        // The writer checks nothing again: ENOSPC there fails the write cleanly.
        {
            const size_t sz_state = llama_state_seq_get_size(ctx, slot.id);
            // the .dft draft sidecar (when a draft context exists) is written beside the state file and
            // counts against the same filesystem; its whole draft state bounds it from above
            const size_t sz_dft   = ctx_dft ? llama_state_seq_get_size(ctx_dft, slot.id) : 0;
            size_t sz_queued = 0;
            {
                std::lock_guard<std::mutex> lk(aw.mtx);
                sz_queued = aw.queued_bytes;
            }
            const size_t sz_need  = sz_state + sz_dft + snap_toks.size() * sizeof(llama_token) + sz_queued;
            std::error_code sec;
            const auto sinfo = std::filesystem::space(params_base.slot_save_path, sec);
            if (sec || sinfo.available < sz_need + sz_need / 10) {
                SLT_DBG(slot, "auto-save: skipped, insufficient free space (need %zu bytes + 10%% slack, available %zu)\n",
                        sz_need, sec ? 0 : (size_t) sinfo.available);
                auto_save_note_failure(slot, sec ? "free-space query failed" : "insufficient free space", snap_toks.size());
                return false;
            }
        }

        // cross-process atomicity: the temp path MUST be unique per writer. The final name (fname) is
        // deterministic (identity + chain hash + tok count), so two processes sharing one
        // --slot-save-path would otherwise both stream a multi-GB state into the SAME "<fname>.tmp" and
        // interleave -> a corrupt temp gets renamed over a good final file. The temp carries pid + a
        // per-process counter, so each writer owns its own complete temp and the deterministic-name
        // rename is the ONLY shared, atomic step (idempotent: identical content). The sidecar temps
        // derive from this same unique base. The pid is also what lets a later instance reap the temps of
        // a writer that died (auto_reap_dead_temps).
        static std::atomic<uint64_t> s_tmp_nonce{0};
        auto job = std::make_shared<aw_job>();
        job->slot_id = slot.id;
        job->fname   = auto_state_filename(hash, snap_toks.size());
        job->tmp     = job->fname + "." + std::to_string((long) getpid()) + "." +
                       std::to_string(s_tmp_nonce.fetch_add(1, std::memory_order_relaxed)) + ".tmp";
        job->toks    = std::make_shared<const llama_tokens>(snap_toks);
        job->hash    = hash;
        job->bhs     = bhs;
        job->kb      = kb;
        job->fp      = fp;
        job->media   = media;

        // Sizes, the delta probe and the U6 check, all from the size query: no state is copied and nothing
        // is written until the unit is known to be publishable.
        // P0.1: a class that does not honour ranges must never publish a delta. The probe compares the
        // range payload with the whole payload of the same prefix (the file headers are identical, so this
        // is the old range-file vs whole-file test without writing either). Once per instance.
        // F1: a NO answer publishes the same prefix as a WHOLE root instead of dropping it.
        std::vector<uint8_t> head(4096);
        size_t n_payload = 0;
        if (is_node && delta_capable == delta_cap::no) {
            auto_save_note_whole_fallback(slot, snap_toks.size(), parent_hi);
            is_node   = false;
            parent_id = 0;
            parent_hi = 0;
        }
        // U5 (mm-delta decision 1): the range save filters by POSITION, so the boundary is
        // slot.prompt.tokens.pos_next(parent_hi), the same function that assigned the cell positions
        // (mtmd decode seeds on pos_next). For text pos_next == parent_hi; for media pos_next is correct.
        const llama_pos lo_pos = is_node ? slot.prompt.tokens.pos_next((llama_pos) lo) : -1;
        if (is_node) {
            n_payload = llama_state_seq_get_size_range(ctx, slot.id, lo_pos, -1, head.data(), head.size());
            if (n_payload > 0 && delta_capable == delta_cap::unknown) {
                const size_t n_whole = llama_state_seq_get_size_range(ctx, slot.id, -1, -1, nullptr, 0);
                if (n_whole == 0) {
                    auto_save_note_failure(slot, "state size query failed", snap_toks.size());
                    return false;
                }
                delta_capable = n_payload < n_whole ? delta_cap::yes : delta_cap::no;
                SRV_INF("auto disk cache: delta capability probed = %s (range %zu B vs whole %zu B)\n",
                        delta_capable == delta_cap::yes ? "YES" : "NO (whole roots only)", hdr + n_payload, hdr + n_whole);
                if (delta_capable == delta_cap::no) {
                    auto_save_note_whole_fallback(slot, snap_toks.size(), parent_hi);
                    is_node   = false;
                    parent_id = 0;
                    parent_hi = 0;
                }
            }
        }
        // 1b) U6 (mm-delta decision 4): a range save selects suffix cells by POSITION, so a mid-chunk anomaly
        //     could leave the boundary value right yet silently drop or duplicate suffix cells, and the
        //     restore's byte-verify is NULL-blind. The size query reports the payload's leading cell count:
        //     require it to equal N - parent_hi, and otherwise fall back to a WHOLE save of this exact prefix
        //     (is_node cleared => the meta is v1/v2 by `media`, byte-identical to a no-parent save).
        if (is_node && n_payload > 0) {
            uint32_t cells = 0;
            const bool ok = state_payload_cell_count(head.data(), std::min(head.size(), n_payload), cells) &&
                            (size_t) cells == snap_toks.size() - parent_hi;
            if (!ok) {
                SLT_WRN(slot, "auto-save: delta cell-count check failed (expected %zu, got %u); "
                              "falling back to a whole snapshot\n",
                        snap_toks.size() - parent_hi, cells);
                is_node   = false;
                parent_id = 0;
                parent_hi = 0;
            }
        }
        if (!is_node) {
            n_payload = llama_state_seq_get_size_range(ctx, slot.id, -1, -1, nullptr, 0);
        }
        if (n_payload == 0) {
            auto_save_note_failure(slot, "state size query failed", snap_toks.size());
            return false;
        }
        // P0.4: refuse to publish a snapshot with NO memory payload. llama_kv_cache::state_write and
        // state_read are SILENT NO-OPS when the cache is a shared view (`if (other) return false;`,
        // src/llama-kv-cache.cpp), as on LLM_ARCH_GEMMA4_ASSISTANT, whose sub-caches both carry mem_other.
        // The .meta would claim N tokens and a later restore would load NOTHING while reporting success.
        // Class-agnostic guard: a real snapshot must be substantially larger than its own header + token
        // array. (The early return in the engine is CORRECT for a pure view sharing v_cells_impl: we refuse
        // the empty SNAPSHOT, never the model.)
        if (hdr + n_payload <= 256 + snap_toks.size() * sizeof(llama_token)) {
            SRV_WRN("auto disk cache: refusing to publish a snapshot with no memory payload "
                    "(%zu B for %zu tokens) - the memory type serialised nothing\n",
                    hdr + n_payload, snap_toks.size());
            aw_cnt.failed++;
            return false;
        }
        job->is_node        = is_node;
        job->parent_id      = is_node ? parent_id : 0;
        job->parent_hi      = is_node ? parent_hi : 0;
        job->parent_fname   = is_node ? auto_state_filename(parent_id, parent_hi) : std::string();
        // Periodic prefill checkpoints of a class that writes no deltas (or with --slot-save-incremental off) are each a
        // whole unit of the prefix so far, so a prefill of L tokens would write N + 2N + ... + L. Each whole checkpoint
        // replaces the previous one of the same prefill once it is published: the deepest is kept (a resume, or a new
        // question at the end of the same long message, needs only that one). Nothing can parent on these units, since
        // this instance writes no deltas.
        if (site == AUTO_SAVE_SITE_PREFILL_CKPT) {
            // (a root written before the delta probe ran counts too: if the next checkpoint turns out to be a delta on
            // it, the record is cleared there and nothing is replaced)
            if (!is_node && (delta_capable != delta_cap::yes || !params_base.slot_save_incremental)) {
                if (!slot.ckpt_prev_whole.empty() && slot.ckpt_prev_whole != job->fname) {
                    job->supersedes = slot.ckpt_prev_whole;
                }
                slot.ckpt_prev_whole = job->fname;
            } else {
                slot.ckpt_prev_whole.clear();
            }
        }
        job->bin.n_expected = hdr + n_payload;

        // 2) logits sidecar (FULL and RS, and only when the captured distribution provably belongs to this
        //    exact state, the same stamp check SLOT_SAVE uses)
        const int64_t t_l0 = ggml_time_us();
        if (logits_sidecar_class() &&
            slot.logits_last_n_tokens == (int32_t) snap_toks.size() && !slot.logits_last.empty()) {
            job->logits  = slot.logits_last;
            job->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
        }
        job->t_logits = ggml_time_us() - t_l0;

        // 2b) draft sidecar, with the same [lo, N) split the state file ended up with. Sized and checked
        //     here; its trailer (the draft's carry-over for the unit's last cell) is captured with it.
        llama_pos dft_p0 = -1, dft_p1 = -1;
        std::vector<uint8_t> dft_trailer;
        if (ctx_dft) {
            const uint32_t cell_lo = is_node ? parent_hi : 0;
            if (auto_draft_sidecar_plan(slot, snap_toks, cell_lo, hdr, dft_p0, dft_p1, job->dft.n_expected, dft_trailer)) {
                job->has_dft         = true;
                job->dft.n_expected += dft_trailer.size();
            }
        }

        // 2c) stage 2: when the memory class has positional cells, capture only the side-state now and leave
        //     the positional K/V in the cache, copied later (aw_def_*). Same bytes, same unit.
        if (aw_capture_deferred(slot, job, ctx, is_node ? lo_pos : -1, dft_p0, dft_p1, dft_trailer, site)) {
            return false;
        }

        // 3) admission, then the copy. The job is queued before the copy starts, so a staged job's first
        //    chunks are already being written while the rest is copied.
        const size_t n_total = job->bin.n_expected + job->dft.n_expected + job->logits.size() * sizeof(float);
        FILE * f_bin = nullptr;
        FILE * f_dft = nullptr;
        if (aw.running && !aw.def.empty()) {
            // The writer is FIFO and owes the deferred captures first. A save captured at once now would hold
            // staging the deferred copies ahead of it need to drain (or wait for a writer that waits for them),
            // so their bytes go first. Only a class or a budget that rules deferral out gets here with copies
            // pending, so this is rare.
            aw_def_force_all("wait");
        }
        if (!aw.running) {
            job->site = site;
            job->mode = aw_mode::sync;
            f_bin = fopen(job->tmp.c_str(), "wb");
            if (!f_bin) {
                auto_save_note_failure(slot, "state write failed (disk full or IO error)", snap_toks.size());
                return false;
            }
        } else {
            job->site = site;
            switch (aw_admit(job, n_total, site == AUTO_SAVE_SITE_IDLE ? aw_policy::defer : aw_policy::wait)) {
                case aw_admit_res::queued:
                    break;
                case aw_admit_res::deferred:
                    aw_cnt.site_requested[site]--; // not an outcome: the idle flush asks again
                    return true;
                case aw_admit_res::dropped:
                    return false;
            }
        }
        const int64_t t_t0 = ggml_time_us();
        const bool ok_bin = aw_capture_stream(*job, job->bin, ctx, is_node ? lo_pos : -1, -1, f_bin, job->bin.n_expected);
        job->t_d2h_tgt = ggml_time_us() - t_t0;
        if (f_bin) {
            const bool ok_file = fflush(f_bin) == 0 &&
#ifndef _WIN32
                                 ::fdatasync(fileno(f_bin)) == 0 &&
#endif
                                 true;
            fclose(f_bin);
            if (!ok_bin || !ok_file) {
                std::error_code ec;
                std::filesystem::remove(job->tmp, ec);
                auto_save_note_failure(slot, "state write failed (disk full or IO error)", snap_toks.size());
                return false;
            }
        } else {
            aw_finish_stream(job->bin, ok_bin);
            if (!ok_bin) {
                aw_finish_stream(job->dft, false);
                auto_save_note_failure(slot, "the state copy failed or changed size during the copy", snap_toks.size());
                return false; // the writer drops the job and its queued children
            }
        }
        if (job->has_dft) {
            const int64_t t_d0 = ggml_time_us();
            bool ok_dft = true;
            if (job->mode == aw_mode::sync) {
                f_dft  = fopen(slot_draft_sidecar_path(job->tmp).c_str(), "wb");
                ok_dft = f_dft != nullptr;
            }
            // the engine produces the state part, the trailer follows it
            ok_dft = ok_dft && aw_capture_stream(*job, job->dft, ctx_dft, dft_p0, dft_p1, f_dft,
                                                 job->dft.n_expected - dft_trailer.size());
            if (ok_dft && !dft_trailer.empty()) {
                aw_sink_ctx sc;
                sc.self = this; sc.job = job.get(); sc.st = &job->dft; sc.direct = f_dft;
                ok_dft = aw_push_bytes(sc, dft_trailer.data(), dft_trailer.size());
                if (ok_dft && sc.cur.n > 0) {
                    aw_push_chunk(sc);
                }
                ok_dft = ok_dft && !sc.failed;
            }
            if (!ok_dft) {
                aw_cnt.draft_skipped++;
                SLT_DBG(slot, "auto-save: no draft sidecar for this unit (the draft copy failed)%s\n", "");
            }
            // the writer reads the timing once the stream is finished, so it is set before that
            job->t_d2h_dft = ggml_time_us() - t_d0;
            if (f_dft) {
                ok_dft = fflush(f_dft) == 0 && ok_dft;
#ifndef _WIN32
                ok_dft = ::fdatasync(fileno(f_dft)) == 0 && ok_dft;
#endif
                fclose(f_dft);
                if (!ok_dft) {
                    std::error_code ec;
                    std::filesystem::remove(slot_draft_sidecar_path(job->tmp), ec);
                }
            }
            if (job->mode == aw_mode::sync) {
                job->has_dft = ok_dft;
            } else {
                aw_finish_stream(job->dft, ok_dft); // an aborted stream makes the writer drop the file
            }
        }
        if (job->mode == aw_mode::sync) {
            aw_publish(*job);
        }
        return false;
    }

    // ===== BACKGROUND WRITER (auto disk cache) ==========================================================
    // A save is split in two. CAPTURE runs on the server thread, at the instant the save is decided: every
    // gate, the delta probe, the parent choice, and the device-to-host copy of the state into host memory
    // (llama_state_seq_save_sink). PUBLISH runs on one writer thread per instance, FIFO: the temp files,
    // fdatasync, the renames under the store lock (.bin, .logits, .dft, then .meta), the index insert, the
    // LRU and the success log. The writer owns no llama object, so it never touches a context.
    //
    // Staging (--slot-save-staging-mb, default 1024): a job whose bytes fit in the free budget is staged
    // whole and the server pays only the copy. One that does not fit streams through a two-chunk ring
    // when the writer is idle (the server then pays about max(copy, write)). When the writer is busy the
    // capture WAITS for it (llamacpp:auto_cache_save_admission_waits_total and _wait_seconds_total), so a
    // save is never dropped for want of staging: no save site is covered by a later one on every class (a
    // prompt node on a class that cannot rewind is the only unit the next turn can restore). The wait is no
    // longer than the synchronous save it replaces would have taken, since it waits only for writes the
    // synchronous code would already have done on this thread. Two exceptions: the idle flush DEFERS
    // instead (nothing is queued, and it retries once the writer is idle, so a request arriving meanwhile
    // is not held), and a writer that makes no progress for AW_STALL_MS (a hung disk) has the waiting save
    // dropped, with a WRN and llamacpp:auto_cache_save_dropped_staging_total, rather than the server loop
    // hung for good. The shutdown flush waits up to its deadline. Staging is allocated per job and freed as
    // the writer drains it, so an idle instance holds none; it is plain pageable host memory, at most the
    // budget plus one ring per instance, on top of --cache-ram and GTT. A budget of 0 publishes on the
    // server thread instead (the synchronous mode, streaming straight into the temp).
    //
    // A unit is visible to restores only once its .meta is renamed into place (restores read the index,
    // never the queue). The queue itself (the PENDING set) is consulted by the save-side dedup and parent
    // choice, so a child can be a delta on a parent that is still queued; a parent that fails takes its
    // queued children with it (llamacpp:auto_cache_save_orphan_dropped_total), because staged delta bytes
    // cannot become a root. Neither loss is final while the slot still holds the conversation: an orphaned
    // child's slot, and every slot whose save was skipped because a queued unit already covered it, are
    // re-armed for the idle flush when that unit fails (aw.resave_slots), and the re-save then picks a
    // published parent or writes a root.

    struct aw_chunk {
        std::unique_ptr<uint8_t[]> buf;
        size_t cap = 0;
        size_t n   = 0;
    };

    // one file's bytes, produced by the server thread and consumed by the writer (guarded by aw.mtx)
    struct aw_stream {
        std::deque<aw_chunk> chunks;    // produced, not yet written
        bool   sealed     = false;      // the producer is done
        bool   aborted    = false;      // the producer gave up: the consumer drops the file
        size_t n_expected = 0;          // exact size the producer announced
        size_t n_produced = 0;
    };

    enum class aw_mode : uint8_t { staged, streamed, sync, deferred };

    struct aw_job;

    // the sink the engine copies the state into, one per stream being captured (or emitted, stage 2)
    struct aw_sink_ctx {
        server_context_impl * self = nullptr;
        aw_job *              job  = nullptr;
        aw_stream *           st   = nullptr;
        aw_chunk              cur;
        FILE *                direct   = nullptr; // sync mode: chunks go straight into this temp file
        bool                  failed   = false;
        bool                  blocking = true;    // deferred emission: wait for staging room (a forced copy)
        bool                  paused   = false;   // deferred emission: no room now, continue later
    };

    // Stage 2: the part of one stream the deferred capture still owes. The engine handle references the
    // positional K/V still in the cache; `tail` follows the engine's bytes (the .dft trailer).
    struct aw_def_part {
        llama_state_deferred * h        = nullptr; // server thread only, freed once emitted
        std::vector<uint8_t>   tail;
        size_t                 tail_off = 0;
        size_t                 host     = 0;       // host bytes the handle holds (counted in aw.deferred_host)
        bool                   active   = false;
        bool                   finished = false;
        aw_sink_ctx            sc;
    };

    struct aw_job {
        int         slot_id = -1;
        std::string fname;                         // final .bin name
        std::string tmp;                           // "<fname>.<pid>.<nonce>.tmp"
        std::string parent_fname;                  // the parent's .bin (delta only)
        std::shared_ptr<const llama_tokens> toks;  // the unit's cell tokens
        uint64_t    hash      = 0;                 // identity over every cell (names the file)
        std::vector<uint64_t> bhs;                 // boundary keys, indexed at [0, kb]
        size_t      kb        = 0;
        model_fp    fp;
        std::vector<server_media_record> media;
        bool        is_node   = false;
        uint64_t    parent_id = 0;
        uint32_t    parent_hi = 0;
        std::vector<float> logits;                 // empty: no .logits sidecar
        int32_t     n_vocab   = 0;
        bool        has_dft   = false;
        aw_stream   bin;
        aw_stream   dft;
        aw_mode     mode      = aw_mode::sync;
        auto_save_site site   = AUTO_SAVE_SITE_RECLAIM;
        std::vector<std::pair<int, auto_save_site>> shadowed; // saves skipped because this job covers them
        std::string supersedes;                    // a whole periodic checkpoint of the same prefill this one replaces
        size_t      reserved  = 0;                 // staged budget still held by this job
        bool        done      = false;
        bool        published = false;             // set with done: the unit is in the store
        // timings, microseconds
        int64_t     t_d2h_tgt = 0;
        int64_t     t_d2h_dft = 0;
        int64_t     t_logits  = 0;
        int64_t     t_enq     = 0;
        // stage 2 (mode deferred): the positional bytes are emitted after the capture, on the server thread
        aw_def_part def_bin;
        aw_def_part def_dft;
        size_t      n_deferred = 0;                // positional bytes left in the cache at the capture
        int64_t     t_emit     = 0;                // time spent emitting them
        int64_t     t_emitted  = 0;                // when the last byte was emitted (us)
        int         n_forced   = 0;                // times a cache mutation forced the copy
        const char * def_how   = "";               // what finished the copy: idle, trickle, forced, wait, sleep, shutdown

        ~aw_job() {
            // normally freed (on the server thread) as soon as emitted; only a teardown leaves one here
            llama_state_deferred_free(def_bin.h);
            llama_state_deferred_free(def_dft.h);
        }
    };

    struct aw_state {
        std::mutex              mtx;
        std::condition_variable cv;              // every wait (jobs, chunks, budget, publish) uses this one
        std::deque<std::shared_ptr<aw_job>> jobs; // FIFO; the front is the one being published
        std::thread             th;
        bool                    started   = false;
        bool                    running   = false;
        bool                    closing   = false;
        bool                    shutdown  = false; // the shutdown flush: admission waits up to the deadline
        std::atomic<int64_t>    deadline_ms{-1};   // shutdown: abandon what is left after this (read lock-free)
        int64_t                 last_progress_ms = -1; // the writer's last sign of life (a chunk, a stage, a job)
        std::vector<int>        resave_slots;      // slots to re-arm for the idle flush (a unit covering them failed)
        size_t                  budget    = 0;
        size_t                  chunk     = 0;
        size_t                  staged    = 0;     // bytes held in chunks not yet written
        size_t                  queued_bytes = 0;  // bytes of queued jobs not yet on disk (capacity check)
        std::deque<uint64_t>    failed_units;      // recent failed unit ids (orphan detection), bounded
        // test hooks (environment, read once): slow the writer down, fail the k-th .meta rename
        int64_t                 test_delay_ms  = 0;
        int64_t                 test_fail_meta = 0;
        int64_t                 test_prefill_delay_ms = 0; // LLAMA_TEST_SLOT_SAVE_PREFILL_DELAY_MS
        int64_t                 n_publishes    = 0;
        int64_t                 reap_age_s     = 600;
        int64_t                 shutdown_ms    = 90 * 1000;
        int64_t                 stall_ms       = 60 * 1000;
        int64_t                 last_reap_ms   = -1;
        // stage 2: deferred captures still owing positional bytes, FIFO (server thread only)
        std::deque<std::shared_ptr<aw_job>> def;
        size_t                  deferred_host  = 0;     // host bytes held by deferred captures (guarded by mtx)
        size_t                  def_count      = 0;     // def.size(), for the metrics (guarded by mtx)
        bool                    defer          = false; // --slot-save-defer, with the writer running
        bool                    defer_idle     = true;  // test hook: false holds deferred copies until forced
        int64_t                 defer_trickle_ms = 10 * 1000; // a copy pending this long is trickled while busy
        size_t                  defer_trickle_slice = 8u << 20; // ... this many bytes per AW_DEF_TRICKLE_PERIOD_US of busy time
        int64_t                 trickle_last_us  = 0;    // the previous busy loop iteration (server thread)
        int64_t                 def_retry_ms   = -1;    // the idle emission found no staging room: retry then
        int64_t                 def_retry_step_ms = 0;  // ... after this delay, doubling while no room appears
    } aw;

    static constexpr size_t  AW_MAX_CHUNK            = 64u << 20;
    // inside TimeoutStopSec 120 / router 150, leaving 30 s for llama_backend_free: past the deadline the writer
    // stops between chunks, and the per-chunk writeback (aw_write_stream) keeps the last fdatasync short
    static constexpr int64_t AW_SHUTDOWN_DEADLINE_MS = 90 * 1000;
    static constexpr int64_t AW_PENDING_WAIT_MS      = 30 * 1000;  // a new task waits this long for a queued prefix
    static constexpr int64_t AW_STALL_MS             = 60 * 1000;  // a writer silent this long is treated as hung
    static constexpr int64_t AW_IDLE_RETRY_MS        = 200;        // a deferred idle flush re-checks this often
    static constexpr size_t  AW_DEF_IDLE_SLICE       = 64u << 20;  // deferred bytes emitted per idle wakeup
    static constexpr size_t  AW_DEF_TRICKLE_SLICE    = 8u << 20;   // ... per AW_DEF_TRICKLE_PERIOD_US of busy time, once overdue
    static constexpr int64_t AW_DEF_TRICKLE_PERIOD_US = 50 * 1000; // about one decode iteration
    static constexpr size_t  AW_DEF_TRICKLE_MAX      = 1024u << 20; // at most this per loop iteration
    static constexpr int64_t AW_DEF_RETRY_MS         = 5;          // idle emission without staging room first re-checks after this

    static int64_t aw_env_i64(const char * name, int64_t def) {
        const char * v = getenv(name);
        return v && *v ? std::strtoll(v, nullptr, 10) : def;
    }

    void aw_start() {
        if (aw.started || !auto_cache_enabled()) {
            return; // once per process (load_model runs again when the server wakes from sleep)
        }
        aw.started = true;
        const int64_t mb = std::max<int64_t>(0, params_base.slot_save_staging_mb);
        aw.budget         = (size_t) mb << 20;
        aw.chunk          = std::clamp<size_t>(aw.budget / 2, 4096, AW_MAX_CHUNK);
        aw.test_delay_ms  = aw_env_i64("LLAMA_TEST_SLOT_SAVE_WRITER_DELAY_MS", 0);
        aw.test_fail_meta = aw_env_i64("LLAMA_TEST_SLOT_SAVE_FAIL_META_AT", 0);
        aw.test_prefill_delay_ms = aw_env_i64("LLAMA_TEST_SLOT_SAVE_PREFILL_DELAY_MS", 0);
        aw.reap_age_s     = aw_env_i64("LLAMA_TEST_SLOT_SAVE_TMP_REAP_AGE_S", 600);
        aw.shutdown_ms    = aw_env_i64("LLAMA_TEST_SLOT_SAVE_SHUTDOWN_DEADLINE_MS", AW_SHUTDOWN_DEADLINE_MS);
        aw.stall_ms       = aw_env_i64("LLAMA_TEST_SLOT_SAVE_WRITER_STALL_MS", AW_STALL_MS);
        aw.defer_idle       = aw_env_i64("LLAMA_TEST_SLOT_SAVE_DEFER_NO_IDLE", 0) == 0;
        aw.defer_trickle_ms = aw_env_i64("LLAMA_TEST_SLOT_SAVE_DEFER_TRICKLE_MS", 10 * 1000);
        aw.defer_trickle_slice = (size_t) std::max<int64_t>(1, aw_env_i64("LLAMA_TEST_SLOT_SAVE_DEFER_TRICKLE_SLICE",
                                                                          (int64_t) AW_DEF_TRICKLE_SLICE));
        if (aw_env_i64("LLAMA_TEST_SLOT_SAVE_DEFER_NO_HOOK", 0) != 0) {
            // test hook, the positive control: a mutation no longer forces a pending copy
            llama_state_deferred_set_hook_enabled(false);
            SRV_WRN("%s", "auto disk cache: TEST HOOK: the flush-on-mutate hook is disabled\n");
        }
        if (params_base.slot_save_prefill_interval > 0) {
            // Each checkpoint is a unit with the whole side-state at its end (about 150 MB per node on a 27B gated
            // delta net), and a delta chain is evictable only from its tip, so the node count of one prefill is the
            // cost to watch: n_ctx / N nodes for a prompt that fills the context.
            const int32_t n_ctx_seq = (int32_t) llama_n_ctx_seq(ctx_tgt);
            const int32_t n_nodes   = n_ctx_seq / params_base.slot_save_prefill_interval;
            if (n_nodes > AUTO_CKPT_NODES_RECOMMENDED) {
                SRV_WRN("auto disk cache: --slot-save-prefill-interval %d gives up to %d checkpoints for a prompt that fills "
                        "this context (%d tokens); each carries the whole side-state, so consider N >= %d "
                        "(n_ctx / %d)\n", params_base.slot_save_prefill_interval, n_nodes, n_ctx_seq,
                        n_ctx_seq / AUTO_CKPT_NODES_RECOMMENDED, AUTO_CKPT_NODES_RECOMMENDED);
            } else {
                SRV_INF("auto disk cache: periodic prefill checkpoints every %d tokens, up to %d per prompt\n",
                        params_base.slot_save_prefill_interval, n_nodes);
            }
        }
        auto_reap_dead_temps();
        if (aw.budget == 0) {
            SRV_INF("%s", "auto disk cache: saves are published on the server thread (--slot-save-staging-mb 0)\n");
            return;
        }
        aw.running = true;
        aw.defer   = params_base.slot_save_defer;
        aw.th = std::thread([this]() { aw_loop(); });
        SRV_INF("auto disk cache: background writer started, staging budget %zu MiB, chunk %zu KiB, positional copy %s\n",
                aw.budget >> 20, aw.chunk >> 10, aw.defer ? "deferred (stage 2)" : "at the capture");
    }

    // Closes the queue and joins the writer. What is still queued at `deadline_ms` is abandoned: the
    // writer stops between chunks, unlinks its temps and never renames a partial unit.
    void aw_stop(int64_t deadline_ms) {
        if (!aw.running) {
            return;
        }
        {
            std::lock_guard<std::mutex> lk(aw.mtx);
            aw.closing     = true;
            aw.deadline_ms = deadline_ms;
        }
        aw.cv.notify_all();
        aw.th.join();
        aw.running = false;
    }

    bool aw_past_deadline() const {
        const int64_t d = aw.deadline_ms.load(std::memory_order_relaxed);
        return d >= 0 && ggml_time_ms() >= d;
    }

    // the writer is alive (called by the writer at every chunk and stage)
    void aw_progress() {
        std::lock_guard<std::mutex> lk(aw.mtx);
        aw.last_progress_ms = ggml_time_ms();
    }

    // under aw.mtx: the writer has work and has shown no sign of life for stall_ms (a hung disk)
    bool aw_stalled_locked() const {
        return !aw_idle_locked() && aw.last_progress_ms >= 0 && ggml_time_ms() - aw.last_progress_ms >= aw.stall_ms;
    }

    // ---- capture side (server thread) -----------------------------------------------------------------

    // hand a filled chunk to the consumer (the writer, or the temp file in sync mode)
    void aw_push_chunk(aw_sink_ctx & sc) {
        if (sc.cur.n == 0) {
            return;
        }
        if (sc.direct) {
            if (fwrite(sc.cur.buf.get(), 1, sc.cur.n, sc.direct) != sc.cur.n) {
                sc.failed = true;
            }
            sc.cur.n = 0; // the buffer is reused
            return;
        }
        {
            std::lock_guard<std::mutex> lk(aw.mtx);
            sc.st->chunks.push_back(std::move(sc.cur));
            if (sc.job->mode == aw_mode::deferred) {
                aw.last_progress_ms = ggml_time_ms(); // a writer waiting for a deferred copy is not stalled
            }
        }
        sc.cur = {};
        aw.cv.notify_all();
    }

    // appends bytes the engine does not produce (the .dft trailer) to a stream being captured
    bool aw_push_bytes(aw_sink_ctx & sc, const void * src, size_t n) {
        const uint8_t * p = (const uint8_t *) src;
        while (n > 0) {
            size_t avail = 0;
            void * dst = aw_reserve(sc, n, &avail);
            if (dst == nullptr) {
                return false;
            }
            memcpy(dst, p, avail);
            aw_sink_commit(&sc, avail);
            p += avail;
            n -= avail;
        }
        return true;
    }

    static void * aw_sink_reserve(void * ud, size_t n, size_t * n_avail) {
        auto & sc = *(aw_sink_ctx *) ud;
        return sc.self->aw_reserve(sc, n, n_avail);
    }

    static void aw_sink_commit(void * ud, size_t n) {
        auto & sc = *(aw_sink_ctx *) ud;
        sc.cur.n          += n;
        sc.st->n_produced += n;
        if (sc.cur.n == sc.cur.cap) {
            sc.self->aw_push_chunk(sc);
        }
    }

    void * aw_reserve(aw_sink_ctx & sc, size_t n, size_t * n_avail) {
        if (sc.failed) {
            return nullptr;
        }
        if (!sc.cur.buf || sc.cur.n == sc.cur.cap) {
            const size_t left = sc.st->n_expected > sc.st->n_produced ? sc.st->n_expected - sc.st->n_produced : n;
            if (sc.direct) {
                // sync mode: one buffer, reused for every piece
                if (!sc.cur.buf) {
                    const size_t cap = std::clamp<size_t>(left, 1, AW_MAX_CHUNK);
                    sc.cur.buf.reset(new (std::nothrow) uint8_t[cap]);
                    if (!sc.cur.buf) {
                        sc.failed = true;
                        return nullptr;
                    }
                    sc.cur.cap = cap;
                }
                sc.cur.n = 0;
                *n_avail = std::min(n, sc.cur.cap);
                return sc.cur.buf.get();
            }
            const size_t cap = std::clamp<size_t>(left, 1, aw.chunk);
            if (sc.job->mode == aw_mode::deferred) {
                // stage 2: chunks in flight share the staging budget; a forced copy waits for room like a
                // streamed save, an idle one pauses and continues on a later wakeup
                std::unique_lock<std::mutex> lk(aw.mtx);
                const size_t room = std::max(aw.budget, 2 * aw.chunk);
                auto fits = [&]() { return aw.staged == 0 || aw.staged + cap <= room; };
                if (!fits() && !sc.blocking) {
                    sc.paused = true;
                    return nullptr;
                }
                bool stalled = false;
                while (!aw.cv.wait_for(lk, std::chrono::milliseconds(500),
                                       [&]() { return fits() || sc.st->aborted || aw_past_deadline(); })) {
                    if (aw_stalled_locked()) {
                        stalled = true;
                        break;
                    }
                }
                if (stalled || sc.st->aborted || aw_past_deadline()) {
                    if (stalled) {
                        SRV_WRN("slot %d: auto-save: the background writer made no progress for %" PRId64 " ms, "
                                "abandoning a deferred save\n", sc.job->slot_id, aw.stall_ms);
                    }
                    sc.failed = true;
                    return nullptr;
                }
                aw.staged += cap;
            } else if (sc.job->mode == aw_mode::streamed) {
                // the ring of two chunks: wait for the writer to drain it, never past the shutdown deadline
                std::unique_lock<std::mutex> lk(aw.mtx);
                const size_t ring = 2 * aw.chunk;
                bool stalled = false;
                while (!aw.cv.wait_for(lk, std::chrono::milliseconds(500),
                                       [&]() { return aw.staged + cap <= ring || sc.st->aborted || aw_past_deadline(); })) {
                    if (aw_stalled_locked()) {
                        stalled = true;
                        break;
                    }
                }
                if (stalled || sc.st->aborted || aw_past_deadline()) {
                    if (stalled) {
                        SRV_WRN("slot %d: auto-save: the background writer made no progress for %" PRId64 " ms, "
                                "abandoning a streamed save\n", sc.job->slot_id, aw.stall_ms);
                    }
                    sc.failed = true;
                    return nullptr;
                }
                aw.staged += cap;
            }
            sc.cur.buf.reset(new (std::nothrow) uint8_t[cap]);
            if (!sc.cur.buf) {
                if (sc.job->mode == aw_mode::streamed || sc.job->mode == aw_mode::deferred) {
                    {
                        std::lock_guard<std::mutex> lk(aw.mtx);
                        aw.staged -= std::min(aw.staged, cap); // the ring slot this chunk reserved
                    }
                    aw.cv.notify_all();
                }
                sc.failed = true;
                return nullptr;
            }
            sc.cur.cap = cap;
            sc.cur.n   = 0;
        }
        *n_avail = std::min(n, sc.cur.cap - sc.cur.n);
        return sc.cur.buf.get() + sc.cur.n;
    }

    // Copies [p0, p1) (whole when both are negative) of `ctx` into `st`, exactly as the file save would
    // write it. Returns false if the engine failed, the sink refused, or the byte count is not the one the
    // size query announced (n_engine) (the state must not change between the two; a difference means a defect, and
    // the unit is dropped rather than published with a wrong length).
    bool aw_capture_stream(aw_job & job, aw_stream & st, llama_context * ctx, llama_pos p0, llama_pos p1,
                           FILE * direct, size_t n_engine) {
        aw_sink_ctx sc;
        sc.self   = this;
        sc.job    = &job;
        sc.st     = &st;
        sc.direct = direct;
        const llama_state_sink sink = { aw_sink_reserve, aw_sink_commit, &sc };
        const size_t n = llama_state_seq_save_sink(ctx, job.slot_id, p0, p1, job.toks->data(), job.toks->size(), &sink);
        if (sc.cur.n > 0 && !sc.failed) {
            aw_push_chunk(sc);
        }
        return n > 0 && !sc.failed && n == n_engine;
    }

    // end of a stream: sealed (complete) or aborted (the consumer drops the file)
    void aw_finish_stream(aw_stream & st, bool ok) {
        {
            std::lock_guard<std::mutex> lk(aw.mtx);
            if (ok) {
                st.sealed = true;
            } else {
                st.aborted = true;
            }
        }
        aw.cv.notify_all();
    }

    // Parses the first non-empty stream's cell count from the head of a state payload (the layout
    // delta_bin_cell_count documents). Returns false when the head is too short or malformed.
    static bool state_payload_cell_count(const uint8_t * p, size_t n, uint32_t & cells_out) {
        size_t off = 0;
        auto rd = [&](uint32_t & v) -> bool {
            if (off + 4 > n) {
                return false;
            }
            v = (uint32_t) p[off] | ((uint32_t) p[off + 1] << 8) | ((uint32_t) p[off + 2] << 16) | ((uint32_t) p[off + 3] << 24);
            off += 4;
            return true;
        };
        uint32_t n_stream = 0;
        if (!rd(n_stream)) {
            return false;
        }
        static constexpr uint32_t DSV4_STATE_MAGIC = 0x34565344; // "DSV4", src/llama-kv-cache-dsv4.cpp
        if (n_stream == DSV4_STATE_MAGIC) {
            uint32_t ver = 0, mode = 0;
            if (!rd(ver) || !rd(mode) || !rd(n_stream)) {
                return false;
            }
        }
        for (uint32_t s = 0; s < n_stream; ++s) {
            uint32_t cell_count = 0;
            if (!rd(cell_count)) {
                return false;
            }
            if (cell_count != 0) {
                cells_out = cell_count;
                return true;
            }
        }
        cells_out = 0;
        return true;
    }

    // the writer has nothing to do (a finished job may still sit at the front for a moment)
    bool aw_idle_locked() const {
        return std::all_of(aw.jobs.begin(), aw.jobs.end(), [](const std::shared_ptr<aw_job> & j) { return j->done; });
    }

    // Admission (section "Staging"): decides staged / streamed for a job of `total` bytes and queues it.
    // When neither fits (staging full, writer busy) a `wait` capture waits for the writer, a `defer` one
    // (the idle flush) returns at once so its caller retries, and the shutdown flush waits up to its
    // deadline. Only a writer that stalls (or the shutdown deadline) drops a job.
    enum class aw_policy : uint8_t { wait, defer };
    enum class aw_admit_res : uint8_t { queued, deferred, dropped };

    aw_admit_res aw_admit(const std::shared_ptr<aw_job> & job, size_t total, aw_policy policy) {
        std::unique_lock<std::mutex> lk(aw.mtx);
        auto decide = [&]() -> bool {
            if (aw.staged + total <= aw.budget) {
                job->mode     = aw_mode::staged;
                job->reserved = total;
                aw.staged    += total;
                return true;
            }
            if (aw_idle_locked()) {
                job->mode = aw_mode::streamed;
                return true;
            }
            return false;
        };
        bool ok = decide();
        const char * why = "the shutdown deadline passed";
        if (!ok && !aw.shutdown && policy == aw_policy::defer) {
            return aw_admit_res::deferred;
        }
        if (!ok) {
            const int64_t t0 = ggml_time_us();
            if (aw.shutdown) {
                ok = aw.cv.wait_until(lk, std::chrono::steady_clock::now() +
                                          std::chrono::milliseconds(std::max<int64_t>(0, aw.deadline_ms.load() - ggml_time_ms())),
                                      [&]() { return decide(); });
            } else {
                why = "the background writer made no progress";
                while (!(ok = aw.cv.wait_for(lk, std::chrono::milliseconds(500), [&]() { return decide(); }))) {
                    if (aw_stalled_locked()) {
                        break;
                    }
                }
            }
            const int64_t t_wait = ggml_time_us() - t0;
            aw_cnt.admission_waits++;
            aw_cnt.admission_wait_us += (uint64_t) t_wait;
            if (ok) {
                SRV_INF("slot %d: auto-save: waited %.1f ms for the writer to take a %zu-token %s (%zu B; staging %zu/%zu MB)\n",
                        job->slot_id, t_wait / 1000.0, job->toks->size(), job->is_node ? "delta" : "root", total,
                        aw.staged >> 20, aw.budget >> 20);
            }
        }
        if (!ok) {
            const size_t used = aw.staged;
            const size_t qd   = aw.jobs.size();
            lk.unlock();
            aw_cnt.dropped_staging++;
            SRV_WRN("slot %d: auto-save: dropped %s %zu tokens (%zu B), %s (staging %zu/%zu MB, queue %zu)\n",
                    job->slot_id, job->is_node ? "delta" : "root", job->toks->size(), total, why, used >> 20,
                    aw.budget >> 20, qd);
            return aw_admit_res::dropped;
        }
        job->t_enq = ggml_time_us();
        aw.queued_bytes += job->bin.n_expected + job->dft.n_expected;
        if (aw_idle_locked()) {
            aw.last_progress_ms = ggml_time_ms(); // the writer's silence is measured from its first job
        }
        aw.jobs.push_back(job);
        aw_cnt.queued++;
        if (job->mode == aw_mode::streamed) {
            aw_cnt.streamed++;
        }
        lk.unlock();
        aw.cv.notify_all();
        return aw_admit_res::queued;
    }

    // Is a unit the writer still holds a usable prefix of this request (at least a block longer than what
    // memory already matches)? Then the task waits for its publish (update_slots defers it, the other slots
    // keep running): waiting out a write beats re-prefilling the prefix, and the restore then reads only
    // published units, never the queue.
    size_t aw_pending_prefix(const server_tokens & request, int32_t n_past) {
        if (!aw.running) {
            return 0;
        }
        std::lock_guard<std::mutex> lk(aw.mtx);
        if (aw.jobs.empty()) {
            return 0;
        }
        const llama_tokens & req = request.get_cell_tokens();
        const size_t floor = (size_t) std::max(0, n_past) + (size_t) std::max(1, params_base.slot_save_block);
        size_t n = 0;
        for (const auto & j : aw.jobs) {
            const llama_tokens & t = *j->toks;
            if (!j->done && t.size() >= floor && t.size() <= req.size() && std::equal(t.begin(), t.end(), req.begin())) {
                n++;
            }
        }
        return n;
    }

    // ---- stage 2: deferred positional copies (server thread) ------------------------------------------
    //
    // A capture whose memory class has positional cells (llama_state_seq_get_size_deferred: cells of an
    // append-only cache, never side-state) copies only the side-state and metadata at the capture and keeps
    // a reference to the rest (llama_state_seq_save_deferred). The job is queued like any other, in mode
    // `deferred`, and its streams are fed later on the server thread, in FIFO order:
    //   - idle: the queue loop's idle wakeups emit AW_DEF_IDLE_SLICE at a time, pausing when the staging is
    //     full, so a task arriving meanwhile is served at once;
    //   - trickle: a copy still pending defer_trickle_ms after its capture gets AW_DEF_TRICKLE_SLICE per
    //     AW_DEF_TRICKLE_PERIOD_US of busy time, so a long generation cannot hold a unit back indefinitely.
    //     The slice scales with the length of the loop iteration (capped at AW_DEF_TRICKLE_MAX): a prefill
    //     iteration is one batch and takes seconds, and a fixed slice per iteration would hold a periodic
    //     prefill checkpoint (a 1.2 GB delta at 16384 tokens on Qwen3.8-27B) back until the prefill ends,
    //     so a crash during the prefill would find none of them on disk;
    //   - forced: before anything changes a referenced cell (the flush-on-mutate hook in the engine calls
    //     aw_def_flush_cb), when a capture or a task must wait for the writer, and before sleep, shutdown
    //     or a restore into the slot.
    // The engine only reads the device with ggml_backend_tensor_get, on every backend. Units are byte for
    // byte what an immediate capture writes. Side-state held by pending captures is bounded by the staging
    // budget (aw.deferred_host); chunks in flight share the budget with staged saves (aw.staged).

    // Captures `job` deferred and queues it. Returns false (nothing done) when the class, the mode or the
    // budget rules it out: the caller then captures it at once (stage 1).
    bool aw_capture_deferred(server_slot & slot, const std::shared_ptr<aw_job> & job, llama_context * ctx,
                             llama_pos p0, llama_pos dft_p0, llama_pos dft_p1, const std::vector<uint8_t> & dft_trailer,
                             auto_save_site site) {
        if (!aw.running || !aw.defer || aw.shutdown || delta_capable == delta_cap::no) {
            return false;
        }
        const llama_tokens & toks = *job->toks;
        size_t n_def = 0;
        const size_t n_q = llama_state_seq_get_size_deferred(ctx, slot.id, p0, -1, toks.size(), &n_def);
        if (n_q != job->bin.n_expected || n_def == 0) {
            return false; // nothing positional in this class (or a size mismatch: stage 1 decides)
        }
        size_t n_def_dft = 0, n_q_dft = 0;
        if (job->has_dft) {
            n_q_dft = llama_state_seq_get_size_deferred(ctx_dft, slot.id, dft_p0, dft_p1, toks.size(), &n_def_dft);
        }
        const size_t n_host = (n_q - n_def) + (job->has_dft ? n_q_dft - n_def_dft : 0);
        if (n_host > aw.budget) {
            return false; // the side-state alone exceeds the budget: copy at once, streamed
        }
        // side-state held by pending captures stays within the budget: older captures go first if needed
        while (!aw.def.empty()) {
            size_t held = 0;
            {
                std::lock_guard<std::mutex> lk(aw.mtx);
                held = aw.deferred_host;
            }
            if (held + n_host <= aw.budget) {
                break;
            }
            aw_def_force_front("wait");
        }

        const int64_t t0 = ggml_time_us();
        llama_state_deferred * hb = llama_state_seq_save_deferred(ctx, slot.id, p0, -1, toks.data(), toks.size());
        if (hb == nullptr || llama_state_deferred_size(hb) != job->bin.n_expected) {
            llama_state_deferred_free(hb);
            return false;
        }
        job->t_d2h_tgt = ggml_time_us() - t0;
        llama_state_deferred * hd = nullptr;
        if (job->has_dft) {
            const int64_t t1 = ggml_time_us();
            hd = llama_state_seq_save_deferred(ctx_dft, slot.id, dft_p0, dft_p1, toks.data(), toks.size());
            if (hd == nullptr || llama_state_deferred_size(hd) + dft_trailer.size() != job->dft.n_expected) {
                llama_state_deferred_free(hd);
                hd = nullptr;
                job->has_dft = false; // the unit restores with a cold draft
                aw_cnt.draft_skipped++;
                SLT_DBG(slot, "auto-save: no draft sidecar for this unit (the draft capture failed)%s\n", "");
            }
            job->t_d2h_dft = ggml_time_us() - t1;
        }

        job->site       = site;
        job->mode       = aw_mode::deferred;
        job->n_deferred = llama_state_deferred_n_deferred(hb) + (hd ? llama_state_deferred_n_deferred(hd) : 0);
        auto arm = [&](aw_def_part & dp, aw_stream & st, llama_state_deferred * h) {
            dp.h         = h;
            dp.active    = true;
            dp.host      = llama_state_deferred_size(h) - llama_state_deferred_n_deferred(h);
            dp.sc.self   = this;
            dp.sc.job    = job.get();
            dp.sc.st     = &st;
            llama_state_deferred_set_flush_cb(h, aw_def_flush_cb, this);
        };
        arm(job->def_bin, job->bin, hb);
        if (hd) {
            arm(job->def_dft, job->dft, hd);
            job->def_dft.tail = dft_trailer;
        }
        {
            std::lock_guard<std::mutex> lk(aw.mtx);
            aw.deferred_host += job->def_bin.host + job->def_dft.host;
            job->t_enq = ggml_time_us();
            aw.queued_bytes += job->bin.n_expected + job->dft.n_expected;
            if (aw_idle_locked()) {
                aw.last_progress_ms = ggml_time_ms();
            }
            aw.jobs.push_back(job);
            aw.def_count++;
        }
        aw.cv.notify_all();
        aw.def.push_back(job);
        aw_cnt.queued++;
        aw_cnt.deferred++;
        aw_cnt.deferred_bytes += job->n_deferred;
        return true;
    }

    // the handle's host bytes leave the deferred budget once it is emitted
    void aw_def_release_part(aw_def_part & dp) {
        if (dp.h) {
            llama_state_deferred_free(dp.h);
            dp.h = nullptr;
        }
        {
            std::lock_guard<std::mutex> lk(aw.mtx);
            aw.deferred_host -= std::min(aw.deferred_host, dp.host);
        }
        dp.host = 0;
    }

    void aw_def_end_part(aw_def_part & dp, aw_stream & st, bool ok) {
        if (dp.finished) {
            return;
        }
        aw_def_release_part(dp);
        dp.finished = true;
        aw_finish_stream(st, ok);
    }

    // Emits what one part still owes. Returns true when the part is finished (completely, or failed).
    bool aw_def_step_part(aw_def_part & dp, aw_stream & st, bool blocking, size_t max, size_t & n) {
        if (!dp.active || dp.finished) {
            return true;
        }
        aw_sink_ctx & sc = dp.sc;
        sc.blocking = blocking;
        sc.paused   = false;
        if (dp.h) {
            if (max != 0 && n >= max) {
                return false;
            }
            const llama_state_sink sink = { aw_sink_reserve, aw_sink_commit, &sc };
            bool done = false;
            n += llama_state_deferred_emit(dp.h, &sink, max != 0 ? max - n : 0, &done);
            if (sc.failed) {
                aw_def_end_part(dp, st, false);
                return true;
            }
            if (!done) {
                return false;
            }
            aw_def_release_part(dp);
        }
        while (dp.tail_off < dp.tail.size()) {
            size_t avail = 0;
            void * dst = aw_reserve(sc, dp.tail.size() - dp.tail_off, &avail);
            if (dst == nullptr) {
                if (sc.failed) {
                    aw_def_end_part(dp, st, false);
                    return true;
                }
                return false;
            }
            memcpy(dst, dp.tail.data() + dp.tail_off, avail);
            aw_sink_commit(&sc, avail);
            dp.tail_off += avail;
            n           += avail;
        }
        if (sc.cur.n > 0) {
            aw_push_chunk(sc);
        }
        aw_def_end_part(dp, st, !sc.failed && st.n_produced == st.n_expected);
        return true;
    }

    // Advances one deferred job (.bin first: the writer consumes it before the .dft). True when finished.
    bool aw_def_step_job(aw_job & job, bool blocking, size_t max, size_t & n) {
        const int64_t t0 = ggml_time_us();
        bool fin = aw_def_step_part(job.def_bin, job.bin, blocking, max, n);
        if (fin && job.def_bin.finished && !job.def_dft.finished && job.def_dft.active) {
            if (job.bin.aborted) {
                aw_def_end_part(job.def_dft, job.dft, false); // the writer drops the unit, the draft with it
            } else {
                fin = aw_def_step_part(job.def_dft, job.dft, blocking, max, n);
            }
        }
        job.t_emit += ggml_time_us() - t0;
        if (fin) {
            job.t_emitted = ggml_time_us();
            if (job.bin.aborted) {
                SRV_WRN("slot %d: auto-save: the deferred copy of a %zu-token unit failed, it is not published\n",
                        job.slot_id, job.toks->size());
            }
        }
        return fin;
    }

    // Advances the deferred jobs in FIFO order (the writer consumes them in that order): blocking until
    // `upto` is finished (or every job when upto is null), or without blocking for at most `max` bytes.
    // Returns the bytes emitted.
    size_t aw_def_drive(bool blocking, size_t max, const aw_job * upto, const char * how) {
        size_t n = 0;
        if (blocking && !aw.def.empty()) {
            // the writer has been waiting for these bytes, not hung: its silence so far is not a stall
            std::lock_guard<std::mutex> lk(aw.mtx);
            aw.last_progress_ms = ggml_time_ms();
        }
        while (!aw.def.empty()) {
            std::shared_ptr<aw_job> job = aw.def.front();
            if (!aw_def_step_job(*job, blocking, max, n)) {
                break; // no staging room (idle) or the slice is used up
            }
            job->def_how = how;
            aw.def.pop_front();
            {
                std::lock_guard<std::mutex> lk(aw.mtx);
                aw.def_count = aw.def.size();
            }
            if (job.get() == upto || (max != 0 && n >= max)) {
                break;
            }
        }
        return n;
    }

    void aw_def_force_all(const char * how) {
        if (!aw.def.empty()) {
            aw_def_drive(/*blocking=*/true, 0, nullptr, how);
        }
    }

    void aw_def_force_front(const char * how) {
        if (!aw.def.empty()) {
            aw_def_drive(/*blocking=*/true, 0, aw.def.front().get(), how);
        }
    }

    // `job` still owes deferred bytes (server thread)
    bool aw_def_has(const aw_job * job) const {
        for (const auto & j : aw.def) {
            if (j.get() == job) {
                return true;
            }
        }
        return false;
    }

    // every deferred capture of `slot_id` (and, FIFO, those before it) before its cells are replaced
    void aw_def_force_slot(int slot_id, const char * how) {
        const aw_job * last = nullptr;
        for (const auto & j : aw.def) {
            if (j->slot_id == slot_id) {
                last = j.get();
            }
        }
        if (last) {
            aw_def_drive(/*blocking=*/true, 0, last, how);
        }
    }

    // The engine's flush-on-mutate hook: a cache operation is about to change cells this capture still
    // references. Small remainders (within the deferred budget) are copied into host memory the capture owns,
    // which never waits for the writer; larger ones are emitted now, FIFO, waiting for staging room, exactly
    // like a stage-1 streamed save. Inside a store-lock scope nothing may wait for the writer (it may need the
    // exclusive lock to drain), so the engine's host copy is used there regardless.
    static bool aw_def_flush_cb(void * ud, llama_state_deferred * h) {
        return ((server_context_impl *) ud)->aw_def_flush(h);
    }

    bool aw_def_flush(llama_state_deferred * h) {
        aw_job * job = nullptr;
        aw_def_part * part = nullptr;
        for (const auto & j : aw.def) {
            if (j->def_bin.h == h) {
                job = j.get(); part = &j->def_bin;
            } else if (j->def_dft.h == h) {
                job = j.get(); part = &j->def_dft;
            }
        }
        if (job == nullptr) {
            return false;
        }
        job->n_forced++;
        aw_cnt.deferred_forced++;
        const size_t pend = llama_state_deferred_n_pending(h);
        size_t held = 0;
        {
            std::lock_guard<std::mutex> lk(aw.mtx);
            held = aw.deferred_host;
        }
        if (auto_store_lock::depth() > 0 || !aw.running || held + pend <= aw.budget) {
            // the engine copies the rest (exactly `pend` bytes) into the capture's own host memory: account for it
            {
                std::lock_guard<std::mutex> lk(aw.mtx);
                aw.deferred_host += pend;
            }
            part->host += pend;
            if (held + pend > aw.budget) {
                // inside a store-lock scope (or with the writer stopped) the copy cannot wait for the writer, so
                // it may exceed the budget; it is released as soon as the job is emitted
                SRV_WRN("slot %d: auto-save: a forced deferred copy holds %zu B over the %zu B staging budget (%s)\n",
                        job->slot_id, held + pend - aw.budget, aw.budget,
                        auto_store_lock::depth() > 0 ? "inside a store-lock scope" : "writer not running");
            }
            return false;
        }
        aw_def_drive(/*blocking=*/true, 0, job, "forced");
        return true;
    }

    // idle wakeup: one slice, never waiting for the writer
    // without staging room the retry backs off from AW_DEF_RETRY_MS to AW_IDLE_RETRY_MS (at once when the
    // writer is stalled), so a writer that cannot drain is not polled every few milliseconds
    void aw_def_idle_step() {
        const size_t n = aw_def_drive(/*blocking=*/false, AW_DEF_IDLE_SLICE, nullptr, "idle");
        if (n == 0 && !aw.def.empty()) {
            bool stalled = false;
            {
                std::lock_guard<std::mutex> lk(aw.mtx);
                stalled = aw_stalled_locked();
            }
            aw.def_retry_step_ms = stalled || aw.def_retry_step_ms > 0
                ? std::min<int64_t>(stalled ? AW_IDLE_RETRY_MS : aw.def_retry_step_ms * 2, AW_IDLE_RETRY_MS)
                : AW_DEF_RETRY_MS;
            aw.def_retry_ms = ggml_time_ms() + aw.def_retry_step_ms;
        } else {
            aw.def_retry_step_ms = 0;
            aw.def_retry_ms      = -1;
        }
    }

    // busy loop: an overdue copy advances a little per iteration
    void aw_def_trickle() {
        const int64_t now = ggml_time_us();
        const int64_t dt  = aw.trickle_last_us > 0 ? now - aw.trickle_last_us : 0;
        aw.trickle_last_us = now;
        if (aw.def.empty() || aw.defer_trickle_ms < 0) {
            return;
        }
        if (now - aw.def.front()->t_enq < aw.defer_trickle_ms * 1000) {
            return;
        }
        // bytes per unit of busy time, not per iteration (see the stage-2 notes above)
        const double periods = std::max(1.0, (double) dt / (double) AW_DEF_TRICKLE_PERIOD_US);
        const size_t slice   = (size_t) std::min((double) AW_DEF_TRICKLE_MAX, periods * (double) aw.defer_trickle_slice);
        aw_def_drive(/*blocking=*/false, std::max<size_t>(1, slice), nullptr, "trickle");
    }

    // teardown after the writer stopped: the captures are dropped (their jobs were abandoned)
    void aw_def_abandon_all() {
        for (auto & job : aw.def) {
            aw_def_end_part(job->def_bin, job->bin, false);
            aw_def_end_part(job->def_dft, job->dft, false);
        }
        aw.def.clear();
        std::lock_guard<std::mutex> lk(aw.mtx);
        aw.def_count = 0;
    }

    // ---- publish side (writer thread, or the server thread in sync mode) -------------------------------

    void aw_loop() {
        while (true) {
            std::shared_ptr<aw_job> job;
            {
                std::unique_lock<std::mutex> lk(aw.mtx);
                aw.cv.wait(lk, [&]() { return !aw.jobs.empty() || aw.closing; });
                if (aw.jobs.empty()) {
                    return; // closing, and nothing left
                }
                job = aw.jobs.front();
                aw.last_progress_ms = ggml_time_ms();
            }
            aw_publish(*job);
            {
                std::lock_guard<std::mutex> lk(aw.mtx);
                aw.jobs.pop_front();
            }
            aw.cv.notify_all();
        }
    }

    // takes the next chunk of `st`, waiting for the producer; false when the stream is finished
    bool aw_next_chunk(aw_job & job, aw_stream & st, aw_chunk & out, bool & aborted) {
        std::unique_lock<std::mutex> lk(aw.mtx);
        if (job.mode != aw_mode::sync) {
            aw.cv.wait(lk, [&]() { return !st.chunks.empty() || st.sealed || st.aborted || aw_past_deadline(); });
        }
        // past the shutdown deadline a stream the producer never finished counts as aborted
        aborted = st.aborted || (!st.sealed && st.chunks.empty() && job.mode != aw_mode::sync && aw_past_deadline());
        if (st.chunks.empty()) {
            return false;
        }
        out = std::move(st.chunks.front());
        st.chunks.pop_front();
        return true;
    }

    // a written chunk frees its staging: a staged job gives back its reservation, a streamed one the ring
    void aw_release_chunk(aw_job & job, const aw_chunk & c) {
        {
            std::lock_guard<std::mutex> lk(aw.mtx);
            size_t dec = c.cap;
            if (job.mode == aw_mode::staged) {
                dec = std::min(dec, job.reserved);
                job.reserved -= dec;
            }
            aw.staged -= std::min(aw.staged, dec);
            aw.last_progress_ms = ggml_time_ms();
        }
        aw.cv.notify_all();
    }

    // drops whatever the stream still holds (or will receive) without writing it
    void aw_drain_stream(aw_job & job, aw_stream & st) {
        if (&st == &job.dft && !job.has_dft) {
            return; // never produced
        }
        aw_chunk c;
        bool aborted = false;
        while (aw_next_chunk(job, st, c, aborted)) {
            aw_release_chunk(job, c);
            c = {};
        }
    }

    enum class aw_wr { ok, aborted, failed, abandoned };

    // writes `st` into `path`, then fdatasyncs it. Timings accumulate into t_write / t_sync (us).
    // Writeback is bounded as it goes: each chunk is pushed to the kernel and its writeback started, and the
    // writer waits for the PREVIOUS chunk's writeback before taking the next, so at most two chunks of this
    // file are dirty at any time. The final fdatasync then has little left to do (it would otherwise flush up
    // to vm.dirty_ratio of RAM in one go, tens of seconds on a multi-GB unit, with no deadline check), and the
    // writer's progress is visible chunk by chunk. Past the shutdown deadline the file is abandoned, also just
    // before that final sync.
    aw_wr aw_write_stream(aw_job & job, aw_stream & st, const std::string & path, int64_t & t_write, int64_t & t_sync) {
        FILE * f = fopen(path.c_str(), "wb");
        bool ok = f != nullptr;
        aw_chunk c;
        bool aborted = false;
        const int64_t t0 = ggml_time_us();
        int64_t t_wb = 0;              // time spent waiting on writeback, counted as sync
        off_t   off = 0, prev_off = 0; // start of the current and of the previous chunk
        off_t   prev_len = 0;
        auto drop = [&]() {
            if (f) {
                fclose(f);
            }
            std::error_code ec;
            std::filesystem::remove(path, ec);
        };
        while (aw_next_chunk(job, st, c, aborted)) {
            if (ok && fwrite(c.buf.get(), 1, c.n, f) != c.n) {
                ok = false;
            }
#if defined(__linux__)
            if (ok && fflush(f) != 0) {
                ok = false;
            }
            if (ok) {
                const int64_t tw = ggml_time_us();
                const int fd = fileno(f);
                sync_file_range(fd, off, (off_t) c.n, SYNC_FILE_RANGE_WRITE);
                if (prev_len > 0) {
                    sync_file_range(fd, prev_off, prev_len,
                                    SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER);
                }
                t_wb += ggml_time_us() - tw;
            }
#endif
            prev_off = off;
            prev_len = (off_t) c.n;
            off     += (off_t) c.n;
            aw_release_chunk(job, c);
            c = {};
            if (job.mode != aw_mode::sync && aw_past_deadline()) {
                // shutdown deadline: stop between chunks, never publish a partial unit
                drop();
                aw_drain_stream(job, st);
                return aw_wr::abandoned;
            }
        }
        t_write += ggml_time_us() - t0 - t_wb;
        t_sync  += t_wb;
        if (aborted) {
            drop();
            return st.aborted ? aw_wr::aborted : aw_wr::abandoned;
        }
        if (job.mode != aw_mode::sync && aw_past_deadline()) {
            drop();
            return aw_wr::abandoned;
        }
        if (f) {
            const int64_t t1 = ggml_time_us();
            ok = fflush(f) == 0 && ok;
#ifndef _WIN32
            ok = ::fdatasync(fileno(f)) == 0 && ok;
#endif
            ok = fclose(f) == 0 && ok;
            t_sync += ggml_time_us() - t1;
        }
        if (!ok) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
            return aw_wr::failed;
        }
        aw_progress();
        return aw_wr::ok;
    }

    // fsync of a small file written by a helper (.logits, .meta), or of the store directory
    static bool aw_sync_path(const std::string & path, bool dir) {
#ifndef _WIN32
        const int fd = ::open(path.c_str(), (dir ? (O_RDONLY | O_DIRECTORY) : O_RDONLY) | O_CLOEXEC);
        if (fd < 0) {
            return false;
        }
        const bool ok = (dir ? ::fsync(fd) : ::fdatasync(fd)) == 0;
        ::close(fd);
        return ok;
#else
        (void) path; (void) dir;
        return true;
#endif
    }

    void aw_mark_failed(uint64_t unit_id) {
        std::lock_guard<std::mutex> lk(aw.mtx);
        aw.failed_units.push_back(unit_id);
        while (aw.failed_units.size() > 256) {
            aw.failed_units.pop_front();
        }
    }

    bool aw_parent_failed(uint64_t parent_id) {
        std::lock_guard<std::mutex> lk(aw.mtx);
        return std::find(aw.failed_units.begin(), aw.failed_units.end(), parent_id) != aw.failed_units.end();
    }

    enum class aw_drop { failed, orphan, abandoned, aborted };

    // Removes every temp of this job, marks it failed (its queued children are dropped), counts and logs.
    void aw_fail(aw_job & job, const char * reason, aw_drop kind = aw_drop::failed) {
        std::error_code ec;
        std::filesystem::remove(job.tmp, ec);
        std::filesystem::remove(slot_logits_sidecar_path(job.tmp), ec);
        std::filesystem::remove(slot_logits_sidecar_path(job.tmp) + ".tmp", ec);
        std::filesystem::remove(slot_meta_sidecar_path(job.tmp), ec);
        std::filesystem::remove(slot_meta_sidecar_path(job.tmp) + ".tmp", ec);
        std::filesystem::remove(slot_draft_sidecar_path(job.tmp), ec);
        aw_mark_failed(job.hash);
        if (kind != aw_drop::abandoned) {
            // The saves this unit covered are lost with it. Where the slot still holds the conversation (a
            // node or an idle flush) its idle flush is re-armed, and so is an orphan's own slot: the re-save
            // finds a published parent or writes a root. Where it does not (a reclaim, a cache-idle clear, the
            // shutdown) the save counts as requested and never published, so it is not silent. A unit's own
            // failure (IO, space) is not retried, as before.
            std::lock_guard<std::mutex> lk(aw.mtx);
            for (const auto & [sid, ssite] : job.shadowed) {
                if (auto_save_site_keeps_slot(ssite)) {
                    aw.resave_slots.push_back(sid);
                } else {
                    aw_cnt.site_requested[ssite]++;
                    SRV_WRN("slot %d: auto-save: a %s save deduplicated against this %zu-token unit is lost with it\n",
                            sid, auto_save_site_name(ssite), job.toks->size());
                }
            }
            if (kind == aw_drop::orphan) {
                aw.resave_slots.push_back(job.slot_id);
            }
            if (aw.resave_slots.size() > 4096) { // nothing takes them while the idle flush is off
                aw.resave_slots.erase(aw.resave_slots.begin(), aw.resave_slots.end() - 4096);
            }
        }
        switch (kind) {
            case aw_drop::failed:
                auto_save_note_failure(job.slot_id, reason, job.toks->size());
                break;
            case aw_drop::orphan:
                aw_cnt.orphan_dropped++;
                SRV_WRN("slot %d: auto-save: dropped a %zu-token delta, %s\n", job.slot_id, job.toks->size(), reason);
                break;
            case aw_drop::abandoned:
                aw_cnt.shutdown_abandoned++;
                SRV_WRN("slot %d: auto-save: abandoned a %zu-token save, %s\n", job.slot_id, job.toks->size(), reason);
                break;
            case aw_drop::aborted:
                break; // the capture counted and logged it
        }
    }

    void aw_publish(aw_job & job) {
        const int64_t t_start = ggml_time_us();
        int64_t t_write = 0, t_sync = 0, t_pub = 0, t_evict = 0;
        const bool sync = job.mode == aw_mode::sync;
        bool published_ok = false;
        auto finish = [&]() {
            {
                std::lock_guard<std::mutex> lk(aw.mtx);
                aw.staged   -= std::min(aw.staged, job.reserved);
                job.reserved = 0;
                if (!sync) {
                    aw.queued_bytes -= std::min(aw.queued_bytes, job.bin.n_expected + job.dft.n_expected);
                }
                job.done = true;
                job.published = published_ok;
            }
            aw.cv.notify_all();
        };
        auto abandon = [&]() {
            aw_drain_stream(job, job.bin);
            aw_drain_stream(job, job.dft);
            aw_fail(job, "the shutdown deadline passed before it was written", aw_drop::abandoned);
            finish();
        };
        if (!sync && aw_past_deadline()) {
            abandon();
            return;
        }
        // a delta whose parent failed to publish can never be restored: its staged bytes are a suffix
        if (job.is_node && aw_parent_failed(job.parent_id)) {
            aw_drain_stream(job, job.bin);
            aw_drain_stream(job, job.dft);
            aw_fail(job, "its parent failed to publish", aw_drop::orphan);
            finish();
            return;
        }
        // 1) temps: .bin, .logits, .dft, .meta, each synced (in sync mode the capture already wrote the
        //    .bin and .dft temps)
        switch (sync ? aw_wr::ok : aw_write_stream(job, job.bin, job.tmp, t_write, t_sync)) {
            case aw_wr::ok:        break;
            case aw_wr::abandoned: abandon(); return;
            case aw_wr::aborted:
                aw_drain_stream(job, job.dft);
                aw_fail(job, "", aw_drop::aborted);
                finish();
                return;
            case aw_wr::failed:
                aw_drain_stream(job, job.dft);
                aw_fail(job, "state write failed (disk full or IO error)");
                finish();
                return;
        }
        if (!job.logits.empty()) {
            const int64_t t0 = ggml_time_us();
            if (slot_logits_write(job.tmp, job.logits, job.n_vocab, (uint32_t) job.toks->size()) > 0) {
                aw_sync_path(slot_logits_sidecar_path(job.tmp), false);
            }
            t_write += ggml_time_us() - t0;
        }
        bool dft_ok = false;
        if (job.has_dft) {
            switch (sync ? aw_wr::ok : aw_write_stream(job, job.dft, slot_draft_sidecar_path(job.tmp), t_write, t_sync)) {
                case aw_wr::ok:        dft_ok = true; break;
                case aw_wr::abandoned: abandon(); return;
                case aw_wr::aborted:   break; // counted at capture: the unit restores with a cold draft
                case aw_wr::failed:    aw_cnt.draft_skipped++; break;
            }
        }
        {
            const int64_t t0 = ggml_time_us();
            const bool meta_ok = job.is_node
                ? slot_meta_write(job.tmp, job.fp, *job.toks, job.hash, job.media, /*is_node=*/true,
                                  job.parent_id, job.parent_hi, (uint32_t) job.toks->size())
                : slot_meta_write(job.tmp, job.fp, *job.toks, job.hash, job.media);
            if (!meta_ok || !aw_sync_path(slot_meta_sidecar_path(job.tmp), false)) {
                aw_fail(job, ".meta sidecar write failed");
                finish();
                return;
            }
            t_sync += ggml_time_us() - t0;
        }
        if (aw.test_delay_ms > 0 && !sync) {
            // test hook: the unit's temps are complete and nothing is published yet
            const int64_t until = ggml_time_ms() + aw.test_delay_ms;
            while (ggml_time_ms() < until && !aw_past_deadline()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (aw_past_deadline()) {
                abandon();
                return;
            }
        }
        // 2) atomic publish under the store lock: .bin, .logits, .dft (or remove a stale one), .meta last
        aw_progress();
        const int64_t t_p0 = ggml_time_us();
        int timeout_ms = auto_store_lock::PUBLISH_TIMEOUT_MS;
        const int64_t deadline_ms = aw.deadline_ms.load(std::memory_order_relaxed);
        if (deadline_ms >= 0) {
            timeout_ms = (int) std::clamp<int64_t>(deadline_ms - ggml_time_ms(), 0, timeout_ms);
        }
        auto_store_lock pub_lock(params_base.slot_save_path, /*exclusive=*/true, timeout_ms);
        if (!pub_lock.held()) {
            aw_fail(job, "timed out waiting for restores in progress to release the store lock");
            finish();
            return;
        }
        aw_progress();
        // a parent evicted (by a peer, or by our own LRU) while this delta was queued: drop it. Checked under
        // the lock, right before the renames, so the window left is a peer's LRU (deletes take no lock).
        if (job.is_node) {
            std::error_code pec;
            if (!std::filesystem::exists(slot_meta_sidecar_path(job.parent_fname), pec)) {
                aw_fail(job, "its parent is no longer in the store", aw_drop::orphan);
                finish();
                return;
            }
        }
        const std::string & fname = job.fname;
        std::error_code ec;
        std::filesystem::rename(job.tmp, fname, ec);
        if (ec) {
            aw_fail(job, "rename of the state file into place failed");
            finish();
            return;
        }
        std::filesystem::rename(slot_logits_sidecar_path(job.tmp), slot_logits_sidecar_path(fname), ec);
        ec.clear();
        {
            // a unit without a .dft must not inherit one an earlier writer left at this name
            std::error_code dec;
            if (dft_ok) {
                std::filesystem::rename(slot_draft_sidecar_path(job.tmp), slot_draft_sidecar_path(fname), dec);
                if (dec) {
                    dft_ok = false;
                    std::filesystem::remove(slot_draft_sidecar_path(job.tmp), dec);
                    std::filesystem::remove(slot_draft_sidecar_path(fname), dec);
                }
            } else {
                std::filesystem::remove(slot_draft_sidecar_path(fname), dec);
            }
        }
        bool meta_fail = false;
        {
            std::lock_guard<std::mutex> lk(aw.mtx);
            meta_fail = aw.test_fail_meta > 0 && ++aw.n_publishes == aw.test_fail_meta;
        }
        if (!meta_fail) {
            std::filesystem::rename(slot_meta_sidecar_path(job.tmp), slot_meta_sidecar_path(fname), ec);
        }
        if (meta_fail || ec) {
            // the .meta is the scan key: a unit whose .meta never landed is not published
            std::error_code rec;
            std::filesystem::remove(fname, rec);
            std::filesystem::remove(slot_logits_sidecar_path(fname), rec);
            std::filesystem::remove(slot_draft_sidecar_path(fname), rec);
            aw_fail(job, "rename of the .meta sidecar into place failed");
            finish();
            return;
        }
        pub_lock.release();
        {
            // published: a later child of this unit is no orphan, whatever an earlier attempt did
            std::lock_guard<std::mutex> lk(aw.mtx);
            aw.failed_units.erase(std::remove(aw.failed_units.begin(), aw.failed_units.end(), job.hash), aw.failed_units.end());
        }
        // power-loss durability of the renames
        aw_sync_path(params_base.slot_save_path.empty() ? std::string(".") : params_base.slot_save_path, true);
        t_pub = ggml_time_us() - t_p0;

        // 3) index insert, then the LRU and the reconcile
        bool published = true;
        {
            std::lock_guard<std::mutex> lk(auto_idx.mtx);
            auto_cache_entry e{ fname, (uint32_t) job.toks->size(), job.fp, /*pinned=*/false, /*id=*/job.hash,
                                /*deepest=*/ job.kb < job.bhs.size() ? job.bhs[job.kb] : 0, /*has_media=*/ !job.media.empty() };
            for (size_t i = 0; i <= job.kb && i < job.bhs.size(); ++i) {
                auto_index_insert_locked(job.bhs[i], e);
            }
            auto_idx.indexed_files.insert(fname);
        }
        const int64_t t_e0 = ggml_time_us();
        if (!job.supersedes.empty()) {
            // the previous whole checkpoint of this prefill: this deeper one replaces it (a pinned one stays)
            std::error_code sec;
            if (!std::filesystem::exists(job.supersedes + ".pin", sec) && std::filesystem::exists(job.supersedes, sec)) {
                std::filesystem::remove(job.supersedes, sec);
                std::filesystem::remove(slot_meta_sidecar_path(job.supersedes), sec);
                std::filesystem::remove(slot_logits_sidecar_path(job.supersedes), sec);
                std::filesystem::remove(slot_draft_sidecar_path(job.supersedes), sec);
                aw_cnt.ckpt_superseded++;
                SRV_INF("auto disk cache: prefill checkpoint %s replaces the shallower whole checkpoint %s\n",
                        fname.c_str(), job.supersedes.c_str());
            }
        }
        if (params_base.slot_save_max_count > 0 || params_base.slot_save_max_bytes > 0) {
            bool oversized = false;
            uint64_t n_evicted = 0;
            uint64_t n_bound   = 0;
            bool chain_rejected = false;
            slot_save_enforce_limits(params_base.slot_save_path,
                                     params_base.slot_save_max_count,
                                     params_base.slot_save_max_bytes,
                                     fname, oversized, &n_evicted, &n_bound, &chain_rejected);
            aw_cnt.evicted        += n_evicted;
            aw_cnt.bound_exceeded += n_bound;
            if (oversized) {
                published = false;
                aw_mark_failed(job.hash);
                auto_save_note_failure(job.slot_id, "snapshot is larger than --slot-save-max-mb", job.toks->size());
            } else if (chain_rejected) {
                published = false;
                aw_mark_failed(job.hash);
                auto_save_note_failure(job.slot_id, "the store cap cannot hold this conversation's delta chain", job.toks->size());
            }
        }
        aw_progress();
        auto_reap_dead_temps();
        {
            std::lock_guard<std::mutex> lk(auto_idx.mtx);
            auto_index_drop_missing_locked();
            std::error_code mec;
            const auto dmt = std::filesystem::last_write_time(params_base.slot_save_path, mec);
            if (!mec) {
                auto_idx.dir_mtime = dmt;
            }
        }
        t_evict = ggml_time_us() - t_e0;
        published_ok = published;
        if (published) {
            aw_cnt.site_published[job.site]++;
            (job.is_node ? aw_cnt.delta : aw_cnt.root)++;
            aw_cnt.bytes += job.bin.n_expected;
            if (dft_ok) {
                aw_cnt.draft++;
            }
        }
        const char * mode = job.mode == aw_mode::staged ? "staged" : job.mode == aw_mode::streamed ? "streamed" :
                            job.mode == aw_mode::deferred ? "deferred" : "sync";
        char defer_info[160] = "";
        if (job.mode == aw_mode::deferred) {
            snprintf(defer_info, sizeof(defer_info), ", positional %zu B copied %.1f ms after the capture (copy %.1f ms, %s%s)",
                     job.n_deferred, std::max<int64_t>(0, job.t_emitted - job.t_enq) / 1000.0, job.t_emit / 1000.0,
                     job.def_how, job.n_forced > 0 ? ", forced by a cache mutation" : "");
        }
        char what[64];
        if (job.media.empty()) {
            snprintf(what, sizeof(what), "%zu tokens", job.toks->size());
        } else {
            snprintf(what, sizeof(what), "%zu cells incl. %zu media chunks", job.toks->size(), job.media.size());
        }
        SRV_INF("slot %d: auto-save: persisted %s to %s (%s, %zu B: capture d2h %.1f ms [tgt %.1f / dft %.1f / logits %.1f], "
                "queue-wait %.1f ms, write %.1f ms, fdatasync %.1f ms, publish %.1f ms, evict %.1f ms, mode %s%s)\n",
                job.slot_id, what, fname.c_str(), job.is_node ? "delta" : "root", job.bin.n_expected,
                (job.t_d2h_tgt + job.t_d2h_dft + job.t_logits) / 1000.0, job.t_d2h_tgt / 1000.0, job.t_d2h_dft / 1000.0,
                job.t_logits / 1000.0, sync ? 0.0 : (t_start - job.t_enq) / 1000.0,
                t_write / 1000.0, t_sync / 1000.0, t_pub / 1000.0, t_evict / 1000.0, mode, defer_info);
        finish();
    }

    // Temp files of a writer that died (crash, SIGKILL) are never renamed and nothing else removes them.
    // Reap "<name>.<pid>.<nonce>.tmp*" whose pid is not alive and that are older than the reap age; a
    // live peer's temps (its pid is alive) are left alone. Runs at startup and after a publish, at most
    // once a minute.
    void auto_reap_dead_temps() {
#ifndef _WIN32
        if (!auto_cache_enabled()) {
            return;
        }
        const int64_t now_ms = ggml_time_ms();
        {
            std::lock_guard<std::mutex> lk(aw.mtx);
            if (aw.last_reap_ms >= 0 && now_ms - aw.last_reap_ms < 60 * 1000) {
                return;
            }
            aw.last_reap_ms = now_ms;
        }
        const auto now = std::filesystem::file_time_type::clock::now();
        std::error_code ec;
        for (std::filesystem::directory_iterator it(params_base.slot_save_path, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = it->path().filename().string();
            const size_t t = name.find(".tmp");
            if (t == std::string::npos) {
                continue;
            }
            // "<base>.<pid>.<nonce>" before ".tmp"
            const size_t d2 = name.rfind('.', t - 1);
            if (d2 == std::string::npos || d2 == 0) {
                continue;
            }
            const size_t d1 = name.rfind('.', d2 - 1);
            if (d1 == std::string::npos) {
                continue;
            }
            const std::string spid   = name.substr(d1 + 1, d2 - d1 - 1);
            const std::string snonce = name.substr(d2 + 1, t - d2 - 1);
            auto digits = [](const std::string & s) {
                return !s.empty() && s.size() <= 12 && std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; });
            };
            if (!digits(spid) || !digits(snonce)) {
                continue;
            }
            const long pid = std::strtol(spid.c_str(), nullptr, 10);
            if (pid <= 0 || pid == (long) getpid() || !(::kill((pid_t) pid, 0) != 0 && errno == ESRCH)) {
                continue;
            }
            std::error_code fec;
            const auto mt = std::filesystem::last_write_time(it->path(), fec);
            if (fec || now - mt < std::chrono::seconds(aw.reap_age_s)) {
                continue;
            }
            std::filesystem::remove(it->path(), fec);
            if (!fec) {
                SRV_INF("auto disk cache: removed %s, a temp of writer pid %ld which is no longer running\n",
                        name.c_str(), pid);
            }
        }
#endif
    }

    // MID-PREFILL SHARED-CONTEXT BASE (Option A): persist the leading shared preamble [0, B_ctx) ONCE
    // as a deduplicated WHOLE-state v1 ROOT, so N chats sharing that prefix each restore this base via
    // the existing longest-prefix restore instead of re-prefilling it (and, with --slot-save-incremental,
    // collapse their own save to a small [B_ctx, N) delta parented on it: the incremental parent-find
    // discovers it for free). Called from update_slots() at the exact instant a COLD-prefilling slot has
    // decoded EXACTLY cells [0, B_ctx) and NOTHING beyond (armed via slot.ctx_save_pos at prompt start,
    // clamped there by the prefill loop). Because the resident sequence IS the true whole state at
    // B_ctx, the whole-save (auto_publish_snapshot lo==0, hi==toks.size() -> llama_state_seq_save_file
    // whole-root branch) serialises the correct recurrent/attention state: SOUND for dense, SWA AND
    // recurrent/hybrid. This is why there is NO model-class gate here (the old idle-flush [0,B) sub-range
    // checkpoint, taken with the slot sitting at N, mislabelled the state-after-N as a B-length prefix
    // and had to be hard-gated to PART && n_swa == 0; the mid-prefill whole-save removes that unsoundness
    // for every class). Every early return is a clean no-op (invariant 4). Remaining gates:
    //   - MEDIA: a prompt carrying media is saved like a text one, from its cell tokens and media records
    //     (the media-aware chain, keyed at its deepest chunk-safe boundary). Arming only ever picks a
    //     position whose previous cell is text, so the cut never splits a chunk.
    //   - LoRA-equal: the fingerprint captures the global LoRA set; refuse a base taken under a per-request
    //     adapter override (invariant 3), same guard as the whole-prefix save path.
    void auto_save_context_base(server_slot & slot) {
        if (!auto_cache_enabled()) {
            return; // off by default
        }
        if (!are_lora_equal(slot.lora, params_base.lora_adapters)) {
            return; // fp captures the global LoRA set (invariant 3)
        }
        // B_ctx = the armed target. The slot is resident at EXACTLY B_ctx here (clamped there and this
        // batch decoded), so the cell list has length B_ctx and a whole-save serialises the whole state
        // at B_ctx. Re-checked defensively so a spurious call can only no-op, never write a mislabelled
        // prefix.
        const int32_t B_ctx = slot.ctx_save_pos;
        if (B_ctx <= 0) {
            return;
        }
        std::vector<server_media_record> media;
        if (slot.prompt.tokens.has_media()) {
            try {
                media = slot.prompt.tokens.extract_media_records();
            } catch (const std::exception & e) {
                SLT_WRN(slot, "auto-save: system node skipped, %s\n", e.what());
                return; // identity-less chunk: never re-verifiable (invariant 4)
            }
        }
        const llama_tokens toks = media.empty() ? slot.prompt.tokens.get_text_tokens()
                                                : slot.prompt.tokens.get_cell_tokens();
        if ((int32_t) toks.size() != B_ctx) {
            return; // defensive: the hook must fire with the slot resident at exactly B_ctx
        }
        const int B = params_base.slot_save_block;
        if (B <= 0 || B_ctx < B) {
            return; // defensive: the arm floor keeps B_ctx >= one block
        }
        // block-hash chain over [0, B_ctx). Like every other unit, the node is indexed at its boundaries
        // up to the deepest one (bhs.back(): the last whole block for text, or a later chunk end) and
        // named by its identity over every cell, so a node at an exact, unaligned preamble end is
        // distinct from another of the same length that differs only in its last partial block.
        uint64_t id = 0;
        const auto bhs = auto_block_hashes(toks, media, B, cur_fp.fp_model, cur_fp.fp_mmproj, &id);
        if (bhs.empty()) {
            return;
        }
        const size_t   kb        = bhs.size() - 1;
        const uint64_t ckpt_hash = bhs[kb];
        // EXACT-unit redundant-write dedup (not equal-or-longer): the 2nd..Nth chat sharing this
        // [0, B_ctx) preamble finds the base already published and writes nothing.
        {
            std::lock_guard<std::mutex> lk(auto_idx.mtx);
            if (auto_index_has_exact_locked(ckpt_hash, (uint32_t) B_ctx, id)) {
                return;
            }
        }
        {
            std::lock_guard<std::mutex> lk(aw.mtx);
            for (const auto & j : aw.jobs) {
                if (!j->done && j->hash == id && j->toks->size() == (size_t) B_ctx) {
                    j->shadowed.emplace_back(slot.id, AUTO_SAVE_SITE_SYSTEM_NODE); // re-armed if that unit fails
                    return; // queued already
                }
            }
        }
        // Write cells [0, B_ctx) as a WHOLE-state ROOT (v1 text / v2 media): lo == 0 and hi == B_ctx ==
        // toks.size() so auto_publish_snapshot takes the llama_state_seq_save_file (whole-root) branch,
        // which serialises the true whole recurrent/attention state at B_ctx. Same capacity pre-flight,
        // pid+nonce temp, temp->rename (meta last) publish and index insert as every other save.
        auto_publish_snapshot(slot, AUTO_SAVE_SITE_SYSTEM_NODE, ctx_tgt, toks, /*lo=*/0, /*hi=*/B_ctx,
                              /*hash=*/id, bhs, kb, cur_fp, media);
    }

    // the sites after which the slot still holds the conversation (another site can save it again later)
    static bool auto_save_site_keeps_slot(auto_save_site site) {
        return site != AUTO_SAVE_SITE_RECLAIM && site != AUTO_SAVE_SITE_CACHE_IDLE && site != AUTO_SAVE_SITE_SHUTDOWN;
    }

    // Returns true only when an idle flush was deferred (staging full, writer busy): see auto_idle_flush.
    bool auto_save_slot_if_useful(server_slot & slot, auto_save_site site) {
        if (!auto_cache_enabled()) {
            return false; // off by default
        }
        // exclusions reuse the existing guards. NOTE: an idle slot has already been reset(), so
        // `slot.task` is null here: the just-finished task survives as `slot.task_prev`. Use it for
        // the generative check (COMPLETION/INFILL only).
        const auto & wtask = slot.task ? slot.task : slot.task_prev;
        if (!wtask || !wtask->need_sampling()) {
            return false;
        }
        // The fingerprint captures the GLOBAL LoRA set; refuse to persist a snapshot taken under a
        // per-request adapter override that differs from it (invariant 3). (Conservative: a future version
        // could fold the slot's adapters into the snapshot fingerprint instead.)
        if (!are_lora_equal(slot.lora, params_base.lora_adapters)) {
            return false;
        }
        // media branch, gated on the PER-REQUEST has_media() (not the server-wide has_mtmd/mctx)
        // so an --mmproj server still persists its text-only turns as byte-identical v1 units.
        // A media prompt saves on exactly the same terms as a text one, including a
        // prompt+generation snapshot on a FULL-seq-rm model. A later conversation turn re-renders
        // the assistant history to byte-identical tokens (qwencode_prefix_stability, verified) and
        // so extends the WHOLE snapshot, which is the FULL restore condition, the same way a
        // text turn does. (An earlier gate refused such media snapshots on the belief they were
        // unrestorable; that was wrong and silently denied multi-turn image conversations any disk
        // cache, so it is gone: media is aligned with text.)
        const bool prompt_has_media = slot.prompt.tokens.has_media();
        std::vector<server_media_record> media;
        if (prompt_has_media) {
            try {
                media = slot.prompt.tokens.extract_media_records();
            } catch (const std::exception & e) {
                // identity-less chunk (e.g. a placeholder bitmap): it can never be re-verified
                // against a future request, so the snapshot must not be persisted (invariant 4)
                SLT_WRN(slot, "auto-save: skipped, %s\n", e.what());
                return false;
            }
        }
        // Text prompts: get_text_tokens() (not get_tokens()): media-safe accessor that never
        // trips the get_tokens() GGML_ASSERT(!has_mtmd) under mmproj. For a no-media prompt it
        // equals the full token-id prefix, so the persisted token stream and the block-hash key
        // are byte-identical to what a text-only server would write.
        // Media prompts: get_cell_tokens(): the cell-aligned list (media cells
        // LLAMA_TOKEN_NULL) whose length equals the KV cell count llama_state_seq_save_file
        // persists; `media` tiles its NULL cells exactly.
        const llama_tokens toks = prompt_has_media ? slot.prompt.tokens.get_cell_tokens()
                                                   : slot.prompt.tokens.get_text_tokens();
        if (!media.empty()) {
            // chunk-completeness tripwire: process_mtmd_chunk() only ever appends whole chunks
            // to prompt.tokens (and keep_first() refuses mid-chunk cuts), so a prompt cannot
            // end inside a media chunk. The records must tile the cell list they are stored
            // with; slot_meta_read rejects the unit otherwise.
            GGML_ASSERT((size_t) media.back().start_idx + media.back().n_tokens <= toks.size());
        }
        // effective floor = max(block, min-tokens): a snapshot must cover >= 1 block AND clear the
        // configured minimum-size threshold. A trivially small prefix saves little prefill against
        // the state-file write + later restore, so it is skipped. Applies to text and media alike.
        const int save_floor = std::max(params_base.slot_save_block, params_base.slot_save_min_tokens);
        if ((int) toks.size() < save_floor) {
            return false; // below the floor: not worth a multi-GB write
        }
        // media-aware chain: text cells contribute their token ids (bit-identical to the
        // pre-media chain for a text-only prompt), media cells their record identity; only
        // chunk-safe block boundaries are emitted, so a media prompt with no safe boundary
        // yields an empty chain and is skipped below.
        uint64_t unit_id = 0; // identity over every cell: names the file, decides the exact-length dedup
        const auto bhs = auto_block_hashes(toks, media, params_base.slot_save_block,
                                           cur_fp.fp_model, cur_fp.fp_mmproj, &unit_id);
        if (bhs.empty()) {
            return false;
        }
        // NOTE: the shared-context preamble base is NO LONGER written here at idle-flush. It is now
        // whole-saved MID-PREFILL (Option A: auto_save_context_base, fired from update_slots when a cold
        // slot is resident at exactly the block-aligned first-user boundary B_ctx), which is sound for
        // dense, SWA AND recurrent/hybrid. The old idle-flush [0,B_ctx) sub-range checkpoint, unsound
        // for FULL/RS and SWA and therefore a no-op on the production qwen3.6 (hybrid), is gone.
        const uint64_t full_hash = bhs.back(); // the deepest boundary key: the index bucket to check
        {
            std::lock_guard<std::mutex> lk(auto_idx.mtx);
            auto it = auto_idx.by_boundary.find(full_hash);
            if (it != auto_idx.by_boundary.end()) {
                for (const auto_cache_entry & c : it->second) {
                    // Whole-prefix classes (FULL, RS, SWA): only an EXACT-length snapshot substitutes,
                    // a longer one can never be rewound to this prefix and a shorter one is a different
                    // resume point, otherwise this save would be suppressed by a snapshot the model can
                    // never restore. PART with n_swa_mem == 0: an equal-or-longer snapshot already
                    // covers this prefix because it can rewind to it.
                    // COST, on the record: moving RS from the >= arm to the == arm makes it WRITE the
                    // saves a longer sibling used to suppress. A snapshot is indexed at every boundary
                    // it reaches, so the sibling is any already-saved prompt that agrees with this one
                    // up to its deepest whole block and runs past it - in practice a branch, a
                    // regenerate, or a second conversation whose first turns are shorter than an
                    // existing release snapshot sharing the same preamble. Each such save is a delta
                    // when --slot-save-incremental finds a strict-prefix parent (normally the
                    // mid-prefill [0, B_ctx) base) and a whole multi-GB write when it does not.
                    // slot_save_enforce_limits enumerates the whole --slot-save-path directory and
                    // evicts by mtime with no fingerprint scoping, so in a SHARED store these writes
                    // do compete with other models' units. The trade is taken deliberately: the
                    // suppressed save is the ONLY snapshot an RS model could ever restore at this
                    // prefix, so suppressing it buys the capacity by guaranteeing zero reuse, which is
                    // the production miss this branch exists to fix. Give a shared store enough
                    // --slot-save-max-mb headroom for both identities, or point each model at its own
                    // directory.
                    // "Exact" means the same unit: equal length AND equal identity over every cell, since
                    // the bucket key does not cover cells after the last whole block. (PART still lets
                    // a longer sibling suppress: it can rewind to the shared prefix, losing at most the
                    // trailing partial block when the two differ there.)
                    const bool same = c.n_tokens == toks.size() && c.id == unit_id;
                    if (restore_is_whole_prefix_only() ? same : (same || c.n_tokens > toks.size())) {
                        return false; // a usable snapshot for this exact prefix already exists
                    }
                }
            }
        }
        // the same rule against the units the writer still holds (the PENDING set): a reclaim save right
        // after the node that covers it must not queue the unit twice
        {
            std::unique_lock<std::mutex> lk(aw.mtx);
            std::shared_ptr<aw_job> cover;
            for (const auto & j : aw.jobs) {
                if (j->done) {
                    continue;
                }
                const bool reaches = std::find(j->bhs.begin(), j->bhs.begin() + std::min(j->kb + 1, j->bhs.size()),
                                               full_hash) != j->bhs.begin() + std::min(j->kb + 1, j->bhs.size());
                if (!reaches) {
                    continue;
                }
                const bool same = j->toks->size() == toks.size() && j->hash == unit_id;
                if (restore_is_whole_prefix_only() ? same : (same || j->toks->size() > toks.size())) {
                    cover = j;
                    break;
                }
            }
            if (cover && cover->is_node && !auto_save_site_keeps_slot(site)) {
                // The slot's conversation is about to be overwritten or cleared and the queued unit covering it
                // is a DELTA: the one failure a second write can recover from is that delta being orphaned (its
                // parent failed or was evicted), and only this slot's state can then write it as a root. So
                // wait for the delta's outcome and write the conversation here if it did not publish. A root
                // cover is not waited for: what fails a root (IO, space, size) fails a second write the same
                // way, so a lost root shows up as requested != published for this site instead.
                const int64_t t0 = ggml_time_us();
                // The writer is FIFO: a deferred job at or ahead of the cover owes positional bytes that only this
                // thread emits, so emit them first, or the wait below ends only at the stall timeout (and the
                // rewrite-as-root recovery is skipped). Never inside a store-lock scope, where nothing may wait
                // for the writer.
                if (auto_store_lock::depth() == 0) {
                    const aw_job * upto = nullptr;
                    for (const auto & j : aw.jobs) {
                        if (aw_def_has(j.get())) {
                            upto = j.get();
                        }
                        if (j == cover) {
                            break;
                        }
                    }
                    if (upto) {
                        lk.unlock();
                        aw_def_drive(/*blocking=*/true, 0, upto, "wait");
                        lk.lock();
                    }
                }
                while (!aw.cv.wait_for(lk, std::chrono::milliseconds(500), [&]() { return cover->done || aw_past_deadline(); })) {
                    if (aw_stalled_locked()) {
                        break;
                    }
                }
                const bool pub = cover->done && cover->published;
                SLT_INF(slot, "auto-save: waited %.1f ms for the queued %zu-token unit that covers this slot (%s)\n",
                        (ggml_time_us() - t0) / 1000.0, cover->toks->size(),
                        pub ? "published" : cover->done ? "not published, saving it again" : "still queued");
                if (pub || !cover->done) {
                    return false; // published; or the writer is stalled / past the shutdown deadline
                }
                cover.reset();
            }
            if (cover) {
                cover->shadowed.emplace_back(slot.id, site); // accounted for (and re-armed) if that unit fails
                return false; // queued already
            }
        }

        // INCREMENTAL SAVE: when --slot-save-incremental, write only the KV cells added since the
        // deepest already-saved snapshot on this branch (a v3 text / v4 media delta node) instead of
        // re-D2H'ing and re-writing the whole prefix. Find the deepest candidate whose persisted cells
        // are a STRICT prefix of this prompt under the same fingerprint; the delta .bin then holds
        // cells [parent_hi, N). The parent-find scans the FULL per-boundary candidate vectors (SIZE_MAX,
        // not the RESTORE 4-cap): the real parent is the SHORTEST snapshot at the deepest SHARED
        // boundary (e.g. a pinned base whose bucket also holds N longer divergent siblings), so a 4-cap
        // hides it and forces a whole (~1.77GB) save instead of a small delta. auto_index_lookup orders
        // candidates deepest-boundary-first / longest-first, so the FIRST strict-prefix match is the
        // deepest parent. Flag off (or no parent found) => the EXACT whole-save path below (byte-
        // identical). Everything after this (nonce temp, logits sidecar, temp+rename publish, index
        // insert, LRU) is SHARED between both modes.
        //
        // TEXT AND MEDIA UNIFY here (decision 1/3): a media prompt now takes exactly the same parent-
        // find as a text one: the cell-token byte compare, the per-record parent verify and the
        // media-aware parent hash all reduce to the pre-media text behaviour when `media` is empty. A
        // text prompt therefore emits a byte-identical v3 delta; a media prompt emits a v4 delta. The
        // parent may itself be text (v1) OR media (v2/v4): a text base can parent a media delta (an
        // image added in the new turn) and vice-versa: the media-aware hash keys the exact parent
        // file for either kind (decision 6).
        bool     have_parent = false;
        uint64_t parent_id   = 0;
        uint32_t parent_hi   = 0;
        // per-record parent verify over the shared prefix [0, parent_hi): every candidate record starting
        // inside the parent must equal THIS prompt's record at that exact start index (id, shape, type).
        // Media cells are NULL==NULL so the byte compare is identity-blind; this restores the byte backstop a
        // text delta gets for free. Without it a ~2^-64 filename-hash collision between same-text/different-
        // image prefixes could compose the WRONG parent KV. A no-op for a text delta (decision 5).
        const auto records_match = [&](const std::vector<server_media_record> & cand_media, size_t parent_hi_sz) {
            for (const auto & rec : cand_media) {
                if ((size_t) rec.start_idx >= parent_hi_sz) {
                    break; // this and all later records start outside the shared prefix
                }
                const auto it = std::lower_bound(media.begin(), media.end(), rec.start_idx,
                    [](const server_media_record & r, uint32_t s) { return r.start_idx < s; });
                const bool match = it != media.end()             &&
                                   it->start_idx == rec.start_idx &&
                                   it->id        == rec.id        &&
                                   it->n_tokens  == rec.n_tokens  &&
                                   it->n_pos     == rec.n_pos     &&
                                   it->nx        == rec.nx        &&
                                   it->ny        == rec.ny        &&
                                   it->is_audio  == rec.is_audio;
                if (!match) {
                    return false;
                }
            }
            return true;
        };
        if (params_base.slot_save_incremental) {
            // `toks`/`media` are this prompt's cell tokens + records; look up against the live prompt.
            for (const auto_cache_entry & cand : auto_index_lookup(slot.prompt.tokens, /*max_attempts=*/SIZE_MAX)) {
                model_fp     disk_fp;
                llama_tokens disk_toks;
                std::vector<server_media_record> disk_media;
                if (!slot_meta_read(cand.state_path, cur_fp.fp_mmproj, disk_fp, disk_toks, disk_media)) {
                    continue; // unreadable meta -> not a usable parent (invariant 4)
                }
                // The restore gate: a parent saved by another rung of this model is a valid
                // parent, because n_ctx is not identity and both rungs name units alike. Whether
                // the link resolves by name is checked below, after the strict-prefix test.
                if (!disk_fp.restore_compatible(cur_fp, !disk_media.empty())) {
                    continue; // invariant 3
                }
                // STRICT prefix of the CELL tokens (media cells LLAMA_TOKEN_NULL on both sides):
                // disk_toks == toks[0:disk_toks.size()] AND disk_toks.size() < toks.size() (a delta
                // must add at least one cell; an equal/longer snapshot is not a parent here). A media
                // parent's NULL cells never equal a text prompt's real tokens, so a text prompt can
                // never pick a media parent: the text path is unchanged.
                if (disk_toks.size() >= toks.size() ||
                    !std::equal(disk_toks.begin(), disk_toks.end(), toks.begin())) {
                    continue;
                }
                const size_t parent_hi_sz = disk_toks.size();
                // the delta .bin covers cells [parent_hi, N); the boundary may not split a media chunk
                // (a chunk's cells + M-RoPE positions must all live in one .bin), so refuse a candidate
                // whose end cuts one. Always true on the text path (no media chunks in `media`).
                if (!boundary_is_chunk_safe(toks, media, parent_hi_sz)) {
                    continue;
                }
                if (!records_match(disk_media, parent_hi_sz)) {
                    continue;
                }
                // parent_id = the parent's own file id, read from its filename: the restore walk resolves
                // the link as auto_state_filename(parent_id, parent_hi), so the name is the authority.
                // (Re-deriving it from the prefix would break the link to a unit named under an
                // earlier naming rule.) A candidate whose name does not parse cannot be a parent.
                uint64_t name_id  = 0;
                uint32_t name_len = 0;
                if (!slot_save_parse_node_id(cand.state_path, name_id, name_len) || name_len != parent_hi_sz) {
                    continue;
                }
                // The delta is published under THIS instance's identity prefix, and the restore walk
                // resolves its parent under the tip's prefix, so the parent must carry the same one. A
                // unit named under an earlier naming rule (before n_ctx left the identity, or before
                // the full-identity prefix) is a fine restore target but would leave the delta
                // pointing at a name that does not exist: take a whole root instead.
                if (cand.state_path != auto_state_filename(name_id, name_len)) {
                    continue;
                }
                parent_hi   = (uint32_t) parent_hi_sz;
                parent_id   = name_id;
                have_parent = true;
                break; // deepest (longest-first) strict-prefix parent
            }
            // a deeper parent may still be queued (a prompt node, then the reclaim seconds later in a tool
            // loop): the writer is FIFO, so it is published before this child, and if it fails the child is
            // dropped with it (llamacpp:auto_cache_save_orphan_dropped_total)
            std::lock_guard<std::mutex> lk(aw.mtx);
            for (const auto & j : aw.jobs) {
                const llama_tokens & jt = *j->toks;
                if (j->done || jt.size() <= parent_hi || jt.size() >= toks.size() ||
                    !std::equal(jt.begin(), jt.end(), toks.begin()) ||
                    !boundary_is_chunk_safe(toks, media, jt.size()) || !records_match(j->media, jt.size())) {
                    continue;
                }
                parent_hi   = (uint32_t) jt.size();
                parent_id   = j->hash;
                have_parent = true;
            }
        }

        // Publish through the SHARED atomic-publish helper: capacity pre-flight, pid+nonce temp,
        // three-file temp->rename (meta last) with orphan cleanup, per-boundary index insert + LRU.
        // Whole root: lo=0, hi=N => byte-identical save_file path. Delta (have_parent): lo=parent_hi>0
        // => v3 node cells [parent_hi, N). Both index every boundary (kb = bhs.size()-1).
        return auto_publish_snapshot(slot, site, ctx_tgt, toks,
                              /*lo=*/ have_parent ? (int32_t) parent_hi : 0,
                              /*hi=*/ (int32_t) toks.size(),
                              /*hash=*/ unit_id, bhs, /*kb=*/ bhs.size() - 1, cur_fp,
                              /*media=*/ media, /*parent_id=*/ have_parent ? parent_id : 0);
    }

    // AFTER-RESPONSE node (--slot-save-node-response, --slot-save-node-tool; both default off). Called
    // with the task still attached, just before release(); the save itself runs after release(), on the
    // idle slot, exactly like the idle flush. Off by default because a user who continues on the same
    // instance gains nothing from it: the idle flush and the reclaim save persist the conversation when
    // that is useful, and they cover tool loops too. The tool variant parses the response once, with the
    // request's chat session (a request that offered no tools parses to no tool calls).
    bool auto_save_after_response_wanted(const server_slot & slot) const {
        if (!auto_cache_enabled() || !slot.task || !slot.task->need_sampling()) {
            return false;
        }
        if (params_base.slot_save_node_response) {
            return true;
        }
        // the chat session owns parsing; a request without tools parses to no tool calls
        if (!params_base.slot_save_node_tool || !slot.task->chat_session_node_tool.has_template() ||
            slot.task->chat_session_node_tool.format() == COMMON_CHAT_FORMAT_CONTENT_ONLY) {
            return false;
        }
        try {
            common_chat_session session = slot.task->chat_session_node_tool;
            const common_chat_msg msg = session.finish(slot.generated);
            return !msg.tool_calls.empty();
        } catch (const std::exception & e) {
            SLT_DBG(slot, "auto-save: could not parse the response for tool calls, %s\n", e.what());
            return false;
        }
    }

    // AUTO-SAVE (shutdown): persist every slot's warm KV on graceful terminate: the third
    // call site of auto_save_slot_if_useful, symmetric with the idle-flush and
    // get_available_slot sites. Called exactly once, from server_context::start_loop()
    // after the queue loop has exited: same (main) thread that owns all slot mutation,
    // update_slots quiescent, strictly before llama_backend_free() in the caller's
    // clean_up(). Invariant 1: gate first.
    //
    // Processing slots ARE saved (deliberately, unlike invariant 5's release-site caveat): the
    // queue loop has fully exited, so we sit on an update_slots() boundary where prompt.tokens ==
    // KV length for both prefill and generation (handle_last_sampled_token pushes the sampled
    // token and llama_decode writes its KV within one update_slots; the next, unprocessed token
    // is staged in slot.sampled, outside both prompt.tokens and the KV). A mid-prefill snapshot of
    // a long prompt is thus coherent AND the highest-value thing to keep: a re-run extends it and
    // skips the reprefill (both attention and recurrent). auto_save_slot_if_useful's own dedup +
    // LRU bound the only residual (a recurrent mid-generation snapshot a shorter re-request can't
    // rewind into: a safe, evictable write).
    //
    // Deadline-boxed: the process is typically inside a supervisor's stop window (systemd SIGKILLs at
    // TimeoutStopSec, 120 s on the rig, the router waits 150 s). The captures here block (the shutdown
    // admission waits for staging room instead of dropping), then the queue is closed and the writer
    // drains what is left, jobs queued before the shutdown first, so parents precede the shutdown
    // children. Whatever is still queued at AW_SHUTDOWN_DEADLINE_MS is abandoned between chunks, its temps
    // unlinked and never renamed (llamacpp:auto_cache_save_shutdown_abandoned_total), which leaves time
    // for llama_backend_free inside the stop window.
    void auto_save_slots_at_shutdown() {
        if (!auto_cache_enabled()) {
            return; // off by default
        }
        const int64_t t_deadline_ms = ggml_time_ms() + aw.shutdown_ms;
        {
            std::lock_guard<std::mutex> lk(aw.mtx);
            aw.shutdown    = true;
            aw.deadline_ms = t_deadline_ms;
        }
        if (!sleeping) { // sleep entry destroy()'d ctx_tgt; the warm KV is already gone
            for (size_t i = 0; i < slots.size(); i++) {
                if (ggml_time_ms() >= t_deadline_ms) {
                    SRV_WRN("auto-save: shutdown flush deadline (%" PRId64 " ms) exceeded, skipping %zu remaining slots\n",
                            aw.shutdown_ms, slots.size() - i);
                    break;
                }
                auto_save_slot_if_useful(slots[i], AUTO_SAVE_SITE_SHUTDOWN);
            }
        }
        aw_def_force_all("shutdown"); // pending deferred copies, before the writer is closed
        aw_stop(t_deadline_ms);
    }

    // AUTO-SAVE (idle-delay): the two other save sites (get_available_slot reclaim, shutdown) only
    // fire on next-task-arrival or terminate, so a lone request's warm KV stays crash-volatile and
    // invisible to peer instances until more traffic lands. This closes that window to <=N seconds:
    // once a slot has been idle for slot_save_idle_seconds it is flushed like any other site (v2 for
    // media), on the main-loop thread, off the request hot path. Driven by the queue's timed idle
    // wait (auto_idle_next_deadline bounds the wait; auto_idle_flush runs the due flushes), so there
    // is no busy poll and it works with sleeping disabled.
    bool auto_idle_flush_enabled() const {
        return auto_cache_enabled() && params_base.slot_save_idle_seconds >= 0;
    }

    // Absolute ggml_time_ms() deadline of the earliest slot due for an idle flush, or -1 if none is
    // pending. Called by the queue loop to bound its idle wait. A slot with no prior task (never ran)
    // or already flushed for its current idle period contributes nothing.
    int64_t auto_idle_next_deadline() {
        if (!aw.def.empty() && aw.defer_idle) {
            // a deferred copy is pending: wake at once (or shortly, when the last wakeup found no room)
            return std::max(ggml_time_ms(), aw.def_retry_ms);
        }
        if (!auto_idle_flush_enabled()) {
            return -1;
        }
        const int64_t idle_ms = (int64_t) params_base.slot_save_idle_seconds * 1000;
        auto_idle_take_resaves();
        int64_t earliest = -1;
        for (const auto & slot : slots) {
            if (slot.is_processing() || slot.auto_idle_flushed || !slot.task_prev || slot.t_last_used < 0) {
                continue;
            }
            // t_last_used is microseconds; a deferred flush is due again at its retry time
            const int64_t deadline = std::max(slot.t_last_used / 1000 + idle_ms, slot.auto_idle_retry_ms);
            if (earliest < 0 || deadline < earliest) {
                earliest = deadline;
            }
        }
        return earliest;
    }

    // Flush ONE idle slot whose idle deadline has passed, then return. Runs on the main-loop thread
    // from the queue's idle wait (no task in flight, so all slots are quiescent). Flushing a single
    // slot per wakeup keeps a request that lands mid-flush from stalling behind more than one
    // (potentially multi-GB) write: the queue loop re-checks for queued tasks between wakeups, and
    // any remaining due slots are flushed on immediately-following wakeups (their deadline has already
    // passed, so the wait returns at once). auto_save_slot_if_useful carries every correctness gate
    // and its dedup makes a repeat flush free.
    void auto_idle_flush() {
        if (!aw.def.empty() && aw.defer_idle) {
            // deferred copies first, one slice per wakeup: they are older than any idle flush
            aw_def_idle_step();
            return;
        }
        if (!auto_idle_flush_enabled()) {
            return;
        }
        const int64_t now_ms  = ggml_time_ms();
        const int64_t idle_ms = (int64_t) params_base.slot_save_idle_seconds * 1000;
        auto_idle_take_resaves();
        for (auto & slot : slots) {
            if (slot.is_processing() || slot.auto_idle_flushed || !slot.task_prev || slot.t_last_used < 0) {
                continue;
            }
            if (now_ms - slot.t_last_used / 1000 < idle_ms || now_ms < slot.auto_idle_retry_ms) {
                continue; // not idle long enough yet, or a deferred flush not due again yet
            }
            if (slot.auto_idle_deferred) {
                bool idle = true;
                {
                    std::lock_guard<std::mutex> lk(aw.mtx);
                    idle = aw_idle_locked();
                }
                if (!idle) {
                    slot.auto_idle_retry_ms = now_ms + AW_IDLE_RETRY_MS; // cheap: no save planned until it is
                    continue;
                }
            }
            // A flush that finds the staging full and the writer busy is DEFERRED, not dropped: nothing was
            // queued, and it is retried once the writer is idle (then it streams). No request is in flight,
            // but one may arrive at any moment, so the idle flush never waits for the writer itself.
            if (auto_save_slot_if_useful(slot, AUTO_SAVE_SITE_IDLE)) {
                if (!slot.auto_idle_deferred) {
                    SLT_INF(slot, "%s", "auto-save: idle flush deferred, the staging is full and the writer busy; "
                                        "retrying once the writer is idle\n");
                }
                slot.auto_idle_deferred = true;
                slot.auto_idle_retry_ms = now_ms + AW_IDLE_RETRY_MS;
                continue;
            }
            slot.auto_idle_flushed  = true; // one attempt per idle period; a reclaim/next task re-arms it
            slot.auto_idle_deferred = false;
            slot.auto_idle_retry_ms = -1;
            return;                         // at most one flush per wakeup, then back to service tasks
        }
    }

    // The writer lost a unit that covered these slots (a failed queued unit their save was deduplicated
    // against, or an orphaned delta): give each idle one another idle flush. Server thread only.
    void auto_idle_take_resaves() {
        std::vector<int> ids;
        {
            std::lock_guard<std::mutex> lk(aw.mtx);
            ids.swap(aw.resave_slots);
        }
        for (const int id : ids) {
            for (auto & slot : slots) {
                if (slot.id == id && !slot.is_processing() && slot.auto_idle_flushed) {
                    SLT_INF(slot, "%s", "auto-save: a queued unit covering this slot was lost, re-arming the idle flush\n");
                    slot.auto_idle_flushed = false;
                }
            }
        }
    }

    // ----- the cache's parts of upstream methods, each called from one place in the method named -----

    // A new task whose prefix the background writer still holds (aw_pending_prefix) is held in STARTED until
    // that unit is published, at most AW_PENDING_WAIT_MS, then proceeds to the disk lookup as usual. The wait
    // never blocks the server thread: other slots keep prefilling and decoding meanwhile. Mirrors the
    // restore's own gates, so a task that would not restore does not wait.
    bool auto_pending_defer(server_slot & slot, const server_tokens & input_tokens) {
        if (!aw.running || input_tokens.empty() || !slot.task->params.cache_prompt || !slot.task->need_sampling() ||
                !are_lora_equal(slot.lora, params_base.lora_adapters)) {
            return false;
        }
        const int32_t n_mem = (int32_t) slot.prompt.tokens.get_common_prefix(input_tokens);
        const size_t  n     = aw_pending_prefix(input_tokens, n_mem);
        const int64_t now   = ggml_time_us();
        if (n > 0 && slot.aw_pending_t0_us < 0) {
            slot.aw_pending_t0_us = now;
            SLT_INF(slot, "auto-restore: %zu queued unit(s) prefix this request, waiting for the writer to publish them\n", n);
        }
        if (slot.aw_pending_t0_us < 0) {
            return false;
        }
        const bool timed_out = now - slot.aw_pending_t0_us >= AW_PENDING_WAIT_MS * 1000;
        if (n > 0 && !timed_out) {
            return true;
        }
        SLT_INF(slot, "auto-restore: waited %.1f ms for the queued unit(s) that prefix this request%s\n",
                (now - slot.aw_pending_t0_us) / 1000.0, timed_out && n > 0 ? " (timed out)" : "");
        slot.aw_pending_t0_us = -1;
        return false;
    }

    // load_model(): the projector fingerprint, false when it cannot be computed (the load fails)
    bool mmproj_fingerprint_load(const std::string & mmproj_path) {
        // projector identity fingerprint (header-only read, ~ms): guards media KV
        // snapshots against projector swap/requant/dimension changes and is exposed
        // in /props. Computed whenever an mmproj is loaded, NOT gated on the auto
        // disk cache, precisely because /props reports it.
        if (!mmproj_header_fingerprint(mmproj_path, fp_mmproj)) {
            SRV_ERR("failed to fingerprint multimodal projector, '%s'\n", mmproj_path.c_str());
            return false;
        }
        SRV_INF("multimodal projector fingerprint: %016" PRIx64 "\n", fp_mmproj);
        return true;
    }

    // load_model(): what the memory type can do for a save or restore, next to n_swa
    void memory_caps_init() {
        // NOTE: --swa-full only enlarges the SWA cache; it does NOT disable masking.
        // llama_kv_cache_iswa still constructs kv_swa with hparams.n_swa/swa_type, and
        // llama_kv_cache::state_write masks on those members unconditionally, so saves stay
        // WINDOWED even under --swa-full. Zeroing n_swa here would disarm the disk-cache SWA
        // guards while the engine keeps writing windowed blobs, letting a restore claim a prefix
        // over a hole (silent wrong attention). Keep the scheduling value for the batch planner,
        // but keep a model-derived value for every save/restore soundness decision.
        n_swa_mem  = llama_model_n_swa(model_tgt);   // what the ENGINE masks with, never 0 for an SWA model
        side_only_supported = params_base.slot_restore_selective && llama_memory_can_skip_positional(llama_get_memory(ctx_tgt));
        side_skip_supported = params_base.slot_restore_selective && llama_memory_can_skip_side(llama_get_memory(ctx_tgt));
    }

    // load_model(): the fingerprint, the startup scan of the store and the background writer
    void auto_cache_on_load() {
        // AUTO disk prompt/KV cache (invariant 1): compute the model fingerprint and build the
        // longest-prefix index ONCE, header-only, but ONLY when the feature is enabled. When OFF
        // this is a single boolean test and nothing else (no fingerprint, no scan, no allocation).
        // The fingerprint alone is also needed whenever --slot-save-path is set: manual media
        // snapshots persist it in their v2 .meta sidecar and manual restores verify against it.
        if (auto_cache_enabled() || !params_base.slot_save_path.empty()) {
            cur_fp = auto_compute_fingerprint();
        }
        if (auto_cache_enabled()) {
            auto_index_scan();
            SRV_INF("auto disk prompt cache enabled: indexed %zu prefix boundaries from %s (block=%d)\n",
                    auto_idx.by_boundary.size(), params_base.slot_save_path.c_str(), params_base.slot_save_block);
            aw_start();
        }
    }

    // load_model(): the system node's preamble cache in chat_params
    void preamble_cache_init() {
        // the system node's template-generic position (server_preamble_cache), only when the node is on
        if (auto_cache_enabled() && params_base.slot_save_node_system) {
            chat_params.preamble_cache = std::make_shared<server_preamble_cache>();
            chat_params.preamble_cache->min_tokens = std::max(params_base.slot_save_block, params_base.slot_save_context_min_tokens);
        }
    }

    // get_available_slot(): save the conversation of the slot `ret` that `task` is about to take over
    void reclaim_save(server_slot * ret, const server_task & task) {
        // RECLAIM save (--slot-save-on-reclaim, default on): get_available_slot just picked `ret` for a
        // task whose prompt shares too little with the slot's (or picked it by LRU), so `update_cache`
        // says the slot's conversation is about to be overwritten. Persist it to disk first. This must
        // NOT depend on cache_idle_slots: that path saves only the slots that stay idle AFTER the new
        // task launched, so with --parallel 1 it never reaches the slot being taken over, and a
        // conversation preempted within --slot-save-idle-seconds of its last request used to vanish
        // from disk (it lived on only in --cache-ram). The two sites never save the same slot for the
        // same task (this one saves `ret`, that one the others), and the callee's exact-length dedup
        // absorbs any repeat. Reads `update_cache` BEFORE the `&& prompt_cache` narrowing so disk save
        // works without --cache-ram. `ret` is idle, so this delays only the new task's first token by
        // one save (a delta under --slot-save-incremental).
        // Gate on "the new task does not extend the slot", NOT on `update_cache`: that flag is the
        // RAM cache's own rule (f_keep < 0.5, or an LRU pick), and two conversations that share a
        // long leading prefix (a common system prompt) keep f_keep >= 0.5 while the slot's own tail
        // is still overwritten. Save whenever at least one block of the slot's tokens lies past the
        // common prefix with the new task; a task that extends the slot (or diverges within the
        // last block) loses nothing worth a write.
        if (auto_cache_enabled() && params_base.slot_save_on_reclaim && !ret->prompt.tokens.empty()) {
            const size_t n_lcp_reclaim = ret->prompt.tokens.get_common_prefix(task.tokens);
            if (n_lcp_reclaim + (size_t) params_base.slot_save_block <= (size_t) ret->prompt.n_tokens()) {
                auto_save_slot_if_useful(*ret, AUTO_SAVE_SITE_RECLAIM);
            }
        }
    }

    // launch_slot_with_task(), first: drop the logits captured for the previous prompt
    void logits_capture_reset(server_slot & slot) {
        // A new task is being assigned to this slot: its prompt may differ from whatever produced
        // the slot's current `logits_last` (even at the same token count). Invalidate the capture so
        // a SLOT_SAVE issued on the new prompt can never serialize a stale, mismatched distribution.
        // The stamp is re-established only by a real decode of the new prompt (the capture point).
        slot.logits_last.clear();
        slot.logits_last_n_tokens = -1;
    }

    // launch_slot_with_task(): the system node's position, kept on the slot's prompt
    void ctx_boundary_stash(server_slot & slot, const server_task & task) {
        // stash the system-preamble boundary for the [0,B) shared-context node onto the persistent
        // prompt, computed once here at task launch: the template-generic preamble end when the chat
        // path measured one, else the first user message from the message delimiters (a /completion
        // client's own message_delimiters). The save site also runs at idle-flush and shutdown, when
        // the transient task may be gone, so it must never read the task for this: it reads
        // slot.prompt.ctx_boundary instead. -1 (no boundary) makes the node a no-op.
        slot.prompt.ctx_boundary = task.params.preamble_end >= 0
            ? task.params.preamble_end
            : task.params.message_spans.first_user_message_pos();
    }

    // process_single_task(), --cache-idle-slots: save an idle slot before the RAM cache takes it
    void idle_slot_save(server_slot & slot) {
        // Auto-save (write path): persist this idle slot's KV to disk before
        // the in-memory cache drops it. Gated so the whole call frame is elided
        // when the feature is OFF; the callee re-checks all correctness guards
        // (fingerprint, need_sampling, LoRA, media identity, >=1 block).
        if (auto_cache_enabled()) {
            auto_save_slot_if_useful(slot, AUTO_SAVE_SITE_CACHE_IDLE);
        }
    }

    // process_single_task(), METRICS: the cache's counters
    void metrics_result_fill(server_task_result_metrics * res) {
        auto_metrics_fill(res->metrics);
        if (const auto & pc = chat_params.preamble_cache) {
            res->metrics.n_sysnode_probed        = pc->n_probed.load();
            res->metrics.n_sysnode_probe_renders = pc->n_probe_renders.load();
            res->metrics.n_sysnode_probe_failed  = pc->n_probe_failed.load();
            res->metrics.n_sysnode_seam_mismatch = pc->n_seam_mismatch.load();
            res->metrics.n_sysnode_probe_short   = pc->n_probe_short.load();
        }
    }

    // process_single_task(), SLOT_SAVE: true when the name is refused (the error is sent)
    bool manual_save_refused(const server_task & task, const std::string & filename) {
        // fork: auto-* names belong to the disk cache. Its units are immutable once published
        // and other instances sharing the directory may be reading one; overwriting it in place
        // (and dropping its .meta) would also orphan its delta children. Restoring one by name
        // stays allowed.
        if (filename.rfind("auto-", 0) == 0) {
            send_error(task, "filenames starting with 'auto-' are reserved for the disk cache", ERROR_TYPE_INVALID_REQUEST);
            return true;
        }
        return false;
    }

    // process_single_task(), SLOT_SAVE: publish the temp file and its sidecars, false when the save failed
    // (the error is sent)
    bool manual_save_publish(const server_task & task, server_slot * slot, const std::string & filepath,
                             const std::string & tmp, size_t nwrite) {
        {
            std::error_code ec;
            std::filesystem::remove(slot_meta_sidecar_path(filepath), ec);
            std::filesystem::remove(slot_logits_sidecar_path(filepath), ec);
            std::filesystem::remove(slot_draft_sidecar_path(filepath), ec);
            std::filesystem::remove(slot_fp_sidecar_path(filepath), ec);
            std::filesystem::rename(tmp, filepath, ec);
            if (ec) {
                std::filesystem::remove(tmp, ec);
                send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                return false;
            }
        }
        // fork: the model / projector identity upstream's format does not carry (see
        // slot_fp_sidecar_path). Best effort: without it the restore is unchecked, as upstream's.
        if (!slot_fp_write(filepath, { cur_fp.fp_model, cur_fp.fp_mmproj_loaded, cur_fp.fp_mmproj })) {
            SLT_WRN(*slot, "%s", "failed to write the fingerprint sidecar; a restore of this file is unchecked\n");
        }

        // fork: the regenerate logits sidecar (FULL, RS and NO classes), next to upstream's file and
        // outside it, so the .bin stays upstream's format. Written only when the captured logits
        // belong to exactly the state just saved (stamped with its token count), so a stale
        // distribution can never be paired with this snapshot. Best effort: without it an exact
        // resend after the restore re-prefills instead of emitting its first token at once.
        if (logits_sidecar_manual()) {
            const size_t n_tok = slot->prompt.tokens.size();
            if (slot->logits_last_n_tokens == (int32_t) n_tok && !slot->logits_last.empty()) {
                const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
                if (slot_logits_write(filepath, slot->logits_last, nv, (uint32_t) n_tok) == 0) {
                    SLT_WRN(*slot, "%s", "failed to write logits sidecar; regenerate fast-path disabled for this snapshot\n");
                }
            } else {
                SLT_DBG(*slot, "no matching captured logits for this state (stamp=%d, n_tokens=%zu); sidecar omitted\n",
                        slot->logits_last_n_tokens, n_tok);
            }
        }

        // enforce the bounded slot-save store (LRU by mtime). If this single
        // snapshot exceeds the byte cap, reject the save instead of evicting everything.
        // Eviction is opt-in: it runs ONLY when --slot-save-auto owns this directory as a
        // server-managed cache. A plain manual --slot-save-path save never deletes files.
        if (nwrite > 0 && auto_cache_enabled() &&
            (params_base.slot_save_max_count > 0 || params_base.slot_save_max_bytes > 0)) {
            bool oversized = false;
            slot_save_enforce_limits(params_base.slot_save_path,
                                     params_base.slot_save_max_count,
                                     params_base.slot_save_max_bytes,
                                     filepath, oversized);
            if (oversized) {
                send_error(task,
                           "slot snapshot exceeds --slot-save-max-mb; save rejected",
                           ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
        }
        return true;
    }

    // process_single_task(), SLOT_RESTORE: a file with a .meta sidecar is restored here, true when it was (the
    // result or the error is sent); otherwise the slot is prepared for upstream's restore below
    bool manual_restore_meta(const server_task & task, server_slot * slot, int id_slot, const std::string & filename,
                             const std::string & filepath, int64_t t_start) {
        // fork: a file with a .meta sidecar is an auto-cache unit, or a manual media save
        // written before this endpoint adopted upstream's format. Neither is upstream's
        // format, so it keeps the fork path, which composes delta chains and rebuilds media
        // from the sidecar. Every other file takes upstream's body below, unchanged.
        std::error_code meta_ec;
        aw_def_force_slot(id_slot, "forced"); // the restore replaces the slot's cells
        if (std::filesystem::exists(slot_meta_sidecar_path(filepath), meta_ec) && !meta_ec) {
            // shared store lock until the chain and its media sidecar are read (auto_store_lock)
            auto_store_lock rd_lock(std::filesystem::path(filepath).parent_path().string(),
                                    /*exclusive=*/false, auto_store_lock::RESTORE_TIMEOUT_MS);
            if (!rd_lock.held()) {
                send_error(task, "Unable to restore slot: the slot-save store is busy, retry",
                           ERROR_TYPE_UNAVAILABLE);
                return true;
            }
            // Compose the restore chain (U9 / decision 2). A whole unit (v1/v2 snapshot, parent 0)
            // restores as a single clearing load. But the same directory may hold an auto-cache
            // v3/v4 DELTA tip whose .bin
            // carries only cells [parent_hi, N); pointing a manual restore at one MUST NOT be
            // refused (decision 2). Read its node tail and, when it is a delta, walk parent
            // links via the SHARED auto_build_restore_chain so base + deltas compose to the
            // full cell set before loading: the identical path the auto restore uses, so a v4
            // media delta tip restores correctly here too. A missing .meta / whole snapshot
            // (parent 0) keeps the single-file chain; only a genuinely broken delta chain
            // (a parent file gone or corrupt) errors, the same way a corrupt whole unit would.
            std::vector<std::string> restore_chain = { filepath };
            {
                model_fp     tip_fp;
                llama_tokens tip_toks;
                std::vector<server_media_record> tip_media;
                uint64_t tip_parent_id = 0;
                uint32_t tip_range_lo  = 0;
                uint32_t tip_range_hi  = 0;
                if (slot_meta_read(filepath, cur_fp.fp_mmproj, tip_fp, tip_toks, tip_media,
                                   &tip_parent_id, &tip_range_lo, &tip_range_hi) &&
                    (tip_parent_id != 0 || tip_range_lo != 0)) {
                    if (!auto_build_restore_chain(filepath, tip_toks, tip_parent_id, tip_range_lo,
                                                  restore_chain)) {
                        send_error(task, "Unable to restore slot: the delta snapshot's parent "
                                         "chain is missing or inconsistent", ERROR_TYPE_INVALID_REQUEST);
                        return true;
                    }
                }
            }

            // Shared restore body (also used by the transparent auto-restore path): loads the
            // composed chain into seq slot->id, sets just_restored + restored_logits, rebuilds
            // the FULL-model checkpoint. On a load failure the slot seq is cleared and we error.
            size_t token_count = 0;
            size_t nread = 0;
            if (!do_slot_restore(*slot, restore_chain, &token_count, &nread)) {
                send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                return true;
            }

            // media snapshots: rebuild the prompt's media chunks as stubs from the v2
            // .meta sidecar: without them the restored NULL cells are untraversable.
            // Any rehydration failure drops the loaded state entirely (a half-rehydrated
            // slot must never survive to serve requests) and errors with the reason.
            {
                std::string err;
                if (!manual_restore_rehydrate_media(*slot, filepath, err)) {
                    auto_restore_drop(*slot);
                    send_error(task, "cannot restore media snapshot: " + err, ERROR_TYPE_INVALID_REQUEST);
                    return true;
                }
            }

            restore_swa_checkpoint(*slot);

            const int64_t t_end = ggml_time_us();
            const double t_restore_ms = (t_end - t_start) / 1000.0;

            auto res = std::make_unique<server_task_result_slot_save_load>();
            res->id       = task.id;
            res->id_slot  = id_slot;
            res->filename = filename;
            res->is_save  = false;
            res->n_tokens = slot->prompt.tokens.size();
            res->n_bytes  = nread;
            res->t_ms     = t_restore_ms;
            queue_results.send(std::move(res));
            return true;
        }

        // fork: nothing in ctx_dft belongs to the state about to be loaded, and a failed load
        // below must not leave the previous restore's flag and logits armed
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot->id, -1, -1);
        }
        slot->just_restored = false;
        slot->restored_logits.clear();
        return false;
    }

    // process_single_task(), SLOT_RESTORE: refuse (throw) a file whose .fp sidecar names another model or projector
    void manual_restore_check_fp(const std::string & filepath, const server_tokens & restored) {
        // fork: the fingerprint sidecar, when the save wrote one (see slot_fp_sidecar_path).
        // Text KV is projector-independent, so the projector is compared only for a media save.
        {
            slot_fp_record rec;
            const int st = slot_fp_read(filepath, rec);
            if (st < 0) {
                throw std::runtime_error("unreadable fingerprint sidecar (.fp)");
            }
            if (st > 0 && rec.fp_model != cur_fp.fp_model) {
                throw std::runtime_error("the file was saved by a different model");
            }
            if (st > 0 && restored.has_media() &&
                (rec.fp_mmproj_loaded != cur_fp.fp_mmproj_loaded || rec.fp_mmproj != cur_fp.fp_mmproj)) {
                throw std::runtime_error("the file holds media encoded by a different projector (--mmproj)");
            }
        }
    }

    // process_single_task(), SLOT_RESTORE: after upstream's load. The checkpoints the save appended to the
    // file (load_slot_checkpoints) are the ones the slot held when it was saved, so a restore that loaded
    // them keeps exactly those: the restored slot is the saved one, and saving it again writes the same
    // file, as upstream's restore does. Only a file without them (written before upstream #26004, or with
    // an appendix that was ignored) gets the checkpoint synthesised at the restored position, which stands
    // in for the ones it could not carry.
    void manual_restore_finish(server_slot * slot, const std::string & filepath, bool has_saved_ckpts) {
        std::list<common_prompt_checkpoint> saved;
        if (has_saved_ckpts) {
            saved = std::move(slot->prompt.checkpoints);
        }

        // fork: restore flags, FULL/SWA checkpoints and the logits sidecar (see restore_finish)
        restore_finish(*slot, &filepath, slot->prompt.tokens.size(), /*synth_ckpt=*/!has_saved_ckpts);
        if (!has_saved_ckpts) {
            restore_swa_checkpoint(*slot);
            return;
        }
        // restore_finish dropped the list (it is there for a previous task's checkpoints)
        slot->prompt.checkpoints = std::move(saved);
    }

    // after the decode loop: the mid-prefill saves of slots resident at a node or checkpoint position
    void prefill_nodes_save() {
        // ===== MID-PREFILL SHARED-CONTEXT BASE whole-save (Option A) =====================
        // Runs AFTER the decode loop, so every batch chunk has been llama_decode'd and the KV
        // cells for the tokens added in pre_decode() actually exist. A slot armed at B_ctx and now
        // resident at EXACTLY [0, B_ctx) (clamped by the prefill loop, still SLOT_STATE_PROCESSING_
        // PROMPT because B_ctx < N is a strict prefix) holds the true whole state at B_ctx: whole-
        // save it ONCE, then disarm so the slot prefills [B_ctx, N) normally. post_decode() is NOT a
        // viable hook: a still-prefilling slot has i_batch == -1 and is skipped by its
        // is_inside_view() early-return. Cost when nothing is armed: one cheap slot scan.
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_PROCESSING_PROMPT) {
                continue;
            }
            if (slot.ctx_save_pos > 0 && slot.prompt.n_tokens() == slot.ctx_save_pos) {
                auto_save_context_base(slot);
                slot.ctx_save_pos = -1; // one-shot: neither re-clamp nor re-save this task
            }
            if (slot.prompt_save_pos > 0 && slot.prompt.n_tokens() == slot.prompt_save_pos) {
                // the slot is resident at exactly [0, prompt_save_pos): the regular save persists that
                // prefix (a delta on the deepest saved node under --slot-save-incremental)
                auto_save_slot_if_useful(slot, AUTO_SAVE_SITE_PROMPT_NODE);
                slot.prompt_save_pos = -1;
            }
            // periodic prefill checkpoints: the same mechanism every --slot-save-prefill-interval tokens
            while (slot.ckpt_save_idx < slot.ckpt_save_pos.size() &&
                   slot.ckpt_save_pos[slot.ckpt_save_idx] <= slot.prompt.n_tokens()) {
                if (slot.ckpt_save_pos[slot.ckpt_save_idx] == slot.prompt.n_tokens()) {
                    SLT_INF(slot, "auto-save: prefill checkpoint at %d of %d prompt tokens\n",
                            slot.prompt.n_tokens(), slot.task->n_tokens());
                    auto_save_slot_if_useful(slot, AUTO_SAVE_SITE_PREFILL_CKPT);
                }
                slot.ckpt_save_idx++;
            }
            if (aw.test_prefill_delay_ms > 0) {
                // test hook (LLAMA_TEST_SLOT_SAVE_PREFILL_DELAY_MS): slow every prefill batch down so a test can
                // interrupt a prefill between two checkpoints
                std::this_thread::sleep_for(std::chrono::milliseconds(aw.test_prefill_delay_ms));
            }
        }
        // ===== end MID-PREFILL BASE =====================================================
    }

    // pre_decode(), new prompt: a task with a shared prompt prefix is neither restored nor saved, say so
    void shared_prefix_note(server_slot & slot) {
        if (auto_cache_enabled() && slot.task->n_tokens_shared > 0) {
            // a shared prompt prefix (decision tasks): the parent and its children neither
            // restore nor save (need_sampling() is false for the parent), so say so
            metrics.n_auto_skipped_shared++;
            const int64_t now_ms = ggml_time_ms();
            if (auto_shared_last_wrn_ms < 0 || now_ms - auto_shared_last_wrn_ms >= 60 * 1000) {
                auto_shared_last_wrn_ms = now_ms;
                SLT_WRN(slot, "auto disk cache: a task with a shared prompt prefix (%d tokens) is neither restored nor saved "
                              "(skipped so far = %" PRIu64 "; this line is rate-limited to one per minute, see "
                              "llamacpp:auto_cache_skipped_shared_total)\n",
                        (int) slot.task->n_tokens_shared, metrics.n_auto_skipped_shared);
            }
        }
    }

    // pre_decode(), new prompt: restore a longer prefix of the request from disk, n_past recomputed
    void auto_restore_lookup(server_slot & slot, const server_tokens & input_tokens, int & n_past) {
        // ===== AUTO-RESTORE: cold/cross-process KV reuse from disk (opt-in) ==========
        // If the in-memory match (n_past) is POOR and the disk index holds a snapshot
        // whose persisted tokens are a verified, fingerprint-matching, longer prefix of
        // this request, restore it INTO the slot and RECOMPUTE n_past so all downstream
        // machinery runs unchanged, agnostic to HOW the tokens arrived. Media requests
        // look up exactly like text ones: the chain hashes fold each chunk's identity
        // record, and auto_restore_into_slot byte-verifies the cells PLUS every media
        // record (id/shape/type) before any multi-GB load, falling back to a normal
        // prefill on any mismatch/failure (invariants 2/3/4). (An earlier revision
        // gated this block on !slot.prompt.tokens.has_media(), which tested the
        // slot's STALE PREVIOUS prompt, not the request; no request property replaces
        // it, media requests are simply first-class now.)
        // Decision tasks (the only ones that set n_tokens_shared, on the group
        // PARENT) never reach this block: need_sampling() is false for them, and
        // auto-save excludes them as well, so they neither restore nor publish.
        if (auto_cache_enabled()
                && slot.task->need_sampling()        // generative only (not embed/rerank)
                && slot.alora_invocation_start <= 0      // aLoRA caching bound (mirror below)
                && are_lora_equal(slot.lora, params_base.lora_adapters)) { // fp captures global LoRA (invariant 3)
            // Try candidates best-first (longest usable snapshot first). Each
            // rejected candidate costs only a small .meta read + byte-compare; the
            // multi-GB state loads only once a candidate passes its gates. This
            // fall-through is what stops a longer superset snapshot (unusable by a
            // FULL model, which needs a whole-prefix match) from shadowing a shorter
            // usable one at the same boundary. Media requests look up first-class:
            // auto_index_lookup folds each chunk's identity into the boundary hashes.
            int n_restored = 0;
            auto_not_prefix_skips = 0;
            // the margin gate compares a disk unit against what the slot can really keep:
            // a class that cannot rewind keeps only a strict extension of its cells, or an
            // in-memory context checkpoint at or below the divergence, not the raw LCP
            const int32_t n_usable = auto_usable_warm_reuse(slot, n_past, (int32_t) input_tokens.size());
            if (n_usable < n_past) {
                SLT_DBG(slot, "auto-restore: the slot matches %d tokens but can keep only %d of them\n",
                        (int) n_past, n_usable);
            }
            for (const auto & cand : auto_index_lookup(input_tokens)) {
                // auto_restore_into_slot may CLEAR the slot
                // (KV seq + prompt.tokens) and then have do_slot_restore FAIL
                // (corrupt/short .bin, KV-capacity exceeded, racing LRU eviction
                // deleting the file mid-read). In that case the slot tokens are now
                // empty. We therefore RECOMPUTE n_past UNCONDITIONALLY after any
                // attempt, not only on success, so a cleared-but-failed restore
                // falls back to n_past=0 (clean cold prefill) instead of carrying a
                // stale n_keep_mem>0 into keep_first() on an empty token vector
                // (which would GGML_ASSERT/abort). The recompute is harmless on the
                // early-return-before-clear paths (margin/fp/verify rejects): those
                // leave prompt.tokens untouched, so the LCP is identical to before.
                const int restored = auto_restore_into_slot(slot, cand, input_tokens, n_usable, (int) n_past);
                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);
                if (restored > 0) {
                    n_restored = restored;
                    break; // restored; stop trying shorter candidates
                }
            }
            // hit / miss accounting (llamacpp:auto_cache_restore_*_total and the
            // per-request timings.cache_disk_n). A MISS is a request that still had
            // at least one whole block beyond its in-memory match and got nothing
            // from disk: that includes genuinely new prompts no cache could hold,
            // so read it next to auto_cache_evicted_total, not as a ratio alone.
            // A restore is only a HIT once the prompt actually keeps it: it is counted
            // where n_prompt_cached is final (auto_restore_account), because a later
            // n_past clamp can still discard it (e.g. an RS exact resend re-prefills).
            if (n_restored > 0) {
                // account what the restore actually loaded, not the block-aligned claim:
                // a plain-attention restore keeps the whole unit and the generic path trims
                // it to the real LCP (n_past, recomputed above), every cell of which came
                // from disk since the restore replaced the slot. auto_restore_account then
                // clamps it to the final n_past.
                slot.n_auto_restored_pending = std::max(n_restored, (int) n_past);
            } else if ((int) input_tokens.size() >= n_usable + params_base.slot_save_block) {
                metrics.n_auto_restore_miss++;
                {
                    // a unit of this model held more of the prefix under another identity
                    std::string why;
                    int32_t     cover = 0;
                    {
                        std::lock_guard<std::mutex> lk(auto_idx.mtx);
                        cover = auto_foreign_cover_locked(input_tokens, why);
                    }
                    if (cover >= n_usable + params_base.slot_save_block) {
                        metrics.n_auto_restore_miss_identity++;
                        SLT_WRN(slot, "auto-restore: miss, a unit of this model covers %d tokens of the prefix but was written under another identity (%s); it is never restored here\n",
                                cover, why.c_str());
                    }
                }
                if (auto_not_prefix_skips > 0) {
                    // a snapshot of this prefix exists, but the request diverges inside it
                    // and this class cannot rewind (FULL, RS, NO, or SWA past one window):
                    // only a node at or before the divergence could have served it
                    metrics.n_auto_restore_not_prefix++;
                    SLT_WRN(slot, "auto-restore: miss, %zu snapshot(s) share this prefix but the request diverges inside them and this memory class cannot rewind (seq_rm_type = %d, n_swa = %d); a node at or before the divergence is needed\n",
                            auto_not_prefix_skips, (int) ctx_tgt_seq_rm_type, n_swa_mem);
                }
            }
        }
        // ===== end AUTO-RESTORE =====================================================
    }

    // pre_decode(), new prompt: true when the first token was emitted from the restored logits (the slot needs
    // no prompt batch); may instead clear the restored state, n_past and pos_next with it
    bool restore_continue(server_slot & slot, int & n_past, llama_pos & pos_next) {
        // ===== restore-continue (regenerate fast-path) ==============================
        // A just-restored FULL/recurrent slot receiving the EXACT restored tokens (no
        // suffix) cannot rewind its memory: the normal path would re-decode into the
        // already-occupied sequence and crash (the pure-recurrent gate at the relaxed
        // checkpoint predicate below is FALSE for this case, so just_restored would never
        // be consumed and [TAG_PROMPT_LOGITS] would decrement n_past and re-decode into
        // the occupied sequence). Detect that case here, BEFORE the n_past>0 guard, and:
        //   - if we have the saved next-token logits: emit the first token with NO decode,
        //     keeping the full restored state, then continue normal autoregression;
        //   - otherwise: fall back to a SAFE clear-then-reprefill (never crash).
        // Gated so it is unreachable for non-recurrent models, with-suffix requests
        // and non-generative slots. Media prompts DO reach it: the auto path never
        // had a server-wide mtmd gate (and the manual /slots endpoints save media
        // slots in upstream's format), so a byte-identical media resend on a FULL-seq_rm model
        // restores its whole snapshot and lands exactly here.
        // The body is media-safe by construction:
        // init_sampler skips LLAMA_TOKEN_NULL cells, and the speculative begin feeds
        // get_text_tokens() (the media-safe accessor), never get_tokens(). With-suffix
        // restore reuse is left entirely to the unchanged relaxed-predicate path below.
        // NOTE: cache_prompt==false sets n_past=0 above, so this gate (which
        // requires n_past == task->n_tokens()) is naturally not entered for a
        // no-cache request; that case safely takes the normal full-clear +
        // reprefill path (common_context_seq_rm at [p0,-1) empties the restored
        // sequence first), so regenerate degrades to a cold reprefill, never a crash.
        // No `n_past < n_ctx` clause: the fast path emits with NO decode so it
        // needs no free context slot; a full-n_ctx no-suffix restore is handled
        // here rather than falling through to a zero-token-added crash window.
        // Every class has a logits sidecar now, but a class that can rewind by one token
        // enters only with logits for exactly n_past tokens: a unit rewound to a shorter
        // request carries the logits of its own end, and without usable logits such a
        // class re-decodes one token below instead of re-prefilling everything.
        const bool restored_logits_ok = !slot.restored_logits.empty() && slot.restored_logits_n == n_past;
        const bool cannot_rewind_one  = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                                        ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS   ||
                                        ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO;
        if (slot.just_restored &&
            (restored_logits_ok || cannot_rewind_one) &&
            slot.task->need_sampling() &&
            slot.alora_invocation_start <= 0 &&
            n_past == slot.task->n_tokens() &&
            n_past == (int) slot.prompt.n_tokens()) {

            slot.just_restored = false; // one-shot consume (this path owns it)

            if (restored_logits_ok) {
                // --- fast path: emit first token from saved logits, no decode ---
                slot.stats.n_prompt_cached    = n_past; // entire prompt "reused"
                slot.stats.n_prompt_processed = 0;      // prompt_n = 0 => observable reuse signal
                auto_restore_account(slot, n_past);

                // prime the sampler over the full restored prompt (penalties/grammar
                // history), exactly as the normal DONE_PROMPT transition (init_sampler) would.
                slot.stats.n_gen = 0;
                slot.init_sampler();

                slot.state   = SLOT_STATE_GENERATING;
                slot.i_batch = -1;

                // rebuild the (cold, unsaved) draft context for the restored prompt,
                // mirroring the normal prompt-done transition.
                if (slot.can_speculate()) {
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }

                const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
                const std::vector<float> saved_logits = std::move(slot.restored_logits);
                slot.restored_logits.clear(); // consumed
                const llama_token id = common_sampler_sample_from_logits(
                        slot.smpl.get(), saved_logits.data(), nv, /*grammar_first=*/false);

                common_sampler_accept(slot.smpl.get(), id, true);

                // mirror the generation accounting from the normal sample path
                // (upstream: stats.n_gen, then update_prompt_last() on the first
                // token, then update_gen_last()). No prompt metrics are queued:
                // this path processes zero prompt tokens by construction.
                const int64_t t_current = ggml_time_us();
                slot.stats.n_gen += 1;
                if (slot.stats.n_gen == 1) {
                    slot.stats.update_prompt_last();
                    slot.t_print_last = t_current;
                    slot.n_gen_last   = 0;
                }
                slot.stats.update_gen_last();

                if (slot.task->params.stream) {
                    // mirror the normal prompt-start streaming signal exactly so a
                    // return_progress client still gets its initial 0% progress event
                    if (slot.task->params.return_progress) {
                        send_partial_response(slot, {}, true);
                    } else {
                        // signal HTTP to send the headers (200 status)
                        send_partial_response(slot, {}, false, true);
                    }
                }

                completion_token_output result;
                result.tok  = id;
                // inline of the accept_special_token lambda (defined later in this method,
                // out of scope here): keep special tokens iff the server allows specials
                // or the request explicitly preserves this token.
                const bool keep_special =
                    params_base.special ||
                    slot.task->params.sampling.preserved_tokens.find(result.tok) !=
                        slot.task->params.sampling.preserved_tokens.end();
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, keep_special);
                result.prob         = 1.0f;

                // First-token logprobs (n_probs>0): the post-sampling variant reads the
                // candidate set (cur_p), which common_sampler_sample_from_logits leaves
                // populated, so we can serve it exactly as the normal path does. The
                // pre-sampling variant is computed from the saved logits themselves (the
                // context holds no row for a decode this path never ran).
                if (slot.task->params.sampling.n_probs > 0) {
                    const bool post = slot.task->params.post_sampling_probs;
                    populate_token_probs(slot, result, post, params_base.special, /*idx=*/-1,
                                         post ? nullptr : saved_logits.data());
                }

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    metrics_on_prediction(slot);
                    slot.release();
                }

                SLT_INF(slot, "%s", "restore-continue: emitted first token from saved logits (prompt_n=0)\n");
                return true; // skip the rest of prompt-batch building for this slot; iterate() proceeds to the next
            }

            // --- safe fallback: no valid sidecar -> clear restored seq, then reprefill ---
            // FULL models support full-sequence removal; clearing first guarantees the
            // subsequent reprefill writes into an EMPTY sequence instead of re-decoding
            // into the already-occupied restored state (which is the crash being fixed).
            SLT_WRN(slot, "%s", "restore-continue: no saved logits; clearing restored state and re-prefilling\n");
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, -1, -1);
            slot.prompt.tokens.clear();
            slot.prompt.checkpoints.clear();
            n_past   = 0;
            pos_next = 0; // mirror the do_reset path; keep the stale full-length value from leaking into the checkpoint-erase loop below
            // fall through to the normal guard below with an empty sequence (safe)
        }
        // ===== end restore-continue =================================================
        return false;
    }

    // pre_decode(), new prompt: consumes the slot's restore flag; true when a restored window memory is strictly
    // extended and the checkpoint search must be skipped
    bool restored_extend_only_get(server_slot & slot, bool has_new_tokens, bool pos_min_is_tail, int n_past) {
        // Consume just_restored UNCONDITIONALLY, before the gate: the tail-aware threshold
        // makes the gate false for the restore-plus-suffix case, so a consume inside the
        // gate would leave the flag set. It is also cleared in server_slot::reset(), which
        // bounds an unconsumed flag to the task that owns it: this consume sits inside the
        // `n_past > 0 && n_past <= prompt.n_tokens()` block, which a fully divergent or
        // cache_prompt == false task never enters.
        const bool slot_was_restored = slot.just_restored; slot.just_restored = false;

        // A just-restored WINDOW memory (PART, in practice SWA) that is being
        // strictly EXTENDED needs no rewind at all, so skip the search below
        // entirely. Every position the model can attend to from n_past is
        // present: llama_kv_cache::state_write persists exactly the unmasked
        // window [n_past - n_swa, n_past). This is NOT the same cell set a live
        // slot holds, and the difference is the whole reason the gate misfires
        // here: find_slot only marks an SWA-masked cell REUSABLE, it does not
        // erase it, and the SWA sub-cache is sized PAD(min(size_base, n_swa *
        // (unified ? n_seq_max : 1) + n_ubatch), 256): normally > n_swa, though
        // NOT when size_base < n_swa (a tiny --ctx-size), where the min clamps it.
        // In the normal case a live slot still physically holds older positions
        // and reports a pos_min strictly BELOW pos_min_thold, which keeps the gate
        // shut for it. A restored slot reports pos_min == n_past - n_swa ==
        // pos_min_thold, so the gate opens.
        // The search can only do harm here: the ONLY checkpoint such a slot has is the one
        // synthesised from the LIVE window after the restore (do_slot_restore
        // clears the list for every class, then auto_restore_into_slot and the
        // manual SWA branch each create one, or NONE under --ctx-checkpoints 0,
        // where create_checkpoint now returns early), whose pos_min IS the window
        // start L - n_swa, the same expression as pos_min_thold when
        // has_new_tokens. The acceptance test is
        // `cur.pos_min == 0 || cur.pos_min < pos_min_thold`, and the second
        // disjunct is then false, so do_reset fires and the SWA disk restore is
        // loaded and thrown away. NOT universal, and the first disjunct is why:
        // when L <= n_swa_mem nothing was masked at save time, pos_min == 0, and
        // the checkpoint IS accepted (losing one position rather than everything).
        // That regime is narrow: the save floor is
        // max(slot_save_block, slot_save_min_tokens), so on gemma-26b-a4b
        // (n_swa 1024, floor 1024) it needs L == 1024 exactly, and it widens only
        // for n_swa > 1024 or a lowered --slot-save-min-tokens. The measured
        // production case (13,611 tokens, n_swa 1024) is the regime above.
        // The four terms, exactly: (1) the slot carries a restore no earlier task
        // consumed, i.e. an auto-restore performed for THIS task or a manual
        // /slots restore with no task since (server_slot::reset() clears the flag,
        // which bounds it to one task); (2) the
        // request strictly extends it, so no rewind is needed; (3) pos_min is a
        // real window start, NOT a tail, so FULL and RS keep the unchanged path,
        // because for them the search is what keeps the NOTE's GGML_ABORT
        // unreachable; (4) the restored KV ends exactly at n_past, so nothing
        // above it needs removing (a diverging manual /slots restore has
        // n_past < prompt.n_tokens() and still takes the search).
        // Inert for PART with n_swa == 0 (no checkpoints are ever created for it,
        // and pos_min == 0 keeps the gate closed) and under --swa-full (n_swa == 0
        // makes pos_min_thold == pos_next, gate closed).
        // KNOWN GAP, deliberately not covered: the SWA EXACT-resend case
        // (has_new_tokens false) keeps today's behaviour, checkpoint rejected by
        // one position, cold reprefill. Covering it would need either an
        // off-by-one relaxation of the acceptance test or widening this skip to a
        // path a warm slot does not take, which is the class of change that
        // produced this file's incident history. Extension is the case with
        // production evidence.
        const bool restored_extend_only = slot_was_restored && has_new_tokens &&
                                          !pos_min_is_tail &&
                                          n_past == slot.prompt.n_tokens();
        return restored_extend_only;
    }

    // pre_decode(), new prompt: arm the system node, the prompt node and the periodic prefill checkpoints
    void prefill_nodes_arm(server_slot & slot, const server_tokens & input_tokens, int n_past,
                           bool restored_now, int32_t n_resident, int32_t restored_unit) {
        // ===== ARM the mid-prefill shared-context base save (Option A) ==============
        // Reset here first (STARTED runs exactly once per task) so a target left over
        // from a released/aborted task never leaks into slot reuse. Then arm iff this is
        // a COLD prefill of a text-only generative request that has a block-aligned
        // first-user boundary B_ctx clearing the floor and strictly inside the prompt.
        // The prefill loop clamps the batch at B_ctx; the post-decode hook whole-saves
        // [0, B_ctx) when the slot is resident at exactly B_ctx. Because the resident
        // sequence IS the true whole state at B_ctx, the whole-save is sound for dense,
        // SWA AND recurrent/hybrid: there is NO model-class gate (unlike the old
        // idle-flush [0,B) sub-range checkpoint this replaces).
        slot.ctx_save_pos    = -1;
        slot.prompt_save_pos = -1;
        slot.ckpt_save_pos.clear();
        slot.ckpt_save_idx   = 0;
        slot.ckpt_prev_whole.clear();
        const bool cold_prefill_was_open = slot.cold_prefill_open;
        slot.cold_prefill_open = false;
        if (auto_cache_enabled() &&
            slot.task->need_sampling() &&               // generative only (not embed/rerank; keeps the can_split path)
            slot.alora_invocation_start <= 0 &&         // aLoRA caching bound (mirror the auto-restore gate)
            are_lora_equal(slot.lora, params_base.lora_adapters)) { // fp captures the global LoRA set (invariant 3)
            const int     B        = params_base.slot_save_block;
            const int32_t n_prompt = slot.task->n_tokens();
            const auto &  spans    = slot.task->params.message_spans;

            // A mid-prefill cut must not split a media chunk, and the prefill loop can only
            // stop right after a TEXT token (it clamps after pushing one; a chunk is
            // processed whole). So a node sits where the previous cell is text: a text
            // position, or the start of a chunk. A candidate inside or right after a chunk
            // moves down to that chunk's start. A no-op for a text-only prompt.
            const auto cut_down = [&](int32_t p) -> int32_t {
                while (p > 0 && input_tokens[p - 1] == LLAMA_TOKEN_NULL) {
                    p--;
                }
                return p;
            };

            // --- SYSTEM node (--slot-save-node-system, default on) ---
            // Position: the end of the leading system preamble, exactly (nodes need not be
            // block-aligned). For a chat request it is template-generic: the prefix of the
            // prompt that the system / developer messages, the tools and the template decide on
            // their own (server_preamble_cache), so a request carrying only a system prompt (a
            // pre-cache of it) caches all of it, and a template's default system prompt counts
            // too. Without one (a raw /completion prompt), the message delimiters: the first
            // user message, or the end of the system span when there is no user message.
            // Capped at n_prompt - 1: the node must be a strict prefix to be cut mid-prefill.
            // Floor: --slot-save-context-min-tokens. COLD only, and only when at least one block
            // of [0, B_ctx) is new: a node a few tokens past one just restored (a system-only
            // pre-cache followed by system + user requests) would be a near-duplicate root.
            if (params_base.slot_save_node_system) {
                const int     floor    = std::max(B, params_base.slot_save_context_min_tokens);
                int32_t       B_ctx    = slot.prompt.ctx_boundary; // stashed at task launch
                if (B_ctx <= 0) {
                    B_ctx = spans.system_only_context_end();
                }
                B_ctx = B_ctx > 0 ? std::min(B_ctx, n_prompt - 1) : -1;
                const auto    ok    = [&](int32_t x) { return x >= floor && x < n_prompt && n_past + B <= x; };
                const int32_t B_cut = B_ctx > 0 ? cut_down(B_ctx) : -1;
                if (ok(B_cut)) {
                    slot.ctx_save_pos = B_cut;
                } else if (B_cut != B_ctx && ok(B_ctx)) {
                    metrics.n_auto_node_media_skipped++; // only the media cut stopped it
                }
            }

            // --- PROMPT node (--slot-save-node-prompt, default cold) ---
            // Position: the end of the last user message, exactly, like the system node
            // (n_prompt - 1 block-aligned down without a user span). A unit saved after the
            // response (idle, reclaim, shutdown, --slot-save-node-response) is prompt +
            // generation, and a memory class that cannot rewind (recurrent, hybrid, FULL,
            // sliding-window past one window) can restore it only for a request that extends
            // it. A resend, a regenerate, an edit of the response or a follow-up whose
            // history is not re-rendered token for token diverges inside it; this node sits
            // before that divergence. 'cold' arms it only for a prompt that got essentially
            // no reuse (n_past < the save floor), so at most once per conversation start. 'on'
            // arms it whenever at least one block of new prompt precedes it: that is the
            // setting for a client that re-renders the previous response differently on
            // every turn (reasoning dropped or trimmed, answers rewritten, tool calls
            // re-serialised), whose after-response units can never serve the next turn on
            // these classes, and for regenerate or edit after a restart. The mid-prefill save
            // is the true whole state at that position, so it is sound for every memory class.
            // Floor: max(--slot-save-block, --slot-save-min-tokens), as for any other save.
            // When the system node is armed, the prompt node must lie at least one block past
            // it: anything closer is a prefix that the system node already covers for every
            // request that could reuse it, and would only cost a second save on a cold first
            // prompt with a short first message.
            // A prefill that resumes an interrupted one counts as cold too (--slot-save-prefill-interval
            // only, so the default behaviour is unchanged): the prompt that was cut short never reached
            // its prompt node. Recognised by (a) this slot still holding all of an unfinished cold
            // prefill, (b) a disk restore of a whole unit (not trimmed) whose length is a multiple of the
            // interval, which is what a periodic checkpoint is (any other unit ends there only by
            // coincidence, about once in N/block system nodes, at the cost of one extra node), or
            // (c) a reuse that already reaches into the last user message, which only a checkpoint
            // inside it (or a warm slot that prefilled part of it) can give. A warm or RAM reuse that
            // happens to end on a multiple of N is not a resume.
            const int32_t interval = params_base.slot_save_prefill_interval;
            const int32_t user_pos = spans.last_user_message_pos();
            const bool    resumed  = interval > 0 && n_past > 0 && (
                    (cold_prefill_was_open && !restored_now && n_past == n_resident) ||
                    (restored_now && restored_unit == n_past && n_past % interval == 0) ||
                    (user_pos > 0 && n_past > user_pos));
            const bool    is_cold  = n_past < std::max(B, params_base.slot_save_min_tokens) || resumed;
            if (params_base.slot_save_node_prompt != COMMON_SLOT_SAVE_NODE_PROMPT_OFF) {
                const int     floor    = std::max(B, params_base.slot_save_min_tokens);
                const int32_t user_end = spans.last_user_message_end();
                const int32_t e        = std::min(user_end > 0 ? user_end : n_prompt - 1, n_prompt - 1);
                // exact at the end of the user message (the history re-renders everything after it,
                // the generation prompt included, so nothing past it is safe); without a user span,
                // block-aligned down from n_prompt - 1 so that it still lies before most generation
                // prompts
                const int32_t B_al     = e <= 0 ? -1 : (user_end > 0 ? e : e - (e % B));
                const int32_t B_p      = B_al > 0 ? cut_down(B_al) : -1;
                const auto    ok       = [&](int32_t x) {
                    const bool want = params_base.slot_save_node_prompt == COMMON_SLOT_SAVE_NODE_PROMPT_ON
                                      ? n_past + B <= x
                                      : is_cold;
                    const bool past_sys = slot.ctx_save_pos <= 0 || x >= slot.ctx_save_pos + B;
                    return want && past_sys && x >= floor && x < n_prompt && n_past < x;
                };
                if (ok(B_p)) {
                    slot.prompt_save_pos = B_p;
                } else if (B_p != B_al && ok(B_al)) {
                    metrics.n_auto_node_media_skipped++; // only the media cut stopped it
                }
            }

            // --- PERIODIC PREFILL CHECKPOINTS (--slot-save-prefill-interval, default off) ---
            // Every multiple of the interval strictly between the reuse and the end of the prompt,
            // at or above the save floor. Like the nodes above, a checkpoint is the true whole state
            // at its position, so it is sound for every memory class, and it is cut where the
            // previous cell is text (a position inside or right after a media chunk moves down to
            // the chunk's start; one that then falls at or below the reuse or the previous
            // checkpoint is dropped). A position the system or the prompt node already takes is not
            // armed twice.
            if (interval > 0) {
                const int32_t floor = std::max(B, params_base.slot_save_min_tokens);
                int32_t       prev  = n_past;
                for (int64_t p = ((int64_t) n_past / interval + 1) * interval; p < n_prompt; p += interval) {
                    const int32_t c = cut_down((int32_t) p);
                    if (c < floor || c <= prev || c == slot.ctx_save_pos || c == slot.prompt_save_pos) {
                        if (c != (int32_t) p && p >= floor) {
                            metrics.n_auto_node_media_skipped++;
                        }
                        continue;
                    }
                    slot.ckpt_save_pos.push_back(c);
                    prev = c;
                }
                if (params_base.slot_save_node_prompt == COMMON_SLOT_SAVE_NODE_PROMPT_COLD && is_cold) {
                    slot.cold_prefill_open = true; // cleared when the prompt is done
                }
                if (!slot.ckpt_save_pos.empty()) {
                    SLT_INF(slot, "auto-save: %zu prefill checkpoint(s) armed every %d tokens, first at %d (reuse %d of %d)%s\n",
                            slot.ckpt_save_pos.size(), interval, slot.ckpt_save_pos.front(), n_past, n_prompt,
                            resumed ? ", resuming an interrupted prefill" : "");
                }
            }
        }
        // ===== end ARM =============================================================
    }

    // pre_decode(), prompt batch: true when the batch must stop here, at a node or checkpoint position
    bool prefill_nodes_stop(const server_slot & slot) {
        // mid-prefill shared-context base (Option A): stop this batch EXACTLY at the
        // block-aligned first-user boundary B_ctx, never crossing it, so that once this
        // batch is decoded the slot's resident sequence is precisely [0, B_ctx): the
        // true whole state there for the post-decode whole-save. One-shot: the hook
        // disarms ctx_save_pos, after which the slot prefills [B_ctx, N) unclamped. If
        // B_ctx exceeds n_batch the loop's own batch.size() < n_batch cap stops it first
        // and a later update_slots() iteration re-enters here to reach B_ctx.
        if ((slot.ctx_save_pos    > 0 && slot.prompt.n_tokens() == slot.ctx_save_pos) ||
            (slot.prompt_save_pos > 0 && slot.prompt.n_tokens() == slot.prompt_save_pos)) {
            return true;
        }
        // the same for the periodic prefill checkpoints (empty unless --slot-save-prefill-interval)
        if (slot.ckpt_save_idx < slot.ckpt_save_pos.size() &&
            slot.prompt.n_tokens() == slot.ckpt_save_pos[slot.ckpt_save_idx]) {
            return true;
        }
        return false;
    }

    // pre_decode(), after the prompt batch: tasks waiting for a queued prefix to be published
    void aw_pending_wait(int n_aw_deferred) {
        if (n_aw_deferred > 0 && !aw.def.empty()) {
            // a task waits for a queued prefix: the writer cannot publish it before the deferred copies
            aw_def_force_all("wait");
        }
        if (n_aw_deferred > 0 && batch.size() == 0) {
            // nothing else to decode: wait briefly for the writer instead of spinning the loop (the queue
            // loop re-runs update_slots at once, and new tasks are taken between iterations)
            std::unique_lock<std::mutex> lk(aw.mtx);
            aw.cv.wait_for(lk, std::chrono::milliseconds(20));
        }
    }

    // post_decode(), sampled: the slot's last-token logits for the regenerate sidecar
    void logits_capture(server_slot & slot, int tok_idx) {
        // Feature A: capture this slot's last-token full-vocab logits for a possible disk
        // save. Cost: one ~n_vocab*4-byte copy per decoded token, incurred ONLY on the FULL and RS
        // classes (logits_sidecar_class) AND only when slot saving is enabled (--slot-save-path
        // set); attention models and servers without slot-save pay nothing at all. The copy is
        // unavoidable for correctness: ctx logits are overwritten by the next slot's decode,
        // so a lazy read at SLOT_SAVE would be wrong under --parallel>1. Captured per-slot
        // from this slot's own tok_idx so it is correct for any N/interleave (never read
        // from the shared ctx at save time). common_sampler_sample already synchronized the
        // context above, so llama_get_logits_ith is valid here.
        if (logits_sidecar_class() && !params_base.slot_save_path.empty()) {
            const float * lg = llama_get_logits_ith(slot.ctx_tgt, tok_idx);
            if (lg) {
                const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
                slot.logits_last.assign(lg, lg + nv);
                // Stamp the capture with the token count of the state it corresponds to.
                // At this point process_token() has NOT yet appended the just-sampled token,
                // so prompt.tokens.size() is exactly the length whose final token produced
                // these logits, i.e. it matches the token_count a SLOT_SAVE would record.
                slot.logits_last_n_tokens = (int32_t) slot.prompt.tokens.size();
            } else {
                // capture failed -> invalidate so a later save never serializes stale logits
                slot.logits_last.clear();
                slot.logits_last_n_tokens = -1;
            }
        }
    }

    // post_decode(), response done: the response node, when auto_save_after_response_wanted said so
    void response_node_save(server_slot & slot, bool save) {
        if (save) {
            auto_save_slot_if_useful(slot, AUTO_SAVE_SITE_RESPONSE_NODE);
        }
    }

    // post_decode(), speculative: the same logits capture for the last accepted token
    void logits_capture_spec(server_slot & slot, int32_t i_row_last, size_t n_draft,
                             const std::vector<llama_token> & accepted) {
        // the same last-token logits capture as the non-speculative path (see Feature A there):
        // the last accepted token was sampled from row i_row_last, and once the accepted tokens
        // are added below, prompt.tokens is exactly the state whose final token produced them
        if (logits_sidecar_class() && !params_base.slot_save_path.empty()) {
            const float * lg = i_row_last >= 0 ? llama_get_logits_ith(slot.ctx_tgt, i_row_last) : nullptr;
            if (lg) {
                const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
                slot.logits_last.assign(lg, lg + nv);
                slot.logits_last_n_tokens = (int32_t) (slot.prompt.n_tokens() - n_draft + accepted.size() - 1);
            } else {
                slot.logits_last.clear();
                slot.logits_last_n_tokens = -1;
            }
        }
    }

    // server_context::get_meta_disk_cache(): the cache's fields of the meta, which get_meta() leaves at their defaults
    void meta_fill(server_context_meta & meta) const {
        meta.fp_mmproj       = fp_mmproj;
        meta.auto_cache_caps = auto_cache_caps();
    }
