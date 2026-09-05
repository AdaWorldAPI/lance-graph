//! D-BLW-5 — the observer-effect probe: does awareness reflect a measured
//! statistic fed back into it as shape × Prozentrang (never a raw scalar)?
//!
//! Spec (binding, in precedence order): `.claude/knowledge/observer-effect-tfpn-doctrine.md`
//! (doctrine) → `.claude/board/exec-runs/d-blw-5-design-main-thread.md`
//! (design (a)-(g)) → `.claude/board/exec-runs/d-blw-5-api-inventory-sonnet.md`
//! (exact signatures) → `.claude/board/exec-runs/d-blw-5-build-spec-main-thread.md`
//! (THIS build, final numbers). Scaffolding provenance: `d_ign_b_lenses.rs`
//! (GREEN) — the fleet/scan/cast/seal machinery is copied verbatim from
//! there, cited at each site, per the build spec §2.
//!
//! ## Two corrections the build spec recorded (carried here verbatim, C1/C2)
//!
//! C1. Readers A/B are injection-invariant ONLY if B reads the OBSERVED
//! layer; B is therefore pinned to the arena's DERIVED layer (rung >= 1,
//! default stamp) so the tactics' propagation actually moves it.
//! C2. The injected rank enters awareness as TYPICALITY — the prior's own
//! mass at the observed rank bucket — bound to every subject via a shared
//! reserved `prior` predicate, never as a raw bucket index. Two ranks with
//! equal mass are indistinguishable to awareness under this encoding — a
//! stated limitation (see the not-claimed block).
//!
//! ## Deviations from the build spec, stated here (no others)
//!
//! 1. **No `run_loop(..., on_sealed: &mut dyn FnMut(...))` higher-order
//!    driver.** The build spec's §6 sketch has `run_loop` own both cycles
//!    and hand measurement to a callback; but the O6 firewall (§7: nothing
//!    with `kappa`/`binary_association(`/`fn reader_a`/`fn reader_b` (the O6 list) may
//!    appear textually before the marker) and the injection step's need for
//!    `&mut Mind` access (readers only need `&Mind`) mean the measurement
//!    code and the injection code cannot both live inside one callback
//!    signature without either violating O6 (defining the readers before
//!    the marker) or granting the callback mutable fleet access it has no
//!    textual right to (the design's own signature passes only shared
//!    refs). This file instead makes `run_loop` a single-CYCLE mechanics
//!    function (cast/scan/seal, returns `CycleOutcome`) defined BEFORE the
//!    marker, called twice from the test body (which lives AFTER the
//!    marker, so it is free to call `measure_cohort`/`binary_association`
//!    between the two calls). O6's actual assertion — the literal
//!    pre-marker text contains none of the forbidden identifiers — holds
//!    either way; this is a narrower, compiling realization of the same
//!    firewall, not a relaxation of it.
//! 2. **`inject` is a plain per-cohort writer, not fused with `reason`.**
//!    The build spec's §5 code fence for `inject` ends before the sentence
//!    "then `reason(&mut arena)`" — read literally that sentence is the
//!    CALLER's next step (matching §6: "inject per cohort ..., reason() on
//!    EVERY mind (CTRL included)" — one uniform reason() pass over every
//!    mind, injected or not). `inject` here therefore only calls
//!    `arena.observe` for the reserved belief family; the main test body
//!    calls `reason` once per mind afterward, uniformly.
//! 3. **B′'s "≥2 distinct Wittgenstein games" fallback is reimplemented
//!    locally**, not routed through `stance::stance_panel` (not in this
//!    file's import list per build spec §2). `distinct_games_local`
//!    reproduces the same games taxonomy `stance_panel`'s Wittgenstein arm
//!    uses (`inh-subj`/`inh-obj`/`impl-cause`/`impl-effect`; this file's
//!    `Mind` never produces rung-lifts, so the `rel-subj`/`rel-obj` games
//!    are structurally absent here — a narrower but consistent subset),
//!    over the SAME `arena`/`out` this file already built — no new arena,
//!    no double-count.
//!
//! ## Not compiled, not run by this lane — orchestrator gates
//!
//! This file was written edit-only (no `cargo` of any kind). Every
//! signature cited was read from source in the same pass that wrote this
//! file — see the build tag-file,
//! `.claude/board/exec-runs/d-blw-5-build-sonnet.md`, for what could and
//! could not be verified.

#[cfg(feature = "cycle-driver")]
mod d_blw_5_observer {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]

    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome, WORDS_PER_FP};
    use jc::stats::{binary_association, fisher_2z, BinaryAssociation};
    use lance_graph_contract::cognitive_shader::MetaWord;
    use lance_graph_contract::collapse_gate::MailboxId;
    use lance_graph_contract::kanban::{ExecTarget, KanbanColumn};
    use lance_graph_contract::mul::i4_eval::gate_decision_i4;
    use lance_graph_contract::qualia::QualiaI4_16D;
    use lance_graph_contract::scheduler::DatasetVersion;
    use lance_graph_contract::shape_rank::{
        RemeasureError, RemeasureKey, RemeasureLedger, ShapeRankPayload, SHAPE_BUCKETS,
    };
    use lance_graph_contract::soa_view::MailboxSoaView;
    use lance_graph_contract::thinking::ThinkingStyle;
    use lance_graph_planner::batch_writer::BatchWriter;
    use lance_graph_planner::ir::Arena;
    use lance_graph_planner::nars::stance::{stream, Interner, ReadOut};
    use lance_graph_planner::nars::tactics::{rcr_abduce, Throttle};
    use lance_graph_planner::nars::{BeliefArena, CStmt, Copula, Stamp, TruthValue};
    use lance_graph_planner::nested_bands::{quantize_2z, NestedBandsBuilder};
    use lance_graph_planner::owner_adapter::emit_bootstrap_intent;
    use lance_graph_planner::persist_sink::{
        CommitError, CommitOutcome, CycleFrame, CycleId, DetachedCycleBatch, FrameMeta, LandedSlot,
        SweepSlot, WalSink, WriteFailed,
    };
    use lance_graph_planner::strategy::style_strategy::StyleStrategy;
    use lance_graph_planner::traits::{
        PlanContext, PlanInput, PlanStrategy, QueryFeatures, StrategyOutcome,
    };
    use lance_graph_supervisor::cycle_driver::{
        run_cognitive_work_gated_over, run_cycle, shade_owner, CycleError, CycleOutcome,
    };

    // ── PRE-REGISTERED run shape (build spec §1-2) — fixed BEFORE any run. ──

    const FLEET_OWNERS: MailboxId = 40;
    const ROWS_PER_OWNER: usize = 64;
    const POPULATED_ROWS: usize = 48;
    const CORPUS_VERSES: usize = FLEET_OWNERS as usize * POPULATED_ROWS; // 1920
    const SCOPE_LO: MailboxId = 0;
    const SCOPE_HI: MailboxId = FLEET_OWNERS; // 0..40
    /// Backstop on `reason()`'s RCR+close fixed-point loop [pinned].
    const MAX_REASON_ROUNDS: usize = 16;

    const T_LO: MailboxId = 0;
    const T_HI: MailboxId = 8;
    const FP_LO: MailboxId = 8;
    const FP_HI: MailboxId = 16;
    const FM_LO: MailboxId = 16;
    const FM_HI: MailboxId = 24;
    const P_LO: MailboxId = 24;
    const P_HI: MailboxId = 32;
    const N_LO: MailboxId = 32;
    const N_HI: MailboxId = 36;
    const CTRL_LO: MailboxId = 36;
    const CTRL_HI: MailboxId = 40;

    const TENANT_THRESHOLD: f32 = 1.0;
    const FIRE_ENERGY: f32 = 2.0;
    const TENANT_W_SLOT: u8 = 0;
    const FIRING_ROWS: usize = 3;
    /// Every owner armed z=1 (Analytical) — build spec §2.
    const ARMED_Z: u8 = 1;

    /// Movement floor, pinned by the D-BLW-3 precedent doctrine cites in §7.
    const MOVEMENT_FLOOR: f64 = 0.10;
    /// The direction-test's own floor for `d = Δκ(FP) − Δκ(FM)` (§7 O5).
    const DIRECTION_FLOOR: f64 = 0.10;
    /// DROP threshold (§7).
    const DROP_FLOOR: f64 = 0.01;
    /// Reader B's confidence floor — [hand-tuned] per build spec §4: the
    /// injected chain reaches c ≈ m·0.9·c_ab·0.9 with c_ab = 0.81m/(0.81m+1);
    /// P's m = 1/16 gives 0.0024 (silent), m >= 0.13 gives >= 0.012 (fires).
    const C_MIN: f32 = 0.01;
    /// F+/F- rank eligibility band — [hand-tuned], the "never clipped, only
    /// excluded" boundary (build spec §5).
    const RANK_ELIGIBLE_LO: f32 = 0.05;
    const RANK_ELIGIBLE_HI: f32 = 0.95;
    /// F+/F- logit shift magnitude, opposite signs (build spec §5).
    const LOGIT_SHIFT: f32 = 1.5;
    /// The bucket-midpoint offset added to `rank_fraction()` before shifting
    /// in logit space (build spec §5).
    const BUCKET_MIDPOINT: f32 = 1.0 / 32.0;
    /// N's bloom-verdict percentile floor — the D-BLW-3 pin (build spec §6).
    const N_QUANTILE: f64 = 0.25;

    /// Permissive throttle (build spec §4) — `c_min=0`, unbounded budget,
    /// unbounded hub in-degree. `Throttle::new` is not `const fn`
    /// (`tactics.rs:126-135`), so this is a plain fn, not a `const`.
    fn throttle() -> Throttle {
        Throttle::new(0.0, 65_536, usize::MAX)
    }
    const MAX_PASSES: u32 = 64;

    type Tenant = MailboxSoA<ROWS_PER_OWNER>;
    type Fleet = HashMap<MailboxId, Tenant>;

    // ── ThinkingStyle dispatch input — copied shape from `d_ign_b_lenses.rs`
    // (provenance: `d_ign_b_lenses.rs:175-202`); every owner here is z=1. ──

    fn thinking_style_for(z: u8) -> ThinkingStyle {
        match z {
            1 => ThinkingStyle::Analytical,
            2 => ThinkingStyle::Creative,
            _ => ThinkingStyle::Reflective,
        }
    }

    fn style_vector_for(z: u8) -> Vec<f64> {
        let mut v = vec![0.0f64; 23];
        match thinking_style_for(z) {
            ThinkingStyle::Analytical => v[4] = 1.0,
            ThinkingStyle::Creative => v[3] = 1.0,
            _ => v[0] = 1.0,
        }
        v
    }

    fn plan_context_for(z: u8) -> PlanContext {
        PlanContext {
            query: String::new(),
            features: QueryFeatures::default(),
            free_will_modifier: 1.0,
            thinking_style: Some(style_vector_for(z)),
            nars_hint: None,
            witness: None,
        }
    }

    fn mantissa_of(owner: &Tenant) -> i8 {
        owner.pending_count().min(7) as i8
    }

    /// Provenance: `d_ign_b_lenses.rs:210-212` (itself re-derived from
    /// `cycle_driver.rs:1669`'s `#[cfg(test)]` fixture, not importable).
    fn flow_qualia() -> QualiaI4_16D {
        QualiaI4_16D(0).with(3, 4).with(14, 3).with(9, 4).with(1, 2)
    }

    // ── corpus + bloom-plane seeding — copied from `d_ign_b_lenses.rs`. ─────

    const BLOOM_K: usize = 4;

    fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325_u64 ^ seed.wrapping_mul(0x100_0000_01b3);
        for &c in bytes {
            h ^= u64::from(c);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    fn bloom_add(plane: &mut [u64], token: &str, salt: u64) {
        for k in 0..BLOOM_K {
            let h = fnv1a(
                token.as_bytes(),
                salt ^ (k as u64).wrapping_mul(0x9E37_79B9),
            );
            let bit = (h % (WORDS_PER_FP as u64 * 64)) as usize;
            plane[bit / 64] |= 1u64 << (bit % 64);
        }
    }

    fn tokens(text: &str) -> impl Iterator<Item = String> + '_ {
        text.split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|t| t.len() >= 2)
            .map(str::to_ascii_lowercase)
    }

    fn encode_plane(text: &str, salt: u64) -> Vec<u64> {
        let mut plane = vec![0u64; WORDS_PER_FP];
        for t in tokens(text) {
            bloom_add(&mut plane, &t, salt);
        }
        plane
    }

    /// Nonsense syllables the synthetic corpus builds its clause predicates
    /// from — copied from `d_ign_b_lenses.rs`'s `SYNTH_STEMS`/`synth_term`
    /// (provenance: `d_ign_b_lenses.rs:273-282`), which cites the reasons a
    /// `{stem}{window:02}{n:02}` shape collides with no catalogue this
    /// machine consults.
    const SYNTH_STEMS: [&str; 8] = ["vor", "lan", "tik", "mez", "qor", "sil", "dun", "fex"];

    fn synth_term(window: usize, n: usize) -> String {
        format!(
            "{}{:02}{:02}",
            SYNTH_STEMS[n % SYNTH_STEMS.len()],
            window,
            n
        )
    }

    /// `sub{w:02}{i:02}` — 7 chars, alphanumeric, no catalogued pronoun/verb
    /// morphology (build spec §3). A bare content word with no preceding
    /// pronoun becomes the clause subject (`stance.rs` "Bare content word:
    /// subject anchoring").
    fn subj(window: usize, i: usize) -> String {
        format!("sub{window:02}{i:02}")
    }

    /// One owner-slice's worth (`POPULATED_ROWS` verses) of the
    /// deterministic synthetic corpus (build spec §3). Copula "was" arms
    /// the predicate; "was not" negates.
    /// Per-window shape bits. Dry run 1 used `w % {2,3,4}`, which repeats every
    /// 12 windows: the 40 owners' phi values collapsed onto 12 atoms and the
    /// 16-bucket prior had empty buckets between them (T's pooled statistic
    /// landed in one: typicality 0 by construction). A splitmix64 fold of `w`
    /// gives every window its own counts — same ranges, no period.
    fn window_bits(w: usize) -> u64 {
        let mut z = (w as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn window(w: usize) -> Vec<String> {
        let bits = window_bits(w);
        let n_subj = 5 + (bits % 4) as usize; // 5..8
        let contra_n = 2 + ((bits >> 2) % 3) as usize; // 2..4
        let mut out: Vec<String> = Vec::with_capacity(POPULATED_ROWS);

        // shared: s0,s1 share T(0); s2,s3 share T(1) on even windows.
        out.push(format!("{} was {}.", subj(w, 0), synth_term(w, 0)));
        out.push(format!("{} was {}.", subj(w, 1), synth_term(w, 0)));
        if (bits >> 4) & 1 == 0 {
            out.push(format!("{} was {}.", subj(w, 2), synth_term(w, 1)));
            out.push(format!("{} was {}.", subj(w, 3), synth_term(w, 1)));
        }

        // own: disjoint `n` namespace 10+10*i+j, per subject.
        for i in 0..n_subj {
            let j_count = 2 + ((bits >> (8 + 2 * i)) % 3) as usize;
            for j in 0..j_count {
                out.push(format!(
                    "{} was {}.",
                    subj(w, i),
                    synth_term(w, 10 + 10 * i + j)
                ));
            }
        }

        // contra: affirm all, then negate all (later — revision -> contradiction 0.85).
        for i in 0..contra_n {
            out.push(format!("{} was {}.", subj(w, i), synth_term(w, 80 + i)));
        }
        for i in 0..contra_n {
            out.push(format!("{} was not {}.", subj(w, i), synth_term(w, 80 + i)));
        }

        assert!(
            out.len() <= POPULATED_ROWS,
            "window {w} over-filled: {} > {POPULATED_ROWS} (max is 4 + 8*4 + 8 = 44)",
            out.len()
        );
        let mut j = out.len();
        while out.len() < POPULATED_ROWS {
            out.push(format!(
                "{} was {}.",
                subj(w, j % n_subj),
                synth_term(w, 100 + j)
            ));
            j += 1;
        }
        out
    }

    fn synthetic_corpus() -> Vec<String> {
        (0..FLEET_OWNERS as usize).flat_map(window).collect()
    }

    /// Owner `owner_idx`'s own text slice — the same slice `build_owner`
    /// bloom-seeds from and `labelled_verses` reads (F1: one text source,
    /// never row-byte decoding). Provenance: `d_ign_b_lenses.rs:551-554`.
    fn owner_verses(all: &[String], owner_idx: MailboxId) -> &[String] {
        let lo = owner_idx as usize * POPULATED_ROWS;
        &all[lo..lo + POPULATED_ROWS]
    }

    /// `(label, text)` pairs `stance::stream` wants. Label format
    /// `"kjv:{global_index:05}"` (build spec's carried label convention).
    /// Provenance: `d_ign_b_lenses.rs:559-566`.
    fn labelled_verses(all: &[String], owner_idx: MailboxId) -> Vec<(String, String)> {
        let lo = owner_idx as usize * POPULATED_ROWS;
        all[lo..lo + POPULATED_ROWS]
            .iter()
            .enumerate()
            .map(|(i, text)| (format!("kjv:{:05}", lo + i), text.clone()))
            .collect()
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct RowSpanDescriptor {
        row_lo: u32,
        row_hi: u32,
        cycle: u32,
    }

    impl RowSpanDescriptor {
        fn to_le_bytes(self) -> [u8; 12] {
            let mut out = [0u8; 12];
            out[0..4].copy_from_slice(&self.row_lo.to_le_bytes());
            out[4..8].copy_from_slice(&self.row_hi.to_le_bytes());
            out[8..12].copy_from_slice(&self.cycle.to_le_bytes());
            out
        }
    }

    fn row_span_payload(owner: &Tenant) -> Vec<u8> {
        RowSpanDescriptor {
            row_lo: 0,
            row_hi: owner.populated() as u32,
            cycle: owner.cycle(),
        }
        .to_le_bytes()
        .to_vec()
    }

    // ── the WAL seam (in-process; NOT durability) — copied from
    // `d_ign_b_lenses.rs`'s `MemWal`. ────────────────────────────────────────

    struct SealedCycle {
        frame: CycleFrame,
        version: DatasetVersion,
        batch_hash: u64,
        landings: Vec<SweepSlot>,
    }

    struct MemWal {
        sealed: Mutex<Vec<SealedCycle>>,
        next_version: AtomicU64,
        #[allow(dead_code)]
        wal_writes: AtomicU64,
    }

    impl MemWal {
        fn new() -> Self {
            Self {
                sealed: Mutex::new(Vec::new()),
                next_version: AtomicU64::new(1),
                wal_writes: AtomicU64::new(0),
            }
        }
        fn head(&self) -> DatasetVersion {
            self.sealed
                .lock()
                .expect("MemWal poisoned")
                .last()
                .map_or(DatasetVersion(0), |s| s.version)
        }
    }

    impl WalSink for MemWal {
        async fn commit_cycle(
            &mut self,
            batch: DetachedCycleBatch,
        ) -> Result<CommitOutcome, CommitError> {
            let mut sealed = self.sealed.lock().expect("MemWal poisoned");
            let head = sealed.last().map_or(DatasetVersion(0), |s| s.version);
            if let Some(rec) = sealed.iter().find(|s| s.frame.cycle == batch.frame.cycle) {
                return if rec.batch_hash == batch.batch_hash {
                    Ok(CommitOutcome::Reconciled {
                        current_head: head,
                        cycle: batch.frame.cycle,
                        batch_hash: batch.batch_hash,
                    })
                } else {
                    Err(CommitError::HashConflict {
                        cycle: batch.frame.cycle,
                        stored_hash: rec.batch_hash,
                        offered_hash: batch.batch_hash,
                    })
                };
            }
            if batch.frame.base_version != head {
                return Err(CommitError::Fenced { current_head: head });
            }
            self.wal_writes.fetch_add(1, Ordering::SeqCst);
            let version = DatasetVersion(self.next_version.fetch_add(1, Ordering::SeqCst));
            let (cycle, batch_hash) = (batch.frame.cycle, batch.batch_hash);
            sealed.push(SealedCycle {
                frame: batch.frame,
                version,
                batch_hash,
                landings: batch.landings,
            });
            Ok(CommitOutcome::Committed {
                version,
                cycle,
                batch_hash,
            })
        }

        async fn scan_sealed(
            &self,
            after_cycle: Option<CycleId>,
        ) -> Result<Vec<LandedSlot>, WriteFailed> {
            Ok(self
                .sealed
                .lock()
                .expect("MemWal poisoned")
                .iter()
                .filter(|s| after_cycle.is_none_or(|c| s.frame.cycle > c))
                .flat_map(|s| {
                    s.landings.iter().map(|slot| LandedSlot {
                        cycle: s.frame.cycle,
                        slot: SweepSlot {
                            payload: Vec::new(),
                            ..slot.clone()
                        },
                    })
                })
                .collect())
        }

        async fn timeline(&self) -> Result<Vec<FrameMeta>, WriteFailed> {
            Ok(self
                .sealed
                .lock()
                .expect("MemWal poisoned")
                .iter()
                .map(|s| FrameMeta {
                    cycle: s.frame.cycle,
                    base_version: s.frame.base_version,
                    batch_hash: s.batch_hash,
                })
                .collect())
        }
    }

    // ── fleet construction ──────────────────────────────────────────────────

    fn build_owner(
        id: MailboxId,
        verses: &[String],
        content_salt: u64,
        armed: u8,
        qualia: QualiaI4_16D,
        firing_rows: usize,
    ) -> Tenant {
        let mut owner: Tenant = MailboxSoA::new(id, TENANT_W_SLOT, TENANT_THRESHOLD);
        let cycle = owner.cycle();
        let meta = MetaWord::new(armed, 0, 0, 0, 0);
        for (row, text) in verses.iter().enumerate() {
            let content = encode_plane(text, content_salt);
            let cell = WriteCell {
                content: Some(content.as_slice()),
                qualia: Some(qualia),
                meta: Some(meta),
                entity_type: Some((row % 251) as u16),
                temporal: Some(row as u64),
                ..WriteCell::default()
            };
            let outcome = owner.write_row(row, cycle, &cell);
            assert_eq!(
                outcome,
                WriteOutcome::Accepted,
                "seeding row {row} of owner {id} must be accepted"
            );
        }
        owner.set_populated(verses.len());
        owner.tick();
        for r in 0..firing_rows {
            owner.energy[r] = FIRE_ENERGY;
        }
        owner
    }

    fn build_fleet(corpus: &[String]) -> Fleet {
        let mut fleet = Fleet::new();
        for id in SCOPE_LO..SCOPE_HI {
            fleet.insert(
                id,
                build_owner(
                    id,
                    owner_verses(corpus, id),
                    u64::from(id),
                    ARMED_Z,
                    flow_qualia(),
                    FIRING_ROWS,
                ),
            );
        }
        fleet
    }

    // ── the scan — copied from `d_ign_b_lenses.rs`. ─────────────────────────

    #[derive(Default)]
    struct ScanResult {
        planning: Vec<MailboxId>,
        cognitive: Vec<MailboxId>,
        evaluation: Vec<MailboxId>,
        absorbed: Vec<MailboxId>,
        missing: usize,
    }

    fn scan_board(fleet: &Fleet, ids: impl IntoIterator<Item = MailboxId>) -> ScanResult {
        let mut r = ScanResult::default();
        for id in ids {
            match fleet.get(&id) {
                None => r.missing += 1,
                Some(owner) => match owner.phase() {
                    KanbanColumn::Planning => r.planning.push(id),
                    KanbanColumn::CognitiveWork => r.cognitive.push(id),
                    KanbanColumn::Evaluation => r.evaluation.push(id),
                    KanbanColumn::Commit | KanbanColumn::Plan | KanbanColumn::Prune => {
                        r.absorbed.push(id);
                    }
                },
            }
        }
        r
    }

    struct ColumnPassOutcome {
        cast: usize,
    }

    fn column_pass(
        fleet: &Fleet,
        ids: &[MailboxId],
        writer: &mut BatchWriter<Vec<u8>>,
        mut think: impl FnMut(&Tenant) -> Option<(StrategyOutcome, Vec<u8>)>,
    ) -> ColumnPassOutcome {
        let mut cast = 0usize;
        for &id in ids {
            let Some(owner) = fleet.get(&id) else {
                continue;
            };
            if let Some((outcome, payload)) = think(owner) {
                if emit_bootstrap_intent(
                    &outcome,
                    owner.mailbox_id(),
                    owner.current_cycle(),
                    writer,
                    payload,
                )
                .is_some()
                {
                    cast += 1;
                }
            }
        }
        ColumnPassOutcome { cast }
    }

    /// Provenance: `d_ign_b_lenses.rs:738-772`, unchanged.
    fn plan_or_evaluate_think(owner: &Tenant) -> Option<(StrategyOutcome, Vec<u8>)> {
        let armed = owner.meta_at(0).thinking();
        if armed == 0 {
            return None;
        }
        let qualia = owner.qualia_at(0);
        let mantissa = mantissa_of(owner);
        let gate = gate_decision_i4(&qualia, mantissa);
        let target = owner.phase().advance_on_gate(&gate)?;

        let mut arena = Arena::new();
        let plan_out = StyleStrategy
            .plan(
                PlanInput {
                    plan: None,
                    context: plan_context_for(armed),
                    outcome: None,
                },
                &mut arena,
            )
            .expect("StyleStrategy::plan never errors over this probe's fixed recipe substrate");
        let style_outcome = plan_out
            .outcome
            .expect("StyleStrategy always surfaces a StrategyOutcome");
        let style_move = style_outcome
            .intended_move
            .expect("StyleStrategy always intends the Planning -> CognitiveWork crossing");

        let outcome = if style_move.to == target {
            style_outcome
        } else {
            shade_owner(owner, &qualia, mantissa, style_outcome.reliability)?
        };
        Some((outcome, row_span_payload(owner)))
    }

    // ── the owner's mind, the reasoning pass, injection (build spec §4-5). ──

    struct Mind {
        arena: BeliefArena,
        intern: Interner,
        out: ReadOut,
        /// Distinct `p.stmt.s` over `out.provenance`, sorted.
        subjects: Vec<u16>,
        /// Empty until injection; the reserved-prior/band terms afterward.
        reserved: HashSet<u16>,
    }

    /// `reason` is the ONLY propagation channel and runs at BOTH versions
    /// (V0 pre-injection, V1 post-injection) for EVERY owner including
    /// CTRL — so V1-V0 on CTRL measures pass idempotence, not injection
    /// (build spec §4).
    /// Bounded RCR + closure to a FIXED POINT. Dry run 1 (2026-09-05, recorded in
    /// the board entry) showed a single RCR+close round is NOT idempotent: the
    /// derived layer feeds RCR new premises, so a second `reason()` on the
    /// un-injected CTRL cohort moved reader B (Δκ(CTRL) = −0.0028). A pass that
    /// runs at both V0 and V1 must be a fixed point, or V1−V0 measures the pass.
    fn reason(arena: &mut BeliefArena) {
        let throttle = throttle();
        for _ in 0..MAX_REASON_ROUNDS {
            let frontier = rcr_abduce(arena, &throttle);
            let mut admitted = 0usize;
            for c in frontier.candidates {
                if arena.admit_derived(c.stmt, c.truth, &c.premises, c.rung) {
                    admitted += 1;
                }
            }
            arena.close_transitive(MAX_PASSES);
            if admitted == 0 {
                return;
            }
        }
    }

    fn build_mind(verses: &[(String, String)]) -> Mind {
        let mut arena = BeliefArena::new();
        let mut intern = Interner::new();
        let mut out = ReadOut::default();
        stream(verses, &mut arena, &mut intern, &mut out, false);
        reason(&mut arena);
        let mut subjects: Vec<u16> = out.provenance.iter().map(|p| p.stmt.s).collect();
        subjects.sort_unstable();
        subjects.dedup();
        Mind {
            arena,
            intern,
            out,
            subjects,
            reserved: HashSet::new(),
        }
    }

    /// Injects the reserved belief family described by `payload` into
    /// `mind`'s arena (build spec §5): 16 bucket-beliefs (`prior Inh
    /// band_k`, truth = the shape's mass fraction) + one typicality belief
    /// per subject (`subject Inh prior`, truth = the prior's own mass
    /// fraction AT the observed rank — C2's typicality encoding). Does NOT
    /// call `reason` itself — see module-doc deviation 2; the caller runs
    /// one uniform `reason` pass over every mind afterward.
    fn inject(mind: &mut Mind, payload: &ShapeRankPayload) {
        // Stamp::source folds `id % 64`; 63 is reserved for this probe's
        // injected family, so injected statements are always NEW admissions
        // (Admitted, not routed through the S4 overlap guard) regardless of
        // which observation-source ids the corpus already used.
        let reserved_stamp = Stamp::source(63);
        let prior = mind.intern.id("blw5:prior");
        let mut bands = Vec::with_capacity(SHAPE_BUCKETS);
        for k in 0..SHAPE_BUCKETS {
            bands.push(mind.intern.id(&format!("blw5:band:{k:02}")));
        }
        let mass = payload.mass() as f32;
        for (k, &band_id) in bands.iter().enumerate() {
            let stmt = CStmt {
                s: prior,
                cop: Copula::Inh,
                p: band_id,
            };
            let f = payload.shape[k] as f32 / mass;
            mind.arena
                .observe(stmt, TruthValue::new(f, 0.9), reserved_stamp);
        }
        let typ = payload.shape[payload.rank as usize] as f32 / mass;
        // C2 AMENDED (dry run 2, 2026-09-05, recorded in the board entry):
        // typicality rides in the CONFIDENCE of `subject Inh prior`, at
        // frequency 1. The spec's first encoding (frequency = typicality,
        // c = 0.9) put f below 0.5, which NARS reads as a confident NEGATION;
        // `admit_derived`'s expectation-CHOICE then preferred a vacuous
        // closure path (c ≈ 3.5e-11, expectation ≈ 0.5) over the confident
        // negative (expectation 0.462) — the arena discarded the payload in
        // favour of ignorance. With f = 1 the expectation is monotone in c
        // and the injected link survives CHOICE. C_MIN and every floor are
        // unchanged.
        let subjects = mind.subjects.clone();
        for s in subjects {
            let stmt = CStmt {
                s,
                cop: Copula::Inh,
                p: prior,
            };
            mind.arena
                .observe(stmt, TruthValue::new(1.0, typ), reserved_stamp);
        }
        mind.reserved.insert(prior);
        mind.reserved.extend(bands);
    }

    /// One cycle's mechanics: scan the board over `SCOPE_LO..SCOPE_HI`,
    /// plan/cognitive/evaluate columns, seal via `run_cycle`. Mirrors
    /// `d_ign_b_lenses.rs`'s main loop body (mechanics unchanged), minus the
    /// lens capture (this probe reads Mind arenas directly, never through
    /// the SoA rows — the SoA loop only supplies the sealed version, per
    /// build spec §6). Defined BEFORE the O6 marker: contains none of
    /// `"fn reader_a"`, `"fn reader_b"`, `"binary_association("`, `"kappa"` (the O6 list).
    async fn run_cycle_mechanics(
        cycle: u32,
        fleet: &mut Fleet,
        sink: &mut MemWal,
        writer: &mut BatchWriter<Vec<u8>>,
        position_base: &mut u64,
        watermarks: &mut HashMap<MailboxId, Option<u64>>,
    ) -> CycleOutcome {
        let scan = scan_board(fleet, SCOPE_LO..SCOPE_HI);
        eprintln!(
            "d_blw_5 scan @c{cycle}: planning={} cognitive={} evaluation={} absorbed={} missing={}",
            scan.planning.len(),
            scan.cognitive.len(),
            scan.evaluation.len(),
            scan.absorbed.len(),
            scan.missing
        );
        assert_eq!(scan.missing, 0, "every owner in SCOPE is inserted");

        let planning_outcome = column_pass(fleet, &scan.planning, writer, plan_or_evaluate_think);
        let cognitive_outcome =
            run_cognitive_work_gated_over(fleet, &scan.cognitive, writer, |owner| {
                let armed = owner.meta_at(0).thinking();
                if armed == 0 {
                    return None;
                }
                let style = thinking_style_for(armed);
                let ctx = plan_context_for(armed);
                let qualia = owner.qualia_at(0);
                let mantissa = mantissa_of(owner);
                let reliability = StyleStrategy::reliability_for(style, &ctx);
                Some((qualia, mantissa, reliability, row_span_payload(owner)))
            });
        let evaluation_outcome =
            column_pass(fleet, &scan.evaluation, writer, plan_or_evaluate_think);

        let total_casts = planning_outcome.cast + cognitive_outcome.cast + evaluation_outcome.cast;
        eprintln!("d_blw_5 @c{cycle}: {total_casts} casts staged");

        let base_version = sink.head();
        let outcome = match run_cycle(
            sink,
            fleet,
            writer,
            CycleFrame::new(CycleId(u64::from(cycle)), base_version),
            *position_base,
            watermarks,
            u64::from,
        )
        .await
        {
            Ok(o) => o,
            Err(CycleError::Seal(_)) => {
                panic!("d_blw_5 @c{cycle}: unexpected seal failure (MemWal never injects one)")
            }
            Err(CycleError::Apply { cause, .. }) => {
                panic!("d_blw_5 @c{cycle}: unexpected apply failure: {cause}")
            }
        };
        *position_base = (*position_base).max(outcome.sealed.next_position_base);
        for t in &outcome.sealed.transitions {
            if t.mv.from == KanbanColumn::Planning && t.mv.to == KanbanColumn::CognitiveWork {
                assert_eq!(
                    t.mv.exec,
                    ExecTarget::Elixir,
                    "Planning->CognitiveWork must be the STYLE's Elixir mint"
                );
            }
        }
        outcome
    }

    // ── N's awareness-free bloom criterion (build spec §6, cites
    // `blw_fusion.rs:314-375`) — defined BEFORE the marker: uses none of
    // the forbidden identifiers. ─────────────────────────────────────────

    fn score_row(owner: &Tenant, row: usize, seed: &[u64]) -> u32 {
        owner
            .content_row(row)
            .iter()
            .zip(seed)
            .map(|(w, s)| (w & s).count_ones())
            .sum()
    }

    /// verdict[row] = true iff `score_row` is in the top `1 - N_QUANTILE`
    /// (the D-BLW-3 `q=0.25` pin: top quartile) of the owner's
    /// `POPULATED_ROWS` rows, ties broken by ascending row index.
    fn bloom_verdicts(owner: &Tenant, seed: &[u64]) -> Vec<bool> {
        let mut scored: Vec<(u32, usize)> = (0..POPULATED_ROWS)
            .map(|row| (score_row(owner, row, seed), row))
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let n_pos = (POPULATED_ROWS as f64 * N_QUANTILE) as usize;
        let mut verdict = vec![false; POPULATED_ROWS];
        for &(_, row) in scored.iter().take(n_pos) {
            verdict[row] = true;
        }
        verdict
    }

    // ── MEASUREMENT BLOCK (O6 marker) ──

    // ── readers A/B/B_shadow (build spec §4, C1's correction). ──────────────

    /// A — evidence, injection-invariant by construction. `∃ p ∈
    /// out.provenance: p.verse == v && arena.get(p.stmt).contradiction > 0.05`.
    fn reader_a(mind: &Mind, verse: &str) -> bool {
        mind.out.provenance.iter().any(|p| {
            p.verse == verse
                && mind
                    .arena
                    .get(p.stmt)
                    .is_some_and(|b| b.contradiction > 0.05)
        })
    }

    /// The verse's FIRST provenance statement's `(s, p)`, if any.
    fn first_stmt_for_verse(mind: &Mind, verse: &str) -> Option<CStmt> {
        mind.out
            .provenance
            .iter()
            .find(|p| p.verse == verse)
            .map(|p| p.stmt)
    }

    /// B — awareness-coupled "inferential corroboration" (C1's DERIVED-layer
    /// pin): a rung>=1, default-stamp (derived) belief `b'.s Inh p` with
    /// `b'.s` a distinct known subject, EXCLUDING the reserved injected
    /// terms, at or above `C_MIN` confidence.
    fn reader_b(mind: &Mind, verse: &str) -> bool {
        let Some(stmt) = first_stmt_for_verse(mind, verse) else {
            return false;
        };
        mind.arena.entries().iter().any(|b| {
            b.stmt.cop == Copula::Inh
                && b.stmt.p == stmt.p
                && b.stmt.s != stmt.s
                && mind.subjects.contains(&b.stmt.s)
                && !mind.reserved.contains(&b.stmt.s)
                && b.stamp == Stamp::default()
                && b.rung >= 1
                && b.truth.confidence >= C_MIN
        })
    }

    /// Deviation 3: reproduces `stance_panel`'s Wittgenstein games taxonomy
    /// (`stance.rs:512-529`) locally over this file's own `arena`/`out`,
    /// restricted to the two games this `Mind` can ever produce
    /// (`inh-subj`/`inh-obj`; `impl-cause`/`impl-effect` never fire — this
    /// corpus emits no `because` cue). `rel-*` (rung-1 lifts) are absent by
    /// construction: `stream` never sees an epistemic verb here.
    fn distinct_games_local(mind: &Mind, subject: u16) -> usize {
        let mut games: HashSet<&'static str> = HashSet::new();
        for b in mind.arena.entries() {
            if b.stmt.cop == Copula::Inh && b.stamp != Stamp::default() {
                if b.stmt.s == subject {
                    games.insert("inh-subj");
                }
                if b.stmt.p == subject {
                    games.insert("inh-obj");
                }
            }
        }
        for (_, cause, effect) in &mind.out.impls {
            if *cause == subject {
                games.insert("impl-cause");
            }
            if *effect == subject {
                games.insert("impl-effect");
            }
        }
        games.len()
    }

    /// B′ — pre-registered FALLBACK, decided once at V0 on CTRL only, never
    /// chosen after seeing T/FP/FM/P/N output.
    fn reader_b_fallback(mind: &Mind, verse: &str) -> bool {
        let Some(stmt) = first_stmt_for_verse(mind, verse) else {
            return false;
        };
        distinct_games_local(mind, stmt.s) >= 2
    }

    /// Runs (reader, verse) over every verse this owner's `Mind` was built
    /// from, skipping verses with no provenance in BOTH vectors (printing
    /// the skip count), yielding the two boolean vectors for
    /// `binary_association`.
    fn reader_vectors(
        mind: &Mind,
        verses: &[(String, String)],
        use_fallback: bool,
    ) -> (Vec<bool>, Vec<bool>, usize) {
        let mut a = Vec::with_capacity(verses.len());
        let mut b = Vec::with_capacity(verses.len());
        let mut skipped = 0usize;
        for (label, _) in verses {
            if first_stmt_for_verse(mind, label).is_none() {
                skipped += 1;
                continue;
            }
            a.push(reader_a(mind, label));
            b.push(if use_fallback {
                reader_b_fallback(mind, label)
            } else {
                reader_b(mind, label)
            });
        }
        (a, b, skipped)
    }

    /// `S(cohort, version)` — pools every owner's reader vectors (owner
    /// order ascending) and computes `binary_association`. `corpus` supplies
    /// each owner's own labelled verses.
    fn measure_cohort(
        minds: &HashMap<MailboxId, Mind>,
        corpus: &[String],
        ids: std::ops::Range<MailboxId>,
        use_fallback: bool,
    ) -> Option<BinaryAssociation> {
        let mut a_pool = Vec::new();
        let mut b_pool = Vec::new();
        let mut skipped_total = 0usize;
        for id in ids {
            let mind = minds
                .get(&id)
                .expect("mind must exist for every scoped owner");
            let verses = labelled_verses(corpus, id);
            let (a, b, skipped) = reader_vectors(mind, &verses, use_fallback);
            skipped_total += skipped;
            a_pool.extend(a);
            b_pool.extend(b);
        }
        eprintln!(
            "d_blw_5.measure_cohort: {} verses skipped (no provenance) across the cohort",
            skipped_total
        );
        binary_association(&a_pool, &b_pool)
    }

    // ── payload construction (build spec §5). ────────────────────────────

    struct ArmPayloads {
        t: Option<ShapeRankPayload>,
        fp: Option<ShapeRankPayload>,
        fm: Option<ShapeRankPayload>,
        p: Option<ShapeRankPayload>,
        n: Option<ShapeRankPayload>,
    }

    #[allow(clippy::too_many_lines)]
    fn build_payloads(
        minds: &HashMap<MailboxId, Mind>,
        corpus: &[String],
        v0: DatasetVersion,
        ledger: &mut RemeasureLedger,
        use_fallback: bool,
    ) -> ArmPayloads {
        // Pool: phi_owner over ALL 40 owners (each owner as a cohort-of-one).
        let mut pool: Vec<i32> = Vec::with_capacity(SCOPE_HI as usize);
        for id in SCOPE_LO..SCOPE_HI {
            if let Some(assoc) = measure_cohort(minds, corpus, id..id + 1, use_fallback) {
                if let Some(phi) = assoc.phi {
                    pool.push(quantize_2z(fisher_2z(phi)));
                }
            }
        }
        assert!(
            pool.len() >= 16,
            "PRECONDITION: prior pool too thin ({} < 16) — a corpus defect, not a finding",
            pool.len()
        );
        let nb = NestedBandsBuilder::new(SHAPE_BUCKETS).calibrate_equal_width(&pool, v0.0);
        eprintln!(
            "d_blw_5.pool: {} owner phi values pooled; boundaries={:?}",
            pool.len(),
            nb.boundaries()
        );

        let mut cohort = |lo: MailboxId,
                          hi: MailboxId,
                          arm: u8,
                          label: &str|
         -> Option<ShapeRankPayload> {
            let assoc = measure_cohort(minds, corpus, lo..hi, use_fallback)?;
            let phi = assoc.phi?;
            let obs = quantize_2z(fisher_2z(phi));
            let payload_true = nb.shape_rank(obs, v0.0);
            eprintln!(
                "d_blw_5.arm {label}: phi={phi:.4} obs={obs} shape={:?} rank={} prozentrang={:.4} version={}",
                payload_true.shape, payload_true.rank, payload_true.prozentrang(), payload_true.version
            );
            let key = RemeasureKey {
                stat_id: 1,
                arm,
                cohort: lo,
                metric: 1,
                dataset_version: v0.0,
            };
            match ledger.seal(key, payload_true) {
                Ok(()) => {}
                Err(
                    RemeasureError::AlreadySealed { .. } | RemeasureError::VersionMismatch { .. },
                ) => {
                    panic!("d_blw_5: unexpected ledger seal failure for arm {label} at V0")
                }
            }
            Some(payload_true)
        };

        let t = cohort(T_LO, T_HI, 1, "T");
        let fp_true = cohort(FP_LO, FP_HI, 2, "FP");
        let fm_true = cohort(FM_LO, FM_HI, 3, "FM");
        let n = cohort(N_LO, N_HI, 5, "N");
        let p = {
            // P: uniform shape over the SAME pool size, rank = 8 (median).
            let mass = pool.len() as u64;
            let q = mass / SHAPE_BUCKETS as u64;
            let r = mass % SHAPE_BUCKETS as u64;
            let mut shape = [0u64; SHAPE_BUCKETS];
            for (k, slot) in shape.iter_mut().enumerate() {
                *slot = if (k as u64) < r { q + 1 } else { q };
            }
            let payload = ShapeRankPayload::new(shape, 8, v0.0);
            let key = RemeasureKey {
                stat_id: 1,
                arm: 4,
                cohort: P_LO,
                metric: 1,
                dataset_version: v0.0,
            };
            ledger
                .seal(key, payload)
                .expect("P's ledger seal must succeed at V0 (never sealed before)");
            eprintln!("d_blw_5.arm P: uniform shape={shape:?} rank=8 (zero-information envelope)");
            Some(payload)
        };
        // F+/F- — shift the TRUE arm's own shape's rank in logit(rank_fraction) space.
        let shift = |payload_true: Option<ShapeRankPayload>,
                     sign: f32,
                     label: &str|
         -> Option<ShapeRankPayload> {
            let payload_true = payload_true?;
            let rf = payload_true.rank_fraction() + BUCKET_MIDPOINT;
            if !(RANK_ELIGIBLE_LO..=RANK_ELIGIBLE_HI).contains(&rf) {
                eprintln!(
                    "d_blw_5.arm {label}: EXCLUDED — rank_fraction {rf} outside eligibility band [{RANK_ELIGIBLE_LO}, {RANK_ELIGIBLE_HI}], never clipped"
                );
                return None;
            }
            let l = (rf / (1.0 - rf)).ln() + sign * LOGIT_SHIFT;
            let rf2 = 1.0 / (1.0 + (-l).exp());
            let rank2 = ((rf2 * SHAPE_BUCKETS as f32).floor() as u8).min((SHAPE_BUCKETS - 1) as u8);
            let payload = ShapeRankPayload::new(payload_true.shape, rank2, v0.0);
            eprintln!(
                "d_blw_5.arm {label}: rf={rf:.4} shifted rank={rank2} (true rank={})",
                payload_true.rank
            );
            Some(payload)
        };
        let fp = shift(fp_true, 1.0, "FP");
        let fm = shift(fm_true, -1.0, "FM");
        if let Some(payload) = fp {
            let key = RemeasureKey {
                stat_id: 1,
                arm: 2,
                cohort: FP_LO,
                metric: 1,
                dataset_version: v0.0,
            };
            let _ = ledger.seal(key, payload); // may double-seal with the true-shape arm above; informational only
        }
        if let Some(payload) = fm {
            let key = RemeasureKey {
                stat_id: 1,
                arm: 3,
                cohort: FM_LO,
                metric: 1,
                dataset_version: v0.0,
            };
            let _ = ledger.seal(key, payload);
        }

        ArmPayloads { t, fp, fm, p, n }
    }

    // ── O4/O5 delta helpers. ─────────────────────────────────────────────

    fn delta_kappa(s0: Option<BinaryAssociation>, s1: Option<BinaryAssociation>) -> Option<f64> {
        match (s0.and_then(|a| a.kappa), s1.and_then(|a| a.kappa)) {
            (Some(k0), Some(k1)) => Some(k1 - k0),
            _ => None,
        }
    }

    // ── the main probe ───────────────────────────────────────────────────

    #[tokio::test]
    async fn d_blw_5_observer_effect_belief_arena() {
        let corpus = synthetic_corpus();
        assert_eq!(corpus.len(), CORPUS_VERSES, "PRE-REGISTERED corpus size");
        println!(
            "d_blw_5 corpus: deterministic synthetic ({} verses)",
            corpus.len()
        );

        let mut fleet = build_fleet(&corpus);
        let mut minds: HashMap<MailboxId, Mind> = HashMap::new();
        for id in SCOPE_LO..SCOPE_HI {
            minds.insert(id, build_mind(&labelled_verses(&corpus, id)));
        }

        let mut sink = MemWal::new();
        let mut writer: BatchWriter<Vec<u8>> = BatchWriter::new();
        let mut position_base: u64 = 0;
        let mut watermarks: HashMap<MailboxId, Option<u64>> = HashMap::new();

        // ── c=1: cast/scan/seal -> V0. ───────────────────────────────────
        let outcome_c1 = run_cycle_mechanics(
            1,
            &mut fleet,
            &mut sink,
            &mut writer,
            &mut position_base,
            &mut watermarks,
        )
        .await;
        let v0 = sink.head();
        eprintln!("d_blw_5: V0 sealed = {:?}", v0);
        let _ = outcome_c1;

        // ── Preconditions on CTRL at V0 (before deciding the B/B' fallback). ──
        let ctrl_b_rate_v0;
        {
            let mut a_all = Vec::new();
            let mut b_all = Vec::new();
            for id in CTRL_LO..CTRL_HI {
                let mind = minds.get(&id).expect("CTRL mind must exist");
                let verses = labelled_verses(&corpus, id);
                let (a, b, _) = reader_vectors(mind, &verses, false);
                a_all.extend(a);
                b_all.extend(b);
            }
            let a_rate = a_all.iter().filter(|x| **x).count() as f64 / a_all.len().max(1) as f64;
            let b_rate = b_all.iter().filter(|x| **x).count() as f64 / b_all.len().max(1) as f64;
            ctrl_b_rate_v0 = b_rate;
            eprintln!("d_blw_5.precondition: CTRL A-rate={a_rate:.4} B-rate={b_rate:.4} at V0");
            assert!(
                a_rate > 0.0 && a_rate < 1.0,
                "PRECONDITION: reader A is degenerate on CTRL at V0 (corpus defect, not a finding)"
            );
        }
        let use_fallback = !(ctrl_b_rate_v0 > 0.0 && ctrl_b_rate_v0 < 1.0);
        if use_fallback {
            eprintln!(
                "d_blw_5: reader B degenerate on CTRL at V0 (rate={ctrl_b_rate_v0:.4}) — falling back to B' (Wittgenstein-games) for ALL cohorts at both versions, pre-registered per build spec §4"
            );
        }

        let mut ledger = RemeasureLedger::new();
        let payloads_v0 = build_payloads(&minds, &corpus, v0, &mut ledger, use_fallback);

        // S(cohort, V0) for all six cohorts.
        let s0_t = measure_cohort(&minds, &corpus, T_LO..T_HI, use_fallback);
        let s0_fp = measure_cohort(&minds, &corpus, FP_LO..FP_HI, use_fallback);
        let s0_fm = measure_cohort(&minds, &corpus, FM_LO..FM_HI, use_fallback);
        let s0_p = measure_cohort(&minds, &corpus, P_LO..P_HI, use_fallback);
        let s0_n = measure_cohort(&minds, &corpus, N_LO..N_HI, use_fallback);
        let s0_ctrl = measure_cohort(&minds, &corpus, CTRL_LO..CTRL_HI, use_fallback);
        for (label, assoc) in [
            ("T", s0_t),
            ("FP", s0_fp),
            ("FM", s0_fm),
            ("P", s0_p),
            ("N", s0_n),
            ("CTRL", s0_ctrl),
        ] {
            print_association_table(&format!("S({label}, V0)"), assoc);
        }

        let n_bloom_v0: HashMap<MailboxId, Vec<bool>> = (N_LO..N_HI)
            .map(|id| {
                let owner = fleet.get(&id).expect("N owner must exist");
                let verse0 = &owner_verses(&corpus, id)[0];
                let seed = encode_plane(verse0, u64::from(id));
                (id, bloom_verdicts(owner, &seed))
            })
            .collect();

        // ── O1 can-fire: re-seal at T's V0 key must ERROR. ───────────────
        {
            let t_payload = payloads_v0
                .t
                .expect("T must be non-degenerate for O1's can-fire");
            let key = RemeasureKey {
                stat_id: 1,
                arm: 1,
                cohort: T_LO,
                metric: 1,
                dataset_version: v0.0,
            };
            let err = ledger.seal(key, t_payload).unwrap_err();
            assert!(
                matches!(err, RemeasureError::AlreadySealed { .. }),
                "O1 can-fire: a second seal at T's sealed V0 key must ERROR, got {err:?}"
            );
            // can-stay-silent: a version-bumped payload at the same shape/rank
            // passes (a fresh (id, scope, V+1) one-shot).
            let v1_payload = ShapeRankPayload::new(t_payload.shape, t_payload.rank, v0.0 + 1);
            let key_v1 = RemeasureKey {
                stat_id: 1,
                arm: 1,
                cohort: T_LO,
                metric: 1,
                dataset_version: v0.0 + 1,
            };
            assert!(
                ledger.seal(key_v1, v1_payload).is_ok(),
                "O1 can-stay-silent: a fresh (id, scope, V+1) one-shot must pass"
            );
            let key_arm2 = RemeasureKey {
                stat_id: 1,
                arm: 2,
                cohort: T_LO,
                metric: 1,
                dataset_version: v0.0,
            };
            let arm2_payload = ShapeRankPayload::new(t_payload.shape, t_payload.rank, v0.0);
            assert!(
                ledger.seal(key_arm2, arm2_payload).is_ok(),
                "O1 can-stay-silent: a different arm at the same (id, version) must pass"
            );
        }

        // ── inject per cohort, then ONE uniform reason() pass over every
        // mind (CTRL included). ──────────────────────────────────────────
        if let Some(p) = payloads_v0.t {
            for id in T_LO..T_HI {
                inject(minds.get_mut(&id).expect("T mind"), &p);
            }
        }
        if let Some(p) = payloads_v0.fp {
            for id in FP_LO..FP_HI {
                inject(minds.get_mut(&id).expect("FP mind"), &p);
            }
        }
        if let Some(p) = payloads_v0.fm {
            for id in FM_LO..FM_HI {
                inject(minds.get_mut(&id).expect("FM mind"), &p);
            }
        }
        if let Some(p) = payloads_v0.p {
            for id in P_LO..P_HI {
                inject(minds.get_mut(&id).expect("P mind"), &p);
            }
        }
        if let Some(p) = payloads_v0.n {
            for id in N_LO..N_HI {
                inject(minds.get_mut(&id).expect("N mind"), &p);
            }
        }
        // CTRL: no injection, but `reason` still runs — pass idempotence.
        for mind in minds.values_mut() {
            reason(&mut mind.arena);
        }

        // ── c=2: cast/scan/seal -> V1. ────────────────────────────────────
        let outcome_c2 = run_cycle_mechanics(
            2,
            &mut fleet,
            &mut sink,
            &mut writer,
            &mut position_base,
            &mut watermarks,
        )
        .await;
        let v1 = sink.head();
        eprintln!("d_blw_5: V1 sealed = {:?}", v1);
        let _ = outcome_c2;

        let s1_t = measure_cohort(&minds, &corpus, T_LO..T_HI, use_fallback);
        let s1_fp = measure_cohort(&minds, &corpus, FP_LO..FP_HI, use_fallback);
        let s1_fm = measure_cohort(&minds, &corpus, FM_LO..FM_HI, use_fallback);
        let s1_p = measure_cohort(&minds, &corpus, P_LO..P_HI, use_fallback);
        let s1_n = measure_cohort(&minds, &corpus, N_LO..N_HI, use_fallback);
        let s1_ctrl = measure_cohort(&minds, &corpus, CTRL_LO..CTRL_HI, use_fallback);
        for (label, assoc) in [
            ("T", s1_t),
            ("FP", s1_fp),
            ("FM", s1_fm),
            ("P", s1_p),
            ("N", s1_n),
            ("CTRL", s1_ctrl),
        ] {
            print_association_table(&format!("S({label}, V1)"), assoc);
        }

        let n_bloom_v1: HashMap<MailboxId, Vec<bool>> = (N_LO..N_HI)
            .map(|id| {
                let owner = fleet.get(&id).expect("N owner must exist");
                let verse0 = &owner_verses(&corpus, id)[0];
                let seed = encode_plane(verse0, u64::from(id));
                (id, bloom_verdicts(owner, &seed))
            })
            .collect();

        // ── O2 placebo. ───────────────────────────────────────────────────
        {
            let dk_p = delta_kappa(s0_p, s1_p);
            match dk_p {
                Some(dk) => {
                    println!("O2 placebo: |Δκ(P)| = {}", dk.abs());
                    assert!(dk.abs() < MOVEMENT_FLOOR, "O2: P must not move");
                }
                None => println!("O2 placebo: DEGENERATE at V0 or V1 — reported, not asserted"),
            }
            for id in P_LO..P_HI {
                let mind = minds.get(&id).expect("P mind");
                let prior = *mind
                    .reserved
                    .iter()
                    .find(|&&r| mind.intern.name(r) == "blw5:prior")
                    .expect("P injection must have landed a reserved prior");
                let band00 = mind
                    .reserved
                    .iter()
                    .find(|&&r| mind.intern.name(r) == "blw5:band:00")
                    .copied()
                    .expect("P injection must have landed band_00");
                assert!(
                    mind.arena
                        .get(CStmt {
                            s: prior,
                            cop: Copula::Inh,
                            p: band00
                        })
                        .is_some(),
                    "O2 twin: P's injection mechanics must have executed (band_00 present)"
                );
            }
        }

        // ── O3 null instrument. ───────────────────────────────────────────
        {
            let mut total_hamming = 0usize;
            for id in N_LO..N_HI {
                let before = &n_bloom_v0[&id];
                let after = &n_bloom_v1[&id];
                let hamming = before.iter().zip(after).filter(|(a, b)| a != b).count();
                total_hamming += hamming;
                assert_eq!(
                    before, after,
                    "O3: N's bloom verdicts must be byte-identical V0->V1 for owner {id}"
                );
            }
            println!(
                "O3 null instrument: total Hamming = {total_hamming} (frozen by construction; pool drift 0 here, since N's rows never change)"
            );
        }

        // ── O4 the observable. ─────────────────────────────────────────────
        let dk_t = delta_kappa(s0_t, s1_t);
        match dk_t {
            Some(dk) => {
                println!("O4 the observable: Δκ(T) = {dk}");
                if dk.abs() >= MOVEMENT_FLOOR {
                    println!("O4 FIRES — awareness reflects the statistic");
                } else {
                    println!("O4 SILENT — the honest null: awareness does not reflect this statistic (at floor {MOVEMENT_FLOOR})");
                }
            }
            None => println!("O4 SATURATED — reader B degenerate at V0 or V1"),
        }

        // ── O5 direction. ──────────────────────────────────────────────────
        let dk_fp = delta_kappa(s0_fp, s1_fp);
        let dk_fm = delta_kappa(s0_fm, s1_fm);
        match (dk_fp, dk_fm) {
            (Some(dfp), Some(dfm)) => {
                let d = dfp - dfm;
                println!("O5 direction: Δκ(FP)={dfp} Δκ(FM)={dfm} d={d}");
                let classification = if d >= DIRECTION_FLOOR {
                    "ANCHORING (testimony-dominance, Goodhart realised)"
                } else if d <= -DIRECTION_FLOOR {
                    "EVIDENCE-DOMINANCE"
                } else if dfp.abs() >= DIRECTION_FLOOR && dfm.abs() >= DIRECTION_FLOOR {
                    "PERTURBATION (value-invariant)"
                } else {
                    "SILENT"
                };
                println!("O5 classification: {classification}");
                assert!(
                    !classification.is_empty(),
                    "O5: a classification must be produced"
                );
            }
            _ => println!(
                "O5 direction: at least one of FP/FM is degenerate or excluded — no classification"
            ),
        }

        // ── O6 firewall self-scan. ─────────────────────────────────────────
        {
            const SRC: &str = include_str!("d_blw_5_observer.rs");
            let marker = "── MEASUREMENT BLOCK (O6 marker) ──";
            let marker_pos = SRC.find(marker).expect("O6 marker must exist in this file");
            let (before, after) = SRC.split_at(marker_pos);
            let forbidden = ["fn reader_a", "fn reader_b", "binary_association(", "kappa"];
            for pat in forbidden {
                let hit = before
                    .lines()
                    .any(|line| !line.contains("O6") && line.contains(pat));
                assert!(
                    !hit,
                    "O6 can-fire: forbidden pattern {pat:?} found before the measurement marker"
                );
            }
            assert!(
                after.contains(marker),
                "O6: post-marker half must contain the marker"
            );
            assert!(
                after.contains("binary_association("),
                "O6: a scan that finds nothing is not evidence — post-marker half must call binary_association("
            );
        }

        // ── O7 exclusion is load-bearing — RESTATED (spec §10). The pinned
        // twin compared B against a shadow reader without the reserved/
        // subjects exclusion; under RCR-only reasoning no reserved-SUBJECT
        // belief ever acquires a corpus predicate, so the two readers were
        // identical by construction on every arm (dry run 3) — an assertion
        // that could not fire. What CAN fire: the payload propagated into
        // the derived layer (reserved-term derived beliefs exist on T at V1),
        // and none of the beliefs reader B accepted carries a reserved term.
        {
            let mut derived_with_reserved = 0usize;
            let mut accepted_with_reserved = 0usize;
            let mut b_positive = 0usize;
            for id in T_LO..T_HI {
                let mind = minds.get(&id).expect("T mind");
                derived_with_reserved += mind
                    .arena
                    .entries()
                    .iter()
                    .filter(|b| {
                        b.stamp == Stamp::default()
                            && (mind.reserved.contains(&b.stmt.s)
                                || mind.reserved.contains(&b.stmt.p))
                    })
                    .count();
                let verses = labelled_verses(&corpus, id);
                for (label, _) in &verses {
                    let Some(stmt) = first_stmt_for_verse(mind, label) else {
                        continue;
                    };
                    if !reader_b(mind, label) {
                        continue;
                    }
                    b_positive += 1;
                    // The beliefs B accepted for this verse: re-derive the
                    // acceptance set and count reserved terms in it.
                    accepted_with_reserved += mind
                        .arena
                        .entries()
                        .iter()
                        .filter(|b| {
                            b.stmt.cop == Copula::Inh
                                && b.stmt.p == stmt.p
                                && b.stmt.s != stmt.s
                                && mind.subjects.contains(&b.stmt.s)
                                && !mind.reserved.contains(&b.stmt.s)
                                && b.stamp == Stamp::default()
                                && b.rung >= 1
                                && b.truth.confidence >= C_MIN
                                && (mind.reserved.contains(&b.stmt.s)
                                    || mind.reserved.contains(&b.stmt.p))
                        })
                        .count();
                }
            }
            println!(
                "O7 exclusion: derived beliefs carrying a reserved term on T at V1 = {derived_with_reserved}; \
                 beliefs reader B accepted that carry a reserved term = {accepted_with_reserved}; B positives = {b_positive}"
            );
            assert!(
                derived_with_reserved > 0,
                "O7 can-fire: the injected family must have propagated into T's derived layer at V1"
            );
            assert_eq!(
                accepted_with_reserved, 0,
                "O7 can-stay-silent: reader B must accept no belief that carries a reserved term"
            );
            assert!(
                b_positive > 0,
                "O7 can-stay-silent: the firewalled reader B must be non-empty (some positives) on T"
            );
        }

        // ── CTRL — pass idempotence. ────────────────────────────────────────
        {
            let dk_ctrl = delta_kappa(s0_ctrl, s1_ctrl);
            match dk_ctrl {
                Some(dk) => {
                    println!("CTRL: Δκ(CTRL) = {dk}");
                    assert_eq!(
                        dk, 0.0,
                        "CTRL: pass idempotence requires Δκ(CTRL) == 0.0 exactly"
                    );
                }
                None => {
                    println!("CTRL: DEGENERATE at V0 or V1 (both should read the same table twice)")
                }
            }
            let mut total_hamming = 0usize;
            for id in CTRL_LO..CTRL_HI {
                let mind = minds.get(&id).expect("CTRL mind");
                let verses = labelled_verses(&corpus, id);
                for (label, _) in &verses {
                    if first_stmt_for_verse(mind, label).is_none() {
                        continue;
                    }
                    // Re-reading the SAME (post-both-reason-passes) arena
                    // twice must be bit-identical — the reader Hamming.
                    let b1 = reader_b(mind, label);
                    let b2 = reader_b(mind, label);
                    if b1 != b2 {
                        total_hamming += 1;
                    }
                }
            }
            println!("CTRL reader Hamming (repeat-read stability) = {total_hamming}");
        }

        // ── DROP. ──────────────────────────────────────────────────────────
        {
            let arms = [
                ("T", dk_t, s0_t, s1_t, T_LO..T_HI),
                ("FP", dk_fp, s0_fp, s1_fp, FP_LO..FP_HI),
                ("FM", dk_fm, s0_fm, s1_fm, FM_LO..FM_HI),
                ("N", delta_kappa(s0_n, s1_n), s0_n, s1_n, N_LO..N_HI),
            ];
            for (label, dk, _s0, _s1, range) in arms {
                let Some(dk) = dk else { continue };
                let mut hamming = 0usize;
                for id in range {
                    let mind = minds.get(&id).expect("arm mind");
                    let verses = labelled_verses(&corpus, id);
                    for (l, _) in &verses {
                        if first_stmt_for_verse(mind, l).is_none() {
                            continue;
                        }
                        let b1 = reader_b(mind, l);
                        let b2 = reader_b(mind, l);
                        if b1 != b2 {
                            hamming += 1;
                        }
                    }
                }
                if dk.abs() < DROP_FLOOR && hamming == 0 {
                    println!("DROP fires for {label}");
                }
            }
        }

        // ── §8 output discipline: not-claimed block. ────────────────────────
        println!();
        println!("== D-BLW-5 — what this test does NOT claim ==");
        println!("1. No validity of the observer effect beyond THIS corpus/instrument.");
        println!(
            "2. No parallelism claim (synchronous loop; fleet-level parallel stays A2-gated)."
        );
        println!("3. No durability claim (MemWal is in-process, not persistence).");
        println!("4. No fusion verdict (this is D-BLW-3's first-order fusion's sibling, not its successor).");
        println!("5. No per-stance dispatch claim (this file never touches stance_panel).");
        println!("6. jc is untouched and one-way: it measures S0/S1, is never modified, never fed its own output.");
        println!("7. The rank enters as TYPICALITY (C2): two ranks with equal mass are indistinguishable to awareness under this encoding.");
        println!("8. N is frozen BY CONSTRUCTION (its rows never change); its pool-drift duty reads 0 here.");
        println!("9. The synthetic corpus is symmetric by design; the V0 reader rates are exactly what the generator makes them.");
    }

    /// C2 of the design's naming discipline: print the FULL `BinaryAssociation`
    /// table, never a bare kappa. Provenance: `blw_fusion.rs:687-709`.
    fn print_association_table(label: &str, assoc: Option<BinaryAssociation>) {
        let Some(assoc) = assoc else {
            println!("  {label}: DEGENERATE (no association — structurally unusable input)");
            return;
        };
        let kappa_str = assoc.kappa.map_or_else(
            || format!("undefined(p_e={:.4})", assoc.expected_agreement),
            |k| format!("{k:.4}"),
        );
        let phi_str = assoc
            .phi
            .map_or_else(|| "undefined(constant)".to_string(), |p| format!("{p:.4}"));
        println!(
            "  {label}: n00={} n01={} n10={} n11={} | rate_a={:.4} rate_b={:.4} | p_o={:.4} p_e={:.4} | kappa={kappa_str} | phi={phi_str}",
            assoc.n00,
            assoc.n01,
            assoc.n10,
            assoc.n11,
            assoc.positive_rate_a,
            assoc.positive_rate_b,
            assoc.observed_agreement,
            assoc.expected_agreement,
        );
    }
}
