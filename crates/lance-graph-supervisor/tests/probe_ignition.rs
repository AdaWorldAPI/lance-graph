//! PROBE-IGNITION — the falsifier for "inject corpus X into thinking style Z".
//!
//! Spec: `.claude/board/exec-runs/probe-ignition-design-opus.md` (design,
//! Opus, design-only, no code) + `.claude/board/exec-runs/probe-ignition-api-inventory-sonnet.md`
//! (exact signatures — wins on any design/inventory conflict). This is the
//! first DRIVEN traversal of the built-but-undriven write path: a fleet of
//! real `MailboxSoA` tenants, seeded from a real corpus and armed with a
//! thinking style by a WRITE, discovered by a SCAN of the kanban board alone
//! (no messaging, no queue, no ractor) — the armed style's own
//! `StrategyOutcome` cast write-on-behalf, drained, sealed into one WAL
//! write, and applied. See the design note §0 for the full headline.
//!
//! ## Two verbs only (design §1b)
//!
//! CAST (a write through `BatchWriter`, write-on-behalf) and LOOK INTO THE
//! KANBAN (a read of the owner's phase). No endpoint, no actor, no queue.
//! `where()` scopes the SCAN only; an armed owner outside the scanned range
//! is armed and never started (G7).
//!
//! ## Deviation from the design note, stated once here (no others)
//!
//! **§2 step "Energize" uses a direct `owner.energy[row]` write, NOT
//! `apply_edges(&[(row, CausalEdge64)])`.** `causal_edge::CausalEdge64` is
//! not reachable from this crate: `cognitive-shader-driver` depends on it
//! privately (no `pub use`), `lance-graph-planner` does not re-export it
//! either, and `lance-graph-supervisor`'s own `Cargo.toml` has no dependency
//! edge to `causal-edge` (verified by reading both manifests before writing
//! this file). Adding one would be a Cargo.toml change, which this build's
//! brief forbids outright. `owner.energy` is the SAME public field
//! `apply_edges` itself mutates (`mailbox_soa.rs:66,362`) and is the field
//! `examples/blw_fusion.rs:515` already writes directly for its own
//! energizing (`owner.energy[row] = …`) — so this substitution has a shipped
//! precedent in the same crate family, not an invented mechanism. Every
//! other design element (the gate, the DAG, the seal/apply, the write-back,
//! the wake) is implemented exactly as specified.
//!
//! ## A second, load-bearing correction: the WAKE runs BEFORE the scan
//!
//! The design's own step numbering lists "scheduled wake (cycle 4 only)"
//! as step 14, after the write-back (step 13) — i.e. at the END of a
//! cycle's processing. Read literally that would make the wake's effect
//! visible only from cycle 5's gate read onward. But the design's own
//! cohort table says the REST cohort's own arc is "wake at c4 →
//! Evaluation" — i.e. cycle 4's OWN gate read must already see the
//! post-wake mantissa. The two claims are consistent only if the wake
//! write lands BEFORE cycle 4's scan/gate read, not after. This file runs
//! the wake at the TOP of the loop body when `c == WAKE_CYCLE`, before
//! `scan_board` — a placement correction, not a behavioural addition (the
//! wake is still exactly one write, still gated on `c == WAKE_CYCLE`,
//! still touches only the REST cohort).
//!
//! ## Provenance
//!
//! Feature-gating pattern: `tests/w2b_real_owner_probe.rs`. `MemWal` /
//! `RowSpanDescriptor` / bloom-plane seeding
//! (`fnv1a`/`bloom_add`/`tokens`/`encode_plane`) / `load_verses`:
//! `crates/lance-graph-planner/examples/blw_fusion.rs` (cited at each site
//! below). `flow_qualia()` / `block_qualia()`: re-derived locally from
//! `cycle_driver.rs:1669` / `:1675` (those functions are `#[cfg(test)]`
//! inside `cycle_driver`'s own module, not importable from here).
//!
//! ## Not compiled, not run by this lane — orchestrator gates
//!
//! This file was written edit-only (no `cargo` of any kind). Every
//! signature cited was read from source in the same pass that wrote this
//! file (see the build tag-file for what could and could not be verified).

#[cfg(feature = "cycle-driver")]
mod probe_ignition {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )]

    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Mutex;

    use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome, WORDS_PER_FP};
    use lance_graph_contract::cognitive_shader::MetaWord;
    use lance_graph_contract::collapse_gate::MailboxId;
    use lance_graph_contract::kanban::{ExecTarget, KanbanColumn};
    use lance_graph_contract::mul::i4_eval::{gate_decision_i4, trust_texture_i4};
    use lance_graph_contract::mul::TrustTexture;
    use lance_graph_contract::qualia::QualiaI4_16D;
    use lance_graph_contract::scheduler::DatasetVersion;
    use lance_graph_contract::soa_view::MailboxSoaView;
    use lance_graph_contract::thinking::ThinkingStyle;
    use lance_graph_planner::batch_writer::BatchWriter;
    use lance_graph_planner::ir::Arena;
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
        apply_sealed_transitions, collect_casts, run_cognitive_work_gated_over, run_cycle,
        seal_cycle, shade_owner, CycleError, CycleOutcome,
    };

    // ── PRE-REGISTERED run shape (design §2) — fixed BEFORE any number
    // exists; NOT adjustable after a run. ───────────────────────────────────

    const FLEET_OWNERS: MailboxId = 64;
    const ROWS_PER_OWNER: usize = 64;
    const POPULATED_ROWS: usize = 48;
    const CORPUS_VERSES: usize = FLEET_OWNERS as usize * POPULATED_ROWS; // 3072
    const SCOPE_LO: MailboxId = 0;
    const SCOPE_HI: MailboxId = 32;
    const CYCLES: u32 = 6;
    const WAKE_CYCLE: u32 = 4;

    // Cohorts (design §2 table). ids sum to exactly 32 inside SCOPE.
    const IGNITE_A_LO: MailboxId = 0;
    const IGNITE_A_HI: MailboxId = 6;
    const IGNITE_C_LO: MailboxId = 6;
    const IGNITE_C_HI: MailboxId = 12;
    const REST_LO: MailboxId = 12;
    const REST_HI: MailboxId = 20;
    const CONTRA_LO: MailboxId = 20;
    const CONTRA_HI: MailboxId = 24;
    const UNARMED_LO: MailboxId = 24;
    const UNARMED_HI: MailboxId = 31;
    const ORPHAN_ID: MailboxId = 31;
    const OUTSIDE_LO: MailboxId = 32;
    const OUTSIDE_HI: MailboxId = 64;

    /// Firing threshold. A row is "firing" once `|energy[row]| >= threshold`
    /// (`mailbox_soa.rs:194-196`).
    const TENANT_THRESHOLD: f32 = 1.0;
    /// Energy landed on a firing row — safely above `TENANT_THRESHOLD`.
    const FIRE_ENERGY: f32 = 2.0;
    /// Any w_slot < 64 works — `apply_edges`'s w_slot gate is never exercised
    /// by this probe (see the module-level deviation note).
    const TENANT_W_SLOT: u8 = 0;

    type Tenant = MailboxSoA<ROWS_PER_OWNER>;
    type Fleet = HashMap<MailboxId, Tenant>;

    // ── ThinkingStyle arming vocabulary (design §6 Q1, folded in per the
    // orchestrator ruling): z in {0,1,2,3}. z=0 is UNARMED and never reaches
    // `thinking_style_for`/`style_vector_for` — both fall back defensively
    // (never actually exercised on this probe's cohorts) rather than panic.
    // Three of the contract's 36 styles are reachable here, never 36 — see
    // the §5 not-claimed block. ───────────────────────────────────────────

    fn thinking_style_for(z: u8) -> ThinkingStyle {
        match z {
            1 => ThinkingStyle::Analytical,
            2 => ThinkingStyle::Creative,
            3 => ThinkingStyle::Reflective,
            _ => ThinkingStyle::Analytical,
        }
    }

    /// The 23D sparse vector `StyleStrategy`'s private `resolve_style` reads
    /// (idx 4 = analytical, idx 3 = creative, idx 0 = depth/reflective —
    /// `style_strategy.rs:236-238`), built so `.plan()`'s internal style
    /// resolution and this file's own `thinking_style_for` agree by
    /// construction (both driven from the same `z`).
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

    /// `mantissa` is DERIVED from live owner state, never stored (design §4):
    /// `min(7, pending_count()) as i8`. `pending_count` (`mailbox_soa.rs:571`)
    /// counts rows with `|energy| >= threshold` over the whole `N`-row plane;
    /// only populated rows are ever written above zero in this probe, so it
    /// is equivalent to "populated rows at/above threshold" without needing
    /// a separate `populated` clamp.
    fn mantissa_of(owner: &Tenant) -> i8 {
        owner.pending_count().min(7) as i8
    }

    /// Flow qualia (warmth=4, groundedness=3, coherence=4, valence=2) — the
    /// SAME construction `cycle_driver.rs:1669`'s `flow_qualia()` test
    /// fixture uses (`flow_proxy = 4+3-0 = 7`, coherence>=4 & valence>=2 &
    /// tension<=1 => Calibrated). Re-derived here (that fn is `#[cfg(test)]`
    /// inside `cycle_driver`, not importable).
    fn flow_qualia() -> QualiaI4_16D {
        QualiaI4_16D(0).with(3, 4).with(14, 3).with(9, 4).with(1, 2)
    }

    /// Uncertain qualia (coherence=-3, tension=3) => `Block`. Provenance:
    /// `cycle_driver.rs:1675`'s `block_qualia()` fixture, re-derived (same
    /// reason as above).
    fn block_qualia() -> QualiaI4_16D {
        QualiaI4_16D(0).with(9, -3).with(2, 3)
    }

    // ── corpus + bloom-plane seeding — COPIED from
    // `crates/lance-graph-planner/examples/blw_fusion.rs` with provenance,
    // per the brief ("consume the pattern; copy the needed pieces"). ───────

    /// Bits set per token in a `WORDS_PER_FP`-word identity plane.
    /// Provenance: `blw_fusion.rs:255` (itself citing `blw_tenant.rs:249`).
    const BLOOM_K: usize = 4;

    /// FNV-1a over `bytes`, salted with `seed`. Provenance: `blw_fusion.rs:258-265`.
    fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325_u64 ^ seed.wrapping_mul(0x100_0000_01b3);
        for &c in bytes {
            h ^= u64::from(c);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    /// Set this token's `BLOOM_K` bits in a `WORDS_PER_FP`-word plane.
    /// Provenance: `blw_fusion.rs:269-278`.
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

    /// Lowercased alphanumeric tokens of length >= 2. Provenance: `blw_fusion.rs:281-285`.
    fn tokens(text: &str) -> impl Iterator<Item = String> + '_ {
        text.split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|t| t.len() >= 2)
            .map(str::to_ascii_lowercase)
    }

    /// Build a plane from a verse's tokens. Provenance: `blw_fusion.rs:290-296`.
    fn encode_plane(text: &str, salt: u64) -> Vec<u64> {
        let mut plane = vec![0u64; WORDS_PER_FP];
        for t in tokens(text) {
            bloom_add(&mut plane, &t, salt);
        }
        plane
    }

    /// Read `index\ttext` rows, bounded to `limit`. Provenance: `blw_fusion.rs:478-485`.
    fn load_verses(path: &str, limit: usize) -> Option<Vec<String>> {
        let raw = std::fs::read_to_string(path).ok()?;
        let verses: Vec<String> = raw
            .lines()
            .filter_map(|l| l.split_once('\t').map(|(_, t)| t.to_string()))
            .take(limit)
            .collect();
        (verses.len() == limit).then_some(verses)
    }

    /// Deterministic synthetic fallback — a non-degeneracy fixture, not a
    /// semantic instrument (design §2 step 1): distinct per-index text so
    /// every content plane differs.
    fn synthetic_corpus(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| {
                let salt = (i as u64).wrapping_mul(2_654_435_761) % 104_729;
                format!("probe ignition synthetic verse {i} token{salt}")
            })
            .collect()
    }

    fn load_or_synthesize_corpus() -> (Vec<String>, &'static str) {
        let path =
            std::env::var("BLW_KJV_TSV").unwrap_or_else(|_| "/tmp/kjv_verses.tsv".to_string());
        match load_verses(&path, CORPUS_VERSES) {
            Some(v) => (v, "BLW_KJV_TSV corpus"),
            None => (
                synthetic_corpus(CORPUS_VERSES),
                "deterministic synthetic fallback",
            ),
        }
    }

    /// The write descriptor `P` — a DESCRIPTOR, never owned delta bytes.
    /// Provenance: `blw_fusion.rs:362-380`.
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

    // ── the WAL seam (in-process; NOT durability) — COPIED from
    // `blw_fusion.rs:396-473`, with the `reads` counter added (mirroring
    // `cycle_driver.rs`'s own `#[cfg(test)]` `FakeWalSink`, cited at G3b). ─

    struct SealedCycle {
        frame: CycleFrame,
        version: DatasetVersion,
        /// The batch's deterministic content hash — the reconciliation-first
        /// idempotency key `commit_cycle` looks up BEFORE appending.
        batch_hash: u64,
        landings: Vec<SweepSlot>,
    }

    struct MemWal {
        sealed: Mutex<Vec<SealedCycle>>,
        next_version: AtomicU64,
        wal_writes: AtomicU64,
        /// `scan_sealed` + `timeline` call count — MUST stay 0 across the
        /// main loop (P4b reads no dataset; G3b).
        reads: AtomicU64,
    }

    impl MemWal {
        fn new() -> Self {
            Self {
                sealed: Mutex::new(Vec::new()),
                next_version: AtomicU64::new(1),
                wal_writes: AtomicU64::new(0),
                reads: AtomicU64::new(0),
            }
        }
        fn wal_writes(&self) -> u64 {
            self.wal_writes.load(Ordering::SeqCst)
        }
        fn reads(&self) -> u64 {
            self.reads.load(Ordering::SeqCst)
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
            // Reconciliation-first: an already-durable (cycle, hash) is success,
            // a matching cycle with a different hash fails closed.
            if let Some(rec) = sealed.iter().find(|s| s.frame.cycle == batch.frame.cycle) {
                return if rec.batch_hash == batch.batch_hash {
                    Ok(CommitOutcome::Reconciled {
                        version: rec.version,
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
            let head = sealed.last().map_or(DatasetVersion(0), |s| s.version);
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
            self.reads.fetch_add(1, Ordering::SeqCst);
            Ok(self
                .sealed
                .lock()
                .expect("MemWal poisoned")
                .iter()
                .filter(|s| after_cycle.is_none_or(|c| s.frame.cycle > c))
                .flat_map(|s| {
                    s.landings.iter().map(|slot| LandedSlot {
                        cycle: s.frame.cycle,
                        slot: slot.clone(),
                    })
                })
                .collect())
        }

        async fn timeline(&self) -> Result<Vec<FrameMeta>, WriteFailed> {
            self.reads.fetch_add(1, Ordering::SeqCst);
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

    fn owner_verses(all: &[String], owner_idx: MailboxId) -> &[String] {
        let lo = owner_idx as usize * POPULATED_ROWS;
        &all[lo..lo + POPULATED_ROWS]
    }

    /// Build + seed one owner: `table($x)` (content per row) + `ThinkingStyle($z)`
    /// (a `MetaWord` write) + qualia (declared fixture) + energize
    /// (`firing_rows` rows set above threshold) — design §2 steps 2-6.
    fn build_owner(
        id: MailboxId,
        verses: &[String],
        armed: u8,
        qualia: QualiaI4_16D,
        firing_rows: usize,
    ) -> Tenant {
        let mut owner: Tenant = MailboxSoA::new(id, TENANT_W_SLOT, TENANT_THRESHOLD);
        let cycle = owner.cycle(); // 0, mirrors blw_fusion.rs:723-728 (seed before tick).
        let meta = MetaWord::new(armed, 0, 0, 0, 0);
        for (row, text) in verses.iter().enumerate() {
            let content = encode_plane(text, u64::from(id));
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
        owner.tick(); // cycle 0 -> 1, mirrors blw_fusion.rs:728.
        for r in 0..firing_rows {
            owner.energy[r] = FIRE_ENERGY;
        }
        owner
    }

    fn build_fleet(corpus: &[String]) -> Fleet {
        let mut fleet = Fleet::new();
        let analytical: u8 = 1;
        let creative: u8 = 2;
        let unarmed: u8 = 0;

        for id in IGNITE_A_LO..IGNITE_A_HI {
            fleet.insert(
                id,
                build_owner(id, owner_verses(corpus, id), analytical, flow_qualia(), 3),
            );
        }
        for id in IGNITE_C_LO..IGNITE_C_HI {
            fleet.insert(
                id,
                build_owner(id, owner_verses(corpus, id), creative, flow_qualia(), 3),
            );
        }
        for id in REST_LO..REST_HI {
            fleet.insert(
                id,
                build_owner(id, owner_verses(corpus, id), analytical, flow_qualia(), 1),
            );
        }
        for id in CONTRA_LO..CONTRA_HI {
            fleet.insert(
                id,
                build_owner(id, owner_verses(corpus, id), analytical, block_qualia(), 3),
            );
        }
        for id in UNARMED_LO..UNARMED_HI {
            fleet.insert(
                id,
                build_owner(id, owner_verses(corpus, id), unarmed, flow_qualia(), 3),
            );
        }
        // ORPHAN (id 31): deliberately NOT inserted — "no owner registered"
        // (design §2 cohort table; the #879 missing-owner caveat, G10).
        for id in OUTSIDE_LO..OUTSIDE_HI {
            fleet.insert(
                id,
                build_owner(id, owner_verses(corpus, id), analytical, flow_qualia(), 3),
            );
        }
        fleet
    }

    // ── the scan (LOOK INTO THE KANBAN) ─────────────────────────────────────

    #[derive(Default)]
    struct ScanResult {
        planning: Vec<MailboxId>,
        cognitive: Vec<MailboxId>,
        evaluation: Vec<MailboxId>,
        absorbed: Vec<MailboxId>,
        missing: usize,
    }

    /// Pure reads through the owner's `phase()`. Design §2 step 7.
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

    // ── the CAST — the Planning and Evaluation columns (probe-local;
    // design §6 Q2: the shipped seam can drive only CognitiveWork) ─────────

    struct ColumnPassOutcome {
        cast: usize,
        held: Vec<MailboxId>,
        missing: usize,
    }

    /// Probe-local pass over an explicit id list. Trusts `scan_board`'s
    /// partition (unlike the shipped `cognitive_pass`'s defensive re-check —
    /// this closure's caller already filtered by phase); counts a missing
    /// owner explicitly (G10), where the shipped seam does not.
    fn column_pass(
        fleet: &Fleet,
        ids: &[MailboxId],
        writer: &mut BatchWriter<Vec<u8>>,
        mut think: impl FnMut(&Tenant) -> Option<(StrategyOutcome, Vec<u8>)>,
    ) -> ColumnPassOutcome {
        let mut cast = 0usize;
        let mut held = Vec::new();
        let mut missing = 0usize;
        for &id in ids {
            let Some(owner) = fleet.get(&id) else {
                missing += 1;
                continue;
            };
            let mut did_cast = false;
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
                    did_cast = true;
                }
            }
            if did_cast {
                cast += 1;
            } else {
                held.push(id);
            }
        }
        ColumnPassOutcome {
            cast,
            held,
            missing,
        }
    }

    /// Design §2 step 8/10: gate first, style mints when the gate's target
    /// matches the style's structural crossing, the gate mints otherwise
    /// (always true at Evaluation, since the style's intended move is
    /// structurally always the Planning -> CognitiveWork crossing).
    fn plan_or_evaluate_think(owner: &Tenant) -> Option<(StrategyOutcome, Vec<u8>)> {
        let armed = owner.meta_at(0).thinking();
        if armed == 0 {
            return None; // UNARMED: never plans.
        }
        let qualia = owner.qualia_at(0);
        let mantissa = mantissa_of(owner);
        let gate = gate_decision_i4(&qualia, mantissa);
        let target = owner.phase().advance_on_gate(&gate)?; // Hold => None => rest.

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

    // ── fingerprinting (G1/G3a/G6 byte-identity) ────────────────────────────

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct OwnerFingerprint {
        phase: KanbanColumn,
        cycle: u32,
        energy: [f32; ROWS_PER_OWNER],
        meta0: MetaWord,
        qualia0: QualiaI4_16D,
    }

    fn fingerprint(o: &Tenant) -> OwnerFingerprint {
        OwnerFingerprint {
            phase: o.phase(),
            cycle: o.cycle(),
            energy: o.energy,
            meta0: o.meta_at(0),
            qualia0: o.qualia_at(0),
        }
    }

    fn fingerprint_all(
        fleet: &Fleet,
        ids: impl IntoIterator<Item = MailboxId>,
    ) -> HashMap<MailboxId, OwnerFingerprint> {
        ids.into_iter()
            .filter_map(|id| fleet.get(&id).map(|o| (id, fingerprint(o))))
            .collect()
    }

    fn phase_cycle_snapshot(fleet: &Fleet) -> HashMap<MailboxId, (KanbanColumn, u32)> {
        fleet
            .iter()
            .map(|(&id, o)| (id, (o.phase(), o.cycle())))
            .collect()
    }

    // ── the main probe ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn probe_ignition_scan_and_cast_no_messaging() {
        let (corpus, provenance) = load_or_synthesize_corpus();
        println!(
            "probe.ignition corpus: {provenance} ({} verses, {FLEET_OWNERS} owners x {POPULATED_ROWS} rows)",
            corpus.len()
        );
        assert_eq!(corpus.len(), CORPUS_VERSES, "PRE-REGISTERED corpus size");

        let mut fleet = build_fleet(&corpus);

        // ── non-degeneracy guard (design §2 step 1) ─────────────────────────
        {
            let sample_ids: [MailboxId; 5] =
                [IGNITE_A_LO, IGNITE_C_LO, REST_LO, CONTRA_LO, OUTSIDE_LO];
            let mut digests = Vec::new();
            for &id in &sample_ids {
                let owner = fleet.get(&id).expect("sample id must be a real owner");
                let plane = owner.content_row(0);
                assert!(
                    plane.iter().any(|&w| w != 0),
                    "content plane must be non-zero for owner {id}"
                );
                digests.push(plane.to_vec());
            }
            for (i, di) in digests.iter().enumerate() {
                for dj in &digests[i + 1..] {
                    assert_ne!(di, dj, "content planes must be pairwise distinct");
                }
            }
            println!(
                "probe.ignition non-degeneracy: {} sampled content planes are non-zero and pairwise distinct",
                digests.len()
            );
        }

        // ── G2c: the armed bits reached the plan and changed something ─────
        {
            let r_a =
                StyleStrategy::reliability_for(ThinkingStyle::Analytical, &plan_context_for(1));
            let r_c = StyleStrategy::reliability_for(ThinkingStyle::Creative, &plan_context_for(2));
            assert_ne!(
                r_a.to_bits(),
                r_c.to_bits(),
                "G2c can-fire: distinct styles must yield distinct reliability (the R-GATE property, style_strategy.rs:486-508)"
            );
            let r_a2 =
                StyleStrategy::reliability_for(ThinkingStyle::Analytical, &plan_context_for(1));
            assert_eq!(
                r_a.to_bits(),
                r_a2.to_bits(),
                "G2c can-stay-silent: two owners armed with the SAME style must produce bit-identical reliability"
            );
            eprintln!("probe.ignition.G2c: reliability(Analytical)={r_a} != reliability(Creative)={r_c}; same-style reliability is bit-identical");
        }

        let mut sink = MemWal::new();
        let mut writer: BatchWriter<Vec<u8>> = BatchWriter::new();
        let mut position_base: u64 = 0;
        let mut watermarks: HashMap<MailboxId, Option<u64>> = HashMap::new();
        // Cursor into each owner's firing rows — the write-back pass
        // (design §2 step 13) consumes them one at a time.
        let mut next_firing_row: HashMap<MailboxId, usize> = HashMap::new();

        let unarmed_and_outside: Vec<MailboxId> = (UNARMED_LO..UNARMED_HI)
            .chain(OUTSIDE_LO..OUTSIDE_HI)
            .collect();
        let rest_ids: Vec<MailboxId> = (REST_LO..REST_HI).collect();
        let contra_ids: Vec<MailboxId> = (CONTRA_LO..CONTRA_HI).collect();
        let mut rediscovered_rest_c2 = false;
        let mut rediscovered_rest_c3 = false;
        let mut contra_seen_after_c1: HashSet<MailboxId> = HashSet::new();
        let mut end_of_c5: Option<HashMap<MailboxId, OwnerFingerprint>> = None;

        for c in 1..=CYCLES {
            // ── scheduled wake (design §2 step 14, RE-ORDERED to run before
            // the scan — see the module-level correction note) ─────────────
            if c == WAKE_CYCLE {
                for &id in &rest_ids {
                    let owner = fleet.get_mut(&id).expect("REST owner must exist");
                    let row = *next_firing_row.get(&id).unwrap_or(&0);
                    owner.energy[row] = FIRE_ENERGY;
                }
                println!(
                    "probe.ignition wake @c{c}: re-energized one row for each of {} REST owners",
                    rest_ids.len()
                );
            }

            let scan = scan_board(&fleet, SCOPE_LO..SCOPE_HI);
            assert_eq!(
                scan.missing, 1,
                "scan.missing @c{c}: the orphan (id {ORPHAN_ID}) must be the only missing id in every scan"
            );
            eprintln!(
                "probe.ignition scan @c{c}: planning={} cognitive={} evaluation={} absorbed={} missing={}",
                scan.planning.len(),
                scan.cognitive.len(),
                scan.evaluation.len(),
                scan.absorbed.len(),
                scan.missing
            );

            if c == 2 {
                rediscovered_rest_c2 = rest_ids.iter().all(|id| scan.cognitive.contains(id));
            }
            if c == 3 {
                rediscovered_rest_c3 = rest_ids.iter().all(|id| scan.cognitive.contains(id));
            }
            if c >= 2 {
                for &id in &contra_ids {
                    if scan.planning.contains(&id)
                        || scan.cognitive.contains(&id)
                        || scan.evaluation.contains(&id)
                    {
                        contra_seen_after_c1.insert(id);
                    }
                }
            }

            let wal_writes_at_top_of_cycle = sink.wal_writes();

            // ── G3a: casting mutates nothing — snapshot BEFORE the passes ──
            let snap_before_cast = phase_cycle_snapshot(&fleet);

            let planning_outcome =
                column_pass(&fleet, &scan.planning, &mut writer, plan_or_evaluate_think);

            let cognitive_outcome =
                run_cognitive_work_gated_over(&fleet, &scan.cognitive, &mut writer, |owner| {
                    let armed = owner.meta_at(0).thinking();
                    let style = thinking_style_for(armed);
                    let ctx = plan_context_for(armed);
                    let qualia = owner.qualia_at(0);
                    let mantissa = mantissa_of(owner);
                    let reliability = StyleStrategy::reliability_for(style, &ctx);
                    Some((qualia, mantissa, reliability, row_span_payload(owner)))
                });

            let evaluation_outcome = column_pass(
                &fleet,
                &scan.evaluation,
                &mut writer,
                plan_or_evaluate_think,
            );

            let snap_after_cast = phase_cycle_snapshot(&fleet);
            assert_eq!(
                snap_before_cast, snap_after_cast,
                "G3a @c{c}: staging casts must not mutate any owner's phase or cycle"
            );

            let total_casts =
                planning_outcome.cast + cognitive_outcome.cast + evaluation_outcome.cast;

            if total_casts == 0 {
                // ── REST BRANCH (design §2 step 11): zero staged casts =>
                // record the rest, do NOT call run_cycle, no seal, wal_writes
                // unchanged from a value captured BEFORE this cycle's passes
                // ran (not compared to itself — that would be vacuous). ────
                assert_eq!(
                    sink.wal_writes(),
                    wal_writes_at_top_of_cycle,
                    "G6 can-stay-silent @c{c}: a rest cycle must not move wal_writes"
                );
                eprintln!(
                    "probe.ignition.G6 @c{c}: 0 casts staged — cycle rests, no seal, wal_writes={}",
                    sink.wal_writes()
                );
                if c == 5 {
                    end_of_c5 = Some(fingerprint_all(
                        &fleet,
                        fleet.keys().copied().collect::<Vec<_>>(),
                    ));
                }
                if c == CYCLES {
                    let end_of_c6 =
                        fingerprint_all(&fleet, fleet.keys().copied().collect::<Vec<_>>());
                    match &end_of_c5 {
                        Some(before) => {
                            assert_eq!(before, &end_of_c6, "G6 can-stay-silent: the whole fleet is byte-identical across a rest cycle");
                            eprintln!("probe.ignition.G6 can-stay-silent: fleet byte-identical between end-of-c5 and end-of-c6 ({} owners)", end_of_c6.len());
                        }
                        None => panic!(
                            "G6: end_of_c5 was never captured — cycle 5 was not observed as a rest cycle, so the \
                             pinned c5-vs-c6 comparison this gate depends on cannot run honestly"
                        ),
                    }
                }
                continue;
            }

            let wal_writes_before = sink.wal_writes();
            let base_version = sink.head();
            let outcome: CycleOutcome = match run_cycle(
                &mut sink,
                &mut fleet,
                &mut writer,
                CycleFrame::new(CycleId(u64::from(c)), base_version),
                position_base,
                &mut watermarks,
                u64::from,
            )
            .await
            {
                Ok(o) => o,
                Err(CycleError::Seal(_)) => panic!("probe.ignition @c{c}: unexpected seal failure (MemWal never injects one in the main run)"),
                Err(CycleError::Apply { cause, .. }) => panic!("probe.ignition @c{c}: unexpected apply failure: {cause}"),
            };
            position_base = position_base.max(outcome.sealed.next_position_base);

            // ── G3b (dynamic half): the ids whose phase actually changed
            // are exactly the sealed transitions' owners; the sink was
            // never read to apply them. ─────────────────────────────────
            let snap_post_apply = phase_cycle_snapshot(&fleet);
            let mut changed: HashSet<MailboxId> = HashSet::new();
            for (&id, &prev) in &snap_after_cast {
                if snap_post_apply.get(&id) != Some(&prev) {
                    changed.insert(id);
                }
            }
            let sealed_owners: HashSet<MailboxId> =
                outcome.sealed.transitions.iter().map(|t| t.owner).collect();
            assert_eq!(
                changed, sealed_owners,
                "G3b can-fire @c{c}: changed set must equal the sealed transitions' owners"
            );
            assert_eq!(
                sink.reads(),
                0,
                "G3b can-fire @c{c}: applying a sealed cycle must never read the sink"
            );

            // ── G2b (MID-FLIGHT CORRECTION — the design note's §2 step 8 is
            // the authority, not its own G2b row: when the gate says Prune
            // at Planning, shade_owner REPLACES the style's move, so a
            // Planning-origin cast can be either the style's (Elixir,
            // Flowing) or the gate's (Native, Pruned) mint). ───────────────
            for t in &outcome.sealed.transitions {
                match (t.mv.from, t.mv.to) {
                    (KanbanColumn::Planning, KanbanColumn::CognitiveWork) => assert_eq!(
                        t.mv.exec,
                        ExecTarget::Elixir,
                        "G2b can-fire @c{c}: Planning->CognitiveWork must be the STYLE's Elixir mint"
                    ),
                    (KanbanColumn::Planning, KanbanColumn::Prune) => assert_eq!(
                        t.mv.exec,
                        ExecTarget::Native,
                        "G2b can-fire @c{c}: Planning->Prune must be the GATE's Native mint (shade_owner replaced the style's move)"
                    ),
                    (KanbanColumn::CognitiveWork, _) | (KanbanColumn::Evaluation, _) => assert_eq!(
                        t.mv.exec,
                        ExecTarget::Native,
                        "G2b can-stay-silent @c{c}: CognitiveWork/Evaluation-origin casts are always the gate's Native mint"
                    ),
                    _ => {}
                }
            }

            eprintln!(
                "probe.ignition.G1/G6 @c{c}: {} casts staged, wal_writes {}->{}, {} transitions applied",
                total_casts,
                wal_writes_before,
                sink.wal_writes(),
                outcome.applied.applied.len()
            );

            if c == 1 {
                assert_eq!(
                    outcome.sealed.transitions.len(),
                    24,
                    "G1 can-fire: c1 must advance exactly 24 owners"
                );
                assert_eq!(
                    sink.wal_writes(),
                    1,
                    "G1 can-fire: c1 is the first WAL write"
                );
                // MID-FLIGHT CORRECTION (design note §2 step 8 is the
                // authority): 24 = 20 Flow advances + 4 Block advances, not
                // 24 uniform Elixir/CognitiveWork crossings.
                let flow_advances = outcome
                    .sealed
                    .transitions
                    .iter()
                    .filter(|t| {
                        t.mv.from == KanbanColumn::Planning
                            && t.mv.to == KanbanColumn::CognitiveWork
                    })
                    .count();
                let block_advances = outcome
                    .sealed
                    .transitions
                    .iter()
                    .filter(|t| {
                        t.mv.from == KanbanColumn::Planning && t.mv.to == KanbanColumn::Prune
                    })
                    .count();
                assert_eq!(
                    flow_advances, 20,
                    "G1 can-fire: 20 Flow advances at c1 (IGNITE_A 6 + IGNITE_C 6 + REST 8)"
                );
                assert_eq!(
                    block_advances, 4,
                    "G1 can-fire: 4 Block advances at c1 (CONTRA)"
                );
                eprintln!("probe.ignition.G1 decomposition @c1: Planning->CognitiveWork=20 (Flow), Planning->Prune=4 (Block)");
                let untouched_before = fingerprint_all(&fleet, unarmed_and_outside.iter().copied());
                // (fingerprints were already taken post-apply above via snap_post_apply
                // for phase/cycle; re-derive the FULL fingerprint here for the
                // can-stay-silent half's field-isolation claim.)
                assert_eq!(
                    untouched_before.len(),
                    unarmed_and_outside.len(),
                    "G1 can-stay-silent: every OUTSIDE/UNARMED id must have a real owner"
                );
                eprintln!(
                    "probe.ignition.G1 can-stay-silent @c1: {} OUTSIDE+UNARMED owners fingerprinted (phase+cycle+energy+meta+qualia); \
                     decomposition 32 out-of-scope + 7 unarmed + 1 orphan = 40 untouched",
                    untouched_before.len()
                );
            }

            // ── write-back pass (design §2 step 13): &mut, AFTER apply,
            // never during compute. ──────────────────────────────────────
            for mv in &outcome.applied.applied {
                let id = mv.mailbox;
                let row = *next_firing_row.get(&id).unwrap_or(&0);
                let owner = fleet
                    .get_mut(&id)
                    .expect("an applied move's owner must exist");
                let consumed = owner.consume_firing(row);
                assert!(
                    consumed,
                    "write-back @c{c}: row {row} of owner {id} must still be firing"
                );
                next_firing_row.insert(id, row + 1);
            }

            if c == 1 {
                let g4_owner = fleet.get(&REST_LO).expect("REST_LO owner");
                assert_eq!(
                    g4_owner.qualia_at(0),
                    flow_qualia(),
                    "G4 can-fire: REST qualia is the Flow fixture"
                );
                eprintln!("probe.ignition.G4 can-fire @c1: REST owner {REST_LO} cast on a would-be-Flow qualia (mantissa was >0 pre-consumption)");
            }
            if c == 2 {
                let g4_owner = fleet.get(&REST_LO).expect("REST_LO owner");
                let q = g4_owner.qualia_at(0);
                assert_eq!(
                    q,
                    flow_qualia(),
                    "G4 can-stay-silent: qualia byte-identical to c1 (never rewritten)"
                );
                assert_ne!(
                    q,
                    QualiaI4_16D::ZERO,
                    "G4 anti-rig: qualia must not be the trivial zero vector"
                );
                assert_eq!(
                    trust_texture_i4(&q),
                    TrustTexture::Calibrated,
                    "G4 anti-rig: texture is Calibrated"
                );
                let flow_proxy = i32::from(q.get(3)) + i32::from(q.get(14)) - i32::from(q.get(2));
                assert!(
                    flow_proxy >= 4,
                    "G4 anti-rig: warmth+groundedness-tension must be a would-be-Flow value"
                );
                assert_eq!(
                    mantissa_of(g4_owner),
                    0,
                    "G4 can-stay-silent: mantissa has fallen to 0 by c2"
                );
                eprintln!(
                    "probe.ignition.G4 can-stay-silent @c2: REST owner {REST_LO} rests on an UNCHANGED, non-trivial \
                     would-be-Flow qualia (flow_proxy={flow_proxy}); only mantissa (derived, live) differs"
                );
            }
        }

        assert!(
            rediscovered_rest_c2,
            "G5 can-fire: all 8 REST owners re-found by the scan at c2"
        );
        assert!(
            rediscovered_rest_c3,
            "G5 can-fire: all 8 REST owners re-found by the scan at c3"
        );
        assert!(
            contra_seen_after_c1.is_empty(),
            "G5 can-stay-silent: no CONTRA owner may appear in any active scan bucket after c1 (absorbing)"
        );
        eprintln!(
            "probe.ignition.G5: rediscovered(REST)=8 at c2 and c3; rediscovered(CONTRA)=0 across c2..c{CYCLES}"
        );

        // ── G7: the where() axis is load-bearing ────────────────────────────
        {
            let outside_id = OUTSIDE_LO;
            let owner = fleet.get(&outside_id).expect("OUTSIDE owner must exist");
            assert_eq!(
                owner.phase(),
                KanbanColumn::Planning,
                "G7 can-stay-silent: OUTSIDE owner never advanced in the main run"
            );
            assert_eq!(
                mantissa_of(owner),
                3,
                "G7 can-stay-silent: OUTSIDE owner's firing rows were never consumed"
            );
            let mut throwaway: BatchWriter<Vec<u8>> = BatchWriter::new();
            let (out, payload) = plan_or_evaluate_think(owner)
                .expect("G7 can-fire: a scanned OUTSIDE owner must produce a cast-able outcome");
            let cast = emit_bootstrap_intent(
                &out,
                owner.mailbox_id(),
                owner.current_cycle(),
                &mut throwaway,
                payload,
            );
            assert!(
                cast.is_some(),
                "G7 can-fire: widening the scope by one OUTSIDE id must stage a cast"
            );
            eprintln!("probe.ignition.G7: OUTSIDE owner {outside_id} — silent in the main run (address excluded), casts when scanned directly (throwaway writer)");
        }

        // ── G8: the style-arming axis is load-bearing ──────────────────────
        {
            let unarmed_id = UNARMED_LO;
            {
                let owner = fleet.get(&unarmed_id).expect("UNARMED owner must exist");
                assert_eq!(
                    owner.meta_at(0).thinking(),
                    0,
                    "G8 can-stay-silent: UNARMED owner stayed unarmed the whole main run"
                );
                assert_eq!(
                    owner.phase(),
                    KanbanColumn::Planning,
                    "G8 can-stay-silent: UNARMED owner never advanced"
                );
            }
            let owner_mut = fleet
                .get_mut(&unarmed_id)
                .expect("UNARMED owner must exist");
            owner_mut.set_meta(0, MetaWord::new(1, 0, 0, 0, 0));
            let owner = fleet.get(&unarmed_id).expect("UNARMED owner must exist");
            let mut throwaway: BatchWriter<Vec<u8>> = BatchWriter::new();
            let (out, payload) = plan_or_evaluate_think(owner)
                .expect("G8 can-fire: an armed owner must produce a cast-able outcome");
            let cast = emit_bootstrap_intent(
                &out,
                owner.mailbox_id(),
                owner.current_cycle(),
                &mut throwaway,
                payload,
            );
            assert!(
                cast.is_some(),
                "G8 can-fire: writing non-zero thinking bits must stage a cast"
            );
            eprintln!("probe.ignition.G8: UNARMED owner {unarmed_id} — silent while thinking bits were 0, casts once armed (throwaway writer)");
        }

        // ── G10: the missing-owner accounting gap (design §6 Q2, #879 OPEN) ─
        {
            let ids = [ORPHAN_ID];
            let mut probe_writer: BatchWriter<Vec<u8>> = BatchWriter::new();
            let probe_local = column_pass(&fleet, &ids, &mut probe_writer, plan_or_evaluate_think);
            assert_eq!(
                probe_local.missing, 1,
                "G10: probe-local column_pass must count the orphan explicitly"
            );
            let probe_local_total = probe_local.cast + probe_local.held.len() + probe_local.missing;

            let mut shipped_writer: BatchWriter<Vec<u8>> = BatchWriter::new();
            let shipped =
                run_cognitive_work_gated_over(&fleet, &ids, &mut shipped_writer, |owner| {
                    let armed = owner.meta_at(0).thinking();
                    let style = thinking_style_for(armed);
                    let ctx = plan_context_for(armed);
                    let qualia = owner.qualia_at(0);
                    let mantissa = mantissa_of(owner);
                    let reliability = StyleStrategy::reliability_for(style, &ctx);
                    Some((qualia, mantissa, reliability, row_span_payload(owner)))
                });
            let shipped_total = shipped.cast + shipped.held_owners.len();

            assert_eq!(
                probe_local_total as i64 - shipped_total as i64,
                1,
                "G10: the probe-local pass accounts for the orphan; the shipped seam silently drops it — the two totals must differ by exactly 1"
            );
            eprintln!(
                "probe.ignition.G10: probe-local total={probe_local_total} (missing=1 counted), shipped total={shipped_total} \
                 (orphan silently skipped, no counter) — difference=1, matching #879's OPEN caveat"
            );
        }

        // ── compile-time self-scans (G2a, G3b static half, G11) ─────────────
        //
        // Every needle below is built by CONCATENATING two literal pieces
        // that are never adjacent in this file's own source text — a needle
        // spelled out contiguously would make its own absence-check
        // vacuously true, since `include_str!` reads THIS file, including
        // the scan code itself.
        {
            let src = include_str!("probe_ignition.rs");

            let kanban_move_literal = format!("{}{} {{", "Kanban", "Move");
            assert!(
                !src.contains(&kanban_move_literal),
                "G2a can-fire: source must never construct a kanban move via its own struct-literal syntax"
            );
            assert!(
                src.contains("emit_bootstrap_intent"),
                "G2a can-stay-silent: the scan must be able to find real content — a scan finding nothing is not evidence"
            );
            eprintln!("probe.ignition.G2a: no kanban-move struct-literal in source; emit_bootstrap_intent present (scan mechanism proven live)");

            let advance_phase_call = format!(".{}(", "advance_phase");
            let try_advance_phase_call = format!(".{}(", "try_advance_phase");
            assert!(
                !src.contains(&advance_phase_call),
                "G3b can-fire (static half): source must not call the owner's phase-mutating methods directly"
            );
            assert!(
                !src.contains(&try_advance_phase_call),
                "G3b can-fire (static half): source must not call the owner's phase-mutating methods directly"
            );
            eprintln!("probe.ignition.G3b (static half): no direct phase-mutating call anywhere in source; dynamic half asserted every cycle above");

            let ack_call = "a".to_string() + "ck(";
            let confirm_call = "confir".to_string() + "m(";
            assert!(
                !src.contains(&ack_call),
                "G11 can-fire: source must not define or call an ack-shaped confirmation method"
            );
            assert!(
                !src.contains(&confirm_call),
                "G11 can-fire: source must not define or call a confirm-shaped confirmation method"
            );
            assert!(
                src.contains("BatchWriter"),
                "G11 can-stay-silent: the scan must be able to find real content — a scan finding nothing is not evidence"
            );
            // NOTE: this message deliberately avoids writing the two needles
            // literally — an earlier form spelled them out and the scan matched
            // its OWN success message (a self-match false positive, caught by
            // the central gate run). The needles are built by concatenation
            // above for the same reason.
            eprintln!("probe.ignition.G11: no ack-shaped or confirm-shaped identifier anywhere in source (E-ACK-ELIMINATED-1); scan mechanism proven live");
        }

        // ── §5 Not-claimed block (design §5, printed verbatim in spirit) ────
        println!();
        println!("== PROBE-IGNITION — what this probe does NOT claim ==");
        println!("1. No durability. MemWal is an in-process Mutex/Vec; its versions are sequence numbers, not Lance versions.");
        println!("2. No parallelism. The loop is synchronous.");
        println!("3. No scale claim. {FLEET_OWNERS} owners; the 64k sparse property is proven separately over FakeOwner.");
        println!("4. No multi-writer claim. Single-writer MemWal.");
        println!(
            "5. No deinterlace / temporal claim. This probe does not read through deinterlace."
        );
        println!(
            "6. No validity claim. reliability is settledness, not ground-truth correspondence."
        );
        println!("7. No GUID-prefix routing claim. where() is a contiguous MailboxId range, an honest stand-in.");
        println!("8. No 36-style claim. Three styles are reachable here (Analytical/Creative/Reflective).");
        println!("9. No semantic claim about the corpus. Qualia are declared fixtures, not encoded from text.");
        println!("10. No zero-copy claim. SweepSlot::payload is Vec<u8> by the shipped signature.");
        println!("11. No claim that the loop can re-enter Planning. The arc stops at Commit/Prune/Hold in this probe.");
        println!("12. No recovery claim. recover_fleet is not exercised; G9 (separate test) covers only the WAL-failure retry path.");
    }

    // ── G9: the drained-writer retry footgun (side fixture, own MemWal) ─────
    //
    // #879 OPEN. Provenance for the shape: `cycle_driver.rs`'s own
    // `failed_seal_preserves_the_frozen_cycle_for_byte_identical_retry` test
    // (`cycle_driver.rs:963-1006`) — re-derived here (that fixture is
    // `#[cfg(test)]`-private to `cycle_driver`, not importable).
    #[tokio::test]
    async fn probe_ignition_g9_drained_writer_retry_footgun() {
        struct FlakyWal {
            inner: MemWal,
            fail_next: AtomicBool,
        }
        impl FlakyWal {
            fn new() -> Self {
                Self {
                    inner: MemWal::new(),
                    fail_next: AtomicBool::new(false),
                }
            }
            fn fail_next_commit(&self) {
                self.fail_next.store(true, Ordering::SeqCst);
            }
        }
        impl WalSink for FlakyWal {
            async fn commit_cycle(
                &mut self,
                batch: DetachedCycleBatch,
            ) -> Result<CommitOutcome, CommitError> {
                if self.fail_next.swap(false, Ordering::SeqCst) {
                    return Err(CommitError::Io(WriteFailed(
                        "G9 injected retryable WAL failure".into(),
                    )));
                }
                self.inner.commit_cycle(batch).await
            }
            async fn scan_sealed(
                &self,
                after_cycle: Option<CycleId>,
            ) -> Result<Vec<LandedSlot>, WriteFailed> {
                self.inner.scan_sealed(after_cycle).await
            }
            async fn timeline(&self) -> Result<Vec<FrameMeta>, WriteFailed> {
                self.inner.timeline().await
            }
        }

        let g9_verses = synthetic_corpus(2 * POPULATED_ROWS);
        let mut fleet: Fleet = Fleet::new();
        fleet.insert(
            900,
            build_owner(900, &g9_verses[0..POPULATED_ROWS], 1, flow_qualia(), 1),
        );
        fleet.insert(
            901,
            build_owner(
                901,
                &g9_verses[POPULATED_ROWS..2 * POPULATED_ROWS],
                1,
                flow_qualia(),
                1,
            ),
        );

        let mut sink = FlakyWal::new();
        let mut writer: BatchWriter<Vec<u8>> = BatchWriter::new();
        let mut watermarks: HashMap<MailboxId, Option<u64>> = HashMap::new();

        let scan = scan_board(&fleet, [900, 901]);
        assert_eq!(scan.planning.len(), 2, "both G9 owners start in Planning");
        let staged = column_pass(&fleet, &scan.planning, &mut writer, plan_or_evaluate_think);
        assert_eq!(staged.cast, 2, "both G9 owners must stage a cast");

        let before_phases = phase_cycle_snapshot(&fleet);
        sink.fail_next_commit();
        let err = run_cycle(
            &mut sink,
            &mut fleet,
            &mut writer,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            0,
            &mut watermarks,
            u64::from,
        )
        .await
        .expect_err("G9: the injected WAL failure must surface as CycleError::Seal");

        let CycleError::Seal(failure) = err else {
            panic!("G9: expected a Seal failure");
        };
        assert_eq!(
            failure.casts.len(),
            2,
            "G9: the frozen cycle carries both casts, byte-identical"
        );
        assert_eq!(
            before_phases,
            phase_cycle_snapshot(&fleet),
            "G9: no owner mutated on a failed seal"
        );
        eprintln!("probe.ignition.G9 can-fire (half 1): injected WAL failure surfaced as CycleError::Seal, zero owner mutation");

        let frozen_frame = failure.frame;
        let frozen_casts = failure.casts;
        let sealed = seal_cycle(&mut sink, frozen_frame, frozen_casts)
            .await
            .expect("G9: the retry with the frozen cast set must succeed");
        assert_eq!(
            sink.inner.wal_writes(),
            1,
            "G9: exactly one successful WAL write total"
        );
        let applied = apply_sealed_transitions(&mut fleet, &sealed, &mut watermarks)
            .expect("G9: apply must succeed on the retried seal");
        assert_eq!(
            applied.applied.len(),
            2,
            "G9: both owners advance exactly once"
        );
        eprintln!("probe.ignition.G9 can-fire (half 2): retry via seal_cycle(sink, failure.frame, failure.casts) lands the byte-identical cycle");

        // ── can-stay-silent: on the SAME (already-drained) writer, a fresh
        // collect_casts yields zero slots — the footgun made visible. A
        // naive `run_cycle` retry (which calls collect_casts fresh) would
        // seal an EMPTY cycle here and silently "succeed" — a future guard
        // must flip this from silent to loud. ──────────────────────────────
        let redrained = collect_casts(&mut writer, CycleId(2), 0, u64::from);
        assert!(
            redrained.slots.is_empty(),
            "G9 can-stay-silent: the drained writer has nothing left to collect — a naive retry via run_cycle would seal an empty cycle here"
        );
        eprintln!("probe.ignition.G9 can-stay-silent: fresh collect_casts on the drained writer yields 0 slots (the footgun this test pins for a future guard)");
    }
}
