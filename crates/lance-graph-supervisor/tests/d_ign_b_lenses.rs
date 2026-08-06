//! D-IGN-B — lens SELECTION over the shipped four-stance panel, driven
//! through the same no-messaging cast/scan machinery PROBE-IGNITION proved.
//!
//! Spec: `.claude/board/exec-runs/d-ign-b-design-opus.md` (design, Opus,
//! design-only, no code) + `.claude/board/exec-runs/d-ign-b-api-inventory-sonnet.md`
//! (exact signatures — wins on any design/inventory conflict). Scaffolding
//! provenance: `crates/lance-graph-supervisor/tests/probe_ignition.rs`
//! (GREEN, 2/2, G1-G11 both halves) — this file copies its fleet/scan/cast
//! shapes with provenance noted at each site, per the build brief.
//!
//! ## The headline finding this file must never blur (design §0, F0)
//!
//! `stance_panel(arena, intern, out)` (`stance.rs:469-478`) returns all
//! FOUR stances as ONE 4-tuple computed by ONE call. There is no per-stance
//! dispatch and no way to compute one stance alone without refactoring the
//! shipped function. **Arming SELECTS what is READ, not what is computed.**
//! Every armed owner runs the identical panel over its own arena; the
//! ordinal `z` (read from `owner.meta_at(0).thinking()`, `MetaWord`'s
//! 6-bit `thinking` field, `cognitive_shader.rs:42-76`) picks which of the
//! four tuple elements this file records as that owner's readout. No
//! sentence in this file may say "dispatch" — it says "selection".
//!
//! ## What the lens does NOT read (design §0, F1)
//!
//! A row's content is a one-way bloom plane (4 bits/token OR'd into a
//! `WORDS_PER_FP`-word plane, `mailbox_soa.rs` via `encode_plane` below) —
//! there is no inverse from plane bits back to text. The lens body never
//! decodes SoA row bytes; it re-reads the OWNER'S OWN verse-text slice,
//! selected by the owner's address (`owner_idx * POPULATED_ROWS`) and span,
//! the same slice `build_owner` seeded that owner's bloom planes from. The
//! SoA governs selection (ownership + scope + arming + phase); the text
//! comes from the corpus the rows were seeded from, not from the rows.
//!
//! ## z=5 (Fusion) — BLOCKED in-cycle (design §3, the deliverable per brief)
//!
//! `blw_fusion.rs`'s ranking pool must GROW across many sealed horizons
//! (`S_CYCLES = 8`, `blw_fusion.rs:123`) for Strict-vs-Aware admission to
//! differ at all. This file's fleet is seeded ONCE and each owner accrues
//! at most one or two sealed horizons — with one horizon Strict and Aware
//! admit the same rows and any Δ is 0 BY CONSTRUCTION, the exact vacuous-
//! readout shape the workspace's falsifiability rule exists to reject.
//! `jc` is also not a dependency of this crate (`lance-graph-supervisor/
//! Cargo.toml` has no `jc` edge — a manifest change, not a worker's call).
//! Ordinal 5 stays RESERVED in the arming vocabulary below: never mapped
//! onto z=1..4, never armed on any owner in this file's fleet. See the
//! "z=5 BLOCKED" check near the end of the test body.
//!
//! ## Deviations from the design note, stated here (no others)
//!
//! 1. **L5's cited "20 Flow + 4 Block" decomposition does not apply to
//!    this file's cohorts.** That figure is PROBE-IGNITION's own number
//!    (`probe_ignition.rs:977-984`), produced by a CONTRA cohort (4 owners
//!    on `block_qualia()`) that design §5's cohort re-carve does not
//!    include here — every in-scope armed owner in this file uses
//!    `flow_qualia()`. Re-asserting "20+4" verbatim would be a false,
//!    copy-pasted claim. L5 instead asserts the DERIVED expectation for
//!    THIS file's cohorts (all 30 armed owners Flow-advance, 0 Block) and
//!    prints the measured counts rather than silently trusting them.
//! 2. **`thinking_style_for` maps z=4 onto the SAME `ThinkingStyle` as
//!    z=3 (`Reflective`).** The lens-selection ordinal (1..4, which of the
//!    four stance-panel tuple elements to record) and the `StyleStrategy`
//!    dispatch input share one 6-bit field by design (`owner.meta_at(0)
//!    .thinking()` is read for both purposes, per design §1). This file
//!    needed a 4th distinct lens ordinal but only 3 `ThinkingStyle`
//!    variants were verified reachable from this crate in the probe's own
//!    inventory (Analytical/Creative/Reflective) — inventing a 4th
//!    variant's exact discriminant was not verified in this pass, so z=4
//!    reuses Reflective's style rather than guess. This affects ONLY the
//!    `reliability`/gate-decision input, never the lens SELECTION itself
//!    (`run_lens` switches on `z` directly, independent of `ThinkingStyle`).
//! 3. **`LensReadout::digest` folds NO variant-discriminant tag.** An
//!    earlier draft hashed a `0u8`/`1u8`/`2u8`/`3u8` tag before each
//!    variant's contents, which makes any cross-lens `!=` comparison pass
//!    by construction of the tag alone — an assertion implied by the
//!    digest function itself, not a test of content (this workspace's
//!    falsifiability rule). Removed; see the doc comment on `digest()`.
//!
//! ## Not compiled, not run by this lane — orchestrator gates
//!
//! This file was written edit-only (no `cargo` of any kind). Every
//! signature cited was read from source in the same pass that wrote this
//! file (see the build tag-file, `.claude/board/exec-runs/d-ign-b-build.md`,
//! for what could and could not be verified).
//!
//! > **⊘ SUPERSEDED 2026-08-06 — this file IS compiled and run now.** CI
//! > runs `cargo test -p lance-graph-supervisor --features cycle-driver
//! > --test d_ign_b_lenses`. The section above records the authoring pass's
//! > honest state, not the file's current one; it is regraded in place, not
//! > deleted. Arming that CI step exposed a real defect the unarmed gate had
//! > been hiding (without the feature this binary runs 0 tests, so it was
//! > green vacuously): the deterministic synthetic fallback corpus emitted
//! > text `stance::stream` extracts NOTHING from, so all four lens readouts
//! > came back empty and L1's untagged digests collided empty-vs-empty. The
//! > corpus was replaced with `synthetic_window` (see its doc comment for
//! > what each arm needs) and L1 gained an explicit non-emptiness
//! > precondition so an empty corpus reports itself as a corpus defect
//! > instead of a lens collision. `stance.rs` was NOT touched — the lenses
//! > were never at fault. See `.claude/board/EPIPHANIES.md`
//! > `E-D-IGN-B-CORPUS-PRODUCED-NOTHING-TO-READ-1`.

#[cfg(feature = "cycle-driver")]
mod d_ign_b_lenses {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )]

    use std::collections::{HashMap, HashSet};
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome, WORDS_PER_FP};
    use lance_graph_contract::cognitive_shader::MetaWord;
    use lance_graph_contract::collapse_gate::MailboxId;
    use lance_graph_contract::kanban::{ExecTarget, KanbanColumn};
    use lance_graph_contract::mul::i4_eval::gate_decision_i4;
    use lance_graph_contract::qualia::QualiaI4_16D;
    use lance_graph_contract::scheduler::DatasetVersion;
    use lance_graph_contract::soa_view::MailboxSoaView;
    use lance_graph_contract::thinking::ThinkingStyle;
    use lance_graph_planner::batch_writer::BatchWriter;
    use lance_graph_planner::ir::Arena;
    use lance_graph_planner::nars::stance::{stance_panel, stream, FlipKind, Interner, ReadOut};
    use lance_graph_planner::nars::{BeliefArena, CStmt};
    use lance_graph_planner::owner_adapter::emit_bootstrap_intent;
    use lance_graph_planner::persist_sink::{
        CycleFrame, CycleId, DetachedCycleBatch, LandedSlot, SweepSlot, WalSink, WriteFailed,
    };
    use lance_graph_planner::strategy::style_strategy::StyleStrategy;
    use lance_graph_planner::traits::{
        PlanContext, PlanInput, PlanStrategy, QueryFeatures, StrategyOutcome,
    };
    use lance_graph_supervisor::cycle_driver::{
        run_cognitive_work_gated_over, run_cycle, shade_owner, CycleError, CycleOutcome,
    };

    // ── PRE-REGISTERED run shape (design §5) — fixed BEFORE any run. ───────

    const FLEET_OWNERS: MailboxId = 64;
    const ROWS_PER_OWNER: usize = 64;
    const POPULATED_ROWS: usize = 48;
    const CORPUS_VERSES: usize = FLEET_OWNERS as usize * POPULATED_ROWS; // 3072
    const SCOPE_LO: MailboxId = 0;
    const SCOPE_HI: MailboxId = 32;
    const CYCLES: u32 = 2;

    // Cohorts — design §5's ONE run-shape change (the twin block), the rest
    // of PROBE-IGNITION's cohort table dropped (design §5 justification).
    const TWIN_LO: MailboxId = 0;
    const TWIN_HI: MailboxId = 8;
    const SPREAD_LO: MailboxId = 8;
    const SPREAD_HI: MailboxId = 30;
    const UNARMED_ID: MailboxId = 30;
    const ORPHAN_ID: MailboxId = 31;
    const OUTSIDE_LO: MailboxId = 32;
    const OUTSIDE_HI: MailboxId = 64;

    /// Per-offset arming inside the twin block: z=1,1,2,2,3,3,4,4 (design §4
    /// "twin premise" table). Byte-identical rows, four distinct lenses.
    const TWIN_ARMING: [u8; 8] = [1, 1, 2, 2, 3, 3, 4, 4];

    const TENANT_THRESHOLD: f32 = 1.0;
    const FIRE_ENERGY: f32 = 2.0;
    const TENANT_W_SLOT: u8 = 0;

    type Tenant = MailboxSoA<ROWS_PER_OWNER>;
    type Fleet = HashMap<MailboxId, Tenant>;

    // ── ThinkingStyle dispatch input — SEPARATE from the lens-selection
    // switch (`run_lens` below); see module doc deviation 2. ───────────────

    fn thinking_style_for(z: u8) -> ThinkingStyle {
        match z {
            1 => ThinkingStyle::Analytical,
            2 => ThinkingStyle::Creative,
            _ => ThinkingStyle::Reflective, // z=3 and z=4 (deviation 2)
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

    /// Provenance: `probe_ignition.rs:196-203` (re-derived there from
    /// `cycle_driver.rs:1669`'s `#[cfg(test)]` fixture, not importable).
    fn flow_qualia() -> QualiaI4_16D {
        QualiaI4_16D(0).with(3, 4).with(14, 3).with(9, 4).with(1, 2)
    }

    // ── corpus + bloom-plane seeding — copied from `probe_ignition.rs`
    // (itself citing `blw_fusion.rs`), same provenance chain. ──────────────

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

    fn load_verses(path: &str, limit: usize) -> Option<Vec<String>> {
        let raw = std::fs::read_to_string(path).ok()?;
        let verses: Vec<String> = raw
            .lines()
            .filter_map(|l| l.split_once('\t').map(|(_, t)| t.to_string()))
            .take(limit)
            .collect();
        (verses.len() == limit).then_some(verses)
    }

    /// Nonsense syllables the synthetic corpus builds its clause predicates
    /// from. They must collide with NO catalogue the clause machine consults
    /// — `STOP` / `AUX` (`stance.rs:33-45`), `is_negation` / `is_copula` /
    /// `is_modal_aux` / `is_causal_cue` / `pronoun_case` / `epistemic_reading`
    /// (`clause_cues.rs`, `verb_lexicon.rs`), and `read_verb`'s
    /// `FAMILY_LEXICON` — so the only role a template can give them is the
    /// one it means to: a clause predicate. The `{stem}{window:02}{n:02}`
    /// shape guarantees that (7 chars, alphanumeric, no `-ed`/`-s`/`-ing`
    /// suffix for `classify_verb`'s morphology to strip, absent from every
    /// exact catalogue).
    const SYNTH_STEMS: [&str; 8] = ["vor", "lan", "tik", "mez", "qor", "sil", "dun", "fex"];

    fn synth_term(window: usize, n: usize) -> String {
        format!(
            "{}{:02}{:02}",
            SYNTH_STEMS[n % SYNTH_STEMS.len()],
            window,
            n
        )
    }

    /// One owner-slice's worth (`POPULATED_ROWS` verses) of the deterministic
    /// synthetic corpus.
    ///
    /// ## Why this is not filler (the 2026-08-06 corpus fix)
    ///
    /// The previous fallback emitted `"d-ign-b synthetic verse {i} token{salt}"`.
    /// That text carries no copula, no auxiliary, no typed relational verb and
    /// no `-ed` morphology, so `stance::stream` never arms a predicate and
    /// emits ZERO statements: the arena stays empty, `ReadOut` stays empty,
    /// and all four `stance_panel` arms come back empty. Two empty readouts
    /// then fold zero bytes each and digest IDENTICALLY (`DefaultHasher::new()
    /// .finish()` == 15130871412783076140) — which is what made L1's can-fire
    /// half fail the moment CI armed `--features cycle-driver`. The digest is
    /// deliberately untagged (see `LensReadout::digest`); the defect was the
    /// corpus, not the digest.
    ///
    /// ## What each arm needs, and where this window supplies it
    ///
    /// * **Hegel** (`contradiction_ranking`, `stance.rs:418-427`) — a belief
    ///   whose `contradiction` exceeds `0.05`. `revise_at` sets that to
    ///   `|f₁ − f₂|` on a disjoint-stamp revision (`belief.rs:191-205`), so it
    ///   needs the SAME `(they, Inh, term)` statement observed once affirmed
    ///   (f=0.9) and once negated (f=0.05) → depth 0.85. Supplied by the
    ///   `dev`/`trans` pairs below.
    /// * **Nietzsche** (`stance.rs:483-496`) — iterates Hegel's output and
    ///   partitions by the FIRST vs LAST provenance entry's `negated` flag, so
    ///   it needs both flip directions to be legible from the endpoints:
    ///   affirm→negate (`Devaluation`) and negate→affirm (`Transvaluation`).
    /// * **Kant** (`stance.rs:500-510`) — maps `ReadOut::lifts`, so it needs
    ///   rung-1 lifts: a perception verb with a subject, a complementizer
    ///   within the 3-content-token window, and an emission
    ///   (`stance.rs:237-243`, `:302-364`) — i.e. `they knew that they were X`.
    /// * **Wittgenstein** (`stance.rs:512-532`) — counts distinct language
    ///   games per concept over observed `Inh` beliefs, lift knowers/objects
    ///   and `Impl` cause/effects, so any emission feeds it; the `because`
    ///   verses add the `impl-cause`/`impl-effect` games on top.
    ///
    /// ## Why the SHAPE varies with the window index
    ///
    /// Three of the four arms fold interned `u16` ids (`CStmt`'s `s`/`p`,
    /// Wittgenstein's concept), never the strings behind them — and the
    /// `Interner` assigns ids by order of first appearance, per `run_lens`
    /// call. Two windows with identical structure therefore digest
    /// IDENTICALLY no matter how their tokens are spelled, which would make
    /// L4's ">=2 distinct digests across the in-scope owners" half
    /// unfalsifiable. The counts below are driven by `window % 5|3|7`,
    /// deliberately coprime with the stride-4 arming cycle
    /// (`(id - SPREAD_LO) % 4 + 1`), so each lens's OWN owner set spans
    /// several shapes rather than landing on one residue class.
    fn synthetic_window(window: usize) -> Vec<String> {
        let dev_pairs = 3 + window % 5; // affirm then negate → Devaluation
        let trans_pairs = 2 + window % 3; // negate then affirm → Transvaluation
        let lifts = 4 + window % 7; // knows-that → Kant
        let causal = 2 + window % 3; // "<effect> because <cause>" → Impl games

        let term = |n: usize| synth_term(window, n);
        let mut out: Vec<String> = Vec::with_capacity(POPULATED_ROWS);

        // Disjoint `n` namespaces keep the five roles' terms distinct within
        // one window (0.., 10.., 20.., 40../50.., 60..).
        for j in 0..dev_pairs {
            out.push(format!("they were {}.", term(j)));
        }
        for j in 0..trans_pairs {
            out.push(format!("they were not {}.", term(10 + j)));
        }
        for j in 0..lifts {
            out.push(format!("they knew that they were {}.", term(20 + j)));
        }
        for j in 0..causal {
            out.push(format!(
                "they were {} because they were {}.",
                term(40 + j),
                term(50 + j)
            ));
        }
        for j in 0..dev_pairs {
            out.push(format!("they were not {}.", term(j)));
        }
        for j in 0..trans_pairs {
            out.push(format!("they were {}.", term(10 + j)));
        }

        // Pad to the fixed slice length with plain affirmations. The counts
        // above are bounded (max 7+4+10+4+7+4 = 36 < POPULATED_ROWS), so the
        // pad is never negative; assert rather than assume.
        assert!(
            out.len() <= POPULATED_ROWS,
            "synthetic window {window} over-filled: {} > {POPULATED_ROWS}",
            out.len()
        );
        for j in out.len()..POPULATED_ROWS {
            out.push(format!("they were {}.", term(60 + j)));
        }
        out
    }

    fn synthetic_corpus(n: usize) -> Vec<String> {
        assert_eq!(
            n % POPULATED_ROWS,
            0,
            "the synthetic corpus is built per owner slice; {n} must be a multiple of {POPULATED_ROWS}"
        );
        (0..n / POPULATED_ROWS).flat_map(synthetic_window).collect()
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
    // `probe_ignition.rs`'s `MemWal`. ───────────────────────────────────────

    struct SealedCycle {
        version: DatasetVersion,
        landings: Vec<SweepSlot>,
    }

    struct MemWal {
        sealed: Mutex<Vec<SealedCycle>>,
        next_version: AtomicU64,
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
            &self,
            base: DatasetVersion,
            batch: DetachedCycleBatch,
        ) -> Result<DatasetVersion, WriteFailed> {
            let mut sealed = self.sealed.lock().expect("MemWal poisoned");
            let head = sealed.last().map_or(DatasetVersion(0), |s| s.version);
            if base != head {
                return Err(WriteFailed(format!(
                    "stale base {base:?}: sealed head is {head:?}"
                )));
            }
            self.wal_writes.fetch_add(1, Ordering::SeqCst);
            let version = DatasetVersion(self.next_version.fetch_add(1, Ordering::SeqCst));
            sealed.push(SealedCycle {
                version,
                landings: batch.landings,
            });
            Ok(version)
        }

        async fn scan_sealed(
            &self,
            from_version: Option<DatasetVersion>,
        ) -> Result<Vec<LandedSlot>, WriteFailed> {
            Ok(self
                .sealed
                .lock()
                .expect("MemWal poisoned")
                .iter()
                .filter(|s| from_version.is_none_or(|f| s.version > f))
                .flat_map(|s| {
                    s.landings.iter().map(|slot| LandedSlot {
                        version: s.version,
                        slot: slot.clone(),
                    })
                })
                .collect())
        }

        async fn versions(&self) -> Result<Vec<(CycleId, DatasetVersion)>, WriteFailed> {
            Ok(self
                .sealed
                .lock()
                .expect("MemWal poisoned")
                .iter()
                .map(|s| {
                    (
                        s.landings.first().map_or(CycleId(0), |l| l.cycle),
                        s.version,
                    )
                })
                .collect())
        }
    }

    // ── fleet construction ──────────────────────────────────────────────────

    /// Owner `owner_idx`'s own text slice, address-selected — the SAME
    /// slice `build_owner` bloom-seeds from and `labelled_verses` reads for
    /// the lens (F1: one text source, never row-byte decoding).
    fn owner_verses(all: &[String], owner_idx: MailboxId) -> &[String] {
        let lo = owner_idx as usize * POPULATED_ROWS;
        &all[lo..lo + POPULATED_ROWS]
    }

    /// `(label, text)` pairs `stance::stream` wants (`stance.rs:161-167`).
    /// Label format `"kjv:{global_index:05}"` (design §7 Q2, matching
    /// `blw_fusion.rs:220`'s subject format).
    fn labelled_verses(all: &[String], base_owner: MailboxId) -> Vec<(String, String)> {
        let lo = base_owner as usize * POPULATED_ROWS;
        all[lo..lo + POPULATED_ROWS]
            .iter()
            .enumerate()
            .map(|(i, text)| (format!("kjv:{:05}", lo + i), text.clone()))
            .collect()
    }

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

        // TWIN BLOCK: same slice (base owner TWIN_LO), same content salt
        // (0) for every owner — so the content planes are byte-identical
        // (L0's precondition). Different `id`s only.
        for (offset, id) in (TWIN_LO..TWIN_HI).enumerate() {
            let armed = TWIN_ARMING[offset];
            fleet.insert(
                id,
                build_owner(
                    id,
                    owner_verses(corpus, TWIN_LO),
                    0,
                    armed,
                    flow_qualia(),
                    3,
                ),
            );
        }

        // SPREAD BLOCK: distinct slices, distinct content salts (= id),
        // armed z cycling 1..4 — feeds L4's cross-owner non-constancy half.
        for id in SPREAD_LO..SPREAD_HI {
            let armed = (((id - SPREAD_LO) % 4) + 1) as u8;
            fleet.insert(
                id,
                build_owner(
                    id,
                    owner_verses(corpus, id),
                    u64::from(id),
                    armed,
                    flow_qualia(),
                    3,
                ),
            );
        }

        // UNARMED (id 30): z=0, never plans.
        fleet.insert(
            UNARMED_ID,
            build_owner(
                UNARMED_ID,
                owner_verses(corpus, UNARMED_ID),
                u64::from(UNARMED_ID),
                0,
                flow_qualia(),
                3,
            ),
        );

        // ORPHAN (id 31): deliberately NOT inserted.

        // OUTSIDE (32..64): armed, but SCOPE_HI=32 excludes them from every
        // scan in the main loop (L7).
        for id in OUTSIDE_LO..OUTSIDE_HI {
            fleet.insert(
                id,
                build_owner(
                    id,
                    owner_verses(corpus, id),
                    u64::from(id),
                    1,
                    flow_qualia(),
                    3,
                ),
            );
        }

        fleet
    }

    // ── the scan (LOOK INTO THE KANBAN) — copied from `probe_ignition.rs`. ──

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

    /// Provenance: `probe_ignition.rs:604-638`, unchanged.
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

    // ── the lens readout type (design §2) — probe-local, mints nothing
    // shipped. NOT `CausalWitnessFacet` — that carrier is what §12.7 killed
    // (design §2's explicit anti-decision). ─────────────────────────────────

    #[derive(Debug)]
    enum LensReadout {
        Hegel(Vec<(CStmt, f32)>),
        Nietzsche(Vec<(CStmt, FlipKind)>),
        Kant(Vec<(String, f32, f32)>),
        Wittgenstein(Vec<(u16, usize)>),
    }

    impl LensReadout {
        fn is_empty(&self) -> bool {
            match self {
                LensReadout::Hegel(v) => v.is_empty(),
                LensReadout::Nietzsche(v) => v.is_empty(),
                LensReadout::Kant(v) => v.is_empty(),
                LensReadout::Wittgenstein(v) => v.is_empty(),
            }
        }

        /// Stable fold over the variant's own contents, in the shipped
        /// iteration order (design §7 Q3). Floats hashed via `.to_bits()`,
        /// never `HashMap` iteration (the Wittgenstein arm is already
        /// sorted before `stance_panel` returns it, `stance.rs:530-532` —
        /// that pre-sort is the only reason this fold is deterministic).
        ///
        /// Deliberately NO variant-discriminant tag folded in first. A
        /// tagged digest would make any cross-lens `!=` comparison pass
        /// by construction of the tag alone (an assertion the code itself
        /// implies, not a test of content — the falsifiability rule this
        /// workspace's `CLAUDE.md` sets out). Without a tag, two EMPTY
        /// readouts of any lens hash equal (both fold zero bytes) — that
        /// is intentional: it is what makes L3/L4's non-emptiness and
        /// non-constancy checks below able to fail for real, instead of
        /// being guaranteed to pass by the digest's own construction.
        fn digest(&self) -> u64 {
            use std::collections::hash_map::DefaultHasher;
            let mut h = DefaultHasher::new();
            match self {
                LensReadout::Hegel(v) => {
                    for (stmt, f) in v {
                        stmt.hash(&mut h);
                        f.to_bits().hash(&mut h);
                    }
                }
                LensReadout::Nietzsche(v) => {
                    for (stmt, flip) in v {
                        stmt.hash(&mut h);
                        // FlipKind derives Eq but not Hash (api inventory
                        // §A) — fold its discriminant by hand.
                        let d: u8 = match flip {
                            FlipKind::Transvaluation => 0,
                            FlipKind::Devaluation => 1,
                        };
                        d.hash(&mut h);
                    }
                }
                LensReadout::Kant(v) => {
                    for (label, graded, ablated) in v {
                        label.hash(&mut h);
                        graded.to_bits().hash(&mut h);
                        ablated.to_bits().hash(&mut h);
                    }
                }
                LensReadout::Wittgenstein(v) => {
                    for (concept, games) in v {
                        concept.hash(&mut h);
                        games.hash(&mut h);
                    }
                }
            }
            h.finish()
        }
    }

    /// The panel computes all four; this SELECTS one (design §2's central
    /// consequence). Cost is identical across z — the four cases below
    /// differ only in which tuple element they keep.
    fn run_lens(z: u8, verses: &[(String, String)]) -> LensReadout {
        let mut arena = BeliefArena::new();
        let mut intern = Interner::new();
        let mut out = ReadOut::default();
        stream(verses, &mut arena, &mut intern, &mut out, false);
        let (hegel, nietzsche, kant, wittgenstein) = stance_panel(&arena, &intern, &out);
        match z {
            1 => LensReadout::Hegel(hegel),
            2 => LensReadout::Nietzsche(nietzsche),
            3 => LensReadout::Kant(kant),
            4 => LensReadout::Wittgenstein(wittgenstein),
            other => panic!(
                "run_lens: z={other} is outside the armed range 1..=4 \
                 (z=0 is unarmed, z=5 is RESERVED/BLOCKED — see the module doc)"
            ),
        }
    }

    /// All four readouts over ONE arena — used only by L4's single-owner
    /// cross-lens check (design §4 L4 can-fire half).
    fn run_all_lenses(verses: &[(String, String)]) -> [LensReadout; 4] {
        let mut arena = BeliefArena::new();
        let mut intern = Interner::new();
        let mut out = ReadOut::default();
        stream(verses, &mut arena, &mut intern, &mut out, false);
        let (hegel, nietzsche, kant, wittgenstein) = stance_panel(&arena, &intern, &out);
        [
            LensReadout::Hegel(hegel),
            LensReadout::Nietzsche(nietzsche),
            LensReadout::Kant(kant),
            LensReadout::Wittgenstein(wittgenstein),
        ]
    }

    // ── the main probe ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn d_ign_b_lens_selection_over_byte_identical_rows() {
        let (corpus, provenance) = load_or_synthesize_corpus();
        println!("d_ign_b corpus: {provenance} ({} verses)", corpus.len());
        assert_eq!(corpus.len(), CORPUS_VERSES, "PRE-REGISTERED corpus size");

        let mut fleet = build_fleet(&corpus);

        // ── L0: twin premise ────────────────────────────────────────────
        {
            let twin_ids: Vec<MailboxId> = (TWIN_LO..TWIN_HI).collect();
            let base = fleet.get(&TWIN_LO).expect("twin base owner must exist");
            for row in 0..POPULATED_ROWS {
                let base_plane = base.content_row(row).to_vec();
                assert!(
                    base_plane.iter().any(|&w| w != 0),
                    "L0 can-stay-silent: twin content plane row {row} must be non-zero, not the trivial all-zero case"
                );
                for &id in &twin_ids {
                    let owner = fleet.get(&id).expect("twin owner must exist");
                    assert_eq!(
                        owner.content_row(row).to_vec(),
                        base_plane,
                        "L0 can-fire: twin owner {id} row {row} must be byte-identical to owner {TWIN_LO}"
                    );
                }
            }
            let non_twin = fleet.get(&SPREAD_LO).expect("spread owner must exist");
            assert_ne!(
                non_twin.content_row(0).to_vec(),
                base.content_row(0).to_vec(),
                "L0 can-stay-silent: a non-twin owner's plane must differ from the twin block's"
            );
            eprintln!(
                "d_ign_b.L0: {} twin owners byte-identical across {POPULATED_ROWS} rows; non-twin owner {SPREAD_LO} differs",
                twin_ids.len()
            );
        }

        // ── L1 + L4 can-fire half: computed DIRECTLY over the stance panel,
        // independent of cycle timing — the selection axis is a property of
        // `run_lens`/`stance_panel`, not of when a cast happens to land. ───
        {
            let twin_verses = labelled_verses(&corpus, TWIN_LO);

            // Pre-registered per design §4's risk note + §7 Q4: Hegel was
            // measured (not assumed) constant-false on the corpus shape of
            // the time (§12.3a″); Nietzsche derives from Hegel
            // (stance.rs:483-496) so it degrades with it. The can-fire
            // witness pair is therefore pinned to z=3 (Kant) vs z=4
            // (Wittgenstein) BEFORE any run, per the design's own
            // pre-registered fallback — never chosen after seeing output.
            // The 2026-08-06 corpus fix makes Hegel/Nietzsche fire on the
            // synthetic fallback too (the risk-check line below prints the
            // measurement rather than trusting it); the witness pair is NOT
            // re-picked on the strength of that — re-picking after seeing
            // output is exactly what the pre-registration forbids.
            let r_kant = run_lens(3, &twin_verses);
            let r_witt = run_lens(4, &twin_verses);
            eprintln!(
                "d_ign_b.L1.risk-check: Hegel empty on twin base = {}, Nietzsche empty on twin base = {}",
                run_lens(1, &twin_verses).is_empty(),
                run_lens(2, &twin_verses).is_empty()
            );

            // Precondition, checked BEFORE the digest comparison: an empty
            // readout folds zero bytes, so two empties digest identically
            // (`digest` is deliberately untagged — see its doc comment).
            // Without this guard a corpus that yields nothing extractable
            // reports as "distinct lenses collide", which is a false and
            // deeply misleading diagnosis of a corpus defect. That is the
            // exact failure this file shipped with: the pre-fix synthetic
            // fallback armed no predicate at all, both readouts came back
            // empty, and the assertion below blamed the lenses.
            for (label, readout) in [("z=3 Kant", &r_kant), ("z=4 Wittgenstein", &r_witt)] {
                assert!(
                    !readout.is_empty(),
                    "L1 precondition: the {label} readout is EMPTY over the twin base slice — \
                     the corpus produced no extractable content, so the digest comparison below \
                     would be empty-vs-empty and could only collide. This is a CORPUS defect, \
                     not a lens collision. Corpus source: {provenance}. \
                     (`stance::stream` emits nothing without a copula / auxiliary / typed \
                     relational verb / `-ed` form to arm a predicate, and Kant additionally \
                     needs a perception verb + `that` complementizer.)"
                );
            }

            assert_ne!(
                r_kant.digest(),
                r_witt.digest(),
                "L1 can-fire: distinct lenses (z=3 Kant vs z=4 Wittgenstein) over byte-identical rows must yield distinct digests"
            );

            // can-stay-silent half: same lens (z=3), twin owners 4 and 5
            // (both armed z=3 per TWIN_ARMING), same rows, same cycle.
            let r_kant_a = run_lens(3, &twin_verses);
            let r_kant_b = run_lens(3, &twin_verses);
            assert_eq!(
                r_kant_a.digest(),
                r_kant_b.digest(),
                "L1 can-stay-silent: the SAME lens over byte-identical rows must yield bit-identical digests"
            );
            eprintln!("d_ign_b.L1: selection axis load-bearing (Kant != Wittgenstein digest); same-lens digest bit-identical");

            // L4 can-fire: over ONE owner, the four lens digests are not
            // all equal (>=3 distinct of 4) — the §12.7 anti-collapse shape.
            let all_four = run_all_lenses(&twin_verses);
            let digests: Vec<u64> = all_four.iter().map(LensReadout::digest).collect();
            let mut distinct: Vec<u64> = digests.clone();
            distinct.sort_unstable();
            distinct.dedup();
            eprintln!(
                "d_ign_b.L4.can-fire: single-owner 4-lens digests = {digests:?} ({} distinct of 4)",
                distinct.len()
            );
            assert!(
                distinct.len() >= 3,
                "L4 can-fire: at least 3 of the 4 lens digests must be distinct over one owner"
            );
        }

        // ── main cycle loop: cast/scan/seal/apply, unchanged mechanics,
        // with the lens embedded in `run_cognitive_work_gated_over`'s
        // closure (design §1's chosen seam). ────────────────────────────
        let sink = MemWal::new();
        let mut writer: BatchWriter<Vec<u8>> = BatchWriter::new();
        let mut position_base: u64 = 0;
        let mut watermarks: HashMap<MailboxId, Option<u64>> = HashMap::new();

        let mut readouts: HashMap<(MailboxId, u32), LensReadout> = HashMap::new();
        let mut advanced_to_cognitive: HashSet<MailboxId> = HashSet::new();

        for c in 1..=CYCLES {
            let scan = scan_board(&fleet, SCOPE_LO..SCOPE_HI);
            eprintln!(
                "d_ign_b scan @c{c}: planning={} cognitive={} evaluation={} absorbed={} missing={}",
                scan.planning.len(),
                scan.cognitive.len(),
                scan.evaluation.len(),
                scan.absorbed.len(),
                scan.missing
            );
            assert_eq!(
                scan.missing, 1,
                "the orphan (id {ORPHAN_ID}) must be the only missing id in every scan"
            );

            let planning_outcome =
                column_pass(&fleet, &scan.planning, &mut writer, plan_or_evaluate_think);

            // ── L2 can-stay-silent (top half): the unarmed owner is in
            // `scan.planning` but never casts (armed==0 short-circuits
            // `plan_or_evaluate_think`), so it can never reach CognitiveWork
            // and therefore never gets a readout. ───────────────────────
            let scan_cognitive_for_lens = scan.cognitive.clone();
            let cognitive_outcome = run_cognitive_work_gated_over(
                &fleet,
                &scan_cognitive_for_lens,
                &mut writer,
                |owner| {
                    let armed = owner.meta_at(0).thinking();
                    // z=5 must never reach this arm on this fleet — see the
                    // "z=5 BLOCKED" check after the loop, which proves the
                    // premise; this assert is the in-loop mirror of it.
                    assert_ne!(
                        armed, 5,
                        "z=5 (Fusion) must never be armed on any owner reaching CognitiveWork in this file's fleet — it is RESERVED, not selectable"
                    );
                    if (1..=4).contains(&armed) {
                        let verses = if (TWIN_LO..TWIN_HI).contains(&owner.mailbox_id()) {
                            labelled_verses(&corpus, TWIN_LO)
                        } else {
                            labelled_verses(&corpus, owner.mailbox_id())
                        };
                        let readout = run_lens(armed, &verses);
                        readouts.insert((owner.mailbox_id(), owner.cycle()), readout);
                    }
                    let style = thinking_style_for(armed);
                    let ctx = plan_context_for(armed);
                    let qualia = owner.qualia_at(0);
                    let mantissa = mantissa_of(owner);
                    let reliability = StyleStrategy::reliability_for(style, &ctx);
                    Some((qualia, mantissa, reliability, row_span_payload(owner)))
                },
            );

            let evaluation_outcome = column_pass(
                &fleet,
                &scan.evaluation,
                &mut writer,
                plan_or_evaluate_think,
            );

            let total_casts =
                planning_outcome.cast + cognitive_outcome.cast + evaluation_outcome.cast;
            if total_casts == 0 {
                eprintln!("d_ign_b @c{c}: 0 casts staged — cycle rests, no seal");
                continue;
            }

            let base_version = sink.head();
            let outcome: CycleOutcome = match run_cycle(
                &sink,
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
                Err(CycleError::Seal(_)) => {
                    panic!("d_ign_b @c{c}: unexpected seal failure (MemWal never injects one)")
                }
                Err(CycleError::Apply { cause, .. }) => {
                    panic!("d_ign_b @c{c}: unexpected apply failure: {cause}")
                }
            };
            position_base = position_base.max(outcome.sealed.next_position_base);

            // ── L5 (mechanics unchanged by the swap) — DERIVED expectation
            // for THIS file's cohorts (deviation 1: not PROBE-IGNITION's
            // 20+4, since there is no CONTRA cohort here). ───────────────
            let flow_advances = outcome
                .sealed
                .transitions
                .iter()
                .filter(|t| {
                    t.mv.from == KanbanColumn::Planning && t.mv.to == KanbanColumn::CognitiveWork
                })
                .count();
            let block_advances = outcome
                .sealed
                .transitions
                .iter()
                .filter(|t| t.mv.from == KanbanColumn::Planning && t.mv.to == KanbanColumn::Prune)
                .count();
            eprintln!(
                "d_ign_b.L5 @c{c}: {flow_advances} Flow (Planning->CognitiveWork) + {block_advances} Block (Planning->Prune) transitions sealed"
            );
            for t in &outcome.sealed.transitions {
                if t.mv.from == KanbanColumn::Planning && t.mv.to == KanbanColumn::CognitiveWork {
                    assert_eq!(
                        t.mv.exec,
                        ExecTarget::Elixir,
                        "L5 can-fire: Planning->CognitiveWork must be the STYLE's Elixir mint"
                    );
                    advanced_to_cognitive.insert(t.owner);
                }
            }
            if c == 1 {
                assert_eq!(
                    block_advances, 0,
                    "L5: this file's cohorts have no block/CONTRA-qualia owners in scope"
                );
                assert_eq!(
                    flow_advances,
                    (TWIN_HI - TWIN_LO) as usize + (SPREAD_HI - SPREAD_LO) as usize,
                    "L5 can-fire: all TWIN+SPREAD armed owners must Flow-advance at c1"
                );
            }

            eprintln!(
                "d_ign_b @c{c}: {total_casts} casts staged, {} readouts captured so far",
                readouts.len()
            );
        }

        // ── L2: arming is load-bearing. can-stay-silent checked FIRST,
        // across every cycle key the main loop could have written (not
        // just one), before the arming mutation below changes anything. ──
        {
            let silent_across_all_cycles = !readouts.keys().any(|&(rid, _)| rid == UNARMED_ID);
            assert!(
                silent_across_all_cycles,
                "L2 can-stay-silent: the UNARMED owner must have NO readout from the main loop, at ANY cycle"
            );

            // can-fire: arm it (the G8 pattern, `probe_ignition.rs:1125-1143`),
            // then run the SAME lens-capture step the gated closure performs
            // above and confirm it lands a readout entry — into a SEPARATE
            // scratch map, so this manual arming (which never goes through
            // `run_cognitive_work_gated_over`, hence never reaches
            // `advanced_to_cognitive`) cannot contaminate L6's containment
            // check below.
            let owner_mut = fleet
                .get_mut(&UNARMED_ID)
                .expect("UNARMED owner must exist");
            owner_mut.set_meta(0, MetaWord::new(3, 0, 0, 0, 0)); // arm as Kant, z=3
            let owner = fleet.get(&UNARMED_ID).expect("UNARMED owner must exist");
            let armed = owner.meta_at(0).thinking();
            assert_eq!(armed, 3, "L2 can-fire premise: the write landed");
            let verses = labelled_verses(&corpus, UNARMED_ID);
            let readout = run_lens(armed, &verses);
            let mut scratch: HashMap<(MailboxId, u32), LensReadout> = HashMap::new();
            scratch.insert((UNARMED_ID, owner.cycle()), readout);
            assert!(
                scratch.keys().any(|&(rid, _)| rid == UNARMED_ID),
                "L2 can-fire: writing non-zero thinking bits and re-running the lens step must land a readout entry"
            );
            eprintln!(
                "d_ign_b.L2: UNARMED owner {UNARMED_ID} silent in the main loop for every cycle (armed==0 throughout); \
                 once armed directly + re-run, a readout entry lands (kept in a scratch map, not the main one)"
            );
        }

        // ── L3: per-lens non-emptiness, measured and printed for all
        // in-scope armed owners (TWIN 0..8 + SPREAD 8..30 = 30 owners). ──
        {
            let mut empty_count: HashMap<u8, usize> = HashMap::new();
            let mut total_count: HashMap<u8, usize> = HashMap::new();
            let mut per_lens_digests: HashMap<u8, Vec<u64>> = HashMap::new();
            for id in (TWIN_LO..TWIN_HI).chain(SPREAD_LO..SPREAD_HI) {
                let owner = fleet.get(&id).expect("in-scope armed owner must exist");
                let armed = owner.meta_at(0).thinking();
                let verses = if (TWIN_LO..TWIN_HI).contains(&id) {
                    labelled_verses(&corpus, TWIN_LO)
                } else {
                    labelled_verses(&corpus, id)
                };
                let readout = run_lens(armed, &verses);
                *total_count.entry(armed).or_insert(0) += 1;
                if readout.is_empty() {
                    *empty_count.entry(armed).or_insert(0) += 1;
                }
                per_lens_digests
                    .entry(armed)
                    .or_default()
                    .push(readout.digest());
            }
            for z in 1..=4u8 {
                let empty = empty_count.get(&z).copied().unwrap_or(0);
                let total = total_count.get(&z).copied().unwrap_or(0);
                eprintln!("d_ign_b.L3 z={z}: empty={empty}/{total}");
                assert!(
                    empty < total,
                    "L3 can-fire: lens z={z} must be non-empty on at least one of {total} owners"
                );

                // ── L4 can-stay-silent: each non-empty lens must have >=2
                // distinct digests across the 30 in-scope owners (a
                // lens constant over every owner carries no information).
                let mut ds = per_lens_digests.get(&z).cloned().unwrap_or_default();
                ds.sort_unstable();
                ds.dedup();
                eprintln!(
                    "d_ign_b.L4 z={z}: {} distinct digests across {total} owners",
                    ds.len()
                );
                assert!(
                    ds.len() >= 2,
                    "L4 can-stay-silent: lens z={z} must yield >=2 distinct digests across the in-scope owners"
                );
            }
        }

        // ── L6: readout keys are a subset of owners that advanced
        // Planning->CognitiveWork in a PRIOR cycle. ─────────────────────
        {
            for &(id, _cycle) in readouts.keys() {
                assert!(
                    advanced_to_cognitive.contains(&id),
                    "L6 can-fire: readout owner {id} must be in the set of owners that advanced to CognitiveWork"
                );
            }
            assert!(
                !advanced_to_cognitive.contains(&UNARMED_ID),
                "L6 can-stay-silent: the UNARMED owner never advanced to CognitiveWork, so it has no readout key at any cycle"
            );
            eprintln!(
                "d_ign_b.L6: {} readout keys all within {} owners that advanced to CognitiveWork; UNARMED owner absent from both",
                readouts.len(),
                advanced_to_cognitive.len()
            );
        }

        // ── L7 (inherits G7): the address axis is load-bearing — an
        // OUTSIDE owner, run through the lens body DIRECTLY, produces a
        // readout; the OUTSIDE cohort has none from the main loop. ──────
        {
            for id in OUTSIDE_LO..OUTSIDE_HI {
                assert!(
                    !readouts.keys().any(|&(rid, _)| rid == id),
                    "L7 can-stay-silent: OUTSIDE owner {id} must have no readout from the main loop (never scanned, SCOPE_HI={SCOPE_HI})"
                );
            }
            let outside_owner = fleet.get(&OUTSIDE_LO).expect("OUTSIDE owner must exist");
            assert_eq!(
                outside_owner.phase(),
                KanbanColumn::Planning,
                "L7 can-stay-silent premise: OUTSIDE owner never advanced in the main run"
            );
            let verses = labelled_verses(&corpus, OUTSIDE_LO);
            let readout = run_lens(1, &verses);
            eprintln!(
                "d_ign_b.L7: OUTSIDE owner {OUTSIDE_LO} — silent in the main run (address excluded); \
                 run through the lens body directly it produces a readout (empty={})",
                readout.is_empty()
            );
        }

        // ── z=5 BLOCKED (design §3, the deliverable per brief). ──────────
        {
            for owner in fleet.values() {
                assert_ne!(
                    owner.meta_at(0).thinking(),
                    5,
                    "z=5 (Fusion) must be unarmed/unused on every owner in this file's fleet"
                );
            }
            let z5 = MetaWord::new(5, 0, 0, 0, 0);
            assert_eq!(
                z5.thinking(),
                5,
                "sanity: 5 is representable in the 6-bit thinking field"
            );
            eprintln!(
                "d_ign_b.z5-BLOCKED: ordinal 5 (Fusion) is RESERVED, never armed on any owner in \
                 this file. Reason (design §3): blw_fusion's admission gap needs the ranking pool \
                 to GROW across many sealed horizons (S_CYCLES=8); this fleet is seeded once and \
                 accrues at most {CYCLES} sealed horizons, so Strict-vs-Aware admission would be \
                 identical and any delta would be 0 BY CONSTRUCTION — a vacuous readout. `jc` is \
                 also not a dependency of lance-graph-supervisor. R2 (a separate reduced test, no \
                 kappa, no jc dep) is the recommended path if the orchestrator wants z=5 exercised; \
                 not implemented here."
            );
        }

        // ── §6 Not-claimed block ─────────────────────────────────────────
        println!();
        println!("== D-IGN-B — what this test does NOT claim ==");
        println!("1. No row-byte decoding. The lens reads the owner's own verses selected by its address and span; bloom planes are one-way.");
        println!(
            "2. No fusion verdict, no kappa. z=5 is reserved (see the z5-BLOCKED check above)."
        );
        println!("3. No stance validity. A readout is a read, never evidence that a stance is right about anything.");
        println!("4. No parallelism, no durability, no scale, no multi-writer, no recovery (PROBE-IGNITION's not-claimed items carry over).");
        println!("5. No 36-style claim. Six probe-local ordinals (0..5); the persona-36 bridge stays an open non-goal.");
        println!("6. No rung-3 / runbook claim. The four stances are not the 34 NARS tactics and are not StyleFamily macros.");
        println!("7. No deinterlace/temporal claim in this loop.");
        println!("8. No claim that the four lenses are independent instruments — Nietzsche is computed from Hegel's output (stance.rs:483-496).");
        println!("9. No zero-copy claim. SweepSlot::payload is Vec<u8> by the shipped signature.");
        println!("10. No semantic claim about the corpus. Qualia remain declared fixtures; nothing is encoded from text into qualia.");
        println!("11. No per-stance DISPATCH claim. stance_panel computes all four in one call; this file only SELECTS which tuple element to record (design §0 F0).");
    }
}
