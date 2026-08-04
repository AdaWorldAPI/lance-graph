//! D-BLW-1 — the tiled 64k verse fleet + the lens body in the cycle-driver
//! seam.
//!
//! ## Why TILED, not one `MailboxSoA<65536>`
//!
//! The corpus is ONE bake, **TILED** across K mailbox owners of
//! `MailboxSoA<1024>`. 64 tiles × 1024 rows = 65,536 verse rows. Three
//! reasons this is the shape, not a single giant SoA:
//!
//! - **Size is invariant to tiling.** `MailboxSoA<N>` allocates
//!   `content`+`topic`+`angle` as `3 × N × WORDS_PER_FP(256) × 8 B` =
//!   6,144 B/row, so 65,536 rows is **384 MiB** no matter how it is tiled.
//!   Tiling only changes how the 384 MiB is *addressed* (K owners vs one).
//! - **`MailboxSoA::new` is a large by-value construction.** A single
//!   `MailboxSoA<65536>` would put ~5 MB of the const-sized columns
//!   (`energy`, `plasticity_counter`, `last_active_cycle`, …) on the stack
//!   during `new()`'s return-by-value, which can overflow a test thread's
//!   stack. Tiling at N=1024 keeps each construction's stack-resident
//!   portion ~84 KB.
//! - **A sparse sealed transition set ("17 dirty, not 64k") is a sparse
//!   set of *owners*.** One `MailboxSoA` is one mailbox owner
//!   (`MailboxFleet::Owner`), so a single giant SoA cannot express "most
//!   owners are byte-identical" at all — there is only one owner to be
//!   dirty or clean. Tiling is a **partition of one corpus** into many
//!   owners, not a second projection of it, so it does not disturb the
//!   zero-copy ruling that rejected a 6-SoA (one-per-lens) shape: this is
//!   still exactly one bake, just addressed through K owner keys instead
//!   of one.
//!
//! ## What this file re-anchors, not what it discovers
//!
//! **The mechanical property — a sealed cycle applies only its sparse
//! transition set, and every unrepresented owner is byte-identical after —
//! is ALREADY proven at 64k scale**, in `cycle_driver.rs`'s own test module:
//! `p4b_applies_only_the_sealed_sparse_set_64k_of_17_advance_rest_byte_identical`
//! (line ~1098, `const FLEET: u32 = 65_536`, 17 represented owners, a cloned
//! `before` fleet, a per-owner byte-identical assertion). That test proves
//! the mechanism over the lightweight in-file `FakeOwner`.
//!
//! What this file adds is two things `FakeOwner` structurally cannot
//! exercise: (1) the driver run over the **production `MailboxSoA` owner**
//! instead of the fake, and (2) a **real lens body** in the `CognitiveWork`
//! seam that actually reads an owner's row slice
//! (`MailboxSoaView::energy()`) to decide whether to cast — `FakeOwner`
//! carries no row columns at all, so no lens reading real data could ever
//! run over it. This is the same precedented gap-closure as
//! `tests/w2b_real_owner_probe.rs`, which exists for the identical reason on
//! the actor side ("`KanbanActor<O>` was only ever exercised against
//! `TestBoard`"). The tests below are named `..._over_the_real_mailbox_soa`
//! for this reason: they are RE-PROOFS on the real owner + a real lens, not
//! first proofs of the sparse-set mechanism itself.

#[cfg(feature = "cycle-driver")]
mod blw_bible_lens_wave {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    use cognitive_shader_driver::mailbox_soa::MailboxSoA;
    use lance_graph_contract::collapse_gate::MailboxId;
    use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
    use lance_graph_contract::scheduler::DatasetVersion;
    use lance_graph_contract::soa_view::MailboxSoaView;
    use lance_graph_planner::batch_writer::BatchWriter;
    use lance_graph_planner::persist_sink::{
        CycleFrame, CycleId, DetachedCycleBatch, LandedSlot, SweepSlot, WalSink, WriteFailed,
    };
    use lance_graph_planner::traits::StrategyOutcome;
    use lance_graph_supervisor::cycle_driver::{run_cognitive_work, run_cycle};

    // ── The one number the CI test and the full-scale test share ───────────────

    /// Rows per tile — `MailboxSoA<TILE_ROWS>`. Fixed at the default capacity
    /// (`cognitive_shader_driver::mailbox_soa::DefaultMailboxSoA`'s width).
    const TILE_ROWS: usize = 1024;

    /// The full KJV-scale tile count: 64 tiles × 1024 rows = 65,536 verse
    /// rows. THE number — `CI_TILES` below is derived from it so the CI test
    /// and the ignored full-scale test relate by one division, not a scatter
    /// of independent literals.
    const FULL_TILES: usize = 64;

    /// A CI-tractable slice of the same shape: 8 tiles × 1024 rows = 8,192
    /// rows (≈48 MiB). `FULL_TILES / 8` so the relationship to the full-scale
    /// constant above is explicit, not a coincidence of two hand-picked
    /// numbers.
    const CI_TILES: usize = FULL_TILES / 8;

    type Tile = MailboxSoA<TILE_ROWS>;
    type Fleet = HashMap<MailboxId, Tile>;

    // ── Deterministic seeding (no RNG, no clock) ────────────────────────────────

    /// Deterministic per-(tile, row) `entity_type` seed. Varies by both tile
    /// and row so the plane is not merely a repeated constant.
    fn seed_entity_type(tile: usize, row: usize) -> u16 {
        ((tile * 37 + row) % 4096) as u16
    }

    /// Deterministic per-(tile, row) `energy` seed. `row == 0` is seeded to
    /// EXACTLY `tile as f32` — the lens predicate below reads this value to
    /// recover which tile it is looking at (a genuine read through
    /// `MailboxSoaView::energy()`, never a shortcut through the tile index
    /// itself). Every other row carries a small deterministic offset so the
    /// whole 1024-row plane is populated, not just row 0.
    fn seed_energy(tile: usize, row: usize) -> f32 {
        tile as f32 + (row as f32) * 1e-4
    }

    /// Bake `n_tiles` mailbox owners of `MailboxSoA<TILE_ROWS>` — the corpus
    /// is ONE bake, tiled across K owners (see the module doc for why tiling,
    /// not a single N=65536 SoA, is the shape). `w_slot = tile % 64` (the
    /// 6-bit W-slot constraint — real for a fleet this wide: a 64-tile fleet
    /// uses every slot exactly once). Rows are seeded deterministically via
    /// [`seed_entity_type`] / [`seed_energy`] — no RNG, no clock — so the
    /// fixture is reproducible byte-for-byte across runs. Returns the fleet
    /// and the total verse-row count (`n_tiles * TILE_ROWS`).
    fn bake_tiles(n_tiles: usize) -> (Fleet, usize) {
        let mut fleet: Fleet = HashMap::with_capacity(n_tiles);
        for tile in 0..n_tiles {
            let mailbox_id = tile as MailboxId;
            let w_slot = (tile % 64) as u8;
            let mut mb: Tile = MailboxSoA::new(mailbox_id, w_slot, 1.0);
            // W1c discipline: declare the logical row count before use.
            mb.set_populated(TILE_ROWS);
            for row in 0..TILE_ROWS {
                mb.set_entity_type(row, seed_entity_type(tile, row));
                mb.energy[row] = seed_energy(tile, row);
            }
            fleet.insert(mailbox_id, mb);
        }
        (fleet, n_tiles * TILE_ROWS)
    }

    // ── Move helpers (mirrors cycle_driver.rs's own `mv()` / `sentinel()`) ─────

    /// A move a test harness casts directly, naming the live owner (used to
    /// stage the initial `Planning -> CognitiveWork` casts).
    fn mv(owner: MailboxId, from: KanbanColumn, to: KanbanColumn) -> KanbanMove {
        KanbanMove {
            mailbox: owner,
            from,
            to,
            witness_chain_position: 0,
            exec: ExecTarget::Native,
        }
    }

    /// A bootstrap-sentinel move (`mailbox 0`, `witness_chain_position 0`)
    /// that `owner_adapter::emit_bootstrap_intent` rebinds to the live owner
    /// — the shape a lens body's `StrategyOutcome::intended_move` must be.
    fn sentinel(from: KanbanColumn, to: KanbanColumn) -> KanbanMove {
        KanbanMove {
            mailbox: 0,
            from,
            to,
            witness_chain_position: 0,
            exec: ExecTarget::Native,
        }
    }

    /// Stage one `Planning -> CognitiveWork` cast per owner.
    fn stage_planning_to_cognitive(owners: &[MailboxId]) -> BatchWriter<Vec<u8>> {
        let mut w: BatchWriter<Vec<u8>> = BatchWriter::new();
        for &id in owners {
            w.cast(
                id,
                vec![mv(id, KanbanColumn::Planning, KanbanColumn::CognitiveWork)],
                vec![0xAB],
            );
        }
        w
    }

    // ── The lens body ────────────────────────────────────────────────────────

    /// The lens: reads the owner's row-0 energy (a genuine read through
    /// [`MailboxSoaView::energy`] — never a mutation, never a shortcut
    /// through the tile index the caller happens to know) and casts a
    /// `CognitiveWork -> Evaluation` bootstrap intent for every owner whose
    /// recovered tile index is a multiple of three. `None` (held) for every
    /// other owner — both the fire and the stay-silent path are real on any
    /// fleet with at least 3 tiles (this fixture always has ≥ 3).
    fn tile_divisible_by_three_lens(owner: &Tile) -> Option<(StrategyOutcome, Vec<u8>)> {
        let energy = owner.energy(); // &[f32] — a read via MailboxSoaView
        let row0 = *energy.first().expect("a populated tile has row 0");
        let tile_index = row0 as i64; // seeded to `tile as f32` exactly (see seed_energy)
        if tile_index % 3 != 0 {
            return None; // held — the can-stay-silent half
        }
        let outcome = StrategyOutcome {
            reliability: 0.9,
            intended_move: Some(sentinel(
                KanbanColumn::CognitiveWork,
                KanbanColumn::Evaluation,
            )),
        };
        Some((outcome, vec![0xEE]))
    }

    /// Independently-derived expected fire count (multiples of 3 in
    /// `0..n_tiles`) — computed over the tile-index LOOP, not over the lens's
    /// energy-row-0 read. Divergence between this and the lens's actual
    /// output would mean the owner ↔ row-0-energy ↔ MailboxId wiring broke
    /// somewhere in bake/seal/apply/lens, not that the `% 3` arithmetic
    /// disagrees with itself.
    fn expected_multiples_of_three(n_tiles: usize) -> usize {
        (0..n_tiles).filter(|&t| t % 3 == 0).count()
    }

    // ── Full observable-column snapshot (anti-vacuity gate) ─────────────────────

    /// Every column [`MailboxSoaView`] exposes for one owner, captured as
    /// owned data. Used to prove an untouched owner is BYTE-IDENTICAL after a
    /// cycle, not merely "still present".
    #[derive(Debug, Clone, PartialEq)]
    struct Snapshot {
        phase: KanbanColumn,
        current_cycle: u32,
        energy: Vec<f32>,
        entity_type: Vec<u16>,
        edges_raw: Vec<u64>,
        meta_raw: Vec<u32>,
    }

    fn snapshot(owner: &Tile) -> Snapshot {
        Snapshot {
            phase: owner.phase(),
            current_cycle: owner.current_cycle(),
            energy: owner.energy().to_vec(),
            entity_type: owner.entity_type().to_vec(),
            edges_raw: owner.edges_raw().to_vec(),
            meta_raw: owner.meta_raw().to_vec(),
        }
    }

    // ── The fake WAL sink (a minimal re-implementation — `cycle_driver.rs`'s
    //    own `FakeWalSink` is private to its `#[cfg(test)] mod tests` and is
    //    NOT reachable from an integration test under `tests/`) ────────────────

    struct SealedRec {
        frame: CycleFrame,
        version: DatasetVersion,
        landings: Vec<SweepSlot>,
    }

    struct FakeWalSink {
        sealed: Mutex<Vec<SealedRec>>,
        next_version: AtomicU64,
        wal_writes: AtomicU64,
    }

    impl FakeWalSink {
        fn new() -> Self {
            Self {
                sealed: Mutex::new(Vec::new()),
                next_version: AtomicU64::new(1),
                wal_writes: AtomicU64::new(0),
            }
        }
        fn wal_writes(&self) -> u64 {
            self.wal_writes.load(Ordering::SeqCst)
        }
    }

    impl WalSink for FakeWalSink {
        async fn commit_cycle(
            &self,
            base: DatasetVersion,
            batch: DetachedCycleBatch,
        ) -> Result<DatasetVersion, WriteFailed> {
            let mut sealed = self.sealed.lock().unwrap();
            let head = sealed.last().map_or(DatasetVersion(0), |s| s.version);
            if base != head {
                return Err(WriteFailed(format!("stale base {base:?}, head {head:?}")));
            }
            self.wal_writes.fetch_add(1, Ordering::SeqCst);
            let version = DatasetVersion(self.next_version.fetch_add(1, Ordering::SeqCst));
            sealed.push(SealedRec {
                frame: batch.frame,
                version,
                landings: batch.landings,
            });
            Ok(version)
        }
        async fn scan_sealed(
            &self,
            from: Option<DatasetVersion>,
        ) -> Result<Vec<LandedSlot>, WriteFailed> {
            let sealed = self.sealed.lock().unwrap();
            Ok(sealed
                .iter()
                .filter(|s| from.map_or(true, |f| s.version > f))
                .flat_map(|s| {
                    s.landings.iter().cloned().map(move |slot| LandedSlot {
                        version: s.version,
                        slot,
                    })
                })
                .collect())
        }
        async fn versions(&self) -> Result<Vec<(CycleId, DatasetVersion)>, WriteFailed> {
            let sealed = self.sealed.lock().unwrap();
            Ok(sealed.iter().map(|s| (s.frame.cycle, s.version)).collect())
        }
    }

    // ── FALSIFIER 1: N tiled casts -> exactly one WAL write, one version,
    //    over the REAL MailboxSoA owner (re-anchors p4a's FakeOwner-side
    //    headline, `p4a_drains_casts_and_seals_one_wal_write_one_version`) ──────

    #[tokio::test]
    async fn blw1_n_casts_seal_to_one_write_and_one_version_over_the_real_mailbox_soa() {
        let (mut fleet, total_rows) = bake_tiles(CI_TILES);
        assert_eq!(total_rows, CI_TILES * TILE_ROWS, "8 tiles x 1024 rows");
        let owners: Vec<MailboxId> = (0..CI_TILES as MailboxId).collect();

        let sink = FakeWalSink::new();
        let mut w = stage_planning_to_cognitive(&owners);
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();

        let out = run_cycle(
            &sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            0,
            &mut wm,
            u64::from,
        )
        .await
        .unwrap();

        assert_eq!(
            sink.wal_writes(),
            1,
            "{CI_TILES} tiled casts -> exactly ONE WAL write"
        );
        assert_eq!(
            out.sealed.version,
            DatasetVersion(1),
            "-> exactly one version"
        );
        assert_eq!(
            out.applied.applied.len(),
            CI_TILES,
            "every tile's Planning->CognitiveWork move landed"
        );
        for &id in &owners {
            assert_eq!(fleet[&id].phase(), KanbanColumn::CognitiveWork);
        }
    }

    // ── FALSIFIER 2: only the sealed sparse set (as decided by a REAL lens
    //    reading real row data) advances; the rest is byte-identical — the
    //    anti-vacuity gate, RE-ANCHORING
    //    `p4b_applies_only_the_sealed_sparse_set_64k_of_17_advance_rest_byte_identical`
    //    on the real `MailboxSoA` owner instead of `FakeOwner` ─────────────────

    #[tokio::test]
    async fn blw1_sparse_set_advances_remainder_byte_identical_over_the_real_mailbox_soa() {
        let (mut fleet, _) = bake_tiles(CI_TILES);
        let owners: Vec<MailboxId> = (0..CI_TILES as MailboxId).collect();
        let sink = FakeWalSink::new();
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let mut w = stage_planning_to_cognitive(&owners);

        // Cycle 1: every tile enters CognitiveWork.
        let out1 = run_cycle(
            &sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            0,
            &mut wm,
            u64::from,
        )
        .await
        .unwrap();
        assert_eq!(out1.applied.applied.len(), CI_TILES);

        // Snapshot EVERY owner's full observable column state before the lens
        // wave — the "before" half of the byte-identical proof.
        let before: HashMap<MailboxId, Snapshot> = owners
            .iter()
            .map(|&id| (id, snapshot(&fleet[&id])))
            .collect();

        // The lens fires for owners whose tile index (read from row-0 energy)
        // is a multiple of three.
        let cw = run_cognitive_work(&fleet, &out1.applied, &mut w, tile_divisible_by_three_lens);

        let expected_fire: Vec<MailboxId> = owners.iter().copied().filter(|&t| t % 3 == 0).collect();
        let expected_held: Vec<MailboxId> = owners.iter().copied().filter(|&t| t % 3 != 0).collect();
        // Hand-derived, pinned literal (CI_TILES=8: tiles 0,3,6 are multiples
        // of three) — `== N`, not `>= N`.
        assert_eq!(expected_fire.len(), 3, "tiles 0, 3, 6 among CI_TILES=8");
        assert_eq!(expected_held.len(), 5, "the other 5 tiles among CI_TILES=8");
        assert_eq!(
            cw.cast,
            expected_fire.len(),
            "exactly the multiples-of-3 tiles cast a next intent"
        );
        assert_eq!(cw.held_owners.len(), expected_held.len());
        assert!(
            !expected_fire.is_empty(),
            "anti-vacuity: the lens genuinely fires on this input"
        );
        assert!(
            !expected_held.is_empty(),
            "anti-vacuity: the lens genuinely stays silent on this input"
        );
        assert!(
            expected_fire.len() < CI_TILES,
            "sparse: strictly fewer than every owner fired"
        );

        // Cycle 2: seal + apply ONLY the fired owners' CognitiveWork -> Evaluation.
        let out2 = run_cycle(
            &sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(2), DatasetVersion(1)),
            out1.sealed.next_position_base,
            &mut wm,
            u64::from,
        )
        .await
        .unwrap();

        assert_eq!(
            out2.applied.applied.len(),
            expected_fire.len(),
            "dirty_count == exactly the fired set, not merely nonzero"
        );
        for &id in &expected_fire {
            assert_eq!(
                fleet[&id].phase(),
                KanbanColumn::Evaluation,
                "fired tile {id} advanced"
            );
        }

        // The anti-vacuity gate proper: every held owner's FULL snapshot
        // compares byte-identical to its pre-wave state — not merely
        // "the phase looks unchanged".
        let mut untouched = 0usize;
        for &id in &expected_held {
            let after = snapshot(&fleet[&id]);
            assert_eq!(
                after,
                before.get(&id).unwrap().clone(),
                "held tile {id} must be BYTE-IDENTICAL to its pre-wave snapshot"
            );
            assert_eq!(
                after.phase,
                KanbanColumn::CognitiveWork,
                "held tile {id} stayed at CognitiveWork"
            );
            untouched += 1;
        }
        assert_eq!(untouched, expected_held.len());
        assert_eq!(
            untouched + expected_fire.len(),
            CI_TILES,
            "every owner is accounted for: fired + untouched == the whole fleet"
        );
    }

    // ── FALSIFIER 3: an Outcome cast by a REAL lens in Vn is applied in
    //    Vn+1 — re-anchors `p4c_cognitive_work_casts_the_next_intent_and_round_trips`
    //    (which uses `FakeOwner` and a hand-built outcome, not a lens reading
    //    real row data) on the real `MailboxSoA` owner, at tile scale ─────────

    #[tokio::test]
    async fn blw1_lens_cast_in_vn_is_applied_in_vn_plus_1_over_the_real_mailbox_soa() {
        let (mut fleet, _) = bake_tiles(CI_TILES);
        let owners: Vec<MailboxId> = (0..CI_TILES as MailboxId).collect();
        let sink = FakeWalSink::new();
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let mut w = stage_planning_to_cognitive(&owners);

        // Vn = V1: every tile enters CognitiveWork.
        let out1 = run_cycle(
            &sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            0,
            &mut wm,
            u64::from,
        )
        .await
        .unwrap();
        assert_eq!(out1.sealed.version, DatasetVersion(1));
        for &id in &owners {
            assert_eq!(fleet[&id].phase(), KanbanColumn::CognitiveWork);
        }

        // The lens thinks over the REAL post-V1 fleet and casts a next-cycle
        // Outcome for every multiple-of-3 tile — staged into the writer for
        // V2, never applied yet (P4c never mutates a mailbox itself).
        let cw = run_cognitive_work(&fleet, &out1.applied, &mut w, tile_divisible_by_three_lens);
        let expected_fire = expected_multiples_of_three(CI_TILES);
        assert_eq!(expected_fire, 3, "pinned: tiles 0, 3, 6 among CI_TILES=8");
        assert_eq!(cw.cast, expected_fire, "one cast per multiple-of-3 tile");
        for &id in &owners {
            assert_eq!(
                fleet[&id].phase(),
                KanbanColumn::CognitiveWork,
                "the cognitive pass alone never mutates a mailbox"
            );
        }

        // Vn+1 = V2: the driver drains the staged casts, seals ONE more
        // version, and applies exactly the fired set's CognitiveWork ->
        // Evaluation step.
        let out2 = run_cycle(
            &sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(2), DatasetVersion(1)),
            out1.sealed.next_position_base,
            &mut wm,
            u64::from,
        )
        .await
        .unwrap();

        assert_eq!(
            out2.sealed.version,
            DatasetVersion(2),
            "the outcome cast in V1 seals into exactly V2, not V1 itself"
        );
        assert_eq!(
            out2.applied.applied.len(),
            expected_fire,
            "V1's outcome cast is exactly what advanced in V2"
        );
        for &id in &owners {
            let expected_phase = if id % 3 == 0 {
                KanbanColumn::Evaluation
            } else {
                KanbanColumn::CognitiveWork
            };
            assert_eq!(
                fleet[&id].phase(),
                expected_phase,
                "tile {id}: V1-cast outcome landed in V2, nothing else moved"
            );
        }
    }

    // ── FALSIFIER 4 (ignored by default): the same three re-anchored
    //    assertions at the full 64-tile / 65,536-row KJV scale, over the
    //    real `MailboxSoA` owner + a real lens ──────────────────────────────────

    #[tokio::test]
    #[ignore = "384 MiB of identity planes; run explicitly"]
    async fn blw1_full_kjv_scale_64_tiles_over_the_real_mailbox_soa() {
        let (mut fleet, total_rows) = bake_tiles(FULL_TILES);
        assert_eq!(
            total_rows,
            FULL_TILES * TILE_ROWS,
            "64 tiles x 1024 rows = 65,536 verse rows"
        );
        let owners: Vec<MailboxId> = (0..FULL_TILES as MailboxId).collect();
        let sink = FakeWalSink::new();
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let mut w = stage_planning_to_cognitive(&owners);

        // Headline 1 (mirrors FALSIFIER 1): N tiled casts -> one write, one version.
        let out1 = run_cycle(
            &sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            0,
            &mut wm,
            u64::from,
        )
        .await
        .unwrap();
        assert_eq!(sink.wal_writes(), 1, "64 tiled casts -> exactly ONE WAL write");
        assert_eq!(out1.sealed.version, DatasetVersion(1));
        assert_eq!(out1.applied.applied.len(), FULL_TILES);

        // Headline 2 (mirrors FALSIFIER 2): sparse set + byte-identical remainder.
        let before: HashMap<MailboxId, Snapshot> = owners
            .iter()
            .map(|&id| (id, snapshot(&fleet[&id])))
            .collect();

        let cw = run_cognitive_work(&fleet, &out1.applied, &mut w, tile_divisible_by_three_lens);
        let expected_fire: Vec<MailboxId> =
            owners.iter().copied().filter(|&t| t % 3 == 0).collect();
        let expected_held: Vec<MailboxId> =
            owners.iter().copied().filter(|&t| t % 3 != 0).collect();
        // Hand-derived, pinned literal: multiples of 3 in [0, 64) are
        // 0, 3, .., 63 -> 22 tiles; the other 42 are held.
        assert_eq!(expected_fire.len(), 22, "0..64 multiples of three");
        assert_eq!(expected_held.len(), 42, "the remaining tiles among 64");
        assert_eq!(cw.cast, expected_fire.len());
        assert_eq!(cw.held_owners.len(), expected_held.len());
        assert!(!expected_fire.is_empty(), "anti-vacuity: fires at full scale");
        assert!(!expected_held.is_empty(), "anti-vacuity: silent at full scale");
        assert!(expected_fire.len() < FULL_TILES, "sparse at full scale");

        // Headline 3 (mirrors FALSIFIER 3): the round trip Vn -> Vn+1.
        let out2 = run_cycle(
            &sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(2), DatasetVersion(1)),
            out1.sealed.next_position_base,
            &mut wm,
            u64::from,
        )
        .await
        .unwrap();
        assert_eq!(sink.wal_writes(), 2, "one more WAL write for the lens wave");
        assert_eq!(out2.sealed.version, DatasetVersion(2));
        assert_eq!(out2.applied.applied.len(), expected_fire.len());

        for &id in &expected_fire {
            assert_eq!(fleet[&id].phase(), KanbanColumn::Evaluation);
        }
        let mut untouched = 0usize;
        for &id in &expected_held {
            let after = snapshot(&fleet[&id]);
            assert_eq!(
                after,
                before.get(&id).unwrap().clone(),
                "held tile {id} must be BYTE-IDENTICAL at full scale"
            );
            assert_eq!(after.phase, KanbanColumn::CognitiveWork);
            untouched += 1;
        }
        assert_eq!(untouched, expected_held.len());
        assert_eq!(untouched + expected_fire.len(), FULL_TILES);
    }
}
