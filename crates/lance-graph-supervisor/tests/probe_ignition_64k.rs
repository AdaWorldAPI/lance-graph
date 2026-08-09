//! PROBE-IGNITION-64K — `start()` at the MAIN MODEL's full population.
//!
//! Answers ONE question the 64-owner probe deliberately did not: does the
//! ignition machinery — arm by `MetaWord` write, discover by board scan,
//! cast write-on-behalf, seal ONCE, apply — hold at **65,536 real
//! `MailboxSoA` owners, 1:1, mutation-exclusive** (the operator-ordered
//! main model, `EPIPHANIES.md` E-64K-1TO1-OWNERS-IS-THE-MAIN-MODEL-1)?
//!
//! Scale is made feasible by shrinking the per-owner slice, not the
//! population: `MailboxSoA<4>` with ONE populated row per owner (the
//! main-model axis is OWNERS; rows-per-owner is the benchmark axis,
//! plan §12.3a′/§12.3a‴). The identity planes are lazily mapped, so RSS is
//! dominated by the one written row per owner.
//!
//! ## What this probe does NOT claim (printed at the end of the run)
//!
//! **No concurrency.** The loop is synchronous — this is the SCALE half of
//! the main-model claim only; "parallel" remains gated by D-KIA-A2's
//! pre-registered protocol. **No timing claim** (wall times are printed as
//! provenance, never asserted). **No durability** (`MemWal`). **No
//! semantic claim** (synthetic single-row content, non-degeneracy sampled).
//!
//! Shapes copied with provenance from `tests/probe_ignition.rs` (GREEN,
//! 2/2): the qualia fixtures, the MemWal seam, the gate->style->emit pass,
//! the one-seal/rest disciplines.

#[cfg(feature = "cycle-driver")]
mod probe_ignition_64k {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )]

    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;

    use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome, WORDS_PER_FP};
    use lance_graph_contract::cognitive_shader::MetaWord;
    use lance_graph_contract::collapse_gate::MailboxId;
    use lance_graph_contract::kanban::{ExecTarget, KanbanColumn};
    use lance_graph_contract::mul::i4_eval::gate_decision_i4;
    use lance_graph_contract::qualia::QualiaI4_16D;
    use lance_graph_contract::scheduler::DatasetVersion;
    use lance_graph_contract::soa_view::MailboxSoaView;
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
        run_cognitive_work_gated_over, run_cycle, CycleError, CycleOutcome,
    };

    // ── PRE-REGISTERED run shape — fixed BEFORE any number exists. ─────────

    /// The full main-model population. 65,536 = the "64k" of the order.
    const FLEET_OWNERS: MailboxId = 65_536;
    /// Minimal slice: rows-per-owner is NOT this probe's axis (§12.3a‴).
    const ROWS_PER_OWNER: usize = 4;
    const POPULATED_ROWS: usize = 1;
    /// Sampled non-degeneracy: this many owners' planes checked non-zero
    /// and pairwise distinct (a full 64k pairwise sweep proves nothing more
    /// for O(n²) cost).
    const DISTINCTNESS_SAMPLE: usize = 8;
    const TENANT_THRESHOLD: f32 = 1.0;
    const FIRE_ENERGY: f32 = 2.0;
    const TENANT_W_SLOT: u8 = 0;
    /// Reliability handed to the cycle-2 gated pass (HAND-TUNED constant,
    /// inert to this probe's assertions — nothing branches on it).
    const GATED_RELIABILITY: f32 = 0.5;

    type Tenant = MailboxSoA<ROWS_PER_OWNER>;
    type Fleet = HashMap<MailboxId, Tenant>;

    /// Flow qualia — the shipped gate falsifiers' own fixture construction
    /// (`probe_ignition.rs:201-203` provenance; `cycle_driver.rs:1669`):
    /// warmth=4, groundedness=3, coherence=4, valence=2 => flow_proxy 7,
    /// Calibrated. NOT an all-zeros rig — the silence at cycle 2 happens on
    /// a would-be-Flow qualia because the derived mantissa fell to 0.
    fn flow_qualia() -> QualiaI4_16D {
        QualiaI4_16D(0).with(3, 4).with(14, 3).with(9, 4).with(1, 2)
    }

    /// Derived mantissa (`probe_ignition.rs:192-194` provenance).
    fn mantissa_of(owner: &Tenant) -> i8 {
        owner.pending_count().min(7) as i8
    }

    /// Synthetic non-zero identity plane, distinct per owner (splitmix-style
    /// scramble of the id — deterministic, no clock, no rng).
    fn plane_for(id: MailboxId) -> Vec<u64> {
        let mut x = u64::from(id) ^ 0x9E37_79B9_7F4A_7C15;
        let mut plane = vec![0u64; WORDS_PER_FP];
        for w in plane.iter_mut() {
            x ^= x >> 30;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 27;
            x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            *w = x | 1; // never a zero word
        }
        plane
    }

    fn payload_for(id: MailboxId) -> Vec<u8> {
        u64::from(id).to_le_bytes().to_vec()
    }

    // ── the WAL seam (in-process; NOT durability) — `probe_ignition.rs`
    // MemWal, trimmed to what this probe asserts. ──────────────────────────

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
    }

    impl MemWal {
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
                        current_head: rec.version,
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

    // ── fleet construction: 65,536 real 1:1 owners ─────────────────────────

    fn build_owner(id: MailboxId) -> Tenant {
        let mut owner: Tenant = MailboxSoA::new(id, TENANT_W_SLOT, TENANT_THRESHOLD);
        let cycle = owner.cycle();
        let plane = plane_for(id);
        let cell = WriteCell {
            content: Some(plane.as_slice()),
            qualia: Some(flow_qualia()),
            meta: Some(MetaWord::new(1, 0, 0, 0, 0)), // armed z=1 (Analytical)
            entity_type: Some((id % 251) as u16),
            temporal: Some(u64::from(id)),
            ..WriteCell::default()
        };
        let outcome = owner.write_row(0, cycle, &cell);
        assert_eq!(outcome, WriteOutcome::Accepted, "seeding owner {id} row 0");
        owner.set_populated(POPULATED_ROWS);
        owner.tick();
        owner.energy[0] = FIRE_ENERGY; // one firing row: exhausts after one advance
        owner
    }

    /// The 23D style vector for z=1 (`probe_ignition.rs:165-173` provenance:
    /// idx 4 = analytical).
    fn style_context() -> PlanContext {
        let mut v = vec![0.0f64; 23];
        v[4] = 1.0;
        PlanContext {
            query: String::new(),
            features: QueryFeatures::default(),
            free_will_modifier: 1.0,
            thinking_style: Some(v),
            nars_hint: None,
            witness: None,
        }
    }

    // ── the probe ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn probe_ignition_64k_start_at_full_population() {
        let t0 = Instant::now();
        let mut fleet = Fleet::with_capacity(FLEET_OWNERS as usize);
        for id in 0..FLEET_OWNERS {
            fleet.insert(id, build_owner(id));
        }
        eprintln!(
            "probe.ignition64k build: {FLEET_OWNERS} real 1:1 MailboxSoA<{ROWS_PER_OWNER}> owners in {:?}",
            t0.elapsed()
        );

        // Non-degeneracy (sampled): planes non-zero + pairwise distinct.
        {
            let ids: Vec<MailboxId> = (0..DISTINCTNESS_SAMPLE as MailboxId)
                .map(|i| i * (FLEET_OWNERS / DISTINCTNESS_SAMPLE as MailboxId))
                .collect();
            let mut planes = Vec::new();
            for &id in &ids {
                let owner = fleet.get(&id).expect("sampled owner exists");
                let p = owner.content_row(0).to_vec();
                assert!(
                    p.iter().any(|&w| w != 0),
                    "owner {id} plane must be non-zero"
                );
                planes.push(p);
            }
            for i in 0..planes.len() {
                for j in (i + 1)..planes.len() {
                    assert_ne!(planes[i], planes[j], "sampled planes must be distinct");
                }
            }
            eprintln!(
                "probe.ignition64k non-degeneracy: {DISTINCTNESS_SAMPLE} sampled planes non-zero + pairwise distinct"
            );
        }

        let mut sink = MemWal::new();
        let mut writer: BatchWriter<Vec<u8>> = BatchWriter::new();
        let mut watermarks: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let position_base: u64 = 0;

        // ── S0: the STYLE's mint, computed ONCE — every owner is armed with
        // the same z, so the style outcome is owner-independent; the
        // PER-OWNER binding is `rebind_bootstrap`'s job inside
        // `emit_bootstrap_intent` (owner_adapter.rs — the no-theft guard).
        // `emit_bootstrap_intent` takes `&StrategyOutcome`, so one plan()
        // serves 65,536 emits. ────────────────────────────────────────────
        let mut arena = Arena::new();
        let plan_out = StyleStrategy
            .plan(
                PlanInput {
                    plan: None,
                    context: style_context(),
                    outcome: None,
                },
                &mut arena,
            )
            .expect("StyleStrategy::plan over the fixed recipe substrate");
        let style_outcome: StrategyOutcome = plan_out
            .outcome
            .expect("StyleStrategy always surfaces a StrategyOutcome");

        // ── CYCLE 1: LOOK INTO THE KANBAN (scan all 64k) + CAST ────────────
        let t1 = Instant::now();
        let mut cast = 0usize;
        for id in 0..FLEET_OWNERS {
            let owner = fleet.get(&id).expect("owner exists");
            assert_eq!(
                owner.phase(),
                KanbanColumn::Planning,
                "scan @c1: owner {id}"
            );
            let qualia = owner.qualia_at(0);
            let mantissa = mantissa_of(owner);
            let gate = gate_decision_i4(&qualia, mantissa);
            let target = owner
                .phase()
                .advance_on_gate(&gate)
                .expect("c1: every owner Flows (flow qualia + firing row)");
            assert_eq!(target, KanbanColumn::CognitiveWork, "c1 gate target");
            let ok = emit_bootstrap_intent(
                &style_outcome,
                owner.mailbox_id(),
                owner.current_cycle(),
                &mut writer,
                payload_for(id),
            );
            assert!(ok.is_some(), "c1: cast for owner {id} must stage");
            cast += 1;
        }
        eprintln!(
            "probe.ignition64k c1 cast: {cast} casts staged in {:?} (scan + gate + style emit)",
            t1.elapsed()
        );
        assert_eq!(
            cast, FLEET_OWNERS as usize,
            "S1 can-fire: every owner casts"
        );

        // ── ONE SEAL — the single deterministic convergence boundary of the
        // main model, at full population. ──────────────────────────────────
        let t2 = Instant::now();
        let wal_before = sink.wal_writes();
        let base = sink.head();
        let outcome: CycleOutcome = match run_cycle(
            &mut sink,
            &mut fleet,
            &mut writer,
            CycleFrame::new(CycleId(1), base),
            position_base,
            &mut watermarks,
            u64::from,
        )
        .await
        {
            Ok(o) => o,
            Err(CycleError::Seal(_)) => panic!("64k: unexpected seal failure"),
            Err(CycleError::Apply { cause, .. }) => {
                panic!("64k: unexpected apply failure: {cause}")
            }
        };
        // Restart-stable contract honored even though this probe seals once:
        // the advanced base is ASSERTED (a second cycle would resume from it).
        let advanced_base = position_base.max(outcome.sealed.next_position_base);
        assert!(
            advanced_base >= FLEET_OWNERS as u64,
            "position_base advances past the 64k sealed positions"
        );
        eprintln!(
            "probe.ignition64k c1 seal+apply: {} transitions in {:?}",
            outcome.sealed.transitions.len(),
            t2.elapsed()
        );

        // S2 can-fire: 65,536 casts converged into EXACTLY ONE WAL write.
        assert_eq!(sink.wal_writes(), wal_before + 1, "S2: one seal, one write");
        assert_eq!(
            outcome.sealed.transitions.len(),
            FLEET_OWNERS as usize,
            "S2: every owner's move sealed in the one cycle"
        );
        // Every sealed move is the STYLE's Planning->CognitiveWork mint.
        for t in &outcome.sealed.transitions {
            assert_eq!(t.mv.from, KanbanColumn::Planning, "sealed move origin");
            assert_eq!(t.mv.to, KanbanColumn::CognitiveWork, "sealed move target");
            assert_eq!(t.mv.exec, ExecTarget::Elixir, "the style's mint");
        }
        // Ordering discipline: positions strictly monotone within the seal
        // (the write-side owns arrival, never cross-mailbox order).
        for w in outcome.sealed.transitions.windows(2) {
            assert!(
                w[0].stream_position < w[1].stream_position,
                "sealed positions strictly monotone"
            );
        }
        // Applied on every owner: phase advanced via the seal, nothing else.
        let mut advanced = 0usize;
        for (_, owner) in fleet.iter() {
            if owner.phase() == KanbanColumn::CognitiveWork {
                advanced += 1;
            }
        }
        assert_eq!(advanced, FLEET_OWNERS as usize, "S3: all 64k advanced");
        eprintln!(
            "probe.ignition64k S1-S3: {FLEET_OWNERS} owners armed->cast->sealed(1 write)->advanced"
        );

        // ── write-back AFTER apply (never during compute): consume the one
        // firing row per owner — the mantissa falls to 0. ──────────────────
        for id in 0..FLEET_OWNERS {
            let owner = fleet.get_mut(&id).expect("owner exists");
            owner.consume_firing(0);
        }

        // ── CYCLE 2: the SILENT twin at full population — the same shipped
        // gated pass, every owner on a would-be-Flow qualia, and the whole
        // 64k fleet RESTS because the derived mantissa is 0.
        //
        // Measurement note (a first draft asserted `writer.casts().len() == 0`
        // and failed at 65,536): `casts()` is the CUMULATIVE board — cycle 1's
        // cast records are retained after `collect_casts` drains the payloads
        // (the documented drained-writer semantics G9 pins in the 64-owner
        // probe). The rest is therefore measured as a DELTA. ────────────────
        let t3 = Instant::now();
        let casts_before_c2 = writer.casts().len();
        let ids: Vec<MailboxId> = (0..FLEET_OWNERS).collect();
        let held = run_cognitive_work_gated_over(&fleet, &ids, &mut writer, |owner| {
            Some((
                owner.qualia_at(0),
                mantissa_of(owner),
                GATED_RELIABILITY,
                payload_for(owner.mailbox_id()),
            ))
        });
        let staged_c2 = writer.casts().len() - casts_before_c2;
        assert_eq!(
            staged_c2, 0,
            "S4 can-stay-silent: zero NEW casts at c2 — the fleet rests on non-trivial qualia (flow_proxy 7), not on an all-zeros rig"
        );
        // The positive half of the silence: every one of the 64k owners was
        // SEEN by the pass and Held — the rest is a decision over the full
        // population, not an empty scan.
        assert_eq!(
            held.held_owners.len(),
            FLEET_OWNERS as usize,
            "S4: every owner was seen and Held (rest is a per-owner decision, not absence)"
        );
        // No seal on a rest cycle: wal_writes frozen at 1.
        assert_eq!(sink.wal_writes(), 1, "S4: resting must not write");
        eprintln!(
            "probe.ignition64k c2 rest: 0 new casts, {} owners seen+Held in {:?}; wal_writes frozen at 1",
            held.held_owners.len(),
            t3.elapsed()
        );

        println!();
        println!("== PROBE-IGNITION-64K — what this probe does NOT claim ==");
        println!("1. No CONCURRENCY. The loop is synchronous; this is the SCALE half of the main-model claim only. 'Parallel' remains gated by D-KIA-A2's pre-registered protocol (median-of-5, >=2x at >=4,096 owners, >=100us bodies).");
        println!("2. No timing claim. Wall times above are provenance, never asserted.");
        println!("3. No durability. MemWal is an in-process Mutex/Vec.");
        println!("4. No semantic claim. One synthetic row per owner; distinctness sampled at {DISTINCTNESS_SAMPLE}.");
        println!("5. No rows-axis claim. ROWS_PER_OWNER=4 on purpose — that axis belongs to D-BLW-4 (plan 12.3a-prime).");
        println!(
            "TOTAL: {FLEET_OWNERS} real 1:1 owners, arm->scan->cast->seal(1)->advance->rest, {:?} end to end.",
            t0.elapsed()
        );
    }
}
