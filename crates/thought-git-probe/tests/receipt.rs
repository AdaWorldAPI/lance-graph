//! D-TGB-4: a receipt instead of history. Every test is a falsifier; its doc
//! comment names the deliberate break of `receipt.rs` under which it must turn
//! red. Results are compared by folds. The independent fold below uses FNV-1a
//! over the row, lane and value bytes, summed per write, so it shares no code
//! and no mixing function with the receipt's own fold.

use lance_graph_contract::revision::{
    BasisView, CodebookId, EncounterEvidence, EvidentialEffect, GadamerRevision, GrammarId,
    HorizonId, InterpretiveHorizon, LanguageId, LensId, QuestionId, RevisionPolicy,
};
use thought_git_probe::fixtures::{self, Cell, Outcome, Thought, cell, main_version};
use thought_git_probe::oracle::{Version, next_version};
use thought_git_probe::receipt::{
    self, BaseRef, CommitError, DERIVE, EXEC_DEFAULT, ExecConfig, MetaKey, MetadataTable, Plan,
    ProgramInputs, Receipt, Refusal, Replayed, VersionStore,
};
use thought_git_probe::slots::{Eviction, Slot, SlotPool};

// ---------------------------------------------------------------- helpers

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Count and order-independent sum of per-write FNV-1a hashes.
fn independent_fold(writes: &[(Cell, u64)]) -> (u64, u64) {
    let mut sum = 0_u64;
    for &(c, v) in writes {
        let mut bytes = Vec::with_capacity(13);
        bytes.extend_from_slice(&c.row.to_le_bytes());
        bytes.push(c.lane);
        bytes.extend_from_slice(&v.to_le_bytes());
        sum = sum.wrapping_add(fnv1a(&bytes));
    }
    (writes.len() as u64, sum)
}

/// Both folds agree between the original run and the replay, and the replay
/// reports the fold the receipt recorded.
fn assert_same_by_folds(original: &Thought, rep: &Replayed, r: &Receipt) {
    assert_eq!(
        independent_fold(&original.writes),
        independent_fold(&rep.thought.writes),
        "independent fold differs"
    );
    assert_eq!(rep.output, r.output, "receipt fold differs");
    assert_eq!(rep.output.count, original.writes.len() as u64);
}

fn base_ref() -> BaseRef {
    BaseRef {
        dataset: 7,
        branch: 0,
        version: 1,
    }
}

fn setup() -> (VersionStore, BaseRef) {
    let mut store = VersionStore::new(fixtures::givens());
    let base = base_ref();
    store.seal(base, main_version()).expect("seal base");
    (store, base)
}

/// Sources: two given cells and one absent cell. Targets: the
/// `rubicon_outcomes` cells.
fn inputs() -> ProgramInputs {
    ProgramInputs::new(
        vec![cell(0, 0), cell(1, 0), cell(9, 3)],
        vec![cell(4, 1), cell(5, 1), cell(6, 1)],
    )
}

fn plan(base: BaseRef, thought: u32, outcome: Outcome, seed: u64) -> Plan {
    Plan {
        thought,
        base,
        program: DERIVE,
        inputs: inputs(),
        seed,
        config: EXEC_DEFAULT,
        roots: 0b0011,
        outcome,
    }
}

/// Branch `thought` and capture its receipt.
fn run(store: &VersionStore, pool: &mut SlotPool, p: Plan) -> (Slot, Thought, Receipt) {
    let (slot, _) = pool.branch(p.thought, None).expect("branch");
    let (t, r) = receipt::capture(pool, slot, store, p).expect("capture");
    (slot, t, r)
}

type Horizon = InterpretiveHorizon<(), u64>;

fn horizon(claims: u64, roots: u64) -> Horizon {
    InterpretiveHorizon {
        id: HorizonId(1),
        awareness: (),
        question: QuestionId(1),
        language: LanguageId(1),
        grammar: GrammarId(1),
        codebook: CodebookId(1),
        lens: LensId(1),
        projected_claims: claims,
        independent_roots: roots,
        inherited_roots: 0,
        unresolved_tension: 0,
        revision_index: 0,
    }
}

fn ancestry(h: &Horizon) -> BasisView<u64> {
    BasisView {
        ancestry_independent_roots: h.independent_roots,
        ancestry_derived_roots: h.inherited_roots,
        ancestor_claims: h.projected_claims,
        closes_cycle: false,
    }
}

fn encounter(claims: u64, roots: u64, inherited: u64, contra: u64) -> EncounterEvidence<u64> {
    EncounterEvidence {
        proposed_claims: claims,
        independent_roots: roots,
        inherited_roots: inherited,
        resistance: 0,
        contradictions: contra,
        affected_parts: 0,
    }
}

// ------------------------------------------------------------------ tests

/// 1. A thought whose slot was freed (and reused) replays exactly from the
///    receipt alone. Disable: `replay` reads `captured_in` from the pool and
///    refuses (or misreads) once the slot is gone or recycled.
#[test]
fn a_released_thought_replays_exactly() {
    let (store, base) = setup();
    let mut pool = SlotPool::new(Eviction::Fifo);
    let outcomes = [
        (10, Outcome::Commit, 11_u64),
        (11, Outcome::Plan, 22),
        (12, Outcome::Prune, 33),
    ];
    let mut captured = Vec::new();
    for (id, outcome, seed) in outcomes {
        captured.push(run(&store, &mut pool, plan(base, id, outcome, seed)));
    }
    for (slot, _, _) in &captured {
        pool.free(*slot).expect("free");
        assert!(pool.thought(*slot).is_err(), "slot still live after free");
    }

    // The freed slot is reused by another thought: same index, new generation.
    let first = captured[0].0;
    let (reused, _) = pool.branch(99, None).expect("rebranch");
    assert_eq!(reused.index, first.index);
    assert_ne!(reused.generation, first.generation);

    for (_, original, r) in &captured {
        let rep = receipt::replay(r, &store, EXEC_DEFAULT).expect("replay");
        assert_same_by_folds(original, &rep, r);
        assert_eq!(&rep.thought, original, "cell-for-cell");
    }
}

/// 2. A receipt without its seed refuses. Disable: default the seed to 0.
#[test]
fn a_missing_seed_refuses() {
    let (store, base) = setup();
    let mut pool = SlotPool::new(Eviction::Fifo);
    let (_, t, mut r) = run(&store, &mut pool, plan(base, 10, Outcome::Commit, 5));
    let rep = receipt::replay(&r, &store, EXEC_DEFAULT).expect("with seed");
    assert_same_by_folds(&t, &rep, &r);
    r.seed = None;
    assert_eq!(
        receipt::replay(&r, &store, EXEC_DEFAULT),
        Err(Refusal::MissingSeed)
    );
}

/// 3. A different seed produces different writes and the output check
///    refuses; the recorded seed replays. Disable: skip the output check.
#[test]
fn a_changed_seed_is_an_output_mismatch() {
    let (store, base) = setup();
    let mut pool = SlotPool::new(Eviction::Fifo);
    let (_, t1, r1) = run(&store, &mut pool, plan(base, 10, Outcome::Commit, 5));
    let (_, t2, _) = run(&store, &mut pool, plan(base, 11, Outcome::Commit, 6));
    // Anti-vacuity: the two seeds really give different writes.
    assert_ne!(
        independent_fold(&t1.writes),
        independent_fold(&t2.writes),
        "seeds 5 and 6 produce the same writes; the test would prove nothing"
    );

    let rep = receipt::replay(&r1, &store, EXEC_DEFAULT).expect("original seed");
    assert_same_by_folds(&t1, &rep, &r1);

    let mut changed = r1.clone();
    changed.seed = Some(6);
    let err = receipt::replay(&changed, &store, EXEC_DEFAULT).expect_err("changed seed");
    assert!(matches!(err, Refusal::OutputMismatch { .. }), "got {err:?}");
}

/// 4. Replaying under another execution configuration refuses. Disable: drop
///    the configuration comparison.
#[test]
fn a_changed_config_refuses() {
    let (store, base) = setup();
    let mut pool = SlotPool::new(Eviction::Fifo);
    let (_, t, r) = run(&store, &mut pool, plan(base, 10, Outcome::Commit, 5));
    let other = ExecConfig {
        modulus: 999,
        ..EXEC_DEFAULT
    };
    assert_eq!(
        receipt::replay(&r, &store, other),
        Err(Refusal::ConfigChanged)
    );
    let rep = receipt::replay(&r, &store, EXEC_DEFAULT).expect("same config");
    assert_same_by_folds(&t, &rep, &r);
}

/// 5. A sealed source changed behind the receipt's back refuses; a change to
///    a cell that is not a source does not. Disable: skip the source witness
///    check (the first half then reaches the output check or passes).
#[test]
fn a_changed_source_refuses_and_a_non_source_change_does_not() {
    // Fire: write the ABSENT source cell.
    let (mut store, base) = setup();
    let mut pool = SlotPool::new(Eviction::Fifo);
    let (_, _, r) = run(&store, &mut pool, plan(base, 10, Outcome::Commit, 5));
    let mut v = Version::default();
    v.cells.insert(cell(9, 3), 5);
    store.tamper(base, v);
    let err = receipt::replay(&r, &store, EXEC_DEFAULT).expect_err("source changed");
    assert!(matches!(err, Refusal::SourceChanged { .. }), "got {err:?}");

    // Silent: write a cell that no source reads.
    let (mut store, base) = setup();
    let mut pool = SlotPool::new(Eviction::Fifo);
    let (_, t, r) = run(&store, &mut pool, plan(base, 10, Outcome::Commit, 5));
    let mut v = Version::default();
    v.cells.insert(cell(11, 3), 5);
    store.tamper(base, v);
    let rep = receipt::replay(&r, &store, EXEC_DEFAULT).expect("non-source change");
    assert_same_by_folds(&t, &rep, &r);
}

/// 6. A stale or reused slot fails closed; a live slot is accepted.
///    Disable: ignore the generation in the slot check.
#[test]
fn stale_and_reused_slots_fail_closed() {
    let (store, base) = setup();
    let mut pool = SlotPool::new(Eviction::Fifo);
    let (slot, t, r) = run(&store, &mut pool, plan(base, 10, Outcome::Commit, 5));

    // Silent half: the live slot replays.
    let rep = receipt::replay_live(&pool, &r, &store, EXEC_DEFAULT).expect("live");
    assert_same_by_folds(&t, &rep, &r);

    // WrongThought: a live slot holding another thought than the plan's.
    let err = receipt::capture(&pool, slot, &store, plan(base, 11, Outcome::Plan, 5))
        .expect_err("wrong thought");
    assert_eq!(
        err,
        Refusal::WrongThought {
            expected: 11,
            found: 10
        }
    );

    // Freed: stale.
    pool.free(slot).expect("free");
    assert!(matches!(
        receipt::replay_live(&pool, &r, &store, EXEC_DEFAULT),
        Err(Refusal::Stale(_))
    ));
    assert!(matches!(
        receipt::capture(&pool, slot, &store, plan(base, 10, Outcome::Commit, 5)),
        Err(Refusal::Stale(_))
    ));

    // Re-branched by a different thought: same index, still stale.
    let (reused, _) = pool.branch(77, None).expect("rebranch");
    assert_eq!(reused.index, slot.index);
    assert_ne!(reused.generation, slot.generation);
    assert!(matches!(
        receipt::replay_live(&pool, &r, &store, EXEC_DEFAULT),
        Err(Refusal::Stale(_))
    ));

    // ABA: the SAME thought id re-branched into the same index. Only the
    // generation tells the old handle from the new one.
    pool.free(reused).expect("free reused");
    let (aba, _) = pool.branch(10, None).expect("aba");
    assert_eq!(aba.index, slot.index);
    assert_eq!(pool.thought(aba), Ok(10));
    assert!(matches!(
        receipt::replay_live(&pool, &r, &store, EXEC_DEFAULT),
        Err(Refusal::Stale(_))
    ));
}

/// 7. A replay keeps its contradictions and roots. Disable: replay drops or
///    rewrites a write, or reports the receipt roots as independent.
#[test]
fn contradictions_and_roots_keep_their_meaning() {
    let (store, base) = setup();
    let mut pool = SlotPool::new(Eviction::Fifo);
    // Target (1, 0) is a given (101); source (2, 0) reads 102, so every
    // offered value is at least 102 and differs from the given.
    let p = Plan {
        inputs: ProgramInputs::new(vec![cell(2, 0), cell(9, 3)], vec![cell(1, 0), cell(4, 1)]),
        ..plan(base, 10, Outcome::Commit, 5)
    };
    let (_, original, r) = run(&store, &mut pool, p);
    let rep = receipt::replay(&r, &store, EXEC_DEFAULT).expect("replay");
    assert_same_by_folds(&original, &rep, &r);

    let g = store.givens.clone();
    let v_orig = next_version(&g, &main_version(), std::slice::from_ref(&original));
    let v_rep = next_version(&g, &main_version(), std::slice::from_ref(&rep.thought));
    assert_eq!(v_orig, v_rep);
    assert_eq!(v_rep.contradictions.len(), 1);
    let c = v_rep.contradictions[0];
    assert_eq!(c.cell, cell(1, 0));
    assert_eq!(c.given, 101);
    assert_ne!(c.offered, 101);
    assert_eq!(c.thought, 10);
    assert_eq!(v_orig.contradictions[0], c);

    assert_eq!(rep.inherited_roots, r.roots);
    assert_eq!(rep.independent_roots, 0);
}

/// 8. A replay creates no new certification. Disable: report the receipt
///    roots as independent (the replay would then look like a new root).
#[test]
fn replay_cannot_raise_evidence_but_a_new_root_can() {
    let (store, base) = setup();
    let mut pool = SlotPool::new(Eviction::Fifo);
    let (_, _, r) = run(&store, &mut pool, plan(base, 10, Outcome::Commit, 5));
    let rep = receipt::replay(&r, &store, EXEC_DEFAULT).expect("replay");

    // The prior horizon predates the receipt's roots: it holds another root
    // (bit 3) and not 0b0011. Labelled correctly, the replay's roots arrive as
    // inherited and earn nothing; labelled independent they would look new.
    let prior = horizon(0b0011, 0b1000);
    assert_eq!(
        prior.independent_roots & r.roots,
        0,
        "receipt roots must be new to the prior"
    );
    let e = encounter(0b0011, rep.independent_roots, rep.inherited_roots, 0);
    let d = GadamerRevision.revise(&prior, &e, &ancestry(&prior));
    assert_ne!(d.evidential_effect, EvidentialEffect::IncreaseEligible);

    // Can-fire twin: same encounter plus a root outside the prior's.
    let new_bit = 0b0100_u64;
    assert_eq!(r.roots & new_bit, 0, "bit must be new");
    let fresh = encounter(0b0111, rep.independent_roots | r.roots | new_bit, 0, 0);
    let d = GadamerRevision.revise(&prior, &fresh, &ancestry(&prior));
    assert_eq!(d.evidential_effect, EvidentialEffect::IncreaseEligible);
}

/// 9. A staged-but-unsealed or torn metadata row is not a landing. Disable:
///    accept rows with `seal == None` (or skip the seal comparison).
#[test]
fn a_partial_metadata_write_cannot_pass_as_a_landing() {
    let (mut store, base) = setup();
    let mut pool = SlotPool::new(Eviction::Fifo);
    let (slot, original, r) = run(&store, &mut pool, plan(base, 10, Outcome::Commit, 5));
    let data = BaseRef {
        dataset: 7,
        branch: 0,
        version: 2,
    };
    let key = MetaKey { data, thought: 10 };
    let mut table = MetadataTable::default();

    // Data first, then metadata: staging before the data is sealed refuses.
    assert!(matches!(
        table.stage(&pool, &store, key, r.clone()),
        Err(Refusal::DataMissing(_))
    ));
    store
        .commit_cycle(data, &main_version(), 1, std::slice::from_ref(&original))
        .expect("commit");

    // A key naming another thought than the receipt's.
    let bad = MetaKey { data, thought: 11 };
    assert!(matches!(
        table.stage(&pool, &store, bad, r.clone()),
        Err(Refusal::KeyMismatch(_))
    ));

    // Never staged.
    let never = MetaKey { data, thought: 999 };
    assert!(matches!(
        table.replay_landed(&store, never, EXEC_DEFAULT),
        Err(Refusal::MissingReceipt(_))
    ));

    // Staged, not sealed.
    table.stage(&pool, &store, key, r.clone()).expect("stage");
    assert!(matches!(
        table.replay_landed(&store, key, EXEC_DEFAULT),
        Err(Refusal::Unsealed(_))
    ));

    // Sealed: a landing.
    table.seal(key).expect("seal row");
    let rep = table
        .replay_landed(&store, key, EXEC_DEFAULT)
        .expect("sealed row");
    assert_same_by_folds(&original, &rep, &r);

    // Torn: a field changed after the seal.
    table.row_mut(key).expect("row").receipt.roots ^= 1;
    assert!(matches!(
        table.replay_landed(&store, key, EXEC_DEFAULT),
        Err(Refusal::Torn(_))
    ));
    table.row_mut(key).expect("row").receipt.roots ^= 1;
    assert!(table.replay_landed(&store, key, EXEC_DEFAULT).is_ok());

    // Torn: the outcome changed after the seal. Commit -> Prune would stop
    // `next_version` landing the writes while every fold still matches.
    table.row_mut(key).expect("row").receipt.outcome = Outcome::Prune;
    assert!(matches!(
        table.replay_landed(&store, key, EXEC_DEFAULT),
        Err(Refusal::Torn(_))
    ));
    table.row_mut(key).expect("row").receipt.outcome = Outcome::Commit;
    assert!(table.replay_landed(&store, key, EXEC_DEFAULT).is_ok());

    // The landing replays without the pool.
    pool.free(slot).expect("free");
    let rep = table
        .replay_landed(&store, key, EXEC_DEFAULT)
        .expect("after free");
    assert_same_by_folds(&original, &rep, &r);
}

/// 10. The same plan in two different slots is the same derivation. Disable:
///     include `captured_in` in `identity`.
#[test]
fn identity_ignores_the_slot_but_not_the_seed() {
    let (store, base) = setup();
    let p = plan(base, 10, Outcome::Commit, 5);

    let mut pool_a = SlotPool::new(Eviction::Fifo);
    let (_, ta, ra) = run(&store, &mut pool_a, p.clone());

    let mut pool_b = SlotPool::new(Eviction::Fifo);
    pool_b.branch(500, None).expect("d1");
    let (d2, _) = pool_b.branch(501, None).expect("d2");
    pool_b.branch(502, None).expect("d3");
    pool_b.free(d2).expect("free d2");
    let (_, tb, rb) = run(&store, &mut pool_b, p.clone());

    // Anti-vacuity: the slots really differ.
    assert_ne!(ra.captured_in, rb.captured_in);

    assert_eq!(ra.identity(), rb.identity());
    let rep_a = receipt::replay(&ra, &store, EXEC_DEFAULT).expect("a");
    let rep_b = receipt::replay(&rb, &store, EXEC_DEFAULT).expect("b");
    assert_same_by_folds(&ta, &rep_a, &ra);
    assert_same_by_folds(&tb, &rep_b, &rb);
    assert_eq!(
        independent_fold(&rep_a.thought.writes),
        independent_fold(&rep_b.thought.writes)
    );

    // Fire: only the seed differs.
    let (_, _, rc) = run(&store, &mut pool_a, Plan { seed: 6, ..p });
    assert_ne!(ra.identity(), rc.identity());
}

fn data_ref(version: u64) -> BaseRef {
    BaseRef {
        dataset: 7,
        branch: 0,
        version,
    }
}

/// 11. A sealed landing is immutable. Disables: (a) `stage` inserts over an
///     existing row unconditionally; (b) `seal` overwrites an existing seal.
#[test]
fn a_sealed_landing_cannot_be_replaced_or_resealed() {
    let (mut store, base) = setup();
    let mut pool = SlotPool::new(Eviction::Fifo);
    let (_, original, r) = run(&store, &mut pool, plan(base, 10, Outcome::Commit, 5));
    let data = data_ref(2);
    store
        .commit_cycle(data, &main_version(), 1, std::slice::from_ref(&original))
        .expect("commit");
    let key = MetaKey { data, thought: 10 };
    let mut table = MetadataTable::default();

    // Crash recovery before the seal: the same receipt may be restaged; a
    // different one (same output and outcome, other roots) may not.
    table.stage(&pool, &store, key, r.clone()).expect("stage");
    table
        .stage(&pool, &store, key, r.clone())
        .expect("restage same");
    let other = Receipt {
        roots: r.roots ^ 0b0100,
        ..r.clone()
    };
    assert_ne!(other.identity(), r.identity(), "anti-vacuity");
    assert_eq!(
        table.stage(&pool, &store, key, other.clone()),
        Err(Refusal::StagedConflict(key))
    );

    table.seal(key).expect("seal");
    let sealed = table.row(key).expect("row").clone();

    // Under a sealed key: the same receipt is a no-op, anything else refuses.
    table
        .stage(&pool, &store, key, r.clone())
        .expect("restage landed");
    assert_eq!(table.row(key), Some(&sealed), "restage changed the row");
    assert_eq!(
        table.stage(&pool, &store, key, other),
        Err(Refusal::AlreadyLanded(key))
    );
    assert_eq!(
        table.row(key),
        Some(&sealed),
        "a refused stage changed the row"
    );

    // A row changed after its seal can neither land nor be resealed.
    table.row_mut(key).expect("row").receipt.roots ^= 0b0100;
    assert_eq!(table.seal(key), Err(Refusal::Torn(key)));
    assert!(matches!(
        table.replay_landed(&store, key, EXEC_DEFAULT),
        Err(Refusal::Torn(_))
    ));
    table.row_mut(key).expect("row").receipt.roots ^= 0b0100;
    table.seal(key).expect("reseal of an intact row is a no-op");
    assert!(table.replay_landed(&store, key, EXEC_DEFAULT).is_ok());

    // `captured_in` is outside identity but inside the seal.
    table
        .row_mut(key)
        .expect("row")
        .receipt
        .captured_in
        .generation ^= 1;
    assert!(matches!(
        table.replay_landed(&store, key, EXEC_DEFAULT),
        Err(Refusal::Torn(_))
    ));
}

/// 12. A landing must be in the cohort the write path committed into that
///     version. Disable: drop the `contribution_check` call from `landed` /
///     `stage` (an existing version then certifies any receipt).
#[test]
fn a_landing_needs_the_write_paths_commit_witness() {
    let (mut store, base) = setup();
    let mut pool = SlotPool::new(Eviction::Fifo);
    let (slot_a, ta, ra) = run(&store, &mut pool, plan(base, 10, Outcome::Commit, 5));
    let (_, tb, rb) = run(&store, &mut pool, plan(base, 11, Outcome::Commit, 9));
    let (_, tc, _) = run(&store, &mut pool, plan(base, 12, Outcome::Commit, 13));

    // Two thoughts coalesced into one version (EXO-55): both land.
    let v2 = data_ref(2);
    let w = store
        .commit_cycle(v2, &main_version(), 41, &[ta.clone(), tb.clone()])
        .expect("commit");
    assert_eq!(w.cohort.len(), 2);
    let mut table = MetadataTable::default();
    let (ka, kb) = (
        MetaKey {
            data: v2,
            thought: 10,
        },
        MetaKey {
            data: v2,
            thought: 11,
        },
    );
    for (k, r) in [(ka, &ra), (kb, &rb)] {
        table.stage(&pool, &store, k, r.clone()).expect("stage");
        table.seal(k).expect("seal");
        assert!(table.landed(&store, k).is_ok(), "{k:?} should land");
    }

    // 1/3. A valid receipt under an unrelated existing version.
    let v3 = data_ref(3);
    store
        .commit_cycle(v3, &main_version(), 42, std::slice::from_ref(&tc))
        .expect("commit v3");
    let k3 = MetaKey {
        data: v3,
        thought: 10,
    };
    assert_eq!(
        table.stage(&pool, &store, k3, ra.clone()),
        Err(Refusal::NotInCohort(k3))
    );
    // A version sealed outside the write path proves nothing.
    let v4 = data_ref(4);
    let landed = next_version(
        &store.givens.clone(),
        &main_version(),
        std::slice::from_ref(&ta),
    );
    store.seal(v4, landed).expect("plain seal");
    let k4 = MetaKey {
        data: v4,
        thought: 10,
    };
    assert_eq!(
        table.stage(&pool, &store, k4, ra.clone()),
        Err(Refusal::NoCommitWitness(k4))
    );
    // In the cohort, but with another output: a different run of thought 10.
    let (_, _, ra2) = run(&store, &mut pool, plan(base, 10, Outcome::Commit, 6));
    assert_ne!(ra2.output, ra.output, "anti-vacuity");
    let mut fresh = MetadataTable::default();
    assert_eq!(
        fresh.stage(&pool, &store, ka, ra2),
        Err(Refusal::ContributionMismatch(ka))
    );
    // Same thought and output, but committed with another outcome.
    let v5 = data_ref(5);
    let pruned = Thought::new(10, ta.writes.clone(), Outcome::Prune);
    store
        .commit_cycle(v5, &main_version(), 43, std::slice::from_ref(&pruned))
        .expect("commit v5");
    let k5 = MetaKey {
        data: v5,
        thought: 10,
    };
    assert_eq!(
        fresh.stage(&pool, &store, k5, ra.clone()),
        Err(Refusal::ContributionMismatch(k5))
    );

    // 4. After the live slot is released, replay needs a valid witness.
    pool.free(slot_a).expect("free");
    let rep = table
        .replay_landed(&store, ka, EXEC_DEFAULT)
        .expect("replay");
    assert_same_by_folds(&ta, &rep, &ra);
    store.witness_mut(v2).expect("witness").cohort.remove(&11);
    assert_eq!(
        table.replay_landed(&store, ka, EXEC_DEFAULT),
        Err(Refusal::WitnessTorn(ka))
    );

    // The write path refuses one thought twice, and a second batch under a
    // sealed version.
    let v6 = data_ref(6);
    assert_eq!(
        store.commit_cycle(v6, &main_version(), 44, &[ta.clone(), ta.clone()]),
        Err(CommitError::DuplicateThought(10))
    );
    assert_eq!(
        store.commit_cycle(v3, &main_version(), 45, std::slice::from_ref(&tc)),
        Err(CommitError::AlreadySealed(v3))
    );
}

/// 13. Missing metadata after the data write is reported, not certified.
///     Disable: `incomplete` returns an empty list.
#[test]
fn a_committed_thought_without_a_sealed_row_is_incomplete() {
    let (mut store, base) = setup();
    let mut pool = SlotPool::new(Eviction::Fifo);
    let (_, ta, ra) = run(&store, &mut pool, plan(base, 10, Outcome::Commit, 5));
    let (_, tb, rb) = run(&store, &mut pool, plan(base, 11, Outcome::Commit, 9));
    let v2 = data_ref(2);
    store
        .commit_cycle(v2, &main_version(), 7, &[ta, tb])
        .expect("commit");
    let mut table = MetadataTable::default();
    assert_eq!(table.incomplete(&store, v2), vec![10, 11]);
    let (ka, kb) = (
        MetaKey {
            data: v2,
            thought: 10,
        },
        MetaKey {
            data: v2,
            thought: 11,
        },
    );
    table.stage(&pool, &store, ka, ra).expect("stage a");
    table.seal(ka).expect("seal a");
    // Thought 11 crashed between its data write and its metadata write.
    table.stage(&pool, &store, kb, rb).expect("stage b");
    assert_eq!(table.incomplete(&store, v2), vec![11]);
    assert_eq!(table.landed(&store, kb), Err(Refusal::Unsealed(kb)));
    table.seal(kb).expect("seal b");
    assert!(table.incomplete(&store, v2).is_empty());
}

/// 14. Malformed inputs are typed refusals, never panics. Disables: drop
///     `inputs.check()` / `cfg.check()` from `run`, or from `stage`.
#[test]
fn malformed_inputs_refuse_without_panicking() {
    let (mut store, base) = setup();
    let mut pool = SlotPool::new(Eviction::Fifo);

    assert_eq!(
        ProgramInputs::try_new(vec![cell(0, 0)], vec![]),
        Err(Refusal::MalformedInputs {
            sources: 1,
            targets: 0
        })
    );

    // Public fields bypass the constructor: capture must refuse.
    let mut short = plan(base, 10, Outcome::Commit, 5);
    short.inputs.sources.pop();
    let (slot, _) = pool.branch(10, None).expect("branch");
    assert_eq!(
        receipt::capture(&pool, slot, &store, short).map(|_| ()),
        Err(Refusal::MalformedInputs {
            sources: 2,
            targets: 3
        })
    );
    let mut zero = plan(base, 10, Outcome::Commit, 5);
    zero.config.modulus = 0;
    assert_eq!(
        receipt::capture(&pool, slot, &store, zero).map(|_| ()),
        Err(Refusal::ZeroModulus)
    );

    // A receipt mutated after capture: replay, stage and seal refuse.
    let (t, r) =
        receipt::capture(&pool, slot, &store, plan(base, 10, Outcome::Commit, 5)).expect("capture");
    let mut bad = r.clone();
    bad.inputs.targets.push(cell(7, 2));
    assert!(matches!(
        receipt::replay(&bad, &store, EXEC_DEFAULT),
        Err(Refusal::MalformedInputs { .. })
    ));
    let mut zero_cfg = r.clone();
    zero_cfg.config.modulus = 0;
    assert_eq!(
        receipt::replay(&zero_cfg, &store, zero_cfg.config).map(|_| ()),
        Err(Refusal::ZeroModulus)
    );

    let data = data_ref(2);
    store
        .commit_cycle(data, &main_version(), 1, std::slice::from_ref(&t))
        .expect("commit");
    let key = MetaKey { data, thought: 10 };
    let mut table = MetadataTable::default();
    assert!(matches!(
        table.stage(&pool, &store, key, bad),
        Err(Refusal::MalformedInputs { .. })
    ));
    assert_eq!(
        table.stage(&pool, &store, key, zero_cfg),
        Err(Refusal::ZeroModulus)
    );
    // A staged row truncated before its seal cannot become a landing.
    table.stage(&pool, &store, key, r).expect("stage");
    table
        .row_mut(key)
        .expect("row")
        .receipt
        .inputs
        .sources
        .clear();
    assert!(matches!(
        table.seal(key),
        Err(Refusal::MalformedInputs { .. })
    ));
    assert!(matches!(
        table.landed(&store, key),
        Err(Refusal::Unsealed(_))
    ));
}
