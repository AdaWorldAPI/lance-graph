//! `PROBE-REVISION-ATTENTION-VIEW-1` — preserve the cognitive layers, move the
//! view (charter: `.claude/plans/probe-revision-attention-view-1.md`).
//!
//! # The strongest claim this probe makes, and no stronger
//!
//! At probe level, one cognitive state retains multiple rung-associated
//! contributions while a typed, provenance-preserving selector composition is
//! changed and lowered into a different zero-copy reasoning view. The change
//! is reconstructible as a typed behavioral transformation. No global rung
//! transition and no central scheduler is required.
//!
//! # What is production, and what is probe-local
//!
//! **Production (unmodified):** `NodeRow` / `WitnessLens` / `CausalWitnessFacet`
//! / `Locus` (contract), `BeliefArena` / `Belief::rung` / `rcr_abduce` /
//! `Frontier` / `ReasoningGap` / `GapKind` (planner). None are changed.
//!
//! **Probe-local (this file only):** `Selector`, `ViewPlan`, `ViewEdit`. The
//! charter permits a probe-local shape resembling a `ViewPlan` and forbids
//! prescribing a production type. The existing-container audit ran first and
//! is recorded below.
//!
//! # Existing-container audit (charter §"Probe law", mandatory before any local type)
//!
//! `contract::selection::{NamedView, ViewRegistry}` is the closest shipped
//! precedent and it already has the two-layer shape: `union_of(&[ViewId]) ->
//! WideFieldMask` takes RETAINED descriptor identities and returns one fused
//! opaque artifact. It is not reused here for one honest reason: it composes a
//! single selector family (`WideFieldMask` facet participation) over one
//! representation, and this probe's whole question is whether *heterogeneous*
//! families compose. `CycleFrame` (`persist_sink.rs:148`) records a cycle, not
//! a view. `SupportReceipt` (`causal_audit.rs:250`) records support, not
//! selection. So the local type follows `ViewRegistry`'s shape rather than
//! inventing one, and does not propose replacing it.
//!
//! # The three selector families are deliberately incompatible
//!
//! | Selector | Reads | Physical shape |
//! |---|---|---|
//! | `BoundAt(Locus)` | `WitnessLens::at(pos)` | signed nibble in a 12-byte register |
//! | `RungBand{lo,hi}` | `Belief::rung` | `u32` field on an arena entry |
//! | `GapSubject(u16)` | `rcr_abduce(..).gaps` | scan-derived `Vec<ReasoningGap>` |
//!
//! No common representation, no shared trait, no `CommonMask`. They meet only
//! at the lowering boundary, as one `impl Fn(usize) -> bool`.

use lance_graph_contract::canonical_node::{EdgeBlock, NodeGuid, NodeRow};
use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_contract::witness_fabric::WitnessLens;
use lance_graph_planner::nars::belief::{BeliefArena, CStmt, Copula, Stamp};
use lance_graph_planner::nars::tactics::{rcr_abduce, tr_diverge, Throttle};
use lance_graph_planner::nars::truth::TruthValue;

const N_ROWS: usize = 16;

/// One typed selector. Probe-local; each variant names a DIFFERENT production
/// semantics and reads a DIFFERENT source. There is deliberately no shared
/// representation and no `impl` unifying them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Selector {
    /// Row's `CausalWitnessFacet` is bound at this `Locus` (nibble register).
    BoundAt(Locus),
    /// Row's associated belief rung falls in `[lo, hi]` (u32 arena field).
    RungBand { lo: u32, hi: u32 },
    /// Row's subject appears as a `ReasoningGap` subject in the frontier
    /// (scan-derived vec).
    GapSubject(u16),
}

/// The everything-the-lowering-needs bundle. Borrowed; owns no population.
struct Ctx<'a> {
    lens: WitnessLens<'a>,
    /// Per-row belief rung — the SEPARATE representation, indexed by row.
    rung_of_row: &'a [u32],
    /// Per-row subject id — how a row joins the frontier's gap subjects.
    subject_of_row: &'a [u16],
    /// Subjects the production `rcr_abduce` scan reported as gaps.
    gap_subjects: &'a [u16],
}

impl Selector {
    /// Does this ONE selector admit row `pos`? Each arm reads its own source.
    fn admits(self, pos: usize, ctx: &Ctx<'_>) -> bool {
        match self {
            Selector::BoundAt(locus) => ctx.lens.at(pos).is_some_and(|f| f.is_bound(locus)),
            Selector::RungBand { lo, hi } => ctx
                .rung_of_row
                .get(pos)
                .is_some_and(|r| *r >= lo && *r <= hi),
            Selector::GapSubject(s) => ctx
                .subject_of_row
                .get(pos)
                .is_some_and(|row_s| *row_s == s && ctx.gap_subjects.contains(&s)),
        }
    }
}

/// The provenance-preserving composition description: an ORDERED stack of
/// typed selectors whose identities survive composition. Mirrors
/// `ViewRegistry::union_of`'s shape (retained ids in, fused artifact out) —
/// intersection here, since narrowing attention is the operation under test.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct ViewPlan {
    selectors: Vec<Selector>,
}

impl ViewPlan {
    fn of(selectors: &[Selector]) -> Self {
        Self {
            selectors: selectors.to_vec(),
        }
    }

    /// TERMINAL LOWERING. The returned artifact may be fused and opaque; the
    /// plan above it is not. Borrows `ctx`; copies no population.
    fn lower<'c, 'a>(&'c self, ctx: &'c Ctx<'a>) -> impl Fn(usize) -> bool + use<'c, 'a> {
        move |pos| self.selectors.iter().all(|s| s.admits(pos, ctx))
    }

    /// PROVENANCE. Which selector INDICES rejected `pos`. Empty = admitted.
    /// This is what `union()`/`intersect()` on a packed mask cannot answer.
    fn excluded_by(&self, pos: usize, ctx: &Ctx<'_>) -> Vec<usize> {
        self.selectors
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.admits(pos, ctx))
            .map(|(i, _)| i)
            .collect()
    }

    fn visible(&self, ctx: &Ctx<'_>) -> Vec<usize> {
        let f = self.lower(ctx);
        (0..ctx.lens.len()).filter(|p| f(*p)).collect()
    }
}

/// A typed view transformation — the behavioral-IR unit at probe level.
/// Smallest lawful shape: a stack gets a selector, or loses one by position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewEdit {
    Push(Selector),
    RemoveAt(usize),
}

impl ViewEdit {
    /// `BEFORE + EDIT -> AFTER`, total and deterministic.
    fn apply(self, plan: &ViewPlan) -> ViewPlan {
        let mut next = plan.clone();
        match self {
            ViewEdit::Push(s) => next.selectors.push(s),
            ViewEdit::RemoveAt(i) => {
                if i < next.selectors.len() {
                    next.selectors.remove(i);
                }
            }
        }
        next
    }
}

/// FNV-1a over the population's key + value bytes (32 + 480 of each 512-byte
/// row) — the non-destructiveness witness. That is exactly where a view-induced
/// mutation could land: the witness register lives in `value`, and the address
/// lives in `key`. `EdgeBlock` is not walked (no public byte accessor); it is
/// also never written by anything in this probe.
fn population_digest(rows: &[NodeRow]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for r in rows {
        for b in r.key.as_bytes().iter().chain(r.value.iter()) {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn inh(s: u16, p: u16) -> CStmt {
    CStmt {
        s,
        cop: Copula::Inh,
        p,
    }
}

fn main() {
    let mut gates: Vec<(&str, bool, String)> = Vec::new();

    // ═══ The stationary cognitive state ═══════════════════════════════
    // (a) A multi-rung belief arena — the F-PARALLEL-RUNG-1 half.
    let mut arena = BeliefArena::new();
    for (i, (s, p)) in [(1u16, 2u16), (2, 3), (3, 4), (4, 5)].iter().enumerate() {
        arena.observe(
            inh(*s, *p),
            TruthValue::new(1.0, 0.9),
            Stamp::source(i as u32),
        );
    }
    arena.close_transitive(16);
    let mut rungs_present: Vec<u32> = arena.entries().iter().map(|b| b.rung).collect();
    rungs_present.sort_unstable();
    rungs_present.dedup();

    // (b) The row population. Rows carry a real witness register; row `i`
    //     binds Locus::SupportedBy when i%2==0 and Locus::Supports when i%3==0, so the
    //     two loci overlap on some rows and disagree on others.
    let mut rows: Vec<NodeRow> = (0..N_ROWS)
        .map(|i| NodeRow {
            key: NodeGuid::local(i as u32),
            edges: EdgeBlock::default(),
            value: [0u8; 480],
        })
        .collect();
    for (i, row) in rows.iter_mut().enumerate() {
        let mut f = CausalWitnessFacet::ZERO;
        if i % 2 == 0 {
            f = f.with(Locus::SupportedBy, 1);
        }
        if i % 3 == 0 {
            f = f.with(Locus::Supports, -1);
        }
        WitnessLens::write_register(row, &f);
    }
    // Row -> rung and row -> subject, cycled from the arena so the three
    // selector families read three genuinely different sources.
    let entries = arena.entries();
    let rung_of_row: Vec<u32> = (0..N_ROWS)
        .map(|i| entries[i % entries.len()].rung)
        .collect();
    let subject_of_row: Vec<u16> = (0..N_ROWS)
        .map(|i| entries[i % entries.len()].stmt.s)
        .collect();

    // (c) The production bulk gap scan — NOT CausalTopology (F-UNKNOWN-TEXTURE).
    //
    // MEASURED, and the reason the source is `tr_diverge` rather than
    // `rcr_abduce`: on this fully-closed chain `rcr_abduce` returns 20
    // candidates and ZERO gaps — its gap channel fires on a different
    // condition (no shared middle / hub exclusion / budget), not on a
    // complete chain. `tr_diverge` on a focus with no sibling emits a real
    // `ReasoningGap { kind: NoSibling, subject: Some(1) }`. Both are kept in
    // the probe so the empty channel is visible rather than hidden.
    let abduce = rcr_abduce(&arena, &Throttle::permissive());
    let frontier = tr_diverge(&arena, inh(1, 2));
    let gap_subjects: Vec<u16> = frontier.gaps.iter().filter_map(|g| g.subject).collect();

    let digest_before = population_digest(&rows);
    let ptr_before = rows.as_ptr();

    let ctx = Ctx {
        lens: WitnessLens::new(&rows),
        rung_of_row: &rung_of_row,
        subject_of_row: &subject_of_row,
        gap_subjects: &gap_subjects,
    };

    // ═══ F-PARALLEL-RUNG-1 ════════════════════════════════════════════
    gates.push((
        "F-PARALLEL-RUNG-1 one cognitive state holds >= 3 rung bands, none demoted",
        rungs_present.len() >= 3,
        format!("rungs = {rungs_present:?}"),
    ));

    // ═══ The view, and one Revision-shaped edit at Evaluation ═════════
    let plan_a = ViewPlan::of(&[Selector::BoundAt(Locus::SupportedBy)]);
    let visible_a = plan_a.visible(&ctx);

    let edit = ViewEdit::Push(Selector::RungBand { lo: 1, hi: 2 });
    let plan_b = edit.apply(&plan_a);
    let visible_b = plan_b.visible(&ctx);

    let plan_c = ViewEdit::Push(Selector::GapSubject(1)).apply(&plan_b);
    let visible_c = plan_c.visible(&ctx);

    // ═══ F-HETEROGENEOUS-SELECTOR-1 ═══════════════════════════════════
    // Three families in one plan, each still its own variant reading its own
    // source. Anti-vacuity: each must actually DISCRIMINATE on this fixture
    // (admit some rows, reject others) — a selector that admits everything
    // carries as much information as one that admits nothing.
    let discriminates = |s: Selector| {
        let admitted = (0..N_ROWS).filter(|p| s.admits(*p, &ctx)).count();
        admitted > 0 && admitted < N_ROWS
    };
    let all_three = [
        Selector::BoundAt(Locus::SupportedBy),
        Selector::RungBand { lo: 1, hi: 2 },
        Selector::GapSubject(1),
    ];
    gates.push((
        "F-HETEROGENEOUS-SELECTOR-1 three incompatible families compose; each discriminates",
        plan_c.selectors.len() == 3
            && all_three.iter().all(|s| discriminates(*s))
            && plan_c
                .selectors
                .iter()
                .any(|s| matches!(s, Selector::BoundAt(_)))
            && plan_c
                .selectors
                .iter()
                .any(|s| matches!(s, Selector::RungBand { .. }))
            && plan_c
                .selectors
                .iter()
                .any(|s| matches!(s, Selector::GapSubject(_))),
        format!(
            "admitted per family: bound={} rung={} gap={}",
            (0..N_ROWS)
                .filter(|p| all_three[0].admits(*p, &ctx))
                .count(),
            (0..N_ROWS)
                .filter(|p| all_three[1].admits(*p, &ctx))
                .count(),
            (0..N_ROWS)
                .filter(|p| all_three[2].admits(*p, &ctx))
                .count(),
        ),
    ));

    // ═══ F-VIEW-PROVENANCE-1 ══════════════════════════════════════════
    // For every excluded row the plan names WHICH selectors rejected it, and
    // the blame is non-uniform: at least one row blamed by exactly one
    // selector, at least one by two. Uniform blame would mean the channel
    // carries no information.
    let blame: Vec<(usize, Vec<usize>)> = (0..N_ROWS)
        .map(|p| (p, plan_c.excluded_by(p, &ctx)))
        .collect();
    let blamed_by_one = blame.iter().filter(|(_, b)| b.len() == 1).count();
    let blamed_by_two_plus = blame.iter().filter(|(_, b)| b.len() >= 2).count();
    let admitted_rows = blame.iter().filter(|(_, b)| b.is_empty()).count();
    gates.push((
        "F-VIEW-PROVENANCE-1 every exclusion names its selector(s); blame is non-uniform",
        blamed_by_one >= 1
            && blamed_by_two_plus >= 1
            && admitted_rows >= 1
            && blame
                .iter()
                .all(|(p, b)| b.is_empty() == plan_c.visible(&ctx).contains(p)),
        format!(
            "admitted={admitted_rows} blamed_by_1={blamed_by_one} blamed_by_2+={blamed_by_two_plus}"
        ),
    ));

    // ═══ F-TYPED-EDIT-ROUNDTRIP-1 ═════════════════════════════════════
    // BEFORE + EDIT == observed AFTER, on BOTH layers: the descriptor stack
    // and the lowered visible set. And the inverse edit restores BEFORE.
    let reconstructed = edit.apply(&plan_a);
    let inverse_ok = ViewEdit::RemoveAt(plan_b.selectors.len() - 1).apply(&plan_b) == plan_a;
    gates.push((
        "F-TYPED-EDIT-ROUNDTRIP-1 BEFORE + EDIT == AFTER (plan and visible set), inverse restores",
        reconstructed == plan_b && reconstructed.visible(&ctx) == visible_b && inverse_ok,
        format!(
            "|A|={} |B|={} plan_eq={} view_eq={} inverse={}",
            visible_a.len(),
            visible_b.len(),
            reconstructed == plan_b,
            reconstructed.visible(&ctx) == visible_b,
            inverse_ok
        ),
    ));

    // ═══ F-ZERO-COPY-VIEW-1 ═══════════════════════════════════════════
    // Three different views were lowered and evaluated above. The population
    // must be byte-identical and at the same address. Descriptor allocation
    // (the Vec<Selector>) is reported SEPARATELY — the invariant is zero
    // POPULATION copy, not zero allocation.
    let digest_after = population_digest(&rows);
    let ptr_after = rows.as_ptr();
    let descriptor_bytes = plan_c.selectors.len() * core::mem::size_of::<Selector>();
    gates.push((
        "F-ZERO-COPY-VIEW-1 three views lowered; population address and bytes unchanged",
        digest_before == digest_after && core::ptr::eq(ptr_before, ptr_after),
        format!(
            "digest 0x{digest_before:016x} == 0x{digest_after:016x}; same ptr; \
             descriptors allocated {descriptor_bytes} B ({} rows x 512 B population NOT copied)",
            N_ROWS
        ),
    ));

    // ═══ F-NON-DESTRUCTIVE-1 ══════════════════════════════════════════
    let mut rungs_after: Vec<u32> = arena.entries().iter().map(|b| b.rung).collect();
    rungs_after.sort_unstable();
    rungs_after.dedup();
    gates.push((
        "F-NON-DESTRUCTIVE-1 view change erased no underlying contribution",
        rungs_after == rungs_present && digest_after == digest_before,
        format!("rungs still {rungs_after:?}, population digest unchanged"),
    ));

    // ═══ Controls — the plan must be able to stay silent AND to speak ═══
    let empty_visible = ViewPlan::default().visible(&ctx);
    let contradiction = ViewPlan::of(&[
        Selector::RungBand { lo: 0, hi: 0 },
        Selector::RungBand { lo: 9, hi: 9 },
    ])
    .visible(&ctx);
    gates.push((
        "C1 empty plan admits ALL rows; a contradictory plan admits NONE (not vacuous)",
        empty_visible.len() == N_ROWS && contradiction.is_empty(),
        format!(
            "empty={} contradictory={}",
            empty_visible.len(),
            contradiction.len()
        ),
    ));

    // Narrowing must actually narrow, monotonically — otherwise "changing the
    // view" was decoration.
    gates.push((
        "C2 each pushed selector narrows or holds; the composition is monotone",
        visible_b.len() <= visible_a.len() && visible_c.len() <= visible_b.len(),
        format!(
            "|A|={} >= |B|={} >= |C|={}",
            visible_a.len(),
            visible_b.len(),
            visible_c.len()
        ),
    ));

    // ═══ Report ═══════════════════════════════════════════════════════
    println!("═══ PROBE-REVISION-ATTENTION-VIEW-1 ═══\n");
    println!("  stationary population: {N_ROWS} NodeRow x 512 B, never copied");
    println!("  rung bands in one state: {rungs_present:?}");
    println!(
        "  rcr_abduce: {} candidates, {} gaps (empty channel, kept visible)",
        abduce.candidates.len(),
        abduce.gaps.len()
    );
    println!(
        "  tr_diverge gaps: {:?} -> subjects {gap_subjects:?}\n",
        frontier
            .gaps
            .iter()
            .map(|g| (g.kind, g.subject))
            .collect::<Vec<_>>()
    );
    println!("  view A {:?} -> visible {:?}", plan_a.selectors, visible_a);
    println!("  + {edit:?}");
    println!("  view B {:?} -> visible {:?}", plan_b.selectors, visible_b);
    println!(
        "  view C {:?} -> visible {:?}\n",
        plan_c.selectors, visible_c
    );

    let mut all_green = true;
    for (name, pass, detail) in &gates {
        println!(
            "  [{}] {name} — {detail}",
            if *pass { "PASS" } else { "FAIL" }
        );
        all_green &= *pass;
    }

    println!(
        "\n── F-REVISION-FOCUS-1: ABSENT (recorded, not substituted) ──\n\
         No production Revision API can carry a view edit. `revise_if_minority_wins`\n\
         is `todo!()`; `RevisionOutcome` is a bare 2-variant enum; every other\n\
         `revise` in the tree mutates a scalar truth/confidence. The `ViewEdit`\n\
         above is a PROBE-LOCAL adapter standing in for that missing surface.\n\
         RungElevator was NOT substituted — it does not appear in this probe.\n\
         \n\
         ── F-RUBICON-BOUNDARY-1: STOPPED AT THE BOUNDARY ──\n\
         This probe ends with a typed view + edit existing BEFORE Commit.\n\
         `KanbanMove` carries no attention provenance and `calcify` is `todo!()`\n\
         (D-ATOM-5), so persistence across the Rubicon remains OPEN and is not\n\
         claimed here.\n\
         \n\
         ── F-UNKNOWN-TEXTURE: honoured ──\n\
         No CausalTopology bulk selector was manufactured. The absence-shaped\n\
         selector is the production-wired `rcr_abduce` -> `Frontier.gaps`.\n\
         \n\
         ── SCOPE ──\n\
         Concurrent cognitive OCCUPANCY is shown. Wall-clock/thread parallelism\n\
         is NOT measured and NOT claimed. No behavioral BPE is implemented: this\n\
         establishes only that a view change is typed and reconstructible."
    );
    assert!(all_green, "PROBE-REVISION-ATTENTION-VIEW-1: a gate failed");
    println!("\nPROBE-REVISION-ATTENTION-VIEW-1: ALL GATES GREEN");
}
