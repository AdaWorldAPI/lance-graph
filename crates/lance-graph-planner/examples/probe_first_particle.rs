//! `PROBE-FIRST-PARTICLE-1` — the first cognitive transformation to exist
//! under the #1001 conservation laws (charter:
//! `.claude/plans/probe-revision-attention-view-1.md`).
//!
//! # The particle
//!
//! ```text
//!   same sealed state
//!      ├─ low-rung contribution      (grounded derivations)
//!      ├─ mid-rung contribution      (deeper derivations)
//!      └─ high-rung contribution     (frontier-depth derivations)
//!
//!   View A ──typed edit sequence──▶ View B
//!
//!   underlying state unchanged
//!   reconstruct(View A, edits) == View B
//! ```
//!
//! The claim after this probe, and nothing stronger: *the substrate changed
//! how it looked at the problem, and it can say exactly what changed.*
//!
//! # Why the rung ceiling is 6, not 8 — a measured constraint, kept visible
//!
//! `Belief::rung` is honest Tarski depth (`max(premise rungs)+1`), and
//! `close_transitive` composes span-doubling per pass, so a chain of L links
//! tops out at rung `ceil(log2 L)`. `Stamp::source(id)` is `1 << (id % 64)`
//! (`belief.rs:36-38`): evidential ids >= 64 ALIAS, so the largest
//! collision-free observed chain is 63 links -> rung ceiling 6. The charter's
//! example named R2/R5/R8 and marked the shape non-mandatory; this probe
//! reaches {1,2,3,4,6} across a 1..6 spread by derivation, not decoration.
//!
//! # Overlapping territories are views, NOT StyleLane identities
//!
//! The three windows below (low 0..=2 / mid 2..=5 / high 4..=6) deliberately
//! OVERLAP, mirroring the ruled reading of rung regions as descriptive
//! activation territories of one medium — not enum thresholds, not one-hot
//! rooms. Nothing here reads or writes `StyleLane`, `RungElevator`,
//! `EpistemicMode`, or any Frozen/Learned/Explore identifier: a territory is
//! a VIEW WINDOW a reader composes, and P2 proves one contribution can be
//! visible under two territories at once.
//!
//! # Production types only, plus two probe-local descriptors
//!
//! Production (unmodified): `BeliefArena`/`Belief::rung`/`close_transitive`,
//! `NodeRow`/`WitnessLens`/`CausalWitnessFacet`/`Locus`. Probe-local (this
//! file, per the charter's permission and its existing-container audit —
//! `ViewRegistry::union_of` is the shipped single-family precedent this
//! mirrors): `Selector`, `ViewPlan`, `ViewEdit`. `ViewEdit` stands in for the
//! Revision surface that production does not have (F-REVISION-FOCUS-1:
//! ABSENT, unchanged).

use lance_graph_contract::canonical_node::{EdgeBlock, NodeGuid, NodeRow};
use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_contract::witness_fabric::WitnessLens;
use lance_graph_planner::nars::belief::{BeliefArena, CStmt, Copula, Stamp};
use lance_graph_planner::nars::truth::TruthValue;

/// 63 links = the largest chain whose evidential stamps stay collision-free
/// (see module docs). Subjects 1..=64.
const CHAIN_NODES: u16 = 64;
/// Rung bands sampled into the row population: low {1,2}, mid {3,4}, high {6}.
/// 2 sits in low∩mid territory, 4 in mid∩high — the overlap witnesses.
const BAND_RUNGS: [u32; 5] = [1, 2, 3, 4, 6];
const ROWS_PER_BAND: usize = 4;

/// A typed selector. Two families, no shared representation: one reads a
/// signed nibble out of the row's witness register through the zero-copy
/// lens; the other reads a `u32` rung tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Selector {
    BoundAt(Locus),
    RungBand { lo: u32, hi: u32 },
}

struct Ctx<'a> {
    lens: WitnessLens<'a>,
    rung_of_row: &'a [u32],
}

impl Selector {
    fn admits(self, pos: usize, ctx: &Ctx<'_>) -> bool {
        match self {
            Selector::BoundAt(locus) => ctx.lens.at(pos).is_some_and(|f| f.is_bound(locus)),
            Selector::RungBand { lo, hi } => ctx
                .rung_of_row
                .get(pos)
                .is_some_and(|r| *r >= lo && *r <= hi),
        }
    }
}

/// The provenance-preserving composition: an ordered stack of typed
/// selectors. The lowered artifact (a closure) may be opaque; this may not.
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
    /// Terminal lowering into the existing predicate seam. Borrows only.
    fn lower<'c, 'a>(&'c self, ctx: &'c Ctx<'a>) -> impl Fn(usize) -> bool + use<'c, 'a> {
        move |pos| self.selectors.iter().all(|s| s.admits(pos, ctx))
    }
    fn visible(&self, ctx: &Ctx<'_>) -> Vec<usize> {
        let f = self.lower(ctx);
        (0..ctx.lens.len()).filter(|p| f(*p)).collect()
    }
}

/// The typed transformation — the behavioral-IR unit at probe level. A
/// sequence of these IS the edit; reconstruction runs the sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewEdit {
    Push(Selector),
    RemoveAt(usize),
}

fn apply_all(edits: &[ViewEdit], plan: &ViewPlan) -> ViewPlan {
    let mut next = plan.clone();
    for e in edits {
        match *e {
            ViewEdit::Push(s) => next.selectors.push(s),
            ViewEdit::RemoveAt(i) => {
                if i < next.selectors.len() {
                    next.selectors.remove(i);
                }
            }
        }
    }
    next
}

/// FNV-1a over key+value bytes of every row — the population witness.
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

/// FNV-1a over every belief's (s, p, rung, truth-bits) — the cognitive-state
/// witness ("same sealed state" is asserted, not assumed).
fn arena_digest(a: &BeliefArena) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for b in a.entries() {
        mix(u64::from(b.stmt.s));
        mix(u64::from(b.stmt.p));
        mix(u64::from(b.rung));
        mix(u64::from(b.truth.frequency.to_bits()));
        mix(u64::from(b.truth.confidence.to_bits()));
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

    // ═══ The sealed cognitive state: one observed chain, closed once ═══
    let mut arena = BeliefArena::new();
    for i in 1..CHAIN_NODES {
        arena.observe(
            inh(i, i + 1),
            TruthValue::new(1.0, 0.99),
            Stamp::source(u32::from(i) - 1), // ids 0..=62: collision-free
        );
    }
    arena.close_transitive(16);
    // Sealed from here on: nothing below mutates the arena.
    let arena_seal = arena_digest(&arena);

    // The field, as measured: per-rung belief counts.
    let max_rung = arena.entries().iter().map(|b| b.rung).max().unwrap_or(0);
    let histogram: Vec<(u32, usize)> = (0..=max_rung)
        .map(|r| (r, arena.entries().iter().filter(|b| b.rung == r).count()))
        .collect();

    println!("═══ PROBE-FIRST-PARTICLE-1 ═══\n");
    println!(
        "  one chain of {} observed links, closed once — {} beliefs, sealed\n",
        CHAIN_NODES - 1,
        arena.entries().len()
    );
    println!("  the field (per-rung activity in ONE state):");
    for (r, n) in &histogram {
        let bar = "▓".repeat((*n / 32).clamp(1, 40));
        println!("    R{r} {bar} ({n})");
    }
    println!();

    // ═══ P1 — the concurrent field spans three regions ═════════════════
    let present: Vec<u32> = histogram
        .iter()
        .filter(|(_, n)| *n > 0)
        .map(|(r, _)| *r)
        .collect();
    gates.push((
        "P1 rungs {1,2,3,4,6} all live in one sealed state (spread 5, three regions)",
        BAND_RUNGS.iter().all(|r| present.contains(r)) && max_rung >= 6,
        format!("present = {present:?}, ceiling = R{max_rung}"),
    ));

    // ═══ The row population: 4 rows per band, witness registers real ═══
    let mut picks: Vec<(u16, u32)> = Vec::new();
    for &band in &BAND_RUNGS {
        let mut found = 0usize;
        for b in arena.entries() {
            if b.rung == band && found < ROWS_PER_BAND {
                picks.push((b.stmt.s, b.rung));
                found += 1;
            }
        }
        assert_eq!(found, ROWS_PER_BAND, "band R{band} must have >= 4 beliefs");
    }
    let n_rows = picks.len();
    let mut rows: Vec<NodeRow> = (0..n_rows)
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
        WitnessLens::write_register(row, &f);
    }
    let rung_of_row: Vec<u32> = picks.iter().map(|(_, r)| *r).collect();
    let pop_seal = population_digest(&rows);
    let pop_ptr = rows.as_ptr();

    let ctx = Ctx {
        lens: WitnessLens::new(&rows),
        rung_of_row: &rung_of_row,
    };

    // ═══ P2 — territories overlap; one contribution, two phases ════════
    // Windows mirror the field reading: low 0..=2 / mid 2..=5 / high 4..=6.
    let low = Selector::RungBand { lo: 0, hi: 2 };
    let mid = Selector::RungBand { lo: 2, hi: 5 };
    let high = Selector::RungBand { lo: 4, hi: 6 };
    let in_view = |s: Selector, pos: usize| s.admits(pos, &ctx);
    let r2_row = rung_of_row.iter().position(|r| *r == 2).unwrap();
    let r4_row = rung_of_row.iter().position(|r| *r == 4).unwrap();
    let r1_row = rung_of_row.iter().position(|r| *r == 1).unwrap();
    let r3_row = rung_of_row.iter().position(|r| *r == 3).unwrap();
    let r6_row = rung_of_row.iter().position(|r| *r == 6).unwrap();
    gates.push((
        "P2 overlap: an R2 row is visible under low AND mid; an R4 row under mid AND high; exclusives exist",
        in_view(low, r2_row) && in_view(mid, r2_row)
            && in_view(mid, r4_row) && in_view(high, r4_row)
            && in_view(low, r1_row) && !in_view(mid, r1_row)
            && in_view(mid, r3_row) && !in_view(low, r3_row) && !in_view(high, r3_row)
            && in_view(high, r6_row) && !in_view(mid, r6_row),
        "R2∈low∩mid, R4∈mid∩high, R1 low-only, R3 mid-only, R6 high-only".to_string(),
    ));

    // ═══ P3 — THE PARTICLE: View A ── typed edits ──▶ View B ═══════════
    // A looks at the grounded territory through the attention base;
    // the Revision-shaped edit swaps the territory window to the frontier.
    let view_a = ViewPlan::of(&[Selector::BoundAt(Locus::SupportedBy), low]);
    let visible_a = view_a.visible(&ctx);

    let edits = [ViewEdit::RemoveAt(1), ViewEdit::Push(high)];
    let view_b = apply_all(&edits, &view_a);
    let visible_b = view_b.visible(&ctx);

    let reconstructed = apply_all(&edits, &view_a);
    let inverse = [ViewEdit::RemoveAt(1), ViewEdit::Push(low)];
    let restored = apply_all(&inverse, &view_b);

    gates.push((
        "P3 the particle: A and B differ, both non-empty; reconstruct(A, edits) == B on BOTH layers; inverse restores A",
        visible_a != visible_b
            && !visible_a.is_empty()
            && !visible_b.is_empty()
            && reconstructed == view_b
            && reconstructed.visible(&ctx) == visible_b
            && restored == view_a
            && restored.visible(&ctx) == visible_a,
        format!(
            "A{:?} -> B{:?}; plan_eq={} view_eq={} inverse_eq={}",
            visible_a,
            visible_b,
            reconstructed == view_b,
            reconstructed.visible(&ctx) == visible_b,
            restored == view_a
        ),
    ));

    // ═══ P4 — the universe did not move ════════════════════════════════
    let descriptor_bytes = view_b.selectors.len() * core::mem::size_of::<Selector>()
        + edits.len() * core::mem::size_of::<ViewEdit>();
    gates.push((
        "P4 sealed state unchanged: arena digest, population digest, and base pointer all identical",
        arena_digest(&arena) == arena_seal
            && population_digest(&rows) == pop_seal
            && core::ptr::eq(pop_ptr, rows.as_ptr()),
        format!(
            "arena 0x{arena_seal:016x} ✓, population 0x{pop_seal:016x} ✓, same ptr; \
             {descriptor_bytes} B of descriptors vs {} B of population never copied",
            n_rows * 512
        ),
    ));

    // ═══ P5 — controls: the view can speak and can stay silent ═════════
    let all = ViewPlan::default().visible(&ctx).len();
    let none = ViewPlan::of(&[Selector::RungBand { lo: 50, hi: 60 }])
        .visible(&ctx)
        .len();
    gates.push((
        "P5 control: empty plan sees every row; an off-field band sees none",
        all == n_rows && none == 0,
        format!("empty={all}/{n_rows}, off-field={none}"),
    ));

    // ═══ Report ════════════════════════════════════════════════════════
    println!("  View A {:?}", view_a.selectors);
    println!(
        "    visible {visible_a:?} (rungs {:?})",
        visible_a
            .iter()
            .map(|p| rung_of_row[*p])
            .collect::<Vec<_>>()
    );
    println!("  edits   {edits:?}");
    println!("  View B {:?}", view_b.selectors);
    println!(
        "    visible {visible_b:?} (rungs {:?})\n",
        visible_b
            .iter()
            .map(|p| rung_of_row[*p])
            .collect::<Vec<_>>()
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
        "\n── SCOPE (the charter's \"nothing stronger\") ──\n\
         F-REVISION-FOCUS-1 remains ABSENT: `ViewEdit` is probe-local; no\n\
         production Revision API carries a view edit. RungElevator, StyleLane,\n\
         EpistemicMode, temporal.rs: none appear in this probe. Territories are\n\
         view windows, not identities. Occupancy is semantic, not wall-clock.\n\
         No behavioral BPE: this is one reconstructible transformation, not a\n\
         learner. Rubicon persistence stays open."
    );
    assert!(all_green, "PROBE-FIRST-PARTICLE-1: a gate failed");
    println!("\nPROBE-FIRST-PARTICLE-1: ALL GATES GREEN");
}
