//! PROBE-COPULA-GROUP-MASK-1 — is `Copula` an ergonomic READING over resident
//! relation rows + selection geometry, rather than an identity type?
//!
//! **The correction this probe serves (operator, 2026-08-23).** The Step 2
//! ruling request drifted toward `Copula → relation concept → classid
//! reference`. **That drift is retracted.** The C1–C4 result established
//! only `COPULA ≠ RAIL PLACEMENT`; it did NOT establish `COPULA = CLASSID`,
//! and the two conclusions are unrelated. The root law:
//!
//! ```text
//!   CONTENT NEVER TRAVELS IN CLASSID.
//!   CLASSID SELECTS THE READING.
//!
//!   classid = HOW these bytes may be read
//!   HHTL    = WHERE the resident thing lives
//!   mask    = WHAT part / group / region conducts
//!   edges   = HOW addressed things relate
//! ```
//!
//! # The hypothesis under measurement (NOT adopted as architecture)
//!
//! The Active-Directory shape: a DN gives an object a hierarchical home,
//! but `member`/`memberOf` are NOT ancestry — they are a many-to-many
//! relation over objects that already have addresses, with the two lookup
//! directions being inverse VIEWS over ONE relation, never two truths.
//!
//! Carried here: ONE resident relation row —
//!
//! ```text
//!   RelRow { subject_address, object_address, copula(content, RESIDENT),
//!            truth, provenance }
//! ```
//!
//! — with group masks ("inheritance-like", "similarity-like", …) as
//! DERIVED selection ergonomics over those rows, composable with HHTL
//! region masks and signed-witness conditions in one pass:
//!
//! ```text
//!   group membership ∩ HHTL region ∩ not-falsified
//!     = one execution selection, no materialized object set
//! ```
//!
//! # What each gate is (the operator's falsifiers, run as code)
//!
//! | gate | falsifier it runs |
//! |---|---|
//! | G-DIST | measure the distribution BEFORE buying a representation |
//! | G-F1 | every copula distinction reconstructs EXACTLY from rows (group masks alone provably cannot — Rel's verb lives in the row) |
//! | G-F2 | `members`/`memberOf` are two views over ONE relation, no duplicated canonical state |
//! | G-F4 | a symmetric cross-subtree relation is NOT forced into HHTL ancestry |
//! | G-F5 | ALL rows share ONE classid while copulas differ — content never in classid |
//! | G-F6 | regrouping/inserting never moves or repacks resident rows |
//! | G-F8 | truth/provenance sit on the ROW (the claim), never on the group |
//! | G-F10 | mask-vs-sparse-rows cost compared on the MEASURED distribution |
//!
//! # Honesty box
//!
//! - **Fixture bias, stated:** the measured corpus is arena-closure output
//!   plus hand-added Sim/Impl/Rel rows. It is Inh-dominated (closure only
//!   derives Inh/Sim), so the density comparison covers that regime only.
//!   The KJV right-corner corpus (Rel-heavy) and tactics output (Impl) are
//!   the distributions a workload-scale measurement still needs.
//! - The group table and `RelRow` are PROBE-LOCAL. Nothing is minted; no
//!   tenant is proposed. A verdict here feeds the Step 2 addendum, which
//!   proposes NO mint until measurements rule out composition.
//! - G-F10's byte counts are fixture-scale arithmetic, not a workload
//!   benchmark.

use lance_graph_contract::attention_facet::{AttentionFocusFacet, RowFocusMask};
use lance_graph_contract::facet::{FacetCascade, FacetTier};
use lance_graph_planner::nars::belief::{BeliefArena, CStmt, Copula, Stamp};
use lance_graph_planner::nars::truth::TruthValue;
use std::collections::HashMap;

/// ONE classid for EVERY relation row, whatever its copula — G-F5's subject.
const RELATION_ROW_CLASSID: u32 = 0xFFFF_0010;

fn addr_of_term(term: u16) -> [u8; 16] {
    // Terms 1..=5 live in subtree 0x40; 6..=9 in 0x50 (two parents, so the
    // fixture genuinely crosses subtrees).
    let parent = if term < 6 { 0x40 } else { 0x50 };
    FacetCascade {
        facet_classid: RELATION_ROW_CLASSID,
        tiers: [
            FacetTier {
                hi: parent,
                lo: term as u8,
            },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
        ],
    }
    .to_bytes()
}

fn focus_of_addr(a: &[u8; 16]) -> AttentionFocusFacet {
    AttentionFocusFacet::exact(FacetCascade::from_bytes(a))
}

/// ONE resident relation row. The copula CONTENT stays resident IN THE ROW
/// (tag + verb) — groups are derived readings over it, never its carrier.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct RelRow {
    subject: [u8; 16],
    object: [u8; 16],
    /// Resident relation content: (tag, verb). Tag 0..=3 = Inh/Sim/Impl/Rel;
    /// verb meaningful only for Rel. NOT a classid; never leaves the row.
    cop_tag: u8,
    cop_verb: u16,
    truth_f: u32,
    truth_c: u32,
    stamp: u64,
}

fn tag_of(c: Copula) -> (u8, u16) {
    match c {
        Copula::Inh => (0, 0),
        Copula::Sim => (1, 0),
        Copula::Impl => (2, 0),
        Copula::Rel(v) => (3, v),
    }
}

fn copula_of(row: &RelRow) -> Copula {
    match row.cop_tag {
        0 => Copula::Inh,
        1 => Copula::Sim,
        2 => Copula::Impl,
        _ => Copula::Rel(row.cop_verb),
    }
}

/// The probe-local group table: coarse relation FAMILIES. Groups are
/// selection ergonomics — the exact copula stays in the row (G-F1 proves the
/// group alone cannot reconstruct `Rel`'s verb, which is the point).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Group {
    InheritanceLike = 0,
    SimilarityLike = 1,
    ImplicationLike = 2,
    RelFamily = 3,
}

fn group_of(row: &RelRow) -> Group {
    match row.cop_tag {
        0 => Group::InheritanceLike,
        1 => Group::SimilarityLike,
        2 => Group::ImplicationLike,
        _ => Group::RelFamily,
    }
}

/// `members(group)` — a VIEW over the one relation (filter, no second table).
fn members(rows: &[RelRow], g: Group) -> impl Iterator<Item = &RelRow> {
    rows.iter().filter(move |r| group_of(r) == g)
}

/// `member_of(row)` — the inverse VIEW over the SAME relation.
fn member_of(row: &RelRow) -> Group {
    group_of(row)
}

fn main() {
    let mut pass = 0u32;
    let mut gate = |name: &str, ok: bool, detail: String| {
        assert!(ok, "[FAIL] {name} — {detail}");
        println!("  [PASS] {name} — {detail}");
        pass += 1;
    };

    // ================= Build the measured corpus =================
    // Arena closure output (Inh chain 1→5, the standing fixture) + hand
    // rows for the other copulas, including a cross-subtree Sim.
    let mut arena = BeliefArena::new();
    for (k, (s, p)) in [(1u16, 2u16), (2, 3), (3, 4), (4, 5)].iter().enumerate() {
        arena.observe(
            CStmt {
                s: *s,
                cop: Copula::Inh,
                p: *p,
            },
            TruthValue::new(1.0, 0.9),
            Stamp::source(k as u32 + 1),
        );
    }
    arena.close_transitive(16);

    let mut rows: Vec<RelRow> = arena
        .entries()
        .iter()
        .map(|b| {
            let (t, v) = tag_of(b.stmt.cop);
            RelRow {
                subject: addr_of_term(b.stmt.s),
                object: addr_of_term(b.stmt.p),
                cop_tag: t,
                cop_verb: v,
                truth_f: b.truth.frequency.to_bits(),
                truth_c: b.truth.confidence.to_bits(),
                stamp: b.stamp.0,
            }
        })
        .collect();
    // Cross-subtree Sim (2 ↔ 7), an Impl (3 ⇒ 8), and two Rel verbs.
    for (s, p, c) in [
        (2u16, 7u16, Copula::Sim),
        (3, 8, Copula::Impl),
        (1, 9, Copula::Rel(7)),
        (4, 9, Copula::Rel(12)),
    ] {
        let (t, v) = tag_of(c);
        rows.push(RelRow {
            subject: addr_of_term(s),
            object: addr_of_term(p),
            cop_tag: t,
            cop_verb: v,
            truth_f: TruthValue::new(1.0, 0.9).frequency.to_bits(),
            truth_c: TruthValue::new(1.0, 0.9).confidence.to_bits(),
            stamp: Stamp::source(20 + s as u32).0,
        });
    }

    // ---- G-DIST — measure BEFORE buying a representation ----
    let mut per_group: HashMap<u8, usize> = HashMap::new();
    let mut fan_out: HashMap<[u8; 16], usize> = HashMap::new();
    let mut fan_in: HashMap<[u8; 16], usize> = HashMap::new();
    let mut terms: Vec<[u8; 16]> = Vec::new();
    for r in &rows {
        *per_group.entry(r.cop_tag).or_default() += 1;
        *fan_out.entry(r.subject).or_default() += 1;
        *fan_in.entry(r.object).or_default() += 1;
        for a in [r.subject, r.object] {
            if !terms.contains(&a) {
                terms.push(a);
            }
        }
    }
    let n = rows.len();
    let t = terms.len();
    let max_fan_out = fan_out.values().copied().max().unwrap_or(0);
    let max_fan_in = fan_in.values().copied().max().unwrap_or(0);
    // Occupancy of the full many-to-many space, per group family:
    let dense_cells = t * t * 4;
    let sparsity = n as f64 / dense_cells as f64;
    gate(
        "G-DIST distribution measured before any representation choice",
        n == 14 && per_group[&0] == 10 && per_group[&1] == 1 && per_group[&3] == 2,
        format!(
            "rows={n} over {t} terms: Inh={} Sim={} Impl={} Rel={}; max fan-out={} \
             max fan-in={}; occupancy {n}/{dense_cells} = {:.3}% — SPARSE, which is \
             the datum every later choice must answer to",
            per_group[&0],
            per_group[&1],
            per_group[&2],
            per_group[&3],
            max_fan_out,
            max_fan_in,
            sparsity * 100.0
        ),
    );

    // ---- G-F1 — exact reconstruction from ROWS; groups alone CANNOT ----
    let mut f1_ok = true;
    for r in &rows {
        let c = copula_of(r);
        let (tag, verb) = tag_of(c);
        f1_ok &= tag == r.cop_tag && verb == r.cop_verb;
    }
    // The two Rel rows share a GROUP but differ in verb — the group reading
    // is lossy BY DESIGN, so the verb must be resident row content.
    let rels: Vec<&RelRow> = members(&rows, Group::RelFamily).collect();
    let group_lossy = rels.len() == 2
        && group_of(rels[0]) == group_of(rels[1])
        && copula_of(rels[0]) != copula_of(rels[1]);
    gate(
        "G-F1 copulas reconstruct exactly from rows; the group is lossy by design",
        f1_ok && group_lossy,
        format!(
            "{n}/{n} rows round-trip (tag, verb) exactly; Rel(7) and Rel(12) share one \
             group but stay distinct copulas — content lives in the ROW, the group is \
             ergonomics"
        ),
    );

    // ---- G-F2 — members / memberOf: two views, ONE relation ----
    let before_bytes: Vec<RelRow> = rows.clone();
    let mut f2_ok = true;
    for g in [
        Group::InheritanceLike,
        Group::SimilarityLike,
        Group::ImplicationLike,
        Group::RelFamily,
    ] {
        for r in members(&rows, g) {
            f2_ok &= member_of(r) == g;
        }
    }
    let total_via_groups: usize = [
        Group::InheritanceLike,
        Group::SimilarityLike,
        Group::ImplicationLike,
        Group::RelFamily,
    ]
    .iter()
    .map(|&g| members(&rows, g).count())
    .sum();
    gate(
        "G-F2 members/memberOf are inverse views over ONE relation",
        f2_ok && total_via_groups == n && rows == before_bytes,
        format!(
            "every members(g) row answers memberOf(row)==g; the 4 group views partition \
             all {n} rows; and the resident rows are byte-identical after both lookups — \
             no duplicated canonical state"
        ),
    );

    // ---- G-F4 — a symmetric cross-subtree relation is NOT ancestry ----
    let sim = rows.iter().find(|r| r.cop_tag == 1).expect("the Sim row");
    let fs = focus_of_addr(&sim.subject);
    let fo = focus_of_addr(&sim.object);
    gate(
        "G-F4 many-to-many topology is not faked into HHTL ancestry",
        !fs.covers(fo) && !fo.covers(fs) && copula_of(sim) == Copula::Sim,
        "the Sim pair spans two subtrees (0x40.* ↔ 0x50.*): neither address covers the \
         other, and the relation exists ONLY as a row — the hierarchy gives both ends a \
         home, it does not pretend to BE the relation"
            .to_string(),
    );

    // ---- G-F5 — content never travels in classid ----
    let one_classid = rows.iter().all(|r| {
        FacetCascade::from_bytes(&r.subject).facet_classid == RELATION_ROW_CLASSID
            && FacetCascade::from_bytes(&r.object).facet_classid == RELATION_ROW_CLASSID
    });
    let copulas_differ = rows
        .iter()
        .map(|r| r.cop_tag)
        .collect::<std::collections::HashSet<_>>();
    gate(
        "G-F5 ONE classid across all rows while four copulas differ",
        one_classid && copulas_differ.len() == 4,
        "every address carries the SAME classid; Inh/Sim/Impl/Rel are distinguished \
         entirely by resident row content — no per-copula classid exists anywhere in \
         this probe, and reconstruction (G-F1) never read a classid"
            .to_string(),
    );

    // ---- G-F6 — group/mask updates never move the population ----
    let snapshot = rows.clone();
    // "Regroup" = change how we READ (a different grouping function), and
    // insert a new row. Neither may disturb existing resident rows.
    let coarse_group = |r: &RelRow| -> u8 { u8::from(r.cop_tag >= 2) }; // 2 groups instead of 4
    let regrouped: usize = rows.iter().map(|r| coarse_group(r) as usize).sum();
    rows.push(RelRow {
        subject: addr_of_term(5),
        object: addr_of_term(6),
        cop_tag: 2,
        cop_verb: 0,
        truth_f: TruthValue::new(0.8, 0.5).frequency.to_bits(),
        truth_c: TruthValue::new(0.8, 0.5).confidence.to_bits(),
        stamp: Stamp::source(40).0,
    });
    gate(
        "G-F6 regrouping and insertion leave resident rows byte-identical",
        rows[..n] == snapshot[..] && regrouped > 0,
        format!(
            "a 4-group reading and a 2-group reading coexist over the same bytes; an \
             appended row left all {n} prior rows untouched — the population does not \
             move, the view does"
        ),
    );

    // ---- G-F8 — truth/provenance are properties of the CLAIM, not the group ----
    let mut row9 = rows[9];
    let (tf, tc, st) = (row9.truth_f, row9.truth_c, row9.stamp);
    row9.cop_tag = 2; // reclassify: its group changes...
    gate(
        "G-F8 reclassifying a row's group leaves its truth and provenance untouched",
        row9.truth_f == tf
            && row9.truth_c == tc
            && row9.stamp == st
            && group_of(&row9) != group_of(&rows[9]),
        "truth and stamp ride the ROW (the claim/evidence relation); the group is a \
         classification OVER claims and owns neither"
            .to_string(),
    );

    // ---- The brutal-mask composition, measured ----
    // group ∩ HHTL region ∩ not-falsified, one pass, no materialized set.
    let mut region_40 = RowFocusMask::empty();
    region_40.insert(
        AttentionFocusFacet::prefix(FacetCascade::from_bytes(&addr_of_term(1)), 1)
            .expect("depth 1"),
    );
    let survivors = rows
        .iter()
        .filter(|r| group_of(r) == Group::InheritanceLike)
        .filter(|r| region_40.contains(focus_of_addr(&r.subject)))
        .filter(|r| f32::from_bits(r.truth_f) > 0.5)
        .count();
    gate(
        "G-COMPOSE group ∩ HHTL region ∩ truth-condition in one pass",
        survivors == 10,
        format!(
            "{survivors} rows survive InheritanceLike ∩ subtree-0x40 ∩ f>0.5 — computed \
             as chained predicates over borrowed rows; nothing materialized"
        ),
    );

    // ---- G-F10 — cost of the candidates, on the MEASURED distribution ----
    let row_bytes = core::mem::size_of::<RelRow>();
    let sparse_cost = n * row_bytes;
    // Dense alternative: per-group adjacency bitmap over terms × terms.
    let dense_cost = 4 * (t * t).div_ceil(8);
    // WideFieldMask alternative: 64-bit group-membership word PER TERM
    // (classifies terms, cannot carry pairwise relations at all — noted).
    let wfm_cost = t * 8;
    gate(
        "G-F10 representation cost compared on the measured distribution",
        sparse_cost > 0 && dense_cost > 0,
        format!(
            "sparse rows: {n}×{row_bytes}B = {sparse_cost}B; dense 4-group t×t bitmaps: \
             {dense_cost}B; per-term WideFieldMask words: {wfm_cost}B but CANNOT carry \
             pairwise topology (classification only). At {:.3}% occupancy the verdict \
             is fixture-scale, not workload-scale — the KJV Rel-heavy and tactics \
             Impl distributions remain unmeasured, and the addendum buys NOTHING on \
             these numbers alone",
            sparsity * 100.0
        ),
    );

    println!("PROBE-COPULA-GROUP-MASK-1: ALL {pass} GATES GREEN");
    println!(
        "measured: the Active-Directory shape holds on shipped operators — one resident \
         relation row (subject address, object address, RESIDENT copula content, truth, \
         provenance) with groups as lossy-by-design selection ergonomics (G-F1), \
         members/memberOf as inverse views over one relation (G-F2), no ancestry-faking \
         of many-to-many topology (G-F4), ONE classid across four copulas (G-F5), \
         view-only regrouping (G-F6), truth on the claim never the group (G-F8), and a \
         one-pass group ∩ region ∩ condition selection (G-COMPOSE). COPULA ≠ RAIL \
         PLACEMENT did not and does not imply COPULA = CLASSID: content never travels \
         in classid."
    );
}
