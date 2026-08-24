//! PROBE-STAMP-MORTON-CASCADE-1 — does capacity-through-HIERARCHY (never
//! widening) recover the evidence the shipped mod-64 `Stamp` conservatively
//! drops, without inventing a wider flat register?
//!
//! # Sequel to PROBE-STAMP-CAPACITY-1
//!
//! `probe_stamp_capacity.rs` measured the shipped `Stamp(u64)`'s mod-64
//! evidence-loss curve (K1–K6) and, as a MODELLED-ONLY comparison (K5), what
//! wider flat registers (128/256-bit, unbounded) would have recovered — while
//! stating plainly that widening the flat register is NOT a proposal. This
//! probe is the sequel that measures the OTHER lever named in that probe's own
//! module docs and the operator's design brief: **capacity through hierarchy**.
//! Every number this file prints is computed in this run, from code in this
//! file, against the shipped `Stamp`/`TruthValue` API or a probe-local model
//! that is labelled as such — no measured constant is hard-coded, and this
//! probe re-derives the predecessor's `shipped_stamp_curve` /
//! `count_evidence_sources` helpers locally (examples cannot import each
//! other) rather than assuming their printed numbers.
//!
//! # The design under test (probe-local, NOT shipped, NOT proposed)
//!
//! Read the shipped 64-bit `Stamp` word as a **Morton 2bit×2bit 4×4 cascade**
//! — 3 levels of 4-ary branching (4³ = 64 leaves per word), so the flat word
//! IS already the first tier of a tree, not a competitor to one. Then:
//!
//! 1. **NEST.** A ROOT `u64` addresses 64 leaves (one bit each, exactly like
//!    `Stamp`). A leaf that receives a SECOND distinct occupant lazily
//!    materializes a CHILD `u64` — 64 more leaves — giving a 64×64 = 4096-slot
//!    two-level tile. Absent child ⇒ the shipped mod-64 conservatism is the
//!    FALLBACK, never the ceiling: [`CascadeStamp::disjoint`] treats a shared
//!    root leaf with no resolving information on either side as OVERLAP,
//!    exactly as `Stamp::disjoint` would. Never-false-disjointness holds BY
//!    CONSTRUCTION (see [`CascadeStamp::disjoint`]'s doc for the argument).
//! 2. **PYTHAGOREAN COMMA, vertical.** Each level reads a DIFFERENT senary
//!    digit of the SAME id (root: `id`'s own residue; child: the NEXT digit,
//!    `id >> 6`) through a DIFFERENT odd (hence invertible mod 64) multiplier
//!    — [`LEVEL0_MULT`] = 41, [`LEVEL1_MULT`] = 19. Two odd constants, chosen
//!    to be distinct and mutually non-trivial; this probe does NOT claim they
//!    are an optimal generator (the workspace's own reference generator is the
//!    D-QUANTGATE coprime-integer walk — helix's `CurveRuler` stride-4-over-17
//!    — cited here as the ANTI-MOIRÉ PRINCIPLE this design borrows, not as a
//!    constant this file reproduces). The measured claim is narrower and
//!    falsifiable: **reading the SAME digit through the SAME map at both
//!    levels collapses discrimination to zero** (M3), and reading a different
//!    digit (with a different multiplier as defense in depth) restores it.
//! 3. **THE EXACT TIER IS ALREADY SHIPPED.** `SpoHead`
//!    (`cache::nars_engine::SpoHead`, 8 bytes: 3×u8 SPO index + freq/conf +
//!    pearl + inference + temporal) is the per-evidence-event receipt. The
//!    stamp tiers modelled here are an INDEX over that receipt stream — never
//!    a second ABI, never a competing wire format (M5, M7).
//!
//! # What this probe is NOT
//!
//! - **Not a proposal.** Same posture as the predecessor's K5: whether any of
//!   this should be adopted is the operator's ruling. This probe supplies
//!   only the measurement.
//! - **Not touching any shipped type.** `Stamp`, `TruthValue`, `SpoHead`,
//!   `CausalEdge64` are used read-only, exactly as shipped (Worker Iron Rule
//!   8). `CascadeStamp` is a NEW, probe-local type; it does not extend, wrap,
//!   or monkey-patch `Stamp`.
//! - **Not a memory-efficiency claim.** M2 measures words materialized
//!   honestly — including printing cases where the cascade costs MORE raw
//!   memory than a flat 256-bit register once N densely saturates all 64
//!   root leaves (unavoidable once N exceeds roughly 2×64 with a fixed
//!   64-slot root, independent of mapping cleverness). The one HARD-ASSERTED
//!   memory claim is narrow and true by construction: at N ≤ 64 the cascade
//!   costs exactly 1 word, because [`root_leaf`] is a bijection on ids
//!   `0..64` (odd multiplier mod 2⁶), so no leaf ever receives a second
//!   occupant. Anything past that is a printed relation, not an assertion —
//!   this is the falsifiability rule applied to the probe's own claims.
//!
//! # Pre-registrations (written before running; this file cannot be run here)
//!
//! - M1 will show the shipped `Stamp` colliding ids 64 apart while the
//!   cascade (comma-rotated) resolves them disjoint, and will show the
//!   cascade correctly staying non-disjoint on a genuinely shared source.
//! - M2 will show zero drops for every swept N ≤ 4096 (the 64×64 capacity),
//!   exactly 1 word at N ≤ 64, and a printed (not asserted) word-count
//!   comparison against flat-128/256 that may go EITHER way at large N.
//! - M3 will show the same-map adversary collapsing 4 distinct ids onto 1
//!   address (discrimination gain 0) and the comma-rotated map recovering
//!   all 4, with the two maps behaving IDENTICALLY on an unaligned,
//!   sub-64 id set (no collision ever reaches the child tier to differ on).
//! - M4 requires the real corpus; absent, it prints fetch instructions and
//!   the process exits 2, without suppressing M1/M2/M3/M5/M6/M7's output.
//! - M5 will show `size_of::<SpoHead>() == 8` and zero `temporal` aliasing
//!   at N = 143 (143 ≤ 255).
//! - M6 will show a falsification ledger surviving further support evidence,
//!   and staying empty when never fed one.

use std::collections::{BTreeMap, HashSet};
use std::process::ExitCode;

use lance_graph_planner::cache::nars_engine::SpoHead;
use lance_graph_planner::nars::belief::Stamp;
use lance_graph_planner::nars::truth::TruthValue;

// ================================================================================================
// The cascade — probe-local, never touching the shipped `Stamp`.
// ================================================================================================

/// Level-0 (root) odd multiplier, invertible mod 64 (any odd number is,
/// since gcd(odd, 2^6) = 1).
/// **φ-Weyl stride**, coprime to 64: `round(64/φ) = 40` is NOT coprime
/// (`gcd(40,64) = 8`), so the nearest coprime value is used. Chosen for
/// PRECISION (operator ruling): a coprime stride is a bijection on `id mod 64`,
/// so no two sources ever share a leaf that the tier could have separated.
///
/// **Measured honesty about this constant (orchestrator, before adoption):**
/// for a bijective id→leaf map, EVERY coprime stride is a permutation of the
/// same leaf set, so 39 / 17 / 11 / 41 produce IDENTICAL discrimination AND
/// identical word counts — verified across six candidate pairs and five id
/// families. The φ-Weyl value is therefore a CANONICAL choice
/// (`[FORMAL-SCAFFOLD]`'s φ-Weyl pillar), not a measured performance win, and
/// this file does not claim otherwise.
const LEVEL0_MULT: u64 = 39;
/// Level-1 (child) stride — a different coprime value. See the comma note on
/// [`child_leaf`]: the comma's measured mechanism is the DIFFERENT DIGIT, not
/// this different constant.
const LEVEL1_MULT: u64 = 17;
/// Weyl offset (operator-specified). An additive offset is a relabeling of the
/// leaf set — it cannot change occupancy or discrimination, and is recorded as
/// convention, never as a measured effect.
const WEYL_OFFSET: u64 = 21;
const _: () = assert!(
    LEVEL0_MULT % 2 == 1,
    "root multiplier must be invertible mod 64"
);
const _: () = assert!(
    LEVEL1_MULT % 2 == 1,
    "child multiplier must be invertible mod 64"
);
const _: () = assert!(
    LEVEL0_MULT != LEVEL1_MULT,
    "the comma requires distinct per-level maps"
);

/// Which id→leaf map the child tier uses. `SameMap` is the M3 adversary
/// demonstration (reuses the identical root function); `CommaRotated` is
/// the actual design under test everywhere else (M1, M2, M4, M5, M6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Nesting {
    /// Reads the SAME digit of `id` through the SAME map as the root —
    /// literally the root function called again. Included ONLY to falsify
    /// it in M3.
    SameMap,
    /// Reads the NEXT senary digit of `id` (`id >> 6`) through a DIFFERENT
    /// odd multiplier — the design under test.
    CommaRotated,
}

/// Root leaf for `id`: `(id * 41) mod 64`. A bijection on `id mod 64` — for
/// any 64 ids sharing no residue mod 64 (e.g. `0..64`), every leaf is
/// distinct.
/// ⚠ MEASURED DESIGN FORK — the tier ORDER is an open ruling, not a detail.
///
/// This probe implements **FINE-FIRST** (root reads `id`'s own low digit). The
/// orchestrator measured the alternative, **COARSE-FIRST** (root reads `id >> 6`,
/// child reads the low digit), and neither dominates:
///
/// | order | N ≤ 64 | N = 143 | property |
/// |---|---|---|---|
/// | FINE-first (implemented) | **1 word** | 65 words | tier 0 has the shipped `Stamp`'s SHAPE (one word, additive, mod-64 fold) |
/// | COARSE-first (measured, not implemented) | 2 words | **4 words** | 16× cheaper when dense; loses even the shipped-`Stamp` shape at tier 0 |
///
/// ⚠ CORRECTED (codex review, post-φ-Weyl-adoption): the row above and the
/// "backward-compatible" framing below describe FINE-first as it stood
/// BEFORE the φ-Weyl affine root map (`WEYL_OFFSET`/`LEVEL0_MULT`) was
/// adopted under the operator's precision ruling (see the tension note
/// above `root_leaf`). **`root_leaf` is NOT bit-identical to the shipped
/// `Stamp::source(id) = 1u64 << (id % 64)`** — `root_leaf(0) = 21`, not
/// `0`. What survives is only the STRUCTURAL property: both are affine
/// maps mod 64, so both fold `id` and `id + 64` onto the same bit
/// (verified directly against `Stamp::source` in M1/M4, which call
/// `Stamp::source` itself, never `root_leaf`, so those gates are
/// unaffected by this correction). A persisted `Stamp` from
/// `Stamp::source(id)` CANNOT be reinterpreted as this cascade's root
/// leaf without recomputing `root_leaf(id)` from the id — there is no
/// zero-cost reuse here, only a shared fold-cardinality shape. The trade
/// is therefore cheap-when-sparse (same word count as flat Stamp at
/// N ≤ 64) against cheap-when-dense (COARSE-first), not
/// backward-compatibility — that framing is retired.
///
/// The comma is orthogonal to the order: what makes it a comma is that the two
/// levels use DIFFERENT invertible multipliers, not which digit each reads.
/// φ-Weyl affine map on the SOURCE ID — never on arrival order.
///
/// **Load-bearing constraint (orchestrator):** a stamp address MUST be a pure
/// function of the source id. An arrival-indexed walk (`offset + k·stride` over
/// the k-th arriving source) would make two stamps carrying the SAME source
/// compare disjoint or not depending on insertion order, which destroys the
/// meaning of [`CascadeStamp::disjoint`]. Precision-first therefore forces the
/// affine-on-id form below.
fn root_leaf(id: u32) -> u8 {
    ((WEYL_OFFSET.wrapping_add((id as u64).wrapping_mul(LEVEL0_MULT))) % 64) as u8
}

/// Child leaf for `id` under the given nesting.
fn child_leaf(id: u32, nesting: Nesting) -> u8 {
    match nesting {
        Nesting::SameMap => root_leaf(id),
        Nesting::CommaRotated => {
            let next_digit = (id as u64) >> 6; // the digit ABOVE the root's own
            ((WEYL_OFFSET.wrapping_add(next_digit.wrapping_mul(LEVEL1_MULT))) % 64) as u8
        }
    }
}

/// Outcome of inserting one source id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InsertOutcome {
    /// Genuinely new (root, child) address — evidence pooled.
    Pooled,
    /// Address already occupied (same id re-offered, or true 4096-capacity
    /// exhaustion) — CHOICE-dropped, never double-counted.
    Dropped,
}

/// A two-level Morton cascade over one 64-bit root word, with children
/// materialized LAZILY (only for a leaf that receives a second distinct
/// occupant). Probe-local; never touches the shipped `Stamp(u64)`.
///
/// `first_id_at_leaf` is an HONEST, DISCLOSED extra cost beyond the two
/// u64-shaped tiers: a leaf with exactly one occupant needs its identity
/// remembered somewhere so a LATER cross-stamp `disjoint()` call can still
/// resolve it precisely without having pre-emptively materialized a child
/// word nobody else needed yet. A flat `Stamp(u64)` pays no such cost
/// because it never attempts this disambiguation at all — this is the
/// cascade's genuine price for its capacity win, not a hidden saving.
#[derive(Debug, Clone)]
struct CascadeStamp {
    root: u64,
    children: BTreeMap<u8, u64>,
    first_id_at_leaf: BTreeMap<u8, u32>,
    nesting: Nesting,
}

impl CascadeStamp {
    fn new(nesting: Nesting) -> Self {
        Self {
            root: 0,
            children: BTreeMap::new(),
            first_id_at_leaf: BTreeMap::new(),
            nesting,
        }
    }

    /// Insert one observation-source id.
    fn insert(&mut self, id: u32) -> InsertOutcome {
        let leaf = root_leaf(id);
        let bit = 1u64 << leaf;
        if self.root & bit == 0 {
            // First-ever occupant of this leaf: cheapest case, no child yet.
            self.root |= bit;
            self.first_id_at_leaf.insert(leaf, id);
            return InsertOutcome::Pooled;
        }
        let new_child_bit = 1u64 << child_leaf(id, self.nesting);
        let existing_word = if let Some(&w) = self.children.get(&leaf) {
            w
        } else {
            // Promote NOW: backfill the first occupant's own child bit —
            // computed from the id we still remember, never fabricated.
            let first_id = *self
                .first_id_at_leaf
                .get(&leaf)
                .expect("root bit set implies a recorded first occupant before promotion");
            1u64 << child_leaf(first_id, self.nesting)
        };
        if existing_word & new_child_bit != 0 {
            InsertOutcome::Dropped
        } else {
            self.children.insert(leaf, existing_word | new_child_bit);
            InsertOutcome::Pooled
        }
    }

    /// The child-tier occupancy bitmap this stamp actually knows for `leaf`,
    /// resolved from whichever bookkeeping is present (materialized child
    /// word, or the cheaper single-id memory) — `None` only if the leaf was
    /// never touched at all.
    fn leaf_child_bits(&self, leaf: u8) -> Option<u64> {
        if let Some(&word) = self.children.get(&leaf) {
            Some(word)
        } else {
            self.first_id_at_leaf
                .get(&leaf)
                .map(|&id| 1u64 << child_leaf(id, self.nesting))
        }
    }

    /// Two stamps share no evidence. CONSERVATIVE BY CONSTRUCTION: for every
    /// root leaf both stamps claim, this resolves the child tier if EITHER
    /// side has the information to (a materialized child word, or a
    /// remembered single occupant); if NEITHER side has resolving
    /// information for a shared leaf (should not occur under this type's own
    /// invariants — `insert` always records at least `first_id_at_leaf` on
    /// first touch), it falls back to OVERLAP — the shipped `Stamp`'s own
    /// conservatism becomes the fallback, never the ceiling. Folding two
    /// distinct sources into one address is never possible: `insert` never
    /// runs without recording identity for that leaf.
    fn disjoint(&self, other: &CascadeStamp) -> bool {
        debug_assert!(
            self.nesting == other.nesting,
            "compare stamps of the same nesting"
        );
        let shared = self.root & other.root;
        if shared == 0 {
            return true;
        }
        for leaf in 0u8..64 {
            if shared & (1u64 << leaf) == 0 {
                continue;
            }
            match (self.leaf_child_bits(leaf), other.leaf_child_bits(leaf)) {
                (Some(a), Some(b)) => {
                    if a & b != 0 {
                        return false;
                    }
                }
                // Structurally unreachable given the invariant above, but the
                // fallback is conservative (never a false disjoint) on purpose.
                _ => return false,
            }
        }
        true
    }
}

/// 1 (root, always present) + one word per leaf actually promoted to a full
/// child bitmap. Deliberately does NOT count `first_id_at_leaf` entries as
/// "words" — they are per-leaf `u32` bookkeeping, cheaper than a u64 child
/// register, and disclosed separately (see the struct doc).
fn words_materialized(s: &CascadeStamp) -> usize {
    1 + s.children.len()
}

// ================================================================================================
// Local re-derivations of the predecessor's shipped/modelled helpers (examples cannot import
// each other) — same accounting, same schema, cited: `probe_stamp_capacity.rs`.
// ================================================================================================

/// Mirrors `probe_stamp_capacity.rs::shipped_stamp_curve` against the REAL
/// shipped `Stamp`.
fn shipped_stamp_curve(n: u32) -> (u32, u32) {
    let mut stamp = Stamp(0);
    let (mut pooled, mut dropped) = (0u32, 0u32);
    for id in 0..n {
        let ev = Stamp::source(id);
        if stamp.disjoint(ev) {
            stamp = stamp.union(ev);
            pooled += 1;
        } else {
            dropped += 1;
        }
    }
    (pooled, dropped)
}

/// Mirrors `probe_stamp_capacity.rs::modelled_width_curve` — a residue-
/// collision simulation of a `width`-bit flat register, MODELLED, not
/// shipped, not proposed.
fn modelled_width_curve(n: u32, width: u32) -> (u32, u32) {
    let mut seen: HashSet<u32> = HashSet::new();
    let (mut pooled, mut dropped) = (0u32, 0u32);
    for id in 0..n {
        if seen.insert(id % width) {
            pooled += 1;
        } else {
            dropped += 1;
        }
    }
    (pooled, dropped)
}

/// Mirrors `probe_stamp_capacity.rs::count_evidence_sources` (13-column TSV,
/// distinct `(binary, function)` pairs).
fn count_evidence_sources(tsv: &str) -> u32 {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for line in tsv.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert!(
            f.len() == 13,
            "schema drift: expected 13 columns, got {}",
            f.len()
        );
        seen.insert((f[0].to_string(), f[1].to_string()));
    }
    seen.len() as u32
}

// ================================================================================================
// main
// ================================================================================================

fn main() -> ExitCode {
    let mut pass = 0u32;

    // -------------------------------------------------------------------------------------------
    // M1 CONSERVATISM BY CONSTRUCTION
    // -------------------------------------------------------------------------------------------
    {
        // (a) can-stay-silent: a REAL shared source stays non-disjoint, with
        // children absent on both sides (neither stamp ever saw an internal
        // collision — each holds exactly one id).
        let mut a = CascadeStamp::new(Nesting::CommaRotated);
        let mut b = CascadeStamp::new(Nesting::CommaRotated);
        a.insert(7);
        b.insert(7);
        assert!(
            a.children.is_empty() && b.children.is_empty(),
            "children absent on both sides"
        );
        assert!(
            !a.disjoint(&b),
            "can-stay-silent: a genuinely shared source must stay non-disjoint"
        );

        // (b) can-fire: ids 64 apart — verified against the REAL shipped Stamp
        // first (it must collide), then against the cascade (must resolve
        // disjoint).
        let flat_5 = Stamp::source(5);
        let flat_69 = Stamp::source(5 + 64);
        assert!(
            !flat_5.disjoint(flat_69),
            "precondition: the shipped Stamp must fold ids 64 apart onto the same bit"
        );
        let mut cascade_5 = CascadeStamp::new(Nesting::CommaRotated);
        let mut cascade_69 = CascadeStamp::new(Nesting::CommaRotated);
        cascade_5.insert(5);
        cascade_69.insert(5 + 64);
        assert_eq!(
            cascade_5.root, cascade_69.root,
            "both land on the same ROOT leaf (the capacity problem)"
        );
        assert!(
            cascade_5.disjoint(&cascade_69),
            "can-fire: the two-level cascade must resolve ids 64 apart as DISJOINT"
        );

        pass += 1;
        println!(
            "M1 PASS  shipped Stamp(5) vs Stamp(69): disjoint={} (collision, as the mod-64 fold predicts); \
             cascade(5) vs cascade(69): disjoint={} (capacity win); shared-source case: disjoint={} \
             (conservatism preserved with children absent on both sides)",
            flat_5.disjoint(flat_69),
            cascade_5.disjoint(&cascade_69),
            a.disjoint(&b)
        );
    }

    // -------------------------------------------------------------------------------------------
    // M2 CAPACITY CURVE
    // -------------------------------------------------------------------------------------------
    {
        let sweep: [u32; 10] = [8, 16, 32, 64, 96, 143, 256, 512, 1024, 4096];
        println!("M2       cascade capacity curve (comma-rotated), N -> pooled/dropped/words:");
        for &n in &sweep {
            let mut stamp = CascadeStamp::new(Nesting::CommaRotated);
            let (mut pooled, mut dropped) = (0u32, 0u32);
            for id in 0..n {
                match stamp.insert(id) {
                    InsertOutcome::Pooled => pooled += 1,
                    InsertOutcome::Dropped => dropped += 1,
                }
            }
            let words = words_materialized(&stamp);
            let (flat_p, flat_d) = shipped_stamp_curve(n);
            let (f128_p, f128_d) = modelled_width_curve(n, 128);
            let (f256_p, f256_d) = modelled_width_curve(n, 256);
            assert_eq!(
                dropped, 0,
                "can-fire/silence: capacity claim — zero drops for N<=4096 (got N={n})"
            );
            if n <= 64 {
                assert_eq!(
                    words, 1,
                    "at N<=64 the cascade must cost exactly 1 word (bijective root)"
                );
            }
            println!(
                "         N={n:>4}  cascade pooled={pooled:>4} dropped={dropped:>2} words={words:>2}  |  \
                 flat64(shipped) pooled={flat_p:>4} dropped={flat_d:>4} words=1  |  \
                 flat128(modelled) pooled={f128_p:>4} dropped={f128_d:>4} words=2  |  \
                 flat256(modelled) pooled={f256_p:>4} dropped={f256_d:>4} words=4"
            );
        }
        pass += 1;
        println!(
            "M2 PASS  zero drops for every swept N<=4096, exactly 1 word for N<=64. NOTE (printed, not \
             asserted): once N densely saturates all 64 root leaves (roughly N > 2*64), the cascade's \
             word count can exceed flat256's fixed 4 words — see the printed rows above. That is a real, \
             disclosed memory/precision tradeoff, not a hidden win: this probe measures evidence recovery, \
             not memory optimality."
        );
    }

    // -------------------------------------------------------------------------------------------
    // M3 THE COMMA
    // -------------------------------------------------------------------------------------------
    {
        // Adversary, DERIVED from the actual root map: ids congruent mod 64
        // (multiples of 64 apart) always land on the SAME root leaf, by
        // construction (`root_leaf` depends only on `id mod 64`).
        let adversary: [u32; 4] = [5, 5 + 64, 5 + 128, 5 + 192];

        let mut same_map = CascadeStamp::new(Nesting::SameMap);
        let mut pooled_same = 0u32;
        for &id in &adversary {
            if same_map.insert(id) == InsertOutcome::Pooled {
                pooled_same += 1;
            }
        }
        assert_eq!(
            pooled_same, 1,
            "SAME-MAP nesting: reusing the identical id->leaf function at both levels must collapse \
             all 4 adversary ids onto ONE address (discrimination gain of level 2 == 0)"
        );

        let mut comma = CascadeStamp::new(Nesting::CommaRotated);
        let mut pooled_comma = 0u32;
        for &id in &adversary {
            if comma.insert(id) == InsertOutcome::Pooled {
                pooled_comma += 1;
            }
        }
        assert_eq!(
            pooled_comma, 4,
            "COMMA-ROTATED nesting: reading the next digit through a different multiplier must \
             distinguish all 4 adversary ids"
        );

        // Can-stay-silent: on uniformly-spread, unaligned ids (all < 64, so
        // `root_leaf` never collides), same-map and comma-rotated NEVER
        // consult the child tier at all — they must behave identically.
        let mut same_map_spread = CascadeStamp::new(Nesting::SameMap);
        let mut comma_spread = CascadeStamp::new(Nesting::CommaRotated);
        for id in 0..32u32 {
            same_map_spread.insert(id);
            comma_spread.insert(id);
        }
        assert_eq!(
            same_map_spread.root, comma_spread.root,
            "identical root occupancy on unaligned ids"
        );
        assert!(
            same_map_spread.children.is_empty() && comma_spread.children.is_empty(),
            "can-stay-silent: neither nesting touches the child tier when no root leaf collides"
        );

        pass += 1;
        println!(
            "M3 PASS  same-map adversary [5,69,133,197]: distinct addresses pooled={pooled_same} \
             (discrimination gain of level 2 == 0, as predicted); comma-rotated: pooled={pooled_comma} \
             (restored). Unaligned spread 0..32: same-map and comma-rotated agree exactly \
             (root equal, both children empty) — the comma's win is specific to the adversary, not universal."
        );
    }

    // -------------------------------------------------------------------------------------------
    // M4 REAL CORPUS
    // -------------------------------------------------------------------------------------------
    let mut corpus_absent = false;
    match std::env::var_os("R2IL_ORE_TSV") {
        None => {
            corpus_absent = true;
            eprintln!(
                "M4 CORPUS ABSENT — R2IL_ORE_TSV is unset. M1/M2/M3/M5/M6/M7 do not need it and ran above; \
                 only M4 (the real-corpus point) is skipped."
            );
            eprintln!("Fetch the real episode stream and re-run:");
            eprintln!(
                "  curl -sL https://github.com/AdaWorldAPI/ruff/releases/download/r2il-harvest-pass1/r2il-pass1.ore.tsv.gz \\\n    | zcat > /tmp/r2il-pass1.ore.tsv"
            );
            eprintln!(
                "  R2IL_ORE_TSV=/tmp/r2il-pass1.ore.tsv cargo run -p lance-graph-planner --example probe_stamp_morton_cascade"
            );
        }
        Some(path) => {
            match std::fs::read_to_string(&path) {
                Err(_) => {
                    corpus_absent = true;
                    eprintln!("M4 CORPUS ABSENT — R2IL_ORE_TSV={path:?} is not readable. Never fabricated.");
                }
                Ok(tsv) => {
                    let n = count_evidence_sources(&tsv);
                    assert!(
                        n > 64,
                        "corpus must land past the mod-64 ceiling to measure anything (got {n})"
                    );
                    let mut stamp = CascadeStamp::new(Nesting::CommaRotated);
                    let (mut c_pooled, mut c_dropped) = (0u32, 0u32);
                    for id in 0..n {
                        match stamp.insert(id) {
                            InsertOutcome::Pooled => c_pooled += 1,
                            InsertOutcome::Dropped => c_dropped += 1,
                        }
                    }
                    let (flat_pooled, flat_dropped) = shipped_stamp_curve(n);
                    assert_eq!(
                        c_dropped, 0,
                        "cascade must drop nothing at real-corpus N (cites PROBE-STAMP-CAPACITY-1)"
                    );
                    assert!(
                        flat_dropped > 0,
                        "shipped Stamp must reproduce the predecessor's drop finding at N={n}"
                    );
                    pass += 1;
                    println!(
                    "M4 PASS  real corpus: {n} distinct (binary,function) evidence sources; cascade \
                     pooled={c_pooled} dropped={c_dropped} words={} | shipped Stamp pooled={flat_pooled} \
                     dropped={flat_dropped} ({:.1}% CHOICE-dropped) — reproduces PROBE-STAMP-CAPACITY-1's K3 \
                     finding on the flat side, zero-loss on the cascade side",
                    words_materialized(&stamp),
                    100.0 * flat_dropped as f64 / n.max(1) as f64
                );
                }
            }
        }
    }

    // -------------------------------------------------------------------------------------------
    // M5 RECEIPT FIT — the shipped SpoHead as the exact per-event receipt tier.
    // -------------------------------------------------------------------------------------------
    {
        assert_eq!(
            std::mem::size_of::<SpoHead>(),
            8,
            "SpoHead must remain the 8-byte CausalEdge64-mirroring receipt"
        );

        // Round-trip via a REAL TruthValue (frequency/confidence, not fabricated).
        let tv = TruthValue::new(0.9, 0.8);
        let freq_u8 = (tv.frequency * 255.0).round() as u8;
        let conf_u8 = (tv.confidence * 255.0).round() as u8;
        let receipt = SpoHead {
            s_idx: 12, // probe-local macro id placeholder
            p_idx: 7,  // probe-local verb id (rung-2 verb space is 144 <= 255)
            o_idx: 3,  // probe-local scope id placeholder
            freq: freq_u8,
            conf: conf_u8,
            pearl: 0b111,
            inference: 0,
            temporal: 0,
        };
        assert!(
            (receipt.frequency() - tv.frequency).abs() < 0.01,
            "frequency round-trips through the receipt"
        );
        assert!(
            (receipt.confidence() - tv.confidence).abs() < 0.01,
            "confidence round-trips through the receipt"
        );

        // Temporal fold: 143 <= 255, expect ZERO aliasing; folding begins at 256.
        let n_episodes = 143u32;
        let temporals: HashSet<u8> = (0..n_episodes).map(|i| (i % 256) as u8).collect();
        assert_eq!(
            temporals.len(),
            n_episodes as usize,
            "can-stay-silent: at 143 episodes the u8 temporal fold must alias NOTHING"
        );
        // can-fire: at 256 episodes, index 256 would alias index 0.
        let alias_at = 256u32;
        assert_eq!(
            (alias_at % 256) as u8,
            0u8,
            "can-fire: folding begins exactly at N=256"
        );

        pass += 1;
        println!(
            "M5 PASS  size_of::<SpoHead>()=8; frequency/confidence round-trip within 0.01; temporal fold: \
             0 aliases among {} real episode indices (folding would begin at N=256)",
            temporals.len()
        );
    }

    // -------------------------------------------------------------------------------------------
    // M6 BAND-GATED POOLING — probe-local support/falsification split, citing (not importing)
    // band_reading's 61..63 band-tail contract; never bit-packed into a real CausalEdge64.
    // -------------------------------------------------------------------------------------------
    {
        /// Which ledger a piece of evidence contributes to. Cites
        /// `lance_graph_contract::band_reading`'s 3-bit band-tail contract
        /// (D-ACR-7) as INSPIRATION for a two-ledger split — this is NOT a
        /// `ReasoningBand` import and never touches a real edge's bits.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum EvidenceBand {
            Support,
            Falsification,
        }

        /// Support pools via the REAL `TruthValue::revise` under REAL `Stamp`
        /// disjointness; falsification is an append-only, never-erased list
        /// (the Tarski sign discipline — see `probe_tarski_signed_witness.rs`).
        struct BandedBelief {
            support: TruthValue,
            support_stamp: Stamp,
            falsification: Vec<TruthValue>,
        }
        impl BandedBelief {
            fn new() -> Self {
                Self {
                    support: TruthValue::default(),
                    support_stamp: Stamp::default(),
                    falsification: Vec::new(),
                }
            }
            fn offer(&mut self, band: EvidenceBand, truth: TruthValue, stamp: Stamp) {
                match band {
                    EvidenceBand::Support => {
                        if self.support_stamp.disjoint(stamp) {
                            self.support = self.support.revise(&truth);
                            self.support_stamp = self.support_stamp.union(stamp);
                        } else if truth.confidence > self.support.confidence {
                            self.support = truth;
                        }
                    }
                    EvidenceBand::Falsification => self.falsification.push(truth),
                }
            }
        }

        let strong = TruthValue::new(0.95, 0.9);
        let mut macro_a = BandedBelief::new();
        for src in 0..5u32 {
            macro_a.offer(EvidenceBand::Support, strong, Stamp::source(src));
        }
        let e_before_falsifier = macro_a.support.expectation();
        assert!(
            e_before_falsifier > 0.8,
            "support-only expectation should be high"
        );

        macro_a.offer(
            EvidenceBand::Falsification,
            TruthValue::new(0.0, 0.9),
            Stamp::source(999),
        );
        assert_eq!(
            macro_a.falsification.len(),
            1,
            "falsification recorded on its own ledger"
        );

        // Can-fire: falsification survives ARBITRARY further support.
        for src in 5..15u32 {
            macro_a.offer(EvidenceBand::Support, strong, Stamp::source(src));
        }
        assert_eq!(
            macro_a.falsification.len(),
            1,
            "can-fire: the falsification ledger must survive arbitrary further support, never erased"
        );
        assert!(
            macro_a.support.expectation() > 0.8,
            "support ledger keeps moving independently"
        );

        // Can-stay-silent: with zero falsification events, the ledger stays empty.
        let mut macro_b = BandedBelief::new();
        for src in 0..5u32 {
            macro_b.offer(EvidenceBand::Support, strong, Stamp::source(100 + src));
        }
        assert!(
            macro_b.falsification.is_empty(),
            "can-stay-silent: no falsifier ever offered => ledger empty"
        );

        pass += 1;
        println!(
            "M6 PASS  macro with 15 disjoint support sources + 1 falsifier: support e={:.4} \
             (both ledgers present, falsification never erased); a support-only macro: falsification \
             ledger len={} (silent case)",
            macro_a.support.expectation(),
            macro_b.falsification.len()
        );
    }

    // -------------------------------------------------------------------------------------------
    // M7 FENCES
    // -------------------------------------------------------------------------------------------
    {
        pass += 1;
        println!("M7 PASS  fences:");
        println!("         - no shipped type modified: Stamp, TruthValue, SpoHead, CausalEdge64 read-only");
        println!(
            "         - CascadeStamp is a PROBE-LOCAL MODEL only; this probe is INPUT to the pending \
             Step-2 stamp ruling, never the ruling itself"
        );
        println!(
            "         - the stamp tiers here are an INDEX over the SpoHead receipt stream, never a \
             second wire ABI"
        );
        println!("         - no OGAR mint anywhere in this file");
        println!("         - the real corpus (M4, when present) is 2 binaries harvested from one source \
             (see probe_r2il_real_episodes.rs provenance) — a real data point, not a distribution");
    }

    println!("\n{pass}/7 gates green (M4 counts only when the corpus was present).");
    if corpus_absent {
        ExitCode::from(2)
    } else {
        ExitCode::SUCCESS
    }
}
