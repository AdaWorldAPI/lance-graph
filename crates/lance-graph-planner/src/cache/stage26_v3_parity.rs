//! **Stage 2.6a — representation invariance on the planner's REAL `CausalEdge64`
//! reasoning path.** Measurement only; compiled out of every non-test build.
//!
//! # Why this seam and not the recipe surface
//!
//! Stage 2.6's original brief assumed `CE64 → recipe/runbook/planner`. **That
//! arrow does not exist.** `lance-graph-contract` — which owns the entire
//! Stage-2 surface (`recipe_kernels`, `recipes`, `recipe_dispatch`,
//! `materialize`) — has **no `causal-edge` dependency** and cannot name the
//! type; `style_strategy` consumes no causal edge; and the only non-test
//! producer of a `ThoughtCtx` reads exactly two `PlanContext` scalars. A V3
//! entrance in front of that surface would rehydrate an edge nothing downstream
//! reads, and the comparison would return `discordance = 0` for an entirely
//! trivial reason — a green result that means nothing. (Recorded as Stage 2.6b:
//! the recipe surface is *causally blind*, which is a Stage-3 substrate-wiring
//! question, not a V3 migration defect. It is NOT patched here.)
//!
//! Where `CausalEdge64` actually reasons **inside the planner** is
//! [`super::nars_engine`]: `SpoHead ↔ CausalEdge64` via
//! [`NarsEngine::to_causal_edge`] / [`NarsEngine::from_causal_edge`], and
//! [`NarsEngine::forward_edge`] over the compose tables. That leg is the one the
//! existing compare-thinking work does not cover — `causal-edge`'s own
//! `syllogize` and `cognitive-shader-driver`'s emission path are already proven
//! (`edge_v3_compare`), the planner's is not. This module closes it.
//!
//! # The comparison
//!
//! ```text
//! direct:  SpoHead --to_causal_edge--> CE64 --forward_edge--> CE64 --from_causal_edge--> SpoHead
//! V3:      SpoHead --to_causal_edge--> CE64 --from_v1--> V3 --rehydrate(resolved SPO)--> CE64
//!                                                            --forward_edge--> ... (the SAME path)
//! ```
//!
//! **There is exactly ONE reasoning implementation.** The V3 arm does not
//! compute anything: it drops the in-edge SPO, resolves it back from the target
//! node's facet, rebuilds a `CausalEdge64`, and hands it to the *same*
//! `NarsEngine` methods. No V3-native NARS exists or is introduced.
//!
//! # What is compared, and what is deliberately NOT
//!
//! Compared, by **exact equality** — the primary contract, never a correlation:
//! the rehydrated `CausalEdge64` itself; the `SpoHead` it converts back to
//! (SPO after resolution, NARS frequency/confidence, causal mask, inference
//! class); the `forward_edge` conclusion; the `SpoHead` of that conclusion; and
//! the `syllogize` conclusion identity (the quantity `edge_v3_compare` already
//! uses, kept so the two harnesses speak the same language).
//!
//! **Not** compared, because they are representation-specific by design and
//! asserting on them would be asserting that V3 is CE64: the V3 Lokal `target`,
//! the V3 `TE` byte, the payload width.
//!
//! # The resolver invariant is the load-bearing one
//!
//! Equivalence is CONDITIONAL: it holds exactly while the resolved node facet
//! agrees with the SPO the original edge carried. So this module ships both
//! halves — a positive pass over correct resolution, and a
//! [`v3_parity_detects_a_corrupted_resolution`] falsifier that deliberately
//! resolves the wrong SPO and requires the comparator to go red. A comparator
//! that stays green after the resolution is corrupted proves nothing.

use std::collections::BTreeMap;

use causal_edge::{CausalEdge64, CausalEdgeV3};

use super::nars_engine::{NarsEngine, SpoDistances, SpoHead};

/// The node-facet SPO resolver: `Lokal target → (s_idx, p_idx, o_idx)`.
///
/// Stands in for "read the SPO out of the target node's 6×256² CAM-PQ facet".
/// The V3 contract is that the edge does not duplicate what the node already
/// holds, so the harness must hold that mapping somewhere; a map keyed by the
/// same `u16` the edge carries is the smallest honest model of it.
#[derive(Default)]
struct FacetResolver {
    spo: BTreeMap<u16, (u8, u8, u8)>,
}

impl FacetResolver {
    fn bind(&mut self, target: u16, spo: (u8, u8, u8)) {
        self.spo.insert(target, spo);
    }
    fn resolve(&self, target: u16) -> Option<(u8, u8, u8)> {
        self.spo.get(&target).copied()
    }
}

/// One paired observation at the seam.
struct Leg {
    /// Label for the discordance artifact.
    case: String,
    /// Direct arm.
    direct_edge: CausalEdge64,
    direct_fwd: CausalEdge64,
    direct_head: SpoHead,
    direct_fwd_head: SpoHead,
    direct_syllogism: Option<CausalEdge64>,
    /// V3 arm — same engine, same methods, edge routed through V3.
    v3_edge: CausalEdge64,
    v3_fwd: CausalEdge64,
    v3_head: SpoHead,
    v3_fwd_head: SpoHead,
    v3_syllogism: Option<CausalEdge64>,
}

impl Leg {
    /// Every compared invariant, as `(name, agrees)`. Exact equality throughout.
    fn invariants(&self) -> Vec<(&'static str, bool)> {
        vec![
            ("rehydrated_ce64", self.v3_edge == self.direct_edge),
            (
                "spo_after_resolution",
                spo_of(self.v3_edge) == spo_of(self.direct_edge),
            ),
            (
                "nars_frequency",
                self.v3_edge.frequency_u8() == self.direct_edge.frequency_u8(),
            ),
            (
                "nars_confidence",
                self.v3_edge.confidence_u8() == self.direct_edge.confidence_u8(),
            ),
            (
                "causal_mask",
                self.v3_edge.causal_mask() as u8 == self.direct_edge.causal_mask() as u8,
            ),
            (
                "inference_class",
                self.v3_edge.inference() as u8 == self.direct_edge.inference() as u8,
            ),
            ("spo_head", self.v3_head == self.direct_head),
            ("forward_edge_result", self.v3_fwd == self.direct_fwd),
            ("forward_spo_head", self.v3_fwd_head == self.direct_fwd_head),
            (
                "syllogism_conclusion",
                self.v3_syllogism == self.direct_syllogism,
            ),
            (
                "truth_frequency",
                self.v3_head.frequency() == self.direct_head.frequency(),
            ),
            (
                "truth_confidence",
                self.v3_head.confidence() == self.direct_head.confidence(),
            ),
            (
                "expectation",
                self.v3_head.expectation() == self.direct_head.expectation(),
            ),
        ]
    }

    fn discordant(&self) -> Vec<&'static str> {
        self.invariants()
            .into_iter()
            .filter(|(_, ok)| !ok)
            .map(|(n, _)| n)
            .collect()
    }
}

fn spo_of(e: CausalEdge64) -> (u8, u8, u8) {
    (e.s_idx(), e.p_idx(), e.o_idx())
}

/// The `SpoHead` sweep the comparison runs over.
///
/// **The PATH is the real one** — every head below is driven through
/// [`NarsEngine::to_causal_edge`], [`NarsEngine::forward_edge`] and
/// [`NarsEngine::from_causal_edge`], never around them. The heads themselves
/// are a deliberate sweep of the field domains the conversion actually
/// branches on, stated plainly rather than dressed up as harvested data:
///
/// - **every `inference` discriminant `to_causal_edge` maps**, including the
///   two Pearl rungs it translates (local `7`→`Intervention`, local
///   `8`→`Counterfactual`) and the `5 | 6` arm that folds to `Synthesis` — the
///   fold is lossy on the way out, so it is exactly where a representation
///   round-trip could diverge and must be swept;
/// - **every 3-bit `pearl` mask** `0b000..=0b111`;
/// - **frequency/confidence at both rails and the midpoint**, since
///   `SpoHead::frequency` divides by 255 and a rail is where a rounding
///   difference would surface;
/// - **SPO indices spanning the palette**, including `0` and `255`.
fn head_sweep() -> Vec<SpoHead> {
    let mut heads = Vec::new();
    let inferences = [0u8, 1, 2, 3, 4, 5, 6, 7, 8];
    let truths = [(0u8, 0u8), (128, 128), (255, 255), (255, 0), (0, 255)];
    for (i, &inference) in inferences.iter().enumerate() {
        for pearl in 0u8..8 {
            let (freq, conf) = truths[(i + pearl as usize) % truths.len()];
            heads.push(SpoHead {
                s_idx: (i as u8).wrapping_mul(29),
                p_idx: pearl.wrapping_mul(31),
                o_idx: (i as u8).wrapping_mul(7).wrapping_add(pearl),
                freq,
                conf,
                pearl,
                inference,
                // `to_causal_edge` passes this into `pack`'s temporal argument,
                // where the v2 layout makes the write a no-op. Swept non-zero on
                // purpose: if that ever stops being a no-op, the V3 arm (whose
                // `rehydrate` packs temporal 0) diverges and this harness says so
                // instead of the change landing silently.
                temporal: (i as u8).wrapping_add(pearl),
            });
        }
    }
    // Both palette rails, at the extremes of the truth range.
    heads.push(SpoHead {
        s_idx: 0,
        p_idx: 0,
        o_idx: 0,
        freq: 0,
        conf: 0,
        pearl: 0,
        inference: 0,
        temporal: 0,
    });
    heads.push(SpoHead {
        s_idx: 255,
        p_idx: 255,
        o_idx: 255,
        freq: 255,
        conf: 255,
        pearl: 0b111,
        inference: 3,
        temporal: 255,
    });
    heads
}

/// The three 256×256 compose tables [`NarsEngine::forward_edge`] takes.
type ComposeTables = (
    Box<[u8; 256 * 256]>,
    Box<[u8; 256 * 256]>,
    Box<[u8; 256 * 256]>,
);

/// One `(input, weight)` pair to run through the seam, with the Lokal targets
/// whose node facets hold their SPO.
struct LegSpec<'a> {
    input: &'a SpoHead,
    weight: &'a SpoHead,
    target_in: u16,
    target_w: u16,
    case: String,
}

/// Compose tables for [`NarsEngine::forward_edge`].
///
/// Deterministic and NON-identity: with an identity table `forward` leaves the
/// SPO untouched, so the composition the tables drive would never actually run
/// under the `forward_edge_result` invariant. Pinned by the `spo_moved` half of
/// [`the_sweep_is_not_degenerate`] — and note the WEAKER "the edge changed at
/// all" form does not catch it, because `forward` also composes the NARS truth.
fn compose_tables() -> ComposeTables {
    let build = |salt: u32| {
        let mut t = Box::new([0u8; 256 * 256]);
        for a in 0..256usize {
            for b in 0..256usize {
                t[a * 256 + b] = ((a * 7 + b * 13 + salt as usize) % 256) as u8;
            }
        }
        t
    };
    (build(0), build(97), build(211))
}

/// Run the seam for one `(input, weight)` pair under a given resolver.
fn run_leg(
    engine: &NarsEngine,
    resolver: &FacetResolver,
    spec: LegSpec<'_>,
    tables: &ComposeTables,
) -> Leg {
    let LegSpec {
        input,
        weight,
        target_in,
        target_w,
        case,
    } = spec;
    let (cs, cp, co) = (&*tables.0, &*tables.1, &*tables.2);

    // ── direct arm: the shipping planner path ──
    let direct_edge = engine.to_causal_edge(input);
    let direct_w = engine.to_causal_edge(weight);
    let direct_fwd = engine.forward_edge(direct_edge, direct_w, cs, cp, co);

    // ── V3 arm: drop SPO, resolve it back, rehydrate, SAME path ──
    let v3_in = CausalEdgeV3::from_v1(direct_edge, target_in);
    let v3_wt = CausalEdgeV3::from_v1(direct_w, target_w);
    let (si, pi, oi) = resolver.resolve(target_in).expect("bound target");
    let (sw, pw, ow) = resolver.resolve(target_w).expect("bound target");
    let v3_edge = v3_in.rehydrate(si, pi, oi);
    let v3_w = v3_wt.rehydrate(sw, pw, ow);
    let v3_fwd = engine.forward_edge(v3_edge, v3_w, cs, cp, co);

    let syl = |a: CausalEdge64, b: CausalEdge64| a.syllogize(b).map(|s| s.conclusion);

    Leg {
        case,
        direct_head: engine.from_causal_edge(direct_edge),
        direct_fwd_head: engine.from_causal_edge(direct_fwd),
        direct_syllogism: syl(direct_edge, direct_w),
        v3_head: engine.from_causal_edge(v3_edge),
        v3_fwd_head: engine.from_causal_edge(v3_fwd),
        v3_syllogism: syl(v3_edge, v3_w),
        direct_edge,
        direct_fwd,
        v3_edge,
        v3_fwd,
    }
}

/// The full census: every head against its successor as the weight.
fn census(resolver_corruption: Option<u16>) -> Vec<Leg> {
    let engine = NarsEngine::new(SpoDistances::new_zero());
    let tables = compose_tables();
    let heads = head_sweep();

    // Bind each head's SPO to its own Lokal target — the node facet "holds" it.
    let mut resolver = FacetResolver::default();
    for (i, h) in heads.iter().enumerate() {
        resolver.bind(0x1000 + i as u16, (h.s_idx, h.p_idx, h.o_idx));
    }
    // The falsifier: one facet now disagrees with the edge it came from.
    if let Some(t) = resolver_corruption {
        let (s, p, o) = resolver.resolve(t).expect("corrupt an existing binding");
        resolver.bind(t, (s.wrapping_add(1), p, o));
    }

    let mut legs = Vec::with_capacity(heads.len());
    for i in 0..heads.len() {
        let j = (i + 1) % heads.len();
        legs.push(run_leg(
            &engine,
            &resolver,
            LegSpec {
                input: &heads[i],
                weight: &heads[j],
                target_in: 0x1000 + i as u16,
                target_w: 0x1000 + j as u16,
                case: format!("inf={} pearl={} i={i}", heads[i].inference, heads[i].pearl),
            },
            &tables,
        ));
    }
    legs
}

#[cfg(test)]
mod tests {
    use super::*;
    use jc::stats::binary_association;

    /// **The primary contract: exact representation invariance, zero
    /// discordance, on the planner's real CE64 path.**
    #[test]
    fn v3_representation_is_invariant_on_the_planner_ce64_path() {
        let legs = census(None);
        assert!(!legs.is_empty(), "the sweep ran nothing");

        let mut discordant = 0usize;
        for leg in &legs {
            let bad = leg.discordant();
            assert!(bad.is_empty(), "{}: V3 diverged on {:?}", leg.case, bad);
            discordant += usize::from(!bad.is_empty());
        }
        assert_eq!(discordant, 0, "planner V3 representation discordance");
    }

    /// **The falsifier.** Corrupt ONE resolved facet SPO and the comparator must
    /// go red. A comparator that stays green after the resolution is corrupted
    /// proves nothing — the equivalence is conditional on
    /// `resolved node facet SPO == the original edge's SPO`, and this is what
    /// makes that condition observable rather than assumed.
    #[test]
    fn v3_parity_detects_a_corrupted_resolution() {
        let legs = census(Some(0x1000));
        let discordant: Vec<&Leg> = legs.iter().filter(|l| !l.discordant().is_empty()).collect();
        assert!(
            !discordant.is_empty(),
            "corrupting a resolved SPO changed nothing — the comparator is vacuous"
        );

        // ...and it must be caught by the SPO-shaped invariants specifically,
        // not merely by some incidental byte: a corrupted resolution is a
        // SEMANTIC divergence, and naming which invariant sees it is the
        // difference between a detector and an alarm.
        let names: Vec<&'static str> = discordant.iter().flat_map(|l| l.discordant()).collect();
        assert!(
            names.contains(&"spo_after_resolution") && names.contains(&"rehydrated_ce64"),
            "expected the SPO invariants to fire, got {names:?}"
        );

        // And it must NOT fire everywhere — a comparator that reports every leg
        // discordant after a single-facet corruption is not localising anything.
        assert!(
            discordant.len() < legs.len(),
            "a one-facet corruption made every leg discordant ({} of {}) — the \
             comparator is not localising",
            discordant.len(),
            legs.len()
        );
    }

    /// The sweep genuinely exercises the branches it claims to.
    ///
    /// Anti-vacuity for the invariants above: if every leg produced the same
    /// edge, or no leg ever syllogized, "they all agree" would be a statement
    /// about a constant rather than about the representation.
    #[test]
    fn the_sweep_is_not_degenerate() {
        let legs = census(None);
        let distinct_edges: std::collections::BTreeSet<u64> =
            legs.iter().map(|l| l.direct_edge.0).collect();
        assert!(
            distinct_edges.len() > legs.len() / 2,
            "the sweep collapsed to {} distinct edges over {} legs",
            distinct_edges.len(),
            legs.len()
        );
        let syllogized = legs.iter().filter(|l| l.direct_syllogism.is_some()).count();
        assert!(
            syllogized > 0 && syllogized < legs.len(),
            "syllogism is constant across the sweep ({syllogized}/{}) — the \
             conclusion-identity invariant would be vacuous",
            legs.len()
        );
        // The forward composition must actually move the edge — and
        // specifically its SPO, which is what the compose tables drive.
        //
        // The weaker form of this ("the edge changed at all") was written first
        // and is NOT a discriminating check: `forward` also composes the NARS
        // truth, so it moves the edge even when every compose table is the
        // identity. Measured — the disable-run that made the tables identity
        // left the weak assertion green. The SPO form is the one that fails
        // there, which is the property the non-identity tables exist to give.
        let moved = legs
            .iter()
            .filter(|l| l.direct_fwd != l.direct_edge)
            .count();
        assert!(
            moved > 0,
            "forward_edge never changed an edge — the compose tables are inert"
        );
        let spo_moved = legs
            .iter()
            .filter(|l| spo_of(l.direct_fwd) != spo_of(l.direct_edge))
            .count();
        assert!(
            spo_moved > 0,
            "forward_edge never changed an SPO across {} legs — the compose \
             tables are effectively the identity and the composition is not \
             being exercised",
            legs.len()
        );
    }

    /// JC summarises; it does not adjudicate.
    ///
    /// Every quantity at this seam is exact — `u8` palette indices, `u8` truth
    /// bytes, a `u64` register — so there is **no naturally continuous quantity
    /// here for a correlation to characterise**, and none is manufactured. What
    /// `jc` legitimately contributes is the cross-tab: syllogism PRESENCE is a
    /// real binary pair, both categories occur across the sweep, so κ is
    /// **defined** and its value is a genuine statement rather than the
    /// degenerate constant-column case `jc` refuses to score.
    #[test]
    fn jc_summary_of_the_agreement_is_perfect_and_defined() {
        let legs = census(None);
        let direct: Vec<bool> = legs.iter().map(|l| l.direct_syllogism.is_some()).collect();
        let v3: Vec<bool> = legs.iter().map(|l| l.v3_syllogism.is_some()).collect();
        let t = binary_association(&direct, &v3).expect("equal-length, non-empty");
        assert_eq!((t.n01, t.n10), (0, 0), "off-diagonal must be empty");
        assert!(
            t.n11 > 0 && t.n00 > 0,
            "both categories must occur or the agreement is vacuous (n11={}, n00={})",
            t.n11,
            t.n00
        );
        let k = t.kappa.expect("kappa defined when both categories occur");
        assert!((k - 1.0).abs() < 1e-12, "kappa = {k}");
    }

    /// Write the compact discordance-only artifact. Header-only IS the result.
    #[test]
    #[ignore = "artifact generator; run explicitly"]
    fn stage26_write_discordance_artifact() {
        let legs = census(None);
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/probes");
        std::fs::create_dir_all(dir).expect("create docs/probes");
        let mut csv = String::from("case,invariant,direct,v3\n");
        for leg in &legs {
            for name in leg.discordant() {
                csv.push_str(&format!(
                    "{},{name},{:#018x},{:#018x}\n",
                    leg.case, leg.direct_edge.0, leg.v3_edge.0
                ));
            }
        }
        std::fs::write(
            format!("{dir}/stage26-v3-planner-parity-discordance.csv"),
            csv,
        )
        .expect("write discordance csv");
    }
}
