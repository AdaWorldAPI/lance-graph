//! S3.0 — the **exact causal literal**: an absolute, deterministic address for
//! one causal proposition, independent of every quantity that can revise.
//!
//! ```text
//! CausalLiteral  (domain, S, P, O)   ← exact proposition identity, THIS module
//!         │
//!         ├── world/causal evidence leaves        (S3.1, orthogonal facet —
//!         ├── epistemic/reasoning leaves            NOT nested path depth;
//!         ├── contradiction / mediator / pothole    see "Evidence placement"
//!         │    leaves                               below)
//!         ▼
//!    CausalEdgeV3   local hot proxy      (S3.2)
//!         ▼
//!    CausalEdge64   NARS register
//! ```
//!
//! # The one thing this module asserts
//!
//! **Identity is the component tuple. Nothing else.** For a fixed canonical
//! `domain + S + P + O` the literal is byte-identical across replay, across
//! sources, across Lance versions, and across every amount of evidence that
//! ever accumulates beneath it. Two papers and a model asserting the same
//! canonical proposition converge on ONE literal with THREE witnesses — never
//! three literals.
//!
//! What identity is NOT, stated because each has been reached for before:
//!
//! | not identity | why | where it belongs |
//! |---|---|---|
//! | CAM-PQ nearest centroid | learned, approximate, re-trainable | candidate discovery, basin search, ranking |
//! | NARS `f` / `c` | revisable evidence state | the Meta accumulator |
//! | evidence count | grows monotonically; identity must not | the leaf set |
//! | the asserting source | many sources, one proposition | witness leaves |
//! | the Lance version | history is immutable, identity is timeless | the horizon |
//! | a V3 `target` u16 | tenant-LOCAL, may be repacked | [`crate::hhtl`] resolution |
//!
//! # Why this is an address and not a packed tenant
//!
//! Operator ruling E (`docs/architecture/ARC-B-OWNERSHIP-AND-ADDRESSING-REASSESSMENT.md`
//! §4) gates every new `ValueTenant` behind one question: *is this genuinely
//! missing canonical information, or a container minted to avoid completing the
//! address transition?* This is the former, and the ruling names it as such —
//! it says the missing canonical reference "is the prerequisite", and that the
//! tenant gap and the addressing gap "are the same problem wearing two hats".
//! So `CausalLiteral` adds no bits to `CausalEdge64` and no slot to any tenant.
//! It is 8 bytes of pure address, const-asserted below so that a future edit
//! cannot quietly hang evidence off it.
//!
//! # `NiblePath`'s depth budget — a fact about ONE consumer type, not about HHTL
//!
//! **Read this section before citing it past its scope — a prior draft of this
//! module did exactly that and was retracted; see
//! `E-NIBLEPATH-DEPTH-IS-NOT-HHTL-DIMENSIONALITY-1` in `EPIPHANIES.md`.**
//!
//! [`crate::hhtl::NiblePath`] is a `u64`-backed SEQUENTIAL router path with
//! [`MAX_DEPTH`](crate::hhtl::MAX_DEPTH) = 16 nibbles, purpose-built for the
//! `subClassOf` Abstammung tree. Four `u16` components are 16 nibbles
//! **exactly**:
//!
//! ```text
//!   domain  4 nibbles ┐
//!   subject 4 nibbles ├─ 16 nibbles = 64 bits = the ENTIRE NiblePath budget
//!   predicate 4       │
//!   object  4         ┘   depth remaining for a NiblePath descent: 0
//! ```
//!
//! That is a real, measured, zero-slack fact about `NiblePath` specifically —
//! `MAX_DEPTH` is a property of one particular single-path router, not of HHTL
//! addressing as a concept. This repo has OTHER HHTL-shaped substrates that do
//! not share this ceiling because they are not modelled as sequential depth
//! descent from one root:
//!
//! - [`crate::facet::FacetCascade`] is a fixed 16-byte register
//!   (`classid(4) | 6×(8:8)`) that supports SEVERAL SIMULTANEOUS
//!   `ClassView`-selected readings of the same bytes (`G3D4`/`G4D3`/`G6D2`,
//!   `24×i4`) — new semantic capability lands as a new *reading*, never a
//!   deeper path. [`crate::tekamolo_facet`] is the shipped instance: four
//!   orthogonal 256:256:256 lanes over one register, not nested depth.
//! - WordNet's real hypernym hierarchy (`E-WORDNET-MAKES-THE-4-ARY-ADDRESS-SEMANTIC-1`,
//!   `probe_wordnet_44_activation.rs`) proves a full-width 4-ary HHTL fold is a
//!   genuinely EXACT structural encoding of real ancestry (corr +0.494 vs
//!   shuffled −0.036) — "HHTL identity" and "one `NiblePath`'s depth ceiling"
//!   are not the same claim.
//! - The 24×i4 anaphora boundary (`E-ANAPHORA-BEYOND-I4-IS-A-BASIN-EDGE-1`)
//!   is the general pattern this module follows: a LOCAL representation
//!   reaching its limit marks a TYPE boundary — switch to a different
//!   sanctioned reading of the SAME address, never invent a deeper/wider path
//!   to route around it.
//!
//! **What follows from the measured fact, and no more:** a `CausalLiteral`'s
//! own identity is not expressible as a STRICTLY SHORTER `NiblePath` prefix
//! while staying exact, and a `NiblePath` built by walking all four
//! components is `is_full()` — nothing can descend beneath it via THAT router.
//! This module therefore exposes [`CausalLiteral::routing_prefix`] as an
//! explicitly LOSSY, depth-truncated cohort projection into the `NiblePath`
//! tree (useful for locality/cohort queries against the existing router), and
//! never treats it as identity. `CausalLiteral` itself is the identity — an
//! 8-byte struct, addressable and hashable on its own terms, independent of
//! whether any particular router can also walk it as a path.
//!
//! # Evidence placement — OPEN, not pre-decided by this module
//!
//! A prior draft of this module concluded that because a full-depth
//! `NiblePath` walk of the literal is `is_full()`, an evidence/meta subtree
//! must therefore "ref-escape" out of address space into some structurally
//! separate mechanism. **That inference is withdrawn as a forced conclusion.**
//! The `NiblePath`-descent option is indeed closed (see above) — but the
//! TEKAMOLO/`FacetCascade` pattern above suggests a live alternative: `S3.1`'s
//! `CausalMeta`/`EpistemicMeta` may land as an ORTHOGONAL FACET/COLUMN keyed
//! by this same 8-byte [`CausalLiteral`] (same shape as "temporal/kausal/
//! modal/lokal are four lanes over one register, not four levels of depth"),
//! rather than a disconnected side structure. This module does not build that
//! — S3.1 decides it, from measurement, when it lands.
//!
//! # Predicate meaning is NOT decided here
//!
//! HHTL supplies hierarchy, locality and exact addressing. It does not decide
//! what `CAUSES` means. A raw `u8`/`u16` ordinal is meaningless without its
//! codebook: the same integer denotes different relations under different
//! families. Resolution — `literal → tenant/ClassView → canonical codebook →
//! ResolvedPredicate`, with unknown predicates failing CLOSED and never
//! composing transitively by default — is S3.3 and lives elsewhere. This
//! module deliberately exposes no `is_transitive`, no relation class, and no
//! composition policy.

use crate::hhtl::{NiblePath, FAN_OUT, MAX_DEPTH};

/// A canonical concept ordinal, resolved through a codebook UPSTREAM of this
/// module (`ogar_codebook::canonical_concept_id` and friends). Raw palette
/// integers are not concepts until a codebook says so.
pub type ConceptId = u16;

/// A canonical semantic-domain / `ClassView` ordinal — the interpretation scope
/// under which `subject`/`predicate`/`object` are resolved.
pub type DomainId = u16;

/// The zero-fallback sentinel, shared by every component: *not routed / not yet
/// bound*, never "concept 0".
///
/// This mirrors the canon's zero-fallback ladder — a zero tier means *not
/// consulted*, never *compacted away*. A literal with an unbound component is
/// still a perfectly well-formed address (construction is total; an address is
/// an address), it is simply not yet fully bound — ask [`CausalLiteral::is_fully_bound`]
/// rather than inferring from the value.
pub const UNBOUND: u16 = 0;

/// The exact, absolute identity of ONE causal proposition.
///
/// Equality is component equality — there is no hash, no learned assignment,
/// and no tolerance, so two distinct canonical tuples **cannot** collide and
/// one canonical tuple **cannot** produce two literals. Both directions are
/// swept in the tests rather than asserted here.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct CausalLiteral {
    domain: DomainId,
    subject: ConceptId,
    predicate: ConceptId,
    object: ConceptId,
}

// 8 bytes of PURE ADDRESS. This assert is the structural guard behind the
// module's headline claim: identity cannot depend on evidence, confidence, a
// source id, or a Lance version, because there is nowhere to put them. Adding
// such a field is a compile error, not a review catch.
const _: () = assert!(core::mem::size_of::<CausalLiteral>() == 8);

/// Nibbles consumed by one `u16` component of the address.
pub const NIBBLES_PER_COMPONENT: u8 = 4;
/// Nibbles consumed by the full `domain·S·P·O` address — exactly [`MAX_DEPTH`].
pub const LITERAL_PATH_NIBBLES: u8 = 4 * NIBBLES_PER_COMPONENT;

// The measured NiblePath-depth fact, compiled in: the full literal path is
// exactly the whole NiblePath budget. If MAX_DEPTH ever widens, this fails
// and the "NiblePath specifically, not HHTL in general" scoping in the module
// doc must be re-derived rather than silently inherited.
const _: () = assert!(LITERAL_PATH_NIBBLES == MAX_DEPTH);

impl CausalLiteral {
    /// Address a canonical proposition. Total by construction — every `u16`
    /// quadruple is a valid address, including partly-[`UNBOUND`] ones.
    ///
    /// Binding *meaning* to the ordinals is a separate, later act (S3.3); this
    /// only says *which* proposition is being spoken about.
    #[must_use]
    pub const fn new(
        domain: DomainId,
        subject: ConceptId,
        predicate: ConceptId,
        object: ConceptId,
    ) -> Self {
        Self {
            domain,
            subject,
            predicate,
            object,
        }
    }

    /// The semantic domain / `ClassView` scope.
    #[must_use]
    pub const fn domain(self) -> DomainId {
        self.domain
    }
    /// The canonical subject ordinal.
    #[must_use]
    pub const fn subject(self) -> ConceptId {
        self.subject
    }
    /// The canonical predicate ordinal. Its *meaning* resolves through the
    /// domain's codebook (S3.3), never from the integer alone.
    #[must_use]
    pub const fn predicate(self) -> ConceptId {
        self.predicate
    }
    /// The canonical object ordinal.
    #[must_use]
    pub const fn object(self) -> ConceptId {
        self.object
    }

    /// Is every component bound (non-[`UNBOUND`])?
    ///
    /// A partly-unbound literal is a legal address but not yet a complete
    /// proposition; callers that require a complete one should gate on this
    /// rather than test components against 0 by hand.
    #[must_use]
    pub const fn is_fully_bound(self) -> bool {
        self.domain != UNBOUND
            && self.subject != UNBOUND
            && self.predicate != UNBOUND
            && self.object != UNBOUND
    }

    /// The packed exact identity, root-first coarse→fine:
    /// `domain << 48 | subject << 32 | predicate << 16 | object`.
    ///
    /// Injective over the component space by construction (four disjoint 16-bit
    /// fields tiling a `u64` exactly), so it is safe as a map key. Swept in
    /// `packed_identity_is_injective` rather than trusted.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        ((self.domain as u64) << 48)
            | ((self.subject as u64) << 32)
            | ((self.predicate as u64) << 16)
            | (self.object as u64)
    }

    /// Inverse of [`as_u64`](Self::as_u64) — total, and exactly reversible.
    #[must_use]
    pub const fn from_u64(v: u64) -> Self {
        Self {
            domain: (v >> 48) as u16,
            subject: (v >> 32) as u16,
            predicate: (v >> 16) as u16,
            object: v as u16,
        }
    }

    /// The 8-byte little-endian persisted form of [`as_u64`](Self::as_u64).
    ///
    /// This is the identity that goes to storage: canonical ordinals, never
    /// runtime strings. A literal minted from the same canonical tuple in a
    /// different process, a different Lance version, or a different source
    /// serializes to the same eight bytes.
    #[must_use]
    pub const fn to_le_bytes(self) -> [u8; 8] {
        self.as_u64().to_le_bytes()
    }

    /// Inverse of [`to_le_bytes`](Self::to_le_bytes).
    #[must_use]
    pub const fn from_le_bytes(b: [u8; 8]) -> Self {
        Self::from_u64(u64::from_le_bytes(b))
    }

    /// A **routing / cohort** projection into the `NiblePath` Abstammung tree
    /// — the first `depth` nibbles of the root-first `domain·S·P·O` sequence.
    ///
    /// This is ONE way to relate a literal to the existing `subClassOf`
    /// router; it is not "the HHTL form" of the literal, and `CausalLiteral`
    /// itself (not this projection) is the identity — see the module doc's
    /// "NiblePath's depth budget" section for why the two are not the same
    /// claim.
    ///
    /// # This is NOT identity
    ///
    /// At `depth < LITERAL_PATH_NIBBLES` the projection is **many-to-one** by
    /// design: that is what makes it useful as a deterministic cohort slice
    /// (all literals in a domain; all literals sharing a domain and subject).
    /// Two different propositions genuinely share a prefix, and
    /// `routing_prefix_is_not_identity` proves it on real values so the
    /// projection can never be quietly promoted into an equality test — the
    /// exact confusion Stage-3 falsifiers #3 and #19 name.
    ///
    /// At `depth == LITERAL_PATH_NIBBLES` it is injective — and simultaneously
    /// `is_full()` for the `NiblePath` router, so nothing can descend beneath
    /// it VIA THAT ROUTER. That is a fact about `NiblePath`'s single-path
    /// design, not a ceiling on where evidence/meta state may live (see
    /// "Evidence placement" above).
    ///
    /// `depth` saturates at [`LITERAL_PATH_NIBBLES`].
    #[must_use]
    pub fn routing_prefix(self, depth: u8) -> NiblePath {
        let depth = depth.min(LITERAL_PATH_NIBBLES);
        let packed = self.as_u64();
        let mut path = NiblePath::EMPTY;
        for i in 0..depth {
            // root-first: nibble 0 is the most significant of the 16.
            let shift = 4 * (LITERAL_PATH_NIBBLES - 1 - i) as u32;
            let nibble = ((packed >> shift) & 0xF) as u8;
            debug_assert!(nibble < FAN_OUT, "a 4-bit value is always < FAN_OUT");
            path = if i == 0 {
                NiblePath::root(nibble)
            } else {
                path.child(nibble)
            };
        }
        path
    }

    /// The full-depth routing projection — exact, and exactly `is_full()` for
    /// the `NiblePath` router.
    ///
    /// Provided for completeness and for the budget falsifier; prefer
    /// [`as_u64`](Self::as_u64) / [`to_le_bytes`](Self::to_le_bytes) as the
    /// identity, and a SHORTER [`routing_prefix`](Self::routing_prefix) as the
    /// cohort. A caller reaching for this as a `NiblePath` tree root is about
    /// to discover it cannot descend further via that router.
    #[must_use]
    pub fn full_path(self) -> NiblePath {
        self.routing_prefix(LITERAL_PATH_NIBBLES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    /// A deliberately varied sweep: every component takes low, mid, high and
    /// boundary values, and the values are REUSED across positions so a
    /// field-order bug shows up as a collision rather than hiding.
    fn sweep() -> Vec<CausalLiteral> {
        let vals: [u16; 6] = [0, 1, 2, 0x00FF, 0x8000, u16::MAX];
        let mut out = Vec::new();
        for &d in &vals {
            for &s in &vals {
                for &p in &vals {
                    for &o in &vals {
                        out.push(CausalLiteral::new(d, s, p, o));
                    }
                }
            }
        }
        out
    }

    /// FALSIFIER #1 — the same canonical tuple always produces the same
    /// literal, and FALSIFIER #2 — distinct tuples never collide.
    ///
    /// Both directions on one sweep of 1,296 tuples. The reused value set means
    /// a swapped field order (e.g. predicate and object transposed) collapses
    /// distinct tuples onto one `u64` and fails the injectivity half.
    #[test]
    fn packed_identity_is_injective() {
        let all = sweep();
        // anti-vacuity: the sweep must actually contain distinct tuples that
        // differ in ONLY one position, or injectivity is trivially satisfiable.
        assert!(all.len() >= 1000, "sweep too small: {}", all.len());

        let mut seen: HashMap<u64, CausalLiteral> = HashMap::new();
        for lit in &all {
            // determinism: rebuilding from the same components is identical
            let again =
                CausalLiteral::new(lit.domain(), lit.subject(), lit.predicate(), lit.object());
            assert_eq!(*lit, again, "same canonical tuple produced two literals");
            assert_eq!(
                lit.as_u64(),
                again.as_u64(),
                "identity is not deterministic"
            );

            if let Some(prev) = seen.insert(lit.as_u64(), *lit) {
                assert_eq!(
                    prev, *lit,
                    "two DISTINCT canonical tuples collided on one identity"
                );
            }
        }
        assert_eq!(seen.len(), all.len(), "identity is not injective");
    }

    /// FALSIFIER #2, sharpened — changing ONLY the predicate must change the
    /// literal. A "causes" and a "treated_with" between the same two concepts
    /// are different propositions, and the address has to say so.
    ///
    /// Two-sided: the paired half proves that changing nothing changes nothing,
    /// so this cannot pass by an implementation that simply returns fresh
    /// values.
    #[test]
    fn changing_only_the_predicate_changes_the_literal() {
        let causes = CausalLiteral::new(7, 100, 42, 200);
        let treated_with = CausalLiteral::new(7, 100, 43, 200);
        assert_ne!(causes, treated_with, "distinct predicates collided");
        assert_ne!(causes.as_u64(), treated_with.as_u64());
        assert_ne!(causes.to_le_bytes(), treated_with.to_le_bytes());
        // …and the silence half
        let same = CausalLiteral::new(7, 100, 42, 200);
        assert_eq!(causes, same, "identical tuples must be one literal");
        assert_eq!(causes.to_le_bytes(), same.to_le_bytes());
    }

    /// FALSIFIER #4 — many sources, ONE proposition.
    ///
    /// Three "sources" independently address the same canonical tuple. They
    /// must converge on a single identity: one literal, three witnesses, never
    /// three literals. Modelled as three separately-constructed values whose
    /// set collapses to size one.
    #[test]
    fn three_sources_asserting_the_same_proposition_mint_one_literal() {
        let paper_1 = CausalLiteral::new(3, 8_001, 42, 9_002);
        let paper_2 = CausalLiteral::new(3, 8_001, 42, 9_002);
        let model_3 = CausalLiteral::from_le_bytes(paper_1.to_le_bytes());
        let distinct: HashSet<u64> = [paper_1, paper_2, model_3]
            .iter()
            .map(|l| l.as_u64())
            .collect();
        assert_eq!(
            distinct.len(),
            1,
            "same proposition minted several literals"
        );
        // anti-vacuity: a genuinely different proposition still separates
        let other = CausalLiteral::new(3, 8_001, 42, 9_003);
        assert!(!distinct.contains(&other.as_u64()));
    }

    /// Exact reversibility — the address resolves back to its components, in
    /// both serialized forms, over the whole sweep.
    #[test]
    fn identity_round_trips_exactly_in_both_forms() {
        for lit in sweep() {
            assert_eq!(CausalLiteral::from_u64(lit.as_u64()), lit, "u64 round trip");
            assert_eq!(
                CausalLiteral::from_le_bytes(lit.to_le_bytes()),
                lit,
                "le-bytes round trip"
            );
            // and the components survive individually, not merely in aggregate
            let r = CausalLiteral::from_le_bytes(lit.to_le_bytes());
            assert_eq!(
                (r.domain(), r.subject(), r.predicate(), r.object()),
                (lit.domain(), lit.subject(), lit.predicate(), lit.object())
            );
        }
    }

    /// Field isolation (the layout discipline `I-LEGACY-API-FEATURE-GATED`
    /// prescribes for any new packing): each component, changed from a FULLY
    /// NON-ZERO baseline, moves its own accessor and leaves the other three
    /// bit-identical. A zeroed baseline can hide a field that ORs into a
    /// neighbour's set bits, so the baseline is deliberately all-`0xABCD`.
    #[test]
    fn component_isolation_matrix() {
        const B: u16 = 0xABCD;
        let base = CausalLiteral::new(B, B, B, B);
        let probe: u16 = 0x1234;

        let cases: [(&str, CausalLiteral); 4] = [
            ("domain", CausalLiteral::new(probe, B, B, B)),
            ("subject", CausalLiteral::new(B, probe, B, B)),
            ("predicate", CausalLiteral::new(B, B, probe, B)),
            ("object", CausalLiteral::new(B, B, B, probe)),
        ];
        for (name, got) in cases {
            assert_ne!(got, base, "{name}: changing it changed nothing");
            let moved = [
                got.domain() != base.domain(),
                got.subject() != base.subject(),
                got.predicate() != base.predicate(),
                got.object() != base.object(),
            ];
            assert_eq!(
                moved.iter().filter(|m| **m).count(),
                1,
                "{name}: exactly one component may move, got {moved:?}"
            );
            assert!(
                match name {
                    "domain" => moved[0],
                    "subject" => moved[1],
                    "predicate" => moved[2],
                    _ => moved[3],
                },
                "{name}: the wrong component moved"
            );
        }
    }

    /// THE ANTI-VACUITY TWIN, and the point of the whole split: the routing
    /// prefix is a **cohort**, not an identity.
    ///
    /// Guards Stage-3 falsifiers #3 and #19 at their shared root — mistaking an
    /// approximate/positional projection for exact proposition identity. If a
    /// future edit made `routing_prefix` injective at shallow depth (say by
    /// folding all four components in), this test goes red and forces the
    /// author to say so out loud.
    #[test]
    fn routing_prefix_is_not_identity() {
        // same domain + subject + predicate, different object
        let a = CausalLiteral::new(0x1111, 0x2222, 0x3333, 0x4444);
        let b = CausalLiteral::new(0x1111, 0x2222, 0x3333, 0x5555);
        assert_ne!(a, b, "fixture is degenerate: the two literals are equal");

        // 12 nibbles = domain+subject+predicate — they MUST share it
        assert_eq!(
            a.routing_prefix(12),
            b.routing_prefix(12),
            "the prefix failed to group two literals that share domain+S+P — \
             it is not usable as a cohort"
        );
        // 4 nibbles = the domain cohort
        assert_eq!(a.routing_prefix(4), b.routing_prefix(4));

        // …and the paired half: at full depth it discriminates, so the
        // projection is lossy by DEPTH rather than simply broken.
        assert_ne!(
            a.full_path(),
            b.full_path(),
            "the full-depth path failed to separate distinct literals"
        );
    }

    /// The `NiblePath` depth-budget finding, pinned as a guard rather than
    /// left in prose: the full-depth path is exactly `MAX_DEPTH`, therefore
    /// `is_full()` for that router — a fact about `NiblePath`'s single-path
    /// design, scoped exactly to that type (see the module doc).
    ///
    /// If `MAX_DEPTH` ever widens this fails alongside the `const _` assert, and
    /// the module-doc scoping gets re-derived instead of silently inherited.
    #[test]
    fn the_full_literal_path_exhausts_the_niblepath_budget() {
        let lit = CausalLiteral::new(0x1234, 0x5678, 0x9ABC, 0xDEF0);
        let full = lit.full_path();
        assert_eq!(full.depth(), MAX_DEPTH, "full path is not the whole budget");
        assert!(full.is_full(), "full path must report is_full");
        // the operational consequence: a descent is a silent no-op…
        assert_eq!(full.child(1), full, "descent past MAX_DEPTH is not a no-op");
        // …and the explicit form refuses instead of colliding.
        assert!(
            full.try_child(1).is_none(),
            "try_child must refuse at the ceiling"
        );
        // a SHORTER prefix, by contrast, still has room — proving the ceiling
        // is a property of the full-depth NiblePath walk, not of the literal
        // or of HHTL addressing in general.
        assert!(!lit.routing_prefix(12).is_full());
    }

    /// The prefix is monotone and deterministic: extending the depth extends
    /// the path, and the same literal always yields the same prefix.
    #[test]
    fn routing_prefix_is_deterministic_and_monotone_in_depth() {
        let lit = CausalLiteral::new(0x0102, 0x0304, 0x0506, 0x0708);
        for d in 0..=LITERAL_PATH_NIBBLES {
            assert_eq!(
                lit.routing_prefix(d),
                lit.routing_prefix(d),
                "prefix is not deterministic at depth {d}"
            );
            assert_eq!(
                lit.routing_prefix(d).depth(),
                d,
                "prefix depth mismatch at {d}"
            );
        }
        // saturates rather than wrapping or panicking
        assert_eq!(
            lit.routing_prefix(200),
            lit.full_path(),
            "depth must saturate at the budget"
        );
    }

    /// Zero-fallback: an unbound component is a legal address that is visibly
    /// incomplete — callers ask, rather than testing components against 0.
    #[test]
    fn unbound_components_are_addressable_but_not_fully_bound() {
        assert!(!CausalLiteral::default().is_fully_bound());
        assert!(!CausalLiteral::new(1, 1, UNBOUND, 1).is_fully_bound());
        assert!(CausalLiteral::new(1, 1, 1, 1).is_fully_bound());
        // an unbound-predicate literal is still a DISTINCT address, not a
        // sentinel that collapses onto something else
        assert_ne!(
            CausalLiteral::new(1, 1, UNBOUND, 1),
            CausalLiteral::new(1, 1, 1, 1)
        );
    }
}
