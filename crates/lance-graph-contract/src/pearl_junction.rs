//! # `pearl_junction` — Pearl's three causal junctions over HHTL identity
//!
//! Classifies a pair of SPO edges into one of Pearl's three causal junctions
//! (chain / fork / collider) plus the reverse chain. The classification is
//! a pure function of identity equality between the four `NiblePath`
//! endpoints (subject and object of each edge); no graph walk is required.
//!
//! ## The mapping (E-4 corrected, per `bardioc/.claude/EPIPHANIES.md`)
//!
//! Reading `s -> o` as `s subClassOf o` (or any other transitive edge):
//!
//! | Junction       | Shared term         | Example                          | NARS rule  | ΔHHTL signature              |
//! |----------------|---------------------|----------------------------------|------------|------------------------------|
//! | **Chain**      | `o1 == s2`          | `dog -> mammal -> animal`        | Deduction  | small, along one lineage     |
//! | **ChainRev**   | `s1 == o2`          | reverse of Chain                 | Deduction  | small, along one lineage     |
//! | **Fork**       | `s1 == s2` (child)  | `dog -> mammal`, `dog -> pet`    | Induction  | 2× up to common child        |
//! | **Collider**   | `o1 == o2` (parent) | `dog -> mammal`, `cat -> mammal` | Abduction  | 2× up to common parent       |
//! | **Unrelated**  | (no shared term)    | —                                | —          | —                            |
//!
//! Anti-swap guard (per peer-review round-2 — the earlier `SharedSubject =
//! sibling-via-parent` / `SharedObject = sibling-via-child` framing inverted
//! the induction⇄abduction chirality; this module's tests use the
//! `dog/cat/mammal` example as the canonical anti-swap guard).
//!
//! ## Why this is in the contract crate
//!
//! The classifier is pure-function — it does NOT touch storage, indexes,
//! or any planner state. It IS the bridge between SPO grammar (figure
//! rules) and HHTL identity addressing. Per the Morris semiotic trichotomy
//! mapped to lance-graph code (see `EPIPHANIES.md`), this is **syntax**
//! (figure rules) operating over **semantics** (HHTL nodes); pragmatics
//! (the cascade fold) consumes the classification at runtime.

use crate::hhtl::NiblePath;

/// Pearl's causal-junction taxonomy applied to a pair of SPO edges.
///
/// The classification is determined by identity equality between the
/// four endpoints (`s1`, `o1`, `s2`, `o2`); no graph walk is required.
/// See module docstring for the canonical mapping + examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PearlJunction {
    /// `o1 == s2` — chain: `s1 -> o1=s2 -> o2`. Head-to-tail. Deduction.
    Chain,
    /// `s1 == o2` — reverse chain: `o1 <- s1=o2 <- s2`. Head-to-tail (other
    /// direction). Deduction.
    ChainRev,
    /// `s1 == s2` — fork (common cause): the shared subject is the
    /// **child**; `o1` and `o2` are co-parents reachable via one common
    /// descendant. Conclusion `o1 -> o2` is Induction.
    Fork,
    /// `o1 == o2` — collider (explaining-away): the shared object is the
    /// **parent**; `s1` and `s2` are siblings under one common ancestor.
    /// Conclusion `s1 -> s2` is Abduction.
    Collider,
    /// No shared term between the two edges.
    Unrelated,
}

impl PearlJunction {
    /// Stable label for reports / logs / diff dimensions.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Chain => "chain",
            Self::ChainRev => "chain_rev",
            Self::Fork => "fork",
            Self::Collider => "collider",
            Self::Unrelated => "unrelated",
        }
    }

    /// The NARS-style inference rule the junction selects. `None` for
    /// `Unrelated`. (Chain / ChainRev select Deduction; Fork selects
    /// Induction; Collider selects Abduction.)
    pub const fn nars_rule(self) -> Option<NarsRule> {
        match self {
            Self::Chain | Self::ChainRev => Some(NarsRule::Deduction),
            Self::Fork => Some(NarsRule::Induction),
            Self::Collider => Some(NarsRule::Abduction),
            Self::Unrelated => None,
        }
    }
}

/// The NARS-style inference rule a Pearl junction selects.
///
/// Mirrors the canonical NARS rule taxonomy (Deduction / Induction /
/// Abduction). The `lance-graph-contract::nars` module owns the full
/// `InferenceType` enum (5 variants); this enum names only the three
/// rules that arise from Pearl-junction classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NarsRule {
    /// Chain figure: `M -> P`, `S -> M` ⊢ `S -> P` (or the reverse).
    Deduction,
    /// Fork figure (common cause): `M -> P`, `M -> S` ⊢ `S -> P` (with
    /// confidence calibrated by Pearl's induction discounting).
    Induction,
    /// Collider figure (explaining-away): `P -> M`, `S -> M` ⊢ `S -> P`
    /// (with confidence calibrated by Pearl's abduction discounting).
    Abduction,
}

/// Classify a pair of SPO edges by Pearl-junction taxonomy.
///
/// The four arguments are the subject and object identities of each edge.
/// The predicate is intentionally not in the classifier — the junction
/// type is determined by the topology of identity equality, not by which
/// relation each edge represents. Consumers that need predicate-aware
/// dispatch (e.g. weighting predicates differently) layer that on top.
///
/// The classifier checks for shared identity in this order:
/// 1. `Chain` (`o1 == s2`)
/// 2. `ChainRev` (`s1 == o2`)
/// 3. `Fork` (`s1 == s2`)
/// 4. `Collider` (`o1 == o2`)
/// 5. otherwise `Unrelated`
///
/// When two edges share BOTH endpoints (e.g. `s1 == s2` AND `o1 == o2`),
/// the classifier returns `Chain` only if the chain check fires first;
/// otherwise it follows the order above. Duplicate edges should be
/// deduplicated by the caller before classification.
pub const fn classify_junction(
    s1: NiblePath,
    o1: NiblePath,
    s2: NiblePath,
    o2: NiblePath,
) -> PearlJunction {
    if niblepath_eq(o1, s2) {
        return PearlJunction::Chain;
    }
    if niblepath_eq(s1, o2) {
        return PearlJunction::ChainRev;
    }
    if niblepath_eq(s1, s2) {
        return PearlJunction::Fork;
    }
    if niblepath_eq(o1, o2) {
        return PearlJunction::Collider;
    }
    PearlJunction::Unrelated
}

/// `const fn` equality for [`NiblePath`] — needed because `PartialEq` for
/// user types is not `const` in stable Rust 1.95. Two paths are equal iff
/// their packed `(path, depth)` agree.
const fn niblepath_eq(a: NiblePath, b: NiblePath) -> bool {
    let (ap, ad) = a.packed();
    let (bp, bd) = b.packed();
    ap == bp && ad == bd
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dog/cat/mammal canonical example — the anti-swap guard.
    ///
    /// Two `subClassOf` edges share the same OBJECT (`mammal`). The shared
    /// term is the parent; the two subjects (`dog`, `cat`) are siblings
    /// under it; the conclusion `dog -> cat` is Abduction. This is the
    /// COLLIDER pattern, not the Fork pattern (the earlier incorrect
    /// framing inverted these).
    #[test]
    fn collider_is_dog_cat_mammal_with_shared_object() {
        let dog = NiblePath::root(0x1).child(0x1);
        let cat = NiblePath::root(0x1).child(0x2);
        let mammal = NiblePath::root(0x1);

        // dog -> mammal, cat -> mammal: shared object (mammal = parent),
        // distinct subjects (dog, cat = siblings).
        let j = classify_junction(dog, mammal, cat, mammal);
        assert_eq!(j, PearlJunction::Collider);
        assert_eq!(j.nars_rule(), Some(NarsRule::Abduction));
        assert_eq!(j.label(), "collider");
    }

    /// The dog->mammal / dog->pet example — the Fork canonical.
    ///
    /// Two edges share the same SUBJECT (`dog`). The shared term is the
    /// child; the two objects (`mammal`, `pet`) are co-parents
    /// reachable via the common descendant; the conclusion `mammal -> pet`
    /// is Induction.
    #[test]
    fn fork_is_dog_mammal_pet_with_shared_subject() {
        let dog = NiblePath::root(0x1).child(0x1);
        let mammal = NiblePath::root(0x1);
        let pet = NiblePath::root(0x2);

        let j = classify_junction(dog, mammal, dog, pet);
        assert_eq!(j, PearlJunction::Fork);
        assert_eq!(j.nars_rule(), Some(NarsRule::Induction));
        assert_eq!(j.label(), "fork");
    }

    /// Chain: `dog -> mammal -> animal`. `o1 == s2`.
    #[test]
    fn chain_is_dog_mammal_animal_head_to_tail() {
        let dog = NiblePath::root(0x1).child(0x1);
        let mammal = NiblePath::root(0x1);
        let animal = NiblePath::root(0x0);
        // dog -> mammal, mammal -> animal: o1 (mammal) == s2 (mammal)
        let j = classify_junction(dog, mammal, mammal, animal);
        assert_eq!(j, PearlJunction::Chain);
        assert_eq!(j.nars_rule(), Some(NarsRule::Deduction));
    }

    /// ChainRev: `s1 == o2`.
    #[test]
    fn chain_rev_is_when_s1_equals_o2() {
        let a = NiblePath::root(0x1);
        let b = NiblePath::root(0x2);
        let c = NiblePath::root(0x3);
        // a -> b, c -> a: s1 (a) == o2 (a)
        let j = classify_junction(a, b, c, a);
        assert_eq!(j, PearlJunction::ChainRev);
        assert_eq!(j.nars_rule(), Some(NarsRule::Deduction));
    }

    /// Unrelated: no shared term.
    #[test]
    fn unrelated_when_no_shared_term() {
        let a = NiblePath::root(0x1);
        let b = NiblePath::root(0x2);
        let c = NiblePath::root(0x3);
        let d = NiblePath::root(0x4);
        let j = classify_junction(a, b, c, d);
        assert_eq!(j, PearlJunction::Unrelated);
        assert_eq!(j.nars_rule(), None);
    }

    /// Order-of-checks: when multiple endpoints match, Chain wins first.
    /// Documents the deterministic behavior for callers.
    #[test]
    fn chain_check_fires_before_other_matches() {
        let x = NiblePath::root(0x1);
        let y = NiblePath::root(0x2);
        // edges x->y and y->x: o1 (y) == s2 (y) → Chain
        // (also s1 == o2 → would-be ChainRev; Chain check fires first)
        let j = classify_junction(x, y, y, x);
        assert_eq!(j, PearlJunction::Chain);
    }

    #[test]
    fn const_eq_works_in_classify() {
        // const-context test for the classifier (proves const fn nature)
        const A: NiblePath = NiblePath::root(0x1);
        const B: NiblePath = NiblePath::root(0x2);
        const C: NiblePath = NiblePath::root(0x3);
        // a->b, b->c (Chain)
        const J: PearlJunction = classify_junction(A, B, B, C);
        assert_eq!(J, PearlJunction::Chain);
    }
}
