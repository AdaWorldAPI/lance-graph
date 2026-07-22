//! `ancestry` — the family radix-trie codebook: **ancestry lives in the KEY,
//! never in derived triples.**
//!
//! The operator ruling that reshaped `D-SRS-2` (`self-reasoning-substrate-v1`):
//! ancestry is classic **HHTL family identity** — Distinguished-Name-like
//! chains, a radix-trie codebook ontology. A node's DN (its root-path) IS its
//! lineage, so `is_ancestor_of(A, Z)` = *A's DN is a strict prefix of Z's DN* —
//! radix-trie containment, the same law as the 4⁴ centroid-hierarchy canon
//! (`is_ancestor_of` = centroid-tree containment) and the `part_of:is_a` rails.
//!
//! This is the amortization that retires the `D-SRS-1` O(N²) finding: a
//! materialized transitive closure stores **every ancestor pair**; the trie
//! stores **one parent pointer per entity** and answers any ancestor query by a
//! prefix read. The closure was the wrong carrier, not merely an unbounded one.
//!
//! ## Contract (pre-registered in the plan, G-SRS2 trie contract)
//!
//! - Each covered entity has ONE parent — the FIRST parent edge seen wins
//!   (primary); later conflicting parent edges are counted as
//!   [`multi_parent_residue`](FamilyTrie::multi_parent_residue) and are NOT in
//!   the forest (they route to the fabric / Escalate, never silently dropped
//!   into the trie).
//! - Entities on a parent-cycle cannot reach a root; they are UNCOVERED
//!   ([`cycle_residue`](FamilyTrie::cycle_residue)) — the trie never lies about
//!   a lineage it cannot ground.
//! - `is_ancestor_of` is exact on the covered forest and `false` for anything
//!   uncovered.

use std::collections::HashMap;

/// The family radix-trie: one primary parent pointer per covered entity, DN =
/// the root-path, ancestry = DN prefix containment.
#[derive(Debug, Clone)]
pub struct FamilyTrie {
    /// child → primary parent (first-wins), for every child that appeared.
    parent: HashMap<u16, u16>,
    /// entity → depth (root = 0), for COVERED entities only.
    depth: HashMap<u16, u32>,
    /// Parent edges rejected because the child already had a primary parent.
    multi_parent: usize,
    /// Entities excluded because their parent walk loops (never reaches a root).
    on_cycle: usize,
}

impl FamilyTrie {
    /// Build the trie from `(parent, child)` edges (for an SPO triple
    /// `(A, begat, B)`, `A` is the parent and `B` the child).
    ///
    /// First parent wins; later parent edges for the same child count as
    /// multi-parent residue. After the parent map is fixed, every entity is
    /// walked to a root; entities whose walk revisits a node (a parent cycle)
    /// are uncovered cycle-residue.
    #[must_use]
    pub fn build(edges: &[(u16, u16)]) -> Self {
        let mut parent: HashMap<u16, u16> = HashMap::new();
        let mut multi_parent = 0usize;
        for &(p, c) in edges {
            if p == c {
                multi_parent += 1; // a self-parent can never ground a lineage
                continue;
            }
            match parent.entry(c) {
                std::collections::hash_map::Entry::Occupied(_) => multi_parent += 1,
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(p);
                }
            }
        }

        // Cover: walk each entity to a root, memoizing depth. A walk that
        // revisits a node in progress is a cycle — everyone on it is uncovered.
        let mut depth: HashMap<u16, u32> = HashMap::new();
        let mut on_cycle_set: HashMap<u16, bool> = HashMap::new(); // true = cyclic
        let mut entities: Vec<u16> = parent.keys().copied().collect();
        entities.extend(parent.values().copied());
        entities.sort_unstable();
        entities.dedup();

        for &e in &entities {
            if depth.contains_key(&e) || on_cycle_set.contains_key(&e) {
                continue;
            }
            // Walk up, recording the chain until a known node or a root.
            let mut chain: Vec<u16> = Vec::new();
            let mut chain_pos: HashMap<u16, usize> = HashMap::new();
            let mut cur = e;
            let outcome: Result<u32, usize> = loop {
                if let Some(&d) = depth.get(&cur) {
                    break Ok(d); // reached an already-covered node
                }
                if on_cycle_set.get(&cur) == Some(&true) {
                    break Err(0); // reached a known-cyclic node — whole chain sinks
                }
                if let Some(&pos) = chain_pos.get(&cur) {
                    break Err(pos); // new cycle discovered within this walk
                }
                chain_pos.insert(cur, chain.len());
                chain.push(cur);
                match parent.get(&cur) {
                    Some(&p) => cur = p,
                    None => break Ok(u32::MAX), // root sentinel (depth -1 conceptually)
                }
            };
            match outcome {
                Ok(base) => {
                    // `chain` was built walking UP (leaf-ward first), so its
                    // LAST element is root-most. Assign depths in reverse:
                    // root-most gets the smallest depth (0 at a true root, or
                    // one below an already-covered node at depth `base`).
                    let start = if base == u32::MAX { 0 } else { base + 1 };
                    for (i, &n) in chain.iter().rev().enumerate() {
                        depth.insert(n, start + i as u32);
                    }
                }
                Err(_) => {
                    for &n in &chain {
                        on_cycle_set.insert(n, true);
                    }
                }
            }
        }

        let on_cycle = on_cycle_set.len();
        Self {
            parent,
            depth,
            multi_parent,
            on_cycle,
        }
    }

    /// Number of covered entities (each stores exactly one parent pointer or is
    /// a root) — the trie's total storage in the amortization gate.
    #[must_use]
    pub fn covered(&self) -> usize {
        self.depth.len()
    }

    /// Parent edges rejected by first-wins (routed to fabric/Escalate).
    #[must_use]
    pub fn multi_parent_residue(&self) -> usize {
        self.multi_parent
    }

    /// Entities uncovered because their lineage loops.
    #[must_use]
    pub fn cycle_residue(&self) -> usize {
        self.on_cycle
    }

    /// The DN (root-path, root first) of a covered entity.
    #[must_use]
    pub fn dn(&self, e: u16) -> Option<Vec<u16>> {
        self.depth.get(&e)?;
        let mut path = vec![e];
        let mut cur = e;
        while let Some(&p) = self.parent.get(&cur) {
            if !self.depth.contains_key(&p) {
                break;
            }
            path.push(p);
            cur = p;
        }
        path.reverse();
        Some(path)
    }

    /// `is_ancestor_of(A, Z)` — A's DN is a STRICT prefix of Z's DN, answered by
    /// walking Z up exactly `depth(Z) − depth(A)` steps (reading the DN tail);
    /// the O(1) form is the nibble-packed HHTL key compare of the same path.
    #[must_use]
    pub fn is_ancestor_of(&self, a: u16, z: u16) -> bool {
        let (Some(&da), Some(&dz)) = (self.depth.get(&a), self.depth.get(&z)) else {
            return false;
        };
        if da >= dz {
            return false;
        }
        let mut cur = z;
        for _ in 0..(dz - da) {
            match self.parent.get(&cur) {
                Some(&p) => cur = p,
                None => return false,
            }
        }
        cur == a
    }

    /// The NUMBER of ancestor pairs the covered forest implies (`Σ depth(v)`) —
    /// the amortization numerator, computed without materializing the set.
    #[must_use]
    pub fn pair_count(&self) -> usize {
        self.depth.values().map(|&d| d as usize).sum()
    }

    /// Every ancestor pair `(ancestor, descendant)` the covered forest implies —
    /// the set the G-SRS2-a falsifier compares against the materialized closure.
    /// Size = Σ depth(v).
    #[must_use]
    pub fn ancestor_pairs(&self) -> std::collections::HashSet<(u16, u16)> {
        let mut out = std::collections::HashSet::new();
        for &e in self.depth.keys() {
            let mut cur = e;
            while let Some(&p) = self.parent.get(&cur) {
                if !self.depth.contains_key(&p) {
                    break;
                }
                out.insert((p, e));
                cur = p;
            }
        }
        out
    }

    /// The covered forest's edges `(parent, child)` — exactly what the trie
    /// represents (residue excluded). The falsifier closes THESE edges.
    #[must_use]
    pub fn forest_edges(&self) -> Vec<(u16, u16)> {
        self.depth
            .keys()
            .filter_map(|&c| {
                let &p = self.parent.get(&c)?;
                self.depth.contains_key(&p).then_some((p, c))
            })
            .collect()
    }

    /// Maximum DN depth over covered entities (root = 0).
    #[must_use]
    pub fn max_depth(&self) -> u32 {
        self.depth.values().copied().max().unwrap_or(0)
    }

    /// The HHTL-packable share: covered entities whose depth ≤ 12 (the native
    /// 3×4-nibble path) AND whose every path node has ≤ 16 children in the
    /// forest (`FAN_OUT = 16`). Deeper/wider lineages are the hierarchy's
    /// registry-resolve + ref-escape job (canon), not a trie failure.
    #[must_use]
    pub fn hhtl_packable(&self) -> usize {
        // children count per covered parent.
        let mut fan: HashMap<u16, u32> = HashMap::new();
        for (p, _) in self.forest_edges() {
            *fan.entry(p).or_insert(0) += 1;
        }
        self.depth
            .iter()
            .filter(|&(&e, &d)| {
                if d > 12 {
                    return false;
                }
                // every ancestor on the path (and the entity's own parent hop)
                // must have fan-out ≤ 16.
                let mut cur = e;
                while let Some(&p) = self.parent.get(&cur) {
                    if !self.depth.contains_key(&p) {
                        break;
                    }
                    if fan.get(&p).copied().unwrap_or(0) > 16 {
                        return false;
                    }
                    cur = p;
                }
                true
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reason::DerivationArena;
    use crate::spo::Spo;

    /// A 4-generation chain: DNs are root-paths, ancestry is prefix, pairs are
    /// the triangular count.
    #[test]
    fn chain_dn_prefix_is_ancestry() {
        // 1 → 2 → 3 → 4
        let t = FamilyTrie::build(&[(1, 2), (2, 3), (3, 4)]);
        assert_eq!(t.covered(), 4);
        assert_eq!(t.multi_parent_residue(), 0);
        assert_eq!(t.cycle_residue(), 0);
        assert_eq!(t.dn(4), Some(vec![1, 2, 3, 4]));
        assert_eq!(t.dn(1), Some(vec![1]));
        assert!(t.is_ancestor_of(1, 4));
        assert!(t.is_ancestor_of(2, 4));
        assert!(t.is_ancestor_of(3, 4));
        assert!(!t.is_ancestor_of(4, 1));
        assert!(!t.is_ancestor_of(2, 1));
        assert!(!t.is_ancestor_of(4, 4)); // strict prefix — never self
        assert_eq!(t.ancestor_pairs().len(), 3 + 2 + 1);
        assert_eq!(t.max_depth(), 3);
    }

    /// Multi-parent edges: first wins, the rest are residue; the forest stays a
    /// forest.
    #[test]
    fn multi_parent_first_wins_rest_is_residue() {
        // 1→3 seen first; 2→3 is residue. 3→4 continues the line.
        let t = FamilyTrie::build(&[(1, 3), (2, 3), (3, 4)]);
        assert_eq!(t.multi_parent_residue(), 1);
        assert!(t.is_ancestor_of(1, 4));
        assert!(!t.is_ancestor_of(2, 4)); // the residue edge is NOT in the trie
        assert_eq!(t.dn(4), Some(vec![1, 3, 4]));
    }

    /// A parent cycle never grounds: its members are uncovered, everyone else
    /// is unaffected.
    #[test]
    fn cycle_members_are_uncovered() {
        // 1→2→3→1 is a cycle; 5→6 is a clean line; 3→7 hangs off the cycle.
        let t = FamilyTrie::build(&[(1, 2), (2, 3), (3, 1), (5, 6), (3, 7)]);
        assert!(t.cycle_residue() >= 3, "cycle members uncovered: {t:?}");
        assert!(!t.is_ancestor_of(1, 3));
        assert!(t.is_ancestor_of(5, 6));
        // 7's lineage passes through the cycle → cannot ground → uncovered.
        assert_eq!(t.dn(7), None);
        assert!(!t.is_ancestor_of(3, 7));
    }

    /// Self-parent edges are residue, never a 1-cycle in the trie.
    #[test]
    fn self_parent_is_residue() {
        let t = FamilyTrie::build(&[(1, 1), (1, 2)]);
        assert_eq!(t.multi_parent_residue(), 1);
        assert!(t.is_ancestor_of(1, 2));
    }

    /// THE mini-falsifier (G-SRS2-a in unit form): on a branching forest, the
    /// trie's implied ancestor pairs EQUAL the uncapped transitive closure of
    /// the same forest edges (base ∪ derived), as sets, both directions.
    #[test]
    fn trie_pairs_equal_uncapped_closure_exactly() {
        // Forest: 1→{2,3}, 2→{4,5}, 5→6; second tree 10→11.
        let edges = [(1, 2), (1, 3), (2, 4), (2, 5), (5, 6), (10, 11)];
        let t = FamilyTrie::build(&edges);
        assert_eq!(t.covered(), 8);

        let p = 42u16;
        let base: Vec<Spo> = t
            .forest_edges()
            .iter()
            .map(|&(a, c)| Spo::new(a, p, c))
            .collect();
        let arena = DerivationArena::derive_transitive(&base);
        let g = arena.gate();
        assert!(g.passed(), "forest closure terminates soundly: {g:?}");

        let closure: std::collections::HashSet<(u16, u16)> = arena
            .entries()
            .iter()
            .map(|d| (d.triple.subject, d.triple.object))
            .collect();
        assert_eq!(
            t.ancestor_pairs(),
            closure,
            "trie prefix-ancestry must equal the materialized closure EXACTLY"
        );
    }

    /// `is_ancestor_of` agrees EXACTLY with `ancestor_pairs` membership over a
    /// branching forest (and is strict + antisymmetric) — the operational-API
    /// pin behind the D-SRS-2 gate.
    #[test]
    fn is_ancestor_of_agrees_with_ancestor_pairs() {
        let edges = [(1, 2), (1, 3), (2, 4), (2, 5), (5, 6), (10, 11)];
        let t = FamilyTrie::build(&edges);
        let pairs = t.ancestor_pairs();
        let covered: Vec<u16> = (0..=11).filter(|&e| t.dn(e).is_some()).collect();
        for &a in &covered {
            for &z in &covered {
                let via_walk = t.is_ancestor_of(a, z);
                let via_set = pairs.contains(&(a, z));
                assert_eq!(via_walk, via_set, "disagreement on ({a},{z})");
                if a == z {
                    assert!(!via_walk, "strictness: no self-ancestry");
                }
                if via_walk {
                    assert!(!t.is_ancestor_of(z, a), "antisymmetry on ({a},{z})");
                }
            }
        }
        assert_eq!(t.pair_count(), pairs.len());
    }

    /// Star (one root, many children): depth-1 trie, pairs == edges — the
    /// shape where relocation buys nothing (the detector routes it to
    /// EdgeTable; the trie still answers correctly).
    #[test]
    fn star_is_flat_amortization() {
        let edges: Vec<(u16, u16)> = (1..=10).map(|c| (0, c)).collect();
        let t = FamilyTrie::build(&edges);
        assert_eq!(t.ancestor_pairs().len(), 10); // == edges: ratio 10/11 < 2
        assert_eq!(t.max_depth(), 1);
    }

    /// HHTL packability: depth ≤ 12 and fan-out ≤ 16 pack; a 17-child hub or a
    /// 13-deep line does not.
    #[test]
    fn hhtl_packable_respects_depth_and_fanout() {
        // 13-deep chain 0→1→…→13: entities at depth ≤ 12 pack, depth 13 not.
        let chain: Vec<(u16, u16)> = (0..13).map(|k| (k, k + 1)).collect();
        let t = FamilyTrie::build(&chain);
        assert_eq!(t.covered(), 14);
        assert_eq!(t.hhtl_packable(), 13); // depths 0..=12 pack; depth 13 does not

        // A 17-child hub: the children fail the fan-out bound, the root packs.
        let hub: Vec<(u16, u16)> = (1..=17).map(|c| (0, c)).collect();
        let t2 = FamilyTrie::build(&hub);
        assert_eq!(t2.hhtl_packable(), 1); // only the root
    }
}
