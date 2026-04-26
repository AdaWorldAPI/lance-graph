//! Factorization-aware schema (from Kuzudb).
//!
//! The schema tracks FactorizationGroups — each group holds a set of expressions,
//! a flat/unflat flag, and a cardinality multiplier. The invariant: at most one
//! group is unflat at any point. This is the IR for factorized intermediate
//! representations — avoiding exponential blowup in multi-hop graph traversals.

#[allow(unused_imports)] // intended for expression-in-schema wiring
use super::ExprNode;

/// Schema: the plan's type system for factorization.
#[derive(Debug, Clone, Default)]
pub struct Schema {
    /// Factorization groups.
    pub groups: Vec<FactorizationGroup>,
}

/// A factorization group: a set of expressions that share the same cardinality multiplier.
#[derive(Debug, Clone)]
pub struct FactorizationGroup {
    /// Expressions in this group.
    pub expressions: Vec<String>,
    /// Whether this group is flat (materialized) or unflat (factorized).
    pub flat: bool,
    /// Whether this group is in a single-state (one value per pipeline).
    pub single_state: bool,
    /// Cardinality multiplier for this group.
    pub cardinality_multiplier: f64,
}

impl Schema {
    pub fn new() -> Self {
        Self { groups: Vec::new() }
    }

    /// Add a new group to the schema.
    pub fn add_group(&mut self, group: FactorizationGroup) {
        self.groups.push(group);
    }

    /// Get the single unflat group, if any (invariant: at most one).
    pub fn unflat_group(&self) -> Option<(usize, &FactorizationGroup)> {
        self.groups.iter().enumerate()
            .find(|(_, g)| !g.flat && !g.single_state)
    }

    /// Check the factorization invariant: at most one unflat group.
    pub fn is_valid(&self) -> bool {
        self.groups.iter().filter(|g| !g.flat && !g.single_state).count() <= 1
    }

    /// Encode the factorization state as a bitmask (for DP table differentiation).
    /// Each bit represents whether a group is flat (1) or unflat (0).
    pub fn factorization_encoding(&self) -> u64 {
        let mut encoding = 0u64;
        for (i, group) in self.groups.iter().enumerate() {
            if group.flat {
                encoding |= 1u64 << i;
            }
        }
        encoding
    }

    /// Merge two schemas (for join result).
    pub fn merge(&self, other: &Schema) -> Schema {
        let mut merged = self.clone();
        merged.groups.extend(other.groups.iter().cloned());
        merged
    }

    /// Flatten a specific group by position.
    pub fn flatten_group(&mut self, pos: usize) {
        if pos < self.groups.len() {
            self.groups[pos].flat = true;
        }
    }

    /// Total cardinality multiplier across all groups.
    pub fn total_cardinality(&self) -> f64 {
        self.groups.iter()
            .map(|g| g.cardinality_multiplier)
            .product()
    }
}

impl FactorizationGroup {
    pub fn new_flat(expressions: Vec<String>, cardinality: f64) -> Self {
        Self {
            expressions,
            flat: true,
            single_state: false,
            cardinality_multiplier: cardinality,
        }
    }

    pub fn new_unflat(expressions: Vec<String>, cardinality: f64) -> Self {
        Self {
            expressions,
            flat: false,
            single_state: false,
            cardinality_multiplier: cardinality,
        }
    }
}
