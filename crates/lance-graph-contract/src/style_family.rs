//! The 12 abstract style FAMILIES for orchestration.
//!
//! **Families vs runbooks (E-STYLE-FAMILY-VS-RUNBOOK-1):** the 12 families
//! are the coarse orchestration space — which KIND of thinking a cycle
//! runs. The 36 [`ThinkingStyle`](crate::thinking::ThinkingStyle) variants
//! are literal NARS runbooks — concrete, executable inference recipes that
//! seed the rung ladder and serve as the replayable chaining unit for
//! graph-flow orchestration. Families orchestrate; runbooks execute.
//! The two spaces are related by [`StyleFamily::default_runbook`] and
//! [`ThinkingStyle::family`], never merged.
//!
//! This module is the M9 dedup survivor for the 12-space: the planner,
//! thinking-engine, and driver 12-style definitions are re-exports of (or
//! keyed by) this type. Before M9, FOUR mutually divergent 12→36 tables
//! existed in the tree (planner `planner_style_to_contract`, driver
//! `ord_to_thinking_style`, contract `parse_style_name`, and the
//! `THINKING_RECONCILIATION.md` exemplars) — no two fully agreed. The
//! canonical table below replaces all four; the arm-change history lives
//! in `.claude/plans/dtsc1-thinkingstyle-dedup-spec-v1.md` §3 S1.

use crate::thinking::ThinkingStyle;

/// The 12 abstract style families for orchestration (coarse dispatch).
///
/// Ordinal order is FROZEN to the driver's `UNIFIED_STYLES` order
/// (`Deliberate = 0` … `Metacognitive = 11`); the discriminants are
/// load-bearing (const tables and `as u8` casts key off them) and are
/// pinned by tests. Names match the deepnsm YAML style-card names.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StyleFamily {
    /// Slow, methodical System-2 processing.
    #[default]
    Deliberate = 0,
    /// Decomposition and precise analysis.
    Analytical = 1,
    /// Depth-first narrowing toward one answer.
    Convergent = 2,
    /// Ordered, hierarchical processing.
    Systematic = 3,
    /// Generative, transformative processing.
    Creative = 4,
    /// Breadth-first opening of alternatives.
    Divergent = 5,
    /// Spiral search of unknown territory.
    Exploratory = 6,
    /// Narrow, compressed attention.
    Focused = 7,
    /// Wide, soft attention.
    Diffuse = 8,
    /// Edge-of-field attention (inversion, weak signals).
    Peripheral = 9,
    /// Fast System-1 resonance.
    Intuitive = 10,
    /// Thinking about the thinking.
    Metacognitive = 11,
}

impl StyleFamily {
    /// All 12 families in frozen ordinal order.
    pub const ALL: [StyleFamily; 12] = [
        Self::Deliberate,
        Self::Analytical,
        Self::Convergent,
        Self::Systematic,
        Self::Creative,
        Self::Divergent,
        Self::Exploratory,
        Self::Focused,
        Self::Diffuse,
        Self::Peripheral,
        Self::Intuitive,
        Self::Metacognitive,
    ];

    /// Lower-case family name — matches the deepnsm YAML style-card names.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Deliberate => "deliberate",
            Self::Analytical => "analytical",
            Self::Convergent => "convergent",
            Self::Systematic => "systematic",
            Self::Creative => "creative",
            Self::Divergent => "divergent",
            Self::Exploratory => "exploratory",
            Self::Focused => "focused",
            Self::Diffuse => "diffuse",
            Self::Peripheral => "peripheral",
            Self::Intuitive => "intuitive",
            Self::Metacognitive => "metacognitive",
        }
    }

    /// Parse a family from its lower-case name (trims + case-folds).
    pub fn from_name(s: &str) -> Option<Self> {
        let lower = s.trim().to_ascii_lowercase();
        Self::ALL.into_iter().find(|f| f.name() == lower)
    }

    /// Family for a frozen ordinal (0..=11).
    pub fn from_ordinal(ord: u8) -> Option<Self> {
        Self::ALL.get(ord as usize).copied()
    }

    /// The representative NARS runbook for this family — THE canonical
    /// 12→36 mapping (replaces the four divergent pre-M9 tables).
    ///
    /// Arms were decided exact-runbook-name-first, reconciliation-doc
    /// collapse-exemplar otherwise; `f.default_runbook().family() == f`
    /// holds for every family (pinned by test).
    pub fn default_runbook(&self) -> ThinkingStyle {
        match self {
            Self::Deliberate => ThinkingStyle::Methodical,
            Self::Analytical => ThinkingStyle::Analytical,
            Self::Convergent => ThinkingStyle::Logical,
            Self::Systematic => ThinkingStyle::Systematic,
            Self::Creative => ThinkingStyle::Creative,
            Self::Divergent => ThinkingStyle::Imaginative,
            Self::Exploratory => ThinkingStyle::Exploratory,
            Self::Focused => ThinkingStyle::Precise,
            Self::Diffuse => ThinkingStyle::Gentle,
            Self::Peripheral => ThinkingStyle::Speculative,
            Self::Intuitive => ThinkingStyle::Poetic,
            Self::Metacognitive => ThinkingStyle::Metacognitive,
        }
    }
}

impl core::fmt::Display for StyleFamily {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl ThinkingStyle {
    /// The orchestration family this runbook belongs to — THE canonical
    /// 36→12 mapping (total).
    ///
    /// Each non-anchor runbook joins the family of an anchor in its own
    /// 36-cluster; the Direct cluster (no anchor) splits by semantics
    /// (focus vs convergence). Every family's default runbook maps back
    /// to its family (round-trip pinned by test).
    pub fn family(&self) -> StyleFamily {
        match self {
            // Analytical cluster
            Self::Logical => StyleFamily::Convergent,
            Self::Analytical | Self::Critical => StyleFamily::Analytical,
            Self::Systematic => StyleFamily::Systematic,
            Self::Methodical => StyleFamily::Deliberate,
            Self::Precise => StyleFamily::Focused,
            // Creative cluster
            Self::Creative | Self::Artistic | Self::Playful => StyleFamily::Creative,
            Self::Imaginative | Self::Innovative => StyleFamily::Divergent,
            Self::Poetic => StyleFamily::Intuitive,
            // Empathic cluster — soft, wide attention
            Self::Empathetic
            | Self::Compassionate
            | Self::Supportive
            | Self::Nurturing
            | Self::Gentle
            | Self::Warm => StyleFamily::Diffuse,
            // Direct cluster — focus vs convergence split
            Self::Direct | Self::Concise | Self::Blunt | Self::Frank => StyleFamily::Focused,
            Self::Efficient | Self::Pragmatic => StyleFamily::Convergent,
            // Exploratory cluster
            Self::Curious | Self::Exploratory | Self::Questioning | Self::Investigative => {
                StyleFamily::Exploratory
            }
            Self::Speculative | Self::Philosophical => StyleFamily::Peripheral,
            // Meta cluster
            Self::Reflective
            | Self::Contemplative
            | Self::Metacognitive
            | Self::Wise
            | Self::Transcendent
            | Self::Sovereign => StyleFamily::Metacognitive,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// G2: every family's default runbook maps back to its family.
    #[test]
    fn default_runbook_round_trips_to_its_family() {
        for f in StyleFamily::ALL {
            assert_eq!(
                f.default_runbook().family(),
                f,
                "round-trip broke for {f:?}"
            );
        }
    }

    /// G2: family() is total over all 36 runbooks (exhaustive match makes
    /// this a compile-time fact; the test documents it dynamically).
    #[test]
    fn family_is_total_over_all_36_runbooks() {
        for rb in ThinkingStyle::ALL {
            let f = rb.family();
            assert!(StyleFamily::ALL.contains(&f));
        }
    }

    /// G7: the canonical 12-arm default_runbook table, pinned literally.
    #[test]
    fn canonical_default_runbook_table_pinned() {
        use StyleFamily as F;
        use ThinkingStyle as T;
        let pins = [
            (F::Deliberate, T::Methodical),
            (F::Analytical, T::Analytical),
            (F::Convergent, T::Logical),
            (F::Systematic, T::Systematic),
            (F::Creative, T::Creative),
            (F::Divergent, T::Imaginative),
            (F::Exploratory, T::Exploratory),
            (F::Focused, T::Precise),
            (F::Diffuse, T::Gentle),
            (F::Peripheral, T::Speculative),
            (F::Intuitive, T::Poetic),
            (F::Metacognitive, T::Metacognitive),
        ];
        for (f, rb) in pins {
            assert_eq!(f.default_runbook(), rb);
        }
    }

    /// G7: frozen discriminants (const tables and `as u8` casts key off
    /// these — e.g. thinking-engine's `config.style as u8`).
    #[test]
    fn discriminants_pinned() {
        assert_eq!(StyleFamily::Deliberate as u8, 0);
        assert_eq!(StyleFamily::Analytical as u8, 1);
        assert_eq!(StyleFamily::Convergent as u8, 2);
        assert_eq!(StyleFamily::Systematic as u8, 3);
        assert_eq!(StyleFamily::Creative as u8, 4);
        assert_eq!(StyleFamily::Divergent as u8, 5);
        assert_eq!(StyleFamily::Exploratory as u8, 6);
        assert_eq!(StyleFamily::Focused as u8, 7);
        assert_eq!(StyleFamily::Diffuse as u8, 8);
        assert_eq!(StyleFamily::Peripheral as u8, 9);
        assert_eq!(StyleFamily::Intuitive as u8, 10);
        assert_eq!(StyleFamily::Metacognitive as u8, 11);
    }

    /// The 12 names, pinned literally (== deepnsm YAML card names).
    #[test]
    fn family_names_pinned() {
        let names: Vec<&str> = StyleFamily::ALL.iter().map(|f| f.name()).collect();
        assert_eq!(
            names,
            [
                "deliberate",
                "analytical",
                "convergent",
                "systematic",
                "creative",
                "divergent",
                "exploratory",
                "focused",
                "diffuse",
                "peripheral",
                "intuitive",
                "metacognitive",
            ]
        );
    }

    /// Ordinal and name round-trips.
    #[test]
    fn ordinal_and_name_round_trips() {
        for (i, f) in StyleFamily::ALL.into_iter().enumerate() {
            assert_eq!(StyleFamily::from_ordinal(i as u8), Some(f));
            assert_eq!(StyleFamily::from_name(f.name()), Some(f));
            assert_eq!(
                StyleFamily::from_name(&f.name().to_ascii_uppercase()),
                Some(f)
            );
        }
        assert_eq!(StyleFamily::from_ordinal(12), None);
        assert_eq!(StyleFamily::from_name("empathetic"), None); // a runbook name, not a family
    }
}
