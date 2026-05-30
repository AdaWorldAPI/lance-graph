// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! OWL/DOLCE skeleton projection — the discovery **output** side of D-ARM-14.
//!
//! aerial discovers `(entity-pattern → class)` rules over the splat codebook
//! ([`crate::aerial::TopKDistance`]); this module renders them as the
//! `wikidata-hhtl-load.md` **skeleton** — `rdf:type` (P31) / `rdfs:subClassOf`
//! (P279) triples — with **DOLCE's four top facets** as the HHTL axis template
//! (`ogit-owl-dolce-ontology-compartments.md`: "DOLCE defines WHICH axes;
//! Wikidata fills WHAT occurs"). The discovered rules are candidate skeleton
//! edges; the D-ARM-7 Jirak floor decides which are significant enough to
//! persist, and codebook-HHTL is the 16ⁿ bucket router downstream.

use crate::rule::Item;
use crate::translator::FeedProjector;

/// DOLCE's four top categories — the "clean top facets" of
/// `ogit-owl-dolce-ontology-compartments.md`, the HHTL axis template. Each owns
/// a basin nibble (`0x0..=0x3`) and a stable IRI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DolceCategory {
    /// Objects that are wholly present at each moment (DOLCE *Endurant*).
    Endurant,
    /// Things that happen / unfold in time (DOLCE *Perdurant*).
    Perdurant,
    /// Qualities inhering in entities (DOLCE *Quality*).
    Quality,
    /// Abstract entities outside space-time (DOLCE *Abstract*).
    Abstract,
}

impl DolceCategory {
    /// The four facets in basin order.
    pub const ALL: [DolceCategory; 4] = [
        DolceCategory::Endurant,
        DolceCategory::Perdurant,
        DolceCategory::Quality,
        DolceCategory::Abstract,
    ];

    /// The basin nibble (`0x0..=0x3`) this facet routes to in the HHTL tree.
    #[must_use]
    pub fn basin(self) -> u8 {
        self as u8
    }

    /// The stable prefixed IRI for this facet.
    #[must_use]
    pub fn iri(self) -> &'static str {
        match self {
            DolceCategory::Endurant => "dolce:Endurant",
            DolceCategory::Perdurant => "dolce:Perdurant",
            DolceCategory::Quality => "dolce:Quality",
            DolceCategory::Abstract => "dolce:Abstract",
        }
    }

    /// Recover a facet from its basin nibble.
    #[must_use]
    pub fn from_basin(n: u8) -> Option<DolceCategory> {
        Self::ALL.get(n as usize).copied()
    }
}

/// A [`FeedProjector`] that renders discovered rules as OWL/DOLCE skeleton SPO.
///
/// The **consequent** is the class axis: its category indexes [`Self::class_iris`]
/// (e.g. the four [`DolceCategory`] IRIs, or a Wikidata class set). The
/// **antecedent** is the entity/feature pattern, rendered under [`Self::namespace`].
#[derive(Debug, Clone)]
pub struct OntologyProjector {
    /// IRI prefix for the discovered entity/feature subject (e.g. `"wd:"`).
    pub namespace: String,
    /// The skeleton relation IRI (`"rdfs:subClassOf"` or `"rdf:type"`).
    pub predicate: String,
    /// Class IRI per consequent category (the skeleton classes).
    pub class_iris: Vec<String>,
}

impl OntologyProjector {
    /// A `rdfs:subClassOf` (P279) projector over an explicit class IRI set.
    #[must_use]
    pub fn subclass_of(namespace: impl Into<String>, class_iris: Vec<String>) -> Self {
        Self {
            namespace: namespace.into(),
            predicate: "rdfs:subClassOf".to_string(),
            class_iris,
        }
    }

    /// A `rdf:type` (P31) projector over an explicit class IRI set.
    #[must_use]
    pub fn instance_of(namespace: impl Into<String>, class_iris: Vec<String>) -> Self {
        Self {
            namespace: namespace.into(),
            predicate: "rdf:type".to_string(),
            class_iris,
        }
    }

    /// A `rdfs:subClassOf` projector whose consequent categories index the four
    /// [`DolceCategory`] facets — the DOLCE axis-template skeleton.
    #[must_use]
    pub fn dolce_subclass(namespace: impl Into<String>) -> Self {
        Self::subclass_of(
            namespace,
            DolceCategory::ALL.iter().map(|c| c.iri().to_string()).collect(),
        )
    }
}

fn render_pattern(namespace: &str, items: &[Item]) -> String {
    let mut parts: Vec<Item> = items.to_vec();
    parts.sort();
    let body = parts
        .iter()
        .map(|it| format!("f{}_{}", it.feature, it.category))
        .collect::<Vec<_>>()
        .join("&");
    format!("{namespace}{body}")
}

impl FeedProjector for OntologyProjector {
    fn subject(&self, antecedent: &[Item]) -> String {
        render_pattern(&self.namespace, antecedent)
    }

    fn predicate(&self) -> String {
        self.predicate.clone()
    }

    fn object(&self, consequent: &[Item]) -> String {
        // The consequent is the class axis; its category indexes class_iris.
        consequent
            .first()
            .and_then(|it| self.class_iris.get(it.category as usize))
            .cloned()
            .unwrap_or_else(|| format!("{}class:unmapped", self.namespace))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aerial::{extract_rules, ExtractParams, TopKDistance};
    use crate::encode::{Dataset, FeatureSpec};
    use crate::translator::{CandidateTriple, NARS_PERSONALITY_K};

    #[test]
    fn dolce_facets_round_trip_through_basins() {
        for (n, c) in DolceCategory::ALL.iter().enumerate() {
            assert_eq!(c.basin() as usize, n);
            assert_eq!(DolceCategory::from_basin(n as u8), Some(*c));
        }
        assert_eq!(DolceCategory::from_basin(4), None);
        assert_eq!(DolceCategory::Quality.iri(), "dolce:Quality");
    }

    /// End-to-end: a splat codebook makes each occupation near its DOLCE class;
    /// aerial discovers the `occupation → class` skeleton edge; the projector
    /// renders it as `wd:… rdfs:subClassOf dolce:…`.
    #[test]
    fn discovers_and_projects_dolce_skeleton() {
        // feature 0 = occupation (3), feature 1 = DOLCE class (4 facets).
        let spec = FeatureSpec::new(vec![3, 4]);
        // Plant occupation k ⇒ class k for k ∈ {0,1,2} (Abstract unused).
        let rows: Vec<Vec<u32>> = (0..300).map(|i| { let k = (i % 3) as u32; vec![k, k] }).collect();
        let data = Dataset::new(spec.clone(), rows);

        // Splat top-k: occupation k is near DOLCE class k (distance 1), else far.
        let edges: Vec<(Item, Item, u32)> = (0..3)
            .map(|k| (Item::new(0, k), Item::new(1, k), 1))
            .collect();
        let oracle = TopKDistance::new(spec, 99, &edges);

        let params = ExtractParams {
            theta: 2, // only the codebook-near class is a candidate consequent
            max_antecedent: 1,
            min_support_ppm: 50_000,     // 5%
            min_confidence_ppm: 700_000, // 70%
        };
        let rules = extract_rules(&oracle, &data, &params);

        // The planted occupation0 ⇒ Endurant(class0) skeleton edge is recovered.
        let rule = rules
            .iter()
            .find(|r| r.antecedent == vec![Item::new(0, 0)] && r.consequent == vec![Item::new(1, 0)])
            .expect("occupation0 → DOLCE class0 not discovered");

        let proj = OntologyProjector::dolce_subclass("wd:");
        let triple = CandidateTriple::from_rule(rule, &proj, NARS_PERSONALITY_K);
        assert_eq!(triple.s, "wd:f0_0");
        assert_eq!(triple.p, "rdfs:subClassOf");
        assert_eq!(triple.o, "dolce:Endurant");
        assert!(triple.c > 0.9, "confident skeleton edge");

        // Abstract (class 3) was never instantiated → never a discovered class.
        assert!(
            !rules.iter().any(|r| r.consequent == vec![Item::new(1, 3)]),
            "unused DOLCE facet must not be invented"
        );
    }
}
