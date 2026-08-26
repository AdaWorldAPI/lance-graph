## 2026-08-19 — E-THE-STRONG-HIERARCHY-EXISTS-AS-FIVE-DISCONNECTED-ISLANDS-1

**Status:** FINDING (ARC A′ five-sweep source audit, operator correction of the
weak "computed view" decode). Deliverable:
`docs/architecture/ARC-A2-STRONG-HIERARCHY-RECONCILIATION.md`.

**The finding:** every major mechanism of the operator's strong hierarchy model
is ALREADY IMPLEMENTED in this workspace, correctly and with tests — and none of
the implementations are connected to each other. The gap is not invention; it is
that no SHARED SHAPE exists, so each island grew its own vocabulary.

- Two independent rails, one leaf, no identity duplication →
  `rail_geometry::{RailAxis::Taxonomy,::Mereology,RailPath}` (shipped
  2026-08-13) with its own disable-test `the_pair_axes_are_two_separate_bytes`.
  Sole caller: a `ClassView::rail_carving` default. **But
  `NiblePath::from_guid_prefix_v3` FUSES both bytes into one route, has no
  per-axis constructor, and the two modules never cross-reference.**
- Shared ancestry walked once, not copied (the DN-tree) →
  `deepnsm-v2::ancestry::FamilyTrie`, whose `dn()` is literally the
  distinguished-name walk. Private `u16` id space, zero tie to
  `NodeGuid`/`NiblePath`, no crate depends on it.
- The intended end-state (`is_a` = prefix containment / zero storage; `part_of`
  = explicit edges) → already written down UNPROMPTED in `soa_bake/mod.rs`,
  self-marked "⚠ TYPE SCAFFOLDING, not a working bake".
- Second-order referencing (record → record of its own kind, NO homunculus) →
  `witness_fabric::resolve_chain`, real multi-hop with horizon+budget
  escalation. Bounded ±8, one locus dimension, facet flagged experimental.
- Grouping → stable addressable coordinate → `wikidata_hhtl` DOLCE basin,
  shipped and tested. **Type-A only; no Type-B (discovered) analogue exists
  anywhere in the workspace.**

**The naming failure is measured, not hypothetical:** the word "basin" already
carries FIVE distinct mechanisms (AriGraph discovered cluster ·
`Locus::BasinAnchor` pointer slot · `EpisodicEdges64` class-family ·
`NodeGuid` `family` tier · Wikidata/DOLCE category) — four Type-A, one Type-B,
sharing a word and nothing else. This is exactly the throwaway-naming failure
the operator's "reusable agnostic shape" ruling exists to prevent, already
present in the tree.

**The sharpest single instance of the gap:** `nars/meta_basin.rs` performs real
higher-order structural analysis today (basin clustering over causal
trajectories, outlier suggestion with evidence) and its own doc states
"Nothing here prunes, commits, or scores." Computed, then discarded, every
cycle. Meta-awareness is therefore PRESENT-BUT-UNMATERIALIZED, not future — a
different and more actionable condition than the ARC A draft claimed.

**Correctly absent, and worth recording as correct:** there is no reasoner in
lance-graph. The OWL hydrators intern IRIs and discard the triples;
`ontology_warrant.rs` names OGAR's `ogar-elk` as the external factfinder and is
structurally prevented from producing an entailment ("deliberately no method
that turns a `NarsTruth` back into an entailment"). The operator's
HHTL-exposes-structure / RO-decides-transfer boundary is already drawn in the
code — merely unwired. And NO live conflation of structural ancestry with
semantic inheritance was found; `ontology_warrant.rs` is evidence of the
opposite discipline, built after a measured incident where treating ontology
SILENCE as DISSENT inverted a finding (~50% apparent disagreement vs a true
99.8% agreement).

**Cross-refs:** `E-HIERARCHY-NODE-IS-ALGEBRA-NEVER-A-CROSSWALK-1` (the earlier,
WEAKER decode — this entry supersedes its generality: a computed projection is
one case, not the definition); `E-ARIGRAPH-IS-AN-ISLAND` (independently
reconfirmed still current); `E-FAMILY-NODE-IS-META-AWARENESS`.

