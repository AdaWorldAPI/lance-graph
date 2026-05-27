# odoo-savant-reasoners-v1 — lance-graph side of the Odoo richness harvest

> **Status:** PROPOSAL. Picks up the explicit cross-repo handover boundary
> declared in `lance-graph/.claude/odoo/SAVANTS.md` §"lance-graph handover
> boundary": the woa-rs session defined the 25-Savant roster + delegation
> tuples + call sites + evidence schemas (all contract-level, BBB-allowed);
> the **lance-graph side** must now implement (a) the `Reasoner` impls, (b)
> the two new OGIT families + Layer-2 alignment axioms for the `None`
> classes, (c) the `StyleCluster` wiring.
>
> **Confidence:** HIGH on (b)/(c) — concrete additive extensions of the
> existing `odoo_alignment.rs` seed table + alignment TTLs. MED on (a) — the
> `Reasoner` dispatch shape is an architectural decision (one impl per
> `ReasoningKind` vs a savant-config registry) that this plan pins but which
> needs a review pass before the larger build.
>
> **Predecessors:** PR #412 (odoo hydrator + `dolce_odoo` classifier + ODOO
> slot 50), PR #413 (briefing pack import). Reads on:
> `lance-graph-contract::reasoning` (Reasoner / ReasoningKind / ReasoningContext),
> `lance-graph-callcenter::{family_table, odoo_alignment, unified_bridge}`,
> `lance-graph-contract::thinking::StyleCluster`.
>
> **Anchored iron rules:** I-VSA-IDENTITIES (savant = Layer-2 role catalogue,
> content in tables not bundles), the AGI-as-glove doctrine (savant dispatch
> rides `MetaColumn`/`EdgeColumn`, no new service), board-hygiene.

## Scope

Three deliverable groups, matching SAVANTS.md (a)/(b)/(c):

### Group B — two new OGIT families + Layer-2 alignment axioms (lowest risk, first)

The roster needs homes for classes that resolve to `None` today:

- **`0x63 ProductCatalog`** (basin: product catalogue + pricelist + UoM;
  default style **Analytical**). Lands `product.template`,
  `product.pricelist`, `product.pricelist.item`, `uom.uom`,
  `product.category`. Currently `product.*` inherits `0x61 BillingCore`
  slot 1 via the prefix fallback — ProductCatalog promotes the
  catalogue/pricing concepts to their own basin so L8's
  `PricelistAssignmentAgent` has a real family instead of `None`.
- **`0x90 HRFoundation`** (basin: employee / org; default style
  **Empathic**). Lands `hr.employee`→`vcard:Individual`,
  `hr.department`→`org:OrganizationalUnit`, `hr.job`, `hr.contract` (base
  only — payroll engine is Enterprise/absent, flagged). All `hr.*`
  resolve `None` today.
- **Layer-2 alignment axioms** for the remaining `None` classes that do
  NOT get a new family (they stay cross-cutting): `stock.*` (stock.move,
  stock.rule, stock.warehouse.orderpoint), `account.analytic.distribution.model`,
  `account.account.tag`. These get `owl:equivalentClass` / `rdfs:subClassOf`
  rows in `data/ontologies/odoo/alignment/` pointing at existing pivots
  (e.g. `stock.move`→`schema:MoveAction`-style, analytic→`fibo:CostCenter`)
  so `resolve_odoo` can land them on an existing family rather than minting
  one. Where no honest pivot exists, the class stays `None` and the plan
  records WHY (genuinely cross-cutting, needs a runtime SPO-G edge not a
  static family).

### Group C — StyleCluster wiring

`FamilyEntry` carries `dolce_marker` + `owl_characteristics` but NOT the
default `StyleCluster`. SAVANTS.md inherits the style from the family. Add a
`default_style: StyleCluster` field to `FamilyEntry` (or a sidecar
`family_style(OgitFamily) -> StyleCluster` map if adding a field churns too
many call sites — decide in review). Seed:
0x60→Direct, 0x61→Analytical, 0x62→Analytical, 0x63→Analytical,
0x80→Empathic, 0x81→Direct, 0x90→Empathic.

### Group A — Reasoner impls (largest, scoped here, built in a follow-up PR)

The 25 savants collapse onto **5 `ReasoningKind` discriminants** ×
**5 `InferenceType` strategies**. Decision to pin: implement **one
`Reasoner` per `ReasoningKind`** (CustomerCategory, PostingAnomaly,
NextBestAction, InvoiceCompleteness, Other) that dispatches on
`ReasoningContext.evidence` + the resolved family's style, rather than 25
separate impls. Each `reason()` selects its `QueryStrategy` from
`InferenceType::default_strategy()` (already mapped:
Deduction→CamExact, Induction→CamWide, Abduction→DnTreeFull,
Revision→BundleInto, Synthesis→BundleAcross) and fuses evidence via the
savant's `SemiringChoice` (NarsTruth = the common case). Conclusion type:
a truth-weighted `SavantConclusion { suggestion, confidence: NarsTruth,
rationale }` that woa-rs applies as a **suggestion only**, never an
un-guarded write (Iron Rule 7, verhaltens-bewahrend). Home crate:
`lance-graph-callcenter` (BBB-allowed, already owns `odoo_alignment.rs`).

## Deliverables

- **D-ODOO-SAV-1** (Group B): `0x63 ProductCatalog` + `0x90 HRFoundation`
  family constants + seed rows in `odoo_alignment.rs`; `data/family_registry.ttl`
  family declarations; product.*/hr.* seed rows. Tests: resolve
  product.template→0x63, hr.employee→0x90, end-to-end against live table.
- **D-ODOO-SAV-2** (Group B): Layer-2 alignment axioms TTL for stock.*,
  analytic.distribution.model, account.account.tag — land on an existing
  pivot where honest, else documented `None`-with-rationale. Tests: the
  newly-aligned classes resolve; the genuinely-cross-cutting ones still
  return `None` with a recorded reason.
- **D-ODOO-SAV-3** (Group C): `StyleCluster` per family. Field-or-sidecar
  decision in review; seed the 7 families; test each family→cluster.
- **D-ODOO-SAV-4** (Group A): `SavantConclusion` + 5 `Reasoner` impls in
  `lance-graph-callcenter`, dispatching on `ReasoningKind` + evidence +
  family style. Per-impl tests with synthetic `EvidenceRef` batches.
  **Gated on a review pass of the dispatch shape.** Its own PR.

## Execution

D-ODOO-SAV-1/2/3 are additive + low-risk → ship together in the first PR
(this session). D-ODOO-SAV-4 (Reasoner impls) is the architectural piece →
scoped here, built in a follow-up PR after a `/code-review` pass on the
dispatch shape. Board-hygiene: this plan + INTEGRATION_PLANS prepend land
in the same commit as D-ODOO-SAV-1.

## Invariants

- Option B holds: odoo classes INHERIT existing slots via OWL pivot; new
  families 0x63/0x90 are genuine new basins (product catalogue, HR), not
  per-class mints. `None` stays `None` when no honest pivot exists.
- Public OWL/RDF sources stay pristine — alignment axioms are NEW TTL in
  `data/ontologies/odoo/alignment/`, not edits to upstream vocabs.
- Savant = Layer-2 role catalogue (I-VSA-IDENTITIES): identity in the
  family table, content (evidence/rules) in Arrow/SPO, never bundled.
- Reasoner output is a suggestion; the deterministic guard stays in woa-rs
  (verhaltens-bewahrend). lance-graph implements the ambiguous core only.
- No brain-crate in the customer binary (Iron Rule 1): impls live in
  `lance-graph-callcenter` behind the contract `Reasoner` trait.
