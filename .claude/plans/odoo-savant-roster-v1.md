# odoo-savant-roster-v1

> **Status:** PROPOSAL (the lance-graph side of the woa-rs Odoo savant delegation).
> **Confidence:** HIGH on the roster + dispatch surface (it rides the shipped `reasoning`/`nars`/`thinking` contract enums); MED on the `Reasoner` impls + new OGIT families; the Layer-2 alignment axioms are net-new.
> **Plan file:** `.claude/plans/odoo-savant-roster-v1.md`
> **Source of truth:** `.claude/odoo/SAVANTS.md` + the L1–L15 lane drafts (`.claude/odoo/L*.md`), imported by PR #413.
> **Predecessors:** PR #412 (odoo→FIBO/SKR ontology alignment + DOLCE classifier), `lance-graph-ontology`, `ada-rewrite-charter` (D1: business = OGIT-inherited sidecar).

## Scope

woa-rs harvested 25 Odoo **savants** — delegated reasoners where woa-rs keeps the deterministic guard (AXIS-A Rust) and delegates the ambiguous, evidence-weighted core (AXIS-B) to lance-graph's thinking surface through the BBB-allowed contract crates. Each savant is a dispatch **tuple**: OGIT family · `reasoning::ReasoningKind` · `nars::InferenceType` · `nars::SemiringChoice` · `thinking::StyleCluster`. The tuple fully determines runtime dispatch (`InferenceType::default_strategy() → QueryStrategy`).

This plan is the **lance-graph implementation** of the handover (SAVANTS.md §"lance-graph handover boundary"): (a) `Reasoner` impls per `ReasoningKind`, (b) two new OGIT families + style wiring, (c) Layer-2 alignment axioms for the `None`-family classes.

## Deliverables

| D-id | Scope | Crate | Status |
|---|---|---|---|
| **D-ODOO-1** | the 25-savant roster as data + dispatch tuple + lookups | `contract::savants` | ✅ **DONE this PR** (3 tests) |
| **D-ODOO-2** | `Reasoner` impls per `ReasoningKind` (CustomerCategory / PostingAnomaly / NextBestAction / InvoiceCompleteness / MailIntent + the 6 `Other` codes) — the AXIS-B experts | planner / a new reasoner crate | Queued |
| **D-ODOO-3** | two new OGIT families **`0x63 ProductCatalog`** + **`0x90 HRFoundation`** in `OgitFamilyTable` + inherited `StyleCluster` per family | `lance-graph-ontology` + `contract::build.rs` | Queued |
| **D-ODOO-4** | Layer-2 alignment axioms for the `None` classes (`stock.*`, `account.analytic.distribution.model`, `account.account.tag`) so the 11 `unaligned()` savants resolve a family | `lance-graph-ontology` `data/ontologies/odoo/alignment/` | Queued |
| **D-ODOO-5** | delegation call-site conformance: `ReasoningContext` + Arrow `EvidenceRef` schemas per savant; woa-rs↔lance-graph contract test | `contract` + conformance harness | Queued |

## Invariants (the delegation contract)

- **Suggestion-only, never an un-guarded write** (woa-rs Iron Rule 7, *verhaltens-bewahrend*) — the reasoner returns a truth-weighted conclusion; woa-rs applies it behind its AXIS-A guard.
- **Deterministic guard stays in woa-rs** (balance==0, residual, sign, prefix-match…); lance-graph only sees the ambiguous core.
- **BBB-allowed crates only** (`lance-graph-contract`, `-ontology`, `-callcenter`); no brain-crate in the customer binary (Iron Rule 1).
- The savant **tuple fully determines dispatch**; `SemiringChoice` selects evidence fusion (NarsTruth = NARS revision, the common case).
- **Business = OGIT-inherited sidecar** (charter D1): odoo classes inherit existing FIBO/SKR family slots (PR #412), they do not get a bespoke CAM family — `0x63`/`0x90` are the only *new* families and must be ratified.

## Open questions

- Ratify families `0x63 ProductCatalog` / `0x90 HRFoundation` (or fold into existing slots?).
- Where do the `Reasoner` impls live — `lance-graph-planner` (has NARS engine + MUL) or a dedicated reasoner crate? (AriGraph-circular-dep caveat applies; see CLAUDE.md p64 convergence note.)
- `ReasoningKind::Other(u32)` code registry — `contract::savants::other_kind` holds the 6 codes; promote to a named enum if the set stabilizes.

## Cross-ref

`.claude/odoo/SAVANTS.md` (roster + delegation contract), `.claude/odoo/L*.md` (per-lane porter specs), `contract::{savants, reasoning, nars, thinking}`, `lance-graph-ontology` (odoo alignment from PR #412), `ada-rewrite-charter.md` (business sidecar).
