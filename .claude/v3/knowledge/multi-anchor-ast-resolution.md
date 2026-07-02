# Multi-Anchor AST Resolution — ruff + odoo landing on OGIT/OGAR/ClassViews

> READ BY: any session doing ruff-harvest / odoo-rs transcode / OGAR
> adapter or codebook work / OGIT ontology landing / ClassView codegen.
> Sibling doctrine: OGAR `docs/OGAR-TRANSPILE-SUBSTRATE.md` (pull-in/
> pull-back), `.claude/knowledge/core-first-transcode-doctrine.md`.

## Status: OPERATOR-STATED 2026-07-02 ([G] for the method's existence per operator; [H] per-mechanism until the ruff/odoo code is ground-truthed — board E-RUFF-ODOO-MULTI-ANCHOR-AST)

---

## The method

ruff + odoo together carry a **new AST resolution method**: a source
construct is not resolved from syntax alone — it is **triangulated from
multiple anchors simultaneously**, any of which can bind the resolution:

| Anchor | What it contributes |
|---|---|
| **Database layout** | the ORM/schema ground truth (Odoo models, e.g. `account.move`) — the state shape the construct actually touches |
| **Duplication** | duplicated routes/implementations cluster; N copies doing one thing VOTE for one canonical concept — duplication is a resolution SIGNAL, not noise |
| **Target adapters** | the OGAR classid-keyed adapter surface the construct should land on (identity = classid; state = SoA value tenants; invocation = UnifiedStep) |
| **Target ontology (OGIT)** | the ontology node to land on — label-inheritance chain, schema, assoc |
| **ClassView stacking** | constructors resolve through STACKED ClassViews (classid → ClassView chain), so codegen REUSES existing constructors instead of emitting new ones |
| **Fuzzy ontology match** | for the tail: semantic match against the target ontology when exact anchors are missing — e.g. *"something that constructs an invoice from account, in different duplicated routes"* resolves to the one canonical invoice-construction concept |

## Why this matters for V3

- **Duplication → entropy milestone feedstock.** The duplication anchor
  is a mechanical detector for N→1 collapses: N duplicated routes → 1
  canonical concept (classid) + N ClassView skins. This is the same shape
  as the classid flip and the DTO dedups — the AST method finds these
  collapses in CONSUMER code automatically.
- **Landing surfaces are V3-native.** What a resolution lands ON is
  exactly the V3 address stack: OGIT node (ontology), OGAR classid
  (identity), ClassView (render/constructor), SoA tenant lanes (state,
  per `soa_layout/le-contract.md`), rails (part_of:is_a / memberof) for
  the resolved relations.
- **Constructor reuse = slot purity's sibling.** Just as labels/positions
  come from the ClassView (never a payload slot), constructors come from
  the stacked ClassView chain (never re-emitted per duplicate). Codegen
  that emits a fresh constructor for a duplicated route is the same
  defect class as a label slot in a facet.
- **Fuzzy matches are the oracle tail.** Exact anchors are the
  deterministic path; the fuzzy ontology match is the <25%-tail resolver
  — same shape as the template/Rig-oracle split: deterministic first,
  semantic match for the remainder, and a successful fuzzy resolution
  should be MINTED (into the codebook/ontology) so the next encounter is
  exact.

## Guardrails

1. A fuzzy match NEVER lands silently: it produces a mint/mapping entry
   (reviewable) — the next resolution of the same construct must be exact.
2. Duplication votes pick ONE canonical target; the duplicates become
   ClassView-differentiated skins or adapters, never N parallel concepts.
3. Landing must target the OGAR Core surfaces (classid / tenants /
   EdgeBlock / ClassView / UnifiedStep) — never a parallel object model
   (core-first doctrine).
4. Mirror-to-OGAR: this doc is the lance-graph-side record; the OGAR
   DISCOVERY-MAP D-entry goes through OGAR's own 5+3 hardening gate
   (queued — do not paste into OGAR canon without that pass).

Cross-ref: board `E-RUFF-ODOO-MULTI-ANCHOR-AST`, `le-contract.md` §2–3,
`compiled-templates.md` (deterministic-first + oracle tail),
OGAR `docs/OGAR-AS-IR.md` (compiler phases; this method is the front-end
+ linker phases gaining multi-anchor symbol resolution).
