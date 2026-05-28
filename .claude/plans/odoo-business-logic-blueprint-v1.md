# odoo-business-logic-blueprint-v1 — typed Odoo entity DTOs as the substrate for the OGIT → OWL → DOLCE → FIBU/FIBO normalization + JITson / recipe codegen

> **Status:** PROPOSAL. **PREREQUISITE for `odoo-savant-reasoners-v2` Group F**
> (per `E-SAVANT-COMPOSITION-1`): v2's `SavantPattern` consts compose *over
> normalized typed DTOs*; without this blueprint they would be ad-hoc
> interpretations of L-doc prose. This plan establishes the missing layer
> between Odoo prose curation and the agnostic shader substrate.
>
> **Confidence:** HIGH on the structure (typed DTOs are the missing inheritance-chain
> input; today every OGIT/OWL/DOLCE/FIBU layer string-keys against `model_name`).
> MED on per-lane density sizing (15 lanes × ~5-30 entities each = ~150-450
> const declarations is an estimate; actual count depends on lane density).
> HIGH on the JITson wiring path (the infrastructure exists at
> `lance-graph-contract::jit` + `thinking-engine` Cranelift integration —
> just not yet pointed at recipes/savants).
>
> **Predecessors:** PR #411 (33-TSV atoms + 34 `Tactic` kernels + `jit`
> infrastructure), PR #412 (DOLCE classifier + `dolce_odoo`), PR #414
> (`D-ODOO-SAV-1/2/3` OGIT families + Layer-2 axioms + style wiring),
> PR #416 (FIBU/FIBO + odoo savant roster `contract::savants`), PR #407/408
> (bO-* OWL hydrators), PR #418 (mailbox-owned SoA; `E-BATON-1`), PR #419
> (25 AXIS-B evidence contracts as prose), PR #420 (v1 reasoners — deprecated
> in v2). Driver epiphany: `E-SAVANT-COMPOSITION-1` (2026-05-28).
>
> **Anchored iron rules:** `I-VSA-IDENTITIES` (typed Layer-2 catalogues —
> identity in const data, content in tables), AGI-as-glove (new column not
> new layer — typed DTOs ARE the column the inheritance chain reads from),
> "consult before guess" (the L-doc curation IS the curated surface; project
> it into typed structure, don't re-derive).

## The diagnosis

lance-graph has shipped most of the cognitive substrate:
- OGIT families + Layer-2 axioms (PR #414/416), OWL hydrators (PR #407/408),
  DOLCE classifier (PR #412), FIBU/FIBO alignment (PR #416)
- 33-TSV atoms + 34 `Tactic` kernels + `jit::{JitCompiler, StyleRegistry, KernelHandle}` (PR #411)
- `CausalEdge64` (`causal-edge` v0.2.0) + `cognitive-shader-driver`
- 25-savant roster in `contract::savants` (PR #416)
- 15 lane drafts (`.claude/odoo/L*.md`, PR #413) + 25 AXIS-B evidence
  contracts (`.claude/odoo/savants/*.md`, PR #419) — **as prose**

**Missing layer**: typed Odoo entity DTOs that the inheritance chain
(Odoo → OGIT → OWL → DOLCE → FIBU/FIBO) operates on. Today every
downstream layer string-keys against `model_name`. There is no shared
typed representation of "what an Odoo entity *is*" — its fields, methods,
decorators, state machine, constraints. The user-named pipeline:

```
Odoo source (exact = ground truth)
  → typed Odoo entity DTOs        ← THIS PLAN
  → normalize: Odoo → OGIT → OWL → DOLCE → FIBU/FIBO inheritance chain
  → recipes (34 Tactic kernels) + JITson / Cranelift codegen
  → DTO-ish NARS atoms (33-TSV) — low-entropy typed surface
  → cognitive-shader-driver fans across the SoA at 10000×10000
  → CausalEdge64 emissions in EdgeColumn = the conclusions
```

## Scope (decisions ratified 2026-05-28)

- **Source**: Both passes — L-docs first as the savant-relevant curated
  filter; Odoo source extraction follows as the exhaustive backing.
- **Lane coverage**: All 15 lanes (L1–L15) in v1. Complete typed blueprint
  of the Odoo domain, not a savant-rich subset.

## The typed DTO surface (agnostic primitives)

New module: **`lance-graph-ontology::odoo_blueprint`** (sits next to the
existing odoo hydrator + `dolce_odoo`). Zero-dep beyond what ontology
already pulls, all const data, no serde:

```rust
pub struct OdooEntity {
    pub model_name: &'static str,         // "account.fiscal.position"
    pub description: &'static str,        // one-line semantic intent
    pub fields: &'static [OdooField],
    pub methods: &'static [OdooMethod],
    pub decorators: &'static [OdooDecorator],
    pub state_machine: Option<&'static OdooStateMachine>,
    pub constraints: &'static [OdooConstraint],
    pub provenance: OdooProvenance,        // L-doc lines + Odoo source paths
}
pub struct OdooField {
    pub name: &'static str,
    pub kind: OdooFieldKind,               // Char / Many2one / One2many / Many2many / Selection / ...
    pub target: Option<&'static str>,
    pub required: bool,
    pub computed: Option<&'static str>,
    pub depends: &'static [&'static str],  // @api.depends targets
    pub semantic_role: OdooSemanticRole,   // Identity / Reference / Quantity / Date / Policy / ...
}
pub struct OdooMethod {
    pub name: &'static str,                // "_compute_fiscal_position", "action_post", ...
    pub kind: OdooMethodKind,              // Compute / Inverse / Constrain / Onchange / Action / Cron / ApiModel / ...
    pub return_kind: OdooReturnKind,       // Unit / Self_ / Record / Recordset / Bool / Number / ...
    pub triggers: &'static [&'static str], // state transitions this method fires (if Action)
}
pub struct OdooDecorator {
    pub kind: OdooDecoratorKind,           // ApiDepends / ApiConstrains / ApiOnchange / ApiModelCreateMulti / ...
    pub targets: &'static [&'static str],
}
pub struct OdooStateMachine {
    pub state_field: &'static str,         // typically "state"
    pub states: &'static [OdooState],
    pub transitions: &'static [OdooTransition],
}
pub struct OdooState { pub name: &'static str, pub semantic: OdooStateSemantic /* Draft/Active/Completed/Cancelled/Terminal/... */ }
pub struct OdooTransition { pub from: &'static str, pub to: &'static str, pub trigger: &'static str, pub guards: &'static [&'static str] }
pub struct OdooConstraint { pub kind: OdooConstraintKind /* Sql/Python(@api.constrains)/Domain/... */, pub condition: &'static str, pub source_method: Option<&'static str> }
pub struct OdooProvenance {
    pub l_doc: &'static str,                                 // "L9-PARTNER-FISCALPOS.md"
    pub l_doc_lines: (u32, u32),
    pub odoo_source: &'static [(&'static str, u32, u32)],    // (path, start, end), multi if entity spans files
    pub confidence: OdooConfidence,                          // Curated / Extracted / Conjecture
}
```

The inheritance chain operates on this typed DTO as input — replacing today's
string-keyed lookups.

## Deliverables

| D-id | Scope | Crate | Lines | Conf | Status |
|---|---|---|---|---|---|
| **D-ODOO-BP-1a** | `OdooEntity` + sub-types (zero-dep, const-only, no serde) — the typed surface | `lance-graph-ontology` | 300 | HIGH | Queued |
| **D-ODOO-BP-1b** | L-doc projection: one `OdooEntity` const per entity in each L-doc, 15 lanes, per-lane module `odoo_blueprint::l{1..15}`, provenance=Curated, line-range citations | `lance-graph-ontology` | 2500 | HIGH | Queued |
| **D-ODOO-BP-1c** | Wire OGIT classifier to take `&OdooEntity` (replaces string-keyed `resolve_odoo`); uses field/method semantics for richer dispatch; covers 0x63/0x90 from PR #414 | `lance-graph-ontology` + `lance-graph-callcenter::family_table` | 250 | HIGH | Queued |
| **D-ODOO-BP-1d** | Wire OWL hydrator to take `&OdooEntity`: relational fields → edges, computed fields → SHACL-equivalent constraints, decorators → axioms | `lance-graph-ontology` | 350 | MED | Queued |
| **D-ODOO-BP-1e** | Wire DOLCE classifier + FIBU/FIBO alignment to take `&OdooEntity`; close out D-ODOO-SAV-2's `None`-class alignment for `stock.*` / `analytic.distribution.model` / `account.account.tag` over typed input | `lance-graph-ontology` | 200 | HIGH | Queued |
| **D-ODOO-BP-1f** | Odoo source extraction tool: walk `/home/user/odoo`, parse Python AST for ORM classes via tree-sitter, emit candidate `OdooEntity` consts with `Confidence=Extracted`; merge into 1b's curated set as a follow-up validation pass | `tools/odoo-blueprint-extractor/` | 800 | MED | Queued |
| **D-ODOO-BP-1g** | Wire JITson → recipes: `jit::JitCompiler` compiles `Tactic` kernels parameterized by `(OdooEntity, AtomTouchMask)`; produces the DTO-ish NARS that lands in the shader-driver | `lance-graph-contract::jit` + `thinking-engine` | 400 | MED | Queued |

## Execution

1. **D-ODOO-BP-1a first** — typed surface. Ships with this plan + board
   hygiene (INTEGRATION_PLANS prepend + STATUS_BOARD section). Additive.
2. **D-ODOO-BP-1b in waves** — one Wave per lane group (L1–L5, L6–L10,
   L11–L15), one subagent per lane (Sonnet, mechanical projection of
   prose → const data, Tier-0 reads of L-doc + savant docs). ~5 entities
   per lane average × 15 lanes ≈ 75–200 consts. Per-lane PRs land
   independently once 1a is in.
3. **D-ODOO-BP-1c/d/e in parallel after 1b** — three inheritance-chain
   hops, independent, each takes `&OdooEntity` as the typed input.
   Replaces the current ad-hoc string maps.
4. **D-ODOO-BP-1f after 1b/c/d/e** — source extractor validates the
   curated set + extends to non-savant-relevant entities. Conflicts
   (curated vs extracted) flag for human ratification.
5. **D-ODOO-BP-1g closes the loop** — JITson compiles `Tactic` kernels
   over the normalized `OdooEntity` → DTO-ish NARS in the shader.
6. **THEN `odoo-savant-reasoners-v2` Group F** is unblocked — per-savant
   `SavantPattern` consts compose over the normalized chain.

`odoo-savant-reasoners-v2` Groups D/E/G remain unblocked by this plan
(they ship independently; F is the only group blocked on the blueprint).

## Open questions / risks

- **L-doc completeness vs Odoo source ground truth**: L-docs are
  human-curated samples; entities the savants need may be partially
  documented. 1f extraction backs them up. If a curated entity disagrees
  with an extracted one, default to the curated (humans filtered for
  relevance) and flag in `OdooProvenance` for review.
- **Const data binary size**: ~150-300 entities × ~20 fields + methods
  + decorators could grow the binary. Mitigation: feature-gate per-lane
  modules (`odoo-blueprint-l9` etc.) so consumers pull only what they
  need; default-on for all lanes for the build-once developer experience.
- **Provenance drift**: L-doc edits would desync the const data. v1f
  extractor becomes the long-term ground truth; v1b is the validated
  cache that re-syncs on demand.
- **OdooSemanticRole / OdooStateSemantic enum coverage**: the enum
  variants must cover what the 15 lanes surface without trailing
  `Other(string)` escape hatches. Mitigation: 1b projection drives
  enum-variant discovery; expand the enums per-Wave, freeze after L15.
- **JITson wiring (1g)**: `jit::JitCompiler` exists but isn't wired
  to recipes yet. Scope-creep risk into Cranelift kernel templates.
  Mitigation: 1g produces ONE proof-of-concept JIT path (FiscalPositionResolver),
  the rest follow in `odoo-savant-reasoners-v2` Group F.

## Invariants

- Zero-dep, const-only, no serde on the DTO surface (AGI-as-glove +
  contract discipline).
- Provenance on EVERY entity (`I-VSA-IDENTITIES`: content in typed
  registries, never bundled — the const IS the registry).
- Inheritance chain operates on typed DTOs, **never on string model
  names** (kills the ad-hoc lookup tax).
- L-doc projection is the canonical v1 source; Odoo source extraction
  is validation/extension.
- This blueprint is **PREREQUISITE for `odoo-savant-reasoners-v2`
  Group F**; v2's composition layer composes ON TOP of normalized DTOs.
- Board hygiene: this plan + INTEGRATION_PLANS PREPEND + STATUS_BOARD
  section land in the same commit as D-ODOO-BP-1a (per CLAUDE.md
  Mandatory Board-Hygiene Rule).
