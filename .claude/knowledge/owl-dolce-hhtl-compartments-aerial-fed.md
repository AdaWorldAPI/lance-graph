<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright The Lance Authors
-->

# KNOWLEDGE: Any OWL/DOLCE Wikidata domain → HHTL compartments, fed by aerial

## READ BY:
- Any worker landing a NEW domain (medicine / finance / geography / law / biology …) into the class-meta-DTO + HHTL router
- Any worker on `lance-graph-arm-discovery` (the aerial proposer feed), `contract::hhtl` / `class_view`, `ontology::wikidata_hhtl`
- `truth-architect`, `integration-lead`, `palette-engineer`

## P0 TRIGGERS:
- About to add a second/third domain ontology to the HHTL → read this first (do NOT grow a per-domain layer)
- About to give aerial a domain corpus to discover skeleton from → this is the feed contract
- About to widen `NiblePath` basins or `StructuralSignature` for "more domains" → see § Scale-freezes

---

## The claim (the potential)

**One machinery compartmentalizes *every* OWL/DOLCE-typed domain — including all
of Wikidata — into the same 16ⁿ HHTL tree, and aerial is the per-domain runtime
feed that discovers what lands where.** A medical ontology (SNOMED/FMA/ICD), a
financial one (FIBO), a geographic one, an ERP (Odoo), and Wikidata itself are
*not* special cases with bespoke loaders. Each yields the same four things —
a **basin** (DOLCE facet), a **nibble path** (P279 `subClassOf` descent), a
**`FieldMask`** (property presence), a **`StructuralSignature`** (shape-family) —
and lands as the identical `(ClassId, StructuralSignature, FieldMask)` row.
Domains differ only in *content*, never in *structure*.

This is the `cognitive-risc-classes.md` **N4** doctrine made operational: *"don't
freeze the SoA schema until ≥2 genuinely different domains run through it … chess
+ Odoo + Wikidata-anatomy all run through the same Class+SoA+HHTL+CAM with no
special-case."* Two domains already proved it — Odoo (#441) and Wikidata (#442);
this doc is the generalization to *N*, and aerial is the part that makes each
domain self-populating from data.

## Why it's domain-independent (the proof chain, not a hope)

| Layer | Type | Domain-agnostic because… | Proven by |
|---|---|---|---|
| Routing | `contract::hhtl::NiblePath` | takes a `basin: u8` + child nibbles; **zero DOLCE/domain knowledge** | #442 (4 teeth-tests) |
| Presence | `contract::class_view::FieldMask` | one bit per present property — any property-id set | #441 |
| Shape | `class_signature::StructuralSignature` | FNV-1a over the canonical property-id set — label/QID/domain-independent | #441 (Odoo) + #442 (Wikidata) |
| Resolution | `contract::class_view::ClassView` | a trait; `RegistryClassView` (Odoo) and `WikidataClassView` impl it **unchanged** | #441 + #442 |
| Categorical axis | DOLCE `dolce_id` `0..3` | resolved LATE from the OGIT cache; `basin = dolce_id`, no enum embedded | OD-DOLCE (#441 `b31464d`) |
| **Discovery feed** | `aerial` (this crate) | similarity is an injected integer `CodebookDistance` (the splat); the proposer emits flat `(s,p,o,f,c)` + `dolce_id` — domain enters only as *data* | #438 + #442 Phase 2 |

The collapse property (structurally-identical classes → one shape-family) is the
falsifier each domain must pass: Odoo's curated corpus collapses (#441 D-CLS-AUDIT),
Wikidata's does too (`film ≡ tv-series`, #442), and the aerial-fed worked example
reproduces it end-to-end (`tests/wikidata_landing.rs`).

## The compartmentalization (how a domain maps to nibbles)

```text
NiblePath (the ONE tree axis = Abstammung / P279 subClassOf):

  root nibble = DOLCE facet (the categorical scaffold, ogit-owl-dolce):
     0x0 Endurant   0x1 Perdurant   0x2 Quality   0x3 Abstract
     0x4..0xF       reserved (append-only) for finer top super-axes

  child nibbles = the domain's subClassOf descent (16-way fan-out per level)
     e.g.  Endurant → …organism → …mammal → …bat        (biology)
           Endurant → …anatomical-structure → …organ    (FMA/SNOMED)
           Perdurant → …audiovisual-work → …film         (Wikidata)

  DOMAIN ≠ a second path. The domain (Wikidata vs SNOMED vs FIBO vs Odoo) is an
  ORTHOGONAL compartment — a namespace/facet tag, NOT a branch. "One tree axis
  only" (wikidata-hhtl-load.md:46): bat = mammal-PATH + flight-BIT in the same
  FieldMask, never two paths. Cross-domain multi-typing is likewise a facet bit.
```

Every entity in any domain therefore gets a single O(1) bit-shift address
(`NiblePath`), a presence mask, and a shape signature — the substrate the
`wikidata-hhtl-load.md` compression (skeleton + basins + CAM-dedup + thin rows)
runs over, **identically per domain**.

> **Open design point (honest):** `ogit-owl-dolce-ontology-compartments.md` sketches
> *byte*-wide domain basins (`0x00..0x0F` universal/DOLCE, `0x10..0x19` healthcare),
> while `contract::hhtl::NiblePath::root` takes a *nibble* (`basin & 0x0F`). Reconcile
> before a multi-domain load: either domains occupy reserved top nibbles `0x4..0xF`
> as super-axes, or the domain is a namespace tag above the path. Not yet decided;
> the structural router is agnostic either way.

## The aerial feed (what makes each domain self-populating)

aerial is the **runtime-data proposer** — the per-domain frontend (sibling to the
static OWL/TTL `AstWalker`). For each domain:

1. **Offline (once, certified):** build the domain's codebook distance from the
   10000² Gaussian splat (`ndarray::hpc::splat3d`), certified by jc
   (`ewa_sandwich` ρ-push-forward, `sigma_codebook_probe` ρ=0.9973, `pflug` Lε).
   Emit the per-node top-k as `TopKDistance` edges.
2. **Online (integer):** `extract_rules(TopKDistance, domain_rows, θ/ppm)` →
   `CandidateRule`s — "entity-shape X ⇒ class/basin Y", the discovered skeleton
   edges for *that* domain.
3. **Emit the seam:** `OntologyProjector::dolce_id()` → the stable basin u8;
   `to_ndjson` → `{s,p,o,f,c}`. The hub lands them on `NiblePath`/`FieldMask`/
   `StructuralSignature`. The only things crossing the firewall: the
   `(ClassId, signature, FieldMask)` triple + the `dolce_id` u8.

So a new domain is "fed via aerial" by pointing it at (a) a domain splat codebook
and (b) a domain row corpus — **no new code on the hub side**. The
`tests/wikidata_landing.rs` worked example is the template; swap the fixture for
the new domain's classes.

## Pipeline — *any* domain → HHTL

```text
domain corpus + domain OWL/DOLCE types
   │  (offline)  10000² splat → jc-certified top-k edges
   ▼
aerial: TopKDistance + extract_rules     → CandidateRule (skeleton edges)
   │  OntologyProjector::dolce_id + ndjson
   ▼  ── firewall seam: (ClassId, signature, FieldMask) + dolce_id u8 ──
hub: NiblePath.root(basin).child(P279…)  +  FieldMask(presence)  +  signature(shape)
   ▼
the wikidata-hhtl-load.md row, per domain — skeleton + basins + CAM-dedup + thin rows
```

## What's proven vs CONJECTURE

- **Proven (shipped):** the per-layer domain-independence (Odoo #441 + Wikidata #442); aerial can feed it end-to-end on a fixture (#438 + Phase 2 worked example). The seams are concrete code.
- **CONJECTURE (the potential):** that *every* OWL/DOLCE domain at full scale compartmentalizes cleanly — untested beyond the two curated corpora. Each new domain is a falsification opportunity (run its corpus; does it collapse + route?).
- **Hard prerequisite:** D-ARM-7 (the Jirak floor, `jc::jirak`) gates promoting any discovered rule to a live skeleton — required before a domain's aerial feed writes to the store.

## Scale-freezes to watch as domains multiply (from the #442 review)

1. **`NiblePath` truncates silently at `MAX_DEPTH=16`.** Deep P279 chains (real medical/biological hierarchies exceed 16) collapse to the same address → cross-class collision. Needs a `depth_exhausted()` signal → switch-to-ref before a deep domain loads.
2. **`StructuralSignature` is `u32`.** Across many domains the distinct-shape count can approach the birthday bound (~77k) → two shapes alias to one family. Widen (or per-domain-namespace the hash) before the load that needs it.
3. **DOLCE basin nibble (4 used / 16) + the byte-vs-nibble reconciliation above** — freeze the domain-compartment addressing consciously, append-only.

## Cross-references

- `crates/lance-graph-arm-discovery/` — the aerial feed; `tests/wikidata_landing.rs` the worked template; `OntologyProjector::dolce_id`.
- `contract::hhtl::NiblePath` (#442), `class_view::FieldMask` (#441), `class_signature::StructuralSignature`, `ontology::wikidata_hhtl` (#442).
- `.claude/specs/wikidata-hhtl-load.md` (the compression), `.claude/knowledge/ogit-owl-dolce-ontology-compartments.md` (the DOLCE scaffold + domain compartments), `.claude/knowledge/splat-codebook-aerial-wikidata-compression.md` (the splat→aerial→Wikidata seam mechanics).
- `crates/jc` (certification: ewa_sandwich / sigma_codebook_probe / pflug / jirak), `ndarray::hpc::splat3d` (the splat).
- `EPIPHANIES.md` E-ARM-JC-RESOLVES-BOTH-SEAMS + the D-CLS↔D-ARM-14 convergence FINDING; `cognitive-risc-classes.md` N4.
