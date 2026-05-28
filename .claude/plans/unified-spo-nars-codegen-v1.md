# unified-spo-nars-codegen-v1 — four-layer codegen unification (predicate / capability / meta-capability) + LabelDTO compression — Stage 2 trunk of `normalized-entity-holy-grail-v1`

> **Status:** PROPOSAL. Stage 2 trunk plan that follows `normalized-entity-holy-grail-v1` Stage 1 (PR #431). Absorbs four architectural layers identified during the 2026-05-28 session: predicate-level NARS auto-emission, capability-level method emergence, meta-capability-level analogical transfer, and the LabelDTO compression. Final consequence: the substrate is one DTO type + one codebook + one chain, all typed shapes codegen'd from `format:domain:ogit` URNs.
>
> **Confidence:** HIGH on the layered structure (each layer is a natural extension of the codegen pipeline, no new inference engine required at any tier). HIGH on the LabelDTO compression — it collapses the existing `NormalizedEntity<S>` carrier to label + row + phantom, every other field becomes a `label.resolve()` projection. MED on the analogical-transfer convergence rate (depends on volume of meta-pattern instances; `ogit:analogyTruthFloor` gate prevents over-propagation). MED on the codegen LOC budget (estimate ~50K LOC of generated const data for the TIER-1 medical + financial union; verify against actual extraction). LOW on the cross-domain join-verb taxonomy (open question whether `ogit:identifies` covers the full medical-↔-financial bridging or we need a new `ogit:isSameRealWorldEntity`).
>
> **Predecessors:**
> - `normalized-entity-holy-grail-v1` Stage 1 (PR #431) — typed carrier + Op trait + 3 transaction contexts + cascade scaffold
> - `odoo-source-extraction-v1` Stage 1 (PR #426) — 12 TIER-1 addons, 229 extracted Odoo entities, 73K LOC; surfaced the 3-hop hot paths
> - OGIT PR #7 (`adaworldapi/OGIT` 3-hop shrink demonstration) — Accounting:JournalEntry + FiscalJurisdiction + shortcut verbs
> - OGIT PR #8 (`adaworldapi/OGIT` cargo crate) — Cargo.toml + build.rs + lib.rs; consumer-side dep
> - `lance-graph-contract::nars` — existing 5 InferenceType (Deduction / Induction / Abduction / Analogy / Revision) + truth-value algebra
> - `lance-graph-contract::callcenter::ogit_uris` — hand-maintained canonical OGIT URI codebook (to be replaced by codegen output)
> - `lance-graph-ontology::hydrators::{dolce, fibo, odoo, owl, owltime, provo, qudt, schemaorg, schematron, skos, skr, skr_datev, xsd, zugferd}` — 14 hydrators with G-slot URIs already declared
> - MedCare-rs releases `bioportal-ontologies-2026-05-05` (2.45 GB, 22 ontologies) + `drug-knowledge-bases-2026-05-05` (68 MB, CPIC + PharmGKB) — the medical-side harvest with F2-F22 + K1-K11 phase plans
>
> **Driver epiphanies** (filed during 2026-05-28 session):
> - `E-NORMALIZED-ENTITY-1` (carrier is one struct, 4-way inheritance chain)
> - `E-OP-FIVE-VERBS-1` (resolve/hydrate/classify/align/think — algebraic closure)
> - `E-OP-THREE-CALLSITES-1` (cold/warm/hot, shared const data)
> - `E-TRANSACTION-CONTEXT-1` (interactive/bulk/periodisch own commit policy)
> - `E-CASCADE-AS-EDGECOLUMN-1` (6 Odoo cascade mechanisms collapse to 1 typed graph)
> - `E-ODOO-AS-PRIOR-ART-1` (decomposition borrowed, compile-time encoding earned)
> - `E-CONSUMER-CANNOT-INTERPRET-1` (regex structurally banned via missing-function)
> - `E-NO-AUTOMATIC-REGIME-PICK-1` (consumer-typed context picks mode)
> - `E-CODEBOOK-INHERITS-FROM-OGIT` (every identity inherits from OGIT codebook)
> - `E-OGIT-DISTRIBUTED-CROSS-ONTOLOGY-CONTRACT-1` (cross-ontology mapping IS the TTLs, no separate table)
> - `E-HYDRATOR-AS-OGIT-CODEGEN-1` (each hydrator is the canonical source for its source-↔-OGIT cross-references)
> - `E-FININT-AS-HOLY-GRAIL-CONSUMER-1` (bank-data → sanctions → SAR is the demanding showcase)
> - `E-MEDICAL-MIRRORS-FINANCIAL-1` (substrate's two flagship domains converge on the same chain)
> - `E-FOUR-AXIS-DECOMPOSITION-1` (HIRO MARS / our SoA convergence: 4 axes is the universal-substrate cardinality)
> - `E-OGIT-ACTIVE-GROWTH-SURFACE-1` (the Boos commits + agent-mysql-transcode pattern are workspace signals for where to invest)
> - `E-LICENSE-AS-TYPED-CAPABILITY-1` (license terms gate ingest via the transaction context)
> - `E-CAPABILITY-FROM-PREDICATE-CONJUNCTION-1` (predicate conjunctions emit emergent methods)
> - `E-CROSS-DOMAIN-CAPABILITY-1` (capabilities span ontology branches)
> - `E-NARS-COMPOUND-TERM-IS-THE-CAPABILITY-1` (compound terms = capability methods)
> - `E-META-CAPABILITY-PATTERN-1` (capabilities cluster into meta-patterns; IQ-test analogical reasoning)
> - `E-NARS-CLOSED-UNDER-META-ELEVATION-1` (NARS algebra works unmodified at each layer)
> - `E-NEW-DOMAIN-AUTO-METABOLIZES-1` (substrate learns by structural analogy; new domains attach via meta-pattern matching)
> - `E-UNIFIED-CHAIN-CROSS-DOMAIN-1` (financial + medical share chain; domain is a data choice)
> - `E-EVERY-VERB-EMITS-NARS-METHODS-1` (codegen output, not hand-written)
> - `E-SPO-IS-THE-COMMON-OUTPUT-1` (every chain step emits SPO triples to the AriGraph)
> - `E-LABEL-DTO-IS-THE-SUBSTRATE-1` (universal carrier is `LabelDTO`; all shapes are projections)
> - `E-FORMAT-DOMAIN-OGIT-IS-THE-URN-SCHEME-1` (3-tier URN specifies parser + domain + class)
> - `E-CODEBOOK-IS-THE-COMPILATION-TARGET-1` (codegen consumes all sources, emits one const array)
>
> **Anchored iron rules:**
> - `I-VSA-IDENTITIES` (identity in const data, content in tables — codebook entries ARE identities)
> - "AGI-as-glove" (the four SoA columns are the surface; this plan adds no new column, only adds typed projections via `LabelDTO::resolve()`)
> - "Lab vs canonical" (codegen is build-time; the runtime surface stays canonical-Op-chain)
> - "No service queries" (the codebook is `const`-data, dispatched in-process; no AGI service to call)
> - "Read before Write" (codegen reads existing TTLs / extracted Rust / hydrator declarations and emits NEW const data; never overwrites)

## The diagnosis

Stage 1 (PR #431) shipped the typed carrier + Op trait + 3 transaction contexts + cascade scaffold. What's still missing — the **codegen-emitted middle** of the substrate:

1. **No SPO emission per chain step.** Today `entity.op(op)` advances the stage but writes nothing into the AriGraph. The chain produces typestate transitions; the cognitive substrate needs SPO triples with NARS truth values flowing into the AriGraph for cumulative reasoning to work.
2. **No auto-emission of NARS methods per OGIT verb.** The `nars::InferenceType` algebra exists; the AriGraph SPO storage exists. What's missing is the binding: per declared OGIT verb, emit 5 method handles (one per inference type) that the chain can dispatch through.
3. **No capability layer.** Predicate-conjunctions (e.g. `JournalEntry ∧ GoBD-compliant`) imply emergent methods (`verifySecureSequenceIntact`, `lockAfterPost`, `exportDatevWithAuditTrail`) that no single predicate emits. Today these emergent capabilities are conceptual; no typed encoding.
4. **No meta-capability / analogical-transfer layer.** The IQ-test primitive (Kontenaudit:GoBD :: Xray:ICD :: MaintenanceLog:FAA-Part43 — all `(Activity, NormativeFramework)`) is unimplementable today. New domain entities can't auto-metabolize via structural analogy.
5. **No LabelDTO compression.** `NormalizedEntity<S>` carries 5 inheritance slots (`odoo`/`ogit`/`owl`/`dolce`/`fibu`) as inline fields. The compression collapses these to **one URN string** (`format:domain:ogit`); the rest are `label.resolve()` projections via codebook lookup.
6. **No HydratorCodegen pipeline.** Each hydrator (FIBO/SKR/ZUGFeRD/BioPortal/CPIC/PharmGKB/Odoo) knows its source-↔-OGIT cross-references in its head; nothing aggregates them into OGIT-side TTL backpropagation or const-data Rust emission.
7. **No cross-domain join verbs.** Financial entities (`Accounting:CommercialPartner`) and medical entities (`Medical:Patient`) can refer to the same real-world person, but the substrate has no typed `ogit:identifies` / `ogit:isSameRealWorldEntity` edge to express it.

Each gap is closeable by codegen; none requires new inference machinery. The plan formalises the codegen pipeline that closes all seven.

## The four layers — one substrate

The substrate has exactly four layers of structure, each emitted by the same codegen pipeline reading the same OGIT TTL + hydrator + capability + meta-capability declarations:

| Layer | Object of generation | Codegen output per object |
|---|---|---|
| **1 — Predicate** | one OGIT verb (e.g. `hasFiscalCountry`) | 5 NARS method handles (deduce/induce/abduce/analogize/revise) bound to that verb's `(s, p, o)` shape |
| **2 — Capability** | one declared `ogit:Capability` (conjunction of N triples; e.g. `GoBdInvoice`, `BoneDiseaseDifferential`) | M OpKind discriminants emitted as available methods when all required triples are present; gating via SPO triple-conjunction check |
| **3 — Meta-capability** | one declared `ogit:MetaCapability` (analogical pattern over N capability instances; e.g. `ActivityConstrainedByFramework`) | Propagated method shapes that auto-instantiate for new domain instances via NARS analogy; new-instance abduction hooks |
| **4 — Compression** | the entire substrate's typed surface | One `OGIT_CODEBOOK: &[OgitCodebookEntry]` const array indexed by `LabelDTO` URN; `NormalizedEntity<S>` shrinks to `label + row + phantom` |

NARS algebra (the 5 inference types) is closed under elevation through all four layers — no new engine at any tier. Each layer adds compound-term shape, not new inference primitives.

### Predicate layer — auto-emit 5 NARS methods per verb

Every OGIT verb declaration (e.g. `ogit.Accounting:hasFiscalCountry`) generates:

```rust
pub const SPO_HAS_FISCAL_COUNTRY: SpoTripleShape = SpoTripleShape {
    subject_class:   ogit_class!("Accounting:JournalEntry"),
    predicate:       ogit_verb!("Accounting:hasFiscalCountry"),
    object_class:    ogit_class!("Accounting:FiscalJurisdiction"),
    inferences:      &NARS_METHODS_FOR_HAS_FISCAL_COUNTRY,
};

pub const NARS_METHODS_FOR_HAS_FISCAL_COUNTRY: NarsMethodSet = NarsMethodSet {
    deduce:   deduce_has_fiscal_country,    // table-driven default; per-verb override allowed
    induce:   induce_has_fiscal_country,
    abduce:   abduce_has_fiscal_country,
    analogize: analogize_has_fiscal_country,
    revise:   revise_has_fiscal_country,
};
```

Default `deduce` body: walk the `ogit:allowed` graph from the object class to find downstream consequences. Default `revise`: existing NARS truth-value merge. Per-verb overrides are opt-in via the OGIT verb's optional `ogit:overrideDeduce true` declaration.

### Capability layer — predicate conjunctions emit emergent methods

A `ogit:Capability` is a first-class OGIT entity declaring an N-predicate conjunction + the methods that emerge when all are present:

```turtle
ogit.Accounting:GoBdInvoiceCapability
    a ogit:Capability ;
    ogit:requires (
        [ ogit:onSubject ogit.Accounting:JournalEntry ;
          ogit:withVerb ogit:hasMoveType ;
          ogit:toValue "invoice"^^xsd:string ]
        [ ogit:onSubject ogit.Accounting:JournalEntry ;
          ogit:withVerb ogit:complies ;
          ogit:toObject ogit.Legal:GoBD-AuditTrail ]
    ) ;
    ogit:emits (
        ogit:method:verifySecureSequenceIntact
        ogit:method:lockAfterPost
        ogit:method:exportDatevWithAuditTrail
        ogit:method:reverseChargeIfApplicable
    ) ;
.
```

Codegen output: `Capability<N>` const + runtime capability registry; consumers access via `entity.capability(GOBD_INVOICE).map(|c| c.export_datev_with_audit_trail())`. Capability gating is runtime-checked at Stage 2 (against the AriGraph SPO state); typestate-refined gating is deferred to Stage 3.

### Meta-capability layer — cross-domain analogical reasoning

A `ogit:MetaCapability` declares an abstract pattern (with metaclass-typed placeholders) + known instances + method shapes that propagate via NARS analogy:

```turtle
ogit.Meta:ActivityConstrainedByFramework
    a ogit:MetaCapability ;
    ogit:abstractTriple [ ogit:abstractSubject ogit.Meta:Activity ;
                          ogit:abstractPredicate ogit:constrainedBy ;
                          ogit:abstractObject ogit.Meta:NormativeFramework ] ;
    ogit:instances (
        [ ogit:concreteSubject ogit.Accounting:Kontenaudit ; ogit:concreteObject ogit.Legal:GoBD ]
        [ ogit:concreteSubject ogit.Medical:XRayImagingStudy ; ogit:concreteObject ogit.Medical:Icd10Coding ]
        [ ogit:concreteSubject ogit.Compliance:TransactionMonitoring ; ogit:concreteObject ogit.Compliance:OfacSanctionsList ]
        [ ogit:concreteSubject ogit.Aviation:MaintenanceLog ; ogit:concreteObject ogit.Aviation:FAA-Part43 ]
    ) ;
    ogit:propagates (
        ogit:method:meta:assertCompliance
        ogit:method:meta:auditTrailFor
        ogit:method:meta:periodicCertification
        ogit:method:meta:detectGap
    ) ;
    ogit:analogyTruthFloor [ ogit:freq 0.85 ; ogit:conf 0.70 ] ;
.
```

New domain entities (e.g. `ogit.Pharmacy:DispenseLog`) that match K-1 of K abstract slots get analogically-propagated methods at attenuated NARS confidence (`ogit:analogyTruthFloor` gates the auto-emission; below floor, the substrate proposes the missing slot for maintainer ratification).

### Compression layer — LabelDTO + OGIT_CODEBOOK

The final reduction: every typed shape in the substrate becomes a projection of one URN string.

```rust
/// Universal carrier — one type, all shapes via codebook projection.
pub struct LabelDTO {
    pub label: &'static str,  // "<format>:<domain>:<ogit-class>"
}

pub struct OgitCodebookEntry {
    pub label:         &'static str,
    pub format:        Format,            // turtle / rdfxml / obo / csv / xsd / json
    pub domain:        &'static str,
    pub class_uri:     &'static str,
    pub allowed_edges: &'static [OgitEdge],
    pub nars_methods:  &'static [NarsMethodHandle],
    pub capabilities:  &'static [&'static Capability],
    pub meta_patterns: &'static [MetaPatternMembership],
    pub equivalents:   &'static [&'static str],  // FIBO/SKR/ZUGFeRD/BioPortal cross-refs
    pub regulations:   &'static [&'static str],
    pub hydrator:      HydratorFn,
}

pub const OGIT_CODEBOOK: &[OgitCodebookEntry] = &[/* codegen output */];
```

`NormalizedEntity<S>` collapses to `{label: LabelDTO, row: MailboxRow, _stage: PhantomData<S>}`. The 5 inheritance slots become `label.resolve()` projections.

## Stage-2 deliverables

### Wave A — typed surface (`lance-graph-contract`)

| D-id | What | Site | LOC | Conf |
|---|---|---|---:|:--:|
| **D-USN-1** | `SpoTripleShape` + `NarsMethodSet` + `NarsMethodHandle` types — typed compile-time SPO grammar | `lance-graph-contract::cognition::ogit_spo` | 250 | HIGH |
| **D-USN-1.5** | `LabelDTO` + `OgitCodebookEntry` types — universal carrier + codebook entry shape; `Format` enum (Turtle/RdfXml/Obo/Csv/Xsd/Json) | `lance-graph-contract::cognition::label_dto` | 200 | HIGH |
| **D-USN-9** | `Capability<N>` type + `TripleConstraint` enum (Exact/Value/Subgraph/Range) + runtime registry | `lance-graph-contract::cognition::capability` | 300 | HIGH |
| **D-USN-12** | `MetaCapability` type + abstract-metaclass placeholders + analogy-truth-floor gating | `lance-graph-contract::cognition::meta_capability` | 350 | MED |

### Wave B — OGIT root vocabulary additions (`adaworldapi/OGIT`)

| D-id | What | Site | LOC |
|---|---|---|---:|
| **D-USN-13a** | `ogit:Capability` class + `ogit:requires` / `ogit:emits` / `ogit:onSubject` / `ogit:withVerb` / `ogit:toValue` / `ogit:toObject` / `ogit:method:*` verbs | OGIT root + ogit.ttl | 60 |
| **D-USN-13b** | `ogit:MetaCapability` class + abstract metaclasses (`ogit.Meta:Activity`, `ogit.Meta:NormativeFramework`, `ogit.Meta:CodingSystem`, `ogit.Meta:Measurement`, `ogit.Meta:Event`, `ogit.Meta:Identity`) | OGIT root + ogit.ttl | 100 |
| **D-USN-13c** | First-wave capabilities (5-8 financial + 5-8 medical): `GoBdInvoice`, `KleinunternehmerReverseCharge`, `EuOssTransaction`, `JahresabschlussWindow`, `SanctionsCrossListed`, `BoneDiseaseDifferential`, `PharmacogenomicDrugResponse`, `ElderlyPolymedication`, `OperativePostFollowup`, `BillingClaimCrossRef` | OGIT TTLs in `NTO/Accounting/capabilities/` + `NTO/Medical/capabilities/` | 800 |
| **D-USN-13d** | First-wave meta-capabilities (5-8 patterns): `ActivityConstrainedByFramework`, `ProcessGeneratesEvidence`, `EntityHasCanonicalIdentifier`, `MeasurementProducesValueInRange`, `EventTriggersCompliance`, `IdentityCrossListedAcrossSources`, `NormativeChangeRevisesPriorClassification` | OGIT TTLs in `NTO/Meta/capabilities/` | 600 |

### Wave C — HydratorCodegen pipeline (`lance-graph-ontology` + `tools/ogit-codegen/`)

| D-id | What | Site | LOC |
|---|---|---|---:|
| **D-USN-2** | `HydratorCodegen` trait + `CrossRef` struct in `hydrators::ogit_codegen`; each existing hydrator (fibo, skr, zugferd, dolce_odoo, schemaorg, qudt, …) gets an `impl HydratorCodegen` that emits its source-↔-OGIT cross-references | `lance-graph-ontology::hydrators::ogit_codegen` + per-hydrator extensions | 800 |
| **D-USN-2a** | `OdooHydratorCodegen` impl — projects EXT-1..6's 229 extracted models as `(ogit_class, model_name, CrossRefKind::SeeAlso)` rows; the 30-50 high-value rows with semantic 1:1 correspondences promote to `EquivalentClass` | `lance-graph-ontology::hydrators::odoo_codegen` | 150 |
| **D-USN-2b** | Pattern-match engine for `format:domain:ogit` URN scheme; resolution from any `LabelDTO` → its `(format, domain, ogit-class)` triple at compile time | `tools/ogit-codegen/src/urn.rs` | 200 |
| **D-USN-2c** | TTL backpropagation emitter — aggregates `CrossRef[]` across all hydrators, emits per-OGIT-class TTL fragments (`rdfs:seeAlso` / `owl:equivalentClass` / `ogit:complies` lines) ready to PR against `adaworldapi/OGIT` | `tools/ogit-codegen/src/ttl_emit.rs` | 350 |
| **D-USN-2d** | First-regeneration PR against `adaworldapi/OGIT` adding the codegen-generated cross-references (replaces planned manual PR #9) | output PR | (output) |

### Wave D — auto-emission (`tools/ogit-codegen` + `lance-graph-contract`)

| D-id | What | Site | LOC |
|---|---|---|---:|
| **D-USN-3** | NARS-method default impls — table-driven `deduce` walks the `ogit:allowed` graph; `revise` uses `nars::truth_value::merge`; `induce` / `abduce` / `analogize` get scaffolded defaults with TODO(stage-2-tune) markers | `lance-graph-contract::nars::default_methods` | 400 |
| **D-USN-3a** | Codegen extension — emit `OGIT_CODEBOOK` const from all sources; 1 entry per OGIT class with all 4 layers populated; ~1500 entries for TIER-1 financial + medical union | `tools/ogit-codegen/src/codebook.rs` | 500 |
| **D-USN-3b** | Codebook lookup in `lance-graph-contract::callcenter::ogit_uris` — replace hand-maintained const data with `include!(concat!(env!("OUT_DIR"), "/codebook.rs"))`; the consumer's build.rs runs `ogit-codegen` if its inputs changed | `lance-graph-contract/build.rs` + `ogit_uris.rs` refactor | 200 |
| **D-USN-3c** | SPO emission instrumentation in `NormalizedEntity::op` / `chk_data` / `review` / `abduct` / `report` — each transition writes the SPO triple to the AriGraph with the NARS truth value | `lance-graph-contract::cognition::advance` extension | 150 |
| **D-USN-3d** | NARS Revision on triple emission — when `(s, p, *)` already exists, apply `nars::truth_value::merge`; otherwise assert new | `lance-graph-contract::cognition::ari_emit` | 200 |

### Wave E — medical-side EXT projection (`lance-graph-ontology::extracted::medical`)

| D-id | What | Site | LOC |
|---|---|---|---:|
| **D-USN-4** | Project the 25 MedCare MySQL-transcoded OGIT entities (NTO/Medical, agent-mysql-transcode 2026-05-07) as typed Rust const data — same shape as EXT-1..6 Odoo extraction | `lance-graph-ontology::extracted::medical::medcare` | 1200 |
| **D-USN-4a** | Project the 22 BioPortal ontologies as `OgitCodebookEntry` rows — one entry per top-level class; per-ontology files (`bioportal_*.rs`) | `lance-graph-ontology::extracted::medical::bioportal::*` | ~3000 (codegen) |
| **D-USN-4b** | Project CPIC (11 JSON endpoints, ~5000 guidelines/recommendations/pairs) as typed Rust const arrays | `lance-graph-ontology::extracted::medical::cpic` | ~2000 (codegen) |
| **D-USN-4c** | Project PharmGKB (13 ZIP archives, ~12000 clinical annotations + 200K relationships) as typed Rust — large; consider feature-gated since PharmGKB License is commercial-restricted | `lance-graph-ontology::extracted::medical::pharmgkb` (feature: `pharmgkb`) | ~5000 (codegen) |
| **D-USN-4d** | Cross-domain join verbs — `ogit:identifies` / `ogit:participatesIn` / `ogit:isSameRealWorldEntity` between Financial (`Accounting:CommercialPartner`, `FinancialMarket:Corporation`) and Medical (`Medical:Patient`, `CustomerSupport:User`) | OGIT PR + codegen extension | 300 |

### Wave F — first capabilities + first meta-capabilities (codegen-driven)

| D-id | What | Site |
|---|---|---|
| **D-USN-10** | First-wave `Capability<N>` const data — codegen produces `GOBD_INVOICE_CAPABILITY`, `BONE_DISEASE_DIFFERENTIAL_CAPABILITY`, etc. from the OGIT TTLs in Wave B (D-USN-13c) | `lance-graph-contract::cognition::capability::generated` |
| **D-USN-11** | Capability gating + dispatch — `entity.capability(GOBD_INVOICE)` runtime-checks the AriGraph for the required SPO triples; if all present, returns a typed `CapabilityHandle` exposing the emitted methods | `lance-graph-contract::cognition::capability::gate` |
| **D-USN-14** | Pattern-matching engine for the meta layer — given an entity's SPO triples, identify which meta-capability instances apply via metaclass-typed placeholder substitution | `lance-graph-contract::cognition::meta_capability::match_engine` |
| **D-USN-15** | NARS analogy inference wired into the meta layer — abduction-for-missing-slot mechanism + analogical method propagation at attenuated truth | `lance-graph-contract::nars::analogy_propagate` |

### Wave Σ — LabelDTO compression of the carrier

| D-id | What | Site |
|---|---|---|
| **D-USN-Σ-1** | Refactor `NormalizedEntity<S>` to `{label: LabelDTO, row: MailboxRow, _stage: PhantomData<S>}`; the 5 inheritance slots become `label.resolve()` projections | `lance-graph-contract::cognition::entity` (rewrite) |
| **D-USN-Σ-2** | Migration helpers — `OdooEntityRef::into_label_dto()`, `OgitUriRef::into_label_dto()`, etc. — preserve external API for one cycle, then remove the old slots | `lance-graph-contract::cognition::entity::migrate` |
| **D-USN-Σ-3** | Cross-domain chain demo — single chain that crosses domains: medical procedure billing → Patient resolution → JournalEntry posting → UStG-§13 + sanctions check, ALL through one typed chain on top of the unified codebook | `examples/cross_domain_billing.rs` |

**Total Stage 2 LOC (hand-written + codegen):** ~3000 LOC hand-written + ~12000 LOC generated const data. The generated data is reproducible from `tools/ogit-codegen` run; hand-written code is structural plumbing.

## Execution ordering — six sequential waves

The waves serialise on the codegen pipeline; within each wave, parallelisable work fans to Sonnet agents.

**Wave A (typed surface)** — D-USN-1, 1.5, 9, 12. Independent of any OGIT changes. Parallel-spawnable. **~1100 LOC, 1 Sonnet agent or 4 in parallel.**

**Wave B (OGIT root vocabulary)** — D-USN-13a, 13b, 13c, 13d. New OGIT TTLs across `NTO/Accounting/capabilities/`, `NTO/Medical/capabilities/`, `NTO/Meta/`. **~1560 LOC, single OGIT PR ('claude/spo-nars-vocab-v1'); 1 Sonnet agent.**

**Wave C (HydratorCodegen pipeline)** — D-USN-2, 2a, 2b, 2c. Requires Wave A's `SpoTripleShape` + `LabelDTO` types. **~1500 LOC + first OGIT regeneration PR (D-USN-2d) output; 2 Sonnet agents in parallel (one for the pipeline, one for per-hydrator codegen impls).**

**Wave D (auto-emission)** — D-USN-3, 3a, 3b, 3c, 3d. Requires Wave B (the OGIT vocabulary additions) + Wave C (the codegen pipeline). Mostly mechanical. **~1450 LOC; 1 Sonnet agent.**

**Wave E (medical-side EXT)** — D-USN-4, 4a, 4b, 4c, 4d. Independent of Wave D (codegen extends to medical surface without changes to the engine). Largest LOC budget; mostly codegen output. **~11500 LOC, ~3000 hand-written; 3-5 Sonnet agents (one per top-level subset: BioPortal-anatomy, BioPortal-disease, BioPortal-drug, CPIC, PharmGKB).**

**Wave F (capabilities + meta-capabilities)** — D-USN-10, 11, 14, 15. Requires Waves A-D. **~1000 LOC; 2 Sonnet agents (one for capability gating, one for meta-capability matching).**

**Wave Σ (compression)** — D-USN-Σ-1, Σ-2, Σ-3. Capstone refactor + demo. Touches every consumer of `NormalizedEntity`. **~400 LOC + an extensive cross-crate sweep; main thread (Opus) review + 1 Sonnet agent for the mechanical sweep.**

## Risks / open questions

- **PharmGKB License (commercial-restricted).** The Wave E projection of PharmGKB into typed Rust const data is fine for development; clinical-product redistribution requires commercial PharmGKB License negotiation with Stanford. Mitigation: feature-gate `pharmgkb` and document the gate in `License compliance` knowledge doc. Defer commercial license to MedCare-rs deployment milestone.
- **Codebook entry stability.** Once `OGIT_CODEBOOK` is consumed by downstream crates, removing an entry breaks every consumer. Mitigation: codegen emits `#[deprecated(since = "...")]` on removed entries; append-only convention matches OGIT's own `dcterms:valid "start=YYYY-MM-DD;"` append-only.
- **OpKind discriminant space exhaustion.** With ~1500 OGIT classes × ~5 capabilities each × ~M meta-pattern propagations, the OpKind discriminant space might exceed `u32::MAX`. Conjecture: `u32` is fine for 4B entries; if pressed, promote to `u64` in Stage 3.
- **Analogical-transfer false positives.** Meta-capability `ogit:analogyTruthFloor` gates above-floor auto-emission; below-floor, propose for ratification. If the floor is wrong, the substrate either over-emits (low floor) or under-discovers (high floor). Mitigation: collect ratification feedback in a `meta_capability_review.log`; learn the optimal floor per pattern. Defer the learning to Stage 3.
- **Cross-domain join verbs.** Open question: is `ogit:identifies` (an existing OGIT verb) sufficient for medical-↔-financial bridging, or do we need a stronger `ogit:isSameRealWorldEntity`? Decision deferred to D-USN-4d implementation; flag in the OGIT PR for upstream feedback.
- **Codegen LOC budget.** Stage 2 may produce ~50K LOC of generated const data (medical-side BioPortal projections dominate at ~10K LOC). Mitigation: structure generated code into per-ontology files (e.g. `bioportal_icd10cm.rs`, `bioportal_loinc.rs`) so individual files stay <2K LOC; consumer-side feature flags for non-default ontologies.
- **Build time regression.** Adding 1500 codebook entries + auto-derived NARS methods + capability checks may slow `cargo build` for downstream crates significantly. Mitigation: codegen emits `pub const` arrays (zero compile-time machinery); compile-time benchmarks gate the merge.
- **HIRO MARS / our SoA cardinality coincidence.** `E-FOUR-AXIS-DECOMPOSITION-1` notes both substrates converge on 4 axes. The codebook entry has 5 layers (predicate / capability / meta / cross-ref / regulation). Conjecture: collapsing to 4 layers (where regulation is a subset of cross-ref) is structurally cleaner. Verify against actual TTL volumes in Wave B.

## Subsequent stages (sketched only)

- **Stage 3 — typestate-refined capability gating.** Move capability gating from runtime registry to phantom-type witnesses on the carrier. `entity.unlock::<GoBdInvoice>()` returns `Option<NormalizedEntity<S, GoBdInvoice>>`; methods exist only on the typestated form. Compile-time enforcement of capability access.
- **Stage 4 — JIT-compiled Op chains.** `lance-graph-contract::jit::JitCompiler` compiles a fixed Op sequence into a Cranelift-emitted function; the hot path becomes a single compiled blob over the SoA. Per `E-OP-THREE-CALLSITES-1`'s warm/hot lines.
- **Stage 5 — `lance-graph-elixir-frontend-v1` (separate plan, sketched).** Elixir `|>` syntax compiles to the typed chain. Consumer ergonomics: domain experts read + write `.exs` flow files; codegen produces the Rust.
- **Stage 6 — `cognition-formal-verification-v1` (separate plan, sketched).** RustyDL-style proof obligations over the typestate transitions + cascade termination + Jahresabrechnung fixed-point existence.
- **Stage 7 — `lance-graph-surrealql-frontend-v1` (separate plan, sketched).** SurrealQL read-side companion to Elixir write-side; graph patterns over normalised entities.
- **Stage 8 — Foundry + OTP parity audits.** When v6 + v7 audits show ≥80% feature parity, the workspace earns the "better palantir foundry / better elixir" framing.

## Board updates that land with this plan

Per CLAUDE.md mandatory board-hygiene rule, the commit that creates this plan file also:
- PREPENDs an entry to `.claude/board/INTEGRATION_PLANS.md` pointing at this file
- PREPENDs 12 new epiphanies to `.claude/board/EPIPHANIES.md` (the driver epiphanies listed in the front matter that aren't already filed)
- Adds D-USN-1..Σ-3 rows to `.claude/board/STATUS_BOARD.md` once Wave A starts

Cross-references to update in subsequent waves:
- `LATEST_STATE.md` Contract Inventory: add `cognition::{label_dto, capability, meta_capability, ogit_spo}` + `nars::default_methods` once shipped
- `INTEGRATION_DEBT_AND_PATHS.md` §"the missing consumer surface": mark resolved when Wave Σ ships
- `knowledge/encoding-ecosystem.md`: add `OGIT_CODEBOOK` as the canonical codebook output

The plan completes the holy-grail trunk: `normalized-entity-holy-grail-v1` Stage 1 (PR #431) established the carrier + chain; this plan establishes the codegen + capabilities + analogical reasoning + LabelDTO compression that make the carrier's promise (one substrate, all domains, audit-by-construction, NARS-typed throughout) structurally true rather than aspirationally true.
