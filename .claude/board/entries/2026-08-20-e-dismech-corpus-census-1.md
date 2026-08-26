## 2026-08-20 — E-DISMECH-CORPUS-CENSUS-1 — the DisMech corpus measured: 87.2 MB of strings, of which the entire causal semantics is bits + codebook ordinals

**Status:** FINDING (measured on upstream `monarch-initiative/dismech`, 2,100
disorder files, fetched ephemeral to `/tmp` — never committed, per the
`/tmp`-fixture rule). **Confidence:** High — every number is a count.

**Total string bytes across all properties: 87,222,222.** The top ~27
properties by volume are free text (descriptions / snippets / explanations /
reference titles). Everything that carries CAUSAL SEMANTICS is tiny.

**`causal_link_type` is EXACTLY the four `CausalTopology` states** — measured,
not assumed, so the CE64 bits 59+60 mapping is source-authoritative and needs
no inference from confidence, edge count, or predicate name:

| state | count |
|---|---|
| DIRECT | 9,073 |
| INDIRECT_UNKNOWN_INTERMEDIATES | 4,539 |
| INDIRECT_KNOWN_INTERMEDIATES | 3,978 |
| UNKNOWN | 408 |
| **total causal edges** | **17,998** |

**The two experimental populations this hands us (better than hiding paths):**
`IndirectKnownIntermediates` (3,978) is the ORACLE population — the source
names the mediators, so they can be hidden and recovery measured.
`IndirectUnknownIntermediates` (4,539) is the RESTRAINT CONTROL — the source
itself does not know, so a reasoner that "recovers" a mediator there is
hallucinating closure. Success is therefore two-sided: recovery sensitivity
AND epistemic restraint.

**Every evidence enum is bits:** `supports` 4 (2 b), `evidence_source` 5 (3 b),
`modifier` 7 (3 b), `frequency` 19 (5 b), `treatment_effect` 5,
`genetic[].relationship_type` 10, `prevalence[].measure_type` 8.
⚠ `phenotypes[].category` is **261 distinct — OVER the 255 `Codebook` cap**,
and its top values include both `Neurologic` AND `Neurological`: lexical noise
inside an "enum". It needs normalization or a deliberate split, never a silent
widening (`Codebook::intern` returning `None` IS the split signal).

**Bibliography — LLM-generated, so identity must not be the title.** 131,904
reference-title occurrences over 31,361 distinct (4.21x reuse); 12,124,736 B
inline -> 3,051,438 B deduped (74.8% saved); with a u32 key per occurrence,
3,579,054 B total = 3.4x smaller. A stable key ALREADY exists on ~104,700
occurrences: PMID (dominant), DOI, ORPHA, CGGV, ClinicalTrials, URL. So the
key is `(namespace, id)` — never a hash of the title, because wording drifts
between regenerations while the citation does not.

**"MONDO-derived" applies to the DISORDERS, not the EDGE ENDPOINTS.** Measured
prefix distribution over edge endpoints: no prefix at all **25.2%**, HP 24.9%,
GO 11.6%, NCIT 10.5%, hgnc 7.6%, CL 7.3%, **MONDO only 4.4%**, UBERON 3.3%,
CHEBI 3.1%, NCBITaxon/ECTO/RO 2.0%. OBO_CORE's five namespaces cover **32.7%**.
This is the single largest correction to the DisMech-overlay plan: grounding
cannot lean on MONDO.

**Deterministic resolution ladder — no LLM used.** Phenotypes carry NO CURIE at
all (25,120 entries, 0.0%), so the mapping must be MADE. Against the real
HP/MONDO/UBERON/PATO labels:

| population | exact | +case/punct | +singular | RESOLVED | unresolved |
|---|---|---|---|---|---|
| phenotype labels (25,120) | 55.3% | 23.2% | 2.1% | **80.7%** | 19.3% (3,647 distinct) |
| unprefixed endpoints (30,872) | 25.6% | 9.9% | 0.9% | **36.4%** | 63.6% (11,385 distinct) |

**The unresolved tails are three DIFFERENT kinds, not one backlog:**
(a) provenance leaking into the endpoint slot — `Orphanet` (120), `OMIM` (32),
`ClinGen` (23) are database names, not concepts; (b) qualified mechanism
PROPOSITIONS — `Impaired Terminal Electron Transfer and ATP Synthesis` (24),
`Impaired Neurodevelopment` (20) — genuinely DisMech-local, and must not be
bullied into an ontology node; (c) lexical variants of real concepts. Note
`Sensorineural Hearing Loss` and `Sensorineural hearing loss` BOTH fail, so
they are a SYNONYM miss (HPO's own label is "Sensorineural hearing
impairment"), not a case miss — the next deterministic rung is the synonym
table, before any model is invoked.

**⚠ PHENOTYPES NEED THEIR OWN RESOLUTION DOMAIN (operator, confirmed by
measurement).** Collapsing HP with MONDO does not merely strip edges — it
MISROUTES them, which is harder to detect. Measured: **1,169 labels exist in
more than one namespace, 1,129 of them HP+MONDO**. Of 19,738 resolved
phenotype labels, **26.2% landed in MONDO rather than HP**, and **23.7%
(4,678) are genuinely ambiguous** — a collapsed resolver picks by insertion
order, silently reattaching a phenotype edge to a same-named disease node.
HP must be scoped FIRST, with any cross-namespace fallback deliberate.

**UBERON is the capstone, measurably:** 10 collisions across 14,975 UBERON
labels = **0.07%**. Anatomy is near-collision-free exactly where phenotype and
disease are not, so it is the one layer safe to anchor against.

> **⊘ ARCHITECTURAL CONCLUSION CORRECTED (2026-08-20, same day, measured).**
> **The MEASUREMENTS above stand unchanged** — 1,169 multi-namespace labels,
> 26.2% mis-landing, 23.7% ambiguous, UBERON at 0.07%. What is corrected is
> the word **"OWN"**: the sanctioned abstraction already exists and this entry
> did not know it.
>
> `medcare-cohorts/src/quad_tenant.rs` ships `Domain` (8 variants) with a
> classid→domain map, an enforced two-witness contract (classid AND TUI must
> agree), and `FacetRegime::PerRowTui` for the CUI horseshoe:
>
> | Domain | vocabularies mapped |
> |---|---|
> | `Domain::Phenomenology` | HP |
> | `Domain::Anatomy` | **UBERON + FMA** |
> | `Domain::Disease` | MONDO + ICD-10-GM + ORDO + OMIM + DECIPHER |
> | `Domain::Lab` / `Substance` / `Procedure` | LOINC / ATC+RxNorm+Gelbe Liste / OPS |
> | `Imaging`, `BiologicalProcess` | declared, no vocabulary yet |
>
> So the corrected statements are:
>
> - *Phenotype resolution must be constrained to the existing
>   `Domain::Phenomenology`; **HP is its currently populated single-facet
>   vocabulary, not a new resolution domain.***
> - *Anatomy resolution targets `Domain::Anatomy` (UBERON **and** FMA);
>   UBERON's 0.07% collision rate makes it a strong current
>   projection/anchor, **not the semantic domain itself**.*
>
> **Why this pedantry earns its place:** the board is executable archaeology.
> Left as written, "HP needs its own resolution domain" is precisely how a
> later session mints an `HpResolutionDomain` beside the already-built
> `Domain::Phenomenology` — the duplicate-vocabulary failure this workspace
> has now recorded several times. Append-only: the original claim is regraded
> here, not deleted.

---

