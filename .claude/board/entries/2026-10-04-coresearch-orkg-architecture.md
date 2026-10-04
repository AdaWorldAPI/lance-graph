# Coresearch: ORKG ideas → AdaWorldAPI architecture (2026-10-04)

**Status:** EXPLORATION MAP. Nothing is adopted; the operator chooses. Harness
`.claude/agents/coresearch-council.md`. Inputs: `.claude/harvest/orkg-reference-wiring.md`,
`.claude/harvest/indra-reference-wiring.md`. 5 scouts (code, prior art, literature, systems,
concepts) → 18-row crosswalk → 3 co-architects (bridge, firewall/fit, falsifier) → two premise
passes. Raw scout output was banked in the session scratchpad and not committed.

**Question as asked:** which ORKG-adjacent ideas fit our architecture as synergy or expansion,
made concrete as one evidence shape for a literature finding, a cohort statistic, and a
text-derived claim.

**Premise gate pass 1: PREMISE-SPLIT.** A single carrier would fold three uncertainties into
one: sampling (n, CI, p), source support (witnesses, `NarsTruth`, `Quorum` counts), and lexical
frequency. Lexical counts are excluded as claim evidence: they are the parser's reference
frame. The question was re-asked as **Q1** address, **Q2** per-kind carriers, **Q3**
projection. Outside models agree: PICO is the address, STATO the statistic, SEPIO/ECO the
support, PROV-O the lineage, and GRADE only a derived view.

**Outside sources read at least section-deep:**
- **Papers (arXiv):** 2608.01711 (pooling as a gated object with NOT_POOLED), 2602.21410
  (overlap exclusion by bit vectors), 2602.10881 (LLM extraction failure modes), 2006.06348
  (nanopubs), 1401.5775 (trusty URIs), 1305.3506 (micropublications), 2606.15246 (provenance
  worlds), 1806.04185 (PICO spans), 2004.14974 (SciFact).
- **Systems:** Wikibase statement model, metafor escalc/vcalc, OMOP measurement, OpenAlex,
  living meta-analysis platforms.
- **Ontologies:** SEPIO, ECO, PROV-O, Cochrane PICO, STATO, OBCS, IAO/OBI, RO.

| # | idea | bridge | firewall | verdict | probe / reason |
|---|---|---|---|---|---|
| X3 | cohort statistic keyed by address | OPPORTUNITY | PASS | **ADOPT-NOW** (gated by its own acceptance probe) — **2nd premise pass: the key lives in the ADAPTER** (`DiseaseStats` / `panel()` column order), never in the agnostic `Correlation` (`cohort_stats.rs:5-8` divider); operator picks ONE key: loinc_id (external code) or OboAddr (baked lookup, may be None, V1 u24 tail) | P-X3: re-key over `house::cohorts(N)`; kill if any name ↔ loinc_id is non-unique (the live name join at :1016); n/r/rho/p bit-identical |
| X7 | sampling never moves support, support never moves sampling | OPPORTUNITY (rule) | PASS | **PROBE, inside P-X14** — 2nd premise pass: `warrant(self)` takes no n (`ontology_warrant.rs:112`), so a test on `warrant` alone is implied by its signature (vacuous). The 2×2 must run through the composing projection | P-X7: vary n at fixed source count, vary sources at fixed n, both through the projection; kill if either moves the other's quantity |
| X5 | sampling record = (measure, yi, vi) from sufficient statistics; stop discarding SE | OPPORTUNITY (narrow) | CONFLICT — seam: outside the zero-dep contract; derived at read time | **PROBE** (run with P-X3) | P-X5: yi=atanh r, vi=1/(n−3), recompute p; kill if ≠ stored p by >1e-9; refuse n<4, vi≤0 |
| X1 | proposition = minted address; each source's report = one witness edge | OPPORTUNITY | PASS | **PROBE** | P-X1: model R44930 + two seeded cohort draws as address + witness rows; kill "needs its own node" if all fit with no extra fields |
| X14 | comparison = version × ClassView mask × filter, recomputed ("version" = system snapshot; study period is a row FILTER, never the version selector) | OPPORTUNITY | PASS | **PROBE** (carries P-X7) | P-X14: build over two Lance versions; kill "pure projection" if any derived row/table must be written; same version twice ⇒ identical view |
| X9 | a document witness source keyed (namespace, id) | OPPORTUNITY | CONFLICT ×2 | **PROBE + OPERATOR QUESTION** — 2nd premise pass: **PREMISE-WRONG as first posed** ("4th origin / fold into Oracle / DOCUMENT 0x080B" mixes ORIGIN, SOURCE and byte-IDENTITY axes). Re-asked: **Q9a** which existing universe a literature finding belongs to — `Oracle` (doc: "a curated model, a guideline") or `Observed` (doc: "a confirmed document extract"), `provenance.rs:83-89`; a 4th only if neither fits. **Q9b** a new `WitnessSource` variant keyed (namespace, id), coherent with Q9a (`provenance.rs:302-308`). DOCUMENT 0x080B = identity of ingested bytes (wrong axis); `OracleModule(u32)` = closed registry (wrong range) | P-X9: express one DOI witness per R44930 contribution; one DOI cited twice = one source. NAMING: never "Publication" (= Lance commit identity). Neither carrier records publication date (`WitnessEpoch` = system/episode time) |
| X18 | repetition ≠ corroboration on import | OPPORTUNITY w/ X9 | CONFLICT — seam: witness edges on the row (§17) | **PROBE** | P-X18: same DOI twice must not give corroborating=2; two DOIs must give 2. Ties to OPEN spo-witness-carrier (operator) |
| X2 | address coordinates: population / period / comparator | WORTH-EXPLORING | CONFLICT — seam: pre-bake join, minted classids | **PROBE** — 2nd premise pass SPLIT: place/population = an address (baked OboAddr or a V3 concept per KIND, never per instance); **period = a typed interval VALUE under one period concept, never a classid per period, never a Lance version, never a WitnessEpoch** | P-X2: resolve 31 R0 cells (place, period, method); kill "just more classids" if >20% places need a new coordinate kind or period has no existing temporal type (Lance version = system time, not study time) |
| X6 | interval declares its kind | OPPORTUNITY | PASS | **PROBE** | P-X6: kill as decoration if all real inputs carry one kind (check R44930 Bayesian rows for credible intervals) |
| X10 | retraction / supersession / unknown vs no-value vs absent | WORTH-EXPLORING | CONFLICT — seam: versions + witness edges; states as classids | **PROBE** | P-X10: retracted claim must not appear unmarked in the current projection |
| X12 | duplicate-cohort exclusion by bit-vector AND + popcount | WORTH-EXPLORING | PASS (registered partition) | **PROBE** | P-X12: 3 seeded cohorts with overlapping ids + R44930 place×period; kill if period needs unbounded bits or the overlap-free set disagrees with true overlap |
| X4 | text SPO → address resolver (proposal side) | WORTH-EXPLORING | CONFLICT — seam: PROPOSE side, consumer crate | **PARK** until P-X2 | P-X4 after P-X2: kill "lookup join" if <50% of S/O slots map |
| X13 | gated pooling with NOT_POOLED as a valid outcome | WORTH-EXPLORING (later) | CONFLICT — seam: consumer-side float projection | **PARK** until P-X5 | P-X13: kill the pooler if <2 groups share a measure address; kill "unweighted is fine" if it differs from inverse-variance by > pooled SE |
| X8 | evidence kind (manual/automatic) + direction labels | WORTH-EXPLORING | CONFLICT — seam: minted kind on witness edge | **PARK** | P-X8: build only if some input separates "inconclusive" from "silent" |
| X17 | register STATO / bake META_STUDY_SPINE | DROP (blocked) | CONFLICT — operator mint decision | **PARK** | operator: numeric-0 pad fix + 0x03 occupancy |
| X15 | GRADE certainty | DROP as stored | CONFLICT | **SKIP** (projection only, never stored) | D-BBB-NARS-4 |
| X11 | content hash as claim identity | DROP | TRAP | **SKIP** | GUID is the key (OGAR P0); no internal pins. Allowed only: transient dedup proposal, or an external Trusty-URI as an external id |
| X16 | imported worlds do not permeate the core | ALREADY-HAVE | PASS | **SKIP** (already the firewall) | — |

## Productive disagreements (named, not resolved)
1. **X9 document witness.** Firewall: fold into `Oracle` ("bewusst kein viertes"). Bridge: a document witness source. Falsifier: `OracleModule(u32)` would need a registry. Second premise pass: these are different axes. Origin (Q9a) and source (Q9b) are separate questions, and the code's own doc comments disagree on the origin (a guideline is Oracle; a confirmed document extract is Observed). → operator, as Q9a and Q9b.
2. **X5 storage form.** Firewall: store integer sufficient statistics plus the measure classid and derive yi/vi when read. Bridge: return (z, se) from `fisher_z_p`. Open: Pearson r is not an integer statistic. Either store per-pair integer sums (Σx, Σy, Σxy, Σx², Σy², n), or treat r as the record and derive vi from n. P-X5 must decide this.

## Open points (not answered by this run)
- `loinc::address` / `OboAddr` carry a 24-bit identity (MedCare `loinc.rs:80-92`, `crosswalk.rs:136-145`). Reading already-baked addresses is fine. Minting new units this way would conflict with OGAR P0 (V3 4+12 facet).
- R0 has no STATO class, so a domain measure needs its own concept.
- The text → address join (X4) has no internal design at all.
- Quorum's independence assumption is untested on real data (primer §7 gap 6).

## Not searched / not read
- SEPIO LinkML model.
- Fetches that failed: PeerJ nanopub paper (403), metaCOVID preprint (403).
- Not read: COVID-NMA method papers; Semantic Scholar intent docs; GRADE-from-features literature; ORKG r0-estimates and virology-dashboard projects.
- ndarray (full CLAM/CHAODA): not checked out.
- Abstract-only leads: arXiv 2607.15247, 2505.20310, 2512.21727, 2606.17041, 2603.28325, 2608.07202, 2608.30145, 2606.01428, 2608.15909.

## Premise gate
- Pass 1 (on the question): PREMISE-SPLIT → Q1/Q2/Q3; lexical counts excluded as claim evidence.
- Pass 2 (on the options): X9 PREMISE-WRONG → re-asked as Q9a/Q9b; X2 SPLIT (period is a value); X3 SPLIT (adapter, not agnostic struct); X5, X14 SOUND; X7 SOUND but vacuous unless run through the projection.

## First probe
**P-X3 with P-X5's recompute on the same rows.** Offline, synthetic data, existing functions, small. It is the only probe that can expose a live defect, and its output (address-keyed (yi, vi, measure) rows) feeds P-X7, P-X13 and P-X14.

**Operator questions this map raises:**
1. **Q9a:** which existing epistemic universe holds a literature-reported finding, Oracle or
   Observed?
2. **Q9b:** add a `WitnessSource` variant keyed (namespace, id)? It must not be named
   "Publication".
3. **X3 key:** loinc_id or OboAddr for cohort statistics, carried in the adapter.
4. **Which probes to run.** P-X3 + P-X5 is the recommended first probe.

CLOSEOUT | STATUS: open | OUTCOME: exploration map; 1 ADOPT-NOW candidate (X3 adapter key, gated
by its probe), 10 probes, 4 park, 3 skip | OPEN: Q9a, Q9b, X3 key choice, probe
selection
