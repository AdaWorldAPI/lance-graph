# Keet & Raboanary's FrameworkQFgen — relevance to the Odoo→OGIT retarget

> **Status:** FINDING (2026-05-28). Audit of `toky-raboanary/FrameworkQFgen`
> (HEAD `147a0b9`) + the accompanying ESWC'26 paper "Ontology-Mediated
> Framework for Generating Answerable Questions and Feedback from
> Ontologies" (Raboanary & Keet).
>
> **Triggered by:** user pointer (same session as the two Keet blog posts on
> the OWL-vs-RDF distinction). This is the *third* Keet input — the blog
> posts critiqued my OWL conformance, this paper hands me the validation
> + competency-question tooling.

> **REFRAMING (2026-05-28, user correction this session):** the framework
> is not just "validation tooling" — **it is the missing link between
> psychometry and ontology.** Every axiom-pattern-derived item is
> simultaneously (a) an ontology query and (b) a psychometric measurement
> probe with calibrable difficulty / discrimination / reliability. The
> existing `thinking-engine/cronbach.rs` (Cronbach α) and
> `thinking-engine/calibrate_lenses.rs` (Spearman ρ + ICC) are the
> psychometric instruments waiting to consume the item bank. Read every
> section below through that lens: APS sketches aren't validators, they
> are *test specifications* whose binding-population gives them
> calibrated difficulty curves over the ontology size.
>
> The convergence is structural: 256 = 3σ = 0.9973 (the workspace's
> palette / Gaussian-tail / byte-coverage confluence) is *also* the
> conventional psychometric reliability threshold. Item-bank readiness =
> 99.73% cross-source agreement under the determination algorithm.

## What the framework is

A **content-determination engine** for ontology-driven NLG, sitting between
ingest (parse OWL) and realisation (SimpleNLG / LLM). It takes:

1. **An OWL 2 DL metamodel** that represents *types of questions and
   feedback* as parameterised axiom-pattern sets (APS). Each pattern is a
   DL axiom template — e.g. `C ⊑ D` (taxonomic), `C ⊑ R only D`
   (universal restriction), `C ⊑ R some D` (existential restriction),
   `C DisjointWith D` — with free variables (`C`, `D`, `R`, `X`, `Y`, ...).
   Variables are typed: `VarClass` for class slots, `VarOP` for property
   slots.

2. **A determination algorithm** (Java implementation in the repo) that,
   given (a) an APS + (b) a target ontology, returns *all concrete tuples
   of entities* that produce instances of every axiom in the APS
   simultaneously. Worst-case complexity `O(n^p)` where `p` is the number
   of patterns in the APS; lower bound `Ω(n)`. Crucially: NO
   post-filtering of irrelevant outputs — the matching is exact.

3. **Templates** for natural-language realisation. Each APS is paired with
   a question-template and a feedback-template, both parameterised over
   the same variables. The realiser fills slots from the matched tuples.

## Example from the repo (awo7-What-1-part-1-rel)

APS (one pattern, `p=1`):

    XX ⊑ prop some YY

where `XX`, `YY` are class slots and `prop` is an object-property slot.

Templates:

    Question: What does [XX] [prop]?
    Feedback: [XX] [prop] [YY].

Run against the African Wildlife Ontology (AWO):

    Question: What does a warthog eat?
    Feedback: A warthog eats a root.

Run against EXMO (BFO-based exercise ontology):

    Question: What does a person participate in?
    Feedback: A person participates in sleep.

The same APS exercises both ontologies — no ontology-specific code path.

## Repo structure (key dirs only)

```
Axiom prerequisites/
  AWO/OWL 2 based/         — 10 APS .owl files (awo1..awo10) + Variables.owl + prerequisiteBase.owl
  EXMO/OWL 2 based/        — 6 APS .owl files (exmo1..exmo6)
DeterminationAlgo/         — Java impl (Maven; src/main/java/{Configuration, Associations, CheckingThings, FileManager, MyOnto, ResultAxs, Answers, ...})
Ontologies/                — AWO.owl, exmo.owl (the targets the APS run against)
Results/                   — Raw algorithm output per APS+target pairing
Detailed_Analysis_Determination_Algorithm.pdf — implementation walkthrough
```

`Variables.owl` declares the variable namespace (`#A`, `#A1`, `#X`, `#XX`,
`#Y`, `#YY`, `#OP1`..`#OP5`, `#prop`, `#VarClass`, `#VarOP`,
`#Prerequisite`). Each APS file imports the relevant variables, types them
(`A ⊑ VarClass`, `OP1 ⊑ VarOP`), and lays out the axiom patterns.

## How this applies to the Odoo→OGIT retarget

Three concrete use cases for my POC's downstream:

### Use 1 — Validate the Odoo→OGIT mapping is COMPLETE

Define APS that encode the contracts I expect every retarget artifact to
satisfy. The algorithm returns *missing* tuples (failed matches) =
violations. Two illustrative APS sketches:

**APS-completeness-1: every Odoo `@api.constrains` decorator must produce a
matching `ogit.SDF:ValidationConstraint` entity that constrains the right
attribute on the right OGIT Entity.**

    Pattern P1:  M ⊑ ogit.SDF:ValidationConstraint
    Pattern P2:  M ⊑ ogit.SDF:decoratorPattern value "@api.constrains('A')"
    Pattern P3:  M ⊑ ogit.SDF:constrains some E
    Pattern P4:  E ⊑ ogit:Entity
    Pattern P5:  E ⊑ ogit:mandatory-attributes value A  ∪  ogit:optional-attributes value A

    Free vars: M (the method), A (the constrained attribute), E (the target entity)

    Templates:
      Question: Does the Odoo method [M] correctly constrain attribute [A] on entity [E]?
      Feedback: Method [M] declared @api.constrains('[A]') but [E] has no [A] attribute.

The algorithm returns the (M, A, E) tuples where P1-P4 match but P5 fails.
That's the missing-attribute audit, scalable to all 3555 methods × 388
families.

**APS-completeness-2: every Odoo family must map to a canonical OGIT Entity
(JournalEntry, SalesOrder, ...) — no orphan `odoo:family_name` IRIs.**

    Pattern P1:  F ⊑ odoo:Family
    Pattern P2:  ∃ E . F ogit:original E ∧ E ⊑ ogit:Entity ∧ E ⊑ ogit:Node

    Free vars: F (the Odoo family)

    Templates:
      Question: Does Odoo family [F] map to a canonical OGIT Entity?
      Feedback: Family [F] has no ogit:original → OGIT NTO mapping.

Returns the orphans = the families I'm currently emitting `odoo:account_move`
for instead of reusing `ogit.Accounting:JournalEntry`.

### Use 2 — Generate competency questions for the retarget

Reuse Keet's awo6/awo7/exmo1/exmo2 APS templates directly against my
OGIT-NTO/Accounting subgraph:

    awo6-Definition:
      Question: What is [XX]?
      Feedback: [XX] is [YY], and [ZZ] can be considered as a specialisation of [XX].

    Against ogit.Accounting:JournalEntry:
      Question: What is a JournalEntry?
      Feedback: A JournalEntry is a Node (ogit:Node — its parent), and a SalesInvoice
                can be considered as a specialisation of a JournalEntry.
                (assuming we add SalesInvoice ⊑ JournalEntry)

    awo7-What-1-part-1-rel:
      Question: What does [XX] [prop]?

    Against ogit.Accounting:JournalEntry + ogit.Accounting:hasFiscalCountry:
      Question: What does a JournalEntry have-fiscal-country?
      Feedback: A JournalEntry has-fiscal-country a FiscalJurisdiction.

The 3-hop optimisation doctrine encoded in JournalEntry's
`dcterms:description` becomes mechanically testable: every shortcut verb
gets a competency question whose feedback should mention BOTH the direct
shortcut AND the promoted leaf attribute. If the feedback only mentions
one, the redundancy invariant has degraded.

### Use 3 — Drive the L-doc cross-source validation loop

The L1-L15 documents in `.claude/odoo/L*.md` are curated regulatory /
domain knowledge. The framework would let me, mechanically:

1. Extract competency questions from each L-doc that EXERCISES the Odoo→OGIT
   mapping (e.g. L11-COA-JOURNALS-LOCKDATES.md exercises
   ogit.Accounting:JournalEntry with German HGB §239 lock-date constraints).
2. Auto-generate the APS that those questions imply.
3. Run the determination algorithm against the live OGIT graph.
4. Surface mismatches: the L-doc says "every JournalEntry with state=posted
   must reference a fiscalyear_lock_date that lies in the past" → APS
   pattern → algorithm finds the entities lacking the constraint axiom.

This is exactly the cross-source-validation triangulation that
`E-OWL-IS-THE-UNIVERSAL-INGRESS-1` envisions: D1 (code extraction) ↔ D3
(L-doc curated knowledge) reconciled via D2 (the OGIT axiom layer), with
the determination algorithm doing the heavy lifting on the reconciliation.

## What this changes about the immediate POC scope

`bundles_to_ttl.py` does NOT need to integrate FrameworkQFgen directly —
the framework is OWL-API-based (Java), my emitter is Python, and the
algorithm runs over an already-loaded ontology in JVM memory.

But it DOES change what "OGIT-conformant output" means for the POC:

| concern                                    | implication |
| ---                                        | ---         |
| OWL 2 EL profile target (cdcbc349)         | aligned with FrameworkQFgen's OWL 2 DL metamodel — the determination algorithm runs over our profile |
| Canonical namespace `<http://www.purl.org/ogit/>` | required so the algorithm can match patterns against the canonical entities (JournalEntry etc.) |
| Reuse of existing NTO entries              | required so the APS don't see two parallel hierarchies (odoo:* vs ogit.Accounting:*) and fail |
| `ogit:mandatory-attributes` / `optional-attributes` / `indexed-attributes` RDF lists | required so the APS patterns can match against the entity's declared shape |

In other words: the OGIT-meta-DTO alignment proposed in
`OGIT-META-DTO-ALIGNMENT.md` is **also** what unlocks Keet's framework as
the validation layer. The two are not separate paths — they converge.

## Concrete next step (gated on user direction)

1. **Execute the OGIT-meta-DTO alignment** (proposed but held in the prior
   commit). This produces OGIT-conformant TTL with the proper RDF-list slots.

2. **Write 4-6 APS files** for the Odoo retarget validation, following the
   awo*/exmo* pattern. Suggested starter APS:

   - `odoo1-Family-Has-Canonical-OGIT-Entity.owl`
     "Every Odoo family X has an `ogit:original` pointer to a canonical
     `ogit.<NTO>:` Entity."

   - `odoo2-Method-Constrains-Real-Attribute.owl`
     "Every `@api.constrains('A')` method ValidationConstraint M, where
     M `ogit.SDF:constrains` E, satisfies A ∈ E's attribute set."

   - `odoo3-Compute-Reads-Match-Depends.owl`
     "Every `@api.depends(A1, ..., An)` ComputeFormula M, where M's
     `ogit.SDF:reads_field` set equals {A1, ..., An}."
     (Catches the case where the depends list and the actual body reads
     diverge — D-RPYDTO-5 priority-classifier finding mechanically.)

   - `odoo4-3-Hop-Doctrine.owl`
     "Every entity E that has a promoted leaf attribute A *also* has a
     shortcut verb V such that E V T → T has-attribute A."
     (Tests the JournalEntry 3-hop doctrine across the whole NTO/Accounting.)

   - `odoo5-Causal-Reference-Has-Source.owl`
     "Every `ogit.SDF:causalReference` value (HGB#239 etc.) is matched
     by at least one L-doc citation in the corpus."

3. **Port a stripped-down determination algorithm to Python** for the POC
   path, so it can run in-process against the rdflib graph. Or stand up
   the Java reference impl as a sidecar service if precision matters.

4. **Wire competency-question generation** to a simple LLM realiser that
   turns the matched tuples into NL questions/feedback (no need for
   SimpleNLG; ChatGPT/Claude API can verbalise the matched axioms).

## Open questions

- Does FrameworkQFgen's metamodel translate to OWL 2 EL (our target
  profile), or does it require OWL 2 DL? The 16 APS shipped in the repo
  use `SubClassOf` + `ObjectSomeValuesFrom` + `ObjectAllValuesFrom` +
  `ObjectComplementOf` + `DisjointClasses` — most are EL-conformant, a
  few (the `not` patterns in awo4/awo5/awo10) need DL. For the Odoo
  validation APS, sticking to taxonomic + existential restrictions keeps
  us in EL.

- What's the operational home for the validation runs? A pre-merge CI
  check that loads the OGIT TBox + the freshly-emitted Odoo retarget +
  runs the APS suite? Or a periodic batch run against the live graph?
  (Probably both; CI for schema-level checks, batch for instance-level.)

- Is there a German-fiscal-domain APS library that would be natural to
  reuse — DATEV / SKR04 / Elster / HGB-specific patterns? Or are we
  writing the first one?

## References

- Raboanary, T. H. & Keet, C. M. *Ontology-Mediated Framework for
  Generating Answerable Questions and Feedback from Ontologies.* ESWC'26
  CRC. PDF (in repo): `Detailed_Analysis_Determination_Algorithm.pdf`.
- Repo: https://github.com/toky-raboanary/FrameworkQFgen
- Companion: Keet's "No, an ontology isn't 'just RDF'" (2025-11-15) and
  "Just Turtle and RDF vs OWL examples" (2025-11-17) — see
  `OGIT-META-DTO-ALIGNMENT.md` for the connection.
