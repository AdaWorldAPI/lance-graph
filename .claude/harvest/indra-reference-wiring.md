# INDRA — reference wiring (harvest, 2026-10-03)

> **REFERENCE / COMPARATOR — NOT DEPENDENCY, NOT AUTHORITY.**
> Architecture reconnaissance only. No INDRA code, schema or implementation
> is copied into AdaWorldAPI. Public type and API names are cited so the
> comparison is checkable; nothing more.
>
> **READ BY:** anyone designing the seam *observation/text → normalized
> evidence-bearing causal assertion → reasoning/model*; anyone touching
> `deepnsm-v2` lexical evidence, `jc`, MedCare `reinforcement.rs`,
> `lance-graph-arm-discovery`, `CausalEdge64` witness/truth bits, OGAR
> `ogar-obo`, or an interoperability export to INDRA-shaped statements.

## 0. Sources and revisions

All claims below were read from source at these revisions. Line numbers are
relative to each repo root at that commit.

| repo | commit | date | license (verified in `LICENSE`) |
|---|---|---|---|
| `gyorilab/indra` (also mirrored as `AdaWorldAPI/indra`, same HEAD) | `7ae3337dcae88baed918d7ae6af2ddc21f9946fa` | 2026-06-30 | BSD-2-Clause |
| `gyorilab/indra_db` | `7dc8bf501407c51f36c0746595d8e63d44d4a671` | 2026-07-09 | **GPL-3.0** |
| `gyorilab/gilda` | `33793ab366de5550ba24cd95064644870badd462` | 2026-06-29 | BSD-2-Clause |
| `gyorilab/emmaa` | `fbfd99fea137f2c510d128c16375af0e230dc31b` | 2024-07-19 | BSD-2-Clause |
| `gyorilab/indra_world` | `bc909e0added428a6110bc764130e885e54a3ead` | 2023-02-27 | BSD-2-Clause |
| `gyorilab/mira` | `b1fb21cd91f531efdc961e0d0187ef0a014e55cf` | 2026-08-31 | BSD-2-Clause |

**License rule for this workspace.** The four BSD repos may be studied and
cited. `indra_db` is GPL-3.0 and differs from the rest of the ecosystem: its
schema and code are **not** to be transplanted in any form. Only its table and
column *names* are cited below, as identifiers. Any future persistence design
here is an independent implementation.

AdaWorldAPI side read at: lance-graph `06061fe`, MedCare-rs `29116c5`, OGAR
`a37383e`. `ndarray` is **not present** on this filesystem, so every
`ndarray::*` capability mentioned below is cited only from lance-graph
comments and is **UNVERIFIED** here.

Status vocabulary for AdaWorldAPI claims: **SHIPPED** (in source on the
checkout above), **PLANNED** (in plans/docs only), **ABSENT** (searched, with
the search space named).

## 1. Scope

INDRA is harvested as a reference architecture for knowledge assembly,
evidence-bearing causal assertions and executable model construction.
It is not adopted as an implementation dependency or semantic authority.

ORKG is primarily useful as a reference for structured scholarly
contributions and comparison views; INDRA is the closer comparator for the
reasoning/assembly layer beneath those views.

*(No ORKG harvest exists yet in `.claude/harvest/`; that line states the
intended division of labour, not a completed comparison.)*

The comparison is made at the seam
**"observation/text → normalized evidence-bearing causal assertion →
reasoning/model"**, not at the bookkeeping or UI layer.

## 2. INDRA system map

```
TEXT (REACH, TRIPS, Sparser, Eidos, …)   DATABASES (BioPAX, BEL, SIGNOR, …)   CURATED KB
        │  indra/sources/*               │  indra/sources/*                      │
        └───────────────┬────────────────┴───────────────────────────────────────┘
                        ▼
                 raw Statements            one Evidence each (typically)
                        │
                        ▼   GROUNDING      indra/preassembler/grounding_mapper/
          grounding map → misgrounding map → Adeft/Gilda disambiguation
                        │  Gilda (separate repo) = lexical→(ns,id) ranking
                        ▼   NORMALIZATION
          map_sequence (site fixing) · normalize_equivalences / _opposites
                        │
                        ▼   PREASSEMBLY    indra/preassembler/__init__.py
        ┌───────────────┼──────────────────────────┐
   combine_duplicates   combine_related            find_contradicts
   (matches_key group,  (RefinementFilter chain,   (pairwise, detection only,
    evidence lists       supports / supported_by)   not used by belief)
    concatenated)
        └───────────────┼──────────────────────────┘
                        ▼   BELIEF         indra/belief/__init__.py, skl.py
          SimpleScorer / BayesianScorer / CountsScorer / HybridScorer
          set_prior_probs → set_hierarchy_probs (evidence of refinements pooled)
                        │
                        ▼
      normalized evidence-bearing Statements  (+ belief, + support graph)
                        │
      ┌─────────────────┼──────────────────────┬─────────────────────┐
      ▼                 ▼                      ▼                     ▼
  IndraNet          PyBEL / CX / SIF       PySB / KAMI            English / HTML
  causal graph      causal graph           executable rules       human-readable
  (belief, hashes,  (exchange formats)     (UUID annotation only)
   source_counts on edges)
                        │
                        ▼   MODEL CHECKING  indra/explanation/model_checker/
         "does model M explain Statement S?"  signed-path search
```

Ecosystem around the core:

| layer | component | what it adds |
|---|---|---|
| extraction | `indra/sources/*` | readers and DB processors produce raw `Statement`s |
| lexical grounding | **Gilda** | surface text → ranked `(ns, id)` candidates, context classifiers |
| representation | `indra/statements` | the Statement algebra (§3) |
| assembly | `indra/preassembler` | dedup, refinement, contradiction detection (§6) |
| epistemic scoring | `indra/belief` | source-error belief model (§7) |
| persistence | **INDRA DB** (GPL) | raw ↔ preassembled store, support links, readonly query schema (§8) |
| model generation | `indra/assemblers/*` | graph and executable projections (§9) |
| model checking | `indra/explanation` | path-based explanation of Statements (§10) |
| continuous maintenance | **EMMAA** | daily literature → reassembly → test → delta (§11) |
| domain transfer | **INDRA World** | causal/event statements outside biology (§12) |
| dynamical models | **MIRA** | template models → ODE/Petri/AMR (§9.3) |
| services | `rest_api/api.py`, `indra_db_service/api.py` | REST over sources, preassembly pipeline, assemblers, DB queries |

## 3. Statement algebra

### 3.1 The data contract

```
Statement                                   indra/statements/statements.py:252
├─ evidence: [Evidence]                     :270-280  (list; many papers/readers per Statement)
├─ supports / supported_by: [Statement]     :283-284  (refinement links, §6)
├─ belief: float = 1                        :285      (ON THE STATEMENT, not on Evidence)
├─ uuid                                     :286
├─ _agent_order  (role names)               :268, per subclass
├─ matches_key()  → identity                abstract :290
└─ get_hash(shallow=True|False)             :296
      shallow = make_hash(matches_key, 14)            (knowledge identity)
      full    = make_hash(matches_key + sorted evidence keys, 16)

Agent(Concept)                              indra/statements/agent.py:19
├─ name, db_refs {ns: id | [(id, score)…]}  concept.py:20
├─ mods [ModCondition]                      agent.py:482
├─ mutations [MutCondition]                 :402
├─ activity ActivityCondition               :602
├─ bound_conditions [BoundCondition]        :345
└─ location
   entity_matches_key = preferred grounding (default_ns_order, :15) else name   :78
   state_matches_key  = mods, mutations, activity, location, bound conditions   :92
   matches_key        = (entity key, state key)                                 :72

Evidence                                    indra/statements/evidence.py
├─ source_api, source_id, pmid
├─ text_refs {PMID, PMCID, DOI, URL, …}
├─ text            (source sentence)
├─ annotations {}  (reader-specific; preassembly adds agents.raw_text,
│                   agents.raw_grounding, prior_uuids)
├─ epistemics {}   (negated, hypothesis, direct, section_type, …)
├─ context  BioContext | WorldContext
├─ source_hash     hash(source_api, source_id, text|pmid)
└─ stmt_tag        full hash of owning Statement
```

### 3.2 Identity: what participates and what does not

| participates in `matches_key` (identity) | does NOT participate |
|---|---|
| statement class (`stmt_type`) | evidence of any kind |
| each agent's **grounded entity** (preferred ns/id, else name) | belief |
| each agent's **state**: mods, mutations, activity, location, bound conditions | supports / supported_by |
| type-specific fields: residue/position (Modification, `:654`), `obj_activity` + `is_activation` (RegulateActivity, `:1051`), polarity count + overall polarity (Influence, `:2005`) | context, epistemics, text refs |

Consequences:

- Agent **state is identity**, not a qualifier. `MAP2K1` and
  `MAP2K1(phospho S218)` are different agents.
- **Negation is not identity.** `negated` lives in `Evidence.epistemics`, so
  a negated and an affirmed extraction of the same claim merge into **one**
  Statement, and negation only enters at belief time (§7).
- Hypothesis and directness are evidence-level too. They are filtered by
  pipeline steps (`filter_no_hypothesis`, `filter_direct`), not by identity.
- `Influence` identity deliberately matches overall polarity, so `+/+` and
  `-/-` collapse together (`:2005` comment).

### 3.3 Roles

Roles are a per-class ordered attribute list (`_agent_order`): `enz/sub`
(Modification), `subj/obj` (RegulateActivity, RegulateAmount, Influence),
`members` (Complex, Association), `subj/obj_from/obj_to` (Conversion),
`concept` (Event). There is no generic subject/predicate/object triple: the
**class is the predicate** and roles are positional fields.

### 3.4 The cited classes

| class | role fields | identity extras | contradicts |
|---|---|---|---|
| `Phosphorylation(AddModification)` | enz, sub | residue, position | its inverse mod class at the same site (`:704`) |
| `Activation` / `Inhibition` (`RegulateActivity`) | subj, obj | obj_activity, is_activation | each other, entities equal or refinements (`:1079`) |
| `IncreaseAmount` / `DecreaseAmount` (`RegulateAmount`) | subj, obj | — | each other (`:1890`) |
| `Complex` | members (sorted) | — | — |
| `Influence` | subj, obj (both `Event`) | polarity count, overall polarity | opposite polarity on compatible concepts, or same polarity with one concept opposite (`:2018`) |
| `Event` | concept | delta (`QualitativeDelta` / `QuantitativeState`) | — |
| `Association` | members (Events) | — | `:2187` |

`QuantitativeState` (`delta.py:84`) carries `entity, value, unit, modifier,
text, polarity`: a *reported quantity*, not a statistic.

### 3.5 Answers

- **Identity** = shallow hash of `matches_key` (`:296`).
- **Uncertainty** lives on the Statement (`belief`); its inputs live on
  Evidence (`source_api`, `epistemics.negated`, reader rule subtype).
- **One Statement, many sources**: yes. `evidence` is a list, and
  preassembly concatenates lists (§6.1).

## 4. Evidence and provenance, end to end

Example: two papers extract `Phosphorylation(MAP2K1, MAPK1, T, 185)`.

| step | what exists | provenance state |
|---|---|---|
| reader / DB processor | raw Statement, one `Evidence(source_api='reach', pmid=…, text=…, annotations.found_by=…, epistemics=…)` | complete |
| grounding map + disambiguation | same Statement; `db_refs` rewritten; Gilda scores written to `evidence[0].annotations['agents']['gilda']` (`disambiguate.py:150`) | **raw `TEXT` kept in `db_refs`**; alternatives survive only as scores in an annotation |
| `combine_duplicates` (`preassembler/__init__.py:97`) | one Statement, evidence = concatenation; exact-duplicate evidence dropped by `ev.matches_key() + raw_text + raw_grounding` | each Evidence gets `annotations.agents.raw_text`, `raw_grounding`, `prior_uuids` — **the original grounding of each raw extraction survives normalization** |
| `combine_related` | general Statement linked to specific ones | per-Statement evidence lists unchanged |
| `flatten_evidence` (optional, `:760`) | specific Statement gains evidence of its supporters, tagged direct vs from-supporters | traceable |
| belief | `stmt.belief` set | belief is a function of the evidence list; recomputable |
| INDRA DB | `raw_statements` row (one evidence each) → `raw_unique_links(raw_stmt_id, pa_stmt_mk_hash)` → `pa_statements(mk_hash)` | lossless: every preassembled hash enumerates its raw rows |
| IndraNet edge | `statements=[{stmt_hash, evidence_count, belief, source_counts, …}]` (`assemblers/indranet/assembler.py:487`) | **hash + counts only**; evidence recoverable by lookup of the hash, not carried |
| PySB rule | `Annotation(rule, stmt.uuid, 'from_indra_statement')` (`assemblers/pysb/assembler.py:1022`) | **UUID only**; requires the original in-memory statement list |
| MIRA template (via SIF) | `provenance=[]`; `class Provenance: pass` (`mira/metamodel/templates.py:973`) | **lost** |

**Answer.** Up to and including INDRA DB and IndraNet, an assembled causal
edge stays traceable to every original Evidence item, by hash. The lossy
boundaries are:

1. **Statement → PySB rule**: rule carries a UUID, not a hash; UUIDs are not
   stable across reassembly, so traceability needs the same in-memory run.
2. **Signed/unsigned graph collapse**: many Statements become one edge;
   per-statement hashes are listed but evidence is summarized to counts.
3. **SIF → MIRA template**: provenance absent.

## 5. Grounding and ontology normalization

### 5.1 Pipeline

```
surface text "ERK" (Agent.db_refs['TEXT'])
  → GroundingMapper.map_agent (mapper.py:218)
      1. agent_map  (whole Agent replacement by text)
      2. grounding_map  (curated text → db_refs)
      3. misgrounding_map  (remove known-bad groundings)
  → DisambManager (disambiguate.py)
      Adeft model if one exists for this exact text, else Gilda model;
      context = full text / abstract / evidence sentence
  → standardize_db_refs (ontology mappings, e.g. MESH ↔ HGNC)
  → Agent.get_grounding() = first ns in default_ns_order (agent.py:15)
  → entity_matches_key = str((ns, id))
```

Gilda itself (separate repo): `normalize` (`process.py:71`) → variant
generation (Greek letters, Roman↔Arabic numerals, depluralization) → exact
dict lookup → `filter_for_organism` → `score = (status·2 + string_score)/9`
(`scorer.py:251`; status curated 4 > name 3 > synonym 2 > former_name 1) →
`_merge_equivalent_matches` (subsumed terms kept) → optional disambiguation
(classifier probability multiplied into the score) → sorted `ScoredMatch`
list. Namespace priority is a **tie-breaker only**.

### 5.2 How the hard cases are handled

| case | handling | file |
|---|---|---|
| aliases | Gilda terms with status `synonym`/`former_name` | gilda `term.py:20` |
| ambiguous surface form | ranked list; Adeft/Gilda classifier in context; **then one winner is written into `db_refs`** | `disambiguate.py:150` |
| multiple namespaces | `db_refs` keeps all; identity uses the first in `default_ns_order` | `agent.py:695` |
| ontology equivalence | `normalize_equivalences(ns)` rewrites to one representative | `preassembler/__init__.py:534` |
| hierarchy | `IndraOntology.isa` / `partof` / `isa_or_partof` over a networkx graph with transitive closure | `ontology/ontology_graph.py:147-195, 669` |
| opposite concepts | `is_opposite`, `normalize_opposites(ns)` flips statement polarity | `ontology_graph.py:548`, `preassembler/__init__.py:553` |
| failed grounding | Agent keyed by **name**; DB pipeline moves it to `discarded_statements(reason='grounding')` | `concept.py:34`; indra_db `preassemble_db.py:625` |
| competing groundings | Gilda list ranked; INDRA keeps the winner + score annotation | — |
| grounding confidence | Gilda score in [0, 1]; **a heuristic, not a calibrated probability**; not used by belief | gilda `scorer.py:251` |

### 5.3 Cross-map (lexical disambiguation is NOT ontology grounding)

INDRA collapses two distinct steps into one write: choosing **which lexical
reading** a surface form has, and choosing **which ontology node** it denotes.
The workspace keeps these apart.

| INDRA | AdaWorldAPI | relation | status |
|---|---|---|---|
| `db_refs['TEXT']` + Gilda candidate list | `deepnsm-v2` `LexicalEvidence` / `LexicalReading` — every COCA reading kept with integer counts (`lexical.rs:34-41, 111, 332`) | ANALOGOUS, ours keeps all readings structurally | SHIPPED (lexicon); the FSM does **not** consume it yet |
| winner-takes-all write into `db_refs` | — (planned `PosSet` / reading mask) | ours STRICTLY RICHER *if built* | `PosSet` **PLANNED/deferred** (`.claude/plans/deepnsm-v2-lexical-evidence-consumer-v1.md:306-308`) |
| `(ns, id)` preferred grounding | OGAR classid / `ogar-obo::Namespace` (`OGAR/crates/ogar-obo/src/lib.rs:95-178`) | ANALOGOUS | SHIPPED |
| `IndraOntology.isa_or_partof` | `ogar-obo/src/reason.rs` EL saturation: is_a transitivity, transitive part_of, existential filler subsumption (`saturate :171`, `ancestors :369`); `ogar-ro` IS_A / PART_OF (`lib.rs:153-154`) | SAME role (refinement anchor); ours is a saturating reasoner | SHIPPED |
| `is_opposite` | — no opposite relation found in `ogar-obo` / `ogar-ro` | NO MATCH | ABSENT in those two crates |
| MONDO / FMA / LOINC | `ogar-obo` namespaces, `ogar-fma`, MedCare LOINC in `medcare-analytics` | ANALOGOUS (different ontologies) | SHIPPED |
| Palette / codebook identity, ClassView, ontology rails | no counterpart; INDRA identity is a string key | ORTHOGONAL | SHIPPED (contract) |

## 6. Preassembly mechanics

### 6.1 Duplicates

`Preassembler.combine_duplicate_stmts` (`preassembler/__init__.py:97`):
sort by `matches_fun` (default `matches_key`), `itertools.groupby`, keep a
generic copy of the first Statement, append every distinct Evidence of the
group. Two Statements are duplicates **iff their matches keys are equal**,
i.e. same class, same grounded entities, same agent states, same
type-specific fields. Evidence is **not reduced**; it is accumulated as a list.
Cost: one sort, O(N log N).

### 6.2 Refinement

`refinement_of(other, ontology)` is defined per class. For `RegulateActivity`
(`statements.py:1063`): same class, same activation polarity, subject and
object each `refinement_of` the other's, and `obj_activity` equal or `isa` in
`INDRA_ACTIVITIES`. `Agent.refinement_of` (`agent.py:135`) requires the entity
to match or `isa_or_partof`, **and** every bound condition / mod / mutation /
activity / location of the general agent to be matched by the specific one.
So **ontology ancestry and agent state both participate**.

Candidate generation avoids N×N: `OntologyRefinementFilter`
(`refinement.py:179`) indexes each statement type by agent key per role,
then for each statement asks the ontology for ancestor/descendant keys and
intersects the hash sets role by role. `RefinementConfirmationFilter` runs the
actual `refinement_of` only on survivors.

Representation: relation tuples `(refiner, refined)` (`_generate_id_maps`,
`:385`). Then `specific.supported_by.append(general)` and
`general.supports.append(specific)` (`combine_related`). Top-level =
statements with **no `supports`**, i.e. the most specific ones. The relation
is therefore **both** a stored link on the objects and (in INDRA DB and
`BeliefEngine`) a materialized DiGraph of hashes (`build_refinements_graph`,
`belief/__init__.py:756`, edges specific → general, cycle-checked).

There is no separate *generalized claim* object. Generality exists only as
"a less specific Statement that happens to have been extracted, linked into
the DAG". INDRA never synthesizes a new general Statement from specific ones.

### 6.3 Contradictions

`Preassembler.find_contradicts` (`:435`): pairwise `itertools.product` over
opposite type pairs (each `AddModification` vs its inverse, `Activation` vs
`Inhibition`, `IncreaseAmount` vs `DecreaseAmount`) and
`itertools.combinations` within `Influence` and `ActiveForm`. It returns a list
of pairs. That is all: **detection only**. No belief update reads it, no
statement is removed, both sides survive. It runs over `self.stmts` (the raw
list), not over unique statements.

### 6.4 Scale

- Duplicates: O(N log N).
- Refinement: filter-pruned, but still pairwise in the worst case within an
  ontology neighbourhood; the confirm filter counts comparisons
  (`_comparison_counter`). INDRA DB's initial build is an outer×inner batch
  loop over the upper triangle, O(N²/B²) batch pairs with checkpoints
  (`preassemble_db.py:367`); incremental supplementation compares new
  statements against all old ones in chunks (`:594`). The newer dump pipeline
  abandons in-DB supplementation for a full file-based rebuild
  (`readonly_dumping/export_assembly.py:795`).
- Contradictions: O(P·N) per opposite pair, unfiltered.

These are exactly the pressures the workspace answers with population masks
(§18).

## 7. Belief semantics

### 7.1 SimpleScorer, exactly

`SimpleScorer.score_evidence_list` (`belief/__init__.py:136`). For a list of
evidence partitioned by `source_api` into sources *s*, each with systematic
error rate `syst[s]` and per-evidence random error rate `rand(e)` (by source,
or by reader-rule subtype via `tag_evidence_subtype`, `:710`):

```
P_pos = 1 − Π_s ( syst[s] + Π_{e ∈ s, ¬negated} rand(e) )
P_neg = 1 − Π_s ( syst[s] + Π_{e ∈ s,  negated} rand(e) )
belief = P_pos · (1 − P_neg)
```

Assumptions, explicitly:

1. Each source fails either systematically (all its evidence wrong together,
   probability `syst`) or per evidence (`rand`, independent).
2. **Sources are independent** of each other.
3. Evidence items within a source are independent given no systematic error.
4. Negated evidence is scored as an *independent* claim of the negation and
   multiplies the positive belief by its complement.
5. Priors are fixed defaults (`resources/default_belief_probs.json`,
   37 sources; e.g. `reach` rand 0.3 / syst 0.05, `biopax` 0.2 / 0.01).

`BayesianScorer.update_probs` (`:296`) turns curated `[correct, incorrect]`
counts per source into `rand = 1 − min(p/(p+n), 0.95) − 0.05`, with `syst`
fixed at 0.05: a point estimate, not a posterior distribution.
`CountsScorer` / `HybridScorer` (`belief/skl.py`) fit any sklearn classifier on
source counts, statement type, number of PMIDs, evidence length, etc.

### 7.2 Hierarchy and inference

- `set_hierarchy_probs` (`:422`): a statement's belief is computed over its
  own evidence **plus the evidence of all more-specific statements** in the
  refinement graph (`get_ev_for_stmts_from_supports`). Specific evidence
  supports the general claim; never the reverse.
- `set_linked_probs` (`:522`): a MechLinker-inferred statement gets the
  **product** of its sources' beliefs.
- Curation: `filter_by_curation(..., update_belief=True)` sets belief to 1
  for statements curated correct, and drops incorrect ones by policy
  (`tools/assemble_corpus.py:1703`). Curation keys are
  `(pa_hash, source_hash)`, i.e. evidence-level.

### 7.3 What belief *is*

**The probability that the Statement is a correct extraction / assertion,
given which sources and rules produced it.** It is a model of *source
reliability*. It contains no term for empirical effect strength, sample size,
or study design. Two RCT-grade evidences and two co-occurrence sentences from
the same reader count identically.

### 7.4 Cross-map

| INDRA | AdaWorldAPI | relation | note |
|---|---|---|---|
| belief ∈ [0,1], P(correct) | `NarsTruth{frequency, confidence}` (`lance-graph-contract/src/exploration.rs:89`) | **ORTHOGONAL axes** | NARS *frequency* = proportion of positive evidence about the proposition; *confidence* = amount of evidence. INDRA belief conflates "how many sources" (≈ confidence) with "how reliable" and has no frequency of the proposition itself. |
| `1 − Π(syst + Π rand)` | NARS revision `w = c/(1−c)`, `f = Σ f_i w_i / Σ w_i`, `c = W/(W+1)` (`exploration.rs:110-120`; also `lance-graph-planner/src/nars/truth.rs:57-69`) | ANALOGOUS (both accumulate independent evidence) | INDRA saturates by multiplying error probabilities; NARS saturates by evidence weight. NARS keeps negative evidence *inside* frequency; INDRA multiplies a separate negated score. |
| negated evidence → `·(1 − P_neg)` | negative evidence lowers `f` at constant evidence mass | ANALOGOUS, NARS STRICTLY RICHER | NARS can represent "well-evidenced, mostly negative" (f low, c high); INDRA collapses it to "low belief", indistinguishable from "barely evidenced". |
| source independence | `revision.rs` `EvidenceMask` — echoes and closed cycles add **zero** weight (`lance-graph-contract/src/revision.rs:31`, tests `:520-659`) | ours STRICTLY RICHER on dependence | INDRA has no echo detection: a review quoting a primary paper counts as a second evidence. Note: `revision.rs` is a policy surface with "no production write capability" (its own header). |
| reader-rule subtype priors | `lance-graph-arm-discovery` `arm_to_truth_u8`: f = cooccur/antecedent, c = m/(m+k) (`translator.rs:68-86`) | ANALOGOUS | count → confidence, the only such mapping in the workspace |
| hierarchy pooling | — | NO MATCH yet | no shipped code pools evidence of refinements into an ancestor |
| Fisher-z / Pearson / α / ICC (`crates/jc`) | — in INDRA | NO MATCH | INDRA has no statistical inference in belief |
| `CausalEdge64` u8 f/c (`causal-edge/src/edge.rs:140-151`) | belief float | ANALOGOUS carrier | no evidence-count field on the edge; v2 W slot is a 6-bit witness corpus root (`layout.rs:52`) |
| JC reliability/validity | — | NO MATCH | |

The central distinction holds: **"probability that a statement is correct"**
(INDRA) and **"accumulated frequency/confidence of observations supporting a
proposition"** (NARS) are different epistemologies. One scores the
*messengers*; the other scores the *message*. They compose (source reliability
could discount NARS evidence weight before revision) but are not
substitutable.

## 8. INDRA DB (GPL-3.0 — names only)

```
raw_statements (one evidence per row; mk_hash, source_hash, text_hash,
                reading_id | db_info_id, json)
     │  distill (exact-duplicate removal), grounding, sequence mapping
     │  rejected → discarded_statements(reason)
     ▼
raw_unique_links (raw_stmt_id → pa_stmt_mk_hash)        lossless provenance
     ▼
pa_statements (mk_hash = shallow hash; matches_key; json)
     ├─ pa_agents / pa_mods / pa_muts / pa_activity   grounding index by (db_name, db_id, role)
     ├─ pa_support_links (supporting_mk_hash, supported_mk_hash)
     └─ curation (pa_hash, source_hash, tag, curator, …)   evidence-level curation
     ▼  rebuilt as a separate readonly schema (22 ordered steps)
readonly: belief(mk_hash), evidence_counts, pa_stmt_src (per-source counts),
          source_meta, name_meta / text_meta / other_meta (denormalized
          per-agent lookup with ev_count + belief), agent_interactions,
          fast_raw_pa_link (raw json joined to pa json per evidence)
```

- **Identity**: preassembled key = INDRA's shallow hash; raw uniqueness =
  `(mk_hash, text_hash, reading_id)` for reader output, `(mk_hash, source_hash,
  db_info_id)` for databases.
- **Belief** is not in the principal schema; it is computed offline
  (`indra_db/belief.py`, memory-light mock statements; newer
  `export_assembly.calculate_belief`) and loaded into the readonly schema.
  No curation-aware belief in the legacy path.
- **Incremental update**: `DbPreassembler.supplement_corpus` — new raw ids =
  those not in `raw_unique_links`, condensed against the in-memory set of
  existing hashes, then compared against all older PA statements in chunks.
  The current dump pipeline does a full rebuild instead.
- **Query model**: a boolean query algebra (`HasAgent`, `HasType`,
  `HasSources`, `FromPapers`, `HasEvidenceBound`, … combined with `&`, `|`,
  `~`; `client/readonly/query.py`) sorted by evidence count or belief; REST in
  `indra_db_service/api.py` (`/statements/from_agents`, `/from_hash`,
  `/curation/submit/<hash>`, …).

The useful lesson is architectural, not code: **raw and preassembled
statements are separate tables joined by a link table**, so assembly is a
view over observations that can be rebuilt. That maps onto the Lance
version model (§13, row "INDRA DB raw Statement").

## 9. Assemblers

### 9.1 Per assembler

| assembler | input | transformation | dropped | provenance | belief | simulable | round trip |
|---|---|---|---|---|---|---|---|
| **IndraNet** (`assemblers/indranet/assembler.py`) | Statements | signed: Activation/Inhibition/Increase/Decrease (+Influence; Conversion split) re-preassembled by agent names + polarity, `MultiDiGraph` keyed by sign; digraph: Complex → pairwise edges | agent states, residues (kept as edge data only), evidence text | `statements=[{stmt_hash, evidence_count, source_counts, belief}]` per edge | edge belief re-scored over pooled evidence (`simple_scorer`) or `1 − Π(1 − b_i)` | no | by hash lookup |
| **PyBEL** | Statements | BEL graph | — | BEL citations per edge | — | no | partial (BEL → INDRA processor exists) |
| **PySB** (`assemblers/pysb/assembler.py:526`) | Statements | policy dispatch `{stmttype}_{monomers|assemble}_{policy}` (`:865`): `one_step`, `two_step`, `interactions_only`, `multi_way`, `atp_dependent`, `michaelis_menten`, `hill`; `BaseAgentSet` → Monomers with sites/states | evidence, belief, context | **rule annotation = statement UUID only** | none | **yes** (ODE / SSA / Kappa) | lookup only (`stmt_from_rule`), no synthesis |
| **KAMI**, **CX**, **SIF**, **CyJS** | Statements | graph/exchange formats | varies | varies | CX can carry belief | no | no |
| **English** | Statements | natural-language sentences | everything structural | — | — | no | no |

### 9.2 Where causal language becomes executable mathematics

Trace `Phosphorylation(MAP2K1, MAPK1, 'T', '185')`, policy `one_step`
(`assembler.py:1210`):

```
Statement
  → BaseAgentSet: Monomer MAPK1 gains site T185 with states ('u','p')
  → Rule  MAP2K1() + MAPK1(T185='u') >> MAP2K1() + MAPK1(T185='p')
  → Parameter kf_mm_phosphorylation_1 = 1e-6        (placeholder constant)
  → Annotation(rule, stmt.uuid, 'from_indra_statement')
```

`Activation` `one_step` (`:1652`): `subj + obj(act='inactive') >> subj +
obj(act='active')`. `michaelis_menten` builds a `kcat/(Km + obs)`
Expression.

**Answer.** The transition happens inside the **assembler policy**, at rule
creation: a typed relation becomes a mass-action (or Michaelis–Menten / Hill)
rate law with **placeholder** parameters. Kinetic constants are not inferred
from evidence; belief is not mapped onto any parameter. The executable model
encodes *topology and mechanism type*, not *quantities*.

### 9.3 MIRA

MIRA's `TemplateModel` (`metamodel/templates.py`, `template_model.py`) with
`ControlledConversion`, `NaturalConversion`, `*Production`, `*Degradation`,
`*Replication`, optional sympy `rate_law`, exports to ODE / Petri / AMR /
SBML. There is **no direct INDRA Statement → template path** in this
checkout; the nearest bridge is `sources/sif.py template_model_from_sif_edges`
(POSITIVE → `ControlledReplication`, NEGATIVE → `ControlledDegradation`), and
provenance is empty there.

## 10. Model checking and mechanistic reasoning

- `ModelChecker.check_statement` (`explanation/model_checker/model_checker.py:278`)
  asks: **is there a sign-consistent path from subject to object in the model
  graph?** Nodes are split into `(n, 0)` and `(n, 1)`; positive edges keep
  sign, negative edges flip it (`signed_edges_to_signed_nodes`, `:580`).
  Target polarity is 1 for Inhibition / Decrease / RemoveModification /
  negative Influence. Refinement agents count as matches. Search is BFS,
  bounded by `max_path_length` (default 5) and `max_paths` (default 1).
- Subclasses: `PysbModelChecker` (Kappa influence map, pruned; `score_paths`
  for data), `PybelModelChecker`, `SignedGraphModelChecker`,
  `UnsignedGraphModelChecker`.
- `PathResult`: `path_found`, `result_code` (`PATHS_FOUND`, `NO_PATHS_FOUND`,
  `MAX_PATH_LENGTH_EXCEEDED`, `STATEMENT_TYPE_NOT_HANDLED`, …), `paths`,
  `path_metrics`. Paths map back to Statements via `explanation/reporting.py`.
- **MechLinker** (`mechlinker/__init__.py:12`) infers new Statements from
  combinations (Modification + ActiveForm → Activation, etc.), returning
  `LinkedStatement(source_stmts, inferred_stmt)`, so inferences keep their
  premises; belief = product of premise beliefs.
- A model **contradicting** a Statement is not represented. A failed check is
  "no path", not "a path of the opposite sign". There is no counterfactual,
  no intervention semantics beyond sign reachability, and no confounder notion.

| INDRA | AdaWorldAPI | relation | status |
|---|---|---|---|
| signed BFS path explanation | `causal-edge/src/network.rs` `forward_chain :61`, `causal_query :91`, `evidence_trail :152` | ANALOGOUS | SHIPPED |
| — (no intervention) | Pearl `CausalMask` 2³ association/intervention/counterfactual/confounder (`causal-edge/src/pearl.rs`), `counterfactual :171`, `detect_simpsons_paradox :120`, `lance-graph-planner/src/dismech_counterfactual.rs` | ours STRICTLY RICHER in rung | SHIPPED |
| `LinkedStatement(sources, inferred)` | NARS deduction/abduction/induction (`lance-graph-planner/src/nars/truth.rs:75-107`) | ANALOGOUS; INDRA multiplies beliefs, NARS has per-rule truth functions | SHIPPED |
| failed check | "epistemic pothole" | NO MATCH | pothole exists only in doc comments (`causal-edge/src/layout.rs:326`, `recipe_vocab.rs`) |
| path result per test | JC proof pillars (`crates/jc`) | ORTHOGONAL (JC proves substrate properties, not model claims) | SHIPPED |

## 11. EMMAA: machine-maintained models

```
daily trigger (CloudWatch, 12pm EST)            emmaa/aws_lambda_functions/README.md
  → search_literature(search_terms, date_limit) → readers / indra_db
  → extend_unique (raw statements accumulate by full hash)   model.py:255
  → AssemblyPipeline(config['assembly']) — FULL reassembly each day
       e.g. filter_no_hypothesis → ground → map_grounding → map_sequence
            → run_preassembly → filter_by_curation(update_belief) → filter_top_level
  → ModelManager per mc_type (pysb | pybel | signed_graph | unsigned_graph)
  → run_tests (StatementCheckingTest = one Statement; corpora on S3;
               other models' statements can become this model's tests)
  → ModelStatsGenerator / TestStatsGenerator: added/removed stmt hashes,
    newly passed/failed tests, curation stats
  → notifications (email, tweets)
```

- **A model** = `EmmaaModel` (`model.py:42`): `stmts` (`EmmaaStatement`:
  statement + date + search terms + metadata including `internal`), configs
  for reading/assembly/tests/queries, search terms, export formats.
- **Tests / questions**: a test is one Statement checked by path search;
  queries add `PathProperty`, `SimpleInterventionProperty`, `DynamicProperty`
  (stochastic simulation + temporal pattern satisfaction, queries only).
- **Change tracking**: per-day `Round`s and hash deltas
  (`analyze_tests_results.py:66 find_delta_hashes`).
- **Feedback**: **none from tests to the model.** A failing test is reported,
  diffed and notified; it does not revise evidence, belief or structure. Only
  human curation feeds back, via `filter_by_curation` at the next assembly.
  `path_stmt_counts` (how often a statement appears in passing paths) is
  reporting only.
- **Human role**: curation (evidence-level tags), model configuration, search
  terms.

**Assessment.** EMMAA *is* continuous model maintenance, and it is the closest
ecosystem comparator to a Rubicon/revision loop in **cadence and
delta-tracking**. It is not a revision loop in the **epistemic** sense: the
reasoning output (test results) never changes the knowledge input. In
workspace terms it is "rebuild projection + diff", not "revise belief". The
loop closes through humans.

## 12. INDRA World and domain independence

INDRA World reuses unchanged: the Statement classes (`Influence`, `Event`,
`Association`, `QualitativeDelta`, `QuantitativeState`), `AssemblyPipeline`,
`run_preassembly`, the `RefinementFilter` protocol, `IndraOntology` base,
and the belief scorers. It replaces: the ontology (`WorldOntology`, WM YAML
under the `WM` namespace), grounding (compositional 4-tuples theme /
theme-property / process / process-property with scores), `matches_fun` /
`refinement_fun` (location and time aware), source priors (Eidos rule
summary), readers, assemblers (CAG), and adds an **incremental assembler**
(`assembly/incremental_assembler.py:24`) that returns an `AssemblyDelta` and
recomputes beliefs over refinement descendants, plus structural curations
(`reverse_relation`, `factor_polarity`, `factor_grounding`) that re-hash and
merge statements.

Note: `compositional_refinement` (`assembly/refinement.py:32-34`) passes
`st1.subj, st2.subj` for the object check too, so the object is never checked
on that path. Recorded as a concrete instance of how refinement correctness is
hand-maintained per domain.

| part | domain-independent? |
|---|---|
| Statement substrate | **partly**: the base, `Evidence`, hashing and `Influence`/`Event` are generic; most subclasses (Modification family, `Gef`, `Gap`, `ActiveForm`) are biology |
| grounding | mechanism generic (`db_refs` + ns priority); content specific; World needs compositional groundings the base `get_grounding` handles as a special case (`concept.py:47-82`) |
| ontology | protocol generic (`isa`, `partof`, `is_opposite`, `get_polarity`); content swapped |
| belief | generic, priors swapped |
| preassembly | generic **only because** `matches_fun` / `refinement_fun` are pluggable; identity semantics are rewritten per domain |
| causal assembly | graph assemblers generic; PySB is biology-only |

So INDRA's domain independence lives in **pluggable identity and refinement
functions over a fixed object model**. Specialization enters *at the Statement
type and identity function*, i.e. upstream of assembly. The workspace aims to
put specialization at the **projection boundary** instead (ClassView per
classid over a content-blind facet). This is a real design difference, not a
naming one.

## 13. Cross-map to AdaWorldAPI

| INDRA | AdaWorldAPI | relation | source | status |
|---|---|---|---|---|
| `Concept` / `Agent` grounded identity | OGAR classid + `ogar-obo` node | ANALOGOUS | OGAR `ogar-obo/src/lib.rs:95` | SHIPPED |
| Agent **state** as identity (mods, activity, location) | — (state would be facet payload or a refined classid) | NO MATCH yet | — | ABSENT (no state-in-identity scheme found in `ogar-obo`, `ogar-ro`) |
| `Statement` | SPO / `CausalEdge64` (S/P/O palette indices, f/c u8, Pearl mask, inference type) | ANALOGOUS; ours has truth + causal rung in-edge, INDRA has class-as-predicate + rich agent state | `causal-edge/src/edge.rs:140-161` | SHIPPED |
| `Evidence` | `CausalWitnessFacet` (experimental), `witness_fabric.rs` (`WitnessLens :146`, `RevisionTrajectory :1424`), W slot (6-bit), MedCare `ProvenanceWitness` | ANALOGOUS, **weaker in practice**: no shipped per-evidence text/source/epistemics record attached to an edge | `lance-graph-contract/src/causal_witness.rs:201`; MedCare `medcare-cohorts/src/provenance.rs:270` (no consumer) | SHIPPED types, unwired |
| grounding pipeline | `deepnsm-v2` lexicon + OGAR | partial | §5.3 | lexicon SHIPPED, consumer PLANNED |
| `combine_duplicates` | population fold over identity key | **ANALOGOUS** — both group by an identity key and keep all members; INDRA keeps them as a list on the survivor | — | fold/mask machinery SHIPPED in contract; no evidence fold over SPO rows found |
| refinement DAG | ontology rail + mask (`ogar-obo` saturation) | ANALOGOUS on the entity side; no statement-level refinement found | `ogar-obo/src/reason.rs:171` | entity SHIPPED; statement-level ABSENT (searched `lance-graph/crates`, `OGAR/crates` for `refinement_of` / statement subsumption) |
| contradiction pairs | `revision.rs` / fusion candidates / contradiction depth | ANALOGOUS in intent; neither changes truth from a contradiction today | `revision.rs:31`; `fusion.rs:199` (candidates only, "carries no confidence scalar", test `:404`) | SHIPPED policy, no truth write |
| belief | NARS truth / evidence weight | ORTHOGONAL axes (§7.4) | `exploration.rs:89` | SHIPPED |
| INDRA DB raw Statement | observation / episodic record; Lance version row | ANALOGOUS | `episodic_basin.rs`; `lance-graph/src/graph/versioned.rs:501-609` | SHIPPED |
| INDRA DB preassembled Statement + link table | folded canonical proposition + version-range read | ANALOGOUS | `lance-graph-planner/src/temporal.rs:72, 188, 206` | SHIPPED (query-time policy, "not storage") |
| IndraNet | causal graph projection (`causal-edge/src/network.rs`) | SAME role | | SHIPPED |
| PySB / KAMI / MIRA | executable causal-model projection | NO MATCH | — | ABSENT (no rule/ODE emitter found in `lance-graph/crates`) |
| EMMAA loop | Rubicon / kanban / revision cycle | ANALOGOUS cadence; EMMAA lacks epistemic feedback, ours lacks a running literature loop | `rubicon_witness.rs:113` | SHIPPED primitives, no loop |
| epistemic basin | — | — | plans only (`belief-abi-restoration-v1.md`, …) | PLANNED |

## 14. Key difference: statement assembly versus evidence field

Tested against the code:

- In INDRA, **evidence is mathematically operable in exactly one place:
  belief**. The scorer reads `source_api`, rule subtype and `negated` per
  Evidence and nothing else (`belief/__init__.py:136-195`). Everywhere else —
  dedup, refinement, contradiction, graph assembly, PySB, model checking —
  evidence is **provenance carried along**. Identity, refinement and
  contradiction are computed on the Statement *after* each extraction has
  committed to one reading, one grounding and one type.
- The commitment happens **before assembly**, per raw extraction:
  disambiguation writes a single winner into `db_refs`; a reader emits one
  Statement class. Alternatives survive only as annotation scores.

So the boundary is: **INDRA's operable state is the Statement; evidence is a
provenance list that feeds one scalar.** The thesis's characterization of
INDRA's central object — *normalized Statement + Evidence[] + belief +
refinement/support relations* — is **accurate**, with two additions the
thesis must not miss:

1. INDRA keeps **agent state as part of identity**, a level of mechanistic
   specificity (site, activity, bound partners) the workspace's SPO identity
   does not yet carry.
2. INDRA keeps **raw ↔ preassembled linkage losslessly** (INDRA DB) and
   **per-evidence raw grounding** (`annotations.agents.raw_grounding`), so
   re-assembly under a different identity function is possible. INDRA's
   statements *are* already rebuildable projections of the raw layer — just
   not of an observation layer.

The workspace side is, today, **partly aspirational**: `LexicalEvidence`
retains all readings but the FSM still commits to one; NARS revision is
evidence-weight based but no shipped path writes evidence-weighted truth onto
an edge (MedCare `reinforcement.rs:42-46` defers it); the witness facet is
experimental.

## 15. Live observations versus published assertions

### 15.1 MedCare trace (current code)

```
synthetic cohort (12 diseases × 50 patients)     medcare-cohorts/src/lib.rs:6
  → StatMatrix                                    medcare-cohorts/src/cohort_stats.rs:20   SHIPPED
      correlations(): Pearson + local fisher_z_p  :99, :389
      coherence():   Cronbach α, ICC(3,1)         :152   (jc crate)
      disease_stat_matrix(cohort)                 :476
  → consumer: display view only                   medcare-server/src/views/studie.rs:19,37

separate path:
  criterion_validity: Pearson + Spearman          medcare-first-thought/src/lib.rs:323,340
  → correlation_truth(r, ρ, agreement)            reinforcement.rs:115   SHIPPED
       f = 0.5 + (r+ρ)/4,  c = |(r+ρ)/2| · agreement
  → ReinforcementLane::reinforce (NarsTruth::revision)  :148
  → CorrelationKey = (LOINC observation, disease)  :97
  → CausalEdge64                                  deferred (:42-46)       NOT IMPLEMENTED
```

Two facts the harvest must state plainly: (a) the StatMatrix path and the
NARS path are **not connected**; (b) `correlation_truth` derives confidence
from **effect size × agreement**, not from sample size, and no Fisher-z or
Jirak gate feeds it. By the NARS reading in §7.4 that conflates effect
magnitude (a frequency-side quantity) with evidence mass. That is a current
weakness of *our* code, surfaced by the comparison.

### 15.2 Could both meet in a common representation?

Yes, at the level of `(subject concept, relation, object concept, truth,
provenance)`: an INDRA `IncreaseAmount(LOINC-observation-concept,
disease-concept)` and a MedCare correlation keyed by `(LOINC, disease)` can
share a key once both are grounded to the same ontology nodes. INDRA has no
statement class for "correlates with"; the honest mapping is `Association`
(undirected) or `Influence` with a `QuantitativeState`, not `IncreaseAmount`.

### 15.3 What reduction of a JC result to one INDRA Statement loses

| quantity | INDRA Statement can hold it? |
|---|---|
| sample size n | only as free text in `annotations` |
| distribution shape | no |
| covariance / correlation matrix | no (pairwise only, and not as a number in identity or belief) |
| Fisher-z, CI, p | annotation text only; **not used by belief** |
| effect size r / ρ | annotation, or `QuantitativeState.value` (a "reported quantity") |
| reliability (α, ICC) | no |
| raw observations | no (INDRA has no observation layer) |
| study / cohort hierarchy | no (one `context` per Evidence) |
| anomaly structure (CLAM / CHAODA) | no |
| temporal versions | no (EMMAA rounds are model snapshots, not observation versions) |
| NARS truth (f, c) | no (belief is a single scalar on a different axis) |
| epistemic basin membership | no |
| competing lexical readings | collapsed before assembly |
| fold history (which observations were merged) | only for evidence lists; not for observations |

Converting a live statistical result into one INDRA Statement keeps the
**claim** and discards the **evidence mechanics**; the belief INDRA would then
compute from it (one evidence, source `medcare`, some prior) would carry no
information about n, effect or reliability at all.

## 16. Statistical evidence in INDRA

Searched `indra/statements`, `indra/belief`, `indra/preassembler` for sample
size, p-value, effect size, confidence interval, odds ratio, hazard, cohort:
**no hits**.

| quantity | representable | parsed from text | used in belief | recomputed | absent |
|---|---|---|---|---|---|
| sample size | annotations (free) | not by core readers | no | no | as a field |
| effect size | `QuantitativeState.value` (World) | Eidos/Hume quantities | no | no | — |
| variance / CI / p | annotations | no | no | no | yes |
| meta-analysis | — | — | — | — | yes |
| cohort observations | — | — | — | — | yes |
| covariance / repeated measures / reliability / heterogeneity | — | — | — | — | yes |
| *source counts* | `source_counts` (DB, IndraNet) | — | **yes** (SimpleScorer, CountsScorer) | yes | — |

Belief is computed from source counts and rule identity; **source-count belief
is not study statistics** and must not be mapped onto them. Workspace side:
`jc` ships Pearson `reliability.rs:94`, Spearman `:172`, Cronbach α `:217`,
ICC `:284`, Fisher-z `stats.rs:1193/1202`, kappa/omega/KR-20/t/ANOVA. Jirak is
a pillar probe (`jc/src/jirak.rs:127 prove()`), **not** a reusable
noise-floor API. No meta-analysis primitive found (searched `crates/jc/src`).

## 17. Contradiction and revision

| question | INDRA |
|---|---|
| contradictory Statements linked? | returned as a list of pairs by `find_contradicts`; not stored on objects; not in INDRA DB schema |
| belief recomputed? | **no** |
| coexist? | yes, always |
| one eliminated? | no (only by human curation) |
| provenance decides survival? | no |
| new evidence alters old Statements? | yes, by appending evidence and re-scoring belief; never by contradiction |
| temporal / version awareness? | EMMAA deltas only; no time in belief |
| local or global? | pairwise local |
| epistemic-pothole analogue? | no |

Workspace: NARS revision changes truth when the *same* proposition receives
negative evidence (frequency drops). `revision.rs` decides *whether* new
evidence is independent (echo = zero weight) — something INDRA lacks — but it
is a policy surface with no production write. `fusion.rs` generates synthesis
candidates without confidence. Contradiction depth / epistemic basins:
PLANNED. Net: neither system currently lets a contradiction alter stored
truth automatically; INDRA by design (detection only), the workspace by
incompleteness.

## 18. Scaling model

| stage | INDRA cost | homologous workspace problem |
|---|---|---|
| grounding | per agent, dict lookup + optional classifier | per token lexicon lookup (COCA) |
| duplicate detection | O(N log N) sort-group | identity-keyed fold; with a fixed-width key (classid + facet) it is a radix/bitmap grouping |
| refinement | ontology-indexed candidate sets, then pairwise `refinement_of`; DB build O(N²/B²) batch pairs | "for each statement, which statements are its ancestors" = per-role ancestor-set intersection; expressible as mask AND over per-role ancestor masks if ancestor closure is precomputed (as `ogar-obo` saturation does for entities) |
| contradiction | O(P·N) product per opposite pair, unindexed | same index as refinement with a polarity flip |
| belief | per statement, linear in evidence; hierarchy pooling walks descendants | revision is a commutative fold over evidence weights |
| INDRA DB query | denormalized per-namespace meta tables, joins avoided | SoA columns + mask predicates |
| EMMAA | full daily reassembly | Lance version-range read + delta |

No cross-language benchmark was run; this table names homologous problems
only.

## 19. What INDRA does that we should learn from

| idea | where | verdict |
|---|---|---|
| shallow vs full hash (knowledge identity vs evidence identity) | `statements.py:296` | **ADOPT CONCEPT** |
| agent state as part of identity | `agent.py:72-103` | **ADOPT CONCEPT** (as a refinement facet, not a type) |
| per-evidence raw text + raw grounding kept through normalization | `preassembler/__init__.py:139-180` | **ADOPT CONCEPT** |
| raw / preassembled separation with a lossless link table | INDRA DB (GPL — concept only) | **ALREADY HAVE** in spirit (Lance versions + episodic rows); the explicit raw→folded link is worth mirroring independently |
| pluggable `matches_fun` / `refinement_fun` | `Preassembler.__init__` | **INCOMPATIBLE WITH OUR SUBSTRATE** as a pattern (identity is fixed by classid; specialization is at projection), keep as reference |
| `RefinementFilter` protocol (cheap prefilter → exact confirm) | `refinement.py:79` | **ADOPT CONCEPT** |
| source systematic vs random error | `belief/__init__.py:136` | **ADOPT CONCEPT** as a *discount on evidence weight* before NARS revision, not as a replacement truth |
| hierarchy belief (specific evidence supports general) | `:422` | **ADOPT CONCEPT** (no shipped equivalent) |
| evidence-level curation keyed `(statement hash, evidence hash)` | `filter_by_curation`, INDRA DB `curation` | **ADOPT CONCEPT** |
| Gilda scoring + context classifiers | gilda | **INTEROPERATE** (could ground our concepts to INDRA namespaces) |
| Statement JSON / REST | `rest_api/api.py` | **INTEROPERATE** |
| assembler abstraction (one corpus, many projections) | `assemblers/*` | **ALREADY HAVE** (ClassView projections) |
| model checker as "explain this claim" | `explanation/` | **KEEP AS REFERENCE**; ours has richer causal rungs |
| testable model + daily delta (EMMAA) | emmaa | **ADOPT CONCEPT** (tests as standing questions over versions) |
| executable model generation (PySB/MIRA) | | **KEEP AS REFERENCE** (no current need) |

## 20. What our substrate appears to add (current code only)

| capability | status | source |
|---|---|---|
| COCA lexicon with every reading retained + counts | SHIPPED | `deepnsm-v2/src/lexical.rs:34-41, 111, 332, 490` |
| `Reading` / `PosSet` / configuration-set FSM / certain-vs-alternative SPO | **PLANNED** | plan `deepnsm-v2-lexical-evidence-consumer-v1.md:306-308`; ABSENT in `lance-graph/crates/**/*.rs` |
| relative-pronoun handling (one level) | SHIPPED | `deepnsm-v2/src/fsm.rs:96, 142-205` |
| temporal stream / version-range reads | SHIPPED (query policy) | `lance-graph-planner/src/temporal.rs:188`; `graph/versioned.rs:501` |
| OGAR / OBO / RO saturation | SHIPPED | `ogar-obo/src/reason.rs:171`; `ogar-ro/src/lib.rs:153` |
| DOLCE / OGIT | not verified in this pass | — |
| ClassView, ontology rails | SHIPPED (contract) | `lance-graph-contract` |
| CausalEdge64 with f/c, Pearl mask, inference mantissa, W slot | SHIPPED | `causal-edge/src/edge.rs:140-161`, `layout.rs:26-52` |
| Pearl counterfactual / Simpson detection | SHIPPED | `causal-edge/src/network.rs:120, 171`; `pearl.rs` |
| episodic basin codec | SHIPPED | `lance-graph-contract/src/episodic_basin.rs` |
| epistemic basins | PLANNED | plans only |
| NARS revision (evidence-weight) | SHIPPED | `exploration.rs:110-120` |
| echo-aware evidence independence (`revision.rs`) | SHIPPED, no production write | `revision.rs:31` |
| Lance versioning | SHIPPED | `versioned.rs:501-609` |
| `jc` Pearson / Spearman / α / ICC / Fisher-z | SHIPPED | `jc/src/reliability.rs`, `stats.rs` |
| Jirak noise floor as reusable API | ABSENT (probe only) | `jc/src/jirak.rs:127` |
| CLAM / CHAODA | SHIPPED as test / lite | `graph/neighborhood/clam.rs` ("a TEST, not a fact"); `perturbation-sim/src/chaoda.rs` |
| ndarray HPC statistics, full ClamTree | UNVERIFIED (ndarray absent here) | — |
| fold / mask, Palette256 | SHIPPED (contract) | — |
| MedCare cohort → StatMatrix | SHIPPED (synthetic cohorts, display only) | `cohort_stats.rs:20` |
| live stats → NARS truth | SHIPPED, effect-based confidence | `reinforcement.rs:115` |
| NARS → CausalEdge64 | NOT IMPLEMENTED (deferred) | `reinforcement.rs:42-46` |
| count → truth from mined rules | SHIPPED | `arm-discovery/src/translator.rs:68` |

## 21. Reference interop projection (design only, no code)

### 21.1 Ada evidence proposition → INDRA-like Statement

```
canonical subject   → Agent(name, db_refs={ns: id})      ns via OGAR namespace → INDRA ns table
canonical predicate → Statement subclass                  closed mapping table; unmapped → Association/Influence
canonical object    → Agent / Concept
each evidence row   → one Evidence                        NEVER merged in the export
   source text      → Evidence.text
   source ids       → Evidence.text_refs, source_id
   source           → Evidence.source_api = "adaworld:<lane>"
   epistemics       → Evidence.epistemics {negated, hypothesis, direct}
   statistics       → Evidence.annotations["adaworld"] = {n, r, rho, fisher_z, alpha, icc, …}
   NARS truth       → Evidence.annotations["adaworld"]["truth"] = {f, c}
belief              → stmt.belief = NARS expectation  c·(f−0.5)+0.5  (labelled as a projection)
```

**Lost** (cannot be represented without loss, only smuggled in
annotations that INDRA ignores): frequency/confidence as two axes; evidence
independence / echo structure; raw observations and covariance; cohort
hierarchy; version history; competing lexical readings; Pearl rung
(intervention vs association); classid facet payload; epistemic basin
membership; any fold history beyond the evidence list.

### 21.2 INDRA Statement + Evidence[] → Ada evidence population

```
for stmt in statements:                      key = shallow hash (kept as foreign id)
  for ev in stmt.evidence:                   ONE observation row per Evidence, not per Statement
     subject/object  ← re-grounded from ev.annotations.agents.raw_grounding
                       (NOT from the preassembled db_refs) so our grounding can disagree
     relation        ← stmt class + polarity + agent state as facet
     provenance      ← source_api, source_id, text_refs, text, source_hash
     epistemics      ← negated → negative evidence (lowers f), hypothesis/direct → flags
     prior weight    ← INDRA rand/syst priors as an evidence-weight discount (documented)
  refinement links   ← kept as references between folded propositions, recomputed on our side
  belief             ← kept as an imported annotation; never used as our truth
```

The reverse **must** keep each Evidence as its own row and re-derive truth by
revision; collapsing to one edge with `belief` as confidence would import the
source-reliability axis as if it were evidence mass.

## 22. Benchmark (proposed, not run)

1. **Corpus**: a small mechanistic set INDRA already covers well, e.g. the
   MAPK/ERK pathway, by retrieving INDRA DB statements for 10–20 HGNC genes
   via the public REST API (read access only), with their Evidence text and
   PMIDs.
2. **INDRA side**: record statements, shallow hashes, evidence lists, belief,
   `pa_support_links`, and `find_contradicts` output on the same set.
3. **Our side**: run the existing lexical + FSM path over the **same evidence
   sentences**; ground to OGAR namespaces; fold by identity key; revise NARS
   truth per proposition.
4. **Compare**: grounding agreement (per agent, against Gilda's top match);
   predicate agreement (our relation vs INDRA class); provenance retention
   (fraction of evidence rows traceable); duplicate-folding agreement;
   contradiction pairs; causal graph topology (edge set Jaccard, sign
   agreement).
5. **Inject** a small measured cohort through `jc` (synthetic, stated as
   such) for one pathway relation, and test whether the observational
   evidence can **update or challenge** the folded proposition, i.e. whether
   a measured null or opposite-sign correlation moves NARS truth, where INDRA
   would not move belief.
6. **Project back** via §21.1 and diff against the INDRA statements.

Report: exact agreement, representational differences, information lost in
each direction, wall-clock cost per stage, unresolved ambiguity. Blocked on:
the FSM does not yet consume `LexicalEvidence`, and no shipped path writes
revised truth to an edge, so step 3 would measure the current
single-reading parser.

## 23. Particular questions

1. **Is "Statement + Evidence[] + belief" sufficient for a meta-study
   result?** No. Belief does not encode effect size, heterogeneity, n or
   design, and statistics would sit in annotations nothing reads. It can
   *reference* a meta-study's conclusion as one evidence item.
2. **Can INDRA recompute a claim from raw observations?** No. It consumes
   extracted or curated assertions. Its only "data" path is
   `PysbModelChecker.score_paths` against data and EMMAA's dynamic queries,
   which test a model, not derive a claim.
3. **What does belief represent?** Source reliability: P(the statement is a
   correct extraction/assertion) under independent systematic and random
   source errors. Not empirical effect strength.
4. **How does contradiction alter belief?** It does not. Only *negated
   evidence of the same statement* lowers belief (`·(1 − P_neg)`).
   Cross-statement contradictions are detected and returned, nothing more.
5. **Is `combine_duplicates` a population fold?** Conceptually yes (group by
   identity key, keep every member), with the difference that INDRA's
   members are Evidence objects appended to a surviving Statement, not
   observations in a column; and the fold key commits to a single grounding
   per extraction beforehand.
6. **Is refinement a materialized hierarchy, a relation, or both?** Both: a
   relation computed by `refinement_of` per class, stored as
   `supports/supported_by` on objects, and materialized as a hash DiGraph in
   `BeliefEngine` and as `pa_support_links` in INDRA DB.
7. **How much provenance survives PySB?** A statement UUID per rule
   (`from_indra_statement` annotation). No evidence, no belief, no hash.
8. **Can INDRA round-trip from model to Statements?** Only by lookup
   (`stmt_from_rule` resolves the UUID against the original statement list);
   there is no synthesis of Statements from rules.
9. **Is EMMAA closer to our Rubicon/revision loop than INDRA core?** Closer in
   cadence, versioning and standing tests; not in epistemics, because test
   outcomes never revise knowledge.
10. **Where would live MedCare evidence enter INDRA without premature
    reduction?** Nowhere inside INDRA's object model. The least-bad entry is
    one `Evidence` per cohort result with statistics in `annotations`, which
    belief ignores. A faithful entry needs a layer below Statements that INDRA
    does not have.
11. **What is lost by converting a JC result to one INDRA Statement?** See
    §15.3: n, distribution, covariance, Fisher-z/CI, reliability, raw
    observations, cohort hierarchy, versions, NARS f/c, basin membership,
    reading alternatives, fold history.
12. **Could INDRA Statements be an interoperability format while remaining a
    projection of our substrate?** Yes, with §21: export one Evidence per
    evidence row, statistics in a namespaced annotation, belief as an
    explicitly labelled projection; import by exploding to evidence rows and
    re-revising. It is a good exchange format for *claims* and a poor one for
    *evidence*.

## 24. Mission connection diagram (hypothesis, corrected)

```
                      RAW SCIENCE
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
         literature                observations
            │                            │
   deepnsm-v2 / MarkovNSM          JC / (ndarray)
   (lexicon keeps all readings;    (StatMatrix, Fisher-z, α, ICC)
    parser still commits to one)         │
            │                            │
            └─────────────┬──────────────┘
                          ▼
                 evidence population           ← target; today two disconnected
                          │                       lanes (MedCare StatMatrix vs
                  ontology / revision             reinforcement → NARS, no edge write)
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
       ORKG view       INDRA view        Ada-native
      comparison        Statement        causal/reasoning
       / facets          / model           projection
```

Correction from the harvest: INDRA's own pipeline places its commitment
point (single grounding, single statement class) **above** the box labelled
"evidence population", in the literature lane. An INDRA view is therefore a
projection that can be produced from the population, but INDRA's *ingest*
cannot feed the population without the §21.2 explosion back to evidence
rows. The diagram holds as a target; the "evidence population" box is not yet
a shipped object.

## 25. Thesis test

> *INDRA makes normalized causal/mechanistic Statements the stable
> intermediate representation from which graphs and executable models are
> assembled. AdaWorldAPI is converging on a lower-level stable representation
> in which observations, lexical alternatives, statistical evidence, ontology
> identity, epistemic state and provenance remain operable before being folded
> into Statement-like propositions.*

**First sentence: confirmed** by code (§3, §6, §9, §14).

**Second sentence: partly true, and must be stated as direction, not state.**
Shipped: all-readings lexicon, NARS evidence-weight revision, `jc`
statistics, OGAR saturation, Lance versions, Pearl rungs. Not shipped: the
parser consuming alternatives, a single evidence population joining text and
observation lanes, truth written onto edges, epistemic basins.

**What INDRA retains that the characterization misses:**

- **Mechanistic agent state in identity** (site, activity, bound partner,
  mutation): finer than our current SPO identity.
- **Lossless raw ↔ assembled linkage and per-evidence raw grounding**: INDRA
  statements are already rebuildable projections of their raw extraction
  layer. The difference is that its raw layer is *assertions*, not
  *observations*, and that each raw assertion has already committed to one
  reading.
- **A calibrated, curation-trainable source-reliability model**: an axis NARS
  does not have and the workspace could use as an evidence-weight discount.

**Verdict on "study/claim as projection of an evidence population":** the
comparison **strengthens** it. INDRA demonstrates the cost of the alternative
precisely: once evidence is a provenance list attached to a committed
Statement, effect size, sample size, independence and reading alternatives
have nowhere operable to live, and contradiction cannot change belief. It
also shows the requirement the thesis must meet to be better rather than
merely different: keep INDRA's lossless raw→assembled linkage and its
specificity of identity, while moving the commitment point below the fold.
