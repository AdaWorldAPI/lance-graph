# Language LEGO: corpus-grounded grammar as executable graph-query algebra

**Status:** RESEARCH / RFC ONLY — not an implemented cross-language query compiler.
**Captured:** 2026-10-18.
**Target:** `lance-graph` SoA ABI, masked RelAlg/Datalog AST, R2IL/Quack, DeepNSM-v2, NAL, and the fixed CausalEdge64 ISA.
**Intent:** Preserve one end-to-end hypothesis and give it a falsifiable landing sequence; do not clone Krill, invent a new parser, redefine CE64 fields, or prematurely add a language runtime.

## 0. One-sentence thesis

**Human language is a typed, composable frontend for queries over the same graph on which knowledge and reasoning run.** Interrogatives introduce variables; morphology and syntax constrain their binding; grammatical-function topology constrains permissible semantic interpretations; Datalog/RelAlg executes the resulting graph pattern; NAL/ONA handle supported, uncertain, and competing claims. A Cypher query, a German question, and an English question should lower to the same canonical algebra when they ask the same thing.

The diagram is not `text → LLM embedding → generated Cypher`. It is:

```text
COCA / DeReKo [candidate lemma, PoS, frequency, provenance]
 UD gold / CorefUD / UniMorph [grammatical and anaphora supervision]
 KorAP/Krill/Koral [linguistic spans, positional operators, query AST]
 multilingual Bible [verse-aligned *contrasts*, not semantic truth]
               │ optional independent input streams
               ▼
  Source-specific adapters: token, span, sentence, foundry, annotation,
  version, offsets, morphology, candidate alternatives, lineage
               ▼
  Canonical grammatical SoA ABI (stable address + columns + masks)
               │
      Haspelmath attested function-region masks
      dependency + verb-frame + discourse + anaphora constraints
               ▼
  Canonical semantic-query AST (variables, roles, scope, negation,
  quantifiers, temporal/evidential constraints, projection)
               ▼
  RelAlg / Datalog → typed R2IL → mask-risc / Quack / native SoA reads
               │
       NAL / ONA oracle-informed, evidence-stamped reasoning
               ▼
  Answer rows + provenance + typed abstention / uncertainty
  CausalEdge64 only when the shared ABI, witness and evidence gates permit
```

The frontend must work for **assertions, questions, explanations, hypotheses, and corrections**, not just for entity mentions. No new natural-language-specific compute universe.

## 1. Current repository reality and non-duplication

Read these before implementing anything:

- `README.md`, "Query paths": Cypher/SQL currently run through DataFusion; Quack's typed builders lower to mask-risc; **there is no production text frontend into Quack**. The experimental Cypher→Quack seam is partial.
- `.arxiv/2026-10_haspelmath-deepnsm-ce64-harness.md` and `crates/deepnsm-v2/src/haspelmath.rs` (#1570): Fig. 8.1 grammatical-function continuity, attested forms, incomplete vs disconnected verdicts, fixed **epoch-wide** operand dictionary and explicit refusal of out-of-ABI cases. Do **not** create a rival Haspelmath implementation.
- `.arxiv/2026-10_haspelmath_semantic_maps_grammar_abi_notes.md` and `docs/research/semantic-morton-qualia-masked-do-rfc.md` (#1545): primary-source limitations; 12×12 / Morton are hypotheses, not results from Haspelmath.
- `crates/lance-graph-quack/examples/semantic_map_probe.rs` and `.claude/board/entries/2026-10-16-grammar-coordinates-annotation-wiring.md` (#1591): plane-based connectedness over additional maps; unification with the `haspelmath.rs` fixtures remains open.
- `crates/deepnsm-v2/src/{coca,lexicon,fsm,tekamolo,loci,wire}.rs` (#1314, #1321, #1591): lexical candidates, multi-reading FSM, German TEKAMOLO hypotheses, local loci, explicit CE64 wiring. No new implementation claimed here.
- `crates/causal-edge/src/orientation.rs` (#1594): bit 43..45 is one global `Orientation` codebook, `Core` plus seven dative functions. Do not invent a private syntax code there. The eighth Haspelmath function, Judicantis, is not a CE64 orientation.
- `.claude/knowledge/epistemic-frontier-capstone.md` and `.claude/board/entries/2026-10-17-ce64-one-layout-moore-synapse-membrane.md` (#1595/#1597): decisions about fixed CE64, layers, reference-frame/spine, presence masks and R2IL. Some decisions supersede still-present **code readers and older docs**; inventory and reconcile in their own PR before production writes.
- `crates/lance-graph-relalg/src/ir.rs`: currently describes string-free selection of one row space using `All/Empty/Plane/Atom/And/Or/Not`; full Datalog joins, recursion and quantified query AST are **not silently present**.
- `crates/lance-graph-nal/src/lib.rs`: typed NAL folds operate over sealed coordinates, evidence stamps and borrowed lanes. ONA/OpenNARS are **comparison oracles**, not an accepted cloned interpreter.

Existing measurement to retain: `.claude/board/entries/2026-10-04-deepnsm-v2-position-rules-on-ud-gold.md` reports independent UD English/German/French POS probes; `...-german-tekamolo-verb-position.md` reports **TEKAMOLO ordering is not a reliable universal grammar law** on the measured German text. Do not reverse that finding by making Te-Ka-Mo-Lo a mandatory sentence-order parser.

## 2. Grammatically meaningful LEGO, not token lookup

A source token is not a concept. Preserve this separation:

| Level | Example | Representation / invariant |
| --- | --- | --- |
| Surface | German `dem`, English `to` | FormId + source offsets; language-local |
| Lexical | lemma and POS alternatives | multivalued `(LemmaId, UposId, features, prior, source)` |
| Grammar | `Case=Dat`, `obl`, `acl:relcl` | typed UD-compatible dependency facts; no semantic role implied by tag alone |
| Semantic function | RECIPIENT, BENEFICIARY, LOCATION, DIRECTION, PURPOSE | versioned palette/function identity, Haspelmath connectivity constraints |
| Event/role | `GIVE(event)`, `RECIPIENT(event, child)` | independent, canonical relational facts |
| Epistemology | observed, associated, supported, causal candidate, causes | NAL evidence and CE64 read/write authority, not parser plausibility |
| Query | projection `?drug`, negation, every/some, why | executable semantic AST with explicit scope and answer mode |

German `Dativ` is *not* BENEFICIARY, and German `Akkusativ` is *not* DIRECTION. Cases give constraints that are interpreted under verbs, prepositions, frames and construction context. `auf dem Tisch` vs `auf den Tisch` is a useful minimal adversarial pair. German `zum Bahnhof` is directional despite Dativ.

Haspelmath (2003, `The Geometry of Grammatical Meaning`, Figs. 8.1 etc.) is about attested **grammatical-function adjacency and multifunctional form regions**, not a 144-verb vocabulary, address system or semantic metric. A complete attested disconnected coverage set can falsify a proposed map; missing/intermittent attestation must stay `Incomplete`. Mere adjacency licenses candidate exploration, not equivalence, entailment or causal certification.

Proposed codebook rule: **stable semantic identity by versioned globally pinned definition**, never "u8 happens to match". Numeric locality (Morton, 12×12 palette, HHTL) is an execution choice to compare against controls, not proof of semantic proximity. If a function set exceeds an 8-bit local palette, use explicit versioned indirection or a separate wider identity. **Do not widen/relabel CE64 8:8:8 on ingestion.**

## 3. Preserve existing TEKAMOLO and anaphora ABI

### TEKAMOLO

`crates/deepnsm-v2/src/tekamolo.rs` describes an existing **16-byte V3 facet** (`classid 4 + 12-byte value`) read as `G4D3`, four coarse→fine 3-byte lanes **Temporal, Kausal, Modal, Lokal** (4×3×u8). Zero lane = *unaddressed*. Its current German-language heuristic and left-corner/right-corner competition are a producer, not the sole authority. Do not replace its payload by a new `[event,predicate,argument]` struct; that was only an earlier explanatory analogy and would break the actual coarse/fine contract. Add a **semantic event-role projection beside** the existing facet, with explicit mapping.

The four axes concern modifiers, not the entire predicate-argument graph. Core AGENT, RECIPIENT, THEME are independent relational facts. Kausal/final denotes linguistic reason/purpose; it does **not** certify a Pearl intervention. Temporal in current Bible-oriented hydration may denote verse/TOC position; do not silently promote narrative position to event time.

### 24×i4 loci

`lance-graph-contract/src/causal_witness.rs` defines `CausalWitnessFacet`: a **reading** of 12 bytes as 24 signed i4 event-context loci. Zero = unbound (not a zero-distance reference). Sixteen named loci already exist (Temporal, Kausal, Modal, Lokal, S/P/O meaning, Antecedent, BasinAnchor, SupportedBy, Supports, RunbookEvidence, QualiaReference, MeaningLevel, Quorum, Contradiction). Eight are reserved; **do not reassign or turn them into arbitrary token registers**.

Offsets index the **event/triple stream**, not raw token positions. In `.claude/board/entries/2026-10-16-grammar-coordinates-annotation-wiring.md`, G1 (relative-clause antecedent to locus) and G2 (TEKAMOLO token-to-event binding) are open because the needed mapping is not exported. A local reference outside the signed window is an **external witnessed basin/link**, not a truncated i4. The source/target unit must be proven equal before storing a nibble.

Test sentence: `Ich gebe dem Kind, das dort wartet, heute gern ein Buch, damit es lesen kann.` Different clause-local case of `das` and its dative antecedent `dem Kind` must not be collapsed; `es` can be grammatically ambiguous between `Kind` and `Buch`. Keep alternatives and selectional evidence. The earlier chat's "24 offsets partitioned evenly across T/K/M/L" is not the in-tree contract.

## 4. Language-standardized DTO: data contract, not false universal semantics

**Canonical input DTO** must round-trip source facts plus annotation provenance:

- corpus ID, version, license/access scope, source hash, foundry/tagger and annotation scheme/version;
- original sentence/verse/document identity, token ordinals, character offsets and segment boundaries;
- language, surface, lemma, UPOS, original XPOS/STTS/CLAWS-like labels, morphological FEATS;
- HEAD/DEPREL, enhanced DEPS, MISC, multiword token ranges and empty nodes;
- candidate analyses, score/count/prior and the exact source reading that supplied each; unresolved/unknown must remain explicit;
- optional CorefUD mentions, entity clusters, discourse connective links, alignment lineage.

UD **normalizes labels**, not semantic equivalence. CoNLL-U as a serialization does **not** guarantee valid UD: the published `KorAP-XML-CoNLL-U` example writes STTS `VVFIN` into an UPOS-looking column. Validate and explicitly map tags to UD UPOS/FEATS before treating them as gold-compatible. Never overwrite the original foundry label.

The **SoA hot path** is IDs, presence masks, offset lanes and typed arrays over a sealed canonical address allocation, with strings in a cold lexical/source dictionary. Multiple readings must be rows/masks or candidate groups, not "first row wins" or a silent overwrite. A new corpus calibration requires a new sealed reference version; do not edit the baked spine in place.

Do not assume all languages align on one tokenization or that one UD dependency tag implies one thematic role.

## 5. Linguistic operators as the existing algebra

Extract *semantics*, not engines, from Krill, Koral, UD, Grew and discourse systems:

| Linguistic operation | Proposed native algebra | Key risk |
| --- | --- | --- |
| `lemma/POS/FEATS` condition | resident selection masks / lane predicates | original foundry mappings, multireadings |
| phrase/proximity | positional interval relation | tokenization and offset concordance |
| contains/overlaps/begins/ends | span interval algebra | exact empty-span/multiword semantics |
| UD dependency | typed relation with source and target ordinal | no foreign-key gather dressed as a mask |
| relative pronoun/coreference | mention/antecedent joins + local locus when addressable | candidate multiplicity, unbounded references |
| temporal, modal, purpose/cause | event-scoped predicate joins | TEKAMOLO role vs causal assertion |
| conjunction/alternative | mask AND/OR on common row space | joins still needed between row spaces |
| not/without | stratified negation / scoped antijoin | distinguish open-world absence from falsehood |
| recursive descendant/reachability | bounded/fixpoint Datalog lowering | termination, provenance and bag/set laws |
| who/which/what | bound variable + projection | answer scope and wh-question typing |
| why/how do we know | evidence-graph traversal and witness projection | explanation is not causal proof |
| might/probably/according to | explicit modality / source constraints | not silently a `CausalEdge64.c` |

Example (illustrative intended semantic IR, not a current parser):

```prolog
answer(Drug, Protein, Study) :-
    drug(Drug),
    inhibits(Drug, Protein),
    relation(Protein, implicated_in, bbb_disruption, Claim),
    supports(Study, Claim),
    population(Study, human),
    publication_year(Study, Year),
    Year > 2020.
```

Natural question: "Which drugs inhibit proteins implicated in blood-brain barrier disruption, supported by human studies after 2020?" The claim binding on `supports` is critical: a study mentioning the protein is not necessarily evidence for the protein→barrier claim.

**Compilation is not embeddings/autocomplete and not free-form text-to-Cypher.** The syntactic frontend must bind named predicates against an authorized graph schema/codebook, preserve quantified and negated scope, expose competing interpretations, and refuse unbound symbols. Cypher remains an optional alternate frontend, not a required intermediary.

**Execution invariant:** push down masks and apertures before reading unnecessary columns; use joins and folds where relation content genuinely requires them. The source graph, vocab IDs and results must share stable row/coordinate semantics. A mask is not a graph traversal, and the current `relalg::Rel` implements Boolean selection only; the wider operators require separately gated work.

## 6. NAL / ONA / CE64: do not turn syntax into truth

- Grammar candidate scores, lemma frequencies and parser probability are **not** epistemic `F/C`. Source/annotation quality remains in its own lane.
- NAL revisions need evidence stamps and independence. Thousands of translations derived from one base edition are correlated; verse agreement is not a thousand independent truth roots.
- ONA/OpenNARS/Hyperon are **external oracles** for recipe and rule fidelity where already used; no automatic interpreter port.
- CE64's **one layout** is governed by `causal-edge/src/layout.rs` and its corresponding current contract. `Orientation` bits 43..45 already encode seven dative-map function readings plus Core, not the whole grammar.
- Subsequent 2026-10-17 decisions in `epistemic-frontier-capstone.md` interpret 46..49 as activation and 53..58 as thought layer; code and older working-model docs still contain contrary readers (inference mantissa, W witness handle). **Do not claim migration complete or create a new writer until the separately planned contract reconciliation lands.**
- The "same semantic fact" should have **one canonical address**, with source/grammatical evidence and competing epistemic hypotheses as typed lanes at that address; immutable givens must not be rewritten as hypotheses.
- For questions, the system can return `Answer`, `MultipleReadings`, `Abstain`, or `UnsupportedByCurrentGraph`. Do not fabricate a causal explanation when graph evidence only supports association.

## 7. The experiment that proves or breaks the thesis

Use an **independently defined semantic-query fixture set** (the corpus itself is not a semantic gold oracle), with German/English/French initially and other languages held out. For each meaning, include:

1. Paraphrases across languages with distinct case/preposition/word-order constructions.
2. Known near-misses with different quantifier, scope, polarity, tense, role or evidence binding.
3. German Wechselpräposition (`wo` vs `wohin`), dative recipient vs beneficiary, and accusative object vs direction.
4. Relative clauses: antecedent gender/number/case agreement *within the relative clause*, long references outside the i4 window, and ambiguous neuter pronouns.
5. TEKAMOLO event scoping: `damit` as PURPOSE, `weil` as REASON, `so` as MODAL, `da` ambiguous, clause attachment.
6. Haspelmath connected-region held-out languages, disconnected complete counterexamples and incomplete attestations.
7. Natural-language graph questions with `which`, `why`, negation, existential/universal scope, time qualifiers and evidence restrictions.
8. Corpus noise: Bible verse mismatch, missing verse, paraphrastic divergence, source/translation lineage and domain shift to current German/English.

**Oracle layers must not be mixed:**
- lexical gold: held-out lemma/POS/FEATS accuracy and candidate recall;
- dependency gold: UD/UAS/LAS etc.;
- coreference gold: CorefUD cluster/mention metrics;
- semantic query gold: *separately authored* canonical AST and answer bindings;
- reasoning gold: exact NAL/ONA rule/evidence oracles and independently witnessed claims.

Baseline arms: (A) dominant corpus PoS; (B) current DeepNSM-v2 FSM; (C) UD facts + typed semantic rules; (D) optional parser/LLM text-to-Cypher baseline; (E) our masked R2IL compiler. Test on *same* inputs/answer sets and report memory, allocations, p50/p95 latency, cold/warm, ambiguity, abstention, AST equivalence, witness completeness and provenance independence.

Expected deliverable: **for truly equivalent questions, identical canonical query ASTs and answer bindings; for meaningful differences, non-equivalent ASTs.** If a translated sentence is not genuinely equivalent, preserve the difference; never edit gold to force convergence.

## 8. Proposed gated implementation slices (none claimed shipped by this RFC)

| Gate | Action | Acceptance / stop |
| --- | --- | --- |
| L-0, documentation | harvest existing in-tree seams and source repos, pin legal restrictions | no duplicated parser/semantic topology; current-code links pinned |
| L-1, SoA DTO | reversible CoNLL-U(+)-to-typed-ID ingest with foundries, masks and multi-reading candidates | round-trip token/annotation metadata; rejected invalid UPOS; no data redistribution |
| L-2, annotation algebra | differential Krill span/proximity/containment and UD dependency predicates | oracle-equivalent sets, offsets, captures; no unjustified mask-only graph hop |
| L-3, local binding | export FSM antecedent event index and token→event association before assigning 24xi4 loci | no wrong-unit pointer, no overflow remap; ambiguity preserved |
| L-4, semantic functions | multilingual Haspelmath coverage fixtures + case/role/TEKAMOLO adapters into immutable codes | explicit incomplete/falsified verdict; preserves canonical epoch dictionary |
| L-5, query AST | grammar questions to typed, scoped Datalog AST; schema binding and refusal | canonical AST equivalence on equivalent cross-language questions, near-miss divergence |
| L-6, lowering | AST → masked RelAlg + R2IL/Quack joins/folds under current substrate rules | results equal a reference relational oracle; no gather/foreign-key shortcut |
| L-7, reasoning | evidential NAL operators over returned graph paths with stamp/gate | no syntax→truth promotion, no false causes, no correlated overcount |
| L-8, performance | benchmark against Krill/Lucene, Tantivy and reference graph queries | no speed/accuracy claim without same fixtures and access patterns |

**Sequencing:** L-1 and L-2 can be independent. L-3 is already an identified seam in #1591. L-4 can reuse `haspelmath.rs` and `semantic_map_probe.rs`, not fork them. L-5/6 must not pretend `relalg` already handles recursive joins.

## 9. Non-goals and refusal contracts

No new CE128; no per-verse operand-byte meanings; no redefinition of 12-byte facets; no indiscriminate re-use of the reserved eight i4 loci; no mandatory reimplementation of Krill/Lucene/UD/ParZu/ANNIS as runtime dependencies; no unsourced reuse of COCA/DeReKo copyrighted corpora; no "Bible as gold semantics"; no commitment to universal 144 predicates before measured validation; no claim all language grammars can be inferred purely from corpus frequency; no ontology identity inferred from spelling, UPOS or Morton proximity.

## 10. Downstream documents

Read alongside:
- `docs/research/linguistic-lego-corpus-and-falsification-matrix.md` (external repository harvest, license gates, probe matrix, exact examples).
- `.claude/knowledge/epistemic-frontier-capstone.md` (source-of-truth current decisions).
- `.claude/board/entries/2026-10-16-grammar-coordinates-annotation-wiring.md` (the already-open antecedent/TEKAMOLO event-index gaps).
- `.arxiv/2026-10_haspelmath-deepnsm-ce64-harness.md` (source-faithful topology, explicit ABI refusals).
