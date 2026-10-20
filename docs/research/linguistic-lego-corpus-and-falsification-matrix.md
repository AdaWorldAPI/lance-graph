# Language LEGO — repository harvest, legal gates and falsification matrix

> **Plan and board (2026-10-20):** owned by [`ce64-grammar-wiring-v1`](../../.claude/plans/ce64-grammar-wiring-v1.md). Board: [2026-10-20-chatgpt-prs-1601-1617-plan-links.md](../../.claude/board/entries/2026-10-20-chatgpt-prs-1601-1617-plan-links.md). Merged in #1603.

**Status:** source/integration inventory, not a dependency decision or measured result.
**Captured:** 2026-10-18. Companion to [the grammar-to-query RFC](linguistic-lego-grammar-to-query-rfc.md).

## 1. Distinguish five types of source before selecting any code

| Kind | What it can establish | What it cannot establish alone |
| --- | --- | --- |
| Corpus frequencies (COCA, DeReKo) | observed distribution of form/lemma/tag by source, domain and method | correct POS in a given sentence, gold dependencies, semantic truth |
| Linguistic treebanks (UD) | *annotated contextual grammar* and gold benchmark targets for supported splits | entity/coreference identity, rhetorical meaning or graph-query truth |
| Parallel translations (multilingual Bible) | comparable verse keys; alternate grammatical realization and apparent semantic agreement | guaranteed sentence-level alignment or independent semantic gold |
| Corpus search/index engines (Krill/Koral/KorAP) | mature positional/span/dependency retrieval operator semantics | semantic reasoner or universal NLP compiler |
| Semantic / discourse inventories (Haspelmath, PDTB, DiMLex, CorefUD) | type-compatible semantic-function hypotheses, discourse/coreference comparison | Pearl causes, full sentence logical forms or universally fixed predicate-address order |

A billion automatically tagged tokens are still **silver** unless independently validated. Gold UD is contextual grammar, not gold causal inference. A thousand translations sharing a source edition are not a thousand independent observations.

## 2. Source systems to read, without making them permanent runtimes

### Lexical priors and grammatical standards

| Source | Primary URL | Harvest as | Cautions |
| --- | --- | --- | --- |
| COCA, Corpus of Contemporary American English | https://www.english-corpora.org/coca/ | lexical form→lemma/POS alternatives; frequencies, genre-conditioned priors | non-open corpus licensing; CLAWS-like corpus tags are not UPOS; source-specific folds |
| DeReKo, IDS Mannheim | https://www.ids-mannheim.de/digspra/kl/projekte/korpora/ | German lemma/POS morphology, usage frequencies and foundry disagreement | corpus is access controlled; STTS/multiple foundries, not automatically UD gold |
| UD organization | https://github.com/UniversalDependencies | canonical UPOS, FEATS, HEAD, DEPREL/DEPS and multilingual treebanks | annotation labels are normalized, not semantic identities |
| UD German-HDT | https://github.com/UniversalDependencies/UD_German-HDT | large German dependency fixtures | check release, split, per-treebank license |
| UD German-GSD | https://github.com/UniversalDependencies/UD_German-GSD | case/Wechselpräposition and train/dev/test targets | no explicit TEKAMOLO gold lanes |
| UD English-EWT | https://github.com/UniversalDependencies/UD_English-EWT | English dependency/lexeme controls | genre differences from Bible/COCA |
| UD French-GSD | https://github.com/UniversalDependencies/UD_French-GSD | cross-language ambiguity, word order, dative/recipient checks | surface alignments aren't automatic |
| UD tools | https://github.com/UniversalDependencies/tools | validators and UD evaluation oracle | run actual tags through validation, not just CoNLL-U parser |
| CorefUD | https://ufal.mff.cuni.cz/corefud | multilingual mention/cluster annotation and scoring | not interchangeable with dependency treebank schema |
| UniMorph | https://unimorph.github.io/ | cross-language feature inventories | separate namespace and mapping from UD FEATS |
| PARSEME | https://parsemefr.lis-lab.fr/ | multiword expressions and discontinuous lexical units | extra alignment scope |
| Universal PropBank | https://universalpropositions.github.io/ | predicate/argument-role annotation research | inspect exact data before asserting gold coverage |
| UMR | https://umr4nlp.github.io/ | sentence/document meaning representation research | not a drop-in 8:8:8 register |

### KorAP: copy operator contracts, not its JVM/Perl runtime

| Repository | URL | Reusable ingredient | Caution |
| --- | --- | --- | --- |
| Kalamar | https://github.com/KorAP/Kalamar | UI inspection, foundry hints (`tt`, `marmot`, `malt`, `spacy`, `ud`), result views | Mojolicious UI, **not** the engine |
| Kustvakt | https://github.com/KorAP/Kustvakt | API access for searches and annotation match metadata | enforces licensing and authorization; do not bypass it |
| Krill | https://github.com/KorAP/Krill | Lucene positional spans, structural/relation searches, match captures | per-position payloads do not map 1:1 to Tantivy |
| Koral | https://github.com/KorAP/Koral | `KoralQuery` JSON-LD AST from corpus QLs (CQP, ANNIS etc.) | query syntax/AST, *not* semantics or NAL |
| KorAP-XML-CoNLL-U | https://github.com/KorAP/KorAP-XML-CoNLL-U | XML ↔ CoNLL-U, foundry/text/offset metadata | experimental converter; published example has **STTS in UPOS-looking column** |
| korapxmltool | https://github.com/KorAP/korapxmltool | modern Java converter/tagger/parser/index exporter; multi-foundry merge, parallel throughput | README declares **GPLv3**, Java 21, experimental combinations |
| Koral-Mapper | https://github.com/KorAP/Koral-Mapper | e.g. `stts-upos.yaml` mappings | context-dependent tag mapping requires extra checks |
| Tantivy | https://github.com/quickwit-oss/tantivy | Rust inverted index, positions, pretokenized streams | positional payload and structural-graph parity need sidecar or new operators |

**Implementation direction:** use `korapxmltool`'s **Krill/Koral JSON or the original annotated KorAP XML** as interchange material for a prototype; avoid trying to read opaque Lucene segment files in Rust. Tantivy can retrieve lexical candidates, while SoA relational masks and graph edges carry spans, dependencies and witnesses. Differential-test result sets and captures against Krill; no performance claim from term search alone.

A published `KorAP-XML-CoNLL-U` readme example includes STTS `VVFIN` under the would-be UPOS column. A validator must reject this until explicit normalization maps it to `VERB` with source tag preserved in XPOS/provenance.

### Grammar, anaphora and discourse prior art

| Repository / system | URL | What to borrow |
| --- | --- | --- |
| ParZu | https://github.com/rsennrich/ParZu | German dependency grammar, constraint/ranking ideas, Prolog lineage |
| Grew | https://github.com/grew-nlp/grew | linguistic graph-pattern and graph-rewrite test oracle |
| nlprule | https://github.com/bminixhofer/nlprule | Rust rule-based grammatical analyses and disambiguation |
| SMOR | https://www.cis.uni-muenchen.de/~schmid/tools/SMOR/ | German finite-state inflection/compound analysis, **check non-commercial conditions** |
| Stanza | https://github.com/stanfordnlp/stanza | full off-the-shelf UD annotation baseline |
| spaCy | https://github.com/explosion/spaCy | German/English/French parser baseline |
| pocores | https://github.com/discourse-lab/pocores | German rule-based coreference / antecedent candidates |
| discoursegraphs | https://github.com/arne-cl/discoursegraphs | merge syntax, coreference, RST and connective annotations on aligned tokens |
| DiMLex | https://github.com/discourse-lab/dimlex | German discourse connectives, possible PDTB cause/purpose/time/concession senses |
| DiscourseSegmenter | https://github.com/discourse-lab/DiscourseSegmenter | clause/discourse segmentation rules |
| eduseg_de | https://github.com/discourse-lab/eduseg_de | more recent German segmentation comparison |
| evidencegraph | https://github.com/discourse-lab/evidencegraph | graph model of argument/support/attack |
| RSTmulti | https://github.com/discourse-lab/RSTmulti | multi-annotator rhetorical relationship disagreement |

Discourse connectivity (`weil`, `damit`, `obwohl`, `während`) should become **typed clause-event relations**, not inferred causal certificates. Reference resolution must distinguish a relative pronoun's internal case from the antecedent's main-clause case. Graph-rewrite or Prolog grammars can be **gold/behavioral oracles** even when we never embed their runtimes.

### Parallel languages and topology

| Source | URL | Role |
| --- | --- | --- |
| Haspelmath (2003) semantic maps | https://zenodo.org/records/831410 | measured cross-language function *connectivity*; user-provided scanned `SemMaps.pdf` already distilled in two in-repo research notes |
| Multilingual Bible Corpus | https://github.com/christos-c/bible-corpus | verse-aligned language comparison |
| eBible parallel corpus | https://huggingface.co/datasets/DavidCBaines/ebible_corpus | large-scale aligned corpus candidate; inspect license per translation and versification metadata |
| Token-based parallel-Bible typology study (LREC-COLING 2024) | https://aclanthology.org/2024.lrec-main.965/ | published method for multilingual grammatical comparison |
| Penn Discourse Treebank | https://catalog.ldc.upenn.edu/LDC2019T05 | formal temporal, contingency, comparison and expansion sense hierarchy; licensing applies |

No public-source statistics in this inventory are a newly reproduced experiment. Pin exact corpus releases and checksums when the prototype begins. Every raw text/annotation extract needs rights review; repo source licenses do not transfer permissions to redistributing underlying corpora or trained models.

## 3. Small, sharp experiments with semantic output fixtures

### G-1: Morphology is not thematic role

| Input construction | Gold question/interpretation to test | Wrong shortcut to reject |
| --- | --- | --- |
| `Ich gebe dem Kind ein Buch.` | recipient(child), theme(book) | all datives are LOCATION or BENEFICIARY |
| `Ich helfe dem Kind.` | affected/benefited child, verb-frame-specific | always `iobj → RECIPIENT` |
| `Das Buch liegt auf dem Tisch.` | stative LOCATION(table) | every `auf` means GOAL |
| `Ich lege das Buch auf den Tisch.` | directed GOAL(table) | every `auf` means LOCATION |
| `Ich gehe zum Bahnhof.` | GOAL(station), despite dative | dative blocks direction |
| `Ich kaufe ein Buch für das Kind.` | BENEFICIARY(child) | beneficiary requires dative |

Rule: UPOS/case/dependency are **admission constraints**, while canonical semantic role requires a verb frame and construction analysis. Add equivalent French/English items and compare canonical roles, not token offsets or word counts.

### G-2: Anaphora and local signed pointers

Sentence: `Ich gebe dem Kind, das dort wartet, ein Buch, damit es lesen kann.`

- `das` in `das dort wartet` is **nominative in the relative clause**, antecedent `dem Kind` is **dative in the main clause**. Different case is valid; do not collapse the two grammatical analyses.
- `es` has more than one grammatically compatible neuter antecedent. Common-sense reading (`Kind` reads) may be preferred by semantic plausibility; retain evidence/alternatives.
- The existing `CausalWitnessFacet::Antecedent` is a signed **event offset**, not a signed token-index difference. Only write when the parse supplies a proven token→event map and the target is addressable within the valid nonzero window. For anything else, produce a witnessed nonlocal reference or abstain.
- Gold CorefUD may provide clusters, but it does not by itself validate which text propositions are true.

### G-3: Clause-semantic scope, not string matching

| Phrase | Candidate relation | Guard |
| --- | --- | --- |
| `weil es regnet` | asserted/explained REASON | does not certify interventional cause |
| `damit er lesen kann` | intended PURPOSE | does not imply goal realized |
| `obwohl es regnet` | CONCESSION | does not reverse actual truth |
| `da` | context-dependent causal or spatial/temporal | retain alternatives until clause resolved |
| `so` | manner/degree/discourse depending on context | do not hard-code Modal for every occurrence |
| `morgen` | event TIME | distinguish deictic time from corpus verse address |

The existing 2026-10-04 German TEKAMOLO probe falsified naïve Te-Ka-Mo-Lo **surface ordering** as a universal deterministic rule. Retain its frequency-vs-position ablations.

### G-4: Haspelmath topology, not numerical aesthetics

- Reuse `haspelmath.rs`'s eight named dative functions and `MapVerdict::Incomplete`.
- Reuse the richer `semantic_map_probe.rs` plane fixtures where appropriate (indefinites, reflexive, instrumental/comitative maps); do **not** silently union distinct semantic maps into one dative wheel.
- Measure whether attested multifunctional marker coverage forms a connected subgraph under a **frozen topology**, with complete/incomplete flags.
- Compare proposed 12×12 or Morton embeddings against adjacency-oracle, shuffled labeling, graph embedding and unconstrained controls. Haspelmath does not establish geometric coordinates or quantized distances.
- The CE64 orientation enum contains 0 Core + 7 dative-function readings; **Judicantis is not encodable as a CE64 orientation**. Do not alias it to an unrelated function.

### G-5: The executable natural-language question

Use one independently annotated canonical AST and answer-set fixture for each question below, then express it in German, English and French:

| Question family | AST requirement | Adversarial near miss |
| --- | --- | --- |
| "Which drugs inhibit proteins implicated in BBB disruption?" | project `Drug`, join inhibition and implicated-in via same protein | "Which proteins inhibit drugs?" must reverse direction |
| "...supported by human studies after 2020?" | bind support to the **implicated-in claim edge**, restrict population/time | a study merely mentioning the protein does not qualify |
| "Which proteins are **not** associated with X?" | explicit bounded complement semantics | open-world absence must not become proven `NOT associated` |
| "Did **every** study support X?" | universal quantifier over an explicit study domain | `some` is different |
| "Why is X believed to cause Y?" | explanation witness path and claim status | related/associated is not equivalent to causes |
| "For whom was the book purchased?" | beneficiary argument projection | recipient/beneficiary conflation |
| "Where is the book / where was it moved?" | LOCATION versus GOAL | dative/accusative shortcut |

Use the following illustrative Datalog oracle for one fixture (not currently supported verbatim by `lance-graph-relalg`):

```prolog
answer(Drug, Protein, Evidence) :-
    drug(Drug),
    inhibits(Drug, Protein),
    relation(Protein, implicated_in, bbb_disruption, Claim),
    supports(Evidence, Claim),
    population(Evidence, human),
    publication_year(Evidence, Year),
    Year > 2020.
```

The **goal is AST/answer equivalence for *genuinely equivalent* questions**, AST divergence for non-equivalent questions, and typed abstention for unknown schema symbols, unsupported spans, evidence or remaining ambiguity. A string similarity score does not prove semantic identity.

## 4. Measurement matrix and kill switches

| Layer | Baseline/reference | Metric | Immediate stop |
| --- | --- | --- | --- |
| Candidate coverage | source `word_forms.csv` + multi-reading `LexicalEvidence`; UD train | candidate-set recall by lemma/POS, OOV, source coverage | first-row-wins silently discards a possible gold reading |
| POS/morphology | UD held-out treebanks / `ud_pos_eval` | UPOS, lemma, FEATS; calibration by language/genre | same test examples used for threshold tuning and reported test |
| Dependency parsing | Stanza/spaCy + gold UD | UAS, LAS, well-formedness | STTS in UPOS accepted as UD without validation |
| Span / relation retrieval | Krill/Lucene fixtures, KoralQuery operator AST | identical matches, offsets, captures, ordering where defined | Tantivy term hits mistaken for span semantics |
| Coreference | CorefUD + source resolver controls | clusters/mention accuracy, unresolved ratio, long-reference correctness | offset overflow, event/token index mixing |
| Semantic functions | Haspelmath language-held-out map fixtures | connectedness verdict, incomplete vs falsified, exception counts | adjacency promoted to equivalence or truth |
| TEKAMOLO | current cue/FSM baseline and independent semantic label set | role/attachment precision/recall and abstention | surface TEKAMOLO order treated as inviolable |
| Query semantics | manually audited cross-language AST/answer oracles | canonical AST equality, scope and binding correctness | language-specific shortcut changes semantic predicate identity |
| Reasoning | NAL/ONA oracles; provenance-stamped observations | independent-root revision parity, no spurious `Causes` | parser confidence written as `CausalEdge64.c` |
| Execution | current DataFusion Cypher, Krill/Lucene, Tantivy baselines | cold/warm latency, allocations, peak RSS, masks/read bytes, results | speed claims without identical query semantics and corpus slice |

Execution should honor the substrate's current `no gather/foreign lookup masquerading as a mask` constraint. A dependency graph hop may need a genuinely supported relational operation; don't claim it is free because candidate selection uses bitmaps.

**Reproducible existing reference commands, not execution claims for this PR:**

```bash
cargo run --release --manifest-path crates/deepnsm-v2/Cargo.toml --example ud_pos_eval -- TRAIN.conllu TEST.conllu
cargo test --manifest-path crates/deepnsm-v2/Cargo.toml haspelmath
cargo run --release --manifest-path crates/deepnsm-v2/Cargo.toml --example tekamolo_de -- de_gsd-ud-train.conllu de_gsd-ud-test.conllu
```

Pin repository revision, corpus/treebank release, source checksums, licenses, resolved wordform candidate counts, train/dev/test separation, execution machine and all command output in the future implementation PR. The documentation PR does not run these commands.

## 5. Final anti-drift checklist

- [ ] No "new language runtime" before verifying the existing in-tree lexical, FSM, Haspelmath, loci, TEKAMOLO, R2IL and NAL seams.
- [ ] Corpus/type licenses checked **before data ingestion**, including individual Bible editions.
- [ ] Source annotations preserved alongside normalized UPOS, lemma, morphological and dependency IDs.
- [ ] Stable classid/epoch/codebook boundary; no per-verse reassignment or overflow aliasing.
- [ ] No new CE64 layout, no private field reinterpretations.
- [ ] Relative-pronoun grammatical agreement, pronoun referential identity and factual truth scored independently.
- [ ] Grammar/semantic-map predictions kept distinct from evidence-stamped epistemic inference.
- [ ] Held-out language AND held-out construction controls; negatives/adversarial near misses.
- [ ] AST round-trip/canonical query equivalence tested against a separate semantic gold set.
- [ ] Performance only discussed after exact result equivalence.
