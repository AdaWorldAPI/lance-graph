# German grammar as position + morphology evidence — rule inventory

READ BY: anyone adding a voter to `crates/deepnsm-v2/examples/ud_pos_eval.rs`;
anyone building a German case, anaphora, TEKAMOLO or plausibility evaluator;
anyone consuming `build_de_codebook.py`'s TSVs.

**Status:** an exploration map from one `/coresearch` council (2026-10-04).
Every rule below is graded by its source. Nothing is adopted. Numbers marked
OURS were measured in this workspace; all others are from the cited source
at the grade shown. Board entry: `.claude/board/entries/2026-10-04-coresearch-german-grammar-evidence.md`.

## 0. The premise split (read first)

A quorum voter has one signature: `Fn(&Tok) -> Option<Pos>`, one word class per token, scored on UPOS. Only Q1 rules are voters. The other four questions get their own evaluators. They may feed Q1 voters as cues; they never become voters (premise-auditor, 2026-10-04).

| Q | question | output | gold | evaluator |
|---|---|---|---|---|
| Q1 | word class (adj/adv, noun/verb, DET/PRON) | Pos per token | UPOS | `ud_pos_eval` quorum |
| Q2 | case and government | Case= per nominal | UD Feats `Case=` on the governed DET/NOUN (ADP carries none) | new `ud_case_eval` |
| Q3 | anaphora | a link per anaphor | coreference (CorefUD PotsdamCC; TüBa-D/Z academic-only) | its own |
| Q4 | TEKAMOLO / topological fields | an order preference | rates with bootstrap CIs | its own |
| Q5 | "does this make sense" | a graded score | AUC original vs swapped | its own |

Naming trap: `contract/grammar/wechsel.rs` means dual-role TOKENS. The Dat/Akk Wechselpräposition data is `build_de_codebook.py`'s `wechsel.tsv`.

## 1. Data facts (measured on UD German-GSD train, 263,791 tokens)

- Case is present on NOUN 98.3 %, DET 99.4 % and PRON 98.1 %. **ADP has no Case** (1 of 29,374).
- **`iobj` = 0** in GSD (110 in 3.4M HDT tokens). A dative object is `obj`/`obl` with Case=Dat.
- ADV carries no Degree (1 of 11,900); ADJ does (19,452 of 19,489). That is a gold fact, not a run-time cue.
- GSD Case gold is noisy at about 10 %: `obj` + Case=Nom 740 against 6,180 Acc. **Case kills must hold on HDT**, which is CC BY-SA, 3.4M tokens and converted from manual annotation.
- `build_de_codebook.py` mines every table from UD **gold**. That is allowed only on the fit split, with scoring on test only, and the tables must be re-mined on fit before held-out weighting. Five of them (`article_decidability`, `valency`, `wechsel`, `relative_pronoun`, `reflexive`) **have no Rust consumer**.

## 2. The ranked inventory

**Columns:**

- Verdicts: bridge = OPPORTUNITY / WORTH-EXPLORING / DROP; firewall = PASS / CONFLICT (seam) / TRAP.
- "Measured / published" gives OURS numbers or the source's number.
- Precision comes from the published source or from OURS. Everything else is CONJECTURE until its probe runs.

**Seams:**

- **S1 (licence):** a table mined from UD is a CC BY-SA derived database. It ships as a fetched, built artefact under its own share-alike notice, never as a `const` or `include_bytes!` table in a code crate, and never in `lance-graph-contract`.
- **S2:** mine on train, score on test, and assert the split in code.
- **S3:** a lexicon is injected; it is never a SoA fact or a contract type.
- **S4:** plausibility and discourse resolution live in `lance-graph-planner`. deepnsm-v2 emits only candidates and agreement features.

| rank | Q | rule (testable) | signal | gold | measured / published | bridge | firewall | kill probe (pre-registered) |
|---|---|---|---|---|---|---|---|---|
| 1 | Q1 | **der/die/das + NOUN/ADJ → DET; + finite verb, or closing a verb-final clause → PRON** | next form/readings; clause end | UPOS of der/die/das | none yet | OPPORTUNITY (needs `next_form` in `Tok`) | PASS | KILL if: PRON precision < 0.80; or PRON recall < 0.50; or fires > 2× gold PRON. S |
| 2 | Q1 | **Copula clause, word at the right edge → ADJ (predicative); full-verb clause, word before the right bracket → ADV** | Satzklammer position + copula | UPOS (ADJ+advmod folds to ADV) | OURS: 88.5 % (26 fires), 90.0 % (20); German adj/adv joint quorum 81.6 → 85.1 % (disable: 81.9 %) (D-LXC-20) | ALREADY-HAVE | PASS | done |
| 3 | Q1 | **Monoflexion: stem + paradigm ending (-e/-en/-em/-er/-es) after DET/ADP and before a NOUN → ADJ (attributive). A bare stem is never attributive.** Exceptions: indeclinables (lila, rosa), city -er adjectives, nominalised adjectives (NOUN), `am -sten` (adverbial) | form vs lemma + position | UPOS, deprel amod | OURS: the broad ending rule is 70.6 % on 17 fires; "uninflected before a noun → adv" 80.0 % | OPPORTUNITY | PASS (hand-written endings) | KILL if: precision ≤ 70.6 % or ≤ frequency on the same tokens (read on HDT); or fires on < 1 % of tokens. Silence: indeclinables give 0 ADJ votes. S |
| 4 | Q1 | **A clause-final particle after a V2 finite verb → ADP (compound:prt)** | position + closed particle list | deprel compound:prt | Batinić & Schmidt GSCL 2017 (READ-IN-FULL slides): 99 % precision on FOLK; 11 % of finite verbs separated | OPPORTUNITY | PASS as a rule (the 7,658-verb list has an unknown licence: measure against it only) | KILL if: precision < 0.95; or > 2 % false fires on `case` ADP / pronominal adverbs; demote to a cue if frequency is already ≥ 98 %. S |
| 5 | Q2 | **Fixed-case prepositions** (Acc: durch/für/gegen/ohne/um; Dat: aus/bei/mit/nach/seit/von/zu/gegenüber; Gen: wegen/trotz/während/statt/innerhalb) | closed list | Case= of the governed DET/NOUN | Duden (SECONDHAND) | OPPORTUNITY | PASS (hand-authored list) | KILL if: precision ≤ the noun form's train-majority case on the same tokens; or Acc/Dat < 0.90 on HDT; or Gen < 0.70. S |
| 6 | Q2 | **Relative pronoun case = its role in the relative clause; gender/number from the nearest NP to the left** | form + in-clause elimination | Case= of PronType=Rel | Klenner & Tuggener RANLP 2011 (READ-IN-FULL): relative → next NP left | OPPORTUNITY | CONFLICT S1–S3 (mined TSV) | KILL if: form-only ≤ form frequency; or elimination lifts syncretic die/das/der by < 10 points. S |
| 7 | Q2 | **One nominative per finite clause** (elimination over syncretic articles; decide only when complete) | article candidate sets + clause segmentation | Case= on nsubj/obj | WOGLI arXiv 2306.04523 (SECTION-READ): only masc sg der/den, ein/einen, dieser/diesen and weak nouns mark Nom vs Acc; Table 7 lists 8 surface-identical patterns | OPPORTUNITY | PASS as Q2; **TRAP as a UPOS voter** | KILL if: precision ≤ the article-majority case; or it completes on < 10 % of syncretic NPs. Silence: the WOGLI residue abstains ≥ 90 %. M |
| 8 | Q2 | **Wechselpräposition: the article's case decides; Dat = location (wo), Acc = goal (wohin); verb-pair prior (stellen/legen/setzen → Acc, stehen/liegen/sitzen → Dat)** | article + verb lemma | Case= of the governed NP (no Wo/Wohin gold exists) | No computational accuracy is published (OPEN); Draye 2014 (ABSTRACT-ONLY) | OPPORTUNITY | CONFLICT S1–S3 | KILL if: verb-prior precision ≤ the preposition's majority case on ambiguous forms (den); or fires < 5 % of Wechsel NPs. M |
| 9 | Q2 | **Valency + uniqueness:** a verb frame licenses ≤ 1 argument per case | verb lemma (particles joined via rank 4) | Case= (Dat = obj/obl + Case=Dat) | Bader & Häussler 2006 (READ-IN-FULL): OS in the Mittelfeld is 93.7 % Dat; Somers et al.: Dat-first 42 % for alternating verbs | WORTH-EXPLORING | CONFLICT S1–S3; E-VALBU measure-against only | KILL if: gain < 1 point over rank 7; or precision ≤ the article baseline. M |
| 10 | Q2 cue | **Auxiliary selection: sein + participle → unaccusative/motion; a directional (Acc) PP co-occurs with sein-motion** | aux form + participle | Acc rate of Wechsel NPs, sein vs haben | Keller & Sorace 2003 (SECTION-READ): the Auxiliary Selection Hierarchy; reflexives always take haben | WORTH-EXPLORING | PASS as a cue; TRAP as a voter | KILL if: the Acc-rate difference is < 10 points; or Fisher p > 0.01; or there are < 200 PPs on HDT. S |
| 11 | Q1 cue | **Strong/weak ending agrees with the article class** (an ending/article mismatch means not attributive to that noun) | ending × article | amod of the next noun | Eisenberg (SECONDHAND) | WORTH-EXPLORING | CONFLICT S1 if baked from DEMorphy/Wiktionary; PASS as hand-written Rust rules | KILL if: precision ≤ base + 5 points; or fires < 1 %. Silence: ≤ 5 % false mismatches. M |
| 12 | Q1 cue | **Topological field anchors: LK (finite verb), C (complementizer), VC (verb cluster)** | finite-verb / complementizer position | downstream gain of the Satzklammer voters | Cheung & Penn ACL 2009 (READ-IN-FULL): LK F1 99.75, C 98.98, VC 98.56, MF 94.99, NF 82.73 (TüBa-D/Z) | WORTH-EXPLORING | PASS | KILL if: the Satzklammer voters gain < 20 % more fires; or their precision drops > 2 points. M |
| 13 | Q3 | **Agreement filter (gender/number/case) + Binding A (sich → subject of the same clause) + B (a pronoun is not bound by its own co-arguments); salience subject > object** | lexicon features, clause boundaries | coreference | Klenner & Tuggener (READ-IN-FULL): TüBa-D/Z CEAF salience-only 51.41; per type personal 58.86, relative 55.97, reflexive 54.16. OURS (English, Aesop, small n): agreement 0.727, noun-only 0.455, agreement-blind 0.273 | OPPORTUNITY | CONFLICT S4 (ranking in the planner) + A6 (`loci.rs` SelectionalFit is float: quarantine it) | KILL if: ≤ nearest-NP + 3 points; or the filter drops the gold antecedent > 5 %; or it removes < 30 % of candidates. Binding A ≤ nearest-NP. L |
| 14 | Q3 | **Expletive es** (weather/impersonal verb, or Vorfeld with a postponed subject) | verb lemma + position | deprel expl (GSD: 373, plus 206 expl:pv to check) | UD counts (READ-IN-FULL) | OPPORTUNITY | PASS as a Q3 filter; TRAP as a voter (es is PRON either way) | KILL if: precision < 0.70; or recall < 0.30; or referential > 20 % of fires. S |
| 15 | Q4 | **TEKAMOLO order is a tendency, not a decision**; Luther vs edited text differ | cue lexicon + Mittelfeld | rates with bootstrap CIs | OURS: the order law 60.3/65.2 % (Mittelfeld 45.7/58.8 %), killed as a decider; cue-by-verb-position 88.5 % vs 84.1 % (D-LXC-17) | ALREADY-HAVE | PASS (Luther 1545/1912 public domain only) | KILL if: \|T<L Luther − edited\| is inside the bootstrap 95 % CI; or cue coverage < 5 %. M |
| 16 | Q5 | **S↔O swap / verb swap scored against held context** | train (subj, verb) / (verb, obj) counts | AUC original vs swapped | PEP-3k (READ-IN-FULL): random 0.50, LR 0.64, NN 0.68, + world-knowledge bins 0.76. Thematic fit ρ 28–59 (SECTION-READ) | WORTH-EXPLORING | CONFLICT S4 + A6 (offline fitted only); WOGLI/PEP-3k licences unchecked; Wino-X MIT | KILL if: AUC ≤ 0.55; or ≤ the 95th percentile of a 1,000-permutation shuffle null; or ≤ frequency. Exclude swaps that change case marking. M |
| 17 | Q5 | **Proto-role animacy from WordNet `ConceptMask`** (the more animate argument is the subject) | integer membership (organism / sentient / artifact) | AUC gain over rank 16 | Dowty 1991 (SECONDHAND); PEP-3k bins (READ-IN-FULL) | OPPORTUNITY (zero new types) | PASS (identity membership, A1-clean) | KILL if: the AUC gain is < 0.03; or fires < 10 %. Silence: equal animacy abstains; psych verbs reported separately. S |
| 18 | Q5 (density evaluator, re-filed by the second premise gate) | **WordNet hypernym path metric → ndarray ClamTree + CHAODA density over ROLE-INDEXED points `(verb, role, synset)`**. A plain path distance is symmetric (d(S,O) = d(O,S)), so an S↔O swap is invisible to it. Density measures rarity, and rare is not implausible. Items: verb swaps only, unless a role-indexed point is defined | exact graph metric over synsets | the same swap AUC | CHAODA: Ishaq et al. 2021 (SECONDHAND via `ndarray/src/hpc/clam.rs`); LFD alone ≈ 0.62 ROC-AUC on a synthetic mixture | NEW (operator 2026-10-04) | CONFLICT: outside the letter of `48405aa2`, inside its spirit; **clean only offline as an evaluator, under a recorded DECISION**; TRAP on the hot path | Items: WOGLI swaps (minus those morphology separates) + frequency-matched verb swaps; senses = first sense or the minimum over senses, never gold. KILL if: AUC ≤ 0.55; or ≤ the 95th percentile of a lemma→synset shuffle null; or ≤ frequency; or ≤ the ConceptMask rule (rank 17) + 0.02. Silence: symmetric predicates (heiraten, treffen) within 0.45–0.55; identical-address swaps abstain ≥ 90 %. Coverage: WordNet resolves S and O on < 40 % → KILL. Spot-check the triangle inequality on 10⁴ triples first; any violation kills the CLAM premise. M |
| 19 | planner / belief (re-filed by the second premise gate) | **Belief-state transitions (Admitted / Revised / Chosen, `belief::ReviseOutcome`) as a revision-history evaluator in the planner's belief arena.** Counts and \|f₁−f₂\| are magnitudes ordered across revisions, not ±8 position offsets, so they are not 24×i4 loci. Only a peer pointer within ±8 (supported_by / contradiction) may be written as a locus on the CausalWitness tenant | revision events | does the transition record predict later contradiction/revision beyond the final TruthValue? | none | NEW (operator 2026-10-04) | CONFLICT S4: the transition rule lives in the planner; loci are offsets only; no support/refute mask layer returns (removed 2026-09-02); one extra 24×i4 register per node is allowed, the 8 reserved loci need a decision | KJV arena revised in book order: prefix books 1–33, suffix 34–66. Baseline = final (f,c); treatment adds Admitted/Revised/Chosen counts, max \|f₁−f₂\| and loci peers. Target: a later contradicting revision or a side flip in the suffix. KILL if: AUC gain < 0.02; or gain ≤ the 95th percentile of a null that shuffles transition features within (f,c) bins. Silence: single-Admitted statements predict identically. Coverage: < 500 multi-event statements → UNTESTED. Kill too if i4 quantisation (±8 reach) loses > 25 % of the gain. M |
| — | ceiling | Higher-order CRF on TIGER: POS 97.44 %, POS+MORPH 88.58 % (91.65 % with an analyser); case needs long context | — | — | Mueller/Schmid/Schütze EMNLP 2013 (SECTION-READ) | DROP as a design | PASS as a citation | none |

## 3. Shared precondition (bridge 0)

Most of ranks 1–9 need the same two changes:

- `next_form` added to `Tok`;
- `Inventory` loading `article_case` / `article_decidability` / `wechsel` / `valency` / `relative_pronoun`, re-mined on the fit split.

Ranks 5–9 are one case-constraint intersection, i.e. one Q2 evaluator rather than five voters.

## 4. Resources (licence decides what may be baked)

- **May be built and fetched:** UD German GSD + HDT (CC BY-SA), the de-codebook TSVs (CC BY-SA, derived), Wiktionary / DEMorphy paradigms (CC BY-SA), Wino-X (MIT), WordNet 3.1 (BSD-style).
- **Measure against only, never bake:** TüBa-D/Z, TIGER, GermaNet, E-VALBU (licence unconfirmed), the Batinić particle list. Check WOGLI and PEP-3k before use.

## 5. Not searched / OPEN

- adverbial vs predicative ADJD accuracy in the literature;
- a computational Wechselpräposition accuracy (rank 8 would be the first);
- German noun/verb homographs by position;
- per-case accuracy on UD;
- German Winograd beyond Wino-X;
- adverbial (TEKAMOLO) order corpus studies (only argument-order studies were found);
- left-corner parsing for German (leads only: arXiv 2109.04939, 2311.16258);
- whether the 85.1 % adj/adv figure used lexicons mined only on train. `UD_DE_INVENTORY` was built from GSD train, so yes, but the D-LXC-19 `Inventory` lemma map is mined from all of train, not the fit split.

## 6. Operator answers

**Recorded 2026-10-04 (D-LXC-22): the WordNet scope below is DECIDED as worded** (operator: *"Yes Wordnet would allow also clam Chaoda"*). See `.claude/board/entries/2026-10-04-wordnet-clam-chaoda-scope-and-grammar-read-params.md`.

**WordNet gives the address, COCA + tokens give the torque (operator, 2026-10-04).**
- WordNet supplies a Cartesian address but no linguistic torque (frequency, collocation, position).
- COCA and the tokens supply the qualia of the text.
- So rank 17's mask is only a coordinate, and the plausibility signal comes from COCA/token counts and the slot position.
- Rank 18's CHAODA points must combine both: (WordNet address of S/O, verb, COCA frequency band, slot position).
- This agrees with D-LXC-15: WordNet did not help noun/verb as a chooser or a filter.

### The questions as they were put

- **A1 scope for rank 18.** The firewall critic's draft wording:

  > **DECISION:** ontology concept ids remain identities; CAM-PQ and any learned/embedded distance over them stay prohibited (`48405aa2` unchanged).
  >
  > **SCOPE:** an exact graph metric derived solely from the ontology's own edges (WordNet hypernym shortest path, minimised over multiple inheritance) MAY be used as a `Distance` for ndarray ClamTree/CHAODA offline, as an evaluation or research instrument. It is never stored as a concept attribute, never mixed into the CAM-PQ/palette space, and never consulted on the substrate hot path.
  >
  > **REVISIT WHEN:** a proposal promotes it to run time, to a quorum voter, or to a non-WordNet ontology; or the swap AUC fails to beat frequency.
- **Rank 19.** The register question lapses (second premise gate). Transition history is a planner belief-arena record; the CausalWitness loci carry only peers within ±8. Open instead: does the AriGraph tenant need a planner-written revision-history lane at all, or is the belief arena enough? Rank 19's probe answers it.
- **`hydrate` = node creation.** It is restored and green. The invariants worth pinning as tests: one verse node per triple, 0 orphans, idempotent re-hydration (32,357 nodes), and every promoted basin key resolving.
