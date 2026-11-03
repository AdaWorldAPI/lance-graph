# D-GRAMMAR-CONSTRAINTS-0 — grammatical constraints × RelAlg × CE64/DT64

> **⊘ CORRECTION (D-GRAMMAR-WHEN-1, 2026-11-03).** This report says the head hop
> `Pred::EqU32Via` "executes; call sites EXO-140-unclassified". That is true but
> understated. ADMISSION B (`entries/2026-10-20-six-decisions.md:26-34`) says "a
> position derived from another relational column or table is not admissible as
> a read address", and the 2026-10-16 banner calls a read "at a position taken
> from another lane's values" a violation. `EqU32Via` reads `foreign[fk[i]]`, so
> it is gather-shaped, and no ruling admits it. The results below stand as LAB.
> The co-addressed alternative, and the owner decisions it raises, are in
> `.claude/probes/2026-11-03-d-grammar-when-1/REPORT.md`.

**Date:** 2026-11-03. **Status:** RESEARCH + LAB PROBE.
- Probe: `crates/grammar-constraints-probe`. It has its own workspace and is
  excluded from the root workspace.
- No parser, no grammar compiler, no new ClassId or `F###` word, no
  production change.

**Question:** can a grammatical constraint be stated as a typed relation over
existing observations and answered by the existing DR / RelAlg / mask-risc
fold? And where does the declaration language (CG-3, spaCy, UD, GF, MeTTa)
stop being the same thing as the fold?

**Answer, short:**
- **Yes, for the read-only selection part.** In the probe, one mask-risc
  program per rule answers German dative role candidates from UD-annotated
  tokens:
  - correct 4/4, 0 false positives, 0 false negatives, 1 abstention;
  - Direction/Purpose ambiguity kept;
  - 0 bytes allocated on the hot path;
  - ≈1.2–1.3 µs for all five rules over 40 tokens;
  - 8 negative controls behave as predicted; 9 source disables go red.
- **The load-bearing operand is the head hop**, `Pred::EqU32Via` with
  fk = HEAD. In UD German the recipient of *geben* and the experiencer of
  *gefallen* are both `obl:arg` + `Case=Dat`, so no constraint on the
  dependent's own lanes can tell them apart.
- **First missing interface:** DT64's WHEN (`Guard` over `SetExpr`) has no
  hop leaf, so it can state the dependent's conditions but not the head's
  frame. Even a guard it can state is refused by R2IL `prepare` as
  `GuardNotExecutable`. Executing the hop needs the EXO-140-unclassified
  `EqU32Via` path.

---

## A. Research synergy map

Each mechanism below lists what the source says, then what it implies here.
Q = quoted, P = paraphrased. Sources were fetched 2026-11-03. The UD
statements were also checked against the treebank.

1. **CG-3 is a reading-elimination system with an ordered, non-monotone
   state.** (manual, Rules chapter)
   - SELECT "Deletes all readings in the cohort except the ones matching the
     target set" (Q).
   - REMOVE "Deletes the readings matching the target set" (Q).
   - "Normally, REMOVE and IFF cannot remove the last reading of a cohort"
     (Q).
   - Sections run in "ascending only" order (Q). Changes by
     SELECT/REMOVE/IFF/… "will rerun the sections" (Q), and `--max-runs`
     "defaults to unlimited" (Q).
   - MAP/ADD append tags; SETPARENT/SETCHILD add dependency links, with loop
     protection unless ALLOWLOOP (Q).
   - **Implication:** CG-3's *contextual tests* are relational selections:
     offsets, `*`/`**` scans, BARRIER, LINK, NOT/NEGATE (P). Its *rule
     application* is an ordered state transition: the last-reading guard
     makes the result depend on rule order, which monotone Datalog cannot
     express.
   - CG-3 does look ON/WHEN/DO-shaped: target ≈ ON, contextual tests ≈ WHEN,
     the action ≈ DO. But its DO mutates a shared reading set in order. That
     is DT64's ordered-effect DO, not a pure derived relation, so the
     resemblance does not prove execution equivalence.
2. **UD separates an exact tree from an annotation convention.**
   - The basic layer "forms a tree, where exactly one word is the head" (Q,
     u/overview/syntax). Enhanced graphs may have multiple heads, cycles and
     empty nodes (Q).
   - German UD: "Bare dative and genitive objects … are considered oblique.
     This means that verbs of giving are not ditransitive predicates in
     German UD, as they have one oblique dative argument" (Q, verified
     verbatim from universaldependencies.org/de).
   - Validation levels: L1–L3 are language-independent; L4–L5 hold
     language-specific labels and rules (P).
   - **Implication:** "one head per word" is STRUCTURAL and makes HEAD a
     functional fk, which is what lets `EqU32Via` read the head's lanes in
     place. `obl:arg` vs `iobj` is a LANGUAGE-SPECIFIC convention.
3. **Haspelmath's map is a co-expression constraint, not a role assigner.**
   - "the functions must be arranged in such a way that all multifunctional
     grams can occupy a contiguous area" (p. 217, Q).
   - "every semantic map can be interpreted as making a universal claim …
     that can be falsified easily" (p. 232, Q).
   - **Implication:** the map constrains which function *sets* one form may
     cover; `haspelmath::continuity` already tests this. It never assigns a
     function to a token, so "Case=Dat → Recipient" is not something the map
     licenses. The census and the probe falsify it below.
4. **GF types the frame, not the surface.**
   - RGL German: `accdatV3 : V -> V3` "geben + dat(c2) + acc(c3)" (Q,
     synopsis). English: `mkV3` "ditransitive, e.g. give,_,_" (Q).
   - One grammar parses and linearizes (P, tutorial).
   - **Implication:** the frame lexicon (verb → which slot plays which role)
     is the reusable, typed object. Realization (case, adposition, order)
     belongs to the concrete grammar. This matches the P1 census below.
5. **Universals are mostly tendencies.** Verkerk et al. 2026, Nat. Hum.
   Behav. 10:126–136:
   - "We find statistical support for about a third of the proposed
     linguistic universals" (Q).
   - Approximate counts, taken from the summarizer and not verbatim: 191
     proposed universals → 89 under spatiophylogenetic control → 60 also
     co-evolving.
   - **Implication:** typological universals are HEURISTIC or EVIDENTIAL
     priors. They must never become a hard WHEN.
6. **spaCy's DependencyMatcher is a conjunctive graph pattern with node
   variables.**
   - `RIGHT_ID` / `LEFT_ID` name nodes; operators `<`, `>`, `<<`, `>>` and
     sibling operators (Q).
   - Returns "A list of (match_id, token_ids)" (Q), i.e. binding tuples.
   - EntityRuler resolves overlaps by length (P).
   - German models use TIGER labels (`da`, `oa`, `sb`), not UD (P, model
     meta).
   - **Implication:** `<`/`>` is exactly the HEAD fk hop. `<<`/`>>` is
     recursion (EXO-73). Overlap policy is HEURISTIC selection. A spaCy
     German rule would need a TIGER→UD label crosswalk first.

## B. Constraint classification

| class | in this probe | existing contract that carries it |
|---|---|---|
| STRUCTURAL | HEAD (one per word), DEPREL, Case features | resident `u32` lanes; HEAD as fk |
| LANGUAGE-SPECIFIC | frame lexicon `de-dative-frames-fixture-v0`; `zu`+Dat `obl` → {Direction, Purpose} | none is versioned in a contract; it is a probe constant (OPEN) |
| HEURISTIC | not applied: Direction vs Purpose is not chosen | none; a selection policy would be an owner decision |
| EVIDENTIAL | none (annotations are inputs, not NAL truths) | NAL `Beliefs` exists but is NOT used: an annotation is not a truth value |
| EFFECTFUL | none (outputs are candidate masks; nothing is written) | DT64 ordered `effects`; CG-3 REMOVE/MAP live here |

Also:
- **UNKNOWN is kept apart.** A missing `Case` gives `unknown_case`; a
  verb with no frame gives `abstain_frame`. Neither is FALSE.
- **R2IL refuses both kinds of guard.** A three-valued guard
  (`Truth::ThreeValued`) is refused at `lance-graph-r2il/src/binding.rs:293`,
  right after `GuardNotExecutable` at `:291`.

## C. Source-backed reuse map

| capability | where | status |
|---|---|---|
| Typed role vocabulary (Direction, Recipient, Experiencer, …) | `causal-edge/src/orientation.rs:17` (CE64 bits 43..45, `Core`+7); `deepnsm-v2/src/haspelmath.rs:27` (8 incl. Judicantis) | EXISTING, two different 8-element vocabularies (#1917, consolidated doc line 213) |
| Map contiguity check | `deepnsm-v2/src/haspelmath.rs:80` `continuity` | EXISTING, tested |
| Head hop read in place | `lance-graph-mask-risc/src/ir.rs:158` `Pred::EqU32Via`; Quack `Filter::EqU32Via` `lance-graph-quack/src/lib.rs:358` | EXECUTES; call sites EXO-140-UNCLASSIFIED |
| Hop as demanded result (case marker → its noun) | `ir.rs:338` `Terminal::ScatterOrU32` | EXISTING |
| Dependent-own WHEN | `lance-graph-contract/src/set_algebra.rs:135` `SetExpr`; DT64 `Guard` (`lance-graph-dt64/src/lib.rs:152`) | EXPRESSIBLE, but no hop leaf; the head frame needs an opaque `Sym(k)` |
| WHEN execution | `lance-graph-r2il/src/binding.rs:291` | REFUSED (`GuardNotExecutable`); 3VL refused at `:293` |
| DR identity and stale frame | `lance-graph-dt64/src/dr.rs` `LocalMap` / `MapTable::map` | EXISTING, used |
| Multi-reading parse | `deepnsm-v2/src/fsm.rs:904` `parse_readings` | EXISTING; not wired to UD HEAD/DEPREL (#1917, consolidated doc line 192) |
| Role as ONE CE64 field | `orientation.rs` | 3 bits = one role per edge; it cannot hold {Direction, Purpose}. Ambiguity needs two candidate edges or a mask, never one CE64 word |
| Recursion (`<<`, `**`) | relalg `μ` not built | BLOCKED (EXO-73) |

**First missing caller:** nothing turns UD CoNLL-U into DR-addressed token
lanes. The probe binds by hand (`Population::bind`), cold.

**Where the existing pieces stop:**
- *The first missing interface* is a WHEN leaf that names the head hop. A
  DT64 binding can carry the dependent's conditions. The rule's meaning
  (which verb frame) is lost: `when_guard_cannot_state_the_head_frame`
  shows the recipient and experiencer guards have equal witnesses.
- *Even a guard DT64 can state cannot run:* R2IL refuses it.

**ON/WHEN/DO verdict:**
- ON: the trigger (a token/annotation change) has no producer (#1942).
- WHEN: expressible for single-row conditions only.
- DO: a candidate role is a *pure derived relation* (a returned mask). A
  write is not needed, so EFFECTFUL DO must not be used for it.

**Shared WHERE scope:** all five rules read one population under one frame.
No rule consumes another rule's result, so no method-to-method mask handoff
exists (consistent with D-GLOBAL-WHERE, #1943).

## D. P0 falsifier — specification and results

**Fixture.** `src/fixtures.rs`, UD German-GSD convention, checked against
GSD master. GSD is CC BY-SA 4.0 and was fetched for measurement only.

**GSD census (train+dev+test), dative argument deprels:**

| verb | `obl:arg` + `Case=Dat` |
|---|---|
| *geben* | 31 (e.g. train-s1712) |
| *gefallen* | 18 (e.g. test-s258) |
| *helfen* | 15 |
| *schenken* | 6 |

- `iobj` occurs once in train and never in dev/test.
- `zur`/`zum` are multiword tokens (599 / 726).
- *bringen* + `zu`… is `obl` + `Dat` (48), e.g. "zur Reparatur gebracht",
  which is a **purpose**. "zur Schule bringen" is a **direction** with the
  identical surface.

**Sentences:**
- A *Die Mutter gibt dem Kind das Buch.*
- B *… bringt das Buch zur Schule.*
- C *Dem Kind gefällt das Buch.*
- D (control) *… gibt dem Kind mit dem Löffel den Brei.*
- E (control) *… hilft dem Kind.*

**Gold** (independent, single annotator, not adjudicated):
- Kind/A = Recipient
- Schule/B = Direction
- Kind/C = Experiencer
- Kind/D = Recipient
- Löffel/D = not on the map (instrument)
- Kind/E = Beneficiary (outside the frame lexicon)

**Rules.** Each is one prepared program; 37 ops in total.
- Recipient: `deprel=obl:arg ∧ Case=Dat ∧ head.lemma ∈ GIVE{geben, schenken}`
  (`EqU32Via`).
- Experiencer: the same with EXPERIENCE{gefallen}.
- Direction-or-Purpose: evaluated at the case marker,
  `deprel=case ∧ lemma=zu ∧ head.deprel=obl ∧ head.Case=Dat`, with result
  `ScatterOrU32(head)`.
- Abstain: `obl:arg ∧ Dat ∧ ¬framed`.
- Unknown: `obl:arg ∧ no Case ∧ framed`.

**Comparison of declarations** (the same Recipient rule):
- CG-3, sketch, not executed:
  `MAP (@Recipient) TARGET (Dat) IF (0 (obl:arg)) (p ("geben"))`. Here `p`
  is the parent test, which needs the dependency-aware VISL extensions.
- spaCy, sketch: `{RIGHT_ID:"v", RIGHT_ATTRS:{LEMMA:"geben"}} →
  {LEFT_ID:"v", REL_OP:">", RIGHT_ID:"r", RIGHT_ATTRS:{DEP:"obl:arg",
  MORPH:{IS_SUPERSET:["Case=Dat"]}}}`. A German model would emit TIGER `da`,
  not UD.
- MeTTa:
  `(match &tok (, (dep $v $n obl:arg) (case $n Dat) (lemma $v geben)) $n)`.
- **All three share the conjunctive core that the native fold runs.** What
  differs is what happens after a match: CG-3 *mutates* readings, spaCy
  returns *bindings*, MeTTa returns a *BindingsSet multiset*. None of those
  post-match semantics is in the fold.

**Results** (`cargo test --manifest-path crates/grammar-constraints-probe/Cargo.toml`):
12/12.

| measure | result |
|---|---|
| Correct vs gold | 4/4 |
| False positives | 0 |
| False negatives | 0 |
| Abstained (UNKNOWN) | 1 (E, helfen) |
| Ambiguity kept | 1 (B: Direction and Purpose both, no selection) |
| D instrument | no role |
| A vs C | same own lanes (`obl:arg`, `Dat`), different roles |

**Negative controls** (each predicted, each holds):

| # | control | predicted and observed |
|---|---|---|
| 1 | GIVE → LIKE (A's verb lemma swapped, dative unchanged) | Recipient gone, Experiencer appears |
| 2 | frame operand removed | 4 false positives (105, 302, 405, 505) |
| 3 | dependency direction swapped | no role anywhere |
| 4 | word order only ("Das Buch gibt die Mutter dem Kind") | identical roles by global id; the local position really moved |
| 5 | ambiguous attachment (PP under verb `obl` vs under noun `nmod`) | two readings, two populations: goal candidate in one, none in the other; not merged |
| 6 | supporting observation deleted | Case removed from Kind → UNKNOWN (not FALSE); `zu` deleted → no goal; Case removed from Schule → no goal |
| 7 | stale (foreign) frame | `Stale` refusal |
| 8 | `deprel` operand disabled | the instrument Löffel becomes a Recipient (false positive) |

**Source disables.** Each anchor was asserted present, and each disable
compiles. Nine guards were disabled, and every one turns ≥1 test red:
- frame hop: 5 tests red;
- `zu` marker: 2 red;
- head `obl`: c5;
- head `Dat`: c6;
- abstain `AndNot`: p0;
- unknown-case: c6, p0;
- frame check: c7;
- scatter lane: 3 red.

**Two disables first printed nothing, and both were caught:**
- *A compile error.* The first splice left a stray `}`. A disable that does
  not compile looks like a guard that does not bind.
- *A genuinely inert guard.* The head-`Dat` guard had no fixture that could
  expose it; the Schule-without-Case control was added for it.

**Cost** (`cargo run --release --example cost`, Xeon @ 2.10 GHz, 4 cores):

| phase | cost |
|---|---|
| Cold, reported not gated: vocab bind | ≈3.4 µs |
| Cold: DR map | ≈5.1 µs |
| Cold: population bind | ≈6.9 µs |
| Cold: rule prepare | ≈4.1 µs |
| Hot: all five programs incl. frame re-check | 1.19–1.33 µs/run over 40 tokens (best of 5 × 100k, three runs) |

- Cold numbers are single measurements, so they are noisy.
- Hot allocation: 0 bytes.
- Binding enumeration is not measured: the answers are masks. Reading out
  a role's tokens is the caller's `LocalMap::global` walk.
- No comparison with spaCy or DuckDB: no equivalent workload was run.

## P1 — cross-linguistic census (measured; not an executable test)

UD English-EWT is CC BY-SA. UD Latin-PROIEL is CC BY-NC-SA. Both were used
for measurement only; neither is committed. Counts are dependents of the
verb, by `(deprel, Case, case-marker)`:

| verb / function | German GSD | Latin PROIEL | English EWT |
|---|---|---|---|
| GIVE recipient | *geben* `obl:arg`+Dat 31 | *do* `obl:arg`+Dat 407 | *give* `iobj` 197 (161 with Case=Acc) · `obl`+*to* 23 |
| EXPERIENCE | *gefallen* `obl:arg`+Dat 18 | *placeo* `obl:arg`+Dat 63 | *like*: experiencer is `nsubj` 191 |

**Reading:**
- The constraint *shape* holds in all three languages: role = frame(head
  verb) × argument slot.
- The *slot* is language-specific:
  - German and Latin share `obl:arg`+Dat;
  - English splits the recipient over `iobj` and `obl`+*to*;
  - English realizes the experiencer of LIKE as the **subject** (the
    *please*-type verb would put it elsewhere).
- So a reusable abstract relation needs a per-language slot map, which is
  GF's concrete-syntax role. A shared surface rule would be wrong.

**Not verified:**
- No executable P1 tests.
- No Arabic.
- Passive voice and null subjects were not tested.

## E. Proposed next steps (smallest connection first)

1. **WIRE (lab), owner D-DT64-6 / EXO-140.** Decide whether a WHEN leaf may
   name a head-fk equality (`EqU32Via` over the population's own HEAD lane).
   It is the one operand the probe shows is load-bearing. Falsifier:
   controls 1, 2 and 8 above.
2. **PROBE.** Add a CoNLL-U → DR token-lane adapter, a cold one-time bind,
   and run the five rules over all GSD *geben* / *gefallen* / *helfen*
   tokens. Report precision, coverage and abstention against an independent
   role gold set. That gold set does not exist in UD and needs a sourced
   annotation (e.g. a valency lexicon) **before** any claim.
3. **OWNER DECISION.** Where a versioned language-specific frame lexicon
   lives: OGAR codebook / DT64 class, or not at all. Also: which of the two
   8-element role vocabularies is canonical for a *candidate* role.
4. **OWNER DECISION.** Candidate roles stay masks or Bag candidates. One
   CE64 orientation field cannot carry an alternative set, and promoting a
   candidate to a fact needs an explicit selection policy (HEURISTIC) or
   evidence (EVIDENTIAL/NAL), never silence.
5. **PARK.**
   - CG-3-style ordered REMOVE/SELECT: an effectful, order-dependent state
     transition; it would need DT64 ordered effects plus an iteration
     owner.
   - `<<` / `**` scans (EXO-73).
   - spaCy TIGER→UD crosswalk.
   - Enhanced UD (non-tree).
   - GF runtime.

**Verdicts:**
- **PROVEN (lab):**
  - the read-only role-candidate subset folds natively;
  - surface case does not decide the role;
  - word-order invariance by global id;
  - UNKNOWN is kept apart;
  - ambiguity is kept.
- **LAB:** `EqU32Via` as the head hop.
- **PROPOSED:** steps 1–4.
- **BLOCKED:**
  - WHEN execution (`GuardNotExecutable`);
  - 3VL execution;
  - recursion (EXO-73).
- **NOT VERIFIED:**
  - executable P1;
  - scale beyond 40 tokens;
  - Arabic;
  - passive / null subject;
  - CG-3 / spaCy run-time parity, since neither was executed.
