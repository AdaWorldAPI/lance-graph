# NAL-1..NAL-9 oracle inventory — notes

Read 2026-10-11. Companion to `nal_inventory.tsv` (115 rows, one per capability; `lance_graph_mapping` left blank).

## Sources and how rows are marked

| tag in TSV | document | reached? |
|---|---|---|
| SPEC | Pei Wang, *Non-Axiomatic Logic (NAL) Specification*, 2009-10-30, https://cis.temple.edu/~pwang/Writing/NAL-Specification.pdf (covers IL, NAL-1..NAL-8; 8 layers) | yes, full text read |
| INTRO | Wang, *A Logical Model of Intelligence — An introduction to NARS*, Aug 2026, https://cis.temple.edu/~pwang/NARS-Intro.html | yes |
| SLIDES | Wang, *NAL-9 Implementation*, 10 slides, https://cis.temple.edu/tagit/presentations/NAL%209.pdf | yes |
| self.pdf | Wang, Li, Hammer, *Self-Awareness and Self-Control in NARS*, https://cis.temple.edu/~pwang/Publication/self.pdf | yes |
| Frontiers 2018 | Wang, Li, Hammer, *Self in NARS, an AGI System*, Frontiers Robot. AI 5:20 | yes (HTML) |
| ROADMAP | Wang, *From NARS to a Thinking Machine* (2007), https://cis.temple.edu/~pwang/Publication/roadmap.pdf | yes |
| book 2013/2025 | *Non-Axiomatic Logic: A Model of Intelligent Reasoning* | NO (World Scientific returns 403) |

`source_status`: SOURCE = read in one of the primary documents above; UNVERIFIED = only from an implementation or a fixture (no Wang text defines it); DISTILLATION = only from the repo `.arxiv/` notes. **No row is DISTILLATION-only**: every claim the distillations make about NAL-1..9 was re-found in a primary source. 7 rows are UNVERIFIED (projection/eternalization, negative goals, ^want, ^believe, ^evaluate, ^hesitate, remind/name/register); 108 are SOURCE. Where a row mixes a Wang concept with an implementation mechanism, the status column says SOURCE and the `source` cell carries a `[status note: ...]`.

Fixture prefixes: `ONR/` = `src/main/resources/nal/` in opennars/opennars at 4d1bdb0f599ccf6ed44edfc398f02f930713d30f (tree listed with `git fetch --filter=blob:none`; file names are real, not guessed). `ONA/` = OpenNARS-for-Applications at dc4efd0 (HEAD on 2026-10-11). `oracle_support_status` for ONA is from **reading** `src/NAL.h` and `src/Truth.c`, never from running ONA.

## Which levels change the grammar, which add rules, which add control

From SPEC §1.2 (2009): "NAL is established in multiple layers, each of which extends the logic by adding new grammar and inference rules, with proper addition of the semantics." INTRO §1.2 (2026) repeats it: "each new layer extends the sets of grammar rule and inference rule, as well as adding details in the memory/control mechanism."

What the specification actually adds per layer:

| level | grammar | inference rules / truth functions | control / execution |
|---|---|---|---|
| NAL-1 | judgment = statement + truth value; question; copula → | deduction, abduction, induction, exemplification, conversion, revision, choice, backward (question) derivation | evidential base (stamp) check |
| NAL-2 | copulas ↔, ◦→, →◦, ◦→◦; singleton sets {T}, [T] | comparison, analogy, resemblance; no rules for ◦→ etc. (I/O only, translated to →) | — |
| NAL-3 | ∩, ∪, −, ⊖; multi-element sets | composition (Fint, Funi, Fdif), decomposition | — |
| NAL-4 | ×, ⊥, ⊤, image placeholder | **none new**: SPEC §6.3 "There is no new inference rule directly defined in NAL-4, except the equivalence and implication propositions in the definitions and theorems" | — |
| NAL-5 | statement-as-term; ¬, ∧, ∨; ⇒, ⇔ | higher-order copies of NAL-1..3 rules (same truth functions), conditional rules, negation, contraposition | — |
| NAL-6 | variables (independent, dependent, query) | variable introduction / elimination / unification | — |
| NAL-7 | sequential/parallel conjunction; /⇒ \⇒ |⇒ /⇔ |⇔; tense | "variants of the rules defined in NAL-5 and NAL-6. The only additional function of these rules is to keep the available temporal information" (SPEC §9.3); temporal induction | internal clock; update vs revision |
| NAL-8 | '!' goals, desire values, operations (⇑op) | goal derivation meta-rule; desire merged by revision | decision rule, execution, sensorimotor interface |
| NAL-9 | — | — | internal operations, internal experience |

So the grammar changes at 1–8. NAL-4 adds grammar without new rules, NAL-7 adds grammar and mostly rule variants, NAL-8 adds grammar plus a control step (decision/execution). NAL-9 is not in the 2009 specification ("In the current design, there are 8 layers").

### Wang's statement about NAL-9 adding no grammar or inference rules

**I could not find an explicit Wang sentence saying "NAL-9 adds no grammar or inference rules."** The closest primary statements:

- INTRO, NAL-9 bullet: "When the operations involved in procedural inference are the internal operations of NARS, a self-referential loop is formed that gives the system the ability of self-awareness and self-control." It is defined as NAL-8 applied to internal operations, with no new syntax or rule named.
- ROADMAP §2(4) folds it into procedural reasoning: "Operations toward the outside of the system correspond to perception and control of the environment. Operations toward the inside of the system correspond to self-perception and self-control."
- Frontiers 2018: "the meta-level knowledge is built into the system and immune to experience-triggered modification. This level includes the grammar rules of Narsese, the inference rules of NAL, the basic routines of memory management and inference control, the set of mental operators, etc. Even taken self-awareness and self-control into consideration, this built-in core is still fixed."
- self.pdf §3: "This meta-level deliberative control does not change the underlying automatic routines, but supplement and adjust them."

Caution: INTRO §1.2's general sentence says each layer extends grammar and inference rules. Nothing in what I read applies that sentence to NAL-9 specifically. "NAL-9 is control/execution, not logic" is a reading of the sources, not a quotation. It may be in the 2013/2025 book, which I could not reach.

## Mental operations: what Wang names, what is implemented, what is open

- **Named in Wang's own texts.** self.pdf and Frontiers 2018 give the groups task generation, evidence disqualification (doubt = "decreasing its confidence value to a certain extent"), concept activation (consider), and feeling. Frontiers examples use `^wonder`, `^want` and `^feelSatisfied`. SLIDES name `^consider` (memory routine), `^observe`, `^request`/`^sense`, `^tell`/`^ask` (channel routine) and `^happy` (meta-goal).
- **Open according to Wang.** SLIDES slide 9 TASKS & ISSUES lists "Local operation: doubt, hesitate", "Communication: trust", task-invention triggers, correlated desire values/budget functions, separating known (confidence) from believed (frequency), and strategy abstraction. The repo distillation's correction holds: on the slides `^doubt` is an open issue, while the papers describe doubt as evidence disqualification.
- **Implemented in OpenNARS-Research** (`src/main/java/org/opennars/operator/mental/`): Anticipate, Believe, Consider, Doubt, Evaluate, Feel, FeelBusy, FeelSatisfied, Hesitate, Name, Register, Remind, Want, Wonder. Fixtures exist for anticipate, believe, doubt, evaluate, hesitate, want, wonder. The truth effects they pin (e.g. doubt halves confidence 0.90→0.45) are implementation choices; Wang gives no formula.
- **ONA:** no mental-operator set found in `src/`. Its NAL-9-adjacent behaviour is `{SELF}`-operations plus sensorimotor learning (examples `self*.nal`, `activequestion.nal`).

## Truth functions verified from a primary source (SPEC Table 11.13 and chapter tables)

Verified: evidence↔truth↔interval mappings, expectation, revision, deduction, abduction, induction, exemplification, conversion, comparison, analogy, resemblance, intersection, union, difference, negation, contraposition, and the decision-making function form.

Divergences an oracle must state up front:

1. **Intersection/union/difference confidence.** SPEC Table 5.3/11.13 uses an extended-Boolean `or(...) + and(...)` confidence. Both OpenNARS (`TruthFunctions.intersection/union`) and ONA (`Truth_Intersection`) use `c = c1*c2`, and fixture nal3.0 expects 0.81 from 0.9, 0.9. Spec-parity and implementation-parity differ here.
2. **Dependent-variable elimination.** SPEC uses Fabd (alternative in a footnote); ONA uses `Truth_AnonymousAnalogy`.
3. **Premise-order conventions** for abduction/induction differ between SPEC tables (J1/J2) and the implementations. Pin the order before comparing numbers.
4. **Resemblance** is labelled Fana in SPEC Table 4.2 (typo); Table 11.13 gives the Fres row.
5. **Decision rule** symbols in SPEC appear swapped relative to its own sentence ("desirability as frequency and plausibility as confidence").
6. **Not in any primary source I read:** desire-value functions per rule (OpenNARS desireStrong/Weak/Ded/Ind), temporal projection/eternalization, confidence discount for temporal rules (SPEC calls it "not implemented yet"), and the doubt/hesitate magnitudes.

## OpenNARS for Research vs ONA, in scope

- **OpenNARS (for Research)**, `opennars/opennars`, is Java. It aims to realize the whole NAL (INTRO §3.3: "OpenNARS is an open-source project that has been under development to realize the most up-to-date results of the research"). Its single-step corpus covers NAL-1..NAL-9 by file prefix: nal1.* 9, nal2.* 20, nal3.* 16, nal4.* 9, nal5.* 30, nal6.* 35, nal7.* 30, nal8.* 51, nal9.* 13, plus desire.1, notcontain. It has the mental operators listed above. It also has NAL-3/4 one-premise structural rules, contraposition, and the full variable machinery.
- **ONA**, `opennars/OpenNARS-for-Applications`, is C and "a completely new platform and not branched from the existing OpenNARS codebase" (README). Its `NAL.h` header says "Declarative NAL 1-6 reasoning abilities not covered by Inference.h" and "Mandatory NAL7/8 is not optional and handled by sensorimotor inference, see Inference.h". It is application-oriented: event handling and procedure learning (from ANSNA/20NAR1), and the ALANN control model. I found no contraposition truth function, no conversion rule in `NAL.h` (the function exists in `Truth.c`), and no mental-operator set. Examples are `.nal` scripts with `//expected:` comments, not `outputMustContain` assertions, so they are weaker oracles.

## Not reached / not done

- The 2013 and 2025 *Non-Axiomatic Logic* book (403).
- Any OpenNARS or ONA *run*. The ONA support column is from source reading, and OpenNARS support is "a fixture exists". Another agent runs the corpus.
- Frontiers 2018 section numbers were not checked (rows say so).
