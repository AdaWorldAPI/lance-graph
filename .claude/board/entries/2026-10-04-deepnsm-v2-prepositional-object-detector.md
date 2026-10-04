# 2026-10-04 — prepositional object vs adverbial: which phrases leave the TEKAMOLO puzzle (D-LXC-24)

**Status:** MEASURED (debug-0, release) on UD r2.15 German HDT, train-a-1 → test. Kill bars were fixed before the run. Instrument: `crates/deepnsm-v2/examples/ud_pp_arg_eval.rs`.

Operator: *sich freuen **auf*** vs *auf* as place. A verb-governed phrase is an argument, not an adverbial, so it fills no TEKAMOLO lane. Each one detected narrows the TEKAMOLO "Sudoku". The verb also fixes its case.

## Gold, and a premise that was wrong at first

- **German `obl:arg` is NOT the prepositional object.** It marks bare dative objects (*der Telekom*, *ihr*) and carries a preposition once in GSD train.
- The first run used it and scored 0 positives on HDT.
- **HDT encodes the prepositional object as `obj` with a `case` ADP child.** The top train pairs are *rechnen mit* (163), *gehören zu* (155), *führen zu* (125), *sorgen für* (116), *setzen auf* (105), *warten auf* (36).
- **German GSD has no prepositional-object label** (its PP objects are plain `obl`), so this probe runs on HDT only.

## Results (HDT test, 20,564 verbal PPs, 1,098 prepositional objects = 5.3 %)

**A1, detector.** Mines (verb lemma, preposition) → object rate from train. At test, every known verb form in the clause (the span between punctuation) votes; the best pair with ≥ 3 train occurrences and rate ≥ 0.5 decides.

| | precision | recall | F1 |
|---|---|---|---|
| clause verb | **77.2 %** | **81.6 %** | 0.794 |
| preposition only | 0 % | 0 % | 0 |

- **PASS.** The TEKAMOLO pool loses 896 of 1,098 objects and only 264 of 19,466 adverbials.
- By preposition: *über* F1 0.92, *mit* 0.88, *auf* 0.83 (recall 90.3 %). The weakest is *in* (0.58).
- Disable run: dropping the rate check takes precision to 7.7 %, so the guard is load-bearing.

**A2, case of a detected object with a Wechsel preposition.** The (verb, preposition) table scores **96.8 %** against 81.0 % for the preposition's majority case, on 247 phrases: **PASS**. The phrase's case is the first `Case=` from the preposition to the head noun, because HDT often marks the article only; scoring on the noun fired 5 times.

## Abstract vs concrete heads (operator: *vor **der Fahrt***)

- The test is morphological: *-ung, -heit, -keit, -schaft, -tion, -nis, -ität, -ismus, -tum*.
- An abstract head is a prepositional object **10.7 %** of the time, against 4.6 % for other heads.
- In the `ud_case_eval` table (HDT train), abstract heads read as **goal or topic, not time**:
  - *auf* + abstract: Acc 243 vs Dat 77 (other heads 1,376 vs 1,550);
  - *über* + abstract: Acc 227 vs Dat 13;
  - *an* + abstract: Dat 73 %, against 96 % for *an* + time.
- **R3a** (case from the (preposition, abstract) table) scores 84.1 % vs 79.5 % on GSD (44 tokens, PASS) and **75.7 % vs 75.7 % on HDT, KILL**. The *auf* flip leaves the case unchanged, because *auf*'s overall Wechsel majority is already Acc.
- The verb prior on abstract heads is 68.9 %, against 61.5 % on other heads.

**OPEN.**
- A1 is not yet wired into a TEKAMOLO reader. The gain to lane accuracy is unmeasured: there is no German lane gold (see the D-LXC-23 addendum).
- Pronominal adverbs (*darauf*, *damit*) are the other half of the prepositional object and are not covered.
