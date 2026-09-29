# deepnsm-v2-lexical-evidence-consumer-v1

**Status:** PROPOSAL (2026-09-29). No code authorized. Written against `main`
`5282dfa3` (the #1299 merge).

## Overview

#1299 made DeepNSM-v2 keep every counted PoS reading per `WordId`
(`crates/deepnsm-v2/src/lexical.rs`). Nothing reads it yet. The only PoS
consumer, `examples/bible_wave.rs::load_pos` (`:1023`), still builds a
`HashMap<String, Pos>` with `entry().or_insert_with` — first reading wins,
counts dropped — from `lemmas_5k.csv` and `word_forms.csv`. The loss #1299
fixed in storage is still live at its one caller.

Next part: one read method on `LexicalEvidence` plus one caller switched to it.
Still inside the #1299 boundary: DeepNSM-v2 preserves evidence; it does not
perform the CausalEdge64 epistemic transition.

## Checklist

- [ ] Tests first (below), each seen red before the change.
- [ ] `LexicalEvidence::fsm_readings(id)` — the source PoS byte mapped to the
      six-state FSM `Pos`, every reading kept, counts of tags that fold into
      one state (`n`,`p` → `Noun`; `a`,`d` → `Det`) summed with `checked_add`,
      unknown stays `None`. No selection inside the method.
- [ ] Move the tag → `Pos` fold out of the example (`coca_pos`,
      `bible_wave.rs:980`) into the library so there is one spelling.
- [ ] `bible_wave`: build `LexicalEvidence` with `load_word_forms_csv` against
      `nsm.vocab`; pick the reading per `WordId` by an explicit, named rule
      (highest known count, fixed tie-break, `None` never beats `Some`),
      falling back to `archaic_pos` then `Pos::Other` exactly as today.
- [ ] Measure before/after on the full KJV run: triple count, and the number
      of tokens whose `Pos` changed. Record both; no expectation pinned in
      advance.
- [ ] Confirm routing ids and Cam96 `codes[word_id]` are byte-identical.
- [ ] Commit `.claude/settings.json` `"attribution": {"commit": "", "pr": ""}`
      in the same PR (operator request, 2026-09-26).

## Tests

1. A homograph (`record`: `n` 120,048 / `v` 13,014) returns both readings from
   `fsm_readings`.
2. A fold sums: `n` + `p` on one surface returns one `Noun` with the summed
   count; overflow is an error, not a wrap.
3. Unknown count on any folded reading gives `None` for that state.
4. The selection rule picks `Noun` for `record`; a disable run (first-wins)
   picks by row order and must turn test 4 red.
5. Routing and Cam96 unchanged (reuse the #1299 invariance tests).

## Details

- The FSM takes one `Pos` per `Tagged` (`fsm.rs:66`). Passing several readings
  into `parse_to_spo` needs an FSM change and is **out of scope**; this PR only
  replaces first-wins with a counted choice made outside the FSM.
- `lemmas_5k.csv` (25 columns, lemma level) has no loader in `lexical.rs`.
  `load_pos` reads it first today. Open decision: drop it from the consumer
  (forms already carry `lemFreq`) or add a second loader. Not decided here.
- `academic_20k.csv` stays out: its three duplicate (word, PoS) rows need an
  operator ruling first.
- Out of scope: CausalEdge64, cognitive-shader-driver, arm-discovery, the
  mask-risc fold → next-cycle seam.

## All candidate next parts (2026-09-26 .. 09-29 thread)

Ranked. Item 1 is the checklist above; the rest are recorded so the
convergence session and the operator see the whole set.

1. **Lexical-evidence consumer** (this plan). Smallest; closes the first-wins
   loss at its only caller; gives any later hydration step a real counted
   input.
2. **FSM takes several readings per token.** The honest end state of item 1:
   `Tagged` carries candidates, `parse_to_spo` resolves ambiguity with
   context instead of a counted guess. Needs an FSM change; separate PR.
3. **`lemmas_5k.csv` loader.** Lemma-level file (25 columns: freq, range,
   dispersion, genre splits). Either a second loader in `lexical.rs` or drop
   it from the consumer because `word_forms.csv` already carries `lemFreq`.
   Operator decision.
4. **`academic_20k.csv` loader.** Blocked on a ruling: 3 pairs of rows share
   (word, PoS) with different counts. Disjoint and summable, or duplicates?
   Until ruled, the builder refuses the second as `DuplicateReading`.
5. **Counts for the Cam96 vocabulary.** `bible_vocab.txt` is surface-only (no
   counts, no PoS). The Tigris bucket (checked 2026-09-26 via the `AWS_*`
   credentials) holds only `academic_20k.csv` (identical to the committed
   copy) and a first-wins codebook TSV — no lemma-level or per-form counts.
   KJV-own counts would have to be computed from the corpus, not fetched.
6. **Mask-risc fold → next cycle.** From the 2026-09-26 `.claude/v3` review:
   the smallest missing executable seam across the cycle boundary is a
   mask-risc fold result at Lance version v entering cycle v+1 as a staged
   cast. Most value, not small, and it touches the CausalEdge64 area whose
   PRs #1293–#1295 were reverted. Needs its own plan.
7. **Repo attribution setting.** `.claude/settings.json`
   `"attribution": {"commit": "", "pr": ""}`; rides with item 1.
8. **History scrub of the `Opus 5.5` trailers.** Deferred by the operator:
   no force push. The two #1299 commits (`31f7d26f`, `eee1b17a`) and seven
   older ones keep the trailer; new commits carry none.

## Convergence

A second session holds additional ideas for this area. Its brief is
`.claude/prompts/deepnsm-v2-lexical-consumer-converge.md`. Its answer lands as
a v2 of this plan, never as parallel code.
