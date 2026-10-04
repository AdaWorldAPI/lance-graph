# 2026-10-04 — deepnsm-v2: a noun/verb homograph is decided by position, not by its lemma tag (D-LXC-13)

**Status:** MEASURED (`cargo run --release --example bible_wave -- pg10.txt`,
`v0.1.0-cam96-data` assets). Reopens D-LXC-3 (operator direction, 2026-10-04:
stop turning a word between subject and object into a noun because the corpus
counts it as a noun more often).

## What changed

- `coca::predicate_alternatives(tag, known)`: a lemma tag that is Noun or Verb
  gains the other reading when the lexicon has it. Function words keep their
  one tag. `bible_wave`'s tagger applies it to the lemma table's first row
  (F9 still names that row; it no longer settles a noun/verb homograph).
- `fsm` **slot rule**: right after a fresh subject (a noun that opened the
  clause in `Start` with no determiner or adjective in front), a token's Noun
  reading is dropped when it also has a Verb reading. Relative, like
  licensing: it never empties a token. A carried object ("gave him charge"),
  a re-anchored noun ("mothers house") and a determined subject ("the guard
  locks") do not trigger it.
- `ReadingParse::{slot_dropped, unlicensed_dropped}` counters; the KJV KILL
  gate accepts them as the explanation for a changed verse.

## Rejected on measurement

A sentence-level **clause rule** ("drop paths with no predicate when another
path has one") was built and removed. On its own Noun→Verb flips it was right
about 7 times in 34 (sampled): KJV verses are often verbless fragments ("the
goats for sin offering", "without blemish and without spot"), and the rule
forced a verb into them.

## KJV (`bible_vocab.txt`)

| | D-LXC-2 (lemma tag forced) | D-LXC-13 |
|---|---|---|
| ambiguous tokens / words | 3,363 / 141 | 34,824 / 451 |
| narrowed / still ambiguous | 1,908 / 1,455 | 14,383 / 20,441 |
| legacy tag eliminated | 179 tokens, 35 words | 1,943 tokens, 162 words |
| triples: certain + alternative | 69,670 + 1,716 | 57,350 + 29,001 |
| legacy→alternative / lost / new certain | 732 / 113 / 122 | 12,984 / 1,231 / 1,190 |
| verses changed | 1,131 | 12,437 |
| peak configurations / overflow | 16 / 0 | 256 / 0 (unchanged at a 1,024 cap) |

Of the 1,943 eliminated legacy tags, 1,817 are Verb→Noun (licensing after a
determiner or possessive: "his work", "the cry", "the mount"), 121 are
Noun→Verb (slot rule), and 5 are Adj. Hand-scored samples: Verb→Noun about 36/38 correct;
slot-rule Noun→Verb about 19/30. Every slot-rule token was a forced Noun
before, so where the rule fires it is a net gain. These are samples, not a
measured accuracy.

The certain set shrinks by 12,320: those triples hang on a homograph that
position does not decide, and they are now alternatives instead of being
asserted on the strength of a corpus count. Downstream gates (G1–G4, D-SRS-1,
-2, -4 PASS; D-SRS-3b's three KILLs) keep their verdicts; D-SRS-2's trie
target moves from `let` to `seen` on the smaller graph.

Ablation: with the lemma tag forced (no widening) and the slot rule on, only
1,112 verses change (certain 69,760) — the rule needs the alternative to be
in the reading set.

## OPEN

- Slot-rule errors: archaic possessive `mine`/`thine` read as a subject
  ("mine hand"), vocatives ("ye fools", "thou fool"), noun compounds after a
  bare noun ("meat offering", "water face"). The first is tagger work
  (`mine` before a noun is a determiner), not FSM work.
- `tesseract-paperless::consistency::seam_readings` still forces the lemma
  tag ("They record the deeds" → `record:Noun`); it adopts
  `coca::predicate_alternatives` after this merges.
- Downstream `--export` still carries certain triples only.
