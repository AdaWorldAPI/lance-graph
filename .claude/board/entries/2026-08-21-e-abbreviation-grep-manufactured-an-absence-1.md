## 2026-08-21 — E-ABBREVIATION-GREP-MANUFACTURED-AN-ABSENCE-1 — I reported a shipped 15-module subsystem as non-existent because `fn .*ppr` matches `approx`, and a `head` limit hid the real hits

**Status:** FINDING (self-inflicted, caught by the operator pointing at
`E-ARIGRAPH-IS-AN-ISLAND`). **Confidence:** High — the mechanism is
reproducible in one line.

**What I claimed, in a deliverable handed to the operator:** *"the assumption
is stale — no `ppr`, `personalized_page*`, `bm25`, `rrf`, or `community`
function exists in the workspace… AriGraph appears only as narrative in doc
comments."*

**What is actually there:** `crates/lance-graph/src/graph/arigraph/` — **15
modules, ~327 KB** — `ppr.rs` (`PersonalizedPageRank`,
`personalized_pagerank()`), `bm25.rs` (`Bm25Index::{build,score,rank}`),
`rrf.rs` (`reciprocal_rank_fusion()`), `community.rs` (`Communities`),
`markov_soa.rs`, `episodic.rs` (`EpisodicBasins`), `witness_corpus.rs`,
`retrieval.rs`, `triplet_graph.rs`, `orchestrator.rs`, and five more, all
re-exported from `mod.rs`.

**The mechanism, in two compounding parts.**

1. **The abbreviation collided with a common word.** My pattern was
   `fn .*ppr\|fn .*bm25\|fn .*rrf\|fn .*community`. `"ppr" in "approx"` is
   **`True`** — `a-p-p-r-o-x`. Every `fn approx(a, b, tol)` test helper in
   `jc`, `holograph`, and `sigker` matched. The real functions matched
   *nothing*, because they are spelled out — `personalized_pagerank`,
   `reciprocal_rank_fusion` — and `bm25` lives in the TYPE name, not the
   function name.
2. **`head -20` then converted noise into a false negative.** `community.rs`
   genuinely had 5 matching lines. They were pushed off the end of the output
   by the `approx` flood. I read "nothing relevant in the first 20 lines" as
   "absent."

Neither part is sufficient alone: a noisy pattern with full output would have
shown the real hits; a clean pattern with a `head` limit would have found them
first. **A pattern with a high false-positive rate and an output limit are
individually survivable and jointly a lie.**

**The rule.** *Searching for a function's NAME is not searching for a
CAPABILITY.* Before reporting a subsystem absent, search the **filesystem** for
it — `find . -path "*arigraph*"` would have ended this in one call, needed no
guess about spelling, and cost less than the grep did. Names encode an author's
abbreviation preference; directories encode the subsystem. Corollaries:

- An abbreviation of ≤4 characters is a **substring**, not a token. Check it
  against ordinary vocabulary before trusting it (`ppr`⊂`approx`,
  `rrf`, `cam`⊂`camera`, `ppr`⊂`suppress`).
- **Never conclude absence from a limited-output search.** Absence is a claim
  about the whole set; a `head` reads a prefix of it. Re-run with a `-c` count
  or no limit before writing "does not exist".
- **Absence and unwiring have opposite remedies.** Absent ⇒ build the organs.
  Unwired ⇒ build nothing, close the seam. I recommended the expensive one.

**This is the second instance of a named class.** The board already carries
*"a false negative manufactured by the intake"* (the sorted-histogram gate that
discarded which cell mapped to which offset, codex P1 on #876). Same class,
different mechanism: there the intake destroyed structure before comparison;
here the intake's own noise outran the output window. Two instances with
disjoint mechanisms make it a pattern rather than an anecdote: **the intake is
part of the measurement, and it fails silently in whichever direction nobody
checked.**

**What the correct answer would have been**, and it was already on this board
under `E-ARIGRAPH-IS-AN-ISLAND`: every AriGraph module exists and tests green;
the chain is open at the joints; `HotWitness` is `todo!()`;
`Ee→EW64(hot)+WitnessCorpus(cold)` is the unwired task. *"The most expensive
kind of gap: invisible in green suites (every crate passes; the system doesn't
do the thing) because the integrating seam was never built."* I cited that
entry as prior art for a different claim **without reading it** — and it
contained the correction to the claim I was making in the same breath.

