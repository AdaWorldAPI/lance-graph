## 2026-08-21 — E-DISMECH-KNOWN-INTERMEDIATES-ARE-PROSE-NOT-IDENTITIES-1 — the 3,978-edge "ORACLE population" is 2,489 edges, and its mediators are 5-word prose, not node references

**Status:** FINDING (measured on upstream `monarch-initiative/dismech`, the same
2,100-file ephemeral `/tmp` corpus as `E-DISMECH-CORPUS-CENSUS-1`).
**Confidence:** High — every number is a count, and the first count was WRONG
in a way worth recording (below).

**This CORRECTS one sentence of `E-DISMECH-CORPUS-CENSUS-1`** (append-only: that
entry is not edited). Its census is right and stands — 9,073 / 4,539 / 3,978 /
408, total 17,998. What it also said is this:

> `IndirectKnownIntermediates` (3,978) is the ORACLE population — **the source
> names the mediators, so they can be hidden and recovery measured.**

The source mostly does not name them, and where it does, they are not
identities.

| | measured |
|---|---|
| `INDIRECT_KNOWN_INTERMEDIATES` edges | 3,978 |
| ...carrying >=1 named intermediate | **2,489** |
| ...with **no** `intermediate_mechanisms` key at all | **1,489 (37.4%)** |
| total intermediate strings | 3,424 (distinct 3,048) |
| strings per edge | mean 1.38, max 4 |
| **strings that are an exact node reference** | **45 / 3,048 (1.5%)** |
| string length | median 4 words, mean 5.2, max 39 |

A representative mediator: *"Classical-pathway inhibition yields serum
resistance, permitting spirochete survival during hematogenous dissemination."*
That is a sentence, not an address into the 48,467 distinct mechanism/target
names the corpus carries.

**Three consequences, in decreasing obviousness.**

1. **The oracle population is 2,489, not 3,978** — a 37.4% overstatement. The
   1,489 key-less edges are `KNOWN_INTERMEDIATES` **in label only**: the authors
   asserted mediators exist and did not write them down. They are neither
   oracle nor restraint control. Scored as positives they are unrecoverable and
   depress every metric; scored as negatives they punish a correct answer. They
   need a **third bucket**, or they poison the gold set. (This is the standing
   iron falsifier *"unknown intermediates are treated as negative examples"*,
   reached from an unexpected direction — via the KNOWN label, not the unknown
   one.)

2. **`Recall@k` over a mediator-identity candidate list has nothing to score
   against.** At 1.5% exact reference there is no identity-typed gold. A gold
   set must be MADE, and the making must not leak: grounding the 3,048 prose
   strings by **label matching alone** shares no machinery with DeepNSM or
   AriGraph, so it cannot launder the answer into the evaluation. Measured
   headroom against the 48,467 names: exact (normalized) 204 (6.7%), a corpus
   name is a substring 1,190 (39.0%), token-Jaccard >= 0.5 415 (13.6%) —
   **1,809 groundable (59.4%), 1,239 ungrounded prose (40.6%)**. ~1,800 gold
   mediators over ~2,489 supervised edges is enough to separate ablation
   levels. **The 40.6% residue must be reported, not dropped** — dropping it
   silently inflates every Recall@k, which is the vacuous-fence pattern this
   board already carries under a different name.

3. **`DismechTopology::source_knows_intermediates()` is correctly implemented
   and its second doc sentence is not.** The function answers *"does the source
   CLAIM to know"*, which is a question about the label and is exactly right.
   Its doc then says *"This is what separates the oracle population from the
   restraint control"* — and that is the falsified claim, because the
   label-KNOWN population is not the oracle population. `LATEST_STATE.md`
   carries the same overstatement ("the 3,978-edge ORACLE population"). **No
   bits move and no API changes**; the correction is to the claim, and the
   consumer that builds the oracle set must additionally require a non-empty
   `intermediate_mechanisms`, which the contract crate cannot see because it is
   deliberately source-side only.

**A free by-product, needing no grounding at all:** predicting the 4-way
`CausalTopology` of a masked edge is a benchmark over all **17,998** labelled
examples with a stated majority-class prior of 50.4% (DIRECT). It can run
before any grounding lands and calibrate the harness — which is worth having
precisely because the identity benchmark now needs a build step first.

**The method note, which generalizes past DisMech.** The first run of this
measurement returned *"4 of 3,978 carry a named intermediate"* and I nearly
recorded that. It was false: YAML block-list items sit at the SAME indent as
their key, and the sibling-scan broke on `indent <= key_indent`, so it stepped
off the list before reading a single item. The tell was the shape of the
result — a 0.1% rate on a field the corpus populates 2,489 times is not a
finding, it is a parser that missed. **A count that collapses to near-zero on a
field the schema clearly uses is a claim about the reader, not the corpus** —
the same rule this board already states for null probe results, applied to
counting rather than to timing. Reading three raw blocks cost thirty seconds
and moved the answer by a factor of 622.
