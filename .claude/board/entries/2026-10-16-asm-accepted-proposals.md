# 2026-10-16 — D-ASM follow-ups: learn measures, DAS attention probe, Atomese front end, MORK oracle

**Status:** DECISION (board rows only; nothing built).

Five proposals from the OpenCog / DAS / MORK mapping were accepted the same day.
This entry records where each one lands. The mapping itself (compute vs data per
repo) is in `2026-10-16-opencog-oracle-scope.md` and the D-ASM council spec; it
is not repeated here.

## Where each piece lands

| D-id | piece | home | gated on |
|---|---|---|---|
| D-ASM-1b | Atomese text front end: parse `(ConceptNode "cat")`-shaped s-expressions into the D-ASM-1 binder, names into the label codebook | `lance-graph-metta` binder scope (D-ASM-1); no new crate | D-ASM-1 |
| D-ASM-8 | `learn`-shaped measures over pair counts: marginals, MI / PMI, cosine, Jaccard, clustering input | `lance-graph-arm-discovery` (measure on the counts it already owns); NAL truth via `translator` | D-ASM-3 keyed counts |
| D-ASM-9 | `lance-graph-das`: attention-broker probe (`w = edge.count / node.count`, rent, spread) | QUEUED crate; created only when the probe runs | D-ASM-2 |
| D-ASM-10 | MORK as a fourth oracle (Hyperon, PeTTa, datafrog, MORK); built as a separate binary outside the workspace, results only | outside the workspace (like `hyperon-probe`) | D-ASM-4 harness |
| D-ASM-11 | leapfrog-join probe vs our semijoin chain on a cyclic 3-way join, written from the paper | probe in the D-MTP-1 harness | D-ASM-2 |

## Why `learn` gets no crate of its own

DECISION: no `lance-graph-learn`. `learn`'s counting pipeline is pair counts →
marginals → a measure (MI, cosine, Jaccard). `lance-graph-arm-discovery` already
owns pair-shaped integer evidence (`CandidateRule { cooccur, antecedent_count,
window }`) and its translation to NARS truth; an MI measure is a second reading
of the same counts. The counts themselves come from D-ASM-3
(`GroupReduce{Count}`), never from a loop in arm-discovery. The parse step that
produces the pairs (link-grammar MST) belongs to the grammar side, not here.

BASIS: separation of concerns holds without a new crate: counting = Quack
(D-ASM-3), measure = arm-discovery, truth = NAL (`lance-graph-nal`, global core
methods).

REVISIT WHEN: the measures need state arm-discovery cannot hold (an iterative
clustering loop, a model), or its dependency set would have to grow.

## OPEN

- The `AdaWorldAPI/learn` fork holds the callers (`scm/pipe-parse/pipe-count.scm`,
  `scm/gram-class/mi-similarity.scm`, both `(use-modules (opencog matrix))`), but
  the measure definitions live in the separate `opencog/matrix` repository
  (upstream HEAD `ee9f462` on 2026-10-16), which has no AdaWorldAPI fork. D-ASM-8
  is specified from its documentation and an oracle run, never from its source.
- No `atomspace-storage` fork exists under AdaWorldAPI; D-ASM-1b's round-trip
  oracle (the s-expression encoder/decoder) needs one, or an oracle run of the
  upstream binary outside the workspace.
- `AdaWorldAPI/atomspace-lance` (README only, 2026-10-16) plans a
  content-derived atom address; D-ASM-1b's binder must produce that same
  address, so the two are one design item, not two.
- MORK's `sink_hash_expr` test expects `XoicVnQv2bk` and gets `-Egab4rQ3Nc` on
  both stable and nightly; root cause unknown. Tests 62–70 of `mork test` never
  ran because the suite stops at the first failure.
