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

## Update (same day, after #1584 merged)

DECISION: AtomSpace runs on two layers. `atomspace-lance` (AGPL) is the thin
`StorageNode` adapter, and it calls `lance-graph-cog` (Apache-2.0). The address
design, a deterministic content → NodeGuid mapping, is a `lance-graph-cog` item.
D-ASM-1b's binder uses that same function, so a MeTTa atom and an AtomSpace atom
with the same content get the same address. Tracked as D-ASM-12 (Queued, not
authorized). The crate is still not created.

DECISION: `lance-graph-cog` also carries the thinking API that OpenCog calls
(the PLN/ECAN-shaped questions). It is a facade over the algebra that already
exists: NAL operations from `lance-graph-nal` (global core methods), the
ONA-shaped term table and rule catalogue (D-ASM-5/6), procedures in
`lance-graph-r2il`, and the `lance-graph-relalg` IR, all lowering to mask-risc.
No new algebra, interpreter or NARS executor. OpenCog's own answers are the
oracle, and nothing is ported from its source. Tracked as D-ASM-13.

Closed open items:
- `AdaWorldAPI/matrix` and `AdaWorldAPI/atomspace-storage` now exist as forks, so
  D-ASM-8's definitions and D-ASM-1b's decoder oracle can run outside the
  workspace. They are read for documented behaviour and oracle runs only.
- MORK stable patch: pushed to `AdaWorldAPI/MORK` branch
  `ccr-9f3b8075-74u47l-stable` (`de9312e`). The upstream PR is opened by the
  maintainer.

D-ASM-11 shape (same day): which zipper the leapfrog runs over is decided by the
probe, not up front. Arm A is the cleanest and already integrated: `seek` is a
`partition_point` over the sealed ordered lanes, the same zipper merge on read
that `atomspace-lance` plans. Arm B is leapfrog triejoin over a prefix-shared
trie zipper, and it is something to learn from: it is worst-case optimal on
cyclic joins, which a semijoin chain is not. The probe records what B teaches
(variable order, prefix sharing, the seek pattern) and carries those lessons into
A even when A wins on cost per element; the margin for adopting B as the carrier
is registered before the run.

### Leapfrog Triejoin, read (Veldhuizen, arXiv 1210.0481v5, ICDT '14)

- Interface: `key`, `atEnd` O(1); `next`, `seek`, `open`, `up` O(log N); `seek`
  amortized O(1 + log(N/m)) when m keys are visited in ascending order, which
  needs a seek that moves forward from the current position (the paper: it
  "ascends just far enough"), not a search from the root.
- Relations need not be stored as tries; a trie is an iterator view over sorted
  data (B-tree-like in practice). A sealed NodeGuid-ordered lane already is that
  view: an aperture of n leading bytes is a trie level, `open` widens the
  aperture by one tile, `up` narrows it. So arm A is the paper's algorithm on our
  layout, and the first lesson is a galloping seek from the cursor in place of a
  fresh `partition_point` per call.
- Each relation's arguments must be a subsequence of the chosen variable order,
  so a relation used out of order needs a second sealed ordering. The variable
  order does not affect the worst-case bound; it does affect real cost.
- Bound: the triangle query R(a,b), S(b,c), T(a,c) with |R|=|S|=|T|=n has at most
  n^{3/2} answers (AGM, fractional edge cover); leapfrog triejoin runs in
  O(n^{3/2} log n); a hash variant (credited to Ken Ross) removes the log factor.
- The paper reports no experiments of its own.

Pre-registration for D-ASM-11 (before any code): fixture = the triangle query
at n = 10^3..10^6 on (i) random edges and (ii) an adversarial instance on which a
pairwise join builds Θ(n²) intermediate pairs; both arms and the semijoin chain
== datafrog as a bag; prediction: the semijoin chain's intermediate size grows
as n² on (ii) while both leapfrog arms stay within n^{3/2} visited keys; cost per
answer on the v4 arm only.
