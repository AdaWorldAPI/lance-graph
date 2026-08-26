## 2026-08-22 — E-BPE-IS-RHYME-VQ-IS-THE-MECHANISM-FOR-6X2X8BIT-1 — the 256:256 tile codebooks are VQ-lineage; BPE shares numerology, not mechanism

**Status:** FINDING (literature-grounded, arxiv-grounder run 2026-08-22; graded
per-claim in the run's report). **Confidence:** High for the disanalogy; the
three probes below are the promotion path for anything stronger.

Asked (operator): does `6×2×8bit` (6× 256:256 centroid tiles) have synergies
with Byte Pair Encoding? Answer: **RHYME, not mechanism.**

- BPE (Gage 1994; Sennrich 2016, arXiv:1508.07909) is a **frequency-greedy
  adjacent-pair merge** approximating an entropy/prefix code — Huffman/LZ
  lineage ("Tokenization and the Noiseless Channel", arXiv:2306.16842). Its
  merge tree carries **no containment semantics**, its tokens are
  variable-length, its process is serial and corpus-global.
- The facet's codebooks are **nearest-centroid quantization over a fixed
  4-ary hierarchy** — VQ lineage. The honest literature bridge for `↑n`
  (exponential space at additive path cost) is **residual / tree-structured
  VQ** (RVQ: hierarchical codebooks, log-time tree search) and VQ-VAE
  (learned codebook as tokenizer) — not BPE. Unigram-LM (Kudo 2018) is the
  one tokenizer-family member with a comparable fixed-cardinality global
  objective, but it optimizes segmentation likelihood, not distortion.
- Named hazards of naive "BPE-ification": cross-axis merges on co-occurrence
  where no adjacency exists; loss of `is_ancestor_of` (BPE guarantees no
  subsumption); frequency-optimal replacing distance-optimal under lookups
  that assume distance coherence; a shared vocabulary violating
  classid-scoped codebooks.

**⊘ Provenance note, same day (transcript audited after the fact — the run's
own limits, so nobody cites this entry as more than it is):** the agent made
**9 real WebSearches and ZERO WebFetches**, and every citation checked
(`1508.07909`, `2306.16842`, `1910.13267`, `2305.07185`, `2602.22958`) appears
in a tool RESULT before it appears in the report — so nothing is fabricated,
but the evidence is **search-snippet level, never a paper read end-to-end**,
and several anchors are secondary summarizer pages rather than primary
sources. Its `[G]` grades are therefore over-graded by its own charter; read
them as strong `[H]`. Second limit, and the sharper one: it made **no Read or
Grep of the local files offered to it** (`facet.rs`, `attention_facet.rs`,
`ogar-loco/src/lib.rs`, OGAR `CLAUDE.md`), so every claim about OUR side is
the prompt's framing echoed back, not independent verification. What the run
genuinely establishes is the **BPE side** and the disanalogy; that our
codebooks are VQ-lineage remains our own (well-sourced, in-repo) claim, not
this run's finding.

Pre-registered probes (run before any code moves): (1) RVQ-vs-current
retrain of ONE axis under the 4-level constraint, measured on containment +
retrieval; (2) does distance-trained k-means already concentrate density on
the high-frequency `(coarse,fine)` pairs a BPE allocator would pick —
if yes, reallocation buys nothing; (3) containment-violation rate of a
BPE-trained table vs the centroid tree on the same address stream.

