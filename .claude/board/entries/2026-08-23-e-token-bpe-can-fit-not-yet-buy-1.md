## 2026-08-23 — E-TOKEN-BPE-CAN-FIT-NOT-YET-BUY-1 — BPE fits the fixed 6×(8:8) geometry reconstructibly; the merge tree is measurably NOT HHTL ancestry; nothing yet justifies a production carrier

**Status:** FINDING — [MEASURED] (`PROBE-TOKEN-BPE-GEOMETRY-1`, 8/8), on
ONE real fixture-scale corpus (the in-tree KJV Genesis 2–3 scene, 1125
bytes). **Scope fence:** this is TOKEN BPE (intake tokenization into the
existing 12-byte payload) — NOT behavioral BPE (recurring typed #1001/R2IL
transformations), which remains a separate queued investigation. Results
do not transfer between the two in either direction.
**Confidence:** High for what is measured; every number is fixture-scale,
and the scale corpora (COCA, whole-KJV, R2IL streams, AST intake) are
ABSENT from this checkout — reported absent, never simulated.

### The question and the verdict

> Can BPE act as a reconstructible intake tokenizer over the fixed
> `6×(8:8)` geometry without changing HHTL, classid semantics, or the
> resident memory ABI?

**CAN-FIT, NOT YET BUY.** It fits: 1125 bytes → 336 tokens (3.35×) at a
255-cap vocabulary, decoded byte-exact, packed into 28 resident `[u8;12]`
`Copy` particles, with no classid anywhere in the token path, no
token-object population proposed as canonical, no hash standing in for
content, and no ML machinery. Nothing at this scale justifies a
production token carrier.

### The three readings, measured

- **A — six independent pair subspaces:** works; slot-scoped word
  vocabularies (sizes 25–31 here) fit the LO lane with the HI lane free
  as a page. `u8:u8` stays two bytes, never a u16.
- **B — hierarchical/refinement pairs:** pair-ENCODABLE by construction
  (every merge is `(left:right)`, both ids u8) — but **measurably NOT
  lawful HHTL ancestry**: 3 same-depth token pairs are prefixes of each
  other, so "siblings" OVERLAP. A binary merge DAG over strings is not a
  radix prefix partition. **Encodability ≠ hierarchy** — the fence "do not
  confuse a merge tree with the ontology tree" is now a measured fact,
  not a warning.
- **C — BPE over already-lawful byte symbols:** the clean candidate.
  Compression and reconstruction both green; cost reported as OPERATION
  COUNTS (81852 encode probes, 1914 decode expansions), never wall time.

### Measured surprises worth keeping

1. **Scoped vocabularies LOST here** — per-chapter tables produced 19%
   MORE tokens than one global table, against the intuition that a scoped
   256-entry codebook wins. Weak signal (two chapters of one book), but it
   converts "scoped is obviously right" into "the comparison must be run
   per real corpus."
2. **The vocabulary saturated at 180 of 255** — merging stopped when no
   adjacent pair repeated ≥2×. The corpus, not the cap, set the vocab.
3. **Overflow is the norm, not the exception:** EVERY verse needs
   continuation (p50=4, max=8 particles per verse). A
   one-particle-per-item reading is refuted at verse granularity; any
   production design must budget continuation rows from the start.
4. **No HHTL locality:** chapter token-usage Jaccard 0.32 with heavy
   sharing — BPE stayed orthogonal to scope on this corpus, exactly as
   the law assumes rather than hopes.

### The authority order (the reconstruction law, enforced)

```
  canonical source        AUTHORITATIVE
  tokenized form          exact, reconstructible (measured byte-exact)
  compressed shorthand    must round-trip or is non-canonical
```

A token may accelerate access; it must not destroy the source semantics
required for reasoning. Falsifiers F4/F5/F13/F14 held structurally; F7
(merge-tree-as-ancestry) was made to FAIL measurably, which is the fence
working.

