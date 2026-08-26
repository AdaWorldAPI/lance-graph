## 2026-08-24 — E-PHI-WEYL-STAMP-CASCADE-PRECISION-RULING-1 — φ-Weyl 2-level Morton stamp cascade: coprime strides give identical discrimination, gcd>1 strides concentrate, operator ruled PRECISION over amortization

**Status:** FINDING — [MEASURED] (`PROBE-STAMP-MORTON-CASCADE-1`, 7/7).
Split out from the R2IL x BPE POC (`E-BPE-OVER-DEFUSE-CHAINS-BEATS-LINEAR-
AND-FITS-LOCO-1`) into its own PR — an orthogonal design decision that
muddied that POC's epistemic receipt (architecture review).

Builds a 2-level Morton cascade over the shipped `Stamp(u64)`
(`source(id) = 1<<(id%64)`). **Measured: for a bijective id->leaf map,
every coprime stride gives IDENTICAL discrimination** (39/17/11/41 all
produce the same word counts and the same conservatism, M3) — φ-Weyl
(`LEVEL0=39`, since `round(64/φ)=40` is NOT coprime; `LEVEL1=17`;
`WEYL_OFFSET=21`) is adopted as the CANONICAL choice (`[FORMAL-SCAFFOLD]`'s
φ-Weyl pillar), explicitly NOT a measured performance win. **The measured
TENSION:** coprime strides maximize spread (65 words at N=143, maximum
discrimination); `gcd>1` strides concentrate (17 or 9 words, coarser
resolution). Anti-moiré wants discrimination, amortization wants
concentration — operator ruled PRECISION, so spread was chosen. On the
real 143-episode corpus: cascade pooled=143 dropped=0, vs shipped flat
Stamp pooled=64 dropped=79 (55.2% CHOICE-dropped) — reproduces
`PROBE-STAMP-CAPACITY-1`'s K3 finding exactly, zero-loss on the cascade
side (M4). **Load-bearing constraint made explicit:** a stamp address
must be a pure function of the SOURCE ID, never arrival order — an
arrival-indexed Weyl walk would make `disjoint()` order-dependent and
meaningless.

**Correction on landing (codex review of the sibling POC PR, same finding
applies here since both probes share the file's history):** a prior
doc-comment claimed FINE-first's tier 0 "IS the shipped `Stamp`,
bit-identical." **Wrong post-φ-Weyl-adoption** — `root_leaf(0) = 21`, not
`0`. Corrected: tier 0 shares the shipped `Stamp`'s FOLD-CARDINALITY
SHAPE (one word, same mod-64 collision behavior) but is NOT bit-reusable
— a persisted `Stamp::source(id)` cannot be reinterpreted as this
cascade's root leaf without recomputing `root_leaf(id)`. M1/M4 compare
against the real `Stamp::source` directly (never `root_leaf`), so both
gates were unaffected by the doc-comment error; only the prose claim was
wrong, fixed before this file was split into its own PR.

**Fences:** no shipped type modified (`Stamp`/`TruthValue`/`SpoHead`/
`CausalEdge64` all read-only); `CascadeStamp` is a probe-local model,
input to the pending Step-2 stamp ruling, never the ruling itself; the
stamp tiers are an index over the `SpoHead` receipt stream, never a
second wire ABI; no OGAR mint anywhere in this file.

**Files:** `probe_stamp_morton_cascade.rs`.

