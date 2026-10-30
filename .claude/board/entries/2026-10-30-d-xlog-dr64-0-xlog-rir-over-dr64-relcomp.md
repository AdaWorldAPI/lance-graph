# D-XLOG-DR64-0 — XLOG's compiled RIR lowered onto the DR64 relation register

**Date 2026-10-30 · Status LAB PROBE · Crate `crates/xlog-relcomp-probe` (standalone, not a member) · XLOG pinned `BrainyBlaze/xlog@bff6da66` (MIT OR Apache-2.0) · Claim: EXO-50 `XLOG-DR64-0-CLAIM-20261030-01`**

XLOG's CPU-only crates (`xlog-core`, `xlog-ir`, `xlog-stats`, `xlog-logic`) compile without CUDA and give parsing, RIR lowering and the magic-set decision. Its executor, PIR/exact probabilistic path and solver (`xlog-runtime`, `xlog-prob`, `xlog-solve`, `xlog-cuda`, `xlog-gpu`) depend on `cudarc` and do not run here. `xlog-core` pins arrow 60; the lance stack pins arrow 58, so XLOG cannot link into the lance-graph graph today. That is the one hard blocker found.

Gate B: `q(Y) :- r(1,Y), s(Y,4)` over R={(1,2),(1,3),(9,7)}, S={(2,4),(3,5),(7,4)}. Forward {2,3}, backward {2,7}, answer {2}, equal to scalar enumeration and to a tuple walk of XLOG's own RIR; 300-case differential passes. Backward is the preimage of {z} by the existing `relcomp::compose` against a one-column selector (no transposed store, no new opcode); typing is the existing `bind_dr64` + `Chain`. Magic sets target bound recursive queries only: `auto` declines, `on` errors.

Running it found two plan shapes XLOG's optimizer emits for the same rule: `ChainJoin` with the plain join as `fallback` (unbound ends; admitted only when both agree), and swapped join sides chosen from statistics (large unselective S). Admission of an optimizing frontend has to be order- and promotion-insensitive.

Refused, each with a test: inverse-needing join, negation, comparison, probabilistic program (deterministic RIR, probabilistic program), non-u32 fact, a scanned relation that a rule also derives (found by the Codex review), cross-domain alias (by the existing chain typing), axis mismatch, stale frame, 65 ids, ChainJoin/fallback disagreement. 9 disable runs red.

Cost (Xeon 2.8 GHz, release): hot bind + typing + propagate 267–406 ns on the selective fixture, where scalar enumeration is 11–15 ns; ~2.5–2.9 µs dense (backward `compose` does one row OR per edge of S); a prepared inverse column is ~0.9 ns. Masks beat scalar ~30x on dense. Cold is dominated by XLOG parsing fact literals (15,686 µs for 8,192 facts; reported, not a gate). Bidirectional filtering did not beat a prepared index.

Gate G: world enumeration gives P = 0.64, derivatives (0.8, 0.4, 0.32); noisy-OR over the two proofs gives 0.688. Scallop `scli` (oracle-only build) agrees: `topkproofs` 0.64, `addmultprob` 0.88, `minmaxprob` 0.6, `bool` true; it has no differentiable provenance, so derivatives are checked by enumeration only. The register path refuses probabilistic programs.

**OPEN:** arrow major alignment (or a facts-free `xlog-core`); facts from the substrate instead of XLOG source text; recursion and semi-naive deltas; cyclic bodies (local consistency is not an answer); bag multiplicity; K3. Gate D: `ShaderDispatch` has no field for an arbitrary row-selection mask (only `ColumnWindow`, meta prefilter, predicate-plane `layer_mask`) — a contract gap, nothing repurposed. Gates E, F not touched.
