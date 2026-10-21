# Four Russians / M4RI and reverse lookup: lookup tables versus row copying

**Sources:** [M4RI repository](https://github.com/malb/m4ri) and Martin R. Albrecht, Gregory Bard, Clément Pernet, *Efficient Dense Gaussian Elimination over the Field with Two Elements* ([PDF](https://martinralbrecht.wordpress.com/wp-content/uploads/2011/11/ple.pdf)); plus Y. Lou and Y. Xu, *An Improved Method of the Four Russians for Inversion using Reverse Lookup Table*, IEEE ISIT 2026, [DOI:10.1109/ISIT62367.2026.11653817](https://doi.org/10.1109/ISIT62367.2026.11653817).  
**Research status (2026-10-21):** Abstract/readme-grounded extracts. IEEE work is **not an arXiv article**. No implementation or timing evidence for Lance-graph.

## Source-grounded excerpts

The classic Four Russians idea precomputes combinations for a **block** of pivot rows so a group of elementary operations may be applied through one indexed combination. In M4RI, those combinations use **GF(2) row XOR**. Blocking limits repeated scanning at the cost of a lookup table. Original M4RI gives an O(n^3/log n)-style reduction for GF(2) elimination.

Lou–Xu (2026) describe a **reverse lookup table** recording a *row index* for a given row value, rather than a cached *row vector*, and report retaining asymptotic time O(n^3/log n) with auxiliary storage O(n log n) instead of O(n^2). Treat those as the authors' algorithmic claims: no independent local verification and no inference that a QR256 truth matrix needs only 256 B.

M4RI (library) is GPLv2+; keep it as an **external oracle/process** or follow a clean-room procedure rather than copying source into a differently licensed production crate.

## Hypothetical transfer to QR256

A valuable **index-not-payload** pattern matches the user's zero-copy SoA aim: keep row combinations, codebook references or replay offsets, not duplicated payload vectors. But there are two distinct domains:

1. **GF(2) M4RI:** XOR cancellations, rank, inversion and parity constraints.
2. **Boolean-semiring blocking:** OR of intersections, existential relational composition, bounded reachability.

Substituting XOR for OR is forbidden. The second case needs its own small-table scheme or independent bit-matrix oracle; never call that M4RI correctness.

## Falsifiers / test path F

- **F1, GF(2) reference:** Compare scalar XOR Gaussian, blocked Four Russians and reverse-index experiment at dimensions 16,64,128,256. Require equal rank and row-equivalent echelon results, not byte-identical pivot order. Measure table construction, bytes, cache misses, XOR count and overall time.
- **F2, Boolean lookup:** For random/sparse/dense QR256 source×target tiles, compare blocked Boolean composition with scalar triple loop and `tile_gemm::BitMat::compose`. Deliberately include duplicate two-hop paths; OR must remain 1 while XOR may cancel to 0.
- **F3, 4-bit vs 8-bit block:** Measure lookup size and build/execute crossover including 1,8,64,1,000,000 subsequent folds. Preregister termination if LUT creation or cache misses outweigh saved Boolean operations.
- **F4, reverse index / zero-copy:** Hold stable borrowed Arrow lanes and compare a stored row vector with an index into the original row. Verify exact value equality, source generation invalidation and no hidden `to_vec` or payload copy. Count index arrays and metadata separately.
- **Silence twin:** Matrix with no profitable reuse chooses ordinary bit-masks or simple scalar path, rather than building a speculative LUT.

**Note:** A 256×256 Boolean matrix is 65,536 bits = 8 KiB, while 256 u8 selectors are only 256 B. The latter can *address* the former but cannot encode arbitrary 65,536 independent truth bits.
