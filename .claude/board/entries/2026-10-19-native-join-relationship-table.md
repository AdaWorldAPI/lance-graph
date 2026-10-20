# Native join over a versioned relationship table (2026-10-19)

**Status:** MEASURED (D-MPS-8), probe only; not admitted. Record: `.claude/plans/mork-pathmap-schnellverschluss-v1.md` §7b. Code: `crates/mork-pathmap-probe/src/join.rs`.

## Measured

- `Q(x,y,z) :- R(x,y), S(y,z)` with R and S laid out by the slot of `y` in one shared domain. The leapfrog `y`-level intersection is one mask-risc `And` over two presence planes; enumeration reads both sides at the surviving slot only. No gather. Native now answers in the differential suite, for set and bag semantics.
- n = 50 000: prepare 2.7 ms (PathMap 14.7, MORK 22.9). Set answers into a map 7.4 ms (PathMap 26.1, MORK 144). Streamed answers 0.33 ms; bag count 36 µs; joint plane 0.15 µs.
- One dataset resealed: 0.21 ms to re-prepare its side vs 2.47 ms for the table. 64 thoughts with their own `y` filter planes: 5.2 µs each over one shared table.
- The presence-plane `And` is an answer only as the existence mask; for enumeration and counts it prunes. Seven guards disable-verified.

## Open

- Cyclic bodies (triangle) need two variables co-addressed at once; chain joins over large domains would need a forbidden scatter.
- Admission of the global-to-local binding as a membrane (D-RMAP-1 BIND review).
