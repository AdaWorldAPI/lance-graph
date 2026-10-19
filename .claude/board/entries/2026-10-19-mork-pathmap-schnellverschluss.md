# MORK / PathMap Schnellverschluss: source switch and differential probe (2026-10-19)

**Status:** MEASURED (D-MPS-0, D-MPS-1). Plan: `.claude/plans/mork-pathmap-schnellverschluss-v1.md`. Probe: `crates/mork-pathmap-probe`.

## Measured

- Forks against upstream: AdaWorldAPI/PathMap `ef02591` = upstream `master` (0 files differ). AdaWorldAPI/MORK `main` `ed57c67` = upstream `main` (0 files). The fork's stable branch `de9312e` differs in 18 files (+421 / −408) and is the only carried change.
- A git dependency on MORK does not resolve on its own: MORK names `pathmap = { path = "../PathMap/" }`, which lies outside its repository. One `[patch."https://github.com/AdaWorldAPI/MORK"] pathmap = { git = …, rev = … }` fixes it with one `pathmap` package in the graph and no edit to MORK.
- Upstream MORK `main` does not build on Rust 1.99 stable (`E0554`, nightly defaults); `de9312e` does. MORK's gxhash fork needs `+aes,+sse2`.
- MORK enables PathMap's `jemalloc` feature, which installs a `#[global_allocator]` in the library; a consumer binary cannot add its own (link error measured).
- MORK has no declared licence; upstream PR #148 (MIT) is not merged.
- `query_multi_leapfrog` takes a concrete `&PathMap<()>`; `SubtermCursor<Z>` is generic over PathMap's public `Zipper + ZipperMoving + ZipperPath + ZipperIteration` traits.
- Differential, `Q(x,y,z) :- R(x,y), S(y,z)`: MORK (product-zipper and leapfrog builds) and a PathMap read-zipper arm equal the nested-loop oracle on 8 cases (shared variable, empty, disjoint, per-atom scope, wide ids, duplicates, stale generation, ABA). Both upstream indexes are sets: bag requests are refused (`SetOnly`). Four disable runs each turned their guard's test red.
- Cost, n = 50 000, release, one run (re-run after review): PathMap index 14.8 ms (236 ns/row), query 24.1 ms; over 64 thoughts a shared index costs 1 556 ms in total vs 2 487 ms rebuilt per thought (1.6×; 24.3 vs 38.9 ms per thought). Sharing saves only the build. (The first version compared a per-thought figure with a 64-thought total and said 110×.) The MORK query time (170 ms) includes the probe's text decode and is not a measure of MORK's join.

## Open

- No admitted native lance-graph arm: the join reads `S` at positions chosen by `R`'s values (no-gather ruling). The native arm waits on a co-addressed producer.
- Whether a Lance-owned PathMap-trait cursor can walk a sealed `RelationOrder` without copying is unproved; `RelationOrder` is not built.
- The generation witness compares numbers, so a reused generation is not detected.
