# 2026-10-21 — D-RU-6-LAB-3: walk_mask with a dirty-frontier skip (arm A′), and the chain sweep

**Status:** MEASURED, lab only. D-RU-6 stays **Queued/Lab**. A′ is a gather path, like A, so it is not an ADMISSION-B candidate. It tests what causes arm C's chain win. It is not a production option. Follows D-RU-6-LAB-2 (`entries/2026-10-21-d-ru-6-sparse-compute.md`, #1648). Order set by the operator (EXO-50, 2026-10-21):
1. A′ first.
2. If C still wins against A′ on oracle-checked chains, a sweep over chain length, fan-out and number of foci.
3. Codebook attestation only if a convincing region remains after the sweep.

## What landed

`crates/crepe-plan-probe/examples/walk_dirty.rs`, a fourth arm. It compiles `walk_sparse.rs`, which compiles B and A, so all four arms run on the same graphs with the same oracles. A is unchanged. `walk_mask.rs`, `walk_tile.rs` and `walk_sparse.rs` gain only `pub(crate)`.

A′ is A's loop with two switches. Each switch can be measured on its own:
- **`skip_clean`.** Each delta plane carries a bitmap of its non-zero rows. Every adjacency word is masked with it before the scan, so a set bit whose delta row did not change costs nothing. A only skips whole lengths with nothing new.
- **`touched_only`.** A clears the whole `new` buffer every iteration. It also runs `andnot`, `any` and `or` over the full `n × w` span of each live length. With this switch, only the rows the hop wrote are updated, and only the rows that held a delta are cleared.

With both switches off, A′ is A. A test asserts equal tuples and an equal count of value-addressed row reads.

**Tests (8, plus 38 of A, B and C):**
- tuple-set equality with ascent, crepe and A for all four switch settings;
- mixed lengths with cycles; alternative path lengths; a zero-length ring with over-length edges; a 300-node chain across word boundaries; clustered, dense and cyclic fixtures; the sweep generator over two foci, where C is also checked;
- `skip_clean` masks bits on a chain, its row reads plus masked bits equal A's, and `touched_only` sweeps under a tenth of A's words;
- the silence twin: an all-dirty frontier masks nothing.

**Disable runs** (committed first, each reverted, all red):

| Disable | Red |
|---|---|
| dirty bitmap never refreshed | 7 tests |
| written rows not recorded | 7 tests |
| skip mask read from the wrong length | 7 tests |
| the "off" variant secretly skips | 6 tests |
| length bound made exclusive | 3 tests |
| old delta rows not cleared | the cleared-buffer check |

A missing row clear cannot change the answer: a stale bit is already in `walk`, and `new &= !walk` removes it. So the first run of that disable stayed green. A test-only assertion that `new` is clear at the start of every iteration now catches it.

CI runs `cargo test --example walk_dirty`. The benchmark and the sweep are not run in CI.

## Measured (release, 4 vCPU, load < 1.1, medians of 5; two full runs, consistent within noise)

**Count-only, ms** (first run):

| fixture | A | A′ skip | A′ touched | A′ both | C | A′both/A | C/A′both |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse(512) | 21.6 | 26.0 | 23.9 | 25.1 | 18.5 | 1.16× | 0.74× |
| clustered(512) | 18.1 | 24.1 | 34.8 | 22.0 | 50.9 | 1.21× | 2.32× |
| dense(128) | 2.79 | 2.24 | 3.46 | 2.75 | 5.49 | 0.99× | 1.99× |
| cyclic(300) | 191 | 212 | 334 | 349 | 712 | 1.82× | 2.04× |
| chain(230) | 6.42 | 7.94 | 10.3 | 4.92 | 4.00 | 0.77× | 0.81× |
| chain(700), UNCHECKED | 107 | 119 | 68.6 | 55.3 | 31.4 | 0.52× | 0.57× |
| mixed(256) | 9.09 | 12.5 | 13.8 | 13.2 | 95.3 | 1.45× | 7.24× |
| edges(512) | 2.71 | 3.01 | 1.58 | 1.50 | 4.56 | 0.55× | 3.03× |
| edges(1024), UNCHECKED | 10.8 | 11.1 | 5.99 | 6.35 | 30.3 | 0.59× | 4.77× |

**End to end with tuples, C/A′both:**
- chains: chain(230) 0.71× and 0.94× (the two runs), chain(700) 0.82× and 0.79×;
- sparse(512): 0.97× and 1.01×;
- edges(512) and edges(1024): 0.92–1.18×;
- clustered, dense, cyclic and mixed: 1.22–2.05×.

**Work at chain(700):**
- C's row ORs: 225,650. A′ makes exactly 225,650 value-addressed row reads with `skip_clean`, against A's 669,729.
- A′ still scans 14.6 M adjacency words and sweeps 9.9 M words. A sweeps 230 M.
- So C and A′ do the same row operations. The gap that remains is A′ scanning every adjacency row of every live length, plus A′'s rows being 11 words wide against a tile row's 4.

**Memory:** A′'s count-only resident growth is at or below A's where the planes are mostly empty: 3.0 against 14.3 MiB at edges(1024). It is slightly above A's on small cyclic or dense graphs.

### Chain sweep (`--sweep`: chain of `len` nodes, `fan - 1` extra edges to a node up to 8 ahead)

| len | fan | foci | A′ count | C count | C/A′ count | A′ e2e | C e2e | C/A′ e2e | oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 128 | 1 | 1 | 1.10 | 1.56 | 1.41× | 3.18 | 3.36 | 1.06× | checked |
| 128 | 2 | 1 | 1.41 | 2.11 | 1.50× | 7.03 | 8.01 | 1.14× | checked |
| 128 | 4 | 1 | 1.13 | 1.53 | 1.36× | 6.63 | 7.30 | 1.10× | checked |
| 256 | 1 | 1 | 6.51 | 5.13 | 0.79× | 15.6 | 15.3 | 0.98× | checked |
| 256 | 2 | 1 | 8.89 | 10.5 | 1.18× | 54.0 | 56.8 | 1.05× | checked |
| 256 | 4 | 1 | 11.5 | 9.01 | 0.78× | 54.5 | 57.9 | 1.06× | checked |
| 384 | 1 | 2 | 16.3 | 10.9 | 0.67× | 38.6 | 30.6 | **0.79×** | checked |
| 384 | 2 | 2 | 34.4 | 34.5 | 1.00× | 171 | 192 | 1.13× | checked |
| 384 | 4 | 2 | 32.1 | 24.6 | 0.77× | 178 | 175 | 0.98× | checked |
| 512 | 1 | 2 | 22.1 | 19.5 | 0.88× | 60.4 | 57.6 | 0.95× | checked |
| 512 | 2 | 2 | 51.1 | 69.2 | 1.35× | 376 | 443 | 1.18× | checked |
| 512 | 4 | 2 | 48.1 | 56.1 | 1.17× | 349 | 422 | 1.21× | checked |
| 768 | 1 | 3 | 68.2 | 39.0 | 0.57× | 155 | 126 | **0.81×** | UNCHECKED |
| 768 | 2 | 3 | 277 | 222 | 0.80× | 1,321 | 1,566 | 1.19× | UNCHECKED |
| 768 | 4 | 3 | 242 | 183 | 0.76× | 1,238 | 1,521 | 1.23× | UNCHECKED |

(ms; one sweep run.)

## Reading

- **A′ against A.** Both switches together make walk_mask 1.3× faster on chain(230), 1.9× on chain(700) and 1.7–1.8× on the benchmark generator. They make it 1.2–1.8× slower on clustered, cyclic and mixed-length graphs and 1.16× slower on sparse(512), where the bookkeeping does not pay. `skip_clean` alone never helps: it cuts the gathers but not the whole-buffer sweeps. `touched_only` alone helps on long chains (68.6 against 107 ms at chain(700)); together they help most.
- **A′ does not remove C's chain advantage.** C stays ahead of A′ on chain(230), count-only and end to end, and on chain(700). It does the same row operations as A′ there.
- **The sweep confines C's end-to-end advantage to unbranched chains.** C beats A′ end to end only at fan-out 1 with at least 384 nodes: 0.79× at 384 and 0.95× at 512 (both checked), 0.81× at 768 (UNCHECKED). At 256 nodes and fan 1 it is at parity (0.98×). At fan-out 2 and 4 it loses end to end on every checked point but one (1.05–1.21×; 384/4 at 0.98×), even where it wins count-only (0.76–0.80×), because decoding dominates once branching multiplies the tuples.
- **The cause is not tiles as such.** It is skipping by delta occupancy instead of scanning every adjacency row: C's loop is driven by the changed delta blocks, A′'s by all rows `x`. A pull formulation of A′ (iterate the dirty rows `y` through a reverse adjacency) would test that directly. It is not built here.

## Against the operator's gates

- C stays faster than A′ on oracle-checked long chains, so the sweep ran.
- After the sweep, the end-to-end region where C wins is narrow and structural: unbranched chains of at least 384 nodes, with a gain of 5–21 %.
- Recommendation for the operator: **defer the codebook attestation.** No region with branching survives, and the remaining gain comes from a loop-order difference that a pull variant of A′ may also capture.
- D-RU-6 stays Queued/Lab. No production admission.

## Open

- A pull-form A′ (dirty `y` through a reverse adjacency), to check whether loop order alone accounts for C's remaining chain gain.
- Decoding cost: it dominates every branched case and is not addressed by any arm.
- Sweep repeatability: one sweep run, on a quiet host; single points such as 512/4 for A vs A′ sit within noise.
