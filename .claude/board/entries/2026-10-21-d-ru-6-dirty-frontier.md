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

**Tests (9, plus 38 of A, B and C):**
- tuple-set equality with ascent, crepe and A for all four switch settings;
- mixed lengths with cycles; alternative path lengths; a zero-length ring with over-length edges; a 300-node chain across word boundaries; clustered, dense and cyclic fixtures; the sweep generator over two foci, where C is also checked; the sweep generator builds distinct successors (exactly `fan` per node away from the tail, no duplicate edge);
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
| extra successors drawn with replacement | the distinct-successor test |

A missing row clear cannot change the answer: a stale bit is already in `walk`, and `new &= !walk` removes it. So the first run of that disable stayed green. A test-only assertion that `new` is clear at the start of every iteration now catches it.

CI runs `cargo test --example walk_dirty`. The benchmark and the sweep are not run in CI.

## Measured (release, 4 vCPU, load ≈ 1, medians of 5)

**Correction (codex review on #1652).** The first version of this entry had three faults, fixed before these numbers:
- **End-to-end timing was not like for like.** For A and A′ it included the count pass; for B and C (`e2e()`) it did not. A and A′ end to end now time the fixpoint without counts, plus decoding.
- **The sweep's fan-out was overstated.** `chain_fan` drew extra successors with replacement, so fan 2 and 4 contained duplicate edges. It now draws without replacement, and the sweep reports the distinct out-degree it built: 1.00, 1.99–2.00 and 3.95–3.99.
- **The claimed cutoff contradicted the entry's own data.** "C wins only on unbranched chains of ≥ 384 nodes" conflicted with the checked chain(230) result. The sweep now samples 128, 192, 230, 256, 320, 384, 512 and 768 nodes, and it was run twice.

**Count-only, ms:**

| fixture | A | A′ skip | A′ touched | A′ both | C | A′both/A | C/A′both |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse(512) | 26.3 | 33.2 | 28.4 | 28.3 | 24.4 | 1.08× | 0.86× |
| clustered(512) | 25.3 | 28.7 | 24.7 | 34.9 | 58.2 | 1.38× | 1.67× |
| dense(128) | 2.95 | 2.29 | 3.57 | 2.77 | 6.15 | 0.94× | 2.22× |
| cyclic(300) | 213 | 279 | 434 | 413 | 880 | 1.94× | 2.13× |
| chain(230) | 5.88 | 5.74 | 7.60 | 4.83 | 4.37 | 0.82× | 0.90× |
| chain(700), UNCHECKED | 135 | 123 | 79.0 | 64.6 | 35.3 | 0.48× | 0.55× |
| mixed(256) | 9.97 | 10.2 | 16.1 | 21.1 | 116 | 2.12× | 5.48× |
| edges(512) | 3.57 | 3.86 | 2.37 | 2.27 | 5.27 | 0.64× | 2.32× |
| edges(1024), UNCHECKED | 15.8 | 16.2 | 7.44 | 8.65 | 35.2 | 0.55× | 4.06× |

**End to end, tuples decoded, C/A′both** (C/A in brackets):
- chain(230) 0.79× (0.77×); chain(700) 0.82× (0.60×), UNCHECKED;
- sparse(512) 0.91× (1.12×); edges(512) 1.10× (1.00×); edges(1024) 1.07× (0.84×), UNCHECKED;
- dense 1.25×, mixed 1.66×, clustered 1.78×, cyclic 2.10×.

**Work at chain(700):**
- C's row ORs: 225,650. A′ with `skip_clean` makes exactly 225,650 value-addressed row reads, against A's 669,729.
- A′ still scans 14.6 M adjacency words and sweeps 9.9 M words; A sweeps 230 M.
- C and A′ therefore do the same row operations. What remains is A′ scanning every adjacency row of every live length, and A′'s rows being 11 words wide against a tile row's 4.

**Memory:** A′'s count-only resident growth is at or below A's where the planes are mostly empty (3.0 against 14.3 MiB at edges(1024)), and slightly above A's on small cyclic or dense graphs.

### Chain sweep, C/A′ end to end (two runs: run A / run B)

`--sweep`: a chain of `len` nodes; each node gets `fan - 1` extra edges to distinct nodes 2–8 ahead. Up to 512 nodes every point is checked against ascent and crepe; 768 is UNCHECKED. Foci: 1 up to 256 nodes, 2 for 320–512, 3 for 768.

| len | fan 1 | fan 2 | fan 4 |
|---:|---:|---:|---:|
| 128 | 1.25 / 1.08 | 1.19 / 1.04 | 1.34 / 1.36 |
| 192 | 0.95 / 0.97 | 1.10 / 1.21 | 0.94 / 1.05 |
| 230 | 0.95 / 0.75 | 0.91 / 0.97 | 1.12 / 0.87 |
| 256 | 1.01 / 0.94 | 1.48 / 0.91 | 1.07 / 1.07 |
| 320 | 0.81 / 0.86 | 1.09 / 1.02 | 1.19 / 1.18 |
| 384 | 0.72 / 0.94 | 1.02 / 1.07 | 1.11 / 1.10 |
| 512 | 0.78 / 0.90 | 1.31 / 1.27 | 1.24 / 1.19 |
| 768 (UNCHECKED) | 0.75 / 0.70 | 1.04 / 1.01 | 1.09 / 1.14 |

Count-only, C leads A′ more often: fan 1 from 192 nodes on (0.46–1.04×), and fan 4 from 192 nodes on except at 512 (0.60–1.00×; 1.14–1.16× at 512). Single cells move by up to 0.3 between runs (256/2: 1.48 against 0.91), so only patterns that hold in both runs are read below.

## Reading

- **A′ against A.** Both switches together make walk_mask 1.2× faster on chain(230), 2.1× on chain(700) and 1.6–1.8× on the benchmark generator. They make it 1.4–2.1× slower on clustered, cyclic and mixed-length graphs, and 1.08× slower on sparse(512), where the bookkeeping does not pay. On chain(230) neither switch alone helps by more than 2 %. On chain(700) `touched_only` alone helps (79.0 against 135 ms) and both together help most.
- **A′ does not remove C's chain lead.** On chain(230) and chain(700), C stays ahead of A′ count-only and end to end, while doing the same row operations.
- **Where C wins end to end against A′ (both runs agree):** unbranched chains (fan 1) from 192 nodes upward, by 3–30 %. At 256 nodes it is at parity (1.01 / 0.94). At 128 nodes C loses.
- **With branching it mostly does not hold.** At fan 2 and 4, C loses end to end in both runs at most points (1.01–1.36×). One branched point wins in both runs: 230 nodes at fan 2 (0.91 / 0.97). Three more win in one run only: 192 and 230 at fan 4, 256 at fan 2. Decoding dominates once the graph branches, even where C wins count-only.
- **The cause is loop order, not tiles.** C's loop is driven by the changed delta blocks; A′'s visits every row `x`. A pull form of A′ (dirty `y` through a reverse adjacency) would test that directly. It is not built here.

## Against the operator's gates

- C stays faster than A′ on oracle-checked long chains, so the sweep ran.
- After the sweep, the end-to-end region where C wins is narrow and structural: unbranched chains of about 192 nodes or more, with a 3–30 % gain (one branched point, 230 nodes at fan 2, also wins in both runs). The earlier "≥ 384 nodes" was wrong (see the correction above); the lower edge lies between 128 and 192 nodes.
- Recommendation for the operator: **defer the codebook attestation.** No branched region survives, and the remaining gain comes from a loop-order difference that a pull form of A′ may also capture.
- D-RU-6 stays Queued/Lab. No production admission.

## Open

- A pull-form A′ (dirty `y` through a reverse adjacency), to test whether loop order alone accounts for C's remaining chain gain.
- Decoding cost: it dominates every branched case and none of the arms addresses it.
- Noise: two sweep runs on a quiet host, with single cells moving by up to 0.3; finer boundary claims need more repetitions.
