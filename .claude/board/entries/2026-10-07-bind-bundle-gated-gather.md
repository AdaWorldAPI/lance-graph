# 2026-10-07 — One bound query, five physical routes; the semijoin is never gated

## MEASURED

`crates/lance-graph-benches/examples/bundle_probe.rs`, 1M rows, release,
median of 15. Query bound to lanes/planes only:
`sel < t AND dept_fk ∈ allowed` → `count`, `sum(age)`; 8 densities × 2 layouts
× 2 terminals = 32 cases.

- **All five routes returned the oracle's answer in 32/32 cases**: `quack::lower`
  (gated), `lower` with conjuncts reversed, `lower_fused`, a sorted-ordinal
  selection vector, and a mask-native gated gather (emulated).
- The three mask-risc routes are flat at ~5.5–6 ms at every density; the
  semijoin alone is ~5.2 ms. `MaskOp::Gather` has no `under`, so survivor
  gating never reaches the dominant op and conjunct order is inert.
- The ordinal vector wins below ~25 % density; the gated mask gather wins
  more at every density below 100 % (0.01 %: 209 µs vs 609 µs; floor ≈ the
  cmp-only cost, ~200 µs), with no selection vector.

## CONSEQUENCE

- quack matrix R1 ("no selection vector") survived its falsifier on this
  workload.
- The gap is an executor one: `Gather { under }`. It is not a new carrier and
  not a new V4 opcode.
- BIND/BUNDLE separation holds on live code. V4 in practice is `quack::Query`;
  BUNDLE is a choice of lowering function; no new IR is needed.

## OPEN

- `Gather { under }` is not built. The emulation predicts ~5× at 10 % density.
- The ungated gather runs at ~5.2 ns/row over a 128-byte plane; it looks
  unvectorised.
- Neither `Binding` nor `Program` carries a generation.

Reports: `.claude/research/{D-BIND-BUNDLE-0,D-V4-FOLD-LAB,cypher-engine-autopsy}.md`.
