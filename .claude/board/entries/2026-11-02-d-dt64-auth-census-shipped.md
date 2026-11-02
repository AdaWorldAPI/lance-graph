# D-DT64-AUTH: the method-authority census ships with a two-way ratchet (2026-11-02)

**Status:** SHIPPED (lab-grade guard on the current tree). Plan [`d-dt64-method-authority-census-v1.md`](../plans/d-dt64-method-authority-census-v1.md).

## What landed
- Table `crates/lance-graph-r2il/dt64-authority-census.tsv` (exactly 10 tab-separated columns per row):
  - `LENS` rows pin each production behaviour lens's DT64 declaration: `POLYFILL_LENS` (`0xFFFF::0xF000`, rev 1, 17 arities) and `XLOG_LENS` (`0xFFFF::0xF100`, rev 1, 3 arities).
  - `LAB` rows list the six method identities and dispatch paths DT64 does not see: `MethodSignature`, the tract-harvest `Registry`, XLOG `Word`, the `FnIndex` opcode, `FoldDialect`, and `r2il::procs`. Each has a why and a removal path.
- `lance-graph-r2il/tests/integration/dt64_census.rs`, six tests:
  - it sweeps every `crates/*/src/**/*.rs` for `compose_classid(HARDCODED…` lens mints and matches them against the `LENS` rows in both directions;
  - it replays the polyfill declaration against its row in a fresh `BindStore`;
  - every `LAB` token must still occur (C2);
  - a can-fire test (C1) and a silence test (C3).
- `lance-graph-xlog/tests/dt64_census.rs` checks the XLOG row through `lens::declare`. It lives in XLOG because `lance-graph-r2il` does not depend on `lance-graph-xlog`, so no new dependency was added.

## Measured
- r2il census 6/6 and xlog census 1/1 pass; clippy `-D warnings` and fmt are clean.
- **Disable runs** (the census was committed first; every anchor was asserted):
  - **D1:** a new lens mint in `polyfill.rs` turns 3 tests red, including the two-way sweep.
  - **D2:** a stale `LAB` token turns `every_lab_exception_still_deviates` and the silence test red.
  - **D3:** appending a word to `polyfill_arities()` without updating the row turns the declaration replay red.
  - **D4:** the sweep comparison switched off turns C1 red.

## Findings
- R2IL's own `src/` never declares the polyfill lens. Only tests and consumers call `declare_algebra(POLYFILL_LENS, …)`; the executor (`binding::prepare`) admits against that declaration. The row therefore pins `polyfill_arities()`, the one declaration source.
- `r2il::procs` is reached from production `src/` (`lance-graph-nal/src/nal.rs` `procs::count`) without admission. It is listed as `LAB`, to be removed once OGAR has `PROC_DEF`/`PROC_CALL`.

## Limits
- The sweep sees only lens mints spelled `compose_classid(HARDCODED…`. A behaviour lens minted as a raw literal is not seen.
- The census pins declarations and lists exceptions. It migrates nothing and executes nothing through DT64.
- The `INVOKES_WORD` association stays with D-CODEGRAPH-0-BIN §8.3.
