# D-GLOBAL-WHERE-P2: criteria-aware WHERE, DR composition, Lance selective read, three parity verdicts

**Status:** MEASURED, lab only. No production change.
**Probe:** `.claude/probes/2026-11-03-d-global-where-p1/`. Read `REPORT.md` for numbers and verdicts, `ENGINEERING-PARITY.md` for the literature matrix, and `LANCE-TRACE.md` for the source-backed caller path. The Lance arm lives in `crates/crepe-plan-probe/examples/where_lance_arm.rs`.

**Semantic parity: reached.**
- Every count and EXISTS answer equals both the oracle and DuckDB 2.0.0, at 1M and 10M, threads 1 and 4.
- Every Lance arm (full, allow/block prefilter, attested range read) equals the oracle and the full-projection oracle at 1M and 10M.
- ClassView → physical column resolution goes through `predicate_iri`. It refuses AbsentField, Unauthorized, UnknownProperty, AmbiguousProperty and StaleSchema, each paired with a success control.
- Two traps prove a shortcut would be caught: an ordinal shortcut is wrong on 16/16 positions, and a label shortcut finds no column on 16/16.

**Performance parity: split.**
- Count beats DuckDB by 14.2× (1M, 1 thread) down to 1.96× (10M, 4 threads), weighted 34/46/14/6.
- EXISTS loses at 10M, 0.43× and 0.16×. `Terminal::Any` has no per-tile early exit (`exec.rs:1793`).
- Lance at 1M: a clustered selection reads 0.1–9% of the full scan's bytes.
- A scattered 0.1% selection already reads 33% of the bytes at k=1 and 99.8% at k=16. At 10–50% the masked read is 1.4–11× slower than a full scan, and `ReadPolicy::DEFAULT` already falls back to a full read there.
- At 10M a clustered 0.1% read is 112× faster than the full scan.
- `ReadPolicy::DEFAULT` decides on covered rows alone, so it also sends a single contiguous 10% range to a full scan: 3.8× slower at 1M, 4.6× at 10M. Range count, not coverage, separates clustered from scattered demand.

**Engineering parity: reached on representation, not on strategy.**
- Missing wiring: EXISTS early exit, and routing once instead of per query (break-even at Q ≥ 2; 158,198 → 1,012 µs per query at Q=256).
- Missing capability: choosing evaluation direction, and an ordered u8-in-u32 compare word. Composing existing words costs one pass per 2-bit prefix, up to 5× slower than a hand loop.

**Costs measured, never assumed free.**
- RowAddrMask ingress is a copy, and scales with ranges, not rows: 0.4–5.4 µs for 1 range; about 4,600 µs and 126 KB for 91,527 scattered ranges; 123,162 µs for 3.26M ranges. One tree is built per read, and its size is measured outside the timer. The first run built two trees per C read and doubled those figures (Codex finding on #1950); the corrected numbers are the ones recorded.
- Lane routing costs about 70–90 ns per id.

**Boundary:** `BLOCKED_D4`. No witness binds a DR Frame to a Lance version. Arm C takes its ids as given and composes no DR ordinal with `_rowid`. No third coordinate map was built.

**Found:** Lance 13 truncates `max_rows_per_file` with `as u32` (`write.rs:1077`). `1 << 40` became 0 and produced one fragment per batch. #1633's NotContiguous guard caught it at 10M.

**OPEN:**
- S-INGRESS / D4: one version-checked DR → `_rowid` authority (owner decision).
- EXISTS early break (mask-risc owner).
- The ordered u8 compare kernel (ndarray / mask-risc owner).
- Why the borrowed lane read is 2.3–2.5× slower at 50% selectivity.
- `plan_read` should weigh the range count (lab #1633 owner).
- Every Lance byte count was served from the page cache, not disk.
