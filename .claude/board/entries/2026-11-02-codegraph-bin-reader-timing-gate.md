# D-CODEGRAPH-0-BIN reader: unchanged refresh 2.58 µs; changed-crate intake 2.5 ms fails the 1 ms gate (2026-11-02)

**Status:** MEASURED (release, `crates/lance-graph-codegraph/examples/bin_timing.rs`, mask-risc store of 17,155 rows written by `ruff_charon import`, N = 41 fixed in advance, 4 CPUs, load ≈ 0.3). Plan [`d-codegraph-0-binary-lanes-v1.md`](../plans/d-codegraph-0-binary-lanes-v1.md) §5.

| measurement | median | gate |
|---|---|---|
| unchanged `refresh_crate` (`rows.bin`, 64-byte header read) | 2.58 µs (range 2.53–14.92) | ≤ 20 µs: met |
| changed-crate intake from `rows.bin` | 2.54–2.84 ms | ≤ 1 ms: **NOT met** |
| — `fs::read` / `CrateFile::parse` / `read_bin_as` | 0.16 / 0.35 / 2.04 ms | |
| same crate from `rows.tsv` (baseline) | 5.08–7.41 ms | |

- The file is 556 KB as `rows.bin` against 3.05 MB as `rows.tsv`.
- **Where the gate fails:** `read_bin_as` spends about 2 ms in `Symbols::intern` (a hash plus an allocation per new name). This is the interner replacement the plan already assigns to chunk B3: `SYM_HASH` → global id, label locators into retained buffers, no intern at open or refresh, falsifiers F4 and F5.
- Verification (`parse`) is 0.35 ms. After B3 it is the next item to look at.
- Chunks B1 and B2 are in lance-graph PR #1929, with the gate reported as open. Merging them before B3 is the operator's decision.

OPEN: B3 (interner) and the re-run of this gate; Quack counting (F7) behind the codec9-column gap.
