# Kythe readings produced from rust-analyzer facts (D-SAG-6 codegraph bridge)

**Date 2026-10-31 · Status MEASURED · Branch `ccr-85acaff0-oszq40-kythe`**

The bridge from witnessed semantic facts to Kythe-compatible readings now exists and is automated. `sag-ra-probe XREFS=1` prints every occurrence rust-analyzer's `StaticIndex` records, `tools/kythe_entries.py` turns those rows into Kythe entries, and `tools/kythe_xrefs_check.py` serves them through Kythe v0.0.76 and compares `/xrefs` back. Details: `crates/sag-ra-probe/KYTHE.md`. The earlier server smoke test used hand-written entries; this replaces it.

Result on `fixtures/sag_fixture`:
- 138 occurrence rows, 591 entries, 4 `Attribute` symbols skipped (no kind mapping).
- Kythe `/xrefs` equals rust-analyzer's occurrence set on all 24 symbols.
- 8 of 11 hand-read pins pass. The 3 that fail are references inside `assert_eq!(...)` arguments, which `StaticIndex` does not record. Control: moving the line-21 calls out of the macro makes all three appear; the calls left in `assert_eq!` on line 31 stay missing. Why `descend_into_macros_exact` misses them is open.
- No `ref/call` edges: no call-fact source (Charon) is available, and references are not calls.

OPEN: the macro-argument gap in the fact source; call edges.
