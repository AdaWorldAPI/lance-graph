# 2026-10-14 — Identity table, thought sessions, and Morton-code rows

**Status:** MEASURED. Arms chosen by the pre-registered rules; the overlay
merge is OPEN as the remaining cost.

**Setup (DECISION, operator 2026-10-14):** the identity is a dense table of
64k rows x 32 lanes (lane 0 key, 1 edge, 2..31 value); the 64 thoughts are
session addresses for temporary materialization. Follow-ups measured: one
shared trie instead of one per thought, and the classid out of the trie.
Operator input: the Morton tenant (`lance-graph-contract::morton8x8`) already
has Cartesian addressing, and 64k rows = one 256 x 256 Morton tile, so a
dense class tile needs no trie. Pre-registered in
`.claude/plans/ce64-hypothesis-layers-v1.md` before each arm was written.

**Probe:** `crates/cognitive-shader-driver/examples/identity_session_probe.rs`
(`cargo run --release --example identity_session_probe [dense]`). Built with
`rustc -O -C target-cpu=native`; tier printed `v4 (avx512f)`. 6 tests against
a brute-force oracle; 6 disable runs red.

Reference cell T = 64, d = 10 %, w = 4, all lanes:

| arm | point read | range per NodeGuid | bytes |
|---|---|---|---|
| one trie per thought, 16 levels | 6,725 ns | 1,121 ns | 41.5 MB |
| one trie per thought, 8 levels (classid out) | 3,844 ns | 1,082 ns | 41.5 MB |
| identity trie + overlays per row, 16 levels | 776 ns | 112 ns | 31.4 MB |
| identity trie + overlays per row, 8 levels | 783 ns | 116 ns | 31.4 MB |
| dense tile, identity trie (8 levels) | 1,256 ns | 134 ns | — |
| dense tile, Morton code is the row | 558 ns | 82 ns | — |

**Outcomes:**
- Identity trie with overlays per row: chosen; first arm under the 1 µs point
  read bound.
- Dense class tiles: the row is the Morton HEEL code (no trie); the trie
  stays for sparse classes.
- Classid out of the trie: 1.75 times faster for per-thought tries, no effect
  once the trie is shared (the merge dominates).
- Sessions: 105 to 147 ns per materialized 512-byte row; materialization is
  per prefix range and temporary (all 64 sessions over a class = 2 GiB).
- Few thoughts on a dense tile: 16 to 60 ns per point read.

**OPEN:** at 64 thoughts the merge over up to 64 entries per row costs
500 ns to 1.7 µs; the overlay layout (entry walk, payload locality) is the
next measurement. Timings are one run on a shared 4-core VM.
