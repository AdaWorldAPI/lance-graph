# 2026-10-30 — D-MIST-NATIVE-1a: HHTL16 Register128 rows replace the name interner

**Status:** LAB, measured by tests. Plan: `.claude/plans/d-mistaber-native-v1.md` §9.

- The Kripke probe holds no String in its solved state. World, proposition and rule nodes are 512-byte rows in the HHTL16 format: Register0/Register1, 16 Morton8x8 columns, column 0 the identity, columns 1..=15 addresses.
- The classid is per population, on `RegisterLanes`; the row bytes carry none.
- Accessibility is derived from world rows, and a 16th parent is refused.
- `holds ∘ children` goes through `bind_dr64` and `relcomp::Chain`.
- 11 goldens exact; 18 tests and 2 `compile_fail` doctests; 7 disable runs red.
- Known limit: the binding is a typing guard, so swapping `Chain` for raw `compose` turns nothing red. Only the mistyped-signature disable (D4) shows the path is live.

STATUS: measured | OUTCOME: Mistaber nodes as HHTL16 rows, no Strings, typed DR64 binding | OPEN: whether the row key also carries the address; real concept ids from the OGAR codebook.
