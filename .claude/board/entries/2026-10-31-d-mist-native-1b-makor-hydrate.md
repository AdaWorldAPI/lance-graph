# 2026-10-31 — D-MIST-NATIVE-1b: Mistaber makor addressed the KJV way, no classid

**Status:** LAB, measured by tests. Plan: `.claude/plans/d-mistaber-native-v1.md` §10. DECISION (operator): no classid; reuse `deepnsm-v2` `hydrate.rs`.

- 141 real `makor` citations from the compiled Mistaber ontology are laid over a Yoreh De'ah outline. Siman is the chapter, seif the verse.
- Each lands on its seif's `NiblePath`. No classid, `promote`, key or `NodeRow` is involved.
- A siman past the 255-per-level capacity is refused into `unaddressed`.
- 8 tests; disable runs M1, M3 and M4 red. M2 (seif dedup) is recorded as representation, not addressing.
- `deepnsm-v2` is a lab crate, not a global concept. The global piece is the contract's `NiblePath`.

STATUS: measured | OUTCOME: Mistaber citations hydrate as NiblePaths with no classid | OPEN: whether the generic outline→tree→addressed-items step moves beside `NiblePath` in the contract.
