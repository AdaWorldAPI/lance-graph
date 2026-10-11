# 2026-10-11 — Residue fixed point: only new evidence lowers it (D-RESIDUE-FP-0)

**Status:** MEASURED. Test only, no production change.
`crates/lance-graph-planner/tests/residue_fixed_point.rs`.

```text
cargo test -p lance-graph-planner --test residue_fixed_point
```

Claim: the reasoning loop is cheap (`pearl::hydrate` over the sealed chain),
so it runs to a fixed point; what remains open is residue. Residue here is
hydration debt in bits 59..63 (`IndirectUnknown` edges).

Pinned:
- extra passes over the same evidence move no bit;
- re-sealing evidence already present does not lower the residue;
- unrelated evidence changes nothing; the missing binding lowers the residue
  exactly where it lands;
- the loop moves only bits 59..60, never certification.

Disable runs, both red: promoting on partial evidence (3 of 4 tests fail);
moving certification inside `hydrate` (the bit-isolation test fails).

OPEN: the surprise level proposed for bits 50..52 is not in the layout (still
plasticity); its residue is untested until O14/O4 decide the v3 layout.
