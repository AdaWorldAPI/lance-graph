# 2026-10-12 — D-CMH-G1-6: ordered I32 conditions and MIN/MAX as method steps

**Status:** In PR. **Plan:** `.claude/plans/gremlin-composable-algebra-v1.md` §9.

`Cond::Order` (<, <=, >, >= on an I32 field, in lane units) and
`Fold::Min/Max` reach both frontends. R2IL's `MIN`/`MAX` fold calls, defined
in OGAR but refused as unimplemented here, are now wired onto mask-risc's
MaskedMinI32/MaxI32. No new opcode anywhere.

MEASURED: `tests/ordered.rs`, both frontends equal a row loop over the
indicator text at n ∈ {1,64,65,4097}, for every comparison at five
thresholds including an empty selection. Eight disable runs red.

DECISION: an empty `MIN`/`MAX` is `None` on both frontends. R2IL's stack has no
null, so it pushes the fold's identity (i64::MAX / i64::MIN), the same seed
mask-risc's grouped MIN/MAX leave in an empty group; the method layer reads
it as `None`. REVISIT WHEN: R2IL gains a null value.

OPEN: an ordered literal is in the lane's units, not the canonical unit a
converted claim names; a unit-aware literal needs the claim's conversion.
