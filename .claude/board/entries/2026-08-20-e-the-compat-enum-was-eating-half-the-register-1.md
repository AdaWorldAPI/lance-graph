## 2026-08-20 — E-THE-COMPAT-ENUM-WAS-EATING-HALF-THE-REGISTER-1

**Status:** FINDING (measured + fixed, PR #971). `CausalEdgeV3::rehydrate`
routed the stored 4-bit signed inference mantissa through
`InferenceType::from_mantissa(...) → pack(...) → to_mantissa()`. That enum is a
**lossy compatibility projection** — 8 mantissa states onto 16 — so the round
trip silently rewrote **8 of the 16 states**:

| m | old rehydrate gave | m | old rehydrate gave |
|---|---|---|---|
| −8 | +1 | −3 | −1 |
| −7 | +7 | −2 | −1 |
| −5 | +5 | **0** | **+1** |
| −4 | +4 | +3 | +5 |

`0 → +1` is the one that matters most: **every `pack_v2` edge defaults to
mantissa 0**, so the neutral/identity state was being rewritten to Deduction on
any lift-and-rehydrate. Fixed by carrying the RAW nibble —
`set_inference_mantissa(m)` after `pack`, with the `InferenceType` argument
demoted to a throwaway placeholder.

**Why it hid, and the reason is the generalisable half.** *Exactly half* the
states survive the projection (−6, −1, +1, +2, +4, +5, +6, +7). A projection
that failed on everything would have been caught by the first test written; one
that is right half the time looks like a working codec until someone sweeps it.
And the sweep that would have caught it was structurally impossible where the
type is actually exercised: the Stage-2.6a planner parity harness only ever
carries `InferenceType::Deduction` (mantissa +1) — **a surviving state** — so a
harness that is correct, green, and genuinely load-bearing was blind to this by
construction. *A parity harness proves the two legs agree; it says nothing about
whether the conversion under them is total.* Conversion correctness needs its
own low-level exhaustive suite, and now has one
(`causal-edge::edge_v3::tests`, 11 tests, 5 disable-runs verified red).

**Same pass, the other half of the leak: three CE64-v2 fields were being
dropped entirely** — `w_slot` (6 bits), the truth/topology register (2), the
spare/`ReasoningBand` register (3) — all of which became meaningful state with
#970. They now land in V3's dormant reserved bytes (`[8]` = `w_slot(6) |
truth(2)<<6`, `[9]` = `spare(3)`), preserved as **RAW ORDINALS**: copying
ordinal `01` across means "ordinal 01 preserved", never "`IndirectKnown` is now
source-authoritative". Under v2 the 64 bits are fully partitioned, so
`CausalEdge64 → V3 → CausalEdge64` with the same resupplied SPO is now
**bit-identical** — asserted as whole-register equality, which is what catches a
field a future session forgets to enumerate.

**The two exclusions are principled, not residue.** (1) The 24-bit in-edge SPO
— intentionally deduplicated into the target node's CAM-PQ facet. (2) The
deprecated v2 `temporal` — *not valid CE64-v2 state at all* (bits 52..63 are the
reclaim zone), so it is NOT mapped into V3's TE; TE stays an independent
producer-set signed chain offset.

**A method note worth keeping.** One of the five disable-runs used a malformed
`sed` pattern that matched nothing, and the resulting green read exactly like
"this carry is not load-bearing." It was re-run with an exact-string edit that
*asserts the anchor exists* before removing it, and went red immediately. This
is `E-ANTI-EIGENVALUE…`'s twin at the tooling layer: **a disable that does not
disable is indistinguishable from a guard that does not guard**, and only the
anchor assertion tells them apart. Cf. tesseract-rs's "turning a knob that does
not bind is not a disable."

---

