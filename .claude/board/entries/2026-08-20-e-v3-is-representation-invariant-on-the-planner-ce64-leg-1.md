## 2026-08-20 — E-V3-IS-REPRESENTATION-INVARIANT-ON-THE-PLANNER-CE64-LEG-1

**Status:** FINDING (Stage 2.6a; `cache/stage26_v3_parity.rs`, artifact
`docs/probes/stage26-v3-planner-parity-discordance.csv`).

The existing compare-thinking work proved V3 thinking-preserving at
`causal-edge`'s own `syllogize` and on `cognitive-shader-driver`'s real emission
path. The **planner's** `CausalEdge64` leg — `cache/nars_engine.rs`:
`SpoHead ↔ CausalEdge64` via `to_causal_edge` / `from_causal_edge`, and
`forward_edge` over the compose tables — was the uncovered third leg. It is now
covered.

**Result: planner V3 representation discordance = 0**, by exact equality across
13 invariants per leg — the rehydrated `CausalEdge64` itself, SPO after
resolution, NARS frequency/confidence, causal mask, inference class, the
`SpoHead` round-trip, the `forward_edge` conclusion, that conclusion's
`SpoHead`, the `syllogize` conclusion edge, and the derived truth/expectation.
Representation-specific fields (V3 Lokal target, TE, payload width) are
deliberately NOT compared — asserting on those would be asserting that V3 *is*
CE64.

**One reasoning implementation.** The V3 arm computes nothing: it drops the
in-edge SPO, resolves it back from the target node's facet, rehydrates a
`CausalEdge64`, and hands it to the *same* `NarsEngine` methods. No V3-native
NARS exists.

**The falsifier is what makes the zero mean anything.** Equivalence is
conditional on `resolved node facet SPO == the original edge's SPO`, so the
harness corrupts ONE facet binding and requires the comparator to go red —
and requires it to stay LOCALISED (fewer than all legs discordant, and the
SPO-shaped invariants specifically the ones that fire). Verified: bypassing
`rehydrate` entirely leaves the primary test green and fails **only** the
falsifier, which is precisely the vacuity the falsifier exists to catch.

**A disable-run corrected one of my own claims.** The degeneracy guard first
asserted "`forward_edge` changed the edge on some leg", documented as proving
the compose tables are not inert. Measured: making every table the identity
left that assertion **green**, because `forward` also composes the NARS truth.
The discriminating form is SPO-specific (`spo_of(fwd) != spo_of(input)`), which
does fail under that disable. The weak form is kept alongside it, with the
measurement written next to both.

**JC's role, and where it stops.** Every quantity at this seam is exact — `u8`
palette indices, `u8` truth bytes, a `u64` register — so **there is no naturally
continuous quantity here for a correlation to characterise, and none was
manufactured**. `jc::stats::binary_association` summarises the syllogism-presence
cross-tab: both categories occur across the sweep, so κ is **defined** and equals
1.0 — a real statement rather than the degenerate constant-column case. Exact
discordance remains the contract.

---

