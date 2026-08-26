## 2026-08-21 — E-FROM-V1-DROPS-PROVENANCE-AND-THE-COUNCIL-CAUGHT-THE-CONTRACT-ABOUT-TO-TRUST-IT-1 — three BLOCK(P0)s in one 5+3 run, and the sharpest one falsified the spec's own asymmetry claim against the code it cited

**Status:** FINDING (D-ACR-7 council, spec ratified as
`.claude/plans/dacr7-band-reading-contract-v1.md`; every claim verified at
source in Phase 4). **Confidence:** High.

The D-ACR-7 reading contract (which lens wrote CE64 bits 59..63 —
`TrustTexture` vs `CausalTopology` on 59-60, `ReasoningBand` presence on
61-63) went through the full 5+3 council. Three BLOCK(P0)s were raised and all
three survived verification; none was argued away.

**BLOCK 1 (overclaim, the consequential one): "V3's bytes were never temporal"
is false for populated instances.** `CausalEdgeV3::from_v1(e, target)`
(`edge_v3.rs:117`) has **no provenance parameter** and copies truth/spare as a
raw bit copy (`:138-139`). The reassuring comment above it — "under the v1
layout every one of these accessors is a documented zero stub" — is a
**compile-time feature condition, not a runtime provenance guarantee**: a CE64
of v1/unknown provenance (bits 61-63 aliasing `temporal >= 512`) lifted under
a v2 build carries the stale bits into V3 byte 9, and the result is
**indistinguishable from a clean register**. The draft had declared
`EdgeProvenance::V3Register` "always readable" — the exact plausible-wrong-
answer the contract exists to refuse, at its own core. Resolution:
`V3Register` means *"the caller asserts this register was minted clean"*,
never *"V3 registers are clean"*; unstated origin = `Unknown` = refuse; the
`from_v1` provenance drop is filed as a `causal-edge` follow-up, not patched
from the contract side. The v1 trap thus applies to BOTH carriers — on CE64
directly, on V3 **transitively through the lift**.

**BLOCK 2 (overclaim): a gate contradicted the fix that preceded it, and
another gate was a tautology.** After the council split the resolver
(total `ClassView::band_reading` lookup / fallible `BandReading::project`),
G5's carried-forward text ("an undeclared one errors") contradicted the total
half — split into G5a (total: zero-fallback, no error) / G5b (fallible:
`Err(UndeclaredClass)` must fire). And G10a ("project is a pure function of
its arguments") fed the same input to the same function twice — the
vacuous-assertion house pattern, deleted. Only G10b survives: compare
`CausalEdge64::truth()` vs `CausalEdgeV3::truth_raw()` on the same edge
post-`from_v1` — which tests the exact bit-copy site BLOCK 1 indicted, is
hosted in `causal-edge` (both crates' zero-dep postures hold — measured:
BOTH refuse the other as a dependency, `causal-edge/Cargo.toml:20-23`
explicitly), and closes a **measured missing test**: `edge_v3.rs`'s module doc
claims truth/spare survive the round trip byte-exact, and no test asserts it.

**BLOCK 3 (firewall): the consolidation dropped board hygiene entirely.**
Draft v2 contained zero occurrences of any board file — Savant 4 had answered
the hygiene question in full, and Phase-2 consolidation lost the finding. The
loss-prevention phase lost a finding; recorded as such, restored as the spec's
§6′ commitment table.

**Meta-catch worth keeping:** the same reviewer caught the fix ledger claiming
a citation correction it had not applied ("fixed in §2.5" while §2.5 still
read the old span) — the draft's own "measured, not assumed" standard applied
to the draft. And the council's scope itself was corrected mid-flight by the
operator (CE64 = muscle memory, `CausalEdgeV3` = granularity), a gap **no
savant could have caught** because the v1 spec never mentioned V3 and savants
answer only their question sets — the spec-writing phase is the only place
scope errors can be prevented, which is why Phase 0 is "where the real work
happens."

Sibling finding folded in rather than given its own id: `TrustTexture` is a
**×4 homonym with three arities** (4/4/5/3 — `causal-edge/layout.rs:141`,
`contract/mul.rs:82`, `planner/mul/trust.rs:30` with `Dissonant`
unrepresentable in 2 bits, `arigraph/orchestrator.rs:114`), against
`TYPE_DUPLICATION_MAP.md:9`'s stale "×2"; lines 9, 16 and 19 of that doc must
change together.

