## 2026-08-23 — E-METACOGNITIVE-TRIANGLE-ARROW-1 — the triangle was write-only: nothing ever READ `StyleLane::Frozen` to choose reasoning, and `promote_family` had zero production callers; both half-arrows now close through the SHIPPED Revision/counterfactual surface

**Status:** FINDING (measured — `PROBE-METACOGNITIVE-TRIANGLE-1`, 12/12 gates
green). **Confidence:** High; reproducible from the commit.

**The audit finding (4 Sonnet survey lanes + direct reads, main @ f5e27c9d).**
The autopoiesis triangle's storage and mechanics are complete and tested
(`StyleLane`, the three `ValueTenant` lanes at 152/164/176,
`MailboxSoA::{set_style_lane, set_style_atom, promote_family}`,
`MailboxSoaView::{style_lane_at, triangle_at}`) — but the CONTROL LOOP did
not exist anywhere: (a) no code read `StyleLane::Frozen` to choose how to
reason; (b) no code consumed a receipt of a reasoning run to decide
keep/explore/promote — the teacher probe computed decisions from fresh grades
and bypassed `promote_family` (zero callers outside its unit tests). This is
`persona-vs-rung-ladder.md` O6 ("triangle structure unbuilt"), now measured.
Premise correction recorded: the CE64 59..60 factual lens is
`CausalTopology::{Direct, IndirectKnownIntermediates,
IndirectUnknownIntermediates, Unknown}` — "IntermediateKnown/Unknown" was
brainstorm prose, not the type.

**The probe** (`crates/cognitive-shader-driver/examples/
probe_metacognitive_triangle.rs`) closes the loop once, falsifier-first, on
the Sudoku corpus: Frozen lane READ → lower rung runs the policy it names →
`RungReceipt` (warranted assignments, fixed point, unresolved count, kernel
side-channel) → higher rung assesses the RECEIPT (its signature carries no
Grid — the meta pass's object is the reasoning, never the puzzle) →
Explore/Learned writes → held-out comparison → the first production-path
`promote_family` call → the NEXT run literally reads the promoted Frozen
lane and solves what previously stalled.

**The Revision hinge is the SHIPPED surface, not a local pseudo-Revision**
(operator correction mid-build: "revision.rs / counterfactual.rs already
exists" — verified: v2 lane real and tested, v3 lane `todo!()` on
D-PERSONA-5). TryExplore is declared a split (frozen commitment = majority
pole, explore hypothesis = minority pole); `deposit_counterfactual` stamps a
`RawEdge` with the −6 mantissa, so the whole Explore arm runs in the
counterfactual lane — never observed truth. Each A-vs-B comparison is a
`FreeEnergyComparison` (residual F = unresolved/81) ruled by
`minority_wins()`; the verdict is a `RevisionOutcome`
(`MajorityHolds` → promotion refused on the base held-out;
`Revised` → `promote_family` on the stall held-out, then the mantissa clears
to 0 per `revise_if_minority_wins`'s documented step-5 protocol). Not
called: the two `todo!()` bodies (`CounterfactualMailbox::*`,
`revise_if_minority_wins` itself) — the probe exercises the decision shape
through the shipped pure pieces and does not fake the actor arm. The
invariant this buys: **exploration may be destructive inside the
counterfactual lane; commitment may not be destructive without warrant.**

**TCP/TCF/CUR as the first metacognitive event, resolved WITHOUT touching
`delta_conf`.** The #995/#997 coarse-signature collision is re-observed live
(side-channel, cloned candidate sets only): where all three kernels see the
same n≥3 set with identical `(fired, sign Δconf))`, the exact
`(len_before, len_after)` transitions already separate TCF (→singleton) from
TCP/CUR (→identity); the meta verdict is `ObserverInsufficient { colliding:
[5,20,26], exact_separates: [20] }` — request a richer receipt, never mutate
recipe semantics for telemetry. TCF's manufactured singleton (n=3 → 1, zero
exclusion warrants) is classified `UnwarrantedCertainty` and refused as
truth; the cell stays unresolved. Every digit actually committed carries ≥1
named exclusion warrant and matches the independent backtracking oracle.

**Gates green (12/12):** F1 keep-frozen on progress; F2 stall → TryExplore;
F4 degenerate explore stays silent (minority does not win → no Learned
write); F5 minority wins → Learned recorded; F6 held-out non-reproduction →
`MajorityHolds`/refuse; F7 held-out reproduction → `Revised`/promote (first
production-path `promote_family`); read-side arrow (next run reads promoted
Frozen, 0 unresolved); counterfactual lane (−6 stamped, 0 cleared on
Revised); F8 `ObserverInsufficient`; F3/F15 unwarranted certainty refused +
warrant-per-commit; F13 bystander row byte-identical; F14 full-loop
determinism. Deferred with scope notes: F9 (reason-context equivalence),
F10/F11 (Revision→Kanban hinge — the `verdict_from → select_tactic`
"designed, not wired" edge is deliberately the NEXT slice); F12 holds by
construction (no `causal_edge` import; CE64 59..63 untouched).

