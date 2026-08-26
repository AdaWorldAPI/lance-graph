## 2026-08-23 — E-TYPE-COMPLEXITY-EXPOSED-A-MEMORY-ABI-ESCAPE-1 — the clippy warning was the surface symptom; `BeliefArena` is an independent AoS cognitive population owner outside the canonical memory ABI

**Status:** FINDING (operator-escalated; #1004 recut to a discovery receipt).
**Confidence:** High for the escape; the restoration is CHARTERED, not done.

**The escalation, in order.** Clippy flagged `snapshot() ->
Vec<((u16,u16),u32,(f32,f32))>` in `probe_parallel_rung.rs`. First reading:
an accidental AoS copy — fixed with a borrowed digest witness. **Both
readings were too small.** The copy was inside an owner that is itself the
escape: `BeliefArena { entries: Vec<Belief> }` with `Belief { stmt, truth,
stamp, rung, premises: Vec<u32>, .. }` (`belief.rs:89,129`) — a second
physical representation of cognition outside the canonical V3 LE substrate
(16-byte `classid + 6×(8:8)` docks, SoA lanes, zero-copy views). Removing
the snapshot removed a copy INSIDE the wrong representation. The hash
witness that replaced it was polish on the violation and is now DELETED;
G4 compares the rung-0 lane against the probe's own authored fixture,
bit-exact, borrowed, no snapshot, no digest.

**Retracted phrasing:** an earlier version of this entry said *"BeliefArena
is not (yet) the canonical 4+12 LE SoA substrate"* — implying an SoA rewrite
is the fix. Wrong frame. The question is whether `BeliefArena` should
physically exist at all: the substrate already expresses relation
(node/edge geometry, SPO), support (`Locus::{SupportedBy,Supports}`),
contradiction (`Locus::Contradiction`), provenance (witness lanes), causal
reading (CE64), attention scope (focus facets). **Only the residue with no
ABI-native home after composing those deserves a new tenant.** Notably, the
sibling `TripletGraph { triplets: Vec<Triplet>, entity_index:
HashMap<String, Vec<usize>> }` (`triplet_graph.rs:86-93`) is the SAME
escape shape — "move belief into AriGraph" as it ships today would move it
between two violations.

**The DOCK/ROUTE separation (operator-sharpened), now the reading of the
compatibility table:**

```
  DOCK ABI    16 B LE = classid(4) + payload(12); content-blind storage
      ↓ classid
  ROUTE ABI   G6D2 / G4D3 / G3D4 / G24N4 / Varnode space→offset→size drill
      ↓
  SEMANTIC VIEW   attention / TEKAMOLO / causal witness / R2IL / …
```

`VarnodeFacet` storing `offset_lo` before `offset_hi` is therefore NOT
malformed — byte-monotone prefix traversal is simply not its route
contract; its typed `prefixes()` drill is. CLASSID CHOOSES THE READING.
THE ROUTE CHOOSES THE TRAVERSAL. THE BYTES NEVER CHANGE SHAPE.

**Evidence tiers, kept separate:** PROVEN — heterogeneous carvings share
one dock without changing it (`ValueTenant::CausalWitness` reads the same
16-byte lane as `G24N4` = 24 signed nibbles). STRONGLY SUPPORTED — V4 as
another tenant of the dock (`VarnodeFacet` is byte-for-byte the envelope,
and independently converged on the `G3D4` carving). NOT YET PROVEN — the
V4 persistence slot/tenant mint (`ruff_r2il` explicitly commits no storage
layout; the classid is PROVISIONAL, mint gated on O5).

**Standing demotions and labels:** **B is removed from the architectural
alphabet** — A (address/mask/traversal) and C (pair-field compute) are
laws; a materialized rotation is an optional derived accelerator that must
EARN existence by measurement, since re-carving the dock is already an
unmaterialized rotation. `RowFocusMask::difference` potholes are a
**conservative candidate-unknown mask (P\*)**, never an exact epistemic
hole — refine before asserting absence. `FlatFact.a/b` are NOT free dock
capacity (per-`FactKind` semantics); an effect facet becomes ANOTHER
ADDRESSED ROW, per the furnace's own rule. The FNV residual grouping hash
is diagnostic bookkeeping only — if it ever becomes an address, key,
identity, token, or route, kill it.

**#1004's honest scope (verdict C):** finding-only plateau. The probe now
carries the escape notice in place; the measured G1..G7 coexistence results
hold FOR THE ARENA'S OBJECT MODEL, and restating them over ABI-resident
state is the restoration charter's job. Falsifiers for that follow-up:
F1 no canonical population as `Vec<RowStruct>` · F2 no nested `Vec` in a
population row · F3 reasoning over immutable ABI views · F4 index
delete/rebuild leaves state intact · F5 premises without per-belief heap
vectors · F6 no representation conversion between graph state and
reasoning · F7 classid+ClassView select interpretation over predefined LE
geometry · F8 same resident bytes consumed by attention/relational/
epistemic/causal contracts · F9 no hash as cognitive representation ·
F10 the population does not move; the view does.

