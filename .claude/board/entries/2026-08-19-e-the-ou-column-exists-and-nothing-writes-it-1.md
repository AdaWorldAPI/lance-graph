## 2026-08-19 — E-THE-OU-COLUMN-EXISTS-AND-NOTHING-WRITES-IT-1

**Status:** FINDING (six-agent read-only sweep across lance-graph, ndarray,
lance-graph-java, OGAR, MedCare-rs, the DisMech corpus; operator rulings of
the same day). Deliverable:
`docs/architecture/ARC-B-OWNERSHIP-AND-ADDRESSING-REASSESSMENT.md`.

**The finding:** HHTL is already the FIRST canonical tenant of every
512-byte row (`key(16) | edges(16) | value(480)`, `canonical_node.rs:706-730`,
independently restated at `OGAR/crates/ogar-obo/src/lib.rs:22-35`) — and it
is **zero on every baked row in both production bakes**. `ogar-obo`
(`lib.rs:344-353`, `EDGE-LANES.md:44-51`): *"The bake has no basin:
HEEL/HIP/TWIG and leaf are zero on all 68,797 rows."* MedCare
(`join-map.md:103`): *"HHTL is dormant on every baked row (heel/hip/twig/
leaf all 0 across 68797 rows)."* Every real HHTL reader either mints its own
keys, recomputes tiers in RAM at load, or asserts null — **none reads the
key of a baked artifact.**

So the addressing gap is not storage, layout, or a missing carrier. The OU
column exists in every object and nothing writes or reads it. The measured
consequence: **five structurally distinct hand-rolled ancestor mechanisms in
ONE repository** (LIFO+bitset closure; stack+HashSet with its own cycle
guard; a load-time chosen-minimal-parent chain; a per-query re-climb; and a
Kahn longest-path fallback), each with its own dedup and its own semantics.

**The operationally load-bearing half:** two of those five **disagree by
design** — `atlas.rs:465,898-904` measures **58.2% agreement** between the
rail-register depth and the Kahn longest-path depth, because spanning-tree
depth and DAG longest path are different quantities on a multi-parent DAG.
"Just use the rails" is therefore a **semantic** change, not merely a faster
one, and any migration owes a written ruling on which quantity is canonical.

**The highest-value single change follows directly:** `obo_store::compute_cascade`
(`:690-760`) already derives HEEL/HIP/TWIG in RAM at load — it is the mint,
misplaced. Moving that derivation into the bake populates the key with what
the system already computes, using bytes that are reserved and zeroed today.

Cross-ref: E-TWO-WITNESS-SHAPES-CONTEST-ONE-LANDING-ZONE-1 (the witness seam
and this gap are the same problem — `Locus::BasinAnchor` points at the same
unwritten `part_of:is_a` rail).

