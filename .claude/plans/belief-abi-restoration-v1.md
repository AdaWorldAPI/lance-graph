# BELIEF-ABI-RESTORATION-1 — which Belief semantics still have no home?

> Status: CHARTER (audit-first — no layout may be invented before the residue
> audit reports). Born from #1004's finding
> (`E-TYPE-COMPLEXITY-EXPOSED-A-MEMORY-ABI-ESCAPE-1`): `BeliefArena` is an
> independent AoS cognitive population owner outside the canonical memory ABI,
> and `TripletGraph` is the same escape shape. Deleting the arena entirely is
> a valid outcome of this charter.

## The law this charter restores

```
  MEMORY ABI
         16-byte content-blind dock (classid(4) + payload(12), LE)
                    │ classid
           route/carving contract (G6D2 / G4D3 / G3D4 / G24N4 / typed drill)
                    │
          ┌─────────┼─────────┐
        mask     traverse    pair
          └─────────┼─────────┘
                operator
                    │
           resident bytes stay
```

CLASSID CHOOSES THE READING. THE ROUTE CHOOSES THE TRAVERSAL.
THE BYTES NEVER CHANGE SHAPE.

There is ONE physical population authority. Everything else is a view or an
operator. Indexes are derived machinery: delete/rebuild must leave cognitive
state intact. Hashes verify bytes; they never stand in for semantic geometry.

## The question (brutal form)

NOT "how do we make BeliefArena SoA?" but:

> Which semantics of `Belief` still have NO ABI-native home after AriGraph
> relation geometry + node support + epistemic/episodic witnesses + V3
> tenants are composed? **Only the residue deserves a new tenant.**

## The homes audit (grounded starting map — VERIFY each row, then fill)

| Belief field | candidate ABI home [grounded] | status to establish |
|---|---|---|
| `stmt (s:u16, cop, p:u16)` | canonical node/edge geometry + SPO store; the `part_of:is_a` rails (le-contract L1–L3) | is the copula expressible in the edge/rail reading? |
| `truth (f32, f32)` | `spo::truth` NARS truth (revision shipped, `spo/truth.rs`) | where does per-relation truth RESIDE (lane? SPO row?) |
| `stamp: u64` | evidential bitset — nearest: witness corpus / merkle (`spo/merkle.rs`) | likely residue candidate; do not invent yet |
| `rung: u32` | derivation depth — `Locus::MeaningLevel` is a context POINTER not a magnitude | likely residue candidate; check attention/rung readings first |
| `premises: Vec<u32>` | `Locus::{SupportedBy, Supports}` context pointers + graph edges; cardinality = MORE ROWS (the furnace rule) | the nested heap Vec must not survive in any outcome |
| `contradiction` | `Locus::Contradiction` — already shipped, already the store-licensed elevated object | wire, don't reinvent |

**Known trap, recorded:** `TripletGraph { triplets: Vec<Triplet>,
entity_index: HashMap<String, Vec<usize>> }` is NOT an ABI home — same
escape shape. "Move it into AriGraph" as shipped today moves the violation.

## Step 1 — Delegation-Verification Audit (2026-08-23)

Grades: **[CODE]** shipped and load-bearing · **[STRUCTURAL FIT]** a shipped
type could carry this, unwired · **[ABSENT]** no implementing code exists ·
**[OPERATOR RULING]** the direction is decided, the mechanism is not yet code.

### `premises: Vec<u32>` — delegation STRENGTHENED, not just confirmed

**[CODE].** Every real construction site in the tree was checked —
`close_transitive`'s `derived: HashMap<CStmt, (TruthValue, u32, [u32; 2])>`
(`belief.rs:296,319`, fixed 2-array) and `tactics.rs`'s
`Candidate.premises: [u32; 2]`, documented *"the pointer fabric"*
(`tactics.rs:76-77`, four mint sites, all exactly 2). Every
`admit_derived(..)` call in the workspace (11 sites, `epiphany.rs`,
`elevation.rs`, `stance.rs`, `insights.rs`, `belief.rs` tests) passes `&[]`,
`&[inner_id]` (1), or a 2-array. **Nothing in the codebase has ever
constructed a `Belief` with 3+ premises.** The `Vec<u32>` signature is more
general than any real caller needs.

**What this establishes, and what it does NOT.** Established: real premise
CARDINALITY ≤ 2, so this is not the "cardinality = more rows" case the
furnace rule guards against. **Not established:** that two `u32` premise
identities FIT in two 8:8 tiles, and emphatically not that they fit in two
signed `i4` context nibbles. Cardinality and physical width are different
facts, and an earlier revision of this section conflated them. A `u32`
arena index is not an address; whether the identity it stands for can be
expressed in a tile or a nibble depends on a locality/address
transformation that has not been designed, let alone proven. The width
question is OPEN and belongs to step 2.

### `stmt`/`truth` homes — one structural fit found, one open

**[STRUCTURAL FIT].** `spo::truth::TruthValue { frequency: f32, confidence:
f32 }` (`truth.rs:15-17`) is byte-identical in shape to
`nars::truth::TruthValue` and is documented *"Each SPO edge carries a
TruthValue"* — a real, shipped per-edge truth residence exists. **Open:**
no code currently writes a `Belief`'s truth into an SPO edge; this is an
unwired but structurally sound target, not yet a home.

**Open, unresolved.** Whether `Copula::{Inh, Sim, Impl, Rel(u16)}` is
expressible in existing edge/rail geometry was not settled this pass —
`Copula` needs its own small audit before step 2 closes it.

### The tree-overlay delegation (rung=depth, stamp=accumulation) — direction ruled, mechanism ABSENT

This is the harder, more important finding, and it changes what step 3 is.

**[ABSENT] — no `Belief` is ever minted an HHTL/`FacetCascade` address.**
Grepped `FacetCascade`/`facet_classid` across every file in
`nars/`: **zero occurrences.** `BeliefArena` indexes entries by plain `u32`
position in a `Vec` — there is no address, so "rung = tree depth" has
nothing to measure depth OF yet. The delegation names a destination; the
bridge from a `Belief` to an addressed node does not exist in any form.

**[ABSENT] — no accumulate-from-children-and-siblings fold exists.**
Every precedent the ruling cited was re-checked individually and each is a
**different** mechanism, not one shared fold: `carried_awareness`
(`recipe_loci.rs:347`) is a lower→higher CARRY, not a children→parent
accumulation; `rail_geometry.rs:183`'s Horner sum is a fixed-depth
positional weighting, not a tree walk; `causal_audit`'s *"evidence
accumulates"* is append-only history on ONE node, not aggregation ACROSS
nodes; `orchestration_mode.rs:8`'s *"truth accumulates on the path"* is the
closest in spirit but is documented, not implemented, at that line;
`FieldMask::inherit` is a bitwise OR of two masks, not a tree fold. **None
of these compose into "read a node's children + siblings, fold their
evidence, inherit from parent."** Grepped `children.*sibling` and
`fn accumulate` in `lance-graph-contract/src/` directly: zero hits.

**What this means for the ladder.** The audit's output is the two [ABSENT]
verdicts above. It does **not** license a mechanism, and this section
deliberately proposes none.

**A withdrawn proposal, recorded so it is not re-derived.** An earlier
revision of this section instructed step 3 to *"mint a `FacetCascade`
address per belief position (even a trivial per-arena-position one)"* and
then build a fold until it reproduced the arena. **That is backwards and is
withdrawn.** A synthetic address derived from arena POSITION would turn a
`Vec` index into a pretty 16-byte `Vec` index — still a second physical
belief universe, and now one wearing the canonical address format as
camouflage. An address must come from the canonical node/relation identity
or it is not an address; inventing one to make a hypothesis testable
invents the result too.

**The two ruled items are therefore hypotheses awaiting step 2, not
directions awaiting implementation:**

- `rung = HHTL tree depth` — the real open question is whether derivation
  depth is reconstructible from SUPPORT TOPOLOGY (the premise DAG) at all,
  which is answerable without any address. (Measured since, on one fixture:
  `PROBE-TARSKI-SIGNED-WITNESS-1` gate A2, PR #1007 — depth derived from
  the premise DAG alone reproduced the arena's stored `rung` 10/10. One
  fixture is not a general result, and it is evidence about SUPPORT
  topology, not about HHTL depth.)
- `stamp = children/sibling accumulation` — any replacement must reproduce
  `Stamp`'s load-bearing IDENTITY semantics: disjointness detection,
  overlap detection, source-set union, and no-double-count
  (`belief.rs:39-48`). A generic commutative fold is not automatically
  equivalent to a source-set union, and must not be assumed to be.

## Bounds (the line not to cross)

- Do NOT optimize BeliefArena. Do NOT mechanically SoA-split its fields.
- Do NOT create BeliefArenaV2 or five Vec columns.
- Do NOT preserve nested premise vectors in any outcome.
- FlatFact.a/b are NOT free capacity (per-FactKind semantics) — an effect or
  premise facet is ANOTHER ADDRESSED ROW.
- B stays out of the alphabet: any materialized rotation must EARN existence
  by measurement; re-carving the dock is already an unmaterialized rotation.
- Scoped-difference potholes are conservative candidate-unknown masks (P*),
  never exact holes — refinement before any absence claim.
- The V4 persistence slot is NOT YET PROVEN (provisional classid, O5 gate):
  nothing here may canonize on it.

## Falsifiers (all ten must hold in the restored state)

- F1  no canonical cognitive population owned as `Vec<RowStruct>`
- F2  no canonical population row owns nested `Vec` state
- F3  reasoning operates over immutable ABI-resident views
- F4  deleting/rebuilding ephemeral indexes does not alter cognitive state
- F5  premise/support structure survives without per-belief heap vectors
- F6  no population representation conversion between graph state and reasoning
- F7  a ClassView/classid determines interpretation over predefined LE geometry
- F8  the same resident bytes serve attention, relational reasoning, epistemic
      witnessing and causal reading where their contracts overlap
- F9  no hash is used as cognitive representation
- F10 the population does not move; the view does

## Deliverable ladder (each gates the next)

1. **Residue audit** — fill the homes table with file:line evidence; output:
   the exact residue list (possibly empty).
2. **Operator ruling** on the residue: existing-tenant composition vs one new
   tenant mint (per residue item, not wholesale).
3. **One bounded probe** restating #1000's G1..G7 coexistence results over
   ABI-resident state — the arena's results are the parity oracle.
4. Only then: any migration of `close_transitive` / tactics onto views.

A future `BeliefArena`, if the name survives at all, is at most
`BeliefArena<'a> { borrowed ABI views + derived ephemeral indexes }`.
