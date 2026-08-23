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
