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

## The poster (operator-issued, 2026-08-23)

`docs/architecture/memory-abi-poster.png` — the visual canon of this
charter's law: the content-blind dock (16 B = classid/ClassView + 6×(8:8),
LE, align(16), zero-copy) as the railway; the seven tenants docked around
it; the Allowed list (pointer · offset · ClassView · mask · route contract
· borrowed lens · ephemeral acceleration index); and the Unacceptable
Drift panel with `BeliefArena → Vec<Belief> → Vec<premises> → AoS owner →
snapshot copies → hash-as-geometry → second physical universe` crossed
out. *"Indexes may accelerate the ABI. They must not become the ABI."*

**Reading note (honesty over gloss):** the ring depicts the RESTORED
architecture, i.e. this charter's target — two of its tenants are not
there yet in code. `EpistemicWitnessV3` has no stored tenant today
(epistemic state = read-side `QueryReference` + `Quorum`/`Contradiction`
loci inside the witness lane), and `AriGraph / Relation Geometry` as
shipped (`TripletGraph`) is itself the escape shape this charter names.
The poster is the destination, the homes table is the map, the ladder is
the route.

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
| `stamp: u64` | **DELEGATES to tree accumulation** (operator-ruled, below): a node's evidence accumulates from children + siblings; no per-belief u64 tenant | verify the accumulation reading in the step-3 probe |
| `rung: u32` | **DELEGATES to the node's HHTL address** (operator-ruled, below): derivation depth IS tree depth; inherit from parent | verify depth≡rung equivalence in the step-3 probe |
| `premises: Vec<u32>` | **anaphora pointers** — `Locus::{SupportedBy, Supports}` ±8 window nibbles for near refs; graph edges for far refs; cardinality = MORE ROWS | the nested heap Vec must not survive in any outcome |
| `contradiction` | `Locus::Contradiction` — already shipped, already the store-licensed elevated object | wire, don't reinvent |

**Known trap, recorded:** `TripletGraph { triplets: Vec<Triplet>,
entity_index: HashMap<String, Vec<usize>> }` is NOT an ABI home — same
escape shape. "Move it into AriGraph" as shipped today moves the violation.

## Operator ruling (2026-08-23, arrived ahead of the audit — discharges most of ladder step 2)

**The residue candidates are NOT new tenants. They delegate.**

> The residue delegates to the node's HHTL, and attention has the same HHTL
> masking algebra. It is a simple overlay mask projected as a tree: nodes in
> HHTL **accumulate from children and siblings** and **inherit from parent**.

Groundings, each verified in shipped code:

- **"EpisodicEdge already has residue"** — [CODE] twice over: the
  `EpisodicEdge` trait carries a signed 4-bit mantissa nibble
  (`set_inference_mantissa`, the −6 counterfactual deposit —
  `counterfactual.rs:178-183`), and `EdgeCodecFlavor::CoarseResidue` is
  *"coarse index + a per-dimension signed-4-bit residue"*
  (`canonical_node.rs:676-678`). Signed-i4 residue is established practice,
  not a new invention.
- **"Relativpronomen anaphora pointers"** — [CODE] literally:
  `Locus::Antecedent = 7` is documented *"relativPronomen → its antecedent"*
  (`causal_witness.rs:131-132`), and `resolves_to(locus, self_pos,
  stream_len)` (`:334-344`) dereferences the signed nibble into a window
  position — pronoun resolution as a bounds-checked pointer walk. The
  anaphora election mask has its own can-fire/stay-silent tests.
- **"TEKAMOLO already practices 24×i4"** — [CODE]: the `G24N4` witness lane;
  Tekamolo loci 0..3 (Temporal/Kausal/Modal/Lokal) live inside the same
  24-nibble register.
- **Accumulate/inherit precedents** — [CODE], scattered but real:
  `carried_awareness` *"carry accumulates lower→higher"*
  (`recipe_loci.rs:347`); `rail_geometry.rs:183` Horner accumulation
  `Σ slots[i]/256^(i+1)`; `causal_audit` append-only *"evidence
  accumulates"*; `orchestration_mode.rs:8` *"NARS truth values accumulate on
  the path, not individual nodes"*; `FieldMask::inherit`; the
  `legacy_outliers` bucket-rollover doctrine. The unified tree-overlay
  reading (one mask, projected as a tree, accumulate-up/inherit-down) is
  [OPERATOR RULING] composed over those shipped pieces — the step-3 probe is
  what promotes it to [CODE].

**Consequence for the ladder:** step 1's audit VERIFIES the delegation
(rather than hunting for homes); step 2 is discharged for `stamp`/`rung`
(ruling: delegate, don't mint) and remains open only for anything the audit
finds that the delegation cannot carry; step 3's probe must additionally
show: (a) depth-as-rung over a real derivation tree, (b) evidence
accumulation from children/siblings matching the arena's stamp-union
semantics, (c) the SAME prefix-cover algebra serving both the attention mask
and the belief overlay — one algebra, two tenants, zero conversion.

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
