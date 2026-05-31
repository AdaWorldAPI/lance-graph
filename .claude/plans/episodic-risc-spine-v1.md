# Episodic RISC Spine — v1 (the converged episodic addressing architecture)

> **Authority:** the 2026-05-31 design dialogue (episodic basins+edges, EW64,
> CLAM/pseudo-radix, 4-bit inherited palette, bitmask-as-attention, bounded-horizon
> compression). **Grounds in:** user-supplied `cognitive-risc-{core,classes}.md`,
> `faiss-homology-campq.md`, `wikidata-hhtl-load.md`; the AriGraph paper (arXiv
> 2407.04363); the #444 locality probe (PASS: 98.6% intra-basin, fan-out ≤ 3, Q=0.325).
> **Iron firewalls (non-negotiable):** identity = CAM/OGIT (frozen, exact); stories /
> similarity = discovery (flexible, NEVER addresses). DeepNSM stays English upstream;
> aerial stays a zero-dep proposer. Same triples, two indexes, never swapped.

## The closure — three structures, by lifecycle (zero overlap)

| concern | structure | property | home |
|---|---|---|---|
| **frozen identity** (which-one / what-it-is) | OGIT class palette + CAM hash | exact, **never moves** | `lance-graph-ontology` + `cam` |
| **cross-session index** (the session log) | Lance append-only version log | append + immutable pointer = **pseudo-radix** (no rebalance) | `Dataset::versions()` / `DatasetVersion` |
| **within-session experience** (accumulating working set) | CLAM cluster tree over an ephemeral KV | grows + prunes | `ndarray` CLAM (build target) + `surrealkv` |

Append-ness, clustering, freezing — three jobs, three structures, no overlap. You
don't *build* a radix; append + never-renumber *gives* you one (the Lance version log).

## EW64 = AriGraph episodic edges (NOT a lens over CausalEdge64)

- A mailbox(=episode) is a **basin** with **multiple edges**. The **temporal arc is
  itself a basin+edges one HHTL level up** (`film › drama › episode{e1 e2 e3 e4}`) —
  NOT a scalar prev-pointer (retracts the earlier "prev column").
- **Sparse common case (~98.6% intra-basin, #444):** an edge is addressed BY its
  family — inherited from the row's HHTL/`class_id` path. ~0 extra bits.
- **Cross-family crossover (~1.4%):** a **4-bit nibble** (16 families) indexes the
  **OGIT-class-inherited cross-family palette** (a CAM_PQ facet code; codebook = the
  OWL closed range). The 16 identities are inherited, NEVER stored on the edge.
- **SoC (the user's iron correction):** temporal (basin one level up) ⊥ witness
  (cross-SoA edge) ⊥ frozen identity. **Never stack** — Markov-chained witnesses need
  ≥2 pointers to stack; rejected.

**Encoding — `EpisodicEdges64(u64)` = 4 × `EdgeRef{ family:4b, local:12b }`:**
`family` 0 = intra-basin (the row's own family, inherited — the cheap default);
`1..=15` = cross-family palette index. `local` = 1-based within-family index (12b,
≤4095). 4 edges/word; the within-basin local handles the small basins the probe
measured; the cross-session **episode store is a separate 16-bit Lance-column
pointer** (64k/session), not this word.

## The bitmask is the discriminator AND the attention mask

- The class-inherited **presence bitmask** (`class_view::FieldMask`) doubles as the
  **attention mask** — "attend to what's present" is structural, NOT the forbidden
  per-instance semantics.
- A **`ViewAngle`** (≤16, 4-bit) selects WHICH inherited view-schema attends — the
  Quartettkarte "edition" / FAISS-view. A leaf/family can bake in N required default
  angles. **Line:** an angle is a *class-inherited* attention pattern, never
  instance-private meaning (the moment "angle 3" means something per-row → CISC slide).
- `head2head` (D-H2H-1, shipped) **competes angles**: infight (`DissonanceMin`) vs
  Raumgewinn (`SupportSpread`). Within-story meta suffices first.

## Lifecycle = two-clock

read a book → **16k–256k hot tombstone-witnesses** in the session ephemeral KV
(`surrealkv`) → CLAM clusters them → **epoch-reset prunes** the transient mass
(core #6; tombstone = forwarding record only) → survivors **distill cold into a
palette256 ranking over 4096 story-arc archetypes** (`bgz17`; the DISCOVERY side —
proposes/ranks, **never the CAM key**) → snapshot = a tagged Lance version
("stories from day xz").

## The compression IS the bounded horizon (not a codec)

A research = a free-energy descent; it **rests at the homeostasis floor** ("call it a
day"). Awareness (MUL / `MetaWord` residual-F) = the stopping rule. 256 inputs → <32
clusters (locality, #444 fan-out ≤ 3); 4096–64k epiphanies/KV = **shock-absorber
headroom** (core #7), not a target. **Lever = horizon-shortening (proposer +
Rubicon-arbiter quality), not compression.** Cheapest research = the one that knows
soonest it's done.

## Deliverables

### Verifiable-now (contract, zero-dep, builds offline) — THIS WAVE
- **D-EW64-1** `episodic_edges::{EpisodicEdges64, EdgeRef}` — 4×[4b family|12b local];
  intra/cross; palette-inherited. **(building now)**
- **D-VIEW-1** `view_angle::ViewAngle` — 4-bit view-schema selector + the
  presence-bitmask-as-attention doctrine. **(building now)**
- *(shipped this session)* D-MBX-9-IN `VersionScheduler` = the **pseudo-radix reader**
  (`DatasetVersion` = the fixed session pointer). D-H2H-1 `head2head` = the story-meta.

### CI-gated core (planned; no offline build — `protoc` absent in sandbox)
- **D-EW64-2** wire `EpisodicEdges64` + basin + temporal-arc-basin as `MailboxSoA`
  columns (`cognitive-shader-driver`).
- **D-STORY-1** CLAM-**as-clusterer** over the ephemeral session KV (wire `ndarray`
  CLAM; the current `lance-graph` CLAM is a probe-only *measurer*).
- **D-STORY-2** pseudo-radix session index = Lance version log + tagged snapshots;
  `VersionScheduler` drives it.
- **D-STORY-3** palette256-over-4096 archetype ranking (`bgz17`, standalone-verifiable)
  — discovery, not identity.
- **D-HORIZON-1** MUL residual-F = the snapshot / "call-it-a-day" trigger.

### Resolved-by-decision (self-resolved per the dialogue)
- **4096 archetypes:** frozen-OGIT (CAM) vs discovered-Aerial+ (proposer) → user chose
  **flexible/ephemeral** (CLAM/KV/snapshot) ⇒ **discovery side, never the CAM key**.
- **cross-family palette source:** per-`class_id`, `owl:disjointWith`-derived
  (collision-free / purely additive, per `wikidata-hhtl-load`).
- **temporal:** a basin one HHTL level up, not a scalar column.

## Sequencing
D-EW64-1 + D-VIEW-1 (now, contract) → D-EW64-2 (SoA columns, CI) → D-STORY-1 (CLAM
clusterer) → D-STORY-2 (session index) → D-STORY-3 (archetype ranking) → D-HORIZON-1
(stopping rule). Each CI-gated step verified in a full checkout (with `protoc`).

---
*v1. Contract slices verifiable offline; core/cold slices CI-gated. Firewalls
non-negotiable: identity exact (CAM/OGIT), stories flexible (CLAM/discovery), never
swapped.*
