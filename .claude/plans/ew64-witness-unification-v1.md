# EW64 × CausalWitness — the witness demarcation and the EpisodicWitness64 landing zone — v1

> **⊘ RE-GRADED 2026-08-19 (operator ruling E) — read
> `docs/architecture/ARC-B-OWNERSHIP-AND-ADDRESSING-REASSESSMENT.md` §4 first.**
> The future architecture is **not** "make EW64/CE64 better, mint another
> packed tenant, wire more semantics into the bit layouts" — that is the
> shape being grown OUT OF. Consequences, all applied below:
>
> - **New F0** gates every tenant proposal behind the ruling-E question.
> - **D-EWU-3's proposed `ValueTenant` is NOT approved by this plan** and is
>   provisionally argued *against* in the reassessment, on this plan's own
>   §2 evidence: EW64's slots are **anonymous, index = recency**, and a
>   recency ordering is a cache property, not canonical information.
> - **The real prerequisite is the ontology gap itself.** The `part_of:is_a`
>   rail that `Locus::BasinAnchor` already points at is *confirmed
>   unwritten* — and HHTL is measured **zero on every baked row** in both
>   production bakes (`ogar-obo/src/lib.rs:344-353`; MedCare
>   `join-map.md:103`). The witness seam and the addressing gap are one
>   problem wearing two hats.
> - **Unaffected and still standing:** the demarcation (D-EWU-1), the
>   THREE→TWO spec correction (D-EWU-2), the bridge location (D-EWU-4), the
>   basin-promotion seam (D-EWU-8).

> **Status:** DESIGN / SEAM-CUT (2026-08-19). **Zero code changes.** Nothing here
> mints a tenant, moves a byte, or writes a lane. The deliverable this plan
> exists to unblock is a *ruling*, not a landing.
> **Charter position:** ARC D (episodic / epistemic spine) of
> `docs/architecture/DUMB-STORAGE-RESET-CHARTER.md` (:820-827). It **gates**
> ARC E (meta-awareness, :828-835): every ARC-E mechanism named there —
> second-order awareness stacking, epistemic holes, top-down observation —
> needs an addressable witness, and there are today **two candidate witness
> shapes that do not reference each other**.
> **Problem statement:** `EpisodicWitness64` is **not a code symbol**
> (`soa_view.rs:272`, verbatim: *"`EpisodicWitness64` is NOT YET a code
> symbol"*). Its migration to the V3 substrate has no plan because its landing
> zone is contested. This plan cuts that seam.
> **Prior art (read, not superseded):** `.claude/specs/episodic-witness64-ce64-prefetch.md`
> (the A–E phase spec); `.claude/plans/episodic-risc-spine-v1.md` (2026-05-31,
> predates the charter and the tenant-14 mint — it does not address the
> collision below); `docs/architecture/ARC-A2-STRONG-HIERARCHY-RECONCILIATION.md`
> (the five-disconnected-islands finding).
> **Board hygiene OWED, not done here** (this session wrote exactly one file):
> `INTEGRATION_PLANS.md` prepend + `STATUS_BOARD.md` rows D-EWU-1..9 +
> `ISSUES.md` entry for the collision, in the commit that lands this plan.

---

## §0 FROZEN DECISIONS (charter constraints — not renegotiable by this plan)

| # | Constraint | Source |
|---|---|---|
| F0 ⊘ | **CE64 and EW64 are CODECS / PROJECTIONS — never the canonical semantic container** (operator ruling E, 2026-08-19). They may stay extremely efficient packed hot reads. Canonical meaning must be recoverable from V3-shaped rows + HHTL locality + masks + provenance + relations + the existing operators; CE64/EW64 then *project* that truth. **Before any new tenant is proposed, D-EWU-3 must answer in writing: is this genuinely missing canonical information, or a container minted to avoid completing the address/mask transition?** See `docs/architecture/ARC-B-OWNERSHIP-AND-ADDRESSING-REASSESSMENT.md` §4 — which answers it *provisionally against* the tenant, on the plan's own §2 evidence: EW64's slots are **anonymous, index = recency**, and recency is a cache ordering, not a canonical fact. | operator 2026-08-19; reassessment §4 |
| F1 | Storage knows **references, hierarchy, ClassView, WideFieldMask, version, temporal** — and MUST NOT know ontology semantics, causal meaning, Pearl rung meaning, known-unknown semantics, NARS, AriGraph semantics, or "awareness" | charter :52-76, :899 |
| F2 | Parent knowledge is **REFERENCED, never copied**; inheritance = follow sparse parent reference + local delta | charter :270-280, :912 |
| F3 | **No promotion DTOs, no HierarchyPlane types, no separate higher-order structs, no generic "structural algebra", no another SoA.** Higher-order / ontology / episodic / meta-awareness are **READINGS** inside the one canonical substrate (32×(4+12), 6×2×8-bit, 256:256) | operator correction quoted at ARC-A2 :241-256 |
| F4 | `256:256` stays **polymorphic**; the registered readings explicitly include **`episodic-witness basin : role`** | charter :202-219 (:211) |
| F5 | Known unknowns are **epistemic topology, not null payload**; storage stores reference topology / masks / version only | charter :303-310, :531, :914 |
| F6 | Episodic witness and epistemic causality **must not be collapsed**; `CausalWitnessFacet` "remains pointer-like/contextual evidence" | charter :505-531 |
| F7 | No freeze / no batch wall / DatasetVersion is never permission to think | charter :369-397, :919-920 |
| F8 | Sub-byte granularity's sanctioned home is a **LANE** (`ValueTenant` variant + `VALUE_TENANTS` descriptor — *"the two places"*), never a `le-contract` §3 payload layout | `causal_witness.rs:22-26` |
| F9 | `I-LEGACY-API-FEATURE-GATED` — any layout reclaim needs a field-isolation matrix + a version gate | `CLAUDE.md`; sprint-11, 5 caught instances |

---

## §1 INPUT INVENTORY (every line read this session)

### 1a — EW64: the shipped hot tier (Phase A)

| Symbol | Where | Shape |
|---|---|---|
| `EdgeRef { family: u8, local: u16 }` | `crates/lance-graph-contract/src/episodic_edges.rs:34-41` | `family==0` intra-basin (~98.6%, #444); `1..=15` cross-family nibble; `local` 1-based `1..=4095` |
| `EpisodicEdges64(pub u64)` | `episodic_edges.rs:103` | 4 × 16-bit slots; `CAPACITY = 4` (:107); slot pack = `family<<12 | local` (:81-83) |
| `promote` / `strongest` / `coldest` / `contains` | `:168` / `:217` / `:227` / `:242` | MRU: fire→slot 0, survivors shift, full+fresh evicts coldest (returned). **Slot order IS strength — recency only, no stored weight** |
| `DemotionSink` + `promote_into` | `:311` / `:206` | the hot→cold exit seam |
| the only `impl DemotionSink` | `:631` — inside `#[cfg(test)]` (`VecSink`) | **no production implementor exists workspace-wide** (grep) |

**EW64 has no `ValueTenant` variant.** `ValueTenant` is `Meta = 0 … CausalWitness = 14`
(`canonical_node.rs:828`, `:908`) — 15 variants, none episodic-edge. Nine sites
name `EpisodicEdges64` in doc-comments (`soa_envelope.rs:7`,
`causal-edge/src/syllogism.rs:47`, `deepnsm/src/comprehension.rs:21`,
`graph/arigraph/community.rs:13`, `planner/src/lib.rs:130`, `contract/src/plan.rs:42`,
`arm-discovery/examples/meta_awareness_probe.rs:209`, …). **Every one is prose.
Nothing reads or writes an EW64 column, because there is none.**

### 1b — tenant 14 `CausalWitness`: the OTHER witness shape

| Fact | Where |
|---|---|
| Tenant 14, row range `[204,220)`, 16 B content-blind V3 facet, read **G24N4** = 24 signed i4 loci; *"each nibble is a context pointer (signed ±8 window offset), not a strength"*; slots 16..24 reserved-zero; **"Status: EXPERIMENTAL — not in the operator-locked §3 catalogue"** | `.claude/v3/soa_layout/tenants.md:57` |
| Code home | `crates/lance-graph-contract/src/causal_witness.rs` — `WITNESS_REGISTER_BYTES = 12` (:76), `WITNESS_LOCI = 24` (:79), `NAMED_LOCI = 16` (:81), `LOCUS_LABELS` (:85-110), `enum Locus` (:116-), `BasinAnchor = 8` (:134) |
| Reading surface | `agrees_at` (:352), `agreement_count` (:361), `quorum` (:371), `contradiction` (:378), `resolves_to` (:334), `project(&WideFieldMask)` (:415), `elected` (:444) |
| Register byte offset | `witness_fabric.rs:78-89` — `WITNESS_REGISTER_START = ValueTenant::CausalWitness.value_offset() + 4`, `const _` asserted `== 176` |
| Multi-hop second-order chase | `witness_fabric.rs:421` `resolve_chain(focal_idx, window, locus: Locus, max_hops)` → `ChainResolution{final_offset, hops, out_of_horizon, budget_exhausted}`; lens twin `:517` |
| Own open proposal **P2** | `.claude/v3/soa_layout/witness-nibble-lane.md:127-131` — *"whether slots 8–10 (`basin_anchor`/`supported_by`/`supports`) belong at all: they compress long-lived AriGraph graph relations into a ±8 **stream** window whose referents are generally not 8 events away"* |

### 1c — the third adjacent edge carrier

Tenant 2 `MaterializedEdges`, `U64 × 4 = 32 B`, row `[48,80)`, *"4 out-of-family
`CausalEdge64`"* (`tenants.md:45`). Three witness/edge carriers now sit in one
row geometry — tenant 2 (materialized causal edges), tenant 14 (±8 context
loci), and EW64 (episodic edge refs, uncolumned). **Value-slab headroom is
292 B** (`tenants.md:62-63`), so the constraint is doctrine, not space.

### 1d — the four W1 scaffolds, each self-documenting the missing seam

All four carry V3 wave `W1` in `.claude/v3/MODULE-TABLE.md` (rows :167, :243, :227, :48):

1. `episodic_edges.rs` — shipped hot tier (above). MODULE-TABLE :167:
   *"W1 envelope/ownership (SoA edge column)"* — the column it is tagged for does not exist.
2. `witness_table.rs:38-41` — *"This file declares the column-type primitive
   **only**. It does **not** wire the table into `CausalEdge64`, `MailboxSoA`, or
   any emission path — those are later slices."* Its `:18-21` states the purpose:
   *"the chain of W-references across edges forms a **Markov-style belief-update
   arc** through episodic-reference vectors."*
3. `soa_view.rs:257-277` — the deferred accessor and the design note:
   *"WHAT EpisodicWitness64 IS: it is **AriGraph living in the mailbox SoA
   view** … EW64 is the *particle* (discrete, addressable, exact witness
   pointer); the windowed projection `arigraph::markov_soa` is the *wave*.
   Both ARE AriGraph."* … *"STATUS: `EpisodicWitness64` is NOT YET a code
   symbol."* (The spec's citation `soa_view.rs:77` is **stale** — the note
   now sits at :257-277; correct it when the spec is next edited.)
4. `graph/arigraph/markov_soa.rs:55-56` — *"The truly-correct home is still
   *inside the EW64-in-SoA seam* (P1+P2 of the three-Markovs ordering); this
   module is the agnostic wave-projector that seam will host."* Verified
   independently: `witness_tombstone.rs` is `todo!()`-only (:6) **and absent
   from `graph/mod.rs`** (grep) — orphaned.

### 1e — live precedents this plan must follow, not re-invent

- **`deepnsm-v2::basin::BasinCode`** (`crates/deepnsm-v2/src/basin.rs:42-55`) — the
  house pattern. A basin = *"one subject's outgoing-object neighborhood … the
  deepnsm-v2 realization of the le-contract L1–L3 `part_of:is_a` episodic
  rail"* (:9-12); carries `self_code`/`width`/`members`/`contradiction`; MUL
  competence is *"a **derived READ** over `max_width`, **not a new tenant**"*
  (:58-60), gated by a **held-out split-half** (`heldout_split_gate`, :25-34).
  **Follow this**: derived reads + held-out gates, never a new carrier.
- **Sub-byte-in-a-u64-lane is already sanctioned**: tenant 1 `Qualia` is
  `U64 × 1` carrying `QualiaI4_16D` = *16 signed-4-bit channels*
  (`tenants.md:44`); tenant 9 `Kanban` is `U64 × 1` with its own packing
  (`tenants.md:52`). EW64's 4×(4b:12b) packing is therefore **not** a doctrine
  problem — provided it lands as a **U64 lane** (F8), never a §3 12-byte
  carving (12-bit locals straddle bytes; §3 is byte-axis).
- **`witness_fabric::resolve_chain`** is the shipped second-order chase — ARC-A2
  :189 grades it *"**REAL**, but ±8 window, single locus dimension, facet
  flagged experimental"*; the ONLY row→row-through-its-own-kind mechanism here.
- **`nars/meta_basin.rs`** — 1477 lines of trajectory clustering, meta-basins,
  perturbation-stability, outlier suggestions. `nars/mod.rs:17` is its **only**
  appearance outside itself. ARC-A2 :190: *"**REAL, and discarded**"*.
- **No basin-promotion seam exists anywhere.** ARC-A2 :167-179, Case B:
  `EpisodicMemory::basins()` (`arigraph/episodic.rs:243`) has **zero callers**
  (verified); `EpisodicBasins` is returned by value, never written to any
  lane/column/row/dataset; `Locus::BasinAnchor` *"confirmed unwritten"*;
  promotion seam *"**DOES NOT EXIST.** Not built, not stubbed"*; *"Reusable
  Type-B pattern to copy: **None exists anywhere**"*.

---

## §2 THE COLLISION (this plan's core)

Two witness shapes, one row, **zero mutual references** — verified by grep in
both directions:

```
grep EpisodicEdges64|EW64|episodic_edges  causal_witness.rs witness_fabric.rs witness-nibble-lane.md → 0 hits
grep CausalWitness|Locus|G24N4            episodic_edges.rs                                          → 0 hits
```

|  | tenant 14 `CausalWitnessFacet` | EW64 `EpisodicEdges64` |
|---|---|---|
| carrier | 12 B of a 16 B facet, `[204,220)`, register at 176 | uncolumned `u64` |
| unit | 24 signed i4 **loci** | 4 × `EdgeRef{family,local}` |
| what a slot means | a **named dimension** (16 of them) | an **anonymous slot**; index = recency |
| what a value means | a **context pointer**: signed offset in a ±8 stream window; `0` = unbound | a **reference**: `(family, local)` in episodic basin space |
| reach | ±8 stream positions, same window | 16 families × 4095 locals; cross-session reach is a *separate* column (`episodic_edges.rs:21-22`) |
| ordering | sign = orientation (before/after) | slot = recency (MRU) |
| second-order | `resolve_chain` multi-hop, budget + horizon | none |
| eviction | none (slots are dimensions) | `coldest()` → `DemotionSink` |
| status | EXPERIMENTAL, outside the locked catalogue | shipped type, no column |

### The two readings, with consequences

**Reading L — two RUNGS of one witness ladder.** Tenant 14 answers *"where in
my immediate context is dimension D grounded, and does my peer agree?"* — a
**positional** question inside a bounded window. EW64 answers *"which basin
members did this episode touch, most-recent-first?"* — a **referential**
question over a durable address space. Under L they compose: EW64 supplies
the durable edge; tenant 14 supplies the local context frame; `Locus::BasinAnchor`
(:134) is the seam between them, and its own doc-comment already says so —
*"The event binding me to my AriGraph basin (`part_of:is_a`, L1)"* — while
ARC-A2 :174 records it as **confirmed unwritten** because *"needs an AriGraph
`part_of:is_a` rail; none is wired"*. That unwritten slot is exactly the join L
predicts. Consequences of L: both land; tenant-14 P2 (slots 8–10) is **resolved
against those slots** — `basin_anchor`/`supported_by`/`supports` are long-lived
graph relations wearing a ±8 stream coordinate, i.e. the wrong rung — and they
either re-point at EW64 refs or vacate to reserved.

**Reading R — rivals; one wins.** Both are "the witness column", and keeping two
carriers for one concern is the duplication this workspace has paid for
repeatedly (`docs/TYPE_DUPLICATION_MAP.md`; the D-TSC-1 lesson). Under R the
loser retires: either tenant 14 absorbs episodic reference (widening loci from
±8 offsets to opaque refs — destroying the "loci, not magnitudes" operator lock,
`causal_witness.rs:46-53`), or EW64 absorbs context (per-slot named dimensions —
destroying the MRU invariant, since a dimension slot cannot also be a recency
slot). Either direction kills an operator-locked invariant.

**Recommendation: READING L, subject to the falsifier in D-EWU-1.** Three
source-grounded reasons. (1) The charter refuses the collapse: *"EPISODIC
WITNESS … EPISTEMIC CAUSALITY … **Do not collapse them.** The existing
CausalWitnessFacet remains pointer-like/contextual evidence"* (:513-529) — a
naming of two rungs. (2) Its registered 256:256 readings already list
**`episodic-witness basin : role`** (:211) *distinct* from `context : role`.
(3) R breaks an operator-locked semantic in either direction, and neither break
has a sponsor. **L is a claim and this plan does not bank it** — D-EWU-1 is the
probe that can kill it.

---

## §3 PROPOSED RESOLUTION — the seam-cut order

The order is load-bearing. Phase D (the column) currently has **two possible
landing zones** and no ruling; landing it before the demarcation is how one gets
a third edge carrier next to tenants 2 and 14.

```
D-EWU-1  demarcation ruling (L vs R)          ← OPERATOR DECISION, probe-backed
     │                                           nothing else may land first
     ├── D-EWU-2  spec correction: THREE decisions → TWO
     │
     ▼
D-EWU-3  Phase D — the EpisodicWitness64 lane, shaped by the D-EWU-1 winner
     │    (under L: a new U64×1 ValueTenant + the soa_view accessor;
     │     under R: no new tenant — the winner's reading is widened)
     ├── D-EWU-4  EpisodicEdge bridge location (independent; may run in parallel)
     ▼
D-EWU-5  Phase B — Hebbian co-fire        (GATED ×3)
D-EWU-6  Phase C — DemotionSink impl      (GATED on OQ-11.6)
D-EWU-7  Phase E — arcuate ±5 wire        (NEEDS-DESIGN)
D-EWU-8  basin-promotion seam (Case B)    ← this is what actually gates ARC E
D-EWU-9  markov_soa re-home (particle/wave co-location)
```

---

## §4 DELIVERABLES

Every gate is pre-registered **before** the work, per the falsifiability rule;
each names what input would make it fail.

### D-EWU-1 — the witness demarcation ruling (READING L vs READING R)

A `docs/architecture/` demarcation doc + an operator ruling. The doc states the
§2 table, the two readings, and the outcome of the falsifier below.

**Gate — `F-EWU-DEMARCATION`, a two-sided worked example.** Build an
offline probe (both types live in the zero-dep contract crate, so it compiles
without the workspace) that poses ONE recall question to both shapes:

- **Arm A — a case only EW64 answers.** Fixture: an episode whose referent sits
  either at stream distance > 8 **or** in another family (`family ∈ 1..=15`).
  Assert `resolve_chain(.., Locus::BasinAnchor, ..)` returns
  `final_offset: None` with `out_of_horizon: true`, while
  `EpisodicEdges64::strongest()` returns the `EdgeRef`.
  *Anti-vacuity (mandatory):* the same fixture with the referent at distance
  ≤ 8 **and** in-family must be answered by BOTH — otherwise Arm A is
  measuring "tenant 14 never works", not a rung boundary.
- **Arm B — a case only tenant 14 answers.** Fixture: two co-window rows;
  question = *"at which context event is my `Kausal` dimension grounded, and
  does my peer agree?"* Assert `agrees_at(other, Locus::Kausal)` /
  `agreement_count` produce it, and that no `EpisodicEdges64` read can:
  `EdgeRef` carries no named dimension and no before/after orientation.
  *Anti-vacuity:* state explicitly what an EW64-side answer would look like;
  if one is constructible, Arm B is empty.
- **Kill condition (pre-registered):** if **both** arms come back empty — each
  shape answers everything the other does — the shapes are **rivals**, READING
  L is falsified, and §3's order re-runs from the R branch. A single empty arm
  is also informative: it names which shape is subsumed.

**Blocked on:** nothing. This is the plan's first and cheapest deliverable.

### D-EWU-2 — correct the spec: THREE decisions → TWO

`.claude/specs/episodic-witness64-ce64-prefetch.md` §3 lists three open operator
decisions. **Decision 2 is shipped**: `RawEdge(i8)` at `counterfactual.rs:456`,
`impl EpisodicEdge for RawEdge` at `:472-479`, `clamp_i4` at `:482`, and the
structural test `raw_edge_is_one_byte_mantissa_only` asserting
`size_of::<RawEdge>() == 1` at `:506-508` (the §9 council verdict recorded it as
D-ATOM-4). What *remains* of Decision 2 is only the **bridge location** — see
D-EWU-4. Also fix the stale `soa_view.rs:77` citation (now :257-277).

**Gate:** the edited spec cites `counterfactual.rs:456` and `:472-479`; a reader
who greps those lines finds shipped code. Append-only — the §3 text is regraded
in place, never deleted.

### D-EWU-3 — Phase D: the `EpisodicWitness64` lane

**Under READING L** (recommended branch): a new `ValueTenant` variant,
`U64 × 1`, 8 B, appended after `CausalWitness = 14` at row `[220,228)` — the
same shape as `Meta`(0) / `Qualia`(1) / `Kanban`(9), which is what makes EW64's
sub-byte 4×(4b:12b) packing legal (F8). **Not** a 16 B facet: `local` is 12 bits
and straddles bytes, so the §3 byte-axis carvings cannot express it. Plus the
`soa_view.rs:258-260` accessor the note already specifies.
**Under READING R:** no new tenant — the surviving shape's reading widens and
this becomes a migration, not a mint.

**Gate (both branches):** ① `ValueSchema::Full` count stays compile-asserted
`== VALUE_TENANTS.len()` (`canonical_node.rs:1198`); ② **field-isolation
matrix** per F9 — write the lane, assert all 14 other tenants byte-unchanged,
and the paired inverse; ③ **zero `ENVELOPE_LAYOUT_VERSION` bump** (append into
the 292 B headroom, RESERVE-DON'T-RECLAIM) — a bump means the design is wrong,
not that the version should move; ④ the charter's `F-HIERARCHY-NOT-AUTHORITY`
read: the lane stores `(family, local)` references only, no ontology semantics
(F1).
**Blocked on:** D-EWU-1; the batched mint (never a solo tenant edit — the §5
mint discipline of `triangle-tenants-gestalt-separation-v1.md`); and the spec's
own Phase-D gate, `cognitive-shader-driver`'s `MailboxSoA<N>` not building
offline.

### D-EWU-4 — `impl EpisodicEdge for CausalEdge64`: pick the bridge location

`counterfactual.rs:172-177` states the blocker verbatim: *"The bridge impl …
is BLOCKED on workspace structure. Options: (a) impl in `causal-edge` gated on
a `lance-graph-contract` feature; (b) newtype in a thin bridge crate."*

**Gate:** whichever option lands, `RawEdge`'s structural guarantee must survive
— the bridge exposes the i4 mantissa at bits 46–49 and **structurally cannot**
reach plasticity (50–52) / W / truth / temporal. Falsifier: a compile-fail test
(or a `size_of`/API-surface assertion) proving the bridge type offers no
plasticity accessor. **Independent of D-EWU-1** — may run in parallel.

### D-EWU-5 — Phase B: Hebbian co-fire (per-plane `PlasticityState`)

The spec's §9 council resolved the model: coarse strength = MRU slot order
(shipped); per-edge Hebbian = per-plane `PlasticityState`
(`crates/causal-edge/src/plasticity.rs:6-25` — 3 bits, bit0=S / bit1=P /
bit2=O, `ALL_FROZEN` = *"Established clinical pattern"* :16-17). The rejected
alternative, `Heel::plasticity()` (`high_heel.rs:168-171` — `(truth_meta >> 24)
& 0xFF`, a per-**basin** u8 over a ≤240-edge container), was dropped as a
phantom join (`E-BASIN-NOT-EDGE-PLASTICITY`). **Still an OPERATOR DECISION**
(§6): a council recommendation is not a ratification.

**Gate (3, all pre-existing):** (a) `causal-edge` builds offline; (b) the
plasticity ruling; (c) F9 field-isolation across the v1 `PLAST_SHIFT=49` /
v2 `=50` boundary — the exact class codex caught 5× in sprint-11. Plus:
co-fire idempotence, and plasticity monotone toward hot with a **can-it-stay-
silent** twin (a non-co-firing input must leave the bits unchanged).

### D-EWU-6 — Phase C: a production `DemotionSink`

Today the only implementor is the test `VecSink` (`episodic_edges.rs:631`).
Gated on OQ-11.6 (the `surreal_container` fork + Lance pin) for surreal-LIVE,
or the LanceDB-LIVE fallback.

**Gate:** demote → persist → **re-prefetch** round-trip: an edge evicted by
`coldest()` reappears in the hot word when its basin re-activates. Anti-vacuity:
an edge that was never demoted must **not** appear (a sink that returns
everything carries the same information as one that returns nothing).
**Charter check:** the sink is a durable frontier, never a wall (F7) — a lagging
sink must not block promotion.

### D-EWU-7 — Phase E: the comprehension ↔ arcuate ±5 wire

Verified blocker stands: `deepnsm::parser::SentenceStructure`
(`crates/deepnsm/src/parser.rs:56-66`) carries `triples` / `modifiers` /
`negations` / `temporals` — **no ambiguity or candidate signal**. NEEDS-DESIGN,
and the load-bearing question is firewall *placement*, not which top-k (spec
§7).

**Gate:** coreference resolved over a ±5 fixture **and** the firewall assertion
— no COCA rank crosses the boundary; only `Binary16K` + the resolved opaque
`(family, local)`. Anti-vacuity: a fixture where the wrong candidate would win
without the chain.

### D-EWU-8 — the basin-promotion seam (Case B) — **this is what gates ARC E**

ARC-A2 §3 Case B: discovery exists (`arigraph/episodic.rs:243`, zero callers),
the pointer slot exists (`Locus::BasinAnchor`, unwritten), and the seam between
them **does not exist, not even as a stub**. Without it, ARC E's
"top-down HHTL observation" has nothing stable to observe: a discovered basin
is re-densified per call with no stable identity (:173).

**Gate — `F-PROMOTION` (ARC-A2 :280):** *a discovered basin can structurally
become a promoted stable ref, not only a sidecar.* Concretely: `basins()` →
a `NodeGuid::mint_for` address → a write reaching `Locus::BasinAnchor` → the
same basin resolvable at a later `DatasetVersion`. **Charter fences:** F2 (the
promoted parent is REFERENCED, children copy nothing) and F3 (**no promotion
DTO** — the promoted basin is a reading of the canonical substrate, not a new
shape). Anti-vacuity: a basin that dissolves under a perturbed hop budget must
**not** promote (the stability guard `meta_basin.rs` already implements one
level down).

### D-EWU-9 — re-home `markov_soa` (particle/wave co-location)

`markov_soa.rs:55-56` names its own staging: *"the truly-correct home is still
inside the EW64-in-SoA seam."* Once D-EWU-3 lands, the wave projector sits
beside the particle it projects.

**Gate:** the 4 `markov_soa` tests (part of the 124-green `graph::arigraph`
suite) stay green byte-identical across the move; the injected distance stays
AriGraph's own `cam_pq` — **not** a language table (`markov_soa.rs:28-36`).

---

## §5 NON-GOALS

- **No reasoner in-repo.** `ogar-elk` stays external; `ontology_warrant.rs` is
  the consumer-side grading contract and stays unwired here (ARC-A2 :163).
- **No `meta_basin` fix** beyond naming it (§7). Wiring a consumer for
  higher-order structure is ARC-E work and needs its own probe.
- **No new SoA, no `HierarchyPlane`, no promotion DTO, no generic structural
  algebra** (F3). D-EWU-3's lane is a `ValueTenant` variant in the *existing*
  slab, or it is nothing.
- **No `MetaAgent` homunculus** (charter :711). `resolve_chain` is already the
  second-order mechanism; ARC E extends it, never replaces it.
- **No collapse of episodic witness into epistemic causality** (F6) — that is
  precisely what §2's ruling protects.
- **No implementation.** Zero code changes in this plan.
- **No board writes in this session** (one file, by instruction); the hygiene
  entries §0 lists are owed by the landing commit.

---

## §6 OPEN OPERATOR DECISIONS

| # | Decision | Unblocks | Recommendation |
|---|---|---|---|
| **OD-1** | **The demarcation** — READING L (two rungs: positional context vs episodic reference) or READING R (rivals, one wins) | D-EWU-3, and transitively ARC E | **L**, subject to D-EWU-1's two-armed falsifier. R requires breaking an operator-locked invariant in either direction |
| **OD-2** | **Plasticity model** for Phase B Hebbian co-fire — `high_heel::Heel` W15 byte-3 scalar (`high_heel.rs:168-171`, per-basin, 0=frozen..3=hot, ≤240 edges) **vs** `causal-edge::PlasticityState` (`plasticity.rs:6-25`, 3 bits, one per S/P/O plane) | D-EWU-5 | **per-plane** — spec §8/§9; S/P/O demonstrably harden independently (`freeze_s`/`heat_s`, `ALL_FROZEN` = clinical pattern). `Heel` is a per-basin roll-up on a different object |
| **OD-3** | **Sense-candidate source** for the Phase E arcuate ±5 wire — `vocabulary` neighbors / `similarity` top-k / net-new | D-EWU-7 | reuse the proposer layer, **placement is the real question**: top-k computed in DeepNSM emitting opaque ranks (spec §6③/§7). Ranked last |
| **OD-4** | **`EpisodicEdge` bridge location** — (a) `impl` in `causal-edge` behind a contract feature, or (b) a thin bridge crate | D-EWU-4 | either, provided the mantissa-only guarantee stays **structural** (`counterfactual.rs:172-177`) |
| **OD-5** | *(conditional on OD-1 = L)* tenant-14 **P2**: do slots 8–10 (`basin_anchor`/`supported_by`/`supports`) belong in a ±8 stream register at all? | tenant-14 catalogue entry | re-point at EW64 refs, or vacate to reserved. `witness-nibble-lane.md:127-131` |

---

## §7 DEFERRED — missing integration (named, not scheduled)

| Item | Evidence | Why deferred |
|---|---|---|
| **Phase B Hebbian co-fire** | spec §2; `plasticity.rs:6-25`; v1/v2 `PLAST_SHIFT` minefield | 3 gates, one of which is an operator decision (OD-2) |
| **Phase C cold-connectome `DemotionSink` impl** (surreal-LIVE / LanceDB-LIVE) | only `impl` is the test `VecSink`, `episodic_edges.rs:631` | OQ-11.6 substrate + lancedb-offline |
| **Phase E arcuate comprehension wire** | `parser.rs:56-66` carries no ambiguity signal | needs-design; firewall placement (OD-3) |
| **The basin-promotion seam** (episodic basin → stable higher-order ref, Case B) | `episodic.rs:243` zero callers; `Locus::BasinAnchor` unwritten; ARC-A2 :175 *"DOES NOT EXIST. Not built, not stubbed"*; :179 *"Reusable Type-B pattern: None exists anywhere"* | D-EWU-8; **the real ARC-E gate** |
| **`meta_basin` discard** | 1477 lines; sole external mention `nars/mod.rs:17`; ARC-A2 :190 *"REAL, and discarded"* | §5; higher-order structure has nowhere to land until D-EWU-8 |
| **`witness_tombstone.rs` orphaned `todo!()`** | `:6` *"D-ATOM-5 scaffold only — `todo!()` bodies"*; **absent from `graph/mod.rs`** (verified) — unreachable | the hot→cold→tombstone lifecycle is Phase C's cold half |
| **`AwarenessRevise` / `awareness.revise`** | `counterfactual.rs:379`-area trait, BLOCKED on signature discovery | ARC E; named so it is not rediscovered |
| **`WitnessTable<64>` emission wiring** | `witness_table.rs:38-41`, explicitly "later slices" | needs the D-EWU-3 column to be wired *into* |
| **`MetaWord`: zero spare bits, zero `ClassView` refs** | ARC-A2 :192 (6+4+8+8+6 = 32, ~40 hand-assembling call sites) | ARC E needs it class-projected; not this arc |

---

## §8 What this plan deliberately does NOT decide

It does not pick the winner (OD-1 is the operator's), mint a tenant, touch
`ENVELOPE_LAYOUT_VERSION`, widen `Locus`, add a locus to EW64, implement
`DemotionSink`, or schedule ARC E. It cuts one seam: **which shape is the
episodic witness and which is the epistemic context frame** — because until
that is ruled, `EpisodicWitness64` has two landing zones and every downstream
phase inherits the ambiguity.
