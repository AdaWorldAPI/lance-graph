# Identity / Temporal / Evidence primer — READ BEFORE PROPOSING ANY CARRIER

> **READ BY:** any agent touching identity, provenance, evidence, versions,
> masks, or ownership. **MANDATORY** before proposing a new struct in
> `lance-graph-contract`.
>
> Written 2026-07-27 after a session repeatedly proposed new carriers before
> establishing what the substrate already represents. Every claim below is
> `file:line`-grounded. Do not infer semantics from names.

---

## 0. Two operator rulings (non-negotiable premises)

1. **The V3 substrate is always a V3-shaped GUID using `ClassId:AppId:ClassView`.**
2. **Always SoA-owned; the SoA owns the kanban per SoA; always zero-copy, never
   serialized.** Verified: `soa_envelope.rs:18-19` — *"Nothing is serialized or
   transmitted; the backing bytes are resident in-place, zero-copy from creation
   to Lance tombstone"*; `soa_envelope.rs:16-17` — *"a Lance version IS a
   coherent LE in-place layout at cycle N"*.

**Consequence that kills a whole class of proposals:** a "side ledger",
"append-only receipt file", or any serialized provenance store is **forbidden by
construction**. The cold/authoritative path is *Lance versions of the same
in-place LE bytes*, never a second serialized representation.

---

## 1. THERE IS ONE CLASS-IDENTITY CARRIER

`ClassId` and the GUID's `classid` are **the same carrier**, seen through
different projections. Do not model them as two systems.

```
NodeGuid  =  [u8; 16]                       canonical_node.rs:33-35  "16-byte canonical instance key"
   ≡  FacetCascade                          facet.rs:94-100          byte-identical, reinterpret no-op
      ├── facet_classid : u32   [0..4)      ← the class address
      │     ├── canon  : u16  (HIGH)  = ClassId  — the shared CONCEPT
      │     └── custom : u16  (LOW)   = AppId    — the render lens
      └── tiers : [FacetTier; 6]  [4..16)   ← 6 × (8:8), coarse→fine:
            HEEL · HIP · TWIG · LEAF · family · identity      ← the INSTANCE lives here
```

- `pub type ClassId = u16` — `class_view.rs:54`. *"Per-row class discriminator —
  the Cognitive-RISC `class_id`/`shape_id`… ≤65,535 shape-families;
  OD-CLASSID-WIDTH ratified."* This is the **concept component**, not a rival id.
- `compose_classid(canon, custom) -> u32` / `split_classid` — `ogar_codebook.rs:337-352`.
  Active order `CLASSID_ORDER = ClassidOrder::CanonHigh` (`ogar_codebook.rs:313`).
- `classid_canon` / `classid_custom` / `classid_concept` — `ogar_codebook.rs:357,364,428`.
  **Accessors on one value**, not separate identities.
- `classid_canon_compat` — `ogar_codebook.rs:385-394`. Reads BOTH stored orders;
  the flip is mint-forward and never reinterprets persisted ids.
- `AppPrefix::render(concept) -> u32` — `ogar_codebook.rs:241`. `Core = 0x0000`
  is documented *"shared canonical core (default ClassView, no render lens)"*.
- `ClassView` is a **trait** — `class_view.rs:903`. *"the parser+schema… projects
  row → typed view, late-bound"*. It is what the address **resolves to**, and it
  chooses **which reading** of the 12-byte register applies (`facet.rs:95-96`:
  *"which ClassView interprets the 6 tiers' 8:8"*).
- `NodeGuid::facet()` — `canonical_node.rs:441-443`, with the byte-identity test
  `nodeguid_facet_bridge_is_byte_identical` (`canonical_node.rs:451`).

### Which operations preserve identity vs collapse to concept
| Operation | Uses | Correct because |
|---|---|---|
| routing / storage / equality of an addressed form | **full `classid` u32** | the app lens is part of the addressed form |
| RBAC, ontology, cross-app concept convergence | `classid_canon`/`classid_concept` | the shared concept is the subject |
| reading corpora spanning the order flip | `classid_canon_compat` | serves both stored forms |
| rendering | `AppId` + `ClassView` | selects the skin, not the subject |

**Comparing only the canon half is WRONG** wherever two apps' addressed forms
must stay distinct (storage, dedup of addressed rows). **Comparing the full u32
is WRONG** wherever the shared concept is the subject (RBAC grants, concept
convergence). Neither is universally right — that is why both accessors exist.

---

## 2. Instance identity

- The instance is the GUID tail: `family` + `identity` (`GuidParts`,
  `canonical_node.rs:601-615`), a.k.a. `local_key()` = trailing 6 bytes,
  *"the only discriminator once the prefix is resolved"*.
- **Scope:** `local_key` is unique **only within a resolved classid prefix** — it
  is not globally canonical.
- **Uniqueness is NOT enforced.** `debug_assert_identity_unique`
  (`canonical_node.rs:338`) is `debug_assert!`-gated and its **only call sites are
  in `mod tests`** (`canonical_node.rs:2026, 2034`). Its own doc says *"Call on
  insert with whatever set/bitmap the mint path keeps"* — an obligation on the
  caller, honoured by no production mint path.
- Its panic message admits reuse: *"or reused — mint a non-zero family to expand
  before this fires in prod"*.
- **No tombstone, deletion, or compaction path exists** in `canonical_node.rs`
  (verified ABSENT). So identity-reuse-after-delete is unspecified, not prevented.
- **No monotonic counter / serial / sequence field exists on any identity type**
  (`NodeGuid`, `GuidParts`, `MailboxId`, `EdgeRef`, `EpisodicEdges64`,
  `RelationId`, `WitnessEntry`). Verified ABSENT.

---

## 3. Temporal model — five distinct things, three of them MISSING

| Concept | Carrier | Status |
|---|---|---|
| dataset snapshot | `LanceVersion = u64` (`temporal.rs:47`, *"the storage frame's clock tick"*); `DatasetVersion(u64)` (`scheduler.rs:33-36`) | EXISTS |
| reader horizon | `QueryReference.ref_version` (`temporal.rs:114-116`, *"The `KnowledgeHorizon` — the Lance version the reader is pinned at"*) | EXISTS |
| epistemic policy | `EpistemicMode {Strict, Aware, Retro}` (`temporal.rs:52-61`) + `TemporalStatus {Contemporary, Anachronistic, Spoiler, Unknowable}` (`temporal.rs:91-102`) | EXISTS |
| registration horizon | `DeinterlaceRow::knowable_from()` (`temporal.rs:294-306`) — distinct from `lance_version()` | EXISTS (trait method, not a persisted field) |
| multi-writer scope | `QueryReference.server_id: u16` + `hlc_tick: Option<u64>` (`temporal.rs:110-119`); module doc calls `(server_id, lance_version, hlc_tick)` *"the deinterlace key"* | **MECHANISM IMPLEMENTED + TESTED; PRODUCTION WIRING DORMANT** — `deinterlace` sorts on `(hlc_tick ?? lance_version, lance_version)` and `deinterlace_hlc_orders_across_frames` / `deinterlace_mixed_hlc_falls_back_to_lance_version` pass; what is dormant is that no substrate call site yet sets non-zero `server_id` / `Some(hlc_tick)` (`default()`/`at()` hardcode `0`/`None`, `temporal.rs:126-151`). See §5.7. |
| **per-row last-modified version** | — | **MISSING** (`row_version` is only a *parameter name* in `classify`, never a field) |
| **transaction / commit identity** | — | **MISSING** (`transaction/mod.rs` holds execution-regime typestates, not commit ids) |
| **append-log / observation / receipt serial** | — | **MISSING** |
| **two changes to one row within one version** | — | **NOT EXPRESSIBLE** — `DeinterlaceRow` gives exactly one `lance_version()` per row |

**Do not substitute the nearest available version field for a missing one.**
Read-at ≠ snapshot ≠ changed-at ≠ observation identity.

---

## 4. Evidence carriers — what each actually answers

| Carrier | Answers | Does NOT answer |
|---|---|---|
| `Stamp(pub u64)` (planner `belief.rs:31`; deepnsm-v2 `belief.rs:33`) | "do two bases share a *source bit*" | which event; which source (bits are folded); replay identity |
| `Belief.stamp` | evidential base **and** (via `!= default()`) "is this observation-grounded" | — **two orthogonal questions on one field** |
| `Belief.premises: Vec<u32>` | derivation inputs, *"Arena indices of premises"* (planner `belief.rs:99`) | anything replay-stable — arena indices are allocation-order |
| `SupportReceipt` / `SupportLedger` (`causal_audit.rs`) | which kind of support, from whom, when, how strong | **which event** — the receipt has no identity of its own |
| `WitnessEntry {mailbox_ref, spo_fact_ref}` (`witness_table.rs:80-95`) | which mailbox + SPO fact a W-slot resolves to | evidence membership |
| `EpisodicEdges64` (`episodic_edges.rs:103`) | up to 4 MRU episodic edges | provenance |

**`Stamp::source(id) = 1u64 << (id % 64)`** (planner `belief.rs:36`; deepnsm-v2
`belief.rs:38`) — 64 sources max, silent modulo aliasing beyond. Documented as
CONSERVATIVE (false overlap only, never false disjointness) — **that doc is
correct**; the defect is that source ≠ event, not that folding is unsound.

### Verified drift between the two BeliefArenas
```rust
// planner/src/nars/belief.rs:193   — HAS the empty-stamp guard
if stamp != Stamp::default() && b.stamp.disjoint(stamp) {
// deepnsm-v2/src/belief.rs:189    — LACKS it
if b.stamp.disjoint(stamp) {
```
`Stamp::default()` (all-zero) is `disjoint` from everything, so in deepnsm-v2 a
repeated unsourced observation **pools unboundedly**. The planner routes it to
CHOICE and has the test `empty_incoming_stamp_does_not_pool`
(planner `belief.rs:447`); deepnsm-v2 has no equivalent test. Both files carry
the *same* explanatory doc text — the prose stayed in sync while the code did not.

---

## 5. Fixed-width discipline — the actual rule

**A bitmask is justified only where each bit has a stable, predefined meaning in
a bounded vocabulary.** The repo mostly gets this right:

| Carrier | Bit meaning | Verdict |
|---|---|---|
| `ThoughtMask(u8)` / `ThoughtField` (`recipe_kernels.rs:137,111-128`) | STATIC, *"bit positions are stable (do not reorder — append-only basis)"* | ✅ correct use |
| `FieldMask(u64)` (`class_view.rs:70`) | STATIC, *"once instances persist, a field's bit position never moves and retired bits are never reused"*; positions ≥64 **ignored, NOT folded** (`:79-83`) | ✅ correct use |
| `StepMask(u64)` (`step_mask.rs:40`) | STATIC per template version; positions ≥64 **ignored, NOT folded** (`:53-56`) | ✅ correct use |
| `WideFieldMask` (`class_view.rs:221`) | STATIC; >256 is *"a loud refusal, never a silent drop"* (`:525-529`) | ✅ correct use |
| `WitnessTable<64>` (`witness_table.rs:112`) | slot index; **N=64 is DOMAIN-derived** — *"matching the 6-bit address space of the W-slot field"* (`:100-102`); out-of-range → `Err`, no panic | ✅ correct use |
| `EpisodicEdges64` (`episodic_edges.rs:103`) | DYNAMIC 4×16-bit slots; `push` returns `None` when full; `promote` evicts slot 3 **to a `DemotionSink`** | ✅ explicit eviction, not silent |
| **`Stamp(u64)`** | **DYNAMIC — bit = a runtime-assigned source id, folded `% 64`** | ❌ **the one violation** |

**The discriminator:** `FieldMask` and `StepMask` refuse to fold and say so in
their docs; `Stamp` folds. Dynamically-arriving identities do not belong in
globally-interpreted Boolean positions.

**Never confuse a `u64` identity (2⁶⁴ values) with a `u64` Boolean mask (64
positions).** `LanceVersion` is the former; `Stamp` is the latter.

Known silent-clamp sites worth auditing before reuse: `CausalEdge64::with_w_slot`
uses a **debug-only** `debug_assert!(w <= 63)` (`causal-edge/src/edge.rs:953`);
`with_inference_mantissa` **silently wraps** out-of-range values (`:966-967`);
`set_temporal` under v2 **silently drops** the write (`:588`).

---

## 5.5 SoA ownership — and the island that is NOT SoA-owned

**The owning SoA** is `MailboxSoA<const N: usize>` (`cognitive-shader-driver/src/mailbox_soa.rs:58`).
True struct-of-arrays — one fixed-size array per column:

```
mailbox_id : MailboxId          ← the SoA's own identity
energy              [f32; N]      plasticity_counter  [u8; N]
last_active_cycle   [u32; N]      last_write_cycle    [u32; N]
edges  [CausalEdge64; N]          qualia [QualiaI4_16D; N]
meta   [MetaWord; N]              entity_type [u16; N]
temporal [u64; N]                 expert [u16; N]        sigma [u8; N]
+ three per-row style lanes, appended AFTER Kanban
```

**This SoA owns its Kanban** (`mailbox_soa.rs:179,222-228`):
- `current_cycle: u32`
- `phase: KanbanColumn` — **`pub(crate)`, not `pub`** — *"Mutated only via
  `MailboxSoaOwner::advance_phase` / `try_advance_phase`; starts at
  `KanbanColumn::Planning`. Read it through the `MailboxSoaView::phase` getter."*

One SoA → one Kanban → one owner (`MailboxSoaOwner`) → sole mutator. Verified.

### ⚠ `BeliefArena` is an object-shaped island, NOT V3 SoA ownership

```rust
// planner/src/nars/belief.rs:129-136   — and deepnsm-v2/src/belief.rs likewise
pub struct BeliefArena {
    entries: Vec<Belief>,          // ← array-of-STRUCTS on the heap
    index: HashMap<CStmt, u32>,    // ← a second heap map
    passes: u32, reached_fixed_point: bool,
}
```

It has **no SoA columns, no Kanban, no `mailbox_id`, no V3 GUID addressing**.
`Belief.premises` are `Vec<u32>` *arena indices*. It is AoS + hashmap — the
opposite shape from `MailboxSoA`.

**Consequence for any evidence work:** the question is never "how should the
arena own evidence identity". Both candidate answers — the old `Stamp` and the
withdrawn `SourceRegistry` — are heap structures inside a structure that is
already outside V3 ownership. `SourceRegistry` added a **third** heap map
(`Vec<SourceId>`) inside the island and called the result "containment".

The V3-shaped question is:
> Which `MailboxSoA` column, mask, edge, or Kanban transition represents this,
> and which SoA owns the row?

Until `BeliefArena` is either (a) backed by SoA columns or (b) explicitly
declared a non-substrate diagnostic surface, **no evidence carrier placed inside
it can be substrate-conformant**, however well-typed it is.

---

## 5.7 Parallel execution + temporal deinterlacing

Claims are labelled. **Repository absence is NOT disproof of owner-specified
architecture** — it marks where the invariant is not yet made explicit.

```
1. V3 GUID resolves the addressed SoA state.
2. The SoA-owned Kanban performs the SYNCHRONOUS legal transition.
3. Large populations of transitions execute IN PARALLEL.
4. Each transition casts its continuation fire-and-forget.
5. BatchWriter records intents + coalesces physical writes asynchronously.
6. Lance versioning records the resulting temporal positions.
7. temporal.rs deinterlaces the parallel standing wave.
8. QueryReference / knowledge horizons prevent hindsight leakage.
9. The cohort converges within the ~64k / 550 ms WALL-CLOCK envelope.
```

### ⚠ The SLA is a cohort envelope, not a per-update budget
**Do NOT compute `550 ms / 64 000`.** That division assumes serial execution.
The ~64k updates occupy **one overlapping wall-clock interval**; their compute
and write costs are not summed sequentially. The question is *"can the parallel
cohort converge, deinterlace, and become available inside 550 ms"* — never *"can
each update finish in 8.6 µs"*.

> An earlier revision of this file contained that division. It was wrong and is
> retracted.

### Capacity: addressing + cache envelope, NOT a concurrency ceiling

**~64k is a preferred operating point, not a limit.** It falls out of *16-bit
addressing* + a *cache-friendly working set* — never from the SLA, never from
"how many updates may run in parallel", never from V3 semantics.

| Scale | Addressing | Working set (× 512 B row) | Regime |
|---|---|---|---|
| ~64k (2¹⁶) | 16-bit | **32 MiB** | preferred, cache-friendly |
| ~262k (2¹⁸) | 18-bit | 128 MiB | wider addressing, more cache pressure; comfortable on newer CPUs |
| ~4M (2²²) | ~22-bit | **2 GiB** | memory-resident, outside ordinary L3 — a different **memory-latency** regime |

**[ARITHMETIC CHECK — the only part of this the repo corroborates]** The
footprints are *exactly* derivable from a code-asserted constant:
`const _: () = assert!(core::mem::size_of::<NodeRow>() == 512);`
(`canonical_node.rs:735`, with `NodeGuid == 16` and `EdgeBlock == 16` at
`:733-734`). `2¹⁶ × 512 B = 32 MiB` and `2²² × 512 B = 2 GiB` land on the nose.
So the capacity ladder is **internally consistent with the canonical node**, not
arbitrary.

**The 512 B stride is UNIFORM — there is no second, smaller row shape.**
Every SoA row reserves 512 B *on paper*, always:
- `pub value: [u8; 480], // 32..512  (reserved — comes after)` (`canonical_node.rs:729`)
- *"32..512 are the class-resolved value slab. **Sum = 512 = stride.**"* (`:764`)
- The 480 B slab is where `energy` / `meta` / `qualia` / `entity_type` live
  (`:708`) — i.e. the named `MailboxSoA` columns are **tenants inside the
  reserved slab**, not a competing row shape.
- `EdgeBlock`: *"Canonical, not mandatory: the 16 bytes are ALWAYS reserved
  (zeroed when unused)"* (`:640`); *"always reserved, never shrunk"* (`:8`).
- **RESERVE, DON'T RECLAIM** (`:16`): *"a zero tier means 'not consulted', never
  'compacted away'"*.
- The envelope **enforces** it — `verify_layout` has an error variant for *"Sum
  of column byte-widths does not equal the declared row stride"*
  (`soa_envelope.rs:124`).

Only the **baked** (Lance columnar) form may omit empties — that is storage-layer
compression, and it does **not** change the logical stride. So the 32 MiB / 2 GiB
arithmetic applies uniformly to `MailboxSoA`, with no per-crate recomputation.

> **Retraction:** an earlier revision of this file claimed `MailboxSoA`'s per-row
> footprint was "a different, materially smaller sum" than `NodeRow`'s 512 B, and
> warned against transferring the 32 MiB figure. That was a category error — it
> listed the hot columns and mistook them for the whole row, when they are
> tenants *within* the reserved 480-byte value slab. Both are the same 512-byte
> row: `NodeRow` is its AoS view, `MailboxSoA` its SoA projection.

**⚠ One caveat that does stand:** **16-bit addressing is not expressed in the SoA
API.** `MailboxSoA` is `<const N: usize>` and its accessors take `row: usize`
(`mailbox_soa.rs:624,630`). The `u16`s present (`entity_type`, `expert`) are
**values, not addresses**. The address widths are [OWNER-SPECIFIED]; the row
footprints are arithmetically checked.

### Four dimensions that must never substitute for each other
| Dimension | Meaning | Do NOT infer |
|---|---|---|
| **Address width** | how many SoA positions can be named | 16-bit ⇏ 64k *sequential* operations |
| **Active population** | how many rows/updates participate concurrently | 64k parallel ⇏ 550 ms / 64k per-update latency |
| **Cache envelope** | does the working set stay near the CPU | 32 MiB-friendly ⇏ larger populations unaddressable |
| **Wall-clock SLA** | how fast the parallel cohort must converge | 4M addressable ⇏ same cache behaviour as 64k |

**The ownership model does not change across these scales.** What changes is
physical locality and memory latency. V3 GUID resolution, `ClassView` dispatch,
SoA ownership and one-Kanban-per-SoA are scale-invariant. Above the cache
envelope the expected structural response is **partitioning into several SoAs —
each retaining its own Kanban** — not a different algorithm. *(Partitioning
behaviour at scale is [CODE-AUDIT REQUIRED]: no multi-SoA partitioning policy is
stated in the substrate crates.)*

### Certainty labels
- **[OWNER-SPECIFIED]** ~64k updates execute in parallel within a 550 ms
  wall-clock SLA. *Not located in code as an SLA* — `550_000` appears only as
  `LIBET_COMMIT_WINDOW_US` (a Libet readiness anchor on the
  `Planning → CognitiveWork` crossing, `kanban.rs:146-154`), and `64k` only in
  `onebrc-probe` preset names. **A target the code does not yet state.** The
  *capacity* half is partly corroborated — see the ARITHMETIC CHECK above.
- **[OWNER-SPECIFIED]** The 16-bit / 18-bit / 22-bit addressing ladder and the
  32 MiB / 2 GiB envelopes. Row footprints check out exactly against
  `size_of::<NodeRow>() == 512`, which is the UNIFORM stride (reserve-don't-
  reclaim); the *address widths* themselves are not encoded in any row-index
  type (`row: usize` throughout).
- **[OWNER-SPECIFIED]** Compute + write cost are masked by the parallel pipeline.
- **[CODE-PROVEN]** `BatchWriter` casts are ahead-firing and physical writes are
  coalesced — *"records intent moves AHEAD of any storage write completing"*;
  *"one physical flush coalesces all earlier intents for a row
  (last-state-wins)"*; `cast() -> CastId`, *"NEVER refused"*.
- **[CODE-PROVEN, NARROW]** `BatchWriter` does **not** execute payload compute —
  *"intent recording, nothing else"*; *"the writer never inspects `P`"*. Do not
  attribute the whole masking property to it.
- **[CODE-PROVEN]** The Kanban step itself is **synchronous inline** — *"the
  in-stream synchronous kanbanstep (`VersionScheduler::on_version →
  try_advance_phase(&mut)`), fired inline"*. Fire-and-forget describes the
  **cast**, not the transition.

### Where compute parallelism actually lives [CODE-AUDIT]
- **Actor-per-mailbox** — `lance-graph-supervisor/src/kanban_actor.rs` uses
  `ractor` (`impl Actor`, `ractor::registry::where_is`, `ractor::call!`). One SoA
  = one mailbox = one Kanban = one actor. This is the parallelism unit.
- **Row sweeps inside one SoA** — `MailboxSoA` batch delivery
  (`mailbox_soa.rs:337`, *"Accept a batch of `(target_row, CausalEdge64)`
  deliveries"*) + cycle-guarded late-batch rejection (`:182,:245`).
- **`MailboxSoA::cast_to`** (`mailbox_soa.rs:748-762`) — the W4a ahead-firing
  cast pairing into `BatchWriter`.
- **NOT rayon/`par_iter`** in the substrate: those appear only in the
  `onebrc-probe` benchmark crate. **Gap: no data-parallel executor over the
  cohort is stated in the substrate crates.**

### Temporal deinterlacing — [CODE-PROVEN, with tests]
`temporal.rs` is the standing-wave resolver, not a snapshot wrapper.

- **Sort key** (`temporal.rs:345-351`):
  `(hlc_tick.unwrap_or(lance_version), lance_version)`. The fallback is
  deliberate — falling back to `0` *"would force every missing-HLC row ahead of
  all HLC rows regardless of its version (Codex P2 on #468)"*.
- **Horizon gate**: each row passes `.dispatchable(v_ref.mode)` before ordering.
- **Tests that exist:** `deinterlace_filters_and_orders_single_server`,
  `deinterlace_hlc_orders_across_frames`,
  `deinterlace_mixed_hlc_falls_back_to_lance_version`, and
  **`no_hindsight_streamed_known_game`** — hindsight gating is named and tested.

**Correction to an earlier reading in this file:** HLC is *implemented and
tested* in `deinterlace`; what is dormant is **production wiring** — no
substrate call site sets a non-zero `server_id` or `Some(hlc_tick)`
(`QueryReference::default`/`at` hardcode `0`/`None`). Mechanism present, callers
not yet multi-writer.

**Physical completion order ≠ epistemic order.** Ordering is by the temporal key
above, never by when a cast landed.

### Performance audit — the right questions
| Concern | Question |
|---|---|
| Parallel width | Can ~64k updates remain concurrently active? |
| Wall-clock convergence | Does the **cohort** converge in 550 ms? |
| Synchronous critical path | What must an update finish **before it may cast**? |
| Compute overlap | Which stages run concurrently across rows/mailboxes? |
| Batch coalescing | How many logical updates become one physical write? |
| Deinterlacing cost | What does temporal ordering cost for the whole cohort? |
| Knowledge gating | Can any update observe a **later** Lance version? |
| **Contention** | Does any shared registry/allocator **serialize** the cohort? |
| Failure behaviour | Can one local capacity condition **refuse or stall** the wave? |

### The real PR #854 danger, correctly framed
Not the cost of an O(≤64) scan. A per-arena **mutable** `SourceRegistry` on the
pre-cast path introduces: shared mutable allocation · **insertion-order-dependent
meaning** (so slot assignment becomes nondeterministic under concurrency, hence
**replay-unstable**) · a 64-entry ceiling · **synchronous refusal**
(`CapacityExceeded`) · and a serialization point that can collapse part of an
otherwise parallel cohort around one allocator.

> **Retraction:** an earlier revision called `SourceRegistry` "the forbidden
> confirmation-ledger shape under another name". The `E-ACK-ELIMINATED-1` ruling
> forbids *"a confirmation ledger (a persisted id→version map)"* — write-durability
> bookkeeping. `SourceRegistry` is an id→**slot** map for source interning:
> structurally similar, **not covered by that ruling**. The contention and
> refusal objections above stand on their own.

---

## 5.8 Can the adjacency layer BE the belief store? Provisionally NO

`nars/mod.rs:3` states the intent — *"**NARS Schema** = truth values stored as
edge properties in the adjacency store"* — and it is **backed by real code**, not
aspirational: `adjacent_truth_propagate` exists and is tested
(`adjacency/propagate.rs:19`, tests at `:104,:114`). Correcting an earlier
reading in this file that treated it as an unbacked doc line.

**But propagation is not storage.** Two different capabilities:

```
truth propagation over an adjacency batch   ≠   unique keyed belief storage with revision
```

Surface as it exists today:

| | evidence |
|---|---|
| `AdjacencyStore::new(rel_type, num_nodes)` / `from_edges(…, edges: &[(u64,u64)])` (`csr.rs:45,123`) | CSR is built **whole**; no incremental admission |
| `adjacent` / `adjacent_incoming` / `edge_ids` / `out_degree` / `in_degree` (`csr.rs:62-92`) | **every accessor is `&self`** — no `&mut`, no upsert, no keyed insert anywhere |
| `EdgeProperties::with_nars_truth(mut self, …) -> Self` (`properties.rs:45`) | **builder-consuming**; truth is supplied at construction |
| `EdgeProperties::truth_value(edge_id) -> Option<(f32, f32)>` (`properties.rs:54`) | **read-only**, and only `(frequency, confidence)` |
| node addressing | positional `u64` over `num_nodes` — **not keyed by `CStmt`** |

**Consequences for the migration:**
- *One statement → one authoritative belief under concurrency* — cannot be
  guaranteed, because there is **no admission path at all**.
- *Atomic revision of `truth + groundedness + evidence + contradiction + rung +
  ancestry`* — cannot be done: `truth_value` carries `(f32, f32)`, so
  `contradiction`, `rung`, `stamp` and `premises` have **nowhere to live**.

**Provisional classification: `propagation-only kernel`.** Not wrong — it does
what it claims. It is simply **not a store**. So PR D is *build the keyed mutable
layer*, not *migrate onto an existing one*; scope accordingly.

### The statement key is concept-level by design, and lossy
```rust
pub struct CStmt { pub s: u16, pub cop: Copula, pub p: u16 }   // belief.rs:77-84
pub enum Copula { Inh, Sim, Impl, Rel(u16) }                   // belief.rs:54-63
```
`s`/`p` are *concept ids* — the canon half only. **No `AppId`, no `ClassView`, no
instance tiers.** The relation term for `Rel` rides inside the copula variant, also
`u16`. So the map is `V3 GUID → concept`: **surjective, not injective** — two
app/view-resolved statements sharing a concept projection **collide by
construction**.

This is deliberate (*"the arena composes concept-level STATEMENTS by their shared
terms"*), and it forces a decision **before** any capability audit:

> **⊘ CORRECTION (operator, 2026-07-27).** An earlier version of the table below
> said *"canonical V3 addressing for beliefs would be a category error."*
> **That was wrong, and it was the same error twice.** Two claims got collapsed:
>
> - **True:** the map `addressed-form subject/predicate GUIDs → CStmt` is
>   surjective, so it **cannot be inverted**. A concept belief's key does not
>   recover the addressed forms that produced it.
> - **False (what I wrote):** therefore a concept-level belief cannot itself be
>   V3-addressed.
>
> A belief instance can carry its own canonical V3 GUID. That GUID addresses
> **the belief**, not its constituents — *"a belief about (canonical subject
> concept + copula + canonical predicate concept)"* is a perfectly good
> addressable entity. "These two values are not the same value" is not
> "these two things cannot coexist" — precisely the `ClassId`-vs-`classid`
> mistake this primer exists to prevent, repeated one layer down.
>
> The real distinction is **what the addressed entity means**:
>
> ```text
> full addressed-form belief   — keyed by addressed subject/view/predicate forms
> canonical concept belief     — keyed by canonical concept projections
> ```
>
> Both are V3-addressable. The existing arena has already chosen the second:
> concept-level convergence across app/render views — consistent with one
> carrier having concept and app/view projections rather than two competing
> identity systems.

| Option | Cost |
|---|---|
| **A. concept-keyed belief only** | cheapest, matches today. Preserves current NARS semantics, but loses which addressed forms or observations grounded a belief unless that is attached elsewhere. |
| **B. full-GUID-form-keyed belief** | preserves addressed-form distinctions, but changes **reasoning semantics** and fragments cross-app convergence unless an `is_a` rail re-unifies. |
| **C. canonical concept belief + separately addressed grounding** | the strongest fit. **NOT** "V3-GUID-keyed evidence attached as metadata" — each observation event becomes **its own canonically addressed V3 entity or relation**, not an annotation keyed by existing object GUIDs. |

> **⊘ HARD CONSTRAINT (operator, 2026-07-27): NEVER any serialization or
> materialization. Ever.** This binds how Option C may be built, and it is the
> single easiest way to get C wrong.
>
> "Each observation event becomes its own canonically addressed V3 entity" means
> the event **is addressable** — it has a canonical GUID and an owner. It does
> **not** license any of the following, all of which are forbidden:
>
> - a provenance **side-store**, ledger, journal, or append-log of events;
> - a **serialized** event record (JSON / bincode / protobuf / any wire form) —
>   `SoaEnvelope` is *"zero-copy from creation to Lance tombstone"*
>   (`soa_envelope.rs:16-19`); Lance's columnar I/O writes the **same LE bytes**
>   from the in-place backing store, and nothing is serialized *in order to be
>   stored*;
> - a **materialized** second structure that duplicates state the substrate
>   already holds — no evidence index built alongside the rows, no cached
>   projection kept in sync, no shadow copy of ancestry;
> - a "rebuild it from the serialized mapping" recovery story. There is no
>   serialized mapping. If a design's safety argument is *"we can reconstruct
>   it"*, the design is already outside the substrate.
>
> An evidence event is **a row in an SoA owned by a mailbox**, written
> on-behalf-of that owner, whose history is Lance **versions of those same
> bytes**. The address is the identity; the bytes are the storage; the version
> axis is the history. Nothing else is created.
>
> Test before proposing any grounding mechanism: *does this add bytes that are a
> copy of, or a translation of, bytes the substrate already owns?* If yes, it is
> materialization and it is rejected — regardless of how convenient it is for
> provenance queries. A query is a **projection over what is already there**
> (`temporal.rs` deinterlace is the reference shape: a version-range read, still
> a projection, zero copies), never a structure maintained on the side.

Option C, expanded — three layers, none folded into another:

```text
canonical concept-level belief
        ├── truth state
        ├── contradiction / rung
        ├── derivation ancestry
        └── evidence edges
                ↓
           canonical evidence-event entities
                ├── source attribution
                ├── temporal birth
                └── affected V3 rows / addressed forms
```

```text
belief semantics   = concept-level
observed target    = addressed V3 form or forms
evidence identity  = observation event
```

This is not a compromise between A and B. It is the **orthogonal decomposition**
the withdrawn `SourceRegistry` accidentally forced into view: source membership,
evidence-event identity, and object identity are three axes, and the registry
failed because it tried to serve all three with one.

## 5.9 Consumer census — COMPLETE (10 of 10 modules)

> **Two corrections to the 7/10 partial reported earlier in this file.** The
> partial said "order-dependence: ZERO" and speculated that `premises` might be
> opaque payload. `tactics.rs` — the module that was still running — **refutes
> both**. Recorded here rather than silently overwritten, because the shape of
> the error matters: the encouraging result came from the 7 modules that had
> *finished*, and the hard cases were concentrated in the one that had not.

### ⚠ Order-dependence is REAL — 3 sites, all in `tactics.rs`, all feeding budget caps

| site | mechanism | severity |
|---|---|---|
| `rcr_abduce` `by_pred` (`:182`, consumed `:196-256`) | grouping preserves arena push order → nested-loop emission order → **budget cap truncates** | **test-locked**: `rcr_floor_and_budget` asserts a deterministic prefix, with a comment at `:190` relying on *"members already in arena-index order"* |
| `tr_diverge` Sim scan (`:293-320`) | candidates pushed in scan order into `out.candidates`, **returned as-is, never sorted** | output order is caller-visible |
| `inh_by_subject` (`:339`) → `cas_abstract` (`:375-450`) | `props` grouping order decides which up/down candidates **survive the budget cut** | changes which beliefs get derived |

**This is not incidental iteration order — it decides which candidates survive
truncation.** Per the "name the real order" rule: the operative order is
**arena-admission order**, and a test currently locks it. PR A therefore
**cannot** be a pure behaviour-preserving refactor of these three unless that
order is either named and preserved explicitly, or the truncation is made
order-independent (e.g. rank-then-cut) — which is a **behaviour change** needing
its own decision.

The other 7 modules (`insight` · `basin_resonance` · `epiphany` · `elevation` ·
`reach_out` · `regulate` · `insights`) remain genuinely order-free: commutative
reductions, `BTreeMap`/`BTreeSet` grouping, explicit terminal `sort_by`.

### ⚠ `premises` ARE positional handles — the speculation is REFUTED

`tactics.rs` stores the *same* `u32` it uses for indexed lookup:
`premises: [r, o]` (`rcr_abduce`), `[sg_idx, pi]` / `[gi, sg_idx]`
(`cas_abstract`). So `premises: Vec<u32>` is **not** opaque payload — it is arena
positions, and `BeliefHandle` must preserve positional identity or the premise
graph breaks. (`insights.rs:119` merely *clones* premises, which is why the
7-module partial read them as inert.)

### Slice-dependence: 9 real sites, all in `tactics.rs`
`arena.entries()[r]` / `[o]` (`:217,221,240,242`), `[sg_idx]` (`:387`),
`[pi]` (`:395,415`), `[gi]` (`:425,445`) — **random access by a stored `u32`
handle**. Everything else across all 10 modules is `.iter()`-chaining or
`.len()`, which is mechanical.

### Missing API, discovered by the census
`tr_diverge` (`:285-288`) runs a **redundant linear `.position()` scan** to
recover the arena index of a belief it *just fetched* via `get()` — because
`get()` returns `&Belief` without its index. An O(n) scan for a value the index
map already knows. **`belief index by statement` is a missing accessor**, not a
new requirement.

### The minimal semantic API — 9 reads, 3 writes, deduplicated across all 10 modules
**Reads:** `count all` · `derived only (rung>=1)` · `beliefs by copula` (± grouped
by subject or predicate) · `grounded is_a grouped by predicate` ·
`belief by statement` · **`belief index by statement`** · `belief by handle` ·
`beliefs sharing a term` (term + copula) · `existence scan over a subject's
parents` · `max term id` · `aggregate/statistical reduction over a scalar field`.
**Writes:** `admit observation` · `admit derived` · `close to fixed point`.

Small and nameable. **A generic boxed iterator would swap a mega-accessor for a
fuzzier one** — and would not serve `belief by handle` at all.

---

## 6. Ownership / persistence / replay

- `SoaEnvelope` (`soa_envelope.rs:170`) — the owner of the in-place backing store.
  `ENVELOPE_LAYOUT_VERSION = 2` (`:54`).
- **Write-on-behalf iron rule** (`soa_envelope.rs:165-169`): every consuming crate
  writes ON BEHALF OF the envelope's mailbox id, never directly.
- Mutation lives on the **owner** type, never on the read trait
  (`soa_envelope.rs:148-149`; mirrors `MailboxSoaView` vs `MailboxSoaOwner`).
- `MailboxId = u32` (`collapse_gate.rs:121`) — *"unique u32 identity of one
  spatial-temporal meaning accumulator"*.
- **Replay stability:** anything whose meaning depends on allocation order
  (arena indices in `premises`, `Stamp` bit assignment) is NOT replay-stable.
  Anything derived from the canonical GUID + Lance version IS.

---

## 7. VERIFIED GAPS (say MISSING; do not substitute)

1. **Evidence-event identity** — no receipt/observation/serial identity exists.
2. **Per-row mutation version** — no persisted last-changed field.
3. **Transaction/commit identity** — none.
4. **Intra-version ordering** — one row = one point on the version axis.
5. **Replay-stable derivation provenance** — `premises` are arena indices.
6. **Dependence model** — nothing represents whether two sources share a common
   cause, though the workspace *measured* non-independence (cloned-lane probe:
   +94 % naive agreement, similarity 1.000000).
7. **Enforced instance uniqueness** — debug-only, test-only call sites.

---

## 8. DO NOT INVENT

1. **Do not create a second classid carrier** because layers expose different
   projections. `ClassId`, `classid_canon`, `classid_concept`, `GuidParts`,
   `facet_classid` are **accessors on one value**.
2. **Do not mistake a component accessor for a separate identity system.**
3. **Do not create an arena-local replacement for canonical persistent identity**
   (this is what `SourceRegistry` did; withdrawn in PR #854).
4. **Do not use source identity as evidence-event identity.**
5. **Do not use a dataset snapshot version as observation-event identity.**
6. **Do not assign dynamically-arriving updates to fixed, globally-interpreted
   Boolean positions.** Follow `FieldMask`/`StepMask`: refuse, never fold.
7. **Do not treat projection/rendering differences as independent evidence**
   without first proving the semantic distinction.
8. **Do not use a Boolean API where the architecture distinguishes true, false,
   and unknown.** "Not known to overlap" ≠ "known disjoint".
9. **NEVER any serialization or materialization. Ever.** (operator, 2026-07-27 —
   promoted from "no serialized side-store for provenance", which was too narrow:
   it banned one shape and left materialization open.) No side-store, ledger,
   journal, or append-log; no serialized record in any wire form; **and no
   materialized second structure that duplicates or translates bytes the
   substrate already owns** — no evidence index alongside the rows, no cached
   projection kept in sync, no shadow ancestry copy. Zero-copy from creation to
   Lance tombstone; the cold path is Lance **versions of the same LE bytes**.
   A query is a **projection over what is already there** (`temporal.rs`
   deinterlace is the reference shape), never a maintained side structure.
   Corollary: *"we can reconstruct it from the serialized mapping"* is not a
   safety argument — there is no serialized mapping.
10. **Do not widen a mask to fit more members** — `WideFieldMask`'s doc calls
    exceeding the cap *"a split signal, not a case to widen the mask type"*.
11. **Do not reuse `Stamp`'s shape as precedent.** It is the one carrier in this
    inventory that violates the mask discipline its siblings document.
12. **Do not evaluate a design by whether it implements `Serialize`, can be
    frozen into a census, or can be rebuilt from serialized records.** V3 is
    never serialized, so "not serializable" is not a safety property and
    "reconstructible from a serialized mapping" is not a solution. Judge by
    *which SoA owns it* and *which Kanban governs its transitions*.
13. **Do not let a Rust struct's shape override the substrate's architecture.**
    That `BeliefArena` is a struct with a `Vec` does not make an arena-owned
    registry legitimate — it makes the arena the thing to question.
14. **Do not treat "arena-local" as a containment guarantee.** In V3, containment
    is SoA ownership + write-on-behalf, not privacy of a heap field.
15. **Do not put fallible, allocating, or shared-mutable work before a cast.**
    `cast()` is *"NEVER refused"*. The danger is not per-update cost — it is
    **serializing a parallel cohort around one allocator** and making slot
    meaning depend on insertion order. Never derive a per-update budget by
    dividing the cohort SLA by the update count.
16. **Do not add a confirmation ledger under any name** — the BatchWriter doc
    forbids it explicitly. Durability evidence is the row's own `LanceVersion`.
17. **Do not ride owned bytes on a cast payload.** `P` is a DESCRIPTOR
    (mailbox, dirty row-range, cycle); deltas stay in the SoA backing store.
18. **Do not use physical completion order as epistemic order.** Ordering is
    `(hlc_tick ?? lance_version, lance_version)` through `deinterlace`, gated by
    `QueryReference.mode`.
19. **Do not describe Lance versions as durability acknowledgements only.** They
    are the temporally sorted standing wave — the substrate replay traverses.
20. **Do not treat repository absence as disproof of owner-specified
    architecture.** Report where an invariant is not yet explicit in code, tests,
    or docs; do not conclude it is false.

---

## 9. Reading order for the next coding session

1. This file.
2. `soa_envelope.rs` module docs (ownership + zero-copy contract).
3. `canonical_node.rs` §CANON block in `CLAUDE.md`, then `facet.rs:88-110`.
4. `ogar_codebook.rs:285-400` (the one flippable classid composition).
5. `temporal.rs` **whole file** (the epistemic model is the spec).
6. `planner/src/nars/belief.rs` **and** `deepnsm-v2/src/belief.rs` side by side —
   they differ at one line and it matters.
7. `.claude/board/EPIPHANIES.md` top 5 entries.

---

## 10. Revised redo sequence (operator, 2026-07-27) — SUPERSEDES the earlier PR A–E

The earlier five-PR sketch is replaced. Two corrections drove the change: the
concept-belief-can-itself-be-V3-addressed ruling (§5.8 ⊘ CORRECTION) and the
adjacency reframing below. Bound by §8 rule 9 — **never any serialization or
materialization, ever.**

### The adjacency question was posed backwards

PR B must **stop asking** *"can `AdjacencyStore` become the mutable belief
store?"* — its existing API already answers that: not without becoming a
fundamentally different component. The CSR layer is an **immutable projection +
propagation kernel**, and that is a clean role, not a deficiency.

The question is upstream instead:

> Which authoritative V3-owned edge/row state **produces** the immutable CSR
> projection that `adjacent_truth_propagate` consumes?

```text
canonical SoA-owned belief/edge state
        ↓
Lance-versioned projection          (a projection — not a materialized copy)
        ↓
CSR adjacency batch
        ↓
adjacent_truth_propagate
        ↓
derived updates cast back on behalf of the owner
```

The CSR is a **compute projection**, never the owner. Remaining audit targets,
all upstream of the kernel:

- Where are canonical **mutable relations** currently owned?
- Which SoA columns or `EdgeBlock`s feed `AdjacencyStore::from_edges`?
- Are truth properties **baked from authoritative state**, or supplied ad hoc by
  callers?
- Is there an existing **write-on-behalf path** from propagation output back to
  the owner?
- Does CSR reconstruction happen **per Lance version, per query, or only in
  tests**?

If no such owner exists, the missing component is **not** "a mutable CSR" — it is
the canonical concept-belief / relation SoA that CSR should project.

### Two caveats that reopen the "7 modules are order-free" result

The census called seven modules order-free because their reductions are
commutative and their groupings use `BTreeMap`/`BTreeSet` with an explicit
terminal `sort_by`. **Commutativity is not sufficient.** Two gaps:

**Floating-point accumulation.** For `f32`, `(a + b) + c ≠ a + (b + c)` in
general. Any sum, average, density, confidence aggregation, or entropy
calculation can vary bit-for-bit under reordered or parallel iteration **even
when the mathematical operator is commutative**. A `HashMap`/`HashSet` iteration
feeding a float fold is the worst case — arbitrary order, not merely admission
order. The perturbation test must therefore assert one of:
- **exact equality**, where integer/fixed-point behaviour is intended;
- a **bounded numerical tolerance**, where float variation is accepted;
- a **deterministic accumulation order**, where reproducibility is required.

Which of the three applies is a per-site decision, and stating it is part of A0.

**Ties.** An explicit terminal `sort_by` is deterministic only when the
comparator defines a **total** order. A score-only comparator leaves ties, and
Rust's `sort_by` is *stable* — so prior iteration order (arena admission order)
leaks straight back in through the tie. `max_by` returns the LAST maximum;
`min_by` returns the FIRST minimum. A tie becomes an observable behaviour
difference the moment a `truncate`/`take`/`[..n]` follows the ranking.

Every ranking needs a stable secondary key:

```text
score
then canonical belief handle / statement key
```

So the census grows two columns:

| Caller | Floating reduction? | Total tie-breaker? |
|---|---|---|

### Premise indices: lightly consumed, still architecturally toxic

Only `tactics.rs` dereferences them and `insights.rs` merely clones them — but
`Vec<u32>` still means:

```text
premise identity = admission position in this particular arena build
```

That is **not replay-stable** and cannot become canonical ancestry (§7 gap 5).
Narrow usage makes the migration cheap; it does not make it optional. The
replacement handle must land **before** premises cross into the authoritative
substrate.

A provisional opaque handle is the right PR A1 move:

```rust
pub struct BeliefHandle(/* private: current arena index */);
```

It may still wrap `u32` initially. The point is to **stop callers from knowing
that**, so the backing can later become a canonical concept-belief reference
without a second caller-wide migration.

### The sequence

**PR A0 — finish the census.** Per call site, record: semantic query · fields
read · direct indexing · premise dereference · order dependence · **floating
reduction** · **tie-breaking** · whether **mutation** is required · expected
**cardinality** and hot-path status. No code changes.

**PR A1 — representation decoupling, behaviour unchanged.**
- private `Belief` fields;
- opaque `BeliefHandle`;
- remove public slice access; replace every `entries()[i]`;
- add named semantic scans and lookups (the 9 reads / 3 writes of §5.9, plus the
  missing `belief index by statement`);
- **retain `Vec<Belief>` internally** — this PR does not change storage;
- add **shuffled-admission tests**;
- add **tie and float-order falsifiers** (per the §"Two caveats" decision above).

> **Avoid boxed trait-object iterators.** The future SoA path needs
> monomorphizable or batch-oriented access, not one virtual call per belief — and
> a `Box<dyn Iterator>` cannot serve `belief by handle` at all. Swapping a
> mega-accessor for a fuzzier one is not decoupling.

**PR B — ratify belief granularity.** Record explicitly, as a decision:

> A belief is keyed by **canonical concept-level** subject, copula and predicate.
> `AppId`/`ClassView`-specific forms **converge** into this concept belief.
> Addressed-form and observation distinctions live in **grounding/evidence
> relations**, not in the belief key.

Then define the bijection for the **belief entity itself** — not for `CStmt`
back to its source object GUIDs, which is surjective and not invertible (§5.8).

**PR C — locate or design the authoritative owner.** Determine whether an
existing V3 relation/edge SoA can own: concept statement identity · truth ·
groundedness · rung · contradiction · derivation reference. If none can, **say
the concept-belief SoA is missing** — do not mutate `AdjacencyStore` into it.

**PR D — CSR projection parity.** Prove the authoritative state projects into the
existing adjacency kernel while preserving current reasoning results.

**PR E — evidence events.** Only after the canonical concept-belief owner exists:
observation event → canonical V3 event identity → evidence edge to the concept
belief → affected addressed-form rows → temporal birth and horizon behaviour.
**Then** replace `Stamp`. Bound throughout by §8 rule 9 — the event is a row in
an owner's SoA, never a serialized or materialized record.
