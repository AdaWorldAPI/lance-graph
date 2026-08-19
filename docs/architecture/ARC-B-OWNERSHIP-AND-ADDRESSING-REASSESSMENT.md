# ARC-B reassessment — the oracle boundary and the addressing gap

> **Status:** ASSESSMENT (2026-08-19). Zero code. This document answers the
> operator's ten-point reassessment request and its two superseding rulings
> (the ownership correction for `dismech-rs` PR #7, and the
> HHTL-addressing correction for lance-graph PR #969). It **supersedes**
> the four plans of PR #969 wherever they conflict; each affected plan
> carries a ⊘ block pointing here.
>
> **Evidence base:** a six-agent read-only sweep over `lance-graph`,
> `ndarray`, `lance-graph-java`, `OGAR`, `MedCare-rs` (private — only
> structural facts are quoted here) and the DisMech corpus. Every claim
> below carries the file:line an agent actually read. Where the sweep
> **refuted** something this session previously asserted, the retraction is
> stated in place rather than quietly dropped.

---

## §0 THE ONE-PARAGRAPH FINDING

HHTL is **already** the first canonical tenant of every 512-byte row —
`key(16) | edges(16) | value(480)`, with the key carrying
`classid(4)` plus the `6×2×8-bit` facet. It is also **zero on every baked
row in both production bakes**, and every consumer that needs hierarchy
rediscovers it by walking edges. The gap is therefore not storage, not
layout, and not a missing carrier: **the OU column exists in every object
and nothing writes or reads it.** Two independent repositories say so in
their own words, and five structurally distinct hand-rolled ancestor
walks in one repository are the measured consequence.

---

## §1 CONFIRMED / REFUTED — does the evidence support the ruling?

### Confirmed

**C1 — The canonical row is `16 | 16 | 480`, HHTL first.**
`crates/lance-graph-contract/src/canonical_node.rs:706-730` —
`NodeRow{key: 0..16, edges: 16..32, value: 32..512}`. Every `VALUE_TENANTS`
entry sits at or after byte 32 (`:935-1034`); **nothing overlaps or
reinterprets 16..32**. `ogar-obo` independently states the identical
geometry: *"`key(16) | edges(16) | value(480)`"*
(`OGAR/crates/ogar-obo/src/lib.rs:22-35`).

**C2 — HHTL is dormant in BOTH production bakes.** Not "partially used" —
zero.
- `OGAR/crates/ogar-obo/src/lib.rs:344-353` + `docs/EDGE-LANES.md:44-51`:
  *"The bake has no basin: HEEL/HIP/TWIG and leaf are zero on all 68,797
  rows."*
- MedCare `docs/joinmap/join-map.md:103`: *"HHTL is **dormant** on every
  baked row (heel/hip/twig/leaf all 0 across 68797 rows)."*
- Every real HHTL *reader* in MedCare falls into one of three classes and
  **none reads the key of a baked artifact**: it mints its own keys
  (`gotham_view.rs:104,308,891`), recomputes tiers in RAM at load
  (`obo_store.rs:690` `compute_cascade`), or asserts HHTL is null.

**C3 — Ontology ancestry is rediscovered by edge walking, everywhere.**
`OGAR/crates/ogar-obo/src/reason.rs:369-391` — `ancestors()` is a BFS
transitive `is_a` closure over the edge lanes, depth-capped at 64 hops.
Hierarchy targets live in the **value slab** edge lanes
(`classid + 4×u24`, lanes 7..30, `edges.rs:80-99`), never in the key.

**C4 — The hand-rolled-route census reproduces, and is worse than
predicted.** Five structurally distinct mechanisms answer "who are my
ancestors" in ONE repository, each with its own dedup, its own cycle
guard, its own semantics:

| # | Site | Mechanism |
|---|---|---|
| 1 | `graph_feed.rs:1114-1129` | LIFO `Vec` + `NodeSetMask` bitset dedup, multi-parent closure |
| 2 | `graph.rs:296-321` | `Vec` stack + `HashSet<(u32,u32)>` + explicit `CYCLE_GUARD_DEPTH` |
| 3 | `obo_store.rs:690-760` | load-time single chosen-minimal-parent chain → HEEL/HIP/TWIG |
| 4 | `obo_store.rs:1113-1142` | per-query re-climb of that same chain (rail-prefix O(1) reject first) |
| 5 | `atlas.rs:453-552` | rail-register positional read **if populated**, else Kahn's-algorithm longest-path |

> **⊘ STORNO 2026-08-19 (same day, re-measured at source and in the object
> store).** Two corrections to this document, recorded here rather than by
> editing the findings above.
>
> **C2's "HHTL is zero on every baked row" is WITHDRAWN.** It holds for
> `obo-core.soa` and is FALSE for `all-lanes.soa`, the current production
> golden image: 352/2,048 sampled records (17.2%) carry non-zero key bytes
> 4..10; the key's 6 bytes are a verbatim prefix of a 24-byte positional
> trie path in the value region; key-prefix ↔ rail-head agreement 314/314.
> The original reading was an **inert-artifact false positive** — the
> artifact was absent from the container, so the read path observed a
> missing substrate rather than a dormant one.
>
> **C5's 58.2% is AGREEMENT and this document states it correctly**
> (`atlas.rs:465`: *"Measured agreement: 58.2 %."*). Two board files
> inverted it to "disagreement"; both are storno'd. Agreement 58.2% /
> disagreement 41.8%. The Kahn fallback reaches depth 18 and calls 2,741
> nodes roots; the register depth is spanning-tree and tops out at the
> 24-byte register.

**C5 — And they do not agree.** `atlas.rs:465,898-904` measures **58.2%
agreement** between the rail-register depth and the Kahn longest-path
depth, because they compute different quantities (spanning-tree depth vs
DAG longest path). **This is the single most important operational fact in
this document:** "just use the rails" is a *semantic* change on a
multi-parent DAG, not merely a faster one. Any migration owes a stated
rule for which quantity is canonical.

**C6 — Provenance/status vocabulary is genuinely absent.** Against
`[asserted, observed, inherited, deduced, extrapolated, synthesized,
counterfactual, revised, unknown]`:

| state | verdict |
|---|---|
| deduced / synthesized / counterfactual / revised | EXISTS — `causal-edge/src/edge.rs:13,21,27,19` (`InferenceType`, 4-bit signed mantissa) |
| asserted | PARTIAL — prose only (`edge.rs:56`, *"bare SPO assertions"*), decodes to `Deduction` |
| unknown | PARTIAL — `spo/truth.rs:38-43` `TruthValue::unknown()` = confidence 0.0 (a scalar, not a tag) |
| observed / inherited / extrapolated | **ABSENT** — no type, field, or variant anywhere in contract or planner |

`MetaWord` (`cognitive_shader.rs:44-76`) is `6+4+8+8+6 = 32` bits — **zero
spare**, hard-coded shifts, no ClassView resolution. `SpoRecord`
(`spo/builder.rs:42-53`) carries **no source/stamp/provenance field at
all**. The nearest real vocabulary is `SourceVerdict{Corroborates, Silent,
Conflicts}` (`ontology_warrant.rs:150-158`), scoped to cross-source
warrant, deliberately without `Unknown` — *"a source that was not
consulted is not a source"* (`:147-149`).

**C7 — `ogar-dismech` is address-authority only.** Two files, two deps
(`ogar-loco`, `ogar-obo`); no `ruff_spo_address`, no `ogar_vocab`, no
`dismech-rs`. It mints `DISMECH_CONCEPT_ID = 0x0333`
(`lib.rs:89`), the canon-high render classid (`:95-97`), **and — more than
expected — the 19 causal predicates as a real `Vocabulary`** (`FnIndex`
consts `0x90..0xA2` at `:141-161`, `DisMechVocabulary` at `:190-222`,
`plug_into` at `:245-249`). Zero external consumers repo-wide. No
`CausalGraph` consumption, no `Facet`/`Mint`, no `ClassView`.

### Refuted — three corrections owed

**R1 — The `~915k / ~10M / ~40k` figures do not reproduce.** Measured:

| quantity | actual | source |
|---|---|---|
| OBO bake rows | **68,797** | `ONTOLOGY_BAKE_STATE.md:131,305-306` |
| MONDO rows | **32,095** | same |
| edges | **152,073** (152,037 joined) | `:307-309`; `atlas.rs:674-677` asserts the joined figure |
| all twelve lanes | **1,225,834** | `RAIL_OFFENE_POSTEN.md:731` — *"graph view can only see 3.8% of it"* |
| SNOMED concepts | **484,031** | `ONTOLOGY_BAKE_STATE.md:587` — separate store, not rail-baked, **no multi-hop walk exists** |

The architecture argument is unaffected — a 68,797-row / 152,073-edge
universe with a 1.2M-row substrate behind it is more than enough to
falsify an addressing claim. But the POC must name the corpus it runs on,
and no document should carry the larger figures.

**R2 — `ogar-elk` DOES produce entailments in-repo. I said the opposite
earlier this session and was wrong.** It is a self-contained pure-Rust EL
subsumption closure (`OGAR/crates/ogar-elk/src/lib.rs:163-166,239-367`) —
`entails()`, `equivalence_cycles()`, `merge()` — implementing R1
reflexivity / R2 transitivity / R3 merge-soundness, deliberately excluding
existentials, role composition, bottom propagation and boolean connectives
(`:70-81`: *"the correct move is to wrap a full reasoner, not to grow this
file"*). There is **no external-reasoner client** anywhere in it. What is
true is the *non-serialization* guarantee — *"There is no serialization
surface — not behind a feature, not optionally. An observer that could
serialize its verdict would invite someone to ship the verdict as if it
were substrate"* (`:42-45`). So: entailment production is real and local;
what is forbidden is *persisting* the verdict as substrate. My earlier
"structurally prevented from producing an entailment" was false.

**R3 — The `32 × (4+12)` "canonical substrate" is a Java-side FIXTURE, not
the storage geometry.** `lance-graph-java/native/lgj-abi/src/rowstore.rs:5-8,33-39`
says so itself: *"The Java side may lay its view out differently… The
64-byte-aligned guarantee arrives with the real `NodeRow`
(`#[repr(C, align(64))]`) wiring, not here."* Grep confirms **zero import
of `NodeRow`/`EdgeBlock`/`NodeGuid`** in the row-store path; the only
`canonical_node` import is `EdgeCodecFlavor` for a trait-default test
(`class_view_provider.rs:186`). And `docs/abi.md:433-438` §10 already
names the intended target — *"`NodeRow`'s `16|16|480` … is already a legal
lane description"* — while §11/§12 shipped the homogeneous 32-lane fixture
instead and §10 was never revised. **PR #969's F1 cited that fixture as
proof of canon. That is the error this reassessment exists to fix.**

---

## §2 WHERE PR #7 CROSSES THE OWNERSHIP BOUNDARY

The boundary the operator ruled:

```
upstream Monarch DisMech  →  dismech-rs (semantic oracle)
                          →  ogar-dismech / ogar-from-dismech (interpretation bridge)
                          →  lance-graph (HHTL / masks / reasoning)
```

PR #7's plan places **integration-layer design inside the oracle**. Named
precisely, with the plan's own section as the citation:

| PR #7 element | Why it crosses |
|---|---|
| Four 512-byte row kinds (disorder/node/edge/predicate) — §PROPOSED RESOLUTION Option A | Generic Lance row architecture. The oracle's job is `CausalGraph`, not row geometry. |
| The edge-row value-slab byte map (`0..8 source_ref` … `64..480 RESERVED`) | A Lance layout decision authored in a crate that must not know Lance. |
| `DISMECH_NODE` / `DISMECH_EDGE` / `DISMECH_PREDICATE` classid mints (D-DCG-10) | Address authority is **already** `ogar-dismech`'s (C7). Minting from the oracle inverts it. |
| The predicate-row ontology (19 frozen ordinals) | **Already exists** in `ogar-dismech` as `FnIndex` `0x90..0xA2` + `DisMechVocabulary` (C7). PR #7 would mint a second, unrelated numbering. |
| The side lane (offset/len windows, field order) | Generic variable-length storage design. |
| The 16-byte edge-block summary (D-DCG-7) | A reading of the canonical edge tenant — lance-graph's contract, not the oracle's. |
| The `causal_link_type` → tag-byte mapping (`0..4`, `255`) | The *distinction* is oracle truth; the *byte encoding* is bridge/substrate. |

**The duplicate-numbering hazard is the sharpest instance.** PR #7 §INPUT
INVENTORY derives 19 predicates and D-DCG-2 proposes freezing them as
ordinals `1..19`. `ogar-dismech` already froze the same 19 as `FnIndex`
`0x90..0xA2`. Two frozen numberings for one vocabulary, in two repos,
neither aware of the other — a `TYPE_DUPLICATION_MAP` entry waiting to
happen. **Only one may be canonical, and C7 says it is the one that
already shipped.**

### What is NOT a violation (keep, unchanged)

Every falsifier and measurement. Specifically: the corpus census
(D-DCG-1); `causal_link_type` four-value preservation; the INDIRECT_KNOWN
vs INDIRECT_UNKNOWN anti-collapse gate (D-DCG-6) with its three-sided
test; `intermediate_mechanisms` order preservation; snapshot pinning
(1,968 vs 1,990 vs 1,996); determinism; the round-trip falsifier with its
**mandatory disable-run**; and the upstream substring bug recorded as an
anti-pattern (`"DIRECT" in "INDIRECT_KNOWN_INTERMEDIATES"` is `True`, so
the `elif` is unreachable and all three values classify as DIRECT). That
last one is exactly the kind of finding an oracle exists to produce.

### The three-way split

**Stays in `dismech-rs` (oracle + falsifiers):** the resolver untouched;
census; the four-value distinction as *corpus truth*; intermediate-list
order; snapshot pins; determinism; a round-trip falsifier **scoped to the
oracle's own artifact**, not to a Lance row; the anti-vacuity disable-run;
the upstream anti-pattern record. Zero-dep posture preserved — and now
*load-bearing*, because it is what makes the oracle independently valuable
(Ruling 8).

**Moves to `ogar-dismech` / a new `ogar-from-dismech`:** the lift+mint
(`CausalGraph` → `Class`/`Facet`), reusing `dismech_render_classid` as the
`classid_of` closure and the existing `DisMechVocabulary` for predicates.
OGAR's own norms put this in a **sibling** crate, not in the guard crate —
`ogar-from-ruff` is the precedent (`mint.rs:49-63` `CompiledClass`,
`:71-74` `mint_graph<P>`, `:81-111` `compile_graph_python<P>`), and
`OGAR-CONSUMER-BEST-PRACTICES.md:382-397` forbids hand-constructing a
bridge inline. *"Pull, never re-mint"* (`OGAR-TRANSPILE-SUBSTRATE.md:258`).

**Moves to `lance-graph`:** row geometry, the edge-tenant reading, the
mask/provenance surface, known-unknown hydration, and every reasoning
operator. None of it belongs in either of the other two.

---

## §3 THE ADDRESSING GAP, STATED AS THE BURDEN-OF-PROOF FLIP

The operator's formulation is now the operative test. Every bespoke route
must answer: **why can I not begin from the HHTL already sitting in the
first 16 bytes?**

| Route | Can HHTL + mask + operator replace it? | Blocker |
|---|---|---|
| `ogar-obo::reason::ancestors` BFS (`reason.rs:369-391`) | **Yes, in principle** | HHTL zeroed (C2). With a populated basin this is a key-prefix test. |
| `obo_store::compute_cascade` (`:690-760`) | **It IS the mint, misplaced** | It computes HEEL/HIP/TWIG *in RAM at load* from edges — i.e. it already derives exactly what the key should carry. Moving it into the bake is the single highest-value change in this document. |
| `obo_store::is_ancestor_of` (`:1113-1142`) | **Partly already does** | Uses an O(1) tier-prefix reject *first*, then re-climbs. The prefix half is the target shape; the climb is the fallback. |
| `graph_feed::disease_mask` / `graph.rs::add_ancestors` | **Yes** | Multi-parent closure ≠ single-parent chain — see C5. Needs the canonical-quantity ruling. |
| `atlas.rs` Kahn longest-path | **Only with a semantics ruling** | 58.2% agreement (C5). This route measures a different thing on purpose. |
| `snomed_edges` 1-hop | **No — different information** | No rail register for lane `0x0319`; no multi-hop walk exists in-tree at all. Genuinely out of scope. |
| `FamilyTrie` (deepnsm-v2 `ancestry.rs`) | **Yes** | Parent-pointer forest in a disconnected `u16` space; the DN walk is the rails walk with different types. |
| `NiblePath` fused-axis (`hhtl.rs:386-402`) | **Layering, not conflict** | It fuses `part_of:is_a` into one routing key; the per-axis rails read the same bytes separately. Both are legal *reads* of one fabric. |

**The migration order this implies** is not "adopt HHTL everywhere". It is:
**(1) mint** — move `compute_cascade`'s derivation into the bake so the key
carries what it already computes; **(2) rule the quantity** — spanning-tree
depth vs DAG longest path, once, with C5's 58.2% as the evidence; **(3)
read** — convert routes in the table above, each behind an equivalence
test against its current answer.

---

## §4 CE64 / EW64 — RE-GRADED AS PROJECTIONS

Per the operator's ruling E, and supported by the evidence:

`CausalEdge64` is a 64-bit packed word whose every bit is spoken for
(`causal-edge/src/layout.rs` `_LAYOUT_COVERAGE` asserts exact 64-bit
tiling). `EpisodicEdges64` is a `u64` 4-slot MRU. Both are excellent
**codecs**. Neither can be the canonical definition of causality or
witness, for a reason now measured rather than asserted: **the canonical
information they would have to define does not fit and partly does not
exist.** `MetaWord` has zero spare bits; `asserted`/`observed`/`inherited`/
`extrapolated` have no representation anywhere (C6); and V3 already retired
the awareness-mantissa on measurement (M20, `D-MTS-6` k\*=1).

The consequence for PR #969's EW64 plan: **D-EWU-3's proposed new
`ValueTenant` is now gated behind the ruling-E question** — *is this
genuinely missing canonical information, or is it a container minted to
avoid completing the address/mask transition?* The plan's own §2 table
answers honestly in one row: EW64's slots are *anonymous, index = recency*.
Recency is a **cache ordering**, not a canonical fact. A tenant is the
wrong home for it until the canonical episodic reference — the
`part_of:is_a` rail `Locus::BasinAnchor` already points at and which is
*confirmed unwritten* — actually exists. **That rail is the prerequisite,
and it is the same rail C2 says is zeroed.** The EW64 tenant and the
ontology addressing gap are the same problem wearing two hats.

---

## §5 THE MINIMAL FALSIFIABLE POC

**Held-out mechanism recovery over the real ontology substrate.**

Measured corpus facts (sweep over 1,968 disorder files):

- **693** files carry ≥1 `INDIRECT_UNKNOWN_INTERMEDIATES` edge.
- **300** carry BOTH a KNOWN-with-non-empty-intermediates edge AND an
  UNKNOWN edge.
- **3,454** `intermediate_mechanisms` entries corpus-wide.
- **0 of 3,454 carry an ontology id.** 100% free prose, always a bare YAML
  string. **This is the POC's hardest constraint and it must be stated up
  front:** the held-out target is text, so recovery is a *ranking over
  ontology candidates* scored against prose, never an id-equality check.

**Three candidates**, each with a KNOWN and an UNKNOWN edge as siblings in
the same pathophysiology section:

| Disorder | MONDO | KNOWN-edge target / intermediate | UNKNOWN-edge target |
|---|---|---|---|
| Wilson Disease | 0010200 | Heart Failure ← *"Copper-related cardiomyopathy and cardiac remodeling."* | Abnormality Of The Menstrual Cycle |
| Homocystinuria | 0004737 | Cysteine ← *"CBS-dependent cystathionine production precedes cystathionase-mediated cysteine synthesis."* | Hepatomegaly |
| Abetalipoproteinemia | 0008692 | Impaired Intestinal Lipid Absorption ← *"absent apoB48-containing chylomicron export from enterocytes"* | Decreased HDL cholesterol concentration |

Homocystinuria is the strongest first target: the named entities (CBS,
cystathionine, cysteine) are **independently ontology-anchored elsewhere in
the same node** via its `genes`/`chemical_entities` blocks, so a text→id
alignment has a checkable answer inside one file.

**The experiment, smallest form:**

1. **Prerequisite (the only substrate change): populate the key.** For the
   MONDO lane of the existing OBO bake, write HEEL/HIP/TWIG from what
   `compute_cascade` already derives at load. No new type, no new tenant,
   no layout change — the bytes are reserved and zeroed today.
2. Hide the KNOWN edge's `intermediate_mechanisms` prose from the input.
3. Give the reader: the disorder's MONDO address, its HHTL prefix, the
   ontology neighbourhood reachable by **prefix test only**, and the
   existing masks.
4. Rank candidate mechanisms. Compare to the held-out oracle string.
5. **The falsifier is comparative, not absolute:** HHTL-prefix-scoped
   ranking must beat the same ranking with the prefix scope removed. If it
   does not, the addressing claim is dead and no amount of downstream
   reasoning rescues it.
6. Only then run the same machinery on the 693 UNKNOWN files, where there
   is no answer key — and require that every produced candidate is tagged
   *inherited* or *deduced*, never *asserted*. **A run that cannot mark
   provenance fails, whatever it recovers.** C6 says that tagging does not
   exist yet: it is the POC's second deliverable, and its home is a
   ClassView-resolved mask read, not a new packed field.

---

## §6 WHAT NOT TO BUILD

- **No new hierarchy abstraction, addressing enum, or graph
  representation.** One fabric, many reads.
- **No `AddressingMode::{Rails, Cartesian}`.** Deleted as a concept.
  Cartesian is a projection.
- **No Morton canonization**, and no HHTL ancestry derived from Morton.
  PR #969's D-HTT-6 is withdrawn on exactly this ground.
- **No HHTL+ Helix** (`6×2×8` **plus** `4×24`). Deferred; not this wave.
- **Never "V1" for HHTL.** Ordinary HHTL is `6×2×8-bit`, read as rails or
  centroid per the class's reading. The retired V1 shape is the flat u24
  *tail*, a different thing.
- **No storage-layout migration to "make HHTL canonical".** It is
  canonical. It is empty.
- **No new `ValueTenant`** until the §4 question is answered in writing.
- **No DisMech reasoning inside `dismech-rs`**, and no second predicate
  numbering.
- **No CausalEdge64 emission from any bake.** No external producer exists
  and none is proposed here.

---

## §7 THE GHIDRA / JAVA ABI STANDARD

The external path is the reference discipline: `binary → Ghidra/SLEIGH/
P-code → r2sleigh → V4 R2IL`, with the domain oracle keeping semantic
authority and the ABI exposing *address + mask + view*. `ghidra-mcp` is a
control plane, never the truth layer.

Applied internally, it yields one falsifier and one debt:

**The falsifier.** If Ghidra's Java object universe can be addressed as
`address + mask + view`, then asking *"at HHTL address X, give me the
masked causal / evidence / inherited / unknown surfaces"* must not require
five domain-specific traversals. C4 says today it requires five.

**The debt.** The Java side does not yet mirror the canonical row. It
ships 32 homogeneous facet lanes (`abi.rs:173-176`, `docs/abi.md:442-448`,
`RowStore.java:10-14`) over a fixture that admits it is one
(`rowstore.rs:5-8,33-39`), and its only address-shaped bytes are a flat
row index — `payload_lo64 = target = (row + offset) mod n_rows`
(`abi.md:536-544`) — with `lgj_hop` symmetric one-hop reachability, not
directional tree traversal. A repo-wide grep of the ABI and Java sources
for `parent|ancestor|hierarch` returns **zero** hierarchy hits (excluding
unrelated handle-lifetime "parent"). Even the live `FixtureClassView`
declines to override `is_a_parent`, whose contract hook *does* exist
(`class_view.rs:1001-1097`) but operates `ClassId → ClassId` and never
touches a row byte (`class_view_provider.rs:87-111`).

So the Java conformance task is concrete and small: **model the row as
`HHTL | edges | value`, and let `hop` take a direction and a rail** — at
which point ARC C's `parentOf`/`ancestorsOf` is a mask transform over
bytes that already exist, exactly as the mask-native invariant requires.

---

## §8 WHAT THIS DOES NOT DECIDE

The canonical-depth quantity (C5); whether `compute_cascade`'s
single-minimal-parent chain is the right mint for a multi-parent DAG; the
provenance mask's bit assignment (C6 says the vocabulary is absent, not
what it should be); the `ogar-from-dismech` crate's existence (an OGAR
call); and the EW64 tenant (§4 gates it, the operator rules it).
