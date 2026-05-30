# Cognitive RISC — Spec ↔ Code Reconciliation (v0.1)

> READ BY: integration-lead, truth-architect, ripple-architect, container-architect
> Status mix: FINDING (verified by file:line greps this session) unless tagged CONJECTURE.
> Date: 2026-05-30. Branch: `claude/hopeful-cori-R7szc`.
> Purpose: map the four `.claude/specs/cognitive-risc-*` + `faiss-homology-cam-pq` +
> `wikidata-hhtl-load` docs onto what ALREADY EXISTS in code, so the next build slice
> targets net-new work and reuses (does not re-propose) what is shipped.
> Method: 4 fan-out Explore agents + 3 main-thread verification greps. Two agent claims
> were CORRECTED on verification (noted inline) — do not trust the raw fan-out over this doc.

## TL;DR — five headlines

1. **The substrate's *cognition* half is real; its *identity/provenance* half is not.**
   Free-energy + Rubicon commit gate + two-tier EFE EXIST. CAM-as-identity, `class_id`,
   `discovery_origin`, and the Derived reasoning store DO NOT.
2. **`discovery_origin` is not in code at all** (0 `.rs` hits; 6 `.md` only). An in-flight
   ARM-discovery handover proposes a **2-bit** proposer field — *narrower* than the spec's
   already-flagged-too-narrow 3 bits. **This is freeze-move N2 being walked backwards.**
3. **"HHTL" is a same-name-different-semantics collision (a third meaning).** In code it is a
   *tensor/embedding quantization cascade* (`bgz-tensor`); the spec's HHTL is a *symbolic
   16^n entity-bucket router over P279 ancestry*. They share names (HEEL/HIP/TWIG/LEAF) and
   nothing else. Same class of trap as the documented dual-`CausalEdge64`.
4. **The facet/CAM_PQ codebook layer is ~60% scaffolded, not absent.** `OgitFamilyTable` +
   `FamilyEntry{OwlCharacteristics, DolceMarker, axiom_blob, dcterms:source}` + `OwlIdentity`
   (family u8 × slot u16 product code) exist; `PerFamilyCodebook` is an explicit *named placeholder*
   for the lossless per-facet CAM-PQ centroids (D-SDR-3c).
5. **Application pipelines split:** Odoo `op_emitter` codegen EXISTS but consumes *already-interpreted*
   recipes (no declarative-strata AstWalker); the Wikidata loader is **docs-only, zero code**.

## Layer-by-layer reconciliation (the FAISS-homology "closed picture")

| Spec layer | Spec intent (1 line) | In code today | Verdict | Net-new gap |
|---|---|---|---|---|
| **SoA substrate** | flat columnar, ID-encoded, `(start,len)` ragged backing | `cognitive-shader-driver/src/bindspace.rs` (`FingerprintColumns`/`EdgeColumn`/`QualiaI4Column`/`MetaWord`); `contract/src/container.rs:1` (`Container=[u64;256]`); `contract/src/witness_table.rs:96` (`WitnessTable<N>`); `MailboxSoA<N>` | **PARTIAL** | no `(start,len)` ragged/variable-length backing; no single named composite SoA |
| **LE byte contract** | explicit LE layout + `canonical_bytes` + runtime **version byte** | `canonical_bytes` EXISTS: `unified_audit.rs:176` (26 B), `unified_bridge.rs:141` (`OwlIdentity`→3 B), `lifecycle_audit.rs:72` (18 B). *(Corrects fan-out "ABSENT".)* | **PARTIAL** | no *uniform* contract; **no runtime version byte** — `causal-edge` uses compile-time feature gate only (`edge.rs:179`) |
| **CAM = identity** | BLAKE3/2-**128** over **sorted/canonicalized** symbolic atoms, exact, lossless | closest: `ndarray/src/hpc/merkle_tree.rs:79` blake3 **truncated to 48 bits** over fixed regions | **PARTIAL / weaker** | no 128-bit content hash; **no canonicalize-(sort)-before-hash** rule wired; 48-bit weakens birthday bound |
| **CAM_PQ = facet codes** | product of per-facet **closed-vocab** indices, **lossless** dict-encoding (NOT lossy PQ) | `family_table.rs`: `OgitFamilyTable` codebook + `FamilyEntry`+`OwlCharacteristics`(1 B bitfield, `:63`)+`DolceMarker`+`axiom_blob`; `OwlIdentity`=family×slot product code; `parse_family_registry` (`ttl_parse.rs:748`) hydrates TTL→table. `PerFamilyCodebook` (`:182`) = **named placeholder** for CAM-PQ centroids | **PARTIAL** | multi-facet *product*-of-bitmasks + presence accumulation absent; `PerFamilyCodebook` is empty stub (D-SDR-3c) |
| **HHTL bucket router** | symbolic 16^n nibble path over subClassOf(P279); arithmetic routing; mask inherits as delta | `bgz-tensor/src/hhtl_d.rs` (HEEL/HIP/TWIG weight-codec Slot D/V); `hhtl_cache.rs` (`RouteAction`) — **tensor cascade, not entity router** | **DIFFERENT-SEMANTICS** | symbolic entity-ancestry router ABSENT; family selection today is `FAMILY_TO_SUPER_DOMAIN` routing, not nibble paths |
| **lossy float-PQ (search only)** | legit ANN/PQ in *discovery* layer, never addressing | `ndarray/src/hpc/cam_pq.rs` (`CamCodebook` 6×256, `CamFingerprint` 6 B, ADC `DistanceTables`); `contract/src/cam.rs` (`CamCodecContract`/`CamStrategy`) | **EXISTS** | correctly separate from identity — keep it that way |
| **Class layer** | `class_id`/`shape_id` SoA column; shape-hash dedup; per-class presence bitmask | no class column in `bindspace.rs`; `arigraph/triplet_graph.rs` dedups by **string eq**, not shape hash | **ABSENT** | N1 + N3 unmet (see freeze table) |
| **Reasoning (Derived tier)** | pre-materialized DL closures (`subClassOf*`/`partOf*`), indexed by Subject/Object/axiom, `provenance=Derived` | `arigraph/triplet_graph.rs` runtime BFS only; single `entity_index: HashMap<String,Vec<usize>>`; `Triplet` has no provenance tier | **ABSENT** | whole Derived store + 3-way index + provenance tag net-new |
| **provenance / `discovery_origin`** | u8: tier(2) + proposer-id(≥3, widen!) + reserved | **0 `.rs` hits**; only specs/board/handovers | **ABSENT** | N2 — and in-flight proposal is *narrower* than spec |
| **Proposers + arbiter** | bounded non-recursive emit-k proposers (AstWalker/PairStats/Aerial+/LLM/dIPC); Rubicon EFE; two-tier free energy | arbiter EXISTS: `sigma-tier-router/src/lib.rs` (Σ10 Rubicon, "never commit on F-rising") + `contract/src/grammar/free_energy.rs` (`FreeEnergy`, `Resolution::{Commit,Epiphany,FailureTicket}`) | **arbiter EXISTS / proposers ABSENT** | no `Proposer` trait, no emit-k interface, no `<f,c>`+origin opaque-payload plumbing |

## Application surfaces

| Surface | In code | Verdict | Gap |
|---|---|---|---|
| **OGIT registry crate** | `OGIT/src/lib.rs` (`OgitTtl`, `TtlKind`, `ALL_TTLS`, `by_branch/by_kind`), `build.rs` `include_str!` | **EXISTS** | distribution mechanism, **not** a semantic engine — no transitive-closure / subsumption / disjointness reasoning |
| **DOLCE axis template** | `DolceMarker` (Endurant/Perdurant/Quality/Abstract) on every `FamilyEntry` | **EXISTS** | matches spec's "DOLCE defines the axes"; Wikidata-fill side unbuilt |
| **Odoo AstWalker** | `lance-graph-ontology/src/odoo_blueprint/op_emitter.rs` (`bucket_corpus`, `emit_op_dispatch`) — Phase-2 codegen over *interpreted* `OdooStyleRecipe` | **PARTIAL / DIFFERENT** | no declarative-strata lift (ORM domains / `ir.rule` / `@api.constrains`/`@api.depends`); op_emitter ≠ AST walker |
| **Odoo 3-hop shrink codebook** | `OGIT/NTO/Accounting/.../FiscalJurisdiction.ttl` (promoted attrs, shortcut verbs, indexed attrs, ISO-3166 closed vocab); commit `c5dc1b8` | **EXISTS (schema)** | HHTL/SoA bit-pack materialization absent |
| **Wikidata load pipeline** | `docs/WIKIDATA_HHTL_TILES.md` + `.claude/specs/wikidata-hhtl-load.md` only | **ABSENT (code)** | Pass-1 skeleton, Pass-2 bucket+AST+CAM, streaming, basin shards all net-new |

## Freeze-time non-negotiables (spec N1–N4) — status

| Move | Spec demand | Status | Note |
|---|---|---|---|
| **N1** | `class_id`/`shape_id` SoA discriminator column before freeze | **UNMET** | `OwlIdentity` (family×slot) is a *de-facto* identity at the OGIT bridge, but not wired as the cognitive-SoA discriminator column |
| **N2** | widen proposer-id (3 → 6 bits / u16) before WAL hardens | **UNMET + AT RISK** | type not built; in-flight ARM-discovery handover proposes **2 bits** — would bake the trap deeper |
| **N3** | stable append-only per-class bitmask bit-positions | **UNMET** | no presence bitmask exists yet; `OwlCharacteristics` shows the append-only+reserved-bit discipline as precedent |
| **N4** | don't freeze column ISA until ≥2 domains run through | **OK (by default)** | unified cognitive-SoA isn't frozen because it isn't built; chess-into-OWL bring-up test not yet run |

## Same-name-different-semantics collisions (the dual-CausalEdge64 class of trap)

- **HHTL** — (a) tensor weight-codec cascade `bgz-tensor` ; (b) attention route-cache `hhtl_cache.rs` ; (c) **spec's symbolic entity router (ABSENT)**. Three meanings, one name.
- **CAM** — identity hash (spec, partial via 48-bit merkle) **vs** CAM-PQ lossy search codec (`cam_pq.rs`, real). Spec is emphatic these must never be swapped; code keeps them separate today — preserve that.
- **"codebook"** — `OgitFamilyTable`/`PerFamilyCodebook` (symbolic, lossless intent) **vs** `CamCodebook` (k-means centroids, lossy). Different layers; don't merge.

## Recommended FIRST slice (and second)

**First slice — freeze-time contract moves N2 + N1 hook (small, zero-dep, highest regret if skipped):**
1. Land `discovery_origin` as a real `lance-graph-contract` type with the **widened** proposer-id
   (steal reserved → 6 bits, or go `u16`) — *before* the ARM-discovery work bakes 2 bits into the WAL grammar.
2. Add the `class_id`/`shape_id` SoA discriminator **column hook** to `bindspace.rs` (the N1 hook —
   inheritance machinery can stay deferred; the *column* must exist pre-freeze).
   Rationale: both are append-only byte-grammar decisions the spec classes "ruinous after WAL hardens,"
   both are pure-contract (no surrealkv/Lance dependency), and one is actively being walked backwards now.
   **Blocking decision for the user:** this collides with the in-flight ARM-discovery 2-bit proposal —
   reconcile width before either lands.

**Second slice — the spec's nominated falsifiable floor (`substrate WAL round-trip`):**
   write a SoA "thought" through surrealkv → commit with a materialized witness → read back after a
   simulated schema bump. Confirmed unstarted (no WAL-round-trip/epoch-reset test in `crates/`).
   Bigger: Agent-1 found surrealkv durability is *scaffolded* (`surrealdb/.claude/lance-backend`), so this
   slice also exercises that integration. Do it *after* the byte grammar is correct, not before.

## Confidence / caveats

- Verdicts are FINDING (grep-verified file:line) except: surrealkv WAL maturity (Agent-1, not re-verified
  here → **CONJECTURE**); "no canonicalize-before-hash anywhere" (proving a negative across all crates →
  **CONJECTURE**, high-confidence).
- This doc reconciles *structure*, not *behavior* — it locates code, it does not audit correctness.
