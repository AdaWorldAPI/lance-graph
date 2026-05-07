# Architecture Patterns — SoA/DTO Graph Traversal

> **READ FIRST** before proposing new types, modules, plans, harvest
> work, or "what's missing" analysis. This file teaches you to
> navigate the lance-graph workspace without producing duplicate
> work. The codebase is large and opinionated — most "interesting
> ideas" are already implemented somewhere. Use these patterns to
> find them before reinventing.
>
> Authored 2026-05-06 by a session that walked through ~10 rounds of
> "is X new?" → "no, X exists at Y" before finally building the map.
> The patterns here are the cost of that walk, captured so the next
> session pays a tax of one read instead of ten rounds.
>
> Companion to:
> - `.claude/knowledge/soa-dto-fma-map.md` — 8-region map (R0-R8)
> - `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` — per-component scoring (OPEN active concerns)
> - `.claude/board/ARCHITECTURE_ENTROPY_LEDGER_RESOLVED.md` — closures archive
> - `.claude/board/SINGLE_BINARY_TOPOLOGY.md` — three-layer invariants
> - `Cargo.toml` workspace.members — canonical crate inventory

---

## TL;DR — Five traversal patterns

| # | Pattern | Action |
|---|---|---|
| **P-1** | **CRATE-FIRST** | `cat Cargo.toml \| grep -E '^\s+"crates/'` BEFORE proposing |
| **P-2** | **REGION-FIRST** | Name your concept's R-id from soa-dto-fma-map; if none fits, that's the review question |
| **P-3** | **ENTROPY-FIRST** | Sort ledger Section A by Entropy DESC; pick highest-leverage cleanup |
| **P-4** | **APPEND-ONLY** | Board files (PR_ARC, INTEGRATION_PLANS, EPIPHANIES, TECH_DEBT, ISSUES, ENTROPY_LEDGER, STATUS_BOARD) — never edit prior entries; only append dated sections |
| **P-5** | **CLUSTER-AWARE** | When fixing a ledger row, check Section B Spaghetti Cluster — siblings need to move together |

---

## The architecture as a graph

The workspace is a directed graph where:

- **Nodes** are (a) regions R0-R8 from the SoA-DTO map, (b) crates, (c) modules within crates.
- **Edges** are (a) data flow (X feeds Y), (b) re-export (`pub use`), (c) trait impl (`impl X for Y`), (d) feature gate.
- **Edge weights** are the entropy scores in the ledger — high-entropy edges are conflict zones (parallel duplicates, namespace clashes, stale stubs).

The R0-R8 region partition is the **structural backbone**. Every concept in this codebase belongs to exactly one region. If you can't place a concept in a region, you're either looking at a workspace-level concern (board hygiene, governance) or genuinely-novel work that needs a new region.

```
                          R8 Lab surface (Wire DTOs / serve / grpc)
                          (cognitive-shader-driver/{wire,serve,grpc,…}.rs)
                                    │ decode once at REST edge
                                    ▼
R3 Cross-domain orchestration ←── R4 External boundary (BBB) ────┐
(contract::orchestration:           (contract::external_membrane  │
  StepDomain, UnifiedStep,           callcenter::lance_membrane)  │
  OrchestrationBridge)                       │                    │
       │                                     │ project()          │
       ▼                                     ▼                    │
R0 BindSpace SoA ←──── R1 Per-cycle DTOs (Φ Ψ B Γ) ←──── R5 ──────┘
(driver::bindspace:    (contract::cognitive_shader:    Carrier algebra
 8 columns, 71777B/row)  ShaderDispatch/Resonance/      (Vsa16kF32 ℝ +
       │                 Bus/Crystal)                   Binary16K GF(2))
       │                       │                              │
       │                       ▼                              ▼
       │              R2 Write airgap                R6 Domain catalogues
       │              (contract::collapse_gate:       (contract::{grammar/
       │               GateDecision, MergeMode)        role_keys,thinking,
       │                       │                       persona,ontology,cam})
       └─── written ←──────────┘                              │
                                                              ▼
                                   R7 Storage (Lance MVCC, audit, drift)
                                   (lance-graph::graph::versioned, audit,
                                    transcode::parallelbetrieb)
```

**Tokio boundary** sits below R7's outbound edge, between Layer 2 and Layer 3 of the Single-Binary Topology. Tokio appears ONLY past `CycleAccumulator` flush. R0–R6 are sync; R7 has tokio inside Lance's internal runtime; R8 has tokio for HTTP/gRPC serving.

---

## Crate inventory — canonical at 2026-05-06

The workspace has **~22 crates**. Listing them ALL because every "is this novel?" question collapses against this list.

### Lance-graph workspace (~16 crates)

| Crate | Region(s) | What it owns |
|---|---|---|
| **`lance-graph-contract`** | R1, R2, R3, R4, R5, R6 | Zero-dep type surface: `cognitive_shader`, `collapse_gate`, `cycle_accumulator`, `crystal`, `cam`, `external_membrane`, `grammar/*`, `nars`, `orchestration`, `persona`, `splat`, `thinking`, … 38+ modules |
| **`lance-graph`** | R3, R6, R7 | Main query engine: `parser::parse_cypher_query` (1932 LOC), `graph::{spo,arigraph,versioned}`, `cam_pq/{ivf,jitson_kernel,storage,udf}`, `nsm/{encoder,parser,similarity,tokenizer,nsm_word}` |
| **`lance-graph-planner`** | R3 | Pipeline DAG, executor, MUL gate, NARS algebra, thinking-style dispatch, orchestration_impl |
| **`lance-graph-callcenter`** | R4, R7, R8 | `LanceMembrane`, `audit::{InMemoryAuditSink,LanceAuditSink}`, `policy::{ColumnMaskRewriter,RowEncryptionPolicy,DifferentialPrivacyPolicy,NotYetWiredHashUdf}`, `postgrest`, `rls`, `version_watcher` (sync, std-only post-WATCHER-1) |
| **`lance-graph-rbac`** | R4, R6 | `Policy`, `Role`, `Operation`, `AccessDecision`, `PermissionSpec`, `smb_policy()` |
| **`lance-graph-archetype`** | R3 | Archetype ECS bridge (per ADR 0001) |
| **`lance-graph-catalog`** | R6, R7 | Catalog management |
| **`lance-graph-cognitive`** | R6 | Grammar Triangle (`GrammarTriangle::from_text`), older 630K-LOC cognitive layer |
| **`cognitive-shader-driver`** | R0, R1, R8 | `BindSpace`, `CognitiveShaderDriver` impl, Wire DTOs, `engine_bridge`, `cypher_bridge` (regex stub), `codec_research` |
| **`deepnsm`** | R5, R6 | NSM crate: `codebook`, `context`, `disambiguator_glue`, `encoder`, `fingerprint16k`, `markov_bundle`, `nsm_primes`, `parser` (30 KB), `pipeline`, `pos`, `quantum_mode`, `similarity`, `spo`, `ticket_emit`, `trajectory`, `trajectory_audit`, `triangle_bridge`, `vocabulary` + 12 grammar-style YAMLs + 5 word-frequency CSVs |
| **`holograph`** (workspace-EXCLUDED) | R5, R6 | Crystal stack: `sentence_crystal`, `crystal_dejavu`, `dntree`, `dn_sparse` (116 KB!), `navigator` (64 KB), `mindmap`, `epiphany`, `hamming`, `bitpack`, `hdr_cascade`, `nntree`, `neural_tree`, `representation`, `resonance`, `rl_ops`, `slot_encoding`, `storage`, `storage_transport`, `ffi`, `query/`, `width_10k/16k/32k/`, `graphblas/` |
| **`bgz-tensor`** | R5 | BGZ17 codec implementation + HHTL_D doc + data + examples |
| **`highheelbgz`** | R5 | High-Heel BGZ precision tier |
| **`reader-lm`** | R6 | Reader-LM model: `classifier`, `inference`, `tokenizer`, `weights` |
| **`jc/`** | (jc) | Justified Concentration prover: 7+ pillars (Cartan, Precond, EWA-sandwich, etc.) + examples (`prove_it`, `osint_edge_traversal`) |
| **`thinking-engine`** | R6 (drift) | Older parallel ThinkingStyle (12-variant) — superseded by contract-36 but still wired |
| **`neural-debug`** | (debug) | Neural debugging |
| **`learning`** | R6 | Reinforcement-learning style selector (`StyleSelector` bandit) |
| **`causal-edge`** | R6 | `CausalEdge64` Pearl 2³ masks, `PackedTruth`, inference type table |

### Consumer-side crates (in separate repos, in MCP allowlist)

| Repo / Crate | Layer | What it owns |
|---|---|---|
| `medcare-rs/crates/medcare-rbac` | L2 | Medcare policy (4 roles × 6 entities, BMV-Ä §57 invariants) |
| `medcare-rs/crates/medcare-realtime` | L2 | `MedCareStack`, `MedCareMembraneGate` (POLICY-1 medcare-side closure) |
| `medcare-rs/crates/medcare-{server,db,core,analytics,pdf}` | L2/L3 | medcare consumer surface |
| `smb-office-rs/crates/smb-realtime` | L2 | `SmbStack`, `SmbMembraneGate` (POLICY-1 smb-side closure, PR #29) |
| `smb-office-rs/crates/smb-{office-bin,db,core,analytics}` | L2 | smb consumer surface |
| `q2/crates/cockpit-server` + `cockpit/` | L3 | Q2 Gotham-equivalent UI; SSE serving on canonical R1 surface (PR #35) |

### Out-of-scope / external

| Repo | Status | Purpose |
|---|---|---|
| `MedCareV2` | OUT-OF-MCP | C# .NET 4.8 desktop probe; calls `/api/__parity/csharp` |
| `ladybug-rs` | EARLIER PROTOTYPE | Predecessor to lance-graph workspace; ALL grammar/crystal/NSM/CAM material already migrated |
| `aiwar-neo4j-harvest` | DATA | Cypher/Neo4j OSINT harvest (q2 cockpit reads from this) |

---

## Equivalence map: ladybug-rs ↔ lance-graph

If a session is asked to harvest from ladybug-rs, the answer is **already done** unless explicitly proven otherwise. Discovered 2026-05-06 across ~10 rounds:

| ladybug-rs/src/spo/ | Lance-graph equivalent | How to verify |
|---|---|---|
| `nsm_substrate` | `crates/deepnsm/{codebook,fingerprint16k,encoder}.rs` | `cargo doc -p deepnsm` |
| `sentence_crystal` | `crates/holograph/src/sentence_crystal.rs` (27 KB) | grep workspace |
| `context_crystal` | `lance-graph-contract::grammar::context_chain` + `deepnsm::context` | mod tree |
| `spo` (the 5^5 crystal) | `lance-graph-contract::crystal::Structured5x5` + `holograph::representation` | contract crate src |
| `spo_harvest` (238× cosine) | `bgz-tensor` + `highheelbgz` + `holograph::{hamming,bitpack,hdr_cascade}` + `ndarray::hpc::cascade` | grep `cosine` in ndarray |
| `causal_trajectory` | `deepnsm::{trajectory,trajectory_audit,triangle_bridge}.rs` | deepnsm/src tree |
| `gestalt`, `meta_resonance` | `deepnsm::ticket_emit` + `holograph::resonance` + `deepnsm::similarity` | grep |
| `nsm_primes` | `deepnsm::nsm_primes` | exact match |
| **`clam_path`** | **`crates/lance-graph/src/cam_pq/`** (5 files) | NOT obvious from name; `clam` ≅ `cam` |
| `crystal_lm` | `crates/reader-lm/src/{classifier,inference,tokenizer,weights}.rs` | reader-lm crate |
| `codebook_*` | `deepnsm::codebook` + `bgz-tensor` | grep |
| `deepnsm_integration` | The deepnsm crate IS this | exists by name |
| **DN-tree ↔ crystal binding** | **`holograph::{dntree,dn_sparse,navigator}.rs`** (~214 KB combined!) | holograph src |

**Verdict:** ladybug-rs/src/spo is a SUBSET of what's in the lance-graph workspace. ladybug-rs is the earlier prototype; nothing remains to harvest.

---

## Anti-patterns observed in this session

### 1. The Discovery Loop

Pattern: session proposes work → user says "X already exists at Y" → session proposes different work → user says "Y also exists" → repeat.

**Cost:** ~10 rounds in this session. Each round consumed both user attention and session context budget.

**Cause:** session didn't run CRATE-FIRST + REGION-FIRST traversal before proposing.

**Cure:** before ANY proposal involving "implement X" or "harvest X from external repo", traverse:
1. `Cargo.toml` workspace.members
2. `soa-dto-fma-map.md` regions
3. `ARCHITECTURE_ENTROPY_LEDGER.md` rows
4. `LATEST_STATE.md` Contract Inventory
5. MCP `search_code` for likely names

### 2. Harvest-from-Stale

Pattern: session treats an external repo (ladybug-rs) as canonical truth and proposes harvest plans against material that already migrated.

**Cause:** session didn't realize the workspace had absorbed the prototype crate's content.

**Cure:** for any external-repo harvest, the FIRST move is comparing the external repo's module list against the workspace crate inventory above.

### 3. Map-Blindness

Pattern: session knows about `lance-graph-contract` and `lance-graph` but misses crates like `holograph`, `bgz-tensor`, `highheelbgz`, `reader-lm`, `lance-graph/src/cam_pq` because they don't surface in obvious search paths.

**Cause:** the workspace has ~22 crates but session focused on the 4-5 obvious ones.

**Cure:** ALWAYS read the full `Cargo.toml` workspace.members list at session start. The crates here are the universe.

### 4. Single-Name Lookup

Pattern: session searches for `clam_path` and finds nothing, declares "novel". User says "clam_path ≅ cam_pq".

**Cause:** session didn't try alternate spellings / domain-equivalent names.

**Cure:** when searching for a concept by name, ALSO search for plausible aliases:
- `clam_path` → `cam_pq` (same acronym, different layout)
- `nsm_substrate` → `deepnsm::codebook`
- `sentence_crystal` → `holograph::sentence_crystal`
- `context_crystal` → `grammar/context_chain`
- `spo_harvest` → `bgz-tensor` / `cascade`
- `gestalt` → `ticket_emit` / `resonance`

The pattern: **acronym match > exact-name match > domain-keyword match**.

### 5. Plan-Doc-Without-Code-Check

Pattern: session writes a plan PR proposing "wire X to Y" without first verifying that X and Y aren't already wired.

**Cause:** plan-writing is cheaper than code-reading; session shortcuts.

**Cure:** before writing a plan with deliverable D-ids, READ the actual files involved. The plan should cite `file:line` for every "currently broken" claim.

---

## Pre-work checklist

Run these BEFORE proposing any new type, module, plan, harvest, or wiring:

```
[ ] Read Cargo.toml workspace.members (canonical crate inventory)
[ ] Find concept's region (R0-R8) in soa-dto-fma-map.md; if none fits → review question
[ ] Search the entropy ledger for an existing row touching this concept
[ ] grep workspace for the concept name AND plausible aliases
[ ] Check LATEST_STATE.md Recently Shipped (last 7 days of PRs)
[ ] If proposing harvest from external repo: list workspace crates by domain
    (NSM=deepnsm+lance-graph/src/nsm; Crystal=holograph+contract; …)
[ ] Cite file:line in any "currently broken" claim
[ ] Check whether the concept's row in the ledger is part of a Section B cluster
    — if so, name the cluster + the suggested-order in the proposal
```

If any checklist box reveals existing work, DO NOT propose new work. Either:
- Update the existing row's status (state change) per ledger Update Protocol
- Cite the existing as the canonical and propose ONLY the missing edge
- Stop and ask the user

---

## Wiring recipes (concrete examples)

These are existing seams from the entropy ledger. Each is wireable using ONLY existing primitives — no new types, no duplicates. Future sessions can pick one and execute.

### Recipe A — CAM-DIST-1: register `cam_distance` UDF globally

**State:** UDF registered at `cam_pq/udf.rs:241,257,326`. Called from `query.rs:470` ONLY when `with_cam_codebook(...)` is opted in. `datafusion_planner/mod.rs::new()` does NOT register, so default Cypher path can't reference `cam_distance`.

**Wire:** add the UDF registration in `DataFusionPlanner::new` so it's always available. Pure additive change. Closes CAM-DIST-1 (entropy 3 → 2).

```rust
// In datafusion_planner/mod.rs::DataFusionPlanner::new
let mut state = SessionState::new_with_config_rt(config, runtime);
state = lance_graph::cam_pq::udf::register_cam_distance(state); // NEW LINE
```

**No duplication:** uses existing `register_cam_distance` from `cam_pq/udf.rs`.

### Recipe B — PARSER-1: wire `cypher_parse::plan` to real parser

**State:** Real parser at `lance-graph::parser::parse_cypher_query` (1932 LOC nom). Stub at `planner::strategy::cypher_parse.rs` (72 LOC substring matching). cypher_bridge.rs uses regex.

**Wire:** replace `CypherParse::plan` body with a call to the real parser, return its AST, route via existing visitor. Closes PARSER-1 (entropy 5 → 3 once first stub is wired).

```rust
// planner::strategy::cypher_parse::CypherParse::plan
fn plan(&self, query: &str) -> PlanResult {
    let ast = lance_graph::parser::parse_cypher_query(query)?; // EXISTING fn
    self.from_ast(&ast)
}
```

**No duplication:** uses existing `parse_cypher_query`. The 35 `format!("{:?}", logical_plan)` Debug-stringify sites become eliminable in a second-pass cleanup once the typed AST flows.

### Recipe C — DEEPNSM-NSM-1: collapse `lance-graph/src/nsm/`

**State:** `lance-graph/src/nsm/{encoder,parser,similarity,tokenizer,nsm_word}.rs` (≈2,405 LOC) parallels `crates/deepnsm/`. CLAUDE.md Phase-3 task "Consolidate nsm/ module" never ran.

**Wire:** delete `lance-graph/src/nsm/`; replace with thin re-export shim:

```rust
// lance-graph/src/nsm/mod.rs (new shim)
pub use deepnsm::encoder;
pub use deepnsm::parser;
pub use deepnsm::similarity;
pub use deepnsm::vocabulary as tokenizer;
```

**No duplication:** keeps the deepnsm canonical; gives `lance-graph::nsm::*` callers the same path.

### Recipe D — VSA-1: methods on Vsa16kF32 (the FMA)

**State:** 8 free functions (`vsa16k_bind`, `vsa16k_bundle`, etc.) on a type alias. Click P-1 violation: free function = reject, method = accept.

**Wire:** newtype `Vsa16kF32` + `impl { fn bind, fn bundle, fn cosine, fn permute }`. Existing free fns become trivial method delegators OR get deleted with their bodies absorbed into the methods.

**No duplication:** the algebra is the same; only the call shape changes.

### Recipe E — MEMBRANE-GATE-1: removed (already done)

State changed to **Wired** by:
- SMB side: PR #29 (`SmbMembraneGate` over `Arc<lance_graph_rbac::Policy>`)
- Medcare side: PR #98 (`MedCareMembraneGate` over `Arc<medcare_rbac::Policy>`)

Ledger row needs the state-change append.

### Recipe F — WATCHER-1: removed (already done)

State changed to **Wired (sync, std-only)** by PR #337 — `LanceVersionWatcher` rewritten to `std::sync::{Arc,RwLock,Mutex,Condvar}` per topology I-2. Ledger row needs the state-change append.

---

## Ledger update protocol (canonical, restated for visibility)

When a row's state changes:
1. Append a new dated entry below the current snapshot. Reference the row ID.
2. Do NOT edit the original snapshot table — append-only.
3. New section: `## YYYY-MM-DD — <ID> resolution / state change` with old-state, new-state, evidence (PR#, file:line).

When a NEW SoA / DTO / bridge enters:
1. Append `## YYYY-MM-DD — <ID> introduction`.
2. Score Region / DupCount / DupPotential / LooseEnds / Plan / PlanStatus / Entropy.
3. If part of an existing cluster, cite the cluster ID.

---

## Cross-references

- `.claude/knowledge/soa-dto-fma-map.md` — the 8-region structural map
- `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` — per-component scoring + clusters (OPEN active)
- `.claude/board/ARCHITECTURE_ENTROPY_LEDGER_RESOLVED.md` — closures archive
- `.claude/board/SINGLE_BINARY_TOPOLOGY.md` — three-layer invariants (I-1/I-2/I-3/I-4)
- `.claude/board/INTEGRATION_PLANS.md` — versioned plan index
- `.claude/board/STATUS_BOARD.md` — deliverable-level status
- `.claude/board/LATEST_STATE.md` — current-state snapshot
- `.claude/board/PR_ARC_INVENTORY.md` — per-PR decision history
- `.claude/board/CROSS_REPO_PRS.md` — external-repo PRs that touch this workspace
- `Cargo.toml` (workspace root) — `members = [...]` is the canonical crate list
- `CLAUDE.md` — Click P-1 (carrier-method-or-reject), three iron rules

## Maintenance

This file is not append-only — it's a living usability guide. Edit when:
1. New crate enters the workspace (extend Crate Inventory)
2. New region added to soa-dto-fma-map (extend graph diagram)
3. New systematic anti-pattern observed (add to Anti-Patterns)
4. New wiring recipe identified (add to Recipes)

When editing, preserve the TL;DR table at the top — that's the load-bearing entry point.
