# Patterns — SoA/DTO Graph Traversal Primer

> **Read by:** every session before touching the SoA / DTO surface.
> **Status:** APPEND-ONLY governance (same rules as
> `ARCHITECTURE_ENTROPY_LEDGER.md`, `EPIPHANIES.md`, `TECH_DEBT.md`).
> **Companion to:** `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md`
> (the scored inventory), `.claude/knowledge/soa-dto-fma-map.md`
> (the producer→consumer map), `CLAUDE.md` (the iron rules).

## Why this file exists

This workspace's SoA/DTO surface is a graph: **41+ scored nodes,
multiple producer→consumer edges per node, 7 cluster-subgraphs of
entangled duplicates, and named seams between regions**. Every
session that touches it pays a re-discovery tax: which copy is
canonical, which carrier method exists vs which free function is
floating, which DTO maps to which agent capability, where the
write airgap is soft-enforced vs hard-typed.

This file is the **navigation primer**. Read it before traversing.
The 15 patterns below are usability primitives that turn the
"frustrating wilderness" into a "navigable city" — they encode the
shortest path from "I want to do X" to "I'm editing the right line
of the right file."

The patterns are NOT prescriptive style rules. They are
**graph-traversal heuristics with a cost model** — each one
identifies a recurring decision point, names the cheap path, and
calls out the expensive anti-pattern.

---

## The graph

### Nodes
Each node is a **type definition** (struct, enum, trait,
type-alias, const) that carries cognitive or governance state.
Naming follows: `crate::module::TypeName`. Each node has a row
in `ARCHITECTURE_ENTROPY_LEDGER.md` § Section A scored on
Maturity (1-4), State (Wired/Stub/Aspirational/Dead),
Smart/Dumb (Click P-1 lens), Duplicate Count, Loose Ends,
Plan/Status, Entropy (1-5), Deficit→Genius.

### Edges
Three edge classes:
1. **Producer → Consumer** (`lance-graph-contract::ThinkingStyle`
   → `cognitive-shader-driver::engine_bridge::UNIFIED_STYLES`).
   Read off `soa-dto-fma-map.md` § per-region tables.
2. **Duplicate** (same logical concept, ×N type definitions).
   Examples: `NarsInference` ×6, `ThinkingStyle` ×4+const+bandit,
   `MulAssessment` ×4. Tracked in ledger DupCount column.
3. **Seam** (cross-region single-instance unfused FMA point —
   `multiplicand × multiplicand + addend` SHOULD chain but
   doesn't). Tracked in ledger § Section C. Examples: R4
   `MembraneGate` ↔ R6 `rbac::Policy`; R7 `LanceMembrane`
   projection cold-path.

### Subgraphs (clusters)
Sets of duplicate edges that are mutually entangled — fixing one
without the others breaks the cluster. Named in ledger §
Section B:
- **NARS** (NARS-1, TRUTH-1, NARS-TRUTH-1, MUL-ASSESS-1)
- **Thinking** (THINK-1, COMPASS-1, TRUST-1, FLOW-1, MUL-ASSESS-1, ADJ-THINK-1)
- **VSA carrier** (VSA-1, PERMUTE-1, CONTENT-FP-1, ROLEKEY-OPS-1, CRYSTAL-1)
- **Parser** (PARSER-1, DEBUG-STRINGIFY-1)
- **Foundry seal** (POLICY-1, MEMBRANE-GATE-1, SEAL-1, WATCHER-1, PROJECT-LANCE-1)
- **HEEL ladder** (HEEL-1, CAM-DIST-1, DNTREE-1)
- **Board hygiene** (CONTRACT-INV-1, PLAN-INDEX-1, AGENT-LOG-1, PR-ARC-1, LATEST-RECENT-1, STATUS-CODEC-1)

### Routing
Sessions traverse the graph for one of four queries:
- **Locate**: "where is X defined / canonical?"
- **Score**: "is this row Stage-3-stalled or Stage-4-genius?"
- **Compose**: "to do Y, which producer + which consumer + which seam?"
- **Resolve**: "this row's deficit → genius is what?"

The patterns below are answers to these queries.

---

## Patterns

Each pattern: **signal** (when this fires), **action** (the cheap
path), **anti-pattern** (the expensive path to avoid),
**citation** (where the discipline comes from).

### P-CANON — Find the canonical before you grep

**Signal:** "I need the canonical type for X" (ThinkingStyle,
NarsInference, TruthValue, GateDecision, etc.).

**Action — order matters:**
1. `.claude/board/LATEST_STATE.md` § Current Contract Inventory.
2. `.claude/knowledge/soa-dto-fma-map.md` per-region producer table.
3. `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` § Section A row.
4. **Then** grep `crates/lance-graph-contract/src/`.
5. Last resort: grep workspace-wide.

**Anti-pattern:** Grep first → land on the most prominent or most
recently-edited copy → propose a parallel implementation. The
ledger has 6 NARS-inference enums, 4 ThinkingStyle definitions,
3 TruthValue copies precisely because every session that grepped
first added another one.

**Citation:** CLAUDE.md § "Consult before you guess".

### P-MATURITY — Score before touching

**Signal:** About to modify a type or add a new one.

**Action:** Read the row in `ARCHITECTURE_ENTROPY_LEDGER.md`
§ Section A. The Maturity stage tells you the move:
- **Stage 4 (Only Source of Truth):** don't touch unless adding
  a method on the existing carrier. Adding a parallel struct =
  sabotage of a finished surface.
- **Stage 3 stalled:** fix Smart/Dumb (carrier methods) before
  adding new functionality. The "stalled" usually IS the
  Click-P-1 violation.
- **Stage 2 (Wired, no canonical):** pick the canonical first
  (the one in `lance-graph-contract`), migrate consumers, THEN
  add functionality. Otherwise you're a 7th NARS copy.
- **Stage 1 (Stub):** fill the body; do NOT add a parallel
  implementation in another crate.
- **Stage 0 (Aspirational):** confirm the plan exists and is
  Active in `INTEGRATION_PLANS.md` before implementing. If the
  plan is Stalled or Missing, that is the deliverable.

**Anti-pattern:** Modify a Stage-4 row by adding a wrapper.
Modify a Stage-2 row by adding the 7th copy. Implement a Stage-0
row whose plan isn't even indexed.

**Citation:** Section F.1 of the entropy ledger; this session's
correction-log.

### P-CLICK-P1 — Method on carrier > free function

**Signal:** About to write `fn foo(state: &Carrier) -> X` or
`pub fn process(c: &mut Carrier, ...)`.

**Action:** Make it a method:
```rust
impl Carrier { fn foo(&self) -> X }
```
**Litmus:** "free function = reject, method = accept" (CLAUDE.md
"The Click", P-1).

**Anti-pattern:** `Vsa16kF32` is the canonical ledger example —
its algebra is currently 8 free functions (`vsa16k_bind`,
`vsa16k_bundle`, `vsa16k_cosine`, etc.) on a `Box<[f32; 16_384]>`
type alias. That is the **single biggest deficit-vs-genius gap
in the workspace** (VSA-1, Stage-3-stalled-Dumb).

**Citation:** CLAUDE.md § "The Click" P-1; `EPIPHANIES.md`
E-OBJECT-SPEAKS-FOR-ITSELF.

### P-REGISTER — HashMap before VSA (register laziness check)

**Signal:** About to bundle something into a `Vsa16kF32`.

**Action — the four tests, IN ORDER, any "no" short-circuits:**
- **Test 0 (register laziness):** Does this thing have a natural
  name / ID / enum variant? If yes → `HashMap`, `enum`, SQL key,
  Lance column. NOT VSA.
- **Test 1 (bundle size):** Is N ≤ √d / 4 ≈ 32 at 16K dim? If
  no → use direct lookup; superposition SNR is below threshold.
- **Test 2 (role orthogonality):** Are role keys mutually
  orthogonal (disjoint slice OR orthogonal bipolar)? If no →
  unbind doesn't recover cleanly.
- **Test 3 (cleanup codebook):** Is there a known codebook to
  match against after unbind? If no → raw bundle inspection is
  unreliable.

**Vsa16kF32 is for ONE job:** Markov chain over identity
fingerprints (per-cycle cognitive state, role-keyed content,
position-braided). Provenance, JWT claims, RBAC roles,
transform IDs, branch IDs — all register territory.

**Anti-pattern (this session, retracted):** "VSA-bundled
algebraic provenance" (E4) — register laziness; provenance is
struct/HashMap/SQL.

**Citation:** CLAUDE.md § I-VSA-IDENTITIES iron rule;
`vsa-switchboard-architecture.md` decision matrix.

### P-PERMUTE-NOT-LOSSLESS — Position-braiding has cross-talk

**Signal:** Reaching for `vsa16k_permute` for "lossless"
position-shifting.

**Action:** The operation is unitary; **the braided bundle is
not lossless**. Position-shifted copies in a bundle have
cross-talk that shrinks the unbinding margin with N. Treat
permute as **SNR-bounded position-braiding for Markov ρ^d**,
bound by N ≤ √d/4 like every other VSA bundle.

**Anti-pattern:** Using permute outside Markov ρ^d to braid
arbitrary content (e.g., chains of unrelated facts) and
expecting clean unbind.

**Citation:** CLAUDE.md § "The Click" footnote; this session's
VSA scope correction.

### P-DUAL-TIER — Business writes vs observability writes

**Signal:** About to add a write path. Decide which tier.

**Action — two tiers, two fail policies:**
- **Business-tier:** synchronous, must succeed under policy,
  propagates `Result`, gates on `MembraneGate::admit`. Goes
  through `BindSpace.apply(GateDecision, MergeMode, RowDelta)`
  (E1 seam).
- **Observability-tier:** async fire-and-forget, fail-soft,
  in-tier-only logging. Goes through `LanceAuditSink::record`,
  `CognitiveEventLanceSink::project`, `LanceProbe::probe`. Trait
  signature returns `()` not `Result`. Internal degraded-mode
  ring buffer for sink-down case.

**Reference implementation in the wild:** MedCareV2 PR #8
`AuthClient.LoginAsync` (fire-and-forget, 5s timeout, never
throws, never blocks UI; failure is acceptable).

**Anti-pattern:** Audit log in the same transaction as the
action (Foundry's design — sluggish UI under sink failure).

**Citation:** EPIPHANY E9 (this session); MedCareV2 #8 doctrine
comment.

### P-INGESTION-COMMIT — Two ingestion modes, one Action API

**Signal:** About to add an ingestion path (parser, splat,
import, stream, scrape).

**Action:** Both modes converge on **`BindSpace.apply()`**:
```
Cypher-parser ingestion: parse → AST → RowDelta → apply()
Splat-deposit ingestion: witness → splat → RowDelta → apply()
```
The Action API is the universal commit gate. Ingestion-mode
just changes how `RowDelta` is constructed. They are not
competing architectures — they are two `RowDelta` constructors
feeding the same E1 seam.

**Anti-pattern:** Ingestion path that bypasses the typed
Action API and writes to `BindSpace` columns directly. Looks
shorter; breaks `MembraneGate::admit`, `LanceAuditSink::record`,
`CommitFilter::validate` invariants.

**Citation:** EPIPHANY E1 (this session); analysis from sibling
session re: PR #289 + SPLAT-1 ingestion.

### P-LINEAGE-IS-COLUMN — Don't query lineage; read the row

**Signal:** "How do I retrieve the lineage of this value?"

**Action:** AGI-as-SoA doctrine: lineage is a **typed column on
the same row**, not a side-graph. `CognitiveEventRow` carries
`{produced_by, inputs, branch, subject, timestamp}` as struct
fields. Single column read; not a graph walk.

**Anti-pattern:** "Lineage is a graph; we'll bolt on Neo4j /
add a separate lineage service / put it in a Vsa16kF32."
Foundry-pattern register laziness.

**Citation:** EPIPHANY E3 (this session, corrected); CLAUDE.md
§ "AriGraph is thinking tissue, not storage".

### P-APPEND-ONLY — Read before Write on board files

**Signal:** About to modify any of:
`LATEST_STATE.md`, `PR_ARC_INVENTORY.md`, `EPIPHANIES.md`,
`TECH_DEBT.md`, `ISSUES.md`, `IDEAS.md`,
`ARCHITECTURE_ENTROPY_LEDGER.md`, `INTEGRATION_PLANS.md`,
`AGENT_LOG.md`.

**Action:**
1. **Read first** (`Read` tool, NOT just memory).
2. New entries: **prepend** the dated block (most-recent-first
   convention) OR **append** with a date header.
3. **Mutable fields ONLY:** `Status:` and `Confidence:` lines;
   corrections append as new dated lines, never edit the
   original.
4. **Tool selection:** `Edit` for status flips on existing rows;
   `tee -a` (heredoc) for new sections; **`Write` only on first
   creation** of a brand-new file.

**Diagnostic signature of failure:** `git diff --stat` shows
`~N insertions / ~N deletions` on a file of size N — same
magnitude, virtually every line different — means the file was
*regenerated from prompt* instead of *built from state*.
Revert with `git restore <path>`.

**Anti-pattern (this session, repeated twice):** `Write`
overwriting an own prior `Write` without `Read` between them.
The ledger lost its Maturity / Smart-Dumb / Deficit columns
once via this. Recovery: `tee -a` append of a Section F band-
aid, then a clean rewrite via `Write` *after* `Read`.

**Citation:** CLAUDE.md § "In-Session Orchestration Discipline";
upstream regression
[anthropics/claude-code#46861](https://github.com/anthropics/claude-code/issues/46861).

### P-CONSULT-FIRST — Agent → knowledge → board → grep

**Signal:** Need information about a domain or surface.

**Action — the order:**
1. **Specialist agent card** in `.claude/agents/*.md` —
   19 specialists + 5 meta-agents. Domain → agent table at
   `.claude/agents/BOOT.md` § Knowledge Activation.
2. **Knowledge doc** in `.claude/knowledge/*.md` with `READ BY:`
   header. `encoding-ecosystem.md`, `lab-vs-canonical-surface.md`,
   `vsa-switchboard-architecture.md`, etc.
3. **Board file** — `LATEST_STATE.md`, `ENTROPY_LEDGER.md`,
   `PR_ARC_INVENTORY.md`.
4. **Grep source** — last resort.

**Anti-pattern:** Hand-exploration via `Glob`/`Grep` first.
Burns turns, finds stale or duplicated definitions, misses
the curated synthesis. Subagent spawn (Opus for accumulation)
that bootloads the curated docs first is almost always cheaper
than a grep session on the main thread.

**Citation:** CLAUDE.md § "Consult before you guess";
`.claude/agents/BOOT.md`.

### P-CROSS-SESSION-BLACKBOARD — Ledger row IDs are the protocol

**Signal:** Cross-repo dependency or coordination question.

**Action:** Cite **ledger row IDs** (NARS-1, THINK-1, POLICY-1,
SPLAT-1) as the unit of cross-repo blocking. Sibling sessions
read the ledger; the ledger is the SoA-DTO graph traversal map
for cross-repo work.

**Reference implementations (this session):**
- q2 PR #35 cited THINK-1 + TRUTH-1 as the rows it resolved.
- MedCareV2 PR #8 implicitly waits on POLICY-1 + MEMBRANE-GATE-1.

**Resolution format (single APPEND-only commit per resolution
event):**
```
## YYYY-MM-DD — <ROW-ID> resolution / state change
- Old state: <Maturity, S/D, Entropy>
- New state: <Maturity, S/D, Entropy>
- Evidence: PR# / file:line
- Cluster downstream: <other rows that just got cheaper to fix>
```

**Anti-pattern:** Coordinating via ad-hoc Slack/Notion threads
or per-session email-style handovers. Lossy, not queryable, not
session-replayable.

**Citation:** CLAUDE.md § Layer-2 A2A; this session's q2 +
MedCareV2 cross-references.

### P-DUAL-SOURCE-CHECK — Source vs claim divergence

**Signal:** A board file claims a feature is shipped or a
deliverable is complete.

**Action:** Verify source against claim:
1. Read the claim (e.g. "PR #243 shipped `content_fp.rs`").
2. Grep `crates/` for the cited symbol or file.
3. If absent or stub: **divergence = `TECH_DEBT` entry**, not a
   "trust the board" pass.
4. Update the relevant ledger row from "Wired" to "Aspirational"
   or "Stub" with citation.

**Reference (this session):** the 2026-05-05 audit found four
divergences (PERMUTE-1 missing, CONTENT-FP-1 missing,
ROLEKEY-OPS-1 removed but board still advertises, AGENT_LOG.md
referenced by CLAUDE.md but does not exist).

**Anti-pattern:** Treat `LATEST_STATE.md` as ground truth without
audit. Boards drift; only source compiles.

**Citation:** This session's audit pass; `TECHNICAL_DEBT_SIGNED_SESSION.md`
("56% useful, 44% bypass — honest review") set the precedent.

### P-CLUSTER-FIX — Fix the cluster, not the row

**Signal:** About to resolve a single ledger row that belongs to
a named cluster.

**Action:** Read `ARCHITECTURE_ENTROPY_LEDGER.md` § Section B.
If the row is a cluster member, plan the fix at cluster scope:
- **NARS cluster:** collapse `NarsInference` first (canonical:
  `contract::grammar::inference::NarsInference` 7-variant) →
  then `TruthValue` → then `NarsTruth`. Order matters; doing
  TruthValue first leaves NarsInference still 6-way.
- **Thinking cluster:** adopt contract-36 ThinkingStyle as
  canonical → migrate planner-12 + engine-12 + engine-5 +
  bandit → drop `UNIFIED_STYLES[12]` const → THEN unlock
  `ThinkingAdjacency` (ADJ-THINK-1).
- **VSA carrier cluster:** add `vsa16k_permute` first (Markov
  ρ^d unblocks D5) → then methods on `Vsa16kF32` (Click-P-1) →
  then content_fp on the f32 carrier → THEN re-introduce
  RoleKey ops cleanly.

**Anti-pattern:** Fix one row, ship, claim victory; later
session discovers the cluster is still entangled because the
other 4 copies still exist and now contradict the canonical.

**Citation:** Ledger § Section B; q2 PR #35 followed
the Thinking-cluster order correctly (canonical first, deps
dropped second).

### P-DEBUG-AS-API-IS-DEBT — `format!("{:?}", x)` is debt

**Signal:** Reaching for `format!("{:?}", logical_plan)` (or
similar Debug-string introspection) to extract structure.

**Action:** Each occurrence is a `TECH_DEBT.md` entry. The
correct surface is a **typed visitor** (e.g., DataFusion's
`TreeNode` trait) over the `LogicalPlan` enum variants.

**Reference (DEBUG-STRINGIFY-1, entropy 5):** 35 sites in this
workspace read DataFusion `LogicalPlan` Debug output as a
"stable" surface. `lance_native_planner.rs:76-79` runs
`to_uppercase().contains("MATCH")` against pretty-printed
Rust struct syntax. Brittle; breaks on rustc upgrades; never
documents the schema.

**Anti-pattern:** "Debug output is good enough; we'll fix it
later." It compounds — every consumer that parses the Debug
string locks the format harder.

**Citation:** Ledger DEBUG-STRINGIFY-1 row.

### P-SCOPE-LOCK — One carrier, one job

**Signal:** Tempted to use a successful carrier (Vsa16kF32,
BindSpace, OrchestrationBridge, MembraneGate) for a second
purpose because "it would fit."

**Action:** Re-read the carrier's iron-rule scope:
- `Vsa16kF32` → Markov ±5 cognitive state. Not provenance.
  Not RBAC. Not JWT claims.
- `BindSpace` → SoA columns of cognitive identity per row.
  Not application data; not configuration.
- `OrchestrationBridge` → cross-domain step composition. Not
  generic routing; not message-passing.
- `MembraneGate` → admit/deny on (Subject, Resource, Op). Not
  business-logic gating; not feature flags.

If the new use violates the scope, **build a new typed register
in the right region** instead. The architecture's genius is in
each carrier's scope being narrow.

**Anti-pattern (this session, retracted):** "VSA-bundled
provenance / EWA-Sandwich-bounded lineage" — two attempts to
overload the cognitive carrier with non-cognitive jobs. Caught
by the user's correction.

**Citation:** CLAUDE.md § Substrate-level iron rules
(I-SUBSTRATE-MARKOV / I-NOISE-FLOOR-JIRAK / I-VSA-IDENTITIES);
this session's VSA scope correction.

### P-SEAM-NAMING — Name the seam, not the layer

**Signal:** About to propose architectural work scoped at a
"layer" or "service" level.

**Action:** Identify the **single seam** (cross-region
single-instance unfused FMA point) that the work closes. Name
the seam. Scope the PR to closing that seam exclusively.

Seams are listed in ledger § Section C. Examples:
- E1 = `BindSpace.apply()` typed Action API. Closes 5 ledger
  rows (SEAL-1, WATCHER-1, PROJECT-LANCE-1, MEMBRANE-GATE-1,
  POLICY-1). One seam, one PR.
- E2 = `impl MembraneGate for Arc<rbac::Policy>`. ~30 LOC.
  One seam, one PR.
- E3 = `CognitiveEventLanceSink` mirror of `LanceAuditSink`.
  ~150 LOC. One seam, one PR.

**Anti-pattern:** "Phase 4: Foundry-Parity" / "Layer 7: Auth"
/ "Service: Lineage" — layer-scoped work is a mile wide and
two inches deep, never lands cleanly, and the seam in the
middle is what actually mattered.

**Citation:** Ledger § Section C; EPIPHANIES E1-E7 (this session).

---

## Critical findings — 2026-05-05/06 session

These are the ground truths that surfaced this session. They
are **descriptive history**, not prescriptions; future sessions
should treat them as evidence the patterns above are real.

1. **VSA scope creep is the workspace's most repeated mistake.**
   Three rows in the entropy ledger document attempts to overload
   `Vsa16kF32` with non-Markov jobs (CONTENT-FP-1's
   `from_centroid_semantic` violates I-VSA-IDENTITIES; my own
   E4/E8 retraction this session; two earlier reverts in
   `crystal/role_keys.rs`). The pattern is robust enough to need
   its own iron rule, which CLAUDE.md added on 2026-04-21.

2. **Cross-session blackboard works as designed.** q2 PR #35
   (merged 09:16 UTC 2026-05-06) cited THINK-1 + TRUTH-1 ledger
   rows hours after the ledger was published to main. MedCareV2
   PRs #7 (09:18 UTC) and #8 (09:32 UTC) implicitly chain on
   POLICY-1 + MEMBRANE-GATE-1. The ledger is becoming the
   single coordination surface other repos block on.

3. **Three axes turn out to be what's useful for ranking.**
   Maturity (1-4) + Smart/Dumb (Click P-1 lens) + Deficit→Genius
   (descriptive next step). Entropy (1-5) ranks badness;
   the three axes rank *next move*. Section F of the ledger
   carries them.

4. **Foundry parity is mostly seam-closing, not new layers.**
   E1 (typed Action API) closes 5 ledger rows in a single seam.
   E2 (one impl) is ~30 LOC. E3 (mirror Sink) is ~150 LOC.
   The substrate is mostly there; the gaps are seams.

5. **Two ingestion modes converge on E1.** Cypher-parser
   ingestion (Option 1, ships now) and splat-deposit ingestion
   (Option 2, blocks on SPLAT-1) are not competing architectures
   — they are two `RowDelta` constructors feeding the same
   typed Action API.

6. **EWA-Sandwich (PR #289) is for cognitive Markov state, not
   lineage.** Pillar-6 σ-sandwich bounds propagation of
   cognitive `Vsa16kF32` across ρ^d cycles. It is NOT a
   provenance-graph error model. (This session's correction.)

7. **The SoA-DTO surface IS a graph.** The framing in this file
   is not a metaphor — it is the literal data structure (nodes
   = type defs, edges = producer-consumer + duplicate + seam,
   subgraphs = clusters). Once you treat it as such, the
   patterns above are graph-traversal heuristics with cost
   models, not opinions.

---

## Update protocol

This file is APPEND-ONLY. To add a pattern:

1. **Append** a new dated section: `## YYYY-MM-DD — <P-NAME>
   pattern addition`. Include signal / action / anti-pattern /
   citation following the template above.
2. **Do NOT edit the patterns list above** in this initial
   2026-05-06 snapshot. New patterns prepend to a "Section:
   Patterns added later" block, dated.
3. **Promote** an "added later" pattern into the canonical list
   only via a documented session capstone — never silently.

To correct an existing pattern:

1. Append `## YYYY-MM-DD — <P-NAME> correction`.
2. Restate the corrected pattern.
3. Cite the failure mode that triggered the correction.
4. Cross-reference: leave the original pattern intact; the
   correction lives below as new history.

To retire a pattern:

1. Append `## YYYY-MM-DD — <P-NAME> retired`.
2. Cite the structural change that obsoleted it (e.g. "P-CANON
   step 3 is now step 1 because the entropy ledger is now the
   primary entry point, supersedes this session's ordering").
3. Old pattern stays in the file as history.

Cross-references:
- `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` — the scored
  inventory each pattern references.
- `.claude/knowledge/soa-dto-fma-map.md` — the producer→consumer
  map P-CANON step 2 reads.
- `CLAUDE.md` — iron rules each pattern cites
  (Click P-1 / I-VSA-IDENTITIES / In-Session Orchestration
  Discipline / etc.).
- `.claude/agents/BOOT.md` — agent registry P-CONSULT-FIRST
  step 1 references.
- `EPIPHANIES.md` — companion narrative source for E1-E9
  cited by P-INGESTION-COMMIT, P-LINEAGE-IS-COLUMN, P-DUAL-TIER.
