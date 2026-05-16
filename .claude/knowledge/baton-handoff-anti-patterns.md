# Baton-Handoff Anti-Pattern Catalogue

> **READ BY:** `baton-handoff-auditor` (canonical); also `PP-14
> convergence-architect` when evaluating whether a LATENT drop is a
> missing alignment opportunity; `PP-16 preflight-drift-auditor` when
> plan-level drift maps to a known BAP; main-thread orchestrator when
> triaging a CSI entry from a sprint meta-review.
>
> **Status:** FINDING (every BAP below was observed at least once in
> the sprint-11, sprint-12, or sprint-13 preflight CSI ledger; grep
> targets verified against the working tree at HEAD = `04620aa`).
>
> **Predecessors:**
> - `.claude/board/sprint-log-11/meta-review-opus.md` CSI-7/8/9/18
> - `.claude/board/sprint-log-12/meta-review.md` CSI-7..18
> - `.claude/board/sprint-log-13/preflight-meta-review-opus.md`
>   CSI-19..23
> - `.claude/knowledge/codex-p1-anti-patterns.md` (PP-13 sibling)
> - `CLAUDE.md §Substrate-level iron rules` (I-VSA-IDENTITIES,
>   I-LEGACY-API-FEATURE-GATED)
>
> **Promotion track:** when a BAP fires across N ≥ 3 sprints, follow
> the ceremony in `.claude/knowledge/iron-rules-doctrine.md` §3 to
> promote it to an iron-rule candidate. The canonical precedent is
> BAP9 → `I-LEGACY-API-FEATURE-GATED` (sprint-11 CSI-18 promotion).

---

## §1. What the baton-handoff-auditor sees — the eight boundary classes

Each class describes what "the baton" looks like and why it can be
dropped cleanly, caught cleanly, or lost without either party noticing.

### B1 DTO↔DTO

The baton is a struct or enum instance passed from a producer module to
a consumer module. The classic drop: the producer's type evolves (field
renamed, field type changed from `Vsa16kF32` to `[u64; 256]`, Option
wrapper added) without a corresponding change in the consumer. Because
Rust is strongly typed, OBVIOUS mismatches are compile errors — but
SILENT drops happen when the types are both "correct" (both `f32` slice
types) but encode semantically different things (identity fingerprint vs
content fingerprint). The baton-handoff-auditor checks field-shape
round-trip: can the consumer reconstruct what the producer encoded?

### B2 Crate↔Crate

The baton is a re-export chain: the producer crate authors a type,
`lance-graph-contract` re-exports it, the consumer crate imports it.
The classic drop: the type is authored but never appears in the
intermediate re-export (`lib.rs` orphan), so the consumer crate sees a
"type not found" compile error. More subtle: the type IS re-exported but
with a DIFFERENT name than the consumer's import path expects (CSI-15
rename drift). The auditor checks the full re-export chain: producer
file → lib.rs pub mod → contract re-export → consumer import.

### B3 Sprint↔Sprint

The baton is a TD-* (tech debt) or CSI-* (cross-spec inconsistency)
entry emitted by sprint-N and consumed by sprint-(N+1)'s spec as if
resolved. The classic drop: the sprint-N worker logs a CSI entry as
OPEN with a suggested fix; the sprint-(N+1) planner reads the entry and
assumes it is resolved because the worker's last commit landed on the
branch. But the sibling repo (`ndarray`, `crewai-rust`, `n8n-rs`) never
received the corresponding PR — the baton sat on the floor of the branch
and nobody picked it up on master. The auditor checks: for every CSI
cited as RESOLVED in a sprint-N+1 spec, verify that resolution against
the target repo's `git log master` — not the branch.

### B4 Planner↔Worker

The baton is a D-CSV-* or OQ-CSV-* identifier referenced in a worker
prompt. The classic drop: two planners independently chose the same ID
(CSI-19) or one planner referenced an ID that another planner later
renumbered (CSI-19's three-error sentence in PP-3 §0). A worker reading
the wrong D-CSV-* finds a different deliverable than the planner
intended — or no deliverable at all. The auditor checks: for every D-id
and OQ-id in a worker prompt, verify it against the canonical ID
assignment in the coordination-authority planner output (PP-1 v3 in
sprint-13) and the OQ catalog (PP-11).

### B5 REST↔Canonical

The baton is a `UnifiedStep` instance routed through
`OrchestrationBridge`. The classic drop: a developer sees the Axum REST
server in `cognitive-shader-driver/src/serve.rs` and adds a new
`/v1/<thing>` endpoint that has its own dispatch logic — bypassing the
canonical `OrchestrationBridge` trait entirely. The handler fires; the
baton never reaches the bridge; the canonical consumer surface grows
stale. The auditor checks: every new route handler constructs a
`UnifiedStep` and delegates to a bridge impl, or the route is explicitly
marked `LAB-ONLY` under a `serve`/`grpc` feature gate. See
`.claude/knowledge/lab-vs-canonical-surface.md` Decision Procedure.

### B6 Lib.rs/Mod.rs Orphan

The baton is a `pub mod <name>;` declaration. The classic drop: a worker
creates `crates/<crate>/src/mymodule.rs` but the matching `pub mod
mymodule;` line never appears in `lib.rs`. The file compiles in
isolation; the worker self-reports DONE. But `cargo test -p <crate>`
never runs the module's tests, and downstream importers cannot reach the
module's types. The auditor checks: for every new `.rs` file added under
`crates/<crate>/src/`, its stem appears in that crate's `lib.rs` (or
the parent `mod.rs` for sub-module files).

### B7 Cargo.toml Workspace

The baton is the workspace membership of a new crate. The classic drop:
a worker authors `crates/<new>/Cargo.toml` with a `[workspace]` table
at the top — standard boilerplate copy-paste — not realizing that a sub-
crate's `[workspace]` declaration OVERRIDES the parent workspace and
silently excludes the crate from `cargo metadata`. The parent workspace
never sees the crate; `cargo -p <new>` works only with `--manifest-path`
awkwardness. The auditor checks: no sub-crate `Cargo.toml` intended as a
workspace member declares its own `[workspace]` table.

### B8 Cross-Repo

The baton is a commit SHA, PR merge status, or file registration in a
sibling repo (`/home/user/ndarray`, `/home/user/crewai-rust`,
`/home/user/n8n-rs`). The classic drop: a planner reads the sibling
repo's local branch state and reports it as merged to master. A
subsequent sprint planner reads this report and treats the work as
resolved — but `git log master` on the sibling repo shows no such merge.
The baton is on a branch floor, not in master. The auditor checks: every
sibling-repo claim is verified against `git log master --oneline` of
THAT repo, not against the local branch or the PR title.

---

## §2. Boundary anti-pattern catalogue BAP1..BAP10

Each entry follows the same shape: **Name | Boundary class | Symptom |
Grep target | The rule | Fix pattern | Workspace instances | Promotion
track**.

---

### BAP1 — DTO Field-Shape Silent Drift

**Boundary class:** B1 (DTO↔DTO)

**Symptom:** Producer emits a type that encodes meaning in one format
(e.g., real-valued VSA carrier `Vsa16kF32 = Box<[f32; 16_384]>`) while
the consumer expects a structurally compatible but semantically
incompatible type (e.g., `Binary16K = [u64; 256]`, a Hamming-compare
format). Both types are "fingerprint-shaped"; neither the compiler nor
the linker complains. Decode produces garbage or silent precision loss.

**Grep target:**

```bash
# Find fingerprint type uses at cross-crate boundaries
grep -rn "Vsa16kF32\|Binary16K\|\[u64; 256\]\|\[f32; 16_384\]" \
     crates/*/src/ --include="*.rs" | grep -v "#\[test\]"
# Find mismatched uses in the same function or struct
grep -n "ShaderDispatch\|ShaderHit" crates/*/src/*.rs \
  | grep "fingerprint"
```

**The rule:** when a type crosses a crate boundary, the producer and
consumer must agree on BOTH the Rust type (structural match) AND the
encoding semantics (VSA identity vs CAM-PQ content vs Hamming binary).
A comment like `// fingerprint: Vsa16kF32 semantics` at the boundary
point is mandatory when the type is `Box<[f32; 16_384]>` used in more
than one encoding role.

**Fix pattern:** add a newtype wrapper at the boundary:

```rust
/// Identity fingerprint (Vsa16kF32 semantics — real-valued multiply-add).
/// NEVER interpret as Binary16K (Hamming compare) or CAM-PQ (content index).
pub struct IdentityFingerprint(pub Vsa16kF32);

/// Hamming-compare fingerprint (Binary16K semantics — popcount distance).
pub struct HammingFingerprint(pub Binary16K);
```

The newtype forces a conversion at the boundary and makes the encoding
mismatch a compile error rather than a silent semantic drift.

**Workspace instances:** CLAUDE.md correction note (2026-04-21): "earlier
session posted a version claiming 'XOR on `[u64; 157]`' — Frankenstein
confusion between Binary16K (Hamming-compare format, `[u64; 256]`) and
the actual VSA carrier (real-valued multiply+add)." D5 Frankenstein
revert, commit `0ae9f90`. I-VSA-IDENTITIES iron rule directly addresses
this boundary class.

**Promotion track:** if a third sprint instance of field-shape silent
drift at a carrier boundary appears, promote to iron-rule candidate
"I-DTO-ENCODING-IDENTITY" under the data-semantics axis.

---

### BAP2 — Rename-Without-Downstream-Sweep

**Boundary class:** B1 (DTO↔DTO) + B4 (Planner↔Worker)

**Symptom:** A type is renamed in a PR branch (or in a planner spec)
but the old name persists in downstream artifacts: worker prompts, sibling
crate imports, knowledge docs, sprint specs, or BOOT.md trigger rows.
Workers spawn against the old name, cannot resolve the type, and stall.

**Grep target:**

```bash
# Find old name usages after a rename:
grep -rn "CamPqIndexPlaceholder" crates/ .claude/ --include="*.rs" \
  --include="*.md"
# After a rename to WitnessIndexHashMap — any stale references?
grep -rn "WitnessIndexHashMap\|CamPqIndexPlaceholder" \
     crates/ .claude/ | sort
```

**The rule:** any PR that renames a public type MUST include a grep
sweep of ALL dependent artifacts: crate imports, worker prompts, agent
cards, knowledge docs, sprint specs. The rename commit is not done until
the sweep is ZERO-result for the old name in non-deprecated contexts.

**Fix pattern:**

1. Run `grep -rn "<old-name>" .` at workspace root.
2. Update every hit: crate imports, worker prompt cites, knowledge doc
   mentions, agent card references.
3. Add a `#[deprecated(note = "renamed to <new-name> in PR #NNN")]`
   type alias for the old name in the crate where it was defined.
4. Log the rename in the board: `.claude/board/LATEST_STATE.md` contract
   inventory update in the same commit.

**Workspace instances:** CSI-15 (sprint-13 preflight) — `CamPqIndexPlaceholder`
renamed to `WitnessIndexHashMap` on PR #390 branch; downstream worker
prompts (PP-3, PP-5) cited the old name until the preflight meta-review
caught the drift. Verified: `grep CamPqIndexPlaceholder crates/lance-graph/src/graph/arigraph/witness_corpus.rs` still returns the old name on main (PR #390 not yet merged as of sprint-13 preflight HEAD `04620aa`).

**Promotion track:** N = 1 sprint-13 confirmed instance. Watch for CSI
entries citing rename drift in sprint-14+.

---

### BAP3 — Lib.rs / Mod.rs Orphan

**Boundary class:** B6 (Lib.rs/Mod.rs orphan)

**Symptom:** A worker creates `crates/<crate>/src/<module>.rs` but
never adds `pub mod <module>;` to `crates/<crate>/src/lib.rs`. The file
compiles standalone when referenced via `--manifest-path`; the worker
self-reports DONE. Downstream consumers cannot import the module;
`cargo test -p <crate>` never runs the module's tests. The orphan is
invisible until a consumer fails to resolve the type.

**Grep target:**

```bash
# New .rs files in any crate src/ directory:
git diff --diff-filter=A --name-only main..HEAD \
  | grep -E '^crates/[^/]+/src/[^/]+\.rs$'

# For each, verify lib.rs registration (substitute <crate> and <stem>):
grep "pub mod <stem>" crates/<crate>/src/lib.rs \
  || echo "ORPHAN: crates/<crate>/src/<stem>.rs not registered"
```

**The rule:** every new `crates/<C>/src/<M>.rs` MUST have a matching
`pub mod <M>;` line in `crates/<C>/src/lib.rs` in the SAME commit.
No orphan modules. If the module is nested under a parent module, the
registration goes in the parent `mod.rs`, not `lib.rs` — but the chain
must be complete from `lib.rs` down.

**Fix pattern:** add the `pub mod <M>;` line to `lib.rs` (or the
appropriate `mod.rs`) in alphabetical order within the existing `pub mod`
block. For a private module, use `mod <M>;`. For a crate-internal
module that shouldn't be re-exported, still register it — just omit the
`pub`.

**Workspace instances:**
- CSI-8 (sprint-12) — `attention_mask.rs` + `attention_mask_actor.rs`
  in `cognitive-shader-driver/src/` were orphaned; caught by W-Meta-Opus
  as CSI-8; fixed in aggregation commit `d4e5bbc` (PR #389).
- CSI-9 (sprint-12) — `qualia.rs` + `splat_field.rs` in
  `/home/user/ndarray/src/hpc/stream/` were orphaned in the cross-repo
  sibling; fixed locally (commit `2a1a1e38`) but NOT on ndarray master as
  of sprint-13 preflight. CSI-9 status: OPEN / HARD BLOCKER on D-CSV-11
  productization. Verified: `cd /home/user/ndarray && git log master --oneline | head -5` shows master HEAD `2a3885d2` (PR #146), no stream module registration.

**Promotion track:** N = 2 sprints (sprint-12 CSI-8 + CSI-9). If a
third sprint instance appears, promote to iron-rule candidate
"I-MODULE-REGISTRATION" under the API-version axis. Worker-template-v2
§5.1 (PP-8) codifies the fix discipline as worker responsibility.

---

### BAP4 — Cargo.toml Workspace Self-Declaration Trap

**Boundary class:** B7 (Cargo.toml workspace)

**Symptom:** A new `crates/<new>/Cargo.toml` contains a `[workspace]`
table at the top. This makes the sub-crate an INDEPENDENT workspace
rooted at its own directory, silently excluding it from the parent
workspace. `cargo metadata --no-deps` run at workspace root does not
list the crate; `cargo test -p <new>` fails without `--manifest-path`.
The bug is invisible during `cargo check` because the sub-workspace
builds fine standalone.

**Grep target:**

```bash
# Any [workspace] table in sub-crate Cargo.toml files:
grep -rn '^\[workspace\]' crates/*/Cargo.toml

# Cross-reference against parent workspace members:
grep 'members' Cargo.toml | head -5
```

**The rule:** sub-crate `Cargo.toml` MUST NOT declare `[workspace]`
when the crate is intended as a workspace member. The ONLY exceptions
are standalone crates explicitly listed in the parent's `[workspace]
exclude =` (e.g., `bgz17`, `deepnsm`, `bgz-tensor`).

**Fix pattern:**

1. Remove the `[workspace]` table from `crates/<new>/Cargo.toml`.
2. Add `"crates/<new>"` to the parent `Cargo.toml` `[workspace]
   members = [...]` list.
3. Verify with `cargo metadata --no-deps | jq '.workspace_members'`.

**Workspace instances:** CSI-7 (sprint-12) — `crates/sigma-tier-router/Cargo.toml`
declared `[workspace]` in PR #388 (W-F1); caught by W-Meta-Opus as CSI-7;
fixed in W-G6 + aggregation commit `d4e5bbc` (PR #389). Codified as AP3
in `codex-p1-anti-patterns.md` (PP-13 sibling).

**Promotion track:** N = 1 (sigma-tier-router). The `workspace
member/exclude consistency` pattern is in `iron-rules-doctrine.md §5.5`
candidate watch list at N = 2 (sigma-tier-router + one prior). Track to
N = 3.

---

### BAP5 — REST Endpoint Added Without OrchestrationBridge Route

**Boundary class:** B5 (REST↔Canonical)

**Symptom:** A new `/v1/<thing>` Axum route is added in
`cognitive-shader-driver/src/serve.rs` (or `grpc.rs`) with its own
dispatch logic — pattern-matching on a request field, calling a function
directly, returning a response. The `OrchestrationBridge` trait is
bypassed. Production consumers who walk `UnifiedStep` via the bridge
cannot reach the new capability. The REST handler is a dead end for
anyone not using the lab surface.

**Grep target:**

```bash
# New route additions in diff:
git diff main..HEAD -- '*.rs' \
  | grep -E '^\+.*\.route\("/v1/'

# New Wire DTO types in diff:
git diff main..HEAD -- '*.rs' \
  | grep -E '^\+.*pub struct Wire[A-Z]'

# Any new handler that doesn't construct a UnifiedStep:
grep -n "UnifiedStep" crates/cognitive-shader-driver/src/serve.rs
```

**The rule:** every new `/v1/<thing>` handler MUST either:
(a) construct a `UnifiedStep` and route it through the canonical bridge
    (`StepDomain` + `OrchestrationBridge::handle_step`), OR
(b) be explicitly marked `LAB-ONLY` under the `serve`/`grpc` feature
    gate in a `codec_research.rs` or `wire.rs` module.

A handler with its own dispatch logic that is not behind a lab feature
gate is a P0 boundary violation. See the Decision Procedure in
`.claude/knowledge/lab-vs-canonical-surface.md`.

**Fix pattern:**

```rust
// WRONG: handler with own dispatch logic
async fn handle_new_thing(Json(req): Json<WireNewThing>) -> Json<WireResult> {
    let result = do_the_thing(req.param);
    Json(WireResult { value: result })
}

// RIGHT: handler constructs UnifiedStep and routes through bridge
async fn handle_new_thing(
    State(bridge): State<Arc<dyn OrchestrationBridge>>,
    Json(req): Json<WireNewThing>,
) -> Json<WireResult> {
    let step = UnifiedStep {
        step_type: "new_thing".into(),
        domain: StepDomain::Research,
        payload: serde_json::to_value(&req)?,
    };
    let result = bridge.handle_step(step).await?;
    Json(WireResult::from(result))
}
```

**Workspace instances:** prevented by the lab-vs-canonical-surface.md
doctrine + AP8 in `codex-p1-anti-patterns.md`. No live instances in
sprints 11-13 preflight; the doctrine exists because the pattern was
anticipated from the REST endpoint structure in `serve.rs`.

**Promotion track:** zero observed instances (doctrine-prevented). If
sprint-13+ impl workers add a route without bridge routing, this becomes
the first sprint-13 BAP5 instance.

---

### BAP6 — Sprint-N TD-* Placeholder Consumed as Resolved by Sprint-(N+1)

**Boundary class:** B3 (Sprint↔Sprint) + B8 (Cross-Repo)

**Symptom:** Sprint-N logs a CSI-* or TD-* entry as OPEN but with a
"resolution underway" note. Sprint-(N+1)'s planner reads the note and
treats the work as RESOLVED — citing the resolution commit SHA or the
PR merge date. The cited commit is real BUT exists only on a feature
branch, not on the sibling repo's master. The sprint-(N+1) impl worker
inherits a foundation that doesn't exist on master.

**Grep target:**

```bash
# Find resolution claims in sprint specs:
grep -rn "CSI-9\|ndarray-PR-#147\|2a1a1e3\|qualia_stream\|splat_field" \
     .claude/ --include="*.md"

# Verify against sibling repo master:
cd /home/user/ndarray && git log master --oneline | grep -i "stream\|qualia\|splat"
# If empty: the resolution is on a branch, not master
```

**The rule:** a CSI-* or TD-* entry is RESOLVED only when:
(a) the fix commit SHA is on the TARGET repo's master branch, AND
(b) `git log master --contains <SHA>` returns a non-empty result for
    the target repo (not just the lance-graph branch).

A branch-local fix that has not been merged to master is OPEN regardless
of how long ago the commit was authored.

**Fix pattern:**

1. In the sprint-N+1 spec, change "RESOLVED" to "PARTIALLY RESOLVED
   (branch only — cross-repo PR pending)."
2. Log a blocking TD-* entry: "Requires ndarray PR #N merge to master
   before D-CSV-11 productization can begin."
3. Add the blocker to the spawn-readiness checklist so the pre-spawn
   aggregation step catches it.

**Workspace instances:**
- CSI-9 (sprint-12 + sprint-13 preflight) — `qualia.rs` +
  `splat_field.rs` not registered in `/home/user/ndarray/src/hpc/stream/mod.rs`
  on master. PP-1 v3 §0.1 and §13.8 cited "Risk REDUCED via d4e5bbc"
  (a lance-graph aggregation commit); PP-12 cross-repo audit falsely
  reported "ndarray PR #147 merged 2026-05-16T04:35:05Z." Verified at
  sprint-13 preflight: ndarray master HEAD `2a3885d2` (PR #146), no
  stream module registration. This is a HARD BLOCKER on D-CSV-11.
- CSI-20 (sprint-13 preflight) — multiple planners cited
  I-LEGACY-API-FEATURE-GATED as a canonical iron rule, but `grep
  I-LEGACY-API-FEATURE-GATED CLAUDE.md` returns empty on main; the
  promotion is on PR #390 branch only.

**Promotion track:** N = 2 sprints (sprint-12 CSI-9 + sprint-13 CSI-20).
If a third sprint shows the same pattern (branch-state consumed as
master-state), promote to iron-rule candidate
"I-CROSS-REPO-GROUND-TRUTH" under a new "state-verification" axis.

---

### BAP7 — D-id / OQ-id Numbering Collision Across Parallel Planners

**Boundary class:** B4 (Planner↔Worker)

**Symptom:** Two or more planners independently choose the same D-CSV-*
or OQ-CSV-* identifier for different deliverables or open questions.
Worker prompts spawned from conflicting planner outputs reference the
same ID for different work items — workers either duplicate effort, skip
work, or build on wrong foundations.

**Grep target:**

```bash
# Find all D-CSV-* assignments in specs:
grep -rn "D-CSV-[0-9]\+" .claude/specs/ .claude/plans/ .claude/board/ \
  --include="*.md" | sort -t'-' -k3 -n

# Find all OQ-CSV-* assignments:
grep -rn "OQ-CSV-[0-9]\+" .claude/ --include="*.md" \
  | sort -t'-' -k3 -n | uniq -d -f2
```

**The rule:** D-CSV-* and OQ-CSV-* IDs are allocated by the COORDINATION-
AUTHORITY planner (PP-1 plan vN in this workspace). Sub-planners MUST
NOT choose IDs independently — they draft the deliverable content and
propose a human-readable name; the coordination planner assigns the
number. When parallel planners run simultaneously (as in Wave H), the
numbering coordinator step runs AFTER all planners and BEFORE the meta-
review aggregation commit.

**Fix pattern:**

1. Identify the canonical ID assignment table (PP-1 v3 §11 D-CSV table
   + PP-11 OQ catalog in sprint-13).
2. Produce a conflict list: IDs with two or more assignments.
3. Resolve by: keeping the coordination-authority assignment and updating
   the sub-planner's cross-reference text.
4. Pre-spawn aggregation commit: update the mis-assigned cross-references
   (~3-15 LOC typically, per CSI-19 analysis).

**Workspace instances:** CSI-19 (sprint-13 preflight) — PP-3 §0 cites
"D-CSV-16 slot reserved by PP-2 (sprint-13 splat on-Think method
migration)": three errors in one sentence (PP-2 = iron-rules-doctrine,
not splat; PP-4 = splat on-Think; D-CSV-16 = CAM-PQ per PP-5). PP-1 v3
§12 OQ table lists OQ-CSV-7..12 (6 entries); PP-11 OQ catalog lists
OQ-CSV-7..19 (13 entries). Each planner chose IDs independently without
a coordinator.

**Promotion track:** N = 1 sprint (sprint-13 preflight). Watch for
sprint-14 multi-planner wave. If IDs collide again in sprint-14, promote
to mandatory process gate: the coordination-authority planner must lock
the ID table before any sub-planner spawns.

---

### BAP8 — Iron-Rule Violation at Carrier-Catalogue Boundary

**Boundary class:** B1 (DTO↔DTO)

**Symptom:** A Layer-2 role catalogue (grammar/role_keys.rs, persona
role_keys, callcenter role_keys) is handed a content fingerprint — a
CAM-PQ code, a quantized index, or a sign-binarized embedding — instead
of an identity fingerprint (`Vsa16kF32` bipolar ±1 in a disjoint slice).
The bundle at the carrier boundary loses the codebook mapping (the
"register-loss problem" per I-VSA-IDENTITIES). Unbind no longer recovers
the original role content; cosine similarity against the codebook
produces garbage.

**Grep target:**

```bash
# Find vsa_bundle calls with non-identity inputs:
grep -n "vsa_bundle\|vsa_bind" crates/*/src/*.rs \
  | grep -v "role_keys\|identity_fp\|from_catalogue"

# Find CAM-PQ codes near bundle calls (the violation pattern):
grep -B5 -A5 "vsa_bundle" crates/*/src/*.rs \
  | grep "cam_pq\|pq_code\|quantized\|compress"
```

**The rule:** `vsa_bundle` and `vsa_bind` take IDENTITY fingerprints
only. An identity fingerprint is one produced by a role-key catalogue
(disjoint `[start:end)` slice, bipolar ±1 in that slice, zero elsewhere).
If the input is a CAM-PQ code, a quantized embedding, or any other
content-derived bit pattern, it violates I-VSA-IDENTITIES and the
bundle is semantically invalid.

The four tests before reaching for VSA (from CLAUDE.md §I-VSA-IDENTITIES):
- Test 0 — register laziness: use HashMap if exact-match suffices
- Test 1 — bundle size: N ≤ √d / 4 ≈ 32 items at 16K dim
- Test 2 — role orthogonality: disjoint slices or orthogonal bipolar
- Test 3 — cleanup codebook: known codebook to match against after unbind

**Fix pattern:**

```rust
// WRONG: bundle a CAM-PQ code (content, not identity)
let bundled = vsa_bundle(&cam_pq_code, &role_key);

// RIGHT: bundle the IDENTITY fingerprint that POINTS TO the content
let entity_identity = catalogue.lookup_identity(entity_id)?;
let bundled = vsa_bundle(&entity_identity, &role_key);
// Content lives in the content store, retrieved by identity after unbind
let content = content_store.get(entity_id)?;
```

**Workspace instances:** D5 Frankenstein (sprint-11) — `Vsa10k = [u64; 157]`
bitpacked + XOR `RoleKey::bind/unbind` + `vsa_xor` / `vsa_similarity`
(Hamming-based reinvention). Three composite errors: content bundled as
identity, XOR instead of multiply, wrong dimensionality. Reverted in
commit `0ae9f90`. This instance established the N ≥ 3 violation count
within a single PR that drove I-VSA-IDENTITIES to iron-rule status.

**Promotion track:** ALREADY PROMOTED — I-VSA-IDENTITIES in CLAUDE.md
(2026-04-21, `iron-rules-doctrine.md §2.3`). BAP8 is the operational
detection pattern for iron-rule violations at the carrier boundary.

---

### BAP9 — Feature-Gated v1 API Alias Collides with v2 Canonical Name

**Boundary class:** B1 (DTO↔DTO) — specifically the API version axis

**Symptom:** A v1 API accessor (`pack()`, `set_temporal()`,
`inference_type()`) exists alongside a v2 layout under a feature flag.
The v1 accessor writes to bit positions that the v2 layout reclaimed.
Under the v2 feature, the write silently corrupts adjacent fields.
The corrupted struct passes any round-trip test that only checks the
written field (not the adjacent fields) — so the bug survives the worker's
self-validation.

**Grep target:**

```bash
# v2-layout feature blocks in the diff:
git diff main..HEAD \
  | grep -E '^\+.*#\[cfg\(feature = "v2-[a-z-]+"\)\]'

# v1 setters that write raw bits:
git diff main..HEAD -- '*.rs' \
  | grep -E '^\+.*pub fn (set_|pack|with_)' \
  | grep -E '\|=|<<|>>'

# Field-isolation matrix tests:
grep -n "field_isolation\|bit_bleed\|non_perturbation" \
     crates/causal-edge/tests/
```

**The rule:** every v1 API path under a v2 feature flag MUST be either:
(a) feature-gated to no-op under v2 with a `// MIGRATION:` pointer, OR
(b) routed through the canonical v2 accessor.

Field-isolation matrix tests (N × (N-1) assertions for N fields) are
MANDATORY at every layout-bit boundary touched by a v2 reclaim. See
`crates/causal-edge/tests/v2_layout_tests.rs` for the reference matrix.

**Fix pattern:** see AP1 in `codex-p1-anti-patterns.md` (exact same
pattern, surfaced by PP-13 for within-crate; BAP9 is the boundary-level
detection companion).

**Workspace instances:** PR #383 — 4 instances in one PR (`pack()`
temporal write, `inference_type()` raw discriminant return, `set_temporal()`,
`forward()`). Commits `42b3215` + `b44ce87`. Codex caught all 4. This
established N ≥ 3 violation count that drove E-META-10 and ultimately
I-LEGACY-API-FEATURE-GATED promotion.

**Promotion track:** ALREADY PROMOTED — I-LEGACY-API-FEATURE-GATED in
CLAUDE.md (sprint-12, `iron-rules-doctrine.md §2.4`). BAP9 is the
operational detection pattern for the API version boundary.

---

### BAP10 — Producer Option<X> / Consumer Unwrap Pattern

**Boundary class:** B1 (DTO↔DTO)

**Symptom:** A producer function returns `Option<X>` to reflect a
legitimate absence case (e.g., "no matching codebook entry," "fingerprint
below similarity threshold," "sprint placeholder not yet resolved"). The
consumer unwraps with `.unwrap()` or `.expect()`, assuming `Some` on the
hot path. In the tail case — the absence case — the consumer panics at
runtime. The panic is not visible in happy-path tests; it surfaces in
production under low-frequency inputs.

**Grep target:**

```bash
# Unwrap calls on function returns in the diff:
git diff main..HEAD -- '*.rs' \
  | grep -E '^\+.*\.unwrap\(\)|^\+.*\.expect\("'

# The producer side — functions returning Option:
git diff main..HEAD -- '*.rs' \
  | grep -E '^\+.*-> Option<'
```

**The rule:** any cross-crate boundary that returns `Option<X>` MUST be
matched by a consumer that explicitly handles the `None` case — either
with `?` (propagate), `unwrap_or(default)` (provide a default), or an
explicit `match`. A naked `.unwrap()` on a cross-crate `Option` return
is P1 (CATCH-LATENT) unless the caller can prove the `None` case is
structurally impossible (which requires a comment explaining WHY).

**Fix pattern:**

```rust
// WRONG: unwrap on cross-crate Option
let fingerprint = catalogue.lookup(id).unwrap();

// RIGHT: propagate with ?
let fingerprint = catalogue.lookup(id)?;

// RIGHT: explicit None handler with rationale
let fingerprint = catalogue.lookup(id)
    .ok_or_else(|| BoundaryError::MissingIdentity { id })?;

// RIGHT: unwrap with proof (structural invariant)
// SAFETY-OPTION: id is always pre-validated by the catalogue constructor;
//               lookup() returns None only for ids not in the catalogue,
//               and this function is only called with ids returned by the
//               same catalogue's list() iterator.
let fingerprint = catalogue.lookup(id).unwrap();
```

**Workspace instances:** no direct sprint-11/12 CSI entry for
Option-unwrap at a boundary specifically. Pattern is observed in worker
template self-validation gaps (PP-8 §5.5 field-isolation tests address
the structural analogue for layout fields). Documented preemptively
because the wide sprint-13 scope (D-CSV-13/14/16/17) introduces new
cross-crate handoffs that will have `Option` returns from the CAM-PQ
and SIMD subsystems.

**Promotion track:** N = 0 confirmed sprint instances. Document with the
first sprint-13 occurrence; promote to iron-rule candidate after N = 3.

---

## §3. Severity convention

Maps directly to the baton-handoff-auditor verdict scale.

| Severity | Meaning | Verdict | Auto-action |
|---|---|---|---|
| **P0** | Baton dropped — breaks compile, corrupts state, or violates an iron rule at the boundary | CATCH-CRITICAL | Block merge; spawn fix worker |
| **P1** | Baton at risk — will drop next sprint if unaddressed | CATCH-LATENT | Log TD-*/CSI-*; next sprint owns fix |
| **P2** | Boundary survives but has a fragile seam | CLEAN (with note) | Log TECH_DEBT; watch in next wave |
| **P3** | Naming / style inconsistency at boundary | CLEAN | Informational; no action |

**Default severity by BAP:**

| BAP | Default P |
|-----|-----------|
| BAP1 DTO field-shape drift (silent, at compile boundary) | P0 |
| BAP2 Rename without downstream sweep | P1 (P0 if blocks compile) |
| BAP3 Lib.rs / mod.rs orphan | P0 (invisible tests + type-not-found) |
| BAP4 Cargo.toml workspace self-declaration | P0 (crate excluded from workspace) |
| BAP5 REST endpoint without OrchestrationBridge | P1 (architectural drift; compiles) |
| BAP6 Sprint placeholder consumed as resolved | P1 (P0 if the sprint-N+1 impl is blocked) |
| BAP7 D-id / OQ-id collision | P1 (P0 if workers are already spawned) |
| BAP8 Iron-rule carrier boundary violation | P0 (I-VSA-IDENTITIES violation) |
| BAP9 v1-API alias under v2 feature | P1 (P0 if corrupts shipped behaviour) |
| BAP10 Option unwrap at boundary | P1 (CATCH-LATENT — will panic on tail case) |

---

## §4. When NOT to spawn the baton-handoff-auditor

**DO NOT spawn for:**

- **Within-crate code review** (clippy errors, unsafe without SAFETY,
  unused imports, rustfmt violations) → route to **PP-13
  brutally-honest-tester** (`.claude/agents/brutally-honest-tester.md`).
- **Pre-spawn plan drift** (planner outputs conflict with spec versions,
  plan is inconsistent with board state, worker prompt cites a non-
  existent deliverable) → route to **PP-16 preflight-drift-auditor**
  (`.claude/agents/preflight-drift-auditor.md`).
- **Positive boundary expansion** (proposing a new cross-crate interface,
  identifying an alignment opportunity that reduces friction) → route to
  **PP-14 convergence-architect** (`.claude/agents/convergence-architect.md`).
- **Single-file edit with no cross-references** (README update, comment
  fix, a test within a single crate that doesn't import across crate
  boundaries) → no boundary auditor needed; save tokens.
- **Known-resolved CSI entries** (the sprint meta-review already closed
  the entry with a verified commit SHA on master) → no re-audit needed;
  the CSI ledger is the authoritative record.

The baton-handoff-auditor fires at BOUNDARIES. A single-crate change
with no cross-crate import, no REST handler, no lib.rs touch, and no
sprint-handover document is not a boundary crossing — do not spawn.

---

## §5. Workflow integration — CCA2A slot diagram

```
[plan]          →  PP-16 preflight-drift-auditor
                   (pre-spawn: plan vs. board state drift check)
                          │
                          ▼
[sprint impl]   →  workers run per worker-template-v2.md (PP-8)
                   worker self-validation: cargo check / test / clippy / fmt
                          │
                          ▼
[DURING-IMPL]   →  ┌─────────────────────────────────┐
                   │   baton-handoff-auditor (PP-15)   │
                   │   boundary class scan B1..B8      │
                   │   BAP catalogue BAP1..BAP10        │
                   │   verdict: CATCH-CRITICAL /        │
                   │            CATCH-LATENT / CLEAN    │
                   └─────────────────────────────────┘
                          │
                   CATCH-CRITICAL → spawn fix worker → loop back
                   CATCH-LATENT   → log TD-*/CSI-* → continue
                   CLEAN          → proceed
                          │
                          ▼
[post-impl]     →  PP-13 brutally-honest-tester
                   (within-crate: clippy / fmt / audit / deny /
                    AP1..AP8 codex-style anti-patterns)
                          │
                          ▼
[post-commit]   →  W-Meta-Opus meta-review
                   (cross-spec cross-wave review; CSI ledger)
                          │
                          ▼
[merge]

sibling connections:
  ├── PP-14 convergence-architect: when CATCH-LATENT resolves to
  │   "boundary doesn't exist yet; design a new alignment"
  └── PP-16 preflight-drift-auditor: when boundary mismatch is
      traced back to a plan spec error (CSI-19 ID collision started
      in the plan, not in the code)
```

---

## §6. Maintenance protocol

This document is **APPEND-ONLY** within the BAP catalogue section.
When the baton-handoff-auditor or a sprint meta-review identifies a new
boundary drop pattern not covered by BAP1..BAP10:

1. **Triage** — add a CSI-* entry in the sprint-log meta-review.
2. **BAP entry** — if the pattern is observed in N ≥ 2 PRs, APPEND a
   BAP11+ entry to §2 with the same shape as BAP1..BAP10.
3. **Iron-rule promotion** — if the pattern accumulates N ≥ 3 sprint
   instances AND has substrate-level consequences, follow the ceremony
   in `iron-rules-doctrine.md §3`.

**Positive-flip protocol:** when a CATCH-LATENT drop turns out to be a
missing boundary alignment rather than a bug (e.g., the two crates
simply haven't defined the interface yet), hand off to **PP-14
convergence-architect** with a note in the CATCH-LATENT finding: "This
boundary is an OPPORTUNITY — see convergence-architect for alignment
proposal." This is the rare positive flip where a latent drop becomes a
design input rather than a fix target.

**Do NOT delete or edit existing BAP entries.** If a BAP becomes
obsolete (e.g., the relevant crate is retired), APPEND a DEPRECATED
annotation citing the retirement PR. Iron rules that were promoted from
a BAP are noted with a `PROMOTED → <iron-rule-name>` annotation.

---

## §7. Cross-references

### Baton-handoff-auditor agent

- `.claude/agents/baton-handoff-auditor.md` (PP-15) — the agent that
  runs this catalogue; load §2 before scanning any boundary.

### Sibling agents in the four-agent quality lifecycle

- `.claude/agents/brutally-honest-tester.md` (PP-13) — within-crate
  post-impl gate; AP3/AP4/AP5 overlap with BAP3/BAP4/BAP7 at the compile
  surface; the two agents are complementary.
- `.claude/agents/convergence-architect.md` (PP-14) — DIVERGENT boundary
  alignment proposer; the positive-flip destination for CATCH-LATENT drops
  that are actually missing alignments.
- `.claude/agents/preflight-drift-auditor.md` (PP-16) — pre-spawn plan
  drift detector; the upstream agent whose output this catalogue refines
  at the code-boundary level.

### Iron rules and doctrine

- `CLAUDE.md §Substrate-level iron rules` — I-VSA-IDENTITIES (BAP8),
  I-LEGACY-API-FEATURE-GATED (BAP9), I-SUBSTRATE-MARKOV (BAP1 carrier
  implications), I-NOISE-FLOOR-JIRAK (statistical boundary context).
- `.claude/knowledge/iron-rules-doctrine.md` (PP-2) — the four-axis
  framing and promotion ceremony for new iron rules.

### Lab-vs-canonical doctrine

- `.claude/knowledge/lab-vs-canonical-surface.md` — BAP5 operational
  doctrine. MANDATORY before any boundary touching REST / gRPC / Wire DTO.

### Sibling knowledge doc

- `.claude/knowledge/codex-p1-anti-patterns.md` (PP-13 sibling) —
  within-crate anti-pattern catalogue. AP3 (sub-crate workspace) / AP4
  (lib.rs orphan) / AP5 (cross-repo mod.rs orphan) / AP8 (REST drift)
  are the within-crate mirror of BAP4 / BAP3 / BAP3 / BAP5.

### Sprint-log CSI ledger

- `.claude/board/sprint-log-13/preflight-meta-review-opus.md`
  §3 CSI-19..23 — verified boundary mismatches for sprint-13.
- `.claude/board/sprint-log-12/meta-review.md` §4 CSI-7..18 — sprint-12
  boundary mismatch ledger; CSI-7 (BAP4 instance), CSI-8 (BAP3 instance),
  CSI-9 (BAP3 cross-repo + BAP6 false-resolution instance).

---

## §8. One sentence that should survive any refactor

**The baton drops at the boundary — not inside the crate, not in the
plan, but in the handoff between them — and this catalogue is the map
of every drop this workspace has documented.**

---

*Authored W-Sprint-13-PP-15 (Opus 4.6, Claude Code subagent),
2026-05-16. Sources: sprint-log-13/preflight-meta-review-opus.md
CSI-7..23; codex-p1-anti-patterns.md (PP-13 sibling);
iron-rules-doctrine.md (PP-2); lab-vs-canonical-surface.md; CLAUDE.md
iron rules; PR_ARC_INVENTORY.md PRs #383/#388/#389; commit SHAs
`42b3215`, `b44ce87`, `d4e5bbc`, `0ae9f90`, `2a1a1e38`, `2a3885d2`.*
