# Cognitive Method–Bag API seam: `crate::{op}::{param}::{DRBagRef}` (proposed v1)

**Status:** SCRATCHPAD / CONTRACT PROPOSAL / NO CODE. **Owner:** existing lance-graph contracts and admission gates, not a second execution runtime. See [mindmap](cognitive-teacher-feedback-mindmap-v1.md), [experiment LG-COG-001](cognitive-teacher-experiment-v1.md), and existing `.claude/plans/d-lfa-lance-graph-flow-rubicon-assimilation-v1.md`.

## North star

The user intent is to stop repeatedly *parsing* instruction text to discover behaviour. Instead, use an explicit **cold resolution and binding boundary** that turns the LLM's proposed plan into the already-existing *typed method and parameter coordinates* over a pinned **virtual tuple bag over DR64³ where semantically applicable**. Hot execution is a method call/relational algebra kernel over resident typed lanes, with a separate typed outcome and evidence witness. Do not mistake natural-language representation for runtime object graphs.

Notation proposed:

```text
EXTERNAL                         COLD RESOLVE                   HOT / BORROWED
Claude tool JSON                 Codegraph + ClassView          validated method dispatch
 "op":"fold.count"                class/MethodRef/word           ┌─────────────────┐
 "param": "selected"   ------->    ParamRef + law witness  ----> │ op(bag, params) │
 "bag": "bag-17"                  DRBagRef + BasisWitness         └─────────────────┘
                                permission, focus, generation          │
                                                           typed OutcomeRef
                                                                  │
                                                 external RolloutDB evidence
```

`crate::{op}::{param}::{DR{bag reference}}` is a useful **call signature concept**, not literal valid `crate` path syntax for dynamic data and NOT evidence of an existing `MethodRef` ABI. Suggested conceptual invocation is `invoke(MethodRef, ParamRef, DRBagRef, ExecutionContext)` on a bound method. Static namespace `crate::op::param` and dynamic DRBagRef are separate categories. Do NOT generate a Rust crate per LLM operation.

## 1. Typed contract proposal (not compilable against current repository)

```rust
// Conceptual API names; do not copy into production without source/type audit.
struct VersionedDRBagRef {
    dataset_id: DatasetId,
    world_version: u64,
    class_id: ClassId,
    bag_id: BagId,
    owner_generation: u64,
    basis_digest: BasisDigest,
    lane_schema_digest: Digest,
    // Optional geometry witness, only for DR64³ spatial methods:
    spatial_frame: Option<SpatialFrameWitness>,
}
struct BoundMethod {
    method: MethodRef,
    positional_word: PolyfillWord,
    lens: LensId,
    law_digest: LawDigest,
    codebook_generation: u64,
    parameters: ParamRef,      // typed positional arguments, not serde_json::Value
}
struct ExecutionAdmission {
    scope: Scope,
    access_policy: PolicyWitness,
    source_generation: u64,
    operation_effect: EffectClass, // PureRead | Proposal | SideEffect
    predicate_projection: TruthProjection,
}
fn invoke(
    method: BoundMethod,
    bag: VersionedDRBagRef,
    admission: ExecutionAdmission,
) -> Result<OutcomeRef, Refusal>; // illustrative
```

Unknown semantic scopes, unknown owner, stale generation, changed law, ambiguous nullable interpretation, missing row-order witness, undocumented cross-bag addressing and missing independent oracle must be explicit refusals. Hashes need canonical encoding, versioning and collision handling; do not silently treat `u64` as a complete immutable semantic proof. No leaked scratch `Val::Slot { epoch }` handle across epochs; a durable reference must rebind against pinned current owner.

### Identity distinctions

| Identity | What it names | Lifetime |
| --- | --- | --- |
| `MethodRef` | class method, positional word and versioned semantics | durable only with law/version metadata |
| `ParamRef` | positional argument/witness, including selected population | until method binding/lease rules expire |
| `DRBagRef` | virtual typed population + logical owner/basis | across process only if explicitly rebindable |
| `ThoughtSlot(index,generation)` | one live speculative overlay, **not** durable thought ID | slot pool lifetime |
| `ThoughtEventId` | durable attempt lineage | rollout/evidence store |
| `LanceVersion` | canonical dataset main snapshot | dataset version |
| `RolloutId` | historical teacher/learning trajectory | RolloutDB record set |

Separate Main `thought=0` vs live `1..63` overlays; squash combines speculative states and frees slots, no implicit canonical commit. Generation-bound IDs cannot become physical pointer casts. No persistent raw bitmap pointer, detached array slice or Rust lifetime in model messages.

## 2. Virtual tuple bags over DR64³: explicit semantic fences

- **DR64/DT64 virtual bags**: methods operate on a declared logical population, lanes, masks, basis and version. A bag is not a `Vec<Row>` to be JSON-rehydrated.
- **DR64³ spatial field**: if the operation involves a 3D neighbourhood or field, the spatial orientation, grid topology, incidence and boundary conditions must be declared and independently validated; don't assume an arbitrary tuple bag has spatial Moore26 semantics.
- **Columnar no-materialization**: obtain borrowed typed lane/selection views; Quack/mask-RISC kernels consume admitted resident masks. Remote requests may serialize metadata only at an explicit network boundary; no row-by-row object payload in the compute loop.
- **Law**: For each operation, say whether the semantics is set, bag/multiplicity, ordered prefix, K3 nullable predicate, spatial incidence or counterfactual overlay. Same shape != same law.
- **Witnesses**: basis/version, source generation, row order, mask residency, population derivation, codebook, method law, input scope and effect authorization are independent gates.

### No invented Gather or side-table semantics

Existing admission rules forbid value-dependent cross-table addressing without an explicitly admitted Gather; `Filter::Semijoin` cannot automatically reuse a just-computed selection as foreign mask. A reduction over one resident bag is not proof of a general relational join. XLOG and relalg exploration must obey these constraints. A `DR64³` vocabulary must not be used to conceal missing operations or claim CFD equivalence.

## 3. Seam map with concrete existing source anchors

### `lance-graph-flow` (proposed)

- Existing D-LFA at `.claude/plans/d-lfa-lance-graph-flow-rubicon-assimilation-v1.md` calls it **not yet landed** and assimilation-only. Source for possible donor is `AdaWorldAPI/rs-graph-llm`, especially `graph-flow` and `graph-flow-lance`.
- Adopt GraphBuilder, Task, conditional edge, NextAction, Pause/Resume as *thin control-plane ergonomics* over method handles, not a runtime executor.
- Context/session contains **only** IDs, codebook/source generation, method refs, thought version and small controls. No SoA or mask materialization; no independent session DB as authority.
- A THINK task produces proposed derived data, a COUNTERFACTUAL task runs a witnessed alternative, DO returns an ActionDef-like proposal. No task can change Kanban phase or Lance version autonomously.
- Verifiable fixture: two method calls + alternative + WaitForInput, stale-generation pause/resume must refuse; optional GKAT guard equivalence and K3 `{T}` commit projection per D-LFA. Distinguish guarded flow proof from runtime version fence.
- **Status gate:** until a real `lance-graph-flow` crate exists, use a lab driver or explicit stub; do not present it as shipped.

### `lance-context` / ContextStore + RolloutDB

- ContextStore: external canonical RAG snippets with URI/SHA/range/claim/evidence status; retrieval helps Claude understand methods, not authorizes invocation.
- RolloutDB: durable histories of attempts, hypothesis, chosen method, oracle, reward, cost, latency and refusal. `RolloutRecord` has string/JSON fields on external API and typed token/reward columns; internal `RolloutStore::add(&[RolloutRecord])` converts to Arrow and uses MemWAL.
- **Three receipt states:** `add` is durable, `flush` makes query-visible, merge/base seal changes Lance base version. Returned `add` base version is NOT per-append identity. Separate from canonical application `thought0` world version.
- Phase 1 adaptation via cold-side schema-versioned metadata for the trial. Phase 2 possible direct typed Arrow RecordBatch interface for numeric method/parameter/bag refs by reusing StorageBase; verify actual API boundaries and schema evolution. No immediate fork/rewrite.
- Feedback examples are from disjoint verified development fixtures only; quarantine unverifiable LLM guesses.
- Required negative: corrupt source metadata, swap semantic word, query before flush, replay with wrong world version; each must be refused or visibly unready.

### `lance-graph-thought-git` / current `thought-git-probe`

- The *requested conceptual* `lance-graph-thought-git` name is not verified as a current crate; implementation probe is `crates/thought-git-probe`. Read `thought-git-api-v1.md` and `thought-git-basics-v1.md` before claiming it is available.
- One canonical `thought 0` main (pinned Lance dataset); live slots `1..63` reusable; FIFO/LeastUsed; dependency-linked squash before eviction; generation invalidates stale refs.
- Speculative method execution over a virtual bag should return an overlay/delta and decision evidence. Squash combines overlays but never implicitly lands to main. Durable event history in RolloutDB can refer to stable thought IDs after slot recycle.
- Probe under >63 sequential thought attempts, >63 simultaneously live pressure, deeply nested branches, stale parent reuse, contradictory deltas, baseline version changes, independent oracle after process teardown.

### `lance-graph-relalg`

- Current crate exists: `src/{ir,join_body,lower,normalize,program}.rs`. Reuse its **admitted** relational operators and lowering, never interpret model text as an alternative relational AST at each step.
- Proposed call: method compiler binds declared relation/bag inputs to a typed relalg program/law, normalizes and lowers through existing verified executor path (where that route is implemented).
- Distinguish boolean vs bag/count multiplicities, positive vs negative atoms, legal composition, filter population and witness validity. Explicitly refuse unavailable cross-table joins or unproven mask-reuse.
- Oracle: simple independent scalar relational implementation plus negative fixtures (reorder, mismatched index, bag count, polarity, nullable predicate). No production claim from a fixture proof.

### `lance-graph-quack`

- Existing `crates/lance-graph-quack/src/{lib,bind,set}.rs`; Quack creates plans and lowered mask-RISC operations, not a separate evaluator. Existing filter/aggregate/semijoin restrictions remain.
- Expose read-only capability `explain_admission(BoundMethod, BagRef)` yielding **known, allowed, blocked, unknown** with exact witness.
- Only invoke after bind validation, no model-generated SQL as proof of native semantics. Quack can act as a convenient *method projection* for source-backed task operations. Materializing full query results belongs on cold/terminal boundary.

### `lance-graph-mask-risc` (user shorthand `risc-mask`)

- Existing `crates/lance-graph-mask-risc/src/{ir,exec,fuse,sel,reference,value,...}.rs`; its admitted program consumes resident lanes/masks and returns bounded typed reductions/selection outcomes.
- Compiler must emit a supported `mask_risc::Program`, not a hidden per-row interpreter and not dynamic JSON dispatch.
- Independent reference/negative tests for wrong-population mask, stale borrowed lane, computed mask reuse, unsupported gather, and null law. Counterexample: change positional opcode while keeping arity and compare against pinned method witness.
- Record execution latency and allocations separately for L1 resident and streaming cases; don't present prior specialized ns/64 values as a generic method-dispatch result.

### `lance-graph-r2il`, `lance-graph-dt64`, Codegraph and Rubicon

- `crates/lance-graph-r2il/src/{lib,binding,binding_native,polyfill}.rs`; polyfill words and epoch-sensitive scratch; no non-admitted hypothetical method execution.
- `crates/lance-graph-dt64/src/{dr,lane}.rs` and external virtual-bag plans. Bind class-view / focus / versioned population to lanes without copying whole owner state. DR64³ is a research branch requiring distinct spatial oracle.
- Codegraph/Ruff ORE/SLAG anchors source of method, implementation proof and changed ABI; run read-only feature discovery before authorizing method dispatch.
- `kanban.rs`: Planning, CognitiveWork, Evaluation, Commit/Plan/Prune. Rubicon legal successor rules and revision/counterfactual gates are authoritative; model score or reward has no effect on admission. `KanbanMove` is ≤16-byte `Copy` *in Rust representation*, not a fixed serialized ABI. No automatic publication from flow.

## 4. Semantics of thinking as config

12 StyleFamilies orchestrate; 36 ThinkingStyles are concrete NARS runbook names with canonical family mapping. Seven FieldModulation parameters are already declared. In phase one, pass config as **external prompt/control** and preserve exact config ID; do not claim code-side runbook execution or a native thinking compiler unless a real method binding exists.

Proposed use:
```text
(StyleFamily, ThinkingStyle, ModulationVersion)
  -> proposed operation trace
  -> Codegraph/ClassView MethodRef resolution
  -> normalized, admitted relalg/R2IL program over DRBagRef
  -> Rubicon Evaluation with evidence
  -> RolloutDB provenance + outcome
```
Teaching signal: which method/representation could perform a step, which cannot, whether results match oracle, and whether repeated LLM inference can be substituted with a stable method. Differentiate *discovery gap* vs *expressibility* vs *binding* vs *execution* vs *verification*.

## 5. MCP and REST membrane

Read-only first: `discover_capability(query,source_scope)`, `explain_method(method_ref)`, `probe_method(method_ref, bag_ref, pinned_version)`, `get_evidence(trace_id)`, `retrieve_feedback(problem_signature)`. REST and Railway MCP are two adapters for the SAME typed registry; `Rig` can own Claude calls but cannot commit. ZITADEL issuer/audience and scoped auth (if deployed) must be verified with explicit deny tests. No open arbitrary SQL, shell, raw filesystem, token/secret access, unrestricted tool call or production BatchWriter path.

Model prompts/tool responses are serialized at network boundary; the **hot method path** stays typed/borrowed and does not hydrate JSON rows or run string parsing per step.

## 6. Falsifier matrix

| Falsifier | Must happen |
| --- | --- |
| Invented `crate::op` path / unknown F### | `UNKNOWN_METHOD`, no dynamic eval |
| Same arity, different positional F### semantics | refuse under law digest or fail independent oracle; no silent pass |
| Owner closed or bag version changed | rebind with pinned witness or explicit stale refusal |
| Thought slot 1 reused at different generation | stale handle refused |
| Thought squash with conflicting causal predecessors | independent oracle detects mismatch/refusal |
| Computed mask reused without admitted witness | refusal, not hidden gather |
| 3D spatial face/edge mapping alters conservation | independent discrete incidence oracle fails |
| Rollout add returns but no flush | must not claim read-visible nor new main version |
| Claude suggests Commit while counterfactual absent | Rubicon refuses/replans |
| Model prompt duplicates holdout answer | exclude/report leakage |
| Flow Pause/Resume after codebook change | refusal or explicit rebind/re-evaluate |
| API tool tries production write or cross-tenant ref | authorization denied |
| A claimed S5 has no independent test | `MISSING_ORACLE`, not PASS |

## 7. Implementation sequence and stop rule

**P0** freeze evidence and map existing actual APIs (read-only). **P1** one cold Claude→capability registry→existing Rubicon/Quack native method or refusal→oracle→RolloutDB record→subsequent RAG retrieve; no new executor. **P2** add method/bag semantic IDs, exact negative fixtures and replay. **P3** optimize typed Arrow batch channel only after baseline conversion and memory measurements. **P4** optional thin flow and Railway MCP after effect/auth fences. **P5** separate DR64³ spatial meaning probe only when geometric shape/independent oracle is defined.

Hard stop: if any step needs a parallel runtime, fresh SQL evaluator, row-object materialization inside hot loop, fabricated method/geometry witness, unadmitted Gather, unsafe branch/resume, or unreviewed canonical write, freeze and report the missing contract. Continue lab analysis, never pretend the gate passed.
