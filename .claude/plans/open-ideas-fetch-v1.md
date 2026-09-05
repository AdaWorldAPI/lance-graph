# Open-ideas fetch — three stale cards re-derived from the tree (v1)

> **Status:** MEASURED / ready-to-execute — PLANNING ONLY in this PR. Every
> claim below carries file:line from a read of the CURRENT tree (origin/main
> `ac9148f7`, #1171, plus the FF to `afeb0458`). NOT COMPILED, NOT RUN — no
> `cargo` was invoked; every "exists / is wired / has callers" statement is
> structural (grep + read), and §8 names the ones that must be confirmed by
> execution before they are believed.
>
> **READ BY:** the orchestrator that dispatches the Sonnet workers (§7 carries
> the per-D briefs), anyone touching `lance-graph-callcenter::policy`,
> `sigma_propagation`, the `MailboxSoA.sigma` column, or `crates/jc`.
>
> **Origin:** `IDEAS.md` had gone 82 days without an entry (newest 2026-06-15)
> while ~900 PRs merged. Three of its Open cards were picked as the highest
> value; **each one changed shape on contact with the tree**, which is the
> reason this plan exists rather than three direct PRs.
>
> **Confidence:** HIGH on the census; LOW-to-none on anything marked
> DECISION — those are the operator's, stated with a recommendation and
> deliberately not taken.

## 0. The three headline corrections

| card (IDEAS.md) | what the card says | what the tree says |
|---|---|---|
| `IDEA-POLICY-HASH-UDF` (`:1162`) | "policy_hash_v1 UDF **registration**" | Registration is not the blocker. The UDF is bound as an `Arc<ScalarUDF>` **object inside the `Expr`** (`policy.rs:137`), and DataFusion executes the embedded object without any by-name lookup — proof: `register_vsa_udfs` (`vsa_udfs.rs:574`) has **zero callers** in the tree and its UDFs still execute in tests. The blocker is the **body**: `invoke_with_args` returns `NotImplemented` (`policy.rs:~330`). **Superseded the same day (§2, W0 census): the body was never the blocker either — the DataFusion masking path has no deployed consumer, and the operator ruled DataFusion out of planning. Ruling A: retire.** |
| `IDEA-B1-HARDWARE-BACKENDS` (`:1062`) | "AMX/MKL backends for sigma_propagation, waits on ndarray #119/#121" | The kernel is a **2×2 f64** sandwich — three scalars, 12 mul + 6 add (`sigma_propagation.rs:210-227`). An AMX tile is 16×16 bf16/int8; an MKL `dgemm` call costs more than the 18 flops it would do. The idea is **mis-shaped for its own kernel**. The real lever is vertical batching — `F64x8` (8 edges/instruction), whose `Mul`/`Add` already exist on every backend. AND: `ewa_sandwich(` has **zero production call expressions** outside `jc` + the contract (grep, §3.2) — every "caller" is a doc-comment mention. |
| `IDEA-CAUSAL-EDGE-TENSOR-SIDECAR` (`:1126`) | "design a 9-byte sidecar (`CausalEdge64` + 1 byte Σ index) OR Block 14/15" | **The 1-byte index already shipped, as a SoA column** — `BindSpace.fingerprints.sigma: Box<[u8]>` (`bindspace.rs:54-58`), `MailboxSoA.sigma: [u8; N]` (`mailbox_soa.rs:125-133`), `set_sigma` (`:670`), `BackingStoreWrite::set_sigma` both arms (`backing.rs:253-257`). Neither the sidecar nor Block 14/15 is live. **What does NOT exist is the codebook the index points at** — §4. |

## 1. Frozen decisions this plan inherits (cited, not re-derived)

- **Zero-dep contract.** `lance-graph-contract/Cargo.toml` `[dependencies]` is empty by design and says why (a path dep there broke every CI invocation, 2026-07-07). Nothing in this plan adds a dependency to it.
- **All SIMD from `ndarray::simd`** (`simd-savant`; W1a consumer contract, ndarray `.claude/knowledge/vertical-simd-consumer-contract.md`). Consumer code composes typed wrappers; it never writes intrinsics.
- **Bit identity between the jc kernel and the contract copy is CI law** — `jc::ewa_sandwich::tests::the_contract_copy_matches_the_certified_kernel_bit_for_bit` (`ewa_sandwich.rs:~425`, 1000 sampled SPD pairs, `to_bits()`, no tolerance; `jc-proof.yml`). Any batched kernel joins that law or does not land.
- **Internal head pins are absolutely and permanently prohibited** (MedCare-rs `CLAUDE.md`, operator 2026-08-30). A frozen codebook artifact gets a **tag + content invariants**, never a digest gate.
- **Indices, not content** (`I-VSA-IDENTITIES`; `mailbox_soa.rs:128-131`): the codebook stays shared/cold; rows carry the 1-byte reference.
- **Masking is projection-side, never a DataFusion rewrite** (operator ruling 2026-09-05, recorded on #1188 as `E-PLANNING-MIGRATES-TO-LOCO-R2IL-DATAFUSION-IS-GRACE-PERIOD-1`): the field population is `ClassView × WideFieldMask`, consumed by Lance reads and loco programs; DataFusion-hosted surfaces are grace-period — maintained, never extended. No new operator, rule, or UDF routes into DataFusion.
- **The falsifiability rule** (`CLAUDE.md`): every guard needs a can-fire AND a can-stay-silent test on non-trivial inputs; a threshold needs an inertness test; a doc claim is not a behaviour.
- **Sonnet workers: edit-only, no `cargo`, no `git`, disjoint files; the orchestrator compiles once** (`agent-cargo-hygiene.md`; `sonnet-worker-guardrails.md` §1 pasted verbatim into every brief).

## 2. D-OIF-1 — `policy_hash_v1`: RULING A, SUPERSEDED — a retirement plan, not a crypto task

> **Re-derived 2026-09-05 (W0 architecture census, four read-only Opus tracers + orchestrator verification).**
> Independent census on #1188 (banked at `.claude/board/exec-runs/d-oif-1-census-main-thread.md` on that branch) reached the same ruling; it scanned lance-graph only, so it reports the rewriter as test-constructed — the one extra constructor is the private-repo MedCare `routes/patient.rs:150`, which is why the removal cone below is narrower than "delete `policy.rs` entire": that deletion breaks MedCare's `lance-phase2-rbac` compile until MedCare retires the feature first. Its proposed `E-THE-UNFINISHED-UDF-WAS-NOT-THE-DEBT-1` is the same finding as the banked `E-THE-UNFINISHED-FUNCTION-WAS-NOT-THE-DEBT-1` — one entry, not two.
>
> The previous §2 asked "which hash?". That question was one abstraction generation behind: it assumed the DataFusion policy-rewriter execution model still owned field-level authorization. The census below shows it never reached a binary, and the operator ruled the same day: **"every planning is in migration to ogar-loco and ogar-r2il; DataFusion is out of the picture; what exists gets a grace period; nothing new will migrate to it."** A hash body for `policy_hash_v1` is *new* work on the DataFusion path and is barred by that ruling regardless of the census. No hash family is chosen; `D-OIF-1-DEC` is **withdrawn**.

### 2.1 Tree verdict (production census; "production" = a real call chain from a binary / server handler, never `pub mod`, a feature flag, a registration helper, a test, or a doc comment)

1. `ColumnMaskRewriter` has exactly one non-test constructor in all seven repos: `MedCare-rs/crates/medcare-server/src/routes/patient.rs:150`, inside `get_one_via_lance` (`:85`, `#[cfg(feature = "lance-phase2-rbac")]`), reached only for `?source=lance` (`:52-53`).
2. `lance-phase2-rbac` is default-off (`medcare-server/Cargo.toml:215` `default = []`) and **no Dockerfile enables it** (`docker/Dockerfile.railway:175`, `Dockerfile.railway.OCR:49`, `Dockerfile.reasoning` all build `lance-phase2,reasoning`).
3. Even when compiled, the path's row decoder `record_batch_to_patient` (`patient.rs:200-211`) is a stub that logs and returns `None`; the handler 500s. No user has ever received a masked row from this path.
4. The rewriter redacts **above the scan**: `rewrite_plan` (`policy.rs:210`) maps expressions and `recompute_schema`s; `TableScan.projection` is never written (`extract_table_name` `:226-241` only reads it). The forbidden column is scanned from Lance, materialized, then overwritten — post-hoc redaction, the exact anti-pattern the invariant forbids.
5. `RedactionMode::Hash` (`policy.rs:130-140`) binds `NotYetWiredHashUdf`; any plan containing it fails at execute with `NotImplemented` (`:339`). `policy_hash_v1` has no implementation and no `register_udf` anywhere. `register_vsa_udfs` (`vsa_udfs.rs:574`) has zero callers in all repos.
6. `PolicyRewriter` (`policy.rs:56`) has one impl and `dyn PolicyRewriter` appears nowhere — a one-impl indirection, not a policy VM.
7. The replacement path is **also not enforced**: `lance-graph-rbac::authorize()`/`authorize_scoped()` (`authorize.rs:66,182`), `contract::ClassRbac` (`rbac.rs:143`), `ActionInvocation::commit_via` (`action.rs:327`) and `lance-graph-ogar::OgarRbac` (`rbac_impl.rs:38`) have **zero non-test callers**. `effective_mask` is not an identifier in any `.rs` file. There is no `ogar-rbac` crate; `ogar-auth` is authentication only (password/TOTP).
8. MedCare's live authorization is entity-string keyed: `patient.rs:280 bridge.authorize_read("Patient", …)` → `lance-graph-rbac::policy::Policy` → `AccessDecision`. No mask reaches it. Field projection in production is `views/project.rs:229` (`WideFieldMask::from_positions` from a **view** spec, no role operand) — a view mask, not RBAC.
9. The only genuine `surface ∩ role` fail-closed projection is `a2ui-server/src/project.rs:70-83` (`WideFieldMask::intersect`, `NoRoleGrant` on empty role) — and `a2ui-server` has no `[[bin]]`, no `main`, and no dependent crate.
10. Rubicon/Kanban: no actor/controller survives (`kanban_actor.rs:1-27` tombstone: "All of it is DELETED"); the crossing `Planning → CognitiveWork` is a checked one-way edge (`kanban.rs:98-106`, `can_transition_to` consulted by `try_advance_phase` `soa_view.rs:314`, no mutation on refusal `:329-346`). `try_advance_phase` has one non-test caller (`cycle_driver.rs:560`) that is itself reached only from tests/examples. RBAC code holds no Kanban state; lifecycle code makes no authorization decision; loco/r2il make no authorization decision.

**So:** the invariant «forbidden fields absent from the authorized projection, not materialized and masked by a UDF» is *violated by the only field-level mechanism that was ever written* (item 4) and *satisfied by nothing that is deployed* (items 7–9). The invariant «a Rubicon transition is an SoA-owned mutation, not an actor command» **holds** in code (item 10) and is contradicted only by stale prose (§2.5).

### 2.2 Obligation table

| old obligation | current owner | production evidence | action |
|---|---|---|---|
| `policy_hash_v1` UDF body (`IDEA-POLICY-HASH-UDF`, PR #301) | none — the DataFusion policy path | items 1–5: no deployed consumer; execute → `NotImplemented` | **RETIRE.** No body, no hash choice. `D-OIF-1-DEC` withdrawn |
| `RedactionMode::Hash` | same | binds a UDF that cannot execute | **REMOVE with the cone** (§2.3) — a landmine, not a mode |
| `NotYetWiredHashUdf` / `policy_hash_v1` | same | `policy.rs:279-339`, zero registrations | **REMOVE with the cone** |
| `ColumnMaskRewriter` / `ColumnMaskRegistry` / `RedactionMode::{Null,Constant,Truncate}` / `PolicyRewriter` / `PolicyKind` | same (`policy.rs`) | one gated, undeployed, stub-terminated MedCare caller | **GRACE PERIOD** (operator ruling): retained as-is, regraded SUPERSEDED, frozen — no new work, no new callers. Removal is its own PR once MedCare's `lance-phase2-rbac` feature is retired (`patient.rs:85-211`, `state.rs` `column_mask_registry`/`rls_registry`/`session_context`, `medcare-analytics::column_mask_bridge`) |
| `RlsRewriter` (`rls.rs`) | same | same single caller (`patient.rs:146`) | **GRACE PERIOD**, same cone, same condition. Not a policy VM: a tenant predicate — but on the retired path |
| `register_vsa_udfs` (`vsa_udfs.rs:574`) | none | zero callers, any repo | **REMOVE** candidate; separate from the policy cone (it is `query`-gated, not `auth-rls-lite`). Not touched in this PR |
| DataFusion RLS / optimizer-rule registration | `patient.rs:146-150` only | the only `with_optimizer_rule` in any repo | **GRACE PERIOD** with the MedCare feature |
| `unified_bridge.rs` `authorize_{read,write,act}` | `lance-graph-callcenter` | live: `patient.rs:280` | **RETAIN.** Entity-level allow/deny before SQL; never touches a plan |
| Field-level authorization (the obligation `policy_hash_v1` was one arm of) | **VACANCY** on the canonical path: `ClassRbac × ClassView × WideFieldMask` | items 7–9 | **RECORD as the missing implementation** on the canonical OGAR/ClassView path (OGAR already says so: `DISCOVERY-MAP.md:1581-1584`, `DOCIR-COMPOSITION-GROUNDING.md:75`, `PROBE-OGAR-RBAC-AUTHORIZE`). Not resurrected on DataFusion. Not built in this plan |
| `authorize_scoped` mask fold | `lance-graph-rbac/authorize.rs:199-216` | zero callers; folds a **union** over roles and returns `FieldMask::FULL` on non-Allow (`:190-196`) — fails OPEN in the mask | **REGRADE** to "TRANSPORT ONLY, fail-open in the mask value" — a design defect to fix on the canonical path, not evidence for the old one |
| Rubicon lifecycle | `MailboxSoaOwner::try_advance_phase` + `KanbanColumn` DAG | item 10 | **RETAIN**, no change. Prose corrections only (§2.5) |
| Thinking/planning | `ogar-loco` (one live consumer: `medcare-cohorts/ddx_loco.rs` ← `views/zugfolge.rs:534`); `ogar-r2il` (probe/test only); planner `thinking/`,`mul/`,`strategy/` linked into medcare-server but uncalled | no DataFusion UDF or rule performs reasoning on any production path (`datafusion_planner/udf.rs:130-663` is vector/hamming distance only) | hypothesis **confirmed for DataFusion** (it never owned reasoning in production); **partially confirmed for loco/r2il** (loco live, r2il probe-only). Nothing to move |

### 2.3 Remove / retain / regrade (exact)

- **REMOVE now (this arc, own PR, after grace-period sign-off):** nothing is deleted in #1185. The *first* removal PR is scoped to the dead-by-construction pieces only — `RedactionMode::Hash` + `NotYetWiredHashUdf` + the `policy_hash_v1` name (`policy.rs:130-140, 262-339`) and the test `redaction_mode_hash_binds_not_yet_wired_udf` (`:598-620`) — because they cannot execute today and no grace-period consumer can depend on them. Disable-run: removing the variant must break exactly that test and nothing else.
- **GRACE PERIOD (frozen, regraded SUPERSEDED, no new callers):** `policy.rs` remainder (`PolicyRewriter`, `PolicyKind`, `ColumnMaskRegistry`, `ColumnMaskPolicy`, `ColumnMaskRewriter`, `RedactionMode::{Null,Constant,Truncate}`, and the same-shape stubs `RowEncryptionPolicy` / `DifferentialPrivacyPolicy` `policy.rs:344-420`, zero callers), the DataFusion forward-stubs `datafusion-dispatch` / `datafusion-plan` / `postgrest.rs:940-960` (`Err("not yet implemented")`) / `MembraneRegistry::with_rls`, `rls.rs`, the `lib.rs:114-125` / `:93-99` gates, feature `auth-rls-lite`; MedCare `patient.rs:52-54, 85-211`, `state.rs` `column_mask_registry` / `rls_registry` / `session_context` / `build_session_context`, `medcare-analytics/src/column_mask_bridge.rs`. Removal condition: MedCare retires `lance-phase2-rbac` (its own private-repo PR) — then the callcenter cone falls in one PR.
- **RETAIN, untouched:** `lance-graph-python/src/graph.rs` (`SessionContext` `:1208,:1518`), `lance-graph-catalog/src/{connector,table_reader}.rs`, `holograph/src/storage.rs`, `callcenter/src/{graph_table,filter_expr,lance_membrane,unified_bridge}.rs`, `callcenter/src/bin/audit_verify.rs` + `audit.rs` (Lance dataset open), `lance-graph-planner/src/optimize/` (an in-house `OptimizerRule` trait of the same name, unrelated), MedCare `views/project.rs` + `medcare-rbac`, `tesseract-paperless/src/store.rs`. These are storage/query translation over Lance — the grace-period storage layer, not the policy VM.
- **REGRADE (board):** `IDEA-POLICY-HASH-UDF` → Superseded; `D-OIF-1` → Superseded (retirement plan); `D-OIF-1-DEC` → Withdrawn; `LATEST_STATE.md:2389,2393` claims (see §2.5) get a dated correction line, not an edit.

### 2.4 Layer audit (Rubicon / Heckhausen vocabulary)

| layer | owner in code | impersonation found? |
|---|---|---|
| 1 Lifecycle control | `KanbanColumn` DAG (`kanban.rs:98-106`) + `MailboxSoaOwner::try_advance_phase` (`soa_view.rs:311`); Planning = pre-decisional, `Planning→CognitiveWork` = the Rubicon crossing, `CognitiveWork` = actional, `Evaluation` = post-actional, `Commit`/`Plan`/`Prune` = outcomes; `Planning→Prune` = pre-Rubicon veto | none. `Commit`'s calcify step is DECLARED only (`kanban.rs:48-56`) |
| 2 Action semantics | `ActionState` (OGAR `ogar-vocab/lib.rs:676` canonical wire type; contract `action.rs:44` the Rust mirror, unconsumed), `ActionDef`, `KausalSpec`, `ActionInvocation::commit` | `commit` inlines its own RBAC check (`action.rs:283`) — documented design, not drift. `CommitHook` **does not exist** (prose only) |
| 3 Authorization | contract `rbac.rs` + `lance-graph-rbac` (types + un-called kernel); live gate = string-keyed `Policy` | none on lifecycle. The ClassView∧role fold is unbuilt |
| 4 Execution/reasoning | `ogar-loco` (live), `ogar-r2il` (probe), planner thinking (linked, uncalled) | none: zero authorization words in loco/r2il |
| 5 Storage/query | Lance / DataFusion (B1–B9 in the census) | the one policy hook (`patient.rs:146-150`) is on the undeployed path; DataFusion owns no lifecycle state |

**The one category error worth naming:** the prior plan treated *authorization* (layer 3) as a *storage/query* concern (layer 5) because the only code that existed lived there. The census shows layers 3 and 5 are currently **disjoint**, not layered — where ClassView×mask genuinely governs columns (MedCare `views/`, atlas/graph reads of the resident bake), DataFusion and Lance are not in the path at all.

### 2.5 Stale prose contradicted by the tree (corrected in this PR where the file is ours and non-append-only; dated correction lines elsewhere)

- `a2ui-rs/CLAUDE.md` § "RBAC is real": "`ClassRbac::field_mask` is being retyped to `WideFieldMask` … `WideFieldMask::ALL`" — `rbac.rs:176` still returns `FieldMask` (u64); `WideFieldMask::ALL` does not exist (`class_view.rs:275` has only `EMPTY`; `full_for` `:360`). *(a2ui-rs repo; not touched here — filed for that repo.)*
- `OGAR/crates/ogar-a2ui-frame/src/lib.rs:32`: states the C1.4 retype as done. *(OGAR repo; not touched here.)*
- `LATEST_STATE.md:2389` lists `commit_via` / `OgarRbac` / `graph-flow-action::dispatch_via` as shipped enforcement — the types exist, the callers do not; `rs-graph-llm` is not on disk. `:2393` "`impl ClassRbac for OgarClassView`" — no such impl (`ogar-class-view/lib.rs:399` impls `ClassView` only). *(Append-only: dated correction line added in this PR.)*
- `.claude/v3/knowledge/mailbox-kanban-model.md:30`, `.claude/v3/COMPONENT-MAP.md:79` ("EXTEND"), `.claude/v3/knowledge/write-on-behalf.md:8` (`ACTOR-OWNED`) and `:68`: present `kanban_actor.rs`/`KanbanActor` as the structural owner / pending W1 work — deleted 2026-08-05 (`kanban_actor.rs:1-27`; `LATEST_STATE.md:1824,1860`). *(Corrected in this PR: dated supersession notes, in place.)*
- This plan's own §0 row 1 and `INTEGRATION_PLANS.md` entry: "the blocker is the UDF body" — true as far as it went and still one generation behind; superseded by this §2.
- `.claude/patterns.md:89` names `NotYetWiredHashUdf` as a live pattern — left as-is until the removal PR lands, then updated in that commit.

### 2.6 What this plan does NOT do (hard constraints honoured)

No hash body, no hash choice, no new actor/controller, no new lifecycle carrier, no new policy VM, no SoA widening, no RBAC moved into loco/r2il, no Rubicon state moved into RBAC, no deletion in this PR, and DataFusion registration was never counted as ownership evidence.

## 3. D-OIF-4 / D-OIF-5 — Σ-propagation: the batched kernel and the hop probe

### 3.1 Why "hardware backends" is the wrong shape

- Kernel: `Spd2 { a, b, c }` f64 (`sigma_propagation.rs:113-117`); `ewa_sandwich` = 8 products for `P = M·Σ`, 8 for `R = P·Mᵀ`, then `b = 0.5·(r01 + r10)` (`:210-227`). Byte-identical copy in `jc/src/ewa_sandwich.rs:185-199`.
- AMX: `hpc/amx_matmul.rs` exposes 16×16 tiles (`tile_dpbusd`, `tile_dpbf16ps`, `:288-359`); `simd_amx.rs` exposes u8×i8 VNNI dot/matvec. Neither has a 2×2 f64 shape. MKL `dgemm` on 2×2 is dominated by call overhead. **Rejected for this kernel** — recorded, not deferred.
- The 3×3 analogue exists as a jc probe (`ewa_sandwich_3d.rs`, `Spd3`, `:378`); the n×n idea is `IDEA-PILLAR5PLUS-HIGHER-DIM-SPD`. AMX becomes plausible only at ≥16×16 in **bf16**, and bf16 rounding on the SPD cone can lose PSD — that is its own falsifier and **out of scope** here.

### 3.2 The real lever, and the real gap

- `F64x8` carries `Add`/`Mul` on **every** backend with no FMA in the path: AVX-512 `impl_bin_op!(F64x8, Mul, mul, _mm512_mul_pd)` (`simd_avx512.rs:446-448`), AVX2 (`simd_avx2.rs:973-987`), NEON (`simd_neon.rs:1019-1047`), scalar via `impl_float_type!(F64x8, f64, 8, …)` (`simd_scalar.rs:510`). `mul_add` exists (`:376`) and **must not be used** — it would break bit identity with the scalar kernel.
- So the batched kernel is ordinary consumer code over typed wrappers: 8 edges per lane-op, same 12-mul/6-add order per lane. **No ndarray change is needed; the STOP rule is not triggered.**
- **Home:** `crates/jc` (deps `ndarray`, `Cargo.toml:21`, "MANDATORY"). Not the contract (zero-dep). Name: `ewa_sandwich_x8(m: &[Spd2; 8], sigma: &[Spd2; 8]) -> [Spd2; 8]` plus a slice driver `ewa_sandwich_batch(m: &[Spd2], sigma: &[Spd2], out: &mut [Spd2])` handling the tail with the scalar kernel.
- **The gap that decides the sequencing:** `grep -rn "ewa_sandwich(" crates/` outside `jc/` and `sigma_propagation.rs` returns **nothing**. The four files that mention it — `cognitive-shader-driver/src/bindspace.rs:41`, `mailbox_soa.rs:129`, `perturbation-sim/src/splat.rs:19`, `lance-graph-arm-discovery/src/aerial/codebook.rs:16` — do so in **doc comments**. A faster kernel for a function nobody calls is `E-A-RULED-HOME-NEEDS-A-FIRST-CONSUMER-OR-IT-IS-A-VACANCY-1` verbatim. Therefore D-OIF-4 ships **only paired with D-OIF-5**, whose result decides whether the first consumer (§3.3) is sound.

### 3.3 The named first consumer (NOT built in this plan)

`MailboxSoA::apply_edges` (`mailbox_soa.rs:348-372`) receives `(row, CausalEdge64)` deliveries and today updates only `energy[row]` and `plasticity_counter[row]`; `sigma[row]` is never advanced. The contract's own planned use-site is "B4 shader-driver-sigma-propagate … propagate `sigma_path = ewa_sandwich(...)` along the resonance chain" (`sigma_propagation.rs:78-80`). With a codebook (§4), one hop is: `Σ' = sandwich(M, codebook[sigma[row]])`, then `sigma[row] = nearest(codebook, Σ')`. Two things block wiring it, and this plan does not pretend otherwise:

1. **Where `M` comes from.** A `CausalEdge64` carries no matrix; its v2 spare is **3 bits** (`edge.rs:194, 557` — bits 61-63), so it cannot carry a 256-entry index. Candidates: the *source* row's own `sigma` (M = codebook[σ_src]), or a per-edge-class M. **DECISION D-OIF-5-DEC, operator's.**
2. **Whether re-quantizing to k=256 after every hop keeps concentration.** That is D-OIF-5.

### 3.4 D-OIF-5 — the hop re-quantization probe (jc)

Pre-registered, and **mirrored on the certified comparison** (`crates/jc/src/ewa_sandwich.rs:294-343`, `cv_measured` / `cv_tightness`) — not a new gate shape:

- **Hop-matrix source:** the SAME seeded generator the certified pillar uses (`ewa_sandwich.rs:95` Box-Muller, `:215` random rotation θ; SplitMix64 state, seed fixed) — one `M` per hop per path.
- **Paths:** `N_PATHS ≥ 1000` per depth, `n ∈ {1, 2, 4, 8, 16}`, seed Σ as in the pillar.
- **Two arms per path:** (i) exact — `Σ_k = ewa_sandwich(M_k, Σ_{k-1})`; (ii) re-quantized — after every hop, `Σ_k ← codebook[nearest(codebook, Σ_k)]` with `nearest` in the affine-invariant metric `d(A,B) = ‖log(B^-½·A·B^-½)‖_F` (the probe's own metric, `sigma_codebook_probe.rs:24`).
- **Reduction (the part the first draft got wrong):** `log_norm_growth` returns an ABSOLUTE change in `‖log Σ‖²_F` (`sigma_propagation.rs:254`) while `pillar_5plus_bound(n)` returns a **coefficient of variation** (`:274-285`); they are not comparable directly. So, exactly as `ewa_sandwich::prove` does: per depth and per arm, take `‖log(Σ_n)‖²_F` over all paths, compute `mean` and `std`, `cv_measured = std / mean`, `tightness = cv_measured / pillar_5plus_bound(n)`, **PASS if `tightness ≤ 1.75`** (`ewa_sandwich.rs:325-343`). Report both arms' tightness and their ratio (does re-quantization widen concentration, and by how much).
- **Verdict semantics:** PASS on arm (ii) at every `n` is **evidence that re-quantization preserves concentration in THIS synthetic model** — it is not proof that `MailboxSoA::apply_edges` is sound. Wiring σ-advance into `apply_edges` stays its own deliverable with its own implementation and consumer tests, outside this plan, and still needs `D-OIF-5-DEC`. Any FAIL at depth `n` shelves that consumer with the number.
- **Arms by artifact:** arm 1 runs on the probe's own k-means codebook (`sigma_codebook_probe.rs`, `N_EDGES=10_000, K=256, 100 iters, SEED` fixed at `:49-52`), so it does not wait on D-OIF-2; arm 2 re-runs on the real codebook once D-OIF-2/D-OIF-7 exist.

### 3.5 D-OIF-4 gates

| # | assertion | disable |
|---|---|---|
| G1 | `ewa_sandwich_x8` is **bit-identical** (`to_bits`) to the scalar contract `ewa_sandwich` on 1000 sampled SPD pairs incl. non-zero off-diagonals (the existing test's anti-vacuity guard) | replace one `*` + `+` with `mul_add` |
| G2 | the slice driver's tail path (len % 8 ≠ 0) is bit-identical too | drop the tail loop |
| G3 | a bench (`jc/examples`) prints ns/edge for scalar vs x8 at N = 65 536; the number is **pinned after measurement**, never predicted | — |
| G4 | `jc-proof.yml` still green (the certified kernel is untouched) | — |

## 4. D-OIF-0 / D-OIF-2 / D-OIF-3 — the Σ codebook that three files point at and none holds

### 4.1 The finding

Every σ byte in the substrate is an index into a codebook that **does not exist**:

- `sigma_propagation.rs:73` — "indexing into a 256-entry static `SigmaCodebook` of `Spd2`".
- `contract/src/splat.rs:308` — "The full SigmaCodebook lives in **lance-graph-cognitive**".
- `arm-discovery/src/aerial/codebook.rs:16-19` — "built and certified **offline by `crates/jc`**".
- `grep -rn SigmaCodebook crates/` → those two doc lines. `crates/lance-graph-cognitive/src` → zero hits. `jc` has the **viability probe** (`sigma_codebook_probe.rs`) and no builder/emitter.

Three claimed homes, zero implementations. The column's writers: `backing.rs:310` (write-shim loop) and one test (`:347`, writes `9`). Readers: two planner examples dumping the byte (`blw_tenant.rs:190`, `blw_rows.rs:289`). Every production row has `sigma = 0`, documented as "untrained / first centroid" (`bindspace.rs:54`) of a codebook with no centroids.

**Two numbers, one claim.** `bindspace.rs:39,57` cite "R²=0.9949 at k=256 (#288)". `arm-discovery/src/lib.rs:13` and `codebook.rs:18` cite "ρ=0.9973". The probe computes **R² in log-Euclidean space** (`sigma_codebook_probe.rs:27, 317`, PASS ≥ 0.99). `0.9973` appears elsewhere as the 3σ coverage constant and the ADC-cosine Spearman band (`probe_adc_cosine_head_to_head.rs:6`) — a different quantity. D-OIF-0 settles which number the codebook claim actually rests on.

### 4.2 Two provenance stories for σ — DECISION D-OIF-2-DEC (operator's)

| story | where it is written | what it implies |
|---|---|---|
| **Fitted** — σ = nearest of 256 k-means centroids over observed Σ | `sigma_codebook_probe.rs` (Lloyd's, affine-invariant Riemannian metric, R² gate) | needs a per-row observed Σ at write time; the codebook is trained once, offline, in jc (float to build, index to use — the CAM-PQ doctrine, `codebook.rs:19-22`) |
| **Declared** — σ = lookup from the typed value's `(PropertyKind, Marking, SemanticType)` + value range | `sigma_propagation.rs:74-77` ("B3 transcode-sigma-assignment") | needs a mapping table, not k-means; the "codebook" is authored, and the R² probe is irrelevant to it |

They are not compatible as the *same* v1. Recommendation: **fitted** for the entries, **declared** only as the assignment *fallback* when no Σ is observable at write time — but that is a design choice with consequences for every consumer, so it is stated and not taken.

### 4.3 D-OIF-0 — re-run the viability probe on the current tree (first, cheap, unconditional)

`cargo run -p jc --example sigma_probe` (the `[[example]] sigma_probe`, `jc/Cargo.toml`). Record R², the PASS verdict, and the recommendation string it prints (`sigma_codebook_probe.rs:342-360`) in `AGENT_LOG` + this plan's §9. Reconcile 0.9949 vs 0.9973 in one sentence with the source of each. ~seconds of compute; no decision depends on skipping it.

### 4.4 D-OIF-2 — the codebook TYPE (contract) + BUILDER/EMITTER (jc) + INSTANCE home

- **Type, in the contract (zero-dep, LE law):** `SigmaCodebook([Spd2; 256])`, `from_le_bytes(&[u8; 256·24])` / `to_le_bytes`, `entry(u8) -> &Spd2`, `nearest(&Spd2) -> u8` (affine-invariant metric via the existing `Spd2::{log_spd, sqrt, pow, eig}` `:130-176`), and a **boundary check** on load — every entry `is_spd(eps)` (`:187`), per the module's own contract that SPD is checked "at boundaries (codebook load, runtime gate)" (`:108-109`). 6 KiB.
- **Builder, in jc:** the k-means the probe already runs, refactored so the emitter writes the frozen table (float to build). Emits **LE bytes + a tag**. **No digest pin, no digest gate** — identity = `K == 256` ∧ all-SPD ∧ `det > 0` ∧ the tag. The operator law on internal pins is absolute.
- **Instance home:** the contract carries the TYPE; the loaded INSTANCE must live where files can be read and where `set_sigma` is called — `cognitive-shader-driver` (owner of `MailboxSoA.sigma`), **not** `lance-graph-cognitive` (which the splat proxy names but which holds nothing). Recommendation only; the crate that owns the column should own the loader. **Owner: W-7 / `D-OIF-7` (§7) — not W-2**, whose scope is the contract type and the `jc` builder only.
- Gates: G1 `from_le_bytes(to_le_bytes(cb)) == cb` bit-exact; G2 a non-SPD entry is **refused** at load (can-fire) and a valid table is not (stay-silent); G3 `nearest` returns the entry's own index for each of the 256 entries (identity), and a perturbed entry returns its unperturbed neighbour (discrimination); G4 the emitted table from the probe's fixed SEED is byte-stable across two runs (determinism).

### 4.5 D-OIF-3 — the first writer of a non-zero σ (gated on D-OIF-2-DEC)

Whichever provenance the decision picks, the first writer is the ingest/write path that today calls `BackingStoreWrite::set_sigma` in its loop (`backing.rs:310`) with a constant. Falsifier: after ingesting a batch with ≥2 genuinely different Σ shapes, `sigma` holds ≥2 distinct non-zero values (can-fire) and a batch of identical shapes holds one (stay-silent). **Not built until the decision lands**; the plan carries both shapes' one-line specs so the brief is a paste.

## 5. What this plan REJECTS, with the reason on record

| item | verdict | why |
|---|---|---|
| AMX / MKL for the 2×2 sandwich | rejected | shape mismatch: 3 scalars vs 16×16 tiles; call overhead ≫ 18 flops (§3.1) |
| 9-byte `CausalEdge64 + u8` sidecar | superseded | the u8 landed as a SoA column (§0); a second projection of the same byte is `SECOND-PROJECTION` |
| SchemaSidecar Block 14/15 | superseded | same |
| Σ index inside the `u64` | impossible | v2 spare is 3 bits (`edge.rs:557`) |
| wiring σ-advance into `apply_edges` | deferred, gated | needs D-OIF-5 PASS **and** D-OIF-5-DEC (the M source) |
| bf16 ≥16×16 SPD on AMX | out of scope | own falsifier (PSD under bf16 rounding) — file it as an idea, do not build |
| a CI gate on IDEAS.md staleness | rejected | a ledger that is legitimately quiet would fail it — a gate that fires on quiet carries no information (`E-ANTI-EIGENVALUE-…`). A dated census line in `LATEST_STATE` instead. |

## 6. Sequencing

```text
D-OIF-0  probe re-run (jc, seconds)            ── unconditional, first
D-OIF-1  policy_hash_v1 RETIRED (ruling A)   ── withdrawn 2026-09-05; no worker, no decision
D-OIF-4  ewa_sandwich_x8 (jc)  ┐
D-OIF-5  hop re-quant probe   ┘ paired         ── D-OIF-5 arm 1 needs no artifact
D-OIF-2  SigmaCodebook type + builder + home   ── needs D-OIF-2-DEC (fitted vs declared)
D-OIF-3  first σ writer                        ── needs D-OIF-2 + D-OIF-2-DEC
D-OIF-7  loaded instance + loader (cognitive-shader-driver) ── needs D-OIF-2; W-7
D-OIF-6  IDEAS ledger: 4 entries + 3 flips     ── THIS PR
```

Two decisions stand between the plan and the workers: **D-OIF-2-DEC**, **D-OIF-5-DEC** (D-OIF-1-DEC withdrawn under ruling A, §2). D-OIF-0, D-OIF-4, D-OIF-5 (arm 1) and D-OIF-6 need none.

## 7. Worker briefs (Sonnet, edit-only; the orchestrator compiles once)

Every brief starts with `sonnet-worker-guardrails.md` §1 **verbatim**, then: *Read `.claude/board/AGENT_LOG.md` first; do NOT write it — leave your record in your own tag-file. Do NOT run `cargo` or `git`. Do not claim it compiles or that tests pass — you did not run them. Your file scope is exactly the files named; another worker owns everything else.*

- **W-1 (D-OIF-1)** — **WITHDRAWN 2026-09-05 (ruling A, §2).** No worker. The first removal PR (§2.3 bullet 1) is orchestrator work after grace-period sign-off, not a Sonnet brief.
- **W-4 (D-OIF-4)** — files: `crates/jc/src/ewa_sandwich.rs` (append only below the existing kernel), `crates/jc/examples/ewa_sandwich_bench.rs` (new). Spec: §3.2; **no `mul_add`, ever**; same op order per lane as `:185-199`. Tests: §3.5 G1-G2 with the disable named. STOP if any test needs a tolerance — that is a bug, not a tolerance.
- **W-5 (D-OIF-5)** — files: `crates/jc/src/sigma_hop_requant_probe.rs` (new), `crates/jc/src/lib.rs` (one `pub mod` line + one pillar-table entry, `lib.rs:150-185` shape). Spec: §3.4. Output: the table `n → (exact growth, requantized growth, bound, PASS/FAIL)` printed by `prove`.
- **W-2 (D-OIF-2, after D-OIF-2-DEC)** — files: `crates/lance-graph-contract/src/sigma_propagation.rs` (type + loader + boundary check, zero deps), `crates/jc/src/sigma_codebook_probe.rs` (emitter arm). Spec: §4.4, gates G1-G4.
- **W-7 (D-OIF-7, after D-OIF-2)** — files: `crates/cognitive-shader-driver/src/sigma_codebook.rs` (new — the loaded INSTANCE: read LE bytes + tag, construct the contract `SigmaCodebook` through its boundary SPD check, expose `&'static`/`Arc` access to the driver), `crates/cognitive-shader-driver/src/lib.rs` (one `pub mod` line). Spec: §4.4 "Instance home". Gates: G2 (a non-SPD table is REFUSED at load; a valid one is not) and G4 (byte-stable across two loads) re-run THROUGH the loader, plus a can-fire/stay-silent pair on tag mismatch. Does NOT touch `mailbox_soa.rs` or `backing.rs` — the first writer is D-OIF-3's.
- **W-0 (D-OIF-0)** — no edits: the orchestrator runs `cargo run -p jc --example sigma_probe` itself and records the result (§4.3). Not a worker task.

Orchestrator gates after the fleet lands: `cargo fmt -p <crate>`, `cargo clippy -p <crate> --all-targets -- -D warnings` per touched crate (`-p`, never `--all` — `tesseract-rs`'s lesson), `cargo test -p <crate>` with the feature sets named above, plus the two board gates and `supersession_index.py` regenerated LAST.

## 8. Claims in this plan that are STRUCTURAL and must be confirmed by execution

1. `DataFrame::collect()` compiles under `query-lite` (datafusion sans default features). Fallback named in §2.4.
2. `F64x8` `Mul`/`Add` are non-FMA on **all four** backends — read from `impl_bin_op!` (AVX-512) and the operator impls (AVX2/NEON); the scalar macro body was not read line-by-line. G1 of §3.5 is what proves it.
3. `cargo run -p jc --example sigma_probe` runs on the default feature set (it is not `required-features`-gated in `Cargo.toml:44-56`; `goursat_substrate_probe` is).
4. The probe's k-means codebook is exposable to D-OIF-5 arm 1 without a public API change — if not, W-5 gets a one-line `pub(crate)` accessor in `sigma_codebook_probe.rs` and its scope grows by that file.

## 9. Results ledger (append as they land)

- D-OIF-0: _(not yet run)_
