# D-CGMM independent falsifier matrix: evidence, demand, field roles, actions and native execution

> **2026-11-04. ALL NEW CASES PROPOSED/NOT RUN.** Historical #1944 MeTTa/XLOG/Crepe bounded parity, #1955 borrowed SoA and #1956 Abakus correctness are predecessor evidence, not new CGMM test results. [Stages](d-cgmm-staged-pr-charter-v1.md), [lazy Codegraph](d-cgmm-lazy-codegraph-fact-demand-v1.md), [Action/Edge](d-cgmm-action-relation-schema-v1.md), [EXO-155](https://linear.app/exo-red/issue/EXO-155).

**Independent oracle rule:** source-of-truth fixture carries stable producer IDs, per-source revision and compiler-grounded facts. Oracle enumerates those original facts, tenant policies and typed value rows without calling Codegraph's `Cube::partners`, candidate Codegraph binding, scratch normalizer, RelAlg lowering, or issued method binding. At least one deliberate disable per stage must make the oracle/representation/refusal check red. An empty expected result is insufficient unless a distinct positive control also passes.

**Source grades:** compiler-confirmed, possible dispatch, unresolved dynamic, missing producer, inferred candidate, owner-admitted. The candidate must not silently use a stronger grade than its input.

## F. Source/Kythe identity and loss (F1–F8)

| ID | Nonvacuous fixture and oracle | Falsifier / expected refusal |
| --- | --- | --- |
| F1 | Two identical symbol names in different crates, source IDs distinct; source-ID oracle gives two nodes | Intern by name only → wrong identity or typed conflict |
| F2 | One method declaration with source span, source revision and producer digest | Stale source span or swapped revision → refusal |
| F3 | Five CALLS, two MAY, one dynamic UNKNOWN; expected grade masks differ | Promote MAY to CALLS; grade oracle goes red |
| F4 | A method reads x, writes y and executes ordered effects (synthetic complete producer) | Drop WRITES/order while claiming complete coverage → fail |
| F5 | Producer omits a write-effect column and marks PARTIAL | Treat omission as pure/no-effect → incorrect admission; ABSTAIN required |
| F6 | Cross-language sources with verified node identity/provenance and one missing Ruby source | Infer Rails method effects from Odoo shape → fail |
| F7 | Codegraph symbol `Pos` changes on source refresh while stable source identity does not | Store `Pos` as durable MethodId → stale identity refusal |
| F8 | Missing/invalid binary schema or placeholder `?` symbol | Typed intake refusal; no partial mutation of committed cube |

## D. Datalog demand / lazy Codegraph (D1–D10)

| ID | Nonvacuous fixture and oracle | Falsifier / expected refusal |
| --- | --- | --- |
| D1 | Root f with two confirmed callees, five unrelated source edges | Demand only f confirmed; unrelated edges not projected or cached |
| D2 | Two query roots share one neighbor, repeated Q=1/2/16/256 | Result equality with eager query; measure route/bind count and costs separately |
| D3 | Same root with one MAY-dispatch target | Demand confirmed-only excludes MAY; include-possible includes it but grade stays POSSIBLE |
| D4 | Dynamic unbounded source call exists | Closed-world NOT/CALLS absence interpreted as proof of safety → refusal |
| D5 | Crepe plan `let` drift fixture against `crepe!` independent output | Unverified `crepe_plan!` index certificate must not be trusted |
| D6 | XLOG magic-set compile report for query with demand and negation | Lost polarity/stratification metadata yields NOT_COMPARABLE/REFUSE, not silently executable |
| D7 | Source generation changes between demand preparation and query | Stale demand memo/hot mask must refuse or recompute |
| D8 | Bound/free direction reversed in one relation | One result differs; deliberate direction reversal makes oracle red |
| D9 | Boolean Set versus counting Bag on duplicate edges | Distinct support stays, total count differs; no multiplicity collapse |
| D10 | Recursive SCC request and cycle with 3 nodes | No admitted recursion driver → UnsupportedRecursion, not guessed closure |

## V. ClassView / FieldViewDef / constraint scratch (V1–V10)

| ID | Nonvacuous fixture and oracle | Falsifier / expected refusal |
| --- | --- | --- |
| V1 | Two field names equal under different ClassId/schema bases | Scratch cannot equate field position by label |
| V2 | Full behavior classid lens known; concept half alone insufficient | Missing `lens_fields(full_classid)` must refuse, never fallback |
| V3 | 109-field Odoo-like wide class with one field at position 100 | Reusing 64-bit position or truncating → fail |
| V4 | FieldMask indicates present zero, source semantics preserve present zero | Treat presence as sign/value → fail |
| V5 | Scalar FieldViewDef borrowed canonical lane and same ClassView basis | Any copied source-value column in admitted read path triggers anti-copy checker |
| V6 | Independent type conflict: source I32, candidate U32 signed compare | Refuse instead of auto-coercing absent/negative values |
| V7 | Same schema, changed source effect facts | Scratch invalidated; admitted unchanged field basis remains stable |
| V8 | Scratch eviction and recomputation | Identical OFFER/REFUSE/ABSTAIN, no mutation of admitted DT64 |
| V9 | Two candidates overlap in source span and disagree on method identity | Preserve conflicting candidates, do not last-write-wins |
| V10 | Model/NNUE changes candidate order but not ground facts | Scores do not change admission or unknown grade |

## R. OGIT/OGAR/RelAlg relationship/action semantics (R1–R10)

| ID | Nonvacuous fixture and oracle | Falsifier / expected refusal |
| --- | --- | --- |
| R1 | M:N Worker W1→P1/P2, W2→P2 with independent global-ID edges | Join on local ordinal gives wrong answer |
| R2 | Duplicate W1→P2 edge | Boolean support and Bag count diverge as expected |
| R3 | Tenant A W1/W2/P1/P2, tenant B W3/P3, forged A→B edge | Dropped tenant check leaks edge; typed unauthorized refusal |
| R4 | Inverse traversal not explicitly declared | Reversing (W1,P2) into (P2,W1) unsupported unless independently admitted |
| R5 | Two ActionDef named Approve with different ordered side effects | Same name alone cannot share MethodBinding |
| R6 | Missing source write/effect for SAP/Rails comparison | ABSTAIN rather than equivalent or nonexistent |
| R7 | Verified U32 status==2 on one source lane | RelAlg U32 equality matches independent scalar oracle |
| R8 | Verified I32 amount>=100 with negative and absent values | RelAlg threshold matches independently for admitted two-valued domain; unknown refuses |
| R9 | RelAlg `join_body::is_alpha_acyclic` classifies acyclic M:N join | It must not be treated as executed join, only planning metadata |
| R10 | Source action has conditional WHEN guard, temporal/modal or data-dependent effect | RelAlg pure query cannot execute it; owner-only method OFFER or REFUSE |

## X. Execution/admission/memo economics (X1–X8)

| ID | Nonvacuous fixture and oracle | Falsifier / expected refusal |
| --- | --- | --- |
| X1 | Valid guard-free 2-valued method approved for lab and source scope | Native R2IL exact parity to scalar baseline with source-owned borrow |
| X2 | Guarded WHEN method and explicit Kleene K3 method | `GuardNotExecutable` / `ThreeValuedNotExecutable`; no alternate interpreter |
| X3 | Two methods with different MethodRef/SigRev but same method name | Prepared cache cannot reuse one for the other |
| X4 | Scope/permission revoked while source identity unchanged | Cache hit cannot bypass permission refusal |
| X5 | Source layout/value changed with unchanged class bit basis | Value-dependent derived mask invalidation independent from class basis |
| X6 | Q=1/2/8/16/64/256 cold + warm for memoized recipe | Whole-query cost measured; invalidate and cache-miss controls; no false speedup inference |
| X7 | Valid native Boolean vs Abakus k-of-n words on same workload | Compare exact results and cost; no forced carry default |
| X8 | Unadmitted M:N action or speculative candidate requests WORLD write | No owner permit or Rubicon → refusal, no write effect |

**Total:** 8 F + 10 D + 10 V + 10 R + 8 X = **46 proposed independent gates**, none executed in CGMM-0.

## Evaluation record for every future PR

~~~text
CASE: D-CGMM/<id>
SOURCES: pinned source producer + artifact/revision + code/line
IDENTITIES: source node key, Codegraph Pos (local), ClassView basis, MethodRef if admitted
DEMAND: root/bound-free columns, semantics/grade/permissions, recursion policy
EXPECTED: independent source fact / result mask / explicit refusal
ANTI: deliberate mutation; observed test failure, no vacuous zero
COPIES: owned canonical value bytes, scratch, result and source ingest separated
COST: intake, derive, query, prepare/run, Q, bytes and p50/p95; eager vs lazy
STATUS: PASS | FAIL | PARTIAL | NOT_RUN | NOT_COMPARABLE | BLOCKED_OWNER
ADMISSION: source/ontology/DT64/execution owners, exact EXO links
~~~

**Strict refusal:** No external Datalog optimizer, MORK pathmap, ANN/NNUE candidate, MeTTa rule, class label or Kythe reference **by itself** grants DT64 execution permission or source data ownership.
