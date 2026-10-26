# D-DT64-5: Versioned DTO behavior → native method bindings, time-series and ONNX leaf compatibility

> **DT64↔Quack coordination Q0 (2026-10-26):** [Minimal source-attested borrowed binding contract, capstone #1799](https://github.com/AdaWorldAPI/lance-graph/blob/research/dt64-predicate-holographic-capstone-20261026/.claude/plans/d-dt64-q0-borrowed-quack-binding-seam-v1.md). DT64 references prepared methods and numerical certificates, but never generates or certifies the numerical implementation. D-NAM/EXO-152 owns calibration, source-specific fingerprints and SourceParity versus CertifiedMath; ndarray executes kernels. ONNX/TFLite are external leaves. All first-landing tests require #1792's detached-\`Keep\` 1,032-vs-31 refusal/recompute. Coordination note only; not new implementation authorization.

**Date:** 2026-10-26. **Status:** ENGINEERING PLAN / docs only. **Parent:** DT64-0–4 candidate drafts; D-NAM/EXO-152 is authoritative for native method crosswalk, Quack and external leaf mapping.
**No:** new runtime, ONNX inference interpreter, tensor AST, model retraining, changes to TFLite parity, or unapproved production code.

## 1. Compile the behavior, not the source business system

Source adapters attest the meaning of a typed DTO/relationship at a version and permission scope. A `ClassView` declares fields and a `compute_dag` where appropriate. A prepared method binder resolves an allowed native algebra recipe. Quack/relalg/R2IL/mask-risc/ndarray execute borrowed lanes; ONNX/TFLite names are downstream leaf facades rather than authority for the relational state.

Examples and verdicts:
- `User -> security memberships -> GroupRoles::role_evidence`: Boolean/Distinct relational fold, exact when source policy complete; no ONNX necessary.
- `User -> assigned SKU -> active licence count`: source licence adapter **missing**, so REFUSE; cannot compile a missing relationship into success.
- `SAP CATS entry -> billable ? hours : 0 -> sum`: Quack mask + typed reduction; `billable` UNKNOWN must not map to zero without explicit policy; `hours` numerical exactness per CATS parse and source scale.
- `Odoo sale.order -> partner/allowed_company`: typed relation, ClassView source fields; materialization only at an authorized output terminal.
- `Odoo HR version -> dates -> active employment at time T`: temporal predicate over source intervals, exact with timezone/domain resolution, not speculative neural inference.
- `Time-series feature window -> ONNX scoring model`: numeric method route, explicit model digest/opset/dtype/layout/rounding, calibrated or bounded accuracy, result `Computed` not source-observed.

## 2. Prepared recipe identities

One possible cold receipt, *using D-NAM vocabulary*, not new ABI:
```text
RecipeBinding {
 source_schema_and_predicate_catalogue_versions,
 method_or_rule_identity, scope_and_permission,
 source/owner/snapshot identity, focus coordinates,
 typed-input lanes and ordering witnesses,
 numerical policy {exact|bounded|calibrated|refuse},
 relevant math dialect {relalg/r2il/ndarray/onnx_leaf/tflite_leaf},
 dtype/shape/axes/unit/time window,
 code space + quantization scales/zero points (when needed),
 runtime-free validated entry point and valid input lifetime,
 output semantic kind {Observed? NO, DerivedRelation | ComputedNumeric | Hypothesis}
}
```
An independently reusable *law or compiled recipe* does not authorize a cached input mask, source data, model output or per-tenant scope. D-DRMC #1740 already measures recipe reuse across fresh bindings (lab). The generalized source-attested production Binding Store is not built.

### Canonical native method crosswalk

| DTO demand | Native path | Semantics | Refusals |
|---|---|---|---|
| direct group membership | Quack/relalg/mask-risc | Boolean mask, row scope | unknown identity, scope |
| transitive effective role | precomputed/compiled closure (dir-rmap) | Boolean/Distinct with support witness | unread policy, stale mapping |
| SQL `IS NULL`, K3 WHERE | existing relalg/K3 | TRUE/FALSE/UNKNOWN | false conflation |
| count/sum over admitted rows | mask-risc terminals; ndarray integer | exact in bounded integer range | overflow, missing values |
| time-windowed property | temporal filter + fold | as-of source validity | mixed clocks, incomplete window |
| ONNX Greater/Where/ReduceSum | `lance-graph-onnx` leaf | semantics checked under shape/dtype | broadcast/axes/dtype/overflow |
| TFLite int8 nonlinear | prepared code LUT source-compatible | fixed-point source parity only in tested domain | wrong code space or calibration |
| Float Tanh/Exp | existing ndarray bounded VML | bounded not exact | numeric error outside bound |
| cross-source compare/join | ClassView/Quack binding with source witness | identity-keyed, authorized | foreign positional gather |
| policy evaluation | source policy + scoped role | externally authorized | not derived from `FieldMask` |

## 3. Softmax, FisherZ, Palette and SIMD must NOT leak into the Boolean DT64 bits

The earlier D-TFLP parity work found source-dependent TFLM fixed-point behavior, `Asym` activation −128 valid, conforming `Sym` weights −128 invalid, `SymNaN` −128 sentinel, signed↔unsigned SIMD operand mappings and rounding-specific behavior. These are numerical **execution metadata**, not the DT64 semantic predicate index. A 64-entry predicate catalogue and Palette256 (256 codes) are distinct. Preserve original model weights/scales/zero points, LUT mapping and rounding, no FisherZ→cosine roundtrip unless source mathematics demands it. TFLM source parity and certified mathematical accuracy are different recipe identities. Never claim a source-compatible table validates a mathematical approximation, or vice versa.

## 4. Annotated method examples

**IAM administrator:** `AD:user objectGUID U`, authenticated role policy at revision P, security-group SID chain in snapshot V, `RoleId Admin`, requested resource scope S -> `effective_admin(U,S)`. `false` is valid only if all relevant role-policy groups were read in S. A model may score likely misconfiguration but cannot grant the role.

**Odoo payroll period:** `hr.employee` + valid `hr.version` intervals -> time-indexed numeric hours/wage only with extracted Odoo R2/R13/R16 rules; country/payroll rules absent in the inspected corpus are NOT inferred. Arithmetic needs currency units, rounding, timezone and financial policy, not just a 1-bit presence.

**SAP timesheet eligibility:** `CatsSchema` 23 fields, original nullable billability, group-specific reduction over `hours_logged`, HASH/BAPI order does not become general adjacency. Native Quack and scalar SAP oracle must agree; BAPI materialization remains explicit and never executed in lab.

**Time-series event:** changes to licence assignment (once an authoritative source exists), membership and mailbox status generate sampled features with separate valid and observed times. Typed ONNX native method consumes a bounded window and emits score + model version/digest, not a permission bit.

## 5. Gated tests and fallbacks

- M0: compile `IS_ADMIN` using exact dir-sim role evidence; compare typed closure and independent oracle, null/unread distinction.
- M1: compile billing condition and SUM using original SAP CATS lanes; test unknown billable, negative/overflow/hour parsing and hash/BAPI order.
- M2: Odoo HR valid-interval scalar oracle on synthetic source-shaped data; time coordinate collision and company ACL refusal.
- M3: ONNX leaf wrapper route one admitted numeric method; version/opset/axis/dtype changes must refuse or choose distinct recipe, no hidden fallback.
- M4: two equal-shape models with different source weight digest, codebook and calibration must never share a prepared table.
- M5: source data changes after borrow, escaped Keep stale; re-evaluate or refuse (#1792).
- M6: byte-exact TFLM SourceParity vs mathematically certified numeric arm and SIMD i8/u8 mapping; demonstrate that different declared semantics may legitimately produce different output bytes.
- M7: time-series feature window changes under late arriving observation vs valid-time correction; no retroactive unlabelled mutation.
- M8: forecast output cannot become identity/authoritative edge without typed policy and independently observed evidence.
- M9: compare direct runtime-free native route with any overly general interpreter; count bind, cold prep, hot call, memory, output and refusal cost.

Negative controls: wrong schema version; same local ordinal different global identity; invalid `SymNaN` mapping of Asym −128; normalization under the wrong zero point; ONNX `Where` vs SQL NULL. Report inert controls as inert. No claims of universal ONNX parity.

## 6. Stages, owners and priority

- **5A (docs / now):** this crosswalk, source/predicate/recipe receipt and denial matrix. Owner DT64 + D-NAM.
- **5B (lab / later):** exact role + SAP CATS numeric query plus independent scalar source oracle; no ONNX needed.
- **5C (lab / later):** Odoo method/validity one class, with authorization and time oracle.
- **5D (lab / later):** one ONNX method leaf matching D-NAM; no new operator engine.
- **5E (lab / later):** calibrated numeric & time-series path with explicit approximate method identity.
- **5F (decision):** reject/keep-direct/adopt based on full end-to-end cost, byte parity gates and source/source-policy closures.

**Dependencies:** DT64-1 semantics, DT64-2 source honesty, DT64-3 exact relation selection, EXO-152 D-NAM ownership, D-NFR-1 borrow integration (#1792), EXO-148 for durability only. A self-contained numeric lab does not require durable thought-git.

**Output:** handoff to EXO-50 with exact gates, independent oracle locations, failure matrix, code owner approvals; no implementation authorized by writing this document.
