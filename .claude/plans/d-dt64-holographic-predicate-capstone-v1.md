# D-DT64-0: Versioned DTO Predicate Adapter, 64-slot registry and holographic relational access

**Date:** 2026-10-26. **Status:** DESIGN / docs only. **Decision:** user requests detailed staged PR plans, mappings, falsifiers and EXO-50/.claude coordination. **NOT a production ABI approval.**
**Owner:** DT64 predicate mapping and evidence contract; coordinate existing source owners via EXO-50 before any code.
**References:** [D-QR-CAP / EXO-121](https://linear.app/exo-red/issue/EXO-121), [D-QR-IAM / EXO-123](https://linear.app/exo-red/issue/EXO-123), [D-QR-SAP / EXO-122](https://linear.app/exo-red/issue/EXO-122), [D-QSP / EXO-126](https://linear.app/exo-red/issue/EXO-126), [D-DRMC / EXO-127](https://linear.app/exo-red/issue/EXO-127), [EXO-46](https://linear.app/exo-red/issue/EXO-46), [EXO-148](https://linear.app/exo-red/issue/EXO-148), [EXO-151](https://linear.app/exo-red/issue/EXO-151), [EXO-152](https://linear.app/exo-red/issue/EXO-152), [EXO-50](https://linear.app/exo-red/issue/EXO-50); [#1792](https://github.com/AdaWorldAPI/lance-graph/pull/1792).

## Coordination lock: first common DT64 ↔ Quack seam

[**D-DT64-Q0 borrowed binding seam**](d-dt64-q0-borrowed-quack-binding-seam-v1.md) records the 2026-10-26 cross-session architecture agreement and supersedes any broader reading of the first executable stage in this charter. It is **coordination, not a new CC instruction**. One source-attested semantic binding contains typed source-object/predicate IDs, schema/catalogue version, source owner and actual pinned snapshot, focus witness, optional codebook generation, permission/completeness scope, **borrowed native input lanes** and a link to an independently reusable compiled recipe. This plan introduces **no production Rust struct/ABI**.

- **DT64 owns semantics and binding**, not evaluation, method execution, ONNX, or a physical tensor.
- **Quack/RelAlg/R2IL/mask-risc own relational execution**, ndarray numerical kernels; **D-NAM/EXO-152 owns prepared numerical method identity/certificates**. SourceParity, CertifiedMath, calibration and implementation fingerprint do not inhabit DT64 predicate bits.
- **First landing is narrow:** eight sourced IAM/Exchange/SAP predicate candidates, owner-borrowed algebra, directed (s,p)→o / (p,o)→s / (s,o)→p access, independent scalar oracle and optional measured DR64². Odoo remains an explicitly documented later source-mapping extension, not a forced prerequisite for the first exact fixture.
- **#1792's 1,032 vs 31 stale Keep** falsifier remains mandatory. The compiled recipe may outlive a borrow; a detached Keep cannot be reused across source changes without a complete dependency witness, so recompute/refuse.
- **No DR64³ construction, new runtime, generalized replica or production writes** follow from the Q0 seam.

## Thesis and precise terminology

DT64 = **DTO Tensor 64**, a versioned *semantic adapter plate*: register up to 64 separately typed Boolean predicate mappings in a stable ordered catalogue. Source DTOs from IAM/AD/Entra/Exchange, Odoo and SAP bind into witnessed canonical identities, typed predicates and borrowed lanes. Quack / relalg / R2IL / mask-risc / ndarray operate over admitted native data. DT64 is **not** a replacement database, copied global entity table, new executor, or inference from similar field names.

The proposed `R_v[s,p,o]` is a Boolean relation **only after** source+owner identity, coordinate map, codebook generation, presence, permission and frame consistency are proven. Predicate `p` is a typed semantics tag, not an Odoo field number, ClassId, palette index, thought slot, or universal property-id assertion. The 64-slot registry is a *logical* bound for a pilot, with 0..63 stable ordinals by version; it does **not** itself decide DR64³ axis semantics (NFR-OP-2 open). Per-predicate domain/range and ordering are witnessed; numerically equal local ordinals do not imply identical entities. Inversion is an operation with direction/type, not silent graph edge reversal.

`ClassView::FieldMask` means structural **field presence** (contract/src/class_view.rs). Never repurpose its bits as graph predicate truth. Predicate truth requires its own relation(s), separate existence/known-ness/authorization. Metadata remains above SoA. `contract/src/property.rs::PropertySpec` is an AriGraph SPO property classification, not an existing 64-slot DT64 registry.

## Logical capacity and materialization

64 subjects × 64 targets × 64 predicates = 262,144 Boolean cells = 32 KiB if dense 1 bit/cell. One 64×64 predicate tile = 512 B; DR256² = 8 KiB; DR256³ dense = 2 MiB. These exclude masks for knownness, validity, provenance, evidence, codebooks, indexes, relations and witnesses. DR64³ physical dense/sparse/tiled realization remains unratified; never eagerly allocate a cube to answer an 8- or 23-field question. NFR-OP-2 decides what the third axis is. Gate DR view shape against [d-dr3-holo-onnx-s3a-s3c-v1.md](d-dr3-holo-onnx-s3a-s3c-v1.md) and its supersession by D-NFC.

## Architectural flow

```
AD/Entra/Exchange   Odoo          SAP CATS / business
      |              |                     |
 source DTOs and immutable snapshots, each with its OWN source + version identity
      \______________|_____________________/
                     v
 cold DTO admission -> ClassView field presence -> typed source field/relationship resolver
                     v
 DT64 Predicate Registry (versioned descriptor, 64 slots, owner + domain/range + evidence)
                     v
 legal global-identity ↔ local-focus binding (EXO-46 admission; no value-steered gather)
                     v
 borrowed mask/lanes / DR64² or DR256² when justified / logical DR³ projection
                     v
 Quack + relalg + R2IL + mask-risc + ndarray -> Count/Exists/Keep/verified relations
                     |
 projected directed SPO, uncertainty and negative provenance NEVER erased
                     v
 optional: edge×edge composition; Thought-Git hypothesis/time; ONNX numeric leaf
```

### Integrity axes, never collapsed
- **Frame:** (source-id, dataset/table/snapshot version, schema and binding generation, immutable source/owner seal). A QueryReference is a temporal policy, not a physical checkout (#1792).
- **Focus:** witnessed global→local coordinate mapping, generation, window, typed domain/range. No value-derived cross-table positional gather outside a separately admitted prepare membrane (EXO-46).
- **Scope:** tenant, authorization policy, actor, source permissions; predicate visibility can differ from presence and truth.
- **Predicate interpretation:** 64-entry catalogue identity/version, signature/arity, derived versus observed, operator/semiring semantics; codebook-generation gate.
- **Epistemic observation:** TRUE / FALSE-with-complete-authority / UNKNOWN / NOT-OBSERVED / REFUSED, plus duplicate/bag multiplicity where applicable. Neither null nor absence is false.
- **Time:** valid time, observation time, source revision, persistence/landing and thought-hypothesis time remain different.
- **Law vs data:** D-DRMC compiled recipe reusable under fresh witnessed operands; detached Keep masks are not reusable by cycle stamp alone (#1792).

## Six independent documentation draft PRs (separate branches from main, no hidden stacked commits)

| Gate | Stage | Detailed deliverable | Exit criterion | Hard dependency |
|---|---|---|---|---|
| DT64-0 | This charter & board | scope, precedence, references, 64-slot logical hypothesis, non-goals | source-grounded docs | none |
| DT64-1 | Catalogue and source crosswalk | 64 registered candidate predicates with source DTO fields, types, authority, knownness and negative fixtures | 64 fully classified slots, core 8 exact oracle candidates | DT64-0 review |
| DT64-2 | Source DTO admission | AD/Entra/Exchange + Odoo + SAP identity reconciliation, role/mailbox/license/cross-tenant contrasts | cross-source fixtures fail closed, no writes | DT64-1 |
| DT64-3 | Relational inversion and DR focus | directed SPO projections, Boolean 64×64, conditional 64³, OR-AND vs bag, borrowed lanes, cost | scalar/Cypher or existing dir-sim/Quack parity, negative controls bind | DT64-2; EXO-46 for production |
| DT64-4 | Edge×edge / Thought-Git × time × hypothesis | versioned dependency derivations, replay receipts, evidence/conflicts, proper temporal guards | replay determinism, retract one support, no implicit truth promotion | DT64-3; EXO-148 unblocked for durability |
| DT64-5 | ONNX / deterministic DTO behavior | compile typed predicates and arithmetic into native method bindings; ONNX as external leaf, no new runtime | explicit exact/approx verdict, no false method equivalence, differential oracle | DT64-3; EXO-152 |

**Documentation PR creation is not stage implementation or authorization.** Implementation after a separately accepted falsifier gate, with each stage rebased on merged main.

## Primary open decisions (named owners, no manufactured closure)

1. **DT64-OP-1:** Does 64 designate predicate catalogue ceiling, per-tenant registry, per-focus availability, or a versioned palette? Proposed: versioned registry, stable order; operator to ratify. *Owner*: DT64/ontology.
2. **DT64-OP-2:** Are third-axis cells predicate ordinals, relation ordinals, or property palette planes? This is already NFR-OP-2; do not preempt. *Owner*: DR physical ABI.
3. **DT64-OP-3:** Cross-system identity reconciliation: proven source key joins only, no email/UPN/PERNR equality assumption; define opaque signed mapping witness and collision protocol. *Owner*: IAM / Odoo / SAP source adapters.
4. **DT64-OP-4:** Negative authority, permission-scope and completeness witnesses for FALSE versus UNKNOWN. *Owner*: auth/identity.
5. **DT64-OP-5:** Escaped Keep and cross-source snapshot consistency (#1792): not resolved by a MailboxSoA cycle/gen counter; mutable public fields. *Owner*: source/mailbox and physical Lance owner.
6. **DT64-OP-6:** Direct field mask vs 512 B relation tile vs 32 KiB cube; measure full prepare+hot+terminal cost, no forced materialization. *Owner*: D-QR/D-DRF.
7. **DT64-OP-7:** ThoughtCommitId/DurableRef still absent (EXO-148). No assertion of replay/persisted identity without that authority.
8. **DT64-OP-8:** Odoo model extraction completeness and ACL, SAP CATS source ≠ generic SAP IAM; never invent a native SAP entitlement DTO without a source anchor.
9. **DT64-OP-9:** Exactly what makes local relational access 'holographic': demonstrate lossless directional reconstruction from sufficiently witnessed local projection, characterize incomplete focus; no claim of physical optical holography or general compressed recovery.
10. **DT64-OP-10:** ONNX/time series numerics vs exact Boolean relations, NULL, signedness, codebook validity and quantization; law admission separate from data provenance (EXO-152).

## Global kill tests
- Identical local index, different globally identified principal: refuse.
- Unread group/tenant scope versus explicit negative: UNKNOWN not FALSE.
- Direct vs nested group, security versus distribution: not a single indiscriminate MEMBER_OF closure.
- Cloud mailbox: match/ambiguous/awaiting backsync not coerced to HAS_MAILBOX TRUE.
- Odoo `res.users` vs `hr.employee` via unverified partner/email: refuse.
- SAP client+PERNR vs AD GUID coincidentally equal integers: refuse.
- Two supporting role chains: retract one and effective role remains TRUE.
- Same recipe, changed source, changed codebook, changed scope, same revision number from another source: refuse detached intermediate.
- 64×64×64 tiny synthetic projection whose directed inverse swaps Subject and Object must not pass.
- 8 properties, 23 fields: direct mask beats cube overhead; KEEP_DIRECT, not forced DR³.
- ONNX/MLIR changes dtype/axis/rounding or output semantics: separate method identity or REFUSE.

## Evidence and non-goals

Existing: IAM `dir-rmap-probe/tests/classview_dr.rs` compiled role closure and speed (lab), SAP `lance-graph-sap/src/{schema,bind,edge}.rs` 23 fields and ordered hash/BAPI, Odoo extracted blueprint corpus, DR logic/Quack lab, #1792 borrow-vs-escaped refusal. Missing: DT64 registry implementation, Odoo method fold, SAP↔IAM↔Odoo trusted identity join, general DR³ physical layout, cross-frame binding store, cross-source snapshot attestation and proof checker.

No live SAP/Odoo/AD/Entra connectivity, provisioning, license changes, account mutation, BAPI writes, rollout, authorization shortcut, runtime graph engine, ONNX interpreter, new FrameStore, or automatic claim that every DT64 bit is a fact.

## Agent handoff

Source-first review and proposed falsifier protocol: [d-dt64-source-first-claude-code-handoff.md](../prompts/d-dt64-source-first-claude-code-handoff.md). No implementation or production authorization is conferred by this prompt.


## DT64-QB coordination precedence: narrow source-attested borrowed binding (2026-10-26)

**Controlling first executable seam:** [D-DT64-QB-0](d-dt64-quack-borrowed-binding-seam-v1.md). This section narrows the six broad docs-only stages above: DT64 is a *versioned semantic binding-store adapter*, not a new executor, method evaluator, numerical certificate owner, source data replica or mandatory physical tensor. Quack/RelAlg/R2IL/mask-risc execute deterministic relational operations; ndarray executes native numerical kernels; D-NAM/EXO-152 owns admitted numerical methods and SourceParity/CertifiedMath certificates. DT64 references those methods, never implements their execution or embeds Palette256 numeric codes into 64 semantic predicate slots.

**One conceptual source-attested binding:** typed source-object and predicate ID; source schema and DT64 catalogue version; owner + actual immutable physical snapshot; typed global/local coordinate witness; codebook generation if present; independent permission and read-completeness scope; borrowed input lanes with source lifetime; reference to separately reusable compiled mathematical recipe. ClassView field-presence masks, local DR ordinals, semantic predicate slots and numerical Palette256 codes remain separate. Nominal mailbox owner is provenance, not ACL. `QueryReference::at` horizon is not a physical dataset checkout.

**First executable proof, only if separately authorized:** eight source-grounded IAM/Exchange/SAP predicates, catalogue candidates 08 direct membership, 11 security group, 12 mail-enabled group, 13 group-role grant, 14 effective role, 17 remote-mailbox recipient, 18 confirmed cloud mailbox and 32 CATS entry→scoped PERNR. Preserve unary classifications as unary; test directional `(s,p)→o`, `(p,o)→s`, `(s,o)→p` on binary predicates only. One source-shaped synthetic fixture, one independently authored scalar oracle, one live immutable borrowed Quack evaluation, optional *cost-justified* DR64² Boolean relation view. No license truth without real Entra source. No fabricated Odoo/AD/SAP principal equivalence.

**#1792 source-boundary gate:** now MERGED, but does not provide escaped-selection attestation. In-borrow Quack runs without redundant mutation checks; reusable recipe can be rebound against fresh source; detached `Keep` cannot reuse by caller cycle or recipe hash. The 1,032-old / 31-fresh predicate selection counterexample must refuse/recompute on source mutation. Source public fields, reset+same-cycle ABA and distinct owners with equal stamps make a lone generation/cycle insufficient. Physical source snapshot and complete mutation witness remain OPEN.

**Dependency correction:** NFR-OP-2 DR³ physical third axis, EXO-148 durable Thought-Git ID types, Odoo method, ONNX/TFLite leaf, numerical code-space and physical 64³ cube are **not blockers for the first eight-binding Quack lab**. Retain them as independently gated successors. No production implementation, ABI change, cache, writer/WAL, connector access or merge is authorized by this amendment. Coordination uses EXO-50 append-only comments, not shared mutable messages.
