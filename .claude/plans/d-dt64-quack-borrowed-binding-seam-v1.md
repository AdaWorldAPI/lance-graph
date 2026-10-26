# D-DT64-QB-0 — DT64 ↔ Quack minimal source-attested borrowed binding

**Date:** 2026-10-26. **Status:** COORDINATION CORRECTION / DESIGN ONLY, not an implemented ABI or approved code PR. **Owner split:** DT64 = semantic source adapter; Quack/RelAlg/R2IL/mask-risc = deterministic relational execution; ndarray = native math kernels; D-NAM/EXO-152 = prepared numerical method admission. **References:** DT64-0 [#1799](https://github.com/AdaWorldAPI/lance-graph/pull/1799), [#1794](https://github.com/AdaWorldAPI/lance-graph/pull/1794)–[#1798](https://github.com/AdaWorldAPI/lance-graph/pull/1798), [#1792](https://github.com/AdaWorldAPI/lance-graph/pull/1792) (MERGED ownership audit), EXO-46, EXO-121/122/123, EXO-126/127/151/152.

## 0. Coordinate architectures by ownership, not machinery

**DT64 describes WHAT is bound, Quack computes WHAT the admitted algebra demands.** DT64 never executes an `IsAdmin` predicate, interprets SQL, evaluates ONNX, calls `action_confirm`, creates a business write transaction, or allocates a mandatory DR64³ cube. A *predicate mapping* describes business meaning + source origin + type + allowed read; a compiled recipe is an independently reusable executable law. Their lifecycle and owners differ.

Canonical data remains under original source/SoA owner. The proposed DT64 binding store is a versioned **semantic registry/descriptor+binding adapter**, not a data replica, persistent mutable FrameStore, separately materialized universal graph, or runtime engine.

This document *narrows* the DT64-0..5 roadmap: a minimal DT64-to-Quack borrowed **adapter lab** is the first executable candidate. All later edge×edge, Thought-Git, time series and ONNX proposals remain downstream and gated. No new runtime or physical tensor is a prerequisite.

### Non-interchangeable identities

| Identity | Owner / meaning | Never conflate with |
|---|---|---|
| `PredicateRef(catalogue/version/slot)` | DT64 semantic relationship/method signature | `ClassView::FieldMask` presence bit |
| `FieldRef` | ClassView schema field position/presence | Boolean truth or permission |
| `GlobalKey(source namespace, kind, key)` | original DTO object | coincident local `u16` ordinal |
| `CoordinateWitness(focus, map)` | admitted global→local focus | DR physical row, palette code |
| `PhysicalSelection/DR tile` | optional Quack/DR execution carrier | versioned business relationship source |
| `Palette256/Codec8 numerical code` | ndarray/D-NAM numerical implementation | any of 64 DT64 predicate slots |
| `PreparedRecipeRef` | compiled reusable native law | detached Keep or source snapshot |
| `SourceSnapshotRef` | source owner’s actual immutable version | `QueryReference::at` reading horizon |
| `ScopeWitness` | separately validated actor/resource permission | field presence / successful identity join |

## 1. One source-attested semantic binding, no proposed production ABI yet

Conceptual **cold** record, preferably a view over existing authoritative types:

```text
SemanticBinding<'source> [PROPOSED DOCUMENT SHAPE ONLY] {
  source_object: (namespace, source_kind, opaque_stable_key),
  predicate: (catalogue_id, catalogue_version, slot,
              typed_domain, typed_range, arity, direction,
              source_field_or_relationship_anchor),
  schema: (source_schema_id, schema_digest, binding_revision),
  source: (owner_identity, actual_physical_snapshot_ref,
           observed_at, valid_time_if_known),
  coordinate: (source_global_to_local_witness, focus_generation, row_order),
  codebook: optional (identity, generation, validity_contract),
  policy: (tenant/resource_scope, actor/authorization_witness,
           completeness_and_negative_observation_witness),
  input: borrowed typed source-lane references<'source>,
  recipe: separately compiled/verified native recipe identifier,
}
```

This is an **obligation checklist**, not permission to add a duplicate struct to the contract. Seek existing `ClassView`, `MailboxSoaView`, `SoaEnvelope`, Quack `Col`/`Query`, mask-risc `LaneRef`, D-DRMC recipe and D-RMAP focus witness shapes first; name which precise fields are absent. If no source owner/physical snapshot witness exists, keep it OPEN and refuse escaped use. Do not invent an owner from a numeric cycle.

At runtime inside a live immutable source borrow, Quack executes direct borrowed lanes with **no extra per-op ownership check**; outer preparation validates source/type/scope/coordinates. If an input must be decoded or widened, charge copy/calibration and label it non-zero-copy. Existing Quack lane types may not accept `u16` class IDs or `f32` energy directly. Use already-supported U32/U64 and typed masks for first fixture; do not build new LaneRef variants as a prerequisite.

**Borrow guarantees data immutability, not permission.** ACL/scope is checked independently by source authority, especially on multi-source bindings.

## 2. First exactly eight test predicates and data oracles

These are **catalogue candidates with confirmed or laboratory source surfaces**, not a claim that all are shipped DT64 mappings. Catalogue slot numbers are from the DT64-1 *proposed* fixed grouping and remain provisional; not an ABI.

| slot (proposed) | predicate | type | source anchor / independent oracle | critical refusal |
|---:|---|---|---|---|
| 08 | `MEMBER_OF_DIRECT` | principal/group → group | `dir-sim/ad.rs` `member`, direct `View`; no transitive substitution | foreign group key, unread membership |
| 11 | `SECURITY_GROUP` | unary group classification | `dir-sim/groups.rs::SecurityEnabled`, SID semantics | mail-enabled SMTP ≠ security SID |
| 12 | `MAIL_ENABLED_GROUP` | unary group classification | `dir-sim/groups.rs::MailEnabled`, primary SMTP semantics | security SID ≠ mail enabled |
| 13 | `GROUP_GRANTS_ROLE` | group → declared role | `dir-sim/actors.rs::GroupRoles`, versioned policy | missing policy/unread role list |
| 14 | `EFFECTIVE_ROLE` | principal → role (derived) | `dir-sim/actors.rs::role_evidence`, pre-existing compiled fold | nested security support, role scope, double-support |
| 17 | `HAS_REMOTE_MAILBOX` | unary AD-recipient kind | `dir-sim/cloud.rs::Recipient::RemoteMailbox` | remote recipient ≠ created EXO mailbox |
| 18 | `HAS_EXO_MAILBOX` | principal → cloud mailbox **only where unique and witnessed** | `dir-sim/cloud.rs::CloudMailboxes`, explicit cloud/exchange identifiers | CloudUnread, two targets, mismatch / backsync |
| 32 | `CATS_ENTRY_EMPLOYEE` | CATS entry → SAP scoped employee key | `lance-graph-sap/src/schema.rs` ordinal 5, `bind.rs` source data | same numeric PERNR ≠ AD/Odoo identity |

**All 8 participate in semantic binding; only binary relations participate in the three SPO orientations.** Unary classifiers are unary WHERE/mask filters, not fictitious entity→Boolean-object edges. To exercise `(s,p)→o`, `(p,o)→s`, `(s,o)→p`, use witnessed binary relations 08, 13, 14, 18, 32, subject to defined endpoint types and complete focus. An inverse *query* returns subjects; it must never assert a reverse business fact.

No licence assignment test is allowed to pass without a separate real Entra SKU/service plan DTO authority. Do not infer licence from mailbox.

### Lab fixture: source-shaped but independently authored

- IAM: 64 principals, 12 groups including a nested diamond, one security group with no SMTP and one distribution group with no SID. Read source policy and roles, include an unread group and explicit complete-negative subdomain.
- Exchange: four cloud mailbox observations, one matched, one CloudUnread, one AwaitingBacksync and one ambiguous foreign link. Preserve mailbox GUID and ExternalDirectoryObjectId meanings; no fake automatic licence proof.
- SAP: eight CATS entries from two client namespaces; identical PERNR in both, optional null project/customer values; compare to existing `CatsBatch` field extraction, not a synthetically asserted AD identity join.
- Scope: two tenants, separate source owners and versions; one read denied under otherwise equal data.
- Prep: witnessed typed focus; no guessed cross-system identity. Odoo remains documented candidate for a *later* source integration, outside first binary fixture.

### Differential oracles

A: independent scalar source DTO traversal/classification, source semantics manually asserted from fixture.  
B: Quack+mask-risc through live immutable source borrow and prepared recipe.  
C: optional DR64² Boolean relation tile over the same witnessed focus, only when it pays; compare parity and costs.  
D: optionally existing direct `dir-sim` and `lance-graph-sap` behavior as an additional cross-check, not as the sole independent oracle.

Run directionally correct `Count/Exists/Keep` with output only at named terminal. Record source snapshots, schema, focus, scope, completeness and predicate catalogue at bind time. Avoid tuples in hot fold. The first proof requires **no DR64³**, no field-mask reinterpretation and no ONNX.

## 3. #1792: two different reuse lifetimes and mandatory falsifier

### Borrow lifetime (admitted execution)
`&Owner` → typed borrowed lanes → prepared Quack `Program`/mask-risc → answer. Mutating the same owner during live borrow must be rejected by safe Rust, as demonstrated by #1755. Per-operation generation checks duplicating a borrow invariant are unnecessary.

### Escaped result lifetime (blocked until source-attested dependency)
`Keep` or compiled data-derived selection becomes owned and survives the `&Owner` borrow. Its recipe hash is **not** a source/witness. Owner mailbox cycle and `last_write_cycle` are insufficient: #1792 Part E demonstrated public setters/fields, reset+same-cycle ABA and same numerical stamps from different mailboxes. `QueryReference::at` is a temporal horizon, not proof of one Lance `checkout_version`.

**Required adversary:** reproduce/reference the P2 1,032 versus 31 stale `R` selection under changed predicate source values; reusing detached selection must RECOMPUTE/REFUSE, not silently return previous `Keep`. A compiled mathematical recipe may be reused on fresh, independently valid borrowed lanes after rebind.

Never solve this by introducing a new FrameStore, invented generation in Quack or cheap hash assumed to represent source authority. A future escaped cache requires separate complete source/owner/snapshot/dependency/focus/codebook/scope witnesses and mutation-path encapsulation; that arc stays OPEN.

## 4. Acceptance and refusal matrix

| Falsifier | Expected |
|---|---|
| all eight source predicates with independent oracle, labels respected | equal where admitted |
| unary classification presented as binary fact | type refusal |
| identity collision `same local index, different source` | refuse |
| SAP client A vs B equal PERNR | distinct |
| mail-only group treated as permission-bearing | refuse/not granted |
| group role grant unread vs explicit complete no-grant | UNKNOWN vs FALSE |
| role supported by two nested SID paths, remove one | remains TRUE |
| remote mailbox with unread or backsync incomplete | no claim of proven EXO mailbox |
| missing authoritative SKU source | LICENSE_ASSIGNED not admitted |
| scope denied after otherwise same bindings | refuse (not FALSE) |
| absent field, explicit NULL, unread, explicit false | distinct |
| QueryReference horizon same, physical dataset differs | no cross-source snapshot identity |
| predicate catalogue version or codebook changes | refuse stale semantic binding |
| source owner mutation during live borrow | compile-fail |
| change source after borrow; old `Keep` 1032, fresh 31 | never silently reuse old |
| transposed query invents reverse edge | refuse |
| 8 predicates vs global dense 64³ | direct baseline; cube not required |
| same compiled law rebound to fresh admitted source | correct fresh answer, reusable recipe |

Each laboratory falsifier has an explicit independent expected answer and a negative/disable mutation that demonstrably changes the outcome. Tests not yet run are **PLANNED**.

## 5. Exact staged engineering path, distinct from six docs-only PRs

**QB-S0: Census and mapping receipt** (current documentation). Confirm real concrete source and lane APIs, compile-time borrow constraints, schema and scope gaps; PR #1792 and D-QSP contracts, no new struct.

**QB-S1: Borrow-scoped thin adapter lab** (future, requires operator approval): one source-shaped fixture, 8 predicates mapped to existing typed Quack/RelAlg lanes. Gate with source/independent oracle and type/refusal tests. No general Registry API or cross-system production binding.

**QB-S2: Directional projections** (future): forward/inverse/predicate query over binary predicates; unary stay WHERE; optional DR64² compared against direct lanes at full cold+hot+terminal cost; preserve K3/Bag and source focus.

**QB-S3: Safe recipe rebinding** (future): same compiled recipe, two different valid source reads. Refuse/recompute escaped old `Keep`; compare to 1,032/31 falsifier and independent oracle. No complete detached cache in this stage.

**QB-S4: Explicit ADOPT/KEEP_DIRECT/DEFER verdict** after cost/source oracle and witness review. Production escaped reuse, cross-source consistency, DR³ and actual connectors need separate source-owner decision. No code authorization from this design note.

## 6. Numerical separation for future

D-NAM/EXO-152 owns typed external numeric-method mapping. ndarrays own native kernels. SourceParity versus CertifiedMath, numerical `Codec8`/Palette256 i8/u8 code-space and −128 semantics, original weights/scales/zero-points, calibration/version fingerprints and prepared numerical certificates are **referenced**, never absorbed into a DT64 predicate descriptor or 64-slot table. ONNX/TFLite are leaf-lowering vocabularies. A model prediction is a computed result, never an observed IAM permission or mailbox fact.

## 7. Coordination and outstanding contracts

- **DT64 catalogue**: candidate 64 semantic slots, not approved ABI; source owner reviews in #1794.
- **#1792 MERGED**: borrow lifetime established; escaped `Keep` and physical source snapshot attestation **OPEN**.
- **EXO-46**: global→local production preparation admission unresolved.
- **NFR-OP-2**: DR³ third physical axis unresolved, *not on critical path QB-S1*.
- **EXO-148**: durable Thought-Git types absent; *not on critical path QB-S1*.
- **EXO-152**: D-NAM numerical methods are separate, *not on critical path QB-S1*.
- **Entra licences**: no source-supported SKU/admission in current dir-sim; never inferred.
- **Odoo**: source-extracted DTOs exist; a native Odoo method bind proof remains later.
- **Access**: nominal mailbox owner `mailbox_owner` and `cast_on_behalf` do not themselves authorize queries.

**Exit for this coordination correction:** owners agree the shared interface is a source-attested borrowed semantic binding plus independent compiled recipe reference; one typed eight-predicate lab is the first requested executable proof, not 64 runtime adapters or a physical tensor.
