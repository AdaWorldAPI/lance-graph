# D-DT64-Q0: Source-attested semantic binding → borrowed Quack execution

**2026-10-26. Status:** INTER-SESSION ARCHITECTURE COORDINATION / DOCS ONLY. **Authority:** DT64-0 [#1799](https://github.com/AdaWorldAPI/lance-graph/pull/1799), DT64-1..5 [#1794](https://github.com/AdaWorldAPI/lance-graph/pull/1794)–[#1798](https://github.com/AdaWorldAPI/lance-graph/pull/1798), Quack/DR #1792 Part E. **Not:** a Claude Code instruction, a new production ABI, new solver, new runtime, an approved implementation or a merge decision.

## One-sentence contract

**DT64 attests *what* a DTO field or source relationship means, who owns it, which source snapshot and authorized focus it belongs to, and which borrowed native operands are currently valid. Quack / RelAlg / R2IL / mask-risc execute the *admitted algebra*. ndarray executes numerical kernels. D-NAM owns numerical method admission and prepared certificates.**

The binding is an *interface seam*, not a second copy of business data and not the persistent Binding Store generalized in old plans. Do not invent a FrameStore, tensor physical allocation, predicate evaluator, Odoo executor, ONNX interpreter or new RelAlg AST.

## First common interface, decomposed to avoid fake ownership

This is **contract vocabulary**, not a proposed Rust public struct. Each identity and policy term must point to an existing truthful owner or an explicitly OPEN integration gap. Most values are checked cold at bind/prepare, not in the SIMD hot path.

| Component | Source of authority / owner | First proof or refusal |
|---|---|---|
| typed source-object identity | actual namespace+tenant/client+opaque stable source key | two SAP clients both PERNR 42 never alias; AD GUID != Odoo integer |
| typed predicate identity | DT64 catalogue id/version/slot, domain/range/direction | slot != ClassView FieldMask, DR coordinate or Palette256 code |
| source schema and catalogue versions | source schema digest/field ordinal plus separately versioned DT64 descriptor | moved field ordinal, changed meaning or catalogue generation refuses |
| source owner and physical snapshot | owning source/reader; explicit opened physical version where persistence crosses boundary | `QueryReference::at` is a temporal horizon, not a dataset checkout (#1792) |
| global↔local coordinate witness | D-RMAP/EXO-46 source-attested preparation membrane; typed identity | equal local ordinal from other source refuses |
| codebook generation | real owner of local ordinal/codebook if relevant, NOT numeric equality | changed codebook refuses; D-NFR S15 still OPEN |
| permission / completeness scope | source ACL/policy/tenant, observation authority | unread or forbidden differs from explicit false; group-role policy completeness |
| borrowed input lanes | original immutable `&source` and existing `LaneRef` / `Planes<'_>` / strided views | no source mutation while borrowed; no hidden `to_vec` data replica |
| separately reusable compiled recipe | Quack/R2IL/D-DRMC prepared law bound to typed schema and supported math semantics | recipe reusable on a NEW valid source binding; old detached mask is not |
| result demand and validity | Quack `Count/Exists/Keep`, typed output, receipt and freshness | `Keep` cannot escape and be reused without complete dependency witness |

**Implementation rule:** Do not silently "fill" missing source attestations with a caller-supplied generation, sample fixtures, `mailbox_owner() == 0`, or the apparently matching codebook integer. The source provides authoritative snapshot/owner identity; DT64 records and validates bindings without becoming the issuer of physical data identity.

**Borrow scope:** For an in-process owner with ordinary safe Rust `&self` borrowed lanes and no mutating interior alias, a Quack kernel running while those borrows remain live needs **no redundant per-operation mutation check**. Preserve all input borrows and their source bind identities until the result's immediate consumption; a borrow does not prove external authorization or cross-source snapshot consistency. After the borrow ends, a detached `Vec<u64>` `Keep` is only a stale candidate, not an attested intermediate. #1792 falsifies reuse: stale R mask 1,032 selected rows vs fresh R mask 31 after changing a predicate property. Until dependency provenance is complete, recompute under fresh borrow or REFUSE. A compiled *recipe* can be retained; it is not a source-data cache.

## First executable proof, not a new execution engine

One read-only source-shaped fixture with **eight sourced predicates**, named independently of the 64-slot proposed registry:

1. `MEMBER_OF_DIRECT` (AD member or dir-sim membership)
2. `SECURITY_GROUP` (SID-based, dir-sim group property)
3. `MAIL_ENABLED_GROUP` (SMTP-address group property)
4. `GROUP_GRANTS_ROLE` (dir-sim GroupRoles, witnessed grant policy)
5. `EFFECTIVE_ROLE` (dir-sim actors::role_evidence; multiple chain support)
6. `HAS_REMOTE_MAILBOX` (dir-sim recipient type)
7. `HAS_EXO_MAILBOX` (CloudMailboxes unique linked read, NOT licence)
8. `CATS_ENTRY_EMPLOYEE` (SAP CATS FIELDS[5] employee_number, namespaced SAP client/PERNR, NOT automatic AD GUID join)

First executable slice may include the IAM/Exchange subset and the SAP field as **two independently borrowed, source-scoped views**; it does not assume a real AD↔SAP identity relation. Odoo mapping is planned DT64-2 but not a prerequisite to this minimal algebra proof. All fixtures synthetic/source-shaped, with independently authored original-source scalar oracle and no live account modifications.

### Phase Q0. Source and lane census, read-only

Inspect `contract::class_view::FieldMask`, `MailboxSoaView`/`SoaEnvelope` and #1792 E.1–E.9; existing `Quack` `LaneRef`, `Planes<'_>` and demand terminals; dir-sim `ad.rs`, `groups.rs`, `actors.rs`, `cloud.rs`; SAP `schema.rs` and `bind.rs`. Verify which *current* U32/U64/strided lanes can be borrowed and which u16/f32 carriers need an explicitly accounted temporary copy or a separately approved type addition. No invented zero-copy assertion.

### Phase Q1. Semantic descriptor and binding acceptance

Build a **lab-only** typed mapping table using source-owned identity, schema/catalogue, permission and source snapshot. Distinguish field **presence**, relation **knownness/validity**, actual **truth** and authorization. Cold-bind a Quack program to the admissible borrowed operands. No new hot-path dispatch, no general-purpose persistent data store, no custom predicate evaluator. If external sources do not expose a pinning contract, mark the fixture as lab owner-borrowed, not production attested.

### Phase Q2. Three directed query orientations

Compare the original source semantics, a separately implemented scalar typed-triple oracle, and existing Quack/relalg algebra:
- `(subject, predicate) → objects`
- `(predicate, object) → subjects`
- `(subject, object) → predicates`

Do not generate the inverse `(object, predicate, subject)` as a new fact. Keep per-predicate typed domain/range and distinguish `direct` versus `derived` group closure. Unknown/permission-denied is not Boolean false.

### Phase Q3. Optional DR64² economics

Where repeated queries genuinely need 64×64 source-target relations under one witnessed focus, compare direct borrowed lane + Quack against 512 B DR64² single-predicate bit tile, and baseline CSR/direct walk. Account **prepare + identity map + tile + source read/prefetch + query + output + provenance**. The DR64³ 32 KiB logical cube is **not** a prerequisite or an implementation authorization; NFR-OP-2 remains OPEN. Choose KEEP_DIRECT if construction cost dominates.

### Phase Q4. Mandatory negative controls and receipts

| Falsifier | Required result |
|---|---|
| P2 property mutation, old `Keep` retained: 1,032 vs 31 | fresh bind recomputes or explicit REFUSE, NEVER silently stale |
| same source stamp, distinct source identity | refuse foreign alias |
| same ordinal, different codebook generation | refuse or rebuild |
| scope or ACL changed | no unauthorized data leak or old negative reuse |
| direct group vs nested, security SID vs mail-enabled | typed predicates differ |
| role grant with two independent support paths | removing one still yields role |
| EXO cloud identity ambiguous / AwaitingBacksync | no unjustified mailbox "confirmed" |
| missing licence assignment DTO | REFUSE, not inference from mailbox |
| SAP PERNR same digits as Odoo id or AD surrogate | no cross-system join absent identity witness |
| missing field versus known complete negative | UNKNOWN/NOT-OBSERVED vs FALSE |
| swapped `ClassView FieldMask` and DT64 predicate slot | reject type/semantics |
| forward versus reversed triple | oracle discrepancy must bind |
| source snapshot and `QueryReference::at` only | refuse falsely pinned physical reader |

For each, record independent oracle, actual changed outcome, and **one disable/mutation that reliably makes the test red**. Record inert controls as NOT BINDING. No recorded test result until executed. No claim that a proof capsule can self-certify.

## Boundaries and stage ownership

| Artifact or behavior | Owner |
|---|---|
| 64 semantic slot descriptor/cold DTO normalization | DT64-1 / #1794 |
| source identity/read/ACL/completeness/snapshot | DT64-2 / #1795 and owning systems |
| directed algebra, optional DR64² physical view | Quack/relalg owners + DT64-3 / #1796 |
| time, edge dependencies and hypothesis history | existing Thought-Git owner, DT64-4 / #1797 only later |
| numerical method, calibration, fingerprint and certificate | D-NAM/EXO-152, ndarray and numerical audits; DT64-5 / #1798 **consumer only** |
| ordinary compile-time borrow guarantee and stale escaped mask evidence | #1755 and #1792 |
| preparatory global↔local identity admission | EXO-46, open for production |
| actual DR³ third-axis ABI | NFR-OP-2, open |
| durable Thought-Git IDs and authenticated replays | EXO-148, open |

Numerical method handles may reference **code-space identity, SourceParity vs CertifiedMath, calibration, prepared numerical certificate and source-specific implementation fingerprint**. The 64 DT64 predicate slots do not encode these quantities, and DT64 must never certify TFLM or ONNX correctness.

## First landing acceptance decision

**Only a lab/probe proposal**, not code authorization:
- precise current source and lane census; one minimal binding seam identified or explicitly blocked;
- eight candidate predicates grounded; actually admitted subset stated (no false claims for an incomplete source);
- same borrowed input data evaluated through direct source, independent scalar oracle, existing Quack; optional DR64² measured;
- forward/inverse/predicate projections with exact typed and negative controls;
- stale `Keep` 1,032-vs-31 refusal/recompute;
- prepare cost measured; no prohibited hot-path allocation or new runtime.
- conclusion ADOPT / KEEP_DIRECT / DEFER / REJECT backed by measured gate and source receipts.

**Coordinated governing statement:** DT64 describes what is bound. Quack executes the admitted algebra. A separately prepared numerical certificate determines whether an implementation satisfies an explicitly chosen numeric contract.
