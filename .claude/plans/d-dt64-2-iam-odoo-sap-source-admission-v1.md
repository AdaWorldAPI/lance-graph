# D-DT64-2: IAM × Odoo × SAP admitted DTO source adapters and identity witnesses

> **DT64↔Quack coordination Q0 (2026-10-26):** [Minimal source-attested borrowed binding contract, capstone #1799](https://github.com/AdaWorldAPI/lance-graph/blob/research/dt64-predicate-holographic-capstone-20261026/.claude/plans/d-dt64-q0-borrowed-quack-binding-seam-v1.md). Source owners attest object/physical snapshot, access and completeness; DT64 binds borrowed input lanes, not copied source rows. Do not assume one borrow covers unrelated datasets or that a QueryReference alone pins Lance bytes. All first-landing tests require #1792's detached-\`Keep\` 1,032-vs-31 refusal/recompute. Coordination note only; not new implementation authorization.

**Date:** 2026-10-26. **Status:** SPECIFICATION / docs only, independent draft. **Parent:** DT64-0, DT64-1 review. **Not:** live integrations, code, entitlement changes or SAP writes.
**Existing owner arcs:** EXO-46, EXO-89, EXO-119–123, EXO-126, EXO-151; Quack borrow audit #1792.

## Purpose

Register real source-side DTO surfaces rather than create invented shared identifiers. A *source record* is not yet a *canonical person*; a value in a typed field is not yet an admitted relationship; a relationship is not yet an authorized fact for the current actor. Keep each phase and its receipt separate.

## 0. Source census with exact current anchors

| Source | As built on main | Extracted semantics | Not proven |
|---|---|---|---|
| AD/dir-sim | `crates/lance-graph-dir-sim/src/ad.rs`: `objectGUID`, `userPrincipalName`, `msExchMailboxGuid`, `member`, `proxyAddresses`; `groups.rs`, `actors.rs` | Direct and nested group, mail enabled vs security SID; role evidence from read grants | Complete real-AD/Entra synchronization, live entitlement, inferred name equality |
| Exchange/Entra | `dir-sim/cloud.rs`: `CloudMailboxes`, `MailboxGuid::{Matched,NoCloudMailbox,CloudUnread,AwaitingBacksync,Mismatch}`; `ad.rs` projection | Exact 1:1 cloud identity fold where witnessed, remote vs actual EXO state | All EXO mailbox lifecycle states, licence SKU/plan authority |
| Odoo | `.claude/odoo/L14-HR-BASE.md` R2,R6,R8 and L12 company access, `lance-graph-ontology` Odoo blueprint | `hr.employee.user_id`, `work_contact_id`, `parent_id`, `hr.version` periods; `res.users.company_id`, `allowed_company_ids` | Any shipped DT64 Odoo method route, live database, ACL proof |
| SAP | `crates/lance-graph-sap/src/{schema,bind,edge}.rs`: 23 `CatsSchema::FIELDS`, 16 HASH / 8 BAPI ordinals, `CatsBatch` | CATS employee_number (PERNR-like), project/customer/task/date, billability with UNKNOWN | Generic SAP authorization objects, personnel↔AD crosswalk, live ERP write |

## 1. Candidate source binding descriptor (not public ABI)

```text
SourceBinding {
  source_namespace: SourceId, tenant_or_client: ScopedId,
  stable_object_key: (kind, opaque_bytes), source_snapshot: SourceVersion,
  schema_identity: (schema_digest, field_ordinal, datatype),
  predicate_registry: (catalogue_version, slot, domain, range, direction),
  relation_mode: Direct | Derived(rule_digest) | ObservedNegative,
  relation_value: TypedKeyPair | TypedScalar,
  presence: PRESENT | ABSENT | UNKNOWN | NULL,
  read_completeness: CompleteInScope | Incomplete | Unread,
  policy_scope: ScopeWitness, focus: CoordinateWitness,
  source_owner: OwnerIdentity, observation_clock: ObservedAt,
  valid_interval: optional ValidFromTo,
  source_provenance: SourceReceipt
}
```

No single scalar `status` or `u64 frame` replaces these. `SourceBinding` is a *plan vocabulary*; re-use existing provenance/witness types where lawful. Raw DTOs are borrowed; copy only sealed identity/index metadata when permitted. Never copy giant payloads into an invented DT64 row store.

## 2. Concrete mapping stories

### A. User → direct AD group → transitive security group → effective role

1. Source principal GUID, group GUID and `member` facts must come from one versioned AD snapshot.
2. `GroupProperty::SecurityEnabled` checks SID, not mail address. A group with only SMTP may forward mail but not role authority.
3. Existing `GroupRoles::from_policy` takes a versioned policy. Join with actual security-identifier membership chain.
4. One role with two independent grant paths remains true when one path disappears. `RoleEvidence::Undetermined { unread }` is not FALSE (distinct from `NotHeld` and `NotDeclared`).
5. `ADMIN_IN_SCOPE` is derived only when the requested role and object/resource scope are bound; the word “admin” in a group name does not grant it.
6. Oracle: `dir-sim::View` / `actors::GroupRoles::role_evidence` on exact same snapshot, not a new duplicate evaluator.

### B. AD remote mailbox ↔ Entra identity ↔ Exchange Online mailbox

1. AD objectGUID anchors on-prem. A cloud target requires exact, unique Entra link, not UPN/proxyAddresses fuzzy match.
2. `Recipient::RemoteMailbox` alone does not prove a provisioned cloud mailbox.
3. Cloud mailbox ExternalDirectoryObjectId/ExchangeGuid is separate from AD objectGUID and on-prem `msExchMailboxGuid`. With backsync incomplete, state = AwaitingBacksync, not matched.
4. `HAS_EXO_MAILBOX`, `MAILBOX_GUID_MATCH`, `MAIL_RECEIPT_ELIGIBLE` are distinct operations; only `CloudMailboxes::delivers_to` under its input completeness drives the latter.
5. Licences and service plans require an actual separate Entra assignment dataset; a mailbox cannot be reverse-used as licence proof.
6. Oracle: existing `dir-sim/cloud.rs` branches including CloudUnread, NoCloudMailbox, Mismatch and duplicate link refusal.

### C. Odoo employee ↔ user ↔ company/department/manager

1. `hr.employee` is a work resource; `res.users` is a login principal; `res.partner` is a partner/contact. `user_id` and `work_contact_id` are different relationships.
2. `res.users.company_id` denotes current active company; `allowed_company_ids` is a permission-related set. Neither implies AD security group.
3. `hr.employee.parent_id` is a directional manager link and may be propagated on department manager changes; preserve event/snapshot time.
4. `hr.version` derives valid employment periods; absence/archiving/record rules and company ACL influence visibility.
5. Do not map `hr.employee.id` = SAP PERNR = AD objectGUID merely because a sample fixture uses equal integers.
6. Oracles: extracted L14 R2/R6/R8 and L12 candidate model semantics, with explicit UNKNOWN if authority cannot be reproduced from source.

### D. SAP CATS ↔ Odoo employee crosswalk

1. `CATS.FIELDS[5]` employee_number belongs to SAP source/client namespace, not directly to any other model.
2. `source_system`, `tenant_id`, `entry_id`, `employee_number`, `work_date_utc` and optional customer/project/task ordinals are preserved.
3. A separate owner-signed cross-system person mapping is needed to connect SAP PERNR with Odoo `hr.employee` or AD GUID. If absent/ambiguous, REFUSE join.
4. Keep `HASH_ORDINALS` and `BAPI_ORDINALS` in their original order. Never reformat strings in a hot query, rewrite Unicode/ALPHA semantics or treat the BAPI output as internal truth.
5. Existing CATS `billable_code` has `BILLABLE_UNKNOWN`; map to three-valued evidence, not false.
6. Oracle: `lance-graph-sap` schema/bind/edge and existing explicit BAPI sink; no actual BAPI call.

## 3. Identity collision and consent/focus tests

| Falsifier | Premise | Expected |
|---|---|---|
| two different AD GUIDs, equal UPN | same human-readable string | DISTINCT, never auto-merge |
| two Entra objects, one AD anchor | ambiguous cloud fold | REFUSE 1:1 mailbox identity |
| Odoo `res.users.id` = SAP employee_number | numeric equality | DISTINCT unless attested crosswalk |
| SAP two clients both PERNR=00000042 | client namespace differs | DISTINCT |
| Odoo active company differs allowed company | company access | do not treat active as complete authorization |
| source licence record unread, cloud mailbox exists | incomplete SKU evidence | LICENSE_ASSIGNED UNKNOWN |
| security group absent SID, SMTP present | distribution group | ROLE FALSE only with complete policy witness; no SID chain |
| mail recipient remote, ExchangeGuid not backsynced | awaiting | MISMATCH not inferred; preserve AwaitingBacksync |
| same source integer revision, two datasets | foreign source | REFUSE detached reuse |
| source field NULL, missing and permission-denied | distinct causes | no false coalescence |

## 4. Minimal lab integration proposal

Use a **synthetic but source-shaped read-only** fixture: 64 principals, 12 groups (nested, cycles, doubly supported), 8 Odoo users/employees, 8 SAP CATS entries, 4 cloud mailboxes, two tenants and two SAP clients. Fixture explicitly writes two cross-source identity witnesses and one ambiguous candidate. An independent scalar join oracle checks exact predicate semantics. Compare: direct original source views; Quack prepared borrowed lanes; optional bit-plane for repeated asks. Do not assert real source connectivity.

The adapter must retain `&source` borrows through the execution window (#1792), explicitly separate source identity from `QueryReference::at`, and refuse escaped `Keep` reuse until the owner/snapshot binding contract exists. `ClassView FieldMask` is only presence. Source revision and ACL changes invalidate applicable bindings. No policy changes, writeback, new FrameStore or new dynamic interpreter.

## 5. Economic gate and alternatives

Measure admission/schema resolve, identity-map seal, cold mapping, focused read build, repeated Quack hot fold, output materialization and source oracle. Compare direct 8/23-field masks, CSR, DR64²/DR256² where repeated use pays for materialization. One-shot SAP/ODOO fields should stay direct if faster. Observe missing/unknown rows and full provenance size, not only 32 KiB relation payload.

## 6. Known blockers and successors

- EXO-46 counted global→focus preparation needs operator admission before production.
- Odoo runtime source method route has not been independently tested.
- No source-attested Entra SKU/plan entity in the inspected IAM-sim surface.
- SAP CATS is not SAP authorization; source-specific SAP auth DTO would require new evidence.
- NFR-OP-1 escaped reuse and physical Lance version binding still open (#1792).
- DT64-1 slot registry/codebook not implemented, DR³ third axis still open.
- Negative assertion needs authoritative closed-world scope, not a `NOT EXISTS` over unread data.

**Exit:** approved mapping decisions with 10 binding falsifiers, independent per-source oracles, no live write and cost accounting. **No implementation authorized here.**
