# D-DT64-1: 64-slot predicate catalogue, source DTO crosswalk and legality

> **DT64↔Quack coordination Q0 (2026-10-26):** [Minimal source-attested borrowed binding contract, capstone #1799](https://github.com/AdaWorldAPI/lance-graph/blob/research/dt64-predicate-holographic-capstone-20261026/.claude/plans/d-dt64-q0-borrowed-quack-binding-seam-v1.md). Catalogue defines the typed meaning, source identity and status of a predicate slot; it does not evaluate that predicate. The first executable fixture may use eight truly grounded slots without making all 64 operational. All first-landing tests require #1792's detached-`Keep` 1,032-vs-31 refusal/recompute. Coordination note only; not new implementation authorization.

**Date:** 2026-10-26. **Status:** DESIGN / docs only. **Parent:** `d-dt64-holographic-predicate-capstone-v1.md` DT64-0 (separate draft). **Scope:** 64 candidate semantic mappings, not a production registry or a claim of 64 verified extraction rules.
**Review owners:** IAM/dir-sim, SAP CATS, Odoo blueprint, ClassView, authorization and DR. **No live source connectivity.**

## Type and truth contract

Each slot is a **versioned semantic predicate identifier**, NOT a bit position of `ClassView::FieldMask` (which records structural presence), a global entity ID, a per-tensor Palette256 index, an untyped 64-lane numeric column, or an approval to use axis 3 of DR64³. A slot cannot be reused for a different meaning within its catalogue version.

Descriptor proposal, not ABI: `{catalogue_id,version,slot,symbol,domain_class,range_class,arity,authority_source,source_object,source_field_or_method,source_schema_digest,source_namespace,identity_join_witness,cardinality,direction,observed_or_derived,closed_world_scope,nullable,valid_time,observed_at,source_snapshot,policy_scope,permission_witness,negative_witness,confidence_kind}`. No new type required by this docs PR.

**Structural field presence vs predicate truth:** four possible logical outcomes TRUE, FALSE (only with authoritative complete negative), UNKNOWN/UNREAD, REFUSED. `NULL`, missing, unobserved, revoked read permission, contradictory duplicate, and inapplicable domain stay individually recorded; never coerce to FALSE. Derived roles and eligibility require their own dependency witnesses. Numeric date, quantity, price and identifier values are typed lanes, not Boolean truth bits.

**Registry layout proposal:** eight groups of eight slots. Group 6 is an **orthogonal evidence/policy catalogue**, not automatically a business SPO fact table. Group 7 is an **optional temporal/hypothesis proposal**; no implementation or claim of learning. A production registry may choose fewer than 64 active predicates and reserve others; unadmitted items must refuse rather than fabricate.

## Source-grounded slots and candidate mappings

Status meanings: SHIPPED source type/field or LAB existing behavior means a source **surface** is present, not that this DT64 mapping exists. EXTRACTED means Odoo blueprint analysis exists but no DT64 runtime binding. OPEN/MISSING/PROPOSED indicate unsourced or unratified semantics.

| Slot | Symbol | Domain → range / meaning | Source field or read seam | Source evidence status | Rule / negative edge |
|---:|---|---|---|---|---|
| 00 | `AD_OBJECT` | AD objectGUID → canonical DirectoryObject | dir-sim/ad.rs::objectGUID | **observed candidate** | Guid128; source-namespace |
| 01 | `ENTRA_LINK` | AD principal → Entra object | dir-sim/cloud.rs; directory fold identity lanes | **derived with uniqueness** | exact corroborated pair |
| 02 | `ODOO_USER_EMPLOYEE` | res.users → hr.employee | Odoo L14 R8 user_id | **extracted candidate** | tenant/company scoped |
| 03 | `ODOO_EMPLOYEE_PARTNER` | hr.employee → work_contact res.partner | Odoo L14 R8 work_contact_id | **extracted candidate** | not same as AD identity |
| 04 | `SAP_EMPLOYEE_REFERENCE` | CATS entry → SAP employee number | sap/schema.rs FIELDS[5] employee_number | **shipped DTO field** | client+PERNR, never AD GUID |
| 05 | `SOURCE_OBJECT_OF` | object → originating source record | source envelope / source-system field | **adapter design** | source-scoped opaque key |
| 06 | `IDENTITY_LINK_CONFIRMED` | cross-system principal ↔ person | no shipped cross-system authority | **UNPROVEN** | dual source evidence required |
| 07 | `IDENTITY_LINK_CONFLICT` | cross-system principals disagree | no shipped cross-system authority | **UNPROVEN** | negative/conflict witness |
| 08 | `MEMBER_OF_DIRECT` | principal/group → group direct | dir-sim/ad.rs member; store.rs | **observed candidate** | keep direct vs nested |
| 09 | `MEMBER_OF_NESTED` | group → group | dir-sim/ad.rs member nested groups | **observed candidate** | typed group-only |
| 10 | `MEMBER_OF_TRANSITIVE` | principal/group → group closure | dir-sim/view.rs; dir-rmap probe | **LAB derived** | distinct closure path |
| 11 | `SECURITY_GROUP` | group → SID/authority marker | dir-sim/groups.rs GroupProperty::SecurityEnabled | **observed candidate** | SID evidence, not label |
| 12 | `MAIL_ENABLED_GROUP` | group → SMTP marker | dir-sim/groups.rs MailEnabled | **observed candidate** | not security |
| 13 | `GROUP_GRANTS_ROLE` | security group → role | dir-sim/actors.rs GroupRoles | **observed policy** | grant source/version |
| 14 | `EFFECTIVE_ROLE` | principal → role via SID chain | dir-sim/actors.rs role_evidence | **LAB derived** | retain paths, unread≠false |
| 15 | `ADMIN_IN_SCOPE` | principal → scoped admin role | RBAC binding/group role policy | **derived UNPROVEN** | never blanket is_admin |
| 16 | `HAS_ONPREM_MAILBOX` | AD object → onprem mailbox identity | dir-sim/ad.rs msExchMailboxGuid; recipient | **observed candidate** | recipient type, GUID confirmed |
| 17 | `HAS_REMOTE_MAILBOX` | AD object → remote recipient | dir-sim/cloud.rs Recipient::RemoteMailbox | **observed candidate** | not equivalent to cloud presence |
| 18 | `HAS_EXO_MAILBOX` | Entra principal → EXO mailbox | dir-sim/cloud.rs CloudMailboxes | **observed candidate** | unique fold only |
| 19 | `MAILBOX_GUID_MATCH` | remote AD mailbox ↔ EXO ExchangeGuid | dir-sim/cloud.rs MailboxGuid::Matched | **derived strict** | AwaitingBacksync not true |
| 20 | `MAIL_RECEIPT_ELIGIBLE` | recipient → delivery eligibility | dir-sim/cloud.rs delivers_to | **derived gated** | never infer from proxy alone |
| 21 | `HAS_SMTP_ADDRESS` | recipient → SMTP address | dir-sim/ad.rs mail/proxyAddresses | **observed candidate** | primary vs proxy typed |
| 22 | `LICENSE_ASSIGNED` | principal → SKU assignment | no authoritative Entra licence DTO here | **MISSING SOURCE** | cannot derive from mailbox |
| 23 | `SERVICE_PLAN_ENABLED` | principal → service-plan entitlement | no authoritative licensed-plan DTO here | **MISSING SOURCE** | distinct assigned/enabled/active |
| 24 | `BELONGS_TO_TENANT` | principal/object → tenant | AD/Entra namespace; sap tenant_id; Odoo company | **candidate cross-source** | namespaced typed tenant |
| 25 | `ODOO_ALLOWED_COMPANY` | res.users → allowed res.company | Odoo L12 RES_USERS_COMPANY_ACCESS | **extracted candidate** | multi-company ACL |
| 26 | `ODOO_ACTIVE_COMPANY` | res.users → current res.company | Odoo L12 company_id | **extracted candidate** | not all allowed companies |
| 27 | `EMPLOYED_BY` | hr.employee → res.company | Odoo L14 HR employee/company | **extracted candidate** | valid-time contract |
| 28 | `REPORTS_TO` | hr.employee → hr.employee manager | Odoo L14 R6 parent_id | **extracted candidate** | direct manager only |
| 29 | `DEPARTMENT_OF` | hr.employee → hr.department | Odoo L14 R8/employee dept | **extracted candidate** | optional nullable |
| 30 | `JOB_OF` | hr.employee → hr.job | Odoo L14 HR_JOB | **extracted candidate** | position vs group role |
| 31 | `COMPANY_PARENT` | res.company → res.company | Odoo L12 company tree | **extracted candidate** | cycle prevention |
| 32 | `CATS_ENTRY_EMPLOYEE` | SAP CATS entry → employee | sap/schema.rs employee_number (5) | **observed DTO field** | PERNR source identity |
| 33 | `CATS_ENTRY_CUSTOMER` | SAP CATS entry → customer | sap/schema.rs customer_number (6) | **observed nullable DTO field** | missing not false |
| 34 | `CATS_ENTRY_PROJECT` | SAP CATS entry → project | sap/schema.rs project_code (7) | **observed nullable DTO field** | no implicit global project link |
| 35 | `CATS_ENTRY_TASK` | SAP CATS entry → task | sap/schema.rs task_code (8) | **observed nullable DTO field** | not edge without matching key |
| 36 | `CATS_ENTRY_WORKDAY` | SAP CATS entry → date key | sap/schema.rs work_date_utc (9) | **observed DTO field** | time is value, date link optional |
| 37 | `CATS_ENTRY_BILLABLE` | SAP CATS entry → Billable marker | sap/bind.rs BILLABLE, BILLABLE_UNKNOWN | **derived ternary** | unknown not false |
| 38 | `ODOO_DEPT_MANAGER` | hr.department → manager employee | Odoo L14 R6 | **extracted candidate** | updates not historical truth |
| 39 | `ODOO_EMPLOYEE_USER` | hr.employee → res.users | Odoo L14 R8 user_id | **extracted candidate** | do not equate users in AD |
| 40 | `SALE_ORDER_CUSTOMER` | sale.order → res.partner | Odoo L6 sale order | **extracted candidate** | company/domain scoping |
| 41 | `SALE_ORDER_COMPANY` | sale.order → res.company | Odoo L6 sale order | **extracted candidate** | ACL and state |
| 42 | `PURCHASE_ORDER_VENDOR` | purchase.order → res.partner | Odoo L6 purchase order | **extracted candidate** | vendor vs employee |
| 43 | `STOCK_MOVE_PRODUCT` | stock.move → product.product | Odoo L7 stock and L8 product | **extracted candidate** | item identity |
| 44 | `STOCK_MOVE_SOURCE` | stock.move → stock.location | Odoo L7 stock move | **extracted candidate** | directed from |
| 45 | `STOCK_MOVE_DESTINATION` | stock.move → stock.location | Odoo L7 stock move | **extracted candidate** | directed to, not inverse edge |
| 46 | `ANALYTIC_LINE_ACCOUNT` | account.analytic.line → analytic account | Odoo L10 inventory | **extracted candidate** | cost record, not numeric sum |
| 47 | `ACCOUNT_MOVE_JOURNAL` | account.move → account.journal | Odoo L1/L11 | **extracted candidate** | accounting lifecycle |
| 48 | `SOURCE_OBSERVED` | record → source observation | DTO observation envelope | **PROPOSED META** | not business fact |
| 49 | `FIELD_PRESENT` | row → structural field | ClassView FieldMask | **SHIPPED META** | never predicate truth |
| 50 | `RELATION_COMPLETE` | source/scope → completeness | authoritative observation contract missing | **OPEN META** | enables explicit false only |
| 51 | `ACCESS_ADMITTED` | actor/scope → read grant | policy engine/scope guard | **OPEN META** | never derived from presence |
| 52 | `IDENTITY_RESOLVED` | mapping → key witness | EXO-46 global→local admission | **LAB META** | typed identity proof |
| 53 | `CODEBOOK_AT_VERSION` | predicate → codebook generation | D-NFR S15 missing | **OPEN META** | not numeric slot equality |
| 54 | `SNAPSHOT_AT_VERSION` | source → physical checkout | Lance checkout_version; #1792 | **PARTIAL META** | horizon alone insufficient |
| 55 | `EXPLICIT_NEGATIVE` | relation → negative observation evidence | no general negative-witness authority | **OPEN META** | not absence=FALSE |
| 56 | `EDGE_SUPPORTS_EDGE` | edge claim → edge claim | D-DRMC/D-RFT proposal | **PROPOSED** | proof vs assertion distinction |
| 57 | `EDGE_DEPENDS_ON_EDGE` | derived edge → input edge | D-RFT provenance proposal | **PROPOSED** | dependency closure |
| 58 | `CLAIM_CONTRADICTS` | claim ↔ claim under frame | Thought-Git/epistemic plan | **PROPOSED** | not resolved automatically |
| 59 | `CLAIM_HYPOTHESIS` | claim → thought slot | D-HL EXO-78 w_slot 0..63 | **PARTIAL PLAN** | 0 givens, 1..63 hypotheses |
| 60 | `CLAIM_VALID_DURING` | claim → validity interval | Odoo L14 hr.version / temporal plan | **PROPOSED** | actual interval separate scalar lane |
| 61 | `CLAIM_OBSERVED_AT` | claim → observation event | planner temporal / Lance versions | **PROPOSED** | different clock than valid time |
| 62 | `CLAIM_REPLAYED_FROM` | claim → replay receipt | EXO-98; EXO-148 missing types | **BLOCKED** | not equality of version counters |
| 63 | `EXECUTION_PRODUCED` | native method run → result | EXO-152 ONNX external leaf | **PROPOSED** | computed != observed fact |

## Worked crosswalks (do not silently equate identities)

### IAM / Entra / Exchange

`AD:objectGUID → DirectoryObject` from `dir-sim/ad.rs`; `member` is a direct user-or-group relation. `dir-sim/groups.rs` separates mail-enabled SMTP and security SID; effective roles require `dir-sim/actors.rs::role_evidence` and the source policy version. Entra identity must be uniquely witnessed by the existing fold: identical UPN or SMTP strings are NOT an identity guarantee. `CloudMailboxes` distinguishes remote recipient, cloud identity, awaiting ExchangeGuid backsync, mismatch and matched. Neither remote mailbox nor an SMTP address alone proves EXO mailbox presence or deliverability. Entra assigned SKUs and individual service-plan licensing are **missing authoritative DTO adapters** here. Name the missing source, do not invent graph values.

### Odoo

`hr.employee.user_id → res.users`, `hr.employee.work_contact_id → res.partner`, `hr.employee.parent_id → hr.employee` and `hr.department` are from `.claude/odoo/L14-HR-BASE.md`, extracted against Odoo source, not a live connector. `res.users.company_id` versus `allowed_company_ids` are distinct (L12). `hr.version` determines temporal employment terms with actual validity windows. The Odoo `res.users.id` integer is scoped to database/company context and is NOT an Entra objectGUID. Odoo business object candidates must be verified against extracted L1/L6/L7/L10 models before binding and may remain unadmitted.

### SAP

Only SAP CATS' pinned 23 append-only `FIELDS` and binding surface are sourced here: `lance-graph-sap/src/schema.rs`, `bind.rs`, `edge.rs`. The 16 HASH ordinals and 8 BAPI ordinals are separately ordered profiles; BAPI formatting is a materialization boundary. `employee_number` is PERNR-like, client/source-scoped, never automatically equivalent to AD GUID or Odoo `hr.employee.id`. Source `customer_number`, `project_code`, `task_code` may be nullable and must not be treated as cross-system foreign keys without an authority witness. No generic SAP IAM/authorization entitlement DTO has been evidenced by this CATS source.

## Eight mandatory end-to-end questions and failure oracles

1. `(user,MEMBER_OF_DIRECT,group)` compares with source AD `member` and dir-sim direct view; transitive never silently substituted.
2. `(user,EFFECTIVE_ROLE,role)` matches existing `role_evidence` including two independent SID chains; retract one chain and preserve role.
3. `(user,HAS_EXO_MAILBOX,mailbox)` is UNKNOWN for ambiguous identity or unread cloud state; `MAILBOX_GUID_MATCH` requires ExchangeGuid match.
4. `(user,LICENSE_ASSIGNED,sku)` REFUSES until a source DTO attests assignment; mailbox presence is not a licence.
5. `(odoo_user,ODOO_ALLOWED_COMPANY,company)` differs from active `ODOO_ACTIVE_COMPANY`.
6. `(sap_cats_entry,CATS_ENTRY_EMPLOYEE,PERNR)` compares to SAP CATS field preserving source+client; same digits as Odoo employee id do not join.
7. `(employee,REPORTS_TO,manager)` is directed, and a manager-to-employee answer must be the explicit inverse, not a symmetrical relationship.
8. A source field that was not read remains UNKNOWN; an explicit complete authoritative negative can be FALSE; a forbidden read returns REFUSED.

For all 64 slots define an `ADOPT / CANDIDATE / BLOCKED / INAPPLICABLE` machine-verifiable decision record during the later lab stage; **this document is a seed mapping, not a certified source crosswalk**.

## Gate and performance

DT64-1 acceptance is review of all 64 descriptors, 8 evidence-grounded question oracles, explicit refusal tests, source-schema revision ownership, and no unsourced cross-system identity. `ClassView::WideFieldMask` handles field presence separately; direct masks for narrow 8 or 23 fields are baseline. Cost includes cold schema mapping, global-to-local identity binding, tile build, and hot fold.

## Loose ends for successor PRs

EXO-46 global-to-focus admission; NFR-OP-2 third DR axis; D-NFR S15 codebook version absent; EXO-148 durable identifiers absent; #1792 escaped Keep witness unresolved; no genuine Entra licensing adapter; Odoo runtime method route untested; SAP CATS is not SAP role/authorization data; positive licence assignment differs from active service plan; Odoo ACL changes affect query scope; explicit negative authority absent. None is papered over by choosing 64 slots.

## No-go

No new `PropertySpec` semantics, no `FieldMask` repurposing, no synthetic source-level joins, no production ingestion, provisioning or BAPI writes, no policy bypass, no runtime tensor engine, no claims that metadata slot 48–63 are already data predicates.
