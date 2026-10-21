# D-QR-IAM: typed QR256 relationship composition over dir-sim's borrowed views

**Date:** 2026-10-21. **Status:** LAB-ONLY PLAN / FALSIFIERS, unimplemented. No AD, Entra, Exchange, LDAP or PowerShell writes.

## Hypothesis

**Avoid handrolling derived IAM edges** by compiling already-known identity, predicate, role and group mappings into a small Boolean relational view, with `ClassView` + `WideFieldMask` identifying **eligible stable field positions**, #1659 providing a candidate borrowed global→local register, and #1660 a typed QR256 Bool `OR/AND` fold plus optional cold proof DAG.

The derivation `EffectiveRole(user,role) = EXISTS group: Member(user,group) AND Grant(group,role)` is one test target. `WideFieldMask` DOES NOT itself encode group membership, evidence origin, a group path, identity correspondence or permission. No automatic assumption that equals numeric row ordinal means equals entity across populations.

## Existing ground truth

- `lance-graph-dir-sim/src/lib.rs`: snapshots have independent users/groups 16-bit ordinal spaces (up to 65,536 each) and sparse many-to-many membership, shared version with overlays, Quack selection and simulated execution only.
- `bind.rs::UserBinder` resolves UPN/SMTP to comparison `KeyId` once using existing Quack binding.
- `groups.rs` separates MailEnabled and SecurityEnabled from nesting; mail chains and SID chains have distinct semantics.
- `View::members_transitive[_through]`, `View::groups_transitive` and `actors` are the **independent oracles**, not code to remove before parity is proven.
- `ClassView::fields` (append-only positions) and `WideFieldMask` in contract are the candidate schema geometry. `OrderedLaneWitness` means order must be sealed, not inferred. EXO-46 / #1640 ADMISSION B still apply.

## Contract sketch (not new Rust API)

```text
BoundRelationship {
  source_class, target_class, predicate_id,
  field_basis: ClassView::fields(...), presence: WideFieldMask,
  source_identity: GUID/CodebookWitness, target_identity: GUID/CodebookWitness,
  source_order, target_order, snapshot_generation, authority_scope,
  method8: BoolCompose, axes: UserLocal8×GroupLocal8 | GroupLocal8×RoleLocal8
}
```

The **QR256 register stores up to 256×256 bits**, not all 65k×65k directory edges at once. Chunked focuses require explicit external-frontier continuation and witnessed remapping. A 64×64 QR64 is 512 B and can serve smaller focuses; never silently drop an external membership.

## Falsifier paths

| Test | Independent reference | Fail condition |
|---|---|---|
| I1 Mapping | stable ClassId + ordered FieldRef and `WideFieldMask` selection; compare `UserBinder` exact UPN/SMTP | unknown field/key or swapped class/ref yields relation anyway |
| I2 One hop | parent group→member, user group→role Boolean existence vs `View`/RBAC oracle | same numerical ordinal used as foreign identity |
| I3 Security chain | nested security groups grant role; existence and complete witness chain match source | a non-security intermediate group unexpectedly transmits permission |
| I4 Mail chain | distinct `MailEnabled` semantics, no SID-derived email permission | an unaddressable group transmits mail |
| I5 Duplicates | Count/BAG matches independent multiset oracle; Bool EXISTS remains idempotent | two supporting paths XOR-cancel or support deletion erases surviving grant |
| I6 Version delta | edit membership, group property or authorization; recompute `BEFORE XOR AFTER` and independent full refresh | stale index, deleted group, old thought incarnation or revoked access still returns authorized true |
| I7 Focus | 256-slot maximum focus with bridges beyond focus, 257th edge external-frontier case | truncated reachability or implicit out-of-scope dereference |
| I8 Cost | direct `View` walk, existing Quack, borrowed co-addressed masks, QR64/QR256; count allocations and source bytes | unprofitable index/materialization promoted as win |

## Execution plan

1. Perform a read-only census of `snapshot.rs`, `view.rs`, `groups.rs`, `actors.rs`, `ogar_rbac::group_policy`; distinguish what's implemented vs planned.
2. Register a prototype typed relationship declaration **outside production directory store**; compile eligible FieldRefs/selection masks through existing contract surfaces.
3. Introduce a bounded borrowed address view with version/authority witness. Do not write an index from a foreign lane's value to select positions in another table; #1640 ADMISSION B forbids this.
4. Evaluate Boolean join/closure via existing mask-risc / ndarray tiles; no new recursive VM, no new permissions engine, and no template-as-string materialization.
5. Verify all positive and counterexample fixtures against `View` and independent multiset reference. If any premise is unknown, return typed UNKNOWN/refusal, not FALSE.
6. Measure 1, 8, 64 and 256 slots, nested depth, 257/external focus, duplicate supports; log cold/index + hot + proof costs separately.
7. Only after parity and permissions, propose cross-domain generalization to actual SAP or HubSPO; NO production IAM changes.

**Explicit non-goals:** no live directory mutation, no group-role schema minting without OGAR evidence, no CE64 encoding change, no privilege inferred from class presence, no new `Edge` heap object, no assumed transitivity for an arbitrary relation.

**References:** [#1659](https://github.com/AdaWorldAPI/lance-graph/pull/1659), [#1660](https://github.com/AdaWorldAPI/lance-graph/pull/1660), [EXO-79](https://linear.app/exo-red/issue/EXO-79), [EXO-86](https://linear.app/exo-red/issue/EXO-86), [EXO-46](https://linear.app/exo-red/issue/EXO-46), [EXO-50](https://linear.app/exo-red/issue/EXO-50).
