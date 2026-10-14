# Access-control model

READ BY: anyone adding an RBAC consumer, a field mask, a row scope, an
output stage that shows records, or anything that decides who may do what.

This document describes the model as it stands in code on 2026-10-14. Each
statement names the item that implements it.

## The rule that comes before everything else

**Access control is policy data, not reasoning.** A decision about who may read,
write or act is computed from roles, grants, scopes and masks, all keyed by
classid. It never consults or changes the reasoning substrate (CausalEdge64,
truth values, revision, inference). If a decision ever seems to need inference,
it goes to the global reasoning layer as a request. It is not hand-rolled as a
local heuristic inside access control.

During the POC, policy overrides go through RBAC and the classid: a class-level
default, a role mask, a grant. They do not go through new code paths.

DECISION (operator, 2026-10-14): "Keep separation of concerns. If something
needs reasoning it should use global reasoning and not handroll. Instead use
policy override during the POC via Rbac and classid." The reasoning layer is
changing quickly, and a local copy of it inside access control would drift
behind it.

## Pipeline

```text
identity ──► actor ──► roles / memberships ──► decision ──► output
(ogar-auth)  (ActorSource)  (ClassRbac)     (authorize_*)  (walk_rails_authorized)
```

1. **Identity.** `ogar-auth` authenticates a user (`AuthenticatedUser`). A
   delegated token carries the acting party as an RFC 8693 `act` claim, which is
   parsed at the token boundary.
2. **Actor.** An `ActorSource` turns the identity into held roles.
   - `ogar_rbac::IdentityActors` reads them from the identity under a binding.
   - `lance_graph_dir_sim::ViewActors` reads them from directory groups,
     transitive members included, with group roles from OGAR's
     `GroupRolePolicy` (`GroupRoles::from_policy`).
3. **Authority.** OGAR owns the policy. `GrantSource` provides `roles_of`,
   `grants_of`, `scope_of`, `field_mask_of` and `default_field_mask_of`.
   `OgarRbac::bind(&RbacPlug)` checks the consumer's plug against the classid
   mirror and returns an `RbacBinding`. With actors attached, the binding
   implements `ClassRbac`. A consumer never builds its own grant table.
4. **Decision.** `lance-graph-rbac` folds the roles:
   - `authorize`: allow or deny.
   - `authorize_scoped`: allow or deny, plus the row scope and field mask.
   - `authorize_memberships`: the same, with scopes taken per membership.
5. **Output.** `selection::walk_rails_authorized` removes what the decision
   hides before anything is emitted.

## The four axes of a decision

| axis | question | where | how it combines |
|---|---|---|---|
| verb × class | may this role read / write / act on this class at all? | `ClassGrant { class, op_mask }`, `grants_permit` | default deny; at least one granting role is required |
| row scope | which instances? | `ScopeSpec` (`row_scope`), `Membership` | `authorize_scoped`: intersection over granting roles. `authorize_memberships`: union over granting memberships, and one unscoped membership means no restriction |
| field projection | which fields? | `ClassRbac::field_mask` → `WideFieldMask` | union over granting roles: a field is visible if any granting role shows it |
| output | what leaves the system? | `walk_rails_authorized` + `visible_fields()` | `view.mask ∩ presence ∩ visible(class)` at every hop |

Field masks are `WideFieldMask`, not `FieldMask`. A `u64` mask silently drops
positions 64 and above, which was a measured loss (a grant of `{1, 7, 92}` lost
92).

## Rules that are easy to get wrong

- **Admin is a role, not a bypass.** `ActionInvocation::commit` and
  `commit_via` both require the action's `required_role`. An admin without that
  role is refused (#1557).
- **A role without a mask must not open every field.** Field masks combine by
  union, so one role that declares no mask would open every field for the user.
  Two pieces prevent it:
  - `RbacBinding::with_class_field_defaults` gives such a role the class
    default instead of everything (#1556).
  - OGAR's authority supplies the default through
    `GrantSource::default_field_mask_of` (OGAR #358).

  A declared role mask is never replaced by the default.
- **A refused decision carries an open field mask.** On a non-`Allow` decision,
  `ScopedDecision.field_mask` and `MembershipDecision.field_mask` hold the
  open default and are documented as irrelevant. Output stages use
  `visible_fields()`, which returns `EMPTY` unless the decision allows.
- **A hidden link hides what is behind it.** A rail is a field position. The
  authorized walk does not follow a rail the reader may not see, so a linked
  object is not reachable through it, even if the reader may read that class
  directly.
- **No permit-all fallback in the output stage.** `walk_rails_authorized` takes
  a mask per class, not an `Option`. A class the reader may not read yields
  `EMPTY` and emits nothing.

## Open

- OGAR `ogar-doc-ir`'s `resolve_doc` calls `walk_rails`. It does not yet take
  a decision.
- Report grids (`lance-graph-report-ogar`) have no per-field positions, so the
  walk's mask does not apply to them.
- Accepted 2026-10-14, not yet implemented:
  - relational algebra as the normal form of a decision, with Quack /
    mask-risc as the executor (after a parity test);
  - both setting keys with the mapping as an edge, and a sibling
    `lance-graph-policy-sim` crate;
  - a shared-mailbox role in Spear;
  - a dir-sim owner gate that checks both uniqueness scopes, with the source as
    a parameter.
