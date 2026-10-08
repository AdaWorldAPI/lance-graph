# 2026-10-08 — rbac: RBAC hot-plug socket (`contract::rbac_plug`)

**Status:** TEST-PINNED (`lance-graph-contract` `rbac_plug` tests; `lance-graph-rbac` `plug_tests`). Additive.

## MEASURED

- No RBAC consumer used the hotplug pattern. MedCare-rs carries a hand-copied policy (`medcare-rbac`, kept equal to lance-graph-rbac by a parity test) beside its capability `HOT_PLUG`; a2ui-rs takes a raw role mask from its caller and checks actions against a classid range only; ogar-rbac's `OgarRbac` was the only real `ClassRbac` impl and nothing bound to it.

## DECISION

- `contract::rbac_plug`, shaped after `contract::hotplug`: `RbacPlug` (one const per consumer: classids + role names), `RbacAuthority::bind`, `RbacBinding` (private fields, no `Default`, `Result` lookups; grants and field masks outside the plug are dropped at construction), `RbacDrift` (named refusals incl. `MirrorDrift`), `ActorSource` + `PluggedRbac` (the `ClassRbac` view; an out-of-plug question denies and gets an empty mask), `verify_concepts_against_mirror`.
- Roles are `RoleId` names, matching every existing grant table. **BASIS:** operator chose names over minted role classids.
- Scopes come from the actor source's memberships, so the view is meant for `authorize_memberships`.

## OPEN

- OGAR authority (`impl RbacAuthority` in ogar-rbac) is the paired next PR.
- Consumer plugs (MedCare-rs, a2ui-rs) follow once both merge; MedCare's grant table must first exist on the authority side.
