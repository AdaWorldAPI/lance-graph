# 2026-10-07 — dir-sim: flag and location of an existing node are simulated changes; kind is identity

**Status:** TEST-PINNED (`crates/lance-graph-dir-sim/tests/properties.rs`, 9 tests; `tests/alloc.rs` covers `SetActive` / `SetLocation`; `tests/nodes.rs` re-pinned).

## DECISION

- **Vocabulary (OGAR #319).** `Change::SetActive` (`Option<bool>`, users only) and `Change::SetLocation` (`Option<Dn128>`) are compare-and-set, like `SetAttribute`.
- **Reference semantics.** `NodeState::apply` is the one definition; the overlay stores its result.
- **Kind is immutable identity.** No change alters it, and `PlanError::Unconvergeable` now means exactly a kind mismatch.
- **"Unknown" is never an operation.** A change towards an unknown flag or location is reportable in a diff and refused by the planner (`PlanError::NotActuatable`).
- **SCOPE:** directory simulation.
- **BASIS:** these are properties a directory changes on an existing object; kind is not (separate populations, typed membership endpoints).

## What changed

- **Overlay.** Per population: `ordinal → Option<bool>` flag overrides and `ordinal → Option<Dn128>` location overrides.
  - Created rows are set in place.
  - A delete drops the node's overrides.
  - Setting a value back to the observed one removes the override (net effect only).
- **Readers of the simulated flag.** `active_users`, `live_owners` and `node_state` all read the overridden flag, so rules and validation see the simulated value. An unknown flag is never active.
- **`subtree`.** A moved base node is gated out of the base run, and its new location runs as a delta row. No new query primitive.
- **`diff`.**
  - Property drift is reported as `SetActive` / `SetLocation`.
  - `diff_shared` walks every overridden ordinal.
  - Across observations, a kind change is reported as delete + create.
- **`plan` / reconcile.**
  - The flag and location intent is rebased on the latest observation.
  - A create that meets an existing node converges every property.

## Gates

Disable runs, each red:
- the active plane reads the observed base;
- `None` is collapsed to `false`;
- the compare-and-set `from` is ignored;
- `diff` misses the overrides;
- `subtree` ignores moves;
- reconcile drops the flag intent;
- a property change copies a plane (allocation test);
- a kind mismatch is converged;
- the flag precondition is read from the target;
- the location precondition is read from the target.

## OPEN

- **One change list cannot set a property of a node it creates.** Property changes sort before creates (OGAR's safe order), so the create's own `state` is the only way.
- **Uniqueness counts only active owners.** A plan that enables a user whose address another user holds passes simulation, though Exchange rejects duplicate addresses regardless of the enabled flag. This is a validator question, not one of these changes.
- **No rule proposes flag or location changes yet.** They come from diffs, or from a caller-built change list.
