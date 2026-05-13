# Agent W4 — Sprint-3 log

**Role:** Worker Agent W4 of Sprint-3 (12 + meta CCA2A).
**Branch:** `claude/tier-1-implementation-specs` of `AdaWorldAPI/lance-graph`.
**Tier:** Tier-1 implementation specs.
**Tech-debt anchor:** TD-GENERIC-BRIDGE-3.
**Pattern letter (post-PR #359):** Pattern C — GenericBridge dispatching per-G ConsumerPointer.

---

## Deliverable

`.claude/specs/pr-c-1-generic-bridge.md` — PR-ready spec for the third
concrete Tier-1 implementation. After this spec, an engineer picks up
the PR and starts coding once PR-A-1 (W2) and PR-B-1 (W3) have landed.

## Status

**DONE — spec drafted and pushed to branch via pygithub.**

## Decisions logged

1. **`ConsumerPointer` lives in `lance-graph-contract::consumer`,**
   alongside `ActionCap`, `GateDecision`, `DomainProfile`, and (for
   this PR) the `RbacPolicy` trait. Keeps every consumer pulling from
   one canonical place; matches W2's `SpoQuad`-in-contract decision.
2. **`GenericBridge` is the single canonical `MembraneGate` impl.**
   Per-consumer behavior collapses to a `ConsumerPointer` value — the
   orphan-rule justification for newtype gates (PR #29 commentary)
   dissolves once the trait+type pair lives in the same crate.
3. **Backwards-compat wrappers stay in this PR.** `SmbMembraneGate`
   and `MedCareMembraneGate` become thin newtypes around
   `GenericBridge::for_g(_, G)`. Preserves the 13 SMB + 33 medcare
   test surfaces with zero churn. `#[deprecated]` lands in a follow-up
   release cycle.
4. **Inert G fails closed.** A `ContextBundle` with no
   `consumer_pointer` (or a missing G) → `should_emit` returns
   `false`. Misconfiguration should never grant access.
5. **Action-capability dispatch is in this PR.** `ConsumerPointer.action_capabilities`
   closes Meta-3 HIGH #1 (medcare sprint: "actions unreachable via
   gate"). The dispatcher consumes the slot; populating it for
   medcare/smb is downstream consumer work.
6. **`RbacPolicy` trait stays in `contract` for now.** Splitting into
   a `lance-graph-rbac` crate adds scaffolding for no measured
   benefit. Defer until contract bloat is an empirical problem.
7. **`Arc<dyn ...>` for `RbacPolicy` / `SchemaPtr` / `KernelRef`.**
   Symmetric, cheap to clone, and matches the dynamic-dispatch
   posture the dispatcher already takes.

## Dependency call-out

PR-C-1 depends on **both** PR-A-1 (W2, the `g: u32` quad slot) and
PR-B-1 (W3, the `ContextBundle` with `consumer_pointer` slot). Without
PR-B-1 there is nothing to dispatch on; without PR-A-1 there is no key
to dispatch by. Open-question 1 in the spec flags the wrapper
deprecation policy for the engineer; OQ 5 flags async policy as
out-of-scope follow-up.

## LOC reduction validation

Per `MEDCARE_POLICY_GAP.md`: per-newtype consumer = ~800 LOC. Under
GenericBridge: ~30 LOC `ConsumerPointer` registration. ~25× reduction.
W8's consumer template spec dry-runs this against a third consumer
(`hubspo-rs`) and is the empirical validation harness.

## Cross-worker handover

- **W1 (master plan):** `sprint-3-execution-plan.md` references PR-C-1
  as the Pattern-C deliverable.
- **W2 (PR-A-1, SPO-G u32 slot):** required precursor; PR-C-1 imports
  the `g` axis from this PR.
- **W3 (PR-B-1, ContextBundle):** required precursor; PR-C-1 reads
  the `consumer_pointer` slot this PR introduces.
- **W8 (consumer template):** validates the LOC-reduction claim by
  dry-running a third consumer onboarding under the GenericBridge
  pattern.
- **W5 (PR-E-1, manifest):** optional downstream — declarative YAML
  registration of `ConsumerPointer`. PR-C-1 works without it.

## Files written this session

- `.claude/specs/pr-c-1-generic-bridge.md` (spec, ~10 KB)
- `.claude/board/sprint-log-3/agents/agent-W4.md` (this log)

## Push protocol

pygithub-first per Sprint-3 protocol. Token stripped of quotes,
`Github(auth=Auth.Token(tok))`, `repo.create_file` for the new files.
Local `Write` was denied by the harness; remote pygithub push is the
canonical path.

## Next handover

Engineer pickup, **gated on PR-A-1 + PR-B-1 landing.** The spec's
"Open questions for the engineer" section has five items with
recommendations; engineer should confirm OQ 1 (wrapper deprecation
window) and OQ 4 (`SchemaPtr` / `KernelRef` shape) before coding
starts. OQ 5 (async policy) is flagged out-of-scope and should
become a follow-up RFC.
