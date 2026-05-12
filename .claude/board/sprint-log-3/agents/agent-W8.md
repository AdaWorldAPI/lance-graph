# Agent W8 — Sprint-3 log

**Role:** Worker Agent W8 of Sprint-3 (12 + meta CCA2A).
**Branch:** `claude/tier-1-implementation-specs` of `AdaWorldAPI/lance-graph`.
**Tier:** Tier-2 architecture-validation deliverable.
**Architectural anchor:** Patterns A + B + C + E + F (consumer-side
end-to-end glue path).
**Worked example:** `hubspo-rs` (HubSpot CRM consumer).

---

## Deliverable

`.claude/specs/consumer-crate-template.md` — step-by-step recipe for
adding the Nth consumer crate, with `hubspo-rs` as the worked example.
The spec is the **architectural proof** that the new Patterns A + B + C
+ E + F surface collapses per-consumer cost from ~800 LOC (medcare-rs
baseline, PR #98) to ~100-150 LOC (hubspo-rs target).

## Status

**DONE — spec drafted and pushed to branch via pygithub.**

## Decisions logged

1. **Spec is a recipe, not an API contract.** Unlike PR-A-1 / PR-B-1
   / PR-C-1 (which spec a code change), W8 specs a **process** — what
   files a consumer-author writes, in what order, with what total LOC
   budget. Validation criteria belong to the architecture, not the
   consumer.
2. **`hubspo-rs` chosen over alternatives** (e.g. `salesforce-rs`,
   `pipedrive-rs`) because (a) HubSpot has a published OWL-friendly
   ontology, (b) CRM is a fresh OGIT slot — no overlap with existing
   consumers, (c) the manifest's escalation set
   (`close_deal`, `delete_contact`, `send_bulk_email`) exercises the
   `action_capabilities: escalate` path that medcare's regulatory
   moves stress-tested upstream.
3. **LOC budget = ~100 LOC (no hydrator) / ~150 LOC (with hydrator).**
   Beats W1 master plan's 30-LOC headline once you net out the
   per-domain `HubSpoMessage` variants and `policy.rs` (both domain
   logic, not architectural overhead). The headline ~30 LOC = the
   manifest alone; the realistic ~100 LOC = manifest + minimal crate
   scaffolding.
4. **Validation gates spelled out:** (a) <300 LOC of glue,
   (b) <1 engineer-day, (c) zero upstream changes. If hubspo-rs blows
   any of these, escalate as architectural debt **before** the next
   consumer (e.g. salesforce-rs) lands.
5. **Hydrator placement recommendation.** Step-5 hydrator should live
   in `lance-graph-ontology` (upstream), not in `hubspo-rs`. Lets
   ontology-only builds run without compiling the live actor — useful
   for ontology browsing / OWL validation passes.
6. **Open question 1 surfaces a real wiring decision.** The symbolic
   `ogit_g: CRM` value needs to land in
   `lance-graph-contract::ogit::OGIT` as a const so the consumer can
   write `const G: u32 = OGIT::CRM` without a runtime string lookup.
   Flagged for PR-B-1 to handle when finalising the OGIT slot
   allocation.

## Architectural proof claim

PR #98 medcare-rs integration: **~1865 LOC** across three sequential
stages (medcare-rbac + medcare-realtime + MedCareMembraneGate, per
`.claude/board/MEDCARE_POLICY_GAP.md`).

Projected hubspo-rs integration under Patterns A + B + C + E + F:
**~100-150 LOC**.

Reduction: **12-18× per consumer.** The orphan-rule tax is paid once
upstream (PR-C-1's `GenericBridge`) instead of N times across N
consumers. If the dry-run validates within budget, this moves from
CONJECTURE to FINDING.

## Dependency call-out

W8 spec **describes** an end-to-end recipe consuming the outputs of
W2 / W3 / W4 / W5 / W6. None of those PRs need to land for the spec
itself to be useful — the spec is the contract those PRs collectively
fulfill. But the **dry-run validation** of the spec (actually
scaffolding hubspo-rs) cannot start until at least PR-B-1 + PR-C-1 +
PR-E-1 have landed.

## Cross-worker handover

- **W1 (master plan):** `sprint-3-execution-plan.md` lists W8 as
  the architecture-validation deliverable for the sprint.
- **W2 (PR-A-1, SPO-G u32 slot):** consumer-author writes to the `g`
  slot via `HubSpoActor`; spec assumes PR-A-1 surface.
- **W3 (PR-B-1, ContextBundle):** consumer-author looks up `OGIT::CRM`
  via the registry; spec assumes PR-B-1 surface and flags the const
  addition as open question 1.
- **W4 (PR-C-1, GenericBridge):** consumer-author depends on the
  bridge dispatching by `ConsumerPointer`; spec assumes no
  per-consumer newtype gate.
- **W5 (PR-E-1, manifest modules):** consumer-author writes
  `modules/hubspo/manifest.yaml`; spec uses PR-E-1's schema verbatim.
- **W6 (PR-F-1, ractor supervisor):** consumer-author's `HubSpoActor`
  is spawned by the canonical supervisor; spec assumes manifest
  enumeration at boot.
- **W9 (PR-D-1, FMA hydrator):** sister Pattern D hydrator; precedent
  for the optional Step-5 ontology hydrator path.

## Files written this session

- `.claude/specs/consumer-crate-template.md` (spec, ~10 KB)
- `.claude/board/sprint-log-3/agents/agent-W8.md` (this log)

## Next handover

**Validation pickup** — once PR-B-1 + PR-C-1 + PR-E-1 land, a
consumer-author runs the recipe end-to-end against `hubspo-rs` and
reports the actual LOC count + elapsed time. Three pass/fail gates
in the spec's "Validation criteria" section determine whether the
architectural claim holds. Failure on any gate triggers an
architectural-debt escalation entry in
`.claude/board/TECH_DEBT.md` BEFORE the next consumer lands.
