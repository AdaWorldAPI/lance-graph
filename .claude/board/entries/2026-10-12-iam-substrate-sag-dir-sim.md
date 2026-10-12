# 2026-10-12 — D-IAM-SAG-0: six IAM views over dir_sim and #1509 SAG

**Status:** RFC / RESEARCH-ONLY; no code, benchmark, provider write, or IAM policy change in this PR. **Base:** merged [#1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509), merge SHA `42fa2a347cda22eaaa327bba514f2029cc9daa82`.

**Plan:** [iam-substrate-sag-dir-sim-v1](../../plans/iam-substrate-sag-dir-sim-v1.md). **Research:** [primary-source IAM excerpts](../../../.arxiv/2026-10_iam_access_paths_temporal_provenance_primary_excerpts.md). **Prompt:** [Claude Code implementation handoff](../../prompts/iam-substrate-after-1509.md).

## Decision

Treat IGA, PAM, access/SSO, directories, device/app and automation as **overlapping projections** of a typed, provenance-aware IAM graph, never as functions automatically supplied by the graph itself. Reuse OGAR/dir_sim, GroupRolePolicy, ClassView/WideFieldMask, Quack/R2IL and #1509 SAG. The native provider retains authentication, sessions, policy enforcement and live-write authority.

## Concrete existing contracts

- `dir_sim::VersionStore` observes/simulates/diffs/plans over immutable roots plus delta overlays.
- `dir_sim::actors` provides source-version-specific group-role evidence, indeterminate policy states, and role impact.
- #1509 `SoftwareGraph` projects prepared methods/compiled programs/provenance tiers and offers Quack one-hop queries; it is in memory and does not yet implement multi-hop composition.
- D-CMH-0 specifies a per-plug class/field/method binding seam; OGAR #357 is an RFC, not production executor.

## Work gates

| ID | Planned slice | First falsifier |
|---|---|---|
| D-IAM-SAG-0-0 | source, authority and identity inventory | AD GUID != DN, Entra objectId and alias cannot silently unify |
| D-IAM-SAG-0-1 | read-only role provenance and impact fixture | alternate path preserves access; never-read group remains Undetermined |
| D-IAM-SAG-0-2 | class/method binding and SAG projection | missing field, stale seal and uncompiled method explicitly refuse |
| D-IAM-SAG-0-3 | legal multi-hop entitlement explanation | parity vs oracle; no output-Mask-as-LIVE shortcut |
| D-IAM-SAG-0-4 | temporal policy, device evidence and proposal | unknown compliance never becomes allow |
| D-IAM-SAG-0-5 | optional ONA/Hyperon-inspired probe prioritization | observer ablation never changes policy authority |
| D-IAM-SAG-0-6 | authorized DO boundary, separate PR | forged confidence/approval cannot generate provider write |

**Principal boundaries:** observed ≠ inferred ≠ authorized; one provenance witness ≠ exhaustive proof; group membership ≠ successful SSO; simulated role impact ≠ executed revocation; append-log version ≠ observation/policy effective time. No new ISA bit reading or CE64 mutation.

**Implementation expectation:** execute [the prompt](../../prompts/iam-substrate-after-1509.md) against actual code, ship narrow test-pinned W1/W2 first, and leave unfinished integration explicitly open.
