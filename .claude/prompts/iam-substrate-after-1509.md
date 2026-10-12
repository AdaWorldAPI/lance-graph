# Claude Code implementation prompt — D-IAM-SAG-0 (start from merged #1509)

> Use this as an **implementation-session prompt**, not as an instruction to deploy/write to any identity provider. The present PR is documentation only. Start from the then-current `main`, which must contain [#1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509); record actual SHAs before code changes. If the interfaces have changed, report drift rather than applying this prompt verbatim.

## Mission

Explore and, where evidence permits, implement a **small read-only IAM reasoning slice** over existing `lance-graph-dir-sim`, OGAR policy, Quack/mask-risc and the #1509 Software Awareness Graph. The six marketing categories (IGA, PAM, SSO, Directory, Device/App, Automation) are *domain views*, not six products to build. Deliver a reproducible access-path explanation and before/after permission-impact simulator with strict provenance, not a replacement for SailPoint, CyberArk, Okta, Ping, Entra or Intune.

**Must read first:**
1. `.claude/plans/iam-substrate-sag-dir-sim-v1.md`
2. `.arxiv/2026-10_iam_access_paths_temporal_provenance_primary_excerpts.md`
3. `.claude/plans/software-awareness-graph-v1.md`; `crates/lance-graph-glove-parity/src/awareness.rs`, `tests/awareness.rs`
4. `.claude/plans/class-method-schema-hotplug-v1.md` and `.claude/knowledge/class-method-hotplug-existing-contracts.md`
5. `crates/lance-graph-dir-sim/src/{lib,store,view,actors,snapshot,validate,image,log}.rs`
6. `crates/lance-graph-quack/src/bind.rs`, `crates/lance-graph-rbac/src/lib.rs`, `.claude/plans/dir-sim-persist-v1.md`.
7. Relevant OGAR authority: `ogar-dir-sim`, `ogar-rbac::group_policy`, `ogar-loco::Inventory`, `OGAR #357` (only if current code confirms those imports).

## Strict method

- **Audit before writing.** Produce a row-by-row `[G] grounded code / [P] proposal / [ABSENT] missing` matrix for all functions and authority boundaries. List file:line, exact tests and versions; prior plans can be stale. Confirm #1509's one-hop limitation; check D-SAG-1-4 and D-CMH-G1-4 status on current main.
- **No duplicate engine, catalog, registry or opcode.** No graph traversal in bespoke Rust loops unless it is an explicit bounded reference oracle for tests. Extend an existing valid relation carrier / `EqU32Via` approach only after parity proof; never feed an output mask into a second program as an undeclared population.
- **No authority drift.** OGAR owns classes/action semantics; OGIT owns imported canonical field identity; `GroupRolePolicy` owns group-role grants; `dir_sim` owns hypothetical directory state; external IAM/PAM/IdP/MDM remain enforcement sources.
- **No semantic byte drift.** The 8-bit lanes are the substrate ISA/ALU register contract. Do not reinterpret them as universal NARS/CARE/IAM semantics, modify CE64 bits 59..63, or conflate full u32 class identity with the high-u16 canonical concept.
- **No side effects.** Do not call PowerShell, Azure Graph write APIs, LDAP modifications, PAM sessions or provisioning connectors. `do()` / ActionDef remains inert until a later separately authorized PR. No new network I/O path is permitted in `lance-graph-dir-sim`.
- **Security fail-closed.** No missing-policy->allow, unknown->false, distribution-group->security-group, forged-method->verified, schema mismatch->default, confidence->permission, or time-travel replay->live authorization.

## Proposed sequence, narrow increments

**A. Inventory + fixture first.** Build a synthetic tenant (Anna, Finance, ERP_Accounting) with source GUIDs, DN, two possible membership paths, one directory distribution group, one policy grant, one missing policy, a device with unknown compliance and a separate immutable source observation. Declare source version, policy revision and observed_at.

**B. Pin the existing role oracle.** Ask dir_sim GroupRoles/role_evidence and RoleImpact::between before/after a simulated `RemoveMembership`; record `Held/NotHeld/Undetermined/NotDeclared`, witness and alternate path. Ensure simulation does not mutate base observations. Prefer reusing existing tests and adding a consumer-level contract test, rather than reimplementing role traversal.

**C. Introduce the minimal *read-only* canonical IAM projection**, if justified. Source GUID, Entra objectId, sync anchor, tenant and mutable DN must remain different concepts. Explicit edges `MEMBER_OF`, `GRANTED_ROLE`, `GRANTS`, `REQUIRES`, `EVIDENCED_BY`, `DERIVED_FROM` with version/knownness; do not mint global new IDs where the native GUID can be held as a typed source key. Explicitly label `one-witness` or `exhaustive`.

**D. Link methods to SAG with actual compiler evidence.** If a `MethodSpec` can be prepared, project it using #1509's `SoftwareGraph::project`. Source of evidence must be `MethodTable::prepare`, real lowered program and observed tests. No generic IAM method table cloned from a diagram; no hand-written `Verified` edge or `MayRequire` claiming shipped behavior. Include one refused method for incomplete field/reading/policy.

**E. Check multi-hop legally.** Compare relation carrier vs `EqU32Via` using a small known-reference evaluator; refuse or document blocked if neither existing lowering can carry two-hop queries. Do **not** write a shadow DFS/BFS execution engine to make the demo seem green. An existing directory membership path API is allowed only within its own declared semantics.

**F. Counterfactual and impact.** Run `observe -> simulate -> validate -> diff -> role impact -> explain -> propose` entirely in memory. Model alternate grant and policy revision changes. Distinguish a path disappearing, a role disappearing, a graph entitlement disappearing, and real IdP access being denied. The latter is unknown without actual enforcement evidence.

**G. Optional observer (separate experimental module/doc, never authority).** Inspired by NARS/ONA and Hyperon, inspect SAG refusals, undetermined policies and non-independent evidence. Rank a *read-only query or ablation* by budget and declared uncertainty. Compare to no-observer baseline; avoid psychological/metacognitive adjectives as claims of implementation. Rubicon pre/post-decision and CARE/CE64 readings are analogies for staged probes, not substituted policy gates.

## Mandatory falsifiers

Implement or visibly report blocked for IAM-01..15 in the plan; in particular:
- cross-tenant UPN alias collision / DN rename;
- nested security vs distribution group;
- two valid access paths and one deleted;
- absent grant-policy observation vs explicit empty policy;
- inconsistent source/policy versions;
- three-valued device compliance;
- missing validity plane and stale schema seal;
- simulated state leaves original untouched and replay matches;
- counted duplicate source events do not inflate corroboration;
- attempted DO with no permission cannot reach any provider connector.

For each implemented guard, disable it once in an isolated run and confirm its named test goes red; restore afterwards. If a test is green after disabling its guard, improve the test before claiming coverage.

## Results and communication

Deliver:
1. A concise baseline table with exact existing/absent boundaries, including #1509 one-hop + SAG in-memory limits.
2. Changed files and the *one smallest* implementation slice; keep separate follow-on issues/PRs for multi-hop, device/IdP adapters and authorized effects.
3. `cargo test` / `cargo clippy` commands and actual pass/fail counts for relevant crates, including workspace-exclusion caveats. Never invent metrics.
4. One runnable end-to-end `explain -> simulate -> compare` example with observable provenance and expected refusal.
5. A red/green falsifier ledger with disable-run evidence.
6. An updated plan with tests, method/source revisions and a security statement about why provider writes are impossible.
7. An **OPEN questions** table where data is unavailable; do not replace missing evidence with inference.

**Definition of done:** one well-specified, tested, read-only IAM access-path/impact claim and SAG-aware method evidence, with no second engine and no unproved external authority. If it cannot compile legally, deliver an explicit, code-grounded blocker instead of a fake implementation.
