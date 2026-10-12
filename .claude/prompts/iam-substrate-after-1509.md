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

## Research extension D-IAM-SAG-1 (required reading, not new scope authorization)

The original task intentionally focused on the positive role path. Also read, before implementing a general IAM policy surface:

- [Frontier architecture addendum](../plans/iam-substrate-frontier-gaps-v1.md), with a decision/authority/time ontology, release train and IAM-16..40;
- [Why-not provenance, robust access policies and agentic reference monitors](../../.arxiv/2026-10_iam_negative_provenance_authority_and_agentic_policy_excerpts.md), with original arXiv/OASIS/IETF sources, transfer limits and falsifiers.

**The extension does not enlarge the first implementation slice.** If this prompt is used for the first Rust PR, implement only a smallest read-only, test-pinned path; mark the remaining requirements open. The extensions guard against accidentally shipping a positive-graph demo as a full IAM policy engine.

### X1. Answer types and data completeness

Design or evaluate (do not prematurely expose a new contract API) **three distinct result planes**:

1. Evidence: positive witness, complete scoped absence, incomplete/unknown or contradictory sources;
2. Policy: Permit, Deny, NotApplicable, Indeterminate (including the *explicitly selected* policy-combining algorithm);
3. Provider: actual accepted/rejected/no observation, which the graph must never infer from a permission edge.

Extend the original synthetic fixture with a missing page, incomplete group-policy read, explicit empty policy, explicit deny, and a changed combining algorithm. Show **which authority** was read and why the negative conclusion does or does not follow. A closed-world claim must cite its complete source scope and versions. Do not collapse absent evidence, unknownness and Deny into false.

### X2. Multi-path, negation and bounded completeness

- Produce two independent granting paths, remove only one, and verify the entitlement remains. Use `RoleImpact::between` for directory role impact; do not equate its one shortest path with exhaustive reachability.
- Record `one_witness / exhaustive / incomplete_due_to_budget` on every explanation. Explicitly refuse a universal “no access” claim where source coverage is incomplete or path enumeration is bounded.
- Include `why_not_access` fixture; test against a **separate small reference model**. Do not silently implement Datalog/XACML/provenance as a second production engine.
- A minimal set of membership removals is not automatically the lowest-risk or authorized intervention. Include business-policy constraints in the simulation oracle.

### X3. Source identity and four time concepts

- Verify source AD GUID, Entra objectId, SCIM id, canonical NodeGuid, full classid, mutable DN, user alias, tenant and SIDHistory are separate unless a **reviewed and versioned** mapping explicitly links them.
- Test UPN reuse, two forests with same sAMAccountName, one-to-many sync anchor ambiguity, and a service/workload principal with no human HR record. No privilege inheritance from probabilistic matching.
- Report effective-at, observed-at, recorded-at, and decision/execution-at separately; use a **version vector** for AD/Entra/HR/MDM/policy rather than pretending `VersionStore::VersionId` is a cross-provider transaction.
- Under source clock skew, partial snapshots or changed policy revision, either refuse a consistency-sensitive claim or explicitly downgrade it. Historical permission never yields present write permission.

### X4. Lifecycle, SoD and JIT as policy problems

Use synthetic examples for: moved employee with a temporary double entitlement, leaver with a stale app session, mutually exclusive roles from two sources, orphaned automation principal, privileged JIT grant that expires, break-glass requiring a separate approval. Refuse a “verified safe” claim when versioned policy constraints are unavailable. Treat robust-under-future-policy-extensions as a research problem, not an achieved theorem.

### X5. Independent executable authority and protocol receipts

No actual provider writes are allowed in this PR. For future protocol contracts, specify:
```text
proposed != approved != requested != applied != re-observed != verified
```
A provider timeout may mean success or failure: **UnknownOutcome**, re-observe before retry. If conditional resource versions/ETags are unsupported, fail safely or explicitly require a different authorized concurrency strategy. SCIM per-resource PATCH atomicity is not cross-provider rollback. Do not let a natural-language prompt, SAG `Tier::Verified`, a CE64 confidence code or a speculative CARE/ONA observer supply authorization. Do not imply revoking a group membership terminates an already issued privileged session.

### X6. Negative-proof privacy and information flow

The answer `why does Anna have access?` can expose privileged group and policy names. Include a principal/tenant/scope-bound explanation gate, field redaction and completeness labeling. A request may legitimately obtain the policy decision but **not** all witness edges. Redaction must not change the underlying decision, and graph traversal must not cross tenant scope. `WideFieldMask` is a projection, not an authorization decision.

### X7. Property-based/metamorphic and disable-run suite

Track IAM-16..40 from the frontier plan. Essential metamorphic checks:
- Source-row order and graph-edge order permutation do not alter effective answers.
- Replaying one upstream event from two adapters does not double-count license or evidential confirmation.
- Adding an independent allow path cannot remove a role grant **under a declared purely positive graph rule**, but this monotonicity must NOT be generalized to deny rules or mutable policies.
- Alternative grant path removal must not imply entitlement loss.
- A source policy version change invalidates a stale certificate if the rule is not proved robust under the permitted extension.
- Actor row-scope, field authorization, source coverage and executable-effect gates each have a named disable-red test.
- Cyclic 1000-path fixture has a stated cost/time/memory bound and returns `incomplete` rather than a false exhaustive proof.
- Observer/no-observer with identical inputs produces identical authorized results, even if query cost and discovery order change.

**Stop sign:** If a capability requires inventing a second policy evaluator, traversal runtime or authority registry to make tests pass, STOP and write an explicit blocker and narrow follow-on PR. Don't mislabel inferred behavior as compiled/verified.

### X8. Handoff output expansion

In addition to the earlier deliverables, report:
- an access-decision/evidence/provider triple for each fixture, including unknown cases;
- exact completeness scope and source/policy version vector;
- one-witness vs exhaustive explanation with witness budget;
- a source-anchor collision ledger and privacy review;
- result parity and disable evidence, including IAM-16..40 as Pass/Fail/Blocked/Not attempted with **no fabricated green**;
- what remains research-only, what is code-grounded, and what would require new authority/legal/security review.

**Ultimate invariant:** graph inference may *propose* a safe action, but it cannot *become* the principal that authorizes it.

### X9. Refresh the baseline for #1514 and the D-PBL device policy proposal

Before doing any traversal work, inspect merged [#1514](https://github.com/AdaWorldAPI/lance-graph/pull/1514), `crates/lance-graph-glove-parity/{src/gremlin.rs,tests/relation.rs}` on current main and [the frontier update](../plans/iam-substrate-frontier-gaps-v1.md#12-post-1509-implementation-drift-1514-earns-one-relation-hop-g-on-main). Correct older “no relation carrier” statements:

- **Available on main:** one functional FK `out(rel)`, liveness via `Semijoin`, filtered target property via `EqU32Via`, anchor preserved, tested in Quack/mask-risc on a synthetic ERP fixture.
- **Still missing:** general many-to-many group memberships/fan-out, `out().out()`, target `values`/sum/dedup, R2IL `GATHER` parity, multi-hop SAG queries. Do not retrofit IAM's recursive group closure into a functional FK without independently proving multiplicity semantics.
- **D-PBL-0..7** in `.claude/plans/policy-baseline-integration-v1.md` is proposal only: GPO configured vs assigned vs observed differs from Intune configured vs assigned vs per-device observed; D-PBL explicitly lacks device ingestion and canonical GPO↔CSP mapping unless sourced. No “compliant” result from an assigned policy alone.
- **New gates IAM-41..45** in frontier plan check Intune exclusions and unknown device membership, GPO observed unknown without RSoP, unmapped policy-setting identity, one-hop Quack/R2IL asymmetry, and rejection of false functional cardinality.

Report the current SHA and whether #1514 has landed in the implementation checkout. This research PR branch is intentionally still based on #1509; implementations should start from the then-current main or explicitly rebase before testing. Stop if a baseline assertion cannot be reconciled with code.
