# D-IAM-SAG-1 — Frontier addendum: negative proof, temporal authority, and safe introspection

> **Aligned by [`sag-capstone-alignment-v1.md`](sag-capstone-alignment-v1.md)** — read it first: canonical `ProposedMethod`, the shared status vocabulary, D-id map, and the claims it marks stale (incl. "second hop refused").

> **2026-10-12 / RFC only / deliberately unshipped architectural hypotheses.** Extends [D-IAM-SAG-0](iam-substrate-sag-dir-sim-v1.md) in draft [#1512](https://github.com/AdaWorldAPI/lance-graph/pull/1512), based on merged [#1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509). [Primary literature and standards excerpts](../../.arxiv/2026-10_iam_negative_provenance_authority_and_agentic_policy_excerpts.md). Existing [Claude implementation prompt](../prompts/iam-substrate-after-1509.md) is extended accordingly.
>
> **Status discipline:** [G] = confirmed in cited code or #1509; [R] = result from cited paper/standard **within its own assumptions**; [H] = proposed mapping not proved in lance-graph; [B] = implementation blocker. An RFC is not implementation. No change to existing ISA/byte contract, OGAR/OGIT mint, any directory schema or authorized provider behavior.

## 0. The question #1512 did not fully resolve

**How can a read-only graph explain both permission and *its absence*, distinguish an incomplete observation from a negative fact, and propose a safe smallest change without accidentally becoming the authority it observes?**

A minimal `Identity MEMBER_OF Group GRANTED_ROLE Role` path is enough to *witness* a graph grant. It is insufficient to prove that:
- no other grant path survives removal;
- an explicit deny has higher precedence than an allow;
- all authoritative sources have been observed;
- the device, network, session and time restrictions hold;
- an externally authenticated session was actually accepted;
- a proposed action is currently authorized or reversible.

The addendum separates **evidence closure**, **decision semantics**, **causal impact**, **operational execution** and **cognitive prioritization** as independent proof obligations.

## 1. Three separate results (do not coerce to bool)

Proposed *analytical* result sidecar, not a new zero-dependency `lance-graph-contract` type:

```text
GraphEntitlement =
  Witnessed{one_or_more_paths, completeness: WitnessScope}
  | AbsentUnderClosedWorld{scope, source_versions, exhaustiveness_proof}
  | Undetermined{missing_sources_or_policy}
  | Conflicted{incompatible_observations}

PolicyEvaluation = Permit | Deny | NotApplicable | Indeterminate
  + selected_combining_algorithm + policy_version + obligations/advice

ProviderEnforcement = ObservedAccepted | ObservedRejected | NoObservation
  + provider + session/request id + expiry + observed_at
```

`Deny` is not `NotApplicable`; `NotApplicable` is not `Undetermined`; *absence of a row* is not an authoritative negative unless the relevant source/query has a complete and scoped negative certificate. The policy-combining algorithm is source-specific and versioned. Never choose `deny-overrides` by default merely because a paper or an IAM product supports it: XACML 3.0 has several algorithms and a richer Indeterminate space. [OASIS XACML 3.0](https://docs.oasis-open.org/xacml/3.0/xacml-3.0-core-spec-os-en.html) is the reference contract [R], not evidence of a shipped XACML engine in LG [H].

**Existing nearby [G]:** `dir_sim::actors::RoleEvidence` already distinguishes `Held`, `NotHeld`, `Undetermined`, `NotDeclared` for a declared role under its own documented group-policy completeness checks. The proposed cross-domain proof is a *narrow extension*, not a reinterpretation of that API.

### Completeness as first-class evidence [H]

Represent read scope and coverage for each authority:

```text
ObservationCoverage {
  source, tenant_or_realm, object_kinds, predicate_subset,
  snapshot_or_cursor, page_window, source_version,
  authoritative_for, complete: yes|no|unknown,
  observed_at, read_scope, retention_policy
}
```

A successful paginated read can still be incomplete (partial sync, missing page, insufficient privileges, filter mismatch). Closed-world negative proofs require **all relevant** sources, policy domains and time intervals to be complete, not merely one green source read. Distinguish `Some([])` from `None` in OGAR GroupRolePolicy as current dir_sim already does.

**Falsifier IAM-16:** remove a data page containing the alternate granting path. An apparent `Absent` must become `Undetermined`; never permit deletion just because the narrower observation looks empty.

## 2. Four temporal coordinates and a coherent authority vector [H]

Never equate:
- **valid/effective time:** when the grant, role, device compliance or policy purports to apply;
- **observation time:** when a source response was obtained;
- **record/ingest time:** when this fact was durably appended;
- **decision/execution time:** when a caller asks or an external PDP authorizes an action.

Persist a coherent query envelope, conceptually:

```text
QueryEnvelope {
  source_versions: {AD:v7, Entra:e12, HR:h5, MDM:m8},
  policy_versions: {group_role:p9, conditional_access:ca3},
  canonical_schema_seal, prepared_method_seal,
  effective_as_of, read_snapshot_horizon, observation_cutoff,
  scope, tenant, query_purpose, witness_budget
}
```

Source versions form a **vector**, not a single global monotonically increasing `VersionId`. An atomic multi-system snapshot is **not** established simply because each source was read at some time. A user request may:
1. Refuse a consistency-sensitive conclusion with `MixedVersion`;
2. report bounds as of a declared observation cut; or
3. use an explicitly weaker “best-effort” comparison with stale-source badges in a UI (never silently promote to verified live access).

Tests: out-of-order observations, policy edited between two reads, source clock skew, delayed HR termination, v7 replay after provider has advanced to v9. A historical positive should never silently become a *present* execution grant.

## 3. Identity equivalence as a *proofed relation*, not string equality [H]

`ExternalIdentityBinding(source, namespace, tenant, source_guid, canonical_id, link_rule, rule_version, verified_anchor, confidence/status, valid_interval)` is a relation, not a claim that equal UPN/displayName/mails are unique. DN is location. ObjectId (Entra), objectGUID (AD), SCIM id, hrPersonId, app account id and workload principal are different identities until explicit verified mapping.

**Crucial counterexample:** A AD and B cloud share `anna@example.com` during reassignment; a former user's SIDHistory could grant access. A naive coalescing keyed on mail merges two people and falsely preserves authorization. Another: two authoritative AD forests yield equal-looking `sAMAccountName`; never cross realm boundary.

Only treat `same_actor` as equivalence *inside declared scope*, from an approved anchor rule. A link's confidence belongs to epistemic matching, **not** RBAC delegation. Transitivity of `CORRESPONDS_TO` needs a separately proven anchor logic; otherwise collision chains A≈B≈C can create privileges.

**IAM-17/18:** UPN reuse or source-namespace collision cannot merge identities; ambiguous many-to-many binding explicitly refuses inherited roles. Include machine/service identities, managed identities, API principals and emergency accounts; do not assume every principal has a human HR lifecycle.

## 4. Access paths form an AND/OR proof, not a simple line [H]

A grant chain may require *all* conditions (member AND active policy AND correct device AND within time) while authorization can have *alternative* paths (group A OR role B). Policy **deny** may override multiple valid grant witnesses. Standard provenance polynomial intuition is helpful, but Boolean edge reachability alone loses constraints and can leak sensitive intermediate memberships.

Proposed normalized explanation:

```text
EntitlementClaim(actor, action, object, context)
  Witness(W1) = member(A,G1) AND security_enabled(G1) AND grant(G1,R)
                AND role_allows(R,action,object) AND policy(p1)
  Witness(W2) = direct_app_assignment(A,app) AND app_role(app,R)
  Constraints = effective_time AND device_posture AND explicit_exclusions
  Unknowns = unread_policy, stale_session, unresolved_anchor ...
```

**One witness vs complete support:** `RoleEvidence::Held` supplies a *concrete* path, not a full provenance enumeration. For `revoke_preview`, evaluate effective role/entitlement **before and after** using the existing versioned oracle. Do not infer loss from the disappearance of a chosen witness. Cycle detection and a bounded witness budget are mandatory.

**Negative explanations:** use “failed derivations + completeness” for `why_not_access?`, not merely no positive path; [arXiv:1701.05699](https://arxiv.org/abs/1701.05699) provides a database provenance research model for failure/negation [R], not an OGAR implementation guarantee.

**Cut sets / minimal interventions:** propose a *candidate* set of changes that disables all surviving grant paths **subject to** SoD, business continuity and effect-authority constraints. Minimum-cardinality cut is not necessarily least-risk intervention. Counterfactual success does not prove a causal claim about observed production session behavior.

**IAM-19/20/21:** double-path removal preserves access until both paths are closed; empty role result with incomplete sources remains Unknown; explicit deny + permit respects the selected external policy algorithm.

## 5. Approval, SoD and lifecycle as temporal policy, not attributes [H]

Represent:
- **Separation of duties:** mutually exclusive capabilities, policy authority and expiry; include combinations of directory + app roles, not just group names.
- **JIT/PAM:** grant/approval with time-to-live, one-time session authorization, break-glass purpose, session receipts; approval ≠ privilege issuance, issuance ≠ session accepted, revocation ≠ session termination.
- **Joiner/Mover/Leaver:** HR effective time, account creation and delayed deprovisioning; a mover can both lose and gain privileges with a risky overlap window.
- **Nonhuman lifecycle:** app secrets, service account owner, key rollover, workload identity, orphaned automation privileges; never treat a robot identity as a human with no HR record.
- **Delegation and impersonation:** actor, on-behalf-of principal, scope, consent, session and authority must stay separate; propagated identities do not authorize their own scope expansion.

**Policy robustness question:** can a pending role assignment or a later independent policy extension violate a proved SoD invariant? [arXiv:2603.13181](https://arxiv.org/abs/2603.13181) studies robust policy properties under extension [R]. Transferring its logic requires an explicit policy language and proof rules [H]. A one-time point-in-time check is not a proof under all possible policy changes.

**IAM-22/23/24:** SoD remains violated even when conflicting roles derive from different sources; expired JIT grant fails before actual execution; a new policy extension invalidates a certificate unless it satisfies the proved stability preconditions.

## 6. A provider operation is an audited protocol, not a mask-risc side effect [H]

Keep `D-IAM-SAG-0` read-only. A *future separate PR* can specify:

```text
ProposedMethod -> SimulatedImpact -> Human/Policy Approval
 -> Fresh Authorization (actor+resource+verb+scope+purpose)
 -> Provider Preconditions (source ETag/version, policy rev, target identity)
 -> External Call (idempotency strategy, declared retry semantics)
 -> Provider Receipt -> Re-observation -> Verified/Unverified outcome
 -> Optional compensating proposal
```

At every step, keep `requested`, `approved`, `applied`, `verified` distinct. A failed or timed-out provider call creates **UnknownOutcome**, not “not applied”, because retry can duplicate side effects. Compensation can fail and is not automatically rollback of all consequences.

**SCIM interoperability:** [RFC 7643](https://www.rfc-editor.org/info/rfc7643) defines its resource model; [RFC 7644](https://www.rfc-editor.org/info/rfc7644) covers PATCH, conditional ETags and optional provider capabilities. SCIM PATCH is atomic **for one resource request** in the described protocol; this is not a multi-provider transaction. Conditional `If-Match` depends on provider support. Do not generalize that all Graph/AD/PowerShell endpoints support it.

**IAM-25/26/27:** stale ETag refuses mutation; timeout after provider success yields UnknownOutcome requiring re-observation before retry; compensating a group change does not claim to terminate existing IdP/PAM sessions. There are zero external calls in the documentation PR.

## 7. SAG method evidence and safety-proof boundary [H]

[D-SAG-1](software-awareness-graph-v1.md) projects prepared MethodTable methods, lowered programs and compiler refusals. It **does not** establish that a method may read every tenant's identities or execute a privileged action.

Propose a *sidecar receipt*:

```text
PreparedIAMRead {
  method_key, class_full_id, schema_seal, semantic_reading,
  source_versions, policy_versions, required_field_mask,
  row_scope_and_actor_context, compiler_verdict, body_digest
}
```

- `WideFieldMask` selects **columns**; tenant-/actor-scoped row authorization is an independent predicate/gate.
- `Tier::Compiled` proves a lowered program exists, not correct domain meaning. `Tier::Verified` needs real oracle/certificate including source versions and query scope.
- `Tier::Hypothesis` never becomes `authorized` even if confidence is high.
- The Quack one-hop limit remains [B] for general entitlement closure; use existing `dir_sim::actors` only for its scoped role evidence. `D-CMH-G1-4` / `D-SAG-1-4` relation carrier remains open.

**IAM-28/29:** same method changes schema seal or tenant row scope: stale certificate refuses; a `WideFieldMask` containing salary/secret fields cannot bypass row-level authorization or expose hidden explanation edges.

## 8. Observer / ONA / Shannon focus without authority [H]

Separate the subject graph **(source + policy)** from the software graph **(SAG prepared method/evidence)** and from a speculative observer's **attention state**. Never write observed facts simply because an inference appeared credible. Proposed optional experiment:

```text
Missing/Contradictory Evidence (coverage gap, policy conflict, stale method)
    -> rank read-only questions under budget (value of information)
    -> select next source read / provenance query / dir_sim ablation
    -> replay and compare resolved cases, cost and false alerts
    -> create explanatory proposal and falsifiable test (not provider action)
```

Three independent measures:
1. data entropy/surprisal over observed attribute distributions;
2. epistemic uncertainty about claims given dependency-aware evidence;
3. operational severity/risk and effort (policy-grounded, not entropy-derived).

**Conjecture:** a budgeted observer may reduce *unanswered* identity questions at lower query cost. Do not claim better identity decisions without an oracle. [arXiv:2602.16708](https://arxiv.org/abs/2602.16708) argues that agent policy enforcement must live outside prompts, using policy-relevant decision points and observability contracts; it is a reference proposal, not proof a deployed LG reference monitor exists.

**IAM-30/31:** observer-on vs observer-off has identical effective role decisions for the same pinned data; adversarial retrieved notes or user-supplied code suggesting `execute()` never cross the capability boundary.

## 9. Model-checking / metamorphic test grid

In addition to IAM-01..15:

| ID | Deliberately broken assumption | Expected falsification |
|---|---|---|
| 16 | incomplete page/scope looks closed | `Undetermined`, not `Absent` |
| 17 | UPN reassigned between people | no cross-person role carryover |
| 18 | alias/transitive uncertain anchors A≈B≈C | refuse accidental cross-tenant merge |
| 19 | alternate path survives mutation | entitlement still effective |
| 20 | missing proof of negation | cannot issue complete `why_not` |
| 21 | explicit deny alongside permit | versioned combining algorithm governs |
| 22 | role conflicts from two providers | SoD check detects effective conflict |
| 23 | JIT expiry / break-glass event | expired grant refused; session separately known |
| 24 | later policy extension | invalid/stability-qualified invariant certificate |
| 25 | SCIM conditional write conflict | version precondition stops effect |
| 26 | provider success followed by timeout | UnknownOutcome, re-observe before retry |
| 27 | compensation attempted after privilege session | does not claim revocation of session |
| 28 | compiler result reused after schema or actor-scope drift | certificate invalid |
| 29 | WideFieldMask selected private column | row/field authorization still controls visibility |
| 30 | ONA priority changed, observations fixed | factual/policy answer remains bit-identical |
| 31 | malicious note fakes approval | zero executable authority gained |
| 32 | evidence order/permutation changed | same versioned result and witness set |
| 33 | source observation replay duplicated | no extra evidence or duplicate licenses |
| 34 | same principal 1000 paths, cyclic nested groups | bounded deterministic closure; no combinatorial explosion |
| 35 | observed policy/HR time lag | no present authority from stale history |
| 36 | software graph `MayRequire` with no compiled body | never labelled deployed |
| 37 | no user-readable clearance for sensitive witness | redact/refuse explanation without changing decision |
| 38 | tenant filter removed from one relation join | cross-tenant test goes red |
| 39 | two simultaneous provisioning plans | conflict / idempotency gate; no double license |
| 40 | license source precedence and exchange recipient mismatch | preserve proved primary SMTP and existing license; unsupported mapping refuses |

The reference evaluator must be **separate from the optimized implementation**, small and explicitly bounded, not a hidden general-purpose graph runtime. Define test generator seeds, exact source graphs, policy/version seals, expected status, completeness and whether each proof is one-witness or exhaustive. A disable-run counts only when the named test goes red and the original is restored.

## 10. Proposed release train, with stop points

- **P0 (no shipping):** decide formal status vocabulary, source coverage and policy versioning; independent oracles for grant and absence. Publish test vectors IAM-16..21.
- **P1:** read-only `dir_sim` role impact + witness scope, multi-source completeness receipt *as a sidecar*; no new runtime/registry.
- **P2:** canonical identity link evidence, partial observability and version vector; guard tenant/security scope at all joins.
- **P3:** legal multihop relation carrier when D-CMH-G1-4 unblocks, plus SoD/JIT graph projections; no live enforcement.
- **P4:** SAG certificate linkage, redacted explanation query and observer ablation; never produce authorize tokens from beliefs.
- **P5 (different PR, explicit review):** provider adapter contract, reference monitor, conditional writes, receipts and reconciliation; default read-only. Until then `do()` is inert.

**Non-goals:** don't reinvent XACML, SCIM, OpenNARS, Hyperon, a general authorization PDP/PEP, a new R2IL opcode, tenant identity mint or directory writes. Treat identity security and privacy review as gating work, not an afterthought.

## 11. Architectural falsification that can collapse the whole thesis

If fully correct explanations of negative claims and policy decisions require a second independent authorization engine *because the current Quack/OGAR/dir_sim interface cannot legally express its proof obligation*, **stop** rather than hide a new runtime behind a hotplug. Document the smallest missing primitive, parity target, data/edge witness cost and authority boundary. That is an informative result.

The high-signal hypothesis remains narrow: **the same SoA substrate can host independent typed readings of identity, policy, evidence and software methods**, while the meaning of those readings and the authority to act stay separately governed.

## 12. Post-#1509 implementation drift: #1514 earns *one* relation hop [G, on main]

**Verified after writing this RFC:** [#1514](https://github.com/AdaWorldAPI/lance-graph/pull/1514) is merged to `main` (the #1512 research branch was cut at the earlier #1509 merge and has not been rebased). `crates/lance-graph-glove-parity/tests/relation.rs` and `src/gremlin.rs` now expose **D-CMH-G1-4a**, a *functional* `out(rel)` carrier preserving one anchor row per traverser. A declared `Relation` uses a foreign key; `out(rel)` tests that it references a **live** target with `Filter::Semijoin`; `out(rel).has` can evaluate `EqU32Via`. The tests use synthetic ERP line->partner rows n=1,64,65,4097 with text-based row oracles and documented disable runs.

**Exactly what it does NOT earn:**
- fan-out or many-to-many access membership, `in`/`both`, or target dedup;
- a second hop `out(rel).out(rel)` (explicit `SecondHop` refusal);
- sums/values or referenced methods after a relation when the method glove refuses them;
- R2IL parity for the foreign-target liveness test: R2IL's current fold band lacks `GATHER`, and the method reports `FrontendGap`;
- a complete IAM authorization result, policy evaluation, source identity merge or production session evidence.

**IAM transfer [H]:** a functional `User -> Manager` or `Line -> Account` read could reuse this exact carrier, but `User -> Groups -> Roles -> Entitlements` is multi-valued and recursive. It cannot use `out(rel)` without proving multiplicity and anchor semantics. This **updates** earlier #1509-era text that described *all* relation carriers as absent. The general D-SAG-1-4 and D-CMH-G1-4 obligations are still open.

**IAM-44:** a synthetic functional one-hop matches #1514's row oracle and refuses an R2IL path that needs `GATHER`. **IAM-45:** a user with two groups and cyclic/nested memberships must not be silently forced into a one-foreign-key functional relation; fan-out/second-hop deliberately refuse pending an earned carrier.

## 13. Device policy is not device compliance: D-PBL intersection [P, on main]

The [policy-baseline integration plan](policy-baseline-integration-v1.md) is a **proposal** on current `main`, not an implemented Intune/GPO reader. It distinguishes:

| Layer | GPO | Intune |
|---|---|---|
| Configured | directory policy object + SYSVOL setting | Graph configuration setting |
| Assigned | GPO link, inheritance, security/WMI filter | target group, exclusion and assignment |
| Observed | only via supplied client-side RSoP/status (not centrally inferred) | per-device status returned by Intune |
| Authorization consequence | only if a **separate** supported access-control policy uses the fact | only if conditional-access policy and evaluation evidence authorize the effect |

A unified policy-setting row needs published **GPO↔CSP mapping evidence**; identical-looking names and values are not proof of the same setting. An unmapped setting remains source-specific. `dir_sim` cannot assume it already contains device populations: D-PBL explicitly identifies device ingestion, device-group path support and read-only source adapters as gaps.

**IAM-41:** assignment to a group with an explicit exclusion is not applied to excluded device; missing device/group reader yields `Unknown`, not compliance.

**IAM-42:** `GPO configured=true` plus link is not the same as `observed compliant`; without RSoP/device evidence, observed GPO state remains unknown.

**IAM-43:** unreviewed GPO↔CSP alias/mapping and a missing device object must fail closed on cross-source policy comparison. A device policy conflict may be *observed* without automatically deciding conditional access.

**Dependency:** D-PBL must separately earn policy-object ingestion, source mapping, assignment precedence and actual device status before the IAM projection can consume it. Do not move GPO/Intune into `dir_sim` as if they shared directory authority or write lifecycle. This new seam is relevant to the advertisement's “Device & App Management” category, but it is not proof of an Intune connector.
