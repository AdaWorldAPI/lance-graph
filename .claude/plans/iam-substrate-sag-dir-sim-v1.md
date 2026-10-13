# D-IAM-SAG-0 — IAM substrate: dir_sim × OGAR × Class/Methods × Software Awareness Graph

> **Aligned by [`sag-capstone-alignment-v1.md`](sag-capstone-alignment-v1.md)** — read it first: canonical `ProposedMethod`, the shared status vocabulary, D-id map, and the claims it marks stale (incl. "second hop refused").

> **RFC / implementation handoff, documentation only.** Based on merged [#1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509), merge commit `42fa2a347cda22eaaa327bba514f2029cc9daa82`. The goal is to specify executable, falsifiable IAM projections, not announce six shipped IAM products. No connector writes, security-authority migration, new opcode, second graph engine, or CE64 byte-semantic change is authorized.
>
> **Companions:** [Software Awareness Graph](software-awareness-graph-v1.md), [Class × Schema × Methods HotPlug](class-method-schema-hotplug-v1.md), [class/method falsifiers](class-method-hotplug-falsifiers-v1.md), [dir-sim persistence](dir-sim-persist-v1.md), [RBAC nested scope](../board/entries/2026-10-08-rbac-nested-scope-path.md), [research excerpts](../../.arxiv/2026-10_iam_access_paths_temporal_provenance_primary_excerpts.md), [implementation prompt](../prompts/iam-substrate-after-1509.md). OGAR semantic sibling [#357](https://github.com/AdaWorldAPI/OGAR/pull/357) is an RFC, not a shipped dynamic-method link.

## 0. Question and answer boundary

Could the six functional domains in a typical IAM stack be *typed views* over one provenance-aware identity/entitlement substrate, using existing OGAR/dir_sim/Quack/SoftwareGraph infrastructure rather than reproducing SailPoint, CyberArk, Okta or Intune?

**Candidate answer:** yes for read-only explanation, simulation and proposal *when the relevant graph edges and source-authority semantics are available*. No for live enforcement, SSO, session control, device management and privileged vault features without independent real provider APIs and permission gates. A syntactic class method is not an authorizing identity provider.

Three distinct questions must survive every projection:
1. **Observed:** what did a version-pinned source actually return? Preserve source, source object identity, effective time, observation time, schema version and completeness.
2. **Inferred:** what follows from membership, role, device, resource and policy relationships? Preserve complete witness sets or explicitly label a *single* witness, assumptions, rule version and uncertainty.
3. **Authorized:** which action may a particular actor request or execute *now*, against which authority and version? Re-evaluate outside the analytical graph; mere high confidence never confers permission.

## 1. First-hand inventory and exclusions (as of #1509)

| Surface | Code-grounded behavior [G] | Remaining boundary [P]/[ABSENT] |
|---|---|---|
| `crates/lance-graph-dir-sim/src/lib.rs` | Separate users/groups SoA populations, GUID identities, Dn128 location, sparse membership, pure rules, validation, readonly LDAP bytes | Not an AD/Entra/PowerShell writer; no live authentication/authorization product |
| `dir-sim/src/store.rs`, `view.rs` | `VersionStore` observe/simulate/diff/plan; shared snapshot + delta-sized overlay; versioned provenance and refusal | Cross-policy/source coherent IAM snapshot and schema bridge must be designed |
| `dir-sim/src/actors.rs` | `GroupRoles::from_policy`, `role_evidence` and `RoleImpact::between`; SID/security-group paths; `Held/NotHeld/Undetermined/NotDeclared` | Role evidence is version-specific and role-scope binding is consumer-owned; don't reinterpret a distribution group as a security group |
| `dir-sim/src/log.rs`, `image.rs`, `lance-graph-dir-sim-lance` | Persist/replay ingredients with logical VersionId distinct from Lance batch versions; torn-save refusal contract | Availability in production does not imply multi-system transaction or tenant snapshot equivalence |
| `crates/lance-graph-contract/src/hotplug.rs` and `class_view.rs` | Per-plug authority activation, checked reading, ordered field basis, WideFieldMask | WideFieldMask is field selection, not row-RBAC or access verdict; slab physical reading is distinct from semantic view |
| `crates/lance-graph-glove-parity/src/awareness.rs` (#1509) — ⊘ removed by #1524 (D-CMR-RESET), code on `backup/session-011BMJ-2026-10-12` | `SoftwareGraph::project` from prepared MethodTable; method/field/program/capability/refusal edges with tier and provenance; Quack Binder | In-memory one-hop projection, caller-supplied revision, limited digest; neither a directory membership store nor persistent general graph engine |
| `crates/lance-graph-quack/src/bind.rs`, `crates/lance-graph-r2il` | `Draft::bind`, Quack lowering into mask-risc; R2IL FoldDialect execution over borrowed planes | Multi-hop relation composition not automatically solved; cannot carry result masks as an undeclared population between programs |
| `crates/lance-graph-rbac`, OGAR `GroupRolePolicy` | Separate policy authority and scope/grant primitives | Not an entitlement-certification UI or PDP/PEP for arbitrary external systems |
| OGAR MethodDecl / `ogar-loco` Inventory / class-method RFC | Source method metadata and keyed body inventory exist; binding composer proposed | `MethodDecl::body_source` is not an executable method; no implicit DO capability |

[#1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509) explicitly leaves **D-SAG-1-4 multi-hop**, **D-SAG-1-6 persistence**, and method identity canonicalization open. Preserve its no-new-opcode/no-second-engine/no-applied-transformation rules. Its H1 SIMD fusion has *not* demonstrated a speedup over the scalar reference. Reuse its `Tier`, `Provenance` and compiler refusals as projected evidence, **not** as a new authority.

## 2. Map the six IAM functions to typed projections

| IAM function | Read-only projection / canonical question | Adapter + side-effect boundary | Honest status |
|---|---|---|---|
| IGA (SailPoint, Saviynt) | Identity lifecycle, role/entitlement history, access certification, SoD finding, `why_effective_access?` | HR and entitlement catalogs, review decisions, independent provisioning executor | Core identities/simulation exist, review workflow proposed |
| PAM (CyberArk, BeyondTrust) | `has_privileged_capability?`, `through_which_grant?`, session and approval evidence | Vault/JIT/session products retain privileged access control, recording and rotation | Projection proposed; not a PAM replacement |
| Access/SSO (Okta, Ping) | Federation principal, application assignment, role mapping, conditional access evidence | IdP does authentication, token issuance, MFA and session policy | Read graph proposed; **no SSO** implemented |
| Directory & identity (AD, Entra) | Principal, group, immutable ID, DN location, synced anchor, nested path, what-if membership | Read observations; eventual writes remain external and separately authorized | dir_sim active; cloud correlation depends on available adapter fields |
| Device/app (Intune) | Device ownership, compliant/unknown/noncompliant, app & CA dependencies | MDM supplies compliance, enrollment and enforcement | New domain schema + adapter required |
| Automation (PowerShell, APIs) | Auditable `propose/rehearse/approve/execute/verify/revert` | Capabilities checked at immutable context and just before any external write | Quack/R2IL + plans exist; guarded provider DO seam proposed |

The marketing categories overlap; none establishes an ontology or guarantees a source can provide every required edge.

## 3. Minimal IAM graph vocabulary, not a second graph engine

**Node classes** (concept roles, not a mandate to mint numeric ClassIds): `Identity`, `Group`, `Role`, `Entitlement`, `Resource`, `Application`, `Device`, `Policy`, `Grant`, `Session`, `Evidence`, `Change`. Optional source-specific subtypes (human/service identity, privileged role, tenant, cloud object) are *declared schema refinements* only.

**Edges** (direction is part of semantic contract):
- `Identity MEMBER_OF Group`, `Group MEMBER_OF Group` (source-specific group type/security semantics, nesting scope, witness);
- `Group GRANTED_ROLE Role`, `Role GRANTS Entitlement`, `Entitlement ALLOWS_ACTION Resource`;
- `Identity ASSIGNED_APP Application`, `Identity USES Device`, `Policy REQUIRES_CONDITION Device/Session`;
- `Grant AUTHORIZED_BY Evidence/Policy`, `Session EVIDENCED_BY Evidence`, `Change DERIVED_FROM Evidence`;
- `ExternalObject CORRESPONDS_TO Identity` only under a reviewed match rule and tenant/sync-anchor evidence.

**Identity:** source AD object GUID, Entra objectId and external provider UUIDs are *source keys* with their own tenant and namespace. DN is a **mutable location** (not the primary key). UPN, mail, aliases and display names are **attributes**, never joins by default. Distinguish OGAR's canonical NodeGuid and per-view ordinals from external GUIDs; map with explicit provenance, don't cast/repack equal-sized values or silently unify tenants. `ClassId` high-u16 concept and full u32 addressed class are different views per D-CMH-0.

**Edges are claims:** identity, source/tenant, (valid_from, valid_to), observed_at, source revision, rule revision, completeness/knownness, provenance and verification tier. `Unknown` is not `false`; `observed` is not `authorized`; a hypothetical delta is not an observation.

**Layered canonicalization:** OGIT canonical field URI and reviewed native alias conversion -> OGAR class/reading authority -> ClassView ordered basis and WideFieldMask for field selection -> Quack numeric binding. Do not equate SAP/Odoo/AD fields by label, and don't move schema semantics into the 8-bit ALU/ISA register contract. CE64, CARE, ONA belief and epistemic tiers can remain contextual readings/evidence policies, not hard-coded identity or action bits.

## 4. Two graphs, one substrate, no semantic conflation

**Business/access graph:** version-pinned directory and authority relationships, entitlements, policy provenance and source observations. `dir_sim` governs directory what-if state; versioned GroupRolePolicy remains policy authority. Real IdP/PAM/MDM behavior is modeled as imported evidence, not delegated to `dir_sim`.

**Software-awareness graph (SAG):** methods, inputs, lowering programs, compiler refusals, capability *claims* and tests from #1509. A proposed IAM `effective_access` method should have a `USES_FIELD`/program dependency projection and can report if it is compiled or refused. A SAG assertion `MayRequire` never supplies an observed entitlement.

The bridge is a **versioned method binding/receipt**:
`(source snapshot ids, policy version, canonical schema seal, prepared method key, body/plan digest, actor/read scope) -> query result + witness + tier + refusal`.
Use the D-CMH composer, not another metadata registry. Do not mint method identity in this PR.

**Multi-hop gate:** #1509's one-hop method queries do not implement `Identity -> Group -> Role -> Entitlement`. Compare two legal candidates in D-SAG-1-4/D-CMH-G1-4: (a) typed relation carrier connecting a population output to the next input; (b) factored `EqU32Via` read. Require compiler/Quack/mask-risc parity with a pinned slow reference on cyclic, nested, multiple-path and deleted-edge fixtures. Until one passes, use existing directory membership path logic for directory subgraph and mark broader entitlement closure as proposed. NEVER pipe an arbitrary output Mask as the next program's `LIVE` population.

## 5. Worked scenario: Anna's accounting entitlement

Illustrative data, never real personal data:
```text
Principal AD-guid(A), Tenant(T) --MEMBER_OF/security--> GRP_Finance
GRP_Finance --MEMBER_OF/security--> APP_ERP_Accounting
APP_ERP_Accounting --GRANTED_ROLE/policy-v7--> ERP_Write
ERP_Write --GRANTS--> Accounting_Write on ERP_Ledger
Device(D) --COMPLIANCE@mdm-v12--> Unknown
```

Requests:
1. `explain_access(A, ERP_Ledger, Write, as_of=V, policy=p7)`: enumerate **all** supporting paths or declare `one witness / nonexhaustive`; include policy, source and snapshot receipt.
2. `revoke_preview(A, GRP_Finance)`: `VersionStore.simulate` on base V, evaluate effective roles for both views with `RoleImpact::between`, compute target entitlement deltas; ensure another valid path preserves access.
3. `why_not_access?`: if the device/CA policy is missing, report `Undetermined/Unknown`, *not* a denial or approval.
4. `propose_action`: stage an inert `ProposedMethod` with preconditions and impact; policy authority separately gates `approve` and eventual external executor.

Distinguish **effective graph grant**, **authentication success**, **conditional access evaluation**, and **live enforcement**. A directory membership cannot prove an IdP token was accepted.

## 6. Method sketch (illustrative, intentionally not a Rust API claim)

```text
class Identity      { memberships, effective_access, access_history,
                      role_evidence, simulate_removal, reconcile }
class Entitlement   { explain, principals, blast_radius, revoke_preview }
class Device        { compliance, linked_identities, access_dependencies }
class Grant         { policy, witnesses, effective_during }
class Change        { preview, validate, approve, execute, verify, revert }
```



> ⊘ 2026-10-13: this type is no longer on `main`. `glove-parity::method::EffectClass` was removed by #1524 (D-CMR-RESET) and is on `backup/session-011BMJ-2026-10-12`; the four names stand as vocabulary until the D-CMR rebuild chooses an anchor. (Not the `EffectClass` still on `main` at `crates/lance-graph-cognitive/src/search/certificate.rs`: that one classifies Cohen's d effect sizes and is unrelated.)

**Execution class:** the names `Pure`, `ReadVersion`, `Propose` and `ExternalMutate` come from the withdrawn pre-reset `glove-parity::method::EffectClass` (see the note above); keep them as vocabulary until the D-CMR rebuild chooses an anchor. Nothing below is available on `main`. As intended: `Pure` uses existing Quack/R2IL and dir_sim primitives. `Propose` builds a `ProposedMethod` and is legal only on an Action body. `ExternalMutate` is not a Quack opcode and must refuse unless independently authorized, with fresh source preconditions and external connector attestation. An absent field, alias, authority, schema seal, body, read scope, source version, evidence completeness or capability causes an explicit typed refusal.

## 7. Falsifiers and acceptance matrix

| ID | Fixture / negative test | Expected oracle |
|---|---|---|
| IAM-01 | Same UPN across tenants, renamed DN | two distinct source keys; stable GUID identity; no accidental join |
| IAM-02 | user -> security group -> nested security group -> role | same effective role and witness as dir_sim `role_evidence` |
| IAM-03 | distribution group in nested chain | no transitive security role gained |
| IAM-04 | two independent routes to the same entitlement; remove one | access remains; lost witness != lost entitlement |
| IAM-05 | unknown group policy (never observed) vs explicitly empty policy | `Undetermined` vs `NotHeld`; cannot certify absence from unknown |
| IAM-06 | stale source observation, policy revision changes mid-query | pinned result or typed `VersionMismatch`; never silently mix |
| IAM-07 | one simulated membership removal + replay | original unchanged; deterministic diff/plan; replay matches; no external IO |
| IAM-08 | revoked and newly minted grant; two time windows | access only inside authorized interval; no time-travel leakage |
| IAM-09 | schema/reading drift or missing required field/validity plane | explicit refusal; no default known/allowed |
| IAM-10 | SAG declares `MayRequire FirstOfRun` but no compiled program | hypothesized/claimed, not shipped; no fabricated authorization |
| IAM-11 | duplicate log/evidence derivations | provenance distinguishes source vs inference; no double confidence count |
| IAM-12 | actor without execute permission; forged approval or policy confidence | all external action attempts refuse, no provider writes |
| IAM-13 | unknown device compliance under conditional-access dependency | indeterminate, not permitted/denied based on guess |
| IAM-14 | cost benchmark of multi-hop vs pinned reference | correctness first, memory/latency measured by dataset and CPU tier; no claimed speedup without data |
| IAM-15 | disable key checks separately (source scope, knownness, external effect gate, witness alternatives, snapshot pinning) | each deliberate mutation makes a named test red |

Represent each test with source data, sealed schema/policy revisions, `actual`, `expected`, refusal/witness and oracle provenance. Record disable tests rather than merely asserting the guard exists.

**Two high-value regression models:** (a) duplicate license assignment from competing sources and (b) legitimate user losing a license during reconciliation. Define synthetic inputs and precedence rules, run proposed changes through `dir_sim`; neither incident should be claimed fixed until the relevant semantics and parity tests exist.

## 8. Execution slices and stop signs

- **W0 Inventory, no Rust changes:** verify #1509 source/tests, OGAR semantic ownership, dir_sim latest exact APIs, policy version model, Quack relation carrier. Label each contract `[G]`, `[P]`, `[ABSENT]`. Review whether analogous methods already exist before adding one.
- **W1 Typed read-only IAM fixture:** reuse version-pinned `VersionStore`, `GroupRolePolicy`, existing membership/role evidence. Produce explainable `Held/NotHeld/Undetermined` rows with provenance, including one-witness vs all-paths semantics. Pin IAM-01..05.
- **W2 Canonical class/method binding:** compose in existing allowed consumer layer under D-CMH-0, OGIT field identity, ClassView field basis, schema seal, per-plug authority and existing Quack Binder; project methods/actual compiler verdicts into SAG. Pin IAM-09/10 and policy-source separation.
- **W3 Multihop/read-only access impact:** select a legal relation carrier or existing `EqU32Via` plan after oracle comparison; don't introduce a traversal runtime by stealth. Implement two-version before/after and alternate-path cases, pin IAM-04/06..08/13.
- **W4 Optional ONA-style introspective proposer:** SAG can inspect `refusal -> gap -> test -> candidate`; choose what to investigate using declared budget/uncertainty but *never* let an uncertain signal mutate provider state. Keep epistemic confidence, causal hypothesis, and authority separate. Pin IAM-11/12 and ablate the prioritizer against no-observer baseline.
- **W5 Authorized execution (separate PR only):** source authority and RBAC signing, actor-bound scope, reviewed change, idempotency, check-and-set, provider-specific receipts, conflict/refusal and revert feasibility. NO execution in this RFC.

### W4 observation/control loop (optional)

```text
observe actual grant/policy evidence -> explain via existing graph
 -> identify unverified/undetermined/violating claim (not an enforcement verdict)
 -> produce candidate explanation/test/dir_sim delta
 -> run read-only counterfactual and dependency ablation
 -> label Verified / Refuted / Undetermined with source receipts
 -> present proposal for human/provider authority, no automatic DO
```

Do not equate Shannon entropy with epistemic uncertainty, policy risk or access authorization; report them as distinct measures. The Rubicon/Heckhausen, ONA, Hyperon metagraph and CARE analogies are **research hypotheses** for investigation priority, not license to repurpose CE64 bits 59..63 or identify a policy action with a belief.

## 9. Research decisions and traceability

See [primary-source excerpts](../../.arxiv/2026-10_iam_access_paths_temporal_provenance_primary_excerpts.md) for paraphrased page/section references, direct publication links, transfer limits and falsifiers. Key non-claims:

- NIST SP 800-162 ABAC and SP 800-207 Zero Trust constrain access evaluation and enforcement; neither validates an IAM product built from a graph query.
- Why-provenance can be computationally hard for recursive queries; do not conflate *one witness* with a complete minimal explanation.
- Temporal RBAC constraints require who/what/when; an immutable snapshot alone does not define real-time authorization.
- Hyperon and NARS are architectural references for self-representation and revisable observation, not security accreditation, a replacement engine, or evidence of subjective cognition.

## 10. Deliverables for the implementation successor PR

- A code-grounded contract matrix and accepted naming/authority for canonical IAM concepts.
- A small sealed synthetic dataset and a strict reference evaluator for graph access/role impact.
- Read-only method projections whose compiler proofs/refusals appear in SAG, not hand-authored 'Verified' tiers.
- IAM-01..15 test matrix with disable/falsification evidence; one passing end-to-end `explain -> simulate -> re-evaluate -> propose` demo.
- Measurement table (baseline, candidate, environment, versions) before any performance claim.
- Security review showing external provider writes impossible in read-only mode; no unreviewed policy/byte-layout/ISA drift.

## 11. Frontier expansion after the first draft: deliberately **not** a shipped capability

**Follow-on RFC:** [D-IAM-SAG-1: negative proof, temporal authority, safe introspection](iam-substrate-frontier-gaps-v1.md) and its [source-distilled excerpts](../../.arxiv/2026-10_iam_negative_provenance_authority_and_agentic_policy_excerpts.md). The [Claude Code handoff](../prompts/iam-substrate-after-1509.md) now carries the additional research and falsification obligations.

The first 15 tests primarily cover *positive* entitlement evidence and directory what-if. The extension adds IAM-16..40, grouped by proof obligation:

- **Proof of absence:** complete source/policy coverage, failed derivations, witness exhaustiveness and alternate paths. A shortest path is not a sufficient revocation proof.
- **Proof of identity and time:** verified cross-tenant anchors, source GUID distinct from DN, source/policy version vectors, observation-vs-valid-vs-record-vs-execution times.
- **Proof of policy:** versioned decision-combining algorithm, explicit Permit/Deny/NotApplicable/Indeterminate, SoD/JIT/approval expiry, bounded negative explanation.
- **Proof of action:** actor-scoped independent authorization, fresh provider preconditions, idempotency/unknown-outcome receipts, provider-specific compensation, no automatic rollback guarantee.
- **Proof of introspection:** SAG compiled vs verified tiers, actor row scope independently checked from WideFieldMask, ONA/Hyperon observer evaluated by no-observer ablation with no authority transfer.
- **Proof of responsible explanation:** tenant-scoped/redacted witness output, machine identities, duplicate-source lineage, adversarial prompt injection and cyclic/high-path-count workloads.

**Release order:** maintain W0/W1 as the smallest read-only implementation. Research the expanded test fixture now; implement only when its domain's existing semantics and compiled oracle can be verified. General multi-hop, policy evaluator and provider DO remain independent later work. These are *candidate requirements*, not claims of new Rust code or completed review.

### Implementation drift observed after #1509

Current `main` has merged [#1514](https://github.com/AdaWorldAPI/lance-graph/pull/1514): **one functional foreign-key `out(rel)`** with `Semijoin` liveness and `EqU32Via` in the glove parity method layer. This is real progress beyond #1509, but IAM's many-to-many group membership, second hop and R2IL `GATHER` parity remain unsolved. The updated [frontier addendum](iam-substrate-frontier-gaps-v1.md#12-post-1509-implementation-drift-1514-earns-one-relation-hop-g-on-main) identifies the precise boundary. It also connects the new [D-PBL GPO/Intune policy baseline proposal](policy-baseline-integration-v1.md) to IAM without conflating configured/assigned/observed state or claiming the Intune connector exists. With IAM-41..45, the full proposed test catalog is **IAM-01..45**, not 40; none is executed by this documentation PR. The #1512 branch remains anchored at #1509 and is behind main pending a later synchronization.
