# IAM access paths, temporal policy and provenance: primary-source excerpts

**Date:** 2026-10-12. **Status:** sourced research distillation / transfer hypotheses, **not** implementation evidence. **Base:** [merged lance-graph #1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509). **Application plan:** [D-IAM-SAG-0](../.claude/plans/iam-substrate-sag-dir-sim-v1.md).

**Method.** Short **paraphrased** excerpts (Exzerpt / sinngemäß), with section or abstract locator; exact source URLs, what the author establishes, what remains speculative for OGAR/dir_sim/lance-graph, and a concrete falsifier. We do not transfer formal guarantees merely because vocabulary matches. arXiv preprints are not all peer-reviewed; NIST and Frontiers are non-arXiv primary sources included because access semantics and NARS require them. Read existing [Hyperon excerpt](2026-10_hyperon_metagraph_folds_chronomorphisms_primary_excerpts.md) and [NARS evidence excerpt](2026-10_nars_self_epistemic_needs_evidence_primary_excerpts.md) instead of duplicating entire papers.

## P1. Why-provenance is a set of sufficient witnesses, not a path label

**Source:** Calautti, Livshits, Pieris, Schneider (2023), *The Complexity of Why-Provenance for Datalog Queries*, [arXiv:2303.12773](https://arxiv.org/abs/2303.12773); abstract and complexity results in full text.

**Exzerpt, abstract (paraphrase):** Why-provenance explains an answer through subsets of input facts sufficient to derive it. Recursive queries pose harder computational problems than many nonrecursive fragments; the paper establishes intractability for recursive cases it studies, even certain restricted recursive forms. It also explores practical solving approaches.

**Transfer to IAM [H]:** `Identity->Group->Role->Entitlement` needs a witness contract: (a) *one concrete sufficient witness* for a positive role, (b) *all witnesses* if making a disappearance claim after removal, or (c) an explicitly bounded search. In `dir_sim::actors`, `RoleEvidence::Held` already returns one (shortest) SID-chain witness, and its documentation correctly says this is not the only reason. Reuse `RoleImpact::between` to decide actual role loss.

**Falsifier:** grant one entitlement by two edge-disjoint paths, remove one path, and verify the explanation changes while effective access stays. Declare a path enumeration budget and prove an unfinished search cannot be reported as complete absence. The paper **does not** prove a specific Query/Masks implementation or supply an IAM authorization oracle.

## P2. Temporal roles must satisfy policy and timing together

**Source:** Combi, Viganò, Zavatteri (2015), *Security Constraints in Temporal Role-Based Access-Controlled Workflows (Extended Version)*, [arXiv:1512.06404](https://arxiv.org/abs/1512.06404); abstract, temporal role/workflow mappings and security constraint sections.

**Exzerpt, abstract (paraphrase):** The authors integrate role timing and temporal workflow constraints, determine which users can perform tasks, and describe runtime propagation of security constraints.

**Transfer [H]:** a directory snapshot VersionId, a policy revision, a grant's effective interval and an observed-at timestamp play different roles. A replay of `V7` at a historical policy version is an *analytical question*, not permission to execute a change now. The latest external PDP still decides live authorization.

**Falsifier:** membership is true at both t0 and t1, policy grants only [t0,t1); an as-of query and an execution request at t1 must differ. Also test an observation arriving late and policy revision changing mid-simulation. The paper does not endorse a particular two-clock encoding or guarantee eventual-consistency safety.

## P3. Aggregating an access explanation requires explicit provenance semantics

**Source:** Amsterdamer, Deutch, Tannen (2011), *Provenance for Aggregate Queries*, [arXiv:1101.1110](https://arxiv.org/abs/1101.1110); abstract and sections introducing aggregate provenance.

**Exzerpt, abstract (paraphrase):** Provenance for aggregate computations cannot in general be handled by naïvely using ordinary tuple annotations; provenance may need to attach to values involved in a computation, and extensions to positive relational algebra with aggregates have additional difficulties.

**Transfer [H]:** IAM review dashboards count people, grants, risky roles and duplicated licenses. Distinguish bag count from distinct canonical identity count; attach evidence to the source values/derivation and preserve group-role policy version. An `any(path)` witness is not a `count(distinct principals)` proof. The typed `SoftwareGraph` in #1509 already distinguishes bag-versus-set semantics in its falsifiers.

**Falsifier:** same principal reaches one role via two paths, two source aliases and two time windows. Counts of entitlements, paths and distinct principals must differ by design; policy conclusions must be invariant under reordering of duplicate evidence and explicit about the chosen unit of counting.

## P4. Hyperon: reflect on the methods, not a second IAM engine

**Source:** Goertzel et al. (2023), *OpenCog Hyperon: A Framework for AGI at the Human Level and Beyond*, [arXiv:2310.18318](https://arxiv.org/abs/2310.18318), especially metagraph representation/rewriting and self-modification discussion (§3.1.3 and surrounding architecture). Related formal proposal: Goertzel (2021), *Reflective Metagraph Rewriting as a Foundation for an AGI "Language of Thought"*, [arXiv:2112.08272](https://arxiv.org/abs/2112.08272), abstract.

**Exzerpt (paraphrase):** Hyperon proposes metagraphs to represent knowledge and cognitive processes; a reflective language can treat its own programs/rewrite rules as graph-represented subjects of computation. These are architectural proposals and directions, **not** a demonstrated runtime parity result for lance-graph or a security certification.

**Transfer [H]:** #1509's `SoftwareGraph` already projects prepared methods, lowering programs and compiler refusals as typed rows queried by Quack. Model an IAM method's dependency and test evidence in the *software graph* while the actual effective access is computed from the *directory/authority graph*. Link via a versioned method/plan receipt, not a second runtime, and never turn SAG `Tier::Hypothesis` into an entitlement.

**Falsifier:** declare a hypothetical optimization method with no compiled body; SAG must distinguish *declared*, *compiled*, *verified*, *refused*, and *hypothesized*. Reproject after changing the method seal and invalidate stale evidence. A typed graph description does not itself execute `do()`.

## P5. NARS/ONA: introspection is revisable evidence, not provider authority

**Source:** Wang, Li, Hammer (2018), *Self in NARS, an AGI System*, *Frontiers in Robotics and AI* 5:20, [DOI:10.3389/frobt.2018.00020](https://doi.org/10.3389/frobt.2018.00020), §§2, 3.1–3.5. Related implementation: [OpenNARS for Applications](https://github.com/opennars/OpenNARS-for-Applications). Existing deeper paper audit: [NARS excerpt](2026-10_nars_self_epistemic_needs_evidence_primary_excerpts.md).

**Exzerpt, §§2–3 (paraphrase):** NARS operates under insufficient knowledge/resources, builds revisable self-knowledge using internal events and reasoning traces, and treats mental operations as a supplement to normal inference control. The article's implementation discussion is not a demonstration of subjective awareness.

**Transfer [H]:** an optional IAM observer consumes `Undetermined`, `RefusedFor`, missing evidence and historical mismatches to rank *read-only probes*: inspect another source, run a `dir_sim` counterfactual, request review. Keep statistical surprisal, confidence in an inferred claim, and enterprise authorization in separate planes. Self-observation of a change proposal is not an approval event.

**Falsifier:** same evidence under observer-on and observer-off; compare discovery coverage, false alerts, CPU time and effect on factual IAM output. Observer state may prioritize *questions*, but must not alter the ground-truth role-evidence oracle or bypass an effect gate.

## P6. Dependent evidence cannot be added as independent evidence

**Source:** Jøsang (2007), *Cumulative and Averaging Fission of Beliefs*, [arXiv:0712.1182](https://arxiv.org/abs/0712.1182), abstract and fusion/fission sections. Existing detailed distillation: [NARS evidence excerpt](2026-10_nars_self_epistemic_needs_evidence_primary_excerpts.md).

**Exzerpt (paraphrase):** Different fusion laws apply according to whether evidence has independent or shared provenance; fission examines the effect of removing a known contribution under explicit algebraic preconditions. These are Subjective Logic rules, **not** NARS or CE64 byte-laws.

**Transfer [H]:** a principal's license status observed through AD sync and Entra sync may originate from one upstream change. Do not count the two read surfaces as independently corroborating the same change. Retraction/ablation of one upstream fact should account for downstream duplicates. The immutable source identity and replayable observation receipt are the meaningful inputs.

**Falsifier:** one observation reported by two adapters vs two truly independent observations; adding a duplicate must not increase evidence confidence. Attempting exact inverse confidence from only saturated u8 values must refuse or report bounds, not fabricate certainty.

## P7. ABAC is an authorization evaluation, not a graph association

**Source:** Hu et al., NIST SP 800-162 (updated 2019), *Guide to Attribute Based Access Control (ABAC) Definition and Considerations*, [DOI:10.6028/NIST.SP.800-162](https://doi.org/10.6028/NIST.SP.800-162); [NIST publication](https://csrc.nist.gov/pubs/sp/800/162/upd2/final), abstract and §2.

**Exzerpt, abstract (paraphrase):** ABAC evaluates subject, resource/object, action and possibly environment attributes against policy rules or relationships. This is **more** than the existence of a directory edge. Attributes require validity, provenance and decision context.

**Transfer [H]:** `member_of` may imply a policy candidate but cannot by itself decide conditional access; unknown device compliance and unverified session posture block a fully determined decision. Use ClassView/WideFieldMask for read fields, not as a permission mask. Refuse missing policy context, and keep the IdP/PDP as policy authority.

**Falsifier:** same identity/group path on one compliant and one unknown-compliance device. Graph entitlement can be equal; policy decision becomes permitted/undetermined under a specified policy. Cannot equate a graph association with a successful SSO session.

## P8. Zero Trust keeps enforcement separate from analytical proposals

**Source:** Rose et al. (2020), NIST SP 800-207, *Zero Trust Architecture*, [NIST PDF](https://doi.org/10.6028/NIST.SP.800-207), §§2–3 and policy decision/enforcement point description.

**Exzerpt, §§2–3 (paraphrase):** Access to enterprise resources is mediated by a policy decision point (PDP) and policy enforcement point (PEP), using subject/request context and enterprise policy; network or directory membership alone does not establish trust for all future interactions.

**Transfer [H]:** Quack may evaluate read-only graphs, OGAR/dir_sim may simulate, and SAG may propose tests. The live SSO/PAM/MDM or directory provider still enforces; an external mutation has a separate actor-bound authorization and precondition check. Inference confidence is not an execute grant.

**Falsifier:** inject a `Tier::Verified` optimization receipt, forge a policy-looking graph edge or replay a historical approval. All three must fail to produce a provider write without fresh external authorization. A policy simulation cannot certify that a third-party enforcement system will behave identically.

## Source and transfer ledger

| Work | Authoritative for | Not evidence for |
|---|---|---|
| arXiv:2303.12773 | complexity/meaning of why-provenance witnesses | automatic complete IAM explanations |
| arXiv:1512.06404 | role/workflow timing constraint models | chosen directory timestamp layout |
| arXiv:1101.1110 | aggregate provenance pitfalls | a particular Quack aggregate implementation |
| arXiv:2310.18318 / 2112.08272 | reflective metagraph language/architecture | new engine necessity, measured AGI |
| Wang et al. 2018 / ONA | internal-event traces and revisability | IAM authorization, phenomenal selfhood |
| arXiv:0712.1182 | evidence fusion/fission assumptions | byte-compatible NARS/CE64 truth rules |
| NIST 800-162 | attribute-policy model | proof an entitlement is allowed |
| NIST 800-207 | PDP/PEP and zero-trust architecture | replacement for an active IdP |

**High-signal conclusion:** the core research-backed contract is a triangle: **sufficient provenance witnesses × version-pinned temporal policy × separately enforced authority**. Graph introspection helps discover *questions* and *counterfactuals*, not a shortcut through either formal witness completeness or an authorization boundary.

**Open questions for the implementation PR:** (1) exact carrier for multi-hop; (2) how to expose and bound witness completeness; (3) version join for dir_sim + GroupRolePolicy + MDM; (4) where OGAR/OGIT identity mapping authority lives; (5) what counts as a verified compiler certificate vs a claimed capability; (6) which provider operations remain only simulated.
