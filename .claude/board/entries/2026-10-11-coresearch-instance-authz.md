# Coresearch — instance authorization and principal expansion (2026-10-11)

**Status:** OPEN — exploration map. It ratifies nothing; the operator chooses.
Council: `.claude/agents/coresearch-council.md`.

## The question, after the premise audit

The first brief asked for "one authorization shape". The premise audit found
it bundled three different mechanisms, so the question was re-asked as two:

- **Q1 — instance authorization.** How do we decide whether a set of
  principals holds a right on one object? This covers V17 (mailbox
  FullAccess / SendAs / SendOnBehalf) and V18 (SharePoint role assignments,
  broken inheritance, expiring links).
- **Q2 — principal expansion as a plane.** How is a user's principal set
  (the SID chain and the mail chain) built once per directory version as a
  resident plane, rather than walked per call? Where does the group→role
  map for #36 live?

Moved out as not authorization: the RFC 8693 `act` claim is request
provenance (ogar-auth, #22); the GUID join is identity resolution (#28).

Anchors are frozen: the group model (V19), keystone I-K5, I-K7 and I-K8,
the V17 and V18 decisions, read-only simulation, hot-plug only, the
DataFusion grace period, and the firewall rules.

## The shape the rows converge on (three pieces)

1. **`PrincipalPlanes`**, one per `(VersionId, chain)`. Built at install,
   outside any program, and resident so that Quack can gather from it.
   - Rows are groups; columns are only the groups that are trustees.
   - A direct/indirect bit lets a witness explain itself.
   - There is one plane per chain, never a union of chains.
   - Because the columns are trustee groups only, the plane also depends on
     the grant set. The grant and directory versions must therefore be
     pinned together.
   - Home: `lance-graph-dir-sim/src/principals.rs`.
2. **`GrantRelation`**, a lane beside membership in `Snapshot`.
   - Fields: object, trustee, rights bits, deny plane, level, witness code,
     and expiry.
   - Endpoints that do not resolve are kept by identity.
   - A sign-aware contested fold: a contested allow empties; a deny that any
     witness asserts survives.
   - V17 and V18 share the lane; only the object population differs.
   - Gap: a V18 sharing-link trustee is a link, not a principal, and has no
     row in any principal plane. How a link bearer resolves (identity
     resolution, A12) must be named before V18 uses the lane.
   - Home: `lance-graph-dir-sim/src/grants.rs`; the meaning lives in OGAR.
3. **One per-bit first-applicable fold** in zero-dep contract code.
   - The fold, for each level L in order:
     `undecided = requested; refuse if deny_L & undecided; undecided &= ~allow_L`.
     After the last level, `granted = requested & ~undecided`; any bit still
     undecided is **denied by default**. A check for `requested` succeeds
     only when `undecided == 0`.
   - V18 runs the same fold with no deny planes, with implied role bits
     (read < write < owner).
   - Its output is an instance-scope predicate on the **row-scope axis**
     (I-K5, `CLASSID-RBAC-KEYSTONE-SPEC.md:67`: deny and instance scope both
     go to the scope axis). In code it rides as `Membership.scope` into
     `authorize_memberships` (`lance-graph-rbac/src/authorize.rs:267`), but
     it is not a membership in I-K6's sense. It never goes through
     `ScopeSpec.deny`. So the absorbing-deny and static-role gaps need no
     contract change.

**Inheritance becomes a coordinate at install.** V17 writes the level; V18
writes an anchor-ordinal lane. Neither is a request-time walk.

## Verdicts

The `F` column is the firewall verdict: PASS, or C (CONFLICT, which means
the idea fits only through a named seam).

| row | idea | F | verdict | probe / reason |
|---|---|---|---|---|
| X1 | ordered per-level allow/deny fold, default deny | PASS | ADOPT-NOW | fixtures X1 (explicit allow vs inherited deny, same-level conflict, distance order, a right in no plane is refused) + X15 |
| X2 | right families as data (combiner per family) | PASS | ADOPT-NOW | X2: three families, one fold, no Combiner enum |
| X3 | unreported deny level → explicit, known plane | PASS | ADOPT-NOW | X3: unknown deny never grants |
| X4 | one fold serves the check and the listing | PASS | ADOPT-NOW | X4: check == listing under permuted grant order |
| X5 | `GrantRelation` lane, all witnesses kept | C | PROBE | X5: 3 witnesses stored; contested allow empties, any deny survives |
| X6 | bulk listing in Quack | C | PROBE | X6: K ∈ {1,8,64,500} trustee slots × 1,000 mailboxes; (a) install-time planes vs (b) one-program K-slot gather |
| X7 | resource-parent as a 4th relation; scope I-K2 to the class lattice | C | PROBE | X7: broken inheritance stops at the cut; doc amendment compiled at install |
| X8 | V18 anchor-ordinal lane; Dn128 only for selection | C | PROBE | X8: nearest anchor equals the reference walk; the 257th child fails closed and is counted |
| X9 | Graph `0001-01-01` expiry = no expiry, mapped at ingest | PASS | ADOPT-NOW | X9: sentinel and an absent field both mean "no expiry" |
| X10 | new evaluator replayed beside Spear's owner-only one | C | PROBE | offline replay only, never in the request path; 0 disagreements on owner-only grants, ≥1 after adding FullAccess |
| X11 | class grant ∧ instance scope ∧ narrow | PASS | ALREADY-HAVE | an AND mask after the fold; `narrow` is the principle, not the carrier |
| X12 | reverse audit ("who holds FullAccess on M") | PASS | PARK | waits for Y1/Y2 |
| X13 | Stalwart conflates mail reach, ACL and send-as | C | PROBE + OPERATOR | fork-only test; fix shape (a) or (b) is an operator choice, below |
| X14 | V17 trustee rules vs Microsoft cmdlet docs | C | OPERATOR | read the two unread pages; until a ruling, drop an inadmissible allow and keep an inadmissible deny |
| X15 | exhaustive small-universe equivalence | PASS | ADOPT-NOW | 10,000 seeds; the reference evaluator is written independently; a one-bit mutation must be caught |
| X16 | ODRL "conflict voids the policy" | C | SKIP | fail-open trap; keep only V17's report-only diagnostic |
| Y1 | principal plane per VersionId (resident) | PASS | ADOPT-NOW | Y1: plane == `security_identifiers`; a program consuming another program's `Kept` is refused |
| Y2 | Leopard shape (groups×groups closure + direct-groups lane) | C | PROBE | Y2: 10k users, 2k groups, depth 6; the overlay merge is rejected |
| Y3 | columns = trustee groups only | PASS | PROBE | Y3: decisions equal the full plane; source is secondhand |
| Y4 | intern distinct direct-group sets | PASS | PROBE | Y4 census; drop the idea if distinct/users ≥ 0.5 |
| Y5 | incremental closure maintenance | PASS | PARK | full rebuild per version is the default |
| Y6 | one plane per chain + a direct/indirect bit | PASS | ADOPT-NOW | Y6: the chains never mix (this is the X13 cure) |
| Y7 | tokenGroups fidelity: primaryGroupID | C | PROBE + OPERATOR | Y7: as a SID-chain-only edge, never a plain membership row; sIDHistory → ELSEWHERE (A12) |
| Y8 | explicit RoleAssignment record for the group→role map | PASS | ADOPT-NOW | Y8: an undeclared role never leaks; the role is never derived from a group name |
| Y9 | two stages (principals, then grants); liberal grant | C | PROBE | seam: stage 1 is only the compiled per-version plane (Y1); RPPM's path-condition evaluator is a runtime walker and is not adopted. Probe Y9: a {read, write} request with read granted to U and write to G is granted (liberal grant) |
| Y10 | subject nesting carries no priority | PASS | ADOPT-NOW | Y10: swapping nesting depths does not change the decision |
| Z1 | `act` / `may_act` | PASS | ELSEWHERE | ogar-auth boundary, #22; `may_act` is an AND gate |
| Z2 | compound principals as trustees | — | SKIP | provenance would decide grants (A11) |
| Z3 | OAuth RAR | C | ELSEWHERE | parsed once into a `narrow` cap |
| Z4 | unredeemed invitation | PASS | ELSEWHERE | identity resolution (A12) |
| Z5 | role mining | PASS | SKIP | design time only; its one useful output is Y4 |

## Open points for the operator

- **X13 fix shape.** The finding has three concepts: mail reach, ACL
  membership, and send-as. Send-as is an instance grant (A5) and belongs in
  `GrantRelation` (SendAs), whichever option is chosen.
  - (a) A fork-local list recipient that expands members, so a distribution
    list is no longer a Stalwart group account.
  - (b) Keep the mapping, but feed each purpose its own chain and never
    merge group addresses into a member's addresses. This is only sound if
    Stalwart gets two carriers. If both chains still land in the one
    `groups` field (`dirsim/mod.rs:180-209`), the conflation survives.
  - Either way, #20 (reachable from config) stays blocked until it lands.
- **X14 V17 trustee rules.** This splits into two questions.
  - (i) Which principals may be NAMED as trustee, per right and platform?
    The Microsoft cmdlet docs contradict V17 on three points: FullAccess
    with non-mail security groups; SendAs in Exchange Online;
    GrantSendOnBehalfTo needing a mail-enabled principal.
  - (ii) How does membership pass a right down? The ruled "SendOnBehalf
    follows the mail chain" answers (ii), and the docs may contradict only
    (i).
- **Y7.** Should primaryGroupID enter? Without it, a grant to Domain Users
  reaches no one. But it is a SID-chain-only fact: Exchange expands lists
  from `member`. A plain membership row is walked by both chains
  (`view.rs:643-668`), so it must be a SID-chain-only edge or its own lane.
  Mapping the RID to a group is identity resolution (A12). sIDHistory adds
  alias SIDs for one principal, so it is identity resolution and moves to
  ELSEWHERE.

## What was not searched

- Google Drive's permission model was read only through Graph.
- Cedar was not read beyond its semantics (no engine code).
- OPA/Rego was not read.
- Postgres RLS was not re-read; the keystone pass already covered it.
- No SMT-based policy analysis.
- The two Microsoft pages in X14 ("Manage permissions for recipients" and
  `Set-DistributionGroup -GrantSendOnBehalfTo`).

"Nothing found" in these areas means they were not searched, not that
nothing exists.

Premise audit on the options (2026-10-11): PrincipalPlanes and
GrantRelation are SOUND. The fold was SPLIT on its name only (corrected
above). X13, X14 and Y7 were SPLIT and have been re-asked above.

Banked raw output (scratchpad, not committed): the scouts (50 items) and
the co-architect rows (bridge 31, firewall 31, falsifier 28 + 3 skipped).
The crosswalk below carries the evidence for each row.

---
## Crosswalk — instance authorization (Q1) and principal expansion (Q2)

Built by the main thread from five scouts (50 items, merged to 24 rows).
Raw scout output is banked in `coresearch-scouts-raw.md` and NOT forwarded.
Relation: ALREADY-HAVE / PARTIAL / NEW / CONFLICTS-ANCHOR / ELSEWHERE.
Grades: outside READ-IN-FULL (RIF), SECTION-READ (SR), ABSTRACT-ONLY (AO),
SECONDHAND (SH); inside VERIFIED-IN-CODE (VIC), CLAIMED, ABSENT.

### Q1 — instance authorization

| # | idea | mechanism and evidence | outside source (grade) | nearest inside surface (grade) | relation |
|---|---|---|---|---|---|
| X1 | **Ordered per-right fold (canonical ACE order)** | Each right bit is decided by the FIRST level in canonical order that names it; inside a level deny wins. Levels: explicit deny, explicit allow, then per ancestor distance (parent before grandparent) deny before allow. Straight-line: `undecided = requested; for L: refuse if deny_L & undecided; undecided &= ~allow_L`. Inputs are per-level allow/deny masks per (object, principal set); no interpreter. The level is a field of the grant record, never a list position. | MS-DTYP 2.5.3.2 + Win32 "Order of ACEs in a DACL" (RIF/SR, three scouts agree) | contract `rbac.rs:114-194` has NO deny input; `ScopeSpec.deny` is only the empty-intersection sentinel; `authorize_memberships` unions positives (`authorize.rs:252-309`) (VIC) | NEW |
| X2 | **Combiner is a declared property of each right family** | V17 = first-applicable across levels with deny-overrides within a level, per bit. V18 = permit-overrides, "highest role" as a join on read<write<owner, cut points instead of denies. Keystone `ScopeSpec.deny` / NGAC prohibitions / Cedar forbid = deny-overrides. Three combiners, so one shared deny flag cannot serve all. | XACML 3.0 §7.10, App. C (SR); Cedar arXiv 2403.04651 §3.2-3.3 (SR); NIST SP 800-178 §4-5 (SR) | none (VIC: absent) | NEW |
| X3 | **Unknown-level deny = Indeterminate{D}** | An unreported deny level must block an allow at its level. V17 promotes it to explicit (stricter than XACML). Compiled form: a third per-bit plane "could-deny-unknown", or fold it into the explicit-deny plane. Same house pattern as dir-sim's value + known planes (`active` + `active_known`). | XACML §7.10 (SR) | dir-sim `cloud.rs:15-21`, board entry 2026-10-07-dir-sim-v4-active (VIC) | PARTIAL |
| X4 | **One pass for check and for effective rights** | Windows' MAXIMUM_ALLOWED pseudocode computes `allowed & ~denied` over ALL ACEs, ignoring order, so it disagrees with the desired-access path on "explicit allow user, inherited deny group". Stalwart has the same split: `has_access_to_container` stops at the first matching grant while `container_acl` unions, so a user in two granted groups gets order-dependent answers. Lesson: one fold, called by both the yes/no check and the listing. | MS-DTYP 2.5.3.2 (RIF); Stalwart `sharing/resources.rs:92-135` vs `sharing/mod.rs:26-37` (VIC) | none yet | NEW (design rule) |
| X5 | **Instance grant record = witnessed relation lane** | `(object, principal, right|role, sign, level, witness, expiry)` stored like membership: sparse ordinal pairs, every witness kept, disagreeing witnesses both stay, a decision fold empties contested rows. HubSPO's handover W-1 already proposes delegations as a sparse ordinal-pair lane like memberships. | NGAC association (SR); Zanzibar tuples (RIF) | OGAR `ogar-dir-core/src/edge.rs:1-25` (DirEdge + evidence code); dir-sim `cloud.rs` contested-fails-closed fold (VIC); no grant types anywhere (ABSENT, closed search) | PARTIAL |
| X6 | **Bulk listing vs point check under mask-risc laws** | For one (principal, object) the four level masks are four `Any` terminals combined as scalars — fits today. Bulk "which objects may U see" needs a hop ACE-row → object-row (`ScatterOrU32`), whose output may not feed a later program, so levels cannot be recombined in bulk. Options: (a) compile per-object effective planes at version install; (b) K trustee-slot lanes per object row, K gathers ORed in one program; (c) a new grouped-any terminal. Limits: ≤500 permission entries per Exchange mailbox bounds (b). | Exchange Add-MailboxPermission (SR) | mask-risc `ir.rs:237, 250-282, 290-292, 322-342` (VIC); quack `lib.rs:366-391` (VIC) | PARTIAL |
| X7 | **Resource-tree inheritance is a 4th relation, not the class lattice** | Inheritance down an instance tree (NTFS, SharePoint) is a resource-parent relation, distinct from membership, role implication and class-parent. Keystone I-K2's "no system inherits grants up the hierarchy" is scoped to the class lattice. Zanzibar models it as `tuple_to_userset` (one stored edge per hop); broken inheritance = the inherit edge is absent while containment stays. | Zanzibar §2.3.1 (RIF); NGAC object-attribute DAG (SR) | keystone `CLASSID-RBAC-KEYSTONE-SPEC.md:66` I-K2 (VIC, doc); board entry 2026-08-23 membership-is-participation-not-ancestry (VIC) | PARTIAL |
| X8 | **Dn128 + `subtree()` as the V18 tree carrier** | Dn128: up to 16 one-byte levels with explicit depth; dir-sim selects a subtree in one Quack program (depth gate + 16-byte ternary match over a strided lane). Nearest-broken-inheritance = subtree(anchor) minus deeper anchors' subtrees, or a stored anchor-ordinal lane read by Gather. Limit: ≤256 children per parent (fails closed on overflow); big SharePoint roots may exceed it. `ScopePath` (4×u64) cannot hold this; Spear spends 2 of 4 segments on one GUID. callcenter `dn_path.rs` hashes segments (destroys containment) — not a carrier. | — | OGAR `ogar-dir-core/src/dn128.rs:1-45, 92-142`; dir-sim `lib.rs:182-240` (VIC) | PARTIAL |
| X9 | **Library regime has no deny; Graph expiry sentinel** | Graph drive-item roles are read/write/owner only, so V18 is positive grants + cut points. `expirationDateTime = 0001-01-01` means "no expiry": an ingest that treats `expiry < now` as expired would expire every non-expiring grant. The sentinel must be mapped once at the boundary. Expiry = per-grant expiry lane checked against a version clock or `now`. | Graph v1.0 permission resource (RIF) | plan V18 `:590-592` (doc) | NEW |
| X10 | **Staged evaluator** | Windows central access policy runs a staged rule beside the enforced one and only logs disagreements. Run the V17/V18 instance evaluator staged beside Spear's current owner-only decision; with owner-only grants the disagreement count must be exactly 0. | MS-DTYP AccessCheck (RIF) | Spear `src/mail/iam.rs` (VIC) | NEW |
| X11 | **Remove-only layer = narrow** | Central access policy ANDs into the DACL result, so it can only remove rights; Windows' own form of "class grant AND instance grant". | MS-DTYP (RIF); NGAC policy-class conjunction (SR) | contract `ScopeSet::narrow` `rbac.rs:281-285` (VIC) | ALREADY-HAVE |
| X12 | **Reverse audit query** | NGAC PReview answers "all users with op on object" by one reverse linear sweep. That is V17's "who holds FullAccess/SendAs on this mailbox" without a per-user loop. | Mell et al., JoWUA 2017 §4-5 (SR) | none | NEW |
| X13 | **Stalwart conflates mail reach, ACL grants and send-as** | Stalwart's single membership relation gives a member the group's ACL grants, passes `assert_is_member` for Identity/EmailSubmission, and adds the group's addresses to the member's (`principals.rs:714-729`), so the member can create an identity with the group's address and send as it. dir-sim's backend feeds it mail chain ∪ SID chain (Stalwart #11). Result: implicit SendAs on any addressed group, distribution lists included, with no grant; DL members receive DL ACL grants. Latent until #20. Fix shape: feed each purpose its own chain, never the union. | Stalwart code (VIC, read not executed) | Stalwart `directory/src/backend/dirsim/mod.rs:180-210`; `cache/directory.rs:118-139`; `cache/principals.rs:714-729`; `jmap/src/identity/set.rs:72-87`; `jmap/src/submission/set.rs:537-545` (VIC) | CONFLICTS-ANCHOR (A1, A5) — in shipped code |
| X14 | **V17 trustee rules vs Microsoft cmdlet reference** | (1) Add-MailboxPermission: security groups that are not mail-enabled "do not work" — the Applicable line includes Exchange Server, not only EXO; V17 says on-premises they are valid. (2) Add-RecipientPermission (EXO, SendAs) accepts "security groups" with no mail-enabled requirement; V17 says EXO requires mail-enabled. (3) Set-Mailbox -GrantSendOnBehalfTo accepts a mailbox, mail user or mail-enabled security group ("a mail-enabled security principal"); V17 says distribution groups included and the mail chain confers it. (4) SendAs targets also include mail users, contacts, dynamic DLs. NOT read: "Manage permissions for recipients", Set-DistributionGroup -GrantSendOnBehalfTo. | Microsoft Learn cmdlet reference (SR) | plan V17 `:474-495` (doc) | CONFLICTS-ANCHOR (A5 detail) — operator decision |
| X15 | **Equivalence/analyzability probe** | Cedar's restricted language admits SMT equivalence of policy sets; usable idea: two compiled instance-grant sets must admit the same (principal, object, right) triples as a reference evaluator. HEALTHCARE §5 states the same falsifier shape. | Cedar §4 (SR) | OGAR `docs/HEALTHCARE-TRANSCODING.md` §5 (doc) | PARTIAL (probe shape) |
| X16 | **Conflict outcome "invalid"** | ODRL lets a permission/prohibition conflict void the policy and be reported. V17 already reports a cancelling deny. | W3C ODRL 2.2 (SH) | plan V17 (doc) | PARTIAL |

### Q2 — principal expansion as a plane

| # | idea | mechanism and evidence | outside source (grade) | nearest inside surface (grade) | relation |
|---|---|---|---|---|---|
| Y1 | **Resident plane, rebuilt per version (never a hop)** | Quack/mask-risc allow a foreign plane only if RESIDENT: never the Keep or `ScatterOrU32` output of another program; dir-sim seals outputs in `Kept` (compile-fail doctest); the capstone refuses a cyclic n-hop. So the closure is built outside the fold once per VersionId and registered as a resident plane that later programs Gather. Keyed by version, it needs no invalidation. Windows does the same per logon session (tokenGroups computed once into the token). | MS-ADTS tokenGroups (SR); Win32 AccessCheck (RIF) | quack `lib.rs:144-155, 366-391`; mask-risc `ir.rs:82-122, 257-342`; dir-sim `exec.rs:26-53`, capstone (VIC); today's walk is per call, scans every row, re-sorts nesting pairs (`view.rs:527-687`), `store.rs:139-153` re-folds lineage per `view()` (VIC) | PARTIAL |
| Y2 | **Leopard shape: groups×groups closure + user→direct-group lane** | Store each group's flattened descendant set and each user's DIRECT groups only; a principal set = OR of the ancestor rows of the user's direct groups. Never densify users×groups. Offline build at a snapshot + timestamped add/delete layer merged at query time. Measured: <150 µs median, 1.56M QPS; one tuple change can fan out to tens of thousands of index events. | Zanzibar §2.4.4, §3.2.4-3.2.5 (RIF) | dir-sim snapshot + overlay already has "offline index + incremental layer" shape (`store.rs:89-140`) (VIC); nesting exists only as identity pairs, no ordinal lane (`view.rs:814-831`) (VIC) | PARTIAL |
| Y3 | **Size the plane by trustee groups only** | Precompute only the permission sets clients choose; the plane can be users × (groups some grant names as trustee), not users × all groups. Dense bound 2³² bits = 512 MiB/version applies only if every group is a trustee. | AuthZed Materialize docs (SH) | dir-sim `snapshot.rs:62-64` (65,536×65,536) (VIC) | NEW |
| Y4 | **Intern distinct principal sets** | Users with identical direct-group (or SID) sets share one closure; store a set id per user + a sets×groups table. Only worth it if distinct sets ≪ users (census first; drop dedup if ratio ≈ 1). No population-sized seen-set (ScatterCountU32 HELD; use key-ordered runs). | Molloy et al. SACMAT 2008 (AO) | plan `2026-10-08-resident-projection-fold-mask-v1.md` D-RPF-2/3 (CLAIMED, plan) | PARTIAL (probe) |
| Y5 | **Rebuild vs incremental maintenance** | Insertions only grow a closure (cheap). Deletions need overdelete/rederive: on a 10k-node/100k-edge DAG, deleting 1,000 edges took 64.9 s modular vs 29.6 s full rematerialization. HubSPO's stored-mask maintenance per Lance version (P-LIST-1: 1.7–2.6 ms for 1,000 changes, matches full recompute) is for single-row predicates; a closure is not single-row. Neither path may be assumed. | Hu/Motik/Horrocks arXiv 1811.02304 (SR); DBSP arXiv 2203.16684 (SR) | HubSPO plan H-ADR-5/9, P-LIST-1 (doc, measured) | PARTIAL (probe) |
| Y6 | **Separate planes per chain, with a direct/indirect tag** | SCIM `User.groups` is a read-only published closure over every group kind, each entry tagged direct/indirect. Three different closures exist here: all-kinds nesting, SID chain, mail chain; each needs its own plane. Keep the direct-vs-reached bit so a witness can explain itself. A consumer with one membership notion is fed per purpose, never a union (see X13). | RFC 7643 §4.1.2, §4.2 (SR) | dir-sim `view.rs:607-641` (VIC); no witness of which chain conferred a SID (`store.rs` explain_membership covers one direct edge) (VIC) | PARTIAL |
| Y7 | **tokenGroups fidelity: user SID, primary group, sIDHistory** | tokenGroups = transitive closure over memberOf ∪ primaryGroupID edges, filtered to security groups, plus every reached object's sIDHistory. `security_identifiers` = tokenGroups minus the user's own SID, the primary group and sIDHistory. ogar-ad does not ingest primaryGroupID; nothing maps SID→GUID. | MS-ADTS tokenGroups, MS-DRSR 4.1.8 (SR) | ogar-dir-core `sid.rs:1-36` (formats only); ogar-ad `lib.rs:35-44`; dir-sim `view.rs:633-641` (VIC); closed search for primaryGroupID/tokenGroups = none (ABSENT) | PARTIAL |
| Y8 | **Group→role map is an explicit assignment record** | SCIM leaves the group→role map to the service provider, never on the group. OGIT Auth reifies it as a `RoleAssignment(team, role, dataset)` node — the NGAC association at class/dataset grain. Stalwart has no group→role map at all (roles come only from the account). HubSPO handover W-2 proposes RBAC data shaped like `GrantSource`, keyed by the group's Guid128. Constraint: `RbacBinding` drops grants outside the plug, so a dynamic map can only name roles the plug declares (`&'static str`); `ActorSource::roles_of` borrows a slice, so the source must own a precomputed role slice per actor. | RFC 7643 §4.1.2 (SR); OGIT NTO/Auth RoleAssignment.ttl (SR) | contract `rbac_plug.rs:36-44, 259-299`, `rbac.rs:338-340`; ogar-rbac `GrantSource`; handover `lance-graph/.claude/handovers/2026-10-10-1600-hubspo-rs-to-lance-graph.md` W-2 (VIC/doc) | PARTIAL |
| Y9 | **Principal matching as the Q1/Q2 boundary** | RPPM: stage 1 maps (subject, object) to named principals by path conditions over relationship labels (π+ for unbounded chains); stage 2 looks up signed rules for those principals. Liberal-grant (rights accumulate across matched principals) vs strict-grant give different answers for multi-right requests; Windows ACEs are liberal-grant. | Crampton & Sellwood arXiv 1406.4988 (SR); OpenMRS arXiv 1503.06154 (SR) | dir-sim `security_identifiers` (stage 1) (VIC) | PARTIAL |
| Y10 | **Subject-side nesting carries no priority** | In Windows/SharePoint, specificity lives only on the object axis (explicit vs inheritance distance); a group reached through nesting is at the level of its own ACE. So the principal plane can be a flat bitset without depth. Falsifier: any V17/V18 rule that ranks a deny by group depth would break this. | FAF / Samarati FOSAD 2000 (SH); plan V17 ("a group entry is at the level of its own ACE") | plan V17 `:500-504` (doc) | ALREADY-HAVE (confirmed) |

### Belongs elsewhere (not authorization)

| # | idea | mechanism and evidence | source (grade) | inside | relation |
|---|---|---|---|---|---|
| Z1 | **RFC 8693 act vs may_act** | §4.1: only top-level claims and the current actor may be considered; prior actors are informational (CONFIRMED). "Attenuation only" is NOT in the RFC — A11 is a stricter local choice, compatible with §1.1's "some of its rights". `may_act` = standing eligibility checked at exchange: an AND gate, never a source of rights. §1.1 impersonation vs delegation maps to SendAs (impersonation, no trace) vs SendOnBehalf (delegation, Sender header carries the actor); FullAccess is neither. | RFC 8693 §1.1, §4.1, §4.4 (RIF) | ogar-auth `user.rs:169-175, 213-234` (VIC) | ELSEWHERE (A11 refined) |
| Z2 | **Compound principals ("B for A")** | ABLP/Taos: an identity carries its transit path; a delegated compound principal holds no more than the delegator. Formal ancestor of `act`. As a trustee it would let provenance decide grants. | Zanzibar §5 citing TOPLAS 1993 (SH) | `ScopeSet::narrow` (VIC) | ELSEWHERE / ALREADY-HAVE as narrow |
| Z3 | **OAuth RAR authorization_details** | A consented ceiling: the server issues only what the standing grant allows; unknown types/fields are refused. Feeds `narrow`; never carries an instance grant. | RFC 9396 §2-9 (SR) | `ScopeSet::narrow` (VIC); ScopePath can't hold a per-instance id | ELSEWHERE (attenuation) |
| Z4 | **Unredeemed Graph invitation** | A sharing invitation to an address that matches no account carries address + role but no grantee until redemption. It is identity resolution: hold it dormant, it decides nothing. Same bucket: V18 `existingAccess` links. | Graph v1.0 permission resource (RIF) | A12; dir-sim `validate.rs` address_owner (VIC) | ELSEWHERE (A12) |
| Z5 | **Role mining by concept lattice** | Derive candidate roles from observed user×right relation. Design-time role engineering, not a decision mechanism; its usable consequence is Y4. | Molloy et al. (AO) | — | ELSEWHERE |

### Anchors affected

- A5 (V17 trustee detail): contradicted by X14 on three points; operator decision.
- A11: refined by Z1 — the RFC permits considering the current actor; "attenuation only" is our stricter choice, and `may_act` is an admissible AND gate.
- A1/A5: violated in shipped Stalwart code by X13.
- I-K2: scoped by X7 — it speaks to the class lattice, not instance trees.
