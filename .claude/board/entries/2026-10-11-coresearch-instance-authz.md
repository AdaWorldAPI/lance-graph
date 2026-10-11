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
| X1 | ordered per-level allow/deny fold | PASS | ADOPT-NOW | fixtures X1 (explicit allow vs inherited deny, same-level conflict, distance order) + X15 |
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
| Y9 | two stages (principals, then grants); liberal grant | C | ADOPT-NOW (framing) | RPPM's path-condition evaluator is a runtime walker: SKIP it |
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

Banked raw output (scratchpad, not committed): scouts (50 items), the
crosswalk (X1–X16, Y1–Y10, Z1–Z5), and the co-architect rows (bridge 31,
firewall 31, falsifier 28 + 3 skipped).
