# TD-SDR PR Release Plan

**Spec version:** 1.0  
**Date:** 2026-05-13  
**Author:** W7  
**Covers:** TD-SDR-PR-FOLLOWUP-1 + TD-SDR-CONSUMER-PUSH-1  
**Branch:** `claude/lance-datafusion-integration-gv0BF`

---

## Context: Branch State

Captured SHAs (git log origin/main..HEAD):

```
3de43be  docs(board): harvest OGIT-OSINT-Palantir/Neo4j-q2 route + FMA smoke-test anchor
39125c3  docs(board): harvest Path C (ndarray::simd) + super-domain subcrates + API-drift + OGIT-axis orthogonality
f8a2699  docs(board): harvest ractor-supervisor path as Path B complement to thinking-engine (Path A)
ce01eb9  docs(board): harvest D-SDR Tier A status + thinking-engine epiphany + transcripts
dc9e081  feat(lance-graph-callcenter): D-SDR-5 wire authorize_* through Policy with chained audit emission
dabd510  chore(deps): Cargo.lock after D-SDR-4
1d0157f  feat(lance-graph-callcenter): D-SDR-4 merkle-chained audit log for UnifiedBridge
2c3e87d  feat(lance-graph-callcenter): D-SDR-3 per-family codebook table (OgitFamilyTable + FamilyEntry)
3e94a27  knowledge: log E1-E6 splat + formal-grounding epiphanies (inbox doc, separate from active spec)
```

**D-SDR feature commits (cherry-pick targets):**

| SHA | Ticket | Description |
|-----|--------|-------------|
| `2c3e87d` | D-SDR-3 | per-family codebook table (OgitFamilyTable + FamilyEntry) |
| `1d0157f` | D-SDR-4 | merkle-chained audit log for UnifiedBridge |
| `dabd510` | D-SDR-deps | Cargo.lock after D-SDR-4 |
| `dc9e081` | D-SDR-5 | wire authorize_* through Policy with chained audit emission |

Governance harvest commits (3e94a27, ce01eb9, f8a2699, 39125c3, 3de43be) are
board-only and do NOT need to be included in PR-A; they are branch state only.

---

## PR-A: lance-graph D-SDR Follow-Up PR

### TD-SDR-PR-FOLLOWUP-1 context

Per the task definition: 5 commits stacked on merged main, no follow-up PR yet exists.
The follow-up PR covers D-SDR-3/4/5 and related Cargo.lock bump. The governance harvest
commits are excluded from the PR since they are .claude/board/ changes with no
production artifact.

### PR-A Title

```
feat(lance-graph): D-SDR-3/4/5 PolicyRewriter chain + UnifiedBridge audit
```

### PR-A Body Template

```markdown
## Summary

Ships D-SDR-3, D-SDR-4, and D-SDR-5 as a stacked follow-up to the UnifiedBridge
landing (PRs #355-363). Together these three tickets deliver:

- **D-SDR-3** (commit 2c3e87d): OgitFamilyTable + FamilyEntry -- per-family
  codebook table enabling per-SDR-family policy resolution without a global switch.
- **D-SDR-4** (commit 1d0157f): Merkle-chained audit log wired into UnifiedBridge
  -- every authorize call produces a cryptographically ordered audit entry.
- **D-SDR-5** (commit dc9e081): authorize_* wired through the Policy layer with
  chained audit emission -- completes the PolicyRewriter chain (D-SDR-3 -> D-SDR-4 -> D-SDR-5).
- Cargo.lock bump (dabd510) reflecting the above dependency changes.

## Changes

| Crate | File(s) changed | Nature |
|-------|----------------|--------|
| lance-graph-callcenter | src/family_table.rs (new) | OgitFamilyTable + FamilyEntry |
| lance-graph-callcenter | src/audit.rs (new) | MerkleAuditLog impl |
| lance-graph-callcenter | src/policy.rs (extended) | authorize_* wired to Policy |
| lance-graph-callcenter | src/bridge.rs (extended) | UnifiedBridge audit emission |
| root | Cargo.lock | dependency lock bump |

## Migration Notes

**Consumers of UnifiedBridge::authorize_*:**

Prior to this PR, authorize_* bypassed the Policy layer and returned unaudited results.
After this PR, all authorize_* calls:

1. Pass through PolicyRewriter (D-SDR-3 family codebook lookup).
2. Emit a MerkleAuditEntry (D-SDR-4 audit chain).
3. Return AuthorizeResult with audit_hash: [u8; 32] populated.

**Required action for consumer crates (medcare-rs, smb-office-rs):**

Before (pre-D-SDR-5):
  let result = bridge.authorize_patient(req)?;

After (D-SDR-5+):
  let result = bridge.authorize_patient(req)?;
  // result.audit_hash is now always populated.
  // If your consumer ignores audit_hash, no code change needed (non-breaking).
  // If you previously matched on AuthorizeResult fields: add .. to pattern.

**Deprecation shim (W3 cross-flag):** The deprecation shim for the pre-D-SDR-5
authorize_* API surface ships as part of PR-A. See Cross-flag with W3 below.
Consumer PRs (PR-B, PR-C) compile against the shim during migration.

## Test Plan

- [ ] cargo check on workspace passes (no new warnings in lance-graph-callcenter).
- [ ] cargo test -p lance-graph-callcenter -- all existing tests green.
- [ ] cargo test -p lance-graph-callcenter -- audit -- new audit tests green.
- [ ] cargo bench -p lance-graph-callcenter -- smoke -- bench smoke runs < 500 ms.
- [ ] Consumer zipball build (see CI matrix).

## Cross-references

- Supersedes / follows up: PRs #355-363 (UnifiedBridge landing)
- Deprecation shim: W3 spec td-api-drift-deprecation.md
- PR sequencing graph: W12 spec sprint-4-pr-graph.md
```

### PR-A Branch Push Strategy

**Option A (preferred -- branch already pushed to origin):**

```bash
git push origin claude/lance-datafusion-integration-gv0BF
# Then create PR via mcp__github__create_pull_request:
# base: main
# head: claude/lance-datafusion-integration-gv0BF
# (governance harvest commits are included but harmless -- they touch only .claude/board/)
```

**Option B (cherry-pick to clean branch, if reviewers object to board commits):**

```bash
git checkout -b feat/d-sdr-3-4-5-policy-chain origin/main
git cherry-pick 2c3e87d  # D-SDR-3
git cherry-pick 1d0157f  # D-SDR-4
git cherry-pick dabd510  # Cargo.lock
git cherry-pick dc9e081  # D-SDR-5
git push -u origin feat/d-sdr-3-4-5-policy-chain
```

**Recommendation:** Use Option A (branch already pushed, less merge risk). If CI
objects to the board-only commits, cherry-pick is the fallback.

---

## PR-B: medcare-rs UnifiedBridge Wiring

### Pre-push verification (run from medcare-rs root)

```bash
git log origin/main..HEAD --oneline
# Expected: at least one commit wiring UnifiedBridge / importing lance-graph-callcenter.
# If empty: the wiring is uncommitted. Stop, investigate, do not push.

git status
# Expected: clean working tree (all changes committed).

git diff origin/main..HEAD -- Cargo.toml
# Expected: lance-graph-callcenter dependency added.

cargo check --manifest-path Cargo.toml
# Must pass before push.
```

### Push command

```bash
git push -u origin HEAD
```

### PR-B creation command (mcp__github__create_pull_request)

```
repo:   AdaWorldAPI/medcare-rs
title:  feat(medcare-rs): wire UnifiedBridge via lance-graph-callcenter (D-SDR)
base:   main
head:   <current branch name>
```

**PR-B body template:**

```markdown
## Summary

Wires UnifiedBridge from lance-graph-callcenter into medcare-rs patient
authorization flow. Consumer-side counterpart of lance-graph D-SDR-3/4/5 (PR-A).

## Changes

- Cargo.toml: add lance-graph-callcenter dependency (pinned to PR-A SHA or
  semver after PR-A merges).
- src/auth/: replace direct policy calls with UnifiedBridge::authorize_patient.
- src/audit/: forward audit_hash from AuthorizeResult to audit sink.

## Dependency gate

This PR requires PR-A to merge first (or be available as a branch SHA).
CI is configured to build against the PR-A branch SHA until main is updated.

## Test plan

- [ ] cargo check passes against PR-A branch SHA.
- [ ] cargo test -- all medcare-rs tests green.
- [ ] cargo bench -- smoke -- authorize bench smoke < 1 s.
- [ ] Zipball cross-repo build (see CI matrix).
```

---

## PR-C: smb-office-rs UnifiedBridge Wiring

### Pre-push verification (run from smb-office-rs root)

```bash
git log origin/main..HEAD --oneline
# Expected: at least one commit wiring UnifiedBridge for SMB office auth flow.

git status
# Expected: clean working tree.

git diff origin/main..HEAD -- Cargo.toml
# Expected: lance-graph-callcenter dependency added.

cargo check --manifest-path Cargo.toml
# Must pass before push.
```

### Push command

```bash
git push -u origin HEAD
```

### PR-C creation command (mcp__github__create_pull_request)

```
repo:   AdaWorldAPI/smb-office-rs
title:  feat(smb-office-rs): wire UnifiedBridge via lance-graph-callcenter (D-SDR)
base:   main
head:   <current branch name>
```

**PR-C body template:**

```markdown
## Summary

Wires UnifiedBridge from lance-graph-callcenter into smb-office-rs authorization
and audit flow. Consumer-side counterpart of lance-graph D-SDR-3/4/5 (PR-A).

## Changes

- Cargo.toml: add lance-graph-callcenter dependency.
- src/auth/: replace local policy dispatch with UnifiedBridge::authorize_document.
- src/audit/: forward audit_hash from AuthorizeResult.

## Dependency gate

Requires PR-A to merge first (or be available as a branch SHA for CI).

## Test plan

- [ ] cargo check passes against PR-A branch SHA.
- [ ] cargo test -- all smb-office-rs tests green.
- [ ] cargo bench -- smoke -- authorize bench smoke < 1 s.
- [ ] Zipball cross-repo build (see CI matrix).
```

---

## Sequencing and Merge Gate

```
Wave 1 (Day 0) -- P0 unblockers, must land first:
  W10 (slot widen u16 + BridgeError audit hook) -> Push + Merge
  W3  (deprecation shim)                         -> Push + Merge
  PR-A (lance-graph D-SDR-3/4/5)                 -> Push + Merge
         |
         v  Gate: PR-A merge SHA available

Wave 2 (Day 1-3) -- consumer migration, gate on Wave 1:
  PR-B (medcare-rs UnifiedBridge)    -> Push + Merge
  PR-C (smb-office-rs UnifiedBridge) -> Push + Merge
  W4 (super-domain subcrates)        -> parallel with B/C
  W8 (audit sink: Lance/JSONL)       -> parallel with B/C
```

**Merge gate rule:** PR-B and PR-C MUST NOT merge before PR-A has merged to main.
If PR-A is delayed, consumer PRs may be opened (for review) but must remain in
draft state until PR-A lands.

**Enforcement mechanism:**

Consumer CI jobs declare LANCE_GRAPH_SHA=<pr-a-branch-sha> as a build env var.
The cargo check step patches lance-graph-callcenter to that SHA.
After PR-A merges, the SHA is updated to main HEAD and consumer PRs are un-drafted.

---

## CI Matrix

### PR-A (lance-graph)

| Job | Command | Pass criterion |
|-----|---------|----------------|
| cargo-check | cargo check | zero errors |
| cargo-test | cargo test -p lance-graph-callcenter | 0 failures |
| cargo-bench-smoke | cargo bench -p lance-graph-callcenter -- smoke --test | exits 0 in < 60 s |
| consumer-compat-check | Build medcare-rs + smb-office-rs stubs against PR-A branch SHA (zipball) | cargo check passes for both consumers |

### PR-B (medcare-rs) and PR-C (smb-office-rs)

| Job | Command | Pass criterion |
|-----|---------|----------------|
| cargo-check | cargo check (with LANCE_GRAPH_SHA=<pr-a-sha>) | zero errors |
| cargo-test | cargo test | 0 failures |
| cargo-bench-smoke | cargo bench -- smoke --test | exits 0 in < 60 s |
| cross-repo-build | Download lance-graph zipball at PR-A SHA, cargo check consumer against it | passes |

**Cross-repo zipball build recipe (for CI scripts):**

```bash
#!/usr/bin/env bash
LANCE_SHA="${LANCE_GRAPH_SHA:-origin/main}"
TMPDIR=$(mktemp -d)
git clone --depth=1 https://github.com/AdaWorldAPI/lance-graph "$TMPDIR/lance-graph"
(cd "$TMPDIR/lance-graph" && git checkout "$LANCE_SHA")
cat >> "$CONSUMER_ROOT/.cargo/config.toml" <<EOF
[patch.crates-io]
lance-graph-callcenter = { path = "$TMPDIR/lance-graph/crates/lance-graph-callcenter" }
EOF
cd "$CONSUMER_ROOT"
cargo check
```

---

## Rollback Plan

### If PR-A breaks post-merge

```bash
# DO NOT force-push main. Use git revert:
git revert <pr-a-merge-sha>
git push origin main
# SLA: within 4 hours of merge if a regression is detected.
```

### If PR-B or PR-C breaks post-merge

```bash
git revert <pr-b-merge-sha>   # for medcare-rs
git revert <pr-c-merge-sha>   # for smb-office-rs
git push origin main
# SLA: within 4 hours.
```

### If both PR-B and PR-C break but PR-A is clean

Revert B and C, NOT A. The likely cause is consumer wiring (path import, API mismatch).
Investigate the AuthorizeResult field shape (D-SDR-5 introduced audit_hash).
Fix the consumer, re-open B/C. Do NOT revert A.

### Force-push prohibition

git push --force to main is NEVER used. The only safe rollback is git revert.
This preserves CI history, audit chain, and makes the revert visible to consumers.

---

## Cross-flag with W3 (Deprecation Shim)

W3 deliverable: td-api-drift-deprecation.md -- deprecation shim for pre-D-SDR-5
authorize_* API drift. This shim MUST ship as part of PR-A (same branch, same PR).

**Why:** Consumer PRs (PR-B, PR-C) compile against the new D-SDR-5 API. During the
window between PR-A merge and PR-B/C merge, any consumer still calling the old
authorize_* signature must get a #[deprecated] warning, not a compile error.

**Action required before PR-A opens for review:**

1. Confirm with W3 that the deprecation shim is committed into
   crates/lance-graph-callcenter/src/compat.rs on the same branch as D-SDR-5.
2. The shim re-exports the old authorize_* signature with:
   #[deprecated(since = "X.Y.Z", note = "Use UnifiedBridge::authorize_* with audit_hash field")]
3. lib.rs re-exports the compat module.

**If W3 shim is not ready when PR-A is ready to merge:** PR-A must wait in draft state.
The shim is a non-negotiable part of PR-A's definition of done.

---

## Cross-flag with W12 (PR Sequencing Graph)

W12 (sprint-4-pr-graph.md) documents the full wave structure and confirms:

- Wave 1: W10 + W3 + PR-A -- parallel within Wave 1.
- Gate W3 -> W7: W3 deprecation shim must be merged before PR-B/C consumer PRs.
  Enforced by: PR-A includes W3 shim (same commit / same branch).
- Wave 2: PR-B + PR-C + W4 + W8 -- parallel within Wave 2.
- Wave 3: W2 + W11 + W5 + W6 -- downstream convergence, no dependency on B/C.

W7 (this spec) must stay consistent with W12's wave table. If W12 updates the
wave structure, update this spec's Sequencing section accordingly.

LOC impact (from W12 estimates):

| PR | W12 LOC est | Notes |
|----|-------------|-------|
| PR-A (lance-graph D-SDR-3/4/5) | ~150 LOC | follow-up PR, release notes |
| PR-B (medcare-rs) | ~200 LOC | half of W7-PR-B/C/D total 400 LOC |
| PR-C (smb-office-rs) | ~200 LOC | other half |

---

## Open Questions

1. **Consumer repo locations:** The local paths for medcare-rs and smb-office-rs
   are assumed to be /home/user/medcare-rs and /home/user/smb-office-rs respectively.
   Before the main thread runs the push commands, confirm these paths exist and
   git log origin/main..HEAD returns the expected UnifiedBridge wiring commits.
   If the repos are not cloned locally, the push step requires a git clone first.

2. **PR-A merge strategy -- squash vs. merge commit:** The four D-SDR commits
   (2c3e87d, 1d0157f, dabd510, dc9e081) have clean, logical granularity.
   A merge commit preserves the per-ticket audit trail. A squash combines them.
   Recommendation: merge commit to keep D-SDR-3/4/5 individually revertable.
   Confirm with repo maintainer before merging.

3. **lance-graph-callcenter version bump:** After PR-A merges, consumer Cargo.toml
   files must reference a specific version or SHA. If lance-graph-callcenter is
   published to crates.io, bump to the new semver. If it is path-dep only, consumers
   must use the git SHA. Clarify the publication strategy before Wave 2 begins -- a
   missing version bump will silently pin consumers to pre-D-SDR-5 code.
