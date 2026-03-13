# CC_SESSION_BOOTSTRAP.md (lance-graph upstream)

## Clone. Sync. Split. Push.

```bash
mkdir adaworld && cd adaworld

# REQUIRED:
git clone https://github.com/AdaWorldAPI/lance-graph
git clone https://github.com/AdaWorldAPI/holograph      # read only, BlasGraph source

cd lance-graph
git remote add upstream https://github.com/lance-format/lance-graph.git
git fetch upstream
cargo test --workspace
```

## First command in every session:

```bash
cat CLAUDE.md
cat .claude/UPSTREAM_PR_SESSIONS.md
```

## What you write to:

```
lance-graph    Push to AdaWorldAPI fork, PR to lance-format upstream
```

## What you read from:

```
holograph      BlasGraph source for PR C. DO NOT MODIFY.
```

## NOT in scope:

```
ladybug-rs     Not touched. Not cloned. Not referenced in upstream PRs.
rustynum       Not touched. No upstream PR may depend on it.
n8n-rs         Not touched.
crewai-rust    Not touched.
```

## Upstream PR target:

```
lance-format/lance-graph    main branch
Maintainer: beinan
PR #146 (ours): CLOSE FIRST, then split into A/B/C
```
