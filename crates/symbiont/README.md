# symbiont — the golden image

> **THE one and only way the full Ada stack is meant to compile+link as one
> binary:** lance-graph + lance `=7.0.0` + lancedb `=0.30.0` + the ndarray
> fork + the ractor fork + surrealdb (kv-lance only) + OGAR.

A successful build IS the golden image — the known-good foundation for the
kanban thinking, whose first test workload is the perturbation simulation
(`perturbation-sim`). The binary itself just prints the linked-stack line and
exits; the point is that the unified dependency graph **compiles and links**.

It is `exclude`d from the lance-graph workspace (own `[workspace]` table) so
the default lance-graph build/CI never pulls surrealdb-core + OGAR.

## Build

```bash
# Local (any session): git-deps resolve from the AdaWorldAPI forks
cargo build --manifest-path crates/symbiont/Cargo.toml

# Container (build/CI validation) — from the lance-graph repo ROOT:
docker build -f crates/symbiont/Dockerfile -t symbiont .
docker run --rm symbiont
```

## How it stays portable

- **Same-repo crates** (`lance-graph`, `perturbation-sim`) → path deps.
- **External forks** (`ractor`, `surrealdb-core`, `ogar-*`) → **git deps**
  pinned to `claude/jirak-math-theorems-harvest-rfii13`. No machine-specific
  paths, so it resolves in any session and on Railway/CI.
- **One wrinkle:** lance-graph's own `Cargo.toml` path-deps the ndarray fork as
  a sibling (`../../../ndarray`). In a session it's already there; the
  Dockerfile clones it next to the repo. (That's why ndarray isn't a git dep in
  this manifest — it enters via lance-graph's existing path dep.)

## Railway (build validation)

1. Point the service at this repo, root directory = repo root.
2. Set the Dockerfile path to `crates/symbiont/Dockerfile`.
3. The build proves the stack links; a green build is the deliverable.

**Auth:** a Railway server has access to all private AdaWorldAPI repositories
by design — no token wiring needed. The Dockerfile sets
`CARGO_NET_GIT_FETCH_WITH_CLI=true` so cargo fetches the git-dep forks through
the system `git` (which honors Railway's ambient credentials) instead of its
built-in transport.

## Notes

- ndarray is linked twice (lance-graph's path fork + surrealdb-core's git-rev
  fork) plus the real crates.io `ndarray 0.16.1` that lance-index needs. This
  is accepted as cosmetic — no ndarray type crosses the surrealdb↔lance-graph
  seam, so the duplication never manifests at a call boundary.
- The local-path variant of this crate (absolute `/home/user/...` paths) lives
  on `claude/perturbation-sim-nan-hardening`; this main version is the portable
  git-deps build.
