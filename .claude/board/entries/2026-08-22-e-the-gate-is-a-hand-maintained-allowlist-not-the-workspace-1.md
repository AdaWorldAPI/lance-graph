## 2026-08-22 — E-THE-GATE-IS-A-HAND-MAINTAINED-ALLOWLIST-NOT-THE-WORKSPACE-1 — nine workspace members are in no CI job at all, and adding one to `members` adds it to nothing

**Status:** FINDING (measured; every claim is a grep over `.github/workflows/`
and root `Cargo.toml`). **Confidence:** High. **Supersedes the cause** given in
`E-A-CRATE-WITH-ZERO-CONSUMERS-IS-BUILT-BY-NOTHING-AND-CAN-BE-MERGED-BROKEN-1`
(same day, same session, mine).

That entry explained a crate that did not compile at `main` by saying a crate
with zero consumers is in nobody's build graph. **That is wrong.**
`crates/lance-graph-hydrate` is listed in `[workspace] members` (root
`Cargo.toml:25`); `cargo build --workspace` would compile it. The explanation
was plausible, fitted the symptom, and was never checked against the manifest.

**The actual mechanism.** No workflow in this repo runs `--workspace` or
`--all`. Every gate names one crate by path:

```
build.yml:78    cargo build   --manifest-path crates/lance-graph/Cargo.toml
rust-test.yml   cargo test    --manifest-path crates/<name>/Cargo.toml   (×14)
style.yml:75+   cargo clippy  --manifest-path crates/<name>/Cargo.toml   (×6)
style.yml:150+  cargo fmt     --manifest-path crates/<name>/Cargo.toml   (×3)
```

`style.yml:152` even says so in a comment: *"`cargo fmt --all` never reaches
it."* The gate is a **hand-maintained allowlist**. Membership in
`[workspace] members` is therefore not a build guarantee — it is a lockfile and
`-p` convenience, nothing more.

**Measured scope, not just my crate.** Of 25 members, ELEVEN appear in no
workflow. Two of those (`lance-graph-catalog`, `lance-graph-planner`) are
dependencies of `lance-graph`, so their LIBS compile inside a gated build
(their tests still never run). The remaining NINE are gated by nothing at all:

```
lance-graph-benches            lance-graph-rbac
neural-debug                   lance-graph-ontology
lance-graph-archetype          lance-graph-consumer-conformance
sigma-tier-router              cognitive-shader-driver
lance-graph-hydrate
```

`lance-graph-hydrate` was simply the one that a semver-compatible upstream
change (`object_store 0.13.2` moving `get`/`put` onto `ObjectStoreExt`) happened
to break. Nine crates carry the same exposure; nothing about this was specific
to having no consumer.

**Why the wrong cause was attractive** — worth naming, because it is the
Kahneman System-1 shape the workspace already warns about: the crate's own
`lib.rs` prominently says it has zero consumers, so a consumer-shaped
explanation was *pre-loaded* by the file being read. It explained the symptom
without contradicting anything visible, which is exactly when a claim needs its
own check and does not get one. The check was one grep of `Cargo.toml`.

**The durable fix is a gate, not a consumer.** Landing
`VersionedGraph::hydrate_from` (this branch) does make `lance-graph-hydrate` a
dependency of a gated crate, so its lib will now compile in CI — but that is a
side effect, and it leaves its TESTS ungated and the other eight untouched.
The fix is a workflow line per crate, or one `--workspace` job. Filed as
`ISSUES.md` `ISS-CI-GATE-IS-AN-ALLOWLIST-NINE-MEMBERS-UNGATED`.

**Outcome, same day (PR #984):** both, and the count above was wrong. A
workflow line per crate for every member — each measured locally before its
gate was armed — PLUS `cargo build --workspace` as the net that covers future
members without a line. Wrong count because the member check extracted with
`"crates/[a-z0-9-]+"`: no underscore, so `crates/surreal_container` was
invisible to it, and `tools/dto-class-check` is not under `crates/` at all.
Eleven, not nine. A membership check blind to two of its inputs is this
entry's own defect class, one level up, in the instrument rather than the
workflow — which is the part worth carrying forward.

