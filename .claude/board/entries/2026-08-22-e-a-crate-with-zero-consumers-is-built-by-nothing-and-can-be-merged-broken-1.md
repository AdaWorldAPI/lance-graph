## 2026-08-22 — E-A-CRATE-WITH-ZERO-CONSUMERS-IS-BUILT-BY-NOTHING-AND-CAN-BE-MERGED-BROKEN-1 — the hydration crate did not compile at `main`, and its own doc says why nobody found out

**Status:** ⊘ SUPERSEDED SAME-DAY by
`E-THE-GATE-IS-A-HAND-MAINTAINED-ALLOWLIST-NOT-THE-WORKSPACE-1` — the measured
breakage is real and unchanged; the MECHANISM named below ("zero consumers ⇒
not in the build graph") is WRONG. `lance-graph-hydrate` is a workspace
MEMBER. Kept in place per append-only; read the successor for the cause.
**Confidence:** High on the failure, RETRACTED on the cause.

`crates/lance-graph-hydrate` was minted 2026-08-17 (PR #957 + follow-ups) as
the generic object-store → local-volume hydration lifecycle, with a 5+3
hardening council on it. Its `lib.rs` states plainly that it has no consumer
yet — the deferral of `ISS-HYDRATE-DIR-AND-FILE-DUPLICATE-THEIR-STAGING-BODIES`
is even justified by that fact (*"cheap only while this crate had zero
consumers"*).

**Measured 2026-08-22, on a clean checkout of `origin/main` (#981):**

```
cargo build -p lance-graph-hydrate
error[E0599]: no method named `get` found for reference `&dyn ObjectStore`
... 5 errors, in the LIB, not the tests
```

`object_store 0.13.2` — the version the workspace lockfile already pins, on
`main`, unchanged by this branch — moved `get`/`put` off the base
`ObjectStore` trait onto an extension trait `ObjectStoreExt`. The fix is two
`use` lines. `cargo fmt -p lance-graph-hydrate` also rewrites four of its
files, so it was merged unformatted as well.

**The finding is not the breakage; it is the mechanism that hid it.** A crate
with zero consumers is not in any consumer's build graph. Nothing that CI
actually runs reaches it, so a semver-COMPATIBLE upstream change (0.13.x, no
major bump, no lockfile movement) silently invalidated it and no gate fired.
The absence of a consumer was recorded in the crate's own doc as a *cost
tradeoff* — it is also, and more importantly, a **verification hole**: the
crate's tests pass only in the one command nobody runs.

Two consequences, both narrow:

1. **A mint without a consumer needs an explicit build gate**, or it is
   documentation with a `Cargo.toml`. Either land the first consumer in the
   same arc, or add the crate to whatever CI job actually compiles.
2. **"Hardened by a 5+3 council" is orthogonal to "builds."** The council
   reviewed intent, duplication, and doctrine conformance — all real, all
   preserved by this fix. None of that is a compiler.

Corrected in the same commit that gives the crate its first mechanism with a
consumer path (`archive::hydrate_archive`), so the hole closes rather than
being recorded and left open.

