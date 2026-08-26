## 2026-08-24 — E-GIT-SOURCED-CRATE-CANNOT-PATH-DEP-OUTSIDE-ITS-REPO-1 (follow-up) — `cognitive-stack`'s OGAR deps fixed the same way; its `ndarray` dep left path-only, and why: a same-version git+path duplicate is a real type-identity hazard, not just style

**Status:** FIX — [MEASURED] (`cargo check -p cognitive-stack` green, 4m52s
clean build under the new deps; `cargo check` also directly demonstrated
the hazard this entry warns about — see below).

`crates/cognitive-stack` carried the identical escaping-path shape as
`lance-graph-ogar` before PR #1019 (`ogar-vocab`/`ogar-ontology`/
`ogar-adapter-surrealql` all `path = "../../../OGAR/crates/..."`, from the
same 2026-07-07 "NO-PIN" policy note that also covered `symbiont`).
Switched to `git = "https://github.com/AdaWorldAPI/OGAR", branch = "main"`
— identical URL+branch to `lance-graph-ogar`'s own pin (which
`cognitive-stack` also depends on), so Cargo unifies both into ONE
resolved OGAR source. Safe specifically because `lance-graph` (also a
dependency here) has NO OGAR dependency at all — nothing else in the
graph could bring in a competing OGAR source.

**`ndarray` was NOT switched — a first attempt was, and directly
demonstrated why not.** `lance-graph` (path-dep'd by `cognitive-stack`)
has its OWN escaping path dep on `ndarray`
(`crates/lance-graph/Cargo.toml`, gated behind the `ndarray-hpc` feature,
which is in `lance-graph`'s DEFAULT feature set), unfixed. Switching
`cognitive-stack`'s OWN direct `ndarray` dep to `git` while `lance-graph`'s
stayed `path` produced a build with **two distinct `ndarray v0.17.2`
package instances** (`git...?rev=0129b5c8...` — a pre-existing pinned
instance, likely from `p64`/`causal-edge` — plus a NEW
`git...#80f0b01f` branch-tracked instance, plus the still-present local
path instance) — visible directly in `cargo check`'s package list, their
`fractal` sub-crate duplicated the same way. Cargo treats path and git
sources as different identities even at the same version (the same
reason the `[patch]` sections elsewhere in this repo exist for
`lance-graph-contract`) — a real type-identity hazard if any code path
ever passes a value between the two instances, not merely a naming
inconsistency. It compiled anyway (Rust tolerates multiple crate
instances until a signature unifies them directly), which makes this an
easy footgun to miss: **a clean `cargo check` does not prove dependency
source unification.** Reverted before commit; `ndarray` stays `path` here.

**The real fix is upstream, out of scope for this commit:**
`lance-graph`'s own `ndarray-hpc` feature (in its DEFAULT feature set)
still path-deps `ndarray` the same escaping way. This means the MAIN
`lance-graph` crate itself would fail the identical way `lance-graph-ogar`
did for any external `git` consumer that enables `ndarray-hpc` (or
anything requiring it) — currently LATENT, not yet triggered, because no
observed external consumer has needed that feature (MedCare-rs's Railway
break was specifically about `ogar-loco`, an unconditional dependency;
`ndarray-hpc` being optional means Cargo's resolver never touches the
escaping path unless the feature is actually activated). Flagged, not
fixed here — touching `lance-graph`'s default feature set is a bigger,
more sensitive change than this follow-up's scope.

**Files:** `crates/cognitive-stack/Cargo.toml`.

## 2026-08-24 — E-GIT-SOURCED-CRATE-CANNOT-PATH-DEP-OUTSIDE-ITS-REPO-1 — `lance-graph-ogar`'s OGAR path deps could not resolve for ANY external `git` consumer; the "sibling-checkout" mental model does not survive a `git` dependency

**Status:** FIX — [MEASURED] (verified: `cargo check` + `cargo test`
green, 72+ lib tests including the codebook COUNT_FUSE, all under the new
`git` deps).

**Root cause, generic to every external consumer, not one project:**
`crates/lance-graph-ogar/Cargo.toml` depended on five OGAR crates via
`path = "../../../OGAR/crates/..."` — an escaping relative path,
justified by a documented "operator policy (2026-07-07): NO PINS" note
for this sandbox's local multi-repo layout. That policy is correct for a
LOCAL build (this session, tesseract-rs CI's sibling-checkout precedent)
but cannot survive `lance-graph-ogar` being pulled as a `git` dependency
by ANY external repo (`some-consumer -> lance-graph-ogar` via
`git = "https://github.com/AdaWorldAPI/lance-graph", branch = "main"`).
Cargo clones a git-sourced crate into its own opaque checkout cache
(`~/.cargo/git/checkouts/lance-graph-<hash>/<rev>/`); a `path =
"../../../OGAR/..."` inside it resolves relative to THAT cache directory,
which has no OGAR sibling — nothing ever puts one there, regardless of
how fresh either repo is on GitHub or on the builder's disk. **A crate
with an escaping path dependency cannot be consumed via `git` by any
external repo, ever — this is a hard Cargo constraint, not a missing
setup step, and it affects every one of `lance-graph-ogar`'s consumers
identically**, not a single project's build. First checked "is OGAR
stale?" and ruled it out: local `/home/user/OGAR` matched `origin/main`
exactly, `ogar-loco` has been on `origin/main` since 2026-08-05
(`89d0d3a9`).

**Fix:** switched all five OGAR deps (`ogar-vocab`, `ogar-class-view`,
`ogar-ontology`, `ogar-loco`, `ogar-adapter-surrealql`) from `path` to
`git = "https://github.com/AdaWorldAPI/OGAR", branch = "main"`.
Verified `cargo check`/`cargo test` green standalone (this crate owns its
own `[workspace]` root); both git deps resolved to the identical rev
(`#719471db`) already referenced by the existing `[patch]` section, so no
new contract-source divergence was introduced. Tradeoff accepted
explicitly: the 2026-07-07 policy's "always current sibling, no
Cargo.lock pin" property is gone for this crate — a pin is now unavoidable
for external consumability. Local/in-sandbox dev is unaffected (same
public repo, still fetched/cached).

**Known residue, NOT fixed here:** `symbiont` and `cognitive-stack`
(named in the same superseded comment as sharing this pattern) carry the
identical escaping-path shape and were left untouched — `symbiont` is
separately marked deprecated/operator no-go; `cognitive-stack` is
unaudited for external git-consumers. Tracked here so a future session
does not assume this fix covers them.

**Fences:** no behavior change to OGAR's contract surface, only the
dependency SOURCE type; the `[patch]` section (already redirecting
`ogar-class-view`'s transitive `lance-graph-contract` onto the path copy)
is untouched and still correct, since patches redirect FROM a git source
TO a path source — the broken direction was the reverse (a path
dependency escaping a git-sourced crate), which `[patch]` cannot fix at
all (path dependencies are not patchable).

**Files:** `crates/lance-graph-ogar/Cargo.toml`.

