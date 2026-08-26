## 2026-08-18 — E-PIN-LANCE9-LANCEDB033-DF541-ARROW58-NO-DF53-1

**Status:** RULING `[operator]` + same-day discharge, measured.

**The ruling.** lance 9 / lancedb 0.33 / datafusion 54.1 (**no DF 53**) /
arrow 58 are ALWAYS pinned across AdaWorldAPI forks, usually consumed via
`[patch]` of the upstream repository git. DF 53 existed only for older syntax
in an out-of-scope research crate; all of lance-graph is DF-54-certified,
and single-major DF 54.1 is the discipline ("without DF 53/54 parallelism
it's much better, easier, faster to handle").

**Measured before acting:** the lock carried datafusion 53.1.0 AND 54.1.0;
the ONLY DF-53 source was `deltalake 0.32.4` (`^53.1.0`) behind the `delta`
feature — which was already NON-default and documented BROKEN (0.32 removed
`DeltaTableProvider::try_new`; the reader was never refactored). Checked the
registry: deltalake-core tops out at 0.32.4 / DF ^53.1.0 — no DF-54 deltalake
exists, so a bump could not discharge the ruling; removal could.

**Discharged:** `delta` feature + `deltalake`/`url` optional deps +
`DeltaTableReader` removed (Cargo.toml carries the dated removal note; a
future Delta reader returns only with a DF-54-compatible deltalake, as its
own deliberate PR). `DataSourceFormat::Delta` (a catalog metadata tag) stays.
**Post-removal lock: exactly ONE datafusion = 54.1.0; zero deltalake
entries.** `cargo check -p lance-graph` green (needed `protobuf-compiler`
installed in-sandbox — lance's prost build; the Dockerfiles already install
it). Docker pins surveyed on request: root + avx512 Dockerfiles = Rust
1.97.1 + protobuf-compiler/cmake, no feature references to delta, versions
purely from Cargo.lock — removal is docker-safe. The symbiont
Dockerfile's stale `rust:1.95` pin is MOOT: **symbiont is DEPRECATED**
(operator, same day — superseded by the supervisor/persistence arc,
PRs #879/#911/#912/#913) and is not live surface.

**Same-day scope pivot (operator, BINDING for RP-SEAL):** rustynum is NOT a
source of truth — "no rustynum, everything ndarray"; it is the historical
donor already ported into AdaWorldAPI/ndarray. The RP-SEAL workflow script
was corrected in place (source map + the two cell briefs naming rustynum);
the in-flight independent pass could not be force-stopped in this harness
build, so the ruling additionally binds consolidation as a hard filter
(re-anchor or surgically re-run any finding citing rustynum/symbiont).
Deltalake removal ratified in the same exchange ("we don't need deltalake").

**Drift cleanup executed (same day, on the operator's "absolute no-go"):**
audit result — A1's research report: 0 citations (clean); the Java/Panama
arc: zero involvement (ndarray::simd enforced throughout); merged docs
carried exactly one live symbiont citation (the lotus audit §3 candidate
row, now ⊘-struck) + CLAUDE.md's two symbiont-as-binding-consumer lines
(now ⊘-annotated). Live-code drift found and FIXED:
`lance-graph-cognitive`'s `rustynum_accel` shim was name-only (zero
rustynum dependency — it delegated to `ndarray::hpc::bitwise`); renamed
`simd_accel` (module + file + 11 call sites) and rerouted through the
sanctioned `ndarray::simd::hamming_distance_raw` re-export. Default
compile green; the `wip` feature's pre-existing not-yet-compiling state
carries zero errors naming the rename. Historical provenance comments
("replacing rustynum as of 2026-03-22", ndarray's port-history docs) kept
— the no-go bans USE, not history. Board history entries citing symbiont
stay append-only, superseded by this entry.

**Same-day factual correction (operator-pointed, verified at source):** the
"no DF-54 deltalake exists" premise holds only for crates.io RELEASES
(deltalake-core <= 0.32.4 / DF ^53). delta-io/delta-rs **main** already pins
`datafusion = 54.0.0` + `arrow = 58` (workspace Cargo.toml, fetched
2026-08-18) — so restoration is available NOW via the house mechanism (git
dep on the upstream repository, pinned rev) + a `DeltaTableReader` refactor
to the current builder API. The removal stands on the NEED ruling, not on
availability; the in-repo notes were corrected so no future session treats
restoration as blocked on a registry release.

**Second discharge — the lotus BLOCKER:** the [patch → upstream-git]
mechanism sanctioned consulting the upstream lance repository as source; the
exact v9.0.0 tag is cloned (`/tmp/sources/lance-9`, matching the Cargo.lock
checksum) plus current upstream (`/tmp/sources/lance-main`) for the RP-SEAL
two-column discipline. D-LOTUS-6's "lance source absent" BLOCKER is LIFTED;
capability findings deliberately enter through the RP-SEAL Domain-A
independent passes, not this entry (independence rule).

**Supersedes in place:** CLAUDE.md's "BOTH MAJORS ARE REQUIRED — do NOT fix
Cargo.lock to one" datafusion note (written when `delta` was default and
load-bearing) — corrected in the same commit.

