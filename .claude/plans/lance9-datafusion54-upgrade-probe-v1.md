# lance9-datafusion54-upgrade-probe-v1 — what breaks on lance 9 / lancedb 0.33 / DataFusion 54 / Rust 1.97.1

> **Status:** MEASURED 2026-08-05. Assessment, not a landed migration — the pins
> and the two code fixes are on `claude/x265-x266-plans-review-h9osnl`; the real
> upgrade is being driven in a parallel session.
> **Method:** variables isolated on operator instruction — toolchain tested ALONE
> on the old pins first, so a toolchain failure could never be mistaken for a
> dependency failure. `target/` cleared between (the rustc-version swap
> invalidates every artifact anyway).

## 1. Version correction — there is no `lancedb` 0.36 Rust crate

The registry tops out at **`lancedb 0.33.0`**, and that release IS the lance-9
pairing (it pins `lance = "=9.0.0"` exactly). **0.36.0 is the PyPI package**,
versioned independently of the Rust crate. So "lance 9 + lancedb 0.36" spells
`lance =9.0.0` + `lancedb =0.33.0` on the Rust side.

## 2. Results

| Leg | Result |
|---|---|
| Rust 1.97.1 on the OLD pins | **clean** — no new language or lint surface. `lance-graph-contract` passes `clippy --all-targets -D warnings`; the only reds anywhere were the 12 **pre-existing** `lance-graph-ontology` lints already on the board (oxrdf deprecated alias, doc-list indentation) |
| Resolution, lance 9 | **succeeds** — no conflict, no unused-patch alert |
| `cargo check -p lance-graph` | **green** (after 2 fixes below) |
| `-p lance-graph-callcenter --features query` | **green** (compiles the 5 `vsa_udfs` UDFs + `policy`) |
| `-p lance-graph-callcenter --features query-lite` | **green** (compiles `transcode/ontology_table.rs`) |

**arrow and `object_store` do not move** — `arrow 58.3.0`, `object_store 0.13.2`,
unchanged. Historically the two that force painful cascades; they sit still.
Only DataFusion crosses a major (53 → 54).

## 3. The whole break surface: `as_any` → `Any` supertrait, 12 sites, 3 traits

DataFusion 54 moved `Any` from a **method** to a **supertrait** on three traits
at once, so every explicit `as_any` impl is `E0407`. Verified per trait against
the 54.1.0 sources — not assumed:

| trait | 53 | 54 |
|---|---|---|
| `TableSource` | `: Sync + Send` + `fn as_any` | `: Any + Sync + Send`, no method |
| `TableProvider` | `: Debug + Sync + Send` + `fn as_any` (`table.rs:54`) | `: Any + Debug + Sync + Send`, no method |
| `ScalarUDFImpl` | `: Debug + DynEq + DynHash + Send + Sync` + `fn as_any` | `: … + Any`, no method |

Downcasting now goes through the blanket `Any` impl; **no call site changed.**
`TableProvider` is NOT untouched by this — a parallel audit read the break as
`TableSource`-only, and `OntologyTableProvider` would have failed the moment
`query-lite` was enabled.

## 4. The `DynEq`/`DynHash` non-break — and the reasoning error that predicted it

Reading 54's `ScalarUDFImpl` supertrait list (`Debug + DynEq + DynHash + Send +
Sync + Any`), with `impl<T: Eq + Any> DynEq for T` / `impl<T: Hash + Any>
DynHash for T` blanket impls, I predicted a second migration: all ~11 UDF types
must gain `Eq + Hash`, and the ones holding `Arc<dyn Fn>` can't derive it. **That
prediction was wrong, and it is recorded here because the error is reusable:**

**DataFusion 53 ALREADY required `DynEq + DynHash`** —
`datafusion-expr-53.1.0/src/udf.rs:498` is
`pub trait ScalarUDFImpl: Debug + DynEq + DynHash + Send + Sync`. The
53 → 54 delta on that trait is EXACTLY `+ Any` / `− fn as_any`. Nothing about
Eq/Hash moved. The codebase already satisfied it — all 10 in-tree UDFs carry
hand-written `PartialEq`/`Eq`/`Hash` keyed on `name()` — **because it had to, to
compile against 53 in the first place.**

**The error:** I read the NEW version's trait bound and inferred "new
requirement" without diffing the OLD one. The differential was one grep away.
Ruling: `E-DIFF-THE-VERSIONS-DONT-READ-THE-NEW-ONE-1`.

## 5. Two code fixes (the entire migration cost, so far)

1. **`as_any` sweep** — 12 deletions across `datafusion_planner/udf.rs` (3),
   `cam_pq/udf.rs`, `nsm/nsm_word.rs`, `callcenter/vsa_udfs.rs` (5),
   `callcenter/policy.rs`, `callcenter/transcode/ontology_table.rs`,
   `catalog/source_catalog.rs`. Mechanical; one comment left at the
   `TableSource` site naming the cause.
2. **Invariant notes on 4 hand-written `PartialEq` impls** (no behaviour change).
   The `name()`-keyed equality is CORRECT for DataFusion — a UDF's identity in
   expression comparison and CSE is its registered name, not its payload — but
   at three sites it makes an *assertion* rather than restating a fact, and that
   was invisible:
   - `VectorDistanceUDF` / `VectorSimilarityUDF` — `func` is derived from the
     compared `metric` at every construction site, so
     `(name, metric)`-equality ⇒ function-equality **by construction**. Safe;
     now stated.
   - `CamDistanceUDF` — `func` captures a caller-supplied **codebook** under the
     fixed name `"cam_distance"`. Nothing in the type enforces
     same-name ⇒ same-codebook: two registrations with different codebooks
     compare equal for CSE while computing different distances. Load-bearing
     assumption, now documented with the "one codebook per session context"
     rule.
   - `NsmSimilarityUdf` — same shape with `Arc<NsmRuntime>`.

## 6. The duplicate-major cost, and where it actually lives

After bumping the two workspace `datafusion` pins (`lance-graph/Cargo.toml:21`,
`lance-graph-catalog/Cargo.toml:16`) to 54, **every one of the 20 `lance-*`
crates resolves to a single major, 9.0.0** — no lance 6/7/9 triple, no two
Lance stacks compiled. A parallel audit saw three lance majors; that state is
pre-pin-bump, and the pin bump is what collapses it.

`datafusion 53.1.0` does remain in the lock, from **exactly one source**:
`deltalake-core 0.32.4`, reached only through the **non-default `delta`
feature** — already quarantined out of `default` with a standing
`TODO(lance-bump-delta)` (deltalake 0.32 removed `DeltaTableProvider::try_new`).
So the duplicate DataFusion major is confined to an already-broken opt-in
feature and costs a default build nothing.

## 7. Two failures that will look like Lance fallout and are not

- **`protoc` is a hard prerequisite** — `lance-encoding`'s build script fails
  without it.
- **A `--all-targets` example fails to resolve `lance_graph_ontology` / `oxrdf`
  / `oxrdfxml` / `oxttl`** — a pre-existing feature-gating bug in the example,
  unrelated to either bump.

## 8. Landed (updated after the operator ruled "bump all now, fix after")

The probe became the bump. Nine repos' toolchains moved 1.95 → 1.97.1
(rust-toolchain.toml + CI pins + Dockerfiles, together — a pin that CI
overrides is not a pin); `q2` was deliberately LEFT on its `nightly-2026-04-28`
(an external project whose nightly pin likely carries nightly features; breaking
it was not in scope). Merged: lance-graph #891-#895, OGAR #243/#244/#245,
ruff #93, stockfish-rs #14, woa-rs #179, a2ui-rs #19, MedCare-rs #351.

**Docker drift found while in there (pre-existing, unrelated to either bump):**
lance-graph's and ndarray's Dockerfiles installed **1.94.0** — a version BEHIND
their own 1.95.0 `rust-toolchain.toml`, so the images had been building on an
older toolchain than the workspace declared. Both now on 1.97.1. `ruff`'s
Dockerfile is the drift-proof pattern and needed no edit: it COPYs
`rust-toolchain.toml` and installs `--default-toolchain none`, so it cannot
disagree with the pin. Worth adopting in the others.

**Clippy fix-after, all green under 1.97.1:** `lance-graph`, `-catalog`,
`-contract`, `-ontology`, `-planner`, `-supervisor`, `cognitive-shader-driver`,
`-callcenter` (both `query` and `query-lite`). **The default-feature pass was a
FALSE GREEN** — `vsa_udfs` and `transcode/ontology_table` are feature-gated, so
the first run never compiled the very files carrying the `as_any` impls. Only
explicit impls break, and only when their module is actually compiled; that is
exactly how a gated module can look fine. ndarray: 6 mechanical lints fixed,
**2186 lib tests pass, 0 fail**.

The 12 pre-existing `lance-graph-ontology` lints are gone (5 oxrdf deprecations
+ 6 doc-indentation + 1 in a paired file), which is what made
`rust-toolchain.toml`'s own precondition — "bump when workspace clippy passes
clean" — satisfiable rather than aspirational. Note the bump ran AHEAD of that
gate on the operator's explicit call; the fix-after discharged it.

## 9. Still open

- OGAR: 22 crates unreached by the lint sweep + one unverified
  `clickhouse-ddl` fix (resume worker dispatched).
- `blockly-rs` and `rig` still pin 1.95.0 — outside the sweep.
- Unchecked lance-dependent crates: `-benches`, `symbiont`, `cognitive-stack`,
  `surreal_container`, `-python`. No workspace-scope clippy (per the `-p`-only
  rule). The `delta` feature stays broken by deltalake 0.32's own API drift
  (`TODO(lance-bump-delta)`), which is what quarantines the duplicate
  DataFusion major.
