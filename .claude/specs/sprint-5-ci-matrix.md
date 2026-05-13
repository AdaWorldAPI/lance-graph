# Sprint-5 CI Matrix and Green-Gate Criteria

> **Spec-ID:** S5-W11
> **Author:** W4 (claude-sonnet-4-6), sprint-log-5-6, 2026-05-13
> **Deliverable type:** Spec-only (no code committed; engineer configures CI)
> **Status:** Draft — ready for engineer pickup
> **Cross-refs:**
> - `.github/workflows/build.yml` (existing linux-build gate)
> - `.github/workflows/rust-test.yml` (existing test + test-with-coverage)
> - `.github/workflows/style.yml` (existing format + clippy)
> - `.github/workflows/jc-proof.yml` (existing substrate proof, jc crate)
> - `.claude/specs/sprint-6-conformance-test.md` (W12 sibling — §6 CI integration)
> - `.claude/board/LATEST_STATE.md` PR #364 row — commit a3c753f (ndarray hpc-extras blake3 opt-in)
> - ndarray#142 — VBMI gate for `permute_bytes` (P0 SIGILL fix)

---

## 1 — Purpose

PR #364 merged 2026-05-13 with all 5 CI checks green:
`format`, `clippy`, `linux-build (stable)`, `test (stable)`, `test-with-coverage`.

Sprint-5 follow-on PRs (PR-D3a, PR-D3b, PR-D4) and the sprint-6 cascade
(PR-E1/E2/E3, PR-F1, PR-G1/G2) must land against a CI matrix that is:

1. **Repeatable** — identical pass criteria for any engineer or agent spawning a PR.
2. **Hardware-safe** — ndarray#142 P0 SIGILL (non-VBMI AVX-512 `permute_bytes`) gates
   which runner features can be exercised.
3. **Coverage-gated** — no silent coverage regression between PRs.
4. **Consumer-conformance-aware** — sprint-6 W10 conformance tests (W12 spec) are a
   blocking step for E-series PRs.
5. **Time-budgeted** — total wall time per PR stays under 20 minutes.

This spec defines the authoritative green-gate table, the target matrix, hardware
runner constraints, coverage thresholds, time budgets, and the delta versus existing
`.github/workflows/`.

---

## 2 — Per-PR Green-Gate Table

### 2.1 Blocking checks (PR cannot merge unless all pass)

| Check ID | Workflow file | Job name | Fail mode | Applies to |
|---|---|---|---|---|
| **GG-1** | `style.yml` | `clippy` (Tier A: contract crate) | Hard fail `-D warnings` | All PRs touching `crates/**` |
| **GG-2** | `style.yml` | `format` | Hard fail `-- --check` | All PRs touching `crates/**` |
| **GG-3** | `build.yml` | `linux-build (stable)` | Hard fail | All PRs touching `crates/**` |
| **GG-4** | `rust-test.yml` | `test (stable)` — lib + doc + contract | Hard fail | All PRs touching `crates/**` |
| **GG-5** | `rust-test.yml` | `test-with-coverage` | Hard fail if build broken; advisory if only coverage drops (see §4) | All PRs |
| **GG-6** | `rust-test.yml` | `consumer-conformance` (NEW — see §6) | Hard fail | PR-E1, PR-E2, PR-E3 and any PR touching `unified_bridge.rs` or `UnifiedAuditEvent` |

### 2.2 Advisory checks (informational; do not block merge)

| Check ID | Workflow file | Job name | Notes |
|---|---|---|---|
| **ADV-1** | `style.yml` | `clippy` (Tier B: lance-graph core) | `continue-on-error: true` — ~91 pre-existing violations; pay down per TD-CLIPPY-LG-1 |
| **ADV-2** | `rust-test.yml` | `test-with-coverage` coverage delta | Advisory until threshold baseline established (see §4) |
| **ADV-3** | `jc-proof.yml` | `prove` | Informational substrate proof; only blocking if `crates/jc/**` or `contract::cam` changed |

### 2.3 Conditional activation

| Trigger condition | Additional blocking check activated |
|---|---|
| PR touches `crates/lance-graph-consumer-conformance/**` | GG-6 (conformance gate) |
| PR touches `crates/lance-graph-callcenter/src/unified_bridge.rs` | GG-6 |
| PR touches `crates/lance-graph-callcenter/src/unified_audit.rs` | GG-6 |
| PR touches `crates/jc/**` or `crates/lance-graph-contract/src/cam.rs` | ADV-3 promotes to blocking |
| PR is PR-D3a (LanceAuditSink) | GG-4 must include `--features lance-sink` on `lance-graph-callcenter` |
| PR is PR-D3b (JSONL verify) | GG-4 must include `--features jsonl` on `lance-graph-callcenter` |

---

## 3 — Target Matrix (OS × Toolchain × Features)

### 3.1 Current matrix (post-#364 baseline)

| Dimension | Value | File |
|---|---|---|
| OS | `ubuntu-24.04` (GitHub-hosted) | all workflows |
| Toolchain | `stable` only | build.yml, rust-test.yml |
| Features (default) | `unity-catalog`, `delta`, `ndarray-hpc` | CLAUDE.md Cargo.toml excerpt |
| RUSTFLAGS | `-C debuginfo=1 -C target-cpu=x86-64-v3` | rust-test.yml, build.yml env |
| ndarray checkout | `AdaWorldAPI/ndarray` master (pin retired post-PR#115) | rust-test.yml |

### 3.2 Feature combinations to exercise per PR

The matrix must run these combinations for every PR touching `crates/lance-graph`:

| Combination ID | Cargo flags | Purpose |
|---|---|---|
| **FC-1** | *(default: ndarray-hpc + unity-catalog + delta)* | Happy path — all features |
| **FC-2** | `--no-default-features` | Fallback mode: no ndarray, no delta, no unity-catalog. Catches implicit ndarray-hpc-only code paths. |
| **FC-3** | `--features ndarray-hpc --no-default-features` | ndarray-hpc in isolation — catches feature-gated crate imports that assume other features |
| **FC-4** | `--features lance-cache` (on `lance-graph-ontology` only) | Exercises `LanceWriter` path; requires `protoc`. Only runs if PR touches `lance-graph-ontology`. |

### 3.3 E-series consumer PRs (sprint-6) — additional combinations

| Combination ID | Crate | Cargo flags | Triggered by |
|---|---|---|---|
| **FC-E1** | `medcare-rs` | `--features consumer-conformance` | PR-E1 |
| **FC-E2** | `smb-office-rs` | `--features consumer-conformance` | PR-E2 |
| **FC-E3** | `woa-rs` | `--features consumer-conformance` | PR-E3 |
| **FC-CC** | `lance-graph-consumer-conformance` (new crate, W12 spec) | `--lib --tests` | Any of PR-E1/E2/E3 |

### 3.4 Toolchain expansion (sprint-6 plan, NOT yet in existing workflows)

The current matrix runs `stable` only. For sprint-6 cascade:

- **beta** toolchain: add as advisory job (catch regressions before stable promotion).
  - **File change required:** `build.yml` matrix `toolchain` array → `[stable, beta]`; add
    `continue-on-error: true` on the `beta` matrix entry only.
- **MSRV** (minimum supported Rust version): deferred — no MSRV policy set; add in sprint-7
  governance PR only after the MSRV is declared in `Cargo.toml`.
- **macOS / Windows:** deferred to sprint-7. Cross-platform concerns are out of scope for
  sprint-5/6 follow-ons; the callcenter + audit crates have no platform-specific code paths.

---

## 4 — Coverage Threshold and Regression Detection

### 4.1 Current coverage setup

`test-with-coverage` uses `cargo-llvm-cov` (lcov output) uploaded to Codecov with
`fail_ci_if_error: false`. Coverage failure (Codecov unreachable, token absent) is
**non-blocking** today.

### 4.2 Threshold policy for sprint-5/6 follow-ons

| Crate | Baseline (post-#364) | Hard floor | Regression action |
|---|---|---|---|
| `lance-graph-contract` | ~100% (97/97 callcenter lib tests pass, full contract suite) | **85% line** | Block PR if below floor |
| `lance-graph` (core) | uncalibrated | **60% line** | Advisory only until calibrated; set hard floor in sprint-7 |
| `lance-graph-callcenter` | uncalibrated (D-SDR-3/4/5 added ~1000 LOC) | **70% line** after PR-D3a/b | Advisory until D3b merges |
| `lance-graph-consumer-conformance` (new, W12) | N/A (new crate) | **90% line** from first PR | Hard from creation; conformance crate must have high coverage by design |

### 4.3 Regression detection mechanism

Because Codecov does not block merges (`fail_ci_if_error: false`), coverage regression
is currently invisible unless manually checked. For sprint-5/6:

1. **PR-D3a** must add a `coverage-gate` job to `rust-test.yml` that runs
   `cargo llvm-cov --fail-under-lines 85 --manifest-path crates/lance-graph-contract/Cargo.toml`.
   This is a **1-line addition** to the existing coverage job — no new job needed.
2. **PR-E1** (sprint-6 first E-series PR) must extend the gate to include
   `lance-graph-consumer-conformance` at 90%.
3. No PR may lower coverage on `lance-graph-contract` below 85% without a
   justified `#[cfg(test)] #[allow(dead_code)]` annotation and a board entry
   in `TECH_DEBT.md`.

---

## 5 — Hardware Concerns: ndarray#142 P0 SIGILL

### 5.1 Background

ndarray#142 (merged 2026-05-13) ships a VBMI (Vector Byte Manipulation Instructions)
runtime gate for `permute_bytes`. Without VBMI, calling `permute_bytes` on an
AVX-512 host that lacks the `vbmi` sub-extension causes a SIGILL:

- **Affected CPUs:** Skylake-X, Cascade Lake, Ice Lake-SP (AVX-512F without VBMI).
- **Safe CPUs:** Ice Lake client, Tiger Lake, Sapphire Rapids, Alder Lake, Raptor Lake
  (all have VBMI). GitHub-hosted `ubuntu-24.04` runners use Intel CPUs in the
  `cascade-lake` / `icelake-server` family — **VBMI is NOT guaranteed**.

### 5.2 Mitigation in current CI

The current `RUSTFLAGS = "-C target-cpu=x86-64-v3"` compiles to x86-64-v3 baseline
(AVX2, no AVX-512). This means the `permute_bytes` SIMD path is **not compiled in** by
default and the SIGILL cannot fire. This is the correct and safe current state.

### 5.3 Rules for sprint-5/6 PRs

| Rule | Enforcement |
|---|---|
| **R-HW-1** Do NOT change `RUSTFLAGS` to `-C target-cpu=native` or any AVX-512 target in CI | Reviewer must reject any PR that touches `env.RUSTFLAGS` to add AVX-512 targets |
| **R-HW-2** Do NOT add a CI job that runs `--features ndarray-hpc` with `RUSTFLAGS=-C target-cpu=skylake-avx512` | Explicitly disallowed — SIGILL risk on GitHub hosted runners |
| **R-HW-3** If a PR needs to benchmark AVX-512 paths, it must use a self-hosted runner tagged `avx512-vbmi` with a verified VBMI-capable CPU | Runner label: `runs-on: [self-hosted, avx512-vbmi]` |
| **R-HW-4** The `ndarray-hpc` feature flag must be tested under FC-1 (default, x86-64-v3) and FC-3 (ndarray-hpc isolated, x86-64-v3); neither uses AVX-512 | Enforced by RUSTFLAGS env var |

### 5.4 blake3 hpc-extras note (commit a3c753f)

Commit a3c753f in PR #364 adds ndarray `hpc-extras` as an opt-in feature for `blake3`
hashing within the ndarray callstack. This feature is **not default** and is not
activated by `ndarray-hpc` in lance-graph. CI runs without `hpc-extras`. If a future
PR enables `hpc-extras` in lance-graph, R-HW-1 through R-HW-3 apply and the feature
must be gated behind a `lance-graph/Cargo.toml` opt-in feature named `ndarray-hpc-extras`.

---

## 6 — Consumer-Conformance Gate (W12 alignment)

W12's spec (`.claude/specs/sprint-6-conformance-test.md`) defines assertions A1-A10
for `UnifiedBridge<B>` consumers. This section defines how those assertions integrate
into the CI matrix as a **blocking step** (GG-6).

### 6.1 New workflow job: `consumer-conformance`

**File:** `rust-test.yml` (extend existing file — no new workflow file)

```yaml
  consumer-conformance:
    runs-on: ubuntu-24.04
    timeout-minutes: 15
    if: |
      github.event_name == 'push' ||
      contains(github.event.pull_request.labels.*.name, 'e-series') ||
      contains(toJson(github.event.pull_request.files.*.filename), 'unified_bridge') ||
      contains(toJson(github.event.pull_request.files.*.filename), 'unified_audit') ||
      contains(toJson(github.event.pull_request.files.*.filename), 'consumer-conformance')
    defaults:
      run:
        working-directory: lance-graph
    steps:
      - uses: actions/checkout@v4
        with:
          path: lance-graph
      - name: Checkout AdaWorldAPI/ndarray
        uses: actions/checkout@v4
        with:
          repository: AdaWorldAPI/ndarray
          path: ndarray
      - name: Setup rust toolchain
        run: |
          rustup toolchain install stable
          rustup default stable
      - uses: Swatinem/rust-cache@v2
        with:
          shared-key: "lance-graph-deps"
          workspaces: lance-graph/crates/lance-graph
      - name: Install dependencies
        run: |
          sudo apt update
          sudo apt install -y protobuf-compiler
      - name: Run consumer conformance tests
        run: |
          cargo test \
            --manifest-path crates/lance-graph-consumer-conformance/Cargo.toml \
            --features consumer-conformance \
            --lib --tests \
            -- --test-threads=1
```

### 6.2 Blocking criteria

The `consumer-conformance` job is **hard-failing** (no `continue-on-error`). A PR
that breaks any of A1-A10 for any E1/E2/E3 consumer cannot merge.

### 6.3 Test thread count

`--test-threads=1` is required because `RecordingSink` uses a `Mutex<Vec<...>>` and
conformance assertions are sequential (A3 requires ordered merkle chain). Parallel
test threads would interleave events and produce false failures.

### 6.4 Relationship to W12's harness

W12 defines the `assert_consumer_conformance` generic function in
`crates/lance-graph-consumer-conformance/src/harness.rs`. GG-6 executes that
harness for each of the three active consumers (E1 MedcareBridge, E2 OgitBridge,
E3 WoaBridge). The job is owned by the `lance-graph-consumer-conformance` crate —
it is not a separate binary or integration-test crate.

---

## 7 — Audit-Sink Integration Tests (W1 + W2 sink-running jobs)

PR-D3a (W1, LanceAuditSink) and PR-D3b (W2, JSONL verify) each ship sink-running
jobs that emit real audit events and verify round-trip. These are **integration tests**
(not unit tests) and must run under a separate job to avoid polluting unit-test output.

### 7.1 New job: `audit-sink-integration`

**File:** `rust-test.yml` (extend — no new file)

| Property | Value |
|---|---|
| Activation | Triggered when PR touches `crates/lance-graph-callcenter/src/audit*` or `crates/lance-graph-callcenter/src/lance_sink*` |
| Runner | `ubuntu-24.04` |
| Timeout | 20 minutes |
| Cargo flags | `--features lance-sink,jsonl --test audit_sink_integration` |
| Blocking | Hard-fail; no `continue-on-error` |

### 7.2 Feature flag discipline

| Feature | Crate | Purpose | Default? |
|---|---|---|---|
| `lance-sink` | `lance-graph-callcenter` | Enables `LanceAuditSink` (requires `protoc`) | No |
| `jsonl` | `lance-graph-callcenter` | Enables JSONL serialization for `UnifiedAuditEvent` | No |
| `consumer-conformance` | `lance-graph-consumer-conformance` | Enables conformance harness (no protoc) | No |

These features must NOT be added to `default-features` of any workspace member —
they are explicitly opt-in to preserve the zero-protoc compile path.

---

## 8 — Time Budget Per Job and Parallelism

### 8.1 Current timeout budgets (post-#364)

| Job | File | Timeout | Actual ~time |
|---|---|---|---|
| `linux-build` | build.yml | 30 min | ~8 min (cached) |
| `test (stable)` | rust-test.yml | 30 min | ~10 min (cached) |
| `test-with-coverage` | rust-test.yml | 30 min | ~15 min (cached) |
| `clippy` | style.yml | 25 min | ~6 min |
| `format` | style.yml | 15 min | ~3 min |
| `prove` | jc-proof.yml | 5 min | ~2 min |

### 8.2 New job time budgets (sprint-5/6)

| Job | File | Proposed timeout | Rationale |
|---|---|---|---|
| `consumer-conformance` | rust-test.yml | 15 min | New crate, small; 3 consumers × A1-A10 assertions ~1-2 min each |
| `audit-sink-integration` | rust-test.yml | 20 min | Lance append I/O + tmpdir teardown |
| `linux-build (beta)` | build.yml | 30 min | Same as stable; advisory |

### 8.3 Parallelism strategy

Jobs that share the Swatinem cache key `lance-graph-deps` benefit from warm cache
on the second job forward within a PR run. The recommended job ordering for total
wall time minimisation:

```
[parallel group A]           [parallel group B — after A]
  clippy                       linux-build (stable)
  format                       test (stable)
                               test-with-coverage

[serial, after group B]
  consumer-conformance (if activated — depends on test artifacts)
  audit-sink-integration (if activated)
```

`consumer-conformance` runs after group B because it exercises the full
callcenter + ontology + conformance crate stack, which benefits from the
warm cache populated by group B.

### 8.4 Total PR wall time target

With the above parallelism and warm cache: **≤ 18 minutes** for a standard PR
(groups A + B in parallel). PRs that activate `consumer-conformance` and
`audit-sink-integration`: **≤ 22 minutes** (within the 25-minute GitHub
Actions concurrency cost budget).

---

## 9 — Delta vs Existing `.github/workflows/`

### 9.1 Files that CHANGE (modifications to existing files)

| File | Section changed | What changes |
|---|---|---|
| `rust-test.yml` | `jobs:` block | Add `consumer-conformance` job (§6.1); add `audit-sink-integration` job (§7.1); add coverage `--fail-under-lines` flag to `test-with-coverage` step (§4.3) |
| `build.yml` | `strategy.matrix.toolchain` | Add `beta` entry with `continue-on-error: true` (§3.4) — sprint-6 only, not sprint-5 |

### 9.2 Files that stay UNCHANGED

| File | Reason |
|---|---|
| `style.yml` | Tier A/B clippy split is correct as-is; no changes needed for sprint-5/6 |
| `jc-proof.yml` | Self-contained substrate proof; no sprint-5/6 changes |
| `release.yml` | Release workflow; out of scope |
| `rust-publish.yml` | Publish workflow; out of scope |

### 9.3 Files that are NEW (do not exist yet)

None. All new CI jobs are added to `rust-test.yml` and `build.yml` rather than
creating new workflow files. This minimises the number of required status checks
visible in the GitHub branch protection rule.

### 9.4 Branch protection rule update required

When GG-6 (`consumer-conformance`) is added, the GitHub branch protection rule for
`main` must add `consumer-conformance` as a required status check. This is a
GitHub UI / API change, not a file change. The engineer enabling GG-6 must update
the branch protection rule in the repository settings after the first successful
run of the new job.

---

## 10 — Sprint-5 Follow-on PR Checklist (PR-D3a, PR-D3b, PR-D4)

Each PR in the sprint-5 follow-on batch must satisfy the following before merge:

### PR-D3a (LanceAuditSink)

- [ ] GG-1 through GG-5 all green
- [ ] `audit-sink-integration` job added to `rust-test.yml` (§7.1) and green
- [ ] Coverage floor for `lance-graph-callcenter` set to 70% in coverage job (§4.2)
- [ ] `lance-sink` feature is NOT in default features of any workspace member
- [ ] ADV-1 (clippy Tier B) checked manually; any new violations documented in TECH_DEBT.md

### PR-D3b (JSONL verify)

- [ ] GG-1 through GG-5 green
- [ ] `--features jsonl` exercises JSONL serialization path in integration test
- [ ] No coverage regression on `lance-graph-callcenter` below 70%

### PR-D4 (family hydration)

- [ ] GG-1 through GG-5 green
- [ ] FC-1 and FC-2 both pass (family hydration must not require ndarray-hpc)
- [ ] OgitFamilyTable sparse `HashMap<u16, FamilyEntry>` path covered by unit tests

---

## 11 — Sprint-6 Cascade PR Checklist (PR-E1/E2/E3, PR-F1, PR-G1/G2)

### PR-E1 (medcare-rs finalisation)

- [ ] GG-1 through GG-6 all green (GG-6 = consumer-conformance, A1-A10 for MedcareBridge)
- [ ] FC-E1 (`--features consumer-conformance` on medcare-rs) passes
- [ ] Coverage floor for `lance-graph-consumer-conformance` at 90% (§4.2)
- [ ] A2: super_domain == SuperDomain::Healthcare on all emitted events
- [ ] Branch protection updated to require `consumer-conformance` status check

### PR-E2 (smb-office-rs retrofit)

- [ ] GG-1 through GG-6 green
- [ ] FC-E2 passes; OgitBridge resolves "WorkOrder" alias to canonical OGIT name (A5)
- [ ] A2: super_domain == SuperDomain::WorkOrderBilling (discriminant 6)

### PR-E3 (woa-rs extraction)

- [ ] GG-1 through GG-6 green
- [ ] FC-E3 passes; WoaBridge g_lock returns non-zero NamespaceId (A10)
- [ ] A8: TenantId isolation verified across WoaBridge and MedcareBridge instances

### PR-F1 (thinking engine wire)

- [ ] GG-1 through GG-5 green (GG-6 not triggered — F1 does not touch unified_bridge)
- [ ] FC-1 and FC-3 both pass (thinking engine must not break ndarray-hpc isolation)
- [ ] `jc-proof.yml` prove job passes (F1 may touch contract::cam indirectly via planner)

### PR-G1 (manifest modules)

- [ ] GG-1 through GG-5 green
- [ ] FC-1 and FC-2 both pass (manifest modules must compile without ndarray-hpc)

### PR-G2 (ractor supervisor)

- [ ] GG-1 through GG-5 green
- [ ] Ractor actor-bind integration test passes under `test (stable)` (§2.1 GG-4)
- [ ] ADV-1 clippy Tier B does not acquire new ractor-related violations without documented rationale

---

## 12 — Summary Table: All Gate IDs

| Gate ID | Type | Workflow | Job | Sprint-5 D-series | Sprint-6 E-series | Sprint-6 F/G series |
|---|---|---|---|---|---|---|
| GG-1 | Blocking | style.yml | clippy (contract) | Required | Required | Required |
| GG-2 | Blocking | style.yml | format | Required | Required | Required |
| GG-3 | Blocking | build.yml | linux-build (stable) | Required | Required | Required |
| GG-4 | Blocking | rust-test.yml | test (stable) | Required | Required | Required |
| GG-5 | Blocking | rust-test.yml | test-with-coverage | Required | Required | Required |
| GG-6 | Blocking (conditional) | rust-test.yml | consumer-conformance | Not applicable | Required for E1/E2/E3 | Not applicable |
| ADV-1 | Advisory | style.yml | clippy (lance-graph core) | Monitor | Monitor | Monitor |
| ADV-2 | Advisory | rust-test.yml | coverage delta | Monitor | Monitor | Monitor |
| ADV-3 | Advisory→Blocking | jc-proof.yml | prove | Monitor | Monitor | Blocking if cam.rs touched |

---

*Spec complete. Engineer pickup: start with §9.1 (delta to rust-test.yml) then §4.3 (coverage floor addition), then §6.1 (consumer-conformance job). Align with W12's harness at `.claude/specs/sprint-6-conformance-test.md` §3 before writing the new job YAML.*
