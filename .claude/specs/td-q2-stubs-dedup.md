# TD-Q2-STUBS-DEDUP-1 — Implementation Spec

> **Priority:** P0
> **Sprint:** sprint-log-4 (2026-05-13)
> **Owner:** W2
> **Blocking:** FMA heart-click demo (W11), q2 graph-notebook compile
> **Branch:** `claude/lance-datafusion-integration-gv0BF` (lance-graph side)
> **q2 HEAD SHA:** `dfe3df477282030b48312b74dc573fddd3660e0b`

---

## Problem Statement

q2 (`AdaWorldAPI/q2`) carries three local stub crates under `crates/stubs/` that
shadow or mock types from canonical upstream crates. Before the FMA demo can
compile end-to-end, those stubs must be replaced with re-exports from the real
crates. The key stubs are:

| Stub crate | Path in q2 | Shadows |
|---|---|---|
| `q2-ndarray` | `crates/stubs/q2-ndarray/` | `ndarray` (AdaWorldAPI fork at `/home/user/ndarray`) |
| `graph-flow` | `crates/stubs/graph-flow/` | `rs-graph-llm` / LangGraph orchestrator (not locally available) |
| `notebook-query` | `crates/stubs/notebook-query/` | `lance-graph` + `lance-graph-planner` query surface |

The workspace `Cargo.toml` already declares the canonical deps (confirmed HEAD):

```toml
[workspace.dependencies.lance-graph]
path = "../lance-graph/crates/lance-graph"
features = ["ndarray-hpc"]

[workspace.dependencies.lance-graph-contract]
path = "../lance-graph/crates/lance-graph-contract"

[workspace.dependencies.q2-ndarray]
path = "./crates/stubs/q2-ndarray"   # <- THIS IS THE STUB, needs replacement
```

The problem: `q2-ndarray` stub is wired as the live dep in the workspace, and
`notebook-query` stub re-exports mock types instead of forwarding to the real
`lance-graph-contract` types. `graph-flow` stub is intentionally flagged
("Uses stub when rs-graph-llm repo is not available locally") and is lower
priority for FMA demo.

---

## 1. Inventory Step

### Files to locate and audit

**Execute these path inspections on q2 repo (AdaWorldAPI/q2 HEAD):**

```
crates/stubs/Cargo.toml              # stub workspace manifest (confirmed exists)
crates/stubs/q2-ndarray/Cargo.toml  # q2-ndarray stub Cargo.toml
crates/stubs/q2-ndarray/src/lib.rs  # stub exports (mock types to enumerate)
crates/stubs/graph-flow/Cargo.toml  # graph-flow stub Cargo.toml
crates/stubs/graph-flow/src/lib.rs  # graph-flow stub exports
crates/stubs/notebook-query/Cargo.toml
crates/stubs/notebook-query/src/lib.rs   # notebook-query stub exports
crates/stubs/lib.rs                  # top-level stub re-export shim (258 bytes)
```

**NOTE:** Direct file inspection via MCP read budget was used on directory
listings and workspace Cargo.toml. The stub `src/lib.rs` files were not read
(budget exhausted at 3 reads). An engineer MUST audit the stub `src/lib.rs`
files before cutover to ensure no q2-internal types are secretly defined there
(vs. just re-exported).

**What to look for in each stub `src/lib.rs`:**

- `pub struct` / `pub type` / `pub trait` definitions -- these are local mock
  types that need a canonical equivalent.
- `pub use` statements pointing to nothing real -- these are the re-export
  holes to fill.
- `compile_error!` / `todo!()` / `unimplemented!()` -- signals actively broken surface.

**Confirmed from workspace Cargo.toml (q2 HEAD `dfe3df`):**

The workspace `[workspace.dependencies.notebook-query]` points to:
```
path = "./crates/stubs/notebook-query"
```

And `[workspace.dependencies.q2-ndarray]` points to:
```
path = "./crates/stubs/q2-ndarray"
```

These are the two P0 stubs for FMA demo. `graph-flow` is P1 (blocked on
rs-graph-llm repo availability).

---

## 2. Migration Recipe

### 2a. `q2-ndarray` stub -> re-export from canonical ndarray

The canonical ndarray is at `/home/user/ndarray` (AdaWorldAPI fork). q2 already
depends on `lance-graph` with `features = ["ndarray-hpc"]` which transitively
pulls the ndarray fork. The stub just needs to become a thin re-export crate.

**`crates/stubs/q2-ndarray/Cargo.toml` -- replace with:**

```toml
[package]
name = "q2-ndarray"
version = "0.1.0"
edition = "2024"

[dependencies]
# Point at the same canonical ndarray the lance-graph workspace uses.
# Adjust relative path based on q2's directory depth vs. lance-graph.
ndarray = { path = "../../../../ndarray", optional = false }
lance-graph-contract = { workspace = true }

[features]
default = ["canonical"]
canonical = []   # presence of this feature = stub was replaced
```

**`crates/stubs/q2-ndarray/src/lib.rs` -- replace content with:**

```rust
//! q2-ndarray -- re-export shim. Do NOT define types here.
//! All consumers must import from `::ndarray` or `lance_graph_contract`.
#![doc = "Canonical re-export. Stub deleted per TD-Q2-STUBS-DEDUP-1."]

// Re-export the ndarray prelude so existing `use q2_ndarray::*` imports
// continue to compile during the transition window.
pub use ndarray::*;

// VSA carrier type -- canonical definition in lance-graph-contract.
pub use lance_graph_contract::crystal::fingerprint::CrystalFingerprint;
pub use lance_graph_contract::exploration::NarsTruth;
pub use lance_graph_contract::nars::InferenceType;
pub use lance_graph_contract::orchestration::OrchestrationBridge;
```

### 2b. `notebook-query` stub -> re-export from lance-graph-contract + lance-graph

**`crates/stubs/notebook-query/Cargo.toml` -- add dependencies:**

```toml
[dependencies]
lance-graph-contract = { workspace = true }
lance-graph = { workspace = true }
lance-graph-planner = { workspace = true }
```

**`crates/stubs/notebook-query/src/lib.rs` -- replace stub bodies with:**

```rust
//! notebook-query -- canonical re-export shim for the FMA demo.
//! Stub deleted per TD-Q2-STUBS-DEDUP-1.

// Contract-level types (zero-dep, safe to always re-export)
pub use lance_graph_contract::exploration::NarsTruth;
pub use lance_graph_contract::nars::InferenceType;
pub use lance_graph_contract::orchestration::OrchestrationBridge;
pub use lance_graph_contract::orchestration::UnifiedStep;
pub use lance_graph_contract::orchestration::BridgeSlot;
pub use lance_graph_contract::orchestration::StepDomain;
pub use lance_graph_contract::crystal::fingerprint::CrystalFingerprint;

// Planner-level types (require lance-graph-planner in dep tree)
pub use lance_graph_planner::api::Planner;

pub mod ogit;
```

**`crates/stubs/notebook-query/src/ogit.rs` -- new file:**

```rust
//! OGIT/OwlIdentity re-exports for the OGIT<->OSINT<->Palantir/Neo4j<->q2 route.
pub use lance_graph_callcenter::unified_bridge::{OgitFamily, OwlIdentity, UnifiedBridge};

/// FamilyId alias -- canonical type is OgitFamily(u8).
/// Any q2 code using `FamilyId` should switch to `OgitFamily`.
pub type FamilyId = OgitFamily;
```

Note: `lance-graph-callcenter` must be added to the workspace deps if not
present. See OQ-4 in Open Questions.

### 2c. Workspace dep block update (q2 `Cargo.toml`)

The workspace `[workspace.dependencies]` keys stay unchanged (paths still point
to the stubs). Only the stub crates' own `Cargo.toml` files change to add real
deps. This preserves backward compat with all `{ workspace = true }` references
in downstream crates.

### 2d. Stub deletion list (after type-eq tests pass -- see section 4)

Files to delete once compile-fail tests pass:

```
crates/stubs/q2-ndarray/src/mock_types.rs   # if exists
crates/stubs/q2-ndarray/src/vsa.rs          # if exists
crates/stubs/notebook-query/src/stub_graph.rs  # if exists
crates/stubs/notebook-query/src/fake_planner.rs  # if exists
```

**Before deleting**, run:
```bash
cargo check --workspace 2>&1 | grep "q2.ndarray\|notebook.query"
cargo test --workspace 2>&1 | grep -E "FAILED|error"
```

---

## 3. Re-export Surface

Minimum types q2 must be able to import after migration. Sources confirmed from
lance-graph-contract HEAD on branch `claude/lance-datafusion-integration-gv0BF`.

| Type | Canonical crate | Module path | Notes |
|---|---|---|---|
| `SpoQuad` | `lance-graph` (core) | `lance_graph::graph::spo::` | NOT found in HEAD grep -- verify existence before re-exporting |
| `OwlIdentity` | `lance-graph-callcenter` | `lance_graph_callcenter::unified_bridge::OwlIdentity` | `u16` newtype, repr(transparent) |
| `OgitFamily` | `lance-graph-callcenter` | `lance_graph_callcenter::unified_bridge::OgitFamily` | `u8` newtype, repr(transparent) |
| `UnifiedBridge` | `lance-graph-callcenter` | `lance_graph_callcenter::unified_bridge::UnifiedBridge` | Struct (not trait); wraps NamespaceBridge + Policy |
| `Vsa16kF32` | `lance-graph-contract` | `lance_graph_contract::crystal::fingerprint::CrystalFingerprint` variant | `Vsa16kF32(Box<[f32; 16_384]>)` variant of CrystalFingerprint enum |
| `NarsTruth` | `lance-graph-contract` | `lance_graph_contract::exploration::NarsTruth` | `{ frequency: f32, confidence: f32 }` |
| `InferenceType` | `lance-graph-contract` | `lance_graph_contract::nars::InferenceType` | Enum, 5 variants |
| `OrchestrationBridge` | `lance-graph-contract` | `lance_graph_contract::orchestration::OrchestrationBridge` | Trait |

**Note on `SpoQuad`:** Grep of lance-graph-contract and lance-graph core found
no `pub struct SpoQuad` definition. It may be a planned type not yet in HEAD,
or lives in `lance-graph::graph::spo` as an alias/type. Engineer must verify
before writing the re-export. If absent, the compile-fail test should target
`CrystalFingerprint` as the SPO-equivalent carrier instead.

**Note on `FamilyId`:** Not found as a distinct type in HEAD; the canonical
equivalent is `OgitFamily(u8)`. Any q2 code using `FamilyId` should alias
to `OgitFamily`.

---

## 4. Compile-Fail Tests

Add to `crates/stubs/notebook-query/tests/type_eq.rs` (new file):

```rust
//! Type-equality assertions: q2 re-exports must resolve to the same
//! concrete type as the canonical crate. Fails at compile time if
//! the stub defines its own type instead of re-exporting the real one.
//!
//! Run: cargo test -p notebook-query --test type_eq

use static_assertions::assert_type_eq_all;

// NarsTruth must be the same type from both import paths
assert_type_eq_all!(
    notebook_query::NarsTruth,
    lance_graph_contract::exploration::NarsTruth
);

// InferenceType enum must be identical
assert_type_eq_all!(
    notebook_query::InferenceType,
    lance_graph_contract::nars::InferenceType
);

// CrystalFingerprint (VSA16k carrier) must be identical
assert_type_eq_all!(
    notebook_query::CrystalFingerprint,
    lance_graph_contract::crystal::fingerprint::CrystalFingerprint
);

// OwlIdentity from q2 re-export must be the callcenter type
assert_type_eq_all!(
    notebook_query::ogit::OwlIdentity,
    lance_graph_callcenter::unified_bridge::OwlIdentity
);

// Trait compatibility: impl OrchestrationBridge must be accepted as
// lance_graph_contract::orchestration::OrchestrationBridge.
// Static dispatch check -- this compiles only if they are the same trait.
fn _bridge_compat_check<B>(_b: B)
where
    B: notebook_query::OrchestrationBridge
     + lance_graph_contract::orchestration::OrchestrationBridge
{}
```

Add to `notebook-query/Cargo.toml`:

```toml
[dev-dependencies]
static-assertions = "1.1"
lance-graph-contract = { workspace = true }
lance-graph-callcenter = { workspace = true }
```

**Run:** `cargo test -p notebook-query --test type_eq`

---

## 5. Cutover Sequence (3 Commits)

### Commit A -- Add canonical deps behind feature flag

**Files changed:**
- `crates/stubs/q2-ndarray/Cargo.toml` -- add `[features] default = ["canonical"]` + real ndarray dep
- `crates/stubs/notebook-query/Cargo.toml` -- add lance-graph-contract + lance-graph-planner deps
- `crates/stubs/notebook-query/Cargo.toml` -- add `static_assertions` dev-dep
- `crates/stubs/notebook-query/tests/type_eq.rs` -- add compile-fail tests (from section 4)

**Gate:** Both stubs still compile with old stub code. Tests in `type_eq.rs`
should FAIL at this point (stub types != canonical types). That failure is
the signal to proceed to Commit B.

**PR description must include these pinned commits:**
```
lance-graph branch: claude/lance-datafusion-integration-gv0BF
lance-graph HEAD:   dfe3df477282030b48312b74dc573fddd3660e0b
ndarray HEAD:       (run: git -C /home/user/ndarray log --oneline -1)
q2 HEAD:            dfe3df477282030b48312b74dc573fddd3660e0b
```

### Commit B -- Flip imports + delete stub type definitions

**Files changed:**
- `crates/stubs/q2-ndarray/src/lib.rs` -- replace stub body with `pub use ndarray::*` + contract re-exports
- `crates/stubs/notebook-query/src/lib.rs` -- replace stub body with canonical re-exports
- `crates/stubs/notebook-query/src/ogit.rs` -- new file with OGIT re-export surface (from section 3)
- Delete any `mock_types.rs`, `fake_planner.rs`, or `stub_graph.rs` files discovered in section 1 audit

**Gate:** `cargo check --workspace` passes. `cargo test -p notebook-query --test type_eq` passes.

### Commit C -- Yank feature flag default, enable CI gate

**Files changed:**
- `crates/stubs/q2-ndarray/Cargo.toml` -- remove the `canonical` feature (migration marker no longer needed)
- `.github/workflows/ci.yml` (or equivalent) -- add step:
  `cargo test -p notebook-query --test type_eq`
- `crates/stubs/Cargo.toml` -- add comment: "All stubs are re-export shims. TD-Q2-STUBS-DEDUP-1 resolved."

**Gate:** CI green. FMA smoke test (W11 spec) can proceed.

---

## 6. Risk: Version-Skew

### Lance-graph vs q2 branch mismatch

q2's workspace Cargo.toml (HEAD `dfe3df`) declares:

```toml
[workspace.dependencies.lance-graph]
path = "../lance-graph/crates/lance-graph"
features = ["ndarray-hpc"]
```

This is a **local path dep** -- it resolves to whatever checkout is at
`../lance-graph` at build time. There is NO pinned commit in q2's lockfile for
lance-graph (path deps are not locked by git SHA in Cargo.lock).

**Risk:** The engineer clones q2 and lance-graph at different commits. The
types in lance-graph's contract crate may not match what q2's stub re-exports
expose.

**Mitigation:**

1. In the PR description (Commit A), pin the exact lance-graph commit (see above).

2. Add a workspace metadata note in q2's Cargo.toml documenting the required
   lance-graph branch:

```toml
[workspace.metadata.required-sibling-checkouts]
lance-graph = "claude/lance-datafusion-integration-gv0BF"
ndarray = "main"  # verify actual branch name from /home/user/ndarray
```

3. The compile-fail tests in section 4 serve as the runtime version-skew
   detector: if the wrong lance-graph is checked out, `assert_type_eq_all!`
   will fail at compile time with a readable message.

### ndarray path resolution

ndarray is at `/home/user/ndarray`. q2 reaches it transitively via lance-graph.
If the ndarray path changes (different developer machine), `q2-ndarray/Cargo.toml`
will fail to resolve. Mitigation: keep `q2-ndarray` as a pure re-export of
whatever `ndarray` resolves transitively rather than adding a second direct
path dep.

---

## 7. Open Questions

**OQ-1: Does `SpoQuad` exist in lance-graph HEAD?**
Grep of `lance-graph-contract` and `lance-graph` core found no `pub struct SpoQuad`.
Either it is planned-but-not-shipped, lives under a different name (e.g.,
`SpoTriple` or `TripletGraph`), or is q2-internal. Engineer must audit before
writing the re-export. If absent from lance-graph, W11 (FMA demo) needs to know
what carrier type to use instead -- likely `CrystalFingerprint::Vsa16kF32`.

**OQ-2: What does `graph-flow` stub export?**
`graph-flow` is intentionally left as a stub. For the FMA demo, does W11
actually import `graph-flow` types? If yes, this becomes P0 alongside
`q2-ndarray`. If no, `graph-flow` stub can stay as-is for this sprint.

**OQ-3: Is `crates/stubs/lib.rs` a top-level re-export aggregator?**
The directory listing shows `crates/stubs/lib.rs` (258 bytes, confirmed SHA
`ed4ddb07e25378a2b7090af59b5da6e8342fedaa`). If this is a crate root that
re-exports all three stubs, it is the single point-of-entry and changing it
may break more callers than expected. Audit before Commit B.

**OQ-4: Is `lance-graph-callcenter` reachable as a workspace dep in q2?**
`OwlIdentity` and `UnifiedBridge` live in `lance-graph-callcenter`. The q2
workspace Cargo.toml does not list `lance-graph-callcenter` in
`[workspace.dependencies]` (not visible in the confirmed HEAD). If it is not
a sibling crate under `../lance-graph/crates/lance-graph-callcenter`, a new
entry must be added before the `ogit.rs` re-export will compile.

**OQ-5: Edition mismatch between q2 and lance-graph-contract?**
q2 declares `edition = "2024"` in `[workspace.package]`. `lance-graph-contract`
edition is unverified here. If there is an edition mismatch in proc-macro or
`use` resolution, the re-export surface may compile but produce subtly different
trait object vtables. Verify editions match before claiming the type-eq tests as
sufficient.
