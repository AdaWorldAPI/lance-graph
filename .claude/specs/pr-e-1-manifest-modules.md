# PR-E-1: /modules/<name>/manifest.yaml + build-script glue

**Tier-2 implementation spec — Pattern E canonical (Compile-Time Consumer Binding).**
**Tech-debt anchor:** TD-MANIFEST-MODULES-4.
**Sprint-3 owner:** W5 (this spec) -> engineer pickup.

---

## Goal

PostNuke (anno 2000) module pattern, ported to Rust + Cargo. Each consumer
of the OGIT-G stack lives in its own directory under `modules/<name>/` and
declares its presence via a single `manifest.yaml`. The manifest binds the
consumer to its OGIT-G slot, declares its entity types, and points at the
Rust crate that supplies the actor implementation.

A build script in `lance-graph-contract` reads every
`modules/*/manifest.yaml` at compile time and emits Rust constants plus
`Consumer` trait scaffolding. The result: adding a new consumer means
shipping a manifest + a small actor crate, not surgery on the contract
crate.

This is the on-ramp pattern for the Tier-2 supervised consumer mesh
(Pattern E + F together). Once the manifest contract exists, the ractor
supervisor (PR-F-1) can boot a consumer set deterministically from
disk-resident config.

---

## Files to touch

| File | Change |
|---|---|
| `modules/dolce/manifest.yaml` | **NEW** — root context (always present, no consumer crate) |
| `modules/medcare/manifest.yaml` | **NEW** — G=Healthcare consumer; binds to `medcare-rs` |
| `modules/smb-office/manifest.yaml` | **NEW** — G=SMB consumer; binds to `smb-office-rs` |
| `modules/q2-cockpit/manifest.yaml` | **NEW** — G=Gotham consumer; binds to `q2-cockpit-rs` |
| `modules/fma/manifest.yaml` | **NEW** — inert OWL bundle (no consumer; data only) |
| `modules/hubspo/manifest.yaml` | **NEW** — G=CRM placeholder (consumer crate doesn't exist yet; inert until `hubspo-rs` lands) |
| `crates/lance-graph-contract/build.rs` | **NEW** — scan `modules/*/manifest.yaml`, emit `OGIT::*` constants + registry seed |
| `crates/lance-graph-contract/src/consumer.rs` | **EXTEND** — `Consumer` trait declaration (W4 PR-C-1 sister also touches this; coordinate) |
| `crates/lance-graph-contract/src/generated/ogit_namespace.rs` | **NEW (generated)** — emitted by `build.rs`; contains `OGIT::*` consts |
| `crates/lance-graph-contract/src/generated/registry_seed.rs` | **NEW (generated)** — emitted by `build.rs`; contains `seed_from_manifests()` factory |
| `crates/lance-graph-contract/Cargo.toml` | **EXTEND** — `serde_yaml` build-dep, declare `build = "build.rs"` |

---

## Sample manifest.yaml (medcare — fully populated)

```yaml
# modules/medcare/manifest.yaml
ogit_g: HEALTHCARE
version: 1
domain_name: medcare

# When false: medcare-rs presence is mandatory (build error if absent).
# When true: missing crate is OK, ContextBundle yields consumer_pointer=None.
inert_when_consumer_absent: false

entity_types:
  Patient:      u16=100
  Diagnosis:    u16=101
  LabResult:    u16=102
  Prescription: u16=103
  Anamnese:     u16=104
  Ueberweisung: u16=105

rbac_policy: medcare_policy

stack_profile:
  audit_retention_days: 3650   # BMV-A section 57 retention floor
  requires_fail_closed: true
  escalation: llm

action_capabilities:
  finalize_diagnosis:       escalate
  issue_btm_prescription:   escalate
  anonymize_patient:        escalate

actor:
  crate: medcare-rs
  type: MedCareActor
  message_type: MedCareMessage

inherits_from: dolce
```

### Sample manifest.yaml (fma — inert OWL bundle)

```yaml
# modules/fma/manifest.yaml
ogit_g: FMA
version: 1
domain_name: fma

inert_when_consumer_absent: true   # always inert; data-only bundle

entity_types: {}      # FMA exposes its types via the OWL hydrator (PR-D-1)

rbac_policy: ~        # no actor, no RBAC
stack_profile: ~
action_capabilities: {}

actor: ~              # no actor crate; ContextBundle.consumer_pointer = None

inherits_from: dolce
```

### Sample manifest.yaml (dolce — root context, no consumer)

```yaml
# modules/dolce/manifest.yaml
ogit_g: DOLCE
version: 1
domain_name: dolce

inert_when_consumer_absent: true

entity_types:
  Endurant:     u16=1
  Perdurant:    u16=2
  Quality:      u16=3
  Abstract:     u16=4

rbac_policy: ~
stack_profile: ~
action_capabilities: {}
actor: ~

inherits_from: ~       # DOLCE is the root; no parent
```

### OGIT-G slot assignments (canonical)

| `ogit_g` token | Slot | Notes |
|---|---|---|
| `DOLCE`      | `0` | root context; reserved (open-question 1 in PR-A-1) |
| `MED`        | `1` | reserved for legacy MEDS bridge (not used this sprint) |
| `HEALTHCARE` | `2` | `medcare-rs` |
| `GOTHAM`     | `3` | `q2-cockpit-rs` |
| `SMB`        | `4` | `smb-office-rs` |
| `FMA`        | `5` | inert OWL bundle |
| `CRM`        | `6` | `hubspo-rs` placeholder |

---

## Build-script algorithm (~150 LOC)

```text
1. Glob modules/*/manifest.yaml from CARGO_MANIFEST_DIR/../../modules/
2. For each path:
   a. Parse via serde_yaml into ManifestRaw struct
   b. Validate:
      - ogit_g token is in the canonical slot table (above)
      - version >= 1
      - domain_name is unique across all manifests parsed so far
      - if actor.crate present, it must appear in workspace.members
        (read by re-parsing the workspace Cargo.toml)
3. After all manifests parsed:
   - Reject if two manifests claim the same ogit_g token
   - Reject if a non-root manifest's inherits_from doesn't resolve
4. Emit src/generated/ogit_namespace.rs:
   pub mod OGIT {
       pub const DOLCE_V1:      (u32, u32) = (0, 1);
       pub const HEALTHCARE_V1: (u32, u32) = (2, 1);
       pub const GOTHAM_V1:     (u32, u32) = (3, 1);
       pub const SMB_V1:        (u32, u32) = (4, 1);
       pub const FMA_V1:        (u32, u32) = (5, 1);
       pub const CRM_V1:        (u32, u32) = (6, 1);
   }
5. Emit src/generated/registry_seed.rs:
   pub fn seed_from_manifests() -> OntologyRegistry { ... }
   (one ContextBundle per manifest; consumer_pointer populated only
    when the consumer crate is detected as a workspace member AND the
    feature flag `module-<name>` is enabled.)
6. println!("cargo:rerun-if-changed=...") on every parsed manifest path
   plus the workspace Cargo.toml so cargo notices new modules.
```

### Generated `ogit_namespace.rs` (sample)

```rust
// AUTO-GENERATED by build.rs. DO NOT EDIT.
// Source: modules/*/manifest.yaml
#![allow(non_snake_case)]
pub mod OGIT {
    /// (slot, manifest_version)
    pub const DOLCE_V1:      (u32, u32) = (0, 1);
    pub const HEALTHCARE_V1: (u32, u32) = (2, 1);
    pub const GOTHAM_V1:     (u32, u32) = (3, 1);
    pub const SMB_V1:        (u32, u32) = (4, 1);
    pub const FMA_V1:        (u32, u32) = (5, 1);
    pub const CRM_V1:        (u32, u32) = (6, 1);
}
```

### Generated `registry_seed.rs` (sketch)

```rust
// AUTO-GENERATED by build.rs. DO NOT EDIT.
use crate::context_bundle::{ContextBundle, ConsumerPointer};
use crate::ontology_registry::OntologyRegistry;

pub fn seed_from_manifests() -> OntologyRegistry {
    let mut reg = OntologyRegistry::new();
    reg.insert(0, ContextBundle::dolce_root());
    reg.insert(2, ContextBundle::healthcare_v1(
        #[cfg(feature = "module-medcare")]
        Some(ConsumerPointer::new::<medcare_rs::MedCareActor>()),
        #[cfg(not(feature = "module-medcare"))]
        None,
    ));
    // ... q2-cockpit, smb-office, fma (None), hubspo (None until crate exists)
    reg
}
```

---

## Manifest schema (engineer-facing detail)

Required top-level keys per manifest:

| Key | Type | Required | Notes |
|---|---|---|---|
| `ogit_g` | string token | yes | must match canonical slot table |
| `version` | u32 | yes | >= 1; bumped on schema-incompatible changes |
| `domain_name` | string | yes | unique; matches the directory name |
| `inert_when_consumer_absent` | bool | yes | true = silent fallback; false = build error |
| `entity_types` | map<string, "u16=NNN"> | yes (may be empty) | reserves entity-type slots inside this G |
| `rbac_policy` | string or `~` | yes (nullable) | resolved by PR-F-1 supervisor |
| `stack_profile` | object or `~` | yes (nullable) | per-domain runtime knobs |
| `action_capabilities` | map<string, mode> | yes (may be empty) | mode in {direct, escalate, deny} |
| `actor` | object or `~` | yes (nullable) | binding to the consumer crate |
| `inherits_from` | string or `~` | yes (nullable) | parent manifest's `domain_name`; `~` only for `dolce` |

Unknown top-level keys are accepted with a `cargo:warning` (forward-compat;
see open-question 1).

---

## Test plan

| Test | Coverage |
|---|---|
| `tests/manifest_parse.rs` | Round-trip `serde_yaml` of all 6 sample manifests; assert key fields. |
| `tests/build_script_emits_namespace.rs` | After `cargo build`, assert `src/generated/ogit_namespace.rs` exists and contains every expected `OGIT::*_V1` const. |
| `tests/duplicate_g_rejected.rs` | Synthesise a temp manifest claiming `ogit_g: HEALTHCARE` (collides with medcare); assert `build.rs` exits non-zero with a useful message. |
| `tests/inert_bundle_consumer_pointer_none.rs` | Call `seed_from_manifests()`; assert `registry.resolve(OGIT::FMA_V1.0).consumer_pointer.is_none()`. |

**Regression gate:** `cargo build --workspace` must succeed both with and
without the optional `module-medcare`, `module-smb-office`, `module-q2-cockpit`
features enabled. (Validates the build script handles consumer-absent paths.)

---

## Dependencies

- **PR-B-1 (W3)** must land first. The build script emits code that
  references `ContextBundle` and `ConsumerPointer` types; both live in
  PR-B-1's contract surface. Without those types, the generated
  `registry_seed.rs` won't compile.
- **PR-C-1 (W4)** is a sister change. The `Consumer` trait declaration in
  `consumer.rs` is shared territory. Coordinate edits to avoid merge
  conflicts: PR-E-1 adds the manifest-driven `seed_from_manifests()`
  factory; PR-C-1 owns the trait shape itself.
- **External crate:** `serde_yaml` as a `build-dependencies` entry. Likely
  already in the workspace lockfile via other crates; if not, pin to the
  current `serde_yaml = "0.9"`.

---

## Acceptance criteria

- [ ] 6 `manifest.yaml` files in `modules/{dolce,medcare,smb-office,q2-cockpit,fma,hubspo}/`.
- [ ] `build.rs` reads, validates, and emits constants for all 6.
- [ ] `OGIT::*_V1` namespace constants accessible to all consumer crates
      via `lance_graph_contract::generated::ogit_namespace::OGIT`.
- [ ] `OntologyRegistry::seed_from_manifests()` factory works and is the
      single source of truth for runtime registry construction.
- [ ] 4 new tests green (parse round-trip, namespace emission, duplicate-G
      rejection, inert-bundle consumer_pointer = None).
- [ ] `cargo build --workspace` succeeds (validates the build script
      doesn't break compilation across feature combinations).
- [ ] Build script is idempotent: clean build and incremental build emit
      the same bytes for both generated files.

---

## Effort

**Medium.** ~330 LOC end-to-end. Estimate **~2 engineer-days** including
review.

Rough breakdown:

- 6 `manifest.yaml` files                              -- ~120 LOC YAML
- `build.rs` (parse + validate + emit)                 -- ~150 LOC
- `Consumer` trait extension                           -- ~30 LOC
- 4 new tests                                          -- ~40 LOC
- `Cargo.toml` build-dep wiring                        -- ~5 LOC
- Touch-ups in consumer crates to import `OGIT::*`     -- ~10 LOC

---

## Open questions for the engineer

1. **Manifest schema strictness: hard fail on missing required field, or
   allow forward-compat (new fields ignored)?** Recommend **hard fail on
   missing required, soft on unknown**. Required fields are listed in the
   schema table above; emit a `cargo:warning` for unknown top-level keys
   so future additions land smoothly. Unknown nested keys under
   `stack_profile` are silently accepted (per-domain knobs vary).
2. **Generated files: commit to git or `.gitignore`?** Recommend
   **commit**. Two reasons: (a) debuggable -- `git blame` on
   `ogit_namespace.rs` shows when slot `7` got added; (b) CI doesn't have
   to regen on every PR -- the build script's `rerun-if-changed` keys
   keep them honest. Cost: a small commit churn whenever a manifest
   changes, but that's the desired audit trail.
3. **`inert_when_consumer_absent` -- how does the build script detect
   "consumer crate compiled in"?** Recommend **feature-flag based**.
   Each consumer crate exposes a `module-<name>` Cargo feature in
   `lance-graph-contract`. The umbrella binary enables the features
   it wants. The build script reads `CARGO_FEATURE_MODULE_<NAME>` env
   vars and emits `#[cfg(feature = "module-<name>")]` gates around the
   `ConsumerPointer::new::<...>()` calls in `seed_from_manifests()`.
   Alternatives considered: scanning `workspace.members` directly
   (rejected -- false positives when a crate exists but isn't wired in)
   and using `cargo metadata` (rejected -- adds a heavy build-dep on
   `cargo_metadata` and slows the build).
4. **Manifest path: `modules/` at workspace root, or
   `crates/lance-graph-contract/modules/`?** Recommend **workspace root**.
   The manifests are visible to all crates and to non-Rust tooling
   (e.g., a future Python script that lists known consumers). Putting
   them under the contract crate would imply they're "owned" by it, but
   they're really workspace-level metadata.
5. **`hubspo` placeholder -- ship the manifest now, or wait for
   `hubspo-rs` to exist?** Recommend **ship now with
   `inert_when_consumer_absent: true`**. The whole point of this spec
   (and the W8 consumer-template dry-run) is to prove that adding a
   consumer requires *only* a manifest + a small actor crate. The
   placeholder is the regression gate: if shipping `hubspo-rs` later
   requires touching anything in `lance-graph-contract`, the
   architecture has regressed.
6. **What does `OGIT::MED_V1` map to (slot 1)?** Reserved slot for the
   legacy MEDS bridge; not used this sprint. The build script should
   *not* emit a constant for unused slots -- only for slots claimed by
   a manifest. Open a follow-up issue if/when we need slot 1 back.

---

## Cross-references

- `.claude/plans/compile-time-consumer-binding-v1.md` -- D-MANIFEST-MODULES
  (W11 sprint-2 sub-plan; canonical pattern E source)
- `.claude/board/TECH_DEBT.md` -- TD-MANIFEST-MODULES-4
- `.claude/specs/pr-b-1-context-bundle.md` -- W3 sister; required precursor
  (ContextBundle + ConsumerPointer types)
- `.claude/specs/pr-c-1-generic-bridge.md` -- W4 sister; shares
  `consumer.rs` trait surface
- `.claude/specs/pr-f-1-ractor-supervisor.md` -- W6 sister; supervisor
  reads manifests at runtime via `seed_from_manifests()`
- `.claude/specs/consumer-crate-template.md` -- W8; uses this spec to add
  `hubspo-rs` in <30 LOC of consumer-side code
- `.claude/specs/sprint-3-execution-plan.md` -- W1 master execution plan
- `.claude/knowledge/tier-0-pattern-recognition.md` -- Pattern E section
