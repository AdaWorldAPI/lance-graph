# PR-G1: `/modules/<name>/manifest.yaml` + compile-time codegen

**Sprint-6 deliverable — Pattern E (Compile-Time Consumer Binding), PR-G1.**
**Canonical plan reference:** `.claude/plans/compile-time-consumer-binding-v1.md` §2.1 (D-MANIFEST-MODULES).
**Tech-debt anchor:** TD-MANIFEST-MODULES-4.
**Depends on:** PR-B-1 (ContextBundle + ConsumerPointer types), PR-C-1 (Consumer trait surface).
**Consumed by:** PR-G2 (ractor supervisor enumerates `inventory::iter::<ConsumerRegistration>()`).

---

## 1. Goal

Replace hand-written OGIT G-slot constants and per-consumer registry wiring with a PostNuke-style directory-of-manifests system. After this PR lands:

- Adding a new consumer = drop one `manifest.yaml` under `modules/<name>/` + write `~50 LOC` of consumer-crate glue (`impl Consumer for FooActor` + `inventory::submit!`).
- Zero edits to `lance-graph-contract` source code after initial build-script ships.
- Zero edits to `lance-graph-callcenter` source for new consumers (supervisor reads registrations via `inventory::iter`).

---

## 2. Manifest format — YAML, justified

**Choice:** YAML (`.yaml`). **Alternatives considered:** TOML, a Rust crate exporting `const` values.

**Justification:**
- The canonical plan doc (`compile-time-consumer-binding-v1.md` §2.1) specifies YAML and provides complete examples for all 6 initial modules. TOML would work but introduces unnecessary divergence from the spec.
- YAML allows `~` (null) for optional blocks (`actor: ~`, `rbac_policy: ~`) which maps cleanly to `Option<T>` in `serde_yaml`. TOML requires explicitly omitting the key, which is less legible in a sparse manifest like `fma` or `dolce`.
- A Rust crate exporting `const` values (alternative C) would create the Cargo dependency-cycle defect the sprint-3 CORRECTION addressed: any crate referencing actor types would need to depend on consumer crates, which already depend on `lance-graph-contract`. Rejected.
- `serde_yaml = "0.9"` is already in the workspace lockfile (via `lance-graph-callcenter` tooling). No new dep introduced for `lance-graph-contract` build-time.

**Schema strictness:** hard-fail on missing required fields at build time (`#[serde(deny_unknown_fields)]` on the outer struct; soft-accept unknown nested fields under `stack_profile` via a `BTreeMap<String, serde_yaml::Value>` catch-all). A typo in a required key (`ogig_g` instead of `ogit_g`) must break the build immediately, never silently mis-register.

---

## 3. Manifest schema (6 initial modules)

### 3.1 Required top-level keys

| Key | Type | Notes |
|---|---|---|
| `ogit_g` | string token | Must match canonical slot table (§3.3) |
| `version` | u32 | `>= 1`; bumped on schema-incompatible ontology changes |
| `domain_name` | string | Unique across all manifests; matches directory name |
| `inert_when_consumer_absent` | bool | `true` = OK if actor crate absent; `false` = build error |
| `entity_types` | map\<string, "u16=NNN"\> | Reserves entity-type IDs inside this G; may be empty `{}` |
| `rbac_policy` | string or `~` | Policy name resolved by supervisor; `~` for inert |
| `stack_profile` | object or `~` | Per-domain runtime knobs (see §3.2) |
| `action_capabilities` | map\<string, mode\> | mode in `{direct, escalate, deny, permit, permit_with_audit}`; may be empty |
| `actor` | object or `~` | Binding to consumer crate; `~` for inert modules |
| `inherits_from` | string or `~` | Parent `domain_name`; `~` only for DOLCE root |

### 3.2 `stack_profile` sub-keys

```yaml
stack_profile:
  audit_retention_days: 3650    # BMV-A §57 / GDPR retention
  requires_fail_closed: true    # escalate on Policy::evaluate error
  escalation: llm               # llm | human | deny
```

Unknown sub-keys are accepted silently (per-domain extensibility).

### 3.3 OGIT-G slot assignments (canonical)

| `ogit_g` token | Slot | Consumer crate | Status |
|---|---|---|---|
| `DOLCE` | 0 | none | inert root context |
| `MED` | 1 | reserved | not used this sprint |
| `HEALTHCARE` | 2 | `medcare-rs` | active |
| `GOTHAM` | 3 | `q2-cockpit-rs` | active |
| `SMB` | 4 | `smb-office-rs` | active |
| `FMA` | 5 | none | inert OWL data bundle |
| `CRM` | 6 | `hubspo-rs` | inert placeholder |

### 3.4 Sample manifests

**`modules/medcare/manifest.yaml`** (active consumer, fail-closed, regulatory):

```yaml
ogit_g: HEALTHCARE
version: 1
domain_name: medcare
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
  audit_retention_days: 3650
  requires_fail_closed: true
  escalation: llm

action_capabilities:
  finalize_diagnosis:       escalate
  issue_btm_prescription:   escalate
  anonymize_patient:        escalate
  read_lab:                 permit_with_audit
  read_anamnese:            permit_with_audit

actor:
  crate: medcare-rs
  type: MedCareActor
  message_type: MedCareMessage

inherits_from: dolce
```

**`modules/fma/manifest.yaml`** (inert data bundle, no actor):

```yaml
ogit_g: FMA
version: 1
domain_name: fma
inert_when_consumer_absent: true

entity_types: {}

rbac_policy: ~
stack_profile: ~
action_capabilities: {}
actor: ~

inherits_from: dolce
```

**`modules/dolce/manifest.yaml`** (root context, always present):

```yaml
ogit_g: DOLCE
version: 1
domain_name: dolce
inert_when_consumer_absent: true

entity_types:
  Endurant:    u16=1
  Perdurant:   u16=2
  Quality:     u16=3
  Abstract:    u16=4
  SocialObject: u16=5
  Information: u16=6
  SocialAct:   u16=7

rbac_policy: ~
stack_profile: ~
action_capabilities: {}
actor: ~
inherits_from: ~
```

**`modules/smb-office/manifest.yaml`** (active, SMB domain):

```yaml
ogit_g: SMB
version: 1
domain_name: smb-office
inert_when_consumer_absent: false

entity_types:
  Customer:   u16=200
  Invoice:    u16=201
  TaxDecl:    u16=202
  Document:   u16=203
  Contact:    u16=204

rbac_policy: smb_policy

stack_profile:
  audit_retention_days: 3650
  requires_fail_closed: false
  escalation: human

action_capabilities:
  send_mahnung:    escalate
  classify_tax:    direct
  read_invoice:    permit
  read_customer:   permit

actor:
  crate: smb-office-rs
  type: SmbOfficeActor
  message_type: SmbOfficeMessage

inherits_from: dolce
```

**`modules/q2-cockpit/manifest.yaml`** (active, q2 Gotham):

```yaml
ogit_g: GOTHAM
version: 1
domain_name: q2-cockpit
inert_when_consumer_absent: false

entity_types:
  WorkOrder:   u16=300
  Asset:       u16=301
  SiteVisit:   u16=302
  Report:      u16=303

rbac_policy: q2_policy

stack_profile:
  audit_retention_days: 730
  requires_fail_closed: false
  escalation: human

action_capabilities:
  assign_order:  direct
  close_order:   escalate
  read_asset:    permit

actor:
  crate: q2-cockpit-rs
  type: Q2CockpitActor
  message_type: Q2CockpitMessage

inherits_from: dolce
```

**`modules/hubspo/manifest.yaml`** (inert placeholder, no crate yet):

```yaml
ogit_g: CRM
version: 1
domain_name: hubspo
inert_when_consumer_absent: true

entity_types:
  Lead:       u16=400
  Deal:       u16=401
  Contact:    u16=402

rbac_policy: ~
stack_profile: ~
action_capabilities: {}
actor: ~

inherits_from: dolce
```

---

## 4. Build-script: `crates/lance-graph-contract/build.rs`

**Home:** `lance-graph-contract`. Every consumer depends on this crate; it's the right home for the G-slot constants they import. The build script itself is `~160 LOC`; it's a build-time tool and its build-dep (`serde_yaml`) does not become a runtime dependency of the crate.

**Contract zero-dep invariant preserved:** `[dependencies]` in `lance-graph-contract/Cargo.toml` stays empty. `serde_yaml = "0.9"` lands under `[build-dependencies]` only.

### 4.1 Algorithm (~160 LOC)

```text
1. Determine workspace root: CARGO_MANIFEST_DIR/../..
2. Glob: workspace_root/modules/*/manifest.yaml
   Sort lexicographically for deterministic output.
3. For each manifest path:
   a. Read UTF-8 bytes
   b. Parse via serde_yaml::from_str::<ManifestRaw> with deny_unknown_fields
      (ManifestRaw covers all required keys; unknown top-level keys = hard fail)
   c. Validate:
      - ogit_g must appear in CANONICAL_SLOTS table
      - version >= 1
      - domain_name equals parent directory name (consistency gate)
      - entity_type codes u16=NNN: NNN in range [1, 65535]; no duplicates within this G
      - if !inert_when_consumer_absent AND actor.is_none(): build error
      - inherits_from resolves to a domain_name seen in a prior manifest
        OR is null AND ogit_g == DOLCE
4. Cross-manifest validation:
   - Reject duplicate ogit_g tokens
   - Reject duplicate entity-type codes globally (u16 uniqueness across all G)
   - Reject domain_name collisions
5. Detect active consumers:
   - For each non-inert manifest with actor.crate set, check
     CARGO_FEATURE_MODULE_<UPPER_DOMAIN_NAME> env var.
     If set: emit ConsumerPointer binding in registry_seed (data-only, no type ref).
     If not set and inert_when_consumer_absent=false: emit compile_error!
6. Emit to OUT_DIR/ogit_namespace.rs:
   - pub mod OGIT { pub const <NAME>_V<N>: (u32, u32) = (slot, version); ... }
7. Emit to OUT_DIR/manifest_metadata.rs:
   - static MANIFEST_METADATA: phf::Map<u32, ManifestMetadata>
     (data only: strings, u16 arrays, bool flags — no Rust type references)
8. println!("cargo:rerun-if-changed={}", manifest_path) for each file
9. println!("cargo:rerun-if-changed={}", workspace_root/Cargo.toml)
```

### 4.2 Codegen output: `ogit_namespace.rs`

```rust
// AUTO-GENERATED by crates/lance-graph-contract/build.rs
// Source: modules/*/manifest.yaml
// DO NOT EDIT — regenerated when any manifest changes.
#![allow(non_snake_case)]

pub mod OGIT {
    /// (g_slot, manifest_version). Import as `use lance_graph_contract::OGIT;`
    pub const DOLCE_V1:      (u32, u32) = (0, 1);
    pub const HEALTHCARE_V1: (u32, u32) = (2, 1);
    pub const GOTHAM_V1:     (u32, u32) = (3, 1);
    pub const SMB_V1:        (u32, u32) = (4, 1);
    pub const FMA_V1:        (u32, u32) = (5, 1);
    pub const CRM_V1:        (u32, u32) = (6, 1);
}

/// All slots seen in any manifest, inert or active.
pub const ALL_G_SLOTS: &[u32] = &[0, 2, 3, 4, 5, 6];
```

### 4.3 Codegen output: `manifest_metadata.rs`

Data-only. No import of consumer crate types. `phf_map!` requires `phf = { version = "0.11", features = ["macros"] }` as a **build-dependency** plus a **runtime dependency** (for the emitted `phf::Map` type).

```rust
// AUTO-GENERATED by crates/lance-graph-contract/build.rs
use phf::phf_map;

/// Per-domain metadata extracted from manifests at compile time.
/// Keyed by G slot (u32). Consumer crates read this to self-populate
/// their ConsumerPointer without touching lance-graph-contract source.
pub static MANIFEST_METADATA: phf::Map<u32, ManifestMetadata> = phf_map! {
    0u32 => ManifestMetadata {
        domain_name:   "dolce",
        g_slot:        0,
        version:       1,
        inert:         true,
        rbac_policy:   None,
        stack:         StackProfile { audit_days: 0, fail_closed: false, escalation: Escalation::Deny },
        actor_crate:   None,
        actor_type:    None,
        entity_count:  7,
    },
    2u32 => ManifestMetadata {
        domain_name:   "medcare",
        g_slot:        2,
        version:       1,
        inert:         false,
        rbac_policy:   Some("medcare_policy"),
        stack:         StackProfile { audit_days: 3650, fail_closed: true, escalation: Escalation::Llm },
        actor_crate:   Some("medcare-rs"),
        actor_type:    Some("MedCareActor"),
        entity_count:  6,
    },
    // ... smb-office (4), gotham (3), fma (5), crm (6) ...
};
```

**`ManifestMetadata`** struct lives in `lance-graph-contract/src/manifest.rs` (hand-written, ~40 LOC). It holds only `&'static str` and primitive types — no generic parameters, no consumer crate references.

### 4.4 Consumer-side self-registration (not in build.rs — in each consumer crate)

The build script emits NO consumer type references. Each consumer crate opts in via:

```rust
// crates/medcare-rs/src/actor.rs
use lance_graph_contract::{OGIT, MANIFEST_METADATA, consumer::ConsumerRegistration};
use inventory;

pub struct MedCareActor;

impl Consumer for MedCareActor {
    const G: u32 = OGIT::HEALTHCARE_V1.0;

    fn pointer() -> ConsumerPointer {
        let meta = &MANIFEST_METADATA[&Self::G];
        ConsumerPointer {
            g:               Self::G,
            version:         OGIT::HEALTHCARE_V1.1,
            domain_name:     meta.domain_name,
            stack_profile:   meta.stack.clone(),
            inert:           false,
        }
    }
}

inventory::submit! {
    ConsumerRegistration::new::<MedCareActor>()
}
```

Cargo dependency graph after fix:

```
medcare-rs ──→ lance-graph-contract   [unchanged]
medcare-rs ──→ lance-graph-callcenter [unchanged]
lance-graph-contract ──✗ (no edge to consumer crates)
lance-graph-callcenter::supervisor reads inventory::iter::<ConsumerRegistration>() at startup
```

No Cargo cycle. Confirmed by `cargo tree -e no-dev -p lance-graph-contract` showing no consumer crates in output.

---

## 5. What gets codegen'd — summary

| Emitted symbol | Location | Purpose |
|---|---|---|
| `pub mod OGIT { pub const *_V*: (u32, u32) }` | `ogit_namespace.rs` | Typed G-slot constants for all consumer crates to import |
| `pub const ALL_G_SLOTS: &[u32]` | `ogit_namespace.rs` | Supervisor boot enumeration |
| `pub static MANIFEST_METADATA: phf::Map<u32, ManifestMetadata>` | `manifest_metadata.rs` | Data-only manifest facts; consumer crates read at compile/runtime |

**What is NOT emitted by build.rs:**
- Actor type references (`medcare_rs::MedCareActor`) — dependency-cycle defect prevention
- `ConsumerRegistration` entries — each consumer crate emits its own via `inventory::submit!`
- `OntologyRegistry` hydration — that is `seed_from_manifests()` in `lance-graph-callcenter`, reading `MANIFEST_METADATA` + `inventory::iter` at startup

---

## 6. Incremental compilation behavior

`cargo:rerun-if-changed` is emitted for:
1. Every `modules/*/manifest.yaml` file individually.
2. `{workspace_root}/Cargo.toml` (so adding a new workspace member triggers rebuild).
3. The build script itself (`build.rs` — Cargo tracks this automatically).

**Guarantee:** modifying only `modules/medcare/manifest.yaml` triggers recompile of `lance-graph-contract` only, not of `smb-office-rs`, `q2-cockpit-rs`, etc. Those consumer crates recompile only if the `ogit_namespace.rs` or `manifest_metadata.rs` bytes change (i.e. the constants they import change).

**Idempotency gate:** Build output must be byte-identical across two consecutive runs with no manifest change. Enforced by the test `tests/idempotency.rs` (§7.3). The `phf_map!` macro uses sorted keys and deterministic hash seeds; output is deterministic.

---

## 7. Failure modes

### 7.1 Malformed manifest

**Symptom:** `serde_yaml::from_str` returns `Err` (unknown field, wrong type, missing required key).

**Build output:**
```
error[E0080]: evaluation of constant value failed
  = note: called `Option::unwrap()` on a `None` value
  ... build script panicked: manifest parse error in modules/medcare/manifest.yaml:
      unknown field `ogig_g`, expected one of `ogit_g`, `version`, `domain_name`, ...
```

**Resolution:** Fix the typo in the manifest. Build is deterministic — the error re-fires on every compile until fixed.

### 7.2 Conflicting G slots (two manifests claim same slot)

**Symptom:** Second manifest parsed with the same `ogit_g` token as an already-seen manifest.

**Build output:**
```
build script panicked: duplicate G slot: HEALTHCARE claimed by both
  modules/medcare/manifest.yaml AND modules/medcare-v2/manifest.yaml
  (if adding HEALTHCARE_V2, bump `version:` in the existing manifest, do not create a parallel directory)
```

### 7.3 Conflicting entity-type codes globally

**Symptom:** Two manifests declare the same `u16=NNN` code for different entity types.

**Build output:**
```
build script panicked: entity-type code collision: u16=100 is declared by
  modules/medcare/manifest.yaml (Patient) AND
  modules/smb-office/manifest.yaml (Customer)
  Entity-type codes must be globally unique across all G slots.
```

### 7.4 Non-inert manifest with missing consumer crate

**Symptom:** `inert_when_consumer_absent: false`, `CARGO_FEATURE_MODULE_<NAME>` env var not set (crate not compiled in), `actor.crate` points to a workspace member that isn't in scope.

**Build output:**
```
build script panicked: consumer crate required but absent:
  modules/medcare/manifest.yaml has inert_when_consumer_absent=false
  but feature `module-medcare` is not enabled.
  Either enable the feature in your binary's Cargo.toml or set inert_when_consumer_absent=true.
```

### 7.5 `inherits_from` does not resolve

**Symptom:** A manifest references a `domain_name` that hasn't been seen yet (alphabetical parse order).

**Resolution:** Build script sorts manifests by `ogit_g` slot (ascending) before processing, so DOLCE (slot 0) always loads first. A reference to an unresolved parent after sorting indicates a genuine unregistered parent.

---

## 8. LOC estimate

| Artifact | LOC |
|---|---|
| `crates/lance-graph-contract/build.rs` | ~160 |
| `crates/lance-graph-contract/src/manifest.rs` (ManifestMetadata type + support enums) | ~60 |
| `modules/dolce/manifest.yaml` | ~20 |
| `modules/medcare/manifest.yaml` | ~30 |
| `modules/smb-office/manifest.yaml` | ~25 |
| `modules/q2-cockpit/manifest.yaml` | ~22 |
| `modules/fma/manifest.yaml` | ~15 |
| `modules/hubspo/manifest.yaml` | ~18 |
| `crates/lance-graph-contract/Cargo.toml` changes | ~5 |
| `tests/manifest_parse.rs` | ~50 |
| `tests/idempotency.rs` | ~25 |
| `tests/duplicate_g_rejected.rs` | ~20 |
| `tests/duplicate_entity_code_rejected.rs` | ~20 |
| `tests/inert_no_consumer_pointer.rs` | ~20 |
| **Total** | **~470 LOC** |

This is slightly above the plan's ~410 LOC estimate; the delta (~60 LOC) comes from: (a) the `manifest.rs` type file that was implicit in the plan, (b) the extra `duplicate_entity_code_rejected` test added per CORRECTION guidance, and (c) the `smb-office` and `q2-cockpit` manifests being more populated than estimated.

---

## 9. DELTA vs `compile-time-consumer-binding-v1.md` Pattern E

Pattern E (§2.1) specified the design at planning level. This spec concretizes the following sub-items:

| Plan item | This spec's concretization |
|---|---|
| "Parse each via `serde_yaml`" (§2.1 step 2) | Concretized: `ManifestRaw` struct with `#[serde(deny_unknown_fields)]` on outer struct; soft-accept on `stack_profile` sub-keys via `BTreeMap<String, Value>` |
| "Validate: actor.crate resolvable as workspace member" (§2.1 step 2) | **Revised:** detection is via `CARGO_FEATURE_MODULE_<NAME>` env var, NOT workspace member scan. Reason: scanning workspace members gives false positives for crates present but not wired in (confirmed by sprint-3 CORRECTION §3). |
| "Emit `Consumer` trait registration shims keyed by (G, version)" (§2.1 step 4) | **Revised:** shims are NOT emitted by build.rs (dependency-cycle defect; sprint-3 CORRECTION). Build.rs emits ONLY `OGIT::*` constants and `MANIFEST_METADATA` phf::Map. Consumer self-registration uses `inventory::submit!` in the consumer crate. |
| "Detect peers: walk workspace Cargo.toml" (§2.1 step 5) | **Superseded** by feature-flag detection. Workspace walking would require `cargo_metadata` as a heavy build-dep and has false-positive issues. |
| "Emit per-G ACTIVE vs INERT markers" (§2.1 step 5) | Concretized: `ManifestMetadata.inert: bool` field in phf::Map; `ALL_G_SLOTS: &[u32]` constant enumerates all registered slots. |
| "Idempotency: re-running produces byte-identical output" (§2.1 step 6) | Concretized: `tests/idempotency.rs` test + `phf_map!` deterministic ordering. |
| "~150 LOC build-script + 6×~30 LOC manifests + ~80 LOC test fixtures = ~410 LOC" | Concretized: ~160 LOC build.rs + ~130 LOC manifests + ~60 LOC `manifest.rs` type + ~135 LOC tests = ~470 LOC (+60 for dependency-cycle fix artifacts). |
| Open question 1: "Build-script home?" → Recommend: contract | **Confirmed:** contract crate. |
| Open question 2: "Strict vs. evolvable schema?" | Confirmed: `deny_unknown_fields` on required outer struct; soft-accept on `stack_profile` nested map. |
| Open question 4: "Inert manifest semantics" | Confirmed: inert = registered in OGIT MANIFEST_METADATA, traversable by supervisor, `ConsumerRegistration` absent from `inventory::iter`. Confirmed semantics: `Route { g: FMA_V1.0 }` returns `NoConsumer`, not a panic. |

**Items from plan §2.1 NOT covered here (scope boundary):**

- `MODULE_TABLE` as a `&[ModuleEntry]` runtime const — the plan's original shape. **Replaced** by `MANIFEST_METADATA: phf::Map<u32, ManifestMetadata>` (same semantics, keyed differently, no dependency-cycle issue).
- D-RACTOR-SUPERVISOR (§2.2) — that is PR-G2 (W11), which consumes this PR's output.
- Entity-type `parent:` references (e.g. `{ code: 100, parent: dolce.Person }`) — the plan's medcare example uses this richer format. This spec simplifies to `u16=NNN` codes without parent-ref in the build.rs validation step (parent resolution requires FMA OWL hydrator which is PR-D-1 scope). Added as open question §10.5.

---

## 10. Open questions for the engineer

1. **`phf` as a runtime dep of `lance-graph-contract`?** The zero-dep invariant in `CLAUDE.md` §Workspace Conventions says "no external crate deps". `phf` would be the first. **Options:** (a) emit the phf::Map only, accept the dep break with explicit governance annotation; (b) emit a const array of `(u32, ManifestMetadata)` pairs sorted by key, accessed via binary search (`O(log N)`; fine for N ≤ 50). Recommendation: option (b) preserves the zero-dep invariant. The binary-search accessor is trivial to write (~10 LOC). Defer `phf` to a future sprint if N grows beyond ~100.

2. **Entity-type `parent:` references in manifests?** The canonical plan's medcare manifest includes `parent: dolce.Person` for each entity type (enables DOLCE class inheritance resolution). This spec omits parent-ref parsing in the build.rs validator. If parent-ref resolution is needed in the `ManifestMetadata`, add it as a follow-up or include the field as `Option<&'static str>` (string-only, no OWL resolution at build time).

3. **`version:` bump policy for ontology evolution?** When `HEALTHCARE_V1` → `HEALTHCARE_V2`, does the old manifest get a version bump or a new directory (`modules/medcare-v2/`)? The plan recommends coexistence via `(G, version)` tuples. Implication: the G slot stays the same; `version: 2` in the same `modules/medcare/manifest.yaml`. The build script emits `HEALTHCARE_V2: (2, 2)` alongside `HEALTHCARE_V1: (2, 1)`. The old constant is deprecated but not removed until all consumers migrate.

4. **Commit generated files to git?** Sprint-3 spec recommends YES (debuggable; CI doesn't regen). This spec concurs: commit `src/generated/ogit_namespace.rs` and `src/generated/manifest_metadata.rs` to version control. The `cargo:rerun-if-changed` chain keeps them honest on developer machines; CI verifies they're up to date by running `cargo build` and checking for dirty files.

5. **`modules/` path: workspace root vs. `crates/lance-graph-contract/modules/`?** Workspace root is the canonical choice per sprint-3 spec (visible to all crates, non-Rust tooling, future Python scripts). The build.rs path glob becomes `${CARGO_MANIFEST_DIR}/../../modules/*/manifest.yaml` (two levels up from the contract crate).

---

## 11. Test plan

| Test file | What it tests |
|---|---|
| `tests/manifest_parse.rs` | Round-trip `serde_yaml` of all 6 initial manifests; assert key fields (g_slot, domain_name, inert flag, entity count). |
| `tests/idempotency.rs` | Run build.rs emission logic twice; assert byte-identical output. Validates `cargo:rerun-if-changed` correctness claim. |
| `tests/duplicate_g_rejected.rs` | Synthesize a 7th manifest claiming `ogit_g: HEALTHCARE` (collides with medcare); assert build.rs exits non-zero with "duplicate G slot" message. |
| `tests/duplicate_entity_code_rejected.rs` | Two manifests each declare `u16=100`; assert collision error. |
| `tests/inert_no_consumer_pointer.rs` | Parse `modules/fma/manifest.yaml`; assert `ManifestMetadata.inert == true` and `actor_crate.is_none()`. |

**Regression gate:** `cargo build -p lance-graph-contract` succeeds with zero additional features (no consumer crates). Also `cargo check --workspace` must pass.

---

## 12. Files to create/modify

| File | Action | Notes |
|---|---|---|
| `modules/dolce/manifest.yaml` | CREATE | Root context |
| `modules/medcare/manifest.yaml` | CREATE | Active healthcare consumer |
| `modules/smb-office/manifest.yaml` | CREATE | Active SMB consumer |
| `modules/q2-cockpit/manifest.yaml` | CREATE | Active Gotham consumer |
| `modules/fma/manifest.yaml` | CREATE | Inert OWL data bundle |
| `modules/hubspo/manifest.yaml` | CREATE | Inert CRM placeholder |
| `crates/lance-graph-contract/build.rs` | CREATE | ~160 LOC parse+validate+emit |
| `crates/lance-graph-contract/src/manifest.rs` | CREATE | `ManifestMetadata` type + `StackProfile` + `Escalation` enums (~60 LOC) |
| `crates/lance-graph-contract/src/generated/ogit_namespace.rs` | GENERATED | OGIT::* constants |
| `crates/lance-graph-contract/src/generated/manifest_metadata.rs` | GENERATED | Static metadata array |
| `crates/lance-graph-contract/src/lib.rs` | MODIFY | `pub mod manifest; pub mod generated;` |
| `crates/lance-graph-contract/Cargo.toml` | MODIFY | `build = "build.rs"`, `[build-dependencies] serde_yaml = "0.9"` |
| `tests/manifest_parse.rs` | CREATE | 5 unit tests |

---

## 13. Acceptance criteria

- [ ] 6 `manifest.yaml` files in `modules/{dolce,medcare,smb-office,q2-cockpit,fma,hubspo}/`.
- [ ] `cargo build -p lance-graph-contract` emits `ogit_namespace.rs` with all 6 `OGIT::*_V1` constants.
- [ ] `cargo build -p lance-graph-contract` emits `manifest_metadata.rs` with all 6 domain entries.
- [ ] All 5 tests in `tests/manifest_parse.rs` pass.
- [ ] `tests/idempotency.rs`: two consecutive builds with no manifest change produce byte-identical output.
- [ ] `tests/duplicate_g_rejected.rs`: build exits non-zero with useful message.
- [ ] `cargo build --workspace` succeeds without any `module-*` features enabled.
- [ ] `cargo check -p lance-graph-contract` passes with zero added deps in `[dependencies]` (zero-dep invariant preserved).

---

## 14. Cross-references

- `.claude/plans/compile-time-consumer-binding-v1.md` — Pattern E canonical source (D-MANIFEST-MODULES §2.1)
- `.claude/specs/pr-e-1-manifest-modules.md` — sprint-3 W5 predecessor spec + 2026-05-12 CORRECTION (dependency-cycle fix)
- `.claude/specs/pr-b-1-context-bundle.md` — required precursor (ContextBundle + ConsumerPointer types)
- `.claude/specs/pr-c-1-generic-bridge.md` — sister spec (`consumer.rs` Consumer trait surface)
- `.claude/specs/pr-g2-ractor-supervisor.md` — W11 sibling; consumes `inventory::iter::<ConsumerRegistration>()` from this PR
- `.claude/board/TECH_DEBT.md` — TD-MANIFEST-MODULES-4
- `.claude/knowledge/tier-0-pattern-recognition.md` — Pattern E section
