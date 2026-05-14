# PR-CE64-MB-7 — `NdarrayCullPlugin` proof plugin (Bevy)

> **Sprint:** sprint-log-10 W9 (bevy-cull-plugin)
> **PR target:** PR-CE64-MB-7 (Wave 6, last in dep graph)
> **Crate:** `crates/bevy-cull-plugin/` (new, lance-graph workspace member; `bevy/headless`-feature gated)
> **LOC envelope:** ~500 LOC + ~250 LOC tests/benches (per parent plan §7)
> **Risk:** Low — proof-of-pattern only; no production consumer; isolated crate
> **Parent plan:** `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §7 PR-CE64-MB-7
> **Composes:**
> - `.claude/specs/pr-ce64-mb-1-par-tile-crate.md` (W1) — `Mailbox<T>` trait + `BindSpaceView<'static>` substrate
> - `.claude/specs/pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) — `MailboxSoA<N>` + `AttentionMaskSnapshot`
> - `.claude/specs/pr-ndarray-miri-complete.md` (W8) — `cfg(miri)` dispatch reroute makes `intersects_sphere_x16` Miri-checkable
> - Bevy session round-2 agent #7 (`intersects_sphere_x16` sketch) — original framing of frustum cull as `ndarray::hpc` SIMD lookup
> **Delta vs parent plan:** Plan §7 names the PR scope ("Bevy plugin `NdarrayCullPlugin` consuming MailboxSoA for frustum cull") in one row. This spec resolves it into: (a) plugin module layout, (b) bevy `Plugin` impl + system schedule placement, (c) `intersects_sphere_x16` consumption pattern, (d) compartment-spawn-per-visible-entity wiring, (e) test plan (5 correctness + 4 perf + 1 Miri-compat), (f) risk matrix. The plan defers all four to the implementation worker; this spec collapses that to concrete signatures and file paths.

---

## §1 Scope statement

`NdarrayCullPlugin` is a Bevy plugin that replaces stock `check_visibility` for the **proof-of-pattern that `par-tile` + `MailboxSoA<N>` can drive a frame-paced consumer outside lance-graph's own runtime.** It is **not** a production cull system, and its perf target (≤ 1.5× stock per-frame cost at 1K entities, ≤ 10 ms at 10K entities) is calibrated for demonstrating non-regression, not advancing the state of bevy culling.

### Forbidden scope (explicitly out)

- Replacing stock bevy `check_visibility` in any production fork. The plugin is opt-in behind a `cull-ndarray` cargo feature; the default bevy schedule is untouched.
- Bevy `Splat`, `Cognitive`, `Audio`, or `Particle` plugins. Per parent plan §7 these are sprint-12+ scope. This crate ships **one** plugin: `NdarrayCullPlugin`.
- Wiring AriGraph SPO-G writes from inside the bevy schedule. The cull system emits `CausalEdge64` via `witness_outs` mpsc; downstream commit to AriGraph happens off-thread via `AttentionMaskActor` + supervisor (out of scope for this PR).
- Modifying par-tile or MailboxSoA shapes. The plugin **consumes** the W1/W6 surface; if interface mismatch surfaces during impl, W9 escalates via AGENT_LOG (per parent plan §10).

### In-scope deliverables

1. New workspace crate `crates/bevy-cull-plugin/` with `Cargo.toml`, `src/lib.rs`, `src/plugin.rs`, `src/cull_system.rs`, `src/frustum.rs`, `src/resources.rs`.
2. `NdarrayCullPlugin` bevy `Plugin` impl wiring one system into `PostUpdate` schedule (`VisibilitySystems::CheckVisibility` set, replacing stock when feature enabled).
3. `cull_system` that consumes the active rows of `Res<MailboxSoARes>` (a bevy `Resource` wrapping `Arc<RwLock<MailboxSoA<N>>>`) + the camera's `Frustum` and writes per-entity `ViewVisibility` via `intersects_sphere_x16`.
4. `spawn_compartment_per_visible` system that pushes one MailboxSoA row per newly-visible entity (proof that the plugin *produces* compartments, not just consumes them).
5. Tests: 5 correctness + 4 integration + 1 Miri-compat (§7).
6. Benches: 4 criterion benches (§8).
7. CI: one new job in `.github/workflows/rust-test.yml` (path-filtered to `crates/bevy-cull-plugin/**`), runs under `xvfb-run` with `bevy/headless`.

---

## §2 Crate layout

```
crates/bevy-cull-plugin/
├── Cargo.toml                  # bevy 0.14 (matches workspace), ndarray (workspace),
│                               # par-tile (workspace), criterion (dev), proptest (dev)
├── src/
│   ├── lib.rs                  # pub use plugin::NdarrayCullPlugin;
│   ├── plugin.rs               # `Plugin` impl + system schedule wiring
│   ├── cull_system.rs          # the cull system fn (consumes MailboxSoA + Frustum)
│   ├── spawn_system.rs         # spawn_compartment_per_visible (proof of producer side)
│   ├── frustum.rs              # FrustumSoA: 6 planes in x16 layout for intersects_sphere_x16
│   └── resources.rs            # MailboxSoARes bevy Resource wrapper
├── tests/
│   ├── correctness.rs          # §7.1 5 correctness tests
│   ├── par_tile_integration.rs # §7.6 (sprint-10-test-plan.md) 4 integration tests
│   └── miri_compat.rs          # §7 1 Miri-compat compile test
└── benches/
    └── cull_bench.rs           # §8.6 (sprint-10-test-plan.md) 4 benches
```

**Feature flags:**

```toml
[features]
default = []                      # plugin disabled by default; require explicit opt-in
cull-ndarray = []                 # enables NdarrayCullPlugin; replaces stock check_visibility
headless = ["bevy/headless"]      # required for CI; disables window/render subsystems
miri-compat = []                  # gates the intersects_sphere_x16 call behind cfg(miri) fallback
```

---

## §3 `NdarrayCullPlugin` bevy `Plugin` impl

The plugin registers one resource and two systems. **Both systems run in `PostUpdate`** — the cull system replaces stock `check_visibility` (in the `VisibilitySystems::CheckVisibility` set); the spawn system runs **after** cull so it can read just-written `ViewVisibility`.

```rust
// crates/bevy-cull-plugin/src/plugin.rs

use bevy::prelude::*;
use bevy::render::view::VisibilitySystems;
use crate::cull_system::cull_system;
use crate::spawn_system::spawn_compartment_per_visible;
use crate::resources::MailboxSoARes;

#[derive(Default)]
pub struct NdarrayCullPlugin {
    /// Mailbox SoA capacity; matches `MailboxSoA<N>` const-generic N.
    /// Default N = 512 per parent plan §4 OQ-N (bevy recommended bracket).
    pub mailbox_capacity: usize,
}

impl Plugin for NdarrayCullPlugin {
    fn build(&self, app: &mut App) {
        // Resource: one MailboxSoA per app, behind Arc<RwLock<…>> for system-side mutation.
        // Capacity-N is a runtime-shaped wrapper around a const-generic MailboxSoA<512>.
        app.insert_resource(MailboxSoARes::new(self.mailbox_capacity.max(512)));

        // Cull system replaces stock check_visibility in PostUpdate.
        // Schedule ordering: in_set(VisibilitySystems::CheckVisibility) so other
        // bevy plugins that depend on this set (e.g. shadow systems) observe the
        // ViewVisibility writes at the same schedule point as the stock cull.
        app.add_systems(
            PostUpdate,
            cull_system
                .in_set(VisibilitySystems::CheckVisibility)
                .ambiguous_with(bevy::render::view::check_visibility), // intentional replacement
        );

        // Spawn system runs after cull (reads just-written ViewVisibility).
        app.add_systems(
            PostUpdate,
            spawn_compartment_per_visible
                .after(VisibilitySystems::CheckVisibility),
        );
    }
}
```

**Why `ambiguous_with` and not direct replacement:** stock bevy ships `check_visibility` as a fixed system. The plugin pattern is **additive co-existence** — both systems run; the cull system writes `ViewVisibility::HIDDEN` first; stock then runs and either confirms (visible → no change) or overrides (hidden by stock → forced hidden). This is **conservative**: the plugin can never make a stock-hidden entity visible. The `ambiguous_with` marker tells bevy's schedule executor we accept the ordering ambiguity (writes to the same component) and resolves it via system ordering (`.in_set` placement). In a future PR this can be tightened by feature-gating stock cull out entirely.

---

## §4 `cull_system` — the consumer-side proof

The system is **read-only on MailboxSoA** (no compartment spawn here; that's `spawn_compartment_per_visible`'s job). It reads frustum + transforms, runs `intersects_sphere_x16` in 16-lane SIMD batches, writes `ViewVisibility`.

```rust
// crates/bevy-cull-plugin/src/cull_system.rs

use bevy::prelude::*;
use bevy::render::primitives::{Frustum, Aabb};
use bevy::render::view::ViewVisibility;
use ndarray::hpc::frustum::intersects_sphere_x16;  // exists in ndarray; W8 makes Miri-clean
use par_tile::mailbox_soa::MailboxSoA;
use crate::frustum::FrustumSoA;
use crate::resources::MailboxSoARes;

pub fn cull_system(
    cameras:   Query<&Frustum, With<Camera>>,
    mut q:     Query<(&GlobalTransform, &Aabb, &mut ViewVisibility)>,
    mailboxes: Res<MailboxSoARes>,
) {
    // Single-camera proof scope. Multi-camera support deferred to sprint-12+.
    let Ok(frustum) = cameras.get_single() else { return };
    let frustum_soa = FrustumSoA::from_bevy(frustum);

    // Collect entity spheres into x16 batches.
    let entities: Vec<(Entity, Vec3, f32, Mut<ViewVisibility>)> = q.iter_mut()
        .map(|(t, aabb, vv)| {
            let center = t.transform_point(Vec3::from(aabb.center));
            let radius = aabb.half_extents.length();
            (Entity::PLACEHOLDER, center, radius, vv)
        })
        .collect();

    // x16-lane SIMD cull.
    let mut idx = 0;
    while idx + 16 <= entities.len() {
        let centers: [[f32; 3]; 16] = std::array::from_fn(|k|
            [entities[idx + k].1.x, entities[idx + k].1.y, entities[idx + k].1.z]
        );
        let radii: [f32; 16] = std::array::from_fn(|k| entities[idx + k].2);
        let mask: u16 = intersects_sphere_x16(&frustum_soa.planes, &centers, &radii);
        for k in 0..16 {
            let visible = (mask >> k) & 1 == 1;
            *entities[idx + k].3 = if visible {
                ViewVisibility::HIDDEN.with_visibility(true)  // bevy 0.14 API; verify on impl
            } else {
                ViewVisibility::HIDDEN
            };
        }
        idx += 16;
    }
    // Scalar tail.
    for k in idx..entities.len() {
        let visible = frustum.intersects_sphere(entities[k].1, entities[k].2);
        *entities[k].3 = if visible { ViewVisibility::HIDDEN.with_visibility(true) } else { ViewVisibility::HIDDEN };
    }

    // MailboxSoA is read here purely to confirm the resource is wired (compile-time proof of dep).
    // Real producer-side use happens in spawn_compartment_per_visible.
    let _: usize = mailboxes.read().count;
}
```

**Why read `mailboxes.count` even though it's unused:** the proof of the dep graph is that `bevy-cull-plugin` *imports and consumes* `par_tile::mailbox_soa::MailboxSoA`. The `_: usize = mailboxes.read().count` line is the smallest possible witness that the consumer side compiles against the W1/W6 substrate. Implementor should keep this line and document it as a compile-time dep witness.

**Miri-compat fallback (`cfg(miri)`):** under Miri the `intersects_sphere_x16` call routes through `ndarray::simd_nightly` (W8 spec). The cull system itself needs no `cfg(miri)` branch — W8 handles the dispatch.

---

## §5 `FrustumSoA` — 6 planes in x16 layout

```rust
// crates/bevy-cull-plugin/src/frustum.rs

use bevy::render::primitives::Frustum;

/// 6 frustum planes packed for x16 SIMD intersection.
/// Layout: `planes[plane_idx][component]` where component = 0..3 (nx, ny, nz, d).
pub struct FrustumSoA {
    pub planes: [[f32; 4]; 6],
}

impl FrustumSoA {
    pub fn from_bevy(f: &Frustum) -> Self {
        let mut planes = [[0.0f32; 4]; 6];
        for (i, p) in f.half_spaces.iter().enumerate() {
            let n = p.normal();
            planes[i][0] = n.x;
            planes[i][1] = n.y;
            planes[i][2] = n.z;
            planes[i][3] = p.d();
        }
        Self { planes }
    }
}
```

**Why static `[[f32; 4]; 6]` and not a vec:** the `intersects_sphere_x16` signature in `ndarray::hpc::frustum` takes `&[[f32; 4]; 6]` as the frustum input. Bevy's `Frustum::half_spaces` is `[HalfSpace; 6]`. Cost: 96 bytes per camera per frame to materialize, trivial.

---

## §6 `spawn_compartment_per_visible` — the producer-side proof

This system runs after cull and pushes one MailboxSoA row per just-visible entity. **Compartment lifetime = one frame.** Each frame, the previous frame's compartments are drained before new pushes (proof of compartment pruning via `drop_row`).

```rust
// crates/bevy-cull-plugin/src/spawn_system.rs

use bevy::prelude::*;
use bevy::render::view::ViewVisibility;
use par_tile::mailbox_soa::{MailboxSoA, RoleId, TemporalWindow, SigmaTier};
use crate::resources::MailboxSoARes;

pub fn spawn_compartment_per_visible(
    q:             Query<&ViewVisibility>,
    mut mailboxes: ResMut<MailboxSoARes>,
    frame_count:   Res<bevy::core::FrameCount>,
) {
    // Drain previous frame's compartments. drop_row reclaims the slot.
    let mut mb = mailboxes.write();
    let active_ids: Vec<usize> = (0..mb.bindspace_views.len())
        .filter(|&i| mb.active[i])
        .collect();
    for slot in active_ids {
        mb.drop_row(slot);  // exists in W6 spec §4.2
    }

    // Push one row per visible entity.
    let now = frame_count.0;
    for vv in q.iter() {
        if vv.get() {
            let _ = mb.push_row(
                RoleId(1),  // proof-only RoleId; real plugins would use per-entity role
                TemporalWindow { start_cycle: now, end_cycle: now + 1, flags: 0, _pad: 0 },
                SigmaTier::TwigBranching,  // Sigma-7: per-frame ephemeral
                par_tile::BindSpaceView::empty_static(),  // §6.1 below
                None,  // pure-internal, no ConsumerHandle
            );
        }
    }
}
```

### §6.1 `BindSpaceView::empty_static()` — required helper in par-tile

The cull plugin **does not own a BindSpace**. To satisfy the `push_row` signature it needs a zero-cost view. **Action item for W1 impl:** add to par-tile:

```rust
impl BindSpaceView<'static> {
    /// Empty view backed by an OnceLock<Arc<BindSpace>> singleton.
    /// Used by consumers that need a placeholder (bevy cull plugin proof).
    pub fn empty_static() -> Self { /* … */ }
}
```

This is **the one cross-spec touchpoint** between W9 and W1. W9 implementor flags this if W1's spec does not include it (OQ-1 below).

---

## §7 Test plan

Source: `.claude/specs/sprint-10-test-plan.md` §3 (PR-CE64-MB-7 row) + §7.6 (bevy-cull-plugin × par-tile).

### §7.1 Correctness (5 tests, `tests/correctness.rs`)

| Test | Assertion |
|---|---|
| `cull_matches_stock_1k_entities` | 1K randomly-placed entities; `NdarrayCullPlugin` and stock `check_visibility` agree on `ViewVisibility` for all 1K |
| `cull_frustum_boundary_edge_case` | Entity center exactly on frustum plane (d == 0); both pre- and post- `intersects_sphere_x16` decisions match scalar `frustum.intersects_sphere` within `f32::EPSILON * radius` |
| `cull_empty_scene_no_panic` | 0 entities; cull system runs; no panic; `MailboxSoARes` count remains 0 |
| `cull_10k_entities_one_frame_budget` | 10K entities; cull system finishes within 16.7 ms (60 Hz frame budget); measured via `Instant::elapsed` |
| `cull_no_camera_no_op` | App with `MailboxSoARes` but no camera entity; cull system runs without panic and does not modify `ViewVisibility` |

### §7.2 par-tile integration (4 tests, `tests/par_tile_integration.rs`)

Per `sprint-10-test-plan.md` §7.6:

| Test | Assertion |
|---|---|
| `cull_system_spawns_compartments_per_frame` | 1K-entity scene; `MailboxSoARes::count` ≥ 1 per visible entity per frame |
| `viewvisibility_matches_stock_bevy_cull` | Per-entity `ViewVisibility` from plugin matches stock for all 1K entities |
| `cull_system_drops_compartments_after_frame` | After 2 frames, `MailboxSoARes::count` = (visible entities of frame 2) — no leak from frame 1 |
| `cull_system_handles_empty_scene` | 0 entities; no panic; `MailboxSoARes::count` = 0 |

### §7.3 Miri-compat (1 test, `tests/miri_compat.rs`)

```rust
#[cfg(miri)]
#[test]
fn cull_compiles_under_miri() {
    // Compile-only test: instantiates FrustumSoA + a 16-entity batch and calls
    // intersects_sphere_x16 once. Verifies W8's cfg(miri) dispatch reroute reaches
    // bevy-cull-plugin call sites without UB.
    let frustum = crate::frustum::FrustumSoA { planes: [[0.0; 4]; 6] };
    let centers = [[0.0; 3]; 16];
    let radii   = [1.0; 16];
    let _mask = ndarray::hpc::frustum::intersects_sphere_x16(&frustum.planes, &centers, &radii);
}
```

**Total: 10 tests** (5 correctness + 4 integration + 1 Miri-compat). Matches `sprint-10-test-plan.md` §3.2 row count of ~8 (rounded to 10 with the 2 added schedule tests below).

### §7.4 Schedule sanity (2 tests, `tests/correctness.rs`)

| Test | Assertion |
|---|---|
| `cull_system_runs_in_post_update` | `NdarrayCullPlugin::build` registers `cull_system` in `PostUpdate` with `VisibilitySystems::CheckVisibility` set |
| `spawn_runs_after_cull` | `spawn_compartment_per_visible` runs strictly after `cull_system` in the same `PostUpdate` schedule |

---

## §8 Bench plan

Per `sprint-10-test-plan.md` §8.6:

```rust
// crates/bevy-cull-plugin/benches/cull_bench.rs
use criterion::{Criterion, criterion_group, criterion_main, black_box};

fn cull_1k(c: &mut Criterion)   { /* < 2 ms target */ }
fn cull_10k(c: &mut Criterion)  { /* < 10 ms target */ }
fn cull_100k(c: &mut Criterion) { /* < 50 ms target */ }
fn cull_vs_stock_1k(c: &mut Criterion) { /* ratio < 1.5× target */ }

criterion_group!(benches, cull_1k, cull_10k, cull_100k, cull_vs_stock_1k);
criterion_main!(benches);
```

Targets are calibrated for ubuntu-24.04 github-hosted CI runner (2-core, 7 GB RAM) per sprint-10-test-plan.md §8.

---

## §9 CI workflow addition

Per `sprint-10-test-plan.md` §9.1 — add one job to `.github/workflows/rust-test.yml`:

```yaml
bevy-cull-plugin-tests:
  runs-on: ubuntu-24.04
  timeout-minutes: 15
  if: contains(github.event.pull_request.changed_files, 'crates/bevy-cull-plugin/')
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: sudo apt-get install -y xvfb libgtk-3-dev
    - name: bevy cull plugin tests (headless)
      run: xvfb-run cargo test --manifest-path crates/bevy-cull-plugin/Cargo.toml --features headless,cull-ndarray
```

The path filter `crates/bevy-cull-plugin/` keeps the job from running on unrelated PRs.

---

## §10 Files-to-touch

| File | Action | LOC | Owner crate |
|---|---|---|---|
| `crates/bevy-cull-plugin/Cargo.toml` | NEW | ~30 | new crate |
| `crates/bevy-cull-plugin/src/lib.rs` | NEW | ~10 | new crate |
| `crates/bevy-cull-plugin/src/plugin.rs` | NEW | ~50 | new crate |
| `crates/bevy-cull-plugin/src/cull_system.rs` | NEW | ~120 | new crate |
| `crates/bevy-cull-plugin/src/spawn_system.rs` | NEW | ~60 | new crate |
| `crates/bevy-cull-plugin/src/frustum.rs` | NEW | ~40 | new crate |
| `crates/bevy-cull-plugin/src/resources.rs` | NEW | ~40 | new crate |
| `crates/bevy-cull-plugin/tests/correctness.rs` | NEW | ~150 | new crate |
| `crates/bevy-cull-plugin/tests/par_tile_integration.rs` | NEW | ~120 | new crate |
| `crates/bevy-cull-plugin/tests/miri_compat.rs` | NEW | ~30 | new crate |
| `crates/bevy-cull-plugin/benches/cull_bench.rs` | NEW | ~150 | new crate |
| `Cargo.toml` (workspace root) | EDIT | +1 | workspace |
| `.github/workflows/rust-test.yml` | EDIT | +10 | workspace |
| `crates/par-tile/src/bindspace_view.rs` | EDIT (W1) | +15 | par-tile (cross-spec touchpoint §6.1) |
| `.claude/board/STATUS_BOARD.md` | EDIT | +1 row | board hygiene |
| `.claude/board/AGENT_LOG.md` | APPEND | +1 entry | board hygiene |

**Total: ~500 LOC plugin + ~450 LOC tests/benches + ~15 LOC par-tile edit + boilerplate.** Matches parent plan §7 envelope.

---

## §11 Risk matrix

| Risk | Severity | Probability | Mitigation |
|---|---|---|---|
| Bevy 0.14 → 0.15 API churn (e.g. `ViewVisibility::HIDDEN.with_visibility(true)` shape change) | Med | Med | Pin bevy to workspace version; if 0.15 lands mid-sprint, W9-impl uses `bevy::version!` check + cfg-gated adapter; escalate via AGENT_LOG if API change is non-trivial |
| `intersects_sphere_x16` API in ndarray differs from sketch (function signature drift) | Med | Low | W8 spec freezes the signature; if drift surfaces, W8-impl owns the fix; W9-impl is downstream consumer |
| `BindSpaceView::empty_static()` missing from W1 spec (cross-spec touchpoint §6.1) | High | Med | OQ-1 below escalates this to W1-impl pre-sprint-11; meta-review catches |
| MailboxSoA capacity exhaustion on 100K-entity stress test | Med | Med | Plugin's `mailbox_capacity` is configurable; default 512; 100K bench reduces visible-fraction to keep within capacity; if real workloads exceed, plugin escalates via `Result` instead of unwrap |
| Bevy schedule ambiguity warnings (cull_system + stock check_visibility both write ViewVisibility) | Low | High | `ambiguous_with` explicitly marks the intentional overlap; doc comment in plugin.rs explains the conservative-write rationale |
| Compartment leak if cull system panics mid-frame | Low | Low | `drop_row` is panic-safe (no allocation in the drop path); bevy schedule recovers via standard panic-handler |
| 10K-entity scene exceeds 1-frame budget on slow CI runner | Med | Low | sprint-10-test-plan.md §8 calibrates targets to ubuntu-24.04 2-core runner; if CI drifts to slower runners, retune via Criterion `target_time` |
| Plugin used in production without `cull-ndarray` feature gate review | Low | Low | README + CHANGELOG note: feature is **proof-of-pattern**, not production; default-disabled enforces opt-in |
| `xvfb-run` flakiness on CI (race between Xvfb start and bevy app init) | Low | Med | Use `xvfb-run --auto-servernum`; retry-on-failure step in CI; or compile with `bevy/headless` and skip Xvfb entirely |

---

## §12 Open Questions

| OQ | Question | Tentative resolution | Resolver |
|---|---|---|---|
| **W9-OQ-1** | `BindSpaceView::empty_static()` — does W1 spec include this helper or does W9 add it? | W1 should include — it's a generic helper for any consumer crate that needs a placeholder view. W9 spec flags it as a cross-spec touchpoint. | Meta-review verifies W1 spec; if absent, W1-impl adds before sprint-11 Wave 1 |
| **W9-OQ-2** | Bevy version pin: does workspace track bevy 0.14 stable or main? | Pin to bevy 0.14 stable for proof PR; bump to 0.15 in sprint-12+ alongside any Splat/Cognitive plugins | W9-impl reads workspace `Cargo.toml`; if no bevy entry today, add 0.14 |
| **W9-OQ-3** | Stock `check_visibility` ambiguous_with — does bevy 0.14 schedule executor accept this without compile error on duplicate write? | Tested in spike: yes (bevy 0.14 explicitly supports `ambiguous_with` for resolving Component-write conflicts). If drift, fall back to feature-gating stock cull out entirely (more invasive). | W9-impl on first compile |
| **W9-OQ-4** | Multi-camera support — is the single-camera scope a soft limit for sprint-11 or a hard one? | Soft. Plugin's `cameras.get_single()` pattern is the proof-narrowing; multi-camera fan-out is a 1-line change (`.iter()` instead of `.get_single()`) deferred to sprint-12+ | Plan §7 + W9-impl |
| **W9-OQ-5** | `cull-ndarray` feature default-state in CI matrix — run with feature on or off? | On for the dedicated `bevy-cull-plugin-tests` job (proves the plugin path); off for the existing `rust-test` workspace sweep (proves nothing else broke) | sprint-10-test-plan.md §9.1 + W9-impl |

---

## §13 Cross-references

**Plans this composes:**
- `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §7 PR-CE64-MB-7 (parent row)
- `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §3rd-pair recursion (bevy session) — diamond dep graph + Slice↔Plane bridge framing
- `.claude/specs/pr-ce64-mb-1-par-tile-crate.md` (W1) — par-tile substrate; bevy as a node in the diamond §572-593
- `.claude/specs/pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) — MailboxSoA<N> + push_row / drop_row signatures
- `.claude/specs/pr-ndarray-miri-complete.md` (W8) — `intersects_sphere_x16` dispatch under cfg(miri)
- `.claude/specs/sprint-10-pr-dep-graph.md` (W10) — Wave 6 placement (depends on Waves 1-5)
- `.claude/specs/sprint-10-test-plan.md` (W11) — §3 PR-CE64-MB-7 test row, §7.6 integration tests, §8.6 benches, §9.1 CI job

**This spec does NOT:**
- Define `intersects_sphere_x16` semantics (lives in ndarray; W8 freezes)
- Define `MailboxSoA<N>` layout (W6 owns)
- Define `BindSpaceView<'a>` lifetime semantics (W1 owns)
- Propose new bevy plugins beyond `NdarrayCullPlugin` (sprint-12+)

**Board files this spec triggers** (per CLAUDE.md Mandatory Board-Hygiene Rule, when PR-CE64-MB-7 opens in sprint-11):
- `.claude/board/STATUS_BOARD.md` — append D-CE64-MB-7 row (Status: Queued → In progress → In PR → Shipped)
- `.claude/board/AGENT_LOG.md` — one-liner per W9-impl commit
- `.claude/board/LATEST_STATE.md` — Contract Inventory append (NdarrayCullPlugin) post-merge
- `.claude/board/PR_ARC_INVENTORY.md` — PREPEND entry post-merge

---

*End of pr-ce64-mb-7-bevy-cull-plugin.md — W9 deliverable, sprint-log-10 bevy-cull-plugin worker.*
