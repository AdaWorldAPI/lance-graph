# PROBE-IGNITION — API inventory (Sonnet grindwork lane)

Read-only mechanical inventory. Every signature copied verbatim from source;
every claim anchored `file:line`. No cargo run. See § NOT VERIFIED for gaps.

---

## A. The gate + loop surfaces — `crates/lance-graph-supervisor/src/cycle_driver.rs`

File read in full (1811 lines).

### `run_cycle`

```rust
// cycle_driver.rs:446-458
pub async fn run_cycle<S, F>(
    sink: &S,
    fleet: &mut F,
    writer: &mut BatchWriter<Vec<u8>>,
    frame: CycleFrame,
    position_base: u64,
    watermarks: &mut HashMap<MailboxId, Option<u64>>,
    row_of: impl FnMut(MailboxId) -> u64,
) -> Result<CycleOutcome, CycleError>
where
    S: WalSink,
    F: MailboxFleet,
```

Body (`cycle_driver.rs:459-470`): calls `collect_casts(writer, frame.cycle,
position_base, row_of)` → `seal_cycle(sink, frame, collected.slots).await`
(mapping any error to `CycleError::Seal`) → `apply_sealed_transitions(fleet,
&sealed, watermarks)`. On `Ok(applied)` returns
`CycleOutcome { sealed, applied, held: collected.held }`; on
`Err((partial, cause))` returns `CycleError::Apply { partial, cause }`.
`CycleOutcome` carries: the sealed cycle (version + sparse transitions +
next stream-position base), the applied effect (advanced owners + `deferred`/
`missing` counters, watermarks already advanced), and the intents `held`
back by the ≤1-move-per-owner partition (re-stage via `restage_held`).

Error paths (`cycle_driver.rs:422-436`):
```rust
pub enum CycleError {
    /// The WAL commit failed — no owner mutated; the boxed SealFailure
    /// carries the byte-identical frozen cycle for retry via seal_cycle.
    Seal(Box<SealFailure>),
    /// A guard tripped mid-apply — the applied prefix (with its watermarks
    /// already advanced) is preserved; re-drive the tail via recover_fleet.
    Apply {
        partial: AppliedCycle,
        cause: PersistError,
    },
}
```

### `cognitive_pass` (shared body — private, `fn` not `pub fn`)

```rust
// cycle_driver.rs:490-495
fn cognitive_pass<F>(
    fleet: &F,
    owners: impl IntoIterator<Item = MailboxId>,
    writer: &mut BatchWriter<Vec<u8>>,
    mut think: impl FnMut(&F::Owner) -> Option<(StrategyOutcome, Vec<u8>)>,
) -> CognitiveWorkOutcome
where
    F: MailboxFleet,
```

Iteration (`cycle_driver.rs:501-527`):
```rust
for id in owners {
    let Some(owner) = fleet.owner(id) else {
        continue;
    };
    if owner.phase() != KanbanColumn::CognitiveWork {
        continue;
    }
    let mut did_cast = false;
    if let Some((outcome, payload)) = think(owner) {
        if emit_bootstrap_intent(
            &outcome,
            owner.mailbox_id(),
            owner.current_cycle(),
            writer,
            payload,
        )
        .is_some()
        {
            did_cast = true;
        }
    }
    if did_cast {
        cast += 1;
    } else {
        held_owners.push(id);
    }
}
```

**The missing-owner silent-skip (the OPEN caveat):** `cycle_driver.rs:501-504`
— `fleet.owner(id)` returning `None` (owner not registered in the fleet, or
not resolvable) just `continue`s with **no counter incremented anywhere**.
Contrast `apply_sealed_transitions` (P4b), which counts this case explicitly
via `AppliedCycle::missing` (`cycle_driver.rs:365-368`). `cognitive_pass` has
no analogous field — an owner silently dropped from the `owners` iterator
(e.g. `entered` in `run_cognitive_work`, or a caller-supplied re-poll list in
`run_cognitive_work_over`) leaves no trace in `CognitiveWorkOutcome` at all;
it is neither in `held_owners` nor counted as cast. This is a real gap versus
P4b's honesty discipline, not something proven safe by a test in this file.

### `shade_owner`

```rust
// cycle_driver.rs:615-620
#[must_use]
pub fn shade_owner<O: MailboxSoaOwner>(
    owner: &O,
    qualia: &QualiaI4_16D,
    mantissa: i8,
    reliability: f32,
) -> Option<StrategyOutcome>
```

Body (`cycle_driver.rs:621-635`):
```rust
let phase = owner.phase();
let gate = gate_decision_i4(qualia, mantissa);
let to = phase.advance_on_gate(&gate)?;
Some(StrategyOutcome {
    reliability,
    intended_move: Some(KanbanMove {
        mailbox: 0,               // bootstrap sentinel
        from: phase,
        to,
        witness_chain_position: 0,
        exec: ExecTarget::Native,
    }),
})
```
`gate_decision_i4` returns `GateDecision::{Flow, Hold{reason}, Block{reason}}`
(see § B). `KanbanColumn::advance_on_gate(&GateDecision) -> Option<KanbanColumn>`
is the DAG lowering — its own definition was **not** read in this file (it
lives in `lance_graph_contract::kanban`; not opened this pass — see § NOT
VERIFIED). From the test evidence at `cycle_driver.rs:1682-1726`: `Flow` at
`CognitiveWork` → `Evaluation` ("forward"); `Block` at `Planning` → `Prune`
("Prune-where-legal"); `Hold` (or no legal successor, e.g. `Block`/`Flow` at
the absorbing `Commit` column) → `None`.

### `run_cognitive_work_gated` / `run_cognitive_work_gated_over`

```rust
// cycle_driver.rs:644-649
pub fn run_cognitive_work_gated<F>(
    fleet: &F,
    applied: &AppliedCycle,
    writer: &mut BatchWriter<Vec<u8>>,
    mut read_gate: impl FnMut(&F::Owner) -> Option<(QualiaI4_16D, i8, f32, Vec<u8>)>,
) -> CognitiveWorkOutcome
where
    F: MailboxFleet,
```
Body (`cycle_driver.rs:653-657`) delegates to `run_cognitive_work(fleet,
applied, writer, |owner| { let (qualia, mantissa, reliability, payload) =
read_gate(owner)?; let outcome = shade_owner(owner, &qualia, mantissa,
reliability)?; Some((outcome, payload)) })`. The extractor closure's exact
type is `impl FnMut(&F::Owner) -> Option<(QualiaI4_16D, i8, f32, Vec<u8>)>`
— tuple order is `(qualia, signed_mantissa, reliability, payload)`.

```rust
// cycle_driver.rs:662-667
pub fn run_cognitive_work_gated_over<F>(
    fleet: &F,
    owners: &[MailboxId],
    writer: &mut BatchWriter<Vec<u8>>,
    mut read_gate: impl FnMut(&F::Owner) -> Option<(QualiaI4_16D, i8, f32, Vec<u8>)>,
) -> CognitiveWorkOutcome
where
    F: MailboxFleet,
```
Same extractor closure shape; delegates to `run_cognitive_work_over` with the
identical `shade_owner`-wrapping closure (`cycle_driver.rs:671-675`).

`held_owners` (on `CognitiveWorkOutcome`, `cycle_driver.rs:475-484`): owners
evaluated this pass that produced **no cast** — a gate `Hold`, a declined/
unfinished thought (`think`/`read_gate` returned `None`), or a `None` from
`shade_owner` (no legal successor). Doc explicitly: "A Hold is a reschedule,
not a strand" — feed `held_owners` back into `run_cognitive_work_over` /
`run_cognitive_work_gated_over` on a later cycle.

### `MailboxFleet` trait

```rust
// cycle_driver.rs:179-186
pub trait MailboxFleet {
    type Owner: MailboxSoaOwner;
    fn owner(&self, id: MailboxId) -> Option<&Self::Owner>;
    fn owner_mut(&mut self, id: MailboxId) -> Option<&mut Self::Owner>;
}
```

`HashMap` blanket impl (`cycle_driver.rs:190-198`):
```rust
impl<O: MailboxSoaOwner> MailboxFleet for HashMap<MailboxId, O> {
    type Owner = O;
    fn owner(&self, id: MailboxId) -> Option<&O> {
        self.get(&id)
    }
    fn owner_mut(&mut self, id: MailboxId) -> Option<&mut O> {
        self.get_mut(&id)
    }
}
```
Bound: `O: MailboxSoaOwner` only — no `Hash`/`Eq` bound stated explicitly on
`O` (those are already required transitively by `HashMap<MailboxId, O>`
itself needing `MailboxId: Hash + Eq`, not `O`). `MailboxId` is
`lance_graph_contract::collapse_gate::MailboxId` (imported
`cycle_driver.rs:68`; underlying type not re-verified in this pass — see
`u32` assumption noted where `u64::from` is used as `row_of` in tests, e.g.
`cycle_driver.rs:933,957` etc., consistent with `MailboxId = u32`).

### Public struct fields — `CycleOutcome`, `SealFailure`, `CycleError`, `HeldIntent`, `CollectedCasts`

```rust
// cycle_driver.rs:412-420
pub struct CycleOutcome {
    pub sealed: SealedCycle,
    pub applied: AppliedCycle,
    pub held: Vec<HeldIntent>,
}
```

```rust
// cycle_driver.rs:120-129
pub struct SealFailure {
    pub frame: CycleFrame,
    pub casts: Vec<SweepSlot>,
    pub cause: PersistError,
}
```

```rust
// cycle_driver.rs:422-436  (see full body above under run_cycle)
pub enum CycleError {
    Seal(Box<SealFailure>),
    Apply { partial: AppliedCycle, cause: PersistError },
}
```

```rust
// cycle_driver.rs:152-158
pub struct HeldIntent {
    pub owner: MailboxId,
    pub mv: KanbanMove,
}
```

```rust
// cycle_driver.rs:163-170
pub struct CollectedCasts {
    pub slots: Vec<SweepSlot>,
    pub held: Vec<HeldIntent>,
}
```

Also relevant (referenced throughout, not explicitly requested but load-bearing):
```rust
// cycle_driver.rs:87-96  SealedTransition
pub struct SealedTransition {
    pub stream_position: u64,
    pub owner: MailboxId,
    pub mv: KanbanMove,
}
// cycle_driver.rs:103-114  SealedCycle
pub struct SealedCycle {
    pub version: DatasetVersion,
    pub transitions: Vec<SealedTransition>,
    pub next_position_base: u64,
}
// cycle_driver.rs:133-148  AppliedCycle
pub struct AppliedCycle {
    pub version: DatasetVersion,
    pub applied: Vec<KanbanMove>,
    pub deferred: usize,
    pub missing: usize,
}
```

### `restage_held`

```rust
// cycle_driver.rs:261
pub fn restage_held(writer: &mut BatchWriter<Vec<u8>>, held: Vec<HeldIntent>) -> usize
```
Body (`cycle_driver.rs:262-267`): for each `HeldIntent`, calls
`writer.cast(h.owner, vec![h.mv], Vec::new())` (intent-only re-cast — empty
payload, since the original cast's payload already sealed with its cycle).
Returns `held.len()`.

---

## B. The style + qualia surfaces

### `resolve_style`

```rust
// crates/lance-graph-planner/src/strategy/style_strategy.rs:231
fn resolve_style(ctx: &PlanContext) -> ThinkingStyle
```
(private `fn`, not `pub`). Body (`style_strategy.rs:232-251`): reads
`ctx.thinking_style: &Option<Vec<f64>>` — filters out `None` and empty
vectors, returning `DEFAULT_STYLE` (= `ThinkingStyle::Analytical`,
`style_strategy.rs:46`) for either. Otherwise reads exactly three indices of
the 23D vector — **the same axes `selector.rs::style_alignment` uses**:
```rust
let analytical = v.get(4).copied().unwrap_or(0.0);
let creative = v.get(3).copied().unwrap_or(0.0);
let depth = v.first().copied().unwrap_or(0.0);   // index 0
let max = analytical.max(creative).max(depth);
if max <= 0.0 {
    DEFAULT_STYLE
} else if (analytical - max).abs() < f64::EPSILON {
    ThinkingStyle::Analytical   // Analytical cluster → TruthAwareInference
} else if (creative - max).abs() < f64::EPSILON {
    ThinkingStyle::Creative     // Creative cluster → StructuralDivergence
} else {
    ThinkingStyle::Reflective   // depth-dominant → Meta cluster → Infrastructure
}
```
Explicit doc note (`style_strategy.rs:229-230`): this is **not** the contract
`style_vector`/i4-32D `StyleRecipe` surface — a separate, deferred decode.

### `reliability_for`

```rust
// style_strategy.rs:328
pub fn reliability_for(style: ThinkingStyle, ctx: &PlanContext) -> f32
```
Body (`style_strategy.rs:329-332`):
```rust
match ctx.witness.as_ref().and_then(|w| w.rung()) {
    Some(rung) => Self::reliability_at(style, ctx, rung),
    None => Self::reliability_of(style, ctx),   // unstratified fallback
}
```
`WitnessWindow::rung()` (`traits.rs:121-128`) returns `Some(RungLevel)` only
on `WaveGrounding::Causal` (via `RungLevel::for_pass(settle_pass)`); `None` on
`Escalate`/`Unbound` — absence must never be read as `RungLevel::Surface`.

Related entry points on `StyleStrategy` (`style_strategy.rs:306,361`):
- `reliability_of(style, ctx) -> f32` — unstratified, calls
  `reliability_at(style, ctx, RungLevel::Transcendent)`.
- `reliability_at(style: ThinkingStyle, ctx: &PlanContext, rung: RungLevel) -> f32`
  — builds `ThoughtCtx` via `thought_ctx_from(ctx)`, runs every kernel in
  `recipes_for_at(style, rung)` (`.run(&mut tc)`, mutating `tc.confidence`),
  returns `tc.confidence.clamp(0.0, 1.0)`.

### `intended_move`

```rust
// style_strategy.rs:391
fn intended_move(_style: ThinkingStyle) -> KanbanMove
```
(private `fn`). Body (`style_strategy.rs:392-398`) — constant regardless of
`style` (the `_style` param is unused, per the doc: the move is a *structural
constant of the Planning→CognitiveWork crossing*, not style-conditioned):
```rust
KanbanMove {
    mailbox: 0,
    from: KanbanColumn::Planning,
    to: KanbanColumn::CognitiveWork,
    witness_chain_position: 0,
    exec: ExecTarget::Elixir,
}
```

### `PlanInput` / `StrategyOutcome` — `crates/lance-graph-planner/src/traits.rs`

```rust
// traits.rs:193-203
pub struct PlanInput {
    pub plan: Option<LogicalPlan>,
    pub context: PlanContext,
    pub outcome: Option<StrategyOutcome>,
}
```

```rust
// traits.rs:181-190
pub struct StrategyOutcome {
    pub reliability: f32,
    pub intended_move: Option<KanbanMove>,
}
```
Derives: `Debug, Clone, Copy, PartialEq` (`traits.rs:181`).

```rust
// traits.rs:132-152
pub struct PlanContext {
    pub query: String,
    pub features: QueryFeatures,
    pub free_will_modifier: f64,
    pub thinking_style: Option<Vec<f64>>,
    pub nars_hint: Option<crate::thinking::NarsInferenceType>,
    pub witness: Option<WitnessWindow>,
}
```
`context` on `PlanInput` is a plain owned `PlanContext` (not generic/typed
beyond this struct); constructed by callers directly as a struct literal
(e.g. test helper `ctx_with` at `style_strategy.rs:450-459`).

```rust
// traits.rs:88-98
pub struct WitnessWindow {
    pub rows: Vec<(usize, CausalWitnessFacet)>,
    pub focal_idx: usize,
    pub locus: Locus,
    pub passes: u8,
}
```

### `gate_decision_i4` / `QualiaI4_16D` / `GateDecision` — `lance_graph_contract::mul::i4_eval`

```rust
// crates/lance-graph-contract/src/mul.rs:575
pub fn gate_decision_i4(qualia: &QualiaI4_16D, signed_mantissa: i8) -> GateDecision
```
Module: `pub mod i4_eval` at `mul.rs:448`, function is `#[inline]`, heap-free
except the `String` reasons on `Hold`/`Block`. Body composes
`trust_texture_i4(qualia)` + `flow_state_i4(qualia, signed_mantissa)`
(`mul.rs:511,543`) via a `match (texture, flow)` (`mul.rs:579-599`):
`Uncertain → Block`; `Underconfident + Anxiety → Block`; `Overconfident → Hold`;
`_ + Anxiety → Hold`; `(Calibrated|Underconfident) + (Flow|Transition) → Flow`;
else `Hold`.

```rust
// mul.rs:144-151
pub enum GateDecision {
    Flow,
    Hold { reason: String },
    Block { reason: String },
}
```
Cannot be `#[repr(u8)]` (carries `String` payloads); `to_disc(&self) -> u8`
(`mul.rs:158-164`) maps `Flow=0, Hold=1, Block=2` (locked mapping, D-CSV-13b).

`QualiaI4_16D` — `crates/lance-graph-contract/src/qualia.rs:173-263`:
```rust
#[repr(C, align(8))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct QualiaI4_16D(pub u64);
```
8 bytes; 16 dims × i4 (range −8..+7), one dim per nibble (`QUALIA_I4_DIMS = 16`,
`qualia.rs:140`; labels `QUALIA_I4_LABELS`, `qualia.rs:146-163` — matches first
16 of the canonical 17 `AXIS_LABELS`, "integration" dim 16 dropped).
Construction: `QualiaI4_16D::ZERO` (`qualia.rs:178`), `.with(dim: usize, value:
i8) -> Self` builder (`qualia.rs:207-211`, clamps to −8..+7), `.set(dim, value)`
in-place mutator (`qualia.rs:195-203`), `.get(dim) -> i8` sign-extending reader
(`qualia.rs:184-190`), `from_f32_17d(&QualiaVector) -> Self` /
`to_f32_17d(self) -> QualiaVector` round-trip converters
(`qualia.rs:219-251`), `.magnitude(self) -> i8` = `coherence(dim9)
.saturating_mul(valence(dim1))` (`qualia.rs:258-262`).

---

### `MetaWord` — `lance_graph_contract::cognitive_shader`

```rust
// crates/lance-graph-contract/src/cognitive_shader.rs:42-44
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct MetaWord(pub u32);
```
Doc (`cognitive_shader.rs:38-41`): "Packed u32 per row: `thinking(6) +
awareness(4) + nars_f(8) + nars_c(8) + free_e(6)`." Bit layout
(`cognitive_shader.rs:46-76`):
```rust
pub const fn new(thinking: u8, awareness: u8, nars_f: u8, nars_c: u8, free_e: u8) -> Self
// thinking  : bits 0..6   (mask 0x3F)
// awareness : bits 6..10  (mask 0x0F << 6)
// nars_f    : bits 10..18 (u8 << 10)
// nars_c    : bits 18..26 (u8 << 18)
// free_e    : bits 26..32 (mask 0x3F << 26)
```
Getters: `.thinking() -> u8`, `.awareness() -> u8`, `.nars_f() -> u8`,
`.nars_c() -> u8`, `.free_e() -> u8` (`cognitive_shader.rs:56-75`), each
masking/shifting the packed `u32`.

**ThinkingStyle mapping — NOT the contract's 36-style `ThinkingStyle` enum.**
Grepped every `MetaWord::new(...)` call site plus `auto_style` (the only
producer with a real style semantic, `cognitive-shader-driver/src/
engine_bridge.rs:295-298,760`): the `thinking` field is populated from a
**separate, local 6-bit ordinal registry**,
`cognitive_shader_driver::auto_style` (`crates/cognitive-shader-driver/src/
auto_style.rs:23-34`):
```rust
pub const DELIBERATE: u8 = 0;
pub const ANALYTICAL: u8 = 1;
pub const CONVERGENT: u8 = 2;
pub const SYSTEMATIC: u8 = 3;
pub const CREATIVE: u8 = 4;
pub const DIVERGENT: u8 = 5;
pub const EXPLORATORY: u8 = 6;
pub const FOCUSED: u8 = 7;
pub const DIFFUSE: u8 = 8;
pub const PERIPHERAL: u8 = 9;
pub const INTUITIVE: u8 = 10;
pub const METACOGNITIVE: u8 = 11;
```
Doc comment on this table (`auto_style.rs:21-22`): "0..11 matches
`thinking_engine::cognitive_stack::ThinkingStyle::all()`" — a **third**,
different `ThinkingStyle` type (in the separate `thinking-engine` crate), not
`lance_graph_contract::thinking::ThinkingStyle` (the 36-style / `StyleCluster`
enum `style_strategy.rs` uses) and not the `StyleFamily` used elsewhere in the
contract. `style_from_qualia(q: &[f32]) -> u8` (`auto_style.rs:37`+) derives
the ordinal from a qualia shape (dominant-axis heuristic among certainty/
arousal/urgency/depth/valence), never from a `lance_graph_contract::thinking::
ThinkingStyle` value. **No code path was found in this pass that writes
`lance_graph_contract::thinking::ThinkingStyle` (or its `cluster()`/`tau()`)
into a `MetaWord`.** `MetaFilter::thinking_mask: u64` (bitset over 64 possible
`auto_style` ordinals; `cognitive_shader.rs:81-107`) is the corresponding
read-side prefilter, AND-combined with `awareness_min`/`nars_f_min`/
`nars_c_min`/`free_e_max`.

---

## C. The owner/tenant surfaces — `crates/cognitive-shader-driver/src/mailbox_soa.rs`

Targeted reads (constructor, write/populate surface, contract-trait impls,
qualia/energy/meta accessors); not read end-to-end (file is large, per the
brief).

### `MailboxSoA<N>` — constructor

```rust
// mailbox_soa.rs:58   struct decl (generic param)
pub struct MailboxSoA<const N: usize> { /* ... */ }

// mailbox_soa.rs:232-233
pub type DefaultMailboxSoA = MailboxSoA<1024>;

// mailbox_soa.rs:292
pub fn new(mailbox_id: MailboxId, w_slot: u8, threshold: f32) -> Self
```
Panics (`mailbox_soa.rs:293-296`) if `w_slot >= 64` ("w_slot must fit in 6
bits (0..=63 per plan §6 L-6), got {w_slot}"). Zero-initializes every column
(`energy`, `plasticity_counter`, `last_active_cycle`/`last_write_cycle` to
`u32::MAX` sentinels, `current_cycle = 0`, `edges/qualia/meta/entity_type`,
`temporal/expert/sigma`, heap-allocated `content`/`topic`/`angle` planes of
`N * WORDS_PER_FP` `u64` each, `frozen_style/learned_style/explore_style`
`[[0u8;12]; N]`, `populated = 0`, `phase: KanbanColumn::Planning`)
(`mailbox_soa.rs:297-334`).

### `write_row` / `WriteOutcome`

```rust
// mailbox_soa.rs:417
pub fn write_row(&mut self, row: usize, cycle: u32, cell: &WriteCell<'_>) -> WriteOutcome
```
```rust
// mailbox_soa.rs:241-254
pub enum WriteOutcome {
    /// cycle == current_cycle — cell applied, last_write_cycle[row] stamped.
    Accepted,
    /// cycle strictly behind current_cycle (wrap-aware) — nothing mutated,
    /// stale_write_count incremented.
    Stale,
    /// cycle strictly ahead of current_cycle (wrap-aware) — nothing mutated.
    Future,
}
```
Gate logic (`mailbox_soa.rs:417-463`): `row >= N` → `Stale` (no mutation, "a
row we do not own is never written"). Otherwise wrap-aware delta =
`self.current_cycle.wrapping_sub(cycle)`: `delta == 0` → apply every `Some`
field of `cell` via the per-column setters (`set_content`/`set_topic`/
`set_angle`/`set_edge`/`set_qualia`/`set_meta`/`set_entity_type`/
`set_temporal`/`set_expert`/`set_sigma`), stamp `last_write_cycle[row] =
cycle`, return `Accepted`; `delta < 0x8000_0000` → `stale_write_count`
saturating-incremented, return `Stale`; else → `Future` (no mutation).

`WriteCell<'a>` (`mailbox_soa.rs:262-283`, `#[derive(Debug, Clone, Default)]`):
```rust
pub struct WriteCell<'a> {
    pub content: Option<&'a [u64]>,      // WORDS_PER_FP u64, borrowed
    pub topic: Option<&'a [u64]>,
    pub angle: Option<&'a [u64]>,
    pub edge: Option<CausalEdge64>,
    pub qualia: Option<QualiaI4_16D>,
    pub meta: Option<MetaWord>,
    pub entity_type: Option<u16>,
    pub temporal: Option<u64>,
    pub expert: Option<u16>,
    pub sigma: Option<u8>,
}
```

### `set_populated` / `current_cycle`

```rust
// mailbox_soa.rs:495
pub fn set_populated(&mut self, n: usize)   // = n.min(N); a DECLARATION, not an implicit counter
// mailbox_soa.rs:486
pub fn populated(&self) -> usize
```
`current_cycle` is read via the `MailboxSoaView::current_cycle(&self) -> u32`
trait impl (`mailbox_soa.rs:872-874`, returns `self.current_cycle`); advanced
via `pub fn tick(&mut self)` (`mailbox_soa.rs:399-401`,
`current_cycle.wrapping_add(1)`) — **not** a field named `current_cycle()` as
an inherent method; the inherent field is `pub(crate)`-scoped implicitly
through the struct (not confirmed `pub` — see § NOT VERIFIED) and reached
through the trait method in all call sites grepped.

### `MailboxSoaOwner` / `MailboxSoaView` impls for `MailboxSoA<N>`

```rust
// mailbox_soa.rs:852   impl block header
impl<const N: usize> MailboxSoaView for MailboxSoA<N> {
    fn mailbox_id(&self) -> MailboxId { self.mailbox_id }                        // :854-856
    fn n_rows(&self) -> usize { self.populated }                                  // :858-866 (NOT N — populated)
    fn w_slot(&self) -> u8 { self.w_slot }                                        // :868-870
    fn current_cycle(&self) -> u32 { self.current_cycle }                        // :872-874
    fn phase(&self) -> KanbanColumn { self.phase }                                // :876-878
    fn identity_plane_at(&self, row: usize, plane: IdentityPlane) -> Option<&[u64]> // :886-895
    fn style_lane_at(&self, row: usize, lane: StyleLane) -> Option<[u8; 12]>      // :902-911
    fn energy(&self) -> &[f32] { &self.energy }                                   // :913-915
    fn edges_raw(&self) -> &[u64]   // unsafe repr(transparent) cast, :917-931
    fn meta_raw(&self) -> &[u32]    // unsafe repr(transparent) cast, :934-941
    fn entity_type(&self) -> &[u16] { &self.entity_type }                         // :944-946
}
```
`identity_plane_at`/`style_lane_at` both guard `row >= self.populated` →
`None` (never reads a zero-padded capacity row) before dispatching to
`content_row`/`topic_row`/`angle_row` or `frozen_style`/`learned_style`/
`explore_style`.

```rust
// mailbox_soa.rs:949
impl<const N: usize> MailboxSoaOwner for MailboxSoA<N> {
    fn advance_phase(&mut self, to: KanbanColumn) -> KanbanMove   // :953-973
}
```
Body: `from = self.phase; self.phase = to;` then constructs `KanbanMove {
mailbox: self.mailbox_id, from, to, witness_chain_position:
self.current_cycle, exec: ExecTarget::Native }`. **`try_advance_phase` is NOT
overridden here** — `MailboxSoA<N>` uses the trait's DEFAULT impl from
`lance_graph_contract::soa_view::MailboxSoaOwner` (see below); no
`fn try_advance_phase` appears anywhere in `mailbox_soa.rs` (grep returned
zero hits in this file).

### `MailboxSoaOwner`/`MailboxSoaView` trait definitions —
`crates/lance-graph-contract/src/soa_view.rs`

```rust
// soa_view.rs:67-89 (required methods only; several defaulted methods omitted, see below)
pub trait MailboxSoaView {
    fn mailbox_id(&self) -> MailboxId;
    fn n_rows(&self) -> usize;
    fn w_slot(&self) -> u8;
    fn current_cycle(&self) -> u32;
    fn phase(&self) -> KanbanColumn;
    fn energy(&self) -> &[f32];
    fn edges_raw(&self) -> &[u64];
    fn meta_raw(&self) -> &[u32];
    fn entity_type(&self) -> &[u16];
    // defaulted (deferred-binding, all return None unless overridden):
    fn class_id(&self) -> &[u16] { self.entity_type() }                    // :99-102
    fn class_id_at(&self, row: usize) -> u16 { self.entity_type()[row] }    // :105-108
    fn row_for_local_key(&self, _local_key: u64) -> Option<usize> { None } // :125-128
    fn hhtl_path_at(&self, _row: usize) -> Option<crate::hhtl::NiblePath> { None } // :143-146
    fn edge_block_at(&self, _row: usize) -> Option<crate::canonical_node::EdgeBlock> { None } // :162-165
    fn identity_plane_at(&self, _row: usize, _plane: IdentityPlane) -> Option<&[u64]> { None } // :176-179
    fn style_lane_at(&self, _row: usize, _lane: StyleLane) -> Option<[u8; 12]> { None } // :195-198
    fn triangle_at(&self, row: usize, family: u8) -> Option<(u8, u8, u8)> { /* composes style_lane_at ×3 */ } // :209-219
    fn style_rails_at(&self, row: usize, lane: StyleLane) -> Option<[(u8,u8);6]> { /* composes style_lane_at */ } // :240-251
    fn energy_at(&self, row: usize) -> f32 { self.energy()[row] }           // :283-286
}
```
**No `fn qualia(&self)` on this trait.** Explicit comment
(`soa_view.rs:253-255`): "the qualia column (`QualiaI4_16D`) accessor is
intentionally omitted — add `fn qualia(&self) -> &[crate::qualia::
QualiaI4_16D]` when the first consumer (planner strategy selection) needs
it; keep the read surface minimal until then." This directly matches
`cycle_driver.rs`'s own doc note that `shade_owner`'s qualia/mantissa are
caller-supplied because `MailboxSoaView` does not yet expose `qualia()`.
Similarly no `episodic_witness` accessor yet (`soa_view.rs:257-277`, deferred
for `EpisodicWitness64`, not yet a code symbol).

```rust
// soa_view.rs:295-321
pub trait MailboxSoaOwner: MailboxSoaView {
    fn advance_phase(&mut self, to: KanbanColumn) -> KanbanMove;

    fn try_advance_phase(
        &mut self,
        to: KanbanColumn,
    ) -> Result<KanbanMove, RubiconTransitionError> {
        let from = self.phase();
        if from.can_transition_to(to) {
            Ok(self.advance_phase(to))
        } else {
            Err(RubiconTransitionError { from, to })
        }
    }
}
```
`try_advance_phase` is a **default trait method** — checks
`KanbanColumn::can_transition_to` before calling the (required, unchecked)
`advance_phase`; returns `RubiconTransitionError { from, to }` on an illegal
edge with no mutation. `MailboxSoA<N>` inherits this default unmodified.

### `MailboxSoaView` read accessors used by `blw_fusion.rs`-style consumers

`identity_plane_at` is the accessor `blw_fusion.rs` and any Hamming/CAM
distance reader would use (confirmed present and overridden on
`MailboxSoA<N>`, `mailbox_soa.rs:886-895`, guarded by `populated`). No other
"identity_plane_at etc." read accessors beyond `style_lane_at`/`triangle_at`/
`style_rails_at`/`energy_at` were found on the trait (full list above is
exhaustive for this file).

---

## D. The seeding surfaces

### `examples/blw_fusion.rs` — seed/seal loop call sequence

Landed 2026-08-04 (per `AGENT_LOG.md:1-8`). Key calls, in execution order,
with line numbers:

```
blw_fusion.rs:723   let mut owner: Tenant = MailboxSoA::new(TENANT_ID, TENANT_W_SLOT, TENANT_THRESHOLD);
blw_fusion.rs:726   seed_slice(&mut owner, 0, &verses[0..SLICE])        -- slice 1, BEFORE the loop
blw_fusion.rs:727   owner.set_populated(seated_total);
blw_fusion.rs:728   owner.tick();                                       -- cycle 0 -> 1
blw_fusion.rs:730   let sink = MemWal::new();                            -- local WalSink impl, mirrors persist_sink::FakeWalSink
blw_fusion.rs:731   let mut writer: BatchWriter<RowSpanDescriptor> = BatchWriter::new();

-- per cycle c in 1..=8 (plan: KanbanColumn DAG Planning->CognitiveWork->Evaluation->Plan->Planning, twice) --
blw_fusion.rs:791   assert_eq!(owner.phase(), spec.from, ...)
blw_fusion.rs:797   seed_slice(&mut owner, (c-1)*SLICE, &verses[(c-1)*SLICE..c*SLICE])   -- c > 1 only
blw_fusion.rs:803   owner.set_populated(seated_total);
blw_fusion.rs:817   rank_verdicts(&owner, seated_total, &seed_a)          -- score+verdict FULL pool
blw_fusion.rs:818   rank_verdicts(&owner, seated_total, &seed_b)
blw_fusion.rs:819-821  contains_all(&owner, row, &god_probe) per row      -- verdict_z
blw_fusion.rs:835   emit_bootstrap_intent(&outcome, owner.mailbox_id(), owner_cycle, &mut writer, span)
blw_fusion.rs:837   writer.on_behalf_of(cast)
blw_fusion.rs:846   writer.intent_moves(cast)
blw_fusion.rs:854-861  build one SweepSlot { cycle: spec.id, stream_position: c as u64,
                        owner: cast_owner, row: 0, paired_move: Some(cast_move),
                        payload: span.to_le_bytes().to_vec() }
blw_fusion.rs:864   persist_cycle(&sink, CycleFrame::new(spec.id, base), slots).await?
blw_fusion.rs:871   sink.scan_sealed(Some(base)).await?
blw_fusion.rs:872   recover_and_apply(&mut owner, &sealed, watermark).map_err(|(_, e)| e)?
blw_fusion.rs:873   watermark = recovered.watermark;
blw_fusion.rs:930   owner.tick();
```

**Load-bearing finding:** this example does **not** call
`cycle_driver::seal_cycle` / `cycle_driver::apply_sealed_transitions` /
`cycle_driver::run_cycle` at all. It calls the lower-level
`lance_graph_planner::persist_sink::{persist_cycle, recover_and_apply}`
directly (confirmed import at `blw_fusion.rs:101`: `SweepSlot, WalSink,
WriteFailed` from that module, plus `persist_cycle`/`recover_and_apply` used
inline — exact `use` line for those two symbols not captured in this pass,
see § NOT VERIFIED) and its own local `MemWal` (`blw_fusion.rs:393-478` per
the earlier grep, `impl WalSink for MemWal` at `:422`) rather than
`cycle_driver::FakeWalSink`. The lifecycle intent is built by hand via
`bootstrap_intent(from, to) -> KanbanMove` (`blw_fusion.rs:530-538`, mailbox
0, witness_chain_position 0, `ExecTarget::Elixir`) and staged through
`emit_bootstrap_intent` (from `lance_graph_planner::owner_adapter`, same
function `cycle_driver.rs` uses in `cognitive_pass`) — so the P4c
rebind-and-cast seam is shared, but the P4a/P4b seal+apply seam
(`collect_casts`/`seal_cycle`/`apply_sealed_transitions`) is **not**
exercised by this example; it reimplements an equivalent single-slot
seal/apply by hand each cycle.

### `crates/deepnsm-v2/src/` — lib surface for verses/triplets

`lib.rs` module list (`crates/deepnsm-v2/src/lib.rs:37-50`): `ancestry`,
`basin`, `belief`, `codebook`, `corpus`, `evidence`, `fsm`, `introspect`,
`reason`, `shape`, `space`, `spo`, `vocab`, `wave`.

**`corpus` module** (`crates/deepnsm-v2/src/corpus.rs`) — text → verses:
```rust
pub const GUTENBERG_FOOTER: &str = "*** END OF THE PROJECT GUTENBERG";   // corpus.rs:13
pub const KJV_OLD_TESTAMENT_VERSES: usize = 23_145;                       // corpus.rs:24 (documentation only, not a threshold)
pub fn is_verse_marker(tok: &str) -> bool                                 // corpus.rs:28
pub fn split_verses(text: &str) -> Vec<String>                            // corpus.rs:56
pub struct CorpusSplit { pub verses: Vec<String>, pub crossed_new_testament: bool } // corpus.rs:71-81
pub fn split_verses_detailed(text: &str) -> CorpusSplit                   // corpus.rs:96
```
`split_verses` splits on `d+:d+` verse markers (e.g. `1:1`), truncating body
text at the FULL `GUTENBERG_FOOTER` string (not a bare `***`, which is only
the OT/NT separator and must not be treated as end-of-file — the historical
truncation bug this module fixes, doc at `corpus.rs:37-54`).

**`bible_wave.rs` corpus-to-triples shape** (referenced in `lib.rs:31-33` doc,
not independently opened this pass): "`examples/bible_wave.rs` runs the whole
KJV (23,145 verses = one 64k tile) through FSM → SPO → `TemporalStream`" — see
§ NOT VERIFIED (file not read directly).

**Top-level `Nsm` engine + `TemporalStream`** (`lib.rs:93-220`):
```rust
pub struct Nsm {
    pub vocab: PaletteVocab,     // frequency-ranked ROUTING address
    pub space: Cam96Space,       // CAM-PQ 96 meaning-DISTRIBUTION space
    codes: Vec<Cam96>,           // private; per-word-id 96-bit meaning code
}
impl Nsm {
    pub fn new(vocab: PaletteVocab, space: Cam96Space) -> Self;                      // lib.rs:106
    pub fn with_codes(vocab: PaletteVocab, space: Cam96Space, codes: Vec<Cam96>) -> Self; // lib.rs:118
    pub fn ingest(&self, tokens: &[Tagged]) -> Vec<Spo>;                             // lib.rs:128, delegates to fsm::parse_to_spo
    pub fn code(&self, word: &str) -> Option<&Cam96>;                                // lib.rs:135
    pub fn word_similarity(&self, a: &str, b: &str) -> Option<f32>;                  // lib.rs:143
    pub fn triple_similarity(&self, a: Spo, b: Spo) -> [Option<f32>; 3];             // lib.rs:151
}

pub struct TemporalStream { entries: Vec<(u64, Spo)> }   // lib.rs:173-176, private field
impl TemporalStream {
    pub fn new() -> Self;                                                            // lib.rs:181
    pub fn push(&mut self, version: u64, triple: Spo);                               // lib.rs:186
    pub fn window_at(&self, ref_version: u64) -> impl Iterator<Item = &Spo> + '_;     // lib.rs:202, borrowing projection via TemporalPov::at
    pub fn window_range(&self, range: VersionRange) -> impl Iterator<Item = &Spo> + '_; // lib.rs:214
}
```
`window_at`/`window_range` are explicitly documented as **borrowing
projections, never a second store** (`lib.rs:194-201`) — consistent with the
`temporal.rs` sorted-stream doctrine referenced in the top-level `CLAUDE.md`
"2026-07-10 supersession" note.

Re-exports at crate root (`lib.rs:54-77`): `FamilyTrie`; `basin_self_code,
heldout_bessel_gate, heldout_constant_n_gate, heldout_split_gate, BasinCode,
HeldOutGate`; `Belief, BeliefArena, CStmt, Copula, ReviseOutcome, Stamp`;
`load_cam96_codes, load_cam96_space, CodebookError`; `evidence_basin,
forward_gate, novelty_rate, open_question_yield, partial_spearman,
shuffle_beliefs_null, shuffle_rungs_null, EvidenceBasin, ForwardGateReport`;
`parse_to_spo, Pos, Tagged`; `confidence_delta_recount,
confidence_delta_self, most_frequent_belief, provenance_check,
ConfidenceAnswer, ProvenanceReport`; `detect, detect_all,
detect_all_measured, detect_measured, MeasuredShape, Representation,
ShapeClass, ShapeReport`; `AdcSpace, Cam96, Cam96Space, SemanticSpace`;
`Spo`; `PaletteVocab, WordId`; `WitnessStream`.

---

## NOT VERIFIED

- **`KanbanColumn::advance_on_gate`** — signature and full match arms not
  read directly in this pass (not opened; behavior inferred only from
  `cycle_driver.rs` test names/asserts at `cycle_driver.rs:1682-1726`). Lives
  in `lance_graph_contract::kanban` per the `use` at `cycle_driver.rs:69`.
- **`KanbanColumn::can_transition_to`** — referenced by `try_advance_phase`'s
  default body (`soa_view.rs:316`) and by a `style_strategy.rs` test
  (`style_strategy.rs:865`), but its own definition/match arms were not
  opened in this pass.
- **`MailboxId` underlying type** — treated as `u32` based on usage
  (`u64::from` conversions in `cycle_driver.rs` tests, `& 0x3F` masking in
  `soa_view.rs` `FakeSoa::w_slot`), but `pub type MailboxId = …` in
  `lance_graph_contract::collapse_gate` was not opened directly.
- **`BatchWriter<P>`** (`cast`, `on_behalf_of`, `intent_moves`,
  `drain_pending_payloads`) and **`emit_bootstrap_intent`**
  (`lance_graph_planner::owner_adapter`) — signatures used pervasively in
  both `cycle_driver.rs` and `blw_fusion.rs` but their own definitions were
  not opened in this pass (out of the requested A–D scope).
- **`persist_cycle` / `recover_and_apply` / `RecoveredCycle` (or whatever the
  return type of `recover_and_apply` is named)** in
  `lance_graph_planner::persist_sink` — signatures not independently
  confirmed; only their call sites in `cycle_driver.rs` tests and
  `blw_fusion.rs:864,872` were read. `recovered.watermark` and
  `recovered.applied` field names are taken on faith from
  `blw_fusion.rs:873,875,879` and `cycle_driver.rs`'s own
  `recover_fleet`/`recover_and_apply` usage (`cycle_driver.rs:725`), not from
  the type's own declaration.
- **`examples/bible_wave.rs`** — not opened in this pass; its FSM → SPO →
  `TemporalStream` pipeline is reported only via the `deepnsm-v2/src/lib.rs`
  module-doc summary (`lib.rs:26-35`), not from the example's own source.
- **`crates/deepnsm-v2/src/vocab.rs` / `spo.rs`** (`PaletteVocab`, `WordId`,
  `Spo` struct fields) — not opened; only inferred from `lib.rs` usage
  (`Spo::new(subject, predicate, object)`-shaped calls in `lib.rs` tests) and
  the `blw_fusion.rs`/`cycle_driver.rs` grep results.
- **`MailboxSoA<N>` private field visibility** — `phase` is confirmed
  `pub(crate)` (`mailbox_soa.rs:229`); other fields' exact visibility
  (`energy`, `current_cycle`, etc.) were not individually confirmed as
  `pub`/`pub(crate)`/private beyond what the accessor methods imply — only
  the accessor methods themselves were verified as the intended read/write
  surface.
- **The exact `use` line(s) importing `persist_cycle`/`recover_and_apply`
  into `blw_fusion.rs`** — the import block was not fully re-read after the
  initial partial grep (`blw_fusion.rs:101` covers `SweepSlot, WalSink,
  WriteFailed` only); the two function imports are inferred from call-site
  usage, not confirmed against an explicit `use` statement.
- **`rank_verdicts` / `contains_all` / `encode_plane` / `bloom_of_terms`**
  (blw_fusion.rs helpers) — call sites read, full bodies not inventoried
  (out of scope per the brief, which asked only for the seed/seal call
  sequence).
