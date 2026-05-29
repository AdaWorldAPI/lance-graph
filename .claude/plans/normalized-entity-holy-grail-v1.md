# normalized-entity-holy-grail-v1 — typed unified normalization + Op chain over OGIT/OWL/DOLCE/Odoo with three-context execution

> **Status:** PROPOSAL. The trunk that unifies the prior workspace work
> (`odoo-business-logic-blueprint-v1` typed surface, `odoo-source-extraction-v1`
> Stage 1 extracted backing, the contract crate's `OrchestrationBridge` +
> `UnifiedStep` + `jit` substrate, PR #427 mailbox SoA, the
> `cognitive-shader-driver`) into ONE consumer-facing pipeline grammar.
> Closes the gap the recent session arc surfaced: lance-graph already has
> the thinking; consumers do not have a typed surface to consume it
> without re-implementing business logic in regex / hand-rolled pseudo-
> code.
>
> **Confidence:** HIGH on the carrier + algebra shape (one struct, 5
> verbs — direct analogue of the SoA-as-AGI unification). HIGH on the
> three-context split (interactive / bulk / periodisch are genuinely
> different SLAs with different commit semantics; Odoo's `env.context`
> flags vs lock-date wizards vs `with norecompute()` already prove the
> decomposition is real). MED on the Op-trait three-call-site
> specialisation (cold/warm/hot); the JIT path exists in
> `lance-graph-contract::jit` but has never been wired against a typed
> Op grammar. LOW on the macro-DSL layer (consumer ergonomics) — needs
> per-repo iteration once the underlying typed pipeline is stable.
>
> **Predecessors:**
> - `D-ODOO-BP-1a..g` typed `OdooEntity` surface (PR #420-ish) + Wave
>   1-3 lane modules (`l{1..15}`)
> - `D-ODOO-EXT-1..6` source-extracted backing (PR #426 merged; 12
>   TIER-1 addons, 73K LOC extracted, 1274+1192 SKR account templates,
>   37 UStVA Kennzahlen, GoBD wiring)
> - PR #411 `lance-graph-contract::jit::{JitCompiler, StyleRegistry,
>   KernelHandle}` — the hot-path compilation substrate
> - PR #427 `MailboxSoA` thoughtspace columns
>   (`edges`/`qualia`/`meta`/`entity_type`) + WitnessTable primitive +
>   §10 architectural refinements
> - `lance-graph-contract::callcenter::ogit_uris` — canonical OGIT
>   codebook
> - `lance-graph-ontology::dolce_odoo` — DOLCE classifier
> - `lance-graph-contract::orchestration::{OrchestrationBridge,
>   UnifiedStep, StepDomain, BridgeSlot}` — the bridge layer
> - `lance-graph-rbac::{SuperDomain, smb_policy}` — entry-point auth
>
> Driver epiphanies (this plan also lands all eight as
> `EPIPHANIES.md` entries):
> - `E-NORMALIZED-ENTITY-1` — single carrier holds the 4-way
>   inheritance chain
> - `E-OP-FIVE-VERBS-1` — only 5 universal verbs
>   (`resolve/hydrate/classify/align/think`)
> - `E-OP-THREE-CALLSITES-1` — same Op trait, three execution
>   speeds, shared const data
> - `E-TRANSACTION-CONTEXT-1` — interactive/bulk/periodisch own
>   commit + Baton epoch + Lance version policy
> - `E-CASCADE-AS-EDGECOLUMN-1` — dependency cascade collapses Odoo's
>   six overlapping mechanisms into one `EdgeColumn` traversal
> - `E-ODOO-AS-PRIOR-ART-1` — Odoo solved the three regimes; we
>   re-encode as compile-time-typed boundaries instead of runtime
>   `env.context` flags
> - `E-CONSUMER-CANNOT-INTERPRET-1` — business heuristics MUST be
>   SIMD-amenable const data; regex / hand-rolled `if line.account.
>   starts_with("84")` is structurally banned because the chain
>   doesn't expose that primitive
> - `E-NO-AUTOMATIC-REGIME-PICK-1` — shader does NOT autonomously
>   choose hot vs cold; the consumer's typed transaction context does
>   (a correction of the cute-but-wrong "shader picks based on flow
>   rate" framing)
>
> **Anchored iron rules:**
> - `I-VSA-IDENTITIES` (identity in const data, content in tables —
>   Op kernels are const-data identities; their kernel logic lives in
>   the shader)
> - `E-CODEBOOK-INHERITS-FROM-OGIT` (OGIT IRIs are the codebook
>   handles; the NormalizedEntity carries the OGIT slot as a
>   `&'static OgitUri`, not a parsed sub-tree)
> - `E-SAVANT-COMPOSITION-1` (savants compose over typed DTOs — this
>   plan ships the DTO shape v2 reasoners read from)
> - "AGI-as-glove" — the four SoA columns ARE the AGI surface; the
>   NormalizedEntity is a typed ROW into those columns, not a wrapper
>   around them
> - "Lab vs canonical" — the consumer surface is the typed
>   pipeline, not a REST endpoint; the macro-DSL is the ergonomic
>   skin, never a wire surface
> - "No service queries" — the chain does not call an AGI service;
>   AGI is the shader's behaviour on the SoA row the chain owns

## The diagnosis

Today the workspace has:
1. A typed business-grammar substrate (`OdooEntity`,
   `OdooAccountTemplate`, `OdooUstvaKennzahl`, `OdooGobdWiring`, savant
   role keys, NARS atoms) — **what to think about**
2. A cognitive substrate (`CognitiveShader`, `MailboxSoA`,
   `CausalEdge64`, Vsa16kF32, JIT compiler, OrchestrationBridge) —
   **how to think**
3. An RBAC + RLS substrate (`SuperDomain`, `smb_policy`,
   `UnifiedBridge`) — **whose data + can they touch it**

What it does NOT have: **one typed surface that ties the three
together into a chainable consumer pipeline**. Each consumer
(woa-rs, medcare-rs, smb-office-rs, medcarev2) builds its own ad-hoc
glue — typically as Axum handlers that call into the bridge with
JSON dicts, lose typing at the boundary, and re-implement business
heuristics (account-range checks, fiscal-position rules, VAT
liability) as regex or hand-rolled `if` chains on the consumer side.

**The structural anti-pattern this leaves available:** consumers
*can* write `if line.account.code.starts_with("84") {
post_to_revenue() }`. They shouldn't, but the type system permits
it. CodeRabbit / Codex will not catch it. The cognitive shader
becomes optional infrastructure that consumers route around when
deadlines hit.

**The Odoo prior art is the reference point:** the same three SLAs
(interactive / bulk / periodisch) are solved by `@api.depends`
strings + `with env.norecompute():` blocks + `account.fiscal.year.
close` wizards + lock-date global state. Six overlapping cascade
mechanisms; runtime-evaluated dependency strings; stringly-typed
context flags. Odoo got the decomposition right; we want to keep
the decomposition and re-encode it as compile-time typestate.

The user-named pipeline:

```text
Customer event (invoice upload / Kontodaten sync / Jahresabrechnung)
  → NormalizedEntity<Raw>                                    ← perimeter ingest
  → .resolve_ogit(ctx)    → NormalizedEntity<WithOgit>       ← codebook hit
  → .hydrate_owl(ctx)     → NormalizedEntity<WithOwl>        ← TTL join
  → .classify_dolce(ctx)  → NormalizedEntity<WithDolce>      ← upper-ontology category
  → .align_fibu(ctx)      → NormalizedEntity<Normalized>     ← domain overlay
  → .op(KontenerkennungSkr04)            ← chain begins; shader dispatch
  → .chk_data(SkrAccountInRange::new(8400..=8499))
  → .review(FiscalPositionResolver)      ← invokes the savant
  → .abduct(VatLiability)                ← NARS abduction inference
  → .op(GoBdLockCheck)
  → .report(UStvaKennzahlAggregator)
  → .output()                            ← Baton fan-out to dependents
```

Same chain shape in all three contexts; the context determines
*how* each `.method()` commits.

## The shape

### The carrier — `NormalizedEntity<S>`

One struct, four inheritance slots + the four SoA columns + a phantom
stage. Lives in `lance-graph-contract::cognition`:

```rust
pub struct NormalizedEntity<S = stages::Raw> {
    // Inheritance chain — populated as stages advance.
    pub(crate) odoo:  &'static OdooEntity,   // source-of-truth from EXT-2..6
    pub(crate) ogit:  Option<&'static OgitUri>,
    pub(crate) owl:   Option<&'static OwlClass>,
    pub(crate) dolce: Option<DolceCategory>,
    pub(crate) fibu:  Option<&'static FibuAlignment>,

    // Cognitive state — typed view into the MailboxSoA row.
    pub(crate) row:   MailboxRow,            // (mailbox_ref: u32, row_idx: u16)

    _stage: PhantomData<S>,
}
```

`row` is a typed handle into the mailbox SoA the entity lives in;
fingerprint / qualia / meta / edges access goes through that handle,
so we never duplicate state. The mailbox owns the SoA row; the
`NormalizedEntity` is a typed lens onto it. Same Baton ownership
guarantees apply (Rust move/ownership semantics make UB a compile
error per §9 E-CE64-MB-4).

### The algebra — five verbs

```rust
impl NormalizedEntity<Raw> {
    pub fn resolve_ogit(self, ctx: &impl OgitCtx) -> NormalizedEntity<WithOgit>;
}
impl NormalizedEntity<WithOgit> {
    pub fn hydrate_owl(self, ctx: &impl OwlCtx) -> NormalizedEntity<WithOwl>;
}
impl NormalizedEntity<WithOwl> {
    pub fn classify_dolce(self, ctx: &impl DolceCtx) -> NormalizedEntity<WithDolce>;
}
impl NormalizedEntity<WithDolce> {
    pub fn align_fibu(self, ctx: &impl FibuCtx) -> NormalizedEntity<Normalized>;
}
impl NormalizedEntity<Normalized> {
    // Chain begins — only typed Ops, no free-form thinking.
    pub fn op<O: Op<Normalized, Normalized>>(self, op: O) -> Self;
    pub fn chk_data<C: Op<Normalized, Checked>>(self, c: C) -> NormalizedEntity<Checked>;
    pub fn review<R: Op<Checked, Reviewed>>(self, r: R) -> NormalizedEntity<Reviewed>;
    pub fn abduct<A: Op<Reviewed, Abducted>>(self, a: A) -> NormalizedEntity<Abducted>;
    pub fn report<P: Op<Abducted, Reported>>(self, p: P) -> NormalizedEntity<Reported>;
    pub fn output(self) -> Output;
}
```

Typestate enforces order: `.abduct()` is not callable on `<Raw>` or
even `<Normalized>` — only on `<Reviewed>`. The compiler is the
review gate.

### The Op trait — three call sites, one definition

```rust
pub trait Op<I: Stage, O: Stage>: Sized + 'static {
    /// Const-data identity of this Op (the kernel handle the shader
    /// dispatches against). Per `I-VSA-IDENTITIES`, this is the
    /// register; the kernel logic lives in the shader.
    fn kind(&self) -> OpKind;

    /// Cold path — single carrier; batch / one-shot. Dispatches the
    /// shader kernel once, eagerly.
    fn apply(&self, entity: NormalizedEntity<I>, ctx: &impl Context)
        -> NormalizedEntity<O>;

    /// Warm path — async stream; one in / one out, flow-controlled.
    /// The shader runs the kernel per element with bounded
    /// parallelism; cascade Batons batch per epoch.
    fn apply_stream<S>(&self, s: S, ctx: &impl Context)
        -> impl Stream<Item = NormalizedEntity<O>>
        where S: Stream<Item = NormalizedEntity<I>>;

    /// Hot path — SoA-swept SIMD kernel over a mailbox; JIT-compiled
    /// from the const-data Op + kernel handle. No allocation, no
    /// virtual call.
    fn apply_soa(&self, mb: &mut MailboxSoA<N>, mask: BitMask,
                 ctx: &impl Context);
}
```

Same const-data Op (`SkrAccountInRange`, `VatLiability`,
`KontenerkennungSkr04`), three call sites. The shader does not pick
between them; the **transaction context** does.

### The transaction context — interactive / bulk / periodisch

Three contexts, each picks the Op call site + Baton epoch + Lance
version policy + cascade traversal mode:

| | **Interactive** | **Bulk** | **Periodisch** |
|---|---|---|---|
| Call site | `apply` (cold) | `apply_stream` (warm) | `apply_soa` (hot, JIT) |
| Lance version | live | per-batch snapshot | frozen point-in-time |
| Baton emission | eager, immediate fan-out | lazy, per-epoch flush | epochal, iterate-to-fixed-point |
| Cascade graph | sync DFS through dependent mailboxes | async, batched | JIT-compiled iteration |
| `output()` blocks on | cascade quiescence | epoch flush | fiscal-cutoff debits=credits |
| Typical example | Kunde lädt Rechnung hoch → Kontenerkennung → Posting → UStVA-Kz-Aggregation | Kontodaten-Sync nächtlich | Jahresabrechnung + UStVA-Q4 |

Sketch:
```rust
woa.interactive(|ctx| {
    invoice.into_entity()
        .resolve_ogit(ctx).hydrate_owl(ctx)
        .classify_dolce(ctx).align_fibu(ctx)
        .op(KontenerkennungSkr04)
        .chk_data(SkrAccountInRange::new(8400..=8499))
        .review(FiscalPositionResolver)
        .abduct(VatLiability)
        .op(GoBdLockCheck)
        .report(UStvaKennzahlAggregator)
        .output()  // BLOCKS on dependent cascade
});

woa.bulk(|ctx| {
    bank_statements
        .into_stream()
        .normalize_stream(ctx)  // resolve/hydrate/classify/align on stream
        .apply_stream(KontodatenSync)  // SoA-swept warm path
        .commit_at_end(ctx)  // single epoch flush
});

woa.periodisch(|ctx| {
    ctx.frozen_lance_version()
       .jit_chain(JahresabrechnungChain)
       .iterate_until(debits_eq_credits)
       .commit_as_fiscal_close()
});
```

### Cascade as EdgeColumn

Odoo's six cascade mechanisms (`@api.depends` strings + `@api.
constrains` + FK ondelete + server actions + `_inherits` forwarding +
implicit model cascades) collapse into ONE typed graph: the
`EdgeColumn` per mailbox. Each cascade dependency is one
`CausalEdge64`:
- `source_mailbox_ref` → `target_mailbox_ref`
- `kind: CascadeKind::ComputeRecompute | ConstrainFire | LedgerUpdate | ReportAggregate`
- `truth: NarsTruth` (frequency + confidence on whether the
  dependency fired)

Cascade traversal is `EdgeColumn::walk_dependents(entity.row,
mode: TraversalMode)` where mode is set by the context. Same graph,
three traversal disciplines.

## Stage-1 deliverables (this plan)

| D-id | Description | Site | LOC | Conf | Status |
|---|---|---|---:|:--:|:--:|
| **D-NEH-1a** | `lance-graph-contract::cognition::{NormalizedEntity, stages, Op, OpKind, MailboxRow, Output}` typed surface — zero-dep, `todo!()` bodies, typestate compile-fail tests | `lance-graph-contract/src/cognition/` | 600 | HIGH | Queued |
| **D-NEH-1b** | `lance-graph-contract::transaction::{Interactive, Bulk, Periodisch, Context, OgitCtx/OwlCtx/DolceCtx/FibuCtx}` — context shapes + commit-policy traits | `lance-graph-contract/src/transaction/` | 400 | HIGH | Queued |
| **D-NEH-1c** | 5-verb advancement methods on `NormalizedEntity<S>` (`resolve_ogit` / `hydrate_owl` / `classify_dolce` / `align_fibu` / chain methods) — typed signatures + provenance + `todo!()` bodies | `lance-graph-contract/src/cognition/advance.rs` | 300 | HIGH | Queued |
| **D-NEH-1d** | `CascadeKind` + cascade-graph traversal trait on `EdgeColumn` + per-context traversal mode (`Sync` / `Batched` / `JitFixedPoint`) | `lance-graph-contract/src/cognition/cascade.rs` | 350 | MED | Queued |
| **D-NEH-1e** | Compile-fail tests proving the typestate gate (`.abduct()` on `<Raw>` fails to compile; `.output()` on `<Reviewed>` fails to compile) | `lance-graph-contract/tests/cognition_typestate.rs` | 250 | HIGH | Queued |
| **D-NEH-1f** | Doc-level example consumer chain (woa-rs invoice flow as a `cargo doc` example) + cross-reference docs to existing primitives | `lance-graph-contract/src/cognition/mod.rs` + `docs/COGNITION_HOLY_GRAIL.md` | 400 | HIGH | Queued |
| **D-NEH-1g** | Board hygiene + AGENT_LOG entries + EPIPHANIES prepend (the 8 driver epiphanies above) | `.claude/board/` | 200 | HIGH | Queued |

**Total Stage 1 LOC:** ~2500 (mostly typed signatures + compile-fail
tests; no kernel bodies yet).

## Subsequent waves (future plans, sketched only)

- **Stage 2 — kernel bodies (`normalized-entity-holy-grail-v2`):**
  port each Op kernel to the shader dispatch table; JIT-compile the
  hot-path form via `lance-graph-contract::jit`. ~50 kernels (one per
  typed Op in the BP-1 / EXT-2..6 surface). 5-10K LOC of shader
  kernel definitions.
- **Stage 3 — consumer DSL macros (`cognition-dsl-v1`):** per-
  repo `medcare_think!` / `woa_think!` / `smb_think!` macros expand to
  the typestate chain. Pipe-style ergonomics on top of the
  compile-time-typed chain. ~1K LOC per macro crate.
- **Stage 4 — Stream + GenServer integration
  (`cognition-stream-v1`):** wire the warm-path Op to a `tokio::
  Stream` interface; wire the interactive context to a
  `lance-graph-supervisor`-managed actor (ractor-style mailbox). The
  Elixir-OTP analogue.
- **Stage 5 — Jahresabrechnung kernel
  (`cognition-fiscal-close-v1`):** JIT-compile the full year-end
  chain; benchmark against Odoo's `account.fiscal.year.close` wizard.
  Target: ≥100× throughput on a 1M-row ledger.
- **Stage 6 — palantir-foundry parity audit
  (`foundry-parity-v3`):** for each palantir-foundry feature
  (ontology objects, branches, lineage, action types, functions on
  objects), document where in this substrate the equivalent lives.
  Closes the "better palantir foundry" framing.
- **Stage 7 — elixir-OTP parity audit (`otp-parity-v1`):** same for
  Elixir/OTP (lightweight processes, supervision trees, pattern
  matching, pipe operator, streams, behaviours). Closes the "better
  elixir" framing.

## Execution ordering (Stage 1)

1. **D-NEH-1g first** (board hygiene + epiphanies) — establishes the
   anchored findings that the rest of the work cites
2. **D-NEH-1a in parallel with D-NEH-1b** — independent module
   skeletons in the contract crate; can be done by parallel Sonnet
   agents with non-overlapping files
3. **D-NEH-1c after D-NEH-1a + 1b** — depends on both surfaces
4. **D-NEH-1d in parallel with D-NEH-1c** — cascade traversal is
   additive
5. **D-NEH-1e after D-NEH-1a..d** — compile-fail tests need the
   typestate machinery in place
6. **D-NEH-1f last** — docs need the surface stable

## Risks / open questions

- **MailboxRow vs owned data**: `NormalizedEntity` as a typed lens
  into a `MailboxSoA` row means the entity's lifetime is bounded by
  the mailbox. Cross-mailbox handoff = Baton emission, not entity
  move. Pin this in `E-NORMALIZED-ENTITY-2` once we hit a use case
  that pushes back.
- **JIT shape for chains**: today `jit::JitCompiler` compiles
  `Tactic` kernels parameterised by `(OdooEntity, AtomTouchMask)`.
  Chains-as-sequence-of-Ops need a new JitChainHandle that
  concatenates kernels with shared SoA-row state. Open whether this
  is one Cranelift function or N composed kernels.
- **Stream backpressure**: the warm-path `apply_stream` returns
  `impl Stream<Item = …>`. Backpressure policy (drop-newest vs
  block-producer vs spill-to-Lance) is a per-context concern; pin in
  `bulk_context::BackpressureMode` enum.
- **Macro DSL timing**: building the macros (Stage 3) requires the
  Stage 1 typestate to be stable. Doing it in lock-step with Stage 1
  is tempting but risks tying the macro shape to early surface
  decisions. Defer.
- **OpKind enum vs trait-impl-per-Op**: Stage 2 will reveal this.
  My lean is OpKind enum (single point of change, table-driven
  shader dispatch) per the earlier architectural discussion; final
  call deferred to when the first 10 kernels are in place.
- **Odoo cascade-mechanism enumeration**: the cascade unification
  (`E-CASCADE-AS-EDGECOLUMN-1`) claims six Odoo mechanisms collapse
  into one. Verify per-mechanism that the `EdgeColumn` traversal
  correctly captures: server actions (configurable),
  `_inherits` field forwarding (silent), and the mail-thread
  auto-subscription cascade. Conjecture until tested.
- **"Better palantir foundry / better elixir" measurement**: the
  ambition needs a yardstick. Proposal: Stage 6 + Stage 7 audits
  emit a feature-parity matrix (rows = foundry/OTP features,
  columns = workspace primitives, cells = present / partial / gap).
  When that matrix is ≥80% present, we have earned the framing.

## Board updates that land with this plan

Per CLAUDE.md mandatory board-hygiene rule, the commit that creates
this plan file also:
- PREPENDs an entry to `.claude/board/INTEGRATION_PLANS.md`
- PREPENDs 8 entries to `.claude/board/EPIPHANIES.md` (the driver
  epiphanies listed in the front matter)
- Adds D-NEH-1a..g rows to `.claude/board/STATUS_BOARD.md` once
  Stage 1 work starts

Future-PR-merge commits will:
- Append to `LATEST_STATE.md` Current Contract Inventory (new
  `cognition::*` + `transaction::*` modules)
- PREPEND `PR_ARC_INVENTORY.md` entries per the standard rule
- PREPEND `AGENT_LOG.md` entries per Sonnet-agent slice

Cross-references to update in subsequent waves:
- `INTEGRATION_DEBT_AND_PATHS.md` § "the missing consumer surface"
- `lab-vs-canonical-surface.md` (add "the consumer chain IS the
  canonical surface" corollary)
- `autoattended-multiagent-pattern.md` (Stage 2's ~50 kernel
  implementations are a natural multi-agent wave)
