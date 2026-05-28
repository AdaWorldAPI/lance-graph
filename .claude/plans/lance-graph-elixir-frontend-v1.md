# lance-graph-elixir-frontend-v1 — Elixir-syntax script frontend → typed Rust codegen against `lance-graph-contract::cognition` (no BEAM)

> **Status:** PROPOSAL. Stage 5/8 of the holy-grail substrate roadmap (sibling to `normalized-entity-holy-grail-v1` + `unified-spo-nars-codegen-v1`). Adds an Elixir-source-file frontend that consumer-side domain experts (accountants, doctors, lawyers) read + write, compiled at build time to typed Rust chains against the codegen-emitted `OGIT_CODEBOOK`. **The BEAM is not a runtime target** — Elixir is the SURFACE only; the substrate stays typed Rust + SIMD + JIT.
>
> **Confidence:** HIGH on the parser front-end (tree-sitter-elixir has solid grammar coverage; embedding it via `tree-sitter-rust` binding is established). HIGH on the typestate-resolver direction (HM-style inference over `|>` pipelines is well-studied; our typestate ladder is small: 9 stages, 5 verbs, ~50 OpKind discriminants per domain). MED on the codegen target stability (depends on `unified-spo-nars-codegen-v1`'s `OGIT_CODEBOOK` shape stabilising; this plan is GATED on Stage-2 trunk shipping). LOW on the per-domain DSL ergonomic acceptance — accountants reading Elixir is a hypothesis worth verifying, not a foregone conclusion.
>
> **Predecessors (gating dependencies):**
> - `normalized-entity-holy-grail-v1` Stage 1 (PR #431 — typed carrier + chain methods)
> - `unified-spo-nars-codegen-v1` Wave D + Σ (codegen-emitted `OGIT_CODEBOOK` + `LabelDTO` compression) — this plan compiles AGAINST that target
> - `lance-graph-contract::cognition::{advance, op, transaction}` — Rust API the codegen emits calls into
> - tree-sitter-elixir grammar (external, BSD-3-clause)
>
> **Driver epiphanies:**
> - `E-CONSUMER-CANNOT-INTERPRET-1` (regex / hand-rolled `if x.starts_with("84")` is structurally banned via missing-function) — the Elixir surface MUST preserve this; `fn x -> x.balance * 0.19 end` closures are compile-time rejected unless they const-fold
> - `E-SHARE-SUBSTRATE-SEPARATE-OPERATIONS-1` (just filed) — the Elixir surface is one parser + one codegen; per-domain ops compile against domain-specific OpKind sets without forcing a per-domain frontend
> - `E-LABEL-DTO-IS-THE-SUBSTRATE-1` — Elixir atoms (`:invoice`, `:patient`, `:gobd_audit_trail`) map naturally to `LabelDTO` URNs at compile time
> - `E-FORMAT-DOMAIN-OGIT-IS-THE-URN-SCHEME-1` — Elixir `defmodule Woa.Invoice do … end` namespace maps to `<format>:Accounting:JournalEntry` URN via codegen-driven name resolution
> - Sketched orally during the 2026-05-28 session as the "better elixir" framing — the workspace earns "better than Elixir/OTP" when domain experts read/write the surface AND the runtime is typed Rust + SIMD + JIT
>
> **Anchored iron rules:**
> - "Lab vs canonical" — the Elixir frontend IS a lab surface for ergonomics; the canonical surface stays `UnifiedStep` via `OrchestrationBridge` + the typed `Op<I,O>` chain. The codegen target IS the canonical surface; the Elixir source is the ergonomic skin
> - "Consult before guess" — Elixir's existing patterns (Phoenix LiveView, Ecto schemas, GenServer, Stream) are prior art; this plan re-uses those patterns where they map to substrate primitives, replaces the BEAM-runtime ones (`spawn`, `receive`, `:erlang.*`) with compile-time errors
> - "The object speaks for itself" — Elixir's `|>` operator naturally expresses `carrier.method1().method2().method3()` — the substrate's chain methods become directly the pipeline-stage primitives

## The diagnosis

Today every consumer of `lance-graph-contract::cognition` writes Rust. Rust is correct but verbose for business-logic authors who think in `|>` pipelines, atoms, and pattern matching. Three concrete pain points:

1. **Domain experts can't read the chains.** A `KontenerkennungSkr04` Op chain written in Rust is opaque to an accountant; a `medcare_think!` macro chain is opaque to a doctor. Both audiences read Elixir-shaped flow code naturally — Phoenix and Ecto trained 15 years of business-logic authors on this syntax.
2. **The macro layer is per-repo + heavy.** Each consumer repo (woa-rs, medcare-rs, smb-office-rs, medcarev2) would need its own `_think!` macro crate; that's N repos × ~1K LOC per macro layer = ~5K LOC of duplicate metaprogramming.
3. **Source-code review happens on Rust, not on intent.** A PR adding "if patient has bone fracture + ICD-10 M80-M89 + recent X-ray, suggest osteoporosis differential" should be reviewed in those terms; today it's reviewed as Rust trait impls + Op kernels.

Elixir solves all three:
- Domain experts read `|>` chains
- One parser + one codegen handles every consumer repo
- Source review is at the intent layer; the generated Rust is compilation output

## The shape — three layers

### 1. Parser front-end (tree-sitter-elixir, ~2K LOC)

Embed `tree-sitter-elixir` via the `tree-sitter` Rust bindings. Parse `.exs` source files into an Elixir AST. Subset-restrict: only the constructs that map to substrate primitives are accepted; the rest emit compile errors:

| Elixir construct | Substrate mapping | Status |
|---|---|---|
| `\|>` pipe operator | typed chain method call | ✅ accepted |
| `case`/`cond`/`with` | const-data pattern match → shader kernel dispatch | ✅ accepted |
| atoms (`:invoice`) | `LabelDTO` URN | ✅ accepted (typed at compile time against `OGIT_CODEBOOK`) |
| `defmodule` / `def` | crate module / typed function | ✅ accepted |
| `defstruct` | const-data record (SIMD layout) | ✅ accepted |
| `use LanceGraph.Cognition` | injects the chain DSL macros | ✅ accepted |
| `GenServer.call`/`cast` | `Baton` emission via `CollapseGateEmission` | ⚡ subset (synchronous-call shape only; selective receive is compile error) |
| `receive do … end` | mailbox SoA selective read | ❌ compile error: "no BEAM target; use `apply_soa` op for SoA sweeps" |
| `:erlang.spawn` / `Process.*` | (BEAM-only) | ❌ compile error: "use `lance-graph-supervisor` actor for owned cohorts" |
| `Mix.install` / `Code.eval_string` | dynamic code loading | ❌ compile error: "compile-time only; runtime eval rejected" |
| anonymous `fn -> end` closures | only allowed where they const-fold to typed Op | ❌ rejected otherwise: "must be a typed Op" |

### 2. Typestate resolver (~3-4K LOC)

HM-style type inference over `|>` chains. For each pipeline:
- Track the carrier's `Stage` at every `|>` step
- Resolve each method call (`resolve_ogit`, `chk_data`, `review`, …) against the substrate's `OGIT_CODEBOOK` to find the typed Op kernel
- Verify the Op's `I` matches the current Stage; advance to `O` for the next step
- Reject out-of-order chains with clear Elixir-source-line errors (column-precise, source-mapped)

```elixir
# This compiles:
invoice
|> resolve_ogit()                                  # Stage: Raw → WithOgit
|> hydrate_owl()                                   # WithOgit → WithOwl
|> classify_dolce()                                # WithOwl → WithDolce
|> align_fibu()                                    # WithDolce → Normalized
|> op(SkrAccountInRange.new(8400..8499))           # Normalized → Normalized
|> chk_data(FiscalPositionResolver)                # Normalized → Checked
|> review(InsuranceCoverage)                       # Checked → Reviewed
|> abduct(VatLiability)                            # Reviewed → Abducted
|> report(UStvaKennzahlAggregator)                 # Abducted → Reported
|> output()                                        # Reported → Output

# This compile-errors at the resolver:
invoice
|> abduct(VatLiability)                            # ← ERROR: cannot abduct on Stage Raw
#  ^^^^^^                                          # ← column-precise
# Reason: `abduct` requires Stage Reviewed; carrier is Stage Raw.
# To advance Stage Raw → Reviewed, chain resolve_ogit → hydrate_owl → classify_dolce → align_fibu → chk_data → review first.
```

### 3. Codegen target (~2K LOC)

Emit Rust source against `lance-graph-contract::cognition` + `transaction`. The generated code is what `cargo` compiles:

```rust
// Generated from `woa/invoice.exs`:
pub mod woa_invoice {
    use lance_graph_contract::cognition::*;
    use lance_graph_contract::transaction::{Interactive, Bulk, Periodisch};

    pub fn post(invoice: LabelDTO, ctx: &Interactive) -> Output {
        NormalizedEntity::<Raw>::from(invoice)
            .resolve_ogit(ctx)
            .hydrate_owl(ctx)
            .classify_dolce(ctx)
            .align_fibu(ctx)
            .op(SkrAccountInRange::new(8400..=8499))
            .chk_data(FiscalPositionResolver)
            .review(InsuranceCoverage)
            .abduct(VatLiability)
            .report(UStvaKennzahlAggregator)
            .output()
    }
}
```

The generated code is identical to what a Rust author would write by hand — no runtime overhead, no dynamic dispatch, just compile-time codegen output.

## Concrete example — the woa-rs invoice flow

```elixir
# woa/invoice.exs
defmodule Woa.Invoice do
  use LanceGraph.Cognition

  context :interactive do
    def post(invoice) do
      invoice
      |> resolve_ogit()
      |> hydrate_owl()
      |> classify_dolce()
      |> align_fibu()
      |> op(KontenerkennungSkr04)
      |> chk_data(SkrAccountInRange.new(8400..8499))
      |> review(FiscalPositionResolver)
      |> abduct(VatLiability)
      |> op(GoBdLockCheck)
      |> report(UStvaKennzahlAggregator)
      |> output()
    end
  end

  context :periodisch do
    def jahresabrechnung(year) do
      frozen_lance_at(year_end(year))
      |> jit_chain(JahresabrechnungChain)
      |> iterate_until(:debits_eq_credits)
      |> commit_as_fiscal_close()
    end
  end

  cascade do
    edge "account.move.line.balance" -> "res.partner.credit",
         kind: :ledger_update,
         truth: nars(freq: 1.0, conf: 0.95)
  end
end
```

The `cascade do … end` block declares `EdgeColumn` entries; the `context :interactive do … end` block declares which transaction context the enclosed functions run in; atoms (`:ledger_update`, `:debits_eq_credits`) are codebook URNs typed at compile time.

## And the medcare-rs medical flow

```elixir
# medcare/encounter.exs
defmodule Medcare.Encounter do
  use LanceGraph.Cognition

  context :interactive do
    def diagnose(encounter) do
      encounter
      |> resolve_ogit()
      |> hydrate_owl()
      |> classify_dolce()
      |> align_clinical()
      |> op(VitalsIngest)
      |> chk_data(IcdValid.icd10_gm())
      |> review(DifferentialDiagnosis)
      |> abduct(CarePathway)
      |> op(ContraindicationCheck)        # Capability auto-emitted: PHARMACOGENOMIC
      |> report(BillingClaim)
      |> output()
    end
  end

  cascade do
    edge "patient.diagnosis" -> "drug.contraindication",
         kind: :clinical_aggregation,
         truth: nars(freq: 0.92, conf: 0.85)
  end
end
```

Same chain shape, different OGIT slot resolution, different Op kernels — but the SUBSTRATE is one. A doctor reads `Medcare.Encounter.diagnose`, an accountant reads `Woa.Invoice.post`; both files compile to the same typed substrate.

## Stage-1 deliverables

| D-id | What | Site | LOC | Conf |
|---|---|---|---:|:--:|
| **D-LGE-1** | Parser front-end — `tree-sitter-elixir` integration; AST extraction; subset gating (reject `receive`/`spawn`/`:erlang.*`/anonymous closures-that-don't-const-fold) | new crate `crates/lance-graph-elixir-parser/` | 2000 | HIGH |
| **D-LGE-2** | Typestate resolver — HM-style stage inference over `\|>` chains; method resolution against `OGIT_CODEBOOK`; compile-error generation with column-precise Elixir source mapping | new crate `crates/lance-graph-elixir-resolver/` | 3500 | HIGH |
| **D-LGE-3** | Codegen — Rust source emission against `lance-graph-contract::cognition` + `transaction`; one .rs per .exs; output goes to `OUT_DIR` for downstream cargo crates to `include!` | new crate `crates/lance-graph-elixir-codegen/` | 2000 | HIGH |
| **D-LGE-4** | Stdlib shim — `LanceGraph.Std` module mapping common Elixir idioms (`Enum.map` over stream, `with {:ok, x} <-` railway, `String.contains?` over codebook lookup) to typed Op chains | `lance-graph-elixir-codegen/stdlib/` | 1500 | MED |
| **D-LGE-5** | Build-script integration — `lance-graph-elixir-build` helper crate that downstream repos drop into their `Cargo.toml`'s `[build-dependencies]`; walks `src/*.exs` and emits `OUT_DIR/elixir_compiled.rs` | new crate `crates/lance-graph-elixir-build/` | 500 | HIGH |
| **D-LGE-6** | First consumer migration — port one woa-rs route from Rust to `.exs` (e.g. the invoice-posting flow); demonstrate compile-time equivalence + zero runtime overhead | `woa-rs` repo PR | 200 | HIGH |
| **D-LGE-7** | Documentation — `docs/ELIXIR_FRONTEND.md` cookbook with 10 typical patterns (interactive flow, bulk import, periodic report, cascade declaration, capability access, meta-pattern lookup) | new doc | 400 | HIGH |
| **D-LGE-8** | LSP integration scaffold — basic `lance-graph-elixir-lsp` server (start with go-to-definition + hover + diagnostics); install in VSCode + JetBrains | new crate | 1500 | MED |

**Total Stage 1:** ~11.5K LOC across 6 new crates + 1 doc + 1 LSP scaffold.

## Execution ordering

**Wave A — parser** (D-LGE-1). Independent of all other waves; tree-sitter-elixir + AST extraction is a self-contained substrate. **Parallel-spawnable.**

**Wave B — resolver** (D-LGE-2). Requires Wave A AST + `OGIT_CODEBOOK` from `unified-spo-nars-codegen-v1` Wave D. Gating dependency on the Stage-2 trunk plan; can't start until that ships.

**Wave C — codegen** (D-LGE-3). Requires Wave A + Wave B. Mostly mechanical — walk the resolved AST, emit Rust source. **Parallel-spawnable with stdlib shim (D-LGE-4).**

**Wave D — build-script integration + first consumer** (D-LGE-5, D-LGE-6). Requires Wave C complete. Smallest LOC but highest validation value.

**Wave E — docs + LSP** (D-LGE-7, D-LGE-8). Independent of all prior; can run in parallel with Waves A-D once the API shapes are stable.

## Risks / open questions

- **Domain expert read-fluency is a hypothesis.** Accountants reading Elixir is the bet behind this plan. Mitigation: Wave D's first consumer migration includes a "read-the-flow" review session with the actual MedCareV2 + WoA stakeholders before scaling to all consumers. If the hypothesis fails, the plan still provides ergonomic Rust DSL via `macro_rules!` as a fallback (less elegant but still typed).
- **Tree-sitter-elixir Rust binding stability.** Tree-sitter parsers occasionally bump their language ABI; the binding crate may lag. Mitigation: vendor the exact grammar version at a pinned SHA; CI bumps require explicit review.
- **Elixir-source error UX vs Rust-source error UX.** If the typestate resolver emits errors that point at the GENERATED Rust source, users will be confused. The resolver must emit errors against the ELIXIR source with column-precise pointers. Mitigation: build a source-map layer; Wave B includes UX testing on intentionally-broken `.exs` files.
- **Build time regression.** Each consumer crate adds a build-script step that parses + resolves + codegens all its `.exs` files. For incremental builds, the codegen must support file-level granularity (only re-codegen changed `.exs` files). Mitigation: D-LGE-5 includes mtime-based incremental codegen; `cargo build` should add <100ms for unchanged Elixir.
- **GenServer-shape ergonomics.** Elixir's `GenServer.call` is synchronous; mapping to Baton emission requires the call to be transactional (the carrier emits a Baton and waits for cascade quiescence before the call returns). Mitigation: D-LGE-4's stdlib shim maps `GenServer.call` to `entity.transit_op(op)` with a closure receiving the cascade result; semantics are clear but the mapping is non-trivial.
- **Macro DSL recursion.** Elixir devs write recursive flows; the substrate's typestate chain is linear. Mitigation: D-LGE-4's stdlib shim provides `Stream.unfold` over `apply_stream` (warm path) for iterative flows; recursion through the chain is compile-error.

## Why this lands as Stage 5 (not Stage 3 of the holy-grail)

`normalized-entity-holy-grail-v1`'s subsequent waves are:
- Stage 2 — kernel bodies (`unified-spo-nars-codegen-v1` — gating dep for this plan)
- Stage 3 — consumer DSL macros (Rust `macro_rules!` — the FALLBACK if the Elixir frontend hypothesis fails)
- Stage 4 — Stream + ractor-supervised actor wiring
- **Stage 5 — this plan (`lance-graph-elixir-frontend-v1`)**
- Stage 6 — Jahresabrechnung JIT kernel
- Stage 7 — palantir-foundry parity audit
- Stage 8 — elixir-OTP parity audit (closes the "better elixir" framing)

Stage 5 sequences AFTER Stage 3 because if domain-expert read-fluency turns out to be a non-event, Stage 3's Rust DSL macros (`woa_think!` / `medcare_think!`) cover the ergonomic gap at lower implementation cost. Stage 5 is the bet — Stage 3 is the hedge.

## Board updates that land with this plan

Per CLAUDE.md mandatory board-hygiene rule, the commit that creates this plan file also:
- PREPENDs an entry to `.claude/board/INTEGRATION_PLANS.md`
- PREPENDs new epiphanies if any (none beyond what `unified-spo-nars-codegen-v1` already filed)
- Adds D-LGE-1..8 rows to `.claude/board/STATUS_BOARD.md` once Wave A starts

The plan is the SURFACE deliverable that earns the "better Elixir" framing — domain experts read + write the surface, the substrate stays typed Rust + SIMD + JIT. When Stage 8's parity audit lands and shows ≥80% Elixir/OTP feature parity (lightweight processes via lance-graph-supervisor, supervision via ractor, streams via warm-path Op, pattern matching via const-data Op dispatch, behaviours via the Op trait, pipe via `|>`), the workspace has earned the framing.
