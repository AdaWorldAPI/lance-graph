# Cognition holy-grail (normalized-entity-holy-grail-v1)

This document is a pointer. The canonical source is
`.claude/plans/normalized-entity-holy-grail-v1.md`.

## What this is

The typed consumer pipeline grammar that unifies the workspace's
OGIT/OWL/DOLCE/Odoo inheritance + cognitive shader + JIT + MailboxSoA
into one surface consumers chain on top of.

## Where the pieces live

- Carrier: `lance_graph_contract::cognition::NormalizedEntity<Stage>`
- Stage markers: `lance_graph_contract::cognition::stages::{Raw, WithOgit, WithOwl, WithDolce, Normalized, Checked, Reviewed, Abducted, Reported}`
- Algebra: `lance_graph_contract::cognition::advance` (5 verbs)
- Op trait: `lance_graph_contract::cognition::op::Op<I,O>`
- Contexts: `lance_graph_contract::transaction::{Interactive, Bulk, Periodisch}`
- Context traits: `lance_graph_contract::transaction::{OgitCtx, OwlCtx, DolceCtx, FibuCtx}`
- Cascade: `lance_graph_contract::cognition::cascade::{CascadeKind, TraversalMode, CascadeWalker}`

## Stage 1 status (D-NEH-1a..g shipped)

Stage 1 ships the typed surface as a zero-dep scaffold. All
advancement verbs past `resolve_ogit` are `todo!()` bodies pending
Stage 2 wiring. The Op trait's `apply_stream` (warm) and `apply_soa`
(hot) call sites are documented but deferred — see `// TODO(Stage 2):` markers
in `op.rs`.

## What stages 2..7 will add

See plan §"Subsequent waves":

- **Stage 2** — kernel bodies: port ~50 Op kernels to the shader
  dispatch table; JIT-compile the hot path; wire the real
  OGIT/OWL/DOLCE/FIBU lookups in all three contexts.
- **Stage 3** — consumer DSL macros: per-repo `medcare_think!` /
  `woa_think!` / `smb_think!` macros expand to the typestate chain.
- **Stage 4** — Stream + GenServer integration: wire warm-path Op to
  `tokio::Stream`; wire Interactive context to a supervised actor.
- **Stage 5** — Jahresabrechnung kernel: JIT-compile the full year-end
  chain; benchmark against Odoo's fiscal-year-close wizard.
- **Stage 6** — palantir-foundry parity audit.
- **Stage 7** — elixir-OTP parity audit.

## Consumer example (from the doc)

```rust
use lance_graph_contract::cognition::*;
use lance_graph_contract::transaction::Interactive;

let ctx = Interactive::new();
let invoice = NormalizedEntity::<Raw>::raw(
    OdooEntityRef("account.move"),
    MailboxRow { mailbox_ref: 0, row_idx: 0 },
);

// Stage 2 onwards — bodies wired, no more todo!():
let result = invoice
    .resolve_ogit(&ctx)       // Raw → WithOgit
    .hydrate_owl(&ctx)        // WithOgit → WithOwl
    .classify_dolce(&ctx)     // WithOwl → WithDolce
    .align_fibu(&ctx)         // WithDolce → Normalized
    .op(KontenerkennungSkr04)
    .chk_data(SkrAccountInRange::new(8400..=8499))
    .review(FiscalPositionResolver)
    .abduct(VatLiability)
    .op(GoBdLockCheck)
    .report(UStvaKennzahlAggregator)
    .output();
```
