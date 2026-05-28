//! Future shape: taxable item via the existing stack.
//!
//! This is NOT a proposal for new types — every primitive here already
//! exists somewhere in the workspace:
//!
//!   - `StyleRecipe` / `RecipeTemplate` / `register_recipe` — `lance-graph-contract::recipe`
//!   - `KernelHandle` / Cranelift JIT compile — `lance-graph-contract::jit`
//!   - `cognition::{advance, op, transaction}` — the API the codegen emits calls into
//!   - `PaletteCompose` SpMV — `bgz17::semiring`
//!   - 400ms ractor mailbox — the actional sweep (Heckhausen-Rubicon)
//!   - LanceDB — context / rule storage (vectors)
//!   - SurrealQL — kanban board state (goal cards + post-actional eval)
//!   - Cognitive shader — `cognitive-shader-driver` (the inner loop)
//!
//! This file shows how `taxable_item` over a tax rate compiles to exactly
//! one method call, with everything else pre-resolved at compile time.

// ─────────────────────────────────────────────────────────────────────
// 1. ELIXIR SOURCE — what a domain-expert accountant writes
// ─────────────────────────────────────────────────────────────────────
//
// File: rules/de/ustg/taxable_item.exs
//
//     defmodule DE.UStG.TaxableItem do
//       use Ada.Rule
//
//       # law :: context() :: emits :: rules()
//       def compute(item, rate) do
//         law(:UStG)                  # pull regulatory anchor
//         |> context(item)            # bind item to fiscal context (jurisdiction, period)
//         |> emits(:taxable_amount)   # name the output the rule emits
//         |> apply(rate, to: item)    # apply rate per the rule
//       end
//     end
//
// At build time the elixir-frontend codegen (per
// `.claude/plans/lance-graph-elixir-frontend-v1.md`) parses this with
// tree-sitter-elixir, HM-infers types over the `|>` pipeline against
// `lance-graph-contract::cognition`, and emits ↓ below ↓.

// ─────────────────────────────────────────────────────────────────────
// 2. CODEGEN OUTPUT — typed Rust that compiles down to one StyleRecipe
// ─────────────────────────────────────────────────────────────────────

use crate::cognition::{advance, transaction, Op};
use crate::jit::{KernelHandle, StyleRegistry};
use crate::recipe::{register_recipe, RecipeId, StyleRecipe};

/// Auto-emitted from `rules/de/ustg/taxable_item.exs`.
///
/// All four pipeline stages collapse to one `advance()` call: the
/// StyleRecipe (registered at app load) carries the `(law, context,
/// emits, rules)` chain as a weighted atom composition; the JIT fused
/// it into one branch-free kernel; the shader invokes the kernel
/// inside its 400ms actional sweep.
pub fn taxable_item(item: &Item, rate: TaxRate, ctx: &CompiledCtx) -> Money {
    // One call. The recipe + kernel were resolved in the pre-actional
    // phase; the shader's hot loop just invokes the handle.
    advance(ctx, Op::TaxableItem { item, rate, recipe: RECIPE_DE_USTG_TAXABLE })
}

// ─────────────────────────────────────────────────────────────────────
// 3. RECIPE REGISTRATION — data, not compute (Elixir-hot-load semantics)
// ─────────────────────────────────────────────────────────────────────
//
// Per `recipe.rs`: adding a *recipe* is a template change (registered
// once, JIT-compiles to a KernelHandle at next activation). Adding a
// new atom would be a data change (no recompile, new row in basis).

pub const RECIPE_DE_USTG_TAXABLE: RecipeId = RecipeId::new(0xDE_01_USTG_TAXBL);

pub const STYLE_DE_USTG_TAXABLE: StyleRecipe = StyleRecipe {
    name: "de.ustg.taxable_item",
    weights: &[
        // I4-32D atom catalogue indices (from D-ATOM-1); weight in [-7..+8].
        (ATOM_LAW_LOOKUP_USTG,       8),  // strong positive: this IS UStG
        (ATOM_PULL_FISCAL_CTX,       6),  // moderate: context-binding
        (ATOM_EMIT_TAXABLE_AMOUNT,   7),  // strong: this is what we yield
        (ATOM_APPLY_RATE,            8),  // strong: the operation
        (ATOM_MENGENMASS_MONEY,      6),  // mengenmass = money
        (ATOM_TEK_QUANTITIES,        5),  // TEKAMOLO slot = QU
        (ATOM_VERB_TRANSITIVE,       4),  // T (returns Money, not a raise)
    ],
    // Pre-computed composition vector populated by register_recipe()
    // (folds weights × basis atoms into one I4-32D vector for cosine match).
    composition: None,
};

// ─────────────────────────────────────────────────────────────────────
// 4. REGULATORY ANCHOR — concrete UStG section, traceable provenance
// ─────────────────────────────────────────────────────────────────────

pub const REGULATORY_ANCHOR_USTG_12: RegulatoryRef = RegulatoryRef {
    code: "UStG",
    paragraph: "§12",
    title_de: "Steuersätze",
    title_en: "Tax rates",
    url: "https://www.gesetze-im-internet.de/ustg_1980/__12.html",
    // Cross-source validation per OGIT-META-DTO-ALIGNMENT.md:
    // D1 = this recipe (Odoo code-extracted)
    // D2 = ogit.Accounting:Tax entity attributes (OGIT schema)
    // D3 = L-doc curated knowledge (.claude/odoo/L3-K7-TAX.md)
    cross_source_d2: Some("ogit.Accounting:Tax/amount"),
    cross_source_d3: Some(".claude/odoo/L3-K7-TAX.md#ustg-rate-table"),
};

// ─────────────────────────────────────────────────────────────────────
// 5. APP-LOAD REGISTRATION (pre-actional, NOT in the 400ms budget)
// ─────────────────────────────────────────────────────────────────────

#[ctor::ctor]
fn register_de_ustg_taxable() {
    register_recipe(STYLE_DE_USTG_TAXABLE);
    // After this fires once at app load, the recipe is in the pool
    // forever. Every kanban card that pulls a "compute taxable amount
    // under UStG" goal resolves to this RecipeId in the pre-actional
    // phase; the shader sweeps for 400ms invoking the JIT-compiled
    // kernel; the post-actional phase reads the Money result and the
    // Δentropy stream to decide goal-satisfaction.
}

// ─────────────────────────────────────────────────────────────────────
// 6. THE PATH AT RUNTIME (zero surprise — everything already in place)
// ─────────────────────────────────────────────────────────────────────
//
//   SurrealQL kanban  →  card pulled: "compute taxable amount for invoice N"
//                          │
//                          │ pre-actional (not budgeted):
//                          │   - lookup RECIPE_DE_USTG_TAXABLE in pool
//                          │   - JIT KernelHandle from STYLE_DE_USTG_TAXABLE
//                          │   - bind X cache (item refs, rate value)
//                          ▼
//   Mailbox spawns  →  400ms actional sweep begins
//                          │
//                          │ shader inner loop:
//                          │   advance(ctx, Op::TaxableItem{..}) →
//                          │     KernelHandle.invoke() →
//                          │       PaletteCompose SpMV (one cycle, ~20-200μs) →
//                          │         emit Δentropy
//                          │
//                          │ (in the typical case the loop terminates
//                          │  on F<floor after a small handful of cycles —
//                          │  taxable_item is a thin compute, not a search)
//                          ▼
//   Mailbox closes  →  post-actional eval (SurrealQL kanban + mailbox state)
//                          │
//                          │ Money result → write back to LanceDB
//                          │ Δentropy stream → episodic memory
//                          │ card → Done
//                          ▼
//   Next pre-actional cycle picks up next card.
//
// ─────────────────────────────────────────────────────────────────────
//
// What was hard about this?  Nothing — every line above maps onto an
// existing type in the workspace.  The Elixir source is 4 lines.  The
// compiled Rust is one `advance()` call.  The shader runs the same
// PaletteCompose semiring it already runs for everything else.  This
// is the future shape, and the future shape is small.