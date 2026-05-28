//! Codegen output as SoA layout — three planes rendered as DTO columns.
//!
//! Same principle as `images/three-planes-SoA-DTO.png`: each named
//! section IS one DTO column, and every rule below fills exactly that
//! column structure.  Read top-to-bottom = read the columns.  Read
//! left-to-right across `ALL_RULES[..]` = SoA sweep across rows.
//!
//! Auto-emitted by the (to-build) `tools/odoo-codegen` from:
//!   - `/tmp/odoo-extract/harvest-full/bundles/<family>.ndjson`   (ruff-py-dto)
//!   - `/tmp/work/delegation-full/delegation.ttl`                 (extract_delegation.py)
//!   - per-domain grammar coding (`.claude/odoo/tax_grammar.py` shape)
//!   - `.claude/odoo/L*.md`                                       (curated regulatory anchors)
//!   - OGIT NTO entity registry                                   (basin resolution)
//!
//! Runtime cost: one `OP_DISPATCH.get(op.kind())` lookup → one row →
//! invoke recipe.  No ontology touch, no schema walk.

use crate::cognition::{Op, RecipeId};
use crate::atoms::{Atom, AtomId};
use crate::recipe::RuntimeContract;

// ─────────────────────────────────────────────────────────────────────
// SoA row layout — one CompiledRule per business rule.
// All rules below fill EXACTLY this column structure.
// ─────────────────────────────────────────────────────────────────────

pub struct CompiledRule {
    // ── COLUMN 1: SEMANTIC SPELL (intent grammar, domain-expert-readable) ──
    pub semantic_spell:    &'static str,            // Elixir source verbatim
    pub semantic_meaning:  &'static str,            // single-sentence English

    // ── COLUMN 2: COMPILED SYSCALL (runtime bone, deterministic) ───────────
    pub op_kind:           Op,                       // enum discriminant
    pub fn_signature:      &'static str,             // for docs / audit only
    pub runtime_contract:  RuntimeContract,          // bitflags: NO_ONTOLOGY | SINGLE_ADVANCE | TYPED | CACHE_READY

    // ── COLUMN 3: COGNITIVE CHECKSUM (StyleRecipe weights = soul-print) ────
    pub recipe_id:         RecipeId,
    pub recipe_weights:    &'static [(AtomId, i8)],  // i4 weights -7..+8 over D-ATOM-1
    pub interpretation:    &'static str,             // what the weights MEAN, plain text

    // ── COLUMN 4: REGULATORY ANCHOR (concrete provenance, cross-source) ────
    pub anchor:            RegRef,                   // primary jurisdictional reference
    pub cross_source_d2:   Option<CrossRef>,         // OWL / ontology cross-pointer
    pub cross_source_d3:   Option<CrossRef>,         // L-doc / case law cross-pointer

    // ── COLUMN 5: GRAMMAR CODING (E-BUSINESS-LOGIC-IS-GRAMMAR-1 axes) ──────
    pub transitivity:      Transitivity,             // T transitive / I intransitive
    pub tekamolo:          TekSlot,                  // TE | KA | MO | LO | QU
    pub mengenmass:        Mengenmass,               // money | percent | rate | count | date | none

    // ── COLUMN 6: COST MODEL (pre-actional budget estimate) ────────────────
    pub cost_estimate_ns:  u32,                      // typical kernel cost in ns
    pub cost_p99_ns:       u32,                      // worst-case for budget guarding

    // ── COLUMN 7: PROVENANCE (audit trail) ─────────────────────────────────
    pub extracted_from:    SourceRef,                // (file, line_start, line_end)
    pub extractor_version: &'static str,             // for re-extraction reproducibility
}

// ─────────────────────────────────────────────────────────────────────
// ROW 1 — DE.UStG.TaxableItem
// ─────────────────────────────────────────────────────────────────────

pub const RULE_DE_USTG_TAXABLE_ITEM: CompiledRule = CompiledRule {
    semantic_spell:    r#"law(:UStG) |> context(item) |> emits(:taxable_amount) |> apply(rate, to: item)"#,
    semantic_meaning:  "Apply German VAT law (UStG) to compute taxable amount of an item at a given tax rate.",

    op_kind:           Op::TaxableItem,
    fn_signature:      "fn(item: ItemId, rate: TaxRate, ctx: CtxId) -> Money",
    runtime_contract:  RuntimeContract::NO_ONTOLOGY
                         .union(RuntimeContract::SINGLE_ADVANCE)
                         .union(RuntimeContract::TYPED)
                         .union(RuntimeContract::CACHE_READY),

    recipe_id:         RecipeId::DE_USTG_TAXABLE,
    recipe_weights:    &[
        (Atom::LAW_LOOKUP_USTG,     8),  // strong + : this IS UStG
        (Atom::PULL_FISCAL_CTX,     6),
        (Atom::EMIT_TAXABLE_AMOUNT, 7),
        (Atom::APPLY_RATE,          8),
        (Atom::MENGENMASS_MONEY,    6),
        (Atom::TEK_QUANTITIES,      5),
        (Atom::VERB_TRANSITIVE,     4),
    ],
    interpretation:    "legal-contextual, fiscal, monetary, quantity-aware, transitive, emits taxable amount",

    anchor:            RegRef { code: "UStG", para: "§12", title: "Steuersätze" },
    cross_source_d2:   Some(CrossRef::EU_VAT_DIRECTIVE_2006_112_EC),
    cross_source_d3:   Some(CrossRef::BFH_CASE_LAW_BMF_GUIDELINES),

    transitivity:      Transitivity::Transitive,
    tekamolo:          TekSlot::Quantities,
    mengenmass:        Mengenmass::Money,

    cost_estimate_ns:  80,
    cost_p99_ns:       200,

    extracted_from:    SourceRef::new("account/models/account_tax.py", 342, 367),
    extractor_version: "ruff-py-dto-0.1.0+tax_grammar.py-2026-05-28",
};

// ─────────────────────────────────────────────────────────────────────
// ROW 2 — DE.UStG.ReverseCharge (Reverse-Charge / §13b UStG)
// ─────────────────────────────────────────────────────────────────────

pub const RULE_DE_USTG_REVERSE_CHARGE: CompiledRule = CompiledRule {
    semantic_spell:    r#"law(:UStG) |> context(invoice) |> when(supplier.country != customer.country) |> emits(:reverse_charge_marker)"#,
    semantic_meaning:  "Mark cross-border B2B invoices as reverse-charge per UStG §13b (recipient owes VAT, not supplier).",

    op_kind:           Op::ReverseChargeMark,
    fn_signature:      "fn(invoice: InvoiceId, ctx: CtxId) -> ReverseChargeFlag",
    runtime_contract:  RuntimeContract::NO_ONTOLOGY
                         .union(RuntimeContract::SINGLE_ADVANCE)
                         .union(RuntimeContract::TYPED)
                         .union(RuntimeContract::CACHE_READY),

    recipe_id:         RecipeId::DE_USTG_REVERSE_CHARGE,
    recipe_weights:    &[
        (Atom::LAW_LOOKUP_USTG,     8),
        (Atom::PULL_FISCAL_CTX,     7),
        (Atom::CHECK_JURISDICTION,  8),   // strong + : the central test
        (Atom::EMIT_FISCAL_FLAG,    6),
        (Atom::MENGENMASS_CATEGORICAL, 5), // flag, not money
        (Atom::TEK_LOCATIVE,        7),   // LO slot dominant (cross-border)
        (Atom::VERB_TRANSITIVE,     4),
    ],
    interpretation:    "legal-contextual, jurisdictional, categorical (boolean flag), locative-dominant, transitive",

    anchor:            RegRef { code: "UStG", para: "§13b", title: "Leistungsempfänger als Steuerschuldner" },
    cross_source_d2:   Some(CrossRef::EU_VAT_DIRECTIVE_2006_112_EC_ART_196),
    cross_source_d3:   Some(CrossRef::BMF_SCHREIBEN_2014_03_25_REVERSE_CHARGE),

    transitivity:      Transitivity::Transitive,
    tekamolo:          TekSlot::Locative,
    mengenmass:        Mengenmass::Categorical,

    cost_estimate_ns:  60,
    cost_p99_ns:       180,

    extracted_from:    SourceRef::new("account/models/account_move.py", 1820, 1849),
    extractor_version: "ruff-py-dto-0.1.0+tax_grammar.py-2026-05-28",
};

// ─────────────────────────────────────────────────────────────────────
// ROW 3 — DE.HGB.FiscalYearLock (Festschreibung)
// ─────────────────────────────────────────────────────────────────────

pub const RULE_DE_HGB_FISCALYEAR_LOCK: CompiledRule = CompiledRule {
    semantic_spell:    r#"law(:HGB) |> context(move) |> when(move.date < ctx.fiscalyear_lock_date) |> raise(:LockedPeriod)"#,
    semantic_meaning:  "Reject any account.move whose date falls in a closed fiscal year (HGB §239 Festschreibung).",

    op_kind:           Op::FiscalYearLockCheck,
    fn_signature:      "fn(move: MoveId, ctx: CtxId) -> Result<(), LockedPeriod>",
    runtime_contract:  RuntimeContract::NO_ONTOLOGY
                         .union(RuntimeContract::SINGLE_ADVANCE)
                         .union(RuntimeContract::TYPED)
                         .union(RuntimeContract::CACHE_READY),

    recipe_id:         RecipeId::DE_HGB_FISCALYEAR_LOCK,
    recipe_weights:    &[
        (Atom::LAW_LOOKUP_HGB,      8),
        (Atom::PULL_FISCAL_CTX,     6),
        (Atom::CHECK_TEMPORAL,      8),   // strong + : it's a temporal check
        (Atom::RAISE_LOCKED_PERIOD, 7),
        (Atom::MENGENMASS_DATE,     6),
        (Atom::TEK_TEMPORAL,        8),   // TE slot dominant
        (Atom::VERB_INTRANSITIVE,   6),   // raises without return → I
    ],
    interpretation:    "legal-contextual, temporal-dominant, intransitive (raise-without-return), date-quantity, audit-trail-grade",

    anchor:            RegRef { code: "HGB", para: "§239", title: "Festschreibung von Buchführungsdaten" },
    cross_source_d2:   Some(CrossRef::OGIT_ACCOUNTING_JOURNALENTRY_LOCK_ATTR),
    cross_source_d3:   Some(CrossRef::GOBD_2019_UNVERAENDERBARKEIT),

    transitivity:      Transitivity::Intransitive,
    tekamolo:          TekSlot::Temporal,
    mengenmass:        Mengenmass::Date,

    cost_estimate_ns:  40,
    cost_p99_ns:       120,

    extracted_from:    SourceRef::new("account/models/account_move.py", 2950, 2978),
    extractor_version: "ruff-py-dto-0.1.0+tax_grammar.py-2026-05-28",
};

// ─────────────────────────────────────────────────────────────────────
// SoA SWEEP HANDLES — what runtime + audit + planner consume
// ─────────────────────────────────────────────────────────────────────

/// All rules in registration order. Iterating this IS the SoA sweep.
/// Audit / coverage / cost-budget / planner all walk this slice.
pub const ALL_RULES: &[&CompiledRule] = &[
    &RULE_DE_USTG_TAXABLE_ITEM,
    &RULE_DE_USTG_REVERSE_CHARGE,
    &RULE_DE_HGB_FISCALYEAR_LOCK,
    // ... 3552 more rows (one per extracted method, filled by codegen)
];

/// O(1) dispatch: Op enum → row.  Built once at app load by the ctor.
/// This IS the entire runtime ontology touch — paid once, then never
/// crossed again until the next mailbox cycle.
pub static OP_DISPATCH: once_cell::sync::Lazy<phf::Map<Op, &'static CompiledRule>> =
    once_cell::sync::Lazy::new(|| {
        let mut m = phf::Map::new();
        for r in ALL_RULES { m.insert(r.op_kind, r); }
        m
    });

// ─────────────────────────────────────────────────────────────────────
// THE HOT PATH — one fn, branch-free, no schema, no ontology
// ─────────────────────────────────────────────────────────────────────

pub fn advance(ctx: &CompiledCtx, op: Op) -> AdvanceOutcome {
    // (1) lookup row — O(1) phf hash
    let rule = OP_DISPATCH.get(&op).expect("op→rule mapping (codegen invariant)");

    // (2) budget guard — already in the row, single subtract+compare
    if rule.cost_estimate_ns > ctx.deadline_remaining_ns() {
        return AdvanceOutcome::BudgetExhausted;
    }

    // (3) invoke the JIT-compiled kernel for this recipe
    //     (kernel handle was resolved + cached in pre-actional phase)
    let delta = ctx.kernel(rule.recipe_id).invoke(ctx, op);

    // (4) emit Δentropy — the shader's only output
    AdvanceOutcome::Advanced { delta, recipe: rule.recipe_id }
}

// ─────────────────────────────────────────────────────────────────────
// WHY THIS IS THE RIGHT SHAPE FOR CODEGEN OUTPUT
// ─────────────────────────────────────────────────────────────────────
//
// Read TOP-TO-BOTTOM in one row    = read all DTO fields of one rule.
// Read LEFT-TO-RIGHT across rows   = SoA sweep (audit, coverage, cost).
// Every rule fills EXACTLY the same column structure → diff-friendly,
// reviewer-friendly, mechanical-codegen-friendly.
//
// Same shape as `images/three-planes-SoA-DTO.png` rendered as source.
// The diagram and the codegen output are literally the same data
// structure, one rendered for human reading and one for the compiler.
//
// No `if codegen_target == elixir { ... } else if target == rust { ... }`
// branching during emit.  The codegen reads the bundles + delegation +
// grammar coding + OGIT basin + regulatory anchor for each method, and
// emits one CompiledRule const filling every column.  Order-of-operations
// across columns is fixed.  Column count is fixed.  Per-column shape is
// fixed (it's a typed Rust field, not a free-form blob).
//
// New domain (audit / X-ray / HR) = same struct, different rows.
// New plane (e.g. quantum/probabilistic semantics) = new column added
// to CompiledRule, every rule's codegen filler updated mechanically.
// Removing a plane = remove the column.  Schema evolves; layout stays.