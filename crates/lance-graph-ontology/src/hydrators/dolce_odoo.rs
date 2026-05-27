//! Odoo DOLCE suffix classifier — Seam decision 2, in its own module.
//!
//! Per Open-question 3 of the four-way alignment seam
//! (`woa-rs/.claude/reference/four_way_alignment_seam.md`), the odoo-specific
//! DOLCE heuristics live in a separate module rather than inline in
//! `dolce.rs`, so each extraction source (odoo, FMA, SNOMED, future ones) owns
//! its own per-source heuristic logic.
//!
//! The odoo namespace uses dotted lowercase model names (`account.move`,
//! `stock.move`, `hr.attendance`) where event semantics are encoded by
//! *suffix* (`.move`, `.message`, `.attendance`). [`classify_odoo`] maps a
//! model name onto its DOLCE upper category from those suffixes, with one
//! explicit special-case (`product.template` — odoo's "template" there means
//! the master product record, an Endurant, not a config template).
//!
//! Litmus (CLAUDE.md): this is a stateless pure function with no carrier — it
//! reads a `&str` and returns a category. That is the sanctioned shape for a
//! classifier; there is no odoo-class struct to hang it on.

/// DOLCE upper categories used by the odoo suffix classifier.
///
/// These are the four DOLCE-Lite-Plus top categories. **Canonical DOLCE+DUL
/// renames `Endurant` → `Object` and `Perdurant` → `Event`** per the DUL
/// ontology header (see `dolce.rs` module docs); this enum keeps the original
/// DOLCE-Lite-Plus names because that is the vocabulary the seam doc's test
/// matrix and the `lance-graph-callcenter::super_domain::DolceMarker` seed use.
///
/// Unlike `DolceMarker` in `lance-graph-callcenter`, there is no `Unknown`
/// variant: [`classify_odoo`] always returns a concrete category (defaulting
/// to [`DolceCategory::Endurant`] for persistent stateful objects), matching
/// the seam's "Default: Endurant" rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DolceCategory {
    /// Persistent stateful object (DUL: `Object`). The default.
    Endurant,
    /// Event / occurrence that unfolds in time (DUL: `Event`).
    Perdurant,
    /// An attribute / rate / classification that characterises something.
    Quality,
    /// A reference / configuration / template — an abstract entity.
    AbstractEntity,
}

/// Name suffixes indicating a Perdurant (event / occurrence).
///
/// `.move.line` precedes `.move` in spirit — a line within a move event is a
/// fact within that Perdurant (seam matrix: `account.move.line → Perdurant`).
/// It is listed explicitly because it does not end with `.move`. Order within
/// this list does not matter (any match returns Perdurant), but the comment
/// records why both are present.
const PERDURANT_SUFFIXES: &[&str] = &[
    ".move.line",   // account.move.line — fact within the move event
    ".move",        // account.move, stock.move, hr.leave.allocation.move
    ".message",     // mail.message
    ".activity",    // mail.activity
    ".attendance",  // hr.attendance
    ".transition",  // workflow transitions
    ".event",       // calendar.event
    ".log",         // any .log model
    ".history",     // change history
    ".transaction", // payment.transaction
    ".picking",     // stock.picking (logistic event)
    ".scrap",       // stock.scrap (logistic event)
];

/// Name suffixes indicating a Quality (attribute / classification / rate).
const QUALITY_SUFFIXES: &[&str] = &[
    ".tag",      // crm.tag, account.account.tag
    ".category", // product.category, res.partner.category
    ".type",     // account.account.type, sale.order.type
    ".group",    // res.groups, account.tax.group
    ".tax",      // account.tax (it's a rate, a quality, not an event)
];

/// Name suffixes indicating an AbstractEntity (reference / config / template).
const ABSTRACT_SUFFIXES: &[&str] = &[
    ".template", // mail.template, account.chart.template
    ".config",   // *.config.settings
    ".policy",   // any *.policy
    ".rule",     // account.reconcile.model rules
    ".formula",  // hr.payroll.structure.line formulas
];

/// Classify an odoo model IRI / name onto its DOLCE upper category.
///
/// Accepts either a bare model name (`"res.partner"`) or a prefixed IRI
/// (`"odoo:res.partner"` / `"https://ada.world/onto/odoo#res.partner"`); the
/// prefix is stripped to the model name before matching.
///
/// Resolution order (first match wins):
/// 1. `product.template` special-case → [`DolceCategory::Endurant`] (odoo uses
///    "template" here for the master product record, not a config template).
/// 2. Perdurant suffix → [`DolceCategory::Perdurant`].
/// 3. Quality suffix → [`DolceCategory::Quality`].
/// 4. Abstract suffix → [`DolceCategory::AbstractEntity`].
/// 5. Default → [`DolceCategory::Endurant`] (persistent stateful object).
pub fn classify_odoo(iri: &str) -> DolceCategory {
    let model = model_name(iri);

    // (1) The single special-case: product.template is a master record
    // (Endurant), NOT an abstract config template — even though `.template`
    // is an Abstract suffix below. Must be checked before the suffix lists.
    if model == "product.template" {
        return DolceCategory::Endurant;
    }

    // (2) Perdurant — event / occurrence by suffix.
    for suffix in PERDURANT_SUFFIXES {
        if model.ends_with(suffix) {
            return DolceCategory::Perdurant;
        }
    }
    // (3) Quality — attribute / classification / rate by suffix.
    for suffix in QUALITY_SUFFIXES {
        if model.ends_with(suffix) {
            return DolceCategory::Quality;
        }
    }
    // (4) AbstractEntity — reference / config / template by suffix.
    for suffix in ABSTRACT_SUFFIXES {
        if model.ends_with(suffix) {
            return DolceCategory::AbstractEntity;
        }
    }

    // (5) Default: Endurant (res.partner, res.users, res.company,
    // product.product, account.account, account.journal, stock.warehouse,
    // crm.lead, hr.employee, …).
    DolceCategory::Endurant
}

/// Strip a leading `odoo:` prefix or the full odoo namespace IRI, returning the
/// bare odoo model name.
fn model_name(iri: &str) -> &str {
    if let Some(rest) = iri.strip_prefix("https://ada.world/onto/odoo#") {
        return rest;
    }
    iri.trim_start_matches("odoo:")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_iri_prefixes() {
        assert_eq!(model_name("odoo:res.partner"), "res.partner");
        assert_eq!(
            model_name("https://ada.world/onto/odoo#account.move"),
            "account.move"
        );
        assert_eq!(model_name("res.partner"), "res.partner");
    }

    #[test]
    fn classifier_handles_prefixed_iris() {
        assert_eq!(classify_odoo("odoo:account.move"), DolceCategory::Perdurant);
        assert_eq!(
            classify_odoo("https://ada.world/onto/odoo#res.partner"),
            DolceCategory::Endurant
        );
    }
}
