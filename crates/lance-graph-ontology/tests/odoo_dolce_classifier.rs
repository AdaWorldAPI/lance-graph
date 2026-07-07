//! Seam decision 2 — the full odoo DOLCE suffix-classifier test matrix.
//!
//! Pins every row of the matrix in
//! `woa-rs/.claude/reference/four_way_alignment_seam.md` §"Seam decision 2"
//! plus the prefixed-IRI accessor. The `product.template` special-case
//! (Endurant despite the `.template` Abstract suffix) is the single override
//! the seam calls out.

use lance_graph_ontology::{classify_odoo, DolceCategory};

#[test]
fn classify_odoo_matrix_matches_seam_table() {
    use DolceCategory::*;

    // (odoo model, expected category) — verbatim from the seam doc's 21-row
    // table, in the same order.
    let matrix: &[(&str, DolceCategory)] = &[
        ("res.partner", Endurant),                  // persistent stateful object
        ("res.users", Endurant),                    // persistent stateful object
        ("res.company", Endurant),                  // persistent stateful object
        ("account.move", Perdurant),                // .move — the journal-entry event
        ("account.move.line", Perdurant),           // child of Perdurant; fact within event
        ("account.account", Endurant),              // chart-of-accounts entry persists
        ("account.tax", Quality),                   // .tax — a rate, characterises a txn
        ("account.journal", Endurant),              // persistent container
        ("product.product", Endurant),              // the product is the stateful thing
        ("product.template", Endurant),             // SPECIAL-CASE: master record, not config
        ("product.category", Quality),              // classification
        ("stock.move", Perdurant),                  // .move — the stock event
        ("stock.picking", Perdurant),               // .picking — the logistic event
        ("stock.warehouse", Endurant),              // the physical warehouse
        ("mail.message", Perdurant),                // .message — communication event
        ("crm.lead", Endurant),                     // the lead persists
        ("crm.tag", Quality),                       // classification
        ("hr.employee", Endurant),                  // persistent person
        ("hr.attendance", Perdurant),               // .attendance — an attendance event
        ("mail.template", AbstractEntity),          // .template — true template/config
        ("account.chart.template", AbstractEntity), // .template — chart template
    ];

    for (model, expected) in matrix {
        assert_eq!(
            classify_odoo(model),
            *expected,
            "classify_odoo({model:?}) should be {expected:?}"
        );
    }

    // Sanity: the matrix is the full ~20-row seam table.
    assert_eq!(matrix.len(), 21, "the seam matrix has 21 rows");
}

#[test]
fn product_template_special_case_overrides_template_suffix() {
    // product.template ends with `.template` (an Abstract suffix) but odoo uses
    // it for the master product record — the seam's single special-case.
    assert_eq!(classify_odoo("product.template"), DolceCategory::Endurant);
    // ...while other `.template` models stay Abstract.
    assert_eq!(
        classify_odoo("mail.template"),
        DolceCategory::AbstractEntity
    );
    assert_eq!(
        classify_odoo("account.chart.template"),
        DolceCategory::AbstractEntity
    );
}

#[test]
fn unknown_model_defaults_to_endurant() {
    // The seam's "Default: Endurant" rule — no Unknown variant.
    assert_eq!(classify_odoo("ir.cron"), DolceCategory::Endurant);
    assert_eq!(classify_odoo("res.country"), DolceCategory::Endurant);
}

#[test]
fn accepts_prefixed_iris() {
    assert_eq!(classify_odoo("odoo:account.move"), DolceCategory::Perdurant);
    assert_eq!(
        classify_odoo("https://ada.world/onto/odoo#product.category"),
        DolceCategory::Quality
    );
}
