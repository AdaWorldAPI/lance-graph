//! Lane L4 (K8K9-REPORTS-DATEV) — typed Odoo entity declarations.
//!
//! Source: `.claude/odoo/L4-K8K9-REPORTS-DATEV.md`.
//!
//! **K-steps covered:** K8 (BWA/SuSa/EÜR/GuV/Bilanz/USt-VA report structure),
//! K9 (DATEV EXTF export + Steuerschlüssel field).
//!
//! ## Entity inventory (6 entities)
//!
//! | Const | Odoo model | Section in L-doc |
//! |---|---|---|
//! | [`ACCOUNT_ACCOUNT_TAG`] | `account.account.tag` | §3–5 (USt-VA + GuV + Bilanz tags) |
//! | [`ACCOUNT_ACCOUNT_DE`] | `account.account` (l10n_de) | §6 (code-lock constraint) |
//! | [`ACCOUNT_TAX_DATEV`] | `account.tax` (l10n_de) | §8 (DATEV Steuerschlüssel field) |
//! | [`PRODUCT_TEMPLATE_DE`] | `product.template` (l10n_de) | §8.1 (income/expense account routing) |
//! | [`RES_COMPANY_DE`] | `res.company` (l10n_de chart template) | §7 (audit-trail + DIN 5008 setup) |
//! | [`ACCOUNT_JOURNAL_DE`] | `account.journal` (l10n_de) | §7 (liquidity tag auto-assignment) |

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooEntityKind, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind,
    OdooProvenance, OdooReturnKind, OdooSemanticRole, OdooSourceRef,
};

// ─── 1. account.account.tag ──────────────────────────────────────────────────
//
// The 60 account.account.tag records are STATIC DATA in community (XML).
// They are the sole driver of the German USt-VA (35 leaf lines + 8 aggregations),
// GuV (21 tags: pl_01..pl_15 where pl_08 splits into 8.1–8.7), and Bilanz
// (14 Aktiva + 22 Passiva = 36 position tags).
//
// Engine: Enterprise `account.report` reads these tags and routes account
// balances to the correct financial-statement line.  The data itself (all 60
// tag XML records) is community.

/// `account.account.tag` — classification labels that drive German financial
/// report line routing (USt-VA / GuV / Bilanz).
///
/// L-doc §§3–5; source: `l10n_de/data/account_account_tags_data.xml:L3-L1106`.
pub const ACCOUNT_ACCOUNT_TAG: OdooEntity = OdooEntity {
    model_name: "account.account.tag",
    kind: OdooEntityKind::Model,
    description: "SKOS-style classification label applied to accounts or taxes; \
                  routes balances to USt-VA / GuV / Bilanz report lines (HGB §275 / §266)",
    fields: &[
        OdooField {
            name: "name",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "applicability",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // Distinguishes tax-applicability ("taxes") from account-applicability
            // ("accounts").  Only "accounts" tags drive report-line routing.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "color",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "country_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.country"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            // For USt-VA lines: `engine="tax_tags"` formula strings such as
            // "-81_BASE" or "+89_TAX".  Sign polarity is load-bearing:
            // negative = credit-side (sales tax), positive = debit-side (input
            // tax).  See L-doc §3 §Porter-gotcha-1.
            name: "tax_negate",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
    ],
    methods: &[],
    decorators: &[],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Sql,
        condition: "unique(name, country_id, applicability) — tag name is \
                    unique per country+applicability combination",
        source_method: None,
    }],
    provenance: OdooProvenance {
        l_doc: "L4-K8K9-REPORTS-DATEV.md",
        l_doc_lines: (42, 295),
        odoo_source: &[OdooSourceRef {
            path: "addons/l10n_de/data/account_account_tags_data.xml",
            line_range: (3, 1106),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── 2. account.account (l10n_de) — code-lock constraint ─────────────────────
//
// Germany-specific override: once an account has posted move lines its
// account code (Kontonummer) must not change — GoBD §§238/239 HGB.
// The lock is enforced at the application layer (UserError), NOT at the DB
// level.  See L-doc §6.

/// `account.account` — Germany-specific override enforcing the GoBD account
/// code immutability constraint once move lines exist.
///
/// L-doc §6; source: `l10n_de/models/account_account.py:L8-L19`.
pub const ACCOUNT_ACCOUNT_DE: OdooEntity = OdooEntity {
    model_name: "account.account",
    kind: OdooEntityKind::Model,
    description: "General ledger account extended by l10n_de: blocks code changes \
                  (Kontonummer) once the account has posted journal entry lines \
                  (GoBD §§238/239 HGB Buchführungspflicht)",
    fields: &[
        OdooField {
            name: "code",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // The Kontonummer — identity key that the code-lock protects.
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            // Bilanz/GuV position (e.g. "A.II.3") derived from the account's
            // tag_ids by the report engine.  Schema-neutral in woa-rs as
            // `bilanz_position: Option<String>`.
            name: "tag_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("account.account.tag"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "account_type",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // Determines Bilanz-Aktiva/Passiva routing independent of tag_ids.
            // Also governs DATEV account-group assignment.
            semantic_role: OdooSemanticRole::Policy,
        },
    ],
    methods: &[
        OdooMethod {
            // l10n_de override: raises UserError if any of the four conditions
            // trigger (code in vals AND DE company AND code differs AND move
            // lines exist).  Application-layer only — no DB constraint.
            name: "write",
            kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
    ],
    decorators: &[],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "account code must not change once any account.move.line \
                    references this account in a German company (UserError, not DB)",
        source_method: Some("write"),
    }],
    provenance: OdooProvenance {
        l_doc: "L4-K8K9-REPORTS-DATEV.md",
        l_doc_lines: (296, 370),
        odoo_source: &[OdooSourceRef {
            path: "addons/l10n_de/models/account_account.py",
            line_range: (8, 19),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── 3. account.tax (l10n_de DATEV extension) ────────────────────────────────
//
// Adds `l10n_de_datev_code`: a 4-character DATEV Steuerschlüssel stored on
// each tax record.  The community module only declares the field; the mapping
// logic (EXTF v700 tax keys 0/2/3/8/9/10/13/18/19/21/39) must be implemented
// from the DATEV specification.  See L-doc §8.

/// `account.tax` — extended by l10n_de with DATEV Steuerschlüssel field.
///
/// L-doc §8; source: `l10n_de/models/datev.py:L1-L15`.
pub const ACCOUNT_TAX_DATEV: OdooEntity = OdooEntity {
    model_name: "account.tax",
    kind: OdooEntityKind::Model,
    description: "Tax record extended for DATEV export: carries a 4-character \
                  DATEV Steuerschlüssel (l10n_de_datev_code) used as the bridge \
                  from Odoo taxes to DATEV EXTF v700 tax key numbers",
    fields: &[
        OdooField {
            // 4-char DATEV Steuerschlüssel, e.g. "0003" (19% USt) or "0009"
            // (19% VSt).  User-entered; community stores it, mapping is external.
            // Representative values from DATEV spec (L-doc §8.2):
            //   "0"  = kein Steuerschlüssel
            //   "2"  = 7% USt,  "3"  = 19% USt
            //   "8"  = 7% VSt,  "9"  = 19% VSt
            //   "10" = ig. Lieferung steuerfreie
            //   "13" = §13b UStG Umkehr 19%
            //   "18" = 7% ig. Erwerb, "19" = 19% ig. Erwerb
            //   "21" = nicht steuerbar
            //   "39" = §24 UStG Pauschalsteuer
            name: "l10n_de_datev_code",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Tax,
        },
        OdooField {
            name: "amount",
            kind: OdooFieldKind::Float,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // Percentage rate (e.g. 19.0, 7.0, 0.0) — the threshold that drives
            // DATEV Steuerschlüssel routing (≥18.5 → key 3/9, ≥6.5 → key 2/8).
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "type_tax_use",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // "sale" vs "purchase" determines USt (sale) vs VSt (purchase)
            // side of the DATEV Steuerschlüssel pairing.
            semantic_role: OdooSemanticRole::Policy,
        },
    ],
    methods: &[],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L4-K8K9-REPORTS-DATEV.md",
        l_doc_lines: (368, 451),
        odoo_source: &[OdooSourceRef {
            path: "addons/l10n_de/models/datev.py",
            line_range: (1, 15),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── 4. product.template (l10n_de income/expense account routing) ─────────────
//
// Germany-specific `_get_product_accounts` override: when a product has no
// explicit income or expense account, the system searches for an account
// matching `internal_group + tax_ids`.  This means income accounts are keyed
// to their applicable tax rate (19% product → SKR03 8400, 7% → SKR03 8300).
// See L-doc §8.2.

/// `product.template` — l10n_de override routing income/expense accounts by
/// applicable tax rate when no explicit account is set.
///
/// L-doc §8.2; source: `l10n_de/models/datev.py:L17-L37`.
pub const PRODUCT_TEMPLATE_DE: OdooEntity = OdooEntity {
    model_name: "product.template",
    kind: OdooEntityKind::Model,
    description: "Product template extended by l10n_de: _get_product_accounts \
                  searches for a matching income/expense account by internal_group \
                  + tax_ids when no explicit account is configured — the Odoo side \
                  of the SKR03/04 Erlöskonto routing (8400/4400 for 19% USt, \
                  8300/4300 for 7% USt)",
    fields: &[
        OdooField {
            name: "property_account_income_id",
            kind: OdooFieldKind::Property,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "property_account_expense_id",
            kind: OdooFieldKind::Property,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "taxes_id",
            kind: OdooFieldKind::Many2many,
            target: Some("account.tax"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Tax,
        },
    ],
    methods: &[OdooMethod {
        // Returns dict {"income": account, "expense": account}.
        // DE override: if no property account is set, searches by
        // (internal_group, tax_ids) to find the rate-appropriate Erlöskonto /
        // Aufwandskonto.  Symmetric for expense/supplier taxes.
        name: "_get_product_accounts",
        kind: OdooMethodKind::Override,
        return_kind: OdooReturnKind::Dict,
        triggers: &[],
    }],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L4-K8K9-REPORTS-DATEV.md",
        l_doc_lines: (376, 422),
        odoo_source: &[OdooSourceRef {
            path: "addons/l10n_de/models/datev.py",
            line_range: (17, 37),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── 5. res.company (l10n_de chart-template setup) ───────────────────────────
//
// The chart-template hook sets `restrictive_audit_trail=True` (GoBD
// Festschreibung — §146/§239 HGB) and assigns the DIN 5008 report layout.
// It also auto-tags the suspense and transfer accounts with `B_II_4`.
// See L-doc §7.

/// `res.company` — l10n_de chart-template hook setting GoBD Festschreibung
/// flag and DIN 5008 layout on company creation.
///
/// L-doc §7; source: `l10n_de/models/chart_template.py:L9-L25`.
pub const RES_COMPANY_DE: OdooEntity = OdooEntity {
    model_name: "res.company",
    kind: OdooEntityKind::Model,
    description: "Company record augmented by l10n_de chart-template setup: \
                  restrictive_audit_trail=True (GoBD Festschreibung §§146/239 HGB), \
                  DIN 5008 paper layout, and auto-tagging of suspense/transfer \
                  accounts with tag_de_asset_bs_B_II_4",
    fields: &[
        OdooField {
            // GoBD Festschreibung flag: once True, posted entries cannot be
            // modified.  Both SKR03 and SKR04 activate this on company creation.
            // woa-rs analogue: ErpFiscalYearClose.status = 'festgestellt'.
            name: "restrictive_audit_trail",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "external_report_layout_id",
            kind: OdooFieldKind::Many2one,
            target: Some("report.layout"),
            required: false,
            computed: None,
            depends: &[],
            // l10n_din5008.external_layout_din5008 — German invoice format.
            semantic_role: OdooSemanticRole::Document,
        },
        OdooField {
            name: "paperformat_id",
            kind: OdooFieldKind::Many2one,
            target: Some("report.paperformat"),
            required: false,
            computed: None,
            depends: &[],
            // l10n_din5008.paperformat_euro_din — A4/DIN 5008.
            semantic_role: OdooSemanticRole::Document,
        },
        OdooField {
            name: "account_journal_suspense_account_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            // Suspense account auto-tagged B_II_4 (Sonstige Vermögensgegenstände).
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "transfer_account_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            // Transfer/clearing account auto-tagged B_II_4 (same catch-all tag).
            semantic_role: OdooSemanticRole::Reference,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_get_de_res_company",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            // Called once at chart-of-accounts install time.  Assigns B_II_4
            // to suspense + transfer accounts if template_code is de_skr03 or
            // de_skr04.
            name: "_setup_utility_bank_accounts",
            kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[OdooDecorator {
        // @template('de_skr03', 'res.company') + @template('de_skr04', 'res.company')
        // on _get_de_res_company — standard Odoo chart-template decorator.
        kind: OdooDecoratorKind::ApiModel,
        targets: &["de_skr03", "de_skr04"],
    }],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L4-K8K9-REPORTS-DATEV.md",
        l_doc_lines: (330, 367),
        odoo_source: &[OdooSourceRef {
            path: "addons/l10n_de/models/chart_template.py",
            line_range: (9, 25),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── 6. account.journal (l10n_de liquidity tag) ───────────────────────────────
//
// l10n_de overrides `_prepare_liquidity_account_vals` to append
// `tag_de_asset_bs_B_IV` (Kassenbestand/Bankguthaben) to bank/cash journal
// accounts at journal-creation time.  See L-doc §7 final bullet.

/// `account.journal` — l10n_de override tagging newly created liquidity
/// (bank/cash) accounts with `tag_de_asset_bs_B_IV` for Bilanz position
/// B IV (Kassenbestand, Bankguthaben, Schecks).
///
/// L-doc §7; source: `l10n_de/models/account_journal.py:L14-L16`.
pub const ACCOUNT_JOURNAL_DE: OdooEntity = OdooEntity {
    model_name: "account.journal",
    kind: OdooEntityKind::Model,
    description: "Journal model extended by l10n_de: bank/cash liquidity accounts \
                  receive tag_de_asset_bs_B_IV (Kassenbestand/Bankguthaben) at \
                  journal creation, routing them to Bilanz position B IV",
    fields: &[
        OdooField {
            name: "type",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // "bank" or "cash" — trigger for the B_IV tag assignment.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "default_account_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            // The liquidity account that receives tag_de_asset_bs_B_IV.
            semantic_role: OdooSemanticRole::Reference,
        },
    ],
    methods: &[OdooMethod {
        // Appends tag_de_asset_bs_B_IV to the liquidity account being created
        // when template_code is de_skr03 or de_skr04.  Setup-time mutation.
        name: "_prepare_liquidity_account_vals",
        kind: OdooMethodKind::Override,
        return_kind: OdooReturnKind::Dict,
        triggers: &[],
    }],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L4-K8K9-REPORTS-DATEV.md",
        l_doc_lines: (363, 367),
        odoo_source: &[OdooSourceRef {
            path: "addons/l10n_de/models/account_journal.py",
            line_range: (14, 16),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── ENTITIES slice ───────────────────────────────────────────────────────────

/// All 6 entities documented in lane L4 (K8 German financial reports + K9
/// DATEV export + GoBD audit-trail).
pub const ENTITIES: &[OdooEntity] = &[
    ACCOUNT_ACCOUNT_TAG,
    ACCOUNT_ACCOUNT_DE,
    ACCOUNT_TAX_DATEV,
    PRODUCT_TEMPLATE_DE,
    RES_COMPANY_DE,
    ACCOUNT_JOURNAL_DE,
];

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::{OdooConfidence, OdooFieldKind, OdooSemanticRole};

    #[test]
    fn entities_slice_has_six_entries() {
        assert_eq!(ENTITIES.len(), 6);
    }

    #[test]
    fn account_account_tag_identity() {
        assert_eq!(ACCOUNT_ACCOUNT_TAG.model_name, "account.account.tag");
        assert_eq!(ACCOUNT_ACCOUNT_TAG.provenance.confidence, OdooConfidence::Curated);
        assert_eq!(ACCOUNT_ACCOUNT_TAG.provenance.l_doc, "L4-K8K9-REPORTS-DATEV.md");
        // applicability field governs report-line routing vs tax-label role
        let applicability = ACCOUNT_ACCOUNT_TAG
            .fields
            .iter()
            .find(|f| f.name == "applicability")
            .expect("applicability field must be present");
        assert_eq!(applicability.kind, OdooFieldKind::Selection);
        assert_eq!(applicability.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn account_account_de_has_code_lock_constraint() {
        assert_eq!(ACCOUNT_ACCOUNT_DE.model_name, "account.account");
        let lock = ACCOUNT_ACCOUNT_DE
            .constraints
            .first()
            .expect("code-lock Python constraint must be present");
        assert_eq!(lock.kind, OdooConstraintKind::Python);
        assert_eq!(lock.source_method, Some("write"));
    }

    #[test]
    fn account_tax_datev_has_steuerschluessel_field() {
        assert_eq!(ACCOUNT_TAX_DATEV.model_name, "account.tax");
        let datev_code = ACCOUNT_TAX_DATEV
            .fields
            .iter()
            .find(|f| f.name == "l10n_de_datev_code")
            .expect("l10n_de_datev_code field must be present");
        assert_eq!(datev_code.kind, OdooFieldKind::Char);
        assert_eq!(datev_code.semantic_role, OdooSemanticRole::Tax);
    }

    #[test]
    fn product_template_de_has_get_product_accounts_override() {
        assert_eq!(PRODUCT_TEMPLATE_DE.model_name, "product.template");
        let method = PRODUCT_TEMPLATE_DE
            .methods
            .iter()
            .find(|m| m.name == "_get_product_accounts")
            .expect("_get_product_accounts override must be present");
        assert_eq!(method.kind, OdooMethodKind::Override);
        assert_eq!(method.return_kind, OdooReturnKind::Dict);
    }

    #[test]
    fn res_company_de_audit_trail_policy() {
        assert_eq!(RES_COMPANY_DE.model_name, "res.company");
        let trail = RES_COMPANY_DE
            .fields
            .iter()
            .find(|f| f.name == "restrictive_audit_trail")
            .expect("restrictive_audit_trail field must be present");
        assert_eq!(trail.kind, OdooFieldKind::Boolean);
        // This IS a policy field — it governs the GoBD Festschreibung lock.
        assert_eq!(trail.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn account_journal_de_liquidity_tag_method() {
        assert_eq!(ACCOUNT_JOURNAL_DE.model_name, "account.journal");
        let method = ACCOUNT_JOURNAL_DE
            .methods
            .iter()
            .find(|m| m.name == "_prepare_liquidity_account_vals")
            .expect("_prepare_liquidity_account_vals override must be present");
        assert_eq!(method.kind, OdooMethodKind::Override);
    }

    #[test]
    fn all_entities_have_curated_confidence() {
        for e in ENTITIES {
            assert_eq!(
                e.provenance.confidence,
                OdooConfidence::Curated,
                "entity {} must be Curated",
                e.model_name
            );
        }
    }

    #[test]
    fn all_entities_reference_l4_l_doc() {
        for e in ENTITIES {
            assert_eq!(
                e.provenance.l_doc,
                "L4-K8K9-REPORTS-DATEV.md",
                "entity {} must reference L4 l_doc",
                e.model_name
            );
        }
    }
}
