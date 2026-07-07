//! Auto-generated curated-vs-extracted pairing table (D-ODOO-EXT-5).
//!
//! For every `model_name` that appears in BOTH a curated lane module
//! (`l{1..15}`) AND a source-extracted module (`extracted::*`), this
//! file records the `(curated_const, extracted_const)` reference pair.
//!
//! Curated stays canonical on conflict (per `odoo-business-logic-blueprint-v1`
//! §"merge ordering"). Mismatches (field/method count deltas) are
//! recorded out-of-tree in `/tmp/pairings.json` for human review.
//! Plan: `.claude/plans/odoo-source-extraction-v1.md`.

use crate::odoo_blueprint::*;

/// One pairing: a model_name that has both a human-curated lane entity
/// (`OdooConfidence::Curated`) and at least one source-extracted entity
/// (`OdooConfidence::Extracted`).
#[derive(Debug, Clone, Copy)]
pub struct OdooEntityPairing {
    pub model_name: &'static str,
    /// Pointer to the curated lane const (canonical reference).
    pub curated: &'static OdooEntity,
    /// Pointer to the extracted const (the source-truth backing).
    pub extracted: &'static OdooEntity,
}

pub static CURATED_EXTRACTED_PAIRS: &[OdooEntityPairing] = &[
    // account.account  (curated: ACCOUNT_ACCOUNT in l11.rs  |  extracted: EXT_ACCOUNT_ACCOUNT in account.rs)
    // delta: fields=+19, methods=+67
    OdooEntityPairing {
        model_name: "account.account",
        curated: &crate::odoo_blueprint::l11::ACCOUNT_ACCOUNT,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_ACCOUNT,
    },
    // account.account.tag  (curated: ACCOUNT_ACCOUNT_TAG in l11.rs  |  extracted: EXT_ACCOUNT_ACCOUNT_TAG in account.rs)
    // delta: fields=+2, methods=+8
    OdooEntityPairing {
        model_name: "account.account.tag",
        curated: &crate::odoo_blueprint::l11::ACCOUNT_ACCOUNT_TAG,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_ACCOUNT_TAG,
    },
    // account.analytic.account  (curated: ANALYTIC_ACCOUNT in l10.rs  |  extracted: EXT_ACCOUNT_ANALYTIC_ACCOUNT in analytic.rs)
    // delta: fields=+5, methods=+6
    OdooEntityPairing {
        model_name: "account.analytic.account",
        curated: &crate::odoo_blueprint::l10::ANALYTIC_ACCOUNT,
        extracted: &crate::odoo_blueprint::extracted::analytic::EXT_ACCOUNT_ANALYTIC_ACCOUNT,
    },
    // account.analytic.applicability  (curated: ANALYTIC_APPLICABILITY in l10.rs  |  extracted: EXT_ACCOUNT_ANALYTIC_APPLICABILITY in account.rs)
    // delta: fields=-1, methods=+2
    OdooEntityPairing {
        model_name: "account.analytic.applicability",
        curated: &crate::odoo_blueprint::l10::ANALYTIC_APPLICABILITY,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_ANALYTIC_APPLICABILITY,
    },
    // account.analytic.distribution.model  (curated: ANALYTIC_DISTRIBUTION_MODEL in l10.rs  |  extracted: EXT_ACCOUNT_ANALYTIC_DISTRIBUTION_MODEL in analytic.rs)
    // delta: fields=-2, methods=+0
    OdooEntityPairing {
        model_name: "account.analytic.distribution.model",
        curated: &crate::odoo_blueprint::l10::ANALYTIC_DISTRIBUTION_MODEL,
        extracted:
            &crate::odoo_blueprint::extracted::analytic::EXT_ACCOUNT_ANALYTIC_DISTRIBUTION_MODEL,
    },
    // account.analytic.line  (curated: ANALYTIC_LINE in l10.rs  |  extracted: EXT_ACCOUNT_ANALYTIC_LINE in analytic.rs)
    // delta: fields=+2, methods=-1
    OdooEntityPairing {
        model_name: "account.analytic.line",
        curated: &crate::odoo_blueprint::l10::ANALYTIC_LINE,
        extracted: &crate::odoo_blueprint::extracted::analytic::EXT_ACCOUNT_ANALYTIC_LINE,
    },
    // account.analytic.plan  (curated: ANALYTIC_PLAN in l10.rs  |  extracted: EXT_ACCOUNT_ANALYTIC_PLAN in analytic.rs)
    // delta: fields=+10, methods=+25
    OdooEntityPairing {
        model_name: "account.analytic.plan",
        curated: &crate::odoo_blueprint::l10::ANALYTIC_PLAN,
        extracted: &crate::odoo_blueprint::extracted::analytic::EXT_ACCOUNT_ANALYTIC_PLAN,
    },
    // account.fiscal.position  (curated: ACCOUNT_FISCAL_POSITION in l9.rs  |  extracted: EXT_ACCOUNT_FISCAL_POSITION in account.rs)
    // delta: fields=+5, methods=+10
    OdooEntityPairing {
        model_name: "account.fiscal.position",
        curated: &crate::odoo_blueprint::l9::ACCOUNT_FISCAL_POSITION,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_FISCAL_POSITION,
    },
    // account.fiscal.position.account  (curated: ACCOUNT_FISCAL_POS_ACCOUNT in l9.rs  |  extracted: EXT_ACCOUNT_FISCAL_POSITION_ACCOUNT in account.rs)
    // delta: fields=+0, methods=+0
    OdooEntityPairing {
        model_name: "account.fiscal.position.account",
        curated: &crate::odoo_blueprint::l9::ACCOUNT_FISCAL_POS_ACCOUNT,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_FISCAL_POSITION_ACCOUNT,
    },
    // account.full.reconcile  (curated: ACCOUNT_FULL_RECONCILE in l2.rs  |  extracted: EXT_ACCOUNT_FULL_RECONCILE in account.rs)
    // delta: fields=+0, methods=+0
    OdooEntityPairing {
        model_name: "account.full.reconcile",
        curated: &crate::odoo_blueprint::l2::ACCOUNT_FULL_RECONCILE,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_FULL_RECONCILE,
    },
    // account.journal  (curated: ACCOUNT_JOURNAL in l11.rs  |  extracted: EXT_ACCOUNT_JOURNAL in account.rs)
    // delta: fields=+34, methods=+59
    OdooEntityPairing {
        model_name: "account.journal",
        curated: &crate::odoo_blueprint::l11::ACCOUNT_JOURNAL,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_JOURNAL,
    },
    // account.move  (curated: ACCOUNT_MOVE in l1.rs  |  extracted: EXT_ACCOUNT_MOVE in account.rs)
    // delta: fields=+118, methods=+325
    OdooEntityPairing {
        model_name: "account.move",
        curated: &crate::odoo_blueprint::l1::ACCOUNT_MOVE,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_MOVE,
    },
    // account.move.line  (curated: ACCOUNT_MOVE_LINE in l1.rs  |  extracted: EXT_ACCOUNT_MOVE_LINE in account.rs)
    // delta: fields=+67, methods=+132
    OdooEntityPairing {
        model_name: "account.move.line",
        curated: &crate::odoo_blueprint::l1::ACCOUNT_MOVE_LINE,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_MOVE_LINE,
    },
    // account.partial.reconcile  (curated: ACCOUNT_PARTIAL_RECONCILE in l2.rs  |  extracted: EXT_ACCOUNT_PARTIAL_RECONCILE in account.rs)
    // delta: fields=+1, methods=+11
    OdooEntityPairing {
        model_name: "account.partial.reconcile",
        curated: &crate::odoo_blueprint::l2::ACCOUNT_PARTIAL_RECONCILE,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_PARTIAL_RECONCILE,
    },
    // account.payment  (curated: PAYMENT in l5.rs  |  extracted: EXT_ACCOUNT_PAYMENT in account.rs)
    // delta: fields=+27, methods=+52
    OdooEntityPairing {
        model_name: "account.payment",
        curated: &crate::odoo_blueprint::l5::PAYMENT,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_PAYMENT,
    },
    // account.payment.term  (curated: PAYMENT_TERM in l5.rs  |  extracted: EXT_ACCOUNT_PAYMENT_TERM in account.rs)
    // delta: fields=+8, methods=+11
    OdooEntityPairing {
        model_name: "account.payment.term",
        curated: &crate::odoo_blueprint::l5::PAYMENT_TERM,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_PAYMENT_TERM,
    },
    // account.payment.term.line  (curated: PAYMENT_TERM_LINE in l5.rs  |  extracted: EXT_ACCOUNT_PAYMENT_TERM_LINE in account.rs)
    // delta: fields=+1, methods=+2
    OdooEntityPairing {
        model_name: "account.payment.term.line",
        curated: &crate::odoo_blueprint::l5::PAYMENT_TERM_LINE,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_PAYMENT_TERM_LINE,
    },
    // account.reconcile.model  (curated: RECONCILE_MODEL in l5.rs  |  extracted: EXT_ACCOUNT_RECONCILE_MODEL in account.rs)
    // delta: fields=+3, methods=+2
    OdooEntityPairing {
        model_name: "account.reconcile.model",
        curated: &crate::odoo_blueprint::l5::RECONCILE_MODEL,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_RECONCILE_MODEL,
    },
    // account.reconcile.model.line  (curated: RECONCILE_MODEL_LINE in l5.rs  |  extracted: EXT_ACCOUNT_RECONCILE_MODEL_LINE in account.rs)
    // delta: fields=+2, methods=+1
    OdooEntityPairing {
        model_name: "account.reconcile.model.line",
        curated: &crate::odoo_blueprint::l5::RECONCILE_MODEL_LINE,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_RECONCILE_MODEL_LINE,
    },
    // account.tax  (curated: ACCOUNT_TAX_DATEV in l4.rs  |  extracted: EXT_ACCOUNT_TAX in account.rs)
    // delta: fields=+33, methods=+113
    OdooEntityPairing {
        model_name: "account.tax",
        curated: &crate::odoo_blueprint::l4::ACCOUNT_TAX_DATEV,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_TAX,
    },
    // account.tax.group  (curated: ACCOUNT_TAX_GROUP in l3.rs  |  extracted: EXT_ACCOUNT_TAX_GROUP in account.rs)
    // delta: fields=+10, methods=+1
    OdooEntityPairing {
        model_name: "account.tax.group",
        curated: &crate::odoo_blueprint::l3::ACCOUNT_TAX_GROUP,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_TAX_GROUP,
    },
    // account.tax.repartition.line  (curated: ACCOUNT_TAX_REPARTITION_LINE in l3.rs  |  extracted: EXT_ACCOUNT_TAX_REPARTITION_LINE in account.rs)
    // delta: fields=+11, methods=+5
    OdooEntityPairing {
        model_name: "account.tax.repartition.line",
        curated: &crate::odoo_blueprint::l3::ACCOUNT_TAX_REPARTITION_LINE,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_ACCOUNT_TAX_REPARTITION_LINE,
    },
    // product.category  (curated: PRODUCT_CATEGORY in l8.rs  |  extracted: EXT_PRODUCT_CATEGORY in product.rs)
    // delta: fields=+7, methods=+6
    OdooEntityPairing {
        model_name: "product.category",
        curated: &crate::odoo_blueprint::l8::PRODUCT_CATEGORY,
        extracted: &crate::odoo_blueprint::extracted::product::EXT_PRODUCT_CATEGORY,
    },
    // product.pricelist  (curated: PRODUCT_PRICELIST in l8.rs  |  extracted: EXT_PRODUCT_PRICELIST in product.rs)
    // delta: fields=+7, methods=+22
    OdooEntityPairing {
        model_name: "product.pricelist",
        curated: &crate::odoo_blueprint::l8::PRODUCT_PRICELIST,
        extracted: &crate::odoo_blueprint::extracted::product::EXT_PRODUCT_PRICELIST,
    },
    // product.pricelist.item  (curated: PRODUCT_PRICELIST_ITEM in l8.rs  |  extracted: EXT_PRODUCT_PRICELIST_ITEM in product.rs)
    // delta: fields=+28, methods=+32
    OdooEntityPairing {
        model_name: "product.pricelist.item",
        curated: &crate::odoo_blueprint::l8::PRODUCT_PRICELIST_ITEM,
        extracted: &crate::odoo_blueprint::extracted::product::EXT_PRODUCT_PRICELIST_ITEM,
    },
    // product.product  (curated: PRODUCT_PRODUCT in l8.rs  |  extracted: EXT_PRODUCT_PRODUCT in product.rs)
    // delta: fields=+37, methods=+71
    OdooEntityPairing {
        model_name: "product.product",
        curated: &crate::odoo_blueprint::l8::PRODUCT_PRODUCT,
        extracted: &crate::odoo_blueprint::extracted::product::EXT_PRODUCT_PRODUCT,
    },
    // product.template  (curated: PRODUCT_TEMPLATE_DE in l4.rs  |  extracted: EXT_PRODUCT_TEMPLATE in product.rs)
    // delta: fields=+43, methods=+99
    OdooEntityPairing {
        model_name: "product.template",
        curated: &crate::odoo_blueprint::l4::PRODUCT_TEMPLATE_DE,
        extracted: &crate::odoo_blueprint::extracted::product::EXT_PRODUCT_TEMPLATE,
    },
    // purchase.order  (curated: PURCHASE_ORDER in l6.rs  |  extracted: EXT_PURCHASE_ORDER in purchase.rs)
    // delta: fields=+26, methods=+74
    OdooEntityPairing {
        model_name: "purchase.order",
        curated: &crate::odoo_blueprint::l6::PURCHASE_ORDER,
        extracted: &crate::odoo_blueprint::extracted::purchase::EXT_PURCHASE_ORDER,
    },
    // purchase.order.line  (curated: PURCHASE_ORDER_LINE in l6.rs  |  extracted: EXT_PURCHASE_ORDER_LINE in purchase.rs)
    // delta: fields=+29, methods=+42
    OdooEntityPairing {
        model_name: "purchase.order.line",
        curated: &crate::odoo_blueprint::l6::PURCHASE_ORDER_LINE,
        extracted: &crate::odoo_blueprint::extracted::purchase::EXT_PURCHASE_ORDER_LINE,
    },
    // res.company  (curated: RES_COMPANY_LOCK_DATE in l11.rs  |  extracted: EXT_RES_COMPANY in account.rs)
    // delta: fields=+67, methods=+45
    OdooEntityPairing {
        model_name: "res.company",
        curated: &crate::odoo_blueprint::l11::RES_COMPANY_LOCK_DATE,
        extracted: &crate::odoo_blueprint::extracted::account::EXT_RES_COMPANY,
    },
    // res.country  (curated: RES_COUNTRY in l9.rs  |  extracted: EXT_RES_COUNTRY in base.rs)
    // delta: fields=+11, methods=+9
    OdooEntityPairing {
        model_name: "res.country",
        curated: &crate::odoo_blueprint::l9::RES_COUNTRY,
        extracted: &crate::odoo_blueprint::extracted::base::EXT_RES_COUNTRY,
    },
    // res.country.group  (curated: RES_COUNTRY_GROUP in l9.rs  |  extracted: EXT_RES_COUNTRY_GROUP in base.rs)
    // delta: fields=+0, methods=+3
    OdooEntityPairing {
        model_name: "res.country.group",
        curated: &crate::odoo_blueprint::l9::RES_COUNTRY_GROUP,
        extracted: &crate::odoo_blueprint::extracted::base::EXT_RES_COUNTRY_GROUP,
    },
    // res.currency  (curated: RES_CURRENCY in l12.rs  |  extracted: EXT_RES_CURRENCY in base.rs)
    // delta: fields=+5, methods=+11
    OdooEntityPairing {
        model_name: "res.currency",
        curated: &crate::odoo_blueprint::l12::RES_CURRENCY,
        extracted: &crate::odoo_blueprint::extracted::base::EXT_RES_CURRENCY,
    },
    // res.currency.rate  (curated: RES_CURRENCY_RATE in l12.rs  |  extracted: EXT_RES_CURRENCY_RATE in base.rs)
    // delta: fields=+2, methods=+15
    OdooEntityPairing {
        model_name: "res.currency.rate",
        curated: &crate::odoo_blueprint::l12::RES_CURRENCY_RATE,
        extracted: &crate::odoo_blueprint::extracted::base::EXT_RES_CURRENCY_RATE,
    },
    // res.partner  (curated: RES_PARTNER_ACCOUNTING in l9.rs  |  extracted: EXT_RES_PARTNER in base.rs)
    // delta: fields=+23, methods=+68
    OdooEntityPairing {
        model_name: "res.partner",
        curated: &crate::odoo_blueprint::l9::RES_PARTNER_ACCOUNTING,
        extracted: &crate::odoo_blueprint::extracted::base::EXT_RES_PARTNER,
    },
    // res.users  (curated: RES_USERS_COMPANY_ACCESS in l12.rs  |  extracted: EXT_RES_USERS in base.rs)
    // delta: fields=+28, methods=+89
    OdooEntityPairing {
        model_name: "res.users",
        curated: &crate::odoo_blueprint::l12::RES_USERS_COMPANY_ACCESS,
        extracted: &crate::odoo_blueprint::extracted::base::EXT_RES_USERS,
    },
    // sale.order  (curated: SALE_ORDER in l6.rs  |  extracted: EXT_SALE_ORDER in sale.rs)
    // delta: fields=+43, methods=+128
    OdooEntityPairing {
        model_name: "sale.order",
        curated: &crate::odoo_blueprint::l6::SALE_ORDER,
        extracted: &crate::odoo_blueprint::extracted::sale::EXT_SALE_ORDER,
    },
    // sale.order.line  (curated: SALE_ORDER_LINE in l6.rs  |  extracted: EXT_SALE_ORDER_LINE in sale.rs)
    // delta: fields=+47, methods=+89
    OdooEntityPairing {
        model_name: "sale.order.line",
        curated: &crate::odoo_blueprint::l6::SALE_ORDER_LINE,
        extracted: &crate::odoo_blueprint::extracted::sale::EXT_SALE_ORDER_LINE,
    },
    // stock.location  (curated: STOCK_LOCATION in l7.rs  |  extracted: EXT_STOCK_LOCATION in stock.rs)
    // delta: fields=+18, methods=+24
    OdooEntityPairing {
        model_name: "stock.location",
        curated: &crate::odoo_blueprint::l7::STOCK_LOCATION,
        extracted: &crate::odoo_blueprint::extracted::stock::EXT_STOCK_LOCATION,
    },
    // stock.lot  (curated: STOCK_LOT in l13.rs  |  extracted: EXT_STOCK_LOT in stock.rs)
    // delta: fields=+8, methods=+22
    OdooEntityPairing {
        model_name: "stock.lot",
        curated: &crate::odoo_blueprint::l13::STOCK_LOT,
        extracted: &crate::odoo_blueprint::extracted::stock::EXT_STOCK_LOT,
    },
    // stock.move  (curated: STOCK_MOVE in l7.rs  |  extracted: EXT_STOCK_MOVE in stock.rs)
    // delta: fields=+51, methods=+116
    OdooEntityPairing {
        model_name: "stock.move",
        curated: &crate::odoo_blueprint::l7::STOCK_MOVE,
        extracted: &crate::odoo_blueprint::extracted::stock::EXT_STOCK_MOVE,
    },
    // stock.move.line  (curated: STOCK_MOVE_LINE in l7.rs  |  extracted: EXT_STOCK_MOVE_LINE in stock.rs)
    // delta: fields=+33, methods=+48
    OdooEntityPairing {
        model_name: "stock.move.line",
        curated: &crate::odoo_blueprint::l7::STOCK_MOVE_LINE,
        extracted: &crate::odoo_blueprint::extracted::stock::EXT_STOCK_MOVE_LINE,
    },
    // stock.picking  (curated: STOCK_PICKING in l7.rs  |  extracted: EXT_STOCK_PICKING in stock.rs)
    // delta: fields=+44, methods=+89
    OdooEntityPairing {
        model_name: "stock.picking",
        curated: &crate::odoo_blueprint::l7::STOCK_PICKING,
        extracted: &crate::odoo_blueprint::extracted::stock::EXT_STOCK_PICKING,
    },
    // stock.quant  (curated: STOCK_QUANT in l7.rs  |  extracted: EXT_STOCK_QUANT in stock.rs)
    // delta: fields=+18, methods=+62
    OdooEntityPairing {
        model_name: "stock.quant",
        curated: &crate::odoo_blueprint::l7::STOCK_QUANT,
        extracted: &crate::odoo_blueprint::extracted::stock::EXT_STOCK_QUANT,
    },
    // stock.rule  (curated: STOCK_RULE in l13.rs  |  extracted: EXT_STOCK_RULE in stock.rs)
    // delta: fields=+11, methods=+25
    OdooEntityPairing {
        model_name: "stock.rule",
        curated: &crate::odoo_blueprint::l13::STOCK_RULE,
        extracted: &crate::odoo_blueprint::extracted::stock::EXT_STOCK_RULE,
    },
    // stock.warehouse  (curated: STOCK_WAREHOUSE in l7.rs  |  extracted: EXT_STOCK_WAREHOUSE in stock.rs)
    // delta: fields=+20, methods=+42
    OdooEntityPairing {
        model_name: "stock.warehouse",
        curated: &crate::odoo_blueprint::l7::STOCK_WAREHOUSE,
        extracted: &crate::odoo_blueprint::extracted::stock::EXT_STOCK_WAREHOUSE,
    },
    // stock.warehouse.orderpoint  (curated: STOCK_WAREHOUSE_ORDERPOINT in l13.rs  |  extracted: EXT_STOCK_WAREHOUSE_ORDERPOINT in stock.rs)
    // delta: fields=+19, methods=+43
    OdooEntityPairing {
        model_name: "stock.warehouse.orderpoint",
        curated: &crate::odoo_blueprint::l13::STOCK_WAREHOUSE_ORDERPOINT,
        extracted: &crate::odoo_blueprint::extracted::stock::EXT_STOCK_WAREHOUSE_ORDERPOINT,
    },
    // uom.uom  (curated: UOM_UOM in l8.rs  |  extracted: EXT_UOM_UOM in uom.rs)
    // delta: fields=+9, methods=+16
    OdooEntityPairing {
        model_name: "uom.uom",
        curated: &crate::odoo_blueprint::l8::UOM_UOM,
        extracted: &crate::odoo_blueprint::extracted::uom::EXT_UOM_UOM,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::OdooConfidence;

    #[test]
    fn pairing_table_is_well_formed() {
        for pair in CURATED_EXTRACTED_PAIRS {
            assert_eq!(
                pair.model_name, pair.curated.model_name,
                "Curated entity model_name mismatch for {}",
                pair.model_name,
            );
            assert_eq!(
                pair.model_name, pair.extracted.model_name,
                "Extracted entity model_name mismatch for {}",
                pair.model_name,
            );
            assert_eq!(
                pair.curated.provenance.confidence,
                OdooConfidence::Curated,
                "Curated confidence wrong for {}",
                pair.model_name,
            );
            assert_eq!(
                pair.extracted.provenance.confidence,
                OdooConfidence::Extracted,
                "Extracted confidence wrong for {}",
                pair.model_name,
            );
        }
    }

    #[test]
    fn pairing_table_has_expected_size() {
        // EXT-5 inventory: 48 model_name overlaps across TIER-1.
        // Adjust if the actual count differs; commit body should explain drift.
        assert!(
            CURATED_EXTRACTED_PAIRS.len() >= 40,
            "Pairing table thinner than expected: {}",
            CURATED_EXTRACTED_PAIRS.len(),
        );
    }
}
