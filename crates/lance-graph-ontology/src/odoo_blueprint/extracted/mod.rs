//! Auto-generated source-extracted `OdooEntity` consts from Odoo's Python ORM,
//! produced by `tools/odoo-blueprint-extractor` (D-ODOO-EXT-1) per
//! `D-ODOO-EXT-2` of `.claude/plans/odoo-source-extraction-v1.md`.
//!
//! These are additive to the curated `l{1..15}` lane modules — they carry
//! `OdooConfidence::Extracted` and `EXT_*` const prefixes to avoid symbol
//! collisions with curated lane consts. The curated set stays canonical
//! on conflict (per BP-1 plan §"merge ordering"); pairing is wired by
//! `D-ODOO-EXT-5`.

// Wave A (foundation)
pub mod analytic;
pub mod base;
pub mod product;
pub mod uom;

// Wave B (value-flow chain)
pub mod account;
pub mod account_payment;
pub mod purchase;
pub mod sale;
pub mod stock;

// Wave C (DE-specific + e-invoice)
pub mod account_edi_ubl_cii;
pub mod account_peppol;
pub mod l10n_de;
