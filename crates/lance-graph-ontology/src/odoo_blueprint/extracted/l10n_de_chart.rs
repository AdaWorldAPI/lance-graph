//! Auto-generated from l10n_de SKR03/SKR04 CSVs by `tools/odoo-blueprint-extractor`.
//! Do NOT edit by hand — re-run: `python -m odoo_blueprint_extractor data --addon l10n_de --out <dir>`
//! Plan: `.claude/plans/odoo-source-extraction-v1.md` (D-ODOO-EXT-4).
//!
//! Provenance:
//!   `/home/user/odoo/addons/l10n_de/data/template/account.account-de_skr03.csv`
//!   `/home/user/odoo/addons/l10n_de/data/template/account.account-de_skr04.csv`

use crate::odoo_blueprint::{OdooAccountTemplate, OdooSkrChart};

// ─── SKR03 account template consts ─────────────────────────────────────

pub const EXT_SKR03_0005: OdooAccountTemplate = OdooAccountTemplate {
    code: "0005",
    name: "Rückständige fällige Einzahlungen auf Geschäftsanteile",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0010: OdooAccountTemplate = OdooAccountTemplate {
    code: "0010",
    name: "Entgeltlich erworbene Konzessionen, gewerbliche Schutzrechte und ähnliche Rechte und Werte sowie Lizenzen an solchen Rechten und Werten",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0015: OdooAccountTemplate = OdooAccountTemplate {
    code: "0015",
    name: "Konzessionen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0020: OdooAccountTemplate = OdooAccountTemplate {
    code: "0020",
    name: "Gewerbliche Schutzrechte",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0025: OdooAccountTemplate = OdooAccountTemplate {
    code: "0025",
    name: "Ähnliche Rechte und Werte",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0027: OdooAccountTemplate = OdooAccountTemplate {
    code: "0027",
    name: "EDV-Software",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0030: OdooAccountTemplate = OdooAccountTemplate {
    code: "0030",
    name: "Lizenzen an gewerblichen Schutzrechten und ähnlichen Rechten und Werten",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0035: OdooAccountTemplate = OdooAccountTemplate {
    code: "0035",
    name: "Geschäfts- oder Firmenwert",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0038: OdooAccountTemplate = OdooAccountTemplate {
    code: "0038",
    name: "Anzahlungen auf Geschäfts- oder Firmenwert",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0039: OdooAccountTemplate = OdooAccountTemplate {
    code: "0039",
    name: "Geleistete Anzahlungen auf immaterielle Vermögensgegenstände",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0040: OdooAccountTemplate = OdooAccountTemplate {
    code: "0040",
    name: "Verschmelzungsmehrwert",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0043: OdooAccountTemplate = OdooAccountTemplate {
    code: "0043",
    name: "Selbst geschaffene immaterielle Vermögensgegenstände",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0045: OdooAccountTemplate = OdooAccountTemplate {
    code: "0045",
    name: "Lizenzen und Franchiseverträge",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0046: OdooAccountTemplate = OdooAccountTemplate {
    code: "0046",
    name: "Konzessionen und gewerbliche Schutzrechte",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0047: OdooAccountTemplate = OdooAccountTemplate {
    code: "0047",
    name: "Rezepte, Verfahren, Prototypen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0048: OdooAccountTemplate = OdooAccountTemplate {
    code: "0048",
    name: "Immaterielle Vermögensgegenstände in Entwicklung",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0050: OdooAccountTemplate = OdooAccountTemplate {
    code: "0050",
    name: "Grundstücke, grundstücksgleiche Rechte und Bauten einschließlich der Bauten auf fremden Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0059: OdooAccountTemplate = OdooAccountTemplate {
    code: "0059",
    name: "Grundstücksanteil des häuslichen Arbeitszimmers",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0060: OdooAccountTemplate = OdooAccountTemplate {
    code: "0060",
    name: "Grundstücksgleiche Rechte ohne Bauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0065: OdooAccountTemplate = OdooAccountTemplate {
    code: "0065",
    name: "Unbebaute Grundstücke",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0070: OdooAccountTemplate = OdooAccountTemplate {
    code: "0070",
    name: "Grundstücksgleiche Rechte (Erbbaurecht, Dauerwohnrecht, unbebaute Grundstücke)",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0075: OdooAccountTemplate = OdooAccountTemplate {
    code: "0075",
    name: "Grundstücke mit Substanzverzehr",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0079: OdooAccountTemplate = OdooAccountTemplate {
    code: "0079",
    name: "Anzahlungen auf Grund und Boden",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0080: OdooAccountTemplate = OdooAccountTemplate {
    code: "0080",
    name: "Bauten auf eigenen Grundstücken und grundstücksgleichen Rechten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0085: OdooAccountTemplate = OdooAccountTemplate {
    code: "0085",
    name: "Grundstückswerte eigener bebauter Grundstücke",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0090: OdooAccountTemplate = OdooAccountTemplate {
    code: "0090",
    name: "Geschäftsbauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0100: OdooAccountTemplate = OdooAccountTemplate {
    code: "0100",
    name: "Fabrikbauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0110: OdooAccountTemplate = OdooAccountTemplate {
    code: "0110",
    name: "Garagen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0111: OdooAccountTemplate = OdooAccountTemplate {
    code: "0111",
    name: "Außenanlagen für Geschäfts-, Fabrik- und andere Bauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0112: OdooAccountTemplate = OdooAccountTemplate {
    code: "0112",
    name: "Hof- und Wegebefestigungen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0113: OdooAccountTemplate = OdooAccountTemplate {
    code: "0113",
    name: "Einrichtungen für Geschäfts-, Fabrik- und andere Bauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0115: OdooAccountTemplate = OdooAccountTemplate {
    code: "0115",
    name: "Andere Bauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0120: OdooAccountTemplate = OdooAccountTemplate {
    code: "0120",
    name: "Geschäfts-, Fabrik- und andere Bauten im Bau auf eigenen Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0129: OdooAccountTemplate = OdooAccountTemplate {
    code: "0129",
    name: "Anzahlungen auf Geschäfts-, Fabrik- und andere Bauten auf eigenen Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0140: OdooAccountTemplate = OdooAccountTemplate {
    code: "0140",
    name: "Wohnbauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0145: OdooAccountTemplate = OdooAccountTemplate {
    code: "0145",
    name: "Garagen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0146: OdooAccountTemplate = OdooAccountTemplate {
    code: "0146",
    name: "Außenanlagen für Geschäfts-, Fabrik- und andere Bauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0147: OdooAccountTemplate = OdooAccountTemplate {
    code: "0147",
    name: "Hof- und Wegebefestigungen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0148: OdooAccountTemplate = OdooAccountTemplate {
    code: "0148",
    name: "Einrichtungen für Wohnbauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0149: OdooAccountTemplate = OdooAccountTemplate {
    code: "0149",
    name: "Gebäudeteil des häuslichen Arbeitszimmers",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0150: OdooAccountTemplate = OdooAccountTemplate {
    code: "0150",
    name: "Wohnbauten im Bau auf eigenen Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0159: OdooAccountTemplate = OdooAccountTemplate {
    code: "0159",
    name: "Anzahlungen auf Wohnbauten auf eigenen Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0160: OdooAccountTemplate = OdooAccountTemplate {
    code: "0160",
    name: "Bauten auf fremden Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0165: OdooAccountTemplate = OdooAccountTemplate {
    code: "0165",
    name: "Geschäftsbauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0170: OdooAccountTemplate = OdooAccountTemplate {
    code: "0170",
    name: "Fabrikbauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0175: OdooAccountTemplate = OdooAccountTemplate {
    code: "0175",
    name: "Garagen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0176: OdooAccountTemplate = OdooAccountTemplate {
    code: "0176",
    name: "Außenanlagen für Geschäfts-, Fabrik- und andere Bauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0177: OdooAccountTemplate = OdooAccountTemplate {
    code: "0177",
    name: "Hof- und Wegebefestigungen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0178: OdooAccountTemplate = OdooAccountTemplate {
    code: "0178",
    name: "Einrichtungen für Geschäfts-, Fabrik- und andere Bauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0179: OdooAccountTemplate = OdooAccountTemplate {
    code: "0179",
    name: "Andere Bauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0180: OdooAccountTemplate = OdooAccountTemplate {
    code: "0180",
    name: "Geschäfts-, Fabrik- und andere Bauten im Bau auf eigenen Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0189: OdooAccountTemplate = OdooAccountTemplate {
    code: "0189",
    name: "Anzahlungen auf Geschäfts-, Fabrik- und andere Bauten auf fremden Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0190: OdooAccountTemplate = OdooAccountTemplate {
    code: "0190",
    name: "Wohnbauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0191: OdooAccountTemplate = OdooAccountTemplate {
    code: "0191",
    name: "Garagen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0192: OdooAccountTemplate = OdooAccountTemplate {
    code: "0192",
    name: "Außenanlagen für Geschäfts-, Fabrik- und andere Bauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0193: OdooAccountTemplate = OdooAccountTemplate {
    code: "0193",
    name: "Hof- und Wegebefestigungen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0194: OdooAccountTemplate = OdooAccountTemplate {
    code: "0194",
    name: "Einrichtungen für Wohnbauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0195: OdooAccountTemplate = OdooAccountTemplate {
    code: "0195",
    name: "Wohnbauten im Bau auf fremden Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0199: OdooAccountTemplate = OdooAccountTemplate {
    code: "0199",
    name: "Anzahlungen auf Wohnbauten auf fremden Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0200: OdooAccountTemplate = OdooAccountTemplate {
    code: "0200",
    name: "Technische Anlagen und Maschinen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0210: OdooAccountTemplate = OdooAccountTemplate {
    code: "0210",
    name: "Maschinen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0220: OdooAccountTemplate = OdooAccountTemplate {
    code: "0220",
    name: "Maschinengebundene Werkzeuge",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0240: OdooAccountTemplate = OdooAccountTemplate {
    code: "0240",
    name: "Technische Anlagen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0260: OdooAccountTemplate = OdooAccountTemplate {
    code: "0260",
    name: "Transportanlagen und Ähnliches",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0280: OdooAccountTemplate = OdooAccountTemplate {
    code: "0280",
    name: "Betriebsvorrichtungen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0290: OdooAccountTemplate = OdooAccountTemplate {
    code: "0290",
    name: "Technische Anlagen und Maschinen im Bau",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0299: OdooAccountTemplate = OdooAccountTemplate {
    code: "0299",
    name: "Anzahlungen auf technische Anlagen und Maschinen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0300: OdooAccountTemplate = OdooAccountTemplate {
    code: "0300",
    name: "Andere Anlagen, Betriebs- und Geschäftsausstattung",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0310: OdooAccountTemplate = OdooAccountTemplate {
    code: "0310",
    name: "Andere Anlagen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0320: OdooAccountTemplate = OdooAccountTemplate {
    code: "0320",
    name: "Pkw",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0350: OdooAccountTemplate = OdooAccountTemplate {
    code: "0350",
    name: "Lkw",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0380: OdooAccountTemplate = OdooAccountTemplate {
    code: "0380",
    name: "Sonstige Transportmittel",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0400: OdooAccountTemplate = OdooAccountTemplate {
    code: "0400",
    name: "Betriebsausstattung",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0410: OdooAccountTemplate = OdooAccountTemplate {
    code: "0410",
    name: "Geschäftsausstattung",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0420: OdooAccountTemplate = OdooAccountTemplate {
    code: "0420",
    name: "Büroeinrichtung",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0430: OdooAccountTemplate = OdooAccountTemplate {
    code: "0430",
    name: "Ladeneinrichtung",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0440: OdooAccountTemplate = OdooAccountTemplate {
    code: "0440",
    name: "Werkzeuge",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0450: OdooAccountTemplate = OdooAccountTemplate {
    code: "0450",
    name: "Einbauten in fremde Grundstücke",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0460: OdooAccountTemplate = OdooAccountTemplate {
    code: "0460",
    name: "Gerüst- und Schalungsmaterial",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0480: OdooAccountTemplate = OdooAccountTemplate {
    code: "0480",
    name: "Geringwertige Wirtschaftsgüter",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0485: OdooAccountTemplate = OdooAccountTemplate {
    code: "0485",
    name: "Wirtschaftsgüter (Sammelposten)",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0490: OdooAccountTemplate = OdooAccountTemplate {
    code: "0490",
    name: "Sonstige Betriebs- und Geschäftsausstattung",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0498: OdooAccountTemplate = OdooAccountTemplate {
    code: "0498",
    name: "Andere Anlagen, Betriebs- und Geschäftsausstattung im Bau",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0499: OdooAccountTemplate = OdooAccountTemplate {
    code: "0499",
    name: "Anzahlungen auf andere Anlagen, Betriebs- und Geschäftsausstattung",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0500: OdooAccountTemplate = OdooAccountTemplate {
    code: "0500",
    name: "Anteile an verbundenen Unternehmen (Anlagevermögen)",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0501: OdooAccountTemplate = OdooAccountTemplate {
    code: "0501",
    name: "Anteile an verbundenen Unternehmen, Personengesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0502: OdooAccountTemplate = OdooAccountTemplate {
    code: "0502",
    name: "Anteile an verbundenen Unternehmen, Kapitalgesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0503: OdooAccountTemplate = OdooAccountTemplate {
    code: "0503",
    name: "Anteile an herrschender oder mehrheitlich beteiligter Gesellschaft, Kapitalgesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0504: OdooAccountTemplate = OdooAccountTemplate {
    code: "0504",
    name: "Anteile an herrschender oder mehrheitlich beteiligter Gesellschaft",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0505: OdooAccountTemplate = OdooAccountTemplate {
    code: "0505",
    name: "Ausleihungen an verbundene Unternehmen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0506: OdooAccountTemplate = OdooAccountTemplate {
    code: "0506",
    name: "Ausleihungen an verbundene Unternehmen, Personengesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0507: OdooAccountTemplate = OdooAccountTemplate {
    code: "0507",
    name: "Ausleihungen an verbundene Unternehmen, Kapitalgesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0508: OdooAccountTemplate = OdooAccountTemplate {
    code: "0508",
    name: "Ausleihungen an verbundene Unternehmen, Einzelunternehmen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0509: OdooAccountTemplate = OdooAccountTemplate {
    code: "0509",
    name: "Anteile an herrschender oder mehrheitlich beteiligter Gesellschaft, Personengesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0510: OdooAccountTemplate = OdooAccountTemplate {
    code: "0510",
    name: "Beteiligungen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0513: OdooAccountTemplate = OdooAccountTemplate {
    code: "0513",
    name: "Typisch stille Beteiligungen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0516: OdooAccountTemplate = OdooAccountTemplate {
    code: "0516",
    name: "Atypisch stille Beteiligungen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0517: OdooAccountTemplate = OdooAccountTemplate {
    code: "0517",
    name: "Beteiligungen an Kapitalgesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0518: OdooAccountTemplate = OdooAccountTemplate {
    code: "0518",
    name: "Beteiligungen an Personengesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0519: OdooAccountTemplate = OdooAccountTemplate {
    code: "0519",
    name: "Beteiligung einer GmbH & Co. KG an einer Komplementär GmbH",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0520: OdooAccountTemplate = OdooAccountTemplate {
    code: "0520",
    name: "Ausleihungen an Unternehmen, mit denen ein Beteiligungsverhältnis besteht",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0523: OdooAccountTemplate = OdooAccountTemplate {
    code: "0523",
    name: "Ausleihungen an Unternehmen, mit denen ein Beteiligungsverhältnis besteht, Personengesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0524: OdooAccountTemplate = OdooAccountTemplate {
    code: "0524",
    name: "Ausleihungen an Unternehmen, mit denen ein Beteiligungsverhältnis besteht, Kapitalgesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0525: OdooAccountTemplate = OdooAccountTemplate {
    code: "0525",
    name: "Wertpapiere des Anlagevermögens",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0530: OdooAccountTemplate = OdooAccountTemplate {
    code: "0530",
    name: "Wertpapiere mit Gewinnbeteiligungsansprüchen, die dem Teileinkünfteverfahren unterliegen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0535: OdooAccountTemplate = OdooAccountTemplate {
    code: "0535",
    name: "Festverzinsliche Wertpapiere",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0540: OdooAccountTemplate = OdooAccountTemplate {
    code: "0540",
    name: "Übrige sonstige Ausleihungen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0550: OdooAccountTemplate = OdooAccountTemplate {
    code: "0550",
    name: "Darlehen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0570: OdooAccountTemplate = OdooAccountTemplate {
    code: "0570",
    name: "Genossenschaftsanteile zum langfristigen Verbleib",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0580: OdooAccountTemplate = OdooAccountTemplate {
    code: "0580",
    name: "Ausleihungen an Gesellschafter",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0582: OdooAccountTemplate = OdooAccountTemplate {
    code: "0582",
    name: "Ausleihungen an GmbH-Gesellschafter",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0584: OdooAccountTemplate = OdooAccountTemplate {
    code: "0584",
    name: "Ausleihungen an persönlich haftende Gesellschafter",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0586: OdooAccountTemplate = OdooAccountTemplate {
    code: "0586",
    name: "Ausleihungen an Kommanditisten",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0590: OdooAccountTemplate = OdooAccountTemplate {
    code: "0590",
    name: "Ausleihungen an nahe stehende Personen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0595: OdooAccountTemplate = OdooAccountTemplate {
    code: "0595",
    name: "Rückdeckungsansprüche aus Lebensversicherungen zum langfristigen Verbleib",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0600: OdooAccountTemplate = OdooAccountTemplate {
    code: "0600",
    name: "Anleihen nicht konvertibel",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0601: OdooAccountTemplate = OdooAccountTemplate {
    code: "0601",
    name: "Anleihen nicht konvertibel - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0605: OdooAccountTemplate = OdooAccountTemplate {
    code: "0605",
    name: "Anleihen nicht konvertibel - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0610: OdooAccountTemplate = OdooAccountTemplate {
    code: "0610",
    name: "Anleihen nicht konvertibel - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0615: OdooAccountTemplate = OdooAccountTemplate {
    code: "0615",
    name: "Anleihen konvertibel",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0616: OdooAccountTemplate = OdooAccountTemplate {
    code: "0616",
    name: "Anleihen konvertibel - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0620: OdooAccountTemplate = OdooAccountTemplate {
    code: "0620",
    name: "Anleihen konvertibel - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0625: OdooAccountTemplate = OdooAccountTemplate {
    code: "0625",
    name: "Anleihen konvertibel - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0630: OdooAccountTemplate = OdooAccountTemplate {
    code: "0630",
    name: "Verbindlichkeiten gegenüber Kreditinstituten",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0631: OdooAccountTemplate = OdooAccountTemplate {
    code: "0631",
    name: "Verbindlichkeiten gegenüber Kreditinstituten - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0640: OdooAccountTemplate = OdooAccountTemplate {
    code: "0640",
    name: "Verbindlichkeiten gegenüber Kreditinstituten - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0650: OdooAccountTemplate = OdooAccountTemplate {
    code: "0650",
    name: "Verbindlichkeiten gegenüber Kreditinstituten - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0660: OdooAccountTemplate = OdooAccountTemplate {
    code: "0660",
    name: "Verbindlichkeiten gegenüber Kreditinstituten aus Teilzahlungsverträgen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0661: OdooAccountTemplate = OdooAccountTemplate {
    code: "0661",
    name: "Verbindlichkeiten gegenüber Kreditinstituten aus Teilzahlungsverträgen - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0670: OdooAccountTemplate = OdooAccountTemplate {
    code: "0670",
    name: "Verbindlichkeiten gegenüber Kreditinstituten aus Teilzahlungsverträgen - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0680: OdooAccountTemplate = OdooAccountTemplate {
    code: "0680",
    name: "Verbindlichkeiten gegenüber Kreditinstituten aus Teilzahlungsverträgen - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0699: OdooAccountTemplate = OdooAccountTemplate {
    code: "0699",
    name: "Gegenkonto 0630-0689 bei Aufteilung der Konten 0690-0698",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0700: OdooAccountTemplate = OdooAccountTemplate {
    code: "0700",
    name: "Verbindlichkeiten gegenüber verbundenen Unternehmen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0701: OdooAccountTemplate = OdooAccountTemplate {
    code: "0701",
    name: "Verbindlichkeiten gegenüber verbundenen Unternehmen - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0705: OdooAccountTemplate = OdooAccountTemplate {
    code: "0705",
    name: "Verbindlichkeiten gegenüber verbundenen Unternehmen - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0710: OdooAccountTemplate = OdooAccountTemplate {
    code: "0710",
    name: "Verbindlichkeiten gegenüber verbundenen Unternehmen - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0715: OdooAccountTemplate = OdooAccountTemplate {
    code: "0715",
    name: "Verbindlichkeiten gegenüber Unternehmen, mit denen ein Beteiligungsverhältnis besteht",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0716: OdooAccountTemplate = OdooAccountTemplate {
    code: "0716",
    name: "Verbindlichkeiten gegenüber Unternehmen, mit denen ein Beteiligungsverhältnis besteht -  Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0720: OdooAccountTemplate = OdooAccountTemplate {
    code: "0720",
    name: "Verbindlichkeiten gegenüber Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0725: OdooAccountTemplate = OdooAccountTemplate {
    code: "0725",
    name: "Verbindlichkeiten gegenüber Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0730: OdooAccountTemplate = OdooAccountTemplate {
    code: "0730",
    name: "Verbindlichkeiten gegenüber Gesellschaftern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0731: OdooAccountTemplate = OdooAccountTemplate {
    code: "0731",
    name: "Verbindlichkeiten gegenüber Gesellschaftern - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0740: OdooAccountTemplate = OdooAccountTemplate {
    code: "0740",
    name: "Verbindlichkeiten gegenüber Gesellschaftern - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0750: OdooAccountTemplate = OdooAccountTemplate {
    code: "0750",
    name: "Verbindlichkeiten gegenüber Gesellschaftern - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0755: OdooAccountTemplate = OdooAccountTemplate {
    code: "0755",
    name: "Verbindlichkeiten gegenüber Gesellschaftern für offene Ausschüttungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0760: OdooAccountTemplate = OdooAccountTemplate {
    code: "0760",
    name: "Verbindlichkeiten gegenüber typisch stillen Gesellschaftern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0761: OdooAccountTemplate = OdooAccountTemplate {
    code: "0761",
    name: "Verbindlichkeiten gegenüber typisch stillen Gesellschaftern - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0764: OdooAccountTemplate = OdooAccountTemplate {
    code: "0764",
    name: "Verbindlichkeiten gegenüber typisch stillen Gesellschaftern - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0767: OdooAccountTemplate = OdooAccountTemplate {
    code: "0767",
    name: "Verbindlichkeiten gegenüber typisch stillen Gesellschaftern - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0770: OdooAccountTemplate = OdooAccountTemplate {
    code: "0770",
    name: "Verbindlichkeiten gegenüber atypisch stillen Gesellschaftern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0771: OdooAccountTemplate = OdooAccountTemplate {
    code: "0771",
    name: "Verbindlichkeiten gegenüber atypisch stillen Gesellschaftern - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0774: OdooAccountTemplate = OdooAccountTemplate {
    code: "0774",
    name: "Verbindlichkeiten gegenüber atypisch stillen Gesellschaftern - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0777: OdooAccountTemplate = OdooAccountTemplate {
    code: "0777",
    name: "Verbindlichkeiten gegenüber atypisch stillen Gesellschaftern - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0780: OdooAccountTemplate = OdooAccountTemplate {
    code: "0780",
    name: "Partiarische Darlehen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0781: OdooAccountTemplate = OdooAccountTemplate {
    code: "0781",
    name: "Partiarische Darlehen - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0784: OdooAccountTemplate = OdooAccountTemplate {
    code: "0784",
    name: "Partiarische Darlehen - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0787: OdooAccountTemplate = OdooAccountTemplate {
    code: "0787",
    name: "Partiarische Darlehen - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0799: OdooAccountTemplate = OdooAccountTemplate {
    code: "0799",
    name: "Gegenkonto 0730-0789 und 1665-1678 und 1695-1698 bei Aufteilung der Konten 0790-0798",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0809: OdooAccountTemplate = OdooAccountTemplate {
    code: "0809",
    name: "Kapitalerhöhung aus Gesellschaftsmitteln",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0810: OdooAccountTemplate = OdooAccountTemplate {
    code: "0810",
    name: "Geschäftsguthaben der verbleibenden Mitglieder",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0811: OdooAccountTemplate = OdooAccountTemplate {
    code: "0811",
    name: "Geschäftsguthaben der ausscheidenden Mitglieder",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0812: OdooAccountTemplate = OdooAccountTemplate {
    code: "0812",
    name: "Geschäftsguthaben aus gekündigten Geschäftsanteilen",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0813: OdooAccountTemplate = OdooAccountTemplate {
    code: "0813",
    name: "Rückständige fällige Einzahlungen auf Geschäftsanteile, vermerkt",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0815: OdooAccountTemplate = OdooAccountTemplate {
    code: "0815",
    name: "Gegenkonto Rückständige fällige Einzahlungen auf Geschäftsanteile, vermerkt",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0819: OdooAccountTemplate = OdooAccountTemplate {
    code: "0819",
    name: "Erworbene eigene Anteile",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0839: OdooAccountTemplate = OdooAccountTemplate {
    code: "0839",
    name: "Nachschüsse (Forderungen, Gegenkonto 0845)",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0845: OdooAccountTemplate = OdooAccountTemplate {
    code: "0845",
    name: "Nachschusskapital (Gegenkonto 0839)",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_II"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0848: OdooAccountTemplate = OdooAccountTemplate {
    code: "0848",
    name: "Andere Gewinnrücklagen aus dem Erwerb eigener Anteile",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0849: OdooAccountTemplate = OdooAccountTemplate {
    code: "0849",
    name: "Rücklage für Anteile an einem herrschenden oder mehrheitlich beteiligten Unternehmen",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0852: OdooAccountTemplate = OdooAccountTemplate {
    code: "0852",
    name: "Andere Ergebnisrücklagen",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0853: OdooAccountTemplate = OdooAccountTemplate {
    code: "0853",
    name: "Gewinnrücklagen aus den Übergangsvorschriften BilMoG",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0854: OdooAccountTemplate = OdooAccountTemplate {
    code: "0854",
    name: "Gewinnrücklagen aus den Übergangsvorschriften BilMoG (Zuschreibung Sachanlagevermögen)",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0857: OdooAccountTemplate = OdooAccountTemplate {
    code: "0857",
    name: "Gewinnrücklagen aus den Übergangsvorschriften BilMoG (Zuschreibung Finanzanlagevermögen)",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0858: OdooAccountTemplate = OdooAccountTemplate {
    code: "0858",
    name: "Gewinnrücklagen aus den Übergangsvorschriften BilMoG (Auflösung der Sonderposten mit Rücklageanteil)",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0859: OdooAccountTemplate = OdooAccountTemplate {
    code: "0859",
    name: "Latente Steuern (Gewinnrücklage Haben) aus erfolgsneutralen Verrechnungen",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0865: OdooAccountTemplate = OdooAccountTemplate {
    code: "0865",
    name: "Gewinnvortrag vor Verwendung (mit Aufteilung für Kapitalkontenentwicklung)",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0867: OdooAccountTemplate = OdooAccountTemplate {
    code: "0867",
    name: "Verlustvortrag vor Verwendung (mit Aufteilung für Kapitalkontenentwicklung)",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0870: OdooAccountTemplate = OdooAccountTemplate {
    code: "0870",
    name: "Festkapital",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0880: OdooAccountTemplate = OdooAccountTemplate {
    code: "0880",
    name: "Variables Kapital",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0890: OdooAccountTemplate = OdooAccountTemplate {
    code: "0890",
    name: "Gesellschafter-Darlehen",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0900: OdooAccountTemplate = OdooAccountTemplate {
    code: "0900",
    name: "Kommandit-Kapital",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0910: OdooAccountTemplate = OdooAccountTemplate {
    code: "0910",
    name: "Verlustausgleichskonto",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0920: OdooAccountTemplate = OdooAccountTemplate {
    code: "0920",
    name: "Gesellschafter-Darlehen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0950: OdooAccountTemplate = OdooAccountTemplate {
    code: "0950",
    name: "Rückstellungen für Pensionen und ähnliche Verpflichtungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0951: OdooAccountTemplate = OdooAccountTemplate {
    code: "0951",
    name: "Rückstellungen für Pensionen und ähnliche Verpflichtungen zur Saldierung mit Vermögensgegenständen zum langfristigen Verbleib nach § 246 Abs. 2 HGB",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0952: OdooAccountTemplate = OdooAccountTemplate {
    code: "0952",
    name: "Rückstellungen für Pensionen und ähnliche Verpflichtungen gegenüber Gesellschaftern oder nahe stehenden Personen (10 % Beteiligung am Kapital)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0953: OdooAccountTemplate = OdooAccountTemplate {
    code: "0953",
    name: "Rückstellungen für Direktzusagen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0954: OdooAccountTemplate = OdooAccountTemplate {
    code: "0954",
    name: "Rückstellungen für Zuschussverpflichtungen für Pensionskassen und Lebensversicherungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0955: OdooAccountTemplate = OdooAccountTemplate {
    code: "0955",
    name: "Steuerrückstellungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0956: OdooAccountTemplate = OdooAccountTemplate {
    code: "0956",
    name: "Gewerbesteuerrückstellung nach § 4 Abs. 5b EStG",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0957: OdooAccountTemplate = OdooAccountTemplate {
    code: "0957",
    name: "Gewerbesteuerrückstellung",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0961: OdooAccountTemplate = OdooAccountTemplate {
    code: "0961",
    name: "Urlaubsrückstellungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0962: OdooAccountTemplate = OdooAccountTemplate {
    code: "0962",
    name: "Steuerrückstellung aus Steuerstundung (BStBK)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0963: OdooAccountTemplate = OdooAccountTemplate {
    code: "0963",
    name: "Körperschaftsteuerrückstellung",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0964: OdooAccountTemplate = OdooAccountTemplate {
    code: "0964",
    name: "Rückstellungen für mit der Altersversorgung vergleichbare langfristige Verpflichtungen zum langfristigen Verbleib",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0965: OdooAccountTemplate = OdooAccountTemplate {
    code: "0965",
    name: "Rückstellungen für Personalkosten",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0966: OdooAccountTemplate = OdooAccountTemplate {
    code: "0966",
    name: "Rückstellungen zur Erfüllung der Aufbewahrungspflichten",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0967: OdooAccountTemplate = OdooAccountTemplate {
    code: "0967",
    name: "Rückstellungen für mit der Altersversorgung vergleichbare langfristige Verpflichtungen zur Saldierung mit Vermögensgegenständen zum langfristigen Verbleib nach § 246 Abs. 2 HGB",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0968: OdooAccountTemplate = OdooAccountTemplate {
    code: "0968",
    name: "Passive latente Steuern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_E"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0969: OdooAccountTemplate = OdooAccountTemplate {
    code: "0969",
    name: "Rückstellung für latente Steuern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0970: OdooAccountTemplate = OdooAccountTemplate {
    code: "0970",
    name: "Sonstige Rückstellungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0971: OdooAccountTemplate = OdooAccountTemplate {
    code: "0971",
    name: "Rückstellungen für unterlassene Aufwendungen für Instandhaltung, Nachholung in den ersten drei Monaten",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0973: OdooAccountTemplate = OdooAccountTemplate {
    code: "0973",
    name: "Rückstellungen für Abraum- und Abfallbeseitigung",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0974: OdooAccountTemplate = OdooAccountTemplate {
    code: "0974",
    name: "Rückstellungen für Gewährleistungen (Gegenkonto 4790)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0976: OdooAccountTemplate = OdooAccountTemplate {
    code: "0976",
    name: "Rückstellungen für drohende Verluste aus schwebenden Geschäften",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0977: OdooAccountTemplate = OdooAccountTemplate {
    code: "0977",
    name: "Rückstellungen für Abschluss- und Prüfungskosten",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0978: OdooAccountTemplate = OdooAccountTemplate {
    code: "0978",
    name: "Aufwandsrückstellungen nach § 249 Abs. 2 HGB a. F",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0979: OdooAccountTemplate = OdooAccountTemplate {
    code: "0979",
    name: "Rückstellungen für Umweltschutz",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0980: OdooAccountTemplate = OdooAccountTemplate {
    code: "0980",
    name: "Aktive Rechnungsabgrenzung",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_C"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0983: OdooAccountTemplate = OdooAccountTemplate {
    code: "0983",
    name: "Aktive latente Steuern",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_D"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0984: OdooAccountTemplate = OdooAccountTemplate {
    code: "0984",
    name: "Als Aufwand berücksichtigte Zölle und Verbrauchsteuern auf Vorräte",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_C"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0985: OdooAccountTemplate = OdooAccountTemplate {
    code: "0985",
    name: "Als Aufwand berücksichtigte Umsatzsteuer auf Anzahlungen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_C"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0986: OdooAccountTemplate = OdooAccountTemplate {
    code: "0986",
    name: "Damnum/Disagio",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_C"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0987: OdooAccountTemplate = OdooAccountTemplate {
    code: "0987",
    name: "Rechnungsabgrenzungsposten (Gewinnrücklage Soll) aus erfolgsneutralen Verrechnungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0988: OdooAccountTemplate = OdooAccountTemplate {
    code: "0988",
    name: "Latente Steuern (Gewinnrücklage Soll) aus erfolgsneutralen Verrechnungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0989: OdooAccountTemplate = OdooAccountTemplate {
    code: "0989",
    name: "Gesamthänderisch gebundene Rücklagen (mit Aufteilung für Kapitalkontenentwicklung)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0990: OdooAccountTemplate = OdooAccountTemplate {
    code: "0990",
    name: "Passive Rechnungsabgrenzung",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_D"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0996: OdooAccountTemplate = OdooAccountTemplate {
    code: "0996",
    name: "Pauschalwertberichtigung auf Forderungen - Restlaufzeit bis zu 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0997: OdooAccountTemplate = OdooAccountTemplate {
    code: "0997",
    name: "Pauschalwertberichtigung auf Forderungen - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0998: OdooAccountTemplate = OdooAccountTemplate {
    code: "0998",
    name: "Einzelwertberichtigungen auf Forderungen - Restlaufzeit bis zu 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_0999: OdooAccountTemplate = OdooAccountTemplate {
    code: "0999",
    name: "Einzelwertberichtigungen auf Forderungen - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1010: OdooAccountTemplate = OdooAccountTemplate {
    code: "1010",
    name: "Nebenkasse 1",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1020: OdooAccountTemplate = OdooAccountTemplate {
    code: "1020",
    name: "Nebenkasse 2",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1100: OdooAccountTemplate = OdooAccountTemplate {
    code: "1100",
    name: "Bank (Postbank)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1110: OdooAccountTemplate = OdooAccountTemplate {
    code: "1110",
    name: "Bank (Postbank 1)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1120: OdooAccountTemplate = OdooAccountTemplate {
    code: "1120",
    name: "Bank (Postbank 2)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1130: OdooAccountTemplate = OdooAccountTemplate {
    code: "1130",
    name: "Bank (Postbank 3)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1190: OdooAccountTemplate = OdooAccountTemplate {
    code: "1190",
    name: "LZB-Guthaben",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1195: OdooAccountTemplate = OdooAccountTemplate {
    code: "1195",
    name: "Bundesbankguthaben",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1210: OdooAccountTemplate = OdooAccountTemplate {
    code: "1210",
    name: "Bank 1",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1220: OdooAccountTemplate = OdooAccountTemplate {
    code: "1220",
    name: "Bank 2",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1230: OdooAccountTemplate = OdooAccountTemplate {
    code: "1230",
    name: "Bank 3",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1240: OdooAccountTemplate = OdooAccountTemplate {
    code: "1240",
    name: "Bank 4",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1250: OdooAccountTemplate = OdooAccountTemplate {
    code: "1250",
    name: "Bank 5",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1290: OdooAccountTemplate = OdooAccountTemplate {
    code: "1290",
    name: "Finanzmittelanlagen im Rahmen der kurzfristigen Finanzdisposition (nicht im Finanzmittelfonds enthalten)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1295: OdooAccountTemplate = OdooAccountTemplate {
    code: "1295",
    name: "Verbindlichkeiten gegenüber Kreditinstituten (nicht im Finanzmittelfonds enthalten)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1300: OdooAccountTemplate = OdooAccountTemplate {
    code: "1300",
    name: "Wechsel aus Lieferungen und Leistungen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1301: OdooAccountTemplate = OdooAccountTemplate {
    code: "1301",
    name: "Wechsel aus Lieferungen und Leistungen - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1302: OdooAccountTemplate = OdooAccountTemplate {
    code: "1302",
    name: "Wechsel aus Lieferungen und Leistungen - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1305: OdooAccountTemplate = OdooAccountTemplate {
    code: "1305",
    name: "Wechsel aus Lieferungen und Leistungen, bundesbankfähig",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1310: OdooAccountTemplate = OdooAccountTemplate {
    code: "1310",
    name: "Besitzwechsel gegen verbundene Unternehmen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1311: OdooAccountTemplate = OdooAccountTemplate {
    code: "1311",
    name: "Besitzwechsel gegen verbundene Unternehmen - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1312: OdooAccountTemplate = OdooAccountTemplate {
    code: "1312",
    name: "Besitzwechsel gegen verbundene Unternehmen - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1315: OdooAccountTemplate = OdooAccountTemplate {
    code: "1315",
    name: "Besitzwechsel gegen verbundene Unternehmen, bundesbankfähig",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1320: OdooAccountTemplate = OdooAccountTemplate {
    code: "1320",
    name: "Besitzwechsel gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1321: OdooAccountTemplate = OdooAccountTemplate {
    code: "1321",
    name: "Besitzwechsel gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1322: OdooAccountTemplate = OdooAccountTemplate {
    code: "1322",
    name: "Bills receivable from other long-term investees and investors - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1325: OdooAccountTemplate = OdooAccountTemplate {
    code: "1325",
    name: "Besitzwechsel gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht, bundesbankfähig",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1327: OdooAccountTemplate = OdooAccountTemplate {
    code: "1327",
    name: "Finanzwechsel",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_III_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1329: OdooAccountTemplate = OdooAccountTemplate {
    code: "1329",
    name: "Andere Wertpapiere mit unwesentlichen Wertschwankungen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_III_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1330: OdooAccountTemplate = OdooAccountTemplate {
    code: "1330",
    name: "Schecks",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1340: OdooAccountTemplate = OdooAccountTemplate {
    code: "1340",
    name: "Anteile an verbundenen Unternehmen (Umlaufvermögen)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_III_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1344: OdooAccountTemplate = OdooAccountTemplate {
    code: "1344",
    name: "Anteile an herrschender oder mehrheitlich beteiligter Gesellschaft",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_III_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1348: OdooAccountTemplate = OdooAccountTemplate {
    code: "1348",
    name: "Sonstige Wertpapiere",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_III_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1349: OdooAccountTemplate = OdooAccountTemplate {
    code: "1349",
    name: "Wertpapieranlagen im Rahmen der kurzfristigen Finanzdisposition",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_III_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1350: OdooAccountTemplate = OdooAccountTemplate {
    code: "1350",
    name: "GmbH-Anteile zum kurzfristigen Verbleib",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1352: OdooAccountTemplate = OdooAccountTemplate {
    code: "1352",
    name: "Genossenschaftsanteile zum kurzfristigen Verbleib",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1353: OdooAccountTemplate = OdooAccountTemplate {
    code: "1353",
    name: "Vermögensgegenstände zur Erfüllung von mit der Altersversorgung vergleichbaren langfristigen Verpflichtungen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1354: OdooAccountTemplate = OdooAccountTemplate {
    code: "1354",
    name: "Vermögensgegenstände zur Saldierung mit der Altersversorgung vergleichbaren langfristigen Verpflichtungen nach § 246 Abs. 2 HGB",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_E"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1355: OdooAccountTemplate = OdooAccountTemplate {
    code: "1355",
    name: "Ansprüche aus Rückdeckungsversicherungen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1356: OdooAccountTemplate = OdooAccountTemplate {
    code: "1356",
    name: "Vermögensgegenstände zur Erfüllung von Pensionsrückstellungen und ähnlichen Verpflichtungen zum langfristigen Verbleib",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1357: OdooAccountTemplate = OdooAccountTemplate {
    code: "1357",
    name: "Vermögensgegenstände zur Saldierung mit Pensionsrückstellungen und ähnlichen Verpflichtungen zum langfristigen Verbleib nach § 246 Abs. 2 HGB",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_E"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1370: OdooAccountTemplate = OdooAccountTemplate {
    code: "1370",
    name: "Gewinnermittlung §4/3 erfolgswirksam",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1371: OdooAccountTemplate = OdooAccountTemplate {
    code: "1371",
    name: "Verrechnungskonto Gewinnermittlung § 4 Abs. 3 EStG, nicht ergebniswirksam",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1372: OdooAccountTemplate = OdooAccountTemplate {
    code: "1372",
    name: "Wirtschaftsgüter des Umlaufvermögens nach § 4 Abs. 3 Satz 4 EStG",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1373: OdooAccountTemplate = OdooAccountTemplate {
    code: "1373",
    name: "Forderungen gegen Kommanditisten und atypisch stille Gesellschafter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1374: OdooAccountTemplate = OdooAccountTemplate {
    code: "1374",
    name: "Forderungen gegen Kommanditisten und atypisch stille Gesellschafter - Restlaufzeit bis 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1375: OdooAccountTemplate = OdooAccountTemplate {
    code: "1375",
    name: "Forderungen gegen Kommanditisten und atypisch stille Gesellschafter - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1376: OdooAccountTemplate = OdooAccountTemplate {
    code: "1376",
    name: "Forderungen gegen typisch stille Gesellschafter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1377: OdooAccountTemplate = OdooAccountTemplate {
    code: "1377",
    name: "Forderungen gegen typisch stille Gesellschafter - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1378: OdooAccountTemplate = OdooAccountTemplate {
    code: "1378",
    name: "Forderungen gegen typisch stille Gesellschafter - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1380: OdooAccountTemplate = OdooAccountTemplate {
    code: "1380",
    name: "Überleitungskonto Kostenstelle",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1381: OdooAccountTemplate = OdooAccountTemplate {
    code: "1381",
    name: "Forderungen gegen GmbH-Gesellschafter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1382: OdooAccountTemplate = OdooAccountTemplate {
    code: "1382",
    name: "Forderungen gegen GmbH-Gesellschafter - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1383: OdooAccountTemplate = OdooAccountTemplate {
    code: "1383",
    name: "Forderungen gegen GmbH-Gesellschafter - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1385: OdooAccountTemplate = OdooAccountTemplate {
    code: "1385",
    name: "Forderungen gegen persönlich haftende Gesellschafter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1386: OdooAccountTemplate = OdooAccountTemplate {
    code: "1386",
    name: "Receivables from general partners - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1387: OdooAccountTemplate = OdooAccountTemplate {
    code: "1387",
    name: "Forderungen gegen persönlich haftende Gesellschafter - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1389: OdooAccountTemplate = OdooAccountTemplate {
    code: "1389",
    name: "Ansprüche aus betrieblicher Altersversorgung und Pensionsansprüche (Mitunternehmer)",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1390: OdooAccountTemplate = OdooAccountTemplate {
    code: "1390",
    name: "Verrechnungskonto Ist-Versteuerung",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1400: OdooAccountTemplate = OdooAccountTemplate {
    code: "1400",
    name: "Verrechnungskonto Ist-Versteuerung",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1401: OdooAccountTemplate = OdooAccountTemplate {
    code: "1401",
    name: "Forderungen aus Lieferungen und Leistungen",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1410: OdooAccountTemplate = OdooAccountTemplate {
    code: "1410",
    name: "Forderungen aus Lieferungen und Leistungen ohne Kontokorrent",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1411: OdooAccountTemplate = OdooAccountTemplate {
    code: "1411",
    name: "Forderungen aus Lieferungen und Leistungen, keine getrennte Forderungs-/Verbindlichkeitsbuchhaltung (PoS)",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1445: OdooAccountTemplate = OdooAccountTemplate {
    code: "1445",
    name: "Forderungen aus Lieferungen und Leistungen zum allgemeinen Umsatzsteuersatz oder eines Kleinunternehmers (EÜR)",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1446: OdooAccountTemplate = OdooAccountTemplate {
    code: "1446",
    name: "Forderungen aus Lieferungen und Leistungen zum ermäßigten Umsatzsteuersatz (EÜR)",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1447: OdooAccountTemplate = OdooAccountTemplate {
    code: "1447",
    name: "Forderungen aus steuerfreien oder nicht steuerbaren Lieferungen und Leistungen (EÜR)",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1448: OdooAccountTemplate = OdooAccountTemplate {
    code: "1448",
    name: "Forderungen aus Lieferungen und Leistungen nach Durchschnittssätzen nach § 24 UStG (EÜR)",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1449: OdooAccountTemplate = OdooAccountTemplate {
    code: "1449",
    name: "Gegenkonto 1445-1448 bei Aufteilung der Forderungen nach Steuersätzen (EÜR)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1450: OdooAccountTemplate = OdooAccountTemplate {
    code: "1450",
    name: "Forderungen nach § 11 Abs. 1 Satz 2 EStG für § 4 Abs. 3 EStG",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1451: OdooAccountTemplate = OdooAccountTemplate {
    code: "1451",
    name: "Forderungen aus Lieferungen und Leistungen ohne Kontokorrent - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1455: OdooAccountTemplate = OdooAccountTemplate {
    code: "1455",
    name: "Forderungen aus Lieferungen und Leistungen ohne Kontokorrent - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1460: OdooAccountTemplate = OdooAccountTemplate {
    code: "1460",
    name: "Zweifelhafte Forderungen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1461: OdooAccountTemplate = OdooAccountTemplate {
    code: "1461",
    name: "Zweifelhafte Forderungen - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1465: OdooAccountTemplate = OdooAccountTemplate {
    code: "1465",
    name: "Zweifelhafte Forderungen - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1470: OdooAccountTemplate = OdooAccountTemplate {
    code: "1470",
    name: "Forderungen aus Lieferungen und Leistungen gegen verbundene Unternehmen",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1471: OdooAccountTemplate = OdooAccountTemplate {
    code: "1471",
    name: "Forderungen aus Lieferungen und Leistungen gegen verbundene Unternehmen - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1475: OdooAccountTemplate = OdooAccountTemplate {
    code: "1475",
    name: "Forderungen aus Lieferungen und Leistungen gegen verbundene Unternehmen - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1478: OdooAccountTemplate = OdooAccountTemplate {
    code: "1478",
    name: "Wertberichtigungen auf Forderungen gegen verbundene Unternehmen - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1479: OdooAccountTemplate = OdooAccountTemplate {
    code: "1479",
    name: "Wertberichtigungen auf Forderungen gegen verbundene Unternehmen - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1480: OdooAccountTemplate = OdooAccountTemplate {
    code: "1480",
    name: "Forderungen aus Lieferungen und Leistungen gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1481: OdooAccountTemplate = OdooAccountTemplate {
    code: "1481",
    name: "Forderungen aus Lieferungen und Leistungen gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1485: OdooAccountTemplate = OdooAccountTemplate {
    code: "1485",
    name: "Forderungen aus Lieferungen und Leistungen gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1488: OdooAccountTemplate = OdooAccountTemplate {
    code: "1488",
    name: "Wertberichtigungen auf Forderungen gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit größer 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1489: OdooAccountTemplate = OdooAccountTemplate {
    code: "1489",
    name: "Wertberichtigungen auf Forderungen gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit bis 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1490: OdooAccountTemplate = OdooAccountTemplate {
    code: "1490",
    name: "Forderungen aus Lieferungen und Leistungen gegen Gesellschafter",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1491: OdooAccountTemplate = OdooAccountTemplate {
    code: "1491",
    name: "Forderungen aus Lieferungen und Leistungen gegen Gesellschafter - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1495: OdooAccountTemplate = OdooAccountTemplate {
    code: "1495",
    name: "Forderungen aus Lieferungen und Leistungen gegen Gesellschafter - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1498: OdooAccountTemplate = OdooAccountTemplate {
    code: "1498",
    name: "Gegenkonto zu sonstigen Vermögensgegenständen bei Buchungen über Debitorenkonto",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1499: OdooAccountTemplate = OdooAccountTemplate {
    code: "1499",
    name: "Gegenkonto 1451-1497 bei Aufteilung Debitorenkonto",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1500: OdooAccountTemplate = OdooAccountTemplate {
    code: "1500",
    name: "Sonstige Vermögensgegenstände",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1501: OdooAccountTemplate = OdooAccountTemplate {
    code: "1501",
    name: "Sonstige Vermögensgegenstände - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1502: OdooAccountTemplate = OdooAccountTemplate {
    code: "1502",
    name: "Sonstige Vermögensgegenstände - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1503: OdooAccountTemplate = OdooAccountTemplate {
    code: "1503",
    name: "Forderungen gegen Vorstandsmitglieder und Geschäftsführer - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1504: OdooAccountTemplate = OdooAccountTemplate {
    code: "1504",
    name: "Forderungen gegen Vorstandsmitglieder und Geschäftsführer - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1505: OdooAccountTemplate = OdooAccountTemplate {
    code: "1505",
    name: "Forderungen gegen Aufsichtsratsund Beiratsmitglieder - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1506: OdooAccountTemplate = OdooAccountTemplate {
    code: "1506",
    name: "Forderungen gegen Aufsichtsratsund Beiratsmitglieder - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1507: OdooAccountTemplate = OdooAccountTemplate {
    code: "1507",
    name: "Forderungen gegen sonstige Gesellschafter - Restlaufzeit bis 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1508: OdooAccountTemplate = OdooAccountTemplate {
    code: "1508",
    name: "Forderungen gegen sonstige Gesellschafter - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1510: OdooAccountTemplate = OdooAccountTemplate {
    code: "1510",
    name: "Geleistete Anzahlungen auf Vorräte",
    account_type: "asset_prepayments",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1511: OdooAccountTemplate = OdooAccountTemplate {
    code: "1511",
    name: "Geleistete Anzahlungen, 7 % Vorsteuer",
    account_type: "asset_prepayments",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1518: OdooAccountTemplate = OdooAccountTemplate {
    code: "1518",
    name: "Geleistete Anzahlungen, 19 % Vorsteuer",
    account_type: "asset_prepayments",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1519: OdooAccountTemplate = OdooAccountTemplate {
    code: "1519",
    name: "Forderungen gegen Arbeitsgemeinschaften",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1520: OdooAccountTemplate = OdooAccountTemplate {
    code: "1520",
    name: "Forderungen gegenüber Krankenkassen aus Aufwendungsausgleichsgesetz",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1521: OdooAccountTemplate = OdooAccountTemplate {
    code: "1521",
    name: "Agenturwarenabrechnung",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1522: OdooAccountTemplate = OdooAccountTemplate {
    code: "1522",
    name: "Genussrechte",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1524: OdooAccountTemplate = OdooAccountTemplate {
    code: "1524",
    name: "Einzahlungsansprüche zu Nebenleistungen oder Zuzahlungen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1525: OdooAccountTemplate = OdooAccountTemplate {
    code: "1525",
    name: "Kautionen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1526: OdooAccountTemplate = OdooAccountTemplate {
    code: "1526",
    name: "Kautionen - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1527: OdooAccountTemplate = OdooAccountTemplate {
    code: "1527",
    name: "Kautionen - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1528: OdooAccountTemplate = OdooAccountTemplate {
    code: "1528",
    name: "Nachträglich abziehbare Vorsteuer nach § 15a Abs. 2 UStG",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1529: OdooAccountTemplate = OdooAccountTemplate {
    code: "1529",
    name: "Zurückzuzahlende Vorsteuer nach § 15a Abs. 2 UStG",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1530: OdooAccountTemplate = OdooAccountTemplate {
    code: "1530",
    name: "Forderungen gegen Personal aus Lohn- und Gehaltsabrechnung",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1531: OdooAccountTemplate = OdooAccountTemplate {
    code: "1531",
    name: "Forderungen gegen Personal aus Lohn- und Gehaltsabrechnung - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1537: OdooAccountTemplate = OdooAccountTemplate {
    code: "1537",
    name: "Forderungen gegen Personal aus Lohn- und Gehaltsabrechnung - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1538: OdooAccountTemplate = OdooAccountTemplate {
    code: "1538",
    name: "Körperschaftssteuergutschrift §37 (b.1 J)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1539: OdooAccountTemplate = OdooAccountTemplate {
    code: "1539",
    name: "Körperschaftssteuergutschrift §37 (g.1 J)",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1540: OdooAccountTemplate = OdooAccountTemplate {
    code: "1540",
    name: "Forderungen aus Gewerbesteuerüberzahlungen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1542: OdooAccountTemplate = OdooAccountTemplate {
    code: "1542",
    name: "Steuererstattungsansprüche gegenüber anderen Ländern",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1543: OdooAccountTemplate = OdooAccountTemplate {
    code: "1543",
    name: "Forderungen an das Finanzamt aus abgeführtem Bauabzugsbetrag",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1544: OdooAccountTemplate = OdooAccountTemplate {
    code: "1544",
    name: "Forderung gegenüber Bundesagentur für Arbeit",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1545: OdooAccountTemplate = OdooAccountTemplate {
    code: "1545",
    name: "USt-Forderungen",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1547: OdooAccountTemplate = OdooAccountTemplate {
    code: "1547",
    name: "Forderungen aus entrichteten Verbrauchsteuern",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1548: OdooAccountTemplate = OdooAccountTemplate {
    code: "1548",
    name: "Vorsteuer in Folgeperiode/im Folgejahr abziehbar",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1549: OdooAccountTemplate = OdooAccountTemplate {
    code: "1549",
    name: "Körperschaftsteuerrückforderung",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1550: OdooAccountTemplate = OdooAccountTemplate {
    code: "1550",
    name: "Darlehen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1551: OdooAccountTemplate = OdooAccountTemplate {
    code: "1551",
    name: "Darlehen - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1555: OdooAccountTemplate = OdooAccountTemplate {
    code: "1555",
    name: "Darlehen - Restlaufzeit größer 1 Jahr",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1556: OdooAccountTemplate = OdooAccountTemplate {
    code: "1556",
    name: "Nachträglich abziehbare Vorsteuer nach § 15a Abs. 1 UStG, bewegliche Wirtschaftsgüter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1557: OdooAccountTemplate = OdooAccountTemplate {
    code: "1557",
    name: "Zurückzuzahlende Vorsteuer nach § 15a Abs. 1 UStG, bewegliche Wirtschaftsgüter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1558: OdooAccountTemplate = OdooAccountTemplate {
    code: "1558",
    name: "Nachträglich abziehbare Vorsteuer nach § 15a Abs. 1 UStG, unbewegliche Wirtschaftsgüter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1559: OdooAccountTemplate = OdooAccountTemplate {
    code: "1559",
    name: "Zurückzuzahlende Vorsteuer nach § 15a Abs. 1 UStG, unbewegliche Wirtschaftsgüter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1560: OdooAccountTemplate = OdooAccountTemplate {
    code: "1560",
    name: "Aufzuteilende Vorsteuer",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1561: OdooAccountTemplate = OdooAccountTemplate {
    code: "1561",
    name: "Aufzuteilende Vorsteuer 7 %",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1562: OdooAccountTemplate = OdooAccountTemplate {
    code: "1562",
    name: "Aufzuteilende Vorsteuer aus innergemeinschaftlichem Erwerb",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1563: OdooAccountTemplate = OdooAccountTemplate {
    code: "1563",
    name: "Aufzuteilende Vorsteuer aus innergemeinschaftlichem Erwerb 19 %",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1566: OdooAccountTemplate = OdooAccountTemplate {
    code: "1566",
    name: "Aufzuteilende Vorsteuer 19 %",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1567: OdooAccountTemplate = OdooAccountTemplate {
    code: "1567",
    name: "Aufzuteilende Vorsteuer nach §§ 13a und 13b UStG",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1569: OdooAccountTemplate = OdooAccountTemplate {
    code: "1569",
    name: "Aufzuteilende Vorsteuer nach §§ 13a und 13b UStG 19 %",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1570: OdooAccountTemplate = OdooAccountTemplate {
    code: "1570",
    name: "Abziehbare Vorsteuer",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1571: OdooAccountTemplate = OdooAccountTemplate {
    code: "1571",
    name: "Abziehbare Vorsteuer 7 %",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1572: OdooAccountTemplate = OdooAccountTemplate {
    code: "1572",
    name: "Abziehbare Vorsteuer aus innergemeinschaftlichem Erwerb",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1574: OdooAccountTemplate = OdooAccountTemplate {
    code: "1574",
    name: "Abziehbare Vorsteuer aus innergemeinschaftlichem Erwerb 19 %",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1576: OdooAccountTemplate = OdooAccountTemplate {
    code: "1576",
    name: "Abziehbare Vorsteuer 19 %",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1577: OdooAccountTemplate = OdooAccountTemplate {
    code: "1577",
    name: "Abziehbare Vorsteuer nach § 13b UStG 19 %",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1578: OdooAccountTemplate = OdooAccountTemplate {
    code: "1578",
    name: "Abziehbare Vorsteuer nach § 13b UStG",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1580: OdooAccountTemplate = OdooAccountTemplate {
    code: "1580",
    name: "Gegenkonto Vorsteuer § 4 Abs. 3 EStG",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1581: OdooAccountTemplate = OdooAccountTemplate {
    code: "1581",
    name: "Auflösung Vorsteuer aus Vorjahr § 4 Abs. 3 EStG",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1582: OdooAccountTemplate = OdooAccountTemplate {
    code: "1582",
    name: "Vorsteuer aus Investitionen § 4 Abs. 3 EStG",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1583: OdooAccountTemplate = OdooAccountTemplate {
    code: "1583",
    name: "Gegenkonto für Vorsteuer nach Durchschnittssätzen für § 4 Abs. 3 EStG",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1584: OdooAccountTemplate = OdooAccountTemplate {
    code: "1584",
    name: "Abziehbare Vorsteuer aus innergemeinschaftlichem Erwerb von Neufahrzeugen von Lieferanten ohne USt-Id-Nr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1585: OdooAccountTemplate = OdooAccountTemplate {
    code: "1585",
    name: "Abziehbare Vorsteuer aus der Auslagerung von Gegenständen aus einem Umsatzsteuerlager",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1587: OdooAccountTemplate = OdooAccountTemplate {
    code: "1587",
    name: "Vorsteuer nach allgemeinen Durchschnittssätzen UStVA Kz. 63",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1588: OdooAccountTemplate = OdooAccountTemplate {
    code: "1588",
    name: "Entstandene Einfuhrumsatzsteuer",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1590: OdooAccountTemplate = OdooAccountTemplate {
    code: "1590",
    name: "Durchlaufende Posten",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1592: OdooAccountTemplate = OdooAccountTemplate {
    code: "1592",
    name: "Fremdgeld",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1593: OdooAccountTemplate = OdooAccountTemplate {
    code: "1593",
    name: "Verrechnungskonto erhaltene Anzahlungen bei Buchung über Debitorenkonto",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1594: OdooAccountTemplate = OdooAccountTemplate {
    code: "1594",
    name: "Forderungen gegen verbundene Unternehmen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1595: OdooAccountTemplate = OdooAccountTemplate {
    code: "1595",
    name: "Forderungen gegen verbundene Unternehmen - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1596: OdooAccountTemplate = OdooAccountTemplate {
    code: "1596",
    name: "Forderungen gegen verbundene Unternehmen - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1597: OdooAccountTemplate = OdooAccountTemplate {
    code: "1597",
    name: "Forderungen gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1598: OdooAccountTemplate = OdooAccountTemplate {
    code: "1598",
    name: "Forderungen gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1599: OdooAccountTemplate = OdooAccountTemplate {
    code: "1599",
    name: "Forderungen gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1600: OdooAccountTemplate = OdooAccountTemplate {
    code: "1600",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1601: OdooAccountTemplate = OdooAccountTemplate {
    code: "1601",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1605: OdooAccountTemplate = OdooAccountTemplate {
    code: "1605",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen zum allgemeinen Umsatzsteuersatz (EÜR)",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1606: OdooAccountTemplate = OdooAccountTemplate {
    code: "1606",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen zum ermäßigten Umsatzsteuersatz (EÜR)",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1607: OdooAccountTemplate = OdooAccountTemplate {
    code: "1607",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen ohne Vorsteuerabzug (EÜR)",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1609: OdooAccountTemplate = OdooAccountTemplate {
    code: "1609",
    name: "Gegenkonto 1605-1607 bei Aufteilung der Verbindlichkeiten nach Steuersätzen (EÜR)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1610: OdooAccountTemplate = OdooAccountTemplate {
    code: "1610",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen ohne Kontokorrent",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1624: OdooAccountTemplate = OdooAccountTemplate {
    code: "1624",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen für Investitionen für § 4 Abs. 3 EStG",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1625: OdooAccountTemplate = OdooAccountTemplate {
    code: "1625",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen ohne Kontokorrent - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1626: OdooAccountTemplate = OdooAccountTemplate {
    code: "1626",
    name: "Trade payables, no separate receivables/payables accounting - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1628: OdooAccountTemplate = OdooAccountTemplate {
    code: "1628",
    name: "Trade payables, no separate receivables/payables accounting - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1630: OdooAccountTemplate = OdooAccountTemplate {
    code: "1630",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber verbundenen Unternehmen",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1631: OdooAccountTemplate = OdooAccountTemplate {
    code: "1631",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber verbundenen Unternehmen - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1635: OdooAccountTemplate = OdooAccountTemplate {
    code: "1635",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber verbundenen Unternehmen - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1638: OdooAccountTemplate = OdooAccountTemplate {
    code: "1638",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber verbundenen Unternehmen - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1640: OdooAccountTemplate = OdooAccountTemplate {
    code: "1640",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber Unternehmen, mit denen ein Beteiligungsverhältnis besteht",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1641: OdooAccountTemplate = OdooAccountTemplate {
    code: "1641",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1645: OdooAccountTemplate = OdooAccountTemplate {
    code: "1645",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1648: OdooAccountTemplate = OdooAccountTemplate {
    code: "1648",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1650: OdooAccountTemplate = OdooAccountTemplate {
    code: "1650",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber Gesellschaftern",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1651: OdooAccountTemplate = OdooAccountTemplate {
    code: "1651",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber Gesellschaftern - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1655: OdooAccountTemplate = OdooAccountTemplate {
    code: "1655",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber Gesellschaftern - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1658: OdooAccountTemplate = OdooAccountTemplate {
    code: "1658",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber Gesellschaftern - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1659: OdooAccountTemplate = OdooAccountTemplate {
    code: "1659",
    name: "Gegenkonto 1625-1658 bei Aufteilung Kreditorenkonto",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1660: OdooAccountTemplate = OdooAccountTemplate {
    code: "1660",
    name: "Wechselverbindlichkeiten",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1661: OdooAccountTemplate = OdooAccountTemplate {
    code: "1661",
    name: "Wechselverbindlichkeiten - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1662: OdooAccountTemplate = OdooAccountTemplate {
    code: "1662",
    name: "Wechselverbindlichkeiten - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1663: OdooAccountTemplate = OdooAccountTemplate {
    code: "1663",
    name: "Wechselverbindlichkeiten - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1665: OdooAccountTemplate = OdooAccountTemplate {
    code: "1665",
    name: "Verbindlichkeiten gegenüber GmbH-Gesellschaftern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1666: OdooAccountTemplate = OdooAccountTemplate {
    code: "1666",
    name: "Verbindlichkeiten gegenüber GmbH-Gesellschaftern - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1667: OdooAccountTemplate = OdooAccountTemplate {
    code: "1667",
    name: "Verbindlichkeiten gegenüber GmbH-Gesellschaftern - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1668: OdooAccountTemplate = OdooAccountTemplate {
    code: "1668",
    name: "Verbindlichkeiten gegenüber GmbH-Gesellschaftern - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1670: OdooAccountTemplate = OdooAccountTemplate {
    code: "1670",
    name: "Verbindlichkeiten gegenüber persönlich haftenden Gesellschaftern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1671: OdooAccountTemplate = OdooAccountTemplate {
    code: "1671",
    name: "Verbindlichkeiten gegenüber persönlich haftenden Gesellschaftern - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1672: OdooAccountTemplate = OdooAccountTemplate {
    code: "1672",
    name: "Verbindlichkeiten gegenüber persönlich haftenden Gesellschaftern - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1673: OdooAccountTemplate = OdooAccountTemplate {
    code: "1673",
    name: "Verbindlichkeiten gegenüber persönlich haftenden Gesellschaftern - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1675: OdooAccountTemplate = OdooAccountTemplate {
    code: "1675",
    name: "Verbindlichkeiten gegenüber Kommanditisten",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1676: OdooAccountTemplate = OdooAccountTemplate {
    code: "1676",
    name: "Verbindlichkeiten gegenüber Kommanditisten - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1677: OdooAccountTemplate = OdooAccountTemplate {
    code: "1677",
    name: "Verbindlichkeiten gegenüber Kommanditisten - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1678: OdooAccountTemplate = OdooAccountTemplate {
    code: "1678",
    name: "Verbindlichkeiten gegenüber Kommanditisten - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1691: OdooAccountTemplate = OdooAccountTemplate {
    code: "1691",
    name: "Verbindlichkeiten gegenüber Arbeitsgemeinschaften",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1695: OdooAccountTemplate = OdooAccountTemplate {
    code: "1695",
    name: "Verbindlichkeiten gegenüber stillen Gesellschaftern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1696: OdooAccountTemplate = OdooAccountTemplate {
    code: "1696",
    name: "Liabilities to silent partners - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1697: OdooAccountTemplate = OdooAccountTemplate {
    code: "1697",
    name: "Verbindlichkeiten gegenüber stillen Gesellschaftern - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1698: OdooAccountTemplate = OdooAccountTemplate {
    code: "1698",
    name: "Verbindlichkeiten gegenüber stillen Gesellschaftern - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1700: OdooAccountTemplate = OdooAccountTemplate {
    code: "1700",
    name: "Sonstige Verbindlichkeiten",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1701: OdooAccountTemplate = OdooAccountTemplate {
    code: "1701",
    name: "Sonstige Verbindlichkeiten - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1702: OdooAccountTemplate = OdooAccountTemplate {
    code: "1702",
    name: "Sonstige Verbindlichkeiten - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1703: OdooAccountTemplate = OdooAccountTemplate {
    code: "1703",
    name: "Sonstige Verbindlichkeiten - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1704: OdooAccountTemplate = OdooAccountTemplate {
    code: "1704",
    name: "Sonstige Verbindlichkeiten nach § 11 Abs. 2 Satz 2 EStG für § 4 Abs. 3 EStG",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1705: OdooAccountTemplate = OdooAccountTemplate {
    code: "1705",
    name: "Darlehen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1706: OdooAccountTemplate = OdooAccountTemplate {
    code: "1706",
    name: "Darlehen - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1707: OdooAccountTemplate = OdooAccountTemplate {
    code: "1707",
    name: "Darlehen - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1708: OdooAccountTemplate = OdooAccountTemplate {
    code: "1708",
    name: "Darlehen -  Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1709: OdooAccountTemplate = OdooAccountTemplate {
    code: "1709",
    name: "Gewinnverfügungskonto stille Gesellschafter",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1710: OdooAccountTemplate = OdooAccountTemplate {
    code: "1710",
    name: "Erhaltene Anzahlungen auf Bestellungen (Verbindlichkeiten)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1711: OdooAccountTemplate = OdooAccountTemplate {
    code: "1711",
    name: "Erhaltene, versteuerte Anzahlungen 7 % USt (Verbindlichkeiten)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1716: OdooAccountTemplate = OdooAccountTemplate {
    code: "1716",
    name: "Erhaltene Anzahlungen 15% MwSt.",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1718: OdooAccountTemplate = OdooAccountTemplate {
    code: "1718",
    name: "Erhaltene, versteuerte Anzahlungen 19 % USt (Verbindlichkeiten)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1719: OdooAccountTemplate = OdooAccountTemplate {
    code: "1719",
    name: "Erhaltene Anzahlungen - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1720: OdooAccountTemplate = OdooAccountTemplate {
    code: "1720",
    name: "Erhaltene Anzahlungen - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1721: OdooAccountTemplate = OdooAccountTemplate {
    code: "1721",
    name: "Erhaltene Anzahlungen - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1722: OdooAccountTemplate = OdooAccountTemplate {
    code: "1722",
    name: "Erhaltene Anzahlungen auf Bestellungen (von Vorräten offen abgesetzt)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1725: OdooAccountTemplate = OdooAccountTemplate {
    code: "1725",
    name: "Umsatzsteuer in Folgeperiode fällig (§§ 13 Abs. 1 Nr. 6 und 13b Abs. 2 UStG)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1728: OdooAccountTemplate = OdooAccountTemplate {
    code: "1728",
    name: "Umsatzsteuer aus im anderen EULand steuerpflichtigen elektronischen Dienstleistungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1729: OdooAccountTemplate = OdooAccountTemplate {
    code: "1729",
    name: "Steuerzahlungen aus im anderen EU-Land steuerpflichtigen Dienstleistungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1730: OdooAccountTemplate = OdooAccountTemplate {
    code: "1730",
    name: "Kreditkartenabrechnung",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1731: OdooAccountTemplate = OdooAccountTemplate {
    code: "1731",
    name: "Agenturwarenabrechnung",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1732: OdooAccountTemplate = OdooAccountTemplate {
    code: "1732",
    name: "Erhaltene Kautionen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1733: OdooAccountTemplate = OdooAccountTemplate {
    code: "1733",
    name: "Erhaltene Kautionen - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1734: OdooAccountTemplate = OdooAccountTemplate {
    code: "1734",
    name: "Erhaltene Kautionen - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1735: OdooAccountTemplate = OdooAccountTemplate {
    code: "1735",
    name: "Erhaltene Kautionen - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1736: OdooAccountTemplate = OdooAccountTemplate {
    code: "1736",
    name: "Verbindlichkeiten aus Steuern und Abgaben",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1737: OdooAccountTemplate = OdooAccountTemplate {
    code: "1737",
    name: "Verbindlichkeiten aus Steuern und Abgaben -  Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1738: OdooAccountTemplate = OdooAccountTemplate {
    code: "1738",
    name: "Verbindlichkeiten aus Steuern und Abgaben - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1739: OdooAccountTemplate = OdooAccountTemplate {
    code: "1739",
    name: "Verbindlichkeiten aus Steuern und Abgaben - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1740: OdooAccountTemplate = OdooAccountTemplate {
    code: "1740",
    name: "Verbindlichkeiten aus Lohn und Gehalt",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1741: OdooAccountTemplate = OdooAccountTemplate {
    code: "1741",
    name: "Verbindlichkeiten aus Lohn- und Kirchensteuer",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1742: OdooAccountTemplate = OdooAccountTemplate {
    code: "1742",
    name: "Verbindlichkeiten im Rahmen der sozialen Sicherheit",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1743: OdooAccountTemplate = OdooAccountTemplate {
    code: "1743",
    name: "Verbindlichkeiten im Rahmen der sozialen Sicherheit - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1744: OdooAccountTemplate = OdooAccountTemplate {
    code: "1744",
    name: "Verbindlichkeiten im Rahmen der sozialen Sicherheit - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1745: OdooAccountTemplate = OdooAccountTemplate {
    code: "1745",
    name: "Verbindlichkeiten im Rahmen der sozialen Sicherheit - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1746: OdooAccountTemplate = OdooAccountTemplate {
    code: "1746",
    name: "Verbindlichkeiten aus Einbehaltungen (KapESt und SolZ, KiSt auf KapESt) für offene Ausschüttungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1747: OdooAccountTemplate = OdooAccountTemplate {
    code: "1747",
    name: "Verbindlichkeiten für Verbrauchsteuern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1748: OdooAccountTemplate = OdooAccountTemplate {
    code: "1748",
    name: "Verbindlichkeiten für Einbehaltungen von Arbeitnehmern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1749: OdooAccountTemplate = OdooAccountTemplate {
    code: "1749",
    name: "Verbindlichkeiten an das Finanzamt aus abzuführendem Bauabzugsbetrag",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1750: OdooAccountTemplate = OdooAccountTemplate {
    code: "1750",
    name: "Verbindlichkeiten aus Vermögensbildung",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1751: OdooAccountTemplate = OdooAccountTemplate {
    code: "1751",
    name: "Verbindlichkeiten aus Vermögensbildung - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1752: OdooAccountTemplate = OdooAccountTemplate {
    code: "1752",
    name: "Verbindlichkeiten aus Vermögensbildung - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1753: OdooAccountTemplate = OdooAccountTemplate {
    code: "1753",
    name: "Verbindlichkeiten aus Vermögensbildung - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1754: OdooAccountTemplate = OdooAccountTemplate {
    code: "1754",
    name: "Steuerzahlungen an andere Länder",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1755: OdooAccountTemplate = OdooAccountTemplate {
    code: "1755",
    name: "Lohn- und Gehaltsverrechnung",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1756: OdooAccountTemplate = OdooAccountTemplate {
    code: "1756",
    name: "Lohn- und Gehaltsverrechnung nach § 11 Abs. 2 Satz 2 EStG für § 4 Abs. 3 EStG",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1758: OdooAccountTemplate = OdooAccountTemplate {
    code: "1758",
    name: "Sonstige Verbindlichkeiten aus genossenschaftlicher Rückvergütung",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1759: OdooAccountTemplate = OdooAccountTemplate {
    code: "1759",
    name: "Voraussichtliche Beitragsschuld gegenüber den Sozialversicherungsträgern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1760: OdooAccountTemplate = OdooAccountTemplate {
    code: "1760",
    name: "Umsatzsteuer nicht fällig",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1761: OdooAccountTemplate = OdooAccountTemplate {
    code: "1761",
    name: "Umsatzsteuer nicht fällig 7 %",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1762: OdooAccountTemplate = OdooAccountTemplate {
    code: "1762",
    name: "Umsatzsteuer nicht fällig aus im Inland steuerpflichtigen EU-Lieferungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1764: OdooAccountTemplate = OdooAccountTemplate {
    code: "1764",
    name: "Umsatzsteuer nicht fällig aus im Inland steuerpflichtigen EU-Lieferungen 19 %",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1766: OdooAccountTemplate = OdooAccountTemplate {
    code: "1766",
    name: "Umsatzsteuer nicht fällig 19 %",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1767: OdooAccountTemplate = OdooAccountTemplate {
    code: "1767",
    name: "Umsatzsteuer aus im anderen EULand steuerpflichtigen Lieferungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1768: OdooAccountTemplate = OdooAccountTemplate {
    code: "1768",
    name: "Umsatzsteuer aus im anderen EULand steuerpflichtigen sonstigen Leistungen/Werklieferungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1769: OdooAccountTemplate = OdooAccountTemplate {
    code: "1769",
    name: "Umsatzsteuer aus der Auslagerung von Gegenständen aus einem Umsatzsteuerlager",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1770: OdooAccountTemplate = OdooAccountTemplate {
    code: "1770",
    name: "Umsatzsteuer",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1771: OdooAccountTemplate = OdooAccountTemplate {
    code: "1771",
    name: "Umsatzsteuer 7 %",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1772: OdooAccountTemplate = OdooAccountTemplate {
    code: "1772",
    name: "Umsatzsteuer aus innergemeinschaftlichem Erwerb",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1774: OdooAccountTemplate = OdooAccountTemplate {
    code: "1774",
    name: "Umsatzsteuer aus innergemeinschaftlichem Erwerb 19 %",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1776: OdooAccountTemplate = OdooAccountTemplate {
    code: "1776",
    name: "Umsatzsteuer 19 %",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1777: OdooAccountTemplate = OdooAccountTemplate {
    code: "1777",
    name: "Umsatzsteuer aus im Inland steuerpflichtigen EU-Lieferungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1778: OdooAccountTemplate = OdooAccountTemplate {
    code: "1778",
    name: "Umsatzsteuer aus im Inland steuerpflichtigen EU-Lieferungen 19 %",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1779: OdooAccountTemplate = OdooAccountTemplate {
    code: "1779",
    name: "Umsatzsteuer aus innergemeinschaftlichem Erwerb ohne Vorsteuerabzug",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1780: OdooAccountTemplate = OdooAccountTemplate {
    code: "1780",
    name: "Umsatzsteuer-Vorauszahlungen",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1781: OdooAccountTemplate = OdooAccountTemplate {
    code: "1781",
    name: "Umsatzsteuer-Vorauszahlungen 1/11",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1782: OdooAccountTemplate = OdooAccountTemplate {
    code: "1782",
    name: "Nachsteuer, UStVA Kz. 65",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1783: OdooAccountTemplate = OdooAccountTemplate {
    code: "1783",
    name: "In Rechnung unrichtig oder unberechtigt ausgewiesene Steuerbeträge, UStVA Kz. 69",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1784: OdooAccountTemplate = OdooAccountTemplate {
    code: "1784",
    name: "Umsatzsteuer aus innergemeinschaftlichem Erwerb von Neufahrzeugen von Lieferanten ohne Umsatzsteuer-Identifikationsnummer",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1785: OdooAccountTemplate = OdooAccountTemplate {
    code: "1785",
    name: "Umsatzsteuer nach § 13b UStG",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1787: OdooAccountTemplate = OdooAccountTemplate {
    code: "1787",
    name: "Umsatzsteuer nach § 13b UStG 19 %",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1788: OdooAccountTemplate = OdooAccountTemplate {
    code: "1788",
    name: "Einfuhrumsatzsteuer aufgeschoben bis",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1789: OdooAccountTemplate = OdooAccountTemplate {
    code: "1789",
    name: "Umsatzsteuer laufendes Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1790: OdooAccountTemplate = OdooAccountTemplate {
    code: "1790",
    name: "Umsatzsteuer Vorjahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1791: OdooAccountTemplate = OdooAccountTemplate {
    code: "1791",
    name: "Umsatzsteuer frühere Jahre",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1792: OdooAccountTemplate = OdooAccountTemplate {
    code: "1792",
    name: "Sonstige Verrechnungskonten (Interimskonten)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1793: OdooAccountTemplate = OdooAccountTemplate {
    code: "1793",
    name: "Verrechnungskonto geleistete Anzahlungen bei Buchung über Kreditorenkonto",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1794: OdooAccountTemplate = OdooAccountTemplate {
    code: "1794",
    name: "Umsatzsteuer aus Erwerb als letzter Abnehmer innerhalb eines Dreiecksgeschäfts",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1795: OdooAccountTemplate = OdooAccountTemplate {
    code: "1795",
    name: "Verbindlichkeiten im Rahmen der sozialen Sicherheit für § 4 Abs. 3 EStG",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1796: OdooAccountTemplate = OdooAccountTemplate {
    code: "1796",
    name: "Ausgegebene Geschenkgutscheine",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1797: OdooAccountTemplate = OdooAccountTemplate {
    code: "1797",
    name: "Verbindlichkeiten aus Umsatzsteuer-Vorauszahlungen",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1800: OdooAccountTemplate = OdooAccountTemplate {
    code: "1800",
    name: "Privatentnahmen allgemein",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1810: OdooAccountTemplate = OdooAccountTemplate {
    code: "1810",
    name: "Privatsteuern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1820: OdooAccountTemplate = OdooAccountTemplate {
    code: "1820",
    name: "Sonderausgaben beschränkt abzugsfähig",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1830: OdooAccountTemplate = OdooAccountTemplate {
    code: "1830",
    name: "Sonderausgaben unbeschränkt abzugsfähig",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1840: OdooAccountTemplate = OdooAccountTemplate {
    code: "1840",
    name: "Zuwendungen, Spenden",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1850: OdooAccountTemplate = OdooAccountTemplate {
    code: "1850",
    name: "Außergewöhnliche Belastungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1860: OdooAccountTemplate = OdooAccountTemplate {
    code: "1860",
    name: "Grundstücksaufwand",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1869: OdooAccountTemplate = OdooAccountTemplate {
    code: "1869",
    name: "Grundstücksaufwand (Umsatzsteuerschlüssel möglich, nur Einzelunternehmen)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1870: OdooAccountTemplate = OdooAccountTemplate {
    code: "1870",
    name: "Grundstücksertrag",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1879: OdooAccountTemplate = OdooAccountTemplate {
    code: "1879",
    name: "Grundstücksertrag (Umsatzsteuerschlüssel möglich, nur Einzelunternehmen)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1880: OdooAccountTemplate = OdooAccountTemplate {
    code: "1880",
    name: "Unentgeltliche Wertabgaben",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1890: OdooAccountTemplate = OdooAccountTemplate {
    code: "1890",
    name: "Privateinlagen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1900: OdooAccountTemplate = OdooAccountTemplate {
    code: "1900",
    name: "Privatentnahmen allgemein (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1910: OdooAccountTemplate = OdooAccountTemplate {
    code: "1910",
    name: "Privatsteuern (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1920: OdooAccountTemplate = OdooAccountTemplate {
    code: "1920",
    name: "Privatsteuern (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1930: OdooAccountTemplate = OdooAccountTemplate {
    code: "1930",
    name: "Sonderausgaben unbeschränkt abzugsfähig (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1940: OdooAccountTemplate = OdooAccountTemplate {
    code: "1940",
    name: "Zuwendungen, Spenden (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1950: OdooAccountTemplate = OdooAccountTemplate {
    code: "1950",
    name: "Außergewöhnliche Belastungen (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1960: OdooAccountTemplate = OdooAccountTemplate {
    code: "1960",
    name: "Grundstücksaufwand (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1970: OdooAccountTemplate = OdooAccountTemplate {
    code: "1970",
    name: "Grundstücksertrag (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1980: OdooAccountTemplate = OdooAccountTemplate {
    code: "1980",
    name: "Unentgeltliche Wertabgaben (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_1990: OdooAccountTemplate = OdooAccountTemplate {
    code: "1990",
    name: "Privateinlagen (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_2000: OdooAccountTemplate = OdooAccountTemplate {
    code: "2000",
    name: "Außerordentliche Ausgaben",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2001: OdooAccountTemplate = OdooAccountTemplate {
    code: "2001",
    name: "Außerordentliche Aufwendungen, die den Jahresüberschuss beeinflussen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2004: OdooAccountTemplate = OdooAccountTemplate {
    code: "2004",
    name: "Verluste durch Verschmelzung und Umwandlung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2005: OdooAccountTemplate = OdooAccountTemplate {
    code: "2005",
    name: "Außerordentliche nicht zahlungswirksame Aufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2006: OdooAccountTemplate = OdooAccountTemplate {
    code: "2006",
    name: "Verluste durch außergewöhnliche Schadensfälle (nur Bilanzierer)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2007: OdooAccountTemplate = OdooAccountTemplate {
    code: "2007",
    name: "Aufwendungen für Restrukturierungs- und Sanierungsmaßnahmen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2008: OdooAccountTemplate = OdooAccountTemplate {
    code: "2008",
    name: "Verluste aus der Veräußerung oder der Aufgabe von Geschäftsaktivitäten nach Steuern",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2010: OdooAccountTemplate = OdooAccountTemplate {
    code: "2010",
    name: "Betriebsfremde Aufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2020: OdooAccountTemplate = OdooAccountTemplate {
    code: "2020",
    name: "Periodenfremde Aufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2090: OdooAccountTemplate = OdooAccountTemplate {
    code: "2090",
    name: "Aufwendungen aus der Anwendung von Übergangsvorschriften",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2091: OdooAccountTemplate = OdooAccountTemplate {
    code: "2091",
    name: "Aufwendungen aus der Anwendung von Übergangsvorschriften (Pensionsrückstellungen)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2092: OdooAccountTemplate = OdooAccountTemplate {
    code: "2092",
    name: "Außerordentlicher Aufwand aus der Anwendung von Übergangsbestimmungen (Bilanzierungshilfen)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2094: OdooAccountTemplate = OdooAccountTemplate {
    code: "2094",
    name: "Aufwendungen aus der Anwendung von Übergangsvorschriften (Latente Steuern)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2100: OdooAccountTemplate = OdooAccountTemplate {
    code: "2100",
    name: "Zinsen und ähnliche Aufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2102: OdooAccountTemplate = OdooAccountTemplate {
    code: "2102",
    name: "Steuerlich nicht abzugsfähige andere Nebenleistungen zu Steuern § 4 Abs. 5b EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2103: OdooAccountTemplate = OdooAccountTemplate {
    code: "2103",
    name: "Steuerlich abzugsfähige andere Nebenleistungen zu Steuern",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2104: OdooAccountTemplate = OdooAccountTemplate {
    code: "2104",
    name: "Steuerlich nicht abzugsfähige andere Nebenleistungen zu Steuern",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2105: OdooAccountTemplate = OdooAccountTemplate {
    code: "2105",
    name: "Zinsaufwendungen § 233a AO nicht abzugsfähig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2106: OdooAccountTemplate = OdooAccountTemplate {
    code: "2106",
    name: "Abzinsung des Steuererhöhungsbetrags § 38",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2107: OdooAccountTemplate = OdooAccountTemplate {
    code: "2107",
    name: "Zinsaufwendungen § 233a AO abzugsfähig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2108: OdooAccountTemplate = OdooAccountTemplate {
    code: "2108",
    name: "Zinsaufwendungen §§ 234 bis 237 AO nicht abzugsfähig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2109: OdooAccountTemplate = OdooAccountTemplate {
    code: "2109",
    name: "Zinsaufwendungen an verbundene Unternehmen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2110: OdooAccountTemplate = OdooAccountTemplate {
    code: "2110",
    name: "Zinsaufwendungen für kurzfristige Verbindlichkeiten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2113: OdooAccountTemplate = OdooAccountTemplate {
    code: "2113",
    name: "Nicht abzugsfähige Schuldzinsen nach § 4 Abs. 4a EStG (Hinzurechnungsbetrag)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2114: OdooAccountTemplate = OdooAccountTemplate {
    code: "2114",
    name: "Zinsen für Gesellschafterdarlehen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2115: OdooAccountTemplate = OdooAccountTemplate {
    code: "2115",
    name: "Zinsen und ähnliche Aufwendungen §§ 3 Nr. 40 und 3c EStG bzw. § 8b Abs. 1 und 4 KStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2116: OdooAccountTemplate = OdooAccountTemplate {
    code: "2116",
    name: "Zinsen und ähnliche Aufwendungen an verbundene Unternehmen §§ 3 Nr. 40 und 3c EStG bzw. § 8b Abs. 1 KStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2117: OdooAccountTemplate = OdooAccountTemplate {
    code: "2117",
    name: "Zinsen an Gesellschafter mit einer Beteiligung von mehr als 25 % bzw. diesen nahe stehende Personen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2118: OdooAccountTemplate = OdooAccountTemplate {
    code: "2118",
    name: "Zinsen auf Kontokorrentkonten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2119: OdooAccountTemplate = OdooAccountTemplate {
    code: "2119",
    name: "Zinsaufwendungen für kurzfristige Verbindlichkeiten an verbundene Unternehmen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2120: OdooAccountTemplate = OdooAccountTemplate {
    code: "2120",
    name: "Zinsaufwendungen für langfristige Verbindlichkeiten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2123: OdooAccountTemplate = OdooAccountTemplate {
    code: "2123",
    name: "Abschreibungen auf ein Agio oder Disagio/Damnum zur Finanzierung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2124: OdooAccountTemplate = OdooAccountTemplate {
    code: "2124",
    name: "Abschreibungen auf ein Agio oder Disagio/Damnum zur Finanzierung des Anlagevermögens",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2125: OdooAccountTemplate = OdooAccountTemplate {
    code: "2125",
    name: "Zinsaufwendungen für Gebäude, die zum Betriebsvermögen gehören",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2126: OdooAccountTemplate = OdooAccountTemplate {
    code: "2126",
    name: "Zinsen zur Finanzierung des Anlagevermögens",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2127: OdooAccountTemplate = OdooAccountTemplate {
    code: "2127",
    name: "Renten und dauernde Lasten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2128: OdooAccountTemplate = OdooAccountTemplate {
    code: "2128",
    name: "Zinsaufwendungen für Kapitalüberlassung durch Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2129: OdooAccountTemplate = OdooAccountTemplate {
    code: "2129",
    name: "Zinsaufwendungen für langfristige Verbindlichkeiten an verbundene Unternehmen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2130: OdooAccountTemplate = OdooAccountTemplate {
    code: "2130",
    name: "Diskontaufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2139: OdooAccountTemplate = OdooAccountTemplate {
    code: "2139",
    name: "Diskontaufwendungen an verbundene Unternehmen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2140: OdooAccountTemplate = OdooAccountTemplate {
    code: "2140",
    name: "Zinsähnliche Aufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2141: OdooAccountTemplate = OdooAccountTemplate {
    code: "2141",
    name: "Kreditprovisionen und Verwaltungskostenbeiträge",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2142: OdooAccountTemplate = OdooAccountTemplate {
    code: "2142",
    name: "Zinsanteil der Zuführungen zu Pensionsrückstellungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2143: OdooAccountTemplate = OdooAccountTemplate {
    code: "2143",
    name: "Zinsaufwendungen aus der Abzinsung von Verbindlichkeiten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2144: OdooAccountTemplate = OdooAccountTemplate {
    code: "2144",
    name: "Zinsaufwendungen aus der Abzinsung von Rückstellungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2145: OdooAccountTemplate = OdooAccountTemplate {
    code: "2145",
    name: "Zinsaufwendungen aus der Abzinsung von Pensionsrückstellungen und ähnlichen/vergleichbaren Verpflichtungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2146: OdooAccountTemplate = OdooAccountTemplate {
    code: "2146",
    name: "Zinsaufwendungen aus der Abzinsung von Pensionsrückstellungen und ähnlichen/vergleichbaren Verpflichtungen zur Verrechnung nach § 246 Abs. 2 HGB",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2147: OdooAccountTemplate = OdooAccountTemplate {
    code: "2147",
    name: "Aufwendungen aus Vermögensgegenständen zur Verrechnung nach § 246 Abs. 2 HGB",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2148: OdooAccountTemplate = OdooAccountTemplate {
    code: "2148",
    name: "Steuerlich nicht abzugsfähige Zinsaufwendungen aus der Abzinsung von Rückstellungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2149: OdooAccountTemplate = OdooAccountTemplate {
    code: "2149",
    name: "Zinsähnliche Aufwendungen an verbundene Unternehmen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2150: OdooAccountTemplate = OdooAccountTemplate {
    code: "2150",
    name: "Aufwendungen aus der Währungsumrechnung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2151: OdooAccountTemplate = OdooAccountTemplate {
    code: "2151",
    name: "Aufwendungen aus der Währungsumrechnung (nicht § 256a HGB)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2166: OdooAccountTemplate = OdooAccountTemplate {
    code: "2166",
    name: "Aufwendungen aus Bewertung Finanzmittelfonds",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2170: OdooAccountTemplate = OdooAccountTemplate {
    code: "2170",
    name: "Nicht abziehbare Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2171: OdooAccountTemplate = OdooAccountTemplate {
    code: "2171",
    name: "Nicht abziehbare Vorsteuer 7 %",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2176: OdooAccountTemplate = OdooAccountTemplate {
    code: "2176",
    name: "Nicht abziehbare Vorsteuer 19 %",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2200: OdooAccountTemplate = OdooAccountTemplate {
    code: "2200",
    name: "Körperschaftsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2203: OdooAccountTemplate = OdooAccountTemplate {
    code: "2203",
    name: "Körperschaftsteuer für Vorjahre",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2204: OdooAccountTemplate = OdooAccountTemplate {
    code: "2204",
    name: "Körperschaftsteuererstattungen für Vorjahre",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2208: OdooAccountTemplate = OdooAccountTemplate {
    code: "2208",
    name: "Solidaritätszuschlag",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2209: OdooAccountTemplate = OdooAccountTemplate {
    code: "2209",
    name: "Solidaritätszuschlag für Vorjahre",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2210: OdooAccountTemplate = OdooAccountTemplate {
    code: "2210",
    name: "Solidaritätszuschlagerstattungen für Vorjahre",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2213: OdooAccountTemplate = OdooAccountTemplate {
    code: "2213",
    name: "Kapitalertragsteuer 25 %",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2216: OdooAccountTemplate = OdooAccountTemplate {
    code: "2216",
    name: "Anrechenbarer Solidaritätszuschlag auf Kapitalertragsteuer 25 %",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2219: OdooAccountTemplate = OdooAccountTemplate {
    code: "2219",
    name: "Anrechnung/Abzug ausländische Quellensteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2250: OdooAccountTemplate = OdooAccountTemplate {
    code: "2250",
    name: "Aufwendungen aus der Zuführung und Auflösung von latenten Steuern",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2255: OdooAccountTemplate = OdooAccountTemplate {
    code: "2255",
    name: "Erträge aus der Zuführung und Auflösung von latenten Steuern",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2260: OdooAccountTemplate = OdooAccountTemplate {
    code: "2260",
    name: "Aufwendungen aus der Zuführung zu Steuerrückstellungen für Steuerstundung (BStBK)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2265: OdooAccountTemplate = OdooAccountTemplate {
    code: "2265",
    name: "Erträge aus der Auflösung von Steuerrückstellungen für Steuerstundung (BStBK)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2280: OdooAccountTemplate = OdooAccountTemplate {
    code: "2280",
    name: "Gewerbesteuernachzahlung Vorjahre",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2281: OdooAccountTemplate = OdooAccountTemplate {
    code: "2281",
    name: "Gewerbesteuernachzahlungen und Gewerbesteuererstattungen für Vorjahre nach § 4 Abs. 5b EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2282: OdooAccountTemplate = OdooAccountTemplate {
    code: "2282",
    name: "Gewerbesteuerrückerstattung Vorjahre",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2283: OdooAccountTemplate = OdooAccountTemplate {
    code: "2283",
    name: "Erträge aus der Auflösung von Gewerbesteuerrückstellungen nach § 4 Abs. 5b EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2284: OdooAccountTemplate = OdooAccountTemplate {
    code: "2284",
    name: "Auflösung der Gewerbesteuerrückstellung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2285: OdooAccountTemplate = OdooAccountTemplate {
    code: "2285",
    name: "Steuernachzahlungen Vorjahre für sonstige Steuern",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2287: OdooAccountTemplate = OdooAccountTemplate {
    code: "2287",
    name: "Steuererstattungen Vorjahre für sonstige Steuern",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2289: OdooAccountTemplate = OdooAccountTemplate {
    code: "2289",
    name: "Erträge aus der Auflösung von Rückstellungen für sonstige Steuern",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2300: OdooAccountTemplate = OdooAccountTemplate {
    code: "2300",
    name: "Sonstige Aufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2307: OdooAccountTemplate = OdooAccountTemplate {
    code: "2307",
    name: "Sonstige Aufwendungen betriebsfremd und regelmäßig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2308: OdooAccountTemplate = OdooAccountTemplate {
    code: "2308",
    name: "Sonstige nicht abziehbare Aufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2309: OdooAccountTemplate = OdooAccountTemplate {
    code: "2309",
    name: "Sonstige Aufwendungen unregelmäßig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2310: OdooAccountTemplate = OdooAccountTemplate {
    code: "2310",
    name: "Anlagenabgänge Sachanlagen (Restbuchwert bei Buchverlust)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2311: OdooAccountTemplate = OdooAccountTemplate {
    code: "2311",
    name: "Anlagenabgänge immaterielle Vermögensgegenstände (Restbuchwert bei Buchverlust)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2312: OdooAccountTemplate = OdooAccountTemplate {
    code: "2312",
    name: "Anlagenabgänge Finanzanlagen (Restbuchwert bei Buchverlust)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2313: OdooAccountTemplate = OdooAccountTemplate {
    code: "2313",
    name: "Anlagenabgänge Finanzanlagen § 3 Nr. 40 EStG bzw. § 8b Abs. 3 KStG (Restbuchwert bei Buchverlust)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2315: OdooAccountTemplate = OdooAccountTemplate {
    code: "2315",
    name: "Anlagenabgänge Sachanlagen (Restbuchwert bei Buchgewinn)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2316: OdooAccountTemplate = OdooAccountTemplate {
    code: "2316",
    name: "Anlagenabgänge immaterielle Vermögensgegenstände (Restbuchwert bei Buchgewinn)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2317: OdooAccountTemplate = OdooAccountTemplate {
    code: "2317",
    name: "Anlagenabgänge Finanzanlagen (Restbuchwert bei Buchgewinn)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2318: OdooAccountTemplate = OdooAccountTemplate {
    code: "2318",
    name: "Anlagenabgänge Finanzanlagen § 3 Nr. 40 EStG bzw. § 8b Abs. 2 KStG (Restbuchwert bei Buchgewinn)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2320: OdooAccountTemplate = OdooAccountTemplate {
    code: "2320",
    name: "Verluste aus dem Abgang von Gegenständen des Anlagevermögens",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2323: OdooAccountTemplate = OdooAccountTemplate {
    code: "2323",
    name: "Verluste aus der Veräußerung von Anteilen an Kapitalgesellschaften (Finanzanlagevermögen) § 3 Nr. 40 EStG bzw. § 8b Abs. 3 KStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2325: OdooAccountTemplate = OdooAccountTemplate {
    code: "2325",
    name: "Verluste aus dem Abgang von Gegenständen des Umlaufvermögens (außer Vorräte)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2326: OdooAccountTemplate = OdooAccountTemplate {
    code: "2326",
    name: "Verluste aus dem Abgang von Gegenständen des Umlaufvermögens (außer Vorräte) § 3 Nr. 40 EStG bzw. § 8b Abs. 3 KStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2327: OdooAccountTemplate = OdooAccountTemplate {
    code: "2327",
    name: "Abgang von Wirtschaftsgütern des Umlaufvermögens nach § 4 Abs. 3 Satz 4 EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2328: OdooAccountTemplate = OdooAccountTemplate {
    code: "2328",
    name: "Abgang von Wirtschaftsgütern des Umlaufvermögens § 3 Nr. 40 EStG bzw. § 8b Abs. 3 KStG nach § 4 Abs. 3 Satz 4 EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2339: OdooAccountTemplate = OdooAccountTemplate {
    code: "2339",
    name: "Einstellungen in die steuerliche Rücklage nach § 4g EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2340: OdooAccountTemplate = OdooAccountTemplate {
    code: "2340",
    name: "Einstellungen SoPo mit Reserveanteil",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2341: OdooAccountTemplate = OdooAccountTemplate {
    code: "2341",
    name: "Einstellungen SoPo § 7g Abs.2 EStG n.F.",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2342: OdooAccountTemplate = OdooAccountTemplate {
    code: "2342",
    name: "Einstellungen in die steuerliche Rücklage nach § 6b Abs. 3 EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2344: OdooAccountTemplate = OdooAccountTemplate {
    code: "2344",
    name: "Einstellungen in die Rücklage für Ersatzbeschaffung nach R 6.6 EStR",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2345: OdooAccountTemplate = OdooAccountTemplate {
    code: "2345",
    name: "Einstellungen in sonstige steuerliche Rücklagen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2347: OdooAccountTemplate = OdooAccountTemplate {
    code: "2347",
    name: "Aufwendungen aus dem Erwerb eigener Anteile",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2350: OdooAccountTemplate = OdooAccountTemplate {
    code: "2350",
    name: "Sonstige Grundstücksaufwendungen (neutral)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2375: OdooAccountTemplate = OdooAccountTemplate {
    code: "2375",
    name: "Grundsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2380: OdooAccountTemplate = OdooAccountTemplate {
    code: "2380",
    name: "Zuwendungen, Spenden, steuerlich nicht abziehbar",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2381: OdooAccountTemplate = OdooAccountTemplate {
    code: "2381",
    name: "Zuwendungen, Spenden für wissenschaftliche und kulturelle Zwecke",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2382: OdooAccountTemplate = OdooAccountTemplate {
    code: "2382",
    name: "Zuwendungen, Spenden für mildtätige Zwecke",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2383: OdooAccountTemplate = OdooAccountTemplate {
    code: "2383",
    name: "Zuwendungen, Spenden für kirchliche, religiöse und gemeinnützige Zwecke",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2384: OdooAccountTemplate = OdooAccountTemplate {
    code: "2384",
    name: "Zuwendungen, Spenden an politische Parteien",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2385: OdooAccountTemplate = OdooAccountTemplate {
    code: "2385",
    name: "Nicht abziehbare Hälfte der Aufsichtsratsvergütungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2386: OdooAccountTemplate = OdooAccountTemplate {
    code: "2386",
    name: "Abziehbare Aufsichtsratsvergütungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2387: OdooAccountTemplate = OdooAccountTemplate {
    code: "2387",
    name: "Zuwendungen, Spenden in das zu erhaltende Vermögen (Vermögensstock) einer Stiftung für gemeinnützige Zwecke",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2388: OdooAccountTemplate = OdooAccountTemplate {
    code: "2388",
    name: "Zuwendungen, Spenden an Stiftungen für gemeinnützige Zwecke i.S.d. des § 52 Abs. 2 Nr. 4 der Abgabenordnung (AO)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2389: OdooAccountTemplate = OdooAccountTemplate {
    code: "2389",
    name: "Zuwendungen, Spenden in das zu erhaltende Vermögen (Vermögensstock) einer Stiftung für kirchliche, religiöse und gemeinnützige Zwecke",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2390: OdooAccountTemplate = OdooAccountTemplate {
    code: "2390",
    name: "Zuwendungen, Spenden an Stiftungen in das zu erhaltende Vermögen (Vermögensstock) für wissenschaftliche, mildtätige, kulturelle Zwecke",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2400: OdooAccountTemplate = OdooAccountTemplate {
    code: "2400",
    name: "Forderungsverluste (übliche Höhe)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2401: OdooAccountTemplate = OdooAccountTemplate {
    code: "2401",
    name: "Forderungsverluste 7 % USt (übliche Höhe)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2402: OdooAccountTemplate = OdooAccountTemplate {
    code: "2402",
    name: "Forderungsverluste aus steuerfreien EU-Lieferungen (übliche Höhe)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2403: OdooAccountTemplate = OdooAccountTemplate {
    code: "2403",
    name: "Forderungsverluste aus im Inland steuerpflichtigen EU-Lieferungen 7 % USt (übliche Höhe)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2406: OdooAccountTemplate = OdooAccountTemplate {
    code: "2406",
    name: "Forderungsverluste 19 % USt (übliche Höhe)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2408: OdooAccountTemplate = OdooAccountTemplate {
    code: "2408",
    name: "Forderungsverluste aus im Inland steuerpflichtigen EU-Lieferungen 19 % USt (übliche Höhe)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2430: OdooAccountTemplate = OdooAccountTemplate {
    code: "2430",
    name: "Forderungsverluste, unüblich hoch",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2431: OdooAccountTemplate = OdooAccountTemplate {
    code: "2431",
    name: "Forderungsverluste 7 % USt (soweit unüblich hoch)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2436: OdooAccountTemplate = OdooAccountTemplate {
    code: "2436",
    name: "Forderungsverluste 19 % USt (soweit unüblich hoch)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2440: OdooAccountTemplate = OdooAccountTemplate {
    code: "2440",
    name: "Abschreibungen auf Forderungen gegenüber Kapitalgesellschaften, an denen eine Beteiligung besteht (soweit unüblich hoch), § 3c EStG bzw. § 8b Abs. 3 KStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2441: OdooAccountTemplate = OdooAccountTemplate {
    code: "2441",
    name: "Abschreibungen auf Forderungen gegenüber Gesellschaftern und nahe stehenden Personen (soweit unüblich hoch), § 8b Abs. 3 KStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2450: OdooAccountTemplate = OdooAccountTemplate {
    code: "2450",
    name: "Einstellungen in die Pauschalwertberichtigung auf Forderungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2451: OdooAccountTemplate = OdooAccountTemplate {
    code: "2451",
    name: "Einstellungen in die Einzelwertberichtigung auf Forderungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2480: OdooAccountTemplate = OdooAccountTemplate {
    code: "2480",
    name: "Einstellungen in die Rücklage für Anteile an einem herrschenden oder mehrheitlich beteiligten Unternehmen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2481: OdooAccountTemplate = OdooAccountTemplate {
    code: "2481",
    name: "Einstellungen in gesamthänderisch gebundene Rücklagen (mit Aufteilung für Kapitalkontenentwicklung)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2485: OdooAccountTemplate = OdooAccountTemplate {
    code: "2485",
    name: "Einstellungen in andere Ergebnisrücklagen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2490: OdooAccountTemplate = OdooAccountTemplate {
    code: "2490",
    name: "Aufwendungen aus Verlustübernahme",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2492: OdooAccountTemplate = OdooAccountTemplate {
    code: "2492",
    name: "Abgeführte Gewinne auf Grund einer Gewinngemeinschaft",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2493: OdooAccountTemplate = OdooAccountTemplate {
    code: "2493",
    name: "Abgeführte Gewinnanteile (Soll) / ausgeglichene Verlustanteile (Haben) bei typisch stiller Beteiligung § 8 GewStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2494: OdooAccountTemplate = OdooAccountTemplate {
    code: "2494",
    name: "Abgeführte Gewinne auf Grund eines Gewinn- oder Teilgewinnabführungsvertrags",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2498: OdooAccountTemplate = OdooAccountTemplate {
    code: "2498",
    name: "Einstellungen in den Ausgleichsposten für aktivierte eigene Anteile",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_2500: OdooAccountTemplate = OdooAccountTemplate {
    code: "2500",
    name: "Außerordentliche Erträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2501: OdooAccountTemplate = OdooAccountTemplate {
    code: "2501",
    name: "Nichtoperative Erträge, die das Nettoergebnis beeinflussen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2504: OdooAccountTemplate = OdooAccountTemplate {
    code: "2504",
    name: "Erträge durch Verschmelzung und Umwandlung",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2505: OdooAccountTemplate = OdooAccountTemplate {
    code: "2505",
    name: "Nicht-operative Erträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2506: OdooAccountTemplate = OdooAccountTemplate {
    code: "2506",
    name: "Erträge aus dem Verkauf von wesentlichen Beteiligungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2507: OdooAccountTemplate = OdooAccountTemplate {
    code: "2507",
    name: "Erträge aus dem Verkauf wesentlicher Immobilien",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2508: OdooAccountTemplate = OdooAccountTemplate {
    code: "2508",
    name: "Gewinn aus der Veräußerung oder der Aufgabe von Geschäftsaktivitäten nach Steuern",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2510: OdooAccountTemplate = OdooAccountTemplate {
    code: "2510",
    name: "Nicht-operative Erträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2520: OdooAccountTemplate = OdooAccountTemplate {
    code: "2520",
    name: "Periodenfremde Erträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2590: OdooAccountTemplate = OdooAccountTemplate {
    code: "2590",
    name: "Erträge aus der Anwendung von Übergangsvorschriften",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2591: OdooAccountTemplate = OdooAccountTemplate {
    code: "2591",
    name: "Außerordentliche Erträge aus der Anwendung von Übergangsbestimmungen (Zuschreibung für Sachanlagen)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2592: OdooAccountTemplate = OdooAccountTemplate {
    code: "2592",
    name: "Außerordentliche Erträge aus der Anwendung von Übergangsbestimmungen (Zuschreibung für Finanzanlagen)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2593: OdooAccountTemplate = OdooAccountTemplate {
    code: "2593",
    name: "Außerordentliche Erträge aus der Anwendung von Übergangsbestimmungen (Wertpapiere des Umlaufvermögens)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2594: OdooAccountTemplate = OdooAccountTemplate {
    code: "2594",
    name: "Erträge aus der Anwendung von Übergangsvorschriften (latente Steuern)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2600: OdooAccountTemplate = OdooAccountTemplate {
    code: "2600",
    name: "Erträge aus Beteiligungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_09"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2603: OdooAccountTemplate = OdooAccountTemplate {
    code: "2603",
    name: "Erträge aus Beteiligungen an Personengesellschaften (verbundene Unternehmen), § 9 GewStG bzw. § 18 EStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_09"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2615: OdooAccountTemplate = OdooAccountTemplate {
    code: "2615",
    name: "Erträge aus Anteilen an Kapitalgesellschaften (Beteiligung) § 3 Nr. 40 EStG bzw. § 8b Abs. 1 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_09"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2616: OdooAccountTemplate = OdooAccountTemplate {
    code: "2616",
    name: "Erträge aus Anteilen an Kapitalgesellschaften (verbundene Unternehmen) § 3 Nr. 40 EStG bzw. § 8b Abs. 1 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_09"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2617: OdooAccountTemplate = OdooAccountTemplate {
    code: "2617",
    name: "Einkünfte aus Anteilen an Kapitalgesellschaften (verbundene Unternehmen) § 3 Nr. 40 EStG/§ 8b (1) KStG (inländische Kap.Ges.)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_09"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2618: OdooAccountTemplate = OdooAccountTemplate {
    code: "2618",
    name: "Gewinnanteile aus gewerblichen und selbständigen Mitunternehmerschaften, § 9 GewStG bzw. § 18 EStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_09"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2619: OdooAccountTemplate = OdooAccountTemplate {
    code: "2619",
    name: "Erträge aus Beteiligungen an verbundenen Unternehmen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_09"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2620: OdooAccountTemplate = OdooAccountTemplate {
    code: "2620",
    name: "Erträge aus anderen Wertpapieren und Ausleihungen des Finanzanlagevermögens",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2621: OdooAccountTemplate = OdooAccountTemplate {
    code: "2621",
    name: "Erträge aus Ausleihungen des Finanzanlagevermögens",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2622: OdooAccountTemplate = OdooAccountTemplate {
    code: "2622",
    name: "Erträge aus Ausleihungen des Finanzanlagevermögens an verbundenen Unternehmen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2623: OdooAccountTemplate = OdooAccountTemplate {
    code: "2623",
    name: "Erträge aus Anteilen an Personengesellschaften (Finanzanlagevermögen)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2625: OdooAccountTemplate = OdooAccountTemplate {
    code: "2625",
    name: "Erträge aus Anteilen an Kapitalgesellschaften (Finanzanlagevermögen) § 3 Nr. 40 EStG bzw. § 8b Abs. 1 und 4 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2626: OdooAccountTemplate = OdooAccountTemplate {
    code: "2626",
    name: "Erträge aus Anteilen an Kapitalgesellschaften (verbundene Unternehmen) § 3 Nr. 40 EStG bzw. § 8b Abs. 1 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2640: OdooAccountTemplate = OdooAccountTemplate {
    code: "2640",
    name: "Zins- und Dividendenerträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2641: OdooAccountTemplate = OdooAccountTemplate {
    code: "2641",
    name: "Erhaltene Ausgleichszahlungen (als außenstehender Aktionär)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2646: OdooAccountTemplate = OdooAccountTemplate {
    code: "2646",
    name: "Erträge aus Anteilen an Personengesellschaften (verbundene Unternehmen)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2647: OdooAccountTemplate = OdooAccountTemplate {
    code: "2647",
    name: "Erträge aus anderen Wertpapieren des Finanzanlagevermögens an Kapitalgesellschaften (verbundene Unternehmen)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2648: OdooAccountTemplate = OdooAccountTemplate {
    code: "2648",
    name: "Erträge aus anderen Wertpapieren des Finanzanlagevermögens an Personengesellschaften (verbundene Unternehmen)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2649: OdooAccountTemplate = OdooAccountTemplate {
    code: "2649",
    name: "Erträge aus anderen Wertpapieren und Ausleihungen des Finanzanlagevermögens aus verbundenen Unternehmen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2650: OdooAccountTemplate = OdooAccountTemplate {
    code: "2650",
    name: "Sonstige Zinsen und ähnliche Erträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2652: OdooAccountTemplate = OdooAccountTemplate {
    code: "2652",
    name: "Aufstockung des Körperschaftsteuerguthabens",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2653: OdooAccountTemplate = OdooAccountTemplate {
    code: "2653",
    name: "Zinserträge § 233a AO und § 4 Abs. 5b EStG, steuerfrei",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2654: OdooAccountTemplate = OdooAccountTemplate {
    code: "2654",
    name: "Erträge aus anderen Wertpapieren und Ausleihungen des Umlaufvermögens",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2655: OdooAccountTemplate = OdooAccountTemplate {
    code: "2655",
    name: "Erträge aus Anteilen an Kapitalgesellschaften (Umlaufvermögen) § 3 Nr. 40 EStG bzw. § 8b Abs. 1 und 4 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2656: OdooAccountTemplate = OdooAccountTemplate {
    code: "2656",
    name: "Erträge aus Anteilen an Kapitalgesellschaften (verbundene Unternehmen) § 3 Nr. 40 EStG bzw. § 8b Abs. 1 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2657: OdooAccountTemplate = OdooAccountTemplate {
    code: "2657",
    name: "Zinserträge § 233a AO, steuerpflichtig",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2658: OdooAccountTemplate = OdooAccountTemplate {
    code: "2658",
    name: "Zinserträge § 233a AO, steuerfrei (Anlage GK KSt)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2659: OdooAccountTemplate = OdooAccountTemplate {
    code: "2659",
    name: "Sonstige Zinsen und ähnliche Erträge aus verbundenen Unternehmen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2660: OdooAccountTemplate = OdooAccountTemplate {
    code: "2660",
    name: "Erträge aus der Währungsumrechnung",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2661: OdooAccountTemplate = OdooAccountTemplate {
    code: "2661",
    name: "Erträge aus der Währungsumrechnung (nicht § 256a HGB)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2666: OdooAccountTemplate = OdooAccountTemplate {
    code: "2666",
    name: "Erträge aus Bewertung Finanzmittelfonds",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2670: OdooAccountTemplate = OdooAccountTemplate {
    code: "2670",
    name: "Diskonterträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2679: OdooAccountTemplate = OdooAccountTemplate {
    code: "2679",
    name: "Diskonterträge aus verbundenen Unternehmen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2680: OdooAccountTemplate = OdooAccountTemplate {
    code: "2680",
    name: "Zinsähnliche Erträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2682: OdooAccountTemplate = OdooAccountTemplate {
    code: "2682",
    name: "Steuerfreie Zinserträge aus der Abzinsung von Rückstellungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2683: OdooAccountTemplate = OdooAccountTemplate {
    code: "2683",
    name: "Zinserträge aus der Abzinsung von Verbindlichkeiten",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2684: OdooAccountTemplate = OdooAccountTemplate {
    code: "2684",
    name: "Zinserträge aus der Abzinsung von Rückstellungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2685: OdooAccountTemplate = OdooAccountTemplate {
    code: "2685",
    name: "Zinserträge aus der Abzinsung von Pensionsrückstellungen und ähnlichen/vergleichbaren Verpflichtungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2686: OdooAccountTemplate = OdooAccountTemplate {
    code: "2686",
    name: "Zinserträge aus der Abzinsung von Pensionsrückstellungen und ähnlichen/vergleichbaren Verpflichtungen zur Verrechnung nach § 246 Abs. 2 HGB",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2687: OdooAccountTemplate = OdooAccountTemplate {
    code: "2687",
    name: "Erträge aus Vermögensgegenständen zur Verrechnung nach § 246 Abs. 2 HGB",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2688: OdooAccountTemplate = OdooAccountTemplate {
    code: "2688",
    name: "Zinserträge Rückzahlung der Steuererhöhung §38",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2689: OdooAccountTemplate = OdooAccountTemplate {
    code: "2689",
    name: "Zinsähnliche Erträge aus verbundenen Unternehmen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2700: OdooAccountTemplate = OdooAccountTemplate {
    code: "2700",
    name: "Andere betriebs- und/oder periodenfremde (neutrale) sonstige Erträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2705: OdooAccountTemplate = OdooAccountTemplate {
    code: "2705",
    name: "Sonstige betriebliche und regelmäßige Erträge (neutral)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2707: OdooAccountTemplate = OdooAccountTemplate {
    code: "2707",
    name: "Sonstige Erträge betriebsfremd und regelmäßig",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2709: OdooAccountTemplate = OdooAccountTemplate {
    code: "2709",
    name: "Sonstige Erträge unregelmäßig",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2710: OdooAccountTemplate = OdooAccountTemplate {
    code: "2710",
    name: "Erträge aus Zuschreibungen des Sachanlagevermögens",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2711: OdooAccountTemplate = OdooAccountTemplate {
    code: "2711",
    name: "Erträge aus Zuschreibungen des immateriellen Anlagevermögens",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2712: OdooAccountTemplate = OdooAccountTemplate {
    code: "2712",
    name: "Erträge aus Zuschreibungen des Finanzanlagevermögens",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2713: OdooAccountTemplate = OdooAccountTemplate {
    code: "2713",
    name: "Erträge aus Zuschreibungen des Finanzanlagevermögens § 3 Nr. 40 EStG bzw. § 8b Abs. 3 Satz 8 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2714: OdooAccountTemplate = OdooAccountTemplate {
    code: "2714",
    name: "Erträge aus Zuschreibungen § 3 Nr. 40 EStG bzw. § 8b Abs. 2 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2715: OdooAccountTemplate = OdooAccountTemplate {
    code: "2715",
    name: "Erträge aus Zuschreibungen des Umlaufvermögens (außer Vorräte)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2716: OdooAccountTemplate = OdooAccountTemplate {
    code: "2716",
    name: "Erträge aus Zuschreibungen des Umlaufvermögens § 3 Nr. 40 EStG bzw. § 8b Abs. 3 Satz 8 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2720: OdooAccountTemplate = OdooAccountTemplate {
    code: "2720",
    name: "Erträge aus dem Abgang von Gegenständen des Anlagevermögens",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2723: OdooAccountTemplate = OdooAccountTemplate {
    code: "2723",
    name: "Erträge aus der Veräußerung von Anteilen an Kapitalgesellschaften (Finanzanlagevermögen) § 3 Nr. 40 EStG bzw. § 8b Abs. 2 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2725: OdooAccountTemplate = OdooAccountTemplate {
    code: "2725",
    name: "Erträge aus dem Abgang von Gegenständen des Umlaufvermögens (außer Vorräte)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2726: OdooAccountTemplate = OdooAccountTemplate {
    code: "2726",
    name: "Erträge aus dem Abgang von Gegenständen des Umlaufvermögens (außer Vorräte) § 3 Nr. 40 EStG bzw. § 8b Abs. 2 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2727: OdooAccountTemplate = OdooAccountTemplate {
    code: "2727",
    name: "Erträge aus der Auflösung einer steuerlichen Rücklage nach § 6b Abs. 3 EStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2728: OdooAccountTemplate = OdooAccountTemplate {
    code: "2728",
    name: "Erträge aus der Auflösung einer steuerlichen Rücklage nach § 6b Abs. 10 EStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2729: OdooAccountTemplate = OdooAccountTemplate {
    code: "2729",
    name: "Erträge aus der Auflösung der Rücklage für Ersatzbeschaffung, R 6.6 EStR",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2730: OdooAccountTemplate = OdooAccountTemplate {
    code: "2730",
    name: "Erträge aus der Herabsetzung der Pauschalwertberichtigung auf Forderungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2731: OdooAccountTemplate = OdooAccountTemplate {
    code: "2731",
    name: "Erträge aus der Herabsetzung der Einzelwertberichtigung auf Forderungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2732: OdooAccountTemplate = OdooAccountTemplate {
    code: "2732",
    name: "Erträge aus abgeschriebenen Forderungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2733: OdooAccountTemplate = OdooAccountTemplate {
    code: "2733",
    name: "Erträge aus der Auflösung einer steuerlichen Rücklage nach § 7g Abs. 7 EStG a.F. (Existenzgründungsrücklage)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2734: OdooAccountTemplate = OdooAccountTemplate {
    code: "2734",
    name: "Einkommen Bewertung Verbindlichkeiten",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2735: OdooAccountTemplate = OdooAccountTemplate {
    code: "2735",
    name: "Erträge aus der Auflösung von Rückstellungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2736: OdooAccountTemplate = OdooAccountTemplate {
    code: "2736",
    name: "Erträge aus der Herabsetzung von Verbindlichkeiten",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2738: OdooAccountTemplate = OdooAccountTemplate {
    code: "2738",
    name: "Erträge aus der Auflösung von Steuerrückstellungen nach § 52 Abs. 16 EStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2739: OdooAccountTemplate = OdooAccountTemplate {
    code: "2739",
    name: "Erträge aus der Auflösung von Steuerrückstellungen (Ansparabschreibung nach § 7g Abs. 2 EStG)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2740: OdooAccountTemplate = OdooAccountTemplate {
    code: "2740",
    name: "Erträge aus der Auflösung sonstiger steuerlicher Rücklagen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2741: OdooAccountTemplate = OdooAccountTemplate {
    code: "2741",
    name: "Erträge aus der Auflösung steuerrechtlicher Sonderabschreibungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2742: OdooAccountTemplate = OdooAccountTemplate {
    code: "2742",
    name: "Versicherungsentschädigungen und Schadenersatzleistungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2743: OdooAccountTemplate = OdooAccountTemplate {
    code: "2743",
    name: "Investitionszuschüsse (steuerpflichtig)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2744: OdooAccountTemplate = OdooAccountTemplate {
    code: "2744",
    name: "Investitionszulagen (steuerfrei)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2746: OdooAccountTemplate = OdooAccountTemplate {
    code: "2746",
    name: "Steuerfreie Erträge aus der Auflösung von steuerlichen Rücklagen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2747: OdooAccountTemplate = OdooAccountTemplate {
    code: "2747",
    name: "Sonstige steuerfreie Betriebseinnahmen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2749: OdooAccountTemplate = OdooAccountTemplate {
    code: "2749",
    name: "Erstattungen Aufwendungsausgleichsgesetz",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2750: OdooAccountTemplate = OdooAccountTemplate {
    code: "2750",
    name: "Grundstückserträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2751: OdooAccountTemplate = OdooAccountTemplate {
    code: "2751",
    name: "Erlöse aus Vermietung und Verpachtung, umsatzsteuerfrei § 4 Nr. 12 UStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2752: OdooAccountTemplate = OdooAccountTemplate {
    code: "2752",
    name: "Erlöse aus Vermietung und Verpachtung 19 % USt",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2760: OdooAccountTemplate = OdooAccountTemplate {
    code: "2760",
    name: "Erträge aus der Aktivierung unentgeltlich erworbener Vermögensgegenstände",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2762: OdooAccountTemplate = OdooAccountTemplate {
    code: "2762",
    name: "Kostenerstattungen, Rückvergütungen und Gutschriften für frühere Jahre",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2764: OdooAccountTemplate = OdooAccountTemplate {
    code: "2764",
    name: "Erträge aus Verwaltungskostenumlage",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2790: OdooAccountTemplate = OdooAccountTemplate {
    code: "2790",
    name: "Erträge aus Verlustübernahme",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2792: OdooAccountTemplate = OdooAccountTemplate {
    code: "2792",
    name: "Erhaltene Gewinne auf Grund einer Gewinngemeinschaft",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2794: OdooAccountTemplate = OdooAccountTemplate {
    code: "2794",
    name: "Erhaltene Gewinne auf Grund eines Gewinn- oder Teilgewinnabführungsvertrags",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2798: OdooAccountTemplate = OdooAccountTemplate {
    code: "2798",
    name: "Entnahmen aus dem Ausgleichsposten für aktivierte eigene Anteile",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2840: OdooAccountTemplate = OdooAccountTemplate {
    code: "2840",
    name: "Entnahmen aus der Rücklage für Anteile an einem herrschenden oder mehrheitlich beteiligten Unternehmen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2841: OdooAccountTemplate = OdooAccountTemplate {
    code: "2841",
    name: "Entnahmen aus gesamthänderisch gebundenen Rücklagen (mit Aufteilung für Kapitalkontenentwicklung)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2850: OdooAccountTemplate = OdooAccountTemplate {
    code: "2850",
    name: "Entnahmen aus anderen Ergebnisrücklagen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2865: OdooAccountTemplate = OdooAccountTemplate {
    code: "2865",
    name: "Gewinnvortrag nach Verwendung (mit Aufteilung für Kapitalkontenentwicklung)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2867: OdooAccountTemplate = OdooAccountTemplate {
    code: "2867",
    name: "Verlustvortrag nach Verwendung (mit Aufteilung für Kapitalkontenentwicklung)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2870: OdooAccountTemplate = OdooAccountTemplate {
    code: "2870",
    name: "Vorabausschüttung",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2890: OdooAccountTemplate = OdooAccountTemplate {
    code: "2890",
    name: "Verrechneter kalkulatorischer Unternehmerlohn",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2891: OdooAccountTemplate = OdooAccountTemplate {
    code: "2891",
    name: "Verrechnete kalkulatorische Miete und Pacht",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2892: OdooAccountTemplate = OdooAccountTemplate {
    code: "2892",
    name: "Verrechnete kalkulatorische Zinsen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2893: OdooAccountTemplate = OdooAccountTemplate {
    code: "2893",
    name: "Verrechnete kalkulatorische Abschreibungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2894: OdooAccountTemplate = OdooAccountTemplate {
    code: "2894",
    name: "Verrechnete kalkulatorische Wagnisse",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2895: OdooAccountTemplate = OdooAccountTemplate {
    code: "2895",
    name: "Verrechneter kalkulatorischer Lohn für unentgeltliche Mitarbeiter",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_2990: OdooAccountTemplate = OdooAccountTemplate {
    code: "2990",
    name: "Aufwendungen/Erträge aus Währungsumrechnungsdifferenzen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_3000: OdooAccountTemplate = OdooAccountTemplate {
    code: "3000",
    name: "Roh-, Hilfs- und Betriebsstoffe",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3010: OdooAccountTemplate = OdooAccountTemplate {
    code: "3010",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3030: OdooAccountTemplate = OdooAccountTemplate {
    code: "3030",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3060: OdooAccountTemplate = OdooAccountTemplate {
    code: "3060",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe, innergemeinschaftlicher Erwerb 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3062: OdooAccountTemplate = OdooAccountTemplate {
    code: "3062",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe, innergemeinschaftlicher Erwerb 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3066: OdooAccountTemplate = OdooAccountTemplate {
    code: "3066",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe, innergemeinschaftlicher Erwerb ohne Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3067: OdooAccountTemplate = OdooAccountTemplate {
    code: "3067",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe, innergemeinschaftlicherErwerb ohne Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3070: OdooAccountTemplate = OdooAccountTemplate {
    code: "3070",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe 5,5 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3071: OdooAccountTemplate = OdooAccountTemplate {
    code: "3071",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe 10,7 % / 9,5 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3075: OdooAccountTemplate = OdooAccountTemplate {
    code: "3075",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe aus einem USt-Lager § 13a UStG 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3076: OdooAccountTemplate = OdooAccountTemplate {
    code: "3076",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe aus einem USt-Lager § 13a UStG 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3089: OdooAccountTemplate = OdooAccountTemplate {
    code: "3089",
    name: "Erwerb Roh-, Hilfs- und Betriebsstoffe als letzter Abnehmer innerhalb Dreiecksgeschäft 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3090: OdooAccountTemplate = OdooAccountTemplate {
    code: "3090",
    name: "Energiestoffe (Fertigung)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3091: OdooAccountTemplate = OdooAccountTemplate {
    code: "3091",
    name: "Energiestoffe (Fertigung) 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3092: OdooAccountTemplate = OdooAccountTemplate {
    code: "3092",
    name: "Energiestoffe (Fertigung) 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3100: OdooAccountTemplate = OdooAccountTemplate {
    code: "3100",
    name: "Fremdleistungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3106: OdooAccountTemplate = OdooAccountTemplate {
    code: "3106",
    name: "Fremdleistungen 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3108: OdooAccountTemplate = OdooAccountTemplate {
    code: "3108",
    name: "Fremdleistungen 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3109: OdooAccountTemplate = OdooAccountTemplate {
    code: "3109",
    name: "Fremdleistungen ohne Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3110: OdooAccountTemplate = OdooAccountTemplate {
    code: "3110",
    name: "Bauleistungen eines im Inland ansässigen Unternehmers 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3113: OdooAccountTemplate = OdooAccountTemplate {
    code: "3113",
    name: "Sonstige Leistungen eines im anderen EU-Land ansässigen Unternehmers 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3115: OdooAccountTemplate = OdooAccountTemplate {
    code: "3115",
    name: "Leistungen eines im Ausland ansässigen Unternehmers 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3120: OdooAccountTemplate = OdooAccountTemplate {
    code: "3120",
    name: "Bauleistungen eines im Inland ansässigen Unternehmers 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3123: OdooAccountTemplate = OdooAccountTemplate {
    code: "3123",
    name: "Sonstige Leistungen eines im anderen EU-Land ansässigen Unternehmers 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3125: OdooAccountTemplate = OdooAccountTemplate {
    code: "3125",
    name: "Leistungen eines im Ausland ansässigen Unternehmers 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3130: OdooAccountTemplate = OdooAccountTemplate {
    code: "3130",
    name: "Bauleistungen eines im Inland ansässigen Unternehmers ohne Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3133: OdooAccountTemplate = OdooAccountTemplate {
    code: "3133",
    name: "Sonstige Leistungen eines im anderen EU-Land ansässigen Unternehmers ohne Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3135: OdooAccountTemplate = OdooAccountTemplate {
    code: "3135",
    name: "Leistungen eines im Ausland ansässigen Unternehmers ohne Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3140: OdooAccountTemplate = OdooAccountTemplate {
    code: "3140",
    name: "Bauleistungen eines im Inland ansässigen Unternehmers ohne Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3143: OdooAccountTemplate = OdooAccountTemplate {
    code: "3143",
    name: "Sonstige Leistungen eines im anderen EU-Land ansässigen Unternehmers ohne Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3145: OdooAccountTemplate = OdooAccountTemplate {
    code: "3145",
    name: "Leistungen eines im Ausland ansässigen Unternehmers ohne Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3150: OdooAccountTemplate = OdooAccountTemplate {
    code: "3150",
    name: "Erhaltene Skonti aus Leistungen, für die als Leistungsempfänger die Steuer nach § 13b UStG geschuldet wird",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3151: OdooAccountTemplate = OdooAccountTemplate {
    code: "3151",
    name: "Erhaltene Skonti aus Leistungen, für die als Leistungsempfänger die Steuer nach § 13b UStG geschuldet wird 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3153: OdooAccountTemplate = OdooAccountTemplate {
    code: "3153",
    name: "Erhaltene Skonti aus Leistungen, für die als Leistungsempfänger die Steuer nach § 13b UStG geschuldet wird ohne Vorsteuer aber mit Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3154: OdooAccountTemplate = OdooAccountTemplate {
    code: "3154",
    name: "Erhaltene Skonti aus Leistungen, für die als Leistungsempfänger die Steuer nach § 13b UStG geschuldet wird ohne Vorsteuer, mit 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3160: OdooAccountTemplate = OdooAccountTemplate {
    code: "3160",
    name: "Leistungen nach § 13b UStG mit Vorsteuerabzug",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3165: OdooAccountTemplate = OdooAccountTemplate {
    code: "3165",
    name: "Leistungen nach § 13b UStG ohne Vorsteuerabzug",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3170: OdooAccountTemplate = OdooAccountTemplate {
    code: "3170",
    name: "Fremdleistungen (Miet- und Pachtzinsen bewegliche Wirtschaftsgüter)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3175: OdooAccountTemplate = OdooAccountTemplate {
    code: "3175",
    name: "Fremdleistungen (Miet- und Pachtzinsen unbewegliche Wirtschaftsgüter)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3180: OdooAccountTemplate = OdooAccountTemplate {
    code: "3180",
    name: "Fremdleistungen (Entgelte für Rechte und Lizenzen)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3185: OdooAccountTemplate = OdooAccountTemplate {
    code: "3185",
    name: "Fremdleistungen (Vergütungen für die Überlassung von Wirtschaftsgütern - mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3200: OdooAccountTemplate = OdooAccountTemplate {
    code: "3200",
    name: "Wareneingang",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3300: OdooAccountTemplate = OdooAccountTemplate {
    code: "3300",
    name: "Wareneingang 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3349: OdooAccountTemplate = OdooAccountTemplate {
    code: "3349",
    name: "Wareneingang ohne Vorsteuerabzug",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3400: OdooAccountTemplate = OdooAccountTemplate {
    code: "3400",
    name: "Wareneingang 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3420: OdooAccountTemplate = OdooAccountTemplate {
    code: "3420",
    name: "Innergemeinschaftlicher Erwerb 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3425: OdooAccountTemplate = OdooAccountTemplate {
    code: "3425",
    name: "Innergemeinschaftlicher Erwerb 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3430: OdooAccountTemplate = OdooAccountTemplate {
    code: "3430",
    name: "Innergemeinschaftlicher Erwerb ohne Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &[],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3435: OdooAccountTemplate = OdooAccountTemplate {
    code: "3435",
    name: "Innergemeinschaftlicher Erwerb ohne Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &[],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3440: OdooAccountTemplate = OdooAccountTemplate {
    code: "3440",
    name: "Innergemeinschaftlicher Erwerb von Neufahrzeugen von Lieferanten ohne USt-Id-Nr. 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3505: OdooAccountTemplate = OdooAccountTemplate {
    code: "3505",
    name: "Wareneingang 5,5 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3540: OdooAccountTemplate = OdooAccountTemplate {
    code: "3540",
    name: "Wareneingang zum Durchschnittssatz nach § 24 UStG 10,7 % / 9,5 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3550: OdooAccountTemplate = OdooAccountTemplate {
    code: "3550",
    name: "Steuerfreier innergemeinschaftlicher Erwerb",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3551: OdooAccountTemplate = OdooAccountTemplate {
    code: "3551",
    name: "Wareneingang im Drittland steuerbar",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3552: OdooAccountTemplate = OdooAccountTemplate {
    code: "3552",
    name: "Erwerb 1. Abnehmer innerhalb eines Dreiecksgeschäftes",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3553: OdooAccountTemplate = OdooAccountTemplate {
    code: "3553",
    name: "Erwerb Waren als letzter Abnehmer innerhalb Dreiecksgeschäft 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3557: OdooAccountTemplate = OdooAccountTemplate {
    code: "3557",
    name: "Wareneingang, steuerpflichtig im Drittland (7%)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3558: OdooAccountTemplate = OdooAccountTemplate {
    code: "3558",
    name: "Wareneingang im anderen EU-Land steuerbar",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3559: OdooAccountTemplate = OdooAccountTemplate {
    code: "3559",
    name: "Steuerfreie Einfuhren",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3560: OdooAccountTemplate = OdooAccountTemplate {
    code: "3560",
    name: "Waren aus einem Umsatzsteuerlager, § 13a UStG 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3565: OdooAccountTemplate = OdooAccountTemplate {
    code: "3565",
    name: "Waren aus einem Umsatzsteuerlager, § 13a UStG 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3600: OdooAccountTemplate = OdooAccountTemplate {
    code: "3600",
    name: "Nicht abziehbare Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3610: OdooAccountTemplate = OdooAccountTemplate {
    code: "3610",
    name: "Nicht abziehbare Vorsteuer 7 %",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3660: OdooAccountTemplate = OdooAccountTemplate {
    code: "3660",
    name: "Nicht abziehbare Vorsteuer 19 %",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3700: OdooAccountTemplate = OdooAccountTemplate {
    code: "3700",
    name: "Nachlässe",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3701: OdooAccountTemplate = OdooAccountTemplate {
    code: "3701",
    name: "Nachlässe aus Einkauf Roh-, Hilfsund Betriebsstoffe",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3710: OdooAccountTemplate = OdooAccountTemplate {
    code: "3710",
    name: "Nachlässe 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3714: OdooAccountTemplate = OdooAccountTemplate {
    code: "3714",
    name: "Nachlässe aus Einkauf Roh-, Hilfsund Betriebsstoffe 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3715: OdooAccountTemplate = OdooAccountTemplate {
    code: "3715",
    name: "Nachlässe aus Einkauf Roh-, Hilfsund Betriebsstoffe 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3717: OdooAccountTemplate = OdooAccountTemplate {
    code: "3717",
    name: "Nachlässe aus Einkauf Roh-, Hilfsund Betriebsstoffe, innergemeinschaftlicher Erwerb 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3718: OdooAccountTemplate = OdooAccountTemplate {
    code: "3718",
    name: "Nachlässe aus Einkauf Roh-, Hilfsund Betriebsstoffe, innergemeinschaftlicher Erwerb 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3720: OdooAccountTemplate = OdooAccountTemplate {
    code: "3720",
    name: "Nachlässe 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3724: OdooAccountTemplate = OdooAccountTemplate {
    code: "3724",
    name: "Nachlässe aus innergemeinschaftlichem Erwerb 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3725: OdooAccountTemplate = OdooAccountTemplate {
    code: "3725",
    name: "Nachlässe aus innergemeinschaftlichem Erwerb 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3730: OdooAccountTemplate = OdooAccountTemplate {
    code: "3730",
    name: "Erhaltene Skonti",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3731: OdooAccountTemplate = OdooAccountTemplate {
    code: "3731",
    name: "Erhaltene Skonti 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3733: OdooAccountTemplate = OdooAccountTemplate {
    code: "3733",
    name: "Erhaltene Skonti aus Einkauf Roh-, Hilfs- und Betriebsstoffe",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3734: OdooAccountTemplate = OdooAccountTemplate {
    code: "3734",
    name: "Erhaltene Skonti aus Einkauf Roh-, Hilfs- und Betriebsstoffe 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3736: OdooAccountTemplate = OdooAccountTemplate {
    code: "3736",
    name: "Erhaltene Skonti 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3738: OdooAccountTemplate = OdooAccountTemplate {
    code: "3738",
    name: "Erhaltene Skonti aus Einkauf Roh-, Hilfs- und Betriebsstoffe 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3741: OdooAccountTemplate = OdooAccountTemplate {
    code: "3741",
    name: "Erhaltene Skonti aus Einkauf Roh-, Hilfs- und Betriebsstoffe aus steuerpflichtigem innergemeinschaftlichem Erwerb 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3743: OdooAccountTemplate = OdooAccountTemplate {
    code: "3743",
    name: "Erhaltene Skonti aus Einkauf Roh-, Hilfs- und Betriebsstoffe aus steuerpflichtigem innergemeinschaftlichem Erwerb 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3744: OdooAccountTemplate = OdooAccountTemplate {
    code: "3744",
    name: "Erhaltene Skonti aus Einkauf Roh-, Hilfs- und Betriebsstoffe aus steuerpflichtigem innergemeinschaftlichem Erwerb",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3745: OdooAccountTemplate = OdooAccountTemplate {
    code: "3745",
    name: "Erhaltene Skonti aus steuerpflichtigem innergemeinschaftlichem Erwerb",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3746: OdooAccountTemplate = OdooAccountTemplate {
    code: "3746",
    name: "Erhaltene Skonti aus steuerpflichtigem innergemeinschaftlichem Erwerb 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3748: OdooAccountTemplate = OdooAccountTemplate {
    code: "3748",
    name: "Erhaltene Skonti aus steuerpflichtigem innergemeinschaftlichem Erwerb 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3750: OdooAccountTemplate = OdooAccountTemplate {
    code: "3750",
    name: "Erhaltene Boni 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3753: OdooAccountTemplate = OdooAccountTemplate {
    code: "3753",
    name: "Erhaltene Boni aus Einkauf Roh-, Hilfs- und Betriebsstoffe",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3754: OdooAccountTemplate = OdooAccountTemplate {
    code: "3754",
    name: "Erhaltene Boni aus Einkauf Roh-, Hilfs- und Betriebsstoffe 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3755: OdooAccountTemplate = OdooAccountTemplate {
    code: "3755",
    name: "Erhaltene Boni aus Einkauf Roh-, Hilfs- und Betriebsstoffe 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3760: OdooAccountTemplate = OdooAccountTemplate {
    code: "3760",
    name: "Erhaltene Boni 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3769: OdooAccountTemplate = OdooAccountTemplate {
    code: "3769",
    name: "Erhaltene Boni",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3770: OdooAccountTemplate = OdooAccountTemplate {
    code: "3770",
    name: "Erhaltene Rabatte",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3780: OdooAccountTemplate = OdooAccountTemplate {
    code: "3780",
    name: "Erhaltene Rabatte 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3783: OdooAccountTemplate = OdooAccountTemplate {
    code: "3783",
    name: "Erhaltene Rabatte aus Einkauf Roh-, Hilfs- und Betriebsstoffe",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3784: OdooAccountTemplate = OdooAccountTemplate {
    code: "3784",
    name: "Erhaltene Rabatte aus Einkauf Roh-, Hilfs- und Betriebsstoffe 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3785: OdooAccountTemplate = OdooAccountTemplate {
    code: "3785",
    name: "Erhaltene Rabatte aus Einkauf Roh-, Hilfs- und Betriebsstoffe 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3788: OdooAccountTemplate = OdooAccountTemplate {
    code: "3788",
    name: "Erhaltene Skonti aus Einkauf Roh-, Hilfs- und Betriebsstoffe 10,7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3790: OdooAccountTemplate = OdooAccountTemplate {
    code: "3790",
    name: "Erhaltene Rabatte 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3792: OdooAccountTemplate = OdooAccountTemplate {
    code: "3792",
    name: "Erhaltene Skonti aus Erwerb Roh-, Hilfs- und Betriebsstoffe als letzter Abnehmer innerhalb Dreiecksgeschäft 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3793: OdooAccountTemplate = OdooAccountTemplate {
    code: "3793",
    name: "Erhaltene Skonti aus Erwerb Waren als letzter Abnehmer innerhalb Dreiecksgeschäft 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3794: OdooAccountTemplate = OdooAccountTemplate {
    code: "3794",
    name: "Erhaltene Skonti 5,5 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3796: OdooAccountTemplate = OdooAccountTemplate {
    code: "3796",
    name: "Erhaltene Skonti 10,7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3798: OdooAccountTemplate = OdooAccountTemplate {
    code: "3798",
    name: "Erhaltene Skonti aus Einkauf Roh-, Hilfs- und Betriebsstoffe 5,5 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3800: OdooAccountTemplate = OdooAccountTemplate {
    code: "3800",
    name: "Bezugsnebenkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3830: OdooAccountTemplate = OdooAccountTemplate {
    code: "3830",
    name: "Leergut",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3850: OdooAccountTemplate = OdooAccountTemplate {
    code: "3850",
    name: "Zölle und Einfuhrabgaben",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3950: OdooAccountTemplate = OdooAccountTemplate {
    code: "3950",
    name: "Bestandsveränderungen Waren",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3955: OdooAccountTemplate = OdooAccountTemplate {
    code: "3955",
    name: "Bestandsveränderungen Roh-, Hilfs- und Betriebsstoffe",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3960: OdooAccountTemplate = OdooAccountTemplate {
    code: "3960",
    name: "Bestandsveränderungen Roh-, Hilfs- und Betriebsstoffe sowie bezogene Waren",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_3970: OdooAccountTemplate = OdooAccountTemplate {
    code: "3970",
    name: "Roh-, Hilfs- und Betriebsstoffe (Bestand)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_3980: OdooAccountTemplate = OdooAccountTemplate {
    code: "3980",
    name: "Waren (Bestand)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_4100: OdooAccountTemplate = OdooAccountTemplate {
    code: "4100",
    name: "Löhne und Gehälter",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4110: OdooAccountTemplate = OdooAccountTemplate {
    code: "4110",
    name: "Löhne",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4120: OdooAccountTemplate = OdooAccountTemplate {
    code: "4120",
    name: "Gehälter",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4124: OdooAccountTemplate = OdooAccountTemplate {
    code: "4124",
    name: "Geschäftsführergehälter der GmbH-Gesellschafter",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4125: OdooAccountTemplate = OdooAccountTemplate {
    code: "4125",
    name: "Ehegattengehalt",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4126: OdooAccountTemplate = OdooAccountTemplate {
    code: "4126",
    name: "Tantiemen Gesellschafter-Geschäftsführer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4127: OdooAccountTemplate = OdooAccountTemplate {
    code: "4127",
    name: "Geschäftsführergehälter",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4128: OdooAccountTemplate = OdooAccountTemplate {
    code: "4128",
    name: "Vergütungen an angestellte Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4129: OdooAccountTemplate = OdooAccountTemplate {
    code: "4129",
    name: "Tantiemen Arbeitnehmer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4130: OdooAccountTemplate = OdooAccountTemplate {
    code: "4130",
    name: "Gesetzliche soziale Aufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4137: OdooAccountTemplate = OdooAccountTemplate {
    code: "4137",
    name: "Gesetzliche soziale Aufwendungen für Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4138: OdooAccountTemplate = OdooAccountTemplate {
    code: "4138",
    name: "Beiträge zur Berufsgenossenschaft",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4139: OdooAccountTemplate = OdooAccountTemplate {
    code: "4139",
    name: "Ausgleichsabgabe nach dem Schwerbehindertengesetz",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4140: OdooAccountTemplate = OdooAccountTemplate {
    code: "4140",
    name: "Freiwillige soziale Aufwendungen, lohnsteuerfrei",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4141: OdooAccountTemplate = OdooAccountTemplate {
    code: "4141",
    name: "Sonstige soziale Abgaben",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4144: OdooAccountTemplate = OdooAccountTemplate {
    code: "4144",
    name: "Soziale Abgaben für Minijobber",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4145: OdooAccountTemplate = OdooAccountTemplate {
    code: "4145",
    name: "Freiwillige soziale Aufwendungen, lohnsteuerpflichtig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4146: OdooAccountTemplate = OdooAccountTemplate {
    code: "4146",
    name: "Freiwillige Zuwendungen an Minijobber",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4147: OdooAccountTemplate = OdooAccountTemplate {
    code: "4147",
    name: "Freiwillige Zuwendungen an Gesellschafter-Geschäftsführer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4148: OdooAccountTemplate = OdooAccountTemplate {
    code: "4148",
    name: "Freiwillige Zuwendungen an angestellte Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4149: OdooAccountTemplate = OdooAccountTemplate {
    code: "4149",
    name: "Pauschale Steuer auf sonstige Bezüge (z. B. Fahrtkostenzuschüsse)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4150: OdooAccountTemplate = OdooAccountTemplate {
    code: "4150",
    name: "Krankengeldzuschüsse",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4151: OdooAccountTemplate = OdooAccountTemplate {
    code: "4151",
    name: "Sachzuwendungen und Dienstleistungen an Minijobber",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4152: OdooAccountTemplate = OdooAccountTemplate {
    code: "4152",
    name: "Sachzuwendungen und Dienstleistungen an Arbeitnehmer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4153: OdooAccountTemplate = OdooAccountTemplate {
    code: "4153",
    name: "Sachzuwendungen und Dienstleistungen an Gesellschafter-Geschäftsführer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4154: OdooAccountTemplate = OdooAccountTemplate {
    code: "4154",
    name: "Sachzuwendungen und Dienstleistungen an angestellte Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4155: OdooAccountTemplate = OdooAccountTemplate {
    code: "4155",
    name: "Zuschüsse der Agenturen für Arbeit (Haben)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4156: OdooAccountTemplate = OdooAccountTemplate {
    code: "4156",
    name: "Aufwendungen aus der Veränderung von Urlaubsrückstellungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4157: OdooAccountTemplate = OdooAccountTemplate {
    code: "4157",
    name: "Aufwendungen aus der Veränderung von Urlaubsrückstellungen für Gesellschafter-Geschäftsführer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4158: OdooAccountTemplate = OdooAccountTemplate {
    code: "4158",
    name: "Aufwendungen aus der Veränderung von Urlaubsrückstellungen für angestellte Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4159: OdooAccountTemplate = OdooAccountTemplate {
    code: "4159",
    name: "Aufwendungen aus der Veränderung von Urlaubsrückstellungen für Minijobber",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4160: OdooAccountTemplate = OdooAccountTemplate {
    code: "4160",
    name: "Versorgungskassen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4165: OdooAccountTemplate = OdooAccountTemplate {
    code: "4165",
    name: "Aufwendungen für Altersversorgung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4166: OdooAccountTemplate = OdooAccountTemplate {
    code: "4166",
    name: "Aufwendungen für Altersversorgung für Gesellschafter-Geschäftsführer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4167: OdooAccountTemplate = OdooAccountTemplate {
    code: "4167",
    name: "Pauschale Steuer auf sonstige Bezüge (z. B. Direktversicherungen)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4168: OdooAccountTemplate = OdooAccountTemplate {
    code: "4168",
    name: "Aufwendungen für Altersversorgung für Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4169: OdooAccountTemplate = OdooAccountTemplate {
    code: "4169",
    name: "Aufwendungen für Unterstützung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4170: OdooAccountTemplate = OdooAccountTemplate {
    code: "4170",
    name: "Vermögenswirksame Leistungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4175: OdooAccountTemplate = OdooAccountTemplate {
    code: "4175",
    name: "Fahrtkostenerstattung - Wohnung/ Arbeitsstätte",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4180: OdooAccountTemplate = OdooAccountTemplate {
    code: "4180",
    name: "Bedienungsgelder",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4190: OdooAccountTemplate = OdooAccountTemplate {
    code: "4190",
    name: "Aushilfslöhne",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4194: OdooAccountTemplate = OdooAccountTemplate {
    code: "4194",
    name: "Pauschale Steuer für Minijobber",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4195: OdooAccountTemplate = OdooAccountTemplate {
    code: "4195",
    name: "Löhne für Minijobs",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4196: OdooAccountTemplate = OdooAccountTemplate {
    code: "4196",
    name: "Pauschale Steuer für GesellschafterGeschäftsführer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4197: OdooAccountTemplate = OdooAccountTemplate {
    code: "4197",
    name: "Pauschale Steuer für angestellte Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4198: OdooAccountTemplate = OdooAccountTemplate {
    code: "4198",
    name: "Pauschale Steuer für Arbeitnehmer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4199: OdooAccountTemplate = OdooAccountTemplate {
    code: "4199",
    name: "Pauschale Steuer für Aushilfen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4200: OdooAccountTemplate = OdooAccountTemplate {
    code: "4200",
    name: "Raumkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4210: OdooAccountTemplate = OdooAccountTemplate {
    code: "4210",
    name: "Miete (unbewegliche Wirtschaftsgüter)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4211: OdooAccountTemplate = OdooAccountTemplate {
    code: "4211",
    name: "Aufwendungen für gemietete oder gepachtete unbewegliche Wirtschaftsgüter, die gewerbesteuerlich hinzuzurechnen sind",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4212: OdooAccountTemplate = OdooAccountTemplate {
    code: "4212",
    name: "Miete/Aufwendungen für doppelte Haushaltsführung Unternehmer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4215: OdooAccountTemplate = OdooAccountTemplate {
    code: "4215",
    name: "Leasing (unbewegliche Wirtschaftsgüter)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4219: OdooAccountTemplate = OdooAccountTemplate {
    code: "4219",
    name: "Vergütungen an Mitunternehmer für die mietweise Überlassung ihrer unbeweglichen Wirtschaftsgüter § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4220: OdooAccountTemplate = OdooAccountTemplate {
    code: "4220",
    name: "Pacht (unbewegliche Wirtschaftsgüter)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4222: OdooAccountTemplate = OdooAccountTemplate {
    code: "4222",
    name: "Vergütungen an Gesellschafter für die miet- oder pachtweise Überlassung ihrer unbeweglichen Wirtschaftsgüter",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4228: OdooAccountTemplate = OdooAccountTemplate {
    code: "4228",
    name: "Miet- und Pachtnebenkosten, die gewerbesteuerlich nicht hinzuzurechnen sind",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4229: OdooAccountTemplate = OdooAccountTemplate {
    code: "4229",
    name: "Vergütungen an Mitunternehmer für die pachtweise Überlassung ihrer unbeweglichen Wirtschaftsgüter § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4230: OdooAccountTemplate = OdooAccountTemplate {
    code: "4230",
    name: "Heizung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4240: OdooAccountTemplate = OdooAccountTemplate {
    code: "4240",
    name: "Gas, Strom, Wasser",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4250: OdooAccountTemplate = OdooAccountTemplate {
    code: "4250",
    name: "Reinigung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4260: OdooAccountTemplate = OdooAccountTemplate {
    code: "4260",
    name: "Instandhaltung betrieblicher Räume",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4270: OdooAccountTemplate = OdooAccountTemplate {
    code: "4270",
    name: "Abgaben für betrieblich genutzten Grundbesitz",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4280: OdooAccountTemplate = OdooAccountTemplate {
    code: "4280",
    name: "Sonstige Raumkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4288: OdooAccountTemplate = OdooAccountTemplate {
    code: "4288",
    name: "Aufwendungen für ein häusliches Arbeitszimmer (abziehbarer Anteil)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4289: OdooAccountTemplate = OdooAccountTemplate {
    code: "4289",
    name: "Aufwendungen für ein häusliches Arbeitszimmer (nicht abziehbarer Anteil)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4290: OdooAccountTemplate = OdooAccountTemplate {
    code: "4290",
    name: "Grundstücksaufwendungen betrieblich",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4300: OdooAccountTemplate = OdooAccountTemplate {
    code: "4300",
    name: "Nicht abziehbare Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4301: OdooAccountTemplate = OdooAccountTemplate {
    code: "4301",
    name: "Nicht abziehbare Vorsteuer 7 %",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4306: OdooAccountTemplate = OdooAccountTemplate {
    code: "4306",
    name: "Nicht abziehbare Vorsteuer 19 %",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4320: OdooAccountTemplate = OdooAccountTemplate {
    code: "4320",
    name: "Gewerbesteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4340: OdooAccountTemplate = OdooAccountTemplate {
    code: "4340",
    name: "Sonstige Betriebssteuern",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4350: OdooAccountTemplate = OdooAccountTemplate {
    code: "4350",
    name: "Verbrauchsteuer (sonstige Steuern)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4355: OdooAccountTemplate = OdooAccountTemplate {
    code: "4355",
    name: "Ökosteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4360: OdooAccountTemplate = OdooAccountTemplate {
    code: "4360",
    name: "Versicherungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4366: OdooAccountTemplate = OdooAccountTemplate {
    code: "4366",
    name: "Versicherungen für Gebäude",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4370: OdooAccountTemplate = OdooAccountTemplate {
    code: "4370",
    name: "Netto-Prämie für Rückdeckung künftiger Versorgungsleistungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4380: OdooAccountTemplate = OdooAccountTemplate {
    code: "4380",
    name: "Beiträge",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4390: OdooAccountTemplate = OdooAccountTemplate {
    code: "4390",
    name: "Sonstige Abgaben",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4396: OdooAccountTemplate = OdooAccountTemplate {
    code: "4396",
    name: "Steuerlich abzugsfähige Verspätungszuschläge und Zwangsgelder",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4397: OdooAccountTemplate = OdooAccountTemplate {
    code: "4397",
    name: "Steuerlich nicht abzugsfähige Verspätungszuschläge und Zwangsgelder",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4500: OdooAccountTemplate = OdooAccountTemplate {
    code: "4500",
    name: "Fahrzeugkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4510: OdooAccountTemplate = OdooAccountTemplate {
    code: "4510",
    name: "Kfz-Steuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4520: OdooAccountTemplate = OdooAccountTemplate {
    code: "4520",
    name: "Kfz-Versicherungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4530: OdooAccountTemplate = OdooAccountTemplate {
    code: "4530",
    name: "Laufende Kfz-Betriebskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4540: OdooAccountTemplate = OdooAccountTemplate {
    code: "4540",
    name: "Kfz-Reparaturen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4550: OdooAccountTemplate = OdooAccountTemplate {
    code: "4550",
    name: "Garagenmiete",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4560: OdooAccountTemplate = OdooAccountTemplate {
    code: "4560",
    name: "Mautgebühren",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4570: OdooAccountTemplate = OdooAccountTemplate {
    code: "4570",
    name: "Mietleasing Kfz",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4580: OdooAccountTemplate = OdooAccountTemplate {
    code: "4580",
    name: "Sonstige Kfz-Kosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4590: OdooAccountTemplate = OdooAccountTemplate {
    code: "4590",
    name: "Kfz-Kosten für betrieblich genutzte zum Privatvermögen gehörende Kraftfahrzeuge",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4595: OdooAccountTemplate = OdooAccountTemplate {
    code: "4595",
    name: "Fremdfahrzeugkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4600: OdooAccountTemplate = OdooAccountTemplate {
    code: "4600",
    name: "Werbekosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4605: OdooAccountTemplate = OdooAccountTemplate {
    code: "4605",
    name: "Streuartikel",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4630: OdooAccountTemplate = OdooAccountTemplate {
    code: "4630",
    name: "Geschenke abzugsfähig ohne § 37b EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4631: OdooAccountTemplate = OdooAccountTemplate {
    code: "4631",
    name: "Geschenke abzugsfähig mit § 37b EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4632: OdooAccountTemplate = OdooAccountTemplate {
    code: "4632",
    name: "Pauschale Steuer für Geschenke und Zuwendungen abzugsfähig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4633: OdooAccountTemplate = OdooAccountTemplate {
    code: "4633",
    name: "Pauschalsteuern für Geschenke und Zuwendungen abzugsfähig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4635: OdooAccountTemplate = OdooAccountTemplate {
    code: "4635",
    name: "Geschenke nicht abzugsfähig ohne § 37b EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4636: OdooAccountTemplate = OdooAccountTemplate {
    code: "4636",
    name: "Geschenke nicht abzugsfähig mit § 37b EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4637: OdooAccountTemplate = OdooAccountTemplate {
    code: "4637",
    name: "Pauschale Steuer für Geschenke und Zuwendungen nicht abzugsfähig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4638: OdooAccountTemplate = OdooAccountTemplate {
    code: "4638",
    name: "Geschenke ausschließlich betrieblich genutzt",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4639: OdooAccountTemplate = OdooAccountTemplate {
    code: "4639",
    name: "Zugaben mit § 37b EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4640: OdooAccountTemplate = OdooAccountTemplate {
    code: "4640",
    name: "Repräsentationskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4650: OdooAccountTemplate = OdooAccountTemplate {
    code: "4650",
    name: "Bewirtungskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4651: OdooAccountTemplate = OdooAccountTemplate {
    code: "4651",
    name: "Sonstige eingeschränkt abziehbare Betriebsausgaben (abziehbarer Anteil)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4652: OdooAccountTemplate = OdooAccountTemplate {
    code: "4652",
    name: "Sonstige eingeschränkt abziehbare Betriebsausgaben (nicht abziehbarer Anteil)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4653: OdooAccountTemplate = OdooAccountTemplate {
    code: "4653",
    name: "Aufmerksamkeiten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4654: OdooAccountTemplate = OdooAccountTemplate {
    code: "4654",
    name: "Nicht abzugsfähige Bewirtungskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4655: OdooAccountTemplate = OdooAccountTemplate {
    code: "4655",
    name: "Nicht abzugsfähige Betriebsausgaben aus Werbe- und Repräsentationskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4660: OdooAccountTemplate = OdooAccountTemplate {
    code: "4660",
    name: "Reisekosten Arbeitnehmer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4663: OdooAccountTemplate = OdooAccountTemplate {
    code: "4663",
    name: "Reisekosten Arbeitnehmer Fahrtkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4664: OdooAccountTemplate = OdooAccountTemplate {
    code: "4664",
    name: "Reisekosten Arbeitnehmer Verpflegungsmehraufwand",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4666: OdooAccountTemplate = OdooAccountTemplate {
    code: "4666",
    name: "Reisekosten Arbeitnehmer Übernachtungsaufwand",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4668: OdooAccountTemplate = OdooAccountTemplate {
    code: "4668",
    name: "Kilometergelderstattung Arbeitnehmer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4670: OdooAccountTemplate = OdooAccountTemplate {
    code: "4670",
    name: "Reisekosten Unternehmer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4672: OdooAccountTemplate = OdooAccountTemplate {
    code: "4672",
    name: "Reisekosten Unternehmer (nicht abziehbarer Anteil)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4673: OdooAccountTemplate = OdooAccountTemplate {
    code: "4673",
    name: "Reisekosten Unternehmer Fahrtkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4674: OdooAccountTemplate = OdooAccountTemplate {
    code: "4674",
    name: "Reisekosten Unternehmer Verpflegungsmehraufwand",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4676: OdooAccountTemplate = OdooAccountTemplate {
    code: "4676",
    name: "Reisekosten Unternehmer Übernachtungsaufwand und Reisenebenkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4678: OdooAccountTemplate = OdooAccountTemplate {
    code: "4678",
    name: "Fahrten zwischen Wohnung und Betriebsstätte und Familienheimfahrten (abziehbarer Anteil)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4679: OdooAccountTemplate = OdooAccountTemplate {
    code: "4679",
    name: "Fahrten zwischen Wohnung und Betriebsstätte und Familienheimfahrten (nicht abziehbarer Anteil)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4680: OdooAccountTemplate = OdooAccountTemplate {
    code: "4680",
    name: "Fahrten zwischen Wohnung und Betriebsstätte und Familienheimfahrten (Haben)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4681: OdooAccountTemplate = OdooAccountTemplate {
    code: "4681",
    name: "Verpflegungsmehraufwendungen im Rahmen der doppelten Haushaltsführung Unternehmer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4700: OdooAccountTemplate = OdooAccountTemplate {
    code: "4700",
    name: "Kosten der Warenabgabe",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4710: OdooAccountTemplate = OdooAccountTemplate {
    code: "4710",
    name: "Verpackungsmaterial",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4730: OdooAccountTemplate = OdooAccountTemplate {
    code: "4730",
    name: "Ausgangsfrachten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4750: OdooAccountTemplate = OdooAccountTemplate {
    code: "4750",
    name: "Transportversicherungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4760: OdooAccountTemplate = OdooAccountTemplate {
    code: "4760",
    name: "Verkaufsprovisionen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4780: OdooAccountTemplate = OdooAccountTemplate {
    code: "4780",
    name: "Fremdarbeiten (Vertrieb)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4790: OdooAccountTemplate = OdooAccountTemplate {
    code: "4790",
    name: "Aufwand für Gewährleistungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_6"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4800: OdooAccountTemplate = OdooAccountTemplate {
    code: "4800",
    name: "Reparaturen und Instandhaltungen von technischen Anlagen und Maschinen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4801: OdooAccountTemplate = OdooAccountTemplate {
    code: "4801",
    name: "Reparaturen und Instandhaltung von Bauten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4805: OdooAccountTemplate = OdooAccountTemplate {
    code: "4805",
    name: "Reparaturen und Instandhaltungen von anderen Anlagen und Betriebsund Geschäftsausstattung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4806: OdooAccountTemplate = OdooAccountTemplate {
    code: "4806",
    name: "Wartungskosten für Hard- und Software",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4808: OdooAccountTemplate = OdooAccountTemplate {
    code: "4808",
    name: "Zuführung zu Aufwandsrückstellungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4809: OdooAccountTemplate = OdooAccountTemplate {
    code: "4809",
    name: "Sonstige Reparaturen und Instandhaltungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4810: OdooAccountTemplate = OdooAccountTemplate {
    code: "4810",
    name: "Operating leases movable assets for technical equipment and machinery",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4815: OdooAccountTemplate = OdooAccountTemplate {
    code: "4815",
    name: "Kaufleasing",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4820: OdooAccountTemplate = OdooAccountTemplate {
    code: "4820",
    name: "Abschreibung für Inbetriebnahme, Erweiterung",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4822: OdooAccountTemplate = OdooAccountTemplate {
    code: "4822",
    name: "Abschreibungen auf immaterielle Vermögensgegenstände",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4823: OdooAccountTemplate = OdooAccountTemplate {
    code: "4823",
    name: "Abschreibungen auf selbst geschaffene immaterielle Vermögensgegenstände",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4824: OdooAccountTemplate = OdooAccountTemplate {
    code: "4824",
    name: "Abschreibungen auf den Geschäftsoder Firmenwert",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4825: OdooAccountTemplate = OdooAccountTemplate {
    code: "4825",
    name: "Außerplanmäßige Abschreibungen auf den Geschäfts- oder Firmenwert",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4826: OdooAccountTemplate = OdooAccountTemplate {
    code: "4826",
    name: "Außerplanmäßige Abschreibungenauf immaterielle Vermögensgegenstände",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4827: OdooAccountTemplate = OdooAccountTemplate {
    code: "4827",
    name: "Außerplanmäßige Abschreibungen auf selbst geschaffene immaterielle Vermögensgegenstände",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4830: OdooAccountTemplate = OdooAccountTemplate {
    code: "4830",
    name: "Abschreibungen auf Sachanlagen (ohne AfA auf Kfz und Gebäude)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4831: OdooAccountTemplate = OdooAccountTemplate {
    code: "4831",
    name: "Abschreibungen auf Gebäude",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4832: OdooAccountTemplate = OdooAccountTemplate {
    code: "4832",
    name: "Abschreibungen auf Kfz",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4833: OdooAccountTemplate = OdooAccountTemplate {
    code: "4833",
    name: "Abschreibungen auf Gebäudeanteil des häuslichen Arbeitszimmers",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4840: OdooAccountTemplate = OdooAccountTemplate {
    code: "4840",
    name: "Außerplanmäßige Abschreibungen auf Sachanlagen",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4841: OdooAccountTemplate = OdooAccountTemplate {
    code: "4841",
    name: "Absetzung für außergewöhnliche technische und wirtschaftliche Abnutzung der Gebäude",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4842: OdooAccountTemplate = OdooAccountTemplate {
    code: "4842",
    name: "Absetzung für außergewöhnliche technische und wirtschaftliche Abnutzung des Kfz",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4843: OdooAccountTemplate = OdooAccountTemplate {
    code: "4843",
    name: "Absetzung für außergewöhnliche technische und wirtschaftliche Abnutzung sonstiger Wirtschaftsgüter",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4850: OdooAccountTemplate = OdooAccountTemplate {
    code: "4850",
    name: "Abschreibungen auf Sachanlagen auf Grund steuerlicher Sondervorschriften",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4851: OdooAccountTemplate = OdooAccountTemplate {
    code: "4851",
    name: "Sonderabschreibungen nach § 7g Abs. 5 EStG (ohne Kfz)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4852: OdooAccountTemplate = OdooAccountTemplate {
    code: "4852",
    name: "Sonderabschreibungen nach § 7g Abs. 5 EStG (für Kfz)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4853: OdooAccountTemplate = OdooAccountTemplate {
    code: "4853",
    name: "Kürzung der Anschaffungs- oder Herstellungskosten nach § 7g Abs. 2 EStG (ohne Kfz)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4854: OdooAccountTemplate = OdooAccountTemplate {
    code: "4854",
    name: "Kürzung der Anschaffungs- oder Herstellungskosten nach § 7g Abs. 2 EStG (für Kfz)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4855: OdooAccountTemplate = OdooAccountTemplate {
    code: "4855",
    name: "Sofortabschreibung geringwertiger Wirtschaftsgüter",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4860: OdooAccountTemplate = OdooAccountTemplate {
    code: "4860",
    name: "Abschreibungen auf aktivierte, geringwertige Wirtschaftsgüter",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4862: OdooAccountTemplate = OdooAccountTemplate {
    code: "4862",
    name: "Abschreibungen auf den Sammelposten Wirtschaftsgüter",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4865: OdooAccountTemplate = OdooAccountTemplate {
    code: "4865",
    name: "Außerplanmäßige Abschreibungen auf aktivierte, geringwertige Wirtschaftsgüter",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4866: OdooAccountTemplate = OdooAccountTemplate {
    code: "4866",
    name: "Abschreibungen auf Finanzanlagen (nicht dauerhaft)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4870: OdooAccountTemplate = OdooAccountTemplate {
    code: "4870",
    name: "Abschreibungen auf Finanzanlagen (dauerhaft)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4871: OdooAccountTemplate = OdooAccountTemplate {
    code: "4871",
    name: "Abschreibungen auf Finanzanlagen § 3 Nr. 40 EStG bzw. § 8b Abs. 3 KStG (dauerhaft)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4872: OdooAccountTemplate = OdooAccountTemplate {
    code: "4872",
    name: "Aufwendungen auf Grund von Verlustanteilen an gewerblichen und selbständigen Mitunternehmerschaften, § 8 GewStG bzw. § 18 EStG",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4873: OdooAccountTemplate = OdooAccountTemplate {
    code: "4873",
    name: "Abschreibungen auf Finanzanlagen auf Grund § 6b EStG-Rücklage, § 3 Nr. 40 EStG bzw. § 8b Abs. 3 KStG",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4874: OdooAccountTemplate = OdooAccountTemplate {
    code: "4874",
    name: "Abschreibungen auf Finanzanlagen auf Grund § 6b EStG-Rücklage",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4875: OdooAccountTemplate = OdooAccountTemplate {
    code: "4875",
    name: "Abschreibungen auf Wertpapiere des Umlaufvermögens",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4876: OdooAccountTemplate = OdooAccountTemplate {
    code: "4876",
    name: "Abschreibungen auf Wertpapiere des Umlaufvermögens § 3 Nr. 40 EStG bzw. § 8b Abs. 3 KStG",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4877: OdooAccountTemplate = OdooAccountTemplate {
    code: "4877",
    name: "Abschreibungen auf Finanzanlagen - verbundene Unternehmen",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4878: OdooAccountTemplate = OdooAccountTemplate {
    code: "4878",
    name: "Abschreibungen auf Wertpapiere des Umlaufvermögens - verbundene Unternehmen",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4880: OdooAccountTemplate = OdooAccountTemplate {
    code: "4880",
    name: "Abschreibungen auf sonstige Vermögensgegenstände des Umlaufvermögens (soweit unüblich hoch)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4882: OdooAccountTemplate = OdooAccountTemplate {
    code: "4882",
    name: "Abschreibungen auf Umlaufvermögen, steuerrechtlich bedingt (soweit unüblich hoch)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4886: OdooAccountTemplate = OdooAccountTemplate {
    code: "4886",
    name: "Abschreibungen auf Umlaufvermögen außer Vorräte und Wertpapiere des Umlaufvermögens (übliche Höhe)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4887: OdooAccountTemplate = OdooAccountTemplate {
    code: "4887",
    name: "Abschreibungen auf Umlaufvermögen außer Vorräte und Wertpapiere des Umlaufvermögens, steuerrechtlich bedingt (übliche Höhe)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4892: OdooAccountTemplate = OdooAccountTemplate {
    code: "4892",
    name: "Abschreibungen auf Roh-, Hilfsund Betriebsstoffe/Waren (soweit unübliche Höhe)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4893: OdooAccountTemplate = OdooAccountTemplate {
    code: "4893",
    name: "Abschreibungen auf fertige und unfertige Erzeugnisse (soweit unübliche Höhe)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4900: OdooAccountTemplate = OdooAccountTemplate {
    code: "4900",
    name: "Sonstige betriebliche Aufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4902: OdooAccountTemplate = OdooAccountTemplate {
    code: "4902",
    name: "Interimskonto für Aufwendungen in einem anderen Land, bei denen eine Vorsteuervergütung möglich ist",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4905: OdooAccountTemplate = OdooAccountTemplate {
    code: "4905",
    name: "Sonstige Aufwendungen betrieblich und regelmäßig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4909: OdooAccountTemplate = OdooAccountTemplate {
    code: "4909",
    name: "Fremdleistungen/Fremdarbeiten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4910: OdooAccountTemplate = OdooAccountTemplate {
    code: "4910",
    name: "Porto",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4920: OdooAccountTemplate = OdooAccountTemplate {
    code: "4920",
    name: "Telefon",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4925: OdooAccountTemplate = OdooAccountTemplate {
    code: "4925",
    name: "Telefax und Internetkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4930: OdooAccountTemplate = OdooAccountTemplate {
    code: "4930",
    name: "Bürobedarf",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4940: OdooAccountTemplate = OdooAccountTemplate {
    code: "4940",
    name: "Zeitschriften, Bücher (Fachliteratur)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4945: OdooAccountTemplate = OdooAccountTemplate {
    code: "4945",
    name: "Fortbildungskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4946: OdooAccountTemplate = OdooAccountTemplate {
    code: "4946",
    name: "Freiwillige Sozialleistungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4948: OdooAccountTemplate = OdooAccountTemplate {
    code: "4948",
    name: "Vergütungen an Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4949: OdooAccountTemplate = OdooAccountTemplate {
    code: "4949",
    name: "Haftungsvergütung an Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4950: OdooAccountTemplate = OdooAccountTemplate {
    code: "4950",
    name: "Rechts- und Beratungskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4955: OdooAccountTemplate = OdooAccountTemplate {
    code: "4955",
    name: "Buchführungskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4957: OdooAccountTemplate = OdooAccountTemplate {
    code: "4957",
    name: "Abschluss- und Prüfungskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4958: OdooAccountTemplate = OdooAccountTemplate {
    code: "4958",
    name: "Vergütungen an Gesellschafter für die miet- oder pachtweise Überlassung ihrer beweglichen Wirtschaftsgüter",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4959: OdooAccountTemplate = OdooAccountTemplate {
    code: "4959",
    name: "Vergütungen an Mitunternehmer für die miet- oder pachtweise Überlassung ihrer beweglichen Wirtschaftsgüter § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4960: OdooAccountTemplate = OdooAccountTemplate {
    code: "4960",
    name: "Mieten für Einrichtungen (bewegliche Wirtschaftsgüter)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4961: OdooAccountTemplate = OdooAccountTemplate {
    code: "4961",
    name: "Pacht (bewegliche Wirtschaftsgüter)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4963: OdooAccountTemplate = OdooAccountTemplate {
    code: "4963",
    name: "Aufwendungen für gemietete oder gepachtete bewegliche Wirtschaftsgüter, die gewerbesteuerlich hinzuzurechnen sind",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4964: OdooAccountTemplate = OdooAccountTemplate {
    code: "4964",
    name: "Aufwendungen für die zeitlich befristete Überlassung von Rechten (Lizenzen, Konzessionen)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4965: OdooAccountTemplate = OdooAccountTemplate {
    code: "4965",
    name: "Mietleasing bewegliche Wirtschaftsgüter für Betriebs- und Geschäftsausstattung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4969: OdooAccountTemplate = OdooAccountTemplate {
    code: "4969",
    name: "Aufwendungen für Abraum- und Abfallbeseitigung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4970: OdooAccountTemplate = OdooAccountTemplate {
    code: "4970",
    name: "Nebenkosten des Geldverkehrs",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4975: OdooAccountTemplate = OdooAccountTemplate {
    code: "4975",
    name: "Aufwendungen aus Anteilen an Kapitalgesellschaften §§ 3 Nr. 40 und 3c EStG bzw. § 8b Abs. 1 und 4 KStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4976: OdooAccountTemplate = OdooAccountTemplate {
    code: "4976",
    name: "Veräußerungskosten § 3 Nr. 40 EStG bzw. § 8b Abs. 2 KStG (bei Veräußerungsgewinn)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4980: OdooAccountTemplate = OdooAccountTemplate {
    code: "4980",
    name: "Mietleasing bewegliche Wirtschaftsgüter für technische Anlagen und Maschinen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4984: OdooAccountTemplate = OdooAccountTemplate {
    code: "4984",
    name: "Genossenschaftliche Rückvergütung an Mitglieder",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4985: OdooAccountTemplate = OdooAccountTemplate {
    code: "4985",
    name: "Werkzeuge und Kleingeräte",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4990: OdooAccountTemplate = OdooAccountTemplate {
    code: "4990",
    name: "Kalkulatorischer Unternehmerlohn",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4991: OdooAccountTemplate = OdooAccountTemplate {
    code: "4991",
    name: "Kalkulatorische Miete und Pacht",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4992: OdooAccountTemplate = OdooAccountTemplate {
    code: "4992",
    name: "Kalkulatorische Zinsen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4993: OdooAccountTemplate = OdooAccountTemplate {
    code: "4993",
    name: "Kalkulatorische Abschreibungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4994: OdooAccountTemplate = OdooAccountTemplate {
    code: "4994",
    name: "Kalkulatorische Wagnisse",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4995: OdooAccountTemplate = OdooAccountTemplate {
    code: "4995",
    name: "Kalkulatorischer Lohn für unentgeltliche Mitarbeiter",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4996: OdooAccountTemplate = OdooAccountTemplate {
    code: "4996",
    name: "Herstellungskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4997: OdooAccountTemplate = OdooAccountTemplate {
    code: "4997",
    name: "Verwaltungskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4998: OdooAccountTemplate = OdooAccountTemplate {
    code: "4998",
    name: "Vertriebskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_4999: OdooAccountTemplate = OdooAccountTemplate {
    code: "4999",
    name: "Gegenkonto 4996-4998",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR03_7000: OdooAccountTemplate = OdooAccountTemplate {
    code: "7000",
    name: "Unfertige Erzeugnisse, unfertige Leistungen (Bestand)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_7050: OdooAccountTemplate = OdooAccountTemplate {
    code: "7050",
    name: "Unfertige Erzeugnisse (Bestand)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_7080: OdooAccountTemplate = OdooAccountTemplate {
    code: "7080",
    name: "Unfertige Leistungen (Bestand)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_7090: OdooAccountTemplate = OdooAccountTemplate {
    code: "7090",
    name: "In Ausführung befindliche Bauaufträge",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_7095: OdooAccountTemplate = OdooAccountTemplate {
    code: "7095",
    name: "In Arbeit befindliche Aufträge",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_2"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_7100: OdooAccountTemplate = OdooAccountTemplate {
    code: "7100",
    name: "Fertige Erzeugnisse und Waren (Bestand)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_7110: OdooAccountTemplate = OdooAccountTemplate {
    code: "7110",
    name: "Fertige Erzeugnisse (Bestand)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_7140: OdooAccountTemplate = OdooAccountTemplate {
    code: "7140",
    name: "Waren (Bestand)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_7200: OdooAccountTemplate = OdooAccountTemplate {
    code: "7200",
    name: "Waren (Inventar)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_3"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR03_8100: OdooAccountTemplate = OdooAccountTemplate {
    code: "8100",
    name: "Steuerfreie Umsätze § 4 Nr. 8 ff. UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8105: OdooAccountTemplate = OdooAccountTemplate {
    code: "8105",
    name: "Steuerfreie Umsätze nach § 4 Nr. 12 UStG (Vermietung und Verpachtung)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8110: OdooAccountTemplate = OdooAccountTemplate {
    code: "8110",
    name: "Sonstige steuerfreie Umsätze Inland",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8120: OdooAccountTemplate = OdooAccountTemplate {
    code: "8120",
    name: "Steuerfreie Umsätze nach § 4 Nr. 1a UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8125: OdooAccountTemplate = OdooAccountTemplate {
    code: "8125",
    name: "Steuerfreie innergemeinschaftliche Lieferungen nach § 4 Nr. 1b UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8130: OdooAccountTemplate = OdooAccountTemplate {
    code: "8130",
    name: "Lieferungen des ersten Abnehmers bei innergemeinschaftlichen Dreiecksgeschäften § 25b Abs. 2 UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8135: OdooAccountTemplate = OdooAccountTemplate {
    code: "8135",
    name: "Steuerfreie innergemeinschaftliche Lieferungen von Neufahrzeugen an Abnehmer ohne USt-Id-Nr",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8140: OdooAccountTemplate = OdooAccountTemplate {
    code: "8140",
    name: "Steuerfreie Umsätze Offshore usw.",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8150: OdooAccountTemplate = OdooAccountTemplate {
    code: "8150",
    name: "Sonstige steuerfreie Umsätze (z. B. § 4 Nr. 2 bis 7 UStG)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8160: OdooAccountTemplate = OdooAccountTemplate {
    code: "8160",
    name: "Steuerfreie Umsätze ohne Vorsteuerabzug zum Gesamtumsatz gehörend, § 4 UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8165: OdooAccountTemplate = OdooAccountTemplate {
    code: "8165",
    name: "Steuerfreie Umsätze ohne Vorsteuerabzug zum Gesamtumsatz gehörend",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8190: OdooAccountTemplate = OdooAccountTemplate {
    code: "8190",
    name: "Erlöse, die mit den Durchschnittssätzen des § 24 UStG versteuert werden",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8191: OdooAccountTemplate = OdooAccountTemplate {
    code: "8191",
    name: "Umsatzerlöse nach §§ 25 und 25a UStG 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8193: OdooAccountTemplate = OdooAccountTemplate {
    code: "8193",
    name: "Umsatzerlöse nach §§ 25 und 25a UStG ohne USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8194: OdooAccountTemplate = OdooAccountTemplate {
    code: "8194",
    name: "Umsatzerlöse aus Reiseleistungen § 25 Abs. 2 UStG, steuerfrei",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8195: OdooAccountTemplate = OdooAccountTemplate {
    code: "8195",
    name: "Erlöse als Kleinunternehmer nach § 19 Abs. 1 UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8196: OdooAccountTemplate = OdooAccountTemplate {
    code: "8196",
    name: "Erlöse aus Geldspielautomaten 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8200: OdooAccountTemplate = OdooAccountTemplate {
    code: "8200",
    name: "Erlöse",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8300: OdooAccountTemplate = OdooAccountTemplate {
    code: "8300",
    name: "Erlöse 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8310: OdooAccountTemplate = OdooAccountTemplate {
    code: "8310",
    name: "Erlöse aus im Inland steuerpflichtigen EU-Lieferungen 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8315: OdooAccountTemplate = OdooAccountTemplate {
    code: "8315",
    name: "Erlöse aus im Inland steuerpflichtigen EU-Lieferungen 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8320: OdooAccountTemplate = OdooAccountTemplate {
    code: "8320",
    name: "Erlöse aus im anderen EU-Land steuerpflichtigen Lieferungen, im Inland nicht steuerbar",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8331: OdooAccountTemplate = OdooAccountTemplate {
    code: "8331",
    name: "Erlöse aus im anderen EU-Land steuerpflichtigen elektronischen Dienstleistungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8335: OdooAccountTemplate = OdooAccountTemplate {
    code: "8335",
    name: "Erlöse aus Lieferungen von Mobilfunkgeräten, Tablet-Computern, Spielekonsolen und integrierten Schaltkreisen, für die der Leistungsempfänger die Umsatzsteuer nach § 13b UStG schuldet",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8336: OdooAccountTemplate = OdooAccountTemplate {
    code: "8336",
    name: "Erlöse aus im anderen EU-Land steuerpflichtigen sonstigen Leistungen, für die der Leistungsempfänger die Umsatzsteuer schuldet",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8337: OdooAccountTemplate = OdooAccountTemplate {
    code: "8337",
    name: "Erlöse aus Leistungen, für die der Leistungsempfänger die Umsatzsteuer nach § 13b UStG schuldet",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8338: OdooAccountTemplate = OdooAccountTemplate {
    code: "8338",
    name: "Erlöse aus im Drittland steuerbaren Leistungen, im Inland nicht steuerbare Umsätze",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8339: OdooAccountTemplate = OdooAccountTemplate {
    code: "8339",
    name: "Erlöse aus im anderen EU-Land steuerpflichtigen Lieferungen, im Inland nicht steuerbar",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8400: OdooAccountTemplate = OdooAccountTemplate {
    code: "8400",
    name: "Erlöse 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8410: OdooAccountTemplate = OdooAccountTemplate {
    code: "8410",
    name: "Erlöse 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8499: OdooAccountTemplate = OdooAccountTemplate {
    code: "8499",
    name: "Nebenerlöse (Bezug zu Materialaufwand)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8500: OdooAccountTemplate = OdooAccountTemplate {
    code: "8500",
    name: "Sonderbetriebseinnahmen, Tätigkeitsvergütung",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8501: OdooAccountTemplate = OdooAccountTemplate {
    code: "8501",
    name: "Sonderbetriebseinnahmen, Miet-/Pachteinnahmen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8502: OdooAccountTemplate = OdooAccountTemplate {
    code: "8502",
    name: "Sonderbetriebseinnahmen, Zinseinnahmen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8503: OdooAccountTemplate = OdooAccountTemplate {
    code: "8503",
    name: "Sonderbetriebseinnahmen, Haftungsvergütung",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8504: OdooAccountTemplate = OdooAccountTemplate {
    code: "8504",
    name: "Sonderbetriebseinnahmen, Pensionszahlungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8505: OdooAccountTemplate = OdooAccountTemplate {
    code: "8505",
    name: "Sonderbetriebseinnahmen, sonstige Sonderbetriebseinnahmen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8510: OdooAccountTemplate = OdooAccountTemplate {
    code: "8510",
    name: "Provisionsumsätze",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8514: OdooAccountTemplate = OdooAccountTemplate {
    code: "8514",
    name: "Provisionsumsätze, steuerfrei § 4 Nr. 8 ff. UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8515: OdooAccountTemplate = OdooAccountTemplate {
    code: "8515",
    name: "Provisionsumsätze, steuerfrei § 4 Nr. 5 UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8516: OdooAccountTemplate = OdooAccountTemplate {
    code: "8516",
    name: "Provisionsumsätze 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8519: OdooAccountTemplate = OdooAccountTemplate {
    code: "8519",
    name: "Provisionsumsätze 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8520: OdooAccountTemplate = OdooAccountTemplate {
    code: "8520",
    name: "Erlöse Abfallverwertung",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8540: OdooAccountTemplate = OdooAccountTemplate {
    code: "8540",
    name: "Erlöse Leergut",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8570: OdooAccountTemplate = OdooAccountTemplate {
    code: "8570",
    name: "Sonstige Erträge aus Provisionen, Lizenzen und Patenten",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8574: OdooAccountTemplate = OdooAccountTemplate {
    code: "8574",
    name: "Sonstige Erträge aus Provisionen, Lizenzen und Patenten, steuerfrei § 4 Nr. 8 ff. UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8575: OdooAccountTemplate = OdooAccountTemplate {
    code: "8575",
    name: "Sonstige Erträge aus Provisionen, Lizenzen und Patenten, steuerfrei § 4 Nr. 5 UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8576: OdooAccountTemplate = OdooAccountTemplate {
    code: "8576",
    name: "Sonstige Erträge aus Provisionen, Lizenzen und Patenten 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8579: OdooAccountTemplate = OdooAccountTemplate {
    code: "8579",
    name: "Sonstige Erträge aus Provisionen, Lizenzen und Patenten 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8589: OdooAccountTemplate = OdooAccountTemplate {
    code: "8589",
    name: "Gegenkonto 8580-8582 bei Aufteilung der Erlöse nach Steuersätzen (EÜR)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8590: OdooAccountTemplate = OdooAccountTemplate {
    code: "8590",
    name: "Verrechnete sonstige Sachbezüge (keine Waren)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8591: OdooAccountTemplate = OdooAccountTemplate {
    code: "8591",
    name: "Sachbezüge 7 % USt (Waren)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8595: OdooAccountTemplate = OdooAccountTemplate {
    code: "8595",
    name: "Sachbezüge 19 % USt (Waren)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8600: OdooAccountTemplate = OdooAccountTemplate {
    code: "8600",
    name: "Sonstige Erlöse betrieblich und regelmäßig",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8603: OdooAccountTemplate = OdooAccountTemplate {
    code: "8603",
    name: "Sonstige betriebliche Erträge",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8604: OdooAccountTemplate = OdooAccountTemplate {
    code: "8604",
    name: "Erstattete Vorsteuer anderer Länder",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8605: OdooAccountTemplate = OdooAccountTemplate {
    code: "8605",
    name: "Sonstige betriebliche und regelmäßige Erträge (neutral)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8606: OdooAccountTemplate = OdooAccountTemplate {
    code: "8606",
    name: "Sonstige betriebliche Erträge von verbundenen Unternehmen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8607: OdooAccountTemplate = OdooAccountTemplate {
    code: "8607",
    name: "Andere Nebenerlöse",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8609: OdooAccountTemplate = OdooAccountTemplate {
    code: "8609",
    name: "Sonstige Erträge betrieblich und regelmäßig, steuerfrei § 4 Nr. 8 ff. UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8610: OdooAccountTemplate = OdooAccountTemplate {
    code: "8610",
    name: "Verrechnete sonstige Sachbezüge",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8611: OdooAccountTemplate = OdooAccountTemplate {
    code: "8611",
    name: "Verrechnete sonstige Sachbezüge aus Kfz-Gestellung 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8613: OdooAccountTemplate = OdooAccountTemplate {
    code: "8613",
    name: "Verrechnete sonstige Sachbezüge 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8614: OdooAccountTemplate = OdooAccountTemplate {
    code: "8614",
    name: "Verrechnete sonstige Sachbezüge ohne Umsatzsteuer",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8625: OdooAccountTemplate = OdooAccountTemplate {
    code: "8625",
    name: "Sonstige betriebliche Erträge, steuerfrei z. B. § 4 Nr. 2 bis 7 UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8630: OdooAccountTemplate = OdooAccountTemplate {
    code: "8630",
    name: "Sonstige Erträge betrieblich und regelmäßig 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8640: OdooAccountTemplate = OdooAccountTemplate {
    code: "8640",
    name: "Sonstige Erträge betrieblich und regelmäßig 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8650: OdooAccountTemplate = OdooAccountTemplate {
    code: "8650",
    name: "Erlöse Zinsen und Diskontspesen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8660: OdooAccountTemplate = OdooAccountTemplate {
    code: "8660",
    name: "Erlöse Zinsen und Diskontspesen aus verbundenen Unternehmen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8700: OdooAccountTemplate = OdooAccountTemplate {
    code: "8700",
    name: "Erlösschmälerungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8701: OdooAccountTemplate = OdooAccountTemplate {
    code: "8701",
    name: "Erlösschmälerungen für steuerfreie Umsätze nach § 4 Nr. 8 ff. UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8702: OdooAccountTemplate = OdooAccountTemplate {
    code: "8702",
    name: "Erlösschmälerungen für steuerfreie Umsätze nach § 4 Nr. 2 bis 7 UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8703: OdooAccountTemplate = OdooAccountTemplate {
    code: "8703",
    name: "Erlösschmälerungen für sonstige steuerfreie Umsätze ohne Vorsteuerabzug",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8704: OdooAccountTemplate = OdooAccountTemplate {
    code: "8704",
    name: "Erlösschmälerungen für sonstige steuerfreie Umsätze mit Vorsteuerabzug",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8705: OdooAccountTemplate = OdooAccountTemplate {
    code: "8705",
    name: "Erlösschmälerungen aus steuerfreien Umsätzen § 4 Nr. 1a UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8710: OdooAccountTemplate = OdooAccountTemplate {
    code: "8710",
    name: "Erlösschmälerungen 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8720: OdooAccountTemplate = OdooAccountTemplate {
    code: "8720",
    name: "Erlösschmälerungen 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8724: OdooAccountTemplate = OdooAccountTemplate {
    code: "8724",
    name: "Erlösschmälerungen aus steuerfreien innergemeinschaftlichen Lieferungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8725: OdooAccountTemplate = OdooAccountTemplate {
    code: "8725",
    name: "Erlösschmälerungen aus im Inlandsteuerpflichtigen EU-Lieferungen 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8726: OdooAccountTemplate = OdooAccountTemplate {
    code: "8726",
    name: "Erlösschmälerungen aus im Inland steuerpflichtigen EU-Lieferungen 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8727: OdooAccountTemplate = OdooAccountTemplate {
    code: "8727",
    name: "Erlösschmälerungen aus im anderen EU-Land steuerpflichtigen Lieferungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8730: OdooAccountTemplate = OdooAccountTemplate {
    code: "8730",
    name: "Gewährte Skonti",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8731: OdooAccountTemplate = OdooAccountTemplate {
    code: "8731",
    name: "Gewährte Skonti 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8736: OdooAccountTemplate = OdooAccountTemplate {
    code: "8736",
    name: "Gewährte Skonti 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8738: OdooAccountTemplate = OdooAccountTemplate {
    code: "8738",
    name: "Gewährte Skonti aus Lieferungen von Mobilfunkgeräten etc., für die der Leistungsempfänger die Umsatzsteuer nach § 13b Abs. 2 Nr. 10 UStG schuldet",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8741: OdooAccountTemplate = OdooAccountTemplate {
    code: "8741",
    name: "Gewährte Skonti aus Leistungen, für die der Leistungsempfänger die Umsatzsteuer nach § 13b UStG schuldet",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8742: OdooAccountTemplate = OdooAccountTemplate {
    code: "8742",
    name: "Gewährte Skonti aus Erlösen aus im anderen EU-Land steuerpflichtigen sonstigen Leistungen, für die der Leistungsempfänger die Umsatzsteuer schuldet",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8743: OdooAccountTemplate = OdooAccountTemplate {
    code: "8743",
    name: "Gewährte Skonti aus steuerfreien innergemeinschaftlichen Lieferungen § 4 Nr. 1b UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8745: OdooAccountTemplate = OdooAccountTemplate {
    code: "8745",
    name: "Gewährte Skonti aus im Inland steuerpflichtigen EU-Lieferungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8746: OdooAccountTemplate = OdooAccountTemplate {
    code: "8746",
    name: "Gewährte Skonti aus im Inland steuerpflichtigen EU-Lieferungen 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8748: OdooAccountTemplate = OdooAccountTemplate {
    code: "8748",
    name: "Gewährte Skonti aus im Inland steuerpflichtigen EU-Lieferungen 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8750: OdooAccountTemplate = OdooAccountTemplate {
    code: "8750",
    name: "Gewährte Boni 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8760: OdooAccountTemplate = OdooAccountTemplate {
    code: "8760",
    name: "Gewährte Boni 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8769: OdooAccountTemplate = OdooAccountTemplate {
    code: "8769",
    name: "Gewährte Boni",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8770: OdooAccountTemplate = OdooAccountTemplate {
    code: "8770",
    name: "Gewährte Rabatte",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8780: OdooAccountTemplate = OdooAccountTemplate {
    code: "8780",
    name: "Gewährte Rabatte 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8790: OdooAccountTemplate = OdooAccountTemplate {
    code: "8790",
    name: "Gewährte Rabatte 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8800: OdooAccountTemplate = OdooAccountTemplate {
    code: "8800",
    name: "Erlöse aus Verkäufen Sachanlagevermögen (bei Buchverlust)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8801: OdooAccountTemplate = OdooAccountTemplate {
    code: "8801",
    name: "Erlöse aus Verkäufen Sachanlagevermögen 19 % USt (bei Buchverlust)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8807: OdooAccountTemplate = OdooAccountTemplate {
    code: "8807",
    name: "Erlöse aus Verkäufen Sachanlagevermögen steuerfrei § 4 Nr. 1a UStG (bei Buchverlust)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8808: OdooAccountTemplate = OdooAccountTemplate {
    code: "8808",
    name: "Erlöse aus Verkäufen Sachanlagevermögen steuerfrei § 4 Nr. 1b UStG (bei Buchverlust)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8817: OdooAccountTemplate = OdooAccountTemplate {
    code: "8817",
    name: "Erlöse aus Verkäufen immaterieller Vermögensgegenstände (bei Buchverlust)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8818: OdooAccountTemplate = OdooAccountTemplate {
    code: "8818",
    name: "Erlöse aus Verkäufen Finanzanlagen (bei Buchverlust)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8819: OdooAccountTemplate = OdooAccountTemplate {
    code: "8819",
    name: "Erlöse aus Verkäufen Finanzanlagen § 3 Nr. 40 EStG bzw. § 8b Abs. 3 KStG (bei Buchverlust)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8820: OdooAccountTemplate = OdooAccountTemplate {
    code: "8820",
    name: "Erlöse aus Verkäufen Sachanlagevermögen 19 % USt (bei Buchgewinn)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8827: OdooAccountTemplate = OdooAccountTemplate {
    code: "8827",
    name: "Erlöse aus Verkäufen Sachanlagevermögen steuerfrei § 4 Nr. 1a UStG (bei Buchgewinn)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8828: OdooAccountTemplate = OdooAccountTemplate {
    code: "8828",
    name: "Erlöse aus Verkäufen Sachanlagevermögen steuerfrei § 4 Nr. 1b UStG (bei Buchgewinn)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8829: OdooAccountTemplate = OdooAccountTemplate {
    code: "8829",
    name: "Erlöse aus Verkäufen Sachanlagevermögen (bei Buchgewinn)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8837: OdooAccountTemplate = OdooAccountTemplate {
    code: "8837",
    name: "Erlöse aus Verkäufen immaterieller Vermögensgegenstände (bei Buchgewinn)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8838: OdooAccountTemplate = OdooAccountTemplate {
    code: "8838",
    name: "Erlöse aus Verkäufen Finanzanlagen (bei Buchgewinn)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8839: OdooAccountTemplate = OdooAccountTemplate {
    code: "8839",
    name: "Erlöse aus Verkäufen Finanzanlagen § 3 Nr. 40 EStG bzw. § 8b Abs. 2 KStG (bei Buchgewinn)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8850: OdooAccountTemplate = OdooAccountTemplate {
    code: "8850",
    name: "Erlöse aus Verkäufen von Wirtschaftsgütern des Umlaufvermögens 19 % USt für § 4 Abs. 3 Satz 4 EStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8851: OdooAccountTemplate = OdooAccountTemplate {
    code: "8851",
    name: "Erlöse aus Verkäufen von Wirtschaftsgütern des Umlaufvermögens, umsatzsteuerfrei § 4 Nr. 8 ff. UStG i. V. m. § 4 Abs. 3 Satz 4 EStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8852: OdooAccountTemplate = OdooAccountTemplate {
    code: "8852",
    name: "Erlöse aus Verkäufen von Wirtschaftsgütern des Umlaufvermögens, umsatzsteuerfrei § 4 Nr. 8 ff. UStG i. V. m. § 4 Abs. 3 Satz 4 EStG und § 3 Nr. 40 EStG bzw. § 8b Abs. 2 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8853: OdooAccountTemplate = OdooAccountTemplate {
    code: "8853",
    name: "Erlöse aus Verkäufen von Wirtschaftsgütern des Umlaufvermögens nach § 4 Abs 3 Satz 4 EStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8900: OdooAccountTemplate = OdooAccountTemplate {
    code: "8900",
    name: "Unentgeltliche Wertabgaben",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8905: OdooAccountTemplate = OdooAccountTemplate {
    code: "8905",
    name: "Entnahme von Gegenständen ohne USt",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8906: OdooAccountTemplate = OdooAccountTemplate {
    code: "8906",
    name: "Verwendung von Gegenständen für Zwecke außerhalb des Unternehmens ohne USt",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8910: OdooAccountTemplate = OdooAccountTemplate {
    code: "8910",
    name: "Entnahme durch den Unternehmer für Zwecke außerhalb des Unternehmens (Waren) 19 % USt",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8915: OdooAccountTemplate = OdooAccountTemplate {
    code: "8915",
    name: "Entnahme durch den Unternehmer für Zwecke außerhalb des Unternehmens (Waren) 7 % USt",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8918: OdooAccountTemplate = OdooAccountTemplate {
    code: "8918",
    name: "Verwendung von Gegenständen für Zwecke außerhalb des Unternehmens ohne USt (Telefon-Nutzung)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8919: OdooAccountTemplate = OdooAccountTemplate {
    code: "8919",
    name: "Entnahme durch den Unternehmer für Zwecke außerhalb des Unternehmens (Waren) ohne USt",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8920: OdooAccountTemplate = OdooAccountTemplate {
    code: "8920",
    name: "Verwendung von Gegenständen für Zwecke außerhalb des Unternehmens 19 % USt",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8921: OdooAccountTemplate = OdooAccountTemplate {
    code: "8921",
    name: "Verwendung von Gegenständen für Zwecke außerhalb des Unternehmens 19 % USt (Kfz-Nutzung)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8922: OdooAccountTemplate = OdooAccountTemplate {
    code: "8922",
    name: "Verwendung von Gegenständen für Zwecke außerhalb des Unternehmens 19 % USt (Telefon-Nutzung)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8924: OdooAccountTemplate = OdooAccountTemplate {
    code: "8924",
    name: "Verwendung von Gegenständen für Zwecke außerhalb des Unternehmens ohne USt (Kfz-Nutzung)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8925: OdooAccountTemplate = OdooAccountTemplate {
    code: "8925",
    name: "Unentgeltliche Erbringung einer sonstigen Leistung 19 % USt",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8929: OdooAccountTemplate = OdooAccountTemplate {
    code: "8929",
    name: "Unentgeltliche Erbringung einer sonstigen Leistung ohne USt",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8930: OdooAccountTemplate = OdooAccountTemplate {
    code: "8930",
    name: "Verwendung von Gegenständen für Zwecke außerhalb des Unternehmens 7 % USt",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8932: OdooAccountTemplate = OdooAccountTemplate {
    code: "8932",
    name: "Unentgeltliche Erbringung einer sonstigen Leistung 7 % USt",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8935: OdooAccountTemplate = OdooAccountTemplate {
    code: "8935",
    name: "Unentgeltliche Zuwendung von Gegenständen 19 % USt",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8939: OdooAccountTemplate = OdooAccountTemplate {
    code: "8939",
    name: "Unentgeltliche Zuwendung von Gegenständen ohne USt",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8940: OdooAccountTemplate = OdooAccountTemplate {
    code: "8940",
    name: "Unentgeltliche Zuwendung von Waren 19 % USt",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8945: OdooAccountTemplate = OdooAccountTemplate {
    code: "8945",
    name: "Unentgeltliche Zuwendung von Waren 7 % USt",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8949: OdooAccountTemplate = OdooAccountTemplate {
    code: "8949",
    name: "Unentgeltliche Zuwendung von Waren ohne USt",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8950: OdooAccountTemplate = OdooAccountTemplate {
    code: "8950",
    name: "Nicht steuerbare Umsätze (Innenumsätze)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8955: OdooAccountTemplate = OdooAccountTemplate {
    code: "8955",
    name: "Umsatzsteuervergütungen, z. B.nach § 24 UStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8959: OdooAccountTemplate = OdooAccountTemplate {
    code: "8959",
    name: "Direkt mit dem Umsatz verbundene Steuern",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8960: OdooAccountTemplate = OdooAccountTemplate {
    code: "8960",
    name: "Bestandsveränderungen - unfertige Erzeugnisse",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_02"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8970: OdooAccountTemplate = OdooAccountTemplate {
    code: "8970",
    name: "Bestandsveränderungen - unfertige Leistungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_02"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8975: OdooAccountTemplate = OdooAccountTemplate {
    code: "8975",
    name: "Bestandsveränderungen - in Ausführung befindliche Bauaufträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_02"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8977: OdooAccountTemplate = OdooAccountTemplate {
    code: "8977",
    name: "Bestandsveränderungen - in Arbeit befindliche Aufträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_02"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8980: OdooAccountTemplate = OdooAccountTemplate {
    code: "8980",
    name: "Bestandsveränderungen - fertige Erzeugnisse",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_02"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8990: OdooAccountTemplate = OdooAccountTemplate {
    code: "8990",
    name: "Andere aktivierte Eigenleistungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_03"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8994: OdooAccountTemplate = OdooAccountTemplate {
    code: "8994",
    name: "Aktivierte Eigenleistungen (den Herstellungskosten zurechenbare Fremdkapitalzinsen)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_03"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_8995: OdooAccountTemplate = OdooAccountTemplate {
    code: "8995",
    name: "Aktivierte Eigenleistungen zur Erstellung von selbst geschaffenen immateriellen Vermögensgegenständen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_03"],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_9000: OdooAccountTemplate = OdooAccountTemplate {
    code: "9000",
    name: "Saldenvorträge, Sachkonten",
    account_type: "income_other",
    tag_xmlids: &[],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_9001: OdooAccountTemplate = OdooAccountTemplate {
    code: "9001",
    name: "Saldenvorträge, Sachkonten",
    account_type: "income_other",
    tag_xmlids: &[],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_9008: OdooAccountTemplate = OdooAccountTemplate {
    code: "9008",
    name: "Saldenvorträge, Debitoren",
    account_type: "income_other",
    tag_xmlids: &[],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_9009: OdooAccountTemplate = OdooAccountTemplate {
    code: "9009",
    name: "Saldenvorträge, Kreditoren",
    account_type: "income_other",
    tag_xmlids: &[],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_9089: OdooAccountTemplate = OdooAccountTemplate {
    code: "9089",
    name: "Offene Posten aus 2019",
    account_type: "income_other",
    tag_xmlids: &[],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR03_9090: OdooAccountTemplate = OdooAccountTemplate {
    code: "9090",
    name: "Summenvortragskonto",
    account_type: "income_other",
    tag_xmlids: &[],
    chart: OdooSkrChart::Skr03,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

// ─── SKR04 account template consts ─────────────────────────────────────

pub const EXT_SKR04_0050: OdooAccountTemplate = OdooAccountTemplate {
    code: "0050",
    name: "Ausstehende Einlagen auf das Komplementär-Kapital, nicht eingefordert",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0060: OdooAccountTemplate = OdooAccountTemplate {
    code: "0060",
    name: "Ausstehende Einlagen auf das Komplementär-Kapital, eingefordert",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0070: OdooAccountTemplate = OdooAccountTemplate {
    code: "0070",
    name: "Ausstehende Einlagen auf das Kommandit-Kapital, nicht eingefordert",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0080: OdooAccountTemplate = OdooAccountTemplate {
    code: "0080",
    name: "Ausstehende Einlagen auf das Kommandit-Kapital, eingefordert",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0090: OdooAccountTemplate = OdooAccountTemplate {
    code: "0090",
    name: "Rückständige fällige Einzahlungen auf Geschäftsanteile",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0100: OdooAccountTemplate = OdooAccountTemplate {
    code: "0100",
    name: "Entgeltlich erworbene Konzessionen, gewerbliche Schutzrechte und ähnliche Rechte und Werte sowie Lizenzen an solchen Rechten und Werten",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0110: OdooAccountTemplate = OdooAccountTemplate {
    code: "0110",
    name: "Konzessionen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0120: OdooAccountTemplate = OdooAccountTemplate {
    code: "0120",
    name: "Gewerbliche Schutzrechte",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0130: OdooAccountTemplate = OdooAccountTemplate {
    code: "0130",
    name: "Ähnliche Rechte und Werte",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0135: OdooAccountTemplate = OdooAccountTemplate {
    code: "0135",
    name: "EDV-Software",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0140: OdooAccountTemplate = OdooAccountTemplate {
    code: "0140",
    name: "Lizenzen an gewerblichen Schutzrechten und ähnlichen Rechten und Werten",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0143: OdooAccountTemplate = OdooAccountTemplate {
    code: "0143",
    name: "Selbst geschaffene immaterielle Vermögensgegenstände",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0144: OdooAccountTemplate = OdooAccountTemplate {
    code: "0144",
    name: "EDV-Software",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0145: OdooAccountTemplate = OdooAccountTemplate {
    code: "0145",
    name: "Lizenzen und Franchiseverträge",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0146: OdooAccountTemplate = OdooAccountTemplate {
    code: "0146",
    name: "Konzessionen und gewerbliche Schutzrechte",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0147: OdooAccountTemplate = OdooAccountTemplate {
    code: "0147",
    name: "Rezepte, Verfahren, Prototypen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0148: OdooAccountTemplate = OdooAccountTemplate {
    code: "0148",
    name: "Immaterielle Vermögensgegenstände in Entwicklung",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0150: OdooAccountTemplate = OdooAccountTemplate {
    code: "0150",
    name: "0 Geschäfts- oder Firmenwert",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0170: OdooAccountTemplate = OdooAccountTemplate {
    code: "0170",
    name: "Geleistete Anzahlungen auf immaterielle Vermögensgegenstände",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0179: OdooAccountTemplate = OdooAccountTemplate {
    code: "0179",
    name: "Anzahlungen auf Geschäfts- oder Firmenwert",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_I_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0200: OdooAccountTemplate = OdooAccountTemplate {
    code: "0200",
    name: "Grundstücke, grundstücksgleiche Rechte und Bauten einschließlich der Bauten auf fremden Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0210: OdooAccountTemplate = OdooAccountTemplate {
    code: "0210",
    name: "Grundstücksgleiche Rechte ohne Bauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0215: OdooAccountTemplate = OdooAccountTemplate {
    code: "0215",
    name: "Unbebaute Grundstücke",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0220: OdooAccountTemplate = OdooAccountTemplate {
    code: "0220",
    name: "Grundstücksgleiche Rechte (Erbbaurecht, Dauerwohnrecht, unbebaute Grundstücke)",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0225: OdooAccountTemplate = OdooAccountTemplate {
    code: "0225",
    name: "Grundstücke mit Substanzverzehr",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0230: OdooAccountTemplate = OdooAccountTemplate {
    code: "0230",
    name: "Bauten auf eigenen Grundstücken und grundstücksgleichen Rechten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0235: OdooAccountTemplate = OdooAccountTemplate {
    code: "0235",
    name: "Grundstückswerte eigener bebauter Grundstücke",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0240: OdooAccountTemplate = OdooAccountTemplate {
    code: "0240",
    name: "Geschäftsbauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0250: OdooAccountTemplate = OdooAccountTemplate {
    code: "0250",
    name: "Fabrikbauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0260: OdooAccountTemplate = OdooAccountTemplate {
    code: "0260",
    name: "Andere Bauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0270: OdooAccountTemplate = OdooAccountTemplate {
    code: "0270",
    name: "Garagen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0280: OdooAccountTemplate = OdooAccountTemplate {
    code: "0280",
    name: "Außenanlagen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0285: OdooAccountTemplate = OdooAccountTemplate {
    code: "0285",
    name: "Hof- und Wegebefestigungen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0290: OdooAccountTemplate = OdooAccountTemplate {
    code: "0290",
    name: "Einrichtungen für Geschäfts-, Fabrik-, Wohn- und andere Bauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0300: OdooAccountTemplate = OdooAccountTemplate {
    code: "0300",
    name: "Wohnbauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0305: OdooAccountTemplate = OdooAccountTemplate {
    code: "0305",
    name: "Garagen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0310: OdooAccountTemplate = OdooAccountTemplate {
    code: "0310",
    name: "Außenanlagen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0315: OdooAccountTemplate = OdooAccountTemplate {
    code: "0315",
    name: "Hof- und Wegebefestigungen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0320: OdooAccountTemplate = OdooAccountTemplate {
    code: "0320",
    name: "Einrichtungen für Wohnbauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0329: OdooAccountTemplate = OdooAccountTemplate {
    code: "0329",
    name: "Gebäudeteil des häuslichen Arbeitszimmers",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0330: OdooAccountTemplate = OdooAccountTemplate {
    code: "0330",
    name: "Bauten auf fremden Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0340: OdooAccountTemplate = OdooAccountTemplate {
    code: "0340",
    name: "Geschäftsbauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0350: OdooAccountTemplate = OdooAccountTemplate {
    code: "0350",
    name: "Fabrikbauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0360: OdooAccountTemplate = OdooAccountTemplate {
    code: "0360",
    name: "Wohnbauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0370: OdooAccountTemplate = OdooAccountTemplate {
    code: "0370",
    name: "Andere Bauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0380: OdooAccountTemplate = OdooAccountTemplate {
    code: "0380",
    name: "Garagen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0390: OdooAccountTemplate = OdooAccountTemplate {
    code: "0390",
    name: "Außenanlagen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0395: OdooAccountTemplate = OdooAccountTemplate {
    code: "0395",
    name: "Hof- und Wegebefestigungen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0398: OdooAccountTemplate = OdooAccountTemplate {
    code: "0398",
    name: "Einrichtungen für Geschäfts-, Fabrik-, Wohn- und andere Bauten",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0400: OdooAccountTemplate = OdooAccountTemplate {
    code: "0400",
    name: "Technische Anlagen und Maschinen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0420: OdooAccountTemplate = OdooAccountTemplate {
    code: "0420",
    name: "Technische Anlagen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0440: OdooAccountTemplate = OdooAccountTemplate {
    code: "0440",
    name: "Maschinen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0450: OdooAccountTemplate = OdooAccountTemplate {
    code: "0450",
    name: "Transportanlagen und Ähnliches",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0460: OdooAccountTemplate = OdooAccountTemplate {
    code: "0460",
    name: "Maschinengebundene Werkzeuge",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0470: OdooAccountTemplate = OdooAccountTemplate {
    code: "0470",
    name: "Betriebsvorrichtungen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0500: OdooAccountTemplate = OdooAccountTemplate {
    code: "0500",
    name: "Andere Anlagen, Betriebs- und Geschäftsausstattung",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0510: OdooAccountTemplate = OdooAccountTemplate {
    code: "0510",
    name: "Andere Anlagen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0520: OdooAccountTemplate = OdooAccountTemplate {
    code: "0520",
    name: "Pkw",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0540: OdooAccountTemplate = OdooAccountTemplate {
    code: "0540",
    name: "Lkw",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0560: OdooAccountTemplate = OdooAccountTemplate {
    code: "0560",
    name: "Sonstige Transportmittel",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0620: OdooAccountTemplate = OdooAccountTemplate {
    code: "0620",
    name: "Werkzeuge",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0630: OdooAccountTemplate = OdooAccountTemplate {
    code: "0630",
    name: "Betriebsausstattung",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0635: OdooAccountTemplate = OdooAccountTemplate {
    code: "0635",
    name: "Geschäftsausstattung",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0640: OdooAccountTemplate = OdooAccountTemplate {
    code: "0640",
    name: "Ladeneinrichtung",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0650: OdooAccountTemplate = OdooAccountTemplate {
    code: "0650",
    name: "Büroeinrichtung",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0660: OdooAccountTemplate = OdooAccountTemplate {
    code: "0660",
    name: "Gerüst- und Schalungsmaterial",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0670: OdooAccountTemplate = OdooAccountTemplate {
    code: "0670",
    name: "Geringwertige Wirtschaftsgüter",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0675: OdooAccountTemplate = OdooAccountTemplate {
    code: "0675",
    name: "Wirtschaftsgüter (Sammelposten)",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0680: OdooAccountTemplate = OdooAccountTemplate {
    code: "0680",
    name: "Einbauten in fremde Grundstücke",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0690: OdooAccountTemplate = OdooAccountTemplate {
    code: "0690",
    name: "Sonstige Betriebs- und Geschäftsausstattung",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0700: OdooAccountTemplate = OdooAccountTemplate {
    code: "0700",
    name: "Geleistete Anzahlungen und Anlagen im Bau",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0705: OdooAccountTemplate = OdooAccountTemplate {
    code: "0705",
    name: "Anzahlungen auf Grund und Boden",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0710: OdooAccountTemplate = OdooAccountTemplate {
    code: "0710",
    name: "Geschäfts-, Fabrik- und andere Bauten im Bau auf eigenen Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0720: OdooAccountTemplate = OdooAccountTemplate {
    code: "0720",
    name: "Anzahlungen auf Geschäfts-, Fabrik- und andere Bauten auf eigenen Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0725: OdooAccountTemplate = OdooAccountTemplate {
    code: "0725",
    name: "Wohnbauten im Bau auf fremden Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0735: OdooAccountTemplate = OdooAccountTemplate {
    code: "0735",
    name: "Anzahlungen auf Wohnbauten auf eigenen Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0740: OdooAccountTemplate = OdooAccountTemplate {
    code: "0740",
    name: "Geschäfts-, Fabrik- und andere Bauten im Bau auf eigenen Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0755: OdooAccountTemplate = OdooAccountTemplate {
    code: "0755",
    name: "Wohnbauten im Bau auf fremden Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0765: OdooAccountTemplate = OdooAccountTemplate {
    code: "0765",
    name: "Anzahlungen auf Wohnbauten auf fremden Grundstücken",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0770: OdooAccountTemplate = OdooAccountTemplate {
    code: "0770",
    name: "Technische Anlagen und Maschinen im Bau",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0780: OdooAccountTemplate = OdooAccountTemplate {
    code: "0780",
    name: "Anzahlungen auf technische Anlagen und Maschinen",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0785: OdooAccountTemplate = OdooAccountTemplate {
    code: "0785",
    name: "Andere Anlagen, Betriebs- und Geschäftsausstattung im Bau",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0795: OdooAccountTemplate = OdooAccountTemplate {
    code: "0795",
    name: "Anzahlungen auf andere Anlagen, Betriebs- und Geschäftsausstattung",
    account_type: "asset_fixed",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0800: OdooAccountTemplate = OdooAccountTemplate {
    code: "0800",
    name: "Anteile an verbundenen Unternehmen (Anlagevermögen)",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0803: OdooAccountTemplate = OdooAccountTemplate {
    code: "0803",
    name: "Anteile an verbundenen Unternehmen, Personengesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0804: OdooAccountTemplate = OdooAccountTemplate {
    code: "0804",
    name: "Anteile an verbundenen Unternehmen, Kapitalgesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0805: OdooAccountTemplate = OdooAccountTemplate {
    code: "0805",
    name: "Anteile an herrschender oder mehrheitlich beteiligter Gesellschaft, Personengesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0808: OdooAccountTemplate = OdooAccountTemplate {
    code: "0808",
    name: "Anteile an herrschender oder mehrheitlich beteiligter Gesellschaft, Kapitalgesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0809: OdooAccountTemplate = OdooAccountTemplate {
    code: "0809",
    name: "Anteile an herrschender oder mehrheitlich beteiligter Gesellschaft",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0810: OdooAccountTemplate = OdooAccountTemplate {
    code: "0810",
    name: "Ausleihungen an verbundene Unternehmen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0813: OdooAccountTemplate = OdooAccountTemplate {
    code: "0813",
    name: "Ausleihungen an verbundene Unternehmen, Personengesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0814: OdooAccountTemplate = OdooAccountTemplate {
    code: "0814",
    name: "Ausleihungen an verbundene Unternehmen, Kapitalgesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0815: OdooAccountTemplate = OdooAccountTemplate {
    code: "0815",
    name: "Ausleihungen an verbundene Unternehmen, Einzelunternehmen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0820: OdooAccountTemplate = OdooAccountTemplate {
    code: "0820",
    name: "Beteiligungen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0829: OdooAccountTemplate = OdooAccountTemplate {
    code: "0829",
    name: "Beteiligung einer GmbH & Co. KG an einer Komplementär-GmbH",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0830: OdooAccountTemplate = OdooAccountTemplate {
    code: "0830",
    name: "Typisch stille Beteiligungen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0840: OdooAccountTemplate = OdooAccountTemplate {
    code: "0840",
    name: "Atypisch stille Beteiligungen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0850: OdooAccountTemplate = OdooAccountTemplate {
    code: "0850",
    name: "Beteiligungen an Kapitalgesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0860: OdooAccountTemplate = OdooAccountTemplate {
    code: "0860",
    name: "Beteiligungen an Personengesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0880: OdooAccountTemplate = OdooAccountTemplate {
    code: "0880",
    name: "Ausleihungen an Unternehmen, mit denen ein Beteiligungsverhältnis besteht",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0883: OdooAccountTemplate = OdooAccountTemplate {
    code: "0883",
    name: "Ausleihungen an Unternehmen, mit denen ein Beteiligungsverhältnis besteht, Personengesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0885: OdooAccountTemplate = OdooAccountTemplate {
    code: "0885",
    name: "Ausleihungen an Unternehmen, mit denen ein Beteiligungsverhältnis besteht, Kapitalgesellschaften",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0900: OdooAccountTemplate = OdooAccountTemplate {
    code: "0900",
    name: "Wertpapiere des Anlagevermögens",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0910: OdooAccountTemplate = OdooAccountTemplate {
    code: "0910",
    name: "Wertpapiere mit Gewinnbeteiligungsansprüchen, die dem Teileinkünfteverfahren unterliegen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0920: OdooAccountTemplate = OdooAccountTemplate {
    code: "0920",
    name: "Festverzinsliche Wertpapiere",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0930: OdooAccountTemplate = OdooAccountTemplate {
    code: "0930",
    name: "Übrige sonstige Ausleihungen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0940: OdooAccountTemplate = OdooAccountTemplate {
    code: "0940",
    name: "Darlehen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0960: OdooAccountTemplate = OdooAccountTemplate {
    code: "0960",
    name: "Ausleihungen an Gesellschafter",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0961: OdooAccountTemplate = OdooAccountTemplate {
    code: "0961",
    name: "Ausleihungen an GmbH-Gesellschafter",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0962: OdooAccountTemplate = OdooAccountTemplate {
    code: "0962",
    name: "Ausleihungen an persönlich haftende Gesellschafter",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0963: OdooAccountTemplate = OdooAccountTemplate {
    code: "0963",
    name: "Ausleihungen an Kommanditisten",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0964: OdooAccountTemplate = OdooAccountTemplate {
    code: "0964",
    name: "Ausleihungen an stille Gesellschafter",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0970: OdooAccountTemplate = OdooAccountTemplate {
    code: "0970",
    name: "Ausleihungen an nahe stehende Personen",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_0980: OdooAccountTemplate = OdooAccountTemplate {
    code: "0980",
    name: "Genossenschaftsanteile zum langfristigen Verbleib",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_A_III_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1000: OdooAccountTemplate = OdooAccountTemplate {
    code: "1000",
    name: "Roh-, Hilfs- und Betriebsstoffe",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1040: OdooAccountTemplate = OdooAccountTemplate {
    code: "1040",
    name: "Unfertige Erzeugnisse, unfertige Leistungen (Bestand)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1050: OdooAccountTemplate = OdooAccountTemplate {
    code: "1050",
    name: "Unfertige Erzeugnisse (Bestand)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1080: OdooAccountTemplate = OdooAccountTemplate {
    code: "1080",
    name: "Unfertige Leistungen (Bestand)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1090: OdooAccountTemplate = OdooAccountTemplate {
    code: "1090",
    name: "In Ausführung befindliche Bauaufträge",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1095: OdooAccountTemplate = OdooAccountTemplate {
    code: "1095",
    name: "In Arbeit befindliche Aufträge",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1100: OdooAccountTemplate = OdooAccountTemplate {
    code: "1100",
    name: "Fertige Erzeugnisse und Waren (Bestand)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1110: OdooAccountTemplate = OdooAccountTemplate {
    code: "1110",
    name: "Fertige Erzeugnisse (Bestand)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1140: OdooAccountTemplate = OdooAccountTemplate {
    code: "1140",
    name: "Waren (Bestand)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1180: OdooAccountTemplate = OdooAccountTemplate {
    code: "1180",
    name: "Geleistete Anzahlungen auf Vorräte",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1181: OdooAccountTemplate = OdooAccountTemplate {
    code: "1181",
    name: "Geleistete Anzahlungen 7 % Vorsteuer",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1186: OdooAccountTemplate = OdooAccountTemplate {
    code: "1186",
    name: "Geleistete Anzahlungen 19 % Vorsteuer",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1190: OdooAccountTemplate = OdooAccountTemplate {
    code: "1190",
    name: "Erhaltene Anzahlungen auf Bestellungen (von Vorräten offen abgesetzt)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_I_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1200: OdooAccountTemplate = OdooAccountTemplate {
    code: "1200",
    name: "Forderungen aus Lieferungen und Leistungen",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1205: OdooAccountTemplate = OdooAccountTemplate {
    code: "1205",
    name: "Forderungen aus Lieferungen und Leistungen (Odoo)",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1206: OdooAccountTemplate = OdooAccountTemplate {
    code: "1206",
    name: "Forderungen aus Lieferungen und Leistungen (PoS)",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1210: OdooAccountTemplate = OdooAccountTemplate {
    code: "1210",
    name: "Forderungen aus Lieferungen und Leistungen ohne Kontokorrent",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1215: OdooAccountTemplate = OdooAccountTemplate {
    code: "1215",
    name: "Forderungen aus Lieferungen und Leistungen zum allgemeinen Steuersatz",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1216: OdooAccountTemplate = OdooAccountTemplate {
    code: "1216",
    name: "Forderungen aus Lieferungen und Leistungen zum ermäßigten Steuersatz",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1217: OdooAccountTemplate = OdooAccountTemplate {
    code: "1217",
    name: "Forderungen aus steuerfreien oder nicht steuerbaren Lieferungen und Leistungen (EÜR)",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1218: OdooAccountTemplate = OdooAccountTemplate {
    code: "1218",
    name: "Forderungen aus Lieferungen und Leistungen gemäß §24 UStG",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1219: OdooAccountTemplate = OdooAccountTemplate {
    code: "1219",
    name: "Gegenkonto 1215-1218 bei Aufteilung der Forderungen nach Steuersätzen (EÜR)",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1220: OdooAccountTemplate = OdooAccountTemplate {
    code: "1220",
    name: "Forderungen nach § 11 Abs. 1 Satz 2 EStG für § 4 Abs. 3 EStG",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1221: OdooAccountTemplate = OdooAccountTemplate {
    code: "1221",
    name: "Forderungen aus Lieferungen und Leistungen ohne Kontokorrent – Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1225: OdooAccountTemplate = OdooAccountTemplate {
    code: "1225",
    name: "Forderungen aus Lieferungen und Leistungen ohne Kontokorrent - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1230: OdooAccountTemplate = OdooAccountTemplate {
    code: "1230",
    name: "Wechsel aus Lieferungen und Leistungen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1231: OdooAccountTemplate = OdooAccountTemplate {
    code: "1231",
    name: "Wechsel aus Lieferungen und Leistungen Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1232: OdooAccountTemplate = OdooAccountTemplate {
    code: "1232",
    name: "Wechsel aus Lieferungen und Leistungen Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1235: OdooAccountTemplate = OdooAccountTemplate {
    code: "1235",
    name: "Wechsel aus Lieferungen und Leistungen, bundesbankfähig",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1240: OdooAccountTemplate = OdooAccountTemplate {
    code: "1240",
    name: "Zweifelhafte Forderungen",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1241: OdooAccountTemplate = OdooAccountTemplate {
    code: "1241",
    name: "Zweifelhafte Forderungen - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1245: OdooAccountTemplate = OdooAccountTemplate {
    code: "1245",
    name: "Zweifelhafte Forderungen - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1246: OdooAccountTemplate = OdooAccountTemplate {
    code: "1246",
    name: "Einzelwertberichtigungen auf Forderungen mit einer – Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1247: OdooAccountTemplate = OdooAccountTemplate {
    code: "1247",
    name: "Einzelwertberichtigungen auf Forderungen mit einer - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1248: OdooAccountTemplate = OdooAccountTemplate {
    code: "1248",
    name: "Pauschalwertberichtigung auf Forderungen mit einer – Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1249: OdooAccountTemplate = OdooAccountTemplate {
    code: "1249",
    name: "Pauschalwertberichtigung auf Forderungen mit einer - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1250: OdooAccountTemplate = OdooAccountTemplate {
    code: "1250",
    name: "Forderungen aus Lieferungen und Leistungen gegen Gesellschafter",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1251: OdooAccountTemplate = OdooAccountTemplate {
    code: "1251",
    name: "Forderungen aus Lieferungen und Leistungen gegen Gesellschafter - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1255: OdooAccountTemplate = OdooAccountTemplate {
    code: "1255",
    name: "Forderungen aus Lieferungen und Leistungen gegen Gesellschafter - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1260: OdooAccountTemplate = OdooAccountTemplate {
    code: "1260",
    name: "Forderungen gegen verbundene Unternehmen",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1261: OdooAccountTemplate = OdooAccountTemplate {
    code: "1261",
    name: "Forderungen gegen verbundene Unternehmen - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1265: OdooAccountTemplate = OdooAccountTemplate {
    code: "1265",
    name: "Forderungen gegen verbundene Unternehmen - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1266: OdooAccountTemplate = OdooAccountTemplate {
    code: "1266",
    name: "Besitzwechsel gegen verbundene Unternehmen",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1267: OdooAccountTemplate = OdooAccountTemplate {
    code: "1267",
    name: "Besitzwechsel gegen verbundene Unternehmen Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1268: OdooAccountTemplate = OdooAccountTemplate {
    code: "1268",
    name: "Besitzwechsel gegen verbundene Unternehmen Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1269: OdooAccountTemplate = OdooAccountTemplate {
    code: "1269",
    name: "Besitzwechsel gegen verbundene Unternehmen, bundesbankfähig",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1270: OdooAccountTemplate = OdooAccountTemplate {
    code: "1270",
    name: "Forderungen aus Lieferungen und Leistungen gegen verbundene Unternehmen",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1271: OdooAccountTemplate = OdooAccountTemplate {
    code: "1271",
    name: "Forderungen aus Lieferungen und Leistungen gegen verbundene Unternehmen - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1275: OdooAccountTemplate = OdooAccountTemplate {
    code: "1275",
    name: "Forderungen aus Lieferungen und Leistungen gegen verbundene Unternehmen - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1276: OdooAccountTemplate = OdooAccountTemplate {
    code: "1276",
    name: "Wertberichtigungen auf Forderungen gegen verbundene Unternehmen – Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1277: OdooAccountTemplate = OdooAccountTemplate {
    code: "1277",
    name: "Wertberichtigungen auf Forderungen gegen verbundene Unternehmen - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1280: OdooAccountTemplate = OdooAccountTemplate {
    code: "1280",
    name: "Forderungen gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1281: OdooAccountTemplate = OdooAccountTemplate {
    code: "1281",
    name: "Forderungen gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1285: OdooAccountTemplate = OdooAccountTemplate {
    code: "1285",
    name: "Forderungen gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1286: OdooAccountTemplate = OdooAccountTemplate {
    code: "1286",
    name: "Besitzwechsel gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1287: OdooAccountTemplate = OdooAccountTemplate {
    code: "1287",
    name: "Besitzwechsel gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1288: OdooAccountTemplate = OdooAccountTemplate {
    code: "1288",
    name: "Besitzwechsel gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1289: OdooAccountTemplate = OdooAccountTemplate {
    code: "1289",
    name: "Besitzwechsel gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht, bundesbankfähig",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1290: OdooAccountTemplate = OdooAccountTemplate {
    code: "1290",
    name: "Forderungen aus Lieferungen und Leistungen gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1291: OdooAccountTemplate = OdooAccountTemplate {
    code: "1291",
    name: "Forderungen aus Lieferungen und Leistungen gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1295: OdooAccountTemplate = OdooAccountTemplate {
    code: "1295",
    name: "Forderungen aus Lieferungen und Leistungen gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1296: OdooAccountTemplate = OdooAccountTemplate {
    code: "1296",
    name: "Wertberichtigungen auf Forderungen gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht – Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1297: OdooAccountTemplate = OdooAccountTemplate {
    code: "1297",
    name: "Wertberichtigungen auf Forderungen gegen Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit größer 1 Jahr",
    account_type: "asset_non_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1298: OdooAccountTemplate = OdooAccountTemplate {
    code: "1298",
    name: "Ausstehende Einlagen auf das gezeichnete Kapital, eingefordert (Forderungen, nicht eingeforderte ausstehende Einlagen s. Konto 2910)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1299: OdooAccountTemplate = OdooAccountTemplate {
    code: "1299",
    name: "Nachschüsse (Forderungen, Gegenkonto 2929)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1300: OdooAccountTemplate = OdooAccountTemplate {
    code: "1300",
    name: "Sonstige Vermögensgegenstände",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1301: OdooAccountTemplate = OdooAccountTemplate {
    code: "1301",
    name: "Sonstige Vermögensgegenstände - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1305: OdooAccountTemplate = OdooAccountTemplate {
    code: "1305",
    name: "Sonstige Vermögensgegenstände - Restlaufzeit größer 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1307: OdooAccountTemplate = OdooAccountTemplate {
    code: "1307",
    name: "Forderungen gegen GmbH-Gesellschafter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1308: OdooAccountTemplate = OdooAccountTemplate {
    code: "1308",
    name: "Forderungen gegen GmbH-Gesellschafter - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1309: OdooAccountTemplate = OdooAccountTemplate {
    code: "1309",
    name: "Forderungen gegen GmbH-Gesellschafter - Restlaufzeit größer 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1310: OdooAccountTemplate = OdooAccountTemplate {
    code: "1310",
    name: "Forderungen gegen Vorstandsmitglieder und Geschäftsführer",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1311: OdooAccountTemplate = OdooAccountTemplate {
    code: "1311",
    name: "Forderungen gegen Vorstandsmitglieder und Geschäftsführer - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1315: OdooAccountTemplate = OdooAccountTemplate {
    code: "1315",
    name: "Forderungen gegen Vorstandsmitglieder und Geschäftsführer - Restlaufzeit größer 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1317: OdooAccountTemplate = OdooAccountTemplate {
    code: "1317",
    name: "Forderungen gegen persönlich haftende Gesellschafter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1318: OdooAccountTemplate = OdooAccountTemplate {
    code: "1318",
    name: "Forderungen gegen persönlich haftende Gesellschafter - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1319: OdooAccountTemplate = OdooAccountTemplate {
    code: "1319",
    name: "Forderungen gegen persönlich haftende Gesellschafter - Restlaufzeit größer 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1320: OdooAccountTemplate = OdooAccountTemplate {
    code: "1320",
    name: "Forderungen gegen Aufsichtsratsund Beirats-Mitglieder",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1321: OdooAccountTemplate = OdooAccountTemplate {
    code: "1321",
    name: "Forderungen gegen Aufsichtsratsund Beirats-Mitglieder - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1325: OdooAccountTemplate = OdooAccountTemplate {
    code: "1325",
    name: "Forderungen gegen Aufsichtsratsund Beirats-Mitglieder - Restlaufzeit größer 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1327: OdooAccountTemplate = OdooAccountTemplate {
    code: "1327",
    name: "Forderungen gegen Kommanditisten und atypisch stille Gesellschafter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1328: OdooAccountTemplate = OdooAccountTemplate {
    code: "1328",
    name: "Forderungen gegen Kommanditisten und atypisch stille Gesellschafter - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1329: OdooAccountTemplate = OdooAccountTemplate {
    code: "1329",
    name: "Forderungen gegen Kommanditisten und atypisch stille Gesellschafter - Restlaufzeit größer 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1330: OdooAccountTemplate = OdooAccountTemplate {
    code: "1330",
    name: "Forderungen gegen sonstige Gesellschafter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1331: OdooAccountTemplate = OdooAccountTemplate {
    code: "1331",
    name: "Forderungen gegen sonstige Gesellschafter - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1335: OdooAccountTemplate = OdooAccountTemplate {
    code: "1335",
    name: "Forderungen gegen sonstige Gesellschafter - Restlaufzeit größer 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1337: OdooAccountTemplate = OdooAccountTemplate {
    code: "1337",
    name: "Forderungen gegen typisch stille Gesellschafter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1338: OdooAccountTemplate = OdooAccountTemplate {
    code: "1338",
    name: "Forderungen gegen typisch stille Gesellschafter - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1339: OdooAccountTemplate = OdooAccountTemplate {
    code: "1339",
    name: "Forderungen gegen typisch stille Gesellschafter - Restlaufzeit größer 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1340: OdooAccountTemplate = OdooAccountTemplate {
    code: "1340",
    name: "Forderungen gegen Personal aus Lohn- und Gehaltsabrechnung",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1341: OdooAccountTemplate = OdooAccountTemplate {
    code: "1341",
    name: "Forderungen gegen Personal aus Lohn- und Gehaltsabrechnung - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1345: OdooAccountTemplate = OdooAccountTemplate {
    code: "1345",
    name: "Forderungen gegen Personal aus Lohn- und Gehaltsabrechnung - Restlaufzeit größer 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1350: OdooAccountTemplate = OdooAccountTemplate {
    code: "1350",
    name: "Kautionen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1351: OdooAccountTemplate = OdooAccountTemplate {
    code: "1351",
    name: "Kautionen - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1355: OdooAccountTemplate = OdooAccountTemplate {
    code: "1355",
    name: "Kautionen - Restlaufzeit größer 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1360: OdooAccountTemplate = OdooAccountTemplate {
    code: "1360",
    name: "Darlehen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1361: OdooAccountTemplate = OdooAccountTemplate {
    code: "1361",
    name: "Darlehen - Restlaufzeit bis 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1365: OdooAccountTemplate = OdooAccountTemplate {
    code: "1365",
    name: "Darlehen - Restlaufzeit größer 1 Jahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1369: OdooAccountTemplate = OdooAccountTemplate {
    code: "1369",
    name: "Forderungen gegenüber Krankenkassen aus Aufwendungsausgleichsgesetz",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1370: OdooAccountTemplate = OdooAccountTemplate {
    code: "1370",
    name: "Durchlaufende Posten",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1374: OdooAccountTemplate = OdooAccountTemplate {
    code: "1374",
    name: "Fremdgeld",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1375: OdooAccountTemplate = OdooAccountTemplate {
    code: "1375",
    name: "Agenturwarenabrechnung",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1378: OdooAccountTemplate = OdooAccountTemplate {
    code: "1378",
    name: "Ansprüche aus Rückdeckungsversicherungen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1380: OdooAccountTemplate = OdooAccountTemplate {
    code: "1380",
    name: "Vermögensgegenstände zur Erfüllung von Pensionsrückstellungen und ähnlichen Verpflichtungen zum langfristigen Verbleib",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1381: OdooAccountTemplate = OdooAccountTemplate {
    code: "1381",
    name: "Vermögensgegenstände zur Saldierung mit Pensionsrückstellungen und ähnlichen Verpflichtungen zum langfristigen Verbleib nach § 246 Abs. 2 HGB",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_E"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1382: OdooAccountTemplate = OdooAccountTemplate {
    code: "1382",
    name: "Vermögensgegenstände zur Erfüllung von mit der Altersversorgung vergleichbaren langfristigen Verpflichtungen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1383: OdooAccountTemplate = OdooAccountTemplate {
    code: "1383",
    name: "Vermögensgegenstände zur Saldierung mit der Altersversorgung vergleichbaren langfristigen Verpflichtungen nach § 246 Abs. 2 HGB",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_E"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1390: OdooAccountTemplate = OdooAccountTemplate {
    code: "1390",
    name: "GmbH-Anteile zum kurzfristigen Verbleib",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1391: OdooAccountTemplate = OdooAccountTemplate {
    code: "1391",
    name: "Forderungen gegen Arbeitsgemeinschaften",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1393: OdooAccountTemplate = OdooAccountTemplate {
    code: "1393",
    name: "Genussrechte",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1394: OdooAccountTemplate = OdooAccountTemplate {
    code: "1394",
    name: "Einzahlungsansprüche zu Nebenleistungen oder Zuzahlungen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1395: OdooAccountTemplate = OdooAccountTemplate {
    code: "1395",
    name: "Genossenschaftsanteile zum kurzfristigen Verbleib",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1396: OdooAccountTemplate = OdooAccountTemplate {
    code: "1396",
    name: "Nachträglich abziehbare Vorsteuer nach § 15a Abs. 1 UStG, bewegliche Wirtschaftsgüter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1397: OdooAccountTemplate = OdooAccountTemplate {
    code: "1397",
    name: "Zurückzuzahlende Vorsteuer nach § 15a Abs. 1 UStG, bewegliche Wirtschaftsgüter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1398: OdooAccountTemplate = OdooAccountTemplate {
    code: "1398",
    name: "Nachträglich abziehbare Vorsteuer nach § 15a Abs. 1 UStG, unbewegliche Wirtschaftsgüter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1399: OdooAccountTemplate = OdooAccountTemplate {
    code: "1399",
    name: "Zurückzuzahlende Vorsteuer nach § 15a Abs. 1 UStG, unbewegliche Wirtschaftsgüter",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1400: OdooAccountTemplate = OdooAccountTemplate {
    code: "1400",
    name: "Abziehbare Vorsteuer",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1401: OdooAccountTemplate = OdooAccountTemplate {
    code: "1401",
    name: "Abziehbare Vorsteuer 7 %",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1402: OdooAccountTemplate = OdooAccountTemplate {
    code: "1402",
    name: "Abziehbare Vorsteuer aus innergemeinschaftlichem Erwerb",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1404: OdooAccountTemplate = OdooAccountTemplate {
    code: "1404",
    name: "Abziehbare Vorsteuer aus innergemeinschaftlichem Erwerb 19 %",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1406: OdooAccountTemplate = OdooAccountTemplate {
    code: "1406",
    name: "Abziehbare Vorsteuer 19 %",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1407: OdooAccountTemplate = OdooAccountTemplate {
    code: "1407",
    name: "Abziehbare Vorsteuer nach § 13b UStG 19 %",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1408: OdooAccountTemplate = OdooAccountTemplate {
    code: "1408",
    name: "Abziehbare Vorsteuer nach § 13b UStG",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1410: OdooAccountTemplate = OdooAccountTemplate {
    code: "1410",
    name: "Aufzuteilende Vorsteuer",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1411: OdooAccountTemplate = OdooAccountTemplate {
    code: "1411",
    name: "Aufzuteilende Vorsteuer 7 %",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1412: OdooAccountTemplate = OdooAccountTemplate {
    code: "1412",
    name: "Aufzuteilende Vorsteuer aus innergemeinschaftlichem Erwerb",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1413: OdooAccountTemplate = OdooAccountTemplate {
    code: "1413",
    name: "Aufzuteilende Vorsteuer aus innergemeinschaftlichem Erwerb 19 %",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1416: OdooAccountTemplate = OdooAccountTemplate {
    code: "1416",
    name: "Aufzuteilende Vorsteuer 19 %",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1417: OdooAccountTemplate = OdooAccountTemplate {
    code: "1417",
    name: "Aufzuteilende Vorsteuer nach §§ 13a und 13b UStG",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1419: OdooAccountTemplate = OdooAccountTemplate {
    code: "1419",
    name: "Aufzuteilende Vorsteuer nach §§ 13a und 13b UStG 19 %",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1420: OdooAccountTemplate = OdooAccountTemplate {
    code: "1420",
    name: "Forderungen aus UmsatzsteuerVorauszahlungen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1421: OdooAccountTemplate = OdooAccountTemplate {
    code: "1421",
    name: "Forderungen aus Umsatzsteuervorauszahlungen für das laufende Jahr",
    account_type: "asset_receivable",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1422: OdooAccountTemplate = OdooAccountTemplate {
    code: "1422",
    name: "Umsatzsteuerforderungen Vorjahr",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1425: OdooAccountTemplate = OdooAccountTemplate {
    code: "1425",
    name: "Umsatzsteuerforderungen frühere Jahre",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1427: OdooAccountTemplate = OdooAccountTemplate {
    code: "1427",
    name: "Forderungen aus entrichteten Verbrauchsteuern",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1431: OdooAccountTemplate = OdooAccountTemplate {
    code: "1431",
    name: "Abziehbare Vorsteuer aus der Auslagerung von Gegenständen aus einem Umsatzsteuerlager",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1432: OdooAccountTemplate = OdooAccountTemplate {
    code: "1432",
    name: "Abziehbare Vorsteuer aus innergemeinschaftlichem Erwerb von Neufahrzeugen von Lieferanten ohne USt-Identifikationsnummer",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1433: OdooAccountTemplate = OdooAccountTemplate {
    code: "1433",
    name: "Entstandene Einfuhrumsatzsteuer",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1434: OdooAccountTemplate = OdooAccountTemplate {
    code: "1434",
    name: "Vorsteuer in Folgeperiode/im Folgejahr abziehbar",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1435: OdooAccountTemplate = OdooAccountTemplate {
    code: "1435",
    name: "Forderungen aus Gewerbesteuerüberzahlungen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1436: OdooAccountTemplate = OdooAccountTemplate {
    code: "1436",
    name: "Vorsteuer aus Erwerb als letzter Abnehmer innerhalb eines Dreiecksgeschäfts",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1440: OdooAccountTemplate = OdooAccountTemplate {
    code: "1440",
    name: "Steuerpflichtige sonstige Leistungen EU 7%USt/7%VSt",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1450: OdooAccountTemplate = OdooAccountTemplate {
    code: "1450",
    name: "Körperschaftsteuerrückforderung",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1456: OdooAccountTemplate = OdooAccountTemplate {
    code: "1456",
    name: "Forderungen an das Finanzamt aus abgeführtem Bauabzugsbetrag",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1457: OdooAccountTemplate = OdooAccountTemplate {
    code: "1457",
    name: "Forderung gegenüber Bundesagentur für Arbeit",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1480: OdooAccountTemplate = OdooAccountTemplate {
    code: "1480",
    name: "Gegenkonto Vorsteuer § 4 Abs. 3 EStG",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1481: OdooAccountTemplate = OdooAccountTemplate {
    code: "1481",
    name: "Auflösung Vorsteuer aus Vorjahr § 4 Abs. 3 EStG",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1482: OdooAccountTemplate = OdooAccountTemplate {
    code: "1482",
    name: "Vorsteuer aus Investitionen § 4 Abs. 3 EStG",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1483: OdooAccountTemplate = OdooAccountTemplate {
    code: "1483",
    name: "Gegenkonto für Vorsteuer nach Durchschnittssätzen für § 4 Abs. 3 EStG",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1485: OdooAccountTemplate = OdooAccountTemplate {
    code: "1485",
    name: "Verrechnungskonto Gewinnermittlung § 4/3 EStG, erfolgswirksam",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1486: OdooAccountTemplate = OdooAccountTemplate {
    code: "1486",
    name: "Verrechnungskonto für Gewinnermittlung § 4 Abs. 3 EStG, nicht ergebniswirksam",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1490: OdooAccountTemplate = OdooAccountTemplate {
    code: "1490",
    name: "Verrechnungskonto Ist-Versteuerung",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1495: OdooAccountTemplate = OdooAccountTemplate {
    code: "1495",
    name: "Verrechnungskonto erhaltene Anzahlungen bei Buchung über Debitorenkonto",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1498: OdooAccountTemplate = OdooAccountTemplate {
    code: "1498",
    name: "Überleitungskonto Kostenstellen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_1500: OdooAccountTemplate = OdooAccountTemplate {
    code: "1500",
    name: "Anteile an verbundenen Unternehmen (Umlaufvermögen)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_III_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1504: OdooAccountTemplate = OdooAccountTemplate {
    code: "1504",
    name: "Anteile an herrschender oder mehrheitlich beteiligter Gesellschaft",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_III_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1510: OdooAccountTemplate = OdooAccountTemplate {
    code: "1510",
    name: "Sonstige Wertpapiere",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_III_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1520: OdooAccountTemplate = OdooAccountTemplate {
    code: "1520",
    name: "Finanzwechsel",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_III_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1525: OdooAccountTemplate = OdooAccountTemplate {
    code: "1525",
    name: "Andere Wertpapiere mit unwesentlichen Wertschwankungen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_III_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1530: OdooAccountTemplate = OdooAccountTemplate {
    code: "1530",
    name: "Wertpapieranlagen im Rahmen der kurzfristigen Finanzdisposition",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_III_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1550: OdooAccountTemplate = OdooAccountTemplate {
    code: "1550",
    name: "Schecks",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1610: OdooAccountTemplate = OdooAccountTemplate {
    code: "1610",
    name: "Nebenkasse 1",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1620: OdooAccountTemplate = OdooAccountTemplate {
    code: "1620",
    name: "Nebenkasse 2",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1700: OdooAccountTemplate = OdooAccountTemplate {
    code: "1700",
    name: "Bank (Postbank)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1710: OdooAccountTemplate = OdooAccountTemplate {
    code: "1710",
    name: "Bank (Postbank 1)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1720: OdooAccountTemplate = OdooAccountTemplate {
    code: "1720",
    name: "Bank (Postbank 2)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1730: OdooAccountTemplate = OdooAccountTemplate {
    code: "1730",
    name: "Bank (Postbank 3)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1780: OdooAccountTemplate = OdooAccountTemplate {
    code: "1780",
    name: "LZB-Guthaben",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1790: OdooAccountTemplate = OdooAccountTemplate {
    code: "1790",
    name: "Bundesbankguthaben",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1810: OdooAccountTemplate = OdooAccountTemplate {
    code: "1810",
    name: "Paypal",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1820: OdooAccountTemplate = OdooAccountTemplate {
    code: "1820",
    name: "Bank 2",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1830: OdooAccountTemplate = OdooAccountTemplate {
    code: "1830",
    name: "Bank 3",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1840: OdooAccountTemplate = OdooAccountTemplate {
    code: "1840",
    name: "Bank 4",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1850: OdooAccountTemplate = OdooAccountTemplate {
    code: "1850",
    name: "Bank 5",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1890: OdooAccountTemplate = OdooAccountTemplate {
    code: "1890",
    name: "Finanzmittelanlagen im Rahmen der kurzfristigen Finanzdisposition (nicht im Finanzmittelfonds enthalten)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1900: OdooAccountTemplate = OdooAccountTemplate {
    code: "1900",
    name: "Aktive Rechnungsabgrenzung",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_C"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1920: OdooAccountTemplate = OdooAccountTemplate {
    code: "1920",
    name: "Als Aufwand berücksichtigte Zölle und Verbrauchsteuern auf Vorräte",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_C"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1930: OdooAccountTemplate = OdooAccountTemplate {
    code: "1930",
    name: "Als Aufwand berücksichtigte Umsatzsteuer auf Anzahlungen",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_C"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1940: OdooAccountTemplate = OdooAccountTemplate {
    code: "1940",
    name: "Damnum/Disagio",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_C"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_1950: OdooAccountTemplate = OdooAccountTemplate {
    code: "1950",
    name: "Aktive latente Steuern",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_C"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2000: OdooAccountTemplate = OdooAccountTemplate {
    code: "2000",
    name: "Festkapital",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2010: OdooAccountTemplate = OdooAccountTemplate {
    code: "2010",
    name: "Variables Kapital",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2020: OdooAccountTemplate = OdooAccountTemplate {
    code: "2020",
    name: "Gesellschafter-Darlehen",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2050: OdooAccountTemplate = OdooAccountTemplate {
    code: "2050",
    name: "Kommandit-Kapital",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2060: OdooAccountTemplate = OdooAccountTemplate {
    code: "2060",
    name: "Verlustausgleichskonto",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2070: OdooAccountTemplate = OdooAccountTemplate {
    code: "2070",
    name: "Gesellschafter-Darlehen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2100: OdooAccountTemplate = OdooAccountTemplate {
    code: "2100",
    name: "Privatentnahmen allgemein",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2130: OdooAccountTemplate = OdooAccountTemplate {
    code: "2130",
    name: "Unentgeltliche Wertabgaben",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2150: OdooAccountTemplate = OdooAccountTemplate {
    code: "2150",
    name: "Privatsteuern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2180: OdooAccountTemplate = OdooAccountTemplate {
    code: "2180",
    name: "Privateinlagen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2200: OdooAccountTemplate = OdooAccountTemplate {
    code: "2200",
    name: "Sonderausgaben beschränkt abzugsfähig",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2230: OdooAccountTemplate = OdooAccountTemplate {
    code: "2230",
    name: "Sonderausgaben unbeschränkt abzugsfähig",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2250: OdooAccountTemplate = OdooAccountTemplate {
    code: "2250",
    name: "Zuwendungen, Spenden",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2280: OdooAccountTemplate = OdooAccountTemplate {
    code: "2280",
    name: "Außergewöhnliche Belastungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2300: OdooAccountTemplate = OdooAccountTemplate {
    code: "2300",
    name: "Grundstücksaufwand",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2349: OdooAccountTemplate = OdooAccountTemplate {
    code: "2349",
    name: "Grundstücksaufwand (Umsatzsteuerschlüssel möglich, nur Einzelunternehmen)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2350: OdooAccountTemplate = OdooAccountTemplate {
    code: "2350",
    name: "Grundstücksertrag",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2399: OdooAccountTemplate = OdooAccountTemplate {
    code: "2399",
    name: "Grundstücksertrag (Umsatzsteuerschlüssel möglich, nur Einzelunternehmen)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2500: OdooAccountTemplate = OdooAccountTemplate {
    code: "2500",
    name: "Privatentnahmen allgemein (TH),FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2530: OdooAccountTemplate = OdooAccountTemplate {
    code: "2530",
    name: "Unentgeltliche Wertabgaben (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2550: OdooAccountTemplate = OdooAccountTemplate {
    code: "2550",
    name: "Privatsteuern (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2580: OdooAccountTemplate = OdooAccountTemplate {
    code: "2580",
    name: "Privateinlagen (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2600: OdooAccountTemplate = OdooAccountTemplate {
    code: "2600",
    name: "Sonderausgaben beschränkt abzugsfähig (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2630: OdooAccountTemplate = OdooAccountTemplate {
    code: "2630",
    name: "Sonderausgaben unbeschränkt abzugsfähig (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2650: OdooAccountTemplate = OdooAccountTemplate {
    code: "2650",
    name: "Zuwendungen, Spenden (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2680: OdooAccountTemplate = OdooAccountTemplate {
    code: "2680",
    name: "Außergewöhnliche Belastungen (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2700: OdooAccountTemplate = OdooAccountTemplate {
    code: "2700",
    name: "Grundstücksaufwand (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2750: OdooAccountTemplate = OdooAccountTemplate {
    code: "2750",
    name: "Grundstücksertrag (TH), FK",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2900: OdooAccountTemplate = OdooAccountTemplate {
    code: "2900",
    name: "Gezeichnetes Kapital",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2901: OdooAccountTemplate = OdooAccountTemplate {
    code: "2901",
    name: "Geschäftsguthaben der verbleibenden Mitglieder",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2902: OdooAccountTemplate = OdooAccountTemplate {
    code: "2902",
    name: "Geschäftsguthaben der ausscheidenden Mitglieder",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2903: OdooAccountTemplate = OdooAccountTemplate {
    code: "2903",
    name: "Geschäftsguthaben aus gekündigten Geschäftsanteilen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2906: OdooAccountTemplate = OdooAccountTemplate {
    code: "2906",
    name: "Rückständige fällige Einzahlungen auf Geschäftsanteile, vermerkt",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2907: OdooAccountTemplate = OdooAccountTemplate {
    code: "2907",
    name: "Gegenkonto Rückständige fällige Einzahlungen auf Geschäftsanteile, vermerkt",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2908: OdooAccountTemplate = OdooAccountTemplate {
    code: "2908",
    name: "Kapitalerhöhung aus Gesellschaftsmitteln",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2909: OdooAccountTemplate = OdooAccountTemplate {
    code: "2909",
    name: "Erworbene eigene Anteile",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2910: OdooAccountTemplate = OdooAccountTemplate {
    code: "2910",
    name: "Ausstehende Einlagen auf das gezeichnete Kapital, nicht eingefordert (Passivausweis, vom gezeichneten Kapital offen abgesetzt; eingeforderte ausstehende Einlagen s. Konto 1298)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_I"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2920: OdooAccountTemplate = OdooAccountTemplate {
    code: "2920",
    name: "Kapitalrücklage",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_II"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2925: OdooAccountTemplate = OdooAccountTemplate {
    code: "2925",
    name: "Kapitalrücklage durch Ausgabe von Anteilen über Nennbetrag",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_II"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2926: OdooAccountTemplate = OdooAccountTemplate {
    code: "2926",
    name: "Kapitalrücklage durch Ausgabe von Schuldverschreibungen für Wandlungsrechte und Optionsrechte zum Erwerb von Anteilen",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_II"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2927: OdooAccountTemplate = OdooAccountTemplate {
    code: "2927",
    name: "Kapitalrücklage durch Zuzahlungen gegen Gewährung eines Vorzugs für Anteile",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_II"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2928: OdooAccountTemplate = OdooAccountTemplate {
    code: "2928",
    name: "Kapitalrücklage durch Zuzahlungen in das Eigenkapital",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_II"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2929: OdooAccountTemplate = OdooAccountTemplate {
    code: "2929",
    name: "Nachschusskapital (Gegenkonto 1299)",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_II"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2930: OdooAccountTemplate = OdooAccountTemplate {
    code: "2930",
    name: "Gesetzliche Rücklage",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2935: OdooAccountTemplate = OdooAccountTemplate {
    code: "2935",
    name: "Rücklage für Anteile an einem herrschenden oder mehrheitlich beteiligten Unternehmen",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2937: OdooAccountTemplate = OdooAccountTemplate {
    code: "2937",
    name: "Andere Ergebnisrücklagen",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2950: OdooAccountTemplate = OdooAccountTemplate {
    code: "2950",
    name: "Satzungsmäßige Rücklagen",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2959: OdooAccountTemplate = OdooAccountTemplate {
    code: "2959",
    name: "Gesamthänderisch gebundene Rücklagen (mit Aufteilung für Kapitalkontenentwicklung)",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2960: OdooAccountTemplate = OdooAccountTemplate {
    code: "2960",
    name: "Andere Gewinnrücklagen",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2961: OdooAccountTemplate = OdooAccountTemplate {
    code: "2961",
    name: "Andere Gewinnrücklagen aus dem Erwerb eigener Anteile",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2962: OdooAccountTemplate = OdooAccountTemplate {
    code: "2962",
    name: "Eigenkapitalanteil von Wertaufholungen",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2963: OdooAccountTemplate = OdooAccountTemplate {
    code: "2963",
    name: "Gewinnrücklagen aus den Übergangsvorschriften BilMoG",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2964: OdooAccountTemplate = OdooAccountTemplate {
    code: "2964",
    name: "Gewinnrücklagen aus den Übergangsvorschriften BilMoG (Zuschreibung Sachanlagevermögen)",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2965: OdooAccountTemplate = OdooAccountTemplate {
    code: "2965",
    name: "Gewinnrücklagen aus den Übergangsvorschriften BilMoG (Zuschreibung Finanzanlagevermögen)",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2966: OdooAccountTemplate = OdooAccountTemplate {
    code: "2966",
    name: "Gewinnrücklagen aus den Übergangsvorschriften BilMoG (Auflösung der Sonderposten mit Rücklageanteil)",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2967: OdooAccountTemplate = OdooAccountTemplate {
    code: "2967",
    name: "Latente Steuern (Gewinnrücklage Haben) aus erfolgsneutralen Verrechnungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2968: OdooAccountTemplate = OdooAccountTemplate {
    code: "2968",
    name: "Latente Steuern (Gewinnrücklage Soll) aus erfolgsneutralen Verrechnungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2969: OdooAccountTemplate = OdooAccountTemplate {
    code: "2969",
    name: "Rechnungsabgrenzungsposten (Gewinnrücklage Soll) aus erfolgsneutralen Verrechnungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_III_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2970: OdooAccountTemplate = OdooAccountTemplate {
    code: "2970",
    name: "Gewinnvortrag vor Verwendung",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2975: OdooAccountTemplate = OdooAccountTemplate {
    code: "2975",
    name: "Gewinnvortrag vor Verwendung (mit Aufteilung für Kapitalkontenentwicklung)",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2977: OdooAccountTemplate = OdooAccountTemplate {
    code: "2977",
    name: "Verlustvortrag vor Verwendung (mit Aufteilung für Kapitalkontenentwicklung)",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2978: OdooAccountTemplate = OdooAccountTemplate {
    code: "2978",
    name: "Verlustvortrag vor Verwendung",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_2979: OdooAccountTemplate = OdooAccountTemplate {
    code: "2979",
    name: "Übertrag auf neue Rechnung ( Bilanz )",
    account_type: "equity",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_A_IV"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3000: OdooAccountTemplate = OdooAccountTemplate {
    code: "3000",
    name: "Rückstellungen für Pensionen und ähnliche Verpflichtungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3005: OdooAccountTemplate = OdooAccountTemplate {
    code: "3005",
    name: "Rückstellungen für Pensionen und ähnliche Verpflichtungen gegenüber Gesellschaftern oder nahe stehenden Personen (10 % Beteiligung am Kapital)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3009: OdooAccountTemplate = OdooAccountTemplate {
    code: "3009",
    name: "Rückstellungen für Pensionen und ähnliche Verpflichtungen zur Saldierung mit Vermögensgegenständen zum langfristigen Verbleib nach § 246 Abs. 2 HGB",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3010: OdooAccountTemplate = OdooAccountTemplate {
    code: "3010",
    name: "Rückstellungen für Direktzusagen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3011: OdooAccountTemplate = OdooAccountTemplate {
    code: "3011",
    name: "Rückstellungen für Zuschussverpflichtungen für Pensionskassen und Lebensversicherungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3015: OdooAccountTemplate = OdooAccountTemplate {
    code: "3015",
    name: "Rückstellungen für pensionsähnliche Verpflichtungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3020: OdooAccountTemplate = OdooAccountTemplate {
    code: "3020",
    name: "Steuerrückstellungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3030: OdooAccountTemplate = OdooAccountTemplate {
    code: "3030",
    name: "Rückstellung für Gewerbesteuer",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3035: OdooAccountTemplate = OdooAccountTemplate {
    code: "3035",
    name: "Gewerbesteuerrückstellung nach § 4 Abs. 5b EStG",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3040: OdooAccountTemplate = OdooAccountTemplate {
    code: "3040",
    name: "Körperschaftsteuerrückstellung",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3050: OdooAccountTemplate = OdooAccountTemplate {
    code: "3050",
    name: "Steuerrückstellung aus Steuerstundung (BStBK)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3060: OdooAccountTemplate = OdooAccountTemplate {
    code: "3060",
    name: "Rückstellung für latente Steuern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3065: OdooAccountTemplate = OdooAccountTemplate {
    code: "3065",
    name: "Passive latente Steuern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_E"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3070: OdooAccountTemplate = OdooAccountTemplate {
    code: "3070",
    name: "Sonstige Rückstellungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3075: OdooAccountTemplate = OdooAccountTemplate {
    code: "3075",
    name: "Rückstellungen für unterlassene Aufwendungen für Instandhaltung, Nachholung in den ersten drei Monaten",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3076: OdooAccountTemplate = OdooAccountTemplate {
    code: "3076",
    name: "Rückstellungen für mit der Altersversorgung vergleichbare langfristige Verpflichtungen zum langfristigen Verbleib",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3077: OdooAccountTemplate = OdooAccountTemplate {
    code: "3077",
    name: "Rückstellungen für mit der Altersversorgung vergleichbare langfristige Verpflichtungen zur Saldierung mit Vermögensgegenständen zum langfristigen Verbleib nach § 246 Abs. 2 HGB",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3079: OdooAccountTemplate = OdooAccountTemplate {
    code: "3079",
    name: "Urlaubsrückstellungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3085: OdooAccountTemplate = OdooAccountTemplate {
    code: "3085",
    name: "Rückstellungen für Abraum- und Abfallbeseitigung",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3090: OdooAccountTemplate = OdooAccountTemplate {
    code: "3090",
    name: "Rückstellungen für Gewährleistungen (Gegenkonto 6790)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3092: OdooAccountTemplate = OdooAccountTemplate {
    code: "3092",
    name: "Rückstellungen für drohende Verluste aus schwebenden Geschäften",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3095: OdooAccountTemplate = OdooAccountTemplate {
    code: "3095",
    name: "Rückstellungen für Abschlussund Prüfungskosten",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3098: OdooAccountTemplate = OdooAccountTemplate {
    code: "3098",
    name: "Aufwandsrückstellungen nach § 249 Abs. 2 HGB a. F.",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3099: OdooAccountTemplate = OdooAccountTemplate {
    code: "3099",
    name: "Rückstellungen für Umweltschutz",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3100: OdooAccountTemplate = OdooAccountTemplate {
    code: "3100",
    name: "Anleihen, nicht konvertibel",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3101: OdooAccountTemplate = OdooAccountTemplate {
    code: "3101",
    name: "Anleihen, nicht konvertibel - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3105: OdooAccountTemplate = OdooAccountTemplate {
    code: "3105",
    name: "Anleihen, nicht konvertibel - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3110: OdooAccountTemplate = OdooAccountTemplate {
    code: "3110",
    name: "Anleihen, nicht konvertibel - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3120: OdooAccountTemplate = OdooAccountTemplate {
    code: "3120",
    name: "Anleihen, konvertibel",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3121: OdooAccountTemplate = OdooAccountTemplate {
    code: "3121",
    name: "Anleihen, konvertibel - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3125: OdooAccountTemplate = OdooAccountTemplate {
    code: "3125",
    name: "Anleihen, konvertibel - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3130: OdooAccountTemplate = OdooAccountTemplate {
    code: "3130",
    name: "Anleihen, konvertibel -  Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3150: OdooAccountTemplate = OdooAccountTemplate {
    code: "3150",
    name: "Verbindlichkeiten gegenüber Kreditinstituten",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3151: OdooAccountTemplate = OdooAccountTemplate {
    code: "3151",
    name: "Verbindlichkeiten gegenüber Kreditinstituten - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3160: OdooAccountTemplate = OdooAccountTemplate {
    code: "3160",
    name: "Verbindlichkeiten gegenüber Kreditinstituten - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3170: OdooAccountTemplate = OdooAccountTemplate {
    code: "3170",
    name: "Verbindlichkeiten gegenüber Kreditinstituten - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3180: OdooAccountTemplate = OdooAccountTemplate {
    code: "3180",
    name: "Verbindlichkeiten gegenüber Kreditinstituten aus Teilzahlungsverträgen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3181: OdooAccountTemplate = OdooAccountTemplate {
    code: "3181",
    name: "Verbindlichkeiten gegenüber Kreditinstituten aus Teilzahlungsverträgen - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3190: OdooAccountTemplate = OdooAccountTemplate {
    code: "3190",
    name: "Verbindlichkeiten gegenüber Kreditinstituten aus Teilzahlungsverträgen - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3200: OdooAccountTemplate = OdooAccountTemplate {
    code: "3200",
    name: "Verbindlichkeiten gegenüber Kreditinstituten aus Teilzahlungsverträgen - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3250: OdooAccountTemplate = OdooAccountTemplate {
    code: "3250",
    name: "Erhaltene Anzahlungen auf Bestellungen (Verbindlichkeiten)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3260: OdooAccountTemplate = OdooAccountTemplate {
    code: "3260",
    name: "Erhaltene, versteuerte Anzahlungen 7 % USt (Verbindlichkeiten)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3272: OdooAccountTemplate = OdooAccountTemplate {
    code: "3272",
    name: "Erhaltene, versteuerte Anzahlungen 19 % USt (Verbindlichkeiten)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3280: OdooAccountTemplate = OdooAccountTemplate {
    code: "3280",
    name: "Erhaltene Anzahlungen – Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3284: OdooAccountTemplate = OdooAccountTemplate {
    code: "3284",
    name: "Erhaltene Anzahlungen - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3285: OdooAccountTemplate = OdooAccountTemplate {
    code: "3285",
    name: "Erhaltene Anzahlungen – Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3300: OdooAccountTemplate = OdooAccountTemplate {
    code: "3300",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3301: OdooAccountTemplate = OdooAccountTemplate {
    code: "3301",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3305: OdooAccountTemplate = OdooAccountTemplate {
    code: "3305",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen zum allgemeinen Umsatzsteuersatz (EÜR)",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3306: OdooAccountTemplate = OdooAccountTemplate {
    code: "3306",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen zum ermäßigten Umsatzsteuersatz (EÜR)",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3307: OdooAccountTemplate = OdooAccountTemplate {
    code: "3307",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen ohne Vorsteuer (EÜR)",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3309: OdooAccountTemplate = OdooAccountTemplate {
    code: "3309",
    name: "Gegenkonto 3305-3307 bei Aufteilung der Verbindlichkeiten nach Steuersätzen (EÜR)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3310: OdooAccountTemplate = OdooAccountTemplate {
    code: "3310",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen ohne Kontokorrent",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3334: OdooAccountTemplate = OdooAccountTemplate {
    code: "3334",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen für Investitionen für § 4 Abs. 3 EStG",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3335: OdooAccountTemplate = OdooAccountTemplate {
    code: "3335",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen ohne Kontokorrent - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3337: OdooAccountTemplate = OdooAccountTemplate {
    code: "3337",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen ohne Kontokorrent - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3338: OdooAccountTemplate = OdooAccountTemplate {
    code: "3338",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen ohne Kontokorrent - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3340: OdooAccountTemplate = OdooAccountTemplate {
    code: "3340",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber Gesellschaftern",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3341: OdooAccountTemplate = OdooAccountTemplate {
    code: "3341",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber Gesellschaftern - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3345: OdooAccountTemplate = OdooAccountTemplate {
    code: "3345",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber Gesellschaftern - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3348: OdooAccountTemplate = OdooAccountTemplate {
    code: "3348",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber Gesellschaftern - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3350: OdooAccountTemplate = OdooAccountTemplate {
    code: "3350",
    name: "Wechselverbindlichkeiten",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3351: OdooAccountTemplate = OdooAccountTemplate {
    code: "3351",
    name: "Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3380: OdooAccountTemplate = OdooAccountTemplate {
    code: "3380",
    name: "Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3390: OdooAccountTemplate = OdooAccountTemplate {
    code: "3390",
    name: "Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3400: OdooAccountTemplate = OdooAccountTemplate {
    code: "3400",
    name: "Verbindlichkeiten gegenüber verbundenen Unternehmen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3401: OdooAccountTemplate = OdooAccountTemplate {
    code: "3401",
    name: "Verbindlichkeiten gegenüber verbundenen Unternehmen - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3405: OdooAccountTemplate = OdooAccountTemplate {
    code: "3405",
    name: "Verbindlichkeiten gegenüber verbundenen Unternehmen - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3410: OdooAccountTemplate = OdooAccountTemplate {
    code: "3410",
    name: "Verbindlichkeiten gegenüber verbundenen Unternehmen - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3420: OdooAccountTemplate = OdooAccountTemplate {
    code: "3420",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber verbundenen Unternehmen",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3421: OdooAccountTemplate = OdooAccountTemplate {
    code: "3421",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber verbundenen Unternehmen - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3425: OdooAccountTemplate = OdooAccountTemplate {
    code: "3425",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber verbundenen Unternehmen - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3430: OdooAccountTemplate = OdooAccountTemplate {
    code: "3430",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber verbundenen Unternehmen - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3450: OdooAccountTemplate = OdooAccountTemplate {
    code: "3450",
    name: "Verbindlichkeiten gegenüber Unternehmen, mit denen ein Beteiligungsverhältnis besteht",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3451: OdooAccountTemplate = OdooAccountTemplate {
    code: "3451",
    name: "Verbindlichkeiten gegenüber Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3455: OdooAccountTemplate = OdooAccountTemplate {
    code: "3455",
    name: "Verbindlichkeiten gegenüber Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3460: OdooAccountTemplate = OdooAccountTemplate {
    code: "3460",
    name: "Verbindlichkeiten gegenüber Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3470: OdooAccountTemplate = OdooAccountTemplate {
    code: "3470",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber Unternehmen, mit denen ein Beteiligungsverhältnis besteht",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3471: OdooAccountTemplate = OdooAccountTemplate {
    code: "3471",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3475: OdooAccountTemplate = OdooAccountTemplate {
    code: "3475",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3480: OdooAccountTemplate = OdooAccountTemplate {
    code: "3480",
    name: "Verbindlichkeiten aus Lieferungen und Leistungen gegenüber Unternehmen, mit denen ein Beteiligungsverhältnis besteht - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3500: OdooAccountTemplate = OdooAccountTemplate {
    code: "3500",
    name: "Sonstige Verbindlichkeiten",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3501: OdooAccountTemplate = OdooAccountTemplate {
    code: "3501",
    name: "Sonstige Verbindlichkeiten - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3504: OdooAccountTemplate = OdooAccountTemplate {
    code: "3504",
    name: "Sonstige Verbindlichkeiten - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3507: OdooAccountTemplate = OdooAccountTemplate {
    code: "3507",
    name: "Sonstige Verbindlichkeiten - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3509: OdooAccountTemplate = OdooAccountTemplate {
    code: "3509",
    name: "Sonstige Verbindlichkeiten nach § 11 Abs. 2 Satz 2 EStG für § 4 Abs. 3 EStG",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3510: OdooAccountTemplate = OdooAccountTemplate {
    code: "3510",
    name: "Verbindlichkeiten gegenüber Gesellschaftern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3511: OdooAccountTemplate = OdooAccountTemplate {
    code: "3511",
    name: "Verbindlichkeiten gegenüber Gesellschaftern - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3514: OdooAccountTemplate = OdooAccountTemplate {
    code: "3514",
    name: "Verbindlichkeiten gegenüber Gesellschaftern - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3517: OdooAccountTemplate = OdooAccountTemplate {
    code: "3517",
    name: "Liabilities to shareholders/partners - remaining term greater than 5 years",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3519: OdooAccountTemplate = OdooAccountTemplate {
    code: "3519",
    name: "Verbindlichkeiten gegenüber Gesellschaftern für offene Ausschüttungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3520: OdooAccountTemplate = OdooAccountTemplate {
    code: "3520",
    name: "Verbindlichkeiten gegenüber typisch stillen Gesellschaftern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3521: OdooAccountTemplate = OdooAccountTemplate {
    code: "3521",
    name: "Verbindlichkeiten gegenüber typisch stillen Gesellschaftern - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3524: OdooAccountTemplate = OdooAccountTemplate {
    code: "3524",
    name: "Verbindlichkeiten gegenüber typisch stillen Gesellschaftern - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3527: OdooAccountTemplate = OdooAccountTemplate {
    code: "3527",
    name: "Verbindlichkeiten gegenüber typisch stillen Gesellschaftern - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3530: OdooAccountTemplate = OdooAccountTemplate {
    code: "3530",
    name: "Verbindlichkeiten gegenüber atypisch stillen Gesellschaftern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3531: OdooAccountTemplate = OdooAccountTemplate {
    code: "3531",
    name: "Verbindlichkeiten gegenüber atypisch stillen Gesellschaftern - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3534: OdooAccountTemplate = OdooAccountTemplate {
    code: "3534",
    name: "Verbindlichkeiten gegenüber atypisch stillen Gesellschaftern - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3537: OdooAccountTemplate = OdooAccountTemplate {
    code: "3537",
    name: "Verbindlichkeiten gegenüber atypisch stillen Gesellschaftern - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3540: OdooAccountTemplate = OdooAccountTemplate {
    code: "3540",
    name: "Partiarische Darlehen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3541: OdooAccountTemplate = OdooAccountTemplate {
    code: "3541",
    name: "Partiarische Darlehen - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3544: OdooAccountTemplate = OdooAccountTemplate {
    code: "3544",
    name: "Partiarische Darlehen - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3547: OdooAccountTemplate = OdooAccountTemplate {
    code: "3547",
    name: "Partiarische Darlehen - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3550: OdooAccountTemplate = OdooAccountTemplate {
    code: "3550",
    name: "Erhaltene Kautionen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3551: OdooAccountTemplate = OdooAccountTemplate {
    code: "3551",
    name: "Erhaltene Kautionen - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3554: OdooAccountTemplate = OdooAccountTemplate {
    code: "3554",
    name: "Erhaltene Kautionen - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3557: OdooAccountTemplate = OdooAccountTemplate {
    code: "3557",
    name: "Erhaltene Kautionen - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3560: OdooAccountTemplate = OdooAccountTemplate {
    code: "3560",
    name: "Darlehen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3561: OdooAccountTemplate = OdooAccountTemplate {
    code: "3561",
    name: "Darlehen - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3564: OdooAccountTemplate = OdooAccountTemplate {
    code: "3564",
    name: "Darlehen - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3567: OdooAccountTemplate = OdooAccountTemplate {
    code: "3567",
    name: "Darlehen - Restlaufzeit größer 5 Jahre",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3600: OdooAccountTemplate = OdooAccountTemplate {
    code: "3600",
    name: "Agenturwarenabrechnung",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3610: OdooAccountTemplate = OdooAccountTemplate {
    code: "3610",
    name: "Kreditkartenabrechnung",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3611: OdooAccountTemplate = OdooAccountTemplate {
    code: "3611",
    name: "Verbindlichkeiten gegenüber Arbeitsgemeinschaften",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3620: OdooAccountTemplate = OdooAccountTemplate {
    code: "3620",
    name: "Gewinnverfügungskonto stille Gesellschafter",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3630: OdooAccountTemplate = OdooAccountTemplate {
    code: "3630",
    name: "Sonstige Verrechnungskonten (Interimskonto)",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3635: OdooAccountTemplate = OdooAccountTemplate {
    code: "3635",
    name: "Sonstige Verbindlichkeiten aus genossenschaftlicher Rückvergütung",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3640: OdooAccountTemplate = OdooAccountTemplate {
    code: "3640",
    name: "Verbindlichkeiten gegenüber GmbH-Gesellschaftern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3641: OdooAccountTemplate = OdooAccountTemplate {
    code: "3641",
    name: "Verbindlichkeiten gegenüber GmbH-Gesellschaftern - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3642: OdooAccountTemplate = OdooAccountTemplate {
    code: "3642",
    name: "Verbindlichkeiten gegenüber GmbH-Gesellschaftern - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3643: OdooAccountTemplate = OdooAccountTemplate {
    code: "3643",
    name: "Verbindlichkeiten gegenüber GmbH-Gesellschaftern - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3645: OdooAccountTemplate = OdooAccountTemplate {
    code: "3645",
    name: "Verbindlichkeiten gegenüber persönlich haftenden Gesellschaftern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3646: OdooAccountTemplate = OdooAccountTemplate {
    code: "3646",
    name: "Verbindlichkeiten gegenüber persönlich haftenden Gesellschaftern - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3647: OdooAccountTemplate = OdooAccountTemplate {
    code: "3647",
    name: "Verbindlichkeiten gegenüber persönlich haftenden Gesellschaftern - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3648: OdooAccountTemplate = OdooAccountTemplate {
    code: "3648",
    name: "Verbindlichkeiten gegenüber persönlich haftenden Gesellschaftern - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3650: OdooAccountTemplate = OdooAccountTemplate {
    code: "3650",
    name: "Verbindlichkeiten gegenüber Kommanditisten",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3651: OdooAccountTemplate = OdooAccountTemplate {
    code: "3651",
    name: "Verbindlichkeiten gegenüber Kommanditisten - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3652: OdooAccountTemplate = OdooAccountTemplate {
    code: "3652",
    name: "Verbindlichkeiten gegenüber Kommanditisten - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3653: OdooAccountTemplate = OdooAccountTemplate {
    code: "3653",
    name: "Verbindlichkeiten gegenüber Kommanditisten - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3655: OdooAccountTemplate = OdooAccountTemplate {
    code: "3655",
    name: "Verbindlichkeiten gegenüber stillen Gesellschaftern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3656: OdooAccountTemplate = OdooAccountTemplate {
    code: "3656",
    name: "Verbindlichkeiten gegenüber stillen Gesellschaftern -  Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3657: OdooAccountTemplate = OdooAccountTemplate {
    code: "3657",
    name: "Verbindlichkeiten gegenüber stillen Gesellschaftern - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3658: OdooAccountTemplate = OdooAccountTemplate {
    code: "3658",
    name: "Verbindlichkeiten gegenüber stillen Gesellschaftern - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3695: OdooAccountTemplate = OdooAccountTemplate {
    code: "3695",
    name: "Verrechnungskonto geleistete Anzahlungen bei Buchung über Kreditorenkonto",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3700: OdooAccountTemplate = OdooAccountTemplate {
    code: "3700",
    name: "Verbindlichkeiten aus Steuern und Abgaben",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3701: OdooAccountTemplate = OdooAccountTemplate {
    code: "3701",
    name: "Verbindlichkeiten aus Steuern und Abgaben – Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3710: OdooAccountTemplate = OdooAccountTemplate {
    code: "3710",
    name: "Verbindlichkeiten aus Steuern und Abgaben – Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3715: OdooAccountTemplate = OdooAccountTemplate {
    code: "3715",
    name: "Verbindlichkeiten aus Steuern und Abgaben – Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3720: OdooAccountTemplate = OdooAccountTemplate {
    code: "3720",
    name: "Verbindlichkeiten aus Lohn und Gehalt",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3725: OdooAccountTemplate = OdooAccountTemplate {
    code: "3725",
    name: "Verbindlichkeiten für Einbehaltungen von Arbeitnehmern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3726: OdooAccountTemplate = OdooAccountTemplate {
    code: "3726",
    name: "Verbindlichkeiten an das Finanzamt aus abzuführendem Bauabzugsbetrag",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3730: OdooAccountTemplate = OdooAccountTemplate {
    code: "3730",
    name: "Verbindlichkeiten aus Lohn- und Kirchensteuer",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3740: OdooAccountTemplate = OdooAccountTemplate {
    code: "3740",
    name: "Verbindlichkeiten im Rahmen der sozialen Sicherheit",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3741: OdooAccountTemplate = OdooAccountTemplate {
    code: "3741",
    name: "erbindlichkeiten im Rahmen der sozialen Sicherheit - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3750: OdooAccountTemplate = OdooAccountTemplate {
    code: "3750",
    name: "Verbindlichkeiten im Rahmen der sozialen Sicherheit - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3755: OdooAccountTemplate = OdooAccountTemplate {
    code: "3755",
    name: "Verbindlichkeiten im Rahmen der sozialen Sicherheit - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3759: OdooAccountTemplate = OdooAccountTemplate {
    code: "3759",
    name: "Voraussichtliche Beitragsschuld gegenüber den Sozialversicherungsträgern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3760: OdooAccountTemplate = OdooAccountTemplate {
    code: "3760",
    name: "Verbindlichkeiten aus Einbehaltungen (KapESt und SolZ, KiSt auf KapESt) für offene Ausschüttungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3761: OdooAccountTemplate = OdooAccountTemplate {
    code: "3761",
    name: "Verbindlichkeiten für Verbrauchsteuern",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3770: OdooAccountTemplate = OdooAccountTemplate {
    code: "3770",
    name: "Verbindlichkeiten aus Vermögensbildung",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3771: OdooAccountTemplate = OdooAccountTemplate {
    code: "3771",
    name: "Verbindlichkeiten aus Vermögensbildung - Restlaufzeit bis 1 Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3780: OdooAccountTemplate = OdooAccountTemplate {
    code: "3780",
    name: "Verbindlichkeiten aus Vermögensbildung - Restlaufzeit 1 bis 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3785: OdooAccountTemplate = OdooAccountTemplate {
    code: "3785",
    name: "Verbindlichkeiten aus Vermögensbildung - Restlaufzeit größer 5 Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3786: OdooAccountTemplate = OdooAccountTemplate {
    code: "3786",
    name: "Ausgegebene Geschenkgutscheine",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3790: OdooAccountTemplate = OdooAccountTemplate {
    code: "3790",
    name: "Lohn- und Gehaltsverrechnungskonto",
    account_type: "asset_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3796: OdooAccountTemplate = OdooAccountTemplate {
    code: "3796",
    name: "Verbindlichkeiten im Rahmen der sozialen Sicherheit für § 4 Abs. 3 EStG",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3798: OdooAccountTemplate = OdooAccountTemplate {
    code: "3798",
    name: "Umsatzsteuer aus im anderen EU-Land steuerpflichtigen elektronischen Dienstleistungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3799: OdooAccountTemplate = OdooAccountTemplate {
    code: "3799",
    name: "Steuerzahlungen aus im anderen EU-Land steuerpflichtigen Dienstleistungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3800: OdooAccountTemplate = OdooAccountTemplate {
    code: "3800",
    name: "Umsatzsteuer",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3801: OdooAccountTemplate = OdooAccountTemplate {
    code: "3801",
    name: "USt 7%",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3802: OdooAccountTemplate = OdooAccountTemplate {
    code: "3802",
    name: "Umsatzsteuer aus innergemeinschaftlichem Erwerb",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3804: OdooAccountTemplate = OdooAccountTemplate {
    code: "3804",
    name: "Umsatzsteuer aus innergemeinschaftlichem Erwerb 19 %",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3806: OdooAccountTemplate = OdooAccountTemplate {
    code: "3806",
    name: "USt 19%",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3807: OdooAccountTemplate = OdooAccountTemplate {
    code: "3807",
    name: "Umsatzsteuer aus im Inland steuerpflichtigen EU-Lieferungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3808: OdooAccountTemplate = OdooAccountTemplate {
    code: "3808",
    name: "Umsatzsteuer aus im Inland steuerpflichtigen EU-Lieferungen 19 %",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3809: OdooAccountTemplate = OdooAccountTemplate {
    code: "3809",
    name: "Umsatzsteuer aus innergemeinschaftlichem Erwerb ohne Vorsteuerabzug",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3810: OdooAccountTemplate = OdooAccountTemplate {
    code: "3810",
    name: "Umsatzsteuer nicht fällig",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3811: OdooAccountTemplate = OdooAccountTemplate {
    code: "3811",
    name: "Umsatzsteuer nicht fällig 7 %",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3812: OdooAccountTemplate = OdooAccountTemplate {
    code: "3812",
    name: "Umsatzsteuer nicht fällig aus im Inland steuerpflichtigen EU-Lieferungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3814: OdooAccountTemplate = OdooAccountTemplate {
    code: "3814",
    name: "Umsatzsteuer nicht fällig aus im Inland steuerpflichtigen EU-Lieferungen 19 %",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3816: OdooAccountTemplate = OdooAccountTemplate {
    code: "3816",
    name: "Umsatzsteuer nicht fällig 19 %",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_B_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3817: OdooAccountTemplate = OdooAccountTemplate {
    code: "3817",
    name: "Umsatzsteuer aus im anderen EU-Land steuerpflichtigen Lieferungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3818: OdooAccountTemplate = OdooAccountTemplate {
    code: "3818",
    name: "Umsatzsteuer aus im anderen EU-Land steuerpflichtigen sonstigen Leistungen/Werklieferungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3819: OdooAccountTemplate = OdooAccountTemplate {
    code: "3819",
    name: "Umsatzsteuer aus Erwerb als letzter Abnehmer innerhalb eines Dreiecksgeschäfts",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3820: OdooAccountTemplate = OdooAccountTemplate {
    code: "3820",
    name: "Umsatzsteuer-Vorauszahlungen",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3830: OdooAccountTemplate = OdooAccountTemplate {
    code: "3830",
    name: "Umsatzsteuer-Vorauszahlungen 1/11",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3832: OdooAccountTemplate = OdooAccountTemplate {
    code: "3832",
    name: "Nachsteuer, UStVA Kz. 65",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3834: OdooAccountTemplate = OdooAccountTemplate {
    code: "3834",
    name: "Umsatzsteuer aus innergemeinschaftlichem Erwerb von Neufahrzeugen von Lieferanten ohne Umsatzsteuer-Identifikationsnummer",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3835: OdooAccountTemplate = OdooAccountTemplate {
    code: "3835",
    name: "Umsatzsteuer nach § 13b UStG",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3837: OdooAccountTemplate = OdooAccountTemplate {
    code: "3837",
    name: "Umsatzsteuer nach § 13b UStG 19 %",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3840: OdooAccountTemplate = OdooAccountTemplate {
    code: "3840",
    name: "Umsatzsteuer laufendes Jahr",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3841: OdooAccountTemplate = OdooAccountTemplate {
    code: "3841",
    name: "Umsatzsteuer Vorjahr",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3845: OdooAccountTemplate = OdooAccountTemplate {
    code: "3845",
    name: "Umsatzsteuer frühere Jahre",
    account_type: "liability_non_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3850: OdooAccountTemplate = OdooAccountTemplate {
    code: "3850",
    name: "Einfuhrumsatzsteuer aufgeschoben bis ...",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3851: OdooAccountTemplate = OdooAccountTemplate {
    code: "3851",
    name: "In Rechnung unrichtig oder unberechtigt ausgewiesene Steuerbeträge, UStVA Kz. 69",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_asset_bs_B_II_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3854: OdooAccountTemplate = OdooAccountTemplate {
    code: "3854",
    name: "Steuerzahlungen an andere Länder",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3860: OdooAccountTemplate = OdooAccountTemplate {
    code: "3860",
    name: "Verbindlichkeiten aus Umsatzsteuer-Vorauszahlungen",
    account_type: "liability_payable",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3865: OdooAccountTemplate = OdooAccountTemplate {
    code: "3865",
    name: "Umsatzsteuer in Folgeperiode fällig (§§ 13 Abs. 1 Nr. 6 und 13b Abs. 2 UStG)",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_C_8"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3900: OdooAccountTemplate = OdooAccountTemplate {
    code: "3900",
    name: "Passive Rechnungsabgrenzung",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_D"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_3950: OdooAccountTemplate = OdooAccountTemplate {
    code: "3950",
    name: "Abgrenzung unterjährig pauschal gebuchter Abschreibungen für BWA",
    account_type: "liability_current",
    tag_xmlids: &["l10n_de.tag_de_liabilities_bs_D"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/hgb/266"],
};

pub const EXT_SKR04_4000: OdooAccountTemplate = OdooAccountTemplate {
    code: "4000",
    name: "Umsatzerlöse (Zur freien Verfügung)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4100: OdooAccountTemplate = OdooAccountTemplate {
    code: "4100",
    name: "Steuerfreie Umsätze § 4 Nr. 8 ff. UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4110: OdooAccountTemplate = OdooAccountTemplate {
    code: "4110",
    name: "Sonstige steuerfreie Umsätze Inland",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4120: OdooAccountTemplate = OdooAccountTemplate {
    code: "4120",
    name: "Steuerfreie Umsätze nach § 4 Nr. 1a UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4125: OdooAccountTemplate = OdooAccountTemplate {
    code: "4125",
    name: "Steuerfreie innergemeinschaftliche Lieferungen nach § 4 Nr. 1b UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4130: OdooAccountTemplate = OdooAccountTemplate {
    code: "4130",
    name: "Lieferungen des ersten Abnehmers bei innergemeinschaftlichen Dreiecksgeschäften § 25b Abs. 2 UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4135: OdooAccountTemplate = OdooAccountTemplate {
    code: "4135",
    name: "Steuerfreie innergemeinschaftliche Lieferungen von Neufahrzeugen an Abnehmer ohne Umsatzsteuer-Identifikationsnummer",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4136: OdooAccountTemplate = OdooAccountTemplate {
    code: "4136",
    name: "Umsatzerlöse nach §§ 25 und 25a UStG 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4138: OdooAccountTemplate = OdooAccountTemplate {
    code: "4138",
    name: "Umsatzerlöse nach §§ 25 und 25a UStG ohne USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4139: OdooAccountTemplate = OdooAccountTemplate {
    code: "4139",
    name: "Umsatzerlöse aus Reiseleistungen § 25 Abs. 2 UStG, steuerfrei",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4140: OdooAccountTemplate = OdooAccountTemplate {
    code: "4140",
    name: "Steuerfreie Umsätze Offshore etc.",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4150: OdooAccountTemplate = OdooAccountTemplate {
    code: "4150",
    name: "Sonstige steuerfreie Umsätze (z. B. § 4 Nr. 2 bis 7 UStG)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4160: OdooAccountTemplate = OdooAccountTemplate {
    code: "4160",
    name: "Steuerfreie Umsätze ohne Vorsteuerabzug zum Gesamtumsatz gehörend, § 4 UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4165: OdooAccountTemplate = OdooAccountTemplate {
    code: "4165",
    name: "Steuerfreie Umsätze ohne Vorsteuerabzug zum Gesamtumsatz gehörend",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4180: OdooAccountTemplate = OdooAccountTemplate {
    code: "4180",
    name: "Erlöse, die mit den Durchschnittssätzen des § 24 UStG versteuert werden",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4185: OdooAccountTemplate = OdooAccountTemplate {
    code: "4185",
    name: "Erlöse als Kleinunternehmer nach § 19 Abs. 1 UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4186: OdooAccountTemplate = OdooAccountTemplate {
    code: "4186",
    name: "Erlöse aus Geldspielautomaten 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4200: OdooAccountTemplate = OdooAccountTemplate {
    code: "4200",
    name: "Erlöse",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4300: OdooAccountTemplate = OdooAccountTemplate {
    code: "4300",
    name: "Erlöse 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4310: OdooAccountTemplate = OdooAccountTemplate {
    code: "4310",
    name: "Erlöse aus im Inland steuerpflichtigen EU-Lieferungen 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4315: OdooAccountTemplate = OdooAccountTemplate {
    code: "4315",
    name: "Erlöse aus im Inland steuerpflichtigen EU-Lieferungen 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4320: OdooAccountTemplate = OdooAccountTemplate {
    code: "4320",
    name: "Erlöse aus im anderen EU-Land steuerbaren Leistungen, im Inland nicht steuerbare Umsätze",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4331: OdooAccountTemplate = OdooAccountTemplate {
    code: "4331",
    name: "Erlöse aus im anderen EU-Land steuerpflichtigen elektronischen Dienstleistungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4335: OdooAccountTemplate = OdooAccountTemplate {
    code: "4335",
    name: "Erlöse aus Lieferungen von Mobilfunkgeräten, Tablet-Computern, Spielekonsolen und integrierten Schaltkreisen, für die der Leistungsempfänger die Umsatzsteuer nach § 13b UStG schuldet",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4336: OdooAccountTemplate = OdooAccountTemplate {
    code: "4336",
    name: "Erlöse aus im anderen EU-Land steuerpflichtigen sonstigen Leistungen, für die der Leistungsempfänger die Umsatzsteuer schuldet",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4337: OdooAccountTemplate = OdooAccountTemplate {
    code: "4337",
    name: "Erlöse aus Leistungen, für die der Leistungsempfänger die Umsatzsteuer nach § 13b UStG schuldet",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4338: OdooAccountTemplate = OdooAccountTemplate {
    code: "4338",
    name: "Erlöse aus im Drittland steuerbaren Leistungen, im Inland nicht steuerbare Umsätze",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4339: OdooAccountTemplate = OdooAccountTemplate {
    code: "4339",
    name: "Erlöse aus im anderen EU-Land steuerbaren Leistungen, im Inland nicht steuerbare Umsätze",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4400: OdooAccountTemplate = OdooAccountTemplate {
    code: "4400",
    name: "Erlöse 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4499: OdooAccountTemplate = OdooAccountTemplate {
    code: "4499",
    name: "Nebenerlöse (Bezug zu Materialaufwand)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4510: OdooAccountTemplate = OdooAccountTemplate {
    code: "4510",
    name: "Erlöse Abfallverwertung",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4520: OdooAccountTemplate = OdooAccountTemplate {
    code: "4520",
    name: "Erlöse Leergut",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4560: OdooAccountTemplate = OdooAccountTemplate {
    code: "4560",
    name: "Provisionsumsätze",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4564: OdooAccountTemplate = OdooAccountTemplate {
    code: "4564",
    name: "Provisionsumsätze, steuerfrei § 4 Nr. 8 ff. UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4566: OdooAccountTemplate = OdooAccountTemplate {
    code: "4566",
    name: "Provisionsumsätze 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4569: OdooAccountTemplate = OdooAccountTemplate {
    code: "4569",
    name: "Provisionsumsätze 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4570: OdooAccountTemplate = OdooAccountTemplate {
    code: "4570",
    name: "Sonstige Erträge aus Provisionen, Lizenzen und Patenten",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4574: OdooAccountTemplate = OdooAccountTemplate {
    code: "4574",
    name: "Sonstige Erträge aus Provisionen, Lizenzen und Patenten, steuerfrei § 4 Nr. 8 ff. UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4575: OdooAccountTemplate = OdooAccountTemplate {
    code: "4575",
    name: "Sonstige Erträge aus Provisionen, Lizenzen und Patenten, steuerfrei § 4 Nr. 5 UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4576: OdooAccountTemplate = OdooAccountTemplate {
    code: "4576",
    name: "Sonstige Erträge aus Provisionen, Lizenzen und Patenten 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4579: OdooAccountTemplate = OdooAccountTemplate {
    code: "4579",
    name: "Sonstige Erträge aus Provisionen, Lizenzen und Patenten 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4600: OdooAccountTemplate = OdooAccountTemplate {
    code: "4600",
    name: "Unentgeltliche Wertabgaben",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4605: OdooAccountTemplate = OdooAccountTemplate {
    code: "4605",
    name: "Entnahme von Gegenständen ohne USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4610: OdooAccountTemplate = OdooAccountTemplate {
    code: "4610",
    name: "Entnahme durch den Unternehmer für Zwecke außerhalb des Unternehmens (Waren) 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4616: OdooAccountTemplate = OdooAccountTemplate {
    code: "4616",
    name: "Entnahme durch den Unternehmer für Zwecke außerhalb des Unternehmens (Waren) 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4619: OdooAccountTemplate = OdooAccountTemplate {
    code: "4619",
    name: "Entnahme durch den Unternehmer für Zwecke außerhalb des Unternehmens (Waren) ohne USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4620: OdooAccountTemplate = OdooAccountTemplate {
    code: "4620",
    name: "Entnahme durch den Unternehmer für Zwecke außerhalb des Unternehmens (Waren) 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4630: OdooAccountTemplate = OdooAccountTemplate {
    code: "4630",
    name: "Verwendung von Gegenständen für Zwecke außerhalb des Unternehmens 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4639: OdooAccountTemplate = OdooAccountTemplate {
    code: "4639",
    name: "Verwendung von Gegenständen für Zwecke außerhalb des Unternehmens ohne USt (Kfz-Nutzung)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4640: OdooAccountTemplate = OdooAccountTemplate {
    code: "4640",
    name: "Verwendung von Gegenständen für Zwecke außerhalb des Unternehmens 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4645: OdooAccountTemplate = OdooAccountTemplate {
    code: "4645",
    name: "Verwendung von Gegenständen für Zwecke außerhalb des Unternehmens 19 % USt (Kfz-Nutzung)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4646: OdooAccountTemplate = OdooAccountTemplate {
    code: "4646",
    name: "Verwendung von Gegenständen für Zwecke außerhalb des Unternehmens 19 % USt (Telefon-Nutzung)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4650: OdooAccountTemplate = OdooAccountTemplate {
    code: "4650",
    name: "Unentgeltliche Erbringung einer sonstigen Leistung 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4659: OdooAccountTemplate = OdooAccountTemplate {
    code: "4659",
    name: "Unentgeltliche Erbringung einer sonstigen Leistung ohne USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4660: OdooAccountTemplate = OdooAccountTemplate {
    code: "4660",
    name: "Unentgeltliche Erbringung einer sonstigen Leistung 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4670: OdooAccountTemplate = OdooAccountTemplate {
    code: "4670",
    name: "Unentgeltliche Zuwendung von Waren 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4679: OdooAccountTemplate = OdooAccountTemplate {
    code: "4679",
    name: "Unentgeltliche Zuwendung von Waren ohne USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4680: OdooAccountTemplate = OdooAccountTemplate {
    code: "4680",
    name: "Unentgeltliche Zuwendung von Waren 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4686: OdooAccountTemplate = OdooAccountTemplate {
    code: "4686",
    name: "Unentgeltliche Zuwendung von Gegenständen 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4689: OdooAccountTemplate = OdooAccountTemplate {
    code: "4689",
    name: "Unentgeltliche Zuwendung von Gegenständen ohne USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4690: OdooAccountTemplate = OdooAccountTemplate {
    code: "4690",
    name: "Nicht steuerbare Umsätze (Innenumsätze)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4695: OdooAccountTemplate = OdooAccountTemplate {
    code: "4695",
    name: "Umsatzsteuervergütungen, z. B. nach § 24 UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4699: OdooAccountTemplate = OdooAccountTemplate {
    code: "4699",
    name: "Direkt mit dem Umsatz verbundene Steuern",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4700: OdooAccountTemplate = OdooAccountTemplate {
    code: "4700",
    name: "Erlösschmälerungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4701: OdooAccountTemplate = OdooAccountTemplate {
    code: "4701",
    name: "Erlösschmälerungen für steuerfreie Umsätze nach § 4 Nr. 8 ff. UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4702: OdooAccountTemplate = OdooAccountTemplate {
    code: "4702",
    name: "Erlösschmälerungen für steuerfreie Umsätze nach § 4 Nr. 2 bis 7 UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4703: OdooAccountTemplate = OdooAccountTemplate {
    code: "4703",
    name: "Erlösschmälerungen für sonstige steuerfreie Umsätze ohne Vorsteuerabzug",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4704: OdooAccountTemplate = OdooAccountTemplate {
    code: "4704",
    name: "Erlösschmälerungen für sonstige steuerfreie Umsätze mit Vorsteuerabzug",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4705: OdooAccountTemplate = OdooAccountTemplate {
    code: "4705",
    name: "Erlösschmälerungen aus steuerfreien Umsätzen § 4 Nr. 1a UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4710: OdooAccountTemplate = OdooAccountTemplate {
    code: "4710",
    name: "Erlösschmälerungen 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4720: OdooAccountTemplate = OdooAccountTemplate {
    code: "4720",
    name: "Erlösschmälerungen 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4724: OdooAccountTemplate = OdooAccountTemplate {
    code: "4724",
    name: "Erlösschmälerungen aus steuerfreien innergemeinschaftlichen Lieferungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4725: OdooAccountTemplate = OdooAccountTemplate {
    code: "4725",
    name: "Erlösschmälerungen aus im Inland steuerpflichtigen EU-Lieferungen 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4726: OdooAccountTemplate = OdooAccountTemplate {
    code: "4726",
    name: "Erlösschmälerungen aus im Inland steuerpflichtigen EU-Lieferungen 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4727: OdooAccountTemplate = OdooAccountTemplate {
    code: "4727",
    name: "Erlösschmälerungen aus im anderen EU-Land steuerpflichtigen Lieferungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4730: OdooAccountTemplate = OdooAccountTemplate {
    code: "4730",
    name: "Gewährte Skonti",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4731: OdooAccountTemplate = OdooAccountTemplate {
    code: "4731",
    name: "Gewährte Skonti 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4736: OdooAccountTemplate = OdooAccountTemplate {
    code: "4736",
    name: "Gewährte Skonti 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4738: OdooAccountTemplate = OdooAccountTemplate {
    code: "4738",
    name: "Gewährte Skonti aus Lieferungen von Mobilfunkgeräten etc., für die der Leistungsempfänger die Umsatzsteuer nach § 13b Abs. 2 Nr. 10 UStG schuldet",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4741: OdooAccountTemplate = OdooAccountTemplate {
    code: "4741",
    name: "Gewährte Skonti aus Leistungen, für die der Leistungsempfänger die Umsatzsteuer nach § 13b UStG schuldet",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4742: OdooAccountTemplate = OdooAccountTemplate {
    code: "4742",
    name: "Gewährte Skonti aus Erlösen aus im anderen EU-Land steuerpflichtigen sonstigen Leistungen, für die der Leistungsempfänger die Umsatzsteuer schuldet",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4743: OdooAccountTemplate = OdooAccountTemplate {
    code: "4743",
    name: "Gewährte Skonti aus steuerfreien innergemeinschaftlichen Lieferungen § 4 Nr. 1b UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4745: OdooAccountTemplate = OdooAccountTemplate {
    code: "4745",
    name: "Gewährte Skonti aus im Inland steuerpflichtigen EU-Lieferungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4746: OdooAccountTemplate = OdooAccountTemplate {
    code: "4746",
    name: "Gewährte Skonti aus im Inland steuerpflichtigen EU-Lieferungen 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4748: OdooAccountTemplate = OdooAccountTemplate {
    code: "4748",
    name: "Gewährte Skonti aus im Inland steuerpflichtigen EU-Lieferungen 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4750: OdooAccountTemplate = OdooAccountTemplate {
    code: "4750",
    name: "Gewährte Boni 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4760: OdooAccountTemplate = OdooAccountTemplate {
    code: "4760",
    name: "Gewährte Boni 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4770: OdooAccountTemplate = OdooAccountTemplate {
    code: "4770",
    name: "Gewährte Rabatte",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4780: OdooAccountTemplate = OdooAccountTemplate {
    code: "4780",
    name: "Gewährte Rabatte 7 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4790: OdooAccountTemplate = OdooAccountTemplate {
    code: "4790",
    name: "Gewährte Rabatte 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_01"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4800: OdooAccountTemplate = OdooAccountTemplate {
    code: "4800",
    name: "Bestandsveränderungen - fertige Erzeugnisse",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_02"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4810: OdooAccountTemplate = OdooAccountTemplate {
    code: "4810",
    name: "Bestandsveränderungen - unfertige Erzeugnisse",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_02"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4815: OdooAccountTemplate = OdooAccountTemplate {
    code: "4815",
    name: "Bestandsveränderungen - unfertige Leistungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_02"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4816: OdooAccountTemplate = OdooAccountTemplate {
    code: "4816",
    name: "Bestandsveränderungen in Ausführung befindlicher Bauaufträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_02"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4818: OdooAccountTemplate = OdooAccountTemplate {
    code: "4818",
    name: "Bestandsveränderungen in Arbeit befindlicher Aufträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_02"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4820: OdooAccountTemplate = OdooAccountTemplate {
    code: "4820",
    name: "Andere aktivierte Eigenleistungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_03"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4824: OdooAccountTemplate = OdooAccountTemplate {
    code: "4824",
    name: "Aktivierte Eigenleistungen (den Herstellungskosten zurechenbare Fremdkapitalzinsen)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_03"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4825: OdooAccountTemplate = OdooAccountTemplate {
    code: "4825",
    name: "Aktivierte Eigenleistungen zur Erstellung von selbst geschaffenen immateriellen Vermögensgegenständen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_03"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4830: OdooAccountTemplate = OdooAccountTemplate {
    code: "4830",
    name: "Sonstige betriebliche Erträge",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4832: OdooAccountTemplate = OdooAccountTemplate {
    code: "4832",
    name: "Sonstige betriebliche Erträge von verbundenen Unternehmen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4833: OdooAccountTemplate = OdooAccountTemplate {
    code: "4833",
    name: "Andere Nebenerlöse",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4835: OdooAccountTemplate = OdooAccountTemplate {
    code: "4835",
    name: "Sonstige Erträge betrieblich und regelmäßig",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4836: OdooAccountTemplate = OdooAccountTemplate {
    code: "4836",
    name: "Sonstige Erträge betrieblich und regelmäßig 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4837: OdooAccountTemplate = OdooAccountTemplate {
    code: "4837",
    name: "Sonstige Erträge betriebsfremd und regelmäßig",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4838: OdooAccountTemplate = OdooAccountTemplate {
    code: "4838",
    name: "Erstattete Vorsteuer anderer Länder",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4839: OdooAccountTemplate = OdooAccountTemplate {
    code: "4839",
    name: "Sonstige Erträge unregelmäßig",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4840: OdooAccountTemplate = OdooAccountTemplate {
    code: "4840",
    name: "Erträge aus der Währungsumrechnung",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4841: OdooAccountTemplate = OdooAccountTemplate {
    code: "4841",
    name: "Sonstige Erträge betrieblich und regelmäßig, steuerfrei § 4 Nr. 8 ff. UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4842: OdooAccountTemplate = OdooAccountTemplate {
    code: "4842",
    name: "Sonstige betriebliche Erträge, steuerfrei z. B. § 4 Nr. 2 bis 7 UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4843: OdooAccountTemplate = OdooAccountTemplate {
    code: "4843",
    name: "Erträge aus Bewertung Finanzmittelfonds",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4844: OdooAccountTemplate = OdooAccountTemplate {
    code: "4844",
    name: "Erlöse aus Verkäufen Sachanlagevermögen steuerfrei § 4 Nr. 1a UStG (bei Buchgewinn)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4845: OdooAccountTemplate = OdooAccountTemplate {
    code: "4845",
    name: "Erlöse aus Verkäufen Sachanlagevermögen 19 % USt (bei Buchgewinn)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4847: OdooAccountTemplate = OdooAccountTemplate {
    code: "4847",
    name: "Erträge aus der Währungsumrechnung (nicht § 256a HGB)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4848: OdooAccountTemplate = OdooAccountTemplate {
    code: "4848",
    name: "Erlöse aus Verkäufen Sachanlagevermögen steuerfrei § 4 Nr. 1b UStG (bei Buchgewinn)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4849: OdooAccountTemplate = OdooAccountTemplate {
    code: "4849",
    name: "Erlöse aus Verkäufen Sachanlagevermögen (bei Buchgewinn)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4850: OdooAccountTemplate = OdooAccountTemplate {
    code: "4850",
    name: "Erlöse aus Verkäufen immaterieller Vermögensgegenstände (bei Buchgewinn)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4851: OdooAccountTemplate = OdooAccountTemplate {
    code: "4851",
    name: "Erlöse aus Verkäufen Finanzanlagen (bei Buchgewinn)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4852: OdooAccountTemplate = OdooAccountTemplate {
    code: "4852",
    name: "Erlöse aus Verkäufen Finanzanlagen § 3 Nr. 40 EStG bzw. § 8b Abs. 2 KStG (bei Buchgewinn)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4855: OdooAccountTemplate = OdooAccountTemplate {
    code: "4855",
    name: "Anlagenabgänge Sachanlagen (Restbuchwert bei Buchgewinn)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4856: OdooAccountTemplate = OdooAccountTemplate {
    code: "4856",
    name: "Anlagenabgänge immaterielle Vermögensgegenstände (Restbuchwert bei Buchgewinn)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4857: OdooAccountTemplate = OdooAccountTemplate {
    code: "4857",
    name: "Anlagenabgänge Finanzanlagen (Restbuchwert bei Buchgewinn)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4858: OdooAccountTemplate = OdooAccountTemplate {
    code: "4858",
    name: "Anlagenabgänge Finanzanlagen § 3 Nr. 40 EStG bzw. § 8b Abs. 2 KStG (Restbuchwert bei Buchgewinn)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4860: OdooAccountTemplate = OdooAccountTemplate {
    code: "4860",
    name: "Grundstückserträge",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4861: OdooAccountTemplate = OdooAccountTemplate {
    code: "4861",
    name: "Erlöse aus Vermietung und Verpachtung, umsatzsteuerfrei § 4 Nr. 12 UStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4862: OdooAccountTemplate = OdooAccountTemplate {
    code: "4862",
    name: "Erlöse aus Vermietung und Verpachtung 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4900: OdooAccountTemplate = OdooAccountTemplate {
    code: "4900",
    name: "Erträge aus dem Abgang von Gegenständen des Anlagevermögens",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4901: OdooAccountTemplate = OdooAccountTemplate {
    code: "4901",
    name: "Erträge aus der Veräußerung von Anteilen an Kapitalgesellschaften (Finanzanlagevermögen) § 3 Nr. 40 EStG bzw. § 8b Abs. 2 KStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4905: OdooAccountTemplate = OdooAccountTemplate {
    code: "4905",
    name: "Erträge aus dem Abgang von Gegenständen des Umlaufvermögens außer Vorräte",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4906: OdooAccountTemplate = OdooAccountTemplate {
    code: "4906",
    name: "Erträge aus dem Abgang von Gegenständen des Umlaufvermögens (außer Vorräte) § 3 Nr. 40 EStG bzw. § 8b Abs. 2 KStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4910: OdooAccountTemplate = OdooAccountTemplate {
    code: "4910",
    name: "Erträge aus Zuschreibungen des Sachanlagevermögens",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4911: OdooAccountTemplate = OdooAccountTemplate {
    code: "4911",
    name: "Erträge aus Zuschreibungen des immateriellen Anlagevermögens",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4912: OdooAccountTemplate = OdooAccountTemplate {
    code: "4912",
    name: "Erträge aus Zuschreibungen des Finanzanlagevermögens",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4913: OdooAccountTemplate = OdooAccountTemplate {
    code: "4913",
    name: "Erträge aus Zuschreibungen des Finanzanlagevermögens § 3 Nr. 40 EStG bzw. § 8b Abs. 3 Satz 8 KStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4914: OdooAccountTemplate = OdooAccountTemplate {
    code: "4914",
    name: "Erträge aus Zuschreibungen § 3 Nr. 40 EStG bzw. § 8b Abs. 2 KStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4915: OdooAccountTemplate = OdooAccountTemplate {
    code: "4915",
    name: "Erträge aus Zuschreibungen des Umlaufvermögens (außer Vorräte)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4916: OdooAccountTemplate = OdooAccountTemplate {
    code: "4916",
    name: "Erträge aus Zuschreibungen des Umlaufvermögens § 3 Nr. 40 EStG bzw. § 8b Abs. 3 Satz 8 KStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4920: OdooAccountTemplate = OdooAccountTemplate {
    code: "4920",
    name: "Erträge aus der Herabsetzung der Pauschalwertberichtigung auf Forderungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4923: OdooAccountTemplate = OdooAccountTemplate {
    code: "4923",
    name: "Erträge aus der Herabsetzung der Einzelwertberichtigung auf Forderungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4925: OdooAccountTemplate = OdooAccountTemplate {
    code: "4925",
    name: "Erträge aus abgeschriebenen Forderungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4927: OdooAccountTemplate = OdooAccountTemplate {
    code: "4927",
    name: "Erträge aus der Auflösung einer steuerlichen Rücklage nach § 6b Abs. 3 EStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4928: OdooAccountTemplate = OdooAccountTemplate {
    code: "4928",
    name: "Erträge aus der Auflösung einer steuerlichen Rücklage nach § 6b Abs. 10 EStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4929: OdooAccountTemplate = OdooAccountTemplate {
    code: "4929",
    name: "Erträge aus der Auflösung der Rücklage für Ersatzbeschaffung, R 6.6 EStR",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4930: OdooAccountTemplate = OdooAccountTemplate {
    code: "4930",
    name: "Erträge aus der Auflösung von Rückstellungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4932: OdooAccountTemplate = OdooAccountTemplate {
    code: "4932",
    name: "Erträge aus der Herabsetzung von Verbindlichkeiten",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4934: OdooAccountTemplate = OdooAccountTemplate {
    code: "4934",
    name: "Erträge aus der Auflösung einer Anlaufrücklage",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4935: OdooAccountTemplate = OdooAccountTemplate {
    code: "4935",
    name: "Erträge aus der Auflösung sonstiger steuerlicher Rücklagen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4936: OdooAccountTemplate = OdooAccountTemplate {
    code: "4936",
    name: "Erträge aus der Auflösung von Rückstellungen (kumulierte Abschreibungen nach § 7g Abs. 2 EStG)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4937: OdooAccountTemplate = OdooAccountTemplate {
    code: "4937",
    name: "Erträge aus der Auflösung steuerrechtlicher Sonderabschreibungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4938: OdooAccountTemplate = OdooAccountTemplate {
    code: "4938",
    name: "Erträge aus der Auflösung einer steuerlichen Rücklage nach § 4g EStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4939: OdooAccountTemplate = OdooAccountTemplate {
    code: "4939",
    name: "Erträge Auflösung von Steuerrückstellungen § 52 (16) EStG",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4940: OdooAccountTemplate = OdooAccountTemplate {
    code: "4940",
    name: "Verrechnete sonstige Sachbezüge (keine Waren)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4941: OdooAccountTemplate = OdooAccountTemplate {
    code: "4941",
    name: "Sachbezüge 7 % USt (Waren)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4945: OdooAccountTemplate = OdooAccountTemplate {
    code: "4945",
    name: "Sachbezüge 19 % USt (Waren)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4947: OdooAccountTemplate = OdooAccountTemplate {
    code: "4947",
    name: "Verrechnete sonstige Sachbezüge aus Kfz-Gestellung 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4948: OdooAccountTemplate = OdooAccountTemplate {
    code: "4948",
    name: "Verrechnete sonstige Sachbezüge 19 % USt",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4949: OdooAccountTemplate = OdooAccountTemplate {
    code: "4949",
    name: "Verrechnete sonstige Sachbezüge ohne Umsatzsteuer",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4960: OdooAccountTemplate = OdooAccountTemplate {
    code: "4960",
    name: "Periodenfremde Erträge",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4970: OdooAccountTemplate = OdooAccountTemplate {
    code: "4970",
    name: "Insurance recoveries and compensation payments",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4972: OdooAccountTemplate = OdooAccountTemplate {
    code: "4972",
    name: "Erstattungen Aufwendungsausgleichsgesetz",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4975: OdooAccountTemplate = OdooAccountTemplate {
    code: "4975",
    name: "Investitionszuschüsse (steuerpflichtig)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4980: OdooAccountTemplate = OdooAccountTemplate {
    code: "4980",
    name: "Investitionszulagen (steuerfrei)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4981: OdooAccountTemplate = OdooAccountTemplate {
    code: "4981",
    name: "Steuerfreie Erträge aus der Auflösung von steuerlichen Rücklagen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4982: OdooAccountTemplate = OdooAccountTemplate {
    code: "4982",
    name: "Sonstige steuerfreie Betriebseinnahmen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4987: OdooAccountTemplate = OdooAccountTemplate {
    code: "4987",
    name: "Erträge aus der Aktivierung unentgeltlich erworbener Vermögensgegenstände",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4989: OdooAccountTemplate = OdooAccountTemplate {
    code: "4989",
    name: "Kostenerstattungen, Rückvergütungen und Gutschriften für frühere Jahre",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_4992: OdooAccountTemplate = OdooAccountTemplate {
    code: "4992",
    name: "Erträge aus Verwaltungskostenumlagen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_5000: OdooAccountTemplate = OdooAccountTemplate {
    code: "5000",
    name: "Aufwendungen für Roh-, Hilfsund Betriebsstoffe und für bezogene Waren",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5100: OdooAccountTemplate = OdooAccountTemplate {
    code: "5100",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5110: OdooAccountTemplate = OdooAccountTemplate {
    code: "5110",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5130: OdooAccountTemplate = OdooAccountTemplate {
    code: "5130",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5160: OdooAccountTemplate = OdooAccountTemplate {
    code: "5160",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe, innergemeinschaftlicher Erwerb 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5162: OdooAccountTemplate = OdooAccountTemplate {
    code: "5162",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe, innergemeinschaftlicher Erwerb 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5166: OdooAccountTemplate = OdooAccountTemplate {
    code: "5166",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe, innergemeinschaftlicher Erwerb ohne Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5167: OdooAccountTemplate = OdooAccountTemplate {
    code: "5167",
    name: "Einkauf Roh-, Hilfs- und Betriebsstoffe, innergemeinschaftlicher Erwerb ohne Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5189: OdooAccountTemplate = OdooAccountTemplate {
    code: "5189",
    name: "Erwerb Roh-, Hilfs- und Betriebsstoffe als letzter Abnehmer innerhalb Dreiecksgeschäft 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5190: OdooAccountTemplate = OdooAccountTemplate {
    code: "5190",
    name: "Energiestoffe (Fertigung)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5191: OdooAccountTemplate = OdooAccountTemplate {
    code: "5191",
    name: "Energiestoffe (Fertigung) 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5192: OdooAccountTemplate = OdooAccountTemplate {
    code: "5192",
    name: "Energiestoffe (Fertigung) 19 % Vorsteue",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5200: OdooAccountTemplate = OdooAccountTemplate {
    code: "5200",
    name: "Wareneingang",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5300: OdooAccountTemplate = OdooAccountTemplate {
    code: "5300",
    name: "Wareneingang 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5349: OdooAccountTemplate = OdooAccountTemplate {
    code: "5349",
    name: "Wareneingang ohne Vorsteuerabzug",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5400: OdooAccountTemplate = OdooAccountTemplate {
    code: "5400",
    name: "Wareneingang 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5420: OdooAccountTemplate = OdooAccountTemplate {
    code: "5420",
    name: "Innergemeinschaftlicher Erwerb 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5425: OdooAccountTemplate = OdooAccountTemplate {
    code: "5425",
    name: "Innergemeinschaftlicher Erwerb 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5430: OdooAccountTemplate = OdooAccountTemplate {
    code: "5430",
    name: "Innergemeinschaftlicher Erwerb ohne Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &[],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5435: OdooAccountTemplate = OdooAccountTemplate {
    code: "5435",
    name: "Innergemeinschaftlicher Erwerb ohne Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &[],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5440: OdooAccountTemplate = OdooAccountTemplate {
    code: "5440",
    name: "Innergemeinschaftlicher Erwerb von Neufahrzeugen von Lieferanten ohne Umsatzsteuer-Identifikationsnummer 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5550: OdooAccountTemplate = OdooAccountTemplate {
    code: "5550",
    name: "Steuerfreier innergemeinschaftlicher Erwerb",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5551: OdooAccountTemplate = OdooAccountTemplate {
    code: "5551",
    name: "Wareneingang im Drittland steuerbar",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5552: OdooAccountTemplate = OdooAccountTemplate {
    code: "5552",
    name: "Erwerb 1. Abnehmer innerhalb eines Dreiecksgeschäftes",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5553: OdooAccountTemplate = OdooAccountTemplate {
    code: "5553",
    name: "Erwerb Waren als letzter Abnehmer innerhalb Dreiecksgeschäft 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5557: OdooAccountTemplate = OdooAccountTemplate {
    code: "5557",
    name: "Wareneingang, steuerpflichtig im Drittland (7%)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5559: OdooAccountTemplate = OdooAccountTemplate {
    code: "5559",
    name: "Steuerfreie Einfuhren",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5600: OdooAccountTemplate = OdooAccountTemplate {
    code: "5600",
    name: "Nicht abziehbare Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5610: OdooAccountTemplate = OdooAccountTemplate {
    code: "5610",
    name: "Nicht abziehbare Vorsteuer 7 %",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5660: OdooAccountTemplate = OdooAccountTemplate {
    code: "5660",
    name: "Nicht abziehbare Vorsteuer 19 %",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5700: OdooAccountTemplate = OdooAccountTemplate {
    code: "5700",
    name: "Nachlässe",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5701: OdooAccountTemplate = OdooAccountTemplate {
    code: "5701",
    name: "Nachlässe aus Einkauf Roh-, Hilfsund Betriebsstoffe",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5710: OdooAccountTemplate = OdooAccountTemplate {
    code: "5710",
    name: "Nachlässe 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5714: OdooAccountTemplate = OdooAccountTemplate {
    code: "5714",
    name: "Nachlässe aus Einkauf Roh-, Hilfsund Betriebsstoffe 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5715: OdooAccountTemplate = OdooAccountTemplate {
    code: "5715",
    name: "Nachlässe aus Einkauf Roh-, Hilfsund Betriebsstoffe 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5717: OdooAccountTemplate = OdooAccountTemplate {
    code: "5717",
    name: "Nachlässe aus Einkauf Roh-, Hilfsund Betriebsstoffe, innergemeinschaftlicher Erwerb 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5718: OdooAccountTemplate = OdooAccountTemplate {
    code: "5718",
    name: "Nachlässe aus Einkauf Roh-, Hilfsund Betriebsstoffe, innergemeinschaftlicher Erwerb 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5720: OdooAccountTemplate = OdooAccountTemplate {
    code: "5720",
    name: "Nachlässe 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5724: OdooAccountTemplate = OdooAccountTemplate {
    code: "5724",
    name: "Nachlässe aus innergemeinschaftlichem Erwerb 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5725: OdooAccountTemplate = OdooAccountTemplate {
    code: "5725",
    name: "Nachlässe aus innergemeinschaftlichem Erwerb 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5730: OdooAccountTemplate = OdooAccountTemplate {
    code: "5730",
    name: "Erhaltene Skonti",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5731: OdooAccountTemplate = OdooAccountTemplate {
    code: "5731",
    name: "Erhaltene Skonti 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5733: OdooAccountTemplate = OdooAccountTemplate {
    code: "5733",
    name: "Erhaltene Skonti aus Einkauf Roh-, Hilfs- und Betriebsstoffe",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5734: OdooAccountTemplate = OdooAccountTemplate {
    code: "5734",
    name: "Erhaltene Skonti aus Einkauf Roh-, Hilfs- und Betriebsstoffe 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5736: OdooAccountTemplate = OdooAccountTemplate {
    code: "5736",
    name: "Erhaltene Skonti 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5738: OdooAccountTemplate = OdooAccountTemplate {
    code: "5738",
    name: "Erhaltene Skonti aus Einkauf Roh-, Hilfs- und Betriebsstoffe 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5741: OdooAccountTemplate = OdooAccountTemplate {
    code: "5741",
    name: "Erhaltene Skonti aus Einkauf Roh-, Hilfs- und Betriebsstoffe aus steuerpflichtigem innergemeinschaftlichem Erwerb 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5743: OdooAccountTemplate = OdooAccountTemplate {
    code: "5743",
    name: "Erhaltene Skonti aus Einkauf Roh-, Hilfs- und Betriebsstoffe aus steuerpflichtigem innergemeinschaftlichem Erwerb 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5744: OdooAccountTemplate = OdooAccountTemplate {
    code: "5744",
    name: "Erhaltene Skonti aus Einkauf Roh-, Hilfs- und Betriebsstoffe aus steuerpflichtigem innergemeinschaftlichem Erwerb",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5750: OdooAccountTemplate = OdooAccountTemplate {
    code: "5750",
    name: "Erhaltene Boni 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5753: OdooAccountTemplate = OdooAccountTemplate {
    code: "5753",
    name: "Erhaltene Boni aus Einkauf Roh-, Hilfs- und Betriebsstoffe",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5754: OdooAccountTemplate = OdooAccountTemplate {
    code: "5754",
    name: "Erhaltene Boni aus Einkauf Roh-, Hilfs- und Betriebsstoffe 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5755: OdooAccountTemplate = OdooAccountTemplate {
    code: "5755",
    name: "Erhaltene Boni aus Einkauf Roh-, Hilfs- und Betriebsstoffe 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5760: OdooAccountTemplate = OdooAccountTemplate {
    code: "5760",
    name: "Erhaltene Boni 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5770: OdooAccountTemplate = OdooAccountTemplate {
    code: "5770",
    name: "Erhaltene Rabatte",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5780: OdooAccountTemplate = OdooAccountTemplate {
    code: "5780",
    name: "Erhaltene Rabatte 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5783: OdooAccountTemplate = OdooAccountTemplate {
    code: "5783",
    name: "Erhaltene Rabatte aus Einkauf Roh-, Hilfs- und Betriebsstoffe",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5784: OdooAccountTemplate = OdooAccountTemplate {
    code: "5784",
    name: "Erhaltene Rabatte aus Einkauf Roh-, Hilfs- und Betriebsstoffe 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5785: OdooAccountTemplate = OdooAccountTemplate {
    code: "5785",
    name: "Erhaltene Rabatte aus Einkauf Roh-, Hilfs- und Betriebsstoffe 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5788: OdooAccountTemplate = OdooAccountTemplate {
    code: "5788",
    name: "Erhaltene Skonti aus Einkauf Roh-, Hilfs- und Betriebsstoffe 10,7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5790: OdooAccountTemplate = OdooAccountTemplate {
    code: "5790",
    name: "Erhaltene Rabatte 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5792: OdooAccountTemplate = OdooAccountTemplate {
    code: "5792",
    name: "Erhaltene Skonti aus Erwerb Roh-, Hilfs- und Betriebsstoffe als letzter Abnehmer innerhalb Dreiecksgeschäft 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5793: OdooAccountTemplate = OdooAccountTemplate {
    code: "5793",
    name: "Erhaltene Skonti aus Erwerb Waren als letzter Abnehmer innerhalb Dreiecksgeschäft 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5794: OdooAccountTemplate = OdooAccountTemplate {
    code: "5794",
    name: "Erhaltene Skonti 5,5 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5796: OdooAccountTemplate = OdooAccountTemplate {
    code: "5796",
    name: "Erhaltene Skonti 10,7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5798: OdooAccountTemplate = OdooAccountTemplate {
    code: "5798",
    name: "Erhaltene Skonti aus Einkauf Roh-, Hilfs- und Betriebsstoffe 5,5 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5800: OdooAccountTemplate = OdooAccountTemplate {
    code: "5800",
    name: "Bezugsnebenkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5820: OdooAccountTemplate = OdooAccountTemplate {
    code: "5820",
    name: "Leergut",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5840: OdooAccountTemplate = OdooAccountTemplate {
    code: "5840",
    name: "Zölle und Einfuhrabgaben",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5880: OdooAccountTemplate = OdooAccountTemplate {
    code: "5880",
    name: "Bestandsveränderungen Roh-, Hilfs- und Betriebsstoffe sowie bezogene Waren",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5881: OdooAccountTemplate = OdooAccountTemplate {
    code: "5881",
    name: "Bestandsveränderungen Waren",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5885: OdooAccountTemplate = OdooAccountTemplate {
    code: "5885",
    name: "Bestandsveränderungen Roh-, Hilfs- und Betriebsstoffe",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5900: OdooAccountTemplate = OdooAccountTemplate {
    code: "5900",
    name: "Fremdleistungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5906: OdooAccountTemplate = OdooAccountTemplate {
    code: "5906",
    name: "Fremdleistungen 19 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5908: OdooAccountTemplate = OdooAccountTemplate {
    code: "5908",
    name: "Fremdleistungen 7 % Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5909: OdooAccountTemplate = OdooAccountTemplate {
    code: "5909",
    name: "Fremdleistungen ohne Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5910: OdooAccountTemplate = OdooAccountTemplate {
    code: "5910",
    name: "Bauleistungen eines im Inland ansässigen Unternehmers 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5913: OdooAccountTemplate = OdooAccountTemplate {
    code: "5913",
    name: "Sonstige Leistungen eines im anderen EU-Land ansässigen Unternehmers 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5915: OdooAccountTemplate = OdooAccountTemplate {
    code: "5915",
    name: "Leistungen eines im Ausland ansässigen Unternehmers 7 % Vorsteuer und 7 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5920: OdooAccountTemplate = OdooAccountTemplate {
    code: "5920",
    name: "Bauleistungen eines im Inland ansässigen Unternehmers 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5923: OdooAccountTemplate = OdooAccountTemplate {
    code: "5923",
    name: "Sonstige Leistungen eines im anderen EU-Land ansässigen Unternehmers 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5925: OdooAccountTemplate = OdooAccountTemplate {
    code: "5925",
    name: "Leistungen eines im Ausland ansässigen Unternehmers 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_05"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5950: OdooAccountTemplate = OdooAccountTemplate {
    code: "5950",
    name: "Erhaltene Skonti aus Leistungen, für die als Leistungsempfänger die Steuer nach § 13b UStG geschuldet wird",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5951: OdooAccountTemplate = OdooAccountTemplate {
    code: "5951",
    name: "Erhaltene Skonti aus Leistungen, für die als Leistungsempfänger die Steuer nach § 13b UStG geschuldet wird 19 % Vorsteuer und 19 % Umsatzsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5970: OdooAccountTemplate = OdooAccountTemplate {
    code: "5970",
    name: "Fremdleistungen (Miet- und Pachtzinsen bewegliche Wirtschaftsgüter)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5975: OdooAccountTemplate = OdooAccountTemplate {
    code: "5975",
    name: "Fremdleistungen (Miet- und Pachtzinsen unbewegliche Wirtschaftsgüter)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5980: OdooAccountTemplate = OdooAccountTemplate {
    code: "5980",
    name: "Fremdleistungen (Entgelte für Rechte und Lizenzen)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_5985: OdooAccountTemplate = OdooAccountTemplate {
    code: "5985",
    name: "Fremdleistungen (Vergütungen für die Überlassung von Wirtschaftsgütern - mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6000: OdooAccountTemplate = OdooAccountTemplate {
    code: "6000",
    name: "Löhne und Gehälter",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6010: OdooAccountTemplate = OdooAccountTemplate {
    code: "6010",
    name: "Löhne",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6020: OdooAccountTemplate = OdooAccountTemplate {
    code: "6020",
    name: "Gehälter",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6024: OdooAccountTemplate = OdooAccountTemplate {
    code: "6024",
    name: "Geschäftsführergehälter der GmbH-Gesellschafter",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6026: OdooAccountTemplate = OdooAccountTemplate {
    code: "6026",
    name: "Tantiemen Gesellschafter-Geschäftsführer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6027: OdooAccountTemplate = OdooAccountTemplate {
    code: "6027",
    name: "Geschäftsführergehälter",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6028: OdooAccountTemplate = OdooAccountTemplate {
    code: "6028",
    name: "Vergütungen an angestellte Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6029: OdooAccountTemplate = OdooAccountTemplate {
    code: "6029",
    name: "Tantiemen Arbeitnehmer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6030: OdooAccountTemplate = OdooAccountTemplate {
    code: "6030",
    name: "Aushilfslöhne",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6035: OdooAccountTemplate = OdooAccountTemplate {
    code: "6035",
    name: "Löhne für Minijobs",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6036: OdooAccountTemplate = OdooAccountTemplate {
    code: "6036",
    name: "Pauschale Steuer für Minijobber",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6037: OdooAccountTemplate = OdooAccountTemplate {
    code: "6037",
    name: "Pauschale Steuer für Gesellschafter-Geschäftsführer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6038: OdooAccountTemplate = OdooAccountTemplate {
    code: "6038",
    name: "Pauschale Steuer für angestellte Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6039: OdooAccountTemplate = OdooAccountTemplate {
    code: "6039",
    name: "Pauschale Steuer für Arbeitnehmer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6040: OdooAccountTemplate = OdooAccountTemplate {
    code: "6040",
    name: "Pauschale Steuer für Aushilfen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6045: OdooAccountTemplate = OdooAccountTemplate {
    code: "6045",
    name: "Bedienungsgelder",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6050: OdooAccountTemplate = OdooAccountTemplate {
    code: "6050",
    name: "Ehegattengehalt",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6060: OdooAccountTemplate = OdooAccountTemplate {
    code: "6060",
    name: "Freiwillige soziale Aufwendungen, lohnsteuerpflichtig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6066: OdooAccountTemplate = OdooAccountTemplate {
    code: "6066",
    name: "Freiwillige Zuwendungen an Minijobber",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6067: OdooAccountTemplate = OdooAccountTemplate {
    code: "6067",
    name: "Freiwillige Zuwendungen an Gesellschafter-Geschäftsführer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6068: OdooAccountTemplate = OdooAccountTemplate {
    code: "6068",
    name: "Freiwillige Zuwendungen an angestellte Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6069: OdooAccountTemplate = OdooAccountTemplate {
    code: "6069",
    name: "Pauschale Steuer auf sonstige Bezüge (z. B. Fahrtkostenzuschüsse)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6070: OdooAccountTemplate = OdooAccountTemplate {
    code: "6070",
    name: "Krankengeldzuschüsse",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6071: OdooAccountTemplate = OdooAccountTemplate {
    code: "6071",
    name: "Sachzuwendungen und Dienstleistungen an Minijobber",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6072: OdooAccountTemplate = OdooAccountTemplate {
    code: "6072",
    name: "Sachzuwendungen und Dienstleistungen an Arbeitnehmer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6073: OdooAccountTemplate = OdooAccountTemplate {
    code: "6073",
    name: "Sachzuwendungen und Dienstleistungen an Gesellschafter-Geschäftsführer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6074: OdooAccountTemplate = OdooAccountTemplate {
    code: "6074",
    name: "Sachzuwendungen und Dienstleistungen an angestellte Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6075: OdooAccountTemplate = OdooAccountTemplate {
    code: "6075",
    name: "Zuschüsse der Agenturen für Arbeit (Haben)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6076: OdooAccountTemplate = OdooAccountTemplate {
    code: "6076",
    name: "Aufwendungen aus der Veränderung von Urlaubsrückstellungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6077: OdooAccountTemplate = OdooAccountTemplate {
    code: "6077",
    name: "Aufwendungen aus der Veränderung von Urlaubsrückstellungen für Gesellschafter-Geschäftsführer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6078: OdooAccountTemplate = OdooAccountTemplate {
    code: "6078",
    name: "Aufwendungen aus der Veränderung von Urlaubsrückstellungen für angestellte Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6079: OdooAccountTemplate = OdooAccountTemplate {
    code: "6079",
    name: "Aufwendungen aus der Veränderung von Urlaubsrückstellungen für Minijobber",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6080: OdooAccountTemplate = OdooAccountTemplate {
    code: "6080",
    name: "Vermögenswirksame Leistungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6090: OdooAccountTemplate = OdooAccountTemplate {
    code: "6090",
    name: "Fahrtkostenerstattung Wohnung/Arbeitsstätte",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6100: OdooAccountTemplate = OdooAccountTemplate {
    code: "6100",
    name: "Soziale Abgaben und Aufwendungen für Altersversorgung und für Unterstützung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6110: OdooAccountTemplate = OdooAccountTemplate {
    code: "6110",
    name: "Gesetzliche soziale Aufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6118: OdooAccountTemplate = OdooAccountTemplate {
    code: "6118",
    name: "Gesetzliche soziale Aufwendungen für Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6120: OdooAccountTemplate = OdooAccountTemplate {
    code: "6120",
    name: "Beiträge zur Berufsgenossenschaft",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6130: OdooAccountTemplate = OdooAccountTemplate {
    code: "6130",
    name: "Freiwillige soziale Aufwendungen, lohnsteuerfrei",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6140: OdooAccountTemplate = OdooAccountTemplate {
    code: "6140",
    name: "Aufwendungen für Altersversorgung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6147: OdooAccountTemplate = OdooAccountTemplate {
    code: "6147",
    name: "Pauschale Steuer auf sonstige Bezüge (z. B. Direktversicherungen)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6148: OdooAccountTemplate = OdooAccountTemplate {
    code: "6148",
    name: "Aufwendungen für Altersversorgung für Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6149: OdooAccountTemplate = OdooAccountTemplate {
    code: "6149",
    name: "Aufwendungen für Altersversorgung für Gesellschafter-Geschäftsführer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6150: OdooAccountTemplate = OdooAccountTemplate {
    code: "6150",
    name: "Versorgungskassen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6160: OdooAccountTemplate = OdooAccountTemplate {
    code: "6160",
    name: "Aufwendungen für Unterstützung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6170: OdooAccountTemplate = OdooAccountTemplate {
    code: "6170",
    name: "Sonstige soziale Abgaben",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6171: OdooAccountTemplate = OdooAccountTemplate {
    code: "6171",
    name: "Soziale Abgaben für Minijobber",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_06"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6200: OdooAccountTemplate = OdooAccountTemplate {
    code: "6200",
    name: "Abschreibungen auf immaterielle Vermögensgegenstände",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6201: OdooAccountTemplate = OdooAccountTemplate {
    code: "6201",
    name: "Abschreibungen auf selbst geschaffene immaterielle Vermögensgegenstände",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6205: OdooAccountTemplate = OdooAccountTemplate {
    code: "6205",
    name: "Abschreibungen auf den Geschäfts- oder Firmenwert",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6209: OdooAccountTemplate = OdooAccountTemplate {
    code: "6209",
    name: "Außerplanmäßige Abschreibungen auf den Geschäfts- oder Firmenwert",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6210: OdooAccountTemplate = OdooAccountTemplate {
    code: "6210",
    name: "Außerplanmäßige Abschreibungen auf immaterielle Vermögensgegenstände",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6211: OdooAccountTemplate = OdooAccountTemplate {
    code: "6211",
    name: "Außerplanmäßige Abschreibungen auf selbst geschaffene immaterielle Vermögensgegenstände",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6220: OdooAccountTemplate = OdooAccountTemplate {
    code: "6220",
    name: "Abschreibungen auf Sachanlagen (ohne AfA auf Kfz und Gebäude)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6221: OdooAccountTemplate = OdooAccountTemplate {
    code: "6221",
    name: "Abschreibungen auf Gebäude",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6222: OdooAccountTemplate = OdooAccountTemplate {
    code: "6222",
    name: "Abschreibungen auf Kfz",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6223: OdooAccountTemplate = OdooAccountTemplate {
    code: "6223",
    name: "Abschreibungen auf Gebäudeteil des häuslichen Arbeitszimmers",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6230: OdooAccountTemplate = OdooAccountTemplate {
    code: "6230",
    name: "Außerplanmäßige Abschreibungen auf Sachanlagen",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6231: OdooAccountTemplate = OdooAccountTemplate {
    code: "6231",
    name: "Absetzung für außergewöhnliche technische und wirtschaftliche Abnutzung der Gebäude",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6232: OdooAccountTemplate = OdooAccountTemplate {
    code: "6232",
    name: "Absetzung für außergewöhnliche technische und wirtschaftliche Abnutzung des Kfz",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6233: OdooAccountTemplate = OdooAccountTemplate {
    code: "6233",
    name: "Absetzung für außergewöhnliche technische und wirtschaftliche Abnutzung sonstiger Wirtschaftsgüter",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6240: OdooAccountTemplate = OdooAccountTemplate {
    code: "6240",
    name: "Abschreibungen auf Sachanlagen auf Grund steuerlicher Sondervorschriften",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6241: OdooAccountTemplate = OdooAccountTemplate {
    code: "6241",
    name: "Sonderabschreibungen nach § 7g Abs. 5 EStG (ohne Kfz)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6242: OdooAccountTemplate = OdooAccountTemplate {
    code: "6242",
    name: "Sonderabschreibungen nach § 7g Abs. 5 EStG (für Kfz)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6243: OdooAccountTemplate = OdooAccountTemplate {
    code: "6243",
    name: "Kürzung der Anschaffungs- oder Herstellungskosten nach § 7g Abs. 2 EStG (ohne Kfz)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6244: OdooAccountTemplate = OdooAccountTemplate {
    code: "6244",
    name: "Kürzung der Anschaffungs- oder Herstellungskosten nach § 7g Abs. 2 EStG (für Kfz)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6250: OdooAccountTemplate = OdooAccountTemplate {
    code: "6250",
    name: "Kaufleasing",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6260: OdooAccountTemplate = OdooAccountTemplate {
    code: "6260",
    name: "Sofortabschreibungen geringwertiger Wirtschaftsgüter",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6262: OdooAccountTemplate = OdooAccountTemplate {
    code: "6262",
    name: "Abschreibungen auf aktivierte, geringwertige Wirtschaftsgüter",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6264: OdooAccountTemplate = OdooAccountTemplate {
    code: "6264",
    name: "Abschreibungen auf den Sammelposten Wirtschaftsgüter",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6266: OdooAccountTemplate = OdooAccountTemplate {
    code: "6266",
    name: "Außerplanmäßige Abschreibungen auf aktivierte, geringwertige Wirtschaftsgüter",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6268: OdooAccountTemplate = OdooAccountTemplate {
    code: "6268",
    name: "Abschreibung. Inbetriebnahme/Erweiterung",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6270: OdooAccountTemplate = OdooAccountTemplate {
    code: "6270",
    name: "Abschreibungen auf sonstige Vermögensgegenstände des Umlaufvermögens (soweit unüblich hoch)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6272: OdooAccountTemplate = OdooAccountTemplate {
    code: "6272",
    name: "Abschreibungen auf Umlaufvermögen, steuerrechtlich bedingt (soweit unüblich hoch)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6278: OdooAccountTemplate = OdooAccountTemplate {
    code: "6278",
    name: "Abschreibungen auf Roh-, Hilfsund Betriebsstoffe/Waren (soweit unüblich hoch)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6279: OdooAccountTemplate = OdooAccountTemplate {
    code: "6279",
    name: "Abschreibungen auf fertige und unfertige Erzeugnisse (soweit unüblich hoch)",
    account_type: "expense_depreciation",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6280: OdooAccountTemplate = OdooAccountTemplate {
    code: "6280",
    name: "Forderungsverluste (soweit unüblich hoch)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6281: OdooAccountTemplate = OdooAccountTemplate {
    code: "6281",
    name: "Forderungsverluste 7 % USt (soweit unüblich hoch)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6286: OdooAccountTemplate = OdooAccountTemplate {
    code: "6286",
    name: "Forderungsverluste 19 % USt (soweit unüblich hoch)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6290: OdooAccountTemplate = OdooAccountTemplate {
    code: "6290",
    name: "Abschreibungen auf Forderungen gegenüber Kapitalgesellschaften, an denen eine Beteiligung besteht (soweit unüblich hoch), § 3c EStG bzw. § 8b Abs. 3 KStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6291: OdooAccountTemplate = OdooAccountTemplate {
    code: "6291",
    name: "Abschreibungen auf Forderungen gegenüber Gesellschaftern und nahe stehenden Personen (soweit unüblich hoch), § 8b Abs. 3 KStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_07"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6300: OdooAccountTemplate = OdooAccountTemplate {
    code: "6300",
    name: "Sonstige betriebliche Aufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6302: OdooAccountTemplate = OdooAccountTemplate {
    code: "6302",
    name: "Interimskonto für Aufwendungen in einem anderen Land, bei denen eine Vorsteuervergütung möglich ist",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6303: OdooAccountTemplate = OdooAccountTemplate {
    code: "6303",
    name: "Fremdleistungen/Fremdarbeiten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6304: OdooAccountTemplate = OdooAccountTemplate {
    code: "6304",
    name: "Sonstige Aufwendungen betrieblich und regelmäßig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6305: OdooAccountTemplate = OdooAccountTemplate {
    code: "6305",
    name: "Raumkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6310: OdooAccountTemplate = OdooAccountTemplate {
    code: "6310",
    name: "Miete (unbewegliche Wirtschaftsgüter)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6312: OdooAccountTemplate = OdooAccountTemplate {
    code: "6312",
    name: "Miete/Aufwendungen für doppelte Haushaltsführung Unternehmer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6313: OdooAccountTemplate = OdooAccountTemplate {
    code: "6313",
    name: "Vergütungen an Gesellschafter für die miet- oder pachtweise Überlassung ihrer unbeweglichen Wirtschaftsgüter",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6314: OdooAccountTemplate = OdooAccountTemplate {
    code: "6314",
    name: "Vergütungen an Mitunternehmer für die mietweise Überlassung ihrer unbeweglichen Wirtschaftsgüter § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6315: OdooAccountTemplate = OdooAccountTemplate {
    code: "6315",
    name: "Pacht (unbewegliche Wirtschaftsgüter)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6316: OdooAccountTemplate = OdooAccountTemplate {
    code: "6316",
    name: "Leasing (unbewegliche Wirtschaftsgüter)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6317: OdooAccountTemplate = OdooAccountTemplate {
    code: "6317",
    name: "Aufwendungen für gemietete oder gepachtete unbewegliche Wirtschaftsgüter, die gewerbesteuerlich hinzuzurechnen sind",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6319: OdooAccountTemplate = OdooAccountTemplate {
    code: "6319",
    name: "Vergütungen an Mitunternehmer für die pachtweise Überlassung ihrer unbeweglichen Wirtschaftsgüter § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6320: OdooAccountTemplate = OdooAccountTemplate {
    code: "6320",
    name: "Heizung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6325: OdooAccountTemplate = OdooAccountTemplate {
    code: "6325",
    name: "Gas, Strom, Wasser",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6330: OdooAccountTemplate = OdooAccountTemplate {
    code: "6330",
    name: "Reinigung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6335: OdooAccountTemplate = OdooAccountTemplate {
    code: "6335",
    name: "Instandhaltung betrieblicher Räume",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6340: OdooAccountTemplate = OdooAccountTemplate {
    code: "6340",
    name: "Abgaben für betrieblich genutzten Grundbesitz",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6345: OdooAccountTemplate = OdooAccountTemplate {
    code: "6345",
    name: "Sonstige Raumkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6348: OdooAccountTemplate = OdooAccountTemplate {
    code: "6348",
    name: "Aufwendungen für ein häusliches Arbeitszimmer (abziehbarer Anteil)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6349: OdooAccountTemplate = OdooAccountTemplate {
    code: "6349",
    name: "Aufwendungen für ein häusliches Arbeitszimmer (nicht abziehbarer Anteil)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6350: OdooAccountTemplate = OdooAccountTemplate {
    code: "6350",
    name: "Grundstücksaufwendungen, betrieblich",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_1"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6390: OdooAccountTemplate = OdooAccountTemplate {
    code: "6390",
    name: "Zuwendungen, Spenden, steuerlich nicht abziehbar",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6391: OdooAccountTemplate = OdooAccountTemplate {
    code: "6391",
    name: "Zuwendungen, Spenden für wissenschaftliche und kulturelle Zwecke",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6392: OdooAccountTemplate = OdooAccountTemplate {
    code: "6392",
    name: "Zuwendungen, Spenden für mildtätige Zwecke",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6393: OdooAccountTemplate = OdooAccountTemplate {
    code: "6393",
    name: "Zuwendungen, Spenden für kirchliche, religiöse und gemeinnützige Zwecke",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6394: OdooAccountTemplate = OdooAccountTemplate {
    code: "6394",
    name: "Zuwendungen, Spenden an politische Parteien",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6395: OdooAccountTemplate = OdooAccountTemplate {
    code: "6395",
    name: "Zuwendungen, Spenden in das zu erhaltende Vermögen (Vermögensstock) einer Stiftung für gemeinnützige Zwecke",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6396: OdooAccountTemplate = OdooAccountTemplate {
    code: "6396",
    name: "Zuwendungen an Stiftungen gemäß § 52 Abs. 2 Nr. 4 Abgabenordnung (AO)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6397: OdooAccountTemplate = OdooAccountTemplate {
    code: "6397",
    name: "Zuwendungen, Spenden in das zu erhaltende Vermögen (Vermögensstock) einer Stiftung für kirchliche, religiöse und gemeinnützige Zwecke",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6398: OdooAccountTemplate = OdooAccountTemplate {
    code: "6398",
    name: "Zuwendungen, Spenden an Stiftungen in das zu erhaltende Vermögen (Vermögensstock) für wissenschaftliche, mildtätige, kulturelle Zwecke",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6400: OdooAccountTemplate = OdooAccountTemplate {
    code: "6400",
    name: "Versicherungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6405: OdooAccountTemplate = OdooAccountTemplate {
    code: "6405",
    name: "Versicherungen für Gebäude",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6410: OdooAccountTemplate = OdooAccountTemplate {
    code: "6410",
    name: "Netto-Prämie für Rückdeckung künftiger Versorgungsleistungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6420: OdooAccountTemplate = OdooAccountTemplate {
    code: "6420",
    name: "Beiträge",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6430: OdooAccountTemplate = OdooAccountTemplate {
    code: "6430",
    name: "Sonstige Abgaben",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6436: OdooAccountTemplate = OdooAccountTemplate {
    code: "6436",
    name: "Steuerlich abzugsfähige Verspätungszuschläge und Zwangsgelder",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6437: OdooAccountTemplate = OdooAccountTemplate {
    code: "6437",
    name: "Steuerlich nicht abzugsfähige Verspätungszuschläge und Zwangsgelder",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_2"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6440: OdooAccountTemplate = OdooAccountTemplate {
    code: "6440",
    name: "Ausgleichsabgabe nach dem Schwerbehindertengesetz",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6450: OdooAccountTemplate = OdooAccountTemplate {
    code: "6450",
    name: "Reparaturen und Instandhaltung von Bauten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6460: OdooAccountTemplate = OdooAccountTemplate {
    code: "6460",
    name: "Reparaturen und Instandhaltung von technischen Anlagen und Maschinen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6470: OdooAccountTemplate = OdooAccountTemplate {
    code: "6470",
    name: "Reparaturen und Instandhaltung von anderen Anlagen und Betriebs- und Geschäftsausstattung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6475: OdooAccountTemplate = OdooAccountTemplate {
    code: "6475",
    name: "Zuführung zu Aufwandsrückstellungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6485: OdooAccountTemplate = OdooAccountTemplate {
    code: "6485",
    name: "Reparaturen und Instandhaltung von anderen Anlagen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6490: OdooAccountTemplate = OdooAccountTemplate {
    code: "6490",
    name: "Sonstige Reparaturen und Instandhaltung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6495: OdooAccountTemplate = OdooAccountTemplate {
    code: "6495",
    name: "Wartungskosten für Hard- und Software",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_3"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6498: OdooAccountTemplate = OdooAccountTemplate {
    code: "6498",
    name: "Mietleasing bewegliche Wirtschaftsgüter für technische Anlagen und Maschinen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6500: OdooAccountTemplate = OdooAccountTemplate {
    code: "6500",
    name: "Fahrzeugkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6520: OdooAccountTemplate = OdooAccountTemplate {
    code: "6520",
    name: "Kfz-Versicherungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6530: OdooAccountTemplate = OdooAccountTemplate {
    code: "6530",
    name: "Laufende Kfz-Betriebskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6540: OdooAccountTemplate = OdooAccountTemplate {
    code: "6540",
    name: "Kfz-Reparaturen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6550: OdooAccountTemplate = OdooAccountTemplate {
    code: "6550",
    name: "Garagenmiete",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6560: OdooAccountTemplate = OdooAccountTemplate {
    code: "6560",
    name: "Mietleasing Kfz",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6570: OdooAccountTemplate = OdooAccountTemplate {
    code: "6570",
    name: "Sonstige Kfz-Kosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6580: OdooAccountTemplate = OdooAccountTemplate {
    code: "6580",
    name: "Mautgebühren",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6590: OdooAccountTemplate = OdooAccountTemplate {
    code: "6590",
    name: "Kfz-Kosten für betrieblich genutzte zum Privatvermögen gehörende Kraftfahrzeuge",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6595: OdooAccountTemplate = OdooAccountTemplate {
    code: "6595",
    name: "Fremdfahrzeugkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_4"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6600: OdooAccountTemplate = OdooAccountTemplate {
    code: "6600",
    name: "Werbekosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6605: OdooAccountTemplate = OdooAccountTemplate {
    code: "6605",
    name: "Streuartikel",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6610: OdooAccountTemplate = OdooAccountTemplate {
    code: "6610",
    name: "Geschenke abzugsfähig ohne § 37b EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6611: OdooAccountTemplate = OdooAccountTemplate {
    code: "6611",
    name: "Geschenke abzugsfähig mit § 37b EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6612: OdooAccountTemplate = OdooAccountTemplate {
    code: "6612",
    name: "Pauschale Steuer für Geschenke und Zuwendungen abzugsfähig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6620: OdooAccountTemplate = OdooAccountTemplate {
    code: "6620",
    name: "Geschenke nicht abzugsfähig ohne § 37b EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6621: OdooAccountTemplate = OdooAccountTemplate {
    code: "6621",
    name: "Geschenke nicht abzugsfähig mit § 37b EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6625: OdooAccountTemplate = OdooAccountTemplate {
    code: "6625",
    name: "Geschenke ausschließlich betrieblich genutzt",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6629: OdooAccountTemplate = OdooAccountTemplate {
    code: "6629",
    name: "Zugaben mit § 37b EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6630: OdooAccountTemplate = OdooAccountTemplate {
    code: "6630",
    name: "Repräsentationskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6640: OdooAccountTemplate = OdooAccountTemplate {
    code: "6640",
    name: "Bewirtungskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6641: OdooAccountTemplate = OdooAccountTemplate {
    code: "6641",
    name: "Sonstige eingeschränkt abziehbare Betriebsausgaben (abziehbarer Anteil)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6642: OdooAccountTemplate = OdooAccountTemplate {
    code: "6642",
    name: "Sonstige eingeschränkt abziehbare Betriebsausgaben (nichtabziehbarer Anteil)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6643: OdooAccountTemplate = OdooAccountTemplate {
    code: "6643",
    name: "Aufmerksamkeiten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6644: OdooAccountTemplate = OdooAccountTemplate {
    code: "6644",
    name: "Nicht abzugsfähige Bewirtungskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6645: OdooAccountTemplate = OdooAccountTemplate {
    code: "6645",
    name: "Nicht abzugsfähige Betriebsausgaben aus Werbe- und Repräsentationskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6650: OdooAccountTemplate = OdooAccountTemplate {
    code: "6650",
    name: "Reisekosten Arbeitnehmer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6660: OdooAccountTemplate = OdooAccountTemplate {
    code: "6660",
    name: "Reisekosten Arbeitnehmer Übernachtungsaufwand",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6663: OdooAccountTemplate = OdooAccountTemplate {
    code: "6663",
    name: "Reisekosten Arbeitnehmer Fahrtkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6664: OdooAccountTemplate = OdooAccountTemplate {
    code: "6664",
    name: "Reisekosten Arbeitnehmer Verpflegungsmehraufwand",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6668: OdooAccountTemplate = OdooAccountTemplate {
    code: "6668",
    name: "Kilometergelderstattung Arbeitnehmer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6670: OdooAccountTemplate = OdooAccountTemplate {
    code: "6670",
    name: "Reisekosten Unternehmer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6673: OdooAccountTemplate = OdooAccountTemplate {
    code: "6673",
    name: "Reisekosten Unternehmer Fahrtkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6674: OdooAccountTemplate = OdooAccountTemplate {
    code: "6674",
    name: "Reisekosten Unternehmer Verpflegungsmehraufwand",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6680: OdooAccountTemplate = OdooAccountTemplate {
    code: "6680",
    name: "Reisekosten Unternehmer Übernachtungsaufwand und Reisenebenkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6688: OdooAccountTemplate = OdooAccountTemplate {
    code: "6688",
    name: "Fahrten zwischen Wohnung und Betriebsstätte und Familienheimfahrten (abziehbarer Anteil)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6689: OdooAccountTemplate = OdooAccountTemplate {
    code: "6689",
    name: "Fahrten zwischen Wohnung und Betriebsstätte und Familienheimfahrten (nicht abziehbarer Anteil)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6690: OdooAccountTemplate = OdooAccountTemplate {
    code: "6690",
    name: "Fahrten zwischen Wohnung und Betriebsstätte und Familienheimfahrten (Haben)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6691: OdooAccountTemplate = OdooAccountTemplate {
    code: "6691",
    name: "Verpflegungsmehraufwendungen im Rahmen der doppelten Haushaltsführung Unternehmer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_5"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6700: OdooAccountTemplate = OdooAccountTemplate {
    code: "6700",
    name: "Kosten der Warenabgabe",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6710: OdooAccountTemplate = OdooAccountTemplate {
    code: "6710",
    name: "Verpackungsmaterial",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6740: OdooAccountTemplate = OdooAccountTemplate {
    code: "6740",
    name: "Ausgangsfrachten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6760: OdooAccountTemplate = OdooAccountTemplate {
    code: "6760",
    name: "Transportversicherungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6770: OdooAccountTemplate = OdooAccountTemplate {
    code: "6770",
    name: "Verkaufsprovisionen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6780: OdooAccountTemplate = OdooAccountTemplate {
    code: "6780",
    name: "Fremdarbeiten (Vertrieb)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_6"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6790: OdooAccountTemplate = OdooAccountTemplate {
    code: "6790",
    name: "Aufwand für Gewährleistung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6800: OdooAccountTemplate = OdooAccountTemplate {
    code: "6800",
    name: "Porto",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6805: OdooAccountTemplate = OdooAccountTemplate {
    code: "6805",
    name: "Telefon",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6810: OdooAccountTemplate = OdooAccountTemplate {
    code: "6810",
    name: "Telefax und Internetkosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6815: OdooAccountTemplate = OdooAccountTemplate {
    code: "6815",
    name: "Bürobedarf",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6820: OdooAccountTemplate = OdooAccountTemplate {
    code: "6820",
    name: "Zeitschriften, Bücher (Fachliteratur)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6821: OdooAccountTemplate = OdooAccountTemplate {
    code: "6821",
    name: "Fortbildungskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6822: OdooAccountTemplate = OdooAccountTemplate {
    code: "6822",
    name: "Freiwillige Sozialleistungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6823: OdooAccountTemplate = OdooAccountTemplate {
    code: "6823",
    name: "Vergütungen an Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6824: OdooAccountTemplate = OdooAccountTemplate {
    code: "6824",
    name: "Haftungsvergütung an Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6825: OdooAccountTemplate = OdooAccountTemplate {
    code: "6825",
    name: "Rechts- und Beratungskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6827: OdooAccountTemplate = OdooAccountTemplate {
    code: "6827",
    name: "Abschluss- und Prüfungskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6830: OdooAccountTemplate = OdooAccountTemplate {
    code: "6830",
    name: "Buchführungskosten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6833: OdooAccountTemplate = OdooAccountTemplate {
    code: "6833",
    name: "Vergütungen an Gesellschafter für die miet- oder pachtweise Überlassung ihrer beweglichen Wirtschaftsgüter",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6834: OdooAccountTemplate = OdooAccountTemplate {
    code: "6834",
    name: "Vergütungen an Mitunternehmer für die miet- oder pachtweise Überlassung ihrer beweglichen Wirtschaftsgüter § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6835: OdooAccountTemplate = OdooAccountTemplate {
    code: "6835",
    name: "Mieten für Einrichtungen (bewegliche Wirtschaftsgüter)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6837: OdooAccountTemplate = OdooAccountTemplate {
    code: "6837",
    name: "Aufwendungen für die zeitlich befristete Überlassung von Rechten (Lizenzen, Konzessionen)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6838: OdooAccountTemplate = OdooAccountTemplate {
    code: "6838",
    name: "Aufwendungen für gemietete oder gepachtete bewegliche Wirtschaftsgüter, die gewerbesteuerlich hinzuzurechnen sind",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6840: OdooAccountTemplate = OdooAccountTemplate {
    code: "6840",
    name: "Mietleasing bewegliche Wirtschaftsgüter für Betriebs- und Geschäftsausstattung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6845: OdooAccountTemplate = OdooAccountTemplate {
    code: "6845",
    name: "Werkzeuge und Kleingeräte",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6850: OdooAccountTemplate = OdooAccountTemplate {
    code: "6850",
    name: "Sonstiger Betriebsbedarf",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6854: OdooAccountTemplate = OdooAccountTemplate {
    code: "6854",
    name: "Genossenschaftliche Rückvergütung an Mitglieder",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6855: OdooAccountTemplate = OdooAccountTemplate {
    code: "6855",
    name: "Nebenkosten des Geldverkehrs",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6856: OdooAccountTemplate = OdooAccountTemplate {
    code: "6856",
    name: "Aufwendungen aus Anteilen an Kapitalgesellschaften §§ 3 Nr. 40 und 3c EStG bzw. § 8b Abs. 1 und 4 KStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6857: OdooAccountTemplate = OdooAccountTemplate {
    code: "6857",
    name: "Veräußerungskosten § 3 Nr. 40 EStG bzw. § 8b Abs. 2 KStG (bei Veräußerungsgewinn)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6859: OdooAccountTemplate = OdooAccountTemplate {
    code: "6859",
    name: "Aufwendungen für Abraum- und Abfallbeseitigung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6860: OdooAccountTemplate = OdooAccountTemplate {
    code: "6860",
    name: "Nicht abziehbare Vorsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6865: OdooAccountTemplate = OdooAccountTemplate {
    code: "6865",
    name: "Nicht abziehbare Vorsteuer 7 %",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6871: OdooAccountTemplate = OdooAccountTemplate {
    code: "6871",
    name: "Nicht abziehbare Vorsteuer 19 %",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6875: OdooAccountTemplate = OdooAccountTemplate {
    code: "6875",
    name: "Nicht abziehbare Hälfte der Aufsichtsratsvergütungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6876: OdooAccountTemplate = OdooAccountTemplate {
    code: "6876",
    name: "Abziehbare Aufsichtsratsvergütungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6880: OdooAccountTemplate = OdooAccountTemplate {
    code: "6880",
    name: "Aufwendungen aus der Währungsumrechnung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6881: OdooAccountTemplate = OdooAccountTemplate {
    code: "6881",
    name: "Aufwendungen aus der Währungsumrechnung (nicht § 256a HGB)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6883: OdooAccountTemplate = OdooAccountTemplate {
    code: "6883",
    name: "Aufwendungen aus Bewertung Finanzmittelfonds",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6884: OdooAccountTemplate = OdooAccountTemplate {
    code: "6884",
    name: "Erlöse aus Verkäufen Sachanlagevermögen steuerfrei § 4 Nr. 1a UStG (bei Buchverlust)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_6885: OdooAccountTemplate = OdooAccountTemplate {
    code: "6885",
    name: "Erlöse aus Verkäufen Sachanlagevermögen 19 % USt (bei Buchverlust)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_6888: OdooAccountTemplate = OdooAccountTemplate {
    code: "6888",
    name: "Erlöse aus Verkäufen Sachanlagevermögen steuerfrei § 4 Nr. 1b UStG (bei Buchverlust)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_6889: OdooAccountTemplate = OdooAccountTemplate {
    code: "6889",
    name: "Erlöse aus Verkäufen Sachanlagevermögen (bei Buchverlust)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_6890: OdooAccountTemplate = OdooAccountTemplate {
    code: "6890",
    name: "Erlöse aus Verkäufen immaterieller Vermögensgegenstände (bei Buchverlust)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_6891: OdooAccountTemplate = OdooAccountTemplate {
    code: "6891",
    name: "Erlöse aus Verkäufen Finanzanlagen (bei Buchverlust)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_6892: OdooAccountTemplate = OdooAccountTemplate {
    code: "6892",
    name: "Erlöse aus Verkäufen Finanzanlagen § 3 Nr. 40 EStG bzw. § 8b Abs. 3 KStG (bei Buchverlust)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_6895: OdooAccountTemplate = OdooAccountTemplate {
    code: "6895",
    name: "Anlagenabgänge Sachanlagen (Restbuchwert bei Buchverlust)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6896: OdooAccountTemplate = OdooAccountTemplate {
    code: "6896",
    name: "Anlagenabgänge immaterielle Vermögensgegenstände (Restbuchwert bei Buchverlust)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6897: OdooAccountTemplate = OdooAccountTemplate {
    code: "6897",
    name: "Anlagenabgänge Finanzanlagen (Restbuchwert bei Buchverlust)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6898: OdooAccountTemplate = OdooAccountTemplate {
    code: "6898",
    name: "Anlagenabgänge Finanzanlagen § 3 Nr. 40 EStG bzw. § 8b Abs. 3 KStG (Restbuchwert bei Buchverlust)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6900: OdooAccountTemplate = OdooAccountTemplate {
    code: "6900",
    name: "Verluste aus dem Abgang von Gegenständen des Anlagevermögens",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6903: OdooAccountTemplate = OdooAccountTemplate {
    code: "6903",
    name: "Verluste aus der Veräußerung von Anteilen an Kapitalgesellschaften (Finanzanlagevermögen) § 3 Nr. 40 EStG bzw. § 8b Abs. 3 KStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6905: OdooAccountTemplate = OdooAccountTemplate {
    code: "6905",
    name: "Verluste aus dem Abgang von Gegenständen des Umlaufvermögens außer Vorräte",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6906: OdooAccountTemplate = OdooAccountTemplate {
    code: "6906",
    name: "Verluste aus dem Abgang von Gegenständen des Umlaufvermögens (außer Vorräte) § 3 Nr. 40 EStG bzw. § 8b Abs. 3 KStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6910: OdooAccountTemplate = OdooAccountTemplate {
    code: "6910",
    name: "Abschreibungen auf Umlaufvermögen außer Vorräte und Wertpapiere des Umlaufvermögens (übliche Höhe)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6912: OdooAccountTemplate = OdooAccountTemplate {
    code: "6912",
    name: "Abschreibungen auf Umlaufvermögen außer Vorräte und Wertpapiere des Umlaufvermögens, steuerrechtlich bedingt (übliche Höhe)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6918: OdooAccountTemplate = OdooAccountTemplate {
    code: "6918",
    name: "Aufwendungen aus dem Erwerb eigener Anteile",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6920: OdooAccountTemplate = OdooAccountTemplate {
    code: "6920",
    name: "Einstellung in die Pauschalwertberichtigung auf Forderungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6922: OdooAccountTemplate = OdooAccountTemplate {
    code: "6922",
    name: "Einstellungen in die steuerliche Rücklage nach § 6b Abs. 3 EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6923: OdooAccountTemplate = OdooAccountTemplate {
    code: "6923",
    name: "Einstellung in die Einzelwertberichtigung auf Forderungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6924: OdooAccountTemplate = OdooAccountTemplate {
    code: "6924",
    name: "Einstellungen in die steuerliche Rücklage nach § 6b Abs. 10 EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6927: OdooAccountTemplate = OdooAccountTemplate {
    code: "6927",
    name: "Einstellungen in sonstige steuerliche Rücklagen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6928: OdooAccountTemplate = OdooAccountTemplate {
    code: "6928",
    name: "Einstellungen in die Rücklage für Ersatzbeschaffung nach R 6.6 EStR",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6929: OdooAccountTemplate = OdooAccountTemplate {
    code: "6929",
    name: "Einstellungen in die steuerliche Rücklage nach § 4g EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6930: OdooAccountTemplate = OdooAccountTemplate {
    code: "6930",
    name: "Forderungsverluste (übliche Höhe)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6931: OdooAccountTemplate = OdooAccountTemplate {
    code: "6931",
    name: "Forderungsverluste 7 % USt (übliche Höhe)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6932: OdooAccountTemplate = OdooAccountTemplate {
    code: "6932",
    name: "Forderungsverluste aus steuerfreien EU-Lieferungen (übliche Höhe)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6933: OdooAccountTemplate = OdooAccountTemplate {
    code: "6933",
    name: "Forderungsverluste aus im Inland steuerpflichtigen EU-Lieferungen 7 % USt (übliche Höhe)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6936: OdooAccountTemplate = OdooAccountTemplate {
    code: "6936",
    name: "Forderungsverluste 19 % USt (übliche Höhe)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6938: OdooAccountTemplate = OdooAccountTemplate {
    code: "6938",
    name: "Forderungsverluste aus im Inland steuerpflichtigen EU-Lieferungen 19 % USt (übliche Höhe)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6960: OdooAccountTemplate = OdooAccountTemplate {
    code: "6960",
    name: "Periodenfremde Aufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6967: OdooAccountTemplate = OdooAccountTemplate {
    code: "6967",
    name: "Sonstige Aufwendungen betriebsfremd und regelmäßig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6968: OdooAccountTemplate = OdooAccountTemplate {
    code: "6968",
    name: "Sonstige nicht abziehbare Aufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_6969: OdooAccountTemplate = OdooAccountTemplate {
    code: "6969",
    name: "Sonstige Aufwendungen unregelmäßig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7000: OdooAccountTemplate = OdooAccountTemplate {
    code: "7000",
    name: "Erträge aus Beteiligungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_09"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7004: OdooAccountTemplate = OdooAccountTemplate {
    code: "7004",
    name: "Erträge aus Beteiligungen an Personengesellschaften (verbundene Unternehmen), § 9 GewStG bzw. § 18 EStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_09"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7005: OdooAccountTemplate = OdooAccountTemplate {
    code: "7005",
    name: "Erträge aus Anteilen an Kapitalgesellschaften (Beteiligung) § 3 Nr. 40 EStG bzw. § 8b Abs. 1 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_09"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7006: OdooAccountTemplate = OdooAccountTemplate {
    code: "7006",
    name: "Erträge aus Anteilen an Kapitalgesellschaften (verbundene Unternehmen) § 3 Nr. 40 EStG bzw. § 8b Abs. 1 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_09"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7008: OdooAccountTemplate = OdooAccountTemplate {
    code: "7008",
    name: "Gewinnanteile aus gewerblichen und selbständigen Mitunternehmerschaften, § 9 GewStG bzw. § 18 EStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_09"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7009: OdooAccountTemplate = OdooAccountTemplate {
    code: "7009",
    name: "Erträge aus Beteiligungen an verbundenen Unternehmen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_09"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7010: OdooAccountTemplate = OdooAccountTemplate {
    code: "7010",
    name: "Erträge aus anderen Wertpapieren und Ausleihungen des Finanzanlagevermögens",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7011: OdooAccountTemplate = OdooAccountTemplate {
    code: "7011",
    name: "Erträge aus Ausleihungen des Finanzanlagevermögens",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7012: OdooAccountTemplate = OdooAccountTemplate {
    code: "7012",
    name: "Erträge aus Ausleihungen des Finanzanlagevermögens an verbundenen Unternehmen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7013: OdooAccountTemplate = OdooAccountTemplate {
    code: "7013",
    name: "Erträge aus Anteilen an Personengesellschaften (Finanzanlagevermögen)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7014: OdooAccountTemplate = OdooAccountTemplate {
    code: "7014",
    name: "Erträge aus Anteilen an Kapitalgesellschaften (Finanzanlagevermögen) § 3 Nr. 40 EStG bzw. § 8b Abs. 1 und 4 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7015: OdooAccountTemplate = OdooAccountTemplate {
    code: "7015",
    name: "Erträge aus Anteilen an Kapitalgesellschaften (verbundene Unternehmen) § 3 Nr. 40 EStG bzw. § 8b Abs. 1 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7016: OdooAccountTemplate = OdooAccountTemplate {
    code: "7016",
    name: "Erträge aus Anteilen an Personengesellschaften (verbundene Unternehmen)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7017: OdooAccountTemplate = OdooAccountTemplate {
    code: "7017",
    name: "Erträge aus anderen Wertpapieren des Finanzanlagevermögens an Kapitalgesellschaften (verbundene Unternehmen)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7018: OdooAccountTemplate = OdooAccountTemplate {
    code: "7018",
    name: "Erträge aus anderen Wertpapieren des Finanzanlagevermögens an Personengesellschaften (verbundene Unternehmen)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7019: OdooAccountTemplate = OdooAccountTemplate {
    code: "7019",
    name: "Erträge aus anderen Wertpapieren und Ausleihungen des Finanzanlagevermögens aus verbundenen Unternehmen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7020: OdooAccountTemplate = OdooAccountTemplate {
    code: "7020",
    name: "Zins- und Dividendenerträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7030: OdooAccountTemplate = OdooAccountTemplate {
    code: "7030",
    name: "Erhaltene Ausgleichszahlungen (als außenstehender Aktionär)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_10"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7100: OdooAccountTemplate = OdooAccountTemplate {
    code: "7100",
    name: "Sonstige Zinsen und ähnliche Erträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7103: OdooAccountTemplate = OdooAccountTemplate {
    code: "7103",
    name: "Erträge aus Anteilen an Kapitalgesellschaften (Umlaufvermögen) § 3 Nr. 40 EStG bzw. § 8b Abs. 1 und 4 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7104: OdooAccountTemplate = OdooAccountTemplate {
    code: "7104",
    name: "Erträge aus Anteilen an Kapitalgesellschaften (verbundene Unternehmen) § 3 Nr. 40 EStG bzw. § 8b Abs. 1 KStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7105: OdooAccountTemplate = OdooAccountTemplate {
    code: "7105",
    name: "Zinserträge § 233a AO, steuerpflichtig",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7106: OdooAccountTemplate = OdooAccountTemplate {
    code: "7106",
    name: "Zinserträge § 233a AO, steuerfrei (Anlage GK KSt)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7107: OdooAccountTemplate = OdooAccountTemplate {
    code: "7107",
    name: "Zinserträge § 233a AO und § 4 Abs. 5b EStG, steuerfrei",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7109: OdooAccountTemplate = OdooAccountTemplate {
    code: "7109",
    name: "Sonstige Zinsen und ähnliche Erträge aus verbundenen Unternehmen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7110: OdooAccountTemplate = OdooAccountTemplate {
    code: "7110",
    name: "Sonstige Zinserträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7119: OdooAccountTemplate = OdooAccountTemplate {
    code: "7119",
    name: "Sonstige Zinserträge aus verbundenen Unternehmen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7120: OdooAccountTemplate = OdooAccountTemplate {
    code: "7120",
    name: "Zinsähnliche Erträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7129: OdooAccountTemplate = OdooAccountTemplate {
    code: "7129",
    name: "Zinsähnliche Erträge aus verbundenen Unternehmen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7130: OdooAccountTemplate = OdooAccountTemplate {
    code: "7130",
    name: "Diskonterträge",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7139: OdooAccountTemplate = OdooAccountTemplate {
    code: "7139",
    name: "Diskonterträge aus verbundenen Unternehmen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7140: OdooAccountTemplate = OdooAccountTemplate {
    code: "7140",
    name: "Steuerfreie Zinserträge aus der Abzinsung von Rückstellungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7141: OdooAccountTemplate = OdooAccountTemplate {
    code: "7141",
    name: "Zinserträge aus der Abzinsung von Verbindlichkeiten",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7142: OdooAccountTemplate = OdooAccountTemplate {
    code: "7142",
    name: "Zinserträge aus der Abzinsung von Rückstellungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7143: OdooAccountTemplate = OdooAccountTemplate {
    code: "7143",
    name: "Zinserträge aus der Abzinsung von Pensionsrückstellungen und ähnlichen/vergleichbaren Verpflichtungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7144: OdooAccountTemplate = OdooAccountTemplate {
    code: "7144",
    name: "Zinserträge aus der Abzinsung von Pensionsrückstellungen und ähnlichen/vergleichbaren Verpflichtungen zur Verrechnung nach § 246 Abs. 2 HGB",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7145: OdooAccountTemplate = OdooAccountTemplate {
    code: "7145",
    name: "Erträge aus Vermögensgegenständen zur Verrechnung nach § 246 Abs. 2 HGB",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_11"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7190: OdooAccountTemplate = OdooAccountTemplate {
    code: "7190",
    name: "Erträge aus Verlustübernahme",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7192: OdooAccountTemplate = OdooAccountTemplate {
    code: "7192",
    name: "Erhaltene Gewinne auf Grund einer Gewinngemeinschaft",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7194: OdooAccountTemplate = OdooAccountTemplate {
    code: "7194",
    name: "Erhaltene Gewinne auf Grund eines Gewinn- oder Teilgewinnabführungsvertrags",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7200: OdooAccountTemplate = OdooAccountTemplate {
    code: "7200",
    name: "Abschreibungen auf Finanzanlagen (dauerhaft)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7201: OdooAccountTemplate = OdooAccountTemplate {
    code: "7201",
    name: "Abschreibungen auf Finanzanlagen (nicht dauerhaft)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7204: OdooAccountTemplate = OdooAccountTemplate {
    code: "7204",
    name: "Abschreibungen auf Finanzanlagen § 3 Nr. 40 EStG bzw. § 8b Abs. 3 KStG (dauerhaft)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7207: OdooAccountTemplate = OdooAccountTemplate {
    code: "7207",
    name: "Abschreibungen auf Finanzanlagen - verbundene Unternehmen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7208: OdooAccountTemplate = OdooAccountTemplate {
    code: "7208",
    name: "Aufwendungen auf Grund von Verlustanteilen an gewerblichen und selbständigen Mitunternehmerschaften, § 8 GewStG bzw. § 18 EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7210: OdooAccountTemplate = OdooAccountTemplate {
    code: "7210",
    name: "Abschreibungen auf Wertpapiere des Umlaufvermögens",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7214: OdooAccountTemplate = OdooAccountTemplate {
    code: "7214",
    name: "Abschreibungen auf Wertpapiere des Umlaufvermögens § 3 Nr. 40 EStG bzw. § 8b Abs. 3 KStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7217: OdooAccountTemplate = OdooAccountTemplate {
    code: "7217",
    name: "Abschreibungen auf Wertpapiere des Umlaufvermögens - verbundene Unternehmen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7250: OdooAccountTemplate = OdooAccountTemplate {
    code: "7250",
    name: "Abschreibungen auf Finanzanlagen auf Grund § 6b EStG-Rücklage",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7255: OdooAccountTemplate = OdooAccountTemplate {
    code: "7255",
    name: "Abschreibungen auf Finanzanlagen auf Grund § 6b EStG-Rücklage, § 3 Nr. 40 EStG bzw. § 8b Abs. 3 KStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7300: OdooAccountTemplate = OdooAccountTemplate {
    code: "7300",
    name: "Zinsen und ähnliche Aufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7302: OdooAccountTemplate = OdooAccountTemplate {
    code: "7302",
    name: "Steuerlich nicht abzugsfähige andere Nebenleistungen zu Steuern § 4 Abs. 5b EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7303: OdooAccountTemplate = OdooAccountTemplate {
    code: "7303",
    name: "Steuerlich abzugsfähige andere Nebenleistungen zu Steuern",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7304: OdooAccountTemplate = OdooAccountTemplate {
    code: "7304",
    name: "Steuerlich nicht abzugsfähige andere Nebenleistungen zu Steuern",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7305: OdooAccountTemplate = OdooAccountTemplate {
    code: "7305",
    name: "Zinsaufwendungen § 233a AO abzugsfähig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7306: OdooAccountTemplate = OdooAccountTemplate {
    code: "7306",
    name: "Zinsaufwendungen §§ 234 bis 237 AO nicht abzugsfähig",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7309: OdooAccountTemplate = OdooAccountTemplate {
    code: "7309",
    name: "Zinsaufwendungen an verbundene Unternehmen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7310: OdooAccountTemplate = OdooAccountTemplate {
    code: "7310",
    name: "Zinsaufwendungen für kurzfristige Verbindlichkeiten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7313: OdooAccountTemplate = OdooAccountTemplate {
    code: "7313",
    name: "Nicht abzugsfähige Schuldzinsen nach § 4 Abs. 4a EStG (Hinzurechnungsbetrag)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7316: OdooAccountTemplate = OdooAccountTemplate {
    code: "7316",
    name: "Zinsen für Gesellschafterdarlehen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7317: OdooAccountTemplate = OdooAccountTemplate {
    code: "7317",
    name: "Zinsen an Gesellschafter mit einer Beteiligung von mehr als 25 % bzw. diesen nahe stehende Personen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7319: OdooAccountTemplate = OdooAccountTemplate {
    code: "7319",
    name: "Zinsaufwendungen für kurzfristige Verbindlichkeiten an verbundene Unternehmen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7320: OdooAccountTemplate = OdooAccountTemplate {
    code: "7320",
    name: "Zinsaufwendungen für langfristige Verbindlichkeiten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7323: OdooAccountTemplate = OdooAccountTemplate {
    code: "7323",
    name: "Abschreibungen auf ein Agio oder Disagio/Damnum zur Finanzierung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7324: OdooAccountTemplate = OdooAccountTemplate {
    code: "7324",
    name: "Abschreibungen auf ein Agio oder Disagio/Damnum zur Finanzierung des Anlagevermögens",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_12"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7325: OdooAccountTemplate = OdooAccountTemplate {
    code: "7325",
    name: "Zinsaufwendungen für Gebäude, die zum Betriebsvermögen gehören",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7326: OdooAccountTemplate = OdooAccountTemplate {
    code: "7326",
    name: "Zinsen zur Finanzierung des Anlagevermögens",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7327: OdooAccountTemplate = OdooAccountTemplate {
    code: "7327",
    name: "Renten und dauernde Lasten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7328: OdooAccountTemplate = OdooAccountTemplate {
    code: "7328",
    name: "Zinsaufwendungen für Kapitalüberlassung durch Mitunternehmer § 15 EStG (mit Sonderbetriebseinnahme korrespondierend)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7329: OdooAccountTemplate = OdooAccountTemplate {
    code: "7329",
    name: "Zinsaufwendungen für langfristige Verbindlichkeiten an verbundene Unternehmen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7330: OdooAccountTemplate = OdooAccountTemplate {
    code: "7330",
    name: "Zinsähnliche Aufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7339: OdooAccountTemplate = OdooAccountTemplate {
    code: "7339",
    name: "Zinsähnliche Aufwendungen an verbundene Unternehmen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7340: OdooAccountTemplate = OdooAccountTemplate {
    code: "7340",
    name: "Diskontaufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7349: OdooAccountTemplate = OdooAccountTemplate {
    code: "7349",
    name: "Diskontaufwendungen an verbundene Unternehmen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7350: OdooAccountTemplate = OdooAccountTemplate {
    code: "7350",
    name: "Zinsen und ähnliche Aufwendungen §§ 3 Nr. 40 und 3c EStG bzw. § 8b Abs. 1 und 4 KStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7351: OdooAccountTemplate = OdooAccountTemplate {
    code: "7351",
    name: "Zinsen und ähnliche Aufwendungen an verbundene Unternehmen §§ 3 Nr. 40 und 3c EStG bzw. § 8b Abs. 1 KStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7355: OdooAccountTemplate = OdooAccountTemplate {
    code: "7355",
    name: "Kreditprovisionen und Verwaltungskostenbeiträge",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7360: OdooAccountTemplate = OdooAccountTemplate {
    code: "7360",
    name: "Zinsanteil der Zuführungen zu Pensionsrückstellungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7361: OdooAccountTemplate = OdooAccountTemplate {
    code: "7361",
    name: "Zinsaufwendungen aus der Abzinsung von Verbindlichkeiten",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7362: OdooAccountTemplate = OdooAccountTemplate {
    code: "7362",
    name: "Zinsaufwendungen aus der Abzinsung von Rückstellungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7363: OdooAccountTemplate = OdooAccountTemplate {
    code: "7363",
    name: "Zinsaufwendungen aus der Abzinsung von Pensionsrückstellungen und ähnlichen/vergleichbaren Verpflichtungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7364: OdooAccountTemplate = OdooAccountTemplate {
    code: "7364",
    name: "Zinsaufwendungen aus der Abzinsung von Pensionsrückstellungen und ähnlichen/vergleichbaren Verpflichtungen zur Verrechnung nach § 246 Abs. 2 HGB",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7365: OdooAccountTemplate = OdooAccountTemplate {
    code: "7365",
    name: "Aufwendungen aus Vermögensgegenständen zur Verrechnung nach § 246 Abs. 2 HGB",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7366: OdooAccountTemplate = OdooAccountTemplate {
    code: "7366",
    name: "Steuerlich nicht abzugsfähige Zinsaufwendungen aus der Abzinsung von Rückstellungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_13"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7390: OdooAccountTemplate = OdooAccountTemplate {
    code: "7390",
    name: "Aufwendungen aus Verlustübernahme",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7392: OdooAccountTemplate = OdooAccountTemplate {
    code: "7392",
    name: "Abgeführte Gewinne auf Grund einer Gewinngemeinschaft",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7394: OdooAccountTemplate = OdooAccountTemplate {
    code: "7394",
    name: "Abgeführte Gewinne auf Grund eines Gewinn- oder Teilgewinnabführungsvertrags",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7399: OdooAccountTemplate = OdooAccountTemplate {
    code: "7399",
    name: "Abgeführte Gewinnanteile (Soll) / ausgeglichene Verlustanteile (Haben) bei typisch stiller Beteiligung § 8 GewStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7400: OdooAccountTemplate = OdooAccountTemplate {
    code: "7400",
    name: "Außerordentliche Erträge",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7401: OdooAccountTemplate = OdooAccountTemplate {
    code: "7401",
    name: "Außerordentliche Erträge, die den Reingewinn beeinflussen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7450: OdooAccountTemplate = OdooAccountTemplate {
    code: "7450",
    name: "Außerordentliche, nicht finanzierungswirksame Erträge",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7451: OdooAccountTemplate = OdooAccountTemplate {
    code: "7451",
    name: "Erträge durch Verschmelzungund Umwandlung",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7452: OdooAccountTemplate = OdooAccountTemplate {
    code: "7452",
    name: "Erträge aus der Veräußerung von wesentlichen Beteiligungen",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7453: OdooAccountTemplate = OdooAccountTemplate {
    code: "7453",
    name: "Erträge aus dem Verkauf wesentlicher Immobilien",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7454: OdooAccountTemplate = OdooAccountTemplate {
    code: "7454",
    name: "Gewinn aus der Veräußerung oder der Aufgabe von Geschäftsaktivitäten nach Steuern",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7460: OdooAccountTemplate = OdooAccountTemplate {
    code: "7460",
    name: "Erträge aus der Anwendung von Übergangsvorschriften",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7461: OdooAccountTemplate = OdooAccountTemplate {
    code: "7461",
    name: "Erträge aus der Anwendung von Übergangsbestimmungen (Zugänge zum Sachanlagevermögen)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7462: OdooAccountTemplate = OdooAccountTemplate {
    code: "7462",
    name: "Erträge aus der Anwendung von Übergangsbestimmungen (Zugänge zu Finanzanlagen)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7463: OdooAccountTemplate = OdooAccountTemplate {
    code: "7463",
    name: "Erträge aus der Anwendung von Übergangsbestimmungen (Wert des Umlaufvermögens)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7464: OdooAccountTemplate = OdooAccountTemplate {
    code: "7464",
    name: "Erträge aus der Anwendung von Übergangsvorschriften (latente Steuern)",
    account_type: "income",
    tag_xmlids: &["l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7500: OdooAccountTemplate = OdooAccountTemplate {
    code: "7500",
    name: "Außerordentliche Ausgaben",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7501: OdooAccountTemplate = OdooAccountTemplate {
    code: "7501",
    name: "Außerordentliche Aufwendungen, die den Jahresüberschuss beeinflussen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7550: OdooAccountTemplate = OdooAccountTemplate {
    code: "7550",
    name: "Außerordentliche, nicht finanzierungswirksame Aufwendungen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7551: OdooAccountTemplate = OdooAccountTemplate {
    code: "7551",
    name: "Verluste durch Verschmelzung und Umwandlung",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7552: OdooAccountTemplate = OdooAccountTemplate {
    code: "7552",
    name: "Verluste durch außergewöhnliche Schadensfälle (nur Bilanzierer)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7553: OdooAccountTemplate = OdooAccountTemplate {
    code: "7553",
    name: "Aufwendungen für Restrukturierungs- und Sanierungsmaßnahmen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7554: OdooAccountTemplate = OdooAccountTemplate {
    code: "7554",
    name: "Verluste aus der Veräußerung oder der Aufgabe von Geschäftsaktivitäten nach Steuern",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7560: OdooAccountTemplate = OdooAccountTemplate {
    code: "7560",
    name: "Aufwendungen aus der Anwendung von Übergangsvorschriften",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7561: OdooAccountTemplate = OdooAccountTemplate {
    code: "7561",
    name: "Aufwendungen aus der Anwendung von Übergangsvorschriften (Pensionsrückstellungen)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7562: OdooAccountTemplate = OdooAccountTemplate {
    code: "7562",
    name: "Aufwendungen aus der Anwendung von Übergangsbestimmungen (Bilanzierungshilfen)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7563: OdooAccountTemplate = OdooAccountTemplate {
    code: "7563",
    name: "Aufwendungen aus der Anwendung von Übergangsvorschriften (Latente Steuern)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7600: OdooAccountTemplate = OdooAccountTemplate {
    code: "7600",
    name: "Körperschaftsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7603: OdooAccountTemplate = OdooAccountTemplate {
    code: "7603",
    name: "Körperschaftsteuer für Vorjahre",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7604: OdooAccountTemplate = OdooAccountTemplate {
    code: "7604",
    name: "Körperschaftsteuer für Vorjahre",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7607: OdooAccountTemplate = OdooAccountTemplate {
    code: "7607",
    name: "Solidaritätszuschlagerstattungen für Vorjahre",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7608: OdooAccountTemplate = OdooAccountTemplate {
    code: "7608",
    name: "Solidaritätszuschlag",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7609: OdooAccountTemplate = OdooAccountTemplate {
    code: "7609",
    name: "Solidaritätszuschlag für Vorjahre",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7610: OdooAccountTemplate = OdooAccountTemplate {
    code: "7610",
    name: "Gewerbesteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7630: OdooAccountTemplate = OdooAccountTemplate {
    code: "7630",
    name: "Kapitalertragsteuer 25 %",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7633: OdooAccountTemplate = OdooAccountTemplate {
    code: "7633",
    name: "Anrechenbarer Solidaritätszuschlag auf Kapitalertragsteuer 25 %",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7639: OdooAccountTemplate = OdooAccountTemplate {
    code: "7639",
    name: "Anrechnung/Abzug ausländischer Quellensteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7640: OdooAccountTemplate = OdooAccountTemplate {
    code: "7640",
    name: "Gewerbesteuernachzahlungen Vorjahre",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7641: OdooAccountTemplate = OdooAccountTemplate {
    code: "7641",
    name: "Gewerbesteuernachzahlungen und Gewerbesteuererstattungen für Vorjahre nach § 4 Abs. 5b EStG",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7642: OdooAccountTemplate = OdooAccountTemplate {
    code: "7642",
    name: "Gewerbesteuererstattungen Vorjahre",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7643: OdooAccountTemplate = OdooAccountTemplate {
    code: "7643",
    name: "Erträge aus der Auflösung von Gewerbesteuerrückstellungen nach § 4 Abs. 5b EStG",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7644: OdooAccountTemplate = OdooAccountTemplate {
    code: "7644",
    name: "Erträge aus der Auflösung von Gewerbesteuerrückstellungen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7646: OdooAccountTemplate = OdooAccountTemplate {
    code: "7646",
    name: "Aufwendungen aus der Zuführung zu Steuerrückstellungen für Steuerstundung (BStBK)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7648: OdooAccountTemplate = OdooAccountTemplate {
    code: "7648",
    name: "Erträge aus der Auflösung von Steuerrückstellungen für Steuerstundung (BStBK)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7649: OdooAccountTemplate = OdooAccountTemplate {
    code: "7649",
    name: "Erträge aus der Zuführung und Auflösung von latenten Steuern",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_14"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7650: OdooAccountTemplate = OdooAccountTemplate {
    code: "7650",
    name: "Sonstige Betriebssteuern",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7675: OdooAccountTemplate = OdooAccountTemplate {
    code: "7675",
    name: "Verbrauchsteuer (sonstige Steuern)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7678: OdooAccountTemplate = OdooAccountTemplate {
    code: "7678",
    name: "Ökosteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7680: OdooAccountTemplate = OdooAccountTemplate {
    code: "7680",
    name: "Grundsteuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7685: OdooAccountTemplate = OdooAccountTemplate {
    code: "7685",
    name: "Kfz-Steuer",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7690: OdooAccountTemplate = OdooAccountTemplate {
    code: "7690",
    name: "Steuernachzahlungen Vorjahre für sonstige Steuern",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7692: OdooAccountTemplate = OdooAccountTemplate {
    code: "7692",
    name: "Steuererstattungen Vorjahre für sonstige Steuern",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7694: OdooAccountTemplate = OdooAccountTemplate {
    code: "7694",
    name: "Erträge aus der Auflösung von Rückstellungen für sonstige Steuern",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7705: OdooAccountTemplate = OdooAccountTemplate {
    code: "7705",
    name: "Gewinnvortrag nach Verwendung (mit Aufteilung für Kapitalkontenentwicklung)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7725: OdooAccountTemplate = OdooAccountTemplate {
    code: "7725",
    name: "Verlustvortrag nach Verwendung (mit Aufteilung für Kapitalkontenentwicklung)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7744: OdooAccountTemplate = OdooAccountTemplate {
    code: "7744",
    name: "Entnahmen aus anderen Ergebnisrücklagen",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7751: OdooAccountTemplate = OdooAccountTemplate {
    code: "7751",
    name: "Entnahmen aus gesamthänderisch gebundenen Rücklagen (mit Aufteilung für Kapitalkontenentwicklung)",
    account_type: "income_other",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_7781: OdooAccountTemplate = OdooAccountTemplate {
    code: "7781",
    name: "Einstellungen in gesamthänderisch gebundene Rücklagen (mit Aufteilung für Kapitalkontenentwicklung)",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_7785: OdooAccountTemplate = OdooAccountTemplate {
    code: "7785",
    name: "Einstellungen in andere Ergebnisrücklagen",
    account_type: "expense",
    tag_xmlids: &["l10n_de.tag_de_pl_15"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

pub const EXT_SKR04_9000: OdooAccountTemplate = OdooAccountTemplate {
    code: "9000",
    name: "Saldenvorträge, Sachkonten",
    account_type: "income_other",
    tag_xmlids: &[],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_9008: OdooAccountTemplate = OdooAccountTemplate {
    code: "9008",
    name: "Saldenvorträge, Debitoren",
    account_type: "income_other",
    tag_xmlids: &[],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_9009: OdooAccountTemplate = OdooAccountTemplate {
    code: "9009",
    name: "Saldenvorträge, Kreditoren",
    account_type: "income_other",
    tag_xmlids: &[],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_9090: OdooAccountTemplate = OdooAccountTemplate {
    code: "9090",
    name: "Summenvortragskonto",
    account_type: "income_other",
    tag_xmlids: &[],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_9991: OdooAccountTemplate = OdooAccountTemplate {
    code: "9991",
    name: "Cash Difference Gain",
    account_type: "income",
    tag_xmlids: &["account.account_tag_operating", "l10n_de.tag_de_pl_04"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/13"],
};

pub const EXT_SKR04_9994: OdooAccountTemplate = OdooAccountTemplate {
    code: "9994",
    name: "Cash Difference Loss",
    account_type: "expense",
    tag_xmlids: &["account.account_tag_operating", "l10n_de.tag_de_pl_08_7"],
    chart: OdooSkrChart::Skr04,
    regulation_iri: &["ogit:regulation/de/ustg/15"],
};

// ─── Canonical iteration handles ────────────────────────────────────────

/// All SKR03 account templates — canonical iteration handle.
pub static SKR03_CHART: &[OdooAccountTemplate] = &[
    EXT_SKR03_0005,
    EXT_SKR03_0010,
    EXT_SKR03_0015,
    EXT_SKR03_0020,
    EXT_SKR03_0025,
    EXT_SKR03_0027,
    EXT_SKR03_0030,
    EXT_SKR03_0035,
    EXT_SKR03_0038,
    EXT_SKR03_0039,
    EXT_SKR03_0040,
    EXT_SKR03_0043,
    EXT_SKR03_0045,
    EXT_SKR03_0046,
    EXT_SKR03_0047,
    EXT_SKR03_0048,
    EXT_SKR03_0050,
    EXT_SKR03_0059,
    EXT_SKR03_0060,
    EXT_SKR03_0065,
    EXT_SKR03_0070,
    EXT_SKR03_0075,
    EXT_SKR03_0079,
    EXT_SKR03_0080,
    EXT_SKR03_0085,
    EXT_SKR03_0090,
    EXT_SKR03_0100,
    EXT_SKR03_0110,
    EXT_SKR03_0111,
    EXT_SKR03_0112,
    EXT_SKR03_0113,
    EXT_SKR03_0115,
    EXT_SKR03_0120,
    EXT_SKR03_0129,
    EXT_SKR03_0140,
    EXT_SKR03_0145,
    EXT_SKR03_0146,
    EXT_SKR03_0147,
    EXT_SKR03_0148,
    EXT_SKR03_0149,
    EXT_SKR03_0150,
    EXT_SKR03_0159,
    EXT_SKR03_0160,
    EXT_SKR03_0165,
    EXT_SKR03_0170,
    EXT_SKR03_0175,
    EXT_SKR03_0176,
    EXT_SKR03_0177,
    EXT_SKR03_0178,
    EXT_SKR03_0179,
    EXT_SKR03_0180,
    EXT_SKR03_0189,
    EXT_SKR03_0190,
    EXT_SKR03_0191,
    EXT_SKR03_0192,
    EXT_SKR03_0193,
    EXT_SKR03_0194,
    EXT_SKR03_0195,
    EXT_SKR03_0199,
    EXT_SKR03_0200,
    EXT_SKR03_0210,
    EXT_SKR03_0220,
    EXT_SKR03_0240,
    EXT_SKR03_0260,
    EXT_SKR03_0280,
    EXT_SKR03_0290,
    EXT_SKR03_0299,
    EXT_SKR03_0300,
    EXT_SKR03_0310,
    EXT_SKR03_0320,
    EXT_SKR03_0350,
    EXT_SKR03_0380,
    EXT_SKR03_0400,
    EXT_SKR03_0410,
    EXT_SKR03_0420,
    EXT_SKR03_0430,
    EXT_SKR03_0440,
    EXT_SKR03_0450,
    EXT_SKR03_0460,
    EXT_SKR03_0480,
    EXT_SKR03_0485,
    EXT_SKR03_0490,
    EXT_SKR03_0498,
    EXT_SKR03_0499,
    EXT_SKR03_0500,
    EXT_SKR03_0501,
    EXT_SKR03_0502,
    EXT_SKR03_0503,
    EXT_SKR03_0504,
    EXT_SKR03_0505,
    EXT_SKR03_0506,
    EXT_SKR03_0507,
    EXT_SKR03_0508,
    EXT_SKR03_0509,
    EXT_SKR03_0510,
    EXT_SKR03_0513,
    EXT_SKR03_0516,
    EXT_SKR03_0517,
    EXT_SKR03_0518,
    EXT_SKR03_0519,
    EXT_SKR03_0520,
    EXT_SKR03_0523,
    EXT_SKR03_0524,
    EXT_SKR03_0525,
    EXT_SKR03_0530,
    EXT_SKR03_0535,
    EXT_SKR03_0540,
    EXT_SKR03_0550,
    EXT_SKR03_0570,
    EXT_SKR03_0580,
    EXT_SKR03_0582,
    EXT_SKR03_0584,
    EXT_SKR03_0586,
    EXT_SKR03_0590,
    EXT_SKR03_0595,
    EXT_SKR03_0600,
    EXT_SKR03_0601,
    EXT_SKR03_0605,
    EXT_SKR03_0610,
    EXT_SKR03_0615,
    EXT_SKR03_0616,
    EXT_SKR03_0620,
    EXT_SKR03_0625,
    EXT_SKR03_0630,
    EXT_SKR03_0631,
    EXT_SKR03_0640,
    EXT_SKR03_0650,
    EXT_SKR03_0660,
    EXT_SKR03_0661,
    EXT_SKR03_0670,
    EXT_SKR03_0680,
    EXT_SKR03_0699,
    EXT_SKR03_0700,
    EXT_SKR03_0701,
    EXT_SKR03_0705,
    EXT_SKR03_0710,
    EXT_SKR03_0715,
    EXT_SKR03_0716,
    EXT_SKR03_0720,
    EXT_SKR03_0725,
    EXT_SKR03_0730,
    EXT_SKR03_0731,
    EXT_SKR03_0740,
    EXT_SKR03_0750,
    EXT_SKR03_0755,
    EXT_SKR03_0760,
    EXT_SKR03_0761,
    EXT_SKR03_0764,
    EXT_SKR03_0767,
    EXT_SKR03_0770,
    EXT_SKR03_0771,
    EXT_SKR03_0774,
    EXT_SKR03_0777,
    EXT_SKR03_0780,
    EXT_SKR03_0781,
    EXT_SKR03_0784,
    EXT_SKR03_0787,
    EXT_SKR03_0799,
    EXT_SKR03_0809,
    EXT_SKR03_0810,
    EXT_SKR03_0811,
    EXT_SKR03_0812,
    EXT_SKR03_0813,
    EXT_SKR03_0815,
    EXT_SKR03_0819,
    EXT_SKR03_0839,
    EXT_SKR03_0845,
    EXT_SKR03_0848,
    EXT_SKR03_0849,
    EXT_SKR03_0852,
    EXT_SKR03_0853,
    EXT_SKR03_0854,
    EXT_SKR03_0857,
    EXT_SKR03_0858,
    EXT_SKR03_0859,
    EXT_SKR03_0865,
    EXT_SKR03_0867,
    EXT_SKR03_0870,
    EXT_SKR03_0880,
    EXT_SKR03_0890,
    EXT_SKR03_0900,
    EXT_SKR03_0910,
    EXT_SKR03_0920,
    EXT_SKR03_0950,
    EXT_SKR03_0951,
    EXT_SKR03_0952,
    EXT_SKR03_0953,
    EXT_SKR03_0954,
    EXT_SKR03_0955,
    EXT_SKR03_0956,
    EXT_SKR03_0957,
    EXT_SKR03_0961,
    EXT_SKR03_0962,
    EXT_SKR03_0963,
    EXT_SKR03_0964,
    EXT_SKR03_0965,
    EXT_SKR03_0966,
    EXT_SKR03_0967,
    EXT_SKR03_0968,
    EXT_SKR03_0969,
    EXT_SKR03_0970,
    EXT_SKR03_0971,
    EXT_SKR03_0973,
    EXT_SKR03_0974,
    EXT_SKR03_0976,
    EXT_SKR03_0977,
    EXT_SKR03_0978,
    EXT_SKR03_0979,
    EXT_SKR03_0980,
    EXT_SKR03_0983,
    EXT_SKR03_0984,
    EXT_SKR03_0985,
    EXT_SKR03_0986,
    EXT_SKR03_0987,
    EXT_SKR03_0988,
    EXT_SKR03_0989,
    EXT_SKR03_0990,
    EXT_SKR03_0996,
    EXT_SKR03_0997,
    EXT_SKR03_0998,
    EXT_SKR03_0999,
    EXT_SKR03_1010,
    EXT_SKR03_1020,
    EXT_SKR03_1100,
    EXT_SKR03_1110,
    EXT_SKR03_1120,
    EXT_SKR03_1130,
    EXT_SKR03_1190,
    EXT_SKR03_1195,
    EXT_SKR03_1210,
    EXT_SKR03_1220,
    EXT_SKR03_1230,
    EXT_SKR03_1240,
    EXT_SKR03_1250,
    EXT_SKR03_1290,
    EXT_SKR03_1295,
    EXT_SKR03_1300,
    EXT_SKR03_1301,
    EXT_SKR03_1302,
    EXT_SKR03_1305,
    EXT_SKR03_1310,
    EXT_SKR03_1311,
    EXT_SKR03_1312,
    EXT_SKR03_1315,
    EXT_SKR03_1320,
    EXT_SKR03_1321,
    EXT_SKR03_1322,
    EXT_SKR03_1325,
    EXT_SKR03_1327,
    EXT_SKR03_1329,
    EXT_SKR03_1330,
    EXT_SKR03_1340,
    EXT_SKR03_1344,
    EXT_SKR03_1348,
    EXT_SKR03_1349,
    EXT_SKR03_1350,
    EXT_SKR03_1352,
    EXT_SKR03_1353,
    EXT_SKR03_1354,
    EXT_SKR03_1355,
    EXT_SKR03_1356,
    EXT_SKR03_1357,
    EXT_SKR03_1370,
    EXT_SKR03_1371,
    EXT_SKR03_1372,
    EXT_SKR03_1373,
    EXT_SKR03_1374,
    EXT_SKR03_1375,
    EXT_SKR03_1376,
    EXT_SKR03_1377,
    EXT_SKR03_1378,
    EXT_SKR03_1380,
    EXT_SKR03_1381,
    EXT_SKR03_1382,
    EXT_SKR03_1383,
    EXT_SKR03_1385,
    EXT_SKR03_1386,
    EXT_SKR03_1387,
    EXT_SKR03_1389,
    EXT_SKR03_1390,
    EXT_SKR03_1400,
    EXT_SKR03_1401,
    EXT_SKR03_1410,
    EXT_SKR03_1411,
    EXT_SKR03_1445,
    EXT_SKR03_1446,
    EXT_SKR03_1447,
    EXT_SKR03_1448,
    EXT_SKR03_1449,
    EXT_SKR03_1450,
    EXT_SKR03_1451,
    EXT_SKR03_1455,
    EXT_SKR03_1460,
    EXT_SKR03_1461,
    EXT_SKR03_1465,
    EXT_SKR03_1470,
    EXT_SKR03_1471,
    EXT_SKR03_1475,
    EXT_SKR03_1478,
    EXT_SKR03_1479,
    EXT_SKR03_1480,
    EXT_SKR03_1481,
    EXT_SKR03_1485,
    EXT_SKR03_1488,
    EXT_SKR03_1489,
    EXT_SKR03_1490,
    EXT_SKR03_1491,
    EXT_SKR03_1495,
    EXT_SKR03_1498,
    EXT_SKR03_1499,
    EXT_SKR03_1500,
    EXT_SKR03_1501,
    EXT_SKR03_1502,
    EXT_SKR03_1503,
    EXT_SKR03_1504,
    EXT_SKR03_1505,
    EXT_SKR03_1506,
    EXT_SKR03_1507,
    EXT_SKR03_1508,
    EXT_SKR03_1510,
    EXT_SKR03_1511,
    EXT_SKR03_1518,
    EXT_SKR03_1519,
    EXT_SKR03_1520,
    EXT_SKR03_1521,
    EXT_SKR03_1522,
    EXT_SKR03_1524,
    EXT_SKR03_1525,
    EXT_SKR03_1526,
    EXT_SKR03_1527,
    EXT_SKR03_1528,
    EXT_SKR03_1529,
    EXT_SKR03_1530,
    EXT_SKR03_1531,
    EXT_SKR03_1537,
    EXT_SKR03_1538,
    EXT_SKR03_1539,
    EXT_SKR03_1540,
    EXT_SKR03_1542,
    EXT_SKR03_1543,
    EXT_SKR03_1544,
    EXT_SKR03_1545,
    EXT_SKR03_1547,
    EXT_SKR03_1548,
    EXT_SKR03_1549,
    EXT_SKR03_1550,
    EXT_SKR03_1551,
    EXT_SKR03_1555,
    EXT_SKR03_1556,
    EXT_SKR03_1557,
    EXT_SKR03_1558,
    EXT_SKR03_1559,
    EXT_SKR03_1560,
    EXT_SKR03_1561,
    EXT_SKR03_1562,
    EXT_SKR03_1563,
    EXT_SKR03_1566,
    EXT_SKR03_1567,
    EXT_SKR03_1569,
    EXT_SKR03_1570,
    EXT_SKR03_1571,
    EXT_SKR03_1572,
    EXT_SKR03_1574,
    EXT_SKR03_1576,
    EXT_SKR03_1577,
    EXT_SKR03_1578,
    EXT_SKR03_1580,
    EXT_SKR03_1581,
    EXT_SKR03_1582,
    EXT_SKR03_1583,
    EXT_SKR03_1584,
    EXT_SKR03_1585,
    EXT_SKR03_1587,
    EXT_SKR03_1588,
    EXT_SKR03_1590,
    EXT_SKR03_1592,
    EXT_SKR03_1593,
    EXT_SKR03_1594,
    EXT_SKR03_1595,
    EXT_SKR03_1596,
    EXT_SKR03_1597,
    EXT_SKR03_1598,
    EXT_SKR03_1599,
    EXT_SKR03_1600,
    EXT_SKR03_1601,
    EXT_SKR03_1605,
    EXT_SKR03_1606,
    EXT_SKR03_1607,
    EXT_SKR03_1609,
    EXT_SKR03_1610,
    EXT_SKR03_1624,
    EXT_SKR03_1625,
    EXT_SKR03_1626,
    EXT_SKR03_1628,
    EXT_SKR03_1630,
    EXT_SKR03_1631,
    EXT_SKR03_1635,
    EXT_SKR03_1638,
    EXT_SKR03_1640,
    EXT_SKR03_1641,
    EXT_SKR03_1645,
    EXT_SKR03_1648,
    EXT_SKR03_1650,
    EXT_SKR03_1651,
    EXT_SKR03_1655,
    EXT_SKR03_1658,
    EXT_SKR03_1659,
    EXT_SKR03_1660,
    EXT_SKR03_1661,
    EXT_SKR03_1662,
    EXT_SKR03_1663,
    EXT_SKR03_1665,
    EXT_SKR03_1666,
    EXT_SKR03_1667,
    EXT_SKR03_1668,
    EXT_SKR03_1670,
    EXT_SKR03_1671,
    EXT_SKR03_1672,
    EXT_SKR03_1673,
    EXT_SKR03_1675,
    EXT_SKR03_1676,
    EXT_SKR03_1677,
    EXT_SKR03_1678,
    EXT_SKR03_1691,
    EXT_SKR03_1695,
    EXT_SKR03_1696,
    EXT_SKR03_1697,
    EXT_SKR03_1698,
    EXT_SKR03_1700,
    EXT_SKR03_1701,
    EXT_SKR03_1702,
    EXT_SKR03_1703,
    EXT_SKR03_1704,
    EXT_SKR03_1705,
    EXT_SKR03_1706,
    EXT_SKR03_1707,
    EXT_SKR03_1708,
    EXT_SKR03_1709,
    EXT_SKR03_1710,
    EXT_SKR03_1711,
    EXT_SKR03_1716,
    EXT_SKR03_1718,
    EXT_SKR03_1719,
    EXT_SKR03_1720,
    EXT_SKR03_1721,
    EXT_SKR03_1722,
    EXT_SKR03_1725,
    EXT_SKR03_1728,
    EXT_SKR03_1729,
    EXT_SKR03_1730,
    EXT_SKR03_1731,
    EXT_SKR03_1732,
    EXT_SKR03_1733,
    EXT_SKR03_1734,
    EXT_SKR03_1735,
    EXT_SKR03_1736,
    EXT_SKR03_1737,
    EXT_SKR03_1738,
    EXT_SKR03_1739,
    EXT_SKR03_1740,
    EXT_SKR03_1741,
    EXT_SKR03_1742,
    EXT_SKR03_1743,
    EXT_SKR03_1744,
    EXT_SKR03_1745,
    EXT_SKR03_1746,
    EXT_SKR03_1747,
    EXT_SKR03_1748,
    EXT_SKR03_1749,
    EXT_SKR03_1750,
    EXT_SKR03_1751,
    EXT_SKR03_1752,
    EXT_SKR03_1753,
    EXT_SKR03_1754,
    EXT_SKR03_1755,
    EXT_SKR03_1756,
    EXT_SKR03_1758,
    EXT_SKR03_1759,
    EXT_SKR03_1760,
    EXT_SKR03_1761,
    EXT_SKR03_1762,
    EXT_SKR03_1764,
    EXT_SKR03_1766,
    EXT_SKR03_1767,
    EXT_SKR03_1768,
    EXT_SKR03_1769,
    EXT_SKR03_1770,
    EXT_SKR03_1771,
    EXT_SKR03_1772,
    EXT_SKR03_1774,
    EXT_SKR03_1776,
    EXT_SKR03_1777,
    EXT_SKR03_1778,
    EXT_SKR03_1779,
    EXT_SKR03_1780,
    EXT_SKR03_1781,
    EXT_SKR03_1782,
    EXT_SKR03_1783,
    EXT_SKR03_1784,
    EXT_SKR03_1785,
    EXT_SKR03_1787,
    EXT_SKR03_1788,
    EXT_SKR03_1789,
    EXT_SKR03_1790,
    EXT_SKR03_1791,
    EXT_SKR03_1792,
    EXT_SKR03_1793,
    EXT_SKR03_1794,
    EXT_SKR03_1795,
    EXT_SKR03_1796,
    EXT_SKR03_1797,
    EXT_SKR03_1800,
    EXT_SKR03_1810,
    EXT_SKR03_1820,
    EXT_SKR03_1830,
    EXT_SKR03_1840,
    EXT_SKR03_1850,
    EXT_SKR03_1860,
    EXT_SKR03_1869,
    EXT_SKR03_1870,
    EXT_SKR03_1879,
    EXT_SKR03_1880,
    EXT_SKR03_1890,
    EXT_SKR03_1900,
    EXT_SKR03_1910,
    EXT_SKR03_1920,
    EXT_SKR03_1930,
    EXT_SKR03_1940,
    EXT_SKR03_1950,
    EXT_SKR03_1960,
    EXT_SKR03_1970,
    EXT_SKR03_1980,
    EXT_SKR03_1990,
    EXT_SKR03_2000,
    EXT_SKR03_2001,
    EXT_SKR03_2004,
    EXT_SKR03_2005,
    EXT_SKR03_2006,
    EXT_SKR03_2007,
    EXT_SKR03_2008,
    EXT_SKR03_2010,
    EXT_SKR03_2020,
    EXT_SKR03_2090,
    EXT_SKR03_2091,
    EXT_SKR03_2092,
    EXT_SKR03_2094,
    EXT_SKR03_2100,
    EXT_SKR03_2102,
    EXT_SKR03_2103,
    EXT_SKR03_2104,
    EXT_SKR03_2105,
    EXT_SKR03_2106,
    EXT_SKR03_2107,
    EXT_SKR03_2108,
    EXT_SKR03_2109,
    EXT_SKR03_2110,
    EXT_SKR03_2113,
    EXT_SKR03_2114,
    EXT_SKR03_2115,
    EXT_SKR03_2116,
    EXT_SKR03_2117,
    EXT_SKR03_2118,
    EXT_SKR03_2119,
    EXT_SKR03_2120,
    EXT_SKR03_2123,
    EXT_SKR03_2124,
    EXT_SKR03_2125,
    EXT_SKR03_2126,
    EXT_SKR03_2127,
    EXT_SKR03_2128,
    EXT_SKR03_2129,
    EXT_SKR03_2130,
    EXT_SKR03_2139,
    EXT_SKR03_2140,
    EXT_SKR03_2141,
    EXT_SKR03_2142,
    EXT_SKR03_2143,
    EXT_SKR03_2144,
    EXT_SKR03_2145,
    EXT_SKR03_2146,
    EXT_SKR03_2147,
    EXT_SKR03_2148,
    EXT_SKR03_2149,
    EXT_SKR03_2150,
    EXT_SKR03_2151,
    EXT_SKR03_2166,
    EXT_SKR03_2170,
    EXT_SKR03_2171,
    EXT_SKR03_2176,
    EXT_SKR03_2200,
    EXT_SKR03_2203,
    EXT_SKR03_2204,
    EXT_SKR03_2208,
    EXT_SKR03_2209,
    EXT_SKR03_2210,
    EXT_SKR03_2213,
    EXT_SKR03_2216,
    EXT_SKR03_2219,
    EXT_SKR03_2250,
    EXT_SKR03_2255,
    EXT_SKR03_2260,
    EXT_SKR03_2265,
    EXT_SKR03_2280,
    EXT_SKR03_2281,
    EXT_SKR03_2282,
    EXT_SKR03_2283,
    EXT_SKR03_2284,
    EXT_SKR03_2285,
    EXT_SKR03_2287,
    EXT_SKR03_2289,
    EXT_SKR03_2300,
    EXT_SKR03_2307,
    EXT_SKR03_2308,
    EXT_SKR03_2309,
    EXT_SKR03_2310,
    EXT_SKR03_2311,
    EXT_SKR03_2312,
    EXT_SKR03_2313,
    EXT_SKR03_2315,
    EXT_SKR03_2316,
    EXT_SKR03_2317,
    EXT_SKR03_2318,
    EXT_SKR03_2320,
    EXT_SKR03_2323,
    EXT_SKR03_2325,
    EXT_SKR03_2326,
    EXT_SKR03_2327,
    EXT_SKR03_2328,
    EXT_SKR03_2339,
    EXT_SKR03_2340,
    EXT_SKR03_2341,
    EXT_SKR03_2342,
    EXT_SKR03_2344,
    EXT_SKR03_2345,
    EXT_SKR03_2347,
    EXT_SKR03_2350,
    EXT_SKR03_2375,
    EXT_SKR03_2380,
    EXT_SKR03_2381,
    EXT_SKR03_2382,
    EXT_SKR03_2383,
    EXT_SKR03_2384,
    EXT_SKR03_2385,
    EXT_SKR03_2386,
    EXT_SKR03_2387,
    EXT_SKR03_2388,
    EXT_SKR03_2389,
    EXT_SKR03_2390,
    EXT_SKR03_2400,
    EXT_SKR03_2401,
    EXT_SKR03_2402,
    EXT_SKR03_2403,
    EXT_SKR03_2406,
    EXT_SKR03_2408,
    EXT_SKR03_2430,
    EXT_SKR03_2431,
    EXT_SKR03_2436,
    EXT_SKR03_2440,
    EXT_SKR03_2441,
    EXT_SKR03_2450,
    EXT_SKR03_2451,
    EXT_SKR03_2480,
    EXT_SKR03_2481,
    EXT_SKR03_2485,
    EXT_SKR03_2490,
    EXT_SKR03_2492,
    EXT_SKR03_2493,
    EXT_SKR03_2494,
    EXT_SKR03_2498,
    EXT_SKR03_2500,
    EXT_SKR03_2501,
    EXT_SKR03_2504,
    EXT_SKR03_2505,
    EXT_SKR03_2506,
    EXT_SKR03_2507,
    EXT_SKR03_2508,
    EXT_SKR03_2510,
    EXT_SKR03_2520,
    EXT_SKR03_2590,
    EXT_SKR03_2591,
    EXT_SKR03_2592,
    EXT_SKR03_2593,
    EXT_SKR03_2594,
    EXT_SKR03_2600,
    EXT_SKR03_2603,
    EXT_SKR03_2615,
    EXT_SKR03_2616,
    EXT_SKR03_2617,
    EXT_SKR03_2618,
    EXT_SKR03_2619,
    EXT_SKR03_2620,
    EXT_SKR03_2621,
    EXT_SKR03_2622,
    EXT_SKR03_2623,
    EXT_SKR03_2625,
    EXT_SKR03_2626,
    EXT_SKR03_2640,
    EXT_SKR03_2641,
    EXT_SKR03_2646,
    EXT_SKR03_2647,
    EXT_SKR03_2648,
    EXT_SKR03_2649,
    EXT_SKR03_2650,
    EXT_SKR03_2652,
    EXT_SKR03_2653,
    EXT_SKR03_2654,
    EXT_SKR03_2655,
    EXT_SKR03_2656,
    EXT_SKR03_2657,
    EXT_SKR03_2658,
    EXT_SKR03_2659,
    EXT_SKR03_2660,
    EXT_SKR03_2661,
    EXT_SKR03_2666,
    EXT_SKR03_2670,
    EXT_SKR03_2679,
    EXT_SKR03_2680,
    EXT_SKR03_2682,
    EXT_SKR03_2683,
    EXT_SKR03_2684,
    EXT_SKR03_2685,
    EXT_SKR03_2686,
    EXT_SKR03_2687,
    EXT_SKR03_2688,
    EXT_SKR03_2689,
    EXT_SKR03_2700,
    EXT_SKR03_2705,
    EXT_SKR03_2707,
    EXT_SKR03_2709,
    EXT_SKR03_2710,
    EXT_SKR03_2711,
    EXT_SKR03_2712,
    EXT_SKR03_2713,
    EXT_SKR03_2714,
    EXT_SKR03_2715,
    EXT_SKR03_2716,
    EXT_SKR03_2720,
    EXT_SKR03_2723,
    EXT_SKR03_2725,
    EXT_SKR03_2726,
    EXT_SKR03_2727,
    EXT_SKR03_2728,
    EXT_SKR03_2729,
    EXT_SKR03_2730,
    EXT_SKR03_2731,
    EXT_SKR03_2732,
    EXT_SKR03_2733,
    EXT_SKR03_2734,
    EXT_SKR03_2735,
    EXT_SKR03_2736,
    EXT_SKR03_2738,
    EXT_SKR03_2739,
    EXT_SKR03_2740,
    EXT_SKR03_2741,
    EXT_SKR03_2742,
    EXT_SKR03_2743,
    EXT_SKR03_2744,
    EXT_SKR03_2746,
    EXT_SKR03_2747,
    EXT_SKR03_2749,
    EXT_SKR03_2750,
    EXT_SKR03_2751,
    EXT_SKR03_2752,
    EXT_SKR03_2760,
    EXT_SKR03_2762,
    EXT_SKR03_2764,
    EXT_SKR03_2790,
    EXT_SKR03_2792,
    EXT_SKR03_2794,
    EXT_SKR03_2798,
    EXT_SKR03_2840,
    EXT_SKR03_2841,
    EXT_SKR03_2850,
    EXT_SKR03_2865,
    EXT_SKR03_2867,
    EXT_SKR03_2870,
    EXT_SKR03_2890,
    EXT_SKR03_2891,
    EXT_SKR03_2892,
    EXT_SKR03_2893,
    EXT_SKR03_2894,
    EXT_SKR03_2895,
    EXT_SKR03_2990,
    EXT_SKR03_3000,
    EXT_SKR03_3010,
    EXT_SKR03_3030,
    EXT_SKR03_3060,
    EXT_SKR03_3062,
    EXT_SKR03_3066,
    EXT_SKR03_3067,
    EXT_SKR03_3070,
    EXT_SKR03_3071,
    EXT_SKR03_3075,
    EXT_SKR03_3076,
    EXT_SKR03_3089,
    EXT_SKR03_3090,
    EXT_SKR03_3091,
    EXT_SKR03_3092,
    EXT_SKR03_3100,
    EXT_SKR03_3106,
    EXT_SKR03_3108,
    EXT_SKR03_3109,
    EXT_SKR03_3110,
    EXT_SKR03_3113,
    EXT_SKR03_3115,
    EXT_SKR03_3120,
    EXT_SKR03_3123,
    EXT_SKR03_3125,
    EXT_SKR03_3130,
    EXT_SKR03_3133,
    EXT_SKR03_3135,
    EXT_SKR03_3140,
    EXT_SKR03_3143,
    EXT_SKR03_3145,
    EXT_SKR03_3150,
    EXT_SKR03_3151,
    EXT_SKR03_3153,
    EXT_SKR03_3154,
    EXT_SKR03_3160,
    EXT_SKR03_3165,
    EXT_SKR03_3170,
    EXT_SKR03_3175,
    EXT_SKR03_3180,
    EXT_SKR03_3185,
    EXT_SKR03_3200,
    EXT_SKR03_3300,
    EXT_SKR03_3349,
    EXT_SKR03_3400,
    EXT_SKR03_3420,
    EXT_SKR03_3425,
    EXT_SKR03_3430,
    EXT_SKR03_3435,
    EXT_SKR03_3440,
    EXT_SKR03_3505,
    EXT_SKR03_3540,
    EXT_SKR03_3550,
    EXT_SKR03_3551,
    EXT_SKR03_3552,
    EXT_SKR03_3553,
    EXT_SKR03_3557,
    EXT_SKR03_3558,
    EXT_SKR03_3559,
    EXT_SKR03_3560,
    EXT_SKR03_3565,
    EXT_SKR03_3600,
    EXT_SKR03_3610,
    EXT_SKR03_3660,
    EXT_SKR03_3700,
    EXT_SKR03_3701,
    EXT_SKR03_3710,
    EXT_SKR03_3714,
    EXT_SKR03_3715,
    EXT_SKR03_3717,
    EXT_SKR03_3718,
    EXT_SKR03_3720,
    EXT_SKR03_3724,
    EXT_SKR03_3725,
    EXT_SKR03_3730,
    EXT_SKR03_3731,
    EXT_SKR03_3733,
    EXT_SKR03_3734,
    EXT_SKR03_3736,
    EXT_SKR03_3738,
    EXT_SKR03_3741,
    EXT_SKR03_3743,
    EXT_SKR03_3744,
    EXT_SKR03_3745,
    EXT_SKR03_3746,
    EXT_SKR03_3748,
    EXT_SKR03_3750,
    EXT_SKR03_3753,
    EXT_SKR03_3754,
    EXT_SKR03_3755,
    EXT_SKR03_3760,
    EXT_SKR03_3769,
    EXT_SKR03_3770,
    EXT_SKR03_3780,
    EXT_SKR03_3783,
    EXT_SKR03_3784,
    EXT_SKR03_3785,
    EXT_SKR03_3788,
    EXT_SKR03_3790,
    EXT_SKR03_3792,
    EXT_SKR03_3793,
    EXT_SKR03_3794,
    EXT_SKR03_3796,
    EXT_SKR03_3798,
    EXT_SKR03_3800,
    EXT_SKR03_3830,
    EXT_SKR03_3850,
    EXT_SKR03_3950,
    EXT_SKR03_3955,
    EXT_SKR03_3960,
    EXT_SKR03_3970,
    EXT_SKR03_3980,
    EXT_SKR03_4100,
    EXT_SKR03_4110,
    EXT_SKR03_4120,
    EXT_SKR03_4124,
    EXT_SKR03_4125,
    EXT_SKR03_4126,
    EXT_SKR03_4127,
    EXT_SKR03_4128,
    EXT_SKR03_4129,
    EXT_SKR03_4130,
    EXT_SKR03_4137,
    EXT_SKR03_4138,
    EXT_SKR03_4139,
    EXT_SKR03_4140,
    EXT_SKR03_4141,
    EXT_SKR03_4144,
    EXT_SKR03_4145,
    EXT_SKR03_4146,
    EXT_SKR03_4147,
    EXT_SKR03_4148,
    EXT_SKR03_4149,
    EXT_SKR03_4150,
    EXT_SKR03_4151,
    EXT_SKR03_4152,
    EXT_SKR03_4153,
    EXT_SKR03_4154,
    EXT_SKR03_4155,
    EXT_SKR03_4156,
    EXT_SKR03_4157,
    EXT_SKR03_4158,
    EXT_SKR03_4159,
    EXT_SKR03_4160,
    EXT_SKR03_4165,
    EXT_SKR03_4166,
    EXT_SKR03_4167,
    EXT_SKR03_4168,
    EXT_SKR03_4169,
    EXT_SKR03_4170,
    EXT_SKR03_4175,
    EXT_SKR03_4180,
    EXT_SKR03_4190,
    EXT_SKR03_4194,
    EXT_SKR03_4195,
    EXT_SKR03_4196,
    EXT_SKR03_4197,
    EXT_SKR03_4198,
    EXT_SKR03_4199,
    EXT_SKR03_4200,
    EXT_SKR03_4210,
    EXT_SKR03_4211,
    EXT_SKR03_4212,
    EXT_SKR03_4215,
    EXT_SKR03_4219,
    EXT_SKR03_4220,
    EXT_SKR03_4222,
    EXT_SKR03_4228,
    EXT_SKR03_4229,
    EXT_SKR03_4230,
    EXT_SKR03_4240,
    EXT_SKR03_4250,
    EXT_SKR03_4260,
    EXT_SKR03_4270,
    EXT_SKR03_4280,
    EXT_SKR03_4288,
    EXT_SKR03_4289,
    EXT_SKR03_4290,
    EXT_SKR03_4300,
    EXT_SKR03_4301,
    EXT_SKR03_4306,
    EXT_SKR03_4320,
    EXT_SKR03_4340,
    EXT_SKR03_4350,
    EXT_SKR03_4355,
    EXT_SKR03_4360,
    EXT_SKR03_4366,
    EXT_SKR03_4370,
    EXT_SKR03_4380,
    EXT_SKR03_4390,
    EXT_SKR03_4396,
    EXT_SKR03_4397,
    EXT_SKR03_4500,
    EXT_SKR03_4510,
    EXT_SKR03_4520,
    EXT_SKR03_4530,
    EXT_SKR03_4540,
    EXT_SKR03_4550,
    EXT_SKR03_4560,
    EXT_SKR03_4570,
    EXT_SKR03_4580,
    EXT_SKR03_4590,
    EXT_SKR03_4595,
    EXT_SKR03_4600,
    EXT_SKR03_4605,
    EXT_SKR03_4630,
    EXT_SKR03_4631,
    EXT_SKR03_4632,
    EXT_SKR03_4633,
    EXT_SKR03_4635,
    EXT_SKR03_4636,
    EXT_SKR03_4637,
    EXT_SKR03_4638,
    EXT_SKR03_4639,
    EXT_SKR03_4640,
    EXT_SKR03_4650,
    EXT_SKR03_4651,
    EXT_SKR03_4652,
    EXT_SKR03_4653,
    EXT_SKR03_4654,
    EXT_SKR03_4655,
    EXT_SKR03_4660,
    EXT_SKR03_4663,
    EXT_SKR03_4664,
    EXT_SKR03_4666,
    EXT_SKR03_4668,
    EXT_SKR03_4670,
    EXT_SKR03_4672,
    EXT_SKR03_4673,
    EXT_SKR03_4674,
    EXT_SKR03_4676,
    EXT_SKR03_4678,
    EXT_SKR03_4679,
    EXT_SKR03_4680,
    EXT_SKR03_4681,
    EXT_SKR03_4700,
    EXT_SKR03_4710,
    EXT_SKR03_4730,
    EXT_SKR03_4750,
    EXT_SKR03_4760,
    EXT_SKR03_4780,
    EXT_SKR03_4790,
    EXT_SKR03_4800,
    EXT_SKR03_4801,
    EXT_SKR03_4805,
    EXT_SKR03_4806,
    EXT_SKR03_4808,
    EXT_SKR03_4809,
    EXT_SKR03_4810,
    EXT_SKR03_4815,
    EXT_SKR03_4820,
    EXT_SKR03_4822,
    EXT_SKR03_4823,
    EXT_SKR03_4824,
    EXT_SKR03_4825,
    EXT_SKR03_4826,
    EXT_SKR03_4827,
    EXT_SKR03_4830,
    EXT_SKR03_4831,
    EXT_SKR03_4832,
    EXT_SKR03_4833,
    EXT_SKR03_4840,
    EXT_SKR03_4841,
    EXT_SKR03_4842,
    EXT_SKR03_4843,
    EXT_SKR03_4850,
    EXT_SKR03_4851,
    EXT_SKR03_4852,
    EXT_SKR03_4853,
    EXT_SKR03_4854,
    EXT_SKR03_4855,
    EXT_SKR03_4860,
    EXT_SKR03_4862,
    EXT_SKR03_4865,
    EXT_SKR03_4866,
    EXT_SKR03_4870,
    EXT_SKR03_4871,
    EXT_SKR03_4872,
    EXT_SKR03_4873,
    EXT_SKR03_4874,
    EXT_SKR03_4875,
    EXT_SKR03_4876,
    EXT_SKR03_4877,
    EXT_SKR03_4878,
    EXT_SKR03_4880,
    EXT_SKR03_4882,
    EXT_SKR03_4886,
    EXT_SKR03_4887,
    EXT_SKR03_4892,
    EXT_SKR03_4893,
    EXT_SKR03_4900,
    EXT_SKR03_4902,
    EXT_SKR03_4905,
    EXT_SKR03_4909,
    EXT_SKR03_4910,
    EXT_SKR03_4920,
    EXT_SKR03_4925,
    EXT_SKR03_4930,
    EXT_SKR03_4940,
    EXT_SKR03_4945,
    EXT_SKR03_4946,
    EXT_SKR03_4948,
    EXT_SKR03_4949,
    EXT_SKR03_4950,
    EXT_SKR03_4955,
    EXT_SKR03_4957,
    EXT_SKR03_4958,
    EXT_SKR03_4959,
    EXT_SKR03_4960,
    EXT_SKR03_4961,
    EXT_SKR03_4963,
    EXT_SKR03_4964,
    EXT_SKR03_4965,
    EXT_SKR03_4969,
    EXT_SKR03_4970,
    EXT_SKR03_4975,
    EXT_SKR03_4976,
    EXT_SKR03_4980,
    EXT_SKR03_4984,
    EXT_SKR03_4985,
    EXT_SKR03_4990,
    EXT_SKR03_4991,
    EXT_SKR03_4992,
    EXT_SKR03_4993,
    EXT_SKR03_4994,
    EXT_SKR03_4995,
    EXT_SKR03_4996,
    EXT_SKR03_4997,
    EXT_SKR03_4998,
    EXT_SKR03_4999,
    EXT_SKR03_7000,
    EXT_SKR03_7050,
    EXT_SKR03_7080,
    EXT_SKR03_7090,
    EXT_SKR03_7095,
    EXT_SKR03_7100,
    EXT_SKR03_7110,
    EXT_SKR03_7140,
    EXT_SKR03_7200,
    EXT_SKR03_8100,
    EXT_SKR03_8105,
    EXT_SKR03_8110,
    EXT_SKR03_8120,
    EXT_SKR03_8125,
    EXT_SKR03_8130,
    EXT_SKR03_8135,
    EXT_SKR03_8140,
    EXT_SKR03_8150,
    EXT_SKR03_8160,
    EXT_SKR03_8165,
    EXT_SKR03_8190,
    EXT_SKR03_8191,
    EXT_SKR03_8193,
    EXT_SKR03_8194,
    EXT_SKR03_8195,
    EXT_SKR03_8196,
    EXT_SKR03_8200,
    EXT_SKR03_8300,
    EXT_SKR03_8310,
    EXT_SKR03_8315,
    EXT_SKR03_8320,
    EXT_SKR03_8331,
    EXT_SKR03_8335,
    EXT_SKR03_8336,
    EXT_SKR03_8337,
    EXT_SKR03_8338,
    EXT_SKR03_8339,
    EXT_SKR03_8400,
    EXT_SKR03_8410,
    EXT_SKR03_8499,
    EXT_SKR03_8500,
    EXT_SKR03_8501,
    EXT_SKR03_8502,
    EXT_SKR03_8503,
    EXT_SKR03_8504,
    EXT_SKR03_8505,
    EXT_SKR03_8510,
    EXT_SKR03_8514,
    EXT_SKR03_8515,
    EXT_SKR03_8516,
    EXT_SKR03_8519,
    EXT_SKR03_8520,
    EXT_SKR03_8540,
    EXT_SKR03_8570,
    EXT_SKR03_8574,
    EXT_SKR03_8575,
    EXT_SKR03_8576,
    EXT_SKR03_8579,
    EXT_SKR03_8589,
    EXT_SKR03_8590,
    EXT_SKR03_8591,
    EXT_SKR03_8595,
    EXT_SKR03_8600,
    EXT_SKR03_8603,
    EXT_SKR03_8604,
    EXT_SKR03_8605,
    EXT_SKR03_8606,
    EXT_SKR03_8607,
    EXT_SKR03_8609,
    EXT_SKR03_8610,
    EXT_SKR03_8611,
    EXT_SKR03_8613,
    EXT_SKR03_8614,
    EXT_SKR03_8625,
    EXT_SKR03_8630,
    EXT_SKR03_8640,
    EXT_SKR03_8650,
    EXT_SKR03_8660,
    EXT_SKR03_8700,
    EXT_SKR03_8701,
    EXT_SKR03_8702,
    EXT_SKR03_8703,
    EXT_SKR03_8704,
    EXT_SKR03_8705,
    EXT_SKR03_8710,
    EXT_SKR03_8720,
    EXT_SKR03_8724,
    EXT_SKR03_8725,
    EXT_SKR03_8726,
    EXT_SKR03_8727,
    EXT_SKR03_8730,
    EXT_SKR03_8731,
    EXT_SKR03_8736,
    EXT_SKR03_8738,
    EXT_SKR03_8741,
    EXT_SKR03_8742,
    EXT_SKR03_8743,
    EXT_SKR03_8745,
    EXT_SKR03_8746,
    EXT_SKR03_8748,
    EXT_SKR03_8750,
    EXT_SKR03_8760,
    EXT_SKR03_8769,
    EXT_SKR03_8770,
    EXT_SKR03_8780,
    EXT_SKR03_8790,
    EXT_SKR03_8800,
    EXT_SKR03_8801,
    EXT_SKR03_8807,
    EXT_SKR03_8808,
    EXT_SKR03_8817,
    EXT_SKR03_8818,
    EXT_SKR03_8819,
    EXT_SKR03_8820,
    EXT_SKR03_8827,
    EXT_SKR03_8828,
    EXT_SKR03_8829,
    EXT_SKR03_8837,
    EXT_SKR03_8838,
    EXT_SKR03_8839,
    EXT_SKR03_8850,
    EXT_SKR03_8851,
    EXT_SKR03_8852,
    EXT_SKR03_8853,
    EXT_SKR03_8900,
    EXT_SKR03_8905,
    EXT_SKR03_8906,
    EXT_SKR03_8910,
    EXT_SKR03_8915,
    EXT_SKR03_8918,
    EXT_SKR03_8919,
    EXT_SKR03_8920,
    EXT_SKR03_8921,
    EXT_SKR03_8922,
    EXT_SKR03_8924,
    EXT_SKR03_8925,
    EXT_SKR03_8929,
    EXT_SKR03_8930,
    EXT_SKR03_8932,
    EXT_SKR03_8935,
    EXT_SKR03_8939,
    EXT_SKR03_8940,
    EXT_SKR03_8945,
    EXT_SKR03_8949,
    EXT_SKR03_8950,
    EXT_SKR03_8955,
    EXT_SKR03_8959,
    EXT_SKR03_8960,
    EXT_SKR03_8970,
    EXT_SKR03_8975,
    EXT_SKR03_8977,
    EXT_SKR03_8980,
    EXT_SKR03_8990,
    EXT_SKR03_8994,
    EXT_SKR03_8995,
    EXT_SKR03_9000,
    EXT_SKR03_9001,
    EXT_SKR03_9008,
    EXT_SKR03_9009,
    EXT_SKR03_9089,
    EXT_SKR03_9090,
];

/// All SKR04 account templates — canonical iteration handle.
pub static SKR04_CHART: &[OdooAccountTemplate] = &[
    EXT_SKR04_0050,
    EXT_SKR04_0060,
    EXT_SKR04_0070,
    EXT_SKR04_0080,
    EXT_SKR04_0090,
    EXT_SKR04_0100,
    EXT_SKR04_0110,
    EXT_SKR04_0120,
    EXT_SKR04_0130,
    EXT_SKR04_0135,
    EXT_SKR04_0140,
    EXT_SKR04_0143,
    EXT_SKR04_0144,
    EXT_SKR04_0145,
    EXT_SKR04_0146,
    EXT_SKR04_0147,
    EXT_SKR04_0148,
    EXT_SKR04_0150,
    EXT_SKR04_0170,
    EXT_SKR04_0179,
    EXT_SKR04_0200,
    EXT_SKR04_0210,
    EXT_SKR04_0215,
    EXT_SKR04_0220,
    EXT_SKR04_0225,
    EXT_SKR04_0230,
    EXT_SKR04_0235,
    EXT_SKR04_0240,
    EXT_SKR04_0250,
    EXT_SKR04_0260,
    EXT_SKR04_0270,
    EXT_SKR04_0280,
    EXT_SKR04_0285,
    EXT_SKR04_0290,
    EXT_SKR04_0300,
    EXT_SKR04_0305,
    EXT_SKR04_0310,
    EXT_SKR04_0315,
    EXT_SKR04_0320,
    EXT_SKR04_0329,
    EXT_SKR04_0330,
    EXT_SKR04_0340,
    EXT_SKR04_0350,
    EXT_SKR04_0360,
    EXT_SKR04_0370,
    EXT_SKR04_0380,
    EXT_SKR04_0390,
    EXT_SKR04_0395,
    EXT_SKR04_0398,
    EXT_SKR04_0400,
    EXT_SKR04_0420,
    EXT_SKR04_0440,
    EXT_SKR04_0450,
    EXT_SKR04_0460,
    EXT_SKR04_0470,
    EXT_SKR04_0500,
    EXT_SKR04_0510,
    EXT_SKR04_0520,
    EXT_SKR04_0540,
    EXT_SKR04_0560,
    EXT_SKR04_0620,
    EXT_SKR04_0630,
    EXT_SKR04_0635,
    EXT_SKR04_0640,
    EXT_SKR04_0650,
    EXT_SKR04_0660,
    EXT_SKR04_0670,
    EXT_SKR04_0675,
    EXT_SKR04_0680,
    EXT_SKR04_0690,
    EXT_SKR04_0700,
    EXT_SKR04_0705,
    EXT_SKR04_0710,
    EXT_SKR04_0720,
    EXT_SKR04_0725,
    EXT_SKR04_0735,
    EXT_SKR04_0740,
    EXT_SKR04_0755,
    EXT_SKR04_0765,
    EXT_SKR04_0770,
    EXT_SKR04_0780,
    EXT_SKR04_0785,
    EXT_SKR04_0795,
    EXT_SKR04_0800,
    EXT_SKR04_0803,
    EXT_SKR04_0804,
    EXT_SKR04_0805,
    EXT_SKR04_0808,
    EXT_SKR04_0809,
    EXT_SKR04_0810,
    EXT_SKR04_0813,
    EXT_SKR04_0814,
    EXT_SKR04_0815,
    EXT_SKR04_0820,
    EXT_SKR04_0829,
    EXT_SKR04_0830,
    EXT_SKR04_0840,
    EXT_SKR04_0850,
    EXT_SKR04_0860,
    EXT_SKR04_0880,
    EXT_SKR04_0883,
    EXT_SKR04_0885,
    EXT_SKR04_0900,
    EXT_SKR04_0910,
    EXT_SKR04_0920,
    EXT_SKR04_0930,
    EXT_SKR04_0940,
    EXT_SKR04_0960,
    EXT_SKR04_0961,
    EXT_SKR04_0962,
    EXT_SKR04_0963,
    EXT_SKR04_0964,
    EXT_SKR04_0970,
    EXT_SKR04_0980,
    EXT_SKR04_1000,
    EXT_SKR04_1040,
    EXT_SKR04_1050,
    EXT_SKR04_1080,
    EXT_SKR04_1090,
    EXT_SKR04_1095,
    EXT_SKR04_1100,
    EXT_SKR04_1110,
    EXT_SKR04_1140,
    EXT_SKR04_1180,
    EXT_SKR04_1181,
    EXT_SKR04_1186,
    EXT_SKR04_1190,
    EXT_SKR04_1200,
    EXT_SKR04_1205,
    EXT_SKR04_1206,
    EXT_SKR04_1210,
    EXT_SKR04_1215,
    EXT_SKR04_1216,
    EXT_SKR04_1217,
    EXT_SKR04_1218,
    EXT_SKR04_1219,
    EXT_SKR04_1220,
    EXT_SKR04_1221,
    EXT_SKR04_1225,
    EXT_SKR04_1230,
    EXT_SKR04_1231,
    EXT_SKR04_1232,
    EXT_SKR04_1235,
    EXT_SKR04_1240,
    EXT_SKR04_1241,
    EXT_SKR04_1245,
    EXT_SKR04_1246,
    EXT_SKR04_1247,
    EXT_SKR04_1248,
    EXT_SKR04_1249,
    EXT_SKR04_1250,
    EXT_SKR04_1251,
    EXT_SKR04_1255,
    EXT_SKR04_1260,
    EXT_SKR04_1261,
    EXT_SKR04_1265,
    EXT_SKR04_1266,
    EXT_SKR04_1267,
    EXT_SKR04_1268,
    EXT_SKR04_1269,
    EXT_SKR04_1270,
    EXT_SKR04_1271,
    EXT_SKR04_1275,
    EXT_SKR04_1276,
    EXT_SKR04_1277,
    EXT_SKR04_1280,
    EXT_SKR04_1281,
    EXT_SKR04_1285,
    EXT_SKR04_1286,
    EXT_SKR04_1287,
    EXT_SKR04_1288,
    EXT_SKR04_1289,
    EXT_SKR04_1290,
    EXT_SKR04_1291,
    EXT_SKR04_1295,
    EXT_SKR04_1296,
    EXT_SKR04_1297,
    EXT_SKR04_1298,
    EXT_SKR04_1299,
    EXT_SKR04_1300,
    EXT_SKR04_1301,
    EXT_SKR04_1305,
    EXT_SKR04_1307,
    EXT_SKR04_1308,
    EXT_SKR04_1309,
    EXT_SKR04_1310,
    EXT_SKR04_1311,
    EXT_SKR04_1315,
    EXT_SKR04_1317,
    EXT_SKR04_1318,
    EXT_SKR04_1319,
    EXT_SKR04_1320,
    EXT_SKR04_1321,
    EXT_SKR04_1325,
    EXT_SKR04_1327,
    EXT_SKR04_1328,
    EXT_SKR04_1329,
    EXT_SKR04_1330,
    EXT_SKR04_1331,
    EXT_SKR04_1335,
    EXT_SKR04_1337,
    EXT_SKR04_1338,
    EXT_SKR04_1339,
    EXT_SKR04_1340,
    EXT_SKR04_1341,
    EXT_SKR04_1345,
    EXT_SKR04_1350,
    EXT_SKR04_1351,
    EXT_SKR04_1355,
    EXT_SKR04_1360,
    EXT_SKR04_1361,
    EXT_SKR04_1365,
    EXT_SKR04_1369,
    EXT_SKR04_1370,
    EXT_SKR04_1374,
    EXT_SKR04_1375,
    EXT_SKR04_1378,
    EXT_SKR04_1380,
    EXT_SKR04_1381,
    EXT_SKR04_1382,
    EXT_SKR04_1383,
    EXT_SKR04_1390,
    EXT_SKR04_1391,
    EXT_SKR04_1393,
    EXT_SKR04_1394,
    EXT_SKR04_1395,
    EXT_SKR04_1396,
    EXT_SKR04_1397,
    EXT_SKR04_1398,
    EXT_SKR04_1399,
    EXT_SKR04_1400,
    EXT_SKR04_1401,
    EXT_SKR04_1402,
    EXT_SKR04_1404,
    EXT_SKR04_1406,
    EXT_SKR04_1407,
    EXT_SKR04_1408,
    EXT_SKR04_1410,
    EXT_SKR04_1411,
    EXT_SKR04_1412,
    EXT_SKR04_1413,
    EXT_SKR04_1416,
    EXT_SKR04_1417,
    EXT_SKR04_1419,
    EXT_SKR04_1420,
    EXT_SKR04_1421,
    EXT_SKR04_1422,
    EXT_SKR04_1425,
    EXT_SKR04_1427,
    EXT_SKR04_1431,
    EXT_SKR04_1432,
    EXT_SKR04_1433,
    EXT_SKR04_1434,
    EXT_SKR04_1435,
    EXT_SKR04_1436,
    EXT_SKR04_1440,
    EXT_SKR04_1450,
    EXT_SKR04_1456,
    EXT_SKR04_1457,
    EXT_SKR04_1480,
    EXT_SKR04_1481,
    EXT_SKR04_1482,
    EXT_SKR04_1483,
    EXT_SKR04_1485,
    EXT_SKR04_1486,
    EXT_SKR04_1490,
    EXT_SKR04_1495,
    EXT_SKR04_1498,
    EXT_SKR04_1500,
    EXT_SKR04_1504,
    EXT_SKR04_1510,
    EXT_SKR04_1520,
    EXT_SKR04_1525,
    EXT_SKR04_1530,
    EXT_SKR04_1550,
    EXT_SKR04_1610,
    EXT_SKR04_1620,
    EXT_SKR04_1700,
    EXT_SKR04_1710,
    EXT_SKR04_1720,
    EXT_SKR04_1730,
    EXT_SKR04_1780,
    EXT_SKR04_1790,
    EXT_SKR04_1810,
    EXT_SKR04_1820,
    EXT_SKR04_1830,
    EXT_SKR04_1840,
    EXT_SKR04_1850,
    EXT_SKR04_1890,
    EXT_SKR04_1900,
    EXT_SKR04_1920,
    EXT_SKR04_1930,
    EXT_SKR04_1940,
    EXT_SKR04_1950,
    EXT_SKR04_2000,
    EXT_SKR04_2010,
    EXT_SKR04_2020,
    EXT_SKR04_2050,
    EXT_SKR04_2060,
    EXT_SKR04_2070,
    EXT_SKR04_2100,
    EXT_SKR04_2130,
    EXT_SKR04_2150,
    EXT_SKR04_2180,
    EXT_SKR04_2200,
    EXT_SKR04_2230,
    EXT_SKR04_2250,
    EXT_SKR04_2280,
    EXT_SKR04_2300,
    EXT_SKR04_2349,
    EXT_SKR04_2350,
    EXT_SKR04_2399,
    EXT_SKR04_2500,
    EXT_SKR04_2530,
    EXT_SKR04_2550,
    EXT_SKR04_2580,
    EXT_SKR04_2600,
    EXT_SKR04_2630,
    EXT_SKR04_2650,
    EXT_SKR04_2680,
    EXT_SKR04_2700,
    EXT_SKR04_2750,
    EXT_SKR04_2900,
    EXT_SKR04_2901,
    EXT_SKR04_2902,
    EXT_SKR04_2903,
    EXT_SKR04_2906,
    EXT_SKR04_2907,
    EXT_SKR04_2908,
    EXT_SKR04_2909,
    EXT_SKR04_2910,
    EXT_SKR04_2920,
    EXT_SKR04_2925,
    EXT_SKR04_2926,
    EXT_SKR04_2927,
    EXT_SKR04_2928,
    EXT_SKR04_2929,
    EXT_SKR04_2930,
    EXT_SKR04_2935,
    EXT_SKR04_2937,
    EXT_SKR04_2950,
    EXT_SKR04_2959,
    EXT_SKR04_2960,
    EXT_SKR04_2961,
    EXT_SKR04_2962,
    EXT_SKR04_2963,
    EXT_SKR04_2964,
    EXT_SKR04_2965,
    EXT_SKR04_2966,
    EXT_SKR04_2967,
    EXT_SKR04_2968,
    EXT_SKR04_2969,
    EXT_SKR04_2970,
    EXT_SKR04_2975,
    EXT_SKR04_2977,
    EXT_SKR04_2978,
    EXT_SKR04_2979,
    EXT_SKR04_3000,
    EXT_SKR04_3005,
    EXT_SKR04_3009,
    EXT_SKR04_3010,
    EXT_SKR04_3011,
    EXT_SKR04_3015,
    EXT_SKR04_3020,
    EXT_SKR04_3030,
    EXT_SKR04_3035,
    EXT_SKR04_3040,
    EXT_SKR04_3050,
    EXT_SKR04_3060,
    EXT_SKR04_3065,
    EXT_SKR04_3070,
    EXT_SKR04_3075,
    EXT_SKR04_3076,
    EXT_SKR04_3077,
    EXT_SKR04_3079,
    EXT_SKR04_3085,
    EXT_SKR04_3090,
    EXT_SKR04_3092,
    EXT_SKR04_3095,
    EXT_SKR04_3098,
    EXT_SKR04_3099,
    EXT_SKR04_3100,
    EXT_SKR04_3101,
    EXT_SKR04_3105,
    EXT_SKR04_3110,
    EXT_SKR04_3120,
    EXT_SKR04_3121,
    EXT_SKR04_3125,
    EXT_SKR04_3130,
    EXT_SKR04_3150,
    EXT_SKR04_3151,
    EXT_SKR04_3160,
    EXT_SKR04_3170,
    EXT_SKR04_3180,
    EXT_SKR04_3181,
    EXT_SKR04_3190,
    EXT_SKR04_3200,
    EXT_SKR04_3250,
    EXT_SKR04_3260,
    EXT_SKR04_3272,
    EXT_SKR04_3280,
    EXT_SKR04_3284,
    EXT_SKR04_3285,
    EXT_SKR04_3300,
    EXT_SKR04_3301,
    EXT_SKR04_3305,
    EXT_SKR04_3306,
    EXT_SKR04_3307,
    EXT_SKR04_3309,
    EXT_SKR04_3310,
    EXT_SKR04_3334,
    EXT_SKR04_3335,
    EXT_SKR04_3337,
    EXT_SKR04_3338,
    EXT_SKR04_3340,
    EXT_SKR04_3341,
    EXT_SKR04_3345,
    EXT_SKR04_3348,
    EXT_SKR04_3350,
    EXT_SKR04_3351,
    EXT_SKR04_3380,
    EXT_SKR04_3390,
    EXT_SKR04_3400,
    EXT_SKR04_3401,
    EXT_SKR04_3405,
    EXT_SKR04_3410,
    EXT_SKR04_3420,
    EXT_SKR04_3421,
    EXT_SKR04_3425,
    EXT_SKR04_3430,
    EXT_SKR04_3450,
    EXT_SKR04_3451,
    EXT_SKR04_3455,
    EXT_SKR04_3460,
    EXT_SKR04_3470,
    EXT_SKR04_3471,
    EXT_SKR04_3475,
    EXT_SKR04_3480,
    EXT_SKR04_3500,
    EXT_SKR04_3501,
    EXT_SKR04_3504,
    EXT_SKR04_3507,
    EXT_SKR04_3509,
    EXT_SKR04_3510,
    EXT_SKR04_3511,
    EXT_SKR04_3514,
    EXT_SKR04_3517,
    EXT_SKR04_3519,
    EXT_SKR04_3520,
    EXT_SKR04_3521,
    EXT_SKR04_3524,
    EXT_SKR04_3527,
    EXT_SKR04_3530,
    EXT_SKR04_3531,
    EXT_SKR04_3534,
    EXT_SKR04_3537,
    EXT_SKR04_3540,
    EXT_SKR04_3541,
    EXT_SKR04_3544,
    EXT_SKR04_3547,
    EXT_SKR04_3550,
    EXT_SKR04_3551,
    EXT_SKR04_3554,
    EXT_SKR04_3557,
    EXT_SKR04_3560,
    EXT_SKR04_3561,
    EXT_SKR04_3564,
    EXT_SKR04_3567,
    EXT_SKR04_3600,
    EXT_SKR04_3610,
    EXT_SKR04_3611,
    EXT_SKR04_3620,
    EXT_SKR04_3630,
    EXT_SKR04_3635,
    EXT_SKR04_3640,
    EXT_SKR04_3641,
    EXT_SKR04_3642,
    EXT_SKR04_3643,
    EXT_SKR04_3645,
    EXT_SKR04_3646,
    EXT_SKR04_3647,
    EXT_SKR04_3648,
    EXT_SKR04_3650,
    EXT_SKR04_3651,
    EXT_SKR04_3652,
    EXT_SKR04_3653,
    EXT_SKR04_3655,
    EXT_SKR04_3656,
    EXT_SKR04_3657,
    EXT_SKR04_3658,
    EXT_SKR04_3695,
    EXT_SKR04_3700,
    EXT_SKR04_3701,
    EXT_SKR04_3710,
    EXT_SKR04_3715,
    EXT_SKR04_3720,
    EXT_SKR04_3725,
    EXT_SKR04_3726,
    EXT_SKR04_3730,
    EXT_SKR04_3740,
    EXT_SKR04_3741,
    EXT_SKR04_3750,
    EXT_SKR04_3755,
    EXT_SKR04_3759,
    EXT_SKR04_3760,
    EXT_SKR04_3761,
    EXT_SKR04_3770,
    EXT_SKR04_3771,
    EXT_SKR04_3780,
    EXT_SKR04_3785,
    EXT_SKR04_3786,
    EXT_SKR04_3790,
    EXT_SKR04_3796,
    EXT_SKR04_3798,
    EXT_SKR04_3799,
    EXT_SKR04_3800,
    EXT_SKR04_3801,
    EXT_SKR04_3802,
    EXT_SKR04_3804,
    EXT_SKR04_3806,
    EXT_SKR04_3807,
    EXT_SKR04_3808,
    EXT_SKR04_3809,
    EXT_SKR04_3810,
    EXT_SKR04_3811,
    EXT_SKR04_3812,
    EXT_SKR04_3814,
    EXT_SKR04_3816,
    EXT_SKR04_3817,
    EXT_SKR04_3818,
    EXT_SKR04_3819,
    EXT_SKR04_3820,
    EXT_SKR04_3830,
    EXT_SKR04_3832,
    EXT_SKR04_3834,
    EXT_SKR04_3835,
    EXT_SKR04_3837,
    EXT_SKR04_3840,
    EXT_SKR04_3841,
    EXT_SKR04_3845,
    EXT_SKR04_3850,
    EXT_SKR04_3851,
    EXT_SKR04_3854,
    EXT_SKR04_3860,
    EXT_SKR04_3865,
    EXT_SKR04_3900,
    EXT_SKR04_3950,
    EXT_SKR04_4000,
    EXT_SKR04_4100,
    EXT_SKR04_4110,
    EXT_SKR04_4120,
    EXT_SKR04_4125,
    EXT_SKR04_4130,
    EXT_SKR04_4135,
    EXT_SKR04_4136,
    EXT_SKR04_4138,
    EXT_SKR04_4139,
    EXT_SKR04_4140,
    EXT_SKR04_4150,
    EXT_SKR04_4160,
    EXT_SKR04_4165,
    EXT_SKR04_4180,
    EXT_SKR04_4185,
    EXT_SKR04_4186,
    EXT_SKR04_4200,
    EXT_SKR04_4300,
    EXT_SKR04_4310,
    EXT_SKR04_4315,
    EXT_SKR04_4320,
    EXT_SKR04_4331,
    EXT_SKR04_4335,
    EXT_SKR04_4336,
    EXT_SKR04_4337,
    EXT_SKR04_4338,
    EXT_SKR04_4339,
    EXT_SKR04_4400,
    EXT_SKR04_4499,
    EXT_SKR04_4510,
    EXT_SKR04_4520,
    EXT_SKR04_4560,
    EXT_SKR04_4564,
    EXT_SKR04_4566,
    EXT_SKR04_4569,
    EXT_SKR04_4570,
    EXT_SKR04_4574,
    EXT_SKR04_4575,
    EXT_SKR04_4576,
    EXT_SKR04_4579,
    EXT_SKR04_4600,
    EXT_SKR04_4605,
    EXT_SKR04_4610,
    EXT_SKR04_4616,
    EXT_SKR04_4619,
    EXT_SKR04_4620,
    EXT_SKR04_4630,
    EXT_SKR04_4639,
    EXT_SKR04_4640,
    EXT_SKR04_4645,
    EXT_SKR04_4646,
    EXT_SKR04_4650,
    EXT_SKR04_4659,
    EXT_SKR04_4660,
    EXT_SKR04_4670,
    EXT_SKR04_4679,
    EXT_SKR04_4680,
    EXT_SKR04_4686,
    EXT_SKR04_4689,
    EXT_SKR04_4690,
    EXT_SKR04_4695,
    EXT_SKR04_4699,
    EXT_SKR04_4700,
    EXT_SKR04_4701,
    EXT_SKR04_4702,
    EXT_SKR04_4703,
    EXT_SKR04_4704,
    EXT_SKR04_4705,
    EXT_SKR04_4710,
    EXT_SKR04_4720,
    EXT_SKR04_4724,
    EXT_SKR04_4725,
    EXT_SKR04_4726,
    EXT_SKR04_4727,
    EXT_SKR04_4730,
    EXT_SKR04_4731,
    EXT_SKR04_4736,
    EXT_SKR04_4738,
    EXT_SKR04_4741,
    EXT_SKR04_4742,
    EXT_SKR04_4743,
    EXT_SKR04_4745,
    EXT_SKR04_4746,
    EXT_SKR04_4748,
    EXT_SKR04_4750,
    EXT_SKR04_4760,
    EXT_SKR04_4770,
    EXT_SKR04_4780,
    EXT_SKR04_4790,
    EXT_SKR04_4800,
    EXT_SKR04_4810,
    EXT_SKR04_4815,
    EXT_SKR04_4816,
    EXT_SKR04_4818,
    EXT_SKR04_4820,
    EXT_SKR04_4824,
    EXT_SKR04_4825,
    EXT_SKR04_4830,
    EXT_SKR04_4832,
    EXT_SKR04_4833,
    EXT_SKR04_4835,
    EXT_SKR04_4836,
    EXT_SKR04_4837,
    EXT_SKR04_4838,
    EXT_SKR04_4839,
    EXT_SKR04_4840,
    EXT_SKR04_4841,
    EXT_SKR04_4842,
    EXT_SKR04_4843,
    EXT_SKR04_4844,
    EXT_SKR04_4845,
    EXT_SKR04_4847,
    EXT_SKR04_4848,
    EXT_SKR04_4849,
    EXT_SKR04_4850,
    EXT_SKR04_4851,
    EXT_SKR04_4852,
    EXT_SKR04_4855,
    EXT_SKR04_4856,
    EXT_SKR04_4857,
    EXT_SKR04_4858,
    EXT_SKR04_4860,
    EXT_SKR04_4861,
    EXT_SKR04_4862,
    EXT_SKR04_4900,
    EXT_SKR04_4901,
    EXT_SKR04_4905,
    EXT_SKR04_4906,
    EXT_SKR04_4910,
    EXT_SKR04_4911,
    EXT_SKR04_4912,
    EXT_SKR04_4913,
    EXT_SKR04_4914,
    EXT_SKR04_4915,
    EXT_SKR04_4916,
    EXT_SKR04_4920,
    EXT_SKR04_4923,
    EXT_SKR04_4925,
    EXT_SKR04_4927,
    EXT_SKR04_4928,
    EXT_SKR04_4929,
    EXT_SKR04_4930,
    EXT_SKR04_4932,
    EXT_SKR04_4934,
    EXT_SKR04_4935,
    EXT_SKR04_4936,
    EXT_SKR04_4937,
    EXT_SKR04_4938,
    EXT_SKR04_4939,
    EXT_SKR04_4940,
    EXT_SKR04_4941,
    EXT_SKR04_4945,
    EXT_SKR04_4947,
    EXT_SKR04_4948,
    EXT_SKR04_4949,
    EXT_SKR04_4960,
    EXT_SKR04_4970,
    EXT_SKR04_4972,
    EXT_SKR04_4975,
    EXT_SKR04_4980,
    EXT_SKR04_4981,
    EXT_SKR04_4982,
    EXT_SKR04_4987,
    EXT_SKR04_4989,
    EXT_SKR04_4992,
    EXT_SKR04_5000,
    EXT_SKR04_5100,
    EXT_SKR04_5110,
    EXT_SKR04_5130,
    EXT_SKR04_5160,
    EXT_SKR04_5162,
    EXT_SKR04_5166,
    EXT_SKR04_5167,
    EXT_SKR04_5189,
    EXT_SKR04_5190,
    EXT_SKR04_5191,
    EXT_SKR04_5192,
    EXT_SKR04_5200,
    EXT_SKR04_5300,
    EXT_SKR04_5349,
    EXT_SKR04_5400,
    EXT_SKR04_5420,
    EXT_SKR04_5425,
    EXT_SKR04_5430,
    EXT_SKR04_5435,
    EXT_SKR04_5440,
    EXT_SKR04_5550,
    EXT_SKR04_5551,
    EXT_SKR04_5552,
    EXT_SKR04_5553,
    EXT_SKR04_5557,
    EXT_SKR04_5559,
    EXT_SKR04_5600,
    EXT_SKR04_5610,
    EXT_SKR04_5660,
    EXT_SKR04_5700,
    EXT_SKR04_5701,
    EXT_SKR04_5710,
    EXT_SKR04_5714,
    EXT_SKR04_5715,
    EXT_SKR04_5717,
    EXT_SKR04_5718,
    EXT_SKR04_5720,
    EXT_SKR04_5724,
    EXT_SKR04_5725,
    EXT_SKR04_5730,
    EXT_SKR04_5731,
    EXT_SKR04_5733,
    EXT_SKR04_5734,
    EXT_SKR04_5736,
    EXT_SKR04_5738,
    EXT_SKR04_5741,
    EXT_SKR04_5743,
    EXT_SKR04_5744,
    EXT_SKR04_5750,
    EXT_SKR04_5753,
    EXT_SKR04_5754,
    EXT_SKR04_5755,
    EXT_SKR04_5760,
    EXT_SKR04_5770,
    EXT_SKR04_5780,
    EXT_SKR04_5783,
    EXT_SKR04_5784,
    EXT_SKR04_5785,
    EXT_SKR04_5788,
    EXT_SKR04_5790,
    EXT_SKR04_5792,
    EXT_SKR04_5793,
    EXT_SKR04_5794,
    EXT_SKR04_5796,
    EXT_SKR04_5798,
    EXT_SKR04_5800,
    EXT_SKR04_5820,
    EXT_SKR04_5840,
    EXT_SKR04_5880,
    EXT_SKR04_5881,
    EXT_SKR04_5885,
    EXT_SKR04_5900,
    EXT_SKR04_5906,
    EXT_SKR04_5908,
    EXT_SKR04_5909,
    EXT_SKR04_5910,
    EXT_SKR04_5913,
    EXT_SKR04_5915,
    EXT_SKR04_5920,
    EXT_SKR04_5923,
    EXT_SKR04_5925,
    EXT_SKR04_5950,
    EXT_SKR04_5951,
    EXT_SKR04_5970,
    EXT_SKR04_5975,
    EXT_SKR04_5980,
    EXT_SKR04_5985,
    EXT_SKR04_6000,
    EXT_SKR04_6010,
    EXT_SKR04_6020,
    EXT_SKR04_6024,
    EXT_SKR04_6026,
    EXT_SKR04_6027,
    EXT_SKR04_6028,
    EXT_SKR04_6029,
    EXT_SKR04_6030,
    EXT_SKR04_6035,
    EXT_SKR04_6036,
    EXT_SKR04_6037,
    EXT_SKR04_6038,
    EXT_SKR04_6039,
    EXT_SKR04_6040,
    EXT_SKR04_6045,
    EXT_SKR04_6050,
    EXT_SKR04_6060,
    EXT_SKR04_6066,
    EXT_SKR04_6067,
    EXT_SKR04_6068,
    EXT_SKR04_6069,
    EXT_SKR04_6070,
    EXT_SKR04_6071,
    EXT_SKR04_6072,
    EXT_SKR04_6073,
    EXT_SKR04_6074,
    EXT_SKR04_6075,
    EXT_SKR04_6076,
    EXT_SKR04_6077,
    EXT_SKR04_6078,
    EXT_SKR04_6079,
    EXT_SKR04_6080,
    EXT_SKR04_6090,
    EXT_SKR04_6100,
    EXT_SKR04_6110,
    EXT_SKR04_6118,
    EXT_SKR04_6120,
    EXT_SKR04_6130,
    EXT_SKR04_6140,
    EXT_SKR04_6147,
    EXT_SKR04_6148,
    EXT_SKR04_6149,
    EXT_SKR04_6150,
    EXT_SKR04_6160,
    EXT_SKR04_6170,
    EXT_SKR04_6171,
    EXT_SKR04_6200,
    EXT_SKR04_6201,
    EXT_SKR04_6205,
    EXT_SKR04_6209,
    EXT_SKR04_6210,
    EXT_SKR04_6211,
    EXT_SKR04_6220,
    EXT_SKR04_6221,
    EXT_SKR04_6222,
    EXT_SKR04_6223,
    EXT_SKR04_6230,
    EXT_SKR04_6231,
    EXT_SKR04_6232,
    EXT_SKR04_6233,
    EXT_SKR04_6240,
    EXT_SKR04_6241,
    EXT_SKR04_6242,
    EXT_SKR04_6243,
    EXT_SKR04_6244,
    EXT_SKR04_6250,
    EXT_SKR04_6260,
    EXT_SKR04_6262,
    EXT_SKR04_6264,
    EXT_SKR04_6266,
    EXT_SKR04_6268,
    EXT_SKR04_6270,
    EXT_SKR04_6272,
    EXT_SKR04_6278,
    EXT_SKR04_6279,
    EXT_SKR04_6280,
    EXT_SKR04_6281,
    EXT_SKR04_6286,
    EXT_SKR04_6290,
    EXT_SKR04_6291,
    EXT_SKR04_6300,
    EXT_SKR04_6302,
    EXT_SKR04_6303,
    EXT_SKR04_6304,
    EXT_SKR04_6305,
    EXT_SKR04_6310,
    EXT_SKR04_6312,
    EXT_SKR04_6313,
    EXT_SKR04_6314,
    EXT_SKR04_6315,
    EXT_SKR04_6316,
    EXT_SKR04_6317,
    EXT_SKR04_6319,
    EXT_SKR04_6320,
    EXT_SKR04_6325,
    EXT_SKR04_6330,
    EXT_SKR04_6335,
    EXT_SKR04_6340,
    EXT_SKR04_6345,
    EXT_SKR04_6348,
    EXT_SKR04_6349,
    EXT_SKR04_6350,
    EXT_SKR04_6390,
    EXT_SKR04_6391,
    EXT_SKR04_6392,
    EXT_SKR04_6393,
    EXT_SKR04_6394,
    EXT_SKR04_6395,
    EXT_SKR04_6396,
    EXT_SKR04_6397,
    EXT_SKR04_6398,
    EXT_SKR04_6400,
    EXT_SKR04_6405,
    EXT_SKR04_6410,
    EXT_SKR04_6420,
    EXT_SKR04_6430,
    EXT_SKR04_6436,
    EXT_SKR04_6437,
    EXT_SKR04_6440,
    EXT_SKR04_6450,
    EXT_SKR04_6460,
    EXT_SKR04_6470,
    EXT_SKR04_6475,
    EXT_SKR04_6485,
    EXT_SKR04_6490,
    EXT_SKR04_6495,
    EXT_SKR04_6498,
    EXT_SKR04_6500,
    EXT_SKR04_6520,
    EXT_SKR04_6530,
    EXT_SKR04_6540,
    EXT_SKR04_6550,
    EXT_SKR04_6560,
    EXT_SKR04_6570,
    EXT_SKR04_6580,
    EXT_SKR04_6590,
    EXT_SKR04_6595,
    EXT_SKR04_6600,
    EXT_SKR04_6605,
    EXT_SKR04_6610,
    EXT_SKR04_6611,
    EXT_SKR04_6612,
    EXT_SKR04_6620,
    EXT_SKR04_6621,
    EXT_SKR04_6625,
    EXT_SKR04_6629,
    EXT_SKR04_6630,
    EXT_SKR04_6640,
    EXT_SKR04_6641,
    EXT_SKR04_6642,
    EXT_SKR04_6643,
    EXT_SKR04_6644,
    EXT_SKR04_6645,
    EXT_SKR04_6650,
    EXT_SKR04_6660,
    EXT_SKR04_6663,
    EXT_SKR04_6664,
    EXT_SKR04_6668,
    EXT_SKR04_6670,
    EXT_SKR04_6673,
    EXT_SKR04_6674,
    EXT_SKR04_6680,
    EXT_SKR04_6688,
    EXT_SKR04_6689,
    EXT_SKR04_6690,
    EXT_SKR04_6691,
    EXT_SKR04_6700,
    EXT_SKR04_6710,
    EXT_SKR04_6740,
    EXT_SKR04_6760,
    EXT_SKR04_6770,
    EXT_SKR04_6780,
    EXT_SKR04_6790,
    EXT_SKR04_6800,
    EXT_SKR04_6805,
    EXT_SKR04_6810,
    EXT_SKR04_6815,
    EXT_SKR04_6820,
    EXT_SKR04_6821,
    EXT_SKR04_6822,
    EXT_SKR04_6823,
    EXT_SKR04_6824,
    EXT_SKR04_6825,
    EXT_SKR04_6827,
    EXT_SKR04_6830,
    EXT_SKR04_6833,
    EXT_SKR04_6834,
    EXT_SKR04_6835,
    EXT_SKR04_6837,
    EXT_SKR04_6838,
    EXT_SKR04_6840,
    EXT_SKR04_6845,
    EXT_SKR04_6850,
    EXT_SKR04_6854,
    EXT_SKR04_6855,
    EXT_SKR04_6856,
    EXT_SKR04_6857,
    EXT_SKR04_6859,
    EXT_SKR04_6860,
    EXT_SKR04_6865,
    EXT_SKR04_6871,
    EXT_SKR04_6875,
    EXT_SKR04_6876,
    EXT_SKR04_6880,
    EXT_SKR04_6881,
    EXT_SKR04_6883,
    EXT_SKR04_6884,
    EXT_SKR04_6885,
    EXT_SKR04_6888,
    EXT_SKR04_6889,
    EXT_SKR04_6890,
    EXT_SKR04_6891,
    EXT_SKR04_6892,
    EXT_SKR04_6895,
    EXT_SKR04_6896,
    EXT_SKR04_6897,
    EXT_SKR04_6898,
    EXT_SKR04_6900,
    EXT_SKR04_6903,
    EXT_SKR04_6905,
    EXT_SKR04_6906,
    EXT_SKR04_6910,
    EXT_SKR04_6912,
    EXT_SKR04_6918,
    EXT_SKR04_6920,
    EXT_SKR04_6922,
    EXT_SKR04_6923,
    EXT_SKR04_6924,
    EXT_SKR04_6927,
    EXT_SKR04_6928,
    EXT_SKR04_6929,
    EXT_SKR04_6930,
    EXT_SKR04_6931,
    EXT_SKR04_6932,
    EXT_SKR04_6933,
    EXT_SKR04_6936,
    EXT_SKR04_6938,
    EXT_SKR04_6960,
    EXT_SKR04_6967,
    EXT_SKR04_6968,
    EXT_SKR04_6969,
    EXT_SKR04_7000,
    EXT_SKR04_7004,
    EXT_SKR04_7005,
    EXT_SKR04_7006,
    EXT_SKR04_7008,
    EXT_SKR04_7009,
    EXT_SKR04_7010,
    EXT_SKR04_7011,
    EXT_SKR04_7012,
    EXT_SKR04_7013,
    EXT_SKR04_7014,
    EXT_SKR04_7015,
    EXT_SKR04_7016,
    EXT_SKR04_7017,
    EXT_SKR04_7018,
    EXT_SKR04_7019,
    EXT_SKR04_7020,
    EXT_SKR04_7030,
    EXT_SKR04_7100,
    EXT_SKR04_7103,
    EXT_SKR04_7104,
    EXT_SKR04_7105,
    EXT_SKR04_7106,
    EXT_SKR04_7107,
    EXT_SKR04_7109,
    EXT_SKR04_7110,
    EXT_SKR04_7119,
    EXT_SKR04_7120,
    EXT_SKR04_7129,
    EXT_SKR04_7130,
    EXT_SKR04_7139,
    EXT_SKR04_7140,
    EXT_SKR04_7141,
    EXT_SKR04_7142,
    EXT_SKR04_7143,
    EXT_SKR04_7144,
    EXT_SKR04_7145,
    EXT_SKR04_7190,
    EXT_SKR04_7192,
    EXT_SKR04_7194,
    EXT_SKR04_7200,
    EXT_SKR04_7201,
    EXT_SKR04_7204,
    EXT_SKR04_7207,
    EXT_SKR04_7208,
    EXT_SKR04_7210,
    EXT_SKR04_7214,
    EXT_SKR04_7217,
    EXT_SKR04_7250,
    EXT_SKR04_7255,
    EXT_SKR04_7300,
    EXT_SKR04_7302,
    EXT_SKR04_7303,
    EXT_SKR04_7304,
    EXT_SKR04_7305,
    EXT_SKR04_7306,
    EXT_SKR04_7309,
    EXT_SKR04_7310,
    EXT_SKR04_7313,
    EXT_SKR04_7316,
    EXT_SKR04_7317,
    EXT_SKR04_7319,
    EXT_SKR04_7320,
    EXT_SKR04_7323,
    EXT_SKR04_7324,
    EXT_SKR04_7325,
    EXT_SKR04_7326,
    EXT_SKR04_7327,
    EXT_SKR04_7328,
    EXT_SKR04_7329,
    EXT_SKR04_7330,
    EXT_SKR04_7339,
    EXT_SKR04_7340,
    EXT_SKR04_7349,
    EXT_SKR04_7350,
    EXT_SKR04_7351,
    EXT_SKR04_7355,
    EXT_SKR04_7360,
    EXT_SKR04_7361,
    EXT_SKR04_7362,
    EXT_SKR04_7363,
    EXT_SKR04_7364,
    EXT_SKR04_7365,
    EXT_SKR04_7366,
    EXT_SKR04_7390,
    EXT_SKR04_7392,
    EXT_SKR04_7394,
    EXT_SKR04_7399,
    EXT_SKR04_7400,
    EXT_SKR04_7401,
    EXT_SKR04_7450,
    EXT_SKR04_7451,
    EXT_SKR04_7452,
    EXT_SKR04_7453,
    EXT_SKR04_7454,
    EXT_SKR04_7460,
    EXT_SKR04_7461,
    EXT_SKR04_7462,
    EXT_SKR04_7463,
    EXT_SKR04_7464,
    EXT_SKR04_7500,
    EXT_SKR04_7501,
    EXT_SKR04_7550,
    EXT_SKR04_7551,
    EXT_SKR04_7552,
    EXT_SKR04_7553,
    EXT_SKR04_7554,
    EXT_SKR04_7560,
    EXT_SKR04_7561,
    EXT_SKR04_7562,
    EXT_SKR04_7563,
    EXT_SKR04_7600,
    EXT_SKR04_7603,
    EXT_SKR04_7604,
    EXT_SKR04_7607,
    EXT_SKR04_7608,
    EXT_SKR04_7609,
    EXT_SKR04_7610,
    EXT_SKR04_7630,
    EXT_SKR04_7633,
    EXT_SKR04_7639,
    EXT_SKR04_7640,
    EXT_SKR04_7641,
    EXT_SKR04_7642,
    EXT_SKR04_7643,
    EXT_SKR04_7644,
    EXT_SKR04_7646,
    EXT_SKR04_7648,
    EXT_SKR04_7649,
    EXT_SKR04_7650,
    EXT_SKR04_7675,
    EXT_SKR04_7678,
    EXT_SKR04_7680,
    EXT_SKR04_7685,
    EXT_SKR04_7690,
    EXT_SKR04_7692,
    EXT_SKR04_7694,
    EXT_SKR04_7705,
    EXT_SKR04_7725,
    EXT_SKR04_7744,
    EXT_SKR04_7751,
    EXT_SKR04_7781,
    EXT_SKR04_7785,
    EXT_SKR04_9000,
    EXT_SKR04_9008,
    EXT_SKR04_9009,
    EXT_SKR04_9090,
    EXT_SKR04_9991,
    EXT_SKR04_9994,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skr03_chart_has_expected_size() {
        assert_eq!(SKR03_CHART.len(), 1274);
    }

    #[test]
    fn skr04_chart_has_expected_size() {
        assert_eq!(SKR04_CHART.len(), 1192);
    }

    #[test]
    fn skr03_chart_entries_have_codes() {
        for entry in SKR03_CHART {
            assert!(!entry.code.is_empty(), "SKR03 entry missing code");
        }
    }

    #[test]
    fn skr04_chart_entries_have_codes() {
        for entry in SKR04_CHART {
            assert!(!entry.code.is_empty(), "SKR04 entry missing code");
        }
    }
}
