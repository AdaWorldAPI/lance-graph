//! Auto-generated from l10n_de UStVA XML by `tools/odoo-blueprint-extractor`.
//! Do NOT edit by hand — re-run: `python -m odoo_blueprint_extractor data --addon l10n_de --out <dir>`
//! Plan: `.claude/plans/odoo-source-extraction-v1.md` (D-ODOO-EXT-4).
//!
//! Provenance:
//!   `/home/user/odoo/addons/l10n_de/data/account_account_tags_data.xml`
//!   `account/models/company.py:268` + `l10n_de/models/res_company.py:32`

use crate::odoo_blueprint::{OdooGobdWiring, OdooKennzahlKind, OdooUstvaKennzahl};

// ─── UStVA Kennzahlen ───────────────────────────────────────────────────

pub const KZ_81: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "81",
    label: "81. Zum Steuersatz von 19 %",
    tag_xmlid: "l10n_de.tax_report_de_81",
    kind: OdooKennzahlKind::Derived,
    regulation_iri: &["ogit:regulation/de/ustg/13", "ogit:regulation/de/ustg/18"],
};

pub const KZ_86: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "86",
    label: "86. Zum Steuersatz von 7 %",
    tag_xmlid: "l10n_de.tax_report_de_86",
    kind: OdooKennzahlKind::Derived,
    regulation_iri: &["ogit:regulation/de/ustg/13", "ogit:regulation/de/ustg/18"],
};

pub const KZ_87: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "87",
    label: "87. Zum Steuersatz von 0 %",
    tag_xmlid: "l10n_de.tax_report_de_87",
    kind: OdooKennzahlKind::Base,
    regulation_iri: &["ogit:regulation/de/ustg/13", "ogit:regulation/de/ustg/18"],
};

pub const KZ_35: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "35",
    label: "35/36. Zu anderen Steuersätzen",
    tag_xmlid: "l10n_de.tax_report_de_35",
    kind: OdooKennzahlKind::Derived,
    regulation_iri: &["ogit:regulation/de/ustg/13", "ogit:regulation/de/ustg/18"],
};

pub const KZ_77: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "77",
    label: "77. Lieferungen land- und forstwirtschaftlicher Betriebe nach § 24 UStG an Abnehmer mit USt-IdNr.",
    tag_xmlid: "l10n_de.tax_report_de_77",
    kind: OdooKennzahlKind::Base,
    regulation_iri: &["ogit:regulation/de/ustg/13", "ogit:regulation/de/ustg/18"],
};

pub const KZ_76: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "76",
    label: "76/80. Umsätze, für die eine Steuer nach § 24 UStG zu entrichten ist",
    tag_xmlid: "l10n_de.tax_report_de_76",
    kind: OdooKennzahlKind::Derived,
    regulation_iri: &["ogit:regulation/de/ustg/13", "ogit:regulation/de/ustg/18"],
};

pub const KZ_41: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "41",
    label: "41. An Abnehmer mit USt-IdNr",
    tag_xmlid: "l10n_de.tax_report_de_41",
    kind: OdooKennzahlKind::Base,
    regulation_iri: &["ogit:regulation/de/ustg/4", "ogit:regulation/de/ustg/18"],
};

pub const KZ_44: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "44",
    label: "44. Neuer Fahrzeuge an Abnehmer ohne USt-IdNr",
    tag_xmlid: "l10n_de.tax_report_de_44",
    kind: OdooKennzahlKind::Base,
    regulation_iri: &["ogit:regulation/de/ustg/4", "ogit:regulation/de/ustg/18"],
};

pub const KZ_49: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "49",
    label: "49. Neuer Fahrzeuge außerhalb eines Unternehmens",
    tag_xmlid: "l10n_de.tax_report_de_49",
    kind: OdooKennzahlKind::Base,
    regulation_iri: &["ogit:regulation/de/ustg/4", "ogit:regulation/de/ustg/18"],
};

pub const KZ_43: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "43",
    label: "43. Weitere steuerfreie Umsätze mit Vorsteuerabzug",
    tag_xmlid: "l10n_de.tax_report_de_43",
    kind: OdooKennzahlKind::Base,
    regulation_iri: &["ogit:regulation/de/ustg/4", "ogit:regulation/de/ustg/18"],
};

pub const KZ_48: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "48",
    label: "48. Steuerfreie Umsätze ohne Vorsteuerabzug",
    tag_xmlid: "l10n_de.tax_report_de_48",
    kind: OdooKennzahlKind::Base,
    regulation_iri: &["ogit:regulation/de/ustg/4", "ogit:regulation/de/ustg/18"],
};

pub const KZ_91: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "91",
    label: "91. Steuerfreie innergemeinschaftliche Erwerbe",
    tag_xmlid: "l10n_de.tax_report_de_91",
    kind: OdooKennzahlKind::Base,
    regulation_iri: &["ogit:regulation/de/ustg/1a", "ogit:regulation/de/ustg/18"],
};

pub const KZ_89: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "89",
    label: "89. Steuerpflichtige innergemeinschaftliche Erwerbe zum Steuersatz von 19 %",
    tag_xmlid: "l10n_de.tax_report_de_89",
    kind: OdooKennzahlKind::Derived,
    regulation_iri: &["ogit:regulation/de/ustg/1a", "ogit:regulation/de/ustg/18"],
};

pub const KZ_93: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "93",
    label: "93. Zum Steuersatz von 7 %",
    tag_xmlid: "l10n_de.tax_report_de_93",
    kind: OdooKennzahlKind::Derived,
    regulation_iri: &["ogit:regulation/de/ustg/1a", "ogit:regulation/de/ustg/18"],
};

pub const KZ_90: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "90",
    label: "90. Zum Steuersatz von 0 %",
    tag_xmlid: "l10n_de.tax_report_de_90",
    kind: OdooKennzahlKind::Base,
    regulation_iri: &["ogit:regulation/de/ustg/1a", "ogit:regulation/de/ustg/18"],
};

pub const KZ_95: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "95",
    label: "95/98. Zu anderen Steuersätzen",
    tag_xmlid: "l10n_de.tax_report_de_95",
    kind: OdooKennzahlKind::Derived,
    regulation_iri: &["ogit:regulation/de/ustg/1a", "ogit:regulation/de/ustg/18"],
};

pub const KZ_94: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "94",
    label: "94/96. Neuer Fahrzeuge von Lieferern ohne",
    tag_xmlid: "l10n_de.tax_report_de_94",
    kind: OdooKennzahlKind::Derived,
    regulation_iri: &["ogit:regulation/de/ustg/1a", "ogit:regulation/de/ustg/18"],
};

pub const KZ_46: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "46",
    label: "46/47. Steuerpflichtige sonstige Leistungen eines im übrigen Gemeinschaftsgebiet ansässigen Unternehmers",
    tag_xmlid: "l10n_de.tax_report_de_46",
    kind: OdooKennzahlKind::Derived,
    regulation_iri: &["ogit:regulation/de/ustg/13b", "ogit:regulation/de/ustg/18"],
};

pub const KZ_73: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "73",
    label: "73/74. Lieferungen sicherungsübereigneter Gegenstände und Umsätze, die unter das GrEStG fallen",
    tag_xmlid: "l10n_de.tax_report_de_73",
    kind: OdooKennzahlKind::Derived,
    regulation_iri: &["ogit:regulation/de/ustg/13b", "ogit:regulation/de/ustg/18"],
};

pub const KZ_84: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "84",
    label: "84/85. Andere Leistungen",
    tag_xmlid: "l10n_de.tax_report_de_84",
    kind: OdooKennzahlKind::Derived,
    regulation_iri: &["ogit:regulation/de/ustg/13b", "ogit:regulation/de/ustg/18"],
};

pub const KZ_42: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "42",
    label: "42. Dreiecksgeschäften",
    tag_xmlid: "l10n_de.tax_report_de_42",
    kind: OdooKennzahlKind::Base,
    regulation_iri: &["ogit:regulation/de/ustg/18"],
};

pub const KZ_60: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "60",
    label: "60. Übrige steuerpflichtige Umsätze, für die der Leistungsempfänger die Steuer nach § 13b Abs. 5 UStG schuldet",
    tag_xmlid: "l10n_de.tax_report_de_60",
    kind: OdooKennzahlKind::Derived,
    regulation_iri: &["ogit:regulation/de/ustg/18"],
};

pub const KZ_21: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "21",
    label: "21. Nicht steuerbare sonstige Leistungen",
    tag_xmlid: "l10n_de.tax_report_de_21",
    kind: OdooKennzahlKind::Base,
    regulation_iri: &["ogit:regulation/de/ustg/18"],
};

pub const KZ_45: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "45",
    label: "45. Übrige nicht steuerbare Umsätze",
    tag_xmlid: "l10n_de.tax_report_de_45",
    kind: OdooKennzahlKind::Base,
    regulation_iri: &["ogit:regulation/de/ustg/18"],
};

pub const KZ_66: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "66",
    label: "66. Vorsteuerbeträge aus Rechnungen von anderen Unternehmern",
    tag_xmlid: "l10n_de.tax_report_de_66",
    kind: OdooKennzahlKind::Tax,
    regulation_iri: &["ogit:regulation/de/ustg/15", "ogit:regulation/de/ustg/18"],
};

pub const KZ_61: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "61",
    label: "61. Vorsteuerbeträge aus dem innergemeinschaftlichen Erwerb von Gegenständen",
    tag_xmlid: "l10n_de.tax_report_de_61",
    kind: OdooKennzahlKind::Tax,
    regulation_iri: &["ogit:regulation/de/ustg/15", "ogit:regulation/de/ustg/18"],
};

pub const KZ_62: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "62",
    label: "62. Entstandene Einfuhrumsatzsteuer",
    tag_xmlid: "l10n_de.tax_report_de_62",
    kind: OdooKennzahlKind::Tax,
    regulation_iri: &["ogit:regulation/de/ustg/15", "ogit:regulation/de/ustg/18"],
};

pub const KZ_67: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "67",
    label: "67. Vorsteuerbeträge aus Leistungen im Sinne des § 13b UStG",
    tag_xmlid: "l10n_de.tax_report_de_67",
    kind: OdooKennzahlKind::Tax,
    regulation_iri: &["ogit:regulation/de/ustg/15", "ogit:regulation/de/ustg/18"],
};

pub const KZ_63: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "63",
    label: "63. Vorsteuerbeträge, die nach allgemeinen Durchschnittssätzen berechnet sind",
    tag_xmlid: "l10n_de.tax_report_de_63",
    kind: OdooKennzahlKind::Tax,
    regulation_iri: &["ogit:regulation/de/ustg/15", "ogit:regulation/de/ustg/18"],
};

pub const KZ_59: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "59",
    label: "59. Vorsteuerabzug für innergemeinschaftliche Lieferungen neuer Fahrzeuge außerhalb eines Unternehmens",
    tag_xmlid: "l10n_de.tax_report_de_59",
    kind: OdooKennzahlKind::Tax,
    regulation_iri: &["ogit:regulation/de/ustg/15", "ogit:regulation/de/ustg/18"],
};

pub const KZ_64: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "64",
    label: "64. Berichtigung des Vorsteuerabzugs",
    tag_xmlid: "l10n_de.tax_report_de_64",
    kind: OdooKennzahlKind::Tax,
    regulation_iri: &["ogit:regulation/de/ustg/15", "ogit:regulation/de/ustg/18"],
};

pub const KZ_65: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "65",
    label: "65. Steuer infolge Wechsels der Besteuerungsform sowie Nachsteuer auf versteuerte Anzahlungen u. ä. wegen Steuersatzänderung",
    tag_xmlid: "l10n_de.tax_report_de_65",
    kind: OdooKennzahlKind::Tax,
    regulation_iri: &["ogit:regulation/de/ustg/18"],
};

pub const KZ_69: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "69",
    label: "69. In Rechnungen unrichtig oder unberechtigt ausgewiesene Steuerbeträge",
    tag_xmlid: "l10n_de.tax_report_de_69",
    kind: OdooKennzahlKind::Tax,
    regulation_iri: &["ogit:regulation/de/ustg/18"],
};

pub const KZ_39: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "39",
    label: "39. Abzug der festgesetzten Sondervorauszahlung für Dauerfristverlängerung",
    tag_xmlid: "l10n_de.tax_report_de_39",
    kind: OdooKennzahlKind::Tax,
    regulation_iri: &["ogit:regulation/de/ustg/18"],
};

pub const KZ_83: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "83",
    label: "83. Verbleibende Umsatzsteuer-Vorauszahlung",
    tag_xmlid: "l10n_de.tax_report_de_tag_83",
    kind: OdooKennzahlKind::Tax,
    regulation_iri: &["ogit:regulation/de/ustg/18"],
};

pub const KZ_50: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "50",
    label: "50. Minderung der Bemessungsgrundlage",
    tag_xmlid: "l10n_de.tax_report_de_50",
    kind: OdooKennzahlKind::Base,
    regulation_iri: &["ogit:regulation/de/ustg/18"],
};

pub const KZ_37: OdooUstvaKennzahl = OdooUstvaKennzahl {
    kz: "37",
    label: "37. Minderung der abziehbaren Vorsteuerbeträge",
    tag_xmlid: "l10n_de.tax_report_de_37",
    kind: OdooKennzahlKind::Tax,
    regulation_iri: &["ogit:regulation/de/ustg/18"],
};

/// All UStVA Kennzahlen — canonical iteration handle.
pub static USTVA_KENNZAHLEN: &[OdooUstvaKennzahl] = &[
    KZ_81,
    KZ_86,
    KZ_87,
    KZ_35,
    KZ_77,
    KZ_76,
    KZ_41,
    KZ_44,
    KZ_49,
    KZ_43,
    KZ_48,
    KZ_91,
    KZ_89,
    KZ_93,
    KZ_90,
    KZ_95,
    KZ_94,
    KZ_46,
    KZ_73,
    KZ_84,
    KZ_42,
    KZ_60,
    KZ_21,
    KZ_45,
    KZ_66,
    KZ_61,
    KZ_62,
    KZ_67,
    KZ_63,
    KZ_59,
    KZ_64,
    KZ_65,
    KZ_69,
    KZ_39,
    KZ_83,
    KZ_50,
    KZ_37,
];

// ─── GoBD audit-trail wiring ─────────────────────────────────────────────

/// GoBD compliance wiring on `res.company` — when `country_code == 'DE'`,
/// `force_restrictive_audit_trail` is computed True, forcing the audit
/// trail per GoBD (Grundsätze zur ordnungsmäßigen Führung und
/// Aufbewahrung von Büchern, Aufzeichnungen und Unterlagen in
/// elektronischer Form sowie zum Datenzugriff).
///
/// Provenance: `account/models/company.py:268` + `l10n_de/models/res_company.py:32`.
pub static GOBD_WIRING: OdooGobdWiring = OdooGobdWiring {
    base_field: "restrictive_audit_trail",
    force_field: "force_restrictive_audit_trail",
    trigger: "country_code == 'DE'",
    regulation_iri: &["ogit:regulation/de/gobd", "ogit:regulation/de/ao/146a"],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ustva_kennzahlen_cover_canonical_boxes() {
        let kz_codes: std::collections::HashSet<&str> =
            USTVA_KENNZAHLEN.iter().map(|k| k.kz).collect();
        assert!(kz_codes.contains("81"), "Missing canonical UStVA Kennzahl Kz.81");
        assert!(kz_codes.contains("86"), "Missing canonical UStVA Kennzahl Kz.86");
        assert!(kz_codes.contains("87"), "Missing canonical UStVA Kennzahl Kz.87");
        assert!(kz_codes.contains("35"), "Missing canonical UStVA Kennzahl Kz.35");
        assert!(kz_codes.contains("41"), "Missing canonical UStVA Kennzahl Kz.41");
        assert!(kz_codes.contains("44"), "Missing canonical UStVA Kennzahl Kz.44");
        assert!(kz_codes.contains("49"), "Missing canonical UStVA Kennzahl Kz.49");
    }

    #[test]
    fn ustva_kennzahlen_non_empty() {
        assert_eq!(USTVA_KENNZAHLEN.len(), 37);
    }

    #[test]
    fn gobd_wiring_has_correct_trigger() {
        assert_eq!(GOBD_WIRING.trigger, "country_code == 'DE'");
        assert!(!GOBD_WIRING.regulation_iri.is_empty());
    }
}
