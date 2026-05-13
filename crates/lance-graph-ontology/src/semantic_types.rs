//! `semantic_types.toml` loader.
//!
//! Maps OGIT URIs (or attribute paths) to `lance_graph_contract::SemanticType`
//! enum values. The TOML file is the only declarative config in this crate;
//! customer-facing ontology data goes through TTL.
//!
//! Embedded at compile time via `include_str!`. Consumers can override
//! mappings by passing a custom TOML string to [`SemanticTypeMap::from_toml`].
//!
//! Only the variants that already exist in `lance-graph-contract::property`
//! are recognised. Adding a new variant is a contract change and must be
//! tracked separately in `LATEST_STATE.md`.

use crate::error::{Error, Result};
use lance_graph_contract::property::{DatePrecision, GeoFormat, SemanticType};
use std::collections::HashMap;
use std::sync::OnceLock;

const DEFAULT_TOML: &str = include_str!("semantic_types.toml");

/// Lookup table from attribute URI to SemanticType.
#[derive(Clone, Debug)]
pub struct SemanticTypeMap {
    by_uri: HashMap<String, SemanticType>,
    default: SemanticType,
}

impl SemanticTypeMap {
    pub fn from_toml(toml_str: &str) -> Result<Self> {
        let value: toml::Value = toml_str
            .parse()
            .map_err(|e| Error::TomlDecode(format!("{e}")))?;

        let mut by_uri = HashMap::new();
        if let Some(mappings) = value.get("mappings").and_then(|v| v.as_table()) {
            for (key, val) in mappings {
                let s = val.as_str().ok_or_else(|| {
                    Error::TomlDecode(format!(
                        "mappings.{key}: expected string SemanticType name, got {val:?}"
                    ))
                })?;
                let st = parse_semantic_type(s).ok_or_else(|| {
                    Error::TomlDecode(format!(
                        "mappings.{key}: `{s}` is not a recognised SemanticType variant"
                    ))
                })?;
                by_uri.insert(key.clone(), st);
            }
        }

        let default = value
            .get("default")
            .and_then(|v| v.get("unmapped"))
            .and_then(|v| v.as_str())
            .and_then(parse_semantic_type)
            .unwrap_or(SemanticType::PlainText);

        Ok(Self { by_uri, default })
    }

    pub fn defaults() -> &'static Self {
        static MAP: OnceLock<SemanticTypeMap> = OnceLock::new();
        MAP.get_or_init(|| {
            SemanticTypeMap::from_toml(DEFAULT_TOML)
                .expect("bundled semantic_types.toml must parse")
        })
    }

    pub fn lookup(&self, attr_uri: &str) -> SemanticType {
        self.by_uri
            .get(attr_uri)
            .cloned()
            .unwrap_or_else(|| self.default.clone())
    }

    pub fn default_type(&self) -> &SemanticType {
        &self.default
    }

    pub fn len(&self) -> usize {
        self.by_uri.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_uri.is_empty()
    }
}

/// Parse a semantic-type name from the TOML config. The set of accepted
/// names mirrors the variants currently in
/// `lance_graph_contract::property::SemanticType`. Names with parameters
/// pick conservative defaults (Date → Day, Geo → LatLon).
fn parse_semantic_type(name: &str) -> Option<SemanticType> {
    Some(match name {
        "PlainText" => SemanticType::PlainText,
        "Iban" => SemanticType::Iban,
        "Email" => SemanticType::Email,
        "Phone" => SemanticType::Phone,
        "Address" => SemanticType::Address,
        "Url" => SemanticType::Url,
        "TaxId" => SemanticType::TaxId,
        "CustomerId" => SemanticType::CustomerId,
        "InvoiceNumber" => SemanticType::InvoiceNumber,
        "Image" => SemanticType::Image,
        "Date" => SemanticType::Date(DatePrecision::Day),
        "DateMonth" => SemanticType::Date(DatePrecision::Month),
        "DateYear" => SemanticType::Date(DatePrecision::Year),
        "DateTime" => SemanticType::Date(DatePrecision::DateTime),
        "GeoLatLon" => SemanticType::Geo(GeoFormat::LatLon),
        "GeoWgs84" => SemanticType::Geo(GeoFormat::Wgs84),
        "GeoPlusCode" => SemanticType::Geo(GeoFormat::PlusCode),
        // Currency / File variants take a `&'static str` parameter we
        // cannot construct from TOML; they require explicit Rust call
        // sites. Skip them in the TOML loader.
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_load() {
        let map = SemanticTypeMap::defaults();
        assert!(!map.is_empty());
        assert_eq!(map.default_type().clone(), SemanticType::PlainText);
    }

    #[test]
    fn lookup_returns_default_for_unmapped() {
        let map = SemanticTypeMap::defaults();
        let st = map.lookup("ogit:ZzzNonexistent");
        assert_eq!(st, SemanticType::PlainText);
    }

    #[test]
    fn from_toml_handles_overrides() {
        let toml_str = r#"
[mappings]
"ogit.Test:Foo.bar" = "Email"

[default]
unmapped = "PlainText"
"#;
        let map = SemanticTypeMap::from_toml(toml_str).unwrap();
        assert_eq!(map.lookup("ogit.Test:Foo.bar"), SemanticType::Email);
        assert_eq!(map.lookup("anything-else"), SemanticType::PlainText);
    }

    #[test]
    fn from_toml_rejects_bad_variant() {
        let toml_str = r#"
[mappings]
"ogit.Bogus:X" = "NotARealVariant"
"#;
        assert!(SemanticTypeMap::from_toml(toml_str).is_err());
    }

    #[test]
    fn parametric_variants_picked() {
        let toml_str = r#"
[mappings]
"a" = "Date"
"b" = "DateTime"
"c" = "GeoLatLon"
"#;
        let map = SemanticTypeMap::from_toml(toml_str).unwrap();
        assert_eq!(map.lookup("a"), SemanticType::Date(DatePrecision::Day));
        assert_eq!(map.lookup("b"), SemanticType::Date(DatePrecision::DateTime));
        assert_eq!(map.lookup("c"), SemanticType::Geo(GeoFormat::LatLon));
    }

    /// WorkOrder namespace mappings cover the WoA-domain attributes emitted
    /// in `OGIT/NTO/WorkOrder/entities/*.ttl`. Each canonical SemanticType
    /// (Email/Phone/Iban/TaxId/CustomerId/InvoiceNumber/Date/DateTime/Image)
    /// must round-trip through the bundled TOML.
    #[test]
    fn workorder_namespace_lookups() {
        let map = SemanticTypeMap::defaults();
        // Customer
        assert_eq!(
            map.lookup("ogit.WorkOrder:Customer.email"),
            SemanticType::Email
        );
        assert_eq!(
            map.lookup("ogit.WorkOrder:Customer.telefon"),
            SemanticType::Phone
        );
        assert_eq!(
            map.lookup("ogit.WorkOrder:Customer.iban"),
            SemanticType::Iban
        );
        assert_eq!(
            map.lookup("ogit.WorkOrder:Customer.taxId"),
            SemanticType::TaxId
        );
        assert_eq!(
            map.lookup("ogit.WorkOrder:Customer.kdnr"),
            SemanticType::CustomerId
        );
        // Order
        assert_eq!(
            map.lookup("ogit.WorkOrder:Order.orderId"),
            SemanticType::InvoiceNumber
        );
        assert_eq!(
            map.lookup("ogit.WorkOrder:Order.datum"),
            SemanticType::Date(DatePrecision::Day)
        );
        assert_eq!(
            map.lookup("ogit.WorkOrder:Order.bezahlt"),
            SemanticType::Date(DatePrecision::Day)
        );
        // LogbookEntry / User
        assert_eq!(
            map.lookup("ogit.WorkOrder:LogbookEntry.datum"),
            SemanticType::Date(DatePrecision::Day)
        );
        assert_eq!(
            map.lookup("ogit.WorkOrder:LogbookEntry.createdAt"),
            SemanticType::Date(DatePrecision::DateTime)
        );
        assert_eq!(
            map.lookup("ogit.WorkOrder:User.email"),
            SemanticType::Email
        );
        assert_eq!(
            map.lookup("ogit.WorkOrder:User.phone"),
            SemanticType::Phone
        );
        // Picture / PasswordEntry
        assert_eq!(
            map.lookup("ogit.WorkOrder:Picture.dateiname"),
            SemanticType::Image
        );
        assert_eq!(
            map.lookup("ogit.WorkOrder:PasswordEntry.url"),
            SemanticType::Url
        );
    }

    /// WorkOrder attributes that are not given a dedicated semantic type
    /// fall through to `PlainText` (the default `unmapped`). And opaque
    /// PlainText labels in the TOML still resolve to PlainText.
    #[test]
    fn workorder_plaintext_and_default_fallback() {
        let map = SemanticTypeMap::defaults();
        // Explicit PlainText mapping (route / firma / artikelnr).
        assert_eq!(
            map.lookup("ogit.WorkOrder:LogbookEntry.route"),
            SemanticType::PlainText
        );
        assert_eq!(
            map.lookup("ogit.WorkOrder:Customer.firma"),
            SemanticType::PlainText
        );
        assert_eq!(
            map.lookup("ogit.WorkOrder:Article.artikelnr"),
            SemanticType::PlainText
        );
        // Unmapped WorkOrder attribute → default PlainText.
        assert_eq!(
            map.lookup("ogit.WorkOrder:Order.bogusFieldThatDoesNotExist"),
            SemanticType::PlainText
        );
    }
}
