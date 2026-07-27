//! Schematron (`.sch`) → `ContextBundle` hydrator.
//!
//! Schematron is ISO/IEC 19757-3, an XML rule language used by every major
//! e-invoicing / e-business standard (EN 16931, PEPPOL, XRechnung,
//! ZUGFeRD, UBL, ISO 20022) to express business rules that can't be
//! captured by XSD. A `.sch` file declares:
//!
//! ```xml
//! <schema xmlns="http://purl.oclc.org/dsdl/schematron">
//!   <pattern id="...">
//!     <rule context="//ram:ApplicableHeaderTradeSettlement">
//!       <assert id="FX-SCH-A-000047" test="(ram:BasisAmount)" flag="error">
//!         [BR-45]-Each VAT breakdown shall have a taxable amount.
//!       </assert>
//!     </rule>
//!   </pattern>
//! </schema>
//! ```
//!
//! Pattern D, minimal-name-extraction shape: every `<assert id="...">`,
//! `<report id="...">`, and `<pattern id="...">` becomes one IRI. The text
//! body is additionally scanned for bracketed business-rule identifiers —
//! `[BR-52]`, `[BR-CO-03]`, `[PEPPOL-EN16931-R008]` — and each distinct
//! identifier is interned as a sibling IRI under the same base.
//!
//! IRI scheme (single contiguous namespace, hydrator-supplied base):
//!
//! - Schema assertion: `{base}/assert/{id}`
//! - Schema report:    `{base}/report/{id}`
//! - Pattern:          `{base}/pattern/{id}` (skipped if no `@id`)
//! - Business rule:    `{base}/rule/{rule-id}`
//!
//! Out of scope (deferred follow-up):
//!
//! - XPath context resolution into edges back to XSD types (would link
//!   `urn:schematron:...rule/BR-52` ⊑ `ram:ApplicableTradeTax`).
//! - Phase / pattern / rule structural edges.
//! - Severity (`@flag="warning|error|fatal"`) projection.
//! - Variable / `<let>` resolution.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use quick_xml::events::Event;
use quick_xml::Reader;

use super::owl::{ContextBundle, EntityId, HydrateErr, OntologySlot};
use crate::registry::OntologyRegistry;

/// Hydrator for Schematron `.sch` files. One instance hydrates one
/// `ContextBundle` keyed by `g`, drawing IRIs from one or more input
/// files. `base_iri` is the URN prefix every interned IRI is built from
/// (typically `urn:schematron:{standard-id}`).
pub struct SchematronHydrator {
    pub g: u32,
    pub version: u32,
    pub domain_name: String,
    pub inherits_from: Option<u32>,
    pub starting_entity_id: EntityId,
    pub base_iri: String,
}

impl SchematronHydrator {
    pub fn hydrate_many(
        &self,
        sch_paths: &[&Path],
        registry: &OntologyRegistry,
    ) -> Result<u32, HydrateErr> {
        let mut iri_to_id: HashMap<String, EntityId> = HashMap::new();
        let mut next_id: EntityId = self.starting_entity_id;

        for path in sch_paths {
            let bytes = fs::read(path).map_err(|e| HydrateErr::Io {
                path: path.to_path_buf(),
                source: e,
            })?;
            walk_sch(path, &bytes, &self.base_iri, &mut iri_to_id, &mut next_id)?;
        }

        let entity_count = iri_to_id.len() as u32;
        let ontology = OntologySlot {
            entity_count,
            iri_to_id,
        };
        let bundle = ContextBundle {
            g: self.g,
            version: self.version,
            domain_name: self.domain_name.clone(),
            inherits_from: self.inherits_from,
            ontology: Some(Arc::new(ontology)),
            edge_types: Vec::new(),
        };
        registry.register_bundle(bundle);
        Ok(self.g)
    }
}

fn walk_sch(
    path: &Path,
    bytes: &[u8],
    base_iri: &str,
    iri_to_id: &mut HashMap<String, EntityId>,
    next_id: &mut EntityId,
) -> Result<(), HydrateErr> {
    let mut reader = Reader::from_reader(bytes);
    reader.config_mut().trim_text(false);
    let mut buf = Vec::new();

    // Track the current assert/report scope so we know whose text body
    // we're collecting for business-rule ID extraction.
    let mut in_assert_or_report = false;
    let mut current_text_buf = String::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Err(e) => {
                return Err(HydrateErr::Parse {
                    path: path.to_path_buf(),
                    message: format!("Schematron: {e}"),
                });
            }
            Ok(Event::Eof) => break,
            Ok(Event::Start(e)) => {
                let qname = e.name();
                let local: Vec<u8> = local_name(qname.as_ref()).to_vec();
                let id = attr_value(&e, b"id");
                match local.as_slice() {
                    b"assert" => {
                        if let Some(id) = id.as_deref() {
                            let iri = format!("{base_iri}/assert/{id}");
                            intern(iri, iri_to_id, next_id);
                        }
                        in_assert_or_report = true;
                        current_text_buf.clear();
                    }
                    b"report" => {
                        if let Some(id) = id.as_deref() {
                            let iri = format!("{base_iri}/report/{id}");
                            intern(iri, iri_to_id, next_id);
                        }
                        in_assert_or_report = true;
                        current_text_buf.clear();
                    }
                    b"pattern" => {
                        if let Some(id) = id.as_deref() {
                            let iri = format!("{base_iri}/pattern/{id}");
                            intern(iri, iri_to_id, next_id);
                        }
                    }
                    _ => {}
                }
            }
            // Self-closing `<assert .../>` / `<report .../>` / `<pattern .../>`
            // emit Event::Empty WITHOUT a matching Event::End. We MUST NOT set
            // `in_assert_or_report = true` here — there is no text body to
            // collect, and leaving the flag stuck true causes downstream
            // unrelated text to be scanned as rule text and produce spurious
            // `/rule/...` IRIs (Codex P2 finding, PR #407 review).
            Ok(Event::Empty(e)) => {
                let qname = e.name();
                let local: Vec<u8> = local_name(qname.as_ref()).to_vec();
                let id = attr_value(&e, b"id");
                match local.as_slice() {
                    b"assert" => {
                        if let Some(id) = id.as_deref() {
                            let iri = format!("{base_iri}/assert/{id}");
                            intern(iri, iri_to_id, next_id);
                        }
                    }
                    b"report" => {
                        if let Some(id) = id.as_deref() {
                            let iri = format!("{base_iri}/report/{id}");
                            intern(iri, iri_to_id, next_id);
                        }
                    }
                    b"pattern" => {
                        if let Some(id) = id.as_deref() {
                            let iri = format!("{base_iri}/pattern/{id}");
                            intern(iri, iri_to_id, next_id);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(t)) if in_assert_or_report => {
                if let Ok(s) = std::str::from_utf8(t.as_ref()) {
                    current_text_buf.push_str(s);
                }
            }
            Ok(Event::End(e)) => {
                let qname = e.name();
                let local: Vec<u8> = local_name(qname.as_ref()).to_vec();
                if matches!(local.as_slice(), b"assert" | b"report") {
                    if in_assert_or_report {
                        for rule_id in extract_business_rule_ids(&current_text_buf) {
                            let iri = format!("{base_iri}/rule/{rule_id}");
                            intern(iri, iri_to_id, next_id);
                        }
                    }
                    in_assert_or_report = false;
                    current_text_buf.clear();
                }
            }
            _ => {}
        }
        buf.clear();
    }
    Ok(())
}

fn intern(iri: String, map: &mut HashMap<String, EntityId>, n: &mut EntityId) {
    if map.contains_key(&iri) {
        return;
    }
    let id = *n;
    *n += 1;
    map.insert(iri, id);
}

fn local_name(qname: &[u8]) -> &[u8] {
    match qname.iter().position(|b| *b == b':') {
        Some(i) => &qname[i + 1..],
        None => qname,
    }
}

fn attr_value(e: &quick_xml::events::BytesStart, attr_name: &[u8]) -> Option<String> {
    for attr in e.attributes().with_checks(false).flatten() {
        if local_name(attr.key.as_ref()) == attr_name {
            return std::str::from_utf8(attr.value.as_ref())
                .ok()
                .map(|s| s.to_string());
        }
    }
    None
}

/// Pull bracketed business-rule identifiers like `[BR-52]`, `[BR-CO-03]`,
/// `[PEPPOL-EN16931-R008]`, `[BR-DEC-19]` out of a Schematron message
/// body. Pattern: opening `[`, one or more uppercase ASCII chars, then
/// one or more `-{uppercase|digit}+` segments, closing `]`.
///
/// Anything more complex (mixed-case, non-ASCII, leading digits) is
/// ignored — EN16931 / PEPPOL / DE rule IDs all fit this shape and a
/// stricter pattern keeps false positives out of the IRI namespace.
fn extract_business_rule_ids(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'[' {
            i += 1;
            continue;
        }
        // Look ahead for closing `]` within reasonable range.
        let start = i + 1;
        let end = match bytes[start..].iter().position(|b| *b == b']') {
            Some(p) => start + p,
            None => break,
        };
        let candidate = &text[start..end];
        if looks_like_business_rule_id(candidate) && !out.contains(&candidate.to_string()) {
            out.push(candidate.to_string());
        }
        i = end + 1;
    }
    out
}

fn looks_like_business_rule_id(s: &str) -> bool {
    if s.len() < 3 || s.len() > 64 {
        return false;
    }
    let mut saw_dash = false;
    let mut prev_dash = false;
    for (idx, c) in s.chars().enumerate() {
        if idx == 0 {
            if !c.is_ascii_uppercase() {
                return false;
            }
            prev_dash = false;
        } else if c == '-' {
            if prev_dash {
                // Reject double dashes.
                return false;
            }
            saw_dash = true;
            prev_dash = true;
        } else if c.is_ascii_uppercase() || c.is_ascii_digit() {
            prev_dash = false;
        } else {
            return false;
        }
    }
    saw_dash && !prev_dash
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    const TINY_SCH: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<schema xmlns="http://purl.oclc.org/dsdl/schematron" queryBinding="xslt2">
  <pattern id="P-1">
    <rule context="//foo">
      <assert id="A-001" test="bar">[BR-99]-Always bar.</assert>
      <report id="R-001" test="baz">[BR-CO-42]-Never baz.</report>
    </rule>
  </pattern>
  <pattern>
    <rule context="//qux">
      <assert id="A-002" test="bar2">[BR-99]-Repeat rule should dedupe.</assert>
    </rule>
  </pattern>
</schema>
"#;

    #[test]
    fn extracts_ids_and_business_rules() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tiny.sch");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(TINY_SCH.as_bytes()).unwrap();
        drop(f);

        let reg = OntologyRegistry::new_in_memory();
        let h = SchematronHydrator {
            g: 8888,
            version: 1,
            domain_name: "tiny-sch".to_string(),
            inherits_from: None,
            starting_entity_id: 100,
            base_iri: "urn:schematron:tiny".to_string(),
        };
        h.hydrate_many(&[&path], &reg).expect("hydrate tiny sch");

        let bundle = reg.bundle_for(8888).expect("bundle");
        // 2 asserts (A-001 dedupes once) + 1 report + 1 pattern + 2 distinct
        // business-rule IDs (BR-99 dedupes, BR-CO-42) = 6 IRIs.
        let resolved = |iri: &str| bundle.resolve_iri(iri).is_some();
        assert!(resolved("urn:schematron:tiny/assert/A-001"));
        assert!(resolved("urn:schematron:tiny/assert/A-002"));
        assert!(resolved("urn:schematron:tiny/report/R-001"));
        assert!(resolved("urn:schematron:tiny/pattern/P-1"));
        assert!(resolved("urn:schematron:tiny/rule/BR-99"));
        assert!(resolved("urn:schematron:tiny/rule/BR-CO-42"));
        assert_eq!(bundle.entity_count(), 6);
    }

    #[test]
    fn business_rule_id_validator() {
        assert!(looks_like_business_rule_id("BR-52"));
        assert!(looks_like_business_rule_id("BR-CO-03"));
        assert!(looks_like_business_rule_id("BR-DE-1"));
        assert!(looks_like_business_rule_id("PEPPOL-EN16931-R008"));
        assert!(looks_like_business_rule_id("BR-DEC-19"));
        // Negatives.
        assert!(!looks_like_business_rule_id("br-52")); // lowercase
        assert!(!looks_like_business_rule_id("BR")); // no dash
        assert!(!looks_like_business_rule_id("-BR")); // leading dash
        assert!(!looks_like_business_rule_id("BR-")); // trailing dash
        assert!(!looks_like_business_rule_id("BR--52")); // double dash
        assert!(!looks_like_business_rule_id("BR 52")); // space
        assert!(!looks_like_business_rule_id("1BR-2")); // leading digit
    }

    #[test]
    fn extract_skips_non_business_rule_brackets() {
        let s = "Each VAT breakdown [shall have] this value [BR-45] and [99] is bad.";
        let ids = extract_business_rule_ids(s);
        assert_eq!(ids, vec!["BR-45".to_string()]);
    }

    const SELF_CLOSING_SCH: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<schema xmlns="http://purl.oclc.org/dsdl/schematron" queryBinding="xslt2">
  <pattern id="P-empty">
    <rule context="//foo">
      <!-- Self-closing assert: no message body, no End event. -->
      <assert id="A-EMPTY-001" test="bar"/>
      <!-- After the self-closing assert, this stray text mentioning
           [BR-99] in a non-assert/non-report context MUST NOT be
           picked up as a rule IRI. -->
      <bar>Stray [BR-99] text outside any assert</bar>
      <!-- Normal Start/End assert AFTER the empty: must work as usual. -->
      <assert id="A-NORMAL-001" test="baz">[BR-42]-Real rule.</assert>
    </rule>
  </pattern>
</schema>
"#;

    #[test]
    fn self_closing_assert_does_not_capture_later_text() {
        // Regression test for the Codex P2 finding (PR #407 review):
        // `<assert .../>` emits Event::Empty with no matching End. Before
        // the fix, `in_assert_or_report` was set true and never reset,
        // so subsequent text bodies were scanned as rule message text
        // and produced spurious /rule/... IRIs.
        let dir = tempdir().unwrap();
        let path = dir.path().join("self-closing.sch");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(SELF_CLOSING_SCH.as_bytes()).unwrap();
        drop(f);

        let reg = OntologyRegistry::new_in_memory();
        let h = SchematronHydrator {
            g: 6666,
            version: 1,
            domain_name: "self-closing".to_string(),
            inherits_from: None,
            starting_entity_id: 100,
            base_iri: "urn:schematron:test".to_string(),
        };
        h.hydrate_many(&[&path], &reg).expect("hydrate");

        let bundle = reg.bundle_for(6666).expect("bundle");

        // Assert IDs from both the empty and the normal assert resolve.
        assert!(bundle
            .resolve_iri("urn:schematron:test/assert/A-EMPTY-001")
            .is_some());
        assert!(bundle
            .resolve_iri("urn:schematron:test/assert/A-NORMAL-001")
            .is_some());
        assert!(bundle
            .resolve_iri("urn:schematron:test/pattern/P-empty")
            .is_some());

        // BR-42 from the NORMAL assert's body must resolve (positive control).
        assert!(
            bundle
                .resolve_iri("urn:schematron:test/rule/BR-42")
                .is_some(),
            "BR-42 from the normal assert's message body must resolve"
        );

        // BR-99 from text OUTSIDE any assert/report must NOT resolve.
        assert!(
            bundle
                .resolve_iri("urn:schematron:test/rule/BR-99")
                .is_none(),
            "BR-99 was in stray text outside an assert; must NOT have been \
             interned as a rule IRI (this was the P2 corruption case)"
        );
    }
}
