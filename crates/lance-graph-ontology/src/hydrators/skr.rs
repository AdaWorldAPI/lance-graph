//! DATEV SKR (Standardkontenrahmen) -> `ContextBundle` hydrator.
//!
//! The Standardkontenrahmen is the German de-facto chart of accounts used
//! for HGB / EStG-compliant bookkeeping. Two parallel schemes coexist:
//!
//! - **SKR 03** (Prozessgliederungsprinzip) — process-oriented family
//!   numbering; used by most German SMEs.
//! - **SKR 04** (Abschlussgliederungsprinzip) — balance-sheet-oriented
//!   numbering aligned with HGB §266 P&L structure; used by larger firms.
//!
//! The two schemes are NOT interchangeable: account `1000` means
//! "Roh-, Hilfs- und Betriebsstoffe" in SKR 04 but "Kasse" in SKR 03.
//! They therefore hydrate into separate G slots
//! (`OGIT::SKR03_V1` and `OGIT::SKR04_V1`).
//!
//! Pattern D, minimal-name-extraction shape:
//!
//! Each row in the source CSV (`data/ontologies/skr-datev/skr0{3,4}.csv`)
//! contributes ONE IRI: `urn:datev:skr0{3,4}:account/{number}`. Family
//! classifications (Anlagevermögen, Erlöse, etc.) are NOT projected as
//! separate IRIs in this minimal hydrator — they live in the CSV's
//! `family` column and can be lifted to a SKOS Collection axiom in a
//! follow-up PR.
//!
//! CSV format:
//!
//! ```csv
//! account_number,account_name,family,source
//! 0050,"Ausstehende Einlagen auf das ...",Anlagevermögen,"DATEV SKR 04 ..."
//! ```
//!
//! Quoted fields (containing commas / quotes) are supported; the CSV is
//! parsed by a small streaming state machine to avoid pulling a new dep.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use super::owl::{ContextBundle, EntityId, HydrateErr, OntologySlot};
use crate::registry::OntologyRegistry;

/// Hydrator for a DATEV SKR CSV. One instance hydrates one `ContextBundle`
/// keyed by `g`, drawing rows from a single CSV file. `iri_prefix` is the
/// URN base each row's `account_number` is appended to.
pub struct SkrHydrator {
    pub g: u32,
    pub version: u32,
    pub domain_name: String,
    pub inherits_from: Option<u32>,
    pub starting_entity_id: EntityId,
    /// e.g. `"urn:datev:skr04:account"` — the `/{account_number}` is appended.
    pub iri_prefix: String,
}

impl SkrHydrator {
    /// Read the CSV at `csv_path` and intern each row's account number as
    /// `{iri_prefix}/{account_number}`.
    pub fn hydrate(&self, csv_path: &Path, registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
        let bytes = fs::read(csv_path).map_err(|e| HydrateErr::Io {
            path: csv_path.to_path_buf(),
            source: e,
        })?;
        let text = std::str::from_utf8(&bytes).map_err(|e| HydrateErr::Parse {
            path: csv_path.to_path_buf(),
            message: format!("CSV not UTF-8: {e}"),
        })?;

        let mut iri_to_id: HashMap<String, EntityId> = HashMap::new();
        let mut next_id: EntityId = self.starting_entity_id;

        let mut header_seen = false;
        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            if !header_seen {
                header_seen = true;
                continue;
            }
            let fields = parse_csv_line(line);
            if fields.is_empty() {
                continue;
            }
            let account_number = fields[0].trim();
            if account_number.is_empty() {
                continue;
            }
            let iri = format!("{}/{}", self.iri_prefix, account_number);
            if let std::collections::hash_map::Entry::Vacant(slot) = iri_to_id.entry(iri) {
                let id = next_id;
                next_id += 1;
                slot.insert(id);
            }
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

/// Minimal CSV-line parser. Handles double-quoted fields with embedded commas
/// and the `""` escape for a literal quote inside a quoted field. Does NOT
/// handle multi-line records (those don't occur in DATEV SKR CSVs).
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                // `""` -> literal `"`.
                if chars.peek() == Some(&'"') {
                    chars.next();
                    cur.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push(c);
            }
        } else if c == '"' {
            in_quotes = true;
        } else if c == ',' {
            out.push(std::mem::take(&mut cur));
        } else {
            cur.push(c);
        }
    }
    out.push(cur);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    const TINY_CSV: &str = r#"account_number,account_name,family,source
0001,"First, account",Anlagevermögen,test
0002,Second account,Umlaufvermögen,test
0003,"He said ""hi""",Eigenkapital,test
"#;

    #[test]
    fn parse_csv_line_handles_quoted_commas_and_escapes() {
        let f = parse_csv_line(r#"0001,"First, account",Anlagevermögen,test"#);
        assert_eq!(f, vec!["0001", "First, account", "Anlagevermögen", "test"]);

        let f = parse_csv_line(r#"0003,"He said ""hi""",Eigenkapital,test"#);
        assert_eq!(f, vec!["0003", r#"He said "hi""#, "Eigenkapital", "test"]);

        let f = parse_csv_line(r#"plain,unquoted,row,end"#);
        assert_eq!(f, vec!["plain", "unquoted", "row", "end"]);
    }

    #[test]
    fn tiny_csv_hydrates_to_three_iris() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tiny.csv");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(TINY_CSV.as_bytes()).unwrap();
        drop(f);

        let reg = OntologyRegistry::new_in_memory();
        let h = SkrHydrator {
            g: 7777,
            version: 1,
            domain_name: "tiny-skr".to_string(),
            inherits_from: None,
            starting_entity_id: 100,
            iri_prefix: "urn:test:skr".to_string(),
        };
        h.hydrate(&path, &reg).expect("hydrate tiny SKR CSV");

        let bundle = reg.bundle_for(7777).expect("bundle");
        assert_eq!(bundle.entity_count(), 3);
        assert!(bundle.resolve_iri("urn:test:skr/0001").is_some());
        assert!(bundle.resolve_iri("urn:test:skr/0002").is_some());
        assert!(bundle.resolve_iri("urn:test:skr/0003").is_some());
    }
}
