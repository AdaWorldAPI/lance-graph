//! Ontology partition-locality probe — the empirical falsifier for the
//! "16 family pointers / inherited nothingness" claim
//! (`.claude/knowledge/delta-card-addressing-integration-map.md`, Probe 1).
//!
//! ## What this measures (and what it does NOT)
//!
//! The integration map claims that the world-spine's per-entity cost falls to
//! its floor because the OWL/DOLCE `rdfs:subClassOf` hierarchy is **highly
//! local**: a class's parents almost always live in the SAME top-level facet
//! ("top-basin", DOLCE-style root ancestor). If that holds, then:
//!   * a 16-bit *local* reference (within-basin) addresses almost every edge,
//!     and the rare cross-basin edge is the only one that needs a wide pointer;
//!   * a per-class "family frontier" of <= 16 distinct parent-basins is enough.
//!
//! This probe measures, on REAL ontology `subClassOf` graphs:
//!   1. **locality** — fraction of subClassOf edges whose endpoints share a
//!      top-basin (the "~90% local" number, measured not asserted);
//!   2. **fan-out** — per-class distribution of distinct parent-basins reached
//!      (is <= 16 enough? we report the max + a histogram);
//!   3. **modularity Q** — of the top-basin partition, via the same
//!      popcount-AND community-edge idea as `splat_louvain_modularity.rs`
//!      (here over an explicit adjacency, the graphs being small).
//!
//! ## HONEST SUBSTRATE CAVEAT (read before quoting any number)
//!
//! There is NO 115M-entity Wikidata dump on disk. This probe runs on the REAL
//! ontology TTLs under `data/ontologies/` (DOLCE-Ultralite, schema.org, PROV-O,
//! QUDT-core, OWL-Time, Odoo-core). These are GENUINE `rdfs:subClassOf` graphs
//! — a real, smaller falsifier — but they are upper/domain ontologies (hundreds
//! to low-thousands of classes), NOT the full Wikidata P279 graph. A result
//! here is "measured on real ontology structure", NEVER "proven on Wikidata".
//! The verdict text states this explicitly.
//!
//! ## Zero-dep
//!
//! `jc` is standalone (std only). The TTL "parser" below is a deliberately
//! minimal, zero-dependency line/statement scanner for the `rdfs:subClassOf`
//! slice of Turtle — NOT a general TTL parser. It tracks the current subject
//! across a Turtle subject-block, collects named superclass objects of
//! `rdfs:subClassOf` (handling `;` predicate- and `,` object-separators), and
//! skips anonymous blank-node restrictions (`[ ... ]`), which are not named
//! superclass edges.
//!
//! Run:
//!   cargo run --manifest-path crates/jc/Cargo.toml \
//!             --example ontology_locality_probe
//!   cargo run --manifest-path crates/jc/Cargo.toml \
//!             --example ontology_locality_probe -- /path/to/ttl/dir

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

// ════════════════════════════════════════════════════════════════════════════
//  Part 1 — zero-dep Turtle `rdfs:subClassOf` scanner
// ════════════════════════════════════════════════════════════════════════════

/// One directed `subClassOf` edge: `child` is a subclass of `parent`.
/// Both are stored as their verbatim Turtle term (prefixed name like
/// `schema:MediaObject` / `:Entity`, or an angle-bracket IRI `<http://...>`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SubClassEdge {
    child: String,
    parent: String,
}

/// Strip a trailing line comment (`# ...`) that is NOT inside an IRI.
/// Turtle comments run to end-of-line; `#` inside `<...>` is a fragment.
fn strip_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut in_iri = false;
    let mut in_str = false;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'<' if !in_str => in_iri = true,
            b'>' if !in_str => in_iri = false,
            b'"' if !in_iri => in_str = !in_str,
            b'#' if !in_iri && !in_str => return &line[..i],
            _ => {}
        }
    }
    line
}

/// Is `tok` a plausible *named* RDF term usable as a class node?
/// Accept prefixed names (`pfx:Local`, `:Local`) and angle-bracket IRIs
/// (`<http://...>`). Reject blank nodes (`[`, `_:b0`), RDF lists, literals,
/// and the bare keyword `a`.
fn is_named_term(tok: &str) -> bool {
    if tok.is_empty() {
        return false;
    }
    if tok == "a" {
        return false;
    }
    if tok.starts_with('[') || tok.starts_with(']') {
        return false;
    }
    if tok.starts_with("_:") {
        return false;
    }
    if tok.starts_with('"') || tok.starts_with('\'') {
        return false;
    }
    if tok.starts_with('<') {
        return tok.ends_with('>') && tok.len() > 2;
    }
    // prefixed name: must contain exactly one ':' separating prefix and local,
    // and the local part must be non-empty. `:Entity` (empty prefix) is valid.
    if let Some(colon) = tok.find(':') {
        let local = &tok[colon + 1..];
        return !local.is_empty() && !local.contains(':');
    }
    false
}

/// Tokenize Turtle source into a flat stream where the structural punctuation
/// `; , . [ ]` are their own single-char tokens and everything else is a
/// whitespace-delimited term. `@prefix`/`@base` directive lines are dropped
/// (they end in `.` but never carry `subClassOf`). Comments are stripped.
fn tokenize(src: &str) -> Vec<String> {
    let mut toks = Vec::new();
    for raw in src.lines() {
        let line = strip_comment(raw).trim_end();
        let trimmed = line.trim_start();
        // Skip Turtle directives: `@prefix ... .`, `@base ... .`,
        // and the SPARQL-style `PREFIX`/`BASE` forms.
        if trimmed.starts_with('@')
            || trimmed.starts_with("PREFIX ")
            || trimmed.starts_with("BASE ")
        {
            continue;
        }
        // Split the line into terms while peeling structural punctuation.
        let mut cur = String::new();
        let mut in_iri = false;
        let mut in_str = false;
        let bytes = line.as_bytes();
        let mut push_cur = |cur: &mut String, toks: &mut Vec<String>| {
            if !cur.is_empty() {
                toks.push(std::mem::take(cur));
            }
        };
        for &b in bytes {
            let c = b as char;
            if in_str {
                cur.push(c);
                if c == '"' {
                    in_str = false;
                }
                continue;
            }
            if in_iri {
                cur.push(c);
                if c == '>' {
                    in_iri = false;
                }
                continue;
            }
            match c {
                '<' => {
                    in_iri = true;
                    cur.push(c);
                }
                '"' => {
                    in_str = true;
                    cur.push(c);
                }
                c if c.is_whitespace() => push_cur(&mut cur, &mut toks),
                ';' | ',' | '.' | '[' | ']' => {
                    push_cur(&mut cur, &mut toks);
                    toks.push(c.to_string());
                }
                _ => cur.push(c),
            }
        }
        push_cur(&mut cur, &mut toks);
    }
    toks
}

/// Parse a `subClassOf` predicate name (prefixed or angle-bracket) — accepts
/// `rdfs:subClassOf`, the bare-prefix `subClassOf`, and the full IRI form.
fn is_subclassof_pred(tok: &str) -> bool {
    tok == "rdfs:subClassOf"
        || tok == "subClassOf"
        || tok == ":subClassOf"
        || tok == "<http://www.w3.org/2000/01/rdf-schema#subClassOf>"
}

/// Scan a Turtle token stream into `subClassOf` edges.
///
/// Turtle statement grammar (the slice we need): `subject (pred objlist (';'
/// pred objlist)* )? '.'`. The subject is the first term of a statement; a
/// `;` keeps the subject and starts a new predicate; a `,` continues the
/// current predicate's object list; a `.` ends the statement (clears subject).
/// Blank-node objects open with `[`; we skip to the matching `]` (tracking
/// nesting) so anonymous OWL restrictions never count as named superclasses.
fn scan_edges(tokens: &[String]) -> Vec<SubClassEdge> {
    let mut edges = Vec::new();
    let mut subject: Option<String> = None;
    let mut cur_pred: Option<String> = None;
    let mut expecting_subject = true;
    let mut i = 0;
    while i < tokens.len() {
        let tok = &tokens[i];
        match tok.as_str() {
            "." => {
                subject = None;
                cur_pred = None;
                expecting_subject = true;
            }
            ";" => {
                cur_pred = None; // next non-punct term is a new predicate
            }
            "," => { /* keep cur_pred; next term is another object */ }
            "[" => {
                // Skip a blank node entirely (to the matching ']').
                let mut depth = 1;
                i += 1;
                while i < tokens.len() && depth > 0 {
                    match tokens[i].as_str() {
                        "[" => depth += 1,
                        "]" => depth -= 1,
                        _ => {}
                    }
                    i += 1;
                }
                continue; // i already advanced past ']'
            }
            "]" => { /* stray close; ignore */ }
            _ => {
                if expecting_subject {
                    subject = Some(tok.clone());
                    expecting_subject = false;
                } else if cur_pred.is_none() {
                    // This term is a predicate.
                    cur_pred = Some(tok.clone());
                } else {
                    // This term is an object of the current predicate.
                    if let (Some(subj), Some(pred)) = (&subject, &cur_pred) {
                        if is_subclassof_pred(pred)
                            && is_named_term(tok)
                            && is_named_term(subj)
                            && tok != subj
                        {
                            edges.push(SubClassEdge {
                                child: subj.clone(),
                                parent: tok.clone(),
                            });
                        }
                    }
                }
            }
        }
        i += 1;
    }
    edges
}

/// Parse all `*.ttl` files directly under `dir` (one level; plus `odoo/`),
/// returning the union of `subClassOf` edges. Returns `(edges, files_read)`.
fn parse_ttl_dir(dir: &Path) -> std::io::Result<(Vec<SubClassEdge>, Vec<PathBuf>)> {
    let mut edges = Vec::new();
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    // One level of subdirectory descent is enough for the curated set
    // (top-level *.ttl + odoo/odoo-core.ttl). Bounded to avoid walking the
    // large fibo-*/qudt-* expansion sets the prompt did not name.
    let wanted: HashSet<&str> = [
        "dul.ttl",
        "schemaorg.ttl",
        "provo.ttl",
        "qudt-core.ttl",
        "time.ttl",
        "odoo-core.ttl",
    ]
    .into_iter()
    .collect();
    while let Some(d) = stack.pop() {
        let rd = match std::fs::read_dir(&d) {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                // descend only into `odoo` (keeps the set curated + fast)
                if p.file_name().and_then(|s| s.to_str()) == Some("odoo") {
                    stack.push(p);
                }
                continue;
            }
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if wanted.contains(name) {
                let src = std::fs::read_to_string(&p)?;
                let toks = tokenize(&src);
                edges.extend(scan_edges(&toks));
                files.push(p);
            }
        }
    }
    files.sort();
    Ok((edges, files))
}
