//! Ontology partition-locality probe — the empirical falsifier for the
//! "16 family pointers / inherited nothingness" claim in
//! `.claude/knowledge/delta-card-addressing-integration-map.md`.
//!
//! ## What this probe measures (and what it does NOT)
//!
//! The addressing map claims a frozen ontology radix gives ~0-bit-per-row
//! cost because (a) `subClassOf` edges are overwhelmingly *local* (both
//! endpoints in the same top-level DOLCE-style basin), so a reference can be
//! a 16-bit local pointer instead of a 27-bit global QID, and (b) the
//! per-class "family frontier" (distinct parent basins reachable) is small —
//! the design pencils in a 4/12/16 split, so the question is whether ≤16
//! distinct basins per class is empirically enough.
//!
//! This probe MEASURES those two numbers — plus the modularity Q of the
//! basin partition — on **real `rdfs:subClassOf` graphs** parsed from the
//! ontology TTLs shipped in `data/ontologies/` (DOLCE-Ultralite, schema.org,
//! Odoo-core, PROV-O, QUDT-core, OWL-Time).
//!
//! ### HONEST SCOPE CAVEAT (read before quoting any number)
//!
//! These are REAL ontology `subClassOf` structures, but they are NOT the
//! full 115M-entity Wikidata `P279` graph. There is NO Wikidata dump on
//! disk. This is a *genuine but smaller* falsifier: it tests the locality
//! hypothesis on the same KIND of structure (hand-curated upper + domain
//! ontologies, exactly the "frozen ISA" the map freezes), at 10^2..10^3
//! classes rather than 10^8. A PASS here means "the locality hypothesis
//! survives on real ontology structure"; it does NOT mean "proven on
//! Wikidata". The verdict text repeats this caveat.
//!
//! ## Definitions
//!
//! - **top-basin** of a class = the root ancestor reached by walking
//!   `subClassOf` parents upward (the DOLCE-style top facet). A class with
//!   no parent is its own basin (a root). Multi-parent classes pick a
//!   deterministic representative root (smallest interned id) so the
//!   partition is well-defined; cycles are broken defensively.
//! - **locality** = fraction of `subClassOf` edges `(child -> parent)` whose
//!   two endpoints share a top-basin. (Edges to a different basin are the
//!   "non-local" references that would need a wider pointer.)
//! - **fan-out** = per class, the number of DISTINCT parent-basins among its
//!   direct `subClassOf` parents. Max + histogram answer "is ≤16 enough?".
//! - **modularity Q** = Newman modularity of the basin partition on the
//!   undirected `subClassOf` graph, computed with the popcount-AND gain idea
//!   reused from `splat_louvain_modularity.rs` (within-partition edge mass
//!   via AND of per-basin membership bitsets).
//!
//! Zero external deps (std only) — jc stays standalone. The TTL "parser" is
//! a minimal line scanner for `rdfs:subClassOf` triples ONLY; it is NOT a
//! general Turtle parser and deliberately skips blank-node restrictions
//! (`rdfs:subClassOf [ ... ]`) since those are anonymous OWL restrictions,
//! not class-to-class edges.
//!
//! Run:
//!   cargo run --manifest-path crates/jc/Cargo.toml \
//!             --example ontology_locality_probe
//!   cargo run --manifest-path crates/jc/Cargo.toml \
//!             --example ontology_locality_probe -- /path/to/ontology/dir

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

// ── TTL subClassOf line scanner (zero-dep, NOT a general Turtle parser) ─────
//
// Turtle predicate-list shape we handle:
//
//     ex:Child  a owl:Class ;                 <- establishes current subject
//               rdfs:subClassOf ex:Parent ,   <- emits (Child -> Parent)
//                               ex:Other ;     <- comma-continued object list
//               rdfs:label "..." .            <- '.' terminates the subject
//
// Rules:
//   * The current subject is the first token of a statement (the token
//     before `a` / `rdf:type`, or simply the first token on a line that is
//     not whitespace-led when no subject is active). It persists until a
//     statement-terminating `.`.
//   * `rdfs:subClassOf` / bare `subClassOf` sets the current predicate so a
//     following line beginning with `,` continues the object list.
//   * Objects that are named IRIs (prefixed `pfx:Local`, `:Local`, or
//     `<iri>`) become edges. An object that is `[` opens an anonymous OWL
//     restriction — SKIPPED (it is not a class-to-class edge).
//   * String literals and `#` comments are stripped first so that the word
//     "subClassOf" inside an `rdfs:comment "..."` is never mistaken for a
//     predicate.

/// Remove `"..."`/`'''...'''`/`"""..."""` string literals and trailing `#`
/// comments from a single physical line, so the tokenizer never sees TTL
/// content text. Multi-line triple-quoted strings are handled by the caller
/// via the `in_long_string` flag.
fn strip_strings_and_comments(line: &str, in_long_string: &mut bool) -> String {
    // Char-based scan (UTF-8 safe: ontology comments contain multi-byte chars
    // like the zero-width space U+200B in the QUDT license text, so byte
    // slicing would panic on a char boundary). All delimiters we look for
    // ('"', '\'', '#', '\\') are single-byte ASCII.
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let n = chars.len();
    let is_triple = |c: &[char], i: usize, q: char| {
        i + 3 <= c.len() && c[i] == q && c[i + 1] == q && c[i + 2] == q
    };
    let mut i = 0;
    while i < n {
        if *in_long_string {
            // Look for the closing """ or ''' (treated the same).
            if is_triple(&chars, i, '"') || is_triple(&chars, i, '\'') {
                *in_long_string = false;
                i += 3;
            } else {
                i += 1;
            }
            continue;
        }
        // Opening of a long ("""/''') string?
        if is_triple(&chars, i, '"') || is_triple(&chars, i, '\'') {
            let q = chars[i];
            i += 3;
            // Does it also close on this same line?
            let mut closed = false;
            while i < n {
                if is_triple(&chars, i, q) {
                    i += 3;
                    closed = true;
                    break;
                }
                i += 1;
            }
            if !closed {
                *in_long_string = true;
            }
            out.push(' ');
            continue;
        }
        let c = chars[i];
        if c == '"' || c == '\'' {
            // Single-line quoted literal: skip to matching quote, honoring \".
            let quote = c;
            i += 1;
            while i < n {
                let d = chars[i];
                if d == '\\' {
                    i += 2;
                    continue;
                }
                if d == quote {
                    i += 1;
                    break;
                }
                i += 1;
            }
            out.push(' ');
            continue;
        }
        if c == '#' {
            // Rest of line is a comment.
            break;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// True iff `tok` is a named-IRI object we accept as a subClassOf target:
/// a prefixed name (`pfx:Local` or `:Local`) or an angle-bracket IRI
/// (`<...>`). Rejects blank nodes (`[`, `_:`), list punctuation, and the
/// `owl:Thing`-style roots are accepted (they ARE named classes / valid
/// basins). We DO reject `rdf:type`-ish predicates by construction because
/// this is only ever called on object position.
fn is_named_iri(tok: &str) -> bool {
    if tok.is_empty() {
        return false;
    }
    if tok.starts_with('<') {
        return tok.len() > 2; // <x>
    }
    if tok.starts_with("_:") {
        return false; // explicit blank node label
    }
    if tok.starts_with('[') || tok.starts_with(']') {
        return false;
    }
    // prefixed name: must contain a ':' and start with an identifier or ':'
    let first = tok.chars().next().unwrap();
    (first == ':' || first.is_alphabetic()) && tok.contains(':')
}

/// Normalize an object/subject token into a stable class key. Strips a
/// trailing `;` `,` `.` punctuation and surrounding `<>`; leaves prefixed
/// names as-is. Returns `None` for things that are not class references.
fn normalize_iri(tok: &str) -> Option<String> {
    let t = tok.trim_matches(|c| c == ';' || c == ',' || c == '.');
    if t.is_empty() {
        return None;
    }
    if t.starts_with('<') && t.ends_with('>') && t.len() > 2 {
        return Some(t.to_string());
    }
    if is_named_iri(t) {
        return Some(t.to_string());
    }
    None
}

/// Parse all `rdfs:subClassOf` / bare-`subClassOf` class-to-class edges from
/// a TTL document. Returns a vec of `(child, parent)` IRI-key pairs.
///
/// Self-loops (`X subClassOf X`) and edges into blank-node restrictions are
/// dropped. This is the function the `#[cfg(test)]` parser test exercises.
pub fn parse_subclass_edges(ttl: &str) -> Vec<(String, String)> {
    const SUBCLASS: &str = "subClassOf"; // matches rdfs:subClassOf AND bare subClassOf
    let mut edges: Vec<(String, String)> = Vec::new();
    let mut current_subject: Option<String> = None;
    let mut predicate_is_subclass = false;
    let mut in_long_string = false;
    // Depth of nested `[ ... ]` blank-node restrictions. While > 0 we are
    // INSIDE an anonymous OWL restriction and emit no edges; the restriction
    // spans multiple physical lines, so this persists across the line loop.
    let mut bracket_depth: i32 = 0;

    for raw_line in ttl.lines() {
        let line = strip_strings_and_comments(raw_line, &mut in_long_string);
        let leading_ws = raw_line.starts_with(char::is_whitespace);

        // Split into whitespace tokens (Turtle is whitespace-delimited at this
        // granularity; we already stripped strings/comments).
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.is_empty() {
            // A blank physical line does not by itself end a statement.
            continue;
        }

        let mut idx = 0;

        // A statement that begins flush-left (no leading whitespace) and whose
        // first token is a named IRI / blank starts a NEW subject — UNLESS the
        // line is a pure object-list continuation beginning with ',' (handled
        // below) or a directive (@prefix / @base / PREFIX / BASE).
        let first = toks[0];
        let is_directive = first.starts_with('@')
            || first.eq_ignore_ascii_case("prefix")
            || first.eq_ignore_ascii_case("base");
        if is_directive {
            // Directives don't carry subjects or edges; but a directive still
            // can be terminated by '.', which must not clobber subject state of
            // a real statement (directives are always flush-left & self
            // contained), so just skip the whole line.
            continue;
        }

        if bracket_depth == 0
            && !leading_ws
            && first != ","
            && first != ";"
            && !first.starts_with('[')
        {
            // New subject candidate (only when not inside a blank node).
            if let Some(subj) = normalize_iri(first) {
                current_subject = Some(subj);
            } else {
                current_subject = None;
            }
            predicate_is_subclass = false;
            idx = 1;
        }

        // Walk remaining tokens, tracking predicate switches and emitting
        // edges while the active predicate is subClassOf AND we are at
        // bracket depth 0 (outside any anonymous restriction).
        while idx < toks.len() {
            let tok = toks[idx];

            // Update bracket depth from any '[' / ']' characters in the token,
            // then move on if the token is pure bracket punctuation. A '['
            // opening means the CURRENT subClassOf object is an anonymous
            // restriction; we suppress emission until the matching ']' but
            // stay in subClassOf predicate mode so a following ',' continues
            // the OUTER object list.
            let opens = tok.matches('[').count() as i32;
            let closes = tok.matches(']').count() as i32;
            if opens > 0 || closes > 0 {
                bracket_depth += opens - closes;
                if bracket_depth < 0 {
                    bracket_depth = 0;
                }
                // If the token is only brackets (possibly with ',' / ';'),
                // there is nothing else to interpret on it.
                let stripped: String = tok
                    .chars()
                    .filter(|&c| c != '[' && c != ']' && c != ',' && c != ';')
                    .collect();
                if stripped.is_empty() {
                    idx += 1;
                    continue;
                }
            }

            // Anything inside a blank node is ignored entirely.
            if bracket_depth > 0 {
                idx += 1;
                continue;
            }

            // Object-list continuation: ',' keeps the current predicate.
            if tok == "," {
                idx += 1;
                continue;
            }
            // ';' ends the current predicate's object list (a new predicate
            // follows on this or a later line).
            if tok == ";" {
                predicate_is_subclass = false;
                idx += 1;
                continue;
            }
            // '.' terminates the whole statement → no active subject.
            if tok.starts_with('.') && tok.len() == 1 {
                current_subject = None;
                predicate_is_subclass = false;
                idx += 1;
                continue;
            }

            // Predicate detection: rdfs:subClassOf or bare subClassOf.
            let bare = tok.trim_end_matches([';', ',']);
            if bare == SUBCLASS || bare.ends_with(":subClassOf") || bare == "rdfs:subClassOf" {
                predicate_is_subclass = true;
                idx += 1;
                continue;
            }
            // In subClassOf object position: emit a named-IRI edge.
            if predicate_is_subclass {
                if let (Some(child), Some(parent)) =
                    (current_subject.clone(), normalize_iri(tok))
                {
                    if child != parent {
                        edges.push((child, parent));
                    }
                }
                idx += 1;
                continue;
            }

            // Not in subClassOf mode: a token like `a`, `rdf:type`,
            // `owl:disjointWith`, `rdfs:label` is a (non-subclass) predicate;
            // it just resets predicate state. We do not need its objects.
            if bare == "a" || bare.contains(':') {
                predicate_is_subclass = false;
            }
            idx += 1;
        }
    }
    edges
}

// ── class graph: intern IRIs, build parent adjacency, assign top-basins ─────

/// Interned subClassOf DAG over class IRIs.
pub struct ClassGraph {
    /// id -> IRI key (for printing).
    pub names: Vec<String>,
    /// Direct parents of each class (deduplicated, sorted).
    pub parents: Vec<Vec<usize>>,
    /// All edges as interned (child, parent) id pairs.
    pub edges: Vec<(usize, usize)>,
}

impl ClassGraph {
    /// Build from `(child, parent)` IRI-key edges. Every IRI appearing in any
    /// position becomes a node (a parent that is never a child is a root).
    pub fn from_edges(iri_edges: &[(String, String)]) -> Self {
        let mut id_of: BTreeMap<String, usize> = BTreeMap::new();
        let mut names: Vec<String> = Vec::new();
        let intern = |s: &str, names: &mut Vec<String>, id_of: &mut BTreeMap<String, usize>| {
            if let Some(&id) = id_of.get(s) {
                id
            } else {
                let id = names.len();
                names.push(s.to_string());
                id_of.insert(s.to_string(), id);
                id
            }
        };
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for (c, p) in iri_edges {
            let ci = intern(c, &mut names, &mut id_of);
            let pi = intern(p, &mut names, &mut id_of);
            edges.push((ci, pi));
        }
        let n = names.len();
        let mut parents: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(c, p) in &edges {
            parents[c].push(p);
        }
        for ps in parents.iter_mut() {
            ps.sort_unstable();
            ps.dedup();
        }
        Self { names, parents, edges }
    }

    pub fn n_classes(&self) -> usize {
        self.names.len()
    }

    /// Assign each class to its top-basin = the root ancestor reached by
    /// walking parents upward. Multi-parent: follow the parent with the
    /// SMALLEST interned id (deterministic representative). Cycles: broken by
    /// a visited-set; the entry node of a cycle becomes its own basin.
    /// Returns `basin[id] = root_id`.
    pub fn assign_basins(&self) -> Vec<usize> {
        let n = self.n_classes();
        let mut basin = vec![usize::MAX; n];
        for start in 0..n {
            if basin[start] != usize::MAX {
                continue;
            }
            // Walk up to a root, recording the path; memoize on the way back.
            let mut path: Vec<usize> = Vec::new();
            let mut visiting: BTreeSet<usize> = BTreeSet::new();
            let mut cur = start;
            let root;
            loop {
                if let Some(&memo) = basin.get(cur) {
                    if memo != usize::MAX {
                        root = memo;
                        break;
                    }
                }
                if visiting.contains(&cur) {
                    // Cycle: treat `cur` as the basin root for this SCC entry.
                    root = cur;
                    break;
                }
                visiting.insert(cur);
                path.push(cur);
                // Pick the smallest-id parent (deterministic). No parent → root.
                match self.parents[cur].iter().min() {
                    Some(&p) => cur = p,
                    None => {
                        root = cur;
                        break;
                    }
                }
            }
            for id in path {
                basin[id] = root;
            }
            if basin[start] == usize::MAX {
                basin[start] = root;
            }
        }
        basin
    }
}

// ── metric 1: locality ──────────────────────────────────────────────────────

/// Fraction of edges whose child and parent share a top-basin.
/// Returns (local_edges, total_edges, fraction). Empty graph → fraction 0.
pub fn locality(edges: &[(usize, usize)], basin: &[usize]) -> (usize, usize, f64) {
    let total = edges.len();
    if total == 0 {
        return (0, 0, 0.0);
    }
    let local = edges
        .iter()
        .filter(|&&(c, p)| basin[c] == basin[p])
        .count();
    (local, total, local as f64 / total as f64)
}

// ── metric 2: fan-out (distinct parent-basins per class) ────────────────────

/// Per-class count of DISTINCT parent-basins among its direct subClassOf
/// parents. Returns (max_fanout, histogram) where histogram[k] = #classes
/// whose fan-out == k. Classes with no parents contribute fan-out 0.
pub fn fan_out(graph: &ClassGraph, basin: &[usize]) -> (usize, BTreeMap<usize, usize>) {
    let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut max_fo = 0usize;
    for c in 0..graph.n_classes() {
        let distinct: BTreeSet<usize> = graph.parents[c].iter().map(|&p| basin[p]).collect();
        let fo = distinct.len();
        max_fo = max_fo.max(fo);
        *hist.entry(fo).or_insert(0) += 1;
    }
    (max_fo, hist)
}

// ── metric 3: modularity Q of the basin partition ──────────────────────────
//
// Newman modularity on the UNDIRECTED subClassOf graph (each subClassOf edge
// contributes one undirected link between child and parent):
//
//     Q = Σ_c [ e_c / m  -  (a_c / 2m)^2 ]
//
// where m = |E|, e_c = number of edges fully inside basin c, a_c = sum of
// degrees of nodes in basin c. We reuse the `splat_louvain_modularity.rs`
// idea — the within-community edge mass is a popcount-AND between a node's
// neighbour bitset and the basin-membership bitset — but with dynamically
// sized `Vec<u64>` planes so the probe handles ontologies with thousands of
// classes (the contract's fixed 16,384-bit `AwarenessPlane16K` is too small
// for schema.org). Self-loops are excluded by construction (the parser drops
// `X subClassOf X`).

/// A dynamically sized bitset (the standalone analogue of `AwarenessPlane16K`).
struct BitPlane(Vec<u64>);

impl BitPlane {
    fn zero(n_bits: usize) -> Self {
        BitPlane(vec![0u64; n_bits.div_ceil(64)])
    }
    #[inline]
    fn set(&mut self, idx: usize) {
        self.0[idx / 64] |= 1u64 << (idx % 64);
    }
    #[inline]
    fn and_popcount(&self, other: &BitPlane) -> u32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a & b).count_ones())
            .sum()
    }
}

/// Compute Newman modularity Q of the basin partition. Returns Q in
/// [-0.5, 1.0]. Empty graph → 0.0.
pub fn modularity_q(graph: &ClassGraph, basin: &[usize]) -> f64 {
    let n = graph.n_classes();
    let m = graph.edges.len();
    if m == 0 || n == 0 {
        return 0.0;
    }
    let two_m = 2.0 * m as f64;

    // Undirected neighbour bitset per node (both directions of each edge).
    let mut neigh: Vec<BitPlane> = (0..n).map(|_| BitPlane::zero(n)).collect();
    let mut degree = vec![0u32; n];
    for &(c, p) in &graph.edges {
        neigh[c].set(p);
        neigh[p].set(c);
        degree[c] += 1;
        degree[p] += 1;
    }

    // Group node ids by basin; build a membership bitset per basin.
    let mut members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (id, &b) in basin.iter().enumerate() {
        members.entry(b).or_default().push(id);
    }

    let mut q = 0.0;
    for ids in members.values() {
        let mut plane = BitPlane::zero(n);
        for &id in ids {
            plane.set(id);
        }
        // e_c counted twice (once per endpoint) via Σ_u popcount(neigh[u] AND plane).
        let mut e_c_times_two = 0u32;
        let mut a_c = 0.0;
        for &id in ids {
            e_c_times_two += neigh[id].and_popcount(&plane);
            a_c += degree[id] as f64;
        }
        let e_c = e_c_times_two as f64 / 2.0;
        q += (e_c / m as f64) - (a_c / two_m).powi(2);
    }
    q
}

// ── verdict ──────────────────────────────────────────────────────────────────

/// Verdict tier for the locality hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// High locality AND fan-out fits the family frontier.
    Pass,
    /// Locality decent but borderline, or fan-out near the cap.
    Marginal,
    /// Locality low — local-pointer assumption does not hold.
    Fail,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Marginal => "MARGINAL",
            Verdict::Fail => "FAIL",
        }
    }
}

/// Decide the verdict from the measured numbers.
///
/// Thresholds (stated, not hand-waved):
///   * locality ≥ 0.90 AND max_fanout ≤ 16          → PASS  (the map's claim)
///   * locality ≥ 0.75 (or max_fanout in 17..=32)   → MARGINAL
///   * otherwise                                     → FAIL
///
/// The "16" frontier is the design's pencilled cap; max_fanout > 16 means a
/// single class needs more than 16 distinct family pointers, breaking the
/// 4/12/16 split as stated (though a wider frontier byte would still work).
pub fn verdict(locality_frac: f64, max_fanout: usize) -> Verdict {
    if locality_frac >= 0.90 && max_fanout <= 16 {
        Verdict::Pass
    } else if locality_frac >= 0.75 || (max_fanout > 16 && max_fanout <= 32) {
        Verdict::Marginal
    } else {
        Verdict::Fail
    }
}

// ── load real ontology TTLs from a directory ────────────────────────────────

/// All parsed `(child, parent)` IRI edges plus the sorted list of TTL files
/// they came from.
type LoadedOntology = (Vec<(String, String)>, Vec<PathBuf>);

/// Recursively collect `*.ttl` files under `dir`, parse subClassOf edges from
/// each, and return (all_edges, sorted_file_list). I/O errors on individual
/// files are skipped with a note to stderr (the probe is best-effort over
/// whatever real ontologies are present).
fn load_dir(dir: &Path) -> std::io::Result<LoadedOntology> {
    let mut edges: Vec<(String, String)> = Vec::new();
    let mut files: Vec<PathBuf> = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let rd = match std::fs::read_dir(&d) {
            Ok(rd) => rd,
            Err(e) => {
                eprintln!("  (skip dir {}: {})", d.display(), e);
                continue;
            }
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().map(|e| e == "ttl").unwrap_or(false) {
                match std::fs::read_to_string(&path) {
                    Ok(text) => {
                        let mut e = parse_subclass_edges(&text);
                        edges.append(&mut e);
                        files.push(path);
                    }
                    Err(e) => eprintln!("  (skip {}: {})", path.display(), e),
                }
            }
        }
    }
    files.sort();
    Ok((edges, files))
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() {
    // Data dir: arg 1, else the repo-default `data/ontologies`.
    let arg = std::env::args().nth(1);
    let dir = arg
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/ontologies"));

    println!("══════════════════════════════════════════════════════════════════════");
    println!("  Ontology partition-locality probe  (probe 1: partition locality)");
    println!("══════════════════════════════════════════════════════════════════════");
    println!();
    println!("  SUBSTRATE: REAL rdfs:subClassOf graphs from {}", dir.display());
    println!("  This is a GENUINE but SMALLER falsifier (10^2..10^3 classes).");
    println!("  It is NOT the full 115M-entity Wikidata P279 graph — there is no");
    println!("  Wikidata dump on disk. A PASS means the locality hypothesis survives");
    println!("  on real ontology structure, NOT that it is proven on Wikidata.");
    println!();

    let (iri_edges, files) = match load_dir(&dir) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("FATAL: cannot read {}: {}", dir.display(), e);
            eprintln!("Pass a data dir as arg 1, e.g.:");
            eprintln!("  cargo run --manifest-path crates/jc/Cargo.toml \\");
            eprintln!("    --example ontology_locality_probe -- /abs/path/to/ontologies");
            std::process::exit(1);
        }
    };

    if iri_edges.is_empty() {
        eprintln!("No rdfs:subClassOf edges found under {}.", dir.display());
        eprintln!("(Found {} .ttl files but no class-to-class subClassOf triples.)", files.len());
        std::process::exit(1);
    }

    println!("  TTL files parsed ({}):", files.len());
    for f in &files {
        // Show the file name + how many edges it alone contributes.
        if let Ok(text) = std::fs::read_to_string(f) {
            let n = parse_subclass_edges(&text).len();
            let name = f.file_name().and_then(|s| s.to_str()).unwrap_or("?");
            println!("    {:<28} {:>5} subClassOf edges", name, n);
        }
    }
    println!();

    let graph = ClassGraph::from_edges(&iri_edges);
    let basin = graph.assign_basins();
    let n_basins: BTreeSet<usize> = basin.iter().copied().collect();

    let (local, total, loc_frac) = locality(&graph.edges, &basin);
    let (max_fo, fo_hist) = fan_out(&graph, &basin);
    let q = modularity_q(&graph, &basin);
    let v = verdict(loc_frac, max_fo);

    println!("──────────────────────────────────────────────────────────────────────");
    println!("  VERDICT TABLE  (measured on real ontology subClassOf structure)");
    println!("──────────────────────────────────────────────────────────────────────");
    println!("    classes (nodes)            : {}", graph.n_classes());
    println!("    subClassOf edges           : {}", total);
    println!("    top-basins (root facets)   : {}", n_basins.len());
    println!();
    println!("    LOCALITY                   : {}/{} = {:.4}  ({:.2}% of edges are intra-basin)",
        local, total, loc_frac, loc_frac * 100.0);
    println!("      (the map's '~90% local' claim — measured value above)");
    println!();
    println!("    FAN-OUT (distinct parent-basins per class)");
    println!("      max                      : {}  (is <=16 enough? {})",
        max_fo, if max_fo <= 16 { "YES" } else { "NO — exceeds the pencilled 16-frontier" });
    println!("      histogram (fanout -> #classes):");
    for (k, cnt) in &fo_hist {
        let bar = "#".repeat((*cnt).min(60));
        println!("        {:>3} -> {:>5}  {}", k, cnt, bar);
    }
    println!();
    println!("    MODULARITY Q (basin partition, Newman)");
    println!("      Q                        : {:.4}", q);
    println!("      (Q>0.3 = clear community structure; Q->1 = near-perfectly modular)");
    println!();

    println!("══════════════════════════════════════════════════════════════════════");
    println!("  VERDICT : {}", v.as_str());
    println!("══════════════════════════════════════════════════════════════════════");
    match v {
        Verdict::Pass => {
            println!("  High locality ({:.1}%) AND max fan-out {} <= 16.", loc_frac * 100.0, max_fo);
            println!("  ⇒ On REAL ontology structure, 16-bit LOCAL references + a <=16");
            println!("    family frontier ARE real: the vast majority of subClassOf");
            println!("    references stay inside one top-basin, and no class needs more");
            println!("    than 16 distinct parent-basin pointers.");
        }
        Verdict::Marginal => {
            println!("  Locality {:.1}% / max fan-out {}. The hypothesis is PARTIALLY", loc_frac * 100.0, max_fo);
            println!("  supported: either locality is below the 90% target, or a few");
            println!("  classes exceed the 16-frontier (a wider frontier byte would fix");
            println!("  those). The local-pointer idea is plausible but not clean here.");
        }
        Verdict::Fail => {
            println!("  Locality {:.1}% / max fan-out {}. The local-pointer assumption", loc_frac * 100.0, max_fo);
            println!("  does NOT hold on this structure: too many subClassOf edges cross");
            println!("  basins, so 16-bit local references would miss their targets.");
        }
    }
    println!();
    println!("  HONEST CAVEAT (mandatory): measured on REAL ontologies");
    println!("  (DOLCE-Ultralite, schema.org, Odoo, PROV-O, QUDT, OWL-Time), NOT on");
    println!("  Wikidata. Same KIND of frozen-ISA structure, ~10^3 classes not 10^8.");
    println!("  This FALSIFIES-or-survives the claim on real data; it does NOT prove");
    println!("  it at Wikidata scale. The Wikidata P279 run remains the open probe.");
    println!("══════════════════════════════════════════════════════════════════════");
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny inline TTL exercising every parser path: prefixed object,
    /// `:Local` object, comma-continued object list, a blank-node restriction
    /// that MUST be skipped, the word "subClassOf" buried in a comment string
    /// that MUST NOT be parsed, and a `.`-terminated statement.
    const TINY_TTL: &str = r#"
@prefix ex: <http://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Animal a owl:Class ;
    rdfs:comment "Top type. Note: every Dog subClassOf Animal informally." ;
    rdfs:label "Animal" .

ex:Dog a owl:Class ;
    rdfs:subClassOf ex:Animal ;
    rdfs:label "Dog" .

ex:Puppy a owl:Class ;
    rdfs:subClassOf ex:Dog ,
                    ex:Animal ,
                    [ rdf:type owl:Restriction ;
                      owl:onProperty ex:hasParent ;
                      owl:someValuesFrom ex:Dog ] ;
    rdfs:label "Puppy" .

:LocalThing a owl:Class ;
    rdfs:subClassOf :OtherLocal .
"#;

    #[test]
    fn parser_extracts_expected_edges_only() {
        let mut edges = parse_subclass_edges(TINY_TTL);
        edges.sort();
        let mut expected = vec![
            ("ex:Dog".to_string(), "ex:Animal".to_string()),
            ("ex:Puppy".to_string(), "ex:Dog".to_string()),
            ("ex:Puppy".to_string(), "ex:Animal".to_string()),
            (":LocalThing".to_string(), ":OtherLocal".to_string()),
        ];
        expected.sort();
        assert_eq!(edges, expected, "parser must emit exactly the 4 named-class edges");
    }

    #[test]
    fn parser_skips_comment_text_and_blank_nodes() {
        let edges = parse_subclass_edges(TINY_TTL);
        // The comment mentions "Dog subClassOf Animal" — must NOT produce an
        // edge with a literal-word subject/object.
        assert!(
            !edges.iter().any(|(c, _)| c == "Dog" || c == "every"),
            "comment text must not become an edge"
        );
        // The blank-node restriction on Puppy must NOT add an edge to a '['.
        assert!(
            !edges.iter().any(|(_, p)| p.starts_with('[')),
            "blank-node restriction must be skipped"
        );
        // Exactly 4 edges total (Puppy has 2 named parents, not 3).
        assert_eq!(edges.len(), 4);
    }

    #[test]
    fn parser_handles_angle_bracket_iri_objects() {
        let ttl = r#"
ex:A a owl:Class ;
    rdfs:subClassOf <http://example.org/Root> .
"#;
        let edges = parse_subclass_edges(ttl);
        assert_eq!(
            edges,
            vec![("ex:A".to_string(), "<http://example.org/Root>".to_string())]
        );
    }

    /// Build a planted 2-basin subClassOf forest with a KNOWN number of
    /// cross-basin edges, then assert the locality fraction is exactly the
    /// hand-computed value.
    ///
    /// Basin A: rootA <- a1, a2, a3   (3 intra-basin edges)
    /// Basin B: rootB <- b1, b2       (2 intra-basin edges)
    /// Cross  : a3 -> rootB           (1 cross-basin edge)
    /// Total 6 edges, 5 local ⇒ locality = 5/6.
    fn planted_two_basin() -> (Vec<(String, String)>, &'static str) {
        let edges = vec![
            ("a1".into(), "rootA".into()),
            ("a2".into(), "rootA".into()),
            ("a3".into(), "rootA".into()),
            ("b1".into(), "rootB".into()),
            ("b2".into(), "rootB".into()),
            ("a3".into(), "rootB".into()), // the one cross-basin edge
        ];
        (edges, "5/6")
    }

    #[test]
    fn locality_on_planted_two_basin_is_five_sixths() {
        let (iri_edges, _) = planted_two_basin();
        let graph = ClassGraph::from_edges(&iri_edges);
        let basin = graph.assign_basins();

        // a3 has parents {rootA, rootB}; smallest interned id wins as its
        // representative basin. Interning order: a1,rootA,a2,a3,b1,rootB,b2.
        // rootA interns before rootB, so a3's basin = rootA.
        let id = |s: &str| graph.names.iter().position(|n| n == s).unwrap();
        assert_eq!(basin[id("a3")], basin[id("rootA")], "a3 should land in basin A");

        let (local, total, frac) = locality(&graph.edges, &basin);
        assert_eq!(total, 6);
        assert_eq!(local, 5, "exactly one edge (a3->rootB) crosses basins");
        assert!((frac - 5.0 / 6.0).abs() < 1e-12, "locality must be exactly 5/6");
    }

    #[test]
    fn two_clean_basins_give_perfect_locality_and_high_q() {
        // No cross edges: two disjoint stars ⇒ locality = 1.0, Q should be
        // clearly positive (two well-separated communities).
        let iri_edges: Vec<(String, String)> = vec![
            ("a1".into(), "rootA".into()),
            ("a2".into(), "rootA".into()),
            ("b1".into(), "rootB".into()),
            ("b2".into(), "rootB".into()),
        ];
        let graph = ClassGraph::from_edges(&iri_edges);
        let basin = graph.assign_basins();
        let (_, _, frac) = locality(&graph.edges, &basin);
        assert!((frac - 1.0).abs() < 1e-12, "fully disjoint basins ⇒ locality 1.0");

        let q = modularity_q(&graph, &basin);
        assert!(q > 0.3, "two clean communities should give Q > 0.3, got {q}");
    }

    #[test]
    fn fan_out_counts_distinct_parent_basins() {
        // a3 has two parents in two different basins ⇒ fan-out 2 for a3.
        let (iri_edges, _) = planted_two_basin();
        let graph = ClassGraph::from_edges(&iri_edges);
        let basin = graph.assign_basins();
        let (max_fo, hist) = fan_out(&graph, &basin);
        // a3's two parents rootA, rootB are in basins {rootA, rootB} → 2 distinct.
        assert_eq!(max_fo, 2, "a3 reaches 2 distinct parent-basins");
        // Most classes have fan-out 0 (roots) or 1 (single parent).
        assert!(hist.contains_key(&0));
        assert!(hist.contains_key(&1));
        assert_eq!(hist.get(&2), Some(&1), "exactly one class has fan-out 2");
    }

    #[test]
    fn verdict_thresholds() {
        assert_eq!(verdict(0.95, 8), Verdict::Pass);
        assert_eq!(verdict(0.95, 17), Verdict::Marginal); // fan-out over 16
        assert_eq!(verdict(0.80, 4), Verdict::Marginal); // locality below 90%
        assert_eq!(verdict(0.50, 4), Verdict::Fail);
        assert_eq!(verdict(0.0, 0), Verdict::Fail);
    }

    #[test]
    fn cycle_is_broken_defensively() {
        // A 2-cycle must not infinite-loop; both nodes get a basin.
        let iri_edges: Vec<(String, String)> =
            vec![("x".into(), "y".into()), ("y".into(), "x".into())];
        let graph = ClassGraph::from_edges(&iri_edges);
        let basin = graph.assign_basins();
        assert_eq!(basin.len(), 2);
        assert!(basin.iter().all(|&b| b != usize::MAX), "every node assigned");
    }
}
