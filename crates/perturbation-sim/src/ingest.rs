//! Ingest real grid topology from open data — primarily the **PyPSA-Eur /
//! OpenStreetMap prebuilt high-voltage network** (Zenodo 13358976, ODbL): the
//! `buses.csv` + `lines.csv` pair, 200–750 kV, 35 European countries incl.
//! Spain, with geographic coordinates and electrical parameters.
//!
//! Zero-dep: a small quote-aware CSV reader (PyPSA geometry/WKT fields embed
//! commas inside quotes, and some exports use `;`), column resolution by
//! name-alias (robust to schema drift across dataset versions and to other
//! sources that follow the PyPSA column convention), and DC-model parameter
//! derivation:
//!
//! - **susceptance** `b = 1/x` where `x` is the line reactance column; if that
//!   column is absent, estimated from `length` and `circuits` at
//!   [`X_PER_KM`] Ω/km (the PyPSA-Eur standard-line-type approach).
//! - **limit** = the `s_nom` thermal-rating column; if absent, estimated by
//!   voltage class via [`estimate_snom_mva`].
//!
//! The lib takes file *contents* (`&str`), not paths — it stays `std`-only and
//! testable with inline fixtures; the caller does the `std::fs::read_to_string`.
//!
//! **Honesty:** OSM carries no measured reactance or as-built thermal limits,
//! so any estimated value is an engineering proxy fit for DC contingency
//! screening, not utility protection data. [`PypsaImport`] reports how many
//! lines used an estimate so the caller can disclose it.

use crate::graph::{Edge, Grid};
use std::collections::HashMap;

/// Typical overhead-line series reactance, Ω per km (PyPSA-Eur default regime).
pub const X_PER_KM: f64 = 0.33;

/// Result of importing a PyPSA-style network.
#[derive(Debug, Clone)]
pub struct PypsaImport {
    pub grid: Grid,
    /// Bus identifier (the CSV `bus_id`/`name`), indexed as the grid's buses.
    pub bus_ids: Vec<String>,
    /// Longitude per bus (the CSV `x`), for plotting the perturbation shape.
    pub lon: Vec<f64>,
    /// Latitude per bus (the CSV `y`).
    pub lat: Vec<f64>,
    /// How many lines had to estimate reactance (no usable `x` column/value).
    pub n_estimated_reactance: usize,
    /// How many lines had to estimate the thermal limit (no `s_nom`).
    pub n_estimated_limit: usize,
    /// Lines dropped because an endpoint bus was filtered out / unknown.
    pub n_dropped_lines: usize,
}

impl PypsaImport {
    /// Restrict to the **largest connected component**. Open-data country
    /// extracts are typically fragmented — cross-border ties are dropped by the
    /// country filter and OSM has coverage gaps — which leaves `λ₂ ≈ 0` before
    /// any trip (the network is already disconnected, so a single-area
    /// contingency study is meaningless on the raw extract). This returns a
    /// re-indexed import containing only the connected core. The estimate/drop
    /// counters are reset to 0 (they describe the original parse, not this view).
    pub fn largest_component(&self) -> PypsaImport {
        let n = self.grid.n;
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for e in &self.grid.edges {
            adj[e.from].push(e.to);
            adj[e.to].push(e.from);
        }
        // Label components by iterative DFS.
        let mut comp = vec![usize::MAX; n];
        let mut sizes: Vec<usize> = Vec::new();
        for start in 0..n {
            if comp[start] != usize::MAX {
                continue;
            }
            let cid = sizes.len();
            let mut size = 0usize;
            let mut stack = vec![start];
            comp[start] = cid;
            while let Some(u) = stack.pop() {
                size += 1;
                for &v in &adj[u] {
                    if comp[v] == usize::MAX {
                        comp[v] = cid;
                        stack.push(v);
                    }
                }
            }
            sizes.push(size);
        }
        let best = sizes
            .iter()
            .enumerate()
            .max_by_key(|(_, &s)| s)
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut new_index = vec![usize::MAX; n];
        let (mut bus_ids, mut lon, mut lat) = (Vec::new(), Vec::new(), Vec::new());
        for old in 0..n {
            if comp[old] == best {
                new_index[old] = bus_ids.len();
                bus_ids.push(self.bus_ids[old].clone());
                lon.push(self.lon[old]);
                lat.push(self.lat[old]);
            }
        }
        let edges = self
            .grid
            .edges
            .iter()
            .filter(|e| comp[e.from] == best)
            .map(|e| Edge::new(new_index[e.from], new_index[e.to], e.susceptance, e.limit))
            .collect();

        PypsaImport {
            grid: Grid::new(bus_ids.len(), edges),
            bus_ids,
            lon,
            lat,
            n_estimated_reactance: 0,
            n_estimated_limit: 0,
            n_dropped_lines: 0,
        }
    }
}

/// Rough thermal rating (MVA) of a single circuit by nominal voltage (kV).
/// Used only when the source lacks an `s_nom` column. Deliberately coarse —
/// disclose it as an estimate.
pub fn estimate_snom_mva(v_nom_kv: f64) -> f64 {
    match v_nom_kv as u32 {
        0..=149 => 100.0,
        150..=240 => 500.0,
        241..=320 => 900.0,
        321..=440 => 1700.0,
        441..=550 => 2200.0,
        _ => 3500.0,
    }
}

/// Import a PyPSA-Eur / OSM network from the contents of `buses.csv` and
/// `lines.csv`. If `country` is `Some("ES")`, keep only buses in that country
/// (matched on the buses `country` column) and lines with both endpoints kept.
///
/// Returns an error string only on a structurally unusable file (missing the
/// minimum columns). Lines with bad/missing data are dropped and counted, not
/// fatal.
pub fn from_pypsa_csv(
    buses_csv: &str,
    lines_csv: &str,
    country: Option<&str>,
) -> Result<PypsaImport, String> {
    // ── buses ──────────────────────────────────────────────────────────────
    let (bh, brows) = parse_table(buses_csv).ok_or("buses.csv: empty/unparseable")?;
    let b_id = col(&bh, &["bus_id", "name", "station_id", "bus"])
        .ok_or("buses.csv: no bus id column (bus_id/name)")?;
    let b_lon = col(&bh, &["x", "lon", "longitude"]);
    let b_lat = col(&bh, &["y", "lat", "latitude"]);
    let b_country = col(&bh, &["country", "country_code"]);

    let mut bus_index: HashMap<String, usize> = HashMap::new();
    let mut bus_ids: Vec<String> = Vec::new();
    let mut lon: Vec<f64> = Vec::new();
    let mut lat: Vec<f64> = Vec::new();

    for row in &brows {
        let id = field(row, b_id);
        if id.is_empty() {
            continue;
        }
        if let (Some(c), Some(want)) = (b_country, country) {
            if !field(row, c).eq_ignore_ascii_case(want) {
                continue;
            }
        }
        if bus_index.contains_key(&id) {
            continue;
        }
        bus_index.insert(id.clone(), bus_ids.len());
        bus_ids.push(id);
        lon.push(
            b_lon
                .and_then(|c| field(row, c).parse().ok())
                .unwrap_or(f64::NAN),
        );
        lat.push(
            b_lat
                .and_then(|c| field(row, c).parse().ok())
                .unwrap_or(f64::NAN),
        );
    }
    if bus_ids.is_empty() {
        return Err("no buses after filtering — check the country code".into());
    }

    // ── lines ──────────────────────────────────────────────────────────────
    let (lh, lrows) = parse_table(lines_csv).ok_or("lines.csv: empty/unparseable")?;
    let l_b0 = col(&lh, &["bus0", "bus_0", "from"]).ok_or("lines.csv: no bus0 column")?;
    let l_b1 = col(&lh, &["bus1", "bus_1", "to"]).ok_or("lines.csv: no bus1 column")?;
    let l_x = col(&lh, &["x", "reactance", "x_pu"]);
    let l_snom = col(&lh, &["s_nom", "s_nom_mva", "rating", "thermal_limit"]);
    let l_vnom = col(&lh, &["v_nom", "voltage", "vn"]);
    let l_len = col(&lh, &["length", "length_km"]);
    let l_circ = col(&lh, &["circuits", "num_parallel"]);

    let mut edges: Vec<Edge> = Vec::new();
    let mut n_estimated_reactance = 0usize;
    let mut n_estimated_limit = 0usize;
    let mut n_dropped_lines = 0usize;

    for row in &lrows {
        let (id0, id1) = (field(row, l_b0), field(row, l_b1));
        let (Some(&from), Some(&to)) = (bus_index.get(&id0), bus_index.get(&id1)) else {
            n_dropped_lines += 1;
            continue;
        };
        if from == to {
            n_dropped_lines += 1;
            continue;
        }

        let circuits = l_circ
            .and_then(|c| field(row, c).parse::<f64>().ok())
            .unwrap_or(1.0)
            .max(1.0);
        let v_nom = l_vnom
            .and_then(|c| field(row, c).parse::<f64>().ok())
            .unwrap_or(220.0);
        let length = l_len.and_then(|c| field(row, c).parse::<f64>().ok());

        // susceptance b = 1/x
        let x_val = l_x
            .and_then(|c| field(row, c).parse::<f64>().ok())
            .filter(|&x| x > 0.0);
        let susceptance = match x_val {
            Some(x) => 1.0 / x,
            None => {
                n_estimated_reactance += 1;
                let len = length.unwrap_or(50.0).max(1e-3);
                1.0 / (X_PER_KM * len / circuits)
            }
        };

        let snom = l_snom
            .and_then(|c| field(row, c).parse::<f64>().ok())
            .filter(|&s| s > 0.0);
        let limit = match snom {
            Some(s) => s,
            None => {
                n_estimated_limit += 1;
                estimate_snom_mva(v_nom) * circuits
            }
        };

        edges.push(Edge::new(from, to, susceptance, limit));
    }

    if edges.is_empty() {
        return Err("no lines connect the selected buses".into());
    }

    Ok(PypsaImport {
        grid: Grid::new(bus_ids.len(), edges),
        bus_ids,
        lon,
        lat,
        n_estimated_reactance,
        n_estimated_limit,
        n_dropped_lines,
    })
}

// ── minimal quote-aware CSV reader ──────────────────────────────────────────

/// Parse a CSV into (lowercased header names, data rows). Returns `None` if
/// there is no header line. Auto-detects `,` vs `;`.
fn parse_table(text: &str) -> Option<(Vec<String>, Vec<Vec<String>>)> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header_line = lines.next()?;
    let delim = detect_delim(header_line);
    let headers: Vec<String> = split_csv(header_line, delim)
        .into_iter()
        .map(|h| h.trim().to_ascii_lowercase())
        .collect();
    let rows: Vec<Vec<String>> = lines.map(|l| split_csv(l, delim)).collect();
    Some((headers, rows))
}

fn detect_delim(header: &str) -> char {
    if header.matches(';').count() > header.matches(',').count() {
        ';'
    } else {
        ','
    }
}

/// Split one CSV line, respecting double-quoted fields (which may contain the
/// delimiter, e.g. WKT geometry).
fn split_csv(line: &str, delim: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                if in_quotes && chars.peek() == Some(&'"') {
                    cur.push('"');
                    chars.next();
                } else {
                    in_quotes = !in_quotes;
                }
            }
            c if c == delim && !in_quotes => {
                out.push(std::mem::take(&mut cur));
            }
            c => cur.push(c),
        }
    }
    out.push(cur);
    out
}

/// First header column whose name matches one of `names` (already lowercased).
fn col(headers: &[String], names: &[&str]) -> Option<usize> {
    names
        .iter()
        .find_map(|want| headers.iter().position(|h| h == want))
}

/// Field at `idx` (trimmed), or "" if out of range / `idx` is `None`.
fn field(row: &[String], idx: usize) -> String {
    row.get(idx)
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUSES: &str = "\
bus_id,x,y,v_nom,country
A,0.0,40.0,400,ES
B,1.0,40.5,400,ES
C,2.0,41.0,400,ES
D,9.0,48.0,400,FR
";

    // Note: line A-D crosses into FR (bus D) and must be dropped under the ES
    // filter; line with an explicit reactance keeps it, others estimate.
    const LINES: &str = "\
line_id,bus0,bus1,x,s_nom,v_nom,length,circuits
L1,A,B,5.0,1000,400,30,1
L2,B,C,,,400,40,2
L3,A,D,4.0,1500,400,500,1
";

    #[test]
    fn imports_and_filters_by_country() {
        let imp = from_pypsa_csv(BUSES, LINES, Some("ES")).unwrap();
        assert_eq!(imp.grid.n, 3, "D (FR) must be filtered out");
        assert_eq!(imp.bus_ids, vec!["A", "B", "C"]);
        // L3 (A-D) dropped: D not in the ES set.
        assert_eq!(imp.grid.edges.len(), 2);
        assert_eq!(imp.n_dropped_lines, 1);
    }

    #[test]
    fn reactance_and_limit_use_columns_when_present() {
        let imp = from_pypsa_csv(BUSES, LINES, Some("ES")).unwrap();
        // L1 has x=5 -> b=0.2, s_nom=1000.
        let l1 = imp
            .grid
            .edges
            .iter()
            .find(|e| e.from == 0 && e.to == 1)
            .unwrap();
        assert!((l1.susceptance - 0.2).abs() < 1e-12);
        assert!((l1.limit - 1000.0).abs() < 1e-12);
    }

    #[test]
    fn missing_reactance_and_limit_are_estimated() {
        let imp = from_pypsa_csv(BUSES, LINES, Some("ES")).unwrap();
        assert_eq!(imp.n_estimated_reactance, 1); // L2
        assert_eq!(imp.n_estimated_limit, 1); // L2
                                              // L2: length 40, circuits 2 -> x = 0.33*40/2 = 6.6 -> b = 1/6.6.
        let l2 = imp
            .grid
            .edges
            .iter()
            .find(|e| e.from == 1 && e.to == 2)
            .unwrap();
        assert!((l2.susceptance - 1.0 / (X_PER_KM * 40.0 / 2.0)).abs() < 1e-9);
        // s_nom estimate: 400 kV -> 1700 * 2 circuits.
        assert!((l2.limit - 1700.0 * 2.0).abs() < 1e-9);
    }

    #[test]
    fn largest_component_picks_the_bigger_island() {
        // Triangle (A,B,C) + a separate edge (D,E). LCC = the triangle.
        let buses = "bus_id,x,y,country\nA,0,0,ES\nB,0,0,ES\nC,0,0,ES\nD,0,0,ES\nE,0,0,ES\n";
        let lines = "line_id,bus0,bus1,x\nl1,A,B,1\nl2,B,C,1\nl3,C,A,1\nl4,D,E,1\n";
        let imp = from_pypsa_csv(buses, lines, Some("ES")).unwrap();
        assert_eq!(imp.grid.n, 5);
        let lcc = imp.largest_component();
        assert_eq!(lcc.grid.n, 3, "largest component is the triangle");
        assert_eq!(lcc.grid.edges.len(), 3);
    }

    #[test]
    fn semicolon_delimiter_and_quoted_geometry() {
        let buses = "name;x;y;country\nN1;0;0;ES\nN2;1;1;ES\n";
        let lines = "line_id;bus0;bus1;x;geometry\nLa;N1;N2;2.5;\"LINESTRING(0 0, 1 1)\"\n";
        let imp = from_pypsa_csv(buses, lines, Some("ES")).unwrap();
        assert_eq!(imp.grid.n, 2);
        assert_eq!(imp.grid.edges.len(), 1);
        assert!((imp.grid.edges[0].susceptance - 0.4).abs() < 1e-12);
    }
}
