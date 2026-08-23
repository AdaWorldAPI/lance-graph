//! PROBE-R2IL-DEFUSE-MACROS-1 — testing the PRESCRIPTION from
//! `probe_r2il_real_episodes`'s E5 headline, not just citing it.
//!
//! # The finding under test
//!
//! `probe_r2il_real_episodes` (this crate, same directory) measured, on this same
//! real corpus, that of 380 occurrences of the top recurring opcode trigram
//! `(int_add, copy, store)` only 1 was dataflow-chained, against a 31.5% base rate
//! of adjacent-op def-use linkage (those numbers are ORCHESTRATOR-MEASURED — cited
//! here, never re-asserted as a constant by this probe). Its conclusion: "sequential
//! adjacency is not composition — the macro carrier must be the DEF-USE CHAIN, never
//! the linear opcode window."
//!
//! This probe stops using the linear window entirely and builds macros from DEF-USE
//! CHAINS instead, then measures whether the chain carrier is actually better — i.e.
//! whether the prescription holds, or whether the chain vocabulary turns out to be
//! sparse/degenerate. Every gate below is pre-registered with a direction and is
//! allowed to fail; a failure IS the finding, not a bug in this probe.
//!
//! # Definition — DEF-USE CHAIN (the whole idea)
//!
//! Within one episode, a def-use edge exists from op X to op Y when some ValueId in
//! X's `OperandOut` set appears in Y's `OperandIn` set, AND X's position precedes
//! Y's position in the episode's smelt (fact_id) order. A length-3 CHAIN is X→Y→Z
//! under two such edges (X→Y, Y→Z) — **regardless of whether X, Y, Z are adjacent in
//! fact_id order.** X<Y<Z in position is a consequence of chaining two forward edges,
//! not an extra check. The chain's signature is `(opcode(X), opcode(Y), opcode(Z))`.
//!
//! Backward uses (an op consuming a value defined by a LATER op — loop-carried
//! values across a back-edge) are excluded from edge construction: this pass-1 chain
//! is forward-only, straight-line def-use, not full program dataflow (see C6).
//!
//! # Data contract (grounded in `probe_r2il_real_episodes`'s module docs; not
//! re-derived)
//!
//! env var `R2IL_ORE_TSV` → TSV, `#`-comments, 13 tab-separated columns:
//! `binary function fact_id at concern kind opcode a b prov_inst prov_block
//! prov_op_site prov_value` (0-indexed: 0=binary, 1=function, 2=fact_id, 5=kind,
//! 6=opcode, 8=b, 11=prov_op_site). `kind` is CamelCase (`Op`/`OperandIn`/
//! `OperandOut`/`Edge`/`CallSite`). For `OperandIn`/`OperandOut`, `b == ValueId + 1`,
//! `b == 0` means no SSA value (skip). An episode = one (binary, function) pair; its
//! ops are the `Op` rows in ascending fact_id order, keyed by `prov_op_site`.
//! Orchestrator-measured (cited, not asserted by this probe): 143 episodes, 2
//! binaries, 5340 Op rows, 7416 OperandIn rows, 4237 OperandOut rows, 0 rows with
//! `b == 0` (every operand row carries a real ValueId).
//!
//! # Pre-registrations — every gate can fail
//!
//! - **C1 CHAIN EXTRACTION** — chains exist (`total > 0`, can-fire) and recur
//!   (`distinct * 10 < total` — margin against the 9^3 pigeonhole bound; a vocabulary this thin
//!   would be all-unique and degenerate as a macro carrier).
//! - **C2 NOT-THE-WINDOW** — the chain-signature set differs from the adjacent-window
//!   opcode-trigram set: symmetric difference non-empty (can-fire, the carriers are
//!   NOT the same thing) and intersection non-empty (can-stay-silent, they are not
//!   disjoint universes either).
//! - **C3 REACHES PAST ADJACENCY** — for the single most-recurring chain signature,
//!   some fraction of its occurrences have `z_pos - x_pos > 2`, i.e. skip at least one
//!   intervening op (can-fire). The "100% dataflow-chained" fact is true by
//!   construction and is explicitly NOT treated as evidence here (falsifiability
//!   rule) — only the skip-past-adjacency claim is asserted.
//! - **C4 CONCENTRATION** — pre-registered: the def-use chain vocabulary's top-10
//!   occupancy share is HIGHER than the linear-window trigram vocabulary's, on the
//!   same episodes (the claim: dataflow recurrence is a cleaner macro signal than
//!   incidental adjacency). Refuted if the inequality does not hold.
//! - **C5 SPAN** — pre-registered: the median chain span (`z_pos - x_pos`) exceeds 2
//!   (the pure-adjacency span), i.e. chains genuinely reach past neighboring ops, not
//!   merely restate them.
//! - **C6 FENCE** — prints the scope fences (recomputed corpus binary count as a live
//!   anchor, not a memorized constant) and states explicitly that this is a pass-1,
//!   single-episode, seven-opcode chain — not full program dataflow.
//!
//! # Fences
//!
//! - No import of `ruff_r2il` (separate cargo workspace) — the TSV schema is cited
//!   and re-derived from the file itself, same discipline as `probe_r2il_real_episodes`.
//! - No mint: no classid, no BPE, no vocabulary table, no learner subsystem.
//! - Not full program dataflow: no alias analysis, no memory-SSA, no interprocedural
//!   edges, no loop-carried (backward) def-use — forward, intra-episode, pass-1 only.
//! - No performance claims. Counts, rates, and one NARS-style confidence readout
//!   (reusing the shipped `revise`/`Stamp` mechanism, nothing new) only.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::process::ExitCode;

use lance_graph_planner::nars::belief::Stamp;
use lance_graph_planner::nars::truth::TruthValue;

// ================================================================================================
// TSV reading (duplicated from `probe_r2il_real_episodes` — separate example binary,
// same schema; no cross-example import exists for cargo `--example` targets)
// ================================================================================================

struct Row {
    binary: String,
    function: String,
    fact_id: u64,
    kind: String,
    opcode: String,
    b: u64,
    op_site: String,
}

fn parse(tsv: &str) -> Vec<Row> {
    let mut rows = Vec::new();
    for line in tsv.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert!(
            f.len() == 13,
            "schema drift: expected 13 columns, got {}",
            f.len()
        );
        rows.push(Row {
            binary: f[0].to_string(),
            function: f[1].to_string(),
            fact_id: f[2].parse().expect("fact_id"),
            kind: f[5].to_string(),
            opcode: f[6].to_string(),
            b: f[8].parse().expect("b"),
            op_site: f[11].to_string(),
        });
    }
    rows
}

/// One real function-episode: the smelt-ordered op sequence plus per-op-site dataflow.
struct Episode {
    ops: Vec<(String, String)>,       // (op_site, opcode), sorted by fact_id
    outs: BTreeMap<String, Vec<u64>>, // op_site -> ValueIds DEFINED
    ins: BTreeMap<String, Vec<u64>>,  // op_site -> ValueIds CONSUMED
}

fn episodes(rows: &[Row]) -> BTreeMap<(String, String), Episode> {
    let mut map: BTreeMap<(String, String), Episode> = BTreeMap::new();
    let mut op_order: BTreeMap<(String, String), Vec<(u64, String, String)>> = BTreeMap::new();
    for r in rows {
        let key = (r.binary.clone(), r.function.clone());
        let ep = map.entry(key.clone()).or_insert_with(|| Episode {
            ops: Vec::new(),
            outs: BTreeMap::new(),
            ins: BTreeMap::new(),
        });
        match r.kind.as_str() {
            "Op" => op_order.entry(key).or_default().push((
                r.fact_id,
                r.op_site.clone(),
                r.opcode.clone(),
            )),
            "OperandOut" if r.b != 0 => {
                ep.outs.entry(r.op_site.clone()).or_default().push(r.b - 1);
            }
            "OperandIn" if r.b != 0 => {
                ep.ins.entry(r.op_site.clone()).or_default().push(r.b - 1);
            }
            _ => {}
        }
    }
    for (key, mut ops) in op_order {
        ops.sort_by_key(|(fid, _, _)| *fid);
        map.get_mut(&key)
            .expect("episode exists for every op row")
            .ops = ops.into_iter().map(|(_, site, opc)| (site, opc)).collect();
    }
    map
}

// ================================================================================================
// Def-use chain extraction — the new carrier under test
// ================================================================================================

struct ChainOcc {
    sig: (String, String, String),
    x_pos: usize,
    y_pos: usize,
    z_pos: usize,
    ep_key: (String, String),
}

/// Extract every length-3 forward def-use chain, per episode, independent of
/// linear/fact_id adjacency. Backward (loop-carried) uses are excluded by
/// construction (edge requires `pos(use) > pos(def)`).
fn extract_chains(eps: &BTreeMap<(String, String), Episode>) -> Vec<ChainOcc> {
    let mut all = Vec::new();
    for (key, ep) in eps.iter() {
        let mut pos: HashMap<&str, usize> = HashMap::new();
        let mut opcode_of: HashMap<&str, &str> = HashMap::new();
        for (i, (site, opc)) in ep.ops.iter().enumerate() {
            pos.insert(site.as_str(), i);
            opcode_of.insert(site.as_str(), opc.as_str());
        }
        // META-REVIEW FIX (P1): the whole chain extraction assumes `prov_op_site` is
        // UNIQUE within an episode — `pos.insert` silently overwrites otherwise, and
        // every span/skip figure below would be computed against a wrong position map
        // while every gate still passed. Make the assumption a gate that can fire.
        assert_eq!(
            pos.len(),
            ep.ops.len(),
            "prov_op_site must be unique within an episode"
        );

        // uses_of: ValueId -> [(pos, op_site)] consuming it
        let mut uses_of: HashMap<u64, Vec<(usize, &str)>> = HashMap::new();
        for (site, vals) in &ep.ins {
            let Some(&p) = pos.get(site.as_str()) else {
                continue; // defensive: an operand row for an op not in the Op set
            };
            for v in vals {
                uses_of.entry(*v).or_default().push((p, site.as_str()));
            }
        }

        // def_of: ValueId -> (pos, op_site) — earliest position defensively, in case
        // of an anomalous multi-def value; SSA on this corpus should give one def.
        let mut def_of: HashMap<u64, (usize, &str)> = HashMap::new();
        for (site, vals) in &ep.outs {
            let Some(&p) = pos.get(site.as_str()) else {
                continue;
            };
            for v in vals {
                def_of
                    .entry(*v)
                    .and_modify(|e| {
                        if p < e.0 {
                            *e = (p, site.as_str());
                        }
                    })
                    .or_insert((p, site.as_str()));
            }
        }

        // forward_edges: X -> {Y} where a value defined at X (pos px) is consumed at
        // Y (pos py), py > px. FORWARD ONLY — this is the back-edge exclusion fence.
        let mut fwd: HashMap<&str, HashSet<&str>> = HashMap::new();
        for (v, &(px, x_site)) in &def_of {
            if let Some(users) = uses_of.get(v) {
                for &(py, y_site) in users {
                    if py > px {
                        fwd.entry(x_site).or_default().insert(y_site);
                    }
                }
            }
        }

        // Chains X -> Y -> Z: chaining two forward edges guarantees x_pos < y_pos <
        // z_pos, so no extra distinctness/ordering check is needed here.
        for (&x_site, ys) in &fwd {
            for &y_site in ys.iter() {
                let Some(zs) = fwd.get(y_site) else {
                    continue;
                };
                for &z_site in zs.iter() {
                    let x_pos = *pos.get(x_site).expect("x_site in pos");
                    let y_pos = *pos.get(y_site).expect("y_site in pos");
                    let z_pos = *pos.get(z_site).expect("z_site in pos");
                    let sig = (
                        (*opcode_of.get(x_site).expect("x_site opcode")).to_string(),
                        (*opcode_of.get(y_site).expect("y_site opcode")).to_string(),
                        (*opcode_of.get(z_site).expect("z_site opcode")).to_string(),
                    );
                    all.push(ChainOcc {
                        sig,
                        x_pos,
                        y_pos,
                        z_pos,
                        ep_key: key.clone(),
                    });
                }
            }
        }
    }
    all
}

fn chain_sig_counts(chains: &[ChainOcc]) -> BTreeMap<(String, String, String), usize> {
    let mut counts = BTreeMap::new();
    for c in chains {
        *counts.entry(c.sig.clone()).or_insert(0usize) += 1;
    }
    counts
}

/// The carrier this probe REPLACES: adjacent-window opcode trigrams, on the same
/// episodes, counted the same way `probe_r2il_real_episodes` counts them.
fn window_trigram_counts(
    eps: &BTreeMap<(String, String), Episode>,
) -> BTreeMap<(String, String, String), usize> {
    let mut counts = BTreeMap::new();
    for ep in eps.values() {
        let opcs: Vec<&str> = ep.ops.iter().map(|(_, o)| o.as_str()).collect();
        for w in opcs.windows(3) {
            let key = (w[0].to_string(), w[1].to_string(), w[2].to_string());
            *counts.entry(key).or_insert(0usize) += 1;
        }
    }
    counts
}

fn top10_share(counts: &BTreeMap<(String, String, String), usize>) -> f64 {
    let mut v: Vec<usize> = counts.values().copied().collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    let total: usize = v.iter().sum();
    if total == 0 {
        return 0.0;
    }
    v.iter().take(10).sum::<usize>() as f64 / total as f64
}

fn median_sorted(v: &[usize]) -> f64 {
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2] as f64
    } else {
        (v[n / 2 - 1] + v[n / 2]) as f64 / 2.0
    }
}

// ================================================================================================
// main
// ================================================================================================

fn main() -> ExitCode {
    let Some(path) = std::env::var_os("R2IL_ORE_TSV") else {
        eprintln!("CORPUS ABSENT — this probe measures only real data and never fabricates.");
        eprintln!("Fetch the real episode stream and re-run:");
        eprintln!(
            "  curl -sL https://github.com/AdaWorldAPI/ruff/releases/download/r2il-harvest-pass1/r2il-pass1.ore.tsv.gz \\\n    | zcat > /tmp/r2il-pass1.ore.tsv"
        );
        eprintln!(
            "  R2IL_ORE_TSV=/tmp/r2il-pass1.ore.tsv cargo run -p lance-graph-planner --example probe_r2il_defuse_macros"
        );
        return ExitCode::from(2);
    };
    let Ok(tsv) = std::fs::read_to_string(&path) else {
        eprintln!("CORPUS ABSENT — R2IL_ORE_TSV={path:?} is not readable. Never fabricated.");
        return ExitCode::from(2);
    };

    let rows = parse(&tsv);
    let eps = episodes(&rows);
    let chains = extract_chains(&eps);
    let chain_counts = chain_sig_counts(&chains);
    let top_sig = chain_counts
        .iter()
        .max_by_key(|(_, n)| *n)
        .map(|(k, _)| k.clone());
    let mut pass = 0u32;

    // -------------------------------------------------------------------------------------------
    // C1 CHAIN EXTRACTION — chains exist and recur.
    // -------------------------------------------------------------------------------------------
    {
        let total = chains.len();
        let distinct = chain_counts.len();
        assert!(
            total > 0,
            "C1 can-fire: length-3 def-use chains must exist in this corpus"
        );
        assert!(
            // META-REVIEW FIX (P1): `distinct < total` is unfalsifiable at this corpus
            // scale by pigeonhole — 9 opcodes bound the signature space at 9^3 = 729, and
            // this corpus yields far more chains than that, so the bare form passes
            // necessarily. Assert recurrence WITH MARGIN so the gate can actually fail.
            distinct * 10 < total,
            "C1 pre-registered: chain signatures recur (distinct {distinct} < total {total}) \
             — otherwise the def-use chain vocabulary is degenerate (all-unique, no macro carrier)"
        );

        // Supplementary confidence readout: how consistently does the single most
        // recurring chain signature show up per-episode, using the SHIPPED
        // revise+Stamp loop (mirrors `probe_r2il_real_episodes`'s E4 mechanism
        // exactly — nothing new minted here).
        let mut e_value = f32::NAN;
        let mut revised = 0usize;
        let mut dropped = 0usize;
        if let Some(sig) = &top_sig {
            let episodes_with_sig: HashSet<&(String, String)> = chains
                .iter()
                .filter(|c| &c.sig == sig)
                .map(|c| &c.ep_key)
                .collect();
            let mut truth = TruthValue::new(0.5, 0.05);
            let mut stamp = Stamp(0);
            for (idx, key) in eps.keys().enumerate() {
                if !episodes_with_sig.contains(key) {
                    continue;
                }
                let ev = Stamp::source(idx as u32);
                if stamp.disjoint(ev) {
                    truth = truth.revise(&TruthValue::new(1.0, 0.9));
                    stamp = stamp.union(ev);
                    revised += 1;
                } else {
                    dropped += 1; // CHOICE: mod-64 stamp collision, same mechanism as E4
                }
            }
            e_value = truth.expectation();
        }
        pass += 1;
        println!(
            "C1 PASS  {total} length-3 def-use chains, {distinct} distinct opcode signatures \
             (recurrence holds); top-signature revise+Stamp e={e_value:.3} (revised={revised} \
             choice_dropped={dropped}, mod-64 ceiling per the shipped E4 mechanism)"
        );
    }

    // -------------------------------------------------------------------------------------------
    // C2 NOT-THE-WINDOW — chain signatures vs adjacent-window trigrams, as sets.
    // -------------------------------------------------------------------------------------------
    let window_counts = window_trigram_counts(&eps);
    {
        let chain_sig_set: HashSet<_> = chain_counts.keys().cloned().collect();
        let window_sig_set: HashSet<_> = window_counts.keys().cloned().collect();
        let inter = chain_sig_set.intersection(&window_sig_set).count();
        let sym_diff = chain_sig_set.symmetric_difference(&window_sig_set).count();
        assert!(
            sym_diff > 0,
            "C2 can-fire: the chain-signature set must differ from the window-trigram set"
        );
        assert!(
            inter > 0,
            "C2 can-stay-silent: the two carriers must still share some signatures"
        );

        let mut top_chain: Vec<_> = chain_counts.iter().collect();
        top_chain.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        let mut top_window: Vec<_> = window_counts.iter().collect();
        top_window.sort_by_key(|(_, n)| std::cmp::Reverse(*n));

        pass += 1;
        println!(
            "C2 PASS  chain sigs={} window sigs={} intersection={} symmetric_diff={}",
            chain_sig_set.len(),
            window_sig_set.len(),
            inter,
            sym_diff
        );
        println!(
            "         top chain sigs:  {:?}",
            top_chain
                .iter()
                .take(3)
                .map(|(k, n)| (k.clone(), **n))
                .collect::<Vec<_>>()
        );
        println!(
            "         top window sigs: {:?}",
            top_window
                .iter()
                .take(3)
                .map(|(k, n)| (k.clone(), **n))
                .collect::<Vec<_>>()
        );
    }

    // -------------------------------------------------------------------------------------------
    // C3 REACHES PAST ADJACENCY — the top chain signature's occurrences are not all
    // linearly adjacent. (100% dataflow-chained is true by construction — NOT the
    // evidence; that would be a vacuous self-consistency check, per the workspace
    // falsifiability rule.)
    // -------------------------------------------------------------------------------------------
    {
        let sig = top_sig.clone().expect("C1 established chains non-empty");
        let occ: Vec<&ChainOcc> = chains.iter().filter(|c| c.sig == sig).collect();
        let skip = occ.iter().filter(|c| c.z_pos - c.x_pos > 2).count();
        let frac = skip as f64 / occ.len() as f64;
        assert!(
            // META-REVIEW FIX (P1): `> 0.0` is the documented weak form — one qualifying
            // occurrence out of hundreds would pass while a regression to incidental
            // adjacency still read convincingly. Pin a MAJORITY floor instead.
            frac > 0.5,
            "C3 can-fire: the top chain signature's occurrences must include some that skip \
             at least one intervening op (z_pos - x_pos > 2) — reaching past adjacency"
        );
        pass += 1;
        println!(
            "C3 PASS  top signature {:?}: {}/{} occurrences ({:.1}%) skip >=1 intervening op \
             (self-consistency note: 100% are dataflow-chained by construction — that is NOT \
             the finding; the skip-past-adjacency fraction is)",
            sig,
            skip,
            occ.len(),
            100.0 * frac
        );
    }

    // -------------------------------------------------------------------------------------------
    // C4 CONCENTRATION — pre-registered: def-use chains concentrate MORE than the
    // linear window (higher top-10 occupancy share). Refuted if false.
    // -------------------------------------------------------------------------------------------
    {
        let chain_share = top10_share(&chain_counts);
        let window_share = top10_share(&window_counts);
        assert!(
            chain_share > window_share,
            "C4 pre-registered: def-use chain signatures should concentrate MORE than \
             linear-window trigrams (top-10 share {chain_share:.3} vs {window_share:.3}) — \
             this is REFUTED if the inequality does not hold"
        );
        pass += 1;
        println!(
            "C4 PASS  top-10 occupancy share: chains {chain_share:.3} vs window {window_share:.3}"
        );
    }

    // -------------------------------------------------------------------------------------------
    // C5 SPAN — pre-registered: chains genuinely reach past pure adjacency (median
    // span > 2).
    // -------------------------------------------------------------------------------------------
    {
        let mut spans: Vec<usize> = chains.iter().map(|c| c.z_pos - c.x_pos).collect();
        spans.sort_unstable();
        let min = *spans.first().expect("C1 established chains non-empty");
        let max = *spans.last().expect("C1 established chains non-empty");
        let med = median_sorted(&spans);
        assert!(
            med > 2.0,
            "C5 pre-registered: median chain span should exceed the pure-adjacency span of 2 \
             (got {med}) — refuted if not"
        );
        pass += 1;
        println!("C5 PASS  span (z_pos - x_pos) distribution: min={min} median={med:.1} max={max}");
    }

    // -------------------------------------------------------------------------------------------
    // C6 FENCE — scope statement + one live-recomputed anchor (not a memorized number).
    // -------------------------------------------------------------------------------------------
    {
        let binaries: HashSet<&String> = eps.keys().map(|(b, _)| b).collect();
        assert_eq!(
            binaries.len(),
            2,
            "C6 anchor: the corpus this probe reads is still 2 binaries (recomputed live \
             from the same rows, not a memorized constant)"
        );
        pass += 1;
        println!(
            "C6 PASS  fences: no mint (no classid/BPE/vocabulary table/learner subsystem); no \
             ruff import (schema cited, not re-ingested); corpus = {} binaries / {} episodes \
             (measured live in this run); a def-use chain here is a PASS-1, FORWARD-ONLY, \
             SEVEN-OPCODE, SINGLE-EPISODE chain over this episode's own operands — not full \
             program dataflow (no alias analysis, no memory-SSA, no interprocedural edges, no \
             loop-carried back-edges)",
            binaries.len(),
            eps.len()
        );
    }

    println!("\n{pass}/6 gates green — def-use chain macro carrier measured on real episodes.");
    println!(
        "Prescription under test (probe_r2il_real_episodes E5): \"the macro carrier must be the \
         DEF-USE CHAIN, never the linear opcode window\" — this probe tests that directly."
    );
    ExitCode::SUCCESS
}
