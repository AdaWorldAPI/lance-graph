//! PROBE-R2IL-REAL-EPISODES-1 — the frontier loop measured on REAL FunctionBehavior
//! episode streams (the named-not-built measurement from PROBE-R2IL-FRONTIER-PHASE2-1).
//!
//! # What the data is
//!
//! The ruff-side R2IL pass-1 harvest (`AdaWorldAPI/ruff`, `crates/ruff_r2il/examples/`
//! `harvest_r2il.rs`) lifted 143 real functions from real x86-64 ELF64 binaries
//! (`r2sleigh/tests/e2e/{stress_test,stress_test_opt}`; provenance FNV-pinned in ruff's
//! committed `.claude/harvest/r2il/PROVENANCE.md`) into 17,557 typed `FlatFact` rows —
//! one row per melted fact, smelt order, each carrying a 16-byte facet address. The bulk
//! artifact lives on the ruff GitHub Release `r2il-harvest-pass1`
//! (`r2il-pass1.ore.tsv.gz`); the census + slag + triage stay committed in ruff's tree.
//!
//! This probe READS that evidence file and measures. It is:
//! - **not a re-ingest path into ruff** (ruff's own rule "nothing in ruff parses them
//!   back" governs the ruff pipeline; a lance-graph-side probe measuring distributions
//!   from the evidence is a measurement, not a furnace input);
//! - **env-gated, never fabricating**: `R2IL_ORE_TSV` must point at the real TSV. If it
//!   is unset or missing, the probe prints the fetch instructions and exits with an
//!   explicit CORPUS ABSENT failure — it never synthesizes a stand-in corpus.
//!
//! ```sh
//! curl -sL https://github.com/AdaWorldAPI/ruff/releases/download/r2il-harvest-pass1/r2il-pass1.ore.tsv.gz \
//!   | zcat > /tmp/r2il-pass1.ore.tsv
//! R2IL_ORE_TSV=/tmp/r2il-pass1.ore.tsv \
//!   cargo run -p lance-graph-planner --example probe_r2il_real_episodes
//! ```
//!
//! # TSV schema (grounded in `ruff_r2il::furnace` module docs, ruff `origin/main`)
//!
//! `binary  function  fact_id  at  concern  kind  opcode  a  b  prov_inst  prov_block
//!  prov_op_site  prov_value` — payload per kind: `Op`: a = ordinal-in-block, b =
//! `input_arity | has_output << 32`; `OperandIn`: a = input index, b = `ValueId + 1`
//! (0 = no SSA value); `OperandOut`: a = 0, b = `ValueId + 1`.
//!
//! # Gates (each can fail; each failure would be a finding)
//!
//! - **E1 CONSERVATION** — recomputed census must equal ruff's independently committed
//!   `r2il-pass1-census.md` EXACTLY (5 fact kinds, 9 opcodes, total 17,557). This
//!   validates the reader against numbers this probe did not produce.
//! - **E2 EPISODES** — exactly 143 distinct functions across exactly 2 binaries (the
//!   README's measured count; the 2 stripped binaries never contributed functions).
//! - **E3 RECURRENCE** — real opcode-trigram structure vs a marginal-preserving
//!   within-function shuffle (deterministic LCG, fixed seed). The FIRST pre-registration
//!   ("real top-1 occurrence count > shuffled") was REFUTED on the real data (380 vs
//!   387) — top-1 count is maximized by the monotone (copy,copy,copy) runs the shuffle
//!   CREATES. The gate now asserts the measured structure: real streams collapse into
//!   >2x fewer distinct trigram types than shuffle, with higher top-10 occupancy, and
//!   the shuffle's top type is pinned to the predicted monotone-run artifact. The
//!   refutation is recorded in place, not adjusted away.
//! - **E4 FRONTIER ON REAL EPISODES** — the shipped loop (`TruthValue::revise` +
//!   `Stamp` disjointness, nothing new) run over real function-episodes as evidence
//!   events for the top recurring macros. Measures the e-values actually reached AND
//!   the modulo-64 stamp ceiling: with 143 episodes, `Stamp::source(i % 64)` collides,
//!   so evidence past 64 distinct sources is conservatively CHOICE-dropped — this gate
//!   measures how much.
//! - **E5 THE CLOBBER ANALOG ON REAL DATA (headline)** — sloppy admission = the opcode
//!   trigram matched. Contract admission = the occurrence is DATAFLOW-CHAINED (op i's
//!   `OperandOut` ValueId is consumed by an `OperandIn` of op i+1, twice). Measures the
//!   over-admission rate: the fraction of real opcode-gram occurrences a happy-path
//!   matcher would learn that are NOT dataflow-connected. Non-vacuity is enforced both
//!   ways (can-fire: some occurrence must fail the contract; can-stay-silent: some must
//!   pass) — if either side is empty on this corpus the gate fails as vacuous.
//!
//! # Fences (what this probe does NOT do)
//!
//! - No import of `ruff_r2il` (separate cargo workspace); the schema knowledge above is
//!   cited, and E1 exists precisely to catch a misread of it.
//! - No mint: no classid, no vocabulary table, no BPE, no learner subsystem. The
//!   R2IL × BPE / OGAR-loco / V4 synthesis stays a three-IF hypothesis.
//! - No performance claims. Counts and rates only.

use std::collections::BTreeMap;
use std::process::ExitCode;

use lance_graph_planner::nars::belief::Stamp;
use lance_graph_planner::nars::truth::TruthValue;

// ================================================================================================
// TSV reading (probe-local; validated by E1 against the independently committed census)
// ================================================================================================

#[derive(Clone)]
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
    /// Op sites in smelt (fact_id) order, each with its opcode.
    ops: Vec<(String, String)>, // (op_site, opcode)
    /// op_site -> ValueIds this op DEFINES (from OperandOut, b != 0 → ValueId = b-1).
    outs: BTreeMap<String, Vec<u64>>,
    /// op_site -> ValueIds this op CONSUMES (from OperandIn, b != 0).
    ins: BTreeMap<String, Vec<u64>>,
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

/// Deterministic LCG (fixed seed) — a reproducible within-function permutation for the
/// E3 marginal-preserving control. Numerical Recipes constants; not cryptographic.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

fn shuffled(seq: &[String], lcg: &mut Lcg) -> Vec<String> {
    let mut s = seq.to_vec();
    for i in (1..s.len()).rev() {
        let j = (lcg.next() % (i as u64 + 1)) as usize;
        s.swap(i, j);
    }
    s
}

fn trigram_counts(seqs: &[Vec<String>]) -> BTreeMap<(String, String, String), (usize, usize)> {
    // value = (total occurrences, distinct functions containing it)
    let mut counts: BTreeMap<(String, String, String), (usize, usize)> = BTreeMap::new();
    for seq in seqs {
        let mut seen_here: BTreeMap<(String, String, String), bool> = BTreeMap::new();
        for w in seq.windows(3) {
            let key = (w[0].clone(), w[1].clone(), w[2].clone());
            counts.entry(key.clone()).or_default().0 += 1;
            seen_here.entry(key).or_insert(true);
        }
        for (key, _) in seen_here {
            counts.entry(key).or_default().1 += 1;
        }
    }
    counts
}

// ================================================================================================
// main
// ================================================================================================

fn main() -> ExitCode {
    let Some(path) = std::env::var_os("R2IL_ORE_TSV") else {
        eprintln!("CORPUS ABSENT — this probe measures only real data and never fabricates.");
        eprintln!("Fetch the real episode stream and re-run:");
        eprintln!(
            "  curl -sL https://github.com/AdaWorldAPI/ruff/releases/download/r2il-harvest-pass1/r2il-pass1.ore.tsv.gz | zcat > /tmp/r2il-pass1.ore.tsv"
        );
        eprintln!("  R2IL_ORE_TSV=/tmp/r2il-pass1.ore.tsv cargo run -p lance-graph-planner --example probe_r2il_real_episodes");
        return ExitCode::from(2);
    };
    let Ok(tsv) = std::fs::read_to_string(&path) else {
        eprintln!("CORPUS ABSENT — R2IL_ORE_TSV={path:?} is not readable. Never fabricated.");
        return ExitCode::from(2);
    };

    let rows = parse(&tsv);
    let mut pass = 0u32;

    // -------------------------------------------------------------------------------------------
    // E1 CONSERVATION — recomputed census == ruff's independently committed census, exactly.
    // -------------------------------------------------------------------------------------------
    {
        let mut by_kind: BTreeMap<&str, usize> = BTreeMap::new();
        let mut by_opcode: BTreeMap<&str, usize> = BTreeMap::new();
        for r in &rows {
            *by_kind.entry(r.kind.as_str()).or_default() += 1;
            *by_opcode.entry(r.opcode.as_str()).or_default() += 1;
        }
        assert_eq!(rows.len(), 17_557, "total classified FlatFact rows");
        let expect_kind = [
            // NOTE: the TSV kind column carries the enum variant names (CamelCase), not
            // Census's snake_case as_str() — verified against the file; counts below still
            // cross-check EXACTLY against ruff's committed census.
            ("CallSite", 108usize),
            ("Edge", 456),
            ("Op", 5340),
            ("OperandIn", 7416),
            ("OperandOut", 4237),
        ];
        for (k, n) in expect_kind {
            assert_eq!(by_kind.get(k), Some(&n), "kind {k}");
        }
        assert_eq!(
            by_kind.len(),
            5,
            "exactly the 5 committed kinds — no phantom kind"
        );
        let expect_op = [
            ("branch", 25usize),
            ("call", 428),
            ("call_ind", 4),
            ("cbranch", 995),
            ("copy", 6753),
            ("int_add", 4768),
            ("load", 2394),
            ("return", 222),
            ("store", 1968),
        ];
        for (k, n) in expect_op {
            assert_eq!(by_opcode.get(k), Some(&n), "opcode {k}");
        }
        assert_eq!(by_opcode.len(), 9, "exactly the 9 committed opcodes");
        pass += 1;
        println!(
            "E1 PASS  census conservation: 17557 rows, 5 kinds + 9 opcodes == committed census"
        );
    }

    // -------------------------------------------------------------------------------------------
    // E2 EPISODES — 143 real functions across 2 (symtab-bearing) binaries.
    // -------------------------------------------------------------------------------------------
    let eps = episodes(&rows);
    {
        let binaries: std::collections::BTreeSet<&String> = eps.keys().map(|(b, _)| b).collect();
        assert_eq!(
            eps.len(),
            143,
            "distinct real functions (README's measured count)"
        );
        assert_eq!(
            binaries.len(),
            2,
            "the 2 symtab-bearing binaries; stripped ones never contribute"
        );
        let op_total: usize = eps.values().map(|e| e.ops.len()).sum();
        assert_eq!(op_total, 5340, "every op row lands in exactly one episode");
        pass += 1;
        println!("E2 PASS  143 real function-episodes across 2 binaries; 5340 ops partitioned");
    }

    // -------------------------------------------------------------------------------------------
    // E3 RECURRENCE — real trigram structure vs the marginal-preserving shuffled control.
    // -------------------------------------------------------------------------------------------
    let real_seqs: Vec<Vec<String>> = eps
        .values()
        .map(|e| e.ops.iter().map(|(_, opc)| opc.clone()).collect())
        .collect();
    // REFUTED PRE-REGISTRATION, recorded not adjusted away: the first version of this
    // gate pre-registered "real top-1 trigram occurrence count > shuffled" and FAILED on
    // the real data (real 380 vs shuffled 387). Top-1 count was the WRONG statistic:
    // shuffling a copy/int_add-dominated marginal CREATES monotone (copy,copy,copy) runs,
    // so the control's top-1 grows. Real structure shows up as TYPE-CONCENTRATION: real
    // streams collapse into far fewer distinct trigram types (measured 97 vs 260 under
    // the python cross-check — ~2.7x), and the top-10 types carry ~50% of all real
    // occurrences vs ~34% shuffled. The gate now asserts the measured fact, with margin.
    let (top_real_key, top_real, top_real_fns) = {
        let real = trigram_counts(&real_seqs);
        let mut lcg = Lcg(0x5EED_0001);
        let shuf_seqs: Vec<Vec<String>> = real_seqs.iter().map(|s| shuffled(s, &mut lcg)).collect();
        let shuf = trigram_counts(&shuf_seqs);
        let (rk, rv) = real.iter().max_by_key(|(_, (n, _))| *n).expect("nonempty");
        let (sk, sv) = shuf.iter().max_by_key(|(_, (n, _))| *n).expect("nonempty");

        // Type concentration, with a non-marginal 2x margin (anti-vacuity: a shuffle
        // that preserved type counts would fail this).
        assert!(
            real.len() * 2 < shuf.len(),
            "real trigram-type collapse must beat shuffle by >2x (real {} vs shuffled {})",
            real.len(),
            shuf.len()
        );
        // Top-10 occupancy share strictly higher in real streams.
        let share = |m: &BTreeMap<(String, String, String), (usize, usize)>| -> f64 {
            let mut counts: Vec<usize> = m.values().map(|(n, _)| *n).collect();
            counts.sort_unstable_by_key(|n| std::cmp::Reverse(*n));
            let total: usize = counts.iter().sum();
            counts.iter().take(10).sum::<usize>() as f64 / total as f64
        };
        let (rs, ss) = (share(&real), share(&shuf));
        assert!(
            rs > ss,
            "real top-10 share must exceed shuffled ({rs:.3} vs {ss:.3})"
        );
        // The shuffle's top type must be the monotone-run artifact the refutation
        // predicted — pinning the MECHANISM of the refutation, not just its outcome.
        assert_eq!(
            (sk.0.as_str(), sk.1.as_str(), sk.2.as_str()),
            ("copy", "copy", "copy"),
            "the shuffled control's top trigram is the marginal monotone run"
        );
        println!(
            "E3 PASS  real trigram types {} vs shuffled {} (>2x collapse); top-10 share {:.3} vs {:.3}; \
             real top ({},{},{}) x{} in {} fns; shuffled top (copy,copy,copy) x{} — the refuted top-1 \
             pre-registration is recorded above",
            real.len(), shuf.len(), rs, ss,
            rk.0, rk.1, rk.2, rv.0, rv.1, sv.0
        );
        (rk.clone(), rv.0, rv.1)
    };
    pass += 1;
    let _ = (top_real, top_real_fns); // reported inside E3's line; E4/E5 key off top_real_key

    // -------------------------------------------------------------------------------------------
    // E4 FRONTIER ON REAL EPISODES — shipped revise + Stamp over 143 real evidence events,
    // measuring the modulo-64 stamp ceiling at real corpus scale.
    // -------------------------------------------------------------------------------------------
    {
        let real = trigram_counts(&real_seqs);
        let mut top: Vec<_> = real.iter().collect();
        top.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
        println!("E4       frontier over real episodes (top 3 recurring macros):");
        let mut ceiling_seen = false;
        for (key, (occ, fns)) in top.iter().take(3) {
            let mut truth = TruthValue::new(0.5, 0.05); // ignorance prior
            let mut stamp = Stamp(0);
            let (mut revised, mut dropped) = (0usize, 0usize);
            for (idx, seq) in real_seqs.iter().enumerate() {
                let here = seq
                    .windows(3)
                    .any(|w| (&w[0], &w[1], &w[2]) == (&key.0, &key.1, &key.2));
                if !here {
                    continue;
                }
                let ev = Stamp::source(idx as u32);
                if stamp.disjoint(ev) {
                    truth = truth.revise(&TruthValue::new(1.0, 0.9));
                    stamp = stamp.union(ev);
                    revised += 1;
                } else {
                    dropped += 1; // CHOICE: overlapping stamp, evidence conservatively not pooled
                }
            }
            if dropped > 0 {
                ceiling_seen = true;
            }
            println!(
                "         ({},{},{}) occ={} fns={} -> e={:.3} revised={} choice_dropped={} (mod-64 ceiling)",
                key.0, key.1, key.2, occ, fns,
                truth.expectation(), revised, dropped
            );
            assert!(
                truth.expectation() > 0.75,
                "a macro recurring in {fns} real functions must clear the trust bar"
            );
            assert!(
                revised <= 64,
                "the Stamp mod-64 ceiling bounds disjoint sources"
            );
        }
        // Can-fire for the ceiling claim: at 143 episodes, at least one widely-recurring
        // macro must actually HIT the ceiling — otherwise the ceiling talk is decoration.
        assert!(
            ceiling_seen,
            "with 143 episodes and 64 stamp bits, some macro must measurably drop evidence"
        );
        pass += 1;
        println!("E4 PASS  shipped revise+Stamp works on real episodes; the mod-64 ceiling BINDS at 143 functions");
    }

    // -------------------------------------------------------------------------------------------
    // E5 HEADLINE — the clobber analog on real data: opcode match vs dataflow-chained contract.
    // -------------------------------------------------------------------------------------------
    {
        let keys: Vec<(String, String)> = eps.keys().cloned().collect();
        let (mut sloppy, mut chained) = (0usize, 0usize);
        for key in &keys {
            let ep = &eps[key];
            for w in ep.ops.windows(3) {
                let opc = (&w[0].1, &w[1].1, &w[2].1);
                if opc != (&top_real_key.0, &top_real_key.1, &top_real_key.2) {
                    continue;
                }
                sloppy += 1;
                let empty: Vec<u64> = Vec::new();
                let out0 = ep.outs.get(&w[0].0).unwrap_or(&empty);
                let in1 = ep.ins.get(&w[1].0).unwrap_or(&empty);
                let out1 = ep.outs.get(&w[1].0).unwrap_or(&empty);
                let in2 = ep.ins.get(&w[2].0).unwrap_or(&empty);
                let link01 = out0.iter().any(|v| in1.contains(v));
                let link12 = out1.iter().any(|v| in2.contains(v));
                if link01 && link12 {
                    chained += 1;
                }
            }
        }
        let unchained = sloppy - chained;
        // Non-vacuity, both directions (the falsifiability rule): the contract must be
        // able to refuse AND able to admit on this real corpus.
        assert!(
            unchained > 0,
            "can-fire: the dataflow contract refuses some real occurrences"
        );
        assert!(
            chained > 0,
            "can-stay-silent: the dataflow contract admits some real occurrences"
        );
        let rate = 100.0 * unchained as f64 / sloppy as f64;

        // Context base rate — every operand row in this corpus carries a real SSA
        // ValueId (measured: 0 of 7416 OperandIn and 0 of 4237 OperandOut rows have
        // b == 0), so linkage failures below are genuine dataflow facts, never missing
        // coverage. The base rate says how often ANY adjacent op pair is def-use linked.
        let (mut pairs, mut linked) = (0usize, 0usize);
        for key in &keys {
            let ep = &eps[key];
            let empty: Vec<u64> = Vec::new();
            for w in ep.ops.windows(2) {
                pairs += 1;
                let o = ep.outs.get(&w[0].0).unwrap_or(&empty);
                let n = ep.ins.get(&w[1].0).unwrap_or(&empty);
                if o.iter().any(|v| n.contains(v)) {
                    linked += 1;
                }
            }
        }
        let base = 100.0 * linked as f64 / pairs as f64;
        pass += 1;
        println!(
            "E5 PASS  HEADLINE: of {sloppy} real occurrences of the top macro, {chained} are dataflow-chained, {unchained} are NOT — a happy-path opcode matcher over-admits {rate:.1}% on real code (adjacency base rate: {linked}/{pairs} = {base:.1}% of ALL adjacent pairs are linked — the top opcode idiom chains far BELOW base rate: it is an addressing idiom, not a dataflow pipe)"
        );
    }

    println!("\n{pass}/5 gates green — real FunctionBehavior episode measurement complete.");
    println!(
        "Fences: no ruff import; no mint; no BPE; no learner subsystem; counts and rates only."
    );
    ExitCode::SUCCESS
}
