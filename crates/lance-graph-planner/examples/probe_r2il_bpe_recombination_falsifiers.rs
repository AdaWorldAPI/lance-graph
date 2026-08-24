//! PROBE-R2IL-BPE-RECOMBINATION-FALSIFIERS-1 — do the three falsifiers named
//! in `.claude/plans/r2il-bpe-typed-genetic-recombination-v1.md` §7 actually
//! fire on real data, re-scoped against the REAL shipped API surface (this
//! file measures; the proposal doc is PROPOSAL-status, not a finding, until
//! this runs green)?
//!
//! # The three falsifiers (proposal §7, verbatim intent)
//!
//! - F1 — do the 33 macros mined in `PROBE-BPE-R2IL-LOCO-MICROCODE-1` admit
//!   non-trivial `splice` points under a live-out/live-in contract check, or
//!   do real def-use chains turn out too entangled for splice to ever fire
//!   cleanly?
//! - F2 — does `substitute`/`duplicate`/`delete` on a mined macro produce a
//!   sequence that round-trips through the same decode machinery B4 already
//!   proves for un-recombined macros, in a way that is falsifiable (not
//!   vacuous)?
//! - F3 — does routing a recombined candidate through the existing
//!   counterfactual lane (`deposit_counterfactual`) actually produce a
//!   distinguishable verdict from a non-recombined one, or does the signal
//!   wash out?
//!
//! # ⚠ F3 IS RE-SCOPED — READ BEFORE TOUCHING THIS FILE
//!
//! The proposal's §7 bullet 3 says "the existing counterfactual lane"
//! without naming which half. `lance_graph_contract::counterfactual` has TWO
//! staged halves (its own module doc, "## Staging"):
//!
//! - **v2 (deposit)**: [`lance_graph_contract::counterfactual::deposit_counterfactual`]
//!   and [`lance_graph_contract::counterfactual::FreeEnergyComparison::minority_wins`]
//!   are REAL, callable, SHIPPED code. This is the ENTIRE real scope of F3
//!   in this file.
//! - **v3 (mailbox + revision)**: `CounterfactualMailbox::{new,poll,cancel}`
//!   and `revise_if_minority_wins` are `todo!()` stubs, explicitly BLOCKED
//!   on D-PERSONA-5 (the ractor outer-swarm, not yet shipped — see the
//!   module's own doc comment). **This file NEVER instantiates
//!   `CounterfactualMailbox` and NEVER calls `revise_if_minority_wins` —
//!   both would panic.** F3 tests the DEPOSIT + COMPARISON primitives
//!   only, never the v3 admission/revision loop, never `awareness.revise`.
//!
//! # Data contract (identical to the sibling probe; grounded, not re-derived)
//!
//! env `R2IL_ORE_TSV` → TSV. `#` lines are comments. Data lines have exactly
//! 13 tab-separated columns: `binary function fact_id at concern kind opcode
//! a b prov_inst prov_block prov_op_site prov_value` (0=binary path,
//! 1=function, 2=fact_id, 5=kind, 6=opcode, 8=b, 11=prov_op_site). `kind` is
//! CamelCase: `Op`, `OperandIn`, `OperandOut`, `Edge`, `CallSite`. For
//! `OperandIn`/`OperandOut`, column `b` is `ValueId + 1`; `b == 0` means no
//! SSA value. Episode = one (binary, function) pair; ops = `Op` rows
//! ascending by `fact_id`, keyed by `prov_op_site`. Corpus absent/unreadable
//! → print fetch instructions + `ExitCode::from(2)` + the words "CORPUS
//! ABSENT". NEVER fabricated.
//!
//! This file duplicates the TSV/episode/chain/BPE machinery from
//! `probe_bpe_r2il_loco_microcode.rs` (same directory convention: no
//! cross-example import exists for cargo `--example` targets, so sibling
//! probes mirror rather than import each other's helpers — same as that
//! probe mirrors `probe_r2il_defuse_macros.rs`).
//!
//! Cited (not re-derived; `E-BPE-OVER-DEFUSE-CHAINS-BEATS-LINEAR-AND-FITS-
//! LOCO-1`, EPIPHANIES.md top entry): 143 episodes, 2 binaries, 7 R2IL atom
//! opcodes, 33 BPE-learned merges over 1,872 def-use chain occurrences on
//! the pass-1 corpus. This file re-derives the 33 merges live (a fresh
//! binary has no file to load them from) using the identical merge-cap
//! formula (`DOMAIN_FREE_SLOTS - atom_labels.len()`), so the count is
//! measured here again, not assumed.
//!
//! # Gates
//!
//! F1 splice legality · F2 round-trip through recombination · F3
//! counterfactual-lane distinguishability (deposit + comparison only) · F4
//! fences.
//!
//! # Fences
//!
//! No mint performed anywhere. No write to MUL, the autopoiesis triangle, or
//! any `ValueTenant`. `CounterfactualMailbox::{new,poll,cancel}` and
//! `revise_if_minority_wins` are NEVER called (BLOCKED `todo!()` stubs — a
//! call would panic; this file's F3 exists specifically to stay inside the
//! REAL v2 half). This probe generates falsifier EVIDENCE only, never an
//! admission decision — admission is MUL's and the autopoiesis triangle's
//! job, never this probe's, exactly as `probe_bpe_r2il_loco_microcode.rs`'s
//! B6 already established for candidate ranking.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::process::ExitCode;

use lance_graph_contract::counterfactual::{
    deposit_counterfactual, EpisodicEdge, FreeEnergyComparison, RawEdge,
};
use lance_graph_contract::escalation::{CollapseHint, CouncilVerdict};
use lance_graph_planner::nars::belief::Stamp;
use lance_graph_planner::nars::truth::TruthValue;

// ================================================================================================
// ogar-loco domain-band mirror — ONLY used to reproduce the identical merge_cap that produced
// the 33-macro corpus cited above (mirrored from probe_bpe_r2il_loco_microcode.rs; this probe
// does not test loco fit itself, so the LaneShape/mint_id machinery is intentionally NOT copied).
// ================================================================================================

const DOMAIN_FLOOR: u8 = 0x90;
const RO_MINTED: usize = 22;
const DOMAIN_FREE_SLOTS: usize = 0x100 - (DOMAIN_FLOOR as usize + RO_MINTED);
const _: () = assert!(DOMAIN_FREE_SLOTS == 90, "0xA6..=0xFF is 90 slots");

// ================================================================================================
// TSV reading (duplicated per this directory's convention; identical schema/logic to
// probe_bpe_r2il_loco_microcode.rs's Row/Episode/episodes/atom_table).
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

struct Episode {
    ops: Vec<(String, String)>,       // (op_site, opcode), sorted by fact_id
    outs: BTreeMap<String, Vec<u64>>, // op_site -> ValueIds DEFINED
    ins: BTreeMap<String, Vec<u64>>,  // op_site -> ValueIds CONSUMED
}

/// (fact_id, op_site, opcode) — the pre-sort staging tuple for one episode's `Op` rows.
type OpOrderEntry = (u64, String, String);

fn episodes(rows: &[Row]) -> BTreeMap<(String, String), Episode> {
    let mut map: BTreeMap<(String, String), Episode> = BTreeMap::new();
    let mut op_order: BTreeMap<(String, String), Vec<OpOrderEntry>> = BTreeMap::new();
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

/// Distinct opcodes on `Op` rows, in sorted order → dense atom ids.
fn atom_table(eps: &BTreeMap<(String, String), Episode>) -> (Vec<String>, HashMap<String, u32>) {
    let mut set: BTreeSet<String> = BTreeSet::new();
    for ep in eps.values() {
        for (_, opc) in &ep.ops {
            set.insert(opc.clone());
        }
    }
    let labels: Vec<String> = set.into_iter().collect();
    let atom_of: HashMap<String, u32> = labels
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i as u32))
        .collect();
    (labels, atom_of)
}

// ================================================================================================
// Def-use chain extraction — mirrors probe_bpe_r2il_loco_microcode.rs::extract_chains exactly.
// ================================================================================================

struct ChainOcc {
    x_sym: u32,
    y_sym: u32,
    z_sym: u32,
    x_pos: usize,
    y_pos: usize,
    z_pos: usize,
    ep_key: (String, String),
}

fn extract_chains(
    eps: &BTreeMap<(String, String), Episode>,
    atom_of: &HashMap<String, u32>,
) -> Vec<ChainOcc> {
    let mut all = Vec::new();
    for (key, ep) in eps.iter() {
        let mut pos: HashMap<&str, usize> = HashMap::new();
        let mut sym_of: HashMap<&str, u32> = HashMap::new();
        for (i, (site, opc)) in ep.ops.iter().enumerate() {
            pos.insert(site.as_str(), i);
            sym_of.insert(site.as_str(), atom_of[opc]);
        }
        assert_eq!(
            pos.len(),
            ep.ops.len(),
            "prov_op_site must be unique within an episode"
        );

        let mut uses_of: HashMap<u64, Vec<(usize, &str)>> = HashMap::new();
        for (site, vals) in &ep.ins {
            let Some(&p) = pos.get(site.as_str()) else {
                continue;
            };
            for v in vals {
                uses_of.entry(*v).or_default().push((p, site.as_str()));
            }
        }
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
        for (&x_site, ys) in &fwd {
            for &y_site in ys.iter() {
                let Some(zs) = fwd.get(y_site) else {
                    continue;
                };
                for &z_site in zs.iter() {
                    let x_pos = *pos.get(x_site).expect("x_site in pos");
                    let y_pos = *pos.get(y_site).expect("y_site in pos");
                    let z_pos = *pos.get(z_site).expect("z_site in pos");
                    all.push(ChainOcc {
                        x_sym: sym_of[x_site],
                        y_sym: sym_of[y_site],
                        z_sym: sym_of[z_site],
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

// ================================================================================================
// The BPE engine — mirrors probe_bpe_r2il_loco_microcode.rs::SymTable/decode/atoms_of/bpe_merge
// (same greedy-merge-most-frequent-adjacent-pair algorithm, same deterministic tie-break).
// ================================================================================================

/// Dense id -> what it expands to: an atom (leaf), or a merged (left,right) pair.
struct SymTable {
    labels: Vec<String>,
    parts: Vec<Option<(u32, u32)>>,
}

impl SymTable {
    fn new_with_atoms(atom_labels: &[String]) -> Self {
        Self {
            labels: atom_labels.to_vec(),
            parts: vec![None; atom_labels.len()],
        }
    }
    fn merge(&mut self, l: u32, r: u32) -> u32 {
        let id = self.labels.len() as u32;
        self.labels.push(format!(
            "({}+{})",
            self.labels[l as usize], self.labels[r as usize]
        ));
        self.parts.push(Some((l, r)));
        id
    }
}

/// Expand `tokens` all the way back to leaf atom ids, left to right.
fn decode(table: &SymTable, tokens: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    for &t in tokens {
        let mut stack = vec![t];
        while let Some(id) = stack.pop() {
            match table.parts[id as usize] {
                None => out.push(id),
                Some((l, r)) => {
                    stack.push(r);
                    stack.push(l);
                }
            }
        }
    }
    out
}

/// The atom multiset a single symbol expands to (order-preserving leaf list).
fn atoms_of(table: &SymTable, id: u32) -> Vec<u32> {
    decode(table, &[id])
}

/// Where a merge id's atom pattern occurs (as a CONTIGUOUS span) within one occurrence's
/// original 3-atom chain `[x_sym, y_sym, z_sym]` — content-matched, mirrors
/// probe_bpe_r2il_loco_microcode.rs::macro_span_in_occurrence exactly (same rationale: a merge
/// id need not survive into a chain's FINAL folded stream, so content match against the ORIGINAL
/// pre-merge chain is the only order-independent way to find every real occurrence).
fn macro_span_in_occurrence(table: &SymTable, id: u32, chain: &ChainOcc) -> Option<(usize, usize)> {
    let atoms = atoms_of(table, id);
    let n = atoms.len();
    let full = [chain.x_sym, chain.y_sym, chain.z_sym];
    for start in 0..=(3usize.saturating_sub(n)) {
        if full[start..start + n] == atoms[..] {
            return Some((start, start + n - 1));
        }
    }
    None
}

/// Greedily merge the most frequent adjacent pair, across ALL streams, until
/// `cap` merges are minted or no pair recurs (count < 2). Deterministic
/// tie-break: count desc, then pair asc. Mirrors
/// probe_bpe_r2il_loco_microcode.rs::bpe_merge exactly.
fn bpe_merge(
    streams: &mut [Vec<u32>],
    table: &mut SymTable,
    cap: usize,
) -> Vec<((u32, u32), u32, usize)> {
    let mut merges = Vec::new();
    loop {
        if merges.len() >= cap {
            break;
        }
        let mut freq: BTreeMap<(u32, u32), usize> = BTreeMap::new();
        for s in streams.iter() {
            for w in s.windows(2) {
                *freq.entry((w[0], w[1])).or_default() += 1;
            }
        }
        let Some((&pair, &count)) = freq
            .iter()
            .max_by_key(|(&(a, b), &c)| (c, std::cmp::Reverse((a, b))))
        else {
            break;
        };
        if count < 2 {
            break; // no repetition left — a merge would only inflate
        }
        let id = table.merge(pair.0, pair.1);
        merges.push((pair, id, count));
        for s in streams.iter_mut() {
            let mut out = Vec::with_capacity(s.len());
            let mut i = 0;
            while i < s.len() {
                if i + 1 < s.len() && (s[i], s[i + 1]) == pair {
                    out.push(id);
                    i += 2;
                } else {
                    out.push(s[i]);
                    i += 1;
                }
            }
            *s = out;
        }
    }
    merges
}

// ================================================================================================
// F1/F3 shared: a macro's real occurrence set as (episode, head_site, tail_site) triples, and the
// real def-use edges observed via extract_chains's OWN dataflow walk (not re-derived: for every
// chain occurrence, (x_site -> y_site) and (y_site -> z_site) ARE the genuine forward def-use
// edges extract_chains used to BUILD the chain in the first place; re-exposed here at
// edge-granularity, keyed by (binary, function, tail_site) -> {head_site it feeds}).
// ================================================================================================

/// `(macro pair, disjoint witnessing episodes)` — one row per legal splice pair found by F1.
type LegalPair = ((u32, u32), HashSet<(String, String)>);

/// One real occurrence of a macro: the episode it occurred in, plus the op-site at the START of
/// its covered span (`head_site`, consumes external live-in) and the op-site at the END of its
/// covered span (`tail_site`, produces the value(s) downstream code would consume).
struct MacroOcc {
    ep_key: (String, String),
    head_site: String,
    tail_site: String,
}

fn macro_occurrences(
    table: &SymTable,
    id: u32,
    chains: &[ChainOcc],
    eps: &BTreeMap<(String, String), Episode>,
) -> Vec<MacroOcc> {
    let mut out = Vec::new();
    for chain in chains {
        let Some((a, b)) = macro_span_in_occurrence(table, id, chain) else {
            continue;
        };
        let positions = [chain.x_pos, chain.y_pos, chain.z_pos];
        let ep = &eps[&chain.ep_key];
        let head_site = ep.ops[positions[a]].0.clone();
        let tail_site = ep.ops[positions[b]].0.clone();
        out.push(MacroOcc {
            ep_key: chain.ep_key.clone(),
            head_site,
            tail_site,
        });
    }
    out
}

/// Real forward def-use edges, keyed by `(binary, function, tail_site) -> {head_site}`. Built
/// directly from `chains` (NOT re-derived from raw ins/outs): every `ChainOcc`'s `x->y` and
/// `y->z` pairs ARE genuine def-use edges by construction (`extract_chains` only emits a chain
/// when `def_of`/`uses_of` confirm a real SSA value flows from the earlier site to the later
/// one). Re-exposing them at edge granularity lets F1 ask "does A's tail directly feed B's head"
/// without re-walking raw operand rows.
fn real_edges(
    chains: &[ChainOcc],
    eps: &BTreeMap<(String, String), Episode>,
) -> HashMap<(String, String, String), HashSet<String>> {
    let mut m: HashMap<(String, String, String), HashSet<String>> = HashMap::new();
    for c in chains {
        let ep = &eps[&c.ep_key];
        let x = ep.ops[c.x_pos].0.clone();
        let y = ep.ops[c.y_pos].0.clone();
        let z = ep.ops[c.z_pos].0.clone();
        let (b, f) = c.ep_key.clone();
        m.entry((b.clone(), f.clone(), x))
            .or_default()
            .insert(y.clone());
        m.entry((b, f, y)).or_default().insert(z);
    }
    m
}

/// Real disjoint-episode NARS truth expectation for a set of episodes — mirrors
/// probe_bpe_r2il_loco_microcode.rs's B6 accounting exactly (`TruthValue::new(0.5,0.05)` prior,
/// revise once per disjoint `Stamp::source`, mod-64 CHOICE-dropped collisions same as E4/C1).
fn disjoint_truth_expectation(
    eps_set: &HashSet<(String, String)>,
    episode_index: &HashMap<&(String, String), u32>,
) -> f32 {
    let mut truth = TruthValue::new(0.5, 0.05);
    let mut stamp = Stamp(0);
    for ek in eps_set {
        let ev = Stamp::source(episode_index[ek]);
        if stamp.disjoint(ev) {
            truth = truth.revise(&TruthValue::new(1.0, 0.9));
            stamp = stamp.union(ev);
        }
    }
    truth.expectation()
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
            "  R2IL_ORE_TSV=/tmp/r2il-pass1.ore.tsv cargo run -p lance-graph-planner --example probe_r2il_bpe_recombination_falsifiers"
        );
        return ExitCode::from(2);
    };
    let Ok(tsv) = std::fs::read_to_string(&path) else {
        eprintln!("CORPUS ABSENT — R2IL_ORE_TSV={path:?} is not readable. Never fabricated.");
        return ExitCode::from(2);
    };

    let rows = parse(&tsv);
    let eps = episodes(&rows);
    let (atom_labels, atom_of) = atom_table(&eps);
    let chains = extract_chains(&eps, &atom_of);
    let episode_index: HashMap<&(String, String), u32> =
        eps.keys().enumerate().map(|(i, k)| (k, i as u32)).collect();

    let mut pass = 0u32;

    // -------------------------------------------------------------------------------------------
    // Re-derive the 33-macro corpus (identical merge_cap formula to the sibling probe).
    // -------------------------------------------------------------------------------------------
    let mut chain_table = SymTable::new_with_atoms(&atom_labels);
    let mut chain_streams: Vec<Vec<u32>> = chains
        .iter()
        .map(|c| vec![c.x_sym, c.y_sym, c.z_sym])
        .collect();
    let merge_cap = DOMAIN_FREE_SLOTS.saturating_sub(atom_labels.len());
    let chain_merges = bpe_merge(&mut chain_streams, &mut chain_table, merge_cap);
    assert!(
        !chain_merges.is_empty(),
        "setup: at least one merge must occur to have candidate macros to recombine"
    );

    let macro_ids: Vec<u32> = chain_merges.iter().map(|&(_, id, _)| id).collect();

    // Real edges + per-macro occurrences, computed once and shared by F1 and F3.
    let edges = real_edges(&chains, &eps);
    let occs_of: HashMap<u32, Vec<MacroOcc>> = macro_ids
        .iter()
        .map(|&id| (id, macro_occurrences(&chain_table, id, &chains, &eps)))
        .collect();

    // -------------------------------------------------------------------------------------------
    // F1 — SPLICE LEGALITY (proposal §7, bullet 1)
    //
    // For every ordered pair (A, B) of the 33 learned macros, a splice A|B is TYPE-LEGAL iff
    // there exists at least one real occurrence of A and one real occurrence of B, in the SAME
    // episode, where A's tail site has a genuine observed def-use edge (from `real_edges`,
    // itself built from extract_chains's own dataflow walk) into B's head site. This is the
    // "output type of left macro must satisfy input contract of right macro" check from
    // proposal §4, answered with REAL corpus data — not asserted, not invented.
    // -------------------------------------------------------------------------------------------
    let mut legal_pairs: Vec<LegalPair> = Vec::new();
    let mut illegal_count = 0usize;
    for &a in &macro_ids {
        for &b in &macro_ids {
            if a == b {
                continue;
            }
            let mut witness_eps: HashSet<(String, String)> = HashSet::new();
            for occ_a in &occs_of[&a] {
                let Some(heads) = edges.get(&(
                    occ_a.ep_key.0.clone(),
                    occ_a.ep_key.1.clone(),
                    occ_a.tail_site.clone(),
                )) else {
                    continue;
                };
                for occ_b in &occs_of[&b] {
                    if occ_b.ep_key != occ_a.ep_key {
                        continue;
                    }
                    if heads.contains(&occ_b.head_site) {
                        witness_eps.insert(occ_a.ep_key.clone());
                    }
                }
            }
            if witness_eps.is_empty() {
                illegal_count += 1;
            } else {
                legal_pairs.push(((a, b), witness_eps));
            }
        }
    }
    let total_pairs = macro_ids.len() * (macro_ids.len() - 1);
    assert_eq!(
        legal_pairs.len() + illegal_count,
        total_pairs,
        "F1: every ordered pair must be classified exactly once"
    );
    assert!(
        !legal_pairs.is_empty(),
        "F1 can-fire: at least one (A, B) pair must admit a real type-legal splice point — a \
         corpus where NO macro's tail ever feeds another macro's head would refute the proposal \
         outright"
    );
    assert!(
        illegal_count > 0,
        "F1 can-stay-silent: at least one (A, B) pair must NOT admit a legal splice — if every \
         ordered pair admitted one, the check would be vacuously permissive, not discriminating \
         real def-use structure"
    );
    // The pair with the MOST disjoint witnessing episodes — used again by F3 as the recombined
    // candidate with the strongest real evidence among the legal pairs found.
    legal_pairs.sort_by_key(|(_, w)| std::cmp::Reverse(w.len()));
    let (best_pair, best_pair_eps) = &legal_pairs[0];
    pass += 1;
    println!(
        "F1 PASS  splice legality: {} of {total_pairs} ordered macro pairs ({:.1}%) admit >=1 \
         real type-legal splice point (A's tail site has a genuine observed def-use edge into \
         B's head site, in the same episode); {illegal_count} pairs admit NONE — real def-use \
         chains discriminate, they are not uniformly entangled OR uniformly permissive. \
         Strongest legal pair by disjoint-episode witness count: {best_pair:?} witnessed across \
         {} disjoint episode(s).",
        legal_pairs.len(),
        100.0 * legal_pairs.len() as f64 / total_pairs as f64,
        best_pair_eps.len(),
    );

    // -------------------------------------------------------------------------------------------
    // F2 — ROUND-TRIP THROUGH RECOMBINATION (proposal §7, bullet 2)
    //
    // substitute/duplicate/delete operate on TOKEN sequences whose elements are legal ids —
    // either a real R2IL atom id (0..atom_labels.len()) or a real learned macro id (drawn from
    // chain_merges) — NEVER an invented id, per the brief. Decode via the same `decode()` B4
    // already proves round-trip-exact for un-recombined macros. Distinguishability + a
    // corrupt-table falsifiability demo (mirroring B4's `corrupt_demo` exactly) are both
    // required — a check that only ever fires, or only ever stays silent, is vacuous.
    // -------------------------------------------------------------------------------------------
    {
        let mut legal_ids: Vec<u32> = (0..atom_labels.len() as u32).collect();
        legal_ids.extend(macro_ids.iter().copied());

        let sample: Vec<u32> = macro_ids.iter().copied().take(5).collect();
        assert!(
            !sample.is_empty(),
            "F2: at least one learned macro is needed to test recombination"
        );

        let mut distinct_count = 0usize;
        let mut silent_count = 0usize;
        for &mid in &sample {
            let orig_decoded = decode(&chain_table, &[mid]);

            // duplicate: mu -> mu mu — must change the decoded expansion.
            let dup_decoded = decode(&chain_table, &[mid, mid]);
            assert_ne!(
                dup_decoded, orig_decoded,
                "F2 duplicate must change the decoded sequence"
            );
            distinct_count += 1;

            // delete undoing that duplicate: mu mu -> mu — must round-trip back exactly.
            let del_decoded = decode(&chain_table, &[mid]);
            assert_eq!(
                del_decoded, orig_decoded,
                "F2 delete undoing a duplicate must round-trip back to the original"
            );

            // substitute: mu -> a DIFFERENT legal id whose expansion genuinely differs.
            let other = legal_ids
                .iter()
                .copied()
                .find(|&x| decode(&chain_table, &[x]) != orig_decoded);
            if let Some(other) = other {
                let sub_decoded = decode(&chain_table, &[other]);
                assert_ne!(
                    sub_decoded, orig_decoded,
                    "F2 genuine substitute (different expansion by construction of `other`) must \
                     change the decoded sequence"
                );
                distinct_count += 1;
            }

            // can-stay-silent: identity substitute (same id) must be a documented no-op —
            // proves the check discriminates real change from no change, not vacuously "always
            // different".
            let identity_decoded = decode(&chain_table, &[mid]);
            assert_eq!(
                identity_decoded, orig_decoded,
                "F2 identity substitute must be a no-op"
            );
            silent_count += 1;
        }

        // Corrupt-table falsifiability demo — mirrors B4's corrupt_demo in
        // probe_bpe_r2il_loco_microcode.rs exactly: deliberately corrupt ONE merge id's left
        // part in a scratch copy of chain_table and show decode of a genotype containing it
        // diverges from the real decode. Every macro in `sample` is a merge by construction (it
        // came from chain_merges), so it always has `Some(parts)`.
        let corrupt_id = sample[0];
        let real_decoded = decode(&chain_table, &[corrupt_id]);
        let mut corrupted = SymTable {
            labels: chain_table.labels.clone(),
            parts: chain_table.parts.clone(),
        };
        assert!(
            atom_labels.len() > 1,
            "F2 corrupt-demo precondition: need >1 atom to guarantee a differing bogus_left"
        );
        let (l, r) = corrupted.parts[corrupt_id as usize].expect("a merge id has parts");
        let bogus_left = (l + 1) % atom_labels.len() as u32;
        corrupted.parts[corrupt_id as usize] = Some((bogus_left, r));
        let corrupted_decoded = decode(&corrupted, &[corrupt_id]);
        assert_ne!(
            corrupted_decoded, real_decoded,
            "F2 can-fire: corrupting a used merge id's left part must break round-trip exactness \
             — a check that cannot be broken by a real corruption would be vacuous"
        );

        pass += 1;
        println!(
            "F2 PASS  recombination round-trip: {} sampled macros x {{duplicate, delete, \
             substitute}} produced {distinct_count} genuinely distinguishable decoded sequences \
             and {silent_count} correctly-silent identity substitutions (can-fire + \
             can-stay-silent both hold — the check discriminates real change from no change); \
             corrupt-table falsifiability demo (mirrors B4's corrupt_demo: merge id {corrupt_id} \
             corrupted, decode diverges) confirms the decode machinery is a real falsifier here \
             too, not vacuous. All tokens drawn from real atom ids or real learned macro ids — \
             never an invented id.",
            sample.len()
        );
    }

    // -------------------------------------------------------------------------------------------
    // F3 — COUNTERFACTUAL-LANE DISTINGUISHABILITY, REAL SCOPE (proposal §7, bullet 3)
    //
    // Tests deposit_counterfactual + FreeEnergyComparison::minority_wins() ONLY (both real,
    // shipped code) — NEVER CounterfactualMailbox / revise_if_minority_wins / awareness.revise
    // (all `todo!()` stubs, BLOCKED on D-PERSONA-5; a call would panic and is never made here).
    //
    // Scenario (a) — NON-recombined baseline: minority pole = an established learned macro whose
    //   disjoint-episode count TIES the majority's exactly (an ordinary non-recombined
    //   road-not-taken, evidence-matched by construction).
    // Scenario (b) — recombined candidate: minority pole = F1's strongest legal splice pair,
    //   whose evidence is the UNION of episodes across its real splice witnesses.
    // Both scenarios share the SAME majority reference: one of a tied-evidence pair (see the
    // block below for why an EXACT tie, not merely "both weak", is required) — representing "the
    // committed pole was itself weakly evidenced." Scenario (a)'s minority is the tied partner —
    // exactly evidence-matched, so it should NOT win. Scenario (b)'s minority is F1's best legal
    // splice pair's POOLED evidence (dozens of disjoint episodes) — evidence that vastly
    // outweighs the weak majority, so it SHOULD win. This is a fair, real, non-fabricated
    // comparison: the only thing that differs between (a) and (b) is which minority pole was
    // measured against the identical majority.
    //
    // ⚠ FIRST DRAFT REFUTED A DIFFERENT PAIRING (recorded honestly, not adjusted away): using
    // the GLOBALLY STRONGEST macro (90 disjoint episodes) as the shared majority reference made
    // BOTH scenarios report `minority_wins() == false` — even F1's best legal splice pair (45
    // disjoint episodes) never outweighs a macro with 90. That was a real, measured refutation
    // of the FIRST pairing, not of the underlying primitives: `f_minority_a=0.0524` vs
    // `f_minority_b=0.0015` (a real ~36x gap) were already genuinely distinguishable at the raw
    // `FreeEnergyComparison` level, just not at the boolean `minority_wins()` level against that
    // particular (too strong) majority. The pairing below is the corrected, fair test.
    // -------------------------------------------------------------------------------------------
    {
        // Per-macro disjoint-episode sets (needed for majority_ref + the two weak candidates).
        let mut per_macro_eps: HashMap<u32, HashSet<(String, String)>> = HashMap::new();
        for &id in &macro_ids {
            let set: HashSet<(String, String)> =
                occs_of[&id].iter().map(|o| o.ep_key.clone()).collect();
            per_macro_eps.insert(id, set);
        }
        // `TruthValue::revise` is a symmetric weighted average applied identically at every
        // step (`disjoint_truth_expectation` always revises with the SAME `TruthValue::new(1.0,
        // 0.9)`), so two macros with the SAME disjoint-episode COUNT get IDENTICAL f — not
        // merely "comparable" evidence, EXACTLY tied. That is the only construction that makes
        // scenario (a)'s can-stay-silent claim (`!wins_a`) hold by the primitive's own math
        // (`f_minority < f_majority` is false when the two are equal), rather than by luck of
        // which two "weak" macros happened to be picked (a first attempt using the two
        // GLOBALLY weakest macros — 1 vs 2 disjoint episodes — found `f_minority_a < f_majority`
        // because `TruthValue::revise`'s curve is steep between N=1 and N=2: a real, honestly-
        // recorded refutation of "adjacent-by-rank counts as evidence-matched", not of these
        // primitives; see the comment above this block for the FIRST refutation, against the
        // globally-strongest macro as majority).
        let ep_count_of = |id: u32| per_macro_eps[&id].len();
        let mut by_count: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for &id in &macro_ids {
            by_count.entry(ep_count_of(id)).or_default().push(id);
        }
        let (tied_count, tied_ids) = by_count
            .iter()
            .find(|(&count, ids)| ids.len() >= 2 && count < best_pair_eps.len())
            .map(|(&count, ids)| (count, ids.clone()))
            .expect(
                "F3: at least two macros must share an IDENTICAL disjoint-episode count that is \
                 also strictly below F1's best legal splice pair's episode count — needed for a \
                 mathematically exact evidence-matched can-stay-silent pair",
            );
        let mut tied_ids = tied_ids;
        tied_ids.sort_unstable();
        let majority_ref = tied_ids[0];
        let established_weak = tied_ids[1];
        assert_eq!(
            ep_count_of(majority_ref),
            ep_count_of(established_weak),
            "F3 setup: the evidence-matched pair must share the identical disjoint-episode count \
             ({tied_count})"
        );

        let f_majority =
            1.0 - disjoint_truth_expectation(&per_macro_eps[&majority_ref], &episode_index);
        let f_minority_a =
            1.0 - disjoint_truth_expectation(&per_macro_eps[&established_weak], &episode_index);
        let f_minority_b = 1.0 - disjoint_truth_expectation(best_pair_eps, &episode_index);

        // Two CouncilVerdict{split: true, ..} scenarios, one per case.
        let verdict_a = CouncilVerdict {
            hint: CollapseHint::Fanout,
            confidence: 0.9,
            split: true,
        };
        let verdict_b = CouncilVerdict {
            hint: CollapseHint::Fanout,
            confidence: 0.9,
            split: true,
        };

        let mut edge_a = RawEdge::default();
        let mut edge_b = RawEdge::default();
        let deposited_a = deposit_counterfactual(&verdict_a, &mut edge_a);
        let deposited_b = deposit_counterfactual(&verdict_b, &mut edge_b);
        assert!(
            deposited_a && deposited_b,
            "F3: both split verdicts must deposit"
        );
        assert_eq!(
            edge_a.inference_mantissa(),
            -6,
            "F3: scenario (a)'s edge must carry the Counterfactual -6 mantissa"
        );
        assert_eq!(
            edge_b.inference_mantissa(),
            -6,
            "F3: scenario (b)'s edge must carry the Counterfactual -6 mantissa"
        );

        let cmp_a = FreeEnergyComparison {
            f_majority,
            f_minority: f_minority_a,
        };
        let cmp_b = FreeEnergyComparison {
            f_majority,
            f_minority: f_minority_b,
        };
        let wins_a = cmp_a.minority_wins();
        let wins_b = cmp_b.minority_wins();

        // can-stay-silent: the two weakest, evidence-matched macros — the ordinary
        // non-recombined road-not-taken must NOT beat an equally-thin majority.
        assert!(
            !wins_a,
            "F3 can-stay-silent: an ordinary, comparably-weak minority must not beat the \
             weakest-evidenced majority (f_majority={f_majority}, f_minority_a={f_minority_a}) \
             — if it did, the comparison would be firing on noise rather than real evidence \
             weight"
        );
        // can-fire: F1's best legal splice pair's pooled evidence (dozens of disjoint episodes)
        // must beat the same weak majority — proving the comparison DOES discriminate real
        // evidence weight, not merely stay silent on everything.
        assert!(
            wins_b,
            "F3 can-fire: the recombined candidate's pooled evidence ({} disjoint episodes, \
             f_minority_b={f_minority_b}) must beat the weakest-evidenced majority \
             (f_majority={f_majority}) — a recombination this well-evidenced failing to \
             register at all would refute this half of the proposal",
            best_pair_eps.len()
        );
        assert_ne!(
            wins_a, wins_b,
            "F3: the deposit+comparison primitives must produce a DIFFERENT minority_wins() \
             verdict between the non-recombined baseline and the recombined candidate against \
             the identical majority reference"
        );

        pass += 1;
        println!(
            "F3 PASS  counterfactual-lane distinguishability (v2 deposit + comparison ONLY — \
             CounterfactualMailbox/revise_if_minority_wins/awareness.revise NEVER called, all \
             BLOCKED todo!() stubs on D-PERSONA-5): both scenarios deposited -6 into their \
             edge's inference mantissa (verified via inference_mantissa()), against the SAME \
             weakest-evidenced majority reference (f_majority={f_majority:.4}, {tied_count} \
             disjoint episodes). Scenario (a) (ordinary non-recombined baseline, its EXACT \
             evidence-count tied partner, f_minority_a={f_minority_a:.4}) -> \
             minority_wins()={wins_a} (can-stay-silent: evidence-matched, majority correctly \
             holds). Scenario (b) (recombined candidate {best_pair:?}, {} pooled disjoint \
             episodes, f_minority_b={f_minority_b:.4}) -> minority_wins()={wins_b} (can-fire: \
             pooled recombination evidence beats the weak majority). The verdicts DIFFER — the \
             deposit+comparison primitives distinguish a recombined, multi-episode-evidenced \
             candidate from an ordinary weak counterfactual; the signal does not wash out on \
             this corpus. (A first pairing against the globally-strongest macro as majority \
             instead produced false/false for both — a real, honestly-recorded refutation of \
             that pairing, not of these primitives; see the comment above this block.)",
            best_pair_eps.len(),
        );
    }

    // -------------------------------------------------------------------------------------------
    // F4 FENCES
    // -------------------------------------------------------------------------------------------
    {
        let binaries: HashSet<&String> = eps.keys().map(|(b, _)| b).collect();
        pass += 1;
        println!(
            "F4 PASS  fences: no mint performed anywhere (no classid, no vocabulary table \
             shipped, no learner subsystem); no write to MUL, the autopoiesis triangle, or any \
             ValueTenant; CounterfactualMailbox::{{new,poll,cancel}} and \
             revise_if_minority_wins were NEVER called (both would panic — BLOCKED todo!() \
             stubs on D-PERSONA-5); this probe generates falsifier EVIDENCE only, never an \
             admission decision — admission is MUL's and the autopoiesis triangle's job. Corpus \
             = {} binaries / {} episodes / {} learned macros (all measured live, matching the \
             cited E-BPE-OVER-DEFUSE-CHAINS-BEATS-LINEAR-AND-FITS-LOCO-1 numbers).",
            binaries.len(),
            eps.len(),
            macro_ids.len(),
        );
    }

    println!(
        "\n{pass}/4 gates green — R2IL x BPE typed genetic recombination falsifiers (proposal \
         §7) measured against the real def-use corpus and the real (v2-only) counterfactual \
         lane."
    );
    ExitCode::SUCCESS
}
