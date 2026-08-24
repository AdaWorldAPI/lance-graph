//! PROBE-BPE-R2IL-LOCO-MICROCODE-1 — do BPE-learned merges over R2IL def-use
//! chains fit `ogar-loco`'s fixed microcode geometry, and does the def-use
//! chain merge carrier already exhibit patterns three SHIPPED lance-graph
//! codecs use elsewhere for the same problem ("capacity through hierarchy,
//! never widening")?
//!
//! # The architecture under test (operator-stated; treated as the spec)
//!
//! - **R2IL is the OPCODE** — the atomic vocabulary. Thinking atoms + NARS.
//!   Not P-code, not SLEIGH.
//! - **IR code lands in V4** — physically identical to V3 (the 16-byte
//!   content-blind dock: `classid(4) + 12-byte payload`; the classid selects
//!   the reading, so no layout version bump). This probe writes no bytes at
//!   rest and never touches V3/V4 storage — it is a pure in-memory
//!   measurement over evidence already on disk.
//! - **BPE merges become MICROCODE MACROS in `ogar-loco`** — turning
//!   low-code orchestration powerful through R2IL capabilities. This probe
//!   generates the CANDIDATE macro table and measures whether it fits the
//!   shipped `ogar-loco` call geometry; it does not mint anything.
//!
//! # Scope fence — admission is NOT this probe's job
//!
//! A sibling architecture review found that MUL
//! (`lance_graph_contract::mul`: `GateDecision`, `Homeostasis`,
//! `FlowState`) is the real admission gate, and the autopoiesis triangle
//! (`ValueTenant::{FrozenStyle,LearnedStyle,ExploreStyle}`) is
//! resonance-based thinking (PROBE-METACOGNITIVE-TRIANGLE-1:
//! `RungReceipt`-only judging — the meta pass's object is the reasoning,
//! never the puzzle; `FreeEnergyComparison::minority_wins()`; Explore runs
//! in a COUNTERFACTUAL lane via `deposit_counterfactual` stamping
//! `RawEdge -6`, never observed truth), NOT an RL policy. **This probe
//! generates CANDIDATES only.** It must NOT claim to gate, promote, or
//! admit anything — B6 below (frontier candidate ranking) ranks candidates
//! via the shipped `TruthValue`/`Stamp` primitives already used in the
//! sibling probes in this directory, explicitly labelled as candidate
//! generation, never as the real MUL/triangle admission path. The words
//! "freeze", "admit", "promote", and "gate" are never used for what B6
//! does — that vocabulary belongs to MUL and the triangle.
//!
//! # The `ogar-loco` geometry (PINNED from source, verified before this
//! file was written; cite, do not re-derive)
//!
//! Read from `AdaWorldAPI/OGAR`, `crates/ogar-loco/src/lib.rs`:
//!
//! - `pub struct Call { function: FnIndex, values: [u8; MAX_VALUES_PER_CALL] }`
//! - `pub struct FnIndex(pub u8)`; `FnIndex::NOP == FnIndex(0x00)` — `0x00`
//!   is reserved, never a real function.
//! - `pub const MAX_VALUES_PER_CALL: usize = 3`.
//! - `pub const BODY_BYTES: usize = 360` (const-asserted in source).
//! - `enum LaneShape { Pairs, Triples, Quads }`:
//!   - `Pairs`   = `6 x (u8:u8)`       → 2 bytes/call, 1 immediate, **180 calls/node**
//!   - `Triples` = `4 x (u8:u8:u8)`    → 3 bytes/call, 2 immediates, **120 calls/node**
//!   - `Quads`   = `3 x (u8:u8:u8:u8)` → 4 bytes/call, 3 immediates, **90 calls/node**
//!   - every shape spends 12 bytes per lane and the same 360 `BODY_BYTES`.
//! - **The `FnIndex` domain is BANDED, stored-byte ABI**
//!   (`pub const DOMAIN_FLOOR: u8 = 0x90`, const-asserted "moving it
//!   reinterprets every persisted node"): bytes `0x01..0x8F` (143 slots)
//!   are the shared computational core, `0x90..0xFF` (112 slots) is the
//!   domain band. Within the domain band, `crates/ogar-ro` has already
//!   minted `0x90..0xA5` (22 RO/BFO predicates — verified live against
//!   `ogar-ro`'s codebook in this checkout) — so only `0xA6..0xFF`
//!   (**90 slots**) are free for new domain mints. This probe's atoms AND
//!   its BPE-learned merges together share that 90-slot budget: this is
//!   the real headroom for merges, not 255.
//!
//! This probe does **NOT** import `ogar-loco` (a separate cargo
//! workspace). The shapes above are mirrored probe-locally, exactly as the
//! sibling probes in this directory mirror external schemas they cite.
//!
//! # Data contract (grounded; not re-derived — cited from
//! `probe_r2il_real_episodes.rs` / `probe_r2il_defuse_macros.rs`, same
//! directory)
//!
//! env `R2IL_ORE_TSV` → TSV. `#` lines are comments. Data lines have
//! exactly 13 tab-separated columns: `binary function fact_id at concern
//! kind opcode a b prov_inst prov_block prov_op_site prov_value`
//! (0=binary path, 1=function, 2=fact_id, 5=kind, 6=opcode, 8=b,
//! 11=prov_op_site). `kind` is CamelCase: `Op`, `OperandIn`, `OperandOut`,
//! `Edge`, `CallSite`. For `OperandIn`/`OperandOut`, column `b` is
//! `ValueId + 1`; `b == 0` means no SSA value. Episode = one (binary,
//! function) pair; ops = `Op` rows ascending by `fact_id`, keyed by
//! `prov_op_site`. Corpus absent/unreadable → print fetch instructions +
//! `ExitCode::from(2)` + the words "CORPUS ABSENT". NEVER fabricated.
//!
//! Orchestrator-measured numbers (citable in comments, NEVER asserted as
//! constants by this probe — this file measures live and reports a
//! RELATION, not a hardcoded number): 143 episodes, 2 binaries, 5340 Op
//! rows, 7 distinct R2IL atom opcodes on Op rows specifically (copy,
//! int_add, load, store, cbranch, return, call — the Op-row-only count,
//! NOT the 9-opcode census that also counts operand-row parent opcodes),
//! 1872 length-3 def-use chains with 27 distinct signatures, adjacency
//! def-use base rate 31.5%.
//!
//! **Two DIFFERENT atom-count quantities appear below — kept explicitly
//! distinct after an orchestrator review caught them conflated in a prior
//! draft.** `total_corpus_atoms` is the TRUE corpus total: every `Op` row
//! across every episode, counted once each (5340 on the cited corpus).
//! `chain_atoms_before` is atom-SLOTS CONSUMED BY CHAIN-OCCURRENCE STREAMS
//! — 3 per def-use chain occurrence (1872 chains × 3 = 5616 on the cited
//! corpus) — which is neither `<=` nor `>=` the corpus total in general,
//! since chain occurrences can overlap/share underlying ops and not every
//! op sits inside a length-3 chain. B1 prints both, labelled. B2's chain
//! BPE run uses `chain_atoms_before`; B3's linear control run uses
//! `total_corpus_atoms` (one stream per episode, covering every Op row
//! exactly once) — asserted equal to `total_corpus_atoms` at B3's setup.
//! The two BPE runs therefore operate over different-sized inputs by
//! design; B3's per-slot metric (tokens saved / merges used) is what makes
//! the comparison fair despite that, not equal input sizes.
//!
//! # Pre-registration — B3 (control)
//!
//! `probe_r2il_defuse_macros.rs`'s C4 gate already found def-use chains
//! concentrate MORE than linear-window trigrams on this corpus. This
//! probe's B3 asks the SAME question through the BPE lens: does BPE over
//! def-use chains achieve better COMPRESSION PER VOCABULARY SLOT SPENT
//! than the same BPE algorithm run over the linear opcode stream of the
//! same episodes? Pre-registered direction: YES (def-use chains carry
//! genuine dataflow recurrence; linear adjacency is largely incidental,
//! per the E5 finding cited above). The assertion is left to FAIL LOUDLY
//! if refuted — a refutation is a finding and must be recorded in place,
//! never adjusted away.
//!
//! # Pre-registration — INV1/INV2/INV3 (cross-checks against shipped
//! precedent)
//!
//! These measure whether the BPE/loco design in THIS probe already
//! exhibits three patterns three SHIPPED lance-graph codecs use for
//! "capacity through hierarchy, never width" on different axes. A
//! negative result is a finding about THIS probe's current shape, not a
//! defect in the codecs cited:
//!
//! - **INV1 (HighHeelBGZ, `crates/highheelbgz/src/lib.rs`)** —
//!   `SpiralAddress` uses `stride` as a ROLE discriminant ("Finger 2 —
//!   STRIDE MATCH… different stride → different ROLE → categorically
//!   different", verified in source). Applied here: does grouping our
//!   learned macros by their SOURCE atom multiset (opcode composition)
//!   produce natural role clusters?
//! - **INV2 (bgz17's HHTL, `crates/bgz17/src/layered.rs`)** —
//!   `LayeredScope::search` does Scent-prune-then-escalate: Layer 0
//!   (scent, 1 byte, ρ=0.937) prunes before Layer 2 (full base L1, 102
//!   bytes, ρ=0.992) runs (`crates/bgz17/src/lib.rs` module docs,
//!   verified in source). Applied here: does a cheap opcode-multiset
//!   check prune most def-use-chain candidates before the expensive exact
//!   dataflow walk?
//! - **INV3 (BGZ-HHTL-D, `crates/bgz-tensor/BGZ_HHTL_D.md`)** — shares ONE
//!   palette across many same-shape/same-role tensors rather than minting
//!   per-instance (480 tensors → 26 palette groups, verified in the doc).
//!   Applied here: do multiple episodes that learn "the same" macro
//!   converge onto ONE mint, or does each risk a duplicate?
//!
//! # Gates (each can fail; each failure would be a finding)
//!
//! B1 atoms · B2 BPE over def-use chains · B3 control (linear BPE) ·
//! B4 reconstructibility · B5 loco fit · B6 candidate ranking (NOT
//! admission) · INV1/INV2/INV3 · B7 fences.
//!
//! # Fences
//!
//! No mint actually performed: no classid minted, no vocabulary table
//! shipped, no learner subsystem, no write to any `ValueTenant`/MUL/
//! triangle type. `ogar-loco`, bgz17's cascade shape, and `ruff_r2il` are
//! MIRRORED/CITED, never imported. Corpus is 2 binaries at the pass-1
//! seven-opcode convention, so nothing here is a claim about x86-64 in
//! general. V4 is physically V3 (no layout bump) and this probe writes no
//! bytes at rest. This probe generates ranked CANDIDATES ONLY — admission
//! is MUL's and the autopoiesis triangle's, not this probe's, and no
//! vocabulary here (freeze/admit/promote/gate) is used for what B6 does.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::process::ExitCode;

use lance_graph_planner::nars::belief::Stamp;
use lance_graph_planner::nars::truth::TruthValue;

// ================================================================================================
// ogar-loco geometry mirror (probe-local; cited from OGAR source above, never imported)
// ================================================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LaneShape {
    Pairs,
    Triples,
    Quads,
}

impl LaneShape {
    /// Max immediate bytes a `Call` in this shape carries alongside its
    /// `FnIndex` (mirrors `MAX_VALUES_PER_CALL` sliced per shape).
    fn max_immediates(self) -> usize {
        match self {
            LaneShape::Pairs => 1,
            LaneShape::Triples => 2,
            LaneShape::Quads => 3,
        }
    }
    fn calls_per_node(self) -> usize {
        match self {
            LaneShape::Pairs => 180,
            LaneShape::Triples => 120,
            LaneShape::Quads => 90,
        }
    }
}

const DOMAIN_FLOOR: u8 = 0x90;
/// Already minted by `ogar-ro` (`0x90..=0xA5`, verified against its
/// codebook: `IS_A` at `0x90` through `CONFOUNDS_TEST` at `0xA5`).
const RO_MINTED: usize = 22;
/// The real headroom for this probe's atoms + merges: `0xA6..=0xFF`.
const DOMAIN_FREE_SLOTS: usize = 0x100 - (DOMAIN_FLOOR as usize + RO_MINTED);
const _: () = assert!(DOMAIN_FREE_SLOTS == 90, "0xA6..=0xFF is 90 slots");

/// Mint id for the `offset`-th new domain-band slot (atoms first, then
/// merges) — `None` once the 90-slot budget is exhausted. Probe-local
/// accounting only; nothing is actually minted anywhere.
fn mint_id(offset: usize) -> Option<u8> {
    let v = DOMAIN_FLOOR as usize + RO_MINTED + offset;
    if v <= 0xFF {
        Some(v as u8)
    } else {
        None
    }
}

// ================================================================================================
// TSV reading (duplicated per this directory's convention — separate example binary, same
// schema as `probe_r2il_real_episodes.rs` / `probe_r2il_defuse_macros.rs`; no cross-example
// import exists for cargo `--example` targets)
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
// Def-use chain extraction — mirrors `probe_r2il_defuse_macros.rs::extract_chains` exactly
// (same directory, same corpus, same forward-only fence), retargeted to atom-symbol ids and
// keeping site strings so INV2's expensive check can re-consult outs/ins.
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

/// Adjacent linear window-of-3 candidates, one per (episode, position) —
/// used by B3's control BPE and INV2's cheap-check candidate pool. Sites
/// are kept so INV2's expensive stage can re-run the exact dataflow check.
struct WinOcc {
    ep_key: (String, String),
    sites: [String; 3],
    syms: [u32; 3],
}

fn extract_windows(
    eps: &BTreeMap<(String, String), Episode>,
    atom_of: &HashMap<String, u32>,
) -> Vec<WinOcc> {
    let mut out = Vec::new();
    for (key, ep) in eps.iter() {
        for w in ep.ops.windows(3) {
            out.push(WinOcc {
                ep_key: key.clone(),
                sites: [w[0].0.clone(), w[1].0.clone(), w[2].0.clone()],
                syms: [atom_of[&w[0].1], atom_of[&w[1].1], atom_of[&w[2].1]],
            });
        }
    }
    out
}

// ================================================================================================
// The BPE engine — generalizes `probe_token_bpe_geometry.rs::BpeTable` from bytes to symbol ids
// (same directory, same greedy-merge-most-frequent-adjacent-pair algorithm, same deterministic
// tie-break). Streams here are either 3-symbol def-use chain occurrences (B2, where positions 0-1
// and 1-2 ARE def-use edges by construction of `extract_chains`) or per-episode linear opcode
// sequences (B3 control).
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

/// Greedily merge the most frequent adjacent pair, across ALL streams, until
/// `cap` merges are minted or no pair recurs (count < 2). Deterministic
/// tie-break: count desc, then pair asc (mirrors `probe_token_bpe_geometry`'s
/// `BpeTable::train`). Returns `((left,right), new_id, count_at_merge_time)`.
fn bpe_merge(
    streams: &mut Vec<Vec<u32>>,
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
            "  R2IL_ORE_TSV=/tmp/r2il-pass1.ore.tsv cargo run -p lance-graph-planner --example probe_bpe_r2il_loco_microcode"
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
    let windows = extract_windows(&eps, &atom_of);
    let episode_index: HashMap<&(String, String), u32> =
        eps.keys().enumerate().map(|(i, k)| (k, i as u32)).collect();

    let mut pass = 0u32;

    // -------------------------------------------------------------------------------------------
    // B2 (computed before B1's print so B1 can report post-merge domain headroom, per spec)
    // -------------------------------------------------------------------------------------------
    assert!(
        chains
            .iter()
            .all(|c| c.x_pos < c.y_pos && c.y_pos < c.z_pos),
        "def-use chain positions must be strictly increasing by construction"
    );

    // Two DIFFERENT atom-count quantities, kept explicitly separate (defect fix — a prior
    // draft printed `chain_atoms_before` as bare "atoms", which reads as the corpus total but
    // is actually 3 * chains.len(): the atom-slots CONSUMED BY CHAIN-OCCURRENCE STREAMS
    // (chains overlap/share underlying ops, and not every op sits in a length-3 chain, so
    // this is neither <= nor >= the corpus total in general). `total_corpus_atoms` below is
    // the TRUE total: every Op row across every episode, counted once each.
    let total_corpus_atoms: usize = eps.values().map(|ep| ep.ops.len()).sum();

    let mut chain_table = SymTable::new_with_atoms(&atom_labels);
    let mut chain_streams: Vec<Vec<u32>> = chains
        .iter()
        .map(|c| vec![c.x_sym, c.y_sym, c.z_sym])
        .collect();
    let chain_streams_orig = chain_streams.clone();
    let merge_cap = DOMAIN_FREE_SLOTS.saturating_sub(atom_labels.len());
    // NOTE: this is chain-occurrence atom-slots (3 * chains.len()), NOT total_corpus_atoms —
    // see the comment above `total_corpus_atoms` for why the two differ.
    let chain_atoms_before: usize = chain_streams.iter().map(|s| s.len()).sum();
    assert_eq!(
        chain_atoms_before,
        chains.len() * 3,
        "sanity: every chain stream is exactly 3 atoms wide"
    );
    let chain_merges = bpe_merge(&mut chain_streams, &mut chain_table, merge_cap);
    let chain_tokens_after: usize = chain_streams.iter().map(|s| s.len()).sum();
    let chain_stopped_by_cap = chain_merges.len() == merge_cap;

    // -------------------------------------------------------------------------------------------
    // B1 ATOMS
    // -------------------------------------------------------------------------------------------
    {
        assert!(
            !atom_labels.is_empty() && atom_labels.len() <= 9,
            "B1: atom count must be a real R2IL opcode vocabulary, bounded by the 9-opcode \
             census ceiling (this can fail on a richer corpus)"
        );
        assert!(
            atom_labels.len() < DOMAIN_FREE_SLOTS,
            "B1: the R2IL atom vocabulary must leave room for merges within the domain band's \
             90 free slots (0xA6..=0xFF) — atoms alone must not exhaust the budget"
        );
        let remaining = DOMAIN_FREE_SLOTS - atom_labels.len() - chain_merges.len();
        pass += 1;
        println!(
            "B1 PASS  {} distinct R2IL atom opcodes ({:?}) across {total_corpus_atoms} total Op \
             rows in the corpus; merge budget = 90 - {} atoms = {} slots; after B2's {} merges, \
             {} domain slots remain (this can fail on a richer corpus with more atom opcodes or \
             a larger merge budget consumed)",
            atom_labels.len(),
            atom_labels,
            atom_labels.len(),
            merge_cap,
            chain_merges.len(),
            remaining
        );
    }

    // -------------------------------------------------------------------------------------------
    // B2 BPE OVER DEF-USE CHAINS
    // -------------------------------------------------------------------------------------------
    {
        assert!(
            !chain_merges.is_empty() || chain_atoms_before == 0,
            "B2: some merge must occur on a nonempty chain corpus"
        );
        let stop_reason = if chain_stopped_by_cap {
            "the 90-slot domain cap"
        } else {
            "corpus exhaustion (no adjacent pair recurred >=2 times)"
        };
        let compression = chain_atoms_before as f64 / chain_tokens_after.max(1) as f64;
        // Probe-local accounting only: mint ids for the first and last merge, drawn from the
        // 90-slot domain band AFTER the atom vocabulary's own slots (never actually written
        // anywhere — see B7's "no mint actually performed" fence). Formatted as plain hex, not
        // the raw `Option<u8>` Debug form, since the slot is always occupied here (merge_cap
        // guarantees `atom_labels.len() + chain_merges.len() - 1 < DOMAIN_FREE_SLOTS`); the
        // OVERFLOW fallback exists only for the theoretical case a richer corpus exhausts it.
        let fmt_mint = |offset: usize| match mint_id(offset) {
            Some(v) => format!("0x{v:02X}"),
            None => "OVERFLOW".to_string(),
        };
        let first_mint = fmt_mint(atom_labels.len());
        let last_mint = fmt_mint(atom_labels.len() + chain_merges.len().saturating_sub(1));
        pass += 1;
        println!(
            "B2 PASS  {} merges over {} def-use chain occurrences (each merge WOULD mint a \
             probe-local domain FnIndex in 0xA6..=0xFF, e.g. first={first_mint} \
             last={last_mint} — accounting only, nothing minted); stopped by {stop_reason}; \
             {chain_atoms_before} atom-slots CONSUMED BY CHAIN-OCCURRENCE STREAMS (3 per \
             occurrence, NOT the {total_corpus_atoms}-atom corpus total — see B1) -> \
             {chain_tokens_after} tokens after merging (compression {compression:.3}x). \
             Deterministic tie-break: count desc then pair asc.",
            chain_merges.len(),
            chains.len(),
        );
    }

    // -------------------------------------------------------------------------------------------
    // B3 CONTROL — chains vs linear (pre-registered direction, may fail loudly if refuted)
    // -------------------------------------------------------------------------------------------
    let (linear_merges, linear_atoms_before, linear_tokens_after) = {
        let mut linear_table = SymTable::new_with_atoms(&atom_labels);
        let mut linear_streams: Vec<Vec<u32>> = eps
            .values()
            .map(|ep| ep.ops.iter().map(|(_, opc)| atom_of[opc]).collect())
            .collect();
        let before: usize = linear_streams.iter().map(|s| s.len()).sum();
        let merges = bpe_merge(&mut linear_streams, &mut linear_table, merge_cap);
        let after: usize = linear_streams.iter().map(|s| s.len()).sum();
        (merges, before, after)
    };
    // The linear control's "before" count IS the true corpus total (one stream per episode,
    // covering every Op row exactly once) — unlike chain_atoms_before, which is chain-occurrence
    // atom-slots. Both BPE runs are still fairly comparable per-slot (see B3 below) because the
    // metric normalizes by merges used, not by the differing input sizes.
    assert_eq!(
        linear_atoms_before, total_corpus_atoms,
        "the linear control's atom stream must equal the true corpus total"
    );
    {
        let chain_per_slot = (chain_atoms_before as f64 - chain_tokens_after as f64)
            / chain_merges.len().max(1) as f64;
        let linear_per_slot = (linear_atoms_before as f64 - linear_tokens_after as f64)
            / linear_merges.len().max(1) as f64;
        assert!(
            chain_per_slot > linear_per_slot,
            "B3 pre-registered: def-use-chain BPE should achieve MORE compression per domain \
             slot spent than linear-window BPE on the same episodes (chain {chain_per_slot:.3} \
             tokens/slot vs linear {linear_per_slot:.3} tokens/slot) — REFUTED if the inequality \
             does not hold; the refutation is recorded here, not adjusted away"
        );
        pass += 1;
        println!(
            "B3 PASS  compression-per-slot: def-use chains {chain_per_slot:.3} tokens/merge \
             ({} merges over {chain_atoms_before} chain-occurrence atom-slots) vs linear stream \
             {linear_per_slot:.3} tokens/merge ({} merges over {linear_atoms_before} corpus-total \
             atoms, == the {total_corpus_atoms} from B1) — two DIFFERENT input sizes, but the \
             metric normalizes by merges used, so the per-slot comparison is fair",
            chain_merges.len(),
            linear_merges.len(),
        );
    }

    // -------------------------------------------------------------------------------------------
    // B4 RECONSTRUCTIBILITY
    // -------------------------------------------------------------------------------------------
    {
        let mut recon_ok = true;
        let mut mismatch_at: Option<usize> = None;
        for (i, (orig, merged)) in chain_streams_orig
            .iter()
            .zip(chain_streams.iter())
            .enumerate()
        {
            let decoded = decode(&chain_table, merged);
            if &decoded != orig {
                recon_ok = false;
                mismatch_at = Some(i);
                break;
            }
        }
        assert!(
            recon_ok,
            "B4: every def-use chain occurrence must decode back to its exact atom sequence \
             (first mismatch at index {mismatch_at:?})"
        );

        // Can-fire: a deliberately corrupted merge table must break round-trip exactness.
        let mut corrupt_demo: Option<(bool, u32)> = None;
        if let Some(&(_, last_id, _)) = chain_merges.last() {
            if atom_labels.len() > 1 {
                let mut corrupted = SymTable {
                    labels: chain_table.labels.clone(),
                    parts: chain_table.parts.clone(),
                };
                let (l, r) = corrupted.parts[last_id as usize].expect("a merge id has parts");
                // Guaranteed different from `l` since atom_labels.len() > 1.
                let bogus_left = (l + 1) % atom_labels.len() as u32;
                corrupted.parts[last_id as usize] = Some((bogus_left, r));
                if let Some((orig_bad, merged_bad)) = chain_streams_orig
                    .iter()
                    .zip(chain_streams.iter())
                    .find(|(_, m)| m.contains(&last_id))
                {
                    let bad_decoded = decode(&corrupted, merged_bad);
                    let broke = &bad_decoded != orig_bad;
                    assert!(
                        broke,
                        "B4 can-fire: corrupting a used merge's left part must break round-trip \
                         exactness for at least one occurrence that used it"
                    );
                    corrupt_demo = Some((broke, last_id));
                }
            }
        }
        let corrupt_demo_text = match corrupt_demo {
            Some((broke, id)) => format!("merge id {id} corrupted, broke={broke}"),
            None => "not applicable (single-atom vocabulary — no pair to corrupt)".to_string(),
        };
        pass += 1;
        println!(
            "B4 PASS  {} occurrences decode byte-exact (well, atom-exact) via the B2 merge \
             table; corruption demo ({corrupt_demo_text}) confirms the check is falsifiable, \
             not vacuous",
            chain_streams_orig.len(),
        );
    }

    // -------------------------------------------------------------------------------------------
    // B5 LOCO FIT
    // -------------------------------------------------------------------------------------------
    {
        let (mut fit_pairs, mut fit_triples, mut fit_quads, mut fit_none) =
            (0usize, 0usize, 0usize, 0usize);
        let mut none_examples: Vec<u32> = Vec::new();
        for &(_, id, _count) in &chain_merges {
            let n_atoms = atoms_of(&chain_table, id).len();
            let immediates_needed = n_atoms.saturating_sub(1);
            if immediates_needed <= LaneShape::Pairs.max_immediates() {
                fit_pairs += 1;
            } else if immediates_needed <= LaneShape::Triples.max_immediates() {
                fit_triples += 1;
            } else if immediates_needed <= LaneShape::Quads.max_immediates() {
                fit_quads += 1;
            } else {
                fit_none += 1;
                none_examples.push(id);
            }
        }

        // Nodes-needed accounting: total def-use chain occurrences per episode, as the call
        // count a real emission would need, checked against BODY_BYTES=360 at each shape's
        // calls/node.
        let mut calls_per_episode: BTreeMap<&(String, String), usize> = BTreeMap::new();
        for c in &chains {
            *calls_per_episode.entry(&c.ep_key).or_insert(0) += 1;
        }
        let mut counts: Vec<usize> = calls_per_episode.values().copied().collect();
        counts.sort_unstable();
        assert!(
            !counts.is_empty(),
            "B5: at least one episode must contain chain occurrences"
        );
        let median_calls = counts[counts.len() / 2];
        let max_calls = *counts.last().expect("non-empty");

        pass += 1;
        println!(
            "B5 PASS  of {} learned macros: fits-Pairs(<=1 imm)={fit_pairs} \
             fits-Triples(<=2 imm)={fit_triples} fits-Quads(<=3 imm)={fit_quads} \
             fits-NONE(>3 imm)={fit_none} {:?} — this corpus's def-use chains are fixed at \
             length 3 by construction (extract_chains), so no macro here can exceed 3 atoms / \
             2 immediates; a fit-NONE macro would require a longer chain than this pass-1 \
             extractor produces. That is a disclosed methodology ceiling, not a hidden result.",
            chain_merges.len(),
            none_examples
        );
        for shape in [LaneShape::Pairs, LaneShape::Triples, LaneShape::Quads] {
            let nodes_median = median_calls.div_ceil(shape.calls_per_node());
            let nodes_max = max_calls.div_ceil(shape.calls_per_node());
            println!(
                "         {shape:?}: {} calls/node (BODY_BYTES=360) -> median episode ({median_calls} \
                 chain calls) needs {nodes_median} node(s); busiest episode ({max_calls} chain \
                 calls) needs {nodes_max} node(s)",
                shape.calls_per_node()
            );
        }
    }

    // -------------------------------------------------------------------------------------------
    // B6 CANDIDATE RANKING (NOT admission)
    // -------------------------------------------------------------------------------------------
    let groups: BTreeMap<Vec<u32>, Vec<usize>> = {
        let mut g: BTreeMap<Vec<u32>, Vec<usize>> = BTreeMap::new();
        for (i, s) in chain_streams.iter().enumerate() {
            g.entry(s.clone()).or_default().push(i);
        }
        g
    };
    {
        let mut group_stats: Vec<(Vec<u32>, usize, usize, f32)> = Vec::new();
        for (key, idxs) in &groups {
            let eps_set: HashSet<&(String, String)> =
                idxs.iter().map(|&i| &chains[i].ep_key).collect();
            let mut truth = TruthValue::new(0.5, 0.05);
            let mut stamp = Stamp(0);
            for ek in &eps_set {
                let ev = Stamp::source(episode_index[*ek]);
                if stamp.disjoint(ev) {
                    truth = truth.revise(&TruthValue::new(1.0, 0.9));
                    stamp = stamp.union(ev);
                }
                // else: mod-64 CHOICE-dropped, same conservatism as E4/C1.
            }
            group_stats.push((key.clone(), idxs.len(), eps_set.len(), truth.expectation()));
        }
        assert!(
            group_stats.len() >= 2,
            "B6: at least two candidate groups are needed to compare rankings"
        );

        let mut by_raw = group_stats.clone();
        by_raw.sort_by_key(|(_, raw, _, _)| std::cmp::Reverse(*raw));
        let mut by_truth = group_stats.clone();
        by_truth.sort_by(|a, b| b.3.partial_cmp(&a.3).expect("finite expectation"));
        let truth_rank_of: HashMap<Vec<u32>, usize> = by_truth
            .iter()
            .enumerate()
            .map(|(r, (k, _, _, _))| (k.clone(), r))
            .collect();

        let mut flip_pair: Option<(usize, usize)> = None;
        let mut agree_pair: Option<(usize, usize)> = None;
        for i in 0..by_raw.len() {
            for j in (i + 1)..by_raw.len() {
                let ki = &by_raw[i].0;
                let kj = &by_raw[j].0;
                if flip_pair.is_none() && truth_rank_of[kj] < truth_rank_of[ki] {
                    flip_pair = Some((i, j));
                }
                if agree_pair.is_none() && truth_rank_of[ki] < truth_rank_of[kj] {
                    agree_pair = Some((i, j));
                }
                if flip_pair.is_some() && agree_pair.is_some() {
                    break;
                }
            }
            if flip_pair.is_some() && agree_pair.is_some() {
                break;
            }
        }
        assert!(
            flip_pair.is_some(),
            "B6 ranking-sensitivity can-fire: some pair of candidate groups must invert order \
             between raw-occurrence ranking and disjoint-evidence (revise+Stamp) ranking — a \
             macro that recurs often within ONE episode is weaker validity evidence than one \
             recurring across many disjoint episodes"
        );
        assert!(
            agree_pair.is_some(),
            "B6 can-stay-silent: some pair must still agree in relative order between both \
             rankings — the two rankings are not universally opposed"
        );
        let fmt_pair = |p: Option<(usize, usize)>| match p {
            Some((a, b)) => format!("({a}, {b})"),
            None => "none".to_string(),
        };
        let flip_pair_s = fmt_pair(flip_pair);
        let agree_pair_s = fmt_pair(agree_pair);

        pass += 1;
        println!(
            "B6 PASS  {} candidate macro groups ranked. NOT admission: this ranks candidates \
             only; freezing/admitting/promoting/gating a macro is MUL's and the autopoiesis \
             triangle's job, never this probe's. Ranking-sensitivity: raw-count top group vs \
             disjoint-evidence top group differ at by_raw indices {flip_pair_s}; some pair \
             still agrees at {agree_pair_s} — the by-raw-count and by-disjoint-evidence \
             rankings are DIFFERENT lenses on the same candidates, not the same ranking twice",
            group_stats.len()
        );
    }

    // -------------------------------------------------------------------------------------------
    // INV1 — stride-as-role (HighHeelBGZ SpiralAddress precedent)
    // -------------------------------------------------------------------------------------------
    {
        let mut role_classes: BTreeMap<Vec<u32>, usize> = BTreeMap::new();
        for &(_, id, _count) in &chain_merges {
            let mut multiset = atoms_of(&chain_table, id);
            multiset.sort_unstable();
            *role_classes.entry(multiset).or_insert(0) += 1;
        }
        assert!(
            !chain_merges.is_empty(),
            "INV1: at least one merge is needed to measure role reuse"
        );
        assert!(
            role_classes.len() < chain_merges.len(),
            "INV1: SOME role reuse must exist (role_classes {} < merges {})",
            role_classes.len(),
            chain_merges.len()
        );
        let largest = role_classes.values().copied().max().unwrap_or(0);
        assert!(
            largest > 1,
            "INV1 non-trivial-reuse can-fire: the largest role class must hold more than 1 \
             macro (got {largest}) — a corpus where every macro is a unique role would fail \
             this"
        );
        pass += 1;
        println!(
            "INV1 PASS  exploratory (HighHeelBGZ stride-as-role precedent): {} learned macros \
             collapse into {} distinct atom-multiset role classes (largest class holds {largest} \
             macros) — grouping by SOURCE opcode composition, like SpiralAddress's stride, \
             produces real role reuse on this corpus",
            chain_merges.len(),
            role_classes.len()
        );
    }

    // -------------------------------------------------------------------------------------------
    // INV2 — escalation cascade (bgz17 LayeredScope::search precedent)
    // -------------------------------------------------------------------------------------------
    {
        let chain_sig_counts: BTreeMap<(u32, u32, u32), usize> = {
            let mut m = BTreeMap::new();
            for c in &chains {
                *m.entry((c.x_sym, c.y_sym, c.z_sym)).or_insert(0usize) += 1;
            }
            m
        };
        let (&top_sig, _) = chain_sig_counts
            .iter()
            .max_by_key(|(_, n)| **n)
            .expect("INV2: at least one chain signature must exist");
        let mut target_multiset = vec![top_sig.0, top_sig.1, top_sig.2];
        target_multiset.sort_unstable();

        let mut pruned_at_cheap = 0usize;
        let mut survivors: Vec<&WinOcc> = Vec::new();
        for w in &windows {
            let mut m = w.syms.to_vec();
            m.sort_unstable();
            if m == target_multiset {
                survivors.push(w);
            } else {
                pruned_at_cheap += 1;
            }
        }

        let mut chained = 0usize;
        for w in &survivors {
            let ep = &eps[&w.ep_key];
            let empty: Vec<u64> = Vec::new();
            let out0 = ep.outs.get(&w.sites[0]).unwrap_or(&empty);
            let in1 = ep.ins.get(&w.sites[1]).unwrap_or(&empty);
            let out1 = ep.outs.get(&w.sites[1]).unwrap_or(&empty);
            let in2 = ep.ins.get(&w.sites[2]).unwrap_or(&empty);
            let link01 = out0.iter().any(|v| in1.contains(v));
            let link12 = out1.iter().any(|v| in2.contains(v));
            if link01 && link12 {
                chained += 1;
            }
        }
        let not_chained = survivors.len() - chained;

        assert!(
            pruned_at_cheap > 0,
            "INV2 can-fire: the cheap opcode-multiset check must prune SOME linear-window \
             candidates before the expensive exact def-use walk runs"
        );
        assert!(
            !survivors.is_empty(),
            "INV2 can-stay-silent: the cheap check must not prune EVERY candidate — some must \
             survive to the expensive stage"
        );
        let prune_frac = 100.0 * pruned_at_cheap as f64 / windows.len().max(1) as f64;

        pass += 1;
        println!(
            "INV2 PASS  exploratory (bgz17 LayeredScope scent-prune-then-escalate precedent): \
             target signature {top_sig:?}; of {} linear-window candidates, cheap opcode-multiset \
             check prunes {pruned_at_cheap} ({prune_frac:.1}%) before the expensive exact def-use \
             walk runs on the {} survivors ({chained} truly chained, {not_chained} multiset-matched \
             but NOT dataflow-chained — reordered/coincidental) — a real prune, not a null-op",
            windows.len(),
            survivors.len()
        );
    }

    // -------------------------------------------------------------------------------------------
    // INV3 — shared-palette amortization (BGZ-HHTL-D precedent); reuses B6's `groups`, per spec
    // -------------------------------------------------------------------------------------------
    {
        let mut per_group_ep_count: Vec<(usize, usize)> = groups
            .values()
            .map(|idxs| {
                let eps_set: HashSet<&(String, String)> =
                    idxs.iter().map(|&i| &chains[i].ep_key).collect();
                (idxs.len(), eps_set.len())
            })
            .collect();
        per_group_ep_count.sort_by_key(|(raw, _)| std::cmp::Reverse(*raw));

        let shared = per_group_ep_count
            .iter()
            .find(|(_, ep_count)| *ep_count > 1);
        let unshared = per_group_ep_count
            .iter()
            .find(|(_, ep_count)| *ep_count == 1);
        assert!(
            shared.is_some(),
            "INV3 can-fire: at least one converged macro group must be shared across more than \
             one episode (amortization actually happens)"
        );
        assert!(
            unshared.is_some(),
            "INV3 can-stay-silent: at least one macro group must stay unshared (exactly one \
             episode) — proving the check isn't vacuously true for every macro"
        );

        let top5: Vec<usize> = per_group_ep_count.iter().take(5).map(|(_, e)| *e).collect();
        let fmt_hit = |h: Option<&(usize, usize)>| match h {
            Some((raw, ep)) => format!("(occurrences={raw}, episodes={ep})"),
            None => "none".to_string(),
        };
        let shared_s = fmt_hit(shared);
        let unshared_s = fmt_hit(unshared);
        pass += 1;
        println!(
            "INV3 PASS  exploratory (BGZ-HHTL-D shared-palette precedent): {} converged macro \
             groups measured; top-5 by occurrence carry episodes-per-mint {:?}; shared example \
             {shared_s}, unshared example {unshared_s} — some macros amortize across \
             episodes (one mint reused), some stay episode-local (would risk a duplicate mint if \
             minted per-instance)",
            per_group_ep_count.len(),
            top5,
        );
    }

    // -------------------------------------------------------------------------------------------
    // B7 FENCES
    // -------------------------------------------------------------------------------------------
    {
        let binaries: HashSet<&String> = eps.keys().map(|(b, _)| b).collect();
        assert_eq!(
            binaries.len(),
            2,
            "B7 anchor: the corpus this probe reads is still 2 binaries (recomputed live from \
             the same rows, not a memorized constant)"
        );
        pass += 1;
        println!(
            "B7 PASS  fences: no mint actually performed (no classid minted anywhere, no \
             vocabulary table shipped, no learner subsystem, no write to any \
             ValueTenant/MUL/triangle type); ogar-loco, bgz17's cascade shape, and ruff_r2il are \
             MIRRORED/CITED, never imported; corpus = {} binaries / {} episodes (measured live), \
             pass-1 seven-opcode convention — nothing here is a claim about x86-64 in general; V4 \
             is physically V3 (no ENVELOPE_LAYOUT_VERSION bump) and this probe writes no bytes at \
             rest; this probe generates ranked CANDIDATES ONLY — admission is MUL's and the \
             autopoiesis triangle's, not this probe's, and no vocabulary here \
             (freeze/admit/promote/gate) is used for what B6 does",
            binaries.len(),
            eps.len()
        );
    }

    println!(
        "\n{pass}/10 gates green — R2IL def-use-chain BPE candidate macros measured against the \
         real ogar-loco call geometry, plus three exploratory cross-checks against shipped \
         precedent (HighHeelBGZ / bgz17 HHTL / BGZ-HHTL-D)."
    );
    ExitCode::SUCCESS
}
