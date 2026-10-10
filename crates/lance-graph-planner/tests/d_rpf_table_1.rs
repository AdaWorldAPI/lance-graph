//! D-RPF-TABLE-1: does the confidence-inert `NarsTables::build(1)` fast path
//! earn its semantic cost?
//!
//! Three of the four axes live here (numeric surface, replay drift,
//! downstream decisions); the fourth, throughput, is the
//! `d_rpf_table_1_bench` example, because a timing is not a test assertion.
//!
//! Laws compared, all through their real implementations:
//! - A   `CausalEdge64::revision` (confidence-aware ISA revision)
//! - Bn  `NarsTables::build(n).revise` for n in {1, 2, 4, 8, 16}
//!
//! The replay for A cannot go through `replay_step`, which takes a table. It
//! goes through `step_with`, which is `replay_step`'s body with the truth law
//! as a parameter. `step_with_a_table_is_replay_step` proves the mirror is
//! bit-exact against the real `replay_step` for every table resolution, so a
//! difference between A and Bn can only come from the law.
//!
//! Report: `.claude/board/entries/2026-10-09-d-rpf-table-1-fast-path-cost.md`.

use causal_edge::edge::InferenceType;
use causal_edge::isa::IsaFault;
use causal_edge::tables::{unpack_c, unpack_f, NarsTables};
use causal_edge::{CausalEdge64, CausalMask, PlasticityState};
use lance_graph_planner::chain_counterfactual::{
    counterfactual_replay, cut_step, CutContext, DEFAULT_FREQUENCY_BAR,
};
use lance_graph_planner::chain_replay::{replay_step, ChainStep, ComposeTables};
use lance_graph_planner::pearl::Reaction;

const LEVELS: [usize; 5] = [1, 2, 4, 8, 16];

/// A revision law over the u8 truth grid.
type Law<'a> = Box<dyn Fn(u8, u8, u8, u8) -> (u8, u8) + 'a>;

#[allow(deprecated)] // v2 `pack` ignores temporal; not under test
fn edge(f: u8, c: u8, op: InferenceType) -> CausalEdge64 {
    CausalEdge64::pack(
        1,
        2,
        3,
        f,
        c,
        CausalMask::SPO,
        0,
        op,
        PlasticityState::ALL_HOT,
        0,
    )
}

fn law_a() -> Law<'static> {
    Box::new(|f1, c1, f2, c2| {
        let r =
            edge(f1, c1, InferenceType::Revision).revision(edge(f2, c2, InferenceType::Revision));
        (r.frequency_u8(), r.confidence_u8())
    })
}

fn law_table(t: &NarsTables) -> Law<'_> {
    Box::new(move |f1, c1, f2, c2| {
        let p = t.revise(f1, c1, f2, c2);
        (unpack_f(p), unpack_c(p))
    })
}

fn tables() -> Vec<NarsTables> {
    LEVELS.iter().map(|&n| NarsTables::build(n)).collect()
}

// ─── Axis 1: the numeric surface ────────────────────────────────────────

/// The R3 domain of D-RPF-CONF-0, for comparability.
const FS: [u8; 5] = [0, 51, 128, 204, 255];
const CELLS: u64 = (FS.len() * FS.len() * 256 * 256) as u64;

fn is_boundary(c: u8) -> bool {
    c == 0 || c >= 254
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Surface {
    differ: u64,
    differ_broad: u64,
    max_df: u8,
    max_dc: u8,
    max_df_broad: u8,
    max_dc_broad: u8,
    sum_df: u64,
    sum_dc: u64,
    /// p50 / p95 / p99 of |dF| and |dC| over all cells.
    pct_df: [u8; 3],
    pct_dc: [u8; 3],
    distinct_c: usize,
}

fn percentiles(hist: &[u64; 256]) -> [u8; 3] {
    let mut out = [0u8; 3];
    for (k, q) in [0.50f64, 0.95, 0.99].iter().enumerate() {
        let want = (q * CELLS as f64).ceil() as u64;
        let mut acc = 0;
        for (v, n) in hist.iter().enumerate() {
            acc += n;
            if acc >= want {
                out[k] = v as u8;
                break;
            }
        }
    }
    out
}

fn surface(law: &Law<'_>, reference: &Law<'_>) -> Surface {
    let mut s = Surface {
        differ: 0,
        differ_broad: 0,
        max_df: 0,
        max_dc: 0,
        max_df_broad: 0,
        max_dc_broad: 0,
        sum_df: 0,
        sum_dc: 0,
        pct_df: [0; 3],
        pct_dc: [0; 3],
        distinct_c: 0,
    };
    let (mut hf, mut hc) = ([0u64; 256], [0u64; 256]);
    let mut seen_c = [false; 256];
    for &f1 in &FS {
        for &f2 in &FS {
            for c1 in 0..=255u8 {
                for c2 in 0..=255u8 {
                    let (fa, ca) = reference(f1, c1, f2, c2);
                    let (fb, cb) = law(f1, c1, f2, c2);
                    seen_c[cb as usize] = true;
                    let (df, dc) = (fa.abs_diff(fb), ca.abs_diff(cb));
                    hf[df as usize] += 1;
                    hc[dc as usize] += 1;
                    s.sum_df += df as u64;
                    s.sum_dc += dc as u64;
                    s.max_df = s.max_df.max(df);
                    s.max_dc = s.max_dc.max(dc);
                    if df != 0 || dc != 0 {
                        s.differ += 1;
                        if !is_boundary(c1) && !is_boundary(c2) {
                            s.differ_broad += 1;
                            s.max_df_broad = s.max_df_broad.max(df);
                            s.max_dc_broad = s.max_dc_broad.max(dc);
                        }
                    }
                }
            }
        }
    }
    s.pct_df = percentiles(&hf);
    s.pct_dc = percentiles(&hc);
    s.distinct_c = seen_c.iter().filter(|x| **x).count();
    s
}

fn all_surfaces() -> Vec<Surface> {
    let a = law_a();
    let ts = tables();
    ts.iter().map(|t| surface(&law_table(t), &a)).collect()
}

#[test]
fn axis1_report() {
    let ts = tables();
    for (n, t) in LEVELS.iter().zip(&ts) {
        println!(
            "B{n:<2} tables={:>3} bytes={:>9}",
            t.revision.len(),
            t.byte_size()
        );
    }
    for (n, s) in LEVELS.iter().zip(all_surfaces()) {
        println!(
            "B{n:<2} differ={:>7} broad={:>7} max=({},{}) broad_max=({},{}) \
             mean=({:.3},{:.3}) p50/95/99 dF={:?} dC={:?} distinct_c={}",
            s.differ,
            s.differ_broad,
            s.max_df,
            s.max_dc,
            s.max_df_broad,
            s.max_dc_broad,
            s.sum_df as f64 / CELLS as f64,
            s.sum_dc as f64 / CELLS as f64,
            s.pct_df,
            s.pct_dc,
            s.distinct_c
        );
    }
}

// ─── Axes 2 and 3: replay drift and downstream decisions ────────────────

/// `replay_step`'s body with the truth law as a parameter: forward for the
/// palettes, mask, direction and plasticity, then the law's truth written
/// back. `step_with_a_table_is_replay_step` pins it against the real one.
fn step_with(
    law: &Law<'_>,
    running: CausalEdge64,
    weight: CausalEdge64,
    compose: ComposeTables<'_>,
) -> Result<CausalEdge64, IsaFault> {
    let (f, c) = law(
        running.frequency_u8(),
        running.confidence_u8(),
        weight.frequency_u8(),
        weight.confidence_u8(),
    );
    let mut out = running.forward(weight, compose.s, compose.p, compose.o)?;
    out.set_frequency_u8(f);
    out.set_confidence_u8(c);
    Ok(out)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn byte(&mut self, lo: u8, hi: u8) -> u8 {
        lo + (self.next() % (hi as u64 - lo as u64 + 1)) as u8
    }
}

fn compose_tables() -> [Box<[u8; 256 * 256]>; 3] {
    let mut r = Lcg(0x9E37_79B9_7F4A_7C15);
    core::array::from_fn(|_| {
        let mut t = Box::new([0u8; 256 * 256]);
        for v in t.iter_mut() {
            *v = r.next() as u8;
        }
        t
    })
}

/// The ops a recorded weight can carry that the ISA executes.
const OPS: [InferenceType; 4] = [
    InferenceType::Deduction,
    InferenceType::Induction,
    InferenceType::Abduction,
    InferenceType::Revision,
];

/// Two workloads. `Uniform` spans the whole truth grid. `W0` is the
/// distribution `dcr_w0_replay_budget` measured the replay kernel on
/// (f in 128..=255, c in 128..=227), i.e. the workload the fast path was
/// sized against.
#[derive(Debug, Clone, Copy)]
enum Corpus {
    Uniform,
    W0,
}

struct Chain {
    seed: CausalEdge64,
    steps: Vec<ChainStep>,
}

fn corpus(kind: Corpus, chains: usize, len: usize) -> Vec<Chain> {
    let mut r = Lcg(match kind {
        Corpus::Uniform => 0x0D17_AB1E_0000_0001,
        Corpus::W0 => 0x051E_D270_B5A1_11E5,
    });
    let pick = |r: &mut Lcg| match kind {
        Corpus::Uniform => (r.byte(0, 255), r.byte(0, 255)),
        Corpus::W0 => (r.byte(128, 255), r.byte(128, 227)),
    };
    (0..chains)
        .map(|_| {
            let (f, c) = pick(&mut r);
            let seed = edge(f, c, InferenceType::Revision);
            let steps = (0..len)
                .map(|_| {
                    let (f, c) = pick(&mut r);
                    let op = OPS[(r.next() % 4) as usize];
                    (0x90u8, edge(f, c, op))
                })
                .collect();
            Chain { seed, steps }
        })
        .collect()
}

/// Every running truth after each step.
fn trajectory(law: &Law<'_>, ch: &Chain, compose: ComposeTables<'_>) -> Vec<CausalEdge64> {
    let mut running = ch.seed;
    ch.steps
        .iter()
        .map(|&(_, w)| {
            running = step_with(law, running, w, compose).expect("ops are executable");
            running
        })
        .collect()
}

fn compose_of(t: &[Box<[u8; 256 * 256]>; 3]) -> ComposeTables<'_> {
    ComposeTables {
        s: &t[0],
        p: &t[1],
        o: &t[2],
    }
}

/// The mirror is the real replay step. Without this every A-vs-Bn replay
/// number below could be a difference between two step functions.
#[test]
fn step_with_a_table_is_replay_step() {
    let c = compose_tables();
    let compose = compose_of(&c);
    let chains = corpus(Corpus::Uniform, 64, 64);
    for t in tables() {
        let law = law_table(&t);
        for ch in &chains {
            let (mut mine, mut real) = (ch.seed, ch.seed);
            for &(_, w) in &ch.steps {
                mine = step_with(&law, mine, w, compose).unwrap();
                real = replay_step(real, w, &t, compose).unwrap();
                assert_eq!(mine, real, "c_levels={}", t.c_levels);
            }
        }
    }
}

const CHECKPOINTS: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

#[derive(Debug, Clone, PartialEq)]
struct Drift {
    /// Per checkpoint: (chains whose truth differs from A, mean |dF|, mean |dC|).
    at: Vec<(usize, f64, f64)>,
    /// Chains whose truth first differs from A at step 1.
    first_at_step1: usize,
    /// Chains that never differ within 64 steps.
    never: usize,
    /// Median first-divergence step over the chains that diverge.
    median_first: usize,
    /// Fields other than F/C that differ from A at any step (palettes, mask,
    /// direction, plasticity, Epi5). Must be zero: the law touches only F/C.
    other_field_diffs: usize,
}

const FC_MASK: u64 = 0xFFFF << 24;

fn drift(law: &Law<'_>, chains: &[Chain], compose: ComposeTables<'_>) -> Drift {
    let a = law_a();
    let mut at = vec![(0usize, 0f64, 0f64); CHECKPOINTS.len()];
    let mut firsts = Vec::new();
    let (mut first_at_step1, mut never, mut other) = (0, 0, 0);
    for ch in chains {
        let ta = trajectory(&a, ch, compose);
        let tb = trajectory(law, ch, compose);
        let first = ta.iter().zip(&tb).position(|(x, y)| x != y);
        match first {
            None => never += 1,
            Some(0) => {
                first_at_step1 += 1;
                firsts.push(1);
            }
            Some(i) => firsts.push(i + 1),
        }
        other += ta
            .iter()
            .zip(&tb)
            .filter(|(x, y)| (x.0 & !FC_MASK) != (y.0 & !FC_MASK))
            .count();
        for (k, &cp) in CHECKPOINTS.iter().enumerate() {
            let (x, y) = (ta[cp - 1], tb[cp - 1]);
            if x != y {
                at[k].0 += 1;
            }
            at[k].1 += x.frequency_u8().abs_diff(y.frequency_u8()) as f64;
            at[k].2 += x.confidence_u8().abs_diff(y.confidence_u8()) as f64;
        }
    }
    for e in &mut at {
        e.1 /= chains.len() as f64;
        e.2 /= chains.len() as f64;
    }
    firsts.sort_unstable();
    Drift {
        at,
        first_at_step1,
        never,
        median_first: firsts.get(firsts.len() / 2).copied().unwrap_or(0),
        other_field_diffs: other,
    }
}

#[test]
fn axis2_report() {
    let c = compose_tables();
    let compose = compose_of(&c);
    let ts = tables();
    for kind in [Corpus::Uniform, Corpus::W0] {
        let chains = corpus(kind, 256, 64);
        for (n, t) in LEVELS.iter().zip(&ts) {
            let d = drift(&law_table(t), &chains, compose);
            println!(
                "{kind:?} B{n:<2} first@1={} never={} median_first={} other={}",
                d.first_at_step1, d.never, d.median_first, d.other_field_diffs
            );
            for (cp, (k, df, dc)) in CHECKPOINTS.iter().zip(&d.at) {
                println!("    step {cp:>2}: differ {k:>3}/256  mean dF {df:6.2}  mean dC {dc:6.2}");
            }
        }
    }
}

/// How often A itself reaches c = 255 (after which its frequency is frozen:
/// `w = MAX` dominates every later revision) and how often a step hits the
/// singular cell (255,255), which is G1 and collapses to (0,0).
fn a_saturation(chains: &[Chain], compose: ComposeTables<'_>) -> (Vec<usize>, usize) {
    let a = law_a();
    let mut sat = vec![0usize; CHECKPOINTS.len()];
    let mut singular = 0;
    for ch in chains {
        let t = trajectory(&a, ch, compose);
        let mut prev = ch.seed;
        for (i, (e, &(_, w))) in t.iter().zip(&ch.steps).enumerate() {
            if prev.confidence_u8() == 255 && w.confidence_u8() == 255 {
                singular += 1;
            }
            prev = *e;
            if let Some(k) = CHECKPOINTS.iter().position(|&cp| cp == i + 1) {
                if e.confidence_u8() == 255 {
                    sat[k] += 1;
                }
            }
        }
    }
    (sat, singular)
}

#[test]
fn axis2_a_saturation_report() {
    let c = compose_tables();
    let compose = compose_of(&c);
    for kind in [Corpus::Uniform, Corpus::W0] {
        let (sat, singular) = a_saturation(&corpus(kind, 256, 64), compose);
        println!("{kind:?}: A at c=255 by checkpoint {sat:?} of 256; singular steps {singular}");
    }
}

// ─── Axis 3: decisions ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    LoadBearing,
    TruthOnly,
    Inert,
}

fn class(r: Reaction) -> Class {
    match r {
        Reaction::LoadBearing { .. } => Class::LoadBearing,
        Reaction::TruthOnly { .. } => Class::TruthOnly,
        Reaction::Inert { .. } => Class::Inert,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Decision {
    factual: bool,
    cut: bool,
    load_bearing: bool,
    reaction: Reaction,
    /// The two arms' terminal frequencies are equal, so a TruthOnly can only
    /// have come from confidence.
    freq_equal: bool,
}

fn terminal(
    law: &Law<'_>,
    seed: CausalEdge64,
    steps: &[ChainStep],
    compose: ComposeTables<'_>,
) -> (u8, u8) {
    let ch = Chain {
        seed,
        steps: steps.to_vec(),
    };
    trajectory(law, &ch, compose)
        .last()
        .map_or((0, 0), |e| (e.frequency_u8(), e.confidence_u8()))
}

/// The cut decision exactly as `chain_counterfactual` + `pearl::reason`
/// make it: verdict = terminal frequency >= the bar (an empty arm is
/// Inconsistent and reads (0,0)); load-bearing = the verdicts differ;
/// reaction = `Reaction::classify` on the two terminal truths.
fn decide(law: &Law<'_>, ch: &Chain, index: usize, compose: ComposeTables<'_>) -> Decision {
    let cut = cut_step(&ch.steps, index).expect("in range");
    let f = terminal(law, ch.seed, &ch.steps, compose);
    let c = if cut.is_empty() {
        (0, 0)
    } else {
        terminal(law, ch.seed, &cut, compose)
    };
    let factual = !ch.steps.is_empty() && f.0 >= DEFAULT_FREQUENCY_BAR;
    let cut_ok = !cut.is_empty() && c.0 >= DEFAULT_FREQUENCY_BAR;
    let load_bearing = factual != cut_ok;
    Decision {
        factual,
        cut: cut_ok,
        load_bearing,
        reaction: Reaction::classify(load_bearing, f, c),
        freq_equal: f.0 == c.0,
    }
}

/// The decision mirror is the real counterfactual path for every table.
#[test]
fn decide_with_a_table_is_counterfactual_replay() {
    let c = compose_tables();
    let compose = compose_of(&c);
    for t in tables() {
        let law = law_table(&t);
        for ch in corpus(Corpus::Uniform, 64, 8).iter() {
            for i in 0..ch.steps.len() {
                let d = decide(&law, ch, i, compose);
                let cf = counterfactual_replay(
                    &ch.steps,
                    i,
                    ch.seed,
                    CutContext {
                        tables: &t,
                        compose,
                        owner: 3,
                        base_seq: 100,
                        bar: DEFAULT_FREQUENCY_BAR,
                    },
                )
                .unwrap()
                .unwrap();
                assert_eq!(d.load_bearing, cf.role.is_load_bearing());
                let tf = cf
                    .factual
                    .last()
                    .map(|r| (r.edge.frequency_u8(), r.edge.confidence_u8()));
                let tc = cf
                    .counterfactual
                    .last()
                    .map(|r| (r.edge.frequency_u8(), r.edge.confidence_u8()));
                assert_eq!(
                    d.reaction,
                    Reaction::classify(
                        cf.role.is_load_bearing(),
                        tf.unwrap_or((0, 0)),
                        tc.unwrap_or((0, 0))
                    )
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Flips {
    cuts: usize,
    /// The chain's own verdict differs from A's.
    verdict: usize,
    load_bearing: usize,
    /// The Reaction variant differs from A's.
    reaction_class: usize,
    /// Of those, cuts where A says TruthOnly with equal frequencies (so the
    /// truth moved only in confidence) and the law says Inert.
    confidence_only: usize,
}

fn flips(law: &Law<'_>, chains: &[Chain], compose: ComposeTables<'_>) -> Flips {
    let a = law_a();
    let mut out = Flips::default();
    for ch in chains {
        for i in 0..ch.steps.len() {
            let (da, db) = (decide(&a, ch, i, compose), decide(law, ch, i, compose));
            out.cuts += 1;
            out.verdict += usize::from(da.factual != db.factual);
            out.load_bearing += usize::from(da.load_bearing != db.load_bearing);
            if class(da.reaction) != class(db.reaction) {
                out.reaction_class += 1;
                if class(da.reaction) == Class::TruthOnly
                    && da.freq_equal
                    && class(db.reaction) == Class::Inert
                {
                    out.confidence_only += 1;
                }
            }
        }
    }
    out
}

#[test]
fn axis3_report() {
    let c = compose_tables();
    let compose = compose_of(&c);
    let ts = tables();
    for kind in [Corpus::Uniform, Corpus::W0] {
        for len in [2usize, 8] {
            let chains = corpus(kind, 256, len);
            for (n, t) in LEVELS.iter().zip(&ts) {
                println!(
                    "{kind:?} len={len} B{n:<2} {:?}",
                    flips(&law_table(t), &chains, compose)
                );
            }
        }
    }
    // The repository's own counterfactual fixtures (chain_counterfactual tests).
    let lb: Vec<ChainStep> = vec![
        (0x91, edge(250, 250, InferenceType::Deduction)),
        (0x92, edge(40, 30, InferenceType::Deduction)),
    ];
    let red: Vec<ChainStep> = (0..4)
        .map(|i| (0x90 + i as u8, edge(250, 250, InferenceType::Deduction)))
        .collect();
    let seed = edge(200, 200, InferenceType::Deduction);
    let a = law_a();
    for (name, steps, cut) in [("load-bearing", &lb, 0usize), ("redundant", &red, 2)] {
        let ch = Chain {
            seed,
            steps: steps.clone(),
        };
        println!("{name}: A {:?}", decide(&a, &ch, cut, compose));
        for (n, t) in LEVELS.iter().zip(&ts) {
            println!(
                "{name}: B{n} {:?}",
                decide(&law_table(t), &ch, cut, compose)
            );
        }
    }
}

// ─── Pins ───────────────────────────────────────────────────────────────

/// Axis 1, per level: (revision tables, bytes, cells differing, broad cells
/// differing, broad max dF, broad max dC, distinct output c).
#[rustfmt::skip]
const AXIS1: [(usize, usize, u64, u64, u8, u8, usize); 5] = [
    (  1,   262_144, 1_636_550, 1_598_385, 128, 168,  1),
    (  4,   655_360, 1_634_496, 1_596_341, 127, 100,  3),
    ( 16, 2_228_224, 1_629_527, 1_591_392, 125,  55, 10),
    ( 64, 8_519_680, 1_616_783, 1_578_704, 121,  28, 32),
    (256, 33_685_504, 1_585_273, 1_547_346, 113,  14, 92),
];

#[test]
fn axis1_is_pinned() {
    let ts = tables();
    let got: Vec<_> = ts
        .iter()
        .zip(all_surfaces())
        .map(|(t, s)| {
            (
                t.revision.len(),
                t.byte_size(),
                s.differ,
                s.differ_broad,
                s.max_df_broad,
                s.max_dc_broad,
                s.distinct_c,
            )
        })
        .collect();
    assert_eq!(got, AXIS1.to_vec());
}

/// Error falls with resolution on the mean, but no level reaches A: even 16
/// buckets differ on 96.8% of cells and by over 100 frequency codes in the
/// broad domain. That is what "full precision" must not be read to mean.
#[test]
fn axis1_no_resolution_reaches_a() {
    for s in all_surfaces() {
        assert!(s.differ * 100 > CELLS * 96, "{s:?}");
        assert!(s.max_df_broad > 100, "{s:?}");
    }
}

/// One AXIS2 row.
type Axis2Row = (&'static str, usize, usize, usize, usize, usize, u64, u64);

/// Axis 2, per corpus and level: (chains diverging at step 1, chains never
/// diverging, median first divergence, non-F/C field diffs, mean |dF| and
/// mean |dC| at step 64 in hundredths). 256 chains x 64 steps.
#[rustfmt::skip]
const AXIS2: [Axis2Row; 10] = [
    ("Uniform",  1, 256, 0, 1, 0, 4386, 8325),
    ("Uniform",  2, 256, 0, 1, 0, 4275, 4557),
    ("Uniform",  4, 254, 0, 1, 0, 3953, 2414),
    ("Uniform",  8, 254, 0, 1, 0, 3377, 1252),
    ("Uniform", 16, 245, 0, 1, 0, 2805,  620),
    ("W0",       1, 256, 0, 1, 0, 1634, 8100),
    ("W0",       2, 255, 0, 1, 0, 1633, 3200),
    ("W0",       4, 252, 0, 1, 0, 1186, 1891),
    ("W0",       8, 251, 0, 1, 0,  656,  974),
    ("W0",      16, 248, 0, 1, 0,  363,  469),
];

#[test]
fn axis2_is_pinned() {
    let c = compose_tables();
    let compose = compose_of(&c);
    let ts = tables();
    let mut got = Vec::new();
    for (name, kind) in [("Uniform", Corpus::Uniform), ("W0", Corpus::W0)] {
        let chains = corpus(kind, 256, 64);
        for (n, t) in LEVELS.iter().zip(&ts) {
            let d = drift(&law_table(t), &chains, compose);
            let last = d.at[CHECKPOINTS.len() - 1];
            got.push((
                name,
                *n,
                d.first_at_step1,
                d.never,
                d.median_first,
                d.other_field_diffs,
                (last.1 * 100.0).round() as u64,
                (last.2 * 100.0).round() as u64,
            ));
        }
    }
    assert_eq!(got, AXIS2.to_vec());
}

/// The law touches only F/C: palettes, mask, direction, plasticity and the
/// epistemic bits agree with A at every step of every chain. So Epi5 and the
/// Pearl operator selection (bits 40..42) cannot be moved by the table.
#[test]
fn axis2_only_truth_bytes_differ() {
    let c = compose_tables();
    let compose = compose_of(&c);
    let ts = tables();
    let chains = corpus(Corpus::Uniform, 64, 64);
    for t in &ts {
        assert_eq!(drift(&law_table(t), &chains, compose).other_field_diffs, 0);
    }
}

/// A itself saturates on the uniform corpus: 43 of 256 chains sit at c = 255
/// by step 64 (frequency frozen from then on) and 13 steps hit the G1 cell.
/// The W0 corpus never reaches it.
#[test]
fn axis2_a_saturation_is_pinned() {
    let c = compose_tables();
    let compose = compose_of(&c);
    assert_eq!(
        a_saturation(&corpus(Corpus::Uniform, 256, 64), compose),
        (vec![1, 2, 6, 7, 12, 24, 43], 13)
    );
    assert_eq!(
        a_saturation(&corpus(Corpus::W0, 256, 64), compose),
        (vec![0; 7], 0)
    );
}

/// Axis 3, per corpus, chain length and level: decision flips against A over
/// every cut of 256 chains.
#[rustfmt::skip]
const AXIS3: [(&str, usize, usize, Flips); 20] = [
    ("Uniform", 2,  1, Flips { cuts: 512, verdict: 104, load_bearing: 125, reaction_class: 156, confidence_only: 1 }),
    ("Uniform", 2,  2, Flips { cuts: 512, verdict: 66, load_bearing: 68, reaction_class: 101, confidence_only: 3 }),
    ("Uniform", 2,  4, Flips { cuts: 512, verdict: 32, load_bearing: 43, reaction_class: 84, confidence_only: 1 }),
    ("Uniform", 2,  8, Flips { cuts: 512, verdict: 16, load_bearing: 21, reaction_class: 58, confidence_only: 2 }),
    ("Uniform", 2, 16, Flips { cuts: 512, verdict: 6, load_bearing: 9, reaction_class: 35, confidence_only: 2 }),
    ("Uniform", 8,  1, Flips { cuts: 2048, verdict: 776, load_bearing: 244, reaction_class: 905, confidence_only: 16 }),
    ("Uniform", 8,  2, Flips { cuts: 2048, verdict: 552, load_bearing: 205, reaction_class: 751, confidence_only: 13 }),
    ("Uniform", 8,  4, Flips { cuts: 2048, verdict: 368, load_bearing: 183, reaction_class: 646, confidence_only: 13 }),
    ("Uniform", 8,  8, Flips { cuts: 2048, verdict: 168, load_bearing: 105, reaction_class: 502, confidence_only: 13 }),
    ("Uniform", 8, 16, Flips { cuts: 2048, verdict: 96, load_bearing: 77, reaction_class: 426, confidence_only: 15 }),
    ("W0", 2,  1, Flips { cuts: 512, verdict: 0, load_bearing: 0, reaction_class: 8, confidence_only: 1 }),
    ("W0", 2,  2, Flips { cuts: 512, verdict: 0, load_bearing: 0, reaction_class: 7, confidence_only: 1 }),
    ("W0", 2,  4, Flips { cuts: 512, verdict: 0, load_bearing: 0, reaction_class: 6, confidence_only: 1 }),
    ("W0", 2,  8, Flips { cuts: 512, verdict: 0, load_bearing: 0, reaction_class: 4, confidence_only: 1 }),
    ("W0", 2, 16, Flips { cuts: 512, verdict: 0, load_bearing: 0, reaction_class: 1, confidence_only: 0 }),
    ("W0", 8,  1, Flips { cuts: 2048, verdict: 0, load_bearing: 0, reaction_class: 632, confidence_only: 54 }),
    ("W0", 8,  2, Flips { cuts: 2048, verdict: 0, load_bearing: 0, reaction_class: 628, confidence_only: 54 }),
    ("W0", 8,  4, Flips { cuts: 2048, verdict: 0, load_bearing: 0, reaction_class: 338, confidence_only: 43 }),
    ("W0", 8,  8, Flips { cuts: 2048, verdict: 0, load_bearing: 0, reaction_class: 244, confidence_only: 36 }),
    ("W0", 8, 16, Flips { cuts: 2048, verdict: 0, load_bearing: 0, reaction_class: 238, confidence_only: 45 }),
];

#[test]
fn axis3_is_pinned() {
    let c = compose_tables();
    let compose = compose_of(&c);
    let ts = tables();
    let mut got = Vec::new();
    for (name, kind) in [("Uniform", Corpus::Uniform), ("W0", Corpus::W0)] {
        for len in [2usize, 8] {
            let chains = corpus(kind, 256, len);
            for (n, t) in LEVELS.iter().zip(&ts) {
                got.push((name, len, *n, flips(&law_table(t), &chains, compose)));
            }
        }
    }
    assert_eq!(got, AXIS3.to_vec());
}

/// The W0 corpus cannot flip a verdict at all: every input frequency is
/// at least 128, so every weighted mean is too, and every chain clears the bar of
/// 128 under every law. Its zero verdict flips are structural, not evidence of
/// stability.
#[test]
fn w0_corpus_cannot_flip_the_frequency_bar() {
    let c = compose_tables();
    let compose = compose_of(&c);
    let a = law_a();
    for ch in corpus(Corpus::W0, 256, 8) {
        let t = trajectory(&a, &ch, compose);
        assert!(t.iter().all(|e| e.frequency_u8() >= DEFAULT_FREQUENCY_BAR));
    }
}

/// The repository's load-bearing fixture (`chain_counterfactual`'s
/// `cutting_a_load_bearing_edge_flips_the_verdict`: seed 200/200, steps
/// 250/250 then 40/30, cut 0) is load-bearing under B1 ONLY. Under A and
/// under every other resolution both arms clear the bar.
#[test]
fn the_load_bearing_fixture_is_load_bearing_only_under_b1() {
    let c = compose_tables();
    let compose = compose_of(&c);
    let ch = Chain {
        seed: edge(200, 200, InferenceType::Deduction),
        steps: vec![
            (0x91, edge(250, 250, InferenceType::Deduction)),
            (0x92, edge(40, 30, InferenceType::Deduction)),
        ],
    };
    let a = decide(&law_a(), &ch, 0, compose);
    assert_eq!(
        a.reaction,
        Reaction::TruthOnly {
            factual: 246,
            counterfactual: 194
        }
    );
    let lb: Vec<bool> = tables()
        .iter()
        .map(|t| decide(&law_table(t), &ch, 0, compose).load_bearing)
        .collect();
    assert_eq!(lb, vec![true, false, false, false, false]);
}

/// G2 re-derivation: a load-bearing fixture under law A, the law
/// recommended for decision-bearing wiring. Seed 200/200, a strong support
/// step 250/250, then a confident refutation 10/230. Cutting the support
/// leaves the refutation to dominate, so the verdict flips under exact
/// revision. B1's verdict on the same chain is pinned beside it as the
/// historical record, not as the contract.
#[test]
fn a_load_bearing_fixture_under_law_a() {
    let c = compose_tables();
    let compose = compose_of(&c);
    let ch = Chain {
        seed: edge(200, 200, InferenceType::Deduction),
        steps: vec![
            (0x91, edge(250, 250, InferenceType::Deduction)),
            (0x92, edge(10, 230, InferenceType::Deduction)),
        ],
    };
    let a = decide(&law_a(), &ch, 0, compose);
    assert_eq!(
        a.reaction,
        Reaction::LoadBearing {
            factual: 210,
            counterfactual: 64
        }
    );
    // The mirror image of the old fixture: B1, B2, B4 and B8 call this cut
    // NOT load-bearing; only B16 agrees with A.
    let lb: Vec<bool> = tables()
        .iter()
        .map(|t| decide(&law_table(t), &ch, 0, compose).load_bearing)
        .collect();
    assert_eq!(lb, vec![false, false, false, false, true]);
}

/// Anti-vacuity for the flip counter: a counter that compared only
/// frequencies would miss TruthOnly decisions that come from confidence
/// alone. This fixture has A moving confidence with frequency held, so a
/// confidence-blind classifier calls it Inert while `Reaction::classify`
/// does not.
#[test]
fn the_flip_counter_sees_confidence_only_changes() {
    let c = compose_tables();
    let compose = compose_of(&c);
    // Cutting one of two identical steps leaves the frequency where it was
    // and lowers only the accumulated confidence.
    let w = edge(200, 120, InferenceType::Revision);
    let ch = Chain {
        seed: edge(200, 120, InferenceType::Revision),
        steps: vec![(0x90, w), (0x91, w)],
    };
    let d = decide(&law_a(), &ch, 1, compose);
    assert!(d.freq_equal, "{d:?}");
    assert_eq!(class(d.reaction), Class::TruthOnly, "{d:?}");
    let blind = if d.freq_equal && !d.load_bearing {
        Class::Inert
    } else {
        class(d.reaction)
    };
    assert_ne!(blind, class(d.reaction));
    // And B1 is exactly that blind classifier here.
    let t1 = NarsTables::build(1);
    assert_eq!(
        class(decide(&law_table(&t1), &ch, 1, compose).reaction),
        Class::Inert
    );
}
