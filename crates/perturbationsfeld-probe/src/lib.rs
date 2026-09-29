//! D-PFP-1 — the Perturbationsfeld probe.
//!
//! Measurement only (workspace-excluded). Spec: `.claude/plans/perturbationsfeld-probe-v1.md`
//! §9 (ratified v3); every constant below is pre-registered in `PREREG.md`
//! and a test asserts the two agree.
//!
//! What it asks, and nothing more:
//! 1. Is the thinking-engine's output sensitive to its input on the tracked
//!    256² tables at the δ level?
//! 2. Given it is, does the E-set (dense energy lowered through
//!    `lance-graph-mask-risc` as `Pred::GtI32` over an order-preserving key
//!    lane) retain the cycle-n top set into cycle n+1 more, less, or no
//!    differently than the C-set (today's `top_k` window lowered as
//!    `Pred::Range`)?
//!
//! Retention is self-consistency, not fidelity. No result is attributed to
//! "address" alone. Not related to `crates/perturbation-sim` (a power-grid
//! simulator).

use lance_graph_mask_risc::{
    execute, materialize_rows, reference_scratch, LaneRef, MaskOp, Operand, Planes, Pred, Program,
    Scratch, Terminal, Value,
};
use thinking_engine::codebook_index::CodebookIndex;
use thinking_engine::dto::PerturbationDto;
use thinking_engine::engine::ThinkingEngine;

/// The pre-registered constants (`PREREG.md`). Never edited after the first run.
pub mod prereg {
    /// Aperture threshold θ — reuses `SCAN_WORTHY_ENERGY`
    /// (`cognitive-shader-driver/src/engine_bridge.rs:107`). Hand-set.
    pub const THETA: f32 = 0.01;
    /// Decision margin δ = 2 of 8 top-set ids per stimulus. Hand-set.
    pub const DELTA_IDS: u32 = 2;
    /// Top-set width (the width of `PerturbationDto::top_k`).
    pub const TOP: u32 = 8;
    /// Number of stimuli.
    pub const Q: usize = 32;
    /// Token ids per stimulus.
    pub const M: usize = 8;
    /// Cycles per `think`.
    pub const MAX_CYCLES: usize = 10;
    /// Stimulus seed (workspace SplitMix64 convention).
    pub const STIMULUS_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
    /// Sabotage permutation seed.
    pub const PERMUTATION_SEED: u64 = 0x5EED_0000_0000_0001;
    /// Degeneracy ceiling, percent of stimuli excluded from a required comparison.
    pub const DEGENERATE_CEILING_PCT: usize = 25;
    /// Minimum eligible stimuli for the relabel sanity to be evaluable.
    pub const SANITY_MIN_ELIGIBLE: usize = 8;
    /// Empty-window fallback width (`engine_bridge.rs:136-137`).
    pub const EMPTY_WINDOW: usize = 64;
}

/// Probe-local helpers. Workspace convention, local copies by house pattern;
/// none of these is a shared primitive.
pub mod helpers {
    /// SplitMix64 (probe-local copy; the workspace has no shared impl).
    pub struct SplitMix64(pub u64);

    impl SplitMix64 {
        /// Next 64-bit value.
        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform in `0..n` (modulo draw; bias is irrelevant at these sizes
        /// and the draw is pre-registered as written).
        pub fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }

    /// Exact f32 → i32 total-order key: `a < b ⇔ key(a) < key(b)` for every
    /// finite, non-NaN pair, except that `-0.0` and `+0.0` get distinct keys
    /// (hence θ must be nonzero). PROBE ONLY — not a lane pattern.
    pub fn key(e: f32) -> i32 {
        let b = e.to_bits() as i32;
        if b < 0 {
            b ^ 0x7FFF_FFFF
        } else {
            b
        }
    }

    /// Fixed Fisher–Yates permutation of `0..n`.
    pub fn permutation(n: usize, seed: u64) -> Vec<u16> {
        let mut p: Vec<u16> = (0..n as u16).collect();
        let mut r = SplitMix64(seed);
        for i in (1..n).rev() {
            let j = r.below(i as u64 + 1) as usize;
            p.swap(i, j);
        }
        p
    }
}

use helpers::{key, permutation, SplitMix64};
use prereg::*;

/// One real lens: a 256² table and its token → centroid index.
pub struct Lens {
    /// Human name for the report.
    pub name: &'static str,
    /// Distance table bytes (N² u8).
    pub table: &'static [u8],
    /// LE u16 token → centroid index.
    pub index: &'static [u8],
}

/// PRIMARY lens: Jina v5 (ground truth per the workspace model registry).
pub const JINA_V5: Lens = Lens {
    name: "jina-v5 (PRIMARY)",
    table: include_bytes!("../../thinking-engine/data/jina-v5-codebook/distance_table_256x256.u8"),
    index: include_bytes!("../../thinking-engine/data/jina-v5-codebook/codebook_index.u16"),
};

/// REPLICATION lens: BGE-M3.
pub const BGE_M3: Lens = Lens {
    name: "bge-m3 (REPLICATION)",
    table: include_bytes!("../../thinking-engine/data/bge-m3-hdr/distance_table_256x256.u8"),
    index: include_bytes!("../../thinking-engine/data/bge-m3-hdr/codebook_index.u16"),
};

/// Active top set: `top_k` ids with `e > 0` (zero-energy padding removed).
pub fn act(p: &PerturbationDto) -> Vec<u16> {
    p.top_k
        .iter()
        .filter(|&&(_, e)| e > 0.0)
        .map(|&(i, _)| i)
        .collect()
}

/// `|act(a) ∩ act(b)|`, 0..=8.
pub fn inter(a: &PerturbationDto, b: &PerturbationDto) -> u32 {
    let aa = act(a);
    act(b).iter().filter(|i| aa.contains(i)).count() as u32
}

/// Number of zero-energy padding slots in `top_k`.
pub fn padding(p: &PerturbationDto) -> u32 {
    TOP - act(p).len() as u32
}

fn all_zero(e: &[f32]) -> bool {
    e.iter().map(|&x| x as f64).sum::<f64>() < 1e-10
}

fn has_nan(e: &[f32]) -> bool {
    e.iter().any(|x| x.is_nan())
}

fn l1(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).sum()
}

/// One `think` from a fresh state (RESET semantics).
pub fn fire(engine: &mut ThinkingEngine, ids: &[u16]) -> PerturbationDto {
    engine.reset();
    engine.perturb(ids);
    engine.think(MAX_CYCLES)
}

/// The C window over the active `top_k` rows with `e > theta`, or the
/// empty-window fallback. Returns `[lo, hi)`.
pub fn window(p: &PerturbationDto, theta: f32, n: usize) -> (u32, u32) {
    let active: Vec<u16> = p
        .top_k
        .iter()
        .filter(|&&(_, e)| e > theta)
        .map(|&(i, _)| i)
        .collect();
    if active.is_empty() {
        (0, n.min(EMPTY_WINDOW) as u32)
    } else {
        let lo = *active.iter().min().unwrap() as u32;
        let hi = (*active.iter().max().unwrap() as u32 + 1).min(n as u32);
        (lo, hi)
    }
}

fn keep_program(pred: Pred) -> Program {
    Program {
        ops: vec![MaskOp::Pred {
            pred,
            under: None,
            dst: 0,
        }],
        terminal: Terminal::Keep {
            mask: Operand::Scratch(0),
        },
        scratch_slots: 1,
    }
}

/// Run a one-predicate `Keep` program through the executor AND the oracle.
/// Returns the executor's ids and whether the oracle's slot agrees bit-for-bit.
fn run_keep(program: &Program, planes: &Planes<'_>, n: usize) -> (Vec<u16>, bool) {
    let mut scratch = Scratch::for_program(program, n).expect("scratch");
    let v = execute(program, planes, &mut scratch, None).expect("execute");
    assert_eq!(v, Value::Mask(Operand::Scratch(0)));
    let words = n.div_ceil(64);
    let slot = scratch.slot(0).expect("slot 0")[..words].to_vec();
    let oracle = reference_scratch(program, planes).expect("oracle");
    let agree = oracle[0][..words] == slot[..];
    let ids = materialize_rows(&slot, n)
        .into_iter()
        .map(|i| i as u16)
        .collect();
    (ids, agree)
}

/// C arm through mask-risc `Pred::Range` + `Keep`.
pub fn c_ids(p: &PerturbationDto, theta: f32, n: usize) -> (Vec<u16>, bool) {
    let (lo, hi) = window(p, theta, n);
    let program = keep_program(Pred::Range { lo, hi });
    let planes = Planes {
        n_rows: n,
        masks: &[],
        lanes: &[],
    };
    let (ids, agree) = run_keep(&program, &planes, n);
    let scalar: Vec<u16> = (lo..hi).map(|i| i as u16).collect();
    (ids.clone(), agree && ids == scalar)
}

/// C' arm: the same window as a scalar id list (TRIPWIRE).
pub fn c_prime_ids(p: &PerturbationDto, theta: f32, n: usize) -> Vec<u16> {
    let (lo, hi) = window(p, theta, n);
    (lo..hi).map(|i| i as u16).collect()
}

/// E arm through the key lane + `Pred::GtI32` + `Keep`.
pub fn e_ids(energy: &[f32], theta: f32) -> (Vec<u16>, bool) {
    let n = energy.len();
    let lane: Vec<i32> = energy.iter().map(|&e| key(e)).collect();
    let lanes = [LaneRef::I32(&lane)];
    let program = keep_program(Pred::GtI32 {
        lane: 0,
        t: key(theta),
    });
    let planes = Planes {
        n_rows: n,
        masks: &[],
        lanes: &lanes,
    };
    let (ids, agree) = run_keep(&program, &planes, n);
    let scalar: Vec<u16> = (0..n)
        .filter(|&i| energy[i] > theta)
        .map(|i| i as u16)
        .collect();
    (ids.clone(), agree && ids == scalar)
}

/// E_m (reported only): top-`k` rows by energy among `e > 0`, ties by lower id.
pub fn e_m_ids(energy: &[f32], k: usize) -> Vec<u16> {
    let mut idx: Vec<usize> = (0..energy.len()).filter(|&i| energy[i] > 0.0).collect();
    idx.sort_by(|&a, &b| energy[b].total_cmp(&energy[a]).then(a.cmp(&b)));
    idx.truncate(k);
    idx.sort_unstable();
    idx.into_iter().map(|i| i as u16).collect()
}

/// The per-lens outcome (spec §9.6), in evaluation order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Invalid(Vec<String>),
    InputInsensitive,
    RelabelInsensitive,
    HigherRetention,
    LowerRetention,
    NoRetentionDifference,
}

impl Outcome {
    /// The pre-registered label.
    pub fn label(&self) -> &'static str {
        match self {
            Outcome::Invalid(_) => "INVALID",
            Outcome::InputInsensitive => "INPUT-INSENSITIVE",
            Outcome::RelabelInsensitive => "RELABEL-INSENSITIVE",
            Outcome::HigherRetention => "HIGHER-RETENTION",
            Outcome::LowerRetention => "LOWER-RETENTION",
            Outcome::NoRetentionDifference => "NO-RETENTION-DIFFERENCE",
        }
    }
}

/// Everything one lens run measured.
#[derive(Debug)]
pub struct Report {
    pub lens: &'static str,
    pub n: usize,
    pub base_valid: usize,
    pub q_valid: usize,
    pub sanity_eligible: usize,
    pub positive_pair: (u16, u16),
    pub positive_inter: u32,
    pub n0: f64,
    pub r_e: u32,
    pub r_c: u32,
    pub r_s: u32,
    pub r_em: u32,
    pub delta_r: i64,
    pub d_es: u32,
    pub mean_c: f64,
    pub mean_e: f64,
    pub padding_rate: f64,
    pub l1_ec: f64,
    pub theta_c_inert: bool,
    pub outcome: Outcome,
}

/// Run the full pre-registered protocol on one lens.
pub fn run_lens(lens: &Lens) -> Report {
    let table = lens.table.to_vec();
    let mut engine = ThinkingEngine::new(table.clone());
    let n = engine.size;
    let vocab = lens.index.len() / 2;
    let index: Vec<u16> = lens
        .index
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    let cb = CodebookIndex::new(index, n as u16, lens.name.to_string());

    let mut invalid: Vec<String> = Vec::new();

    // Stimuli.
    let mut rng = SplitMix64(STIMULUS_SEED);
    let stimuli: Vec<Vec<u16>> = (0..Q)
        .map(|_| {
            let toks: Vec<u32> = (0..M).map(|_| rng.below(vocab as u64) as u32).collect();
            cb.lookup_many(&toks)
        })
        .collect();

    let pi = permutation(n, PERMUTATION_SEED);

    // Cycle n (twice, V1 tripwire).
    let p_n: Vec<PerturbationDto> = stimuli.iter().map(|s| fire(&mut engine, s)).collect();
    for (s, p) in stimuli.iter().zip(&p_n) {
        let again = fire(&mut engine, s);
        if again.top_k != p.top_k {
            invalid.push("V1 determinism (cycle n)".into());
            break;
        }
        if has_nan(&p.energy) {
            invalid.push("NaN in e_n".into());
            break;
        }
    }

    // Arms.
    struct Arms {
        c: Vec<u16>,
        cp: Vec<u16>,
        e: Vec<u16>,
        em: Vec<u16>,
        s: Vec<u16>,
    }
    let mut arms: Vec<Arms> = Vec::with_capacity(Q);
    let mut oracle_ok = true;
    let (mut e_shrinks, mut e_grows, mut c_moves) = (false, false, false);
    for p in &p_n {
        let (c, c_ok) = c_ids(p, THETA, n);
        let (e, e_ok) = e_ids(&p.energy, THETA);
        oracle_ok &= c_ok && e_ok;
        let (e_hi, _) = e_ids(&p.energy, THETA * 2.0);
        let (e_lo, _) = e_ids(&p.energy, THETA / 2.0);
        e_shrinks |= e_hi.len() < e.len();
        e_grows |= e_lo.len() > e.len();
        let (c_hi, _) = c_ids(p, THETA * 2.0, n);
        let (c_lo, _) = c_ids(p, THETA / 2.0, n);
        c_moves |= c_hi != c || c_lo != c;
        let em = e_m_ids(&p.energy, c.len());
        let s: Vec<u16> = e.iter().map(|&i| pi[i as usize]).collect();
        arms.push(Arms {
            cp: c_prime_ids(p, THETA, n),
            c,
            e,
            em,
            s,
        });
    }
    if !oracle_ok {
        invalid.push("V3 lowering oracle".into());
    }
    if !(e_shrinks && e_grows) {
        invalid.push("theta inertness (E)".into());
    }

    // Cycle n+1 per arm (C, C', E, S twice for V1; E_m once).
    let mut next: Vec<[PerturbationDto; 5]> = Vec::with_capacity(Q);
    for a in &arms {
        let run = |eng: &mut ThinkingEngine, ids: &[u16]| fire(eng, ids);
        let c = run(&mut engine, &a.c);
        let cp = run(&mut engine, &a.cp);
        let e = run(&mut engine, &a.e);
        let s = run(&mut engine, &a.s);
        let em = run(&mut engine, &a.em);
        for (ids, first) in [(&a.c, &c), (&a.e, &e), (&a.s, &s)] {
            if run(&mut engine, ids).top_k != first.top_k {
                invalid.push("V1 determinism (cycle n+1)".into());
            }
        }
        for x in [&c, &cp, &e, &s, &em] {
            if has_nan(&x.energy) {
                invalid.push("NaN in e_n+1".into());
            }
        }
        next.push([c, cp, e, s, em]);
    }
    invalid.dedup();

    // Degeneracy.
    let base_ok: Vec<bool> = (0..Q)
        .map(|i| !all_zero(&p_n[i].energy) && !arms[i].e.is_empty())
        .collect();
    let ok = |i: usize, k: usize| !all_zero(&next[i][k].energy);
    let base_valid = base_ok.iter().filter(|&&b| b).count();
    let cmp_valid: Vec<usize> = (0..Q)
        .filter(|&i| base_ok[i] && ok(i, 0) && ok(i, 2))
        .collect();
    let q_valid = cmp_valid.len();
    if (Q - q_valid) * 100 > DEGENERATE_CEILING_PCT * Q {
        invalid.push(format!(
            "degeneracy ceiling ({} of {} excluded)",
            Q - q_valid,
            Q
        ));
    }

    // C vs C' tripwire.
    for &i in &cmp_valid {
        if ok(i, 1) && inter(&next[i][0], &next[i][1]) != act(&next[i][0]).len() as u32 {
            invalid.push("C vs C' tripwire".into());
            break;
        }
        if arms[i].c != arms[i].cp {
            invalid.push("C vs C' ids differ".into());
            break;
        }
    }

    // Metrics.
    let r_of = |k: usize| -> u32 {
        (0..Q)
            .filter(|&i| base_ok[i] && ok(i, k))
            .map(|i| inter(&p_n[i], &next[i][k]))
            .sum()
    };
    let r_c = r_of(0);
    let r_e = r_of(2);
    let r_s = r_of(3);
    let r_em = r_of(4);
    let delta_r: i64 = cmp_valid
        .iter()
        .map(|&i| inter(&p_n[i], &next[i][2]) as i64 - inter(&p_n[i], &next[i][0]) as i64)
        .sum();

    // N0: cross-stimulus overlap of P_n.
    let nz: Vec<usize> = (0..Q).filter(|&i| !all_zero(&p_n[i].energy)).collect();
    let (mut pair_sum, mut pairs) = (0u64, 0u64);
    for (a, &i) in nz.iter().enumerate() {
        for &j in &nz[a + 1..] {
            pair_sum += inter(&p_n[i], &p_n[j]) as u64;
            pairs += 1;
        }
    }
    let n0 = if pairs == 0 {
        1.0
    } else {
        pair_sum as f64 / (pairs as f64 * TOP as f64)
    };

    // Positive control: the least-similar pair of distinct centroids.
    let (mut pa, mut pb, mut best) = (0u16, 1u16, u8::MAX);
    for a in 0..n {
        for b in 0..n {
            if a != b && table[a * n + b] < best {
                best = table[a * n + b];
                pa = a as u16;
                pb = b as u16;
            }
        }
    }
    let px = fire(&mut engine, &[pa]);
    let py = fire(&mut engine, &[pb]);
    let positive_inter = inter(&px, &py);

    // Sanity eligibility.
    let sanity: Vec<usize> = (0..Q)
        .filter(|&i| base_ok[i] && ok(i, 2) && ok(i, 3) && arms[i].e.len() <= n / 2)
        .collect();
    let d_es: u32 = sanity
        .iter()
        .map(|&i| TOP - inter(&next[i][2], &next[i][3]))
        .sum();

    let mean_c =
        cmp_valid.iter().map(|&i| arms[i].c.len()).sum::<usize>() as f64 / q_valid.max(1) as f64;
    let mean_e =
        cmp_valid.iter().map(|&i| arms[i].e.len()).sum::<usize>() as f64 / q_valid.max(1) as f64;
    let (mut pad, mut slots) = (0u32, 0u32);
    for i in 0..Q {
        for x in std::iter::once(&p_n[i]).chain(next[i].iter()) {
            pad += padding(x);
            slots += TOP;
        }
    }
    let l1_ec = cmp_valid
        .iter()
        .map(|&i| l1(&next[i][2].energy, &next[i][0].energy))
        .sum::<f64>()
        / q_valid.max(1) as f64;

    // Outcome (evaluation order is the pre-registration).
    let outcome = if !invalid.is_empty() {
        Outcome::Invalid(invalid)
    } else if positive_inter > TOP - DELTA_IDS || (1.0 - n0) * (TOP as f64) < DELTA_IDS as f64 {
        Outcome::InputInsensitive
    } else if sanity.len() >= SANITY_MIN_ELIGIBLE
        && (d_es as usize) < DELTA_IDS as usize * sanity.len()
    {
        Outcome::RelabelInsensitive
    } else {
        let margin = DELTA_IDS as i64 * q_valid as i64;
        if delta_r >= margin {
            Outcome::HigherRetention
        } else if delta_r <= -margin {
            Outcome::LowerRetention
        } else {
            Outcome::NoRetentionDifference
        }
    };

    Report {
        lens: lens.name,
        n,
        base_valid,
        q_valid,
        sanity_eligible: sanity.len(),
        positive_pair: (pa, pb),
        positive_inter,
        n0,
        r_e,
        r_c,
        r_s,
        r_em,
        delta_r,
        d_es,
        mean_c,
        mean_e,
        padding_rate: pad as f64 / slots as f64,
        l1_ec,
        theta_c_inert: !c_moves,
        outcome,
    }
}

/// Combine PRIMARY and REPLICATION into the reported verdict (spec §9.4).
pub fn verdict(primary: &Outcome, replication: &Outcome) -> String {
    match (primary, replication) {
        (Outcome::Invalid(_), _) => "INVALID".to_string(),
        (p, Outcome::Invalid(_)) => format!("{}, REPLICATION INVALID", p.label()),
        (p, r) if p.label() != r.label() => format!("{}, LENS-SPECIFIC", p.label()),
        (p, _) => p.label().to_string(),
    }
}
