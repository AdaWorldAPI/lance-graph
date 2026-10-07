//! **D-PUZZLE-ATTN-0 — entropy as the focus of attention, on crosswords.**
//!
//! #1387 showed that crossword solving is a fixed point: placing a word ANDs
//! its letters into every crossing slot, and every order of propagation
//! reaches the same state. Deduction needs no attention. Attention starts
//! where deduction runs out: when no slot is forced, a search has to choose
//! WHERE to look next. This probe changes only that choice and holds
//! everything else fixed: the same puzzles, the same propagation
//! (`shared/crossword_core.rs`, moved unchanged from #1387), the same value
//! order (most frequent word first).
//!
//! # The slot policies
//!
//! | policy | next slot | reads |
//! |---|---|---|
//! | `Fixed` | the first unplaced slot | nothing |
//! | `Random` | a seeded random unplaced slot | nothing |
//! | `Widest` | the most candidates | popcount (negative control) |
//! | `Popcount` | the fewest candidates: `log2(popcount)`, uniform entropy | popcount |
//! | `Shannon` | the lowest `H = -Σ p log2 p`, `p ∝` COCA frequency | the frequency prior |
//! | `DomWdeg` | the fewest candidates per accumulated failure | popcount + past contradictions |
//!
//! `Popcount` is #1387's `min_by_key(count)`: entropy under a uniform prior.
//! `Shannon` differs only when candidates carry unequal weight, which the
//! frequency-ranked vocabulary gives them. `DomWdeg` is Boussemart et al.'s
//! weighted-degree heuristic (2004) in its simplest form: every contradiction
//! adds one to the weight of the slot that ran out of candidates, and later
//! choices divide a slot's count by that weight. It is the one policy whose
//! choices depend on what earlier folds did.
//!
//! A slot with no candidate is a contradiction: it has no entropy and is never
//! read as settled.
//!
//! # Workloads
//!
//! - **prove**: created puzzles (unique by construction); count fills up to 2.
//! - **fill**: an empty grid; find one fill.
//! - **open**: created puzzles with only the first half of their givens; count
//!   fills up to 50.
//!
//! Run: `cargo run --release -p cognitive-shader-driver --example crossword_attention_probe`
//! Tests (CI): `cargo test -p cognitive-shader-driver --example crossword_attention_probe`

use std::time::Instant;

use deepnsm_v2::vocab::WordId;

#[path = "shared/population_fold.rs"]
mod population_fold;
use population_fold::Rng;

#[path = "shared/crossword_core.rs"]
mod crossword_core;
use crossword_core::*;

const ACADEMIC: &str = include_str!("../../deepnsm/word_frequency/academic_20k.csv");

/// The frequency prior: per `WordId`, its COCA-All count `z` and `z ln z`.
/// Surfaces that collapse when lowercased sum.
struct Prior {
    z: Vec<f64>,
    zlnz: Vec<f64>,
}

impl Prior {
    fn coca(cold: &Cold) -> Self {
        let n = cold.vocab.len();
        let mut z = vec![0.0f64; n];
        for line in ACADEMIC.lines().skip(1) {
            let c: Vec<&str> = line.split(',').collect();
            let (Some(w), Some(f)) = (c.get(3), c.get(5)) else {
                continue;
            };
            if let (Some(id), Ok(f)) = (cold.vocab.id(&w.to_lowercase()), f.parse::<f64>()) {
                z[id as usize] += f;
            }
        }
        // A word with no count still counts as a candidate.
        for v in z.iter_mut() {
            *v = v.max(1.0);
        }
        let zlnz = z.iter().map(|v| v * v.ln()).collect();
        Prior { z, zlnz }
    }

    /// Shannon entropy in bits of the candidates in `mask` under this prior;
    /// `None` for an empty mask (a contradiction, not certainty).
    fn entropy(&self, mask: &[u64]) -> Option<f64> {
        let (mut zs, mut s) = (0.0, 0.0);
        let mut any = false;
        for (b, &word) in mask.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let i = b * 64 + w.trailing_zeros() as usize;
                zs += self.z[i];
                s += self.zlnz[i];
                any = true;
                w &= w - 1;
            }
        }
        any.then(|| ((zs.ln() - s / zs) / std::f64::consts::LN_2).max(0.0))
    }
}

/// Uniform entropy: `log2(popcount)`; `None` for an empty mask.
#[cfg(test)]
fn uniform_entropy(count: u64) -> Option<f64> {
    (count > 0).then(|| (count as f64).log2())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    Fixed,
    Random(u64),
    Widest,
    Popcount,
    Shannon,
    DomWdeg,
}

const POLICIES: [Policy; 6] = [
    Policy::Fixed,
    Policy::Random(0x5EED),
    Policy::Widest,
    Policy::Popcount,
    Policy::Shannon,
    Policy::DomWdeg,
];

/// What one search did.
#[derive(Debug, Clone, Default, PartialEq)]
struct Stats {
    /// Words placed and propagated.
    nodes: u64,
    /// Placements whose propagation emptied a slot.
    contradictions: u64,
    /// Fills found (capped).
    fills: u64,
    /// The first fill found.
    first: Option<Vec<WordId>>,
    /// Nodes spent before the first fill.
    nodes_to_first: Option<u64>,
    /// Ran out of budget.
    exhausted: bool,
    /// Per node, in order: did the placement contradict? (Bounded.)
    fail_trace: Vec<bool>,
    /// Decisions where Shannon and popcount would pick different slots.
    disagreements: u64,
    decisions: u64,
}

struct Search<'a> {
    hot: &'a Hot,
    puz: &'a Puzzle,
    prior: &'a Prior,
    policy: Policy,
    rng: Rng,
    /// DomWdeg: per slot, contradictions it caused by running dry.
    weight: Vec<u64>,
    budget: u64,
    cap: u64,
    stats: Stats,
}

const TRACE_CAP: usize = 1 << 20;

impl<'a> Search<'a> {
    fn new(
        hot: &'a Hot,
        puz: &'a Puzzle,
        prior: &'a Prior,
        policy: Policy,
        budget: u64,
        cap: u64,
    ) -> Self {
        let seed = match policy {
            Policy::Random(s) => s,
            _ => 0,
        };
        Search {
            hot,
            puz,
            prior,
            policy,
            rng: Rng(seed),
            weight: vec![0; puz.slots()],
            budget,
            cap,
            stats: Stats::default(),
        }
    }

    /// The slot to branch on, or `None` when every slot is placed.
    fn select(&mut self, st: &State) -> Option<usize> {
        let open: Vec<usize> = (0..self.puz.slots())
            .filter(|&s| st.placed[s] == UNSET)
            .collect();
        if open.is_empty() {
            return None;
        }
        let by_count = |f: &dyn Fn(usize) -> f64| {
            open.iter()
                .copied()
                .min_by(|&a, &b| f(a).total_cmp(&f(b)).then(a.cmp(&b)))
        };
        // How often the two entropy readings disagree, measured on the
        // popcount search's own decisions.
        let pc = by_count(&|s| st.count(s) as f64);
        if self.policy == Policy::Popcount {
            let sh = by_count(&|s| self.prior.entropy(st.mask(s)).unwrap_or(-1.0));
            self.stats.decisions += 1;
            self.stats.disagreements += u64::from(pc != sh);
        }
        match self.policy {
            Policy::Fixed => Some(open[0]),
            Policy::Random(_) => Some(open[self.rng.below(open.len() as u64) as usize]),
            Policy::Widest => by_count(&|s| -(st.count(s) as f64)),
            Policy::Popcount => pc,
            Policy::Shannon => by_count(&|s| self.prior.entropy(st.mask(s)).unwrap_or(-1.0)),
            Policy::DomWdeg => by_count(&|s| st.count(s) as f64 / (1 + self.weight[s]) as f64),
        }
    }

    /// Depth-first: fills under `st`, up to the cap. `false` once the
    /// budget or the cap stops the search.
    fn go(&mut self, st: State) -> bool {
        let Some(s) = self.select(&st) else {
            if no_repeats(&st.placed) {
                self.stats.fills += 1;
                if self.stats.first.is_none() {
                    self.stats.first = Some(st.placed.clone());
                    self.stats.nodes_to_first = Some(self.stats.nodes);
                }
            }
            return self.stats.fills < self.cap;
        };
        for w in bits(st.mask(s)) {
            if st.placed.contains(&w) {
                continue;
            }
            if self.stats.nodes >= self.budget {
                self.stats.exhausted = true;
                return false;
            }
            self.stats.nodes += 1;
            let mut next = st.clone();
            next.place(s, w);
            let mut q = vec![s as u8];
            let failed = match propagate_token(self.hot, self.puz, &mut next, &mut q) {
                Err(Stop::Contradiction(t)) => {
                    self.weight[t as usize] += 1;
                    true
                }
                Err(Stop::Language) => unreachable!("one language"),
                Ok(()) => !no_repeats(&next.placed),
            };
            self.stats.contradictions += u64::from(failed);
            if self.stats.fail_trace.len() < TRACE_CAP {
                self.stats.fail_trace.push(failed);
            }
            if !failed && !self.go(next) {
                return false;
            }
        }
        true
    }
}

/// Search from the fixed point of `givens`.
fn search(
    hot: &Hot,
    puz: &Puzzle,
    prior: &Prior,
    givens: &[(u8, WordId)],
    policy: Policy,
    budget: u64,
    cap: u64,
) -> Stats {
    let mut s = Search::new(hot, puz, prior, policy, budget, cap);
    if let Ok((mut st, mut q)) = start(hot, puz, givens) {
        if propagate_token(hot, puz, &mut st, &mut q).is_ok() {
            s.go(st);
        }
    }
    s.stats
}

/// The workloads, built once.
struct Bench {
    hot: Hot,
    prior: Prior,
    created: Vec<Created>,
    empty: Vec<Puzzle>,
}

fn bench(sides: &[usize], per_side: usize, seed: u64) -> Bench {
    let (hot, cold) = build_lexicon(Lang::En, &english_ranked());
    let prior = Prior::coca(&cold);
    let mut rng = Rng(seed);
    let mut created = Vec::new();
    let mut empty = Vec::new();
    for &side in sides {
        let mut made = 0;
        while made < per_side {
            if let Ok(c) = create(&hot, side, &mut rng, 100_000) {
                created.push(c);
                made += 1;
            }
        }
        for _ in 0..per_side {
            empty.push(compile(&Grid::random_nyt(side, &mut rng), Lang::En));
        }
    }
    Bench {
        hot,
        prior,
        created,
        empty,
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Row {
    runs: u64,
    solved: u64,
    nodes: u64,
    contradictions: u64,
    nodes_to_first: u64,
    disagreements: u64,
    decisions: u64,
}

impl Row {
    fn add(&mut self, s: &Stats, solved: bool) {
        self.runs += 1;
        self.solved += u64::from(solved);
        self.nodes += s.nodes;
        self.contradictions += s.contradictions;
        self.nodes_to_first += s.nodes_to_first.unwrap_or(s.nodes);
        self.disagreements += s.disagreements;
        self.decisions += s.decisions;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Work {
    Prove,
    Fill,
    Open,
}

fn run_work(b: &Bench, work: Work, policy: Policy, budget: u64) -> (Row, Vec<Stats>) {
    let mut row = Row::default();
    let mut all = Vec::new();
    match work {
        Work::Prove => {
            for c in &b.created {
                let s = search(&b.hot, &c.puz, &b.prior, &c.givens, policy, budget, 2);
                row.add(&s, !s.exhausted);
                all.push(s);
            }
        }
        Work::Fill => {
            for p in &b.empty {
                let s = search(&b.hot, p, &b.prior, &[], policy, budget, 1);
                row.add(&s, s.fills > 0);
                all.push(s);
            }
        }
        Work::Open => {
            for c in &b.created {
                let half = &c.givens[..c.givens.len() / 2];
                let s = search(&b.hot, &c.puz, &b.prior, half, policy, budget, 50);
                row.add(&s, !s.exhausted);
                all.push(s);
            }
        }
    }
    (row, all)
}

/// Contradiction rate per tenth of a search's node sequence, pooled.
fn fail_deciles(all: &[Stats]) -> [f64; 10] {
    let mut n = [0u64; 10];
    let mut f = [0u64; 10];
    for s in all {
        let len = s.fail_trace.len();
        for (i, &x) in s.fail_trace.iter().enumerate() {
            let d = (i * 10 / len.max(1)).min(9);
            n[d] += 1;
            f[d] += u64::from(x);
        }
    }
    core::array::from_fn(|d| f[d] as f64 / n[d].max(1) as f64)
}

fn main() {
    println!("D-PUZZLE-ATTN-0: entropy as the focus of attention on crosswords");
    let t0 = Instant::now();
    let b = bench(&[5, 7], 10, 0xA77E);
    println!(
        "  workload: {} created puzzles, {} empty grids (sides 5/7), built in {:.1} s",
        b.created.len(),
        b.empty.len(),
        t0.elapsed().as_secs_f64()
    );
    let budget = 200_000;
    for (name, work) in [
        ("prove (count to 2)", Work::Prove),
        ("fill (empty grid)", Work::Fill),
        ("open (half the givens, count to 50)", Work::Open),
    ] {
        println!("\n{name}, budget {budget} nodes per run:");
        println!(
            "  {:<10} {:>6} {:>10} {:>10} {:>12} {:>8} {:>13}",
            "policy", "solved", "nodes", "contra", "to first", "ms", "Sh≠Pc"
        );
        for p in POLICIES {
            let t = Instant::now();
            let (r, all) = run_work(&b, work, p, budget);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            println!(
                "  {:<10} {:>3}/{:<2} {:>10} {:>10} {:>12} {:>8.1} {:>13}",
                format!("{p:?}").split('(').next().unwrap_or(""),
                r.solved,
                r.runs,
                r.nodes,
                r.contradictions,
                r.nodes_to_first,
                ms,
                if p == Policy::Popcount {
                    format!("{:.3}", r.disagreements as f64 / r.decisions.max(1) as f64)
                } else {
                    "-".into()
                },
            );
            if matches!(p, Policy::Popcount | Policy::DomWdeg) {
                let d = fail_deciles(&all);
                println!(
                    "             contradiction rate by tenth of the search: {}",
                    d.iter()
                        .map(|x| format!("{x:.2}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
        }
    }
}

// ── Falsifiers ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    /// Small and shared: sides 5 and 7, a few of each.
    fn bench_small() -> &'static Bench {
        static B: OnceLock<Bench> = OnceLock::new();
        B.get_or_init(|| bench(&[5], 6, 0xA77E))
    }

    const BUDGET: u64 = 50_000;

    /// Measurement helper (not an assertion): the small-fixture table.
    #[test]
    #[ignore]
    fn dump_small() {
        let b = bench_small();
        for work in [Work::Prove, Work::Fill, Work::Open] {
            for p in POLICIES {
                let (r, _) = run_work(b, work, p, BUDGET);
                println!(
                    "{work:?} {p:?} solved {}/{} nodes {} first {} contra {}",
                    r.solved, r.runs, r.nodes, r.nodes_to_first, r.contradictions
                );
            }
        }
    }

    /// A contradiction is not certainty: an empty slot has no entropy under
    /// either reading; a single candidate has zero.
    #[test]
    fn a_contradiction_has_no_entropy() {
        let b = bench_small();
        assert_eq!(b.prior.entropy(&[0, 0]), None);
        assert_eq!(uniform_entropy(0), None);
        assert_eq!(b.prior.entropy(&[1, 0]), Some(0.0));
        assert_eq!(uniform_entropy(1), Some(0.0));
    }

    /// Shannon under the frequency prior is not log2(popcount): two slots
    /// with the same 128 candidates, one of them holding "the" (the most
    /// frequent word), differ by more than a bit while log2(popcount) reads
    /// both as exactly 7.
    #[test]
    fn shannon_is_not_log2_popcount() {
        let b = bench_small();
        let n = b.prior.z.len();
        let mut flat = vec![0u64; b.hot.blocks];
        for id in n - 200..n - 72 {
            flat[id / 64] |= 1 << (id % 64);
        }
        let mut skewed = flat.clone();
        let first = n - 200;
        skewed[first / 64] &= !(1 << (first % 64));
        skewed[0] |= 1; // id 0 = "the"
        let count = |m: &[u64]| m.iter().map(|w| w.count_ones()).sum::<u32>();
        assert_eq!((count(&flat), count(&skewed)), (128, 128));
        let (hf, hs) = (
            b.prior.entropy(&flat).unwrap(),
            b.prior.entropy(&skewed).unwrap(),
        );
        assert!(hs < hf - 1.0, "{hs} vs {hf}");
        assert!((uniform_entropy(128).unwrap() - 7.0).abs() < 1e-12);
    }

    /// Attention never changes the answer: on unique puzzles every policy
    /// that finishes finds exactly one fill, and it is the solution.
    #[test]
    fn attention_changes_cost_never_the_answer() {
        let b = bench_small();
        for p in POLICIES {
            let (_, all) = run_work(b, Work::Prove, p, BUDGET);
            for (c, s) in b.created.iter().zip(&all) {
                if s.exhausted {
                    continue;
                }
                assert_eq!(s.fills, 1, "{p:?}");
                assert_eq!(s.first.as_ref(), Some(&c.solution), "{p:?}");
            }
        }
    }

    /// Focus matters: most-constrained-first spends far fewer nodes than
    /// a fixed order and than widest-first, and solves at least as many.
    #[test]
    fn most_constrained_first_beats_fixed_order_and_widest_first() {
        let b = bench_small();
        for work in [Work::Prove, Work::Fill, Work::Open] {
            let pc = run_work(b, work, Policy::Popcount, BUDGET).0;
            for base in [Policy::Fixed, Policy::Widest] {
                let r = run_work(b, work, base, BUDGET).0;
                assert!(pc.solved >= r.solved, "{work:?} {base:?}");
                assert!(
                    2 * pc.nodes < r.nodes,
                    "{work:?} {base:?}: {} vs {}",
                    pc.nodes,
                    r.nodes
                );
            }
        }
    }

    /// The measured loss: frequency-weighted Shannon attention spends more
    /// nodes than popcount. Search cost is the number of candidates to try,
    /// not how predictable the slot is. If this ever flips, re-read it.
    #[test]
    fn shannon_attention_costs_more_than_popcount() {
        let b = bench_small();
        for work in [Work::Fill, Work::Open] {
            let pc = run_work(b, work, Policy::Popcount, BUDGET).0;
            let sh = run_work(b, work, Policy::Shannon, BUDGET).0;
            assert!(
                sh.nodes > pc.nodes,
                "{work:?}: {} vs {}",
                sh.nodes,
                pc.nodes
            );
            assert!(sh.solved <= pc.solved, "{work:?}");
        }
    }

    /// DomWdeg's weights only grow from contradictions, and a search with
    /// none behaves exactly like popcount.
    #[test]
    fn dom_wdeg_is_popcount_until_something_fails() {
        let b = bench_small();
        let c = &b.created[0];
        // From the full givens propagation finishes the puzzle: no choice,
        // no contradiction, and the two policies are identical.
        let pc = search(
            &b.hot,
            &c.puz,
            &b.prior,
            &c.givens,
            Policy::Popcount,
            BUDGET,
            2,
        );
        let dw = search(
            &b.hot,
            &c.puz,
            &b.prior,
            &c.givens,
            Policy::DomWdeg,
            BUDGET,
            2,
        );
        if pc.contradictions == 0 {
            assert_eq!(pc, dw);
        }
        // Can fire: on empty grids contradictions happen and the weights
        // move the search somewhere else.
        let (_, a) = run_work(b, Work::Fill, Policy::Popcount, BUDGET);
        let (_, d) = run_work(b, Work::Fill, Policy::DomWdeg, BUDGET);
        assert!(a.iter().map(|s| s.contradictions).sum::<u64>() > 0);
        assert_ne!(a, d);
    }

    /// Same puzzles, same seed, same policy: the same search.
    #[test]
    fn replay_is_deterministic() {
        let b = bench_small();
        for p in POLICIES {
            assert_eq!(
                run_work(b, Work::Fill, p, 5_000).1,
                run_work(b, Work::Fill, p, 5_000).1,
                "{p:?}"
            );
        }
    }
}
