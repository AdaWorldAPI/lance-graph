//! P-D4: the answer/abstain gate, judged by a strictly proper score
//! (pre-registered in
//! `.claude/board/entries/2026-10-22-coresearch-pal256-evidence-probes.md`).
//!
//! Synthetic world, fixed before the first run:
//! - 40 blocks × 25 questions, 4 candidate answers each. A block shares its
//!   8 sources, so questions inside a block are dependent; every interval is
//!   a block bootstrap, never an iid one (I-NOISE-FLOOR-JIRAK).
//! - Block kinds: 26 reliable (source reliability 0.6–0.9), 10 weak (0.3),
//!   4 out of support (0.25, chance). The support flag is simulated with
//!   the error rates P-D3 measured (2% false positives, 100% recall).
//! - Each question gets 3–6 observations, each a positive vote (f = 230,
//!   c = 128) for one candidate; in 30% of questions the first observation
//!   is echoed 3 more times under the same source stamp.
//!
//! Arms:
//! - (a) always answer: production revision without stamps; output the
//!   top candidate with p = its NARS expectation.
//! - (b) gate: stamp-gated revision (P-E2), and answer only if the top
//!   candidate's interval (mean ± 2 sd of the Jeffreys posterior) clears the
//!   runner-up's, it rests on ≥ 2 distinct sources, and the support flag is
//!   clear. Abstentions output p = π, the always-answer accuracy measured on
//!   the other half of the blocks.
//!
//! Pre-registered kill: Brier(a) − Brier(b) < 0.02, or its 95% block-
//! bootstrap interval includes 0.
//!
//! RESULT OF THE FIRST RUN: the pre-registered margin rule never answers
//! (coverage 0). A candidate with no evidence has Jeffreys sd 0.35, and a
//! runner-up with one vote has an interval reaching above 1, so mean ± 2 sd
//! cannot clear with 3–9 observations. The rule is killed by vacuity, and
//! `preregistered_interval_margin_is_vacuous` pins that.
//!
//! CORRECTION (codex review of #1724): π is estimated from the same blocks
//! being resampled, so it is refitted inside every bootstrap draw. With that,
//! the post-hoc gain is 0.0242 [−0.0062, 0.0552]: the interval includes 0 and
//! the kill fires. The earlier [0.0079, 0.0409] held π fixed.
//!
//! POST-HOC arm (b′), declared after that run and before any other: the
//! margin is on evidence counted once — the top candidate needs at least 2
//! more distinct sources than the runner-up. The other two conditions are
//! unchanged. Being post-hoc, its pass is weaker evidence than a
//! pre-registered one.
//!
//! AMENDMENT (2026-10-22, before the first run): the shuffled-label control
//! in the board entry compared Brier scores of the gate and always-answer
//! on shuffled labels. That cannot stay silent: replacing a p ≠ π with π
//! lowers the Brier score on any labels, so a gate that sees no labels still
//! "gains". The label-blind check used instead is selective accuracy:
//! - real labels: accuracy on answered items exceeds accuracy on abstained
//!   items (interval excludes 0);
//! - labels shuffled within blocks: the same difference has an interval that
//!   includes 0.

use causal_edge::Opcode;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn unit(&mut self) -> f64 {
        (self.next() as f64) / ((1u64 << 53) as f64)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const CANDIDATES: usize = 4;
const VOTE: (u8, u8) = (230, 128);

#[derive(Clone, Copy, Default)]
struct Belief {
    f: u8,
    c: u8,
    stamp: u64,
}

fn revise(a: Belief, b: Belief, gated: bool) -> Belief {
    if a.c == 0 {
        return b;
    }
    if gated && a.stamp & b.stamp != 0 {
        return if b.c > a.c { b } else { a };
    }
    let (f, c) = Opcode::Revision.truth_code(a.f, a.c, b.f, b.c);
    Belief {
        f,
        c,
        stamp: a.stamp | b.stamp,
    }
}

/// Jeffreys posterior mean and sd from codes (horizon k = 1).
fn posterior(b: Belief) -> (f64, f64) {
    if b.c == 0 {
        return (0.5, 0.5_f64.sqrt() / 2.0);
    }
    let c = f64::from(b.c);
    let w = c / (255.0 - c);
    let wp = w * f64::from(b.f) / 255.0;
    let (x, y) = (wp + 0.5, w - wp + 0.5);
    (
        x / (x + y),
        (x * y / ((x + y) * (x + y) * (x + y + 1.0))).sqrt(),
    )
}

struct Question {
    block: usize,
    truth: usize,
    ood_flag: bool,
    /// (candidate, source stamp) per observation.
    votes: Vec<(usize, u64)>,
}

fn world() -> Vec<Question> {
    let mut rng = Lcg(0xD4_D4D4);
    let mut out = Vec::new();
    for block in 0..40 {
        let (lo, hi, ood) = match block {
            0..=25 => (0.6, 0.9, false),
            26..=35 => (0.3, 0.3, false),
            _ => (0.25, 0.25, true),
        };
        let rel: Vec<f64> = (0..8).map(|_| lo + (hi - lo) * rng.unit()).collect();
        for _ in 0..25 {
            let truth = rng.below(CANDIDATES);
            let n = 3 + rng.below(4);
            let mut votes = Vec::new();
            for _ in 0..n {
                let s = rng.below(8);
                let cand = if rng.unit() < rel[s] {
                    truth
                } else {
                    (truth + 1 + rng.below(CANDIDATES - 1)) % CANDIDATES
                };
                votes.push((cand, 1u64 << s));
            }
            if rng.unit() < 0.3 {
                let first = votes[0];
                votes.extend([first; 3]);
            }
            let ood_flag = if ood { true } else { rng.unit() < 0.02 };
            out.push(Question {
                block,
                truth,
                ood_flag,
                votes,
            });
        }
    }
    out
}

struct Outcome {
    block: usize,
    p_a: f64,
    y_a: f64,
    answered: bool,
    p_b_answer: f64,
    y_b: f64,
}

fn beliefs(q: &Question, gated: bool) -> [Belief; CANDIDATES] {
    let mut bs = [Belief::default(); CANDIDATES];
    for &(cand, stamp) in &q.votes {
        let obs = Belief {
            f: VOTE.0,
            c: VOTE.1,
            stamp,
        };
        bs[cand] = revise(bs[cand], obs, gated);
    }
    bs
}

fn ranked(bs: &[Belief; CANDIDATES]) -> (usize, usize) {
    let mut idx: Vec<usize> = (0..CANDIDATES).collect();
    idx.sort_by(|&i, &j| posterior(bs[j]).0.total_cmp(&posterior(bs[i]).0));
    (idx[0], idx[1])
}

#[derive(Clone, Copy)]
enum Margin {
    /// Pre-registered: mean ± 2 sd of the Jeffreys posterior.
    Interval,
    /// Post-hoc: ≥ 2 more distinct sources than the runner-up.
    Sources,
}

fn evaluate(qs: &[Question], rule: Margin) -> Vec<Outcome> {
    qs.iter()
        .map(|q| {
            let ua = beliefs(q, false);
            let (ta, _) = ranked(&ua);
            let gb = beliefs(q, true);
            let (tb, rb) = ranked(&gb);
            let (mt, st) = posterior(gb[tb]);
            let (mr, sr) = posterior(gb[rb]);
            let sources = gb[tb].stamp.count_ones();
            let interval = mt - 2.0 * st > mr + 2.0 * sr;
            let source_margin = sources >= gb[rb].stamp.count_ones() + 2;
            let margin = match rule {
                Margin::Interval => interval,
                Margin::Sources => source_margin,
            };
            let answered = margin && sources >= 2 && !q.ood_flag;
            Outcome {
                block: q.block,
                p_a: posterior(ua[ta]).0,
                y_a: f64::from(u8::from(ta == q.truth)),
                answered,
                p_b_answer: mt,
                y_b: f64::from(u8::from(tb == q.truth)),
            }
        })
        .collect()
}

/// π for blocks of one parity: always-answer accuracy on the other parity.
fn base_rate(os: &[Outcome], parity: usize) -> f64 {
    let other: Vec<&Outcome> = os.iter().filter(|o| o.block % 2 != parity).collect();
    other.iter().map(|o| o.y_a).sum::<f64>() / other.len() as f64
}

/// The same base rate refitted on the drawn blocks (with multiplicity), so a
/// bootstrap interval carries the base rate's own estimation error.
fn base_rate_in(os: &[Outcome], blocks: &[usize], parity: usize) -> f64 {
    let (mut s, mut n) = (0.0_f64, 0.0_f64);
    for &blk in blocks.iter().filter(|&&b| b % 2 != parity) {
        for o in os.iter().filter(|o| o.block == blk) {
            s += o.y_a;
            n += 1.0;
        }
    }
    s / n.max(1.0)
}

fn brier_diff(os: &[Outcome], blocks: &[usize], pi: [f64; 2]) -> f64 {
    let (mut a, mut b, mut n) = (0.0, 0.0, 0.0);
    for &blk in blocks {
        for o in os.iter().filter(|o| o.block == blk) {
            a += (o.p_a - o.y_a).powi(2);
            let p = if o.answered {
                o.p_b_answer
            } else {
                pi[o.block % 2]
            };
            b += (p - o.y_b).powi(2);
            n += 1.0;
        }
    }
    (a - b) / n
}

/// Answered minus abstained accuracy, pooled over the drawn blocks. Not
/// label-blind: the gate abstains on whole out-of-support blocks, whose
/// accuracy is low, and shuffling inside a block keeps that.
fn selective_gap(os: &[Outcome], ys: &[f64], blocks: &[usize]) -> f64 {
    let (mut ans, mut na, mut abs, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for &blk in blocks {
        for (o, &y) in os.iter().zip(ys).filter(|(o, _)| o.block == blk) {
            if o.answered {
                ans += y;
                na += 1.0;
            } else {
                abs += y;
                nb += 1.0;
            }
        }
    }
    ans / na.max(1.0) - abs / nb.max(1.0)
}

/// The same gap computed inside each block (blocks with both answered and
/// abstained items) and averaged. Shuffling inside a block makes this zero
/// in expectation for any gate that does not see the labels.
fn stratified_gap(os: &[Outcome], ys: &[f64], blocks: &[usize]) -> f64 {
    let (mut sum, mut n) = (0.0_f64, 0.0_f64);
    for &blk in blocks {
        let one = [blk];
        let (na, nb) = os
            .iter()
            .filter(|o| o.block == blk)
            .fold(
                (0, 0),
                |(a, b), o| {
                    if o.answered {
                        (a + 1, b)
                    } else {
                        (a, b + 1)
                    }
                },
            );
        if na > 0 && nb > 0 {
            sum += selective_gap(os, ys, &one);
            n += 1.0;
        }
    }
    sum / n.max(1.0)
}

/// 95% block-bootstrap interval of `stat`, 2000 resamples.
fn block_ci(stat: impl Fn(&[usize]) -> f64) -> (f64, f64, f64) {
    let blocks: Vec<usize> = (0..40).collect();
    let point = stat(&blocks);
    let mut rng = Lcg(0xB007);
    let mut draws: Vec<f64> = (0..2000)
        .map(|_| {
            let s: Vec<usize> = (0..40).map(|_| rng.below(40)).collect();
            stat(&s)
        })
        .collect();
    draws.sort_by(f64::total_cmp);
    (point, draws[50], draws[1949])
}

#[test]
fn preregistered_interval_margin_is_vacuous() {
    let os = evaluate(&world(), Margin::Interval);
    let answered = os.iter().filter(|o| o.answered).count();
    println!("interval margin: answered {answered}/{}", os.len());
    assert!(answered * 20 < os.len(), "the interval rule now answers");
}

#[test]
/// Codex review of #1724: with π refitted inside every bootstrap draw (π is
/// estimated from the same blocks), the interval includes 0. The post-hoc
/// gate therefore does NOT clear the kill condition; this pins that. The
/// earlier fixed-π interval [0.0079, 0.0409] omitted π's estimation error.
fn post_hoc_gate_gain_is_not_established() {
    let os = evaluate(&world(), Margin::Sources);
    let coverage = os.iter().filter(|o| o.answered).count() as f64 / os.len() as f64;
    let acc_a = os.iter().map(|o| o.y_a).sum::<f64>() / os.len() as f64;
    let pi = [base_rate(&os, 0), base_rate(&os, 1)];
    let (_, flo, fhi) = block_ci(|b| brier_diff(&os, b, pi));
    let (d, lo, hi) =
        block_ci(|b| brier_diff(&os, b, [base_rate_in(&os, b, 0), base_rate_in(&os, b, 1)]));
    println!(
        "coverage {coverage:.3}, always-answer accuracy {acc_a:.3}, pi {pi:.3?}, \
         Brier(a)-Brier(b) {d:.4}, refitted-pi interval [{lo:.4}, {hi:.4}] \
         (fixed-pi interval [{flo:.4}, {fhi:.4}], which omits pi's error)"
    );
    assert!(
        (0.2..0.95).contains(&coverage),
        "gate is vacuous: coverage {coverage}"
    );
    assert!(
        lo <= 0.0,
        "the gain is now established: {d} [{lo}, {hi}] — re-pin P-D4"
    );
}

#[test]
fn gate_selects_on_evidence_not_on_labels() {
    let os = evaluate(&world(), Margin::Sources);
    let real: Vec<f64> = os.iter().map(|o| o.y_b).collect();
    let mut shuffled = real.clone();
    let mut rng = Lcg(0x5AFF);
    for blk in 0..40 {
        let idx: Vec<usize> = (0..os.len()).filter(|&i| os[i].block == blk).collect();
        for k in (1..idx.len()).rev() {
            let j = rng.below(k + 1);
            shuffled.swap(idx[k], idx[j]);
        }
    }
    let (pg, plo, phi) = block_ci(|b| selective_gap(&os, &real, b));
    let (pgs, pslo, pshi) = block_ci(|b| selective_gap(&os, &shuffled, b));
    println!(
        "pooled gap (confounded): real {pg:.3} [{plo:.3}, {phi:.3}], \
         shuffled {pgs:.3} [{pslo:.3}, {pshi:.3}]"
    );
    let (g, lo, hi) = block_ci(|b| stratified_gap(&os, &real, b));
    let (gs, slo, shi) = block_ci(|b| stratified_gap(&os, &shuffled, b));
    println!(
        "within-block gap: real {g:.3} [{lo:.3}, {hi:.3}], \
         shuffled {gs:.3} [{slo:.3}, {shi:.3}]"
    );
    assert!(lo > 0.0, "gate does not select on evidence: [{lo}, {hi}]");
    assert!(
        slo <= 0.0 && shi >= 0.0,
        "gate gains on shuffled labels: [{slo}, {shi}]"
    );
}
