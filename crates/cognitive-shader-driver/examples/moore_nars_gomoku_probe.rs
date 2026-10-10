//! D-MOORE-NARS-0 — can the NARS recipe palette learn, from the consequences
//! of its own reasoning, which recipe to run in which recurring Moore-local
//! situation of a coupled world (Gomoku)?
//!
//! The loop, one learner move:
//!
//! ```text
//! board → Moore hydration (empty cells with a stone among their 8 neighbours)
//!       → per-cell line readings in the 4 Moore directions (the snapshot)
//!       → mechanical propagation: own win, else forced block
//!       → if ≥ 2 candidates remain: structural signature → choose a recipe
//!       → the recipe reasons over the population → a move
//!       → CE64 record; W names the pending credit slot
//!       → opponent replies → at the learner's next turn the consequence is
//!         read off the board → signed activation → credit (signature, recipe)
//! ```
//!
//! What is learned is `signature × recipe → usefulness`, never
//! `position → move`. The signature is relative (mine / theirs) and holds
//! only line classes, never coordinates.
//!
//! Recipes are realized here from their catalogue IDEA on Moore lines; the
//! catalogue's own `substrate` (CLAM, VSA, InnerCouncil …) is not present in
//! this world. [`census`] says, for all 34, which are realized and why the
//! rest are not.
//!
//! Run: `cargo run --release -p cognitive-shader-driver --example moore_nars_gomoku_probe`

use std::collections::HashMap;

use causal_edge::layout::{PLAST_MASK, PLAST_SHIFT};
use causal_edge::{CausalEdge64, CausalMask};
use lance_graph_contract::epistemic_state5::{Certification3, Epi5Gen, EpistemicState5, Topology2};
#[cfg(test)]
use lance_graph_contract::recipes::RECIPES;

// ── Census of the 34 catalogue recipes ─────────────────────────────────────

/// How a catalogue recipe stands in this domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Standing {
    /// Realized as a move-reasoning operator over the Moore population.
    Applicable(Op),
    /// What it describes is the chooser itself (recipe selection, memory,
    /// exploration), not an operator the chooser selects.
    ChooserLevel,
    /// Its idea is already carried by a realized operator here.
    Redundant(Op),
    /// Needs a substrate this world does not have (VSA bundle, fingerprint
    /// clusters, personas, a second modality).
    Unreachable,
    /// Meaningful here, not built in this probe.
    NotWired,
    /// Its action has no legal meaning here (Gomoku has no pass / HOLD).
    Unsafe,
}

/// `(code, standing, reason)` for all 34, in catalogue order.
fn census() -> [(&'static str, Standing, &'static str); 34] {
    use Op::*;
    use Standing::*;
    [
        (
            "RTE",
            Redundant(Asc),
            "recursion depth = lookahead depth, carried by ASC/ICR",
        ),
        (
            "HTD",
            NotWired,
            "split the board into independent sub-fights",
        ),
        ("SMAD", Redundant(Tcf), "multi-agent vote = TCF agreement"),
        (
            "RCR",
            Applicable(Rcr),
            "backward from the opponent's strongest line",
        ),
        (
            "TCP",
            Applicable(Tcp),
            "prune cells that touch no line, then extend",
        ),
        ("TR", Applicable(Tr), "seeded choice among the top three"),
        (
            "ASC",
            Applicable(Asc),
            "2-ply adversarial refutation over top-4 x top-4",
        ),
        (
            "CAS",
            Redundant(Cur),
            "abstraction scaling = CUR coarse-to-fine",
        ),
        ("IRS", Unreachable, "no persona kernels in this world"),
        (
            "MCP",
            ChooserLevel,
            "calibrating own competence = the recipe chooser",
        ),
        (
            "CR",
            Applicable(Cr),
            "own attack vs block conflict, resolved by counterfactual",
        ),
        ("TCA", NotWired, "temporal context = move history window"),
        (
            "CDT",
            ChooserLevel,
            "explore/exploit = the chooser's prior on untried recipes",
        ),
        ("MCT", Unreachable, "single modality"),
        ("LSI", Unreachable, "no fingerprint clusters"),
        (
            "PSO",
            ChooserLevel,
            "TD-learned template slots = the learned stats",
        ),
        ("CDI", Unsafe, "HOLD = pass; Gomoku has no pass"),
        (
            "CWS",
            ChooserLevel,
            "episodic memory = the persistent learned state",
        ),
        ("ARE", Unreachable, "no VSA bind to invert"),
        ("TCF", Applicable(Tcf), "majority of HPM, RCR, TCP"),
        (
            "SSR",
            Redundant(Asc),
            "challenge schedule = adversarial refutation",
        ),
        ("ETD", NotWired, "cluster-derived subtasks"),
        (
            "AMP",
            ChooserLevel,
            "Q-values over styles = the learned stats",
        ),
        ("ZCF", Unreachable, "no VSA bind"),
        (
            "HPM",
            Applicable(Hpm),
            "pattern-table score over Moore lines, both sides",
        ),
        (
            "CUR",
            Applicable(Cur),
            "coarse class first, fine score within the top class",
        ),
        ("MPC", Redundant(Tcf), "majority-vote bundle = TCF"),
        (
            "SSAM",
            NotWired,
            "analogy would reuse moves: position -> move, excluded",
        ),
        ("IDR", Unreachable, "no agent/action/patient grammar here"),
        ("SPP", Redundant(Tcf), "parallel paths + agreement = TCF"),
        (
            "ICR",
            Applicable(Icr),
            "counterfactual: play, take the opponent's reply, read",
        ),
        ("SDD", Unreachable, "the world is exact; no noise floor"),
        (
            "DTMF",
            ChooserLevel,
            "switch template on BLOCK = credit on contradiction",
        ),
        ("HKF", Unreachable, "no cross-domain bind"),
    ]
}

// ── The recipe operators realized here ─────────────────────────────────────

/// One operator per realized recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Op {
    Hpm,
    Rcr,
    Tcp,
    Cur,
    Tr,
    Icr,
    Asc,
    Cr,
    Tcf,
}

const OPS: [Op; 9] = [
    Op::Hpm,
    Op::Rcr,
    Op::Tcp,
    Op::Cur,
    Op::Tr,
    Op::Icr,
    Op::Asc,
    Op::Cr,
    Op::Tcf,
];
const NO: usize = OPS.len();

impl Op {
    fn code(self) -> &'static str {
        match self {
            Op::Hpm => "HPM",
            Op::Rcr => "RCR",
            Op::Tcp => "TCP",
            Op::Cur => "CUR",
            Op::Tr => "TR",
            Op::Icr => "ICR",
            Op::Asc => "ASC",
            Op::Cr => "CR",
            Op::Tcf => "TCF",
        }
    }
    fn index(self) -> usize {
        OPS.iter().position(|&o| o == self).unwrap()
    }
}

// ── Board and Moore lines ──────────────────────────────────────────────────

const MAXC: usize = 13 * 13;
const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

/// Line classes, weakest to strongest. `FIVE` wins.
const OPEN2: u8 = 1;
const THREE: u8 = 2;
const OPEN3: u8 = 3;
const FOUR: u8 = 4;
const OPEN4: u8 = 5;
const FIVE: u8 = 6;
const WEIGHT: [u32; 7] = [0, 4, 12, 60, 200, 5000, 100_000];

#[derive(Clone)]
struct Board {
    n: usize,
    win: usize,
    cell: [u8; MAXC],
    stones: usize,
}

impl Board {
    fn new(n: usize, win: usize) -> Self {
        assert!(n * n <= MAXC);
        Board {
            n,
            win,
            cell: [0; MAXC],
            stones: 0,
        }
    }
    fn get(&self, r: i32, c: i32) -> Option<u8> {
        let n = self.n as i32;
        (r >= 0 && c >= 0 && r < n && c < n).then(|| self.cell[(r * n + c) as usize])
    }
    fn play(&mut self, i: usize, p: u8) {
        debug_assert_eq!(self.cell[i], 0);
        self.cell[i] = p;
        self.stones += 1;
    }
    fn full(&self) -> bool {
        self.stones == self.n * self.n
    }
    /// Length and open ends of `p`'s line through `i` in direction `d`, as
    /// if `p` stood on `i`.
    fn line(&self, i: usize, p: u8, (dr, dc): (i32, i32)) -> (usize, u8) {
        let n = self.n as i32;
        let (r0, c0) = (i as i32 / n, i as i32 % n);
        let mut len = 1;
        let mut open = 0;
        for s in [1, -1] {
            let (mut r, mut c) = (r0 + s * dr, c0 + s * dc);
            loop {
                match self.get(r, c) {
                    Some(x) if x == p => {
                        len += 1;
                        r += s * dr;
                        c += s * dc;
                    }
                    Some(0) => {
                        open += 1;
                        break;
                    }
                    _ => break,
                }
            }
        }
        (len, open)
    }
    /// Score and strongest class of placing `p` on `i` (one fold).
    fn eval(&self, i: usize, p: u8) -> (u32, u8, u8) {
        let mut score = 0;
        let mut best = 0;
        let mut best_dir = 0;
        let (mut fours, mut open3) = (0, 0);
        for (k, &d) in DIRS.iter().enumerate() {
            let (len, open) = self.line(i, p, d);
            let w = self.win;
            let class = if len >= w {
                FIVE
            } else if len + 1 == w {
                if open == 2 {
                    OPEN4
                } else if open == 1 {
                    FOUR
                } else {
                    0
                }
            } else if len + 2 == w {
                if open == 2 {
                    OPEN3
                } else if open == 1 {
                    THREE
                } else {
                    0
                }
            } else if len + 3 == w && open == 2 {
                OPEN2
            } else {
                0
            };
            if class >= FOUR {
                fours += 1;
            }
            if class == OPEN3 {
                open3 += 1;
            }
            score += WEIGHT[class as usize];
            if class > best {
                best = class;
                best_dir = k as u8;
            }
        }
        // Two simultaneous threats cannot both be blocked.
        if best < FIVE && (fours >= 2 || (fours >= 1 && open3 >= 1)) {
            best = OPEN4;
        } else if best < FOUR && open3 >= 2 {
            best = FOUR;
        }
        (score, best, best_dir)
    }
    /// Moore hydration: empty cells with a stone among their 8 neighbours.
    fn candidates(&self) -> Vec<usize> {
        let n = self.n as i32;
        if self.stones == 0 {
            return vec![(self.n / 2) * self.n + self.n / 2];
        }
        let mut out = Vec::new();
        for i in 0..self.n * self.n {
            if self.cell[i] != 0 {
                continue;
            }
            let (r, c) = (i as i32 / n, i as i32 % n);
            let near = (-1..=1).any(|dr| {
                (-1..=1)
                    .any(|dc| (dr, dc) != (0, 0) && matches!(self.get(r + dr, c + dc), Some(1 | 2)))
            });
            if near {
                out.push(i);
            }
        }
        out
    }
    fn hash(&self, me: u8) -> u64 {
        let mut h = 0xcbf29ce484222325u64 ^ u64::from(me);
        for &x in &self.cell[..self.n * self.n] {
            h = (h ^ u64::from(x)).wrapping_mul(0x100000001b3);
        }
        h
    }
}

/// Per-candidate readings for both sides: the hydrated population.
#[derive(Clone, Copy, Debug)]
struct Cell {
    i: usize,
    s_me: u32,
    c_me: u8,
    s_op: u32,
    c_op: u8,
    dir: u8,
}

#[derive(Clone)]
struct Snap {
    cells: Vec<Cell>,
}

fn snapshot(b: &Board, me: u8, folds: &mut u64) -> Snap {
    let op = 3 - me;
    let cells = b
        .candidates()
        .into_iter()
        .map(|i| {
            let (s_me, c_me, dme) = b.eval(i, me);
            let (s_op, c_op, dop) = b.eval(i, op);
            Cell {
                i,
                s_me,
                c_me,
                s_op,
                c_op,
                dir: if c_me >= c_op { dme } else { dop },
            }
        })
        .collect::<Vec<_>>();
    *folds += 2 * cells.len() as u64;
    Snap { cells }
}

impl Snap {
    fn best_me(&self) -> u8 {
        self.cells.iter().map(|c| c.c_me).max().unwrap_or(0)
    }
    fn best_op(&self) -> u8 {
        self.cells.iter().map(|c| c.c_op).max().unwrap_or(0)
    }
    fn argmax(&self, f: impl Fn(&Cell) -> i64) -> Cell {
        let mut best = self.cells[0];
        let mut bv = f(&best);
        for c in &self.cells[1..] {
            let v = f(c);
            if v > bv {
                bv = v;
                best = *c;
            }
        }
        best
    }
    fn top(&self, k: usize, f: impl Fn(&Cell) -> i64) -> Vec<Cell> {
        let mut v = self.cells.clone();
        v.sort_by_key(|c| (std::cmp::Reverse(f(c)), c.i));
        v.truncate(k);
        v
    }
}

/// The mechanical half: an own win, else a forced block. `None` = reasoning.
fn propagate(s: &Snap) -> Option<usize> {
    if s.cells.len() == 1 {
        return Some(s.cells[0].i);
    }
    if let Some(c) = s.cells.iter().find(|c| c.c_me == FIVE) {
        return Some(c.i);
    }
    s.cells.iter().find(|c| c.c_op == FIVE).map(|c| c.i)
}

/// Static balance from `me`'s side, read off the hydrated population.
fn balance(s: &Snap) -> i32 {
    i32::from(s.best_me()) - i32::from(s.best_op())
}

fn hpm(c: &Cell) -> i64 {
    i64::from(c.s_me) + i64::from(c.s_op)
}

// ── Opponents (fixed, deterministic) ───────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Opp {
    /// Threat heuristic: own weight 1.1, block weight 1.0.
    Threat,
    /// Ignores blocking except a forced one.
    Aggressor,
    /// 2-ply over the top 5.
    Lookahead,
}

fn opp_move(b: &Board, me: u8, kind: Opp) -> usize {
    let mut f = 0;
    let s = snapshot(b, me, &mut f);
    if let Some(i) = propagate(&s) {
        return i;
    }
    match kind {
        Opp::Threat => {
            s.argmax(|c| 11 * i64::from(c.s_me) + 10 * i64::from(c.s_op))
                .i
        }
        Opp::Aggressor => s.argmax(|c| i64::from(c.s_me)).i,
        Opp::Lookahead => minimax(b, me, &s, 5, 5, &mut f),
    }
}

// ── Recipe realizations ────────────────────────────────────────────────────

/// After `me` plays `i`, the reply the threat opponent would make and the
/// balance `me` then reads. Counts its folds.
fn counterfactual(b: &Board, me: u8, i: usize, folds: &mut u64) -> i32 {
    let mut c = b.clone();
    c.play(i, me);
    if c.line(i, me, (0, 1)).0 >= c.win || DIRS.iter().any(|&d| c.line(i, me, d).0 >= c.win) {
        return 7;
    }
    let s1 = snapshot(&c, 3 - me, folds);
    if s1.cells.is_empty() {
        return 0;
    }
    let r = propagate(&s1).unwrap_or_else(|| {
        s1.argmax(|x| 11 * i64::from(x.s_me) + 10 * i64::from(x.s_op))
            .i
    });
    if s1.cells.iter().any(|x| x.i == r && x.c_me == FIVE) {
        return -7;
    }
    c.play(r, 3 - me);
    let s2 = snapshot(&c, me, folds);
    if s2.cells.is_empty() {
        return 0;
    }
    balance(&s2)
}

fn minimax(b: &Board, me: u8, s: &Snap, k1: usize, k2: usize, folds: &mut u64) -> usize {
    let mut best = (i32::MIN, s.cells[0].i);
    for m in s.top(k1, hpm) {
        let mut c = b.clone();
        c.play(m.i, me);
        let worst = if m.c_me == FIVE {
            7
        } else {
            let s1 = snapshot(&c, 3 - me, folds);
            let mut worst = i32::MAX;
            for r in s1.top(k2, hpm) {
                let v = if r.c_me == FIVE {
                    -7
                } else {
                    let mut cc = c.clone();
                    cc.play(r.i, 3 - me);
                    let s2 = snapshot(&cc, me, folds);
                    if s2.cells.is_empty() {
                        0
                    } else {
                        balance(&s2)
                    }
                };
                worst = worst.min(v);
            }
            if worst == i32::MAX {
                0
            } else {
                worst
            }
        };
        if worst > best.0 {
            best = (worst, m.i);
        }
    }
    best.1
}

/// D-LAB-P1: `minimax` with a deterministic certificate. A candidate's value
/// is the minimum over its replies, so the running minimum is an upper bound
/// that only falls. Once it is `<=` the best finished value, the candidate
/// cannot be chosen (the choice needs a strict `>`), and its remaining
/// replies are skipped. Values never exceed 7, so a finished 7 ends the
/// search. Picks the same move as `minimax` by construction; the probe
/// measures what the skipped replies were worth in folds.
fn minimax_certified(b: &Board, me: u8, s: &Snap, k1: usize, k2: usize, folds: &mut u64) -> usize {
    let mut best = (i32::MIN, s.cells[0].i);
    for m in s.top(k1, hpm) {
        if best.0 >= 7 {
            break;
        }
        let mut c = b.clone();
        c.play(m.i, me);
        let worst = if m.c_me == FIVE {
            7
        } else {
            let s1 = snapshot(&c, 3 - me, folds);
            let mut worst = i32::MAX;
            for r in s1.top(k2, hpm) {
                let v = if r.c_me == FIVE {
                    -7
                } else {
                    let mut cc = c.clone();
                    cc.play(r.i, 3 - me);
                    let s2 = snapshot(&cc, me, folds);
                    if s2.cells.is_empty() {
                        0
                    } else {
                        balance(&s2)
                    }
                };
                worst = worst.min(v);
                if worst <= best.0 {
                    break;
                }
            }
            if worst == i32::MAX {
                0
            } else {
                worst
            }
        };
        if worst > best.0 {
            best = (worst, m.i);
        }
    }
    best.1
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Run one recipe over the population. Returns the move; adds its folds.
fn run_op(op: Op, b: &Board, me: u8, s: &Snap, rng: &mut Rng, folds: &mut u64) -> usize {
    match op {
        Op::Hpm => s.argmax(hpm).i,
        Op::Rcr => {
            let worst = s.best_op();
            s.argmax(|c| {
                if c.c_op == worst {
                    1_000_000 + i64::from(c.s_me)
                } else {
                    i64::from(c.s_op)
                }
            })
            .i
        }
        Op::Tcp => {
            let live = s.cells.iter().any(|c| c.c_me >= OPEN2 || c.c_op >= OPEN2);
            s.argmax(|c| {
                if live && c.c_me < OPEN2 && c.c_op < OPEN2 {
                    i64::MIN / 2
                } else {
                    i64::from(c.s_me)
                }
            })
            .i
        }
        Op::Cur => {
            let top = s.cells.iter().map(|c| c.c_me.max(c.c_op)).max().unwrap();
            s.argmax(|c| {
                if c.c_me.max(c.c_op) == top {
                    hpm(c)
                } else {
                    i64::MIN / 2
                }
            })
            .i
        }
        Op::Tr => {
            let t = s.top(3, hpm);
            t[rng.below(t.len())].i
        }
        Op::Icr => {
            let mut best = (i32::MIN, 0);
            for c in s.top(5, hpm) {
                let v = counterfactual(b, me, c.i, folds);
                if v > best.0 {
                    best = (v, c.i);
                }
            }
            best.1
        }
        Op::Asc => minimax(b, me, s, 4, 4, folds),
        Op::Cr => {
            let a = s.argmax(|c| i64::from(c.s_me)).i;
            let d = s.argmax(|c| i64::from(c.s_op)).i;
            if a == d {
                a
            } else {
                let va = counterfactual(b, me, a, folds);
                let vd = counterfactual(b, me, d, folds);
                if va >= vd {
                    a
                } else {
                    d
                }
            }
        }
        Op::Tcf => {
            let picks = [
                run_op(Op::Hpm, b, me, s, rng, folds),
                run_op(Op::Rcr, b, me, s, rng, folds),
                run_op(Op::Tcp, b, me, s, rng, folds),
            ];
            if picks[1] == picks[2] {
                picks[1]
            } else {
                picks[0]
            }
        }
    }
}

// ── The structural signature ───────────────────────────────────────────────

/// Relative, coordinate-free: best own class, best opponent class, how many
/// opponent cells reach an open three, whether attack and block disagree,
/// population size.
fn signature(s: &Snap) -> u64 {
    let own = u64::from(s.best_me());
    let opp = u64::from(s.best_op());
    let threats = s.cells.iter().filter(|c| c.c_op >= OPEN3).count().min(2) as u64;
    let a = s.argmax(|c| i64::from(c.s_me)).i;
    let d = s.argmax(|c| i64::from(c.s_op)).i;
    let conflict = u64::from(a != d);
    let size = match s.cells.len() {
        0..=10 => 0,
        11..=25 => 1,
        _ => 2,
    };
    own | opp << 3 | threats << 6 | conflict << 8 | size << 9
}

fn coarse(s: &Snap) -> u64 {
    u64::from(s.best_me()) | u64::from(s.best_op()) << 3 | 1 << 40
}

// ── The learner ────────────────────────────────────────────────────────────

/// What a learning event is credited from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Signal {
    /// Balance `h` learner turns later minus balance before the move, read
    /// through W; contradiction = -7. `Retained(1)` = after one reply.
    Retained(u8),
    /// Balance right after the move, before the opponent has answered.
    Immediate,
    /// Only contradictions (negative); everything else is no evidence.
    Contradiction,
    /// Only the game result, spread over the game's steps.
    Final,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Memory {
    /// Structural signature, backing off to (own, opp) classes.
    Structural,
    /// The literal board.
    Literal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Chooser {
    /// One recipe always.
    Fixed(Op),
    /// Seeded uniform over the nine.
    Uniform,
    /// Learned NARS expectation.
    Learned,
}

#[derive(Debug, Clone, Copy)]
struct Cfg {
    chooser: Chooser,
    signal: Signal,
    memory: Memory,
    /// Clear learned state at every game start.
    reset_per_game: bool,
    /// Evidence decay per update of a pairing (1.0 = none).
    decay: f64,
    /// Fold-cost weight in the chooser (0 = cost-blind).
    cost_weight: f64,
    /// Update from experience (false = frozen evaluation).
    learn: bool,
}

const BASE: Cfg = Cfg {
    chooser: Chooser::Learned,
    signal: Signal::Retained(1),
    memory: Memory::Structural,
    reset_per_game: false,
    decay: 1.0,
    cost_weight: 0.02,
    learn: true,
};

/// NARS evidence for one (signature, recipe) pairing.
#[derive(Clone, Copy, Default, Debug)]
struct Ev {
    pos: f64,
    neg: f64,
}

impl Ev {
    fn expectation(self) -> f64 {
        let w = self.pos + self.neg;
        if w == 0.0 {
            return 0.5;
        }
        let f = self.pos / w;
        let c = w / (w + 1.0);
        c * (f - 0.5) + 0.5
    }
    fn state(self, topology: Topology2) -> EpistemicState5 {
        let w = self.pos + self.neg;
        let f = if w > 0.0 { self.pos / w } else { 0.5 };
        let c = w / (w + 1.0);
        let cert = if w == 0.0 || f <= 0.5 {
            Certification3::Open
        } else if c < 0.5 {
            Certification3::Associated
        } else if f < 0.7 || c < 0.8 {
            Certification3::Supports
        } else {
            Certification3::CausalCandidate
        };
        EpistemicState5::new(Epi5Gen::V1, topology, cert)
    }
}

/// A pending credit: written at the move, read at the next turn through W.
#[derive(Clone, Copy, Debug)]
struct Pending {
    key: u64,
    back: u64,
    op: Op,
    before: i32,
    immediate: i32,
    /// Moore direction of the move's strongest line.
    dir: u8,
    /// The question the step asked (bits 40..42).
    question: CausalMask,
}

#[derive(Default)]
struct Learner {
    stats: HashMap<u64, [Ev; NO]>,
    /// Global mean folds per recipe (for the cost term).
    folds: [(u64, u64); NO],
}

impl Learner {
    fn choose(&self, cfg: &Cfg, key: u64, back: u64, rng: &mut Rng) -> (Op, bool) {
        match cfg.chooser {
            Chooser::Fixed(op) => (op, false),
            Chooser::Uniform => (OPS[rng.below(NO)], false),
            Chooser::Learned => {
                let (row, hit) = match self.stats.get(&key) {
                    Some(r) => (*r, true),
                    None => match (cfg.memory, self.stats.get(&back)) {
                        (Memory::Structural, Some(r)) => (*r, true),
                        _ => ([Ev::default(); NO], false),
                    },
                };
                let mut best = (f64::MIN, Op::Hpm);
                for (k, &op) in OPS.iter().enumerate() {
                    let (n, f) = self.folds[k];
                    let mean = if n == 0 { 0.0 } else { f as f64 / n as f64 };
                    let v = row[k].expectation() - cfg.cost_weight * (1.0 + mean).ln();
                    if v > best.0 {
                        best = (v, op);
                    }
                }
                (best.1, hit)
            }
        }
    }
    fn credit(&mut self, cfg: &Cfg, p: &Pending, activation: i32) -> EpistemicState5 {
        let mut out = Ev::default();
        let keys: &[u64] = match cfg.memory {
            Memory::Structural => &[p.key, p.back],
            Memory::Literal => &[p.key],
        };
        for &k in keys {
            let row = self.stats.entry(k).or_insert([Ev::default(); NO]);
            let e = &mut row[p.op.index()];
            e.pos *= cfg.decay;
            e.neg *= cfg.decay;
            let a = f64::from(activation) / 7.0;
            if a > 0.0 {
                e.pos += a;
            } else if a < 0.0 {
                e.neg -= a;
            }
            if k == p.key {
                out = *e;
            }
        }
        let topology = if cfg.signal == Signal::Final {
            Topology2::IndirectKnown
        } else {
            Topology2::Direct
        };
        out.state(topology)
    }
}

/// The signed reaction of a reasoning step, from board readings alone. It
/// never sees which recipe ran.
fn activation(before: i32, after: i32, contradiction: bool) -> i32 {
    if contradiction {
        return -7;
    }
    (2 * (after - before)).clamp(-7, 7)
}

fn question(s: &Snap) -> CausalMask {
    if s.best_op() > s.best_me() {
        CausalMask::PO
    } else if s.best_me() > s.best_op() {
        CausalMask::SO
    } else {
        CausalMask::SP
    }
}

/// The CE64 record of one credited reasoning step.
fn record(p: &Pending, w: u8, act: i32, surprise: u8, epi: EpistemicState5) -> CausalEdge64 {
    let mut e = CausalEdge64(0);
    e.set_causal_mask(p.question);
    e.set_direction(p.dir);
    e.set_inference_mantissa(act.clamp(-7, 7) as i8);
    e.0 = (e.0 & !PLAST_MASK) | (u64::from(surprise & 7) << PLAST_SHIFT);
    e.set_w_slot(w);
    e.with_epistemic_raw5(epi.raw())
}

// ── One game ───────────────────────────────────────────────────────────────

#[derive(Default, Clone, Debug)]
struct Stats {
    games: u64,
    wins: u64,
    losses: u64,
    moves: u64,
    mech: u64,
    steps: u64,
    folds: u64,
    ops: [u64; NO],
    productive: u64,
    contradictions: u64,
    hits: u64,
    /// Sum over evaluated steps of (best recipe's activation − chosen's).
    regret: i64,
    regret_n: u64,
    /// Steps where the chosen recipe was among the best.
    optimal: u64,
}

impl Stats {
    fn add(&mut self, o: &Stats) {
        self.games += o.games;
        self.wins += o.wins;
        self.losses += o.losses;
        self.moves += o.moves;
        self.mech += o.mech;
        self.steps += o.steps;
        self.folds += o.folds;
        for k in 0..NO {
            self.ops[k] += o.ops[k];
        }
        self.productive += o.productive;
        self.contradictions += o.contradictions;
        self.hits += o.hits;
        self.regret += o.regret;
        self.regret_n += o.regret_n;
        self.optimal += o.optimal;
    }
    fn score(&self) -> f64 {
        (self.wins as f64 + 0.5 * (self.games - self.wins - self.losses) as f64)
            / self.games.max(1) as f64
    }
    fn folds_per_step(&self) -> f64 {
        self.folds as f64 / self.steps.max(1) as f64
    }
    fn mean_regret(&self) -> f64 {
        self.regret as f64 / self.regret_n.max(1) as f64
    }
    fn optimal_frac(&self) -> f64 {
        self.optimal as f64 / self.regret_n.max(1) as f64
    }
    fn productive_frac(&self) -> f64 {
        self.productive as f64 / self.steps.max(1) as f64
    }
    fn hit_frac(&self) -> f64 {
        self.hits as f64 / self.steps.max(1) as f64
    }
    fn top_share(&self) -> f64 {
        *self.ops.iter().max().unwrap() as f64 / self.steps.max(1) as f64
    }
}

#[derive(Clone, Copy)]
struct World {
    n: usize,
    win: usize,
    opp: Opp,
    /// Score every reasoning step's recipe choice against all nine.
    oracle: bool,
}

/// The consequence every recipe would have had at this step, read exactly
/// (the opponents are deterministic).
fn oracle_regret(
    b: &Board,
    me: u8,
    s: &Snap,
    chosen: usize,
    opp: Opp,
    rng_seed: u64,
) -> (i32, bool) {
    let before = balance(s);
    let mut acts = [0i32; NO];
    for (k, &op) in OPS.iter().enumerate() {
        let mut f = 0;
        let mut rng = Rng(rng_seed);
        let mv = run_op(op, b, me, s, &mut rng, &mut f);
        acts[k] = consequence(b, me, mv, opp, before);
    }
    let best = *acts.iter().max().unwrap();
    (best - acts[chosen], acts[chosen] == best)
}

/// Play `mv`, let `opp` answer, read the activation at `me`'s next turn.
fn consequence(b: &Board, me: u8, mv: usize, opp: Opp, before: i32) -> i32 {
    let mut c = b.clone();
    c.play(mv, me);
    if DIRS.iter().any(|&d| c.line(mv, me, d).0 >= c.win) {
        return 7;
    }
    if c.full() {
        return 0;
    }
    let r = opp_move(&c, 3 - me, opp);
    c.play(r, 3 - me);
    if DIRS.iter().any(|&d| c.line(r, 3 - me, d).0 >= c.win) {
        return -7;
    }
    let mut f = 0;
    let s2 = snapshot(&c, me, &mut f);
    if s2.cells.is_empty() {
        return 0;
    }
    let contradiction = s2.cells.iter().filter(|x| x.c_op == FIVE).count() >= 2
        && !s2.cells.iter().any(|x| x.c_me == FIVE);
    activation(before, balance(&s2), contradiction)
}

/// One game. The learner is `me`; black (1) moves first.
fn game(
    w: World,
    cfg: &Cfg,
    l: &mut Learner,
    seed: u64,
    me: u8,
    trace: &mut Vec<CausalEdge64>,
) -> Stats {
    let mut st = Stats {
        games: 1,
        ..Stats::default()
    };
    let mut rng = Rng(seed);
    let mut b = Board::new(w.n, w.win);
    // Seeded opening near the centre: 2..=4 stones, alternating colours.
    let k = 2 + rng.below(3);
    let mid = w.n as i32 / 2;
    let mut placed = 0;
    while placed < k {
        let r = mid - 2 + rng.below(5) as i32;
        let c = mid - 2 + rng.below(5) as i32;
        let i = (r * w.n as i32 + c) as usize;
        if b.cell[i] == 0 {
            b.play(i, 1 + (placed % 2) as u8);
            placed += 1;
        }
    }
    let mut turn = 1 + (k % 2) as u8;
    // (W slot, learner turn the step was taken on).
    let mut queue: std::collections::VecDeque<(u8, u64)> = std::collections::VecDeque::new();
    let horizon = match cfg.signal {
        Signal::Retained(h) => usize::from(h.max(1)),
        _ => 1,
    };
    let mut ring = [None::<Pending>; 64];
    let mut wslot = 0u8;
    let mut game_steps: Vec<Pending> = Vec::new();
    let mut result = 0i32;
    loop {
        if b.full() {
            break;
        }
        if turn != me {
            let i = opp_move(&b, turn, w.opp);
            b.play(i, turn);
            if DIRS.iter().any(|&d| b.line(i, turn, d).0 >= b.win) {
                result = -1;
                break;
            }
            turn = me;
            continue;
        }
        let mut folds = 0;
        let s = snapshot(&b, me, &mut folds);
        st.moves += 1;
        // Read the consequences of earlier reasoning steps through W.
        let contradiction = s.cells.iter().filter(|x| x.c_op == FIVE).count() >= 2
            && !s.cells.iter().any(|x| x.c_me == FIVE);
        st.contradictions += u64::from(contradiction);
        // A step is read once `horizon` learner turns have passed, forced
        // moves included.
        while queue
            .front()
            .is_some_and(|&(_, t)| st.moves - t >= horizon as u64)
        {
            let (slot, _) = queue.pop_front().unwrap();
            let Some(p) = ring[slot as usize].take() else {
                continue;
            };
            // Counted once per step, from the reading it is credited with.
            let retained = activation(p.before, balance(&s), contradiction);
            if retained > 0 {
                st.productive += 1;
            }
            let act = match cfg.signal {
                Signal::Retained(_) => Some(retained),
                Signal::Immediate => Some(p.immediate),
                Signal::Contradiction => contradiction.then_some(-7),
                Signal::Final => None,
            };
            if let (Some(a), true, Chooser::Learned) = (act, cfg.learn, cfg.chooser) {
                let epi = l.credit(cfg, &p, a);
                let expect = 2.0 * epi_expectation(l, cfg, &p) - 1.0;
                let surprise = ((f64::from(a) / 7.0 - expect).abs() * 3.5).round() as u8;
                trace.push(record(&p, slot, a, surprise, epi));
            }
        }
        let mv = match propagate(&s) {
            Some(i) => {
                st.mech += 1;
                i
            }
            None => {
                let key = match cfg.memory {
                    Memory::Structural => signature(&s),
                    Memory::Literal => b.hash(me),
                };
                let back = coarse(&s);
                let (op, hit) = l.choose(cfg, key, back, &mut rng);
                st.hits += u64::from(hit);
                let step_seed = rng.next();
                let mut r2 = Rng(step_seed);
                let mut f = 0;
                let mv = run_op(op, &b, me, &s, &mut r2, &mut f);
                st.steps += 1;
                st.folds += f;
                st.ops[op.index()] += 1;
                let k = op.index();
                if cfg.learn {
                    l.folds[k].0 += 1;
                    l.folds[k].1 += f;
                }
                if w.oracle {
                    let (reg, opt) = oracle_regret(&b, me, &s, k, w.opp, step_seed);
                    st.regret += i64::from(reg);
                    st.regret_n += 1;
                    st.optimal += u64::from(opt);
                }
                let mut after = b.clone();
                after.play(mv, me);
                let mut f2 = 0;
                let immediate = {
                    let si = snapshot(&after, me, &mut f2);
                    if si.cells.is_empty() {
                        0
                    } else {
                        activation(balance(&s), balance(&si), false)
                    }
                };
                let p = Pending {
                    key,
                    back,
                    op,
                    before: balance(&s),
                    immediate,
                    dir: s.cells.iter().find(|c| c.i == mv).map_or(0, |c| c.dir),
                    question: question(&s),
                };
                ring[wslot as usize] = Some(p);
                queue.push_back((wslot, st.moves));
                game_steps.push(p);
                wslot = (wslot + 1) % 64;
                mv
            }
        };
        b.play(mv, me);
        if DIRS.iter().any(|&d| b.line(mv, me, d).0 >= b.win) {
            result = 1;
            break;
        }
        turn = 3 - me;
    }
    // Steps still waiting are answered by the game's end.
    if result == -1 {
        st.contradictions += 1;
    }
    while let Some((slot, _)) = queue.pop_front() {
        let Some(p) = ring[slot as usize].take() else {
            continue;
        };
        let a = match result {
            1 => 7,
            -1 => -7,
            _ => 0,
        };
        let act = match cfg.signal {
            Signal::Retained(_) => Some(a),
            // The pre-reply reading was taken when the step was made.
            Signal::Immediate => Some(p.immediate),
            Signal::Contradiction => (result == -1).then_some(-7),
            Signal::Final => None,
        };
        if let (Some(a), true, Chooser::Learned) = (act, cfg.learn, cfg.chooser) {
            l.credit(cfg, &p, a);
        }
    }
    if let (Signal::Final, true, Chooser::Learned) = (cfg.signal, cfg.learn, cfg.chooser) {
        for p in &game_steps {
            l.credit(cfg, p, 7 * result);
        }
    }
    match result {
        1 => st.wins += 1,
        -1 => st.losses += 1,
        _ => {}
    }
    st
}

fn epi_expectation(l: &Learner, cfg: &Cfg, p: &Pending) -> f64 {
    let _ = cfg;
    l.stats
        .get(&p.key)
        .map_or(0.5, |row| row[p.op.index()].expectation())
}

// ── Runs ───────────────────────────────────────────────────────────────────

/// `episodes` games, alternating colour; windows of `win` games.
fn run(
    w: World,
    cfg: &Cfg,
    l: &mut Learner,
    seed: u64,
    episodes: usize,
    window: usize,
) -> Vec<Stats> {
    let mut out = Vec::new();
    let mut cur = Stats::default();
    let mut trace = Vec::new();
    for e in 0..episodes {
        if cfg.reset_per_game {
            *l = Learner::default();
        }
        let me = 1 + (e % 2) as u8;
        let g = game(
            w,
            cfg,
            l,
            seed.wrapping_mul(1_000_003).wrapping_add(e as u64),
            me,
            &mut trace,
        );
        cur.add(&g);
        trace.clear();
        if (e + 1) % window == 0 {
            out.push(std::mem::take(&mut cur));
        }
    }
    if cur.games > 0 {
        out.push(cur);
    }
    out
}

/// Pool runs window by window.
fn pool(per: &[Vec<Stats>]) -> Vec<Stats> {
    (0..per[0].len())
        .map(|k| total(&per.iter().map(|p| p[k].clone()).collect::<Vec<_>>()))
        .collect()
}

fn total(v: &[Stats]) -> Stats {
    let mut t = Stats::default();
    for s in v {
        t.add(s);
    }
    t
}

fn line(name: &str, s: &Stats) {
    println!(
        "  {name:<34} score {:.3}  W/L {:>4}/{:<4} folds/step {:>7.1}  regret {:.3}  opt {:.3}  prod {:.3}  hits {:.3}  top {:.2}",
        s.score(),
        s.wins,
        s.losses,
        s.folds_per_step(),
        s.mean_regret(),
        s.optimal_frac(),
        s.productive_frac(),
        s.hit_frac(),
        s.top_share()
    );
}

fn dist(s: &Stats) -> String {
    OPS.iter()
        .zip(s.ops)
        .map(|(o, n)| format!("{}:{:.2}", o.code(), n as f64 / s.steps.max(1) as f64))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let t0 = std::time::Instant::now();
    println!("D-MOORE-NARS-0: recipe learning on Gomoku\n");
    println!("Census of the 34 catalogue recipes:");
    let c = census();
    for (code, st, why) in c {
        println!("  {code:<5} {:<22} {why}", format!("{st:?}"));
    }
    let realized = c
        .iter()
        .filter(|x| matches!(x.1, Standing::Applicable(_)))
        .count();
    println!("  realized: {realized} of 34\n");

    let w9 = World {
        n: 9,
        win: 5,
        opp: Opp::Threat,
        oracle: true,
    };
    let seeds = [11u64, 22, 33];

    println!("Stage 1: fixed policies vs Threat (9x9, five), 3 seeds x 300 games");
    let mut fixed = Vec::new();
    for op in OPS {
        let cfg = Cfg {
            chooser: Chooser::Fixed(op),
            ..BASE
        };
        let s =
            total(&seeds.map(|sd| total(&run(w9, &cfg, &mut Learner::default(), sd, 300, 300))));
        line(op.code(), &s);
        fixed.push((op, s));
    }
    let uni = Cfg {
        chooser: Chooser::Uniform,
        ..BASE
    };
    let su = total(&seeds.map(|sd| total(&run(w9, &uni, &mut Learner::default(), sd, 300, 300))));
    line("uniform recipe", &su);

    println!("\nStage 1b: the same fixed policies vs Lookahead (2-ply), 3 x 100 games");
    let wl = World {
        opp: Opp::Lookahead,
        oracle: false,
        ..w9
    };
    for op in OPS {
        let cfg = Cfg {
            chooser: Chooser::Fixed(op),
            ..BASE
        };
        let s =
            total(&seeds.map(|sd| total(&run(wl, &cfg, &mut Learner::default(), sd, 100, 100))));
        line(op.code(), &s);
    }

    let episodes = 2000;
    let window = 400;
    println!("\nStage 2-3: learning vs Threat, {episodes} episodes, windows of {window}, pooled over 3 seeds");
    let arms: [(&str, Cfg); 10] = [
        ("learned, persistent (base)", BASE),
        (
            "learned, reset every game",
            Cfg {
                reset_per_game: true,
                ..BASE
            },
        ),
        (
            "retained over 3 turns (W h=3)",
            Cfg {
                signal: Signal::Retained(3),
                ..BASE
            },
        ),
        (
            "contradiction only",
            Cfg {
                signal: Signal::Contradiction,
                ..BASE
            },
        ),
        (
            "immediate (no W delay)",
            Cfg {
                signal: Signal::Immediate,
                ..BASE
            },
        ),
        (
            "final win/loss only",
            Cfg {
                signal: Signal::Final,
                ..BASE
            },
        ),
        (
            "literal board memory",
            Cfg {
                memory: Memory::Literal,
                ..BASE
            },
        ),
        (
            "cost-blind",
            Cfg {
                cost_weight: 0.0,
                ..BASE
            },
        ),
        (
            "decay 0.98",
            Cfg {
                decay: 0.98,
                ..BASE
            },
        ),
        ("uniform (no learning)", uni),
    ];
    for (name, cfg) in arms {
        let per: Vec<Vec<Stats>> = seeds
            .iter()
            .map(|&sd| run(w9, &cfg, &mut Learner::default(), sd, episodes, window))
            .collect();
        let pooled = pool(&per);
        println!("  {name}");
        for (k, s) in pooled.iter().enumerate() {
            line(&format!("  window {k}"), s);
        }
        let firsts: Vec<String> = per.iter().map(|p| format!("{:.3}", p[0].score())).collect();
        let lasts: Vec<String> = per
            .iter()
            .map(|p| format!("{:.3}", p.last().unwrap().score()))
            .collect();
        println!(
            "    per seed: first [{}]  last [{}]",
            firsts.join(" "),
            lasts.join(" ")
        );
        println!(
            "    recipes (last window): {}",
            dist(pooled.last().unwrap())
        );
    }

    println!(
        "\nWhat the base learner prefers, per structural signature (seed 11, {episodes} episodes):"
    );
    let mut l = Learner::default();
    run(w9, &BASE, &mut l, 11, episodes, episodes);
    let mut prefer = [0usize; NO];
    let mut confident = 0;
    for (key, row) in &l.stats {
        if key >> 40 == 1 {
            continue;
        }
        let k = (0..NO)
            .max_by(|&a, &b| row[a].expectation().total_cmp(&row[b].expectation()))
            .unwrap();
        prefer[k] += 1;
        confident += usize::from(row[k].pos + row[k].neg >= 3.0);
    }
    let shown: Vec<String> = OPS
        .iter()
        .zip(prefer)
        .map(|(o, n)| format!("{}:{n}", o.code()))
        .collect();
    println!(
        "  {} signatures, {confident} with >= 3 evidence on their favourite; favourite counts: {}",
        prefer.iter().sum::<usize>(),
        shown.join(" ")
    );

    println!(
        "\nStage 4: transfer. Train base on seeds A, freeze, evaluate on unseen openings / 11x11"
    );
    let mut trained = Learner::default();
    for sd in seeds {
        run(w9, &BASE, &mut trained, sd, episodes, episodes);
    }
    let frozen = Cfg {
        learn: false,
        ..BASE
    };
    let unseen = [101u64, 202, 303];
    let ft = total(&unseen.map(|sd| {
        let mut l = Learner {
            stats: trained.stats.clone(),
            folds: trained.folds,
        };
        total(&run(w9, &frozen, &mut l, sd, 300, 300))
    }));
    let fu =
        total(&unseen.map(|sd| total(&run(w9, &frozen, &mut Learner::default(), sd, 300, 300))));
    line("9x9 unseen, trained frozen", &ft);
    line("9x9 unseen, untrained frozen", &fu);
    let w11 = World { n: 11, ..w9 };
    let gt = total(&unseen.map(|sd| {
        let mut l = Learner {
            stats: trained.stats.clone(),
            folds: trained.folds,
        };
        total(&run(w11, &frozen, &mut l, sd, 150, 150))
    }));
    let gu =
        total(&unseen.map(|sd| total(&run(w11, &frozen, &mut Learner::default(), sd, 150, 150))));
    line("11x11, trained frozen", &gt);
    line("11x11, untrained frozen", &gu);

    for (opp, after_n, win_n) in [(Opp::Lookahead, 600, 150), (Opp::Aggressor, 600, 150)] {
        println!("\nStage 5: opponent switch Threat -> {opp:?} after {episodes} episodes");
        let wo = World {
            opp,
            oracle: false,
            ..w9
        };
        for (name, cfg) in [
            ("no decay", BASE),
            (
                "decay 0.98",
                Cfg {
                    decay: 0.98,
                    ..BASE
                },
            ),
        ] {
            let mut after = Vec::new();
            for sd in seeds {
                let mut l = Learner::default();
                run(w9, &cfg, &mut l, sd, episodes, episodes);
                after.push(run(wo, &cfg, &mut l, sd + 7, after_n, win_n));
            }
            let pooled = pool(&after);
            println!("  pretrained, {name}");
            for (k, s) in pooled.iter().enumerate() {
                line(&format!("  window {k}"), s);
                println!("      recipes: {}", dist(s));
            }
        }
        let fresh: Vec<Vec<Stats>> = seeds
            .iter()
            .map(|&sd| run(wo, &BASE, &mut Learner::default(), sd + 7, after_n, win_n))
            .collect();
        println!("  fresh learner");
        for (k, s) in pool(&fresh).iter().enumerate() {
            line(&format!("  window {k}"), s);
        }
        for op in [Op::Hpm, Op::Icr] {
            let f = total(&seeds.map(|sd| {
                total(&run(
                    wo,
                    &Cfg {
                        chooser: Chooser::Fixed(op),
                        ..BASE
                    },
                    &mut Learner::default(),
                    sd + 7,
                    after_n,
                    after_n,
                ))
            }));
            line(&format!("fixed {}", op.code()), &f);
        }
    }

    let _ = &fixed;
    println!("\nwall time {:.1}s", t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board(stones: &[(usize, usize, u8)]) -> Board {
        let mut b = Board::new(9, 5);
        for &(r, c, p) in stones {
            b.play(r * 9 + c, p);
        }
        b
    }

    fn wins(b: &Board, i: usize, p: u8) -> bool {
        let mut c = b.clone();
        c.play(i, p);
        DIRS.iter().any(|&d| c.line(i, p, d).0 >= c.win)
    }

    /// Positions where reasoning (not propagation) decides: played from the
    /// empty 9x9 board by a noisy threat player, stopped at a random depth.
    fn reasoning_corpus(count: usize) -> Vec<(Board, u8)> {
        let mut out = Vec::new();
        let mut rng = Rng(0x5EED_0001);
        while out.len() < count {
            let mut b = Board::new(9, 5);
            let depth = 4 + rng.below(30);
            let mut me = 1u8;
            let mut ok = true;
            for _ in 0..depth {
                let mut f = 0;
                let s = snapshot(&b, me, &mut f);
                if s.cells.is_empty() {
                    ok = false;
                    break;
                }
                let i = if rng.below(10) < 7 {
                    opp_move(&b, me, Opp::Threat)
                } else {
                    s.cells[rng.below(s.cells.len())].i
                };
                b.play(i, me);
                if DIRS.iter().any(|&d| b.line(i, me, d).0 >= b.win) {
                    ok = false;
                    break;
                }
                me = 3 - me;
            }
            if !ok {
                continue;
            }
            let mut f = 0;
            let s = snapshot(&b, me, &mut f);
            if s.cells.len() >= 2 && propagate(&s).is_none() {
                out.push((b, me));
            }
        }
        out
    }

    /// D-LAB-P1: the certificate never changes the pick, and it saves folds.
    #[test]
    fn certified_minimax_picks_what_minimax_picks_with_fewer_folds() {
        let corpus = reasoning_corpus(400);
        // Measured, deterministic: (k1, k2, full folds, certified folds).
        const PINNED: [(usize, usize, u64, u64); 3] = [
            (4, 4, 442_720, 282_480),
            (5, 5, 665_076, 367_344),
            (8, 8, 1_601_258, 624_386),
        ];
        for (k1, k2, pin_full, pin_cert) in PINNED {
            let (mut full, mut cert, mut cut_positions) = (0u64, 0u64, 0usize);
            for (b, me) in &corpus {
                let mut f0 = 0;
                let s = snapshot(b, *me, &mut f0);
                let (mut ff, mut fc) = (0u64, 0u64);
                let a = minimax(b, *me, &s, k1, k2, &mut ff);
                let c = minimax_certified(b, *me, &s, k1, k2, &mut fc);
                assert_eq!(a, c, "the certificate changed the pick (k {k1}/{k2})");
                assert!(fc <= ff);
                if fc < ff {
                    cut_positions += 1;
                }
                full += ff;
                cert += fc;
            }
            eprintln!(
                "P1 k {k1}/{k2}: folds {full} -> {cert} ({:.1}% saved), cut on {cut_positions}/{} positions",
                100.0 * (1.0 - cert as f64 / full as f64),
                corpus.len()
            );
            assert_eq!((full, cert), (pin_full, pin_cert), "k {k1}/{k2}");
        }
    }

    #[test]
    fn census_covers_the_catalogue_exactly() {
        let c = census();
        let codes: Vec<&str> = RECIPES.iter().map(|r| r.code).collect();
        let ours: Vec<&str> = c.iter().map(|x| x.0).collect();
        assert_eq!(ours, codes, "one census row per catalogue recipe, in order");
        let realized: Vec<Op> = c
            .iter()
            .filter_map(|x| match x.1 {
                Standing::Applicable(op) => Some(op),
                _ => None,
            })
            .collect();
        assert_eq!(realized.len(), NO);
        for op in OPS {
            assert_eq!(realized.iter().filter(|&&o| o == op).count(), 1, "{op:?}");
            let row = c.iter().find(|x| x.1 == Standing::Applicable(op)).unwrap();
            assert_eq!(row.0, op.code(), "an operator carries its catalogue code");
        }
    }

    #[test]
    fn propagation_takes_the_win_before_the_block() {
        let mine: Vec<_> = (1..5).map(|c| (4, c, 1)).collect();
        let theirs: Vec<_> = (1..5).map(|c| (1, c, 2)).collect();
        let mut f = 0;
        // Own four only: the win.
        let b = board(&[mine.clone(), vec![(7, 7, 2)]].concat());
        let i = propagate(&snapshot(&b, 1, &mut f)).expect("a win is mechanical");
        assert!(wins(&b, i, 1));
        // Opponent four only: the block.
        let b = board(&[theirs.clone(), vec![(7, 7, 1)]].concat());
        let i = propagate(&snapshot(&b, 1, &mut f)).expect("a forced block is mechanical");
        assert!(
            wins(&b, i, 2),
            "the block stands where the opponent would win"
        );
        // Both: the win, never the block.
        let b = board(&[mine, theirs].concat());
        let i = propagate(&snapshot(&b, 1, &mut f)).unwrap();
        assert!(wins(&b, i, 1));
        // A quiet position leaves reasoning to do.
        let b = board(&[(4, 4, 1), (4, 5, 2), (5, 5, 1)]);
        assert_eq!(propagate(&snapshot(&b, 2, &mut f)), None);
    }

    #[test]
    fn the_register_never_names_the_recipe() {
        let p = |op| Pending {
            key: 1,
            back: 2,
            op,
            before: 0,
            immediate: 0,
            dir: 2,
            question: CausalMask::SO,
        };
        let epi = Ev::default().state(Topology2::Direct);
        for a in [-7, 0, 3] {
            let r: Vec<u64> = OPS
                .iter()
                .map(|&op| record(&p(op), 5, a, 1, epi).0)
                .collect();
            assert!(r.windows(2).all(|x| x[0] == x[1]), "activation {a}");
            assert_eq!(CausalEdge64(r[0]).inference_mantissa(), a.max(-7) as i8);
        }
    }

    #[test]
    fn w_follows_the_pending_slots() {
        let w = World {
            n: 9,
            win: 5,
            opp: Opp::Threat,
            oracle: false,
        };
        let mut l = Learner::default();
        let mut trace = Vec::new();
        for s in 0..6 {
            game(w, &BASE, &mut l, s, 1 + (s % 2) as u8, &mut trace);
            let ws: Vec<u8> = trace.iter().map(|e| e.w_slot()).collect();
            for (k, x) in ws.iter().enumerate() {
                assert_eq!(
                    *x as usize,
                    k % 64,
                    "credit reads the slot written one step earlier"
                );
            }
            trace.clear();
        }
    }

    /// The later games of a learning run, pooled over two seeds.
    fn late(cfg: &Cfg) -> Stats {
        let w = World {
            n: 9,
            win: 5,
            opp: Opp::Threat,
            oracle: false,
        };
        let per: Vec<Vec<Stats>> = [11u64, 22]
            .iter()
            .map(|&sd| run(w, cfg, &mut Learner::default(), sd, 900, 300))
            .collect();
        let p = pool(&per);
        total(&p[1..])
    }

    #[test]
    fn persistent_structural_learning_beats_its_controls() {
        let base = late(&BASE);
        let reset = late(&Cfg {
            reset_per_game: true,
            ..BASE
        });
        let hpm = late(&Cfg {
            chooser: Chooser::Fixed(Op::Hpm),
            ..BASE
        });
        let literal = late(&Cfg {
            memory: Memory::Literal,
            ..BASE
        });
        let immediate = late(&Cfg {
            signal: Signal::Immediate,
            ..BASE
        });
        for (n, s) in [
            ("base", &base),
            ("reset", &reset),
            ("hpm", &hpm),
            ("literal", &literal),
            ("immediate", &immediate),
        ] {
            println!(
                "{n:<10} score {:.3} hits {:.3} top {:.2} {}",
                s.score(),
                s.hit_frac(),
                s.top_share(),
                dist(s)
            );
        }
        // Experience improves play over the best single recipe and over a
        // learner that forgets between games (measured 0.528 / 0.500 / 0.467).
        assert!(
            base.score() >= hpm.score() + 0.015,
            "base {:.3} hpm {:.3}",
            base.score(),
            hpm.score()
        );
        assert!(
            base.score() >= reset.score() + 0.03,
            "base {:.3} reset {:.3}",
            base.score(),
            reset.score()
        );
        // It reuses structure, and does not collapse onto one recipe.
        assert!(base.hit_frac() > 0.9);
        assert!(base.top_share() < 0.7, "top share {:.2}", base.top_share());
        // Literal boards almost never recur, so they teach nothing.
        assert!(
            literal.hit_frac() < 0.05,
            "literal hits {:.3}",
            literal.hit_frac()
        );
        assert!(base.score() >= literal.score() + 0.015);
        // Crediting before the opponent answers leaves the learner on HPM:
        // the delayed consequence read through W is what moves it.
        assert!(
            immediate.top_share() > 0.9,
            "immediate top {:.2}",
            immediate.top_share()
        );
        assert!(base.score() >= immediate.score() + 0.015);
    }

    #[test]
    fn a_frozen_evaluation_leaves_the_learner_unchanged() {
        let w = World {
            n: 9,
            win: 5,
            opp: Opp::Threat,
            oracle: false,
        };
        let mut l = Learner::default();
        run(w, &BASE, &mut l, 3, 40, 40);
        let snap = |l: &Learner| {
            let mut v: Vec<_> = l
                .stats
                .iter()
                .map(|(k, r)| (*k, format!("{r:?}")))
                .collect();
            v.sort();
            (v, l.folds)
        };
        let before = snap(&l);
        assert!(l.folds.iter().any(|f| f.0 > 0), "training ran recipes");
        run(
            w,
            &Cfg {
                learn: false,
                ..BASE
            },
            &mut l,
            4,
            40,
            40,
        );
        assert_eq!(snap(&l), before);
    }

    #[test]
    fn replay_is_deterministic() {
        let w = World {
            n: 9,
            win: 5,
            opp: Opp::Threat,
            oracle: true,
        };
        let a = run(w, &BASE, &mut Learner::default(), 5, 20, 20);
        let b = run(w, &BASE, &mut Learner::default(), 5, 20, 20);
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }
}
