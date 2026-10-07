//! D-PUZZLE-0, step 2: crosswords on the same population algebra as Sudoku.
//!
//! Everything domain-free (the propagation reading, the five questions, the
//! folds, the oracle) comes unchanged from `shared/population_fold.rs`, the
//! module the Sudoku probe uses. This file holds only the crossword: its
//! claims, its law, its own counters. If a crossword question needed code in
//! the shared half, "same algebra" would be false; the shared module's fence
//! test is what would catch that.
//!
//! # The corpus
//!
//! A 5×5 grid with two black squares and ten crossing slots:
//!
//! ```text
//! . . . . #
//! . . . . .
//! . . . . .
//! . . . . .
//! # . . . .
//! ```
//!
//! Each instance fills the 23 white squares with random letters from a small
//! alphabet; the slot words of that fill are the solution. The dictionary is
//! those words plus random distractors of each length. A few slots are
//! given. An instance is kept only if the dictionary admits exactly one fill
//! consistent with the givens (checked by backtracking), so "the true word" is
//! well defined. Each instance is snapshotted after `k` forced placements.
//!
//! # The law (the only domain logic)
//!
//! A slot's candidates are the dictionary words of its length that agree with
//! every letter already fixed by a placed word. A slot with exactly one
//! candidate is forced (lowest slot first) and fixes its letters. This is the
//! crossword's analogue of a naked single.
//!
//! # The claims
//!
//! Claim = "slot s holds word w", one edge per claim still consistent, in the
//! shared propagation reading: a given slot is `GIVEN`, a forced placement
//! `FORCED`, the true word of a slot not yet placed `ENTAILED`, every other
//! surviving candidate `CANDIDATE`. Class `0x0907` is this probe's own.
//!
//! Run: `cargo run --release -p cognitive-shader-driver --example crossword_population_fold_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example crossword_population_fold_probe`

use std::time::Instant;

use lance_graph_contract::class_view::ClassId;

#[path = "shared/population_fold.rs"]
mod population_fold;
use population_fold::{
    admit, check_three_ways, declarations, per_group, report, Lane, Rng, CANDIDATE, ENTAILED,
    FORCED, GIVEN,
};

/// The probe's own class (no production class declares a crossword reading).
const CROSSWORD_CLASS: ClassId = 0x0907;

const SIDE: usize = 5;
const TEMPLATE: [&str; SIDE] = ["....#", ".....", ".....", ".....", "#...."];
const ALPHABET: u8 = 3;
const DISTRACTORS_PER_LENGTH: usize = 10;
const GIVEN_SLOTS: usize = 2;

type Word = Vec<u8>;

/// A slot is the ordered list of the squares it covers.
fn slots() -> Vec<Vec<usize>> {
    let white = |r: usize, c: usize| TEMPLATE[r].as_bytes()[c] == b'.';
    let mut out = Vec::new();
    for across in [true, false] {
        for line in 0..SIDE {
            let mut run = Vec::new();
            for k in 0..=SIDE {
                let (r, c) = if across { (line, k) } else { (k, line) };
                if k < SIDE && white(r, c) {
                    run.push(r * SIDE + c);
                } else {
                    if run.len() >= 2 {
                        out.push(run.clone());
                    }
                    run.clear();
                }
            }
        }
    }
    out
}

/// One instance: the dictionary (deduplicated), the given slots, and the
/// solution word of every slot.
struct Instance {
    dictionary: Vec<Word>,
    given: Vec<bool>,
    solution: Vec<Word>,
}

/// Letters fixed so far (`None` = open).
type Fill = [Option<u8>; SIDE * SIDE];

fn fits(word: &[u8], slot: &[usize], fill: &Fill) -> bool {
    slot.iter()
        .zip(word)
        .all(|(&sq, &l)| fill[sq].is_none_or(|f| f == l))
}

fn place(word: &[u8], slot: &[usize], fill: &mut Fill) {
    for (&sq, &l) in slot.iter().zip(word) {
        fill[sq] = Some(l);
    }
}

fn candidates<'a>(inst: &'a Instance, slot: &[usize], fill: &Fill) -> Vec<&'a Word> {
    inst.dictionary
        .iter()
        .filter(|w| w.len() == slot.len() && fits(w, slot, fill))
        .collect()
}

/// Number of complete fills consistent with the givens, counted up to `cap`.
fn count_solutions(inst: &Instance, slots: &[Vec<usize>], cap: usize) -> usize {
    fn go(
        inst: &Instance,
        slots: &[Vec<usize>],
        i: usize,
        fill: &mut Fill,
        n: &mut usize,
        cap: usize,
    ) {
        if *n >= cap {
            return;
        }
        if i == slots.len() {
            *n += 1;
            return;
        }
        for w in candidates(inst, &slots[i], fill) {
            let saved = *fill;
            place(w, &slots[i], fill);
            go(inst, slots, i + 1, fill, n, cap);
            *fill = saved;
        }
    }
    let mut fill: Fill = [None; SIDE * SIDE];
    for (s, slot) in slots.iter().enumerate() {
        if inst.given[s] {
            place(&inst.solution[s], slot, &mut fill);
        }
    }
    let mut n = 0;
    go(inst, slots, 0, &mut fill, &mut n, cap);
    n
}

/// Draw instances until one has exactly one solution.
fn draw_instance(rng: &mut Rng, slots: &[Vec<usize>]) -> Instance {
    loop {
        let squares: Vec<u8> = (0..SIDE * SIDE)
            .map(|_| rng.below(u64::from(ALPHABET)) as u8)
            .collect();
        let solution: Vec<Word> = slots
            .iter()
            .map(|s| s.iter().map(|&sq| squares[sq]).collect())
            .collect();
        let mut dictionary = solution.clone();
        let mut lengths: Vec<usize> = slots.iter().map(Vec::len).collect();
        lengths.sort_unstable();
        lengths.dedup();
        for len in lengths {
            for _ in 0..DISTRACTORS_PER_LENGTH {
                dictionary.push(
                    (0..len)
                        .map(|_| rng.below(u64::from(ALPHABET)) as u8)
                        .collect(),
                );
            }
        }
        dictionary.sort();
        dictionary.dedup();
        let mut order: Vec<usize> = (0..slots.len()).collect();
        rng.shuffle(&mut order);
        let mut given = vec![false; slots.len()];
        for &s in &order[..GIVEN_SLOTS] {
            given[s] = true;
        }
        let inst = Instance {
            dictionary,
            given,
            solution,
        };
        if count_solutions(&inst, slots, 2) == 1 {
            return inst;
        }
    }
}

/// The givens, then at most `depth` forced placements. Returns which slots are
/// placed (`Some(true)` given, `Some(false)` forced) and the fill.
fn propagate(inst: &Instance, slots: &[Vec<usize>], depth: usize) -> (Vec<Option<bool>>, Fill) {
    let mut placed: Vec<Option<bool>> = vec![None; slots.len()];
    let mut fill: Fill = [None; SIDE * SIDE];
    for (s, slot) in slots.iter().enumerate() {
        if inst.given[s] {
            place(&inst.solution[s], slot, &mut fill);
            placed[s] = Some(true);
        }
    }
    let mut forced = 0;
    while forced < depth {
        let next = (0..slots.len()).find_map(|s| {
            if placed[s].is_some() {
                return None;
            }
            let c = candidates(inst, &slots[s], &fill);
            (c.len() == 1).then(|| (s, c[0].clone()))
        });
        let Some((s, w)) = next else { break };
        place(&w, &slots[s], &mut fill);
        placed[s] = Some(false);
        forced += 1;
    }
    (placed, fill)
}

/// About `min_edges` claim edges; each instance is one group.
fn build_lane(min_edges: usize, seed: u64) -> Lane {
    let slots = slots();
    assert_eq!(slots.len(), 10);
    let mut rng = Rng(seed);
    let mut lane = Lane::default();
    while lane.edges.len() < min_edges {
        let inst = draw_instance(&mut rng, &slots);
        let depth = rng.below(slots.len() as u64 - GIVEN_SLOTS as u64 + 1) as usize;
        let (placed, fill) = propagate(&inst, &slots, depth);
        for (s, slot) in slots.iter().enumerate() {
            match placed[s] {
                Some(true) => {
                    lane.push(GIVEN);
                    lane.expected[0] += 1;
                }
                Some(false) => {
                    // A forced word is the true word: the instance is unique.
                    assert_eq!(
                        slot.iter().map(|&sq| fill[sq].unwrap()).collect::<Word>(),
                        inst.solution[s]
                    );
                    lane.push(FORCED);
                    lane.expected[1] += 1;
                }
                None => {
                    let c = candidates(&inst, slot, &fill);
                    // The law never rules out the true word.
                    assert!(c.contains(&&inst.solution[s]));
                    lane.push(ENTAILED);
                    lane.expected[2] += 1;
                    for _ in 1..c.len() {
                        lane.push(CANDIDATE);
                        lane.expected[3] += 1;
                    }
                }
            }
        }
        lane.close_group();
    }
    lane
}

fn main() {
    let decl = declarations(CROSSWORD_CLASS);
    admit(&decl, CROSSWORD_CLASS).expect("the crossword class declares the canonical reading");
    let t = Instant::now();
    let lane = build_lane(1_000_000, 0xC0_55_u64);
    println!("D-PUZZLE-0 / crosswords on the same algebra");
    println!(
        "  lane: {} edges from {} unique-solution instances, built in {:.2?}",
        lane.edges.len(),
        lane.groups,
        t.elapsed()
    );
    let counts = check_three_ways(&lane, &decl, CROSSWORD_CLASS);
    assert!(
        per_group(&lane, GIVEN.bit() | FORCED.bit() | ENTAILED.bit())
            .iter()
            .all(|&n| n == 10)
    );
    println!("  every instance: exactly 10 asserted claims, one per slot (group fold)");
    report(&lane, &decl, CROSSWORD_CLASS, &counts);
}

#[cfg(test)]
mod tests {
    use super::population_fold::{count_in, queries, UNKNOWN_CAUSES};
    use super::*;
    use lance_graph_contract::epistemic_state5::fact::{CAUSES, DIRECT, IND_KNOWN};
    use lance_graph_contract::epistemic_state5::facts_population;

    fn small_lane() -> Lane {
        build_lane(20_000, 0x7E57)
    }

    /// The template has the ten slots drawn in the module doc, and every white
    /// square sits in exactly two of them.
    #[test]
    fn the_template_has_ten_crossing_slots() {
        let slots = slots();
        let lens: Vec<usize> = slots.iter().map(Vec::len).collect();
        assert_eq!(lens, [4, 5, 5, 5, 4, 4, 5, 5, 5, 4]);
        let mut cover = [0u8; SIDE * SIDE];
        for s in &slots {
            for &sq in s {
                cover[sq] += 1;
            }
        }
        let white = cover.iter().filter(|&&n| n > 0).count();
        assert_eq!(white, 23);
        assert!(cover.iter().all(|&n| n == 0 || n == 2));
    }

    /// Every count three ways, each equal to the instance's own counters, and
    /// every question actually exercised.
    #[test]
    fn every_question_counts_the_same_three_ways() {
        let lane = small_lane();
        let counts = check_three_ways(&lane, &declarations(CROSSWORD_CLASS), CROSSWORD_CLASS);
        assert!(counts.iter().all(|&n| n > 0), "a question went unexercised");
    }

    /// Same partition as Sudoku: disjoint states covering every edge, and no
    /// `Unknown × Causes`.
    #[test]
    fn the_four_states_partition_the_lane() {
        let lane = small_lane();
        let union = queries()[1..].iter().fold(0, |u, q| u | q.population);
        assert_eq!(count_in(&lane.edges, union), lane.edges.len());
        assert_eq!(count_in(&lane.edges, UNKNOWN_CAUSES.bit()), 0);
    }

    /// Every instance keeps exactly one asserted claim per slot and its given
    /// slots, whatever its depth; the depths actually vary.
    #[test]
    fn the_group_fold_sees_every_instance_whole() {
        let lane = small_lane();
        assert!(per_group(&lane, facts_population(CAUSES))
            .iter()
            .all(|&n| n == 10));
        assert!(per_group(&lane, facts_population(DIRECT | CAUSES))
            .iter()
            .all(|&n| n as usize == GIVEN_SLOTS));
        let forced = per_group(&lane, facts_population(IND_KNOWN | CAUSES));
        assert_eq!(*forced.iter().min().unwrap(), 0);
        assert!(*forced.iter().max().unwrap() >= 5, "forcing never got deep");
    }

    /// Every kept instance has exactly one solution, and the generator does
    /// reject ambiguous ones (otherwise the check would be decoration).
    #[test]
    fn instances_are_unique_and_ambiguous_ones_are_rejected() {
        let slots = slots();
        let mut rng = Rng(9);
        for _ in 0..50 {
            assert_eq!(
                count_solutions(&draw_instance(&mut rng, &slots), &slots, 3),
                1
            );
        }
        // With no givens and a wide dictionary, ambiguity is common.
        let mut rejected = 0;
        for _ in 0..50 {
            let mut inst = draw_instance(&mut rng, &slots);
            inst.given = vec![false; slots.len()];
            if count_solutions(&inst, &slots, 2) > 1 {
                rejected += 1;
            }
        }
        assert!(rejected > 0, "uniqueness never bit");
    }

    /// The gate is the declaration, for this domain's class as for Sudoku's.
    #[test]
    fn an_undeclared_lane_is_refused_before_any_edge() {
        let decl = declarations(CROSSWORD_CLASS);
        assert!(admit(&decl, CROSSWORD_CLASS).is_some());
        assert!(admit(&decl, 0x0906).is_none());
    }
}
