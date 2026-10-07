//! D-PUZZLE-0, step 2b: crosswords filled with real words.
//!
//! Step 2 (`crossword_population_fold_probe`) used a synthetic three-letter
//! alphabet. This probe fills the same 5×5 template from a real word list:
//!
//! - **English** (default, always available): the two vocabularies DeepNSM
//!   reads. v1's 4096-word table (`word_rank_lookup.csv`, ranks 1..=4096) and
//!   v2's academic list (`academic_20k.csv`, 18,559 distinct words), both
//!   committed under `crates/deepnsm/word_frequency/`. Their union, alphabetic
//!   ASCII only, scored by the larger COCA frequency. No other English source.
//! - **German** (only with `DEREKO_PATH` set): the 20,000 most frequent forms
//!   of the DeReKo-2014 STTS list (`form, lemma, tag, freq`, tab-separated),
//!   after excluding proper nouns (`NE`) and non-alphabetic forms and summing
//!   frequencies per lowercase form. DeReKo is CC BY-NC 3.0 (IDS Mannheim); it
//!   is read from the given path at run time and nothing derived from it is
//!   committed. Without the variable the German run is skipped, never failed.
//!
//! The dictionary is every word of a slot's length in that vocabulary. An
//! instance is a random fill of the template from that dictionary; given
//! slots are added in random order until the dictionary admits exactly one
//! fill consistent with them, so "the true word" is well defined. The law,
//! the claims and every question are the step-2 ones; the fold, the reading
//! and the oracle come unchanged from `shared/population_fold.rs`.
//!
//! Run: `cargo run --release -p cognitive-shader-driver --example crossword_real_words_probe`
//! German too: `DEREKO_PATH=/path/DeReKo-2014-II-MainArchive-STT.100000.freq cargo run ...`
//! Tests: `cargo test -p cognitive-shader-driver --example crossword_real_words_probe`

use std::collections::HashMap;
use std::time::Instant;

use lance_graph_contract::class_view::ClassId;

#[path = "shared/population_fold.rs"]
mod population_fold;
use population_fold::{
    admit, check_three_ways, declarations, per_group, report, Lane, Rng, CANDIDATE, ENTAILED,
    FORCED, GIVEN,
};

/// Same class as step 2: the same domain, a different dictionary.
const CROSSWORD_CLASS: ClassId = 0x0907;

/// The step-2 template: two black squares, ten crossing slots.
const TEMPLATE: [&str; 5] = ["....#", ".....", ".....", ".....", "#...."];
/// NYT grid rules: shortest word, and the most black squares (a sixth).
const MIN_WORD: usize = 3;
const MAX_BLACK_DIVISOR: usize = 6;
/// Side of the random NYT-rule grids.
const NYT_SIDE: usize = 7;
/// v1's vocabulary size (`deepnsm::vocabulary::VOCAB_SIZE`).
const V1_VOCAB: u32 = 4096;
/// Forms kept from the German frequency list.
const GERMAN_VOCAB: usize = 20_000;
/// Search nodes allowed for one random fill before a fresh start.
const FILL_BUDGET: usize = 20_000;

const RANK_LOOKUP: &str = include_str!("../../deepnsm/word_frequency/word_rank_lookup.csv");
const ACADEMIC: &str = include_str!("../../deepnsm/word_frequency/academic_20k.csv");

type Word = Vec<char>;
/// Letters fixed so far, one per square (`None` = open or black).
type Fill = Vec<Option<char>>;

/// A square grid: its side and which squares are white.
#[derive(Clone, Debug, PartialEq)]
struct Grid {
    side: usize,
    white: Vec<bool>,
}

impl Grid {
    fn template() -> Self {
        let white = TEMPLATE
            .iter()
            .flat_map(|r| r.bytes().map(|b| b == b'.'))
            .collect();
        Self {
            side: TEMPLATE.len(),
            white,
        }
    }

    fn cells(&self) -> usize {
        self.side * self.side
    }

    /// Every maximal white run, across then down.
    fn runs(&self) -> Vec<Vec<usize>> {
        let n = self.side;
        let mut out = Vec::new();
        for across in [true, false] {
            for line in 0..n {
                let mut run = Vec::new();
                for k in 0..=n {
                    let (r, c) = if across { (line, k) } else { (k, line) };
                    if k < n && self.white[r * n + c] {
                        run.push(r * n + c);
                    } else {
                        if !run.is_empty() {
                            out.push(run.clone());
                        }
                        run.clear();
                    }
                }
            }
        }
        out
    }

    /// The word slots: runs of two or more squares.
    fn slots(&self) -> Vec<Vec<usize>> {
        self.runs().into_iter().filter(|r| r.len() >= 2).collect()
    }

    fn blacks(&self) -> usize {
        self.white.iter().filter(|&&w| !w).count()
    }

    /// 180° rotational symmetry: square `i` and square `cells - 1 - i` agree.
    fn symmetric(&self) -> bool {
        let n = self.cells();
        (0..n).all(|i| self.white[i] == self.white[n - 1 - i])
    }

    /// The white squares form one 4-connected region.
    fn connected(&self) -> bool {
        let n = self.side;
        let Some(start) = self.white.iter().position(|&w| w) else {
            return false;
        };
        let mut seen = vec![false; self.cells()];
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(i) = stack.pop() {
            let (r, c) = (i / n, i % n);
            let up = (r > 0).then(|| i - n);
            let down = (r + 1 < n).then(|| i + n);
            let left = (c > 0).then(|| i - 1);
            let right = (c + 1 < n).then(|| i + 1);
            for j in [up, down, left, right].into_iter().flatten() {
                if self.white[j] && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        (0..self.cells()).all(|i| !self.white[i] || seen[i])
    }

    /// The NYT grid rules: rotational symmetry, every word at least
    /// `MIN_WORD` letters (so every white square is checked), one connected
    /// white region, at most a sixth of the squares black.
    fn is_nyt_valid(&self) -> bool {
        self.symmetric()
            && self.runs().iter().all(|r| r.len() >= MIN_WORD)
            && self.connected()
            && self.blacks() * MAX_BLACK_DIVISOR <= self.cells()
    }

    /// A random grid obeying the NYT rules: symmetric black-square pairs drawn
    /// until the grid is valid and has at least one black square.
    fn random_nyt(side: usize, rng: &mut Rng) -> Self {
        let cells = side * side;
        loop {
            let mut white = vec![true; cells];
            for i in 0..cells.div_ceil(2) {
                if rng.below(8) == 0 {
                    white[i] = false;
                    white[cells - 1 - i] = false;
                }
            }
            let g = Self { side, white };
            if g.blacks() > 0 && g.is_nyt_valid() {
                return g;
            }
        }
    }
}

/// Where each instance's grid comes from.
#[derive(Clone, Copy, Debug)]
enum Grids {
    /// The fixed step-2 template.
    Template,
    /// A fresh random NYT-rule grid of this side per instance.
    Nyt(usize),
}

impl Grids {
    fn draw(self, rng: &mut Rng) -> Grid {
        match self {
            Grids::Template => Grid::template(),
            Grids::Nyt(side) => Grid::random_nyt(side, rng),
        }
    }

    /// Every slot length an instance can need.
    fn lengths(self) -> Vec<usize> {
        match self {
            Grids::Template => {
                let mut l: Vec<usize> = Grid::template().slots().iter().map(Vec::len).collect();
                l.sort_unstable();
                l.dedup();
                l
            }
            Grids::Nyt(side) => (MIN_WORD..=side).collect(),
        }
    }
}

/// Insert `word` with `freq`, keeping the larger frequency, if it is
/// alphabetic ASCII.
fn keep_max(out: &mut HashMap<String, f64>, word: &str, freq: &str) {
    let w = word.trim().to_lowercase();
    if w.is_empty() || !w.chars().all(|c| c.is_ascii_alphabetic()) {
        return;
    }
    let n: f64 = freq.trim().parse().unwrap_or(0.0);
    let e = out.entry(w).or_insert(0.0);
    *e = e.max(n);
}

/// DeepNSM's English vocabulary: v1's 4096 (`rank,word,pos,freq`, rank cut)
/// united with v2's academic list (`ID,band,status,word,Pos,COCA-All,...`).
fn english_vocabulary() -> HashMap<String, f64> {
    let mut out = HashMap::new();
    for line in RANK_LOOKUP.lines().skip(1) {
        let f: Vec<&str> = line.trim_end_matches('\r').split(',').collect();
        let [rank, word, _, freq] = f.as_slice() else {
            continue;
        };
        if rank.trim().parse::<u32>().is_ok_and(|r| r <= V1_VOCAB) {
            keep_max(&mut out, word, freq);
        }
    }
    for line in ACADEMIC.lines().skip(1) {
        let f: Vec<&str> = line.trim_end_matches('\r').split(',').collect();
        if let (Some(word), Some(freq)) = (f.get(3), f.get(5)) {
            keep_max(&mut out, word, freq);
        }
    }
    out
}

/// DeReKo forms: alphabetic (umlauts and ß included), lowercase, proper nouns
/// excluded, frequencies summed over lemma/tag rows.
fn dereko_frequencies(text: &str) -> HashMap<String, f64> {
    let mut out: HashMap<String, f64> = HashMap::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let [form, _, tag, freq] = f.as_slice() else {
            continue;
        };
        if *tag == "NE" {
            continue;
        }
        let w = form.to_lowercase();
        if w.is_empty() || !w.chars().all(char::is_alphabetic) {
            continue;
        }
        *out.entry(w).or_insert(0.0) += freq.trim().parse::<f64>().unwrap_or(0.0);
    }
    out
}

/// Crossword answers from the Down-and-Across clue/answer files
/// (`*.target`, one answer per line, multi-word answers space-split by the
/// dataset's own post-processing). Spaces are removed to recover the grid
/// string; each answer is scored by how often it occurs.
fn xword_answers(targets: &[String]) -> HashMap<String, f64> {
    let mut out: HashMap<String, f64> = HashMap::new();
    for text in targets {
        for line in text.lines() {
            let w: String = line
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>()
                .to_lowercase();
            if w.is_empty() || !w.chars().all(|c| c.is_ascii_alphabetic()) {
                continue;
            }
            *out.entry(w).or_insert(0.0) += 1.0;
        }
    }
    out
}

/// The `GERMAN_VOCAB` most frequent DeReKo forms (ties alphabetical).
fn german_vocabulary(text: &str) -> HashMap<String, f64> {
    let f = dereko_frequencies(text);
    let mut ranked: Vec<(String, f64)> = f.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(GERMAN_VOCAB);
    ranked.into_iter().collect()
}

/// The `top` most frequent words of one length, ties broken alphabetically
/// so the list is deterministic.
fn top_words(freq: &HashMap<String, f64>, len: usize, top: usize) -> Vec<Word> {
    let mut ws: Vec<(&String, f64)> = freq
        .iter()
        .filter(|(w, _)| w.chars().count() == len)
        .map(|(w, &f)| (w, f))
        .collect();
    ws.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    ws.into_iter()
        .take(top)
        .map(|(w, _)| w.chars().collect())
        .collect()
}

/// One length's words, indexed by (position, letter) as a bitset over words.
struct Bucket {
    words: Vec<Word>,
    index: Vec<HashMap<char, Vec<u64>>>,
}

impl Bucket {
    fn new(words: Vec<Word>, len: usize) -> Self {
        let blocks = words.len().div_ceil(64);
        let mut index = vec![HashMap::new(); len];
        for (i, w) in words.iter().enumerate() {
            for (p, &ch) in w.iter().enumerate() {
                let bits = index[p].entry(ch).or_insert_with(|| vec![0u64; blocks]);
                bits[i / 64] |= 1 << (i % 64);
            }
        }
        Self { words, index }
    }

    /// Indices of the words agreeing with every letter fixed in `slot`.
    fn candidates(&self, slot: &[usize], fill: &Fill) -> Vec<usize> {
        let blocks = self.words.len().div_ceil(64);
        let mut acc = vec![u64::MAX; blocks];
        if !self.words.len().is_multiple_of(64) {
            acc[blocks - 1] = (1u64 << (self.words.len() % 64)) - 1;
        }
        for (p, &sq) in slot.iter().enumerate() {
            if let Some(ch) = fill[sq] {
                match self.index[p].get(&ch) {
                    Some(bits) => acc.iter_mut().zip(bits).for_each(|(a, b)| *a &= b),
                    None => return Vec::new(),
                }
            }
        }
        let mut out = Vec::new();
        for (b, &word) in acc.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                out.push(b * 64 + w.trailing_zeros() as usize);
                w &= w - 1;
            }
        }
        out
    }
}

struct Dictionary {
    by_len: HashMap<usize, Bucket>,
}

impl Dictionary {
    fn new(freq: &HashMap<String, f64>, lengths: &[usize]) -> Self {
        let by_len = lengths
            .iter()
            .copied()
            .map(|l| (l, Bucket::new(top_words(freq, l, usize::MAX), l)))
            .collect();
        Self { by_len }
    }

    fn bucket(&self, slot: &[usize]) -> &Bucket {
        &self.by_len[&slot.len()]
    }

    fn candidates(&self, slot: &[usize], fill: &Fill) -> Vec<&Word> {
        let b = self.bucket(slot);
        b.candidates(slot, fill)
            .into_iter()
            .map(|i| &b.words[i])
            .collect()
    }
}

fn place(word: &[char], slot: &[usize], fill: &mut Fill) {
    for (&sq, &l) in slot.iter().zip(word) {
        fill[sq] = Some(l);
    }
}

/// The open slot with the fewest candidates (most-constrained first).
fn tightest(dict: &Dictionary, slots: &[Vec<usize>], done: &[bool], fill: &Fill) -> Option<usize> {
    (0..slots.len())
        .filter(|&s| !done[s])
        .min_by_key(|&s| dict.bucket(&slots[s]).candidates(&slots[s], fill).len())
}

/// Complete fills consistent with `fill`, counted up to `cap`.
fn count_fills(
    dict: &Dictionary,
    slots: &[Vec<usize>],
    done: &mut [bool],
    fill: &mut Fill,
    cap: usize,
) -> usize {
    let Some(s) = tightest(dict, slots, done, fill) else {
        return 1;
    };
    let mut n = 0;
    done[s] = true;
    for w in dict.candidates(&slots[s], fill) {
        let saved = fill.clone();
        place(w, &slots[s], fill);
        n += count_fills(dict, slots, done, fill, cap - n);
        *fill = saved;
        if n >= cap {
            break;
        }
    }
    done[s] = false;
    n
}

/// One random complete fill, or `None` when the node budget runs out.
fn random_fill(
    dict: &Dictionary,
    slots: &[Vec<usize>],
    cells: usize,
    rng: &mut Rng,
) -> Option<Vec<Word>> {
    fn go(
        dict: &Dictionary,
        slots: &[Vec<usize>],
        done: &mut [bool],
        fill: &mut Fill,
        rng: &mut Rng,
        budget: &mut usize,
    ) -> bool {
        let Some(s) = tightest(dict, slots, done, fill) else {
            return true;
        };
        let mut cs = dict.candidates(&slots[s], fill);
        rng.shuffle(&mut cs);
        done[s] = true;
        for w in cs {
            if *budget == 0 {
                break;
            }
            *budget -= 1;
            let saved = fill.clone();
            place(w, &slots[s], fill);
            if go(dict, slots, done, fill, rng, budget) {
                return true;
            }
            *fill = saved;
        }
        done[s] = false;
        false
    }
    let mut done = vec![false; slots.len()];
    let mut fill: Fill = vec![None; cells];
    let mut budget = FILL_BUDGET;
    go(dict, slots, &mut done, &mut fill, rng, &mut budget).then(|| {
        slots
            .iter()
            .map(|s| s.iter().map(|&sq| fill[sq].unwrap()).collect())
            .collect()
    })
}

struct Instance {
    grid: Grid,
    slots: Vec<Vec<usize>>,
    given: Vec<bool>,
    solution: Vec<Word>,
}

impl Instance {
    /// The fill holding every given slot's true word.
    fn given_fill(&self) -> Fill {
        let mut fill: Fill = vec![None; self.grid.cells()];
        for (s, slot) in self.slots.iter().enumerate() {
            if self.given[s] {
                place(&self.solution[s], slot, &mut fill);
            }
        }
        fill
    }
}

/// Number of fills consistent with the instance's givens, up to `cap`.
fn count_solutions(dict: &Dictionary, inst: &Instance, cap: usize) -> usize {
    let mut fill = inst.given_fill();
    let mut done = inst.given.clone();
    count_fills(dict, &inst.slots, &mut done, &mut fill, cap)
}

/// A random fill, then given slots added in random order until it is the only
/// fill the dictionary admits. Every placed word must be a dictionary word, so
/// the solution is checked against the dictionary too.
/// A grid comes fresh from `grids` on every attempt, so a grid the
/// dictionary cannot fill within the budget is simply replaced.
fn draw_instance(dict: &Dictionary, grids: Grids, rng: &mut Rng) -> Instance {
    loop {
        let grid = grids.draw(rng);
        let slots = grid.slots();
        let Some(solution) = random_fill(dict, &slots, grid.cells(), rng) else {
            continue;
        };
        let mut order: Vec<usize> = (0..slots.len()).collect();
        rng.shuffle(&mut order);
        let mut inst = Instance {
            given: vec![false; slots.len()],
            grid,
            slots,
            solution,
        };
        for &s in &order {
            inst.given[s] = true;
            if count_solutions(dict, &inst, 2) == 1 {
                return inst;
            }
        }
        unreachable!("with every slot given the fill is unique");
    }
}

fn propagate(dict: &Dictionary, inst: &Instance, depth: usize) -> (Vec<Option<bool>>, Fill) {
    let slots = &inst.slots;
    let mut fill = inst.given_fill();
    let mut placed: Vec<Option<bool>> = inst.given.iter().map(|&g| g.then_some(true)).collect();
    let mut forced = 0;
    while forced < depth {
        let next = (0..slots.len()).find_map(|s| {
            if placed[s].is_some() {
                return None;
            }
            let c = dict.candidates(&slots[s], &fill);
            (c.len() == 1).then(|| (s, c[0].clone()))
        });
        let Some((s, w)) = next else { break };
        place(&w, &slots[s], &mut fill);
        placed[s] = Some(false);
        forced += 1;
    }
    (placed, fill)
}

/// Per-lane bookkeeping the shared `Lane` does not keep.
#[derive(Default)]
struct Stats {
    givens: Vec<usize>,
    slots: Vec<usize>,
    blacks: Vec<usize>,
    forced: usize,
}

fn build_lane(dict: &Dictionary, grids: Grids, min_edges: usize, seed: u64) -> (Lane, Stats) {
    let mut rng = Rng(seed);
    let mut lane = Lane::default();
    let mut stats = Stats::default();
    while lane.edges.len() < min_edges {
        let inst = draw_instance(dict, grids, &mut rng);
        let slots = &inst.slots;
        let g = inst.given.iter().filter(|&&b| b).count();
        stats.givens.push(g);
        stats.slots.push(slots.len());
        stats.blacks.push(inst.grid.blacks());
        let depth = rng.below((slots.len() - g) as u64 + 1) as usize;
        let (placed, fill) = propagate(dict, &inst, depth);
        for (s, slot) in slots.iter().enumerate() {
            match placed[s] {
                Some(true) => {
                    lane.push(GIVEN);
                    lane.expected[0] += 1;
                }
                Some(false) => {
                    assert_eq!(
                        slot.iter().map(|&sq| fill[sq].unwrap()).collect::<Word>(),
                        inst.solution[s]
                    );
                    lane.push(FORCED);
                    lane.expected[1] += 1;
                    stats.forced += 1;
                }
                None => {
                    let c = dict.candidates(slot, &fill);
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
    (lane, stats)
}

fn run(name: &str, freq: &HashMap<String, f64>, grids: Grids) {
    let decl = declarations(CROSSWORD_CLASS);
    admit(&decl, CROSSWORD_CLASS).expect("the crossword class declares the canonical reading");
    let dict = &Dictionary::new(freq, &grids.lengths());
    let t = Instant::now();
    let (lane, stats) = build_lane(dict, grids, 1_000_000, 0xC0_CA_u64);
    let mut hist = vec![0usize; stats.slots.iter().max().unwrap() + 1];
    for &g in &stats.givens {
        hist[g] += 1;
    }
    println!("D-PUZZLE-0 / real-word crosswords: {name}, grids {grids:?}");
    for (len, b) in {
        let mut v: Vec<_> = dict.by_len.iter().collect();
        v.sort_by_key(|(l, _)| **l);
        v
    } {
        println!("  dictionary: {} words of length {len}", b.words.len());
    }
    println!(
        "  lane: {} edges from {} unique-solution instances, built in {:.2?}",
        lane.edges.len(),
        lane.groups,
        t.elapsed()
    );
    let spread: Vec<String> = hist
        .iter()
        .enumerate()
        .filter(|(_, &n)| n > 0)
        .map(|(g, n)| format!("{g}:{n}"))
        .collect();
    println!(
        "  givens needed for uniqueness (givens:instances): {}",
        spread.join(" ")
    );
    let mean = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len() as f64;
    println!(
        "  per grid: {:.1} slots, {:.1} black squares on average",
        mean(&stats.slots),
        mean(&stats.blacks)
    );
    let counts = check_three_ways(&lane, &decl, CROSSWORD_CLASS);
    let asserted: Vec<usize> = per_group(&lane, GIVEN.bit() | FORCED.bit() | ENTAILED.bit())
        .into_iter()
        .map(|n| n as usize)
        .collect();
    assert_eq!(asserted, stats.slots);
    println!("  every instance: exactly one asserted claim per slot (group fold)");
    report(&lane, &decl, CROSSWORD_CLASS, &counts);
}

fn main() {
    let english = english_vocabulary();
    let name = "English, DeepNSM v1 4096 + academic 20k (committed)";
    run(name, &english, Grids::Template);
    run(name, &english, Grids::Nyt(NYT_SIDE));
    match std::env::var("XWORD_ANSWERS_DIR") {
        Ok(dir) => {
            let targets: Vec<String> = ["train", "val", "test"]
                .iter()
                .map(|s| {
                    std::fs::read_to_string(format!("{dir}/{s}.target"))
                        .expect("XWORD_ANSWERS_DIR holds train/val/test.target")
                })
                .collect();
            run(
                "crossword answers, Down-and-Across clue set (read at run time)",
                &xword_answers(&targets),
                Grids::Nyt(NYT_SIDE),
            );
        }
        Err(_) => println!(
            "Crossword-answer run skipped: set XWORD_ANSWERS_DIR to the folder of *.target files"
        ),
    }
    match std::env::var("DEREKO_PATH") {
        Ok(path) => {
            let text = std::fs::read_to_string(&path).expect("DEREKO_PATH is readable");
            run(
                "German, DeReKo-2014 top 20k (CC BY-NC 3.0, read at run time)",
                &german_vocabulary(&text),
                Grids::Template,
            );
        }
        Err(_) => println!("German run skipped: set DEREKO_PATH to the DeReKo-2014 .freq file"),
    }
}

#[cfg(test)]
mod tests {
    use super::population_fold::{count_in, queries, UNKNOWN_CAUSES};
    use super::*;
    use lance_graph_contract::epistemic_state5::fact::{CAUSES, DIRECT, IND_KNOWN};
    use lance_graph_contract::epistemic_state5::facts_population;

    fn coca() -> Dictionary {
        Dictionary::new(&english_vocabulary(), &Grids::Nyt(NYT_SIDE).lengths())
    }

    /// The English dictionary is exactly DeepNSM's two vocabularies. Each
    /// source boundary has a witness: v1 core words, academic-only words,
    /// words ranked past 4096 in v1's table and absent from the academic list
    /// (the rank cut), and an inflected form only `word_forms.csv` carries.
    #[test]
    fn the_english_dictionary_is_deepnsm_v1_plus_academic() {
        let d = coca();
        for (len, n) in [(4, 1250), (5, 1697)] {
            let b = &d.by_len[&len];
            assert_eq!(b.words.len(), n);
            assert!(b.words.iter().all(|w| w.len() == len));
        }
        let has = |w: &str| {
            let w: Word = w.chars().collect();
            d.by_len[&w.len()].words.contains(&w)
        };
        assert_eq!(d.by_len[&4].words[0], "have".chars().collect::<Word>());
        assert!(has("that") && has("with"), "v1 core words");
        assert!(has("abyss") && has("acorn"), "academic-only words");
        assert!(
            !has("gosh") && !has("kinda"),
            "v1 rank cut at 4096 not applied"
        );
        assert!(
            !has("says"),
            "an inflected form from word_forms.csv leaked in"
        );
    }

    /// The bitset index returns exactly the words a direct scan returns.
    #[test]
    fn the_index_agrees_with_a_scan() {
        let d = coca();
        let mut rng = Rng(3);
        for _ in 0..200 {
            let grid = Grid::random_nyt(NYT_SIDE, &mut rng);
            let slots = grid.slots();
            let mut fill: Fill = vec![None; grid.cells()];
            for sq in fill.iter_mut() {
                if rng.below(4) == 0 {
                    *sq = Some((b'a' + rng.below(26) as u8) as char);
                }
            }
            for slot in &slots {
                let b = d.bucket(slot);
                let scan: Vec<usize> = (0..b.words.len())
                    .filter(|&i| {
                        slot.iter()
                            .zip(&b.words[i])
                            .all(|(&sq, &l)| fill[sq].is_none_or(|f| f == l))
                    })
                    .collect();
                assert_eq!(b.candidates(slot, &fill), scan);
            }
        }
    }

    fn small_lane() -> (Lane, Stats) {
        build_lane(&coca(), Grids::Template, 30_000, 0x7E57)
    }

    fn nyt_lane() -> (Lane, Stats) {
        build_lane(&coca(), Grids::Nyt(NYT_SIDE), 30_000, 0x7E58)
    }

    #[test]
    fn every_question_counts_the_same_three_ways() {
        for (lane, _) in [small_lane(), nyt_lane()] {
            let counts = check_three_ways(&lane, &declarations(CROSSWORD_CLASS), CROSSWORD_CLASS);
            assert!(counts.iter().all(|&n| n > 0), "a question went unexercised");
        }
    }

    #[test]
    fn the_four_states_partition_the_lane() {
        let (lane, _) = small_lane();
        let union = queries()[1..].iter().fold(0, |u, q| u | q.population);
        assert_eq!(count_in(&lane.edges, union), lane.edges.len());
        assert_eq!(count_in(&lane.edges, UNKNOWN_CAUSES.bit()), 0);
    }

    /// One asserted claim per slot; the given count per instance is what the
    /// generator recorded; forcing happens on real words.
    #[test]
    fn the_group_fold_sees_every_instance_whole() {
        let (lane, stats) = small_lane();
        assert!(per_group(&lane, facts_population(CAUSES))
            .iter()
            .all(|&n| n == 10));
        assert!(stats.slots.iter().all(|&n| n == 10));
        let given: Vec<usize> = per_group(&lane, facts_population(DIRECT | CAUSES))
            .into_iter()
            .map(|n| n as usize)
            .collect();
        assert_eq!(given, stats.givens);
        let forced = per_group(&lane, facts_population(IND_KNOWN | CAUSES));
        assert!(
            forced.iter().sum::<u32>() > 0,
            "no real-word slot was ever forced"
        );
    }

    /// Kept instances are unique, every solution word is a dictionary word,
    /// and the givens are doing work: with all of them removed, the
    /// dictionary admits more than one fill of the same grid.
    #[test]
    fn instances_are_unique_and_givens_are_needed() {
        let d = coca();
        let mut rng = Rng(11);
        let mut ambiguous_without = 0;
        for grids in [Grids::Template, Grids::Nyt(NYT_SIDE)] {
            for _ in 0..10 {
                let inst = draw_instance(&d, grids, &mut rng);
                assert_eq!(count_solutions(&d, &inst, 3), 1);
                for (s, w) in inst.solution.iter().enumerate() {
                    assert!(d.bucket(&inst.slots[s]).words.contains(w));
                }
                let open = Instance {
                    grid: inst.grid.clone(),
                    slots: inst.slots.clone(),
                    given: vec![false; inst.slots.len()],
                    solution: inst.solution.clone(),
                };
                if count_solutions(&d, &open, 2) > 1 {
                    ambiguous_without += 1;
                }
            }
        }
        assert!(ambiguous_without > 0, "uniqueness never needed a given");
    }

    /// German parsing: umlauts and ß kept, proper nouns dropped, rows summed.
    #[test]
    fn dereko_rows_are_parsed_by_tag_and_summed() {
        let text = "Haus\tHaus\tNN\t10\nhaus\thausen\tVVIMP\t2\nBerlin\tBerlin\tNE\t99\n\
                    Füße\tFuß\tNN\t5\n,\t,\t$,\t1000\n";
        let f = dereko_frequencies(text);
        assert_eq!(f.get("haus"), Some(&12.0));
        assert_eq!(f.get("füße"), Some(&5.0));
        assert!(!f.contains_key("berlin"));
        assert!(!f.contains_key(","));
        assert_eq!(
            top_words(&f, 4, 10),
            vec!["haus".chars().collect::<Word>(), "füße".chars().collect()]
        );
    }

    /// German keeps the 20,000 most frequent forms and drops the rest.
    #[test]
    fn german_keeps_the_top_20k_forms() {
        let mut text = String::new();
        for i in 0..GERMAN_VOCAB + 5 {
            // Distinct alphabetic forms, frequency falling with i.
            let w: String = format!("{i:05}")
                .bytes()
                .map(|b| (b'a' + (b - b'0')) as char)
                .collect();
            text.push_str(&format!("{w}\t{w}\tNN\t{}\n", 1_000_000 - i));
        }
        let v = german_vocabulary(&text);
        assert_eq!(v.len(), GERMAN_VOCAB);
        assert!(v.contains_key("aaaaa"), "most frequent form dropped");
        assert!(!v.contains_key("caaae"), "form ranked past 20k kept"); // i = 20004
    }

    fn grid(rows: &[&str]) -> Grid {
        Grid {
            side: rows.len(),
            white: rows
                .iter()
                .flat_map(|r| r.bytes().map(|b| b == b'.'))
                .collect(),
        }
    }

    /// Each NYT rule refuses on its own: the step-2 template passes, and one
    /// grid per rule breaks exactly that rule.
    #[test]
    fn each_nyt_rule_refuses_its_own_violation() {
        assert!(Grid::template().is_nyt_valid());
        let ok = grid(&[
            "...#...", ".......", ".......", "#.....#", ".......", ".......", "...#...",
        ]);
        assert!(ok.is_nyt_valid());
        // asymmetric: one black square without its mirror
        let asym = grid(&[
            "...#...", ".......", ".......", ".......", ".......", ".......", ".......",
        ]);
        assert!(!asym.symmetric() && !asym.is_nyt_valid());
        // a two-letter word
        let short = grid(&[
            "..#....", ".......", ".......", ".......", ".......", ".......", "....#..",
        ]);
        assert!(short.symmetric() && !short.is_nyt_valid());
        // two white regions: a full black column cuts the grid
        let cut = grid(&[
            "...#...", "...#...", "...#...", "...#...", "...#...", "...#...", "...#...",
        ]);
        assert!(cut.symmetric() && !cut.connected() && !cut.is_nyt_valid());
        // too many black squares (12 of 49 > a sixth), every other rule kept
        let heavy = grid(&[
            "...#...", "...#...", "###.###", ".......", "###.###", "...#...", "...#...",
        ]);
        assert!(heavy.symmetric() && heavy.blacks() * MAX_BLACK_DIVISOR > heavy.cells());
        assert!(!heavy.is_nyt_valid());
    }

    /// Random NYT grids are valid, carry black squares, and actually vary.
    #[test]
    fn random_nyt_grids_are_valid_and_vary() {
        let mut rng = Rng(5);
        let grids: Vec<Grid> = (0..50)
            .map(|_| Grid::random_nyt(NYT_SIDE, &mut rng))
            .collect();
        assert!(grids.iter().all(|g| g.is_nyt_valid() && g.blacks() > 0));
        let unique: std::collections::HashSet<Vec<bool>> =
            grids.iter().map(|g| g.white.clone()).collect();
        assert!(unique.len() > 10, "only {} distinct grids", unique.len());
    }

    /// On NYT grids every instance still holds one asserted claim per slot,
    /// and the slot count follows the grid.
    #[test]
    fn nyt_lane_holds_one_claim_per_slot_of_its_own_grid() {
        let (lane, stats) = nyt_lane();
        let asserted: Vec<usize> = per_group(&lane, facts_population(CAUSES))
            .into_iter()
            .map(|n| n as usize)
            .collect();
        assert_eq!(asserted, stats.slots);
        let lo = stats.slots.iter().min().unwrap();
        let hi = stats.slots.iter().max().unwrap();
        assert!(lo < hi, "every grid had the same number of slots");
    }

    /// Crossword answers: the dataset's word splits are rejoined, punctuation
    /// is refused, repeats are counted.
    #[test]
    fn crossword_answers_are_rejoined_and_counted() {
        let targets = vec![
            "egg white\nlimos\nen nea\n".to_string(),
            "limos\nst. louis\n".to_string(),
        ];
        let a = xword_answers(&targets);
        assert_eq!(a.get("eggwhite"), Some(&1.0));
        assert_eq!(a.get("ennea"), Some(&1.0));
        assert_eq!(a.get("limos"), Some(&2.0));
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn an_undeclared_lane_is_refused_before_any_edge() {
        let decl = declarations(CROSSWORD_CLASS);
        assert!(admit(&decl, CROSSWORD_CLASS).is_some());
        assert!(admit(&decl, 0x0906).is_none());
    }
}
