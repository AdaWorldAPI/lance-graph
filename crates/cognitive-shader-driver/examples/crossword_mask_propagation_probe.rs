//! D-PUZZLE-0, step 3: is crossword propagation the same substrate operation
//! as Sudoku's?
//!
//! The question is not whether a crossword can be solved. It is whether, once a
//! puzzle is compiled, propagation is `mask → intersect → popcount → promote →
//! expose → adjacent masks → repeat`, with nothing crossword-shaped left on the
//! hot path. Steps 2 and 2b (`crossword_population_fold_probe`,
//! `crossword_real_words_probe`) re-derive candidates from a cell array on every
//! step; they stay as the oracle here.
//!
//! # Three coordinate systems, kept apart
//!
//! | what | encoding | where it comes from |
//! |---|---|---|
//! | WHERE a letter sits | cell = `Morton8x8::from_xy(col, row)` (u16) | `lance_graph_contract::morton8x8` |
//! | WHICH slot position | `(slot:offset)` = `FacetTier { hi: slot, lo: offset }.as_u16()` | `facet::FacetTier`, the `(group:member)` reading |
//! | WHAT word | `WordId` (u16) from DeepNSM-v2 `PaletteVocab::from_frequency_ranked` | `deepnsm_v2::vocab` |
//! | what is BELIEVED | CE64 bits 59..63, the shared GIVEN 20 / FORCED 21 / ENTAILED 22 / CANDIDATE 4 | `shared/population_fold.rs` |
//!
//! A crossing is equality of two cell codes. The contract stores no topology
//! (`morton8x8.rs`: "no neighbour list and no stored edge"), so the equality is
//! evaluated once, at compile time: positions are sorted by cell code and every
//! equal pair becomes `cross[slot:offset] = (other slot:other offset)`, one u16
//! per position. The runtime reads that lane; it never sees the grid.
//!
//! # Letters
//!
//! No canonical Moore symbol space exists to borrow: every Moore in the repo is
//! a direction table, a grid-edge validity byte or a palette-law operand, and
//! none of them may be read as a letter. So a letter is a probe-local code: the
//! sorted alphabet of the admitted words, code `k + 1` for the `k`-th letter,
//! `0` = no letter. The alphabet is read from the vocabulary, never assumed:
//! English gives 27 (a–z and the é of `sauté`, `cliché`), German its own.
//!
//! # Vocabularies (DeepNSM-v2)
//!
//! - **English** (committed): `crates/deepnsm/word_frequency/academic_20k.csv`,
//!   column `word`, lowercased, into `PaletteVocab::from_frequency_ranked`:
//!   exactly the vocabulary DeepNSM-v2 builds (`examples/genre_shapes.rs`),
//!   18,555 distinct ids (`vocab.rs` quotes 18,559 distinct surfaces; four
//!   collapse when lowercased). Words with a hyphen or apostrophe keep their id but
//!   are not crossword words.
//! - **German** (runtime only): DeepNSM-v2 builds no German vocabulary and the
//!   repository holds none. With `DEREKO_PATH` set, the 20,000 most frequent
//!   DeReKo-2014 forms (lowercased, proper nouns and non-alphabetic forms
//!   dropped, frequencies summed) go through the same `from_frequency_ranked`.
//!   CC BY-NC 3.0; nothing derived is committed. ä ö ü ß stay letters of their
//!   own (no `ae`/`ss` folding: the vocabulary has none); text is assumed NFC,
//!   and a form with a combining mark is refused because it is not alphabetic.
//!
//! # The candidate population
//!
//! For a language, a word length `L`, an offset `i` and a letter code `x`,
//! `P(L, i, x)` is a bitset over that language's `WordId`s. A slot's
//! candidates start as `P_all(L)` and every exposed crossing letter ANDs one
//! `P` into them (`ndarray::simd::mask_and_assign`); `popcount_batch_u64`
//! decides: 0 = contradiction, 1 = forced, more = still candidates. A forced
//! word exposes its letters to its crossings through the `cross` lane, and the
//! queue runs to a fixed point. Backtracking is used only to CREATE puzzles and
//! to check uniqueness; solving is the fixed point alone.
//!
//! Run: `cargo run --release -p cognitive-shader-driver --example crossword_mask_propagation_probe`
//! German too: `DEREKO_PATH=/path/DeReKo-2014-II-MainArchive-STT.100000.freq cargo run ...`
//! Tests: `cargo test -p cognitive-shader-driver --example crossword_mask_propagation_probe`

use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

use deepnsm_v2::vocab::{PaletteVocab, WordId};
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::facet::FacetTier;
use lance_graph_contract::morton8x8::Morton8x8;
use ndarray::simd::{mask_and_assign, popcount_batch_u64};

#[path = "shared/population_fold.rs"]
mod population_fold;
use population_fold::{
    admit, check_three_ways, declarations, per_group, report, Lane, Rng, CANDIDATE, ENTAILED,
    FORCED, GIVEN,
};

/// Same class as steps 2 and 2b: the same domain.
const CROSSWORD_CLASS: ClassId = 0x0907;

/// NYT rules: shortest word; most black squares (a sixth).
const MIN_WORD: usize = 3;
const MAX_BLACK_DIVISOR: usize = 6;
/// Longest slot (NYT 15×15) and the per-slot stride of every position lane.
const MAX_LEN: usize = 15;
const STRIDE: usize = 16;
/// No crossing at this position.
const NONE: u16 = u16::MAX;
/// No word placed in this slot.
const UNSET: WordId = WordId::MAX;
/// German forms kept.
const GERMAN_VOCAB: usize = 20_000;

const ACADEMIC: &str = include_str!("../../deepnsm/word_frequency/academic_20k.csv");

// ─────────────────────────────── vocabulary ───────────────────────────────

/// The language a lexicon and a puzzle belong to. A puzzle is only ever
/// solved against its own language's populations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lang {
    En,
    De,
}

/// Cold side: strings and the letter codebook. Construction and checks only.
struct Cold {
    vocab: PaletteVocab,
    letters: Vec<char>,
}

impl Cold {
    fn code(&self, c: char) -> Option<u8> {
        self.letters.binary_search(&c).ok().map(|i| i as u8 + 1)
    }

    fn word(&self, w: WordId) -> &str {
        self.vocab.word(w).expect("word id in range")
    }
}

/// Hot side: integers and masks only. No `String`, no `char`.
struct Hot {
    lang: Lang,
    blocks: usize,
    /// Alphabet size + 1 (code 0 = no letter).
    codes: usize,
    /// Per `WordId`: crossword length, 0 = not a crossword word.
    len: Vec<u8>,
    /// `[id * STRIDE + offset]` = letter code.
    spell: Vec<u8>,
    /// By length: every crossword word of that length.
    all: Vec<Vec<u64>>,
    /// By length: `[(offset * codes + code) * blocks ..]` = `P(L, offset, code)`.
    at: Vec<Vec<u64>>,
}

impl Hot {
    fn pop(&self, len: usize, offset: usize, code: u8) -> &[u64] {
        let i = (offset * self.codes + code as usize) * self.blocks;
        &self.at[len][i..i + self.blocks]
    }

    fn letter(&self, w: WordId, offset: usize) -> u8 {
        self.spell[w as usize * STRIDE + offset]
    }
}

/// A crossword word: `MIN_WORD..=MAX_LEN` letters, every one alphabetic and
/// lowercase.
fn admissible(w: &str) -> bool {
    let n = w.chars().count();
    (MIN_WORD..=MAX_LEN).contains(&n) && w.chars().all(|c| c.is_alphabetic() && !c.is_uppercase())
}

/// Build both sides from a frequency-ranked word list (most frequent first).
fn build_lexicon(lang: Lang, ranked: &[String]) -> (Hot, Cold) {
    let mut vocab = PaletteVocab::new();
    vocab.from_frequency_ranked(ranked.iter().map(String::as_str));
    let n = vocab.len();
    assert!(n < UNSET as usize, "WordId::MAX is the empty marker");
    let mut letters: Vec<char> = (0..n)
        .filter_map(|id| vocab.word(id as WordId))
        .filter(|w| admissible(w))
        .flat_map(str::chars)
        .collect();
    letters.sort_unstable();
    letters.dedup();
    assert!(letters.len() < 255, "letter codes are one byte");
    let cold = Cold { vocab, letters };
    let codes = cold.letters.len() + 1;
    let blocks = n.div_ceil(64);
    let mut hot = Hot {
        lang,
        blocks,
        codes,
        len: vec![0; n],
        spell: vec![0; n * STRIDE],
        all: vec![Vec::new(); MAX_LEN + 1],
        at: vec![Vec::new(); MAX_LEN + 1],
    };
    for l in MIN_WORD..=MAX_LEN {
        hot.all[l] = vec![0; blocks];
        hot.at[l] = vec![0; l * codes * blocks];
    }
    for id in 0..n {
        let w = cold.word(id as WordId);
        if !admissible(w) {
            continue;
        }
        let l = w.chars().count();
        hot.len[id] = l as u8;
        let bit = 1u64 << (id % 64);
        hot.all[l][id / 64] |= bit;
        for (i, c) in w.chars().enumerate() {
            let code = cold.code(c).expect("letter in codebook");
            hot.spell[id * STRIDE + i] = code;
            hot.at[l][(i * codes + code as usize) * blocks + id / 64] |= bit;
        }
    }
    (hot, cold)
}

/// English: DeepNSM-v2's academic vocabulary, built as `genre_shapes` builds it.
fn english_ranked() -> Vec<String> {
    ACADEMIC
        .lines()
        .skip(1)
        .filter_map(|l| l.split(',').nth(3))
        .map(str::to_lowercase)
        .filter(|w| !w.is_empty())
        .collect()
}

/// German: the `GERMAN_VOCAB` most frequent DeReKo forms (`form, lemma, tag,
/// freq`), lowercased, `NE` and non-alphabetic forms dropped, summed.
fn german_ranked(text: &str) -> Vec<String> {
    let mut f: HashMap<String, f64> = HashMap::new();
    for line in text.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let [form, _, tag, freq] = c.as_slice() else {
            continue;
        };
        if *tag == "NE" {
            continue;
        }
        let w = form.to_lowercase();
        if w.is_empty() || !w.chars().all(char::is_alphabetic) {
            continue;
        }
        *f.entry(w).or_insert(0.0) += freq.trim().parse::<f64>().unwrap_or(0.0);
    }
    let mut r: Vec<(String, f64)> = f.into_iter().collect();
    r.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    r.truncate(GERMAN_VOCAB);
    r.into_iter().map(|(w, _)| w).collect()
}

// ─────────────────────────────── layout ───────────────────────────────

/// A board as row masks: bit `c` of `rows[r]` = cell `(r, c)` is white.
#[derive(Clone, Debug, PartialEq)]
struct Grid {
    side: usize,
    rows: Vec<u16>,
}

/// Every white bit of a line lies in a run of at least three: OR of the
/// three-wide windows covers the line.
fn line_runs_ok(w: u16) -> bool {
    let t = w & (w >> 1) & (w >> 2);
    (t | (t << 1) | (t << 2)) == w
}

impl Grid {
    #[cfg(test)]
    fn from_rows(rows: &[&str]) -> Self {
        let rows: Vec<u16> = rows
            .iter()
            .map(|r| {
                r.bytes()
                    .enumerate()
                    .fold(0u16, |m, (c, b)| m | (u16::from(b == b'.') << c))
            })
            .collect();
        Self {
            side: rows.len(),
            rows,
        }
    }

    fn full_line(&self) -> u16 {
        ((1u32 << self.side) - 1) as u16
    }

    fn cols(&self) -> Vec<u16> {
        (0..self.side)
            .map(|c| (0..self.side).fold(0u16, |m, r| m | (((self.rows[r] >> c) & 1) << r)))
            .collect()
    }

    fn reverse(&self, w: u16) -> u16 {
        w.reverse_bits() >> (16 - self.side)
    }

    fn blacks(&self) -> usize {
        self.side * self.side
            - self
                .rows
                .iter()
                .map(|r| r.count_ones() as usize)
                .sum::<usize>()
    }

    /// 180° symmetry: row `r` reversed is row `side - 1 - r`.
    fn symmetric(&self) -> bool {
        (0..self.side).all(|r| self.reverse(self.rows[r]) == self.rows[self.side - 1 - r])
    }

    /// Every across and down run is at least `MIN_WORD` long.
    fn runs_ok(&self) -> bool {
        self.rows
            .iter()
            .chain(self.cols().iter())
            .all(|&w| line_runs_ok(w))
    }

    /// One white region: grow from the first white cell by row-mask dilation
    /// until nothing changes.
    fn connected(&self) -> bool {
        let Some(r0) = self.rows.iter().position(|&w| w != 0) else {
            return false;
        };
        let full = self.full_line();
        let mut reach = vec![0u16; self.side];
        reach[r0] = self.rows[r0] & self.rows[r0].wrapping_neg();
        loop {
            let mut changed = false;
            for r in 0..self.side {
                let up = if r > 0 { reach[r - 1] } else { 0 };
                let down = if r + 1 < self.side { reach[r + 1] } else { 0 };
                let grown = (reach[r] | (reach[r] << 1) | (reach[r] >> 1) | up | down)
                    & full
                    & self.rows[r];
                if grown != reach[r] {
                    reach[r] = grown;
                    changed = true;
                }
            }
            if !changed {
                return reach == self.rows;
            }
        }
    }

    fn is_nyt_valid(&self) -> bool {
        self.symmetric()
            && self.runs_ok()
            && self.connected()
            && self.blacks() * MAX_BLACK_DIVISOR <= self.side * self.side
    }

    /// Symmetric black pairs drawn until the grid is NYT-valid with at least
    /// one black square.
    fn random_nyt(side: usize, rng: &mut Rng) -> Self {
        let cells = side * side;
        loop {
            let mut g = Self {
                side,
                rows: vec![((1u32 << side) - 1) as u16; side],
            };
            for i in 0..cells.div_ceil(2) {
                if rng.below(8) == 0 {
                    for j in [i, cells - 1 - i] {
                        g.rows[j / side] &= !(1 << (j % side));
                    }
                }
            }
            if g.blacks() > 0 && g.is_nyt_valid() {
                return g;
            }
        }
    }
}

/// A compiled puzzle: everything the runtime may read.
#[derive(Clone)]
struct Puzzle {
    lang: Lang,
    /// Per slot: its length.
    len: Vec<u8>,
    /// `[slot * STRIDE + offset]`: the crossing `(slot:offset)` tile, or `NONE`.
    cross: Vec<u16>,
    /// `[slot * STRIDE + offset]`: the Morton cell code. Construction and the
    /// oracle only; propagation never reads it.
    cell: Vec<u16>,
}

impl Puzzle {
    fn slots(&self) -> usize {
        self.len.len()
    }
}

/// Compile a grid: slots from the row and column masks, each position's cell
/// from `Morton8x8::checked_offset`, crossings from equal cell codes.
fn compile(grid: &Grid, lang: Lang) -> Puzzle {
    let mut len = Vec::new();
    let mut cell = Vec::new();
    for (across, lines) in [(true, grid.rows.clone()), (false, grid.cols())] {
        for (line, &w) in lines.iter().enumerate() {
            let mut c = 0;
            while c < grid.side {
                if w >> c & 1 == 0 {
                    c += 1;
                    continue;
                }
                let start = c;
                while c < grid.side && w >> c & 1 == 1 {
                    c += 1;
                }
                let l = c - start;
                if l < 2 {
                    continue;
                }
                let (x, y) = if across { (start, line) } else { (line, start) };
                let (dx, dy) = if across { (1i8, 0i8) } else { (0, 1) };
                let origin = Morton8x8::from_xy(x as u8, y as u8);
                let mut codes = [NONE; STRIDE];
                for (off, code) in codes.iter_mut().enumerate().take(l) {
                    *code = origin
                        .checked_offset(dx * off as i8, dy * off as i8)
                        .expect("on the board")
                        .code();
                }
                len.push(l as u8);
                cell.extend_from_slice(&codes);
            }
        }
    }
    assert!(len.len() < 255, "slot ids are one byte");
    let mut pos: Vec<(u16, u8, u8)> = len
        .iter()
        .enumerate()
        .flat_map(|(s, &l)| (0..l).map(move |o| (s, o)))
        .map(|(s, o)| (cell[s * STRIDE + o as usize], s as u8, o))
        .collect();
    pos.sort_unstable();
    let mut cross = vec![NONE; len.len() * STRIDE];
    for w in pos.windows(2) {
        if w[0].0 == w[1].0 {
            let tile = |(_, s, o): (u16, u8, u8)| FacetTier { hi: s, lo: o }.as_u16();
            cross[w[0].1 as usize * STRIDE + w[0].2 as usize] = tile(w[1]);
            cross[w[1].1 as usize * STRIDE + w[1].2 as usize] = tile(w[0]);
        }
    }
    Puzzle {
        lang,
        len,
        cross,
        cell,
    }
}

// ─────────────────────────────── runtime ───────────────────────────────

/// Why a solve did not start or did not finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stop {
    /// The puzzle and the populations are different languages.
    Language,
    /// A slot was left with no candidate, or two placed words disagree.
    Contradiction(u8),
}

/// Per-slot candidate masks over `WordId` and the placed word, if any.
#[derive(Clone)]
struct State {
    blocks: usize,
    cand: Vec<u64>,
    placed: Vec<WordId>,
    events: usize,
}

impl State {
    fn new(hot: &Hot, puz: &Puzzle) -> Result<Self, Stop> {
        if hot.lang != puz.lang {
            return Err(Stop::Language);
        }
        let b = hot.blocks;
        let mut cand = vec![0u64; puz.slots() * b];
        for (s, &l) in puz.len.iter().enumerate() {
            cand[s * b..(s + 1) * b].copy_from_slice(&hot.all[l as usize]);
        }
        Ok(Self {
            blocks: b,
            cand,
            placed: vec![UNSET; puz.slots()],
            events: 0,
        })
    }

    fn mask(&self, s: usize) -> &[u64] {
        &self.cand[s * self.blocks..(s + 1) * self.blocks]
    }

    fn count(&self, s: usize) -> u64 {
        popcount_batch_u64(self.mask(s))
    }

    fn place(&mut self, s: usize, w: WordId) {
        let b = self.blocks;
        self.cand[s * b..(s + 1) * b].fill(0);
        self.cand[s * b + w as usize / 64] = 1 << (w % 64);
        self.placed[s] = w;
    }

    /// Promote every unplaced slot already down to one candidate; refuse any
    /// with none.
    fn seed(&mut self, queue: &mut Vec<u8>) -> Result<(), Stop> {
        for s in 0..self.placed.len() {
            if self.placed[s] == UNSET {
                match self.count(s) {
                    0 => return Err(Stop::Contradiction(s as u8)),
                    1 => {
                        let w = first_bit(self.mask(s));
                        self.place(s, w);
                        queue.push(s as u8);
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

fn first_bit(m: &[u64]) -> WordId {
    let (b, w) = m
        .iter()
        .enumerate()
        .find(|(_, &w)| w != 0)
        .expect("non-empty mask");
    (b * 64 + w.trailing_zeros() as usize) as WordId
}

fn bits(m: &[u64]) -> Vec<WordId> {
    let mut out = Vec::new();
    for (b, &word) in m.iter().enumerate() {
        let mut w = word;
        while w != 0 {
            out.push((b * 64 + w.trailing_zeros() as usize) as WordId);
            w &= w - 1;
        }
    }
    out
}

/// The fixed point. Each queued slot exposes its word's letters through the
/// `cross` lane; an unplaced crossing slot ANDs `P(len, offset, letter)` into
/// its mask and is promoted at popcount 1. Reads only `Hot` and the compiled
/// `len` / `cross` lanes.
fn propagate(hot: &Hot, puz: &Puzzle, st: &mut State, queue: &mut Vec<u8>) -> Result<(), Stop> {
    let b = st.blocks;
    while let Some(s) = queue.pop() {
        let s = s as usize;
        let w = st.placed[s];
        for off in 0..puz.len[s] as usize {
            let c = puz.cross[s * STRIDE + off];
            if c == NONE {
                continue;
            }
            let (t, j) = ((c >> 8) as usize, (c & 0xFF) as usize);
            let x = hot.letter(w, off);
            if st.placed[t] != UNSET {
                if hot.letter(st.placed[t], j) != x {
                    return Err(Stop::Contradiction(t as u8));
                }
                continue;
            }
            let m = &mut st.cand[t * b..(t + 1) * b];
            mask_and_assign(m, hot.pop(puz.len[t] as usize, j, x));
            st.events += 1;
            match popcount_batch_u64(m) {
                0 => return Err(Stop::Contradiction(t as u8)),
                1 => {
                    let w2 = first_bit(m);
                    st.placed[t] = w2;
                    queue.push(t as u8);
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Solve from the givens: place them, seed singles, run to the fixed point.
fn solve(hot: &Hot, puz: &Puzzle, givens: &[(u8, WordId)]) -> Result<State, Stop> {
    let mut st = State::new(hot, puz)?;
    let mut queue = Vec::new();
    for &(s, w) in givens {
        st.place(s as usize, w);
        queue.push(s);
    }
    st.seed(&mut queue)?;
    propagate(hot, puz, &mut st, &mut queue)?;
    Ok(st)
}

// ─────────────────────────────── creation ───────────────────────────────

/// Depth-first fill over the same masks, most-constrained slot first, random
/// order within it. `budget` counts tried words. Creation only.
fn random_fill(hot: &Hot, puz: &Puzzle, rng: &mut Rng, budget: &mut usize) -> Option<Vec<WordId>> {
    fn go(hot: &Hot, puz: &Puzzle, st: State, rng: &mut Rng, budget: &mut usize) -> Option<State> {
        let Some(s) = (0..puz.slots())
            .filter(|&s| st.placed[s] == UNSET)
            .min_by_key(|&s| st.count(s))
        else {
            return Some(st);
        };
        let mut cs = bits(st.mask(s));
        rng.shuffle(&mut cs);
        for w in cs {
            if *budget == 0 {
                return None;
            }
            *budget -= 1;
            let mut next = st.clone();
            next.place(s, w);
            let mut q = vec![s as u8];
            if propagate(hot, puz, &mut next, &mut q).is_ok() {
                if let Some(done) = go(hot, puz, next, rng, budget) {
                    return Some(done);
                }
            }
        }
        None
    }
    let mut st = State::new(hot, puz).ok()?;
    let mut q = Vec::new();
    st.seed(&mut q).ok()?;
    propagate(hot, puz, &mut st, &mut q).ok()?;
    go(hot, puz, st, rng, budget).map(|st| st.placed)
}

/// Complete fills consistent with `givens`, up to `cap`; `None` when the
/// budget runs out. The uniqueness oracle for creation.
fn count_fills(
    hot: &Hot,
    puz: &Puzzle,
    givens: &[(u8, WordId)],
    cap: usize,
    budget: &mut usize,
) -> Option<usize> {
    fn go(hot: &Hot, puz: &Puzzle, st: State, cap: usize, budget: &mut usize) -> Option<usize> {
        let Some(s) = (0..puz.slots())
            .filter(|&s| st.placed[s] == UNSET)
            .min_by_key(|&s| st.count(s))
        else {
            return Some(1);
        };
        let mut n = 0;
        for w in bits(st.mask(s)) {
            if *budget == 0 {
                return None;
            }
            *budget -= 1;
            let mut next = st.clone();
            next.place(s, w);
            let mut q = vec![s as u8];
            if propagate(hot, puz, &mut next, &mut q).is_ok() {
                n += go(hot, puz, next, cap - n, budget)?;
                if n >= cap {
                    break;
                }
            }
        }
        Some(n)
    }
    match solve(hot, puz, givens) {
        Ok(st) => go(hot, puz, st, cap, budget),
        Err(_) => Some(0),
    }
}

/// A created puzzle: its compiled layout, its solution and the givens that make
/// the solution unique.
struct Created {
    puz: Puzzle,
    solution: Vec<WordId>,
    givens: Vec<(u8, WordId)>,
}

/// Grid, random fill, then givens in random order until unique. `None` when
/// the budget runs out; the caller draws again.
fn create(hot: &Hot, side: usize, rng: &mut Rng, budget: usize) -> Option<Created> {
    let grid = Grid::random_nyt(side, rng);
    let puz = compile(&grid, hot.lang);
    let mut b = budget;
    let solution = random_fill(hot, &puz, rng, &mut b)?;
    let mut order: Vec<u8> = (0..puz.slots() as u8).collect();
    rng.shuffle(&mut order);
    let mut givens = Vec::new();
    for s in order {
        givens.push((s, solution[s as usize]));
        let mut b = budget;
        if count_fills(hot, &puz, &givens, 2, &mut b)? == 1 {
            return Some(Created {
                puz,
                solution,
                givens,
            });
        }
    }
    unreachable!("every slot given is unique")
}

// ─────────────────────────────── the lane ───────────────────────────────

/// Content lanes beside the epistemic edges: the claim "slot holds word".
#[derive(Default)]
struct Claims {
    slot: Vec<u8>,
    word: Vec<WordId>,
}

/// One instance's claims at the fixed point: given and forced slots one claim
/// each, an unplaced slot its true word as ENTAILED and every other surviving
/// word as CANDIDATE.
fn emit(c: &Created, st: &State, lane: &mut Lane, claims: &mut Claims) {
    for s in 0..c.puz.slots() {
        let given = c.givens.iter().any(|g| g.0 as usize == s);
        let truth = c.solution[s];
        let mut push = |state, w: WordId, lane: &mut Lane| {
            lane.push(state);
            claims.slot.push(s as u8);
            claims.word.push(w);
        };
        if given {
            assert_eq!(st.placed[s], truth);
            push(GIVEN, truth, lane);
            lane.expected[0] += 1;
        } else if st.placed[s] != UNSET {
            assert_eq!(st.placed[s], truth, "a forced word is the true word");
            push(FORCED, truth, lane);
            lane.expected[1] += 1;
        } else {
            let alive = bits(st.mask(s));
            assert!(alive.contains(&truth), "the law never rules out the truth");
            push(ENTAILED, truth, lane);
            lane.expected[2] += 1;
            for w in alive.into_iter().filter(|&w| w != truth) {
                push(CANDIDATE, w, lane);
                lane.expected[3] += 1;
            }
        }
    }
    lane.close_group();
}

// ─────────────────────────────── oracles ───────────────────────────────

/// Every cell shared by two placed words carries the same letter, checked
/// from the Morton cell lanes alone (never from `cross`).
fn cell_consistent(hot: &Hot, puz: &Puzzle, placed: &[WordId]) -> bool {
    let mut at: HashMap<u16, u8> = HashMap::new();
    for (s, &w) in placed.iter().enumerate() {
        if w == UNSET {
            continue;
        }
        for off in 0..puz.len[s] as usize {
            let x = hot.letter(w, off);
            if *at.entry(puz.cell[s * STRIDE + off]).or_insert(x) != x {
                return false;
            }
        }
    }
    true
}

/// Steps 2/2b's method, as the reference: letters on a cell array, every
/// unplaced slot's candidates re-scanned from the vocabulary STRINGS each
/// round, singles placed until nothing changes. Returns the placed word per
/// slot, or `None` on contradiction. (Slot-indexed on purpose: it mirrors the
/// step-2 loop it stands in for.)
#[allow(clippy::needless_range_loop)]
fn cell_scan_solve(cold: &Cold, puz: &Puzzle, givens: &[(u8, WordId)]) -> Option<Vec<WordId>> {
    let words: Vec<(WordId, Vec<char>)> = (0..cold.vocab.len() as WordId)
        .map(|id| (id, cold.word(id)))
        .filter(|(_, w)| admissible(w))
        .map(|(id, w)| (id, w.chars().collect()))
        .collect();
    let mut board: HashMap<u16, char> = HashMap::new();
    let mut placed = vec![UNSET; puz.slots()];
    let put = |s: usize, w: &[char], board: &mut HashMap<u16, char>| -> bool {
        (0..w.len()).all(|o| *board.entry(puz.cell[s * STRIDE + o]).or_insert(w[o]) == w[o])
    };
    for &(s, w) in givens {
        let chars: Vec<char> = cold.word(w).chars().collect();
        if !put(s as usize, &chars, &mut board) {
            return None;
        }
        placed[s as usize] = w;
    }
    loop {
        let mut changed = false;
        for s in 0..puz.slots() {
            if placed[s] != UNSET {
                continue;
            }
            let l = puz.len[s] as usize;
            let fits: Vec<&(WordId, Vec<char>)> = words
                .iter()
                .filter(|(_, w)| {
                    w.len() == l
                        && (0..l).all(|o| {
                            board
                                .get(&puz.cell[s * STRIDE + o])
                                .is_none_or(|&c| c == w[o])
                        })
                })
                .collect();
            match fits.len() {
                0 => return None,
                1 => {
                    if !put(s, &fits[0].1, &mut board) {
                        return None;
                    }
                    placed[s] = fits[0].0;
                    changed = true;
                }
                _ => {}
            }
        }
        if !changed {
            return Some(placed);
        }
    }
}

// ─────────────────────────────── benchmark ───────────────────────────────

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Size sweep: how big can puzzles be created, and how long does solving take.
fn sweep(name: &str, hot: &Hot, cold: &Cold) {
    println!("\n{name}: creation and solving by board size (budget 200,000 tried words per attempt, 20 s per size)");
    println!(
        "  {:>4} {:>7} {:>6} {:>7} {:>9} {:>11} {:>11} {:>9} {:>11}",
        "side",
        "puzzles",
        "slots",
        "givens",
        "solved",
        "create ms",
        "solve us",
        "events",
        "scan us"
    );
    let mut rng = Rng(0x5EED);
    for side in [5, 7, 9, 11, 13, 15] {
        let t0 = Instant::now();
        let (mut made, mut tried) = (Vec::new(), 0usize);
        let mut create_ms = Vec::new();
        while t0.elapsed() < Duration::from_secs(20) && made.len() < 40 {
            tried += 1;
            let t = Instant::now();
            if let Some(c) = create(hot, side, &mut rng, 200_000) {
                create_ms.push(t.elapsed().as_secs_f64() * 1e3);
                made.push(c);
            }
        }
        if made.is_empty() {
            println!(
                "  {side:>4} {:>7} (none created in {tried} attempts within 20 s)",
                0
            );
            continue;
        }
        let (mut solve_us, mut scan_us, mut events, mut solved) = (Vec::new(), Vec::new(), 0, 0);
        for c in &made {
            let t = Instant::now();
            let st =
                black_box(solve(hot, &c.puz, &c.givens)).expect("a created puzzle is consistent");
            solve_us.push(t.elapsed().as_secs_f64() * 1e6);
            events += st.events;
            if st.placed.iter().all(|&w| w != UNSET) {
                solved += 1;
            }
            assert!(cell_consistent(hot, &c.puz, &st.placed));
            if side <= 9 {
                let t = Instant::now();
                let scan = cell_scan_solve(cold, &c.puz, &c.givens).expect("consistent");
                scan_us.push(t.elapsed().as_secs_f64() * 1e6);
                assert_eq!(scan, st.placed, "mask fixed point == cell-scan fixed point");
            }
        }
        let n = made.len();
        let slots = made.iter().map(|c| c.puz.slots()).sum::<usize>() as f64 / n as f64;
        let givens = made.iter().map(|c| c.givens.len()).sum::<usize>() as f64 / n as f64;
        let scan = if scan_us.is_empty() {
            "-".to_string()
        } else {
            format!("{:.0}", median(scan_us))
        };
        println!(
            "  {side:>4} {n:>4}/{tried:<3} {slots:>6.1} {givens:>7.1} {:>8.0}% {:>11.1} {:>11.1} {:>9.1} {:>11}",
            100.0 * solved as f64 / n as f64,
            median(create_ms),
            median(solve_us),
            events as f64 / n as f64,
            scan
        );
    }
}

/// Filter micro-benchmark: one slot pattern (length + revealed letters) as a
/// mask AND chain vs a direct scan over the vocabulary's spell lane.
fn filter_bench(name: &str, hot: &Hot) {
    let mut rng = Rng(0xF17);
    let words: Vec<WordId> = (0..hot.len.len() as WordId)
        .filter(|&w| hot.len[w as usize] as usize >= 5)
        .collect();
    let patterns: Vec<(usize, Vec<(usize, u8)>)> = (0..2000)
        .map(|_| {
            let w = words[rng.below(words.len() as u64) as usize];
            let l = hot.len[w as usize] as usize;
            let k = 1 + rng.below(3) as usize;
            let fixed = (0..k)
                .map(|_| {
                    let o = rng.below(l as u64) as usize;
                    (o, hot.letter(w, o))
                })
                .collect();
            (l, fixed)
        })
        .collect();
    let mut buf = vec![0u64; hot.blocks];
    let mut claims = 0u64;
    let t = Instant::now();
    for (l, fixed) in &patterns {
        buf.copy_from_slice(&hot.all[*l]);
        for &(o, x) in fixed {
            mask_and_assign(&mut buf, hot.pop(*l, o, x));
        }
        claims += popcount_batch_u64(black_box(&buf));
    }
    let mask_ns = t.elapsed().as_nanos() as f64;
    let t = Instant::now();
    let mut scanned = 0u64;
    for (l, fixed) in &patterns {
        scanned += (0..hot.len.len())
            .filter(|&id| {
                hot.len[id] as usize == *l
                    && fixed.iter().all(|&(o, x)| hot.spell[id * STRIDE + o] == x)
            })
            .count() as u64;
    }
    let scan_ns = t.elapsed().as_nanos() as f64;
    assert_eq!(black_box(claims), scanned, "mask filter == direct scan");
    let p = patterns.len() as f64;
    println!(
        "  {name}: candidate filter over {} patterns, {:.1} surviving claims each",
        patterns.len(),
        claims as f64 / p
    );
    println!(
        "    mask AND chain + popcount: {:>9.0} ns/pattern  {:>7.2} ns/claim",
        mask_ns / p,
        mask_ns / claims as f64
    );
    println!(
        "    direct scan of the spell lane: {:>5.0} ns/pattern  {:>7.2} ns/claim",
        scan_ns / p,
        scan_ns / claims as f64
    );
}

/// About a million claims at one size, folded the shared three ways.
fn lane_run(name: &str, hot: &Hot, side: usize) {
    let decl = declarations(CROSSWORD_CLASS);
    admit(&decl, CROSSWORD_CLASS).expect("the crossword class declares the canonical reading");
    let mut rng = Rng(0x1A4E);
    let (mut lane, mut claims, mut slots) = (Lane::default(), Claims::default(), Vec::new());
    let t = Instant::now();
    while lane.edges.len() < 1_000_000 {
        let Some(c) = create(hot, side, &mut rng, 200_000) else {
            continue;
        };
        let st = solve(hot, &c.puz, &c.givens).expect("consistent");
        slots.push(c.puz.slots());
        emit(&c, &st, &mut lane, &mut claims);
    }
    println!(
        "\n  {name}: {} claims from {} {side}x{side} puzzles, built in {:.2?}",
        lane.edges.len(),
        lane.groups,
        t.elapsed()
    );
    let counts = check_three_ways(&lane, &decl, CROSSWORD_CLASS);
    let asserted: Vec<usize> = per_group(&lane, GIVEN.bit() | FORCED.bit() | ENTAILED.bit())
        .into_iter()
        .map(|n| n as usize)
        .collect();
    assert_eq!(asserted, slots);
    println!("  every puzzle: one asserted claim per slot (group fold)");
    report(&lane, &decl, CROSSWORD_CLASS, &counts);
}

fn run_language(name: &str, hot: &Hot, cold: &Cold) {
    let words = hot.len.iter().filter(|&&l| l > 0).count();
    println!(
        "\n=== {name}: {} WordIds, {} crossword words, {} letters ({})",
        cold.vocab.len(),
        words,
        cold.letters.len(),
        cold.letters.iter().collect::<String>()
    );
    filter_bench(name, hot);
    sweep(name, hot, cold);
    lane_run(name, hot, 7);
}

fn main() {
    println!("D-PUZZLE-0 step 3: crossword propagation as Cartesian-addressed population masking");
    let (hot, cold) = build_lexicon(Lang::En, &english_ranked());
    run_language("English, DeepNSM-v2 academic_20k", &hot, &cold);
    match std::env::var("DEREKO_PATH") {
        Ok(path) => {
            let text = std::fs::read_to_string(&path).expect("DEREKO_PATH is readable");
            let (hot, cold) = build_lexicon(Lang::De, &german_ranked(&text));
            run_language("German, DeReKo-2014 top 20k (runtime only)", &hot, &cold);
        }
        Err(_) => println!("\nGerman skipped: set DEREKO_PATH to the DeReKo-2014 .freq file"),
    }
}

#[cfg(test)]
mod tests {
    use super::population_fold::{count_in, queries, raw5, UNKNOWN_CAUSES};
    use super::*;
    use causal_edge::layout::EPISTEMIC_MASK;

    fn english() -> (Hot, Cold) {
        build_lexicon(Lang::En, &english_ranked())
    }

    fn words(ws: &[&str]) -> Vec<String> {
        ws.iter().map(|w| w.to_string()).collect()
    }

    /// One across slot (row 0) crossing one down slot (column 1) at its
    /// second letter.
    fn plus() -> Grid {
        Grid::from_rows(&["...", "#.#", "#.#"])
    }

    fn word(cold: &Cold, w: &str) -> WordId {
        cold.vocab.id(w).expect("in the fixture")
    }

    /// The English vocabulary is DeepNSM-v2's: 18,555 distinct ids in
    /// frequency order, and the hyphen/apostrophe rows keep ids but no length.
    #[test]
    fn english_is_the_deepnsm_v2_academic_vocabulary() {
        let (hot, cold) = english();
        assert_eq!(cold.vocab.len(), 18_555);
        assert_eq!(cold.word(0), "the");
        let id = word(&cold, "so-called");
        assert_eq!(hot.len[id as usize], 0);
        // The letters come from the data: a-z plus the é of `sauté` and
        // `cliché` (DeepNSM-v2 lowercases and does not fold accents).
        let mut az: Vec<char> = ('a'..='z').collect();
        az.push('é');
        assert_eq!(cold.letters, az);
        assert_eq!(hot.len[word(&cold, "cliché") as usize], 6);
    }

    /// The letter codes round-trip: every crossword word decodes from its
    /// spell lane back to its own string.
    #[test]
    fn every_word_decodes_from_its_letter_codes() {
        let (hot, cold) = english();
        let mut n = 0;
        for id in 0..cold.vocab.len() as WordId {
            let l = hot.len[id as usize] as usize;
            if l == 0 {
                continue;
            }
            let back: String = (0..l)
                .map(|o| cold.letters[hot.letter(id, o) as usize - 1])
                .collect();
            assert_eq!(back, cold.word(id));
            n += 1;
        }
        assert!(n > 10_000);
    }

    /// Indexed positional populations equal a direct scan over the vocabulary
    /// strings for random patterns.
    #[test]
    fn positional_populations_equal_a_string_scan() {
        let (hot, cold) = english();
        let mut rng = Rng(2);
        for _ in 0..500 {
            let l = MIN_WORD + rng.below((MAX_LEN - MIN_WORD + 1) as u64) as usize;
            let k = rng.below(4) as usize;
            let fixed: Vec<(usize, char)> = (0..k)
                .map(|_| {
                    let o = rng.below(l as u64) as usize;
                    (
                        o,
                        cold.letters[rng.below(cold.letters.len() as u64) as usize],
                    )
                })
                .collect();
            let mut m = hot.all[l].clone();
            for &(o, c) in &fixed {
                mask_and_assign(&mut m, hot.pop(l, o, cold.code(c).unwrap()));
            }
            let scan: Vec<WordId> = (0..cold.vocab.len() as WordId)
                .filter(|&id| {
                    let w: Vec<char> = cold.word(id).chars().collect();
                    admissible(cold.word(id))
                        && w.len() == l
                        && fixed.iter().all(|&(o, c)| w[o] == c)
                })
                .collect();
            assert_eq!(bits(&m), scan, "pattern {l} {fixed:?}");
        }
    }

    /// A slot narrowed to one candidate is promoted.
    #[test]
    fn a_single_candidate_is_forced() {
        let (hot, cold) = build_lexicon(Lang::En, &words(&["cat", "cow", "dog", "ant"]));
        let puz = compile(&plus(), Lang::En);
        let (across, down) = (0u8, 1u8);
        assert_eq!(puz.len, [3, 3]);
        let st = solve(&hot, &puz, &[(across, word(&cold, "cat"))]).unwrap();
        assert_eq!(st.placed[down as usize], word(&cold, "ant"));
    }

    /// A forced word constrains EVERY crossing slot, not just the first.
    #[test]
    fn a_placed_word_masks_every_crossing() {
        let (hot, cold) = build_lexicon(
            Lang::En,
            &words(&["cat", "cow", "cub", "toe", "tea", "dog", "ant"]),
        );
        let puz = compile(&Grid::from_rows(&["...", ".#.", ".#."]), Lang::En);
        assert_eq!(puz.len, [3, 3, 3]);
        let st = solve(&hot, &puz, &[(0, word(&cold, "cat"))]).unwrap();
        let first = |s: usize| -> Vec<char> {
            bits(st.mask(s))
                .into_iter()
                .map(|w| cold.word(w).chars().next().unwrap())
                .collect()
        };
        assert!(first(1).iter().all(|&c| c == 'c') && first(1).len() == 3);
        assert!(first(2).iter().all(|&c| c == 't') && first(2).len() == 2);
    }

    /// No candidate left is a contradiction, never a silent empty slot.
    #[test]
    fn an_empty_slot_is_a_contradiction() {
        let (hot, cold) = build_lexicon(Lang::En, &words(&["cat", "cow", "dog"]));
        let puz = compile(&plus(), Lang::En);
        assert_eq!(
            solve(&hot, &puz, &[(0, word(&cold, "cat"))]).err(),
            Some(Stop::Contradiction(1))
        );
    }

    /// The crossing comes from the compiled lane. Re-pointing one crossing
    /// lets an incompatible word survive, and the cell oracle sees it; with
    /// the lane emptied nothing propagates at all.
    #[test]
    fn crossings_are_read_from_the_compiled_lane() {
        let (hot, cold) = build_lexicon(Lang::En, &words(&["cat", "cow", "dog", "ant"]));
        let puz = compile(&plus(), Lang::En);
        let given = [(0u8, word(&cold, "cat"))];
        let ok = solve(&hot, &puz, &given).unwrap();
        assert!(cell_consistent(&hot, &puz, &ok.placed));

        // across offset 1 really crosses down offset 0; re-point the pair
        // (both directions) at down offset 1.
        let mut wrong = puz.clone();
        assert_eq!(wrong.cross[1], FacetTier { hi: 1, lo: 0 }.as_u16());
        wrong.cross[1] = FacetTier { hi: 1, lo: 1 }.as_u16();
        wrong.cross[STRIDE] = NONE;
        wrong.cross[STRIDE + 1] = FacetTier { hi: 0, lo: 1 }.as_u16();
        let bad = solve(&hot, &wrong, &given).unwrap();
        assert_eq!(bad.placed[1], word(&cold, "cat"));
        assert!(!cell_consistent(&hot, &wrong, &bad.placed));

        let mut cut = puz.clone();
        cut.cross.fill(NONE);
        let none = solve(&hot, &cut, &given).unwrap();
        assert_eq!(none.placed[1], UNSET);
        assert_eq!(none.events, 0);
    }

    /// Crossing tiles are symmetric and land on the same Morton cell.
    #[test]
    fn every_crossing_names_the_same_cell_both_ways() {
        let mut rng = Rng(4);
        for side in [5, 7, 9, 15] {
            let puz = compile(&Grid::random_nyt(side, &mut rng), Lang::En);
            for s in 0..puz.slots() {
                for o in 0..puz.len[s] as usize {
                    let c = puz.cross[s * STRIDE + o];
                    assert_ne!(c, NONE, "an NYT grid checks every square");
                    let (t, j) = ((c >> 8) as usize, (c & 0xFF) as usize);
                    assert_eq!(puz.cell[t * STRIDE + j], puz.cell[s * STRIDE + o]);
                    assert_eq!(
                        puz.cross[t * STRIDE + j],
                        FacetTier {
                            hi: s as u8,
                            lo: o as u8
                        }
                        .as_u16()
                    );
                }
            }
        }
    }

    /// Languages are separate populations: German keeps ä ö ü ß as letters,
    /// the same WordId names different words, and a puzzle is refused against
    /// the other language's populations.
    #[test]
    fn languages_do_not_share_ordinals_or_populations() {
        let (en, en_cold) = build_lexicon(Lang::En, &words(&["the", "and", "house", "street"]));
        let (de, de_cold) = build_lexicon(
            Lang::De,
            &words(&["der", "und", "haus", "straße", "über", "größe", "mädchen"]),
        );
        assert_ne!(en_cold.word(0), de_cold.word(0));
        for c in ['ä', 'ö', 'ü', 'ß'] {
            assert!(de_cold.code(c).is_some() && en_cold.code(c).is_none());
        }
        let de_puz = compile(&plus(), Lang::De);
        assert_eq!(State::new(&en, &de_puz).err(), Some(Stop::Language));
        assert!(State::new(&de, &de_puz).is_ok());
    }

    /// Bits 59..63 carry state only: every claim edge is zero outside them,
    /// its code is one of the four shared states, and two different words in
    /// the same state have the identical edge.
    #[test]
    fn ce64_carries_state_never_content() {
        let (hot, _) = english();
        let mut rng = Rng(7);
        let (mut lane, mut claims) = (Lane::default(), Claims::default());
        while lane.groups < 5 {
            if let Some(c) = create(&hot, 5, &mut rng, 200_000) {
                let st = solve(&hot, &c.puz, &c.givens).unwrap();
                emit(&c, &st, &mut lane, &mut claims);
            }
        }
        let codes: Vec<u32> = population_fold::STATES
            .iter()
            .map(|s| u32::from(s.raw()))
            .collect();
        for e in &lane.edges {
            assert_eq!(e.0 & !EPISTEMIC_MASK, 0);
            assert!(codes.contains(&raw5(*e)));
        }
        let cands: Vec<usize> = (0..lane.edges.len())
            .filter(|&i| raw5(lane.edges[i]) == u32::from(CANDIDATE.raw()))
            .collect();
        let (a, b) = (cands[0], cands[1]);
        assert_ne!(claims.word[a], claims.word[b]);
        assert_eq!(lane.edges[a], lane.edges[b]);
        // reserved codes 24..31 never appear
        assert_eq!(count_in(&lane.edges, !0u32 << 24), 0);
        assert_eq!(count_in(&lane.edges, UNKNOWN_CAUSES.bit()), 0);
    }

    /// The mask fixed point equals steps 2/2b's cell-scan fixed point on
    /// created puzzles, and every created puzzle is unique and consistent.
    #[test]
    fn mask_propagation_equals_the_cell_scan_oracle() {
        let (hot, cold) = english();
        let mut rng = Rng(9);
        let mut made = 0;
        while made < 8 {
            let side = if made % 2 == 0 { 5 } else { 7 };
            let Some(c) = create(&hot, side, &mut rng, 200_000) else {
                continue;
            };
            made += 1;
            let st = solve(&hot, &c.puz, &c.givens).unwrap();
            assert_eq!(
                cell_scan_solve(&cold, &c.puz, &c.givens).unwrap(),
                st.placed
            );
            assert!(cell_consistent(&hot, &c.puz, &c.solution));
            let mut b = 1_000_000;
            assert_eq!(count_fills(&hot, &c.puz, &c.givens, 3, &mut b), Some(1));
        }
    }

    /// The mask grid rules agree with a cell-by-cell check, and refuse.
    #[test]
    fn mask_grid_rules_equal_a_cell_check() {
        fn cell_valid(g: &Grid) -> bool {
            let n = g.side;
            let white = |r: usize, c: usize| g.rows[r] >> c & 1 == 1;
            let sym = (0..n).all(|r| (0..n).all(|c| white(r, c) == white(n - 1 - r, n - 1 - c)));
            let mut runs_ok = true;
            for across in [true, false] {
                for a in 0..n {
                    let mut run = 0;
                    for b in 0..=n {
                        let w = b < n && if across { white(a, b) } else { white(b, a) };
                        if w {
                            run += 1;
                        } else {
                            if run > 0 && run < MIN_WORD {
                                runs_ok = false;
                            }
                            run = 0;
                        }
                    }
                }
            }
            let cells: Vec<(usize, usize)> = (0..n)
                .flat_map(|r| (0..n).map(move |c| (r, c)))
                .filter(|&(r, c)| white(r, c))
                .collect();
            let mut seen = vec![cells[0]];
            let mut i = 0;
            while i < seen.len() {
                let (r, c) = seen[i];
                for (dr, dc) in [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                    let (r2, c2) = (r as i32 + dr, c as i32 + dc);
                    if r2 >= 0 && c2 >= 0 && (r2 as usize) < n && (c2 as usize) < n {
                        let p = (r2 as usize, c2 as usize);
                        if white(p.0, p.1) && !seen.contains(&p) {
                            seen.push(p);
                        }
                    }
                }
                i += 1;
            }
            let blacks = n * n - cells.len();
            sym && runs_ok && seen.len() == cells.len() && blacks * MAX_BLACK_DIVISOR <= n * n
        }
        let mut rng = Rng(11);
        let (mut valid, mut invalid) = (0, 0);
        for _ in 0..3000 {
            let side = [5, 7, 9][rng.below(3) as usize];
            let mut g = Grid {
                side,
                rows: vec![((1u32 << side) - 1) as u16; side],
            };
            for i in 0..side * side {
                if rng.below(9) == 0 {
                    g.rows[i / side] &= !(1 << (i % side));
                }
            }
            if g.rows.iter().all(|&r| r == 0) {
                continue;
            }
            let ok = g.is_nyt_valid();
            assert_eq!(ok, cell_valid(&g), "{:?}", g.rows);
            if ok {
                valid += 1;
            } else {
                invalid += 1;
            }
        }
        assert!(
            valid > 10 && invalid > 10,
            "{valid} valid, {invalid} invalid"
        );
    }

    /// Solving runs without the strings: `Cold` is dropped before the solve,
    /// and the result is the same.
    #[test]
    fn the_hot_path_needs_no_strings() {
        let (hot, cold) = english();
        let mut rng = Rng(13);
        let c = loop {
            if let Some(c) = create(&hot, 7, &mut rng, 200_000) {
                break c;
            }
        };
        let expect = cell_scan_solve(&cold, &c.puz, &c.givens).unwrap();
        drop(cold);
        assert_eq!(solve(&hot, &c.puz, &c.givens).unwrap().placed, expect);
    }

    /// Three ways, partition, declaration gate: the shared fold, unchanged.
    #[test]
    fn the_shared_fold_counts_the_crossword_lane_three_ways() {
        let (hot, _) = english();
        let mut rng = Rng(17);
        let (mut lane, mut claims) = (Lane::default(), Claims::default());
        while lane.edges.len() < 20_000 {
            if let Some(c) = create(&hot, 5, &mut rng, 200_000) {
                let st = solve(&hot, &c.puz, &c.givens).unwrap();
                emit(&c, &st, &mut lane, &mut claims);
            }
        }
        let decl = declarations(CROSSWORD_CLASS);
        let counts = check_three_ways(&lane, &decl, CROSSWORD_CLASS);
        assert!(counts.iter().all(|&n| n > 0), "a question went unexercised");
        let union = queries()[1..].iter().fold(0, |u, q| u | q.population);
        assert_eq!(count_in(&lane.edges, union), lane.edges.len());
        assert!(admit(&decl, CROSSWORD_CLASS).is_some());
        assert!(admit(&decl, 0x0906).is_none());
    }
}
