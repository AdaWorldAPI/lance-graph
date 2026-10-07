//! The crossword core shared by the D-PUZZLE-0 crossword probes: the
//! `MooreSymbol8` codebook, lexicons and `P(L, i, x)` populations, NYT grids,
//! compilation into slot / crossing lanes, the token-mask state with its fixed
//! point, and puzzle creation. Moved unchanged from
//! `crossword_mask_propagation_probe.rs` (#1387) so the attention probe
//! (D-PUZZLE-ATTN-0) runs on the same code; included with `#[path]`.
#![allow(dead_code)]

use std::collections::HashMap;
use std::time::Instant;

use deepnsm_v2::vocab::{PaletteVocab, WordId};
use lance_graph_contract::facet::FacetTier;
use lance_graph_contract::morton8x8::Morton8x8;
use ndarray::simd::{mask_and_assign, popcount_batch_u64};

use super::population_fold::Rng;

/// NYT rules: shortest word; most black squares (a sixth).
pub const MIN_WORD: usize = 3;
pub const MAX_BLACK_DIVISOR: usize = 6;
/// Longest slot (21×21 Sunday board) and the per-slot stride of position lanes.
pub const MAX_LEN: usize = 21;
pub const STRIDE: usize = 32;
/// No crossing / no occupant.
pub const NONE: u16 = u16::MAX;
/// No word placed in this slot.
pub const UNSET: WordId = WordId::MAX;
/// German forms kept.
pub const GERMAN_VOCAB: usize = 20_000;

pub const ACADEMIC: &str = include_str!("../../../deepnsm/word_frequency/academic_20k.csv");

// ─────────────────────────────── symbols ───────────────────────────────

/// The declared letter reading of one Moore-local byte: `0` unknown, `1..=26`
/// a–z, `27` ä, `28` ö, `29` ü, `30` ß, `31..=255` reserved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct MooreSymbol8(pub u8);

impl MooreSymbol8 {
    pub const UNKNOWN: Self = Self(0);
    /// Codes in use, `0..=30`.
    pub const COUNT: usize = 31;

    /// The symbol of a lowercase character after accent folding.
    pub fn of(c: char) -> Option<Self> {
        match fold(c) {
            c @ 'a'..='z' => Some(Self(c as u8 - b'a' + 1)),
            'ä' => Some(Self(27)),
            'ö' => Some(Self(28)),
            'ü' => Some(Self(29)),
            'ß' => Some(Self(30)),
            _ => None,
        }
    }

    /// The letter of a symbol; `None` for unknown and reserved codes.
    pub fn letter(self) -> Option<char> {
        match self.0 {
            1..=26 => Some((b'a' + self.0 - 1) as char),
            27 => Some('ä'),
            28 => Some('ö'),
            29 => Some('ü'),
            30 => Some('ß'),
            _ => None,
        }
    }
}

/// Crossword spelling ignores accents: an accented Latin letter folds to its
/// base. ä ö ü ß are letters of their own and do not fold.
pub fn fold(c: char) -> char {
    match c {
        'à' | 'á' | 'â' | 'ã' | 'å' | 'ā' => 'a',
        'ç' => 'c',
        'è' | 'é' | 'ê' | 'ë' | 'ē' => 'e',
        'ì' | 'í' | 'î' | 'ï' => 'i',
        'ñ' => 'n',
        'ò' | 'ó' | 'ô' | 'õ' => 'o',
        'ù' | 'ú' | 'û' => 'u',
        'ý' | 'ÿ' => 'y',
        c => c,
    }
}

/// A word's crossword spelling, or `None` if it is not a crossword word
/// (wrong length, or a character with no symbol).
pub fn spelling(w: &str) -> Option<Vec<MooreSymbol8>> {
    let s: Option<Vec<MooreSymbol8>> = w.chars().map(MooreSymbol8::of).collect();
    s.filter(|s| (MIN_WORD..=MAX_LEN).contains(&s.len()))
}

// ─────────────────────────────── vocabulary ───────────────────────────────

/// The language a lexicon and a puzzle belong to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    En,
    De,
}

/// Cold side: the strings. Construction, the literal arm and checks only.
pub struct Cold {
    pub vocab: PaletteVocab,
}

impl Cold {
    pub fn word(&self, w: WordId) -> &str {
        self.vocab.word(w).expect("word id in range")
    }
}

/// Hot side: integers and masks only. No `String`, no `char`.
pub struct Hot {
    pub lang: Lang,
    pub blocks: usize,
    /// Per `WordId`: crossword length, 0 = not a crossword word.
    pub len: Vec<u8>,
    /// `[id * STRIDE + offset]` = the symbol.
    pub spell: Vec<MooreSymbol8>,
    /// By length: every crossword word of that length.
    pub all: Vec<Vec<u64>>,
    /// By length: `[(offset * COUNT + symbol) * blocks ..]` = `P(L, offset, symbol)`.
    pub at: Vec<Vec<u64>>,
}

impl Hot {
    pub fn pop(&self, len: usize, offset: usize, x: MooreSymbol8) -> &[u64] {
        let i = (offset * MooreSymbol8::COUNT + x.0 as usize) * self.blocks;
        &self.at[len][i..i + self.blocks]
    }

    pub fn letter(&self, w: WordId, offset: usize) -> MooreSymbol8 {
        self.spell[w as usize * STRIDE + offset]
    }

    /// Bytes held by the populations (the fixed per-language cost).
    pub fn population_bytes(&self) -> usize {
        8 * (self.all.iter().map(Vec::len).sum::<usize>()
            + self.at.iter().map(Vec::len).sum::<usize>())
    }
}

/// Both sides from a frequency-ranked word list (most frequent first).
pub fn build_lexicon(lang: Lang, ranked: &[String]) -> (Hot, Cold) {
    let mut vocab = PaletteVocab::new();
    vocab.from_frequency_ranked(ranked.iter().map(String::as_str));
    let n = vocab.len();
    assert!(n < UNSET as usize, "WordId::MAX is the empty marker");
    let blocks = n.div_ceil(64);
    let mut hot = Hot {
        lang,
        blocks,
        len: vec![0; n],
        spell: vec![MooreSymbol8::UNKNOWN; n * STRIDE],
        all: vec![Vec::new(); MAX_LEN + 1],
        at: vec![Vec::new(); MAX_LEN + 1],
    };
    for l in MIN_WORD..=MAX_LEN {
        hot.all[l] = vec![0; blocks];
        hot.at[l] = vec![0; l * MooreSymbol8::COUNT * blocks];
    }
    for id in 0..n {
        let Some(sp) = spelling(vocab.word(id as WordId).expect("in range")) else {
            continue;
        };
        let l = sp.len();
        hot.len[id] = l as u8;
        let bit = 1u64 << (id % 64);
        hot.all[l][id / 64] |= bit;
        for (i, &x) in sp.iter().enumerate() {
            hot.spell[id * STRIDE + i] = x;
            hot.at[l][(i * MooreSymbol8::COUNT + x.0 as usize) * blocks + id / 64] |= bit;
        }
    }
    (hot, Cold { vocab })
}

/// English: DeepNSM-v2's academic vocabulary, built as `genre_shapes` builds it.
pub fn english_ranked() -> Vec<String> {
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
pub fn german_ranked(text: &str) -> Vec<String> {
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
pub struct Grid {
    pub side: usize,
    pub rows: Vec<u32>,
}

/// Every white bit of a line lies in a run of at least three: the OR of the
/// three-wide windows covers the line.
pub fn line_runs_ok(w: u32) -> bool {
    let t = w & (w >> 1) & (w >> 2);
    (t | (t << 1) | (t << 2)) == w
}

impl Grid {
    #[cfg(test)]
    pub fn from_rows(rows: &[&str]) -> Self {
        let rows: Vec<u32> = rows
            .iter()
            .map(|r| {
                r.bytes()
                    .enumerate()
                    .fold(0u32, |m, (c, b)| m | (u32::from(b == b'.') << c))
            })
            .collect();
        Self {
            side: rows.len(),
            rows,
        }
    }

    pub fn full_line(side: usize) -> u32 {
        ((1u64 << side) - 1) as u32
    }

    pub fn cols(&self) -> Vec<u32> {
        (0..self.side)
            .map(|c| (0..self.side).fold(0u32, |m, r| m | (((self.rows[r] >> c) & 1) << r)))
            .collect()
    }

    pub fn blacks(&self) -> usize {
        self.side * self.side
            - self
                .rows
                .iter()
                .map(|r| r.count_ones() as usize)
                .sum::<usize>()
    }

    /// 180° symmetry: row `r` reversed is row `side - 1 - r`.
    pub fn symmetric(&self) -> bool {
        let rev = |w: u32| w.reverse_bits() >> (32 - self.side);
        (0..self.side).all(|r| rev(self.rows[r]) == self.rows[self.side - 1 - r])
    }

    /// Every across and down run is at least `MIN_WORD` long.
    pub fn runs_ok(&self) -> bool {
        self.rows
            .iter()
            .chain(self.cols().iter())
            .all(|&w| line_runs_ok(w))
    }

    /// One white region: grow from the first white cell by row-mask dilation.
    pub fn connected(&self) -> bool {
        let Some(r0) = self.rows.iter().position(|&w| w != 0) else {
            return false;
        };
        let full = Self::full_line(self.side);
        let mut reach = vec![0u32; self.side];
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

    pub fn is_nyt_valid(&self) -> bool {
        self.symmetric()
            && self.runs_ok()
            && self.connected()
            && self.blacks() * MAX_BLACK_DIVISOR <= self.side * self.side
    }

    /// Symmetric black pairs drawn until the grid is NYT-valid with at least
    /// one black square.
    pub fn random_nyt(side: usize, rng: &mut Rng) -> Self {
        let cells = side * side;
        loop {
            let mut g = Self {
                side,
                rows: vec![Self::full_line(side); side],
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
pub struct Puzzle {
    pub lang: Lang,
    /// Per slot: its length.
    pub len: Vec<u8>,
    /// `[slot * STRIDE + offset]`: the crossing `(slot:offset)` tile, or `NONE`.
    pub cross: Vec<u16>,
    /// `[slot * STRIDE + offset]`: the Morton cell code.
    pub cell: Vec<u16>,
    /// `[cell]`: the (at most two) `(slot:offset)` tiles on a cell.
    pub occupant: Vec<[u16; 2]>,
}

impl Puzzle {
    pub fn slots(&self) -> usize {
        self.len.len()
    }
}

pub fn tile(s: usize, o: usize) -> u16 {
    FacetTier {
        hi: s as u8,
        lo: o as u8,
    }
    .as_u16()
}

pub fn untile(t: u16) -> (usize, usize) {
    ((t >> 8) as usize, (t & 0xFF) as usize)
}

/// Compile a grid: slots from the row and column masks, each position's cell
/// from `Morton8x8::checked_offset`, crossings from equal cell codes.
pub fn compile(grid: &Grid, lang: Lang) -> Puzzle {
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
    let tiles = cell
        .iter()
        .filter(|&&c| c != NONE)
        .max()
        .map_or(0, |&m| m as usize + 1);
    let mut occupant = vec![[NONE; 2]; tiles];
    let mut cross = vec![NONE; len.len() * STRIDE];
    for (s, &l) in len.iter().enumerate() {
        for o in 0..l as usize {
            let c = cell[s * STRIDE + o] as usize;
            let slot = if occupant[c][0] == NONE { 0 } else { 1 };
            assert_eq!(occupant[c][slot], NONE, "at most two positions per cell");
            occupant[c][slot] = tile(s, o);
            if slot == 1 {
                let other = occupant[c][0];
                cross[s * STRIDE + o] = other;
                let (t, j) = untile(other);
                cross[t * STRIDE + j] = tile(s, o);
            }
        }
    }
    Puzzle {
        lang,
        len,
        cross,
        cell,
        occupant,
    }
}

/// Why a solve did not start or did not finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// The puzzle and the populations are different languages.
    Language,
    /// A slot was left with no candidate, or two letters disagree on a cell.
    Contradiction(u8),
}

/// Per-slot candidate masks plus the placed word, shared by arms A and D.
#[derive(Clone)]
pub struct State {
    pub blocks: usize,
    pub cand: Vec<u64>,
    pub placed: Vec<WordId>,
    pub ands: usize,
}

impl State {
    pub fn new(hot: &Hot, puz: &Puzzle) -> Result<Self, Stop> {
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
            ands: 0,
        })
    }

    pub fn mask(&self, s: usize) -> &[u64] {
        &self.cand[s * self.blocks..(s + 1) * self.blocks]
    }

    pub fn count(&self, s: usize) -> u64 {
        popcount_batch_u64(self.mask(s))
    }

    pub fn place(&mut self, s: usize, w: WordId) {
        let b = self.blocks;
        self.cand[s * b..(s + 1) * b].fill(0);
        self.cand[s * b + w as usize / 64] = 1 << (w % 64);
        self.placed[s] = w;
    }

    /// Promote every unplaced slot already down to one candidate.
    pub fn seed(&mut self, queue: &mut Vec<u8>) -> Result<(), Stop> {
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

    /// AND one population into slot `t`; promote at popcount 1.
    pub fn narrow(&mut self, t: usize, p: &[u64], queue: &mut Vec<u8>) -> Result<(), Stop> {
        let b = self.blocks;
        let m = &mut self.cand[t * b..(t + 1) * b];
        mask_and_assign(m, p);
        self.ands += 1;
        match popcount_batch_u64(m) {
            0 => Err(Stop::Contradiction(t as u8)),
            1 => {
                self.placed[t] = first_bit(m);
                queue.push(t as u8);
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

pub fn first_bit(m: &[u64]) -> WordId {
    let (b, w) = m
        .iter()
        .enumerate()
        .find(|(_, &w)| w != 0)
        .expect("non-empty mask");
    (b * 64 + w.trailing_zeros() as usize) as WordId
}

pub fn bits(m: &[u64]) -> Vec<WordId> {
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

/// Arm A's fixed point: each queued slot hands its letter at every crossed
/// offset to the crossing `(slot:offset)` read from `cross`.
pub fn propagate_token(
    hot: &Hot,
    puz: &Puzzle,
    st: &mut State,
    queue: &mut Vec<u8>,
) -> Result<(), Stop> {
    while let Some(s) = queue.pop() {
        let s = s as usize;
        let w = st.placed[s];
        for off in 0..puz.len[s] as usize {
            let c = puz.cross[s * STRIDE + off];
            if c == NONE {
                continue;
            }
            let (t, j) = untile(c);
            let x = hot.letter(w, off);
            if st.placed[t] != UNSET {
                if hot.letter(st.placed[t], j) != x {
                    return Err(Stop::Contradiction(t as u8));
                }
                continue;
            }
            st.narrow(t, hot.pop(puz.len[t] as usize, j, x), queue)?;
        }
    }
    Ok(())
}

pub fn start(hot: &Hot, puz: &Puzzle, givens: &[(u8, WordId)]) -> Result<(State, Vec<u8>), Stop> {
    let mut st = State::new(hot, puz)?;
    let mut queue = Vec::new();
    for &(s, w) in givens {
        st.place(s as usize, w);
        queue.push(s);
    }
    st.seed(&mut queue)?;
    Ok((st, queue))
}

// ─────────────────────────────── creation ───────────────────────────────

/// Why creation gave up on one attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Miss {
    /// No complete fill found within the budget.
    Fill,
    /// A fill was found, but uniqueness could not be decided within the budget.
    Unique,
}

/// Crossword fills never repeat an entry: no two placed slots hold the same
/// word. Both creation searches apply this rule, so the generator and the
/// uniqueness count agree on what a fill is.
pub fn no_repeats(placed: &[WordId]) -> bool {
    let mut seen: Vec<WordId> = placed.iter().copied().filter(|&w| w != UNSET).collect();
    let n = seen.len();
    seen.sort_unstable();
    seen.dedup();
    seen.len() == n
}

/// Depth-first fill over arm A's masks, most-constrained slot first, random
/// order within it. `budget` counts tried words. Creation only.
pub fn random_fill(
    hot: &Hot,
    puz: &Puzzle,
    rng: &mut Rng,
    budget: &mut usize,
) -> Option<Vec<WordId>> {
    pub fn go(
        hot: &Hot,
        puz: &Puzzle,
        st: State,
        rng: &mut Rng,
        budget: &mut usize,
    ) -> Option<State> {
        let Some(s) = (0..puz.slots())
            .filter(|&s| st.placed[s] == UNSET)
            .min_by_key(|&s| st.count(s))
        else {
            return no_repeats(&st.placed).then_some(st);
        };
        let mut cs = bits(st.mask(s));
        cs.retain(|w| !st.placed.contains(w));
        rng.shuffle(&mut cs);
        for w in cs {
            if *budget == 0 {
                return None;
            }
            *budget -= 1;
            let mut next = st.clone();
            next.place(s, w);
            let mut q = vec![s as u8];
            if propagate_token(hot, puz, &mut next, &mut q).is_ok() && no_repeats(&next.placed) {
                if let Some(done) = go(hot, puz, next, rng, budget) {
                    return Some(done);
                }
            }
        }
        None
    }
    let (mut st, mut q) = start(hot, puz, &[]).ok()?;
    propagate_token(hot, puz, &mut st, &mut q).ok()?;
    go(hot, puz, st, rng, budget).map(|st| st.placed)
}

/// Complete fills consistent with `givens`, up to `cap`; `None` when the
/// budget runs out. The uniqueness oracle for creation.
pub fn count_fills(
    hot: &Hot,
    puz: &Puzzle,
    givens: &[(u8, WordId)],
    cap: usize,
    budget: &mut usize,
) -> Option<usize> {
    pub fn go(hot: &Hot, puz: &Puzzle, st: State, cap: usize, budget: &mut usize) -> Option<usize> {
        let Some(s) = (0..puz.slots())
            .filter(|&s| st.placed[s] == UNSET)
            .min_by_key(|&s| st.count(s))
        else {
            return Some(usize::from(no_repeats(&st.placed)));
        };
        let mut n = 0;
        for w in bits(st.mask(s))
            .into_iter()
            .filter(|w| !st.placed.contains(w))
        {
            if *budget == 0 {
                return None;
            }
            *budget -= 1;
            let mut next = st.clone();
            next.place(s, w);
            let mut q = vec![s as u8];
            if propagate_token(hot, puz, &mut next, &mut q).is_ok() && no_repeats(&next.placed) {
                n += go(hot, puz, next, cap - n, budget)?;
                if n >= cap {
                    break;
                }
            }
        }
        Some(n)
    }
    let Ok((mut st, mut q)) = start(hot, puz, givens) else {
        return Some(0);
    };
    if propagate_token(hot, puz, &mut st, &mut q).is_err() {
        return Some(0);
    }
    go(hot, puz, st, cap, budget)
}

/// A created puzzle: its compiled layout, its solution and the givens that
/// make the solution unique.
pub struct Created {
    pub puz: Puzzle,
    pub solution: Vec<WordId>,
    pub givens: Vec<(u8, WordId)>,
    pub compile_ns: f64,
}

/// Grid, compile, random fill, then givens in random order until unique.
pub fn create(hot: &Hot, side: usize, rng: &mut Rng, budget: usize) -> Result<Created, Miss> {
    let grid = Grid::random_nyt(side, rng);
    let t = Instant::now();
    let puz = compile(&grid, hot.lang);
    let compile_ns = t.elapsed().as_nanos() as f64;
    let mut b = budget;
    let solution = random_fill(hot, &puz, rng, &mut b).ok_or(Miss::Fill)?;
    let mut order: Vec<u8> = (0..puz.slots() as u8).collect();
    rng.shuffle(&mut order);
    let mut givens = Vec::new();
    for s in order {
        givens.push((s, solution[s as usize]));
        let mut b = budget;
        if count_fills(hot, &puz, &givens, 2, &mut b).ok_or(Miss::Unique)? == 1 {
            return Ok(Created {
                puz,
                solution,
                givens,
                compile_ns,
            });
        }
    }
    unreachable!("every slot given is unique")
}
