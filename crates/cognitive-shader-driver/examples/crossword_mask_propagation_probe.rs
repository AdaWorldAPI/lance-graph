//! D-PUZZLE-0, step 3: is crossword propagation the same substrate operation
//! as Sudoku's, and which physical representation of it is best?
//!
//! The question is not whether a crossword can be solved. It is whether, once a
//! puzzle is compiled, propagation is `mask → intersect → popcount → promote →
//! expose → adjacent masks → repeat`, with nothing crossword-shaped on the hot
//! path, and how four representations of that one logic compare on the SAME
//! puzzles, givens and fixed points.
//!
//! # Coordinates, kept apart
//!
//! | what | encoding | source |
//! |---|---|---|
//! | WHERE a letter sits | cell = `Morton8x8::from_xy(col, row)` (u16) | `lance_graph_contract::morton8x8` |
//! | WHICH slot position | `(slot:offset)` = `FacetTier { hi: slot, lo: offset }.as_u16()` | `facet::FacetTier` |
//! | WHAT letter is in a cell | [`MooreSymbol8`] (u8) | declared here (see below) |
//! | WHAT word | `WordId` (u16), DeepNSM-v2 `PaletteVocab::from_frequency_ranked` | `deepnsm_v2::vocab` |
//! | what is BELIEVED | CE64 bits 59..63: GIVEN 20 / FORCED 21 / ENTAILED 22 / CANDIDATE 4 | `shared/population_fold.rs` |
//!
//! A crossing is equality of two cell codes. The contract stores no topology
//! (`morton8x8.rs`: "no neighbour list and no stored edge"), so the equality is
//! evaluated once, at compile time, into two lanes: `cross[slot:offset]` = the
//! other `(slot:offset)` on the same cell, and `occupant[cell]` = the (at most
//! two) positions on a cell. The runtime reads lanes; it never sees the grid.
//!
//! # `MooreSymbol8`: the letter reading of one byte
//!
//! No existing Moore byte may be read as a letter: every Moore in the repo is a
//! direction table, a grid-edge validity byte or a palette-law operand. So this
//! probe declares one reading of a Moore-local byte as a symbol codebook:
//! `0` unknown, `1..=26` a–z, `27` ä, `28` ö, `29` ü, `30` ß, `31..=255`
//! reserved. Crossword spelling ignores accents: an accented Latin letter folds
//! to its base (`cliché` is spelled `cliche`); ä ö ü ß are letters of their
//! own and do not fold. A word with any other character is not a crossword
//! word; it keeps its `WordId`. Whether this reading becomes a canonical value
//! tenant is a contract decision, not taken here.
//!
//! # Vocabularies
//!
//! - **English** (committed): `crates/deepnsm/word_frequency/academic_20k.csv`,
//!   column `word`, lowercased, through `PaletteVocab::from_frequency_ranked`,
//!   exactly as DeepNSM-v2 builds it (`examples/genre_shapes.rs`): 18,555 ids.
//!   (`vocab.rs` quotes 18,559 distinct surfaces; four collapse when
//!   lowercased.)
//! - **German** (runtime only): DeepNSM-v2 builds no German vocabulary and the
//!   repository holds none. With `DEREKO_PATH` set, the 20,000 most frequent
//!   DeReKo-2014 forms (lowercased, `NE` and non-alphabetic forms dropped,
//!   frequencies summed) go through the same `from_frequency_ranked`. CC BY-NC
//!   3.0; nothing derived is committed. Text is assumed NFC.
//!
//! # The four arms (identical puzzles, givens and fixed points)
//!
//! | arm | content | addressing | propagation |
//! |---|---|---|---|
//! | A token | `WordId` masks | `cross[slot:offset]` | a placed word's letter at offset `i` ANDs `P(L, j, letter)` into the crossing slot |
//! | B Cartesian | `MooreSymbol8` board | `occupant[cell]` | a placed word writes its symbols; each affected slot's mask is recomputed from its cells' symbols |
//! | C literal | strings, chars | cell array | every unplaced slot re-scans the vocabulary strings each round (steps 2/2b) |
//! | D hybrid | `WordId` masks + `MooreSymbol8` board | `occupant[cell]` | a placed word writes its symbols; each NEW symbol ANDs one `P` into the cell's other slot |
//!
//! `P(L, i, x)` is a bitset over a language's `WordId`s: words of length `L`
//! with symbol `x` at offset `i`. `popcount_batch_u64` decides: 0 =
//! contradiction, 1 = forced, more = still candidates. Backtracking (arm A's
//! masks) only CREATES puzzles and checks uniqueness; solving is the fixed point.
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
/// Longest slot (21×21 Sunday board) and the per-slot stride of position lanes.
const MAX_LEN: usize = 21;
const STRIDE: usize = 32;
/// No crossing / no occupant.
const NONE: u16 = u16::MAX;
/// No word placed in this slot.
const UNSET: WordId = WordId::MAX;
/// German forms kept.
const GERMAN_VOCAB: usize = 20_000;

const ACADEMIC: &str = include_str!("../../deepnsm/word_frequency/academic_20k.csv");

// ─────────────────────────────── symbols ───────────────────────────────

/// The declared letter reading of one Moore-local byte: `0` unknown, `1..=26`
/// a–z, `27` ä, `28` ö, `29` ü, `30` ß, `31..=255` reserved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
struct MooreSymbol8(u8);

impl MooreSymbol8 {
    const UNKNOWN: Self = Self(0);
    /// Codes in use, `0..=30`.
    const COUNT: usize = 31;

    /// The symbol of a lowercase character after accent folding.
    fn of(c: char) -> Option<Self> {
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
    fn letter(self) -> Option<char> {
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
fn fold(c: char) -> char {
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
fn spelling(w: &str) -> Option<Vec<MooreSymbol8>> {
    let s: Option<Vec<MooreSymbol8>> = w.chars().map(MooreSymbol8::of).collect();
    s.filter(|s| (MIN_WORD..=MAX_LEN).contains(&s.len()))
}

// ─────────────────────────────── vocabulary ───────────────────────────────

/// The language a lexicon and a puzzle belong to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lang {
    En,
    De,
}

/// Cold side: the strings. Construction, the literal arm and checks only.
struct Cold {
    vocab: PaletteVocab,
}

impl Cold {
    fn word(&self, w: WordId) -> &str {
        self.vocab.word(w).expect("word id in range")
    }
}

/// Hot side: integers and masks only. No `String`, no `char`.
struct Hot {
    lang: Lang,
    blocks: usize,
    /// Per `WordId`: crossword length, 0 = not a crossword word.
    len: Vec<u8>,
    /// `[id * STRIDE + offset]` = the symbol.
    spell: Vec<MooreSymbol8>,
    /// By length: every crossword word of that length.
    all: Vec<Vec<u64>>,
    /// By length: `[(offset * COUNT + symbol) * blocks ..]` = `P(L, offset, symbol)`.
    at: Vec<Vec<u64>>,
}

impl Hot {
    fn pop(&self, len: usize, offset: usize, x: MooreSymbol8) -> &[u64] {
        let i = (offset * MooreSymbol8::COUNT + x.0 as usize) * self.blocks;
        &self.at[len][i..i + self.blocks]
    }

    fn letter(&self, w: WordId, offset: usize) -> MooreSymbol8 {
        self.spell[w as usize * STRIDE + offset]
    }

    /// Bytes held by the populations (the fixed per-language cost).
    fn population_bytes(&self) -> usize {
        8 * (self.all.iter().map(Vec::len).sum::<usize>()
            + self.at.iter().map(Vec::len).sum::<usize>())
    }
}

/// Both sides from a frequency-ranked word list (most frequent first).
fn build_lexicon(lang: Lang, ranked: &[String]) -> (Hot, Cold) {
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
    rows: Vec<u32>,
}

/// Every white bit of a line lies in a run of at least three: the OR of the
/// three-wide windows covers the line.
fn line_runs_ok(w: u32) -> bool {
    let t = w & (w >> 1) & (w >> 2);
    (t | (t << 1) | (t << 2)) == w
}

impl Grid {
    #[cfg(test)]
    fn from_rows(rows: &[&str]) -> Self {
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

    fn full_line(side: usize) -> u32 {
        ((1u64 << side) - 1) as u32
    }

    fn cols(&self) -> Vec<u32> {
        (0..self.side)
            .map(|c| (0..self.side).fold(0u32, |m, r| m | (((self.rows[r] >> c) & 1) << r)))
            .collect()
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
        let rev = |w: u32| w.reverse_bits() >> (32 - self.side);
        (0..self.side).all(|r| rev(self.rows[r]) == self.rows[self.side - 1 - r])
    }

    /// Every across and down run is at least `MIN_WORD` long.
    fn runs_ok(&self) -> bool {
        self.rows
            .iter()
            .chain(self.cols().iter())
            .all(|&w| line_runs_ok(w))
    }

    /// One white region: grow from the first white cell by row-mask dilation.
    fn connected(&self) -> bool {
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
struct Puzzle {
    lang: Lang,
    /// Per slot: its length.
    len: Vec<u8>,
    /// `[slot * STRIDE + offset]`: the crossing `(slot:offset)` tile, or `NONE`.
    cross: Vec<u16>,
    /// `[slot * STRIDE + offset]`: the Morton cell code.
    cell: Vec<u16>,
    /// `[cell]`: the (at most two) `(slot:offset)` tiles on a cell.
    occupant: Vec<[u16; 2]>,
}

impl Puzzle {
    fn slots(&self) -> usize {
        self.len.len()
    }
}

fn tile(s: usize, o: usize) -> u16 {
    FacetTier {
        hi: s as u8,
        lo: o as u8,
    }
    .as_u16()
}

fn untile(t: u16) -> (usize, usize) {
    ((t >> 8) as usize, (t & 0xFF) as usize)
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

// ─────────────────────────────── arms ───────────────────────────────

/// Why a solve did not start or did not finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stop {
    /// The puzzle and the populations are different languages.
    Language,
    /// A slot was left with no candidate, or two letters disagree on a cell.
    Contradiction(u8),
}

/// What a solve leaves, and what it cost.
#[derive(Clone, Default)]
struct Solved {
    placed: Vec<WordId>,
    /// `[slot * blocks ..]`: final candidates (a placed slot holds one bit).
    cand: Vec<u64>,
    /// Mask intersections (`P` ANDs) performed.
    ands: usize,
    /// Symbols written to the board (arms B and D).
    writes: usize,
    /// Bytes of runtime state the arm holds.
    bytes: usize,
}

/// One mask arm: puzzle and givens in, fixed point out.
type Arm = fn(&Hot, &Puzzle, &[(u8, WordId)]) -> Result<Solved, Stop>;

/// Per-slot candidate masks plus the placed word, shared by arms A and D.
#[derive(Clone)]
struct State {
    blocks: usize,
    cand: Vec<u64>,
    placed: Vec<WordId>,
    ands: usize,
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
            ands: 0,
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

    /// Promote every unplaced slot already down to one candidate.
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

    /// AND one population into slot `t`; promote at popcount 1.
    fn narrow(&mut self, t: usize, p: &[u64], queue: &mut Vec<u8>) -> Result<(), Stop> {
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

/// Arm A's fixed point: each queued slot hands its letter at every crossed
/// offset to the crossing `(slot:offset)` read from `cross`.
fn propagate_token(
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

fn start(hot: &Hot, puz: &Puzzle, givens: &[(u8, WordId)]) -> Result<(State, Vec<u8>), Stop> {
    let mut st = State::new(hot, puz)?;
    let mut queue = Vec::new();
    for &(s, w) in givens {
        st.place(s as usize, w);
        queue.push(s);
    }
    st.seed(&mut queue)?;
    Ok((st, queue))
}

/// Arm A: token masks over the `cross` lane.
fn solve_token(hot: &Hot, puz: &Puzzle, givens: &[(u8, WordId)]) -> Result<Solved, Stop> {
    let (mut st, mut queue) = start(hot, puz, givens)?;
    propagate_token(hot, puz, &mut st, &mut queue)?;
    let bytes = st.cand.len() * 8 + st.placed.len() * 2;
    Ok(Solved {
        placed: st.placed,
        cand: st.cand,
        ands: st.ands,
        writes: 0,
        bytes,
    })
}

/// Arm D: token masks, with the letters routed through a `MooreSymbol8` board
/// and the `occupant` lane. Each NEW symbol on a cell ANDs one `P` into the
/// cell's other slot.
fn solve_hybrid(hot: &Hot, puz: &Puzzle, givens: &[(u8, WordId)]) -> Result<Solved, Stop> {
    let (mut st, mut queue) = start(hot, puz, givens)?;
    let mut board = vec![MooreSymbol8::UNKNOWN; puz.occupant.len()];
    let mut writes = 0;
    while let Some(s) = queue.pop() {
        let s = s as usize;
        let w = st.placed[s];
        for off in 0..puz.len[s] as usize {
            let c = puz.cell[s * STRIDE + off] as usize;
            let x = hot.letter(w, off);
            let cur = board[c];
            if cur != MooreSymbol8::UNKNOWN {
                if cur != x {
                    return Err(Stop::Contradiction(s as u8));
                }
                continue;
            }
            board[c] = x;
            writes += 1;
            for &occ in &puz.occupant[c] {
                if occ == NONE {
                    continue;
                }
                let (t, j) = untile(occ);
                if t == s || st.placed[t] != UNSET {
                    continue;
                }
                st.narrow(t, hot.pop(puz.len[t] as usize, j, x), &mut queue)?;
            }
        }
    }
    let bytes = st.cand.len() * 8 + st.placed.len() * 2 + board.len();
    Ok(Solved {
        placed: st.placed,
        cand: st.cand,
        ands: st.ands,
        writes,
        bytes,
    })
}

/// Arm B: the board is the state. A placed word writes its symbols; every
/// unplaced slot on a newly written cell recomputes its mask from ALL of its
/// cells' symbols. (Slot-indexed: `placed` and the lanes share the index.)
#[allow(clippy::needless_range_loop)]
fn solve_cartesian(hot: &Hot, puz: &Puzzle, givens: &[(u8, WordId)]) -> Result<Solved, Stop> {
    if hot.lang != puz.lang {
        return Err(Stop::Language);
    }
    let b = hot.blocks;
    let mut board = vec![MooreSymbol8::UNKNOWN; puz.occupant.len()];
    let mut placed = vec![UNSET; puz.slots()];
    let mut scratch = vec![0u64; b];
    let (mut ands, mut writes) = (0usize, 0usize);
    let recompute = |t: usize, board: &[MooreSymbol8], scratch: &mut [u64], ands: &mut usize| {
        let l = puz.len[t] as usize;
        scratch.copy_from_slice(&hot.all[l]);
        for j in 0..l {
            let x = board[puz.cell[t * STRIDE + j] as usize];
            if x != MooreSymbol8::UNKNOWN {
                mask_and_assign(scratch, hot.pop(l, j, x));
                *ands += 1;
            }
        }
        popcount_batch_u64(scratch)
    };
    let mut queue: Vec<u8> = Vec::new();
    for &(s, w) in givens {
        placed[s as usize] = w;
        queue.push(s);
    }
    for t in 0..puz.slots() {
        if placed[t] == UNSET {
            match recompute(t, &board, &mut scratch, &mut ands) {
                0 => return Err(Stop::Contradiction(t as u8)),
                1 => {
                    placed[t] = first_bit(&scratch);
                    queue.push(t as u8);
                }
                _ => {}
            }
        }
    }
    while let Some(s) = queue.pop() {
        let s = s as usize;
        let w = placed[s];
        for off in 0..puz.len[s] as usize {
            let c = puz.cell[s * STRIDE + off] as usize;
            let x = hot.letter(w, off);
            let cur = board[c];
            if cur != MooreSymbol8::UNKNOWN {
                if cur != x {
                    return Err(Stop::Contradiction(s as u8));
                }
                continue;
            }
            board[c] = x;
            writes += 1;
            for &occ in &puz.occupant[c] {
                if occ == NONE {
                    continue;
                }
                let (t, _) = untile(occ);
                if t == s || placed[t] != UNSET {
                    continue;
                }
                match recompute(t, &board, &mut scratch, &mut ands) {
                    0 => return Err(Stop::Contradiction(t as u8)),
                    1 => {
                        placed[t] = first_bit(&scratch);
                        queue.push(t as u8);
                    }
                    _ => {}
                }
            }
        }
    }
    let bytes = board.len() + placed.len() * 2 + scratch.len() * 8;
    let mut cand = vec![0u64; puz.slots() * b];
    for t in 0..puz.slots() {
        if placed[t] == UNSET {
            recompute(t, &board, &mut scratch, &mut 0);
            cand[t * b..(t + 1) * b].copy_from_slice(&scratch);
        } else {
            cand[t * b + placed[t] as usize / 64] = 1 << (placed[t] % 64);
        }
    }
    Ok(Solved {
        placed,
        cand,
        ands,
        writes,
        bytes,
    })
}

/// Arm C, steps 2/2b's method: letters on a cell array, every unplaced slot
/// re-scanned from the vocabulary STRINGS each round, singles placed until
/// nothing changes. `None` on contradiction. (Slot-indexed on purpose: it
/// mirrors the loop it stands in for.)
#[allow(clippy::needless_range_loop)]
fn solve_literal(cold: &Cold, puz: &Puzzle, givens: &[(u8, WordId)]) -> Option<Vec<WordId>> {
    let words: Vec<(WordId, Vec<char>)> = (0..cold.vocab.len() as WordId)
        .filter_map(|id| {
            let w = cold.word(id);
            spelling(w)?;
            Some((id, w.chars().map(fold).collect()))
        })
        .collect();
    let mut board: HashMap<u16, char> = HashMap::new();
    let mut placed = vec![UNSET; puz.slots()];
    let put = |s: usize, w: &[char], board: &mut HashMap<u16, char>| -> bool {
        (0..w.len()).all(|o| *board.entry(puz.cell[s * STRIDE + o]).or_insert(w[o]) == w[o])
    };
    for &(s, w) in givens {
        let chars: Vec<char> = cold.word(w).chars().map(fold).collect();
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

// ─────────────────────────────── creation ───────────────────────────────

/// Why creation gave up on one attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Miss {
    /// No complete fill found within the budget.
    Fill,
    /// A fill was found, but uniqueness could not be decided within the budget.
    Unique,
}

/// Depth-first fill over arm A's masks, most-constrained slot first, random
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
            if propagate_token(hot, puz, &mut next, &mut q).is_ok() {
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
            if propagate_token(hot, puz, &mut next, &mut q).is_ok() {
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
struct Created {
    puz: Puzzle,
    solution: Vec<WordId>,
    givens: Vec<(u8, WordId)>,
    compile_ns: f64,
}

/// Grid, compile, random fill, then givens in random order until unique.
fn create(hot: &Hot, side: usize, rng: &mut Rng, budget: usize) -> Result<Created, Miss> {
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

// ─────────────────────────────── ablation ───────────────────────────────

/// What removing one given does: the fixed point re-run without it, and
/// whether the remaining givens still pin one fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Ablation {
    slot: u8,
    /// Slots placed (given or forced) with every given that are no longer
    /// placed without this one, the ablated slot itself not counted.
    lost_placed: usize,
    /// Change in surviving candidates summed over all slots (a placed slot
    /// counts one).
    popcount_delta: i64,
    /// Fills consistent with the remaining givens, capped at 2; `None` when
    /// the uniqueness budget runs out.
    fills: Option<usize>,
}

/// The four outcomes of one ablation. The readout nominates; it does not
/// prove cause: a given can be necessary in this set and replaceable by
/// another set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reaction {
    /// The fixed point is unchanged and the fill stays unique.
    Redundant,
    /// The fixed point weakens, but search still finds only one fill.
    ShortcutOnly,
    /// The fixed point is unchanged, yet the fill is no longer unique:
    /// propagation never used this given, search did.
    SilentlyNecessary,
    /// The fixed point weakens and the fill is no longer unique.
    Necessary,
}

impl Ablation {
    fn reaction(self) -> Option<Reaction> {
        let moved = self.lost_placed > 0 || self.popcount_delta != 0;
        let unique = self.fills? == 1;
        Some(match (moved, unique) {
            (false, true) => Reaction::Redundant,
            (true, true) => Reaction::ShortcutOnly,
            (false, false) => Reaction::SilentlyNecessary,
            (true, false) => Reaction::Necessary,
        })
    }
}

fn total_popcount(sol: &Solved, blocks: usize) -> i64 {
    sol.cand
        .chunks_exact(blocks)
        .map(|m| popcount_batch_u64(m) as i64)
        .sum()
}

/// Remove each given in turn, re-run arm A's fixed point and the uniqueness
/// count, and report the difference against the full set of givens.
fn ablate(hot: &Hot, puz: &Puzzle, givens: &[(u8, WordId)], budget: usize) -> Vec<Ablation> {
    let base = solve_token(hot, puz, givens).expect("the full givens are consistent");
    let base_pop = total_popcount(&base, hot.blocks);
    givens
        .iter()
        .enumerate()
        .map(|(i, &(slot, _))| {
            let rest: Vec<(u8, WordId)> = givens
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != i)
                .map(|(_, &g)| g)
                .collect();
            let sol = solve_token(hot, puz, &rest).expect("a subset of consistent givens");
            let lost_placed = (0..puz.slots())
                .filter(|&s| s != slot as usize)
                .filter(|&s| base.placed[s] != UNSET && sol.placed[s] == UNSET)
                .count();
            let mut b = budget;
            Ablation {
                slot,
                lost_placed,
                popcount_delta: total_popcount(&sol, hot.blocks) - base_pop,
                fills: count_fills(hot, puz, &rest, 2, &mut b),
            }
        })
        .collect()
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
fn emit(c: &Created, sol: &Solved, blocks: usize, lane: &mut Lane, claims: &mut Claims) {
    for s in 0..c.puz.slots() {
        let given = c.givens.iter().any(|g| g.0 as usize == s);
        let truth = c.solution[s];
        let mut push = |state, w: WordId, lane: &mut Lane| {
            lane.push(state);
            claims.slot.push(s as u8);
            claims.word.push(w);
        };
        if given {
            assert_eq!(sol.placed[s], truth);
            push(GIVEN, truth, lane);
            lane.expected[0] += 1;
        } else if sol.placed[s] != UNSET {
            assert_eq!(sol.placed[s], truth, "a forced word is the true word");
            push(FORCED, truth, lane);
            lane.expected[1] += 1;
        } else {
            let alive = bits(&sol.cand[s * blocks..(s + 1) * blocks]);
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

/// Every cell shared by two placed words carries the same symbol, checked from
/// the Morton cell lanes alone (never from `cross` or `occupant`).
fn cell_consistent(hot: &Hot, puz: &Puzzle, placed: &[WordId]) -> bool {
    let mut at: HashMap<u16, MooreSymbol8> = HashMap::new();
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

// ─────────────────────────────── benchmark ───────────────────────────────

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn time_us<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let r = black_box(f());
    (r, t.elapsed().as_secs_f64() * 1e6)
}

/// Filter micro-benchmark: one slot pattern (length + revealed symbols) as a
/// mask AND chain vs a direct scan of the spell lane.
fn filter_bench(hot: &Hot) {
    let mut rng = Rng(0xF17);
    let words: Vec<WordId> = (0..hot.len.len() as WordId)
        .filter(|&w| hot.len[w as usize] as usize >= 5)
        .collect();
    let patterns: Vec<(usize, Vec<(usize, MooreSymbol8)>)> = (0..2000)
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
        "  candidate filter, {} patterns, {:.1} surviving claims each:",
        patterns.len(),
        claims as f64 / p
    );
    println!(
        "    mask AND chain + popcount  {:>8.0} ns/pattern {:>8.2} ns/claim",
        mask_ns / p,
        mask_ns / claims as f64
    );
    println!(
        "    direct spell-lane scan     {:>8.0} ns/pattern {:>8.2} ns/claim",
        scan_ns / p,
        scan_ns / claims as f64
    );
}

/// Timed rounds per puzzle and arm (after one warm-up call each).
const ROUNDS: usize = 5;

/// Size sweep: what can be created, and every arm's solve on the same puzzles.
fn sweep(hot: &Hot, cold: &Cold) -> Vec<(usize, Vec<Created>)> {
    println!("  creation by board size (budget 100,000 tried words per step, 30 s per size):");
    println!(
        "    {:>4} {:>8} {:>9} {:>11} {:>6} {:>6} {:>10}",
        "side", "made", "miss f/u", "create ms", "slots", "givens", "compile us"
    );
    let mut rng = Rng(0x5EED);
    let mut per_size: Vec<(usize, Vec<Created>)> = Vec::new();
    for side in [5, 7, 9, 11, 13, 15, 17, 19, 21] {
        let t0 = Instant::now();
        let (mut made, mut miss_fill, mut miss_unique, mut ms) = (Vec::new(), 0, 0, Vec::new());
        while t0.elapsed() < Duration::from_secs(30) && made.len() < 30 {
            match time_us(|| create(hot, side, &mut rng, 100_000)) {
                (Ok(c), us) => {
                    ms.push(us / 1e3);
                    made.push(c);
                }
                (Err(Miss::Fill), _) => miss_fill += 1,
                (Err(Miss::Unique), _) => miss_unique += 1,
            }
        }
        if made.is_empty() {
            println!("    {side:>4} {:>8} {:>4}/{:<4}", 0, miss_fill, miss_unique);
            continue;
        }
        let n = made.len() as f64;
        println!(
            "    {side:>4} {:>8} {:>4}/{:<4} {:>11.1} {:>6.1} {:>6.1} {:>10.1}",
            made.len(),
            miss_fill,
            miss_unique,
            median(ms),
            made.iter().map(|c| c.puz.slots()).sum::<usize>() as f64 / n,
            made.iter().map(|c| c.givens.len()).sum::<usize>() as f64 / n,
            median(made.iter().map(|c| c.compile_ns / 1e3).collect())
        );
        per_size.push((side, made));
    }
    println!(
        "  solving the same puzzles, four arms (median over puzzles of each arm's best of {ROUNDS} warm runs, us; ANDs, symbol writes, bytes per puzzle):"
    );
    println!(
        "    {:>4} {:>7} {:>9} {:>9} {:>9} {:>9} {:>7} {:>7} {:>7} {:>8} {:>8}",
        "side",
        "solved",
        "A token",
        "D hybrid",
        "B cart",
        "C literal",
        "ANDs A",
        "ANDs B",
        "writes",
        "bytes A",
        "bytes B"
    );
    for (side, made) in &per_size {
        let (mut ta, mut td, mut tb, mut tc) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut and_a, mut and_b, mut writes, mut bytes_a, mut bytes_b, mut solved) =
            (0, 0, 0, 0, 0, 0);
        for c in made {
            // One warm-up call per arm, then ROUNDS rounds in rotated order;
            // each arm keeps its best time. A single first-call timing would
            // charge whichever arm runs first for the cold populations.
            let a = solve_token(hot, &c.puz, &c.givens).unwrap();
            let d = solve_hybrid(hot, &c.puz, &c.givens).unwrap();
            let b = solve_cartesian(hot, &c.puz, &c.givens).unwrap();
            let arms: [Arm; 3] = [solve_token, solve_hybrid, solve_cartesian];
            let mut best = [f64::INFINITY; 3];
            for round in 0..ROUNDS {
                for k in 0..3 {
                    let i = (k + round) % 3;
                    let (_, us) = time_us(|| arms[i](hot, &c.puz, &c.givens).unwrap());
                    best[i] = best[i].min(us);
                }
            }
            ta.push(best[0]);
            td.push(best[1]);
            tb.push(best[2]);
            assert_eq!(a.placed, d.placed);
            assert_eq!(a.placed, b.placed);
            assert_eq!(a.cand, d.cand);
            assert_eq!(a.cand, b.cand);
            assert!(cell_consistent(hot, &c.puz, &a.placed));
            if *side <= 9 {
                let (lit, us) = time_us(|| solve_literal(cold, &c.puz, &c.givens).unwrap());
                tc.push(us);
                assert_eq!(lit, a.placed, "literal fixed point == mask fixed point");
            }
            and_a += a.ands;
            and_b += b.ands;
            writes += d.writes;
            bytes_a += a.bytes;
            bytes_b += b.bytes;
            if a.placed.iter().all(|&w| w != UNSET) {
                solved += 1;
            }
        }
        let n = made.len();
        let lit = if tc.is_empty() {
            "-".to_string()
        } else {
            format!("{:.0}", median(tc))
        };
        println!(
            "    {side:>4} {:>6.0}% {:>9.1} {:>9.1} {:>9.1} {:>9} {:>7.1} {:>7.1} {:>7.1} {:>8} {:>8}",
            100.0 * solved as f64 / n as f64,
            median(ta),
            median(td),
            median(tb),
            lit,
            and_a as f64 / n as f64,
            and_b as f64 / n as f64,
            writes as f64 / n as f64,
            bytes_a / n,
            bytes_b / n
        );
    }
    per_size
}

/// Ablate every given of every created puzzle and tally the reactions.
fn ablation_report(hot: &Hot, per_size: &[(usize, Vec<Created>)]) {
    println!("  cui bono: remove each given, re-run the fixed point and the uniqueness count:");
    println!(
        "    {:>4} {:>7} {:>10} {:>9} {:>9} {:>9} {:>8} {:>13} {:>10}",
        "side",
        "givens",
        "redundant",
        "shortcut",
        "silent",
        "necessary",
        "budget",
        "lost placed",
        "pop delta"
    );
    for (side, made) in per_size {
        let mut tally = [0usize; 4];
        let (mut out_of_budget, mut n, mut lost, mut pop) = (0usize, 0usize, 0usize, 0i64);
        for c in made {
            for a in ablate(hot, &c.puz, &c.givens, 100_000) {
                n += 1;
                lost += a.lost_placed;
                pop += a.popcount_delta;
                match a.reaction() {
                    Some(Reaction::Redundant) => tally[0] += 1,
                    Some(Reaction::ShortcutOnly) => tally[1] += 1,
                    Some(Reaction::SilentlyNecessary) => tally[2] += 1,
                    Some(Reaction::Necessary) => tally[3] += 1,
                    None => out_of_budget += 1,
                }
            }
        }
        if n == 0 {
            continue;
        }
        println!(
            "    {side:>4} {n:>7} {:>10} {:>9} {:>9} {:>9} {out_of_budget:>8} {:>13.2} {:>10.1}",
            tally[0],
            tally[1],
            tally[2],
            tally[3],
            lost as f64 / n as f64,
            pop as f64 / n as f64
        );
    }
}

/// About a million claims at one size, folded the shared three ways.
fn lane_run(hot: &Hot, side: usize) {
    let decl = declarations(CROSSWORD_CLASS);
    admit(&decl, CROSSWORD_CLASS).expect("the crossword class declares the canonical reading");
    let mut rng = Rng(0x1A4E);
    let (mut lane, mut claims, mut slots) = (Lane::default(), Claims::default(), Vec::new());
    let t = Instant::now();
    while lane.edges.len() < 1_000_000 {
        let Ok(c) = create(hot, side, &mut rng, 100_000) else {
            continue;
        };
        let sol = solve_token(hot, &c.puz, &c.givens).expect("consistent");
        slots.push(c.puz.slots());
        emit(&c, &sol, hot.blocks, &mut lane, &mut claims);
    }
    println!(
        "  lane: {} claims from {} {side}x{side} puzzles, built in {:.2?}",
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
    let used: String = (1..MooreSymbol8::COUNT as u8)
        .map(MooreSymbol8)
        .filter(|&x| (MIN_WORD..=MAX_LEN).any(|l| hot.pop(l, 0, x).iter().any(|&w| w != 0)))
        .filter_map(MooreSymbol8::letter)
        .collect();
    println!(
        "\n=== {name}: {} WordIds, {} crossword words, first letters in use [{used}], populations {:.1} MB",
        cold.vocab.len(),
        words,
        hot.population_bytes() as f64 / 1e6
    );
    filter_bench(hot);
    let per_size = sweep(hot, cold);
    ablation_report(hot, &per_size);
    lane_run(hot, 5);
}

fn main() {
    println!("D-PUZZLE-0 step 3: crossword propagation, four representations of one fold");
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

    const ARMS: [(&str, Arm); 3] = [
        ("token", solve_token),
        ("hybrid", solve_hybrid),
        ("cartesian", solve_cartesian),
    ];

    fn created(hot: &Hot, side: usize, seed: u64) -> Created {
        let mut rng = Rng(seed);
        loop {
            if let Ok(c) = create(hot, side, &mut rng, 100_000) {
                return c;
            }
        }
    }

    /// The codebook is the declared one: a–z 1..=26, ä ö ü ß 27..=30, 0 for
    /// unknown; accents fold; anything else has no symbol.
    #[test]
    fn moore_symbol8_is_the_declared_codebook() {
        assert_eq!(MooreSymbol8::of('a'), Some(MooreSymbol8(1)));
        assert_eq!(MooreSymbol8::of('z'), Some(MooreSymbol8(26)));
        for (c, n) in [('ä', 27), ('ö', 28), ('ü', 29), ('ß', 30)] {
            assert_eq!(MooreSymbol8::of(c), Some(MooreSymbol8(n)));
        }
        assert_eq!(MooreSymbol8::of('é'), MooreSymbol8::of('e'));
        assert_eq!(MooreSymbol8::of('ç'), MooreSymbol8::of('c'));
        for c in ['-', '\'', ' ', 'ø', 'A'] {
            assert_eq!(MooreSymbol8::of(c), None, "{c}");
        }
        for n in 1..=30u8 {
            let x = MooreSymbol8(n);
            assert_eq!(MooreSymbol8::of(x.letter().unwrap()), Some(x));
        }
        assert_eq!(MooreSymbol8::UNKNOWN.letter(), None);
        assert_eq!(MooreSymbol8(31).letter(), None);
    }

    /// The English vocabulary is DeepNSM-v2's, and the crossword projection
    /// only spells words: `cliché` keeps its id and is spelled `cliche`;
    /// hyphenated rows keep their id and have no spelling.
    #[test]
    fn english_is_the_deepnsm_v2_academic_vocabulary() {
        let (hot, cold) = english();
        assert_eq!(cold.vocab.len(), 18_555);
        assert_eq!(cold.word(0), "the");
        assert_eq!(hot.len[word(&cold, "so-called") as usize], 0);
        let cliche = word(&cold, "cliché");
        let spelled: String = (0..6)
            .map(|o| hot.letter(cliche, o).letter().unwrap())
            .collect();
        assert_eq!(spelled, "cliche");
        // English has no umlaut words, so their populations are empty.
        assert!(hot.pop(5, 0, MooreSymbol8(27)).iter().all(|&w| w == 0));
    }

    /// Every crossword word decodes from its symbols to its folded string.
    #[test]
    fn every_word_decodes_from_its_symbols() {
        let (hot, cold) = english();
        let mut n = 0;
        for id in 0..cold.vocab.len() as WordId {
            let l = hot.len[id as usize] as usize;
            if l == 0 {
                continue;
            }
            let back: String = (0..l)
                .map(|o| hot.letter(id, o).letter().unwrap())
                .collect();
            assert_eq!(back, cold.word(id).chars().map(fold).collect::<String>());
            n += 1;
        }
        assert!(n > 15_000);
    }

    /// Positional populations equal a direct scan over the vocabulary strings.
    #[test]
    fn positional_populations_equal_a_string_scan() {
        let (hot, cold) = english();
        let mut rng = Rng(2);
        for _ in 0..500 {
            let l = MIN_WORD + rng.below(10) as usize;
            let k = rng.below(4) as usize;
            let fixed: Vec<(usize, char)> = (0..k)
                .map(|_| {
                    let o = rng.below(l as u64) as usize;
                    (o, (b'a' + rng.below(26) as u8) as char)
                })
                .collect();
            let mut m = hot.all[l].clone();
            for &(o, c) in &fixed {
                mask_and_assign(&mut m, hot.pop(l, o, MooreSymbol8::of(c).unwrap()));
            }
            let scan: Vec<WordId> = (0..cold.vocab.len() as WordId)
                .filter(|&id| {
                    let w: Vec<char> = cold.word(id).chars().map(fold).collect();
                    spelling(cold.word(id)).is_some()
                        && w.len() == l
                        && fixed.iter().all(|&(o, c)| w[o] == c)
                })
                .collect();
            assert_eq!(bits(&m), scan, "pattern {l} {fixed:?}");
        }
    }

    /// A slot narrowed to one candidate is promoted, in every arm.
    #[test]
    fn a_single_candidate_is_forced() {
        let (hot, cold) = build_lexicon(Lang::En, &words(&["cat", "cow", "dog", "ant"]));
        let puz = compile(&plus(), Lang::En);
        assert_eq!(puz.len, [3, 3]);
        for (name, arm) in ARMS {
            let sol = arm(&hot, &puz, &[(0, word(&cold, "cat"))]).unwrap();
            assert_eq!(sol.placed[1], word(&cold, "ant"), "{name}");
        }
    }

    /// A placed word constrains EVERY crossing slot, in every arm.
    #[test]
    fn a_placed_word_masks_every_crossing() {
        let (hot, cold) = build_lexicon(
            Lang::En,
            &words(&["cat", "cow", "cub", "toe", "tea", "dog", "ant"]),
        );
        let puz = compile(&Grid::from_rows(&["...", ".#.", ".#."]), Lang::En);
        assert_eq!(puz.len, [3, 3, 3]);
        for (name, arm) in ARMS {
            let sol = arm(&hot, &puz, &[(0, word(&cold, "cat"))]).unwrap();
            let first = |s: usize| -> Vec<char> {
                bits(&sol.cand[s * hot.blocks..(s + 1) * hot.blocks])
                    .into_iter()
                    .map(|w| cold.word(w).chars().next().unwrap())
                    .collect()
            };
            assert_eq!(first(1), ['c', 'c', 'c'], "{name}");
            assert_eq!(first(2), ['t', 't'], "{name}");
        }
    }

    /// No candidate left is a contradiction, in every arm.
    #[test]
    fn an_empty_slot_is_a_contradiction() {
        let (hot, cold) = build_lexicon(Lang::En, &words(&["cat", "cow", "dog"]));
        let puz = compile(&plus(), Lang::En);
        for (name, arm) in ARMS {
            assert_eq!(
                arm(&hot, &puz, &[(0, word(&cold, "cat"))]).err(),
                Some(Stop::Contradiction(1)),
                "{name}"
            );
        }
    }

    /// Crossings come from the compiled lanes. Re-pointing one crossing in
    /// `cross` lets an incompatible word survive in arm A, and the cell oracle
    /// sees it. Emptying `cross` (A) or `occupant` (B, D) stops propagation.
    #[test]
    fn crossings_are_read_from_the_compiled_lanes() {
        let (hot, cold) = build_lexicon(Lang::En, &words(&["cat", "cow", "dog", "ant"]));
        let puz = compile(&plus(), Lang::En);
        let given = [(0u8, word(&cold, "cat"))];
        for (_, arm) in ARMS {
            assert!(cell_consistent(
                &hot,
                &puz,
                &arm(&hot, &puz, &given).unwrap().placed
            ));
        }

        let mut wrong = puz.clone();
        assert_eq!(wrong.cross[1], tile(1, 0));
        wrong.cross[1] = tile(1, 1);
        wrong.cross[STRIDE] = NONE;
        wrong.cross[STRIDE + 1] = tile(0, 1);
        let bad = solve_token(&hot, &wrong, &given).unwrap();
        assert_eq!(bad.placed[1], word(&cold, "cat"));
        assert!(!cell_consistent(&hot, &wrong, &bad.placed));

        let mut cut = puz.clone();
        cut.cross.fill(NONE);
        let a = solve_token(&hot, &cut, &given).unwrap();
        assert_eq!((a.placed[1], a.ands), (UNSET, 0));

        let mut cut = puz.clone();
        cut.occupant.iter_mut().for_each(|o| *o = [NONE; 2]);
        for arm in [solve_hybrid as Arm, solve_cartesian] {
            let s = arm(&hot, &cut, &given).unwrap();
            assert_eq!(s.placed[1], UNSET);
        }
    }

    /// Crossing tiles are symmetric, land on one Morton cell, and match the
    /// occupant lane, up to 21x21.
    #[test]
    fn every_crossing_names_the_same_cell_both_ways() {
        let mut rng = Rng(4);
        for side in [5, 7, 15, 21] {
            let puz = compile(&Grid::random_nyt(side, &mut rng), Lang::En);
            for s in 0..puz.slots() {
                for o in 0..puz.len[s] as usize {
                    let c = puz.cross[s * STRIDE + o];
                    assert_ne!(c, NONE, "an NYT grid checks every square");
                    let (t, j) = untile(c);
                    let cell = puz.cell[s * STRIDE + o];
                    assert_eq!(puz.cell[t * STRIDE + j], cell);
                    assert_eq!(puz.cross[t * STRIDE + j], tile(s, o));
                    let occ = puz.occupant[cell as usize];
                    assert!(occ.contains(&tile(s, o)) && occ.contains(&c));
                }
            }
        }
    }

    /// Languages are separate populations: WordId 0 names different words,
    /// only German has umlaut populations, and every arm refuses a puzzle
    /// against the other language's populations.
    #[test]
    fn languages_do_not_share_ordinals_or_populations() {
        let (en, en_cold) = build_lexicon(Lang::En, &words(&["the", "and", "house", "street"]));
        let (de, de_cold) = build_lexicon(
            Lang::De,
            &words(&["der", "und", "haus", "straße", "über", "größe", "mädchen"]),
        );
        assert_ne!(en_cold.word(0), de_cold.word(0));
        let umlaut_u = MooreSymbol8::of('ü').unwrap();
        assert!(en.pop(4, 0, umlaut_u).iter().all(|&w| w == 0));
        assert!(de.pop(4, 0, umlaut_u).iter().any(|&w| w != 0));
        let de_puz = compile(&plus(), Lang::De);
        for (name, arm) in ARMS {
            assert_eq!(arm(&en, &de_puz, &[]).err(), Some(Stop::Language), "{name}");
            assert!(arm(&de, &de_puz, &[]).is_ok(), "{name}");
        }
    }

    /// Bits 59..63 carry state only: every claim edge is zero outside them,
    /// its code is one of the four shared states, two different words in one
    /// state have the identical edge, and reserved codes never appear.
    #[test]
    fn ce64_carries_state_never_content() {
        let (hot, _) = english();
        let (mut lane, mut claims) = (Lane::default(), Claims::default());
        for seed in 0..4 {
            let c = created(&hot, 5, 70 + seed);
            let sol = solve_token(&hot, &c.puz, &c.givens).unwrap();
            emit(&c, &sol, hot.blocks, &mut lane, &mut claims);
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
        assert_eq!(count_in(&lane.edges, !0u32 << 24), 0);
        assert_eq!(count_in(&lane.edges, UNKNOWN_CAUSES.bit()), 0);
    }

    /// The four arms reach one fixed point on created puzzles: same placed
    /// words (A, B, C, D) and same surviving masks (A, B, D); every created
    /// puzzle is unique and consistent.
    #[test]
    fn the_four_arms_agree_on_created_puzzles() {
        let (hot, cold) = english();
        for seed in 0..6 {
            let c = created(&hot, 5, 90 + seed);
            let a = solve_token(&hot, &c.puz, &c.givens).unwrap();
            let b = solve_cartesian(&hot, &c.puz, &c.givens).unwrap();
            let d = solve_hybrid(&hot, &c.puz, &c.givens).unwrap();
            assert_eq!(a.placed, b.placed);
            assert_eq!(a.placed, d.placed);
            assert_eq!(a.cand, b.cand);
            assert_eq!(a.cand, d.cand);
            assert_eq!(solve_literal(&cold, &c.puz, &c.givens).unwrap(), a.placed);
            assert!(cell_consistent(&hot, &c.puz, &c.solution));
            let mut budget = 1_000_000;
            assert_eq!(
                count_fills(&hot, &c.puz, &c.givens, 3, &mut budget),
                Some(1)
            );
        }
    }

    /// The row-mask grid rules agree with a cell-by-cell check, both ways.
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
            let side = [5, 7, 9, 15, 21][rng.below(5) as usize];
            let mut g = Grid {
                side,
                rows: vec![Grid::full_line(side); side],
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

    /// Solving runs without the strings: `Cold` is dropped first, and every
    /// mask arm still reaches the literal arm's fixed point.
    #[test]
    fn the_hot_path_needs_no_strings() {
        let (hot, cold) = english();
        let c = created(&hot, 5, 13);
        let expect = solve_literal(&cold, &c.puz, &c.givens).unwrap();
        drop(cold);
        for (name, arm) in ARMS {
            assert_eq!(
                arm(&hot, &c.puz, &c.givens).unwrap().placed,
                expect,
                "{name}"
            );
        }
    }

    /// Cui bono, on fixtures with known answers: a given the fixed point
    /// re-derives is redundant; a given whose loss weakens the fixed point
    /// but leaves one fill is a shortcut; a given whose loss admits a second
    /// fill is necessary.
    #[test]
    fn ablation_separates_redundant_shortcut_and_necessary_givens() {
        // across cat crossing down ant at across offset 1 / down offset 0
        let (hot, cold) = build_lexicon(Lang::En, &words(&["cat", "ant", "dog"]));
        let puz = compile(&plus(), Lang::En);
        let (cat, ant) = (word(&cold, "cat"), word(&cold, "ant"));
        let both = [(0u8, cat), (1u8, ant)];
        let a = ablate(&hot, &puz, &both, 1_000);
        // dropping ant: cat still forces it back
        assert_eq!(a[1].reaction(), Some(Reaction::Redundant));
        assert_eq!((a[1].lost_placed, a[1].popcount_delta), (0, 0));
        // dropping cat from {cat}: nothing propagates, search still unique
        let only = ablate(&hot, &puz, &[(0, cat)], 1_000);
        assert_eq!(only[0].reaction(), Some(Reaction::ShortcutOnly));
        assert_eq!(only[0].lost_placed, 1);
        assert!(only[0].popcount_delta > 0);

        // with art as a second a-word, ant is no longer forced by cat
        let (hot, cold) = build_lexicon(Lang::En, &words(&["cat", "ant", "art", "dog"]));
        let (cat, ant) = (word(&cold, "cat"), word(&cold, "ant"));
        let a = ablate(&hot, &puz, &[(0, cat), (1, ant)], 1_000);
        assert_eq!(a[1].fills, Some(2));
        assert_eq!(a[1].reaction(), Some(Reaction::Necessary));
        assert_eq!(a[1].popcount_delta, 1);
    }

    /// On created puzzles the last given added is necessary by construction
    /// (the generator stops at the first unique set), so ablation must find
    /// at least one necessary given per puzzle.
    #[test]
    fn every_created_puzzle_has_a_necessary_given() {
        let (hot, _) = english();
        for seed in 0..4 {
            let c = created(&hot, 5, 120 + seed);
            let a = ablate(&hot, &c.puz, &c.givens, 1_000_000);
            assert_eq!(a.len(), c.givens.len());
            let last = a.last().unwrap();
            assert!(matches!(
                last.reaction(),
                Some(Reaction::Necessary | Reaction::SilentlyNecessary)
            ));
        }
    }

    /// Three ways, partition, declaration gate: the shared fold, unchanged.
    #[test]
    fn the_shared_fold_counts_the_crossword_lane_three_ways() {
        let (hot, _) = english();
        let mut rng = Rng(17);
        let (mut lane, mut claims) = (Lane::default(), Claims::default());
        while lane.edges.len() < 20_000 {
            if let Ok(c) = create(&hot, 5, &mut rng, 100_000) {
                let sol = solve_token(&hot, &c.puz, &c.givens).unwrap();
                emit(&c, &sol, hot.blocks, &mut lane, &mut claims);
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
