//! Codebook construction and lookup — transcoded from `cb.c` (algorithmic
//! shape) and adapted to Rust.
//!
//! The codebook role: map phrase-tuples (sequences of word-IDs at one of
//! the Fibonacci-aligned levels) to integer codebook indices. The upstream
//! C reference exposes 11 codebook tiers (unigram .. 144-gram); this
//! transcode covers the same 11 tiers via the [`Codebook`] trait.
//!
//! Two implementations live behind that trait:
//!
//! - [`FlatCodebook`] — direct port of `cb.c`'s storage shape: flat
//!   `Vec`s for each tier + `HashMap` for lookup. Simpler, smaller code,
//!   not COW-friendly.
//!
//! - [`CowRadixCodebook`] — the variant. Stores phrase-tuples as paths in
//!   an Adaptive Radix Tree (ART, Leis 2013) with Copy-on-Write
//!   semantics: every mutation produces a new root, old roots remain
//!   valid for prior consumers. Fits the workspace's append-only doctrine.
//!
//! Both implementations satisfy the same trait and pass the same
//! round-trip tests.

use std::collections::HashMap;
use std::sync::Arc;

use crate::constants::N_LEVELS;

/// Phrase lengths per n-gram codebook level: `[3, 5, 8, 13, 21, 34, 55, 89, 144]`.
pub const NG_LENS: [usize; N_LEVELS] = [3, 5, 8, 13, 21, 34, 55, 89, 144];

/// Tier sizing budget for the 11 codebooks.
///
/// Direct port of `qtc_cb_sizes_t`. `auto_codebook_sizes` produces sensible
/// defaults by corpus size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodebookSizes {
    /// Unigram codebook size.
    pub uni: u32,
    /// Bigram codebook size.
    pub bi: u32,
    /// Trigram codebook size.
    pub tri: u32,
    /// 5-gram codebook size.
    pub fg: u32,
    /// 8-gram codebook size.
    pub eg: u32,
    /// 13-gram codebook size.
    pub tg: u32,
    /// 21-gram codebook size.
    pub vg: u32,
    /// 34-gram codebook size.
    pub tfg: u32,
    /// 55-gram codebook size.
    pub ffg: u32,
    /// 89-gram codebook size.
    pub efg: u32,
    /// 144-gram codebook size.
    pub ofg: u32,
}

impl CodebookSizes {
    /// Recommended sizes by corpus word count, matching `auto_codebook_sizes`
    /// in `cb.c`.
    #[must_use]
    pub const fn auto(nw: u32) -> Self {
        match nw {
            0..=4_999 => Self {
                uni: 509,
                bi: 509,
                tri: 350,
                fg: 100,
                eg: 50,
                tg: 0,
                vg: 0,
                tfg: 0,
                ffg: 0,
                efg: 0,
                ofg: 0,
            },
            5_000..=49_999 => Self {
                uni: 1_000,
                bi: 509,
                tri: 500,
                fg: 200,
                eg: 100,
                tg: 50,
                vg: 0,
                tfg: 0,
                ffg: 0,
                efg: 0,
                ofg: 0,
            },
            50_000..=199_999 => Self {
                uni: 4_000,
                bi: 2_000,
                tri: 1_000,
                fg: 500,
                eg: 200,
                tg: 100,
                vg: 50,
                tfg: 0,
                ffg: 0,
                efg: 0,
                ofg: 0,
            },
            200_000..=499_999 => Self {
                uni: 8_000,
                bi: 4_000,
                tri: 2_000,
                fg: 1_000,
                eg: 500,
                tg: 300,
                vg: 100,
                tfg: 50,
                ffg: 0,
                efg: 0,
                ofg: 0,
            },
            500_000..=1_999_999 => Self {
                uni: 16_000,
                bi: 8_000,
                tri: 4_000,
                fg: 2_000,
                eg: 1_000,
                tg: 1_000,
                vg: 500,
                tfg: 200,
                ffg: 100,
                efg: 0,
                ofg: 0,
            },
            2_000_000..=9_999_999 => Self {
                uni: 32_000,
                bi: 16_000,
                tri: 8_000,
                fg: 4_000,
                eg: 2_000,
                tg: 2_000,
                vg: 1_000,
                tfg: 500,
                ffg: 200,
                efg: 200,
                ofg: 100,
            },
            _ => Self {
                uni: 64_000,
                bi: 32_000,
                tri: 32_000,
                fg: 16_000,
                eg: 4_000,
                tg: 4_000,
                vg: 2_000,
                tfg: 2_000,
                ffg: 1_000,
                efg: 1_000,
                ofg: 500,
            },
        }
    }

    /// Sizes for an n-gram tier by level (0=trigram .. 8=144-gram).
    #[must_use]
    pub const fn ngram_budget(&self, level: usize) -> u32 {
        match level {
            0 => self.tri,
            1 => self.fg,
            2 => self.eg,
            3 => self.tg,
            4 => self.vg,
            5 => self.tfg,
            6 => self.ffg,
            7 => self.efg,
            8 => self.ofg,
            _ => 0,
        }
    }
}

/// Common interface satisfied by both codebook implementations.
///
/// All methods are `&self` after construction — codebooks are immutable
/// at query time (matching the upstream model: build once per corpus,
/// query many times during encode/decode).
pub trait Codebook: Send + Sync {
    /// Number of unique words across the input.
    fn n_unique(&self) -> u32;
    /// Unigram codebook size (≤ `n_unique`).
    fn n_uni(&self) -> u32;
    /// Bigram codebook size.
    fn n_bi(&self) -> u32;
    /// N-gram codebook size at `level` (0=trigram .. 8=144-gram).
    fn n_ngram(&self, level: usize) -> u32;

    /// `unigram_index(word_id)` → codebook index, if present.
    fn unigram_index(&self, word_id: u32) -> Option<u32>;
    /// `bigram_index(w1, w2)` → codebook index, if present.
    fn bigram_index(&self, w1: u32, w2: u32) -> Option<u32>;
    /// `ngram_index(level, &[w0..wn-1])` → codebook index, if present.
    fn ngram_index(&self, level: usize, words: &[u32]) -> Option<u32>;

    /// Reverse lookup: unigram codebook index → word ID.
    fn unigram_word(&self, idx: u32) -> Option<u32>;
    /// Reverse lookup: bigram codebook index → word-ID pair.
    fn bigram_words(&self, idx: u32) -> Option<(u32, u32)>;
    /// Reverse lookup: n-gram codebook index → word-ID tuple.
    fn ngram_words(&self, level: usize, idx: u32) -> Option<Vec<u32>>;
}

/// Direct flat-storage port of `cb.c`'s codebook layout.
///
/// Storage shape mirrors `qtc_cbs_t`: flat `Vec`s per tier (unigram word
/// IDs, bigram word-ID pairs, n-gram word-ID tuples) plus `HashMap`s
/// for lookup.
#[derive(Debug, Clone, Default)]
pub struct FlatCodebook {
    n_unique: u32,
    /// `uni_wids[i]` = word_id of the i-th unigram entry.
    uni_wids: Vec<u32>,
    /// Reverse map: word_id → unigram index.
    uni_lookup: HashMap<u32, u32>,
    /// `bi_wids[2i], bi_wids[2i+1]` = word-ID pair of the i-th bigram entry.
    bi_wids: Vec<u32>,
    /// `(w1, w2)` → bigram index.
    bi_lookup: HashMap<(u32, u32), u32>,
    /// Per-level n-gram word-ID tuples (flat `Vec` of length `n * level_size`).
    ng_wids: [Vec<u32>; N_LEVELS],
    /// Per-level reverse map: word-ID tuple → n-gram index.
    ng_lookup: [HashMap<Vec<u32>, u32>; N_LEVELS],
}

impl FlatCodebook {
    /// Build a flat codebook from the corpus.
    ///
    /// `word_ids` is the per-word ID sequence after interning;
    /// `n_unique` is the number of distinct word IDs (≥ max value in `word_ids` + 1);
    /// `sizes` is the per-tier budget.
    #[must_use]
    pub fn build(word_ids: &[u32], n_unique: u32, sizes: CodebookSizes) -> Self {
        let mut cb = Self {
            n_unique,
            ..Default::default()
        };
        build_unigrams(&mut cb, word_ids, n_unique, sizes.uni);
        build_bigrams(&mut cb, word_ids, sizes.bi);
        for (level, &ng_len) in NG_LENS.iter().enumerate() {
            let budget = sizes.ngram_budget(level);
            if budget > 0 && word_ids.len() >= ng_len {
                build_ngrams(&mut cb, level, ng_len, word_ids, budget);
            }
        }
        cb
    }
}

impl Codebook for FlatCodebook {
    fn n_unique(&self) -> u32 {
        self.n_unique
    }
    fn n_uni(&self) -> u32 {
        self.uni_wids.len() as u32
    }
    fn n_bi(&self) -> u32 {
        (self.bi_wids.len() / 2) as u32
    }
    fn n_ngram(&self, level: usize) -> u32 {
        if level >= N_LEVELS {
            return 0;
        }
        (self.ng_wids[level].len() / NG_LENS[level]) as u32
    }

    fn unigram_index(&self, word_id: u32) -> Option<u32> {
        self.uni_lookup.get(&word_id).copied()
    }
    fn bigram_index(&self, w1: u32, w2: u32) -> Option<u32> {
        self.bi_lookup.get(&(w1, w2)).copied()
    }
    fn ngram_index(&self, level: usize, words: &[u32]) -> Option<u32> {
        if level >= N_LEVELS || words.len() != NG_LENS[level] {
            return None;
        }
        self.ng_lookup[level].get(words).copied()
    }

    fn unigram_word(&self, idx: u32) -> Option<u32> {
        self.uni_wids.get(idx as usize).copied()
    }
    fn bigram_words(&self, idx: u32) -> Option<(u32, u32)> {
        let i = idx as usize * 2;
        if i + 1 < self.bi_wids.len() {
            Some((self.bi_wids[i], self.bi_wids[i + 1]))
        } else {
            None
        }
    }
    fn ngram_words(&self, level: usize, idx: u32) -> Option<Vec<u32>> {
        if level >= N_LEVELS {
            return None;
        }
        let ng = NG_LENS[level];
        let start = idx as usize * ng;
        if start + ng > self.ng_wids[level].len() {
            return None;
        }
        Some(self.ng_wids[level][start..start + ng].to_vec())
    }
}

fn build_unigrams(cb: &mut FlatCodebook, word_ids: &[u32], n_unique: u32, budget: u32) {
    let mut freq = vec![0u32; n_unique as usize];
    for &w in word_ids {
        freq[w as usize] += 1;
    }
    let mut ents: Vec<(u32, u32)> = freq
        .iter()
        .copied()
        .enumerate()
        .map(|(i, c)| (i as u32, c))
        .collect();
    ents.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let take = (budget as usize).min(ents.len());
    cb.uni_wids = ents.iter().take(take).map(|&(w, _)| w).collect();
    cb.uni_lookup = cb
        .uni_wids
        .iter()
        .enumerate()
        .map(|(i, &w)| (w, i as u32))
        .collect();
}

fn build_bigrams(cb: &mut FlatCodebook, word_ids: &[u32], budget: u32) {
    if word_ids.len() < 2 {
        return;
    }
    let mut freq: HashMap<(u32, u32), u32> = HashMap::new();
    for w in word_ids.windows(2) {
        *freq.entry((w[0], w[1])).or_insert(0) += 1;
    }
    let mut ents: Vec<((u32, u32), u32)> = freq
        .into_iter()
        .filter(|&((w1, w2), _)| cb.uni_lookup.contains_key(&w1) && cb.uni_lookup.contains_key(&w2))
        .collect();
    ents.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let take = (budget as usize).min(ents.len());
    for &((w1, w2), _) in ents.iter().take(take) {
        let idx = (cb.bi_wids.len() / 2) as u32;
        cb.bi_wids.push(w1);
        cb.bi_wids.push(w2);
        cb.bi_lookup.insert((w1, w2), idx);
    }
}

fn build_ngrams(cb: &mut FlatCodebook, level: usize, ng_len: usize, word_ids: &[u32], budget: u32) {
    let mut freq: HashMap<Vec<u32>, u32> = HashMap::new();
    for w in word_ids.windows(ng_len) {
        *freq.entry(w.to_vec()).or_insert(0) += 1;
    }
    let mut ents: Vec<(Vec<u32>, u32)> = freq
        .into_iter()
        .filter(|(words, c)| *c >= 2 && words.iter().all(|w| cb.uni_lookup.contains_key(w)))
        .collect();
    ents.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let take = (budget as usize).min(ents.len());
    for (words, _) in ents.into_iter().take(take) {
        let idx = (cb.ng_wids[level].len() / ng_len) as u32;
        cb.ng_wids[level].extend_from_slice(&words);
        cb.ng_lookup[level].insert(words, idx);
    }
}

// ──────────────────────────────────────────────────────────────────────
// COW Adaptive Radix Trie codebook variant
// ──────────────────────────────────────────────────────────────────────

/// Adaptive Radix Tree node variants (Leis et al, 2013).
///
/// Three node types instead of the full four (Node4 / Node16 / Node48 /
/// Node256). Node48 is an optimization for 17..48-child density; this
/// implementation skips it and grows Node16 → Node256 directly. Less
/// dense but simpler; sufficient for the codebook role.
#[derive(Debug, Clone)]
enum ArtNode {
    /// 4-key direct array — for low-fan-out internal nodes.
    Node4 {
        keys: [u32; 4],
        children: [Option<Arc<ArtNode>>; 4],
        count: u8,
        leaf: Option<u32>,
    },
    /// 16-key direct array — for medium-fan-out internal nodes.
    Node16 {
        keys: Box<[u32; 16]>,
        children: Box<[Option<Arc<ArtNode>>; 16]>,
        count: u8,
        leaf: Option<u32>,
    },
    /// 256-key direct array — for high-fan-out internal nodes.
    Node256 {
        children: Box<[Option<Arc<ArtNode>>; 256]>,
        leaf: Option<u32>,
    },
}

impl ArtNode {
    fn new_empty() -> Self {
        Self::Node4 {
            keys: [0; 4],
            children: [const { None }; 4],
            count: 0,
            leaf: None,
        }
    }

    fn leaf(&self) -> Option<u32> {
        match self {
            Self::Node4 { leaf, .. } | Self::Node16 { leaf, .. } | Self::Node256 { leaf, .. } => {
                *leaf
            }
        }
    }

    fn set_leaf(&mut self, value: Option<u32>) {
        match self {
            Self::Node4 { leaf, .. } | Self::Node16 { leaf, .. } | Self::Node256 { leaf, .. } => {
                *leaf = value
            }
        }
    }

    fn child(&self, key: u32) -> Option<&Arc<ArtNode>> {
        match self {
            Self::Node4 {
                keys,
                children,
                count,
                ..
            } => {
                for i in 0..*count as usize {
                    if keys[i] == key {
                        return children[i].as_ref();
                    }
                }
                None
            }
            Self::Node16 {
                keys,
                children,
                count,
                ..
            } => {
                for i in 0..*count as usize {
                    if keys[i] == key {
                        return children[i].as_ref();
                    }
                }
                None
            }
            Self::Node256 { children, .. } => {
                if key < 256 {
                    children[key as usize].as_ref()
                } else {
                    None
                }
            }
        }
    }

    /// Insert/replace a child for `key`. Returns the new node if the variant
    /// had to grow.
    fn with_child(&self, key: u32, child: Arc<ArtNode>) -> Self {
        let mut new = self.clone();
        new.put_child(key, child);
        new
    }

    fn put_child(&mut self, key: u32, child: Arc<ArtNode>) {
        // Try to replace existing.
        if self.replace_child(key, child.clone()) {
            return;
        }
        // Need to insert; grow if necessary.
        loop {
            match self {
                Self::Node4 {
                    keys,
                    children,
                    count,
                    ..
                } => {
                    if (*count as usize) < 4 {
                        keys[*count as usize] = key;
                        children[*count as usize] = Some(child);
                        *count += 1;
                        return;
                    }
                    self.grow_to_16();
                }
                Self::Node16 {
                    keys,
                    children,
                    count,
                    ..
                } => {
                    if (*count as usize) < 16 && key < 256 {
                        keys[*count as usize] = key;
                        children[*count as usize] = Some(child);
                        *count += 1;
                        return;
                    }
                    self.grow_to_256();
                }
                Self::Node256 { children, .. } => {
                    if key < 256 {
                        children[key as usize] = Some(child);
                    }
                    return;
                }
            }
        }
    }

    fn replace_child(&mut self, key: u32, child: Arc<ArtNode>) -> bool {
        match self {
            Self::Node4 {
                keys,
                children,
                count,
                ..
            } => {
                for i in 0..*count as usize {
                    if keys[i] == key {
                        children[i] = Some(child);
                        return true;
                    }
                }
                false
            }
            Self::Node16 {
                keys,
                children,
                count,
                ..
            } => {
                for i in 0..*count as usize {
                    if keys[i] == key {
                        children[i] = Some(child);
                        return true;
                    }
                }
                false
            }
            Self::Node256 { children, .. } => {
                if key < 256 && children[key as usize].is_some() {
                    children[key as usize] = Some(child);
                    true
                } else {
                    false
                }
            }
        }
    }

    fn grow_to_16(&mut self) {
        let (old_keys, old_children, old_count, old_leaf) = match self {
            Self::Node4 {
                keys,
                children,
                count,
                leaf,
            } => (*keys, std::mem::take(children), *count, *leaf),
            _ => return,
        };
        let mut new_keys = Box::new([0u32; 16]);
        let mut new_children: Box<[Option<Arc<ArtNode>>; 16]> =
            Box::new(std::array::from_fn(|_| None));
        for i in 0..old_count as usize {
            new_keys[i] = old_keys[i];
            new_children[i] = old_children[i].clone();
        }
        *self = Self::Node16 {
            keys: new_keys,
            children: new_children,
            count: old_count,
            leaf: old_leaf,
        };
    }

    fn grow_to_256(&mut self) {
        let (old_keys, old_children, old_count, old_leaf) = match self {
            Self::Node16 {
                keys,
                children,
                count,
                leaf,
            } => {
                let keys = **keys;
                let children = std::mem::replace(children, Box::new(std::array::from_fn(|_| None)));
                (keys, children, *count, *leaf)
            }
            _ => return,
        };
        let mut new_children: Box<[Option<Arc<ArtNode>>; 256]> =
            Box::new(std::array::from_fn(|_| None));
        for i in 0..old_count as usize {
            let k = old_keys[i] as usize;
            if k < 256 {
                new_children[k] = old_children[i].clone();
            }
        }
        *self = Self::Node256 {
            children: new_children,
            leaf: old_leaf,
        };
    }
}

/// COW Adaptive Radix Trie keyed by `&[u32]`, valued by `u32` codebook index.
///
/// Insertion produces a new root via path-copy; the previous root remains
/// valid for prior consumers. Reads are `&self` and share structure across
/// versions via `Arc`.
#[derive(Debug, Clone)]
pub struct CowArt {
    root: Arc<ArtNode>,
    len: u32,
}

impl Default for CowArt {
    fn default() -> Self {
        Self::new()
    }
}

impl CowArt {
    /// Empty trie.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: Arc::new(ArtNode::new_empty()),
            len: 0,
        }
    }

    /// Number of key-value pairs.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> u32 {
        self.len
    }

    /// True iff empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Look up `key`. Returns `Some(value)` if present.
    #[must_use]
    pub fn get(&self, key: &[u32]) -> Option<u32> {
        let mut node = self.root.as_ref();
        for &k in key {
            node = node.child(k)?.as_ref();
        }
        node.leaf()
    }

    /// Insert `key → value`. Returns a new trie sharing structure with
    /// `self` everywhere the key path does not touch.
    #[must_use]
    pub fn insert(&self, key: &[u32], value: u32) -> Self {
        let (new_root, inserted) = insert_rec(&self.root, key, value);
        Self {
            root: Arc::new(new_root),
            len: self.len + u32::from(inserted),
        }
    }
}

fn insert_rec(node: &Arc<ArtNode>, key: &[u32], value: u32) -> (ArtNode, bool) {
    if key.is_empty() {
        let mut new_node = node.as_ref().clone();
        let was_present = new_node.leaf().is_some();
        new_node.set_leaf(Some(value));
        return (new_node, !was_present);
    }
    let head = key[0];
    let tail = &key[1..];

    let (child_new, inserted) = match node.child(head) {
        Some(child) => insert_rec(child, tail, value),
        None => {
            let leaf = ArtNode::new_empty();
            insert_rec(&Arc::new(leaf), tail, value)
        }
    };

    let new_node = node.with_child(head, Arc::new(child_new));
    (new_node, inserted)
}

/// Codebook backed by [`CowArt`] tries — one per tier.
///
/// Each tier (unigram / bigram / n-gram per level) gets its own COW trie.
/// The forward direction (key → codebook index) is the trie; the reverse
/// direction (codebook index → key) is the auxiliary `Vec` filled at
/// construction time.
#[derive(Debug, Clone, Default)]
pub struct CowRadixCodebook {
    n_unique: u32,
    uni_trie: CowArt,
    uni_wids: Vec<u32>,
    bi_trie: CowArt,
    bi_wids: Vec<u32>,
    ng_tries: [CowArt; N_LEVELS],
    ng_wids: [Vec<u32>; N_LEVELS],
}

impl CowRadixCodebook {
    /// Build a COW-radix codebook from the corpus.
    ///
    /// Currently uses the same frequency-sort selection as
    /// [`FlatCodebook::build`]; the COW property is exercised at insertion
    /// time (each entry produces a new trie root via path-copy).
    #[must_use]
    pub fn build(word_ids: &[u32], n_unique: u32, sizes: CodebookSizes) -> Self {
        // Reuse the flat construction logic to select entries by frequency,
        // then materialize them into COW tries.
        let flat = FlatCodebook::build(word_ids, n_unique, sizes);
        let mut cb = Self {
            n_unique,
            ..Default::default()
        };

        for (i, &w) in flat.uni_wids.iter().enumerate() {
            cb.uni_trie = cb.uni_trie.insert(&[w], i as u32);
            cb.uni_wids.push(w);
        }
        for i in 0..flat.bi_wids.len() / 2 {
            let w1 = flat.bi_wids[2 * i];
            let w2 = flat.bi_wids[2 * i + 1];
            cb.bi_trie = cb.bi_trie.insert(&[w1, w2], i as u32);
            cb.bi_wids.push(w1);
            cb.bi_wids.push(w2);
        }
        for (level, ngs) in flat.ng_wids.iter().enumerate() {
            let ng = NG_LENS[level];
            for (idx, chunk) in ngs.chunks_exact(ng).enumerate() {
                cb.ng_tries[level] = cb.ng_tries[level].insert(chunk, idx as u32);
                cb.ng_wids[level].extend_from_slice(chunk);
            }
        }

        cb
    }
}

impl Codebook for CowRadixCodebook {
    fn n_unique(&self) -> u32 {
        self.n_unique
    }
    fn n_uni(&self) -> u32 {
        self.uni_wids.len() as u32
    }
    fn n_bi(&self) -> u32 {
        (self.bi_wids.len() / 2) as u32
    }
    fn n_ngram(&self, level: usize) -> u32 {
        if level >= N_LEVELS {
            return 0;
        }
        (self.ng_wids[level].len() / NG_LENS[level]) as u32
    }

    fn unigram_index(&self, word_id: u32) -> Option<u32> {
        self.uni_trie.get(&[word_id])
    }
    fn bigram_index(&self, w1: u32, w2: u32) -> Option<u32> {
        self.bi_trie.get(&[w1, w2])
    }
    fn ngram_index(&self, level: usize, words: &[u32]) -> Option<u32> {
        if level >= N_LEVELS || words.len() != NG_LENS[level] {
            return None;
        }
        self.ng_tries[level].get(words)
    }

    fn unigram_word(&self, idx: u32) -> Option<u32> {
        self.uni_wids.get(idx as usize).copied()
    }
    fn bigram_words(&self, idx: u32) -> Option<(u32, u32)> {
        let i = idx as usize * 2;
        if i + 1 < self.bi_wids.len() {
            Some((self.bi_wids[i], self.bi_wids[i + 1]))
        } else {
            None
        }
    }
    fn ngram_words(&self, level: usize, idx: u32) -> Option<Vec<u32>> {
        if level >= N_LEVELS {
            return None;
        }
        let ng = NG_LENS[level];
        let start = idx as usize * ng;
        if start + ng > self.ng_wids[level].len() {
            return None;
        }
        Some(self.ng_wids[level][start..start + ng].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_corpus() -> Vec<u32> {
        // 30 words, vocabulary {0..=4}, with deterministic bigram/trigram repeats.
        vec![
            0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 0, 1,
            2,
        ]
    }

    #[test]
    fn codebook_sizes_auto_increases_with_corpus() {
        let s_small = CodebookSizes::auto(1_000);
        let s_big = CodebookSizes::auto(5_000_000);
        assert!(s_big.uni > s_small.uni);
        assert!(s_big.ofg > 0);
        assert_eq!(s_small.ofg, 0); // 144-gram inactive at small scale
    }

    #[test]
    fn flat_codebook_roundtrips_unigrams() {
        let corpus = small_corpus();
        let sizes = CodebookSizes::auto(corpus.len() as u32);
        let cb = FlatCodebook::build(&corpus, 5, sizes);
        for w in 0..5u32 {
            let idx = cb.unigram_index(w).expect("present");
            assert_eq!(cb.unigram_word(idx), Some(w));
        }
    }

    #[test]
    fn flat_codebook_roundtrips_bigrams() {
        let corpus = small_corpus();
        let sizes = CodebookSizes::auto(corpus.len() as u32);
        let cb = FlatCodebook::build(&corpus, 5, sizes);
        // (0,1), (1,2), (2,0), (2,3), (3,4), (4,0) are present in corpus.
        for (w1, w2) in [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 0)] {
            let idx = cb.bigram_index(w1, w2).expect("bigram present");
            assert_eq!(cb.bigram_words(idx), Some((w1, w2)));
        }
    }

    #[test]
    fn cow_radix_codebook_roundtrips_unigrams() {
        let corpus = small_corpus();
        let sizes = CodebookSizes::auto(corpus.len() as u32);
        let cb = CowRadixCodebook::build(&corpus, 5, sizes);
        for w in 0..5u32 {
            let idx = cb.unigram_index(w).expect("present");
            assert_eq!(cb.unigram_word(idx), Some(w));
        }
    }

    #[test]
    fn cow_radix_codebook_roundtrips_bigrams() {
        let corpus = small_corpus();
        let sizes = CodebookSizes::auto(corpus.len() as u32);
        let cb = CowRadixCodebook::build(&corpus, 5, sizes);
        for (w1, w2) in [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 0)] {
            let idx = cb.bigram_index(w1, w2).expect("present");
            assert_eq!(cb.bigram_words(idx), Some((w1, w2)));
        }
    }

    #[test]
    fn cow_radix_codebook_agrees_with_flat_on_lookups() {
        let corpus = small_corpus();
        let sizes = CodebookSizes::auto(corpus.len() as u32);
        let flat = FlatCodebook::build(&corpus, 5, sizes);
        let cow = CowRadixCodebook::build(&corpus, 5, sizes);
        for w in 0..5u32 {
            assert_eq!(flat.unigram_index(w), cow.unigram_index(w), "uni {w}");
        }
        for w1 in 0..5u32 {
            for w2 in 0..5u32 {
                assert_eq!(
                    flat.bigram_index(w1, w2),
                    cow.bigram_index(w1, w2),
                    "bi ({w1},{w2})"
                );
            }
        }
    }

    #[test]
    fn cow_art_path_copy_preserves_old_root() {
        let art_v0 = CowArt::new();
        let art_v1 = art_v0.insert(&[1, 2, 3], 42);
        let art_v2 = art_v1.insert(&[1, 2, 4], 99);

        // v0 still empty.
        assert_eq!(art_v0.len(), 0);
        assert_eq!(art_v0.get(&[1, 2, 3]), None);
        // v1 has the first insert, v2 has both.
        assert_eq!(art_v1.len(), 1);
        assert_eq!(art_v1.get(&[1, 2, 3]), Some(42));
        assert_eq!(art_v1.get(&[1, 2, 4]), None);
        assert_eq!(art_v2.len(), 2);
        assert_eq!(art_v2.get(&[1, 2, 3]), Some(42));
        assert_eq!(art_v2.get(&[1, 2, 4]), Some(99));
    }

    #[test]
    fn cow_art_grows_node_variants() {
        // Force Node4 → Node16 → Node256 growth on a single root.
        let mut art = CowArt::new();
        for k in 0..200u32 {
            art = art.insert(&[k], k);
        }
        assert_eq!(art.len(), 200);
        for k in 0..200u32 {
            assert_eq!(art.get(&[k]), Some(k));
        }
    }
}
