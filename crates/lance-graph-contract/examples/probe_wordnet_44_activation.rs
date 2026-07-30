//! PROBE-WORDNET-44-ACTIVATION — does a 4⁴ address make HHTL adjacency
//! *semantic*, and does 4-ary buy resolution the nibble cannot?
//!
//! ## The claim under test (operator, 2026-07-30)
//!
//! > "wordnet makes CLAM HHTL spatial activation via 4^4"
//!
//! and, one message earlier:
//!
//! > "4^4 would be a sparse adjacent if the sentence refers to out of bounds
//! > meaning"
//!
//! Read together: when a reference lands outside the current cell, the first
//! hypothesis set should NOT be "the whole codebook" and should NOT be an
//! immediate cross-domain jump — it should be the **sparse adjacent band**, the
//! ancestry-neighbours reachable by flipping the last 2-bit group, then the one
//! above it, and so on. At 4-ary that band is `3 siblings × 4 levels = 12` cells
//! out of 256 (**4.7 %**) — a graded ladder. At 16-ary ([`NiblePath`], the
//! shipped router, `FAN_OUT = 16`) a byte has only TWO levels, so the ladder has
//! two rungs and no sub-nibble structure exists at all.
//!
//! The reason WordNet is the right corpus is that it removes the hard part:
//! [`Palette::build_hierarchical`]-style codebooks must *discover* ancestry with
//! k-means and hope it survives, whereas WordNet's `@` hypernym relation IS
//! ground-truth ancestry. So this probe does not ask "did clustering find
//! structure?" — it asks the sharper question: **given real taxonomic ancestry,
//! does folding it into a fixed 4-ary depth-4 address preserve semantic
//! distance well enough that address-adjacency is a usable search prior?**
//!
//! ## Honest boundaries (read before citing)
//!
//! - **This is a STRUCTURE probe, not a codec probe.** No embeddings, no
//!   Base17, no k-means. It measures the fold, nothing downstream of it. It
//!   says nothing about whether an *embedding-trained* 4⁴ codebook inherits
//!   these properties — that is the separate Phase-B question, and
//!   `PROBE-CODEBOOK-44`'s real-data leg already warns that a Base17 fold
//!   ceiling (ρ≈0.26 on single words) caps codebook fidelity independently.
//! - **WordNet's noun graph is a DAG; this takes the FIRST `@` parent only**,
//!   turning it into a tree. Multiple inheritance is real and is dropped. Both
//!   sides of every comparison below see the same tree, so the simplification
//!   cannot flatter the result — but a concept with two genuine parents is
//!   representable at only one address, which is a real limit of any
//!   fixed-arity address and is reported, not hidden.
//! - **Path length is a crude semantic metric** (it ignores information
//!   content — Resnik/Lin would weight by corpus frequency). It is used because
//!   it is the metric the ADDRESS is trying to approximate: both are pure
//!   structure. A frequency-weighted rerun is future work.
//!
//! ## Gates
//!
//! - **W1 ancestry-is-by-construction** — a prefix of the address is a genuine
//!   fold-ancestor, and a SHUFFLED address assignment (same cell sizes, permuted
//!   labels) destroys it. Anti-vacuity: the shuffle arm must fail.
//! - **W2 monotone ladder** — mean WordNet path distance must increase strictly
//!   with adjacency rung: same-cell < share-3 < share-2 < share-1 < share-0.
//!   This IS the sparse-adjacency claim. Falsifier: any inversion, or a flat
//!   profile (address distance carrying no semantic signal).
//! - **W3 spatial activation** — for an anchor concept, do its 12 ancestry-
//!   adjacent cells actually hold its semantic neighbours?
//!   - **PRIMARY statistic: OUT-OF-CELL recall.** Among the anchor's nearest
//!     WordNet neighbours that fall OUTSIDE its home cell, what fraction land in
//!     the 12-cell band, vs 12 distinct random cells? Out-of-cell is the claim:
//!     a neighbour already in the home cell needs no adjacency band to reach it.
//!     The null is self-checking — a random 12 of the remaining 255 cells should
//!     score ≈ 12/255 ≈ 4.7 %.
//!   - **SECONDARY, retained as a saturating diagnostic:** recall over ALL 32
//!     nearest distinct neighbours, where both arms credit the home cell. This
//!     was the original primary and it is the WRONG statistic — home-cell hits
//!     inflate the baseline to ≈0.62, capping the achievable ratio near 1.6 and
//!     MASKING the effect. Kept only so the two are comparable in the record.
//!   - Twin-tested per the can-it-fire/can-it-stay-silent rule: the band must
//!     beat random (fire) AND must not be a cover (silent). The cover guard is
//!     CALIBRATED, not asserted — a deliberately coarse 2-level address (6+1 of
//!     16 cells) is measured alongside and must be REJECTED by it.
//!   - **Specification check that this gate failed once:** before running a twin
//!     gate, compute the maximum achievable value of the fire statistic *under*
//!     the silent guard. Here that was 1/0.62 ≈ 1.61 against a `>1.5` fire bar
//!     and a `<0.95` cover bar — an 0.018-wide window, i.e. a mis-specified
//!     gate whose first pass was luck rather than evidence.
//! - **W4 4-ary beats the nibble** — the 16-ary ladder's coarsest useful rung
//!   ("same top nibble") is a single bucket of 16 cells. The 4-ary ladder splits
//!   that same population into rungs 1 and 2. Gate: those two sub-rungs must
//!   differ in mean distance by a real margin — i.e. the sub-nibble structure
//!   the nibble router CANNOT see is semantically load-bearing. Falsifier: the
//!   split is flat, meaning 4⁴ micro-traversal is decoration and the nibble
//!   loses nothing.
//! - **W5 fold balance** — report min/max/median cell occupancy. The
//!   le-contract's own warning ("lacking proper bucket rollover ... saturates
//!   silently") applies to this fold; a wildly unbalanced fold would make W2/W3
//!   look good for the wrong reason (one giant cell).
//!
//! ## Running
//!
//! ```sh
//! curl -sSL -o /tmp/wn31.tar.gz https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz
//! mkdir -p /tmp/wn31 && tar xzf /tmp/wn31.tar.gz -C /tmp/wn31
//! python3 crates/lance-graph-planner/examples/data/wordnet/build_isa_tree.py \
//!     /tmp/wn31/dict /tmp/wordnet_isa_tree.tsv
//! cargo run -p lance-graph-contract --example probe_wordnet_44_activation
//! ```
//!
//! The corpus is LOCAL-ONLY and gitignored (workspace convention); the
//! generator is committed, the data is not.
//!
//! [`NiblePath`]: lance_graph_contract::hhtl::NiblePath
//! [`Palette::build_hierarchical`]: https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/bgz17/src/palette.rs

use std::collections::HashMap;
use std::path::Path;

// ── the fold ────────────────────────────────────────────────────────────────

/// Address levels. 4 levels × 2 bits = one byte = 256 cells.
const LEVELS: usize = 4;
/// Arity per level. `4^4 = 256` — the operator's shape.
const ARITY: usize = 4;
const CELLS: usize = 256;
/// Sibling cells reachable by flipping ONE level's 2-bit group:
/// `(ARITY - 1) * LEVELS = 12` of 256 = 4.7 %.
const BAND: usize = (ARITY - 1) * LEVELS;

struct Tree {
    /// Dense index space. `parent[i]` = index of first `@` hypernym, or `usize::MAX`.
    parent: Vec<usize>,
    children: Vec<Vec<usize>>,
    depth: Vec<u16>,
    lemma: Vec<String>,
}

impl Tree {
    fn load(path: &Path) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut offsets: Vec<u32> = Vec::new();
        let mut raw: Vec<(u32, u32, u16, String)> = Vec::new();
        for line in text.lines() {
            if line.starts_with('#') {
                continue;
            }
            let mut f = line.split('\t');
            let (Some(c), Some(p), Some(d), Some(l)) = (f.next(), f.next(), f.next(), f.next())
            else {
                continue;
            };
            let (c, p) = (c.parse::<u32>().unwrap_or(0), p.parse::<u32>().unwrap_or(0));
            raw.push((c, p, d.parse::<u16>().unwrap_or(0), l.to_string()));
            offsets.push(c);
        }
        offsets.sort_unstable();
        let index: HashMap<u32, usize> = offsets.iter().enumerate().map(|(i, &o)| (o, i)).collect();

        let n = raw.len();
        let mut t = Tree {
            parent: vec![usize::MAX; n],
            children: vec![Vec::new(); n],
            depth: vec![0; n],
            lemma: vec![String::new(); n],
        };
        for (c, p, d, l) in raw {
            let i = index[&c];
            t.depth[i] = d;
            t.lemma[i] = l;
            if p != 0 {
                if let Some(&pi) = index.get(&p) {
                    t.parent[i] = pi;
                    t.children[pi].push(i);
                }
            }
        }
        Ok(t)
    }

    fn is_leaf(&self, i: usize) -> bool {
        self.children[i].is_empty()
    }

    /// Number of leaves under `i` (memoized iteratively — depth 19 is safe for
    /// recursion, but the iterative form keeps this honest for any corpus).
    fn leaf_counts(&self) -> Vec<u32> {
        let n = self.parent.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(self.depth[i]));
        let mut counts = vec![0u32; n];
        for &i in &order {
            if self.is_leaf(i) {
                counts[i] = 1;
            }
            if self.parent[i] != usize::MAX {
                counts[self.parent[i]] += counts[i];
            }
        }
        counts
    }

    /// WordNet path distance: `depth(a) + depth(b) − 2·depth(LCA)`.
    fn path_distance(&self, a: usize, b: usize) -> u32 {
        let (mut x, mut y) = (a, b);
        while self.depth[x] > self.depth[y] {
            x = self.parent[x];
        }
        while self.depth[y] > self.depth[x] {
            y = self.parent[y];
        }
        while x != y {
            if self.parent[x] == usize::MAX || self.parent[y] == usize::MAX {
                return u32::MAX; // different roots (should not happen: 1 root)
            }
            x = self.parent[x];
            y = self.parent[y];
        }
        (self.depth[a] + self.depth[b] - 2 * self.depth[x]) as u32
    }
}

/// Greedily bin a forest's roots into `ARITY` buckets balanced by leaf count —
/// the "proper bucket rollover" the le-contract demands of any fixed-width
/// carving. Largest-first into the currently-lightest bucket (LPT scheduling).
fn balance(roots: &[usize], weight: &[u32]) -> [Vec<usize>; ARITY] {
    let mut sorted: Vec<usize> = roots.to_vec();
    sorted.sort_by_key(|&r| std::cmp::Reverse(weight[r]));
    let mut buckets: [Vec<usize>; ARITY] = Default::default();
    let mut load = [0u64; ARITY];
    for r in sorted {
        let (mut best, mut best_load) = (0, u64::MAX);
        for (b, &l) in load.iter().enumerate() {
            if l < best_load {
                best_load = l;
                best = b;
            }
        }
        load[best] += weight[r].max(1) as u64;
        buckets[best].push(r);
    }
    buckets
}

/// Fold the taxonomy into 4⁴ addresses. Every LEAF synset gets one byte; the
/// two-bit group at level `k` names the bucket it fell into at that level, so
/// `addr >> (2·(LEVELS−k))` is a genuine fold-ancestor by construction.
///
/// A node whose subtree is still being subdivided contributes its children to
/// the next level; internal nodes are not addressed (they are the skeleton, not
/// the concepts) — see the module's honest-boundaries note.
fn fold(tree: &Tree, root: usize, leaves: &[u32]) -> Vec<Option<u8>> {
    let mut addr = vec![None; tree.parent.len()];
    // Work queue: (forest roots, address prefix, level)
    let mut stack: Vec<(Vec<usize>, u8, usize)> = vec![(vec![root], 0u8, 0usize)];
    while let Some((forest, prefix, level)) = stack.pop() {
        if level == LEVELS {
            // Terminal cell: every leaf under this forest gets `prefix`.
            let mut work = forest;
            while let Some(i) = work.pop() {
                if tree.is_leaf(i) {
                    addr[i] = Some(prefix);
                } else {
                    work.extend_from_slice(&tree.children[i]);
                }
            }
            continue;
        }
        // Expand until the forest is FINE-GRAINED enough to balance, not merely
        // splittable. Expanding to exactly ARITY roots is what produced the
        // first run's 15,769-leaf cell against a median of 20: with 4 coarse
        // roots of wildly unequal weight, LPT can only isolate the giant one and
        // hope the next level splits it — and at the terminal level there is no
        // next level, so it lands whole. This is precisely the le-contract's
        // "lacking proper bucket rollover … saturates silently". Expanding to
        // ARITY·GRAIN gives the balancer small enough items to actually level.
        const GRAIN: usize = 24;
        let mut roots = forest;
        while roots.len() < ARITY * GRAIN {
            let expandable = roots
                .iter()
                .enumerate()
                .filter(|(_, &r)| !tree.is_leaf(r))
                .max_by_key(|(_, &r)| leaves[r])
                .map(|(pos, _)| pos);
            match expandable {
                Some(pos) => {
                    let r = roots.swap_remove(pos);
                    roots.extend_from_slice(&tree.children[r]);
                    // A leaf-bearing internal node keeps its own leaf identity
                    // only if it IS a leaf, which it is not here — nothing lost.
                }
                None => break, // all leaves, fewer than ARITY: buckets under-fill
            }
        }
        for (b, bucket) in balance(&roots, leaves).into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            stack.push((bucket, (prefix << 2) | b as u8, level + 1));
        }
    }
    addr
}

// ── metrics ─────────────────────────────────────────────────────────────────

/// Levels of shared address prefix (0..=LEVELS). 4 = same cell.
fn shared_levels(a: u8, b: u8) -> usize {
    for k in 0..LEVELS {
        let shift = 2 * (LEVELS - k) - 2;
        if (a >> shift) != (b >> shift) {
            return k;
        }
    }
    LEVELS
}

/// The sparse-adjacent band: cells reachable by changing exactly one 2-bit
/// group. `BAND` = 12 cells, never including `cell` itself.
/// The band for an address of arbitrary depth — used by the calibration arm to
/// build a deliberately COARSE (2-level, 16-cell) address space whose band is
/// large enough that it must behave like a cover. That is what turns the
/// `< 0.95` upper guard from a hand-chosen constant into a threshold shown to
/// discriminate (the workspace's inertness rule: raising a knob must silence
/// something, lowering it must admit something).
fn band_generic(cell: u8, levels: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity((ARITY - 1) * levels);
    for level in 0..levels {
        let shift = 2 * (levels - 1 - level);
        let cur = (cell >> shift) & 0b11;
        for v in 0..ARITY as u8 {
            if v != cur {
                out.push((cell & !(0b11 << shift)) | (v << shift));
            }
        }
    }
    out
}

fn band_of(cell: u8) -> [u8; BAND] {
    let mut out = [0u8; BAND];
    let mut n = 0;
    for level in 0..LEVELS {
        let shift = 2 * (LEVELS - 1 - level);
        let cur = (cell >> shift) & 0b11;
        for v in 0..ARITY as u8 {
            if v != cur {
                out[n] = (cell & !(0b11 << shift)) | (v << shift);
                n += 1;
            }
        }
    }
    debug_assert_eq!(n, BAND);
    out
}

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn mean(v: &[u32]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64
}

// ── gates ───────────────────────────────────────────────────────────────────

static FAILURES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn gate(name: &str, pass: bool, detail: String) {
    println!(
        "  [{}] {name}\n        {detail}",
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/wordnet_isa_tree.tsv".to_string());
    let path = Path::new(&path);
    if !path.exists() {
        eprintln!(
            "corpus not found at {}\nsee the module docs for the 3-line fetch + generate recipe",
            path.display()
        );
        std::process::exit(2);
    }

    let tree = Tree::load(path).expect("load is-a tree");
    let n = tree.parent.len();
    let root = (0..n)
        .find(|&i| tree.parent[i] == usize::MAX)
        .expect("root");
    let leaves_under = tree.leaf_counts();
    let leaf_ids: Vec<usize> = (0..n).filter(|&i| tree.is_leaf(i)).collect();

    println!("PROBE-WORDNET-44-ACTIVATION");
    println!(
        "corpus: {n} synsets, root='{}', {} leaves, max depth {}\n",
        tree.lemma[root],
        leaf_ids.len(),
        tree.depth.iter().max().unwrap()
    );

    let addr = fold(&tree, root, &leaves_under);
    let addressed: Vec<usize> = leaf_ids
        .iter()
        .copied()
        .filter(|&i| addr[i].is_some())
        .collect();

    // ── W5 fold balance (reported first: W2/W3 are only meaningful if sane) ──
    let mut occupancy = [0usize; CELLS];
    for &i in &addressed {
        occupancy[addr[i].unwrap() as usize] += 1;
    }
    let used = occupancy.iter().filter(|&&c| c > 0).count();
    let mut occ_sorted: Vec<usize> = occupancy.iter().copied().filter(|&c| c > 0).collect();
    occ_sorted.sort_unstable();
    let (lo, hi) = (occ_sorted[0], *occ_sorted.last().unwrap());
    let median = occ_sorted[occ_sorted.len() / 2];
    gate(
        "W5 fold balance",
        used >= CELLS / 2 && hi <= 40 * median.max(1),
        format!(
            "{used}/{CELLS} cells used; occupancy min={lo} median={median} max={hi} \
             (addressed leaves={})",
            addressed.len()
        ),
    );

    // ── W1 ancestry-by-construction, with a shuffle arm as the falsifier ────
    let mut rng = SplitMix64(0x9E37_79B9_7F4A_7C15);
    let sample: Vec<usize> = (0..4000)
        .map(|_| addressed[rng.below(addressed.len())])
        .collect();

    // Real: two leaves sharing k address levels must share a fold-ancestor —
    // operationally, their WordNet LCA must be at least as deep as k allows.
    // Shuffled: permute the cell labels, destroying prefix meaning.
    let mut perm: Vec<u8> = (0..=255u8).collect();
    for i in (1..perm.len()).rev() {
        perm.swap(i, rng.below(i + 1));
    }
    let real_corr = prefix_vs_lca_corr(&tree, &sample, &addr, None);
    let shuf_corr = prefix_vs_lca_corr(&tree, &sample, &addr, Some(&perm));
    gate(
        "W1 ancestry-is-by-construction (shuffle falsifier)",
        real_corr > 0.30 && shuf_corr.abs() < 0.05 && real_corr > shuf_corr + 0.25,
        format!(
            "corr(shared address levels, LCA depth): real={real_corr:+.4}  \
             shuffled={shuf_corr:+.4}  (shuffle must collapse to ~0)"
        ),
    );

    // ── W2 monotone ladder ──────────────────────────────────────────────────
    let mut rungs: [Vec<u32>; LEVELS + 1] = Default::default();
    for _ in 0..200_000 {
        let a = addressed[rng.below(addressed.len())];
        let b = addressed[rng.below(addressed.len())];
        if a == b {
            continue;
        }
        let k = shared_levels(addr[a].unwrap(), addr[b].unwrap());
        let d = tree.path_distance(a, b);
        if d != u32::MAX {
            rungs[k].push(d);
        }
    }
    let means: Vec<f64> = rungs.iter().map(|r| mean(r)).collect();
    let monotone = (0..LEVELS).all(|k| means[k] > means[k + 1]);
    let spread = means[0] - means[LEVELS];
    gate(
        "W2 monotone ladder (share-0 → same-cell)",
        monotone && spread > 1.0,
        format!(
            "mean WordNet path distance by shared address levels: \
             0:{:.2} (n={})  1:{:.2} (n={})  2:{:.2} (n={})  3:{:.2} (n={})  4/same-cell:{:.2} (n={})  \
             — strictly decreasing={monotone}, spread={spread:.2} hops",
            means[0], rungs[0].len(),
            means[1], rungs[1].len(),
            means[2], rungs[2].len(),
            means[3], rungs[3].len(),
            means[4], rungs[4].len(),
        ),
    );

    // ── W3 spatial activation: band recall vs random-12 baseline ────────────
    const ANCHORS: usize = 300;
    const NEAR: usize = 32;
    let mut band_recall = Vec::new();
    let mut rand_recall = Vec::new();
    // Pool the search: comparing every anchor against 82k leaves is O(n²); use a
    // deterministic candidate pool per anchor, identical for both arms, so the
    // comparison is exact and the cost is bounded.
    // DISTINCT candidates. Sampling with replacement and then taking the
    // NEAR-smallest lets one synset occupy several slots, so the metric would be
    // a weighted sampled-entry recall — not "recall of the 32 nearest
    // neighbours" as documented. Both arms shared the bias so the ratio stayed
    // meaningful, but the LABEL was wrong, which is the defect. Dedupe first.
    let pool: Vec<usize> = {
        let mut seen = std::collections::HashSet::new();
        let mut v = Vec::with_capacity(20_000);
        while v.len() < 20_000 {
            let c = addressed[rng.below(addressed.len())];
            if seen.insert(c) {
                v.push(c);
            }
        }
        v
    };
    // Calibration arm for the upper guard (see the gate below): the SAME recall
    // measured against a deliberately COARSE address — only the top 2 levels,
    // so 16 cells and a 6-cell band = 37.5 % of that codebook instead of 4.7 %.
    // A band that big must behave like a cover, which is what makes the 0.95
    // threshold something that bites rather than decoration.
    let mut coarse_recall = Vec::new();
    // Out-of-cell arms — the primary statistic (see the in-loop note).
    let mut oob_band: Vec<f64> = Vec::new();
    let mut oob_rand: Vec<f64> = Vec::new();
    for _ in 0..ANCHORS {
        let anchor = addressed[rng.below(addressed.len())];
        let acell = addr[anchor].unwrap();
        let mut scored: Vec<(u32, usize)> = pool
            .iter()
            .filter(|&&p| p != anchor)
            .map(|&p| (tree.path_distance(anchor, p), p))
            .filter(|&(d, _)| d != u32::MAX)
            .collect();
        scored.sort_unstable();
        let near: Vec<usize> = scored.iter().take(NEAR).map(|&(_, p)| p).collect();
        if near.len() < NEAR {
            continue;
        }

        let band = band_of(acell);
        let in_band = |c: u8| c == acell || band.contains(&c);
        let hit = near.iter().filter(|&&p| in_band(addr[p].unwrap())).count();
        band_recall.push(hit as f64 / NEAR as f64);

        // PRIMARY statistic — OUT-OF-CELL neighbours only. The claim under test
        // is about references that land outside the current cell; a neighbour
        // already in the home cell needs no adjacency band to reach and is
        // credited to BOTH arms, which is what compressed the earlier ratio (see
        // the gate note). Restricting to out-of-cell neighbours measures what
        // the 12-cell band actually contributes.
        let out: Vec<usize> = near
            .iter()
            .copied()
            .filter(|&p| addr[p].unwrap() != acell)
            .collect();
        if !out.is_empty() {
            let b = out
                .iter()
                .filter(|&&p| band.contains(&addr[p].unwrap()))
                .count();
            oob_band.push(b as f64 / out.len() as f64);
        }

        // Baseline: BAND random cells + the anchor cell (same budget).
        let mut rcells: Vec<u8> = Vec::with_capacity(BAND);
        while rcells.len() < BAND {
            let c = rng.below(CELLS) as u8;
            if c != acell && !rcells.contains(&c) {
                rcells.push(c);
            }
        }
        let hit_r = near
            .iter()
            .filter(|&&p| {
                let c = addr[p].unwrap();
                c == acell || rcells.contains(&c)
            })
            .count();
        rand_recall.push(hit_r as f64 / NEAR as f64);
        if !out.is_empty() {
            let r = out
                .iter()
                .filter(|&&p| rcells.contains(&addr[p].unwrap()))
                .count();
            oob_rand.push(r as f64 / out.len() as f64);
        }

        // Calibration: same anchors, same neighbours, COARSE 2-level address.
        let coarse_anchor = acell >> 4;
        let cband = band_generic(coarse_anchor, 2);
        let hit_c = near
            .iter()
            .filter(|&&p| {
                let c = addr[p].unwrap() >> 4;
                c == coarse_anchor || cband.contains(&c)
            })
            .count();
        coarse_recall.push(hit_c as f64 / NEAR as f64);
    }
    let band_mean = band_recall.iter().sum::<f64>() / band_recall.len() as f64;
    let rand_mean = rand_recall.iter().sum::<f64>() / rand_recall.len() as f64;
    let coarse_mean = coarse_recall.iter().sum::<f64>() / coarse_recall.len() as f64;
    let oob_b = oob_band.iter().sum::<f64>() / oob_band.len() as f64;
    let oob_r = oob_rand.iter().sum::<f64>() / oob_rand.len() as f64;
    gate(
        "W3 spatial activation (can-fire AND can-stay-silent)",
        oob_b > oob_r * 2.0 && band_mean < 0.95 && coarse_mean >= 0.95,
        format!(
            "PRIMARY (out-of-cell neighbours — the actual sparse-adjacency claim): \
             band={oob_b:.3} vs random={oob_r:.3}, ratio {:.2}×, over {} cells (4.7 % of \
             the codebook).\n        SECONDARY (all {NEAR} nearest DISTINCT neighbours, \
             both arms crediting the home cell): band={band_mean:.3} vs random={rand_mean:.3} \
             (ratio {:.2}× — SATURATING, see note).\n        Upper guard band<0.95 is \
             CALIBRATED, not asserted: the same measurement against a coarse 2-level \
             address (6+1 of 16 cells) gives {coarse_mean:.3}, which the guard REJECTS.",
            oob_b / oob_r.max(1e-9),
            BAND,
            band_mean / rand_mean.max(1e-9),
        ),
    );

    // ── W4 does 4-ary see what the nibble cannot? ───────────────────────────
    // The nibble router's finest distinction inside a byte is "same top nibble"
    // (= shared levels ≥ 2 in 4-ary terms). 4-ary splits that population into
    // shared-2 and shared-3. If those differ, the sub-nibble decision is real.
    let sub_nibble_gap = means[2] - means[3];
    let nibble_blind: Vec<u32> = rungs[2].iter().chain(rungs[3].iter()).copied().collect();
    gate(
        "W4 sub-nibble structure is load-bearing",
        sub_nibble_gap > 0.5,
        format!(
            "inside one top nibble the nibble router sees ONE bucket \
             (mean {:.2}, n={}); 4-ary splits it into shared-2 {:.2} vs shared-3 {:.2} \
             — gap {sub_nibble_gap:.2} hops that 16-ary cannot address",
            mean(&nibble_blind),
            nibble_blind.len(),
            means[2],
            means[3],
        ),
    );

    // A worked example, so the numbers have a face. Show the band's NEAREST
    // residents by WordNet distance — an earlier draft printed the first six in
    // file order, which reads as a semantic result while being arbitrary.
    for probe_word in ["violin", "dog", "oak", "hammer"] {
        let Some(&anchor) = addressed.iter().find(|&&i| tree.lemma[i] == probe_word) else {
            continue;
        };
        let acell = addr[anchor].unwrap();
        let band = band_of(acell);
        let mut in_band: Vec<(u32, usize)> = addressed
            .iter()
            .filter(|&&p| p != anchor && band.contains(&addr[p].unwrap()))
            .map(|&p| (tree.path_distance(anchor, p), p))
            .filter(|&(d, _)| d != u32::MAX)
            .collect();
        in_band.sort_unstable();
        let nearest: Vec<String> = in_band
            .iter()
            .take(6)
            .map(|&(d, p)| format!("{}({d})", tree.lemma[p]))
            .collect();
        // The same budget spent anywhere else in the codebook, for contrast.
        let out_of_band_best = addressed
            .iter()
            .filter(|&&p| p != anchor && !band.contains(&addr[p].unwrap()))
            .filter(|&&p| addr[p] != Some(acell))
            .map(|&p| tree.path_distance(anchor, p))
            .filter(|&d| d != u32::MAX)
            .min()
            .unwrap_or(u32::MAX);
        println!(
            "\n  worked example: '{probe_word}' @ cell {acell:#04x} \
             ({:02b}|{:02b}|{:02b}|{:02b})\n    nearest in the 12-cell band (hops): {}\
             \n    best distance OUTSIDE the band: {out_of_band_best} hops",
            (acell >> 6) & 3,
            (acell >> 4) & 3,
            (acell >> 2) & 3,
            acell & 3,
            nearest.join(", "),
        );
        break;
    }

    let failures = FAILURES.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "\n{}",
        if failures == 0 {
            "ALL GATES GREEN".to_string()
        } else {
            format!("{failures} GATE(S) FAILED")
        }
    );
    if failures > 0 {
        std::process::exit(1);
    }
}

/// Correlation between (shared address levels) and (LCA depth) over sampled
/// pairs. `perm` optionally relabels cells — the falsifier arm.
fn prefix_vs_lca_corr(
    tree: &Tree,
    sample: &[usize],
    addr: &[Option<u8>],
    perm: Option<&[u8]>,
) -> f64 {
    let relabel = |c: u8| match perm {
        Some(p) => p[c as usize],
        None => c,
    };
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for w in sample.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a == b {
            continue;
        }
        let k = shared_levels(relabel(addr[a].unwrap()), relabel(addr[b].unwrap()));
        // DIFFERENT-CELL PAIRS ONLY. A label permutation is a bijection, so
        // same-cell pairs stay same-cell under any relabelling — including them
        // let the falsifier arm score +0.645 on pure cell identity, masking
        // whether the INTERMEDIATE prefix levels carry anything. The claim
        // under test is about levels 1..3, so level 4 is excluded from both arms.
        if k == LEVELS || shared_levels(addr[a].unwrap(), addr[b].unwrap()) == LEVELS {
            continue;
        }
        // LCA depth via path distance identity.
        let d = tree.path_distance(a, b);
        if d == u32::MAX {
            continue;
        }
        let lca_depth = (tree.depth[a] as i32 + tree.depth[b] as i32 - d as i32) / 2;
        xs.push(k as f64);
        ys.push(lca_depth as f64);
    }
    pearson(&xs, &ys)
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut num, mut dx, mut dy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        num += (a - mx) * (b - my);
        dx += (a - mx) * (a - mx);
        dy += (b - my) * (b - my);
    }
    if dx == 0.0 || dy == 0.0 {
        return 0.0;
    }
    num / (dx.sqrt() * dy.sqrt())
}
