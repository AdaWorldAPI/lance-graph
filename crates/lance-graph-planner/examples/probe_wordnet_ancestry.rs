//! PROBE-WORDNET-ANCESTRY — the falsifier `E-HYPERNYM-CLIMB-IS-A-CASCADE-TIER-DELTA-1`
//! specified and never ran: **does centroid-tree ancestry track hypernym ancestry?**
//!
//! The claim under test (operator: *"wordnet IS HHTL"*). The substrate has two
//! independent vertical structures that have never been compared on real data:
//!   - WordNet's hypernym DAG — explicit, symbolic, 117k+ noun `is_a` edges;
//!   - the hierarchical 16x16 codebook (`bgz17::Palette::build_hierarchical`,
//!     #823) whose `code>>4 == coarse` IS centroid ancestry, reusing
//!     `hhtl.rs::NiblePath`.
//! If a shared code PREFIX predicts a shared HYPERNYM, the HHTL address is
//! semantically grounded rather than distance-only. If it does not, the prefix
//! is a distance address that happens to be hierarchical — still useful, but
//! not "wordnet IS HHTL".
//!
//! **Why this probe is gated on the ceiling probe.** `Palette::build*` accepts
//! only `&[Base17]`, and `probe_base17_fold_ceiling` measured that the 17-dim
//! fold caps rho at ~0.27 vs cosine on exactly this input class — DIMENSIONALLY,
//! not fixably. Running ancestry through the shipped API alone would therefore
//! read an upstream cap as this claim's failure. So BOTH arms run:
//!   - `base17` — the SHIPPED path (what production would actually do), and
//!   - `fulldim` — the same 16x16 hierarchy built over FULL-dimension vectors
//!     via `ndarray::simd::kmeans` (no hand-rolled trainer).
//! The delta between the arms is itself the answer about whether Base17 blocks
//! this question too.
//!
//! **Metric.** For a lemma pair: `prefix_depth` in {0,1,2} (2 = same fine code,
//! 1 = same coarse nibble, 0 = neither) vs `wn_depth` = depth of the deepest
//! common hypernym ancestor over all sense pairs (0 = only the root, or none).
//! Report Spearman rho and the mean `wn_depth` per prefix level — the latter is
//! the readable form: if prefix 2 pairs are not deeper-related than prefix 0
//! pairs, there is no signal, whatever rho says.
//!
//! **Falsifiers (this probe can fail, and can fail in both directions):**
//!   - CAN-IT-FIRE: a SHUFFLED-code control must destroy the signal. If shuffled
//!     scores like the real codebook, the harness is measuring nothing.
//!   - CAN-IT-STAY-SILENT: `wn_depth` must have real spread (not all pairs at
//!     the same ancestor), and prefix levels must be non-degenerate (each of
//!     0/1/2 must actually occur). A prefix that is 2 for every pair carries as
//!     much information as one that is never 2.
//!   - FLAT NULL: the flat-256 codebook's `code>>4` is meaningless by
//!     construction (#823 measured prefix purity 0.1602 vs hierarchical 1.0000),
//!     so it is the null the hierarchical arm must beat to claim ancestry.
//!
//! > **ENCODER-MISMATCH CAVEAT (operator, 2026-07-27) — read before citing the
//! > Base17 arms.** bgz17/Base17 encodes **PHASE** (cyclic position over
//! > golden-ordered residue classes); **spatial DIRECTION** is `helix`'s job
//! > (`fisher_z.rs`: `hyperbolic_depth = 2*arctanh(s)` — the "Fisher-2z" form,
//! > geometry keeping the arc-length factor 2 that statistics drops). Hypernym
//! > ancestry is a direction/semantic question, so the two `*-base17` arms below
//! > compare a PHASE prefix against a TAXONOMY and are ill-posed by
//! > construction. They are retained as the SHIPPED-path baseline, not as a
//! > verdict on Base17. This is the same category error `bf16-hhtl-terrain.md`
//! > correction 6 already names ("the codec should never decode to f32 to score
//! > cosine"); a helix-carried arm is the honest direction measurement and is
//! > NOT yet built. The full-dimension arms do not route through Base17 and are
//! > unaffected.
//!
//! Real bytes only (Rule 23): real WordNet 3.1 WNDB rails + real MiniLM
//! embeddings. Deterministic SplitMix64, seed 0x9E3779B97F4A7C15.
//!
//! ```text
//! # WordNet rail (gitignored build product of the committed generator):
//! curl -sSL -o /tmp/wn31.tar.gz https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz
//! #   sha256 3f7d8be8ef6ecc7167d39b10d66954ec734280b5bdcd57f7d9eafe429d11c22a
//! tar xzf /tmp/wn31.tar.gz -C /tmp
//! WNDB_DIR=/tmp/dict python3 \
//!   crates/lance-graph-planner/examples/data/wordnet/build_wordnet_rail.py
//! # Embeddings + vocab (all-MiniLM-L6-v2):
//! #   model.safetensors sha256 53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db
//! #   vocab.txt         sha256 07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3
//! cargo run --release -p lance-graph-planner --example probe_wordnet_ancestry -- \
//!   <emb.f32> <vocab.txt> <wordnet31_isa_v2.tsv>
//! ```
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use bgz17::base17::Base17;
use bgz17::palette::Palette;
use bgz17::{BASE_DIM, FP_SCALE, GOLDEN_STEP};
use ndarray::simd::kmeans;
use std::collections::{HashMap, HashSet};

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const N_LEMMAS: usize = 3000;
const N_PAIRS: usize = 40_000;
const KMEANS_ITERS: usize = 15;

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

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        let (dx, dy) = (a - mx, b - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    sxy / (sxx.sqrt() * syy.sqrt()).max(1e-300)
}
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).expect("finite").then(a.cmp(&b)));
    let mut r = vec![0f64; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1 + j) as f64) / 2.0;
        for &k in &idx[i..j] {
            r[k] = avg;
        }
        i = j;
    }
    r
}
fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&ranks(x), &ranks(y))
}

// ── WordNet noun hypernym DAG ────────────────────────────────────────────────

/// Parsed rail: lemma -> its noun synset offsets; synset -> hypernym parents.
struct WordNet {
    senses: HashMap<String, Vec<u32>>,
    parents: HashMap<u32, Vec<u32>>,
}

impl WordNet {
    fn load(path: &str) -> Self {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {path}: {e}\nBuild it first (see the header)."));
        let mut senses: HashMap<String, Vec<u32>> = HashMap::new();
        let mut parents: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut data_rows = 0usize;
        for line in text.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let c: Vec<&str> = line.split('\t').collect();
            // word, pos, sense_num, synset_offset, kind, hypernym_word, hypernym_offset
            assert_eq!(
                c.len(),
                7,
                "rail schema changed: expected 7 columns, got {} in {line:?}",
                c.len()
            );
            data_rows += 1;
            if c[1] != "n" {
                continue; // nouns only: the `@` taxonomy (matches rail scope)
            }
            let (Ok(off), Ok(hyp)) = (c[3].parse::<u32>(), c[6].parse::<u32>()) else {
                continue;
            };
            senses.entry(c[0].to_string()).or_default().push(off);
            parents.entry(off).or_default().push(hyp);
        }
        for v in senses.values_mut() {
            v.sort_unstable();
            v.dedup();
        }
        for v in parents.values_mut() {
            v.sort_unstable();
            v.dedup();
        }
        println!(
            "wordnet: {data_rows} rail rows -> {} noun lemmas, {} synsets with parents",
            senses.len(),
            parents.len()
        );
        WordNet { senses, parents }
    }

    /// All ancestors of `s` (excluding `s`), with the shortest hop count to each.
    fn ancestors(&self, s: u32) -> HashMap<u32, u32> {
        let mut out: HashMap<u32, u32> = HashMap::new();
        let mut frontier = vec![s];
        let mut depth = 0u32;
        let mut seen: HashSet<u32> = HashSet::from([s]);
        while !frontier.is_empty() && depth < 32 {
            depth += 1;
            let mut next = Vec::new();
            for n in frontier.drain(..) {
                for &p in self.parents.get(&n).map_or(&[][..], |v| v.as_slice()) {
                    if seen.insert(p) {
                        out.insert(p, depth);
                        next.push(p);
                    }
                }
            }
            frontier = next;
        }
        out
    }

    /// Depth of a synset = hops to the deepest root reachable (its own height
    /// above the taxonomy top). Used to score HOW SPECIFIC a shared ancestor is.
    fn root_distance(&self, s: u32) -> u32 {
        let anc = self.ancestors(s);
        anc.values().copied().max().unwrap_or(0)
    }

    /// Shared-ancestor specificity for two lemmas: over all sense pairs, the
    /// root-distance of the deepest common ancestor. 0 = nothing in common
    /// beyond the top of the taxonomy.
    fn shared_depth(&self, a: &str, b: &str, cache: &mut HashMap<u32, u32>) -> Option<f64> {
        let (sa, sb) = (self.senses.get(a)?, self.senses.get(b)?);
        let mut best = 0u32;
        let mut any = false;
        for &x in sa {
            let ax = self.ancestors(x);
            for &y in sb {
                if x == y {
                    continue;
                }
                let ay = self.ancestors(y);
                for k in ax.keys() {
                    if ay.contains_key(k) {
                        any = true;
                        let d = *cache.entry(*k).or_insert_with(|| self.root_distance(*k));
                        best = best.max(d);
                    }
                }
            }
        }
        if any {
            Some(f64::from(best))
        } else {
            Some(0.0)
        }
    }
}

// ── encoders ─────────────────────────────────────────────────────────────────

fn base17_from_f32(v: &[f32]) -> Base17 {
    let mut pos = [0usize; BASE_DIM];
    for (i, p) in pos.iter_mut().enumerate() {
        *p = (i * GOLDEN_STEP) % BASE_DIM;
    }
    let n = v.len();
    let mut sum = [0f64; BASE_DIM];
    let mut cnt = [0u32; BASE_DIM];
    for octave in 0..n.div_ceil(BASE_DIM) {
        for (bi, &p) in pos.iter().enumerate() {
            let d = octave * BASE_DIM + p;
            if d < n {
                sum[bi] += f64::from(v[d]);
                cnt[bi] += 1;
            }
        }
    }
    let mut dims = [0i16; BASE_DIM];
    for (i, slot) in dims.iter_mut().enumerate() {
        if cnt[i] > 0 {
            *slot = ((sum[i] / f64::from(cnt[i])) * FP_SCALE)
                .round()
                .clamp(-32768.0, 32767.0) as i16;
        }
    }
    Base17 { dims }
}

fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn nearest(v: &[f32], cents: &[Vec<f32>]) -> usize {
    let mut best = (f32::INFINITY, 0usize);
    for (i, c) in cents.iter().enumerate() {
        let d = sq_l2(v, c);
        if d < best.0 {
            best = (d, i);
        }
    }
    best.1
}

/// A cascade codebook over FULL-dimension vectors: `levels` levels of `fan`-ary
/// k-means, so `fan.pow(levels) == 256` and the code's leading
/// `log2(fan)`-bit groups ARE the centroid's ancestry (coarse -> fine).
///
/// `fan=16, levels=2` reproduces the SHIPPED `Palette::build_hierarchical`
/// shape (one ancestry step, `code>>4`). `fan=4, levels=4` is the canon
/// `256 = 4^4` cascade (three ancestry steps, `code>>6 / >>4 / >>2`), where each
/// level is a 2-bit Morton refinement — one bit per axis, alternating.
/// Which shape actually predicts hypernym ancestry is the question, not an
/// assumption: both are measured.
///
/// Trainer is `ndarray::simd::kmeans` at every level (no hand-rolled Lloyd).
struct Cascade {
    /// `nodes[level]` = the centroid set for each parent path at that level.
    nodes: Vec<Vec<Vec<Vec<f32>>>>,
    fan: usize,
    levels: usize,
}

impl Cascade {
    fn build(rows: &[Vec<f32>], dim: usize, fan: usize, levels: usize, iters: usize) -> Self {
        assert_eq!(
            fan.pow(levels as u32),
            256,
            "cascade must carve exactly 256 leaves; {fan}^{levels} does not"
        );
        // Groups at the current level, in path order.
        let mut groups: Vec<Vec<Vec<f32>>> = vec![rows.to_vec()];
        let mut nodes = Vec::with_capacity(levels);
        for _ in 0..levels {
            let mut level_cents = Vec::with_capacity(groups.len());
            let mut next: Vec<Vec<Vec<f32>>> = Vec::with_capacity(groups.len() * fan);
            for g in &groups {
                // An under-populated node pads with its own mean so the
                // `code >> k` ancestry invariant holds for every leaf.
                let cents = if g.len() >= fan {
                    kmeans(g, fan, dim, iters)
                } else {
                    let mut m = vec![0f32; dim];
                    for r in g {
                        for (a, b) in m.iter_mut().zip(r) {
                            *a += *b;
                        }
                    }
                    if !g.is_empty() {
                        for a in &mut m {
                            *a /= g.len() as f32;
                        }
                    }
                    vec![m; fan]
                };
                let mut buckets: Vec<Vec<Vec<f32>>> = vec![Vec::new(); fan];
                for r in g {
                    buckets[nearest(r, &cents)].push(r.clone());
                }
                level_cents.push(cents);
                next.extend(buckets);
            }
            nodes.push(level_cents);
            groups = next;
        }
        Cascade { nodes, fan, levels }
    }

    /// Encode to a byte whose leading `log2(fan)`-bit groups are the path.
    fn encode(&self, v: &[f32]) -> u8 {
        let bits = self.fan.trailing_zeros() as usize;
        let mut path = 0usize;
        let mut code = 0u8;
        for l in 0..self.levels {
            let c = nearest(v, &self.nodes[l][path]);
            code = (code << bits) | c as u8;
            path = path * self.fan + c;
        }
        code
    }

    /// Shared prefix depth in LEVELS (0..=levels).
    fn prefix_depth(&self, a: u8, b: u8) -> f64 {
        let bits = self.fan.trailing_zeros() as usize;
        let mut d = 0;
        for l in 1..=self.levels {
            let shift = (self.levels - l) * bits;
            if a >> shift == b >> shift {
                d = l;
            } else {
                break;
            }
        }
        d as f64
    }
}

/// Shipped 16x16 shape: 2 = same code, 1 = same coarse nibble, 0 = neither.
#[inline]
fn prefix_depth_16x16(a: u8, b: u8) -> f64 {
    if a == b {
        2.0
    } else if a >> 4 == b >> 4 {
        1.0
    } else {
        0.0
    }
}

fn report(name: &str, prefix: &[f64], wn: &[f64], max_level: usize) -> f64 {
    let rho = spearman(prefix, wn);
    let mut sum = vec![0f64; max_level + 1];
    let mut cnt = vec![0usize; max_level + 1];
    for (p, w) in prefix.iter().zip(wn) {
        let i = *p as usize;
        sum[i] += w;
        cnt[i] += 1;
    }
    let cells: Vec<String> = (0..=max_level)
        .map(|i| {
            if cnt[i] == 0 {
                format!("{i}:-")
            } else {
                format!("{i}:{:.2}(n={})", sum[i] / cnt[i] as f64, cnt[i])
            }
        })
        .collect();
    println!(
        "  {name:<24} rho {rho:>7.4}   mean wn_depth by prefix  {}",
        cells.join("  ")
    );
    rho
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        eprintln!(
            "usage: probe_wordnet_ancestry <emb.f32> <vocab.txt> <wordnet31_isa_v2.tsv>\n\n\
             Requires REAL bytes (Rule 23). See the module header for the exact\n\
             fetch + build commands and the sha256 of every input."
        );
        std::process::exit(2);
    }
    let (emb_path, vocab_path, rail_path) = (&a[0], &a[1], &a[2]);

    let buf = std::fs::read(emb_path).unwrap_or_else(|e| panic!("read {emb_path}: {e}"));
    assert!(buf.len() > 8, "{emb_path}: too short for a header");
    let n = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    assert_eq!(
        buf.len(),
        8 + n * dim * 4,
        "{emb_path}: header/size mismatch"
    );
    let all: Vec<f32> = buf[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let vocab: Vec<String> = std::fs::read_to_string(vocab_path)
        .unwrap_or_else(|e| panic!("read {vocab_path}: {e}"))
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(
        vocab.len(),
        n,
        "vocab has {} entries but the embedding matrix has {n} rows - they must \
         be the same tokenizer",
        vocab.len()
    );
    println!("embeddings: {n} x {dim}; vocab: {} tokens", vocab.len());

    let wn = WordNet::load(rail_path);

    let mut lemmas: Vec<(usize, String)> = vocab
        .iter()
        .enumerate()
        .filter(|(_, w)| {
            w.len() > 2 && w.chars().all(|c| c.is_ascii_lowercase()) && wn.senses.contains_key(*w)
        })
        .map(|(i, w)| (i, w.clone()))
        .collect();
    println!("usable lemmas (vocab AND wordnet noun): {}", lemmas.len());
    assert!(
        lemmas.len() >= N_LEMMAS,
        "only {} usable lemmas, need >= {N_LEMMAS}",
        lemmas.len()
    );
    lemmas.truncate(N_LEMMAS);

    let rows: Vec<Vec<f32>> = lemmas
        .iter()
        .map(|(i, _)| all[i * dim..(i + 1) * dim].to_vec())
        .collect();

    // ── codebooks: the SHIPPED shape, the canon 4^4 shape, and the nulls ────
    let b17: Vec<Base17> = rows.iter().map(|r| base17_from_f32(r)).collect();
    let hier_b17 = Palette::build_hierarchical(&b17, KMEANS_ITERS);
    let flat_b17 = Palette::build(&b17, 256, KMEANS_ITERS);
    let c16 = Cascade::build(&rows, dim, 16, 2, KMEANS_ITERS);
    let c4 = Cascade::build(&rows, dim, 4, 4, KMEANS_ITERS);

    let code_hier_b17: Vec<u8> = b17.iter().map(|p| hier_b17.leaves.nearest(p)).collect();
    let code_flat_b17: Vec<u8> = b17.iter().map(|p| flat_b17.nearest(p)).collect();
    let code_16: Vec<u8> = rows.iter().map(|r| c16.encode(r)).collect();
    let code_4: Vec<u8> = rows.iter().map(|r| c4.encode(r)).collect();

    // SHUFFLED control: same code MULTISET, assignment destroyed.
    let mut rng = SplitMix64(SEED);
    let mut code_shuf = code_4.clone();
    for i in (1..code_shuf.len()).rev() {
        code_shuf.swap(i, rng.below(i + 1));
    }

    // ── pair sample + WordNet ground truth ──────────────────────────────────
    let mut cache: HashMap<u32, u32> = HashMap::new();
    let mut wn_depth = Vec::with_capacity(N_PAIRS);
    let mut idx_pairs = Vec::with_capacity(N_PAIRS);
    while idx_pairs.len() < N_PAIRS {
        let i = rng.below(lemmas.len());
        let j = rng.below(lemmas.len());
        if i == j {
            continue;
        }
        let Some(d) = wn.shared_depth(&lemmas[i].1, &lemmas[j].1, &mut cache) else {
            continue;
        };
        idx_pairs.push((i, j));
        wn_depth.push(d);
    }

    // CAN-IT-STAY-SILENT: the ground truth must discriminate. If every pair
    // shares the same ancestor depth, every rho below is meaningless.
    let distinct: HashSet<u64> = wn_depth.iter().map(|d| d.to_bits()).collect();
    println!(
        "pairs: {} | wn_depth distinct values: {} | mean {:.3}",
        wn_depth.len(),
        distinct.len(),
        wn_depth.iter().sum::<f64>() / wn_depth.len() as f64
    );
    assert!(
        distinct.len() >= 3,
        "wn_depth takes only {} distinct values - no ground-truth spread",
        distinct.len()
    );

    let map = |f: &dyn Fn(u8, u8) -> f64, codes: &[u8]| -> Vec<f64> {
        idx_pairs
            .iter()
            .map(|&(i, j)| f(codes[i], codes[j]))
            .collect()
    };
    let p_hier_b17 = map(&prefix_depth_16x16, &code_hier_b17);
    let p_flat_b17 = map(&prefix_depth_16x16, &code_flat_b17);
    let p_16 = map(&|a, b| c16.prefix_depth(a, b), &code_16);
    let p_4 = map(&|a, b| c4.prefix_depth(a, b), &code_4);
    let p_shuf = map(&|a, b| c4.prefix_depth(a, b), &code_shuf);

    // CAN-IT-FIRE: prefix depth must be non-degenerate. A prefix that is always
    // 0 (or always maximal) carries exactly no information.
    for (name, p) in [
        ("16x16-base17", &p_hier_b17),
        ("16x16-fulldim", &p_16),
        ("4^4-fulldim", &p_4),
    ] {
        let hits = p.iter().filter(|v| **v > 0.0).count();
        assert!(
            hits > 0 && hits < p.len(),
            "{name}: prefix depth degenerate ({hits}/{} pairs share a prefix)",
            p.len()
        );
    }

    println!("\nSpearman rho( centroid prefix depth , wordnet shared-ancestor depth ):");
    let rho_hier_b17 = report("16x16-base17 SHIPPED", &p_hier_b17, &wn_depth, 2);
    let rho_flat_b17 = report("flat-256-base17 NULL", &p_flat_b17, &wn_depth, 2);
    let rho_16 = report("16x16-fulldim", &p_16, &wn_depth, 2);
    let rho_4 = report("4^4-fulldim CANON", &p_4, &wn_depth, 4);
    let rho_shuf = report("SHUFFLED control", &p_shuf, &wn_depth, 4);

    println!("\n--- verdict ---");
    println!("  shuffled control          rho {rho_shuf:+.4}  (must sit at ~0)");
    println!(
        "  4^4-fulldim over control      {:+.4}",
        rho_4 - rho_shuf.abs()
    );
    println!(
        "  16x16-fulldim over control    {:+.4}",
        rho_16 - rho_shuf.abs()
    );
    println!(
        "  SHIPPED over its flat null    {:+.4}",
        rho_hier_b17 - rho_flat_b17
    );
    println!("  4^4 over 16x16 (same data)    {:+.4}", rho_4 - rho_16);
    println!(
        "  fulldim over base17 (16x16)   {:+.4}",
        rho_16 - rho_hier_b17
    );

    let best = rho_4.max(rho_16);
    if best - rho_shuf.abs() > 0.05 {
        println!(
            "\n  SIGNAL: centroid ancestry tracks hypernym ancestry above the\n  \
             shuffled control => the HHTL prefix carries semantic grounding,\n  \
             not distance only. Strength is the margin, never the raw rho."
        );
    } else {
        println!(
            "\n  NO SIGNAL at this resolution: centroid prefix does not predict\n  \
             hypernym ancestry beyond the shuffled control. 'wordnet IS HHTL' is\n  \
             NOT supported by this measurement - the prefix is a DISTANCE address\n  \
             that happens to be hierarchical."
        );
    }
    println!(
        "\n  Shape (4^4 vs 16x16, same full-dim data, same 256 leaves) and carrier\n  \
         (full-dim vs Base17, same 16x16 shape) are reported separately, so a\n  \
         null result names WHICH factor is responsible instead of blaming both."
    );
}
