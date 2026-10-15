//! D-HOLO H14: how a dirty row of one (classid x thought x 64k tile) is held.
//!
//! Pre-registered in `.claude/plans/d-holo-address-indexed-algebra-v1.md`
//! ("Next falsifier: H14", with the two 2026-10-15 amendments). Arms:
//! - A, full row: per dirty (thought, row) the 512-byte row image.
//! - B, connected delta: per dirty (thought, row) one record: row key, lane
//!   mask, the changed 16-byte lanes packed in lane order (`Table[i][a][b]`).
//! - C, inverted fold: per (thought, lane) one table indexed directly by the
//!   row address (`Lane_l[i]`) plus a presence mask. Lanes of row i are read
//!   at position i of each table; counting queries fold presence masks.
//!
//! The session level (which thoughts wrote) is one u64 mask, shared by all
//! arms and not part of any arm's lane storage.
//!
//! Run (release, the tier is printed): `cargo run --release --example
//! holo_lane_storage_probe`. Tests: `cargo test --release --example
//! holo_lane_storage_probe`.
#![allow(dead_code)]

use std::hint::black_box;
use std::time::Instant;

use ndarray::simd::mask_or;

const N: usize = 65_536;
const WORDS: usize = N / 64;
const ROW: usize = 512;
const LANE: usize = 16;
const LANES: usize = 32;
const NONE: u32 = u32::MAX;
const APERTURE: usize = 4_096;
/// The lane set of F1.
const F1_LANES: [usize; 4] = [1, 8, 15, 22];

type Row = [u8; ROW];
type Val = [u8; LANE];

fn tier() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "v4 (avx512f)"
    } else if cfg!(target_feature = "avx2") {
        "v3 (avx2): timings not reportable"
    } else {
        "below v3: timings not reportable"
    }
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn lane_bytes(row: &Row, l: usize) -> &Val {
    row[l * LANE..(l + 1) * LANE].try_into().unwrap()
}

fn base() -> Vec<Row> {
    (0..N)
        .map(|r| {
            let mut row = [0u8; ROW];
            for (w, c) in row.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                *c = mix((r * 64 + w) as u64).to_le_bytes();
            }
            row
        })
        .collect()
}

/// One thought's writes: dirty rows ascending, each with its lane mask and
/// the new values in lane order.
struct Writes {
    rows: Vec<u32>,
    masks: Vec<u32>,
    vals: Vec<Val>,
}

/// The j-th lane a dirty row writes: distinct for j < 31, never lane 0 (key).
fn lane_of(t: usize, r: usize, j: usize) -> usize {
    1 + (r * 7 + t * 13 + j * 5) % (LANES - 1)
}

fn writes(base: &[Row], t: usize, per_10k: u64, k: usize) -> Writes {
    let mut w = Writes {
        rows: Vec::new(),
        masks: Vec::new(),
        vals: Vec::new(),
    };
    for (r, b) in base.iter().enumerate() {
        if mix(((t as u64) << 32) ^ r as u64 ^ 0xA5A5) % 10_000 >= per_10k {
            continue;
        }
        let mut mask = 0u32;
        for j in 0..k {
            mask |= 1 << lane_of(t, r, j);
        }
        w.rows.push(r as u32);
        w.masks.push(mask);
        for l in BitIter(mask) {
            let x = mix(((t as u64) << 40) ^ ((r as u64) << 8) ^ l as u64);
            let mut v = [0u8; LANE];
            v[..8].copy_from_slice(&x.to_le_bytes());
            v[8..].copy_from_slice(&x.rotate_left(17).to_le_bytes());
            if &v == lane_bytes(b, l) {
                v[0] ^= 1;
            }
            w.vals.push(v);
        }
    }
    w
}

struct BitIter(u32);
impl Iterator for BitIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let t = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1;
        Some(t)
    }
}

/// Which thoughts of a session wrote anything: one u64.
fn session_mask(all: &[Writes]) -> u64 {
    all.iter()
        .enumerate()
        .filter(|(_, w)| !w.rows.is_empty())
        .fold(0, |m, (t, _)| m | 1 << t)
}

// ---------------------------------------------------------------- A

struct ArmA {
    index: Vec<u32>,
    images: Vec<Row>,
}

impl ArmA {
    fn seal(base: &[Row], w: &Writes) -> Self {
        let mut a = ArmA {
            index: vec![NONE; N],
            images: Vec::with_capacity(w.rows.len()),
        };
        let mut v = 0;
        for (&r, &m) in w.rows.iter().zip(&w.masks) {
            let mut img = base[r as usize];
            for l in BitIter(m) {
                img[l * LANE..(l + 1) * LANE].copy_from_slice(&w.vals[v]);
                v += 1;
            }
            a.index[r as usize] = a.images.len() as u32;
            a.images.push(img);
        }
        a
    }
    fn bytes(&self) -> usize {
        self.images.len() * (4 + ROW)
    }
    fn read(&self, base: &[Row], r: usize) -> Row {
        match self.index[r] {
            NONE => base[r],
            i => self.images[i as usize],
        }
    }
    /// A holds no lane mask: a changed lane is found by comparing with base.
    fn changed(&self, base: &[Row], r: usize, l: usize) -> bool {
        match self.index[r] {
            NONE => false,
            i => lane_bytes(&self.images[i as usize], l) != lane_bytes(&base[r], l),
        }
    }
}

// ---------------------------------------------------------------- B

struct ArmB {
    index: Vec<u32>,
    rows: Vec<u32>,
    masks: Vec<u32>,
    starts: Vec<u32>,
    words: Vec<Val>,
}

impl ArmB {
    fn seal(w: &Writes) -> Self {
        let mut b = ArmB {
            index: vec![NONE; N],
            rows: Vec::with_capacity(w.rows.len()),
            masks: Vec::with_capacity(w.rows.len()),
            starts: Vec::with_capacity(w.rows.len()),
            words: Vec::with_capacity(w.vals.len()),
        };
        let mut v = 0;
        for (&r, &m) in w.rows.iter().zip(&w.masks) {
            let n = m.count_ones() as usize;
            b.index[r as usize] = b.rows.len() as u32;
            b.rows.push(r);
            b.masks.push(m);
            b.starts.push(b.words.len() as u32);
            b.words.extend_from_slice(&w.vals[v..v + n]);
            v += n;
        }
        b
    }
    fn bytes(&self) -> usize {
        self.rows.len() * 12 + self.words.len() * LANE
    }
    fn read(&self, base: &[Row], r: usize) -> Row {
        let mut out = base[r];
        let i = self.index[r];
        if i != NONE {
            let i = i as usize;
            let w0 = self.starts[i] as usize;
            for (w, l) in BitIter(self.masks[i]).enumerate() {
                out[l * LANE..(l + 1) * LANE].copy_from_slice(&self.words[w0 + w]);
            }
        }
        out
    }
    fn mask(&self, r: usize) -> u32 {
        match self.index[r] {
            NONE => 0,
            i => self.masks[i as usize],
        }
    }
}

// ---------------------------------------------------------------- C

struct Plane {
    present: Vec<u64>,
    /// Positional: the value of row i is `values[i]`.
    values: Vec<Val>,
    count: usize,
}

struct ArmC {
    planes: Vec<Option<Plane>>,
    touched: Vec<u8>,
}

#[cfg(test)]
thread_local! {
    static VALUE_READS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static MERGES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

impl ArmC {
    fn seal(w: &Writes) -> Self {
        let mut c = ArmC {
            planes: (0..LANES).map(|_| None).collect(),
            touched: Vec::new(),
        };
        let mut v = 0;
        for (&r, &m) in w.rows.iter().zip(&w.masks) {
            for l in BitIter(m) {
                let p = c.planes[l].get_or_insert_with(|| Plane {
                    present: vec![0; WORDS],
                    values: vec![[0; LANE]; N],
                    count: 0,
                });
                p.present[r as usize / 64] |= 1 << (r % 64);
                p.values[r as usize] = w.vals[v];
                p.count += 1;
                v += 1;
            }
        }
        c.touched = (0..LANES as u8)
            .filter(|&l| c.planes[l as usize].is_some())
            .collect();
        c
    }
    /// Bytes handed to the writer: per touched lane the dirty positions
    /// (as a bitmap or a u32 list, whichever is smaller) and their values.
    fn bytes(&self) -> usize {
        self.planes
            .iter()
            .flatten()
            .map(|p| (WORDS * 8).min(4 * p.count) + LANE * p.count)
            .sum()
    }
    fn resident(&self) -> usize {
        self.touched.len() * (WORDS * 8 + N * LANE)
    }
    fn value(&self, l: usize, r: usize) -> Option<&Val> {
        let p = self.planes[l].as_ref()?;
        if p.present[r / 64] >> (r % 64) & 1 == 0 {
            return None;
        }
        #[cfg(test)]
        VALUE_READS.with(|c| c.set(c.get() + 1));
        Some(&p.values[r])
    }
    /// Merge-on-read at one address: the base row overlaid with position r
    /// of every touched lane table.
    fn read(&self, base: &[Row], r: usize) -> Row {
        #[cfg(test)]
        MERGES.with(|c| c.set(c.get() + 1));
        let mut out = base[r];
        for &l in &self.touched {
            if let Some(v) = self.value(l as usize, r) {
                out[l as usize * LANE..(l as usize + 1) * LANE].copy_from_slice(v);
            }
        }
        out
    }
    /// The presence words of `lanes` over an aperture, ORed: no row is built.
    fn fold_lanes(&self, lanes: &[usize], a0: usize, acc: &mut [u64]) {
        let (w0, w1) = (a0 / 64, (a0 + APERTURE) / 64);
        acc.fill(0);
        let mut tmp = vec![0u64; w1 - w0];
        for &l in lanes {
            if let Some(p) = &self.planes[l] {
                mask_or(acc, &p.present[w0..w1], &mut tmp);
                acc.copy_from_slice(&tmp);
            }
        }
    }
}

fn popcount(words: &[u64]) -> u64 {
    words.iter().map(|w| u64::from(w.count_ones())).sum()
}

// ---------------------------------------------------------------- queries

fn lmask(lanes: &[usize]) -> u32 {
    lanes.iter().fold(0, |m, &l| m | 1 << l)
}

fn f1_a(a: &ArmA, base: &[Row], a0: usize) -> u64 {
    (a0..a0 + APERTURE)
        .filter(|&r| F1_LANES.iter().any(|&l| a.changed(base, r, l)))
        .count() as u64
}
fn f1_b(b: &ArmB, a0: usize) -> u64 {
    let m = lmask(&F1_LANES);
    (a0..a0 + APERTURE).filter(|&r| b.mask(r) & m != 0).count() as u64
}
fn f1_c(c: &ArmC, a0: usize, acc: &mut [u64]) -> u64 {
    c.fold_lanes(&F1_LANES, a0, acc);
    popcount(acc)
}
fn f2_a(arms: &[ArmA], base: &[Row], l: usize, a0: usize) -> u64 {
    (a0..a0 + APERTURE)
        .filter(|&r| arms.iter().any(|a| a.changed(base, r, l)))
        .count() as u64
}
fn f2_b(arms: &[ArmB], l: usize, a0: usize) -> u64 {
    (a0..a0 + APERTURE)
        .filter(|&r| arms.iter().any(|b| b.mask(r) >> l & 1 != 0))
        .count() as u64
}
fn f2_c(arms: &[ArmC], l: usize, a0: usize, acc: &mut [u64], tmp: &mut [u64]) -> u64 {
    let (w0, w1) = (a0 / 64, (a0 + APERTURE) / 64);
    acc.fill(0);
    for c in arms {
        if let Some(p) = &c.planes[l] {
            mask_or(acc, &p.present[w0..w1], tmp);
            acc.copy_from_slice(tmp);
        }
    }
    popcount(acc)
}
/// F3 on C: fold first, then merge-on-read only the rows the fold names.
fn f3_c(c: &ArmC, base: &[Row], a0: usize, acc: &mut [u64], out: &mut Vec<Row>) {
    c.fold_lanes(&F1_LANES, a0, acc);
    out.clear();
    for (wi, &w) in acc.iter().enumerate() {
        for b in BitIter64(w) {
            out.push(c.read(base, a0 + wi * 64 + b));
        }
    }
}
fn f3_b(b: &ArmB, base: &[Row], a0: usize, out: &mut Vec<Row>) {
    let m = lmask(&F1_LANES);
    out.clear();
    for r in a0..a0 + APERTURE {
        if b.mask(r) & m != 0 {
            out.push(b.read(base, r));
        }
    }
}
fn f3_a(a: &ArmA, base: &[Row], a0: usize, out: &mut Vec<Row>) {
    out.clear();
    for r in a0..a0 + APERTURE {
        if F1_LANES.iter().any(|&l| a.changed(base, r, l)) {
            out.push(a.read(base, r));
        }
    }
}

struct BitIter64(u64);
impl Iterator for BitIter64 {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let t = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1;
        Some(t)
    }
}

// ---------------------------------------------------------------- timing

/// Median of 11 samples; each sample is the mean over at least 3 calls and
/// 2 ms. Returns ns per call.
fn time<F: FnMut()>(mut f: F) -> f64 {
    f();
    let mut v: Vec<f64> = (0..11)
        .map(|_| {
            let t = Instant::now();
            let mut n = 0u32;
            while n < 3 || t.elapsed().as_micros() < 2_000 {
                f();
                n += 1;
            }
            t.elapsed().as_nanos() as f64 / f64::from(n)
        })
        .collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v[5]
}

fn main() {
    println!("D-HOLO H14: full row (A) vs connected delta (B) vs inverted fold (C)");
    println!("tier: {}", tier());
    println!("rows per tile {N}; aperture {APERTURE}; timings ns, median of 11\n");
    let base = base();
    println!(
        "{:>3} {:>5} {:>3} | {:>9} {:>9} {:>9} {:>10} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} | {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} | {:>6} {:>6} {:>6}",
        "T", "dirty", "k", "A KiB", "B KiB", "C KiB", "C res KiB",
        "sealA", "sealB", "sealC", "readA", "readB", "readC", "sessA", "sessB", "sessC",
        "F1 A", "F1 B", "F1 C", "F2 A", "F2 B", "F2 C", "F3 A", "F3 B", "F3 C"
    );
    let mut b_rule = true;
    let mut c_rule = true;
    let mut c_fold_wins = true;
    let mut c_read_ok = true;
    let mut acc = vec![0u64; APERTURE / 64];
    let mut tmp = vec![0u64; APERTURE / 64];
    let mut out: Vec<Row> = Vec::with_capacity(APERTURE);
    for t_n in [1usize, 8, 64] {
        for (label, per_10k) in [("1%", 100u64), ("10%", 1_000), ("100%", 10_000)] {
            for k in [1usize, 4, 16, 31] {
                let ws: Vec<Writes> = (0..t_n).map(|t| writes(&base, t, per_10k, k)).collect();
                let dirty: usize = ws.iter().map(|w| w.rows.len()).sum();
                let session = session_mask(&ws);
                let a: Vec<ArmA> = ws.iter().map(|w| ArmA::seal(&base, w)).collect();
                let b: Vec<ArmB> = ws.iter().map(ArmB::seal).collect();
                let c: Vec<ArmC> = ws.iter().map(ArmC::seal).collect();
                let (ba, bb): (usize, usize) = (
                    a.iter().map(ArmA::bytes).sum(),
                    b.iter().map(ArmB::bytes).sum(),
                );
                let bc: usize = c.iter().map(ArmC::bytes).sum();
                let rc: usize = c.iter().map(ArmC::resident).sum();

                // Seal, ns per dirty (thought, row). Timed on thought 0 only
                // and scaled by its dirty count, so a 64-thought cell does
                // not rebuild gigabytes eleven times.
                let d0 = ws[0].rows.len().max(1) as f64;
                let sa = time(|| { black_box(ArmA::seal(&base, black_box(&ws[0]))); }) / d0;
                let sb = time(|| { black_box(ArmB::seal(black_box(&ws[0]))); }) / d0;
                let sc = time(|| { black_box(ArmC::seal(black_box(&ws[0]))); }) / d0;

                // Full-row read: 65,536 random (thought, row) pairs.
                let probe: Vec<(u16, u32)> = (0..N as u64)
                    .map(|i| {
                        let x = mix(i ^ 0x5151);
                        ((x % t_n as u64) as u16, ((x >> 32) % N as u64) as u32)
                    })
                    .collect();
                let per = probe.len() as f64;
                let ra = time(|| { for &(t, r) in &probe { black_box(a[t as usize].read(&base, r as usize)); } }) / per;
                let rb = time(|| { for &(t, r) in &probe { black_box(b[t as usize].read(&base, r as usize)); } }) / per;
                let rcn = time(|| { for &(t, r) in &probe { black_box(c[t as usize].read(&base, r as usize)); } }) / per;

                // Session read: every thought of the session over one aperture.
                let a0 = 8 * APERTURE;
                let thoughts: Vec<usize> = (0..t_n).filter(|t| session >> t & 1 == 1).collect();
                let sper = (thoughts.len() * APERTURE).max(1) as f64;
                let qa = time(|| { for r in a0..a0 + APERTURE { for &t in &thoughts { black_box(a[t].read(&base, r)); } } }) / sper;
                let qb = time(|| { for r in a0..a0 + APERTURE { for &t in &thoughts { black_box(b[t].read(&base, r)); } } }) / sper;
                let qc = time(|| { for r in a0..a0 + APERTURE { for &t in &thoughts { black_box(c[t].read(&base, r)); } } }) / sper;

                // F1 / F2 / F3, ns per aperture row (F3 per emitted row).
                let ap = APERTURE as f64;
                let want1 = f1_b(&b[0], a0);
                assert_eq!(f1_a(&a[0], &base, a0), want1, "F1 A");
                assert_eq!(f1_c(&c[0], a0, &mut acc), want1, "F1 C");
                let want2 = f2_b(&b, 5, a0);
                assert_eq!(f2_a(&a, &base, 5, a0), want2, "F2 A");
                assert_eq!(f2_c(&c, 5, a0, &mut acc, &mut tmp), want2, "F2 C");
                let f1a = time(|| { black_box(f1_a(&a[0], &base, black_box(a0))); }) / ap;
                let f1b = time(|| { black_box(f1_b(&b[0], black_box(a0))); }) / ap;
                let f1c = time(|| { black_box(f1_c(&c[0], black_box(a0), &mut acc)); }) / ap;
                let f2a = time(|| { black_box(f2_a(&a, &base, 5, black_box(a0))); }) / ap;
                let f2b = time(|| { black_box(f2_b(&b, 5, black_box(a0))); }) / ap;
                let f2c = time(|| { black_box(f2_c(&c, 5, black_box(a0), &mut acc, &mut tmp)); }) / ap;
                let emitted = want1.max(1) as f64;
                let f3a = time(|| { f3_a(&a[0], &base, black_box(a0), &mut out); black_box(&out); }) / emitted;
                let f3b = time(|| { f3_b(&b[0], &base, black_box(a0), &mut out); black_box(&out); }) / emitted;
                let f3c = time(|| { f3_c(&c[0], &base, black_box(a0), &mut acc, &mut out); black_box(&out); }) / emitted;

                if k <= 4 {
                    let b_ok = rb <= 1.25 * ra && qb <= 1.25 * qa && bb as f64 <= 0.5 * ba as f64;
                    let folds = f1b >= 2.0 * f1c && f2b >= 2.0 * f2c;
                    let reads = rcn <= 1.25 * rb && qc <= 1.25 * qb;
                    let c_ok = folds && reads && bc <= bb;
                    b_rule &= b_ok;
                    c_rule &= c_ok;
                    c_fold_wins &= folds;
                    c_read_ok &= reads;
                }
                let _ = dirty;
                println!(
                    "{t_n:>3} {label:>5} {k:>3} | {:>9} {:>9} {:>9} {:>10} | {sa:>6.1} {sb:>6.1} {sc:>6.1} | {ra:>6.2} {rb:>6.2} {rcn:>6.2} | {qa:>6.2} {qb:>6.2} {qc:>6.2} | {f1a:>7.3} {f1b:>7.3} {f1c:>7.3} | {f2a:>7.3} {f2b:>7.3} {f2c:>7.3} | {f3a:>6.1} {f3b:>6.1} {f3c:>6.1}",
                    ba >> 10, bb >> 10, bc >> 10, rc >> 10
                );
            }
        }
    }
    println!("\nseal: ns per dirty row; read/sess: ns per row read; F1/F2: ns per aperture row; F3: ns per emitted row");
    println!("rule at k <= 4, every (dirty, thoughts) cell:");
    println!("  C folds >= 2x faster than B on F1 and F2: {}", if c_fold_wins { "yes" } else { "no" });
    println!("  C reads (merge-on-read) <= 1.25x B: {}", if c_read_ok { "yes" } else { "no" });
    println!("  C default: {}", if c_rule { "HOLDS" } else { "not met" });
    println!("  B default over A: {}", if b_rule { "HOLDS" } else { "not met" });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(t_n: usize, per_10k: u64, k: usize) -> (Vec<Row>, Vec<Writes>) {
        let b = base();
        let ws = (0..t_n).map(|t| writes(&b, t, per_10k, k)).collect();
        (b, ws)
    }

    /// The dense reference: each thought's rows written in full.
    fn dense(base: &[Row], w: &Writes) -> Vec<Row> {
        let mut d = base.to_vec();
        let mut v = 0;
        for (&r, &m) in w.rows.iter().zip(&w.masks) {
            for l in BitIter(m) {
                d[r as usize][l * LANE..(l + 1) * LANE].copy_from_slice(&w.vals[v]);
                v += 1;
            }
        }
        d
    }

    #[test]
    fn every_arm_reads_the_dense_reference() {
        for k in [1, 4, 31] {
            let (base, ws) = fixture(2, 1_000, k);
            for w in &ws {
                let d = dense(&base, w);
                let (a, b, c) = (ArmA::seal(&base, w), ArmB::seal(w), ArmC::seal(w));
                let mut changed = 0;
                for r in 0..N {
                    assert_eq!(a.read(&base, r), d[r], "A row {r} k {k}");
                    assert_eq!(b.read(&base, r), d[r], "B row {r} k {k}");
                    assert_eq!(c.read(&base, r), d[r], "C row {r} k {k}");
                    changed += usize::from(d[r] != base[r]);
                }
                assert_eq!(changed, w.rows.len(), "every dirty row changes, no other");
            }
        }
    }

    #[test]
    fn the_session_mask_names_exactly_the_writing_thoughts() {
        let b = base();
        let ws: Vec<Writes> = (0..4)
            .map(|t| writes(&b, t, if t == 2 { 0 } else { 1_000 }, 4))
            .collect();
        assert!(ws[2].rows.is_empty() && !ws[3].rows.is_empty());
        assert_eq!(session_mask(&ws), 0b1011);
    }

    #[test]
    fn a_lane_mask_is_the_set_of_lanes_that_differ_from_the_base() {
        let (base, ws) = fixture(1, 1_000, 4);
        let b = ArmB::seal(&ws[0]);
        let d = dense(&base, &ws[0]);
        for r in 0..N {
            let diff = (0..LANES).fold(0u32, |m, l| {
                m | u32::from(lane_bytes(&d[r], l) != lane_bytes(&base[r], l)) << l
            });
            assert_eq!(b.mask(r), diff, "row {r}");
        }
        assert!(b.masks.iter().all(|m| m.count_ones() == 4 && m & 1 == 0));
    }

    #[test]
    fn c_planes_name_exactly_the_rows_whose_lane_differs() {
        let (base, ws) = fixture(1, 1_000, 4);
        let c = ArmC::seal(&ws[0]);
        let d = dense(&base, &ws[0]);
        for l in 0..LANES {
            for r in 0..N {
                let differs = lane_bytes(&d[r], l) != lane_bytes(&base[r], l);
                let present = c.planes[l]
                    .as_ref()
                    .is_some_and(|p| p.present[r / 64] >> (r % 64) & 1 == 1);
                assert_eq!(present, differs, "lane {l} row {r}");
            }
        }
    }

    #[test]
    fn folds_and_terminals_agree_on_every_arm() {
        let (base, ws) = fixture(3, 1_000, 4);
        let a: Vec<ArmA> = ws.iter().map(|w| ArmA::seal(&base, w)).collect();
        let b: Vec<ArmB> = ws.iter().map(ArmB::seal).collect();
        let c: Vec<ArmC> = ws.iter().map(ArmC::seal).collect();
        let d0 = dense(&base, &ws[0]);
        let (mut acc, mut tmp) = (vec![0u64; APERTURE / 64], vec![0u64; APERTURE / 64]);
        for a0 in [0, 3 * APERTURE, N - APERTURE] {
            let want1 = (a0..a0 + APERTURE)
                .filter(|&r| F1_LANES.iter().any(|&l| lane_bytes(&d0[r], l) != lane_bytes(&base[r], l)))
                .count() as u64;
            assert!(want1 > 0, "fixture changes F1 lanes in the aperture");
            assert_eq!(f1_a(&a[0], &base, a0), want1);
            assert_eq!(f1_b(&b[0], a0), want1);
            assert_eq!(f1_c(&c[0], a0, &mut acc), want1);
            for l in [1, 5, 30] {
                let want2 = f2_b(&b, l, a0);
                assert_eq!(f2_a(&a, &base, l, a0), want2, "lane {l}");
                assert_eq!(f2_c(&c, l, a0, &mut acc, &mut tmp), want2, "lane {l}");
            }
            let (mut oa, mut ob, mut oc) = (Vec::new(), Vec::new(), Vec::new());
            f3_a(&a[0], &base, a0, &mut oa);
            f3_b(&b[0], &base, a0, &mut ob);
            f3_c(&c[0], &base, a0, &mut acc, &mut oc);
            assert_eq!(oa.len() as u64, want1);
            assert_eq!(oa, ob);
            assert_eq!(oa, oc);
        }
    }

    #[test]
    fn c_counts_without_building_rows_and_builds_only_what_f3_emits() {
        let (base, ws) = fixture(2, 1_000, 4);
        let c: Vec<ArmC> = ws.iter().map(ArmC::seal).collect();
        let (mut acc, mut tmp) = (vec![0u64; APERTURE / 64], vec![0u64; APERTURE / 64]);
        MERGES.with(|m| m.set(0));
        let n1 = f1_c(&c[0], 0, &mut acc);
        let _ = f2_c(&c, 5, 0, &mut acc, &mut tmp);
        assert_eq!(MERGES.with(|m| m.get()), 0, "F1 and F2 build no row");
        let mut out = Vec::new();
        f3_c(&c[0], &base, 0, &mut acc, &mut out);
        assert!(n1 > 0);
        assert_eq!(MERGES.with(|m| m.get()), n1, "one merge per emitted row");
    }

    #[test]
    fn reading_a_lane_touches_one_position() {
        let (_, ws) = fixture(1, 10_000, 31);
        let c = ArmC::seal(&ws[0]);
        VALUE_READS.with(|v| v.set(0));
        let mut found = 0;
        for r in [0usize, 777, N - 1] {
            for l in 1..LANES {
                found += u64::from(c.value(l, r).is_some());
            }
        }
        assert_eq!(found, 3 * 31, "every lane of a fully written row is present");
        assert_eq!(VALUE_READS.with(|v| v.get()), found, "one value access per lane read");
    }
}
