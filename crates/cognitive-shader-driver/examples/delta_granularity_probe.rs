//! Delta granularity economics: 16-byte slot deltas vs 512-byte row images.
//!
//! Pre-registered in `.claude/plans/ce64-hypothesis-layers-v1.md`
//! ("Delta granularity: 16-byte slot or 512-byte row").
//!
//! Arms, over the same 65,536-row base of 512-byte rows:
//! - R: per dirty row, the full 512-byte image.
//! - S: per dirty row, a 32-bit slot mask plus the changed 16-byte slots.
//!   S-known gets the mask from the writer; S-diff compares with the base.
//!
//! Folds accumulate wide per (row, slot) and are rounded once at seal; that
//! path is the same for both arms and is timed once.
//!
//! Uses no crate types: it is pure byte movement, so it can also be built
//! with plain `rustc -O` without the crate's dependencies.
//!
//! Run (release, the tier is printed): `cargo run --release --example
//! delta_granularity_probe`. Tests: `cargo test --example delta_granularity_probe`.

use std::hint::black_box;
use std::time::Instant;

const ROW: usize = 512;
const SLOT: usize = 16;
const SLOTS: usize = 32;
const NONE: u32 = u32::MAX;

type Row = [u8; ROW];

fn tier() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "v4 (avx512f)"
    } else if cfg!(target_feature = "avx2") {
        "v3 (avx2): timings not reportable"
    } else {
        "below v3: timings not reportable"
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// The j-th slot a dirty row writes: distinct for j < 31, never slot 0 (key).
fn slot_of(row: usize, j: usize) -> usize {
    1 + (row * 7 + j * 5) % (SLOTS - 1)
}

fn base(n: usize) -> Vec<Row> {
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            let mut row = [0u8; ROW];
            for c in row.as_chunks_mut::<8>().0 {
                *c = r.next().to_le_bytes();
            }
            row
        })
        .collect()
}

/// Every `stride`-th row is dirty.
fn dirty_rows(n: usize, stride: usize) -> Vec<u32> {
    (0..n).step_by(stride).map(|r| r as u32).collect()
}

/// The value a sealed accumulator writes into its slot: always differs from
/// the base slot so every written slot is a real change.
fn sealed(acc: [u64; 2], base_slot: &[u8]) -> [u8; SLOT] {
    let f = acc[1].checked_div(acc[0]).unwrap_or(0);
    let mut w = [0u8; SLOT];
    w[..8].copy_from_slice(&f.to_le_bytes());
    w[8..].copy_from_slice(&acc[0].to_le_bytes());
    if w == base_slot {
        w[0] ^= 1;
    }
    w
}

// ---------- R: row images ----------

struct RowStore {
    index: Vec<u32>,
    rows: Vec<u32>,
    images: Vec<Row>,
}

impl RowStore {
    fn seal(base: &[Row], dirty: &[u32], k: usize, acc: &[[u64; 2]]) -> Self {
        let mut s = RowStore {
            index: vec![NONE; base.len()],
            rows: Vec::with_capacity(dirty.len()),
            images: Vec::with_capacity(dirty.len()),
        };
        for (i, &r) in dirty.iter().enumerate() {
            let mut img = base[r as usize];
            for j in 0..k {
                let sl = slot_of(r as usize, j);
                let w = sealed(acc[i * k + j], &img[sl * SLOT..(sl + 1) * SLOT]);
                img[sl * SLOT..(sl + 1) * SLOT].copy_from_slice(&w);
            }
            s.index[r as usize] = s.images.len() as u32;
            s.rows.push(r);
            s.images.push(img);
        }
        s
    }
    fn stored_bytes(&self) -> usize {
        self.images.len() * (4 + ROW)
    }
    fn read(&self, base: &[Row], r: usize) -> Row {
        match self.index[r] {
            NONE => base[r],
            i => self.images[i as usize],
        }
    }
    fn read_slot(&self, base: &[Row], r: usize, sl: usize) -> [u8; SLOT] {
        let src = match self.index[r] {
            NONE => &base[r],
            i => &self.images[i as usize],
        };
        src[sl * SLOT..(sl + 1) * SLOT].try_into().unwrap()
    }
}

// ---------- S: slot deltas ----------

struct SlotStore {
    index: Vec<u32>,
    rows: Vec<u32>,
    masks: Vec<u32>,
    starts: Vec<u32>,
    words: Vec<[u8; SLOT]>,
}

impl SlotStore {
    fn empty(n: usize, dirty: usize, k: usize) -> Self {
        SlotStore {
            index: vec![NONE; n],
            rows: Vec::with_capacity(dirty),
            masks: Vec::with_capacity(dirty),
            starts: Vec::with_capacity(dirty),
            words: Vec::with_capacity(dirty * k),
        }
    }
    fn push(&mut self, r: u32, mask: u32, words: impl Iterator<Item = [u8; SLOT]>) {
        if mask == 0 {
            return;
        }
        self.index[r as usize] = self.rows.len() as u32;
        self.rows.push(r);
        self.masks.push(mask);
        self.starts.push(self.words.len() as u32);
        self.words.extend(words);
    }
    /// The writer knows its slots; words are pushed in slot order.
    fn seal_known(base: &[Row], dirty: &[u32], k: usize, acc: &[[u64; 2]]) -> Self {
        let mut s = Self::empty(base.len(), dirty.len(), k);
        let mut tmp = [[0u8; SLOT]; SLOTS];
        for (i, &r) in dirty.iter().enumerate() {
            let b = &base[r as usize];
            let mut mask = 0u32;
            for j in 0..k {
                let sl = slot_of(r as usize, j);
                tmp[sl] = sealed(acc[i * k + j], &b[sl * SLOT..(sl + 1) * SLOT]);
                mask |= 1 << sl;
            }
            s.push(r, mask, BitIter(mask).map(|sl| tmp[sl]));
        }
        s
    }
    /// The mask is found by comparing each new row with the base.
    fn seal_diff(base: &[Row], dirty: &[u32], new: &[Row], k: usize) -> Self {
        let mut s = Self::empty(base.len(), dirty.len(), k);
        for (i, &r) in dirty.iter().enumerate() {
            let (b, n) = (&base[r as usize], &new[i]);
            let mut mask = 0u32;
            for sl in 0..SLOTS {
                if b[sl * SLOT..(sl + 1) * SLOT] != n[sl * SLOT..(sl + 1) * SLOT] {
                    mask |= 1 << sl;
                }
            }
            s.push(
                r,
                mask,
                BitIter(mask).map(|sl| n[sl * SLOT..(sl + 1) * SLOT].try_into().unwrap()),
            );
        }
        s
    }
    fn stored_bytes(&self) -> usize {
        self.rows.len() * (4 + 4) + self.words.len() * SLOT
    }
    fn read(&self, base: &[Row], r: usize) -> Row {
        let mut out = base[r];
        let i = self.index[r];
        if i != NONE {
            let i = i as usize;
            let w0 = self.starts[i] as usize;
            for (w, sl) in BitIter(self.masks[i]).enumerate() {
                out[sl * SLOT..(sl + 1) * SLOT].copy_from_slice(&self.words[w0 + w]);
            }
        }
        out
    }
    fn read_slot(&self, base: &[Row], r: usize, sl: usize) -> [u8; SLOT] {
        let i = self.index[r];
        if i != NONE {
            let m = self.masks[i as usize];
            if m & (1 << sl) != 0 {
                let off = (m & ((1u32 << sl) - 1)).count_ones() as usize;
                return self.words[self.starts[i as usize] as usize + off];
            }
        }
        base[r][sl * SLOT..(sl + 1) * SLOT].try_into().unwrap()
    }
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

// ---------- folds ----------

/// Fold stream: (accumulator index, weight, weighted value). Precomputed so
/// the timed loop measures the fold, not the generator.
fn fold_stream(n_acc: usize, len: usize) -> Vec<(u32, u32, u32)> {
    let mut r = Rng(0xD1B5_4A32_D192_ED03);
    (0..len)
        .map(|_| {
            let x = r.next();
            let w = (x >> 32) as u32 & 0xFF | 1;
            let f = (x >> 48) as u32 & 0xFF;
            ((x as u32) % n_acc as u32, w, w * f)
        })
        .collect()
}

fn fold(acc: &mut [[u64; 2]], stream: &[(u32, u32, u32)], passes: usize) {
    for _ in 0..passes {
        for &(i, w, wf) in stream {
            let a = &mut acc[i as usize];
            a[0] += w as u64;
            a[1] += wf as u64;
        }
    }
}

// ---------- timing ----------

/// Median over 5 runs of ns per element.
fn ns_per<F: FnMut()>(elements: usize, mut f: F) -> f64 {
    let mut v: Vec<f64> = (0..5)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64 / elements as f64
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[2]
}

fn main() {
    const N: usize = 65_536;
    println!("tier: {}", tier());
    let base = base(N);

    // M2: folds, arm-independent. 10^9 folds each, over the smallest and
    // the largest accumulator sets in the grid (every row dirty, k = 1 and
    // k = 31), in random order and, for k = 31, sorted by accumulator.
    for (k, sorted) in [(1usize, false), (31, false), (31, true)] {
        let n_acc = N * k;
        let mut stream = fold_stream(n_acc, 1 << 20);
        if sorted {
            stream.sort_unstable_by_key(|e| e.0);
        }
        let passes = 1_000_000_000usize.div_ceil(stream.len());
        let mut acc = vec![[0u64; 2]; n_acc];
        let t = Instant::now();
        fold(&mut acc, &stream, passes);
        let total = passes * stream.len();
        let el = t.elapsed();
        black_box(&acc);
        println!(
            "M2 folds k={k:>2} {}: {} folds in {:.3} s = {:.3} ns/fold ({} accumulators, {} KiB)",
            if sorted { "sorted" } else { "random" },
            total,
            el.as_secs_f64(),
            el.as_nanos() as f64 / total as f64,
            n_acc,
            (n_acc * 16) >> 10
        );
    }

    println!(
        "\n{:>5} {:>3} | {:>10} {:>10} {:>6} | {:>7} {:>7} {:>7} | {:>7} {:>7} | {:>7} {:>7} | {:>8} {:>8}",
        "dirty", "k", "M1 R B", "M1 S B", "S/R", "M3 R", "M3 Sk", "M3 Sd",
        "M4f R", "M4f S", "M4s R", "M4s S", "M5 R GiB", "M5 S GiB"
    );
    let mut r = Rng(7);
    let probe: Vec<(u32, u8)> = (0..1 << 16)
        .map(|_| {
            let x = r.next();
            ((x as u32) % N as u32, 1 + ((x >> 32) % 31) as u8)
        })
        .collect();
    for (label, stride) in [("1%", 100), ("10%", 10), ("100%", 1)] {
        let dirty = dirty_rows(N, stride);
        for k in [1usize, 2, 4, 8, 16, 31] {
            let acc: Vec<[u64; 2]> = (0..dirty.len() * k)
                .map(|i| [i as u64 + 1, (i as u64 + 1) * 3])
                .collect();
            let rs = RowStore::seal(&base, &dirty, k, &acc);
            let sk = SlotStore::seal_known(&base, &dirty, k, &acc);
            let sd = SlotStore::seal_diff(&base, &dirty, &rs.images, k);
            assert_eq!(sk.masks, sd.masks);
            let m3r = ns_per(dirty.len(), || {
                black_box(RowStore::seal(&base, &dirty, k, &acc));
            });
            let m3k = ns_per(dirty.len(), || {
                black_box(SlotStore::seal_known(&base, &dirty, k, &acc));
            });
            let m3d = ns_per(dirty.len(), || {
                black_box(SlotStore::seal_diff(&base, &dirty, &rs.images, k));
            });
            let m4fr = ns_per(probe.len(), || {
                for &(row, _) in &probe {
                    black_box(rs.read(&base, row as usize));
                }
            });
            let m4fs = ns_per(probe.len(), || {
                for &(row, _) in &probe {
                    black_box(sk.read(&base, row as usize));
                }
            });
            let m4sr = ns_per(probe.len(), || {
                for &(row, sl) in &probe {
                    black_box(rs.read_slot(&base, row as usize, sl as usize));
                }
            });
            let m4ss = ns_per(probe.len(), || {
                for &(row, sl) in &probe {
                    black_box(sk.read_slot(&base, row as usize, sl as usize));
                }
            });
            let (br, bs) = (rs.stored_bytes(), sk.stored_bytes());
            let gib = |b: usize| (b * 64) as f64 / (1u64 << 30) as f64;
            println!(
                "{:>5} {:>3} | {:>10} {:>10} {:>6.3} | {:>7.1} {:>7.1} {:>7.1} | {:>7.2} {:>7.2} | {:>7.2} {:>7.2} | {:>8.3} {:>8.3}",
                label, k, br, bs, bs as f64 / br as f64, m3r, m3k, m3d, m4fr, m4fs, m4sr, m4ss,
                gib(br), gib(bs)
            );
        }
    }
    println!(
        "\nM3 ns per dirty row; M4 ns per read (f = full row, s = one slot); M5 for 64 tables."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(k: usize) -> (Vec<Row>, Vec<u32>, Vec<[u64; 2]>) {
        let b = base(512);
        let d = dirty_rows(512, 3);
        let acc = (0..d.len() * k)
            .map(|i| [i as u64 + 1, (i as u64 + 1) * 5])
            .collect();
        (b, d, acc)
    }

    #[test]
    fn both_arms_read_the_same_bytes() {
        for k in [1, 4, 31] {
            let (b, d, acc) = setup(k);
            let rs = RowStore::seal(&b, &d, k, &acc);
            let sk = SlotStore::seal_known(&b, &d, k, &acc);
            let mut changed = 0;
            for r in 0..b.len() {
                let (x, y) = (rs.read(&b, r), sk.read(&b, r));
                assert_eq!(x, y, "row {r}, k {k}");
                changed += (x != b[r]) as usize;
                for sl in 0..SLOTS {
                    assert_eq!(rs.read_slot(&b, r, sl), sk.read_slot(&b, r, sl));
                }
            }
            assert_eq!(changed, d.len(), "every dirty row changes, no other");
        }
    }

    #[test]
    fn slot_store_keeps_only_written_slots() {
        let (b, d, acc) = setup(2);
        let sk = SlotStore::seal_known(&b, &d, 2, &acc);
        assert_eq!(sk.words.len(), d.len() * 2);
        assert_eq!(sk.stored_bytes(), d.len() * (8 + 2 * SLOT));
        let rs = RowStore::seal(&b, &d, 2, &acc);
        assert_eq!(rs.stored_bytes(), d.len() * (4 + ROW));
    }

    #[test]
    fn diff_seal_finds_the_known_mask() {
        for k in [1, 8, 31] {
            let (b, d, acc) = setup(k);
            let rs = RowStore::seal(&b, &d, k, &acc);
            let sk = SlotStore::seal_known(&b, &d, k, &acc);
            let sd = SlotStore::seal_diff(&b, &d, &rs.images, k);
            assert_eq!(sk.masks, sd.masks);
            assert_eq!(sk.words, sd.words);
            assert!(sk
                .masks
                .iter()
                .all(|m| m.count_ones() as usize == k && m & 1 == 0));
        }
    }

    #[test]
    fn unchanged_rows_store_nothing() {
        let b = base(64);
        let d = dirty_rows(64, 4);
        let same: Vec<Row> = d.iter().map(|&r| b[r as usize]).collect();
        let sd = SlotStore::seal_diff(&b, &d, &same, 1);
        assert_eq!(sd.stored_bytes(), 0);
        assert!(sd.index.iter().all(|&i| i == NONE));
    }

    #[test]
    fn folds_accumulate_wide_and_round_once() {
        let mut acc = vec![[0u64; 2]; 4];
        fold(&mut acc, &[(1, 3, 3 * 200), (1, 1, 100)], 2);
        assert_eq!(acc[1], [8, 1400]);
        assert_eq!(acc[0], [0, 0]);
        let w = sealed(acc[1], &[0u8; SLOT]);
        assert_eq!(u64::from_le_bytes(w[..8].try_into().unwrap()), 175);
    }
}
