//! D-DRISA-3 falsifiers. The oracle works per row on small integer states and
//! shares nothing with the planes, the carry words or the comparator.

use drisa_register_probe::alu::{bag_at_least, kofn, kofn_bytes, kofn_kleene, three_of_five_sop};

/// Per-row input state: 0 false, 1 true, 2 unknown (Kleene only).
struct Case {
    n_rows: usize,
    present: Vec<bool>,
    /// `state[i][row]`.
    state: Vec<Vec<u8>>,
}

fn rng(mut z: u64) -> impl FnMut() -> u64 {
    move || {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Zero,
    Full,
    Alternating,
    OneHot,
    Random,
    RandomKleene,
}

fn case(n: usize, n_rows: usize, shape: Shape, seed: u64) -> Case {
    let mut r = rng(seed);
    let present = (0..n_rows).map(|row| row % 13 != 7).collect();
    let state = (0..n)
        .map(|i| {
            (0..n_rows)
                .map(|row| match shape {
                    Shape::Zero => 0,
                    Shape::Full => 1,
                    Shape::Alternating => ((row + i) % 2) as u8,
                    Shape::OneHot => u8::from(row % n.max(1) == i),
                    Shape::Random => (r() % 2) as u8,
                    Shape::RandomKleene => (r() % 3) as u8,
                })
                .collect()
        })
        .collect();
    Case { n_rows, present, state }
}

fn plane(n_rows: usize, f: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut w = vec![0u64; n_rows.div_ceil(64)];
    for row in 0..n_rows {
        if f(row) {
            w[row / 64] |= 1 << (row % 64);
        }
    }
    w
}

fn bit(words: &[u64], row: usize) -> bool {
    words[row / 64] >> (row % 64) & 1 == 1
}

fn tail_clear(words: &[u64], n_rows: usize) -> bool {
    (n_rows..words.len() * 64).all(|row| !bit(words, row))
}

const NS: [usize; 6] = [1, 2, 4, 5, 16, 24];
const ROWS: [usize; 3] = [100, 4096, 65_536 + 37];

fn ks(n: usize) -> [usize; 5] {
    [0, 1, n.div_ceil(2), n, n + 1]
}

/// A1/A2: two-valued k-of-n, every n, k and shape, three row counts with
/// tails; carry arm and byte-counter arm both equal the oracle.
#[test]
fn kofn_matches_the_per_row_oracle_everywhere() {
    let mut cells = 0;
    for shape in [Shape::Zero, Shape::Full, Shape::Alternating, Shape::OneHot, Shape::Random] {
        for &n in &NS {
            for &rows in &ROWS {
                let c = case(n, rows, shape, (n * 7 + rows) as u64);
                let presence = plane(rows, |r| c.present[r]);
                let ins: Vec<Vec<u64>> = c.state.iter().map(|s| plane(rows, |r| s[r] == 1)).collect();
                let refs: Vec<&[u64]> = ins.iter().map(|v| v.as_slice()).collect();
                for k in ks(n) {
                    let got = kofn(&refs, k, &presence, rows);
                    let bytes = kofn_bytes(&refs, k, &presence, rows);
                    for row in 0..rows {
                        let cnt = c.state.iter().filter(|s| s[row] == 1).count();
                        let want = c.present[row] && cnt >= k;
                        assert_eq!(bit(&got, row), want, "{shape:?} n={n} rows={rows} k={k} row={row}");
                        assert_eq!(bit(&bytes, row), want, "bytes {shape:?} n={n} k={k} row={row}");
                    }
                    assert!(tail_clear(&got, rows) && tail_clear(&bytes, rows));
                    cells += 1;
                }
            }
        }
    }
    assert_eq!(cells, 5 * 6 * 3 * 5);
}

/// Silence twins: k = 1 is OR, k = n is AND, both under presence; k = 0 is
/// exactly the present rows (absent rows never become true); k > n is empty.
#[test]
fn boundary_ks_are_or_and_presence_and_empty() {
    for &n in &NS {
        let rows = 65_536 + 37;
        let c = case(n, rows, Shape::Random, 99 + n as u64);
        let presence = plane(rows, |r| c.present[r]);
        let ins: Vec<Vec<u64>> = c.state.iter().map(|s| plane(rows, |r| s[r] == 1)).collect();
        let refs: Vec<&[u64]> = ins.iter().map(|v| v.as_slice()).collect();
        let or: Vec<u64> = (0..presence.len()).map(|w| refs.iter().fold(0, |a, m| a | m[w]) & presence[w]).collect();
        let and: Vec<u64> = (0..presence.len()).map(|w| refs.iter().fold(!0, |a, m| a & m[w]) & presence[w]).collect();
        assert_eq!(kofn(&refs, 1, &presence, rows), or);
        assert_eq!(kofn(&refs, n, &presence, rows), and);
        assert_eq!(kofn(&refs, 0, &presence, rows), presence);
        assert!(kofn(&refs, n + 1, &presence, rows).iter().all(|&w| w == 0));
        // Anti-vacuity: the middle k is neither the OR nor the AND.
        if n >= 4 {
            let mid = kofn(&refs, n.div_ceil(2), &presence, rows);
            assert!(mid != or && mid != and, "n={n}");
        }
    }
}

/// A5: Kleene. TRUE iff the TRUE count reaches k; FALSE iff TRUE + UNKNOWN
/// does not; UNKNOWN otherwise. UNKNOWN is never turned into FALSE before the
/// terminal, and an absent row is neither TRUE nor UNKNOWN.
#[test]
fn kleene_kofn_keeps_unknown_and_matches_the_oracle() {
    let mut unknowns = 0;
    for &n in &NS {
        for &rows in &ROWS {
            let c = case(n, rows, Shape::RandomKleene, (n * 31 + rows) as u64);
            let presence = plane(rows, |r| c.present[r]);
            let t: Vec<Vec<u64>> = c.state.iter().map(|s| plane(rows, |r| s[r] == 1)).collect();
            let u: Vec<Vec<u64>> = c.state.iter().map(|s| plane(rows, |r| s[r] == 2)).collect();
            let tr: Vec<&[u64]> = t.iter().map(|v| v.as_slice()).collect();
            let ur: Vec<&[u64]> = u.iter().map(|v| v.as_slice()).collect();
            for k in ks(n) {
                let (yes, unk) = kofn_kleene(&tr, &ur, k, &presence, rows);
                for row in 0..rows {
                    let ct = c.state.iter().filter(|s| s[row] == 1).count();
                    let cu = c.state.iter().filter(|s| s[row] == 2).count();
                    let (want_t, want_u) = if !c.present[row] {
                        (false, false)
                    } else if ct >= k {
                        (true, false)
                    } else if ct + cu >= k {
                        (false, true)
                    } else {
                        (false, false)
                    };
                    assert_eq!((bit(&yes, row), bit(&unk, row)), (want_t, want_u), "n={n} k={k} row={row}");
                    unknowns += usize::from(want_u);
                }
                assert!(tail_clear(&yes, rows) && tail_clear(&unk, rows));
            }
        }
    }
    assert!(unknowns > 1000, "the fixture must produce UNKNOWN results: {unknowns}");
}

/// A6: Bag multiplicity is not support. Same support (every input present at
/// least once), different sums: the Bag threshold separates them, the support
/// count cannot.
#[test]
fn bag_sums_differ_from_support_counts() {
    let rows = 4096 + 5;
    let mut r = rng(5);
    let n = 5;
    // multiplicity m[i][row] in 0..=3
    let m: Vec<Vec<u8>> = (0..n).map(|_| (0..rows).map(|_| (r() % 4) as u8).collect()).collect();
    let present: Vec<bool> = (0..rows).map(|row| row % 11 != 3).collect();
    let presence = plane(rows, |row| present[row]);
    let lo: Vec<Vec<u64>> = m.iter().map(|v| plane(rows, |row| v[row] & 1 == 1)).collect();
    let hi: Vec<Vec<u64>> = m.iter().map(|v| plane(rows, |row| v[row] & 2 == 2)).collect();
    let support: Vec<Vec<u64>> = m.iter().map(|v| plane(rows, |row| v[row] > 0)).collect();
    let lr: Vec<&[u64]> = lo.iter().map(|v| v.as_slice()).collect();
    let hr: Vec<&[u64]> = hi.iter().map(|v| v.as_slice()).collect();
    let sr: Vec<&[u64]> = support.iter().map(|v| v.as_slice()).collect();
    let mut differ = 0;
    for k in [0, 1, 3, 5, 8, 12, 15, 16] {
        let bag = bag_at_least(&lr, &hr, k, &presence, rows);
        let sup = kofn(&sr, k, &presence, rows);
        for row in 0..rows {
            let sum: usize = m.iter().map(|v| usize::from(v[row])).sum();
            assert_eq!(bit(&bag, row), present[row] && sum >= k, "k={k} row={row}");
        }
        differ += usize::from(bag != sup);
        assert!(tail_clear(&bag, rows));
    }
    assert!(differ >= 4, "Bag and support must disagree on several k: {differ}");
}

/// A7 negative control: 3-of-5 by carry words equals the 10-term native
/// sum of products (the timing comparison is in the bench, not asserted).
#[test]
fn three_of_five_matches_its_boolean_expansion() {
    let rows = 65_536 + 37;
    let c = case(5, rows, Shape::Random, 77);
    let presence = plane(rows, |r| c.present[r]);
    let ins: Vec<Vec<u64>> = c.state.iter().map(|s| plane(rows, |r| s[r] == 1)).collect();
    let refs: Vec<&[u64]> = ins.iter().map(|v| v.as_slice()).collect();
    let arr: [&[u64]; 5] = [refs[0], refs[1], refs[2], refs[3], refs[4]];
    assert_eq!(kofn(&refs, 3, &presence, rows), three_of_five_sop(&arr, &presence, rows));
}

/// A presence plane whose tail word carries stray bits past `n_rows` (a
/// caller that did not clear it) must not leak them into any result.
#[test]
fn a_dirty_presence_tail_never_reaches_the_result() {
    let rows = 4096 + 37;
    let c = case(5, rows, Shape::Full, 3);
    let mut presence = plane(rows, |r| c.present[r]);
    *presence.last_mut().unwrap() |= !((1u64 << (rows % 64)) - 1);
    let ins: Vec<Vec<u64>> = c.state.iter().map(|s| plane(rows, |r| s[r] == 1)).collect();
    let refs: Vec<&[u64]> = ins.iter().map(|v| v.as_slice()).collect();
    for k in [0, 1, 3, 5] {
        assert!(tail_clear(&kofn(&refs, k, &presence, rows), rows), "kofn k={k}");
        assert!(tail_clear(&kofn_bytes(&refs, k, &presence, rows), rows), "bytes k={k}");
    }
    let none = vec![0u64; presence.len()];
    let unknown: Vec<&[u64]> = vec![none.as_slice(); refs.len()];
    let (yes, unk) = kofn_kleene(&refs, &unknown, 0, &presence, rows);
    assert!(tail_clear(&yes, rows) && tail_clear(&unk, rows));
}
