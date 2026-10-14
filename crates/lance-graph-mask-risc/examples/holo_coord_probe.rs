//! D-HOLO H5: coordinate-native vs coordinate-crossing queries on one tile.
//!
//! Pre-registered in `.claude/plans/d-holo-address-indexed-algebra-v1.md` (H5).
//!
//! - Q1 `identity[$prefix].count()`: `Range` under the occupancy plane (no key
//!   read) vs a `MatchU64` sweep of the key lane, plus the closed form; an
//!   adversarial run on a permuted tile, where `Range` must disagree.
//! - Q2 `identity[$node].thoughts($all).count()`: one popcount of the row's
//!   written-by word vs 64 mask-risc programs vs 64 bit reads.
//! - Q3 reverse `memberOf`: a mask-risc scan of the four membership lanes vs a
//!   CSR inverse vs per-group inverse mask planes, with build and update cost.
//!
//! `cargo run --release -p lance-graph-mask-risc --example holo_coord_probe`
//! `cargo test -p lance-graph-mask-risc --example holo_coord_probe`

use std::hint::black_box;
use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_compiled, Scratch};
use lance_graph_mask_risc::{
    touched_words, words_for, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program,
    Terminal, Value,
};

const N: usize = 65_536;
const W: usize = N / 64;
const CLASSID: u64 = 0x0C0C_0000;
const THOUGHTS: usize = 64;
const GROUPS: u32 = 1024;
const SLOTS: usize = 4;

fn tier() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "v4 (avx512f)"
    } else if cfg!(target_feature = "avx2") {
        "v3 (avx2): timings not reportable"
    } else {
        "below v3: timings not reportable"
    }
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn count(v: Value) -> u64 {
    match v {
        Value::Count(n) => n as u64,
        other => panic!("expected a count, got {other:?}"),
    }
}

fn run(p: &Program, planes: &Planes<'_>, s: &mut Scratch<'_>, ext: std::ops::Range<usize>) -> u64 {
    count(execute_compiled(&p.compile(), planes, &Foreign::NONE, s, Out::None, ext).expect("run"))
}

/// Median of 11 runs, ns per call.
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

// ---------- Q1 ----------

/// The key of row `r` on a dense tile: row = HEEL code.
fn key(r: usize) -> u64 {
    (CLASSID << 32) | ((r as u64) << 16)
}

fn occupancy(density_pct: u64, seed: u64) -> Vec<u64> {
    let mut o = vec![0u64; W];
    for r in 0..N {
        if mix(seed ^ r as u64) % 100 < density_pct {
            o[r / 64] |= 1 << (r % 64);
        }
    }
    o
}

/// Prefix of `p` HEEL nibbles with value `pv`: its row interval.
fn prefix_rows(p: u32, pv: u64) -> (usize, usize) {
    let span = 1usize << (4 * (4 - p));
    let lo = (pv as usize) * span;
    (lo, lo + span)
}

fn prefix_match(p: u32, pv: u64) -> Pred {
    let heel_care: u64 = (0xFFFFu64 << (4 * (4 - p))) & 0xFFFF;
    let pattern = (CLASSID << 32) | ((pv << (4 * (4 - p))) << 16);
    Pred::MatchU64 {
        lane: 0,
        pattern,
        care: (0xFFFF_FFFFu64 << 32) | (heel_care << 16),
    }
}

fn q1_range(lo: usize, hi: usize) -> Program {
    Program::new(
        vec![MaskOp::Pred {
            pred: Pred::Range {
                lo: lo as u32,
                hi: hi as u32,
            },
            under: Some(Operand::Plane(0)),
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    )
}

fn q1_sweep(p: u32, pv: u64) -> Program {
    Program::new(
        vec![MaskOp::Pred {
            pred: prefix_match(p, pv),
            under: Some(Operand::Plane(0)),
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    )
}

fn q1_oracle(keys: &[u64], o: &[u64], p: u32, pv: u64) -> u64 {
    let Pred::MatchU64 { pattern, care, .. } = prefix_match(p, pv) else {
        unreachable!()
    };
    (0..N)
        .filter(|&r| (o[r / 64] >> (r % 64)) & 1 == 1 && (keys[r] ^ pattern) & care == 0)
        .count() as u64
}

// ---------- Q2 ----------

fn coverage(seed: u64) -> (Vec<u64>, Vec<Vec<u64>>) {
    let mut w = vec![0u64; N];
    let mut m = vec![vec![0u64; W]; THOUGHTS];
    for (r, word) in w.iter_mut().enumerate() {
        for (t, plane) in m.iter_mut().enumerate() {
            if mix(seed ^ ((r as u64) << 8) ^ t as u64) % 10 == 0 {
                *word |= 1 << t;
                plane[r / 64] |= 1 << (r % 64);
            }
        }
    }
    (w, m)
}

fn q2_prog(t: u16, row: usize) -> Program {
    Program::new(
        vec![MaskOp::Pred {
            pred: Pred::Range {
                lo: row as u32,
                hi: row as u32 + 1,
            },
            under: Some(Operand::Plane(t)),
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    )
}

// ---------- Q3 ----------

struct Members {
    /// `lanes[j][r]`: the j-th group of identity r, 0 = none.
    lanes: Vec<Vec<u32>>,
    v: Vec<i32>,
}

fn members(skewed: bool, seed: u64) -> Members {
    // Zipf(1) over 1..=GROUPS by inverse CDF.
    let h: Vec<f64> = (1..=GROUPS)
        .scan(0.0, |acc, k| {
            *acc += 1.0 / f64::from(k);
            Some(*acc)
        })
        .collect();
    let total = *h.last().unwrap();
    let mut lanes = vec![vec![0u32; N]; SLOTS];
    for r in 0..N {
        let mut have = [0u32; SLOTS];
        for (j, lane) in lanes.iter_mut().enumerate() {
            let x = mix(seed ^ ((r as u64) << 4) ^ j as u64);
            if x % 4 == 0 {
                continue; // a quarter of the slots are empty
            }
            let u = (x >> 8) as f64 / (1u64 << 56) as f64;
            let g = if skewed {
                h.partition_point(|&c| c < u * total) as u32 + 1
            } else {
                (x >> 8) as u32 % GROUPS + 1
            };
            if !have.contains(&g) {
                have[j] = g;
                lane[r] = g;
            }
        }
    }
    let v = (0..N)
        .map(|r| (mix(seed ^ 0xABCD ^ r as u64) % 1000) as i32)
        .collect();
    Members { lanes, v }
}

fn q3_scan(g: u32, lo: usize, hi: usize, t: Option<i32>) -> Program {
    let mut ops = vec![MaskOp::Pred {
        pred: Pred::Range {
            lo: lo as u32,
            hi: hi as u32,
        },
        under: None,
        dst: 0,
    }];
    if let Some(t) = t {
        ops.push(MaskOp::Pred {
            pred: Pred::LtI32 {
                lane: SLOTS as u16,
                t,
            },
            under: Some(Operand::Scratch(0)),
            dst: 1,
        });
        ops.push(MaskOp::Or {
            a: Operand::Scratch(1),
            b: Operand::Scratch(1),
            dst: 0,
        });
    }
    for j in 0..SLOTS as u16 {
        ops.push(MaskOp::Pred {
            pred: Pred::EqU32 { lane: j, v: g },
            under: Some(Operand::Scratch(0)),
            dst: 2 + j,
        });
    }
    ops.push(MaskOp::Or {
        a: Operand::Scratch(2),
        b: Operand::Scratch(3),
        dst: 6,
    });
    ops.push(MaskOp::Or {
        a: Operand::Scratch(4),
        b: Operand::Scratch(5),
        dst: 7,
    });
    ops.push(MaskOp::Or {
        a: Operand::Scratch(6),
        b: Operand::Scratch(7),
        dst: 6,
    });
    Program::new(
        ops,
        Terminal::Count {
            mask: Operand::Scratch(6),
        },
    )
}

struct Csr {
    off: Vec<u32>,
    rows: Vec<u32>,
}

fn build_csr(m: &Members) -> Csr {
    let mut deg = vec![0u32; GROUPS as usize + 2];
    for lane in &m.lanes {
        for &g in lane {
            if g != 0 {
                deg[g as usize + 1] += 1;
            }
        }
    }
    for i in 1..deg.len() {
        deg[i] += deg[i - 1];
    }
    let mut fill = deg.clone();
    let mut rows = vec![0u32; *deg.last().unwrap() as usize];
    for r in 0..N {
        for lane in &m.lanes {
            let g = lane[r];
            if g != 0 {
                rows[fill[g as usize] as usize] = r as u32;
                fill[g as usize] += 1;
            }
        }
    }
    Csr { off: deg, rows }
}

impl Csr {
    fn members(&self, g: u32) -> &[u32] {
        &self.rows[self.off[g as usize] as usize..self.off[g as usize + 1] as usize]
    }
    fn count(&self, g: u32, lo: usize, hi: usize, v: Option<(&[i32], i32)>) -> u64 {
        let m = self.members(g);
        let a = m.partition_point(|&r| (r as usize) < lo);
        let b = m.partition_point(|&r| (r as usize) < hi);
        match v {
            None => (b - a) as u64,
            Some((v, t)) => m[a..b].iter().filter(|&&r| v[r as usize] < t).count() as u64,
        }
    }
}

fn build_group_planes(m: &Members) -> Vec<Vec<u64>> {
    let mut p = vec![vec![0u64; W]; GROUPS as usize + 1];
    for lane in &m.lanes {
        for (r, &g) in lane.iter().enumerate() {
            if g != 0 {
                p[g as usize][r / 64] |= 1 << (r % 64);
            }
        }
    }
    p
}

fn q3_plane(lo: usize, hi: usize, t: Option<i32>) -> Program {
    let mut ops = vec![MaskOp::Pred {
        pred: Pred::Range {
            lo: lo as u32,
            hi: hi as u32,
        },
        under: Some(Operand::Plane(0)),
        dst: 0,
    }];
    if let Some(t) = t {
        ops.push(MaskOp::Pred {
            pred: Pred::LtI32 { lane: 0, t },
            under: Some(Operand::Scratch(0)),
            dst: 1,
        });
        return Program::new(
            ops,
            Terminal::Count {
                mask: Operand::Scratch(1),
            },
        );
    }
    Program::new(
        ops,
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    )
}

fn q3_oracle(m: &Members, g: u32, lo: usize, hi: usize, t: Option<i32>) -> u64 {
    (lo..hi)
        .filter(|&r| m.lanes.iter().any(|l| l[r] == g) && t.is_none_or(|t| m.v[r] < t))
        .count() as u64
}

struct Q3 {
    m: Members,
    csr: Csr,
    gp: Vec<Vec<u64>>,
}

impl Q3 {
    fn scan(&self, g: u32, lo: usize, hi: usize, t: Option<i32>, s: &mut Scratch<'_>) -> u64 {
        let lanes: [LaneRef<'_>; SLOTS + 1] = std::array::from_fn(|j| {
            if j < SLOTS {
                LaneRef::U32(&self.m.lanes[j])
            } else {
                LaneRef::I32(&self.m.v)
            }
        });
        let planes = Planes {
            n_rows: N,
            masks: &[],
            lanes: &lanes,
        };
        run(&q3_scan(g, lo, hi, t), &planes, s, lo..hi)
    }
    fn inverse_planes(
        &self,
        g: u32,
        lo: usize,
        hi: usize,
        t: Option<i32>,
        s: &mut Scratch<'_>,
    ) -> u64 {
        let masks = [self.gp[g as usize].as_slice()];
        let lanes = [LaneRef::I32(&self.m.v)];
        let planes = Planes {
            n_rows: N,
            masks: &masks,
            lanes: &lanes,
        };
        run(&q3_plane(lo, hi, t), &planes, s, lo..hi)
    }
}

fn main() {
    println!("tier: {}", tier());

    // ---- Q1 ----
    let keys: Vec<u64> = (0..N).map(key).collect();
    let mut perm: Vec<u64> = keys.clone();
    for i in (1..N).rev() {
        perm.swap(i, mix(0xFEED ^ i as u64) as usize % (i + 1));
    }
    println!("# Q1 identity[$prefix].count()");
    println!("occ\tp\trows\tR_ns\tS_ns\tR_words_read\tS_rows_read\tanswer");
    for &occ in &[100u64, 30] {
        let o = occupancy(occ, 7);
        for &p in &[1u32, 2, 3] {
            let pv = (1u64 << (4 * p)) / 2 + 3;
            let (lo, hi) = prefix_rows(p, pv);
            let masks = [o.as_slice()];
            let lanes = [LaneRef::U64(&keys)];
            let planes = Planes {
                n_rows: N,
                masks: &masks,
                lanes: &lanes,
            };
            let (pr, ps) = (q1_range(lo, hi), q1_sweep(p, pv));
            let mut s = Scratch::for_program(&ps, N).expect("scratch");
            let want = q1_oracle(&keys, &o, p, pv);
            let r = run(&pr, &planes, &mut s, 0..N);
            let sw = run(&ps, &planes, &mut s, 0..N);
            assert_eq!((r, sw), (want, want), "Q1 occ {occ} p {p}");
            if occ == 100 {
                assert_eq!(want, 1u64 << (4 * (4 - p)), "closed form");
            }
            let rn = time(|| {
                black_box(run(&pr, &planes, &mut s, 0..N));
            });
            let sn = time(|| {
                black_box(run(&ps, &planes, &mut s, 0..N));
            });
            println!(
                "{occ}%\t{p}\t{}\t{rn:.1}\t{sn:.1}\t{}\t{N}\t{want}",
                hi - lo,
                touched_words(lo as u32, hi as u32).len()
            );
        }
    }
    {
        // Adversarial: the key lane is permuted, so rows are no longer in key
        // order. `Range` reads no key, so it must now be wrong.
        let o = occupancy(30, 7);
        let (p, pv) = (1u32, 11u64);
        let (lo, hi) = prefix_rows(p, pv);
        let masks = [o.as_slice()];
        let lanes = [LaneRef::U64(&perm)];
        let planes = Planes {
            n_rows: N,
            masks: &masks,
            lanes: &lanes,
        };
        let mut s = Scratch::for_program(&q1_sweep(p, pv), N).expect("scratch");
        let r = run(&q1_range(lo, hi), &planes, &mut s, 0..N);
        let sw = run(&q1_sweep(p, pv), &planes, &mut s, 0..N);
        let want = q1_oracle(&perm, &o, p, pv);
        assert_eq!(sw, want);
        assert_ne!(
            r, want,
            "the permuted tile must expose a Range without an order witness"
        );
        println!(
            "# adversarial permuted tile: Range {r}, sweep {sw}, oracle {want}: Range disagrees"
        );
    }

    // ---- Q2 ----
    let (w, m) = coverage(11);
    let probes: Vec<usize> = (0..1024).map(|i| mix(i as u64) as usize % N).collect();
    let thought_masks: [&[u64]; THOUGHTS] = std::array::from_fn(|t| m[t].as_slice());
    let planes = Planes {
        n_rows: N,
        masks: &thought_masks,
        lanes: &[],
    };
    let progs: Vec<Vec<Program>> = probes[..16]
        .iter()
        .map(|&row| (0..THOUGHTS as u16).map(|t| q2_prog(t, row)).collect())
        .collect();
    let mut s = Scratch::for_program(&progs[0][0], N).expect("scratch");
    for (i, &row) in probes[..16].iter().enumerate() {
        let want = u64::from(w[row].count_ones());
        let viap: u64 = progs[i]
            .iter()
            .map(|p| run(p, &planes, &mut s, row..row + 1))
            .sum();
        let bits: u64 = (0..THOUGHTS)
            .map(|t| (m[t][row / 64] >> (row % 64)) & 1)
            .sum();
        assert_eq!((viap, bits), (want, want));
    }
    let wn = time(|| {
        black_box(
            probes
                .iter()
                .map(|&r| w[black_box(r)].count_ones())
                .sum::<u32>(),
        );
    }) / probes.len() as f64;
    let bn = time(|| {
        black_box(
            probes
                .iter()
                .map(|&r| {
                    (0..THOUGHTS)
                        .map(|t| (m[t][r / 64] >> (r % 64)) & 1)
                        .sum::<u64>()
                })
                .sum::<u64>(),
        );
    }) / probes.len() as f64;
    let pn = time(|| {
        for (i, &row) in probes[..16].iter().enumerate() {
            black_box(
                progs[i]
                    .iter()
                    .map(|p| run(p, &planes, &mut s, row..row + 1))
                    .sum::<u64>(),
            );
        }
    }) / 16.0;
    println!("# Q2 identity[$node].thoughts($all).count(), ns per identity");
    println!("W (row word popcount) {wn:.2}\tB (64 plane bit reads) {bn:.2}\tP (64 mask-risc programs) {pn:.0}");

    // ---- Q3 ----
    println!("# Q3 reverse memberOf");
    println!("dist\tquery\tS_ns\tI_ns\tIM_ns\tS_rows_read");
    for &skewed in &[false, true] {
        let mem = members(skewed, 21);
        let edges: usize = mem
            .lanes
            .iter()
            .map(|l| l.iter().filter(|&&g| g != 0).count())
            .sum();
        let t0 = Instant::now();
        let csr = black_box(build_csr(&mem));
        let csr_ns = t0.elapsed().as_nanos() as f64;
        let t0 = Instant::now();
        let gp = black_box(build_group_planes(&mem));
        let gp_ns = t0.elapsed().as_nanos() as f64;
        let csr_bytes = csr.off.len() * 4 + csr.rows.len() * 4;
        let gp_bytes = gp.len() * W * 8;
        // One-edge insert: CSR shifts every later row and offset; the mask
        // plane sets one bit.
        let mut c2 = Csr {
            off: csr.off.clone(),
            rows: csr.rows.clone(),
        };
        let t0 = Instant::now();
        for i in 0..100u32 {
            let g = i % GROUPS + 1;
            let at = c2.off[g as usize + 1] as usize;
            c2.rows.insert(at, 7);
            for o in &mut c2.off[g as usize + 1..] {
                *o += 1;
            }
        }
        let csr_ins = t0.elapsed().as_nanos() as f64 / 100.0;
        let mut gp2 = vec![0u64; W];
        let t0 = Instant::now();
        for i in 0..100usize {
            gp2[black_box(i * 37 % W)] |= 1 << (i % 64);
        }
        let gp_ins = t0.elapsed().as_nanos() as f64 / 100.0;
        black_box(&gp2);
        let dist = if skewed { "zipf" } else { "uniform" };
        println!(
            "# {dist}: {edges} edges; CSR build {:.2} ns/edge, {} KiB, insert {csr_ins:.0} ns; group planes build {:.2} ns/edge, {} KiB, insert {gp_ins:.1} ns",
            csr_ns / edges as f64,
            csr_bytes / 1024,
            gp_ns / edges as f64,
            gp_bytes / 1024
        );
        let q = Q3 { m: mem, csr, gp };
        let groups: Vec<u32> = (0..256)
            .map(|i| (mix(0x77 ^ i) % u64::from(GROUPS)) as u32 + 1)
            .collect();
        let mut s = Scratch::for_program(&q3_scan(1, 0, N, Some(1)), N).expect("scratch");
        let (plo, phi) = (5 * 4096, 6 * 4096);
        let shapes: [(&str, usize, usize, Option<i32>); 3] = [
            ("Q3a reverse(g).count", 0, N, None),
            ("Q3b reverse(g)&prefix", plo, phi, None),
            ("Q3c reverse(g)&(v<t)", 0, N, Some(100)),
        ];
        for (name, lo, hi, t) in shapes {
            for &g in &groups[..32] {
                let want = q3_oracle(&q.m, g, lo, hi, t);
                assert_eq!(q.scan(g, lo, hi, t, &mut s), want, "{name} scan g={g}");
                assert_eq!(
                    q.csr.count(g, lo, hi, t.map(|t| (q.m.v.as_slice(), t))),
                    want,
                    "{name} csr"
                );
                assert_eq!(
                    q.inverse_planes(g, lo, hi, t, &mut s),
                    want,
                    "{name} planes"
                );
            }
            let sn = time(|| {
                for &g in &groups {
                    black_box(q.scan(g, lo, hi, t, &mut s));
                }
            }) / groups.len() as f64;
            let iv = t.map(|t| (q.m.v.as_slice(), t));
            let inn = time(|| {
                for &g in &groups {
                    black_box(q.csr.count(g, lo, hi, iv));
                }
            }) / groups.len() as f64;
            let imn = time(|| {
                for &g in &groups {
                    black_box(q.inverse_planes(g, lo, hi, t, &mut s));
                }
            }) / groups.len() as f64;
            let be_i = if sn > inn {
                csr_ns / (sn - inn)
            } else {
                f64::INFINITY
            };
            let be_im = if sn > imn {
                gp_ns / (sn - imn)
            } else {
                f64::INFINITY
            };
            println!(
                "{dist}\t{name}\t{sn:.0}\t{inn:.1}\t{imn:.0}\t{}\tbreak-even N: CSR {be_i:.0}, planes {be_im:.0}",
                hi - lo
            );
        }
    }
    let _ = words_for(N);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FAILS IF: Range or the sweep disagree with the oracle on any prefix
    /// depth or occupancy.
    #[test]
    fn q1_range_and_sweep_match_the_oracle() {
        let keys: Vec<u64> = (0..N).map(key).collect();
        for &occ in &[100u64, 30] {
            let o = occupancy(occ, 3);
            for &p in &[1u32, 2, 3] {
                for pv in [0u64, (1 << (4 * p)) - 1, 5] {
                    let (lo, hi) = prefix_rows(p, pv);
                    let masks = [o.as_slice()];
                    let lanes = [LaneRef::U64(&keys)];
                    let planes = Planes {
                        n_rows: N,
                        masks: &masks,
                        lanes: &lanes,
                    };
                    let mut s = Scratch::for_program(&q1_sweep(p, pv), N).expect("scratch");
                    let want = q1_oracle(&keys, &o, p, pv);
                    assert!(want > 0);
                    assert_eq!(run(&q1_range(lo, hi), &planes, &mut s, 0..N), want);
                    assert_eq!(run(&q1_sweep(p, pv), &planes, &mut s, 0..N), want);
                }
            }
        }
    }

    /// FAILS IF: a permuted key lane does not expose a Range used without an
    /// order witness (the fixture could not detect the violation).
    #[test]
    fn a_permuted_tile_exposes_range_without_a_witness() {
        let mut perm: Vec<u64> = (0..N).map(key).collect();
        for i in (1..N).rev() {
            perm.swap(i, mix(0xFEED ^ i as u64) as usize % (i + 1));
        }
        let o = occupancy(30, 3);
        let (lo, hi) = prefix_rows(1, 11);
        let masks = [o.as_slice()];
        let lanes = [LaneRef::U64(&perm)];
        let planes = Planes {
            n_rows: N,
            masks: &masks,
            lanes: &lanes,
        };
        let mut s = Scratch::for_program(&q1_sweep(1, 11), N).expect("scratch");
        let want = q1_oracle(&perm, &o, 1, 11);
        assert_eq!(run(&q1_sweep(1, 11), &planes, &mut s, 0..N), want);
        assert_ne!(run(&q1_range(lo, hi), &planes, &mut s, 0..N), want);
    }

    /// FAILS IF: any Q3 arm differs from the oracle, for any shape, either
    /// distribution, on groups of every size.
    #[test]
    fn q3_arms_match_the_oracle() {
        for &skewed in &[false, true] {
            let m = members(skewed, 5);
            let csr = build_csr(&m);
            let gp = build_group_planes(&m);
            let q = Q3 { m, csr, gp };
            let mut s = Scratch::for_program(&q3_scan(1, 0, N, Some(1)), N).expect("scratch");
            for g in [1u32, 2, 17, 500, GROUPS] {
                for (lo, hi, t) in [
                    (0, N, None),
                    (4096, 8192, None),
                    (0, N, Some(100)),
                    (100, 3000, Some(500)),
                ] {
                    let want = q3_oracle(&q.m, g, lo, hi, t);
                    assert_eq!(q.scan(g, lo, hi, t, &mut s), want, "scan g={g}");
                    assert_eq!(
                        q.csr.count(g, lo, hi, t.map(|t| (q.m.v.as_slice(), t))),
                        want,
                        "csr"
                    );
                    assert_eq!(q.inverse_planes(g, lo, hi, t, &mut s), want, "planes");
                }
            }
            // Anti-vacuity: groups of different sizes, and skew visible.
            let big = q3_oracle(&q.m, 1, 0, N, None);
            let small = q3_oracle(&q.m, GROUPS, 0, N, None);
            if skewed {
                assert!(
                    big > 20 * small.max(1),
                    "zipf must be skewed: {big} vs {small}"
                );
            } else {
                assert!(big > 0 && small > 0);
            }
        }
    }

    /// FAILS IF: Q2's three forms disagree with the row word's popcount.
    #[test]
    fn q2_forms_agree() {
        let (w, m) = coverage(5);
        let masks: [&[u64]; THOUGHTS] = std::array::from_fn(|t| m[t].as_slice());
        let planes = Planes {
            n_rows: N,
            masks: &masks,
            lanes: &[],
        };
        let mut s = Scratch::for_program(&q2_prog(0, 0), N).expect("scratch");
        for row in [0usize, 63, 64, 4097, N - 1] {
            let want = u64::from(w[row].count_ones());
            let p: u64 = (0..THOUGHTS as u16)
                .map(|t| run(&q2_prog(t, row), &planes, &mut s, row..row + 1))
                .sum();
            assert_eq!(p, want);
        }
    }
}
