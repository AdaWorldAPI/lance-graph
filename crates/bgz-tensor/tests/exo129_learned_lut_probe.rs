//! EXO-129: does a 256×256 table learned from observed one-hop transitions
//! compose over k hops, and does its purity predict how well? Pre-registered
//! in `.claude/board/entries/2026-10-23-exo129-learned-lut-preregistration.md`.
//!
//! The learned table has `PaletteLut`'s shape (row-major `a << 8 | b`, one
//! byte per cell). It is a plain array here because bgz-tensor does not
//! depend on cognitive-shader-driver.

use bgz_tensor::attention::ComposeTable;
use bgz_tensor::palette::WeightPalette;
use bgz_tensor::projection::{Base17, BASE_DIM};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn sym(&mut self, r: i32) -> i32 {
        (self.next() % (2 * r as u64 + 1)) as i32 - r
    }
    fn upto(&mut self, n: i32) -> i32 {
        (self.next() % (n as u64 + 1)) as i32
    }
    fn idx(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn random_row(rng: &mut Lcg, r: i32) -> Base17 {
    let mut dims = [0i16; BASE_DIM];
    for d in dims.iter_mut() {
        *d = rng.sym(r) as i16;
    }
    Base17 { dims }
}

/// P-S2's clustered generator, unchanged.
fn clustered(rng: &mut Lcg) -> Vec<Base17> {
    let centres: Vec<Base17> = (0..64).map(|_| random_row(rng, 1500)).collect();
    let mut rows = Vec::with_capacity(1024);
    for c in &centres {
        for _ in 0..16 {
            let mut dims = [0i16; BASE_DIM];
            for (d, x) in dims.iter_mut().zip(c.dims.iter()) {
                *d = (i32::from(*x) + rng.sym(120)).clamp(-32768, 32767) as i16;
            }
            rows.push(Base17 { dims });
        }
    }
    rows
}

/// P-S2's uniform generator, unchanged.
fn uniform(rng: &mut Lcg) -> Vec<Base17> {
    (0..1024).map(|_| random_row(rng, 1500)).collect()
}

const BASIS: i16 = 4096;

fn group_entry(g: u8) -> Base17 {
    let mut dims = [0i16; BASE_DIM];
    for (i, d) in dims.iter_mut().enumerate().take(8) {
        if g >> i & 1 == 1 {
            *d = BASIS;
        }
    }
    Base17 { dims }
}

/// W3: the palette is the XOR group itself; members carry non-negative noise.
fn group_world(noise: i32, rng: &mut Lcg) -> (WeightPalette, Vec<Base17>) {
    let entries: Vec<Base17> = (0..=255u8).map(group_entry).collect();
    let mut rows = Vec::with_capacity(1024);
    for e in &entries {
        for _ in 0..4 {
            let mut dims = e.dims;
            for d in dims.iter_mut() {
                *d += rng.upto(noise) as i16;
            }
            rows.push(Base17 { dims });
        }
    }
    let palette = WeightPalette {
        radii: vec![0; 256],
        counts: vec![4; 256],
        entries,
    };
    (palette, rows)
}

struct World {
    name: String,
    palette: WeightPalette,
    rows: Vec<Base17>,
    compose: ComposeTable,
}

fn worlds() -> Vec<World> {
    let mut rng = Lcg(0x0005_2C0D);
    let mut out = Vec::new();
    for (name, rows) in [
        ("W1 clustered", clustered(&mut rng)),
        ("W2 uniform", uniform(&mut rng)),
    ] {
        let palette = WeightPalette::build(&rows, 256);
        assert_eq!(palette.len(), 256, "{name}: palette is not full");
        let compose = ComposeTable::build(&palette);
        out.push(World {
            name: name.to_string(),
            palette,
            rows,
            compose,
        });
    }
    let mut grng = Lcg(0x0129_0003);
    for n in [0, 500, 1500, 3000] {
        let (palette, rows) = group_world(n, &mut grng);
        let compose = ComposeTable::build(&palette);
        out.push(World {
            name: format!("W3 group n={n}"),
            palette,
            rows,
            compose,
        });
    }
    out
}

const HOPS: usize = 6;
const TRAIN: usize = 20_000;
const TEST: usize = 2_000;

fn snap(w: &World, v: &Base17) -> u8 {
    w.palette.assign(v).index
}

struct Learned {
    table: Vec<u8>,
    purity: f64,
    unobserved: f64,
}

fn learn(w: &World) -> Learned {
    let mut counts = vec![0u32; 65_536 * 256];
    let mut rng = Lcg(0xA129_0001);
    let mut total = 0u64;
    for _ in 0..TRAIN {
        let mut x = w.rows[rng.below(w.rows.len())].clone();
        let mut s = snap(w, &x);
        for _ in 0..HOPS {
            let b = rng.idx();
            x = x.xor_bind(&w.palette.entries[b as usize]);
            let t = snap(w, &x);
            counts[((s as usize) << 8 | b as usize) * 256 + t as usize] += 1;
            total += 1;
            s = t;
        }
    }
    let mut table = vec![0u8; 65_536];
    let mut modal_hits = 0u64;
    let mut unobserved = 0usize;
    for cell in 0..65_536usize {
        let row = &counts[cell * 256..cell * 256 + 256];
        let (best, &n) = row
            .iter()
            .enumerate()
            .max_by_key(|&(i, &c)| (c, std::cmp::Reverse(i)))
            .unwrap();
        if n == 0 {
            unobserved += 1;
            table[cell] = w.compose.compose((cell >> 8) as u8, cell as u8);
        } else {
            table[cell] = best as u8;
            modal_hits += u64::from(n);
        }
    }
    Learned {
        table,
        purity: modal_hits as f64 / total as f64,
        unobserved: unobserved as f64 / 65_536.0,
    }
}

/// Agreement of a table-driven chain with the truth, k = 1..HOPS.
fn agreement(w: &World, step: &dyn Fn(u8, u8) -> u8) -> Vec<f64> {
    let mut rng = Lcg(0xB129_0002);
    (1..=HOPS)
        .map(|k| {
            let hits = (0..TEST)
                .filter(|_| {
                    let x0 = w.rows[rng.below(w.rows.len())].clone();
                    let keys: Vec<u8> = (0..k).map(|_| rng.idx()).collect();
                    let truth = snap(
                        w,
                        &keys.iter().fold(x0.clone(), |x, &b| {
                            x.xor_bind(&w.palette.entries[b as usize])
                        }),
                    );
                    let arm = keys.iter().fold(snap(w, &x0), |s, &b| step(s, b));
                    arm == truth
                })
                .count();
            hits as f64 / TEST as f64
        })
        .collect()
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &t in &idx[i..=j] {
            r[t] = avg;
        }
        i = j + 1;
    }
    r
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = a.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let cov: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = ra.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = rb.iter().map(|y| (y - mb).powi(2)).sum();
    if va == 0.0 || vb == 0.0 {
        return 0.0;
    }
    cov / (va * vb).sqrt()
}

fn fmt(v: &[f64]) -> String {
    v.iter()
        .map(|a| format!("{a:.3}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn learned_lut_under_multi_hop_composition() {
    let mut purities = Vec::new();
    let mut agree2 = Vec::new();
    let mut verdict_w12 = Vec::new();
    for w in worlds() {
        let l = learn(&w);
        let lt = l.table.clone();
        let al = agreement(&w, &|a, b| lt[(a as usize) << 8 | b as usize]);
        let ac = agreement(&w, &|a, b| w.compose.compose(a, b));

        // D1: shuffle the learned bytes across cells.
        let mut shuffled = l.table.clone();
        let mut srng = Lcg(0xD129_0001);
        for i in (1..shuffled.len()).rev() {
            shuffled.swap(i, srng.below(i + 1));
        }
        let ad1 = agreement(&w, &|a, b| shuffled[(a as usize) << 8 | b as usize]);

        println!(
            "{}: purity {:.4}  unobserved {:.4}",
            w.name, l.purity, l.unobserved
        );
        println!("{}: L k=1..6 {}", w.name, fmt(&al));
        println!("{}: C k=1..6 {}", w.name, fmt(&ac));
        println!(
            "{}: P^k {}",
            w.name,
            fmt(&(1..=HOPS)
                .map(|k| l.purity.powi(k as i32))
                .collect::<Vec<_>>())
        );
        println!("{}: D1 shuffled k=2 {:.3}", w.name, ad1[1]);

        // Rule 4, D1: a shuffled table must not compose.
        assert!(
            ad1[1] <= 0.10,
            "{}: D1 shuffled table still agrees {:.3}",
            w.name,
            ad1[1]
        );
        // Rule 1: the harness control.
        if w.name == "W3 group n=0" {
            assert!(
                al.iter().chain(&ac).all(|&a| a >= 0.99),
                "harness control failed: L {} C {}",
                fmt(&al),
                fmt(&ac)
            );
        }
        if w.name.starts_with("W1") || w.name.starts_with("W2") {
            verdict_w12.push(al[1]);
        }
        purities.push(l.purity);
        agree2.push(al[1]);
    }
    let rho = spearman(&purities, &agree2);
    println!("M: spearman(purity, agreement_L(2)) over 6 settings = {rho:.3}");
    let torque = if verdict_w12.iter().all(|&a| a >= 0.90) {
        "PROMOTE"
    } else if verdict_w12.iter().all(|&a| a < 0.90) {
        "KILL"
    } else {
        "MIXED"
    };
    let m = if rho >= 0.9 { "PASS" } else { "FAIL" };
    println!("EXO-129 torque verdict: {torque}; mechanism M: {m}");

    // Pinned results (2026-10-23). A change in either fails here, which means
    // the probe must be re-registered, not that the pin may be edited.
    assert_eq!(
        torque, "KILL",
        "the learned table now composes in W1/W2; re-register EXO-129"
    );
    assert_eq!(
        m, "PASS",
        "purity no longer ranks k=2 agreement; re-register EXO-129"
    );
}

/// Rule 4, D2: with L := C the harness reproduces C exactly.
#[test]
fn compose_table_as_learned_reproduces_c() {
    for w in worlds() {
        let ac = agreement(&w, &|a, b| w.compose.compose(a, b));
        let table: Vec<u8> = (0..65_536usize)
            .map(|c| w.compose.compose((c >> 8) as u8, c as u8))
            .collect();
        let al = agreement(&w, &|a, b| table[(a as usize) << 8 | b as usize]);
        assert_eq!(ac, al, "{}: D2 L:=C differs from C", w.name);
    }
}

/// W3's palette really is an XOR group: the closure the control relies on.
#[test]
fn group_world_is_closed_under_bind() {
    for a in 0..=255u8 {
        for b in 0..=255u8 {
            assert_eq!(
                group_entry(a).xor_bind(&group_entry(b)).dims,
                group_entry(a ^ b).dims
            );
        }
    }
}
