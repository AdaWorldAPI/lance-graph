//! P-S2: does a Palette256 compose table survive multi-hop composition?
//! Pre-registered in
//! `.claude/board/entries/2026-10-23-ps2-compose-chain-preregistration.md`.
//!
//! `ComposeTable` stores, for each pair `(a, b)`, the palette index nearest to
//! `entry[a].xor_bind(entry[b])`. A chain feeds each looked-up index into the
//! next lookup, so every hop snaps back onto the palette. XOR bind is
//! associative and its own inverse; this probe measures how much of that the
//! table keeps.
//!
//! Synthetic worlds, production code: a mechanism probe, not a claim about
//! real weights.

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
    fn idx(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
}

fn random_row(rng: &mut Lcg, r: i32) -> Base17 {
    let mut dims = [0i16; BASE_DIM];
    for d in dims.iter_mut() {
        *d = rng.sym(r) as i16;
    }
    Base17 { dims }
}

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

fn uniform(rng: &mut Lcg) -> Vec<Base17> {
    (0..1024).map(|_| random_row(rng, 1500)).collect()
}

struct World {
    name: &'static str,
    palette: WeightPalette,
    table: ComposeTable,
}

fn worlds() -> Vec<World> {
    let mut rng = Lcg(0x0005_2C0D);
    [
        ("clustered", clustered(&mut rng)),
        ("uniform", uniform(&mut rng)),
    ]
    .into_iter()
    .map(|(name, rows)| {
        let palette = WeightPalette::build(&rows, 256);
        assert_eq!(palette.len(), 256, "{name}: palette is not full");
        let table = ComposeTable::build(&palette);
        World {
            name,
            palette,
            table,
        }
    })
    .collect()
}

fn nearest(w: &World, v: &Base17) -> u8 {
    w.palette.assign(v).index
}

fn table_chain(w: &World, ops: &[u8]) -> u8 {
    ops[1..]
        .iter()
        .fold(ops[0], |acc, &b| w.table.compose(acc, b))
}

/// One bind per hop in Base17 space, one snap at the end.
fn exact_chain(w: &World, ops: &[u8]) -> u8 {
    let v = ops[1..]
        .iter()
        .fold(w.palette.entries[ops[0] as usize].clone(), |acc, &b| {
            acc.xor_bind(&w.palette.entries[b as usize])
        });
    nearest(w, &v)
}

const CHAINS: usize = 2000;
const MAX_HOPS: usize = 6;

fn agreement(w: &World, chain: fn(&World, &[u8]) -> u8) -> Vec<f64> {
    let mut rng = Lcg(0x5EED_0002);
    (1..=MAX_HOPS)
        .map(|k| {
            let hits = (0..CHAINS)
                .filter(|_| {
                    let ops: Vec<u8> = (0..=k).map(|_| rng.idx()).collect();
                    chain(w, &ops) == exact_chain(w, &ops)
                })
                .count();
            hits as f64 / CHAINS as f64
        })
        .collect()
}

fn unbind_recovery(w: &World, compose: impl Fn(u8, u8) -> u8) -> f64 {
    let mut ok = 0usize;
    for a in 0..=255u8 {
        for b in 0..=255u8 {
            if compose(compose(a, b), b) == a {
                ok += 1;
            }
        }
    }
    let _ = w;
    ok as f64 / 65536.0
}

#[test]
fn compose_table_under_multi_hop_composition() {
    for w in worlds() {
        let agree = agreement(&w, table_chain);
        let recovery = unbind_recovery(&w, |a, b| w.table.compose(a, b));

        let mut images: Vec<usize> = (0..=255u8)
            .map(|b| {
                let mut seen = [false; 256];
                for x in 0..=255u8 {
                    seen[w.table.compose(x, b) as usize] = true;
                }
                seen.iter().filter(|&&s| s).count()
            })
            .collect();
        images.sort_unstable();

        let mut rng = Lcg(0x5EED_0003);
        let mut reach = [true; 256];
        let mut reach_sizes = Vec::new();
        for _ in 0..MAX_HOPS {
            let b = rng.idx();
            let mut next = [false; 256];
            for (x, &r) in reach.iter().enumerate() {
                if r {
                    next[w.table.compose(x as u8, b) as usize] = true;
                }
            }
            reach = next;
            reach_sizes.push(reach.iter().filter(|&&s| s).count());
        }

        println!(
            "{}: agreement(k=1..{MAX_HOPS}) {:?}",
            w.name,
            agree.iter().map(|a| format!("{a:.3}")).collect::<Vec<_>>()
        );
        println!("{}: unbind recovery {recovery:.4}", w.name);
        println!(
            "{}: per-relation image size min {} median {} max {}",
            w.name, images[0], images[128], images[255]
        );
        println!("{}: reachable set after k hops {reach_sizes:?}", w.name);
        let stable = agree[1] >= 0.90 && recovery >= 0.90;
        println!(
            "{}: verdict {}",
            w.name,
            if stable { "STABLE" } else { "UNSTABLE" }
        );

        // Harness guard: one hop is one bind and one snap.
        assert_eq!(agree[0], 1.0, "{}: harness guard failed", w.name);
        // Pinned result (2026-10-23): the table does not survive two hops.
        assert!(
            !stable,
            "{}: the compose table now passes; re-register P-S2",
            w.name
        );
    }
}

/// Post-hoc diagnostic (not pre-registered): is a bound pair still inside the
/// codebook's support? P-D3's radius flag (`distortion > radii[index]`),
/// applied to the bind result before it is snapped.
#[test]
fn bound_pairs_fall_outside_the_codebook_support() {
    for w in worlds() {
        let mut outside = 0usize;
        for a in 0..256usize {
            for b in 0..256usize {
                let v = w.palette.entries[a].xor_bind(&w.palette.entries[b]);
                let s = w.palette.assign(&v);
                if s.distortion > w.palette.radii[s.index as usize] {
                    outside += 1;
                }
            }
        }
        let share = outside as f64 / 65536.0;
        println!("{}: bound pairs outside support {share:.4}", w.name);
    }
}

/// The measures can report stability: the exact chain agrees with itself.
#[test]
fn agreement_reads_one_for_the_exact_chain() {
    for w in worlds() {
        assert!(agreement(&w, exact_chain).iter().all(|&a| a == 1.0));
    }
}

/// The measures can report failure: a constant law recovers almost nothing.
#[test]
fn unbind_recovery_collapses_for_a_constant_law() {
    for w in worlds() {
        let r = unbind_recovery(&w, |_, _| 0);
        assert!(r < 0.01, "{}: constant law recovered {r}", w.name);
    }
}
