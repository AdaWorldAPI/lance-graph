use std::collections::BTreeMap;
use std::time::Instant;
const ROW: usize = 512;

fn lcg(s: &mut u64) -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s }

fn main() {
    let n: usize = 65536;              // rows in the cycle image
    let img_bytes = n * ROW;           // 32 MiB
    let mut image = vec![0u8; img_bytes];
    let src = vec![0xABu8; ROW];

    // ---- A. scatter 512B rows into the 32 MiB image: sequential vs random ----
    for &(label, random) in &[("seq_place", false), ("rand_place", true)] {
        let mut s = 0x1234_5678u64;
        let t = Instant::now();
        for i in 0..n {
            let idx = if random { (lcg(&mut s) as usize) % n } else { i };
            image[idx*ROW..idx*ROW+ROW].copy_from_slice(&src);
        }
        let e = t.elapsed();
        println!("{label:<12} {:>8.2} ms  {:>6.2} GB/s", e.as_secs_f64()*1e3, img_bytes as f64/e.as_secs_f64()/1e9);
    }
    std::hint::black_box(&image);

    // ---- B. 4x4 petal tiling (8 KiB work field) sequential vs random petal order ----
    let petal_rows = 16usize; let petals = n / petal_rows;   // 4096 petals of 8 KiB
    for &(label, random) in &[("seq_petal", false), ("rand_petal", true)] {
        let mut s = 0xDEAD_BEEFu64;
        let t = Instant::now();
        let mut acc = 0u64;
        for p in 0..petals {
            let pp = if random { (lcg(&mut s) as usize) % petals } else { p };
            let base = pp * petal_rows * ROW;
            for r in 0..petal_rows {
                // touch the row (read 8B + write 8B) — a "breath" on a hot field
                let o = base + r*ROW;
                let mut b8 = [0u8;8]; b8.copy_from_slice(&image[o..o+8]); acc ^= u64::from_le_bytes(b8);
                image[o..o+8].copy_from_slice(&acc.to_le_bytes());
            }
        }
        let e = t.elapsed();
        println!("{label:<12} {:>8.2} ms  ({} petals x {} rows, acc={acc:#x})", e.as_secs_f64()*1e3, petals, petal_rows);
    }

    // ---- C. BTreeMap image build: sequential vs random row keys ----
    for &(label, random) in &[("btree_seq", false), ("btree_rand", true)] {
        let mut s = 0xF00Du64;
        let t = Instant::now();
        let mut m: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
        for i in 0..n {
            let k = if random { lcg(&mut s) % n as u64 } else { i as u64 };
            m.insert(k, src.clone());
        }
        let e = t.elapsed();
        println!("{label:<12} {:>8.2} ms  ({} live)", e.as_secs_f64()*1e3, m.len());
        std::hint::black_box(&m);
    }

    // ---- D. open-petal buffer RSS: K petals held open, 8 KiB each ----
    for k in [64usize, 256, 1024, 4096] {
        let bufs: Vec<Vec<u8>> = (0..k).map(|_| vec![0u8; petal_rows*ROW]).collect();
        println!("open_petals k={k:<5} resident {:>6.1} MiB (8 KiB each)", (k*petal_rows*ROW) as f64/1048576.0);
        std::hint::black_box(&bufs);
    }

    // ---- E. RS(k=8,m=2) parity RMW for a single dirty row vs full-stripe ----
    // model: stripe = 8 data petals + 2 parity petals, petal = 8 KiB
    let petal = petal_rows*ROW;
    let mut p = vec![0u8; petal]; let mut q = vec![0u8; petal];
    let dat: Vec<Vec<u8>> = (0..8).map(|i| vec![i as u8; petal]).collect();
    let t = Instant::now();
    for _ in 0..1000 {  // RMW: read old data + old P + old Q, xor deltas, write P,Q
        for j in 0..petal { p[j] ^= dat[0][j] ^ dat[1][j]; q[j] ^= dat[0][j]; }
    }
    let e = t.elapsed();
    println!("parity_rmw_1k  {:>8.2} ms  ({:.1} us per single-row RMW over an 8KiB petal)",
        e.as_secs_f64()*1e3, e.as_secs_f64()*1e6/1000.0);
    std::hint::black_box((&p,&q));
}
