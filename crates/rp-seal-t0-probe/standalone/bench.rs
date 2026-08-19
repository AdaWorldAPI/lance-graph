use std::collections::BTreeMap;
use std::time::Instant;

const ROW: usize = 512;

#[inline(never)]
fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(PRIME);
    }
    h
}

// 8 independent FNV lanes over interleaved bytes — an ILP-friendly variant
// (NOT the same digest; measures the dependency-chain ceiling only).
#[inline(never)]
fn fnv1a_8lane(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = [OFFSET; 8];
    let n = bytes.len() / 8 * 8;
    let mut i = 0;
    while i < n {
        for l in 0..8 {
            h[l] ^= u64::from(bytes[i + l]);
            h[l] = h[l].wrapping_mul(PRIME);
        }
        i += 8;
    }
    h.iter().fold(0u64, |a, b| a ^ b)
}

#[inline(never)]
fn xor_bytes(dst: &mut [u8], src: &[u8]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d ^= *s;
    }
}

fn main() {
    let c: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(65536);
    let d_frac: f64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let d = ((c as f64) * d_frac) as usize;
    eprintln!("C(casts)={c} D(distinct rows)={d} row={ROW}B payload_total={} MiB", (c*ROW) >> 20);

    // ---- payload pool -------------------------------------------------
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(c);
    for i in 0..c {
        let mut v = vec![0u8; ROW];
        v[0] = (i & 0xff) as u8;
        v[1] = ((i >> 8) & 0xff) as u8;
        payloads.push(v);
    }
    let flat: Vec<u8> = vec![7u8; c * ROW];

    // ---- 1. FNV-1a byte-at-a-time (the actual content_hash inner loop) --
    let t = Instant::now();
    let h = fnv1a(&flat);
    let e = t.elapsed();
    println!("fnv1a_serial      {:>9.2} ms  {:>7.2} GB/s (h={h:#x})", e.as_secs_f64()*1e3, flat.len() as f64/ e.as_secs_f64()/1e9);

    let t = Instant::now();
    let h = fnv1a_8lane(&flat);
    let e = t.elapsed();
    println!("fnv1a_8lane_ILP   {:>9.2} ms  {:>7.2} GB/s (h={h:#x})", e.as_secs_f64()*1e3, flat.len() as f64/ e.as_secs_f64()/1e9);

    // ---- 2. memcpy baseline (Vec clone of the whole payload set) --------
    let t = Instant::now();
    let cloned: Vec<Vec<u8>> = payloads.clone();
    let e = t.elapsed();
    println!("clone_payloads    {:>9.2} ms  {:>7.2} GB/s ({} allocs)", e.as_secs_f64()*1e3, (c*ROW) as f64/e.as_secs_f64()/1e9, cloned.len());
    std::hint::black_box(&cloned);

    // ---- 3. freeze image: BTreeMap insert with per-cast payload clone ----
    let t = Instant::now();
    let mut image: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
    for (i, p) in payloads.iter().enumerate() {
        image.insert((i % d.max(1)) as u64, p.clone());
    }
    let e = t.elapsed();
    println!("freeze_image      {:>9.2} ms  ({} live rows of {} inserts)", e.as_secs_f64()*1e3, image.len(), c);

    // ---- 4. sort_by_key over SweepSlot-sized structs ---------------------
    #[allow(dead_code)]
    struct Slot { cycle: u64, sp: u64, owner: u32, row: u64, mv: Option<[u8;16]>, payload: Vec<u8> }
    let mut slots: Vec<Slot> = (0..c).map(|i| Slot {
        cycle: 1, sp: ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 20, owner: (i%4096) as u32,
        row: (i % d.max(1)) as u64, mv: None, payload: Vec::new() }).collect();
    let t = Instant::now();
    slots.sort_by_key(|s| s.sp);
    let e = t.elapsed();
    println!("sort_slots        {:>9.2} ms  (size_of Slot = {}B)", e.as_secs_f64()*1e3, std::mem::size_of::<Slot>());
    std::hint::black_box(&slots);

    // ---- 5. arrow FixedSizeBinary null padding (zero-fill per landing) ---
    let t = Instant::now();
    let mut values: Vec<u8> = Vec::with_capacity((1 + c + d) * ROW);
    values.extend(std::iter::repeat_n(0u8, ROW));           // frame row
    for _ in 0..c { values.extend(std::iter::repeat_n(0u8, ROW)); }  // landing nulls
    for (_, p) in image.iter() { values.extend_from_slice(p); }      // image values
    let e = t.elapsed();
    println!("arrow_payload_col {:>9.2} ms  ({} MiB buffer, {} MiB of it zeros)",
        e.as_secs_f64()*1e3, values.len()>>20, ((1+c)*ROW)>>20);
    std::hint::black_box(&values);

    // ---- 6. XOR parity over a stripe (RS P-parity inner loop) ------------
    let mut par = vec![0u8; ROW];
    let t = Instant::now();
    for p in payloads.iter() { xor_bytes(&mut par, p); }
    let e = t.elapsed();
    println!("xor_parity_scalar {:>9.2} ms  {:>7.2} GB/s", e.as_secs_f64()*1e3, (c*ROW) as f64/e.as_secs_f64()/1e9);
    std::hint::black_box(&par);
}
