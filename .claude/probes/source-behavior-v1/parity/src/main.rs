//! Amendment 6.2 lookup/admission probe. Reads the prepared artifacts written by
//! `prepare.py` (never regenerates them) and links NO source runtime: no TFLM, no
//! exp/tanh. Execution is `table[input_code as u8 as usize]`.
//!
//! Artifact layout: b"PAL256LAB1", u32 n, then per calibration:
//!   32 receipt-hash SourceParity | 32 receipt-hash CertifiedMath | 256 src | 256 math
//!   | 32 sha256(src) | 32 sha256(math) | u16 k | k x (u8 index, i8 source byte)
use sha2::{Digest, Sha256};
use std::hint::black_box;
use std::time::Instant;

struct Cal { rh_src: [u8; 32], rh_math: [u8; 32], src: [i8; 256], math: [i8; 256], th_src: [u8; 32], th_math: [u8; 32], patch: Vec<(u8, i8)> }

fn parse(b: &[u8]) -> Vec<Cal> {
    assert_eq!(&b[..10], b"PAL256LAB1");
    let n = u32::from_le_bytes(b[10..14].try_into().unwrap()) as usize;
    let mut p = 14; let mut out = Vec::with_capacity(n);
    let take = |p: &mut usize, k: usize| { let s = &b[*p..*p + k]; *p += k; s };
    for _ in 0..n {
        let rh_src = take(&mut p, 32).try_into().unwrap(); let rh_math = take(&mut p, 32).try_into().unwrap();
        let mut src = [0i8; 256]; let mut math = [0i8; 256];
        for (d, s) in src.iter_mut().zip(take(&mut p, 256)) { *d = *s as i8; }
        for (d, s) in math.iter_mut().zip(take(&mut p, 256)) { *d = *s as i8; }
        let th_src = take(&mut p, 32).try_into().unwrap(); let th_math = take(&mut p, 32).try_into().unwrap();
        let k = u16::from_le_bytes(take(&mut p, 2).try_into().unwrap()) as usize;
        let patch = (0..k).map(|_| { let e = take(&mut p, 2); (e[0], e[1] as i8) }).collect();
        out.push(Cal { rh_src, rh_math, src, math, th_src, th_math, patch });
    }
    assert_eq!(p, b.len(), "trailing bytes");
    out
}

fn sha(t: &[i8; 256]) -> [u8; 32] { let mut h = Sha256::new(); h.update(t.map(|v| v as u8)); h.finalize().into() }

#[derive(Debug, PartialEq)]
enum Refusal { NoArtifactForReceipt, TableDigestMismatch }

/// Lab admission: the caller names the semantic receipt it requires; the table is
/// returned only if that receipt was prepared AND the table still hashes to its digest.
fn admit<'a>(cals: &'a [Cal], want: &[u8; 32]) -> Result<&'a [i8; 256], Refusal> {
    for c in cals {
        if &c.rh_src == want { return if sha(&c.src) == c.th_src { Ok(&c.src) } else { Err(Refusal::TableDigestMismatch) }; }
        if &c.rh_math == want { return if sha(&c.math) == c.th_math { Ok(&c.math) } else { Err(Refusal::TableDigestMismatch) }; }
    }
    Err(Refusal::NoArtifactForReceipt)
}

fn main() {
    let path = std::env::args().nth(1).expect("artifacts.bin");
    let bytes = std::fs::read(&path).expect("read");
    let t = Instant::now(); let mut cals = parse(&bytes); let t_parse = t.elapsed();
    // 1. cold validation: every table digest
    let t = Instant::now();
    let bad = cals.iter().filter(|c| sha(&c.src) != c.th_src || sha(&c.math) != c.th_math).count();
    let t_validate = t.elapsed();
    // 2. reconstruction from CertifiedMath + sparse patch must equal SourceParity bit for bit
    // DISABLE=patch drops each patch's last entry; DISABLE=xor drops the ^0x80 storage mapping.
    let disable = std::env::var("DISABLE").unwrap_or_default();
    let (mut recon_bad, mut diff_cells, mut patched) = (0, 0, 0);
    for c in &cals {
        let k = if disable == "patch" { c.patch.len().saturating_sub(1) } else { c.patch.len() };
        let mut r = c.math; for &(i, v) in &c.patch[..k] { r[i as usize] = v; }
        recon_bad += (r != c.src) as usize;
        diff_cells += c.src.iter().zip(c.math.iter()).filter(|(a, b)| a != b).count();
        patched += !c.patch.is_empty() as usize;
    }
    // 3. exhaustive index mapping: position i of a table is input code i - 128. The runtime
    //    index from an i8 code (q + 128) must equal the index from Codec8 U8 storage (code ^ 0x80)
    //    for all 256 codes of every table (a wrong mapping is the realistic lookup bug).
    let mut lookup_bad = 0;
    for c in &cals { for q in -128i32..128 {
        let from_i8 = (q + 128) as usize;
        let from_u8_storage = if disable == "xor" { (q as i8 as u8) as usize } else { ((q as i8 as u8) ^ 0x80) as usize };
        lookup_bad += (c.src[from_i8] != c.src[from_u8_storage] || from_i8 != from_u8_storage) as usize; } }
    // 4. admission controls
    let rh0 = cals.iter().position(|c| !c.patch.is_empty()).unwrap();
    let eq = cals.iter().position(|c| c.patch.is_empty()).unwrap();
    let ok_src = admit(&cals, &cals[rh0].rh_src).map(|t| *t == cals[rh0].src);
    let ok_math = admit(&cals, &cals[rh0].rh_math).map(|t| *t == cals[rh0].math);
    // asking for SourceParity on a byte-equal calibration returns the SOURCE artifact, never the math one
    let swap = admit(&cals, &cals[eq].rh_src).map(|t| std::ptr::eq(t, &cals[eq].src));
    let mut unknown = cals[rh0].rh_src; unknown[0] ^= 1;
    let unk = admit(&cals, &unknown).map(|_| ());
    let saved = cals[rh0].src[17]; cals[rh0].src[17] = saved.wrapping_add(1);
    let corrupt = admit(&cals, &cals[rh0].rh_src).map(|_| ());
    cals[rh0].src[17] = saved;
    let restored = admit(&cals, &cals[rh0].rh_src).map(|_| ());
    // 5. timing: 16 Mi input bytes, xorshift, release; median of 7
    let n = 1 << 24; let mut x: u64 = 0x9E3779B97F4A7C15;
    let inp: Vec<u8> = (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 32) as u8 }).collect();
    let mut out = vec![0i8; n];
    let med = |f: &mut dyn FnMut()| { let mut v: Vec<f64> = (0..7).map(|_| { let t = Instant::now(); f(); t.elapsed().as_nanos() as f64 / n as f64 }).collect(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[3] };
    let tab = cals[rh0].src;
    let t_direct = med(&mut || { for (o, &q) in out.iter_mut().zip(&inp) { *o = tab[q as usize]; } black_box(&out); });
    let base = cals[rh0].math; let patch = cals[rh0].patch.clone();
    let t_overlay = med(&mut || { for (o, &q) in out.iter_mut().zip(&inp) {
        let mut v = base[q as usize]; for &(i, s) in &patch { if i == q { v = s; } } *o = v; } black_box(&out); });
    let t_recon = { let t = Instant::now(); for _ in 0..1000 { let mut r = black_box(base); for &(i, s) in &patch { r[i as usize] = s; } black_box(r); } t.elapsed().as_nanos() as f64 / 1000.0 };
    let t_sha = { let t = Instant::now(); for _ in 0..1000 { black_box(sha(black_box(&tab))); } t.elapsed().as_nanos() as f64 / 1000.0 };
    println!("calibrations {} parse_ms {:.2} validate_all_ms {:.2} digest_failures {}", cals.len(), t_parse.as_secs_f64() * 1e3, t_validate.as_secs_f64() * 1e3, bad);
    println!("reconstruct_failures {recon_bad} source_vs_math_diff_cells {diff_cells} calibrations_with_patch {patched} exhaustive_lookup_failures {lookup_bad}");
    println!("admit SourceParity {:?} | CertifiedMath {:?} | equal-bytes SourceParity returns source artifact {:?}", ok_src, ok_math, swap);
    println!("admit unknown receipt {:?} | corrupted byte {:?} | restored {:?}", unk, corrupt, restored);
    println!("ns/elem: direct lookup {:.3} | math+sparse overlay lookup {:.3} | one-time reconstruct ns {:.1} | sha256 of one 256-byte table ns {:.1}", t_direct, t_overlay, t_recon, t_sha);
}
