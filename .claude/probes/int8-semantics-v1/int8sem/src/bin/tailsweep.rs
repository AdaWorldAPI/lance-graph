//! D-TFLP Amendment 6.4 follow-up: N-tail / K-tail sweep of every dot arm against the i64 oracle.
//!
//! The 6.4 `arms` run used n = 16 everywhere (plus n = 1 in the extreme pairs), so the N-tail
//! padding of the VPDPBUSD paths (`int8_tile_gemm.rs:169-186`, padded lanes get weight 0 and are
//! dropped at the store) was barely exercised. This sweep covers m in {1, 2, 3, 16, 17},
//! n in 1..=33, k in {1, 2, 3, 4, 5, 31, 63, 64, 65, 128, 129}, each shape with three inputs:
//! uniform random codes over the full -128..127 range, all (-128, -128), all (127, -128).
//! Every arm must equal the oracle; `uncorrected_u8xi8` is the deliberately wrong negative control
//! (no 128*colsum correction) and must differ wherever the column sum is non-zero.
//! Usage: tailsweep OUT.json
use int8sem::*;
use std::collections::BTreeMap;

fn main() {
    let out = std::env::args().nth(1).expect("OUT.json");
    let ms = [1usize, 2, 3, 16, 17];
    let ks = [1usize, 2, 3, 4, 5, 31, 63, 64, 65, 128, 129];
    let mut rng = Rng::new(0x647A_115E_ED64);
    // arm -> (cases run, cells, mismatches, not run)
    let mut st: BTreeMap<&str, [u64; 4]> = BTreeMap::new();
    let mut tiers: BTreeMap<&str, u64> = BTreeMap::new();
    let mut ctrl_should_differ = 0u64;
    let mut first_bad: Vec<String> = Vec::new();
    for &m in &ms {
        for n in 1..=33usize {
            for &k in &ks {
                for fill in 0..3 {
                    let (a, w): (Vec<i8>, Vec<i8>) = match fill {
                        0 => ((0..m * k).map(|_| rng.next_u32() as i8).collect(), (0..n * k).map(|_| rng.next_u32() as i8).collect()),
                        1 => (vec![-128; m * k], vec![-128; n * k]),
                        _ => (vec![127; m * k], vec![-128; n * k]),
                    };
                    let wt = transpose(&w, n, k);
                    let d = DotInputs { a: &a, w_rows: &w, wt: &wt, m, n, k };
                    let orc = oracle(&d);
                    *tiers.entry(matmul_tier(m, n, k)).or_default() += 1;
                    let colsum_nonzero = (0..n).filter(|&c| w[c * k..(c + 1) * k].iter().map(|&x| x as i64).sum::<i64>() != 0).count() as u64;
                    ctrl_should_differ += colsum_nonzero * m as u64;
                    for arm in DOT_ARMS {
                        let e = st.entry(arm).or_default();
                        match run_dot_arm(arm, &d) {
                            ArmOut::NotRun(_) => e[3] += 1,
                            ArmOut::Cells(v) => {
                                e[0] += 1;
                                e[1] += v.len() as u64;
                                for (i, (&g, &o)) in v.iter().zip(orc.iter()).enumerate() {
                                    if g != o {
                                        e[2] += 1;
                                        if arm != "uncorrected_u8xi8" && first_bad.len() < 10 {
                                            first_bad.push(format!("{arm} m={m} n={n} k={k} fill={fill} cell={i} got={g} oracle={o}"));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let mut s = String::from("{\n \"shapes\": {\"m\": [1,2,3,16,17], \"n\": \"1..=33\", \"k\": [1,2,3,4,5,31,63,64,65,128,129], \"fills\": [\"uniform\", \"(-128,-128)\", \"(127,-128)\"]},\n");
    s += " \"matmul_i8_to_i32_tier_by_case_count\": {";
    s += &tiers.iter().map(|(k, v)| format!("\"{k}\": {v}")).collect::<Vec<_>>().join(", ");
    s += "},\n \"arms\": {\n";
    s += &st
        .iter()
        .map(|(a, e)| format!("  \"{a}\": {{\"cases_run\": {}, \"cells\": {}, \"mismatches_vs_i64_oracle\": {}, \"not_run\": {}}}", e[0], e[1], e[2], e[3]))
        .collect::<Vec<_>>()
        .join(",\n");
    s += &format!("\n }},\n \"control_cells_with_nonzero_colsum\": {ctrl_should_differ},\n \"first_unexpected_mismatches\": [{}]\n}}\n",
        first_bad.iter().map(|x| format!("\"{x}\"")).collect::<Vec<_>>().join(", "));
    std::fs::write(&out, &s).unwrap();
    print!("{s}");
}
