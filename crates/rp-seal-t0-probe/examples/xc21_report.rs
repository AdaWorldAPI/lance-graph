//! The X-C2-1 report run: full matrix (both hash schemes × I1–I9 with the
//! cascade-congruent I9 strides), the charter 10⁶-trial null control, and
//! one full-size (65,536 × 512 B = 32 MiB) pass. Counts only — no
//! wall-clock quantity is measured or reported (T0.3).

use rp_seal_t0_probe::*;

fn main() {
    let schemes: [&dyn Scheme; 2] = [&S1Unbound, &S6Bound];
    let n = 4096;
    println!(
        "X-C2-1 injection matrix  (n = {n} chunks × {CHUNK_BYTES} B, multiplicity 1 unless stated)"
    );
    println!("FA = false-accepted slots / affected;  alarms = clean slots flagged\n");
    for s in schemes {
        println!("── {}", s.name());
        let mut cases: Vec<(String, Injection)> = vec![
            ("I1 erasure".into(), Injection::I1 { pos: 100 }),
            ("I2 double erasure".into(), Injection::I2 { a: 7, b: 3000 }),
            (
                "I3 silent corruption".into(),
                Injection::I3 { pos: 555, bit: 77 },
            ),
            ("I4 wrong-slot".into(), Injection::I4 { src: 11, dst: 2222 }),
            (
                "I5 stale".into(),
                Injection::I5 {
                    pos: 900,
                    old_version: 1,
                },
            ),
            ("I6 duplicate".into(), Injection::I6 { src: 42, dst: 4000 }),
            (
                "I7 domain (32)".into(),
                Injection::I7 {
                    start: 512,
                    len: 32,
                    bit: 9,
                },
            ),
            (
                "I8 boundary (g=64)".into(),
                Injection::I8 { group: 64, bit: 3 },
            ),
        ];
        for stride in [4usize, 16, 64, 256, 1024, 4096] {
            cases.push((
                format!("I9 phase-aligned (stride {stride})"),
                Injection::I9 {
                    stride,
                    phase: 1 % stride,
                    bit: 21,
                },
            ));
        }
        for (label, inj) in &cases {
            let (affected, _, fa, alarms) = run_one(s, n, inj);
            println!(
                "  {label:<28} affected {:>5}   FA {:>5}   alarms {:>3}",
                affected.len(),
                fa.len(),
                alarms.len()
            );
        }
        println!();
    }

    println!("null control (charter: 10⁶ DISTINCT clean chunks per scheme):");
    for s in schemes {
        let spurious = null_control(s, 1_000_000);
        println!("  {:<44} spurious flags: {spurious}", s.name());
        assert_eq!(spurious, 0, "a scheme that flags clean cycles is broken");
    }

    println!("\nfull-size pass (n = {FULL_CHUNKS} = 32 MiB), I9 stride 4096 under S6:");
    let (affected, _, fa, alarms) = run_one(
        &S6Bound,
        FULL_CHUNKS,
        &Injection::I9 {
            stride: 4096,
            phase: 5,
            bit: 63,
        },
    );
    println!(
        "  affected {}   FA {}   alarms {}",
        affected.len(),
        fa.len(),
        alarms.len()
    );
    assert_eq!(fa.len(), 0);
    assert_eq!(alarms.len(), 0);
    println!("\nOK — ground truth is the injection record; schemes S2–S5 plug in via X-C2-3.");
}
