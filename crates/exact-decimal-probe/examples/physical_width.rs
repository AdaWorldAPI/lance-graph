//! D-AED-2 measurement: encoded cost per value of a sealed slice at PM9,
//! PM17 or `i32`: a notional descriptor, the magnitudes and the polarity or
//! presence plane. `Vec` headers and struct padding are not counted, for
//! either column.
//! Run: `cargo run --release --example physical_width`.

use exact_decimal_probe::{seal, Exp, PhysicalSlice, SealedSlice, DESCRIPTOR_BYTES};

/// Encoded cost of the same slice as `i32` coefficients plus a presence
/// plane, counted the same way.
fn i32_bytes(s: &SealedSlice) -> usize {
    DESCRIPTOR_BYTES + s.coefficients.len() * 4 + s.coefficients.len().div_ceil(64) * 8
}

fn row(name: &str, slices: &[SealedSlice]) {
    let values: usize = slices.iter().map(|s| s.coefficients.len()).sum();
    let stored: usize = slices.iter().map(|s| PhysicalSlice::store(s).bytes()).sum();
    let wide: usize = slices.iter().map(i32_bytes).sum();
    let widths: Vec<&str> = slices
        .iter()
        .map(|s| PhysicalSlice::store(s).width())
        .collect();
    for s in slices {
        assert_eq!(PhysicalSlice::store(s).load(), *s, "{name}: round trip");
    }
    println!(
        "{name:<42} {values:>8} values  {:>6.3} B/value encoded  {:>6.3} B/value as i32  widths {widths:?}",
        stored as f64 / values as f64,
        wide as f64 / values as f64,
    );
}

fn main() {
    let s2 = Exp::scale(2);
    let s3 = Exp::scale(3);
    let cells = |n: usize, f: &dyn Fn(usize) -> Option<String>| -> SealedSlice {
        let v: Vec<Option<String>> = (0..n).map(f).collect();
        let r: Vec<Option<&str>> = v.iter().map(|c| c.as_deref()).collect();
        seal(&r, s2).unwrap()
    };
    // The #2182 fixture: one slice of integers, two at finer scales.
    let fixture: Vec<SealedSlice> = [
        vec![Some("250.00"), Some("100"), None, Some("1000"), Some("-50")],
        vec![
            Some("37.5"),
            Some("-12.25"),
            Some("0.00"),
            None,
            Some("0.125"),
        ],
        vec![
            Some("-1"),
            Some("2.5"),
            Some("40"),
            Some("0.5"),
            Some("-0.75"),
        ],
    ]
    .iter()
    .map(|s| seal(s, s3).unwrap())
    .collect();
    row("#2182 fixture (3 slices of 5)", &fixture);

    let n = 1 << 16;
    // Half integers, half one decimal, sealed per half: whole amounts seal
    // at exponent 0, so each half is its own slice.
    let ints = cells(n / 2, &|i| (i % 9 != 0).then(|| format!("{}", i % 200)));
    let tenths = cells(n / 2, &|i| {
        (i % 9 != 0).then(|| format!("{}.{}", i % 20, i % 10))
    });
    row("half integers, half scale 1 (2 slices)", &[ints, tenths]);
    row(
        "PM9 limit: |c| <= 255",
        &[cells(n, &|i| Some(format!("{}", i as i64 % 511 - 255)))],
    );
    row(
        "PM17 limit: |c| <= 65535",
        &[cells(n, &|i| {
            Some(format!("{}", (i as i64 * 7) % 131_071 - 65_535))
        })],
    );
    row(
        "one value past PM17 widens the slice",
        &[cells(n, &|i| {
            Some(if i == 0 {
                "70000".to_owned()
            } else {
                format!("{}", i % 100)
            })
        })],
    );
    row(
        "worst: 1-value slices (descriptor)",
        &(0..64)
            .map(|i| cells(1, &|_| Some(format!("{i}"))))
            .collect::<Vec<_>>(),
    );
    row("worst: all absent", &[cells(n, &|_| None)]);
}
