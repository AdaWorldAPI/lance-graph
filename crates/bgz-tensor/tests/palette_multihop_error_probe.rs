//! Quantized XOR composition is exact only when the frozen codebook is closed.
//! A table lookup does not itself assert correctness of a multi-hop inference.
use bgz_tensor::{Base17, ComposeTable, WeightPalette};

fn value(n: i16) -> Base17 {
    let mut dims = [0i16; 17];
    dims[0] = n;
    Base17 { dims }
}

fn frozen(values: &[i16]) -> WeightPalette {
    WeightPalette {
        entries: values.iter().copied().map(value).collect(),
        radii: vec![0; values.len()],
        counts: vec![1; values.len()],
    }
}

#[test]
fn closed_codebook_64_hops_are_exact_and_leave_frame_unchanged() {
    let palette = frozen(&[0, 1]);
    let snapshot = palette.entries.clone();
    let table = ComposeTable::build(&palette);
    let mut got = 0u8;
    let mut exact = value(0);
    for hop in 1..=64 {
        let input = if hop % 2 == 0 { 0 } else { 1 };
        got = table.compose(got, input);
        exact = exact.xor_bind(&palette.entries[input as usize]);
        assert_eq!(palette.entries[got as usize], exact, "hop {hop}");
    }
    assert_eq!(palette.entries, snapshot);
}

#[test]
fn immutable_but_nonclosed_codebook_has_a_three_hop_counterexample() {
    let palette = frozen(&[0, 1, 2]); // 1 XOR 2 = 3 is not in the codebook
    let table = ComposeTable::build(&palette);
    let direct = value(1).xor_bind(&value(2)).xor_bind(&value(2));
    assert_eq!(direct, value(1)); // XOR is associative and self-inverse
    let first = table.compose(1, 2); // missing 3 snaps to nearest 2
    assert_eq!(first, 2);
    let quantized = table.compose(first, 2); // then 2 XOR 2 returns 0
    assert_eq!(quantized, 0);
    assert_ne!(palette.entries[quantized as usize], direct);
}
