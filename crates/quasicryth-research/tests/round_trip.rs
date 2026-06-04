//! Integration tests: cross-variant round-trip + edge cases.
//!
//! Run with `cargo test --manifest-path crates/quasicryth-research/Cargo.toml`.

use quasicryth_research::pipeline::{compress, decompress, Variant};

fn round_trip(input: &[u8], variant: Variant) {
    let compressed = compress(input, variant).expect("compress");
    let decompressed = decompress(&compressed).expect("decompress");
    assert_eq!(decompressed, input, "round-trip mismatch under {variant:?}");
}

#[test]
fn variants_agree_on_long_natural_text() {
    let text = b"\
The Fibonacci substitution sigma takes L to LS and S to L. It generates an \
aperiodic sequence of two tile types with a precisely irrational frequency \
ratio phi colon one. The hierarchy never collapses because phi is a Pisot \
Vijayaraghavan number whose conjugate has absolute value less than one. \
This is the algebraic reason for the non-collapse theorem. Period five \
collapses at level three because log five over log phi is approximately \
three point three. Every periodic tiling collapses within order log p \
levels for any rational L frequency p over q.";
    round_trip(text, Variant::Flat);
    round_trip(text, Variant::CowRadix);

    // Both variants must produce decompressed output that equals input.
    let from_flat = decompress(&compress(text, Variant::Flat).unwrap()).unwrap();
    let from_cow = decompress(&compress(text, Variant::CowRadix).unwrap()).unwrap();
    assert_eq!(from_flat, from_cow);
    assert_eq!(from_flat, text);
}

#[test]
fn round_trip_at_5kb_scale() {
    let phrase = "the quick brown fox jumps over the lazy dog ";
    let mut input = Vec::new();
    while input.len() < 5_000 {
        input.extend_from_slice(phrase.as_bytes());
    }
    round_trip(&input, Variant::Flat);
    round_trip(&input, Variant::CowRadix);
}

#[test]
fn round_trip_single_word() {
    round_trip(b"hello", Variant::Flat);
    round_trip(b"hello", Variant::CowRadix);
}

#[test]
fn round_trip_only_whitespace() {
    round_trip(b"     ", Variant::Flat);
    round_trip(b"     ", Variant::CowRadix);
}

#[test]
fn round_trip_mixed_punctuation_lines() {
    let text = b"\
First line. (Has parens.) Second-line: with hyphens, commas; semicolons!
Third \"quoted\" line. 'Single quotes' too. Fourth\tline with tabs.
";
    round_trip(text, Variant::Flat);
    round_trip(text, Variant::CowRadix);
}

#[test]
fn round_trip_repeated_uppercase_word() {
    let text = b"HELLO WORLD HELLO WORLD HELLO WORLD";
    round_trip(text, Variant::Flat);
    round_trip(text, Variant::CowRadix);
}

#[test]
fn cross_variant_independence() {
    // Each variant must round-trip on its own AND produce identical decoded
    // output (the compressed bytes themselves may differ — that's fine).
    let text = b"alpha beta gamma alpha beta delta epsilon alpha";
    let compressed_flat = compress(text, Variant::Flat).unwrap();
    let compressed_cow = compress(text, Variant::CowRadix).unwrap();
    assert_eq!(decompress(&compressed_flat).unwrap(), text);
    assert_eq!(decompress(&compressed_cow).unwrap(), text);
}
