//! Immutable reference vs relational value: lab contract for Palette256.
//! A palette byte is an address, never a relationship or a confidence score.
//! Pairwise distribution values live in the frozen, family-calibrated LUT.
//! No hot-path API or memory layout is changed by this regression probe.
use bgz_tensor::fisher_z::FisherZTable;

#[test]
fn baked_pairwise_table_is_stable_across_repeated_reads_and_roundtrip() {
    let representatives = vec![vec![1.0_f32, 0.0], vec![0.8, 0.6], vec![-0.6, 0.8]];
    let table = FisherZTable::build(&representatives, 3);
    let baked = table.to_bytes();
    assert_eq!(baked.len(), 3 * 3 + 8);
    for _thought in 0..64 {
        for a in 0..3_u8 {
            for b in 0..3_u8 {
                assert_eq!(
                    table.lookup_i8(a, b),
                    table.entries[a as usize * 3 + b as usize]
                );
            }
        }
        assert_eq!(
            table.to_bytes(),
            baked,
            "lookup cannot rewrite calibrated reference"
        );
    }
    let restored = FisherZTable::from_bytes(&baked, 3);
    assert_eq!(restored.to_bytes(), baked);
    assert_eq!(restored.lookup_i8(0, 1), table.lookup_i8(0, 1));
}

#[test]
fn needle_identity_is_not_the_same_question_as_relation_strength() {
    let representatives = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
    let table = FisherZTable::build(&representatives, 3);
    let needle: u8 = 1;
    assert_eq!(
        needle, 1,
        "the address names an entry without measuring distance"
    );
    assert_ne!(table.lookup_i8(0, 1), table.lookup_i8(0, 2));
    // Do not infer self-cosine == 1 from the calibrated diagonal. The diagonal
    // consumes family gamma whose calibration excludes self-pairs by design.
    assert_eq!(table.lookup_i8(1, 1), table.entries[4]);
}
