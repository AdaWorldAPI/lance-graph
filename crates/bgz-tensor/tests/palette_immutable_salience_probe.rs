//! Read-only Palette256 frame and PCDVQ salience contract.
//! This does not assert that dropping weak signals eliminates hallucinations.
use bgz_tensor::{Base17, WeightPalette};

fn axis(axis: usize, amplitude: i16) -> Base17 {
    let mut dims = [0i16; 17];
    dims[axis] = amplitude;
    Base17 { dims }
}

#[test]
fn directional_signal_has_higher_salience_than_same_amplitude_detail() {
    let origin = axis(0, 0);
    let directional = axis(0, 8);
    let detail = axis(16, 8);
    assert_eq!(origin.l1(&directional), origin.l1(&detail));
    assert_eq!(origin.l1_weighted(&directional), 20 * 8);
    assert_eq!(origin.l1_weighted(&detail), 8);
}

#[test]
fn inference_reads_do_not_retrain_the_baked_reference() {
    let rows = [axis(0, 0), axis(0, 12), axis(16, 2), axis(0, -12)];
    let palette = WeightPalette::build(&rows, 3);
    let original_entries = palette.entries.clone();
    let original_radii = palette.radii.clone();
    let original_counts = palette.counts.clone();
    for _ in 0..64 {
        for row in &rows {
            let assignment = palette.assign(row);
            assert!((assignment.index as usize) < palette.len());
        }
    }
    assert_eq!(palette.entries, original_entries);
    assert_eq!(palette.radii, original_radii);
    assert_eq!(palette.counts, original_counts);
}
