//! Every code constant appears verbatim in PREREG.md (pre-registration integrity).

use perturbationsfeld_probe::prereg::*;

const PREREG: &str = include_str!("../PREREG.md");

#[test]
fn prereg_matches_code() {
    let lines = [
        format!("THETA = {}", THETA),
        format!("DELTA_IDS = {}", DELTA_IDS),
        format!("TOP = {}", TOP),
        format!("Q = {}", Q),
        format!("M = {}", M),
        format!("MAX_CYCLES = {}", MAX_CYCLES),
        format!("STIMULUS_SEED = {:#018X}", STIMULUS_SEED),
        format!("PERMUTATION_SEED = {:#018X}", PERMUTATION_SEED),
        format!("DEGENERATE_CEILING_PCT = {}", DEGENERATE_CEILING_PCT),
        format!("SANITY_MIN_ELIGIBLE = {}", SANITY_MIN_ELIGIBLE),
        format!("EMPTY_WINDOW = {}", EMPTY_WINDOW),
    ];
    for l in &lines {
        assert!(PREREG.contains(l.as_str()), "PREREG.md is missing `{l}`");
    }
}
