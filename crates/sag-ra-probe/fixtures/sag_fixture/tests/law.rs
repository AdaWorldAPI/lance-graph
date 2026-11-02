//! Independent behavioural oracle for the candidate law
//! count(filter(P, X)) == popcount(mask(P, X)), P = `x < t`.
use sag_fixture::*;

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

#[test]
fn pure_filter_count_equals_mask_popcount() {
    let mut s = 0x9E3779B97F4A7C15;
    for n in [0usize, 1, 63, 64, 65, 127, 128, 1000] {
        for _ in 0..200 {
            // A small value range so `x == t` occurs: random u32 almost never
            // ties, and a `<` / `<=` slip would then go unseen.
            let xs: Vec<u32> = (0..n).map(|_| (lcg(&mut s) % 16) as u32).collect();
            let t = (lcg(&mut s) % 17) as u32;
            assert_eq!(count_below(&xs, t), popcount_mask(&mask_below(&xs, t)));
        }
    }
}

#[test]
fn the_tally_closure_has_an_effect_the_mask_form_drops() {
    let xs = [1u32, 5, 9];
    let mut seen = 0u64;
    let c = count_with_tally(&xs, 6, &mut seen);
    assert_eq!(c, popcount_mask(&mask_below(&xs, 6)));
    // Same count, but the original also advanced `seen`; the fused form would not.
    assert_eq!(seen, 3);
}

#[test]
#[should_panic]
fn the_division_predicate_panics_where_the_mask_form_cannot() {
    count_div(&[1, 2, 3], 0);
}
