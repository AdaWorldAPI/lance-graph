//! `PmLedger`: every key's value equals an `i64` model (or is never
//! touched), through single events, block absorbs, carries past ±255 and
//! demotion back after cancellation.

use xlog_relcomp_probe::pm_ledger::{PmLedger, PmOverflow};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, k: u64) -> u64 {
        self.next() % k
    }
}

fn assert_model(l: &PmLedger, m: &[Option<i64>], what: &str) {
    for (k, want) in m.iter().enumerate() {
        assert_eq!(l.value(k), *want, "{what}: key {k}");
    }
}

#[test]
fn a_touched_net_zero_is_plus_zero_and_untouched_stays_untouched() {
    let mut l = PmLedger::new(3);
    l.add(0, -5).unwrap();
    l.add(0, 5).unwrap();
    assert_eq!(l.value(0), Some(0), "-5 + +5 = +0");
    assert_eq!(l.value(1), None, "never touched");
    l.add(2, 0).unwrap();
    assert_eq!(l.value(2), Some(0), "+0 is touched");
    assert_eq!(l.touched_keys(), 2);
}

#[test]
fn a_carry_that_cancels_is_counted_and_demoted() {
    let mut l = PmLedger::new(1);
    l.add(0, 255).unwrap();
    l.add(0, 255).unwrap();
    assert_eq!((l.carries, l.escalated(), l.value(0)), (1, 1, Some(510)));
    l.add(0, -255).unwrap();
    assert_eq!(l.value(0), Some(255));
    assert_eq!(l.escalated(), 0, "back within ±255: a lane again");
    assert_eq!((l.carries, l.demotions), (1, 1));
}

#[test]
fn absorb_demotes_a_wide_value_cancelled_by_a_lane() {
    let mut base = PmLedger::new(70);
    base.add(65, 200).unwrap();
    base.add(65, 100).unwrap();
    assert_eq!(base.escalated(), 1);
    let mut d = PmLedger::new(70);
    d.add(65, -100).unwrap();
    base.absorb(&d).unwrap();
    assert_eq!(base.value(65), Some(200));
    assert_eq!(base.escalated(), 0);
    assert_eq!(base.demotions, 1);
}

#[test]
fn absorb_of_a_wide_delta_that_cancels_does_not_carry() {
    let mut base = PmLedger::new(4);
    base.add(1, -200).unwrap();
    let mut d = PmLedger::new(4);
    d.add(1, 300).unwrap();
    assert_eq!(d.escalated(), 1, "the delta itself is wide");
    let before = base.carries;
    base.absorb(&d).unwrap();
    assert_eq!(base.value(1), Some(100));
    assert_eq!(base.escalated(), 0);
    assert_eq!(base.carries, before, "|-200 + 300| fits: no carry");
}

#[test]
fn absorb_carries_where_the_lane_sum_leaves_the_range() {
    let mut base = PmLedger::new(64);
    let mut d = PmLedger::new(64);
    for k in 0..64 {
        base.add(k, 200).unwrap();
        d.add(k, if k % 2 == 0 { 100 } else { -100 }).unwrap();
    }
    base.absorb(&d).unwrap();
    assert_eq!(base.carries, 32, "every even key passes 255");
    assert_eq!(base.escalated(), 32);
    for k in 0..64 {
        let want = if k % 2 == 0 { 300 } else { 100 };
        assert_eq!(base.value(k), Some(want));
    }
}

/// Random events and absorbs against an `i64` model, on a key count that
/// is not a multiple of 64. Anti-vacuity: the run must carry, demote, and
/// end with wide keys and never-touched keys.
#[test]
fn random_streams_match_the_i64_model() {
    let (mut carries, mut demotions, mut wide_end, mut untouched_end) = (0, 0, 0, 0);
    for seed in 1..=40u64 {
        let mut r = Lcg(seed);
        let n = 130;
        let mut base = PmLedger::new(n);
        let mut model: Vec<Option<i64>> = vec![None; n];
        for _ in 0..30 {
            let mut d = PmLedger::new(n);
            let mut dm: Vec<Option<i64>> = vec![None; n];
            for _ in 0..200 {
                let k = r.below(100) as usize; // keys 100.. stay untouched
                let w = r.below(161) as i64 - 80;
                let w = if r.below(20) == 0 { w * 6 } else { w };
                d.add(k, w).unwrap();
                dm[k] = Some(dm[k].unwrap_or(0) + w);
            }
            assert_model(&d, &dm, "delta");
            base.absorb(&d).unwrap();
            for k in 0..n {
                if let Some(x) = dm[k] {
                    model[k] = Some(model[k].unwrap_or(0) + x);
                }
            }
            assert_model(&base, &model, "base");
        }
        carries += base.carries;
        demotions += base.demotions;
        wide_end += base.escalated();
        untouched_end += model.iter().filter(|v| v.is_none()).count();
    }
    assert!(carries > 0 && demotions > 0, "{carries} {demotions}");
    assert!(
        wide_end > 0 && untouched_end > 0,
        "{wide_end} {untouched_end}"
    );
}

/// The exact escape is `i64` and checked. `i64::MIN` is a legal value (its
/// magnitude does not fit `i64`, so nothing may take `abs` of it).
#[test]
fn i64_min_is_held_exactly() {
    let mut l = PmLedger::new(2);
    l.add(0, i64::MIN).unwrap();
    assert_eq!(l.value(0), Some(i64::MIN));
    assert_eq!(l.escalated(), 1);
    l.add(0, 1).unwrap();
    assert_eq!(l.value(0), Some(i64::MIN + 1));
    l.add(1, -5).unwrap();
    l.add(1, i64::MIN + 5).unwrap();
    assert_eq!(l.value(1), Some(i64::MIN));
}

/// An addition whose exact sum leaves `i64` is refused, naming the key and
/// both operands, and the key keeps its value.
#[test]
fn an_add_past_i64_is_refused_and_changes_nothing() {
    let mut l = PmLedger::new(1);
    l.add(0, i64::MAX).unwrap();
    let before = (l.value(0), l.carries, l.demotions);
    assert_eq!(
        l.add(0, 1),
        Err(PmOverflow {
            key: 0,
            have: i64::MAX,
            add: 1
        })
    );
    assert_eq!((l.value(0), l.carries, l.demotions), before);
    l.add(0, -i64::MAX).unwrap();
    l.add(0, -1).unwrap();
    l.add(0, i64::MIN).unwrap_err();
    assert_eq!(l.value(0), Some(-1), "refused add left the value as it was");
}

/// A refused absorb changes nothing, not even the keys of the same block
/// that would not have overflowed.
#[test]
fn an_absorb_past_i64_is_refused_and_changes_nothing() {
    let mut base = PmLedger::new(70);
    base.add(65, i64::MAX).unwrap();
    base.add(3, 7).unwrap();
    base.add(66, 200).unwrap();
    let mut d = PmLedger::new(70);
    d.add(3, 1).unwrap();
    d.add(66, 100).unwrap(); // would carry
    d.add(65, 1).unwrap(); // overflows
    let before: Vec<Option<i64>> = (0..70).map(|k| base.value(k)).collect();
    let counters = (base.carries, base.demotions, base.escalated());
    assert_eq!(
        base.absorb(&d),
        Err(PmOverflow {
            key: 65,
            have: i64::MAX,
            add: 1
        })
    );
    let after: Vec<Option<i64>> = (0..70).map(|k| base.value(k)).collect();
    assert_eq!(after, before);
    assert_eq!((base.carries, base.demotions, base.escalated()), counters);
    // The same delta without the overflowing key applies.
    let mut d2 = PmLedger::new(70);
    d2.add(3, 1).unwrap();
    d2.add(66, 100).unwrap();
    base.absorb(&d2).unwrap();
    assert_eq!((base.value(3), base.value(66)), (Some(8), Some(300)));
}

/// Two wide values whose sum leaves `i64`: refused the same way.
#[test]
fn an_absorb_of_two_wide_values_past_i64_is_refused() {
    let mut base = PmLedger::new(1);
    base.add(0, i64::MIN).unwrap();
    let mut d = PmLedger::new(1);
    d.add(0, -1000).unwrap();
    assert_eq!(
        base.absorb(&d),
        Err(PmOverflow {
            key: 0,
            have: i64::MIN,
            add: -1000
        })
    );
    assert_eq!(base.value(0), Some(i64::MIN));
}
