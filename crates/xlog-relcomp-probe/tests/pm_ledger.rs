//! `PmLedger`: every key's value equals an `i64` model (or is never
//! touched), through single events, block absorbs, carries past ±255 and
//! demotion back after cancellation.

use xlog_relcomp_probe::pm_ledger::PmLedger;

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
    l.add(0, -5);
    l.add(0, 5);
    assert_eq!(l.value(0), Some(0), "-5 + +5 = +0");
    assert_eq!(l.value(1), None, "never touched");
    l.add(2, 0);
    assert_eq!(l.value(2), Some(0), "+0 is touched");
    assert_eq!(l.touched_keys(), 2);
}

#[test]
fn a_carry_that_cancels_is_counted_and_demoted() {
    let mut l = PmLedger::new(1);
    l.add(0, 255);
    l.add(0, 255);
    assert_eq!((l.carries, l.escalated(), l.value(0)), (1, 1, Some(510)));
    l.add(0, -255);
    assert_eq!(l.value(0), Some(255));
    assert_eq!(l.escalated(), 0, "back within ±255: a lane again");
    assert_eq!((l.carries, l.demotions), (1, 1));
}

#[test]
fn absorb_demotes_a_wide_value_cancelled_by_a_lane() {
    let mut base = PmLedger::new(70);
    base.add(65, 200);
    base.add(65, 100);
    assert_eq!(base.escalated(), 1);
    let mut d = PmLedger::new(70);
    d.add(65, -100);
    base.absorb(&d);
    assert_eq!(base.value(65), Some(200));
    assert_eq!(base.escalated(), 0);
    assert_eq!(base.demotions, 1);
}

#[test]
fn absorb_of_a_wide_delta_that_cancels_does_not_carry() {
    let mut base = PmLedger::new(4);
    base.add(1, -200);
    let mut d = PmLedger::new(4);
    d.add(1, 300);
    assert_eq!(d.escalated(), 1, "the delta itself is wide");
    let before = base.carries;
    base.absorb(&d);
    assert_eq!(base.value(1), Some(100));
    assert_eq!(base.escalated(), 0);
    assert_eq!(base.carries, before, "|-200 + 300| fits: no carry");
}

#[test]
fn absorb_carries_where_the_lane_sum_leaves_the_range() {
    let mut base = PmLedger::new(64);
    let mut d = PmLedger::new(64);
    for k in 0..64 {
        base.add(k, 200);
        d.add(k, if k % 2 == 0 { 100 } else { -100 });
    }
    base.absorb(&d);
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
                d.add(k, w);
                dm[k] = Some(dm[k].unwrap_or(0) + w);
            }
            assert_model(&d, &dm, "delta");
            base.absorb(&d);
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
