//! Arm Q, masked count over several tiles: selecting tiles (pages of 256
//! users) with a tile mask and rows inside each tile with a row mask must
//! count exactly the users a page-by-page resolve finds, for roles and for
//! effective group membership.

use dir_rmap_probe::World;
use dir_rmap_probe::fixture::{Fixture, Spec};
use dir_rmap_probe::focus::Row;
use dir_rmap_probe::qr::{Pages, Qr, Runs};

const ALL: Row = [!0; 4];
const EVEN: Row = [0x5555_5555_5555_5555; 4];
const ODD: Row = [0xAAAA_AAAA_AAAA_AAAA; 4];

fn bit(r: &Row, i: usize) -> bool {
    r[i >> 6] >> (i & 63) & 1 == 1
}

/// Tile masks: one tile, every other tile, every third tile, all tiles.
fn tile_masks(pages: usize) -> Vec<Vec<u64>> {
    let words = pages.div_ceil(64);
    let by = |f: &dyn Fn(usize) -> bool| {
        let mut m = vec![0u64; words];
        for p in (0..pages).filter(|&p| f(p)) {
            m[p >> 6] |= 1 << (p & 63);
        }
        m
    };
    vec![
        by(&|p| p == pages / 2),
        by(&|p| p % 2 == 0),
        by(&|p| p % 3 == 1),
        by(&|_| true),
    ]
}

/// Per-page row masks: all rows, even rows, odd rows, or alternating by
/// page (interlaced activation).
fn row_masks(pages: usize) -> Vec<Vec<Row>> {
    vec![
        vec![ALL; pages],
        vec![EVEN; pages],
        vec![ODD; pages],
        (0..pages)
            .map(|p| if p % 2 == 0 { EVEN } else { ODD })
            .collect(),
    ]
}

/// Does a user hold the answer, given its roles and effective row?
type Holds = Box<dyn Fn(u64, &Row) -> bool>;

#[test]
fn masked_counts_equal_a_page_by_page_resolve() {
    let mut spec = Spec::small();
    spec.users = 3000;
    let w = World::new(Fixture::generate(spec));
    let b = w.prepare();
    let qr = Qr::bind(&b, &w.witness()).unwrap();
    let mut pages = Pages::new(&qr);
    let np = qr.pages();
    let mut scratch = vec![[0u64; 4]; np];
    let mut runs = Runs::new(&qr);

    // Questions: every role, and membership of a few security groups.
    let mut questions: Vec<(Row, Holds)> = Vec::new();
    for r in 0..64u32 {
        if qr.role_slots[r as usize] != [0; 4] {
            questions.push((
                qr.role_slots[r as usize],
                Box::new(move |roles: u64, _: &Row| roles >> r & 1 == 1),
            ));
        }
    }
    for &g in qr.ids.iter().step_by(17) {
        let t = qr.slot(g).unwrap();
        questions.push((
            qr.slots_reaching(g).unwrap(),
            Box::new(move |_: u64, eff: &Row| bit(eff, t)),
        ));
    }
    assert!(questions.len() > 10);

    let (mut nonzero, mut checked) = (0, 0);
    for (slots, holds) in &questions {
        for tiles in tile_masks(np) {
            for rows in row_masks(np) {
                let got = qr
                    .count_masked(&b, slots, &tiles, &rows, &mut scratch)
                    .unwrap();
                let mut want = 0u64;
                for p in 0..np {
                    if tiles[p >> 6] >> (p & 63) & 1 == 0 {
                        continue;
                    }
                    for i in 0..256 {
                        let u = p * 256 + i;
                        if u >= b.users || !bit(&rows[p], i) {
                            continue;
                        }
                        let page = pages.of(&qr, &b, u as u16).unwrap();
                        let eff = qr.effective(&page.member[i]);
                        want += u64::from(holds(page.roles[i], &eff));
                    }
                }
                assert_eq!(got, want);
                assert_eq!(
                    runs.count(&qr, &b, slots, &tiles, &rows).unwrap(),
                    want,
                    "runs"
                );
                assert!(scratch.iter().all(|t| *t == [0; 4]), "scratch left dirty");
                nonzero += usize::from(got != 0);
                checked += 1;
            }
        }
    }
    // Anti-vacuity: most combinations select somebody.
    assert!(nonzero * 2 > checked, "{nonzero} of {checked}");
}

#[test]
fn a_stale_snapshot_refuses_the_masked_count() {
    let mut w = World::new(Fixture::generate(Spec::small()));
    let b = w.prepare();
    let qr = Qr::bind(&b, &w.witness()).unwrap();
    let (u, g) = w.direct[0];
    w.apply(&[dir_rmap_probe::Edit::RemoveDirect(u, g)]);
    let b2 = w.prepare();
    let np = qr.pages();
    let mut scratch = vec![[0u64; 4]; np];
    let r = qr.count_masked(
        &b2,
        &ALL,
        &vec![!0; np.div_ceil(64)],
        &vec![ALL; np],
        &mut scratch,
    );
    assert_eq!(r, Err(dir_rmap_probe::prep::Refusal::Stale));
}
