//! V4: the Quack filters [`effectively_active`] / [`effectively_inactive`]
//! agree with the one definition, `ogar_dir_sim::effective_active`, on every
//! row of the 3×3 table, executed through mask-risc.
//!
//! Each source is a known (validity) plane plus an enabled plane. Where a
//! source is unknown its enabled bit is a stale payload, so the table is run
//! twice: once with every stale bit set (it would vote enabled) and once with
//! every stale bit clear (it would vote disabled). Neither may change a row.

use lance_graph_dir_sim::{effectively_active, effectively_inactive};
use lance_graph_mask_risc::{
    execute_into, materialize_rows, words_for, Foreign, Out, Planes, Scratch,
};
use lance_graph_quack::{lower, Agg, Filter, Mask, Query};
use ogar_dir_sim::effective_active;

const STATES: [Option<bool>; 3] = [Some(true), Some(false), None];
const AD_KNOWN: Mask = Mask(0);
const AD_ENABLED: Mask = Mask(1);
const ENTRA_KNOWN: Mask = Mask(2);
const ENTRA_ENABLED: Mask = Mask(3);

/// The nine `(ad, entra)` rows, in a fixed order.
fn table() -> Vec<(Option<bool>, Option<bool>)> {
    STATES
        .iter()
        .flat_map(|&ad| STATES.iter().map(move |&entra| (ad, entra)))
        .collect()
}

/// `[ad_known, ad_enabled, entra_known, entra_enabled]` over the table,
/// with every unknown source's enabled bit set to `stale`.
fn planes(rows: &[(Option<bool>, Option<bool>)], stale: bool) -> [Vec<u64>; 4] {
    let mut p: [Vec<u64>; 4] = std::array::from_fn(|_| vec![0; words_for(rows.len())]);
    let mut put = |plane: usize, i: usize| p[plane][i / 64] |= 1 << (i % 64);
    for (i, &(ad, entra)) in rows.iter().enumerate() {
        for (src, flag) in [(0, ad), (2, entra)] {
            if flag.is_some() {
                put(src, i);
            }
            if flag.unwrap_or(stale) {
                put(src + 1, i);
            }
        }
    }
    p
}

fn kept(filter: Filter, p: &[Vec<u64>; 4], n: usize) -> Vec<usize> {
    let prog = lower(&Query {
        filter,
        agg: Agg::Rows,
    })
    .unwrap();
    let masks: Vec<&[u64]> = p.iter().map(Vec::as_slice).collect();
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let mut scratch = Scratch::for_program(&prog, n).unwrap();
    let mut out = vec![0u64; words_for(n)];
    execute_into(
        &prog,
        &planes,
        &Foreign::NONE,
        &mut scratch,
        Out::Mask(&mut out),
    )
    .unwrap();
    materialize_rows(&out, n)
}

#[test]
fn the_quack_filters_agree_with_effective_active_on_every_row() {
    let rows = table();
    let n = rows.len();
    for stale in [true, false] {
        let p = planes(&rows, stale);
        let active = kept(
            effectively_active(AD_KNOWN, AD_ENABLED, ENTRA_KNOWN, ENTRA_ENABLED),
            &p,
            n,
        );
        let inactive = kept(
            effectively_inactive(AD_KNOWN, AD_ENABLED, ENTRA_KNOWN, ENTRA_ENABLED),
            &p,
            n,
        );
        for (i, &(ad, entra)) in rows.iter().enumerate() {
            let got = match (active.contains(&i), inactive.contains(&i)) {
                (true, false) => Some(true),
                (false, true) => Some(false),
                (false, false) => None,
                (true, true) => panic!("row {i} kept by both filters"),
            };
            assert_eq!(
                got,
                effective_active(ad, entra),
                "ad={ad:?} entra={entra:?} stale={stale}"
            );
        }
        // Anti-vacuity: all three values occur.
        assert_eq!(
            active.len(),
            3,
            "enabled+enabled, enabled+unknown, unknown+enabled"
        );
        assert_eq!(inactive.len(), 5);
    }
}
