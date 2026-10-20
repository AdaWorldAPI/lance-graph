//! D-RMAP focus probe against an independent oracle.
//!
//! The oracle scans the canonical fact slice directly: it uses no index, no
//! focus binding and no prepared operand, so agreement is not one structure
//! agreeing with itself.

use std::collections::BTreeSet;

use lance_graph_mask_risc::Scratch;
use mork_pathmap_probe::boundary::PostingTrie;
use mork_pathmap_probe::rmap::*;

const PREDS: [u32; 3] = [0, 1, 2];

/// Scoped reachability by scanning all facts: the induced subgraph of the
/// focus, BFS by levels, escapes collected from reached nodes within budget.
fn oracle(
    facts: &[Fact],
    focus: &[u32],
    preds: &[u32],
    seeds: &[u32],
    hops: usize,
) -> (BTreeSet<u32>, BTreeSet<u32>) {
    let inf: BTreeSet<u32> = focus.iter().copied().collect();
    let mut reached: BTreeSet<u32> = seeds.iter().copied().collect();
    let mut frontier = reached.clone();
    let mut escaped = BTreeSet::new();
    for _ in 0..hops {
        let mut next = BTreeSet::new();
        for &[s, p, o] in facts {
            if frontier.contains(&s) && preds.contains(&p) {
                if inf.contains(&o) {
                    next.insert(o);
                } else {
                    escaped.insert(o);
                }
            }
        }
        frontier = next.difference(&reached).copied().collect();
        reached.extend(next);
        if frontier.is_empty() {
            break;
        }
    }
    (reached, escaped)
}

fn ids_of(focus: &Focus, set: &SlotSet) -> BTreeSet<u32> {
    slots_of(set).into_iter().map(|s| focus.id(s)).collect()
}

struct World {
    facts: Vec<Fact>,
}

fn world() -> World {
    World {
        facts: graph(65_536, 3, 2, 512, 0.9, 0x5EED),
    }
}

fn w(seal: u64, focus: u64) -> Witness {
    Witness {
        facts: 1,
        seal,
        focus,
        scope: 7,
    }
}

/// One hop, two hops, three hops, single and multiple predicates: every
/// prepared arm equals the oracle, and so does the direct walk over both
/// indexes (sorted run and PathMap trie).
#[test]
fn every_arm_matches_the_oracle_on_one_two_and_three_hops() {
    let wd = world();
    let run = SortedRun::build(&wd.facts, false, 1);
    let trie = SpogTrie::build(&wd.facts);
    let focus = discover(&run, &[17], &PREDS, SLOTS);
    assert_eq!(focus.len(), SLOTS, "the fixture must yield a full focus");
    let focus_trie = discover(&trie, &[17], &PREDS, SLOTS);
    assert_eq!(
        focus.ids(),
        focus_trie.ids(),
        "both indexes discover the same focus in the same order"
    );
    let cur = w(1, focus.generation);
    let csr = Csr::prepare(&run, &focus, &PREDS, cur);
    let tile = Tile::from_csr(&csr);
    let mut ex = TileExec::new();
    let mut checked = 0;
    for preds in [&PREDS[..1], &PREDS[..]] {
        for hops in 1..=3 {
            for seed_slot in [0u8, 40, 255] {
                let seeds = slot_set([seed_slot]);
                let (want_r, want_e) =
                    oracle(&wd.facts, focus.ids(), preds, &[focus.id(seed_slot)], hops);
                let direct = reach_direct(&run, &focus, preds, &seeds, hops);
                let via_trie = reach_direct(&trie, &focus, preds, &seeds, hops);
                let c = csr.reach(&cur, preds, &seeds, hops).unwrap();
                let t = ex
                    .reach(&tile, &csr, &cur, None, None, preds, &seeds, hops)
                    .unwrap();
                for (name, got) in [
                    ("direct", &direct),
                    ("trie", &via_trie),
                    ("csr", &c),
                    ("tile", &t),
                ] {
                    assert_eq!(
                        ids_of(&focus, &got.reached),
                        want_r,
                        "{name} reached, preds {preds:?}, hops {hops}"
                    );
                    assert_eq!(
                        got.escaped, want_e,
                        "{name} escaped, preds {preds:?}, hops {hops}"
                    );
                }
                // The mask mode gives the same reach, and its escape sources
                // turn into exactly the escaped identities.
                let cm = csr.reach_sources(&cur, preds, &seeds, hops).unwrap();
                let tm = ex
                    .reach_sources(&tile, &csr, &cur, None, None, preds, &seeds, hops)
                    .unwrap();
                assert_eq!(cm, tm);
                assert_eq!(reach_direct_sources(&run, &focus, preds, &seeds, hops), cm);
                assert_eq!(ids_of(&focus, &cm.reached), want_r);
                assert_eq!(csr.escape_ids(&cm.escape_sources, preds), want_e);
                assert_eq!(cm.global().is_ok(), want_e.is_empty());
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 18);
}

/// Anti-vacuity: the fixture exercises what the assertions above claim.
/// Three hops reach more than one hop, edges leave the focus, and a reached
/// set is neither empty nor the whole focus.
#[test]
fn the_fixture_is_not_degenerate() {
    let wd = world();
    let run = SortedRun::build(&wd.facts, false, 1);
    let focus = discover(&run, &[17], &PREDS, SLOTS);
    let csr = Csr::prepare(&run, &focus, &PREDS, w(1, 1));
    assert!(csr.edges() > SLOTS, "in-focus edges: {}", csr.edges());
    assert!(csr.escapes() > 0, "some edges must leave the focus");
    let seeds = slot_set([40]);
    let one = csr.reach(&w(1, 1), &PREDS, &seeds, 1).unwrap();
    let three = csr.reach(&w(1, 1), &PREDS, &seeds, 3).unwrap();
    let n1 = slots_of(&one.reached).len();
    let n3 = slots_of(&three.reached).len();
    assert!(n1 > 1 && n3 > n1 && n3 < SLOTS, "1 hop {n1}, 3 hops {n3}");
    assert!(!three.escaped.is_empty());
}

/// An out-of-focus target is never a relationship: it is reported, and the
/// global question is refused while it exists.
#[test]
fn an_out_of_focus_target_is_reported_never_reached() {
    // 0 -> 1 -> 9 -> 2; the focus is {0, 1, 2}. The global answer from 0 in
    // three hops contains 2, but only through 9, which is not focused.
    let facts: Vec<Fact> = vec![[0, 0, 1], [1, 0, 9], [9, 0, 2]];
    let run = SortedRun::build(&facts, false, 1);
    let focus = Focus::bind(&[0, 1, 2], 1);
    let csr = Csr::prepare(&run, &focus, &[0], w(1, 1));
    let r = csr.reach(&w(1, 1), &[0], &slot_set([0]), 3).unwrap();
    assert_eq!(
        ids_of(&focus, &r.reached),
        BTreeSet::from([0, 1]),
        "2 is not reachable INSIDE the focus"
    );
    assert_eq!(r.escaped, BTreeSet::from([9]));
    assert_eq!(r.global(), Err(Refusal::IncompleteFocus));
    // Silence twin: with 9 focused, nothing escapes and the global answer is given.
    let focus = Focus::bind(&[0, 1, 2, 9], 1);
    let csr = Csr::prepare(&run, &focus, &[0], w(1, 1));
    let r = csr.reach(&w(1, 1), &[0], &slot_set([0]), 3).unwrap();
    assert!(r.escaped.is_empty());
    assert_eq!(
        ids_of(&focus, r.global().unwrap()),
        BTreeSet::from([0, 1, 2, 9])
    );
}

/// Every in-focus edge keeps the canonical fact it came from.
#[test]
fn prepared_edges_keep_their_canonical_rows() {
    let wd = world();
    let run = SortedRun::build(&wd.facts, false, 1);
    let focus = discover(&run, &[17], &PREDS, SLOTS);
    let csr = Csr::prepare(&run, &focus, &PREDS, w(1, 1));
    let mut n = 0;
    for (p, s, t, row) in csr.edge_set() {
        assert_eq!(wd.facts[row as usize], [focus.id(s), p, focus.id(t)]);
        n += 1;
    }
    assert_eq!(n, csr.edges());
}

/// Each validity dimension refuses on its own, and a matching witness passes.
#[test]
fn a_stale_operand_is_refused_per_dimension() {
    let wd = world();
    let run = SortedRun::build(&wd.facts, false, 1);
    let focus = discover(&run, &[17], &PREDS, SLOTS);
    let bound = w(1, focus.generation);
    let csr = Csr::prepare(&run, &focus, &PREDS, bound);
    let tile = Tile::from_csr(&csr);
    let seeds = slot_set([0]);
    let mut ex = TileExec::new();
    assert!(csr.reach(&bound, &PREDS, &seeds, 2).is_ok());
    let cases = [
        (Witness { facts: 2, ..bound }, Refusal::FactsChanged),
        (Witness { seal: 2, ..bound }, Refusal::IndexResealed),
        (
            Witness {
                focus: bound.focus + 1,
                ..bound
            },
            Refusal::FocusChanged,
        ),
        (Witness { scope: 8, ..bound }, Refusal::ScopeChanged),
    ];
    for (cur, why) in cases {
        assert_eq!(csr.reach(&cur, &PREDS, &seeds, 2).err(), Some(why));
        assert_eq!(
            ex.reach(&tile, &csr, &cur, None, None, &PREDS, &seeds, 2)
                .err(),
            Some(why)
        );
    }
}

/// A recycled thought slot: a delta built for one thought incarnation is
/// refused for another, even when the slot number is the same.
#[test]
fn a_thought_delta_is_refused_for_another_incarnation() {
    let wd = world();
    let run = SortedRun::build(&wd.facts, false, 1);
    let focus = discover(&run, &[17], &PREDS, SLOTS);
    let cur = w(1, focus.generation);
    let csr = Csr::prepare(&run, &focus, &PREDS, cur);
    let tile = Tile::from_csr(&csr);
    let delta = ThoughtDelta::new(41, 0, &[(0, 200)], &[]);
    let mut ex = TileExec::new();
    let seeds = slot_set([0]);
    assert!(
        ex.reach(&tile, &csr, &cur, Some(&delta), Some(41), &PREDS, &seeds, 1)
            .is_ok()
    );
    assert_eq!(
        ex.reach(&tile, &csr, &cur, Some(&delta), Some(42), &PREDS, &seeds, 1)
            .err(),
        Some(Refusal::ThoughtMismatch)
    );
    assert_eq!(
        ex.reach_sources(&tile, &csr, &cur, Some(&delta), Some(42), &PREDS, &seeds, 1)
            .err(),
        Some(Refusal::ThoughtMismatch)
    );
}

/// Thought deltas over one shared tile: added and retracted edges change only
/// that thought's answer, equal an oracle over the edited fact set, and leave
/// the shared tile and other thoughts untouched.
#[test]
fn thought_deltas_change_only_their_own_thought() {
    let wd = world();
    let run = SortedRun::build(&wd.facts, false, 1);
    let focus = discover(&run, &[17], &PREDS, SLOTS);
    let cur = w(1, focus.generation);
    let csr = Csr::prepare(&run, &focus, &[0], cur);
    let tile = Tile::from_csr(&csr);
    let mut ex = TileExec::new();
    let seeds = slot_set([0]);
    let base = ex
        .reach(&tile, &csr, &cur, None, None, &[0], &seeds, 2)
        .unwrap();
    // Retract every edge out of slot 0, add 0 -> 255.
    let out_of_0: Vec<(u8, u8)> = csr
        .edge_set()
        .into_iter()
        .filter(|e| e.1 == 0)
        .map(|e| (0u8, e.2))
        .collect();
    assert!(!out_of_0.is_empty());
    let delta = ThoughtDelta::new(5, 0, &[(0, 255)], &out_of_0);
    let t = ex
        .reach(&tile, &csr, &cur, Some(&delta), Some(5), &[0], &seeds, 2)
        .unwrap();
    // Oracle over the edited facts (in-focus part only; escapes are the CSR's).
    let mut edited: Vec<Fact> = wd
        .facts
        .iter()
        .copied()
        .filter(|f| !(f[0] == focus.id(0) && f[1] == 0 && focus.slot_of(f[2]).is_some()))
        .collect();
    edited.push([focus.id(0), 0, focus.id(255)]);
    let (want, _) = oracle(&edited, focus.ids(), &[0], &[focus.id(0)], 2);
    assert_eq!(ids_of(&focus, &t.reached), want);
    assert_ne!(
        t.reached, base.reached,
        "the delta must change this thought's answer"
    );
    let again = ex
        .reach(&tile, &csr, &cur, None, None, &[0], &seeds, 2)
        .unwrap();
    assert_eq!(again.reached, base.reached, "the shared tile is unchanged");
}

/// Rebinding k slots incrementally gives exactly what a full preparation of
/// the new focus gives, for k = 1, 8, 16, 128, 256.
#[test]
fn incremental_rebind_equals_full_preparation() {
    let wd = world();
    let fwd = SortedRun::build(&wd.facts, false, 1);
    let focus = discover(&fwd, &[17], &PREDS, SLOTS);
    let csr = Csr::prepare(&fwd, &focus, &PREDS, w(1, focus.generation));
    // Replacement identities: a second community's focus (some may already be
    // in the first, which exercises slot swaps).
    let other = discover(&fwd, &[40_000], &PREDS, SLOTS);
    // Two sources of replacements: another community (few edges into the
    // focus), and the focus's own escape targets (focus expansion: every
    // replacement already has an edge INTO it from the focus, which is what
    // the escape re-resolution must recover).
    let mut escaped: Vec<u32> = csr.escape_set().into_iter().map(|e| e.2).collect();
    escaped.sort_unstable();
    escaped.dedup();
    assert!(
        escaped.len() >= 256,
        "expansion needs 256 escape targets: {}",
        escaped.len()
    );
    let mut recovered_incoming = 0;
    for (mode, source) in [
        ("other community", other.ids().to_vec()),
        ("expansion", escaped),
    ] {
        for k in [1usize, 8, 16, 128, 256] {
            let mut taken: BTreeSet<u32> = focus.ids().iter().copied().collect();
            let mut changes = Vec::new();
            for (s, &g) in (0..k).zip(source.iter()) {
                if taken.insert(g) {
                    changes.push(((SLOTS - 1 - s) as u8, g));
                }
            }
            let new = focus.replace(&changes);
            let nw = w(1, new.generation);
            let (inc, inc_tile) = rebind(&fwd, &focus, &new, &csr, nw);
            let full = Csr::prepare(&fwd, &new, &PREDS, nw);
            assert_eq!(
                inc.edge_set(),
                full.edge_set(),
                "{mode}: edges after rebinding {k}"
            );
            // Edges from an unchanged slot into a changed one: the case only the
            // escape re-resolution finds.
            let changed: BTreeSet<u8> = changes.iter().map(|c| c.0).collect();
            recovered_incoming += full
                .edge_set()
                .iter()
                .filter(|e| !changed.contains(&e.1) && changed.contains(&e.2))
                .count();
            assert_eq!(
                inc.escape_set(),
                full.escape_set(),
                "escapes after rebinding {k}"
            );
            let full_tile = Tile::from_csr(&full);
            for &p in &PREDS {
                assert_eq!(
                    inc_tile.plane(p),
                    full_tile.plane(p),
                    "tile after rebinding {k}"
                );
            }
            // A changed focus refuses the old preparation.
            assert_eq!(
                csr.reach(&nw, &PREDS, &slot_set([0]), 1).err(),
                Some(Refusal::FocusChanged)
            );
        }
    }
    assert!(
        recovered_incoming > 0,
        "the fixture must contain edges into replaced slots"
    );
}

/// PathMap's `restrict` cuts the focus's outgoing facts out of the fact trie:
/// exactly the facts whose subject is focused, and nothing else.
#[test]
fn trie_restrict_yields_exactly_the_focus_facts() {
    let wd = world();
    let trie = SpogTrie::build(&wd.facts);
    let run = SortedRun::build(&wd.facts, false, 1);
    let focus = discover(&run, &[17], &PREDS, SLOTS);
    let pairs = PREDS
        .iter()
        .flat_map(|&p| focus.ids().iter().map(move |&s| (p, s)));
    let cut = trie.restrict_to(pairs);
    let inf: BTreeSet<u32> = focus.ids().iter().copied().collect();
    let want = wd.facts.iter().filter(|f| inf.contains(&f[0])).count();
    assert_eq!(cut.len(), want);
    assert!(want > 0 && want < wd.facts.len());
    for &s in focus.ids() {
        for &p in &PREDS {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            cut.objects(p, s, |o, r| a.push((o, r)));
            trie.objects(p, s, |o, r| b.push((o, r)));
            assert_eq!(a, b);
        }
    }
}

/// Fixture 2: the inverted index. Every arm returns Doc21 and Doc39 as
/// canonical identities, on the small case and at scale.
#[test]
fn inverted_index_intersection_agrees_across_arms() {
    let mut postings: Vec<Vec<u32>> = vec![vec![17, 21, 39], vec![21, 39, 82]];
    // Scale: 254 more terms over 65 536 docs.
    let mut rng = Rng(9);
    for _ in 0..254 {
        let mut v: Vec<u32> = (0..40).map(|_| rng.below(65_536) as u32).collect();
        v.sort_unstable();
        v.dedup();
        postings.push(v);
    }
    let pairs: Vec<[u32; 2]> = postings
        .iter()
        .enumerate()
        .flat_map(|(t, d)| d.iter().map(move |&x| [t as u32, x]))
        .collect();
    let trie = PostingTrie::build(&pairs);
    // Focus: the union of the two queried terms' docs (the scope).
    let mut scope: Vec<u32> = postings[0].iter().chain(&postings[1]).copied().collect();
    scope.sort_unstable();
    scope.dedup();
    let focus = Focus::bind(&scope, 1);
    let planes = membership_planes(&[&postings[0], &postings[1]], &focus);
    let mut scratch = Scratch::new(SLOT_WORDS, 2);
    let a = intersect_sorted(&postings[0], &postings[1]);
    let b = trie.intersect(0, 1);
    let c = intersect_planes(&planes[0], &planes[1], &focus, &mut scratch);
    let d = intersect_count(&planes[0], &planes[1], &mut scratch);
    assert_eq!(a, vec![21, 39]);
    assert_eq!(b, a);
    assert_eq!(c, a);
    assert_eq!(d, 2);
    // At scale, every pair of terms, sorted merge vs trie.
    for x in 0..postings.len() as u32 {
        let y = (x * 7 + 3) % postings.len() as u32;
        assert_eq!(
            trie.intersect(x, y),
            intersect_sorted(&postings[x as usize], &postings[y as usize])
        );
    }
}

/// Fixture 1: a focus over 65 536 places is a presence plane over the
/// canonical lanes. The native count equals reading each identity's lanes,
/// and no coordinate is copied.
#[test]
fn place_focus_counts_natively_over_canonical_lanes() {
    let places = places(65_536, 3);
    let mut rng = Rng(11);
    let mut ids: Vec<u32> = Vec::new();
    while ids.len() < SLOTS {
        let i = rng.below(65_536) as u32;
        if !ids.contains(&i) {
            ids.push(i);
        }
    }
    let focus = FocusMask::over(65_536, &ids, Ordering::Canonical, 1);
    let mut scratch = Scratch::new(lance_graph_mask_risc::words_for(65_536), 2);
    let lanes = FocusLanes::load(&places, &ids, 1);
    assert_eq!(
        lanes.bytes(),
        SLOTS * 12,
        "the register copy is the only payload copied"
    );
    let mut small = Scratch::new(SLOT_WORDS, 2);
    let mut nonzero = 0;
    for k in 0..64 {
        let q = BoxQuery {
            lat: (-90_000_000 + k * 2_000_000, 90_000_000),
            lon: (-180_000_000, 180_000_000 - k * 4_000_000),
            class: (k as u32) % 16,
        };
        let native = box_count(&places, &focus, &q, &mut scratch).unwrap();
        assert_eq!(native, box_count_by_id(&places, &ids, &q));
        assert_eq!(lanes.count(1, &q, &mut small).unwrap(), native);
        nonzero += usize::from(native > 0);
    }
    assert!(nonzero > 8, "the queries must select something: {nonzero}");
    // The copy is tied to its focus generation.
    let q = BoxQuery {
        lat: (-90_000_000, 90_000_000),
        lon: (-180_000_000, 180_000_000),
        class: 0,
    };
    assert_eq!(lanes.count(2, &q, &mut small), Err(Refusal::FocusChanged));
}

/// Two row orders both use 16-bit positions. A focus built in Morton order is
/// refused against canonical lanes; read as if canonical, it would select
/// different places (the falsifier for "equal ordinals, equal identities").
#[test]
fn a_mask_in_another_row_order_is_refused() {
    let places = places(65_536, 3);
    let morton = morton_order(&places);
    let ids: Vec<u32> = (0..SLOTS as u32).collect(); // Morton positions 0..256
    let as_morton = FocusMask::over(65_536, &ids, Ordering::Morton(1), 1);
    let mut scratch = Scratch::new(lance_graph_mask_risc::words_for(65_536), 2);
    let q = BoxQuery {
        lat: (-90_000_000, 90_000_000),
        lon: (-180_000_000, 180_000_000),
        class: 3,
    };
    assert_eq!(
        box_count(&places, &as_morton, &q, &mut scratch),
        Err(Refusal::OrderingMismatch)
    );
    // What the refusal prevents: positions 0..256 in Morton order are not
    // rows 0..256.
    let rows: Vec<u32> = morton[..SLOTS].to_vec();
    let right = box_count_by_id(&places, &rows, &q);
    let wrong = box_count_by_id(&places, &ids, &q);
    assert_ne!(rows, ids);
    assert_ne!(
        right, wrong,
        "the two readings must disagree on this fixture"
    );
}
