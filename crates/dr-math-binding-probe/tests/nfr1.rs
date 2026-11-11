//! D-NFR-1: a dt64 frame on a bound Quack query.
//!
//! The source owner ([`Owner`]) holds the table and the data together; its
//! mutation methods are the only way content or mapping changes, and each one
//! advances the table. The oracle reads only the owner's plain `rows`; it
//! shares no code with Quack, mask-risc, dt64 or `nfr1`. The engine side
//! encodes the same rows into resident lanes and planes (local order over
//! tile 1) and asks Quack.
//!
//! "Raw" below means executing a query obtained earlier directly, bypassing
//! `admit`: the negative controls that show what happens without the guard.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};

use dr_math_binding_probe::nfr1::{FramedBinding, Kept, Nfr1Refusal, ResolvedUnder, Reuse};
use lance_graph_dt64::dr::{FrameRefusal, LocalMap, MapRead, MapTable, TileId, TileMap, Width};
use lance_graph_mask_risc::{
    Foreign, LaneRef, Out, Planes, Scratch, Value, execute_into, words_for,
};
use lance_graph_quack::bind::{Binder, BoundField, FieldKind, TableId, table as draft};
use lance_graph_quack::{Col, Mask, Query, lower};

// ── allocation counter ────────────────────────────────────────────────────

struct Counting;
thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}
fn allocs() -> u64 {
    ALLOCS.with(Cell::get)
}
// SAFETY: forwards to `System` unchanged; only counts calls.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static GLOBAL_ALLOC: Counting = Counting;

// ── fixture ───────────────────────────────────────────────────────────────

const EMP: u32 = 21;
const OTHER: u32 = 9;
const EMP_TILE: TileId = 1;
const OTHER_TILE: TileId = 2;
const EMP_N: u64 = 200;
const OTHER_N: u64 = 50;
const SEED: u64 = 0x5EED_0001;

#[derive(Clone, Copy, Debug)]
struct Row {
    class: u32,
    sel: Option<i32>,
    live: bool,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn shuffle(v: &mut [u64], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        v.swap(i, j);
    }
}

/// Shuffle until the order differs from the one it had.
fn reshuffle(v: &mut [u64], rng: &mut Rng) {
    let before = v.to_vec();
    while v == before.as_slice() {
        shuffle(v, rng);
    }
}

fn set_bit(words: &mut [u64], at: usize, on: bool) {
    if on {
        words[at / 64] |= 1 << (at % 64);
    } else {
        words[at / 64] &= !(1 << (at % 64));
    }
}

const LIVE: Mask = Mask(0);
const VSEL: Mask = Mask(1);
const CLASS: Col = Col(0);
const SEL: Col = Col(1);

/// The source owner: table and data live together.
struct Owner {
    table: MapTable,
    /// Tile 1's local order (mirror of the layout, for re-laying lanes).
    ids: Vec<u64>,
    other_ids: Vec<u64>,
    /// Plain values keyed by global id. The oracle reads only this.
    rows: BTreeMap<u64, Row>,
    // resident, in local order over tile 1
    live: Vec<u64>,
    vsel: Vec<u64>,
    class: Vec<u32>,
    sel: Vec<i32>,
    shift: u32,
    catalog_rev: u64,
}

impl Owner {
    fn new(seed: u64) -> Self {
        let mut rng = Rng(seed);
        let mut ids: Vec<u64> = (0..EMP_N).map(|i| 10_000 + 7 * i).collect();
        shuffle(&mut ids, &mut rng);
        let mut other_ids: Vec<u64> = (0..OTHER_N).map(|i| 50_000 + 3 * i).collect();
        shuffle(&mut other_ids, &mut rng);
        let mut rows = BTreeMap::new();
        for &id in &ids {
            let row = Row {
                class: if rng.below(10) < 6 { EMP } else { OTHER },
                sel: (rng.below(6) != 0).then(|| rng.below(3) as i32),
                live: rng.below(10) != 0,
            };
            rows.insert(id, row);
        }
        let mut table = MapTable::new(1, 1);
        table.advance(
            1,
            1,
            vec![
                (
                    EMP_TILE,
                    Some(LocalMap::prepare(Width::W256, &ids).expect("prepares")),
                ),
                (
                    OTHER_TILE,
                    Some(LocalMap::prepare(Width::W256, &other_ids).expect("prepares")),
                ),
            ],
        );
        let mut o = Self {
            table,
            ids,
            other_ids,
            rows,
            live: Vec::new(),
            vsel: Vec::new(),
            class: Vec::new(),
            sel: Vec::new(),
            shift: 0,
            catalog_rev: 1,
        };
        o.relay();
        o
    }

    /// Re-lay every resident lane and plane in the order of `ids`.
    fn relay(&mut self) {
        let n = self.ids.len();
        let mut live = vec![0u64; words_for(n)];
        let mut vsel = vec![0u64; words_for(n)];
        let (mut class, mut sel) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for (at, id) in self.ids.iter().enumerate() {
            let r = self.rows[id];
            set_bit(&mut live, at, r.live);
            set_bit(&mut vsel, at, r.sel.is_some());
            class.push(r.class + self.shift);
            sel.push(r.sel.unwrap_or(1));
        }
        self.live = live;
        self.vsel = vsel;
        self.class = class;
        self.sel = sel;
    }

    fn resolved(&self) -> ResolvedUnder {
        ResolvedUnder {
            relation_map: self.catalog_rev,
        }
    }

    /// Advance the table by bumping its source and relation-map revisions.
    fn advance(
        &mut self,
        source: u64,
        relation_map: u64,
        changes: Vec<(TileId, Option<LocalMap>)>,
    ) {
        let r = self.table.revisions();
        self.table
            .advance(r.source + source, r.relation_map + relation_map, changes);
    }

    /// Content only: the table's source revision moves, no mapping does.
    fn set_sel(&mut self, id: u64, v: Option<i32>) {
        self.rows.get_mut(&id).expect("row").sel = v;
        let at = self.ids.iter().position(|&g| g == id).expect("local");
        set_bit(&mut self.vsel, at, v.is_some());
        self.sel[at] = v.unwrap_or(1);
        self.advance(1, 0, vec![]);
    }

    /// Coordinates: tile 1 is re-laid in a new order; source does not move.
    fn reorder(&mut self, rng: &mut Rng) {
        reshuffle(&mut self.ids, rng);
        self.relay();
        let map = LocalMap::prepare(Width::W256, &self.ids).expect("prepares");
        self.advance(0, 0, vec![(EMP_TILE, Some(map))]);
    }

    /// Relation map: the class codes are renumbered.
    fn renumber_codebook(&mut self) {
        self.shift += 1;
        self.relay();
        self.catalog_rev += 1;
        self.advance(0, 1, vec![]);
    }

    /// A pure extension of tile 1 (the owner's content grew, so source moves).
    fn append(&mut self, id: u64, row: Row) {
        self.ids.push(id);
        self.rows.insert(id, row);
        self.relay();
        let map = LocalMap::prepare(Width::W256, &self.ids).expect("prepares");
        self.advance(1, 0, vec![(EMP_TILE, Some(map))]);
    }

    /// Tile 2 is re-laid; source and relation map do not move.
    fn touch_other_tile(&mut self, rng: &mut Rng) {
        reshuffle(&mut self.other_ids, rng);
        let map = LocalMap::prepare(Width::W256, &self.other_ids).expect("prepares");
        self.advance(0, 0, vec![(OTHER_TILE, Some(map))]);
    }

    /// A new frame at equal revisions.
    fn reseal(&mut self) {
        self.advance(0, 0, vec![]);
    }

    fn remove_tile(&mut self) {
        self.advance(0, 0, vec![(EMP_TILE, None)]);
    }
}

impl Binder for Owner {
    fn table(&self, name: &str) -> Option<TableId> {
        (name == "emp").then_some(TableId(0))
    }
    fn live(&self, _t: TableId) -> Mask {
        LIVE
    }
    fn field(&self, _t: TableId, name: &str) -> Option<BoundField> {
        match name {
            "class" => Some(BoundField {
                col: CLASS,
                kind: FieldKind::Code,
                validity: None,
            }),
            "sel" => Some(BoundField {
                col: SEL,
                kind: FieldKind::I32,
                validity: Some(VSEL),
            }),
            _ => None,
        }
    }
    fn code(&self, _t: TableId, _f: Col, literal: &str) -> Option<u32> {
        (literal == "employee").then_some(EMP + self.shift)
    }
}

/// `class = 'employee' AND sel = want`, bound through Quack's `Draft::bind`.
fn bind(o: &Owner, want: i32, count: bool) -> Query {
    let d = draft("emp")
        .where_eq("class", "employee")
        .where_eq("sel", want);
    let d = if count { d.count() } else { d.rows() };
    d.bind(o).expect("binds")
}

// ── engine side ───────────────────────────────────────────────────────────

fn run_rows(o: &Owner, q: &Query, n: usize) -> Vec<u64> {
    assert_eq!(n, o.class.len(), "n must be the resident length");
    let prog = lower(q).expect("lowers");
    let masks: [&[u64]; 2] = [&o.live, &o.vsel];
    let lanes = [LaneRef::U32(&o.class), LaneRef::I32(&o.sel)];
    let pl = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };
    let mut out = vec![0u64; words_for(n)];
    let mut s = Scratch::for_program(&prog, n).expect("carves");
    execute_into(&prog, &pl, &Foreign::NONE, &mut s, Out::Mask(&mut out)).expect("runs");
    out
}

fn run_count(o: &Owner, q: &Query, n: usize) -> usize {
    assert_eq!(n, o.class.len(), "n must be the resident length");
    let prog = lower(q).expect("lowers");
    let masks: [&[u64]; 2] = [&o.live, &o.vsel];
    let lanes = [LaneRef::U32(&o.class), LaneRef::I32(&o.sel)];
    let pl = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };
    let mut s = Scratch::for_program(&prog, n).expect("carves");
    match execute_into(&prog, &pl, &Foreign::NONE, &mut s, Out::None).expect("runs") {
        Value::Count(c) => c,
        other => panic!("count expected, got {other:?}"),
    }
}

// ── oracle side ───────────────────────────────────────────────────────────

/// Ids of `o.ids` whose row is live, an employee, and has `sel = want`
/// (three-valued: a NULL `sel` is not TRUE).
fn oracle(o: &Owner, want: i32) -> BTreeSet<u64> {
    o.ids
        .iter()
        .copied()
        .filter(|id| {
            let r = o.rows.get(id).expect("row");
            r.live && r.class == EMP && r.sel == Some(want)
        })
        .collect()
}

/// Test-side comparison only: set bits to the ids the layout names.
fn ids_of(mask: &[u64], layout: TileMap<'_>) -> BTreeSet<u64> {
    (0..layout.len())
        .filter(|&l| (mask[l / 64] >> (l % 64)) & 1 == 1)
        .map(|l| {
            layout
                .global(u8::try_from(l).expect("local fits u8"))
                .expect("local in layout")
        })
        .collect()
}

fn popcount(mask: &[u64]) -> usize {
    mask.iter().map(|w| w.count_ones() as usize).sum()
}

fn current_layout(o: &Owner) -> TileMap<'_> {
    o.table
        .map(&o.table.seal(), EMP_TILE)
        .expect("tile present")
}

fn fixture_ok(o: &Owner) {
    let want = oracle(o, 1);
    assert!(!want.is_empty(), "the oracle set must not be empty");
    assert!(
        want.len() * 3 < EMP_N as usize,
        "the oracle set must be under a third of the rows"
    );
    let rows = || o.ids.iter().map(|id| o.rows[id]);
    assert!(
        rows().any(|r| r.live && r.class == EMP && r.sel.is_none()),
        "a NULL-sel live employee must exist"
    );
    assert!(rows().any(|r| !r.live), "a non-live row must exist");
    assert!(rows().any(|r| r.class != EMP), "a non-employee must exist");
}

fn owner() -> Owner {
    let o = Owner::new(SEED);
    fixture_ok(&o);
    o
}

/// Flip the first `k` selected ids from `sel = 1` to `sel = 0`.
fn flip_selected(o: &mut Owner, k: usize) {
    let ids: Vec<u64> = oracle(o, 1).into_iter().take(k).collect();
    assert_eq!(ids.len(), k);
    for id in ids {
        o.set_sel(id, Some(0));
    }
}

fn seal(o: &Owner, q: Query) -> FramedBinding<Query> {
    FramedBinding::seal(&o.table, EMP_TILE, o.resolved(), q).expect("seals")
}

fn kept_rows(o: &Owner, b: &FramedBinding<Query>, calls: &Cell<u32>) -> Kept<Vec<u64>> {
    b.keep(&o.table, |q, l| {
        calls.set(calls.get() + 1);
        run_rows(o, q, l.len())
    })
    .expect("keeps")
}

// ── T1 ────────────────────────────────────────────────────────────────────

#[test]
fn the_first_execution_equals_the_oracle() {
    let o = owner();
    let want = oracle(&o, 1);

    let b = seal(&o, bind(&o, 1, false));
    let a = b.admit(&o.table).expect("admits");
    let bits = run_rows(&o, a.query(), a.layout().len());
    assert_eq!(ids_of(&bits, a.layout()), want);

    let c = seal(&o, bind(&o, 1, true));
    let a = c.admit(&o.table).expect("admits");
    assert_eq!(run_count(&o, a.query(), a.layout().len()), want.len());
}

// ── T2 ────────────────────────────────────────────────────────────────────

#[test]
fn mapping_replay_reproduces_the_assignment_after_every_advance() {
    let mut o = owner();
    let mut rng = Rng(0xA1);
    let b = seal(&o, bind(&o, 1, false));
    let captured = b.replay(&o.table).expect("replays").globals().to_vec();
    assert_eq!(captured.len(), EMP_N as usize);

    let check = |o: &Owner| {
        let replayed = b.replay(&o.table).expect("replays");
        assert_eq!(replayed.globals(), captured.as_slice());
        assert_eq!(replayed.len(), EMP_N as usize);
    };

    let first = oracle(&o, 1).into_iter().next().expect("selected");
    o.set_sel(first, Some(0));
    check(&o);
    o.reorder(&mut rng);
    assert_ne!(captured, o.ids, "the reorder must have moved the layout");
    check(&o);
    o.append(
        99_999,
        Row {
            class: EMP,
            sel: Some(1),
            live: true,
        },
    );
    check(&o);
    o.touch_other_tile(&mut rng);
    check(&o);
    o.renumber_codebook();
    check(&o);
    o.remove_tile();
    check(&o);
}

// ── T3 ────────────────────────────────────────────────────────────────────

#[test]
fn a_content_only_advance_reads_the_new_content() {
    let mut o = owner();
    let q = bind(&o, 1, false);
    let b = seal(&o, q);
    let captured = b.replay(&o.table).expect("replays").globals().to_vec();
    let before = oracle(&o, 1);

    flip_selected(&mut o, 3);
    let null_id = *o
        .ids
        .iter()
        .find(|id| {
            let r = o.rows[*id];
            r.live && r.class == EMP && r.sel.is_none()
        })
        .expect("a NULL-sel live employee");
    o.set_sel(null_id, Some(1));
    let after = oracle(&o, 1);
    assert_ne!(after, before);

    assert_ne!(
        b.frame().revisions().source,
        o.table.revisions().source,
        "the source revision moved"
    );
    assert!(!b.frame().same_issue(&o.table.seal()));

    let a = b
        .admit(&o.table)
        .expect("a content change is not a mapping change");
    let bits = run_rows(&o, a.query(), a.layout().len());
    let got = ids_of(&bits, a.layout());
    assert_eq!(got, after);
    assert_ne!(got, before);
    assert_eq!(a.layout().globals(), captured.as_slice());
}

// ── T4 ────────────────────────────────────────────────────────────────────

#[test]
fn a_coordinates_advance_refuses() {
    let mut o = owner();
    let mut rng = Rng(0x00C0_FFEE);
    let q = bind(&o, 1, false);
    let b = seal(&o, q.clone());

    o.reorder(&mut rng);
    let want = oracle(&o, 1);
    assert_eq!(
        b.admit(&o.table).err(),
        Some(Nfr1Refusal::Frame(FrameRefusal::CoordinatesChanged))
    );

    // Control: the stale query run raw, read through the layout it was bound
    // under, names other ids; read through the current layout it is right.
    let raw = run_rows(&o, &q, o.class.len());
    let old = b.replay(&o.table).expect("replays");
    assert_ne!(ids_of(&raw, old), want);
    assert_eq!(ids_of(&raw, current_layout(&o)), want);

    // A fresh seal admits and equals the oracle.
    let fresh = seal(&o, q);
    let a = fresh.admit(&o.table).expect("admits");
    let bits = run_rows(&o, a.query(), a.layout().len());
    assert_eq!(ids_of(&bits, a.layout()), want);
}

// ── T5 ────────────────────────────────────────────────────────────────────

#[test]
fn a_relation_map_advance_refuses() {
    let mut o = owner();
    let q = bind(&o, 1, true);
    let b = seal(&o, q.clone());

    o.renumber_codebook();
    let want = oracle(&o, 1);
    assert_eq!(
        b.admit(&o.table).err(),
        Some(Nfr1Refusal::Frame(FrameRefusal::MapChanged))
    );
    assert_eq!(
        b.replay(&o.table).expect("replays").globals(),
        current_layout(&o).globals(),
        "the layout did not move"
    );

    // Control: the old literal code means nothing on the renumbered lane.
    let raw = run_count(&o, &q, o.class.len());
    assert_ne!(raw, want.len());

    // A fresh bind + seal equals the oracle.
    let fresh = seal(&o, bind(&o, 1, true));
    let a = fresh.admit(&o.table).expect("admits");
    assert_eq!(run_count(&o, a.query(), a.layout().len()), want.len());
}

// ── T6 ────────────────────────────────────────────────────────────────────

#[test]
fn an_unrelated_tile_advance_stays_admitted() {
    let mut o = owner();
    let mut rng = Rng(0xB6);
    let b = seal(&o, bind(&o, 1, false));
    let before = o.table.revisions();

    o.touch_other_tile(&mut rng);
    let after = o.table.revisions();
    assert_ne!(
        before.coordinates, after.coordinates,
        "the whole-frame digest moved"
    );
    assert_eq!(before.source, after.source);
    assert_eq!(before.relation_map, after.relation_map);

    let a = b.admit(&o.table).expect("another tile moved, not this one");
    let bits = run_rows(&o, a.query(), a.layout().len());
    assert_eq!(ids_of(&bits, a.layout()), oracle(&o, 1));
}

// ── T7 ────────────────────────────────────────────────────────────────────

/// A pure extension is refused in the lab: a conservative choice. Whether a
/// prefix of the old layout may be admitted is left to S7.
#[test]
fn a_pure_extension_refuses_in_the_lab() {
    let mut o = owner();
    let b = seal(&o, bind(&o, 1, false));

    o.append(
        99_999,
        Row {
            class: EMP,
            sel: Some(1),
            live: true,
        },
    );
    assert_eq!(
        b.admit(&o.table).err(),
        Some(Nfr1Refusal::Frame(FrameRefusal::CoordinatesChanged))
    );
    let replayed = b.replay(&o.table).expect("replays");
    assert_eq!(replayed.len(), EMP_N as usize);
    let now = current_layout(&o);
    assert_eq!(now.len(), EMP_N as usize + 1);
    assert!(now.extends(replayed));
}

// ── T8 ────────────────────────────────────────────────────────────────────

#[test]
fn a_removed_tile_and_a_foreign_table_refuse() {
    let mut o = owner();
    let b = seal(&o, bind(&o, 1, false));
    assert!(b.admit(&o.table).is_ok(), "the binding's own table admits");

    let twin = Owner::new(SEED);
    assert_eq!(o.table.revisions(), twin.table.revisions());
    assert_eq!(o.table.seal().version(), twin.table.seal().version());
    assert_eq!(
        b.admit(&twin.table).err(),
        Some(Nfr1Refusal::Read(MapRead::Stale(
            FrameRefusal::ForeignFrame
        )))
    );

    o.remove_tile();
    assert!(
        b.replay(&o.table).is_ok(),
        "replay still answers the bound version"
    );
    assert_eq!(
        b.admit(&o.table).err(),
        Some(Nfr1Refusal::Read(MapRead::NoSuchTile(EMP_TILE)))
    );
}

// ── T9 ────────────────────────────────────────────────────────────────────

#[test]
fn seal_refuses_a_binder_resolved_under_another_relation_map() {
    let mut o = owner();
    let stale = o.resolved();
    let q = bind(&o, 1, false);
    o.renumber_codebook();

    assert_eq!(
        FramedBinding::seal(&o.table, EMP_TILE, stale, q.clone()).err(),
        Some(Nfr1Refusal::BinderNotFrame)
    );
    assert!(FramedBinding::seal(&o.table, EMP_TILE, o.resolved(), q.clone()).is_ok());
    assert_eq!(
        FramedBinding::seal(&o.table, 99, o.resolved(), q).err(),
        Some(Nfr1Refusal::Read(MapRead::NoSuchTile(99)))
    );
}

// ── T10 ───────────────────────────────────────────────────────────────────

#[test]
fn a_kept_mask_is_refused_after_a_content_change_and_recomputes() {
    let mut o = owner();
    let calls = Cell::new(0u32);
    let b = seal(&o, bind(&o, 1, false));
    let mut kept = kept_rows(&o, &b, &calls);
    let before = oracle(&o, 1);

    let first = kept.read(&b, &o.table).expect("reads").clone();
    assert_eq!(ids_of(&first, b.replay(&o.table).expect("replays")), before);
    assert_eq!(calls.get(), 1);

    flip_selected(&mut o, 3);
    let after = oracle(&o, 1);
    assert_eq!(after.len() + 3, before.len());

    assert_eq!(
        kept.read(&b, &o.table).err(),
        Some(Nfr1Refusal::Frame(FrameRefusal::SourceChanged))
    );
    assert!(
        b.admit(&o.table).is_ok(),
        "the address binding survives the same content change"
    );
    // Control: reusing the stale kept bits answers another count.
    assert_ne!(popcount(&first), after.len());

    let (bits, how) = kept
        .read_or_recompute(&b, &o.table, |q, l| {
            calls.set(calls.get() + 1);
            run_rows(&o, q, l.len())
        })
        .expect("recomputes");
    assert_eq!(how, Reuse::Recomputed);
    assert_eq!(ids_of(bits, b.replay(&o.table).expect("replays")), after);
    assert_eq!(calls.get(), 2);

    let (_, how) = kept
        .read_or_recompute(&b, &o.table, |q, l| {
            calls.set(calls.get() + 1);
            run_rows(&o, q, l.len())
        })
        .expect("reads");
    assert_eq!(how, Reuse::Reused);
    assert_eq!(calls.get(), 2);
}

// ── T11 ───────────────────────────────────────────────────────────────────

#[test]
fn a_kept_mask_survives_an_advance_that_moves_neither_content_nor_mapping() {
    let mut o = owner();
    let mut rng = Rng(0xB11);
    let calls = Cell::new(0u32);
    let b = seal(&o, bind(&o, 1, false));
    let mut kept = kept_rows(&o, &b, &calls);

    o.touch_other_tile(&mut rng);
    o.reseal();
    assert_ne!(
        b.frame().version().0,
        o.table.seal().version().0,
        "the version moved"
    );

    assert!(kept.read(&b, &o.table).is_ok());
    let (bits, how) = kept
        .read_or_recompute(&b, &o.table, |q, l| {
            calls.set(calls.get() + 1);
            run_rows(&o, q, l.len())
        })
        .expect("reads");
    assert_eq!(how, Reuse::Reused);
    assert_eq!(
        ids_of(bits, b.replay(&o.table).expect("replays")),
        oracle(&o, 1)
    );
    assert_eq!(calls.get(), 1);
}

// ── T12 ───────────────────────────────────────────────────────────────────

/// Keep a mask, apply `mutate`, then require both `read` and
/// `read_or_recompute` to refuse with `expect`, producing nothing more.
/// Returns the bits kept before the mutation and the owner after it.
fn kept_across_mapping_advance(
    mutate: impl FnOnce(&mut Owner),
    expect: FrameRefusal,
) -> (Vec<u64>, Owner) {
    let mut o = owner();
    let calls = Cell::new(0u32);
    let b = seal(&o, bind(&o, 1, false));
    let mut kept = kept_rows(&o, &b, &calls);
    let stale = kept.read(&b, &o.table).expect("reads").clone();

    mutate(&mut o);
    let refusal = Some(Nfr1Refusal::Frame(expect));
    assert_eq!(kept.read(&b, &o.table).err(), refusal);
    let again = kept
        .read_or_recompute(&b, &o.table, |q, l| {
            calls.set(calls.get() + 1);
            run_rows(&o, q, l.len())
        })
        .err();
    assert_eq!(again, refusal);
    assert_eq!(calls.get(), 1, "a mapping change never recomputes");
    (stale, o)
}

#[test]
fn a_kept_mask_never_survives_a_mapping_advance_even_by_recompute() {
    let (stale, o) = kept_across_mapping_advance(
        |o| o.reorder(&mut Rng(0x12)),
        FrameRefusal::CoordinatesChanged,
    );
    // Control: the kept bits through the current layout name other ids.
    assert_ne!(ids_of(&stale, current_layout(&o)), oracle(&o, 1));

    kept_across_mapping_advance(|o| o.renumber_codebook(), FrameRefusal::MapChanged);
}

// ── T13 ───────────────────────────────────────────────────────────────────

#[test]
fn a_kept_mask_belongs_to_the_binding_that_produced_it() {
    let o = owner();
    let calls = Cell::new(0u32);
    let a = seal(&o, bind(&o, 1, false));
    let b = seal(&o, bind(&o, 2, false));
    let ka = kept_rows(&o, &a, &calls);
    let kb = kept_rows(&o, &b, &calls);

    let bits_a = ka.read(&a, &o.table).expect("reads").clone();
    let bits_b = kb.read(&b, &o.table).expect("reads").clone();
    assert_ne!(bits_a, bits_b);

    assert_eq!(
        ka.read(&b, &o.table).err(),
        Some(Nfr1Refusal::NotThisBinding)
    );
    assert_eq!(
        kb.read(&a, &o.table).err(),
        Some(Nfr1Refusal::NotThisBinding)
    );
}

// ── T14 ───────────────────────────────────────────────────────────────────

#[test]
fn admission_allocates_nothing() {
    let o = owner();
    let calls = Cell::new(0u32);
    let b = seal(&o, bind(&o, 1, false));
    let kept = kept_rows(&o, &b, &calls);

    let measure = |extra_once: bool| -> u64 {
        let before = allocs();
        for i in 0..1_000 {
            assert!(b.admit(&o.table).is_ok());
            assert!(kept.read(&b, &o.table).is_ok());
            if extra_once && i == 0 {
                std::hint::black_box(Vec::<u64>::with_capacity(1));
            }
        }
        allocs() - before
    };

    assert!(b.admit(&o.table).is_ok(), "warm-up");
    assert_eq!(measure(false), 0, "admit and read allocate nothing");
    assert!(
        measure(true) >= 1,
        "the counter must see an allocation made in the measured loop"
    );
}
