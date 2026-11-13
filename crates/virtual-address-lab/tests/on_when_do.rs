//! Plan section 16: DT64's ON x WHEN x DO applies to the address. A method is
//! found, admitted and its demand known before any value is read.
//!
//! Lab fixture only: the behaviour lens, its polyfill word and the `Grid` are
//! this lab's. No guard is evaluated and no effect is executed.

use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::{
    AlgebraRef, Arg, Effect, Guard, MethodBinding, MethodId, Refusal, SchemaRev, SigRev,
    ValueDemand,
};
use virtual_address_lab::grid::{Grid, World, BEHAVIOUR, POLYFILL};
use virtual_address_lab::method::{Cell, Method};
use virtual_address_lab::recipe::Recipe;
use virtual_address_lab::view::Ledger;

const FIELDS: usize = 3;
const ROWS: usize = 64;
const AMOUNT: u8 = 0;
const RATE: u8 = 1;
const TOTAL: u8 = 2;
const RECOMPUTE: MethodId = MethodId(1);
const FLAG: MethodId = MethodId(2);

fn value(lane: usize, pos: usize) -> i32 {
    (lane * 1_000 + (pos * 37) % 1_000) as i32
}

fn mask(positions: &[u8]) -> WideFieldMask {
    WideFieldMask::from_positions(positions)
}

/// ON amount or rate, unguarded: total := word0(amount, rate). Needs values.
fn recompute() -> MethodBinding {
    MethodBinding {
        id: RECOMPUTE,
        sig_rev: SigRev(1),
        name: "recompute_total".into(),
        on: mask(&[AMOUNT, RATE]),
        when: None,
        reads: mask(&[AMOUNT, RATE]),
        effects: vec![Effect {
            target: TOTAL,
            algebra: AlgebraRef {
                lens: POLYFILL,
                word: 0,
            },
            args: vec![Arg::Field(AMOUNT), Arg::Field(RATE)],
        }],
        population: Population::Bag,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

/// ON total, guarded, map only: it needs no value at all.
fn flag() -> MethodBinding {
    MethodBinding {
        id: FLAG,
        sig_rev: SigRev(1),
        name: "flag_total_changed".into(),
        on: mask(&[TOTAL]),
        when: Some(Guard::new(SetExpr::Sym(7), Logic::TwoValued)),
        reads: mask(&[]),
        effects: vec![],
        population: Population::Bag,
        truth: Truth::TwoValued,
        demand: ValueDemand::MapOnly,
    }
}

fn positions_of(m: &WideFieldMask) -> Vec<u8> {
    (0..FIELDS as u8).filter(|&p| m.has(p)).collect()
}

/// FAILS IF: finding, admitting or sizing a method reads a value or borrows a
/// selection; the orientation is not consulted for ON; a `Values` method
/// reads more than its `reads`; a `MapOnly` method reads anything.
#[test]
fn behaviour_is_decided_on_the_address_before_any_value_is_read() {
    let mut grid = Grid::new(FIELDS, ROWS, value);
    let world = World::with_behaviour(FIELDS, vec![2], vec![recompute(), flag()]);
    let sel = grid
        .issue_order(RATE)
        .expect("the source produces the order");
    let binding = world.bind(&grid).unwrap();
    let ledger = Ledger::default();
    let lens = world
        .store
        .resolve(&world.view, BEHAVIOUR, SchemaRev(1))
        .expect("the behaviour lens resolves by its full classid");

    // ---- 1. ON, from the address alone -----------------------------------
    let entity = Method::new(sel, Recipe::TRANSPOSE);
    let turned = Method::new(sel, Recipe::ANTI_TRANSPOSE);
    let (row, col) = (12, 0);
    let field = entity.field(FIELDS, row, col).unwrap();
    assert_eq!(field, AMOUNT);
    let fired: Vec<MethodId> = lens.methods_on(field).map(|m| m.id).collect();
    assert_eq!(fired, [RECOMPUTE]);
    // The same view cell under the other orientation is another field, and
    // another method fires: the address decided, not the cell's number.
    let other = turned.field(FIELDS, row, col).unwrap();
    assert_eq!(other, TOTAL);
    let fired_turned: Vec<MethodId> = lens.methods_on(other).map(|m| m.id).collect();
    assert_eq!(fired_turned, [FLAG]);
    // Every row of one view column dispatches alike: one decision for the
    // whole column, taken before any of its values exists as a read.
    for r in 0..ROWS {
        assert_eq!(entity.field(FIELDS, r, col), Some(AMOUNT));
    }
    assert_eq!((ledger.reads.get(), ledger.resolutions.get()), (0, 0));

    // ---- 2. Admission ----------------------------------------------------
    let reference = world.store.method_ref(BEHAVIOUR, RECOMPUTE).unwrap();
    let admitted = world.store.admit(&reference).expect("admitted");
    assert_eq!(admitted.id, RECOMPUTE);
    let mut dropped = world.store.method_ref(BEHAVIOUR, FLAG).unwrap();
    assert!(world.store.admit(&dropped).is_ok());
    dropped.guard = None;
    assert_eq!(
        world.store.admit(&dropped),
        Err(Refusal::GuardDropped {
            class: BEHAVIOUR,
            method: FLAG
        })
    );
    assert_eq!(ledger.reads.get(), 0, "admission read no value");

    // ---- 3. Demand: only now, and only what the binding reads -------------
    assert_eq!(admitted.demand, ValueDemand::Values);
    let n = grid.raw_positions(sel).len();
    let cell = entity.cell(FIELDS, n, row, col).unwrap();
    assert_eq!(
        cell,
        Cell {
            field: usize::from(AMOUNT),
            ordinal: row
        }
    );
    let wanted = positions_of(&admitted.reads);
    assert_eq!(wanted, [AMOUNT, RATE]);
    let operands: Vec<i32> = wanted
        .iter()
        .map(|&f| {
            entity
                .read_cell(
                    &world,
                    &binding,
                    &grid,
                    Cell {
                        field: usize::from(f),
                        ordinal: cell.ordinal,
                    },
                    &ledger,
                )
                .unwrap()
        })
        .collect();
    assert_eq!(ledger.reads.get(), 2, "exactly the demanded operands");
    let source_row = grid.raw_positions(sel)[row] as usize;
    assert_ne!(source_row, row, "the order is not the row order here");
    assert_eq!(
        operands,
        [grid.cell(0, source_row), grid.cell(1, source_row)]
    );

    // A map-only method demands nothing, so nothing is ever read for it.
    let map_only = world
        .store
        .admit(&world.store.method_ref(BEHAVIOUR, FLAG).unwrap())
        .unwrap();
    assert_eq!(map_only.demand, ValueDemand::MapOnly);
    assert!(positions_of(&map_only.reads).is_empty());
    assert_eq!(ledger.reads.get(), 2);

    // ---- 4. Control: materialise first ------------------------------------
    let eager = Ledger::default();
    let mut tile = vec![vec![0i32; FIELDS]; n];
    for (r, line) in tile.iter_mut().enumerate() {
        for (c, slot) in line.iter_mut().enumerate() {
            *slot = entity.read(&world, &binding, &grid, r, c, &eager).unwrap();
        }
    }
    assert_eq!(eager.reads.get(), (FIELDS * n) as u64);
    assert_eq!(
        [tile[row][usize::from(AMOUNT)], tile[row][usize::from(RATE)]],
        operands[..]
    );
    assert!(ledger.reads.get() * 90 < eager.reads.get());
}

/// FAILS IF: the methods can be bound to the data class's basis by its left
/// half, or a method outside the lens is admitted.
#[test]
fn the_behaviour_is_the_f_lens_and_nothing_else() {
    let world = World::with_behaviour(FIELDS, vec![2], vec![recompute(), flag()]);
    assert!(lance_graph_contract::class_view::is_behaviour_lens(
        BEHAVIOUR
    ));
    // The data class carries no method; the lens carries both.
    let data = world
        .store
        .resolve(&world.view, virtual_address_lab::grid::CLASS, SchemaRev(1))
        .unwrap();
    assert_eq!(data.methods.len(), 0);
    assert_eq!(
        world
            .store
            .method_ref(virtual_address_lab::grid::CLASS, RECOMPUTE),
        Err(Refusal::UnknownMethod {
            class: virtual_address_lab::grid::CLASS,
            method: RECOMPUTE
        })
    );
    // A lens the view does not know is refused, never read from its left half.
    assert_eq!(
        world
            .store
            .resolve(&world.view, 0x0A1A_F004, SchemaRev(1))
            .err(),
        Some(Refusal::UnknownClass(0x0A1A_F004))
    );
    assert_eq!(
        world.store.users_of(AlgebraRef {
            lens: POLYFILL,
            word: 0
        }),
        [(BEHAVIOUR, RECOMPUTE)]
    );
}
