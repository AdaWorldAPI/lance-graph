//! D-DRISA-1 falsifiers. Every refusal test has a success control; every
//! answer is compared with the [`Fixture`] oracle, which reads only global ids
//! and values and never a map, a position or a register.

use drisa_register_probe::{
    build_soa, build_table, lookups, Checks, Fixture, MethodOp, Refusal, RegisterFile,
};
use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_dt64::dr::{LocalMap, MapTable, Scope, TileId, Width};
use lance_graph_dt64::ValueDemand;
use lance_graph_mask_risc::{MaskOp, Operand, Program, Scratch, Terminal, Value};

const FIELDS: usize = 64;
const DEMANDED: [u8; 3] = [5, 17, 42];
const SCOPE: Scope = Scope { id: 7 };

struct World {
    fixture: Fixture,
    table: MapTable,
    tiles: Vec<TileId>,
    width: Width,
    selection: Vec<u64>,
}

fn world(n: usize, width: Width, seed: u64) -> World {
    let fixture = Fixture::new(n, FIELDS, seed);
    let (table, tiles) = build_table(&fixture, width, 1, seed ^ 0xBEEF);
    // Every 10th identity, so the selection includes absent identities too.
    let selection: Vec<u64> = fixture.ids().into_iter().step_by(10).collect();
    World {
        fixture,
        table,
        tiles,
        width,
        selection,
    }
}

fn authorized() -> WideFieldMask {
    WideFieldMask::from_positions(&DEMANDED)
}

fn bind(
    w: &World,
    soa: &drisa_register_probe::Soa,
    checks: Checks,
) -> Result<RegisterFile, Refusal> {
    RegisterFile::bind(
        &w.table,
        &w.table.seal(),
        &w.tiles,
        w.width,
        SCOPE,
        FIELDS,
        &authorized(),
        &DEMANDED,
        ValueDemand::Values,
        &w.selection,
        soa,
        checks,
    )
}

fn run(
    file: &RegisterFile,
    w: &World,
    soa: &drisa_register_probe::Soa,
    op: MethodOp,
) -> Result<Value, Refusal> {
    let c = file.compile(op)?;
    let mut s = file.scratch(&c)?;
    file.run(&w.table, soa, SCOPE, &c, &mut s)
}

const OPS: [MethodOp; 4] = [
    MethodOp::Count,
    MethodOp::Exists,
    MethodOp::CountWhereGt { field: 17, t: -5 },
    MethodOp::SumWhereGt {
        field: 17,
        t: -5,
        sum: 42,
    },
];

fn expect(w: &World, op: MethodOp) -> Value {
    match op {
        MethodOp::Count => Value::Count(w.fixture.count(&w.selection)),
        MethodOp::Exists => Value::Bool(w.fixture.count(&w.selection) > 0),
        MethodOp::CountWhereGt { field, t } => {
            Value::Count(w.fixture.where_gt(&w.selection, field, t, field).0)
        }
        MethodOp::SumWhereGt { field, t, sum } => {
            Value::SumI64(w.fixture.where_gt(&w.selection, field, t, sum).1)
        }
        MethodOp::CountDerived => unreachable!(),
    }
}

/// DR64, DR256 and 256 DR256 tiles: every method equals the oracle.
#[test]
fn every_method_matches_the_oracle_on_dr64_dr256_and_multi_tile() {
    for (n, width) in [(64, Width::W64), (256, Width::W256), (65_536, Width::W256)] {
        let w = world(n, width, 11);
        let soa = build_soa(&w.fixture, &w.table, &w.tiles, w.width);
        let file = bind(&w, &soa, Checks::ALL).unwrap();
        for op in OPS {
            assert_eq!(
                run(&file, &w, &soa, op).unwrap(),
                expect(&w, op),
                "{width:?} n={n} {op:?}"
            );
        }
        // Anti-vacuity: the value predicate keeps some and drops some.
        let (kept, _) = w.fixture.where_gt(&w.selection, 17, -5, 17);
        let present = w.fixture.count(&w.selection);
        assert!(kept > 0 && kept < present, "n={n}: {kept} of {present}");
    }
}

/// The executed planes borrow the SoA's own bytes; the register file owns
/// only the selection bits.
#[test]
fn execution_borrows_the_soa_and_copies_no_value() {
    let w = world(65_536, Width::W256, 12);
    let soa = build_soa(&w.fixture, &w.table, &w.tiles, w.width);
    let file = bind(&w, &soa, Checks::ALL).unwrap();
    file.with_planes(&w.table, &soa, SCOPE, ValueDemand::Values, |p| {
        assert_eq!(p.masks[0].as_ptr(), soa.presence_ptr());
        for (lane, &field) in p.lanes.iter().zip(&DEMANDED) {
            let lance_graph_mask_risc::LaneRef::I32(v) = lane else {
                panic!("i32 lane")
            };
            assert_eq!(v.as_ptr(), soa.field_ptr(field), "field {field}");
        }
    })
    .unwrap();
    assert_eq!(file.owned_bytes(), soa.n_rows() / 8);
}

/// Routing happens once, at bind: 256 runs of four different methods add no
/// global → local lookup.
#[test]
fn many_methods_reuse_one_binding_without_rerouting() {
    let w = world(65_536, Width::W256, 13);
    let soa = build_soa(&w.fixture, &w.table, &w.tiles, w.width);
    let before = lookups(&w.table, &w.tiles);
    let file = bind(&w, &soa, Checks::ALL).unwrap();
    let after_bind = lookups(&w.table, &w.tiles);
    assert!(after_bind - before >= w.selection.len() as u64);
    let compiled: Vec<_> = OPS.iter().map(|&op| file.compile(op).unwrap()).collect();
    let mut scratch: Vec<_> = compiled.iter().map(|c| file.scratch(c).unwrap()).collect();
    for _ in 0..256 {
        for ((c, s), &op) in compiled.iter().zip(&mut scratch).zip(&OPS) {
            assert_eq!(
                file.run(&w.table, &soa, SCOPE, c, s).unwrap(),
                expect(&w, op)
            );
        }
    }
    assert_eq!(
        lookups(&w.table, &w.tiles),
        after_bind,
        "no routing per run"
    );
}

/// A frame advance makes the old register file refuse; a fresh bind under
/// the new frame answers once the SoA owner has rebuilt the layout. With the
/// frame check off the old file still answers: the refusal is that check's.
#[test]
fn a_stale_frame_is_refused_and_a_fresh_bind_answers() {
    // 200 ids in one DR256 tile, so the tile has room for an extension.
    let mut w = world(200, Width::W256, 14);
    let soa = build_soa(&w.fixture, &w.table, &w.tiles, w.width);
    let file = bind(&w, &soa, Checks::ALL).unwrap();
    let lax = bind(
        &w,
        &soa,
        Checks {
            frame: false,
            ..Checks::ALL
        },
    )
    .unwrap();
    assert!(run(&file, &w, &soa, MethodOp::Count).is_ok());
    let frame = w.table.seal();
    let mut globals = w.table.map(&frame, w.tiles[0]).unwrap().globals().to_vec();
    globals.push(0xDEAD_BEEF);
    w.table.advance(
        1,
        1,
        vec![(
            w.tiles[0],
            Some(LocalMap::prepare(Width::W256, &globals).unwrap()),
        )],
    );
    assert!(matches!(
        run(&file, &w, &soa, MethodOp::Count),
        Err(Refusal::StaleFrame { .. })
    ));
    assert_eq!(
        run(&lax, &w, &soa, MethodOp::Count).unwrap(),
        expect(&w, MethodOp::Count)
    );
    // Under the new frame this SoA's layout digest is the old one: refused
    // until its owner rebuilds it.
    assert!(matches!(
        bind(&w, &soa, Checks::ALL),
        Err(Refusal::CoordinateMismatch { .. })
    ));
    let rebuilt = build_soa(&w.fixture, &w.table, &w.tiles, w.width);
    let fresh = bind(&w, &rebuilt, Checks::ALL).unwrap();
    assert_eq!(
        run(&fresh, &w, &rebuilt, MethodOp::Count).unwrap(),
        expect(&w, MethodOp::Count)
    );
}

/// A value write: map-only methods keep answering (and stay correct); value
/// methods and the derived register refuse until the value witness is
/// re-taken, which re-routes nothing.
#[test]
fn a_value_write_invalidates_values_but_not_the_map() {
    let mut w = world(65_536, Width::W256, 15);
    let mut soa = build_soa(&w.fixture, &w.table, &w.tiles, w.width);
    let mut file = bind(&w, &soa, Checks::ALL).unwrap();
    file.derive_gt(&w.table, &soa, SCOPE, 17, -5).unwrap();
    let derived_before = w.fixture.where_gt(&w.selection, 17, -5, 17).0;
    assert_eq!(
        run(&file, &w, &soa, MethodOp::CountDerived).unwrap(),
        Value::Count(derived_before)
    );

    // Flip one selected, present identity across the threshold.
    let g = *w
        .selection
        .iter()
        .find(|&&g| w.fixture.present(g) && w.fixture.where_gt(&[g], 17, -5, 17).0 == 1)
        .unwrap();
    let row = soa.row_of(&w.table, g).unwrap();
    soa.write(row, 17, -50);
    w.fixture.set(g, 17, -50);

    assert_eq!(
        run(&file, &w, &soa, MethodOp::Count).unwrap(),
        expect(&w, MethodOp::Count)
    );
    for op in [
        MethodOp::CountWhereGt { field: 17, t: -5 },
        MethodOp::CountDerived,
    ] {
        assert!(
            matches!(run(&file, &w, &soa, op), Err(Refusal::StaleValues { .. })),
            "{op:?}"
        );
    }
    let routed = lookups(&w.table, &w.tiles);
    file.rebind_values(&w.table, &soa).unwrap();
    assert_eq!(
        lookups(&w.table, &w.tiles),
        routed,
        "rebinding values routes nothing"
    );
    let op = MethodOp::CountWhereGt { field: 17, t: -5 };
    assert_eq!(run(&file, &w, &soa, op).unwrap(), expect(&w, op));
    // The derived register is still the old one: it must refuse, not answer.
    assert!(matches!(
        run(&file, &w, &soa, MethodOp::CountDerived),
        Err(Refusal::StaleValues { .. })
    ));
    file.derive_gt(&w.table, &soa, SCOPE, 17, -5).unwrap();
    assert_eq!(
        run(&file, &w, &soa, MethodOp::CountDerived).unwrap(),
        Value::Count(derived_before - 1)
    );
}

/// Disable: without the value check, the stale derived register answers, and
/// the answer is wrong.
#[test]
fn disabling_the_value_check_returns_a_wrong_derived_answer() {
    let mut w = world(65_536, Width::W256, 16);
    let mut soa = build_soa(&w.fixture, &w.table, &w.tiles, w.width);
    let mut file = bind(
        &w,
        &soa,
        Checks {
            values: false,
            ..Checks::ALL
        },
    )
    .unwrap();
    file.derive_gt(&w.table, &soa, SCOPE, 17, -5).unwrap();
    let g = *w
        .selection
        .iter()
        .find(|&&g| w.fixture.present(g) && w.fixture.where_gt(&[g], 17, -5, 17).0 == 1)
        .unwrap();
    let row = soa.row_of(&w.table, g).unwrap();
    soa.write(row, 17, -50);
    w.fixture.set(g, 17, -50);
    let truth = w.fixture.where_gt(&w.selection, 17, -5, 17).0;
    assert_ne!(
        run(&file, &w, &soa, MethodOp::CountDerived).unwrap(),
        Value::Count(truth)
    );
}

/// A SoA of another source is refused; the same SoA under its own table is
/// accepted.
#[test]
fn a_soa_of_another_source_is_refused() {
    let w = world(256, Width::W256, 17);
    let (other, other_tiles) = build_table(&w.fixture, Width::W256, 2, 17 ^ 0xBEEF);
    let foreign = build_soa(&w.fixture, &other, &other_tiles, Width::W256);
    assert!(matches!(
        bind(&w, &foreign, Checks::ALL),
        Err(Refusal::SourceMismatch { frame: 1, soa: 2 })
    ));
    let own = build_soa(&w.fixture, &w.table, &w.tiles, w.width);
    assert!(bind(&w, &own, Checks::ALL).is_ok());
}

/// Same ids, another local order: refused; with the coordinate check off the
/// value answers diverge from the oracle (the permutation is not harmless).
#[test]
fn a_permuted_coordinate_assignment_is_refused_and_would_be_wrong() {
    let w = world(65_536, Width::W256, 18);
    let (permuted, _) = build_table(&w.fixture, Width::W256, 1, 0x1234);
    let soa_other_layout = build_soa(&w.fixture, &permuted, &w.tiles, Width::W256);
    assert!(matches!(
        bind(&w, &soa_other_layout, Checks::ALL),
        Err(Refusal::CoordinateMismatch { .. })
    ));
    let lax = bind(
        &w,
        &soa_other_layout,
        Checks {
            coordinates: false,
            ..Checks::ALL
        },
    )
    .unwrap();
    let op = MethodOp::SumWhereGt {
        field: 17,
        t: -5,
        sum: 42,
    };
    assert_ne!(
        run(&lax, &w, &soa_other_layout, op).unwrap(),
        expect(&w, op)
    );
}

/// Unknown, unauthorized and unbound fields each refuse with their own
/// reason; the authorized demand binds.
#[test]
fn field_refusals_are_typed() {
    let w = world(256, Width::W256, 19);
    let soa = build_soa(&w.fixture, &w.table, &w.tiles, w.width);
    let try_bind = |demanded: &[u8]| {
        RegisterFile::bind(
            &w.table,
            &w.table.seal(),
            &w.tiles,
            w.width,
            SCOPE,
            FIELDS,
            &authorized(),
            demanded,
            ValueDemand::Values,
            &w.selection,
            &soa,
            Checks::ALL,
        )
    };
    assert!(matches!(try_bind(&[5, 64]), Err(Refusal::UnknownField(64))));
    assert!(matches!(try_bind(&[5, 6]), Err(Refusal::Unauthorized(6))));
    let file = try_bind(&DEMANDED).unwrap();
    assert!(matches!(
        file.compile(MethodOp::CountWhereGt { field: 9, t: 0 }),
        Err(Refusal::FieldNotBound(9))
    ));
    assert!(file
        .compile(MethodOp::CountWhereGt { field: 5, t: 0 })
        .is_ok());
}

/// Another scope refuses; with the scope check off it would answer.
#[test]
fn another_scope_is_refused() {
    let w = world(256, Width::W256, 20);
    let soa = build_soa(&w.fixture, &w.table, &w.tiles, w.width);
    let file = bind(&w, &soa, Checks::ALL).unwrap();
    let c = file.compile(MethodOp::Count).unwrap();
    let mut s = file.scratch(&c).unwrap();
    assert!(matches!(
        file.run(&w.table, &soa, Scope { id: 8 }, &c, &mut s),
        Err(Refusal::ScopeChanged)
    ));
    assert!(file.run(&w.table, &soa, SCOPE, &c, &mut s).is_ok());
    let lax = bind(
        &w,
        &soa,
        Checks {
            scope: false,
            ..Checks::ALL
        },
    )
    .unwrap();
    assert!(lax.run(&w.table, &soa, Scope { id: 8 }, &c, &mut s).is_ok());
}

/// A program that reads a scratch slot it never wrote is refused, even when a
/// previous run left a full mask there: scratch carries nothing across runs.
#[test]
fn stale_scratch_is_refused_by_mask_risc() {
    let w = world(256, Width::W256, 21);
    let soa = build_soa(&w.fixture, &w.table, &w.tiles, w.width);
    let file = bind(&w, &soa, Checks::ALL).unwrap();
    let c = file
        .compile(MethodOp::CountWhereGt { field: 17, t: -5 })
        .unwrap();
    let mut s = file.scratch(&c).unwrap();
    file.run(&w.table, &soa, SCOPE, &c, &mut s).unwrap();
    let reads_stale = Program::new(
        vec![MaskOp::And {
            a: Operand::Plane(0),
            b: Operand::Plane(1),
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(1),
        },
    );
    let r = file.with_planes(&w.table, &soa, SCOPE, ValueDemand::MapOnly, |p| {
        lance_graph_mask_risc::execute(&reads_stale, p, &mut s, None)
    });
    assert!(matches!(
        r,
        Ok(Err(
            lance_graph_mask_risc::ExecError::ScratchReadBeforeWrite { slot: 1 }
        ))
    ));
    let _ = Scratch::new(1, 1);
}

/// Disable: dropping the presence plane counts absent identities.
#[test]
fn disabling_presence_counts_absent_identities() {
    let w = world(65_536, Width::W256, 22);
    let soa = build_soa(&w.fixture, &w.table, &w.tiles, w.width);
    let file = bind(
        &w,
        &soa,
        Checks {
            presence: false,
            ..Checks::ALL
        },
    )
    .unwrap();
    assert_ne!(
        run(&file, &w, &soa, MethodOp::Count).unwrap(),
        expect(&w, MethodOp::Count)
    );
}

/// An id no bound tile holds, and an id given twice, are refused at bind.
#[test]
fn selection_refusals_are_typed() {
    let mut w = world(256, Width::W256, 23);
    let soa = build_soa(&w.fixture, &w.table, &w.tiles, w.width);
    w.selection.push(0x0123);
    assert!(matches!(
        bind(&w, &soa, Checks::ALL),
        Err(Refusal::UnmappedIdentity(0x0123))
    ));
    w.selection.pop();
    let first = w.selection[0];
    w.selection.push(first);
    assert!(
        matches!(bind(&w, &soa, Checks::ALL), Err(Refusal::DuplicateIdentity(g)) if g == first)
    );
}
