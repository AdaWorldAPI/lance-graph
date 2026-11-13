//! Plan sections 16 and 17: ON -> WHEN -> demanded operands on a witnessed
//! address, with no row or relation materialised first.
//!
//! Lab fixture only: the behaviour lenses, the polyfill word and the `Grid`
//! are this lab's. No effect is executed and nothing is written back.

use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_algebra::{Cmp, Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::FrameRefusal;
use lance_graph_dt64::{
    AlgebraRef, Arg, CoordSpace, Effect, Guard, LaneRefusal, MethodBinding, MethodId, Refusal,
    SigRev, ValueDemand,
};
use virtual_address_lab::behaviour::{admit_routed, behaviour, guard_mask, operands, Ledgers};
use virtual_address_lab::grid::{Grid, World, BEHAVIOUR, CLASS, POLYFILL, POSITION, UNRELATED};
use virtual_address_lab::method::Method;
use virtual_address_lab::recipe::Recipe;
use virtual_address_lab::view::Refused;

const FIELDS: usize = 3;
const ROWS: usize = 64;
const AMOUNT: u8 = 0;
const RATE: u8 = 1;
const TOTAL: u8 = 2;
const RECOMPUTE: MethodId = MethodId(1);
const FLAG: MethodId = MethodId(2);
const NULLABLE: MethodId = MethodId(3);
const THRESHOLD: i32 = 500;

fn value(lane: usize, pos: usize) -> i32 {
    (lane * 1_000 + (pos * 37) % 1_000) as i32
}

fn mask(positions: &[u8]) -> WideFieldMask {
    WideFieldMask::from_positions(positions)
}

fn amount_over(t: i32) -> Guard {
    Guard::new(
        SetExpr::i32(u16::from(AMOUNT), Cmp::Gt, t),
        Logic::TwoValued,
    )
}

fn binding(
    id: MethodId,
    name: &str,
    on: &[u8],
    when: Option<Guard>,
    reads: &[u8],
    demand: ValueDemand,
) -> MethodBinding {
    let effects = if reads.is_empty() {
        vec![]
    } else {
        vec![Effect {
            target: TOTAL,
            algebra: AlgebraRef {
                lens: POLYFILL,
                word: 0,
            },
            args: vec![Arg::Field(reads[0]), Arg::Field(reads[1])],
        }]
    };
    MethodBinding {
        id,
        sig_rev: SigRev(1),
        name: name.into(),
        on: mask(on),
        when,
        reads: mask(reads),
        effects,
        population: Population::Bag,
        truth: Truth::TwoValued,
        demand,
    }
}

/// The routed lens: a guarded method needing values, one guarded by a mask
/// plane, one three-valued.
fn methods() -> Vec<MethodBinding> {
    vec![
        binding(
            RECOMPUTE,
            "recompute_total",
            &[AMOUNT, RATE],
            Some(amount_over(THRESHOLD)),
            &[AMOUNT, RATE],
            ValueDemand::Values,
        ),
        binding(
            FLAG,
            "flag_total_changed",
            &[TOTAL],
            Some(Guard::new(SetExpr::Sym(7), Logic::TwoValued)),
            &[],
            ValueDemand::MapOnly,
        ),
        binding(
            NULLABLE,
            "nullable_rate",
            &[RATE],
            Some(Guard::new(
                SetExpr::i32(u16::from(RATE), Cmp::Ge, 0),
                Logic::Kleene,
            )),
            &[],
            ValueDemand::MapOnly,
        ),
    ]
}

/// The lens nothing routes to: the same method identity, ON the same field,
/// unguarded and reading everything.
fn impostor() -> Vec<MethodBinding> {
    vec![binding(
        RECOMPUTE,
        "impostor",
        &[AMOUNT],
        None,
        &[AMOUNT, TOTAL],
        ValueDemand::Values,
    )]
}

fn world() -> World {
    World::routed_to_behaviour(FIELDS, vec![2], methods(), impostor())
}

fn grid() -> Grid {
    Grid::new(FIELDS, ROWS, value).for_class(BEHAVIOUR)
}

fn checksum(grid: &Grid) -> i64 {
    (0..FIELDS)
        .flat_map(|l| grid.raw_lane(l).iter())
        .enumerate()
        .map(|(i, &v)| (i as i64 + 1) * i64::from(v))
        .sum()
}

/// Gate 1. FAILS IF: a lens the position does not route to supplies a method
/// for X, or the routed lens does not.
#[test]
fn only_the_routed_lens_is_the_behaviour_of_x() {
    let world = world();
    let lens = behaviour(&world).unwrap();
    assert_eq!(lens.class, BEHAVIOUR);
    assert_eq!(
        world
            .store
            .routed_at(world.route.version, POSITION)
            .unwrap(),
        Some(BEHAVIOUR)
    );

    // ON, from the address: only the routed lens's methods are candidates.
    let entity = Method::new(0, Recipe::TRANSPOSE);
    let field = entity.field(FIELDS, 12, 0).unwrap();
    assert_eq!(field, AMOUNT);
    let names: Vec<&str> = lens.methods_on(field).map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["recompute_total"]);

    // The unrelated lens is registered, has a compatible basis and a method
    // ON the same field under the same MethodId.
    let foreign = world.store.method_ref(UNRELATED, RECOMPUTE).unwrap();
    let alone = world.store.admit(&foreign).expect("admit alone accepts it");
    assert_eq!(alone.name, "impostor");
    assert!(alone.on.has(field));
    // The route refuses it for X.
    assert_eq!(
        admit_routed(&world, &foreign).err(),
        Some(Refused::NotRoutedBehaviour {
            position: POSITION,
            routed: Some(BEHAVIOUR),
            offered: UNRELATED
        })
    );
    let own = world.store.method_ref(BEHAVIOUR, RECOMPUTE).unwrap();
    assert_eq!(admit_routed(&world, &own).unwrap().name, "recompute_total");

    // A position routed to the DATA class has no behaviour at all: nothing
    // ties a separately registered lens to it (section 17.1, BLOCKED).
    let data = World::new(FIELDS);
    let resolved = behaviour(&data).unwrap();
    assert_eq!((resolved.class, resolved.methods.len()), (CLASS, 0));
}

/// Gate 2. FAILS IF: the guard mask differs from the scalar oracle, is all
/// true or all false, a bag's repeats are lost, or a guard is evaluated
/// under a moved frame, a foreign source, another row order, a mask plane
/// nobody issued or three-valued logic.
#[test]
fn when_filters_the_source_population_it_was_witnessed_for() {
    let mut grid = grid();
    let world = world();
    let order = grid.issue_order(RATE).unwrap();
    let bag = grid.issue_list(vec![14, 14, 5, 20, 14, 27]);
    let ledgers = Ledgers::default();
    let before = checksum(&grid);

    let own = world.store.method_ref(BEHAVIOUR, RECOMPUTE).unwrap();
    let admitted = admit_routed(&world, &own).unwrap();
    let guard = admitted.when.as_ref().unwrap();
    let held = guard_mask(&world, &grid, guard, grid.space(), &ledgers).unwrap();
    assert_eq!(
        (ledgers.guard_runs.get(), ledgers.guard_rows.get()),
        (1, ROWS as u64)
    );
    assert_eq!(ledgers.operands.get(), 0, "a guard reads no DO operand");

    // Independent oracle, over every source row and every ordinal.
    let oracle = |pos: usize| value(usize::from(AMOUNT), pos) > THRESHOLD;
    let mut truths = [0usize; 2];
    for pos in 0..ROWS {
        assert_eq!(held.holds(&world, &grid, pos as u32), Ok(oracle(pos)));
        truths[usize::from(oracle(pos))] += 1;
    }
    assert!(truths[0] > 5 && truths[1] > 5, "both outcomes occur");
    assert_eq!(held.rows_held(), truths[1] as u64);
    for (ordinal, &pos) in grid.raw_positions(order).iter().enumerate() {
        assert_eq!(
            held.holds(&world, &grid, pos),
            Ok(oracle(pos as usize)),
            "ordinal {ordinal}"
        );
    }

    // A bag is walked by occurrence. The mask is a membership set.
    let occurrences: Vec<u32> = grid.raw_positions(bag).to_vec();
    let surviving = occurrences
        .iter()
        .filter(|&&p| held.holds(&world, &grid, p).unwrap())
        .count();
    let mut distinct = occurrences.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let members = distinct.iter().filter(|&&p| oracle(p as usize)).count();
    assert_eq!((surviving, members), (5, 3));

    // Another guard expression is refused at admission.
    let mut other = own.clone();
    other.guard = Some(amount_over(THRESHOLD + 1).witness());
    assert_eq!(
        admit_routed(&world, &other).err(),
        Some(Refused::Bind(Refusal::GuardMismatch {
            class: BEHAVIOUR,
            method: RECOMPUTE
        }))
    );

    // A mask plane nobody issued, and three-valued logic: refused, 0 runs.
    let flag = admit_routed(&world, &world.store.method_ref(BEHAVIOUR, FLAG).unwrap()).unwrap();
    assert_eq!(
        guard_mask(
            &world,
            &grid,
            flag.when.as_ref().unwrap(),
            grid.space(),
            &ledgers
        )
        .err(),
        Some(Refused::GuardPlaneUnissued { sym: 7 })
    );
    let nullable = admit_routed(
        &world,
        &world.store.method_ref(BEHAVIOUR, NULLABLE).unwrap(),
    )
    .unwrap();
    assert_eq!(
        guard_mask(
            &world,
            &grid,
            nullable.when.as_ref().unwrap(),
            grid.space(),
            &ledgers
        )
        .err(),
        Some(Refused::GuardUnsupported("three-valued logic"))
    );
    assert_eq!(ledgers.guard_runs.get(), 1);

    // Another row order than the lanes': refused at the bind.
    let elsewhere = CoordSpace {
        population: 9,
        coordinates: 9,
    };
    assert!(matches!(
        guard_mask(&world, &grid, guard, elsewhere, &ledgers).err(),
        Some(Refused::Lane(LaneRefusal::CoordinateSpace { .. }))
    ));

    // A foreign source: the mask's binding belongs to another frame store.
    let twin = self::grid();
    assert_eq!(
        held.holds(&world, &twin, 0),
        Err(Refused::Lane(LaneRefusal::Frame(
            FrameRefusal::ForeignFrame
        )))
    );

    // The source moves its frame: the mask evaluated before is refused.
    assert_eq!(checksum(&grid), before, "the source was never written");
    grid.advance_source();
    assert!(matches!(
        held.holds(&world, &grid, 0),
        Err(Refused::Lane(LaneRefusal::Frame(_)))
    ));
    assert_eq!(ledgers.effects.get(), 0);
}

/// Gate 3. FAILS IF: an operand is read before ON, admission and a true
/// WHEN; a field outside `reads` can be read through the operand binding; or
/// a false WHEN still reads.
#[test]
fn only_a_surviving_method_demands_only_its_declared_operands() {
    let mut grid = grid();
    let world = world();
    let sel = grid.issue_order(RATE).unwrap();
    let ledgers = Ledgers::default();
    let before = checksum(&grid);
    let entity = Method::new(sel, Recipe::TRANSPOSE);

    // ON and admission, with every ledger at zero.
    let lens = behaviour(&world).unwrap();
    let field = entity.field(FIELDS, 0, 0).unwrap();
    let candidate = lens.methods_on(field).next().unwrap();
    let reference = world.store.method_ref(lens.class, candidate.id).unwrap();
    let admitted = admit_routed(&world, &reference).unwrap();
    assert_eq!(
        (ledgers.guard_runs.get(), ledgers.operands.get()),
        (0, 0),
        "ON and admission read nothing"
    );

    // WHEN, once for the population.
    let held = guard_mask(
        &world,
        &grid,
        admitted.when.as_ref().unwrap(),
        grid.space(),
        &ledgers,
    )
    .unwrap();
    assert_eq!(ledgers.operands.get(), 0);

    let positions: Vec<u32> = grid.raw_positions(sel).to_vec();
    let (mut fired, mut skipped) = (0u64, 0u64);
    for (ordinal, &pos) in positions.iter().enumerate() {
        let before_reads = ledgers.operands.get();
        if !held.holds(&world, &grid, pos).unwrap() {
            // A false WHEN: no operand binding is made, nothing is read.
            assert_eq!(ledgers.operands.get(), before_reads);
            skipped += 1;
            continue;
        }
        let (lanes, values) = operands(&world, &grid, admitted, entity, ordinal, &ledgers).unwrap();
        assert_eq!(ledgers.operands.get(), before_reads + 2);
        let p = pos as usize;
        assert_eq!(
            values,
            [
                (AMOUNT, value(usize::from(AMOUNT), p)),
                (RATE, value(usize::from(RATE), p))
            ]
        );
        assert!(values[0].1 > THRESHOLD);
        // The operand binding holds `reads` and nothing else.
        assert!(lanes.fields().has(AMOUNT) && lanes.fields().has(RATE));
        assert!(!lanes.fields().has(TOTAL));
        assert_eq!(
            lanes
                .selected(
                    &world.store,
                    &world.view,
                    grid.frame_store(),
                    &grid,
                    TOTAL,
                    sel
                )
                .err(),
            Some(LaneRefusal::NotBound { field: TOTAL })
        );
        fired += 1;
    }
    assert!(fired > 5 && skipped > 5, "both branches ran");
    assert_eq!(fired + skipped, ROWS as u64);
    assert_eq!(ledgers.operands.get(), 2 * fired);
    assert!(ledgers.operands.get() < (FIELDS * ROWS) as u64);
    assert_eq!(
        ledgers.guard_runs.get(),
        1,
        "one guard run for the population"
    );
    assert_eq!(ledgers.effects.get(), 0, "no effect ran");
    assert_eq!(checksum(&grid), before, "the source is unchanged");

    // A selection in another row order cannot supply the operands.
    let foreign_sel = grid.issue_list_in(
        vec![1, 2, 3],
        CoordSpace {
            population: 9,
            coordinates: 9,
        },
    );
    assert!(matches!(
        operands(
            &world,
            &grid,
            admitted,
            Method::new(foreign_sel, Recipe::TRANSPOSE),
            0,
            &ledgers
        )
        .err(),
        Some(Refused::Lane(LaneRefusal::SelectionSpace { .. }))
    ));
}

/// Gate 4 (review finding). FAILS IF: a 256-lane class binds fewer than 256
/// lanes.
#[test]
fn a_256_lane_class_binds_every_lane() {
    let grid = Grid::new(256, 4, |l, p| (l * 10 + p) as i32);
    let world = World::new(256);
    let binding = world.bind(&grid).unwrap();
    assert_eq!(binding.fields().count(), 256);
    assert!(binding.addr(0).is_some() && binding.addr(255).is_some());
    let lane = binding
        .lane(&world.store, &world.view, grid.frame_store(), &grid, 255)
        .unwrap();
    let lance_graph_mask_risc::LaneRef::I32(values) = lane else {
        panic!("an i32 lane")
    };
    assert_eq!(values, [2550, 2551, 2552, 2553]);
}
