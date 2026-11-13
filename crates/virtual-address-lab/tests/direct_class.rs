//! Plan section 18.5: routed behaviour on a normal data class. No `0xF###`
//! lens: the position routes to `CLASS`, which carries its own methods.
//! Lab fixture (`Grid`); no guard and no effect.

use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::{
    ClassSchema, Extent, LaneRefusal, MethodBinding, MethodId, SchemaRev, SigRev, ValueDemand,
};
use virtual_address_lab::behaviour::{admit_routed, behaviour, operands, Ledgers};
use virtual_address_lab::grid::{Grid, World, CLASS, POSITION, UNRELATED};
use virtual_address_lab::method::Method;
use virtual_address_lab::recipe::Recipe;
use virtual_address_lab::view::Refused;

const FIELDS: usize = 3;
const ROWS: usize = 64;
const AMOUNT: u8 = 0;
const RATE: u8 = 1;
const TOTAL: u8 = 2;
const READ: MethodId = MethodId(1);

fn value(lane: usize, pos: usize) -> i32 {
    (lane * 1_000 + (pos * 37) % 1_000) as i32
}

fn binding(name: &str, on: &[u8], reads: &[u8]) -> MethodBinding {
    MethodBinding {
        id: READ,
        sig_rev: SigRev(1),
        name: name.into(),
        on: WideFieldMask::from_positions(on),
        when: None,
        reads: WideFieldMask::from_positions(reads),
        effects: vec![],
        population: Population::Bag,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

/// The data class carries `read_rate`; a second class over the same basis
/// carries a method of the same id that reads more. Nothing routes to it.
fn world() -> World {
    let mut world = World::with(FIELDS, vec![binding("read_rate", &[AMOUNT], &[RATE])]);
    world
        .store
        .register(
            &world.view,
            UNRELATED,
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: FIELDS,
                extent: Extent::DR256_SQ_X64,
            },
            vec![binding("impostor", &[AMOUNT], &[AMOUNT, TOTAL])],
        )
        .expect("the foreign class registers");
    world
}

#[test]
fn a_routed_data_class_supplies_its_own_methods() {
    let mut grid = Grid::new(FIELDS, ROWS, value);
    let mut world = world();
    let sel = grid.issue_order(RATE).unwrap();
    let ledgers = Ledgers::default();
    let entity = Method::new(sel, Recipe::TRANSPOSE);

    // ON: the routed class is the data class, and it has the method.
    let class = behaviour(&world).unwrap();
    assert_eq!(class.class, CLASS);
    let field = entity.field(FIELDS, 0, 0).unwrap();
    assert_eq!(field, AMOUNT);
    let names: Vec<&str> = class.methods_on(field).map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["read_rate"]);

    // The foreign method: same id, ON the same field, accepted by `admit`
    // alone, refused for the position.
    let foreign = world.store.method_ref(UNRELATED, READ).unwrap();
    assert_eq!(world.store.admit(&foreign).unwrap().name, "impostor");
    assert_eq!(
        admit_routed(&world, &foreign).err(),
        Some(Refused::NotRoutedBehaviour {
            position: POSITION,
            routed: Some(CLASS),
            offered: UNRELATED
        })
    );
    let own = world.store.method_ref(CLASS, READ).unwrap();
    let admitted = admit_routed(&world, &own).unwrap().clone();
    assert_eq!(ledgers.operands.get(), 0, "ON and admission read nothing");

    // One demanded read: one value, of the declared field only.
    let positions: Vec<u32> = grid.raw_positions(sel).to_vec();
    let (lanes, values) = operands(&world, &grid, &admitted, entity, 9, &ledgers).unwrap();
    assert_eq!(ledgers.operands.get(), 1);
    assert_eq!(
        values,
        [(RATE, value(usize::from(RATE), positions[9] as usize))]
    );
    assert!(lanes.fields().has(RATE));
    for outside in [AMOUNT, TOTAL] {
        assert_eq!(
            lanes
                .selected(
                    &world.store,
                    &world.view,
                    grid.frame_store(),
                    &grid,
                    outside,
                    sel
                )
                .err(),
            Some(LaneRefusal::NotBound { field: outside })
        );
    }

    // Every ordinal against the generator, in the selection's order.
    assert!(positions.windows(2).any(|w| w[0] > w[1]), "a real reorder");
    for (ordinal, &pos) in positions.iter().enumerate() {
        let (_, v) = operands(&world, &grid, &admitted, entity, ordinal, &ledgers).unwrap();
        assert_eq!(v, [(RATE, value(usize::from(RATE), pos as usize))]);
    }
    assert_eq!(ledgers.operands.get(), 1 + ROWS as u64);
    assert_eq!((ledgers.guard_runs.get(), ledgers.effects.get()), (0, 0));

    // The source moves its frame: the binding made before is refused.
    grid.advance_source();
    assert!(matches!(
        lanes
            .selected(
                &world.store,
                &world.view,
                grid.frame_store(),
                &grid,
                RATE,
                sel
            )
            .err(),
        Some(LaneRefusal::Frame(_))
    ));

    // The position is routed elsewhere: the data class's method is refused.
    world.store.route(POSITION, UNRELATED).unwrap();
    assert_eq!(
        admit_routed(&world, &own).err(),
        Some(Refused::NotRoutedBehaviour {
            position: POSITION,
            routed: Some(UNRELATED),
            offered: CLASS
        })
    );
}
