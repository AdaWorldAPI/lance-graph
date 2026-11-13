//! H7 and H13: admission is unchanged by a recipe, and a DT64 method can
//! name one without naming a layout.

use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::FrameRefusal;
use lance_graph_dt64::{
    CoordSpace, Guard, LaneRefusal, MethodBinding, MethodId, SigRev, ValueDemand,
};
use virtual_address_lab::grid::{Grid, World, CLASS};
use virtual_address_lab::recipe::Recipe;
use virtual_address_lab::route::plan_sum_where_eq;
use virtual_address_lab::view::{TableRefusal, Trace, VirtualTable};

const LANES: usize = 6;
const LEN: usize = 500;

fn grid() -> Grid {
    Grid::new(LANES, LEN, |l, p| ((l * 31 + p * 7) % 13) as i32)
}

fn guard(recipe: Recipe) -> Guard {
    Guard::new(
        SetExpr::And(vec![SetExpr::Sym(recipe.digest())]),
        Logic::TwoValued,
    )
}

fn method(id: u32, recipe: Recipe) -> MethodBinding {
    let all: Vec<u8> = (0..LANES as u8).collect();
    MethodBinding {
        id: MethodId(id),
        sig_rev: SigRev(1),
        name: format!("view_{}", recipe.name()),
        on: WideFieldMask::from_positions(&all),
        when: Some(guard(recipe)),
        reads: WideFieldMask::from_positions(&all),
        effects: Vec::new(),
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

/// H7. FAILS IF: a recipe lets a table be formed over a moved frame, another
/// row order or another source.
#[test]
fn a_recipe_does_not_change_what_is_admitted() {
    let world = World::new(LANES);
    let mut g = grid();
    let trace = Trace::default();
    let binding = world.bind(&g).unwrap();
    for &recipe in &Recipe::ALL {
        assert!(VirtualTable::bound(&world, &binding, &g, recipe, &trace).is_ok());
    }
    // Another row order than the reader asked for.
    let asked = CoordSpace {
        population: 2,
        coordinates: 1,
    };
    assert!(matches!(
        world.bind_as(&g, asked),
        Err(LaneRefusal::CoordinateSpace { .. })
    ));
    // Another source of the same shape.
    let other = grid();
    for &recipe in &Recipe::ALL {
        assert_eq!(
            VirtualTable::bound(&world, &binding, &other, recipe, &trace).err(),
            Some(TableRefusal::Lanes(LaneRefusal::Frame(
                FrameRefusal::ForeignFrame
            )))
        );
    }
    // The frame moves after binding.
    g.advance_source();
    for &recipe in &Recipe::ALL {
        assert!(matches!(
            VirtualTable::bound(&world, &binding, &g, recipe, &trace).err(),
            Some(TableRefusal::Lanes(LaneRefusal::Frame(_)))
        ));
    }
    let fresh = world.bind(&g).unwrap();
    g.reorder(17);
    assert!(matches!(
        VirtualTable::bound(&world, &fresh, &g, Recipe::TRANSPOSE, &trace).err(),
        Some(TableRefusal::Lanes(_))
    ));
}

/// H13. FAILS IF: a method naming one recipe is admitted under another
/// recipe's description, or the answer depends on the source's lane ids.
/// The last block pins the limit: DT64 compares the number and reads nothing
/// into it.
#[test]
fn a_method_names_a_recipe_and_no_layout() {
    let world = World::with(
        LANES,
        vec![method(1, Recipe::TRANSPOSE), method(2, Recipe::ROTATE90)],
    );
    let admitted = |id: u32, described: Recipe| -> Option<Recipe> {
        let mref = world.store.method_ref(CLASS, MethodId(id)).ok()?;
        let bound = world.store.admit(&mref).ok()?;
        (bound.when.as_ref()?.witness() == guard(described).witness()).then_some(described)
    };
    assert_eq!(admitted(1, Recipe::TRANSPOSE), Some(Recipe::TRANSPOSE));
    assert_eq!(admitted(2, Recipe::ROTATE90), Some(Recipe::ROTATE90));
    assert_eq!(admitted(1, Recipe::ROTATE90), None);
    assert_eq!(admitted(2, Recipe::TRANSPOSE), None);
    assert_eq!(admitted(3, Recipe::TRANSPOSE), None);

    // The same method over two sources with other lane ids: one answer.
    let a = grid();
    let b = grid().with_lane_ids(vec![40, 31, 22, 13, 4, 55]);
    let want: i64 = (0..LEN)
        .filter(|&p| a.cell(4, p) == 3)
        .map(|p| i64::from(a.cell(2, p)))
        .sum();
    assert!(want > 0);
    let recipe = admitted(2, Recipe::ROTATE90).unwrap();
    for source in [&a, &b] {
        let binding = world.bind(source).unwrap();
        let trace = Trace::default();
        let table = VirtualTable::bound(&world, &binding, source, recipe, &trace).unwrap();
        let mut plan = plan_sum_where_eq(&table, 2, 4, 3).unwrap();
        assert_eq!(plan.sum(&table), Ok(want));
    }
    assert_ne!(
        world.bind(&a).unwrap().addr(2).unwrap().lane,
        world.bind(&b).unwrap().addr(2).unwrap().lane
    );

    // The limit: the guard holds a number. Two recipes with one digest would
    // be one method to DT64, and nothing derives the address map from it.
    assert_ne!(Recipe::TRANSPOSE.digest(), Recipe::ROTATE90.digest());
    assert_eq!(
        guard(Recipe::TRANSPOSE).witness(),
        Guard::new(
            SetExpr::And(vec![SetExpr::Sym(Recipe::TRANSPOSE.digest())]),
            Logic::TwoValued
        )
        .witness()
    );
}
