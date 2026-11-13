//! Plan section 10: one version-bound address recipe reads exactly the
//! demanded property from a real `MailboxSoA` owner, through DT64's selection
//! contract. Oracles are plain arrays kept by the tests.

use causal_edge::CausalEdge64;
use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::soa_view::MailboxSoaView;
use lance_graph_dt64::{CoordSpace, LaneRefusal, LaneSource};
use lance_graph_mask_risc::{ExecError, LaneRef};
use lance_graph_report::bound_sel::{fold_bound_selection, BoundFold, BoundSelRefusal};
use virtual_address_lab::grid::World;
use virtual_address_lab::mailbox::{
    revisions_of, visit_selected, Cell64, MailboxLanes, EDGE, FIELDS, META,
};
use virtual_address_lab::recipe::Recipe;
use virtual_address_lab::view::{Refused, RowObject};

const N: usize = 64;
type Owner = MailboxSoA<N>;
const ENTITY: [Recipe; 1] = [Recipe::TRANSPOSE];

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The oracle: what row `r` holds. `meta` repeats (ties) on purpose.
fn meta_of(r: usize) -> u32 {
    (mix(r as u64) % 23) as u32 * 1_000
}
fn edge_of(r: usize) -> u64 {
    mix(1_000 + r as u64)
}

fn owner(populated: usize) -> Box<Owner> {
    let mut m: Box<Owner> = Box::new(MailboxSoA::new(7, 3, 1.0));
    m.set_populated(populated);
    for r in 0..populated {
        m.set_meta(r, MetaWord(meta_of(r)));
        m.set_edge(r, CausalEdge64(edge_of(r)));
    }
    m
}

fn world() -> World {
    World::named(&FIELDS)
}

/// The values of one property over one selection, in visit order.
fn collect(
    w: &World,
    binding: &lance_graph_dt64::LaneBinding,
    source: &MailboxLanes<'_, Owner>,
    chain: &[Recipe],
    property: usize,
    sel: u16,
) -> Result<Vec<(u32, Cell64)>, Refused> {
    let mut out = Vec::new();
    visit_selected(
        &w.store,
        &w.view,
        binding,
        source,
        source.rows(),
        chain,
        property,
        sel,
        |p, c| out.push((p, c)),
    )?;
    Ok(out)
}

/// 10.2 item 1. FAILS IF: the value is not the oracle's, the lane is not the
/// mailbox's own array, or the descriptor grew past two words.
#[test]
fn one_identity_one_property_is_read_from_the_owner() {
    let mailbox = owner(N);
    let mut source = MailboxLanes::new(&*mailbox);
    let sel = source.issue_list(vec![37]);
    let w = world();
    let binding = w.bind_source(&source, source.space()).unwrap();

    let (lane, positions) = binding
        .selected(&w.store, &w.view, source.frames(), &source, META, sel)
        .unwrap();
    let LaneRef::U32(values) = lane else {
        panic!("meta is a u32 lane");
    };
    // The owner's own allocation: same address as its `meta` column.
    assert!(core::ptr::eq(values.as_ptr(), mailbox.meta_raw().as_ptr()));
    assert_eq!(values.len(), N);
    assert_eq!(positions, &[37]);

    assert_eq!(
        collect(&w, &binding, &source, &ENTITY, usize::from(META), sel),
        Ok(vec![(37, Cell64::U32(meta_of(37)))])
    );
    // The virtual row: a recipe and a row number, no value.
    assert!(core::mem::size_of::<RowObject>() <= 16);
    assert_eq!(source.borrows(), [2, 0], "only the meta column was lent");
}

/// 10.2 item 2. FAILS IF: the 64 values do not arrive in (key, row) order,
/// the positions are a copy, more than 64 values are read, or a column that
/// was not demanded is lent.
#[test]
fn sixty_four_identities_in_source_issued_order_read_one_property() {
    let mailbox = owner(N);
    let mut source = MailboxLanes::new(&*mailbox);
    let sel = source.issue_order_by_meta().expect("the keys fit i32");
    let w = world();
    let binding = w.bind_source(&source, source.space()).unwrap();

    let mut want: Vec<u32> = (0..N as u32).collect();
    want.sort_by_key(|&r| (meta_of(r as usize), r));
    assert_ne!(
        want,
        (0..N as u32).collect::<Vec<_>>(),
        "the order must not be row order"
    );

    let (_, positions) = binding
        .selected(&w.store, &w.view, source.frames(), &source, META, sel)
        .unwrap();
    assert!(core::ptr::eq(
        positions.as_ptr(),
        source.raw_positions(sel).as_ptr()
    ));
    let before = source.borrows();

    let got = collect(&w, &binding, &source, &ENTITY, usize::from(META), sel).unwrap();
    assert_eq!(got.len(), N);
    let want_cells: Vec<(u32, Cell64)> = want
        .iter()
        .map(|&r| (r, Cell64::U32(meta_of(r as usize))))
        .collect();
    assert_eq!(got, want_cells);
    let after = source.borrows();
    assert_eq!(after[0] - before[0], 1, "one borrow of meta for 64 reads");
    assert_eq!(after[1], 0, "edge was never lent");
}

/// 10.2 item 3. FAILS IF: a second property, or a property made of two
/// regions, is not read at the same source-issued positions.
#[test]
fn two_properties_and_one_property_spanning_two_columns() {
    let mailbox = owner(N);
    let mut source = MailboxLanes::new(&*mailbox);
    let sel = source.issue_order_by_meta().unwrap();
    let w = world();
    let binding = w.bind_source(&source, source.space()).unwrap();

    let metas = collect(&w, &binding, &source, &ENTITY, usize::from(META), sel).unwrap();
    let edges = collect(&w, &binding, &source, &ENTITY, usize::from(EDGE), sel).unwrap();
    assert_eq!(source.borrows(), [1, 1]);
    // The combined property: (meta, edge) of the same identity.
    for ((pm, m), (pe, e)) in metas.iter().zip(&edges) {
        assert_eq!(pm, pe, "both columns are visited at the same position");
        let r = *pm as usize;
        assert_eq!((*m, *e), (Cell64::U32(meta_of(r)), Cell64::U64(edge_of(r))));
    }
    assert_eq!(metas.len(), N);
}

/// 10.2 item 4. FAILS IF: a duplicate is folded away or the list reordered.
#[test]
fn a_bag_keeps_its_multiplicity_and_order() {
    let mailbox = owner(N);
    let mut source = MailboxLanes::new(&*mailbox);
    let bag = vec![5, 5, 60, 5, 0];
    let sel = source.issue_list(bag.clone());
    let w = world();
    let binding = w.bind_source(&source, source.space()).unwrap();
    let got = collect(&w, &binding, &source, &ENTITY, usize::from(EDGE), sel).unwrap();
    let want: Vec<(u32, Cell64)> = bag
        .iter()
        .map(|&r| (r, Cell64::U64(edge_of(r as usize))))
        .collect();
    assert_eq!(got, want);
}

/// 10.2 item 5. FAILS IF: a chain that normalises to the entity view reads
/// differently, an Identity chain reads anything, or the recipe is ignored.
#[test]
fn the_recipe_is_composed_before_any_byte_is_demanded() {
    let mailbox = owner(N);
    let mut source = MailboxLanes::new(&*mailbox);
    let sel = source.issue_list((0..N as u32).rev().collect());
    let w = world();
    let binding = w.bind_source(&source, source.space()).unwrap();

    let direct = collect(&w, &binding, &source, &ENTITY, 0, sel).unwrap();
    let long = [Recipe::TRANSPOSE, Recipe::ROTATE90, Recipe::ROTATE270];
    assert_eq!(Recipe::normalize(&long), Recipe::TRANSPOSE);
    assert_eq!(
        collect(&w, &binding, &source, &long, 0, sel),
        Ok(direct.clone())
    );

    let lent = source.borrows();
    let undone = [Recipe::ROTATE90, Recipe::ROTATE270];
    assert_eq!(Recipe::normalize(&undone), Recipe::IDENTITY);
    assert_eq!(
        collect(&w, &binding, &source, &undone, 0, sel),
        Err(Refused::PropertyCrossesFields {
            recipe: Recipe::IDENTITY
        })
    );
    assert_eq!(source.borrows(), lent, "nothing was demanded");

    // The reversed property axis names the other column.
    let flipped = collect(&w, &binding, &source, &[Recipe::ANTI_TRANSPOSE], 0, sel).unwrap();
    assert_eq!(flipped[0], (63, Cell64::U64(edge_of(63))));
    assert_ne!(flipped, direct);
}

/// 10.2 item 6. Each refusal, and what is NOT fabricated.
#[test]
fn stale_or_foreign_addresses_are_refused() {
    let w = world();

    // Field mapping changed under the same frame.
    let mailbox = owner(N);
    let mut source = MailboxLanes::new(&*mailbox);
    let sel = source.issue_list(vec![1, 2, 3]);
    let binding = w.bind_source(&source, source.space()).unwrap();
    assert!(collect(&w, &binding, &source, &ENTITY, 0, sel).is_ok());
    source.swap_columns();
    let got = collect(&w, &binding, &source, &ENTITY, 0, sel);
    assert!(
        matches!(got, Err(Refused::Lane(LaneRefusal::MappingChanged { .. }))),
        "{got:?}"
    );
    source.swap_columns();

    // A selection in another row order.
    let foreign = CoordSpace {
        population: 99,
        coordinates: 1,
    };
    let other = source.issue_list_in(vec![1, 2, 3], foreign);
    let binding = w.bind_source(&source, source.space()).unwrap();
    let got = collect(&w, &binding, &source, &ENTITY, 0, other);
    assert!(
        matches!(got, Err(Refused::Lane(LaneRefusal::SelectionSpace { .. }))),
        "{got:?}"
    );
    assert!(matches!(
        w.bind_source(&source, foreign),
        Err(LaneRefusal::CoordinateSpace { .. })
    ));

    // A property the class does not have.
    assert_eq!(
        collect(&w, &binding, &source, &ENTITY, FIELDS.len(), sel),
        Err(Refused::OutsideTheView)
    );
    drop(source);

    // The owner advances: a binding sealed before is refused afterwards.
    let mut mailbox = mailbox;
    let old_binding = {
        let source = MailboxLanes::new(&*mailbox);
        w.bind_source(&source, source.space()).unwrap()
    };
    mailbox.tick();
    let mut source = MailboxLanes::new(&*mailbox);
    let sel = source.issue_list(vec![1, 2, 3]);
    let got = collect(&w, &old_binding, &source, &ENTITY, 0, sel);
    assert!(
        matches!(got, Err(Refused::Lane(LaneRefusal::Frame(_)))),
        "{got:?}"
    );
    let fresh = w.bind_source(&source, source.space()).unwrap();
    assert!(collect(&w, &fresh, &source, &ENTITY, 0, sel).is_ok());
}

/// 10.2 item 6, absent rows. FAILS IF: the zero padding of a capacity row is
/// handed out as a value.
#[test]
fn a_position_past_the_populated_rows_is_refused_not_zero_filled() {
    let mailbox = owner(40);
    let mut source = MailboxLanes::new(&*mailbox);
    assert_eq!(source.rows(), 40);
    assert_eq!(
        mailbox.meta_raw().len(),
        N,
        "the owner's array has capacity rows"
    );
    let sel = source.issue_list(vec![3, 39, 40, 4]);
    let w = world();
    let binding = w.bind_source(&source, source.space()).unwrap();
    let mut seen = Vec::new();
    let got = visit_selected(
        &w.store,
        &w.view,
        &binding,
        &source,
        source.rows(),
        &ENTITY,
        0,
        sel,
        |p, c| seen.push((p, c)),
    );
    assert_eq!(
        got,
        Err(Refused::PositionOutsideLane {
            at: 2,
            position: 40
        })
    );
    assert_eq!(seen.len(), 2, "the two valid positions before it were read");
}

/// 10.2 item 7, RECORDED LIMITATION (#2058). The owner's bytes change with no
/// cycle advance: the revisions a socket derives do not move. FAILS IF: that
/// changes (then the limitation moved and the plan must be corrected).
#[test]
fn a_write_without_a_cycle_advance_leaves_the_derived_frame_unchanged() {
    let mut mailbox = owner(N);
    let before = revisions_of(&*mailbox);
    mailbox.set_meta(11, MetaWord(424_242));
    assert_eq!(revisions_of(&*mailbox), before);

    let mut source = MailboxLanes::new(&*mailbox);
    let sel = source.issue_list(vec![11]);
    let w = world();
    let binding = w.bind_source(&source, source.space()).unwrap();
    assert_eq!(
        collect(&w, &binding, &source, &ENTITY, 0, sel),
        Ok(vec![(11, Cell64::U32(424_242))])
    );
    assert_ne!(meta_of(11), 424_242);
    // What does move the derived frame: the owner's own cycle.
    mailbox.tick();
    assert_ne!(revisions_of(&*mailbox), before);
}

/// 10.2 item 8, RECORDED GAP. The existing fold over a source-issued
/// selection admits `i32` lanes only; the mailbox lends none. FAILS IF: the
/// fold starts accepting the lane (then the gap closed).
#[test]
fn the_existing_selection_fold_does_not_take_a_mailbox_column() {
    let mailbox = owner(N);
    let mut source = MailboxLanes::new(&*mailbox);
    let sel = source.issue_order_by_meta().unwrap();
    let w = world();
    let binding = w.bind_source(&source, source.space()).unwrap();
    for field in [META, EDGE] {
        let got = fold_bound_selection(
            &binding,
            &w.store,
            &w.view,
            source.frames(),
            &source,
            field,
            sel,
            BoundFold::SumI32,
        );
        assert!(
            matches!(got, Err(BoundSelRefusal::Fold(ExecError::LaneKind { .. }))),
            "field {field}: {got:?}"
        );
    }
}
