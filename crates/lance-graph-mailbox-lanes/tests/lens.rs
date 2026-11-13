//! D-VAS-C1 on a real `MailboxSoA<64>`: a behaviour lens (right half
//! `0xF###`) supplies behaviour for a position only by being what the
//! position routes to. The route log is the one versioned association; no
//! second table links a data class to a lens.
//!
//! Oracle: the owner's public arrays and the generators below.

use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::class_view::{is_behaviour_lens, ClassId, ClassView, WideFieldMask};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::Scope;
use lance_graph_dt64::{
    view_class, BindStore, ClassSchema, Extent, Guard, LaneRequest, MethodBinding, MethodId,
    Refusal, RouteAt, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mailbox_lanes::when::{when, WhenRequest};
use lance_graph_mailbox_lanes::{Column, MailboxLanes};
use lance_graph_mask_risc::LaneRef;

const N: usize = 64;
type Owner = MailboxSoA<N>;
const POPULATED: usize = 40;

/// The data class and two behaviour lenses that share its left half.
const DATA: u32 = 0x0A1A_0003;
const LENS: u32 = 0x0A1A_F001;
const OTHER_LENS: u32 = 0x0A1A_F002;
const UNKNOWN_LENS: u32 = 0x0A1A_F003;
const POSITION: u16 = 11;
const READER: Scope = Scope { id: 3 };
const FIRE: MethodId = MethodId(1);

fn meta_of(r: usize) -> u32 {
    (r as u32 * 7 + 3) % 4
}
fn active_of(r: usize) -> u32 {
    (r as u32 * 5 + 1) % 3
}

fn owner() -> Box<Owner> {
    let mut m: Box<Owner> = Box::new(MailboxSoA::new(7, 3, 1.0));
    m.set_populated(POPULATED);
    for r in 0..POPULATED {
        m.set_meta(r, MetaWord(meta_of(r)));
        m.last_active_cycle[r] = active_of(r);
    }
    m
}

/// The data class reads `[temporal, meta, edges, last_active_cycle]`; both
/// lenses read `[meta, last_active_cycle]`. Field 0 therefore means a
/// different column under the data class than under a lens.
struct View {
    data: Vec<FieldRef>,
    lens: Vec<FieldRef>,
}

impl View {
    fn new() -> View {
        let field = |c: Column| FieldRef::new(c.iri(), c.iri());
        View {
            data: vec![
                field(Column::Temporal),
                field(Column::Meta),
                field(Column::Edges),
                field(Column::LastActiveCycle),
            ],
            lens: vec![field(Column::Meta), field(Column::LastActiveCycle)],
        }
    }
}

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == view_class(DATA) {
            &self.data
        } else {
            &[]
        }
    }
    fn lens_fields(&self, lens: u32) -> Option<&[FieldRef]> {
        (lens == LENS || lens == OTHER_LENS).then_some(&self.lens[..])
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

fn method(name: &str, when: Option<SetExpr>, reads: &[u8]) -> MethodBinding {
    MethodBinding {
        id: FIRE,
        sig_rev: SigRev(1),
        name: name.into(),
        on: WideFieldMask::from_positions(&[0]),
        when: when.map(|e| Guard::new(e, Logic::TwoValued)),
        reads: WideFieldMask::from_positions(reads),
        effects: vec![],
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

fn schema(field_count: usize) -> ClassSchema {
    ClassSchema {
        schema_rev: SchemaRev(1),
        field_count,
        extent: Extent::DR256_SQ_X64,
    }
}

/// Lens field 0 is `meta`, lens field 1 is `last_active_cycle`.
fn lens_guard() -> SetExpr {
    SetExpr::And(vec![SetExpr::eq_u32(0, 2), SetExpr::ne_u32(1, 1)])
}

fn store(view: &View) -> BindStore {
    let mut s = BindStore::new();
    s.register(view, DATA, schema(4), vec![method("data", None, &[1])])
        .expect("the data class registers");
    s.register(
        view,
        LENS,
        schema(2),
        vec![method("lens", Some(lens_guard()), &[0, 1])],
    )
    .expect("the lens registers");
    s.register(
        view,
        OTHER_LENS,
        schema(2),
        vec![method("other", Some(SetExpr::eq_u32(0, 0)), &[0, 1])],
    )
    .expect("the other lens registers");
    s
}

/// Positive. FAILS IF: a position routed to a lens does not get the lens's
/// own method, basis and guard, over the owner's own columns.
#[test]
fn a_routed_lens_supplies_its_own_behaviour_over_the_owners_columns() {
    assert!(is_behaviour_lens(LENS) && is_behaviour_lens(OTHER_LENS));
    assert!(!is_behaviour_lens(DATA));
    let m = owner();
    let view = View::new();
    let mut s = store(&view);
    let version = s.route(POSITION, LENS).unwrap();
    let route = RouteAt {
        position: POSITION,
        version,
    };
    let lanes = MailboxLanes::new(m.published(), READER);

    let class = s
        .resolve_at(&view, version, POSITION, SchemaRev(1))
        .unwrap();
    assert_eq!(class.class, LENS);
    assert_eq!(
        class.fields.len(),
        2,
        "the lens's basis, not the data class's"
    );
    let admitted = s
        .admit_routed(route, &s.method_ref(LENS, FIRE).unwrap())
        .unwrap();
    assert_eq!(admitted.name, "lens");

    // Field 0 under the lens is the owner's `meta` column, in place.
    let frame = lanes.seal();
    let binding = s
        .bind_lanes(
            &view,
            lanes.frame_store(),
            &lanes,
            LaneRequest {
                route,
                rev: SchemaRev(1),
                fields: &admitted.reads,
                frame: &frame,
                scope: READER,
                selection: lanes.space(),
            },
        )
        .unwrap();
    let LaneRef::U32(meta) = binding
        .lane(&s, &view, lanes.frame_store(), &lanes, 0)
        .unwrap()
    else {
        panic!("a u32 lane")
    };
    assert_eq!(meta.as_ptr().cast::<u8>(), m.meta.as_ptr().cast::<u8>());

    // The lens's guard, read in the lens's field positions.
    let fired = when(
        &lanes,
        &s,
        &view,
        WhenRequest {
            route,
            rev: SchemaRev(1),
            scope: READER,
        },
        admitted.when.as_ref().unwrap(),
    )
    .unwrap();
    let want: Vec<u32> = (0..POPULATED as u32)
        .filter(|&r| meta_of(r as usize) == 2 && active_of(r as usize) != 1)
        .collect();
    assert!(want.len() > 3 && want.len() < POPULATED / 2);
    let got: Vec<u32> = (0..POPULATED as u32)
        .filter(|&r| fired.holds(&s, &view, &lanes, r).unwrap())
        .collect();
    assert_eq!(got, want);
}

/// Negative. FAILS IF: a lens the position does not route to supplies a
/// method for it: a second lens with the same basis and the same `MethodId`,
/// or the data class the lens shares its left half with.
#[test]
fn an_unrouted_lens_with_a_compatible_basis_supplies_nothing() {
    let view = View::new();
    let mut s = store(&view);
    let version = s.route(POSITION, LENS).unwrap();
    let route = RouteAt {
        position: POSITION,
        version,
    };
    for (class, name) in [(OTHER_LENS, "other"), (DATA, "data")] {
        let r = s.method_ref(class, FIRE).unwrap();
        // The store knows the method; the position does not have it.
        assert_eq!(s.admit(&r).unwrap().name, name);
        assert_eq!(
            s.admit_routed(route, &r).err(),
            Some(Refusal::NotRoutedClass {
                position: POSITION,
                captured: Some(LENS),
                now: Some(LENS),
                class
            })
        );
    }
}

/// Negative, versioned. FAILS IF: behaviour admitted under one route version
/// survives the position being routed to another lens, or a route captured
/// before the lens was routed admits the lens.
#[test]
fn the_association_is_the_route_and_moves_with_it() {
    let view = View::new();
    let mut s = store(&view);
    let v_data = s.route(POSITION, DATA).unwrap();
    let v_lens = s.route(POSITION, LENS).unwrap();
    let lens_ref = s.method_ref(LENS, FIRE).unwrap();
    assert!(s
        .admit_routed(
            RouteAt {
                position: POSITION,
                version: v_lens
            },
            &lens_ref
        )
        .is_ok());
    // Captured while the position was the data class.
    assert_eq!(
        s.admit_routed(
            RouteAt {
                position: POSITION,
                version: v_data
            },
            &lens_ref
        )
        .err(),
        Some(Refusal::NotRoutedClass {
            position: POSITION,
            captured: Some(DATA),
            now: Some(LENS),
            class: LENS
        })
    );
    // Routed on to the other lens: the first lens's method is gone for it.
    s.route(POSITION, OTHER_LENS).unwrap();
    assert_eq!(
        s.admit_routed(
            RouteAt {
                position: POSITION,
                version: v_lens
            },
            &lens_ref
        )
        .err(),
        Some(Refusal::NotRoutedClass {
            position: POSITION,
            captured: Some(LENS),
            now: Some(OTHER_LENS),
            class: LENS
        })
    );
    // The replay answer is unchanged: at `v_lens` the position was the lens.
    assert_eq!(s.routed_at(v_lens, POSITION), Ok(Some(LENS)));
}

/// Negative. FAILS IF: a lens the view does not know by its full classid is
/// bound through the basis of its left half.
#[test]
fn a_lens_is_never_read_through_its_left_half() {
    let view = View::new();
    let mut s = BindStore::new();
    assert_eq!(view_class(UNKNOWN_LENS), view_class(DATA));
    assert_eq!(
        s.register(&view, UNKNOWN_LENS, schema(4), vec![]),
        Err(Refusal::UnknownLensBasis(UNKNOWN_LENS))
    );
}
