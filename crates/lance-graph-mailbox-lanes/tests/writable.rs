//! Which class fields a method may target on the mailbox.
//!
//! A field naming a mailbox column the socket knows but cannot write must
//! refuse the method. Only a field that names no mailbox column at all is a
//! scratch field whose scalar stays out of the owner.

use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Focus, Scope};
use lance_graph_dt64::{
    view_class, AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, Guard, LaneRequest,
    MethodBinding, MethodId, Refusal, RouteAt, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mailbox_lanes::act::{act, ActContext, ActRefusal};
use lance_graph_mailbox_lanes::{Column, Keep, MailboxLanes};
use lance_graph_mask_risc::LaneRef;
use lance_graph_r2il::binding::{ExecPermit, R2ilLens};
use lance_graph_r2il::polyfill::{AliasLanguage, POLYFILL, POLYFILL_LENS};

const N: usize = 64;
type Owner = MailboxSoA<N>;
const POPULATED: usize = 40;
const CLASS: u32 = 0x0A1A_0006;
const POSITION: u16 = 13;
const REV: SchemaRev = SchemaRev(1);

const META: u8 = 0;
const ACTIVE: u8 = 1;
/// Names `sigma`: a mailbox column (`u8`) the socket does not write.
const SIGMA: u8 = 2;
const MASK: u8 = 3;
/// Names no mailbox column.
const SCRATCH: u8 = 4;
const FIELDS: usize = 5;

const TO_SIGMA: MethodId = MethodId(1);
const TO_SCRATCH: MethodId = MethodId(2);
/// Targets `meta`: a write.
const TO_META: MethodId = MethodId(3);
/// As `TO_SCRATCH`, declared over a bag (`Population::Bag`).
const OVER_BAG: MethodId = MethodId(4);

const W_EQUAL: u8 = 4;
const W_POP_COUNT: u8 = 10;

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

struct View {
    class: Vec<FieldRef>,
    lens: Vec<FieldRef>,
}

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == view_class(CLASS) {
            &self.class
        } else {
            &[]
        }
    }
    fn lens_fields(&self, lens: u32) -> Option<&[FieldRef]> {
        (lens == POLYFILL_LENS).then_some(&self.lens[..])
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

fn word(target: u8, word: u8, args: Vec<Arg>) -> Effect {
    Effect {
        target,
        algebra: AlgebraRef {
            lens: POLYFILL_LENS,
            word,
        },
        args,
    }
}

fn method(id: MethodId, on: u8, meta: u32, active: i64, target: u8) -> MethodBinding {
    MethodBinding {
        id,
        sig_rev: SigRev(1),
        name: format!("m{}", id.0),
        on: WideFieldMask::from_positions(&[on]),
        when: Some(Guard::new(
            SetExpr::eq_u32(u16::from(META), meta),
            Logic::TwoValued,
        )),
        reads: WideFieldMask::from_positions(&[ACTIVE]),
        effects: vec![
            word(MASK, W_EQUAL, vec![Arg::Field(ACTIVE), Arg::Const(active)]),
            word(target, W_POP_COUNT, vec![Arg::Field(MASK)]),
        ],
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

struct World {
    store: BindStore,
    view: View,
    route: RouteAt,
    permit: ExecPermit,
}

fn world() -> World {
    let field = |iri: &str| FieldRef::new(iri, iri);
    let view = View {
        class: vec![
            field(Column::Meta.iri()),
            field(Column::LastActiveCycle.iri()),
            field("urn:mailbox:sigma"),
            field("urn:lab:mask"),
            field("urn:lab:scratch"),
        ],
        lens: POLYFILL
            .iter()
            .enumerate()
            .map(|(i, w)| {
                FieldRef::new(format!("urn:polyfill:{i}"), w.name_in(AliasLanguage::R2il))
            })
            .collect(),
    };
    let mut store = BindStore::new();
    store
        .declare_algebra(
            POLYFILL_LENS,
            REV,
            Extent::DR256_SQ_X64,
            POLYFILL.iter().map(|w| w.arity).collect(),
        )
        .unwrap();
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: REV,
                field_count: FIELDS,
                extent: Extent::DR256_SQ_X64,
            },
            vec![
                method(TO_SIGMA, META, 2, 0, SIGMA),
                method(TO_SCRATCH, META, 2, 0, SCRATCH),
                method(TO_META, META, 2, 0, META),
                MethodBinding {
                    population: Population::Bag,
                    ..method(OVER_BAG, META, 2, 0, SCRATCH)
                },
            ],
        )
        .expect("the class registers");
    let version = store.route(POSITION, CLASS).unwrap();
    let permit = ExecPermit {
        methods: [TO_SIGMA, TO_SCRATCH, TO_META, OVER_BAG]
            .iter()
            .map(|&m| store.method_ref(CLASS, m).unwrap())
            .collect(),
    };
    World {
        store,
        view,
        route: RouteAt {
            position: POSITION,
            version,
        },
        permit,
    }
}

fn ctx(w: &World) -> ActContext<'_> {
    ActContext {
        route: w.route,
        rev: REV,
        scope: Scope { id: 3 },
        focus: Focus { revision: 1 },
        lens: R2ilLens {
            lens: POLYFILL_LENS,
            schema_rev: REV,
        },
        permit: &w.permit,
    }
}

/// FAILS IF: a method whose effect targets a mailbox column the socket
/// cannot write runs, reads a lane, or has its result filed as scratch.
#[test]
fn a_known_unwritable_column_is_not_a_scratch_field() {
    let m = owner();
    let w = world();
    let lanes = MailboxLanes::new(m.published(), Scope { id: 3 });
    let to_sigma = w.store.method_ref(CLASS, TO_SIGMA).unwrap();
    let got = act(&lanes, &w.store, &w.view, &ctx(&w), &to_sigma);
    assert_eq!(
        got,
        Err(ActRefusal::UnsupportedColumn {
            field: SIGMA,
            kind: "u8"
        })
    );
    assert_eq!(lanes.lends(), 0);

    // Silence twin: a field that names no mailbox column is scratch. The
    // method runs, asks for no write, and its scalar is reported.
    let to_scratch = w.store.method_ref(CLASS, TO_SCRATCH).unwrap();
    let acted = act(&lanes, &w.store, &w.view, &ctx(&w), &to_scratch).unwrap();
    assert!(acted.intents.is_empty());
    let want = (0..POPULATED)
        .filter(|&r| meta_of(r) == 2 && active_of(r) == 0)
        .count() as i64;
    assert!(want > 0);
    assert_eq!(acted.unwritten, vec![(SCRATCH, want)]);
}

/// The guard's rows and the fold, by hand: `meta == 2`, then how many of
/// those rows have `active == 0`.
fn oracle() -> (Vec<u32>, i64) {
    let rows: Vec<u32> = (0..POPULATED as u32)
        .filter(|&r| meta_of(r as usize) == 2)
        .collect();
    let count = rows.iter().filter(|&&r| active_of(r as usize) == 0).count() as i64;
    (rows, count)
}

/// A read-only projection, composed from what exists:
/// `ClassView IN published rows WHEN guard DO fold`, the scalar returned in
/// `Acted::unwritten`. FAILS IF: it returns another population or scalar
/// than the hand-written fold, asks the owner for a write, or changes it.
#[test]
fn a_projection_is_a_scalar_and_never_a_write() {
    let m = owner();
    let w = world();
    let before: Vec<(u32, u64)> = (0..POPULATED)
        .map(|r| (m.meta[r].0, m.temporal[r]))
        .collect();
    let lanes = MailboxLanes::new(m.published(), Scope { id: 3 });
    let (rows, count) = oracle();
    assert!(rows.len() > 4 && count > 0 && (count as usize) < rows.len());

    let to_scratch = w.store.method_ref(CLASS, TO_SCRATCH).unwrap();
    let got = act(&lanes, &w.store, &w.view, &ctx(&w), &to_scratch).unwrap();
    // No write is asked for: there is nothing to hand to `apply`.
    assert!(got.intents.is_empty());
    assert_eq!(got.rows, rows);
    assert_eq!(got.unwritten, vec![(SCRATCH, count)]);
    assert_eq!(got.receipt.output.values, vec![(SCRATCH, count)]);
    assert_eq!(got.receipt.method, to_scratch);
    assert_eq!(got.guard_rows, POPULATED as u64);
    // One lane for the guard's field, one for the declared read.
    assert_eq!(lanes.lends(), 2);

    // Silence twin: the same fold targeting `meta` IS a write request.
    let to_meta = w.store.method_ref(CLASS, TO_META).unwrap();
    let writes = act(&lanes, &w.store, &w.view, &ctx(&w), &to_meta).unwrap();
    assert_eq!(writes.intents.len(), 1);
    assert!(writes.unwritten.is_empty());
    drop(lanes);
    let after: Vec<(u32, u64)> = (0..POPULATED)
        .map(|r| (m.meta[r].0, m.temporal[r]))
        .collect();
    assert_eq!(after, before);
}

/// FAILS IF: the projection runs for a position that does not route to the
/// method's class, without an execution permit, or under a reader scope
/// other than the socket's. The scope check is consistency of two values
/// the caller asserts; no authority grants either.
#[test]
fn a_projection_is_refused_at_route_permit_and_scope() {
    let m = owner();
    let w = world();
    let lanes = MailboxLanes::new(m.published(), Scope { id: 3 });
    let to_scratch = w.store.method_ref(CLASS, TO_SCRATCH).unwrap();

    let elsewhere = ActContext {
        route: RouteAt {
            position: POSITION + 1,
            version: w.route.version,
        },
        ..ctx(&w)
    };
    assert!(matches!(
        act(&lanes, &w.store, &w.view, &elsewhere, &to_scratch),
        Err(ActRefusal::Admission(Refusal::NotRoutedClass { .. }))
    ));
    assert_eq!(lanes.lends(), 0);

    let none = ExecPermit::default();
    let unpermitted = ActContext {
        permit: &none,
        ..ctx(&w)
    };
    assert!(matches!(
        act(&lanes, &w.store, &w.view, &unpermitted, &to_scratch),
        Err(ActRefusal::Exec(_))
    ));

    let other_scope = ActContext {
        scope: Scope { id: 4 },
        ..ctx(&w)
    };
    assert!(matches!(
        act(&lanes, &w.store, &w.view, &other_scope, &to_scratch),
        Err(ActRefusal::When(_))
    ));
}

/// Multiplicity. The socket issues a bag in which every guard row occurs
/// twice; `act` gates its fold with one bit per row.
/// FAILS IF: a method declared over a bag runs on that Boolean gate, i.e.
/// membership is silently taken for multiplicity.
#[test]
fn a_bag_method_does_not_run_on_a_membership_gate() {
    let m = owner();
    let w = world();
    let mut lanes = MailboxLanes::new(m.published(), Scope { id: 3 });
    let (rows, count) = oracle();

    // The bag, as the socket issues it: the guard rows, then the same again.
    let once = lanes.issue_where(Column::Meta, Keep::Eq(2)).unwrap();
    let twice = lanes.issue_concat(once, once).unwrap();
    let frame = lanes.seal();
    let binding = w
        .store
        .bind_lanes(
            &w.view,
            lanes.frame_store(),
            &lanes,
            LaneRequest {
                route: w.route,
                rev: REV,
                fields: &WideFieldMask::from_positions(&[ACTIVE]),
                frame: &frame,
                scope: Scope { id: 3 },
                selection: lanes.space(),
            },
        )
        .unwrap();
    let (lane, positions) = binding
        .selected(
            &w.store,
            &w.view,
            lanes.frame_store(),
            &lanes,
            ACTIVE,
            twice,
        )
        .unwrap();
    let LaneRef::U32(active) = lane else {
        panic!("a u32 lane")
    };
    // The bag fold by hand: every occurrence counts.
    let over_bag = positions
        .iter()
        .filter(|&&p| active[p as usize] == 0)
        .count() as i64;
    assert_eq!(positions.len(), 2 * rows.len());
    assert_eq!(over_bag, 2 * count);
    assert_ne!(over_bag, count, "the two readings differ on this fixture");

    // A method declared over distinct rows computes the set reading.
    let distinct = w.store.method_ref(CLASS, TO_SCRATCH).unwrap();
    let got = act(&lanes, &w.store, &w.view, &ctx(&w), &distinct).unwrap();
    assert_eq!(got.unwritten, vec![(SCRATCH, count)]);
    // The same fold declared over a bag is refused, before anything is lent.
    let lent = lanes.lends();
    let bag = w.store.method_ref(CLASS, OVER_BAG).unwrap();
    assert_eq!(
        act(&lanes, &w.store, &w.view, &ctx(&w), &bag),
        Err(ActRefusal::BagPopulation { method: OVER_BAG })
    );
    assert_eq!(lanes.lends(), lent);
}
