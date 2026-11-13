//! WHEN on a real `MailboxSoA<64>` (D-VAS-C3 to C6): a guard of the routed,
//! admitted method, evaluated over the owner's own columns.
//!
//! Oracle: a scalar evaluator over the owner's public arrays, written here
//! from the definition of `SetExpr`. It shares no code with Quack, mask-risc
//! or the socket.

use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Cmp, Logic, Res, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::Scope;
use lance_graph_dt64::{
    view_class, BindStore, ClassSchema, Extent, Guard, LaneRefusal, MethodBinding, MethodId,
    Refusal, RouteAt, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mailbox_lanes::when::{when, WhenRefusal, WhenRequest};
use lance_graph_mailbox_lanes::{Column, MailboxLanes};
use lance_graph_mask_risc::ExecError;

const N: usize = 64;
type Owner = MailboxSoA<N>;
const POPULATED: usize = 40;

const CLASS: u32 = 0x0A1A_0003;
const POSITION: u16 = 11;
const UNROUTED: u16 = 12;
const READER: Scope = Scope { id: 3 };

/// Field positions of the class basis. Deliberately not the column order.
const TEMPORAL: u16 = 0;
const META: u16 = 1;
const EDGES: u16 = 2;
const ACTIVE: u16 = 3;
const FIELDS: usize = 4;
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
    // Values past the populated prefix that WOULD satisfy most guards: a
    // guard that read them would report rows the mailbox does not have.
    for r in POPULATED..N {
        m.meta[r] = MetaWord(2);
        m.last_active_cycle[r] = 0;
    }
    m
}

struct View(Vec<FieldRef>);

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == view_class(CLASS) {
            &self.0
        } else {
            &[]
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

/// `meta == 2 AND last_active_cycle != 1`.
fn fire_guard() -> SetExpr {
    SetExpr::And(vec![SetExpr::eq_u32(META, 2), SetExpr::ne_u32(ACTIVE, 1)])
}

fn fire(when: SetExpr) -> MethodBinding {
    MethodBinding {
        id: FIRE,
        sig_rev: SigRev(1),
        name: "fire".into(),
        on: WideFieldMask::from_positions(&[META as u8]),
        when: Some(Guard::new(when, Logic::TwoValued)),
        reads: WideFieldMask::from_positions(&[META as u8, ACTIVE as u8]),
        effects: vec![],
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

struct World {
    store: BindStore,
    view: View,
    route: RouteAt,
}

fn schema() -> ClassSchema {
    ClassSchema {
        schema_rev: SchemaRev(1),
        field_count: FIELDS,
        extent: Extent::DR256_SQ_X64,
    }
}

fn world() -> World {
    let field = |c: Column| FieldRef::new(c.iri(), c.iri());
    let view = View(vec![
        field(Column::Temporal),
        field(Column::Meta),
        field(Column::Edges),
        field(Column::LastActiveCycle),
    ]);
    let mut store = BindStore::new();
    store
        .register(&view, CLASS, schema(), vec![fire(fire_guard())])
        .expect("the guarded class registers");
    let version = store.route(POSITION, CLASS).unwrap();
    World {
        store,
        view,
        route: RouteAt {
            position: POSITION,
            version,
        },
    }
}

fn request(w: &World) -> WhenRequest {
    WhenRequest {
        route: w.route,
        rev: SchemaRev(1),
        scope: READER,
    }
}

/// The scalar evaluator: `e` at row `r` of the owner's own arrays.
fn holds(m: &Owner, e: &SetExpr, r: usize) -> bool {
    match e {
        SetExpr::All => true,
        SetExpr::Empty => false,
        SetExpr::Res(Res::U32 { lane, eq, v }) => {
            let x = match *lane {
                META => m.meta[r].0,
                ACTIVE => m.last_active_cycle[r],
                other => panic!("the oracle knows no u32 field {other}"),
            };
            (x == *v) == *eq
        }
        SetExpr::Not(c) => !holds(m, c, r),
        SetExpr::And(cs) => cs.iter().all(|c| holds(m, c, r)),
        SetExpr::Or(cs) => cs.iter().any(|c| holds(m, c, r)),
        other => panic!("the oracle does not evaluate {other:?}"),
    }
}

struct Gen(u64);
impl Gen {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn leaf(&mut self) -> SetExpr {
        let lane = if self.next().is_multiple_of(2) {
            META
        } else {
            ACTIVE
        };
        SetExpr::Res(Res::U32 {
            lane,
            eq: self.next().is_multiple_of(2),
            v: (self.next() % 4) as u32,
        })
    }
    fn expr(&mut self, depth: u32) -> SetExpr {
        if depth == 0 || self.next().is_multiple_of(4) {
            return match self.next() % 16 {
                0 => SetExpr::All,
                1 => SetExpr::Empty,
                _ => self.leaf(),
            };
        }
        match self.next() % 3 {
            0 => SetExpr::negated(self.expr(depth - 1)),
            k => {
                let n = 1 + self.next() % 3;
                let cs = (0..n).map(|_| self.expr(depth - 1)).collect();
                if k == 1 {
                    SetExpr::And(cs)
                } else {
                    SetExpr::Or(cs)
                }
            }
        }
    }
}

/// Positive. FAILS IF: a guard holds at a row where the owner's own columns
/// say it does not (or the reverse), a row past the populated prefix is
/// answered, or the guard read a field it does not name.
#[test]
fn a_guard_holds_exactly_where_the_owners_columns_say() {
    let m = owner();
    let w = world();
    let lanes = MailboxLanes::new(m.published(), READER);
    let mut g = Gen(0xC4);
    let (mut some, mut none, mut all, mut one_field) = (0, 0, 0, 0);
    for _ in 0..400 {
        let e = g.expr(3);
        let guard = Guard::new(e.clone(), Logic::TwoValued);
        let fired = when(&lanes, &w.store, &w.view, request(&w), &guard).unwrap();
        assert_eq!(fired.rows(), POPULATED);
        let mut count = 0;
        for r in 0..POPULATED {
            let want = holds(&m, &e, r);
            count += u64::from(want);
            assert_eq!(
                fired.holds(&w.store, &w.view, &lanes, r as u32),
                Ok(want),
                "row {r} of {e:?}"
            );
        }
        assert_eq!(fired.count(), count, "{e:?}");
        assert_eq!(
            fired.holds(&w.store, &w.view, &lanes, POPULATED as u32),
            Err(WhenRefusal::RowOutsideLane {
                row: POPULATED as u32,
                rows: POPULATED
            })
        );
        // Only fields of the two u32 columns are ever bound.
        assert!(fired
            .fields()
            .iter()
            .all(|f| [META as u8, ACTIVE as u8].contains(f)));
        one_field += usize::from(fired.fields().len() == 1);
        match count as usize {
            0 => none += 1,
            POPULATED => all += 1,
            _ => some += 1,
        }
    }
    // Anti-vacuity: mixed answers dominate, and both constants occurred.
    assert!(some > 200, "{some}");
    assert!(none > 10 && all > 10, "{none} {all}");
    assert!(one_field > 20, "{one_field}");
}

/// Positive, end of the admission chain. FAILS IF: the guard evaluated is
/// not the routed, admitted method's own.
#[test]
fn the_admitted_methods_guard_is_what_fires() {
    let m = owner();
    let w = world();
    let lanes = MailboxLanes::new(m.published(), READER);
    let class = w
        .store
        .resolve_at(&w.view, w.route.version, POSITION, SchemaRev(1))
        .unwrap();
    let on: Vec<MethodId> = class.methods_on(META as u8).map(|x| x.id).collect();
    assert_eq!(on, [FIRE]);
    let admitted = w
        .store
        .admit_routed(w.route, &w.store.method_ref(CLASS, FIRE).unwrap())
        .unwrap();
    let guard = admitted.when.as_ref().expect("fire is guarded");
    let fired = when(&lanes, &w.store, &w.view, request(&w), guard).unwrap();
    let want: Vec<u32> = (0..POPULATED as u32)
        .filter(|&r| meta_of(r as usize) == 2 && active_of(r as usize) != 1)
        .collect();
    assert!(
        want.len() > 3 && want.len() < POPULATED / 2,
        "{}",
        want.len()
    );
    let got: Vec<u32> = (0..POPULATED as u32)
        .filter(|&r| fired.holds(&w.store, &w.view, &lanes, r).unwrap())
        .collect();
    assert_eq!(got, want);
    assert_eq!(fired.fields(), &[META as u8, ACTIVE as u8]);
}

/// C3 at the owner: a guarded method whose guard names a field outside the
/// class basis never becomes behaviour of the mailbox's class.
#[test]
fn a_guard_outside_the_basis_never_registers() {
    let w = world();
    let mut store = BindStore::new();
    assert_eq!(
        store.register(
            &w.view,
            CLASS,
            schema(),
            vec![fire(SetExpr::eq_u32(FIELDS as u16, 1))]
        ),
        Err(Refusal::GuardLaneOutOfBasis {
            class: CLASS,
            method: FIRE,
            lane: FIELDS as u16
        })
    );
}

/// Negative: what the socket cannot evaluate is refused by name, and no
/// answer is produced.
#[test]
fn what_the_mailbox_cannot_evaluate_is_refused() {
    let m = owner();
    let w = world();
    let lanes = MailboxLanes::new(m.published(), READER);
    let run = |expr: SetExpr, logic: Logic| {
        when(
            &lanes,
            &w.store,
            &w.view,
            request(&w),
            &Guard::new(expr, logic),
        )
        .err()
    };
    // C5: nothing issues a mask plane for this population.
    assert_eq!(
        run(
            SetExpr::And(vec![SetExpr::eq_u32(META, 2), SetExpr::Sym(5)]),
            Logic::TwoValued
        ),
        Some(WhenRefusal::PlaneUnissued { plane: 5 })
    );
    // C6: the mailbox states no validity for its columns.
    assert_eq!(
        run(SetExpr::eq_u32(META, 2), Logic::Kleene),
        Some(WhenRefusal::ValidityUnstated)
    );
    // A signed comparison over an unsigned column, and a 32-bit comparison
    // over a 64-bit column: the executor's own kind refusal.
    for expr in [SetExpr::i32(META, Cmp::Lt, 3), SetExpr::eq_u32(EDGES, 1)] {
        assert!(
            matches!(
                run(expr.clone(), Logic::TwoValued),
                Some(WhenRefusal::Exec(ExecError::LaneKind { .. }))
            ),
            "{expr:?}"
        );
    }
    // A lane that is no field of the class, and one no field position names.
    assert_eq!(
        run(SetExpr::eq_u32(FIELDS as u16, 1), Logic::TwoValued),
        Some(WhenRefusal::Lane(LaneRefusal::FieldNotInBasis {
            class: CLASS,
            field: FIELDS as u8
        }))
    );
    assert_eq!(
        run(SetExpr::eq_u32(300, 1), Logic::TwoValued),
        Some(WhenRefusal::FieldOutOfRange { lane: 300 })
    );
    // Silence twin: the same call with a guard it can evaluate answers.
    assert!(run(SetExpr::eq_u32(META, 2), Logic::TwoValued).is_none());
    let _ = TEMPORAL;
}

/// Negative: an answer belongs to the route and the publication it was
/// evaluated under.
#[test]
fn a_fired_mask_does_not_outlive_its_route_or_publication() {
    let m = owner();
    let mut w = world();
    let lanes = MailboxLanes::new(m.published(), READER);
    let guard = Guard::new(fire_guard(), Logic::TwoValued);

    // An unrouted position has no behaviour to evaluate a guard for.
    let unrouted = WhenRequest {
        route: RouteAt {
            position: UNROUTED,
            version: w.route.version,
        },
        ..request(&w)
    };
    assert!(matches!(
        when(&lanes, &w.store, &w.view, unrouted, &guard),
        Err(WhenRefusal::Lane(_))
    ));

    let fired = when(&lanes, &w.store, &w.view, request(&w), &guard).unwrap();
    assert!(fired.holds(&w.store, &w.view, &lanes, 0).is_ok());

    // Another publication of the same mailbox is another source.
    let later = MailboxLanes::new(m.published(), READER);
    assert!(matches!(
        fired.holds(&w.store, &w.view, &later, 0),
        Err(WhenRefusal::Lane(_))
    ));

    // The position is routed to other behaviour.
    w.store
        .register(&w.view, CLASS + 1, schema(), vec![])
        .unwrap();
    w.store.route(POSITION, CLASS + 1).unwrap();
    assert!(matches!(
        fired.holds(&w.store, &w.view, &lanes, 0),
        Err(WhenRefusal::Lane(_))
    ));
}
