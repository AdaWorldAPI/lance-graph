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
    view_class, AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, Guard, MethodBinding,
    MethodId, RouteAt, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mailbox_lanes::act::{act, ActContext, ActRefusal};
use lance_graph_mailbox_lanes::{Column, MailboxLanes};
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
            ],
        )
        .expect("the class registers");
    let version = store.route(POSITION, CLASS).unwrap();
    let permit = ExecPermit {
        methods: [TO_SIGMA, TO_SCRATCH]
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
