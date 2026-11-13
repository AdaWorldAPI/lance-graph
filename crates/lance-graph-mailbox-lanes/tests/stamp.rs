//! The owner stamps `last_write_cycle` on every accepted write. A method ON
//! that column is therefore triggered by any write that moves a row's stamp,
//! whichever column the write was for.
//!
//! Oracle: the two methods written out by hand over the owner's columns as
//! they were before the cascade.

use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome};
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
use lance_graph_mailbox_lanes::act::{cascade, ActContext};
use lance_graph_mailbox_lanes::Column;
use lance_graph_r2il::binding::{ExecPermit, R2ilLens};
use lance_graph_r2il::polyfill::{AliasLanguage, POLYFILL, POLYFILL_LENS};

const N: usize = 64;
type Owner = MailboxSoA<N>;
const POPULATED: usize = 40;
const CLASS: u32 = 0x0A1A_0005;
const POSITION: u16 = 12;
const REV: SchemaRev = SchemaRev(1);

const META: u8 = 0;
const ACTIVE: u8 = 1;
const TEMPORAL: u8 = 2;
const MASK: u8 = 3;
const WRITTEN: u8 = 4;
const FIELDS: usize = 5;

/// ON meta; WHEN `meta == 2`; DO `temporal := COUNT(kept rows, active == 0)`.
const TALLY: MethodId = MethodId(1);
/// ON last_write_cycle; WHEN `meta == 1`;
/// DO `meta := COUNT(kept rows, active == 2)`.
const ON_STAMP: MethodId = MethodId(2);

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
            field(Column::Temporal.iri()),
            field("urn:lab:mask"),
            field(Column::LastWriteCycle.iri()),
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
                method(TALLY, META, 2, 0, TEMPORAL),
                method(ON_STAMP, WRITTEN, 1, 2, META),
            ],
        )
        .expect("the class registers");
    let version = store.route(POSITION, CLASS).unwrap();
    let permit = ExecPermit {
        methods: [TALLY, ON_STAMP]
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

fn rows_with_meta(v: u32) -> Vec<u32> {
    (0..POPULATED as u32)
        .filter(|&r| meta_of(r as usize) == v)
        .collect()
}

fn count_active(rows: &[u32], v: u32) -> u64 {
    rows.iter().filter(|&&r| active_of(r as usize) == v).count() as u64
}

/// Positive. FAILS IF: a write that moves a row's write-cycle stamp does not
/// trigger the method ON that column, or the cascade reports settled before
/// that method ran.
#[test]
fn a_moved_write_stamp_triggers_the_method_on_it() {
    let mut m = owner();
    let w = world();
    let tally_rows = rows_with_meta(2);
    let tally_value = count_active(&tally_rows, 0);
    let stamp_rows = rows_with_meta(1);
    let stamp_value = count_active(&stamp_rows, 2);
    // The fixture can express the defect: `tally` changes cells, and no
    // method is ON the column it writes.
    assert!(tally_rows.len() > 1 && tally_value > 0);
    assert!(stamp_rows.len() > 1 && stamp_value != 1);
    for &r in &tally_rows {
        assert_eq!(m.last_write_cycle[r as usize], u32::MAX);
    }

    let run = cascade(&mut m, &w.store, &w.view, &ctx(&w), &[Column::Meta], 8).unwrap();

    assert!(run.settled);
    assert_eq!(run.rounds[0].fired.len(), 1);
    let tally = &run.rounds[0].fired[0];
    assert_eq!((tally.method, &tally.rows), (TALLY, &tally_rows));
    assert_eq!(tally.applied.changed, tally_rows.len() as u32);
    assert_eq!(tally.applied.stamped, tally_rows.len() as u32);

    assert_eq!(run.rounds[1].fired.len(), 1);
    let on_stamp = &run.rounds[1].fired[0];
    assert_eq!((on_stamp.method, &on_stamp.rows), (ON_STAMP, &stamp_rows));
    assert_eq!(on_stamp.value, stamp_value);
    assert_eq!(on_stamp.applied.changed, stamp_rows.len() as u32);
    for &r in &stamp_rows {
        assert_eq!(u64::from(m.meta[r as usize].0), stamp_value);
    }
}

/// Silence. FAILS IF: a write to a row already stamped with this cycle is
/// reported as a change of the stamp column.
#[test]
fn a_write_that_leaves_the_stamp_triggers_nothing_on_it() {
    let mut m = owner();
    let w = world();
    let tally_rows = rows_with_meta(2);
    // The owner writes every one of those rows in this cycle first, with the
    // value they hold: the stamps move now, not in the cascade.
    let cycle = m.current_cycle;
    for &r in &tally_rows {
        let cell = WriteCell {
            meta: Some(MetaWord(2)),
            ..WriteCell::default()
        };
        assert_eq!(
            m.write_row(r as usize, cycle, &cell),
            WriteOutcome::Accepted
        );
    }

    let run = cascade(&mut m, &w.store, &w.view, &ctx(&w), &[Column::Meta], 8).unwrap();

    assert!(run.settled);
    let tally = &run.rounds[0].fired[0];
    assert_eq!(tally.applied.changed, tally_rows.len() as u32);
    assert_eq!(tally.applied.stamped, 0);
    // `temporal` changed, and no method is ON it.
    assert_eq!(run.rounds.len(), 2);
    assert!(run.rounds[1].fired.is_empty());
    for r in rows_with_meta(1) {
        assert_eq!(m.meta[r as usize].0, 1);
    }
}
