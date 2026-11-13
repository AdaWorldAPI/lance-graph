//! E1 (roadmap stage E, full behaviour) on a real `MailboxSoA<64>`:
//! a real write -> ON -> route-bound admission -> WHEN over the owner's
//! columns -> operands over `reads` -> R2IL under the guard's gate ->
//! write-back through `write_row` -> the next ON, until nothing changes.
//!
//! Oracle: `Model`, a scalar interpreter of the same two methods over plain
//! vectors. It shares no code with DT64, Quack, mask-risc, R2IL or the
//! socket; it is the methods written out by hand.
//!
//! Not in this chain: `NodeGuid` (B1 is blocked: the mailbox stores no key),
//! mask planes (C5) and three-valued guards on the mailbox (C6).

use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome};
use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Focus, Scope};
use lance_graph_dt64::{
    view_class, AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, Guard, MethodBinding,
    MethodId, Refusal, RouteAt, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mailbox_lanes::act::{
    act, apply, cascade, ActContext, ActRefusal, Applied, ApplyRefusal,
};
use lance_graph_mailbox_lanes::{Column, MailboxLanes};
use lance_graph_r2il::binding::{ExecPermit, R2ilLens};
use lance_graph_r2il::polyfill::{AliasLanguage, POLYFILL, POLYFILL_LENS};

const N: usize = 64;
type Owner = MailboxSoA<N>;
const POPULATED: usize = 40;

const CLASS: u32 = 0x0A1A_0003;
const OTHER: u32 = 0x0A1A_0004;
const POSITION: u16 = 11;
const READER: Scope = Scope { id: 3 };
const FOCUS: Focus = Focus { revision: 1 };
const REV: SchemaRev = SchemaRev(1);

/// Field positions of the class basis. `MASK` names no mailbox column: it is
/// where a method keeps an intermediate mask.
const META: u8 = 0;
const ACTIVE: u8 = 1;
const TEMPORAL: u8 = 2;
const MASK: u8 = 3;
const FIELDS: usize = 4;

/// `tally`: ON meta; WHEN `meta == 2 AND active != 1`;
/// DO `temporal := COUNT(rows the guard keeps where active == 0)`.
const TALLY: MethodId = MethodId(1);
/// `mark`: ON temporal; WHEN `meta == 1`;
/// DO `meta := COUNT(rows the guard keeps where active == 2)`.
const MARK: MethodId = MethodId(2);
/// Writes a stamp column the owner does not let a method write.
const STAMP: MethodId = MethodId(3);

/// Polyfill positions (`lance_graph_r2il::polyfill::POLYFILL`).
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

impl View {
    fn new() -> View {
        let field = |iri: &str| FieldRef::new(iri, iri);
        View {
            class: vec![
                field(Column::Meta.iri()),
                field(Column::LastActiveCycle.iri()),
                field(Column::Temporal.iri()),
                field("urn:lab:mask"),
            ],
            lens: POLYFILL
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    FieldRef::new(format!("urn:polyfill:{i}"), w.name_in(AliasLanguage::R2il))
                })
                .collect(),
        }
    }
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

/// `target := COUNT(rows where field == v)`, through the scratch mask field.
fn count_where(field: u8, v: i64, target: u8) -> Vec<Effect> {
    vec![
        word(MASK, W_EQUAL, vec![Arg::Field(field), Arg::Const(v)]),
        word(target, W_POP_COUNT, vec![Arg::Field(MASK)]),
    ]
}

fn method(id: MethodId, on: u8, when: SetExpr, effects: Vec<Effect>) -> MethodBinding {
    MethodBinding {
        id,
        sig_rev: SigRev(1),
        name: format!("m{}", id.0),
        on: WideFieldMask::from_positions(&[on]),
        when: Some(Guard::new(when, Logic::TwoValued)),
        reads: WideFieldMask::from_positions(&[ACTIVE]),
        effects,
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

fn tally_guard() -> SetExpr {
    SetExpr::And(vec![
        SetExpr::eq_u32(u16::from(META), 2),
        SetExpr::ne_u32(u16::from(ACTIVE), 1),
    ])
}

struct World {
    store: BindStore,
    view: View,
    route: RouteAt,
    permit: ExecPermit,
}

fn world() -> World {
    let view = View::new();
    let mut store = BindStore::new();
    store
        .declare_algebra(
            POLYFILL_LENS,
            REV,
            Extent::DR256_SQ_X64,
            POLYFILL.iter().map(|w| w.arity).collect(),
        )
        .unwrap();
    let schema = ClassSchema {
        schema_rev: REV,
        field_count: FIELDS,
        extent: Extent::DR256_SQ_X64,
    };
    store
        .register(
            &view,
            CLASS,
            schema,
            vec![
                method(TALLY, META, tally_guard(), count_where(ACTIVE, 0, TEMPORAL)),
                method(
                    MARK,
                    TEMPORAL,
                    SetExpr::eq_u32(u16::from(META), 1),
                    count_where(ACTIVE, 2, META),
                ),
                method(
                    STAMP,
                    MASK,
                    SetExpr::eq_u32(u16::from(META), 0),
                    count_where(ACTIVE, 0, ACTIVE),
                ),
            ],
        )
        .expect("the class registers");
    store
        .register(
            &view,
            OTHER,
            schema,
            vec![method(
                TALLY,
                META,
                SetExpr::eq_u32(u16::from(META), 0),
                count_where(ACTIVE, 0, TEMPORAL),
            )],
        )
        .expect("the second class registers");
    let version = store.route(POSITION, CLASS).unwrap();
    let permit = ExecPermit {
        methods: [TALLY, MARK, STAMP]
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
        scope: READER,
        focus: FOCUS,
        lens: R2ilLens {
            lens: POLYFILL_LENS,
            schema_rev: REV,
        },
        permit: &w.permit,
    }
}

/// The scalar interpreter: the two methods by hand, over plain vectors.
#[derive(Clone, PartialEq, Debug)]
struct Model {
    meta: Vec<u32>,
    active: Vec<u32>,
    temporal: Vec<u64>,
}

/// What one model round did: per fired method, the rows it kept, the value
/// it computed and the cells it changed.
#[derive(Debug, PartialEq, Clone)]
struct ModelFire {
    method: MethodId,
    rows: Vec<u32>,
    value: u64,
    changed: u32,
}

impl Model {
    fn of(m: &Owner) -> Model {
        Model {
            meta: (0..POPULATED).map(|r| m.meta[r].0).collect(),
            active: (0..POPULATED).map(|r| m.last_active_cycle[r]).collect(),
            temporal: (0..POPULATED).map(|r| m.temporal[r]).collect(),
        }
    }

    /// One round for a change of `changed` columns. Every method reads the
    /// state as it was at the start of the round; writes land afterwards.
    fn round(&mut self, changed: &[Column]) -> (Vec<ModelFire>, Vec<Column>) {
        let before = self.clone();
        let mut fires = Vec::new();
        let mut next = Vec::new();
        if changed.contains(&Column::Meta) {
            let rows: Vec<u32> = (0..POPULATED as u32)
                .filter(|&r| before.meta[r as usize] == 2 && before.active[r as usize] != 1)
                .collect();
            let value = rows
                .iter()
                .filter(|&&r| before.active[r as usize] == 0)
                .count() as u64;
            let mut n = 0;
            for &r in &rows {
                if self.temporal[r as usize] != value {
                    self.temporal[r as usize] = value;
                    n += 1;
                }
            }
            if n > 0 {
                next.push(Column::Temporal);
            }
            fires.push(ModelFire {
                method: TALLY,
                rows,
                value,
                changed: n,
            });
        }
        if changed.contains(&Column::Temporal) {
            let rows: Vec<u32> = (0..POPULATED as u32)
                .filter(|&r| before.meta[r as usize] == 1)
                .collect();
            let value = rows
                .iter()
                .filter(|&&r| before.active[r as usize] == 2)
                .count() as u64;
            let mut n = 0;
            for &r in &rows {
                if u64::from(self.meta[r as usize]) != value {
                    self.meta[r as usize] = value as u32;
                    n += 1;
                }
            }
            if n > 0 {
                next.push(Column::Meta);
            }
            fires.push(ModelFire {
                method: MARK,
                rows,
                value,
                changed: n,
            });
        }
        (fires, next)
    }
}

/// The real change that starts the chain: the owner's own cycle-aware write.
fn first_write(m: &mut Owner) {
    let cycle = m.current_cycle;
    let cell = WriteCell {
        meta: Some(MetaWord(2)),
        ..WriteCell::default()
    };
    assert_eq!(m.write_row(5, cycle, &cell), WriteOutcome::Accepted);
}

/// E1, positive. FAILS IF: any round fires another method, keeps another
/// row, computes another value or changes another cell than the scalar
/// interpreter, or the owner's final columns differ from the model's.
#[test]
fn the_whole_chain_equals_the_scalar_interpreter() {
    let mut m = owner();
    let w = world();
    first_write(&mut m);
    let mut model = Model::of(&m);

    let run = cascade(&mut m, &w.store, &w.view, &ctx(&w), &[Column::Meta], 8).unwrap();

    let mut changed = vec![Column::Meta];
    let mut model_rounds = Vec::new();
    while !changed.is_empty() {
        let (fires, next) = model.round(&changed);
        model_rounds.push(fires);
        changed = next;
    }
    // The chain is not trivial: three rounds, both methods, real writes.
    assert_eq!(model_rounds.len(), 3, "{model_rounds:?}");
    assert!(run.settled);
    assert_eq!(run.rounds.len(), model_rounds.len());

    let (mut writes, mut programs, mut guard_rows) = (0, 0, 0);
    for (round, want) in run.rounds.iter().zip(&model_rounds) {
        let got: Vec<ModelFire> = round
            .fired
            .iter()
            .map(|f| ModelFire {
                method: f.method,
                rows: f.rows.clone(),
                value: f.value,
                changed: f.applied.changed,
            })
            .collect();
        assert_eq!(&got, want);
        for f in &round.fired {
            // Ledger 1, guard: one program over the populated rows.
            assert_eq!(f.guard_rows, POPULATED as u64);
            guard_rows += f.guard_rows;
            // Ledger 2, operands: the guard's two fields (one for `mark`),
            // then the method's one declared read. Nothing else was lent.
            let guard_fields = if f.method == TALLY { 2 } else { 1 };
            assert_eq!(f.lanes_lent, guard_fields + 1, "{:?}", f.method);
            // Ledger 3, effects: two steps, one finalized program.
            assert_eq!(f.steps, 2);
            assert_eq!(f.programs, 1);
            programs += f.programs;
            writes += f.applied.changed;
            // Every changed cell carries the owner's stamp of this cycle.
            assert_eq!(f.applied.refused, None);
        }
    }
    assert!(writes > 4, "{writes}");
    assert_eq!(programs, 3);
    assert_eq!(guard_rows, 3 * POPULATED as u64);

    // The owner's columns are the model's, cell for cell.
    assert_eq!(Model::of(&m), model);
    // Rows past the populated prefix were never written.
    for r in POPULATED..N {
        assert_eq!((m.meta[r].0, m.temporal[r]), (0, 0));
        assert_eq!(m.last_write_cycle[r], u32::MAX);
    }
    // A row a method wrote carries the cycle the write was made for.
    let written = model_rounds[0][0].rows[0] as usize;
    assert_eq!(m.last_write_cycle[written], m.current_cycle);
}

/// D4, negative. FAILS IF: the cascade runs past its bound, or reports a
/// chain it cut short as settled.
#[test]
fn the_cascade_stops_at_its_bound_and_says_so() {
    let mut m = owner();
    let w = world();
    first_write(&mut m);
    let run = cascade(&mut m, &w.store, &w.view, &ctx(&w), &[Column::Meta], 2).unwrap();
    assert_eq!(run.rounds.len(), 2);
    assert!(!run.settled);
    // A change of a column no method is ON settles at once, firing nothing.
    let mut m = owner();
    let run = cascade(
        &mut m,
        &w.store,
        &w.view,
        &ctx(&w),
        &[Column::LastActiveCycle],
        8,
    )
    .unwrap();
    assert!(run.settled);
    assert_eq!(run.rounds.len(), 1);
    assert!(run.rounds[0].fired.is_empty());
}

/// Admission, negative. FAILS IF: a method the position does not route to
/// executes, or any column is lent before a method was admitted.
#[test]
fn nothing_is_read_before_a_method_is_admitted() {
    let m = owner();
    let w = world();
    let lanes = MailboxLanes::new(m.published(), READER);
    let foreign = w.store.method_ref(OTHER, TALLY).unwrap();
    assert!(matches!(
        act(&lanes, &w.store, &w.view, &ctx(&w), &foreign),
        Err(ActRefusal::Admission(Refusal::NotRoutedClass { .. }))
    ));
    assert_eq!(lanes.lends(), 0);
    // Admitted and executed, but not permitted to execute.
    let none = ExecPermit::default();
    let unpermitted = ActContext {
        permit: &none,
        ..ctx(&w)
    };
    let own = w.store.method_ref(CLASS, TALLY).unwrap();
    assert!(matches!(
        act(&lanes, &w.store, &w.view, &unpermitted, &own),
        Err(ActRefusal::Exec(_))
    ));
    // Silence twin: the routed, permitted method runs and lends lanes.
    assert!(act(&lanes, &w.store, &w.view, &ctx(&w), &own).is_ok());
    assert!(lanes.lends() > 0);
}

/// D3, negative. FAILS IF: a write lands without the owner's cycle agreeing,
/// on another mailbox, or on a column a method may not write.
#[test]
fn a_write_lands_only_under_the_owners_rules() {
    let mut m = owner();
    let w = world();
    first_write(&mut m);
    let own = w.store.method_ref(CLASS, TALLY).unwrap();
    let acted = {
        let lanes = MailboxLanes::new(m.published(), READER);
        act(&lanes, &w.store, &w.view, &ctx(&w), &own).unwrap()
    };
    assert_eq!(acted.intents.len(), 1);
    let intent = &acted.intents[0];
    assert_eq!(intent.column(), Column::Temporal);

    // Another mailbox.
    let mut other: Box<Owner> = Box::new(MailboxSoA::new(8, 3, 1.0));
    other.set_populated(POPULATED);
    assert_eq!(
        apply(&mut other, intent),
        Err(ApplyRefusal::ForeignOwner {
            intent: 7,
            owner: 8
        })
    );
    assert!(other.temporal.iter().all(|&t| t == 0));

    // The owner moved to the next cycle: the intent is stale, nothing lands.
    let before = Model::of(&m);
    m.tick();
    let applied: Applied = apply(&mut m, intent).unwrap();
    assert_eq!(applied.changed, 0);
    assert_eq!(applied.refused, Some(WriteOutcome::Stale));
    assert_eq!(Model::of(&m), before);
    assert!(m.stale_write_count() > 0);

    // Silence twin: a fresh intent for the current cycle lands, stamped.
    let fresh = {
        let lanes = MailboxLanes::new(m.published(), READER);
        act(&lanes, &w.store, &w.view, &ctx(&w), &own).unwrap()
    };
    let applied = apply(&mut m, &fresh.intents[0]).unwrap();
    assert!(applied.changed > 0);
    assert_eq!(applied.refused, None);
    let row = fresh.rows[0] as usize;
    assert_eq!(m.last_write_cycle[row], m.current_cycle);
    // Applying it again changes nothing and writes nothing.
    let again = apply(&mut m, &fresh.intents[0]).unwrap();
    assert_eq!((again.changed, again.unchanged), (0, applied.changed));

    // A method whose effect targets a stamp column is refused whole.
    let stamp = w.store.method_ref(CLASS, STAMP).unwrap();
    let lanes = MailboxLanes::new(m.published(), READER);
    assert!(matches!(
        act(&lanes, &w.store, &w.view, &ctx(&w), &stamp),
        Err(ActRefusal::NotWritable {
            field: ACTIVE,
            column: Column::LastActiveCycle
        })
    ));
}
