//! D-QUACK-FIELD-OPERAND-PROBE: how a DT64 field becomes the `Col` a program
//! reads.
//!
//! Three index spaces are kept apart on purpose in this fixture:
//!
//! - the field's position in the `ClassView` basis (`amount` is 1,
//!   `quantity` is 5, because this view exchanges positions 2 and 5);
//! - the source's own lane id (`LaneAddr::lane`), renumbered per source;
//! - the operand index in `Planes::lanes`, which is the field's rank among
//!   the bound fields (`operand`).
//!
//! Only fields 1 and 5 are bound. The query is
//! `SUM(quantity) WHERE amount > T`.

use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::FrameRefusal;
use lance_graph_dt64::{
    LaneBinding, LaneRefusal, MethodBinding, MethodId, Refusal, SigRev, ValueDemand,
};
use lance_graph_mask_risc::{ExecError, Foreign, Out, Program, Scratch, Value};
use lance_graph_quack::bind::{table, Binder, BoundField, FieldKind, TableId};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Mask, Query};
use quack_boundary_lab::{
    admitted, bound_positions, operand, oracle, unbound, Prepared, Refused, RowOrder, Source, View,
    World, AMOUNT, DISC_VALID, FIELDS, ID, QTY, READER, ROWS,
};

const T: i32 = 25;
/// `quantity`'s position in the view this probe registers.
const QTY_AT: u8 = DISC_VALID;

/// The registered view: `quantity` at 5, `discountValid` at 2.
fn view() -> View {
    View::swapped(QTY, DISC_VALID)
}

/// Lane ids that agree with no field position and no operand index.
const LANE_IDS: [u16; FIELDS] = [40, 31, 22, 13, 4, 55, 16];
const OTHER_LANE_IDS: [u16; FIELDS] = [6, 5, 4, 3, 2, 1, 0];

fn sparse(w: &World, s: &Source) -> LaneBinding {
    w.bind_fields(s, &[AMOUNT, QTY_AT], READER, s.space())
        .expect("fields 1 and 5 bind")
}

/// `SUM(sum) WHERE filter > T`, over operand indices.
fn program(filter: Col, sum: Col) -> Program {
    lower(&Query {
        filter: Filter::cmp(filter, Cmp::GtI32(T)),
        agg: Agg::SumI32(sum),
    })
    .unwrap()
}

/// The program lowered through the binding: operands from `operand`.
fn bound_program(b: &LaneBinding) -> Program {
    program(
        operand(b, AMOUNT).expect("amount is bound"),
        operand(b, QTY_AT).expect("quantity is bound"),
    )
}

fn scratch(p: &Program) -> Scratch<'static> {
    Scratch::for_program(p, ROWS).unwrap()
}

fn run(p: &Prepared, w: &World, b: &LaneBinding, s: &Source) -> Result<Value, Refused> {
    admitted(p, w, &w.view, b, s, &mut scratch(&p.program), Out::None)
}

/// Falsifiers 1 and 2. FAILS IF: a sparse, non-contiguous binding puts a lane at
/// another operand index than `operand` reports, the sum differs from the
/// oracle, or the three index spaces coincide (the probe would then be the
/// old equal case again).
#[test]
fn a_sparse_binding_reads_the_bound_fields_at_their_ranks() {
    let w = World::with(view(), Vec::new());
    let s = Source::new(RowOrder::ById).with_lane_ids(LANE_IDS);
    let b = sparse(&w, &s);
    assert_eq!(bound_positions(&b), vec![AMOUNT, QTY_AT]);
    assert_eq!(operand(&b, AMOUNT), Some(Col(0)));
    assert_eq!(operand(&b, QTY_AT), Some(Col(1)));
    // Position, lane id and operand index are three different numbers.
    assert_eq!(b.addr(AMOUNT).unwrap().lane, 31);
    assert_eq!(b.addr(QTY_AT).unwrap().lane, 22);

    let want = oracle::sum_qty_amount_gt(&s, T);
    assert!(want > 1_000);
    let p = Prepared::over(bound_program(&b), &s, &b);
    assert_eq!(run(&p, &w, &b, &s), Ok(Value::SumI64(want)));

    // The old equal case: every field bound, standard view, identity ids.
    let w0 = World::new();
    let s0 = Source::new(RowOrder::ById);
    let b0 = w0.bind(&s0).unwrap();
    assert_eq!(operand(&b0, AMOUNT), Some(Col(u16::from(AMOUNT))));
    let p0 = Prepared::over(
        program(operand(&b0, AMOUNT).unwrap(), operand(&b0, QTY).unwrap()),
        &s0,
        &b0,
    );
    assert_eq!(run(&p0, &w0, &b0, &s0), Ok(Value::SumI64(want)));
}

/// Falsifier 2, the negative half. FAILS IF: a program whose operands were numbered by
/// field POSITION is run over a sparse binding, loudly or silently.
#[test]
fn operands_numbered_by_position_are_not_run_over_a_sparse_binding() {
    let w = World::with(view(), Vec::new());
    let s = Source::new(RowOrder::ById).with_lane_ids(LANE_IDS);
    let b = sparse(&w, &s);
    let lanes: Vec<_> = b
        .borrow_all(&w.store, &w.view, s.frame_store(), &s)
        .unwrap()
        .map(|l| l.unwrap().1)
        .collect();

    // Loud: positions 1 and 5 over two operands. The executor refuses.
    let by_position = program(Col(u16::from(AMOUNT)), Col(u16::from(QTY_AT)));
    assert_eq!(
        unbound(
            &by_position,
            &lanes,
            &[],
            &Foreign::NONE,
            &mut scratch(&by_position),
            Out::None
        ),
        Err(ExecError::LaneOutOfRange(u16::from(QTY_AT)))
    );

    // Silent: a program meant for fields 0 and 1 (`SUM(amount) WHERE id > T`)
    // fits two operands and reads `amount` and `quantity` instead.
    let for_other_fields = program(Col(u16::from(ID)), Col(u16::from(AMOUNT)));
    let got = unbound(
        &for_other_fields,
        &lanes,
        &[],
        &Foreign::NONE,
        &mut scratch(&for_other_fields),
        Out::None,
    );
    assert_eq!(
        got,
        Ok(Value::SumI64(oracle::sum_qty_amount_gt(&s, T))),
        "it answers another question"
    );
    let meant: i64 = (0..ROWS)
        .map(|r| s.row(r))
        .filter(|r| r.id > T)
        .map(|r| i64::from(r.amount))
        .sum();
    assert_ne!(got, Ok(Value::SumI64(meant)));

    // Admitted: the program says which fields its operands were numbered
    // over, and they are not the bound ones.
    let p = Prepared {
        program: for_other_fields,
        frame: s.frame_store().seal(),
        fields: vec![ID, AMOUNT],
    };
    assert_eq!(
        run(&p, &w, &b, &s),
        Err(Refused::Operands {
            program: vec![ID, AMOUNT],
            binding: vec![AMOUNT, QTY_AT],
        })
    );
}

/// Falsifier 3. FAILS IF: the answer depends on which lane ids a source uses.
#[test]
fn another_lane_numbering_gives_the_same_answer() {
    let w = World::with(view(), Vec::new());
    let a = Source::new(RowOrder::ById).with_lane_ids(LANE_IDS);
    let c = Source::new(RowOrder::ById).with_lane_ids(OTHER_LANE_IDS);
    let (ba, bc) = (sparse(&w, &a), sparse(&w, &c));
    assert_ne!(ba.addr(AMOUNT).unwrap().lane, bc.addr(AMOUNT).unwrap().lane);
    // The program text is the same for both: operands are ranks.
    assert_eq!(bound_program(&ba), bound_program(&bc));
    let want = oracle::sum_qty_amount_gt(&a, T);
    let pa = Prepared::over(bound_program(&ba), &a, &ba);
    let pc = Prepared::over(bound_program(&bc), &c, &bc);
    assert_eq!(run(&pa, &w, &ba, &a), Ok(Value::SumI64(want)));
    assert_eq!(run(&pc, &w, &bc, &c), Ok(Value::SumI64(want)));
    // One source's binding does not read the other.
    assert_eq!(
        run(&pa, &w, &ba, &c),
        Err(Refused::Binding(LaneRefusal::Frame(
            FrameRefusal::ForeignFrame
        )))
    );
}

/// Falsifier 4. FAILS IF: a changed field basis or a changed source mapping reaches
/// execution.
#[test]
fn a_changed_basis_or_mapping_is_refused_before_execution() {
    let w = World::with(view(), Vec::new());
    let mut s = Source::new(RowOrder::ById).with_lane_ids(LANE_IDS);
    let b = sparse(&w, &s);
    let p = Prepared::over(bound_program(&b), &s, &b);
    assert!(run(&p, &w, &b, &s).is_ok());

    // The basis moved under the binding: position 5 names another field.
    let standard = View::standard();
    assert_eq!(
        admitted(
            &p,
            &w,
            &standard,
            &b,
            &s,
            &mut scratch(&p.program),
            Out::None
        ),
        Err(Refused::Binding(LaneRefusal::MappingChanged {
            field: QTY_AT
        }))
    );
    // A store that never saw this basis refuses to bind it at all.
    let other_world = World::new();
    assert!(matches!(
        other_world.store.resolve(
            &view(),
            quack_boundary_lab::CLASS,
            lance_graph_dt64::SchemaRev(1)
        ),
        Err(Refusal::PredicateBasisChanged { .. })
    ));

    // The source renumbered its lanes without a new frame.
    s.renumber(OTHER_LANE_IDS);
    assert_eq!(
        run(&p, &w, &b, &s),
        Err(Refused::Binding(LaneRefusal::MappingChanged {
            field: AMOUNT
        }))
    );
}

/// Falsifier 5. FAILS IF: an unbound field yields an operand or a lane.
#[test]
fn an_unbound_field_has_no_operand() {
    let w = World::with(view(), Vec::new());
    let s = Source::new(RowOrder::ById).with_lane_ids(LANE_IDS);
    let b = sparse(&w, &s);
    for field in [ID, QTY, 3, 4, 6, 200] {
        assert_eq!(operand(&b, field), None, "field {field}");
        assert!(matches!(
            b.lane(&w.store, &w.view, s.frame_store(), &s, field),
            Err(LaneRefusal::NotBound { .. })
        ));
    }
    // A field outside the basis cannot be bound.
    assert!(matches!(
        w.bind_fields(&s, &[AMOUNT, 9], READER, s.space()),
        Err(LaneRefusal::FieldNotInBasis { field: 9, .. })
    ));
}

/// Quack's own name-to-`Col` surface, answered from a binding.
struct BoundTable<'a> {
    names: [(&'static str, u8); 2],
    binding: &'a LaneBinding,
}

impl Binder for BoundTable<'_> {
    fn table(&self, name: &str) -> Option<TableId> {
        (name == "orders").then_some(TableId(1))
    }
    fn live(&self, _table: TableId) -> Mask {
        Mask(0)
    }
    fn field(&self, _table: TableId, name: &str) -> Option<BoundField> {
        let field = self.names.iter().find(|(n, _)| *n == name)?.1;
        Some(BoundField {
            col: operand(self.binding, field)?,
            kind: FieldKind::I32,
            validity: None,
        })
    }
    fn code(&self, _table: TableId, _field: Col, _literal: &str) -> Option<u32> {
        None
    }
}

/// FAILS IF: Quack's `Binder` cannot carry the rank translation, or a
/// `Draft`-bound query turns out to be admissible although it reads a
/// liveness plane.
#[test]
fn the_existing_binder_carries_the_translation_and_a_draft_reads_a_plane() {
    let w = World::with(view(), Vec::new());
    let s = Source::new(RowOrder::ById).with_lane_ids(LANE_IDS);
    let b = sparse(&w, &s);
    let binder = BoundTable {
        names: [("amount", AMOUNT), ("quantity", QTY_AT)],
        binding: &b,
    };
    let orders = binder.table("orders").unwrap();
    assert_eq!(binder.field(orders, "quantity").unwrap().col, Col(1));
    assert!(binder.field(orders, "discount").is_none());

    let q = table("orders")
        .where_eq("quantity", 7)
        .count()
        .bind(&binder)
        .unwrap();
    // `Draft::bind` always conjoins the table's liveness plane.
    assert_eq!(
        q.filter,
        Filter::and([Filter::plane(Mask(0)), Filter::cmp(Col(1), Cmp::EqI32(7))])
    );
    let p = Prepared::over(lower(&q).unwrap(), &s, &b);
    assert_eq!(
        run(&p, &w, &b, &s),
        Err(Refused::Blocked(
            "mask plane: the lane socket issues lanes, not planes"
        ))
    );
    // The same comparison without the plane is admitted and right.
    let direct = lower(&Query {
        filter: Filter::cmp(Col(1), Cmp::EqI32(7)),
        agg: Agg::Count,
    })
    .unwrap();
    let want = (0..ROWS).filter(|&r| s.row(r).qty == 7).count();
    assert!(want > 1_000);
    let p = Prepared::over(direct, &s, &b);
    assert_eq!(run(&p, &w, &b, &s), Ok(Value::Count(want)));
}

fn method(reads: &[u8]) -> MethodBinding {
    MethodBinding {
        id: MethodId(1),
        sig_rev: SigRev(1),
        name: "recompute".to_string(),
        on: WideFieldMask::from_positions(&[AMOUNT]),
        when: None,
        reads: WideFieldMask::from_positions(reads),
        effects: Vec::new(),
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

/// Control: two stores, one class basis, different behaviour.
///
/// FAILS IF: a method reference of one store is admitted by the other. The
/// lane half records what `LaneBinding::check` does today and takes no
/// position on whether it should.
#[test]
fn two_stores_one_basis_other_behaviour() {
    let a = World::with(view(), vec![method(&[AMOUNT])]);
    let c = World::with(view(), vec![method(&[AMOUNT, QTY_AT])]);
    let s = Source::new(RowOrder::ById).with_lane_ids(LANE_IDS);

    // Behaviour: each store admits its own reference and refuses the other's.
    let class = quack_boundary_lab::CLASS;
    let ra = a.store.method_ref(class, MethodId(1)).unwrap();
    let rc = c.store.method_ref(class, MethodId(1)).unwrap();
    assert!(a.store.admit(&ra).is_ok());
    assert!(c.store.admit(&rc).is_ok());
    assert_eq!(
        c.store.admit(&ra),
        Err(Refusal::BindingMismatch {
            class,
            method: MethodId(1)
        })
    );
    assert_eq!(
        a.store.admit(&rc),
        Err(Refusal::BindingMismatch {
            class,
            method: MethodId(1)
        })
    );
    // Same behaviour, another store: refused by issuer.
    let twin = a.store.clone();
    assert_eq!(twin.admit(&ra), Err(Refusal::ForeignStore { class }));

    // Lanes: a binding made through store `a` passes `check` against store
    // `c`. Both route the position to the class and hold the same basis
    // witness; the binding does not record which store made it.
    let b = sparse(&a, &s);
    assert_eq!(b.check(&c.store, &c.view, s.frame_store(), &s), Ok(()));
    let p = Prepared::over(bound_program(&b), &s, &b);
    assert_eq!(
        run(&p, &c, &b, &s),
        Ok(Value::SumI64(oracle::sum_qty_amount_gt(&s, T)))
    );
}
