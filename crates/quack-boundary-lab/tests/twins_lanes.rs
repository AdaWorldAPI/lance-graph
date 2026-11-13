//! D-QUACK-DERIVED-PLANE-PROBE, P1: the twins rule shape over bound lanes.
//!
//! Pass conditions: `.claude/plans/d-quack-derived-plane-probe.md` section 4,
//! committed before this file existed.
//!
//! Two paths decide the same candidates:
//!
//! - raw: `hidden_state_twins_probe::Bound::decide` over `Vec<u32>` lanes the
//!   caller copied out. The negative control, unchanged.
//! - bound: the same guards registered as methods of this lab's class, the
//!   lanes taken through one `LaneBinding` (`admitted`).
//!
//! `discountValid` stands for a producer-derived lane: the fixture's producer
//! derives it as `id % 5 != 0`.

use hidden_state_twins_probe::{digest, Family, Rule, TwinRefusal};
use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::FrameRefusal;
use lance_graph_dt64::{
    CoordSpace, LaneBinding, LaneRefusal, MethodBinding, MethodId, Refusal, SigRev, ValueDemand,
};
use lance_graph_mask_risc::{Out, Scratch};
use lance_graph_quack::{lower, Agg, Cmp, Filter, Query};
use quack_boundary_lab::{
    admitted, operand, Prepared, Refused, RowOrder, Source, View, World, CLASS, DISC_VALID, FIELDS,
    READER, REGION, ROWS,
};

/// The rules, over lanes `region` and `valid` of some row class.
fn family(class: u32, lanes: usize, region: u16, valid: u16, producer: u32) -> Family {
    Family {
        class,
        lanes,
        rules: vec![
            Rule {
                name: "low_region_with_discount",
                when: vec![(region, vec![0, 1, 2]), (valid, vec![1])],
            },
            Rule {
                name: "high_region",
                when: vec![(region, vec![5, 6, 7])],
            },
        ],
        producer,
    }
}

fn producer() -> u32 {
    digest(&["quack-boundary-lab", "validity", "id % 5 != 0"])
}

/// The rules over this lab's class: lanes are field positions.
fn lab_family(producer: u32) -> Family {
    family(
        CLASS,
        FIELDS,
        u16::from(REGION),
        u16::from(DISC_VALID),
        producer,
    )
}

/// The same rules over a two-lane raw row class (lane 0 region, lane 1 valid).
fn raw_family(producer: u32) -> Family {
    family(0x7A11_0001, 2, 0, 1, producer)
}

fn methods(f: &Family) -> Vec<MethodBinding> {
    let fields = WideFieldMask::from_positions(&[REGION, DISC_VALID]);
    (0..f.rules.len())
        .map(|i| MethodBinding {
            id: MethodId(i as u32 + 1),
            sig_rev: SigRev(1),
            name: f.rules[i].name.to_string(),
            on: fields.clone(),
            when: Some(f.guard(i, f.producer)),
            reads: fields.clone(),
            effects: Vec::new(),
            population: Population::Distinct,
            truth: Truth::TwoValued,
            demand: ValueDemand::Values,
        })
        .collect()
}

fn world() -> World {
    World::with(View::standard(), methods(&lab_family(producer())))
}

fn bind(w: &World, s: &Source) -> LaneBinding {
    w.bind_fields(s, &[REGION, DISC_VALID], READER, s.space())
        .expect("region and discountValid bind")
}

/// The selector: which rules are proposed for the row holding `id`.
///
/// The moduli must not both divide `ROWS - 1`: with 2 and 3 they do
/// (40,002), the selector is then the same for id `i` and its mirror, and a
/// reversed source cannot be told from candidates in the wrong row order.
fn proposed(id: i32) -> u8 {
    u8::from(id % 2 == 0) | (u8::from(id % 7 == 0) << 1)
}

/// The candidate bytes in `s`'s row order. Caller-computed in both paths.
fn candidates(s: &Source) -> Vec<u8> {
    (0..ROWS).map(|r| proposed(s.row(r).id)).collect()
}

/// Row-at-a-time answer for `s` as it is now.
fn oracle(s: &Source) -> Vec<u8> {
    (0..ROWS)
        .map(|r| {
            let row = s.row(r);
            let low = row.region <= 2 && row.discount.is_some();
            let high = row.region >= 5;
            proposed(row.id) & (u8::from(low) | (u8::from(high) << 1))
        })
        .collect()
}

/// What a caller of the raw path holds: copies of two lanes.
fn raw_lanes(s: &Source) -> Vec<Vec<u32>> {
    vec![
        (0..ROWS).map(|r| s.row(r).region).collect(),
        (0..ROWS)
            .map(|r| u32::from(s.row(r).discount.is_some()))
            .collect(),
    ]
}

fn raw(described: &Family, s: &Source, cands: &[u8]) -> Result<Vec<u8>, TwinRefusal> {
    Ok(raw_family(producer())
        .bind()?
        .decide(described, &raw_lanes(s), cands)?
        .holds)
}

#[derive(Debug, PartialEq)]
enum Stop {
    Method(Refusal),
    Description,
    Lanes(Refused),
}

/// The bound path: every proposed rule is admitted as a method, checked
/// against the caller's description, lowered over the binding's operands and
/// run over lanes taken through the binding.
fn bound(
    w: &World,
    b: &LaneBinding,
    s: &Source,
    described: &Family,
    cands: &[u8],
) -> Result<Vec<u8>, Stop> {
    let mut holds = vec![0u8; cands.len()];
    for i in 0..described.rules.len() {
        let bit = 1u8 << i;
        if !cands.iter().any(|c| c & bit != 0) {
            continue;
        }
        let mref = w
            .store
            .method_ref(CLASS, MethodId(i as u32 + 1))
            .map_err(Stop::Method)?;
        let method = w.store.admit(&mref).map_err(Stop::Method)?;
        let stored = method.when.as_ref().expect("every rule is guarded");
        if described.guard(i, described.producer).witness() != stored.witness() {
            return Err(Stop::Description);
        }
        let filter = Filter::and(described.rules[i].when.iter().map(|(lane, values)| {
            let col = operand(b, *lane as u8).expect("the guard reads bound fields only");
            Filter::or(values.iter().map(|&v| Filter::cmp(col, Cmp::EqU32(v))))
        }));
        let program = lower(&Query {
            filter,
            agg: Agg::Rows,
        })
        .expect("the guard lowers");
        let mut out = vec![0u64; ROWS.div_ceil(64)];
        let mut scratch = Scratch::for_program(&program, ROWS).unwrap();
        let p = Prepared::over(program, s, b);
        admitted(&p, w, &w.view, b, s, &mut scratch, Out::Mask(&mut out)).map_err(Stop::Lanes)?;
        for (r, h) in holds.iter_mut().enumerate() {
            if cands[r] & bit != 0 && out[r / 64] >> (r % 64) & 1 == 1 {
                *h |= bit;
            }
        }
    }
    Ok(holds)
}

/// Case 1. FAILS IF: the bound path or the raw path differs from the row
/// loop, or the answer is trivial (no row, every row, or one rule only).
#[test]
fn the_right_source_gives_the_oracle_answer_on_both_paths() {
    let w = world();
    let s = Source::new(RowOrder::ById);
    let b = bind(&w, &s);
    let cands = candidates(&s);
    let want = oracle(&s);
    for bit in [1u8, 2] {
        let n = want.iter().filter(|&&h| h & bit != 0).count();
        let proposed = cands.iter().filter(|&&c| c & bit != 0).count();
        assert!(
            n > 1_000 && n * 2 < proposed,
            "rule bit {bit}: {n} of {proposed}"
        );
    }
    assert_eq!(
        bound(&w, &b, &s, &lab_family(producer()), &cands),
        Ok(want.clone())
    );
    assert_eq!(raw(&raw_family(producer()), &s, &cands), Ok(want));
}

/// Case 2. FAILS IF: a binding made over one source reads another source of
/// the same row count. The raw path is shown to accept it.
#[test]
fn another_source_of_the_same_size_is_refused() {
    let w = world();
    let s = Source::new(RowOrder::ById);
    let other = Source::new(RowOrder::Reversed);
    let b = bind(&w, &s);
    let cands = candidates(&s);
    assert_eq!(
        bound(&w, &b, &other, &lab_family(producer()), &cands),
        Err(Stop::Lanes(Refused::Binding(LaneRefusal::Frame(
            FrameRefusal::ForeignFrame
        ))))
    );
    // Raw: the other source's lanes under this source's candidates. Accepted,
    // and the answer is neither source's.
    let got = raw(&raw_family(producer()), &other, &cands).expect("the raw path accepts");
    assert_ne!(got, oracle(&s));
    assert_ne!(got, oracle(&other));
}

/// Case 3. FAILS IF: lanes are read through a binding after the source
/// declared a new revision.
#[test]
fn a_frame_advanced_after_binding_is_refused() {
    let w = world();
    let mut s = Source::new(RowOrder::ById);
    let b = bind(&w, &s);
    let cands = candidates(&s);
    assert!(bound(&w, &b, &s, &lab_family(producer()), &cands).is_ok());
    s.advance_source();
    assert!(matches!(
        bound(&w, &b, &s, &lab_family(producer()), &cands),
        Err(Stop::Lanes(Refused::Binding(LaneRefusal::Frame(_))))
    ));
    assert!(raw(&raw_family(producer()), &s, &cands).is_ok());
}

/// Case 4. FAILS IF: lanes in another row order than the reader asked for
/// are bound, or a reorder after binding is read through.
#[test]
fn another_row_order_is_refused() {
    let w = world();
    let mut s = Source::new(RowOrder::ById);
    let asked = CoordSpace {
        population: 2,
        coordinates: 1,
    };
    assert!(matches!(
        w.bind_fields(&s, &[REGION, DISC_VALID], READER, asked),
        Err(LaneRefusal::CoordinateSpace { .. })
    ));

    let b = bind(&w, &s);
    let cands = candidates(&s);
    let before = oracle(&s);
    s.reorder(1_000);
    assert!(matches!(
        bound(&w, &b, &s, &lab_family(producer()), &cands),
        Err(Stop::Lanes(Refused::Binding(_)))
    ));
    // Raw: the rotated lanes under the old candidates. Accepted and wrong for
    // both row orders.
    let got = raw(&raw_family(producer()), &s, &cands).expect("the raw path accepts");
    assert_ne!(got, before);
    assert_ne!(got, oracle(&s));
}

/// Case 5a, RECORDED. The validity lane is rewritten as another producer
/// would derive it; the source does not move its frame.
///
/// FAILS IF: the binding turns out to refuse (then the plan's "unsatisfied
/// contract" would be wrong), or the rewritten lane does not change the
/// answer (then the case shows nothing). It does NOT assert that the bound
/// path is safe here: it pins that it is not.
#[test]
fn a_lane_rewritten_by_another_producer_under_the_same_frame_is_admitted() {
    let w = world();
    let mut s = Source::new(RowOrder::ById);
    let b = bind(&w, &s);
    let cands = candidates(&s);
    let described = lab_family(producer());
    let before = bound(&w, &b, &s, &described, &cands).unwrap();

    s.rederive_validity(|id| id % 3 != 0);
    let after =
        bound(&w, &b, &s, &described, &cands).expect("the binding has no witness of lane contents");
    assert_ne!(after, before);
    assert_eq!(after, oracle(&s));
    // The digest the caller states is the bound one, and nothing compares it
    // with the lanes.
    assert_eq!(described.producer, producer());

    // What the digest does bind: the method description.
    let other = digest(&["quack-boundary-lab", "validity", "id % 3 != 0"]);
    assert_ne!(other, producer());
    assert_eq!(
        bound(&w, &b, &s, &lab_family(other), &cands),
        Err(Stop::Description)
    );
}

/// Case 5b. FAILS IF: a rewrite the source DOES declare is read through.
#[test]
fn a_declared_rewrite_is_refused() {
    let w = world();
    let mut s = Source::new(RowOrder::ById);
    let b = bind(&w, &s);
    let cands = candidates(&s);
    s.rederive_validity(|id| id % 3 != 0);
    s.advance_source();
    assert!(matches!(
        bound(&w, &b, &s, &lab_family(producer()), &cands),
        Err(Stop::Lanes(Refused::Binding(LaneRefusal::Frame(_))))
    ));
}
