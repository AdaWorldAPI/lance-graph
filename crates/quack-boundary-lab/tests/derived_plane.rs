//! D-QUACK-DERIVED-PLANE-PROBE, P2 and P3: planes derived from bound lanes.
//!
//! Pass conditions: `.claude/plans/d-quack-derived-plane-probe.md` sections
//! 5, 6 and 12, committed before this file existed.

use dr256_ternlog_lab::bit_planes::BitPlanes;
use hidden_state_twins_probe::{digest, Family, Rule};
use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::FrameRefusal;
use lance_graph_dt64::{LaneBinding, LaneRefusal, MethodBinding, MethodId, SigRev, ValueDemand};
use lance_graph_mask_risc::{Foreign, MaskOp, Operand, Out, Program, Scratch, Terminal, Value};
use lance_graph_quack::{lower, Agg, Cmp, Filter, Mask, Query};
use quack_boundary_lab::derived::{admitted_planes, DerivedPlane, PlaneRefusal, Recipe};
use quack_boundary_lab::{
    admitted, operand, unbound, Prepared, Refused, RowOrder, Source, View, World, CLASS,
    DISC_VALID, FIELDS, PARTNER, PARTNERS, READER, REGION, ROWS,
};

fn producer() -> u32 {
    digest(&["quack-boundary-lab", "validity", "id % 5 != 0"])
}

fn family(producer: u32) -> Family {
    Family {
        class: CLASS,
        lanes: FIELDS,
        rules: vec![
            Rule {
                name: "low_region_with_discount",
                when: vec![
                    (u16::from(REGION), vec![0, 1, 2]),
                    (u16::from(DISC_VALID), vec![1]),
                ],
            },
            Rule {
                name: "high_region",
                when: vec![(u16::from(REGION), vec![5, 6, 7])],
            },
        ],
        producer,
    }
}

fn world() -> World {
    let f = family(producer());
    let fields = WideFieldMask::from_positions(&[REGION, DISC_VALID]);
    let methods = (0..f.rules.len())
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
        .collect();
    World::with(View::standard(), methods)
}

fn bind(w: &World, s: &Source) -> LaneBinding {
    w.bind_fields(s, &[REGION, PARTNER, DISC_VALID], READER, s.space())
        .expect("three fields bind")
}

fn bit(w: &World, b: &LaneBinding, s: &Source, field: u8, k: u8) -> DerivedPlane {
    DerivedPlane::bit(w, &w.view, b, s, field, k).expect("the bit-plane derives")
}

/// Rows whose region is in `subset`: one ternlog over planes (bit 2, bit 1,
/// bit 0), band index `(a << 2) | (b << 1) | c`.
fn band(subset: u8) -> Program {
    Program::new(
        vec![MaskOp::Ternlog {
            imm: subset,
            a: Operand::Plane(0),
            b: Operand::Plane(1),
            c: Operand::Plane(2),
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    )
}

const REGION_BITS: [Recipe; 3] = [
    Recipe::Bit {
        field: REGION,
        k: 2,
    },
    Recipe::Bit {
        field: REGION,
        k: 1,
    },
    Recipe::Bit {
        field: REGION,
        k: 0,
    },
];

fn count_planes(
    program: Program,
    expects: &[Recipe],
    planes: &[&DerivedPlane],
    w: &World,
    b: &LaneBinding,
    s: &Source,
) -> Result<Value, Refused> {
    let mut scratch = Scratch::for_program(&program, ROWS).unwrap();
    let p = Prepared::over(program, s, b);
    admitted_planes(
        &p,
        expects,
        planes,
        w,
        &w.view,
        b,
        s,
        &mut scratch,
        Out::None,
    )
}

fn count_plane0() -> Program {
    lower(&Query {
        filter: Filter::plane(Mask(0)),
        agg: Agg::Count,
    })
    .unwrap()
}

fn plane_of(set: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut words = vec![0u64; ROWS.div_ceil(64)];
    for r in (0..ROWS).filter(|&r| set(r)) {
        words[r / 64] |= 1u64 << (r % 64);
    }
    words
}

/// P2.1. FAILS IF: a derived bit-plane differs from PR 2057's slicing of the
/// same codes, or a band-subset count differs from the row loop, or fewer
/// than 200 of the 256 subsets have a non-trivial answer.
#[test]
fn bit_planes_of_a_bound_lane_answer_every_band_subset() {
    let w = world();
    let s = Source::new(RowOrder::ById);
    let b = bind(&w, &s);
    let planes = [
        bit(&w, &b, &s, REGION, 2),
        bit(&w, &b, &s, REGION, 1),
        bit(&w, &b, &s, REGION, 0),
    ];
    let codes: Vec<u8> = (0..ROWS).map(|r| s.row(r).region as u8).collect();
    let sliced = BitPlanes::of(&codes);
    for (plane, k) in planes.iter().zip([2usize, 1, 0]) {
        assert_eq!(plane.words(), sliced.planes[k].as_slice(), "bit {k}");
    }
    assert!(bit(&w, &b, &s, REGION, 3).words().iter().all(|&x| x == 0));

    let mut per_region = [0usize; 8];
    for &c in &codes {
        per_region[usize::from(c)] += 1;
    }
    let refs: Vec<&DerivedPlane> = planes.iter().collect();
    let mut inside = 0;
    for subset in 0..=255u8 {
        let want: usize = (0..8)
            .filter(|r| (subset >> r) & 1 == 1)
            .map(|r| per_region[r])
            .sum();
        assert_eq!(
            count_planes(band(subset), &REGION_BITS, &refs, &w, &b, &s),
            Ok(Value::Count(want)),
            "subset {subset:#04x}"
        );
        inside += usize::from(want > 0 && want < ROWS);
    }
    assert!(inside >= 200, "only {inside} non-trivial subsets");
}

/// P2.2. FAILS IF: a valid binding lets a plane of another bit or another
/// field stand in for the one the program expects. The raw path is shown to
/// accept the substitute and answer another question.
#[test]
fn a_valid_binding_does_not_admit_another_plane() {
    let w = world();
    let s = Source::new(RowOrder::ById);
    let b = bind(&w, &s);
    let (b2, b1, b0) = (
        bit(&w, &b, &s, REGION, 2),
        bit(&w, &b, &s, REGION, 1),
        bit(&w, &b, &s, REGION, 0),
    );
    let valid = bit(&w, &b, &s, DISC_VALID, 0);
    let region_1 = 0b0000_0010u8;
    let want = (0..ROWS).filter(|&r| s.row(r).region == 1).count();
    assert_eq!(
        count_planes(band(region_1), &REGION_BITS, &[&b2, &b1, &b0], &w, &b, &s),
        Ok(Value::Count(want))
    );

    // Another bit of the right field.
    assert_eq!(
        count_planes(band(region_1), &REGION_BITS, &[&b0, &b1, &b2], &w, &b, &s),
        Err(Refused::Plane(PlaneRefusal::Recipe {
            at: 0,
            expected: REGION_BITS[0],
            got: REGION_BITS[2],
        }))
    );
    // Bit 0 of another field.
    assert_eq!(
        count_planes(
            band(region_1),
            &REGION_BITS,
            &[&b2, &b1, &valid],
            &w,
            &b,
            &s
        ),
        Err(Refused::Plane(PlaneRefusal::Recipe {
            at: 2,
            expected: REGION_BITS[2],
            got: Recipe::Bit {
                field: DISC_VALID,
                k: 0,
            },
        }))
    );
    // One plane short.
    assert_eq!(
        count_planes(band(region_1), &REGION_BITS, &[&b2, &b1], &w, &b, &s),
        Err(Refused::Plane(PlaneRefusal::Count {
            expected: 3,
            got: 2,
        }))
    );

    // Raw: the same substitute, handed straight to the executor.
    let program = band(region_1);
    let got = unbound(
        &program,
        &s.raw_lanes(),
        &[b2.words(), b1.words(), valid.words()],
        &Foreign::NONE,
        &mut Scratch::for_program(&program, ROWS).unwrap(),
        Out::None,
    );
    assert!(matches!(got, Ok(Value::Count(n)) if n != want && n > 0));
}

/// P2.3. FAILS IF: a plane derived under an earlier frame or row order is
/// read after the source moved, including through a fresh, valid binding.
#[test]
fn a_plane_does_not_outlive_its_frame() {
    let w = world();
    let mut s = Source::new(RowOrder::ById);
    let b = bind(&w, &s);
    let valid = bit(&w, &b, &s, DISC_VALID, 0);
    let expects = [Recipe::Bit {
        field: DISC_VALID,
        k: 0,
    }];
    assert!(count_planes(count_plane0(), &expects, &[&valid], &w, &b, &s).is_ok());

    s.advance_source();
    // The old binding is refused by its own check.
    assert!(matches!(
        count_planes(count_plane0(), &expects, &[&valid], &w, &b, &s),
        Err(Refused::Binding(LaneRefusal::Frame(_)))
    ));
    // A fresh binding is valid; the plane is not.
    let fresh = bind(&w, &s);
    assert_eq!(
        count_planes(count_plane0(), &expects, &[&valid], &w, &fresh, &s),
        Err(Refused::Plane(PlaneRefusal::Frame {
            at: 0,
            why: FrameRefusal::SourceChanged,
        }))
    );

    // A reorder: the plane is in the old row order.
    let valid = bit(&w, &fresh, &s, DISC_VALID, 0);
    s.reorder(1_000);
    let fresh = bind(&w, &s);
    assert_eq!(
        count_planes(count_plane0(), &expects, &[&valid], &w, &fresh, &s),
        Err(Refused::Plane(PlaneRefusal::Space { at: 0 }))
    );
}

/// P2.4, kind (b), RECORDED. The lane is rewritten under an unmoved frame
/// after the plane was derived.
///
/// FAILS IF: either path refuses (the boundary would then be closed and the
/// plan wrong), or the two paths agree (the case would show nothing).
#[test]
fn a_rewrite_under_an_unmoved_frame_leaves_a_stale_plane_admitted() {
    let w = world();
    let mut s = Source::new(RowOrder::ById);
    let b = bind(&w, &s);
    let valid = bit(&w, &b, &s, DISC_VALID, 0);
    let expects = [Recipe::Bit {
        field: DISC_VALID,
        k: 0,
    }];
    let lane_count = |s: &Source| {
        let program = lower(&Query {
            filter: Filter::cmp(operand(&b, DISC_VALID).unwrap(), Cmp::EqU32(1)),
            agg: Agg::Count,
        })
        .unwrap();
        let mut scratch = Scratch::for_program(&program, ROWS).unwrap();
        let p = Prepared::over(program, s, &b);
        admitted(&p, &w, &w.view, &b, s, &mut scratch, Out::None)
    };
    let before = (0..ROWS).filter(|&r| s.row(r).discount.is_some()).count();
    assert_eq!(lane_count(&s), Ok(Value::Count(before)));
    assert_eq!(
        count_planes(count_plane0(), &expects, &[&valid], &w, &b, &s),
        Ok(Value::Count(before))
    );

    s.rederive_validity(|id| id % 3 != 0);
    let now = (0..ROWS).filter(|&r| s.row(r).discount.is_some()).count();
    assert_ne!(now, before);
    // The lane path reads the rewritten bytes; the plane path the old ones.
    assert_eq!(lane_count(&s), Ok(Value::Count(now)));
    assert_eq!(
        count_planes(count_plane0(), &expects, &[&valid], &w, &b, &s),
        Ok(Value::Count(before))
    );
}

fn rule_filter(f: &Family, i: usize, b: &LaneBinding) -> Filter {
    Filter::and(f.rules[i].when.iter().map(|(lane, values)| {
        let col = operand(b, *lane as u8).unwrap();
        Filter::or(values.iter().map(|&v| Filter::cmp(col, Cmp::EqU32(v))))
    }))
}

/// P3, derived interpretation. FAILS IF: a guard plane differs from the row
/// loop, a plane of another method is accepted for it, or a description of
/// another producer derives one. The last block pins what is NOT checked.
#[test]
fn a_guard_plane_checks_the_method_and_not_the_lowering() {
    let w = world();
    let s = Source::new(RowOrder::ById);
    let b = bind(&w, &s);
    let f = family(producer());
    let derive = |method: u32, described: usize, lowered: usize, fam: &Family| {
        DerivedPlane::guard(
            &w,
            &w.view,
            &b,
            &s,
            MethodId(method),
            &fam.guard(described, fam.producer),
            rule_filter(fam, lowered, &b),
        )
    };
    let low = derive(1, 0, 0, &f).expect("rule 0 derives");
    let high = derive(2, 1, 1, &f).expect("rule 1 derives");
    let want_low = plane_of(|r| s.row(r).region <= 2 && s.row(r).discount.is_some());
    let want_high = plane_of(|r| s.row(r).region >= 5);
    assert_eq!(low.words(), want_low.as_slice());
    assert_eq!(high.words(), want_high.as_slice());
    assert_ne!(want_low, want_high);

    let expects = [Recipe::Guard {
        method: MethodId(1),
    }];
    let n_low = want_low.iter().map(|x| x.count_ones() as usize).sum();
    assert_eq!(
        count_planes(count_plane0(), &expects, &[&low], &w, &b, &s),
        Ok(Value::Count(n_low))
    );
    assert!(matches!(
        count_planes(count_plane0(), &expects, &[&high], &w, &b, &s),
        Err(Refused::Plane(PlaneRefusal::Recipe { at: 0, .. }))
    ));

    // Another producer's description, or another rule's, is not the bound guard.
    let other = family(digest(&["quack-boundary-lab", "validity", "id % 3 != 0"]));
    assert_eq!(
        derive(1, 0, 0, &other).err(),
        Some(Refused::Plane(PlaneRefusal::Description))
    );
    assert_eq!(
        derive(1, 1, 1, &f).err(),
        Some(Refused::Plane(PlaneRefusal::Description))
    );

    // NOT checked: that the lowered filter is the described guard. Rule 0's
    // description with rule 1's filter derives a plane recorded as method 1
    // and holding rule 1's rows.
    let mislowered = derive(1, 0, 1, &f).expect("the lowering is the caller's");
    assert_eq!(
        mislowered.recipe(),
        Recipe::Guard {
            method: MethodId(1)
        }
    );
    assert_eq!(mislowered.words(), want_high.as_slice());
}

/// P3, cross-population. FAILS IF: two different foreign visibility sets
/// give planes the local record can tell apart (then the fk plane would need
/// nothing beyond the local binding), or either plane is wrong for its set.
#[test]
fn an_fk_plane_s_record_does_not_name_the_foreign_side() {
    let w = world();
    let s = Source::new(RowOrder::ById);
    let b = bind(&w, &s);
    let visible = |keep: fn(usize) -> bool| {
        let mut words = vec![0u64; PARTNERS.div_ceil(64)];
        for k in (0..PARTNERS).filter(|&k| keep(k)) {
            words[k / 64] |= 1u64 << (k % 64);
        }
        words
    };
    let (even, thirds) = (
        visible(|k| k.is_multiple_of(2)),
        visible(|k| k.is_multiple_of(3)),
    );
    let derive = |set: &[u64]| DerivedPlane::foreign_visible(&w, &w.view, &b, &s, PARTNER, set);
    let (a, c) = (derive(&even).unwrap(), derive(&thirds).unwrap());
    assert_eq!(
        a.words(),
        plane_of(|r| s.row(r).partner.is_multiple_of(2)).as_slice()
    );
    assert_eq!(
        c.words(),
        plane_of(|r| s.row(r).partner.is_multiple_of(3)).as_slice()
    );
    assert_ne!(a.words(), c.words());

    // Same recipe, same row order, both accepted for the same expectation.
    assert_eq!(a.recipe(), c.recipe());
    assert_eq!(a.space(), c.space());
    let expects = [Recipe::ForeignVisible { fk: PARTNER }];
    let n = |p: &DerivedPlane| count_planes(count_plane0(), &expects, &[p], &w, &b, &s);
    assert!(matches!(n(&a), Ok(Value::Count(x)) if x > 1_000));
    assert!(matches!(n(&c), Ok(Value::Count(x)) if x > 1_000));
    assert_ne!(n(&a), n(&c));

    // A visibility set over a smaller foreign domain.
    assert!(matches!(
        derive(&even[..10]).err(),
        Some(Refused::Plane(PlaneRefusal::ForeignKeyOutOfRange { .. }))
    ));
}
