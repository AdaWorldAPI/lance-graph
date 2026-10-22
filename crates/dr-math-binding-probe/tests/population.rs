use dr_math_binding_probe::{
    Axis, AxisKind, BoundDr64, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Operand,
    Scope, Truth, bind_dr64,
    population::{ClaimRefusal, PopulationReport, observe_focus},
};

const FOCUS: Focus = Focus { revision: 3 };
const SCOPE: Scope = Scope { id: 4 };

fn store() -> FrameStore {
    FrameStore::new(FrameRevisions {
        source: 8,
        relation_map: 1,
        coordinates: 2,
    })
}
fn sig(method: u16, column: AxisKind) -> MethodSignature {
    MethodSignature::presence(
        7,
        method,
        Axis {
            class: 7,
            kind: AxisKind::Identity,
        },
        Axis {
            class: 7,
            kind: column,
        },
    )
}
fn three_valued(mut s: MethodSignature) -> MethodSignature {
    s.truth = Truth::ThreeValued;
    s
}
fn bind_at<'a>(
    store: &FrameStore,
    focus: Focus,
    s: MethodSignature,
    rows: &'a [u64; 64],
    validity: Option<&'a [u64; 64]>,
) -> BoundDr64<'a> {
    let at = Guards {
        frame: store.seal(),
        focus,
        scope: SCOPE,
    };
    bind_dr64(store, s, s, &at, (focus, SCOPE), Operand { rows, validity }).unwrap()
}
fn bind<'a>(store: &FrameStore, s: MethodSignature, rows: &'a [u64; 64]) -> BoundDr64<'a> {
    bind_at(store, FOCUS, s, rows, None)
}
fn gamma_bits() -> [u64; 64] {
    let mut rows = [0u64; 64];
    rows[1] = (1 << 2) | (1 << 3);
    rows[5] = 1 << 6;
    rows
}
/// R holds every Gamma pair AND pairs outside Gamma (bits 4 and 9), so a
/// checker that reads Gamma in place of R, or counts R outside the focus,
/// gives a different answer.
fn relation_bits() -> [u64; 64] {
    let mut rows = [0u64; 64];
    rows[1] = (1 << 2) | (1 << 3) | (1 << 4);
    rows[5] = (1 << 6) | (1 << 9);
    rows
}

#[test]
fn exact_complete_focus_passes_without_causal_claim() {
    let st = store();
    let (selected, relation) = (gamma_bits(), relation_bits());
    // Anti-vacuity: Gamma and R are different bits, and R reaches outside Gamma.
    assert_ne!(selected, relation);
    assert!((0..64).any(|i| relation[i] & !selected[i] != 0));
    let g = bind(&st, sig(1, AxisKind::Property), &selected);
    let r = bind(&st, sig(2, AxisKind::Property), &relation);
    let summary = observe_focus(&st, &g, &r).unwrap();
    assert_eq!(
        summary,
        PopulationReport {
            eligible: 3,
            true_pairs: 3,
            false_pairs: 0,
            unknown_pairs: 0
        }
    );
    assert_eq!(summary.universal_on_focus().unwrap().checked(), 3);
}

#[test]
fn one_concrete_counterexample_refuses_claim() {
    let st = store();
    let selected = gamma_bits();
    let mut relation = relation_bits();
    relation[5] &= !(1 << 6);
    let g = bind(&st, sig(1, AxisKind::Property), &selected);
    let r = bind(&st, sig(2, AxisKind::Property), &relation);
    assert_eq!(
        observe_focus(&st, &g, &r).unwrap().universal_on_focus(),
        Err(ClaimRefusal::Counterexamples(1))
    );
}

#[test]
fn unknown_is_not_false_or_proof() {
    let st = store();
    let (selected, relation) = (gamma_bits(), relation_bits());
    let mut known = [u64::MAX; 64];
    known[1] &= !(1 << 3);
    let g = bind(&st, sig(1, AxisKind::Property), &selected);
    let r = bind_at(
        &st,
        FOCUS,
        three_valued(sig(2, AxisKind::Property)),
        &relation,
        Some(&known),
    );
    let observed = observe_focus(&st, &g, &r).unwrap();
    assert_eq!(
        (
            observed.true_pairs,
            observed.false_pairs,
            observed.unknown_pairs
        ),
        (2, 0, 1)
    );
    assert_eq!(observed.universal_on_focus(), Err(ClaimRefusal::Unknown(1)));
}

/// Certain-answer semantics: a refuted claim is refuted whatever the unknown
/// pairs hold, so `Counterexamples` must win over `Unknown`. Gamma 0b111, R
/// 0b001, validity 0b011: one true, one false, one unknown.
#[test]
fn refuted_beats_unknown() {
    let st = store();
    let report = |relation: u64, validity: u64| {
        let (mut g, mut r, mut v) = ([0u64; 64], [0u64; 64], [0u64; 64]);
        g[0] = 0b111;
        r[0] = relation;
        v[0] = validity;
        let gb = bind(&st, sig(1, AxisKind::Property), &g);
        let rb = bind_at(
            &st,
            FOCUS,
            three_valued(sig(2, AxisKind::Property)),
            &r,
            Some(&v),
        );
        observe_focus(&st, &gb, &rb).unwrap()
    };
    let both = report(0b001, 0b011);
    assert_eq!(
        (both.true_pairs, both.false_pairs, both.unknown_pairs),
        (1, 1, 1)
    );
    assert_eq!(
        both.universal_on_focus(),
        Err(ClaimRefusal::Counterexamples(1))
    );
    // Twins: unknown only, false only.
    assert_eq!(
        report(0b011, 0b011).universal_on_focus(),
        Err(ClaimRefusal::Unknown(1))
    );
    assert_eq!(
        report(0b011, 0b111).universal_on_focus(),
        Err(ClaimRefusal::Counterexamples(1))
    );
}

#[test]
fn empty_population_is_a_vacuity_not_a_law() {
    let st = store();
    let empty = [0u64; 64];
    let relation = relation_bits();
    let g = bind(&st, sig(1, AxisKind::Property), &empty);
    let r = bind(&st, sig(2, AxisKind::Property), &relation);
    assert_eq!(
        observe_focus(&st, &g, &r).unwrap().universal_on_focus(),
        Err(ClaimRefusal::Vacuous)
    );
}

#[test]
fn axes_frame_and_focus_are_checked_before_reading_bits() {
    let mut st = store();
    let (selected, relation) = (gamma_bits(), relation_bits());
    let g = bind(&st, sig(1, AxisKind::Property), &selected);
    let wrong_axis = bind(&st, sig(2, AxisKind::Identity), &relation);
    assert_eq!(
        observe_focus(&st, &g, &wrong_axis),
        Err(ClaimRefusal::DifferentAxes)
    );
    let known = [u64::MAX; 64];
    let k3_gamma = bind_at(
        &st,
        FOCUS,
        three_valued(sig(1, AxisKind::Property)),
        &selected,
        Some(&known),
    );
    let r = bind(&st, sig(2, AxisKind::Property), &relation);
    assert_eq!(
        observe_focus(&st, &k3_gamma, &r),
        Err(ClaimRefusal::SelectionNotTwoValued)
    );
    let other_focus = bind_at(
        &st,
        Focus { revision: 9 },
        sig(2, AxisKind::Property),
        &relation,
        None,
    );
    assert_eq!(
        observe_focus(&st, &g, &other_focus),
        Err(ClaimRefusal::DifferentFocus)
    );
    let other = store();
    let foreign = bind(&other, sig(2, AxisKind::Property), &relation);
    assert_eq!(
        observe_focus(&st, &g, &foreign),
        Err(ClaimRefusal::DifferentFrame)
    );
    st.advance(FrameRevisions {
        source: 9,
        relation_map: 1,
        coordinates: 2,
    });
    assert_eq!(observe_focus(&st, &g, &r), Err(ClaimRefusal::StaleFrame));
}

#[test]
fn packed_bit_math_matches_independent_scalar_pair_enumeration() {
    let st = store();
    let selected = gamma_bits();
    let mut relation = relation_bits();
    relation[1] &= !(1 << 2);
    let mut known = [u64::MAX; 64];
    known[5] &= !(1 << 6);
    let g = bind(&st, sig(1, AxisKind::Property), &selected);
    let r = bind_at(
        &st,
        FOCUS,
        three_valued(sig(2, AxisKind::Property)),
        &relation,
        Some(&known),
    );
    let report = observe_focus(&st, &g, &r).unwrap();
    let mut scalar = [0u32; 4]; // eligible, true, false, unknown
    for (i, row) in selected.iter().enumerate() {
        for j in 0..64 {
            if row & (1u64 << j) == 0 {
                continue;
            }
            scalar[0] += 1;
            if known[i] & (1u64 << j) == 0 {
                scalar[3] += 1
            } else if relation[i] & (1u64 << j) != 0 {
                scalar[1] += 1
            } else {
                scalar[2] += 1
            }
        }
    }
    assert_eq!(
        report,
        PopulationReport {
            eligible: scalar[0],
            true_pairs: scalar[1],
            false_pairs: scalar[2],
            unknown_pairs: scalar[3]
        }
    );
    assert_eq!(
        (
            report.eligible,
            report.true_pairs,
            report.false_pairs,
            report.unknown_pairs
        ),
        (3, 1, 1, 1)
    );
    assert_eq!(
        report.universal_on_focus(),
        Err(ClaimRefusal::Counterexamples(1))
    );
}
