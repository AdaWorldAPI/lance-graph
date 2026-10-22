use dr_math_binding_probe::{
    bind_dr64, Axis, AxisKind, BindingWitness, BoundDr64, MethodSignature,
    population::{observe_focus, ClaimRefusal, PopulationReport},
};

fn w() -> BindingWitness {
    BindingWitness { source_revision: 8, relation_map_revision: 1,
        coordinate_revision: 2, focus_revision: 3, authorization_scope: 4 }
}
fn sig(method: u16, column: AxisKind) -> MethodSignature {
    MethodSignature::presence(7, method,
        Axis { class: 7, kind: AxisKind::Identity },
        Axis { class: 7, kind: column })
}
fn bind<'a>(s: MethodSignature, rows: &'a [u64; 64]) -> BoundDr64<'a> {
    bind_dr64(s, s, w(), w(), rows).unwrap()
}
fn first_bits() -> [u64; 64] {
    let mut rows = [0u64;64];
    rows[1] = (1<<2) | (1<<3) | (1<<4);
    rows[5] = 1<<6;
    rows
}
#[test]
fn exact_complete_focus_passes_without_causal_claim() {
    let selected = first_bits();
    let g = bind(sig(1, AxisKind::Property), &selected);
    let r = bind(sig(2, AxisKind::Property), &selected);
    let summary = observe_focus(&g, &r, None, w()).unwrap();
    assert_eq!(summary, PopulationReport { eligible: 4, true_pairs: 4, false_pairs: 0, unknown_pairs: 0 });
    assert_eq!(summary.universal_on_focus().unwrap().checked(), 4);
}
#[test]
fn one_concrete_counterexample_refuses_claim() {
    let selected = first_bits();
    let mut outcomes = selected;
    outcomes[5] = 0;
    let g = bind(sig(1, AxisKind::Property), &selected);
    let r = bind(sig(2, AxisKind::Property), &outcomes);
    assert_eq!(observe_focus(&g, &r, None, w()).unwrap().universal_on_focus(),
        Err(ClaimRefusal::Counterexamples(1)));
}
#[test]
fn unknown_is_not_false_or_proof() {
    let selected = first_bits();
    let mut known = selected;
    known[1] &= !(1<<3);
    let g = bind(sig(1, AxisKind::Property), &selected);
    let r = bind(sig(2, AxisKind::Property), &selected);
    let v = bind(sig(3, AxisKind::Property), &known);
    let observed = observe_focus(&g, &r, Some(&v), w()).unwrap();
    assert_eq!((observed.true_pairs,observed.false_pairs,observed.unknown_pairs),(3,0,1));
    assert_eq!(observed.universal_on_focus(), Err(ClaimRefusal::Unknown(1)));
}
#[test]
fn empty_population_is_a_vacuity_not_a_law() {
    let empty = [0u64;64];
    let g = bind(sig(1, AxisKind::Property), &empty);
    let r = bind(sig(2, AxisKind::Property), &empty);
    assert_eq!(observe_focus(&g,&r,None,w()).unwrap().universal_on_focus(),
        Err(ClaimRefusal::Vacuous));
}
#[test]
fn axes_and_snapshot_are_checked_before_reading_bits() {
    let selected = first_bits();
    let g = bind(sig(1, AxisKind::Property), &selected);
    let wrong = bind(sig(2, AxisKind::Identity), &selected);
    assert_eq!(observe_focus(&g,&wrong,None,w()),Err(ClaimRefusal::DifferentAxes));
    let mut stale = w();
    stale.source_revision += 1;
    assert_eq!(observe_focus(&g,&g,None,stale),Err(ClaimRefusal::ChangedWitness));
}
#[test]
fn packed_bit_math_matches_independent_scalar_pair_enumeration() {
    let selected = first_bits();
    let mut outcomes = selected;
    outcomes[1] &= !(1<<2);
    let mut known = selected;
    known[1] &= !(1<<4);
    let g=bind(sig(1,AxisKind::Property),&selected);
    let r=bind(sig(2,AxisKind::Property),&outcomes);
    let v=bind(sig(3,AxisKind::Property),&known);
    let report=observe_focus(&g,&r,Some(&v),w()).unwrap();
    let mut scalar=[0u32;4]; // eligible, true, false, unknown
    for (i, row) in selected.iter().enumerate() {
        for j in 0..64 {
            if row & (1u64<<j)==0 {continue}
            scalar[0]+=1;
            if known[i] & (1u64<<j)==0 {scalar[3]+=1}
            else if outcomes[i] & (1u64<<j)!=0 {scalar[1]+=1}
            else {scalar[2]+=1}
        }
    }
    assert_eq!(report,PopulationReport {
        eligible:scalar[0],true_pairs:scalar[1],
        false_pairs:scalar[2],unknown_pairs:scalar[3]
    });
    assert_eq!((report.eligible,report.true_pairs,report.false_pairs,report.unknown_pairs),(4,2,1,1));
}
