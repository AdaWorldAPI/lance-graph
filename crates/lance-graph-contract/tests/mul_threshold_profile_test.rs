//! Tests for `MulThresholdProfile` (D-ONTO-V5-9).
//!
//! Per `lance-graph-ontology-v5.md` §D-9, medical contexts must select
//! stricter trust / flow / drift thresholds than callcenter contexts;
//! unmapped contexts fall through to a moderate default profile.

use lance_graph_contract::mul::MulThresholdProfile;

#[test]
fn medical_context_selects_medical_profile() {
    // Healthcare namespace id (per the v5 D-9 mapping).
    assert_eq!(
        MulThresholdProfile::for_context(2),
        MulThresholdProfile::MEDICAL
    );
}

#[test]
fn workorder_context_selects_callcenter_profile() {
    // WorkOrder namespace id (per the v5 D-9 mapping).
    assert_eq!(
        MulThresholdProfile::for_context(1),
        MulThresholdProfile::CALLCENTER
    );
}

#[test]
fn medical_subnamespace_range_selects_medical_profile() {
    // Medical/* subnamespaces are 10..=19 (BioPortal stubs land here per
    // D-CASCADE-V1-4). Spot-check 10, 15, and 19.
    assert_eq!(
        MulThresholdProfile::for_context(10),
        MulThresholdProfile::MEDICAL
    );
    assert_eq!(
        MulThresholdProfile::for_context(15),
        MulThresholdProfile::MEDICAL
    );
    assert_eq!(
        MulThresholdProfile::for_context(19),
        MulThresholdProfile::MEDICAL
    );
}

#[test]
fn unmapped_context_falls_through_to_default() {
    assert_eq!(
        MulThresholdProfile::for_context(99),
        MulThresholdProfile::DEFAULT
    );
    assert_eq!(
        MulThresholdProfile::for_context(0),
        MulThresholdProfile::DEFAULT
    );
    // Just above the medical range.
    assert_eq!(
        MulThresholdProfile::for_context(20),
        MulThresholdProfile::DEFAULT
    );
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn medical_is_stricter_than_callcenter() {
    // The fundamental ordering invariant of D-9: medical demands more
    // trust, more flow, and tolerates less angular drift. The values
    // are const so clippy flags the assertion as compile-time-known,
    // but the test is still valuable as a regression guard if the
    // const profile values are ever touched.
    assert!(MulThresholdProfile::MEDICAL.trust_min > MulThresholdProfile::CALLCENTER.trust_min);
    assert!(MulThresholdProfile::MEDICAL.flow_min > MulThresholdProfile::CALLCENTER.flow_min);
    assert!(MulThresholdProfile::MEDICAL.compass_max < MulThresholdProfile::CALLCENTER.compass_max);
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn default_sits_between_medical_and_callcenter() {
    let m = MulThresholdProfile::MEDICAL;
    let c = MulThresholdProfile::CALLCENTER;
    let d = MulThresholdProfile::DEFAULT;
    assert!(d.trust_min > c.trust_min && d.trust_min < m.trust_min);
    assert!(d.flow_min > c.flow_min && d.flow_min < m.flow_min);
    assert!(d.compass_max > m.compass_max && d.compass_max < c.compass_max);
}

#[test]
fn profile_labels_are_stable() {
    assert_eq!(MulThresholdProfile::MEDICAL.label, "medical");
    assert_eq!(MulThresholdProfile::CALLCENTER.label, "callcenter");
    assert_eq!(MulThresholdProfile::DEFAULT.label, "default");
}
