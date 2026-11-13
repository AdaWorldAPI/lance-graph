//! P0: orientation is a real CE64 reading, not an 8-bit grammar opcode.
//!
//! Read-only differential tests against an independent scalar bit oracle.
//! The Python P0 reference fixture covers Rails receiver and river coal twins;
//! this file checks the actual causal-edge accessors. No class/bit minting.

use causal_edge::{CausalEdge64, Orientation};

const ROLE_SHIFT: u32 = 43;
const ROLE_CARE: u64 = 0b111 << ROLE_SHIFT;

fn scalar_role_match(word: u64, role: u8) -> bool {
    let pattern = u64::from(role) << ROLE_SHIFT;
    ((word ^ pattern) & ROLE_CARE) == 0
}

#[test]
fn real_ce64_orientation_accessor_matches_independent_bit_oracle() {
    // Covers 8 role readings x 32 raw Epi5 patterns. The 32 patterns are
    // *raw bits*, NOT 32 admitted epistemic certifications.
    for orientation in Orientation::ALL {
        for epi5 in 0u64..32 {
            let mut edge = CausalEdge64(0x00_12_34_56);
            edge.set_orientation(orientation);
            edge.0 |= epi5 << 59;
            assert_eq!(edge.orientation(), orientation);
            for expected in 0u8..8 {
                assert_eq!(
                    scalar_role_match(edge.0, expected),
                    orientation.code() == expected,
                    "orientation={orientation:?}, epi5={epi5}, expected={expected}"
                );
            }
        }
    }
}

#[test]
fn same_spo_and_evidence_can_have_opposite_role_selection() {
    let mut recipient = CausalEdge64(0x00_12_34_56);
    let mut core = recipient;
    recipient.set_orientation(Orientation::Recipient);
    core.set_orientation(Orientation::Core);

    // The low 40 bits: SPO and frequency/confidence. Identical facts under
    // these projections can still disagree on the declared semantic role.
    let low40 = (1u64 << 40) - 1;
    assert_eq!(recipient.0 & low40, core.0 & low40);
    assert!(scalar_role_match(
        recipient.0,
        Orientation::Recipient.code()
    ));
    assert!(!scalar_role_match(core.0, Orientation::Recipient.code()));
}

#[test]
fn epistemic_topology_must_not_be_a_proxy_for_semantic_role() {
    let mut recipient = CausalEdge64(0x00_12_34_56);
    recipient.set_orientation(Orientation::Recipient);
    let changed_epi5 = CausalEdge64(recipient.0 | (0b10101u64 << 59));
    assert_eq!(recipient.orientation(), changed_epi5.orientation());
    assert!(scalar_role_match(
        changed_epi5.0,
        Orientation::Recipient.code()
    ));
    let mut changed_role = changed_epi5;
    changed_role.set_orientation(Orientation::Core);
    assert!(!scalar_role_match(
        changed_role.0,
        Orientation::Recipient.code()
    ));
}
