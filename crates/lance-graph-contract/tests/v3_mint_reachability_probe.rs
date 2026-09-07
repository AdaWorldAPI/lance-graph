// SPDX-License-Identifier: Apache-2.0

//! D-V3-MINT-1 — is the V3 tail actually reachable through `mint_for` in the
//! build configuration production uses?
//!
//! `mint_for`'s doc justifies its V1 fallback arm as dead code: *"With the
//! feature off no classid registers a V2/V3 `tail_variant` (`classid_read_mode`
//! returns V1), so the fallback arm is dead — it exists purely so the crate
//! compiles `--no-default-features`."*
//!
//! That justification is checkable. The registry's V3 entries are gated on
//! `guid-v3-tail` (default ON); `mint_for`'s V3 arm is gated on `guid-v2-tail`
//! (default OFF). Those are two different features, so the premise and the
//! conclusion can come apart.
//!
//! The discriminator is `leaf`: the V3 arm feeds it to `new_v2`, the V1
//! fallback discards it (`let _ = leaf;`). Two mints differing only in `leaf`
//! are therefore byte-equal iff the fallback ran.

use lance_graph_contract::canonical_node::{classid_read_mode, NodeGuid, TailVariant};

/// The PREMISE half of `mint_for`'s dead-code claim: does a classid actually
/// register a V2/V3 tail variant in this build?
#[test]
fn a_v3_registered_classid_resolves_to_the_v3_tail_variant() {
    let m = classid_read_mode(NodeGuid::CLASSID_OSINT_V3);
    #[cfg(feature = "guid-v3-tail")]
    assert_eq!(
        m.tail_variant,
        TailVariant::V3,
        "CLASSID_OSINT_V3 must register the V3 tail when guid-v3-tail is on"
    );
    #[cfg(not(feature = "guid-v3-tail"))]
    assert_eq!(m.tail_variant, TailVariant::V1);
}

/// The CONCLUSION half: given a classid that registers V3, does `mint_for`
/// reach the V3 arm — or silently fall back to the deprecated V1 layout?
///
/// `leaf` is the discriminator, because the V3 arm forwards it to `new_v2`
/// while the V1 fallback discards it. That keeps this probe meaningful even if
/// the feature graph is later flattened.
#[test]
fn minting_through_the_registered_tail_variant_honours_leaf() {
    let c = NodeGuid::CLASSID_OSINT_V3;
    let tv = classid_read_mode(c).tail_variant;

    // The canonical consumer incantation, verbatim from mint_for's doc.
    let a = NodeGuid::mint_for(tv, c, 0x1111, 0x2222, 0x3333, 0x00AA, 0x0001, 0x0002);
    let b = NodeGuid::mint_for(tv, c, 0x1111, 0x2222, 0x3333, 0x00BB, 0x0001, 0x0002);

    let leaf_is_live = a.as_bytes() != b.as_bytes();

    eprintln!(
        "tail_variant={tv:?} guid-v3-tail={} guid-v2-tail={} leaf_is_live={leaf_is_live}\n  a={:02x?}\n  b={:02x?}",
        cfg!(feature = "guid-v3-tail"),
        cfg!(feature = "guid-v2-tail"),
        a.as_bytes(),
        b.as_bytes(),
    );

    if tv == TailVariant::V3 {
        assert!(
            leaf_is_live,
            "classid {c:#010x} registers TailVariant::V3, but mint_for produced a \
             leaf-insensitive key — the V1 fallback arm ran. A V3 class is being \
             minted into the deprecated V1 layout, silently."
        );
    }
}
