// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! D-TEMPORAL-1 — does canonical address identity survive the cycle store?
//!
//! The claim under test, stated as bytes rather than as prose:
//!
//! ```text
//! NodeGuid G -> NodeRow -> 512-byte LE packet -> cycle Vn commit
//!   -> drop writer -> reopen -> scan_image(cycle)
//!   -> payload == submitted bytes  AND  payload[0..16] == G.as_bytes()
//! ```
//!
//! Three assertions are kept apart, because they fail independently:
//!
//! 1. the raw key bytes survive byte-exact (a storage claim);
//! 2. the V3 address decodes as V3 (a reader claim);
//! 3. reading a V3 key through the V1 accessors is WRONG (a selection claim) —
//!    pinned here as a positive assertion so that a future reader who reaches
//!    for `identity()` on a V3 key breaks this test instead of shipping.
//!
//! Assertion 3 exists because that exact false-green has already happened: a
//! byte-exact V3 round-trip read back through the V1 `identity()` accessor
//! collapsed a five-figure row count to a three-figure one, and the round-trip
//! stayed green throughout. Byte fidelity does not imply address fidelity.
//!
//! Keys are minted through `mint_for(classid_read_mode(c).tail_variant, …)` —
//! never by hardcoding `new` (V1, deprecated) or `new_v2`. The V1 rail is
//! retained only as a read-side legacy control (RESERVE, DON'T RECLAIM); no new
//! unit is minted V1.
//!
//! Two Arrow claims are also kept apart: the schema's column 0 is `kind`, not
//! an address. The address claim is about the first 16 bytes of the
//! `FixedSizeBinary(512)` payload blob.

use lance_graph::graph::cycle_sink::{LanceCycleWriter, EPISODIC_WITNESS_BYTES};
use lance_graph_contract::canonical_node::{
    classid_read_mode, EdgeBlock, NodeGuid, NodeRow, NodeRowPacket, TailVariant,
};
use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_planner::persist_sink::{
    persist_cycle, CommitOutcome, CycleFrame, CycleId, SweepSlot,
};

/// The canonical consumer incantation — the registry picks the tail, not us.
fn mint(classid: u32, leaf: u16, family: u32, identity: u32) -> NodeGuid {
    let tv = classid_read_mode(classid).tail_variant;
    NodeGuid::mint_for(tv, classid, 0x1111, 0x2222, 0x3333, leaf, family, identity)
}

fn row_of(guid: NodeGuid, fill: u8) -> NodeRow {
    NodeRow {
        key: guid,
        edges: EdgeBlock::default(),
        value: [fill; 480],
    }
}

fn packet_bytes(rows: &[NodeRow], cycle: u32) -> Vec<u8> {
    use lance_graph_contract::soa_envelope::SoaEnvelope;
    NodeRowPacket::new(rows, cycle).as_le_bytes().to_vec()
}

fn mv(owner: MailboxId) -> KanbanMove {
    KanbanMove {
        mailbox: owner,
        from: KanbanColumn::Planning,
        to: KanbanColumn::CognitiveWork,
        witness_chain_position: 1,
        exec: ExecTarget::Elixir,
    }
}

fn cast(cycle: u64, sp: u64, owner: MailboxId, row: u64, payload: Vec<u8>) -> SweepSlot {
    SweepSlot {
        cycle: CycleId(cycle),
        stream_position: sp,
        owner,
        row,
        paired_move: Some(mv(owner)),
        payload,
    }
}

/// Write one cycle per (row, guid, fill) and return the writer's head.
async fn write_cycles(
    w: &mut LanceCycleWriter,
    units: &[(u64, NodeGuid, u8)],
) -> Vec<(u64, Vec<u8>, u64)> {
    let mut out = Vec::new();
    let mut base = w.head();
    for (i, (row, g, fill)) in units.iter().enumerate() {
        let cycle = (i + 1) as u64;
        let bytes = packet_bytes(&[row_of(*g, *fill)], cycle as u32);
        assert_eq!(bytes.len(), EPISODIC_WITNESS_BYTES);
        let outcome = persist_cycle(
            w,
            CycleFrame::new(CycleId(cycle), base),
            vec![cast(cycle, 0, 7, *row, bytes.clone())],
        )
        .await
        .unwrap();
        match outcome {
            CommitOutcome::Committed { version, .. } => {
                assert_eq!(
                    version,
                    DatasetVersion(base.0 + 1),
                    "cycle {cycle} minted more than one version"
                );
                base = version;
                out.push((cycle, bytes, *row));
            }
            other => panic!("cycle {cycle} did not commit: {other:?}"),
        }
    }
    out
}

/// RAIL V3 (the live rail) — bytes survive, and the V3 address decodes as V3.
#[tokio::test]
async fn a_v3_address_survives_two_cycles_and_a_reopen_byte_for_byte() {
    let c = NodeGuid::CLASSID_OSINT_V3;
    assert_eq!(
        classid_read_mode(c).tail_variant,
        TailVariant::V3,
        "this rail is only meaningful if the classid actually registers V3"
    );

    let g1 = mint(c, 0x00AA, 0x1234, 0x5678);
    let g2 = mint(c, 0x00BB, 0x4321, 0x8765);
    assert_ne!(
        g1.as_bytes(),
        g2.as_bytes(),
        "leaf must discriminate on the V3 rail — if this fails the V1 fallback ran"
    );

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cycles.lance");
    let p = path.to_str().unwrap();

    let mut w = LanceCycleWriter::open(p).await.unwrap();
    let written = write_cycles(&mut w, &[(100, g1, 0xA1), (200, g2, 0xB2)]).await;
    assert_eq!(w.head(), DatasetVersion(2));

    drop(w);
    let w2 = LanceCycleWriter::open(p).await.unwrap();
    assert_eq!(w2.head(), DatasetVersion(2), "head did not survive restart");

    let expect = [g1, g2];
    for (i, (cycle, submitted, row)) in written.iter().enumerate() {
        let img = w2.scan_image(CycleId(*cycle)).await.unwrap();
        let back = img
            .get(row)
            .unwrap_or_else(|| panic!("cycle {cycle} row {row} missing after reopen"));

        // (1) raw bytes
        assert_eq!(back, submitted, "cycle {cycle} payload bytes changed");
        assert_eq!(
            &back[0..16],
            expect[i].as_bytes(),
            "cycle {cycle} payload[0..16] is not the submitted NodeGuid"
        );
        // the value slab travelled too, so (1) is not a degenerate zero-match
        assert_eq!(back[32], if i == 0 { 0xA1 } else { 0xB2 });
    }

    // (2) V3 semantic decode — the V3-aware accessors, never the V1 ones.
    assert_eq!(g1.leaf(), 0x00AA);
    assert_eq!(g1.family_v2(), 0x1234);
    assert_eq!(g1.identity_v2(), 0x5678);
    assert_eq!(g2.leaf(), 0x00BB);
    assert_eq!(g2.family_v2(), 0x4321);
    assert_eq!(g2.identity_v2(), 0x8765);
    assert_ne!(g1.to_hex_v2(), g2.to_hex_v2(), "V3 rail addresses differ");
}

/// (3) The selection claim: the V1 accessors are WRONG on a V3 key, and wrong
/// in the specific way that looks like success — many distinct addresses
/// collapsing onto one apparent identity while every byte round-trips.
#[test]
fn reading_a_v3_key_through_the_v1_accessor_collapses_distinct_addresses() {
    let c = NodeGuid::CLASSID_OSINT_V3;
    assert_eq!(classid_read_mode(c).tail_variant, TailVariant::V3);

    // 256 genuinely distinct V3 addresses, varying only the leaf tier.
    let keys: Vec<NodeGuid> = (0..256u16).map(|l| mint(c, l, 0x1234, 0x5678)).collect();

    let distinct_bytes: std::collections::HashSet<[u8; 16]> =
        keys.iter().map(|k| *k.as_bytes()).collect();
    assert_eq!(distinct_bytes.len(), 256, "the 256 addresses are distinct");

    // The V3-aware full-tier decode keeps all 256 apart. `local_key_v2` alone
    // does NOT and must not: leaf is an HHTL routing tier (bytes 10..12), part
    // of the addressing PREFIX, deliberately outside the basin-local key
    // (bytes 12..16). Both facts are pinned, because picking the wrong one of
    // the two V3 accessors is the same class of error as picking the V1 one.
    let v3_view: std::collections::HashSet<String> = keys.iter().map(NodeGuid::to_hex_v2).collect();
    assert_eq!(
        v3_view.len(),
        256,
        "the V3-aware full-tier decode must preserve all 256"
    );
    let v3_basin: std::collections::HashSet<u32> =
        keys.iter().map(NodeGuid::local_key_v2).collect();
    assert_eq!(
        v3_basin.len(),
        1,
        "leaf is a routing tier, not part of the basin-local key — these 256 \
         addresses share one basin by construction"
    );

    let v1_view: std::collections::HashSet<u32> = keys.iter().map(NodeGuid::identity).collect();
    assert_eq!(
        v1_view.len(),
        1,
        "PINNED FALSE-GREEN: the V1 identity() accessor reads bytes 13..16 of a \
         V3 key, so 256 distinct addresses present as ONE. If this assertion \
         ever fails, the tail layout moved and every V1-accessor call site on \
         V3 data must be re-audited — do not simply update the number."
    );
}

/// RAIL V1 (legacy control) — a pre-flip class still decodes as V1. No new unit
/// is minted this way; this only proves old rows stay readable.
#[tokio::test]
async fn a_legacy_v1_address_still_round_trips_and_decodes_as_v1() {
    let c = NodeGuid::CLASSID_OSINT; // pre-V3 exemplar
    assert_eq!(
        classid_read_mode(c).tail_variant,
        TailVariant::V1,
        "this control is only meaningful if the classid still registers V1"
    );

    let g = mint(c, 0, 0x00AB_CDEF, 0x0012_3456);
    assert_eq!(g.family(), 0x00AB_CDEF);
    assert_eq!(g.identity(), 0x0012_3456);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cycles.lance");
    let p = path.to_str().unwrap();

    let mut w = LanceCycleWriter::open(p).await.unwrap();
    let written = write_cycles(&mut w, &[(300, g, 0xC3)]).await;
    drop(w);

    let w2 = LanceCycleWriter::open(p).await.unwrap();
    let (cycle, submitted, row) = &written[0];
    let img = w2.scan_image(CycleId(*cycle)).await.unwrap();
    let back = img.get(row).expect("legacy row missing after reopen");

    assert_eq!(back, submitted, "V1 payload bytes changed");
    assert_eq!(&back[0..16], g.as_bytes(), "V1 payload[0..16] != G");
}
