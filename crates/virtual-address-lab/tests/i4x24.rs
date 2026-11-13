//! 24 x i4 = 12 bytes, read in place from two owners (plan section 9.3):
//! a `[NodeRow]` slab (ABI512) and a `MailboxSoA` style lane.

use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::canonical_node::{
    NodeRow, NodeRowPacket, ValueTenant, NODE_ROW_STRIDE, VALUE_TENANTS,
};
use lance_graph_contract::causal_witness::CausalWitnessFacet;
use lance_graph_contract::soa_envelope::SoaEnvelope;
use lance_graph_contract::soa_view::StyleLane;
use lance_graph_mask_risc::{
    execute_into, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program, Scratch,
    StridedRef, Terminal, Value,
};

const LOCI: usize = 24;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The oracle table: row `r`, locus `s`, a value in `-8..=7`.
fn locus(r: usize, s: usize) -> i8 {
    (mix((r * 31 + s) as u64) % 16) as i8 - 8
}

fn table(rows: usize) -> Vec<[i8; LOCI]> {
    (0..rows)
        .map(|r| core::array::from_fn(|s| locus(r, s)))
        .collect()
}

/// The test's own packer: two nibbles per byte, even slot in the low nibble.
fn pack(values: &[i8; LOCI]) -> [u8; 12] {
    core::array::from_fn(|k| {
        let lo = (values[2 * k] as u8) & 0x0F;
        let hi = (values[2 * k + 1] as u8) & 0x0F;
        lo | (hi << 4)
    })
}

/// Rows where locus `slot` equals `value`, from the table alone.
fn oracle_count(table: &[[i8; LOCI]], slot: usize, value: i8) -> usize {
    table.iter().filter(|row| row[slot] == value).count()
}

/// Count, through mask-RISC, the records whose locus `slot` equals `value`.
/// One byte of the 12 is cared for, and of it one nibble.
fn isa_count(view: StridedRef<'_>, slot: usize, value: i8, care_nibble: u8) -> usize {
    let mut pattern = [0u8; 12];
    let mut care = [0u8; 12];
    let nibble = (value as u8) & 0x0F;
    pattern[slot / 2] = if slot & 1 == 0 { nibble } else { nibble << 4 };
    care[slot / 2] = care_nibble;
    let lanes = [LaneRef::Strided(view)];
    let planes = Planes {
        n_rows: view.records,
        masks: &[],
        lanes: &lanes,
    };
    let program = Program::new(
        vec![MaskOp::Pred {
            pred: Pred::MatchFacetStrided {
                lane: 0,
                pattern,
                care,
            },
            under: None,
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    let mut scratch = Scratch::for_program(&program, view.records).unwrap();
    match execute_into(&program, &planes, &Foreign::NONE, &mut scratch, Out::None) {
        Ok(Value::Count(n)) => n,
        other => panic!("unexpected {other:?}"),
    }
}

fn care_of(slot: usize) -> u8 {
    if slot & 1 == 0 {
        0x0F
    } else {
        0xF0
    }
}

/// The slots and values checked: a low-nibble and a high-nibble slot, a
/// negative, a zero and a positive value.
const CASES: [(usize, i8); 4] = [(4, -3), (7, -8), (10, 0), (23, 5)];

/// FAILS IF: the payload is not where the carve says, a nibble is read from
/// the wrong half or with the wrong sign, the read is a copy, or the in-place
/// count differs from the table's.
#[test]
fn twenty_four_i4_are_read_in_place_from_a_node_row_slab() {
    const ROWS: usize = 300;
    let table = table(ROWS);
    let tenant = &VALUE_TENANTS[ValueTenant::CausalWitness as usize];
    let unit = tenant.row_offset as usize;
    let payload = unit + 4;
    // The 4 + 12 unit starts at facet 12, byte 12; its payload is facet 13.
    assert_eq!((unit / 16, unit % 16), (12, 12));
    assert_eq!((payload / 16, payload % 16), (13, 0));

    let mut rows = vec![
        NodeRow {
            key: lance_graph_contract::canonical_node::NodeGuid::local(0),
            edges: Default::default(),
            value: [0u8; 480],
        };
        ROWS
    ];
    for (row, values) in rows.iter_mut().zip(&table) {
        row.value[payload - 32..payload - 32 + 12].copy_from_slice(&pack(values));
    }

    let packet = NodeRowPacket::new(&rows, 1);
    let bytes = packet.as_le_bytes();
    assert_eq!(bytes.len(), ROWS * NODE_ROW_STRIDE);
    assert!(core::ptr::eq(bytes.as_ptr(), rows.as_ptr().cast::<u8>()));

    // Every nibble, read through the contract's borrowed lens over the slab.
    for (r, values) in table.iter().enumerate() {
        let at = r * NODE_ROW_STRIDE + payload;
        let register: &[u8; 12] = bytes[at..at + 12].try_into().unwrap();
        let facet = CausalWitnessFacet::from_register_ref(register);
        assert!(core::ptr::eq(
            (facet as *const CausalWitnessFacet).cast::<u8>(),
            bytes[at..].as_ptr()
        ));
        for (s, want) in values.iter().enumerate() {
            assert_eq!(facet.get(s), *want, "row {r} locus {s}");
        }
    }

    let view = StridedRef {
        bytes,
        first_offset: payload,
        stride: NODE_ROW_STRIDE,
        records: ROWS,
    };
    for (slot, value) in CASES {
        let want = oracle_count(&table, slot, value);
        assert!(
            want > 0 && want < ROWS,
            "slot {slot}: uninformative count {want}"
        );
        assert_eq!(
            isa_count(view, slot, value, care_of(slot)),
            want,
            "slot {slot}"
        );
        // Reading the whole byte as one value is a different question.
        assert_ne!(
            isa_count(view, slot, value, 0xFF),
            want,
            "slot {slot} byte care"
        );
    }
}

/// FAILS IF: the mailbox's own lane is copied to be read, or the count over
/// it differs from the table's.
#[test]
fn twenty_four_i4_are_read_in_place_from_a_lane_the_mailbox_owns() {
    const N: usize = 64;
    let table = table(N);
    let mut mailbox: Box<MailboxSoA<N>> = Box::new(MailboxSoA::new(7, 3, 1.0));
    mailbox.set_populated(N);
    for (row, values) in table.iter().enumerate() {
        mailbox.set_style_lane(row, StyleLane::Frozen, pack(values));
    }

    let lane: &[u8] = mailbox.frozen_style.as_flattened();
    assert_eq!(lane.len(), N * 12);
    assert!(core::ptr::eq(
        lane.as_ptr(),
        mailbox.frozen_style.as_ptr().cast::<u8>()
    ));
    let view = StridedRef {
        bytes: lane,
        first_offset: 0,
        stride: 12,
        records: N,
    };
    let mut informative = 0;
    for (slot, value) in CASES {
        let want = oracle_count(&table, slot, value);
        informative += usize::from(want > 0);
        assert_eq!(
            isa_count(view, slot, value, care_of(slot)),
            want,
            "slot {slot}"
        );
    }
    assert!(informative >= 3, "the 64-row table must hit most cases");

    // The owner writes; the next borrow sees it.
    let mut changed = table[9];
    changed[4] = 7;
    let before = isa_count(view, 4, 7, 0x0F);
    mailbox.set_style_lane(9, StyleLane::Frozen, pack(&changed));
    let lane: &[u8] = mailbox.frozen_style.as_flattened();
    let view = StridedRef {
        bytes: lane,
        first_offset: 0,
        stride: 12,
        records: N,
    };
    assert_eq!(
        isa_count(view, 4, 7, 0x0F),
        before + usize::from(table[9][4] != 7)
    );
}
