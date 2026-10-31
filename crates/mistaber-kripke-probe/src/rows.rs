//! HHTL16 node rows, address-only.
//!
//! One node is one [`NodeRow`] whose `Register0` rail holds the `x` bytes and
//! whose `Register1` rail the `y` bytes of sixteen [`Morton8x8`] columns.
//! Column 0 is the node's own identity; columns 1..=15 hold the addresses of
//! other nodes (a world's parents). Code 0 (`(0, 0)`) is unbound.
//!
//! The row carries no name and no classid: the classid is the concept of the
//! [`RegisterLanes`] the population was resolved under. The row key carries
//! the node's address (V3 `identity` tier, bytes 14..16) and equals column 0;
//! classid bytes 0..4 and every other tier stay zero.

use lance_graph_contract::canonical_node::{NodeGuid, NodeRow, ReadMode, TailVariant, ValueSchema};
use lance_graph_contract::hotplug::{Activation, SlabDeclaration, SlabReading};
use lance_graph_contract::morton8x8::Morton8x8;
use lance_graph_contract::register128::{Register128, RegisterLanes, RegisterRails};
use lance_graph_contract::soa_envelope::ENVELOPE_LAYOUT_VERSION;

use crate::Refusal;

/// Columns per row.
pub const COLUMNS: usize = 16;
/// Parents one world row can hold (column 0 is the identity).
pub const MAX_PARENTS: usize = COLUMNS - 1;
/// The empty column.
pub const UNBOUND: Morton8x8 = Morton8x8::from_xy(0, 0);

/// Lab stand-ins (same 0x7F range as xlog-relcomp-probe/tests/dto_binding.rs);
/// not minted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Concepts {
    pub world: u16,
    pub proposition: u16,
    pub rule: u16,
}

impl Concepts {
    pub const LAB: Concepts = Concepts {
        world: 0x7F11,
        proposition: 0x7F12,
        rule: 0x7F13,
    };
}

/// One bound population: its lanes and its rows.
pub struct Population {
    pub lanes: RegisterLanes,
    pub rows: Vec<NodeRow>,
}

/// Address rule: population-local ordinal `o` (first appearance) maps to
/// Morton code `o + 1`. `from_code` is a bijection on `u16` and `o + 1 >= 1`,
/// so two ordinals never collide and no identity is ever [`UNBOUND`]. The
/// address is meaningful only under its population's concept.
pub fn address(ordinal: u16) -> Morton8x8 {
    Morton8x8::from_code(ordinal + 1)
}

const SLAB: SlabDeclaration = SlabDeclaration {
    reading: SlabReading::Register128,
    value_schema: ValueSchema::Full,
    layout_version: ENVELOPE_LAYOUT_VERSION,
};

/// One activation declaring all three concepts, then resolve and bind two
/// register rails per concept. Order: `[world, proposition, rule]`.
pub fn bind_populations(c: &Concepts) -> Result<[RegisterLanes; 3], Refusal> {
    let concepts = [c.world, c.proposition, c.rule];
    let activation = Activation::new(
        Vec::new(),
        Vec::new(),
        concepts
            .iter()
            .map(|&k| (k, ReadMode::PLUG_AND_PLAY_V3))
            .collect(),
    );
    let mut out = Vec::with_capacity(3);
    for k in concepts {
        let lanes = activation
            .resolve_for_context(k, Some(&SLAB))
            .and_then(|r| r.bind_register128(RegisterRails::Two))
            .map_err(Refusal::Register)?;
        out.push(lanes);
    }
    Ok([out[0], out[1], out[2]])
}

/// A V3 row whose key carries `address` in the `identity` tier, minted through
/// `mint_for`; classid and every other tier are zero. Edges and value are
/// empty. The address equals column 0 of the row.
pub fn node_row(address: Morton8x8) -> NodeRow {
    NodeRow {
        key: NodeGuid::mint_for(
            TailVariant::V3,
            NodeGuid::CLASSID_DEFAULT,
            0,
            0,
            0,
            0,
            0,
            u32::from(address.code()),
        ),
        edges: Default::default(),
        value: [0; 480],
    }
}

/// The address the row key carries.
pub fn key_address(row: &NodeRow) -> Morton8x8 {
    Morton8x8::from_code(row.key.identity_v2())
}

/// `R0[k] = cols[k].x()`, `R1[k] = cols[k].y()` for `k < cols.len()`; the
/// remaining columns stay `(0, 0)`. Callers refuse more than [`COLUMNS`]
/// columns first.
pub fn write_columns(lanes: &RegisterLanes, row: &mut NodeRow, cols: &[Morton8x8]) {
    assert!(cols.len() <= COLUMNS, "a row holds {COLUMNS} columns");
    let (mut xs, mut ys) = ([0u8; COLUMNS], [0u8; COLUMNS]);
    for (k, a) in cols.iter().enumerate() {
        xs[k] = a.x();
        ys[k] = a.y();
    }
    assert!(lanes.set(row, 0, Register128(xs)), "rail 0 granted");
    assert!(lanes.set(row, 1, Register128(ys)), "rail 1 granted");
}

/// Column `k` is `Morton8x8::from_xy(R0[k], R1[k])`.
pub fn columns(lanes: &RegisterLanes, row: &NodeRow) -> [Morton8x8; COLUMNS] {
    let xs = lanes.get(row, 0).expect("rail 0 granted").0;
    let ys = lanes.get(row, 1).expect("rail 1 granted").0;
    std::array::from_fn(|k| Morton8x8::from_xy(xs[k], ys[k]))
}

/// Column 0: the node's own address.
pub fn identity(lanes: &RegisterLanes, row: &NodeRow) -> Morton8x8 {
    columns(lanes, row)[0]
}
