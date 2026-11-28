//! Demo of the resident thought spaces: 30 slot-placed thoughts over 256 rows
//! in one table, and the bytes they store. The spaces (`ThoughtTables`,
//! `RowDelta`, `Placement`) and their tests moved to
//! `lance_graph_thought_git::spaces` (D-TGB-7 slice S2); this example only
//! runs them. Plan: `.claude/plans/ce64-hypothesis-layers-v1.md`.
//!
//! Run: `cargo run --example thought_layers_probe`.

use lance_graph_contract::alpha::AlphaAllocation;
use lance_graph_contract::canonical_node::{EdgeBlock, NodeGuid, NodeRow, TailVariant};
use lance_graph_thought_git::spaces::{row_bytes, Placement, ThoughtId, ThoughtTables, SLOT};

fn base(n: u32) -> Vec<NodeRow> {
    (0..n)
        .map(|i| {
            let mut value = [0u8; 480];
            for (k, b) in value.iter_mut().enumerate() {
                *b = (i as u8).wrapping_mul(31).wrapping_add(k as u8);
            }
            NodeRow {
                key: NodeGuid::mint_for(TailVariant::V3, 0x0C0C_0000, 1, 2, 3, 4, 0x22, i + 1),
                edges: EdgeBlock::default(),
                value,
            }
        })
        .collect()
}

fn main() {
    let rows = base(256);
    let alloc = AlphaAllocation::over(&rows);
    let mut tt = ThoughtTables::over(&alloc, 1);
    let shared = tt.open_table().expect("budget");
    // 30 thoughts, each changing one slot of every row, all in one table.
    for s in 2..32u8 {
        tt.place(ThoughtId(s), Placement::Slot(shared, s))
            .expect("free slot");
    }
    for s in 2..32u8 {
        for r in &rows {
            let mut b = row_bytes(r);
            b[usize::from(s) * SLOT] ^= 0xFF;
            tt.write(ThoughtId(s), r.key, 1, &b).expect("own slot");
        }
    }
    println!(
        "thought_layers_probe: 30 thoughts × 256 rows in {} table(s), {} bytes stored (full rows would be {} bytes)",
        tt.tables_used(),
        tt.stored_bytes(),
        30 * 256 * 512
    );
}
