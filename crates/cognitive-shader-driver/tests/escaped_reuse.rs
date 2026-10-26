//! D-NFR-1 ownership correction: what the mailbox's own stamps can and cannot
//! witness once a derived mask has ESCAPED the borrow it was computed under.
//!
//! Inside a borrow nothing needs witnessing: the owner's `&mut` cannot start
//! while a view is live (`soa-view-probe`'s `compile_fail` examples, #1755).
//! A `Vec<u64>` mask computed from that view is an owned value with no
//! lifetime, so it outlives the borrow. The question here is whether anything
//! the owner already records (`current_cycle`, `last_write_cycle`) tells a
//! later reader that the mask is stale. Measured answer: only for a write in a
//! LATER cycle through `write_row`. A same-cycle `write_row` and every raw
//! setter mutate without moving either stamp.

use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome};
use lance_graph_contract::soa_view::MailboxSoaView;

const ROWS: usize = 64;
const A: u16 = 7;
const B: u16 = 9;

/// Selection mask `entity_type == t` over the view. Scalar on purpose: this is
/// the test oracle, not a kernel.
fn mask_eq(v: &impl MailboxSoaView, t: u16) -> u64 {
    v.entity_type()
        .iter()
        .take(ROWS)
        .enumerate()
        .fold(0u64, |m, (i, &e)| m | (u64::from(e == t) << i))
}

/// Every stamp the owner keeps that a reader could hold beside a mask.
fn stamps(mb: &MailboxSoA<ROWS>) -> (u32, Vec<u32>) {
    (
        mb.current_cycle(),
        (0..ROWS).map(|r| mb.last_write_cycle_at(r)).collect(),
    )
}

fn mailbox() -> MailboxSoA<ROWS> {
    let mut mb = MailboxSoA::<ROWS>::new(1, 0, 0.5);
    mb.set_populated(ROWS);
    mb.tick();
    let c = mb.current_cycle();
    for r in 0..ROWS {
        let cell = WriteCell {
            entity_type: Some(if r % 3 == 0 { A } else { B }),
            ..Default::default()
        };
        assert_eq!(mb.write_row(r, c, &cell), WriteOutcome::Accepted);
    }
    mb
}

/// FAILS IF: a raw setter moves a stamp. It does not, so `(cycle,
/// last_write_cycle)` held beside an escaped mask cannot detect the change,
/// while a fresh recompute does.
#[test]
fn a_raw_setter_changes_the_answer_without_moving_any_stamp() {
    let mut mb = mailbox();
    let kept = mask_eq(&mb, A); // borrow ends here; the mask escapes
    let before = stamps(&mb);
    mb.set_entity_type(1, A); // row 1 was B
    assert_eq!(stamps(&mb), before, "a setter moved a stamp");
    let fresh = mask_eq(&mb, A);
    assert_ne!(
        kept, fresh,
        "the mutation did not reach the predicate input"
    );
    assert_eq!(fresh, kept | 0b10);
}

/// FAILS IF: a same-cycle `write_row` moves a stamp. It does not: the gate
/// accepts it and re-stamps the row with the cycle it already carried.
#[test]
fn a_same_cycle_write_row_changes_the_answer_without_moving_any_stamp() {
    let mut mb = mailbox();
    let kept = mask_eq(&mb, A);
    let before = stamps(&mb);
    let cell = WriteCell {
        entity_type: Some(A),
        ..Default::default()
    };
    assert_eq!(
        mb.write_row(1, mb.current_cycle(), &cell),
        WriteOutcome::Accepted
    );
    assert_eq!(stamps(&mb), before, "a same-cycle write moved a stamp");
    assert_ne!(kept, mask_eq(&mb, A));
}

/// The twin that must detect: a write in a LATER cycle moves `current_cycle`
/// and the row's `last_write_cycle`. FAILS IF the gate stops distinguishing
/// cycles, i.e. if the two tests above are vacuous because no stamp ever
/// moves.
#[test]
fn a_later_cycle_write_moves_both_stamps() {
    let mut mb = mailbox();
    let kept = mask_eq(&mb, A);
    let before = stamps(&mb);
    mb.tick();
    let cell = WriteCell {
        entity_type: Some(A),
        ..Default::default()
    };
    assert_eq!(
        mb.write_row(1, mb.current_cycle(), &cell),
        WriteOutcome::Accepted
    );
    let after = stamps(&mb);
    assert_ne!(after.0, before.0, "current_cycle did not move");
    assert_ne!(after.1[1], before.1[1], "row 1's write stamp did not move");
    assert_eq!(after.1[2], before.1[2], "an unwritten row's stamp moved");
    assert_ne!(kept, mask_eq(&mb, A));
}
