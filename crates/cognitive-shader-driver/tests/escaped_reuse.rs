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

/// D-NFR-1 amendment, mutation path 2 of 3: the columns are PUBLIC FIELDS, so
/// a write needs neither `write_row` nor a setter. A generation bumped inside
/// those methods would not see this one. FAILS IF a direct field write moved
/// a stamp, or did not reach the predicate input.
#[test]
fn a_direct_public_field_write_bypasses_every_stamp() {
    let mut mb = mailbox();
    let kept = mask_eq(&mb, A);
    let before = stamps(&mb);
    mb.entity_type[1] = A; // public field, no method
    assert_eq!(stamps(&mb), before, "a field write moved a stamp");
    let fresh = mask_eq(&mb, A);
    assert_ne!(kept, fresh);
    assert_eq!(fresh, kept | 0b10, "expected exactly row 1 to join");
}

/// The stamps are themselves public fields, so even a reader that compared
/// them could be shown values that no longer describe the data. Here only
/// owner METHODS are used: `reset_row` returns the row's stamp to the
/// never-written sentinel and a same-cycle `write_row` puts back the value the
/// reader recorded, with different data underneath. FAILS IF the stamp pair
/// differs after the round trip (it would then witness the change).
#[test]
fn reset_then_same_cycle_write_restores_the_stamps_over_new_data() {
    let mut mb = mailbox();
    let kept = mask_eq(&mb, A);
    let before = stamps(&mb);
    mb.reset_row(1);
    assert_eq!(
        mb.last_write_cycle_at(1),
        u32::MAX,
        "reset did not clear the stamp"
    );
    let cell = WriteCell {
        entity_type: Some(A),
        ..Default::default()
    };
    assert_eq!(
        mb.write_row(1, mb.current_cycle(), &cell),
        WriteOutcome::Accepted
    );
    assert_eq!(stamps(&mb), before, "the round trip left a trace");
    assert_ne!(kept, mask_eq(&mb, A));
}

/// Two owners, one numeric stamp. Mailboxes 1 and 2 are both at cycle 1 with
/// every row written in cycle 1, so `(current_cycle, last_write_cycle)` is
/// identical; their `entity_type` columns differ. A reuse key without the
/// owner's identity would alias them. Expected masks are computed from the
/// fixture rule, not from the code under test. FAILS IF the stamps differ (the
/// premise is false) or the two answers coincide (nothing to alias).
#[test]
fn equal_stamps_on_two_owners_do_not_identify_one_answer() {
    let one = mailbox();
    let mut two = MailboxSoA::<ROWS>::new(2, 0, 0.5);
    two.set_populated(ROWS);
    two.tick();
    let c = two.current_cycle();
    for r in 0..ROWS {
        let cell = WriteCell {
            entity_type: Some(if r % 2 == 0 { A } else { B }),
            ..Default::default()
        };
        assert_eq!(two.write_row(r, c, &cell), WriteOutcome::Accepted);
    }
    assert_eq!(stamps(&one), stamps(&two), "premise: equal numeric stamps");
    assert_ne!(one.mailbox_id(), two.mailbox_id());
    let expect = |m: usize| (0..ROWS).fold(0u64, |a, r| a | (u64::from(r % m == 0) << r));
    assert_eq!(mask_eq(&one, A), expect(3));
    assert_eq!(mask_eq(&two, A), expect(2));
    assert_ne!(mask_eq(&one, A), mask_eq(&two, A));
}
