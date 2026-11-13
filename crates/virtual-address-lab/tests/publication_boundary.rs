//! Plan section 18.4: the mailbox's cycle counter is not a source revision.
//! Each case changes owner bytes and shows which stamp did NOT move. These
//! tests pin the absence of a publication authority (A1 BLOCKED); they are to
//! be rewritten when the owner gains one.

use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::soa_view::MailboxSoaView;
use lance_graph_dt64::dr::{FrameRefusal, FrameRevisions, FrameStore};

const N: usize = 64;
type Owner = MailboxSoA<N>;
const ROW: usize = 5;
const NEVER: u32 = u32::MAX;

fn owner() -> Box<Owner> {
    let mut m: Box<Owner> = Box::new(MailboxSoA::new(7, 3, 1.0));
    m.set_populated(N);
    m
}

fn stamps(m: &Owner) -> (u32, u32) {
    (m.current_cycle(), m.last_write_cycle_at(ROW))
}

/// FAILS IF: a setter moves a stamp (then the cycle would track that write).
#[test]
fn set_meta_without_tick_moves_no_stamp() {
    let mut m = owner();
    let before = (stamps(&m), m.meta_raw()[ROW]);
    m.set_meta(ROW, MetaWord(0xDEAD_BEEF));
    assert_ne!(m.meta_raw()[ROW], before.1, "the bytes changed");
    assert_eq!(stamps(&m), before.0);
    assert_eq!(before.0, (0, NEVER));
}

/// FAILS IF: assignment to a public column is impossible or stamped.
#[test]
fn a_public_column_write_moves_no_stamp() {
    let mut m = owner();
    let before = stamps(&m);
    m.meta[ROW] = MetaWord(0x0BAD_F00D);
    m.energy[ROW] = 2.5;
    assert_eq!(m.meta_raw()[ROW], 0x0BAD_F00D);
    assert_eq!(stamps(&m), before);
}

/// FAILS IF: two writes in one cycle are distinguishable by the row stamp.
#[test]
fn two_writes_in_one_cycle_share_a_stamp() {
    let mut m = owner();
    m.tick();
    let cell = |v: u32| WriteCell {
        meta: Some(MetaWord(v)),
        ..WriteCell::default()
    };
    assert_eq!(m.write_row(ROW, 1, &cell(11)), WriteOutcome::Accepted);
    let first = (stamps(&m), m.meta_raw()[ROW]);
    assert_eq!(m.write_row(ROW, 1, &cell(22)), WriteOutcome::Accepted);
    let second = (stamps(&m), m.meta_raw()[ROW]);
    assert_eq!(first.0, (1, 1));
    assert_eq!(second.0, first.0, "one stamp");
    assert_ne!(second.1, first.1, "two states");
}

/// FAILS IF: a write at cycle `u32::MAX` is told apart from "never written",
/// or the counter does not wrap.
#[test]
fn the_cycle_wraps_onto_the_never_written_sentinel() {
    let mut m = owner();
    m.current_cycle = u32::MAX;
    let cell = WriteCell {
        meta: Some(MetaWord(33)),
        ..WriteCell::default()
    };
    assert_eq!(m.write_row(ROW, u32::MAX, &cell), WriteOutcome::Accepted);
    assert_eq!(m.meta_raw()[ROW], 33);
    assert_eq!(m.last_write_cycle_at(ROW), NEVER);
    assert_eq!(m.last_write_cycle_at(ROW + 1), NEVER, "an unwritten row");
    m.tick();
    assert_eq!(m.current_cycle(), 0);
}

/// FAILS IF: a frame of another store is accepted at equal revisions, or a
/// store refuses its own current frame.
#[test]
fn a_foreign_store_at_equal_revisions_is_refused() {
    let revs = FrameRevisions {
        source: 7,
        relation_map: 0,
        coordinates: 0,
    };
    let (a, b) = (FrameStore::new(revs), FrameStore::new(revs));
    assert_eq!(a.revisions(), b.revisions());
    assert_eq!(a.check(&a.seal()), Ok(()));
    assert_eq!(b.check(&a.seal()), Err(FrameRefusal::ForeignFrame));
}
