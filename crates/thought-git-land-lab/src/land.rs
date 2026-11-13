//! Lab only: land the thoughts of one supervisor cycle through the planner's
//! [`WalSink`] contract and read them back from the durable image.
//!
//! The flow is: the probe's from-scratch oracle ([`next_version`]) merges the
//! cycle's thoughts into the next [`Version`]; [`plan_cycle`] turns the rows
//! that changed into one [`SweepSlot`] cast each; [`land_cycle`] hands all of
//! them to exactly one [`persist_cycle`] call (one batch, one version).
//!
//! **The write is fire and forget; nothing is read back to answer a read.**
//! [`land_cycle`] returns the resident [`Version`] it computed before the
//! write. That resident version is what the live path reads, under the
//! probe's contract ([`Version::read`]: given, else landed, else absent). The
//! store is not consulted for it.
//!
//! [`replay_landed`] is the recovery path only: it folds the durable images
//! back into a [`Version`] once, at startup or after a crash, to rebuild the
//! resident version. It is not a read path, and a caller that uses it to
//! answer a read after its own commit has left the write path's contract
//! (`WalSink`'s reads are documented as recovery and audit, never the normal
//! path).
//!
//! **The row encoding here is a LAB encoding.** It is not a decision about
//! where a thought's values live in the canonical 512-byte row. It only gives
//! the lab a 512-byte payload per row in which presence and value are kept
//! apart, so an observed zero and a lane never written stay different.

use std::collections::{BTreeMap, BTreeSet};

use lance_graph_planner::persist_sink::{
    CommitOutcome, CycleFrame, CycleId, MailboxId, PersistError, SweepSlot, WalSink, persist_cycle,
};
use thought_git_probe::fixtures::{Cell, Thought};
use thought_git_probe::oracle::{Givens, Version, next_version};

/// Bytes in one lab row payload.
pub const ROW_BYTES: usize = 512;

/// Lanes one lab row can hold: (512 - 24) / 8.
pub const MAX_LANES: u8 = 61;

/// First byte of the presence bitmap.
const BITMAP_AT: usize = 16;
/// First byte of lane 0.
const VALUES_AT: usize = 24;

/// Why a lab row could not be encoded or decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowError {
    /// The payload is not [`ROW_BYTES`] long.
    Length(usize),
    /// Encode: a lane at or above [`MAX_LANES`]. Decode: a presence bit at or
    /// above [`MAX_LANES`] is set.
    LaneOutOfRange(u8),
    /// Decode: bytes 0..16 (the stamp region) are not all zero.
    StampNotZero,
    /// Decode: a non-zero value sits in a lane whose presence bit is off.
    ValueWithoutPresence(u8),
}

/// Encode the lanes of one row as a lab row payload.
///
/// Bytes 0..16 are zero (the stamp region, untouched here); bytes 16..24 are
/// a presence bitmap, `u64` little-endian, bit `lane`; the value of a lane is
/// at `24 + 8 * lane`, `u64` little-endian. A lane whose bit is off holds zero
/// bytes. Presence is separate from the value, so an observed zero (bit on,
/// value 0) and a lane never written (bit off) are different rows.
///
/// # Errors
///
/// [`RowError::LaneOutOfRange`] when a lane is `>= MAX_LANES`.
pub fn encode_row(lanes: &BTreeMap<u8, u64>) -> Result<Vec<u8>, RowError> {
    let mut payload = vec![0u8; ROW_BYTES];
    let mut bitmap = 0u64;
    for (&lane, &value) in lanes {
        if lane >= MAX_LANES {
            return Err(RowError::LaneOutOfRange(lane));
        }
        bitmap |= 1u64 << lane;
        let at = VALUES_AT + 8 * usize::from(lane);
        payload[at..at + 8].copy_from_slice(&value.to_le_bytes());
    }
    payload[BITMAP_AT..BITMAP_AT + 8].copy_from_slice(&bitmap.to_le_bytes());
    Ok(payload)
}

/// Read the 8 bytes at `at` as a little-endian `u64`.
fn u64_at(payload: &[u8], at: usize) -> u64 {
    let mut word = [0u8; 8];
    word.copy_from_slice(&payload[at..at + 8]);
    u64::from_le_bytes(word)
}

/// Decode a lab row payload back into its lanes.
///
/// # Errors
///
/// [`RowError::Length`] for a payload that is not [`ROW_BYTES`] long,
/// [`RowError::StampNotZero`] for a set stamp byte,
/// [`RowError::LaneOutOfRange`] for a presence bit `>= MAX_LANES`, and
/// [`RowError::ValueWithoutPresence`] for a non-zero value under an off bit.
pub fn decode_row(payload: &[u8]) -> Result<BTreeMap<u8, u64>, RowError> {
    if payload.len() != ROW_BYTES {
        return Err(RowError::Length(payload.len()));
    }
    if payload[..BITMAP_AT].iter().any(|&b| b != 0) {
        return Err(RowError::StampNotZero);
    }
    let bitmap = u64_at(payload, BITMAP_AT);
    let high = bitmap >> MAX_LANES;
    if high != 0 {
        let bit = u32::from(MAX_LANES) + high.trailing_zeros();
        return Err(RowError::LaneOutOfRange(
            u8::try_from(bit).unwrap_or(u8::MAX),
        ));
    }
    let mut lanes = BTreeMap::new();
    for lane in 0..MAX_LANES {
        let value = u64_at(payload, VALUES_AT + 8 * usize::from(lane));
        if bitmap & (1u64 << lane) != 0 {
            lanes.insert(lane, value);
        } else if value != 0 {
            return Err(RowError::ValueWithoutPresence(lane));
        }
    }
    Ok(lanes)
}

/// The casts of one cycle and the version the oracle computes for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CyclePlan {
    /// One cast per touched row, in ascending row order.
    pub casts: Vec<SweepSlot>,
    /// The version after the cycle, from the oracle.
    pub next: Version,
}

/// Group landed cells by row.
fn rows_of(cells: &BTreeMap<Cell, u64>) -> BTreeMap<u32, BTreeMap<u8, u64>> {
    let mut rows: BTreeMap<u32, BTreeMap<u8, u64>> = BTreeMap::new();
    for (cell, &value) in cells {
        rows.entry(cell.row).or_default().insert(cell.lane, value);
    }
    rows
}

/// Plan the casts of one cycle.
///
/// `next = next_version(givens, prior, batch)`. A row is TOUCHED when the
/// landed cells of that row differ between `prior.cells` and `next.cells`.
/// There is one cast per touched row, in ascending row order, with
/// `stream_position = first_position + i`, `owner` as given,
/// `row = u64::from(cell.row)` and `paired_move = None`. The payload is
/// [`encode_row`] of ALL landed lanes of that row in `next.cells` — the full
/// row after the cycle, not only this cycle's cells, because a newer row
/// replaces an older row whole when read. Givens are never encoded: they are
/// layer 0.
///
/// # Errors
///
/// [`RowError::LaneOutOfRange`] when a touched row has a lane `>= MAX_LANES`.
pub fn plan_cycle(
    givens: &Givens,
    prior: &Version,
    cycle: CycleId,
    owner: MailboxId,
    first_position: u64,
    batch: &[Thought],
) -> Result<CyclePlan, RowError> {
    let next = next_version(givens, prior, batch);
    let before = rows_of(&prior.cells);
    let after = rows_of(&next.cells);
    let keys: BTreeSet<u32> = before.keys().chain(after.keys()).copied().collect();

    let mut casts = Vec::new();
    let mut position = first_position;
    for row in keys {
        let was = before.get(&row);
        let now = after.get(&row);
        if was == now {
            continue;
        }
        let lanes = now.cloned().unwrap_or_default();
        casts.push(SweepSlot {
            cycle,
            stream_position: position,
            owner,
            row: u64::from(row),
            paired_move: None,
            payload: encode_row(&lanes)?,
        });
        position = position.saturating_add(1);
    }
    Ok(CyclePlan { casts, next })
}

/// Why a cycle did not land.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LandError {
    /// A touched row could not be encoded.
    Row(RowError),
    /// The persistence cycle refused or failed.
    Persist(PersistError),
}

/// [`plan_cycle`], then exactly one [`persist_cycle`] call.
///
/// Returns the outcome and the resident next version (the oracle's), which is
/// what the live path reads from here on; the sink is not read. A cycle that touches no
/// row yields [`CommitOutcome::NoChange`] and never calls the sink; that is
/// `persist_cycle`'s own rule and is not special-cased here.
///
/// # Errors
///
/// [`LandError::Row`] when planning fails, [`LandError::Persist`] when the
/// persistence cycle does.
pub async fn land_cycle<S: WalSink>(
    sink: &mut S,
    frame: CycleFrame,
    owner: MailboxId,
    first_position: u64,
    givens: &Givens,
    prior: &Version,
    batch: &[Thought],
) -> Result<(CommitOutcome, Version), LandError> {
    let plan = plan_cycle(givens, prior, frame.cycle, owner, first_position, batch)
        .map_err(LandError::Row)?;
    let outcome = persist_cycle(sink, frame, plan.casts)
        .await
        .map_err(LandError::Persist)?;
    Ok((outcome, plan.next))
}

/// Where RECOVERY gets a sealed cycle's rows. `WalSink::scan_sealed` returns
/// no payload, so the durable image is a separate read. Recovery clock only:
/// the live path reads the resident version and never calls this.
pub trait ReplayImage {
    /// The coalesced image of `cycle` (row to payload), if it is sealed.
    fn image(&self, cycle: CycleId) -> Option<&BTreeMap<u64, Vec<u8>>>;
}

/// Why landed rows could not be read back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayRefusal {
    /// No sealed image exists for the cycle.
    MissingCycle(CycleId),
    /// A row payload could not be decoded.
    Row {
        /// The cycle holding the row.
        cycle: CycleId,
        /// The row key.
        row: u64,
        /// Why the payload was refused.
        error: RowError,
    },
    /// A row key does not fit `u32` (`Cell::row` is `u32`).
    RowTooLarge(u64),
}

/// Recovery only: rebuild the resident version from the durable images of
/// `cycles`, folded in the order given (a row in a later cycle replaces that
/// row whole). Run once at startup or after a crash, never to answer a read
/// after a commit: the resident version [`land_cycle`] returned is the answer.
///
/// Returns a [`Version`] with `cells` only; contradictions and divergences are
/// not landed, so they stay empty.
///
/// # Errors
///
/// [`ReplayRefusal`] for a missing cycle, an undecodable row, or a row key that
/// does not fit `u32`.
pub fn replay_landed<I: ReplayImage>(
    images: &I,
    cycles: &[CycleId],
) -> Result<Version, ReplayRefusal> {
    let mut rows: BTreeMap<u32, BTreeMap<u8, u64>> = BTreeMap::new();
    for &cycle in cycles {
        let image = images
            .image(cycle)
            .ok_or(ReplayRefusal::MissingCycle(cycle))?;
        for (&row, payload) in image {
            let key = u32::try_from(row).map_err(|_| ReplayRefusal::RowTooLarge(row))?;
            let lanes =
                decode_row(payload).map_err(|error| ReplayRefusal::Row { cycle, row, error })?;
            rows.insert(key, lanes);
        }
    }
    let mut cells = BTreeMap::new();
    for (row, lanes) in rows {
        for (lane, value) in lanes {
            cells.insert(Cell { row, lane }, value);
        }
    }
    Ok(Version {
        cells,
        contradictions: Vec::new(),
        divergences: Vec::new(),
    })
}
