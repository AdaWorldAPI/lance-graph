//! DR's versioned global → local table ([`MapTable`]) made durable through the
//! fire-and-forget cycle writer (D-DT64-7 O5 / D10', D-DR64-WAL-1).
//!
//! The owner of a DR map table never waits for storage:
//!
//! ```text
//! owner:   advance ── claim v ── casts ──try_submit(v, casts)──▶ returns v
//! writer:                                   │ one background task owns the sink
//!                                           ▼
//!                       seal cycle v (LanceCycleWriter: one Lance version)
//! owner:   settle ── ticket Ok ── advance_at(v)        ticket Err ── v is a gap
//! ```
//!
//! - **The WAL is `LanceCycleWriter` behind `CycleWriter`.** This crate writes
//!   no dataset itself and keeps no confirmation ledger, replay queue or retry
//!   loop. Measured at DR64 sizes it commits an advance in 2.2–5.7 ms p50 and
//!   keeps one Lance version per advance (`.claude/plans/wal-shootout-v1.md` §8).
//! - **One advance is one cycle**, with cycle id = route version. The route
//!   version is claimed from the table's clock when the advance is built
//!   ([`MapTable::claim`]) and applied at that version only after the cycle's
//!   seal ticket resolves `Ok` ([`MapTable::advance_at`]). A seal that fails
//!   applies nothing: its version is a gap in the table and in the store.
//! - **Ownership travels in the casts.** Every cast is written on behalf of the
//!   mailbox that owns the table (`SweepSlot::owner`). Restore refuses a
//!   landing on behalf of any other mailbox. There is one writer by
//!   construction: [`MapLog`] owns the table mutably and submits through the
//!   writer handle it was given; it does not arbitrate between writers.
//! - **One 512-byte cast per 64 ids.** A DR64 tile's map is 64 global ids,
//!   `8 × 64 = 512` bytes: exactly one artifact payload. A 256-wide tile is
//!   four. Each advance also carries one marker cast with its route version,
//!   source and relation-map revisions.
//! - **Restore replays, exactly.** [`restore`] reads every sealed cycle and
//!   re-applies each advance at its own route version, so every version
//!   answers as before. Rows that do not form a valid log are refused; nothing
//!   is repaired.
//!
//! What a restore does **not** carry: frames. A [`Frame`] is a witness issued
//! by one table instance, so frames sealed before a restore are foreign to the
//! restored table (`Stale(ForeignFrame)`). Past versions stay readable through
//! [`MapTable::diagnose_at`]. Issuing frames at a past version after a restore
//! is a DR owner decision this crate does not make (OPEN).
//!
//! [`Frame`]: lance_graph_dt64::dr::Frame

use std::collections::{BTreeMap, VecDeque};

use lance_graph::graph::cycle_sink::LanceCycleWriter;
use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_dt64::dr::{
    AdvanceRefusal, LocalMap, MapRefusal, MapTable, RouteVersion, TileId, VersionClock, Width,
};
use lance_graph_planner::persist_sink::{CycleId, SweepSlot, WalSink, WriteFailed};
use lance_graph_supervisor::cycle_driver::SealedCycle;
use lance_graph_supervisor::cycle_writer::{CycleWriterHandle, SealError, SealTicket, SubmitError};

/// Bytes in one cast payload (the cycle writer's artifact width).
pub const CAST_BYTES: usize = 512;
/// Global ids carried by one cast.
const IDS_PER_CAST: usize = CAST_BYTES / 8;
/// The image row of an advance's marker cast.
const MARKER_ROW: u64 = u64::MAX;
/// Tile part that marks a removal.
const REMOVED_PART: u64 = 7;
/// An unused id slot. A map holding this id is refused before anything is
/// claimed or submitted.
pub const RESERVED_ID: u64 = u64::MAX;

/// Marker kinds.
const GENESIS: u8 = 0;
const ADVANCE: u8 = 1;

/// The image row of `part` of `tile`.
fn tile_row(tile: TileId, part: u64) -> u64 {
    (u64::from(tile) << 3) | part
}

fn parts(width: Width) -> u64 {
    match width {
        Width::W64 => 1,
        Width::W256 => 4,
    }
}

/// Why an advance was not submitted, or did not become durable.
#[derive(Debug)]
pub enum AdvanceError {
    /// One advance named the same tile twice. Nothing was claimed.
    DuplicateTile(TileId),
    /// A map holds [`RESERVED_ID`]. Nothing was claimed.
    ReservedId(TileId),
    /// An earlier advance did not become durable, so this log submits
    /// nothing further (the writer seals nothing after a failure either).
    /// Nothing was claimed.
    Poisoned { failed: RouteVersion },
    /// The writer refused the cycle. `version` was claimed and is a gap; the
    /// casts come back inside `error`.
    Submit {
        version: RouteVersion,
        error: SubmitError,
    },
}

/// What happened to one submitted advance once its seal resolved.
#[derive(Debug)]
pub enum Settled {
    /// The cycle is durable and the advance is applied at `version`.
    Applied(RouteVersion),
    /// The cycle did not become durable; nothing was applied and `version` is
    /// a gap. A failed seal hands its casts back inside `error`.
    NotDurable {
        version: RouteVersion,
        error: SealError,
    },
    /// The cycle is durable but the table refused to apply it. `MapLog`
    /// applies its own advances in claimed order and owns its table, so its
    /// API cannot produce this; it is reported rather than dropped so a
    /// disagreement between store and table can never pass silently. A
    /// restore replays the store.
    Refused {
        version: RouteVersion,
        refusal: AdvanceRefusal,
    },
}

/// An advance's `(source, relation_map, changes)`.
type Advance = (u64, u64, Vec<(TileId, Option<LocalMap>)>);

struct Pending {
    version: RouteVersion,
    /// `None` for the genesis cycle, which applies nothing.
    advance: Option<Advance>,
    ticket: SealTicket,
}

/// The durable owner side of one DR map table.
pub struct MapLog {
    table: MapTable,
    writer: CycleWriterHandle,
    on_behalf: MailboxId,
    next_position: u64,
    pending: VecDeque<Pending>,
    failed: Option<RouteVersion>,
}

impl MapLog {
    /// Start a new durable table on an empty store: submits the genesis
    /// cycle (the table's source and relation-map revisions at its current
    /// version) and returns at once. `clock` may be shared with a `RouteLog`.
    pub fn create(
        writer: CycleWriterHandle,
        on_behalf: MailboxId,
        source: u64,
        relation_map: u64,
        clock: VersionClock,
    ) -> Result<MapLog, AdvanceError> {
        let table = MapTable::with_clock(source, relation_map, clock);
        let version = table.current_version();
        let mut log = MapLog {
            table,
            writer,
            on_behalf,
            next_position: 0,
            pending: VecDeque::new(),
            failed: None,
        };
        let casts = vec![log.marker(GENESIS, version, source, relation_map)];
        log.submit(version, None, casts)?;
        Ok(log)
    }

    /// Continue a restored table: further advances follow its last sealed
    /// cycle. `writer` must be framed on the store's current head.
    pub fn resume(restored: Restored, writer: CycleWriterHandle, on_behalf: MailboxId) -> MapLog {
        MapLog {
            table: restored.table,
            writer,
            on_behalf,
            next_position: restored.next_position,
            pending: VecDeque::new(),
            failed: None,
        }
    }

    /// The table as of the last applied (durable) advance.
    pub fn table(&self) -> &MapTable {
        &self.table
    }

    /// Submitted advances whose seal has not been settled yet.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Submit an advance and return its claimed route version at once. It is
    /// applied to [`MapLog::table`] only when [`MapLog::settle`] sees its
    /// cycle sealed.
    pub fn advance(
        &mut self,
        source: u64,
        relation_map: u64,
        changes: Vec<(TileId, Option<LocalMap>)>,
    ) -> Result<RouteVersion, AdvanceError> {
        if let Some(failed) = self.failed {
            return Err(AdvanceError::Poisoned { failed });
        }
        for (i, (tile, map)) in changes.iter().enumerate() {
            if changes[..i].iter().any(|(t, _)| t == tile) {
                return Err(AdvanceError::DuplicateTile(*tile));
            }
            if let Some(m) = map {
                if m.view().globals().contains(&RESERVED_ID) {
                    return Err(AdvanceError::ReservedId(*tile));
                }
            }
        }
        let version = self.table.claim();
        let mut casts = vec![self.marker(ADVANCE, version, source, relation_map)];
        for (tile, map) in &changes {
            casts.extend(self.tile_casts(version, *tile, map.as_ref()));
        }
        self.submit(version, Some((source, relation_map, changes)), casts)?;
        Ok(version)
    }

    /// Apply every advance whose seal has resolved, in submission order,
    /// without waiting. Stops at the first cycle still being sealed.
    pub fn settle(&mut self) -> Vec<Settled> {
        let mut out = Vec::new();
        while let Some(front) = self.pending.front_mut() {
            let Some(result) = front.ticket.try_sealed() else {
                break;
            };
            let Pending {
                version, advance, ..
            } = self.pending.pop_front().expect("front exists");
            if let Some(s) = self.resolve(version, advance, result) {
                out.push(s);
            }
        }
        out
    }

    /// Wait for the oldest submitted advance and apply it. `None` when
    /// nothing is pending. Used at shutdown and in tests; the owner's loop
    /// calls [`MapLog::settle`].
    pub async fn settle_next(&mut self) -> Option<Settled> {
        loop {
            let Pending {
                version,
                advance,
                ticket,
            } = self.pending.pop_front()?;
            let result = ticket.sealed().await;
            if let Some(s) = self.resolve(version, advance, result) {
                return Some(s);
            }
        }
    }

    fn resolve(
        &mut self,
        version: RouteVersion,
        advance: Option<Advance>,
        result: Result<SealedCycle, SealError>,
    ) -> Option<Settled> {
        match result {
            Err(error) => {
                self.failed.get_or_insert(version);
                Some(Settled::NotDurable { version, error })
            }
            Ok(_) => {
                let (source, relation_map, changes) = advance?;
                Some(
                    match self
                        .table
                        .advance_at(version, source, relation_map, changes)
                    {
                        Ok(v) => Settled::Applied(v),
                        Err(refusal) => Settled::Refused { version, refusal },
                    },
                )
            }
        }
    }

    fn submit(
        &mut self,
        version: RouteVersion,
        advance: Option<Advance>,
        casts: Vec<SweepSlot>,
    ) -> Result<(), AdvanceError> {
        match self.writer.try_submit(CycleId(version.0), casts) {
            Ok(ticket) => {
                self.pending.push_back(Pending {
                    version,
                    advance,
                    ticket,
                });
                Ok(())
            }
            Err(error) => Err(AdvanceError::Submit { version, error }),
        }
    }

    fn cast(&mut self, version: RouteVersion, row: u64, payload: Vec<u8>) -> SweepSlot {
        let stream_position = self.next_position;
        self.next_position += 1;
        SweepSlot {
            cycle: CycleId(version.0),
            stream_position,
            owner: self.on_behalf,
            row,
            paired_move: None,
            payload,
        }
    }

    fn marker(&mut self, kind: u8, version: RouteVersion, source: u64, rm: u64) -> SweepSlot {
        let mut p = vec![0u8; CAST_BYTES];
        p[0] = kind;
        p[8..16].copy_from_slice(&version.0.to_le_bytes());
        p[16..24].copy_from_slice(&source.to_le_bytes());
        p[24..32].copy_from_slice(&rm.to_le_bytes());
        self.cast(version, MARKER_ROW, p)
    }

    fn tile_casts(
        &mut self,
        version: RouteVersion,
        tile: TileId,
        map: Option<&LocalMap>,
    ) -> Vec<SweepSlot> {
        let Some(map) = map else {
            let p = vec![0u8; CAST_BYTES];
            return vec![self.cast(version, tile_row(tile, REMOVED_PART), p)];
        };
        let view = map.view();
        let globals = view.globals();
        (0..parts(map.width()))
            .map(|part| {
                let mut p = vec![0u8; CAST_BYTES];
                for slot in 0..IDS_PER_CAST {
                    let i = part as usize * IDS_PER_CAST + slot;
                    let id = globals.get(i).copied().unwrap_or(RESERVED_ID);
                    p[slot * 8..slot * 8 + 8].copy_from_slice(&id.to_le_bytes());
                }
                self.cast(version, tile_row(tile, part), p)
            })
            .collect()
    }
}

/// A sealed store that can hand back each cycle's coalesced image (the
/// payloads `WalSink::scan_sealed` deliberately does not carry).
#[allow(async_fn_in_trait)]
pub trait SealedImages: WalSink {
    /// `row → payload` for one sealed cycle.
    async fn sealed_image(&self, cycle: CycleId) -> Result<BTreeMap<u64, Vec<u8>>, WriteFailed>;
}

impl SealedImages for LanceCycleWriter {
    async fn sealed_image(&self, cycle: CycleId) -> Result<BTreeMap<u64, Vec<u8>>, WriteFailed> {
        self.scan_image(cycle).await
    }
}

/// A table rebuilt from its store.
pub struct Restored {
    pub table: MapTable,
    /// The next cast stream position (one past the last stored landing).
    pub next_position: u64,
    /// The last sealed cycle (the last advance's route version).
    pub last: RouteVersion,
}

/// Why a restore failed.
#[derive(Debug)]
pub enum RestoreError {
    /// The store could not be read.
    Read(WriteFailed),
    /// The stored cycles do not form a valid log.
    Replay(ReplayError),
}

/// Why stored cycles are not a valid log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    /// The store holds no cycles.
    Empty,
    /// The first cycle is not a genesis, or a second genesis follows.
    Genesis { cycle: u64 },
    /// A cycle has no marker cast.
    MissingMarker { cycle: u64 },
    /// A marker's route version is not its cycle id.
    VersionMismatch { cycle: u64, marker: u64 },
    /// A marker kind this crate does not write.
    UnknownKind { cycle: u64, kind: u8 },
    /// A landing was written on behalf of another mailbox.
    ForeignOwner { cycle: u64, owner: MailboxId },
    /// A cast length other than [`CAST_BYTES`].
    PayloadLength { cycle: u64, row: u64, len: usize },
    /// A tile's parts are not one width's parts or one removal, or its ids
    /// are not a prefix followed only by unused slots.
    Positions { cycle: u64, tile: TileId },
    /// The ids do not form a map (duplicate id, too many).
    Map {
        cycle: u64,
        tile: TileId,
        refusal: MapRefusal,
    },
    /// Re-applying the advance was refused by the table.
    Apply { cycle: u64, refusal: AdvanceRefusal },
}

/// Rebuild a table from every sealed cycle in `sink`, re-applying each
/// advance at its own route version. Every landing must be on behalf of
/// `owner`. `clock` becomes the restored table's clock.
pub async fn restore<S: SealedImages>(
    sink: &S,
    owner: MailboxId,
    clock: VersionClock,
) -> Result<Restored, RestoreError> {
    let landings = sink.scan_sealed(None).await.map_err(RestoreError::Read)?;
    let mut cycles: Vec<u64> = Vec::new();
    let mut next_position = 0;
    for l in &landings {
        if l.slot.owner != owner {
            return Err(RestoreError::Replay(ReplayError::ForeignOwner {
                cycle: l.cycle.0,
                owner: l.slot.owner,
            }));
        }
        next_position = next_position.max(l.slot.stream_position + 1);
        if cycles.last() != Some(&l.cycle.0) {
            cycles.push(l.cycle.0);
        }
    }
    let mut images = Vec::with_capacity(cycles.len());
    for &c in &cycles {
        let image = sink
            .sealed_image(CycleId(c))
            .await
            .map_err(RestoreError::Read)?;
        images.push((c, image));
    }
    let (table, last) = replay(&images, clock).map_err(RestoreError::Replay)?;
    Ok(Restored {
        table,
        next_position,
        last,
    })
}

/// A decoded marker: `(kind, version, source, relation_map)`.
fn decode_marker(cycle: u64, image: &BTreeMap<u64, Vec<u8>>) -> Result<(u8, u64, u64, u64), ReplayError> {
    let p = image
        .get(&MARKER_ROW)
        .ok_or(ReplayError::MissingMarker { cycle })?;
    let word = |at: usize| u64::from_le_bytes(p[at..at + 8].try_into().expect("8 bytes"));
    Ok((p[0], word(8), word(16), word(24)))
}

fn replay(
    images: &[(u64, BTreeMap<u64, Vec<u8>>)],
    clock: VersionClock,
) -> Result<(MapTable, RouteVersion), ReplayError> {
    for (cycle, image) in images {
        for (row, p) in image {
            if p.len() != CAST_BYTES {
                return Err(ReplayError::PayloadLength {
                    cycle: *cycle,
                    row: *row,
                    len: p.len(),
                });
            }
        }
    }
    let ((first, genesis), rest) = images.split_first().ok_or(ReplayError::Empty)?;
    let (kind, version, source, relation_map) = decode_marker(*first, genesis)?;
    if kind != GENESIS || genesis.len() != 1 {
        return Err(ReplayError::Genesis { cycle: *first });
    }
    if version != *first {
        return Err(ReplayError::VersionMismatch {
            cycle: *first,
            marker: version,
        });
    }
    while clock.current().0 < version {
        clock.tick();
    }
    let mut table = MapTable::with_clock(source, relation_map, clock.clone());
    let mut last = RouteVersion(version);
    for (cycle, image) in rest {
        let (kind, version, source, relation_map) = decode_marker(*cycle, image)?;
        match kind {
            ADVANCE => {}
            GENESIS => return Err(ReplayError::Genesis { cycle: *cycle }),
            other => {
                return Err(ReplayError::UnknownKind {
                    cycle: *cycle,
                    kind: other,
                })
            }
        }
        if version != *cycle {
            return Err(ReplayError::VersionMismatch {
                cycle: *cycle,
                marker: version,
            });
        }
        let changes = changes_of(*cycle, image)?;
        while clock.current().0 < version {
            clock.tick();
        }
        table
            .advance_at(RouteVersion(version), source, relation_map, changes)
            .map_err(|refusal| ReplayError::Apply {
                cycle: *cycle,
                refusal,
            })?;
        last = RouteVersion(version);
    }
    Ok((table, last))
}

/// The changes of one advance from its tile casts.
fn changes_of(
    cycle: u64,
    image: &BTreeMap<u64, Vec<u8>>,
) -> Result<Vec<(TileId, Option<LocalMap>)>, ReplayError> {
    let mut by_tile: BTreeMap<TileId, Vec<(u64, &[u8])>> = BTreeMap::new();
    for (&row, p) in image {
        if row == MARKER_ROW {
            continue;
        }
        let tile = TileId::try_from(row >> 3).map_err(|_| ReplayError::Positions {
            cycle,
            tile: TileId::MAX,
        })?;
        by_tile.entry(tile).or_default().push((row & 7, p));
    }
    let mut changes = Vec::with_capacity(by_tile.len());
    for (tile, mut rows) in by_tile {
        rows.sort_by_key(|(part, _)| *part);
        let found: Vec<u64> = rows.iter().map(|(part, _)| *part).collect();
        let bad = ReplayError::Positions { cycle, tile };
        let width = match found.as_slice() {
            [REMOVED_PART] => {
                changes.push((tile, None));
                continue;
            }
            [0] => Width::W64,
            [0, 1, 2, 3] => Width::W256,
            _ => return Err(bad),
        };
        let mut globals = Vec::new();
        let mut ended = false;
        for (_, p) in &rows {
            for slot in p.chunks_exact(8) {
                let id = u64::from_le_bytes(slot.try_into().expect("8 bytes"));
                if id == RESERVED_ID {
                    ended = true;
                } else if ended {
                    return Err(bad);
                } else {
                    globals.push(id);
                }
            }
        }
        let map = LocalMap::prepare(width, &globals).map_err(|refusal| ReplayError::Map {
            cycle,
            tile,
            refusal,
        })?;
        changes.push((tile, Some(map)));
    }
    Ok(changes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(entries: &[(u64, Vec<u8>)]) -> BTreeMap<u64, Vec<u8>> {
        entries.iter().cloned().collect()
    }

    fn ids(list: &[u64]) -> Vec<u8> {
        let mut p = vec![0u8; CAST_BYTES];
        for slot in 0..IDS_PER_CAST {
            let id = list.get(slot).copied().unwrap_or(RESERVED_ID);
            p[slot * 8..slot * 8 + 8].copy_from_slice(&id.to_le_bytes());
        }
        p
    }

    #[test]
    fn a_w64_tile_is_one_cast_and_reads_back_as_its_ids() {
        let c = changes_of(1, &image(&[(tile_row(5, 0), ids(&[9, 8, 7]))])).unwrap();
        assert_eq!(c.len(), 1);
        let (tile, map) = &c[0];
        assert_eq!(*tile, 5);
        let map = map.as_ref().unwrap();
        assert_eq!(map.width(), Width::W64);
        assert_eq!(map.view().globals(), &[9, 8, 7]);
    }

    #[test]
    fn tile_parts_that_are_not_one_width_are_refused() {
        // Two of a W256 tile's four parts.
        let two = image(&[(tile_row(1, 0), ids(&[1])), (tile_row(1, 1), ids(&[]))]);
        assert_eq!(
            changes_of(3, &two).unwrap_err(),
            ReplayError::Positions { cycle: 3, tile: 1 }
        );
        // A removal next to a part.
        let mixed = image(&[
            (tile_row(1, 0), ids(&[1])),
            (tile_row(1, REMOVED_PART), vec![0; CAST_BYTES]),
        ]);
        assert!(changes_of(3, &mixed).is_err());
    }

    #[test]
    fn an_id_after_an_unused_slot_is_refused() {
        let mut p = ids(&[1, 2]);
        p[5 * 8..6 * 8].copy_from_slice(&3u64.to_le_bytes());
        let err = changes_of(4, &image(&[(tile_row(2, 0), p)])).unwrap_err();
        assert_eq!(err, ReplayError::Positions { cycle: 4, tile: 2 });
    }

    #[test]
    fn replay_refuses_a_log_without_genesis_or_marker() {
        let advance = |v: u64| {
            let mut p = vec![0u8; CAST_BYTES];
            p[0] = ADVANCE;
            p[8..16].copy_from_slice(&v.to_le_bytes());
            p
        };
        let genesis = {
            let mut p = vec![0u8; CAST_BYTES];
            p[0] = GENESIS;
            p
        };
        let replay_of = |cycles: Vec<(u64, BTreeMap<u64, Vec<u8>>)>| {
            replay(&cycles, VersionClock::new()).map(|_| ()).unwrap_err()
        };
        assert_eq!(replay_of(vec![]), ReplayError::Empty);
        assert_eq!(
            replay_of(vec![(1, image(&[(MARKER_ROW, advance(1))]))]),
            ReplayError::Genesis { cycle: 1 }
        );
        assert_eq!(
            replay_of(vec![
                (0, image(&[(MARKER_ROW, genesis.clone())])),
                (1, image(&[(tile_row(1, 0), ids(&[1]))])),
            ]),
            ReplayError::MissingMarker { cycle: 1 }
        );
        assert_eq!(
            replay_of(vec![
                (0, image(&[(MARKER_ROW, genesis.clone())])),
                (2, image(&[(MARKER_ROW, advance(1))])),
            ]),
            ReplayError::VersionMismatch {
                cycle: 2,
                marker: 1
            }
        );
        assert_eq!(
            replay_of(vec![
                (0, image(&[(MARKER_ROW, genesis.clone())])),
                (1, image(&[(MARKER_ROW, genesis)])),
            ]),
            ReplayError::Genesis { cycle: 1 }
        );
    }
}
