//! DR's versioned global → local table ([`MapTable`]) stored as a separate,
//! Lance-versioned dataset (D-DT64-7 O5 / D10', D-LITHO Q5).
//!
//! [`MapTable`] is the in-memory DR log: append-only, every past version
//! answerable, frames replaying the layout they were sealed under. This crate
//! makes that log durable without changing it.
//!
//! - **One advance is one Lance commit.** [`LanceMapLog::advance`] writes the
//!   advance's rows in one commit, and only then applies the same advance to
//!   the table. An advance is in the dataset whole or not at all.
//! - **One row per local position** (the D10' working model). A tile map at a
//!   version is `width` plus its `local → global` rows in local order. A
//!   removal and the advance itself are marker rows. Ids only, never values.
//! - **Two clocks, never mixed.** The DR route version travels in the rows.
//!   The Lance dataset version is the physical commit counter: it refuses a
//!   stale writer, detects a race, and names a past commit to read with
//!   [`LanceMapLog::restore_at`]. It is never written into, or read as, a
//!   route version.
//! - **Restore replays, exactly.** [`LanceMapLog::restore`] reads every row in
//!   one scan and re-applies each advance to a fresh table at the same route
//!   version, so every version answers as before: an extension stays an
//!   extension, a replacement a new generation, a removal an event. Rows that
//!   do not form a valid log are refused; nothing is repaired.
//!
//! What a restore does **not** carry: frames. A [`Frame`] is a witness issued
//! by one table instance, so frames sealed before a restore are foreign to
//! the restored table (`Stale(ForeignFrame)`). Past versions stay readable
//! through [`MapTable::diagnose_at`]; executing at a past version after a
//! restore would need a frame issued at that version, which DR does not
//! offer (OPEN, DR owner).
//!
//! [`Frame`]: lance_graph_dt64::dr::Frame

use std::sync::Arc;

use arrow_array::{
    Array, RecordBatch, RecordBatchIterator, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use lance::dataset::{WriteMode, WriteParams};
use lance::Dataset;
use lance_graph_dt64::dr::{
    LocalMap, MapRefusal, MapTable, RouteVersion, TileId, VersionClock, Width,
};

/// Row kinds.
const GENESIS: u8 = 0;
const ADVANCE: u8 = 1;
const POSITION: u8 = 2;
const REMOVED: u8 = 3;

/// One stored row. `source` / `relation_map` are set on genesis and advance
/// rows; `tile`, `width`, `local`, `global` on position rows (`tile` also on
/// removals); unused fields are zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Row {
    version: u64,
    kind: u8,
    source: u64,
    relation_map: u64,
    tile: u32,
    width: u16,
    local: u16,
    global: u64,
}

impl Row {
    fn marker(version: u64, kind: u8, source: u64, relation_map: u64) -> Row {
        Row {
            version,
            kind,
            source,
            relation_map,
            tile: 0,
            width: 0,
            local: 0,
            global: 0,
        }
    }
}

/// Why a save or a restore failed.
#[derive(Debug)]
pub enum StoreError {
    /// Lance refused an operation.
    Lance(lance::Error),
    /// A stored column is missing, has another type, or holds a null.
    Schema(String),
    /// The stored rows do not form a valid log.
    Replay(ReplayError),
    /// One advance named the same tile twice. Nothing was written.
    DuplicateTile(TileId),
    /// The dataset is not at the version this writer last saw (`None`: no
    /// dataset). Nothing was written.
    StaleWriter {
        expected: Option<u64>,
        found: Option<u64>,
    },
    /// The commit landed, but not as the next dataset version: another
    /// writer committed between this writer's check and its commit. The rows
    /// are stored; the table was not advanced.
    Race { expected: u64, found: u64 },
    /// The rows were stored at route version `stored`, but the table's clock
    /// had moved, so the advance was issued at `issued`. Happens only when a
    /// clone of the table's clock ticked between the check and the advance.
    ClockMoved { stored: u64, issued: u64 },
}

impl From<lance::Error> for StoreError {
    fn from(e: lance::Error) -> Self {
        StoreError::Lance(e)
    }
}

/// Why stored rows are not a valid log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    /// The dataset holds no rows.
    Empty,
    /// The first row is not the genesis row, or a second genesis follows.
    Genesis,
    /// A row's route version is not after the previous advance's.
    VersionOrder { previous: u64, found: u64 },
    /// A version's rows do not start with its advance marker.
    MissingAdvance { version: u64 },
    /// A row kind this crate does not write.
    UnknownKind(u8),
    /// A width other than 64 or 256.
    Width(u16),
    /// A tile's positions are not `0..len` in order, or mix widths.
    Positions { version: u64, tile: u32 },
    /// A tile appears twice in one advance.
    DuplicateTile { version: u64, tile: u32 },
    /// The positions do not form a map (duplicate id, too many).
    Map {
        version: u64,
        tile: u32,
        refusal: MapRefusal,
    },
    /// Re-applying the advance issued another route version.
    Reissued { stored: u64, issued: u64 },
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("version", DataType::UInt64, false),
        Field::new("kind", DataType::UInt8, false),
        Field::new("source", DataType::UInt64, false),
        Field::new("relation_map", DataType::UInt64, false),
        Field::new("tile", DataType::UInt32, false),
        Field::new("width", DataType::UInt16, false),
        Field::new("local", DataType::UInt16, false),
        Field::new("global", DataType::UInt64, false),
    ]))
}

fn to_batch(rows: &[Row]) -> Result<RecordBatch, ArrowError> {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|r| r.version),
            )),
            Arc::new(UInt8Array::from_iter_values(rows.iter().map(|r| r.kind))),
            Arc::new(UInt64Array::from_iter_values(rows.iter().map(|r| r.source))),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|r| r.relation_map),
            )),
            Arc::new(UInt32Array::from_iter_values(rows.iter().map(|r| r.tile))),
            Arc::new(UInt16Array::from_iter_values(rows.iter().map(|r| r.width))),
            Arc::new(UInt16Array::from_iter_values(rows.iter().map(|r| r.local))),
            Arc::new(UInt64Array::from_iter_values(rows.iter().map(|r| r.global))),
        ],
    )
}

fn column<'a, T: 'static + Array>(b: &'a RecordBatch, name: &str) -> Result<&'a T, StoreError> {
    let c = b
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<T>())
        .ok_or_else(|| StoreError::Schema(format!("column `{name}`")))?;
    if c.null_count() != 0 {
        return Err(StoreError::Schema(format!("null in `{name}`")));
    }
    Ok(c)
}

fn from_batch(b: &RecordBatch) -> Result<Vec<Row>, StoreError> {
    let version: &UInt64Array = column(b, "version")?;
    let kind: &UInt8Array = column(b, "kind")?;
    let source: &UInt64Array = column(b, "source")?;
    let relation_map: &UInt64Array = column(b, "relation_map")?;
    let tile: &UInt32Array = column(b, "tile")?;
    let width: &UInt16Array = column(b, "width")?;
    let local: &UInt16Array = column(b, "local")?;
    let global: &UInt64Array = column(b, "global")?;
    Ok((0..b.num_rows())
        .map(|i| Row {
            version: version.value(i),
            kind: kind.value(i),
            source: source.value(i),
            relation_map: relation_map.value(i),
            tile: tile.value(i),
            width: width.value(i),
            local: local.value(i),
            global: global.value(i),
        })
        .collect())
}

fn encode_width(w: Width) -> u16 {
    match w {
        Width::W64 => 64,
        Width::W256 => 256,
    }
}

fn decode_width(w: u16) -> Result<Width, ReplayError> {
    match w {
        64 => Ok(Width::W64),
        256 => Ok(Width::W256),
        other => Err(ReplayError::Width(other)),
    }
}

/// The rows of one advance at `version`: its marker, then each change.
fn advance_rows(
    version: u64,
    source: u64,
    relation_map: u64,
    changes: &[(TileId, Option<LocalMap>)],
) -> Result<Vec<Row>, StoreError> {
    let mut seen: Vec<TileId> = changes.iter().map(|(t, _)| *t).collect();
    seen.sort_unstable();
    if let Some(w) = seen.windows(2).find(|w| w[0] == w[1]) {
        return Err(StoreError::DuplicateTile(w[0]));
    }
    let mut rows = vec![Row::marker(version, ADVANCE, source, relation_map)];
    for (tile, map) in changes {
        match map {
            None => rows.push(Row {
                tile: *tile,
                ..Row::marker(version, REMOVED, 0, 0)
            }),
            Some(m) => {
                let width = encode_width(m.width());
                for (local, &global) in m.view().globals().iter().enumerate() {
                    rows.push(Row {
                        tile: *tile,
                        width,
                        // A map holds at most 256 positions.
                        local: local as u16,
                        global,
                        ..Row::marker(version, POSITION, 0, 0)
                    });
                }
                if m.is_empty() {
                    // An empty map still replaces the tile: keep it visible.
                    rows.push(Row {
                        tile: *tile,
                        width,
                        local: u16::MAX,
                        ..Row::marker(version, POSITION, 0, 0)
                    });
                }
            }
        }
    }
    Ok(rows)
}

/// Rebuild a table from its rows, re-applying every advance at its own route
/// version.
fn replay(rows: &[Row]) -> Result<MapTable, ReplayError> {
    let (genesis, rest) = rows.split_first().ok_or(ReplayError::Empty)?;
    if genesis.kind != GENESIS {
        return Err(ReplayError::Genesis);
    }
    let clock = VersionClock::new();
    while clock.current().0 < genesis.version {
        clock.tick();
    }
    let mut table = MapTable::with_clock(genesis.source, genesis.relation_map, clock.clone());
    let mut i = 0;
    let mut previous = genesis.version;
    while i < rest.len() {
        let marker = rest[i];
        match marker.kind {
            GENESIS => return Err(ReplayError::Genesis),
            ADVANCE => {}
            POSITION | REMOVED => {
                return Err(ReplayError::MissingAdvance {
                    version: marker.version,
                })
            }
            other => return Err(ReplayError::UnknownKind(other)),
        }
        let version = marker.version;
        if version <= previous {
            return Err(ReplayError::VersionOrder {
                previous,
                found: version,
            });
        }
        let end = i
            + 1
            + rest[i + 1..]
                .iter()
                .take_while(|r| r.version == version && r.kind != ADVANCE)
                .count();
        let changes = changes_of(version, &rest[i + 1..end])?;
        while clock.current().0 + 1 < version {
            clock.tick();
        }
        let issued = table.advance(marker.source, marker.relation_map, changes);
        if issued.0 != version {
            return Err(ReplayError::Reissued {
                stored: version,
                issued: issued.0,
            });
        }
        previous = version;
        i = end;
    }
    Ok(table)
}

/// The changes of one advance from its position and removal rows.
fn changes_of(version: u64, rows: &[Row]) -> Result<Vec<(TileId, Option<LocalMap>)>, ReplayError> {
    let mut changes: Vec<(TileId, Option<LocalMap>)> = Vec::new();
    let mut j = 0;
    while j < rows.len() {
        let tile = rows[j].tile;
        if changes.iter().any(|(t, _)| *t == tile) {
            return Err(ReplayError::DuplicateTile { version, tile });
        }
        match rows[j].kind {
            REMOVED => {
                changes.push((tile, None));
                j += 1;
            }
            POSITION => {
                let run = rows[j..]
                    .iter()
                    .take_while(|r| r.kind == POSITION && r.tile == tile)
                    .count();
                let span = &rows[j..j + run];
                let width = decode_width(span[0].width)?;
                let empty = run == 1 && span[0].local == u16::MAX;
                let ordered = span
                    .iter()
                    .enumerate()
                    .all(|(k, r)| usize::from(r.local) == k && r.width == span[0].width);
                if !empty && !ordered {
                    return Err(ReplayError::Positions { version, tile });
                }
                let globals: Vec<u64> = if empty {
                    Vec::new()
                } else {
                    span.iter().map(|r| r.global).collect()
                };
                let map =
                    LocalMap::prepare(width, &globals).map_err(|refusal| ReplayError::Map {
                        version,
                        tile,
                        refusal,
                    })?;
                changes.push((tile, Some(map)));
                j += run;
            }
            other => return Err(ReplayError::UnknownKind(other)),
        }
    }
    Ok(changes)
}

fn is_not_found(e: &lance::Error) -> bool {
    matches!(e, lance::Error::DatasetNotFound { .. })
}

/// The dataset version a commit on top of `seen` makes: Lance numbers a new
/// dataset 1 and each later commit one higher.
fn next_version(seen: Option<u64>) -> u64 {
    seen.map_or(1, |v| v + 1)
}

/// A writer's handle on one DR map table's dataset at `uri`.
#[derive(Debug, Clone)]
pub struct LanceMapLog {
    uri: String,
    /// The dataset version this writer last saw.
    version: Option<u64>,
}

impl LanceMapLog {
    /// Create the dataset with its genesis row and return a new table with
    /// the given revisions and its writer. Refuses if a dataset is already at
    /// `uri`.
    pub async fn create(
        uri: impl Into<String>,
        source: u64,
        relation_map: u64,
    ) -> Result<(MapTable, LanceMapLog), StoreError> {
        let mut log = LanceMapLog {
            uri: uri.into(),
            version: None,
        };
        let table = MapTable::new(source, relation_map);
        let genesis = Row::marker(table.current_version().0, GENESIS, source, relation_map);
        log.commit(&[genesis]).await?;
        Ok((table, log))
    }

    /// The dataset version this writer last saw (the physical commit
    /// counter, not a route version).
    pub fn dataset_version(&self) -> Option<u64> {
        self.version
    }

    /// Store one advance as one Lance commit, then apply it to `table`.
    /// Returns the route version it was issued at.
    ///
    /// # Errors
    ///
    /// - [`StoreError::DuplicateTile`] or [`StoreError::StaleWriter`]:
    ///   nothing is written and the table is unchanged.
    /// - [`StoreError::Race`]: the rows landed as another dataset version;
    ///   the table is unchanged.
    /// - [`StoreError::ClockMoved`]: see the variant.
    pub async fn advance(
        &mut self,
        table: &mut MapTable,
        source: u64,
        relation_map: u64,
        changes: Vec<(TileId, Option<LocalMap>)>,
    ) -> Result<RouteVersion, StoreError> {
        let stored = table.current_version().0 + 1;
        let rows = advance_rows(stored, source, relation_map, &changes)?;
        self.commit(&rows).await?;
        let issued = table.advance(source, relation_map, changes);
        if issued.0 != stored {
            return Err(StoreError::ClockMoved {
                stored,
                issued: issued.0,
            });
        }
        Ok(issued)
    }

    async fn commit(&mut self, rows: &[Row]) -> Result<u64, StoreError> {
        let found = match Dataset::open(&self.uri).await {
            Ok(ds) => Some(ds.latest_version_id().await?),
            Err(e) if is_not_found(&e) => None,
            Err(e) => return Err(e.into()),
        };
        if found != self.version {
            return Err(StoreError::StaleWriter {
                expected: self.version,
                found,
            });
        }
        let rb = to_batch(rows).map_err(|e| StoreError::Schema(e.to_string()))?;
        let reader = RecordBatchIterator::new([Ok(rb)], schema());
        let mode = if found.is_some() {
            WriteMode::Append
        } else {
            WriteMode::Create
        };
        let params = WriteParams {
            mode,
            ..WriteParams::default()
        };
        let ds = match Dataset::write(reader, self.uri.as_str(), Some(params)).await {
            Ok(ds) => ds,
            Err(lance::Error::DatasetAlreadyExists { .. }) => {
                return Err(StoreError::StaleWriter {
                    expected: None,
                    found: Dataset::open(&self.uri)
                        .await
                        .ok()
                        .map(|d| d.version().version),
                })
            }
            Err(e) => return Err(e.into()),
        };
        let made = ds.version().version;
        let next = next_version(found);
        if made != next {
            return Err(StoreError::Race {
                expected: next,
                found: made,
            });
        }
        self.version = Some(made);
        Ok(made)
    }

    /// Read the whole table at `uri` once and replay it, with a writer at the
    /// dataset's current version.
    pub async fn restore(uri: impl Into<String>) -> Result<(MapTable, LanceMapLog), StoreError> {
        let uri = uri.into();
        let ds = Dataset::open(&uri).await?;
        let table = read(&ds).await?;
        let version = Some(ds.version().version);
        Ok((table, LanceMapLog { uri, version }))
    }

    /// The table as of dataset version `dataset_version` (a past commit),
    /// read through Lance's own version history. Read-only: no writer is
    /// returned, since appending on top of a past commit would fork the log.
    pub async fn restore_at(
        uri: impl Into<String>,
        dataset_version: u64,
    ) -> Result<MapTable, StoreError> {
        let uri = uri.into();
        let ds = Dataset::open(&uri)
            .await?
            .checkout_version(dataset_version)
            .await?;
        read(&ds).await
    }
}

async fn read(ds: &Dataset) -> Result<MapTable, StoreError> {
    let all = ds.scan().try_into_batch().await?;
    let rows = from_batch(&all)?;
    replay(&rows).map_err(StoreError::Replay)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(w: Width, ids: &[u64]) -> Option<LocalMap> {
        Some(LocalMap::prepare(w, ids).unwrap())
    }

    #[test]
    fn rows_round_trip_through_a_record_batch() {
        let rows = advance_rows(7, 3, 4, &[(1, map(Width::W64, &[9, 8])), (2, None)]).unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(from_batch(&to_batch(&rows).unwrap()).unwrap(), rows);
    }

    #[test]
    fn an_advance_naming_a_tile_twice_is_refused() {
        let err = advance_rows(1, 1, 1, &[(5, None), (5, map(Width::W64, &[1]))]).unwrap_err();
        assert!(matches!(err, StoreError::DuplicateTile(5)));
    }

    #[test]
    fn replay_refuses_rows_that_are_not_a_log() {
        let genesis = Row::marker(0, GENESIS, 1, 1);
        let good = advance_rows(1, 1, 1, &[(1, map(Width::W64, &[4, 5]))]).unwrap();
        let mut rows = vec![genesis];
        rows.extend(&good);
        assert!(replay(&rows).is_ok());

        assert_eq!(replay(&rows[1..]).unwrap_err(), ReplayError::Genesis);
        assert_eq!(replay(&[]).unwrap_err(), ReplayError::Empty);

        let mut swapped = rows.clone();
        swapped.swap(2, 3);
        assert_eq!(
            replay(&swapped).unwrap_err(),
            ReplayError::Positions {
                version: 1,
                tile: 1
            }
        );

        let mut dup = rows.clone();
        dup[3].global = dup[2].global;
        assert!(matches!(
            replay(&dup).unwrap_err(),
            ReplayError::Map {
                refusal: MapRefusal::Duplicate,
                ..
            }
        ));

        let mut back = rows.clone();
        back.extend(advance_rows(1, 1, 1, &[]).unwrap());
        assert_eq!(
            replay(&back).unwrap_err(),
            ReplayError::VersionOrder {
                previous: 1,
                found: 1
            }
        );

        let orphan = vec![genesis, good[1]];
        assert_eq!(
            replay(&orphan).unwrap_err(),
            ReplayError::MissingAdvance { version: 1 }
        );
    }

    #[test]
    fn the_next_version_is_one_more_or_one_for_a_new_dataset() {
        assert_eq!(next_version(None), 1);
        assert_eq!(next_version(Some(4)), 5);
    }
}
