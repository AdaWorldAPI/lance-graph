//! A sealed, persistable [`OrderIndex`]: the order of one key, kept beside a
//! table, with the witness that says which table state it is valid for.
//!
//! An order index is a content-dependent preparation: its rows and keys are
//! read from the source's contents. Coordinates, class and population can all
//! stay the same while the contents move, so an order is admitted only by
//!
//! * an O(n) check minted once: at [`SealedOrder::seal`] (the cached keys
//!   against the lane) and at [`SealedOrder::from_bytes`] (the bytes are a
//!   permutation, in the stated order law); or
//! * an O(1) attest ([`SealedOrder::attest`]): the witness fields equal the
//!   table's.
//!
//! Nothing in this module orders rows. The order is read from the index or
//! from the file and only checked.
//!
//! What the witness covers: the table's class, table population, index
//! population, row count, source revision, relation-map revision and
//! coordinates revision, the key lane (which key the order is over), the
//! presence version, and the tie-break rule ([`TIE_KEY_THEN_FRAME_POSITION`]).
//!
//! * Rows are frame-relative canonical positions: table row `0..n_rows` in
//!   the table's own row order at the witnessed coordinates revision. They
//!   are not global ids and not storage row addresses. The table's lanes are
//!   never moved.
//! * The keys are a derived cache of source content, not a versioned copy.
//!   They are checked against the lane once, at seal. A source revision that
//!   differs refuses the order; no contents are kept to compare against.
//! * The presence version is stated by the caller, because the lab `Table`
//!   carries none. It is an input, like the scope plane.
//! * There is no checksum. A file damaged in a way that keeps it a valid
//!   permutation in order is not detected here.
//!
//! Reading is one decode into a `Vec<u32>` and a `Vec<i32>`; the crate
//! forbids `unsafe`, so the file is parsed with `from_le_bytes`.

use crate::{OrderIndex, Origin, Table};
use lance_graph_dt64::dr::FrameRevisions;
use lance_graph_mask_risc::LaneRef;
use std::path::Path;

/// The one tie-break rule this layout knows: ascending key, equal keys by
/// ascending frame-relative position (the row's position in the bound frame).
pub const TIE_KEY_THEN_FRAME_POSITION: u32 = 1;

/// Byte layout constants (public so a test can patch a field).
pub mod layout {
    pub const MAGIC: [u8; 8] = *b"DRORDIX1";
    pub const VERSION: u32 = 1;
    pub const OFF_MAGIC: usize = 0;
    pub const OFF_VERSION: usize = 8;
    pub const OFF_TIE_BREAK: usize = 12;
    pub const OFF_CLASS: usize = 16;
    pub const OFF_KEY_KIND: usize = 20;
    pub const OFF_KEY_ID: usize = 24;
    pub const OFF_TABLE_POPULATION: usize = 32;
    pub const OFF_INDEX_POPULATION: usize = 40;
    pub const OFF_N_ROWS: usize = 48;
    pub const OFF_SOURCE: usize = 56;
    pub const OFF_RELATION_MAP: usize = 64;
    pub const OFF_COORDINATES: usize = 72;
    pub const OFF_PRESENCE: usize = 80;
    pub const HEADER_BYTES: usize = 88;
}

/// Which key the order is over (ordered-lane identity).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyLane {
    /// An `i32` field of the table's class, by basis position.
    Field(u8),
    /// A computed key (`OrderIndex::by_key`), named by a caller-chosen tag.
    Computed(u64),
}

/// Why sealing, reading or attesting was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreRefused {
    Io(std::io::ErrorKind),
    BadMagic,
    UnknownLayout {
        version: u32,
    },
    UnknownTieBreak {
        rule: u32,
    },
    UnknownKeyKind {
        kind: u32,
    },
    /// The byte length is not exactly `HEADER_BYTES + 8 * n_rows`.
    Length {
        expected: u64,
        found: u64,
    },
    /// The stored index claims the table's own row order.
    OwnRowOrderClaimed,
    /// `order` is not a permutation of `0..n_rows`; `at` is the first bad
    /// index position.
    NotAPermutation {
        at: usize,
    },
    /// `(key, row)` at `at` is not strictly greater than at `at - 1`.
    OutOfOrder {
        at: usize,
    },
    /// seal: `KeyLane::Field(f)` names no `i32` lane of the table.
    KeyLaneMissing {
        field: u8,
    },
    /// seal: the cached key at index position `at` is not the lane's value
    /// there.
    KeyMismatch {
        at: usize,
    },
    /// seal: the index was built over another table or revision.
    IndexOfAnotherTable,
    /// attest: class, table population or row count differ.
    AnotherTable,
    /// attest: relation map or coordinates revision differ.
    MappingChanged,
    /// attest: same mapping, another source revision (content may have
    /// changed).
    SourceChanged {
        sealed: u64,
        found: u64,
    },
    AnotherKey,
    PresenceChanged {
        sealed: u64,
        found: u64,
    },
}

/// An order index with its full witness. Not `Clone`: one owner of the
/// permutation.
pub struct SealedOrder {
    /// Index position -> frame-relative table row.
    order: Vec<u32>,
    /// The key at each index position (derived cache of source content).
    keys: Vec<i32>,
    index_population: u64,
    origin: Origin,
    key: KeyLane,
    presence: u64,
}

impl core::fmt::Debug for SealedOrder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SealedOrder")
            .field("len", &self.order.len())
            .field("key", &self.key)
            .field("revisions", &self.origin.revisions)
            .field("presence", &self.presence)
            .finish()
    }
}

/// The `i32` lane of field `f`, if the table holds one.
fn i32_lane<'t>(table: &Table<'t>, f: u8) -> Option<&'t [i32]> {
    match table.lanes.get(usize::from(f)).copied().flatten() {
        Some(LaneRef::I32(lane)) => Some(lane),
        _ => None,
    }
}

fn u32_at(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn u64_at(bytes: &[u8], off: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[off..off + 8]);
    u64::from_le_bytes(b)
}

impl SealedOrder {
    /// Mint the seal: the O(n) check, once.
    pub fn seal(
        index: OrderIndex,
        table: &Table<'_>,
        key: KeyLane,
        presence: u64,
    ) -> Result<SealedOrder, StoreRefused> {
        if index.origin != Origin::of(table) {
            return Err(StoreRefused::IndexOfAnotherTable);
        }
        if index.order.len() != index.origin.n_rows {
            return Err(StoreRefused::IndexOfAnotherTable);
        }
        let field = match key {
            KeyLane::Field(f) => Some(f),
            KeyLane::Computed(_) => None,
        };
        let lane = field.and_then(|f| i32_lane(table, f));
        if let Some(field) = field.filter(|_| lane.is_none()) {
            return Err(StoreRefused::KeyLaneMissing { field });
        }
        if let Some(lane) = lane {
            for (at, (&row, &k)) in index.order.iter().zip(index.keys.iter()).enumerate() {
                if lane.get(row as usize) != Some(&k) {
                    return Err(StoreRefused::KeyMismatch { at });
                }
            }
        }
        Ok(SealedOrder {
            order: index.order,
            keys: index.keys,
            index_population: index.population,
            origin: index.origin,
            key,
            presence,
        })
    }

    /// The file form: exactly `HEADER_BYTES + 8 * n_rows` bytes, little
    /// endian, deterministic.
    pub fn to_bytes(&self) -> Vec<u8> {
        let (kind, id) = match self.key {
            KeyLane::Field(f) => (0u32, u64::from(f)),
            KeyLane::Computed(tag) => (1u32, tag),
        };
        let n = self.order.len();
        let mut out = Vec::with_capacity(layout::HEADER_BYTES + 8 * n);
        out.extend_from_slice(&layout::MAGIC);
        out.extend_from_slice(&layout::VERSION.to_le_bytes());
        out.extend_from_slice(&TIE_KEY_THEN_FRAME_POSITION.to_le_bytes());
        out.extend_from_slice(&self.origin.class.to_le_bytes());
        out.extend_from_slice(&kind.to_le_bytes());
        out.extend_from_slice(&id.to_le_bytes());
        out.extend_from_slice(&self.origin.population.to_le_bytes());
        out.extend_from_slice(&self.index_population.to_le_bytes());
        out.extend_from_slice(&(n as u64).to_le_bytes());
        out.extend_from_slice(&self.origin.revisions.source.to_le_bytes());
        out.extend_from_slice(&self.origin.revisions.relation_map.to_le_bytes());
        out.extend_from_slice(&self.origin.revisions.coordinates.to_le_bytes());
        out.extend_from_slice(&self.presence.to_le_bytes());
        debug_assert_eq!(out.len(), layout::HEADER_BYTES);
        for &row in &self.order {
            out.extend_from_slice(&row.to_le_bytes());
        }
        for &k in &self.keys {
            out.extend_from_slice(&k.to_le_bytes());
        }
        out
    }

    /// Validate `bytes` in O(n) with no reordering: header, length,
    /// permutation, order law. The header is read first, so a forged row
    /// count never allocates.
    pub fn from_bytes(bytes: &[u8]) -> Result<SealedOrder, StoreRefused> {
        if bytes.len() < layout::HEADER_BYTES {
            return Err(StoreRefused::Length {
                expected: layout::HEADER_BYTES as u64,
                found: bytes.len() as u64,
            });
        }
        if bytes[layout::OFF_MAGIC..layout::OFF_MAGIC + 8] != layout::MAGIC {
            return Err(StoreRefused::BadMagic);
        }
        let version = u32_at(bytes, layout::OFF_VERSION);
        if version != layout::VERSION {
            return Err(StoreRefused::UnknownLayout { version });
        }
        let rule = u32_at(bytes, layout::OFF_TIE_BREAK);
        if rule != TIE_KEY_THEN_FRAME_POSITION {
            return Err(StoreRefused::UnknownTieBreak { rule });
        }
        let kind = u32_at(bytes, layout::OFF_KEY_KIND);
        let id = u64_at(bytes, layout::OFF_KEY_ID);
        if kind > 1 {
            return Err(StoreRefused::UnknownKeyKind { kind });
        }
        if kind == 0 && id > u64::from(u8::MAX) {
            return Err(StoreRefused::UnknownKeyKind { kind });
        }
        let key = if kind == 0 {
            KeyLane::Field(id as u8)
        } else {
            KeyLane::Computed(id)
        };
        let n_rows = u64_at(bytes, layout::OFF_N_ROWS);
        let expected = n_rows
            .checked_mul(8)
            .and_then(|b| b.checked_add(layout::HEADER_BYTES as u64))
            .unwrap_or(u64::MAX);
        if expected != bytes.len() as u64 {
            return Err(StoreRefused::Length {
                expected,
                found: bytes.len() as u64,
            });
        }
        let table_population = u64_at(bytes, layout::OFF_TABLE_POPULATION);
        let index_population = u64_at(bytes, layout::OFF_INDEX_POPULATION);
        if index_population == table_population {
            return Err(StoreRefused::OwnRowOrderClaimed);
        }
        let n = n_rows as usize;
        let rows_at = layout::HEADER_BYTES;
        let keys_at = rows_at + 4 * n;
        let order: Vec<u32> = bytes[rows_at..keys_at]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let keys: Vec<i32> = bytes[keys_at..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect();
        let mut seen = vec![0u64; n.div_ceil(64)];
        for (at, &row) in order.iter().enumerate() {
            let r = row as usize;
            if r >= n || (seen[r / 64] >> (r % 64)) & 1 == 1 {
                return Err(StoreRefused::NotAPermutation { at });
            }
            seen[r / 64] |= 1 << (r % 64);
        }
        for i in 1..n {
            if keys[i - 1] > keys[i] {
                return Err(StoreRefused::OutOfOrder { at: i });
            }
            if keys[i - 1] == keys[i] && order[i - 1] >= order[i] {
                return Err(StoreRefused::OutOfOrder { at: i });
            }
        }
        Ok(SealedOrder {
            order,
            keys,
            index_population,
            origin: Origin {
                class: u32_at(bytes, layout::OFF_CLASS),
                population: table_population,
                n_rows: n,
                revisions: FrameRevisions {
                    source: u64_at(bytes, layout::OFF_SOURCE),
                    relation_map: u64_at(bytes, layout::OFF_RELATION_MAP),
                    coordinates: u64_at(bytes, layout::OFF_COORDINATES),
                },
            },
            key,
            presence: u64_at(bytes, layout::OFF_PRESENCE),
        })
    }

    pub fn write_to(&self, path: &Path) -> Result<(), StoreRefused> {
        std::fs::write(path, self.to_bytes()).map_err(|e| StoreRefused::Io(e.kind()))
    }

    pub fn read_from(path: &Path) -> Result<SealedOrder, StoreRefused> {
        let bytes = std::fs::read(path).map_err(|e| StoreRefused::Io(e.kind()))?;
        SealedOrder::from_bytes(&bytes)
    }

    /// O(1): do the witness fields equal this table, key and presence
    /// version? No lane is read; the cost does not depend on `n_rows`.
    pub fn attest(
        &self,
        table: &Table<'_>,
        key: KeyLane,
        presence: u64,
    ) -> Result<(), StoreRefused> {
        let now = Origin::of(table);
        if now.class != self.origin.class
            || now.population != self.origin.population
            || now.n_rows != self.origin.n_rows
        {
            return Err(StoreRefused::AnotherTable);
        }
        if now.revisions.relation_map != self.origin.revisions.relation_map
            || now.revisions.coordinates != self.origin.revisions.coordinates
        {
            return Err(StoreRefused::MappingChanged);
        }
        if now.revisions.source != self.origin.revisions.source {
            return Err(StoreRefused::SourceChanged {
                sealed: self.origin.revisions.source,
                found: now.revisions.source,
            });
        }
        if key != self.key {
            return Err(StoreRefused::AnotherKey);
        }
        if presence != self.presence {
            return Err(StoreRefused::PresenceChanged {
                sealed: self.presence,
                found: presence,
            });
        }
        Ok(())
    }

    /// `attest`, then hand the permutation to the existing kernel. The
    /// vectors move; nothing is copied or reordered.
    pub fn into_index(
        self,
        table: &Table<'_>,
        key: KeyLane,
        presence: u64,
    ) -> Result<OrderIndex, StoreRefused> {
        self.attest(table, key, presence)?;
        Ok(OrderIndex {
            order: self.order,
            keys: self.keys,
            population: self.index_population,
            origin: self.origin,
        })
    }

    /// Whether this seal holds exactly `index`: permutation, keys, index
    /// population and origin. Read-only; compares lanes by position.
    pub fn matches_index(&self, index: &OrderIndex) -> bool {
        self.order == index.order
            && self.keys == index.keys
            && self.index_population == index.population
            && self.origin == index.origin
    }

    /// Index position -> frame-relative table row.
    pub fn rows(&self) -> &[u32] {
        &self.order
    }

    /// The key at each index position (a derived cache of source content).
    pub fn keys(&self) -> &[i32] {
        &self.keys
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub fn key(&self) -> KeyLane {
        self.key
    }

    pub fn presence(&self) -> u64 {
        self.presence
    }

    pub fn source_revision(&self) -> u64 {
        self.origin.revisions.source
    }
}
