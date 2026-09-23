//! The resident source a report runs over: fixed-width SoA lanes the host
//! already owns.
//!
//! An [`AbiBatch`] is built ONCE, downstream of canonicalization (raw values
//! land in the KV first; see [`crate::boundary`]), and is then only ever
//! BORROWED: every report hands `&[u32]` / `&[i32]` / `&[u64]` views to the
//! evaluator. Nothing here copies a lane, gathers rows, or builds a per-row
//! object; cloning an `Arc<AbiBatch>` clones a pointer.
//!
//! **String-free by construction.** A field is a [`FieldId`], a coordinate is a
//! `u32` ordinal in a declared domain, a raw variable-size value is a `u64`
//! [`ContentId`](lance_graph_contract::content_store::ContentId) KV reference.
//! Names and labels are not stored here at all — they live in the catalog and
//! the CAM label store at the boundary.

use std::sync::Arc;

use lance_graph_mask_risc::{words_for, LaneRef};

use crate::ids::{FieldId, MaskId, SourceId};

/// One resident fixed-width lane.
#[derive(Debug, Clone)]
pub enum LaneData {
    /// Signed 32-bit values (measures, ordered compares, canonical integer time).
    I32(Arc<[i32]>),
    /// Unsigned 32-bit values (coordinate ordinals / label ids, codes, keys).
    U32(Arc<[u32]>),
    /// 64-bit references (KV content addresses, GUIDs). Never folded; carried
    /// so selected rows can be dereferenced at the terminal boundary.
    U64(Arc<[u64]>),
}

impl LaneData {
    fn len(&self) -> usize {
        match self {
            LaneData::I32(v) => v.len(),
            LaneData::U32(v) => v.len(),
            LaneData::U64(v) => v.len(),
        }
    }

    fn as_ref(&self) -> LaneRef<'_> {
        match self {
            LaneData::I32(v) => LaneRef::I32(v),
            LaneData::U32(v) => LaneRef::U32(v),
            LaneData::U64(v) => LaneRef::U64(v),
        }
    }

    fn addr(&self) -> usize {
        match self {
            LaneData::I32(v) => v.as_ptr() as usize,
            LaneData::U32(v) => v.as_ptr() as usize,
            LaneData::U64(v) => v.as_ptr() as usize,
        }
    }
}

/// A resident field. It knows its lane and, if it is a coordinate, how many
/// ordinals its domain has. It never knows what it means or what it is called.
#[derive(Debug, Clone)]
pub struct Column {
    /// Canonical field identity.
    pub field: FieldId,
    /// The resident lane.
    pub lane: LaneData,
    /// Domain size for a `U32` coordinate field (ordinals `0..domain`). An
    /// ordinal at or past it lies outside the domain and belongs to no cell.
    pub domain: Option<u32>,
}

impl Column {
    /// A plain value lane (measure source, filter field, KV reference).
    pub fn value(field: FieldId, lane: LaneData) -> Self {
        Self {
            field,
            lane,
            domain: None,
        }
    }

    /// A `U32` coordinate field over `domain` ordinals.
    pub fn coordinate(field: FieldId, ordinals: Arc<[u32]>, domain: u32) -> Self {
        Self {
            field,
            lane: LaneData::U32(ordinals),
            domain: Some(domain),
        }
    }
}

/// Errors building a batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchError {
    /// A lane's length differs from the batch's row count.
    LenMismatch {
        /// The offending field (`None` for a mask).
        field: Option<FieldId>,
        /// The batch row count (or word count for a mask).
        expected: usize,
        /// What the lane holds.
        found: usize,
    },
    /// A domain was declared on a non-`U32` lane.
    DomainOnNonOrdinalLane(FieldId),
    /// Two columns share a field id.
    DuplicateField(FieldId),
    /// Two masks share an id, or a mask claimed [`MaskId::ALPHA`].
    DuplicateMask(MaskId),
}

/// The resident population a report is evaluated over.
///
/// Plane 0 is always the validity plane ([`MaskId::ALPHA`]): the table IS its
/// validity plane, so "every row" lowers to zero ops.
#[derive(Debug, Clone)]
pub struct AbiBatch {
    source: SourceId,
    generation: u32,
    n_rows: usize,
    columns: Vec<Column>,
    masks: Vec<(MaskId, Arc<[u64]>)>,
}

impl AbiBatch {
    /// A batch over `n_rows` rows whose validity plane is all-set.
    pub fn new(source: SourceId, generation: u32, n_rows: usize) -> Self {
        let mut alpha = vec![u64::MAX; words_for(n_rows)];
        let tail = n_rows % 64;
        if tail != 0 {
            if let Some(last) = alpha.last_mut() {
                *last = (1u64 << tail) - 1;
            }
        }
        Self {
            source,
            generation,
            n_rows,
            columns: Vec::new(),
            masks: vec![(MaskId::ALPHA, alpha.into())],
        }
    }

    /// Attach a column. The lane is moved in (an `Arc`), never copied.
    pub fn with_column(mut self, column: Column) -> Result<Self, BatchError> {
        if column.lane.len() != self.n_rows {
            return Err(BatchError::LenMismatch {
                field: Some(column.field),
                expected: self.n_rows,
                found: column.lane.len(),
            });
        }
        if column.domain.is_some() && !matches!(column.lane, LaneData::U32(_)) {
            return Err(BatchError::DomainOnNonOrdinalLane(column.field));
        }
        if self.columns.iter().any(|c| c.field == column.field) {
            return Err(BatchError::DuplicateField(column.field));
        }
        self.columns.push(column);
        Ok(self)
    }

    /// Attach a RESIDENT mask (a cached facet mask, a focus, a class mask)
    /// supplied by the owner. A report never produces one of these for
    /// another report to consume.
    pub fn with_mask(mut self, id: MaskId, words: Arc<[u64]>) -> Result<Self, BatchError> {
        if words.len() != words_for(self.n_rows) {
            return Err(BatchError::LenMismatch {
                field: None,
                expected: words_for(self.n_rows),
                found: words.len(),
            });
        }
        if self.masks.iter().any(|(m, _)| *m == id) {
            return Err(BatchError::DuplicateMask(id));
        }
        self.masks.push((id, words));
        Ok(self)
    }

    /// The source this batch was published as.
    pub fn source(&self) -> SourceId {
        self.source
    }

    /// Publication generation (a republished source bumps it; a plan minted
    /// against an older generation fails closed).
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Row count.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// The columns, in lane order (column `i` is lane `i`).
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Lane index + column of a field.
    pub fn column(&self, field: FieldId) -> Option<(u16, &Column)> {
        self.columns
            .iter()
            .enumerate()
            .find(|(_, c)| c.field == field)
            .map(|(i, c)| (i as u16, c))
    }

    /// Plane index of a resident mask.
    pub fn plane_of(&self, id: MaskId) -> Option<u16> {
        self.masks
            .iter()
            .position(|(m, _)| *m == id)
            .map(|i| i as u16)
    }

    /// Borrowed plane views for the evaluator: O(columns + masks) pointers.
    pub(crate) fn views(&self) -> (Vec<&[u64]>, Vec<LaneRef<'_>>) {
        (
            self.masks.iter().map(|(_, w)| &w[..]).collect(),
            self.columns.iter().map(|c| c.lane.as_ref()).collect(),
        )
    }

    /// Address of a field's first element — the identity the zero-copy
    /// falsifiers compare before and after a report runs.
    pub fn lane_addr(&self, field: FieldId) -> Option<usize> {
        self.column(field).map(|(_, c)| c.lane.addr())
    }

    /// The `U64` reference lane of a field, for terminal dereference.
    pub fn refs(&self, field: FieldId) -> Option<&[u64]> {
        match self.column(field).map(|(_, c)| &c.lane) {
            Some(LaneData::U64(v)) => Some(v),
            _ => None,
        }
    }
}
