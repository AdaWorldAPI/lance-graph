//! D-NFR-1: a dt64 frame on a bound query; mapping validity apart from content
//! witness (seam S5 x S6 of `d-native-repr-contract-v1.md` §4).
//!
//! A bound query does not know which mapping it was bound under. This module
//! pairs any bound query `Q` with a store-issued dt64 [`Frame`] and one tile
//! of dt64's [`MapTable`], and admits or refuses each later use by the
//! artifact's real dependencies. There are two classes:
//!
//! - **A pure address binding** (the bound query itself, [`FramedBinding`])
//!   depends on the mapping only: the tile's layout and the relation-map
//!   revision. It stays admitted across a content change and is refused when
//!   either mapping dependency moves ([`FramedBinding::admit`]).
//! - **A content-dependent preparation** (a kept WHERE mask, [`Kept`])
//!   additionally depends on the content it was computed from. It is refused,
//!   or recomputed, after a content change, and is never reused because the
//!   coordinates are unchanged ([`Kept::read`], [`Kept::read_or_recompute`]).
//!
//! No store, cache or map of bindings is added: the mapping is
//! [`MapTable`], and every value defined here is caller-owned.
//!
//! # Limits, stated
//!
//! - The content witness is the owner-declared `source` revision at table
//!   granularity. A change to an unrelated lane of the same table also
//!   refuses a kept preparation: safe, coarse.
//! - An owner that mutates content without advancing its table is outside
//!   what this lab can detect (NFR-OP-1 stays open).
//! - A pure extension of the tile (an id appended, every old position kept)
//!   is refused here. That is a conservative lab choice; prefix admission is
//!   left to S7.
//! - The codebook generation (D-NFR-2) is not part of this slice. `Q` may
//!   later be Quack's `TypedQuery<W>`; this module is generic over `Q` and
//!   names dt64 only.
//! - Authorization is not checked here. It is independent and belongs at the
//!   read boundary.
use std::sync::{Arc, Weak};

use lance_graph_dt64::dr::{Frame, FrameRefusal, MapRead, MapTable, TileId, TileMap};

/// The relation-map revision the consumer's Binder says it resolved names and
/// codes under (`d-native-repr-contract-v1.md` §4, first bullet).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedUnder {
    /// The relation-map revision the Binder resolved under.
    pub relation_map: u64,
}

/// Why a framed binding or a kept preparation was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nfr1Refusal {
    /// The table did not issue the frame (`Stale(ForeignFrame)`), or holds no
    /// such tile at the bound or at the current version (`NoSuchTile`).
    Read(MapRead),
    /// A dependency moved. Only three values are produced here:
    /// `MapChanged` (relation map), `CoordinatesChanged` (this tile's layout),
    /// `SourceChanged` (content; kept preparations only).
    Frame(FrameRefusal),
    /// `seal`: the Binder resolved under another relation map than the
    /// table's.
    BinderNotFrame,
    /// A kept preparation offered to a binding that did not produce it.
    NotThisBinding,
}

/// A bound query with the frame and tile it was bound under. Not `Clone`: the
/// issue marker identifies this binding.
#[derive(Debug)]
pub struct FramedBinding<Q> {
    query: Q,
    frame: Frame,
    tile: TileId,
    issue: Arc<()>,
}

/// The only way to reach the query for execution: the query and the tile
/// layout it was admitted under.
#[derive(Debug)]
pub struct Admitted<'b, 't, Q> {
    query: &'b Q,
    layout: TileMap<'t>,
}

/// A content-dependent preparation, stamped by the binding that produced it.
/// No public constructor; made only by [`FramedBinding::keep`].
#[derive(Debug)]
pub struct Kept<P> {
    value: P,
    binding: Weak<()>,
    produced: Frame,
    tile: TileId,
}

/// Whether [`Kept::read_or_recompute`] served the held value or made a new one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reuse {
    /// The held value was still valid.
    Reused,
    /// The content moved; the value was produced again.
    Recomputed,
}

impl<Q> FramedBinding<Q> {
    /// Bind `query` to the table's current frame and `tile`.
    ///
    /// Refuses a tile the table does not hold, and a Binder that resolved
    /// under another relation map than the table's.
    pub fn seal(
        table: &MapTable,
        tile: TileId,
        resolved: ResolvedUnder,
        query: Q,
    ) -> Result<Self, Nfr1Refusal> {
        let frame = table.seal();
        table.map(&frame, tile).map_err(Nfr1Refusal::Read)?;
        if resolved.relation_map != frame.revisions().relation_map {
            return Err(Nfr1Refusal::BinderNotFrame);
        }
        Ok(Self {
            query,
            frame,
            tile,
            issue: Arc::new(()),
        })
    }

    /// The frame this binding was sealed under.
    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    /// The tile this binding addresses.
    pub fn tile(&self) -> TileId {
        self.tile
    }

    /// Mapping replay: the tile's layout at the bound version. Never a
    /// currency check; a frame of this table always replays.
    pub fn replay<'t>(&self, table: &'t MapTable) -> Result<TileMap<'t>, Nfr1Refusal> {
        table.map(&self.frame, self.tile).map_err(Nfr1Refusal::Read)
    }

    /// Admit the pure address binding: the mapping dependencies and nothing
    /// else. The content (`source`), the frame version and the whole-frame
    /// `coordinates` digest are deliberately not compared; the tile layout is
    /// compared as an exact slice of at most 256 ids.
    pub fn admit<'b, 't>(
        &'b self,
        table: &'t MapTable,
    ) -> Result<Admitted<'b, 't, Q>, Nfr1Refusal> {
        let bound = table
            .map(&self.frame, self.tile)
            .map_err(Nfr1Refusal::Read)?;
        if self.frame.revisions().relation_map != table.revisions().relation_map {
            return Err(Nfr1Refusal::Frame(FrameRefusal::MapChanged));
        }
        let now = table
            .map(&table.seal(), self.tile)
            .map_err(Nfr1Refusal::Read)?;
        if !(bound.width() == now.width() && bound.globals() == now.globals()) {
            return Err(Nfr1Refusal::Frame(FrameRefusal::CoordinatesChanged));
        }
        Ok(Admitted {
            query: &self.query,
            layout: now,
        })
    }

    /// Produce a content-dependent preparation under an admitted binding and
    /// stamp it with this binding and the table's current frame.
    pub fn keep<'t, P>(
        &self,
        table: &'t MapTable,
        produce: impl FnOnce(&Q, TileMap<'t>) -> P,
    ) -> Result<Kept<P>, Nfr1Refusal> {
        let a = self.admit(table)?;
        Ok(Kept {
            value: produce(a.query, a.layout),
            binding: Arc::downgrade(&self.issue),
            produced: table.seal(),
            tile: self.tile,
        })
    }
}

impl<'b, 't, Q> Admitted<'b, 't, Q> {
    /// The bound query.
    pub fn query(&self) -> &'b Q {
        self.query
    }

    /// The tile layout the query was admitted under.
    pub fn layout(&self) -> TileMap<'t> {
        self.layout
    }
}

impl<P> Kept<P> {
    /// Every refusal that applies to a kept preparation, in order.
    fn check<Q>(&self, binding: &FramedBinding<Q>, table: &MapTable) -> Result<(), Nfr1Refusal> {
        if !Weak::ptr_eq(&self.binding, &Arc::downgrade(&binding.issue)) {
            return Err(Nfr1Refusal::NotThisBinding);
        }
        binding.admit(table)?;
        table
            .map(&self.produced, self.tile)
            .map_err(Nfr1Refusal::Read)?;
        if self.produced.revisions().source != table.revisions().source {
            return Err(Nfr1Refusal::Frame(FrameRefusal::SourceChanged));
        }
        Ok(())
    }

    /// The held value, if it still depends on nothing that moved. Every
    /// mapping refusal of [`FramedBinding::admit`] applies too.
    pub fn read<Q>(&self, binding: &FramedBinding<Q>, table: &MapTable) -> Result<&P, Nfr1Refusal> {
        self.check(binding, table)?;
        Ok(&self.value)
    }

    /// [`Kept::read`], producing the value again when, and only when, the
    /// content witness moved. Every other refusal is returned unchanged and
    /// `produce` is not called: a mapping change needs a re-bind, never a
    /// recompute under the stale binding.
    pub fn read_or_recompute<'t, Q>(
        &mut self,
        binding: &FramedBinding<Q>,
        table: &'t MapTable,
        produce: impl FnOnce(&Q, TileMap<'t>) -> P,
    ) -> Result<(&P, Reuse), Nfr1Refusal> {
        match self.check(binding, table) {
            Ok(()) => return Ok((&self.value, Reuse::Reused)),
            Err(Nfr1Refusal::Frame(FrameRefusal::SourceChanged)) => {}
            Err(e) => return Err(e),
        }
        let a = binding.admit(table)?;
        self.value = produce(a.query(), a.layout());
        self.produced = table.seal();
        Ok((&self.value, Reuse::Recomputed))
    }
}
