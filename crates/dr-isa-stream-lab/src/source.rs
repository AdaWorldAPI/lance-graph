//! Source-owned bf16 lanes and the DT64 socket over them.
//!
//! A [`SlabLane`] is a borrowed view that says where the lane sits: which
//! allocation, which row, at which pitch. DT64 hands out lane addresses; the
//! physical geometry stays with the source and its lane type, so DR carries
//! no ISA knowledge.

use std::cell::Cell;

use lance_graph_contract::ontology::FieldRef;
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{CoordSpace, LaneAddr, LaneSource, SchemaRev};

use crate::bind::{CLASS, READER, SPACE_A};
use crate::fixture::{bf16_exact, Fixture, BLOCK};

/// One borrowed lane: `n` bf16 values at `slab[row * pitch..][..n]`.
#[derive(Debug, Clone, Copy)]
pub struct SlabLane<'a> {
    pub slab: &'a [u16],
    pub row: usize,
    pub pitch: usize,
    pub n: usize,
}

impl<'a> SlabLane<'a> {
    /// The lane's values, in the lane's own row order.
    pub fn values(&self) -> &'a [u16] {
        &self.slab[self.row * self.pitch..self.row * self.pitch + self.n]
    }
}

/// Why a source could not be built from the fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestRefusal {
    /// The value cannot be held exactly in bf16.
    NotBf16Exact {
        identity: usize,
        prop: usize,
        value: i32,
    },
}

fn frames() -> FrameStore {
    FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    })
}

/// How a source lays out its lanes.
#[derive(Debug)]
enum Layout {
    /// One allocation; lane `r` at `data[r * pitch..]`, `pitch >= n` and a
    /// multiple of [`BLOCK`], padding zeroed.
    Slab { data: Box<[u16]>, pitch: usize },
    /// One allocation per lane.
    Scattered(Vec<Box<[u16]>>),
}

/// The data owner: bf16 lanes, the frame store that lives with them, and the
/// field-to-row map (`rows[f]` is the physical row holding field `f`).
#[derive(Debug)]
pub struct Bf16Lanes {
    layout: Layout,
    n: usize,
    frames: FrameStore,
}

impl Bf16Lanes {
    /// Lanes in one slab, field `f` stored at physical row `rows[f]`.
    pub fn slab(fx: &Fixture, rows: &[usize]) -> Result<Bf16Lanes, IngestRefusal> {
        Bf16Lanes::slab_with_pitch(fx, rows, fx.n.div_ceil(BLOCK) * BLOCK)
    }

    /// Lanes in one slab at an explicit `pitch >= n`, with no guarantee that
    /// the pitch is padded to a [`BLOCK`] multiple.
    pub fn slab_with_pitch(
        fx: &Fixture,
        rows: &[usize],
        pitch: usize,
    ) -> Result<Bf16Lanes, IngestRefusal> {
        assert!(pitch >= fx.n, "pitch {pitch} shorter than n {}", fx.n);
        let physical = rows.iter().max().map_or(0, |&r| r + 1);
        let mut data = vec![0u16; physical * pitch].into_boxed_slice();
        for (p, &r) in rows.iter().enumerate() {
            fill(fx, p, &mut data[r * pitch..r * pitch + fx.n])?;
        }
        Ok(Bf16Lanes {
            layout: Layout::Slab { data, pitch },
            n: fx.n,
            frames: frames(),
        })
    }

    /// `props` lanes over `n` identities in one slab at the padded pitch, in
    /// field order, from already-encoded bf16 values (for inputs that were
    /// rounded on purpose; the fixture path refuses inexact values).
    pub fn slab_encoded(n: usize, props: usize, value: impl Fn(usize, usize) -> u16) -> Bf16Lanes {
        let pitch = n.div_ceil(BLOCK) * BLOCK;
        let mut data = vec![0u16; props * pitch].into_boxed_slice();
        for p in 0..props {
            for i in 0..n {
                data[p * pitch + i] = value(i, p);
            }
        }
        Bf16Lanes {
            layout: Layout::Slab { data, pitch },
            n,
            frames: frames(),
        }
    }

    /// Lanes in one slab, field `f` at row `f`.
    pub fn slab_in_order(fx: &Fixture) -> Result<Bf16Lanes, IngestRefusal> {
        let rows: Vec<usize> = (0..fx.props).collect();
        Bf16Lanes::slab(fx, &rows)
    }

    /// Every lane in its own allocation.
    pub fn scattered(fx: &Fixture) -> Result<Bf16Lanes, IngestRefusal> {
        let mut lanes = Vec::with_capacity(fx.props);
        for p in 0..fx.props {
            let mut lane = vec![0u16; fx.n].into_boxed_slice();
            fill(fx, p, &mut lane)?;
            lanes.push(lane);
        }
        Ok(Bf16Lanes {
            layout: Layout::Scattered(lanes),
            n: fx.n,
            frames: frames(),
        })
    }

    /// The frame owner of this data.
    pub fn frames(&self) -> &FrameStore {
        &self.frames
    }

    /// Mutable frame owner, for advancing the source after a change.
    pub fn frames_mut(&mut self) -> &mut FrameStore {
        &mut self.frames
    }

    /// Mutable view of physical row `r` (to model a source-side change).
    pub fn row_mut(&mut self, r: usize) -> &mut [u16] {
        let n = self.n;
        match &mut self.layout {
            Layout::Slab { data, pitch } => &mut data[r * *pitch..r * *pitch + n],
            Layout::Scattered(lanes) => &mut lanes[r],
        }
    }

    /// Physical row `r`, borrowed in place.
    pub fn row(&self, r: usize) -> Option<SlabLane<'_>> {
        match &self.layout {
            Layout::Slab { data, pitch } => {
                (r * pitch + self.n <= data.len()).then_some(SlabLane {
                    slab: data,
                    row: r,
                    pitch: *pitch,
                    n: self.n,
                })
            }
            Layout::Scattered(lanes) => lanes.get(r).map(|lane| SlabLane {
                slab: lane,
                row: 0,
                pitch: self.n,
                n: self.n,
            }),
        }
    }

    /// Bytes this source holds.
    pub fn resident_bytes(&self) -> usize {
        match &self.layout {
            Layout::Slab { data, .. } => data.len() * 2,
            Layout::Scattered(lanes) => lanes.iter().map(|l| l.len() * 2).sum(),
        }
    }
}

fn fill(fx: &Fixture, p: usize, lane: &mut [u16]) -> Result<(), IngestRefusal> {
    for (i, slot) in lane.iter_mut().enumerate() {
        let value = fx.value(i, p);
        *slot = bf16_exact(value).ok_or(IngestRefusal::NotBf16Exact {
            identity: i,
            prop: p,
            value,
        })?;
    }
    Ok(())
}

/// The DT64 socket over a [`Bf16Lanes`]: field `f` lives in physical row
/// `rows[f]`, in row order `spaces[f]`. Counts what was borrowed.
pub struct Socket<'a> {
    pub lanes: &'a Bf16Lanes,
    pub rows: Vec<usize>,
    pub spaces: Vec<CoordSpace>,
    pub scope: Scope,
    pub borrows: Cell<u64>,
}

impl<'a> Socket<'a> {
    /// Field `f` at row `rows[f]`, every lane in [`SPACE_A`].
    pub fn new(lanes: &'a Bf16Lanes, rows: Vec<usize>) -> Socket<'a> {
        let spaces = vec![SPACE_A; rows.len()];
        Socket {
            lanes,
            rows,
            spaces,
            scope: READER,
            borrows: Cell::new(0),
        }
    }

    /// Field `f` at row `f`.
    pub fn in_order(lanes: &'a Bf16Lanes, props: usize) -> Socket<'a> {
        Socket::new(lanes, (0..props).collect())
    }
}

impl LaneSource for Socket<'_> {
    type Lane<'b>
        = SlabLane<'b>
    where
        Self: 'b;

    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        let f = usize::from(field);
        if class != CLASS || rev != SchemaRev(1) || f >= self.rows.len() {
            return None;
        }
        Some(LaneAddr {
            space: self.spaces[f],
            lane: u16::try_from(self.rows[f]).ok()?,
            scope: self.scope,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<SlabLane<'_>> {
        self.borrows.set(self.borrows.get() + 1);
        self.lanes.row(usize::from(addr.lane))
    }

    fn frames(&self) -> &FrameStore {
        self.lanes.frames()
    }
}
