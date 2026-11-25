//! D-PM-TILE (EXO-398): P×M sufficient statistics folded per sealed DR tile
//! and combined across tiles.
//!
//! A Codec9 field is one lane of `u8` magnitudes with its presence plane, the
//! polarity bits (`abakus::pm`): untouched is `P = 0 ∧ M = 0`, which is also
//! the absent rule of the lane socket. An untouched row has `M = 0`, so it
//! adds nothing to `S = Σx`, `Q = Σx²` or `A = Σ|x|`; only the count `N`
//! reads `touched`. Padding a tile to a whole block is therefore neutral.
//!
//! The four statistics add exactly, so a population split into tiles folds
//! each tile on its own and sums the parts. What has to be witnessed is that
//! a part still describes its tile: a [`TilePart`] keeps the frame its lane
//! was borrowed under, and [`combine`] refuses a part whose tile's store has
//! moved, or that is presented with another tile. Nothing is decoded, copied
//! or stored per row; a part is four numbers and a frame.
//!
//! Lab only. No public contract; nothing depends on this crate.

use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_dt64::dr::{Frame, FrameRefusal, FrameStore};
use lance_graph_dt64::{
    BindStore, LaneBinding, LaneRefusal, LaneRequest, LaneSource, PresenceSource,
};
use ndarray::hpc::abakus::pm::{touched, Pm};
use ndarray::simd::{U16x32, U8x64};

pub mod fixture;

/// `N` touched selected values, `S = Σx`, `Q = Σx²`, `A = Σ|x|`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub n: u64,
    pub s: i64,
    pub q: u64,
    pub a: u64,
}

impl Stats {
    /// The sum of two parts, or `None` if any field would overflow.
    pub fn checked_add(self, o: Stats) -> Option<Stats> {
        Some(Stats {
            n: self.n.checked_add(o.n)?,
            s: self.s.checked_add(o.s)?,
            q: self.q.checked_add(o.q)?,
            a: self.a.checked_add(o.a)?,
        })
    }

    /// Population variance as the exact rational `(N·Q − S², N²)`; `None`
    /// for an empty population, whose variance is undefined, not zero.
    pub fn variance_parts(self) -> Option<(i128, i128)> {
        if self.n == 0 {
            return None;
        }
        let n = i128::from(self.n);
        let s = i128::from(self.s);
        Some((n * i128::from(self.q) - s * s, n * n))
    }
}

/// Why a lane could not be folded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FoldRefusal {
    /// The lane, its plane and the selection do not describe the same rows:
    /// the lane must be whole 64-row blocks, one plane word and one selection
    /// word per block.
    Shape {
        rows: usize,
        plane: usize,
        sel: usize,
    },
    /// The socket refused the read.
    Lane(LaneRefusal),
    /// The binding has no lane for this field.
    NoField { field: u8 },
}

/// Sum of the magnitudes in `mask`'s lanes.
#[inline(always)]
fn msum(mask: u64, m: U8x64) -> u64 {
    U8x64::mask_blend(mask, U8x64::splat(0), m).sum_bytes_u64()
}

/// `Σ a_i²` over 64 lanes, exact: a square is at most 255² and fits a `u16`;
/// each square is split into its two bytes so the byte sums stay in range.
#[inline(always)]
fn square_sum(a: U8x64) -> u64 {
    let lo = U16x32::from_u8x64_lo(a);
    let hi = U16x32::from_u8x64_hi(a);
    let (lo, hi) = (lo.mullo(lo), hi.mullo(hi));
    let (lo_h, hi_h) = (lo.shr(8), hi.shr(8));
    let (lo_l, hi_l) = (lo - lo_h.shl(8), hi - hi_h.shl(8));
    lo_l.pack_saturate_u8(hi_l).sum_bytes_u64() + (lo_h.pack_saturate_u8(hi_h).sum_bytes_u64() << 8)
}

/// Fold one Codec9 lane: `mags` are the magnitudes, `plane` the polarity
/// words, `sel` the selected rows, all in the lane's row order.
pub fn fold_lane(mags: &[u8], plane: &[u64], sel: &[u64]) -> Result<Stats, FoldRefusal> {
    if !mags.len().is_multiple_of(64) || mags.len() / 64 != plane.len() || plane.len() != sel.len()
    {
        return Err(FoldRefusal::Shape {
            rows: mags.len(),
            plane: plane.len(),
            sel: sel.len(),
        });
    }
    let mut st = Stats::default();
    for ((block, &pos), &m) in mags.as_chunks::<64>().0.iter().zip(plane).zip(sel) {
        let x = Pm {
            mag: U8x64::from_slice(&block[..]),
            pos,
        };
        st.n += u64::from((m & touched(x)).count_ones());
        let a = msum(m, x.mag);
        st.a += a;
        st.s += a as i64 - 2 * msum(m & !x.pos, x.mag) as i64;
        st.q += square_sum(U8x64::mask_blend(m, U8x64::splat(0), x.mag));
    }
    Ok(st)
}

/// One tile's statistics and the frame its lane was borrowed under. Made
/// only by [`fold_tile`]; read through [`combine`].
#[derive(Clone, Debug)]
pub struct TilePart {
    stats: Stats,
    frame: Frame,
}

impl TilePart {
    /// The statistics, without the witness. For reporting; a caller that
    /// wants a combined answer goes through [`combine`].
    pub fn stats_unwitnessed(&self) -> Stats {
        self.stats
    }
}

/// Fold `field` of a tile through its binding: one `check_bound`, then the
/// lane and its plane borrowed in place.
pub fn fold_tile<S>(
    binding: &LaneBinding,
    store: &BindStore,
    frames: &FrameStore,
    source: &S,
    field: u8,
    sel: &[u64],
) -> Result<TilePart, FoldRefusal>
where
    S: for<'a> PresenceSource<Lane<'a> = &'a [u8]>,
{
    let lanes = binding
        .borrow_all_present_bound(store, frames, source)
        .map_err(FoldRefusal::Lane)?;
    for lane in lanes {
        let (f, mags, plane) = lane.map_err(FoldRefusal::Lane)?;
        if f == field {
            return Ok(TilePart {
                stats: fold_lane(mags, plane, sel)?,
                frame: binding.frame().clone(),
            });
        }
    }
    Err(FoldRefusal::NoField { field })
}

/// Why parts could not be combined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineRefusal {
    /// Part `tile` is not its tile's current seal: the tile moved since the
    /// fold, or the part belongs to another tile.
    Frame { tile: usize, refusal: FrameRefusal },
    /// Adding part `tile` would overflow a statistic.
    Overflow { tile: usize },
}

/// Sum the parts, each checked against the store of the tile it is
/// presented with. Order does not change the answer.
pub fn combine(parts: &[(&TilePart, &FrameStore)]) -> Result<Stats, CombineRefusal> {
    let mut total = Stats::default();
    for (tile, (part, frames)) in parts.iter().enumerate() {
        if !cfg!(feature = "disable-combine-frame-check") {
            frames
                .check(&part.frame)
                .map_err(|refusal| CombineRefusal::Frame { tile, refusal })?;
        }
        total = total
            .checked_add(part.stats)
            .ok_or(CombineRefusal::Overflow { tile })?;
    }
    Ok(total)
}

/// Bind `field` of `class` at `route` over a tile source.
pub fn bind_field<V, S>(
    store: &BindStore,
    view: &V,
    source: &S,
    route: lance_graph_dt64::RouteAt,
    space: lance_graph_dt64::CoordSpace,
    scope: lance_graph_dt64::dr::Scope,
    field: u8,
) -> Result<LaneBinding, LaneRefusal>
where
    V: lance_graph_contract::class_view::ClassView + ?Sized,
    S: LaneSource,
{
    let frame = source.frames().seal();
    store.bind_lanes(
        view,
        source.frames(),
        source,
        LaneRequest {
            route,
            rev: lance_graph_dt64::SchemaRev(1),
            fields: &WideFieldMask::from_positions(&[field]),
            frame: &frame,
            scope,
            selection: space,
        },
    )
}
