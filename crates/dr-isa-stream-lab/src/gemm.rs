//! ARM B: the cube's 16 x 16 answer as a BF16 tile GEMM.
//!
//! `C[p][m] = sum_i lane_p[i] * mask_m(i)`: M = 16 demanded lanes, N = 16
//! masks, K = identities in blocks of 32. B is the selection, a 0/1 matrix
//! staged straight into the VNNI layout from the mask words; it is derived
//! from the WHERE result, never source data.
//!
//! A is the source's lanes. When 16 demanded lanes are consecutive rows of
//! one slab, A is read in place through
//! `bf16_tile_gemm_16x16_packed_strided`, with the slab pitch as the row
//! stride. Otherwise each block's 16 x 32 A is copied into a staging tile,
//! and [`Report::a_staged_bytes`] says how much was copied.

use lance_graph_dt64::{LaneBinding, LaneRefusal};
use ndarray::simd::{
    bf16_tile_gemm_16x16_packed, bf16_tile_gemm_16x16_packed_strided, bf16_tile_gemm_tier,
    PackedBf16B,
};

use crate::bind::World;
use crate::cube::{check_masks, ArmRefusal, WhereMasks};
use crate::fixture::{BLOCK, MASKS, ROWS};
use crate::source::{SlabLane, Socket};

/// bf16 1.0.
const ONE: u16 = 0x3F80;

/// How A reaches the tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// In place from the slab; refuse if the geometry does not allow it.
    Direct,
    /// Always through a staging tile.
    Staged,
}

/// Why A cannot be read in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectRefusal {
    /// Fewer than 16 demanded lanes in this row tile.
    PartialRowTile { lanes: usize },
    /// The lane is not in the same allocation as the first lane of its tile.
    NotOneSlab { field: u8 },
    /// The lane's pitch differs from the first lane's.
    PitchMismatch { field: u8 },
    /// The lane is not the row after the previous lane.
    RowsNotConsecutive { field: u8 },
    /// The pitch does not cover the last block with padding: a 32-wide read
    /// would run into the next row instead of zeros.
    ShortPitch { pitch: usize, needed: usize },
}

/// Why ARM B refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemmRefusal {
    Arm(ArmRefusal),
    Direct(DirectRefusal),
}

impl From<ArmRefusal> for GemmRefusal {
    fn from(r: ArmRefusal) -> Self {
        GemmRefusal::Arm(r)
    }
}

impl From<LaneRefusal> for GemmRefusal {
    fn from(r: LaneRefusal) -> Self {
        GemmRefusal::Arm(ArmRefusal::Lane(r))
    }
}

/// What ARM B did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    /// The GEMM tier that ran.
    pub tier: &'static str,
    /// Row tiles read in place.
    pub direct_tiles: usize,
    /// Row tiles staged.
    pub staged_tiles: usize,
    /// Bytes of A copied into staging tiles.
    pub a_staged_bytes: usize,
    /// Nonzero B entries written (each is written once and cleared once).
    pub b_entries: usize,
    /// GEMM calls.
    pub gemms: usize,
}

/// In-place geometry of one row tile: A's row `r` is
/// `slab[(first_row + r) * pitch..]`.
struct Geometry<'a> {
    slab: &'a [u16],
    first_row: usize,
    pitch: usize,
}

fn geometry<'a>(fields: &[u8], lanes: &[SlabLane<'a>]) -> Result<Geometry<'a>, DirectRefusal> {
    if lanes.len() != ROWS {
        return Err(DirectRefusal::PartialRowTile { lanes: lanes.len() });
    }
    let first = lanes[0];
    for (k, (lane, &field)) in lanes.iter().zip(fields).enumerate().skip(1) {
        if !std::ptr::eq(lane.slab, first.slab) {
            return Err(DirectRefusal::NotOneSlab { field });
        }
        if lane.pitch != first.pitch {
            return Err(DirectRefusal::PitchMismatch { field });
        }
        if lane.row != first.row + k {
            return Err(DirectRefusal::RowsNotConsecutive { field });
        }
    }
    let needed = first.n.div_ceil(BLOCK) * BLOCK;
    if first.pitch < needed {
        return Err(DirectRefusal::ShortPitch {
            pitch: first.pitch,
            needed,
        });
    }
    Ok(Geometry {
        slab: first.slab,
        first_row: first.row,
        pitch: first.pitch,
    })
}

/// Stage block `t` of the selection into `b`: `B[j][m] = mask_m(32t + j)`.
/// Returns the number of entries written.
fn stage_b(masks: &WhereMasks, t: usize, b: &mut PackedBf16B, value: u16) -> usize {
    let mut written = 0;
    for m in 0..MASKS {
        let word = masks.words(m)[t / 2];
        let mut bits = (word >> ((t % 2) * 32)) & 0xFFFF_FFFF;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            b.data_mut()[PackedBf16B::vnni_index(j, m)] = value;
            written += 1;
            bits &= bits - 1;
        }
    }
    written
}

/// `SUM(field) WHERE mask m` for every listed field and mask, as tile GEMMs.
/// Fields are taken 16 at a time; a final partial group is staged.
pub fn sums(
    world: &World,
    binding: &LaneBinding,
    socket: &Socket<'_>,
    masks: &WhereMasks,
    fields: &[u8],
    mode: Mode,
) -> Result<(Vec<[f32; MASKS]>, Report), GemmRefusal> {
    check_masks(binding, masks)?;
    let mut lanes = Vec::with_capacity(fields.len());
    for &f in fields {
        lanes.push(binding.lane(&world.store, &world.view, socket.lanes.frames(), socket, f)?);
    }
    let groups: Vec<(&[u8], &[SlabLane<'_>])> =
        fields.chunks(ROWS).zip(lanes.chunks(ROWS)).collect();
    let mut plans = Vec::with_capacity(groups.len());
    for (fs, ls) in &groups {
        plans.push(match (mode, geometry(fs, ls)) {
            (Mode::Direct, Ok(g)) => Some(g),
            (Mode::Direct, Err(e)) => return Err(GemmRefusal::Direct(e)),
            (Mode::Staged, _) => None,
        });
    }

    let mut report = Report {
        tier: bf16_tile_gemm_tier(),
        ..Report::default()
    };
    let mut c = vec![[0f32; ROWS * MASKS]; groups.len()];
    let mut b = PackedBf16B::zeroed(BLOCK);
    let mut a = vec![0u16; ROWS * BLOCK];
    for t in 0..masks.n.div_ceil(BLOCK) {
        let written = stage_b(masks, t, &mut b, ONE);
        report.b_entries += written;
        if written != 0 {
            for (g, plan) in plans.iter().enumerate() {
                match plan {
                    Some(geo) => {
                        let start = geo.first_row * geo.pitch + t * BLOCK;
                        bf16_tile_gemm_16x16_packed_strided(
                            &geo.slab[start..],
                            geo.pitch,
                            &b,
                            &mut c[g],
                        );
                    }
                    None => {
                        a.fill(0);
                        for (r, lane) in groups[g].1.iter().enumerate() {
                            let values = lane.values();
                            let lo = t * BLOCK;
                            let hi = (lo + BLOCK).min(values.len());
                            a[r * BLOCK..r * BLOCK + (hi - lo)].copy_from_slice(&values[lo..hi]);
                            report.a_staged_bytes += (hi - lo) * 2;
                        }
                        bf16_tile_gemm_16x16_packed(&a, &b, &mut c[g]);
                    }
                }
                report.gemms += 1;
            }
            stage_b(masks, t, &mut b, 0);
        }
    }
    for plan in &plans {
        if plan.is_some() {
            report.direct_tiles += 1;
        } else {
            report.staged_tiles += 1;
        }
    }
    let mut out = Vec::with_capacity(fields.len());
    for (g, (fs, _)) in groups.iter().enumerate() {
        for r in 0..fs.len() {
            let mut row = [0f32; MASKS];
            row.copy_from_slice(&c[g][r * MASKS..(r + 1) * MASKS]);
            out.push(row);
        }
    }
    Ok((out, report))
}

/// The selection as one prepacked B over the whole population:
/// `B[i][m] = mask_m(i)` for `i < n`, zero to the padded K. It depends only
/// on the masks, so it is staged once and reused by every row tile and every
/// query over the same masks. It is a numeric realization of the WHERE
/// result, tied to the masks' coordinate space; it is never source data.
pub struct SelectionB {
    pub space: lance_graph_dt64::CoordSpace,
    pub n: usize,
    b: PackedBf16B,
}

impl SelectionB {
    /// Stage the masks in VNNI layout: a sequential write of every entry.
    pub fn stage(masks: &WhereMasks) -> SelectionB {
        let k = masks.n.div_ceil(BLOCK) * BLOCK;
        let mut b = PackedBf16B::zeroed(k);
        let data = b.data_mut();
        for m in 0..MASKS {
            for (w, &word) in masks.words(m).iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let i = w * 64 + bits.trailing_zeros() as usize;
                    data[PackedBf16B::vnni_index(i, m)] = ONE;
                    bits &= bits - 1;
                }
            }
        }
        SelectionB {
            space: masks.space,
            n: masks.n,
            b,
        }
    }

    /// Bytes the staged selection occupies.
    pub fn bytes(&self) -> usize {
        self.b.data().len() * 2
    }
}

/// Like [`sums`], over a prepacked [`SelectionB`]: one GEMM per 16-lane row
/// tile with K = the padded population. In [`Mode::Direct`] A is each tile's
/// 16 lanes read in place at the slab pitch; in [`Mode::Staged`] they are
/// first copied into a dense 16 x K buffer.
pub fn sums_whole(
    world: &World,
    binding: &LaneBinding,
    socket: &Socket<'_>,
    sel: &SelectionB,
    fields: &[u8],
    mode: Mode,
) -> Result<(Vec<[f32; MASKS]>, Report), GemmRefusal> {
    if sel.space != binding.selection() {
        return Err(GemmRefusal::Arm(ArmRefusal::MaskSpace {
            masks: sel.space,
            selection: binding.selection(),
        }));
    }
    let k = sel.b.k();
    let mut report = Report {
        tier: bf16_tile_gemm_tier(),
        ..Report::default()
    };
    let mut out = Vec::with_capacity(fields.len());
    let mut dense = Vec::new();
    for fs in fields.chunks(ROWS) {
        let mut lanes = Vec::with_capacity(fs.len());
        for &f in fs {
            lanes.push(binding.lane(
                &world.store,
                &world.view,
                socket.lanes.frames(),
                socket,
                f,
            )?);
        }
        let mut c = [0f32; ROWS * MASKS];
        match (mode, geometry(fs, &lanes)) {
            (Mode::Direct, Ok(geo)) => {
                let start = geo.first_row * geo.pitch;
                bf16_tile_gemm_16x16_packed_strided(&geo.slab[start..], geo.pitch, &sel.b, &mut c);
                report.direct_tiles += 1;
            }
            (Mode::Direct, Err(e)) => return Err(GemmRefusal::Direct(e)),
            (Mode::Staged, _) => {
                dense.clear();
                dense.resize(ROWS * k, 0u16);
                for (r, lane) in lanes.iter().enumerate() {
                    let values = lane.values();
                    dense[r * k..r * k + values.len()].copy_from_slice(values);
                    report.a_staged_bytes += values.len() * 2;
                }
                bf16_tile_gemm_16x16_packed(&dense, &sel.b, &mut c);
                report.staged_tiles += 1;
            }
        }
        report.gemms += 1;
        for r in 0..fs.len() {
            let mut row = [0f32; MASKS];
            row.copy_from_slice(&c[r * MASKS..(r + 1) * MASKS]);
            out.push(row);
        }
    }
    Ok((out, report))
}
