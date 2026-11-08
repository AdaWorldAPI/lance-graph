//! ARM A: 16 WHERE masks over a 256 x 256 identity cube, folded in place.
//!
//! Identity `i` sits at `tile(i) = i >> 8`, `slot(i) = i & 255`. A mask is one
//! bit per identity in the binding's row order, so 16 masks over 65,536
//! identities are 16 x 1,024 words. The fold reads each demanded lane once,
//! at the positions the mask words name; nothing is copied out.

use lance_graph_dt64::{CoordSpace, LaneBinding, LaneRefusal};

use crate::bind::World;
use crate::fixture::{bf16_value, Fixture, MASKS};
use crate::source::Socket;

/// Identities per cube tile.
pub const TILE: usize = 256;

/// Cube tile of identity `i`.
pub const fn tile(i: usize) -> usize {
    i >> 8
}

/// Slot of identity `i` within its tile.
pub const fn slot(i: usize) -> usize {
    i & 255
}

/// The 16 WHERE masks, in one coordinate space.
pub struct WhereMasks {
    pub space: CoordSpace,
    pub n: usize,
    words: Vec<Box<[u64]>>,
}

impl WhereMasks {
    /// The fixture's masks, declared to be in `space`.
    pub fn from_fixture(fx: &Fixture, space: CoordSpace) -> WhereMasks {
        let len = fx.n.div_ceil(64);
        let words = (0..MASKS)
            .map(|m| {
                let mut w = vec![0u64; len].into_boxed_slice();
                for i in 0..fx.n {
                    if fx.selected(i, m) {
                        w[i / 64] |= 1 << (i % 64);
                    }
                }
                w
            })
            .collect();
        WhereMasks {
            space,
            n: fx.n,
            words,
        }
    }

    /// Mask `m`'s words.
    pub fn words(&self, m: usize) -> &[u64] {
        &self.words[m]
    }

    /// Selected identities per mask (a popcount terminal).
    pub fn counts(&self) -> [u32; MASKS] {
        let mut c = [0u32; MASKS];
        for (m, slot) in c.iter_mut().enumerate() {
            *slot = self.words[m].iter().map(|w| w.count_ones()).sum();
        }
        c
    }
}

/// Why an arm refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArmRefusal {
    /// The binding refused (frame, space, scope, route, ...).
    Lane(LaneRefusal),
    /// The masks are in another row order than the binding's selection.
    MaskSpace {
        masks: CoordSpace,
        selection: CoordSpace,
    },
    /// The graph's rows and columns are in another row order than the
    /// binding's selection.
    GraphSpace {
        graph: CoordSpace,
        selection: CoordSpace,
    },
}

impl From<LaneRefusal> for ArmRefusal {
    fn from(r: LaneRefusal) -> Self {
        ArmRefusal::Lane(r)
    }
}

/// Refuse masks whose row order is not the binding's.
pub fn check_masks(binding: &LaneBinding, masks: &WhereMasks) -> Result<(), ArmRefusal> {
    if masks.space == binding.selection() {
        Ok(())
    } else {
        Err(ArmRefusal::MaskSpace {
            masks: masks.space,
            selection: binding.selection(),
        })
    }
}

/// `SUM(field) WHERE mask m` for every listed field and every mask, by a
/// positional fold over the cube: tile by tile, each mask's four words name
/// the selected slots, and the lane is read at exactly those positions.
pub fn fold(
    world: &World,
    binding: &LaneBinding,
    socket: &Socket<'_>,
    masks: &WhereMasks,
    fields: &[u8],
) -> Result<Vec<[i64; MASKS]>, ArmRefusal> {
    check_masks(binding, masks)?;
    let mut out = Vec::with_capacity(fields.len());
    for &f in fields {
        let lane = binding.lane(&world.store, &world.view, socket.lanes.frames(), socket, f)?;
        let values = lane.values();
        let mut row = [0i64; MASKS];
        for t in 0..masks.n.div_ceil(TILE) {
            for (m, slot) in row.iter_mut().enumerate() {
                let words = masks.words(m);
                let lo = t * 4;
                let hi = (lo + 4).min(words.len());
                for (w, &word) in words[lo..hi].iter().enumerate() {
                    let w = lo + w;
                    let mut bits = word;
                    while bits != 0 {
                        let i = w * 64 + bits.trailing_zeros() as usize;
                        *slot += bf16_value(values[i]) as i64;
                        bits &= bits - 1;
                    }
                }
            }
        }
        out.push(row);
    }
    Ok(out)
}
