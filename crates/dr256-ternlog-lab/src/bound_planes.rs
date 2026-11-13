//! Eight bit-planes of one bound I32 lane, derived per slab through the
//! slab's `LaneBinding`. Each plane keeps the frame it was derived under and
//! its row order.
//!
//! A lab-local record of the checks `quack_boundary_lab::derived::check_planes`
//! makes (frame against the store, row order against the binding), for a table
//! of more than one slab. Not an authority; nothing here issues anything.
//!
//! Pass conditions: `.claude/board/entries/2026-11-13-wx-calibrated-plane-admission.md`
//! (the "Cost" condition).

use super::bit_planes::{and, gt, lt, plane_op};
use super::{bind_slabs, Plane, Refused, Table, World};
use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_dt64::dr::Frame;
use lance_graph_dt64::{CoordSpace, LaneBinding, LaneRefusal};
use lance_graph_mask_risc::{
    execute_into, fuse_program, words_for, Foreign, LaneRef, Out, Planes, Program, Scratch,
    Terminal, Value,
};

/// One slab: its binding, its row count, and eight planes each with the frame
/// and row order it was derived under.
struct SlabBits {
    binding: LaneBinding,
    rows: usize,
    spaces: [CoordSpace; 8],
    frames: [Frame; 8],
    planes: [Vec<u64>; 8],
}

/// Bit `k` (k in 0..8) of one I32 lane, as plane `k`, for every slab of a
/// table.
pub struct BoundBitPlanes {
    field: u8,
    fields: WideFieldMask,
    slabs: Vec<SlabBits>,
}

impl BoundBitPlanes {
    /// Bind `fields` per slab and slice bit `k` of `field`'s I32 lane into
    /// plane `k`. Each plane is stamped with a frame sealed at derivation and
    /// the slab binding's row order.
    pub fn derive(
        world: &World,
        table: &Table<'_>,
        fields: &WideFieldMask,
        field: u8,
    ) -> Result<BoundBitPlanes, Refused> {
        let mut slabs = Vec::new();
        for (s, (binding, rows)) in bind_slabs(world, table, fields)?.into_iter().enumerate() {
            let source = table.slab(s);
            let mut planes: [Vec<u64>; 8] = std::array::from_fn(|_| vec![0u64; words_for(rows)]);
            let mut found = false;
            for item in binding.borrow_all(&world.store, &world.view, &table.frames, &source)? {
                let (f, lane) = item?;
                if f != field {
                    continue;
                }
                let LaneRef::I32(values) = lane else {
                    return Err(Refused::FieldNotSelected { field });
                };
                for (i, &v) in values.iter().enumerate() {
                    let bits = v as u32;
                    for (k, plane) in planes.iter_mut().enumerate() {
                        if (bits >> k) & 1 == 1 {
                            plane[i / 64] |= 1u64 << (i % 64);
                        }
                    }
                }
                found = true;
            }
            if !found {
                return Err(Refused::FieldNotSelected { field });
            }
            let space = binding.selection();
            slabs.push(SlabBits {
                binding,
                rows,
                spaces: std::array::from_fn(|_| space),
                frames: std::array::from_fn(|_| table.frames.seal()),
                planes,
            });
        }
        Ok(BoundBitPlanes {
            field,
            fields: fields.clone(),
            slabs,
        })
    }

    /// The field whose bits the planes hold.
    pub fn field(&self) -> u8 {
        self.field
    }

    /// Slabs held.
    pub fn slabs(&self) -> usize {
        self.slabs.len()
    }

    /// Bytes of plane words held.
    pub fn bytes(&self) -> usize {
        self.slabs
            .iter()
            .flat_map(|s| s.planes.iter())
            .map(|p| p.len() * 8)
            .sum()
    }

    /// Replace plane `k` of slab `slab` with `with`, words and row order both
    /// (a test seam). The plane keeps the frame it had.
    pub fn set_plane(&mut self, slab: usize, k: usize, with: Plane) {
        self.slabs[slab].planes[k] = with.words;
        self.slabs[slab].spaces[k] = with.space;
    }

    /// The checks for slab `s`: the binding's joint check (every bound lane
    /// borrowed and dropped), then per plane its frame against the table's
    /// store and its row order against the binding. Allocates nothing.
    fn check_slab(&self, s: usize, world: &World, table: &Table<'_>) -> Result<(), Refused> {
        let slab = &self.slabs[s];
        if slab.binding.fields() != &self.fields {
            return Err(Refused::SelectionChanged);
        }
        let source = table.slab(s);
        for item in slab
            .binding
            .borrow_all(&world.store, &world.view, &table.frames, &source)?
        {
            item?;
        }
        let selection = slab.binding.selection();
        for k in 0..8 {
            table
                .frames
                .check(&slab.frames[k])
                .map_err(|why| Refused::Lane(LaneRefusal::Frame(why)))?;
            if slab.spaces[k] != selection {
                return Err(Refused::Misaligned { slab: s, plane: k });
            }
        }
        Ok(())
    }

    /// Execute `program` over slab `s` with no check.
    fn fold_slab(
        &self,
        s: usize,
        program: &Program,
        scratch: &mut Scratch<'_>,
    ) -> Result<u64, Refused> {
        let slab = &self.slabs[s];
        let masks: [&[u64]; 8] = std::array::from_fn(|k| slab.planes[k].as_slice());
        let planes = Planes {
            n_rows: slab.rows,
            masks: &masks,
            lanes: &[],
        };
        let value = execute_into(
            program,
            &planes,
            &Foreign {
                planes: &[],
                lanes: &[],
            },
            scratch,
            Out::None,
        )?;
        match value {
            Value::Count(c) => Ok(c as u64),
            _ => Err(Refused::NotACount),
        }
    }

    /// The checks alone, every slab. Allocates nothing.
    pub fn check(&self, world: &World, table: &Table<'_>) -> Result<(), Refused> {
        for s in 0..self.slabs.len() {
            self.check_slab(s, world, table)?;
        }
        Ok(())
    }

    /// Execute `program` (reads planes 0..=7, no lane) per slab WITHOUT any
    /// check and sum the counts. `scratch` must be sized for one slab.
    pub fn count_unchecked(
        &self,
        program: &Program,
        scratch: &mut Scratch<'_>,
    ) -> Result<u64, Refused> {
        let mut total = 0u64;
        for s in 0..self.slabs.len() {
            total += self.fold_slab(s, program, scratch)?;
        }
        Ok(total)
    }

    /// Check a slab, then fold it, slab by slab; sum the counts.
    pub fn count(
        &self,
        world: &World,
        table: &Table<'_>,
        program: &Program,
        scratch: &mut Scratch<'_>,
    ) -> Result<u64, Refused> {
        let mut total = 0u64;
        for s in 0..self.slabs.len() {
            self.check_slab(s, world, table)?;
            total += self.fold_slab(s, program, scratch)?;
        }
        Ok(total)
    }
}

/// `lo < code < hi` over planes 0..=7 as one fused count program. `None` when
/// no code can satisfy it (`lo == 255` or `hi == 0`).
pub fn range_program(lo: u8, hi: u8) -> Option<Program> {
    let expr = and(gt(lo, plane_op)?, lt(hi, plane_op)?);
    fuse_program(&expr, 0, |mask| Terminal::Count { mask }).ok()
}
