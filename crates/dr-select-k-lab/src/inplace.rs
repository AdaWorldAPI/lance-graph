//! The arm without staging: one selection bound to source positions, folded
//! over the source's own lanes.
//!
//! [`crate::Bound`] copies K rows of every lane and folds over the copy. This
//! binding copies no value. It keeps the position table it was given, one bit
//! per source row, and, when the positions are one ascending run, that run's
//! bounds. Each fold reads the source lanes where they are:
//!
//! - [`InPlace::count_b_gt`] / [`InPlace::sum_a`] through the N-bit plane
//!   (mask-RISC over N rows, program and scratch prepared once);
//! - the same two over a contiguous run (mask-RISC over the K rows of the
//!   run, a sub-slice of each lane);
//! - the same two through the position table (a scalar loop; no admitted
//!   executor folds over a position list, plan §11).
//!
//! Every fold is handed the source and refuses a version other than the one
//! the binding was made at. Nothing here reads the selection again.

use crate::counters::Counters;
use crate::ops::{count_b_gt_program, run, set_threshold, sum_a_program};
use crate::source::{Selection, Source, Version};
use lance_graph_mask_risc::{words_for, Program, Scratch, Value};

/// Why an in-place bind or fold refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InPlaceRefusal {
    /// The source is at another version than the binding.
    Stale { bound: Version, source: Version },
    /// The fold needs a contiguous run and the positions are not one.
    NotARun,
}

/// One selection bound to source positions. No lane value is copied.
pub struct InPlace {
    version: Version,
    rows: Vec<u32>,
    /// One bit per source row.
    plane: Vec<u64>,
    /// `Some((lo, len))` when `rows` is `lo, lo + 1, ..`.
    run: Option<(usize, usize)>,
    count_over_n: Program,
    sum_over_n: Program,
    count_over_run: Program,
    scratch_n: Scratch<'static>,
    scratch_run: Scratch<'static>,
    /// All ones over the run, for the sum (mask-RISC has no unmasked sum).
    ones: Vec<u64>,
}

impl InPlace {
    /// Bind `selection`. Builds the N-bit plane (counted as one conversion and
    /// its bytes); touches no lane.
    pub fn bind(
        source: &Source,
        selection: Selection,
        counters: &Counters,
    ) -> Result<InPlace, InPlaceRefusal> {
        if selection.version != source.version() {
            return Err(InPlaceRefusal::Stale {
                bound: selection.version,
                source: source.version(),
            });
        }
        let n = source.n_rows();
        let rows = selection.rows;
        let run = rows
            .first()
            .filter(|_| rows.windows(2).all(|p| p[1] == p[0] + 1))
            .map(|&lo| (lo as usize, rows.len()));
        let mut plane = vec![0u64; words_for(n)];
        match run {
            // A run is a range of bits: whole words in the middle.
            Some((lo, len)) => {
                let hi = lo + len;
                for (w, word) in plane
                    .iter_mut()
                    .enumerate()
                    .take(hi.div_ceil(64))
                    .skip(lo / 64)
                {
                    let (from, to) = ((w * 64).max(lo), ((w + 1) * 64).min(hi));
                    let width = to - from;
                    let bits = if width == 64 {
                        u64::MAX
                    } else {
                        (1u64 << width) - 1
                    };
                    *word = bits << (from % 64);
                }
            }
            None => {
                for &r in &rows {
                    plane[r as usize / 64] |= 1 << (r % 64);
                }
            }
        }
        Counters::bump(&counters.binds, 1);
        Counters::bump(&counters.conversions, 1);
        Counters::bump(&counters.bytes_copied, (plane.len() * 8) as u64);
        let k = run.map_or(0, |(_, len)| len);
        let count_over_n = count_b_gt_program(0, true);
        let count_over_run = count_b_gt_program(0, false);
        let mut ones = vec![u64::MAX; words_for(k)];
        if !k.is_multiple_of(64) {
            if let Some(last) = ones.last_mut() {
                *last = (1u64 << (k % 64)) - 1;
            }
        }
        Ok(InPlace {
            version: selection.version,
            scratch_n: Scratch::for_program(&count_over_n, n.max(1))
                .expect("scratch for a one-slot lab program (lab bug)"),
            scratch_run: Scratch::for_program(&count_over_run, k.max(1))
                .expect("scratch for a one-slot lab program (lab bug)"),
            rows,
            plane,
            run,
            count_over_n,
            sum_over_n: sum_a_program(),
            count_over_run,
            ones,
        })
    }

    pub fn k(&self) -> usize {
        self.rows.len()
    }

    /// Whether the positions are one ascending run of source rows.
    pub fn is_run(&self) -> bool {
        self.run.is_some()
    }

    /// Bytes the binding retains beside the position table it was given.
    pub fn retained_bytes(&self) -> usize {
        (self.plane.len() + self.ones.len()) * 8
    }

    fn fresh(&self, source: &Source) -> Result<(), InPlaceRefusal> {
        if source.version() == self.version {
            Ok(())
        } else {
            Err(InPlaceRefusal::Stale {
                bound: self.version,
                source: source.version(),
            })
        }
    }

    /// `count(selected AND b > t)` over the N source rows, gated by the plane.
    pub fn count_b_gt(
        &mut self,
        source: &Source,
        t: i32,
        counters: &Counters,
    ) -> Result<u64, InPlaceRefusal> {
        self.fresh(source)?;
        let n = source.n_rows();
        Counters::bump(&counters.operand_rows, n as u64);
        if n == 0 {
            return Ok(0);
        }
        let (a, b) = source.operand_lanes();
        set_threshold(&mut self.count_over_n, t);
        Ok(
            match run(
                &self.count_over_n,
                &mut self.scratch_n,
                n,
                Some(&self.plane),
                a,
                b,
            ) {
                Value::Count(c) => c as u64,
                _ => unreachable!("Count terminal returns Value::Count"),
            },
        )
    }

    /// `sum(a)` over the selected source rows, through the plane.
    pub fn sum_a(&mut self, source: &Source, counters: &Counters) -> Result<i64, InPlaceRefusal> {
        self.fresh(source)?;
        let n = source.n_rows();
        Counters::bump(&counters.operand_rows, n as u64);
        if n == 0 {
            return Ok(0);
        }
        let (a, b) = source.operand_lanes();
        Ok(
            match run(
                &self.sum_over_n,
                &mut self.scratch_n,
                n,
                Some(&self.plane),
                a,
                b,
            ) {
                Value::SumI64(s) => s,
                _ => unreachable!("MaskedSumI32 returns Value::SumI64"),
            },
        )
    }

    /// `count(b > t)` over the run: the K rows of each lane, as a sub-slice.
    pub fn run_count_b_gt(
        &mut self,
        source: &Source,
        t: i32,
        counters: &Counters,
    ) -> Result<u64, InPlaceRefusal> {
        self.fresh(source)?;
        let (lo, k) = self.run.ok_or(InPlaceRefusal::NotARun)?;
        Counters::bump(&counters.operand_rows, k as u64);
        let (a, b) = source.operand_lanes();
        set_threshold(&mut self.count_over_run, t);
        Ok(
            match run(
                &self.count_over_run,
                &mut self.scratch_run,
                k,
                None,
                &a[lo..lo + k],
                &b[lo..lo + k],
            ) {
                Value::Count(c) => c as u64,
                _ => unreachable!("Count terminal returns Value::Count"),
            },
        )
    }

    /// `sum(a)` over the run.
    pub fn run_sum_a(
        &mut self,
        source: &Source,
        counters: &Counters,
    ) -> Result<i64, InPlaceRefusal> {
        self.fresh(source)?;
        let (lo, k) = self.run.ok_or(InPlaceRefusal::NotARun)?;
        Counters::bump(&counters.operand_rows, k as u64);
        let (a, b) = source.operand_lanes();
        Ok(
            match run(
                &self.sum_over_n,
                &mut self.scratch_run,
                k,
                Some(&self.ones),
                &a[lo..lo + k],
                &b[lo..lo + k],
            ) {
                Value::SumI64(s) => s,
                _ => unreachable!("MaskedSumI32 returns Value::SumI64"),
            },
        )
    }

    /// `count(b > t)` through the position table: a scalar loop.
    pub fn via_positions_count_b_gt(
        &self,
        source: &Source,
        t: i32,
        counters: &Counters,
    ) -> Result<u64, InPlaceRefusal> {
        self.fresh(source)?;
        Counters::bump(&counters.operand_rows, self.rows.len() as u64);
        let (_, b) = source.operand_lanes();
        Ok(self.rows.iter().filter(|&&r| b[r as usize] > t).count() as u64)
    }

    /// `sum(a)` through the position table: a scalar loop.
    pub fn via_positions_sum_a(
        &self,
        source: &Source,
        counters: &Counters,
    ) -> Result<i64, InPlaceRefusal> {
        self.fresh(source)?;
        Counters::bump(&counters.operand_rows, self.rows.len() as u64);
        let (a, _) = source.operand_lanes();
        Ok(self.rows.iter().map(|&r| i64::from(a[r as usize])).sum())
    }
}
