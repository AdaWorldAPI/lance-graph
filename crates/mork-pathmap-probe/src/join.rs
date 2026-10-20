//! A versioned relationship table that puts both inputs of
//! `Q(x, y, z) :- R(x, y), S(y, z)` at one coordinate, so the join runs as a
//! fold.
//!
//! STATUS: PROBE. This is a candidate for the "co-addressed producer" the
//! Schnellverschluss plan (§7) names as the precondition of a native arm. It is
//! not admitted: D-RMAP-1 says a pre-fold global-to-local binding is a candidate
//! membrane that needs its own review, and nothing here is wired into Quack,
//! R2IL or the production mask-risc path.
//!
//! # Layout
//!
//! - [`JoinDomain`]: the sorted distinct values of the shared variable `y`
//!   across both datasets. A value's position is its **slot**. This is the one
//!   global-to-local translation, done once in [`JoinTable::prepare`] by binary
//!   search and timed there.
//! - [`Side`]: one input relation laid out by `y` slot: a presence plane (bit
//!   `y` set when the relation has a row with that `y`), `off[y]..off[y + 1]`,
//!   and the other column's values in that range. Each side is prepared from
//!   its own dataset and carries that dataset's generation, so R and S can come
//!   from different sources and be resealed independently.
//!
//! # Evaluation, and why it is not a gather
//!
//! Leapfrog triejoin with `y` first intersects R's `y` level with S's `y` level,
//! then enumerates `x` and `z` independently (they share no variable). Here the
//! intersection is one mask-risc `And` over the two presence planes, which sit
//! over the same slot space. Enumeration then reads `off[y]` and the value range
//! of each side at the surviving slot `y`: every read is at the coordinate the
//! mask names, in every operand. The `x` and `z` values are emitted, never used
//! as addresses. No operand is read at a position taken from another operand's
//! values (2026-10-16 ruling).
//!
//! Multiplicities are kept, so this arm answers [`Semantics::Bag`] as well as
//! [`Semantics::Set`]; both upstream indexes are sets.

use lance_graph_mask_risc::{
    Foreign, MaskOp, Operand, Out, Planes, Program, Scratch, Terminal, execute_into, words_for,
};

use crate::{Answers, RelationView, Semantics};

/// Why the table did not answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JoinRefusal {
    /// A side was prepared from another generation than the caller holds now.
    Stale {
        side: usize,
        bound: u64,
        current: u64,
    },
    /// A re-prepared side has a `y` the domain does not contain. The domain is
    /// the shared coordinate space; changing it re-addresses every side, so the
    /// whole table must be prepared again.
    OutsideDomain { side: usize, y: u32 },
    /// The filter plane is not over this table's slot space.
    FilterShape { words: usize, expected: usize },
}

/// The shared coordinate space: sorted distinct `y` values.
#[derive(Clone, Debug)]
pub struct JoinDomain {
    ids: Vec<u32>,
}

impl JoinDomain {
    /// The union of every side's `y` values.
    pub fn build(ys: impl IntoIterator<Item = u32>) -> Self {
        let mut ids: Vec<u32> = ys.into_iter().collect();
        ids.sort_unstable();
        ids.dedup();
        JoinDomain { ids }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Global `y` to slot. Preparation only.
    pub fn slot_of(&self, y: u32) -> Option<usize> {
        self.ids.binary_search(&y).ok()
    }

    /// Slot to global `y`, for emitting answers.
    pub fn id(&self, slot: usize) -> u32 {
        self.ids[slot]
    }

    pub fn bytes(&self) -> usize {
        self.ids.len() * 4
    }
}

/// One relation laid out by `y` slot.
#[derive(Clone, Debug)]
pub struct Side {
    pub generation: u64,
    present: Vec<u64>,
    off: Vec<u32>,
    vals: Vec<u32>,
}

impl Side {
    /// Lays `rows` out by the slot of column `key`; the other column is the
    /// value. A counting sort over slots, so the value order within one slot
    /// is the input order.
    pub fn prepare(
        domain: &JoinDomain,
        rows: &[[u32; 2]],
        key: usize,
        generation: u64,
        side: usize,
    ) -> Result<Side, JoinRefusal> {
        let n = domain.len();
        let mut slots = Vec::with_capacity(rows.len());
        let mut off = vec![0u32; n + 1];
        for row in rows {
            let y = row[key];
            let s = domain
                .slot_of(y)
                .ok_or(JoinRefusal::OutsideDomain { side, y })?;
            slots.push(s as u32);
            off[s + 1] += 1;
        }
        for i in 0..n {
            off[i + 1] += off[i];
        }
        let mut fill = off.clone();
        let mut vals = vec![0u32; rows.len()];
        for (row, &s) in rows.iter().zip(&slots) {
            let at = &mut fill[s as usize];
            vals[*at as usize] = row[1 - key];
            *at += 1;
        }
        let mut present = vec![0u64; words_for(n)];
        for s in 0..n {
            if off[s + 1] > off[s] {
                present[s >> 6] |= 1 << (s & 63);
            }
        }
        Ok(Side {
            generation,
            present,
            off,
            vals,
        })
    }

    /// The values at slot `y`.
    pub fn at(&self, y: usize) -> &[u32] {
        &self.vals[self.off[y] as usize..self.off[y + 1] as usize]
    }

    pub fn present(&self) -> &[u64] {
        &self.present
    }

    pub fn bytes(&self) -> usize {
        (self.present.len() * 8) + (self.off.len() * 4) + (self.vals.len() * 4)
    }
}

/// The versioned relationship table: one domain, one side per relation, each
/// side tied to its own dataset generation.
#[derive(Clone, Debug)]
pub struct JoinTable {
    pub domain: JoinDomain,
    pub sides: Vec<Side>,
}

/// Where the shared variable sits in each input row: `R(x, y)` keys column 1,
/// `S(y, z)` keys column 0.
pub const R_KEY: usize = 1;
pub const S_KEY: usize = 0;

impl JoinTable {
    /// Prepares the table for `R(x, y), S(y, z)`.
    pub fn prepare(r: RelationView<'_>, s: RelationView<'_>) -> JoinTable {
        Self::prepare_star(&[(r, R_KEY), (s, S_KEY)])
    }

    /// Prepares a star of relations that all share `y`: every relation is laid
    /// out over the same domain. `key` names the column holding `y`.
    pub fn prepare_star(inputs: &[(RelationView<'_>, usize)]) -> JoinTable {
        let domain = JoinDomain::build(
            inputs
                .iter()
                .flat_map(|(v, key)| v.rows.iter().map(move |r| r[*key])),
        );
        let sides = inputs
            .iter()
            .enumerate()
            .map(|(i, (v, key))| {
                Side::prepare(&domain, v.rows, *key, v.generation, i).expect("domain is the union")
            })
            .collect();
        JoinTable { domain, sides }
    }

    /// Re-prepares one side from a resealed dataset, keeping the domain and the
    /// other sides. Refuses if the new rows leave the domain.
    pub fn replace_side(
        &mut self,
        side: usize,
        view: RelationView<'_>,
        key: usize,
    ) -> Result<(), JoinRefusal> {
        self.sides[side] = Side::prepare(&self.domain, view.rows, key, view.generation, side)?;
        Ok(())
    }

    /// Refuses unless every side's generation is the one the caller holds.
    pub fn check(&self, current: &[u64]) -> Result<(), JoinRefusal> {
        for (i, (side, &now)) in self.sides.iter().zip(current).enumerate() {
            if side.generation != now {
                return Err(JoinRefusal::Stale {
                    side: i,
                    bound: side.generation,
                    current: now,
                });
            }
        }
        Ok(())
    }

    pub fn bytes(&self) -> usize {
        self.domain.bytes() + self.sides.iter().map(Side::bytes).sum::<usize>()
    }

    pub fn words(&self) -> usize {
        words_for(self.domain.len())
    }
}

/// Runs the presence-plane intersection through mask-risc. Owns its output
/// plane and scratch, so repeated queries allocate nothing in the fold.
pub struct JoinExec {
    out: Vec<u64>,
    scratch: Scratch<'static>,
}

impl JoinExec {
    pub fn new(table: &JoinTable) -> Self {
        let words = table.words();
        JoinExec {
            out: vec![0; words],
            scratch: Scratch::new(words, 1),
        }
    }

    /// `joint = present_0 ∧ … ∧ present_k [∧ filter]`: the slots where every
    /// relation has a row (and the thought admits the slot).
    pub fn joint(
        &mut self,
        table: &JoinTable,
        filter: Option<&[u64]>,
    ) -> Result<&[u64], JoinRefusal> {
        let words = table.words();
        if let Some(f) = filter
            && f.len() != words
        {
            return Err(JoinRefusal::FilterShape {
                words: f.len(),
                expected: words,
            });
        }
        assert!(table.sides.len() >= 2, "a join needs two relations");
        let mut masks: Vec<&[u64]> = table.sides.iter().map(|s| s.present()).collect();
        if let Some(f) = filter {
            masks.push(f);
        }
        let mut ops = Vec::with_capacity(masks.len());
        let mut acc = Operand::Plane(0);
        for i in 1..masks.len() {
            ops.push(MaskOp::And {
                a: acc,
                b: Operand::Plane(i as u16),
                dst: 0,
            });
            acc = Operand::Scratch(0);
        }
        let prog = Program::new(
            ops,
            Terminal::Keep {
                mask: Operand::Scratch(0),
            },
        );
        let pl = Planes {
            n_rows: table.domain.len(),
            masks: &masks,
            lanes: &[],
        };
        execute_into(
            &prog,
            &pl,
            &Foreign::NONE,
            &mut self.scratch,
            Out::Mask(&mut self.out),
        )
        .expect("join program is valid");
        Ok(&self.out)
    }

    /// The answers of `R(x, y), S(y, z)`, enumerated at the joint slots.
    pub fn answers(
        &mut self,
        table: &JoinTable,
        current: (u64, u64),
        filter: Option<&[u64]>,
        semantics: Semantics,
    ) -> Result<Answers, JoinRefusal> {
        table.check(&[current.0, current.1])?;
        let n = table.domain.len();
        let joint = self.joint(table, filter)?;
        let (r, s) = (&table.sides[0], &table.sides[1]);
        let mut out = Answers::default();
        for_each_set(joint, n, |slot| {
            let y = table.domain.id(slot);
            for &x in r.at(slot) {
                for &z in s.at(slot) {
                    *out.rows.entry([x, y, z]).or_insert(0) += 1;
                }
            }
        });
        Ok(match semantics {
            Semantics::Bag => out,
            Semantics::Set => out.to_set(),
        })
    }

    /// Every bag answer `[x, y, z]`, handed to `f` in slot order, without
    /// building a map: the join's own cost, separated from the output's.
    pub fn for_each_answer(
        &mut self,
        table: &JoinTable,
        current: (u64, u64),
        mut f: impl FnMut([u32; 3]),
    ) -> Result<(), JoinRefusal> {
        table.check(&[current.0, current.1])?;
        let n = table.domain.len();
        let joint = self.joint(table, None)?;
        let (r, s) = (&table.sides[0], &table.sides[1]);
        for_each_set(joint, n, |slot| {
            let y = table.domain.id(slot);
            for &x in r.at(slot) {
                for &z in s.at(slot) {
                    f([x, y, z]);
                }
            }
        });
        Ok(())
    }

    /// The number of bag answers of the star, `Σ_y Π_i |side_i at y|`, without
    /// enumerating a row.
    pub fn bag_count(
        &mut self,
        table: &JoinTable,
        current: &[u64],
        filter: Option<&[u64]>,
    ) -> Result<u64, JoinRefusal> {
        table.check(current)?;
        let n = table.domain.len();
        let joint = self.joint(table, filter)?;
        let mut total = 0u64;
        for_each_set(joint, n, |slot| {
            total += table
                .sides
                .iter()
                .map(|s| s.at(slot).len() as u64)
                .product::<u64>();
        });
        Ok(total)
    }
}

/// Calls `f` for every set bit below `n`, in slot order.
pub fn for_each_set(words: &[u64], n: usize, mut f: impl FnMut(usize)) {
    for (wi, &w) in words.iter().enumerate() {
        let mut w = w;
        while w != 0 {
            let b = w.trailing_zeros() as usize;
            let slot = wi * 64 + b;
            if slot < n {
                f(slot);
            }
            w &= w - 1;
        }
    }
}

/// A filter plane over the table's slots from a predicate on the global `y`.
/// Built per thought, from the domain, never from the sides.
pub fn filter_plane(table: &JoinTable, mut keep: impl FnMut(u32) -> bool) -> Vec<u64> {
    let n = table.domain.len();
    let mut p = vec![0u64; words_for(n)];
    for slot in 0..n {
        if keep(table.domain.id(slot)) {
            p[slot >> 6] |= 1 << (slot & 63);
        }
    }
    p
}
