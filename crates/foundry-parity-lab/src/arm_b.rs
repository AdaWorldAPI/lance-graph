//! ARM B: the virtual operating model.
//!
//! `source-owned planes → DT64 LaneBinding (cold, once) → borrow_all (one
//! check) → mask-RISC programs (per question)`. Every answer is a mask
//! program over borrowed planes; nothing is copied and no per-store object
//! exists. Positions become store identities only by reading the bound
//! `storeId` lane at the set bits of a result mask.
//!
//! What this arm can NOT answer, by construction of the admitted algebra
//! (mask-RISC terminals reduce over rows; there is no element-wise cross-lane
//! arithmetic): the per-store limiting resource (an argmin across lanes) and
//! the per-store cost (a per-row weighted sum across lanes). It answers the
//! per-store maximum only through a layer-cake of feasibility masks, which is
//! lawful because feasibility is monotone in the quantity; the sweep checks
//! that monotonicity instead of assuming it.

use std::cell::RefCell;

use lance_graph_contract::class_view::ClassView;
use lance_graph_dt64::dr::FrameStore;
use lance_graph_dt64::{BindStore, LaneBinding, LaneRefusal};
use lance_graph_mask_risc::{
    execute, tile_words_for, ExecError, LaneRef, MaskOp, Operand, Planes, Pred, Program, Scratch,
    Terminal, Value,
};

use crate::bind::{COST0, EQUIP0, KNOWN0, MINUTES, OFFERED0, SLOTS, STAFF0, STOCK0, STORE_ID};
use crate::fixture::{self as fx, Recipe, DENIED_STORE, STORES, STORE_ID_BASE};
use crate::source::Plane;

/// Quantities above this are refused rather than swept.
pub const Q_MAX: i32 = 128;

/// The fields a question about item `m` demands. Nothing else is bound.
pub fn demanded_fields(m: usize, cost: bool, delivery: bool) -> Vec<u8> {
    let r = fx::recipe(m);
    let mut f = Vec::new();
    for &(k, _) in &r.demand {
        f.push(STOCK0 + k);
        f.push(KNOWN0 + k);
        if cost {
            f.push(COST0 + k);
        }
    }
    f.extend([
        STAFF0 + r.skill,
        EQUIP0 + r.equipment,
        OFFERED0 + m,
        STORE_ID,
    ]);
    if delivery {
        f.extend([SLOTS, MINUTES]);
    }
    f.sort_unstable();
    f.into_iter().map(|x| x as u8).collect()
}

/// The guards a program applies. Every guard is on in a real answer; the
/// negative controls turn exactly one off to show it is load-bearing.
#[derive(Debug, Clone, Copy)]
pub struct Guards {
    /// Stores the reader is denied are excluded (IAM stand-in).
    pub authorization: bool,
    /// A demanded category with unknown stock excludes the store.
    pub known: bool,
}

impl Guards {
    pub const ALL: Guards = Guards {
        authorization: true,
        known: true,
    };
}

/// One store's answer as this arm produces it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Answer {
    NotAuthorized,
    Unknown,
    Known { feasible: bool, max: i32 },
}

/// Why this arm could not answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArmRefusal {
    Lane(LaneRefusal),
    Exec(ExecError),
    /// A bound plane had another kind than its field requires.
    PlaneKind {
        field: u8,
    },
    /// A demanded field was not part of the binding.
    NotBound {
        field: u8,
    },
    /// Feasibility at `q` was not a subset of feasibility at `q - 1`.
    NotMonotone {
        q: i32,
    },
    /// Some store is still feasible at [`Q_MAX`].
    Ceiling,
    /// `q` outside `1..=Q_MAX`.
    Quantity {
        q: i32,
    },
}

impl From<LaneRefusal> for ArmRefusal {
    fn from(r: LaneRefusal) -> Self {
        ArmRefusal::Lane(r)
    }
}

impl From<ExecError> for ArmRefusal {
    fn from(e: ExecError) -> Self {
        ArmRefusal::Exec(e)
    }
}

/// Where a bound field sits in the mask-RISC planes.
#[derive(Debug, Clone, Copy)]
enum Slot {
    Mask(u16),
    Lane(u16),
}

/// The pointer table for one item: borrowed planes, made once per binding.
pub struct ItemQuery<'s> {
    item: usize,
    recipe: Recipe,
    masks: Vec<&'s [u64]>,
    lanes: Vec<LaneRef<'s>>,
    slots: [Option<Slot>; 256],
    ids: &'s [u32],
    scratch: RefCell<Scratch<'static>>,
}

impl<'s> ItemQuery<'s> {
    /// Check the binding once and take every bound lane (`borrow_all`).
    pub fn open<V: ClassView + ?Sized, S>(
        item: usize,
        binding: &LaneBinding,
        store: &BindStore,
        view: &V,
        frames: &FrameStore,
        source: &'s S,
    ) -> Result<ItemQuery<'s>, ArmRefusal>
    where
        S: for<'a> lance_graph_dt64::LaneSource<Lane<'a> = Plane<'a>>,
    {
        let mut masks = Vec::new();
        let mut lanes = Vec::new();
        let mut slots = [None; 256];
        let mut ids = None;
        for lane in binding.borrow_all(store, view, frames, source)? {
            let (field, plane) = lane?;
            slots[usize::from(field)] = Some(match plane {
                Plane::Mask(m) => {
                    masks.push(m);
                    Slot::Mask((masks.len() - 1) as u16)
                }
                Plane::I32(v) => {
                    lanes.push(LaneRef::I32(v));
                    Slot::Lane((lanes.len() - 1) as u16)
                }
                Plane::U32(v) => {
                    if usize::from(field) == STORE_ID {
                        ids = Some(v);
                    }
                    lanes.push(LaneRef::U32(v));
                    Slot::Lane((lanes.len() - 1) as u16)
                }
            });
        }
        let ids = ids.ok_or(ArmRefusal::NotBound {
            field: STORE_ID as u8,
        })?;
        Ok(ItemQuery {
            item,
            recipe: fx::recipe(item),
            masks,
            lanes,
            slots,
            ids,
            scratch: RefCell::new(Scratch::new(tile_words_for(STORES), 2)),
        })
    }

    fn mask(&self, f: usize) -> Result<Operand, ArmRefusal> {
        match self.slots[f] {
            Some(Slot::Mask(i)) => Ok(Operand::Plane(i)),
            Some(Slot::Lane(_)) => Err(ArmRefusal::PlaneKind { field: f as u8 }),
            None => Err(ArmRefusal::NotBound { field: f as u8 }),
        }
    }

    fn lane(&self, f: usize) -> Result<u16, ArmRefusal> {
        match self.slots[f] {
            Some(Slot::Lane(i)) => Ok(i),
            Some(Slot::Mask(_)) => Err(ArmRefusal::PlaneKind { field: f as u8 }),
            None => Err(ArmRefusal::NotBound { field: f as u8 }),
        }
    }

    /// Ops leaving `CanCook(item, q)` in scratch 0.
    fn cook_ops(&self, q: i32, g: Guards) -> Result<Vec<MaskOp>, ArmRefusal> {
        if !(1..=Q_MAX).contains(&q) {
            return Err(ArmRefusal::Quantity { q });
        }
        let r = &self.recipe;
        let acc = Operand::Scratch(0);
        let tmp = Operand::Scratch(1);
        let and = |b| MaskOp::And { a: acc, b, dst: 0 };
        let mut ops = Vec::with_capacity(4 + 3 * r.demand.len());
        let offered = self.mask(OFFERED0 + self.item)?;
        let equip = self.mask(EQUIP0 + r.equipment)?;
        if g.authorization {
            ops.push(self.visible(0)?);
            ops.push(and(offered));
        } else {
            ops.push(MaskOp::And {
                a: offered,
                b: offered,
                dst: 0,
            });
        }
        ops.push(and(equip));
        for &(k, d) in &r.demand {
            if g.known {
                ops.push(and(self.mask(KNOWN0 + k)?));
            }
            ops.push(MaskOp::Pred {
                pred: Pred::GeI32 {
                    lane: self.lane(STOCK0 + k)?,
                    t: d * q,
                },
                under: None,
                dst: 1,
            });
            ops.push(and(tmp));
        }
        ops.push(MaskOp::Pred {
            pred: Pred::GeI32 {
                lane: self.lane(STAFF0 + r.skill)?,
                t: (q + r.per_staff - 1) / r.per_staff,
            },
            under: None,
            dst: 1,
        });
        ops.push(and(tmp));
        Ok(ops)
    }

    /// The IAM stand-in: the reader is denied exactly [`DENIED_STORE`],
    /// matched on the global identity lane, so it holds in any row order.
    fn visible(&self, dst: u16) -> Result<MaskOp, ArmRefusal> {
        Ok(MaskOp::Pred {
            pred: Pred::NeU32 {
                lane: self.lane(STORE_ID)?,
                v: DENIED_STORE,
            },
            under: None,
            dst,
        })
    }

    fn run(&self, ops: Vec<MaskOp>, terminal: Terminal) -> Result<Value, ArmRefusal> {
        let program = Program::new(ops, terminal);
        let planes = Planes {
            n_rows: STORES,
            masks: &self.masks,
            lanes: &self.lanes,
        };
        let mut scratch = self.scratch.borrow_mut();
        Ok(execute(&program, &planes, &mut scratch, None)?)
    }

    fn keep(&self, ops: Vec<MaskOp>) -> Result<u64, ArmRefusal> {
        match self.run(
            ops,
            Terminal::Keep {
                mask: Operand::Scratch(0),
            },
        )? {
            Value::Mask(_) => Ok(self.scratch.borrow().slot(0).map_or(0, |s| s[0])),
            _ => unreachable!("Keep yields a mask"),
        }
    }

    /// `CanCook(item, q)` over the whole network: bit `r` = physical row `r`.
    pub fn cook_mask(&self, q: i32, g: Guards) -> Result<u64, ArmRefusal> {
        self.keep(self.cook_ops(q, g)?)
    }

    /// Stores the reader is denied.
    pub fn denied_mask(&self) -> Result<u64, ArmRefusal> {
        // 64 rows fill the one mask word exactly (fixture asserts it).
        Ok(!self.keep(vec![self.visible(0)?])?)
    }

    /// Visible stores where some demanded category's stock is unknown.
    pub fn unknown_mask(&self) -> Result<u64, ArmRefusal> {
        let mut ops = vec![self.visible(0)?];
        let first = self.mask(KNOWN0 + self.recipe.demand[0].0)?;
        ops.push(MaskOp::And {
            a: first,
            b: first,
            dst: 1,
        });
        for &(k, _) in &self.recipe.demand[1..] {
            ops.push(MaskOp::And {
                a: Operand::Scratch(1),
                b: self.mask(KNOWN0 + k)?,
                dst: 1,
            });
        }
        ops.push(MaskOp::AndNot {
            a: Operand::Scratch(0),
            b: Operand::Scratch(1),
            dst: 0,
        });
        self.keep(ops)
    }

    /// The global identity at physical row `r`.
    pub fn store_at(&self, r: usize) -> u32 {
        self.ids[r]
    }

    /// The per-store maximum by layer-cake: `max[s]` is the largest `q` whose
    /// feasibility mask holds `s`. Indexed by store row (`id - BASE`), not by
    /// physical row. Stops at the first empty mask.
    pub fn max_quantities(&self, g: Guards) -> Result<[i32; STORES], ArmRefusal> {
        let mut max = [0i32; STORES];
        let mut prev = u64::MAX;
        for q in 1..=Q_MAX {
            let m = self.cook_mask(q, g)?;
            if m & !prev != 0 {
                return Err(ArmRefusal::NotMonotone { q });
            }
            if m == 0 {
                return Ok(max);
            }
            let mut bits = m;
            while bits != 0 {
                let r = bits.trailing_zeros() as usize;
                max[(self.ids[r] - STORE_ID_BASE) as usize] = q;
                bits &= bits - 1;
            }
            prev = m;
        }
        Err(ArmRefusal::Ceiling)
    }

    /// Every store's answer at quantity `q`, indexed by store row.
    pub fn answers(&self, q: i32, g: Guards) -> Result<[Answer; STORES], ArmRefusal> {
        let denied = self.denied_mask()?;
        let unknown = self.unknown_mask()?;
        let max = self.max_quantities(g)?;
        let mut out = [Answer::Unknown; STORES];
        for r in 0..STORES {
            let s = (self.ids[r] - STORE_ID_BASE) as usize;
            out[s] = if g.authorization && denied >> r & 1 == 1 {
                Answer::NotAuthorized
            } else if g.known && unknown >> r & 1 == 1 {
                Answer::Unknown
            } else {
                Answer::Known {
                    feasible: max[s] >= q,
                    max: max[s],
                }
            };
        }
        Ok(out)
    }

    /// `NetworkCapacity(item)` = Σ_q Count(CanCook(item, q)): each term is
    /// one mask-RISC `Count` program; monotonicity makes the sum Σ_s max_s.
    pub fn network_capacity(&self) -> Result<i64, ArmRefusal> {
        let mut total = 0i64;
        for q in 1..=Q_MAX {
            match self.run(
                self.cook_ops(q, Guards::ALL)?,
                Terminal::Count {
                    mask: Operand::Scratch(0),
                },
            )? {
                Value::Count(0) => return Ok(total),
                Value::Count(c) => total += c as i64,
                _ => unreachable!("Count yields a count"),
            }
        }
        Err(ArmRefusal::Ceiling)
    }

    /// Ingredient cost of making `q` at every store that can:
    /// Σ_k d_k·q · MaskedSumI32(CanCook(q), cost_k). Needs the cost lanes.
    pub fn network_cost(&self, q: i32) -> Result<i64, ArmRefusal> {
        let mut total = 0i64;
        for &(k, d) in &self.recipe.demand {
            let lane = self.lane(COST0 + k)?;
            match self.run(
                self.cook_ops(q, Guards::ALL)?,
                Terminal::MaskedSumI32 {
                    mask: Operand::Scratch(0),
                    lane,
                },
            )? {
                Value::SumI64(s) => total += i64::from(d * q) * s,
                _ => unreachable!("MaskedSumI32 yields a sum"),
            }
        }
        Ok(total)
    }

    /// `CanFulfill(item, q, deadline)`: cook ∧ a free slot ∧ on time.
    pub fn fulfill_mask(&self, q: i32, deadline: i32) -> Result<u64, ArmRefusal> {
        let mut ops = self.cook_ops(q, Guards::ALL)?;
        let acc = Operand::Scratch(0);
        for pred in [
            Pred::GeI32 {
                lane: self.lane(SLOTS)?,
                t: 1,
            },
            Pred::LeI32 {
                lane: self.lane(MINUTES)?,
                t: deadline - self.recipe.prep_minutes,
            },
        ] {
            ops.push(MaskOp::Pred {
                pred,
                under: None,
                dst: 1,
            });
            ops.push(MaskOp::And {
                a: acc,
                b: Operand::Scratch(1),
                dst: 0,
            });
        }
        self.keep(ops)
    }
}
