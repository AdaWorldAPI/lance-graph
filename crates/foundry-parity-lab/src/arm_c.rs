//! ARM C: per-store ingredient cost as one INT8 GEMM (gap G1, partial).
//!
//! `C[store][item] = Σ_k cost[store][k] · demand[k][item]`. A is the cost
//! matrix (stores × categories, `u8`), B the menu's demand matrix (categories
//! × items, `i8`), and `u8 × i8 → i32` is exact here: costs are 50..=249 and
//! demands 1..=3. The product runs through `ndarray::simd::int8_gemm_packed_strided`
//! (AMX `TDPBUSD` where the host has it, the scalar tier otherwise; both exact).
//!
//! What this arm costs, stated rather than hidden:
//!
//! * **A is staged.** The source owns one `i32` lane per category over stores,
//!   which is A's transpose, and the kernel reads `u8` rows. So the bound cost
//!   lanes are copied into a 64 × 64 `u8` buffer once per binding (4 096 bytes,
//!   reported by [`CostQuery::staged_bytes`]). A cost that does not fit `u8` is
//!   refused, never truncated. Categories no item demands are not bound; their
//!   columns stay zero.
//! * **B is prepared once** per menu revision and reused under the menu's
//!   identity and revision only, never a digest (the rule
//!   `dr-isa-stream-lab`'s `ReuseKey` applies to its operands, #1992).
//!
//! What it still cannot answer: the limiting resource (an argmin across lanes)
//! is not a matrix product and stays BLOCKED (G1).

use std::cell::RefCell;

use lance_graph_contract::class_view::ClassView;
use lance_graph_dt64::dr::FrameStore;
use lance_graph_dt64::{BindStore, LaneBinding, LaneRefusal};
use lance_graph_mask_risc::{
    execute, tile_words_for, ExecError, LaneRef, MaskOp, Operand, Planes, Pred, Program, Scratch,
    Terminal, Value,
};
use ndarray::simd::{int8_gemm_packed_strided, int8_tile_gemm_tier, PackedI8B};

use crate::bind::{COST0, STORE_ID};
use crate::fixture::{
    self as fx, CATEGORIES, DENIED_STORE, ITEMS, MENU_REV, STORES, STORE_ID_BASE,
};
use crate::source::{Network, Plane};

/// Output columns of the one B band: 8 items, padded to the kernel's 16.
const BAND: usize = 16;

/// The categories any menu item demands, ascending. Only these are bound.
pub fn demanded_categories() -> Vec<usize> {
    let mut ks: Vec<usize> = (0..ITEMS)
        .flat_map(|m| fx::recipe(m).demand.into_iter().map(|(k, _)| k))
        .collect();
    ks.sort_unstable();
    ks.dedup();
    ks
}

/// The fields arm C binds: the demanded categories' cost lanes and `storeId`.
pub fn cost_fields() -> Vec<u8> {
    let mut f: Vec<u8> = demanded_categories()
        .into_iter()
        .map(|k| (COST0 + k) as u8)
        .collect();
    f.push(STORE_ID as u8);
    f
}

/// Why arm C could not answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CostRefusal {
    Lane(LaneRefusal),
    Exec(ExecError),
    /// A cost does not fit the kernel's unsigned 8-bit A operand.
    NotU8 {
        field: u8,
        row: usize,
        value: i32,
    },
    /// A bound plane had another kind than its field requires.
    PlaneKind {
        field: u8,
    },
    /// `storeId` was not bound.
    NoIdentity,
    /// The prepared demand band was made for another menu revision.
    MenuRevision {
        prepared: u32,
        asked: u32,
    },
    /// `q` outside `1..=128`.
    Quantity {
        q: i32,
    },
}

impl From<LaneRefusal> for CostRefusal {
    fn from(r: LaneRefusal) -> Self {
        CostRefusal::Lane(r)
    }
}

impl From<ExecError> for CostRefusal {
    fn from(e: ExecError) -> Self {
        CostRefusal::Exec(e)
    }
}

/// The menu's demand matrix, VNNI-packed once.
pub struct PreparedDemand {
    menu: u32,
    band: [PackedI8B; 1],
}

impl PreparedDemand {
    /// Pack the demand of menu revision `menu` (K = 64 categories × 16 cols).
    pub fn prepare(menu: u32) -> PreparedDemand {
        let mut b = vec![0i8; CATEGORIES * BAND];
        for m in 0..ITEMS {
            for (k, d) in fx::recipe(m).demand {
                b[k * BAND + m] = i8::try_from(d).expect("demand fits i8");
            }
        }
        PreparedDemand {
            menu,
            band: [PackedI8B::pack(&b, CATEGORIES)],
        }
    }

    /// Reuse rule: the menu revision must match. Nothing else is consulted.
    pub fn admit(&self, menu: u32) -> Result<(), CostRefusal> {
        if self.menu == menu {
            Ok(())
        } else {
            Err(CostRefusal::MenuRevision {
                prepared: self.menu,
                asked: menu,
            })
        }
    }
}

/// Per-store cost over one binding: staged A, the identity lane, the denied mask.
pub struct CostQuery<'s> {
    a: Box<[u8]>,
    ids: &'s [u32],
    denied: u64,
    out: RefCell<Box<[i32]>>,
}

impl<'s> CostQuery<'s> {
    /// Check the binding once, borrow every bound lane, stage A, and compute
    /// the IAM stand-in's denied mask with one mask-RISC program.
    pub fn open<V: ClassView + ?Sized>(
        binding: &LaneBinding,
        store: &BindStore,
        view: &V,
        frames: &FrameStore,
        source: &'s Network,
    ) -> Result<CostQuery<'s>, CostRefusal> {
        let mut a = vec![0u8; STORES * CATEGORIES].into_boxed_slice();
        let mut ids = None;
        for lane in binding.borrow_all(store, view, frames, source)? {
            let (field, plane) = lane?;
            match (usize::from(field), plane) {
                (STORE_ID, Plane::U32(v)) => ids = Some(v),
                (f, Plane::I32(v)) if (COST0..COST0 + CATEGORIES).contains(&f) => {
                    let k = f - COST0;
                    for (row, &value) in v.iter().enumerate() {
                        a[row * CATEGORIES + k] = u8::try_from(value)
                            .map_err(|_| CostRefusal::NotU8 { field, row, value })?;
                    }
                }
                _ => return Err(CostRefusal::PlaneKind { field }),
            }
        }
        let ids = ids.ok_or(CostRefusal::NoIdentity)?;
        let lanes = [LaneRef::U32(ids)];
        let planes = Planes {
            n_rows: STORES,
            masks: &[],
            lanes: &lanes,
        };
        let program = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::NeU32 {
                    lane: 0,
                    v: DENIED_STORE,
                },
                under: None,
                dst: 0,
            }],
            Terminal::Keep {
                mask: Operand::Scratch(0),
            },
        );
        let mut scratch = Scratch::new(tile_words_for(STORES), 1);
        let visible = match execute(&program, &planes, &mut scratch, None)? {
            Value::Mask(_) => scratch.slot(0).map_or(0, |s| s[0]),
            _ => unreachable!("Keep yields a mask"),
        };
        Ok(CostQuery {
            a,
            ids,
            denied: !visible,
            out: RefCell::new(vec![0i32; STORES * BAND].into_boxed_slice()),
        })
    }

    /// Bytes copied from the source to stage A.
    pub fn staged_bytes(&self) -> usize {
        self.a.len()
    }

    /// Which INT8 tier runs (`"AMX TDPBUSD"` or the scalar reference).
    pub fn tier() -> &'static str {
        int8_tile_gemm_tier()
    }

    /// Ingredient cost of making `q` of every item at every store, indexed
    /// `[store row][item]` (store row = id − base). `None` for a store the
    /// reader is denied. One GEMM per call.
    pub fn per_store(
        &self,
        demand: &PreparedDemand,
        q: i32,
    ) -> Result<[[Option<i64>; ITEMS]; STORES], CostRefusal> {
        if !(1..=crate::arm_b::Q_MAX).contains(&q) {
            return Err(CostRefusal::Quantity { q });
        }
        demand.admit(MENU_REV)?;
        let mut c = self.out.borrow_mut();
        int8_gemm_packed_strided(&self.a, CATEGORIES, &demand.band, &mut c, STORES);
        let mut out = [[None; ITEMS]; STORES];
        for r in 0..STORES {
            if self.denied >> r & 1 == 1 {
                continue;
            }
            let s = (self.ids[r] - STORE_ID_BASE) as usize;
            for m in 0..ITEMS {
                out[s][m] = Some(i64::from(q) * i64::from(c[r * BAND + m]));
            }
        }
        Ok(out)
    }
}
