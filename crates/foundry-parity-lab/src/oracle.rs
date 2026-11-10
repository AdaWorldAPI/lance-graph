//! The independent scalar oracle. Reads [`crate::fixture`] directly; it
//! knows nothing of DT64, DR, lanes, masks or mask-RISC.

use crate::fixture::{self as fx, DENIED_STORE, STORES};

/// Why a store cannot make more of an item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Limit {
    NotOffered,
    Equipment(usize),
    Staff(usize),
    Ingredient(usize),
}

/// One store's answer to `CanCook(store, item, quantity)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cook {
    /// The reader may not see this store. Never reported as "cannot cook".
    NotAuthorized,
    /// A demanded category's stock is unknown. Never read as zero.
    Unknown { category: usize },
    /// `max` units can be made; `feasible` is `max >= quantity`.
    Known {
        feasible: bool,
        max: i32,
        limit: Limit,
    },
}

/// The oracle's view of the network at one source version.
#[derive(Debug, Clone, Copy)]
pub struct Oracle {
    pub restocked: bool,
    /// Store rows the scenario closes (Level S); empty for the baseline.
    pub closed: &'static [usize],
}

impl Oracle {
    pub const BASELINE: Oracle = Oracle {
        restocked: false,
        closed: &[],
    };

    /// `CanCook(row s, item m, quantity q)`.
    pub fn can_cook(&self, s: usize, m: usize, q: i32) -> Cook {
        if fx::store_id(s) == DENIED_STORE {
            return Cook::NotAuthorized;
        }
        let r = fx::recipe(m);
        if let Some(&(k, _)) = r.demand.iter().find(|(k, _)| !fx::known(s, *k)) {
            return Cook::Unknown { category: k };
        }
        let (max, limit) = if !fx::offered(s, m) || self.closed.contains(&s) {
            (0, Limit::NotOffered)
        } else if !fx::equipment(s, r.equipment) {
            (0, Limit::Equipment(r.equipment))
        } else {
            // Smallest cap wins; ties go to the earlier constraint
            // (ingredients in category order, then staff).
            let mut best = (i32::MAX, Limit::NotOffered);
            for &(k, d) in &r.demand {
                let cap = fx::stock(s, k, self.restocked) / d;
                if cap < best.0 {
                    best = (cap, Limit::Ingredient(k));
                }
            }
            let staff_cap = fx::staff(s, r.skill) * r.per_staff;
            if staff_cap < best.0 {
                best = (staff_cap, Limit::Staff(r.skill));
            }
            best
        };
        Cook::Known {
            feasible: max >= q,
            max,
            limit,
        }
    }

    /// `CanFulfill(row s, item m, quantity q, deadline minutes)`.
    pub fn can_fulfill(&self, s: usize, m: usize, q: i32, deadline: i32) -> Option<bool> {
        match self.can_cook(s, m, q) {
            Cook::Known { feasible, .. } => Some(
                feasible
                    && fx::delivery_slots(s) >= 1
                    && fx::recipe(m).prep_minutes + fx::delivery_minutes(s) <= deadline,
            ),
            _ => None,
        }
    }

    /// `NetworkCapacity(item m)`: Σ max over visible stores with known stock.
    pub fn network_capacity(&self, m: usize) -> i64 {
        (0..STORES)
            .map(|s| match self.can_cook(s, m, 1) {
                Cook::Known { max, .. } => i64::from(max),
                _ => 0,
            })
            .sum()
    }

    /// Ingredient cost of making `q` of `m` at every store that can.
    pub fn network_cost(&self, m: usize, q: i32) -> i64 {
        let r = fx::recipe(m);
        (0..STORES)
            .filter(|&s| matches!(self.can_cook(s, m, q), Cook::Known { feasible: true, .. }))
            .map(|s| {
                r.demand
                    .iter()
                    .map(|&(k, d)| i64::from(d * q) * i64::from(fx::unit_cost(s, k)))
                    .sum::<i64>()
            })
            .sum()
    }
}
