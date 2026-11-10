//! ARM A: the conventional materialized null model.
//!
//! Builds one owned object per store from the fixture, then answers by
//! walking objects. This is a locally constructed comparator, NOT a claim
//! about how Foundry represents objects internally. It deliberately makes the
//! second copy the virtual arm avoids, and its byte count is reported.

use crate::fixture::{self as fx, CATEGORIES, DENIED_STORE, EQUIPMENT, ITEMS, SKILLS, STORES};
use crate::oracle::{Cook, Limit};

/// One materialized store.
#[derive(Debug, Clone)]
pub struct StoreObject {
    pub id: u32,
    /// `None` = unknown stock.
    pub stock: Vec<Option<i32>>,
    pub cost: Vec<i32>,
    pub staff: Vec<i32>,
    pub equipment: Vec<bool>,
    pub offered: Vec<bool>,
    pub slots: i32,
    pub minutes: i32,
}

/// The materialized network.
pub struct Objects {
    pub stores: Vec<StoreObject>,
}

impl Objects {
    /// Build every object (the preparation cost of this arm).
    pub fn build(restocked: bool) -> Objects {
        let stores = (0..STORES)
            .map(|s| StoreObject {
                id: fx::store_id(s),
                stock: (0..CATEGORIES)
                    .map(|k| fx::known(s, k).then(|| fx::stock(s, k, restocked)))
                    .collect(),
                cost: (0..CATEGORIES).map(|k| fx::unit_cost(s, k)).collect(),
                staff: (0..SKILLS).map(|j| fx::staff(s, j)).collect(),
                equipment: (0..EQUIPMENT).map(|e| fx::equipment(s, e)).collect(),
                offered: (0..ITEMS).map(|m| fx::offered(s, m)).collect(),
                slots: fx::delivery_slots(s),
                minutes: fx::delivery_minutes(s),
            })
            .collect();
        Objects { stores }
    }

    /// Heap bytes the objects hold (vectors' payloads only).
    pub fn bytes(&self) -> usize {
        self.stores
            .iter()
            .map(|o| {
                o.stock.len() * 8
                    + o.cost.len() * 4
                    + o.staff.len() * 4
                    + o.equipment.len()
                    + o.offered.len()
                    + 16
            })
            .sum()
    }

    /// `CanCook` on one object.
    pub fn can_cook(&self, s: usize, m: usize, q: i32) -> Cook {
        let o = &self.stores[s];
        if o.id == DENIED_STORE {
            return Cook::NotAuthorized;
        }
        let r = fx::recipe(m);
        let mut stock = Vec::with_capacity(r.demand.len());
        for &(k, d) in &r.demand {
            match o.stock[k] {
                Some(v) => stock.push((k, v / d)),
                None => return Cook::Unknown { category: k },
            }
        }
        let (max, limit) = if !o.offered[m] {
            (0, Limit::NotOffered)
        } else if !o.equipment[r.equipment] {
            (0, Limit::Equipment(r.equipment))
        } else {
            let mut best = (i32::MAX, Limit::NotOffered);
            for (k, cap) in stock {
                if cap < best.0 {
                    best = (cap, Limit::Ingredient(k));
                }
            }
            let cap = o.staff[r.skill] * r.per_staff;
            if cap < best.0 {
                best = (cap, Limit::Staff(r.skill));
            }
            best
        };
        Cook::Known {
            feasible: max >= q,
            max,
            limit,
        }
    }

    /// `NetworkCapacity` on the objects.
    pub fn network_capacity(&self, m: usize) -> i64 {
        (0..STORES)
            .map(|s| match self.can_cook(s, m, 1) {
                Cook::Known { max, .. } => i64::from(max),
                _ => 0,
            })
            .sum()
    }
}
