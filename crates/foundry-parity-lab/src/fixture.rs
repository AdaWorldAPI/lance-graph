//! The deterministic operating network. The only definition of its values:
//! the source, the oracle and the materialized null model all read it here,
//! and nothing else defines a value.
//!
//! All data is synthetic. Store rows `0..64` carry global identities
//! `STORE_ID_BASE + row`; rows `0..9` are hand-set for the named cases
//! (see [`case`]), every other row is derived from a hash of its coordinates.

/// Stores (X).
pub const STORES: usize = 64;
// Every mask in this lab is exactly one u64 word.
const _: () = assert!(STORES == 64);
/// Inventory / resource categories (Y).
pub const CATEGORIES: usize = 64;
/// Equipment types.
pub const EQUIPMENT: usize = 4;
/// Staff skills.
pub const SKILLS: usize = 4;
/// Menu items.
pub const ITEMS: usize = 8;
/// Revision of the menu (recipes). A prepared demand operand is valid for
/// exactly one revision.
pub const MENU_REV: u32 = 1;
/// Global identity of row 0.
pub const STORE_ID_BASE: u32 = 1000;

/// Named rows (row index = store number - 1).
pub mod case {
    /// Has all ingredients and sufficient staff for item 0.
    pub const ALL_GOOD: usize = 0;
    /// Lacks one critical ingredient (cheese = 0).
    pub const NO_CHEESE: usize = 1;
    /// Has inventory but no qualified cook.
    pub const NO_COOK: usize = 2;
    /// Has staff, but the oven is unavailable.
    pub const NO_OVEN: usize = 3;
    /// Stock for exactly one unit of item 0, not two.
    pub const EXACTLY_ONE: usize = 4;
    /// No cheese until an inventory update gives it 50.
    pub const RESTOCKED: usize = 5;
    /// Not visible to the lab reader (IAM stand-in denies it).
    pub const HIDDEN: usize = 6;
    /// Used by the reordered-source test; ordinary values.
    pub const REORDERED: usize = 7;
    /// Cheese stock is UNKNOWN; the slot holds a stale 1000.
    pub const UNKNOWN_CHEESE: usize = 8;
}

/// Category indices item 0 uses.
pub const FLOUR: usize = 0;
pub const CHEESE: usize = 1;
pub const SAUCE: usize = 2;
/// The stale value an unknown slot carries.
pub const STALE: i32 = 1000;

/// A recipe: what one unit of a menu item consumes and needs.
#[derive(Debug, Clone)]
pub struct Recipe {
    /// `(category, units per item)`, categories distinct and ascending.
    pub demand: Vec<(usize, i32)>,
    pub equipment: usize,
    pub skill: usize,
    /// Items one qualified employee can produce in the window.
    pub per_staff: i32,
    pub prep_minutes: i32,
}

pub(crate) fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

fn h(tag: u64, a: usize, b: usize) -> u64 {
    mix((tag << 56) ^ ((a as u64) << 24) ^ b as u64)
}

/// The recipe of item `m`.
pub fn recipe(m: usize) -> Recipe {
    if m == 0 {
        return Recipe {
            demand: vec![(FLOUR, 2), (CHEESE, 1), (SAUCE, 1)],
            equipment: 0,
            skill: 0,
            per_staff: 10,
            prep_minutes: 15,
        };
    }
    let mut demand: Vec<(usize, i32)> = (0..3)
        .map(|t| {
            (
                (m * 7 + t * 11 + 3) % CATEGORIES,
                1 + (t as i32 + m as i32) % 3,
            )
        })
        .collect();
    demand.sort_unstable();
    demand.dedup_by_key(|(k, _)| *k);
    Recipe {
        demand,
        equipment: m % EQUIPMENT,
        skill: m % SKILLS,
        per_staff: 4 + (m as i32 % 5),
        prep_minutes: 10 + 5 * (m as i32 % 4),
    }
}

/// Whether row `s` is one of the hand-set cases.
fn named(s: usize) -> bool {
    s <= case::UNKNOWN_CHEESE
}

/// Stock of category `k` at row `s`, at source version `restocked`.
pub fn stock(s: usize, k: usize, restocked: bool) -> i32 {
    if named(s) && k <= SAUCE {
        return match (s, k) {
            (case::NO_CHEESE, CHEESE) => 0,
            (case::EXACTLY_ONE, FLOUR) => 2,
            (case::EXACTLY_ONE, _) => 1,
            (case::RESTOCKED, CHEESE) => {
                if restocked {
                    50
                } else {
                    0
                }
            }
            (case::UNKNOWN_CHEESE, CHEESE) => STALE,
            _ => 100,
        };
    }
    20 + (h(1, s, k) % 60) as i32
}

/// Whether the stock of `k` at `s` is known.
pub fn known(s: usize, k: usize) -> bool {
    if named(s) {
        return !(s == case::UNKNOWN_CHEESE && k == CHEESE);
    }
    !h(2, s, k).is_multiple_of(23)
}

/// Unit cost of `k` at `s`, in cents.
pub fn unit_cost(s: usize, k: usize) -> i32 {
    50 + (h(3, s, k) % 200) as i32
}

/// Whether equipment `e` is available at `s`.
pub fn equipment(s: usize, e: usize) -> bool {
    if named(s) {
        return !(s == case::NO_OVEN && e == 0);
    }
    !h(4, s, e).is_multiple_of(9)
}

/// Qualified staff with skill `j` at `s`.
pub fn staff(s: usize, j: usize) -> i32 {
    if named(s) && j == 0 {
        return if s == case::NO_COOK { 0 } else { 3 };
    }
    (h(5, s, j) % 4) as i32
}

/// Whether item `m` is offered at `s`.
pub fn offered(s: usize, m: usize) -> bool {
    if named(s) {
        return true;
    }
    !h(6, s, m).is_multiple_of(7)
}

/// Free delivery slots at `s` in the window.
pub fn delivery_slots(s: usize) -> i32 {
    if named(s) {
        return 2;
    }
    (h(7, s, 0) % 4) as i32
}

/// Delivery minutes from `s` to the order's destination.
pub fn delivery_minutes(s: usize) -> i32 {
    if named(s) {
        return 20;
    }
    10 + (h(8, s, 0) % 40) as i32
}

/// Global identity of row `s`.
pub fn store_id(s: usize) -> u32 {
    STORE_ID_BASE + s as u32
}

/// The IAM stand-in's single denial: the reader may not see this store.
pub const DENIED_STORE: u32 = STORE_ID_BASE + case::HIDDEN as u32;
