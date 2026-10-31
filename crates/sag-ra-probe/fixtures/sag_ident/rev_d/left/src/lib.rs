//! D-SAG-6-2 identity fixture, revision D: the old path `util::helper` is a
//! re-export of a DIFFERENT function. The path continues; the item does not.

pub mod ops {
    pub fn helper(x: u32) -> u32 {
        x.wrapping_add(1)
    }

    pub fn other(x: u32) -> u32 {
        x.wrapping_mul(3)
    }
}

pub mod util {
    pub use super::ops::other as helper;
}

pub trait Shape {
    fn area(&self) -> u32;
}
