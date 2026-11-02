//! D-SAG-6-2 identity fixture, revision C: `helper` moved from `util` to
//! `ops`, body unchanged, and the old path kept as a re-export.

pub mod ops {
    pub fn helper(x: u32) -> u32 {
        x.wrapping_add(1)
    }
}

pub mod util {
    pub use super::ops::helper;
}

pub trait Shape {
    fn area(&self) -> u32;
}
