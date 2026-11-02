//! D-SAG-6-2 identity fixture, revision B: `helper` moved from `util` to
//! `ops`, body unchanged.

pub mod ops {
    pub fn helper(x: u32) -> u32 {
        x.wrapping_add(1)
    }
}

pub trait Shape {
    fn area(&self) -> u32;
}
