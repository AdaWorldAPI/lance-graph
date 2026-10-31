//! D-SAG-6-2 identity fixture, revision A.

pub mod util {
    pub fn helper(x: u32) -> u32 {
        x.wrapping_add(1)
    }
}

pub trait Shape {
    fn area(&self) -> u32;
}
