//! Revision B: same path as in A, body changed.

pub mod util {
    pub fn helper(x: u32) -> u32 {
        x.wrapping_add(2)
    }
}

pub struct Sq(pub u32);

impl left::Shape for Sq {
    fn area(&self) -> u32 {
        self.0.wrapping_mul(self.0)
    }
}
