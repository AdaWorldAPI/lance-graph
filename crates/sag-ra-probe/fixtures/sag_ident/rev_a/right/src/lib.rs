//! Same module path and same body as `left::util::helper`, another crate.

pub mod util {
    pub fn helper(x: u32) -> u32 {
        x.wrapping_add(1)
    }
}

pub struct Sq(pub u32);

impl left::Shape for Sq {
    fn area(&self) -> u32 {
        self.0.wrapping_mul(self.0)
    }
}
