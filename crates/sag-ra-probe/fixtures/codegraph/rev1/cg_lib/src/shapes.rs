use crate::util::double;
use crate::util::triple;

pub trait Shape {
    fn area(&self) -> u32;
}

pub struct Square(pub u32);

pub struct Rect(pub u32, pub u32);

impl Shape for Square {
    fn area(&self) -> u32 {
        double(self.0) / 2 * self.0
    }
}

impl Shape for Rect {
    fn area(&self) -> u32 {
        self.0 * self.1
    }
}

impl Square {
    pub fn side(&self) -> u32 {
        self.0
    }
}
