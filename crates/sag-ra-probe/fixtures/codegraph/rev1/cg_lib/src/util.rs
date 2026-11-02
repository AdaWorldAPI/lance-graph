use crate::shapes::Square;

pub fn double(x: u32) -> u32 {
    x * 2
}

pub fn triple(x: u32) -> u32 {
    x * 3
}

pub fn unit() -> Square {
    Square(1)
}

fn private_unused() -> u32 {
    0
}
