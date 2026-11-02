use cg_lib::shapes::{Rect, Shape};
use cg_lib::Sq;

pub fn run() -> u32 {
    let s = Sq(3);
    let r = Rect(1, 2);
    cg_lib::area_of(&s) + cg_lib::area_dyn(&r) + s.area() + cg_lib::util::double(2)
}
