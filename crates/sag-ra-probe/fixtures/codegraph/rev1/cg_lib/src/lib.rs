pub mod macro_use;
pub mod shapes;
pub mod util;

pub use shapes::Square as Sq;

pub fn area_of<S: shapes::Shape>(s: &S) -> u32 {
    s.area()
}

pub fn area_dyn(s: &dyn shapes::Shape) -> u32 {
    s.area()
}

#[cfg(any())]
pub fn disabled() -> u32 {
    util::double(1)
}
