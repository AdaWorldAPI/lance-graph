pub fn run(x: u32) -> u32 {
    left::ops::helper(x).wrapping_add(right::util::helper(x))
}

/// Dynamic dispatch across crates: the impl lives in `right`.
pub fn area_dyn(s: &dyn left::Shape) -> u32 {
    s.area()
}
