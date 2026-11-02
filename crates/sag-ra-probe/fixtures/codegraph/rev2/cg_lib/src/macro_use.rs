use crate::util::twice as t2;

pub fn checked() -> u32 {
    assert_eq!(t2(1), 2);
    1
}
