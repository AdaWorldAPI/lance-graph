use crate::util::triple as t3;

pub fn checked() -> u32 {
    assert_eq!(t3(1), 3);
    1
}
