//! Two symbols that share a name in different namespaces, and the two
//! value kinds (`const`, `static`) the Kythe producer must map.
#![allow(non_snake_case)]

pub struct Foo {}

pub fn Foo() -> u32 {
    LIMIT + COUNTER
}

pub const LIMIT: u32 = 1;
pub static COUNTER: u32 = 2;

pub fn uses() -> (Foo, u32) {
    (Foo {}, Foo())
}
