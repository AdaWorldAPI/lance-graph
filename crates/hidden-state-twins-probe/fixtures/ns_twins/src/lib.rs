//! One name in two namespaces, and two sites that look the same.
//!
//! `Foo` is a struct (type namespace) and a function (value namespace). In
//! `as_type` and `as_value` the text `(Foo,` occurs once each: in the first it
//! is the struct, in the second the function.
#![allow(non_snake_case)]

pub struct Foo {}

pub fn Foo() -> u32 {
    1
}

pub fn as_type() -> (Foo, u32) {
    (Foo {}, Foo())
}

pub fn as_value() -> (fn() -> u32, u32) {
    (Foo, 1)
}
