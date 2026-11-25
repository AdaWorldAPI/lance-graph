//! The program lists of `recipe.rs` and `presence.rs`, shared with the
//! upstream differential (`upstream.rs`) so a program added to a suite is
//! compared against upstream VRL too.

/// Programs inside the subset, each exercising a different part of it.
pub const RECIPE: &[&str] = &[
    ".name = .key\n.amount = .properties.amount.value",
    ".name = .key\n.amount = .properties.amount.value\nif .properties.status.value != 2 { abort }",
    ".id = .key\nif .properties.region.value == 0 { abort }\nif .properties.status.value == 3 { abort }",
    ".orig = .key\n.key = .properties.region.value\n.reg = .key\nif .key != 1 { abort }",
    ".a = .key; .b = .properties.amount.value; del(.a)",
    ".a = .properties.status.value; del(.key); if .a == 0 { abort }",
    ".s = .properties.status.value\n.amt = .properties.amount.value\nif .s != 1 { abort }",
    ".id = .key\nif .key < 1010 { abort }\nif .key >= 1050 { abort }",
    ".a = .properties.amount.value\nif .a <= 0 { abort }\nif .properties.status.value > 1 { abort }",
    ".r = .properties.region.value\nif .r < 1 { abort }\nif .r >= 2 { abort }",
];

pub const PRESENCE: &[&str] = &[
    ".n = .key\n.a = .properties.amount.value",
    ".a = .properties.amount.value\nif .a < 0 { abort }",
    ".a = .properties.amount.value\nif .properties.status.value != 2 { abort }",
    ".a = .properties.amount.value\nif .properties.region.value == 0 { abort }\nif .a >= 200 { abort }",
    ".r = .properties.region.value\n.a = .properties.amount.value\nif .properties.status.value > 2 { abort }\nif .a <= 10 { abort }\nif .r != 1 { abort }",
];
