//! Only demanded lanes are ever borrowed, and the counter can fire.

use soa64_u64_lab::arms::SoaU64;
use soa64_u64_lab::fixture::{value, Fixture, FIELDS};
use soa64_u64_lab::socket::{bind_demanded, fold_via_socket, U64Source, World, SPACE_A};

const SEED: u64 = 0x5EED_0064;
const N: usize = 513;

fn expected(fx: &Fixture, fields: &[u8]) -> u128 {
    let mut acc = 0u64;
    for &f in fields {
        for i in 0..fx.n {
            acc = acc.wrapping_add(value(fx.seed, i, usize::from(f)) ^ u64::from(f));
        }
    }
    acc as u128
}

#[test]
fn only_the_demanded_lanes_are_borrowed() {
    let fx = Fixture::new(SEED, N);
    let soa = SoaU64::build(&fx);
    let world = World::new();
    let source = U64Source::new(&soa);
    let fields = [0u8, 2, 4];

    let binding = bind_demanded(&world, &source, &fields, SPACE_A).unwrap();
    assert_eq!(source.borrows.get(), 0);
    assert_eq!(source.borrowed.get(), 0);

    let got = fold_via_socket(&world, &binding, &source, &fields).unwrap();
    assert_eq!(got, expected(&fx, &fields));
    assert_eq!(source.borrowed.get(), 0b10101);
    assert_eq!(source.borrows.get(), 3);
}

#[test]
fn demanding_every_field_borrows_every_lane() {
    let fx = Fixture::new(SEED, N);
    let soa = SoaU64::build(&fx);
    let world = World::new();
    let source = U64Source::new(&soa);
    let fields: Vec<u8> = (0..FIELDS as u8).collect();

    let binding = bind_demanded(&world, &source, &fields, SPACE_A).unwrap();
    let got = fold_via_socket(&world, &binding, &source, &fields).unwrap();
    assert_eq!(got, expected(&fx, &fields));
    assert_eq!(source.borrowed.get(), u64::MAX);
    assert_eq!(source.borrows.get(), FIELDS as u64);
}
