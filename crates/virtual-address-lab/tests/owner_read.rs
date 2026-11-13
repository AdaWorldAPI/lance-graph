//! P0 (plan section 11): one property of one source-local row, read from a
//! real `MailboxSoA` through `MailboxSoaView`. Oracle: plain functions here.

use causal_edge::CausalEdge64;
use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::soa_view::{MailboxSoaView, StyleLane};
use virtual_address_lab::owner::{Column, Reads, Value, VirtualProperty};
use virtual_address_lab::recipe::Recipe;
use virtual_address_lab::view::Refused;

const N: usize = 64;
type Owner = MailboxSoA<N>;
const ENTITY: [Recipe; 1] = [Recipe::TRANSPOSE];
const META: usize = 0;
const EDGE: usize = 1;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
fn meta_of(r: usize) -> u32 {
    mix(r as u64) as u32 | 1
}
fn edge_of(r: usize) -> u64 {
    mix(1_000 + r as u64) | 1
}

fn owner(populated: usize) -> Box<Owner> {
    let mut m: Box<Owner> = Box::new(MailboxSoA::new(7, 3, 1.0));
    m.set_populated(populated);
    for r in 0..populated {
        m.set_meta(r, MetaWord(meta_of(r)));
        m.set_edge(r, CausalEdge64(edge_of(r)));
    }
    m
}

fn within<T>(p: *const T, slab: *const u8, bytes: usize) -> bool {
    let a = p as usize;
    a >= slab as usize && a < slab as usize + bytes
}

/// FAILS IF: the value is not the oracle's, the slice lent is not the owner's
/// own array, a second value or the other column is read, or the descriptor
/// carries a value.
#[test]
fn one_property_of_one_row_is_read_from_the_owners_own_column() {
    let mailbox = owner(N);
    let reads = Reads::default();
    let property = VirtualProperty {
        row: 37,
        property: META,
        chain: &ENTITY,
    };
    assert_eq!(property.column(N), Ok(Column::Meta));
    assert_eq!(reads.values.get(), 0, "resolving read nothing");
    assert_eq!(
        property.read(&*mailbox, &reads),
        Ok(Value::U32(meta_of(37)))
    );
    assert_eq!(
        (
            reads.values.get(),
            reads.meta_borrows.get(),
            reads.edge_borrows.get()
        ),
        (1, 1, 0)
    );

    // The borrowed slices ARE the owner's arrays, inside the owner's allocation.
    let slab = (&*mailbox as *const Owner).cast::<u8>();
    let bytes = core::mem::size_of::<Owner>();
    assert!(core::ptr::eq(
        mailbox.meta_raw().as_ptr().cast::<u8>(),
        mailbox.meta.as_ptr().cast::<u8>()
    ));
    assert!(core::ptr::eq(
        mailbox.edges_raw().as_ptr().cast::<u8>(),
        mailbox.edges.as_ptr().cast::<u8>()
    ));
    assert!(within(mailbox.meta_raw().as_ptr(), slab, bytes));
    assert!(within(mailbox.edges_raw().as_ptr(), slab, bytes));
    // Two separate columns, not one record: 4-byte and 8-byte strides.
    assert_eq!(core::mem::size_of_val(mailbox.meta_raw()), N * 4);
    assert_eq!(core::mem::size_of_val(mailbox.edges_raw()), N * 8);

    // The descriptor: a row, a property index and a borrowed recipe chain.
    assert_eq!(core::mem::size_of::<VirtualProperty<'_>>(), 32);

    let other = VirtualProperty {
        row: 37,
        property: EDGE,
        chain: &ENTITY,
    };
    assert_eq!(other.read(&*mailbox, &reads), Ok(Value::U64(edge_of(37))));
    assert_eq!(reads.edge_borrows.get(), 1);
}

/// `style_lane_at` is NOT such a borrow: it returns 12 bytes by value.
/// FAILS IF: it starts lending the owner's bytes (then it is a candidate).
#[test]
fn the_style_lane_accessor_returns_a_copy_not_a_borrow() {
    let mut mailbox = owner(N);
    mailbox.set_style_lane(5, StyleLane::Frozen, [9; 12]);
    let lane = mailbox.style_lane_at(5, StyleLane::Frozen).unwrap();
    assert_eq!(lane, [9; 12]);
    let slab = (&*mailbox as *const Owner).cast::<u8>();
    assert!(!within(lane.as_ptr(), slab, core::mem::size_of::<Owner>()));
}

/// FAILS IF: an identity-normalised tail changes the read, an Identity view
/// reads anything, or the recipe is not consulted.
#[test]
fn the_chain_is_normalised_before_the_read() {
    let mailbox = owner(N);
    let reads = Reads::default();
    let direct = Value::U32(mailbox.meta_raw()[12]);
    for chain in [
        &[Recipe::TRANSPOSE, Recipe::ROTATE90, Recipe::ROTATE270][..],
        &[Recipe::TRANSPOSE, Recipe::ROTATE180, Recipe::ROTATE180][..],
        &[Recipe::ROTATE90, Recipe::ROTATE270, Recipe::TRANSPOSE][..],
    ] {
        assert_eq!(Recipe::normalize(chain), Recipe::TRANSPOSE);
        let p = VirtualProperty {
            row: 12,
            property: META,
            chain,
        };
        assert_eq!(p.read(&*mailbox, &reads), Ok(direct));
    }
    assert_eq!(
        reads.values.get(),
        3,
        "one load per request, none for the chain"
    );

    let undone = VirtualProperty {
        row: 12,
        property: META,
        chain: &[Recipe::ROTATE90, Recipe::ROTATE270],
    };
    assert_eq!(
        undone.read(&*mailbox, &reads),
        Err(Refused::PropertyCrossesFields {
            recipe: Recipe::IDENTITY
        })
    );
    assert_eq!(reads.values.get(), 3);

    // The reversed property axis is the other column.
    let flipped = VirtualProperty {
        row: 12,
        property: META,
        chain: &[Recipe::ANTI_TRANSPOSE],
    };
    assert_eq!(flipped.column(N), Ok(Column::Edge));
    assert_eq!(flipped.read(&*mailbox, &reads), Ok(Value::U64(edge_of(12))));
}

/// Absent property, absent row. FAILS IF: a value is fabricated.
#[test]
fn an_absent_property_or_row_yields_no_value() {
    let mailbox = owner(40);
    let reads = Reads::default();
    assert_eq!(
        mailbox.meta_raw().len(),
        N,
        "the owner's array spans capacity"
    );
    assert_eq!(mailbox.meta_raw()[40], 0, "row 40 is zero padding");
    let padded = VirtualProperty {
        row: 40,
        property: META,
        chain: &ENTITY,
    };
    assert_eq!(
        padded.read(&*mailbox, &reads),
        Err(Refused::PositionOutsideLane {
            at: 0,
            position: 40
        })
    );
    let no_such = VirtualProperty {
        row: 3,
        property: 2,
        chain: &ENTITY,
    };
    assert_eq!(
        no_such.read(&*mailbox, &reads),
        Err(Refused::OutsideTheView)
    );
    assert_eq!(reads.values.get(), 0);
    let last = VirtualProperty {
        row: 39,
        property: META,
        chain: &ENTITY,
    };
    assert_eq!(last.read(&*mailbox, &reads), Ok(Value::U32(meta_of(39))));
}

/// RECORDED, not solved. The owner resolves no `NodeGuid`, and a setter write
/// moves none of its own stamps. FAILS IF: either changes (then the boundary
/// in plan section 11 moved).
#[test]
fn the_owner_has_no_identity_seam_and_no_stamp_for_a_setter_write() {
    let mut mailbox = owner(N);
    for key in [0u64, 37, u64::MAX] {
        assert_eq!(mailbox.row_for_local_key(key), None);
    }
    let cycle = mailbox.current_cycle();
    let stamp = mailbox.last_write_cycle_at(11);
    let before = mailbox.meta_raw()[11];
    mailbox.set_meta(11, MetaWord(before ^ 0xFFFF));
    assert_ne!(mailbox.meta_raw()[11], before);
    assert_eq!(mailbox.current_cycle(), cycle);
    assert_eq!(mailbox.last_write_cycle_at(11), stamp);
    assert_eq!(stamp, u32::MAX, "set_meta never stamped the row");
}
