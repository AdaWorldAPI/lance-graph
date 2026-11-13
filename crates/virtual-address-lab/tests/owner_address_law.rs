//! Independent reference probe (plan section 14): `Read(S, x, f) = S[f(x)]`
//! against a real `MailboxSoA`. It does not use `virtual_address_lab::owner`;
//! it reuses only `Recipe::normalize` and `Recipe::resolve`.
//!
//! A source-local coordinate demonstration. It resolves no `NodeGuid`, binds
//! no `ClassView` (fields 0 and 1 are lab field positions), authenticates no
//! version and is no admission: it shows what the address function denotes
//! and that one load from the owner's own column returns it.

use causal_edge::CausalEdge64;
use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::soa_view::MailboxSoaView;
use virtual_address_lab::physical::{apply, Moved};
use virtual_address_lab::recipe::Recipe;

const N: usize = 64;

#[derive(Debug, PartialEq, Eq)]
enum Scalar {
    Meta(u32),
    Edge(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Address {
    field: usize,
    row: usize,
}

/// PURE: no memory read, no allocation, no storage mutation.
fn resolve(
    recipe: Recipe,
    fields: usize,
    rows: usize,
    view_row: usize,
    view_property: usize,
) -> Option<Address> {
    let (field, row) = recipe.resolve(fields, rows, view_row, view_property)?;
    Some(Address { field, row })
}

/// EFFECT: one demanded scalar load from the original owner. The returned
/// value is copied; the backing column is not.
fn read_owner<V: MailboxSoaView>(owner: &V, address: Address) -> Option<Scalar> {
    if address.row >= owner.n_rows() {
        return None;
    }
    match address.field {
        0 => owner.meta_raw().get(address.row).copied().map(Scalar::Meta),
        1 => owner
            .edges_raw()
            .get(address.row)
            .copied()
            .map(Scalar::Edge),
        _ => None,
    }
}

/// COMPOSE BEFORE READ. No virtual object or transposed buffer is built.
fn read_virtual<V: MailboxSoaView>(
    owner: &V,
    chain: &[Recipe],
    view_row: usize,
    view_property: usize,
) -> Option<Scalar> {
    let recipe = Recipe::normalize(chain);
    let address = resolve(recipe, 2, owner.n_rows(), view_row, view_property)?;
    read_owner(owner, address)
}

fn owner() -> Box<MailboxSoA<N>> {
    let mut owner = Box::new(MailboxSoA::<N>::new(7, 3, 1.0));
    owner.set_populated(N);
    for row in 0..N {
        owner.set_meta(row, MetaWord(row as u32 + 100));
        owner.set_edge(row, CausalEdge64(row as u64 + 1000));
    }
    owner
}

#[test]
fn virtual_rotation_resolves_both_coordinates() {
    let owner = owner();

    // Natural entity view: row = entity, column = property.
    assert_eq!(
        read_virtual(&*owner, &[Recipe::TRANSPOSE], 12, 0),
        Some(Scalar::Meta(112))
    );

    // Rotate90 + Rotate270 cancels before the read.
    assert_eq!(
        Recipe::normalize(&[Recipe::ROTATE90, Recipe::ROTATE270]),
        Recipe::IDENTITY
    );
    assert_eq!(
        read_virtual(
            &*owner,
            &[Recipe::TRANSPOSE, Recipe::ROTATE90, Recipe::ROTATE270],
            12,
            0,
        ),
        Some(Scalar::Meta(112))
    );

    // ANTI_TRANSPOSE changes BOTH coordinates:
    // (view_row 12, property 0) -> (field 1, source_row 63 - 12) -> edge[51].
    assert_eq!(
        resolve(Recipe::ANTI_TRANSPOSE, 2, N, 12, 0),
        Some(Address { field: 1, row: 51 })
    );
    assert_eq!(
        read_virtual(&*owner, &[Recipe::ANTI_TRANSPOSE], 12, 0),
        Some(Scalar::Edge(1051))
    );
    assert_ne!(
        read_virtual(&*owner, &[Recipe::ANTI_TRANSPOSE], 12, 0),
        Some(Scalar::Edge(1012)),
        "the field-only reading"
    );

    // Only source-owned data, no copied backing store.
    assert_eq!(
        owner.meta_raw().as_ptr().cast::<u8>(),
        owner.meta.as_ptr().cast::<u8>()
    );

    // An invalid virtual coordinate yields nothing.
    assert_eq!(read_virtual(&*owner, &[Recipe::TRANSPOSE], 64, 0), None);
}

/// Independent scalar parity: for all eight recipes and every cell of the
/// view, the one read returns the value at the coordinate a data-moving
/// reference shows. FAILS IF: either coordinate is resolved differently.
#[test]
fn every_recipe_reads_the_cell_the_moved_matrix_shows() {
    let owner = owner();
    let matrix: Vec<Vec<i32>> = (0..2)
        .map(|lane| (0..N).map(|pos| (lane * 1000 + pos) as i32).collect())
        .collect();
    let (mut cells, mut moved_rows) = (0, 0);
    for recipe in Recipe::ALL {
        let mut moved = Moved::default();
        let built = apply(&matrix, recipe, &mut moved);
        let (view_rows, view_properties) = recipe.view_dims(2, N);
        assert_eq!((built.len(), built[0].len()), (view_rows, view_properties));
        for (r, line) in built.iter().enumerate() {
            for (c, &cell) in line.iter().enumerate() {
                let (lane, pos) = (cell as usize / 1000, cell as usize % 1000);
                let want = if lane == 0 {
                    Scalar::Meta(pos as u32 + 100)
                } else {
                    Scalar::Edge(pos as u64 + 1000)
                };
                assert_eq!(
                    read_virtual(&*owner, &[recipe], r, c),
                    Some(want),
                    "{} cell ({r}, {c})",
                    recipe.name()
                );
                // The position a field-only reading would take.
                let naive = if view_rows == N { r } else { c };
                moved_rows += usize::from(pos != naive);
                cells += 1;
            }
        }
        // One past each edge of the view is refused.
        assert_eq!(read_virtual(&*owner, &[recipe], view_rows, 0), None);
        assert_eq!(read_virtual(&*owner, &[recipe], 0, view_properties), None);
    }
    assert_eq!(cells, 8 * 2 * N);
    // Anti-vacuity: four of the eight recipes reverse positions, so half the
    // cells read another position than the one the view names.
    assert_eq!(moved_rows, 4 * 2 * N);
}
