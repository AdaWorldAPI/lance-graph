//! Test F: the `MoorePalettePairs` tenant addresses the #1336 Palette256 law
//! the same way `PairAddress` does.
//!
//! Byte `2i` of the tenant is the first operand and byte `2i + 1` the second,
//! so the law entry for slot `i` is `lut[(first << 8) | second]`. The test
//! reads each pair through the contract's tenant view and checks it against
//! the shipped `PaletteLut::at` and `Quad8::palette_pairs` on the same bytes.

use cognitive_shader_driver::palette_perturbation::{
    PairAddress, PaletteLut, PaletteState, PALETTE_LUT_LEN,
};
use cognitive_shader_driver::quad8::Quad8;
use lance_graph_contract::canonical_node::{ValueTenant, VALUE_SLAB_LEN};
use lance_graph_contract::moore_tenant::{MooreSlot, MooreTenantMut, MooreTenantView};

/// An asymmetric law, so swapping the operands changes the answer.
fn law() -> Box<[u8; PALETTE_LUT_LEN]> {
    let mut t = vec![0u8; PALETTE_LUT_LEN].into_boxed_slice();
    for a in 0..256usize {
        for b in 0..256usize {
            t[a * 256 + b] = (a * 31 + b * 7 + (a ^ b)) as u8;
        }
    }
    t.try_into().expect("64 KiB")
}

fn next(x: &mut u64) -> u8 {
    *x = x
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*x >> 56) as u8
}

#[test]
fn tenant_pairs_address_the_lut_like_pair_address() {
    let entries = law();
    let lut = PaletteLut::new(&entries);
    let mut seed = 0x1336u64;
    let mut swapped_differs = 0;
    for _ in 0..512 {
        let mut slab = [0u8; VALUE_SLAB_LEN];
        let mut m = MooreTenantMut::new(&mut slab);
        for slot in MooreSlot::ALL {
            m.set_palette_pair(slot, next(&mut seed), next(&mut seed));
        }
        let view = MooreTenantView::new(&slab);
        for slot in MooreSlot::ALL {
            let (first, second) = view.palette_pair(slot);
            let addr = PairAddress::new(PaletteState(first), PaletteState(second));
            assert_eq!(addr.0, (u16::from(first) << 8) | u16::from(second));
            let via_tenant = entries[(usize::from(first) << 8) | usize::from(second)];
            assert_eq!(
                lut.at(PaletteState(first), PaletteState(second)).0,
                via_tenant
            );
            assert_eq!(lut.at_address(addr).0, via_tenant);
            if lut.at(PaletteState(second), PaletteState(first)).0 != via_tenant {
                swapped_differs += 1;
            }
        }
        // Two adjacent slots' four bytes, read as the #1336 Quad8, give the
        // same two addresses the tenant gives.
        let base = ValueTenant::MoorePalettePairs.value_offset();
        for k in [0usize, 2, 4, 6] {
            let b = &slab[base + 2 * k..base + 2 * k + 4];
            let (p0, p1) = Quad8::from_bytes([b[0], b[1], b[2], b[3]]).palette_pairs();
            let (a0, a1) = (
                view.palette_pair(MooreSlot::ALL[k]),
                view.palette_pair(MooreSlot::ALL[k + 1]),
            );
            assert_eq!(p0, PairAddress::new(PaletteState(a0.0), PaletteState(a0.1)));
            assert_eq!(p1, PairAddress::new(PaletteState(a1.0), PaletteState(a1.1)));
        }
    }
    // Anti-vacuity: the orientation is load-bearing on this law.
    assert!(
        swapped_differs > 3000,
        "swapped orientation differed only {swapped_differs} times"
    );
}
