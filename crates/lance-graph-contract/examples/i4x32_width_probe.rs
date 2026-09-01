//! Probe: which bytes do `I4x32`'s 32 lanes occupy, relative to a V3 facet?
use lance_graph_contract::atoms::I4x32;
use lance_graph_contract::facet::FacetCascade;

fn main() {
    // Lane-distinguishable pattern: lane k -> value ((k % 15) as i8) - 7, so
    // every lane differs from its neighbour and no lane is 0 except by design.
    let mut v = [0i8; 32];
    for (k, s) in v.iter_mut().enumerate() {
        *s = ((k % 15) as i8) - 7;
    }
    let packed = I4x32::pack(&v);
    let bytes: [u8; 16] = unsafe { std::mem::transmute_copy(&packed) };
    println!("I4x32 size_of      = {}", std::mem::size_of::<I4x32>());
    println!(
        "FacetCascade size  = {}",
        std::mem::size_of::<FacetCascade>()
    );
    println!("I4x32 bytes        = {bytes:02x?}");

    // Which byte does each lane live in? Flip one lane at a time, observe the byte.
    for k in 0..32usize {
        let mut w = v;
        w[k] = if v[k] == 7 { 6 } else { v[k] + 1 };
        let b2: [u8; 16] = unsafe { std::mem::transmute_copy(&I4x32::pack(&w)) };
        let moved: Vec<usize> = (0..16).filter(|&i| b2[i] != bytes[i]).collect();
        if !(10..30).contains(&k) {
            println!("lane {k:2} -> byte(s) {moved:?}");
        }
    }

    // Facet: vary ONLY the classid, observe which of the 16 bytes move.
    let f1 = FacetCascade::from_bytes(&[0u8; 16]);
    let mut raw = f1.to_bytes();
    raw[0] ^= 0xFF;
    raw[3] ^= 0xFF;
    let f2 = FacetCascade::from_bytes(&raw);
    let a = f1.to_bytes();
    let b = f2.to_bytes();
    let moved: Vec<usize> = (0..16).filter(|&i| a[i] != b[i]).collect();
    println!("facet classid-only change moves bytes {moved:?}");
    println!(
        "=> i4 lanes on those bytes: {:?}",
        moved
            .iter()
            .flat_map(|&i| [2 * i, 2 * i + 1])
            .collect::<Vec<_>>()
    );
}
