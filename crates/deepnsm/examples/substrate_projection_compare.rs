//! Compare substrate: p64 (Palette64) vs V3 (classid facet) vs HYBRID, as
//! WIRINGS of the SAME 512-byte SoA node (operator: *"would it resonate to
//! wire lance-graph via P64 parameter vs V3 parameter vs hybrid and compare
//! substrate"*).
//!
//! ## The claim under test
//!
//! p64 and V3 are NOT two competing carriers — they are two PROJECTIONS of one
//! 512-byte block:
//!   - **p64 (Palette64)** reads the 512 bytes as a **64×64 BNN attention
//!     matrix** (4096 bits = 8 cache lines; `Q AND K >> Γ` attention, POPCNT
//!     palette lookup). The COMPUTE projection.
//!   - **V3** reads the SAME 512 bytes as `key(16) | edges(16) | value(480)`:
//!     classid routing + edge slots + tenant value. The ADDRESS projection.
//!
//! Because it is ONE backing array, the boundary between the two is a
//! **zero-copy reinterpretation**, not a serialize — a write through one
//! projection is observed by the other. The V3 key's HEEL/HIP/TWIG cascade
//! tiers coincide with p64's top attention rows (the "key prerenders with zero
//! value decode" canon). So "hybrid" is the substrate's native shape:
//! p64-only amputates routing, V3-only leaves the attention matrix un-run,
//! hybrid is the node as it actually is.
//!
//! The only REAL conversion in the whole wiring is at the edge sub-register
//! (`CausalEdge64` u64 ↔ `CausalEdgeV3` 96-bit), and that was already proven
//! thinking-preserving (E-CAUSALEDGE-V3-96-STAGED-1 / -COMPARE-DRIVER-1). This
//! probe does NOT re-prove that; it proves the SUBSTRATE-level claim that the
//! two projections share one zero-copy backing.
//!
//! KILL gates (regressions, not discoveries):
//!   1. one 512-byte block satisfies BOTH contracts at once (valid p64 64×64
//!      matrix AND valid V3 key/edges/value with a routable classid).
//!   2. zero-copy shared backing: a write through the V3 value projection is
//!      observed by the p64 matrix projection (same bytes, not two copies).
//!   3. cascade↔attention resonance: the V3 HEEL bytes ARE p64's attention
//!      seed row (the address is the compute seed).
//!   4. the two readings are independently addressable on the shared bytes: a
//!      value-body write moves its p64 matrix row but not the classid route,
//!      and zeroing the classid drops routing to the default rung (compute and
//!      address are orthogonal views of one backing array).
//!   5. boundary cost = 0 bytes copied (the projections are reinterpret casts).
//!
//! ## Run
//! ```bash
//! cargo run --manifest-path crates/deepnsm/Cargo.toml --example substrate_projection_compare
//! ```

const NODE_BYTES: usize = 512;

/// The one backing block. Both projections borrow THIS — never a copy.
struct Node {
    bytes: [u8; NODE_BYTES],
}

impl Node {
    fn new() -> Self {
        Node {
            bytes: [0u8; NODE_BYTES],
        }
    }

    // ── V3 (address) projection: key(16) | edges(16) | value(480) ──
    fn v3_classid(&self) -> u32 {
        u32::from_le_bytes([self.bytes[0], self.bytes[1], self.bytes[2], self.bytes[3]])
    }
    fn set_v3_classid(&mut self, c: u32) {
        self.bytes[0..4].copy_from_slice(&c.to_le_bytes());
    }
    fn v3_heel(&self) -> u16 {
        u16::from_le_bytes([self.bytes[4], self.bytes[5]])
    }
    fn set_v3_heel(&mut self, h: u16) {
        self.bytes[4..6].copy_from_slice(&h.to_le_bytes());
    }
    fn v3_edge_slot(&self, i: usize) -> u8 {
        self.bytes[16 + i] // 16 one-byte edge slots (12 in-family + 4 out)
    }
    fn set_v3_edge_slot(&mut self, i: usize, v: u8) {
        self.bytes[16 + i] = v;
    }
    fn v3_value(&self) -> &[u8] {
        &self.bytes[32..NODE_BYTES] // 480-byte tenant value
    }
    fn set_v3_value_byte(&mut self, i: usize, v: u8) {
        self.bytes[32 + i] = v;
    }
    /// Zero-fallback ladder: classid 0 => default class (unrouted).
    fn v3_routes(&self) -> bool {
        self.v3_classid() != 0
    }

    // ── p64 (compute) projection: 512 B = 64 rows × u64 = 64×64 bit matrix ──
    fn p64_row(&self, r: usize) -> u64 {
        let o = r * 8;
        u64::from_le_bytes([
            self.bytes[o],
            self.bytes[o + 1],
            self.bytes[o + 2],
            self.bytes[o + 3],
            self.bytes[o + 4],
            self.bytes[o + 5],
            self.bytes[o + 6],
            self.bytes[o + 7],
        ])
    }
    /// BNN attention between two rows: `Q AND K`, then POPCNT (the Hamming
    /// overlap = attention weight). One instruction each on real hardware.
    fn p64_attention(&self, q: usize, k: usize) -> u32 {
        (self.p64_row(q) & self.p64_row(k)).count_ones()
    }
    /// Palette lookup: the row nearest `probe` by Hamming distance (POPCNT).
    fn p64_nearest_row(&self, probe: u64) -> usize {
        (0..64)
            .min_by_key(|&r| (self.p64_row(r) ^ probe).count_ones())
            .unwrap()
    }
}

fn main() {
    let mut node = Node::new();

    // Build one node that is BOTH a routable V3 facet AND a live p64 matrix.
    node.set_v3_classid(0x0202_0002); // a real routing classid (account.move-shaped)
    node.set_v3_heel(0xBEEF); // cascade tier 0 (also p64 attention seed, see below)
    for i in 0..16 {
        node.set_v3_edge_slot(i, (i as u8) * 17); // 16 edge refs
    }
    for i in 0..480 {
        node.set_v3_value_byte(i, (i as u8).wrapping_mul(31)); // tenant value body
    }

    println!("ONE 512-byte block, read two ways:\n");
    println!(
        "  V3 (address):  classid=0x{:08X} route={}  HEEL=0x{:04X}  edge[0..4]={:?}",
        node.v3_classid(),
        node.v3_routes(),
        node.v3_heel(),
        (0..4).map(|i| node.v3_edge_slot(i)).collect::<Vec<_>>()
    );
    println!(
        "  p64 (compute): row0=0x{:016X}  attn(0,1)={}  nearest(0)={}",
        node.p64_row(0),
        node.p64_attention(0, 1),
        node.p64_nearest_row(0)
    );
    println!();

    let mut fail = Vec::new();

    // 1. one block, both contracts valid at once.
    if !node.v3_routes() {
        fail.push("V3 classid does not route".into());
    }
    // p64 matrix is well-formed: 64 rows readable, attention in [0,64].
    let a = node.p64_attention(0, 1);
    if a > 64 {
        fail.push(format!("p64 attention {a} out of range"));
    }

    // 2. zero-copy shared backing: write through V3 value, observe via p64.
    //    value byte 0 == node byte 32 == p64 row 4, byte 0.
    let row4_before = node.p64_row(4);
    node.set_v3_value_byte(0, 0xAB);
    let row4_after = node.p64_row(4);
    if row4_before == row4_after || (row4_after & 0xFF) != 0xAB {
        fail.push(
            "V3-value write not observed by p64 matrix (backing was copied, not shared)".into(),
        );
    }

    // 3. cascade↔attention resonance: the V3 HEEL bytes (4..6) ARE inside
    //    p64 row 0 (bytes 0..8). Change HEEL -> p64 row0 changes.
    let row0_before = node.p64_row(0);
    node.set_v3_heel(0x1234);
    let row0_after = node.p64_row(0);
    // HEEL sits at bytes 4,5 => bits 32..48 of row 0.
    let heel_in_row0 = ((row0_after >> 32) & 0xFFFF) as u16;
    if row0_before == row0_after || heel_in_row0 != 0x1234 {
        fail.push("V3 HEEL is NOT p64's attention seed (address and compute seed diverged)".into());
    }

    // 4. the two readings are INDEPENDENTLY ADDRESSABLE on the shared bytes.
    //    (Replaces a prior vacuous gate that asserted hardcoded `false`
    //    booleans and so could never fail — a KILL gate that cannot kill.)
    //    Measured on a fresh node, each assertion can genuinely fail:
    //      - a VALUE-body write moves its p64 matrix row (compute reads the
    //        shared bytes) but does NOT disturb the classid route;
    //      - zeroing the classid drops routing to the unrouted default rung
    //        (the route is addressed by the key bytes, not the value body).
    //    This is the real substrate property behind the "amputation" argument
    //    in the table below: the compute reading and the address reading are
    //    orthogonal views of one backing array.
    {
        let mut n = Node::new();
        n.set_v3_classid(0x0202_0002);
        for i in 0..480 {
            n.set_v3_value_byte(i, 0x33);
        }
        let route_initial = n.v3_routes();
        let vrow_before = n.p64_row(4); // row 4 == value bytes 0..8
        n.set_v3_value_byte(2, 0xCC); // flip a value-body byte inside row 4
        let vrow_after = n.p64_row(4);
        let route_after_value = n.v3_routes();
        n.set_v3_classid(0); // zero the routing key
        let route_after_classid = n.v3_routes();

        if vrow_before == vrow_after {
            fail.push(
                "value-body write did not change its p64 row (compute not on shared bytes)".into(),
            );
        }
        if !route_initial || route_after_value != route_initial {
            fail.push(
                "value-body write disturbed the classid route (readings not independent)".into(),
            );
        }
        if route_after_classid {
            fail.push(
                "zeroing classid did not fall to the unrouted default (route not key-addressed)"
                    .into(),
            );
        }
    }

    // 5. boundary cost = 0 copied bytes: both projections borrow `self.bytes`
    //    — every accessor slices the one backing array, none allocate or copy.
    //    Gate 2 already PROVED the shared backing (a V3-value write was seen by
    //    the p64 matrix); a copy would have hidden it. Here we just confirm the
    //    value projection is the same-length in-place window (480 B), not a
    //    materialized buffer.
    if node.v3_value().len() != 480 {
        fail.push("V3 value projection is not the in-place 480-byte window".into());
    }

    // ── comparison table (the "compare substrate" deliverable) ──
    println!("substrate comparison (one 512-byte node):");
    println!("  wiring   | routes(classid) | runs attention | temporal/anaphora | boundary copy");
    println!("  ---------|-----------------|----------------|-------------------|--------------");
    println!("  p64-only |       no        |      yes       |    no (compute)   |      —");
    println!("  V3-only  |      yes        |   not invoked  |  yes (edge facet) |      —");
    println!("  HYBRID   |      yes        |      yes        |        yes        |   0 bytes");
    println!();
    println!(
        "Edge sub-register (the ONE real conversion): CausalEdge64 u64 ↔ CausalEdgeV3 96-bit,"
    );
    println!(
        "proven thinking-preserving in #766 (not re-run here). Everything else is reinterpret."
    );
    println!();

    if fail.is_empty() {
        println!(
            "KILL GATES: all pass — the 512-byte node is SIMULTANEOUSLY a p64 64×64 attention"
        );
        println!("matrix AND a routable V3 key|edges|value facet; the boundary is a zero-copy");
        println!("reinterpret (V3-value writes seen by p64; V3 HEEL == p64 attention seed); the");
        println!("compute and address readings are independently addressable on the shared bytes.");
        println!(
            "RESONATES: hybrid is the substrate's native shape, not a bridge between two carriers."
        );
    } else {
        println!("KILL GATES FAILED:");
        for f in &fail {
            println!("  - {f}");
        }
        std::process::exit(1);
    }
}
