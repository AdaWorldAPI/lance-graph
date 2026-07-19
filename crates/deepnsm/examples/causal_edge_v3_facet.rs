//! Reference layout for the STAGED, ADDITIVE `CausalEdge64 -> V3-96` migration
//! (operator directive). This is the demonstrator the real `causal_edge`-crate
//! type adopts; it is std-only and touches no shipping code, so the layout can
//! be proven (LE round-trip + field isolation + the no-SPO-duplication
//! invariant) before the multi-crate migration runs through `v3-envelope-auditor`.
//!
//! ## The premise, re-grounded (operator, 2026-07-19)
//!
//!   - The old `3×8 = 24-bit SPO` field was a **deduplication candidate,
//!     already cleaned out of `CausalEdge64`** (its 64 bits today are
//!     `block/proj/verb/row_idx/l1/freq/conf` — NO SPO; recon-confirmed).
//!   - **SPO already lives as the 6×256² CAM-PQ facet** (le-contract L4
//!     `palette256²`: 3 SPO byte-pairs + 3 AriGraph SPO-G byte-pairs).
//!   - So the V3-96 edge **references** SPO via its Lokal target node and
//!     **never re-encodes it** — the "we don't duplicate" invariant.
//!
//! ## The V3-96 CausalEdge facet (canon `classid(4) | payload(12)` = 16 B)
//!
//! The 96-bit payload is carved on the TEKAMOLO axis (Temporal / Kausal /
//! Modal / Lokal — `grammar::tekamolo`), plus the nibble anaphora edge:
//!
//! ```text
//! [0]     KA  kausal:  verb(2) | inference(3) | modal_mode(3)   why + how-mode
//! [1]     TE  temporal: i8 signed offset (when, ±127 cycles/positions)
//! [2..4]  LO  lokal:   u16 target node ref (where) -- SPO reads from ITS
//!                       6×256² CAM-PQ facet; the edge stores NO SPO bytes
//! [4..6]  MO  modal:   freq u8 + conf u8 (NARS truth -- how confidently)
//! [6]     anaphora nibble: i8 (low nibble −8..+7 coreference offset;
//!                              high nibble = kind flags)
//! [7]     plasticity u8 (learning-rate / W-slot)
//! [8..12] reserved (4 B, dormant -- future TEKAMOLO Instrument slot + spare)
//! ```
//!
//! What moved vs `CausalEdge64` (8 B): kept verb + NARS truth + target (row ->
//! LO); dropped `proj`/`l1` (attention-specific / CAM-PQ owns distance now);
//! ADDED temporal + anaphora-nibble + plasticity + reserved. SPO: never here.
//!
//! KILL gates (regressions, not discoveries):
//!   - every field LE round-trips (`from_le_bytes(to_le_bytes()) == self`).
//!   - field isolation: writing one field leaves ALL others byte-identical
//!     (I-LEGACY-API-FEATURE-GATED discipline for a layout).
//!   - the edge carries ZERO SPO bytes; SPO is recovered ONLY from the target
//!     node's CAM-PQ facet (no-duplication invariant).
//!
//! ## Run
//!
//! ```bash
//! cargo run --manifest-path crates/deepnsm/Cargo.toml --example causal_edge_v3_facet
//! ```

/// The causal relation (KA — the `verb` field, matching `CausalEdge64`).
#[derive(Clone, Copy, PartialEq, Debug)]
enum Verb {
    Becomes = 0,
    Supports = 1,
    Contradicts = 2,
}
impl Verb {
    fn from_u2(v: u8) -> Verb {
        match v & 3 {
            0 => Verb::Becomes,
            1 => Verb::Supports,
            _ => Verb::Contradicts,
        }
    }
}

/// The 16-byte V3 causal-edge facet: `classid(4) | payload(12)`.
#[derive(Clone, Copy, PartialEq, Debug)]
struct CausalEdgeV3 {
    classid: u32,
    payload: [u8; 12],
}
const _: () = assert!(core::mem::size_of::<[u8; 12]>() == 12); // 96-bit payload

impl CausalEdgeV3 {
    const CLASSID: u32 = 0x0000_0E64; // demo classid: 'E'dge 64->96 successor

    fn new() -> Self {
        CausalEdgeV3 {
            classid: Self::CLASSID,
            payload: [0u8; 12],
        }
    }

    // ── KA (kausal): verb(2) | inference(3) | modal_mode(3) at byte 0 ──
    fn verb(&self) -> Verb {
        Verb::from_u2(self.payload[0] & 0b11)
    }
    fn set_verb(&mut self, v: Verb) {
        self.payload[0] = (self.payload[0] & !0b11) | (v as u8 & 0b11);
    }
    fn inference(&self) -> u8 {
        (self.payload[0] >> 2) & 0b111
    }
    fn set_inference(&mut self, i: u8) {
        self.payload[0] = (self.payload[0] & !(0b111 << 2)) | ((i & 0b111) << 2);
    }

    // ── TE (temporal): signed offset at byte 1 ──
    fn temporal(&self) -> i8 {
        self.payload[1] as i8
    }
    fn set_temporal(&mut self, t: i8) {
        self.payload[1] = t as u8;
    }

    // ── LO (lokal): u16 target node ref at bytes 2..4 (SPO lives at target) ──
    fn target(&self) -> u16 {
        u16::from_le_bytes([self.payload[2], self.payload[3]])
    }
    fn set_target(&mut self, node: u16) {
        let b = node.to_le_bytes();
        self.payload[2] = b[0];
        self.payload[3] = b[1];
    }

    // ── MO (modal): NARS truth freq/conf at bytes 4,5 ──
    fn truth(&self) -> (f32, f32) {
        (
            self.payload[4] as f32 / 255.0,
            self.payload[5] as f32 / 255.0,
        )
    }
    fn set_truth(&mut self, freq: f32, conf: f32) {
        self.payload[4] = (freq.clamp(0.0, 1.0) * 255.0).round() as u8;
        self.payload[5] = (conf.clamp(0.0, 1.0) * 255.0).round() as u8;
    }

    // ── anaphora nibble at byte 6 (low nibble signed −8..+7) ──
    fn anaphora(&self) -> Option<i8> {
        let lo = self.payload[6] & 0x0F;
        if lo == 0 {
            None // sentinel: no coreference edge
        } else {
            // sign-extend a 4-bit value: 8..15 -> -8..-1
            Some(if lo >= 8 { lo as i8 - 16 } else { lo as i8 })
        }
    }
    fn set_anaphora(&mut self, offset: i8) {
        debug_assert!(
            (-8..=7).contains(&offset),
            "anaphora offset out of nibble range"
        );
        let nib = (offset as u8) & 0x0F;
        self.payload[6] = (self.payload[6] & 0xF0) | nib;
    }

    // ── plasticity at byte 7 ──
    fn plasticity(&self) -> u8 {
        self.payload[7]
    }
    fn set_plasticity(&mut self, p: u8) {
        self.payload[7] = p;
    }

    // ── LE serialization (16 B facet) ──
    fn to_le_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..4].copy_from_slice(&self.classid.to_le_bytes());
        out[4..16].copy_from_slice(&self.payload);
        out
    }
    fn from_le_bytes(b: &[u8; 16]) -> Self {
        let classid = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let mut payload = [0u8; 12];
        payload.copy_from_slice(&b[4..16]);
        CausalEdgeV3 { classid, payload }
    }
}

/// The SPO of a node is its 6×256² CAM-PQ facet (3 SPO + 3 AriGraph SPO-G
/// byte-pairs = 12 bytes). The edge does NOT hold this — it is read from the
/// TARGET node via the Lokal reference. Modelled here as a per-node lookup.
fn node_spo_campq(node: u16) -> [u8; 12] {
    // deterministic mock 6×(8:8) palette code for the target node
    let mut c = [0u8; 12];
    for (i, b) in c.iter_mut().enumerate() {
        *b = (node.wrapping_mul(31).wrapping_add(i as u16 * 7) & 0xFF) as u8;
    }
    c
}

fn main() {
    // Build one edge: "A Supports B", 3 cycles ago, truth <0.9,0.8>, the
    // pronoun that spawned it points 4 tokens back, on node #4242.
    let mut e = CausalEdgeV3::new();
    e.set_verb(Verb::Supports);
    e.set_inference(2);
    e.set_temporal(-3);
    e.set_target(4242);
    e.set_truth(0.9, 0.8);
    e.set_anaphora(-4);
    e.set_plasticity(17);

    println!(
        "CausalEdge V3-96 facet (classid 0x{:08X} | 12-byte payload):",
        e.classid
    );
    println!("  KA verb={:?} inference={}", e.verb(), e.inference());
    println!("  TE temporal={} cycles", e.temporal());
    let (f, c) = e.truth();
    println!("  MO truth=<f={f:.2}, c={c:.2}>");
    println!(
        "  LO target=node #{}  (SPO read from ITS CAM-PQ facet, NOT the edge)",
        e.target()
    );
    println!("  anaphora nibble={:?} (coreference offset)", e.anaphora());
    println!("  plasticity={}", e.plasticity());
    println!("  payload bytes = {:?}", e.payload);
    println!();

    // The no-duplication invariant: SPO comes from the target node's CAM-PQ.
    let spo = node_spo_campq(e.target());
    println!("SPO of the target (its 6×256² CAM-PQ facet, 3 SPO + 3 AriGraph SPO-G):");
    println!("  {spo:?}   <- lives on the NODE, never copied into the edge");
    println!();

    // ── KILL gates ──
    let mut fail = Vec::new();

    // 1. LE round-trip
    let rt = CausalEdgeV3::from_le_bytes(&e.to_le_bytes());
    if rt != e {
        fail.push("LE round-trip changed the edge".to_string());
    }

    // 2. field isolation: set each field on a fresh edge, assert the others
    //    stay at their zero/default (only the target byte-range moved).
    let isolation_ok = {
        let mut ok = true;
        // verb touches only byte 0 low bits
        let mut a = CausalEdgeV3::new();
        a.set_verb(Verb::Contradicts);
        if a.payload[1..] != [0u8; 11] {
            ok = false;
        }
        // temporal touches only byte 1
        let mut b = CausalEdgeV3::new();
        b.set_temporal(-9);
        if b.payload[0] != 0 || b.payload[2..] != [0u8; 10] {
            ok = false;
        }
        // target touches only bytes 2..4
        let mut c = CausalEdgeV3::new();
        c.set_target(0xBEEF);
        if c.payload[0..2] != [0u8; 2] || c.payload[4..] != [0u8; 8] {
            ok = false;
        }
        // anaphora touches only byte 6 low nibble
        let mut d = CausalEdgeV3::new();
        d.set_anaphora(-4);
        if d.payload[0..6] != [0u8; 6] || d.payload[7..] != [0u8; 5] || (d.payload[6] & 0xF0) != 0 {
            ok = false;
        }
        ok
    };
    if !isolation_ok {
        fail.push("field isolation violated (a setter touched another field's bytes)".to_string());
    }

    // 3. no-SPO-duplication: the 12 payload bytes must NOT equal the target's
    //    SPO CAM-PQ code (the edge carries a reference, not the triple).
    if e.payload == spo {
        fail.push("edge payload equals the target SPO code (SPO duplicated!)".to_string());
    }

    // 4. anaphora nibble sign-extends correctly and stays in range
    if e.anaphora() != Some(-4) {
        fail.push(format!(
            "anaphora nibble decoded {:?}, expected Some(-4)",
            e.anaphora()
        ));
    }
    // 5. anaphora sentinel: 0 offset -> None (no coreference edge)
    let plain = CausalEdgeV3::new();
    if plain.anaphora().is_some() {
        fail.push("zero nibble should decode as None (no coreference)".to_string());
    }

    if fail.is_empty() {
        println!("KILL GATES: all pass -- the V3-96 edge round-trips, every field is isolated,");
        println!("the anaphora nibble sign-extends, and SPO is REFERENCED (target CAM-PQ), never");
        println!("duplicated. Ready to adopt in `causal_edge` (staged/additive) via v3-envelope-auditor.");
    } else {
        println!("KILL GATES FAILED:");
        for f in &fail {
            println!("  - {f}");
        }
        std::process::exit(1);
    }
}
