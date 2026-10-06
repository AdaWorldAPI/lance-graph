//! D-CE64-NEXTSTATE-0: EpistemicState5 as a sparse next-state mutation of the
//! `CausalEdge64` register.
//!
//! Claim under test:
//!
//! ```text
//! next = (current & !EPISTEMIC_MASK) | (code << 59)       bits 59..63
//! ```
//!
//! Every other bit of the register is carried into cycle k+1 bit-identically,
//! the write crosses the real `MailboxSoA` cycle seam, and the word survives
//! a canonical little-endian round trip.
//!
//! # Bit numbering and byte order
//!
//! Logical bit 0 is the least significant bit of the `u64`. The canonical
//! persisted form is `u64::to_le_bytes`, so bits 59..63 sit in byte 7, bits
//! 3..7. Nothing here writes a byte: the update is a `u64` mask and insert.
//!
//! # Tests
//!
//! | test | what it pins |
//! |---|---|
//! | N1 | only bits 59..63 move, for every code on an edge with every field set |
//! | N2 | the certified state is read by the next cycle through the real mailbox |
//! | N3 | what the mailbox does and does not enforce about same-cycle authority |
//! | N4 | asymmetric words round-trip through canonical LE bytes |
//! | N5 | a fresh mailbox built from the persisted word continues identically |
//! | N6 | the existing two-writer path and the masked insert agree on all inputs |
//!
//! # Codegen (N6, measured, not a test)
//!
//! `cargo rustc --release --example ce64_nextstate_probe -- --emit asm` on
//! x86-64: `via_two_writers` (the shipped `with_spare` + `with_truth`) is
//! `bzhi rax, rdi, 59; shl rsi, 59; or rax, rsi; ret`, i.e. clear bits
//! 59..63 and insert. `via_mask` compiled to the identical body and LLVM
//! folded the two into one symbol. The shipped writers do not force a
//! decode/repack, so this probe adds no production writer for the 5-bit
//! field: it would be a fourth name over bits that already have four lenses
//! (`truth`, `topology`, `spare`, `reasoning_band`) and would not change the
//! machine code.
//!
//! Run: `cargo run -p cognitive-shader-driver --example ce64_nextstate_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example ce64_nextstate_probe`

#[path = "shared/affordance_law.rs"]
mod affordance_law;

use affordance_law::{measure, raw5, LawGen};
use causal_edge::edge::CausalEdge64;
use causal_edge::layout::{TrustTexture, SPARE_MASK, TRUTH_MASK, TRUTH_SHIFT};
use causal_edge::pearl::CausalMask;
use causal_edge::PlasticityState;
use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome};
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::soa_view::MailboxSoaView;

/// Bits 59..63.
const EPISTEMIC_MASK: u64 = TRUTH_MASK | SPARE_MASK;

/// The register row.
const ROW: usize = 0;

const CODE_OBSERVED: u8 = 3;
const CODE_ASSOCIATED: u8 = 7;

/// The existing field-update path: two shipped writers, one per sub-field.
#[inline(never)]
fn via_two_writers(edge: CausalEdge64, code: u8) -> CausalEdge64 {
    edge.with_spare(code >> 2)
        .with_truth(TrustTexture::from_bits_2(code & 0b11))
}

/// The minimal sparse update: one mask and one insert on the `u64`.
#[inline(never)]
fn via_mask(edge: CausalEdge64, code: u8) -> CausalEdge64 {
    CausalEdge64((edge.0 & !EPISTEMIC_MASK) | ((u64::from(code) & 0x1F) << TRUTH_SHIFT))
}

/// An edge with every field non-zero, including bits 59..63.
fn busy() -> CausalEdge64 {
    CausalEdge64::pack_v2(
        0xA5,
        0x3C,
        0x7E,
        201,
        149,
        CausalMask::SO,
        0b101,
        PlasticityState::from_bits(0b011),
    )
    .with_inference_mantissa(-6)
    .with_w_slot(41)
    .with_spare(0b101)
    .with_truth(TrustTexture::from_bits_2(0b10))
}

/// The register read through the production view; `None` outside the
/// declared rows.
fn read(view: &dyn MailboxSoaView) -> Option<CausalEdge64> {
    (ROW < view.n_rows()).then(|| CausalEdge64(view.edges_raw()[ROW]))
}

/// The register as committed by an earlier cycle: `None` while the row was
/// written in the current cycle (its `last_write_cycle` equals `cycle`).
fn committed(mb: &MailboxSoA<4>) -> Option<CausalEdge64> {
    (mb.last_write_cycle_at(ROW) != mb.cycle())
        .then(|| read(mb))
        .flatten()
}

/// A mailbox with the register row declared and seeded, already ticked so
/// the seed is committed.
fn mailbox(seed: CausalEdge64) -> MailboxSoA<4> {
    let mut mb: MailboxSoA<4> = MailboxSoA::new(1, 0, 0.5);
    mb.set_populated(ROW + 1);
    write(&mut mb, seed);
    mb.tick();
    mb
}

fn write(mb: &mut MailboxSoA<4>, edge: CausalEdge64) {
    let cell = WriteCell {
        edge: Some(edge),
        ..WriteCell::default()
    };
    assert_eq!(mb.write_row(ROW, mb.cycle(), &cell), WriteOutcome::Accepted);
}

fn eligible(edge: CausalEdge64) -> u64 {
    measure(LawGen::V1, edge, EdgeProvenance::V2Stamped).expect("declared code")
}

/// The canonical persisted form of one register (the shipped
/// `CausalEdge64::to_le_bytes`).
fn persist(edge: CausalEdge64) -> [u8; 8] {
    edge.to_le_bytes()
}

fn restore(bytes: [u8; 8]) -> CausalEdge64 {
    CausalEdge64::from_le_bytes(bytes)
}

fn main() {
    let start = via_mask(busy(), CODE_OBSERVED);
    let mut mb = mailbox(start);
    let now = committed(&mb).expect("seed committed");
    let next = via_mask(now, CODE_ASSOCIATED);
    write(&mut mb, next);
    println!("D-CE64-NEXTSTATE-0");
    println!(
        "  cycle k   word {:016x}  code {}  eligible {:07b}",
        now.0,
        raw5(now),
        eligible(now)
    );
    println!(
        "  written   word {:016x}  changed bits {:016x}",
        next.0,
        now.0 ^ next.0
    );
    println!(
        "  pending in cycle k: committed() = {:?}",
        committed(&mb).map(|e| e.0)
    );
    mb.tick();
    let k1 = committed(&mb).expect("committed after tick");
    println!("  cycle k+1 word {:016x}  code {}", k1.0, raw5(k1));
    println!("  persisted LE {:02x?}", persist(k1));
    assert_eq!(restore(persist(k1)), k1);
    // Both update paths, kept out of the optimiser's reach so their code can
    // be compared (`--emit asm`).
    let (e, c) = (
        std::hint::black_box(k1),
        std::hint::black_box(CODE_OBSERVED),
    );
    println!(
        "  two writers == mask: {}",
        via_two_writers(e, c) == via_mask(e, c)
    );
}

#[cfg(test)]
mod tests {
    use super::affordance_law::{EPI_LAW, OBSERVE_FOLD, STRATIFY};
    use super::*;

    const OBS: u64 = 1 << OBSERVE_FOLD;
    const STRAT: u64 = 1 << STRATIFY;

    /// N1: setting any code moves no bit outside 59..63.
    #[test]
    fn n1_only_bits_59_to_63_move() {
        let cur = busy();
        assert!(cur.0 & !EPISTEMIC_MASK != 0 && cur.0 & EPISTEMIC_MASK != 0);
        let mut moved = 0;
        for code in 0u8..32 {
            let next = via_mask(cur, code);
            assert_eq!((cur.0 ^ next.0) & !EPISTEMIC_MASK, 0, "code {code}");
            assert_eq!(raw5(next), code);
            if next != cur {
                moved += 1;
            }
        }
        assert_eq!(moved, 31, "every other code changes the word");
    }

    /// N2: written in cycle k through the real seam, read in cycle k+1.
    #[test]
    fn n2_the_next_cycle_reads_the_certified_state() {
        let mut mb = mailbox(via_mask(busy(), CODE_OBSERVED));
        let now = committed(&mb).unwrap();
        assert_eq!(eligible(now), OBS);
        let next = via_mask(now, CODE_ASSOCIATED);
        write(&mut mb, next);
        mb.tick();
        let k1 = committed(&mb).unwrap();
        assert_eq!(k1, next);
        assert_eq!(eligible(k1), OBS | STRAT);
        assert_eq!((now.0 ^ k1.0) & !EPISTEMIC_MASK, 0);
    }

    /// N3: the mailbox does not double-buffer. A write in cycle k lands in
    /// place and a plain read sees it at once; the only distinction the
    /// mailbox keeps is the write stamp, which marks the row as written this
    /// cycle. A committed-only read therefore refuses the row until `tick`,
    /// but the pre-write value is gone; a cycle that needs it must keep the
    /// copy it read at cycle start.
    #[test]
    fn n3_same_cycle_authority_is_a_read_discipline() {
        let mut mb = mailbox(via_mask(busy(), CODE_OBSERVED));
        let at_start = committed(&mb).unwrap();
        let eligible_k = eligible(at_start);
        write(&mut mb, via_mask(at_start, CODE_ASSOCIATED));
        // Not enforced: the plain read already sees the new state.
        assert_eq!(raw5(read(&mb).unwrap()), CODE_ASSOCIATED);
        // The stamp marks it pending; a committed-only read refuses it.
        assert_eq!(committed(&mb), None);
        // Cycle k's eligibility came from the start-of-cycle copy and stays.
        assert_eq!(eligible(at_start), eligible_k);
        assert_eq!(eligible_k, OBS);
        mb.tick();
        assert_eq!(raw5(committed(&mb).unwrap()), CODE_ASSOCIATED);
    }

    /// N4: asymmetric words survive the canonical LE round trip, and bits
    /// 59..63 sit in byte 7, bits 3..7.
    #[test]
    fn n4_canonical_le_round_trip() {
        let words = [
            0x0123_4567_89AB_CDEF_u64,
            0xF800_0000_0000_0001,
            0x0800_0000_0000_0000,
            0x8000_0000_0000_0000,
            0xA5C3_0F1E_2D3C_4B5A,
            u64::MAX ^ 1,
            busy().0,
        ];
        for w in words {
            let bytes = persist(CausalEdge64(w));
            assert_eq!(restore(bytes).0, w, "{w:016x}");
            assert_eq!(bytes[0], (w & 0xFF) as u8, "byte 0 is the low byte");
            assert_eq!(bytes[7] >> 3, ((w & EPISTEMIC_MASK) >> TRUTH_SHIFT) as u8);
        }
    }

    /// N5: a fresh mailbox seeded only from the persisted word continues
    /// exactly like the original; no evidence is replayed.
    #[test]
    fn n5_restart_from_the_persisted_word() {
        let mut mb = mailbox(via_mask(busy(), CODE_OBSERVED));
        let now = committed(&mb).unwrap();
        write(&mut mb, via_mask(now, CODE_ASSOCIATED));
        mb.tick();
        let original = committed(&mb).unwrap();
        let fresh = mailbox(restore(persist(original)));
        let restarted = committed(&fresh).unwrap();
        assert_eq!(restarted, original);
        assert_eq!(eligible(restarted), eligible(original));
        assert_eq!(raw5(restarted), CODE_ASSOCIATED);
    }

    /// N6: the existing two-writer path and the masked insert produce the
    /// same word for every code on varied edges, so the shipped API does not
    /// force a decode/repack to get the sparse update.
    #[test]
    fn n6_existing_path_equals_the_masked_insert() {
        let edges = [
            CausalEdge64::ZERO,
            busy(),
            CausalEdge64(u64::MAX),
            CausalEdge64(0xA5C3_0F1E_2D3C_4B5A),
        ];
        for e in edges {
            for code in 0u8..32 {
                assert_eq!(
                    via_two_writers(e, code),
                    via_mask(e, code),
                    "{:016x} {code}",
                    e.0
                );
            }
        }
        // Every declared code of the #1370 law is reachable this way.
        let declared = (0u8..32).filter(|c| EPI_LAW[*c as usize].is_some()).count();
        assert_eq!(declared, 10);
    }
}
