//! Small deterministic CLI fixture for the Haspelmath wiring seam.
//! Run: cargo run --manifest-path crates/deepnsm-v2/Cargo.toml --example haspelmath_wiring
//! The witness and grammatical-function assignment are TEST INPUTS, not mined
//! evidence, and no CE64 truth/causal headers are minted by this example.

use deepnsm_v2::fsm::{Pos, PosSet};
use deepnsm_v2::haspelmath::{
    bind_tokens, run_harness, DativeFunction, FormAttestation, FrozenOperandAbi,
};
use deepnsm_v2::vocab::PaletteVocab;

fn main() {
    let mut vocab = PaletteVocab::new();
    vocab.from_frequency_ranked(["doctor", "gave", "letter", "to"]);
    let p = |pos| PosSet::single(pos);
    let tokens = bind_tokens(
        &vocab,
        &[
            ("doctor", p(Pos::Noun)),
            ("gave", p(Pos::Verb)),
            ("letter", p(Pos::Noun)),
            ("to", p(Pos::Other)),
        ],
    )
    .expect("known words");

    // English 'to' covers this four-function region in Fig. 8.1.
    // "coverage_complete" is a controlled fixture assumption, not
    // automatically inferred by observing one token occurrence.
    let attestation = FormAttestation {
        word: vocab.id("to").expect("known"),
        functions: DativeFunction::Direction.bit()
            | DativeFunction::Recipient.bit()
            | DativeFunction::Purpose.bit()
            | DativeFunction::Experiencer.bit(),
        coverage_complete: true,
        evidence_root: 1,
    };
    // Register ABI is version-pinned across the entire execution epoch.
    // This tiny shared dictionary is a fixture, not the deployed global ISA.
    let abi = FrozenOperandAbi::new(
        1,
        &[
            vocab.id("doctor").unwrap(),
            vocab.id("gave").unwrap(),
            vocab.id("letter").unwrap(),
            vocab.id("to").unwrap(),
        ],
    )
    .expect("shared dictionary fixture");
    let out = run_harness(&[(1, 1)], &[tokens], &[vec![attestation]], 3, &abi)
        .expect("fixture is addressable");
    for edge in &out.encoded {
        let decoded = out.decode(edge, &abi).expect("frozen ABI decoding");
        println!(
            "verse={:?} order={} SPO=({}, {}, {}) ce64={:016x} map={:?}",
            edge.path,
            edge.order,
            vocab.word(decoded.subject).unwrap_or("OOV"),
            vocab.word(decoded.predicate).unwrap_or("OOV"),
            vocab.word(decoded.object).unwrap_or("OOV"),
            edge.word.0,
            out.attestations[0].2
        );
    }
    assert_eq!(out.encoded.len(), 1);
    assert_eq!(out.hydration.unaddressed, 0);
    assert_eq!(out.encoded[0].word.0 >> 24, 0);
}
