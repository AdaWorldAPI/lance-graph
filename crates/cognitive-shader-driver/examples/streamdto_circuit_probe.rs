//! D-STREAMDTO-0 — a real `StreamDto` through the v3 ladder to next-cycle
//! eligibility, on the shipped pieces only.
//!
//! The v3 ladder (`.claude/v3/COMPONENT-MAP.md`, `v3-substrate-primer.md` §3):
//! Φ `StreamDto` (perturbation ingress) → Ψ `PerturbationDto` → B `BusDto`
//! (commit) → Γ `ThoughtStruct`. In `cognitive-shader-driver` the same loop is
//! `ingest_codebook_indices` → `ShaderDriver::dispatch` (reads each row's
//! `CausalEdge64`, emits up to 8) → `EmitMode::Persist` names the row →
//! `persist_cycle` writes `emitted_edges[0]` back → the next dispatch reads it.
//!
//! What this probe found, link by link:
//!
//! | link | status |
//! |---|---|
//! | `StreamDto` → SoA | `ingest_codebook_indices` wrote only the BindSpace singleton (superseded by the SoA). **Missing** → completed by `ingest_codebook_indices_soa`, cycle-gated `write_row`, same per-index encoding |
//! | `dispatch` reads the mailbox | shipped (`mailbox-thoughtspace` read shim) |
//! | emitted edge → mailbox row | **missing**: the driver owned the mailbox read-only and `run()` is `&self`. Completed by `ShaderDriver::mailbox_mut` |
//! | commit | `MailboxSoA::tick` (shipped) |
//! | next cycle reads it | shipped; the edge's `s_idx` is the cascade query |
//! | #1370 eligibility | reads the committed word; the shader writes code 0 (`Open`), see `ISS-NO-EVIDENCE-WRITER-FOR-EPISTEMIC-STATE` |
//!
//! Two limits, measured here. (1) The edge's only read in `run()` is `s_idx`
//! as the cascade query, keyed on `s_idx / 4`; F/C are revised and discarded,
//! bits 59..63 are not read. A write-back is visible only when it moves the
//! query to another block whose hits differ. (2) Visibility does not wait for
//! `tick`: there is no double buffer, so a dispatch would see the write at
//! once. What separates k from k+1 is the owner's `&mut` (no dispatch runs
//! during the write) and the cycle gate (`tick` makes late writes `Stale`).
//!
//! `StreamDto::timestamp` lands in the temporal column as the caller's number.
//! It is not the cycle, and the cycle never reaches the edge: the driver passes
//! `cycle_index` as `pack`'s v1 `temporal` argument, which v2 drops.

#[path = "shared/affordance_law.rs"]
mod affordance_law;

use std::sync::Arc;

use affordance_law::{measure, raw5, LawGen};
use bgz17::base17::Base17;
use bgz17::palette::Palette;
use bgz17::palette_semiring::PaletteSemiring;
use causal_edge::edge::CausalEdge64;
#[cfg(test)]
use causal_edge::layout::{SPARE_MASK, TRUTH_MASK, TRUTH_SHIFT};
#[cfg(test)]
use cognitive_shader_driver::bindspace::BindSpace;
#[cfg(test)]
use cognitive_shader_driver::engine_bridge::ingest_codebook_indices;
use cognitive_shader_driver::engine_bridge::ingest_codebook_indices_soa;
use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome};
use cognitive_shader_driver::{
    auto_style, CognitiveShaderBuilder, CognitiveShaderDriver, ColumnWindow, EmitMode, MetaFilter,
    ShaderCrystal, ShaderDispatch, ShaderDriver, StyleSelector,
};
use lance_graph_contract::band_reading::EdgeProvenance;
use thinking_engine::dto::{SourceType, StreamDto};

const MAILBOX: u32 = 0;
const ROWS: usize = 12;

fn semiring() -> PaletteSemiring {
    let entries: Vec<Base17> = (0..16)
        .map(|i| {
            let mut dims = [0i16; 17];
            dims[0] = (i as i16) * 100;
            dims[1] = ((i as i16) * 37) % 200;
            Base17 { dims }
        })
        .collect();
    PaletteSemiring::build(&Palette { entries })
}

/// A CAUSES chain over the 16 palettes; only palette block 0 SUPPORTS itself.
///
/// The edge's only read in `run()` is `s_idx`, as the cascade query. The
/// cascade keys on the query's block (`s_idx / 4`), and a hit records the
/// scanned row, not the target. So if the query's block holds a distance-0
/// self-hit with the same predicates as block 0 (true for every block in the
/// W2 demo topology, where each block SUPPORTS itself), the persisted edge
/// leaves the output unchanged. Here block 1 only CAUSES block 2: no self-hit.
fn planes() -> [[u64; 64]; 8] {
    let mut p = [[0u64; 64]; 8];
    for (i, causes) in p[0].iter_mut().take(16).enumerate() {
        *causes |= 1u64 << (i + 1);
    }
    p[2][0] |= 1;
    p
}

fn stream(timestamp: u64) -> StreamDto {
    StreamDto {
        source: SourceType::DeepNsm,
        codebook_indices: (0..ROWS as u16).map(|i| i * 3 + 1).collect(),
        timestamp,
    }
}

/// Φ ingress into the SoA: the `StreamDto`'s fields go straight into the
/// owner's mailbox through `ingest_codebook_indices_soa` (cycle-gated
/// `write_row`). BindSpace is not on the path.
fn ingest(dto: &StreamDto) -> MailboxSoA<1024> {
    let mut mb: MailboxSoA<1024> = MailboxSoA::new(MAILBOX, 0, 1.0);
    ingest_codebook_indices_soa(
        &mut mb,
        &dto.codebook_indices,
        dto.source as u8,
        dto.timestamp,
        0,
    );
    mb
}

/// A mailbox driver: no BindSpace (D-MBX-CUTOVER-0).
fn driver(mb: MailboxSoA<1024>) -> ShaderDriver {
    CognitiveShaderBuilder::new()
        .semiring(Arc::new(semiring()))
        .planes(planes())
        .with_mailbox(MAILBOX, mb)
        .build()
}

fn request() -> ShaderDispatch {
    ShaderDispatch {
        // Start at row 4. The driver emits `s_palette = row % 256`; the
        // cascade keys on `s_idx / 4`, so a persisted query in block 0 (rows
        // 0..4) is the block every unwritten row (edge 0) already uses, and
        // k+1 cannot see it. The wire paths already pass a non-zero
        // `row_start`.
        rows: ColumnWindow::new(4, ROWS as u32),
        meta_prefilter: MetaFilter::ALL,
        layer_mask: 0xFF,
        radius: u16::MAX,
        style: StyleSelector::Ordinal(auto_style::CREATIVE),
        emit: EmitMode::Persist,
        ..Default::default()
    }
}

/// B → state: the row `EmitMode::Persist` names receives `emitted_edges[0]`
/// (the same choice `engine_bridge::persist_cycle` makes for the singleton),
/// written by the owner at the current cycle. Returns `(row, edge)`.
fn persist(d: &mut ShaderDriver, c: &ShaderCrystal) -> (usize, CausalEdge64) {
    let row = c.persisted_row.expect("EmitMode::Persist names a row") as usize;
    assert!(c.bus.emitted_edge_count > 0, "no edge was emitted");
    let edge = CausalEdge64(c.bus.emitted_edges[0]);
    let mb = d.mailbox_mut(MAILBOX).expect("the designated mailbox");
    let cell = WriteCell {
        edge: Some(edge),
        ..Default::default()
    };
    assert_eq!(mb.write_row(row, mb.cycle(), &cell), WriteOutcome::Accepted);
    (row, edge)
}

/// Commit: the existing cycle boundary.
fn commit(d: &mut ShaderDriver) {
    d.mailbox_mut(MAILBOX)
        .expect("the designated mailbox")
        .tick();
}

/// What a dispatch produced, comparable across runs.
fn signature(c: &ShaderCrystal) -> (Vec<(u32, u32)>, [u64; 8], u8) {
    let hits = c
        .bus
        .resonance
        .top_k
        .iter()
        .map(|h| (h.row, h.resonance.to_bits()))
        .collect();
    (hits, c.bus.emitted_edges, c.bus.emitted_edge_count)
}

/// Cycle k then cycle k+1 on one owned driver. `write` disables the persist.
fn run(
    write: bool,
) -> (
    ShaderDriver,
    ShaderCrystal,
    ShaderCrystal,
    usize,
    CausalEdge64,
) {
    let mut d = driver(ingest(&stream(1_000)));
    let k = d.dispatch(&request());
    let (row, edge) = if write {
        persist(&mut d, &k)
    } else {
        (k.persisted_row.unwrap_or(0) as usize, CausalEdge64(0))
    };
    commit(&mut d);
    let k1 = d.dispatch(&request());
    (d, k, k1, row, edge)
}

fn main() {
    let (d, k, k1, row, edge) = run(true);
    let mb = d.mailbox(MAILBOX).unwrap();
    println!("D-STREAMDTO-0");
    println!(
        "cycle k:   hits {} emitted {} persisted_row {:?}",
        k.bus.resonance.hit_count, k.bus.emitted_edge_count, k.persisted_row
    );
    println!(
        "persisted: row {row} edge {:#018x} s_idx {} raw5 {} pearl {:03b}",
        edge.0,
        edge.s_idx(),
        raw5(edge),
        edge.causal_mask() as u8
    );
    println!(
        "cycle k+1: mailbox cycle {} hits {} changed {}",
        mb.cycle(),
        k1.bus.resonance.hit_count,
        signature(&k) != signature(&k1)
    );
    println!(
        "eligibility at k+1: {:?}",
        measure(LawGen::V1, mb.edge(row), EdgeProvenance::V2Stamped)
    );
    println!(
        "temporal[{row}] = {} (StreamDto.timestamp)",
        mb.temporal_at(row)
    );
    let t = LawGen::V1.tables();
    let pearl = t.pearl[edge.causal_mask() as usize];
    for code in [0usize, 1, 3, 5, 7, 12, 17, 20, 25, 30] {
        println!(
            "  code {code:2} under this Pearl: {:#x}",
            t.state[code] & pearl
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// S1 — every link produces something. The SoA ingest writes one bit per
    /// index and encodes each row exactly as the retiring singleton arm does
    /// (that arm is only the oracle here); cycle k hits, emits, names a row.
    #[test]
    fn every_link_is_live() {
        let dto = stream(1_000);
        let mb = ingest(&dto);
        assert_eq!(mb.populated(), ROWS);
        let mut oracle = BindSpace::zeros(ROWS);
        ingest_codebook_indices(
            &mut oracle,
            &dto.codebook_indices,
            dto.source as u8,
            dto.timestamp,
            0,
        );
        for row in 0..ROWS {
            let ones: u32 = mb.content_row(row).iter().map(|w| w.count_ones()).sum();
            assert_eq!(ones, 1, "row {row}");
            assert_eq!(mb.content_row(row), oracle.fingerprints.content_row(row));
            assert_eq!(mb.meta_at(row), oracle.meta.get(row));
            assert_eq!(mb.temporal_at(row), oracle.temporal[row]);
            assert_eq!(
                mb.last_write_cycle_at(row),
                mb.cycle(),
                "ingress went through write_row"
            );
        }
        let d = driver(mb);
        let k = d.dispatch(&request());
        assert!(k.bus.resonance.hit_count > 0);
        assert!(k.bus.emitted_edge_count > 0);
        assert!(k.persisted_row.is_some());
    }

    /// S2 — the circuit closes: cycle k+1 reads the edge cycle k emitted, and
    /// reading it changes what k+1 computes. Without the write, k+1 repeats k.
    #[test]
    fn the_next_cycle_reads_what_this_cycle_emitted() {
        let (d, k, k1, row, edge) = run(true);
        assert_ne!(
            edge.s_idx(),
            0,
            "fixture: the persisted edge must move the query"
        );
        assert_eq!(d.mailbox(MAILBOX).unwrap().edge(row), edge);
        assert_ne!(
            signature(&k),
            signature(&k1),
            "k+1 ignored the persisted edge"
        );

        let (_, k_off, k1_off, _, _) = run(false);
        assert_eq!(
            signature(&k_off),
            signature(&k1_off),
            "nothing else changes k+1"
        );
        assert_eq!(
            signature(&k),
            signature(&k_off),
            "cycle k does not depend on the write"
        );
    }

    /// S3 — the commit boundary is `tick`. Before it the write is this
    /// cycle's; after it a late write for the old cycle is refused.
    #[test]
    fn tick_is_the_commit_and_the_cycle_gate_holds() {
        let mut d = driver(ingest(&stream(1_000)));
        let k = d.dispatch(&request());
        let cycle_k = d.mailbox(MAILBOX).unwrap().cycle();
        let (row, edge) = persist(&mut d, &k);
        let mb = d.mailbox(MAILBOX).unwrap();
        assert_eq!(
            mb.last_write_cycle_at(row),
            mb.cycle(),
            "uncommitted: written this cycle"
        );

        commit(&mut d);
        let mb = d.mailbox_mut(MAILBOX).unwrap();
        assert_eq!(mb.cycle(), cycle_k.wrapping_add(1));
        assert!(mb.last_write_cycle_at(row) != mb.cycle(), "committed");
        let late = WriteCell {
            edge: Some(CausalEdge64(0)),
            ..Default::default()
        };
        assert_eq!(mb.write_row(row, cycle_k, &late), WriteOutcome::Stale);
        assert_eq!(mb.edge(row), edge);
    }

    /// S4 — #1370 eligibility at k+1 is measured on the committed row, and
    /// the circuit does not move it: the shader stamps code 0 (`Open`), which
    /// under the edge's Pearl planes grants exactly what an unwritten word
    /// grants. The second half proves the measurement reads the row: an
    /// evidence code committed into the same row changes it.
    #[test]
    fn eligibility_at_k1_reads_the_committed_row_and_the_circuit_leaves_it_open() {
        let (mut d, _, _, row, edge) = run(true);
        let committed = d.mailbox(MAILBOX).unwrap().edge(row);
        assert_eq!(raw5(committed), 0, "the shader writes no epistemic state");
        let t = LawGen::V1.tables();
        let open = measure(LawGen::V1, committed, EdgeProvenance::V2Stamped);
        assert_eq!(open, Ok(t.state[0] & t.pearl[edge.causal_mask() as usize]));
        assert_eq!(
            open,
            measure(LawGen::V1, CausalEdge64(0), EdgeProvenance::V2Stamped),
            "the circuit moved eligibility"
        );

        // Code 7 (DIRECT | OBSERVED | ASSOCIATED) into the same row, committed
        // by tick. Not code 3: under Pearl S it grants the same 0x20 as code 0
        // (the table `main` prints).
        let observed =
            CausalEdge64((committed.0 & !(TRUTH_MASK | SPARE_MASK)) | (7u64 << TRUTH_SHIFT));
        let mb = d.mailbox_mut(MAILBOX).unwrap();
        let cell = WriteCell {
            edge: Some(observed),
            ..Default::default()
        };
        assert_eq!(mb.write_row(row, mb.cycle(), &cell), WriteOutcome::Accepted);
        mb.tick();
        let reread = measure(LawGen::V1, mb.edge(row), EdgeProvenance::V2Stamped);
        assert_eq!(
            reread,
            Ok(t.state[7] & t.pearl[edge.causal_mask() as usize])
        );
        assert_ne!(reread, open, "the measurement does not read the row");
    }

    /// S5 — timestamp is the caller's number in the temporal column; the cycle
    /// is the mailbox's counter. Neither moves the other, and the dispatch's
    /// `cycle_index` never reaches the edge (v2 drops `pack`'s temporal).
    #[test]
    fn timestamp_is_not_the_cycle() {
        let early = ingest(&stream(5));
        let late = ingest(&stream(9_999_999));
        assert_eq!(early.cycle(), late.cycle());
        assert_eq!(early.temporal_at(0), 5);
        assert_eq!(late.temporal_at(0), 9_999_999);

        let (d, k, _, row, _) = run(true);
        assert_eq!(
            d.mailbox(MAILBOX).unwrap().temporal_at(row),
            1_000,
            "tick left temporal"
        );
        for &e in &k.bus.emitted_edges[..k.bus.emitted_edge_count as usize] {
            assert_eq!(e >> 59, 0, "cycle_index leaked into bits 59..63");
        }
    }

    /// S6 — restart: the committed word persisted as LE bytes and restored into
    /// a fresh mailbox at a different cycle gives the same cycle k+1.
    #[test]
    fn restart_from_the_persisted_word_gives_the_same_next_cycle() {
        let (d, _, k1, row, _) = run(true);
        let bytes = d.mailbox(MAILBOX).unwrap().edge(row).to_le_bytes();

        let mut mb = ingest(&stream(1_000));
        for _ in 0..7 {
            mb.tick();
        }
        let cell = WriteCell {
            edge: Some(CausalEdge64::from_le_bytes(bytes)),
            ..Default::default()
        };
        assert_eq!(mb.write_row(row, mb.cycle(), &cell), WriteOutcome::Accepted);
        mb.tick();
        let restored = driver(mb).dispatch(&request());
        assert_eq!(signature(&restored), signature(&k1));
    }
}
