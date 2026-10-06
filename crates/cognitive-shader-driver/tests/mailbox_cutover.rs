//! D-MBX-CUTOVER-0 — a `ShaderDriver` built on mailboxes stands without BindSpace.
//!
//! One test per cutover property:
//! - no dummy BindSpace is needed;
//! - `backing()` never falls back to the singleton (the old code fell back
//!   whenever no mailbox sat under id 0);
//! - each registered mailbox keeps its own CE64 population;
//! - the ontology is the driver's own handle, not the substrate's;
//! - `row_count` / `byte_footprint` report the selected mailbox;
//! - `tick` and the cycle stay on the mailbox; an attached BindSpace is never
//!   touched.

#![cfg(feature = "mailbox-thoughtspace")]

use std::sync::Arc;

use bgz17::base17::Base17;
use bgz17::palette::Palette;
use bgz17::palette_semiring::PaletteSemiring;
use causal_edge::CausalEdge64;
use cognitive_shader_driver::bindspace::{BindSpace, WORDS_PER_FP};
use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome};
use cognitive_shader_driver::{
    auto_style, CognitiveShaderBuilder, CognitiveShaderDriver, ColumnWindow, MetaFilter, MetaWord,
    ShaderCrystal, ShaderDispatch, ShaderDriver, StyleSelector,
};
use lance_graph_ontology::OntologyRegistry;

const ROWS: usize = 8;

fn semiring() -> Arc<PaletteSemiring> {
    let entries: Vec<Base17> = (0..16)
        .map(|i| {
            let mut dims = [0i16; 17];
            dims[0] = (i as i16) * 100;
            dims[1] = ((i as i16) * 37) % 200;
            Base17 { dims }
        })
        .collect();
    Arc::new(PaletteSemiring::build(&Palette { entries }))
}

/// A CAUSES chain over the 16 palettes; only block 0 SUPPORTS itself, so a
/// query in another block has no distance-0 self-hit and its edge shows in
/// the output (see `streamdto_circuit_probe`).
fn planes() -> [[u64; 64]; 8] {
    let mut p = [[0u64; 64]; 8];
    for (i, causes) in p[0].iter_mut().take(16).enumerate() {
        *causes |= 1u64 << (i + 1);
    }
    p[2][0] |= 1;
    p
}

/// A mailbox whose rows all carry the CE64 `s_idx = query`.
fn mailbox(id: u32, rows: usize, query: u64) -> MailboxSoA<1024> {
    let mut mb: MailboxSoA<1024> = MailboxSoA::new(id, 0, 1.0);
    for row in 0..rows {
        let mut content = [0u64; WORDS_PER_FP];
        content[row % WORDS_PER_FP] = 1 << (row % 64);
        let cell = WriteCell {
            content: Some(&content),
            meta: Some(MetaWord::new(2, 2, 200, 200, 1)),
            edge: Some(CausalEdge64(query)),
            ..Default::default()
        };
        assert_eq!(mb.write_row(row, mb.cycle(), &cell), WriteOutcome::Accepted);
    }
    mb.set_populated(rows);
    mb
}

fn request() -> ShaderDispatch {
    ShaderDispatch {
        rows: ColumnWindow::new(0, ROWS as u32),
        meta_prefilter: MetaFilter::ALL,
        layer_mask: 0xFF,
        radius: u16::MAX,
        style: StyleSelector::Ordinal(auto_style::CREATIVE),
        ..Default::default()
    }
}

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

fn on(mb: MailboxSoA<1024>, id: u32) -> ShaderDriver {
    CognitiveShaderBuilder::new()
        .semiring(semiring())
        .planes(planes())
        .with_mailbox(id, mb)
        .build()
}

/// C1 — a mailbox driver builds and dispatches with no BindSpace at all.
#[test]
fn a_mailbox_driver_needs_no_bindspace() {
    let d = on(mailbox(0, ROWS, 0), 0);
    assert!(d.bindspace().is_none());
    assert_eq!(d.selected_mailbox(), Some(0));
    assert!(d.dispatch(&request()).bus.resonance.hit_count > 0);
}

/// C2 — no fallback: a mailbox under a non-zero id is read, even with a
/// legacy BindSpace attached. The old `backing()` looked up id 0 only and
/// read the singleton otherwise.
#[test]
fn a_mailbox_under_any_id_is_read_never_the_singleton() {
    let legacy = Arc::new(BindSpace::zeros(3));
    let d = CognitiveShaderBuilder::new()
        .bindspace(legacy)
        .semiring(semiring())
        .planes(planes())
        .with_mailbox(7, mailbox(7, ROWS, 0))
        .build();
    assert_eq!(d.selected_mailbox(), Some(7));
    assert_eq!(d.row_count(), ROWS as u32, "read the 3-row singleton");
    assert_eq!(
        signature(&d.dispatch(&request())),
        signature(&on(mailbox(7, ROWS, 0), 7).dispatch(&request()))
    );
}

/// C3 — two mailboxes keep two CE64 populations: selecting each gives its
/// own dispatch, and a write into one leaves the other untouched.
#[test]
fn each_mailbox_keeps_its_own_ce64_population() {
    let mut d = CognitiveShaderBuilder::new()
        .semiring(semiring())
        .planes(planes())
        .with_mailbox(1, mailbox(1, ROWS, 0))
        .with_mailbox(2, mailbox(2, ROWS, 4))
        .select_mailbox(1)
        .build();
    let a = signature(&d.dispatch(&request()));
    assert!(d.select_mailbox(2));
    let b = signature(&d.dispatch(&request()));
    assert_ne!(a, b, "the two populations dispatch alike");
    assert!(!d.select_mailbox(9), "an unregistered id was accepted");
    assert_eq!(d.selected_mailbox(), Some(2));

    let before = d.mailbox(1).unwrap().edge(0);
    let mb2 = d.mailbox_mut(2).unwrap();
    let cell = WriteCell {
        edge: Some(CausalEdge64(8)),
        ..Default::default()
    };
    assert_eq!(mb2.write_row(0, mb2.cycle(), &cell), WriteOutcome::Accepted);
    assert_eq!(d.mailbox(1).unwrap().edge(0), before);
    assert!(d.select_mailbox(1));
    assert_eq!(signature(&d.dispatch(&request())), a);
}

#[test]
#[should_panic(expected = "select one")]
fn several_mailboxes_without_a_selection_refuse_to_build() {
    let _ = CognitiveShaderBuilder::new()
        .semiring(semiring())
        .with_mailbox(1, mailbox(1, ROWS, 0))
        .with_mailbox(2, mailbox(2, ROWS, 0))
        .build();
}

#[test]
#[should_panic(expected = "bindspace required")]
fn no_mailbox_and_no_bindspace_refuses_to_build() {
    let _ = CognitiveShaderBuilder::new().semiring(semiring()).build();
}

/// C4 — the ontology is the driver's: a mailbox driver takes it only from
/// `ontology()`, never from a BindSpace handed in alongside; a singleton
/// driver still inherits its BindSpace's handle.
#[test]
fn the_ontology_is_carried_by_the_driver_not_the_substrate() {
    let reg = Arc::new(OntologyRegistry::new_in_memory());
    let mut bs = BindSpace::zeros(3);
    bs.set_ontology(reg.clone());
    let bs = Arc::new(bs);

    let mailbox_with_legacy = CognitiveShaderBuilder::new()
        .bindspace(bs.clone())
        .semiring(semiring())
        .with_mailbox(0, mailbox(0, ROWS, 0))
        .build();
    assert!(mailbox_with_legacy.ontology().is_none());

    let mailbox_with_reg = CognitiveShaderBuilder::new()
        .semiring(semiring())
        .with_mailbox(0, mailbox(0, ROWS, 0))
        .ontology(reg.clone())
        .build();
    assert!(Arc::ptr_eq(mailbox_with_reg.ontology().unwrap(), &reg));

    let singleton = CognitiveShaderBuilder::new()
        .bindspace(bs)
        .semiring(semiring())
        .build();
    assert!(Arc::ptr_eq(singleton.ontology().unwrap(), &reg));
}

/// C5 — row count and footprint are the selected mailbox's.
#[test]
fn row_count_and_footprint_report_the_selected_mailbox() {
    let mut d = CognitiveShaderBuilder::new()
        .semiring(semiring())
        .with_mailbox(1, mailbox(1, 3, 0))
        .with_mailbox(2, mailbox(2, 5, 0))
        .select_mailbox(1)
        .build();
    let tables = d.byte_footprint() - d.mailbox(1).unwrap().byte_footprint();
    assert_eq!(d.row_count(), 3);
    assert!(d.select_mailbox(2));
    assert_eq!(d.row_count(), 5);
    assert_eq!(
        d.byte_footprint(),
        d.mailbox(2).unwrap().byte_footprint() + tables
    );
    assert!(
        d.mailbox(2).unwrap().byte_footprint() < BindSpace::zeros(1024).byte_footprint(),
        "reported a BindSpace-sized footprint"
    );
}

/// C6 — the cycle stays on the mailbox. A write, a `tick` and the next
/// dispatch leave an attached legacy BindSpace bit-identical, and the
/// dispatch matches a driver that has no BindSpace.
#[test]
fn tick_and_cycle_never_route_through_bindspace() {
    let legacy = Arc::new(BindSpace::zeros(ROWS));
    let edges_before: Vec<u64> = (0..ROWS).map(|r| legacy.edges.get(r)).collect();
    let mut with_legacy = CognitiveShaderBuilder::new()
        .bindspace(legacy.clone())
        .semiring(semiring())
        .planes(planes())
        .with_mailbox(0, mailbox(0, ROWS, 0))
        .build();
    let mut bare = on(mailbox(0, ROWS, 0), 0);
    for d in [&mut with_legacy, &mut bare] {
        let mb = d.mailbox_mut(0).unwrap();
        let cell = WriteCell {
            edge: Some(CausalEdge64(4)),
            ..Default::default()
        };
        assert_eq!(mb.write_row(2, mb.cycle(), &cell), WriteOutcome::Accepted);
        mb.tick();
        assert_eq!(mb.cycle(), 1);
    }
    assert_eq!(
        signature(&with_legacy.dispatch(&request())),
        signature(&bare.dispatch(&request()))
    );
    let edges_after: Vec<u64> = (0..ROWS).map(|r| legacy.edges.get(r)).collect();
    assert_eq!(edges_before, edges_after);
}
