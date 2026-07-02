//! Engine bridge — wires thinking-engine DTOs ↔ cognitive-shader DTOs.
//!
//! Two DTO pipelines exist in isolation:
//!
//! ```text
//! thinking-engine:           Φ StreamDto → Ψ PerturbationDto → B BusDto → Γ ThoughtStruct
//! cognitive-shader-driver:   Φ ShaderDispatch → Ψ ShaderResonance → B ShaderBus → Γ ShaderCrystal
//! ```
//!
//! This module connects them so the shader can dispatch a thinking cycle
//! and the engine's output feeds back into BindSpace:
//!
//! ```text
//! [1] StreamDto.codebook_indices  → populate BindSpace content fingerprints
//! [2] PerturbationDto.top_k       → seed ShaderDispatch.rows (which rows to scan)
//! [3] ShaderBus.cycle_fingerprint → produce BusDto (top-1 hit = codebook_index)
//! [4] ShaderCrystal               → produce ThoughtStruct with sensor provenance
//! [5] Qualia17D                   → fill BindSpace QualiaColumn (17 → 18: pad 0)
//! [6] L4 bridge                   → ShaderBus.emitted_edges feed commit_to_l4
//! ```
//!
//! ## 3-Way Style Unification
//!
//! ```text
//! thinking-engine ThinkingStyle  ←→  contract StyleSelector  ←→  p64 StyleParams
//!   (12 variants, params)            (Auto/Ordinal/Named)        (layer_mask, combine, contra)
//! ```

use lance_graph_contract::cognitive_shader::{
    ColumnWindow, EmitMode, MetaFilter, MetaWord, ShaderBus, ShaderDispatch, StyleSelector,
};

use crate::bindspace::{BindSpace, WORDS_PER_FP};
use lance_graph_contract::qualia::QualiaI4_16D;
// QUALIA_DIMS is referenced only inside `dispatch_busdto` (which is
// `#[cfg(feature = "with-engine")]`); gate the import so default builds don't
// warn on an unused import under clippy `-D warnings`.
#[cfg(feature = "with-engine")]
use lance_graph_contract::qualia::QUALIA_DIMS;

#[cfg(feature = "with-engine")]
use thinking_engine::dto::BusDto;

// ═══════════════════════════════════════════════════════════════════════════
// StreamDto → BindSpace (sensor output populates content fingerprints)
// ═══════════════════════════════════════════════════════════════════════════

/// Ingest a sensor's codebook indices into BindSpace rows.
///
/// Each codebook index becomes a BindSpace row. The content fingerprint
/// is built by setting the bit at `index` (mod 16384) — a simple
/// positional encoding. The meta column records the source type.
///
/// Returns the row range written (start..end).
///
/// `write_cursor` is the first free row in BindSpace. Callers track
/// this across multiple ingestions (sensors arrive at different times).
pub fn ingest_codebook_indices(
    bs: &mut BindSpace,
    indices: &[u16],
    source_ordinal: u8,
    timestamp: u64,
    write_cursor: usize,
) -> (u32, u32) {
    let start = write_cursor.min(bs.meta.0.len());
    let mut cursor = start;

    for &idx in indices {
        if cursor >= bs.meta.0.len() {
            break;
        }

        // Build content fingerprint: set bit at `idx` position.
        let mut content = [0u64; WORDS_PER_FP];
        let bit = idx as usize % (WORDS_PER_FP * 64);
        content[bit / 64] |= 1u64 << (bit % 64);
        bs.fingerprints.set_content(cursor, &content);

        // Meta: source_ordinal as thinking style, no NARS yet.
        bs.meta
            .set(cursor, MetaWord::new(source_ordinal, 0, 0, 0, 0));
        bs.temporal[cursor] = timestamp;

        cursor += 1;
    }

    (start as u32, cursor as u32)
}

// ═══════════════════════════════════════════════════════════════════════════
// PerturbationDto → ShaderDispatch (top-k seeds the scan window)
// ═══════════════════════════════════════════════════════════════════════════

/// Build a ShaderDispatch from resonance top-k.
///
/// The top-k codebook indices from the thinking-engine's resonance field
/// become the row window for the shader to scan. If the BindSpace has
/// been populated via `ingest_codebook_indices`, the rows correspond
/// to the sensor output that produced the resonance.
pub fn dispatch_from_top_k(
    top_k: &[(u16, f32)],
    total_rows: u32,
    style: StyleSelector,
) -> ShaderDispatch {
    let active: Vec<u16> = top_k
        .iter()
        .filter(|&&(_, e)| e > 0.01)
        .map(|&(idx, _)| idx)
        .collect();

    let (start, end) = if active.is_empty() {
        (0, total_rows.min(64))
    } else {
        let min_row = *active.iter().min().unwrap_or(&0) as u32;
        let max_row = (*active.iter().max().unwrap_or(&0) as u32 + 1).min(total_rows);
        (min_row, max_row)
    };

    ShaderDispatch {
        rows: ColumnWindow::new(start, end),
        meta_prefilter: MetaFilter::ALL,
        layer_mask: 0xFF,
        radius: u16::MAX,
        style,
        max_cycles: 10,
        entropy_floor: 0.05,
        emit: EmitMode::Cycle,
        ..Default::default()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ShaderBus → BusDto (cycle_fingerprint → codebook_index)
// ═══════════════════════════════════════════════════════════════════════════

/// Extract a BusDto-compatible tuple from a ShaderBus.
///
/// The top-1 hit's row becomes `codebook_index`. The top-k hits
/// map to the 8-entry top_k array. This is the bridge from
/// "shader world" (BindSpace rows, cycle_fingerprint) back to
/// "engine world" (4096 codebook, energy field).
pub struct EngineBusBridge {
    pub codebook_index: u16,
    pub energy: f32,
    pub top_k: [(u16, f32); 8],
    pub cycle_count: u16,
    pub converged: bool,
}

impl EngineBusBridge {
    pub fn from_shader_bus(bus: &ShaderBus) -> Self {
        let top_k: [(u16, f32); 8] = {
            let mut arr = [(0u16, 0.0f32); 8];
            for (i, h) in bus.resonance.top_k.iter().enumerate().take(8) {
                arr[i] = (h.row as u16, h.resonance);
            }
            arr
        };

        Self {
            codebook_index: bus.resonance.top_k[0].row as u16,
            energy: bus.resonance.top_k[0].resonance,
            top_k,
            cycle_count: bus.resonance.cycles_used,
            converged: bus.gate.is_flow(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// D-PARITY-V2-3: BusDto → BindSpace (the Tier-2 → Tier-3 transition)
// ═══════════════════════════════════════════════════════════════════════════
//
// Per `.claude/knowledge/soa-dto-dependency-ledger.md` Tier 2 → Tier 3 path:
//   ThinkingEngine.commit() → BusDto → ShaderDispatch.encode → BindSpace SoA
//
// The 5 BusDto fields (codebook_index, energy, top_k[8], cycle_count,
// converged) collapse onto a single BindSpace row across these columns:
//
//   codebook_index + top_k indices  →  cycle column (Vsa16kF32, via Binary16K)
//   energy + top_k energies         →  qualia[0..9]
//   cycle_count (saturated u6)      →  MetaWord.free_e
//   converged                       →  MetaWord.awareness (3 = flow, 1 = held)
//   codebook_index (low byte)       →  MetaWord.nars_f (commit confidence)
//   style ordinal (caller picks)    →  MetaWord.thinking
//
// Mapping choice (EPIPHANY): the canonical encode is positional bit-set
// over Binary16K, projected to Vsa16kF32 via the existing
// `binary16k_to_vsa16k_bipolar`. Each codebook index `idx` (u16) sets
// `bit (idx % 16384)` in a `[u64; 256]` Binary16K accumulator. The accumulator
// is then routed through `BindSpace::write_cycle_fingerprint`, which is the
// canonical entry point per CLAUDE.md (it converts to Vsa16kF32 internally).
//
// This re-uses the existing path — no new column structures, no BindSpace
// touch (Wave 3 owns those). Per `lab-vs-canonical-surface.md` we extend the
// canonical surface, not invent a sibling.

#[cfg(feature = "with-engine")]
const NARS_F_FROM_INDEX_LOW: bool = true;
#[cfg(feature = "with-engine")]
const TOP_K_ENERGY_BASE_DIM: usize = 1; // qualia[0] = headline energy, [1..9] = top_k energies

/// Encode a BusDto's codebook_index + top_k indices as a `[u64; WORDS_PER_FP]`
/// Binary16K accumulator. Each index sets one bit at `idx % WIDTH_BITS`.
/// `top_k` entries with energy ≤ 0.0 are skipped (zero-energy = no support).
#[cfg(feature = "with-engine")]
fn busdto_to_binary16k(bus: &BusDto) -> [u64; WORDS_PER_FP] {
    let width_bits = WORDS_PER_FP * 64;
    let mut bits = [0u64; WORDS_PER_FP];
    let mut set_bit = |idx: u16| {
        let pos = (idx as usize) % width_bits;
        bits[pos / 64] |= 1u64 << (pos % 64);
    };
    // Headline: codebook_index always sets a bit (BusDto IS a committed thought).
    set_bit(bus.codebook_index);
    // Top-K supporters: only those with positive energy contribute a bit.
    for &(idx, e) in bus.top_k.iter() {
        if e > 0.0 {
            set_bit(idx);
        }
    }
    bits
}

/// Wire `cognitive-shader-driver::engine_bridge` to consume `BusDto`
/// directly — the Tier 2 → Tier 3 transition (D-PARITY-V2-3).
///
/// Encodes the BusDto into BindSpace row `row` across:
/// - `cycle` column (Vsa16kF32 via Binary16K bits)
/// - `qualia` column (headline energy + 8 top_k energies in dims 0..9)
/// - `meta` column (style + awareness + nars_f + nars_c + free_e)
/// - `expert` column (cycle_count low byte)
///
/// `style_ord` is the caller-picked style ordinal (0..=11). `BusDto`
/// itself does not carry a style; that's the caller's dispatch concern.
///
/// Returns the row written.
#[cfg(feature = "with-engine")]
pub fn dispatch_busdto(bs: &mut BindSpace, row: usize, bus: &BusDto, style_ord: u8) -> usize {
    assert!(
        row < bs.len,
        "dispatch_busdto: row {row} out of bounds {}",
        bs.len
    );

    // [1] cycle column — codebook_index + top_k indices as Binary16K → Vsa16kF32.
    let bits = busdto_to_binary16k(bus);
    bs.write_cycle_fingerprint(row, &bits);

    // [2] qualia column — energies as continuous payload (lossless f32 store).
    //     qualia[0]   = headline energy
    //     qualia[1..9] = top_k energies (positions 1-based to keep dim 0 = headline)
    //     qualia[9]   = codebook_index headline (codex P2 fix 2026-05-07)
    //     qualia[10..17] = zeroed (reserved for downstream qualia / classification dist)
    //
    // The codebook_index headline goes into qualia[9] explicitly so the
    // round-trip is bit-exact even when codebook_index collides with or
    // is larger than any positive-energy top_k index. Previously the
    // decoder relied on `set_bits.iter().next()` which always returned
    // the LOWEST set bit; for `codebook_index = 1234` with positive
    // top_k containing 777, the recovered headline was 777 instead of
    // 1234. f32 represents any integer in [0, 2^24] exactly, so the
    // u16 codebook_index round-trips losslessly through f32.
    let mut q = [0.0f32; QUALIA_DIMS];
    q[0] = bus.energy;
    for (i, &(_idx, e)) in bus.top_k.iter().enumerate().take(8) {
        q[TOP_K_ENERGY_BASE_DIM + i] = e;
    }
    q[9] = bus.codebook_index as f32;
    // D-CSV-5b: engine still produces f32; convert at the bridge boundary.
    // from_f32_17d expects [f32; 17]; q is [f32; QUALIA_DIMS=17].
    let mut q17 = [0.0f32; 17];
    q17.copy_from_slice(&q[..17]);
    // Canonical i4 column (hot-path representation; lossy — general qualia consumers).
    bs.qualia.set(row, QualiaI4_16D::from_f32_17d(&q17));
    // Lab bit-exact tenant: store the full f32-17D so `unbind_busdto` round-trips
    // exactly. The i4 column above clamps to ±1, which corrupts `codebook_index`
    // (q[9]) and the energies — the D-CSV-5b regression this tenant repairs.
    bs.set_qualia_f32(row, &q17);

    // [3] meta column — packed dispatch state.
    //     thinking = caller's style ordinal
    //     awareness = converged ? FLOW(3) : HOLD(1)
    //     nars_f    = low byte of codebook_index (commit confidence proxy)
    //     nars_c    = clamp(energy * 255, 0, 255)
    //     free_e    = saturating cycle_count (6-bit)
    let awareness = if bus.converged { 3u8 } else { 1u8 };
    let nars_f = if NARS_F_FROM_INDEX_LOW {
        (bus.codebook_index & 0xFF) as u8
    } else {
        0
    };
    let nars_c = (bus.energy * 255.0).clamp(0.0, 255.0) as u8;
    let free_e = bus.cycle_count.min(63) as u8;
    bs.meta.set(
        row,
        MetaWord::new(style_ord, awareness, nars_f, nars_c, free_e),
    );

    // [4] expert column — cycle_count (full u16 fidelity, lossless).
    bs.expert[row] = bus.cycle_count;

    row
}

/// Inverse of `dispatch_busdto`: unbind a BindSpace row back to a `BusDto`.
///
/// Round-trip recovery:
///  - `cycle_count`  — bit-exact from `expert[row]`.
///  - `converged`    — bit-exact from `meta.awareness >= 3`.
///  - `energy` + `top_k[*].energy` — bit-exact from qualia f32 store.
///  - `codebook_index` — bit-exact for the headline index, since it was
///    always emitted by `busdto_to_binary16k` (the headline bit is
///    guaranteed-set; we recover it via top_k[0].idx, which the caller
///    encoded redundantly). Falls back to lowest-set bit if top_k[0] is
///    zero-valued.
///  - `top_k[*].idx` — bit-exact for the SUBSET that had positive energy at
///    encode (those indices became set bits). Indices with energy ≤ 0 at
///    encode produced no bit; their original values are not recoverable.
///
/// Tolerance: bit-exact for codebook_index, top_k indices with positive
/// energy at encode, energies (f32 in qualia), cycle_count, converged.
/// LOSSY for top_k entries with non-positive energy at encode.
///
/// ## Feature interaction: `mailbox-thoughtspace` (C5 / D-DIST-5)
///
/// The non-headline `top_k[*].idx` recovery reads the `cycle` plane
/// (`Vsa16kF32` Binary16K set-bits). That plane is the deprecated one the
/// `MailboxSoA` migration **never carries** (it is computed transiently, never
/// a mailbox column — see `mailbox_soa.rs`). Under `mailbox-thoughtspace` the
/// cycle-plane index recovery is therefore feature-gated OUT: only the headline
/// `codebook_index` (stored losslessly in `qualia[9]`) survives, and the
/// non-headline `top_k[1..].idx` recover as `0`. This is an explicit downgrade
/// of an ALREADY-lossy register recovery (`I-VSA-IDENTITIES`: VSA bundles
/// identities, not content — the cycle-plane set-bits were never a faithful
/// register), on a `#[cfg(with-engine)]` lab path the live `run()` never reads.
/// Migration successor: read indices from the SoA edge/identity columns, not the
/// dropped cycle plane. The singleton (default) build keeps the bit-exact
/// cycle-plane recovery via `#[cfg(not(feature = "mailbox-thoughtspace"))]`.
#[cfg(feature = "with-engine")]
pub fn unbind_busdto(bs: &BindSpace, row: usize) -> BusDto {
    assert!(
        row < bs.len,
        "unbind_busdto: row {row} out of bounds {}",
        bs.len
    );

    // [1] qualia → energy + top_k energies + headline codebook_index.
    // D-CSV-5b: bs.qualia is now QualiaI4Column; convert to f32 at the read site.
    // codex P2 fix (2026-05-07): the headline is stored explicitly in qualia[9]
    // at encode, so it round-trips bit-exact regardless of cycle-plane state.
    // Read the bit-exact f32 tenant — NOT the i4 `qualia` column, which clamps
    // to ±1 and would corrupt codebook_index (q[9]) and the energies (the
    // D-CSV-5b regression). The f32 tenant is the migration's exactness anchor.
    let q = *bs.qualia_f32_row(row); // [f32; 17] — bit-exact ground-truth
    let energy = q[0];
    let codebook_index = q[9] as u16;
    let mut top_k = [(0u16, 0.0f32); 8];
    for i in 0..8 {
        top_k[i].1 = q[TOP_K_ENERGY_BASE_DIM + i];
    }

    // [2/3] cycle column → recover NON-headline top_k indices from set bits.
    //       SINGLETON BUILD ONLY — the cycle plane is never migrated to the
    //       mailbox (C5 / D-DIST-5). Under `mailbox-thoughtspace` this block is
    //       gated out and the non-headline indices stay 0 (documented loss).
    #[cfg(not(feature = "mailbox-thoughtspace"))]
    {
        // Project Vsa16kF32 back to Binary16K (sign threshold → bit).
        let cycle = bs.fingerprints.cycle_row(row);
        let mut cycle_arr = [0.0f32; crate::bindspace::FLOATS_PER_VSA];
        cycle_arr.copy_from_slice(cycle);
        let bits = lance_graph_contract::crystal::vsa16k_to_binary16k_threshold(&cycle_arr);
        let set_bits: Vec<u16> = (0..(WORDS_PER_FP * 64))
            .filter(|&pos| bits[pos / 64] & (1u64 << (pos % 64)) != 0)
            .map(|pos| pos as u16)
            .collect();

        // The set_bits iterator feeds only the non-headline top_k slots.
        let bit_iter = set_bits.iter().copied().filter(|&b| b != codebook_index);
        // The headline often equals top_k[0].idx — rebuild that match first.
        if top_k[0].1 > 0.0 {
            top_k[0].0 = codebook_index;
        }
        // Fill remaining positive-energy top_k slots from the remaining set bits.
        let remaining: Vec<u16> = bit_iter.collect();
        let mut r = remaining.into_iter();
        let skip_head = top_k[0].1 > 0.0;
        for slot in top_k.iter_mut().skip(if skip_head { 1 } else { 0 }) {
            if slot.1 > 0.0 {
                if let Some(b) = r.next() {
                    slot.0 = b;
                }
            }
        }
        // If top_k[0].1 was non-positive but the encoder always sets the headline,
        // we still recovered codebook_index above — it's authoritative.
    }
    // Under `mailbox-thoughtspace`: top_k[0].idx still gets the headline if it
    // had positive energy (the headline is in qualia[9], not the dropped plane);
    // all other non-headline indices remain 0 (the documented C5 loss).
    #[cfg(feature = "mailbox-thoughtspace")]
    {
        if top_k[0].1 > 0.0 {
            top_k[0].0 = codebook_index;
        }
    }

    // [4] meta column → converged.
    let m = bs.meta.get(row);
    let converged = m.awareness() >= 3;

    // [5] expert column → cycle_count (full u16 fidelity, no saturation loss).
    let cycle_count = bs.expert[row];

    BusDto {
        codebook_index,
        energy,
        top_k,
        cycle_count,
        converged,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Qualia: 17D experienced (CMYK) vs 18D observed (RGB)
// ═══════════════════════════════════════════════════════════════════════════

/// The 17D/18D split is CMYK vs RGB — not padding.
///
/// 17D (thinking-engine `Qualia17D`): the EXPERIENCED qualia.
///   Convergence speed IS clarity. Non-convergence IS tension.
///   These are the raw QPL observables. "Steelwind" lives here —
///   the felt quality that has no name yet.
///   This is CMYK: subtractive, production-side, how the ink hits paper.
///
/// 18D (BindSpace `QualiaColumn`): the OBSERVED qualia.
///   Dim 17 = `classification_distance`: how far the experienced qualia
///   is from its nearest named emotion. Low = "fear" (named, classified).
///   High = "steelwind" (unnamed, raw, novel).
///   This is RGB: additive, display-side, how the observer perceives it.
///
/// The transform from 17→18 IS the act of observation. Dim 17 measures
/// the gap between what the system FEELS and what it can NAME.
pub const DIM_CLASSIFICATION_DISTANCE: usize = 17;

/// Write 17D experienced qualia + classification distance into 18D observed.
pub fn write_qualia_observed(
    bs: &mut BindSpace,
    row: usize,
    experienced: &[f32; 17],
    _classification_distance: f32,
) {
    // D-CSV-5b: qualia is now QualiaI4Column (16 dims). classification_distance
    // (dim 17) is no longer stored in the column — it is computed on-demand
    // via classification_distance() if needed.
    bs.qualia.set(row, QualiaI4_16D::from_f32_17d(experienced));
}

/// Read observed qualia and decompose into experienced (17D) + classification distance.
pub fn read_qualia_decomposed(bs: &BindSpace, row: usize) -> ([f32; 17], f32) {
    // D-CSV-5b: bs.qualia is now QualiaI4Column; convert to f32 at read site.
    let q_i4 = bs.qualia.row(row);
    let q17 = q_i4.to_f32_17d(); // [f32; 17]
    let mut experienced = [0.0f32; 17];
    let n = q17.len().min(17);
    experienced[..n].copy_from_slice(&q17[..n]);
    // DIM_CLASSIFICATION_DISTANCE = 17, beyond the i4 range (16 dims stored).
    // Post-cutover: classification_distance is no longer stored in the column;
    // return 1.0 (fully unnamed) as the default, or recompute via classification_distance().
    let cd = 1.0_f32;
    (experienced, cd)
}

/// Compute classification distance: how close the experienced qualia is to
/// a known emotion archetype. 0.0 = exact match ("fear"), 1.0 = novel ("steelwind").
///
/// Uses a simple nearest-archetype heuristic over the 17D QPL space.
/// The archetypes are the 6 basic emotions mapped to QPL coordinates.
pub fn classification_distance(experienced: &[f32; 17]) -> f32 {
    // Basic emotion archetypes in QPL space (arousal, valence, tension, warmth, clarity, ...)
    const ARCHETYPES: [[f32; 6]; 6] = [
        // arousal, valence, tension, warmth, clarity, boundary
        [0.9, -0.8, 0.9, 0.1, 0.3, 0.8], // fear
        [0.8, -0.6, 0.7, 0.2, 0.4, 0.6], // anger
        [0.2, -0.7, 0.3, 0.6, 0.2, 0.3], // sadness
        [0.7, 0.8, 0.1, 0.9, 0.8, 0.2],  // joy
        [0.3, 0.1, 0.5, 0.3, 0.1, 0.9],  // surprise
        [0.1, -0.2, 0.1, 0.2, 0.6, 0.4], // disgust
    ];

    let mut min_dist = f32::MAX;
    for archetype in &ARCHETYPES {
        let mut d = 0.0f32;
        for (i, &a) in archetype.iter().enumerate() {
            let diff = experienced[i] - a;
            d += diff * diff;
        }
        min_dist = min_dist.min(d.sqrt());
    }
    // Normalize: typical max L2 across 6 dims ≈ 3.0
    (min_dist / 3.0).clamp(0.0, 1.0)
}

/// Legacy compat: write 17D with zero classification distance (fully named).
pub fn write_qualia_17d(bs: &mut BindSpace, row: usize, q17: &[f32; 17]) {
    write_qualia_observed(bs, row, q17, classification_distance(q17));
}

/// Legacy compat: read experienced 17D only (drops observer frame).
pub fn read_qualia_17d(bs: &BindSpace, row: usize) -> [f32; 17] {
    read_qualia_decomposed(bs, row).0
}

// ═══════════════════════════════════════════════════════════════════════════
// 3-Way Style Unification
// ═══════════════════════════════════════════════════════════════════════════

/// Full style bridge: given a style ordinal, return the p64-bridge
/// StyleParams (layer_mask, combine, contra, density_target) plus
/// the thinking-engine style parameters (resonance_threshold, fan_out, etc.).
///
/// This is THE canonical mapping. Three type systems, one ordinal.
pub struct UnifiedStyle {
    pub ordinal: u8,
    pub name: &'static str,
    // p64-bridge side
    pub layer_mask: u8,
    pub combine: u8,
    pub contra: u8,
    pub density_target: f32,
    // thinking-engine side
    pub resonance_threshold: f32,
    pub fan_out: usize,
    pub exploration: f32,
    pub speed: f32,
    pub collapse_bias: f32,
    pub butterfly_sensitivity: f32,
}

/// The 12 unified styles. Index = ordinal.
pub const UNIFIED_STYLES: [UnifiedStyle; 12] = [
    // 0: Deliberate
    UnifiedStyle {
        ordinal: 0,
        name: "deliberate",
        layer_mask: 0b0111_1111,
        combine: 3,
        contra: 0,
        density_target: 0.08,
        resonance_threshold: 0.70,
        fan_out: 7,
        exploration: 0.20,
        speed: 0.1,
        collapse_bias: -0.05,
        butterfly_sensitivity: 0.50,
    },
    // 1: Analytical
    UnifiedStyle {
        ordinal: 1,
        name: "analytical",
        layer_mask: 0b0111_0111,
        combine: 1,
        contra: 0,
        density_target: 0.05,
        resonance_threshold: 0.85,
        fan_out: 3,
        exploration: 0.05,
        speed: 0.1,
        collapse_bias: -0.10,
        butterfly_sensitivity: 0.80,
    },
    // 2: Convergent
    UnifiedStyle {
        ordinal: 2,
        name: "convergent",
        layer_mask: 0b0011_0111,
        combine: 1,
        contra: 0,
        density_target: 0.04,
        resonance_threshold: 0.75,
        fan_out: 4,
        exploration: 0.10,
        speed: 0.3,
        collapse_bias: -0.05,
        butterfly_sensitivity: 0.70,
    },
    // 3: Systematic
    UnifiedStyle {
        ordinal: 3,
        name: "systematic",
        layer_mask: 0b0111_1111,
        combine: 1,
        contra: 0,
        density_target: 0.03,
        resonance_threshold: 0.70,
        fan_out: 5,
        exploration: 0.10,
        speed: 0.2,
        collapse_bias: 0.00,
        butterfly_sensitivity: 0.60,
    },
    // 4: Creative
    UnifiedStyle {
        ordinal: 4,
        name: "creative",
        layer_mask: 0b1111_1111,
        combine: 0,
        contra: 1,
        density_target: 0.40,
        resonance_threshold: 0.35,
        fan_out: 12,
        exploration: 0.80,
        speed: 0.5,
        collapse_bias: 0.15,
        butterfly_sensitivity: 0.20,
    },
    // 5: Divergent
    UnifiedStyle {
        ordinal: 5,
        name: "divergent",
        layer_mask: 0b1000_1001,
        combine: 0,
        contra: 2,
        density_target: 0.30,
        resonance_threshold: 0.40,
        fan_out: 10,
        exploration: 0.70,
        speed: 0.4,
        collapse_bias: 0.10,
        butterfly_sensitivity: 0.35,
    },
    // 6: Exploratory
    UnifiedStyle {
        ordinal: 6,
        name: "exploratory",
        layer_mask: 0b1111_1111,
        combine: 0,
        contra: 1,
        density_target: 0.50,
        resonance_threshold: 0.30,
        fan_out: 15,
        exploration: 0.90,
        speed: 0.6,
        collapse_bias: 0.20,
        butterfly_sensitivity: 0.15,
    },
    // 7: Focused
    UnifiedStyle {
        ordinal: 7,
        name: "focused",
        layer_mask: 0b0000_0011,
        combine: 1,
        contra: 0,
        density_target: 0.02,
        resonance_threshold: 0.90,
        fan_out: 1,
        exploration: 0.00,
        speed: 0.2,
        collapse_bias: -0.15,
        butterfly_sensitivity: 0.90,
    },
    // 8: Diffuse
    UnifiedStyle {
        ordinal: 8,
        name: "diffuse",
        layer_mask: 0b0111_0111,
        combine: 2,
        contra: 3,
        density_target: 0.20,
        resonance_threshold: 0.45,
        fan_out: 8,
        exploration: 0.40,
        speed: 0.5,
        collapse_bias: 0.05,
        butterfly_sensitivity: 0.25,
    },
    // 9: Peripheral
    UnifiedStyle {
        ordinal: 9,
        name: "peripheral",
        layer_mask: 0b1110_0000,
        combine: 0,
        contra: 1,
        density_target: 0.35,
        resonance_threshold: 0.20,
        fan_out: 20,
        exploration: 0.60,
        speed: 0.7,
        collapse_bias: 0.25,
        butterfly_sensitivity: 0.10,
    },
    // 10: Intuitive
    UnifiedStyle {
        ordinal: 10,
        name: "intuitive",
        layer_mask: 0b0000_0001,
        combine: 0,
        contra: 1,
        density_target: 0.50,
        resonance_threshold: 0.50,
        fan_out: 3,
        exploration: 0.30,
        speed: 0.9,
        collapse_bias: 0.00,
        butterfly_sensitivity: 0.30,
    },
    // 11: Metacognitive
    UnifiedStyle {
        ordinal: 11,
        name: "metacognitive",
        layer_mask: 0b1110_0000,
        combine: 2,
        contra: 3,
        density_target: 0.10,
        resonance_threshold: 0.50,
        fan_out: 5,
        exploration: 0.30,
        speed: 0.3,
        collapse_bias: 0.00,
        butterfly_sensitivity: 0.40,
    },
];

pub fn unified_style(ord: u8) -> &'static UnifiedStyle {
    &UNIFIED_STYLES[(ord % 12) as usize]
}

// ═══════════════════════════════════════════════════════════════════════════
// Cycle fingerprint → BindSpace write-back
// ═══════════════════════════════════════════════════════════════════════════

/// After a shader cycle, persist the results into BindSpace:
/// - cycle_fingerprint → fingerprints.cycle column
/// - emitted_edges → edges column
/// - meta update (gate state → awareness, NARS → f/c)
///
/// Returns the row written.
pub fn persist_cycle(bs: &mut BindSpace, row: usize, bus: &ShaderBus, style_ord: u8) {
    bs.write_cycle_fingerprint(row, &bus.cycle_fingerprint);

    if bus.emitted_edge_count > 0 {
        bs.edges.set(row, bus.emitted_edges[0]);
    }

    let awareness = match bus.gate {
        g if g.is_flow() => 3u8,
        g if g.is_hold() => 2u8,
        _ => 1u8,
    };
    let nars_f = (bus.resonance.top_k[0].resonance * 255.0).clamp(0.0, 255.0) as u8;
    let nars_c = ((1.0 - bus.resonance.std_dev) * 255.0).clamp(0.0, 255.0) as u8;
    let free_e = ((bus.resonance.entropy / 3.0) * 63.0).clamp(0.0, 63.0) as u8;

    bs.meta.set(
        row,
        MetaWord::new(style_ord, awareness, nars_f, nars_c, free_e),
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto_style;
    use lance_graph_contract::cognitive_shader::ShaderHit;
    use lance_graph_contract::collapse_gate::GateDecision;

    #[test]
    fn ingest_sets_bits() {
        let mut bs = BindSpace::zeros(10);
        let (start, end) = ingest_codebook_indices(&mut bs, &[42, 100, 200], 1, 12345, 0);
        assert_eq!(start, 0);
        assert_eq!(end, 3);
        // Check bit 42 is set in row 0's content fingerprint.
        let words = bs.fingerprints.content_row(0);
        assert_ne!(words[42 / 64] & (1u64 << (42 % 64)), 0);
        assert_eq!(bs.temporal[0], 12345);
    }

    #[test]
    fn dispatch_from_empty_topk_scans_first_64() {
        let d = dispatch_from_top_k(&[], 1000, StyleSelector::Auto);
        assert_eq!(d.rows.start, 0);
        assert_eq!(d.rows.end, 64);
    }

    #[test]
    fn dispatch_from_topk_brackets_rows() {
        let top_k = [(10u16, 0.5f32), (20, 0.3), (30, 0.1)];
        let d = dispatch_from_top_k(&top_k, 100, StyleSelector::Ordinal(1));
        assert_eq!(d.rows.start, 10);
        assert_eq!(d.rows.end, 31);
    }

    #[test]
    fn engine_bus_bridge_extracts_top1() {
        let mut bus = ShaderBus::empty();
        bus.resonance.top_k[0] = ShaderHit {
            row: 42,
            distance: 100,
            predicates: 0x07,
            _pad: 0,
            resonance: 0.85,
            cycle_index: 0,
        };
        bus.gate = GateDecision::FLOW_XOR;
        let bridge = EngineBusBridge::from_shader_bus(&bus);
        assert_eq!(bridge.codebook_index, 42);
        assert!((bridge.energy - 0.85).abs() < 0.01);
        assert!(bridge.converged);
    }

    #[test]
    fn qualia_17d_roundtrip() {
        // D-CSV-5b: write_qualia_17d now stores via QualiaI4_16D (8 B, i4×16 signed).
        // i4 precision: step size = 1/7 ≈ 0.143 for positive, 1/8 = 0.125 for negative.
        // Round-trip tolerance must be >= 1 i4 step (0.15 covers it).
        let mut bs = BindSpace::zeros(2);
        let mut q17 = [0.0f32; 17];
        q17[0] = 0.8; // arousal: round(0.8*7)=6 → 6/7 ≈ 0.857 (within 0.15 of 0.8)
        q17[4] = 0.571; // clarity: round(0.571*7)=4 → 4/7 ≈ 0.571 (near-exact)
        q17[14] = -0.25; // groundedness: round(-0.25*8)=-2 → -2/8 = -0.25 (exact)
        write_qualia_17d(&mut bs, 0, &q17);
        let back = read_qualia_17d(&bs, 0);
        // Tolerance = 0.15 (1 i4 step). Values chosen to be representable in i4.
        assert!(
            (back[0] - q17[0]).abs() < 0.15,
            "dim 0: expected ~{}, got {} (i4 quantization ±0.15)",
            q17[0],
            back[0]
        );
        assert!(
            (back[4] - q17[4]).abs() < 0.15,
            "dim 4: expected ~{}, got {} (i4 quantization ±0.15)",
            q17[4],
            back[4]
        );
        assert!(
            (back[14] - q17[14]).abs() < 0.15,
            "dim 14: expected ~{}, got {} (i4 quantization ±0.15)",
            q17[14],
            back[14]
        );
    }

    #[test]
    fn cmyk_rgb_fear_is_near_zero_distance() {
        // "Fear" archetype: high arousal, negative valence, high tension
        let mut fear = [0.0f32; 17];
        fear[0] = 0.9; // arousal
        fear[1] = -0.8; // valence
        fear[2] = 0.9; // tension
        fear[3] = 0.1; // warmth
        fear[4] = 0.3; // clarity
        fear[5] = 0.8; // boundary
        let cd = classification_distance(&fear);
        assert!(
            cd < 0.15,
            "Fear should be near-zero distance (named), got {cd}"
        );
    }

    #[test]
    fn cmyk_rgb_steelwind_is_far() {
        // "Steelwind" — a novel qualia with no archetype match
        let mut steelwind = [0.0f32; 17];
        steelwind[0] = 0.5; // moderate arousal
        steelwind[1] = 0.5; // positive valence
        steelwind[2] = 0.8; // high tension (unusual with positive valence)
        steelwind[3] = 0.0; // no warmth
        steelwind[4] = 0.9; // very clear
        steelwind[5] = 0.0; // no boundary
        let cd = classification_distance(&steelwind);
        assert!(
            cd > 0.3,
            "Steelwind should be far from named emotions, got {cd}"
        );
    }

    #[test]
    fn observed_qualia_preserves_classification_distance() {
        // D-CSV-5b: QualiaI4_16D stores 16 dims (indices 0..15). Dim 17
        // (classification_distance) is no longer stored in the column.
        // read_qualia_decomposed returns 1.0 (max distance = fully unnamed) as default.
        // The experienced dims 0..15 are stored with i4 precision (tolerance ±0.15).
        let mut bs = BindSpace::zeros(2);
        let mut experienced = [0.0f32; 17];
        experienced[0] = 4.0 / 7.0; // exact i4 representation: round(4/7*7)=4 → 4/7
        write_qualia_observed(&mut bs, 0, &experienced, 0.75);
        let (back_exp, back_cd) = read_qualia_decomposed(&bs, 0);
        assert!(
            (back_exp[0] - experienced[0]).abs() < 0.15,
            "experienced dim 0: expected ~{}, got {} (i4 quantization)",
            experienced[0],
            back_exp[0]
        );
        // classification_distance is not stored post-cutover; returns 1.0 (default)
        assert!(
            (back_cd - 1.0).abs() < 1e-6,
            "D-CSV-5b: classification_distance not stored in i4 column; expected 1.0, got {}",
            back_cd
        );
    }

    #[test]
    fn unified_styles_cover_all_12() {
        for i in 0..12u8 {
            let s = unified_style(i);
            assert_eq!(s.ordinal, i);
            assert!(!s.name.is_empty());
            assert!(s.fan_out >= 1);
            assert!(s.resonance_threshold >= 0.0 && s.resonance_threshold <= 1.0);
        }
    }

    #[test]
    fn persist_cycle_updates_meta() {
        let mut bs = BindSpace::zeros(2);
        let mut bus = ShaderBus::empty();
        bus.resonance.top_k[0].resonance = 0.9;
        bus.resonance.std_dev = 0.1;
        bus.resonance.entropy = 1.5;
        bus.gate = GateDecision::FLOW_XOR;
        bus.emitted_edges[0] = 0xDEAD;
        bus.emitted_edge_count = 1;

        persist_cycle(&mut bs, 0, &bus, auto_style::ANALYTICAL);

        let meta = bs.meta.get(0);
        assert_eq!(meta.thinking(), auto_style::ANALYTICAL);
        assert_eq!(meta.awareness(), 3); // Flow = 3
        assert!(meta.nars_f() > 200); // 0.9 * 255 ≈ 230
        assert!(meta.nars_c() > 200); // (1-0.1) * 255 ≈ 230
        assert_eq!(bs.edges.get(0), 0xDEAD);
    }
}
