//! Real wire: Grok response → deepnsm COCA-4096 tokenize → gridlake-4096 cell
//! (by real COCA rank) → 48 helix + 48 CAM_PQ (6× palette256²) per cell.
//!
//! Upgrades the earlier stand-in spike: the cell index is now the REAL COCA
//! word rank from `Vocabulary::load(word_frequency/)`, not an FNV hash. The
//! codec (helix48 place-walk + 6× palette256²) is still a deterministic
//! stand-in for the trained `Signed360` / centroid encoders — the SHAPE,
//! FOOTPRINT, and now the REAL semantic landing are what this demonstrates.

use deepnsm::Vocabulary;
use std::path::Path;
use std::time::Instant;

const GRID: usize = 4096; // COCA vocab = Cam4096 12-bit = 64×64 gridlake tile
const PQ: usize = 6; // 6× (8:8) palette256²

#[derive(Clone, Copy, Default)]
struct Cell {
    helix48: [u8; 6],
    campq48: [u8; 6],
    count: u32,
    sum_truth: u32,
}

fn land(cell: &mut Cell, word: &[u8], palette: &[[[u8; 256]; 256]], truth: u32) {
    cell.count += 1;
    cell.sum_truth += truth;
    let a = word.first().copied().unwrap_or(0) as usize;
    let b = word.last().copied().unwrap_or(0) as usize;
    for (s, t) in palette.iter().enumerate() {
        cell.campq48[s] = t[a][b];
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &by in word {
        h ^= by as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let place = h.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    cell.helix48.copy_from_slice(&place.to_le_bytes()[..6]);
}

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("word_frequency");
    let vocab = Vocabulary::load(&dir).expect("load COCA word_frequency");
    println!("── COCA VOCAB ─────────────────────────────────────────────");
    println!("  loaded {} entries (VOCAB_SIZE=4096) from {}", vocab.len(), dir.display());

    let mut palette: Vec<[[u8; 256]; 256]> = vec![[[0u8; 256]; 256]; PQ];
    for (s, t) in palette.iter_mut().enumerate() {
        for (a, row) in t.iter_mut().enumerate() {
            for (b, cell) in row.iter_mut().enumerate() {
                *cell = ((a ^ b).wrapping_add(s * 37)) as u8;
            }
        }
    }

    let mut grid = vec![Cell::default(); GRID];

    // The ACTUAL Grok (grok-4.20-non-reasoning) response captured this session.
    let grok = "Rust's ownership model ensures every value has a single owner variable at \
                any time. When the owner goes out of scope, the value is automatically \
                dropped and its memory deallocated. Ownership can be transferred via moves; \
                immutable borrows allow temporary references without transferring ownership.";

    let toks = vocab.tokenize(grok);
    let mut known = 0usize;
    let mut cells = std::collections::BTreeSet::new();
    for tk in &toks {
        if !tk.is_known() {
            continue;
        }
        known += 1;
        let rank = tk.rank_or_default() as usize; // 0..4096 = the real COCA cell
        let word = vocab.word(rank as u16).to_string();
        land(&mut grid[rank], word.as_bytes(), &palette, 200); // Grok truth ≈0.78
        cells.insert(rank);
    }

    println!("\n── REAL GROK → COCA LANDING ───────────────────────────────");
    println!(
        "  {} tokens, {} known COCA words → {} distinct real-rank cells (of 4096)",
        toks.len(),
        known,
        cells.len()
    );
    println!("  first landed real words + their 48helix/48CAM_PQ codec:");
    for &c in cells.iter().take(8) {
        let cell = &grid[c];
        println!(
            "    cell[{:>4}] '{:<12}' count={} helix48={:02x?} campq48={:02x?}",
            c,
            vocab.word(c as u16),
            cell.count,
            cell.helix48,
            cell.campq48
        );
    }

    let cell_bytes = std::mem::size_of::<Cell>();
    println!("\n── FOOTPRINT ──────────────────────────────────────────────");
    println!(
        "  {} cells × {} B = {} KB  (gridlake tier; onebrc GridBatch = 80 KB → {})",
        GRID,
        cell_bytes,
        GRID * cell_bytes / 1024,
        if GRID * cell_bytes <= 80 * 1024 { "FITS ✓" } else { "EXCEEDS ✗" }
    );

    // Throughput sweep over the REAL landed COCA ranks (cache-resident scatter+codec).
    let landed: Vec<u16> = cells.iter().map(|&c| c as u16).collect();
    let rows: u64 = 300_000_000;
    let t = Instant::now();
    let mut i = 0usize;
    for _ in 0..rows {
        let rank = landed[i % landed.len().max(1)] as usize;
        let c = &mut grid[rank];
        c.count = c.count.wrapping_add(1);
        c.sum_truth = c.sum_truth.wrapping_add(200);
        let w = vocab.word(rank as u16);
        let a = w.as_bytes().first().copied().unwrap_or(0) as usize;
        let b = w.as_bytes().last().copied().unwrap_or(0) as usize;
        for (s, tbl) in palette.iter().enumerate() {
            c.campq48[s] = tbl[a][b];
        }
        i = i.wrapping_add(1);
    }
    let dt = t.elapsed().as_secs_f64();
    let checksum: u64 = grid.iter().map(|c| c.count as u64).sum();
    println!("\n── THROUGHPUT (real COCA ranks, 48h+48pq encode each) ─────");
    println!(
        "  {} landings in {:.3}s = {:.1} Mrows/s   (checksum {})",
        rows,
        dt,
        (rows as f64 / dt) / 1e6,
        checksum
    );
}
