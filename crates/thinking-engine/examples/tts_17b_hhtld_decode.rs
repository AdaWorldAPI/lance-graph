//! BGZ-HHTL-D safetensors → inference-ready weight matrices.
//!
//! Reads the compressed safetensors produced by tts_17b_hhtld_encode and
//! rehydrates back to approximate f32 weight matrices via palette lookup.
//!
//! Rehydration pipeline per row:
//!   HhtlDEntry (4 bytes)
//!     → twig_centroid → palette[centroid_idx] → Base17 centroid
//!       → polarity + residual_bf16 → corrected Base17
//!         → f32[17] (golden-step unfold for inference)
//!
//! For TTS inference, the rehydrated weights feed into the HHTL cascade
//! which skips 95% of attention computation via RouteAction dispatch.
//!
//! ```sh
//! cargo run --release --example tts_17b_hhtld_decode \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /path/to/model_hhtld.safetensors
//! ```

use bgz_tensor::projection::{Base17, BASE_DIM};
use bgz_tensor::hhtl_d::HhtlDEntry;
use bgz_tensor::stacked_n::bf16_to_f32;

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

/// Parse safetensors header (minimal: header_size + JSON).
fn parse_hhtld_header(reader: &mut BufReader<File>) -> (serde_json::Value, u64) {
    let mut size_buf = [0u8; 8];
    reader.read_exact(&mut size_buf).unwrap();
    let header_size = u64::from_le_bytes(size_buf);

    let mut header_bytes = vec![0u8; header_size as usize];
    reader.read_exact(&mut header_bytes).unwrap();

    let header: serde_json::Value = serde_json::from_slice(&header_bytes).unwrap();
    let data_offset = 8 + header_size;
    (header, data_offset)
}

/// Read a raw tensor blob from the safetensors file.
fn read_tensor_bytes(reader: &mut BufReader<File>, header: &serde_json::Value,
                     data_offset: u64, tensor_name: &str) -> Option<Vec<u8>> {
    let entry = header.get(tensor_name)?;
    let offsets = entry.get("data_offsets")?.as_array()?;
    let begin = offsets[0].as_u64()? as u64;
    let end = offsets[1].as_u64()? as u64;
    let len = (end - begin) as usize;

    reader.seek(SeekFrom::Start(data_offset + begin)).ok()?;
    let mut data = vec![0u8; len];
    reader.read_exact(&mut data).ok()?;
    Some(data)
}

/// Reconstruct palette from raw bytes ([k × 34] = k Base17 entries).
fn decode_palette(bytes: &[u8]) -> Vec<Base17> {
    let k = bytes.len() / 34;
    let mut entries = Vec::with_capacity(k);
    for i in 0..k {
        let mut dims = [0i16; 17];
        for d in 0..17 {
            let offset = i * 34 + d * 2;
            dims[d] = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        }
        entries.push(Base17 { dims });
    }
    entries
}

/// Rehydrate a single row from HhtlDEntry + palette → approximate Base17.
///
/// The centroid provides the bulk signal. The residual + polarity provide
/// a first-order correction. This is lossy — the full f32 precision is lost —
/// but for inference via the HHTL cascade, only the centroid index matters
/// (the cascade uses palette distances, not full-precision values).
fn rehydrate_row(entry: &HhtlDEntry, palette: &[Base17]) -> Base17 {
    let centroid_idx = entry.twig_centroid() as usize;
    if centroid_idx >= palette.len() {
        return Base17 { dims: [0i16; 17] };
    }

    let centroid = &palette[centroid_idx];
    let residual = entry.residual_f32() as f64;
    let sign = if entry.polarity() { 1.0 } else { -1.0 };

    // Apply residual correction: centroid + sign * residual * |centroid|
    let centroid_mag: f64 = centroid.dims.iter()
        .map(|&d| (d as f64).abs())
        .sum::<f64>()
        / BASE_DIM as f64;

    let mut dims = [0i16; 17];
    for d in 0..BASE_DIM {
        let corrected = centroid.dims[d] as f64 + sign * residual * centroid_mag;
        dims[d] = corrected.clamp(-32768.0, 32767.0) as i16;
    }

    Base17 { dims }
}

/// Discover all role names from the safetensors header.
fn discover_roles(header: &serde_json::Value) -> Vec<String> {
    let obj = match header.as_object() {
        Some(o) => o,
        None => return Vec::new(),
    };

    let mut roles: Vec<String> = obj.keys()
        .filter(|k| k.ends_with(".hhtld_entries"))
        .map(|k| k.trim_end_matches(".hhtld_entries").to_string())
        .collect();
    roles.sort();
    roles
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let hhtld_path = if args.len() > 1 {
        &args[1]
    } else {
        "/home/user/models/Qwen3-TTS-12Hz-1.7B-Base/model_hhtld.safetensors"
    };

    println!("═══ BGZ-HHTL-D SAFETENSORS DECODER ═══");
    println!();

    // ─── Step 1: Parse header ──────────────────────────────────────────
    println!("[1] Parsing HHTL-D safetensors header...");
    let t0 = Instant::now();
    let mut reader = BufReader::new(File::open(hhtld_path).expect("open hhtld safetensors"));
    let (header, data_offset) = parse_hhtld_header(&mut reader);

    // Print metadata
    if let Some(meta) = header.get("__metadata__") {
        println!("    Encoding:    {}", meta.get("encoding").and_then(|v| v.as_str()).unwrap_or("?"));
        println!("    Original:    {}", meta.get("original_model").and_then(|v| v.as_str()).unwrap_or("?"));
        println!("    Palette k:   {}", meta.get("palette_k").and_then(|v| v.as_str()).unwrap_or("?"));
        println!("    Entries:     {}", meta.get("total_entries").and_then(|v| v.as_str()).unwrap_or("?"));
        println!("    Compression: {}:1", meta.get("compression_ratio").and_then(|v| v.as_str()).unwrap_or("?"));
    }
    println!("    Parsed in {:?}", t0.elapsed());

    // ─── Step 2: Discover and load roles ───────────────────────────────
    let roles = discover_roles(&header);
    println!("\n[2] Found {} encoded roles:", roles.len());
    for r in &roles { println!("    {}", r); }

    // ─── Step 3: Rehydrate each role ───────────────────────────────────
    println!("\n[3] Rehydrating weight matrices...");
    let mut total_rehydrated_rows = 0usize;

    for role_name in &roles {
        let t0 = Instant::now();

        // Load entries
        let entries_key = format!("{}.hhtld_entries", role_name);
        let entries_bytes = match read_tensor_bytes(&mut reader, &header, data_offset, &entries_key) {
            Some(b) => b,
            None => { eprintln!("    MISS: {}", entries_key); continue; }
        };
        let entries = bgz_tensor::hhtl_d::HhtlDTensor::entries_from_bytes(&entries_bytes);
        let n_rows = entries.len();

        // Load palette
        let palette_key = format!("{}.palette", role_name);
        let palette_bytes = match read_tensor_bytes(&mut reader, &header, data_offset, &palette_key) {
            Some(b) => b,
            None => { eprintln!("    MISS: {}", palette_key); continue; }
        };
        let palette = decode_palette(&palette_bytes);

        // Load distance table
        let dist_key = format!("{}.distance_table", role_name);
        let dist_bytes = read_tensor_bytes(&mut reader, &header, data_offset, &dist_key);

        // Load route table
        let route_key = format!("{}.route_table", role_name);
        let route_bytes = read_tensor_bytes(&mut reader, &header, data_offset, &route_key);

        // Load original shape
        let shape_key = format!("{}.original_shape", role_name);
        let shape = if let Some(sb) = read_tensor_bytes(&mut reader, &header, data_offset, &shape_key) {
            let r = u32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]) as usize;
            let c = u32::from_le_bytes([sb[4], sb[5], sb[6], sb[7]]) as usize;
            [r, c]
        } else {
            [n_rows, 0]
        };

        // Rehydrate all rows
        let rehydrated: Vec<Base17> = entries.iter()
            .map(|e| rehydrate_row(e, &palette))
            .collect();
        total_rehydrated_rows += n_rows;

        // Compute route statistics for this role
        let (n_skip, n_attend, n_compose, n_escalate) = if let (Some(rb), Some(db)) = (&route_bytes, &dist_bytes) {
            let k = palette.len();
            let mut skip = 0usize;
            let mut attend = 0usize;
            let mut compose = 0usize;
            let mut escalate = 0usize;
            for a in 0..k {
                for b in 0..k {
                    match rb[a * k + b] {
                        0 => skip += 1,
                        1 => attend += 1,
                        2 => compose += 1,
                        3 => escalate += 1,
                        _ => {}
                    }
                }
            }
            (skip, attend, compose, escalate)
        } else {
            (0, 0, 0, 0)
        };

        // Basin distribution
        let mut basin_counts = [0usize; 4];
        for e in &entries {
            let b = e.heel_basin() as usize;
            if b < 4 { basin_counts[b] += 1; }
        }

        println!("    {}: {} rows (orig {}×{}), palette k={}, skip={:.0}% ({:?})",
            role_name, n_rows, shape[0], shape[1], palette.len(),
            n_skip as f64 / (palette.len() * palette.len()).max(1) as f64 * 100.0,
            t0.elapsed());

        // Verify self-consistency: centroid → lookup → same centroid
        let mut mismatches = 0usize;
        for (i, entry) in entries.iter().enumerate().take(100) {
            let rehy = &rehydrated[i];
            // Find nearest centroid of rehydrated
            let nearest = (0..palette.len())
                .min_by_key(|&c| rehy.l1(&palette[c]))
                .unwrap_or(0);
            if nearest != entry.twig_centroid() as usize {
                mismatches += 1;
            }
        }
        if mismatches > 0 {
            println!("      ⚠ {} / 100 centroid mismatches after rehydration", mismatches);
        }
    }

    // ─── Step 4: Check passthrough tensors ─────────────────────────────
    let passthrough: Vec<String> = header.as_object()
        .map(|o| o.keys()
            .filter(|k| k.starts_with("passthrough."))
            .cloned()
            .collect())
        .unwrap_or_default();

    println!("\n[4] Passthrough tensors (norms, embeddings): {}", passthrough.len());
    for pt in &passthrough {
        if let Some(entry) = header.get(pt) {
            let offsets = entry.get("data_offsets").and_then(|o| o.as_array());
            if let Some(o) = offsets {
                let begin = o[0].as_u64().unwrap_or(0);
                let end = o[1].as_u64().unwrap_or(0);
                let size = end - begin;
                let name = pt.trim_start_matches("passthrough.");
                if size < 100000 {
                    println!("      {}: {} bytes", name, size);
                }
            }
        }
    }

    // ─── Summary ───────────────────────────────────────────────────────
    println!("\n[5] Summary:");
    println!("    Total rehydrated rows: {}", total_rehydrated_rows);
    println!("    Total roles decoded:   {}", roles.len());
    println!("    Passthrough tensors:   {}", passthrough.len());
    println!("    Ready for HHTL cascade inference.");
    println!();

    println!("═══ DONE ═══");
}
