//! Per-role variance audit on real bgz7 model weights.
//!
//! Usage: cargo run --manifest-path crates/bgz-tensor/Cargo.toml --example variance_audit -- /path/to/model.bgz7
//!
//! Reads bgz7 files, classifies tensors by role (Q/K/V/Gate/Up/Down),
//! and computes inter-role vs intra-role variance to validate NeuronPrint 6D.

fn main() {
    // Discover bgz7 files
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bgz7_paths: Vec<String> = if args.iter().any(|a| a == "--synthetic") {
        // Force synthetic mode
        Vec::new()
    } else if !args.is_empty() {
        args
    } else {
        // Default: look for known locations
        let candidates = vec![
            "/home/user/ndarray/src/hpc/openchat/weights/openchat-3.5-0106.bgz7",
            "/home/user/ndarray/src/hpc/openchat/weights/llama4_scout_shard1.bgz7",
            "/home/user/ndarray/src/hpc/openchat/weights/llama4_scout_shard2.bgz7",
            "/home/user/ndarray/src/hpc/openchat/weights/llama4_scout_shard3.bgz7",
            "/home/user/ndarray/src/hpc/openchat/weights/llama4_scout_shard4.bgz7",
            "/home/user/ndarray/src/hpc/openchat/weights/llama4_scout_shard5.bgz7",
        ];
        candidates
            .into_iter()
            .filter(|p| std::path::Path::new(p).exists())
            .map(|s| s.to_string())
            .collect()
    };

    if bgz7_paths.is_empty() {
        eprintln!("No bgz7 files found. Pass paths as arguments.");
        // Run synthetic audit instead
        run_synthetic_audit();
        return;
    }

    for path in &bgz7_paths {
        println!("\n============================================================");
        println!("Auditing: {}", path);
        println!("============================================================");
        audit_bgz7_file(path);
    }
}

fn audit_bgz7_file(path: &str) {
    // Read the bgz7 file directly (parse the binary format)
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Cannot read {}: {}", path, e);
            return;
        }
    };

    // Parse bgz7: [BGZ7 magic][n_tensors:u32][tensor*]
    if data.len() < 8 || &data[0..4] != b"BGZ7" {
        eprintln!("Not a valid bgz7 file: {}", path);
        return;
    }

    let n_tensors = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    println!("Tensors: {}", n_tensors);

    let mut pos = 8;
    let mut labeled = Vec::new();
    let mut tensor_count_by_role = [0usize; 6];
    let mut total_rows = 0;

    for _ in 0..n_tensors {
        if pos + 4 > data.len() {
            break;
        }

        // Parse: [name_len:u32][name][layer_type:u8][n_rows:u32][n_cols:u32][base17 × n_rows]
        let name_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if pos + name_len > data.len() {
            break;
        }

        let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
        pos += name_len;

        if pos + 1 > data.len() {
            break;
        }
        let _layer_type = data[pos];
        pos += 1;

        if pos + 8 > data.len() {
            break;
        }
        let n_rows =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let _n_cols =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        // Parse role from tensor name
        let role = bgz_tensor::variance::Role::from_name(&name);

        for _ in 0..n_rows {
            if pos + 34 > data.len() {
                break;
            }
            let b17 = bgz_tensor::Base17::from_bytes(&data[pos..pos + 34]);
            pos += 34;

            if let Some(r) = role {
                labeled.push((r, b17));
                tensor_count_by_role[r as usize] += 1;
            }
            total_rows += 1;
        }
    }

    println!("Total rows: {}, Labeled: {}", total_rows, labeled.len());
    for role in bgz_tensor::variance::Role::ALL {
        println!(
            "  {}: {} rows",
            role.label(),
            tensor_count_by_role[role as usize]
        );
    }

    if labeled.is_empty() {
        println!("No labeled tensors found.");
        return;
    }

    // Compute variance report
    let model_name = std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".into());

    let report = bgz_tensor::variance::compute_variance(&model_name, &labeled);
    println!("\n{}", report.summary());

    // Also compute pairwise L1 sample for Belichtungsmesser
    let sample_size = labeled.len().min(500);
    let mut l1_distances = Vec::new();
    for i in 0..sample_size {
        for j in (i + 1)..sample_size.min(i + 50) {
            l1_distances.push(labeled[i].1.l1(&labeled[j].1));
        }
    }

    if !l1_distances.is_empty() {
        let bel = bgz_tensor::belichtungsmesser::Belichtungsmesser::calibrate(&l1_distances);
        println!("\n{}", bel.summary());
    }
}

fn run_synthetic_audit() {
    println!("\n=== Synthetic Variance Audit (no bgz7 files available) ===\n");

    // Simulate what real models look like based on known findings:
    // - FfnGate has highest magnitude (0.6% of total diff)
    // - K is most stable across model sizes
    // - Q and V are intermediate

    let mut labeled = Vec::new();

    // Simulate Q: moderate magnitude, moderate variance
    for i in 0..200 {
        let mut dims = [0i16; 17];
        for d in 0..17 {
            dims[d] = 50 + ((i * 13 + d * 7) % 200) as i16 - 100;
        }
        labeled.push((bgz_tensor::variance::Role::Q, bgz_tensor::Base17 { dims }));
    }

    // Simulate K: low variance (most stable)
    for i in 0..200 {
        let mut dims = [0i16; 17];
        for d in 0..17 {
            dims[d] = 30 + ((i * 5 + d * 3) % 50) as i16 - 25;
        }
        labeled.push((bgz_tensor::variance::Role::K, bgz_tensor::Base17 { dims }));
    }

    // Simulate V: similar to Q
    for i in 0..200 {
        let mut dims = [0i16; 17];
        for d in 0..17 {
            dims[d] = -80 + ((i * 11 + d * 9) % 180) as i16 - 90;
        }
        labeled.push((bgz_tensor::variance::Role::V, bgz_tensor::Base17 { dims }));
    }

    // Simulate Gate: HIGH magnitude (dominant role per findings)
    for i in 0..200 {
        let mut dims = [0i16; 17];
        for d in 0..17 {
            dims[d] = 500 + ((i * 17 + d * 11) % 1000) as i16 - 500;
        }
        labeled.push((
            bgz_tensor::variance::Role::Gate,
            bgz_tensor::Base17 { dims },
        ));
    }

    // Simulate Up: moderate-high
    for i in 0..200 {
        let mut dims = [0i16; 17];
        for d in 0..17 {
            dims[d] = 200 + ((i * 19 + d * 13) % 400) as i16 - 200;
        }
        labeled.push((bgz_tensor::variance::Role::Up, bgz_tensor::Base17 { dims }));
    }

    // Simulate Down: moderate, offset from Up
    for i in 0..200 {
        let mut dims = [0i16; 17];
        for d in 0..17 {
            dims[d] = -300 + ((i * 23 + d * 17) % 600) as i16 - 300;
        }
        labeled.push((
            bgz_tensor::variance::Role::Down,
            bgz_tensor::Base17 { dims },
        ));
    }

    let report = bgz_tensor::variance::compute_variance("synthetic_model", &labeled);
    println!("{}", report.summary());

    // NeuronPrint 6D palette compression test
    println!("\n=== NeuronPrint 6D Palette Compression Test ===\n");

    // Build palette from all vectors
    let all_b17: Vec<bgz_tensor::Base17> = labeled.iter().map(|(_, b)| b.clone()).collect();
    let palette = bgz_tensor::WeightPalette::build(&all_b17, 256);
    println!(
        "Palette: {} entries, max distortion: {}",
        palette.len(),
        palette.max_distortion()
    );
    println!("Mean distortion: {:.1}", palette.mean_distortion());

    // Compress each role and measure fidelity
    for role in bgz_tensor::variance::Role::ALL {
        let role_vecs: Vec<&bgz_tensor::Base17> = labeled
            .iter()
            .filter(|(r, _)| *r == role)
            .map(|(_, b)| b)
            .collect();

        if role_vecs.is_empty() {
            continue;
        }

        let mut total_distortion = 0u64;
        let mut max_dist = 0u32;
        for v in &role_vecs {
            let assignment = palette.assign(v);
            total_distortion += assignment.distortion as u64;
            max_dist = max_dist.max(assignment.distortion);
        }
        let mean_dist = total_distortion as f64 / role_vecs.len() as f64;

        // Measure cosine preservation: random pairs before vs after compression
        let n_sample = role_vecs.len().min(50);
        let mut cos_before = Vec::new();
        let mut cos_after = Vec::new();
        for i in 0..n_sample {
            for j in (i + 1)..n_sample.min(i + 10) {
                let orig_cos = role_vecs[i].cosine(role_vecs[j]);
                let idx_i = palette.assign(role_vecs[i]).index;
                let idx_j = palette.assign(role_vecs[j]).index;
                let comp_cos =
                    palette.entries[idx_i as usize].cosine(&palette.entries[idx_j as usize]);
                cos_before.push(orig_cos);
                cos_after.push(comp_cos);
            }
        }

        let cos_pearson = if cos_before.len() > 1 {
            bgz_tensor::quality::pearson(&cos_before, &cos_after)
        } else {
            0.0
        };

        println!(
            "  {:<5}: mean_dist={:>8.1}, max_dist={:>6}, cos_preservation={:.4}",
            role.label(),
            mean_dist,
            max_dist,
            cos_pearson
        );
    }

    // Full NeuronPrint: 6 indices = 6 bytes per neuron
    println!("\nCompression: 204 bytes (6×34) → 6 bytes (6×u8 palette index)");
    println!("Ratio: {:.0}×", 204.0 / 6.0);

    // Cross-role distance preservation
    println!("\n=== Cross-Role Distance After Compression ===");
    let centroids_by_role: Vec<(bgz_tensor::variance::Role, bgz_tensor::Base17)> = report
        .per_role
        .iter()
        .map(|rs| (rs.role, rs.centroid.clone()))
        .collect();

    for i in 0..centroids_by_role.len() {
        for j in (i + 1)..centroids_by_role.len() {
            let (r1, ref c1) = centroids_by_role[i];
            let (r2, ref c2) = centroids_by_role[j];
            let orig_l1 = c1.l1(c2);
            let idx1 = palette.assign(c1).index;
            let idx2 = palette.assign(c2).index;
            let comp_l1 = palette.entries[idx1 as usize].l1(&palette.entries[idx2 as usize]);
            println!(
                "  {} ↔ {}: orig_L1={:>5}, compressed_L1={:>5}, ratio={:.2}",
                r1.label(),
                r2.label(),
                orig_l1,
                comp_l1,
                comp_l1 as f64 / orig_l1.max(1) as f64
            );
        }
    }
}
