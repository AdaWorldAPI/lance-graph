//! Measure BF16 conversion loss separately from projection loss.
//!
//! cargo run --manifest-path crates/bgz-tensor/Cargo.toml --example bf16_loss

fn main() {
    // Load Jina embeddings
    let mut raw: Vec<Vec<f32>> = Vec::new();
    for path in &["/tmp/jina_batch1.json", "/tmp/jina_batch2.json"] {
        let texts: Vec<&str> = (0..20).map(|_| "x").collect();
        if let Ok(json) = std::fs::read_to_string(path) {
            if let Ok(embs) = bgz_tensor::jina::parse_jina_response(&json, &texts) {
                for e in embs {
                    raw.push(e.vector);
                }
            }
        }
    }
    if raw.is_empty() {
        eprintln!("No Jina data");
        return;
    }

    let n = raw.len();
    println!(
        "=== BF16 Conversion Loss Analysis ({} vectors, dim={}) ===\n",
        n,
        raw[0].len()
    );

    // Ground truth: f32 cosine (no conversion at all)
    let mut gt: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            gt.push((
                i,
                j,
                bgz_tensor::stacked_n::cosine_f32_slice(&raw[i], &raw[j]),
            ));
        }
    }
    let gt_cos: Vec<f64> = gt.iter().map(|p| p.2).collect();

    // === Test 1: BF16 truncation ONLY (no projection) ===
    // Convert f32→BF16→f32, compute cosine on full 1024 dims
    let bf16_vecs: Vec<Vec<f32>> = raw
        .iter()
        .map(|v| {
            v.iter()
                .map(|&x| {
                    let bits = (x.to_bits() >> 16) as u16;
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        })
        .collect();

    let bf16_cos: Vec<f64> = gt
        .iter()
        .map(|&(i, j, _)| bgz_tensor::stacked_n::cosine_f32_slice(&bf16_vecs[i], &bf16_vecs[j]))
        .collect();

    let bf16_errors: Vec<f64> = gt_cos
        .iter()
        .zip(&bf16_cos)
        .map(|(a, b)| (a - b).abs())
        .collect();
    let bf16_pearson = bgz_tensor::quality::pearson(&gt_cos, &bf16_cos);
    let bf16_spearman = bgz_tensor::quality::spearman(&gt_cos, &bf16_cos);

    println!("1. BF16 truncation only (f32→BF16→f32, no projection):");
    println!("   Pearson:  {:.6}", bf16_pearson);
    println!("   Spearman: {:.6}", bf16_spearman);
    println!(
        "   Mean err: {:.6}",
        bf16_errors.iter().sum::<f64>() / bf16_errors.len() as f64
    );
    println!(
        "   Max err:  {:.6}\n",
        bf16_errors.iter().cloned().fold(0.0f64, f64::max)
    );

    // === Test 2: Projection ONLY (no BF16, f64 precision) ===
    // Golden-step project f32→Base17 using f64 accumulators (no BF16 involved)
    let base17_cos: Vec<f64> = gt
        .iter()
        .map(|&(i, j, _)| {
            let a = bgz_tensor::Base17::from_f32(&raw[i]);
            let b = bgz_tensor::Base17::from_f32(&raw[j]);
            a.cosine(&b)
        })
        .collect();

    let proj_pearson = bgz_tensor::quality::pearson(&gt_cos, &base17_cos);
    let proj_spearman = bgz_tensor::quality::spearman(&gt_cos, &base17_cos);
    let proj_errors: Vec<f64> = gt_cos
        .iter()
        .zip(&base17_cos)
        .map(|(a, b)| (a - b).abs())
        .collect();

    println!("2. Projection only (f32→i16[17] via f64 accumulator, no BF16):");
    println!("   Pearson:  {:.6}", proj_pearson);
    println!("   Spearman: {:.6}", proj_spearman);
    println!(
        "   Mean err: {:.6}",
        proj_errors.iter().sum::<f64>() / proj_errors.len() as f64
    );
    println!(
        "   Max err:  {:.6}\n",
        proj_errors.iter().cloned().fold(0.0f64, f64::max)
    );

    // === Test 3: BF16 + Projection (Base17::from_f32 on BF16-truncated data) ===
    let bf16_base17_cos: Vec<f64> = gt
        .iter()
        .map(|&(i, j, _)| {
            let a = bgz_tensor::Base17::from_f32(&bf16_vecs[i]);
            let b = bgz_tensor::Base17::from_f32(&bf16_vecs[j]);
            a.cosine(&b)
        })
        .collect();

    let both_pearson = bgz_tensor::quality::pearson(&gt_cos, &bf16_base17_cos);
    let both_spearman = bgz_tensor::quality::spearman(&gt_cos, &bf16_base17_cos);

    println!("3. BF16 + Projection (f32→BF16→f32→i16[17]):");
    println!("   Pearson:  {:.6}", both_pearson);
    println!("   Spearman: {:.6}\n", both_spearman);

    // === Test 4: Stacked at various SPD (BF16 + golden-step sampling) ===
    for spd in [4, 8, 16, 32, 64] {
        let enc: Vec<bgz_tensor::StackedN> = raw
            .iter()
            .map(|v| bgz_tensor::StackedN::from_f32(v, spd))
            .collect();
        let stacked_cos: Vec<f64> = gt.iter().map(|&(i, j, _)| enc[i].cosine(&enc[j])).collect();
        let sp = bgz_tensor::quality::spearman(&gt_cos, &stacked_cos);
        let pr = bgz_tensor::quality::pearson(&gt_cos, &stacked_cos);

        // Also measure hydrated (BF16→f32) cosine
        let hydrated: Vec<Vec<f32>> = enc.iter().map(|e| e.hydrate_f32()).collect();
        let hyd_cos: Vec<f64> = gt
            .iter()
            .map(|&(i, j, _)| bgz_tensor::stacked_n::cosine_f32_slice(&hydrated[i], &hydrated[j]))
            .collect();
        let hyd_pr = bgz_tensor::quality::pearson(&gt_cos, &hyd_cos);

        println!(
            "4. Stacked SPD={:>2}: Pearson={:.6} Spearman={:.6} Hydrated_Pearson={:.6}",
            spd, pr, sp, hyd_pr
        );
    }

    // === Summary ===
    println!("\n=== Loss Attribution ===");
    println!(
        "BF16 truncation alone:     Pearson loss = {:.6}",
        1.0 - bf16_pearson
    );
    println!(
        "Projection alone (17D):    Pearson loss = {:.6}",
        1.0 - proj_pearson
    );
    println!(
        "Both combined:             Pearson loss = {:.6}",
        1.0 - both_pearson
    );
    println!("Stacked SPD=32:            Pearson loss = {:.6}", {
        let enc: Vec<bgz_tensor::StackedN> = raw
            .iter()
            .map(|v| bgz_tensor::StackedN::from_f32(v, 32))
            .collect();
        let sc: Vec<f64> = gt.iter().map(|&(i, j, _)| enc[i].cosine(&enc[j])).collect();
        1.0 - bgz_tensor::quality::pearson(&gt_cos, &sc)
    });
}
