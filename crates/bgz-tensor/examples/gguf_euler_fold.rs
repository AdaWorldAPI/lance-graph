//! Euler-gamma fold on real Jina GGUF weight tensors.
//!
//! cargo run --release --manifest-path crates/bgz-tensor/Cargo.toml --example gguf_euler_fold

use bgz_tensor::euler_fold::{clam_group, euler_gamma_fold, euler_gamma_unfold, gate_test};
use bgz_tensor::stacked_n::cosine_f32_slice;
use bgz_tensor::variance::Role;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/tmp/hf_cache/models--gaianet--jina-embeddings-v3-GGUF/snapshots/d7d998aab1a7f2ea0aa256d8b3f035cbd0af682a/jina-embeddings-v3-Q8_0.gguf".into()
    });

    println!("=== Euler-Gamma Fold on Real GGUF Weights ===\n");

    let mut file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Cannot open: {}", e);
            return;
        }
    };

    let header = match parse_gguf_header(&mut file) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            return;
        }
    };

    println!("Model: {} tensors\n", header.tensors.len());

    // Collect weight rows by role
    let mut role_rows: HashMap<Role, Vec<Vec<f32>>> = HashMap::new();
    let mut tensors_done = 0;

    for tensor in &header.tensors {
        let role = match Role::from_name(&tensor.name) {
            Some(r) => r,
            None => continue,
        };
        if tensor.n_elements < 1024 {
            continue;
        }

        let data = match read_tensor_f32(&mut file, &header, tensor) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let (n_rows, n_cols) = if tensor.dims.len() >= 2 {
            (
                tensor.dims[0] as usize,
                tensor.dims[1..].iter().map(|&d| d as usize).product(),
            )
        } else {
            (1, data.len())
        };

        let entry = role_rows.entry(role).or_default();
        for r in 0..n_rows.min(500) {
            // limit per tensor for speed
            let start = r * n_cols;
            let end = (start + n_cols).min(data.len());
            if end > start {
                entry.push(data[start..end].to_vec());
            }
        }

        tensors_done += 1;
        if tensors_done % 20 == 0 {
            eprint!("\r{} tensors...", tensors_done);
        }
    }
    eprintln!("\r{} tensors processed", tensors_done);

    // ═══ PART 1: Measure inter-row cosine, then CLAM group, then fold ══
    println!("\n=== PART 1: CLAM Family → Euler Fold on Real Weight Rows ===\n");

    for role in Role::ALL {
        let rows = match role_rows.get(&role) {
            Some(r) if r.len() >= 50 => r,
            _ => continue,
        };

        let sample = &rows[..rows.len().min(300)];
        let dim = sample[0].len();

        // First: what does inter-row cosine look like?
        let mut cosines = Vec::new();
        for i in 0..sample.len().min(50) {
            for j in (i + 1)..sample.len().min(50) {
                cosines.push(cosine_f32_slice(&sample[i], &sample[j]));
            }
        }
        cosines.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mean_cos = cosines.iter().sum::<f64>() / cosines.len() as f64;
        let top10_cos = cosines[..10.min(cosines.len())].iter().sum::<f64>()
            / 10.0f64.min(cosines.len() as f64);
        println!(
            "--- {} ({} rows, dim={}) ---",
            role.label(),
            sample.len(),
            dim
        );
        println!(
            "  Inter-row cosine: mean={:.4}, top-10 mean={:.4}",
            mean_cos, top10_cos
        );

        // CLAM group: find families of genuinely similar rows
        let families = clam_group(sample, 0.80);
        let multi_member: Vec<_> = families
            .iter()
            .filter(|f| f.member_indices.len() >= 2)
            .collect();
        println!(
            "  CLAM families: {} total, {} with >=2 members",
            families.len(),
            multi_member.len()
        );

        if multi_member.is_empty() {
            println!("  No families with >=2 members (rows too dissimilar). Skip fold.\n");
            continue;
        }

        // Family size distribution
        let sizes: Vec<usize> = multi_member
            .iter()
            .map(|f| f.member_indices.len())
            .collect();
        println!(
            "  Family sizes: min={}, max={}, mean={:.1}",
            sizes.iter().min().unwrap(),
            sizes.iter().max().unwrap(),
            sizes.iter().sum::<usize>() as f64 / sizes.len() as f64
        );

        // Gate test: fold each multi-member family and measure recovery
        let mut all_pearsons: Vec<f64> = Vec::new();
        let mut all_ratios: Vec<f64> = Vec::new();

        for fam in multi_member.iter().take(10) {
            let members: Vec<Vec<f32>> = fam
                .member_indices
                .iter()
                .map(|&i| sample[i].clone())
                .collect();
            let n = members.len().min(8); // cap at 8 for speed
            let result = gate_test(&members[..n], 32);
            all_pearsons.push(result.mean_pearson);
            all_ratios.push(result.compression_ratio);
        }

        let mean_p = all_pearsons.iter().sum::<f64>() / all_pearsons.len() as f64;
        let min_p = all_pearsons.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean_r = all_ratios.iter().sum::<f64>() / all_ratios.len() as f64;
        println!(
            "  Fold recovery (within families): mean_ρ={:.4}, min_ρ={:.4}, mean_ratio={:.1}×\n",
            mean_p, min_p, mean_r
        );
    }

    // ═══ PART 2: NeuronPrint 6D fold — 6 roles → 1 container ═══════════
    println!("=== PART 2: NeuronPrint 6D Fold ===\n");

    // For each "neuron" (row index), take Q[i], K[i], V[i], Gate[i], Up[i], Down[i]
    // and fold them into one container.
    let roles_needed = [Role::Q, Role::Up, Role::Down]; // K/V/Gate may be missing from Jina
    let available: Vec<Role> = roles_needed
        .iter()
        .filter(|r| role_rows.get(r).map_or(false, |v| !v.is_empty()))
        .copied()
        .collect();

    if available.len() >= 2 {
        // Find common dimensionality: pad shorter vectors to longest
        let max_dim = available
            .iter()
            .map(|r| role_rows[r][0].len())
            .max()
            .unwrap_or(0);
        let min_rows = available
            .iter()
            .map(|r| role_rows[r].len())
            .min()
            .unwrap_or(0);

        let test_count = min_rows.min(20);
        println!(
            "Folding {} roles × {} neurons (padded to dim={}):",
            available.len(),
            test_count,
            max_dim
        );

        let mut all_pearsons: Vec<f64> = Vec::new();
        for neuron in 0..test_count {
            let members: Vec<Vec<f32>> = available
                .iter()
                .map(|r| {
                    let mut v = role_rows[r][neuron].clone();
                    v.resize(max_dim, 0.0); // pad to common dim
                    v
                })
                .collect();

            let result = gate_test(&members, 32);
            all_pearsons.extend(result.pearson_per_member.iter());

            if neuron < 5 {
                println!(
                    "  Neuron {:>2}: mean_ρ={:.4}, per_role: [{}]",
                    neuron,
                    result.mean_pearson,
                    result
                        .pearson_per_member
                        .iter()
                        .zip(available.iter())
                        .map(|(p, r)| format!("{}={:.3}", r.label(), p))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }

        let global_mean = all_pearsons.iter().sum::<f64>() / all_pearsons.len() as f64;
        let global_min = all_pearsons.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "\nNeuronPrint fold ({} roles): mean_ρ={:.4}, min_ρ={:.4}, ratio={:.1}×",
            available.len(),
            global_mean,
            global_min,
            available.len() as f64 / 2.0
        );
    } else {
        println!("Not enough roles available for NeuronPrint test.");
    }

    // ═══ PART 3: CLAM family compression on real data ════════════════════
    println!("\n=== PART 3: CLAM Family + Euler Fold Compression ===\n");

    for role in Role::ALL {
        let rows = match role_rows.get(&role) {
            Some(r) if r.len() >= 50 => r,
            _ => continue,
        };

        let sample = &rows[..rows.len().min(500)];

        // Group into families
        let families = clam_group(sample, 0.85);
        let raw_bytes = sample.len() * sample[0].len() * 4; // f32

        // Fold each family
        let mut total_folded_bytes = 0;
        let mut total_recovery_pearson = 0.0f64;
        let mut n_recovered = 0;

        for fam in &families {
            if fam.member_indices.len() < 2 {
                continue;
            }

            let members: Vec<Vec<f32>> = fam
                .member_indices
                .iter()
                .map(|&i| sample[i].clone())
                .collect();

            let folded = euler_gamma_fold(&members, 32);
            total_folded_bytes += folded.byte_size();

            // Sample recovery quality
            let test_count = members.len().min(4);
            for j in 0..test_count {
                let recovered = euler_gamma_unfold(&folded, j);
                let enc_orig = bgz_tensor::StackedN::from_f32(&members[j], 32);
                let orig_h = enc_orig.hydrate_f32();
                let r = bgz_tensor::quality::pearson(
                    &orig_h.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                    &recovered.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                );
                total_recovery_pearson += r;
                n_recovered += 1;
            }
        }

        let ratio = raw_bytes as f64 / total_folded_bytes.max(1) as f64;
        let mean_r = if n_recovered > 0 {
            total_recovery_pearson / n_recovered as f64
        } else {
            0.0
        };
        println!(
            "{:<5}: {} vecs → {} families, ratio={:.1}×, recovery_ρ={:.4}",
            role.label(),
            sample.len(),
            families.len(),
            ratio,
            mean_r
        );
    }

    println!("\n=== DONE ===");
}

// ═══════════════════════════════════════════════════════════════════════════
// Minimal GGUF reader (copy from gguf_families.rs)
// ═══════════════════════════════════════════════════════════════════════════

struct GgufHeader {
    tensors: Vec<TensorMeta>,
    data_offset: u64,
}
struct TensorMeta {
    name: String,
    dims: Vec<u64>,
    dtype: u32,
    offset: u64,
    n_elements: u64,
}

fn parse_gguf_header<R: Read + Seek>(r: &mut R) -> Result<GgufHeader, String> {
    let mut b4 = [0u8; 4];
    let mut b8 = [0u8; 8];
    r.read_exact(&mut b4).map_err(|e| e.to_string())?;
    if u32::from_le_bytes(b4) != 0x46554747 {
        return Err("bad magic".into());
    }
    r.read_exact(&mut b4).map_err(|e| e.to_string())?;
    r.read_exact(&mut b8).map_err(|e| e.to_string())?;
    let nt = u64::from_le_bytes(b8) as usize;
    r.read_exact(&mut b8).map_err(|e| e.to_string())?;
    let nm = u64::from_le_bytes(b8) as usize;
    for _ in 0..nm {
        skip_kv(r)?;
    }
    let mut tensors = Vec::with_capacity(nt);
    for _ in 0..nt {
        r.read_exact(&mut b8).map_err(|e| e.to_string())?;
        let nl = u64::from_le_bytes(b8) as usize;
        let mut nb = vec![0u8; nl];
        r.read_exact(&mut nb).map_err(|e| e.to_string())?;
        let name = String::from_utf8_lossy(&nb).to_string();
        r.read_exact(&mut b4).map_err(|e| e.to_string())?;
        let nd = u32::from_le_bytes(b4) as usize;
        let mut dims = Vec::with_capacity(nd);
        for _ in 0..nd {
            r.read_exact(&mut b8).map_err(|e| e.to_string())?;
            dims.push(u64::from_le_bytes(b8));
        }
        r.read_exact(&mut b4).map_err(|e| e.to_string())?;
        let dtype = u32::from_le_bytes(b4);
        r.read_exact(&mut b8).map_err(|e| e.to_string())?;
        let offset = u64::from_le_bytes(b8);
        let ne: u64 = dims.iter().product();
        tensors.push(TensorMeta {
            name,
            dims,
            dtype,
            offset,
            n_elements: ne,
        });
    }
    let pos = r.stream_position().map_err(|e| e.to_string())?;
    Ok(GgufHeader {
        tensors,
        data_offset: (pos + 31) / 32 * 32,
    })
}

fn skip_kv<R: Read + Seek>(r: &mut R) -> Result<(), String> {
    let mut b4 = [0u8; 4];
    let mut b8 = [0u8; 8];
    r.read_exact(&mut b8).map_err(|e| e.to_string())?;
    let kl = u64::from_le_bytes(b8) as usize;
    let mut kb = vec![0u8; kl];
    r.read_exact(&mut kb).map_err(|e| e.to_string())?;
    r.read_exact(&mut b4).map_err(|e| e.to_string())?;
    skip_val(r, u32::from_le_bytes(b4))
}

fn skip_val<R: Read + Seek>(r: &mut R, vt: u32) -> Result<(), String> {
    let mut b4 = [0u8; 4];
    let mut b8 = [0u8; 8];
    match vt {
        0 | 1 | 7 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
        }
        2 | 3 => {
            r.read_exact(&mut [0u8; 2]).map_err(|e| e.to_string())?;
        }
        4 | 5 | 6 => {
            r.read_exact(&mut b4).map_err(|e| e.to_string())?;
        }
        8 => {
            r.read_exact(&mut b8).map_err(|e| e.to_string())?;
            let l = u64::from_le_bytes(b8) as usize;
            let mut s = vec![0u8; l];
            r.read_exact(&mut s).map_err(|e| e.to_string())?;
        }
        9 => {
            r.read_exact(&mut b4).map_err(|e| e.to_string())?;
            let et = u32::from_le_bytes(b4);
            r.read_exact(&mut b8).map_err(|e| e.to_string())?;
            let c = u64::from_le_bytes(b8) as usize;
            for _ in 0..c {
                skip_val(r, et)?;
            }
        }
        10 | 11 | 12 => {
            r.read_exact(&mut b8).map_err(|e| e.to_string())?;
        }
        _ => return Err(format!("unknown vtype {}", vt)),
    }
    Ok(())
}

fn read_tensor_f32<R: Read + Seek>(
    r: &mut R,
    h: &GgufHeader,
    t: &TensorMeta,
) -> Result<Vec<f32>, String> {
    r.seek(SeekFrom::Start(h.data_offset + t.offset))
        .map_err(|e| e.to_string())?;
    let n = t.n_elements as usize;
    match t.dtype {
        0 => {
            let mut buf = vec![0u8; n * 4];
            r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        8 => {
            let nb = (n + 31) / 32;
            let bpb = 34;
            let mut buf = vec![0u8; nb * bpb];
            r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            let mut res = Vec::with_capacity(n);
            for b in 0..nb {
                let o = b * bpb;
                let sb = u16::from_le_bytes([buf[o], buf[o + 1]]);
                let s = f32::from_bits((sb as u32) << 16);
                for i in 0..32 {
                    if res.len() >= n {
                        break;
                    }
                    res.push(buf[o + 2 + i] as i8 as f32 * s);
                }
            }
            Ok(res)
        }
        _ => Err(format!("unsupported dtype {}", t.dtype)),
    }
}
