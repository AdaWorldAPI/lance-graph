//! CAM-PQ diagnostic: ICC vs calibration row count.
//!
//! Runs `train_geometric` on an isolated tensor at increasing row counts
//! ({128, 256, 512, 1024, n_rows}) and reports the ICC_3_1 score between
//! pairwise cosines of the original vs decoded rows, measured on the
//! training population itself.
//!
//! **Purpose:** demonstrate whether the small-row-count ICC values from
//! `codec_rnd_bench.rs` (which measured 128 rows and saw ICC ≈ 0.9998)
//! extrapolate to production-size tensors.
//!
//! Run:
//! ```sh
//! cargo run --release --features calibrate --example cam_pq_row_count_probe \
//!     --manifest-path crates/bgz-tensor/Cargo.toml \
//!     -- <safetensors_path> <tensor_name>
//! ```

use ndarray::hpc::cam_pq::{self, CamCodebook, NUM_SUBSPACES};
use ndarray::hpc::gguf::read_tensor_f32;
use ndarray::hpc::safetensors::read_safetensors_header;
use std::fs::File;
use std::io::BufReader;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: cam_pq_row_count_probe <safetensors_path> <tensor_name_substring>");
        std::process::exit(1);
    }
    let path = &args[1];
    let pattern = &args[2];

    let file = File::open(path).expect("open safetensors");
    let mut reader = BufReader::new(file);
    let gguf = read_safetensors_header(&mut reader).expect("read header");

    let tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name.contains(pattern))
        .unwrap_or_else(|| {
            eprintln!("No tensor name contains {pattern:?}");
            std::process::exit(1);
        });

    println!("tensor: {}  dims: {:?}", tensor.name, tensor.dimensions);

    let flat = read_tensor_f32(&mut reader, &gguf, tensor).expect("read tensor");

    let (row_dim, n_rows) = if tensor.dimensions.len() == 2 {
        (tensor.dimensions[1] as usize, tensor.dimensions[0] as usize)
    } else {
        eprintln!("Expected 2D tensor; got {:?}", tensor.dimensions);
        std::process::exit(1);
    };
    let adjusted_dim = (row_dim / NUM_SUBSPACES) * NUM_SUBSPACES;
    let rows: Vec<Vec<f32>> = flat
        .chunks_exact(row_dim)
        .map(|c| c[..adjusted_dim].to_vec())
        .collect();
    assert_eq!(rows.len(), n_rows);

    let test_counts = [128, 256, 512, 1024, n_rows];

    println!();
    println!(
        "{:>8} | {:>10} | {:>12} | {:>10} | {:>12}",
        "n_train", "icc_train", "icc_all_rows", "rel_err", "time_s"
    );
    println!("{}", "-".repeat(62));

    let mut seen = std::collections::BTreeSet::new();
    for &n_train in &test_counts {
        if n_train > n_rows || !seen.insert(n_train) {
            continue;
        }
        let start = std::time::Instant::now();
        let training = &rows[..n_train];
        let cb = cam_pq::train_geometric(training, adjusted_dim, 20);
        let elapsed = start.elapsed();

        let icc_train = measure_icc(training, &cb, 512);
        let icc_all = measure_icc(&rows, &cb, 512);
        let rel_err = relative_l2_error(&cb, &rows[..rows.len().min(512)]);

        println!(
            "{:>8} | {:>10.4} | {:>12.4} | {:>10.4} | {:>12.2}",
            n_train,
            icc_train,
            icc_all,
            rel_err,
            elapsed.as_secs_f32(),
        );
    }

    println!();
    println!("Hypothesis: `icc_train` stays high (codebook fits training data);");
    println!("`icc_all_rows` collapses as n_train increases relative to codebook");
    println!("capacity (6 subspaces × 256 centroids = 256^6 possible fingerprints,");
    println!("but only 256 per-subspace partitions — ~n_train/256 rows land per");
    println!("centroid at saturation).");
}

fn measure_icc(rows: &[Vec<f32>], cb: &CamCodebook, samples: usize) -> f32 {
    let n = rows.len();
    if n < 3 {
        return f32::NAN;
    }
    let samples = samples.min(n * (n - 1) / 2);
    let mut rng = SimpleRng::new(0x9E3779B97F4A7C15);
    let mut truth: Vec<f32> = Vec::with_capacity(samples);
    let mut pred: Vec<f32> = Vec::with_capacity(samples);
    let mut count = 0;
    while count < samples {
        let i = (rng.next() as usize) % n;
        let j = (rng.next() as usize) % n;
        if i == j {
            continue;
        }
        let t = cosine(&rows[i], &rows[j]);
        let di = cb.decode(&cb.encode(&rows[i]));
        let dj = cb.decode(&cb.encode(&rows[j]));
        let p = cosine(&di, &dj);
        truth.push(t);
        pred.push(p);
        count += 1;
    }
    icc_3_1(&truth, &pred)
}

fn relative_l2_error(cb: &CamCodebook, rows: &[Vec<f32>]) -> f32 {
    let mut sum_err = 0.0f64;
    let mut sum_norm = 0.0f64;
    for row in rows {
        let decoded = cb.decode(&cb.encode(row));
        for (a, b) in row.iter().zip(decoded.iter()) {
            sum_err += ((a - b) as f64).powi(2);
        }
        for &a in row {
            sum_norm += (a as f64).powi(2);
        }
    }
    if sum_norm > 0.0 {
        (sum_err / sum_norm).sqrt() as f32
    } else {
        0.0
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let d = (na * nb).sqrt();
    if d > 0.0 {
        (dot / d) as f32
    } else {
        0.0
    }
}

fn icc_3_1(truth: &[f32], pred: &[f32]) -> f32 {
    let n = truth.len();
    if n < 2 {
        return f32::NAN;
    }
    let mut grand = 0.0f64;
    for i in 0..n {
        grand += (truth[i] + pred[i]) as f64;
    }
    grand /= (2 * n) as f64;
    let mut ms_r = 0.0f64;
    let mut ms_w = 0.0f64;
    for i in 0..n {
        let row_mean = ((truth[i] + pred[i]) as f64) / 2.0;
        ms_r += 2.0 * (row_mean - grand).powi(2);
        ms_w += (truth[i] as f64 - row_mean).powi(2) + (pred[i] as f64 - row_mean).powi(2);
    }
    ms_r /= (n - 1) as f64;
    ms_w /= n as f64;
    ((ms_r - ms_w) / (ms_r + ms_w)) as f32
}

struct SimpleRng {
    state: u64,
}
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}
