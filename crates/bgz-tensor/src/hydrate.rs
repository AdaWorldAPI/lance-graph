//! Hydrate binary: download or reindex bgz7 model shards.
//!
//! ```bash
//! cargo run --manifest-path crates/bgz-tensor/Cargo.toml \
//!   --features hydrate --bin hydrate -- --list
//! ```

use bgz_tensor::manifest::{
    self, bgz7_path, enabled_models, is_enabled, is_hydrated, load_manifest, verify_sha256,
};
use std::{env, fs, process};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        usage();
        process::exit(1);
    }

    let command = &args[1];
    let model = if args.len() > 2 { &args[2] } else { "" };

    let manifest = load_manifest().expect("Failed to load data/manifest.json");

    match command.as_str() {
        "--list" => cmd_list(&manifest),
        "--download" if model == "--enabled" || model.is_empty() => cmd_download_enabled(&manifest),
        "--download" => cmd_download(&manifest, model),
        "--reindex" => cmd_reindex(&manifest, model),
        "--verify" => cmd_verify(&manifest, model),
        "--help" | "-h" => usage(),
        _ => {
            eprintln!("Unknown command: {command}");
            usage();
            process::exit(1);
        }
    }
}

fn usage() {
    eprintln!("bgz-tensor hydrate — manage model tensor indexes");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  hydrate --list                 Show all models and hydration status");
    eprintln!("  hydrate --download             Download all feature-enabled models");
    eprintln!("  hydrate --download MODEL       Download a specific model");
    eprintln!("  hydrate --reindex MODEL        Stream from HuggingFace, build bgz7 locally");
    eprintln!("  hydrate --verify MODEL         Check SHA256 of existing shards");
    eprintln!();
    eprintln!("Feature flags control which models are enabled (zero download by default):");
    eprintln!("  qwen35-9b      80 MB  — quick thinking, shallow routing");
    eprintln!("  qwen35-27b-v1  174 MB — Opus 4.5 behavior (deep reasoning)");
    eprintln!("  qwen35-27b-v2  174 MB — Opus 4.6 precision (code/format)");
    eprintln!("  qwen35-full    430 MB — all variants");
}

fn cmd_list(manifest: &manifest::Manifest) {
    let enabled = enabled_models();
    eprintln!("bgz-tensor model index");
    if enabled.is_empty() {
        eprintln!("  No models enabled. Add features: qwen35-9b, qwen35-27b-v1, qwen35-27b-v2");
    } else {
        eprintln!("  Enabled: {}", enabled.join(", "));
    }
    eprintln!();
    for (name, entry) in &manifest.models {
        let flag = if is_enabled(name) { "►" } else { " " };
        let status = if is_hydrated(name, entry.shards) {
            "HYDRATED"
        } else if is_enabled(name) {
            "ENABLED"
        } else {
            "disabled"
        };
        println!(
            " {flag} {status:>10}  {name:<35} {shards:>2} shards  {mb:>6.0} MB  ({source})",
            shards = entry.shards,
            mb = entry.total_bytes_bgz7 as f64 / 1_000_000.0,
            source = entry.source,
        );
    }
}

fn cmd_download_enabled(manifest: &manifest::Manifest) {
    let enabled = enabled_models();
    if enabled.is_empty() {
        eprintln!("No models enabled. Add features to Cargo.toml:");
        eprintln!("  bgz-tensor = {{ features = [\"qwen35-9b\"] }}");
        process::exit(1);
    }
    for model in &enabled {
        let entry = match manifest.models.get(*model) {
            Some(e) => e,
            None => continue,
        };
        if is_hydrated(model, entry.shards) {
            println!("{model}: already hydrated, skipping");
            continue;
        }
        println!("\n═══ Downloading {model} ═══");
        cmd_download(manifest, model);
    }
}

fn cmd_download(manifest: &manifest::Manifest, model: &str) {
    let entry = manifest.models.get(model).unwrap_or_else(|| {
        eprintln!("Unknown model: {model}");
        eprintln!(
            "Available: {}",
            manifest
                .models
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
        process::exit(1)
    });

    let dir = bgz7_path(model, 0).parent().unwrap().to_path_buf();
    fs::create_dir_all(&dir).expect("Failed to create data directory");

    let repo = "AdaWorldAPI/lance-graph";
    let tag = &entry.release_tag;

    for shard in 0..entry.shards {
        let local_filename = format!("shard-{shard:02}.bgz7");
        let dest = dir.join(&local_filename);

        if dest.exists() && fs::metadata(&dest).map(|m| m.len() > 0).unwrap_or(false) {
            println!("  {local_filename}: already present, skipping");
            continue;
        }

        // Release assets are 1-indexed (shard-01..shard-11),
        // local storage is 0-indexed (shard-00..shard-10) matching manifest.
        let release_shard = shard + 1;
        let asset_name = format!("{model}--shard-{release_shard:02}.bgz7");
        let url = format!("https://github.com/{repo}/releases/download/{tag}/{asset_name}");
        println!("  Downloading {local_filename} (from asset {asset_name})...");

        let status = process::Command::new("curl")
            .args([
                "-fSL",
                "--retry",
                "4",
                "--retry-delay",
                "2",
                "-o",
                dest.to_str().unwrap(),
                &url,
            ])
            .status()
            .expect("curl not found");

        if !status.success() {
            eprintln!("  FAILED to download {local_filename}");
            let _ = fs::remove_file(&dest);
            process::exit(1);
        }
    }

    println!("Done. Verify: hydrate --verify {model}");
}

fn cmd_reindex(manifest: &manifest::Manifest, model: &str) {
    let entry = manifest.models.get(model).unwrap_or_else(|| {
        eprintln!("Unknown model: {model}");
        process::exit(1)
    });

    eprintln!("Reindexing {model} from {} ...", entry.source);
    eprintln!("This streams BF16 safetensors from HuggingFace and builds bgz7 shards.");
    eprintln!("Expected time: ~1-4 hours depending on model size and bandwidth.");
    eprintln!();
    eprintln!("For now, run indexing from the ndarray test suite:");
    eprintln!(
        "  cd ../../../ndarray && cargo test -p ndarray --lib test_index_{} --release -- --ignored --nocapture",
        model.replace('-', "_")
    );
    eprintln!();
    eprintln!("Then copy the shards:");
    let dir = bgz7_path(model, 0).parent().unwrap().to_path_buf();
    for shard in 0..entry.shards {
        let src = format!(
            "/tmp/{}_{}_shard{:02}.bgz7",
            model.replace('-', "_").replace("distilled_", ""),
            if model.contains("distilled") { "" } else { "" },
            shard + 1
        );
        let dest = dir.join(format!("shard-{shard:02}.bgz7"));
        eprintln!("  cp {} {}", src, dest.display());
    }
}

fn cmd_verify(manifest: &manifest::Manifest, model: &str) {
    let entry = manifest.models.get(model).unwrap_or_else(|| {
        eprintln!("Unknown model: {model}");
        process::exit(1)
    });

    let mut all_ok = true;
    for shard in 0..entry.shards {
        let filename = format!("shard-{shard:02}.bgz7");
        let path = bgz7_path(model, shard);

        if !path.exists() {
            println!("  {filename}: MISSING");
            all_ok = false;
            continue;
        }

        let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        if size == 0 {
            println!("  {filename}: EMPTY (0 bytes)");
            all_ok = false;
            continue;
        }

        if let Some(expected) = entry.sha256.get(&filename) {
            match verify_sha256(&path, expected) {
                Ok(true) => println!("  {filename}: OK ({size} bytes)"),
                Ok(false) => {
                    println!("  {filename}: SHA256 MISMATCH ({size} bytes)");
                    all_ok = false;
                }
                Err(e) => {
                    println!("  {filename}: ERROR: {e}");
                    all_ok = false;
                }
            }
        } else {
            println!("  {filename}: present ({size} bytes, no SHA256 in manifest yet)");
        }
    }

    if all_ok {
        println!("All {n} shards verified.", n = entry.shards);
    } else {
        println!("Some shards missing or corrupt.");
        process::exit(1);
    }
}
