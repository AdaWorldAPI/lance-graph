//! neural-scan — CLI tool that scans the graph engine stack and outputs diagnosis JSON.
//!
//! Usage:
//!   neural-scan [--repos <path1:name1,path2:name2,...>] [--output <path>]
//!
//! Default: scans sibling directories (lance-graph, ndarray, q2, aiwar-neo4j-harvest)

use neural_debug::scanner::{scan_stack, ScanConfig};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut config = ScanConfig::default();
    let mut output_path: Option<PathBuf> = None;

    // Parse args
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--repos" => {
                i += 1;
                if i < args.len() {
                    for pair in args[i].split(',') {
                        let parts: Vec<&str> = pair.split(':').collect();
                        if parts.len() == 2 {
                            config.repos.push((
                                parts[0].to_string(),
                                PathBuf::from(parts[1]),
                            ));
                        }
                    }
                }
            }
            "--output" | "-o" => {
                i += 1;
                if i < args.len() {
                    output_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--help" | "-h" => {
                eprintln!("neural-scan — scan Rust source files for dead/stub/NaN functions");
                eprintln!();
                eprintln!("Usage: neural-scan [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --repos <name:path,...>  Repos to scan (default: auto-detect siblings)");
                eprintln!("  --output, -o <path>      Output JSON file (default: stdout)");
                eprintln!("  --help, -h               Show this help");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Auto-detect repos if none specified
    if config.repos.is_empty() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent() // crates/
            .and_then(|p| p.parent()) // lance-graph/
            .and_then(|p| p.parent()) // workspace root
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));

        let candidates = [
            ("lance-graph", "lance-graph"),
            ("ndarray", "ndarray"),
            ("q2", "q2"),
            ("aiwar-neo4j-harvest", "aiwar-neo4j-harvest"),
            ("neo4j-rs", "neo4j-rs"),
        ];

        for (name, dir) in &candidates {
            let path = base.join(dir);
            if path.exists() {
                eprintln!("Found repo: {} at {}", name, path.display());
                config.repos.push((name.to_string(), path));
            }
        }

        if config.repos.is_empty() {
            // Try current directory structure
            let cwd = std::env::current_dir().unwrap_or_default();
            let parent = cwd.parent().unwrap_or(&cwd);
            for (name, dir) in &candidates {
                let path = parent.join(dir);
                if path.exists() {
                    eprintln!("Found repo: {} at {}", name, path.display());
                    config.repos.push((name.to_string(), path));
                }
            }
        }
    }

    if config.repos.is_empty() {
        eprintln!("No repos found. Use --repos to specify paths.");
        std::process::exit(1);
    }

    eprintln!(
        "Scanning {} repos...",
        config.repos.len()
    );

    let diagnosis = scan_stack(&config);

    // Print summary to stderr
    eprintln!();
    eprintln!("=== NEURAL SCAN RESULTS ===");
    eprintln!(
        "Total: {} functions across {} repos",
        diagnosis.total_functions,
        diagnosis.repos.len()
    );
    eprintln!(
        "Dead: {} | Stub: {} | NaN risk: {} | Health: {:.1}%",
        diagnosis.total_dead,
        diagnosis.total_stub,
        diagnosis.total_nan_risk,
        diagnosis.health_pct
    );
    eprintln!("Scan time: {}ms", diagnosis.scan_duration_ms);
    eprintln!();

    for repo in &diagnosis.repos {
        eprintln!(
            "  {} — {} functions, {:.0}% health, {} dead, {} stub",
            repo.name,
            repo.total_functions,
            repo.health_pct,
            repo.total_dead,
            repo.total_stub,
        );
        for module in &repo.modules {
            if module.dead > 0 || module.nan_risk > 0 {
                eprintln!(
                    "    {} — {} dead, {} nan_risk",
                    module.name, module.dead, module.nan_risk
                );
            }
        }
    }

    // Output JSON
    let json = serde_json::to_string_pretty(&diagnosis).unwrap();

    if let Some(path) = output_path {
        std::fs::write(&path, &json).unwrap();
        eprintln!("\nWrote diagnosis to {}", path.display());
    } else {
        println!("{}", json);
    }
}
