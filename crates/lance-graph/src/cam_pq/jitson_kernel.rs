//! Jitson template for compiled ADC scan kernels.
//!
//! Generates a JITSON JSON config that ndarray's jitson engine can compile
//! into a tight scan loop. The compiled kernel runs the stroke cascade
//! without branch misprediction or virtual dispatch.
//!
//! # Pipeline
//!
//! ```text
//! Stage 1: LOAD_HEEL      — read stroke1 array (1 byte/candidate)
//! Stage 2: GATHER_HEEL     — table[0][heel_byte] → partial distance
//! Stage 3: FILTER_HEEL     — reject if partial > heel_threshold
//! Stage 4: LOAD_BRANCH     — read stroke2 array (2 bytes/survivor)
//! Stage 5: GATHER_BRANCH   — + table[1][branch_byte]
//! Stage 6: FILTER_BRANCH   — reject if partial > branch_threshold
//! Stage 7: LOAD_FULL       — read stroke3 array (6 bytes/finalist)
//! Stage 8: GATHER_FULL     — + table[2..5] lookups
//! Stage 9: TOP_K           — maintain heap of top-K results
//! ```
//!
//! The AVX-512 version processes 16 candidates per iteration at each stage.

/// Generate a JITSON template for CAM-PQ stroke cascade scan.
///
/// This returns a JSON string that can be fed to ndarray's
/// `jitson::template::from_json()` to produce a `JitsonTemplate`,
/// which in turn feeds the Cranelift JIT compiler.
pub fn cam_pq_cascade_template(
    heel_threshold: f32,
    branch_threshold: f32,
    top_k: usize,
    lance_table: &str,
) -> String {
    format!(
        r#"{{
  "kernel": "cam_pq_cascade",
  "scan": {{
    "name": "cam_pq_cascade_scan",
    "threshold": {},
    "branch_threshold": {},
    "top_k": {},
    "batch_size": 65536,
    "distance_type": "adc_l2"
  }},
  "pipeline": [
    {{
      "stage": "load_heel",
      "avx512_instr": "VMOVDQU8",
      "fallback": "memcpy",
      "backend": "lancedb",
      "backend_key": "{}.stroke1"
    }},
    {{
      "stage": "gather_heel",
      "avx512_instr": "VPGATHERDD",
      "fallback": "scalar_lookup"
    }},
    {{
      "stage": "filter_heel",
      "avx512_instr": "VCMPPS+KMOV",
      "fallback": "scalar_cmp"
    }},
    {{
      "stage": "load_branch",
      "avx512_instr": "VMOVDQU8",
      "fallback": "memcpy",
      "backend": "lancedb",
      "backend_key": "{}.stroke2"
    }},
    {{
      "stage": "gather_branch",
      "avx512_instr": "VPGATHERDD",
      "fallback": "scalar_lookup"
    }},
    {{
      "stage": "filter_branch",
      "avx512_instr": "VCMPPS+KMOV",
      "fallback": "scalar_cmp"
    }},
    {{
      "stage": "load_full",
      "avx512_instr": "VMOVDQU8",
      "fallback": "memcpy",
      "backend": "lancedb",
      "backend_key": "{}.stroke3"
    }},
    {{
      "stage": "gather_full",
      "avx512_instr": "VPGATHERPS",
      "fallback": "scalar_lookup_6x"
    }},
    {{
      "stage": "top_k",
      "avx512_instr": null,
      "fallback": "heap_insert"
    }}
  ],
  "features": [
    ["avx512f", true],
    ["avx512bw", true],
    ["cam_pq", true],
    ["stroke_cascade", true]
  ],
  "backends": [
    {{
      "name": "lancedb",
      "uri": "{}",
      "table": "{}"
    }}
  ],
  "cranelift_preset": "speed",
  "cranelift_opt_level": "speed"
}}"#,
        heel_threshold,
        branch_threshold,
        top_k,
        lance_table,
        lance_table,
        lance_table,
        lance_table,
        lance_table,
    )
}

/// Generate a JITSON template for full ADC scan (no cascade, brute force).
///
/// Used when the dataset is small enough that stroke cascade overhead
/// exceeds brute force ADC.
pub fn cam_pq_full_adc_template(top_k: usize, lance_table: &str) -> String {
    format!(
        r#"{{
  "kernel": "cam_pq_full_adc",
  "scan": {{
    "name": "cam_pq_full_adc_scan",
    "threshold": 1e30,
    "top_k": {},
    "batch_size": 65536,
    "distance_type": "adc_l2"
  }},
  "pipeline": [
    {{
      "stage": "load_cam",
      "avx512_instr": "VMOVDQU8",
      "fallback": "memcpy",
      "backend": "lancedb",
      "backend_key": "{}.cam"
    }},
    {{
      "stage": "gather_6x",
      "avx512_instr": "VPGATHERPS",
      "fallback": "scalar_lookup_6x"
    }},
    {{
      "stage": "accumulate",
      "avx512_instr": "VADDPS",
      "fallback": "scalar_add"
    }},
    {{
      "stage": "top_k",
      "avx512_instr": null,
      "fallback": "heap_insert"
    }}
  ],
  "features": [
    ["avx512f", true],
    ["cam_pq", true]
  ],
  "backends": [
    {{
      "name": "lancedb",
      "uri": "{}",
      "table": "{}"
    }}
  ],
  "cranelift_preset": "speed",
  "cranelift_opt_level": "speed"
}}"#,
        top_k, lance_table, lance_table, lance_table,
    )
}

/// Kernel selection: choose cascade vs full ADC based on dataset size.
///
/// Below `cascade_threshold` candidates, full ADC is faster because
/// the cascade filter/branch overhead exceeds the savings.
pub fn select_kernel_template(
    num_candidates: u64,
    heel_threshold: f32,
    branch_threshold: f32,
    top_k: usize,
    lance_table: &str,
) -> String {
    const CASCADE_THRESHOLD: u64 = 100_000;

    if num_candidates >= CASCADE_THRESHOLD {
        cam_pq_cascade_template(heel_threshold, branch_threshold, top_k, lance_table)
    } else {
        cam_pq_full_adc_template(top_k, lance_table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cascade_template_valid_json() {
        let json = cam_pq_cascade_template(50.0, 25.0, 10, "vectors");
        // Should be valid JSON
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("cascade template should be valid JSON");
        assert_eq!(parsed["kernel"], "cam_pq_cascade");
        assert_eq!(parsed["pipeline"].as_array().unwrap().len(), 9);
        assert_eq!(parsed["scan"]["top_k"], 10);
    }

    #[test]
    fn test_full_adc_template_valid_json() {
        let json = cam_pq_full_adc_template(20, "my_table");
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("full ADC template should be valid JSON");
        assert_eq!(parsed["kernel"], "cam_pq_full_adc");
        assert_eq!(parsed["pipeline"].as_array().unwrap().len(), 4);
    }

    #[test]
    fn test_kernel_selection() {
        // Small dataset → full ADC
        let small = select_kernel_template(50_000, 50.0, 25.0, 10, "vecs");
        assert!(small.contains("cam_pq_full_adc"));

        // Large dataset → cascade
        let large = select_kernel_template(1_000_000, 50.0, 25.0, 10, "vecs");
        assert!(large.contains("cam_pq_cascade"));

        // Boundary
        let boundary = select_kernel_template(100_000, 50.0, 25.0, 10, "vecs");
        assert!(boundary.contains("cam_pq_cascade"));
    }
}
