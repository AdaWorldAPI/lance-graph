//! Fixture B: layer 0's input RMSNorm of the published Jina v5 small
//! (retrieval) ONNX export, Qwen3 architecture. The data is CC-BY-NC-4.0 and
//! never committed; fetch it first, then run the ignored tests:
//!
//! ```text
//! python3 tools/fetch_jina_slice.py
//! cargo test --release --test jina -- --ignored --nocapture
//! ```
//!
//! The tests are `#[ignore]` so a fresh clone reports them as not run instead
//! of passing on missing data.

use ndarray::{ArrayD, IxDyn};
use prost::Message;
use serde_json::Value;
use tract_harvest_probe::admit::{self, Numeric, Refusal};
use tract_harvest_probe::gather;
use tract_harvest_probe::source::{sha256_hex, SourceInventory};
use tract_onnx::pb;
use tract_onnx::prelude::*;

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures");
const IGNORE: &str = "needs fixtures/jina: python3 tools/fetch_jina_slice.py";

fn manifest() -> Value {
    serde_json::from_str(&std::fs::read_to_string(format!("{FIX}/jina_manifest.json")).unwrap())
        .unwrap()
}

fn read_f32(name: &str) -> Vec<f32> {
    let b = std::fs::read(format!("{FIX}/jina/{name}"))
        .unwrap_or_else(|e| panic!("{name}: {e} ({IGNORE})"));
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn slice_bytes() -> Vec<u8> {
    let b = std::fs::read(format!("{FIX}/jina/slice.onnx")).unwrap();
    assert_eq!(
        sha256_hex(&b),
        manifest()["slice"]["sha256"].as_str().unwrap(),
        "slice is the pinned one"
    );
    b
}

fn shape() -> Vec<usize> {
    manifest()["slice"]["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}

/// Distance in units in the last place between two finite f32 of either sign.
fn ulp(a: f32, b: f32) -> u32 {
    let key = |v: f32| {
        let i = v.to_bits() as i32;
        if i < 0 {
            i32::MIN.wrapping_sub(i)
        } else {
            i
        }
    };
    (key(a) as i64 - key(b) as i64).unsigned_abs() as u32
}

struct Cmp {
    exact: usize,
    total: usize,
    max_ulp: u32,
}

fn cmp(a: &[f32], b: &[f32]) -> Cmp {
    assert_eq!(a.len(), b.len());
    Cmp {
        exact: a
            .iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() == y.to_bits())
            .count(),
        total: a.len(),
        max_ulp: a.iter().zip(b).map(|(x, y)| ulp(*x, *y)).max().unwrap_or(0),
    }
}

fn admit_slice(bytes: &[u8]) -> Result<admit::RmsNormRecipe, Refusal> {
    let src = SourceInventory::from_bytes(bytes).unwrap();
    admit::admit_rmsnorm(&src, &admit::typed(bytes)?)
}

/// The slice is admitted with its 9 compute nodes named as in the source,
/// executed natively, and compared with ORT: the output and, separately, the
/// one bounded step (the mean). Tract's typed graph and its decluttered graph
/// (where Pow..Mul collapses into one fused `RmsNorm` op) are run as
/// comparison arms only.
#[test]
#[ignore = "needs fixtures/jina: python3 tools/fetch_jina_slice.py"]
fn rmsnorm_slice_against_ort() {
    let bytes = slice_bytes();
    let r = admit_slice(&bytes).unwrap();
    assert!(matches!(r.receipt.numeric, Numeric::Bounded { .. }));
    let p = "/layers.0/input_layernorm/";
    let want: Vec<String> = [
        "Cast",
        "Pow",
        "ReduceMean",
        "Add",
        "Sqrt",
        "Div",
        "Mul",
        "Cast_1",
        "Mul_1",
    ]
    .iter()
    .map(|n| format!("{p}{n}"))
    .collect();
    assert_eq!(r.receipt.source_nodes, want);
    assert_eq!(r.eps.to_bits(), 1e-6f32.to_bits());
    assert_eq!(r.weight.len(), 1024);

    let sh = shape();
    let x = ArrayD::from_shape_vec(IxDyn(&sh), read_f32("gather_rows.f32")).unwrap();
    let (out, mean) = r.run(&x).unwrap();
    // `ort_*.f32` is ORT with its graph optimizer off (each source node as
    // written); `ort_*_opt.f32` is ORT's default, which rewrites the pattern.
    let ort_out = read_f32("ort_out.f32");
    let ort_mid = read_f32("ort_mid.f32");
    let ort_opt = read_f32("ort_out_opt.f32");
    let c_opt = cmp(&read_f32("ort_out.f32"), &ort_opt);
    eprintln!(
        "ORT literal vs ORT optimized: {}/{} exact, max {} ulp",
        c_opt.exact, c_opt.total, c_opt.max_ulp
    );
    let c_mid = cmp(mean.as_slice().unwrap(), &ort_mid);
    let c_out = cmp(out.as_slice().unwrap(), &ort_out);
    eprintln!(
        "native mean vs ORT: {}/{} exact, max {} ulp",
        c_mid.exact, c_mid.total, c_mid.max_ulp
    );
    eprintln!(
        "native out  vs ORT: {}/{} exact, max {} ulp",
        c_out.exact, c_out.total, c_out.max_ulp
    );
    let c_nopt = cmp(out.as_slice().unwrap(), &ort_opt);
    eprintln!(
        "native out  vs ORT optimized: {}/{} exact, max {} ulp",
        c_nopt.exact, c_nopt.total, c_nopt.max_ulp
    );

    // Tract arms (oracles, not the native path).
    let run_tract = |declutter: bool| -> (Vec<f32>, Vec<f32>) {
        let m = admit::typed(&bytes).unwrap();
        let m = if declutter {
            m.into_decluttered().unwrap()
        } else {
            m
        };
        let xt = x.clone().into_tensor();
        let o = m.into_runnable().unwrap().run(tvec![xt.into()]).unwrap();
        let v = |i: usize| {
            o[i].to_plain_array_view::<f32>()
                .unwrap()
                .iter()
                .copied()
                .collect()
        };
        (v(0), v(1))
    };
    let (ty_out, ty_mid) = run_tract(false);
    let (de_out, de_mid) = run_tract(true);
    let c_ty = cmp(&ty_out, &ort_out);
    let c_de = cmp(&de_out, &ort_out);
    // Reference mean: the f32 squares summed in f64 (exact to ~1e-16 relative
    // for 1024 terms), divided, rounded once to f32. That is the correctly
    // rounded mean of the squares unless a row sits within ~1e-16 of a
    // rounding boundary; it scores each arm against mathematics, not ORT.
    let hidden = sh[sh.len() - 1];
    let reference: Vec<f32> = x
        .as_slice()
        .unwrap()
        .chunks_exact(hidden)
        .map(|r| (r.iter().map(|v| f64::from(v * v)).sum::<f64>() / hidden as f64) as f32)
        .collect();
    let mut vs_ref = Vec::new();
    for (arm, mid) in [
        ("native", mean.as_slice().unwrap()),
        ("ORT literal", &ort_mid[..]),
        ("tract typed", &ty_mid[..]),
    ] {
        let c = cmp(mid, &reference);
        eprintln!(
            "{arm} mean vs f64 reference: {}/{} exact, max {} ulp",
            c.exact, c.total, c.max_ulp
        );
        vs_ref.push(c.max_ulp);
    }
    assert!(
        vs_ref[0] <= 4,
        "native mean vs the mathematical mean: {} ulp",
        vs_ref[0]
    );
    for (arm, mid) in [("typed", &ty_mid), ("decluttered", &de_mid)] {
        let c = cmp(mid, &ort_mid);
        eprintln!(
            "tract {arm} mean vs ORT: {}/{} exact, max {} ulp",
            c.exact, c.total, c.max_ulp
        );
    }
    eprintln!(
        "tract typed       vs ORT: {}/{} exact, max {} ulp",
        c_ty.exact, c_ty.total, c_ty.max_ulp
    );
    eprintln!(
        "tract decluttered vs ORT: {}/{} exact, max {} ulp",
        c_de.exact, c_de.total, c_de.max_ulp
    );

    // The measured bound, pinned two-sided: a regression or an unexplained
    // improvement both fail and force a re-measure.
    assert!(
        c_out.max_ulp <= 4,
        "native RMSNorm drifted: {} ulp",
        c_out.max_ulp
    );
    assert!(
        c_mid.max_ulp <= 4,
        "native mean drifted: {} ulp",
        c_mid.max_ulp
    );
    // Every step after the mean is a single correctly rounded IEEE op, so an
    // exact mean must give an exact output: check that on rows whose mean
    // matched ORT bit for bit.
    for (row, (m, o)) in mean.iter().zip(&ort_mid).enumerate() {
        if m.to_bits() == o.to_bits() {
            let a = &out.as_slice().unwrap()[row * hidden..(row + 1) * hidden];
            let b = &ort_out[row * hidden..(row + 1) * hidden];
            assert_eq!(
                cmp(a, b).exact,
                hidden,
                "row {row}: exact mean but inexact output"
            );
        }
    }
}

/// `Gather` resolved from the source placement equals the byte ranges the
/// fetch tool pinned, and the fetched rows hash as recorded. Out-of-range and
/// negative indices follow the ONNX rule.
#[test]
#[ignore = "needs fixtures/jina: python3 tools/fetch_jina_slice.py"]
fn gather_rows_from_source_placement() {
    let m = manifest();
    let model = std::fs::read(format!("{FIX}/jina/onnx_model.onnx")).unwrap();
    assert_eq!(
        sha256_hex(&model),
        m["pins"]["onnx/model.onnx"].as_str().unwrap()
    );
    let src = SourceInventory::from_bytes(&model).unwrap();
    let ids: Vec<i64> = m["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let ranges = gather::row_ranges(&src, "embed_tokens.weight", &ids).unwrap();
    let rows = std::fs::read(format!("{FIX}/jina/gather_rows.f32")).unwrap();
    for (i, (r, want)) in ranges
        .iter()
        .zip(m["embedding"]["ranges"].as_array().unwrap())
        .enumerate()
    {
        assert_eq!(r.offset, want["offset"].as_u64().unwrap());
        assert_eq!(r.length, 4096);
        let chunk = &rows[i * 4096..(i + 1) * 4096];
        assert_eq!(sha256_hex(chunk), want["sha256"].as_str().unwrap());
    }
    let vocab = 151_936i64;
    assert!(matches!(
        gather::row_ranges(&src, "embed_tokens.weight", &[vocab]),
        Err(Refusal::RewriteAssumption { .. })
    ));
    assert!(matches!(
        gather::row_ranges(&src, "embed_tokens.weight", &[-vocab - 1]),
        Err(Refusal::RewriteAssumption { .. })
    ));
    let last = gather::row_ranges(&src, "embed_tokens.weight", &[-1]).unwrap()[0];
    assert_eq!(last.row, vocab as u64 - 1);
}

// ── negative controls: mutate the pinned slice in its protobuf ──────────────

fn mutate(f: impl FnOnce(&mut pb::ModelProto)) -> Vec<u8> {
    let mut m = pb::ModelProto::decode(&*slice_bytes()).unwrap();
    f(&mut m);
    m.encode_to_vec()
}

fn constant_named<'a>(m: &'a mut pb::ModelProto, name: &str) -> &'a mut pb::TensorProto {
    let g = m.graph.as_mut().unwrap();
    let n = g.node.iter_mut().find(|n| n.name == name).unwrap();
    n.attribute[0].t.as_mut().unwrap()
}

fn set_f32(t: &mut pb::TensorProto, v: f32) {
    t.raw_data = v.to_le_bytes().to_vec();
    t.float_data.clear();
}

#[test]
#[ignore = "needs fixtures/jina: python3 tools/fetch_jina_slice.py"]
fn pow_exponent_other_than_two_is_refused() {
    let b = mutate(|m| {
        set_f32(
            constant_named(m, "/layers.0/input_layernorm/Constant"),
            2.000_000_2,
        )
    });
    match admit_slice(&b) {
        Err(Refusal::RewriteAssumption { node, .. }) => assert!(node.ends_with("/Pow"), "{node}"),
        other => panic!("{other:?}"),
    }
}

#[test]
#[ignore = "needs fixtures/jina: python3 tools/fetch_jina_slice.py"]
fn reduction_over_another_axis_is_refused() {
    let b = mutate(|m| {
        let t = constant_named(m, "/layers.0/input_layernorm/Constant_1");
        t.raw_data = (-2i64).to_le_bytes().to_vec();
        t.int64_data.clear();
    });
    assert!(
        matches!(
            admit_slice(&b),
            Err(Refusal::Pattern { .. }) | Err(Refusal::Frontend(_))
        ),
        "{:?}",
        admit_slice(&b)
    );
}

#[test]
#[ignore = "needs fixtures/jina: python3 tools/fetch_jina_slice.py"]
fn cast_to_half_precision_is_refused() {
    let b = mutate(|m| {
        let g = m.graph.as_mut().unwrap();
        let n = g
            .node
            .iter_mut()
            .find(|n| n.name == "/layers.0/input_layernorm/Cast_1")
            .unwrap();
        n.attribute[0].i = 10; // FLOAT16: a change of rounding the recipe does not model
    });
    // Tract types the f16 cast and inserts its own cast back to f32 for the
    // final Mul, so the refusal is either the dtype check or the pattern
    // breaking at Cast_1. Either way it names Cast_1 and nothing runs.
    match admit_slice(&b) {
        Err(Refusal::Dtype { node, .. }) | Err(Refusal::Pattern { node, .. }) => {
            assert!(node.ends_with("Cast_1"), "{node}")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
#[ignore = "needs fixtures/jina: python3 tools/fetch_jina_slice.py"]
fn new_weight_generation_is_refused_by_the_old_recipe() {
    let r = admit_slice(&slice_bytes()).unwrap();
    let b = mutate(|m| {
        let w = m
            .graph
            .as_mut()
            .unwrap()
            .initializer
            .iter_mut()
            .find(|t| t.name == "layers.0.input_layernorm.weight")
            .unwrap();
        w.raw_data[0] ^= 1; // one bit of one weight
    });
    match r
        .receipt
        .check_generation(&SourceInventory::from_bytes(&b).unwrap())
    {
        Err(Refusal::StaleGeneration { initializer, .. }) => {
            assert_eq!(initializer, "layers.0.input_layernorm.weight")
        }
        other => panic!("{other:?}"),
    }
}
