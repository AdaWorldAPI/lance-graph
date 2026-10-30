//! Fixture B: layer 0's input RMSNorm of the published Jina v5 small
//! (retrieval) ONNX export, Qwen3 architecture, run as a DR#3 graph through
//! the registry. The data is CC-BY-NC-4.0 and never committed; fetch it first,
//! then run the ignored tests:
//!
//! ```text
//! python3 tools/fetch_jina_slice.py
//! cargo test --release --test jina -- --ignored --nocapture
//! ```
//!
//! The tests are `#[ignore]` so a fresh clone reports them as not run instead
//! of passing on missing data.

use dr_math_binding_probe::FrameStore;
use ndarray::{ArrayD, IxDyn};
use prost::Message;
use serde_json::Value as Json;
use tract_harvest_probe::dr3::{tract_facts, OnnxGraph};
use tract_harvest_probe::gather;
use tract_harvest_probe::plan::{resolve, revisions, Plan};
use tract_harvest_probe::registry::{registry, Numeric, Refusal, Registry, K3};
use tract_harvest_probe::source::{sha256_hex, SourceInventory};
use tract_harvest_probe::value::Value;
use tract_onnx::pb;
use tract_onnx::prelude::*;

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures");
const IGNORE: &str = "needs fixtures/jina: python3 tools/fetch_jina_slice.py";
const P: &str = "/layers.0/input_layernorm/";

fn manifest() -> Json {
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

fn name(key: &str) -> String {
    manifest()["slice"][key].as_str().unwrap().to_string()
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

fn plan_of<'r>(bytes: &[u8], reg: &'r Registry) -> (FrameStore, Result<Plan<'r>, Refusal>) {
    let src = SourceInventory::from_bytes(bytes).unwrap();
    let facts = match tract_facts(bytes, &[(0, DatumType::F32, shape())]) {
        Ok(f) => f,
        Err(e) => {
            return (
                FrameStore::new(dr_math_binding_probe::FrameRevisions {
                    source: 0,
                    relation_map: 0,
                    coordinates: 0,
                }),
                Err(e),
            )
        }
    };
    let g = OnnxGraph::build(&src, &facts).unwrap();
    let store = FrameStore::new(revisions(&src, reg, &g));
    let p = resolve(&src, g, reg, &store);
    (store, p)
}

fn input() -> ArrayD<f32> {
    ArrayD::from_shape_vec(IxDyn(&shape()), read_f32("gather_rows.f32")).unwrap()
}

fn f32_out(m: &std::collections::HashMap<String, Value>, k: &str) -> ArrayD<f32> {
    match &m[k] {
        Value::F32(a) => a.clone(),
        other => panic!("{k}: {other:?}"),
    }
}

/// The slice laid onto the tile, every node bound by its op signature, run
/// node by node, compared with ORT literal (the output, and separately the one
/// Bounded step, the mean).
#[test]
#[ignore = "needs fixtures/jina: python3 tools/fetch_jina_slice.py"]
fn rmsnorm_slice_through_the_graph_against_ort() {
    let reg = registry();
    let bytes = slice_bytes();
    let (store, p) = plan_of(&bytes, &reg);
    let p = p.unwrap();
    let steps: Vec<(String, u16)> = p
        .steps
        .iter()
        .map(|s| (s.name.trim_start_matches(P).to_string(), s.key.ordinal))
        .collect();
    eprintln!("schedule: {steps:?}");
    let folded: Vec<&str> = p
        .folded
        .iter()
        .map(|s| s.name.trim_start_matches(P))
        .collect();
    eprintln!("folded: {folded:?}");
    assert_eq!(
        (steps.len(), folded.len()),
        (9, 4),
        "13 source nodes: the 4 Constants fold at resolution, 9 run"
    );
    assert!(p.folded.iter().all(|b| [0, 1, 2].contains(&b.key.ordinal)));
    let cid = |n: &str| steps.iter().find(|(s, _)| s == n).unwrap().1;
    assert_eq!(
        [
            cid("Cast"),
            cid("Pow"),
            cid("ReduceMean"),
            cid("Add"),
            cid("Sqrt"),
            cid("Div"),
            cid("Mul"),
            cid("Cast_1"),
            cid("Mul_1")
        ],
        [3, 11, 12, 5, 10, 8, 7, 3, 7]
    );
    let bounded: Vec<&str> = p
        .steps
        .iter()
        .filter(|s| matches!(s.numeric, Numeric::Bounded(_)))
        .map(|s| s.name.trim_start_matches(P))
        .collect();
    assert_eq!(bounded, ["ReduceMean"], "the only Bounded method");

    let x = input();
    let out = p
        .run(&store, &[(name("input").as_str(), Value::F32(x.clone()))])
        .unwrap();
    let y = f32_out(&out, &name("output"));
    let mean = f32_out(&out, &name("intermediate"));
    assert_eq!(y.shape(), shape().as_slice());

    let ort_out = read_f32("ort_out.f32");
    let ort_mid = read_f32("ort_mid.f32");
    let ort_opt = read_f32("ort_out_opt.f32");
    let c_mid = cmp(mean.as_slice().unwrap(), &ort_mid);
    let c_out = cmp(y.as_slice().unwrap(), &ort_out);
    let c_opt = cmp(y.as_slice().unwrap(), &ort_opt);
    eprintln!(
        "graph mean vs ORT literal: {}/{} exact, max {} ulp",
        c_mid.exact, c_mid.total, c_mid.max_ulp
    );
    eprintln!(
        "graph out  vs ORT literal: {}/{} exact, max {} ulp",
        c_out.exact, c_out.total, c_out.max_ulp
    );
    eprintln!(
        "graph out  vs ORT optimized (report only): {}/{} exact, max {} ulp",
        c_opt.exact, c_opt.total, c_opt.max_ulp
    );

    // The mean against mathematics: f32 squares summed in f64, divided, rounded once.
    let hidden = *shape().last().unwrap();
    let reference: Vec<f32> = x
        .as_slice()
        .unwrap()
        .chunks_exact(hidden)
        .map(|r| (r.iter().map(|v| f64::from(v * v)).sum::<f64>() / hidden as f64) as f32)
        .collect();
    for (arm, mid) in [
        ("graph", mean.as_slice().unwrap()),
        ("ORT literal", &ort_mid[..]),
    ] {
        let c = cmp(mid, &reference);
        eprintln!(
            "{arm} mean vs f64 reference: {}/{} exact, max {} ulp",
            c.exact, c.total, c.max_ulp
        );
    }

    // Measured bound, pinned two-sided-in-spirit: a regression fails here.
    assert!(
        c_mid.max_ulp <= 4,
        "graph mean drifted: {} ulp",
        c_mid.max_ulp
    );
    assert!(
        c_out.max_ulp <= 4,
        "graph RMSNorm drifted: {} ulp",
        c_out.max_ulp
    );
    // Every step after the mean is a single correctly rounded IEEE op, so a
    // row whose mean matches ORT bit for bit must match on every element.
    let mut exact_rows = 0;
    for (row, (m, o)) in mean.iter().zip(&ort_mid).enumerate() {
        if m.to_bits() == o.to_bits() {
            exact_rows += 1;
            let a = &y.as_slice().unwrap()[row * hidden..(row + 1) * hidden];
            let b = &ort_out[row * hidden..(row + 1) * hidden];
            assert_eq!(
                cmp(a, b).exact,
                hidden,
                "row {row}: exact mean but inexact output"
            );
        }
    }
    eprintln!("rows with a bit-exact mean: {exact_rows}/{}", mean.len());
}

/// `Gather` resolved from the source placement equals the byte ranges the
/// fetch tool pinned; out-of-range and negative indices follow the ONNX rule.
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
        assert_eq!(
            sha256_hex(&rows[i * 4096..(i + 1) * 4096]),
            want["sha256"].as_str().unwrap()
        );
    }
    let vocab = 151_936i64;
    for bad in [vocab, -vocab - 1] {
        assert!(matches!(
            gather::row_ranges(&src, "embed_tokens.weight", &[bad]),
            Err(Refusal::Runtime { .. })
        ));
    }
    assert_eq!(
        gather::row_ranges(&src, "embed_tokens.weight", &[-1]).unwrap()[0].row,
        vocab as u64 - 1
    );
}

// ── negative controls: mutate the pinned slice in its protobuf ──────────────

fn mutate(f: impl FnOnce(&mut pb::ModelProto)) -> Vec<u8> {
    let mut m = pb::ModelProto::decode(&*slice_bytes()).unwrap();
    f(&mut m);
    m.encode_to_vec()
}

fn node_named<'a>(m: &'a mut pb::ModelProto, n: &str) -> &'a mut pb::NodeProto {
    m.graph
        .as_mut()
        .unwrap()
        .node
        .iter_mut()
        .find(|x| x.name == format!("{P}{n}"))
        .unwrap()
}

/// `Pow(x, 2.0000002)`: the Pow registration's WHEN reads the constant
/// exponent from the source and is FALSE.
#[test]
#[ignore = "needs fixtures/jina: python3 tools/fetch_jina_slice.py"]
fn pow_exponent_other_than_two_is_when_false() {
    let b = mutate(|m| {
        let t = node_named(m, "Constant").attribute[0].t.as_mut().unwrap();
        t.raw_data = 2.000_000_2f32.to_le_bytes().to_vec();
        t.float_data.clear();
    });
    match plan_of(&b, &registry()).1 {
        Err(Refusal::When {
            node, key, truth, ..
        }) => {
            assert_eq!(
                (node.trim_start_matches(P), key.ordinal, truth),
                ("Pow", 11, K3::False)
            )
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// The mean over axis -2 instead of -1 is a different, valid program. Nothing
/// assumed the RMSNorm pattern, so the graph runs the program as written: the
/// mean has the shape the source axis gives.
#[test]
#[ignore = "needs fixtures/jina: python3 tools/fetch_jina_slice.py"]
fn another_reduction_axis_is_a_different_program_not_a_refusal() {
    let b = mutate(|m| {
        let t = node_named(m, "Constant_1").attribute[0].t.as_mut().unwrap();
        t.raw_data = (-2i64).to_le_bytes().to_vec();
        t.int64_data.clear();
    });
    let reg = registry();
    let (store, p) = plan_of(&b, &reg);
    let out = p
        .unwrap()
        .run(&store, &[(name("input").as_str(), Value::F32(input()))])
        .unwrap();
    let sh = shape();
    assert_eq!(
        f32_out(&out, &name("intermediate")).shape(),
        &[sh[0], 1, sh[2]]
    );
    assert_eq!(f32_out(&out, &name("output")).shape(), sh.as_slice());
}

/// `Cast_1` to FLOAT16: the signature `Cast{to=10}` is unregistered.
#[test]
#[ignore = "needs fixtures/jina: python3 tools/fetch_jina_slice.py"]
fn cast_to_half_is_unregistered() {
    let b = mutate(|m| node_named(m, "Cast_1").attribute[0].i = 10);
    match plan_of(&b, &registry()).1 {
        Err(Refusal::Unregistered { node, signature }) => {
            assert!(node.ends_with("Cast_1"), "{node}");
            assert!(signature.contains("to=10"), "{signature}");
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// The first Cast widened to f64 (and the constants with it): `Cast{to=11}` is
/// unregistered, and it is the first node of the schedule to fail.
#[test]
#[ignore = "needs fixtures/jina: python3 tools/fetch_jina_slice.py"]
fn cast_to_double_is_unregistered() {
    let b = mutate(|m| {
        node_named(m, "Cast").attribute[0].i = 11;
        for c in ["Constant", "Constant_2", "Constant_3"] {
            let t = node_named(m, c).attribute[0].t.as_mut().unwrap();
            let v = f32::from_le_bytes(t.raw_data[..4].try_into().unwrap());
            t.data_type = 11;
            t.raw_data = f64::from(v).to_le_bytes().to_vec();
        }
    });
    match plan_of(&b, &registry()).1 {
        Err(Refusal::Unregistered { node, signature }) => {
            assert!(node.ends_with("/Cast"), "{node}");
            assert!(signature.contains("to=11"), "{signature}");
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// One bit of one weight changed: a new source generation. The plan sealed for
/// the pinned slice refuses to run once the store advances to it.
#[test]
#[ignore = "needs fixtures/jina: python3 tools/fetch_jina_slice.py"]
fn new_weight_generation_refuses_the_old_plan() {
    let reg = registry();
    let (mut store, p) = plan_of(&slice_bytes(), &reg);
    let p = p.unwrap();
    let b = mutate(|m| {
        let w = m
            .graph
            .as_mut()
            .unwrap()
            .initializer
            .iter_mut()
            .find(|t| t.name == "layers.0.input_layernorm.weight")
            .unwrap();
        w.raw_data[0] ^= 1;
    });
    let src2 = SourceInventory::from_bytes(&b).unwrap();
    let g2 = OnnxGraph::build(
        &src2,
        &tract_facts(&b, &[(0, DatumType::F32, shape())]).unwrap(),
    )
    .unwrap();
    store.advance(revisions(&src2, &reg, &g2));
    match p.run(&store, &[(name("input").as_str(), Value::F32(input()))]) {
        Err(Refusal::Frame(dr_math_binding_probe::Refusal::SourceChanged)) => {}
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// The full graph (inputs pinned to [1, 10]): the inference stage labels all
/// 7,620 source values with a full fact; the typed stage labels 2,546, of
/// which 1,988 are admitted (a typed fact is admitted only from the outlet
/// that is the value's source producer, `dr3::admit`); the two agree wherever
/// both speak. Registry coverage with these facts, cold folding included,
/// pinned exactly so any change forces a re-measure. The one external-data
/// Constant (the rotary `inv_freq`) is inlined from its pinned range.
#[test]
#[ignore = "needs fixtures/jina: python3 tools/fetch_jina_slice.py && python3 tools/fetch_jina_constants.py"]
fn full_graph_facts_and_coverage() {
    use tract_harvest_probe::dr3::tract_fact_sources_path;
    use tract_harvest_probe::plan::coverage;
    let path = std::path::PathBuf::from(format!("{FIX}/jina/onnx_model.onnx"));
    let mut src = SourceInventory::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    let inlined = src
        .inline_constants_from_manifest(
            std::path::Path::new(&format!("{FIX}/jina_constants.json")),
            std::path::Path::new(&format!("{FIX}/jina")),
        )
        .unwrap();
    assert_eq!(inlined, 1);
    let pins: Vec<_> = (0..2).map(|i| (i, DatumType::I64, vec![1, 10])).collect();
    let fs = tract_fact_sources_path(&path, &pins).unwrap();
    let values: Vec<&String> = src
        .nodes
        .iter()
        .flat_map(|n| &n.outputs)
        .filter(|o| !o.is_empty())
        .collect();
    let full = |m: &std::collections::HashMap<String, tract_harvest_probe::registry::Fact>| {
        values
            .iter()
            .filter(|o| {
                m.get(o.as_str())
                    .is_some_and(|f| f.dtype.is_some() && f.shape.is_some())
            })
            .count()
    };
    assert_eq!(values.len(), 7620);
    assert_eq!(full(&fs.inference), 7620);
    assert_eq!(full(&fs.typed), 1988);
    assert!(
        fs.rejected.iter().all(|r| r.stage == "typed"),
        "no inference fact is rejected"
    );
    assert_eq!(fs.rejected.len(), 558);
    assert!(
        fs.disagreements().is_empty(),
        "{:?}",
        &fs.disagreements()[..fs.disagreements().len().min(5)]
    );
    let reg = registry();
    let tot = |m| {
        coverage(&src, m, &reg)
            .values()
            .fold([0; 3], |a, c| [a[0] + c[0], a[1] + c[1], a[2] + c[2]])
    };
    // Typed facts alone leave shapes unknown, so some guards stay UNKNOWN.
    assert_eq!(tot(&fs.typed), [4384, 286, 2950]);
    // With the merged facts every node is registered and WHEN TRUE (was
    // 7,274 / 0 / 346 before the fields of plan §0.10).
    assert_eq!(tot(&fs.merged()), [7620, 0, 0]);
    // The same graph as a bind-store consumer: every registered node has a
    // route to its codebook item and resolves through it; the rest have none.
    let routes = tract_harvest_probe::bind::route(&src, &fs.merged(), &reg);
    let resolved = (0..src.nodes.len())
        .filter(|&n| tract_harvest_probe::bind::resolve(&routes, &reg, n).is_ok())
        .count();
    assert_eq!(
        (
            routes.map.len(),
            resolved,
            src.nodes.len() - routes.map.len()
        ),
        (7620, 7620, 0)
    );
    // DR64 is global to local: the whole graph lays over tiles, `dep`
    // composed tile by tile is the source dataflow, and it schedules.
    let merged = fs.merged();
    let g = OnnxGraph::build(&src, &merged).unwrap();
    assert_eq!(g.nodes, 7620);
    let (dep, _) = g.dep();
    assert_eq!(dep, g.dep_from_edges());
    assert!(dep.tiles.len() > 100, "{} tiles", dep.tiles.len());
    let order = g.schedule().unwrap();
    let pos: Vec<usize> = {
        let mut p = vec![0; order.len()];
        for (k, &n) in order.iter().enumerate() {
            p[n] = k;
        }
        p
    };
    assert!(dep.pairs().all(|(i, j)| pos[i] < pos[j]));
    println!(
        "jina tiles: {} props, {} dep tiles, {} dep edges",
        g.props.len(),
        dep.tiles.len(),
        dep.pairs().count()
    );
}

/// The real Jina rotary embedding, folded cold at `[1, 10]` from the source
/// and the pinned `inv_freq` range, against ORT (`ORT_DISABLE_ALL`) running
/// the same 40 nodes extracted from the graph (`tools/fetch_jina_constants.py`;
/// its outputs are gitignored, their digests committed).
#[test]
#[ignore = "needs fixtures/jina: python3 tools/fetch_jina_slice.py && python3 tools/fetch_jina_constants.py"]
fn jina_rotary_fold_matches_ort() {
    use tract_harvest_probe::dr3::tract_fact_sources_path;
    use tract_harvest_probe::fold::fold;
    let path = std::path::PathBuf::from(format!("{FIX}/jina/onnx_model.onnx"));
    let mut src = SourceInventory::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    let manifest = std::path::PathBuf::from(format!("{FIX}/jina_constants.json"));
    src.inline_constants_from_manifest(&manifest, std::path::Path::new(&format!("{FIX}/jina")))
        .unwrap();
    let pins: Vec<_> = (0..2).map(|i| (i, DatumType::I64, vec![1, 10])).collect();
    let fs = tract_fact_sources_path(&path, &pins).unwrap();
    let f = fold(&src, &fs.merged(), &registry());
    assert!(f.refused.is_empty(), "{:?}", f.refused);
    let m: Json = serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    let mut worst = 0;
    for o in m["rotary"]["outputs"].as_array().unwrap() {
        let name = o["name"].as_str().unwrap();
        let bytes = std::fs::read(format!("{FIX}/jina/{}", o["file"].as_str().unwrap())).unwrap();
        assert_eq!(sha256_hex(&bytes), o["sha256"].as_str().unwrap(), "{name}");
        let want: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let got = match f.values.get(name) {
            Some(Value::F32(a)) => a,
            other => panic!("{name} did not fold: {other:?}"),
        };
        let shape: Vec<usize> = o["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(got.shape(), shape.as_slice(), "{name}");
        let d = got
            .iter()
            .zip(&want)
            .map(|(&g, &w)| ulp(g, w))
            .max()
            .unwrap();
        eprintln!("{name}: {} values, max ULP vs ORT {d}", want.len());
        worst = worst.max(d);
    }
    assert!(worst <= 1, "{worst} ULP");
}
