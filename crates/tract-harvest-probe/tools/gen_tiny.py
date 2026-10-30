#!/usr/bin/env python3
"""D-TRACT-HARVEST-0 fixture A: a tiny ONNX graph plus its negative variants,
and the ONNX Runtime outputs for one pinned input set.

ORT is the independent oracle. The Rust probe never calls it; it reads the
outputs written here. Run from the crate root:

    python3 tools/gen_tiny.py

Outputs go to fixtures/tiny/: one .onnx per variant, inputs.json (the pinned
inputs), ort.json (ORT outputs as exact bit patterns, plus the ORT version).
"""
import hashlib, json, os, struct
import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
import onnxruntime as ort

OUT = os.path.join(os.path.dirname(__file__), "..", "fixtures", "tiny")
OPSET = 18


def session(model):
    """ORT with its graph optimizer off, so every source node runs as written:
    the oracle for source semantics (D-TRACT-HARVEST-0 found ORT's default
    optimizer changes RMSNorm bytes on the Jina slice)."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])


def build(variant="base"):
    """Graph: S = ReduceSum(Where(Greater(X, T), X, Z), axes=[0], keepdims=0)
              Q = Div(Sqrt(Abs(Mul(Add(F, B), F))), B)
    X: int32[N] (symbolic N), F: float[2,3]; T, Z, axes, B are initializers.
    B broadcasts [3] against [2,3]: the shape constraint the admission checks."""
    t_val, z_val, b = 0, 0, np.array([0.5, -2.0, 4.0], np.float32)
    abs_op, domain, opset = "Abs", "", OPSET
    if variant == "neg_shape":
        b = np.array([0.5, -2.0, 4.0, 1.0], np.float32)  # [4] vs [2,3]: not broadcastable
    if variant == "neg_where_nonzero":
        z_val = 1  # ReduceSum(Where(c, X, 1)) is not a masked sum of X
    if variant == "neg_unsupported":
        abs_op = "Erf"  # no admitted native method
    if variant == "neg_domain":
        domain = "com.example"  # same op_type in a foreign domain
    if variant == "div3":
        b = np.array([0.5, -2.0, 3.0], np.float32)  # 1/3 is inexact: Div != Mul(recip)
    if variant == "gen2":
        b = np.array([0.5, -2.0, 8.0], np.float32)  # same shape, new weight generation
    inits = [
        nh.from_array(np.array(t_val, np.int32), "T"),
        nh.from_array(np.array(z_val, np.int32), "Z"),
        nh.from_array(np.array([0], np.int64), "axes"),
        nh.from_array(b, "B"),
    ]
    nodes = [
        h.make_node("Greater", ["X", "T"], ["C"], name="n_greater"),
        h.make_node("Where", ["C", "X", "Z"], ["W"], name="n_where"),
        h.make_node("ReduceSum", ["W", "axes"], ["S"], name="n_reducesum", keepdims=0),
        h.make_node("Add", ["F", "B"], ["G"], name="n_add"),
        h.make_node("Mul", ["G", "F"], ["H"], name="n_mul"),
        h.make_node(abs_op, ["H"], ["A"], name="n_abs", domain=domain if abs_op == "Abs" else ""),
        h.make_node("Sqrt", ["A"], ["Y"], name="n_sqrt"),
        h.make_node("Div", ["Y", "B"], ["Q"], name="n_div"),
    ]
    g = h.make_graph(
        nodes, f"tiny_{variant}",
        [h.make_tensor_value_info("X", T.INT32, ["N"]),
         h.make_tensor_value_info("F", T.FLOAT, [2, 3])],
        [h.make_tensor_value_info("S", T.INT32, []),
         h.make_tensor_value_info("Q", T.FLOAT, [2, 3])],
        inits)
    imports = [h.make_opsetid("", opset)]
    if domain:
        imports.append(h.make_opsetid(domain, 1))
    m = h.make_model(g, opset_imports=imports, producer_name="d-tract-harvest-0", ir_version=8)
    return m


INPUTS = {
    "X": np.array([3, -1, 4, -1, 5, -9, 2], np.int32),
    "F": np.array([[1.5, -0.25, 3.0], [0.1, 7.0, -2.5]], np.float32),
}


def bits(a):
    a = np.ascontiguousarray(a)
    if a.dtype == np.float32:
        return [int(v) for v in a.view(np.uint32).ravel()]
    return [int(v) for v in a.ravel()]


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {"ort_version": ort.__version__, "ort_graph_optimization_level": "ORT_DISABLE_ALL", "onnx_version": onnx.__version__, "variants": {}}
    for v in ["base", "gen2", "div3", "neg_shape", "neg_where_nonzero", "neg_unsupported", "neg_domain"]:
        m = build(v)
        data = m.SerializeToString()
        path = os.path.join(OUT, f"{v}.onnx")
        with open(path, "wb") as f:
            f.write(data)
        entry = {"sha256": hashlib.sha256(data).hexdigest()}
        try:
            sess = session(data)
            outs = sess.run(None, INPUTS)
            entry["ort"] = {o.name: {"shape": list(r.shape), "dtype": str(r.dtype), "bits": bits(r)}
                            for o, r in zip(sess.get_outputs(), outs)}
        except Exception as e:  # ORT refusing a variant is itself evidence
            entry["ort_error"] = type(e).__name__ + ": " + str(e).splitlines()[0][:200]
        manifest["variants"][v] = entry
    # div3 sweep: 256 seeded F inputs through ORT, so a property that holds for
    # some values only (Div vs Mul(recip)) is tested on a population.
    rng = np.random.default_rng(20261030)
    fs = (rng.standard_normal((256, 2, 3)) * 8).astype(np.float32)
    sess = session(build("div3").SerializeToString())
    manifest["div3_sweep"] = {
        "seed": 20261030,
        "F_bits": [bits(f) for f in fs],
        "Q_bits": [bits(sess.run(["Q"], {"X": INPUTS["X"], "F": f})[0]) for f in fs],
    }
    manifest["inputs"] = {k: {"shape": list(a.shape), "dtype": str(a.dtype), "bits": bits(a)} for k, a in INPUTS.items()}
    with open(os.path.join(OUT, "ort.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: list(v.keys()) for k, v in manifest["variants"].items()}))


if __name__ == "__main__":
    main()
