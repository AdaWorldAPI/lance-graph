#!/usr/bin/env python3
"""D-TRACT-HARVEST-0 fixture B: one contiguous subgraph of the published Jina v5
small retrieval ONNX export, with ONNX Runtime outputs as the oracle.

The model weights are CC-BY-NC-4.0, so nothing here is committed except the
manifest of digests (fixtures/jina_manifest.json). Every fetched byte range is
SHA-256 pinned; a mismatch aborts. Fetched data goes to fixtures/jina/
(gitignored). Run from the crate root:

    python3 tools/fetch_jina_slice.py

The slice is layer 0's input RMSNorm, exactly as exported:
    /embed_tokens/Gather -> /layers.0/input_layernorm/{Cast, Pow, ReduceMean,
    Add, Sqrt, Div, Mul, Cast_1, Mul_1}
The Gather is resolved by fetching only the embedding rows of the pinned token
ids (one 4096-byte HTTP range each); the rest of the slice is extracted node for
node, with its Constants, and the one weight it reads inlined.
"""
import hashlib, json, os, sys, urllib.request
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort
from tokenizers import Tokenizer

REPO = "jinaai/jina-embeddings-v5-text-small-retrieval"
REV = "6856e76bb72982e58de0620458a4e8b3614da340"
PINS = {
    "onnx/model.onnx": "463fb3fec2adf9c47a946626dc2a0884e43091f95466083fb1754950000a7c1a",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}
TEXT = "Harvest the algebra. Keep the native substrate."
HERE = os.path.join(os.path.dirname(__file__), "..", "fixtures")
OUT = os.path.join(HERE, "jina")
SLICE_NODES = ["/layers.0/input_layernorm/" + n for n in
               ["Cast", "Constant", "Pow", "Constant_1", "ReduceMean", "Constant_2", "Add",
                "Sqrt", "Constant_3", "Div", "Mul", "Cast_1", "Mul_1"]]
SLICE_IN = "/embed_tokens/Gather_output_0"
SLICE_MID = "/layers.0/input_layernorm/ReduceMean_output_0"
SLICE_OUT = "/layers.0/input_layernorm/Mul_1_output_0"


def url(path):
    return f"https://huggingface.co/{REPO}/resolve/{REV}/{path}"


def fetch(path, rng=None):
    req = urllib.request.Request(url(path))
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[0] + rng[1] - 1}")
    with urllib.request.urlopen(req) as r:
        data = r.read()
    if rng and len(data) != rng[1]:
        sys.exit(f"range {path} {rng}: got {len(data)} bytes")
    return data


def pinned(path):
    local = os.path.join(OUT, path.replace("/", "_"))
    if not os.path.exists(local):
        with open(local, "wb") as f:
            f.write(fetch(path))
    data = open(local, "rb").read()
    got = hashlib.sha256(data).hexdigest()
    if got != PINS[path]:
        sys.exit(f"{path}: sha256 {got} != pinned {PINS[path]}")
    return local, data


def ext(t):
    d = {e.key: e.value for e in t.external_data}
    return d["location"], int(d["offset"]), int(d["length"])


def main():
    os.makedirs(OUT, exist_ok=True)
    model_path, model_bytes = pinned("onnx/model.onnx")
    tok_path, _ = pinned("tokenizer.json")
    enc = Tokenizer.from_file(tok_path).encode(TEXT, add_special_tokens=False)
    ids = list(enc.ids)

    m = onnx.load(model_path, load_external_data=False)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    emb, w = inits["embed_tokens.weight"], inits["layers.0.input_layernorm.weight"]
    hidden = emb.dims[1]
    row_bytes = hidden * 4
    loc, base, length = ext(emb)
    assert emb.data_type == T.FLOAT and length == emb.dims[0] * row_bytes

    ranges = []
    rows = []
    for tid in ids:
        r = (base + tid * row_bytes, row_bytes)
        data = fetch("onnx/" + loc, r)
        ranges.append({"token_id": tid, "offset": r[0], "length": r[1],
                       "sha256": hashlib.sha256(data).hexdigest()})
        rows.append(np.frombuffer(data, "<f4"))
    wloc, woff, wlen = ext(w)
    wdata = fetch("onnx/" + wloc, (woff, wlen))
    weight = np.frombuffer(wdata, "<f4").copy()

    x = np.stack(rows)[None, :, :].astype(np.float32)  # [1, S, hidden]
    x.tofile(os.path.join(OUT, "gather_rows.f32"))
    weight.tofile(os.path.join(OUT, "norm_weight.f32"))

    # Extract the slice node for node.
    by_name = {n.name: n for n in g.node}
    nodes = [by_name[n] for n in SLICE_NODES]
    wt = nh.from_array(weight, "layers.0.input_layernorm.weight")
    sg = h.make_graph(
        nodes, "jina_v5_layer0_input_rmsnorm",
        [h.make_tensor_value_info(SLICE_IN, T.FLOAT, [1, len(ids), hidden])],
        [h.make_tensor_value_info(SLICE_OUT, T.FLOAT, [1, len(ids), hidden]),
         h.make_tensor_value_info(SLICE_MID, T.FLOAT, [1, len(ids), 1])],
        [wt])
    sm = h.make_model(sg, opset_imports=list(m.opset_import), ir_version=m.ir_version,
                      producer_name="d-tract-harvest-0 slice of " + REPO)
    slice_bytes = sm.SerializeToString()
    with open(os.path.join(OUT, "slice.onnx"), "wb") as f:
        f.write(slice_bytes)

    # Two ORT arms. "literal" disables ORT's graph optimizer, so each source
    # node runs as written: that is the source-semantics oracle. "optimized"
    # is ORT's default, which may fuse the pattern into its own kernel; it is
    # recorded for comparison and never asserted against.
    arms = {}
    for arm, level in [("literal", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
                       ("optimized", ort.GraphOptimizationLevel.ORT_ENABLE_ALL)]:
        so = ort.SessionOptions()
        so.graph_optimization_level = level
        sess = ort.InferenceSession(slice_bytes, so, providers=["CPUExecutionProvider"])
        out, mid = sess.run([SLICE_OUT, SLICE_MID], {SLICE_IN: x})
        sfx = "" if arm == "literal" else "_opt"
        out.astype("<f4").tofile(os.path.join(OUT, f"ort_out{sfx}.f32"))
        mid.astype("<f4").tofile(os.path.join(OUT, f"ort_mid{sfx}.f32"))
        arms[arm] = (out, mid)
    out, mid = arms["literal"]

    sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    manifest = {
        "repo": REPO, "revision": REV, "license": "cc-by-nc-4.0 (weights never committed)",
        "pins": PINS, "text": TEXT, "token_ids": ids, "offsets": [list(o) for o in enc.offsets],
        "graph": {"ir_version": m.ir_version, "opset": [(o.domain or "ai.onnx", o.version) for o in m.opset_import],
                  "nodes": len(g.node), "initializers": len(g.initializer)},
        "slice": {"nodes": SLICE_NODES, "input": SLICE_IN, "output": SLICE_OUT, "intermediate": SLICE_MID,
                  "shape": [1, len(ids), hidden], "sha256": hashlib.sha256(slice_bytes).hexdigest()},
        "embedding": {"name": emb.name, "dims": list(emb.dims), "location": loc, "base_offset": base, "ranges": ranges},
        "norm_weight": {"name": w.name, "offset": woff, "length": wlen, "sha256": hashlib.sha256(wdata).hexdigest()},
        "ort": {"version": ort.__version__, "provider": "CPUExecutionProvider",
                "literal": {"graph_optimization_level": "ORT_DISABLE_ALL",
                            "out_sha256": sha(out.astype("<f4")), "mid_sha256": sha(mid.astype("<f4"))},
                "optimized": {"graph_optimization_level": "ORT_ENABLE_ALL",
                              "out_sha256": sha(arms["optimized"][0].astype("<f4")),
                              "mid_sha256": sha(arms["optimized"][1].astype("<f4"))}},
    }
    with open(os.path.join(HERE, "jina_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("token ids", ids)
    print("slice sha", manifest["slice"]["sha256"]); print("literal == optimized:", manifest["ort"]["literal"]["out_sha256"] == manifest["ort"]["optimized"]["out_sha256"])


if __name__ == "__main__":
    main()
