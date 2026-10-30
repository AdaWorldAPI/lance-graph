#!/usr/bin/env python3
"""D-TRACT-SHAPE-1: the Jina graph's Constant nodes whose tensor lives in the
external weight file (the rotary inv_freq), fetched as HTTP ranges.

The model data is CC-BY-NC-4.0: the fetched bytes go to fixtures/jina/
(gitignored); only fixtures/jina_constants.json (offsets, lengths, dims and
SHA-256) is committed. A range whose digest differs from the committed one
aborts. Run from the crate root, after tools/fetch_jina_slice.py:

    python3 tools/fetch_jina_constants.py
"""
import hashlib, json, os, sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h, numpy_helper as nh

from fetch_jina_slice import HERE, OUT, ext, fetch

MANIFEST = os.path.join(HERE, "jina_constants.json")
# The rotary subgraph's outputs: cos and sin after their scaling Mul and Cast.
ROTARY_OUT = ["/rotary_emb/Cast_4_output_0", "/rotary_emb/Cast_5_output_0"]
SEQ = 10
# Only small constants: anything larger is a weight, not a shape-path constant.
MAX_BYTES = 64 * 1024
DTYPES = {1: "f32", 7: "i64"}


def main():
    model = os.path.join(OUT, "onnx_model.onnx")
    if not os.path.exists(model):
        sys.exit("run tools/fetch_jina_slice.py first")
    pinned = {}
    if os.path.exists(MANIFEST):
        pinned = {c["output"]: c for c in json.load(open(MANIFEST))["constants"]}
    m = onnx.load(model, load_external_data=False)
    g = m.graph
    found = []
    for n in g.node:
        if n.op_type != "Constant":
            continue
        for a in n.attribute:
            if a.name != "value" or a.t.data_location != 1:
                continue
            loc, off, length = ext(a.t)
            if length > MAX_BYTES or a.t.data_type not in DTYPES:
                sys.exit(f"{n.name}: {length} bytes of type {a.t.data_type} is not a small constant")
            data = fetch("onnx/" + loc, (off, length))
            digest = hashlib.sha256(data).hexdigest()
            out = n.output[0]
            if out in pinned and pinned[out]["sha256"] != digest:
                sys.exit(f"{out}: sha256 {digest} != pinned {pinned[out]['sha256']}")
            fname = "const_" + "".join(c if c.isalnum() else "_" for c in out) + ".bin"
            with open(os.path.join(OUT, fname), "wb") as f:
                f.write(data)
            found.append({"node": n.name, "output": out, "dtype": DTYPES[a.t.data_type],
                          "dims": list(a.t.dims), "location": loc, "offset": off,
                          "length": length, "sha256": digest, "file": fname})
    rotary = rotary_oracle(m, found)
    with open(MANIFEST, "w") as f:
        json.dump({"license": "cc-by-nc-4.0 (data never committed)", "constants": found,
                   "rotary": rotary}, f, indent=1)
    for c in found:
        print(c["node"], c["output"], c["dims"], c["sha256"][:16])


def rotary_oracle(m, found):
    """The rotary subgraph, extracted node for node with the fetched
    constants inlined, run by ORT with optimizations disabled for input_ids of
    shape [1, SEQ]. Its outputs are written next to the constants (gitignored);
    only their digests are committed."""
    g = m.graph
    by_out = {o: n for n in g.node for o in n.output}
    inits = {t.name: t for t in g.initializer}
    data = {c["output"]: np.fromfile(os.path.join(OUT, c["file"]), "<f4" if c["dtype"] == "f32" else "<i8")
            .reshape(c["dims"]) for c in found}
    need, seen, stack = [], set(), list(ROTARY_OUT)
    used_inits, graph_in = set(), set()
    while stack:
        v = stack.pop()
        if v in seen or not v:
            continue
        seen.add(v)
        if v in inits:
            used_inits.add(v)
        elif v in by_out:
            n = by_out[v]
            if n not in need:
                need.append(n)
            stack.extend(n.input)
        else:
            graph_in.add(v)
    order = [n for n in g.node if n in need]
    nodes = []
    for n in order:
        if n.op_type == "Constant" and n.output[0] in data:
            n = h.make_node("Constant", [], list(n.output), name=n.name,
                            value=nh.from_array(data[n.output[0]].astype(np.float32), "v"))
        nodes.append(n)
    for t in used_inits:
        assert inits[t].data_location != 1, f"{t} is external"
    assert graph_in == {"input_ids"}, graph_in
    sg = h.make_graph(nodes, "jina_rotary", [h.make_tensor_value_info("input_ids", 7, [1, SEQ])],
                      [h.make_tensor_value_info(o, 1, None) for o in ROTARY_OUT],
                      [inits[t] for t in sorted(used_inits)])
    sm = h.make_model(sg, opset_imports=list(m.opset_import), ir_version=m.ir_version)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(sm.SerializeToString(), so, providers=["CPUExecutionProvider"])
    outs = sess.run(ROTARY_OUT, {"input_ids": np.zeros((1, SEQ), np.int64)})
    rec = {"seq": SEQ, "nodes": len(nodes), "ort_version": ort.__version__,
           "graph_optimization_level": "ORT_DISABLE_ALL", "outputs": []}
    for name, arr in zip(ROTARY_OUT, outs):
        arr = np.ascontiguousarray(arr.astype("<f4"))
        fname = "rotary_" + "".join(c if c.isalnum() else "_" for c in name) + ".f32"
        arr.tofile(os.path.join(OUT, fname))
        rec["outputs"].append({"name": name, "shape": list(arr.shape), "file": fname,
                               "sha256": hashlib.sha256(arr.tobytes()).hexdigest()})
    return rec


if __name__ == "__main__":
    main()
