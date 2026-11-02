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
        abs_op = "Acosh"  # no admitted native method
    if variant == "erf":
        abs_op = "Erf"  # admitted: ndarray::hpc::vml::vserf (Bounded)
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
    outs = [h.make_tensor_value_info("S", T.INT32, []),
            h.make_tensor_value_info("Q", T.FLOAT, [2, 3])]
    if variant == "erf":
        outs.append(h.make_tensor_value_info("A", T.FLOAT, [2, 3]))  # the Erf output itself
    g = h.make_graph(
        nodes, f"tiny_{variant}",
        [h.make_tensor_value_info("X", T.INT32, ["N"]),
         h.make_tensor_value_info("F", T.FLOAT, [2, 3])],
        outs,
        inits)
    imports = [h.make_opsetid("", opset)]
    if domain:
        imports.append(h.make_opsetid(domain, 1))
    m = h.make_model(g, opset_imports=imports, producer_name="d-tract-harvest-0", ir_version=8)
    return m


def i64(name, v):
    return nh.from_array(np.array(v, np.int64), name)


_mm = np.random.default_rng(20261031)
MATMUL_FEEDS = {"A": _mm.standard_normal((1, 4, 64)).astype(np.float32),
                "Q": _mm.standard_normal((1, 2, 4, 16)).astype(np.float32),
                "K": _mm.standard_normal((1, 2, 16, 4)).astype(np.float32)}


def build_shape(variant):
    """D-TRACT-SHAPE-1 shape plumbing.

    shape_fold: T = Concat(Unsqueeze(Gather(Shape(F), 1), [0]), [-1]) is
      [3, -1], a function of F's (known) shape only, so it folds;
      R = Reshape(F, T) and U = Unsqueeze(F, [0]) read F's values and run.
    shape_sym: the same chain over X float[N]: Shape(X) has a symbolic
      dimension, nothing folds, the Reshape target is not a constant and its
      WHEN is UNKNOWN.
    neg_gather_index: Gather(Shape(F), 5): index outside [-2, 2).
    neg_concat: Concat(F[2,3], G[2,2], axis=0): dimension 1 differs."""
    inits = [i64("one", 1), i64("ax0", [0]), i64("m1", [-1])]
    if variant == "shape_fold":
        inp = [h.make_tensor_value_info("F", T.FLOAT, [2, 3])]
        nodes = [
            h.make_node("Shape", ["F"], ["s"], name="n_shape"),
            h.make_node("Gather", ["s", "one"], ["d"], name="n_gather", axis=0),
            h.make_node("Unsqueeze", ["d", "ax0"], ["u"], name="n_unsq_d"),
            h.make_node("Concat", ["u", "m1"], ["t"], name="n_concat", axis=0),
            h.make_node("Reshape", ["F", "t"], ["R"], name="n_reshape"),
            h.make_node("Unsqueeze", ["F", "ax0"], ["U"], name="n_unsq_f"),
        ]
        outs = [h.make_tensor_value_info("R", T.FLOAT, [3, 2]),
                h.make_tensor_value_info("U", T.FLOAT, [1, 2, 3])]
    elif variant == "shape_sym":
        inp = [h.make_tensor_value_info("X", T.FLOAT, ["N"])]
        inits.append(i64("zero", 0))
        nodes = [
            h.make_node("Shape", ["X"], ["s"], name="n_shape"),
            h.make_node("Gather", ["s", "zero"], ["d"], name="n_gather", axis=0),
            h.make_node("Unsqueeze", ["d", "ax0"], ["u"], name="n_unsq_d"),
            h.make_node("Concat", ["u", "m1"], ["t"], name="n_concat", axis=0),
            h.make_node("Reshape", ["X", "t"], ["R"], name="n_reshape"),
        ]
        outs = [h.make_tensor_value_info("R", T.FLOAT, ["N", 1])]
    elif variant == "neg_gather_index":
        inp = [h.make_tensor_value_info("F", T.FLOAT, [2, 3])]
        inits.append(i64("five", 5))
        nodes = [
            h.make_node("Shape", ["F"], ["s"], name="n_shape"),
            h.make_node("Gather", ["s", "five"], ["d"], name="n_gather", axis=0),
        ]
        outs = [h.make_tensor_value_info("d", T.INT64, [])]
    elif variant == "neg_concat":
        inp = [h.make_tensor_value_info("F", T.FLOAT, [2, 3])]
        inits.append(nh.from_array(np.zeros((2, 2), np.float32), "G"))
        nodes = [h.make_node("Concat", ["F", "G"], ["C"], name="n_concat", axis=0)]
        outs = [h.make_tensor_value_info("C", T.FLOAT, [4, 3])]
    elif variant == "shape_i64":
        # int64 compares, Where, ConstantOfShape, Expand over Shape(F) = [2, 3].
        inp = [h.make_tensor_value_info("F", T.FLOAT, [2, 3])]
        inits += [i64("k24", [2, 4]), i64("seven", 7), i64("k33", [3, 3]), i64("row", [[1, 2, 3]]),
                  i64("k223", [2, 2, 3])]
        nodes = [
            h.make_node("Shape", ["F"], ["s"], name="n_shape"),
            h.make_node("Equal", ["s", "k24"], ["eq"], name="n_equal"),
            h.make_node("Where", ["eq", "s", "seven"], ["W"], name="n_where"),
            h.make_node("LessOrEqual", ["W", "k33"], ["LE"], name="n_le"),
            h.make_node("ConstantOfShape", ["s"], ["CS"], name="n_cos",
                        value=nh.from_array(np.array([5], np.int64))),
            h.make_node("Expand", ["row", "s"], ["ER"], name="n_expand_row"),
            h.make_node("Expand", ["F", "k223"], ["EF"], name="n_expand_f"),
        ]
        outs = [h.make_tensor_value_info("W", T.INT64, [2]),
                h.make_tensor_value_info("LE", T.BOOL, [2]),
                h.make_tensor_value_info("CS", T.INT64, [2, 3]),
                h.make_tensor_value_info("ER", T.INT64, [2, 3]),
                h.make_tensor_value_info("EF", T.FLOAT, [2, 2, 3])]
    elif variant == "i64_arith":
        # The int64 shape arithmetic no ndarray word covers yet (the ndarray
        # ask): truncating Div on negatives, Mul, Add, Sub, Range.
        inp = [h.make_tensor_value_info("F", T.FLOAT, [2, 3])]
        inits += [i64("negs", [-7, 7]), i64("mul_k", [3, -4]), i64("add_k", [-5, 10]),
                  i64("r0", 0), i64("r1", 1)]
        nodes = [
            h.make_node("Shape", ["F"], ["s"], name="n_shape"),
            h.make_node("Div", ["negs", "s"], ["DV"], name="n_div"),
            h.make_node("Mul", ["s", "mul_k"], ["ML"], name="n_mul"),
            h.make_node("Add", ["s", "add_k"], ["AD"], name="n_add"),
            h.make_node("Sub", ["s", "add_k"], ["SB"], name="n_sub"),
            h.make_node("Gather", ["s", "r1"], ["n3"], name="n_gather", axis=0),
            h.make_node("Range", ["r0", "n3", "r1"], ["RG"], name="n_range"),
        ]
        outs = [h.make_tensor_value_info(o, T.INT64, None) for o in ["DV", "ML", "AD", "SB", "RG"]]
    elif variant == "i64_data":
        # int64 arithmetic on a run-time input: the DO runs per run, and a
        # run whose values overflow refuses.
        inp = [h.make_tensor_value_info("F", T.FLOAT, [2, 3]),
               h.make_tensor_value_info("I", T.INT64, [2])]
        nodes = [h.make_node("Shape", ["F"], ["s"], name="n_shape"),
                 h.make_node("Mul", ["I", "s"], ["MI"], name="n_mul")]
        outs = [h.make_tensor_value_info("MI", T.INT64, [2])]
    elif variant in ("neg_i64_overflow", "neg_i64_div0", "neg_range_zero", "neg_i64_bcast"):
        inp = [h.make_tensor_value_info("F", T.FLOAT, [2, 3])]
        node = {
            "neg_i64_overflow": (h.make_node("Mul", ["s", "k"], ["Y"], name="n_bad"), i64("k", [2**62, 1])),
            "neg_i64_div0": (h.make_node("Div", ["s", "k"], ["Y"], name="n_bad"), i64("k", [1, 0])),
            "neg_range_zero": (h.make_node("Range", ["r0", "n3", "k"], ["Y"], name="n_bad"), i64("k", 0)),
            "neg_i64_bcast": (h.make_node("Add", ["s", "k"], ["Y"], name="n_bad"),
                              nh.from_array(np.array([[1], [2]], np.int64), "k")),
        }[variant]
        inits += [node[1], i64("r0", 0), i64("r1", 1)]
        nodes = [h.make_node("Shape", ["F"], ["s"], name="n_shape"),
                 h.make_node("Gather", ["s", "r1"], ["n3"], name="n_gather", axis=0),
                 node[0]]
        outs = [h.make_tensor_value_info("Y", T.INT64, None)]
    elif variant == "rope":
        # The Jina rotary-embedding and causal-mask structure, self-authored:
        # positions come from the input's shape (Shape -> Gather -> Range), so
        # with a fixed shape the whole chain is shape-only and folds.
        # inv_freq is synthetic, 1 / 10000^(2i/64) for i < 32.
        inp = [h.make_tensor_value_info("ids", T.INT64, [1, 10])]
        inv = (1.0 / (10000.0 ** (np.arange(0, 64, 2, dtype=np.float64) / 64.0))).astype(np.float32)
        inits += [i64("r0", 0), i64("r1", 1), i64("ax0", [0]), i64("ax1", [1]), i64("ax2", [2]),
                  i64("mshape", [10, 10]),
                  nh.from_array(inv.reshape(1, 32, 1), "inv_freq"),
                  nh.from_array(np.array(1.0, np.float32), "scale")]
        nodes = [
            h.make_node("Shape", ["ids"], ["s"], name="n_shape"),
            h.make_node("Gather", ["s", "r1"], ["n"], name="n_gather", axis=0),
            h.make_node("Range", ["r0", "n", "r1"], ["pos"], name="n_range"),
            h.make_node("Unsqueeze", ["pos", "ax0"], ["pos1"], name="n_unsq0"),
            h.make_node("Unsqueeze", ["pos1", "ax1"], ["pos2"], name="n_unsq1"),
            h.make_node("Cast", ["pos2"], ["posf"], name="n_cast_pos", to=1),
            h.make_node("MatMul", ["inv_freq", "posf"], ["fr"], name="n_matmul"),
            h.make_node("Transpose", ["fr"], ["frt"], name="n_transpose", perm=[0, 2, 1]),
            h.make_node("Concat", ["frt", "frt"], ["emb"], name="n_concat", axis=-1),
            h.make_node("Cos", ["emb"], ["c"], name="n_cos"),
            h.make_node("Sin", ["emb"], ["sn"], name="n_sin"),
            h.make_node("Mul", ["c", "scale"], ["COS"], name="n_mul_cos"),
            h.make_node("Mul", ["sn", "scale"], ["SIN"], name="n_mul_sin"),
            # Causal mask: key position <= query position, broadcast [10,1] vs [1,10].
            h.make_node("Unsqueeze", ["pos", "ax1"], ["q"], name="n_unsq_q"),
            h.make_node("Unsqueeze", ["pos", "ax0"], ["k"], name="n_unsq_k"),
            h.make_node("LessOrEqual", ["k", "q"], ["le"], name="n_le"),
            h.make_node("Cast", ["le"], ["leb"], name="n_cast_bool", to=9),
            h.make_node("ConstantOfShape", ["mshape"], ["ones"], name="n_ones",
                        value=nh.from_array(np.array([True]), "v")),
            h.make_node("And", ["ones", "leb"], ["MASK"], name="n_and"),
        ]
        outs = [h.make_tensor_value_info("COS", T.FLOAT, None), h.make_tensor_value_info("SIN", T.FLOAT, None),
                h.make_tensor_value_info("MASK", T.BOOL, None)]
    elif variant == "matmul":
        # f32 MatMul with K > 1 (accumulation order matters): an activation
        # against a weight, and an attention-shaped batched product.
        rng = np.random.default_rng(20261030)
        inp = [h.make_tensor_value_info("A", T.FLOAT, [1, 4, 64]),
               h.make_tensor_value_info("Q", T.FLOAT, [1, 2, 4, 16]),
               h.make_tensor_value_info("K", T.FLOAT, [1, 2, 16, 4])]
        inits += [nh.from_array(rng.standard_normal((64, 8)).astype(np.float32), "W")]
        nodes = [h.make_node("MatMul", ["A", "W"], ["Y1"], name="n_mm_weight"),
                 h.make_node("MatMul", ["Q", "K"], ["Y2"], name="n_mm_attn")]
        outs = [h.make_tensor_value_info("Y1", T.FLOAT, [1, 4, 8]),
                h.make_tensor_value_info("Y2", T.FLOAT, [1, 2, 4, 4])]
    elif variant in ("lstm_skip", "lstm_full"):
        # An omitted optional output before a present one: Tract numbers a
        # node's outlets after dropping empty names (onnx/src/model.rs:150).
        hs = 2
        inp = [h.make_tensor_value_info("X", T.FLOAT, [5, 1, 3])]
        inits = [nh.from_array(np.full((1, 4 * hs, 3), 0.1, np.float32), "W"),
                 nh.from_array(np.full((1, 4 * hs, hs), 0.2, np.float32), "R")]
        outs_ = ["", "Y_h"] if variant == "lstm_skip" else ["Y", "Y_h"]
        nodes = [h.make_node("LSTM", ["X", "W", "R"], outs_, name="n_lstm", hidden_size=hs)]
        outs = [h.make_tensor_value_info("Y_h", T.FLOAT, [1, 1, hs])]
    g = h.make_graph(nodes, f"tiny_{variant}", inp, outs, inits)
    return h.make_model(g, opset_imports=[h.make_opsetid("", OPSET)], producer_name="d-tract-shape-1", ir_version=8)


_jn = np.random.default_rng(20261101)
JINA_FEEDS = {
    "X": _jn.standard_normal((1, 2, 3, 8)).astype(np.float32),
    "M": _jn.integers(0, 2, (1, 1, 3, 3)).astype(np.bool_),
    "C": _jn.integers(0, 2, (1, 2, 3, 3)).astype(np.bool_),
    "Y": _jn.standard_normal((1, 2, 3, 3)).astype(np.float32),
    "NAN_IN": _jn.standard_normal((1, 2, 3, 3)).astype(np.float32),
    "SM_IN": _jn.standard_normal((1, 2, 3, 3)).astype(np.float32),
    "SG_IN": _jn.standard_normal((1, 3, 6)).astype(np.float32),
    "CI": np.array([[1, 0, 1, 1]], np.int64),
    "RI": np.array([[1, 0, 1, 1]], np.int64),
    "BE": _jn.integers(0, 2, (1, 1, 3, 3)).astype(np.bool_),
    "GE": _jn.standard_normal((1, 4, 5)).astype(np.float32),
}
JINA_FEEDS["NAN_IN"][0, 0, 0, 0] = np.nan
JINA_FEEDS["NAN_IN"][0, 1, 2, 2] = np.nan
JINA_FEEDS["NAN_IN"][0, 0, 1, 1] = np.inf
JINA_FEEDS["NAN_IN"][0, 1, 0, 1] = -np.inf
JINA_FEEDS["SG_IN"].reshape(-1)[:5] = np.array([-20.0, -1.0, 0.0, 1.0, 20.0], np.float32)


def build_jina_ops():
    """Jina's 14 unregistered op types, one node per shape pattern.

    Every node's output is also a graph output, so each op is checked
    separately against ORT. Independent graph inputs per op where the pattern
    allows it, so a failure points at one op."""
    f32min = np.array([-3.4028234663852886e38], np.float32)
    inits = [
        i64("st0", [0]), i64("st4", [4]), i64("en3", [3]), i64("en4", [4]), i64("en8", [8]),
        i64("ax3", [3]), i64("step1", [1]),
        nh.from_array(f32min, "Xc"),
        nh.from_array(np.array([0.0], np.float32), "Xc0"),
        nh.from_array(f32min.copy(), "Yc"),
        nh.from_array(_jn.standard_normal((5, 4)).astype(np.float32), "GD"),
        i64("gidx", [[4, 0, 2]]),
        i64("gidx_b", [[[[0, 1, 2, 3]]]]),
        i64("shp4", [4]), i64("shp1114", [1, 1, 1, 4]), i64("shp1133", [1, 1, 3, 3]),
        i64("rs_axes", [1]),
        i64("ge_idx", np.full((1, 1, 5), 3, np.int64)),
        i64("sq_axes", [1]),
    ]
    nodes = [
        h.make_node("Slice", ["X", "st4", "en8", "ax3", "step1"], ["slice_hi"], name="n_slice_hi"),
        h.make_node("Slice", ["X", "st0", "en4", "ax3", "step1"], ["slice_lo"], name="n_slice_lo"),
        h.make_node("Neg", ["slice_lo"], ["neg"], name="n_neg"),
        h.make_node("Slice", ["M", "st0", "en3", "ax3", "step1"], ["slice_bool"], name="n_slice_bool"),
        h.make_node("Where", ["C", "Xc", "Y"], ["where_a"], name="n_where_a"),
        h.make_node("Where", ["slice_bool", "Xc0", "Yc"], ["where_b"], name="n_where_b"),
        h.make_node("IsNaN", ["NAN_IN"], ["isnan"], name="n_isnan"),
        h.make_node("Softmax", ["SM_IN"], ["softmax"], name="n_softmax", axis=-1),
        h.make_node("Sigmoid", ["SG_IN"], ["sigmoid"], name="n_sigmoid"),
        h.make_node("Gather", ["GD", "gidx"], ["gather_f"], name="n_gather_f", axis=0),
        h.make_node("Cast", ["CI"], ["cast_bool"], name="n_cast_bool", to=9),
        h.make_node("Flatten", ["cast_bool"], ["flatten_b"], name="n_flatten_b", axis=2),
        h.make_node("Gather", ["flatten_b", "gidx_b"], ["gather_b"], name="n_gather_b", axis=0),
        h.make_node("Reshape", ["gather_b", "shp4"], ["reshape_b1"], name="n_reshape_b1"),
        h.make_node("Reshape", ["reshape_b1", "shp1114"], ["reshape_b2"], name="n_reshape_b2"),
        h.make_node("Expand", ["BE", "shp1133"], ["expand_b"], name="n_expand_b"),
        h.make_node("ReduceSum", ["RI", "rs_axes"], ["reduce_i64"], name="n_reduce_i64", keepdims=0),
        h.make_node("GatherElements", ["GE", "ge_idx"], ["gather_el"], name="n_gather_el", axis=1),
        h.make_node("Squeeze", ["gather_el", "sq_axes"], ["squeeze"], name="n_squeeze"),
    ]
    vi = h.make_tensor_value_info
    inp = [vi("X", T.FLOAT, [1, 2, 3, 8]), vi("M", T.BOOL, [1, 1, 3, 3]),
           vi("C", T.BOOL, [1, 2, 3, 3]), vi("Y", T.FLOAT, [1, 2, 3, 3]),
           vi("NAN_IN", T.FLOAT, [1, 2, 3, 3]), vi("SM_IN", T.FLOAT, [1, 2, 3, 3]),
           vi("SG_IN", T.FLOAT, [1, 3, 6]), vi("CI", T.INT64, [1, 4]),
           vi("RI", T.INT64, [1, 4]), vi("BE", T.BOOL, [1, 1, 3, 3]),
           vi("GE", T.FLOAT, [1, 4, 5])]
    outs = [vi("slice_hi", T.FLOAT, [1, 2, 3, 4]), vi("slice_lo", T.FLOAT, [1, 2, 3, 4]),
            vi("neg", T.FLOAT, [1, 2, 3, 4]), vi("slice_bool", T.BOOL, [1, 1, 3, 3]),
            vi("where_a", T.FLOAT, [1, 2, 3, 3]), vi("where_b", T.FLOAT, [1, 1, 3, 3]),
            vi("isnan", T.BOOL, [1, 2, 3, 3]), vi("softmax", T.FLOAT, [1, 2, 3, 3]),
            vi("sigmoid", T.FLOAT, [1, 3, 6]), vi("gather_f", T.FLOAT, [1, 3, 4]),
            vi("cast_bool", T.BOOL, [1, 4]), vi("flatten_b", T.BOOL, [4, 1]),
            vi("gather_b", T.BOOL, [1, 1, 1, 4, 1]), vi("reshape_b1", T.BOOL, [4]),
            vi("reshape_b2", T.BOOL, [1, 1, 1, 4]), vi("expand_b", T.BOOL, [1, 1, 3, 3]),
            vi("reduce_i64", T.INT64, [1]), vi("gather_el", T.FLOAT, [1, 1, 5]),
            vi("squeeze", T.FLOAT, [1, 5])]
    g = h.make_graph(nodes, "tiny_jina_ops", inp, outs, inits)
    return h.make_model(g, opset_imports=[h.make_opsetid("", OPSET)], producer_name="d-tract-shape-1", ir_version=8)


def build_erf_sweep():
    g = h.make_graph([h.make_node("Erf", ["X"], ["Y"], name="n_erf")], "erf_sweep",
                     [h.make_tensor_value_info("X", T.FLOAT, ["N"])],
                     [h.make_tensor_value_info("Y", T.FLOAT, ["N"])])
    return h.make_model(g, opset_imports=[h.make_opsetid("", OPSET)], producer_name="d-tract-shape-1", ir_version=8)


def build_trig_sweep():
    g = h.make_graph([h.make_node("Cos", ["X"], ["C"], name="n_cos"),
                      h.make_node("Sin", ["X"], ["S"], name="n_sin")], "trig_sweep",
                     [h.make_tensor_value_info("X", T.FLOAT, ["N"])],
                     [h.make_tensor_value_info("C", T.FLOAT, ["N"]),
                      h.make_tensor_value_info("S", T.FLOAT, ["N"])])
    return h.make_model(g, opset_imports=[h.make_opsetid("", OPSET)], producer_name="d-tract-shape-1", ir_version=8)


def trig_sweep_inputs():
    """Rotary angles: every position up to 8192 times every synthetic inv_freq,
    plus a dense grid over [-4 pi, 4 pi], signed zeros, infinities and NaN."""
    inv = (1.0 / (10000.0 ** (np.arange(0, 64, 2, dtype=np.float64) / 64.0))).astype(np.float32)
    pos = np.arange(0, 8192, 7, dtype=np.float32)
    rot = (pos[:, None] * inv[None, :]).ravel()
    grid = np.linspace(-4 * np.pi, 4 * np.pi, 4097, dtype=np.float64).astype(np.float32)
    special = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-30, 3.4e38, -3.4e38, 1e6, 8191.0], np.float32)
    return np.concatenate([rot, grid, special])


def erf_sweep_inputs():
    """A dense grid over [-6, 6] (the tails saturate past ~3.9), plus signed
    zeros, infinities, NaN, subnormals and tiny values."""
    grid = np.linspace(-6.0, 6.0, 4097, dtype=np.float64).astype(np.float32)
    special = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-30, -1e-30, 1e-10, -1e-10,
                        np.float32(1.4e-45), 3.4e38, -3.4e38, 0.5, 1.0, 2.0, 3.9, 4.0, 5.0], np.float32)
    return np.concatenate([grid, special])


INPUTS = {
    "X": np.array([3, -1, 4, -1, 5, -9, 2], np.int32),
    "F": np.array([[1.5, -0.25, 3.0], [0.1, 7.0, -2.5]], np.float32),
}


def bits(a):
    a = np.ascontiguousarray(a)
    if a.dtype == np.bool_:
        return [int(v) for v in a.ravel()]
    if a.dtype == np.float32:
        return [int(v) for v in a.view(np.uint32).ravel()]
    return [int(v) for v in a.ravel()]


def dumps_compact(v, depth=0):
    """JSON with one key per line, but every list of scalars on one line, so
    a sweep is one line instead of one line per element."""
    pad = " " * (depth + 1)
    if isinstance(v, dict):
        items = [f"{pad}{json.dumps(k)}: {dumps_compact(x, depth + 1)}" for k, x in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * depth + "}" if items else "{}"
    if isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v):
        return json.dumps(v, separators=(",", ":"))
    if isinstance(v, list):
        items = [pad + dumps_compact(x, depth + 1) for x in v]
        return "[\n" + ",\n".join(items) + "\n" + " " * depth + "]" if items else "[]"
    return json.dumps(v)


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {"ort_version": ort.__version__, "ort_graph_optimization_level": "ORT_DISABLE_ALL", "onnx_version": onnx.__version__, "variants": {}}
    for v in ["base", "gen2", "div3", "neg_shape", "neg_where_nonzero", "neg_unsupported", "neg_domain", "erf"]:
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
    for v in ["shape_fold", "shape_sym", "neg_gather_index", "neg_concat", "shape_i64", "lstm_skip", "lstm_full", "i64_arith",
              "i64_data", "rope", "matmul", "jina_ops", "neg_i64_overflow", "neg_i64_div0", "neg_range_zero", "neg_i64_bcast"]:
        data = (build_jina_ops() if v == "jina_ops" else build_shape(v)).SerializeToString()
        with open(os.path.join(OUT, f"{v}.onnx"), "wb") as f:
            f.write(data)
        entry = {"sha256": hashlib.sha256(data).hexdigest()}
        if v == "matmul":
            entry["inputs"] = {k: {"shape": list(a.shape), "bits": bits(a)} for k, a in MATMUL_FEEDS.items()}
        feeds = {"shape_sym": {"X": np.arange(6, dtype=np.float32)},
                 "lstm_skip": {"X": np.ones((5, 1, 3), np.float32)},
                 "lstm_full": {"X": np.ones((5, 1, 3), np.float32)},
                 "i64_data": {"F": INPUTS["F"], "I": np.array([-4, 5], np.int64)},
                 "rope": {"ids": np.arange(10, dtype=np.int64).reshape(1, 10)},
                 "matmul": MATMUL_FEEDS,
                 "jina_ops": JINA_FEEDS}.get(v, {"F": INPUTS["F"]})
        if v == "jina_ops":
            entry["inputs"] = {k: {"shape": list(a.shape), "dtype": str(a.dtype), "bits": bits(a)}
                               for k, a in JINA_FEEDS.items()}
        if v == "neg_i64_div0":
            # An integer zero divisor is undefined behaviour in ORT's kernel;
            # it is not run. The refusal here needs no oracle.
            manifest["variants"][v] = entry
            continue
        try:
            sess = session(data)
            outs = sess.run(None, feeds)
            entry["ort"] = {o.name: {"shape": list(r.shape), "dtype": str(r.dtype), "bits": bits(r)}
                            for o, r in zip(sess.get_outputs(), outs)}
        except Exception as e:
            entry["ort_error"] = type(e).__name__ + ": " + str(e).splitlines()[0][:200]
        manifest["variants"][v] = entry
    # Erf sweep: ORT's output, and erf in f64 rounded once to f32 (the
    # reference the ULP bound is stated against).
    import math
    data = build_erf_sweep().SerializeToString()
    with open(os.path.join(OUT, "erf_sweep.onnx"), "wb") as f:
        f.write(data)
    xs = erf_sweep_inputs()
    ref = np.array([math.erf(float(x)) if not math.isnan(x) else math.nan for x in xs], np.float64).astype(np.float32)
    manifest["erf_sweep"] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "X_bits": bits(xs),
        "ort_bits": bits(session(data).run(["Y"], {"X": xs})[0]),
        "f64_ref_bits": bits(ref),
    }
    # Cos/Sin sweep: ORT's outputs and f64 references rounded once to f32.
    data = build_trig_sweep().SerializeToString()
    with open(os.path.join(OUT, "trig_sweep.onnx"), "wb") as f:
        f.write(data)
    xs = trig_sweep_inputs()
    oc, os_ = session(data).run(["C", "S"], {"X": xs})
    manifest["trig_sweep"] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "X_bits": bits(xs),
        "ort_cos_bits": bits(oc),
        "ort_sin_bits": bits(os_),
        "f64_cos_bits": bits(np.cos(xs.astype(np.float64)).astype(np.float32)),
        "f64_sin_bits": bits(np.sin(xs.astype(np.float64)).astype(np.float32)),
    }
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
        f.write(dumps_compact(manifest) + "\n")
    print(json.dumps({k: list(v.keys()) for k, v in manifest["variants"].items()}))


if __name__ == "__main__":
    main()
