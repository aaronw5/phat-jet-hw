"""Localize where quantization loses accuracy.

The architecture's float ceiling is 81.59%; the quantized model sits at 78.64%.
That 2.95-point loss is larger than the 2.27-point gap to JEDI-Linear, so the
deficit is a quantization problem, not an architecture problem. This script
widens one group of quantizers at a time back toward float and reports which
restoration recovers the most accuracy -- that group is the culprit.
"""
import os, sys, json, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("GRID_NAME", "core7")
from train_qat import build_qat_model
from extract_hw import _gmp_edges
from preprocess import load_jets

CK = os.environ.get("QD_CKPT", "ckpt_snapshots_run6/epoch=140-val_acc=0.7864-ebops=926777.keras")
N = int(os.environ.get("QD_N", "0")) or None

Xv, Yv = load_jets("val")
if N: Xv, Yv = Xv[:N], Yv[:N]


def build():
    m = build_qat_model(aggregation="mean", gmp_bounds=3.7, gmp_edges=_gmp_edges())
    m.load_weights(CK, skip_mismatch=False)
    return m


def acc(mm):
    return float((np.asarray(mm(Xv, training=False)).argmax(1) == Yv.argmax(-1)).mean())


m = build()
base = acc(m)
print(f"baseline quantized acc: {base*100:.2f}%", flush=True)

# --- HGQ2 stores bitwidths as plain layer WEIGHTS named i/f/k/b, not as
# attributes on quantizer objects. Operate on layer.weights directly.
# i = integer bits, f = fractional bits, k = sign/keep flag.

def bit_weights(layer):
    # ONLY f (fractional bits). Overwriting i moves the binary point and
    # rescales every value by 2^delta -> accuracy collapses to chance (20%).
    # Adding fractional bits strictly increases precision at fixed scale.
    return [w for w in layer.weights if w.name == "f"]


probe = collections.defaultdict(int)
for l in m.layers:
    probe[type(l).__name__] += len(bit_weights(l))
print("bit tensors per layer type:", dict(probe), flush=True)


def widen(layer, add_bits=8.0):
    """add fractional bits (monotone precision increase, scale unchanged)"""
    n = 0
    for w in bit_weights(layer):
        arr = np.asarray(w)
        try:
            w.assign(arr + add_bits)
            n += 1
        except Exception:
            pass
    return n


GROUPS = {
    "softmax_exp_LUT":  lambda l: type(l).__name__ == "QUnaryFunctionLUT",
    "softmax":          lambda l: type(l).__name__ == "QSoftmax",
    "attention_einsum": lambda l: type(l).__name__ == "QEinsum",
    "dense":            lambda l: type(l).__name__ == "QDense",
    "conv_gmp":         lambda l: type(l).__name__ == "QConv2D",
    "ALL":              lambda l: True,
}

res = {"checkpoint": CK, "baseline_acc": base, "float_ceiling": 0.8159}
for g, pred in GROUPS.items():
    m2 = build()
    touched = sum(widen(l) for l in m2.layers if pred(l))
    a = acc(m2)
    res[g] = {"acc": a, "delta_pts": (a - base) * 100, "n_tensors": touched}
    print(f"{g:18} widened {touched:4d} tensors -> {a*100:6.2f}%  ({(a-base)*100:+.2f} pts)", flush=True)

json.dump(res, open("qdiag.json", "w"), indent=1)
print("DONE", flush=True)
