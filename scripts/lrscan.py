"""LR scan for the recovery fine-tune (scripts/recover.py).

Why: at floor-arm bitwidths the WEIGHT quantization step is coarse (median
2^0 = 1.0). LR=3e-4 over ~570 steps/epoch walks weights off that grid and
accuracy collapses even though the quantizers are verifiably frozen. This
scan finds the largest LR that does not drift.
"""
import os, sys
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, keras
from hgq.layers.core.base import Quantizer
from train_qat import build_qat_model
from extract_hw import _gmp_edges
from preprocess import load_jets

CK = sys.argv[1]; N = int(sys.argv[2]); PS = int(sys.argv[3])
xtr, ytr, xva, yva = load_jets(n=N)

for lr in [3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7]:
    m = build_qat_model(aggregation="mean", gmp_bounds=3.7,
                        gmp_edges=_gmp_edges(), num_particles=N, patch_size=PS)
    m.load_weights(CK)
    nq = 0
    for l in m._flatten_layers():
        if isinstance(l, Quantizer): l.trainable = False; nq += 1
    assert not any(isinstance(l, Quantizer) and l.trainable_variables
                   for l in m._flatten_layers()), "quantizer freeze failed"
    nb = 0
    for l in m._flatten_layers():
        if hasattr(l, "_beta"):
            l._beta.assign(keras.ops.zeros_like(l._beta)); nb += 1
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    a0 = m.evaluate(xva, yva, verbose=0, batch_size=8192)[1]
    accs = []
    for _ in range(3):
        m.fit(xtr, ytr, epochs=1, batch_size=1024, verbose=0)
        accs.append(m.evaluate(xva, yva, verbose=0, batch_size=8192)[1])
    print(f"lr={lr:.0e}  frozen_q={nq} beta0={nb}  base {a0*100:6.2f}%  -> "
          + "  ".join(f"{a*100:6.2f}%" for a in accs), flush=True)
