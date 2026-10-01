"""Fast LR scan for the recovery fine-tune.

Prior finding: LR=3e-4 collapses accuracy within one epoch -- at these coarse
bitwidths the weight quantization step is O(1) and that LR walks weights off
the representable grid. Scan DOWN from there using partial epochs so the whole
sweep finishes in minutes rather than hours.
"""
import os, sys
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, keras
from train_qat import build_qat_model
from extract_hw import _gmp_edges
from preprocess import load_jets
from hgq.layers.core.base import Quantizer

CK = sys.argv[1]; N = int(sys.argv[2]); PS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xtr, ytr, xva, yva = load_jets(n=N)
SPE = int(os.environ.get("STEPS", "120"))   # partial epoch

def fresh():
    m = build_qat_model(aggregation="mean", gmp_bounds=3.7, gmp_edges=_gmp_edges(),
                        num_particles=N, patch_size=PS)
    m.load_weights(CK)
    for lyr in m._flatten_layers():
        if isinstance(lyr, Quantizer): lyr.trainable = False
        if getattr(lyr, "_beta", None) is not None:
            lyr._beta.assign(keras.ops.convert_to_tensor(0.0, dtype=lyr._beta.dtype))
    assert not any(isinstance(l, Quantizer) and l.trainable_variables
                   for l in m._flatten_layers()), "quantizer still trainable"
    return m

m0 = fresh(); m0.compile(loss="categorical_crossentropy", metrics=["accuracy"])
base = m0.evaluate(xva, yva, verbose=0, batch_size=8192)[1]
print(f"BASELINE val_acc={base*100:.3f}%   ckpt={os.path.basename(CK)}", flush=True)

for lr in [1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7]:
    m = fresh()
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="categorical_crossentropy",
              metrics=["accuracy"])
    accs = []
    for _ in range(3):
        m.fit(xtr, ytr, epochs=1, steps_per_epoch=SPE, batch_size=1024, verbose=0)
        accs.append(m.evaluate(xva, yva, verbose=0, batch_size=8192)[1])
    best = max(accs)
    print(f"lr={lr:>8.0e}  " + " ".join(f"{a*100:7.3f}%" for a in accs)
          + f"   best {best*100:7.3f}%  delta {(best-base)*100:+6.3f}", flush=True)
