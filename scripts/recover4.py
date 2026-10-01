"""Recovery fine-tune at FROZEN cost -- sub-epoch checkpointing.

Findings that shaped this:
  * Only LR <= 1e-6 is stable; 3e-7 gains monotonically (logs/lrscan2.log).
  * The LR scan reached 80.958% using 120-step partial epochs, BETTER than the
    full-epoch run's 80.8954% -- accuracy peaks EARLY within an epoch and then
    decays, so evaluating only at epoch boundaries misses the peak.
  * NaN batches poison Adam's moment estimates permanently; restoring weights is
    not enough. So on NaN we rebuild the optimizer state from scratch.

Evaluate every EVAL_STEPS batches and keep the best.
"""
import os, sys, json
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, keras
from train_qat import build_qat_model
from extract_hw import _gmp_edges
from preprocess import load_jets
from hgq.layers.core.base import Quantizer

CK = sys.argv[1]; N = int(sys.argv[2]); PS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
TAG = os.environ.get("RUN_TAG", "rec4"); LR = float(os.environ.get("LR", "3e-7"))
STEPS = int(os.environ.get("EVAL_STEPS", "40")); TOT = int(os.environ.get("TOTAL_STEPS", "2400"))
BS = 1024
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xtr, ytr, xva, yva = load_jets(n=N)

def make():
    m = build_qat_model(aggregation="mean", gmp_bounds=3.7, gmp_edges=_gmp_edges(),
                        num_particles=N, patch_size=PS)
    for lyr in m._flatten_layers():
        if isinstance(lyr, Quantizer): lyr.trainable = False
        if getattr(lyr, "_beta", None) is not None:
            lyr._beta.assign(keras.ops.convert_to_tensor(0.0, dtype=lyr._beta.dtype))
    assert not any(isinstance(l, Quantizer) and l.trainable_variables
                   for l in m._flatten_layers()), "quantizer still trainable -- cost NOT frozen"
    m.compile(optimizer=keras.optimizers.Adam(LR, clipnorm=1.0),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m

m = make(); m.load_weights(CK)
base = float(m.evaluate(xva, yva, verbose=0, batch_size=8192)[1])
print(f"[{TAG}] baseline {base*100:.4f}%  LR={LR:g}  eval every {STEPS} steps", flush=True)

OUT = f"{ROOT}/recovered_{TAG}.weights.h5"
best = base; rng = np.random.default_rng(1); hist = []; nreset = 0
idx = rng.permutation(len(xtr)); ptr = 0
for blk in range(TOT // STEPS):
    for _ in range(STEPS):
        if ptr + BS > len(idx): idx = rng.permutation(len(xtr)); ptr = 0
        s = idx[ptr:ptr+BS]; ptr += BS
        L = float(m.train_on_batch(xtr[s], ytr[s], return_dict=True)["loss"])
        if not np.isfinite(L):
            # NaN poisons Adam permanently -> reload best weights, fresh optimizer
            nreset += 1
            m = make(); m.load_weights(OUT if os.path.exists(OUT) else CK)
            break
    a = float(m.evaluate(xva, yva, verbose=0, batch_size=8192)[1]); hist.append(a)
    tag = ""
    if a > best:
        best = a; m.save_weights(OUT); tag = " <- saved"
    print(f"  [{TAG}] step {(blk+1)*STEPS:>5} {a*100:.4f}% ({(a-base)*100:+.4f}pt) resets={nreset}{tag}", flush=True)
    if os.path.exists(f"{ROOT}/STOP_{TAG}"): break
json.dump({"tag": TAG, "ckpt": os.path.basename(CK), "N": N, "patch_size": PS, "lr": LR,
           "baseline_acc": base, "best_acc": best, "gain_pt": (best-base)*100,
           "opt_resets": nreset, "eval_steps": STEPS, "history": hist},
          open(f"{ROOT}/recover_{TAG}.json", "w"), indent=1)
print(f"[{TAG}] DONE {base*100:.4f}% -> {best*100:.4f}%  gain {(best-base)*100:+.4f}pt", flush=True)
