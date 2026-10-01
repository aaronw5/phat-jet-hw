"""Accuracy-recovery fine-tune at FROZEN cost -- explicit loop.

Contract: freeze every quantizer (LUT exactly invariant), beta=0, train only the
weights. Any accuracy gained is free because the arithmetic is unchanged.

Why an explicit loop instead of model.fit: fit() produced loss=nan from epoch 0
on these deeply-compressed checkpoints even at LR=3e-7 with clipnorm, while a
manual 400-step walk over the same data at the same LR stayed finite (loss ~4.7,
acc ~79%). Rather than chase that, we step manually and simply SKIP any batch
whose loss is non-finite, restoring pre-step weights when it happens.
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
TAG = os.environ.get("RUN_TAG", "rec3"); LR = float(os.environ.get("LR", "3e-7"))
EPO = int(os.environ.get("EPOCHS", "25")); BS = 1024
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xtr, ytr, xva, yva = load_jets(n=N)

m = build_qat_model(aggregation="mean", gmp_bounds=3.7, gmp_edges=_gmp_edges(),
                    num_particles=N, patch_size=PS)
m.load_weights(CK)
for lyr in m._flatten_layers():
    if isinstance(lyr, Quantizer): lyr.trainable = False
    if getattr(lyr, "_beta", None) is not None:
        lyr._beta.assign(keras.ops.convert_to_tensor(0.0, dtype=lyr._beta.dtype))
assert not any(isinstance(l, Quantizer) and l.trainable_variables
               for l in m._flatten_layers()), "quantizer still trainable -- cost NOT frozen"
m.compile(optimizer=keras.optimizers.Adam(LR, clipnorm=1.0),
          loss="categorical_crossentropy", metrics=["accuracy"])
base = float(m.evaluate(xva, yva, verbose=0, batch_size=8192)[1])
print(f"[{TAG}] baseline {base*100:.4f}%  LR={LR:g}  (quantizers frozen, beta=0)", flush=True)

OUT = f"{ROOT}/recovered_{TAG}.weights.h5"
best = base; hist = []; rng = np.random.default_rng(0); nskip = 0
for ep in range(EPO):
    idx = rng.permutation(len(xtr))
    for step in range(len(idx) // BS):
        s = idx[step*BS:(step+1)*BS]
        snap = [v.numpy() for v in m.trainable_variables]
        L = float(m.train_on_batch(xtr[s], ytr[s], return_dict=True)["loss"])
        if not np.isfinite(L):
            for v, w in zip(m.trainable_variables, snap): v.assign(w)
            nskip += 1
    a = float(m.evaluate(xva, yva, verbose=0, batch_size=8192)[1]); hist.append(a)
    star = ""
    if a > best:
        best = a; m.save_weights(OUT); star = " <- saved"
    print(f"  [{TAG}] ep{ep:>2} {a*100:.4f}% ({(a-base)*100:+.4f}pt) skipped={nskip}{star}", flush=True)
    if os.path.exists(f"{ROOT}/STOP_{TAG}"): print(f"  [{TAG}] stop file", flush=True); break
json.dump({"tag": TAG, "ckpt": os.path.basename(CK), "N": N, "patch_size": PS, "lr": LR,
           "baseline_acc": base, "best_acc": best, "gain_pt": (best-base)*100,
           "batches_skipped": nskip, "history": hist},
          open(f"{ROOT}/recover_{TAG}.json", "w"), indent=1)
print(f"[{TAG}] DONE {base*100:.4f}% -> {best*100:.4f}%  gain {(best-base)*100:+.4f}pt  skipped {nskip}", flush=True)
