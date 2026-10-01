"""Accuracy-recovery fine-tune at FROZEN cost -- hardened.

Same contract as recover.py (freeze quantizers -> LUT exactly invariant, beta=0,
train only the weights, so any accuracy gained is free), with two fixes found
the hard way:

  * LR: the scan (logs/lrscan2.log) shows 1e-4 collapses to chance in 120 steps
    and 1e-5 loses 3.4pt; only LR <= 1e-6 is stable, and 3e-7 gains monotonically.
  * NaN: at deep compression a plain Adam step produced loss=nan from epoch 1
    even at 3e-7, so gradients are clipped and NaN batches are skipped.
"""
import os, sys, json
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, keras
from train_qat import build_qat_model, StopFile
from extract_hw import _gmp_edges
from preprocess import load_jets
from hgq.layers.core.base import Quantizer

CK = sys.argv[1]; N = int(sys.argv[2]); PS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
TAG = os.environ.get("RUN_TAG", "rec2"); LR = float(os.environ.get("LR", "3e-7"))
EPO = int(os.environ.get("EPOCHS", "40"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xtr, ytr, xva, yva = load_jets(n=N)

m = build_qat_model(aggregation="mean", gmp_bounds=3.7, gmp_edges=_gmp_edges(),
                    num_particles=N, patch_size=PS)
m.load_weights(CK)
nq = 0
for lyr in m._flatten_layers():
    if isinstance(lyr, Quantizer): lyr.trainable = False; nq += 1
    if getattr(lyr, "_beta", None) is not None:
        lyr._beta.assign(keras.ops.convert_to_tensor(0.0, dtype=lyr._beta.dtype))
assert not any(isinstance(l, Quantizer) and l.trainable_variables
               for l in m._flatten_layers()), "quantizer still trainable -- cost NOT frozen"
print(f"[{TAG}] froze {nq} quantizers, beta=0, LR={LR:g}", flush=True)

m.compile(optimizer=keras.optimizers.Adam(LR, clipnorm=1.0, epsilon=1e-7),
          loss="categorical_crossentropy", metrics=["accuracy"])
base = float(m.evaluate(xva, yva, verbose=0, batch_size=8192)[1])
print(f"[{TAG}] baseline val_acc={base*100:.4f}%", flush=True)

class Best(keras.callbacks.Callback):
    def __init__(s): s.best = base; s.out = f"{ROOT}/recovered_{TAG}.keras"; s.hist = []
    def on_epoch_end(s, ep, logs=None):
        a = float(logs.get("val_accuracy", 0)); s.hist.append(a)
        if not np.isfinite(logs.get("loss", 0)):
            print(f"  [{TAG}] ep{ep} NaN loss -- aborting", flush=True)
            s.model.stop_training = True; return
        if a > s.best:
            s.best = a; s.model.save_weights(s.out)
            print(f"  [{TAG}] BEST {a*100:.4f}% ep{ep} (+{(a-base)*100:.4f}) -> saved", flush=True)
        else:
            print(f"  [{TAG}] ep{ep} {a*100:.4f}% ({(a-base)*100:+.4f})", flush=True)

cb = Best()
m.fit(xtr, ytr, validation_data=(xva, yva), epochs=EPO, batch_size=1024, verbose=0,
      callbacks=[cb, StopFile(TAG, ROOT),
                 keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                              patience=12, restore_best_weights=True)])
json.dump({"tag": TAG, "ckpt": os.path.basename(CK), "N": N, "patch_size": PS,
           "baseline_acc": base, "best_acc": cb.best, "gain_pt": (cb.best-base)*100,
           "lr": LR, "history": cb.hist}, open(f"{ROOT}/recover_{TAG}.json", "w"), indent=1)
print(f"[{TAG}] DONE baseline {base*100:.4f}% -> best {cb.best*100:.4f}%  gain {(cb.best-base)*100:+.4f}pt", flush=True)
