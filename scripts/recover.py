"""Accuracy-recovery fine-tune at FROZEN cost.

Every frontier point so far was captured while the EBOPs penalty was actively
pushing bitwidths/channels down -- the model never got to re-converge at the
sparsity it had just reached. This script does the complementary half:

  1. load a frontier checkpoint,
  2. FREEZE every quantizer bitwidth variable (so LUT is exactly invariant),
  3. set beta=0 (no cost penalty at all),
  4. train the remaining weights to convergence.

LUT cannot increase (the arithmetic is fixed), so any accuracy gained is free.
"""
import os, sys, json, shutil
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, keras
from train_qat import build_qat_model, StopFile
from extract_hw import _gmp_edges
from preprocess import load_jets

CK   = sys.argv[1]
N    = int(sys.argv[2]); PS = int(sys.argv[3])
TAG  = os.environ.get("RUN_TAG", "recover")
EPO  = int(os.environ.get("EPOCHS", "60"))
LR   = float(os.environ.get("LR", "3e-4"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

xtr, ytr, xva, yva = load_jets(n=N)

m = build_qat_model(aggregation="mean", gmp_bounds=3.7, gmp_edges=_gmp_edges(),
                    num_particles=N, patch_size=PS)
m.load_weights(CK)

# Freeze the quantizers at the LAYER level. Per-variable `trainable=False` is
# ignored in Keras 3 (the layer owns it), and the quantizer vars are named
# f/i/k/b -- not _f/_i/_k/_b. Getting this wrong lets the MonoL1 bitwidth
# regularizers keep shrinking bitwidths with beta=0 removing all counter-
# pressure, so values wrap and accuracy collapses to chance in one epoch.
from hgq.quantizer import Quantizer
nq = nv = 0
for lyr in m._flatten_layers():
    if isinstance(lyr, Quantizer):
        nv += len(lyr.trainable_variables); lyr.trainable = False; nq += 1
kept = m.trainable_variables
print(f"[{TAG}] froze {nq} quantizer layers ({nv} bitwidth vars) -> LUT is invariant")
print(f"[{TAG}] training {len(kept)} weight tensors ({sum(int(v.numpy().size) for v in kept):,} scalars)")
assert not any(isinstance(l, Quantizer) and l.trainable_variables for l in m._flatten_layers()), \
    "a quantizer is still trainable -- cost would not be frozen"

# beta=0: no EBOPs penalty (belt and braces; the quantizers are frozen anyway)
nb = 0
for lyr in m._flatten_layers():
    if getattr(lyr, "_beta", None) is not None:
        lyr._beta.assign(keras.ops.convert_to_tensor(0.0, dtype=lyr._beta.dtype)); nb += 1
print(f"[{TAG}] beta zeroed on {nb} layers")

m.compile(optimizer=keras.optimizers.Adam(LR),
          loss="categorical_crossentropy", metrics=["accuracy"])
base = m.evaluate(xva, yva, verbose=0, batch_size=4096)
print(f"[{TAG}] baseline val_acc={base[1]:.4f}   (recovering at frozen cost)")

class Best(keras.callbacks.Callback):
    def __init__(s): s.best = base[1]; s.out = f"{ROOT}/recovered_{TAG}.keras"
    def on_epoch_end(s, ep, logs=None):
        a = logs.get("val_accuracy", 0)
        if a > s.best:
            s.best = a; s.model.save_weights(s.out)
            print(f"  [{TAG}] BEST val_acc={a:.4f} ep{ep} -> {os.path.basename(s.out)}", flush=True)

m.fit(xtr, ytr, validation_data=(xva, yva), epochs=EPO, batch_size=1024, verbose=2,
      callbacks=[Best(), StopFile(TAG, ROOT),
                 keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max",
                                                   factor=0.5, patience=6, min_lr=1e-6),
                 keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                               patience=15, restore_best_weights=True)])
json.dump({"tag": TAG, "ckpt": CK, "N": N, "patch_size": PS,
           "baseline_acc": float(base[1])}, open(f"{ROOT}/recover_{TAG}.json", "w"), indent=1)
