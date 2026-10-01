"""
Float pretrain of the EXACT hardware architecture (LN-free, core7 GMP grid).

Why: jsc150 (JEDI-Linear's recipe) initializes QAT from a trained float model.
Run 3 tried this with the user's public checkpoint and COLLAPSED because that
model was trained WITH LayerNorms -> huge activation ranges -> EBOPs 5.9e9 ->
beta penalty crushed the net. Training our own LN-free float model removes that
mismatch entirely: identical layer names, identical topology, 1:1 weight transfer.

Usage: python float_pretrain.py [epochs]
Writes: float_phatjet_core7.keras
"""
import os, sys, json

os.environ.setdefault("KERAS_BACKEND", "jax")
import numpy as np, keras, random

np.random.seed(42); random.seed(42); keras.utils.set_random_seed(42)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phat_jet_k3 import build_phat_jet_k3
from grid_shootout import GRIDS

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
BATCH = 2790
LR = 3e-3
GRID = os.environ.get("GRID_NAME", "core7")


def load_data(root):
    stats = json.load(open(f"{root}/data/robust_stats.json"))
    from preprocess import load_jets
    xt, yt, xv, yv = load_jets(stats=stats, root=root)
    iqr = 0.5 * (stats[1]["iqr"] + stats[2]["iqr"])
    return xt, yt, xv, yv, (GRIDS[GRID] / iqr).tolist()


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xt, yt, xv, yv, edges = load_data(root)
    print(f"[float:{GRID}] edges {['%.3f' % e for e in edges]}", flush=True)
    model = build_phat_jet_k3(quantized=False, use_ln=False, aggregation="mean", gmp_edges=edges)
    print(f"[float] params {model.count_params()}", flush=True)
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"], steps_per_execution=8,
    )
    cbs = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=6, min_lr=1e-5, verbose=1),
        keras.callbacks.ModelCheckpoint(f"{root}/float_phatjet_{GRID}.keras", monitor="val_accuracy",
                                        mode="max", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=15, restore_best_weights=True),
    ]
    h = model.fit(xt, yt, validation_data=(xv, yv), batch_size=BATCH, epochs=EPOCHS, verbose=2, callbacks=cbs)
    json.dump({k: [float(v) for v in vs] for k, vs in h.history.items()},
              open(f"{root}/float_{GRID}_hist.json", "w"))
    print(f"[float] best val_acc {max(h.history['val_accuracy']):.4f}", flush=True)


if __name__ == "__main__":
    main()
