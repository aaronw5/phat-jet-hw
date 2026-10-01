"""
GMP grid shootout: short QAT runs differing ONLY in the GMP grid.

Physics-informed candidates (edges in RAW eta/phi units, divided by the
robust-scale IQR at build time). See docs/grid_study.md for the derivation:
  * uniform8  — 8x8, +/-0.4, d=0.1 (current baseline; origin on a bin BOUNDARY,
                so the jet core pT is split across up to 4 cells)
  * core7     — 7x7 non-uniform, +/-0.4: fine 0.08 bins in the core (|x|<0.2,
                where 95% of jet pT lives), coarse 0.2 edge bins. Odd count ->
                origin is a CELL CENTER: the leading-prong core (median |r|=0.04)
                lands in one cell instead of 4. 49 cells = 23% cheaper scatter/
                gather than 64.
  * uniform9  — 9x9 uniform +/-0.45, d=0.1: paper's proven d=0.1 resolution,
                odd count (cell-centered core), edge bins pushed to 0.45 so
                clamping is negligible. 81 cells = 27% costlier than 64.

Usage:  python grid_shootout.py <name>     (name in GRIDS)
Writes: grid_shootout/<name>_hist.json
"""
import os, sys, json

os.environ.setdefault("KERAS_BACKEND", "jax")
# Threads are set by the launcher (SHOOTOUT_THREADS); default 4/run for 3-way parallel.
_thr = os.environ.get("SHOOTOUT_THREADS", "4")
os.environ.setdefault("XLA_FLAGS", f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={_thr}")

import numpy as np
import keras
import random

np.random.seed(42); random.seed(42)
keras.utils.set_random_seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_qat import build_qat_model, cosine_restarts, LR
from hgq.utils.sugar import BetaScheduler, FreeEBOPs, PieceWiseSchedule

EPOCHS = 25
SUBSET = 300_000
BATCH = 2790

# raw-unit edges; converted to robust-scaled units below
GRIDS = {
    "uniform8": np.linspace(-0.4, 0.4, 9),
    "core7":    np.array([-0.40, -0.20, -0.12, -0.04, 0.04, 0.12, 0.20, 0.40]),
    "uniform9": np.linspace(-0.45, 0.45, 10),
}

def main(name):
    edges_raw = GRIDS[name]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stats = json.load(open(os.path.join(root, "data/robust_stats.json")))
    iqr_eta = stats[1]["iqr"]  # eta and phi iqr agree to <0.2%; use per-feature anyway
    # model uses ONE edge list for both coords -> use mean iqr (delta < 1e-3 bins)
    iqr = 0.5 * (stats[1]["iqr"] + stats[2]["iqr"])
    edges = (edges_raw / iqr).tolist()
    print(f"[{name}] raw edges {edges_raw.tolist()} -> scaled {['%.3f' % e for e in edges]}")

    from preprocess import load_jets
    x_train, y_train, x_val, y_val = load_jets(stats=stats, root=root)
    x_val, y_val = x_val[:100_000], y_val[:100_000]
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x_train))[:SUBSET]
    x_train, y_train = x_train[idx], y_train[idx]

    model = build_qat_model(gmp_edges=edges)
    beta_sched = BetaScheduler(PieceWiseSchedule(
        [(0, 2e-8, "linear"), (int(EPOCHS * 2000 / 7000), 3e-7, "log"), (EPOCHS, 3e-6, "constant")]))
    lr_cb = keras.callbacks.LearningRateScheduler(cosine_restarts(LR, max(int(500 * EPOCHS / 7000), 10)))
    model.compile(optimizer=keras.optimizers.Adam(LR),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"], steps_per_execution=4)
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                     batch_size=BATCH, epochs=EPOCHS, verbose=2,
                     callbacks=[FreeEBOPs(), lr_cb, beta_sched])
    os.makedirs(os.path.join(root, "grid_shootout"), exist_ok=True)
    out = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    out["edges_raw"] = edges_raw.tolist()
    out["edges_scaled"] = edges
    json.dump(out, open(os.path.join(root, f"grid_shootout/{name}_hist.json"), "w"))
    model.save_weights(os.path.join(root, f"grid_shootout/{name}.weights.h5"))
    print(f"[{name}] done. best val_acc={max(out['val_accuracy']):.4f}")

if __name__ == "__main__":
    main(sys.argv[1])
