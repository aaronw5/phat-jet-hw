"""Float PHAT-JeT at reduced N, warm-started from the trained N=128 float model.

JEDI-Linear trains a separate model at each N (their Table II sweeps
N in {8,16,32,64,128}). Truncating our trained N=128 model instead of training
at N collapses it -- measured: 78.6% -> 63.3% at N=32, 46.8% at N=16 -- because
the model uses the soft-particle tail. So we must train at each N to compete on
their axis.

Warm start is exact and free: in FLOAT mode every weight is N-agnostic (the GMP
bin indicators are weightless Lambda layers; dense/attention projections act on
the feature axis with weights shared across particles and patches). Verified by
assertion below: shapes match 1:1 between the N=128 and N<128 float models, so
no slicing is involved. Only the QAT models have (1, N) quantizer tensors.

Usage:  N_PARTICLES=32 python float_n.py [epochs]
Writes: float_phatjet_<grid>_n<N>.keras, float_<grid>_n<N>_hist.json
"""
import os, sys, json
os.environ.setdefault("KERAS_BACKEND", "jax")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, keras, random
np.random.seed(42); random.seed(42); keras.utils.set_random_seed(42)
from phat_jet_k3 import build_phat_jet_k3
from float_pretrain import load_data

N = int(os.environ.get("N_PARTICLES", "32"))
PS = int(os.environ.get("PATCH_SIZE", "8"))
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
BATCH = 2790
LR = float(os.environ.get("LR", "1e-3"))   # fine-tune, not from scratch
GRID = os.environ.get("GRID_NAME", "core7")

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xt, yt, xv, yv, edges = load_data(root)
xt, xv = xt[:, :N, :], xv[:, :N, :]        # pT-sorted data -> top-N by pT
print(f"[float n={N}] data {xt.shape} patch_size={PS}", flush=True)

DM = int(os.environ.get("D_MODEL", "16")); NH = int(os.environ.get("NUM_HEADS", "4"))
print(f"[float n={N}] d_model={DM} num_heads={NH}", flush=True)
PAR = bool(int(os.environ.get("PARALLEL_ATTN", 0)))
model = build_phat_jet_k3(quantized=False, use_ln=False, aggregation="mean",
                          gmp_edges=edges, num_particles=N, patch_size=PS,
                          d_model=DM, num_heads=NH, parallel_attn=PAR)
WARM = os.environ.get("WARM", "128")  # "128" | "self" | "none"
_donor = {"128": f"{root}/float_phatjet_{GRID}.keras",
          "self": f"{root}/float_phatjet_{GRID}_n{N}.keras"}.get(WARM)
if WARM == "none":
    print(f"[float n={N}] FROM SCRATCH (no warm start)", flush=True)
    sw = None
else:
    src = build_phat_jet_k3(quantized=False, use_ln=False, aggregation="mean",
                            gmp_edges=edges,
                            num_particles=(128 if WARM == "128" else N),
                            patch_size=(8 if WARM == "128" else PS),
                            parallel_attn=PAR)
    src.load_weights(_donor)
    sw = [np.asarray(w) for w in src.weights]
    assert len(sw) == len(model.weights), (len(sw), len(model.weights))
    for w, _s in zip(model.weights, sw):
        assert tuple(w.shape) == _s.shape, f"float weights NOT N-agnostic: {w.path} {tuple(w.shape)} vs {_s.shape}"
        w.assign(_s)
    print(f"[float n={N}] warm-started all {len(sw)} weights from {os.path.basename(_donor)}", flush=True)

model.compile(optimizer=keras.optimizers.Adam(LR),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"], steps_per_execution=8)
print(f"[float n={N}] warm-start val_acc {model.evaluate(xv, yv, batch_size=4096, verbose=0)[1]:.4f}", flush=True)

TAG = os.environ.get("TAG", "")
ck = f"{root}/float_phatjet_{GRID}_n{N}{TAG}.keras"

class StopFile(keras.callbacks.Callback):
    """Cooperative stop: `touch STOP_<RUN_TAG>` (or STOP_ALL) ends training at the
    next epoch boundary. The sandbox cannot send signals to processes it did not
    spawn (EPERM even at matching uid), so signals are not an option for stopping
    a run launched in an earlier cell -- this file check is.
    """
    def __init__(self, tag, root):
        super().__init__()
        self.paths = [os.path.join(root, f"STOP_{tag}"), os.path.join(root, "STOP_ALL")]
    def on_epoch_end(self, epoch, logs=None):
        for p in self.paths:
            if os.path.exists(p):
                print(f"[stop] {os.path.basename(p)} found -> stopping at epoch {epoch}", flush=True)
                self.model.stop_training = True

cbs = [StopFile(f'float_n{N}{TAG}', root),
       keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5,
                                         patience=int(os.environ.get("RLR_PAT","6")), min_lr=1e-6, verbose=1),
       keras.callbacks.ModelCheckpoint(ck, monitor="val_accuracy", mode="max",
                                       save_best_only=True, verbose=1),
       keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=int(os.environ.get("ES_PAT","18")),
                                     restore_best_weights=True)]
h = model.fit(xt, yt, validation_data=(xv, yv), batch_size=BATCH, epochs=EPOCHS,
              verbose=2, callbacks=cbs)
json.dump({k: [float(v) for v in vs] for k, vs in h.history.items()},
          open(f"{root}/float_{GRID}_n{N}{TAG}_hist.json", "w"))
print(f"[float n={N}] BEST val_acc {max(h.history['val_accuracy']):.4f}", flush=True)
