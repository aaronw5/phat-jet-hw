"""Converge AT the device-fitting operating point instead of compressing past it.

WHY THIS RUN EXISTS
-------------------
run6/run8 used a beta schedule that compresses jsc150's 7000-epoch EBOPs ramp
into 1000 epochs. Measured consequence (run6): accuracy peaks per unit cost at
epoch ~140 (78.64% @ 927k EBOPs, 1.19M LUT, which FITS the VU13P at 69%
occupancy) and then falls off a cliff -- 76.23% by ep208, 71.21% by ep233,
58.40% by ep579 -- while EBOPs keeps dropping to 161k. That extra compression
buys nothing: the design already fit at epoch 140.

Meanwhile the float ceiling for this architecture/grid is 81.59%, so at ep140
quantization is costing 2.95 points. That gap is unconverged QAT, not a
hardware limit.

So: start FROM the ep140 checkpoint, HOLD beta at its ep140 value (1.58e-7),
and spend the whole budget converging at that operating point. No architecture
change, no new attention -- same model, same grid, better training recipe.

Expected outcome: EBOPs roughly flat (beta constant), accuracy climbing from
78.64% toward the 81.59% float ceiling.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("GRID_NAME", "core7")
import numpy as np, keras
from hgq.utils.sugar import BetaScheduler, FreeEBOPs, ParetoFront, PieceWiseSchedule
import train_qat as T

RUN = os.environ.get("HOLD_TAG", "run9_hold")
EPOCHS = int(os.environ.get("HOLD_EPOCHS", 400))
BETA = float(os.environ.get("HOLD_BETA", 1.58e-7))
INIT = os.environ.get("HOLD_INIT", "ckpt_snapshots_run6/epoch=140-val_acc=0.7864-ebops=926777.keras")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from preprocess import gmp_edges_scaled, load_jets
xt, yt, xv, yv = load_jets()

edges = gmp_edges_scaled(os.environ["GRID_NAME"])

model = T.build_qat_model(aggregation="mean", gmp_bounds=3.7, gmp_edges=edges)
model.load_weights(os.path.join(root, INIT), skip_mismatch=False)
P = np.concatenate([np.array(model(xv[i:i+8192], training=False)) for i in range(0, len(xv), 8192)])
print(f"[{RUN}] resumed from {os.path.basename(INIT)}: val_acc {(P.argmax(-1)==yv.argmax(-1)).mean()*100:.2f}%", flush=True)
print(f"[{RUN}] holding beta={BETA:.3e} for {EPOCHS} epochs (no further compression)", flush=True)

pdir = os.path.join(root, f"pareto_{RUN}")
os.makedirs(pdir, exist_ok=True)
cbs = [
    BetaScheduler(PieceWiseSchedule([(0, BETA, "constant"), (EPOCHS, BETA, "constant")])),
    T.ClampSoftmaxExpBits(T.EXP_I_MAX),
    FreeEBOPs(),
    ParetoFront(pdir, ["val_accuracy", "ebops"], [1, -1],
                fname_format="epoch={epoch}-val_acc={val_accuracy:.4f}-ebops={ebops}.keras",
                enable_if=lambda x: x["val_accuracy"] > 0.6),
    keras.callbacks.CSVLogger(os.path.join(root, f"{RUN}_log.csv"), append=True),
    keras.callbacks.ModelCheckpoint(os.path.join(root, f"{RUN}_latest.weights.h5"), save_weights_only=True),
    # Lower, flatter LR than the main run: we are converging, not exploring.
    keras.callbacks.LearningRateScheduler(T.cosine_restarts(5e-4, 40)),
]
model.compile(optimizer=keras.optimizers.Adam(5e-4),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(xt, yt, validation_data=(xv, yv), epochs=EPOCHS, batch_size=T.BATCH,
          callbacks=cbs, verbose=2)
print(f"[{RUN}] DONE", flush=True)
