"""
Continuation of run 4 (train_qat.py): 200 more epochs from qat_final.keras with
beta CONSTANT at 3e-6 (the plateau value) and the same cosine-restart LR schedule.
Rationale: jsc150 spends most of its 7000 epochs with beta near its final value —
that's when accuracy recovers at low EBOPs. Our 200-epoch ramp only just reached
3e-6 at the end. Pareto checkpoints go to the SAME pareto/ dir so the frontier merges.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KERAS_BACKEND", "jax")

import json

import keras
import numpy as np

from train_qat import BATCH, LR, build_qat_model, cosine_restarts
from hgq.utils.sugar import BetaScheduler, FreeEBOPs, ParetoFront, PieceWiseSchedule

EPOCHS = 1000
BETA = 3e-6


def main():
    from preprocess import load_jets
    x_train, y_train, x_val, y_val = load_jets()

    model = build_qat_model()
    model.load_weights("qat_final.keras")
    print("resumed from qat_final.keras (end of run 4)")

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    ebops = FreeEBOPs()
    beta_sched = BetaScheduler(PieceWiseSchedule([(0, BETA, "constant")]))
    lr_cb = keras.callbacks.LearningRateScheduler(cosine_restarts(LR, max(int(500 * 200 / 7000), 10)))
    pareto = ParetoFront(
        "pareto",
        metrics=["val_accuracy", "ebops"],
        sides=[1, -1],
        fname_format="epoch=c{epoch}-val_acc={val_accuracy:.4f}-ebops={ebops:.0f}.keras",
    )

    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH,
        epochs=EPOCHS,
        verbose=2,
        callbacks=[ebops, lr_cb, beta_sched, pareto],
    )
    json.dump(
        {k: [float(v) for v in vs] for k, vs in hist.history.items()},
        open("qat_hist_continue.json", "w"),
    )
    model.save("qat_final_continue.keras")


if __name__ == "__main__":
    main()
