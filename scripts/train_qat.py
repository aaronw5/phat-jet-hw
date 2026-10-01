"""
HGQ2 quantization-aware training of PHAT-JeT (trigger-scale, 128p x 3f).

RECIPE PROVENANCE (for humans and other agents)
-----------------------------------------------
Quantizer config + callbacks copied from HGQ2-examples/jsc150/run_train.py +
model.py `get_model` — the same flow the JEDI-Linear paper used:
  * weights/bias:  q_type='kbi', b0=7, i0=0, overflow_mode='wrap',
                   fr=ir=MonoL1(l1_reg), i_decay_speed=1e-3
  * datalane:      q_type='kif', f0=7, ic=MinMax(0,12), overflow_mode='wrap',
                   fr=MonoL1(l1_reg)
  * LayerConfigScope(beta0=0) + BetaScheduler ramping the EBOPs penalty beta
    during training -> single run traces the accuracy-vs-EBOPs Pareto frontier.
  * ParetoFront callback checkpoints every (val_accuracy, ebops)-nondominated
    epoch to pareto/ as .keras files.

DEVIATIONS from the reference recipe (documented honestly):
  * 7000 epochs on GPU -> 400 epochs on M2 Max CPU; beta schedule compressed
    proportionally (linear to 3e-7 by epoch ~115, log to 3e-6 by 400).
  * We initialize from the trained float model's weights (float_phatjet.keras)
    instead of from scratch — compensates for the shorter schedule.
  * batch 4096 (CPU throughput) vs 2790.

Run in `fpga` env:  python train_qat.py
Outputs: pareto/*.keras checkpoints, qat_hist.json
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import json
import random
from math import cos, pi

import keras
import numpy as np

np.random.seed(42)
random.seed(42)

from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.constraints import MinMax
from hgq.regularizers import MonoL1
from hgq.utils.sugar import BetaScheduler, FreeEBOPs, ParetoFront, PieceWiseSchedule

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from phat_jet_k3 import build_phat_jet_k3

# Cap on integer bits of the QSoftmax exp-table input quantizer. See the long
# rationale in phat_jet_k3._softmax: i<=5 cuts total LUT -24.6% at +0.05% acc,
# i<=4 costs -3.1% acc, i<=3 costs -16.2%.
EXP_I_MAX = 5.0


class ClampSoftmaxExpBits(keras.callbacks.Callback):
    """Reassert the exp-table integer-bit cap every epoch.

    Needed because hgq 0.1.9's FixedPointQuantizerKIF ignores its own `ic`
    constraint in the WRAP training path (it applies the constraint to a
    temporary that is then discarded -- see phat_jet_k3._softmax). Without this,
    `i` tracks the running max of (max - score), which has a long tail: measured
    p99.9 = 14 but max = 189.5, so `i` climbs to ~9 and the exp table grows to
    1024+ entries whose outputs all quantize to the same near-zero value.

    Clamping at epoch end rather than per batch is sufficient: `i` only ever
    grows by observing a new maximum, so at worst one epoch's tables are wide,
    and the value is re-clamped before ebops logging / Pareto checkpointing.
    """

    def __init__(self, i_max=EXP_I_MAX):
        super().__init__()
        self.i_max = float(i_max)
        self._n = 0

    def on_train_begin(self, logs=None):
        self._n = sum(1 for _ in self._quantizers())
        print(f"[clamp] exp-table i capped at {self.i_max} on {self._n} softmax layers", flush=True)

    def _quantizers(self):
        for lyr in self.model.layers:
            tbl = getattr(lyr, "exp_table", None)
            q = getattr(getattr(tbl, "iq", None), "quantizer", None)
            if q is not None and hasattr(q, "_i"):
                yield lyr.name, q

    def on_epoch_end(self, epoch, logs=None):
        for _, q in self._quantizers():
            iv = np.array(q._i)
            if iv.max() > self.i_max:
                q._i.assign(np.minimum(iv, self.i_max))

EPOCHS = int(os.environ.get("QAT_EPOCHS", 1000))  # jsc150 uses 7000 (GPU); 1000 on CPU.
                     # All other hyperparameters below match jsc150/run_train.py exactly.
GRID_NAME = os.environ.get("GRID_NAME", "core7")  # winner of the grid shootout
RUN_TAG = os.environ.get("RUN_TAG", "run6_core7_1000ep")
TRAIN_SUBSET = None  # full train set (620k), as in the paper recipe
BATCH = 2790         # jsc150 default --batch-size
LR = 3e-3            # jsc150 default --learning-rate
L1_REG = 1e-8
# Initial bitwidths. jsc150 uses BW=7/i0=0 because it trains from scratch: weights
# and activations start << 1 and the integer bits grow on demand. Initializing from
# a float checkpoint instead, the incoming ranges are already large, and starting
# too narrow WRAPS them -> instant collapse to chance (this is what killed run 3,
# misdiagnosed then as a LayerNorm mismatch). Measured on float_phatjet_core7:
# max|w|=1.8 (needs i>=2), max|activation|=85.8 at ffn2 (needs i>=8).
# Verified post-transfer val_acc, 20k val jets:
#   bw=7,  i0_a=0 -> 0.2056   (wrap collapse, == chance)
#   bw=7,  i0_a=8 -> 0.7437
#   bw=8,  i0_a=8 -> 0.8044
#   bw=10, i0_a=8, i0_w=3 -> 0.8113  <- chosen; float model is 0.8159, so ~lossless
# beta then compresses bitwidths DOWNWARD from here, which is the intended
# direction of travel; starting wide costs nothing in the final hardware.
BW_K = int(os.environ.get('BW_K', 10))  # initial weight fractional bits
BW_A = int(os.environ.get('BW_A', 10))  # initial datalane fractional bits
I0_TRANSFER = int(os.environ.get('I0_W', 3))    # initial weight integer bits when init'ing from a float ckpt
I0_A_TRANSFER = int(os.environ.get('I0_A', 8))  # initial datalane integer bits when init'ing from a float ckpt


def cosine_restarts(lr0, first_decay_steps, alpha=1e-6):
    def schedule(epoch):
        cycle_step, cycle_len = epoch, first_decay_steps
        while cycle_step >= cycle_len:
            cycle_step -= cycle_len
        t = min(cycle_step / cycle_len, 1.0)
        return alpha + 0.5 * (lr0 - alpha) * (1 + cos(pi * t))

    return schedule


def build_qat_model(aggregation="mean", gmp_bounds=3.7, gmp_edges=None, i0=None,
                    num_particles=None, patch_size=8, d_model=None, num_heads=None):
    """num_particles: N. Defaults to env N_PARTICLES or 128.

    The architecture is N-agnostic — every learned weight is shared across the
    particle and patch axes, so only the per-element HGQ quantizer variables
    (shape (1, N)) depend on N. That is what makes the N-sweep in
    scripts/nsweep.py a weight-SLICE rather than a retrain. The hls4ml jet data
    is pT-sorted (verified: 100% of jets non-increasing in feature 0), so
    taking the first N rows is exactly JEDI-Linear's top-N-by-pT input scheme.
    """
    if num_particles is None:
        num_particles = int(os.environ.get("N_PARTICLES", "128"))
    # Width is the strongest LUT lever we have: attention cost scales ~d_model^2
    # and the head projections ~d_model. At small N the stock d_model=16 is far
    # wider than the jet needs (measured: N=8 QAT stalls at 9.6M EBOPs, where the
    # deployable regime is ~1.3e5), so sweep width rather than only beta.
    if d_model is None:
        d_model = int(os.environ.get("D_MODEL", "16"))
    if num_heads is None:
        num_heads = int(os.environ.get("NUM_HEADS", "4"))
    # i0 = initial INTEGER bits of the weight quantizer. jsc150 uses 0 because it
    # trains from scratch (weights start << 1 and i_decay_speed grows i on demand).
    # Initializing from a float checkpoint, weights reach |w|=1.8, and i0=0 with
    # overflow_mode='wrap' wraps them (1.796 -> -0.2) -> instant collapse to chance.
    # I0_TRANSFER=2 covers |w| < 4; MonoL1 + i_decay_speed decay the slack away.
    from_float = os.environ.get("QAT_INIT", "float") == "float"
    if i0 is None:
        i0 = I0_TRANSFER if from_float else 0
    i0_a = I0_A_TRANSFER if from_float else 0
    # FLOOR_BITS: minimum fractional bits any quantizer may be driven to.
    #
    # Without a floor, the EBOPs penalty (cost ~ bits_a * bits_b per multiply)
    # discovers that crushing ONE operand toward zero makes the whole product
    # nearly free. On the ep140 checkpoint this drove 10 of 36 quantizers below
    # 2 total bits -- the entire GMP path (gmp_gather/scatter/cell/dwconv/
    # pointwise) sits at ~1.0 bit, i.e. binarized, while head1/head_out keep
    # ~10. That is why quantization costs 2.95 pts against an 81.59% float
    # ceiling, which is larger than the 2.27 pt gap to JEDI-Linear.
    #
    # fc=MinMax(floor, 16) bounds the fractional bits from below so the
    # regularizer must spread cost reduction instead of sacrificing whole
    # tensors. 0 disables (original behaviour).
    floor = float(os.environ.get("FLOOR_BITS", "0"))
    fc = MinMax(floor, 16) if floor > 0 else None

    scope0 = QuantizerConfigScope(
        default_q_type="kbi",
        b0=BW_K,
        overflow_mode="wrap",
        i0=i0,
        fr=MonoL1(L1_REG),
        ir=MonoL1(L1_REG),
        i_decay_speed=1e-3,
        **({"fc": fc} if fc is not None else {}),
    )
    scope1 = QuantizerConfigScope(
        default_q_type="kif",
        place="datalane",
        overflow_mode="wrap",
        f0=BW_A,
        i0=i0_a,
        fr=MonoL1(L1_REG),
        ic=MinMax(0, 12),
        **({"fc": fc} if fc is not None else {}),
    )
    scope2 = LayerConfigScope(beta0=0)
    with scope0, scope1, scope2:
        return build_phat_jet_k3(quantized=True, use_ln=False, aggregation=aggregation,
                                 gmp_bounds=gmp_bounds, gmp_edges=gmp_edges,
                                 num_particles=num_particles, patch_size=patch_size,
                                 d_model=d_model, num_heads=num_heads,
                                 # PARALLEL_ATTN=1 runs local/patch attention as
                                 # concurrent branches: 51 -> 38 stages at N=32
                                 # (169.8 -> 126.5 ns) for +0.7% LUT.
                                 parallel_attn=bool(int(os.environ.get("PARALLEL_ATTN", 0))))


def transfer_float_weights(qmodel, float_path="float_phatjet.keras", gmp_edges=None):
    """Copy weights from the float model into same-named Q layers.

    The GMP bin indicators are Lambda layers wrapping closures, which Keras 3
    refuses to deserialize (arbitrary-code-execution guard) — so we do NOT
    load_model(). We rebuild the identical float architecture in code and load
    only the weights into it, then transfer layer-by-layer by name. Same reason
    extract_hw.py rebuilds instead of deserializing.
    """
    # Float donor is built at the Q model's N so shapes match 1:1 (float weights
    # are N-agnostic; only Q models carry (1, N) quantizer tensors).
    _npart = qmodel.input_shape[1]
    _ps = int(os.environ.get("PATCH_SIZE", "8"))
    # Width MUST match the Q model or the donor silently rebuilds at d_model=16
    # and every transferred layer shape-mismatches (or worse, partially matches).
    _dm = int(os.environ.get("D_MODEL", "16"))
    _nh = int(os.environ.get("NUM_HEADS", "4"))
    # Connectivity MUST match the donor checkpoint too. A parallel-attn donor
    # rebuilt as serial loads without error (shapes are identical) but evaluates
    # at 0.17 instead of 0.77, which then looks like a quantizer-range collapse.
    _par = bool(int(os.environ.get("PARALLEL_ATTN", 0)))
    fmodel = build_phat_jet_k3(quantized=False, use_ln=False, aggregation="mean",
                               gmp_edges=gmp_edges, num_particles=_npart, patch_size=_ps,
                               d_model=_dm, num_heads=_nh, parallel_attn=_par)
    fmodel.load_weights(float_path)
    fw = {l.name: l.get_weights() for l in fmodel.layers if l.get_weights()}
    n_copied = 0
    for layer in qmodel.layers:
        if layer.name not in fw:
            continue
        src = fw[layer.name]
        # Q layers hold extra quantizer variables; match by shape prefix
        dst = layer.get_weights()
        replaced, si = list(dst), 0
        for di, w in enumerate(dst):
            if si < len(src) and src[si].shape == w.shape:
                replaced[di] = src[si]
                si += 1
        if si == len(src):
            layer.set_weights(replaced)
            n_copied += 1
    print(f"transferred weights into {n_copied} layers")
    return n_copied



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


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Loading + robust scaling + padding check: single source of truth in preprocess.py.
    from preprocess import load_jets
    x_train, y_train, x_val, y_val = load_jets(root=root)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x_train))[:TRAIN_SUBSET]
    x_train, y_train = x_train[idx], y_train[idx]

    # N_PARTICLES: compete on JEDI-Linear's own axis (their Table II sweeps
    # N in {8,16,32,64,128}). The data is pT-sorted, so [:, :N] is top-N by pT --
    # their exact input scheme. Cost falls ~linearly in N, so N=32 trains ~4x
    # faster per epoch than N=128, which is what makes QAT feasible here.
    N_PART = int(os.environ.get("N_PARTICLES", "128"))
    PATCH_SZ = int(os.environ.get("PATCH_SIZE", "8"))
    if N_PART != 128:
        x_train, x_val = x_train[:, :N_PART, :], x_val[:, :N_PART, :]
        print(f"[{RUN_TAG}] N={N_PART} patch_size={PATCH_SZ} -> x_train {x_train.shape}", flush=True)

    # GMP grid: winner of the 3-way shootout (docs/grid_study.md). Edges are in
    # raw eta/phi units in grid_shootout.GRIDS; convert to robust-scaled units.
    from grid_shootout import GRIDS
    from preprocess import gmp_edges_scaled
    gmp_edges = gmp_edges_scaled(GRID_NAME, root=root)
    print(f"[{RUN_TAG}] grid={GRID_NAME} raw={GRIDS[GRID_NAME].tolist()}", flush=True)

    D_MODEL = int(os.environ.get("D_MODEL", "16"))
    NUM_HEADS = int(os.environ.get("NUM_HEADS", "4"))
    model = build_qat_model(gmp_edges=gmp_edges, num_particles=N_PART, patch_size=PATCH_SZ,
                            d_model=D_MODEL, num_heads=NUM_HEADS)
    # Float init: run 3 collapsed because the donor ckpt was trained WITH
    # LayerNorms -> LN-free Q model saw exploding activation ranges -> EBOPs 5.9e9
    # -> beta penalty crushed it. Fix: float_pretrain.py trains the EXACT LN-free
    # core7 architecture, so the transfer is 1:1 by layer name and shape.
    # Donor must be trained at the SAME N. Truncating an N=128 model instead of
    # training at N collapses it (measured: 78.6% -> 63.3% at N=32, 46.8% at
    # N=16), so scripts/float_n.py produces a per-N float donor.
    # FLOAT_CK overrides the N-derived donor name. This matters: the auto-derived
    # name points at the FIRST float run at that N, which in this project was still
    # training (30 epochs). The converged donors live under _long/_scratch/_ps4
    # suffixes and are 1.3-1.8 pt better -- seeding QAT from the stale ones threw
    # away more accuracy than the entire remaining gap to JEDI-linear.
    _fck = os.environ.get("FLOAT_CK", "")
    float_ckpt = os.path.join(root, _fck) if _fck else os.path.join(
        root, f"float_phatjet_{GRID_NAME}_n{N_PART}.keras"
              if N_PART != 128 else f"float_phatjet_{GRID_NAME}.keras")
    # QAT_RESUME: continue compression from an ALREADY-QUANTIZED checkpoint under a
    # different beta ramp. Distinct from QAT_INIT, which selects quantizer i0 init
    # (i0=2 for a float donor to avoid wrap; 0 for from-scratch) -- leave QAT_INIT
    # at its default "float" when resuming, since the incoming weights are the same
    # magnitude as the float-transferred ones.
    _qres = os.environ.get("QAT_RESUME", "")
    if _qres:
        model.load_weights(_qres, skip_mismatch=False)
        _lg = model.predict(x_val[:50_000], batch_size=8192, verbose=0)
        _acc = float((_lg.argmax(1) == y_val[:50_000].argmax(1)).mean())
        print(f"QAT_RESUME from {os.path.basename(_qres)}: val_acc={_acc:.4f}", flush=True)
        assert _acc > 0.5, f"QAT_RESUME load collapsed to {_acc:.4f} (chance=0.2)"
    elif os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SKIP_PRETRAINED_INIT")):
        print("SKIP_PRETRAINED_INIT set: training QAT from scratch", flush=True)
    elif os.path.exists(float_ckpt):
        n = transfer_float_weights(model, float_ckpt, gmp_edges=gmp_edges)
        # Guard against silent wrap-collapse (run 3): a healthy transfer lands
        # near the float model's 0.816; chance is 0.20. predict() not evaluate(),
        # since compile() has not happened yet.
        _lg = model.predict(x_val[:50_000], batch_size=8192, verbose=0)
        _acc = float((_lg.argmax(1) == y_val[:50_000].argmax(1)).mean())
        print(f"init from {os.path.basename(float_ckpt)}: {n} layers, "
              f"post-transfer val_acc={_acc:.4f}", flush=True)
        assert _acc > 0.5, (
            f"transfer collapsed to {_acc:.4f} (chance=0.2): quantizer range too "
            f"narrow for the incoming float weights/activations — see BW_K/BW_A notes")
    else:
        print(f"no {float_ckpt}; training QAT from scratch", flush=True)

    # Beta schedule: compressed version of jsc150's
    # [(0, 2e-8, linear), (2000/7000, 3e-7, log), (7000/7000, 3e-6, const)]
    # jsc150: [(0, 2e-8, linear), (2000, 3e-7, log), (7000, 3e-6, const)] over 7000
    # epochs. Same values + shapes, breakpoints scaled by 200/7000 (2000 -> 57).
    # BETA_MULT scales the whole EBOPs-penalty ramp. The stock jsc150 schedule is
    # tuned for 7000 GPU epochs; on a few-hundred-epoch CPU budget it compresses
    # far too slowly (measured: 27M EBOPs still at epoch 3 of qat_n32, where the
    # deployable regime is ~1e5). BETA_MULT>1 buys compression at some accuracy.
    BM = float(os.environ.get("BETA_MULT", "1"))
    B0 = float(os.environ.get("BETA0", "2e-8")) * BM
    # BETA_WARMUP: hold beta at ~0 for the first W epochs so the transferred float
    # weights can re-converge under quantization BEFORE the cost penalty starts
    # pushing bitwidths down. Measured without it: qat_n64_c2 initialised at 81.41%,
    # fell to 78.96% at epoch 0, and needed 6 epochs to climb back to 81.24% -- all
    # while the penalty was already compressing, so the recovery happened against
    # a moving target. With a warmup the model re-peaks at full precision first.
    W = int(os.environ.get("BETA_WARMUP", "0"))
    if W > 0:
        knee = max(W + 1, int(EPOCHS * 2000 / 7000))
        beta_sched = BetaScheduler(
            PieceWiseSchedule([(0, 1e-12, "constant"),
                               (W, B0, "linear"),
                               (knee, 3e-7 * BM, "log"),
                               (EPOCHS, 3e-6 * BM, "constant")])
        )
        print(f"[{RUN_TAG}] beta ramp x{BM} with {W}-epoch warmup: "
              f"~0 -> {B0:.2e} -> {3e-7*BM:.2e} -> {3e-6*BM:.2e}", flush=True)
    else:
        beta_sched = BetaScheduler(
            PieceWiseSchedule([(0, B0, "linear"),
                               (int(EPOCHS * 2000 / 7000), 3e-7 * BM, "log"),
                               (EPOCHS, 3e-6 * BM, "constant")])
        )
        print(f"[{RUN_TAG}] beta ramp x{BM}: {B0:.2e} -> {3e-7*BM:.2e} -> {3e-6*BM:.2e}", flush=True)
    exp_clamp = ClampSoftmaxExpBits(EXP_I_MAX)
    ebops = FreeEBOPs()
    pareto_dir = os.path.join(root, f"pareto_{RUN_TAG}")
    os.makedirs(pareto_dir, exist_ok=True)
    pareto = ParetoFront(
        pareto_dir,
        ["val_accuracy", "ebops"],
        [1, -1],
        fname_format="epoch={epoch}-val_acc={val_accuracy:.4f}-ebops={ebops}.keras",
        enable_if=lambda x: x["val_accuracy"] > 0.6,
    )
    # A 1000-epoch CPU run spans days: persist per-epoch history so progress is
    # inspectable and a crash doesn't lose the log.
    csv_cb = keras.callbacks.CSVLogger(os.path.join(root, f"{RUN_TAG}_log.csv"), append=True)
    # Rolling latest-weights checkpoint for resume-after-interrupt.
    resume_cb = keras.callbacks.ModelCheckpoint(
        os.path.join(root, f"{RUN_TAG}_latest.weights.h5"), save_weights_only=True, verbose=0)
    # jsc150: cosine_decay_restarts(lr, first_decay_steps=500) over 7000 epochs
    # -> scaled 500 * 200/7000 ≈ 14 epochs per restart cycle.
    lr_cb = keras.callbacks.LearningRateScheduler(cosine_restarts(LR, max(int(500 * EPOCHS / 7000), 10)))

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        steps_per_execution=4,
    )
    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH,
        epochs=EPOCHS,
        verbose=2,
        # exp_clamp FIRST: it must reassert the exp-table bit cap before FreeEBOPs
        # logs ebops and before ParetoFront decides/saves, so both see clamped bits.
        callbacks=[exp_clamp, ebops, lr_cb, beta_sched, pareto, csv_cb, resume_cb, StopFile(RUN_TAG, root)],
    )
    json.dump(
        {k: [float(v) for v in vs] for k, vs in hist.history.items()},
        open(os.path.join(root, f"{RUN_TAG}_hist.json"), "w"),
    )
    model.save(os.path.join(root, f"{RUN_TAG}_final.keras"))
    print(f"[{RUN_TAG}] best val_acc {max(hist.history['val_accuracy']):.4f}", flush=True)


if __name__ == "__main__":
    main()
