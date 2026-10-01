"""Apples-to-apples: JEDI-linear's OWN pT-sorted checkpoints, unmodified
architecture (get_gnn verbatim), evaluated on OUR validation split and traced
with OUR da4ml estimator at OUR fixed 3.33ns clock.

No widening, no re-architecting: their published scaling knob is N, and we use
exactly the N values they shipped weights for (8/16/32/64/128).
"""
import os, sys, json, glob
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, keras
import build_jedi as BJ
from extract_hw import CLOCK_NS, INPUTS_KIF, LATENCY_CUTOFF
from da4ml.converter import trace_model
from da4ml.trace import comb_trace, HWConfig, to_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load("data/jets_128x3.npz")
Yv = d["y_val"]
OUT = "jedi_family_bench.json"
N = int(sys.argv[1])

ck = glob.glob(f"refs/JEDI-linear-master/official_models/3-feature/jet_classifier_large_{N}/ckpts/*.keras")[0]
m = BJ.build_jedi(N=N, n=3)
BJ.load_official_weights(m, ck)
# JEDI-linear's own preprocessing (src/dataloader.py): per-feature standardize
# with TRAIN-set mean/std. Our robust_scale (median/IQR) is a different
# distribution and their weights are not valid under it.
# NB: they truncate to n_constituents BEFORE computing mean/std, so the
# standardization stats are per-N. Truncate first, then standardize.
_xt = d["x_train"].astype("float32")[:, :N, :]
_shift = _xt.mean(axis=(0, 1), keepdims=True)
_scale = _xt.std(axis=(0, 1), keepdims=True)
X = (d["x_val"].astype("float32")[:, :N, :] - _shift) / _scale
m.compile(loss="categorical_crossentropy", metrics=["accuracy"])
acc = float(m.evaluate(X, Yv, batch_size=4096, verbose=0)[1])

# JEDI-linear's own trace settings: no inputs_kif override (their quantizers
# define the input format), hard_dc=2, HWConfig(1,-1,-1), latency_cutoff=2.
inp, out = trace_model(m, solver_options={"hard_dc": 2}, hwconf=HWConfig(1, -1, -1))
sol = comb_trace(inp, out); psol = to_pipeline(sol, latency_cutoff=LATENCY_CUTOFF)
r = dict(tag=f"jedi_pt_N{N}", N=N, ckpt=os.path.basename(ck), val_acc=acc,
         lut=float(psol.cost), ff=int(psol.reg_bits), stages=len(psol.solutions),
         latency_ns=len(psol.solutions)*CLOCK_NS, clock_ns=CLOCK_NS,
         params=int(m.count_params()))
print(f"RESULT N={N:3d} acc={acc*100:6.2f}% LUT={r['lut']:11,.0f} stages={r['stages']:3d} "
      f"lat={r['latency_ns']:6.1f}ns", flush=True)
rows = json.load(open(OUT)) if os.path.exists(OUT) else []
rows = [x for x in rows if x.get("N") != N] + [r]
json.dump(rows, open(OUT, "w"), indent=1)
