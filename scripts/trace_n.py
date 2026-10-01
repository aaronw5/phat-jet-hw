"""Trace a per-N QAT (or float-init Q) checkpoint -> LUT/FF/latency estimate.

Reports on JEDI-Linear's clock convention: they pipeline every 2 adders for
Fmax ~= 300 MHz. Our N=128/f3 JEDI rebuild gives 23 stages, and 23 * 3.33ns =
77ns against their published 82ns for that row, so 3.33ns is the clock that
reproduces their table. Do NOT report at 2.0ns.

Usage:  python trace_n.py <ckpt.keras> <N> [patch_size]
Appends to nsweep_results.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("GRID_NAME", "core7")
import numpy as np
from train_qat import build_qat_model
from extract_hw import _gmp_edges, CLOCK_NS, INPUTS_KIF, LATENCY_CUTOFF
from preprocess import load_jets
from da4ml.converter import trace_model
from da4ml.trace import comb_trace, HWConfig, to_pipeline

CK = sys.argv[1]; N = int(sys.argv[2]); PS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
OUT = os.environ.get("TRACE_OUT", "nsweep_results.json")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X, Y = load_jets("val", n=N, check=True)

DM = int(os.environ.get("D_MODEL", "16")); NH = int(os.environ.get("NUM_HEADS", "4"))
m = build_qat_model(aggregation="mean", gmp_bounds=3.7, gmp_edges=_gmp_edges(),
                    num_particles=N, patch_size=PS, d_model=DM, num_heads=NH)
m.load_weights(CK, skip_mismatch=False)
m.compile(loss="categorical_crossentropy", metrics=["accuracy"])
acc = float(m.evaluate(X, Y, batch_size=4096, verbose=0)[1])

inp, out = trace_model(m, solver_options={"hard_dc": 2}, hwconf=HWConfig(1, -1, -1),
                       inputs_kif=INPUTS_KIF)
sol = comb_trace(inp, out)
psol = to_pipeline(sol, latency_cutoff=LATENCY_CUTOFF)
lut = float(psol.cost); ff = int(psol.reg_bits); stages = len(psol.solutions)
r = dict(ckpt=os.path.basename(CK), N=N, patch_size=PS, val_acc=acc, lut_est=lut,
         ff_est=ff, stages=stages, latency_ns=stages * CLOCK_NS, clock_ns=CLOCK_NS,
         DSP=0, BRAM=0, II=1, comb_lut=float(sol.cost))
print(f"N={N:3d} acc={acc*100:6.2f}% LUT={lut:11,.0f} FF={ff:10,d} "
      f"stages={stages:4d} lat={r['latency_ns']:6.1f}ns", flush=True)
rows = json.load(open(OUT)) if os.path.exists(OUT) else []
rows = [x for x in rows if not (x.get("N") == N and x.get("ckpt") == r["ckpt"])] + [r]
json.dump(rows, open(OUT, "w"), indent=1)
