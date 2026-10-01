"""PHAT-JeT on JEDI-Linear's own N in {16,32,64,128} axis, WITHOUT retraining.

Why this is legitimate and not a trick
--------------------------------------
Every LEARNED weight in PHAT-JeT is shared across the particle axis and across
patches (GMP is a per-bin grid op; patch attention is applied per patch with
shared projections; the head acts on a pooled vector). The only N-dependent
variables are the 72 per-element HGQ quantizer tensors of shape (1, N) --
verified by inspection: 98,688 of the checkpoint's 196,112 "params" are these
bitwidth variables, not weights.

So an N=128 checkpoint defines a valid N<128 model by SLICING those quantizer
tensors to their first N columns and keeping every weight untouched. Because the
hls4ml data is pT-sorted (verified: 100% of jets non-increasing in feature 0),
the first N rows are the top-N particles by pT -- exactly JEDI-Linear's input
scheme, so the comparison is on their axis, with their selection rule.

This is a zero-training measurement. It is a LOWER bound on reduced-N accuracy:
these weights were optimized for N=128 and the sliced model is never fine-tuned.
Any N<128 point reported here can only improve with training.

Usage:  python nsweep.py [ckpt.keras]
Writes: nsweep_results.json
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("GRID_NAME", "core7")

import numpy as np
import keras
from train_qat import build_qat_model
from extract_hw import _gmp_edges, CLOCK_NS, INPUTS_KIF, LATENCY_CUTOFF
from preprocess import load_jets
from da4ml.converter import trace_model
from da4ml.trace import comb_trace, HWConfig, to_pipeline

CK = sys.argv[1] if len(sys.argv) > 1 else \
     "ckpt_snapshots_run6/epoch=140-val_acc=0.7864-ebops=926777.keras"
NS = [int(x) for x in os.environ.get("NS", "16,32,64,128").split(",")]
# JEDI-Linear pipelines every 2 adders -> ~300 MHz. Our N=128/f3 rebuild gives
# 23 stages, and 23 * 3.33ns = 77ns vs their published 82ns for that row, so
# 3.33ns is the clock that reproduces their table. Do NOT use 2.0ns.
OUT = "nsweep_results.json"

Xv_full, Yv = load_jets("val")

ref = build_qat_model(aggregation="mean", gmp_bounds=3.7, gmp_edges=_gmp_edges(),
                      num_particles=128)
ref.load_weights(CK, skip_mismatch=False)
# Match POSITIONALLY, never by w.path. HGQ quantizer sublayers carry a GLOBAL
# auto-increment counter (fixed_point_quantizer_kif_1 in the first model built
# this process, _51 in the second), so path-keyed lookup silently misses every
# quantizer: the first attempt matched 122 of 518 weights, left the quantizers
# at their full-bitwidth init, and reported 48M LUT at N=16 (larger than N=128)
# with 19.6% = chance accuracy. Verified safe: for any N the two models have
# identical layer names, identical layer count (44) and identical weight order
# (518), so zip() over .weights pairs corresponding tensors exactly.
ref_w = [np.asarray(w) for w in ref.weights]
assert len(ref_w) == 518, len(ref_w)

rows = []
for N in NS:
    ps = 8 if N >= 8 else N
    m = build_qat_model(aggregation="mean", gmp_bounds=3.7, gmp_edges=_gmp_edges(),
                        num_particles=N, patch_size=ps)
    assert len(m.weights) == len(ref_w), (len(m.weights), len(ref_w))
    assert [l.name for l in m.layers] == [l.name for l in ref.layers]
    n_slice = n_exact = n_miss = 0
    for w, src in zip(m.weights, ref_w):
        tgt = tuple(w.shape)
        if src.shape == tgt:
            w.assign(src); n_exact += 1
        else:
            # slice every axis down to the target extent (the N axis shrinks)
            sl = tuple(slice(0, t) for t in tgt)
            if src[sl].shape == tgt:
                w.assign(src[sl]); n_slice += 1
            else:
                n_miss += 1
    Xv = Xv_full[:, :N, :]
    m.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    acc = float(m.evaluate(Xv, Yv, batch_size=4096, verbose=0)[1])

    inp, out = trace_model(m, solver_options={"hard_dc": 2},
                           hwconf=HWConfig(1, -1, -1), inputs_kif=INPUTS_KIF)
    sol = comb_trace(inp, out)
    psol = to_pipeline(sol, latency_cutoff=LATENCY_CUTOFF)
    # Pipeline.reg_bits is already a scalar total; stages == len(solutions).
    lut = float(psol.cost); ff = int(psol.reg_bits)
    stages = len(psol.solutions)
    r = dict(N=N, patch_size=ps, val_acc=acc, lut_est=lut, ff_est=ff,
             stages=stages, latency_ns=stages * CLOCK_NS, clock_ns=CLOCK_NS,
             DSP=0, BRAM=0, II=1,
             w_exact=n_exact, w_sliced=n_slice, w_missing=n_miss)
    print(f"N={N:3d} acc={acc*100:6.2f}% LUT={lut:12,.0f} stages={stages:4d} "
          f"lat={r['latency_ns']:6.1f}ns  (exact {n_exact}, sliced {n_slice}, miss {n_miss})",
          flush=True)
    rows.append(r)
    json.dump(rows, open(OUT, "w"), indent=1)
print("wrote", OUT)
