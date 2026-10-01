"""Trace the untraced HIGH-ACCURACY / LOW-EBOPs checkpoints -- the band that decides
the rebuttal headline.

Rationale: the deliverable the reviewers need is the most ACCURATE design that still
fits a VU13P (1,728,000 LUT). Our current best fitting point is 78.64% at 1.19M LUT
(69% of the device). Several checkpoints reach ~78.4-78.6% at LOWER EBOPs than that
one, so they may trace cheaper at equal accuracy -- which both improves the headline
and buys device headroom for the jet-clustering logic that must share CTL2.

All these runs use the N=128 builder (train_qat / train_hold both call
build_qat_model with aggregation="mean", gmp_bounds=3.7, gmp_edges=_gmp_edges()),
so one model object can load each checkpoint's weights in turn.

Same instrument as every other number: JEDI-linear's own da4ml settings via
bench_headtohead.trace_cost -> HWConfig(1,-1,-1), hard_dc=2, clock 2.0 ns.
"""
import os, sys, json, time, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("GRID_NAME", "core7")
import numpy as np
from train_qat import build_qat_model
from extract_hw import _gmp_edges
from preprocess import load_jets
from bench_headtohead import trace_cost

OUT = "budget_band_traces.json"
CKPTS = [
    "pareto_run9_hold/epoch=27-val_acc=0.7857-ebops=880426.keras",
    "pareto_run9_hold/epoch=24-val_acc=0.7841-ebops=874750.keras",
    "pareto_run8_core7_fixed/epoch=134-val_acc=0.7861-ebops=977813.keras",
    "pareto_run8_core7_fixed/epoch=140-val_acc=0.7858-ebops=954727.keras",
    "pareto_run6_core7_1000ep/epoch=141-val_acc=0.7854-ebops=922791.keras",
]

Xv, Yv = load_jets("val")

done = {}
if os.path.exists(OUT):
    for r in json.load(open(OUT)):
        done[r["ckpt"]] = r

model = build_qat_model(aggregation="mean", gmp_bounds=3.7, gmp_edges=_gmp_edges())
out = list(done.values())
for f in CKPTS:
    base = os.path.basename(f)
    if base in done:
        print(f"skip {base} (cached)", flush=True)
        continue
    if not os.path.exists(f):
        print(f"MISSING {f}", flush=True)
        continue
    model.load_weights(f, skip_mismatch=False)
    # Re-evaluate rather than trusting the filename: at least one checkpoint in this
    # project has a filename accuracy that disagrees with re-evaluation.
    P = np.concatenate([np.array(model(Xv[i:i + 8192], training=False))
                        for i in range(0, len(Xv), 8192)])
    acc = float((P.argmax(-1) == Yv.argmax(-1)).mean())
    eb = int(re.search(r"ebops=(\d+)", base).group(1))
    t = time.time()
    r = trace_cost(model, f"band_{eb}", outdir="bench_prjs")
    r.update(val_acc=acc, ebops=eb, ckpt=base, N=128,
             params=int(model.count_params()),
             fits_vu13p=bool(r["da_est_LUT"] <= 1_728_000),
             pct_vu13p=round(r["da_est_LUT"] / 1_728_000 * 100, 1))
    print(f"RESULT {base[:44]:44s} acc {acc*100:6.2f} LUT {r['da_est_LUT']:>9,} "
          f"({r['pct_vu13p']:4.1f}% VU13P) lat {r['latency_ns']}ns ({time.time()-t:.0f}s)",
          flush=True)
    out.append(r)
    json.dump(out, open(OUT, "w"), indent=1)
print("DONE", flush=True)
