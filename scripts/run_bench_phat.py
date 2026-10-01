"""Trace PHAT-JeT Pareto checkpoints on JEDI-linear's exact da4ml settings."""
import os, sys, json, time, glob, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("GRID_NAME", "core7")
import numpy as np
from train_qat import build_qat_model
from extract_hw import _gmp_edges
from preprocess import load_jets
from bench_headtohead import trace_cost

OUT = "bench_phat_points.json"
Xv, Yv = load_jets("val")

cks = []
for pat in ["pareto_run8_core7_fixed/*.keras", "ckpt_snapshots_run6/*.keras"]:
    for f in glob.glob(pat):
        mm = re.search(r"epoch=(\d+)-val_acc=([\d.]+)-ebops=(\d+)", f)
        if mm:
            cks.append((int(mm.group(3)), f, "run8" if "run8" in f else "run6"))
cks.sort()

# Point selection. The FIRST version of this spread indices evenly over the
# count-sorted checkpoint list, which is wrong: the frontier is dense at low
# EBOPs and sparse at high EBOPs, so even index spacing sampled 1.06M LUT and
# then jumped to 5.12M, skipping the entire region that fits the VU13P
# (1,728,000 LUT). It also let the reported "best" point be epoch 14 -- a
# snapshot from before the EBOPs regularizer had pulled bitwidths down, which
# costs 555 LUT per data-dependent multiply versus 18-26 for converged points.
# That single mis-sampled point produced a 228x resource claim that is an
# artifact of sampling, not of the architecture.
#
# Instead: take the ACCURACY-BEST checkpoint in each EBOPs decade-ish bucket,
# which is what a Pareto frontier actually means, and bias coverage toward the
# 0.5M-6M EBOPs window where the design fits the device.
BUCKETS = [(4e5, 6e5), (6e5, 8e5), (8e5, 1.1e6), (1.1e6, 2e6),
           (2e6, 4e6), (4e6, 6e6), (6e6, 3e7), (3e7, 1e9)]


def _acc(path):
    return float(re.search(r"val_acc=([\d.]+)", path).group(1))


sel = []
for lo, hi in BUCKETS:
    cand = [c for c in cks if lo <= c[0] < hi]
    if cand:
        sel.append(max(cand, key=lambda c: _acc(c[1])))
# de-duplicate identical checkpoints appearing in several snapshot dirs
_seen, _sel = set(), []
for eb, f, run in sel:
    key = os.path.basename(f)
    if key not in _seen:
        _seen.add(key)
        _sel.append((eb, f, run))
sel = _sel

done = {}
if os.path.exists(OUT):
    for r in json.load(open(OUT)):
        done[r["tag"]] = r

model = build_qat_model(aggregation="mean", gmp_bounds=3.7, gmp_edges=_gmp_edges())
out = list(done.values())
for eb, f, run in sel:
    tag = f"phat_{run}_eb{eb}"
    if tag in done:
        print(f"skip {tag} (cached)", flush=True); continue
    model.load_weights(f, skip_mismatch=False)
    P = np.concatenate([np.array(model(Xv[i:i+8192], training=False))
                        for i in range(0, len(Xv), 8192)])
    acc = float((P.argmax(-1) == Yv.argmax(-1)).mean())
    t = time.time()
    r = trace_cost(model, tag, outdir="bench_prjs")
    r.update(val_acc=acc, ebops=eb, run=run, ckpt=os.path.basename(f),
             params=int(model.count_params()))
    print(f"{tag}: acc {acc:.4f} ebops {eb:,} LUT {r['da_est_LUT']:,} "
          f"FF {r['da_est_FF']:,} lat {r['latency_ns']}ns stages {r['pipeline_stages']} "
          f"({time.time()-t:.0f}s)", flush=True)
    out.append(r)
    json.dump(out, open(OUT, "w"), indent=1)
print("DONE", flush=True)
