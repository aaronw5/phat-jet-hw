"""Per-op cost breakdown + latency_cutoff sweep at the device-fitting point.

latency_cutoff is a SYNTHESIS knob, not an architecture change: it sets the
max combinational delay per pipeline stage. JEDI-linear used 4.0; applying the
same sweep to BOTH models keeps the comparison apples-to-apples.
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("GRID_NAME", "core7")
import numpy as np
from train_qat import build_qat_model
from extract_hw import _gmp_edges, INPUTS_KIF
from da4ml.converter import trace_model
from da4ml.trace import comb_trace, HWConfig
from da4ml.trace.pipeline import to_pipeline

CK = "ckpt_snapshots_run6/epoch=140-val_acc=0.7864-ebops=926777.keras"
m = build_qat_model(aggregation="mean", gmp_bounds=3.7, gmp_edges=_gmp_edges())
m.load_weights(CK, skip_mismatch=False)
inps, outs = trace_model(m, hwconf=HWConfig(1, -1, -1), inputs_kif=INPUTS_KIF)
comb = comb_trace(inps, outs)

NAMES = {0: "add", 1: "sub", 2: "relu", 3: "copy/neg", 5: "const", 6: "msb_mux",
         7: "VMUL (data x data)", 8: "LOOKUP (LUT table)", 9: "bit_unary"}
by, cnt = collections.Counter(), collections.Counter()
for op in comb.ops:
    k = NAMES.get(abs(op.opcode), f"op{op.opcode}")
    if op.opcode in (0, 1):
        k = "add/sub"
    by[k] += float(op.cost); cnt[k] += 1
tot = comb.cost
print(f"ep140 combinational: {tot:,.0f} LUT over {len(comb.ops):,} ops", flush=True)
for k, v in by.most_common():
    print(f"  {v:>11,.0f} {100*v/tot:>5.1f}%  {k:<22} n={cnt[k]:>7,}  {v/max(cnt[k],1):>6.1f} LUT/op", flush=True)
lts = comb.lookup_tables
print(f"  lookup tables: {len(lts)}  sizes min={min(len(t) for t in lts)} max={max(len(t) for t in lts)}", flush=True)

res = []
for lc in [4.0, 6.0, 8.0, 12.0, 16.0]:
    p = to_pipeline(comb, latency_cutoff=lc)
    st = len(p.solutions)
    r = dict(latency_cutoff=lc, LUT=int(p.cost), FF=int(p.reg_bits),
             stages=st, latency_ns=st * 2.0)
    res.append(r)
    print(f"  lc={lc:5.1f}  LUT {r['LUT']:>10,}  FF {r['FF']:>10,}  "
          f"stages {st:>4}  latency {r['latency_ns']:>6.0f} ns", flush=True)
json.dump(res, open("sweep_latency_ep140.json", "w"), indent=1)
