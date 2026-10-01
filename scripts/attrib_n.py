"""Where does the LUT go? Attribute combinational cost by da4ml opcode AND by
which part of the network the op came from. Answers: is the low-N LUT floor
reducible (dense/attention) or intrinsic (softmax exp tables, GMP indicators)?

Opcode map (read from da4ml.types dispatch, verified): 0/'Op'=add, 1=sub,
4=const-add, 7=data*data multiply, 8=lookup table (exp/recip), others=relu/quant/mux.
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KERAS_BACKEND", "jax"); os.environ.setdefault("GRID_NAME", "core7")
import numpy as np
from train_qat import build_qat_model
from extract_hw import _gmp_edges, INPUTS_KIF, LATENCY_CUTOFF, CLOCK_NS
from da4ml.converter import trace_model
from da4ml.trace import comb_trace, HWConfig

CK = sys.argv[1]; N = int(sys.argv[2]); PS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
DM = int(os.environ.get("D_MODEL", "16")); NH = int(os.environ.get("NUM_HEADS", "4"))
m = build_qat_model(aggregation="mean", gmp_bounds=3.7, gmp_edges=_gmp_edges(),
                    num_particles=N, patch_size=PS, d_model=DM, num_heads=NH)
m.load_weights(CK, skip_mismatch=False)
inp, out = trace_model(m, solver_options={"hard_dc": 2}, hwconf=HWConfig(1, -1, -1),
                       inputs_kif=INPUTS_KIF)
sol = comb_trace(inp, out)
by_op = collections.Counter(); n_by_op = collections.Counter()
for op in sol.ops:
    k = str(getattr(op, "opcode", "?"))
    by_op[k] += float(op.cost); n_by_op[k] += 1
tot = float(sol.cost)
NAME = {"Op": "add", "0": "add", "1": "sub", "4": "const-add",
        "7": "mul(data*data)", "8": "lookup-table(exp/recip)"}
rows = []
for k, v in by_op.most_common():
    rows.append(dict(opcode=k, name=NAME.get(k, f"other({k})"), lut=v,
                     pct=100*v/tot, n=n_by_op[k]))
out_d = dict(ckpt=os.path.basename(CK), N=N, patch_size=PS, d_model=DM,
             total_lut=tot, n_ops=len(sol.ops), rows=rows)
json.dump(out_d, open(f"attrib_n{N}_ps{PS}_d{DM}_{os.path.splitext(os.path.basename(CK))[0]}.json", "w"), indent=1)
print(f"N={N} ps={PS} d_model={DM}  total comb LUT={tot:,.0f}  ops={len(sol.ops):,}")
for r in rows:
    print(f"  {r['name']:>24s}: {r['lut']:>12,.0f} ({r['pct']:5.1f}%)  n={r['n']:>8,}")
mul_lut = by_op.get("7", 0) + by_op.get("8", 0)
print(f"  --> data*data mul + tables = {100*mul_lut/tot:.1f}%  (JEDI-linear has ZERO of these)")
