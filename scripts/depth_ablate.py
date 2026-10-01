"""Price each PHAT-JeT block in PIPELINE STAGES (float, untrained).

JEDI-linear pT-sorted traces to 10-12 stages under our own convention; our
frontier designs are 40-63. Depth is architectural, so it can be measured on
an untrained model -- this tells us which block to attack before spending any
training compute.
"""
import os, sys, json, itertools
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phat_jet_k3 import build_phat_jet_k3
from extract_hw import CLOCK_NS, INPUTS_KIF, LATENCY_CUTOFF
from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.constraints import MinMax
from hgq.regularizers import MonoL1
from da4ml.converter import trace_model
from da4ml.trace import comb_trace, HWConfig, to_pipeline

N = int(os.environ.get("N_PART", 32))
rows = []
PAR = int(os.environ.get("PARALLEL", 0))
for la, pa, ff, gmp in itertools.product([1,0],[1,0],[1,0],[1,0]):
    tag = f"la{la}_pa{pa}_ff{ff}_gmp{gmp}" + ("_par" if PAR else "")
    try:
        # same quantizer scope stack as train_qat.build_model (BW_K=6, BW_A=6)
        with QuantizerConfigScope(default_q_type="kbi", b0=6, overflow_mode="wrap",
                                  i0=2, fr=MonoL1(1e-9), ir=MonoL1(1e-9), i_decay_speed=1e-3), \
             QuantizerConfigScope(default_q_type="kif", place="datalane",
                                  overflow_mode="wrap", f0=6, i0=2,
                                  fr=MonoL1(1e-9), ic=MinMax(0, 12)), \
             LayerConfigScope(beta0=0):
            m = build_phat_jet_k3(num_particles=N, num_feats=3, d_model=16, num_heads=4,
                                  patch_size=8, quantized=True, use_gmp=bool(gmp),
                                  use_local_attn=bool(la), use_patch_attn=bool(pa),
                                  use_ffn=bool(ff), parallel_attn=bool(PAR))
        inp, out = trace_model(m, hwconf=HWConfig(1,-1,-1), inputs_kif=INPUTS_KIF)
        p = to_pipeline(comb_trace(inp, out), latency_cutoff=LATENCY_CUTOFF, verbose=False)
        r = dict(tag=tag, N=N, parallel=PAR, local_attn=la, patch_attn=pa, ffn=ff, gmp=gmp,
                 stages=len(p.solutions), latency_ns=round(len(p.solutions)*CLOCK_NS,1),
                 lut=round(float(p.cost)), params=int(m.count_params()))
        rows.append(r)
        print(f"RESULT {tag:22s} {r['stages']:3d}st {r['latency_ns']:6.1f}ns  LUT {r['lut']:>9,}", flush=True)
    except Exception as e:
        print(f"FAIL {tag}: {type(e).__name__}: {str(e)[:90]}", flush=True)
json.dump(rows, open(f"depth_ablation_n{N}" + ("_par" if PAR else "") + ".json","w"), indent=1)
