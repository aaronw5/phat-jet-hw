"""Scaled-up JEDI-linear: same topology, wider hidden dim.
Purpose: answer "if we get a bigger FPGA, so do they" -- measure THEIR
architecture at the logic budget where OUR scaled design lives, using the
same da4ml estimator that reproduces their published N=128 row to +1.9%.
Untrained: we report LUT/latency only (capacity, not accuracy).
"""
import os, sys, json
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math import log2
import keras, numpy as np
import build_jedi as BJ
from hgq.config import QuantizerConfigScope, LayerConfigScope
from hgq.layers import QEinsumDenseBatchnorm, QSum, QAdd
from hgq.regularizers import MonoL1
from hgq.constraints import MinMax
from extract_hw import CLOCK_NS, INPUTS_KIF, LATENCY_CUTOFF
from da4ml.converter import trace_model
from da4ml.trace import comb_trace, HWConfig, to_pipeline

def get_gnn_w(N=128, n=3, W=64):
    """get_gnn with hidden width W instead of the hardcoded 64."""
    with (QuantizerConfigScope(place=('weight','bias'), overflow_mode='SAT_SYM'),
          QuantizerConfigScope(place='datalane', heterogeneous_axis=None)):
        inp = keras.layers.Input((N, n))
        ps = 2.0 ** -round(log2(N))
        x = QEinsumDenseBatchnorm('bnc,cC->bnC', (N, W), bias_axes='C', activation='relu')(inp)
        s = QEinsumDenseBatchnorm('bnc,cC->bnC', (N, W), bias_axes='C', activation='relu')(x)
        d = QEinsumDenseBatchnorm('bnc,cC->bnC', (1, W), bias_axes='C', activation='relu')(
            QSum(axes=1, scale=ps, keepdims=True)(x))
        x = QAdd()([s, d])
        x = QEinsumDenseBatchnorm('bnc,cC->bnC', (N, W), bias_axes='C', activation='relu')(x)
        x = QSum(axes=1, scale=1/16, keepdims=False)(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', W, bias_axes='C', activation='relu')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', W//2, bias_axes='C', activation='relu')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', W//4, bias_axes='C', activation='relu')(x)
        out = QEinsumDenseBatchnorm('bc,cC->bC', 5, bias_axes='C')(x)
    return keras.Model(inputs=inp, outputs=out)

def build(N, n, W, bw=7):
    s0 = QuantizerConfigScope(default_q_type='kbi', b0=bw, overflow_mode='wrap', i0=0,
                              fr=MonoL1(1e-8), ir=MonoL1(1e-8))
    s1 = QuantizerConfigScope(default_q_type='kif', place='datalane', overflow_mode='wrap',
                              f0=bw, fr=MonoL1(1e-8), ic=MinMax(0,12))
    with s0, s1, LayerConfigScope(beta0=0):
        return get_gnn_w(N=N, n=n, W=W)

if __name__ == "__main__":
    N = int(sys.argv[1]); W = int(sys.argv[2])
    m = build(N, 3, W)
    inp, out = trace_model(m, solver_options={"hard_dc": 2}, hwconf=HWConfig(1, -1, -1),
                           inputs_kif=INPUTS_KIF)
    sol = comb_trace(inp, out); psol = to_pipeline(sol, latency_cutoff=LATENCY_CUTOFF)
    r = dict(tag=f"jedi_N{N}_W{W}", N=N, W=W, lut=float(psol.cost), ff=int(psol.reg_bits),
             stages=len(psol.solutions), latency_ns=len(psol.solutions)*CLOCK_NS,
             params=int(m.count_params()))
    print("RESULT " + json.dumps({k: r[k] for k in ("tag","N","W","lut","stages","latency_ns","params") if k in r}), flush=True)
    json.dump(r, open(f"jedi_scaled_N{N}_W{W}.json","w"), indent=1)
