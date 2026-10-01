"""Attention-factorization shootout: hardware cost of the SAME architecture with
the attention pattern varied, everything else held fixed.

Why this exists (rebuttal, reviewer rHHc): the reviewers' objection is that FLOPs
and parameter counts do not establish anything about the target deployment. This
script measures the design space through ONE instrument (da4ml, JEDI-linear's own
synthesis settings) so the FLOP-vs-LUT divergence is measured, not asserted.

Key configurations, all with identical d_model / heads / GMP / FFN / N:
  patch_size == N, use_patch_attn=False  ->  EXACT full self-attention baseline
  patch_size == 8, use_patch_attn=True   ->  PHAT-JeT hierarchical (ours)
plus intermediate patch sizes to show the trend, and the parallel-branch variant.

Untrained float-init weights at a FIXED quantizer scope (BW 6), identical for every
row, so rows are comparable to each other. These are NOT the deployable trained
numbers -- those come from the trained Pareto frontier in nsweep_results.json.
"""
import os, sys, json

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phat_jet_k3 import build_phat_jet_k3
from extract_hw import CLOCK_NS, INPUTS_KIF, LATENCY_CUTOFF
from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.constraints import MinMax
from hgq.regularizers import MonoL1
from da4ml.converter import trace_model
from da4ml.trace import comb_trace, HWConfig, to_pipeline

D_MODEL, HEADS = 16, 4


def macs(N, patch_size, use_patch_attn, use_ffn=True, d=D_MODEL):
    """Analytic multiply-accumulate counts, split by operand type.

    'static' = weight x activation  (foldable into shift-add adder trees)
    'dynamic' = activation x activation (needs a real multiplier in fabric)

    The split is the whole point: FLOP counters add these together, hardware
    does not treat them alike.
    """
    NP = N // patch_size
    static = N * d * 3 * d + N * d * d          # QKV + out proj, local stage
    dynamic = 2 * NP * patch_size * patch_size * d  # scores + AV
    if use_patch_attn:
        static += NP * d * 3 * d + NP * d * d
        dynamic += 2 * NP * NP * d
    if use_ffn:
        static += N * (d * 4 * d + 4 * d * d)
    return static, dynamic


def trace(N, patch_size, use_patch_attn, parallel, use_gmp=True):
    with QuantizerConfigScope(default_q_type="kbi", b0=6, overflow_mode="wrap",
                              i0=2, fr=MonoL1(1e-9), ir=MonoL1(1e-9), i_decay_speed=1e-3), \
         QuantizerConfigScope(default_q_type="kif", place="datalane",
                              overflow_mode="wrap", f0=6, i0=2,
                              fr=MonoL1(1e-9), ic=MinMax(0, 12)), \
         LayerConfigScope(beta0=0):
        m = build_phat_jet_k3(num_particles=N, num_feats=3, d_model=D_MODEL,
                              num_heads=HEADS, patch_size=patch_size, quantized=True,
                              use_gmp=use_gmp, use_local_attn=True,
                              use_patch_attn=use_patch_attn, use_ffn=True,
                              parallel_attn=parallel)
    inp, out = trace_model(m, hwconf=HWConfig(1, -1, -1), inputs_kif=INPUTS_KIF)
    p = to_pipeline(comb_trace(inp, out), latency_cutoff=LATENCY_CUTOFF, verbose=False)
    return len(p.solutions), float(p.cost), int(m.count_params())


CONFIGS = []
for N in (32, 64):
    CONFIGS.append((f"full_self_attn_N{N}", N, N, False, 0))
    for ps in (16, 8, 4):
        if ps < N:
            CONFIGS.append((f"hier_ps{ps}_N{N}", N, ps, True, 0))
    CONFIGS.append((f"hier_ps8_N{N}_parallel", N, 8, True, 1))

rows = []
for tag, N, ps, pa, par in CONFIGS:
    try:
        stages, lut, params = trace(N, ps, bool(pa), bool(par))
        st, dyn = macs(N, ps, bool(pa))
        r = dict(tag=tag, N=N, patch_size=ps, patch_attn=int(pa), parallel=int(par),
                 stages=stages, latency_ns=round(stages * CLOCK_NS, 1),
                 lut=round(lut), params=params,
                 static_macs=st, dynamic_macs=dyn, total_macs=st + dyn)
        rows.append(r)
        print(f"RESULT {tag:26s} {stages:3d}st {r['latency_ns']:6.1f}ns "
              f"LUT {r['lut']:>9,}  MACs {r['total_macs']:>8,} "
              f"(dyn {dyn:>7,})", flush=True)
    except Exception as e:
        print(f"FAIL {tag}: {type(e).__name__}: {str(e)[:100]}", flush=True)
    json.dump(rows, open("attn_shootout.json", "w"), indent=1)
print("DONE", len(rows), flush=True)
