"""Head-to-head FPGA-estimate benchmark: PHAT-JeT vs JEDI-linear.

SAME INSTRUMENT for both models -- this is the whole point. We reproduce
JEDI-linear's own syn_test_verilog settings exactly (src/syn_test.py):

    trace_model(model, solver_options={'hard_dc': 2}, hwconf=HWConfig(1, -1, -1))
    solution = comb_trace(inp, out)
    VerilogModel(..., clock_period=2, clock_uncertainty=0.0, latency_cutoff=2)
    da_est_LUT = round(vm._pipe.cost)
    da_est_FF  = round(vm._pipe.reg_bits)

Note _pipe.cost (POST-PIPELINING), not comb.cost (combinational). Our own
extract_hw.py reported comb.cost at 300 MHz, which is a DIFFERENT quantity --
that mismatch is why earlier LUT comparisons were not valid.

These are da4ml ESTIMATES, not Vivado place-and-route results. No Vivado on
this machine. Estimates are only comparable to other estimates from the same
pipeline -- which is exactly what this script produces.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import keras

# ---- JEDI-linear's exact trace settings ----
SOLVER_OPTIONS = {"hard_dc": 2}
HWCONF_ARGS = (1, -1, -1)
CLOCK_PERIOD = 2.0        # ns -> 500 MHz
CLOCK_UNCERTAINTY = 0.0
LATENCY_CUTOFF = 2
PART = "xcvu13p-flga2577-2-e"


def trace_cost(model, tag, outdir="bench_prjs", emulate=False, X=None, Y=None):
    """Trace a keras model -> (LUT, FF, latency_ns, stages). JEDI-linear settings."""
    from da4ml.converter import trace_model      # da4ml 0.6.0 path
    from da4ml.trace import comb_trace, HWConfig, to_pipeline
    from da4ml.codegen import VerilogModel

    inp, out = trace_model(model, solver_options=SOLVER_OPTIONS,
                           hwconf=HWConfig(*HWCONF_ARGS))
    sol = comb_trace(inp, out)
    os.makedirs(outdir, exist_ok=True)
    # JEDI-linear passes latency_cutoff to VerilogModel, which internally calls
    # to_pipeline. da4ml 0.6.0 exposes to_pipeline directly; same computation,
    # and it lets us read stages without constructing Verilog.
    pipe = to_pipeline(sol, latency_cutoff=LATENCY_CUTOFF, verbose=False)
    stages = len(pipe.solutions)
    res = dict(
        tag=tag,
        da_est_LUT=round(float(pipe.cost)),
        da_est_FF=round(float(pipe.reg_bits)),
        comb_cost=round(float(sol.cost)),
        comb_latency=float(np.max(sol.latency)),
        pipeline_stages=stages,
        latency_ns=round(stages * CLOCK_PERIOD, 1),
        clock_period_ns=CLOCK_PERIOD,
        DSP=0, BRAM=0, II=1,
    )
    if emulate and X is not None:
        try:
            from extract_hw import compile_emulator  # macOS-safe Verilator build
            vm = VerilogModel(sol, prj_name=f"bench_{tag}", path=os.path.join(outdir, tag),
                              part_name=PART, clock_period=CLOCK_PERIOD,
                              clock_uncertainty=CLOCK_UNCERTAINTY, latency_cutoff=LATENCY_CUTOFF)
            compile_emulator(vm)
            pred = vm.predict(np.ascontiguousarray(X).astype(np.float32))
            res["rtl_acc"] = float((pred.argmax(1) == np.asarray(Y).argmax(-1)).mean())
            ref = np.asarray(model(X, training=False))
            res["bit_exact_ndiff"] = int(np.sum(np.any(pred - ref != 0, axis=1)))
        except Exception as e:
            res["rtl_error"] = str(e)[:200]
    return res
