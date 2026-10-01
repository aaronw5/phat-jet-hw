"""Verilator RTL-vs-Keras bit-exactness check for the deployable checkpoint.

Two macOS incompatibilities in da4ml's generated build_binder.mk had to be
worked around; NEITHER is a Verilator problem (verilator 5.050 builds fine):
  1. WARNINGS = -Wl,--no-undefined  -- GNU ld only; Apple ld errors with
     "ld: unknown options: --no-undefined".
  2. N_JOBS ?= $(shell nproc)       -- nproc does not exist on macOS.
  3. compile(openmp=True) adds -fopenmp, which Apple clang++ rejects.
We patch the generated makefile in the project dir (site-packages is read-only)
and pass openmp=False + explicit nproc.
"""
import os, sys, json, time, re, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("GRID_NAME", "core7")
from train_qat import build_qat_model
from extract_hw import _gmp_edges
from preprocess import load_jets
from da4ml.converter import trace_model
from da4ml.trace import comb_trace, HWConfig
from da4ml.codegen import VerilogModel

CK = os.environ.get("BE_CKPT", "ckpt_snapshots_run6/epoch=140-val_acc=0.7864-ebops=926777.keras")
PRJ = os.environ.get("BE_PRJ", "bitexact_prj3")
N = int(os.environ.get("BE_N", 2000))

Xv, Yv = load_jets("val")
m = build_qat_model(aggregation="mean", gmp_bounds=3.7, gmp_edges=_gmp_edges())
m.load_weights(CK, skip_mismatch=False)

inp, out = trace_model(m, solver_options={"hard_dc": 2}, hwconf=HWConfig(1, -1, -1))
sol = comb_trace(inp, out)
vm = VerilogModel(sol, prj_name="be", path=PRJ, latency_cutoff=2)
vm.write()

mk = os.path.join(PRJ, "sim", "build_binder.mk")


def patch_mk():
    src = open(mk).read()
    src = src.replace("WARNINGS = -Wl,--no-undefined", "WARNINGS =")
    src = src.replace("N_JOBS ?= $(shell nproc)", "N_JOBS ?= 8")
    open(mk, "w").write(src)


# compile(clean=True) runs `make clean`, which REGENERATES build_binder.mk from
# the (read-only) site-packages template and silently reverts the patch -- the
# verilator step then succeeds and only the final link fails. So: clean first,
# patch after, and compile with clean=False.
# Patching the makefile is not reliable: write() copies it fresh from the
# read-only template (rtl_model.py:221) and `make clean` re-triggers that, so
# the -Wl,--no-undefined line keeps coming back. Instead, interpose a CXX
# wrapper that strips the GNU-only flag before invoking the real compiler --
# nothing da4ml regenerates can undo this.
import subprocess, stat
patch_mk()
wrapper = os.path.abspath(os.path.join(PRJ, "sim", "cxx_nognu"))
os.makedirs(os.path.dirname(wrapper), exist_ok=True)
with open(wrapper, "w") as f:
    f.write('#!/bin/sh\n'
            '# drop flags Apple ld does not understand\n'
            'args=""\n'
            'for a in "$@"; do\n'
            '  case "$a" in -Wl,--no-undefined|--no-undefined) ;; \n'
            '  *) args="$args \'$a\'" ;; esac\n'
            'done\n'
            'eval exec clang++ $args\n')
os.chmod(wrapper, os.stat(wrapper).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
print("installed CXX wrapper stripping --no-undefined", flush=True)

t = time.time()
os.environ["CXX"] = wrapper  # make picks $(CXX) up from the environment

# The real link error that -Wl,--no-undefined was reporting: verilated.a
# references sc_time_stamp(), which the embedding application must define.
# da4ml's binder is purely combinational, so time never advances.
with open(os.path.join(PRJ, "sim", "vl_stub.cc"), "w") as f:
    f.write("#include <verilated.h>\ndouble sc_time_stamp() { return 0; }\n")
mksrc = open(mk).read()
if "vl_stub.cc" not in mksrc:
    mksrc = mksrc.replace("$(VM_PREFIX)_binder.cc ./obj_dir/",
                          "$(VM_PREFIX)_binder.cc vl_stub.cc ./obj_dir/")
    open(mk, "w").write(mksrc)
assert "vl_stub.cc" in open(mk).read(), "stub not wired into link line"

# compile() calls write() first (rtl_model.py:364), which re-copies the makefile
# template and reverts our patches. Call the inner _compile directly instead.
vm._compile(nproc=8, openmp=False, clean=False)
vm._load_lib()
print(f"verilator + link OK in {time.time()-t:.0f}s", flush=True)

X = np.ascontiguousarray(Xv[:N]).astype(np.float32)
pred = vm.predict(X)
ref = np.asarray(m(Xv[:N], training=False))
ndiff = int(np.sum(np.any(pred - ref != 0, axis=1)))
res = dict(ckpt=os.path.basename(CK), n_samples=N, n_mismatch_rows=ndiff,
           max_abs_diff=float(np.max(np.abs(pred - ref))),
           rtl_acc=float((pred.argmax(1) == Yv[:N].argmax(-1)).mean()),
           keras_acc=float((ref.argmax(1) == Yv[:N].argmax(-1)).mean()),
           bit_exact=bool(ndiff == 0))
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open("bitexact_result.json", "w"), indent=1)
print("RESULT:", "BIT-EXACT" if ndiff == 0 else "MISMATCH", flush=True)
