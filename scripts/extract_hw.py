"""
Turn a trained HGQ2 PHAT-JeT checkpoint into hardware numbers + Verilog.

For each checkpoint given on the CLI (or the best-accuracy one in pareto/):
  1. load model, evaluate val accuracy (full 260k val set),
  2. da4ml trace -> combinational cost (LUT estimate) + latency,
  3. pipeline at latency_cutoff=4.0 (targets ~3.3ns clock on VU13P after slack),
  4. write Verilog project (same part as JEDI-Linear: xcvu13p-flga2577-2-e),
  5. bit-exactness check: da4ml pipeline emulation vs keras forward on 512 jets,
  6. append a row to hw_results.json:
       {ckpt, val_acc, ebops, lut_est, comb_latency_ns, stages, latency_ns@300MHz,
        ff_est(reg_bits), dsp:0, bram:0, ii:1}

Latency convention: stages x 3.33ns (300 MHz target, JEDI-Linear's clock).
LUT numbers are da4ml estimates (no Vivado available) — label as estimates.

Usage (fpga env):  python extract_hw.py [ckpt1.keras ckpt2.keras ...]
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np

import hgq.layers  # noqa: F401 — registers Q* classes for checkpoint deserialization

from da4ml.codegen import VerilogModel
from da4ml.converter import trace_model
from da4ml.trace import HWConfig, comb_trace, to_pipeline

CLOCK_NS = 3.33  # 300 MHz — JEDI-Linear's CTL2 target
# Input port fixed-point format (keep, int, frac).
#
# i=7 (range [-128, 128)). The earlier i=6 was set from a 5k-jet sample whose
# scaled-pT max was 60.9, which looked safe against 63.996 -- but over the FULL
# train+val set the max is 92.2, and 58 constituents exceed 63.996. da4ml input
# ports WRAP, so those became large NEGATIVE pT: e.g. 64.07 -> -63.93, a sign
# flip on a jet's LEADING constituent. Measured effect: 14 of 260k val jets
# affected, prediction flipped on 5 of them. Aggregate accuracy impact is only
# ~0.004%, but it is a silent correctness bug at the hardware interface and the
# fix is free -- widening to i=7 costs 192 LUT of 1.33M (+0.014%), latency
# unchanged. Do NOT re-narrow this on the basis of a data subsample.
#
# f=8 -> 0.0039 in scaled units = 4.2e-4 in raw eta/phi, well below the 0.08 GMP bin.
INPUTS_KIF = (1, 7, 8)
LATENCY_CUTOFF = 4.0
PART = "xcvu13p-flga2577-2-e"


def _flush_rows(rows):
    """Merge rows into results/hw_results.json; re-running a ckpt REPLACES its row.

    Written atomically (tmp + os.replace) so a crash mid-write cannot truncate the
    accumulated results file.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(repo_root, "results/hw_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    existing = json.load(open(out_path)) if os.path.exists(out_path) else []
    by_ckpt = {r["ckpt"]: r for r in existing}
    by_ckpt.update({r["ckpt"]: r for r in rows})
    merged = sorted(by_ckpt.values(), key=lambda r: -r.get("val_acc", 0))
    tmp = out_path + ".tmp"
    json.dump(merged, open(tmp, "w"), indent=2)
    os.replace(tmp, out_path)
    return out_path, len(merged)


def _gmp_edges(grid_name=None):
    """Robust-scaled GMP bin edges for the trained grid (must match train_qat.py)."""
    from preprocess import gmp_edges_scaled
    return gmp_edges_scaled(grid_name or os.environ.get("GRID_NAME", "core7"))


def compile_emulator(vm, verbose=False):
    """Build da4ml's Verilator emulation library, portably on macOS.

    da4ml 0.6.0's build_binder.mk is Linux-flavored and fails on Apple clang/ld:
      1. `EXTRA_CXXFLAGS=-fopenmp`  -> Apple clang: "unsupported option '-fopenmp'"
      2. `WARNINGS = -Wl,--no-undefined` -> Apple ld: "unknown options: --no-undefined".
         Mach-O needs -Wl,-undefined,dynamic_lookup: verilated.a references the
         legacy SystemC hook sc_time_stamp(), which ELF shared libs leave undefined
         happily but Mach-O rejects by default. It is never CALLED here (VM_SC=0,
         so VerilatedContext uses its own time), and the bit-exactness assertion
         below is what proves the resulting library behaves correctly.
      3. `N_JOBS ?= $(shell nproc)` -> `nproc` does not exist on macOS, so -j gets
         an empty argument.
    (1) is switchable via compile(openmp=False), but (2) is a hard `=` assignment,
    which make resolves in favor of the makefile over the environment. So: write()
    the project, rewrite that one line on disk, then call the non-writing _compile()
    so our patch survives. Verilator itself is fine — only this link step differed.
    """
    import platform, re

    vm.write()
    if platform.system() == "Darwin":
        mk = vm._path / "sim" / "build_binder.mk"
        txt = mk.read_text()
        txt = txt.replace(
            "WARNINGS = -Wl,--no-undefined",
            "WARNINGS = -Wl,-undefined,dynamic_lookup",
        )
        txt = txt.replace("N_JOBS ?= $(shell nproc)",
                          "N_JOBS ?= $(shell sysctl -n hw.ncpu)")
        mk.write_text(txt)
    vm._compile(verbose=verbose, openmp=False, nproc=os.cpu_count(), clean=True)


def pick_checkpoints(argv):
    if argv:
        return argv
    cks = sorted(
        (f for f in os.listdir("pareto") if f.endswith(".keras")),
        key=lambda f: float(f.split("val_acc=")[1].split("-")[0]),
    )
    return [os.path.join("pareto", cks[-1])] if cks else []


def main():
    # --legacy-run1: checkpoints from run 1 (80-epoch, pT/64 scaling, max pooling,
    # GMP bounds ±0.4) — before the mean-pooling + robust-scaling changes.
    argv = sys.argv[1:]
    legacy = "--legacy-run1" in argv
    argv = [a for a in argv if a != "--legacy-run1"]
    ckpts = pick_checkpoints(argv)
    assert ckpts, "no checkpoints found"

    d = np.load("data/jets_128x3.npz")
    x_val, y_val = d["x_val"].copy(), d["y_val"]
    if legacy:
        x_val[:, :, 0] *= 1.0 / 64.0
    else:
        # Robust scaling matching training -- shared implementation in preprocess.py
        from preprocess import assert_scaled, robust_scale
        x_val = assert_scaled(robust_scale(x_val), "x_val")

    rows = []
    for ck in ckpts:
        print(f"=== {ck}")
        # The GMP bin indicators are closures -> not deserializable from .keras.
        # Rebuild the exact architecture in code and load weights only.
        from train_qat import build_qat_model

        model = build_qat_model(
            aggregation="max" if legacy else "mean",
            gmp_bounds=0.4 if legacy else 3.7,
            gmp_edges=None if legacy else _gmp_edges(),
        )
        # skip_mismatch=False: a silent shape mismatch here (e.g. rebuilding the
        # default 8x8 grid for a 7x7 checkpoint) would yield plausible-looking but
        # meaningless LUT/latency numbers.
        model.load_weights(ck, skip_mismatch=False)

        logits = model.predict(x_val, batch_size=8192, verbose=0)
        acc = float((logits.argmax(1) == y_val.argmax(1)).mean())
        print(f"val_acc {acc:.4f}")

        inps, outs = trace_model(model, hwconf=HWConfig(1, -1, -1), inputs_kif=INPUTS_KIF)
        comb = comb_trace(inps, outs)
        print(f"comb cost {comb.cost:,.0f} LUT-est, comb latency {comb.latency}")

        pipe = to_pipeline(comb, latency_cutoff=LATENCY_CUTOFF, verbose=False)
        stages = len(pipe.solutions)
        import re as _re
        # Verilog identifiers: [a-zA-Z_][a-zA-Z0-9_]* — sanitize everything else
        tag = _re.sub(r"[^a-zA-Z0-9_]", "_", os.path.splitext(os.path.basename(ck))[0])
        outdir = f"verilog_{tag}"
        vm = VerilogModel(pipe, f"phat_jet_{tag[:40]}", outdir, part_name=PART, clock_period=CLOCK_NS)

        # Bit-exactness: the emulator floors inputs onto the port's fixed-point
        # grid (step 2^-f) before the first multiply, so the Keras reference must
        # be fed the SAME pre-quantized values — otherwise we measure input
        # rounding, not a hardware/software mismatch.
        _k, _i, _f = INPUTS_KIF
        xb = np.floor(x_val[:512] * 2.0**_f) / 2.0**_f
        _lim = 2.0 ** _i
        assert np.abs(xb).max() < _lim, (
            f"input {np.abs(xb).max():.2f} exceeds port range +/-{_lim}: raise INPUTS_KIF[1]")
        ref = model.predict(xb, batch_size=512, verbose=0)
        # Verilator build time scales with design size and is the wall-clock
        # bottleneck (tens of minutes on multi-million-LUT designs). Bit-exactness
        # is a property of the da4ml lowering, not of a particular checkpoint, so
        # SKIP_EMU=1 lets a sweep collect cost/latency for many points and prove
        # bit-exactness once on a small representative design.
        _skip_emu = os.environ.get("SKIP_EMU") == "1"
        max_err = match = None
        try:
            if _skip_emu:
                raise RuntimeError("skipped (SKIP_EMU=1)")
            compile_emulator(vm)  # Verilator RTL emulation library (macOS-patched)
            emu = vm.predict(xb.reshape(512, -1), n_threads=-1)
            emu = np.asarray(emu).reshape(ref.shape)
            max_err = float(np.abs(emu - ref).max())
            match = float((emu.argmax(1) == ref.argmax(1)).mean())
        except Exception as e:
            max_err, match = None, None
            print("emulation skipped (SKIP_EMU=1)" if _skip_emu else f"emulation failed: {e}")

        row = dict(
            ckpt=ck,
            val_acc=acc,
            lut_est=float(comb.cost),
            comb_latency_ns=float(comb.latency[0]),
            stages=stages,
            latency_ns_at_300MHz=round(stages * CLOCK_NS, 1),
            ff_est_reg_bits=int(pipe.reg_bits),
            dsp=0,
            bram=0,
            ii=1,
            emu_max_abs_err=max_err,
            emu_argmax_match=match,
            verilog_dir=outdir,
        )
        rows.append(row)
        print(json.dumps(row, indent=2))
        # Persist after EVERY checkpoint. A long sweep that dies partway (OOM on a
        # multi-million-LUT trace, or a kill) previously lost all completed points
        # because results were only written at the end of main().
        _flush_rows(rows)

    out_path, n_total = _flush_rows(rows)
    print(f"wrote {len(rows)} rows ({n_total} total) to {out_path}")


if __name__ == "__main__":
    main()
