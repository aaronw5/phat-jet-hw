# PHAT-JeT on FPGA — grid + QAT configuration of record

Everything below is the *verified* configuration: each number was measured, not
assumed. Reproduction commands are at the bottom. Dataset is the hls4ml 5-class
jet-tagging set (g/q/W/Z/t), 128 constituents × 3 features (pT, Δη, Δφ), matched to
JEDI-Linear's configuration.

---

## 1. GMP grid (the fixed-grid question)

### What changed and why

The paper's GMP uses a **per-jet dynamic grid**: for each jet the constituent
coordinates are min-shifted and scaled so the occupied region fills the grid. That is
data-dependent addressing — the bin boundaries are functions of the input — and it is
untraceable for the distributed-arithmetic converter (da4ml), which needs a fixed
dataflow graph of constant-threshold comparisons. So the hardware model uses a
**fixed grid**: bin edges are compile-time constants.

The reviewer-relevant question is whether freezing the grid degrades the method. It
does not, *provided the fixed edges are placed where the jet physics is*. That is what
this section establishes.

### Chosen grid: `core7` — 7×7 non-uniform

```python
# raw eta/phi units (jet-relative), identical on both axes
EDGES_RAW = [-0.40, -0.20, -0.12, -0.04, 0.04, 0.12, 0.20, 0.40]
# bin widths:    0.20   0.08   0.08  0.08   0.08   0.08  0.20
```

7 bins per axis → **49 cells** (vs 64 for the 8×8 uniform baseline).

The network sees robust-scaled coordinates, so the edges are divided by the
angular IQR at build time:

```python
iqr = 0.5 * (robust_stats[1]["iqr"] + robust_stats[2]["iqr"])  # = 0.10804
gmp_edges = (EDGES_RAW / iqr).tolist()
```

`data/robust_stats.json` (median, IQR per feature, train-set, nonzero constituents only):

| feature | median | IQR |
|---|---|---|
| pT | 6.5956 | 17.65079 |
| Δη | −1.2412e-05 | 0.108128 |
| Δφ | 6.8791e-06 | 0.107955 |

### Why these edges — three measured physics arguments

**(1) Half-extent ±0.40 is set by pT containment, not by the jet radius.**
Measured on 200k train jets, the fraction of total constituent pT falling outside
±B (and therefore clamped into the edge bins):

| B | pT outside | constituents outside |
|---|---|---|
| 0.25 | 0.62% | 2.71% |
| 0.30 | 0.21% | 1.30% |
| 0.35 | 0.06% | 0.51% |
| **0.40** | **0.007%** | **0.13%** |
| 0.45 | 0.001% | 0.03% |

±0.40 clamps 0.007% of the jet's energy. Going wider buys nothing measurable and
costs cells quadratically. This is the fixed-grid replacement for the dynamic
grid's adaptive extent: the outermost bins are open-ended (clamping), so nothing
is *lost* outside ±0.40 — it is merely spatially coarse, which is appropriate
because almost no energy is there.

**(2) The bins must be fine in the core and coarse outside, because that is how
the energy is distributed.** pT-weighted radial containment (fraction of total jet
pT within radius r), per class:

| class | 50% | 90% | 95% | 99% |
|---|---|---|---|---|
| g | 0.062 | 0.204 | 0.253 | 0.336 |
| q | 0.043 | 0.175 | 0.230 | 0.323 |
| W | 0.055 | 0.170 | 0.212 | 0.302 |
| Z | 0.056 | 0.171 | 0.213 | 0.303 |
| t | 0.072 | 0.203 | 0.246 | 0.328 |

Half the jet's momentum sits inside r ≈ 0.05, i.e. inside a *single* 0.1-wide
uniform bin. A uniform grid spends the same resolution on the empty periphery as on
the core. `core7` spends 0.08-wide bins on |Δη|,|Δφ| < 0.20 (where the pT is) and
one wide 0.20 bin on each outer edge (where it is not). Non-uniform edges cost
nothing extra in hardware — each bin is one threshold comparison either way.

**(3) The bin count must be ODD.** This is the argument that actually decided the
configuration, and it is specific to the fixed grid. With an even, symmetric bin
count, a bin *boundary* falls exactly at Δη=Δφ=0 — the jet axis — so the leading
prong's energy is split across four cells. The paper's dynamic grid never had this
problem: min-shifting per jet moved the core away from the boundary. An odd count
centers a cell on the origin. Measured median pT share captured by the single
hottest cell (core coherence), B=0.40:

| G | spacing | origin lands on | hottest-cell pT share |
|---|---|---|---|
| 7 | 0.114 | **cell center** | **47.5%** |
| 8 | 0.100 | boundary | 33.9% |
| 9 | 0.089 | **cell center** | 43.9% |
| 11 | 0.073 | **cell center** | 39.6% |
| 12 | 0.067 | boundary | 27.4% |
| 13 | 0.062 | **cell center** | 36.6% |
| 16 | 0.050 | boundary | 22.6% |

The odd/even alternation is larger than the trend with spacing. This is a
fixed-grid–specific artifact and the single most important design choice here.

Supporting scales (why 0.08 core bins are the right resolution): the expected
2-prong opening angle 2m/pT is 0.146 (W), 0.152 (Z), 0.335 (t) median — so 0.08
bins resolve both prongs of a W/Z into distinct cells, while the 3×3 depthwise
conv still sees them as neighbors. pT-weighted jet width √⟨r²⟩ is 0.113 (g),
0.099 (q), 0.101 (W), 0.101 (Z), 0.126 (t).

### Grid shootout (the accuracy-per-resource tiebreaker)

The static metrics rank the candidates but cannot price them, and the paper's own
grid-spacing appendix shows accuracy is flat across the working spacing range. So
three candidates were trained **identically** (same seed, data, schedule, 25 epochs,
300k jets) differing *only* in the grid:

| Grid | Cells | Best val acc | EBOPs at best | acc @ep10 | acc @ep15 | acc @ep20 |
|---|---|---|---|---|---|---|
| 8×8 uniform ±0.40 (baseline) | 64 | 0.6993 | 1.64e7 | 0.6905 | 0.6981 | 0.6764 |
| **7×7 non-uniform (`core7`)** | **49** | **0.7190** | **1.37e7** | 0.7093 | 0.7190 | 0.7018 |
| 9×9 uniform ±0.45 | 81 | 0.7255 | 1.71e7 | 0.7135 | 0.7190 | 0.6673 |

**`core7` dominates the baseline outright**: +2.0 accuracy points at 17% lower EBOPs
and 23% fewer cells. It ties 9×9 at matched epochs (0.7190 both at epoch 15) using
40% fewer cells and 20% lower EBOPs. Both odd-count grids beat the even-count
baseline by ~2 points, exactly as argument (3) predicts.

(The dip at epoch ~20–25 in all three is the β schedule compressed into a 25-epoch
proxy run over-penalizing at the end — it hits all three identically, so it does not
affect the ranking, and it is absent from the 1000-epoch run.)

### Does the fixed grid "act the same" as the dynamic one?

Yes — and better than the even-count fixed grid it replaces. Float accuracy of the
LN-free hardware architecture with `core7` fixed edges:

| | Accuracy |
|---|---|
| JEDI-Linear, published, pT-sorted | 80.9% |
| JEDI-Linear, published, perm-invariant | 81.6% |
| **This model, fixed `core7` grid, float** | **81.59%** |

The fixed grid is not costing accuracy relative to the dynamic grid: it lands level
with JEDI-Linear's permutation-invariant row and above their pT-sorted headline.

---

## 2. QAT configuration

Recipe follows HGQ2's `jsc150` reference (`run_train.py`) except where noted.

### Architecture / data

* LN-free hardware variant (`use_ln=False`) — LayerNorm is not synthesized.
* `aggregation="mean"`.
* Robust scaling by `data/robust_stats.json`; zero-pad rows re-zeroed *after* scaling.
* Full 620k train set, 208k val.

### Two-stage training (this is the part that mattered)

**Stage 1 — float pretrain of the EXACT hardware architecture**
(`scripts/float_pretrain.py`): same LN-free, `core7`-grid model, no quantizers.
Adam, LR 3e-3, batch 2790, early stopping patience 15.
Result: **81.59% val accuracy** (best at epoch ~45–47 of 60).

Why this stage exists: `jsc150` initializes QAT from a float model. An earlier
attempt (run 3) collapsed to chance and was blamed on a LayerNorm mismatch in the
donor checkpoint. Training a donor with *identical* topology removes that variable
so the transfer is 1:1 by layer name and shape.

**Stage 2 — QAT from those weights** (`scripts/train_qat.py`), 1000 epochs.

### Initial bitwidths — the real cause of the run-3 collapse

`jsc150` uses fractional bits 7 and integer bits 0, which is correct *from scratch*
(weights and activations start ≪ 1; integer bits grow on demand via
`i_decay_speed`). Initializing from a float checkpoint, the incoming ranges are
already large, and `overflow_mode="wrap"` **wraps** them — a weight of 1.796 becomes
negative. Measured on `float_phatjet_core7.keras`:

* max|w| = 1.796 (`embed`) → needs ≥2 integer bits
* max|activation| = **85.75** (`ffn2` output) → needs ≥8 integer bits
  (`embed` 30.67, `ffn1` 37.52, `head1` 29.13, `head_out` 12.73)

The float model was never trained under an activation-magnitude penalty, so it has
no reason to keep activations small. Post-transfer val accuracy (20k val jets),
measured:

| weight frac bits | datalane int bits | post-transfer val_acc |
|---|---|---|
| 7 | 0 | 0.2056 ← **chance; this was run 3** |
| 7 | 8 | 0.7437 |
| 8 | 8 | 0.8044 |
| 10 | 8, weight int 3 | **0.8113** ← chosen |
| 12 | 9 | 0.8115 |

Note more weight integer bits alone did *not* help (0.2056 → 0.2130 at i0_w=3 with
i0_a=0): the collapse was **datalane** integer-bit starvation, not weight wrapping.
Saturation instead of wrap does not fix it either (0.2505).

Config of record:

```python
BW_K = 10          # initial weight fractional bits
BW_A = 10          # initial datalane fractional bits
I0_TRANSFER = 3    # initial weight integer bits   (float-init only)
I0_A_TRANSFER = 8  # initial datalane integer bits (float-init only)
L1_REG = 1e-8      # MonoL1 on f and i
i_decay_speed = 1e-3
ic = MinMax(0, 12) # datalane integer-bit constraint
overflow_mode = "wrap"
```

Starting wide costs nothing in the final hardware: β compresses bitwidths *downward*
from here, which is the intended direction of travel. `train_qat.py` now asserts
post-transfer accuracy > 0.5 so this failure can never again be silent.

### Optimizer / schedule

```python
EPOCHS = 1000            # jsc150 uses 7000 on GPU; 1000 is the CPU budget
BATCH  = 2790            # jsc150 default
LR     = 3e-3            # jsc150 default, cosine restarts
beta_sched = BetaScheduler(PieceWiseSchedule([
    (0,                        2e-8, "linear"),
    (int(EPOCHS * 2000/7000),  3e-7, "log"),     # = epoch 285
    (EPOCHS,                   3e-6, "constant"),
]))
```

The β ramp traces the accuracy-vs-EBOPs Pareto frontier in a *single* run;
`ParetoFront` checkpoints every non-dominated (val_accuracy, EBOPs) epoch.
Deviation from the reference recipe, stated honestly: 7000 GPU epochs → 1000 CPU
epochs, β schedule scaled proportionally.

### Status / result

Post-transfer start **81.21%**; epoch 0 val accuracy 79.13% (versus ~69% for the
best from-scratch run — the two-stage change is worth ~10 points at this budget).
Frontier snapshot (`ckpt_snapshots_run6/`):

| epoch | val_acc | EBOPs |
|---|---|---|
| 2 | 0.8113 | 115.8M |
| 11 | 0.8079 | 36.0M |
| 15 | 0.7970 | 18.9M |
| 34 | 0.7911 | 5.65M |
| 141 | 0.7854 | 0.92M |
| 235 | 0.7359 | 0.55M |

Note the 126× EBOPs compression from epoch 2 → 141 for 2.6 accuracy points.

---

## 3. Hardware extraction

```python
PART           = "xcvu13p-flga2577-2-e"   # matches JEDI-Linear
CLOCK_NS       = 3.33                     # 300 MHz, JEDI-Linear's CTL2 target
LATENCY_CUTOFF = 4.0
INPUTS_KIF     = (1, 6, 8)                # (keep, integer, fractional)
```

`INPUTS_KIF` integer bits = 6, not 5: robust-scaled pT reaches **60.90** for the
hardest constituents, and i=5 saturates at 32 and wraps — which showed up as
max|err| ≈ 9.8 in the bit-exactness check. f=8 gives 0.0039 in scaled units =
4.2e-4 raw, far below the 0.08 core bin width.

### Bit-exactness verification

RTL is generated by da4ml, simulated with Verilator, and compared against Keras on
512 validation jets. Verified result on `qat_final.keras`:

**max absolute error = 0.0, argmax match = 1.000** — bit-exact.

Two things had to be right for this to be meaningful:

1. The Keras reference must be fed inputs **pre-quantized to the port grid**
   (`floor(x · 2^f)/2^f`), because the emulator floors them before the first
   multiply. Otherwise the check measures input rounding, not a HW/SW mismatch.
2. `load_weights(ck, skip_mismatch=False)` — a silent shape mismatch (e.g.
   rebuilding the default 8×8 grid for a 7×7 checkpoint) would produce
   plausible-looking but meaningless LUT/latency numbers.

### macOS portability (da4ml 0.6.0)

`build_binder.mk` is Linux-flavored; three things fail on Apple clang/ld. Patched in
`extract_hw.py::compile_emulator()` between `write()` and `_compile()`:

| Problem | Fix |
|---|---|
| `EXTRA_CXXFLAGS=-fopenmp` — Apple clang has no OpenMP | `_compile(openmp=False)` |
| `WARNINGS = -Wl,--no-undefined` — GNU-ld spelling | rewrite to `-Wl,-undefined,dynamic_lookup` |
| `N_JOBS ?= $(shell nproc)` — no `nproc` on macOS | rewrite to `sysctl -n hw.ncpu` |

`dynamic_lookup` is needed because `verilated.a` references the legacy SystemC hook
`sc_time_stamp()`, which ELF shared libs leave undefined happily but Mach-O rejects.
It is never *called* (`VM_SC=0`), and the bit-exactness result above is the proof
that the resulting library behaves correctly.

Note `WARNINGS` is a hard `=` assignment, so make resolves it in favor of the
makefile over the environment — it must be patched on disk, and `_compile()` (the
non-writing entry point) must be used so the patch survives.

### Caveat to state in the paper/rebuttal

These are **pre-place-and-route** estimates from da4ml's LUT model, not Vivado
post-P&R numbers (no Vivado license available). JEDI-Linear's published numbers are
post-P&R. Label accordingly in any comparison table.

---

## 4. Reproduction

```bash
cd /Users/anrunw/Documents/phat-jet-hw
PY=$(python -c "import sys; print(sys.executable)")   # conda env: fpga

# grid shootout (3 runs, ~25 epochs each; run in parallel, 4 threads each)
for g in uniform8 core7 uniform9; do
  SHOOTOUT_THREADS=4 $PY scripts/grid_shootout.py $g > logs/shootout_$g.log 2>&1 &
done

# stage 1: float pretrain of the exact hardware architecture
GRID_NAME=core7 OMP_NUM_THREADS=12 $PY scripts/float_pretrain.py 60

# stage 2: 1000-epoch QAT from those weights
QAT_EPOCHS=1000 GRID_NAME=core7 RUN_TAG=run6_core7_1000ep QAT_INIT=float \
  OMP_NUM_THREADS=12 $PY scripts/train_qat.py

# hardware numbers + bit-exactness for chosen Pareto points
GRID_NAME=core7 QAT_INIT=float $PY scripts/extract_hw.py ckpt_snapshots_run6/epoch=*.keras
```

Key files: `scripts/phat_jet_k3.py` (model, `onehot_bins` takes an ascending edge
list), `scripts/grid_shootout.py` (`GRIDS` dict — the grid definitions of record),
`scripts/float_pretrain.py`, `scripts/train_qat.py`, `scripts/extract_hw.py`,
`data/robust_stats.json`, `results/hw_results.json`, `docs/grid_study.md`.
