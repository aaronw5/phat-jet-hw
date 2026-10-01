# PHAT-JeT FPGA Hardware Evidence

Hardware implementation flow for **PHAT-JeT** (Patch Hierarchical Attention
Transformer for jet tagging, NeurIPS 2026 submission 28409): HGQ2
quantization-aware training → da4ml distributed-arithmetic conversion →
Verilog for AMD VU13P, producing latency / LUT / FF / DSP / BRAM numbers
comparable to JEDI-Linear ([arXiv:2508.15468](https://arxiv.org/abs/2508.15468)).

Produced for the NeurIPS rebuttal: reviewers asked for FPGA resource evidence
against JEDI-Linear's post-place-and-route results.

## Summary

- **PHAT-JeT can be deployed in the CMS L1 trigger.** At N=64 constituents it
  reaches **74.6%** accuracy at **96.6 ns** latency (300 MHz, II=1). It uses
  **167k LUTs (9.7% of a VU13P)** and no DSPs or BRAM.
- **It was run at five sequence lengths, N = 8, 16, 32, 64 and 128.** N is the
  number of highest-pT constituents per jet. Every N from 16 to 64 has a design
  inside the trigger budget: **67.5% / 71.4% / 74.6%** at **5.0% / 5.7% / 9.7%** of
  the device, all at the same 96.6 ns. Pipeline depth, not N, sets the latency; a
  larger N costs LUTs. Details: [Results by sequence length](#results-by-sequence-length-n).
- Using more of the device raises accuracy: **78.6% at 1.19M LUTs (69%)** at N=128,
  and **78.4% at 1.22M LUTs (71%)** at N=64. Both take 126–133 ns, above the 100 ns budget.
- **The generated RTL matches Keras bit for bit.** A Verilator check over 2,000
  jets gave a max absolute difference of 0.0 for the 78.64% N=128 design.
- **Patch attention saves more hardware than FLOP counts predict.** Compared
  with the same model using full self-attention, it cuts LUTs **2.98x at N=64**
  (FLOPs predict 1.47x) and **1.88x at N=32** (FLOPs predict 1.18x). The reason
  is that activation-by-activation multiplies fall 7.1x and 3.8x respectively,
  and those need real multipliers.
- **Quantization, not the architecture, limits accuracy.** The float model reaches
  81.6% at N=128 and 81.9% at N=64. Short QAT runs lose 6–8 points; the one long
  run lost 2.95. Bitwidth inspection shows the EBOPs regularizer squeezing the GMP
  block to about 1.3 bits.
- **JEDI-Linear remains smaller at equal accuracy.** For example, it reaches
  80.9% at 98k LUTs (post-route). This study claims feasibility only, not a win
  over JEDI-Linear.

## Results

All PHAT-JeT resource numbers are **da4ml estimates**, not Vivado post-place-and-route
(there was no Vivado run). To calibrate the estimator, JEDI-Linear's own released
N=128 model was traced with the same settings. The estimate was 99,690 LUTs against
their published post-route 97,822 (+1.9%), so these estimates lean slightly
conservative. FF estimates are pessimistic (+41%), so treat them as upper bounds.
Latency = pipeline stages × 3.33 ns (300 MHz, the JEDI-Linear firmware target). Each
design was traced at that clock, not rescaled from another one.

**Trigger budget:** under 100 ns latency, II = 1, under 10% of a VU13P (1.728M LUTs),
since Correlator Layer 2 also runs jet clustering.

| PHAT-JeT design | Acc (%) | LUT | % VU13P | Stages | Latency (ns) | Within budget |
|---|---|---|---|---|---|---|
| N=16 | 67.45 | 87,206 | 5.0 | 29 | 96.6 | yes |
| N=32 | 71.36 | 98,069 | 5.7 | 29 | 96.6 | yes |
| **N=64** | **74.58** | **167,304** | **9.7** | 29 | **96.6** | **yes** |
| N=64, max accuracy | 78.42 | 1,221,621 | 70.7 | 40 | 133.2 | LUTs only |
| N=128, bit-exact verified | 78.64 | 1,185,319 | 68.6 | 38 | 126.5 | LUTs only |
| N=128, best accuracy per LUT | 78.60 | 902,337 | 52.2 | — | not traced at 300 MHz | LUTs only |

All rows: 0 DSP, 0 BRAM, II=1. Among the 31 run-6 N=128 checkpoints traced at
300 MHz, none is under 100 ns: the shallowest is 35 stages (116.5 ns, 74.7%).

> **Correction:** earlier versions of this table, and of `comparison_final`, gave the
> N=128 rows as 76 stages / 253.1 ns and 71 stages / 236.4 ns. Those stage counts
> came from traces at a 2 ns cutoff (500 MHz settings) and were then multiplied by
> 3.33 ns. Re-traced at 300 MHz with the same settings as the N=16–64 rows, the
> bit-exact checkpoint is **38 stages = 126.5 ns**, at the same LUT count. The
> 78.60% / 902k point was only traced at the 2 ns cutoff (71 stages × 2 ns = 142 ns
> at 500 MHz), so it has no 300 MHz latency yet. LUT is unaffected, since it doesn't
> depend on the cutoff.

Published designs for reference (post-route, from their papers):

| Design | N | Acc (%) | LUT | Latency (ns) | DSP |
|---|---|---|---|---|---|
| JEDI-Linear (pT-sorted) | 32 / 64 / 128 | 78.0 / 80.9 / 80.9 | 45k / 71k / 98k | 63 / 61 / 82 | 0 |
| JEDI-Linear (perm.-inv.) | 64 / 128 | 81.8 / 81.6 | 164k / 296k | 78 / 138 | 0 |
| MLP-Mixer | 64 / 128 | 79.7 / 79.8 | 159k / 83k | 72 / 72 | 0 |
| Deep Sets (MLST'24) | 32 | 75.9 | 434k | 130 | 903 |
| GNN (MLST'24) | 32 | 75.8 | 2.12M | 205 | 1162 |

Full table with every row and stage counts: `docs/comparison_final.md` /
`results/comparison_final.csv`. Per-checkpoint hardware rows: `results/hw_results.json`.

![Accuracy vs LUT and attention saving](docs/figures/accuracy_vs_lut.png)
![Accuracy vs latency and the L1T envelope](docs/figures/accuracy_vs_latency.png)

## Results by sequence length (N)

PHAT-JeT's learned weights are shared across particles and patches, so the same
model runs at any N divisible by the patch size. Inputs are the N highest-pT
constituents (the dataset is pT-sorted), the same selection JEDI-Linear uses for its
N axis. **Each N was trained separately**: float pretraining, then QAT, then tracing
every Pareto checkpoint at 300 MHz. In total **681 serial designs** were traced across
N = 16, 32 and 64, plus 23 parallel-attention designs at N=64 and 31 at N=128.

### What was run at each N

| N | Float runs (best val acc) | QAT runs | Traced designs |
|---|---|---|---|
| 8 | converged float: 64.69% | `qat_n8_bm30` (EBOPs ramp ×30): 21 epochs, best 63.9%, stalled | 0 (dropped) |
| 16 | 30 ep warm start: 71.28% · **120 ep scratch: 73.30%** | `qat_n16`, `qat_n16_f2`, `qat_n16_v3` | 60 |
| 32, ps=8 | 30 ep: 78.04% · **120 ep: 79.11%** · parallel 30 ep: 77.41% | `qat_n32`, `_fast`, `_floor`, `_f2`, `_c1`, `_c2`, `_v3` | 193 |
| 32, ps=4 | **120 ep: 79.32%** | `qat_n32p4_c1`, `qat_n32p4_v3` | 34 |
| 64 | 30 ep: 80.47% · **100 ep: 81.90%** · parallel 30 ep: 80.64% | `qat_n64`, `_f2`, `_c1`, `_c2`, `_w1`, `_w2`, `_w3`, `_v3`; parallel `par_n64_a/b` (200 ep) | 394 + 23 parallel |
| 128 | **60 ep: 81.59%** · parallel 40 ep: 81.24% | `run6` (1000-ep schedule, stopped at 788), `run8`, `run9_hold`; parallel `par_n128_a` (stopped at epoch 4) | 31 at 300 MHz + 5 at 2 ns |

Run-name suffixes (taken from the training logs):
- `f2` / `floor`: `FLOOR_BITS=2`
- `c1` / `c2`: stronger / gentler EBOPs ramp (β starts at 1.6e-7 / 6e-8), 135–205 epochs
- `w1` / `w2` / `w3`: β warms up from ~0, then ramps to 4.0e-6 / 1.9e-6 / 1.3e-6 over about 300 epochs
- `v3`: short runs (2–4 epochs) started from the converged float donors
- `p4` / `ps4`: patch size 4

### Best design at each N (quantized, traced at 300 MHz)

| N | Float ceiling | Within budget (<100 ns, ≤10% LUT) | Fits one VU13P | Any size (uncompressed) |
|---|---|---|---|---|
| 8 | 64.69% | — not traced | — | — |
| 16 | 73.30% | **67.45%** · 87k LUT (5.0%) · 96.6 ns | 70.40% · 901k (52%) · 153.2 ns | 71.09% · 7.1M · 239.8 ns |
| 32 | 79.11% | **71.36%** · 98k (5.7%) · 96.6 ns | 76.95% · 1.46M (84%) · 163.2 ns | 78.34% · 14.2M · 239.8 ns |
| 32, ps=4 | 79.32% | 68.87% · 130k (7.5%) · **89.9 ns** | 71.33% · 470k (27%) · 129.9 ns | 78.60% · 11.9M · 243.1 ns |
| 64 | 81.90% | **74.58%** · 167k (9.7%) · 96.6 ns | 78.42% · 1.22M (71%) · 133.2 ns | 81.60% · 11.2M · 209.8 ns |
| 64, parallel | 80.64%* | none traced | 72.93% · 1.12M (65%) · 106.6 ns | 80.96% · 24.6M · 173.2 ns |
| 128 | 81.59% | none (≥116.5 ns) | **78.64%** · 1.19M (69%) · 126.5 ns | 81.13% · 185M · 256.4 ns |

\*30-epoch float donor; the parallel QAT run itself reached 81.0% val accuracy.
The "Any size" column shows the quantized model barely compressed: QAT holds float
accuracy, and the cost is all in compressing it down to fit.

JEDI-Linear (pT-sorted, post-route) for reference: N=16 71.9% · 32 78.0% · 64 80.9% ·
128 80.9%, at 44k–98k LUT and 54–82 ns.

### What changes with N

- **Accuracy rises with N and saturates at 64.** Float ceilings are 64.7 → 73.3 →
  79.1 → 81.9 → 81.6% for N = 8 → 128. Inside the trigger budget, each doubling from
  16 to 64 adds about 3–4 points (67.5 → 71.4 → 74.6%).
- **Latency doesn't change with N; LUTs do.** The best in-budget design at N = 16, 32
  and 64 is 29 stages (96.6 ns) in every case. Going from N=16 to N=64 costs 87k →
  167k LUTs. N=32 is the best fit for a tight LUT budget (71.4% at 5.7%); N=64 buys
  +3.2 points for 4 more percent of the device.
- **The quantization gap is largest at the tightest budget.** In-budget designs are
  5.9 / 7.8 / 7.3 points below float at N = 16 / 32 / 64. The longest schedule
  (N=128, run 6) loses 2.95 points at 69% of the device. That points to training
  schedule length, not architecture, as the limit.
- **Every N needs its own training.** Truncating a trained N=128 model to fewer
  particles collapses it (78.6 → 63.3% at N=32, 46.8% at N=16), because it relies on
  the soft-particle tail. Slicing an N=128 *quantized* checkpoint's per-element
  quantizers gave chance accuracy, so those results were discarded. A float warm start
  from N=128 works but is no better than training from scratch (N=16: 71.28% vs 71.56%).
- **The donor bug cost about 1.5 points at every N.** QAT originally picked the float
  donor by N alone, which was the first 30-epoch run, not the converged one. That
  threw away 1.29 / 1.78 / 1.51 points at N = 16 / 32 / 64, about the size of the whole
  gap to JEDI-Linear. Fixed with the `FLOAT_CK` override, which names the donor explicitly.
- **N=8 was dropped.** The converged float model reaches only 64.69%, and the one QAT
  run (EBOPs ramp ×30) stalled at about 63–64% within 21 epochs, before any design was traced.

### Patch size, width and depth (studied at N=32 and N=64)

**Attention pattern**: same model, only the attention changed. These designs are
untrained, with every quantizer fixed at 6 bits, so the rows compare only with each other:

| N | Attention | Stages | Latency | LUT | Act.×act. MACs |
|---|---|---|---|---|---|
| 32 | full self-attention | 38 | 126.5 ns | 5.89M | 32,768 |
| 32 | patch, ps=16 | 53 | 176.5 ns | 4.05M | 16,512 |
| 32 | patch, ps=8 | 51 | 169.8 ns | 3.14M | 8,704 |
| 32 | patch, ps=4 | 52 | 173.2 ns | 2.92M | 6,144 |
| 32 | patch, ps=8, parallel | 37 | 123.2 ns | 3.17M | 8,704 |
| 64 | full self-attention | 40 | 133.2 ns | 19.27M | 131,072 |
| 64 | patch, ps=16 | 53 | 176.5 ns | 8.03M | 33,280 |
| 64 | patch, ps=8 | 56 | 186.5 ns | 6.46M | 18,432 |
| 64 | patch, ps=4 | 55 | 183.2 ns | 6.24M | 16,384 |
| 64 | patch, ps=8, parallel | 39 | 129.9 ns | 6.35M | 18,432 |

- The LUT saving from patch attention grows with N: **1.88x at N=32, 2.98x at N=64.**
  Activation-by-activation multiplies fall 3.8x and 7.1x.
- In serial form, patch attention adds pipeline stages over full attention, since local
  then patch attention is a deeper chain. Wiring them in parallel recovers the latency
  at about the same LUT count.
- **Patch size 4 vs 8** (N=32): float accuracy is the same (79.32% vs 79.11%), and
  ps=4 saves only 7% of LUTs untrained. After QAT, ps=4 reached 68.87% at 89.9 ns
  in budget, vs 71.36% for ps=8. That came from a single QAT arm, against seven for ps=8.
- **Width:** `d_model` 16 → 8 costs 2.0 points (N=16) and 3.5 points (N=32) of float
  accuracy. It halves LUTs once, then saturates (8 → 4 saves only 6%). Rejected.

**Depth ablation**: pipeline stages / LUTs with each block removed (untrained, 6-bit):

| N | Full model | − GMP | − patch attn | − local attn | − FFN | Parallel attn |
|---|---|---|---|---|---|---|
| 32 | 51 st · 3.15M | 41 st · 1.82M | 36 st · 3.04M | 36 st · 1.92M | 48 st · 2.72M | 38 st · 3.17M |
| 64 | 52 st · 6.37M | 42 st · 3.74M | 36 st · 6.02M | 40 st · 3.97M | 49 st · 5.54M | 38 st · 6.38M |

GMP is about 41% of the LUTs at N=64 and 10 stages. Patch attention is the deepest
block (−15 to −16 stages), which is why the parallel wiring helps.

### Parallel attention after training

- **N=64** (`par_n64_a/b`, 200 epochs): QAT reached 81.0% val accuracy. The traced
  checkpoints (epochs 13–50) reach 78.51% at 126.5 ns, but at 3.96M LUTs (2.3x the
  device). The best one that fits is 72.93% at 65% of the device and 106.6 ns. The
  later, far more compressed checkpoints (EBOPs down to 36k–74k by epoch 200) **have
  not been traced yet**. They're the most likely route to an in-budget parallel design.
- **N=128** (`par_n128_a`): stopped at epoch 4 of 1000, at 80.4% val and still
  compressing. Not traced.

### Other things tried at each N

- **EBOPs ramp strength** (`BETA_MULT`, N=32): ×8 compressed 38.0M → 4.22M EBOPs while
  accuracy *rose* 77.46 → 78.08%. ×24 lost 7.5 points in one epoch and was rejected.
- **`FLOOR_BITS=2`** (N=32): without a floor, LUTs stalled around 1.45M as EBOPs fell.
  With the floor, the penalty prunes whole channels instead of binarizing tensors:
  3.35M → 293k LUTs (11.4x), op count down 7.7x.
- **Accuracy-recovery fine-tune** (freeze the quantizers, retrain the weights; N=32 and
  N=64, LR 3e-7 to 1e-6): gained at most 0.01 points. It didn't work.

Data behind this section is in `results/per_n/`:
- `traced_designs.json`: all 681 serial traces at 300 MHz
- `traced_designs_parallel_n64.json`
- `float_ceilings.json`
- `attn_shootout_untrained.json` and `depth_ablation_*_untrained.json`
- `n128_budget_band_traces_2ns_cutoff.json`

## What it does

1. **Port** PHAT-JeT to a hardware-synthesizable Keras 3 / HGQ2 model (`phat_jet_k3.py`).
2. **Train** it quantization-aware, scanning the accuracy-vs-EBOPs Pareto front (`train_qat.py`).
3. **Trace** each checkpoint with da4ml into a pipelined distributed-arithmetic
   netlist: LUT / FF estimates, pipeline stages, latency at 300 MHz (`extract_hw.py`).
4. **Emit Verilog** for the VU13P and check it bit-exactly against Keras with Verilator.
5. **Benchmark** against JEDI-Linear using its own da4ml settings, and make the
   rebuttal figures and tables.

The aim is to show feasibility: the model fits the CMS L1T envelope (<100 ns, II=1,
~10% of VU13P). It doesn't claim to beat JEDI-Linear. See `TASK_PROMPT.md` /
`docs/HANDOFF.md`.

## How PHAT-JeT was implemented in hardware

The public PHAT-JeT code is TF/Keras 2. It uses dynamic shapes,
`tf.scatter_nd`/`tf.gather_nd`, and a grid that changes per jet. None of that can
go through HGQ2 → da4ml. That flow needs static shapes, only ops in da4ml's trace
registry, and no dynamic indexing. The model was therefore rewritten in
`scripts/phat_jet_k3.py` as a single builder. The same code produces both the float
reference and the quantized (HGQ2) model, so trained float weights load 1:1 into
the quantized model.

### Changes to the architecture

| # | Paper model | Hardware model | Why |
|---|---|---|---|
| 1 | GMP on a per-jet dynamic grid, `scatter_nd` / `gather_nd` | **Fixed grid** within ±0.4 in η/φ. Per-bin one-hot indicator LUTs, an outer product to cells, and einsum scatter-add → depthwise conv → einsum gather-back. Same math. | Dynamic indexing can't be traced. Inputs are jet-relative, and 99.9% of constituents fall within ±0.4. |
| 2 | Uniform δ=0.1 bins | **`core7`**: 7×7 non-uniform bins, 0.08 wide in the core (\|x\|<0.2), wider outside | Fine bins where the jet core sits. In a short QAT comparison it beat the uniform 8×8 grid by **+2.0 points** at lower cost (`docs/grid_study.md`). |
| 3 | 8×8 depthwise kernel on a δ=0.05 grid | 3×3 kernel on the fixed grid | Similar physical receptive field (~0.3 vs ~0.4) |
| 4 | 150 particles, patch size 10 | **N = 16 / 32 / 64 / 128** highest-pT particles (N=8 tried), patch size 8 (also 4 at N=32) | Matches JEDI-Linear's N axis and input selection. Learned weights are shared across particles, so the architecture is the same at every N; only the per-element quantizer tensors scale with N. |
| 5 | LayerNorm | **Removed** | A runtime divide and square root don't fit a fully unrolled II=1 design. JEDI-Linear has no runtime normalization either. |
| 6 | Attention scaled by 1/√d_head; softmax output | Scale folded into W_q; the model outputs logits | Exact reparameterization; argmax doesn't need softmax on chip |
| 7 | — | Softmax exp-table input capped at 5 integer bits | HGQ sized the exp table from rare outliers (1024 entries). The cap cut lookup logic 4.1x, **−24.6% total LUTs** for +0.05% accuracy. 4 bits loses 3.1%. |
| 8 | — | Optional `parallel_attn`: local and patch attention as parallel branches | Pipeline depth becomes max(local, patch) instead of their sum: **1.44x lower latency** at N=64. This computes a different function, so the model has to be trained this way. |

### Fixes needed to make it trace and train

- **Tracer limits:** `ops.stack` → `Concatenate` + `Reshape`. `ops.mean`/`ops.sum` →
  `hgq.layers.QSum(scale=1/8)`, a power-of-two shift. Einsum indices must be lowercase.
  Custom `Layer` subclasses are invisible to the tracer, so everything is functional
  Keras ops or registered Q-layers.
- **Float donor must match the hardware model.** QAT initialized from a donor trained
  *with* LayerNorm collapsed. `float_pretrain.py` therefore trains the exact LN-free
  `core7` architecture. Each N needs its own donor (`float_n.py`): cutting an N=128
  model down to N=32 drops it from 78.6% to 63.3%.
- **QAT recipe:** quantizers and callbacks are copied from HGQ2's `jsc150` example,
  the flow JEDI-Linear used. It runs 1000 CPU epochs instead of 7000 GPU epochs.
  `FLOOR_BITS` adds a minimum on fractional bits, to stop the EBOPs penalty from
  shrinking the GMP block to about 1 bit.
- **HGQ 0.1.9 bug:** the integer-bit constraint on the softmax exp table is applied to
  the wrong variable during training. The `ClampSoftmaxExpBits` callback in
  `train_qat.py` re-applies the cap after every epoch.
- **Input preprocessing:** robust scaling per feature (median/IQR). The padding mask
  must be computed *before* scaling. Doing it after silently dropped accuracy to
  24.5%, so `load_jets()` now fails loudly if the padding fraction is off.
- **Input port width:** widened from `(1,6,8)` to `(1,7,8)`. Over the full dataset,
  58 constituents exceed ±64 and were wrapping to negative pT. The extra bit costs
  +192 LUTs.
- **Checkpoints:** `.keras` files can't be deserialized because the GMP indicators
  are closures. The extraction scripts rebuild the model in code and call
  `load_weights(skip_mismatch=False)`.
- **macOS Verilator build:** da4ml's emulator makefile is written for Linux.
  `extract_hw.compile_emulator()` patches the linker flags and `nproc`.

## Contents

**Start with the PHAT-JeT pipeline scripts.** These define the model and take it
from data to Verilog. Everything else is an experiment built on top of them.

```
scripts/
  # ── PHAT-JeT architecture + pipeline (start here) ─────────────────────────
  phat_jet_k3.py      THE MODEL: hardware-synthesizable PHAT-JeT
                      (build_phat_jet_k3: float or quantized, same code path)
  preprocess.py       data: load_jets(), robust_scale(), gmp_edges_scaled()
  grid_shootout.py    defines GRIDS (the fixed GMP bin edges, incl. core7)
  float_pretrain.py   step 1: float training at N=128
  float_n.py          step 1b: float donor at another N (16/32/64)
  train_qat.py        step 2: HGQ2 QAT -> Pareto checkpoints (build_qat_model)
  extract_hw.py       step 3: checkpoint -> LUT/FF/latency + Verilog + RTL check
  trace_n.py          step 3 (quick): LUT/FF/latency only, no Verilog
  bitexact.py         step 4: standalone RTL-vs-Keras bit-exactness check

  # ── experiments: training variants ────────────────────────────────────────
  train_hold.py, train_floor.py, train_qat_continue.py  QAT held at a fixed operating point
  recover.py .. recover4.py, lrscan.py, lrscan2.py     fine-tune for accuracy at fixed cost
  init_pretrained.py                                   load the original repo's float weights

  # ── experiments: hardware studies / benchmarks ────────────────────────────
  nsweep.py, watch_trace.py, trace_budget_band.py      trace many checkpoints
  attn_shootout.py, depth_ablate.py, attrib_n.py, sweep_latency.py   where LUTs/stages go
  qdiag.py                                             where quantization loses accuracy
  bench_headtohead.py, run_bench_phat.py               PHAT-JeT on JEDI-Linear's da4ml settings
  build_jedi.py, bench_jedi_family.py, jedi_scaled.py  JEDI-Linear reference points
  dominance.py, make_rebuttal_figs.py                  iso-accuracy comparison, figures
docs/       PROJECT_NOTES.md (decision log), HANDOFF.md (lab notebook), grid study, rebuttal text
results/    hw_results.json, comparison_final.csv, component cost breakdowns
```

Only code, docs and the small result files are tracked. Checkpoints, traces,
Verilog projects and the dataset are local-only (see `.gitignore`).

## Setup

```bash
conda create -n fpga python=3.11 numpy scipy h5py pandas scikit-learn matplotlib
conda activate fpga
pip install "keras>=3.10" jax hgq2 da4ml hls4ml tensorflow
conda install -c conda-forge verilator   # for bit-exact RTL emulation
```

## Data

[hls4ml LHC jet dataset, 150 particles](https://zenodo.org/records/3602260)
(`hls4ml_LHCjet_150p_train.tar.gz` + `_val.tar.gz`). Preprocess to the
JEDI-Linear-matched config — 128 highest-pT constituents × (pT, ηrel, φrel),
zero-padded, 5-class one-hot — with the snippet in `docs/PROJECT_NOTES.md`
(§ Dataset); output is `data/jets_128x3.npz` (620k train / 260k val).

## How to use

Run everything from the repo root in the `fpga` env. Outputs (checkpoints, logs,
Verilog projects) are written to the repo root.

```bash
export KERAS_BACKEND=jax GRID_NAME=core7
```

**1. Train the float model** (the QAT model starts from these weights)

```bash
python scripts/float_pretrain.py 60                    # N=128 -> float_phatjet_core7.keras
N_PARTICLES=64 WARM=128 python scripts/float_n.py 30   # N=64  -> float_phatjet_core7_n64.keras
```

**2. Quantization-aware training.** A single run sweeps the EBOPs penalty upward,
keeping every non-dominated (accuracy, EBOPs) epoch.

```bash
N_PARTICLES=64 RUN_TAG=qat_n64 FLOOR_BITS=2 QAT_EPOCHS=1000 python scripts/train_qat.py
# -> pareto_qat_n64/epoch=..-val_acc=..-ebops=...keras, qat_n64_log.csv
```

| Env var | Default | Meaning |
|---|---|---|
| `N_PARTICLES` / `PATCH_SIZE` | 128 / 8 | constituents per jet / particles per patch |
| `D_MODEL` / `NUM_HEADS` | 16 / 4 | width / attention heads |
| `PARALLEL_ATTN` | 0 | 1 = parallel local/patch attention (lower latency; needs a parallel float donor) |
| `FLOAT_CK` | `float_phatjet_core7[_nN].keras` | float donor to start from |
| `QAT_RESUME` | — | continue from an already-quantized checkpoint |
| `FLOOR_BITS` | 0 | minimum fractional bits per quantizer |
| `QAT_EPOCHS`, `BETA0`, `BETA_MULT` | 1000, 2e-8, 1 | schedule length and EBOPs-penalty ramp |
| `RUN_TAG` | — | **must be unique per run** (CSV logs append) |

**3. Hardware numbers + Verilog.** Always pass the checkpoint explicitly, and
re-evaluate accuracy rather than trusting the filename: several filenames were
more than 1 point optimistic.

```bash
N_PARTICLES=64 python scripts/extract_hw.py pareto_qat_n64/<ckpt>.keras
#   -> Verilog in verilog_<ckpt>/, row in results/hw_results.json
#   N_PARTICLES must match the checkpoint (default 128). SKIP_EMU=1 skips the slow Verilator build.
python scripts/trace_n.py <ckpt>.keras 64    # quick estimate only (args: ckpt N [patch_size])
```

Latency = pipeline stages × 3.33 ns (300 MHz), traced with a latency cutoff of 4.0.
Don't requote a design at another clock; retrace it instead.

**4. Bit-exactness check**

```bash
BE_CKPT=<ckpt>.keras BE_PRJ=bitexact_prj python scripts/bitexact.py
```

**Use the model directly**

```python
import sys; sys.path.insert(0, "scripts")
from preprocess import gmp_edges_scaled
from phat_jet_k3 import build_phat_jet_k3
from train_qat import build_qat_model

edges = gmp_edges_scaled("core7")                  # fixed GMP grid, robust-scaled units
model  = build_phat_jet_k3(num_particles=64, gmp_edges=edges)   # float reference
qmodel = build_qat_model(gmp_edges=edges, num_particles=64)      # HGQ2, same layer names
qmodel.load_weights("pareto_qat_n64/<ckpt>.keras", skip_mismatch=False)
```

Resource numbers are **da4ml estimates** (no Vivado run). JEDI-Linear's numbers
are post-place-and-route; every comparison table labels which is which.

## Repository notes

The code was cleaned up before publishing:

- **Shared data loading.** About 20 scripts each had their own copy of the dataset
  loading and scaling code. They now call `preprocess.load_jets()`. Its output was
  checked to be bit-identical to the old code.
- **Shared GMP bin edges.** Scripts that recomputed the edges by hand now use
  `preprocess.gmp_edges_scaled()`.
- **Bug fix in `bench_headtohead.py`.** With `emulate=True`, the script never built
  its Verilog model. The error was silently caught, so the RTL check never ran there.
- **Removed a hard-coded absolute path** in `init_pretrained.py`.

`docs/HANDOFF.md` is the full lab notebook: pitfalls hit along the way, numbers that
were later superseded, and how to restart the runs that were stopped.
