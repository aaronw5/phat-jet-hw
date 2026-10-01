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
- Using more of the device raises accuracy: **78.6% at 902k LUTs (52%)**.
  Latency at that point (236 ns) is above the 100 ns trigger budget.
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
| N=128, best accuracy per LUT | 78.60 | 902,337 | 52.2 | 71 | 236.4 | LUTs only |
| N=128, bit-exact verified | 78.64 | 1,188,245 | 68.8 | 76 | 253.1 | LUTs only |

All rows: 0 DSP, 0 BRAM, II=1. At N=128 no traced design is under 100 ns: the
serial model needs at least 71 stages. Running the attention branches in parallel
cut latency 1.44x at N=64 (186.5 → 129.9 ns) for 1.6% fewer LUTs. An N=128 parallel
QAT run was started but stopped at epoch 4, before it converged, and was never traced.

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

## Contents

```
scripts/
  # core pipeline
  phat_jet_k3.py      Keras-3 port of PHAT-JeT (float + HGQ2 builder, one code path,
                      1:1 weight transfer; fully da4ml-traceable)
  preprocess.py       single source of truth for data: load_jets() (load + robust
                      scale + padding sanity check), gmp_edges_scaled()
  train_qat.py        HGQ2 QAT with EBOPs Pareto scan (jsc150 hyperparameters)
  extract_hw.py       checkpoint -> val acc -> da4ml trace -> pipeline -> Verilog
                      -> bit-exact Verilator check -> results/hw_results.json
  bitexact.py         standalone RTL-vs-Keras bit-exactness check

  # training variants
  float_pretrain.py, float_n.py, init_pretrained.py   float pretrain / warm starts
  train_hold.py, train_floor.py, train_qat_continue.py  QAT at a fixed operating point
  recover.py .. recover4.py, lrscan.py, lrscan2.py   frozen-cost accuracy recovery

  # hardware studies / benchmarks
  nsweep.py, trace_n.py, watch_trace.py, trace_budget_band.py   trace checkpoints across N
  attn_shootout.py, depth_ablate.py, attrib_n.py, sweep_latency.py   where the LUTs/stages go
  grid_shootout.py, qdiag.py                           GMP grid choice, quantization loss
  bench_headtohead.py, run_bench_phat.py               PHAT-JeT on JEDI-Linear's exact settings
  build_jedi.py, bench_jedi_family.py, jedi_scaled.py  JEDI-Linear reference points
  dominance.py, make_rebuttal_figs.py                  iso-accuracy comparison, figures
docs/                 decision log (PROJECT_NOTES.md), handoff, debug findings, rebuttal text
results/              hw_results.json and component cost breakdowns
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

## Run

```bash
python scripts/train_qat.py            # QAT; Pareto checkpoints -> pareto/
python scripts/extract_hw.py pareto/<ckpt>.keras   # hardware numbers + Verilog
```

Latency convention: pipeline stages × 3.33 ns (300 MHz, JEDI-Linear's target
clock). Resource numbers are **da4ml estimates** (no Vivado run); JEDI-Linear
reports post-P&R, which is typically *lower* than pre-synthesis estimates —
stated explicitly wherever compared.

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
