# PHAT-JeT FPGA Hardware Evidence

Hardware implementation flow for **PHAT-JeT** (Patch Hierarchical Attention
Transformer for jet tagging, NeurIPS 2026 submission 28409): HGQ2
quantization-aware training → da4ml distributed-arithmetic conversion →
Verilog for AMD VU13P, producing latency / LUT / FF / DSP / BRAM numbers
comparable to JEDI-Linear ([arXiv:2508.15468](https://arxiv.org/abs/2508.15468)).

Produced for the NeurIPS rebuttal: reviewers asked for FPGA resource evidence
against JEDI-Linear's post-place-and-route results.

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

## Comparison target (JEDI-Linear, their Table II, pT-sorted, VU13P post-P&R)

| Model | Particles | Feats | Acc (%) | Latency (ns) | DSP | LUT (k) | FF (k) | BRAM | II |
|---|---|---|---|---|---|---|---|---|---|
| JEDI-linear | 128 | 3 | 80.9 | 82 | 0 | 98 | 48 | 0 | 1 |
