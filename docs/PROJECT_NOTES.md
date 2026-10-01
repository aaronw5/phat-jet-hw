# PHAT-JeT NeurIPS Rebuttal: Hardware Evidence Project

**Audience:** Aaron (author) + any agent (Claude, Codex, etc.) picking up this work.
**Last updated:** 2026-07-24 (session start)

## Goal

The NeurIPS 2026 meta-review's decisive criticism of PHAT-JeT (submission 28409,
"Patch Hierarchical Attention Transformers for Efficient Particle Jet Tagging") is the
**absence of hardware evidence**: the paper motivates LHC trigger deployment but reports
no FPGA latency, LUT/FF/DSP/BRAM, or initiation-interval numbers. Reviewers explicitly
asked for a comparison against JEDI-Linear (arXiv:2508.15468), which reports
post-place-and-route numbers.

**We must produce, within ~5 days:**
1. An HGQ2 quantization-aware-trained (QAT) PHAT-JeT at the JEDI-Linear-matched config
   (**128 particles × 3 features** on the hls4ml jet dataset), accuracy ≥ ~81% to beat
   JEDI-Linear's 80.9% at that config.
2. Generated hardware (Verilog via da4ml, or HLS via hls4ml) with **latency (ns), II,
   LUT, FF, DSP, BRAM** numbers. No Vivado access → we report **pre-P&R estimates**
   (da4ml LUT estimate + yosys synth_xilinx independent count) and say so honestly.
3. A comparison table + Pareto figure (accuracy vs LUT) overlaying our points on
   JEDI-Linear's published rows.
4. Drafted rebuttal paragraphs.

## Comparison target (from JEDI-Linear paper, Table II, VU13P, ~300 MHz)

| Config | Acc | Latency | LUT | FF | DSP | BRAM | II |
|---|---|---|---|---|---|---|---|
| JEDI-Linear 128p×3f (pT-sorted) | 80.9% | 82 ns | ~98k | ~48k | 0 | 0 | 1 |
| JEDI-Linear 128p×3f (perm-inv) | 81.6% | 138 ns | ~296k | — | 0 | 0 | 1 |

Their flow: HGQ (per-weight bitwidth QAT, EBOPs regularizer swept in one run for the
accuracy-vs-resource Pareto) → da4ml (multiplier-free distributed arithmetic, direct
Verilog, no DSPs) → Verilator bit-exactness check → Vivado P&R.
We replicate everything except the final Vivado step.

## Decisions made (user-confirmed)

- **Run everything LOCALLY** on the M2 Max (64 GB, 12 CPU). Nautilus k8s was explored
  but abandoned: OIDC token expired + sandbox proxy drops TLS to the API server IP.
  Do NOT retry cluster auth. (Kubeconfig copy: `/Users/anrunw/nautilus-config/config`,
  example job specs: `/Users/anrunw/Documents/kubernetes/{jobs,pods}/`.)
- **One matched config**: 128 particles × 3 features (pT-sorted, like their headline row).
- **Quantization from scratch**: port public PHAT-JeT (TF/Keras 2) → Keras 3 → HGQ2 QAT.
- **Synthesize GMP too** (user wants the full model on hardware, incl. the geometric
  message-passing scatter→depthwise-conv→gather on the (η,φ) grid).
- **No Vivado** → da4ml/yosys estimates, honestly labeled.

## Resources

| Resource | Location / URL |
|---|---|
| PHAT-JeT paper (submitted PDF) | artifact `a20e60ea-2b2a-490d-bd10-37f5590d9aee` (28409.pdf) |
| Reviews + meta-review | artifact `a9da0822-2408-4cff-81ee-78b0c4510253` |
| PHAT-JeT code (public) | https://github.com/aaronw5/PHAT-JeT → cloned at `./phat-jet/` |
| JEDI-Linear paper | arXiv:2508.15468 → `./jedi_linear.pdf`, key pages in `./jedi_sections.txt` |
| PHAT-JeT paper key pages | `./sections.txt` |
| hls4ml (synthesis) | https://github.com/fastmachinelearning/hls4ml |
| HGQ2 (QAT) | https://github.com/calad0i/HGQ2 ; examples: https://github.com/calad0i/HGQ2-examples |
| da4ml (Verilog gen) | https://github.com/calad0i/da4ml |
| hls4ml jet dataset | OpenML "hls4ml_lhc_jets_hlf" is the 16-HLF version; we need the
  **constituent-level** 150-particle dataset (Zenodo record 3602254 / same as used in
  JEDI-Linear & MLPM papers) |
| Conda env | `fpga` (py3.11: numpy/scipy/matplotlib/sklearn/h5py/pandas; keras3+jax,
  hgq2, da4ml, hls4ml added via pip) |

## Model facts (from paper + repo)

- PHAT-JeT @ trigger scale: ~6.7k params, ~768K FLOPs variant in README table;
  patch size P=10, GMP on (η,φ) grid (δ≈0.1–0.2, ~16×16 cells), depthwise conv,
  exact intra-patch attention + pooled patch-token global attention.
- Public repo: `phat-jet/models/PHAT_JeT.py` (TF/Keras), training scripts under
  `phat-jet/scripts/`. Repo also has its own JEDI_Linear.py baseline reimplementation.
- Paper accuracy @ 150p×3f: 81.80% (hls4ml dataset). Expect slightly different at 128p.

## Feasibility spike findings (2026-07-24)

Toolchain versions in `fpga` env: python 3.11.15, keras 3.15.0 (JAX backend),
hgq 0.1.9 (HGQ2), da4ml 0.6.0, hls4ml 1.3.0.

1. **Quantized attention works end-to-end**: `hgq.layers.QMultiHeadAttention`
   (P=10 tokens, d=12, 4 heads) + QDense traces through `da4ml.converter.trace_model`
   (HGQ2 registers itself as the `keras` DAIS-tracer plugin). With placeholder 12-bit
   inputs the toy traced to cost≈427k LUT / 88 latency stages — meaningless numbers
   before QAT bit-shrinking, but proves the ATTENTION PATH IS TRACEABLE. `comb_trace`
   gives (cost, latency) directly; this is the da4ml LUT estimate we'll report.
2. **GMP must be re-expressed**: dynamic scatter/gather is not in the da4ml op
   registry (109 traceable ops: QEinsum, QConv2D, QDense, QSoftmax, LUTs, arithmetic,
   reshape/transpose/concatenate — no gather, no dynamic indexing). Reformulation:
   one-hot bin indicators via `QUnaryFunctionLUT` per grid bin (η and φ separately),
   outer-product to cell indicator [N, Gη, Gφ], scatter-add = `QEinsum("bnep,bnc->bepc")`,
   depthwise `QConv2D`, gather-back = `QEinsum("bnep,bepc->bnc")`. Forward pass works.
   Trace hit `keras.ops.stack` (unsupported) — fix: replace `ops.stack` with
   `Concatenate` + `Reshape` (both registered). Mathematically identical to the paper's
   GMP with min-shift replaced by fixed grid bounds (fine: trigger inputs are
   detector-frame Δη/Δφ in a bounded cone).
3. **hls4ml dataset**: Zenodo 3602260 (150p) = 2.7 GB train + 1.1 GB val compressed;
   ~880k jets, 5 classes, 150×16 per jet. We slice to 128×3 (pT-sorted) per JEDI-Linear.

## Dataset (prepared 2026-07-24)

Zenodo record 3602260 "HLS4ML LHC Jet dataset (150 particles)": 2.7 GB train +
1.1 GB val compressed (~4 GB extracted, 88 HDF5 files). Each jet: 150 constituents
× 16 features + 59 jet-level features + calorimeter images (unused).

Our slice (matches JEDI-Linear 128p×3f): features `j1_pt, j1_etarel, j1_phirel`
(indices 5, 8, 11 of jetConstituentList), truncated to the 128 highest-pT
constituents (already stored pT-sorted; verified monotonic), zero-padded.
Labels = one-hot [g, q, W, Z, t] (jets columns 53:58).

- train: 620,000 jets → x (620000, 128, 3); val: 260,000 jets. Classes balanced (~20% each).
- Mean nonzero constituents per jet: ~49 (so 128 slots are mostly padding — same as JEDI-Linear's setup).
- Checkpoint artifact: `jets_128x3.npz` (497 MB, artifact 109e5c34).
- NOTE: raw arrays, no standardization. The repo's training pipeline uses
  robust-scaled inputs (`x_*_robust_*const_ptetaphi.npy`, scaler not in repo);
  for the hardware model we standardize with fixed per-feature shift/scale folded
  into the input quantizer (hardware-friendly), computed on train set.

## Keras-3 port (`phat_jet_k3.py`, 2026-07-24)

One builder emits both the float reference and the HGQ2 QAT model (identical
structure → weight transfer is 1:1). Full docstring in the file. Documented
deviations from the paper model (all needed for static/traceable hardware form):
fixed GMP grid [-0.4,0.4]² 8×8 (δ=0.1; 99.9% coverage measured), 3×3 depthwise
kernel, P=8 (16 patches × 8, no padding), no LayerNorm in hardware variant,
1/√d_head folded into W_q, logits output (softmax off-chip; argmax is monotone).

Trace fixes discovered:
- einsum equations must use only lowercase indices (keras `ops.einsum` rejects caps).
- `keras.ops.stack`, `ops.mean`, raw `ops.sum` are NOT traceable; use
  Concatenate+Reshape, and `hgq.layers.QSum(axes=, scale=)` for the mean tokenizer
  (scale=1/8 → power-of-two shift, free in fixed point).
- Custom Layer subclasses are invisible to the tracer — everything must be
  functional Keras ops or registered (Q)layers.

**Float model: 5,429 params. Full quantized model TRACES END-TO-END through
da4ml** (pre-QAT placeholder cost ~16.1M LUT / 279 stages — meaningless until QAT
shrinks bitwidths; listed only as proof of traceability).

## Hardware-flow-first pivot (2026-07-24, user request)

User: "dont need to train this right now ... start with the hgq and implementations
of hls4ml. i am more interested in latency and luts and bram than reproducing rn."
Float training interrupted at epoch 4 (val_acc 0.72 and climbing — recipe works).

Flow validated so far:
- Fixed-bitwidth (no QAT) quantized builds trace end-to-end:
  w6/a6 → 28.0M LUT-est, w4/a4 → 15.9M LUT-est, comb latency ~230-280 (da4ml ns est).
  THESE NUMBERS ARE MEANINGLESS as resource claims — random dense weights mean no
  CMVM sparsity. They prove only that the trace works. JEDI-Linear's 98k LUT comes
  from EBOPs-driven QAT pruning most weight bits to zero. A short QAT run is
  REQUIRED for reportable numbers → compressed QAT next (background) while the
  Verilog path is validated on the w4/a4 trace.
- Key structural insight for the rebuttal: JEDI-Linear has ZERO data×data
  multiplies (all MACs are constant-weight CMVM → DA adder graphs). Attention
  fundamentally has data×data products (Q·K, weights·V) + softmax LUTs; those
  cannot be folded into DA. PHAT-JeT's hardware cost will be dominated by these.
  Honest framing: report the measured gap + the accuracy gain.
- JEDI-Linear reference rows (their Tables I/II, VU13P post-P&R):
  * perm-inv 128p×3f: 81.6%, 138 ns, 296k LUT, 163k FF, 0 DSP, 0 BRAM, II=1, 203 MHz
  * pT-sorted 128p×3f: 80.9%, 82 ns, 98k LUT, 48k FF, 0 DSP, 0 BRAM, II=1, 258 MHz
  (we compare against the pT-sorted row; our inputs are pT-sorted too)
- da4ml pipeline path: comb_trace → to_pipeline(comb, latency_cutoff≈clock_ns)
  → VerilogModel(pipe, part_name='xcvu13p-flga2577-2-e', clock_period=3.33).write()

## QAT + extraction pipeline (2026-07-24 late)

- QAT running detached (`train_qat.py` → `qat_log2.txt`, pid 29261): 80 epochs,
  300k-jet subset, jsc150 recipe, from scratch (float ckpt was never saved — training
  was interrupted pre-save). ~160 s/epoch ≈ 3.5 h total. Pareto checkpoints land in
  `pareto/` (only non-dominated (val_acc, ebops) epochs are kept).
- `extract_hw.py`: checkpoint → val acc (full 260k val set) → da4ml trace →
  to_pipeline(cutoff=4.0) → VerilogModel(VU13P, 3.33 ns) → bit-exact emulation check
  → row in `hw_results.json`. Latency = stages × 3.33 ns @ 300 MHz.
  GOTCHAS (cost a few iterations):
  * `.keras` deserialization fails on GMP lambda activations → rebuild architecture
    via `train_qat.build_qat_model()` + `load_weights()` instead of `load_model()`.
  * `pareto/` files get renamed/deleted by the ParetoFront callback while training
    runs → snapshot (`cp`) to `ckpt_snapshots/` before extracting.
  * `vm.compile()` needed before `vm.predict()` (C emulation lib).
  * Sandbox forbids `PYTHONPATH=... cmd` prefixes → script does sys.path.insert itself.
- Early-checkpoint dry run (epoch 4, val_acc 0.625): trace 29.8M LUT-est, 66 stages
  → 220 ns @ 300 MHz. Still meaningless resource-wise (beta ramp barely started —
  EBOPs is what shrinks this by orders of magnitude in the jsc150 recipe).

## Pretrained-init run (run 3, 2026-07-24, user request)

User supplied trained float weights (`~/Documents/phat-jet-hw/model.weights.h5`,
19,220 params) from run config: hls4ml 150p pt-sorted, enc_dims 12/24/32 (small =
d16?), patch_size=150 (FULL attention, no patching!), heads=4, agg=mean,
patch_tokenizer=mean, cpe_k=8 (8x8 conv kernel), grid_size=0.2, GELU FFN,
robust-scaled inputs. "you should train this instead of he other one running right now"

Transfer (`init_pretrained.py`): 16/16 layers mapped — embed, local+patch attention
q/k/v/o (scale 0.5 folded into wq), ffn, gmp dwconv (center 3x3 crop of their 8x8),
gmp pointwise, head1/head_out. Non-transferable: LayerNorms (we have none),
GELU->ReLU, 150p full-attn -> 128p/16-patch context, input scaling.

Zero-QAT sanity: transferred weights give only ~20-30% val acc (vs 81.8% paper) —
expected: missing LNs + activation + patch-context changes are big distribution
shifts. The value is the INIT, not zero-shot accuracy; attention/embedding features
are task-aligned, so QAT converges much faster (this is also jsc150's own flow:
they init QAT from a float model).

Changes for fidelity to the checkpoint's training:
- agg switched max -> mean (QSum scale=1/128, power-of-two, HW-free)
- input scaling switched pT/64 -> robust median/IQR (stats: data/robust_stats.json,
  computed on train set, pad rows re-zeroed). pt iqr 17.65, eta/phi iqr ~0.108.
- GMP bin bounds rescaled to robust units: ±0.4 raw / 0.108 ≈ ±3.7.

Runs so far: run1 (80ep, subset, lr2e-3/b4096) peaked 69.4% @ epoch 17 before kill;
run2 (200ep exact-match, from scratch) killed at epoch ~2 for pretrained-init;
run3 = 200ep exact-match + pretrained init, log qat_log4.txt, pareto/ dir
(run1/run2 checkpoints kept in pareto_run1_80ep/, pareto_run2_scratch/).

## Run 3 collapse + run 4 (2026-07-24)

Run 3 (pretrained init) COLLAPSED by epoch 3: the float ckpt was trained WITH
LayerNorms; our hardware model has none, so trained-scale weights produce huge
activation ranges. HGQ2 quantizers track ranges -> EBOPs started at 5.9e9
(vs ~4e7 from scratch, 150x) -> beta*EBOPs dominated the loss (571 vs ~1.6 CE)
-> network crushed to random (20%) and stuck. Killed at epoch ~7. Log:
qat_log_run3_pretrained_collapsed.txt. Lesson: do NOT init an LN-free HGQ2 model
from LN-trained float weights without folding LN statistics into the weights first
(idea shelved: calibration-batch scale folding — user chose plain from-scratch).

Run 4 (CURRENT, user-approved): from scratch, exact jsc150 match, 200 epochs,
full train set, robust-scaled inputs, SKIP_PRETRAINED_INIT flag file set.
Log: qat_log5.txt, pid 34039, ~140 s/epoch -> ~8 h. Pareto ckpts -> pareto/.
Old runs preserved: pareto_run1_80ep/ (best 69.4%), pareto_run2_scratch/.

## Session end state (2026-07-24, user cancelled all running work)

All processes stopped. Nothing training. Where things stand:

DONE:
- Run 4 complete: 200 epochs, exact jsc150 recipe, from scratch. 22-point Pareto
  frontier in pareto/ (75.0% @ 13.4M EBOPs -> 61.1% @ 67k EBOPs), final model
  qat_final.keras, history qat_hist.json.
- Hardware extracted for 5 frontier points (hw_results.json, artifact 661e844e;
  Verilog in verilog_run4_*/). Convention: latency = stages/300MHz, VU13P part,
  II=1, DSP=0, BRAM=0. Best trade-off rows:
  * 74.0% | 116.5 ns | 1.28M LUT-est   (epoch 53)
  * 68.1% |  73.3 ns | 219k LUT-est    (epoch 121)
  * 61.1% |  56.6 ns | 55.7k LUT-est   (epoch 191)
  vs JEDI-Linear pT-sorted 128p: 80.9% | 82 ns | 98k LUT (post-P&R);
  perm-inv 128p: 81.6% | 138 ns | 296k LUT. NOTE our numbers are da4ml
  ESTIMATES, not P&R.
- Verilator emulation: module-name sanitization fixed (extract_hw.py now strips
  non-identifier chars); RTL compiles; final .so link step still fails
  ("Compilation failed!!" after g++ -shared with -Wl,--no-undefined — likely
  that flag on macOS clang). Latency/LUT numbers unaffected.
- Continuation run (1000ep, beta const 3e-6) killed at epoch 4 per user; its
  epoch=c* checkpoints in pareto/ are low-value (LR-restart dip) — EXCLUDE from
  frontier (distinguish by "c" prefix).

OPEN QUESTIONS / NEXT STEPS (if resumed):
- 64-particle variant: free data slice (x[:, :64, :]); JEDI-Linear 64p row =
  80.9% | 61 ns | 71k LUT. Faster to train, cheaper attention.
- Accuracy gap honest framing: our 200-epoch CPU budget vs their 7000-epoch GPU
  run; float ceiling 81.8% (paper). GPU run with these exact scripts would close
  most of the gap.
- JEDI-Linear has NO attention heads (linear interaction net: proj -> avg-pool
  context -> broadcast -> dense -> avg-pool -> MLP). User's "2 heads" memory was
  another model (MLPM/JEDI-net variant).
- Comparison table + rebuttal text not yet drafted; all raw numbers exist here
  and in hw_results.json.

## Status log

- [x] Kubernetes/Nautilus access — ABANDONED (user: run locally)
- [x] kubectl + kubelogin downloaded to `./bin/` (unused now)
- [ ] `fpga` conda env creation (in progress)
- [ ] Feasibility spike: HGQ2 attention + GMP through da4ml
- [ ] Dataset prep (128×3, JEDI-Linear preprocessing)
- [ ] Keras 3 port + float training
- [ ] HGQ2 QAT + EBOPs Pareto scan
- [ ] Hardware generation + resource extraction
- [ ] Comparison table + figures
- [ ] Rebuttal text
