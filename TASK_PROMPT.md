# Task prompt: PHAT-JeT FPGA resource study for the NeurIPS rebuttal

Hand this to a competent ML-hardware person with no prior context. Everything needed
to reproduce or extend the study is here.

## 1. Objective

Reviewers asked whether PHAT-JeT (patch hierarchical attention transformer for jet
tagging) is deployable in the CMS Level-1 Trigger. Produce FPGA latency and resource
numbers showing it fits the L1T envelope. The claim is FEASIBILITY, i.e. "this fits",
NOT superiority over any baseline. Do not claim a win over JEDI-linear (arXiv
2508.15468) on accuracy, latency or resources. JEDI-linear is not a baseline in the
paper's Table 2.

## 2. The envelope to hit

- Latency below 100 ns end to end.
- Initiation interval of 1 cycle, i.e. a new jet accepted every clock.
- Target device AMD Xilinx VU13P, 1,728,000 LUTs, which CMS specifies for Correlator
  Layer 2. CTL2 shares the device with jet clustering, so aim for under 10% of LUTs.
- 300 MHz, i.e. 3.33 ns per stage. Latency_ns = pipeline_stages * clock_period_ns.

CRITICAL: 40 MHz is the LHC bunch crossing rate, NOT a firmware clock. Firmware runs
at a multiple of it. Do not quote latency at 40 MHz. Also do not requote a design at a
faster clock by arithmetic, because pipeline depth is a function of the pipelining
cutoff applied at trace time, so a design traced for one frequency is invalid at
another. Retrace instead.

## 3. Toolchain

- Model in Keras 3 with the JAX backend.
- Quantization-aware training in HGQ2 (github.com/calad0i/HGQ2, examples at
  HGQ2-examples). Layers used are QDense, QEinsum for attention products, QSum for
  the exact mean tokenizer, QSoftmax and QUnaryFunctionLUT for LUT-approximated
  nonlinearities, QConv2D.
- Logic synthesis estimates via da4ml, which compiles quantized Keras to Verilog as
  a distributed-arithmetic shift-and-add network. NOT Vivado, since no vendor
  licenses are available. da4ml gives LUT and pipeline-stage estimates only.
- Bit-exactness verified with Verilator against the Keras model.
- Dataset is the hls4ml jet tagging dataset in JEDI-linear's 128 particles by 3
  features configuration (pt, eta, phi), 5 classes (g, q, W, Z, t).

## 4. Synthesis configuration, must match JEDI-linear exactly for comparability

    HWConfig(1, -1, -1)
    hard_dc = 2
    pipelining every 2 adders
    latency_cutoff set from the target clock period

## 5. What was already done and what the numbers are

Three operating points clear both constraints at 300 MHz, all with 0 DSP, 0 BRAM,
II=1, and all at 96.6 ns because pipeline depth follows network depth rather than
particle count:

    N=16   67.4%   87,206 LUT   5.0% of VU13P
    N=32   71.4%   98,069 LUT   5.7%
    N=64   74.6%  167,304 LUT   9.7%

LUT budget only, exceeding 100 ns: 78.60% at 902,337 LUT (52.2%), from
pareto_run9_hold epoch 27. Also 78.64% at 1,188,245 LUT, verified bit-exact.

N=128 serial clears NO point at 300 MHz, needing at least 76 stages = 253 ns. Parallel
attention wiring is the only lever, since depth becomes max(local, patch) instead of
sum. A parallel N=128 QAT run is in progress off a float donor at 80.5%.

Calibration of the estimator: tracing JEDI-linear's own released N=128 f3 model
through this identical flow gives 99,690 LUT against their published post-route
97,822, so the flow reads 1.9% high. Report this. It makes every margin conservative.

Float ceilings per N, needed for the training-budget argument:
    n16 73.30, n32 79.11, n64 81.90, n128 81.59, n128-parallel 80.5 and climbing.
Quantization gap is 6 to 8 points on short runs (161 to 300 epochs on CPU) versus
2.95 points on the one long run. JEDI-linear's released configs specify 7000 epochs
and their checkpoints sit at epochs 1294 to 4472. This is the evidence that the
accuracy deficit is training schedule, not architecture.

Architecture-versus-quantization evidence, from tracing untrained models at a fixed
6-bit quantizer scope so rows are mutually comparable:
    full self-attention N=64   19,266,018 LUT   40 stages   131,072 dynamic MACs
    patch hierarchy ps8 N=64    6,456,512 LUT   56 stages    18,432 dynamic MACs
    same, parallel branches     6,354,246 LUT   39 stages
So the hierarchy saves 2.98x in LUT while total MACs fall only 1.47x, because
two-dynamic-operand products fall 7.11x. Parallel wiring cuts depth 1.44x for 1.6%
less logic. Neither quantity is reachable by quantization, which only scales the width
of operations the architecture already committed to. This justifies choosing the
architecture first.

Component cost: geometric message passing is 41% of total LUT at N=64.

## 6. Implementation notes that cost real time to discover

- Geometric message passing must be static. Dynamic gathers have no fixed-latency
  form. Use threshold comparators for eta and phi binning, an outer product for a
  one-hot cell assignment, then express scatter, 3x3 depthwise mixing and gather as
  contractions against that one-hot tensor.
- Attention's QK and AV products have two dynamic operands and must synthesize as
  real multipliers, unlike weight multiplies which fold into constant shift-and-add.
- patch_size == N together with use_patch_attn=False is exactly full self-attention.
  Use this to get an apples-to-apples full-attention reference in the same flow.
- Parallel float weights do NOT transfer functionally across N even when shapes match.
  A warm start across N evaluates at chance and behaves as a partial reinitialization.
- Checkpoint filename accuracies are unreliable and can be over a point optimistic.
  Always re-evaluate a checkpoint before quoting it.
- Set a minimum fractional-bit floor during QAT. Without it the EBOPs penalty drives
  quantizers to near zero bits, since cost per multiply is nearly free there, and the
  model collapses.
- When transferring float weights into a quantized model, give weight quantizers
  enough integer bits, since overflow_mode='wrap' turns 1.796 into -0.2 and collapses
  to chance instantly.

## 7. Deliverables

1. A comparison table with accuracy, latency, LUT, FF, DSP, BRAM and II, including
   pipeline stages so any clock can be requoted correctly.
2. Figures showing accuracy against LUT with the device budget marked, and accuracy
   against latency with the 100 ns bound marked.
3. Rebuttal prose. Constraints on the writing: no em dashes, no colons in prose,
   succinct enough that a reviewer will read it, and free of machine-writing tells
   (no tricolons, no "not X but Y" antithesis, no enumerated scaffolding, no inflated
   closers). Vary sentence length. State limitations plainly.
4. Every resource figure labeled as a da4ml estimate, never as post-place-and-route.

## 8. Framing rules for the prose

State that the accuracies are not competitive and say why, since a resource budget
reported without an accuracy is unfalsifiable, as an arbitrarily bad model fits any
envelope. Report the calibration overestimate. Make no Pareto or efficiency claim.
The single positive claim available is that an attention model with the geometric
block intact fits the CMS L1T envelope at II=1 with no DSPs and no BRAMs.
