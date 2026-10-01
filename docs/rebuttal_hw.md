# Rebuttal paragraphs — FPGA hardware evidence

Drop-in text. Numbers are measured; every one is reproducible from
`docs/CONFIG.md`. **Read the status note at the end before using §1's numbers** —
the QAT run they come from is still in flight.

---

## §1 — Hardware evidence (response to "no FPGA numbers")

We have synthesized the model to RTL and measured it. Using distributed-arithmetic
lowering (da4ml) targeting the same device as JEDI-Linear (Xilinx VU13P,
`xcvu13p-flga2577-2-e`) at a 300 MHz target, with identical inputs (128
constituents × 3 features, pT-sorted), we obtain a full accuracy–resource frontier
rather than a single operating point, because our EBOPs-penalized QAT traces the
frontier in one run. Representative points: 78.6% at 1.19M LUT-est / 126.5 ns, and
74.7% at 0.65M LUT-est / 116.5 ns, all fully unrolled at initiation interval 1 with
zero DSPs and zero BRAMs — da4ml lowers every multiply into LUT-based adder graphs,
so logic is the only contended resource. The generated RTL is verified **bit-exact**
against the Keras model (max absolute error 0.0, argmax agreement 1.000 over 512
validation jets), so these are measurements of the deployed arithmetic, not of a
software proxy.

**Caveat we state plainly:** our LUT and FF figures are pre-place-and-route
estimates from da4ml's cost model, whereas JEDI-Linear's are post-P&R Vivado
results. We do not have a Vivado license and will not present the two as
equivalent. Additionally, JEDI-Linear's designs close timing at 258 MHz (pT-sorted)
and 203 MHz (permutation-invariant) while ours are quoted at a 300 MHz target, so
pipeline depth in cycles — not nanoseconds — is the like-for-like latency
comparison at this stage.

## §2 — Why our logic cost exceeds JEDI-Linear's (the structural argument)

The gap is architectural and we think it is the interesting part of the comparison,
not an implementation deficiency. **JEDI-Linear contains zero data×data
multiplies.** Every one of its MACs is a constant-weight matrix–vector product, and
constant-weight multiplication is exactly the case distributed arithmetic
annihilates: the multiplier collapses into a shared shift-add graph whose cost falls
as QAT drives weight bits to zero. That is where their ~98k LUT comes from.

Attention cannot access that reduction. Q·Kᵀ and (attention weights)·V are
data×data products — neither operand is a compile-time constant — so no
constant-folding applies and each product must be realized as a general multiplier,
plus softmax lookup logic that has no DA equivalent. This is the dominant term in
our cost, and it is a property of attention itself rather than of our lowering. The
honest framing for the reader is therefore a trade, not a win on every axis:
attention buys accuracy and permutation-invariance-by-construction, and it costs
logic that a CMVM-only architecture does not spend.

## §3 — Grid choice (response to "is the fixed grid principled?")

The submitted model used a per-jet **dynamic** grid, which is data-dependent
addressing and untraceable for the DA converter, so the hardware model uses a fixed
grid. We selected its edges from jet physics, not convenience, and validated the
choice by training.

Extent ±0.40 in (Δη, Δφ) comes from pT containment: measured over 200k training
jets, only **0.007%** of constituent pT falls outside ±0.40 (versus 0.62% at ±0.25),
and the outer bins clamp rather than discard, so nothing is lost. Resolution is
non-uniform — 0.08-wide bins inside |Δη|,|Δφ| < 0.20 and one wide 0.20 bin at each
edge — because the energy is concentrated: 50% of jet pT lies within r ≈ 0.05 and
90% within r ≈ 0.17–0.20 depending on class, so a uniform grid would spend equal
resolution on an essentially empty periphery. Non-uniform edges are free in
hardware: each bin is one threshold comparison either way. The 0.08 core width also
resolves both prongs of a W/Z decay, whose median 2m/pT opening angle is 0.146 and
0.152 respectively.

The decisive constraint is **parity of the bin count**, and it is specific to the
fixed grid. With an even symmetric count a bin boundary falls exactly on the jet
axis, splitting the leading prong across four cells; the dynamic grid never suffered
this because per-jet min-shifting moved the core off the boundary. Measured median
pT fraction captured by the single hottest cell: **47.5% for 7 bins, 33.9% for 8,
43.9% for 9, 27.4% for 12**. The odd/even alternation dominates the trend with
spacing. A controlled shootout (identical seed, data and schedule; only the grid
varies) confirms the prediction: our 7×7 non-uniform grid reaches 71.90% versus
69.93% for the 8×8 uniform baseline, at 17% lower EBOPs and 23% fewer cells. Both
odd grids beat the even baseline by ≈2 points.

Fixing the grid therefore does not degrade the method: the LN-free hardware
architecture with these fixed edges reaches **81.59%** in float, level with
JEDI-Linear's permutation-invariant row (81.6%) and above their pT-sorted headline
(80.9%).

## §4 — Status note (internal — resolve before submitting)

The frontier in §1 is a snapshot at **epoch 261 of a 1000-epoch QAT run that is
still training**. Two consequences:

1. The accuracy numbers in §1 are *not* our best. The run starts at 81.21% after
   float transfer and the β penalty then trades accuracy for logic as it ramps. At
   epoch 261 β had reached only 2.8e-7 of its final 3e-6, so most of the
   compression is still ahead. Early high-accuracy checkpoints exist (81.13% at
   epoch 2) but are not synthesizable — 185M LUT-est, ~100× the device.
2. **Update §1 with the final frontier before submitting**, and re-check the
   "6.2 points / 6.7×" gap quoted in the figure. If the closing frontier does not
   reach JEDI-Linear's accuracy at comparable logic, say so directly and lean on
   §2 — the structural argument stands on its own and is more persuasive than a
   contested claim of parity.

Do not describe the estimates as post-P&R under any rewrite.
