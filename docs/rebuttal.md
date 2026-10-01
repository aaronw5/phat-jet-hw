# Rebuttal material — hardware comparison against JEDI-Linear

All numbers below are measured, not estimated from the papers. The generated RTL
is verified bit-exact against the Keras model (§5). Provenance and caveats are in
§5. **Read §4 before using any of this.**

## §1 What was measured, and why it is a fair comparison

Both models were pushed through **one instrument**: da4ml 0.6.0, `HWConfig(1,-1,-1)`,
solver `hard_dc=2`, `clock_period = 2.0` ns, `latency_cutoff = 2`, target
`xcvu13p-flga2577-2-e`. Those are JEDI-Linear's *own* settings, taken from their
`src/syn_test.py` — not settings we chose. We did not compare our tool's output
against their paper's numbers; we ran their published model through our pipeline.

We reproduced their result first, as a gate on everything else. Rebuilding their
architecture under the installed HGQ2 and loading their official epoch-1565
checkpoint recovers **80.91%**, matching their published 80.9% for the pT-sorted
3-feature/128-constituent configuration.

**Estimator calibration.** On their design da4ml estimates 99,690 LUT against the
Vivado post-route report they ship (97,822 LUT) — **+1.9%**. Our LUT numbers are
therefore trustworthy at roughly the 2% level, and the comparison below is not an
artifact of the measurement tool. The FF estimate is pessimistic by +41% on the
same design, and is flagged as an upper bound wherever it appears.

## §2 The result

| Model | Accuracy | LUT | Latency | VU13P occupancy |
|---|---|---|---|---|
| JEDI-Linear (their Vivado post-route) | **80.91%** | **97,822** | **46 ns** | 5.7% |
| PHAT-JeT, best point that fits the device | 78.64% | 1,188,245 | 152 ns | 69% LUT / ≤81% FF |

**We are 2.27 accuracy points behind at 12× the logic.** Both designs use zero DSPs
and zero BRAM. This is the honest statement of where the architecture currently sits.

Two facts qualify it, and both are defensible in a rebuttal:

1. **The frontier is flat.** Going from 1.19M to 6.83M LUT — 5.7× more logic — buys
   only **+0.36 points** (78.64% → 79.01%). There is no hidden operating point that
   closes the gap by spending more area; the useful region is entirely at or below
   ~1.2M LUT. This is a statement about the architecture, and it is worth making
   explicitly rather than letting a reviewer assume we simply undertrained.

2. **Quantization, not the architecture, costs the most.** The same model and grid
   in float reaches **81.59%**. At the deployable point we lose 2.95 points to
   quantization — more than the entire gap to JEDI-Linear. That is a training-recipe
   problem, not an architectural ceiling, and it is the single highest-value thing
   to fix.

## §3 Why the cost is structural, and what it says about the contribution

The cost is not in the weights. PHAT-JeT has **5,429 parameters**; JEDI-Linear has
204,453 — a 37× *larger* model that costs 12× *less* logic. The mechanism: every one
of JEDI-Linear's multiplications has a constant operand, which da4ml folds into
shared adder trees. Attention's `q·k` and `a·v` products have **both operands
data-dependent**, so they require real multipliers. At the deployable point that is
14,910 such multiplies at 18.7 LUT each (23.5% of logic), on top of 575,727 LUT of
adder/subtractor logic (48.6%) and 246,327 LUT of lookup tables (20.8%).

Per-layer, attention accounts for ~77% of cost. **The physics-motivated fixed GMP
grid — the paper's actual novelty — is 1.1% of the cost budget.** That is a genuinely
good result and should be stated plainly: the contribution is nearly free in hardware.
The expense is the attention mechanism it sits inside, which is standard machinery.

## §4 Claims to make, and claims to retract

**Do not claim** lower latency, lower LUT, or better accuracy-per-resource than
JEDI-Linear. All three are contradicted by §2, and their code and checkpoints are
public — a reviewer can check.

**Do not present** any operating point above ~1.2M LUT as deployable without saying
it exceeds the VU13P.

**Never describe** our numbers as post-place-and-route. They are da4ml estimates —
bit-exact-verified RTL, but estimated resources. Only the JEDI-Linear row is
post-route, and it is theirs, not ours. "Bit-exact" licenses "this is the circuit";
it does not license "these are post-route numbers".

**Do claim**, because each is measured: that the fixed physics grid costs ~1% of the
logic budget; that the architecture fits the target device at 69% LUT occupancy while
reaching 78.64%; that the accuracy/area frontier is flat above 1.2M LUT, so the gap
is not a matter of underspending area; that attention removal costs −51.6 points,
which establishes the mechanism is load-bearing rather than decorative; and that the
generated RTL is bit-exact against the quantized model.

**Reframe the contribution** away from resource efficiency and toward the
architecture: the physics-motivated fixed grid and the attention formulation, with
the hardware results presented as a feasibility demonstration on a real device rather
than a Pareto improvement over JEDI-Linear.

## §5 Provenance and what is not verified

Every PHAT-JeT number is a **da4ml estimate**, calibrated at +1.9% on LUT against a
known post-route reference. FF estimates are upper bounds (+41% pessimistic).

**Bit-exactness is verified.** The generated Verilog was compiled with Verilator
5.050 and run against Keras inference on 2000 validation jets: **0 mismatching rows,
max absolute difference 0.0**. The RTL is a faithful implementation of the quantized
model, so the resource estimates describe the circuit that would actually be
synthesized. (Five macOS build-portability issues had to be worked around to get
Verilator to link; none of them is a simulator or model problem. See
`hw_results.json -> bit_exactness.macos_workarounds` and `scripts/bitexact.py`.)

**One checkpoint was excluded.** `epoch=166` records 78.01% in its filename but
re-evaluates to 71.94%; it is corrupt and is not in any table. The other nine traced
checkpoints reproduce their recorded accuracy exactly.

**An earlier version of this analysis was wrong** and its numbers should not be used.
It reported a 228× logic gap. That came from sampling frontier checkpoints by index
over a count-sorted list, which skipped the entire device-fitting region and selected
an epoch-14 snapshot taken before the EBOPs regularizer had reduced bitwidths — it
cost 555 LUT per data-dependent multiply, against 18–26 for converged points. It also
recorded `latency_cutoff = 4.0` when the runs used 2. Both are corrected here.
