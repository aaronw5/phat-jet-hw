# Debugging high LUT / latency: two real defects found

Both were found by measuring the traced design rather than reasoning about the
architecture. Two hypotheses were tested and **refuted** first, recorded here so
they are not re-tried.

## Refuted hypotheses

**H1 — the one-hot GMP tensor instantiates real multipliers.** Plausible: the
grid-cell tensor is one-hot, so `x * onehot` should be a mask, and fractional
bits on its quantizer would force full multipliers. Refuted: a per-layer cost
breakdown showed the GMP ops are a negligible share of total logic; the softmax
layers dominate.

**H2 — attention saturated into a hard argmax.** Raw local-attention scores span
-508..+512, which for a softmax over 8 elements suggested one-hot output and a
missing `1/sqrt(d)` scale. Refuted by measuring attention entropy directly:
**1.989 nats out of a maximum 2.079** (effective 7.4 of 8 attended). Attention is
near-uniform; the extreme scores are rare tail events, not the typical case.
The scale is absorbed into W_q as documented.

> Method note: the first attempt at H2 measured 24.5% accuracy (~chance) because
> the ad-hoc preprocessing omitted robust scaling. Any diagnostic that reports
> near-chance accuracy is measuring its own harness, not the model. See defect 3.

## Defect 1 — exp-table integer bits sized by outliers (the dominant cost)

`QSoftmax`'s exp table is addressed by the quantized value of `(max - score)`.
HGQ sizes that quantizer's integer bits from the **observed maximum**, and the
score distribution has a long tail:

| softmax | p99.9 of (max-score) | max | i allocated | needed for p99.9 |
|---|---|---|---|---|
| local (over 8)  | 14.0 | 189.5 | 8-9 | 4 |
| patch (over 16) | 23.3 | 43.0  | 6   | 5 |

Those extra bits are pure waste, because the table's **output** is only 3 bits:
stable softmax computes `exp(-d)` for `d >= 0`, and `exp(-d) < 2^-3` for every
`d > 2.1`. Every one of the extra entries encodes a value that quantizes to the
same near-zero output. Measured: 123 of 181 tables had >= 1024 entries; lookup
ops were **29% of total logic** (389k of 1.33M LUT).

Cap sweep on the epoch-141 checkpoint (no retraining, clamp applied post-hoc):

| exp i cap | val acc | total LUT | lookup LUT |
|---|---|---|---|
| baseline (8.97) | 78.33% | 1,333,682 | 389,343 |
| i <= 6 | 78.34% (+0.01) | 1,103,052 (-17.3%) | 184,873 |
| **i <= 5** | **78.38% (+0.05)** | **1,005,050 (-24.6%)** | **94,309** |
| i <= 4 | 75.21% (-3.12) | 953,613 (-28.5%) | 48,914 |
| i <= 3 | 62.12% (-16.21) | 923,325 (-30.8%) | 26,200 |

`i<=5` is the last safe setting -- the cliff at 4 is where real signal starts
being clipped. End-to-end with 6 epochs of fine-tuning at the cap:

**1,333,874 -> 934,703 LUT (-29.9%), lookup 389,343 -> 55,118 (7.1x),
latency 151 -> 143 ns, accuracy 78.33% -> 78.00%.**

### Why the obvious fix does not work: an upstream HGQ bug

Setting `ic=MinMax(-16, 5)` on the exp input quantizer looks correct and does
nothing. In `hgq 0.1.9`, `FixedPointQuantizerKIF.call` (the class QSoftmax's exp
table uses) applies the constraint to the wrong variable in its WRAP training
branch:

```python
new_i = stop_gradient(maximum(self._i - decay, _new_i))  # from UNconstrained
if self._i.constraint is not None:
    _new_i = self._i.constraint(_new_i)   # constrained value -> _new_i
self._i.assign(new_i)                     # ...but new_i is assigned
```

The constrained result lands in `_new_i` and is discarded. Sister class
`FixedPointQuantizerKBI` has the same block correctly ordered, and KIF's own
*tracing* branch is correct -- only the KIF training path is wrong, which is why
it survives inspection. Keras's constraint machinery never runs either, because
under WRAP `i` is non-trainable (`i_trainable = overflow_mode != 'WRAP'`).

Verified empirically: with `MinMax(-16, 5)` attached, `i` grew 8.0 -> 10.99 over
8 gradient steps. With `ClampSoftmaxExpBits`, 3 epochs end at exactly 5.0/5.0.

Fix: `ClampSoftmaxExpBits` in `train_qat.py` reasserts the cap at each epoch end,
ordered **before** `FreeEBOPs` and `ParetoFront` so both see clamped bits.

## Defect 2 — input port wraps on real data (silent correctness bug)

`INPUTS_KIF` was `(1,6,8)`, i.e. range `[-64, +63.996)`. That was set from a
5,000-jet sample whose scaled-pT max was 60.9. Over the **full** train+val set
the max is **92.2**, and 58 constituents exceed 63.996. da4ml input ports WRAP,
so those became large **negative** pT:

    64.07  ->  -63.93     (sign flip on a jet's LEADING constituent)

Measured: 14 of 260,000 val jets affected; **prediction flipped on 5 of them**
(78.57% -> 71.43% on that subset). Aggregate impact ~0.004% -- immaterial to the
headline number, but it is a correctness defect at the hardware interface, and
the fix is free: widening to `i=7` costs **192 LUT of 1.33M (+0.014%)** with
latency unchanged. Now `INPUTS_KIF = (1,7,8)`.

## Defect 3 — preprocessing duplicated in four scripts

The robust-scaling block was copy-pasted into `train_qat.py`,
`float_pretrain.py`, `grid_shootout.py` and `extract_hw.py`. All four were
verified **bit-identical**, so no result was affected -- but an ad-hoc
reimplementation during this session got the mask order wrong and produced
near-chance accuracy while running normally.

The mask must be computed **before** scaling: pT median is 6.6, so after
subtracting it, a `> 0` test misclassifies **50% of all real constituents** as
padding. Now centralized in `scripts/preprocess.py` with `assert_scaled()`,
which checks the padding fraction (expected ~0.617) and fails loudly on exactly
this error.

## Status

The 1000-epoch run stopped at epoch 305 and was training the **unfixed** model.
It should be restarted with these fixes: the resulting frontier should sit
~30% lower in LUT at equal accuracy. All numbers above are da4ml trace-level
estimates, **not** post-place-and-route.

---

# Attention is NOT collapsed (checked against Laatu et al., arXiv:2510.24784)

That paper reports their 64-particle multi-head-attention block "consistently
collapsing over several trained models despite the bitwidth constrained to at
least one bit, turning it into a Deep Set". Since PHAT-JeT is attention-based at
128 particles, the same failure would invalidate the architectural claim. Tested
on ckpt epoch=141 (val 78.71% on the 8k subset):

| softmax | n | entropy / max | mean max weight | uniform weight | std |
|---|---|---|---|---|---|
| local_attn_softmax | 8 | 2.004 / 2.079 | 0.151 | 0.125 | 0.0124 |
| patch_attn_softmax | 16 | 1.844 / 2.773 | 0.402 | 0.0625 | 0.0972 |

Ablation: forcing both softmaxes to uniform weights (exactly the Deep Set the
paper describes) drops accuracy **78.71% -> 27.12% (-51.6 points)**.

So attention is load-bearing, and the patch level is strongly selective (max
weight 6.4x uniform). The local level is closer to uniform but still not
degenerate -- and note it is the local softmax whose exp table was oversized,
i.e. it spent bits on outliers while making near-uniform decisions, which is
exactly the waste that `EXP_I_MAX` removes.

# Cause of the run7 crashes: duplicate RUN_TAG, not OOM

Two training processes were launched with the same `RUN_TAG`, so they shared
`pareto_<tag>/` and `<tag>_log.csv`. `ParetoFront` in one process deletes
dominated checkpoints; the sibling then failed with `FileNotFoundError` on a
file it was about to write. The interleaved epoch numbers in the shared CSV
(0,1,0,2,1,3...) are the signature. Earlier "OOM" and "backgrounding" diagnoses
were both wrong.

Note: `ps` is blocked in this sandbox and returns "Operation not permitted",
which silently made liveness checks report live runs as dead. Use
`os.kill(pid, 0)` (PermissionError = alive, owned by another uid) or, more
robustly, check whether the CSV/log is still growing.

# Post-hoc clamping is not a substitute for training with the cap

| ckpt | baseline | post-hoc clamp | clamp + finetune |
|---|---|---|---|
| ep141 | 78.33% | 78.38% (+0.05) | 78.00%, -29.9% LUT |
| ep163 | 77.16% | 75.27% (-1.89) | -- |
| ep195 | 76.25% | 76.05% (-0.20) | -- |
| ep229 | 74.38% | 71.96% (-2.42) | 72.22%, -24.3% LUT |

Checkpoints further along the EBOPs ramp have already tightened their bit
allocations, so clipping removes bits they were using. The cap must be active
during training (run8), not applied afterwards.

---

# The real comparison target: JEDI-linear Table II (pT-sorted), not Table I

PHAT-JeT is pT-sorted, so the correct reference row is Table II
(non-permutation-invariant), NOT Table I (permutation-invariant).

| Model | N | Feat | Acc | Latn | LUT | FF | DSP | BRAM | II | Fmax |
|---|---|---|---|---|---|---|---|---|---|---|
| JEDI-linear (Tab II) | 128 | 3 | **80.9%** | 82 ns | **98k** | 48k | 0 | 0 | 1 | 257.6 MHz |
| JEDI-linear (Tab I)  | 128 | 3 | 81.6% | 138 ns | 296k | 163k | 0 | 0 | 1 | 203.1 MHz |
| MLPM (MLST'25)       | 128 | 3 | 79.8% | 72 ns | 83k | 21k | 0 | 0 | 1 | 208.7 MHz |

The pT-sorted target (98k LUT @ 80.9%) is ~3x HARDER on LUT than the
permutation-invariant row I first quoted. Do not cite Table I.

# Float ceiling: the architecture is fine, quantization is the gap

FLOAT PHAT-JeT, full 260k val set: **81.59%** with only **5,429 params**.

That is ABOVE JEDI-linear's 80.9%. So the 78.5% QAT number is a ~3.1-point
quantization loss, not an architectural deficit. This is the single most
important number for the rebuttal.

# Where the EBOPs actually go (ep141, 919k total)

| Block | EBOPs | Share |
|---|---|---|
| local_attn | 398,523 | 43.4% |
| patch_attn | 309,161 | 33.6% |
| other | 129,821 | 14.1% |
| embed/head | 71,702 | 7.8% |
| GMP | 9,985 | **1.1%** |

Top single layer: `local_attn_softmax` at 129,536 EBOPs (14.1%) -- the
oversized exp table. run8 is the first run with EXP_I_MAX active from epoch 0.

Note GMP is only 1.1% of cost. The grid study was worth doing for accuracy,
but grid choice is NOT a resource lever. Attention is 77% of the budget.

# HARD CONSTRAINT: no Vivado/Vitis on this machine

Checked: `vivado`, `vitis_hls`, `vsim` all NOT FOUND. Only `verilator` exists.

Every LUT/FF number obtainable here is a **da4ml trace estimate**, never a
post-synthesis or post-place-and-route result. JEDI-linear's 98k IS a real
VU13P implementation number. Estimate-vs-implementation is not a valid
comparison and must never be presented as one in the rebuttal.

# Periodic accuracy drops are cosine warm restarts, NOT failures

`cosine_decay_restarts(LR=3e-3, first_decay_steps ~= 14 epochs)` (matching
jsc150 / JEDI-linear's recipe, scaled from 500 steps over 7000 epochs).

At each cycle boundary the LR jumps ~1200x (e.g. ep70 2.47e-06 -> ep71 3.00e-03)
and val_acc drops sharply (run8: 77.9% -> 68.5% -> 62.8%). run6 shows the SAME
drop at the SAME epoch (78.5% -> 70.6% at ep71), then recovers.

Do not interpret these dips as divergence or as a bug. Read the Pareto
checkpoint list, not the last CSV row -- `ParetoFront` only keeps
non-dominated points, so it is immune to mid-cycle dips.
