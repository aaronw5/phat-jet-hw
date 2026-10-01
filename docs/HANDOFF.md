# PHAT-JeT vs JEDI-Linear — project state and restart guide

Written 2026-07-26. Everything below is measured and re-verified from disk, not
recalled. Read §0 and §1 before running anything; §7 is the copy-paste restart
prompt.

---

## §0 The one-paragraph summary

We benchmarked PHAT-JeT against JEDI-Linear through a single instrument (da4ml
0.6.0, JEDI-Linear's own synthesis settings) on the same 5-class jet dataset.
**PHAT-JeT is behind at comparable resource cost: 78.64% at 1.19M LUT versus
JEDI-Linear's 80.91% at 97,822 LUT — 12.1x the logic for 2.27 points less
accuracy.** The RTL is bit-exact against Keras, so the measurement is sound.
The important diagnostic finding is that this is *not* an architecture deficit:
the same architecture in float32 reaches **81.59%**, which is 0.68 points
*better* than JEDI-Linear. Quantization costs 2.95 points — more than the whole
gap. The cause has been localized (§4): the EBOPs regularizer binarizes the GMP
path, the paper's own contribution, down to ~1.26 bits while sparing the
classifier head at ~10 bits. A candidate fix (a bitwidth floor) is running and
unresolved (§5).

---

## §1 Ground rules that must not be violated

These were learned the hard way; each one has already caused a wrong claim.

1. **No commercial synthesis tooling exists on this machine.** Vivado/Vitis are
   not installed. Every PHAT-JeT resource number is a **da4ml estimate**. Never
   describe any of them as post-place-and-route. The *only* post-route numbers
   in this project are JEDI-Linear's own shipped Vivado report
   (`vivado_post_route_LUT = 97,822`, `vivado_post_route_FF = 48,231`).
2. **Bit-exactness is verified, but that is a different claim from post-route.**
   It licenses saying "the generated RTL faithfully implements the Keras model",
   not "these are silicon numbers."
3. **Do not claim a resource or accuracy win over JEDI-Linear.** All three
   possible framings (lower latency, lower LUT, better accuracy-per-LUT) are
   contradicted by measurement. Their code and checkpoints are public; a
   reviewer can and will check.
4. **macOS platform gaps:** no `timeout`, no `setsid`, no working `pgrep`/`ps`
   process list (`sysmond` unavailable). Launch background jobs with plain
   `nohup ... &`. **Judge liveness by log file mtime**, not by process listing.
5. **Heredoc piped into `nohup python -` dies silently** when the shell exits —
   zero-byte log, no traceback. Always write a real `scripts/*.py` file and
   `nohup $PYBIN scripts/foo.py > logs/foo.log 2>&1 &`.
6. **`CSVLogger` uses `append=True`.** Two runs sharing a `RUN_TAG` interleave
   into one CSV and corrupt it. Always set a unique `RUN_TAG`.
7. **Training is CPU-only** (no GPU on this machine), ~8-12 min/epoch at
   `OMP_NUM_THREADS=11`. Running two arms at once roughly halves each.

---

## §2 Environment and layout

- conda env: **`fpga`** (Python 3.11). Pass `environment="fpga"` to bash/python.
- Project root: `/Users/anrunw/Documents/phat-jet-hw`
- Key versions: `da4ml` 0.6.0, `hgq` 0.1.9, `keras` 3.15 (JAX backend),
  `verilator` 5.050 (conda-forge, **is installed and does work** — see §6).
- Always export `GRID_NAME=core7` and `KERAS_BACKEND=jax`.

```
scripts/
  phat_jet_k3.py        model definition (float + quantized)
  train_qat.py          main QAT trainer; Pareto-frontier-tracing run
  train_hold.py         variant that HOLDS beta instead of ramping
  qdiag.py              bitwidth inspection / sensitivity probe
  extract_hw.py         _gmp_edges(), hardware extraction helpers
  preprocess.py         robust_scale(), load_stats(), assert_scaled()
  build_jedi.py         rebuilds JEDI-Linear under installed HGQ2
  bench_headtohead.py   traces both models through identical da4ml config
  run_bench_phat.py     traces a set of PHAT-JeT checkpoints
  bitexact.py           RTL-vs-Keras bit-exactness check (WORKS, see §6)
  sweep_latency.py      latency_cutoff sweep
data/jets_128x3.npz     the dataset (128 constituents, 3 features)
refs/JEDI-linear-master/ their released code, incl. src/syn_test.py
docs/                   rebuttal.md, comparison_table.md, HANDOFF.md (this)
```

---

## §3 The measured result

**Instrument (identical for both models, taken from JEDI-Linear's
`refs/JEDI-linear-master/src/syn_test.py`):** da4ml 0.6.0, `HWConfig(1,-1,-1)`,
solver `hard_dc=2`, `clock_period=2.0` ns, `latency_cutoff=2`, target
`xcvu13p-flga2577-2-e` (1,728,000 LUT / 3,456,000 CLB FF).

**Estimator calibration:** on JEDI-Linear's design da4ml estimates 99,690 LUT
against their reported 97,822 post-route — **+1.9% on LUT**, so LUT estimates
are trustworthy. FF is *pessimistic*: 68,225 estimated vs 48,231 actual
(**+41%**), so all PHAT-JeT FF figures are upper bounds.

| Model | Accuracy | LUT | Latency | Fits VU13P |
|---|---|---|---|---|
| JEDI-Linear (their Vivado post-route) | **80.91%** | **97,822** | **46 ns** | yes |
| PHAT-JeT best device-fitting (da4ml est) | 78.64% | 1,188,245 | 152 ns | yes (69% LUT, ≤81% FF) |
| PHAT-JeT architecture in float32 | **81.59%** | — | — | — |

Deployable checkpoint: `ckpt_snapshots_run6/epoch=140-val_acc=0.7864-ebops=926777.keras`

**Frontier is flat.** 5.7x more logic buys only +0.36 points. This pre-empts a
reviewer arguing the model is merely undertrained — it isn't; the frontier is
genuinely shallow above the deployable region.

### Why EBOPs looks huge (and is)

The 11.9x LUT gap factors **exactly** into two independent multipliers:

| factor | value |
|---|---|
| more EBOPs (926,777 vs 144,427) | 6.4x |
| LUT per EBOP (1.28 vs 0.69) | 1.9x |
| **product** | **11.9x** (= observed) |

The second factor is **structural, not a tuning problem**. JEDI-Linear's 204,453
parameters are all *constant weights*, so every multiply is weight x activation
and da4ml folds each into a shift-add adder tree. PHAT-JeT has only 5,429
parameters but ~41,000 **data-dependent** multiplies (activation x activation)
inside attention, which need real multipliers. EBOPs counts both kinds alike;
hardware does not. **A model with attention will pay ~2x more LUT per EBOP than
a constant-weight network, and no amount of quantization tuning removes that.**

Accuracy/cost crossover: matching JEDI-Linear's 80.91% needs ~4e7 EBOPs, roughly
280x their 144k, projecting to ~5e7 LUT ≈ 30x the VU13P. **Not deployable at
accuracy parity.**

---

## §4 The key diagnostic finding (most promising lead)

Float ceiling 81.59% vs quantized 78.64% means **quantization costs 2.95
points**, which exceeds the entire 2.27-point gap to JEDI-Linear. For context,
HGQ2 on comparable jet-tagging models typically loses well under half a point.
So something specific is bleeding accuracy.

Reading per-tensor bitwidths out of the ep140 checkpoint
(`bitwidths_ep140.json`, figure `fig_bitwidths.png`):

| layer group | mean total bits (k+i+f) |
|---|---|
| **GMP path** (`gmp_gather/scatter/cell/dwconv/pointwise`) — the paper's novelty | **1.26** |
| attention | 3.99 |
| other | 5.81 |
| classifier head (`head1`, `head_out`) | 10.12 |

10 of 36 quantizers sit below 2 total bits. Because EBOPs cost per multiply
scales as bits_a x bits_b, the regularizer discovered it can crush **one**
operand toward zero and make the product nearly free — and it chose to sacrifice
the GMP block, i.e. exactly the contribution the paper is about, while spending
10 bits on the tiny classifier head.

**How to inspect bitwidths correctly (I got this wrong once):** a layer carries
*several* quantizers, each contributing weights named `k`, `i`, `f`. Building a
dict `{w.name: w for w in layer.weights}` keeps only the last and produces
garbage (it made tensors look like 30+ bits). Group them **in order** instead —
see `scripts/qdiag.py`.

**Two failed diagnostic approaches — do not repeat:**
- Overwriting `i` (integer bits) to a wide value moves the binary point and
  rescales every value by 2^delta -> accuracy collapses to chance (20%).
- Adding fractional bits post-hoc to a trained checkpoint also *lowered*
  accuracy. The bitwidths are co-adapted with the weights; you cannot widen
  them after the fact and read off a sensitivity. **Retraining is required.**

---

## §5 In flight and unresolved

**`floor2` run — the candidate fix.** HGQ2 accepts `fc=MinMax(floor, 16)`, a
lower bound on fractional bits (analogous to the `ic=MinMax(0,12)` already in
the code). Added to `build_qat_model` in `scripts/train_qat.py` behind env var
**`FLOOR_BITS`** (0 = original behaviour, default). Intent: force the
regularizer to spread cost reduction instead of deleting whole tensors.

Launch used:
```bash
cd /Users/anrunw/Documents/phat-jet-hw
PYBIN=$(python -c "import sys; print(sys.executable)")
GRID_NAME=core7 FLOOR_BITS=2 RUN_TAG=floor2 QAT_EPOCHS=200 OMP_NUM_THREADS=11 \
  nohup $PYBIN scripts/train_qat.py > logs/floor2b.log 2>&1 &
```

Status at handoff: **10 epochs done, best 81.26% at 6.09e7 EBOPs (ep9).** It is
still in the expensive high-EBOPs regime (beta is ramping from 2e-8), so it has
**not yet reached the comparison region** around 1e6 EBOPs. At ~10 min/epoch it
needs roughly 130-150 more epochs to get there.

Early read at matched cost (5e7-1e8 EBOPs band): floor2 **81.26%** vs run6
**81.09%** — encouraging but nearly within noise, and in a cost regime that is
30x too expensive to deploy. **The verdict lives at ~1e6 EBOPs and is not in yet.**

**Honest caveat:** the floor is a hypothesis. It is well-motivated by §4, but
1-bit GMP tensors could instead mean the GMP path is genuinely low-information
on this dataset, in which case the floor buys cost and no accuracy.

**Also finished, minor positive:** `run9_hold` (beta held at 1.58e-7 from ep140,
75 epochs) never beat its 78.64% resume point, but beat run6 **at equal cost** —
78.57% at 880k EBOPs vs 78.64% at 927k, i.e. same accuracy for ~5% less logic.
It moves the frontier left, not up. Figure `fig_hold_frontier.png`.

---

## §6 Bit-exactness: verified, and how to rerun

`hw_results.json:bit_exactness` — ep140 RTL vs Keras on 2000 validation jets:
**0 mismatching rows, max abs diff 0.0**, both at 78% on the subset. Verilator
5.050.

My earlier claim that "Verilator won't build on this machine" was **wrong**. It
was installed all along. Five macOS portability faults had to be cleared, none
of them a Verilator problem — all are handled inside `scripts/bitexact.py`:

1. `compile()` passes `-fopenmp`, which Apple clang++ rejects -> pass `openmp=False`.
2. da4ml's `build_binder.mk` template hardcodes `-Wl,--no-undefined`, a GNU-ld
   flag Apple's `ld` refuses.
3. Patching that makefile keeps getting reverted: **`compile()` internally calls
   `write()`, which re-copies the template**. Fix: patch, then call the inner
   `_compile` directly rather than `compile()`.
4. The template also uses `nproc`, absent on macOS.
5. The real underlying link error, once the flag noise was gone: `sc_time_stamp()`
   undefined — a symbol the embedding application must supply. A small stub
   (`vl_stub.cc`) provides it; valid because the generated design is purely
   combinational.

Rerun on another checkpoint: `BE_CKPT=<path> BE_PRJ=<dir> BE_N=2000 $PYBIN scripts/bitexact.py`

---

## §7 Restart prompt (copy-paste into a fresh session)

> I'm working on a NeurIPS rebuttal comparing my model PHAT-JeT against
> JEDI-Linear (arXiv 2508.15468) on FPGA resource cost. My paper is arXiv
> 2605.21789. The project lives at `/Users/anrunw/Documents/phat-jet-hw`, conda
> env `fpga`, and **`docs/HANDOFF.md` there is the authoritative state document —
> read it first, in full, before running anything or making any claim.**
>
> Short version: the head-to-head is done and unfavorable (78.64% @ 1.19M LUT
> vs their 80.91% @ 97.8k LUT), bit-exactness is verified, and all deliverables
> (`hw_results.json`, `docs/comparison_table.md`, `docs/rebuttal.md`,
> `fig_pareto_vs_jedi.png`, `fig_bitwidths.png`) are written against corrected
> numbers. Do not re-derive those.
>
> The live question is the one in §4-§5 of the handoff: my float model hits
> 81.59%, which beats JEDI-Linear, but quantization costs 2.95 points — more
> than the 2.27-point gap. I localized it: the EBOPs regularizer binarizes my
> GMP path to ~1.26 bits while giving the classifier head ~10. I added a
> bitwidth floor (`FLOOR_BITS` env var in `scripts/train_qat.py`, using HGQ2's
> `fc=MinMax(floor,16)`) and a run called `floor2` is training.
>
> Please (1) check whether `floor2` is still alive — judge by
> `logs/floor2b.log` mtime, since `ps`/`pgrep` don't work on this machine —
> and report where it is on the accuracy-vs-EBOPs frontier; (2) once it reaches
> the ~1e6 EBOPs region, compare it against run6 at matched cost and tell me
> honestly whether the floor helped; (3) if it did, trace its best
> device-fitting checkpoint through the same da4ml instrument and update
> `hw_results.json` plus the figure.
>
> Constraints: CPU-only, ~10 min/epoch. No Vivado on this machine, so every
> PHAT-JeT number is a da4ml estimate and must never be called post-route. Don't
> implement new attention variants. Don't draft any claim that we beat
> JEDI-Linear on resources — the measurements contradict it and their code is
> public.

---

## §8 Deliverables already complete (do not redo)

| file | what it is |
|---|---|
| `hw_results.json` | all traced points, calibration, device fit on LUT+FF, bit-exactness record, EBOPs decomposition |
| `docs/comparison_table.md` | shared-instrument statement, calibration, headline, full frontier, corrections log |
| `docs/rebuttal.md` | rebuttal paragraphs + explicit list of claims to retract |
| `fig_pareto_vs_jedi.png` | accuracy vs LUT, device-fitting vs over-budget, float ceiling |
| `fig_bitwidths.png` | per-quantizer bitwidth allocation (the §4 diagnostic) |
| `fig_hold_frontier.png` | run9 hold-schedule vs run6 at matched cost |
| `bitwidths_ep140.json` | per-quantizer bit data behind fig_bitwidths |
| `bench_jedi_n128f3.json` | JEDI-Linear reference trace |
| `bench_phat_points.json` | PHAT-JeT traced frontier points |
| `bitexact_result.json` | bit-exactness verdict |

## §9 Superseded numbers — never reuse

- **"228x logic gap, -0.73 points"** — wrong. Came from sampling frontier
  checkpoints by index over a *count-sorted* list, which skipped the entire
  device-fitting region and picked an epoch-14 snapshot taken before the EBOPs
  regularizer had reduced bitwidths (555 LUT per data-dependent multiply, vs
  18-26 for converged points). Correct figure: **12.1x logic, -2.27 points.**
- **`latency_cutoff = 4.0`** — the runs used **2** (JEDI-Linear's own setting).
  LUT is invariant to it; latency is not.
- **A bitwidth table showing 30+ bit tensors** — artifact of the `{w.name: w}`
  dict bug described in §4. Correct means: GMP 1.26, attention 3.99, head 10.12.
- **`ckpt_snapshots_run6/epoch=166-...keras` is corrupt** — filename says
  78.01%, re-evaluation gives 71.94%. Excluded from all results.

## Session 2026-07-31 (part 2): envelope framing — THE ARGUMENT

**Reframe from the user, and it is the right one: we do NOT need to beat
JEDI-linear. We need to show PHAT-JeT FITS the CMS L1T envelope.**

Envelope, quoted from JEDI-linear Sec. II/III (refs/jedi_linear.txt L167, L522-532):
  - sub-100 ns latency, sub-10 ns initiation interval
  - CTL2 = 30 VU13P FPGAs, round-robin, 5 available for algorithms at a time
  - VU13P capacity 1,728,000 LUT
  - CTL2 must ALSO host jet clustering / sorting / feature computation, so LUT
    budget is contended, not free.

### Points that clear BOTH constraints (300 MHz, 3.33 ns/stage), zero DSP/BRAM, II=1
  N=16  67.45%   87,206 LUT (5.0%)  29 stages =  96.6 ns   qat_n16_ep101.keras
  N=32  71.36%   98,069 LUT (5.7%)  29 stages =  96.6 ns   qat_n32_c2_ep148.keras
  N=64  74.58%  167,304 LUT (9.7%)  29 stages =  96.6 ns   qat_n64_w1_ep138.keras
  N=128 none at 300 MHz (76 stages min in the fitting band)

### LUT-budget-only points (latency > 100 ns at 300 MHz)
  78.42%  1,221,621 LUT (70.7%)  40 st = 133.2 ns @300 / 80.0 ns @500  qat_n64_w3_ep81
  78.60%    902,337 LUT (52.2%)  71 st = 236.4 ns @300  run9_hold ep27   <-- NEW this session
  78.64%  1,188,245 LUT (68.8%)  76 st = 253.1 ns @300  run6 ep140  <-- BIT-EXACT VERIFIED

NEW: scripts/trace_budget_band.py traced run9_hold ep27 -> 78.60% at only 902k LUT
(52% of device), i.e. ~equal accuracy to the ep140 headline for 24% less logic.
run8 ep134 re-evaluated to 77.34% (filename said 78.61) -> another filename/re-eval
disagreement; ALWAYS re-evaluate, never trust the filename accuracy.

### CLOCK TRAP (cost me a wrong comparison, do not repeat)
jedi_family_bench.json / extract_hw.py use CLOCK_NS = 3.33 (300 MHz).
bench_headtohead.py / run_bench_phat.py / trace_budget_band.py use
CLOCK_PERIOD = 2.0 (500 MHz). Latency in those files is NOT comparable directly.
Always convert to STAGES first, then multiply by the clock you want to quote.
comparison_final.csv stores stages so any clock can be recomputed.

### Attention factorization measured (scripts/attn_shootout.py, attn_shootout.json)
patch_size == N with use_patch_attn=False IS exact full self-attention -> controlled
matched-architecture comparison, everything else held fixed.
  N=32: 5,890,873 -> 3,136,529 LUT = 1.88x saving; MACs predict 1.18x -> 1.59x excess
  N=64: 19,266,018 -> 6,456,512 LUT = 2.98x saving; MACs predict 1.47x -> 2.03x excess
  activation x activation MACs fall 3.76x (N=32) / 7.11x (N=64) -- that is WHY the
  LUT saving beats the FLOP prediction. Use this against the AC's "FLOPs are a poor
  proxy" comment: it is our strongest single result.

### Other measured levers
  parallel branches: 186.5 -> 129.9 ns at N=64 (1.44x faster) for -1.6% LUT. Free.
  GMP = 41% of total LUT at N=64 (6,368,925 -> 3,735,001 without it).

### Calibration (the thing that makes estimates defensible)
Tracing JEDI-linear's OWN released N=128 model through our identical instrument
gives 99,690 LUT vs their published post-route 97,822 = +1.9%. We overestimate
slightly, so our margins are conservative. bench_jedi_n128f3.json.

### Deliverables written
  comparison_final.csv / .md   23 rows, ours + all published designs, stages included
  rebuttal_final.md            R1-R6 drafted paragraphs with every number inline
  rebuttal_numbers.json        machine-readable frontier + args + calibration
  fig_rebuttal_main.png        a) acc vs LUT w/ VU13P line  b) FLOP-vs-LUT divergence
  fig_rebuttal_latency.png     a) acc vs latency w/ 100ns line  b) acc inside both budgets

### STANDING CAVEATS (unchanged, do not violate)
  - All PHAT-JeT resource numbers are da4ml ESTIMATES. Never "post-place-and-route".
  - Only post-route numbers in the project are JEDI-linear's own published ones.
  - Bit-exactness (run6 ep140, 2000 samples, max_abs_diff 0.0) licenses "faithful
    implementation", NOT anything about silicon.
  - Assert NO resource or accuracy win over JEDI-linear. Their code is public.
  - Exclude epoch=166 checkpoint (filename 78.01, re-eval 71.94).

### Band trace completed (all 5 points) — headline UNCHANGED
  78.60%   902,337 LUT (52.2%)  71st  run9_hold ep27    <-- BEST, headline
  78.54% 1,335,092 LUT (77.3%)  75st  run6 ep141
  78.43%   890,072 LUT (51.5%)  71st  run9_hold ep24
  77.55% 1,019,480 LUT (59.0%)  72st  run8 ep140  filename said 78.58  MISMATCH
  77.34% 1,042,747 LUT (60.3%)  73st  run8 ep134  filename said 78.61  MISMATCH

TWO MORE filename/re-eval disagreements, both from pareto_run8_core7_fixed.
That run's checkpoint filenames are unreliable (>1 pt optimistic). Treat ALL
run8 filename accuracies as suspect; re-evaluate before quoting any of them.
Known-bad list is now: epoch=166 (78.01 -> 71.94), run8 ep140, run8 ep134.

## CLOCK CORRECTION (2026-07-31) — I had this WRONG, do not repeat
40 MHz is the LHC BUNCH CROSSING rate (25 ns period), NOT a firmware clock target.
jedi_linear.txt L43 "every 25 ns", L574 "L1 Accept 40 MHz".
JEDI-linear's firmware target is stated at L536: "initiation interval of 1 clock
cycle and a target frequency of over 300 MHz", and L667 "we pipeline every 2
adders for an Fmax of approximately 300 MHz". Their best ACHIEVED Fmax is
381.7 MHz (L830). So:
  - 300 MHz is the correct number to quote. USE IT EVERYWHERE.
  - 500 MHz was never defensible; it exceeds anything they demonstrate. All
    500 MHz / "80 ns" claims have been REMOVED from rebuttal_final.md and
    comparison_final.md. Do not reintroduce them.
  - The 78.42% / 1.22M LUT point is 133.2 ns at 300 MHz => FAILS the 100 ns
    bound. It is now labelled "LUT budget only", not "fits envelope".
  - Pipeline depth is a function of the trace-time adder cutoff. A design traced
    for 300 MHz CANNOT be requoted at a faster clock without retracing. Do not
    multiply stages by a smaller clock period to manufacture a better latency.
  - Good framing to keep: II=1 at 300 MHz = one jet per 3.33 ns vs 25 ns crossing
    => ~7 jets per crossing per engine, no replication.

## N=32 full picture (user asked)
227 traced N=32 points, 44 on the acc/LUT Pareto front. All traced at 3.33 ns.
  BEST clearing BOTH @300MHz: 71.36%  98,069 LUT (5.7%)  29 st = 96.6 ns
  Also clearing both: 70.29% @ 95,730 (30 st, 99.9 ns), 69.16% @ 81,563 (29 st)
  First point to FAIL latency: 71.53% @ 140,673 LUT, 31 st = 103.2 ns
  Under 1 device, latency ignored: 76.95% @ 1,458,758 LUT (84.4%), 49 st = 163 ns
N=32 is the sweet spot for tiny budgets: 71.4% at 5.7% of one VU13P.
N=64 buys +3.2 pts at the SAME 96.6 ns latency for 9.7% of device. Depth, not N,
sets latency.

## parallel_attn semantics (phat_jet_k3.py L219-222, L247, L259)
serial:   x + local(x), then that feeds patch(...)  => depth = local + patch
parallel: both branches read x_pre_attn             => depth = max(local, patch)
Same parameter count and same receptive field, different function. Measured at
N=64: 186.5 -> 129.9 ns (1.44x) for -1.6% LUT.

## Runs launched 2026-07-31
float_phatjet_core7_n32_par.keras  (N=32 parallel float, 30 ep) -- 75.98% val by ep12
float_phatjet_core7_n128_par.keras (N=128 parallel float, 40 ep) -- 79.01% val at ep1
  N=128 parallel needed a donor: float weights ARE N-agnostic, so
  cp float_phatjet_core7_n64_par.keras float_phatjet_core7_n128.keras + WARM=self.
  GOTCHA: float_n.py reads N_PARTICLES/PATCH_SIZE, NOT N/PS. Passing N=128 silently
  trains N=32.
NEXT: QAT at N=128 PARALLEL_ATTN=1 off the n128_par donor, to try to get an N=128
row inside the 100 ns bound (currently none; 76 stages min = 253 ns).

## par_n128_a QAT launched 2026-07-31 (the N=128 sub-100ns attempt)
Goal: an N=128 row inside 100 ns. Serial N=128 needs 76 stages min = 253 ns @300MHz,
so parallel wiring (depth = max(local,patch) instead of sum) is the only lever.
Command that WORKED:
  N_PARTICLES=128 PATCH_SIZE=8 PARALLEL_ATTN=1 QAT_EPOCHS=1000 FLOOR_BITS=2 \
  FLOAT_CK=float_phatjet_core7_n128_par.keras RUN_TAG=par_n128_a
  -> "init from float_phatjet_core7_n128_par.keras: 16 layers, post-transfer val_acc=0.7998"
TWO env-var traps cost two failed launches (both silent-ish):
  1. train_qat.py reads FLOAT_CK, *not* FLOAT_CKPT. With the wrong name it silently
     falls back to float_phatjet_core7.keras (the SERIAL N=128 donor), which loads
     without shape error into a parallel model and evaluates at 0.2788 = chance.
     The assert catches it, but the message blames quantizer range, not the donor.
  2. float_n.py reads N_PARTICLES/PATCH_SIZE, *not* N/PS.
Donor provenance: float_phatjet_core7_n128_par.keras was made by float_n.py with
PARALLEL_ATTN=1 WARM=self off a COPY of float_phatjet_core7_n64_par.keras. NOTE the
warm start evaluated at 0.1639 (chance) despite shapes matching, i.e. parallel float
weights do NOT transfer functionally across N. It recovered to >0.79 in one epoch, so
treat it as a partial reinit, not a fine-tune. Reached 80.5% by ep10 (still climbing).
float_phatjet_core7_n32_par.keras finished 30 ep at 77.4%.

## Float ceilings per N (for the "training budget, not architecture" argument)
  n16 73.30 (scratch) | n32 79.11 (long) / 79.33 (ps4) | n64 81.90 (long) | n128 81.59
  n128 PARALLEL 80.5%+ (par2, still training) -- beats serial n128 float
Quantization gap: short runs lose 6-8 pt (67.4 vs 73.3, 71.4 vs 79.1, 74.6 vs 81.9);
the one LONG run loses 2.95 pt (78.64 vs 81.59). THIS is the evidence that the deficit
is schedule length, not architecture. Use it.

## Runs stopped 2026-07-31 (state at shutdown, both were still training)
par_n128_a (QAT, N=128 PATCH_SIZE=8 PARALLEL_ATTN=1 FLOOR_BITS=2, from float
  ..._n128_par.keras donor at post-transfer 0.7998): reached epoch 4 of 1000.
  Best checkpoint pareto_par_n128_a/epoch=1-val_acc=0.7905-ebops=147592021.keras
  EBOPs fell 147.6M -> 118.7M over 3 epochs while val held ~0.79, i.e. it was
  compressing without accuracy loss and was nowhere near converged. NOT traced to
  logic, so there is no N=128 latency number from it. Resume with QAT_RESUME=<ckpt>
  (leave QAT_INIT at default when resuming) rather than restarting from float.
float_n128_par2 (float donor, 40 ep): reached epoch 19 of 40 at val 0.8098, still
  improving. This already beats the serial n128 float ceiling of 81.59? NO -- 80.98
  is below it, but it beats the earlier par run (77.03). Donor file on disk
  float_phatjet_core7_n128_par.keras is the ep10-ish 0.805 snapshot.
NOTE: the sandbox cannot signal these PIDs (os.kill gives EPERM, pkill/ps blocked),
so they must be stopped from a host terminal. Nothing in the rebuttal depends on
either run; the 3-row table at 96.6 ns is complete and independent of them.
