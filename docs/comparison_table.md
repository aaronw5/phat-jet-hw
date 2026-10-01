# PHAT-JeT vs JEDI-Linear — hardware comparison

**One instrument for both models.** da4ml 0.6.0, `HWConfig(1,-1,-1)`, solver
`hard_dc=2`, `clock_period=2.0` ns, `latency_cutoff=2`, target `xcvu13p-flga2577-2-e`.
These are JEDI-Linear's *own* settings, read from their `src/syn_test.py`, so the
comparison is apples-to-apples rather than our-tool-vs-their-paper.

**Estimator calibration.** On JEDI-Linear's own design da4ml estimates 99,690 LUT
against their shipped Vivado post-route report of 97,822 LUT — **+1.9% error**.
LUT estimates are therefore trustworthy at the ~2% level. FF is pessimistic (+41%)
and is quoted with that caveat. All PHAT-JeT rows are **estimates, not post-route**.

## Headline

| Model | Accuracy | LUT | FF | DSP | BRAM | Latency | Fits VU13P |
|---|---|---|---|---|---|---|---|
| JEDI-Linear (Vivado post-route, their report) | **80.91%** | **97,822** | 48,231 | 0 | 0 | **46 ns** | yes (5.7%) |
| PHAT-JeT, best point that fits (ep140) | 78.64% | 1,188,245 | 2,815,168 | 0 | 0 | 152 ns | yes (69%) |

**Gap: −2.27 accuracy points at 12× the logic.**
The architecture's float ceiling on the same grid is 81.59%, so quantization
currently costs 2.95 points at this operating point.

## Full PHAT-JeT frontier

| Epoch | Accuracy | EBOPs | LUT (est) | FF (est) | Latency | Occupancy | Fits |
|---|---|---|---|---|---|---|---|
| 233 | 71.21% | 549,875 | 721,677 | 2,122,498 | 140 ns | 42% | yes |
| 208 | 76.23% | 574,114 | 786,240 | 2,242,798 | 142 ns | 46% | yes |
| 163 | 77.38% | 758,147 | 1,061,829 | 2,387,497 | 144 ns | 61% | yes |
| 140 | 78.64% | 926,777 | 1,188,245 | 2,815,168 | 152 ns | 69% | yes |
| 69 | 78.82% | 3,862,223 | 4,052,644 | 6,934,344 | 200 ns | 234% | no |
| 61 | 78.75% | 3,585,418 | 5,120,440 | 7,163,681 | 208 ns | 296% | no |
| 56 | 78.78% | 3,690,338 | 5,304,642 | 7,132,017 | 208 ns | 307% | no |
| 37 | 79.08% | 5,556,131 | 5,487,302 | 7,876,290 | 204 ns | 318% | no |
| 38 | 79.01% | 5,007,026 | 6,828,561 | 7,819,451 | 208 ns | 395% | no |
| 14 | 80.18% | 25,122,748 | 22,738,715 | 12,103,670 | 240 ns | 1316% | no |
| 13 | 80.48% | 29,184,954 | 32,485,236 | 12,763,639 | 246 ns | 1880% | no |

The frontier is **flat above ~1.2M LUT**: 5.7× more logic (1.19M → 6.83M) buys only
+0.36 accuracy points. Operating points beyond the device budget are therefore not
worth pursuing — the useful region is entirely at or below ~1.2M LUT.

## Corrections applied to an earlier version of this table

- latency_cutoff is 2 (JEDI-linear's setting), NOT 4.0 as an earlier version of this file said.
- Frontier points are now the ACCURACY-BEST checkpoint per EBOPs bucket. The earlier version spread indices over a count-sorted list, which skipped the whole device-fitting region and reported epoch 14 -- an unconverged snapshot at 555 LUT per data-dependent multiply vs 18-26 for converged points -- as the headline.
- epoch=166 excluded: filename says 78.01% but it re-evaluates to 71.94% (corrupt ckpt).
- LUT is INVARIANT to latency_cutoff (1,185,489 at lc=4,6,8,12,16); only FF and latency move.

## Verification

**RTL is bit-exact.** The deployable point's generated Verilog was compiled with
Verilator 5.050 and compared against Keras inference on 2000 validation jets:
0 mismatching rows, max abs diff 0.0.

Resource figures remain **da4ml estimates** (calibrated +1.9% on LUT against a known
post-route reference). No PHAT-JeT number may be described as post-place-and-route;
only the JEDI-Linear row is, and it is their own shipped Vivado report.
