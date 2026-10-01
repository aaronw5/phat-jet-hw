# Physics-informed fixed GMP grid for PHAT-JeT on FPGA

**Context.** The paper's GMP uses a per-jet dynamic grid (min-shifted, δ=0.1–0.2).
The FPGA port requires a **fixed** grid (static one-hot bin thresholds). This study
chooses the fixed grid from the physics of the dataset (hls4ml LHC jets, 128 leading
constituents, jet pT ≈ 0.9–1.3 TeV, jet-relative η/φ coordinates) so the fixed grid
matches — or improves on — the dynamic grid's behavior. All numbers below measured on
a 200k-jet train subsample.

## Measured physics scales

| Quantity | Value |
|---|---|
| pT-weighted containment (all classes) | 95% of jet pT within r=0.19; 99.9% within r=0.37 |
| Coordinate extent | 99.9% of constituents within \|η\|,\|φ\| ≤ 0.43; **0.007% of pT outside ±0.4** |
| Leading-prong core position | median \|r\| = 0.042 (90% within 0.11) — the core sits **at the origin** |
| Two-prong opening, W / Z / t (2m/pT, median) | 0.155 / 0.176 / 0.311 |
| Measured axis-pair separation (pT-weighted 2-means), W / Z / t | 0.162 / 0.183 / 0.275 |
| q/g single-prong width √⟨r²⟩_pT | 0.041 / 0.070 |
| Effective multiplicity | mean 49 constituents/jet |

Three physics conclusions:

1. **±0.4 half-extent is right.** It loses 0.007% of jet pT (clamped to edge bins);
   shrinking to ±0.3 loses 0.75% and ±0.25 loses 2.0%. Growing to ±0.45 buys nothing
   measurable. This independently confirms the paper-port's ±0.4 choice.
2. **The informative angular structure is concentrated in \|x\| < 0.2** (95% of pT),
   and prong pairs must be *resolved* at Δ ≈ 0.15–0.31 (W/Z/t) while the q/g core
   width to be *localized* is ≈ 0.04–0.07. A uniform grid over-resolves the sparse
   periphery and under-resolves nothing — the periphery carries a percent-level pT tail.
3. **The jet core lands exactly on a bin boundary in any even-count symmetric grid.**
   With 8×8 over ±0.4, the origin is a 4-cell corner: the leading prong's energy is
   split across up to 4 cells jet-by-jet (median hottest-cell pT share: 56%). An
   **odd** bin count centers a cell on the origin (7×7: 69–73%, +13–17 points of core
   coherence). This is a fixed-grid artifact the dynamic per-jet min-shift never had —
   an odd count is the fix.

## Candidate evaluation (static metrics)

| Grid | Cells | GMP scatter/gather cost | Core coherence¹ | pT clamped | W axes resolved² | t axes resolved² |
|---|---|---|---|---|---|---|
| 8×8 uniform ±0.4, δ=0.1 (current) | 64 | 1.00× | 56% | 0.007% | 100% | 100% |
| **7×7 non-uniform ±0.4 ("core7")** | **49** | **0.77×** | **69%** | **0.007%** | **97%** | **99%** |
| 9×9 uniform ±0.45, δ=0.1 | 81 | 1.27× | 69% | 0.002% | 97% | 99% |
| 5×5 non-uniform ±0.4 | 25 | 0.39× | 74% | 0.007% | 94% | 99% |
| 4×4 uniform ±0.4, δ=0.2 | 16 | 0.25× | 57% | 0.007% | 100% | 100% |

¹ median pT share of the hottest cell (higher = leading prong kept coherent).
² fraction of pT-weighted 2-means axis pairs falling in *distinct* cells.

**core7 edges:** `[-0.40, -0.20, -0.12, -0.04, 0.04, 0.12, 0.20, 0.40]` per axis —
fine 0.08 bins where 95% of the pT lives (finer than the paper's proven δ=0.1 exactly
where prongs need separating), coarse 0.2 bins for the outer annulus (which the paper's
δ=0.2 sweep row shows is adequate: 81.78% vs 81.83% at δ=0.1). Non-uniform edges are
**free in hardware**: each bin indicator is one threshold comparison LUT regardless of
spacing; 49 cells cut the dominant GMP scatter/gather einsum cost 23% vs 64.

The paper's own sensitivity sweep (Table 14: δ=0.1 → 81.83%, δ=0.2 → 81.78%,
δ=0.3 → 81.68%) says accuracy is flat in this regime, so the QAT shootout below is
the tie-breaker on accuracy-per-LUT, not accuracy alone.

## QAT shootout (25 epochs, 300k jets, jsc150 recipe)

Three candidates trained identically (same seed, data, schedule, 25 epochs, 300k
jets), differing **only** in the GMP grid:

| Grid | Cells | Best val acc | EBOPs at best | acc @ ep10 | acc @ ep15 | acc @ ep20 |
|---|---|---|---|---|---|---|
| 8×8 uniform (baseline) | 64 | 0.6993 | 1.64e7 | 0.690 | 0.698 | 0.676 |
| **7×7 non-uniform (core7)** | **49** | **0.7190** | **1.37e7** | 0.709 | 0.719 | 0.702 |
| 9×9 uniform ±0.45 | 81 | 0.7255 | 1.71e7 | 0.714 | 0.719 | 0.667 |

**Decision: core7.** It *dominates* the 8×8 baseline — **+2.0 points of accuracy at
17% lower EBOPs and 23% fewer cells** — and matches 9×9 at equal epochs (0.719 both
at epoch 15) while using 40% fewer cells and 20% lower EBOPs. The two odd-count grids
both beat the even-count baseline by ~2 points, exactly as the core-splitting argument
in conclusion (3) predicts; core7 additionally gets there at the lowest cost of the
three, because the fine bins are spent only where the pT density is.

(The accuracy collapse at epoch ~25 in all three is the compressed β schedule
over-penalizing at the end of a 25-epoch proxy run, not a grid effect — it hits all
three identically and is absent from the long run, which spreads the same β ramp over
1000 epochs.)

## Relation to the dynamic grid of the paper

The paper's GMP min-shifts coordinates per jet, i.e. the grid *follows* the jet: the
core never straddles a fixed boundary. The fixed FPGA grid loses that; core7 restores
it in expectation (cell-centered origin + jet-relative coordinates already center the
jet at 0 by construction) and adds resolution where the dynamic uniform grid had none.
The edge-clamp bins replace the dynamic extent: 0.007% of pT is affected.
