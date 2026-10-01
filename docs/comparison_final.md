# PHAT-JeT hardware comparison (rebuttal)

All PHAT-JeT rows are **da4ml logic estimates**, not post-place-and-route. Calibration: tracing JEDI-linear's own released N=128 model through the identical instrument gives 99,690 LUT against their published post-route 97,822 — an overestimate of 1.9%. Published rows are post-P&R from their respective papers.

Envelope (from JEDI-linear Sec. III): CTL2 = 30 VU13P FPGAs, sub-100 ns latency, sub-10 ns initiation interval. VU13P capacity = 1,728,000 LUT.

| Design | Acc (%) | LUT | % VU13P | Latency @300MHz (ns) | DSP | BRAM | II | Source | Note |
|---|---|---|---|---|---|---|---|---|---|
| PHAT-JeT N=64 (L1T-envelope point) | 74.58 | 167,304 | 9.7 | 96.6 | 0 | 0 | 1 | da4ml estimate | clears sub-100 ns AND fits device |
| PHAT-JeT N=64 (max acc, LUT budget only) | 78.42 | 1,221,621 | 70.7 | 133.2 | 0 | 0 | 1 | da4ml estimate | exceeds 100 ns bound |
| PHAT-JeT N=128 (max acc in LUT budget) | 78.6 | 902,337 | 52.2 | 236.4 | 0 | 0 | 1 | da4ml estimate | 52% of device |
| PHAT-JeT N=128 (bit-exact verified) | 78.64 | 1,188,245 | 68.8 | 253.1 | 0 | 0 | 1 | da4ml estimate | RTL bit-exact vs Keras, 2000 samples |
| JEDI-linear N=16 (pT-sorted) | 71.9 | 44,000 | 2.5 | 54 | 0 | 0 | 1 | published post-P&R |  |
| JEDI-linear N=32 (pT-sorted) | 78.0 | 45,000 | 2.6 | 63 | 0 | 0 | 1 | published post-P&R |  |
| JEDI-linear N=64 (pT-sorted) | 80.9 | 71,000 | 4.1 | 61 | 0 | 0 | 1 | published post-P&R |  |
| JEDI-linear N=128 (pT-sorted) | 80.9 | 98,000 | 5.7 | 82 | 0 | 0 | 1 | published post-P&R |  |
| JEDI-linear N=8 (perm-inv) | 66.5 | 136,000 | 7.9 | 79 | 0 | 0 | 1 | published post-P&R |  |
| JEDI-linear N=16 (perm-inv) | 73.6 | 136,000 | 7.9 | 75 | 0 | 0 | 1 | published post-P&R |  |
| JEDI-linear N=32 (perm-inv) | 79.0 | 136,000 | 7.9 | 80 | 0 | 0 | 1 | published post-P&R |  |
| JEDI-linear N=64 (perm-inv) | 81.8 | 164,000 | 9.5 | 78 | 0 | 0 | 1 | published post-P&R |  |
| JEDI-linear N=128 (perm-inv) | 81.6 | 296,000 | 17.1 | 138 | 0 | 0 | 1 | published post-P&R |  |
| MLP-Mixer N=16 | 71.7 | 75,000 | 4.3 | 68 | 0 | n/r | n/r | published post-P&R |  |
| MLP-Mixer N=32 | 78.0 | 63,000 | 3.6 | 62 | 0 | n/r | n/r | published post-P&R |  |
| MLP-Mixer N=64 | 79.7 | 159,000 | 9.2 | 72 | 0 | n/r | n/r | published post-P&R |  |
| MLP-Mixer N=128 | 79.8 | 83,000 | 4.8 | 72 | 0 | n/r | n/r | published post-P&R |  |
| DS N=8 (MLST'24) | 64.0 | 626,000 | 36.2 | 95 | 386 | n/r | n/r | published post-P&R |  |
| DS N=16 (MLST'24) | 69.4 | 555,000 | 32.1 | 115 | 747 | n/r | n/r | published post-P&R |  |
| DS N=32 (MLST'24) | 75.9 | 434,000 | 25.1 | 130 | 903 | n/r | n/r | published post-P&R |  |
| GNN N=8 (MLST'24) | 64.9 | 2,120,000 | 122.7 | 160 | 472 | n/r | n/r | published post-P&R |  |
| GNN N=16 (MLST'24) | 70.8 | 5,362,000 | 310.3 | 180 | 1388 | n/r | n/r | published post-P&R |  |
| GNN N=32 (MLST'24) | 75.8 | 2,120,000 | 122.7 | 205 | 1162 | n/r | n/r | published post-P&R |  |
