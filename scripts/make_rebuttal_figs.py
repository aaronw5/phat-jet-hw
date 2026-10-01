"""Rebuttal figure + table: PHAT-JeT vs JEDI-Linear on their own N axis."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = json.load(open(f"{root}/jedi_published.json"))
ours = json.load(open(f"{root}/nsweep_results.json")) if os.path.exists(f"{root}/nsweep_results.json") else []
ours = sorted([r for r in ours if r.get("val_acc", 0) > 0.4], key=lambda r: r["N"])

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
C_OURS, C_JEDI, C_MLPM, C_PRIOR = "#c1272d", "#0b5394", "#7a9a01", "#888888"

def sc(ax, xs, ys, **kw):
    ax.plot(xs, ys, marker=kw.pop("marker", "o"), lw=kw.pop("lw", 2.0),
            ms=kw.pop("ms", 7), **kw)

# --- panel 1: accuracy vs N ---
ax = axes[0]
pi, pt = J["perm_invariant_f3"], J["pt_sorted_f3"]
sc(ax, [r["N"] for r in pi], [r["acc"] for r in pi], color=C_JEDI, label="JEDI-linear (perm-inv)")
sc(ax, [r["N"] for r in pt], [r["acc"] for r in pt], color=C_JEDI, ls="--", marker="s", label="JEDI-linear (pT-sorted)")
mm = J["mlpmixer_f3_MLST25"]
sc(ax, [r["N"] for r in mm], [r["acc"] for r in mm], color=C_MLPM, ls=":", marker="^", label="MLP-Mixer (MLST'25)")
if ours:
    sc(ax, [r["N"] for r in ours], [r["val_acc"] * 100 for r in ours], color=C_OURS, marker="D", ms=9,
       label="PHAT-JeT (ours, quantized)", zorder=5)
ax.set_xscale("log", base=2); ax.set_xticks([8, 16, 32, 64, 128]); ax.set_xticklabels([8, 16, 32, 64, 128])
ax.set_xlabel("particles per jet, $N$"); ax.set_ylabel("accuracy [%]")
ax.set_title("(a) Accuracy vs input size, 3 features", loc="left", fontweight="bold")
ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")

# --- panel 2: accuracy vs LUT ---
ax = axes[1]
ax.scatter([r["LUT_k"] * 1e3 for r in pt], [r["acc"] for r in pt], color=C_JEDI, marker="s", s=60,
           label="JEDI-linear (pT-sorted)")
for r in pt:
    ax.annotate(f"N={r['N']}", (r["LUT_k"] * 1e3, r["acc"]), fontsize=7,
                textcoords="offset points", xytext=(4, -9), color=C_JEDI)
ax.scatter([r["LUT_k"] * 1e3 for r in J["prior_gnn_f3_MLST24"]],
           [r["acc"] for r in J["prior_gnn_f3_MLST24"]], color=C_PRIOR, marker="x", s=45,
           label="prior GNN/DS (MLST'24)")
if ours:
    ax.scatter([r["lut_est"] for r in ours], [r["val_acc"] * 100 for r in ours],
               color=C_OURS, marker="D", s=85, zorder=5, label="PHAT-JeT (ours)")
    for r in ours:
        ax.annotate(f"N={r['N']}", (r["lut_est"], r["val_acc"] * 100), fontsize=7,
                    textcoords="offset points", xytext=(4, 5), color=C_OURS)
ax.axvline(1_728_000, color="k", ls="-.", lw=1.2)
ax.text(1_728_000 * .93, ax.get_ylim()[0] + 1, "VU13P LUT budget", rotation=90, fontsize=7, ha="right")
ax.set_xscale("log"); ax.set_xlabel("LUT (log scale)"); ax.set_ylabel("accuracy [%]")
ax.set_title("(b) Accuracy vs logic cost", loc="left", fontweight="bold")
ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")

# --- panel 3: latency, incl. prior art ---
ax = axes[2]
names, lats, cols = [], [], []
for r in J["prior_gnn_f16"]:
    names.append(f"{r['model']} N={r['N']}"); lats.append(r["latency_ns"]); cols.append(C_PRIOR)
if ours:
    b = max(ours, key=lambda r: r["val_acc"])
    names.append(f"PHAT-JeT N={b['N']} (ours)"); lats.append(b["latency_ns"]); cols.append(C_OURS)
names.append("JEDI-linear N=128"); lats.append(82); cols.append(C_JEDI)
o = np.argsort(lats)[::-1]
ax.barh([names[i] for i in o], [lats[i] for i in o], color=[cols[i] for i in o])
for y, i in enumerate(o):
    ax.text(lats[i] * 1.05, y, f"{lats[i]:.0f} ns", va="center", fontsize=7.5)
ax.axvline(10_000, color="k", ls="-.", lw=1.2)
ax.text(10_000 * .9, 0.3, "LHC L1 trigger budget ~10 $\\mu$s", rotation=90, fontsize=7, ha="right")
ax.set_xscale("log"); ax.set_xlabel("latency [ns] (log scale)")
ax.set_title("(c) Latency vs published FPGA jet taggers", loc="left", fontweight="bold")
ax.grid(alpha=.3, axis="x"); ax.tick_params(labelsize=8)

fig.tight_layout()
fig.savefig(f"{root}/fig_rebuttal_nsweep.png", dpi=200, bbox_inches="tight")
print("wrote fig_rebuttal_nsweep.png")

# --- markdown table ---
L = ["| Model | N | feats | Acc [%] | Latency [ns] | LUT | FF | DSP | BRAM | II |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for r in pt:
    L.append(f"| JEDI-linear (pT-sorted, published) | {r['N']} | 3 | {r['acc']:.1f} | {r['latency_ns']} | "
             f"{r['LUT_k']*1000:,} | {r['FF_k']*1000:,} | 0 | 0 | 1 |")
for r in ours:
    L.append(f"| **PHAT-JeT (ours, est.)** | {r['N']} | 3 | {r['val_acc']*100:.2f} | {r['latency_ns']:.0f} | "
             f"{r['lut_est']:,.0f} | {r['ff_est']:,} | 0 | 0 | 1 |")
open(f"{root}/table_rebuttal.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
