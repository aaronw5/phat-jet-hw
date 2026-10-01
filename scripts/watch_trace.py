"""Watch pareto_* dirs; trace each NEW frontier checkpoint -> accuracy/LUT/latency.

Why a watcher: ParetoFront deletes dominated checkpoints, so a manually-launched
trace can lose its file mid-run (observed). We copy to traced/ first, then trace.
Appends one row per checkpoint to frontier_qat.json. Idempotent on (tag, epoch).
Stop with: touch STOP_WATCH
"""
import os, sys, json, time, shutil, subprocess, re
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root)
OUT = "frontier_qat.json"
CFG = {"qat_n64_w1": (64, 8), "qat_n64_w2": (64, 8), "qat_n64_w3": (64, 8)}
seen = set()
if os.path.exists(OUT):
    for r in json.load(open(OUT)):
        seen.add((r["tag"], r["epoch"]))
os.makedirs("traced", exist_ok=True)
while not os.path.exists("STOP_WATCH"):
    for tag, (N, PS) in CFG.items():
        d = f"pareto_{tag}"
        if not os.path.isdir(d):
            continue
        cks = sorted(os.listdir(d))
        if not cks:
            continue
        # trace the newest (most-compressed) frontier point only
        newest = max(cks, key=lambda f: os.path.getmtime(os.path.join(d, f)))
        ep = int(re.search(r"epoch=(\d+)", newest).group(1))
        if (tag, ep) in seen:
            continue
        seen.add((tag, ep))
        snap = f"traced/{tag}_ep{ep}.keras"
        try:
            shutil.copy(os.path.join(d, newest), snap)
        except FileNotFoundError:
            continue
        env = dict(os.environ, OMP_NUM_THREADS="2", KERAS_BACKEND="jax", GRID_NAME="core7")
        p = subprocess.run([sys.executable, "scripts/trace_n.py", snap, str(N), str(PS)],
                           capture_output=True, text=True, env=env)
        m = re.search(r"acc=\s*([\d.]+)%\s*LUT=\s*([\d,]+)\s*FF=\s*([\d,]+)\s*stages=\s*(\d+)\s*lat=\s*([\d.]+)ns",
                      p.stdout)
        if not m:
            print(f"[{tag} ep{ep}] trace FAILED: {p.stdout[-200:]} {p.stderr[-300:]}", flush=True)
            continue
        row = dict(tag=tag, N=N, patch_size=PS, epoch=ep,
                   val_acc=float(m.group(1)) / 100, lut=int(m.group(2).replace(",", "")),
                   ff=int(m.group(3).replace(",", "")), stages=int(m.group(4)),
                   latency_ns=float(m.group(5)), ckpt=newest)
        rows = json.load(open(OUT)) if os.path.exists(OUT) else []
        rows.append(row); json.dump(rows, open(OUT, "w"), indent=1)
        print(f"[{tag} ep{ep}] acc={row['val_acc']*100:.2f}% LUT={row['lut']:,} "
              f"lat={row['latency_ns']:.0f}ns", flush=True)
    time.sleep(240)
print("STOP_WATCH -> exiting", flush=True)
