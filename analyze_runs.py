# R1-R4 comparison for the arXiv note. Expects each job dir to contain
# sscurve.csv (from post_mp.py). Produces sscurve_compare.png + a text
# summary of flow-stress deltas at 2/5/10% strain.
#   python analyze_runs.py
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [
    ("oxford_mp/sscurve_R1_frozen.csv", "R1 midpoint, exact $d_b$ (frozen)", "k-"),
    ("oxford_mp_rt/sscurve.csv", "R2 midpoint, ray-traced $d_b(t)$", "b-"),
    ("oxford_mp_st/sscurve.csv", "R3 midpoint, staircase $d_b$", "r--"),
    ("oxford_vox/sscurve.csv", "R4 voxel, staircase $d_b$", "g:"),
]


def load(path):
    e, s = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            e.append(float(row["strain"]))
            s.append(float(row["avgS11"]))
    return np.array(e), np.array(s)


fig, ax = plt.subplots(figsize=(5, 3.6))
data = {}
for path, label, style in RUNS:
    try:
        e, s = load(path)
    except OSError:
        print(f"skip {path} (missing)")
        continue
    data[label] = (e, s)
    ax.plot(e * 100, s, style, lw=1.4, label=label)
ax.set_xlabel("tensile strain (%)")
ax.set_ylabel(r"$\langle\sigma_{11}\rangle$ (MPa)")
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig("sscurve_compare.png", dpi=300)
print("wrote sscurve_compare.png")

if len(data) > 1:
    ref = "R1 midpoint, exact $d_b$ (frozen)"
    er, sr = data[ref]
    for lab, (e, s) in data.items():
        if lab == ref:
            continue
        for target in (0.02, 0.05, 0.10):
            if er.max() >= target and e.max() >= target:
                a = np.interp(target, er, sr)
                b = np.interp(target, e, s)
                print(f"{lab} vs R1 @ {target*100:.0f}%: "
                      f"{b:.2f} vs {a:.2f} MPa ({(b-a)/a*100:+.1f}%)")
