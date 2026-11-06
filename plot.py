import csv
import numpy as np
import matplotlib.pyplot as plt

rows = []
with open("results/rep_code.csv") as f:
    for r in csv.DictReader(f):
        rows.append((int(r["d"]), float(r["p"]), int(r["shots"]), int(r["errors"])))

for d in sorted({r[0] for r in rows}):
    pts = sorted((p, n, e) for dd, p, n, e in rows if dd == d)
    p = np.array([x[0] for x in pts])
    n = np.array([x[1] for x in pts])
    e = np.array([x[2] for x in pts])
    pl = e / n
    err = np.sqrt(pl * (1 - pl) / n)  # binomial error bar
    plt.errorbar(p, pl, yerr=err, marker="o", capsize=3, label=f"d = {d}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("physical error rate p")
plt.ylabel("logical error rate (per shot)")
plt.title("Repetition code memory, MWPM decoding")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.savefig("results/rep_code.png", dpi=150, bbox_inches="tight")
plt.show()