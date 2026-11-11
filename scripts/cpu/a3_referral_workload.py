"""A3 - referral workload for selective prediction, with a random-referral baseline.

Reproduces Phase-4 coverage_curve exactly (rank by calibrated margin |sigma(logit(p)/T) - 0.5|,
stable sort, k = round(c*n), decisions at the frozen RAW threshold), then adds per-coverage
counts (retained/referred malignant and benign, misses kept/removed, specificity among
retained) and the misses removed by 2,000 random referrals of the same size.

usage: a3_referral_workload.py NAME LABELS.npy PROBS.npy THR T OUT.csv
"""
import sys
import numpy as np, pandas as pd

name, y_p, p_p, thr, T, out_p = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
y = np.load(y_p).astype(int).ravel(); p = np.load(p_p).astype(float).ravel()
assert len(y) == len(p)
z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
conf = np.abs(1 / (1 + np.exp(-z / T)) - 0.5)
order = np.argsort(-conf, kind="stable")
pred = p >= thr
miss = (y == 1) & ~pred
fp_all = (y == 0) & pred
n, M = len(y), int(miss.sum())
rng = np.random.RandomState(42)
rows = []
for c in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
    k = max(1, int(round(c * n)))
    keep = np.zeros(n, bool); keep[order[:k]] = True
    ref = ~keep
    r_rand = []
    for _ in range(2000):
        kr = np.zeros(n, bool); kr[rng.permutation(n)[:k]] = True
        r_rand.append(int((miss & ~kr).sum()))
    yr = y[keep]
    rows.append({
        "cohort": name, "coverage": c, "retained": k, "referred": n - k,
        "retained_malignant": int((keep & (y == 1)).sum()),
        "retained_benign": int((keep & (y == 0)).sum()),
        "referred_malignant": int((ref & (y == 1)).sum()),
        "referred_benign": int((ref & (y == 0)).sum()),
        "misses_total": M,
        "misses_remaining": int((miss & keep).sum()),
        "misses_referred": int((miss & ref).sum()),
        "misses_referred_pct": round(100 * (miss & ref).sum() / max(M, 1), 1),
        "random_referral_misses_removed_mean": round(float(np.mean(r_rand)), 1),
        "random_referral_95pct": f"{np.percentile(r_rand, 2.5):.0f}-{np.percentile(r_rand, 97.5):.0f}",
        "sens_retained": round(float(pred[keep & (y == 1)].mean()), 4),
        "spec_retained": round(float((~pred[keep & (y == 0)]).mean()), 4) if (keep & (y == 0)).any() else np.nan,
        "false_pos_remaining": int((fp_all & keep).sum()),
        "malignant_share_retained": round(float((keep & (y == 1)).sum() / max((y == 1).sum(), 1)), 4),
    })
df = pd.DataFrame(rows)
df.to_csv(out_p, index=False)
print(f"== {name}: n={n}, malignant={int(y.sum())}, benign={int((y == 0).sum())}, thr={thr:.4f}, T={T:.4f}")
print(df.drop(columns=["cohort"]).to_string(index=False))
