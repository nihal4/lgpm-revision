"""A1c - external operating points on DDTI and TNUI-train (ERM, R1p, JOINT-g, JOINT-img).

usage: a1c_ddti_tnuitrain_operating_points.py EXT_PROBS THRESHOLDS_JSON DDTI_NPZ TNUITRAIN_NPZ OUTDIR
  THRESHOLDS_JSON = a1b_thresholds.json written by a1b (each model's TN5000-val raw threshold)
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ext_p, thr_p, ddti_p, tr_p, out = sys.argv[1:6]
out = Path(out); out.mkdir(parents=True, exist_ok=True)
E = json.load(open(ext_p)); THR = json.load(open(thr_p))
Y = {"DDTI": np.load(ddti_p, allow_pickle=True)["labels"].astype(int),
     "TNUI-train": np.load(tr_p, allow_pickle=True)["labels"].astype(int)}
rng = np.random.RandomState(42); rows = []
for c, y in Y.items():
    idxs = [rng.randint(0, len(y), len(y)) for _ in range(2000)]
    for m in ["ERM", "R1p", "JOINTg", "JOINTimg"]:
        p = np.asarray(E[c][m], float).ravel(); assert len(p) == len(y); yh = p >= THR[m]
        tp, fn = int((yh & (y == 1)).sum()), int((~yh & (y == 1)).sum())
        tn, fp = int((~yh & (y == 0)).sum()), int((yh & (y == 0)).sum())
        sp = [(~yh[i][y[i] == 0]).mean() for i in idxs]
        rows.append({"cohort": c, "model": m, "n": len(y), "benign": int((y == 0).sum()),
                     "auroc": round(roc_auc_score(y, p), 4), "sens": round(tp / (tp + fn), 4),
                     "spec": round(tn / (tn + fp), 4),
                     "spec_CI": f"{np.percentile(sp, 2.5):.3f}-{np.percentile(sp, 97.5):.3f}",
                     "TP": tp, "FN": fn, "TN": tn, "FP": fp,
                     "sens_minus_fpr": round(tp / (tp + fn) - fp / (fp + tn), 4)})
df = pd.DataFrame(rows); df.to_csv(out / "a1c_ddti_tnuitrain_operating_points.csv", index=False)
print(df.to_string(index=False))
