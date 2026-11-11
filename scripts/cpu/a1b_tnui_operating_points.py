"""A1b - external operating points (sens AND spec) on TNUI-2021 test, val and the
pre-declared pool, for ERM, R1p (3c head), JOINT-g and JOINT-img.

Probabilities: phase4_external_probs.json (row order = locked 3c/3d banks, whose labels we
read here). Thresholds: each model's own TN5000-val raw probs, max sens at spec >= 0.90
(JOINT thresholds read from phase4_joint_probs_tn5000.json). Nothing refitted.

usage: a1b_tnui_operating_points.py EXT_PROBS JOINT_JSON ERM_JSON HEADS3C VAL_NPZ TNUI_TEST_NPZ TNUI_VAL_NPZ OUTDIR
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

ext_p, joint_p, erm_p, heads_p, val_p, tt_p, tv_p, out = sys.argv[1:9]
out = Path(out); out.mkdir(parents=True, exist_ok=True)
E = json.load(open(ext_p)); J = json.load(open(joint_p)); R = json.load(open(erm_p)); H = json.load(open(heads_p))
y_val = np.load(val_p, allow_pickle=True)["labels"].astype(int)
Y = {"TNUI-test": np.load(tt_p, allow_pickle=True)["labels"].astype(int),
     "TNUI-val": np.load(tv_p, allow_pickle=True)["labels"].astype(int)}


def spec90(y, p):
    fpr, tpr, thr = roc_curve(y, p); m = (1 - fpr) >= 0.90
    i = np.where(m)[0][np.argmax(tpr[m])] if m.any() else int(np.argmax(1 - fpr))
    return float(thr[i])


THR = {"JOINTg": J["thr"]["JOINTg"], "JOINTimg": J["thr"]["JOINTimg"],
       "ERM": spec90(y_val, np.asarray(R["probs"]["val"], float).ravel()),
       "R1p": spec90(y_val, np.asarray(H["R1p_lesion"]["probs"]["val"], float).ravel())}
print("thresholds:", {k: round(v, 4) for k, v in THR.items()})
P = {c: {m: np.asarray(E[c][m], float).ravel() for m in THR} for c in Y}
Y["TNUI pool"] = np.concatenate([Y["TNUI-test"], Y["TNUI-val"]])
P["TNUI pool"] = {m: np.concatenate([P["TNUI-test"][m], P["TNUI-val"][m]]) for m in THR}
rng = np.random.RandomState(42)
rows = []
for c in ["TNUI-test", "TNUI-val", "TNUI pool"]:
    y = Y[c]; idxs = [rng.randint(0, len(y), len(y)) for _ in range(2000)]
    for m in ["ERM", "R1p", "JOINTg", "JOINTimg"]:
        p = P[c][m]; yh = p >= THR[m]
        tp, fn = int((yh & (y == 1)).sum()), int((~yh & (y == 1)).sum())
        tn, fp = int((~yh & (y == 0)).sum()), int((yh & (y == 0)).sum())
        sp = [(~yh[i][y[i] == 0]).mean() for i in idxs if (y[i] == 0).any()]
        rows.append({"cohort": c, "model": m, "n": len(y), "benign": int((y == 0).sum()),
                     "auroc": round(roc_auc_score(y, p), 4), "sens": round(tp / (tp + fn), 4),
                     "spec": round(tn / (tn + fp), 4),
                     "spec_CI": f"{np.percentile(sp, 2.5):.3f}-{np.percentile(sp, 97.5):.3f}",
                     "TP": tp, "FN": fn, "TN": tn, "FP": fp,
                     "sens_minus_fpr": round(tp / (tp + fn) - fp / (fp + tn), 4)})
df = pd.DataFrame(rows); df.to_csv(out / "a1b_tnui_operating_points.csv", index=False)
print(df.to_string(index=False))
np.save(out / "y_tnui_pool.npy", Y["TNUI pool"]); np.save(out / "p_jointg_tnui_pool.npy", P["TNUI pool"]["JOINTg"])
np.save(out / "p_r1p_tnui_pool.npy", P["TNUI pool"]["R1p"])
json.dump(THR, open(out / "a1b_thresholds.json", "w"), indent=1)
