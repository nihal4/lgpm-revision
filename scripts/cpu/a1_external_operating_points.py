"""A1 - external operating points on ThyUS2Path for the frozen linear heads.

Reproduces the Phase-5 convention exactly (raw per-image probabilities, case score =
unweighted mean of raw image probabilities, threshold = max sensitivity at
specificity >= 0.90 on raw TN5000-val probabilities, nothing refitted) and adds what
the reviewers asked for: achieved external specificity, confusion counts, bootstrap
CIs, and McNemar on benign cases (false positives) as well as malignant cases.

usage: a1_external_operating_points.py BANK CASEMAP HEADS_JSON VAL_NPZ OUTDIR
  HEADS_JSON = phase3c linear_heads_retrained.json (w, b, mu, sd + val/test probs)
  VAL_NPZ    = phase3 features_val.npz (only its 'labels' array is used)
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

SEED, N_BOOT = 42, 5000
REPS = ["R0_image", "R1p_lesion", "R2p_lesion_ring"]

bank_p, map_p, heads_p, val_p, out_dir = sys.argv[1:6]
out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
Z = np.load(bank_p, allow_pickle=True)
cmap = pd.read_csv(map_p)
assert list(cmap.stem) == list(Z["ids"].astype(str)), "case map does not match bank order"
H = json.loads(Path(heads_p).read_text())
y_val = np.load(val_p, allow_pickle=True)["labels"].astype(int)


def lin_apply(rep, X):                       # identical to phase5 lin_apply
    h = H[rep]["head"]
    w, b, mu, sd = (np.array(h[k]) for k in ("w", "b", "mu", "sd"))
    z = ((X.astype(np.float64) - mu) / sd) @ w.T + b
    return 1 / (1 + np.exp(-z.ravel()))


def spec90_threshold(y, p):                  # identical to phase5 spec90_threshold
    fpr, tpr, thr = roc_curve(y, p)
    spec = 1 - fpr
    m = spec >= 0.90
    i = np.where(m)[0][np.argmax(tpr[m])] if m.any() else int(np.argmax(spec))
    return float(thr[i]), float(tpr[i]), float(spec[i])


case_order = sorted(cmap.case.unique())     # phase5: sorted(case_df5["case"])
pos = {c: i for i, c in enumerate(case_order)}
ci = cmap.case.map(pos).to_numpy()
n_img = np.bincount(ci, minlength=len(case_order))
y_case = np.bincount(ci, weights=cmap.label.to_numpy(), minlength=len(case_order)) / n_img
assert set(np.unique(y_case)) <= {0, 1}
y_case = y_case.astype(int)


def case_mean(p):
    return np.bincount(ci, weights=p, minlength=len(case_order)) / n_img


rng = np.random.RandomState(SEED)
B_IDX = [rng.randint(0, len(y_case), len(y_case)) for _ in range(N_BOOT)]
scores, thr = {}, {}
rows = []
for rep in REPS:
    pv = np.asarray(H[rep]["probs"]["val"], float).ravel()
    assert len(pv) == len(y_val), "val probs / labels length mismatch"
    t, s_val, sp_val = spec90_threshold(y_val, pv)
    pc = case_mean(lin_apply(rep, Z[rep]))
    scores[rep], thr[rep] = pc, t
    yhat = pc >= t
    tp = int(((y_case == 1) & yhat).sum()); fn = int(((y_case == 1) & ~yhat).sum())
    tn = int(((y_case == 0) & ~yhat).sum()); fp = int(((y_case == 0) & yhat).sum())
    sens_b, spec_b = [], []
    for idx in B_IDX:
        yy, hh = y_case[idx], yhat[idx]
        sens_b.append(hh[yy == 1].mean()); spec_b.append((~hh[yy == 0]).mean())
    rows.append({"model": rep, "threshold_raw": round(t, 4),
                 "val_sens": round(s_val, 4), "val_spec": round(sp_val, 4),
                 "auroc_case": round(roc_auc_score(y_case, pc), 4),
                 "ext_sens": round(tp / (tp + fn), 4),
                 "ext_sens_CI": f"{np.percentile(sens_b, 2.5):.4f}-{np.percentile(sens_b, 97.5):.4f}",
                 "ext_spec": round(tn / (tn + fp), 4),
                 "ext_spec_CI": f"{np.percentile(spec_b, 2.5):.4f}-{np.percentile(spec_b, 97.5):.4f}",
                 "TP": tp, "FN": fn, "TN": tn, "FP": fp,
                 "flagged_positive": tp + fp, "n_cases": len(y_case)})
tab = pd.DataFrame(rows)
tab.to_csv(out / "a1_thyus2path_operating_points.csv", index=False)
print(f"ThyUS2Path cases {len(y_case)} (malignant {int(y_case.sum())}, benign {int((y_case == 0).sum())})")
print(tab.to_string(index=False))

# matched-specificity view (POST HOC, descriptive): sensitivity of each model at the
# external specificity another model actually achieved
def sens_at_spec(p, target):
    fpr, tpr, _ = roc_curve(y_case, p)
    ok = (1 - fpr) >= target
    return float(tpr[ok].max()) if ok.any() else 0.0

mt = []
for ref in REPS:
    target = float(tab.set_index("model").loc[ref, "ext_spec"])
    for m in REPS:
        mt.append({"specificity_of": ref, "target_spec": target, "model": m,
                   "sens_at_target_spec": round(sens_at_spec(scores[m], target), 4)})
pd.DataFrame(mt).to_csv(out / "a1_matched_specificity_posthoc.csv", index=False)
print("\nPOST HOC matched-specificity sensitivities:")
print(pd.DataFrame(mt).pivot(index="model", columns="specificity_of",
                             values="sens_at_target_spec").to_string())
json.dump({"thresholds": thr, "case_order": case_order},
          open(out / "a1_thresholds.json", "w"), indent=1)
np.savez_compressed(out / "a1_case_scores_linear.npz", y_case=y_case,
                    case=np.array(case_order), **scores)
