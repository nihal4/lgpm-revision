"""X2 - head-replicate variability (answers R2.3: which R1p head, and does it matter?).

The Phase-3 head weights were not archived; external phases use one Phase-3c retrain.
Here we retrain N heads per representation with the exact Phase-3 recipe (standardise on
train, Linear, AdamW lr 3e-3 wd 1e-2, BCE pos_weight, batch 512, <=300 epochs, early stop
on val AUROC, patience 25), differing only in the torch seed, and apply each unchanged to
ThyUS2Path (case mean of raw probs, threshold = max sens at val spec >= 0.90).

usage: x2_head_replicates.py FEAT_DIR BANK CASEMAP N_SEEDS OUTDIR
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

feat_dir, bank_p, map_p, n_seeds, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)
LIN_EPOCHS, LIN_PATIENCE, LIN_LR, LIN_WD = 300, 25, 3e-3, 1e-2
F = {s: np.load(feat_dir / f"features_{s}.npz", allow_pickle=True) for s in ("train", "val", "test")}
Y = {s: F[s]["labels"].astype(np.float32) for s in F}
Z = np.load(bank_p, allow_pickle=True)
cmap = pd.read_csv(map_p)
assert list(cmap.stem) == list(Z["ids"].astype(str))
case_order = sorted(cmap.case.unique()); pos = {c: i for i, c in enumerate(case_order)}
ci = cmap.case.map(pos).to_numpy(); nimg = np.bincount(ci)
y_case = (np.bincount(ci, weights=cmap.label.to_numpy()) / nimg).astype(int)


def spec90(y, p):
    fpr, tpr, thr = roc_curve(y, p); m = (1 - fpr) >= 0.90
    i = np.where(m)[0][np.argmax(tpr[m])] if m.any() else int(np.argmax(1 - fpr))
    return float(thr[i])


def train_head(rep, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    xtr, xva, xte = (torch.from_numpy(F[s][rep].astype(np.float32)) for s in ("train", "val", "test"))
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp(min=1e-6)
    xtr, xva, xte = (xtr - mu) / sd, (xva - mu) / sd, (xte - mu) / sd
    ytr = torch.from_numpy(Y["train"])
    lin = nn.Linear(xtr.shape[1], 1)
    pos_w = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)])
    opt = torch.optim.AdamW(lin.parameters(), lr=LIN_LR, weight_decay=LIN_WD)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best, wait, bw, n = -1, 0, None, len(xtr)
    for ep in range(1, LIN_EPOCHS + 1):
        perm = torch.randperm(n); lin.train()
        for i0 in range(0, n, 512):
            idx = perm[i0:i0 + 512]
            loss = bce(lin(xtr[idx]).squeeze(1), ytr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        lin.eval()
        with torch.no_grad():
            va = roc_auc_score(Y["val"], torch.sigmoid(lin(xva).squeeze(1)).numpy())
        if va > best + 1e-4:
            best, wait = va, 0; bw = {k: v.clone() for k, v in lin.state_dict().items()}
        else:
            wait += 1
            if wait >= LIN_PATIENCE:
                break
    lin.load_state_dict(bw); lin.eval()
    with torch.no_grad():
        pv = torch.sigmoid(lin(xva).squeeze(1)).numpy(); pt = torch.sigmoid(lin(xte).squeeze(1)).numpy()
        xe = (torch.from_numpy(Z[rep].astype(np.float32)) - mu) / sd
        pe = torch.sigmoid(lin(xe).squeeze(1)).numpy().astype(np.float64)
    return pv, pt, pe, ep


rows = []
for rep in ("R1p_lesion", "R0_image"):
    for seed in range(n_seeds):
        pv, pt, pe, ep = train_head(rep, seed)
        t = spec90(Y["val"], pv)
        pc = np.bincount(ci, weights=pe) / nimg
        yh = pc >= t
        sens = yh[y_case == 1].mean(); fpr = yh[y_case == 0].mean()
        rows.append({"rep": rep, "seed": seed, "epochs": ep,
                     "val_auroc": roc_auc_score(Y["val"], pv), "tn5000_test_auroc": roc_auc_score(Y["test"], pt),
                     "test_sens_at_thr": float((pt[Y["test"] == 1] >= t).mean()),
                     "thy_case_auroc": roc_auc_score(y_case, pc),
                     "thy_sens": sens, "thy_spec": 1 - fpr, "thy_sens_minus_fpr": sens - fpr,
                     "thy_flagged": int(yh.sum())})
        r = rows[-1]
        print(f"{rep:10s} seed {seed:2d}: test AUROC {r['tn5000_test_auroc']:.4f} | ThyUS2Path AUROC {r['thy_case_auroc']:.4f} "
              f"sens {sens:.3f} spec {1-fpr:.3f} sens-FPR {sens-fpr:+.3f}", flush=True)
df = pd.DataFrame(rows); df.to_csv(out / "x2_head_replicates.csv", index=False)
summ = df.groupby("rep").agg(['mean', 'std', 'min', 'max']).drop(columns=["seed", "epochs"], level=0)
summ.to_csv(out / "x2_head_replicates_summary.csv")
pd.set_option("display.width", 250)
print("\n", df.groupby("rep")[["tn5000_test_auroc", "thy_case_auroc", "thy_sens", "thy_spec", "thy_sens_minus_fpr"]]
      .describe().T.round(4).to_string())
