"""s01 - ThyUS2Path operating points for ALL six models (R2.2, R1.2).  GPU, ~15 min on T4.

Phase 5 saved descriptors (for R0/R1p/R2p) but not ERM or JOINT probabilities, and it
never computed external specificity. This script:
  1. rebuilds the Phase-5 image manifest with the Phase-5 rules (per-batch prefix join),
     drops the cases Phase 5 excluded, and asserts row-for-row equality with the bank;
  2. scores ERM, JOINT-g and JOINT-img exactly as Phase 5 did (ERM under autocast, JOINT
     in fp32), and R0/R1p/R2p from the bank with the 3c heads;
  3. GATES: case-level AUROC and sensitivity of all six models must reproduce Table S15;
  4. reports, per model, the threshold, achieved specificity, confusion counts, bootstrap
     CIs, sensitivity - FPR, and PPV; J1-style McNemar on malignant AND benign cases; and
     a post hoc matched-specificity view.  Nothing is refitted.

Attach: ThyUS2Path, Thyroid_phase1_results, phase2 prototype seg (decoder not needed here),
Phase_3_classification, phase3_c_, phase4_joint, phase5_thyus2path(_v2).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from lgpm_rev.common import (DEVICE, FEAT_SCALES, PRED_THRESH, SEED, Tee, Timer, find, gate,
                             joint_arm, erm_model, lin_apply, letterbox_image, masked_gap,
                             mcnemar, operating_point_row, out_dir, read_gray, save_json,
                             spec90_threshold, to_tensor, IMG_EXTS)

OUT = out_dir("s01_thyus2path_all_models")
Tee(OUT / "log.txt")
T = Timer()
BATCH = 16
LOCKED = {  # Table S15 / phase5_main_table.csv (case level)
    "ERM": (0.5479, 0.0134), "R0_image": (0.5400, 0.0460), "R1p_lesion": (0.5081, 0.1590),
    "R2p_lesion_ring": (0.5211, 0.0881), "JOINT-g": (0.5112, 0.0613), "JOINT-img": (0.4780, 0.0153)}

# ------------------------------------------------------------------ artifacts
bank_p = find(["phase5_bank_thyus2path.npz"])
man_p = find(["phase5_manifest_unique_cases.csv"])
heads_p = find(["linear_heads_retrained.json"])
erm_ck = find(["erm_resnet34_tn5000.pt"])
erm_pj = find(["erm_resnet34_probs.json"])
fval_p = find(["features_val.npz"], exclude=("tnui",))
jg_ck, ji_ck = find(["joint_g_trunk.pt"]), find(["joint_img_trunk.pt"])
jp_p = find(["phase4_joint_probs_tn5000.json"])
for k, v in dict(bank=bank_p, manifest=man_p, heads=heads_p, erm=erm_ck, erm_probs=erm_pj,
                 val=fval_p, joint_g=jg_ck, joint_img=ji_ck, joint_probs=jp_p).items():
    print(f"  {k:12s} {v}")

BANK = np.load(bank_p, allow_pickle=True)
bank_ids = BANK["ids"].astype(str)
CASES = pd.read_csv(man_p)

# ----------------------------------------------- 1. rebuild the Phase-5 image manifest
first = Path(CASES.first_image.iloc[0])
if first.exists():
    root = [p for p in first.parents if p.name.lower().startswith("batch")][-1].parent
else:  # different mount path: find the folder that holds the batch folders
    cands = [p for p in Path("/kaggle/input").rglob("batch1_image") if p.is_dir()
             and "batch" not in p.parent.name.lower()]
    assert cands, "ThyUS2Path images not found - attach the ThyUS2Path dataset"
    root = cands[0].parent
print("ThyUS2Path root:", root)
imgs = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def batch_key_of(p):
    for part in Path(p).relative_to(root).parts[:-1]:
        if "batch" in part.lower():
            return part
    return "root"


lab = CASES.set_index("case")["label"].to_dict()
rows = []
for p in imgs:
    b = batch_key_of(p); pref = p.stem.split("_")[0]; c = f"{b}:{pref}"
    if c in lab:
        rows.append({"case": c, "batch": b, "stem": p.stem, "image": str(p), "label": int(lab[c])})
img_df = pd.DataFrame(rows).drop_duplicates("image").sort_values("image").reset_index(drop=True)
# Phase-5 A3 excluded whole cases (near-duplicates of training pools). The bank keeps
# img_df's path order minus those cases, so a greedy walk recovers them; the gate below
# then requires exact equality.
print(f"images rebuilt: {len(img_df)} | bank rows: {len(bank_ids)} | "
      f"images to exclude: {len(img_df) - len(bank_ids)}")
j, drop_idx = 0, []
for i, s in enumerate(img_df.stem):
    if j < len(bank_ids) and s == bank_ids[j]:
        j += 1
    else:
        drop_idx.append(i)
excluded_cases = sorted(set(img_df.case.iloc[drop_idx]))
manifest5 = img_df[~img_df.case.isin(excluded_cases)].reset_index(drop=True)
gate(list(manifest5.stem) == list(bank_ids),
     f"rebuilt manifest equals bank order ({len(manifest5)} images; excluded cases {excluded_cases})")
gate((manifest5.label.to_numpy() == BANK["labels"]).all(), "image labels equal bank labels")
manifest5.to_csv(OUT / "thyus2path_image_manifest.csv", index=False)

case_order = sorted(manifest5.case.unique())
pos = {c: i for i, c in enumerate(case_order)}
ci = manifest5.case.map(pos).to_numpy(); nimg = np.bincount(ci)
y_case = (np.bincount(ci, weights=manifest5.label.to_numpy()) / nimg).astype(int)
gate(len(case_order) == 839 and int(y_case.sum()) == 522, "cohort is 839 cases / 522 malignant")


def case_mean(p):
    return np.bincount(ci, weights=np.asarray(p, float)) / nimg


# ------------------------------------------------------------------- 2. probabilities
H = json.loads(Path(heads_p).read_text())
P_IMG = {rep: lin_apply(H[rep]["head"], BANK[rep]) for rep in ["R0_image", "R1p_lesion", "R2p_lesion_ring"]}

erm = erm_model(erm_ck)
TRG, HDG, _ = joint_arm(jg_ck)
TRI, HDI, _ = joint_arm(ji_ck)
p_erm, p_jg, p_ji = [], [], []
for i0 in range(0, len(manifest5), BATCH):
    chunk = manifest5.iloc[i0:i0 + BATCH]
    x = torch.stack([to_tensor(letterbox_image(read_gray(r).astype(np.float32)))
                     for r in chunk.image]).to(DEVICE)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):   # Phase-5 ERM policy
            p_erm.append(torch.sigmoid(erm(x)).squeeze(1).float().cpu())
        lg, fg = TRG(x)                                                   # Phase-5 JOINT: fp32
        pm = (torch.sigmoid(lg.float()) >= PRED_THRESH).float()
        for bi in range(len(x)):
            if pm[bi].sum() < 4:
                pm[bi] = 1.0
        dg = torch.cat([masked_gap(fg[nm], pm) for nm in FEAT_SCALES], 1)
        p_jg.append(torch.sigmoid(HDG(dg.float()).squeeze(1)).float().cpu())
        fi = TRI.features(x)
        di = torch.cat([fi[nm].mean(dim=(2, 3)) for nm in FEAT_SCALES], 1)
        p_ji.append(torch.sigmoid(HDI(di.float()).squeeze(1)).float().cpu())
    if (i0 // BATCH) % 100 == 0:
        print(f"  scored {min(i0 + BATCH, len(manifest5))}/{len(manifest5)} images ({T})", flush=True)
P_IMG["ERM"] = torch.cat(p_erm).numpy().astype(float)
P_IMG["JOINT-g"] = torch.cat(p_jg).numpy().astype(float)
P_IMG["JOINT-img"] = torch.cat(p_ji).numpy().astype(float)

# thresholds: each model's own TN5000-val raw probabilities (Phase-5 rule)
y_val = np.load(fval_p, allow_pickle=True)["labels"].astype(int)
JP = json.loads(Path(jp_p).read_text()); ERMJ = json.loads(Path(erm_pj).read_text())
PV = {"ERM": np.asarray(ERMJ["probs"]["val"], float).ravel(),
      "JOINT-g": np.asarray(JP["val"]["JOINTg"], float).ravel(),
      "JOINT-img": np.asarray(JP["val"]["JOINTimg"], float).ravel()}
for rep in ["R0_image", "R1p_lesion", "R2p_lesion_ring"]:
    PV[rep] = np.asarray(H[rep]["probs"]["val"], float).ravel()
THR = {m: spec90_threshold(y_val, PV[m])[0] for m in PV}
gate(abs(THR["JOINT-g"] - JP["thr"]["JOINTg"]) < 1e-6 and abs(THR["JOINT-img"] - JP["thr"]["JOINTimg"]) < 1e-6,
     "JOINT thresholds equal the archived Phase-4 values")

P_CASE = {m: case_mean(P_IMG[m]) for m in LOCKED}
print("\nReproduction of Table S15 (case level):")
for m, (a_ref, s_ref) in LOCKED.items():
    a = roc_auc_score(y_case, P_CASE[m]); s = float((P_CASE[m][y_case == 1] >= THR[m]).mean())
    gate(abs(a - a_ref) <= 0.002 and abs(s - s_ref) <= 2.5 / 522,
         f"{m:16s} AUROC {a:.4f} (locked {a_ref}) | sens {s:.4f} (locked {s_ref})")

# ------------------------------------------------------------------- 3. new analyses
rows = [operating_point_row(m, y_case, P_CASE[m], THR[m]) for m in LOCKED]
OP = pd.DataFrame(rows)
OP.to_csv(OUT / "thyus2path_operating_points_all_models.csv", index=False)
pd.set_option("display.width", 250)
print("\nThyUS2Path case-level operating points (frozen source thresholds; nothing refitted):")
print(OP.to_string(index=False))

yh = {m: P_CASE[m] >= THR[m] for m in LOCKED}
mc = {"J1_malignant (sensitivity)": mcnemar(yh["R1p_lesion"][y_case == 1], yh["ERM"][y_case == 1]),
      "J1_benign (false positives; a_only = R1p FP not ERM)": mcnemar(yh["R1p_lesion"][y_case == 0], yh["ERM"][y_case == 0])}
print("\nMcNemar R1p vs ERM:", json.dumps(mc, indent=1))

def sens_at_spec(p, target):
    fpr, tpr, _ = roc_curve(y_case, p); ok = (1 - fpr) >= target
    return float(tpr[ok].max()) if ok.any() else 0.0

mt = pd.DataFrame([{"specificity_of": ref, "target_spec": float(OP.set_index("model").loc[ref, "spec"]),
                    "model": m, "sens_at_target_spec": round(sens_at_spec(P_CASE[m], float(OP.set_index("model").loc[ref, "spec"])), 4)}
                   for ref in LOCKED for m in LOCKED])
mt.to_csv(OUT / "thyus2path_matched_specificity_POSTHOC.csv", index=False)
print("\nPOST HOC sensitivity at matched achieved specificity:")
print(mt.pivot(index="model", columns="specificity_of", values="sens_at_target_spec").round(4).to_string())

np.savez_compressed(OUT / "thyus2path_scores.npz", y_case=y_case, case=np.array(case_order),
                    **{f"case_{m}": P_CASE[m] for m in LOCKED}, **{f"img_{m}": P_IMG[m] for m in LOCKED},
                    img_stem=manifest5.stem.to_numpy(), img_case=manifest5.case.to_numpy())
save_json({"thresholds": THR, "mcnemar": mc, "excluded_cases": excluded_cases,
           "n_cases": len(case_order), "n_images": len(manifest5)}, OUT / "summary.json")
print(f"\ndone in {T} -> {OUT}")
