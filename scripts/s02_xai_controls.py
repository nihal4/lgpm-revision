"""s02 - explanation controls on TN5000-test (R2.4, R1.7).  GPU, ~10 min on T4.

The Phase-6 K1 "R1p localisation ratio" is mask-region precision (share of predicted-mask
PIXELS inside the expert box), not attribution energy. This script recomputes the frozen
masks and Grad-CAM exactly as Phase 6 did, GATES every image against the saved Phase-6
panel, and then adds the comparisons the reviewers asked for:

  uniform predicted mask   = K1 as published (restriction alone)
  R1p-ELA energy           = the actual exact-linear-attribution map, ReLU -> l1, in-box energy
  Grad-CAM                 = K1 comparator as published
  masked Grad-CAM          = Grad-CAM x predicted mask (the "constrained CAM" R1.7 asks for)
  Eigen-CAM / masked Eigen-CAM (descriptive)

Paired bootstrap (5,000, seed 42) on the localisation ratio for:
  ELA - Grad-CAM, ELA - uniform mask, masked Grad-CAM - uniform mask, ELA - masked Grad-CAM.

Raw data (Kaggle input): TN5000, TN3K_Data. Needs artifacts/phase2/best_fusion_decoder.pt.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from lgpm_rev.common import (DEVICE, FEAT_SCALES, IMG_SIZE, PRED_THRESH, SEED, Tee, Timer,
                             erm_model, find, find_json_with, fixed_support_prototypes, frozen_decoder,
                             frozen_trunk, gate, letterbox_image, predict_logits_384,
                             read_gray, save_json, tn3k_train_phase3c, to_tensor, out_dir)

OUT = out_dir("s02_xai_controls")
Tee(OUT / "log.txt")
T = Timer()
N_BOOT = 5000

panel_p = find(["phase6_panel_tn5000.npz"])
heads_p = find(["linear_heads_retrained.json"])
p3s_p = find_json_with("run_summary.json", ("config", "support_episode_ids"))
ck1 = find(["best_resnet34_seed42.pt"]); ckd = find(["best_fusion_decoder.pt"])
erm_ck = find(["erm_resnet34_tn5000.pt"])
tn3k_root = find(["tn3k-data/Thyroid Dataset/tn3k", "*Thyroid Dataset/tn3k"], kind="dir")
t5_root = find(["tn5000/TN5000_forReview", "*TN5000_forReview"], kind="dir")
P3S = json.loads(Path(p3s_p).read_text())
gate("support_episode_ids" in P3S.get("config", {}), f"Phase-3 run_summary found ({p3s_p})")
PANEL = np.load(panel_p, allow_pickle=False)
IDS = [str(x) for x in PANEL["ids"]]
BOXM = PANEL["boxm"].astype(np.uint8)
print(f"panel: {len(IDS)} TN5000-test images | ({T})")

trunk = frozen_trunk(ck1); decoder = frozen_decoder(ckd); erm = erm_model(erm_ck)
PF_PB = fixed_support_prototypes(trunk, tn3k_train_phase3c(tn3k_root),
                                 P3S["config"]["support_episode_ids"])
H = json.loads(Path(heads_p).read_text())["R1p_lesion"]["head"]
w = np.asarray(H["w"], np.float64).ravel(); sd = np.asarray(H["sd"], np.float64).ravel()
WV = {"e3": w[:256] / sd[:256], "e4": w[256:768] / sd[256:768]}       # Phase-6 head_blocks_linr
FEAT_HW = {"e3": (IMG_SIZE // 16, IMG_SIZE // 16), "e4": (IMG_SIZE // 32, IMG_SIZE // 32)}

ACTS = {}
def erm_gradcam(x1):                                                     # Phase-6 erm_forward_cam
    h = erm.layer4.register_forward_hook(lambda m, i, o: ACTS.__setitem__("e4", o))
    try:
        xg = x1.clone().requires_grad_(True)
        erm.zero_grad(set_to_none=True)
        with torch.enable_grad():
            lg = erm(xg)
            g = torch.autograd.grad(lg.sum(), ACTS["e4"])[0]
    finally:
        h.remove()
    a = ACTS["e4"][0].detach().cpu().numpy().astype(np.float64)
    gg = g[0].cpu().numpy().astype(np.float64)
    return np.maximum((gg.mean(axis=(1, 2), keepdims=True) * a).sum(0), 0.0), a

def eigencam(a):
    A = a.reshape(a.shape[0], -1)
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    proj = (u[:, 0][:, None] * A).sum(0).reshape(a.shape[1:])
    return np.maximum(-proj if proj.sum() < 0 else proj, 0.0)

def relu_l1(cam):
    a = np.maximum(np.asarray(cam, np.float64), 0.0); s = a.sum()
    return (a / s if s > 0 else a), bool(s > 0)

def up384(m):
    return cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

def native(m384):
    t = torch.from_numpy(np.asarray(m384, np.float32)[None, None])
    out = {}
    for sc, hw in FEAT_HW.items():
        m = F.interpolate(t, size=hw, mode="nearest").numpy()[0, 0].astype(np.float64)
        out[sc] = (m, max(float(m.sum()), 1.0))
    return out

rows = []
for i0 in range(0, len(IDS), 16):
    ids = IDS[i0:i0 + 16]
    x = torch.stack([to_tensor(letterbox_image(read_gray(t5_root / "JPEGImages" / f"{k}.jpg").astype(np.float32)))
                     for k in ids]).to(DEVICE)
    with torch.no_grad():
        feats = trunk.features(x)
        pm = (torch.sigmoid(predict_logits_384(feats, PF_PB, decoder)) >= PRED_THRESH).float()
    for bi in range(len(ids)):
        gi = i0 + bi
        if pm[bi].sum() < 4:
            pm[bi] = 1.0
        les = pm[bi, 0].cpu().numpy(); bm = BOXM[gi]; boxfrac = max(bm.mean(), 1e-9)
        mn = native(les)
        ela = sum(up384(mn[sc][0] * np.tensordot(WV[sc], feats[sc][bi].cpu().numpy().astype(np.float64),
                                                 axes=([0], [0])) / mn[sc][1]) for sc in FEAT_SCALES)
        e01, ela_nonzero = relu_l1(ela)
        gc, acts = erm_gradcam(x[bi:bi + 1])
        gc384 = up384(gc); g01, _ = relu_l1(gc384)
        mg01, mg_nonzero = relu_l1(gc384 * les)
        eg384 = up384(eigencam(acts)); eg01, _ = relu_l1(eg384)
        meg01, _ = relu_l1(eg384 * les)
        uni_inbox = float(bm[les > 0].mean()) if les.sum() else 0.0
        hit = lambda c: float(bm[np.unravel_index(int(np.argmax(c)), c.shape)] > 0)
        rows.append({"id": ids[bi], "label": int(PANEL["labels"][gi]), "boxfrac": boxfrac,
                     "mask_area": float(les.mean()),
                     "uniform_inbox": uni_inbox, "ela_inbox": float(e01[bm > 0].sum()),
                     "gc_inbox": float(g01[bm > 0].sum()), "mgc_inbox": float(mg01[bm > 0].sum()),
                     "eig_inbox": float(eg01[bm > 0].sum()), "meig_inbox": float(meg01[bm > 0].sum()),
                     "ela_hit": hit(e01), "gc_hit": hit(g01), "mgc_hit": hit(mg01),
                     "ela_nonzero": ela_nonzero, "mgc_nonzero": mg_nonzero,
                     "gc_energy_in_mask": float(g01[les > 0].sum())})
    if (i0 // 16) % 10 == 0:
        print(f"  {min(i0 + 16, len(IDS))}/{len(IDS)} ({T})", flush=True)

D = pd.DataFrame(rows)
for k in ["uniform", "ela", "gc", "mgc", "eig", "meig"]:
    D[f"{k}_ratio"] = D[f"{k}_inbox"] / D["boxfrac"]

# ---- reproduction gates against the saved Phase-6 panel (per image) ----
def maxdev(a, b):
    return float(np.nanmax(np.abs(np.asarray(a, float) - np.asarray(b, float))))
gate(maxdev(D.mask_area, PANEL["mask_area"]) < 1e-4, f"predicted-mask areas match panel (max dev {maxdev(D.mask_area, PANEL['mask_area']):.2e})")
gate(maxdev(D.uniform_inbox, PANEL["met_r1p_inbox"]) < 1e-4, f"uniform-mask in-box equals published K1 R1p column (max dev {maxdev(D.uniform_inbox, PANEL['met_r1p_inbox']):.2e})")
gate(maxdev(D.gc_inbox, PANEL["met_gc_inbox"]) < 2e-3, f"Grad-CAM in-box matches panel (max dev {maxdev(D.gc_inbox, PANEL['met_gc_inbox']):.2e})")
gate(abs(D.uniform_ratio.mean() - 41.63) < 0.05 and abs(D.gc_ratio.mean() - 3.60) < 0.05,
     f"K1 means reproduced: uniform {D.uniform_ratio.mean():.2f} (41.6), Grad-CAM {D.gc_ratio.mean():.2f} (3.6)")
gate((D.ela_hit.to_numpy() == PANEL["met_r1p_hit"]).mean() > 0.99, "ELA pointing-game hits match panel")
D.to_csv(OUT / "xai_controls_per_image.csv", index=False)

# ---- summary + paired bootstrap ----
names = {"uniform": "Uniform predicted mask (= published K1 R1p)", "ela": "R1p-ELA attribution energy",
         "gc": "ERM Grad-CAM (published comparator)", "mgc": "Grad-CAM x predicted mask",
         "eig": "ERM Eigen-CAM", "meig": "Eigen-CAM x predicted mask"}
summ = pd.DataFrame([{"map": names[k], "mean_ratio": D[f"{k}_ratio"].mean(), "median_ratio": D[f"{k}_ratio"].median(),
                      "mean_inbox_share": D[f"{k}_inbox"].mean(),
                      "pointing_hit": D[f"{k}_hit"].mean() if f"{k}_hit" in D else np.nan} for k in names]).round(4)
summ.to_csv(OUT / "xai_controls_summary.csv", index=False)
print("\n", summ.to_string(index=False))
print(f"  ELA maps with no positive mass: {int((~D.ela_nonzero).sum())}; masked Grad-CAM with no mass: {int((~D.mgc_nonzero).sum())}")

rng = np.random.RandomState(SEED); N = len(D); idx = [rng.randint(0, N, N) for _ in range(N_BOOT)]
tests = []
for a, b in [("ela", "gc"), ("ela", "uniform"), ("mgc", "uniform"), ("ela", "mgc")]:
    d = (D[f"{a}_ratio"] - D[f"{b}_ratio"]).to_numpy()
    bs = np.array([d[i].mean() for i in idx])
    tests.append({"A": names[a], "B": names[b], "delta_mean": d.mean(), "CI95": f"{np.percentile(bs, 2.5):.3f}..{np.percentile(bs, 97.5):.3f}",
                  "p_two_sided": min(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1.0)})
TT = pd.DataFrame(tests).round(4)
TT.to_csv(OUT / "xai_controls_paired_tests_POSTHOC.csv", index=False)
print("\nPaired localisation-ratio contrasts (post-review, not pre-registered):\n", TT.to_string(index=False))
save_json({"n": N, "runtime": str(T)}, OUT / "summary.json")
print(f"\ndone in {T} -> {OUT}")
