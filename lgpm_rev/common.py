"""Shared, frozen LG-PM machinery for the revision analyses.

Every function here is copied from the locked phase notebooks (Phases 2-6) so that the
revision scripts reproduce the published numbers before computing anything new. Where
two notebooks differ (e.g. the Phase-2 vs Phase-3c TN3K chains) both are provided and
named after the notebook they come from.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_curve

SEED = 42
IMG_SIZE, MASK_THRESH, PRED_THRESH = 384, 127, 0.5
SEG_K, RING_PX = 20, 15
PROTO_SCALES, SCORE_HW = ["e2", "e3", "e4"], 48
FEAT_SCALES = ["e3", "e4"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
# Kaggle defaults; override with LGPM_INPUT_ROOTS="dir1:dir2" and LGPM_OUT=dir for local runs
INPUT_ROOTS = tuple(os.environ.get("LGPM_INPUT_ROOTS", "/kaggle/input:/kaggle/working").split(":"))
OUT_ROOT = Path(os.environ.get("LGPM_OUT", "/kaggle/working/revision_outputs"))

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ----------------------------------------------------------------------------- utils
def set_seed(seed: int = SEED, deterministic: bool = False) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def out_dir(name: str) -> Path:
    d = OUT_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


class Tee:
    """Mirror stdout to a log file inside the output folder."""
    def __init__(self, path: Path):
        self.f = open(path, "w")
        self.stdout = sys.stdout
        sys.stdout = self
    def write(self, s):
        self.stdout.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.stdout.flush(); self.f.flush()


def gate(ok: bool, msg: str) -> None:
    """Reproduction gate: stop loudly if a locked value is not reproduced."""
    print(("  [GATE OK]   " if ok else "  [GATE FAIL] ") + msg, flush=True)
    if not ok:
        raise SystemExit(f"Reproduction gate failed: {msg}. Nothing new was computed. "
                         f"Check that the right Kaggle datasets are attached.")


def find(patterns, kind="file", roots=INPUT_ROOTS, required=True, exclude=()):
    """First match (deterministic sort) for any glob under the Kaggle input roots."""
    hits = []
    for pat in patterns:
        for root in roots:
            if Path(root).exists():
                hits += sorted(Path(root).rglob(pat))
    hits = [h for h in hits if (h.is_dir() if kind == "dir" else h.is_file())]
    hits = [h for h in hits if not any(e in str(h) for e in exclude)]
    if not hits and required:
        raise FileNotFoundError(f"none of {patterns} found under {roots} - attach the dataset")
    return hits[0] if hits else None


def find_json_with(name, key_path, roots=INPUT_ROOTS):
    """First `name` JSON file (sorted) that contains the nested key path, e.g.
    ("config", "support_episode_ids") picks the Phase-3 run_summary.json among many."""
    for root in roots:
        if not Path(root).exists():
            continue
        for p in sorted(Path(root).rglob(name)):
            try:
                d = json.loads(p.read_text())
                for k in key_path:
                    d = d[k]
                return p
            except Exception:
                continue
    raise FileNotFoundError(f"no {name} containing {'/'.join(key_path)} - attach Phase_3_classification")


def read_gray(p) -> np.ndarray:
    return np.asarray(Image.open(p).convert("L"))


def image_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def load_ckpt(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


# ------------------------------------------------------------------- preprocessing
def letterbox_image(img, size=IMG_SIZE):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    ri = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    t, l = (size - nh) // 2, (size - nw) // 2
    ci = np.zeros((size, size), np.float32); ci[t:t + nh, l:l + nw] = ri / 255.0
    return ci


def letterbox_mask(msk_raw, size=IMG_SIZE):
    h, w = msk_raw.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    rm = cv2.resize(msk_raw, (nw, nh), interpolation=cv2.INTER_NEAREST)
    t, l = (size - nh) // 2, (size - nw) // 2
    cm = np.zeros((size, size), np.uint8); cm[t:t + nh, l:l + nw] = (rm > 0).astype(np.uint8)
    return cm


def letterbox_pair(img, msk, size=IMG_SIZE):           # Phase-2 convention
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    ri = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rm = cv2.resize(msk, (nw, nh), interpolation=cv2.INTER_NEAREST)
    t, l = (size - nh) // 2, (size - nw) // 2
    ci = np.zeros((size, size), np.float32); ci[t:t + nh, l:l + nw] = ri / 255.0
    cm = np.zeros((size, size), np.uint8); cm[t:t + nh, l:l + nw] = (rm > 0).astype(np.uint8)
    return ci, cm


def make_ring(mask, r=RING_PX):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dil = cv2.dilate(mask.astype(np.uint8), k)
    ring = np.clip(dil.astype(np.int16) - mask.astype(np.int16), 0, 1).astype(np.uint8)
    if ring.sum() == 0:
        ring = (1 - mask.astype(np.uint8))
    return ring


def to_tensor(img):
    x = torch.from_numpy(np.stack([img, img, img])).float()
    return (x - MEAN) / STD


# --------------------------------------------------------------------------- models
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = conv_block(cin + cskip, cout)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class JointUNet(nn.Module):
    """Phase-1 ResNet34-UNet (state-dict compatible with every saved trunk)."""
    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        bb = tvm.resnet34(weights=None)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu)
        self.pool = bb.maxpool
        self.enc1, self.enc2, self.enc3, self.enc4 = bb.layer1, bb.layer2, bb.layer3, bb.layer4
        self.up4, self.up3 = Up(512, 256, 256), Up(256, 128, 128)
        self.up2, self.up1 = Up(128, 64, 64), Up(64, 64, 64)
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_block(64, 32), nn.Conv2d(32, 1, 1))
    def features(self, x):
        e0 = self.stem(x)
        e1 = self.enc1(self.pool(e0)); e2 = self.enc2(e1)
        e3 = self.enc3(e2); e4 = self.enc4(e3)
        return {"e0": e0, "e1": e1, "e2": e2, "e3": e3, "e4": e4}
    def decode(self, f):
        d = self.up1(self.up2(self.up3(self.up4(f["e4"], f["e3"]), f["e2"]), f["e1"]), f["e0"])
        return self.head(d)
    def forward(self, x):
        f = self.features(x)
        return self.decode(f), f


class FusionDecoder(nn.Module):
    def __init__(self, cin=2 * len(PROTO_SCALES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1))
    def forward(self, s):
        return self.net(s)


def frozen_trunk(ckpt_path):
    trunk = JointUNet().to(DEVICE)
    ck = load_ckpt(ckpt_path)
    trunk.load_state_dict(ck["model"], strict=True)
    trunk.eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    return trunk


def frozen_decoder(ckpt_path):
    dec = FusionDecoder().to(DEVICE)
    dec.load_state_dict(load_ckpt(ckpt_path)["decoder"], strict=True)
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


def erm_model(ckpt_path):
    import torchvision.models as tvm
    erm = tvm.resnet34(weights=None); erm.fc = nn.Linear(erm.fc.in_features, 1)
    erm.load_state_dict(load_ckpt(ckpt_path)["model"])
    erm = erm.to(DEVICE).eval()
    for p in erm.parameters():
        p.requires_grad_(False)
    return erm


def joint_arm(ckpt_path):
    st = load_ckpt(ckpt_path)
    trk = JointUNet().to(DEVICE); hd = nn.Linear(768, 1).to(DEVICE)
    trk.load_state_dict(st["trunk"], strict=True); hd.load_state_dict(st["head"], strict=True)
    trk.eval(); hd.eval()
    for p in list(trk.parameters()) + list(hd.parameters()):
        p.requires_grad_(False)
    return trk, hd, st


# ----------------------------------------------------------------- prototypes / pooling
def masked_proto(feats, masks):
    h, w = feats.shape[-2:]
    m = F.interpolate(masks, size=(h, w), mode="nearest")
    fg_den = m.sum(dim=(0, 2, 3)) + 1e-6
    bg_den = (1 - m).sum(dim=(0, 2, 3)) + 1e-6
    pf = (feats * m).sum(dim=(0, 2, 3)) / fg_den
    pb = (feats * (1 - m)).sum(dim=(0, 2, 3)) / bg_den
    return pf, pb


def neg_sq_dist(feats, proto):
    return -(feats - proto.view(1, -1, 1, 1)).pow(2).sum(1, keepdim=True)


def score_maps(que_feats, pf_pb_per_scale):
    outs = []
    for name in PROTO_SCALES:
        qf = que_feats[name]
        pf, pb = pf_pb_per_scale[name]
        outs.append(F.interpolate(neg_sq_dist(qf, pf), size=(SCORE_HW, SCORE_HW),
                                  mode="bilinear", align_corners=False))
        outs.append(F.interpolate(neg_sq_dist(qf, pb), size=(SCORE_HW, SCORE_HW),
                                  mode="bilinear", align_corners=False))
    return torch.cat(outs, dim=1)


def predict_logits_384(que_feats, pf_pb_per_scale, decoder):
    return F.interpolate(decoder(score_maps(que_feats, pf_pb_per_scale)),
                         size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)


def masked_gap(feat, msk01):
    h, w = feat.shape[-2:]
    m = F.interpolate(msk01, size=(h, w), mode="nearest")
    den = m.sum(dim=(2, 3)).clamp(min=1.0)
    return (feat * m).sum(dim=(2, 3)) / den


# ------------------------------------------------------------------- TN3K (Phase 3c chain)
def pair_images_masks(image_dir, mask_dir, dataset, split):
    imgs = {p.stem: p for p in image_files(image_dir)}
    msks = {p.stem: p for p in image_files(mask_dir)}
    common = sorted(set(imgs) & set(msks))
    return pd.DataFrame([{"id": k, "image": str(imgs[k]), "mask": str(msks[k]),
                          "dataset": dataset, "split": split} for k in common])


def tn3k_train_phase3c(tn3k_root):
    """Canonical Phase-3c/5/6 chain: sha256-dedup trainval, seed-42 15% val carve-out."""
    tn3k = pair_images_masks(Path(tn3k_root) / "trainval-image", Path(tn3k_root) / "trainval-mask",
                             "TN3K", "trainval")
    assert len(tn3k) > 2700, f"TN3K trainval raw pool drift: {len(tn3k)} (canonical 2879)"
    tn3k["sha256"] = tn3k["image"].apply(sha256_file)
    tn3k = tn3k.sort_values("id", kind="stable").drop_duplicates("sha256", keep="first")
    rs = np.random.RandomState(SEED)
    perm = rs.permutation(len(tn3k))
    val_ids = set(tn3k.iloc[perm[: int(round(0.15 * len(tn3k)))]].id)
    tn3k["split"] = np.where(tn3k.id.isin(val_ids), "val", "train")
    return tn3k[tn3k.split == "train"].reset_index(drop=True)


def fixed_support_prototypes(trunk, tn3k_train, archived_ids):
    """The fixed K=20 support episode (seed 52), asserted against the Phase-3 archive."""
    rng = np.random.RandomState(SEED + 10)
    sup = tn3k_train.iloc[rng.permutation(len(tn3k_train))[:SEG_K]].reset_index(drop=True)
    gate(list(sup.id) == list(archived_ids),
         f"fixed support episode reproduces the Phase-3 archive ({SEG_K} ids)")
    xs, ms = [], []
    for _, r in sup.iterrows():
        xs.append(to_tensor(letterbox_image(read_gray(r["image"]).astype(np.float32))))
        ms.append(torch.from_numpy(letterbox_mask(
            (read_gray(r["mask"]) >= MASK_THRESH).astype(np.uint8))[None]).float())
    sx = torch.stack(xs).to(DEVICE); sm = torch.stack(ms).to(DEVICE)
    with torch.no_grad():
        sf = trunk.features(sx)
    return {name: masked_proto(sf[name], sm) for name in PROTO_SCALES}


# ------------------------------------------------------------------------- metrics
def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def spec90_threshold(y_val, p_val):
    fpr, tpr, thr = roc_curve(y_val, p_val)
    spec = 1 - fpr
    m = spec >= 0.90
    i = np.where(m)[0][np.argmax(tpr[m])] if m.any() else int(np.argmax(spec))
    return float(thr[i]), float(tpr[i]), float(spec[i])


def lin_apply(head, X):
    w, b, mu, sd = (np.array(head[k]) for k in ("w", "b", "mu", "sd"))
    z = ((X.astype(np.float64) - mu) / sd) @ w.T + b
    return (1 / (1 + np.exp(-z.ravel()))).astype(float)


def confusion(y, yhat):
    y = np.asarray(y).astype(int); yhat = np.asarray(yhat).astype(bool)
    tp = int((yhat & (y == 1)).sum()); fn = int((~yhat & (y == 1)).sum())
    tn = int((~yhat & (y == 0)).sum()); fp = int((yhat & (y == 0)).sum())
    return tp, fn, tn, fp


def operating_point_row(name, y, p, thr, n_boot=2000, seed=SEED):
    """Sensitivity, achieved specificity, counts, bootstrap CIs, sens - FPR."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float); yhat = p >= thr
    tp, fn, tn, fp = confusion(y, yhat)
    rng = np.random.RandomState(seed)
    se, sp, j = [], [], []
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y)); yy, hh = y[i], yhat[i]
        if (yy == 1).any() and (yy == 0).any():
            a, b = hh[yy == 1].mean(), (~hh[yy == 0]).mean()
            se.append(a); sp.append(b); j.append(a - (1 - b))
    from sklearn.metrics import roc_auc_score
    q = lambda v: f"{np.percentile(v, 2.5):.4f}..{np.percentile(v, 97.5):.4f}"
    return {"model": name, "n": len(y), "malignant": int(y.sum()), "benign": int((y == 0).sum()),
            "auroc": round(float(roc_auc_score(y, p)), 4), "threshold": round(float(thr), 6),
            "sens": round(tp / max(tp + fn, 1), 4), "sens_CI": q(se),
            "spec": round(tn / max(tn + fp, 1), 4), "spec_CI": q(sp),
            "TP": tp, "FN": fn, "TN": tn, "FP": fp,
            "sens_minus_fpr": round(tp / max(tp + fn, 1) - fp / max(fp + tn, 1), 4),
            "sens_minus_fpr_CI": q(j),
            "ppv": round(tp / max(tp + fp, 1), 4) if tp + fp else float("nan")}


def mcnemar(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b)); c = int(np.sum(~correct_a & correct_b))
    chi2 = None if b + c == 0 else (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import binomtest
    p = None if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"a_only": b, "b_only": c, "chi2_cc": chi2, "exact_p": p}


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


class Timer:
    def __init__(self):
        self.t0 = time.time()
    def __str__(self):
        return f"{(time.time() - self.t0) / 60:.1f} min"
