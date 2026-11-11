"""s03 - support-free decoder control (R2.1, R1.3).  GPU, ~60-80 min on T4.

Question: how much of the few-shot model's segmentation comes from SUPPORT CONDITIONING,
and how much from the supervised Phase-1 encoder plus a learned decoder?

Arms, all on the SAME frozen Phase-1 encoder and the SAME Phase-2 evaluation episodes
(200 per condition, 8 queries, identical seeds, so every query image is shared):
  A  Phase-2 fusion decoder, K=20 source supports     (published; GATE: 0.7926/0.7079/0.6782)
  B  support-free decoder, capacity-matched           (NEW: learned 1x1 projections of
     e2/e3/e4 to 2 channels each replace the 6 prototype score maps; then the identical
     6->32->16->1 head; trained on the identical Phase-2 episode stream, queries only)
  C  Phase-1 U-Net, its own full decoder, threshold 0.60, single view (supervised reference)

Reported: episode-mean Dice per arm and paired episode-bootstrap deltas (B-A, C-A).
Also per-image Dice on the full test pools for B and C (GATE for C: Table S2 non-TTA).

Attach: TN3K_Data (contains DDTI), TNUI-2021, Thyroid_phase1_results, phase2 prototype seg.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from lgpm_rev.common import (DEVICE, IMG_SIZE, MASK_THRESH, PROTO_SCALES, SCORE_HW, SEED, Tee,
                             Timer, find, frozen_decoder, frozen_trunk, gate, letterbox_pair,
                             masked_proto, out_dir, read_gray, save_json, set_seed, sha256_file,
                             to_tensor, pair_images_masks, score_maps)

OUT = out_dir("s03_support_free_decoder")
Tee(OUT / "log.txt")
T = Timer()
set_seed(SEED, deterministic=True)                    # Phase-2 seed policy

# Phase-2 protocol (verbatim)
K_LIST, K_MAX, Q_QUERY = [1, 5, 10, 20], 20, 8
N_EP_TRAIN, N_EP_VAL, N_EP_EVAL = 300, 100, 200
EPOCHS, PATIENCE, LR_DEC, PRED_THRESH = 30, 8, 1e-3, 0.5
PHASE1_THR = 0.60
LOCKED_A = {"TN3K test | in-domain": 0.7926, "TNUI test | zero-shot": 0.7079, "DDTI | zero-shot": 0.6782}
LOCKED_C_POOL = {"TN3K test": 0.8441, "TNUI test": 0.7367, "DDTI": 0.7431}   # Table S2, no TTA

ck1 = find(["best_resnet34_seed42.pt"]); ckd = find(["best_fusion_decoder.pt"])
tn3k_root = find(["tn3k-data/Thyroid Dataset/tn3k", "*Thyroid Dataset/tn3k"], kind="dir")
ddti_root = tn3k_root.parent / "DDTI dataset" / "DDTI"
tnui_root = find(["tnui-2021/TNUI-2021-/thyroid_data", "*TNUI-2021-*/thyroid_data", "*thyroid_data"], kind="dir")
print("tn3k:", tn3k_root, "| ddti:", ddti_root, ddti_root.exists(), "| tnui:", tnui_root)

# ---------------------------------------------------------------- Phase-2 pools (verbatim)
def add_hashes(df):
    df = df.copy(); df["sha256"] = df["image"].apply(sha256_file); return df

def drop_dups(df, priority):
    df = df.copy(); df["_p"] = df["split"].map(priority).fillna(99)
    df = df.sort_values(["_p", "id"], kind="stable")
    return df[~df.duplicated("sha256", keep="first")].drop(columns="_p").reset_index(drop=True)

tn3k = pd.concat([pair_images_masks(tn3k_root / "trainval-image", tn3k_root / "trainval-mask", "TN3K", "trainval"),
                  pair_images_masks(tn3k_root / "test-image", tn3k_root / "test-mask", "TN3K", "test")], ignore_index=True)
tn3k = drop_dups(add_hashes(tn3k), {"trainval": 0, "test": 1})
rng0 = np.random.RandomState(SEED)
tv = tn3k[tn3k.split == "trainval"].copy()
perm = rng0.permutation(len(tv))
val_ids = set(tv.iloc[perm[: int(round(0.15 * len(tv)))]].id)
tv["split"] = np.where(tv.id.isin(val_ids), "val", "train")
tn3k = pd.concat([tv, tn3k[tn3k.split == "test"]], ignore_index=True)
tnui = drop_dups(add_hashes(pd.concat([pair_images_masks(tnui_root / s / "images", tnui_root / s / "masks", "TNUI-2021", s)
                                       for s in ["train", "val", "test"]], ignore_index=True)),
                 {"train": 0, "val": 1, "test": 2})
ddti = drop_dups(add_hashes(pair_images_masks(ddti_root / "1_or_data" / "image", ddti_root / "1_or_data" / "mask",
                                              "DDTI", "external")), {"external": 0})
POOLS = {"tn3k_train": tn3k[tn3k.split == "train"].reset_index(drop=True),
         "tn3k_val": tn3k[tn3k.split == "val"].reset_index(drop=True),
         "tn3k_test": tn3k[tn3k.split == "test"].reset_index(drop=True),
         "tnui_test": tnui[tnui.split == "test"].reset_index(drop=True),
         "ddti_all": ddti.reset_index(drop=True)}
for k, v in POOLS.items():
    print(f"  {k:10s} n={len(v)}")
src_masks = len(POOLS["tn3k_train"])
print(f"SOURCE SUPERVISION: TN3K internal-training masks = {src_masks} (Phase-1 encoder and both decoders)")

# ------------------------------------------------------------ Phase-2 loaders (verbatim)
def load_item(row):
    img = read_gray(row["image"]).astype(np.float32)
    msk = (read_gray(row["mask"]) >= MASK_THRESH).astype(np.uint8)
    return letterbox_pair(img, msk, IMG_SIZE)

def joint_aug(imgs, msks, rng):
    do_flip = rng.rand() < 0.5
    angle = rng.uniform(-10, 10) if rng.rand() < 0.5 else 0.0
    out_i, out_m = [], []
    for im, mk in zip(imgs, msks):
        if do_flip:
            im, mk = np.ascontiguousarray(im[:, ::-1]), np.ascontiguousarray(mk[:, ::-1])
        if angle:
            M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angle, 1.0)
            im = cv2.warpAffine(im, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR)
            mk = cv2.warpAffine(mk, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_NEAREST)
        if rng.rand() < 0.5:
            im = np.clip(im * rng.uniform(0.8, 1.2), 0, 1)
        out_i.append(im.astype(np.float32)); out_m.append((mk > 0).astype(np.uint8))
    return out_i, out_m

class NestedEpisodeSampler:
    def __init__(self, support_df, query_df, q=Q_QUERY, seed=SEED):
        self.sup = support_df.reset_index(drop=True); self.que = query_df.reset_index(drop=True)
        self.q = q; self.rng = np.random.RandomState(seed)
        self.shared = len(set(self.sup.sha256) & set(self.que.sha256)) > 0
    def sample(self, kmax=K_MAX):
        if self.shared:
            idx = self.rng.permutation(len(self.sup)); sup_rows = self.sup.iloc[idx[:kmax]]
            avail = self.que[~self.que.sha256.isin(set(sup_rows.sha256))]
            que_rows = avail.iloc[self.rng.permutation(len(avail))[:self.q]]
        else:
            sup_rows = self.sup.iloc[self.rng.permutation(len(self.sup))[:kmax]]
            que_rows = self.que.iloc[self.rng.permutation(len(self.que))[:self.q]]
        return sup_rows.reset_index(drop=True), que_rows.reset_index(drop=True)

def episode_tensors(sup_rows, que_rows, aug=False, rng=None):
    data = [load_item(r) for _, r in pd.concat([sup_rows, que_rows]).iterrows()]
    imgs, msks = [d[0] for d in data], [d[1] for d in data]
    if aug and rng is not None:
        imgs, msks = joint_aug(imgs, msks, rng)
    ns = len(sup_rows)
    st = lambda L: torch.stack([to_tensor(i) for i in L])
    sm = lambda L: torch.stack([torch.from_numpy(m[None]).float() for m in L])
    return st(imgs[:ns]), sm(msks[:ns]), st(imgs[ns:]), sm(msks[ns:])

def dice_np(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    if (2 * tp + fp + fn) == 0:
        return 1.0 if not (gt.any() or pred.any()) else 0.0
    return (2 * tp) / (2 * tp + fp + fn)

def seg_loss(logits, targets, eps=1e-6):
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * targets).sum(dim=(1, 2, 3)) + eps) / (p.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps)
    return F.binary_cross_entropy_with_logits(logits, targets) + dice.mean()

# ------------------------------------------------------------------------ models
trunk = frozen_trunk(ck1)
fusion = frozen_decoder(ckd)

@torch.no_grad()
def encode(x):                                         # Phase-2 encode: autocast, fp32 out
    with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
        f = trunk.features(x)
    return {k: v.float() for k, v in f.items()}

class SupportFreeDecoder(nn.Module):
    """Same head as the fusion decoder; its 6 input channels are learned 1x1 projections
    of the query's own e2/e3/e4 features instead of support-prototype distances."""
    def __init__(self):
        super().__init__()
        ch = {"e2": 128, "e3": 256, "e4": 512}
        self.proj = nn.ModuleDict({s: nn.Conv2d(ch[s], 2, 1) for s in PROTO_SCALES})
        self.net = nn.Sequential(nn.Conv2d(6, 32, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 1, 1))
    def forward(self, f):
        s = torch.cat([F.interpolate(self.proj[n](f[n]), size=(SCORE_HW, SCORE_HW), mode="bilinear",
                                     align_corners=False) for n in PROTO_SCALES], 1)
        return F.interpolate(self.net(s), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

sf_dec = SupportFreeDecoder().to(DEVICE)
n_sf = sum(p.numel() for p in sf_dec.parameters()); n_fu = sum(p.numel() for p in fusion.parameters())
print(f"parameters: support-free decoder {n_sf} | fusion decoder {n_fu} (difference = the 1x1 projections)")

# ---------------------------------------------- training on the identical Phase-2 stream
opt = torch.optim.AdamW(sf_dec.parameters(), lr=LR_DEC, weight_decay=1e-4)

def run_episodes(n_ep, support_pool, query_pool, train, rng):
    sm = NestedEpisodeSampler(support_pool, query_pool, seed=int(rng.randint(1e9)))
    dices = []
    for _ in range(n_ep):
        sup_rows, que_rows = sm.sample()
        _, _, que_x, que_m = episode_tensors(sup_rows, que_rows, aug=train, rng=rng)  # supports loaded for RNG parity
        _ = int(rng.choice(K_LIST))                                                    # RNG parity with Phase 2
        que_x, que_m = que_x.to(DEVICE), que_m.to(DEVICE)
        qf = encode(que_x)
        if train:
            sf_dec.train(); logits = sf_dec(qf); loss = seg_loss(logits, que_m)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        else:
            sf_dec.eval()
            with torch.no_grad():
                logits = sf_dec(qf)
        pred = (torch.sigmoid(logits) >= PRED_THRESH).cpu().numpy(); gt = que_m.cpu().numpy()
        dices += [dice_np(pred[i, 0], gt[i, 0]) for i in range(len(pred))]
    return float(np.mean(dices))

rng_tr = np.random.RandomState(SEED + 1)
best, best_ep, wait, hist = -1.0, -1, 0, []
for ep in range(1, EPOCHS + 1):
    tr = run_episodes(N_EP_TRAIN, POOLS["tn3k_train"], POOLS["tn3k_train"], True, rng_tr)
    va = run_episodes(N_EP_VAL, POOLS["tn3k_train"], POOLS["tn3k_val"], False, rng_tr)
    hist.append({"epoch": ep, "train_dice": tr, "val_dice": va})
    if va > best + 1e-4:
        best, best_ep, wait = va, ep, 0
        torch.save({"decoder": sf_dec.state_dict(), "epoch": ep, "val_dice": va}, OUT / "best_support_free_decoder.pt")
    else:
        wait += 1
    print(f"ep {ep:02d} | train dice {tr:.4f} | val dice {va:.4f}{'  *' if wait == 0 else ''} ({T})", flush=True)
    if wait >= PATIENCE:
        break
pd.DataFrame(hist).to_csv(OUT / "training_history.csv", index=False)
sf_dec.load_state_dict(torch.load(OUT / "best_support_free_decoder.pt", map_location=DEVICE)["decoder"]); sf_dec.eval()
print(f"best val Dice {best:.4f} @ epoch {best_ep} (Phase-2 fusion decoder: 0.7776 @ 13)")

# ----------------------------------------- evaluation on the identical Phase-2 episodes
conditions = [("TN3K test | in-domain", POOLS["tn3k_train"], POOLS["tn3k_test"]),
              ("TNUI test | zero-shot", POOLS["tn3k_train"], POOLS["tnui_test"]),
              ("DDTI | zero-shot", POOLS["tn3k_train"], POOLS["ddti_all"])]
rows = []
for name, sp, qp in conditions:
    sm = NestedEpisodeSampler(sp, qp, seed=SEED)
    for ep_i in range(N_EP_EVAL):
        sup_rows, que_rows = sm.sample()
        sup_x, sup_m, que_x, que_m = episode_tensors(sup_rows, que_rows)
        sup_x, sup_m, que_x = sup_x.to(DEVICE), sup_m.to(DEVICE), que_x.to(DEVICE)
        gt = que_m.numpy()
        with torch.no_grad():
            sf_, qf = encode(sup_x), encode(que_x)
            protos = {n: masked_proto(sf_[n][:20], sup_m[:20]) for n in PROTO_SCALES}
            pa = torch.sigmoid(F.interpolate(fusion(score_maps(qf, protos)), size=(IMG_SIZE, IMG_SIZE),
                                             mode="bilinear", align_corners=False)) >= PRED_THRESH
            pb = torch.sigmoid(sf_dec(qf)) >= PRED_THRESH
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                lc, _ = trunk(que_x)
            pc = torch.sigmoid(lc.float()) >= PHASE1_THR
        pa, pb, pc = pa.cpu().numpy(), pb.cpu().numpy(), pc.cpu().numpy()
        for qi in range(len(gt)):
            rows.append({"condition": name, "episode": ep_i, "query_id": que_rows.iloc[qi]["id"],
                         "A_fusion_K20": dice_np(pa[qi, 0], gt[qi, 0]),
                         "B_support_free": dice_np(pb[qi, 0], gt[qi, 0]),
                         "C_phase1_unet": dice_np(pc[qi, 0], gt[qi, 0])})
    print(f"  evaluated {name} ({T})", flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT / "episode_query_dice.csv", index=False)

out, rngb = [], np.random.RandomState(SEED)
for name, g in R.groupby("condition", sort=False):
    em = g.groupby("episode")[["A_fusion_K20", "B_support_free", "C_phase1_unet"]].mean()
    gate(abs(em.A_fusion_K20.mean() - LOCKED_A[name]) <= 0.003,
         f"{name}: fusion K=20 Dice {em.A_fusion_K20.mean():.4f} reproduces locked {LOCKED_A[name]}")
    idx = [rngb.randint(0, len(em), len(em)) for _ in range(2000)]
    rec = {"condition": name, "n_episodes": len(em)}
    for col in em:
        bs = [em[col].to_numpy()[i].mean() for i in idx]
        rec[col] = round(em[col].mean(), 4); rec[col + "_CI"] = f"{np.percentile(bs, 2.5):.4f}..{np.percentile(bs, 97.5):.4f}"
    for b in ["B_support_free", "C_phase1_unet"]:
        d = (em[b] - em.A_fusion_K20).to_numpy(); bs = np.array([d[i].mean() for i in idx])
        rec[f"{b}_minus_A"] = round(d.mean(), 4); rec[f"{b}_minus_A_CI"] = f"{np.percentile(bs, 2.5):.4f}..{np.percentile(bs, 97.5):.4f}"
        rec[f"{b}_minus_A_p"] = min(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1.0)
    out.append(rec)
S = pd.DataFrame(out); S.to_csv(OUT / "support_free_vs_fusion_episodic.csv", index=False)
pd.set_option("display.width", 250)
print("\nEpisode-level Dice on identical queries (post-review, not pre-registered):\n", S.T.to_string())

# ------------------------------------------- full-pool per-image Dice (B and C; C gated)
pool_rows = []
for name, pool in [("TN3K test", POOLS["tn3k_test"]), ("TNUI test", POOLS["tnui_test"]), ("DDTI", POOLS["ddti_all"])]:
    db, dc = [], []
    for i0 in range(0, len(pool), 16):
        items = [load_item(r) for _, r in pool.iloc[i0:i0 + 16].iterrows()]
        x = torch.stack([to_tensor(i) for i, _ in items]).to(DEVICE)
        with torch.no_grad():
            pb = (torch.sigmoid(sf_dec(encode(x))) >= PRED_THRESH).cpu().numpy()
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                lc, _ = trunk(x)
            pc = (torch.sigmoid(lc.float()) >= PHASE1_THR).cpu().numpy()
        for k, (_, m) in enumerate(items):
            db.append(dice_np(pb[k, 0], m)); dc.append(dice_np(pc[k, 0], m))
    ok = abs(np.mean(dc) - LOCKED_C_POOL[name]) <= 0.005
    print(f"  [{'GATE OK' if ok else 'NOTE: differs'}] {name}: Phase-1 U-Net single-view {np.mean(dc):.4f} (Table S2 {LOCKED_C_POOL[name]})")
    pool_rows.append({"pool": name, "n": len(pool), "support_free_dice": round(np.mean(db), 4),
                      "phase1_unet_dice": round(np.mean(dc), 4), "phase1_locked_S2": LOCKED_C_POOL[name]})
P = pd.DataFrame(pool_rows); P.to_csv(OUT / "full_pool_dice.csv", index=False)
print(P.to_string(index=False))
save_json({"source_training_masks": src_masks, "support_free_params": n_sf, "fusion_params": n_fu,
           "best_val_dice": best, "best_epoch": best_ep, "runtime": str(T)}, OUT / "summary.json")
print(f"\ndone in {T} -> {OUT}")
