"""s05 - CPU re-analyses from saved predictions (A1, A1b, A1c, A3, X2).  CPU, ~3 min.

These were first run outside Kaggle on the downloaded artifacts; this runner repeats them
on Kaggle from the attached datasets so the revision is fully reproducible. Each step
gates on the locked published values before reporting.

  A1   ThyUS2Path operating points, linear heads (bank + 3c heads)
  A1b  TNUI test / val / pool operating points (ERM, R1p, JOINT-g, JOINT-img)
  A1c  DDTI and TNUI-train operating points
  A3   referral workload + random-referral baseline (TN5000-test and TNUI pool)
  X2   20-seed head replicates applied to ThyUS2Path

Attach: Phase_3_classification, phase3_c_, phase3d_external, phase3e_external,
phase4_joint, phase5_thyus2path(_v2).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import subprocess
import numpy as np
import pandas as pd

from lgpm_rev.common import Tee, find, gate, out_dir

OUT = out_dir("s05_cpu_analyses")
Tee(OUT / "log.txt")
CPU = Path(__file__).resolve().parent / "cpu"


def run(script, *args):
    cmd = [sys.executable, "-I", str(CPU / script), *map(str, args)]
    print("\n$", " ".join(cmd[2:]), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-6000:]); print(r.stderr[-3000:]) if r.returncode else None
    if r.returncode:
        raise SystemExit(f"{script} failed")


F = {"bank": find(["phase5_bank_thyus2path.npz"]), "cases": find(["phase5_manifest_unique_cases.csv"]),
     "heads3c": find(["linear_heads_retrained.json"]), "heads3": find(["linear_heads_results.json"]),
     "val": find(["features_val.npz"], exclude=("tnui",)), "train": find(["features_train.npz"]),
     "test": find(["features_test.npz"], exclude=("tnui",)),
     "tnui_test": find(["features_tnui_test.npz"]), "tnui_val": find(["features_tnui_val.npz"]),
     "ddti": find(["features_tnui_ddti.npz"]), "tnui_train": find(["features_tnui_tnuitrain.npz"]),
     "ext": find(["phase4_external_probs.json"]), "joint": find(["phase4_joint_probs_tn5000.json"]),
     "erm": find(["erm_resnet34_probs.json"])}
for k, v in F.items():
    print(f"  {k:10s} {v}")
gate(F["train"].parent == F["val"].parent == F["test"].parent, "TN5000 train/val/test features come from one folder")

# A1 -----------------------------------------------------------------------------
run("map_cases.py", F["bank"], F["cases"], OUT / "thy_image_case_map.csv")
run("a1_external_operating_points.py", F["bank"], OUT / "thy_image_case_map.csv", F["heads3c"], F["val"], OUT / "a1")
a1 = pd.read_csv(OUT / "a1" / "a1_thyus2path_operating_points.csv").set_index("model")
for m, (auc, sens) in {"R0_image": (0.5400, 0.0460), "R1p_lesion": (0.5081, 0.1590), "R2p_lesion_ring": (0.5211, 0.0881)}.items():
    gate(abs(a1.loc[m, "auroc_case"] - auc) < 1e-3 and abs(a1.loc[m, "ext_sens"] - sens) < 1e-3,
         f"A1 {m} reproduces Table S15 (AUROC {a1.loc[m, 'auroc_case']}, sens {a1.loc[m, 'ext_sens']})")

# A1b / A1c ------------------------------------------------------------------------
run("a1b_tnui_operating_points.py", F["ext"], F["joint"], F["erm"], F["heads3c"], F["val"],
    F["tnui_test"], F["tnui_val"], OUT / "a1b")
b = pd.read_csv(OUT / "a1b" / "a1b_tnui_operating_points.csv").set_index(["cohort", "model"])
for (c, m), s in {("TNUI-test", "ERM"): 0.5373, ("TNUI-test", "R1p"): 0.694, ("TNUI-val", "R1p"): 0.6512,
                  ("TNUI pool", "R1p"): 0.6658, ("TNUI pool", "ERM"): 0.5459}.items():
    gate(abs(b.loc[(c, m), "sens"] - s) < 1e-3, f"A1b {c} {m} sens {b.loc[(c, m), 'sens']} (locked {s})")
run("a1c_ddti_tnuitrain_operating_points.py", F["ext"], OUT / "a1b" / "a1b_thresholds.json",
    F["ddti"], F["tnui_train"], OUT / "a1c")
c = pd.read_csv(OUT / "a1c" / "a1c_ddti_tnuitrain_operating_points.csv").set_index(["cohort", "model"])
for (co, m), s in {("DDTI", "ERM"): 0.4828, ("DDTI", "R1p"): 0.3379, ("TNUI-train", "R1p"): 0.6776}.items():
    gate(abs(c.loc[(co, m), "sens"] - s) < 1e-3, f"A1c {co} {m} sens {c.loc[(co, m), 'sens']} (locked {s})")

# A3 ------------------------------------------------------------------------------
J = json.loads(F["joint"].read_text()); L = json.loads(F["heads3"].read_text())
THR = json.loads((OUT / "a1b" / "a1b_thresholds.json").read_text())
a3 = OUT / "a3"; a3.mkdir(exist_ok=True)
np.save(a3 / "y_test.npy", np.load(F["test"], allow_pickle=True)["labels"])
np.save(a3 / "p_jg_test.npy", np.asarray(J["test"]["JOINTg"])); np.save(a3 / "p_r1p_test.npy", np.asarray(L["R1p_lesion"]["probs_test"]))
run("a3_referral_workload.py", "TN5000-test JOINT-g", a3 / "y_test.npy", a3 / "p_jg_test.npy", J["thr"]["JOINTg"], J["temp"]["JOINTg"], a3 / "a3_tn5000_jointg.csv")
run("a3_referral_workload.py", "TN5000-test R1p", a3 / "y_test.npy", a3 / "p_r1p_test.npy", J["thr"]["R1p"], J["temp"]["R1p"], a3 / "a3_tn5000_r1p.csv")
run("a3_referral_workload.py", "TNUI pool JOINT-g", OUT / "a1b" / "y_tnui_pool.npy", OUT / "a1b" / "p_jointg_tnui_pool.npy", J["thr"]["JOINTg"], J["temp"]["JOINTg"], a3 / "a3_tnui_pool_jointg.csv")
run("a3_referral_workload.py", "TNUI pool R1p", OUT / "a1b" / "y_tnui_pool.npy", OUT / "a1b" / "p_r1p_tnui_pool.npy", THR["R1p"], J["temp"]["R1p"], a3 / "a3_tnui_pool_r1p.csv")
s14 = pd.read_csv(a3 / "a3_tn5000_jointg.csv")
gate(np.allclose(s14.sens_retained, [0.5642, 0.6327, 0.7174, 0.8398, 0.8510, 0.8738], atol=1e-4), "A3 reproduces Table S14 (TN5000)")
s14t = pd.read_csv(a3 / "a3_tnui_pool_jointg.csv")
gate(np.allclose(s14t.sens_retained, [0.6122, 0.6780, 0.7643, 0.8406, 0.8602, 0.8974], atol=1e-4), "A3 reproduces Table S14 (TNUI pool)")

# X2 ------------------------------------------------------------------------------
run("x2_head_replicates.py", F["val"].parent, F["bank"], OUT / "thy_image_case_map.csv", 20, OUT / "x2")
print("\nall CPU analyses done ->", OUT)
