# Results of the revision analyses

`kaggle_run_2026-11-12/` holds the outputs of `notebooks/lgpm_revision_kaggle.ipynb` run on Kaggle
(T4 x2) at commit 57e000a. All 38 reproduction gates passed (see each `log.txt`).

| Folder | Reviewer point | Headline |
|---|---|---|
| `s01_thyus2path_all_models/` | R2.2, R1.2 | At frozen source thresholds, external specificity 0.87-0.997; sensitivity - FPR includes 0 for all six models |
| `s02_xai_controls/` | R2.4, R1.7 | R1p-ELA ratio 20.5 vs Grad-CAM 3.6, but vs mask-restricted Grad-CAM 24.4; published K1 (41.6) is the predicted mask itself |
| `s03_support_free_decoder/` | R2.1, R1.3 | Support-free decoder beats fusion K=20 by +0.048 to +0.073 Dice on identical queries (2,447 source masks) |
| `s04_ddti_label_profile/` | R2.7 | All 189 DDTI labels decided by TI-RADS alone (145 TR4-5 / 44 TR1-3) |
| `s05_cpu_analyses/` | R2.2, R2.3, R2.5 | TNUI/DDTI specificity, 20-seed head replicates, referral vs random |
