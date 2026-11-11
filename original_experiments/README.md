# Original experiments (as run for the submitted manuscript)

These are the eleven Kaggle notebooks exactly as they produced the submitted results
(outputs were cleared before archiving). They were run in this order; each phase reads the
outputs of earlier phases, which are archived in `../artifacts/<phase>/`.

| Order | Notebook | Produces (archived in) | Paper |
|---|---|---|---|
| 0 | `phase0_dataset_audit.ipynb` | clean manifests, hashes, mask statistics (`artifacts/phase0/`) | 2.2–2.3 |
| 1 | `phase1_baseline_segmentation.ipynb` | supervised ResNet34-UNet `best_resnet34_seed42.pt`, Table S2 (`artifacts/phase1/`) | 3.2 |
| 2 | `phase2_prototype_segmentation.ipynb` | fusion decoder `best_fusion_decoder.pt`, Table 5 / S3 (`artifacts/phase2/`) | 3.3 |
| 3 | `phase3_segmentation_guided_classification.ipynb` | TN5000 feature banks, linear heads (probabilities only), ERM, Tables S4–S8 (`artifacts/phase3/`) | 3.4 |
| 3b | `phase3b_paired_significance.ipynb` | family T (`artifacts/phase3/phase3b_*.csv`) | 3.4 |
| 3c | `phase3c_external_classification.ipynb` | **retrained** linear heads `linear_heads_retrained.json`, TNUI-test (`artifacts/phase3c/`) | 3.5 |
| 3d | `phase3d_tnui_val_external.ipynb` | TNUI-val + pooled family P (`artifacts/phase3d/`) | 3.5 |
| 3e | `phase3e_ddti_tnuitrain_external.ipynb` | DDTI + TNUI-train (`artifacts/phase3e/`) | 3.5 |
| 4 | `phase4_joint_finetune.ipynb` | JOINT-g / JOINT-img, families G/H, selective prediction (`artifacts/phase4/`) | 3.6 |
| 5 | `phase5_thyus2path.ipynb` | ThyUS2Path bank + family J (`artifacts/phase5/`) | 3.7 |
| 6 | `phase6_xai.ipynb` | explanation battery, family K (`artifacts/phase6/`) | 3.8 |

To rerun a phase on Kaggle: attach the raw datasets plus the earlier phases' artifacts
(the notebooks find inputs by file name under `/kaggle/input`). Notes from the revision
audit, kept here so a re-runner is not surprised:

- Phase 3 did not save the linear-head **weights**; Phase 3c retrained them from the cached
  features (test AUROC 0.8889 vs 0.8902). Phases 3d–6 load the 3c heads.
- Phase 5 averages **raw** image probabilities per case (the paper text said "calibrated").
- Phase 6's K1 "R1p localisation ratio" is predicted-mask precision, not ELA energy.
- Phase 3e's DDTI parser takes a malignant vote from any source (no precedence) and prints
  no vote profile; see `scripts/s04_ddti_label_profile.py`.
