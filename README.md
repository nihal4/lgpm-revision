# LG-PM — reproducible re-analyses for the major revision

Re-analyses requested by the reviewers of *"Lesion-grounded prototype meta-learning for
few-shot thyroid ultrasound"* (JIMH, major revision, due 20 Nov 2026).

**Rule followed by every script:** first reproduce the locked published values from the
original artifacts (printed as `[GATE OK]`). If any value is not reproduced, the script
stops with `[GATE FAIL]` and computes nothing new. All new analyses are post-review and
are labelled as not pre-registered.

| Script | Device / time (T4) | Reviewer point | What it adds |
|---|---|---|---|
| `scripts/s04_ddti_label_profile.py` | CPU, ~2 min | R2.7 | Which evidence (literal word, diagnosis code, TI-RADS) decided each of the 189 DDTI labels |
| `scripts/s05_cpu_analyses.py` | CPU, ~3 min | R2.2, R2.3, R2.5 | External specificity + counts on ThyUS2Path / TNUI / DDTI; 20-seed head replicates; referral workload vs random referral |
| `scripts/s01_thyus2path_all_models.py` | GPU, ~15 min | R2.2, R1.2 | ThyUS2Path operating points for all six models incl. ERM and JOINT (their probabilities were never saved) |
| `scripts/s02_xai_controls.py` | GPU, ~10 min | R2.4, R1.7 | True R1p-ELA in-box energy, uniform predicted-mask map (= published K1), Grad-CAM × predicted mask |
| `scripts/s03_support_free_decoder.py` | GPU, ~60–80 min | R2.1, R1.3 | Capacity-matched support-free decoder on the frozen encoder, evaluated on the identical Phase-2 episodes |

`lgpm_rev/common.py` holds the frozen machinery copied from the phase notebooks
(letterboxing, ResNet34-UNet, fusion decoder, prototypes, masked pooling, spec-0.90
threshold rule). `scripts/cpu/` holds the stand-alone CPU analyses that `s05` runs.

## Run on Kaggle

1. Push this repo to GitHub (public, or private with a token).
2. Upload `notebooks/lgpm_revision_kaggle.ipynb` to Kaggle; set **GPU T4 x2** and **Internet On**.
3. Attach the datasets listed in the notebook's first cell; edit `REPO_URL`.
4. Run all. Download `revision_outputs.zip` from the Output panel.

## Run locally (CPU parts)

```bash
pip install -r requirements.txt
LGPM_INPUT_ROOTS=/path/to/downloaded/outputs LGPM_OUT=./revision_outputs python scripts/s05_cpu_analyses.py
```

## Findings so far (from the CPU analyses, 11 Nov 2026)

- On ThyUS2Path the frozen R1p threshold gives sensitivity 0.159 at **specificity 0.868** (not 0.90),
  and sensitivity − FPR is +0.027 (95% CI −0.022 to +0.076): no better than chance at that point.
- Every external operating-point contrast (P5, F5, E5, J1) follows where each model's
  threshold landed after the shift (external specificity 0.45–0.87), not discrimination.
- 20 retrained R1p heads: TN5000-test AUROC 0.8922 ± 0.0022; ThyUS2Path behaviour is the
  same for every replicate, so the unarchived Phase-3 head does not change any conclusion.
- Confidence-based referral removes 86% (JOINT-g) / 76% (R1p) of TN5000 malignant misses at
  50% coverage, against ~50% for random referral of the same size.
