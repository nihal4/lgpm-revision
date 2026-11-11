# Artifacts

Every output the original phases saved, organised by phase. Small files are in git. The
8 large files listed in `large_files.json` (checkpoints and the two biggest feature banks)
are hosted on Hugging Face and placed into these folders by
`python scripts/fetch_large_files.py`, which verifies each SHA-256.

| Folder | Key contents |
|---|---|
| `phase0/` | clean manifests and SHA-256 hashes for TN3K / TNUI-2021 / DDTI, TN5000 XML manifest |
| `phase1/` | `best_resnet34_seed42.pt`* , per-case and aggregate segmentation metrics, threshold sweep |
| `phase2/` | `best_fusion_decoder.pt`, episode metrics, summary |
| `phase3/` | `features_{train*,val,test}.npz`, `linear_heads_results.json`, `erm_resnet34_tn5000.pt`*, `erm_resnet34_probs.json` |
| `phase3c/` | `linear_heads_retrained.json` (the head used by every external analysis), TNUI-test bank |
| `phase3d/` | TNUI-val bank, pooled tables |
| `phase3e/` | DDTI and TNUI-train banks, tables |
| `phase4/` | `joint_{g,img}_trunk.pt`*, TN5000 and external probabilities, selective-prediction tables |
| `phase5/` | `phase5_bank_thyus2path.npz`*, case manifest, main table, family J, audit report |
| `phase6/` | TN5000 and ThyUS-600 panels, family K, faithfulness, stability, cascade, `phase6_montage_store.pkl`* |

\* fetched from Hugging Face.
