"""Rebuild the image->case mapping for the Phase-5 ThyUS2Path bank.

Bank rows follow manifest5 sorted by full image path, so all batch1 images come
before batch2. Within a batch, rows are sorted by file name; the batch boundary is
the first place where the stem order goes backwards.
"""
import sys
import numpy as np
import pandas as pd

bank_path, manifest_path, out_path = sys.argv[1:4]
z = np.load(bank_path, allow_pickle=True)
ids = z["ids"].astype(str)
labels = z["labels"].astype(int)
cases = pd.read_csv(manifest_path)

# full-path sort order within a batch == sort order of "<stem><ext>" names; use the stems
breaks = [i for i in range(1, len(ids)) if ids[i] < ids[i - 1]]
assert len(breaks) == 1, f"expected exactly one batch boundary, found {len(breaks)}"
b = breaks[0]
batch = np.where(np.arange(len(ids)) < b, "batch1_image", "batch2_image")
prefix = np.array([s.split("_")[0] for s in ids])
case = np.char.add(np.char.add(batch.astype(str), ":"), prefix)

lab = cases.set_index("case")["label"]
missing = sorted(set(case) - set(lab.index))
assert not missing, f"cases not in manifest: {missing[:5]}"
mismatch = int((lab.loc[case].to_numpy() != labels).sum())
assert mismatch == 0, f"{mismatch} image labels disagree with case labels"

df = pd.DataFrame({"row": np.arange(len(ids)), "stem": ids, "batch": batch,
                   "case": case, "label": labels})
per_case = df.groupby("case").agg(n=("row", "size"), label=("label", "first"))
excluded = sorted(set(lab.index) - set(per_case.index))
print(f"boundary at row {b}: batch1 {b} images, batch2 {len(ids) - b} images")
print(f"cases in bank: {len(per_case)} (malignant {int(per_case.label.sum())}); "
      f"excluded vs manifest: {len(excluded)} -> {excluded}")
print("per-batch cases:", df.drop_duplicates("case").batch.value_counts().to_dict())
df.to_csv(out_path, index=False)
