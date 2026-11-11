"""s04 - DDTI label provenance (R2.7).  CPU, ~2 min.

Methods 2.2 / Supplement S1.3 describe a fixed precedence (literal words > diagnosis codes
> TI-RADS) and a per-run vote profile. The Phase-3e parser (`ddti_xml_label`) actually
collects votes from every source and labels a case malignant if ANY vote is malignant, and
it printed no profile. This script reruns that exact parser while recording which source
cast each vote, rebuilds the locked 189-case cohort (GATE: 189 cases, 145 malignant), and
reports how every final label was reached, including cases decided only by TI-RADS and
cases whose sources disagree.

Raw data (Kaggle input): TN3K_Data (contains DDTI) and, if Phase 3e used it,
the "DDTI: Thyroid Ultrasound Images" dataset.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re
import collections
import xml.etree.ElementTree as ET
import pandas as pd

from lgpm_rev.common import Tee, find, gate, image_files, out_dir, save_json, sha256_file

OUT = out_dir("s04_ddti_label_profile")
Tee(OUT / "log.txt")

tn3k_root = find(["tn3k-data/Thyroid Dataset/tn3k", "*Thyroid Dataset/tn3k"], kind="dir")
DDTI_ROOT = tn3k_root.parent / "DDTI dataset" / "DDTI"
AUTHOR_ROOT = tn3k_root.parents[2]           # .../<owner>  (Phase-3e: AUTHOR_ROOT / "tn3k-data")
XML_ROOTS = [DDTI_ROOT, AUTHOR_ROOT / "tn3k-data"] + \
            [p for p in Path("/kaggle/input").rglob("ddti-thyroid-ultrasound-images") if p.is_dir()][:1]
print("DDTI root:", DDTI_ROOT, DDTI_ROOT.exists(), "| XML roots:", [str(r) for r in XML_ROOTS])


def ddti_xml_votes(path):
    """Phase-3e ddti_xml_label, unchanged in logic, returning (label, [(source, vote)])."""
    pairs, raw, votes = [], "", []
    try:
        root = ET.parse(path).getroot()
        pairs = [(str(e.tag).lower(), (e.text or "").strip().lower()) for e in root.iter()]
    except Exception:
        try:
            raw = Path(path).read_text(errors="ignore").lower()
        except Exception:
            return None, []
    def add_code(t, src):
        if t.isdigit():
            v = int(t)
            if v == 1:
                votes.append((src, 0))
            elif v == 3:
                votes.append((src, 1))
    if pairs:
        texts = " ".join(t for _, t in pairs if t)
        if "malignant" in texts:
            votes.append(("word:malignant", 1))
        if "benign" in texts:
            votes.append(("word:benign", 0))
        for tag, t in pairs:
            if tag.endswith("diagnosis"):
                add_code(t, "diagnosis_code")
            if ("tirads" in tag or "ti-rads" in tag) and t:
                m = re.match(r"([1-5])[abc]?", t)
                if m:
                    votes.append(("tirads_tag", 1 if m.group(1) in "45" else 0))
        for mm in re.finditer(r"ti-?rads[^0-9]{0,6}([1-5])[abc]?", texts):
            votes.append(("tirads_text", 1 if mm.group(1) in "45" else 0))
    else:
        if "malignant" in raw:
            votes.append(("word:malignant", 1))
        if "benign" in raw:
            votes.append(("word:benign", 0))
        for mm in re.finditer(r"<diagnosis>\s*([0-9]+)", raw, flags=re.I):
            add_code(mm.group(1), "diagnosis_code")
        for mm in re.finditer(r"ti-?rads[^0-9]{0,6}([1-5])[abc]?", raw):
            votes.append(("tirads_text", 1 if mm.group(1) in "45" else 0))
    label = (1 if any(v == 1 for _, v in votes) else 0) if votes else None
    return label, votes


xmls = []
for r_ in XML_ROOTS:
    if Path(r_).exists():
        xmls += sorted(Path(r_).rglob("*.xml"))
info, tags = {}, collections.Counter()
for xp in xmls:                         # later roots overwrite earlier ones, as in Phase 3e
    lab, votes = ddti_xml_votes(xp)
    if lab is not None:
        info[xp.stem] = {"label": lab, "votes": votes, "xml": str(xp)}
    try:
        tags.update(str(e.tag).lower() for e in ET.parse(xp).getroot().iter())
    except Exception:
        pass
print(f"XML files found: {len(xmls)} | labelled stems: {len(info)}")
print("XML tags seen (count):", dict(tags.most_common(25)))

imgs = {p.stem: p for p in image_files(DDTI_ROOT / "1_or_data" / "image")}
msks = {p.stem: p for p in image_files(DDTI_ROOT / "1_or_data" / "mask")}
rows = []
for k in sorted(set(imgs) & set(msks)):
    for ck in [k, k.split("_")[0]]:
        if ck in info:
            rows.append({"id": k, "case_key": ck, "image": str(imgs[k]), **info[ck]}); break
D = pd.DataFrame(rows)
D["sha256"] = D["image"].apply(sha256_file)
D = D.drop_duplicates("sha256", keep="first").reset_index(drop=True)   # Phase-3e within-DDTI dedup
gate(len(D) == 189 and int(D.label.sum()) == 145, f"DDTI cohort rebuilt: n={len(D)}, malignant={int(D.label.sum())} (locked 189/145)")


def describe(votes, label):
    srcs = sorted({s for s, v in votes if v == label})
    against = sorted({s for s, v in votes if v != label})
    kind = ("TI-RADS only" if srcs and all(s.startswith("tirads") for s in srcs)
            else "word/diagnosis code only" if srcs and not any(s.startswith("tirads") for s in srcs)
            else "TI-RADS + word/code")
    return "+".join(srcs), "+".join(against), kind, bool(against)


D[["deciding_sources", "contrary_sources", "decided_by", "conflict"]] = D.apply(
    lambda r: pd.Series(describe(r["votes"], r["label"])), axis=1)
D["votes"] = D["votes"].astype(str)
D.drop(columns=["image", "sha256"]).to_csv(OUT / "ddti_label_provenance_per_case.csv", index=False)
prof = (D.assign(label=D.label.map({1: "malignant", 0: "benign"}))
          .groupby(["label", "decided_by"]).size().unstack(fill_value=0))
print("\nHow each final DDTI label was reached (cases):\n", prof.to_string())
print(f"\nCases whose sources disagree (some vote against the final label): {int(D.conflict.sum())}")
print(D[D.conflict].groupby(["label", "deciding_sources", "contrary_sources"]).size().to_string())
prof.to_csv(OUT / "ddti_label_profile.csv")
save_json({"n": len(D), "malignant": int(D.label.sum()), "profile": prof.to_dict(),
           "conflicts": int(D.conflict.sum()), "n_xml": len(xmls)}, OUT / "summary.json")
print(f"\n-> {OUT}")
