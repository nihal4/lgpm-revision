"""Download the large artifacts (checkpoints, big feature banks) from Hugging Face into
artifacts/<phase>/ and verify each file's SHA-256 against artifacts/large_files.json.

usage:
  python scripts/fetch_large_files.py                 # required files only
  python scripts/fetch_large_files.py --all           # also the optional ones
  python scripts/fetch_large_files.py --from-dir DIR  # copy from a local folder instead
"""
import argparse, hashlib, json, shutil, sys, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
M = json.loads((REPO / "artifacts" / "large_files.json").read_text())


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--all", action="store_true"); ap.add_argument("--from-dir")
ap.add_argument("--hf-repo", default=M["hf_repo"])
a = ap.parse_args()
if not a.from_dir and "REPLACE" in a.hf_repo:
    sys.exit("Set hf_repo in artifacts/large_files.json (or pass --hf-repo user/repo).")
bad = []
for f in M["files"]:
    if not (a.all or f["required_by_revision"]):
        continue
    dst = REPO / f["dest"] / f["name"]; dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == f["bytes"] and sha256(dst) == f["sha256"]:
        print(f"  ok (cached)  {f['name']}"); continue
    if a.from_dir:
        shutil.copy(Path(a.from_dir) / f["name"], dst)
    else:
        url = M["url_template"].format(hf_repo=a.hf_repo, name=f["name"])
        print(f"  downloading  {f['name']} ({f['bytes'] / 1e6:.0f} MB) from {url}", flush=True)
        urllib.request.urlretrieve(url, dst)
    ok = dst.stat().st_size == f["bytes"] and sha256(dst) == f["sha256"]
    print(f"  {'verified' if ok else 'SHA-256 MISMATCH'}  {f['name']}")
    if not ok:
        bad.append(f["name"])
sys.exit(f"checksum failures: {bad}" if bad else 0)
