"""Copy the held-out test photos into demo_photos/ for the live demo (CPECOG2 Group 2).

The model never trained on or tuned with these images, so they are a fair demo. The Kaggle
dataset is downloaded with kagglehub (internet needed once); each photo is saved as
<true label>_<file name>, so the audience can compare the true label with the prediction.

  python get_demo_photos.py                      # test photos -> demo_photos/
  python get_demo_photos.py --source path/to/dataset/folder   # skip the download
"""
import argparse
import glob
import os
import shutil

import pandas as pd

SPLIT_FILES = ["results/run3/splits_v2.csv", "results/run2/splits_v2.csv", "data/splits.csv"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser(description="Copy the held-out test photos into a folder for the demo.")
    ap.add_argument("--source", help="folder with the Kaggle images (default: download with kagglehub)")
    ap.add_argument("--out", default="demo_photos")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    split_file = next((f for f in SPLIT_FILES if os.path.exists(f)), None)
    if split_file is None:
        raise SystemExit("No split file found. Run this from the repository folder.")
    s = pd.read_csv(split_file)
    wanted = {(os.path.basename(p), lab) for p, lab, sp in zip(s["path"], s["label"], s["split"]) if sp == args.split}

    src = args.source
    if src is None:
        import kagglehub
        print("Downloading the Kaggle dataset (about 26 MB, one time only)...")
        src = kagglehub.dataset_download("vencerlanz09/healthy-and-bleached-corals-image-classification")

    def label_of(p):
        parent = os.path.basename(os.path.dirname(p)).lower()
        return "bleached" if "bleach" in parent else "healthy" if "health" in parent else None

    os.makedirs(args.out, exist_ok=True)
    n = {"bleached": 0, "healthy": 0}
    for p in sorted(glob.glob(os.path.join(src, "**", "*"), recursive=True)):
        lab = label_of(p)
        if p.lower().endswith(IMG_EXTS) and (os.path.basename(p), lab) in wanted:
            shutil.copy(p, os.path.join(args.out, f"{lab}_{os.path.basename(p)}"))
            n[lab] += 1
    print(f"Copied {sum(n.values())} {args.split} photos ({n['bleached']} bleached, {n['healthy']} healthy) "
          f"from {split_file} into {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
