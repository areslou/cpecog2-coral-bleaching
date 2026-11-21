"""Copy the held-out test photos into demo_photos/ for the live demo (CPECOG2 Group 2).

The model never trained on or tuned with these images, so they are a fair demo. The Kaggle
dataset is downloaded with kagglehub (internet needed once); each photo is saved as
<true label>_<file name>, so the audience can compare the true label with the prediction.

  python get_demo_photos.py                      # test photos -> demo_photos/
  python get_demo_photos.py --source path/to/dataset/folder   # skip the download
  python get_demo_photos.py --sites              # also make 3 example sites: demo_survey_sites.csv + demo_survey/

The example sites are simulated: test photos mixed in known proportions, so the survey ranking can be checked.
"""
import argparse
import glob
import os
import random
import shutil

import pandas as pd

SPLIT_FILES = ["results/run3/splits_v2.csv", "results/run2/splits_v2.csv", "data/splits.csv"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser(description="Copy the held-out test photos into a folder for the demo.")
    ap.add_argument("--source", help="folder with the Kaggle images (default: download with kagglehub)")
    ap.add_argument("--out", default="demo_photos")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--sites", action="store_true", help="also build demo_survey/ with 3 simulated sites (known bleached share)")
    args = ap.parse_args()

    existing = glob.glob(os.path.join(args.out, "*_*.*"))
    if args.sites and args.source is None and len(existing) >= 100:
        print(f"Using the {len(existing)} photos already in {args.out}")
        build_sites(args.out)
        return

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
    if args.sites:
        build_sites(args.out)


SITES = [("Site_A", 42, 8), ("Site_B", 20, 21), ("Site_C", 5, 35)]   # (name, bleached, healthy): all 67 + 64 test photos


def build_sites(photo_dir, survey_dir="demo_survey", csv_path="demo_survey_sites.csv", seed=42):
    """Simulated survey from the test photos in known proportions, two ways: a sites CSV (photo, site) for one-upload
    use with all of demo_photos/, and one folder per site under demo_survey/."""
    files = sorted(glob.glob(os.path.join(photo_dir, "*")))
    pools = {lab: [f for f in files if os.path.basename(f).startswith(lab + "_")] for lab in ("bleached", "healthy")}
    rng = random.Random(seed)
    for v in pools.values():
        rng.shuffle(v)
    need = {"bleached": sum(b for _, b, _ in SITES), "healthy": sum(h for _, _, h in SITES)}
    if any(len(pools[k]) < need[k] for k in need):
        raise SystemExit(f"Not enough photos in {photo_dir} to build the sites (need {need}).")
    shutil.rmtree(survey_dir, ignore_errors=True)
    rows = []
    print(f"Simulated survey in {os.path.abspath(survey_dir)} (true bleached share per site):")
    for name, nb, nh in SITES:
        d = os.path.join(survey_dir, name); os.makedirs(d)
        for f in pools["bleached"][:nb] + pools["healthy"][:nh]:
            shutil.copy(f, d)
            rows.append(dict(photo=os.path.basename(f), site=name))
        pools["bleached"], pools["healthy"] = pools["bleached"][nb:], pools["healthy"][nh:]
        print(f"  {name}: {nb + nh} photos, {nb} truly bleached ({nb / (nb + nh):.0%})")
    pd.DataFrame(rows).sort_values(["site", "photo"]).to_csv(csv_path, index=False)
    print(f"Sites CSV: {os.path.abspath(csv_path)} ({len(rows)} photos)")
    print(f"Next: python survey.py {survey_dir}   or   python survey.py {photo_dir} --sites-csv {csv_path}")
    print(f"GUI: in Survey summary, upload all photos in {photo_dir} together with {csv_path}")


if __name__ == "__main__":
    main()
