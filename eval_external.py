"""External test set (BU + BHD) for the Run 2 final model, with the BHD file names parsed correctly.

Run 2 section 8 matched BHD images by the text "bhd", but the Hugging Face zip names them
"healthy_hbd_ID.png" / "unhealthy_hbd_ID.png", so only the BU images were evaluated. This script
repeats section 8 (same pHash de-duplication, same padding, same YOLO prediction) for both sources.
It only evaluates; it does not train or change the model.

Colab (T4 or CPU), from the repository folder:
  !pip -q install ultralytics==8.4.211 imagehash huggingface_hub scikit-learn pandas
  !python eval_external.py --weights results/run2/final_model_s42.pt
Several weights files (e.g. all three seeds) are reported one by one and as mean ± std.
"""
import argparse
import glob
import json
import os
import re
import shutil
import zipfile

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

NEAR_DUP_T, IMGSZ = 5, 224
SPLITS = "data/splits.csv"   # pHashes of every primary (Kaggle) image, all splits
SPLITS_URL = "https://raw.githubusercontent.com/areslou/cpecog2-coral-bleaching/main/data/splits.csv"
CLASSES = ["healthy", "bleached"]


def phash_of(p):
    try:
        with Image.open(p) as im:
            return str(imagehash.phash(im.convert("RGB")))
    except Exception:
        return None


def hamming(a, b):
    x = np.array([int(h, 16) for h in a], dtype=np.uint64)[:, None] ^ np.array([int(h, 16) for h in b], dtype=np.uint64)[None, :]
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int16)
    pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return pop[np.ascontiguousarray(x).view(np.uint8)].reshape(x.shape[0], x.shape[1], 8).sum(-1).astype(np.int16)


def pad_square(p):
    im = Image.open(p).convert("RGB"); s = max(im.size)
    sq = Image.new("RGB", (s, s), (0, 0, 0)); sq.paste(im, ((s - im.size[0]) // 2, (s - im.size[1]) // 2))
    return sq


def parse(name):
    """Label and source from names like healthy_hbd_12.png, unhealthy_bu_3.png, dead_eilat_T01.jpg."""
    s = name.lower()
    lab = "bleached" if "unhealthy" in s else "healthy" if "healthy" in s else "dead" if "dead" in s else None
    src = "BHD" if ("hbd" in s or "bhd" in s) else "BU" if re.search(r"(?<![a-z])bu(?![a-z])", s) else "EILAT" if "eilat" in s else None
    return lab, src


def metrics(y, p, threshold=0.5):
    yp = (p >= threshold).astype(int)
    return dict(n=len(y), macro_f1=f1_score(y, yp, average="macro"), bleached_recall=recall_score(y, yp, zero_division=0),
                bleached_precision=precision_score(y, yp, zero_division=0), healthy_recall=recall_score(y, yp, pos_label=0, zero_division=0),
                accuracy=accuracy_score(y, yp), roc_auc=roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
                cm=confusion_matrix(y, yp, labels=[0, 1]).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", default=["results/run3/final_model_s42.pt"])
    ap.add_argument("--work", default="/content" if os.path.isdir("/content") else ".", help="folder for the downloaded images")
    ap.add_argument("--out", default="results/run3", help="folder for the result files")
    ap.add_argument("--threshold", type=float, default=0.5, help="decision cutoff for P(bleached); selected on validation only")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    zpath = hf_hub_download("esahit/coral-health-classification", "total_dataset.zip", repo_type="dataset")
    raw = os.path.join(args.work, "ext_raw"); shutil.rmtree(raw, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(raw)
    files = sorted(p for p in glob.glob(os.path.join(raw, "**", "*"), recursive=True)
                   if os.path.isfile(p) and p.lower().endswith((".png", ".jpg", ".jpeg")))
    ext = pd.DataFrame([(p, *parse(os.path.basename(p))) for p in files], columns=["path", "label", "source"])
    print("All external images (rows source, columns label):")
    print(pd.crosstab(ext["source"].fillna("unknown"), ext["label"].fillna("unknown")))
    print("Dataset card: BU 124 healthy / 134 bleached; BHD 537 healthy / 374 bleached / 150 dead; EILAT 280 dead")

    ext = ext[ext["label"].isin(CLASSES) & ext["source"].isin(["BU", "BHD"])].reset_index(drop=True)
    ext["phash"] = ext["path"].map(phash_of)
    ext = ext.dropna(subset=["phash"])
    n_raw = len(ext)
    ext = ext.drop_duplicates("phash").reset_index(drop=True)
    n_within = n_raw - len(ext)
    primary = pd.read_csv(SPLITS if os.path.exists(SPLITS) else SPLITS_URL)["phash"].astype(str).tolist()
    overlap = hamming(ext["phash"], primary).min(1) <= NEAR_DUP_T
    print(f"\nBU + BHD healthy/bleached: {n_raw}; removed {n_within} repeats inside the set and {int(overlap.sum())} matches "
          f"with the Kaggle images (by source: {ext.loc[overlap, 'source'].value_counts().to_dict()})")
    ext = ext[~overlap].reset_index(drop=True)
    print("Evaluated (rows source, columns label):")
    print(pd.crosstab(ext["source"], ext["label"], margins=True))

    sq = os.path.join(args.work, "coral_external_v2"); shutil.rmtree(sq, ignore_errors=True); os.makedirs(sq)
    ext["sq"] = [os.path.join(sq, f"{r.source}_{i:05d}.jpg") for i, r in ext.iterrows()]
    for r in ext.itertuples():
        pad_square(r.path).save(r.sq, quality=95)
    y = (ext["label"] == "bleached").astype(int).to_numpy()
    src = ext["source"].to_numpy()

    rows = []
    for w in args.weights:
        m = YOLO(w)
        b = next(k for k, n in m.names.items() if n == "bleached")
        p = []
        for k in range(0, len(ext), 64):
            p += [float(r.probs.data[b]) for r in m.predict(ext["sq"].tolist()[k:k + 64], imgsz=IMGSZ, verbose=False)]
        p = np.array(p)
        for name, mask in (("BU + BHD", np.ones(len(y), bool)), ("BU", src == "BU"), ("BHD", src == "BHD")):
            if mask.sum():
                rows.append(dict(weights=os.path.basename(w), set=name, **metrics(y[mask], p[mask], args.threshold)))
        ext[f"p_bleached_{os.path.splitext(os.path.basename(w))[0]}"] = p.round(4)

    res = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    res.to_csv(os.path.join(args.out, "external_bu_bhd.csv"), index=False)
    ext.drop(columns=["path", "sq"]).to_csv(os.path.join(args.out, "external_bu_bhd_predictions.csv"), index=False)
    cols = ["n", "macro_f1", "bleached_recall", "bleached_precision", "healthy_recall", "accuracy", "roc_auc"]
    print(f"\nEXTERNAL TEST (BU + BHD), evaluated once with the Run 2 final model (P(bleached) cutoff={args.threshold:.3f})")
    for r in rows:
        print(f"  {r['weights']} | {r['set']:<8} n={r['n']:<4} macro F1 {r['macro_f1']:.3f}, bleached recall {r['bleached_recall']:.3f}, "
              f"accuracy {r['accuracy']:.3f}, ROC AUC {r['roc_auc']:.3f}, confusion matrix (rows true, cols predicted; healthy, bleached) {r['cm']}")
    if len(args.weights) > 1:
        g = res.groupby("set")[cols[1:]].agg(["mean", "std"]).round(3)
        print("\nMean ± std over the weights files:\n", g.to_string())
    print(f"\nSaved {os.path.join(args.out, 'external_bu_bhd.csv')} and external_bu_bhd_predictions.csv")


if __name__ == "__main__":
    main()
