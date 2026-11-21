"""Survey summary: from single-photo checks to a site-level bleaching indicator (CPECOG2 Group 2).

demo.py answers "is this photo bleached?". A reef monitor needs "which site needs attention, and
which photos should an expert check first?". This script classifies every photo from a survey and
reports, for each site:

  - the share of photos flagged as bleached, with a 95% confidence interval (Wilson), so a site with
    few photos is not over-read;
  - how many photos are uncertain (P(bleached) close to the cutoff), which an expert should review;
  - a review list of all photos, most likely bleached first.

Sites come from a sites CSV when one is given (columns: photo, site; one row per photo file name),
otherwise from the subfolders of the survey folder (survey/SiteA/*.jpg, survey/SiteB/*.jpg, ...). Photos
directly in the survey folder count as one site named after the folder; photos missing from a sites CSV
are reported under "Unassigned".

The rate is the share of PHOTOS flagged, not the share of coral cover that is bleached; it is a
screening indicator for deciding where to send experts, and a trained reef monitor makes the final call.

  python survey.py demo_photos
  python survey.py surveys/2026-11 --out survey_report --top 15
  python survey.py demo_photos --sites-csv demo_survey_sites.csv     # all photos in one folder, sites from the CSV
"""
import argparse
import math
import os

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError

from demo import DEFAULT_WEIGHTS, default_threshold, list_images, load_model, pad_square, spec_note

UNCERTAIN_MARGIN = 0.15   # photos with |P(bleached) - cutoff| below this are marked uncertain


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k flagged photos out of n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def classify(clf, items, threshold):
    """items: (site, path, display_name). Returns one row per readable photo; unreadable files are reported."""
    rows, skipped = [], []
    torch.set_grad_enabled(False)
    for site, path, name in items:
        try:
            im = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError, ImportError, ValueError):
            skipped.append(name)
            continue
        probs = clf.predict(clf.to_input(pad_square(im)[0]))
        p_b = float(probs[clf.bleached])
        rows.append(dict(site=site, photo=name, path=path, p_bleached=round(p_b, 4),
                         prediction="bleached" if p_b >= threshold else "healthy",
                         uncertain=abs(p_b - threshold) < UNCERTAIN_MARGIN, note=spec_note(im)))
    return pd.DataFrame(rows), skipped


def summarize(photos):
    """One row per site, sorted so the site that most needs attention comes first."""
    out = []
    for site, g in photos.groupby("site", sort=False):
        n, k = len(g), int((g["prediction"] == "bleached").sum())
        lo, hi = wilson(k, n)
        out.append(dict(site=site, photos=n, flagged_bleached=k, bleaching_rate=k / n, ci95_low=lo, ci95_high=hi,
                        mean_p_bleached=g["p_bleached"].mean(), uncertain=int(g["uncertain"].sum()),
                        outside_specs=int((g["note"] != "").sum())))
    return pd.DataFrame(out).sort_values(["bleaching_rate", "mean_p_bleached"], ascending=False).reset_index(drop=True)


def review_list(photos):
    """All photos, most likely bleached first: the order an expert should check them in."""
    r = photos.sort_values("p_bleached", ascending=False).reset_index(drop=True)
    r.insert(0, "rank", range(1, len(r) + 1))
    return r


def report_text(sites, review, threshold, top=10, skipped=()):
    """Plain-text (Markdown) report for the terminal, the GUI and report.md."""
    n, k = int(sites["photos"].sum()), int(sites["flagged_bleached"].sum())
    lines = [f"Survey: {n} photos from {len(sites)} site(s); {k} flagged bleached ({k / max(n, 1):.0%}). "
             f"Decision rule: bleached when P(bleached) >= {threshold:.3f}.", "",
             "| Site | Photos | Flagged bleached | Bleaching rate (95% CI) | Mean P(bleached) | Uncertain | Outside specs |",
             "| :-- | --: | --: | :-- | --: | --: | --: |"]
    for _, s in sites.iterrows():
        lines.append(f"| {s['site']} | {s['photos']} | {s['flagged_bleached']} | {s['bleaching_rate']:.0%} "
                     f"({s['ci95_low']:.0%} to {s['ci95_high']:.0%}) | {s['mean_p_bleached']:.2f} | {s['uncertain']} | {s['outside_specs']} |")
    lines += ["", f"Review first (top {min(top, len(review))} by P(bleached)):", "",
              "| Rank | Site | Photo | P(bleached) | Prediction | Note |", "| --: | :-- | :-- | --: | :-- | :-- |"]
    for _, r in review.head(top).iterrows():
        note = "; ".join(x for x in ("uncertain" if r["uncertain"] else "", r["note"]) if x)
        lines.append(f"| {r['rank']} | {r['site']} | {r['photo']} | {r['p_bleached']:.3f} | {r['prediction']} | {note} |")
    lines += ["", f"Uncertain = P(bleached) within {UNCERTAIN_MARGIN} of the cutoff; an expert should check these. "
              "The rate is the share of photos flagged, not the share of coral cover bleached. "
              "Screening aid only: a trained reef monitor makes the final call."]
    if skipped:
        lines.append(f"Skipped {len(skipped)} unreadable file(s): {', '.join(skipped[:5])}{' ...' if len(skipped) > 5 else ''}")
    return "\n".join(lines)


PHOTO_COLS = ("photo", "file", "filename", "file_name", "image", "path")


def load_site_map(csv_path):
    """{photo file name (lower case): site} from a CSV with a photo column and a site column."""
    df = pd.read_csv(csv_path)
    cols = {c.strip().lower(): c for c in df.columns}
    pc = next((cols[c] for c in PHOTO_COLS if c in cols), None)
    sc = cols.get("site")
    if pc is None or sc is None:
        raise ValueError(f"{os.path.basename(csv_path)} needs a 'photo' column and a 'site' column (found: {list(df.columns)})")
    return {os.path.basename(str(p)).strip().lower(): str(st).strip() for p, st in zip(df[pc], df[sc]) if pd.notna(p) and pd.notna(st)}


def items_from_csv(csv_path, folder):
    """Photos listed in a sites CSV, found on disk: as written, or inside `folder`. Returns (items, missing names)."""
    df = pd.read_csv(csv_path)
    cols = {c.strip().lower(): c for c in df.columns}
    pc = next((cols[c] for c in PHOTO_COLS if c in cols), None)
    sc = cols.get("site")
    if pc is None or sc is None:
        raise ValueError(f"{os.path.basename(csv_path)} needs a 'photo' column and a 'site' column (found: {list(df.columns)})")
    items, missing = [], []
    for ph, st in zip(df[pc], df[sc]):
        if pd.isna(ph):
            continue
        ph, st = str(ph).strip(), (str(st).strip() if pd.notna(st) else "Unassigned")
        for cand in (ph, os.path.join(folder, ph), os.path.join(folder, os.path.basename(ph))):
            if os.path.isfile(cand):
                items.append((st, cand, os.path.basename(ph)))
                break
        else:
            missing.append(ph)
    return items, missing


def assign_sites(items, site_map):
    """Replace each item's site with the one in site_map (by file name); photos not listed become 'Unassigned'."""
    return [(site_map.get(os.path.basename(path).lower(), "Unassigned"), path, name) for _, path, name in items]


def survey_items(folder):
    """(site, path, display name) for every image; the site is the first subfolder, else the folder's own name."""
    root = os.path.abspath(folder)
    items = []
    for p in list_images(folder):
        rel = os.path.relpath(os.path.abspath(p), root)
        parts = rel.split(os.sep)
        site = parts[0] if len(parts) > 1 else os.path.basename(root.rstrip(os.sep)) or "survey"
        items.append((site, p, rel))
    return items


def main():
    ap = argparse.ArgumentParser(description="Site-level bleaching summary and review list for a folder of survey photos.")
    ap.add_argument("folder", help="survey folder; each subfolder is one site")
    ap.add_argument("--weights", help="model weights (default: the final model in results/)")
    ap.add_argument("--threshold", type=float, help="P(bleached) cutoff (default: the validation-selected cutoff saved with the model, else 0.5)")
    ap.add_argument("--out", default="survey_report", help="folder for site_summary.csv, review_list.csv and report.md")
    ap.add_argument("--top", type=int, default=10, help="how many photos to list in the review-first table")
    ap.add_argument("--sites-csv", help="CSV with photo and site columns; overrides the subfolder sites")
    args = ap.parse_args()

    weights = args.weights or next((w for w in DEFAULT_WEIGHTS if os.path.exists(w)), None)
    if not weights or not os.path.exists(weights):
        raise SystemExit("No weights found. Put the final model in results/run3/ or results/run2/, or pass --weights path/to/model.pt")
    threshold = args.threshold if args.threshold is not None else default_threshold(weights)[0]
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    clf = load_model(weights)

    out_dir = os.path.abspath(args.out) + os.sep
    items = [it for it in survey_items(args.folder) if not os.path.abspath(it[1]).startswith(out_dir)]
    if args.sites_csv:
        items = assign_sites(items, load_site_map(args.sites_csv))
    photos, skipped = classify(clf, items, threshold)
    if photos.empty:
        raise SystemExit("No readable photos in " + args.folder)
    sites, review = summarize(photos), review_list(photos)
    text = report_text(sites, review, threshold, args.top, skipped)

    os.makedirs(args.out, exist_ok=True)
    sites.round(4).to_csv(os.path.join(args.out, "site_summary.csv"), index=False)
    review.drop(columns=["path"]).to_csv(os.path.join(args.out, "review_list.csv"), index=False)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("# Coral bleaching survey summary\n\n" + text + "\n")
    print(text)
    print(f"\nSaved site_summary.csv, review_list.csv and report.md in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
