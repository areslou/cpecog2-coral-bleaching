"""Coral bleaching screening demo (CPECOG2 Group 2).

Classifies one reef photo or every photo in a folder as healthy or bleached, prints the
predicted class and its probability, and saves a Grad-CAM heatmap of the regions that drove
each prediction. Runs on a laptop CPU with no internet once the weights file is local.

Examples
  python demo.py photo.jpg
  python demo.py photos/ --out demo_output --csv predictions.csv
  python demo.py photos/ --show                      # also open each heatmap in a window
  python demo.py photos/ --benchmark                 # CPU latency, proposal protocol

The model defaults to results/run10/final_model.json written by notebooks/Group2_Coral_Run10.ipynb:
three YOLOv8s-cls students (seeds 42, 43, 44, distilled from DINOv2) whose P(bleached) values are
averaged, with the cross-validated cutoff saved in that file. --weights also accepts a single .pt file;
the model card saved next to it (<weights>.json) gives the architecture, class order and preprocessing;
without it the type is detected from the file. The tool is a screening aid: a trained person makes the final call.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

# Ultralytics tries to pip-install missing optional packages (e.g. pi-heif) when it meets an unreadable
# file. The demo must run offline and never install anything mid-demo, so auto-install is turned off.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
MIN_SIDE = 100      # px; smaller images are outside the supported conditions (proposal, system specifications)
MIN_DETAIL = 8.0    # std of grayscale pixel values; below this the image is almost uniform (blank, black, or solid colour)


def spec_note(im):
    """Return a warning when the photo is outside the supported conditions; the prediction is still shown."""
    w, h = im.size
    if min(w, h) < MIN_SIDE:
        return f"outside specs: only {w}x{h} px"
    if float(np.asarray(im.convert("L"), dtype=np.float32).std()) < MIN_DETAIL:
        return "outside specs: almost no detail (blank or solid image)"
    return ""
DEFAULT_WEIGHTS = [
    "results/run10/final_model.json",
    "results/run5/final_model_s42.pt",
    "results/run5/final_model_s43.pt",
    "results/run5/final_model_s44.pt",
    "results/run3/final_model_s42.pt",
    "results/run2/final_model_s42.pt",
]


def default_threshold(weights):
    """Decision cutoff on P(bleached) for these weights.

    Run 5 and Run 3 selected their cutoffs on validation only and save them as selected_threshold.csv next to their models.
    The cutoff belongs to the model it was selected for, so it is only used when that file sits in the same
    folder as the weights; any other model uses 0.5 (the plain highest-probability class).
    For an averaged model (a final_model.json that lists several model files), the cutoff saved in that file is used."""
    if weights.lower().endswith(".json"):
        card = json.load(open(weights))
        if card.get("threshold") is not None:
            return float(card["threshold"]), os.path.relpath(weights)
    path = os.path.join(os.path.dirname(os.path.abspath(weights)), "selected_threshold.csv")
    if os.path.exists(path):
        with open(path, newline="") as f:
            row = next(csv.DictReader(f), None)
        if row and row.get("threshold"):
            return float(row["threshold"]), os.path.relpath(path)
    return 0.5, "default"
IMNET_MEAN, IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- models
class SmallCNN(nn.Module):
    """Scratch baseline from the Run 2 notebook (3 convolutional blocks)."""

    def __init__(self, n_classes=2):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class Classifier:
    """Wraps the network with its class order, input size, normalization and Grad-CAM layer."""

    def __init__(self, net, cam_layer, classes, imgsz, mean, std, arch):
        self.net, self.cam_layer, self.classes, self.imgsz, self.arch = net, cam_layer, list(classes), imgsz, arch
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self.bleached = self.classes.index("bleached")
        for p in self.net.parameters():
            p.requires_grad_(True)

    def to_input(self, square):
        """Padded-square PIL image -> normalized (1, 3, imgsz, imgsz) tensor."""
        arr = np.asarray(square.resize((self.imgsz, self.imgsz), Image.BILINEAR), dtype=np.float32) / 255
        return ((torch.from_numpy(arr).permute(2, 0, 1) - self.mean) / self.std)[None]

    def logits(self, x):
        out = self.net(x)
        return out[1] if isinstance(out, (tuple, list)) else out   # Ultralytics Classify returns (softmax, logits) in eval

    def predict(self, x):
        with torch.no_grad():
            return self.logits(x).softmax(1)[0]

    def decide(self, probs, threshold=0.5):
        """Predicted class index: bleached when P(bleached) >= threshold, otherwise the other class."""
        return self.bleached if float(probs[self.bleached]) >= threshold else 1 - self.bleached

    def gradcam(self, x, threshold=0.5, cls=None):
        """Probabilities and a Grad-CAM map (imgsz x imgsz, 0..1) for the predicted class (or for `cls` if given)."""
        store = {}
        hook = self.cam_layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
        try:
            logits = self.logits(x)
        finally:
            hook.remove()
        if cls is None:
            cls = self.decide(logits.detach().softmax(1)[0], threshold)
        a = store["a"]
        grad = torch.autograd.grad(logits[0, cls], a)[0]
        cam = F.relu((grad.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return logits.detach().softmax(1)[0], cam


class Ensemble:
    """Several models with the same input whose P(bleached) values are averaged (Run 7 and Run 10 final models).

    Same interface as Classifier. Grad-CAM comes from the first model, for the class the averaged prediction decides."""

    def __init__(self, members, arch):
        self.members, self.arch = members, arch
        first = members[0]
        self.classes, self.imgsz, self.bleached = first.classes, first.imgsz, first.bleached
        for m in members[1:]:
            if m.imgsz != first.imgsz or m.classes != first.classes:
                sys.exit("The averaged models must share the input size and class order.")

    def to_input(self, square):
        return self.members[0].to_input(square)

    def decide(self, probs, threshold=0.5):
        return self.members[0].decide(probs, threshold)

    def _probs(self, x, member):
        p = member.predict(x)                      # each member's probabilities in this ensemble's class order
        return p if member.classes == self.classes else p[[member.classes.index(c) for c in self.classes]]

    def predict(self, x):
        return torch.stack([self._probs(x, m) for m in self.members]).mean(0)

    def gradcam(self, x, threshold=0.5):
        probs = self.predict(x)
        with torch.enable_grad():
            _, cam = self.members[0].gradcam(x, threshold, cls=self.decide(probs, threshold))
        return probs, cam


def model_files(weights):
    """The weight files behind `weights` (one .pt file, or the files listed in a final_model.json)."""
    if weights.lower().endswith(".json"):
        card = json.load(open(weights))
        return [os.path.join(os.path.dirname(os.path.abspath(weights)), f) for f in card["models"]]
    return [weights]


def model_size_mb(weights):
    return sum(os.path.getsize(f) for f in model_files(weights)) / 1e6


def load_model(weights):
    if weights.lower().endswith(".json"):          # averaged model: load every listed file
        files = model_files(weights)
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            sys.exit(f"{weights} lists model files that are missing: {', '.join(os.path.basename(f) for f in missing)}")
        members = [load_model(f) for f in files]
        if len(members) == 1:
            return members[0]
        return Ensemble(members, f"{len(members)} x {members[0].arch} (mean of P(bleached))")
    card_path = os.path.splitext(weights)[0] + ".json"
    card = json.load(open(card_path)) if os.path.exists(card_path) else {}
    imgsz = int(card.get("imgsz", 224))
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    arch = card.get("arch")
    if arch is None:   # detect from the file
        if isinstance(ckpt, dict) and "model" in ckpt and hasattr(ckpt.get("model"), "names"):
            arch = "yolov8-cls"
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            arch = "small_cnn"
        elif isinstance(ckpt, dict) and any(k.startswith("features.") for k in ckpt) and "classifier.1.weight" in ckpt:
            arch = "efficientnet_b0"
        else:
            sys.exit(f"Cannot tell which model {weights} holds. Put the model card ({card_path}) next to it.")

    if arch.startswith("yolov8"):
        from ultralytics import YOLO
        net = YOLO(weights).model.float().eval()
        names = net.names if isinstance(net.names, dict) else dict(enumerate(net.names))
        classes = [names[i] for i in range(len(names))]
        # Ultralytics classification input is RGB in [0, 1] with no mean/std normalization
        return Classifier(net, net.model[-1].conv, classes, imgsz, [0, 0, 0], [1, 1, 1], arch)
    if arch == "efficientnet_b0":
        import torchvision
        net = torchvision.models.efficientnet_b0(weights=None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, 2)
        net.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        return Classifier(net.eval(), net.features[-1], card.get("classes", ["healthy", "bleached"]), imgsz,
                          card.get("mean", IMNET_MEAN), card.get("std", IMNET_STD), arch)
    if arch == "small_cnn":
        net = SmallCNN()
        net.load_state_dict(ckpt["state_dict"])
        return Classifier(net.eval(), net.features[-1], card.get("classes", ["healthy", "bleached"]), imgsz,
                          [float(v) for v in ckpt["mean"]], [float(v) for v in ckpt["std"]], arch)
    sys.exit(f"Unknown architecture in the model card: {arch}")


# ---------------------------------------------------------------- images
def pad_square(im):
    """Pad to a square with black borders (the training preprocessing); return the square and the image box."""
    s = max(im.size)
    left, top = (s - im.size[0]) // 2, (s - im.size[1]) // 2
    sq = Image.new("RGB", (s, s), (0, 0, 0))
    sq.paste(im, (left, top))
    return sq, (left, top, left + im.size[0], top + im.size[1])


def overlay(im, cam_square, box, side):
    """Original photo | photo with the Grad-CAM map in red, cropped back to the photo's own shape."""
    cam = Image.fromarray(np.uint8(cam_square * 255)).resize((side, side), Image.BILINEAR).crop(box).resize(im.size, Image.BILINEAR)
    a = np.asarray(cam, dtype=np.float32)[..., None] / 255 * 0.65
    base = np.asarray(im, dtype=np.float32)
    red = np.array([227, 74, 72], dtype=np.float32)
    heat = (base * (1 - a) + red * a).clip(0, 255).astype(np.uint8)
    out = Image.new("RGB", (im.size[0] * 2 + 8, im.size[1]), (255, 255, 255))
    out.paste(im, (0, 0))
    out.paste(Image.fromarray(heat), (im.size[0] + 8, 0))
    return out


def list_images(target):
    if os.path.isdir(target):
        files = sorted(p for p in glob.glob(os.path.join(target, "**", "*"), recursive=True) if p.lower().endswith(IMG_EXTS))
    elif os.path.isfile(target):
        files = [target]
    else:
        sys.exit(f"Not found: {target}")
    if not files:
        sys.exit(f"No images ({', '.join(IMG_EXTS)}) in {target}")
    return files


def benchmark(clf, files, n=100, warm=10):
    """Proposal protocol: mean time per image over n images, batch size 1, after `warm` warm-up runs, on the CPU."""
    torch.set_grad_enabled(False)

    def readable(p):
        try:
            with Image.open(p) as im:
                im.verify()
            return True
        except Exception:
            return False

    files = [p for p in files if readable(p)]
    if not files:
        sys.exit("No readable images to benchmark.")
    pool = (files * (1 + (n + warm) // len(files)))[: n + warm]

    def one(p):
        clf.predict(clf.to_input(pad_square(Image.open(p).convert("RGB"))[0]))

    for p in pool[:warm]:
        one(p)
    t0 = time.perf_counter()
    for p in pool[warm:]:
        one(p)
    return (time.perf_counter() - t0) / n * 1000


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Classify reef photos as healthy or bleached, with Grad-CAM heatmaps.")
    ap.add_argument("target", help="an image file or a folder of images")
    ap.add_argument("--weights", help="final_model.json (averaged model) or one .pt file (default: results/run10/final_model.json)")
    ap.add_argument("--out", default="demo_output", help="folder for the heatmap images (default: demo_output)")
    ap.add_argument("--csv", help="also write the predictions to this CSV file")
    ap.add_argument("--no-heatmap", action="store_true", help="skip Grad-CAM (faster)")
    ap.add_argument("--show", action="store_true", help="open each heatmap in a window")
    ap.add_argument("--benchmark", action="store_true", help="measure CPU latency (100 images, batch 1, 10 warm-ups) and exit")
    ap.add_argument("--threshold", type=float, help="P(bleached) cutoff (default: the validation-selected cutoff saved with the model, else 0.5)")
    args = ap.parse_args()

    weights = args.weights or next((w for w in DEFAULT_WEIGHTS if os.path.exists(w)), None)
    if not weights or not os.path.exists(weights):
        sys.exit("No model found. Put the Run 10 files (final_model.json and final_model_s42/43/44.pt) in results/run10/, "
                 "or pass --weights path/to/final_model.json or path/to/model.pt")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    clf = load_model(weights)
    out_dir = os.path.abspath(args.out) + os.sep
    files = [p for p in list_images(args.target) if not os.path.abspath(p).startswith(out_dir)]   # skip earlier heatmaps
    if not files:
        sys.exit(f"No images to classify in {args.target}")
    threshold, thr_src = (args.threshold, "--threshold") if args.threshold is not None else default_threshold(weights)
    print(f"Model: {clf.arch} ({model_size_mb(weights):.1f} MB) from {weights}; classes {clf.classes}")
    print(f"Decision rule: bleached when P(bleached) >= {threshold:.3f} ({thr_src})")

    if args.benchmark:
        ms = benchmark(clf, files)
        print(f"CPU latency: {ms:.1f} ms/image ({os.cpu_count()} logical cores, batch 1, 100 images after 10 warm-ups"
              f"{'; images reused because the folder has fewer than 110' if len(files) < 110 else ''})")
        return

    if not args.no_heatmap:
        os.makedirs(args.out, exist_ok=True)
    rows = []
    print(f"\n{'file':<40} {'prediction':<10} {'probability':>11} {'P(bleached)':>11}")
    for p in files:
        try:
            im = Image.open(p).convert("RGB")
        except (UnidentifiedImageError, OSError, ImportError, ValueError):
            print(f"{os.path.basename(p)[:40]:<40} skipped: not a readable image file")
            continue
        note = spec_note(im)
        sq, box = pad_square(im)
        x = clf.to_input(sq)
        if args.no_heatmap:
            probs, cam = clf.predict(x), None
        else:
            probs, cam = clf.gradcam(x, threshold)
        k = clf.decide(probs, threshold)
        row = dict(file=p, prediction=clf.classes[k], probability=round(float(probs[k]), 4),
                   p_bleached=round(float(probs[clf.bleached]), 4), heatmap="", note=note)
        if cam is not None:
            fig = overlay(im, cam, box, sq.size[0])
            row["heatmap"] = os.path.join(args.out, os.path.splitext(os.path.basename(p))[0] + "_gradcam.png")
            fig.save(row["heatmap"])
            if args.show:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 4.5))
                plt.imshow(fig); plt.axis("off")
                plt.title(f"{os.path.basename(p)}: {row['prediction']} ({row['probability']:.2f})  |  red = regions that drove the prediction")
                plt.tight_layout(); plt.show()
        rows.append(row)
        print(f"{os.path.basename(p)[:40]:<40} {row['prediction']:<10} {row['probability']:>11.3f} {row['p_bleached']:>11.3f}"
              + (f"   ! {note}" if note else ""))

    n_b = sum(r["prediction"] == "bleached" for r in rows)
    print(f"\n{len(rows)} image(s): {n_b} bleached, {len(rows) - n_b} healthy."
          + (f" Heatmaps in {args.out}/" if not args.no_heatmap else ""))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file"])
            w.writeheader(); w.writerows(rows)
        print("Predictions written to", args.csv)
    print("Screening aid only: a trained reef monitor makes the final call.")


if __name__ == "__main__":
    main()
