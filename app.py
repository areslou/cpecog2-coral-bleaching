"""Coral bleaching screening GUI (CPECOG2 Group 2).

A drag-and-drop window around the same model code as demo.py. The "Single photo" tab shows healthy or
bleached, P(bleached), the decision rule, and the Grad-CAM heatmap; the "Survey summary" tab takes all the
photos from one site and reports the share flagged as bleached and the order an expert should review them. It runs on the
laptop's own CPU at http://127.0.0.1:7860 and needs no internet once requirements.txt is installed.

  python app.py                          # opens the browser automatically
  python app.py --weights path/to/model.pt --threshold 0.5 --port 7861

If anything goes wrong during a presentation, demo.py does the same job from the terminal.
"""
import argparse
import os
import socket
import tempfile

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # no usage pings; the demo must work offline

import torch

from demo import DEFAULT_WEIGHTS, default_threshold, load_model, overlay, pad_square, spec_note
from survey import assign_sites, classify as survey_classify, load_site_map, report_text, review_list, summarize


def build(weights, threshold):
    import gradio as gr

    clf = load_model(weights)
    thr, thr_src = (threshold, "--threshold") if threshold is not None else default_threshold(weights)
    size_mb = os.path.getsize(weights) / 1e6
    rule = f"bleached when P(bleached) ≥ {thr:.3f}" + (" (validation-selected cutoff)" if thr_src.endswith(".csv") else "")

    def classify(im):
        if im is None:
            return "Drop a reef photo on the left, then press **Classify**.", None
        im = im.convert("RGB")
        note = spec_note(im)
        sq, box = pad_square(im)
        probs, cam = clf.gradcam(clf.to_input(sq), thr)
        k = clf.decide(probs, thr)
        label, p_b = clf.classes[k], float(probs[clf.bleached])
        text = (f"## {label.upper()}\n\n"
                f"**P(bleached): {p_b:.3f}**  \n"
                f"Decision rule: {rule}\n\n")
        if note:
            text += f"> **Warning, {note}.** The model was trained on reef photos, so this result may not be reliable.\n\n"
        text += "Red in the heatmap marks the regions that drove the prediction.  \n" \
                "*Screening aid only: a trained reef monitor makes the final call.*"
        return text, overlay(im, cam, box, sq.size[0])

    def survey(files, site):
        if not files:
            return "Upload the photos from one survey or site, then press **Summarize survey**.", [], None
        site = (site or "").strip() or "Survey"
        paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]
        csvs = [p for p in paths if p.lower().endswith(".csv")]
        items = [(site, p, os.path.basename(p)) for p in paths if not p.lower().endswith(".csv")]
        if csvs:   # sites CSV uploaded with the photos: one upload, many sites
            try:
                items = assign_sites(items, load_site_map(csvs[0]))
            except ValueError as e:
                return f"Sites CSV problem: {e}", [], None
        if not items:
            return "Upload photos too, not only the sites CSV.", [], None
        photos, skipped = survey_classify(clf, items, thr)
        if photos.empty:
            return "None of the uploaded files is a readable image.", [], None
        sites, review = summarize(photos), review_list(photos)
        text = report_text(sites, review, thr, top=10, skipped=skipped)
        flagged = review[review["prediction"] == "bleached"].head(8)
        gallery = [(r["path"], f"#{r['rank']}  P(bleached) {r['p_bleached']:.2f}") for _, r in flagged.iterrows()]
        out_dir = tempfile.mkdtemp(prefix="survey_")
        csv_path = os.path.join(out_dir, f"review_list_{site.replace(' ', '_')}.csv")
        review.drop(columns=["path"]).to_csv(csv_path, index=False)
        return text, gallery, csv_path

    with gr.Blocks(title="Coral Bleaching Screening") as ui:
        gr.Markdown("# Coral Bleaching Screening\n"
                    f"CPECOG2 Group 2 · model: {clf.arch}, {size_mb:.1f} MB, runs on this computer's CPU · {rule}")
        with gr.Tabs():
            with gr.Tab("Single photo"):
                with gr.Row():
                    with gr.Column():
                        inp = gr.Image(type="pil", label="Reef photo (JPG, PNG, BMP, WEBP or TIFF)", height=360)
                        with gr.Row():
                            clear = gr.ClearButton(value="Clear")
                            run = gr.Button("Classify", variant="primary")
                    with gr.Column():
                        verdict = gr.Markdown("Drop a reef photo on the left, then press **Classify**.")
                        heat = gr.Image(type="pil", label="Photo | Grad-CAM heatmap", interactive=False)
                run.click(classify, inputs=inp, outputs=[verdict, heat])
                inp.upload(classify, inputs=inp, outputs=[verdict, heat])
                clear.add([inp, heat])
                clear.click(lambda: "Drop a reef photo on the left, then press **Classify**.", outputs=verdict)
            with gr.Tab("Survey summary"):
                gr.Markdown("Upload every photo from a survey. For **several sites at once**, add a sites CSV in the same upload "
                            "(columns `photo` and `site`, one row per photo file name); otherwise all photos count as the one site named below. "
                            "The tool reports the share of photos flagged as bleached per site (with a 95% confidence interval), "
                            "how many are uncertain, and the order in which an expert should review them.")
                with gr.Row():
                    with gr.Column(scale=1):
                        files = gr.File(file_count="multiple", file_types=["image", ".csv"], label="Survey photos (+ optional sites CSV)")
                        site = gr.Textbox(label="Site name (used when no sites CSV is uploaded)", placeholder="e.g. Anilao, Site 3")
                        go = gr.Button("Summarize survey", variant="primary")
                        csv_out = gr.File(label="Review list (CSV)", interactive=False)
                    with gr.Column(scale=2):
                        report = gr.Markdown("Upload the photos from one survey or site, then press **Summarize survey**.")
                        gallery = gr.Gallery(label="Review first: photos most likely bleached", columns=4, height="auto")
                go.click(survey, inputs=[files, site], outputs=[report, gallery, csv_out])
    return ui


def free_port(start, tries=50):
    """First port from `start` that can be opened on this computer (skips busy and Windows-reserved ports)."""
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if os.name != "nt":   # a port left in TIME_WAIT by a closed app.py is reusable (the server sets this too)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port between {start} and {start + tries - 1}. Close other app.py windows or pass --port 8500.")


def main():
    ap = argparse.ArgumentParser(description="Drag-and-drop GUI for the coral bleaching model.")
    ap.add_argument("--weights", help="model weights (default: the final model in results/)")
    ap.add_argument("--threshold", type=float, help="P(bleached) cutoff (default: the validation-selected cutoff saved with the model, else 0.5)")
    ap.add_argument("--port", type=int, default=7860, help="first port to try; the next free one is used if it is busy")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = ap.parse_args()

    weights = args.weights or next((w for w in DEFAULT_WEIGHTS if os.path.exists(w)), None)
    if not weights or not os.path.exists(weights):
        raise SystemExit("No weights found. Put the final model in results/run3/ or results/run2/, or pass --weights path/to/model.pt")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    port = free_port(args.port)
    if port != args.port:
        print(f"Port {args.port} is busy (another app.py still running, or reserved by Windows), so using port {port}.")
    build(weights, args.threshold).launch(server_name="127.0.0.1", server_port=port, inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
