# Coral Bleaching Detection with Transfer Learning

CPECOG2 (Deep Neural Networks) design project, Group 2.

The project classifies underwater reef photographs as **healthy** or **bleached** coral with a small pretrained CNN (YOLOv8s-cls), so that bleaching can be screened faster than by manual photo review.

## Targets

| Metric | Target |
| :-- | :-- |
| Macro F1 (held-out test set) | 0.90 or higher |
| Bleached-class recall | 0.90 or higher |
| Model size | under 50 MB |
| CPU latency | under 200 ms per image |

## Data

- **Primary:** [Healthy and Bleached Corals Image Classification](https://www.kaggle.com/datasets/vencerlanz09/healthy-and-bleached-corals-image-classification) (Kaggle), 923 images (485 bleached, 438 healthy).
- **External test set:** healthy and bleached BHD and BU images from [Coral Health Classification](https://huggingface.co/datasets/esahit/coral-health-classification) (Hugging Face).

Images are not stored in this repository. The notebook downloads them with `kagglehub`. The fixed split (after removing 21 perceptual-hash duplicates) is in [`data/splits.csv`](data/splits.csv): 631 train, 135 validation, 136 test, stratified 70/15/15 with seed 42.

## Repository layout

```
notebooks/Group2_Coral_Run1.ipynb   download, EDA, de-duplication, split, Run 1 training and evaluation
notebooks/Group2_Coral_Run2.ipynb   data fixes, baselines, tuning sweep, EfficientNet-B0, final model (3 seeds),
                                     test and external evaluation, size and latency, Grad-CAM
demo.py                              demo: one image or a folder -> class, probability, Grad-CAM heatmap
data/splits.csv                      file name, label, split, perceptual hash for every image used (Run 1 split)
results/run1/                        Run 1 metrics, confusion matrix, training curves, summary
results/run2/                        Run 2 outputs (unzip run2_results.zip here)
requirements.txt                     library versions (exact ultralytics version)
```

## How to run

### Training and evaluation (Google Colab)

1. Open `notebooks/Group2_Coral_Run2.ipynb` in Google Colab (Run 1 is kept for the record).
2. Runtime > Change runtime type > T4 GPU.
3. Runtime > Run all (about 25 to 30 minutes). The notebook reads `data/splits.csv` from this repository's GitHub URL, so the repository must be public.
4. Download `run2_results.zip` and unzip it into `results/run2/`. It holds every table and figure, `run2_summary.txt`, the final model `final_model_s42.pt` (or `final_model_s42_effnet_b0.pt`) and its model card (`.json`, class order and preprocessing).

The held-out test set and the external BU + BHD set are evaluated once, in sections 7 and 8. Run the notebook end to end only when the tuning plan is final.

### Demo (laptop CPU, no GPU or internet needed)

```
pip install -r requirements.txt
python demo.py path/to/photo.jpg                     # one image
python demo.py path/to/folder --csv predictions.csv  # every image in a folder
python demo.py path/to/folder --show                 # also open each heatmap in a window
python demo.py path/to/folder --benchmark            # CPU latency: 100 images, batch 1, after 10 warm-ups
```

The demo uses the final model in `results/run2/` (or `--weights path/to/model.pt`). For each image it prints the predicted class, its probability and P(bleached), and saves the photo next to a Grad-CAM overlay (red = regions that drove the prediction) in `demo_output/`. `--benchmark` is the latency test the success criteria ask for on a team laptop CPU. The model is a screening aid; a trained reef monitor makes the final call.

## Results so far

| Run | Model | Settings | Val accuracy | Val macro F1 | Bleached recall | Size | CPU latency |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 1 (Nov. 13, 2026) | YOLOv8s-cls, ImageNet weights | Ultralytics defaults, 15 epochs, imgsz 224, batch 32, seed 42 | 0.859 | 0.859 | 0.847 | 10.3 MB | 28 ms |

![Run 1 training curves](results/run1/training_curves.png)

Run 2 (`notebooks/Group2_Coral_Run2.ipynb`) is ready but has not been run yet; its results go here. The test set has not been used; it is evaluated once after tuning is final.

## Next steps

Run 2 covers the Run 1 next steps (conflicting-label duplicates and near-duplicate leakage, scratch CNN and majority-class baselines, macro F1 checkpointing, tuning sweep, EfficientNet-B0). What remains:

- Run `Group2_Coral_Run2.ipynb` on Colab and commit `results/run2/`.
- Measure CPU latency on a team laptop with `python demo.py <test images> --benchmark`.
- Test the demo from a clean environment, and record the backup video.
- Write the results, error analysis and Grad-CAM findings into the report and slides.
