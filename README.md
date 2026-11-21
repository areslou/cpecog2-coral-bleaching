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
notebooks/Group2_Coral_Run3.ipynb   leakage-safe validation-threshold experiment; outputs are kept separate
demo.py                              demo: one image or a folder -> class, probability, Grad-CAM heatmap
eval_external.py                     external BU + BHD evaluation of the final model
download_results.py                  package and download the external evaluation CSV results
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

After running `eval_external.py`, package and download the external result CSVs:

```python
!python download_results.py
```

In Colab, this asks the browser to download `coral_external_results.zip`. Outside Colab, it creates the ZIP in the repository folder.

### Leakage-safe threshold tuning

Run 2 now separates threshold selection from final evaluation:

1. The operating cutoff is selected only from seed-42 validation probabilities.
2. The cutoff search prefers the highest validation macro F1 among cutoffs that achieve at least 0.90 bleached recall.
3. The selected cutoff is saved in `results/run2/selected_threshold.csv`.
4. The held-out test set is evaluated once with that fixed cutoff.
5. The external BU/BHD set is evaluated once with the same validation-selected cutoff.

Pass `--threshold 0.xxx` to `eval_external.py` after selecting the cutoff. The test and external sets must not be used to change the cutoff.

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

### Run 1 (Nov. 13, 2026): validation only

| Model | Settings | Val accuracy | Val macro F1 | Bleached recall | Size | CPU latency |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| YOLOv8s-cls, ImageNet weights | Ultralytics defaults, 15 epochs, imgsz 224, batch 32, seed 42 | 0.859 | 0.859 | 0.847 | 10.3 MB | 28 ms |

![Run 1 training curves](results/run1/training_curves.png)

### Run 2 (Nov. 21, 2026): baselines, tuning, final model, test

Data: the 9 photos filed under both labels (18 files) and 1 near-duplicate across splits were removed. The final split is 629 train / 132 validation / 131 test.

Validation macro F1 (seed 42) for each candidate. The final model is the one with the highest value.

| Model / run | Val macro F1 | Val bleached recall |
| :-- | :-- | :-- |
| Majority class | 0.343 | |
| Scratch CNN (3 blocks, 0.29M params) | 0.788 | 0.754 |
| R2 YOLOv8s-cls, Run 1 settings | 0.818 | 0.826 |
| R3 light augmentation | 0.833 | 0.739 |
| **R4 dropout 0.3 (final)** | **0.864** | 0.797 |
| R5 AdamW, lr0 5e-4, cosine | 0.841 | 0.797 |
| R6 two-stage fine-tuning | 0.833 | 0.797 |
| R7 R3 to R6 combined | 0.833 | 0.754 |
| EfficientNet-B0 | 0.833 | 0.812 |

Final model on the held-out test set (n = 131, evaluated once, mean ± std over seeds 42, 43, 44):

| Metric | Target | Result | Status |
| :-- | :-- | :-- | :-- |
| Macro F1 | ≥ 0.90 | 0.839 ± 0.046 | not met |
| Bleached recall | ≥ 0.90 | 0.801 ± 0.091 | not met |
| Gain over the scratch CNN (test macro F1 0.771) | ≥ +10 points | +6.8 points | not met |
| Accuracy / ROC AUC | | 0.840 ± 0.046 / 0.921 ± 0.007 | |
| Model size | < 50 MB | 10.3 MB | met |
| CPU latency (Colab CPU) | < 200 ms | 36 ms | met |

By seed, test macro F1 was 0.793, 0.885 and 0.840. Seed 42 confusion matrix (rows true healthy/bleached, columns predicted): [[57, 7], [20, 47]]. Most errors are bleached photos predicted as healthy.

![Run 2 sweep](results/run2/sweep_macro_f1.png)

Error analysis (seed 42, [`test_errors.csv`](results/run2/test_errors.csv), [`gradcam_test.png`](results/run2/gradcam_test.png)):

- 20 of the 27 test errors are bleached photos predicted healthy. 12 of the 20 have P(bleached) below 0.35, so a slightly lower cutoff would recover only a few of them.
- Grad-CAM on correct bleached predictions falls on the pale coral, not on the water or fish.
- Missed bleached photos are mostly wide reef scenes where the bleached coral is a small part of the frame (partial bleaching), or naturally pale coral such as a brain coral. In one, the heatmap sits on dark coral next to a clearly white colony.
- Some heatmaps spread faintly onto the black padding bars.

![Grad-CAM on test images](results/run2/gradcam_test.png)

External set: Run 2 section 8 evaluated only the BU images (n = 238: macro F1 0.768 ± 0.034, bleached recall 0.672 ± 0.081). The BHD files are named `*_hbd_*` in the Hugging Face zip, so the name check missed them. The notebook is fixed, and `eval_external.py` evaluates the final model on BU + BHD.

## Next steps

- Run `python eval_external.py --weights results/run2/final_model_s42.pt` on Colab for the BU + BHD external result, and commit `results/run2/`.
- Measure CPU latency on a team laptop with `python demo.py <test images> --benchmark`.
- Test the demo from a clean environment, and record the backup video.
- Write the results, error analysis and Grad-CAM findings into the report and slides.
