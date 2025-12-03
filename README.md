# Ingrown-Tissue-Quantification

Tissue segmentation and quantification toolkit for histological images using a UNET-based CNN.

This repository contains Jupyter notebooks and helper functions to create an image dataset (with augmentation), train a UNet model in TensorFlow, compute optimal thresholds from the ROC curve, and evaluate segmentation performance using standard metrics (accuracy, precision, recall, F1, Jaccard/IoU). The notebooks included are optimized for working with image/mask pairs and a simple workflow for dataset generation, model training, and postprocessing of predictions.

---

## Table of Contents

- [Features](#features)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Dataset structure and image formats](#dataset-structure-and-image-formats)
- [Quick start](#quick-start)
- [Dataset creation and augmentation](#dataset-creation-and-augmentation)
- [Model (training) workflow](#model-training-workflow)
- [Evaluation & Metrics](#evaluation--metrics)
- [Results & Visualization](#results--visualization)
- [Troubleshooting & Tips](#troubleshooting--tips)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- UNET architecture implemented in TensorFlow / Keras for binary image segmentation
- Data loader and tf.data pipeline (images resized, normalized, and batched)
- Albumentations-based augmentations (HorizontalFlip, CoarseDropout, RandomBrightness, RandomContrast)
- Training utilities with ModelCheckpoint, ReduceLROnPlateau, and EarlyStopping
- Post-processing / Predictions: thresholding strategies and ROC-based optimal threshold detection (Youden's index, distance-to-(0,1))
- Metric computation and CSV export for experiment results

---

## Project Structure

At a glance:

- `DatasetMakerTensorFlow.ipynb` — Notebook to load raw images and masks, augment images (albumentations), and save a structured dataset (train/valid/test) under `dataset/`.
- `ModelTensorFlow.ipynb` — Notebook containing model architecture (UNET), training pipeline, evaluation, prediction and ROC-based thresholding.
- `model.png`, `distance.png`, `youden.png` — Visual results and helper images used in the notebook markdowns.
- `README.md` — This document

---

## Requirements

- Python >= 3.8 (macOS default shell/zsh instructions below)
- Some commonly used packages (install via pip):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jupyterlab jupyter numpy pandas matplotlib opencv-python scikit-learn scikit-image tensorflow albumentations tqdm pillow
```

Recommended versions (you can pin them if needed):

- tensorflow >= 2.6 or 2.x (match available GPU / CUDA on your machine)
- opencv-python
- albumentations

Note: If you have GPU support, install a TensorFlow variant that supports your GPU (e.g., `tensorflow` or `tensorflow-macos` + `tensorflow-metal` on M1/M2 macs).

---

## Dataset structure and image formats

This repository expects images and binary masks to be provided in pairs. Typical folder layout produced by `DatasetMakerTensorFlow.ipynb`:

```
dataset/
	├─ train/
	│   ├─ images/
	│   └─ masks/
	├─ valid/
	│   ├─ images/
	│   └─ masks/
	└─ test/
			├─ images/
			└─ masks/
```

- Images: color images (e.g., .TIF) — they will be read with OpenCV and normalized to [0,1].
- Masks: grayscale binary masks where foreground/in-growth area uses the value 255 (0–255), saved as grayscale. In the pipeline the masks are normalized to float (0–1) and expanded to shape (H, W, 1).

Target height / width used by the notebook: 1100 x 1344 (adjustable in `ModelTensorFlow.ipynb`).

---

## Quick start

1. Install the required packages and create a Python environment (see Requirements).

2. Place your raw `images/` and `masks/` into a folder (for the dataset creator to pick up). For example: `raw_images/` and `raw_masks/` in the repo root.

3. Run `DatasetMakerTensorFlow.ipynb` in JupyterLab / Jupyter Notebook: this will create `dataset/train`, `dataset/valid`, and `dataset/test` directories and will augment the training dataset if the `augment` option is enabled.

4. Open `ModelTensorFlow.ipynb`. Update the path variables at the top of the notebook if needed:

```python
dataset_path = './dataset'  # default used by the notebook
files_dir = os.path.join(os.getcwd(), "files")  # where the model & log will be saved
model_file = os.path.join(files_dir, "unet.h5")
```

5. Configure the hyperparameters at the top of the `ModelTensorFlow.ipynb` notebook:

```python
batch_size = 4
lr = 1e-4
epochs = 25
height = 1100
width = 1344
```

6. Run each cell in the notebook in order to train the model and save the best weights. The `files_dir` will contain `unet.h5` after training.

---

## Dataset creation and augmentation

Summary of `DatasetMakerTensorFlow.ipynb` steps:

- Loads raw image/mask pairs using OpenCV (cv2).
- Splits the dataset into train/valid/test using scikit-learn's `train_test_split` while preserving pairing between images and masks.
- Augmentation (optional) is handled via Albumentations. Augmentations included are:
  - HorizontalFlip
  - CoarseDropout (random holes)
  - RandomBrightness
  - RandomContrast

Augmented filenames are saved as `{basename}_{augment_index}.TIF`.

---

## Model (training) workflow

Key components in `ModelTensorFlow.ipynb`:

- UNet architecture:
  - Implemented using `conv_block`, `encoder_block`, `decoder_block` utilities and assembled by `build_unet()`.
  - Output is single-channel sigmoid for binary segmentation (shape: (height, width, 1)).
- Data pipeline:
  - Uses `read_image()` and `read_mask()` helper functions to load and preprocess images / masks (resize, normalize to [0,1])
  - `tf_parse()` wraps the numpy function calls into TensorFlow `tf.data` pipeline
- Training: compiled with `binary_crossentropy`, Adam optimizer, and metric `acc`. Callbacks include `ModelCheckpoint`, `ReduceLROnPlateau`, `EarlyStopping`.

To train:

1. Ensure the dataset structure (under `dataset/`) is created by running the Dataset maker.
2. Run the `ModelTensorFlow.ipynb` cells sequentially to load the data, build the model, and call `model.fit()`.
3. Best model weights will be saved to `files/unet.h5`.

---

## Evaluation & Metrics

Postprocessing / thresholding methods are provided in the training notebook. The pipeline:

1. Model inference is performed to produce predicted probability masks (grayscale float arrays saved to `predictions/` or processed inline).
2. For each predicted pixel probability map and corresponding ground truth, compute an ROC curve (TPR/FPR vs. threshold).
3. Determine two methods of optimal threshold:
   - Youden's Index: maximize (TPR - FPR)
   - Distance-to-(0,1): minimize distance to the (FPR, TPR) = (0,1) corner.
4. Apply the chosen threshold to binarize predicted masks.
5. Evaluate metrics at the binarized threshold: Accuracy, Precision, Recall, F1-score, Jaccard (IoU). Results are exported to `./metrics/youden_score.csv` and `./metrics/distance_score.csv`.

There are helper functions for:

- `compute_roc_curve(true_y, pred_y)` — generates arrays of FPR/TPR/thresholds (grid of 100 thresholds used).
- `find_optimal_threshold(fprs, tprs, thresholds, method='youden'|'distance')` — returns the threshold value.
- `calculate_metrics(true, pred)` — returns accuracy, precision, recall, f1, jaccard.

---

## Results & Visualization

- The repo contains example visualizations `distance.png`, `youden.png` and `model.png` to indicate typical outputs and the UNET diagram.
- After evaluation, binary mask images produced using Youden or distance-based thresholds are saved to `./metrics/youden_{name}.png` and `./metrics/distance_{name}.png` respectively.

---

## Troubleshooting & Tips

- If your images or masks are a different size, update `height` and `width` at the top of `ModelTensorFlow.ipynb`. Ensure you generate all dataset/volume data using `DatasetMakerTensorFlow.ipynb` with the same size or adjust your `read_image/read_mask` resize settings.
- If training runs slowly on CPU-only systems, reduce batch size or image size while debugging.
- For reproducibility, the notebook sets seeds via `os.environ["PYTHONHASHSEED"]`, `np.random.seed(42)`, and `tf.random.set_seed(42)`.
- If you want to perform inference and save predictions to a particular folder, change `predictions/` target path accordingly.

---

## Contributing

Contributions are welcome. If you want to improve the pipeline, please:

1. Fork the repository
2. Create a new working branch for your feature/fix
3. Submit a PR with a clean summary of changes

Please ensure your changes are tested and documented in the README.

---

## License

None 
---

For any other questions, please open an issue or contact the repository owner.
