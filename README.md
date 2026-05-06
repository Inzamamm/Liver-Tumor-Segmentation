# Liver Tumor Segmentation Framework

This repository contains a complete PyTorch implementation of the proposed liver tumor segmentation model. The model follows the paper methodology: preprocessing, hierarchical feature encoding, contextual attention representation, decoder reconstruction with skip fusion, composite loss optimization, evaluation, prediction, and visualization.

## 1. Project Structure

```text
liver_tumor_segmentation_project/
├── configs/
│   └── config.yaml
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── visualize.py
│   ├── plot_results.py
│   ├── ablation_template.py
│   └── utils.py
├── outputs/
├── requirements.txt
└── README.md
```

## 2. Dataset Preparation

Download the dataset from Kaggle:

```text
https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation/data
```

Place the dataset in the following format:

```text
data/
├── images/
│   ├── case_001.png
│   ├── case_002.png
│   └── ...
└── masks/
    ├── case_001.png
    ├── case_002.png
    └── ...
```

The image and mask filenames must match. For example:

```text
images/case_001.png
masks/case_001.png
```

Supported formats include PNG, JPG, TIF, NPY, NII, and NII.GZ.

## 3. Installation

Create a virtual environment and install requirements:

```bash
pip install -r requirements.txt
```

## 4. Configuration

Edit `configs/config.yaml` and update these paths:

```yaml
image_dir: ./data/images
mask_dir: ./data/masks
output_dir: ./outputs
```

Main hyperparameters:

```yaml
image_size: 256
batch_size: 8
num_epochs: 100
learning_rate: 0.0001
attention_heads: 4
base_channels: 64
window_min: -200
window_max: 250
```

## 5. Training

Run:

```bash
python src/train.py configs/config.yaml
```

Training produces:

```text
outputs/best_model.pth
outputs/training_log.csv
outputs/training_curves.png
outputs/train_split.csv
outputs/val_split.csv
outputs/test_split.csv
```

## 6. Evaluation

Run:

```bash
python src/evaluate.py configs/config.yaml outputs/best_model.pth
```

Evaluation produces:

```text
outputs/test_metrics.csv
outputs/prediction_panels/
```

The reported metrics include Dice, IoU, precision, recall, HD95, and ASD.

## 7. Single Image Prediction

Run:

```bash
python src/predict.py configs/config.yaml path/to/image.png outputs/best_model.pth
```

Prediction produces:

```text
outputs/single_prediction_mask.png
outputs/single_prediction_probability.png
```

## 8. Model Summary

The proposed model contains:

1. Preprocessing pipeline with intensity clipping, z-score normalization, min-max normalization, and resizing.
2. Hierarchical encoder with residual convolutional blocks.
3. Contextual representation module based on multi-head self-attention.
4. Decoder reconstruction with transposed convolution and skip fusion.
5. Composite loss using Dice loss and binary cross-entropy.

## 9. Notes for Kaggle

On Kaggle, upload this project, install requirements, update the dataset path in `configs/config.yaml`, and run:

```bash
python src/train.py configs/config.yaml
python src/evaluate.py configs/config.yaml outputs/best_model.pth
```

If memory is limited, reduce:

```yaml
batch_size: 4
base_channels: 32
```

## 10. Citation Note

Dataset reference:

```bibtex
@misc{lits_dataset,
  author = {andrewmvd},
  title = {Liver Tumor Segmentation Dataset},
  year = {2020},
  howpublished = {https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation}
}
```
