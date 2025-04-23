# Vess-Shape-Dataset

## Overview

Vess-Shape-Dataset is a Python toolkit for automatically generating synthetic datasets that mimic the appearance and structure of blood vessels. The main goal is to provide realistic data for training and evaluating image segmentation models, especially in the context of blood vessel segmentation.

## Motivation

This project is inspired by the findings of "IMAGENET-TRAINED CNNS ARE BIASED TOWARDS TEXTURE; INCREASING SHAPE BIAS IMPROVES ACCURACY AND ROBUSTNESS" ([Stylized-ImageNet](https://github.com/rgeirhos/Stylized-ImageNet)). The work demonstrates that Convolutional Neural Networks (CNNs) are forced to learn object shapes when exposed to images with swapped textures. Here, we apply this idea to blood vessel segmentation by generating vessel-like curves and applying diverse textures, encouraging models to focus on shape rather than texture.

## Features

- **Synthetic Vessel Generation:** Uses random Bezier curves to create realistic vessel-like masks.
- **Texture Transfer:** Applies textures from a dataset (e.g., ImageNet) to both foreground (vessel) and background, with options to ensure textures come from different classes.
- **Matting/Blending:** Smoothly blends vessel and background textures using Gaussian blur to avoid hard transitions.
- **Flexible Parameters:** Control the complexity, curvature, radius, and number of vessels per image.
- **Metadata Generation:** Produces metadata for each generated image, including texture sources and class labels.
- **Grid Generation:** Optionally generates a grid of images for easy visualization and comparison.

![output_sample](/example_imgs/output_sample.png)

## Example Usage

```python
from vessel_shape import VesselShape

vs = VesselShape(
    image_size=256,
    n_control_points=5,
    max_vd=100,
    radius=2,
    num_curves=3,
    texture_dir="imagenet_val_sample",
    annotation_csv="imagenet_val_sample/ILSVRC2012_img_val_annotation.csv"
)

results = vs.generate(grid_dir=True, n_samples=3)
```

## Requirements

- Python 3.8+
- numpy
- pandas
- pillow
- matplotlib
- tqdm
- scipy

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

- Place your texture images (e.g., ImageNet samples) in a directory.
- Provide an annotation CSV with image IDs and class labels (see `imagenet_val_sample/ILSVRC2012_img_val_annotation.csv`).

## References

- [Stylized-ImageNet](https://github.com/rgeirhos/Stylized-ImageNet)
- Geirhos, R. et al. "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness." ICLR 2019.
