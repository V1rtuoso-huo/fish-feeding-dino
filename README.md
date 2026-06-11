# Attention Calibration with Frozen DINO Masks for Aquaculture Bioinformatics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official implementation of the paper:

> **Attention Calibration with Frozen DINO Masks: A Bioinformatics Perspective on Interpretable Fish Feeding Intensity Classification with Small-Scale Aquaculture Data**
>
> YuChen Huo, Weibin Huang, Xiaohui Dong, Jiangmao Zheng
>
> *PRICAI 2026 (under review)*

## Overview

This repository provides a complete pipeline for injecting frozen DINO Vision Transformer attention maps as external priors into a ResNet50 classifier. The method is designed for **small-data bioinformatics applications**---specifically, fish feeding intensity recognition in aquaculture, where annotated datasets are typically limited to ~1,000 images per farm. The frozen attention prior steers the classifier toward biologically relevant image regions (fish schools) without requiring any additional annotations.

### Key contributions
- **Mask-Guided ResNet50**: 4-channel input (`[RGB ‖ DINO mask]`) with zero-initialised 4th channel weights
- **Cross-domain evaluation**: Public AV-FFIA benchmark + self-collected aquaculture dataset
- **Comprehensive ablations**: Mask source (ViT-S/8, DINOv2, Uniform, Random), fusion strategy (early concat, gated, distillation, two-stream), augmentation alignment
- **Attention analysis**: Grad-CAM visualisation + quantitative metrics (Spearman ρ, Pearson r, entropy, background response)

## Repository structure

```
├── README.md
├── requirements.txt
├── src/
│   ├── models.py          # MaskGuidedResNet, PlainResNet50, PR50
│   ├── dataset.py         # MaskDataset, CoAugMaskDataset
│   ├── gradcam.py         # Grad-CAM extraction
│   ├── dino_mask.py       # DINO ViT-B/8 mask generation (offline)
│   ├── metrics.py         # Spearman, Pearson, Entropy, Background Response
│   └── train_utils.py     # Training loop + evaluation utilities
├── experiments/
│   ├── main_comparison.py       # Table 2+3+4: Plain vs Mask-guided
│   ├── avffia_benchmark.py      # Table 1: AV-FFIA public benchmark
│   ├── mask_source_ablation.py  # Table 5: ViT-S/8, DINOv2, Uniform, Random
│   ├── fusion_ablation.py       # Table 6: Distillation + Gated + Two-stream
│   └── coaug_control.py         # Table 7: Augmentation-alignment control
├── figures/
│   └── plot_metrics.py          # Generate Fig. 3 (quantitative attention plots)
└── generate_masks.py            # Standalone DINO mask precomputation script
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

The self-collected fish feeding intensity dataset is available on Google Drive:

📂 [**Google Drive link**](https://drive.google.com/your-dataset-link)

Download and extract the `dataset/` folder to the project root. It should contain:

```
dataset/
├── train/          # 757 images, 3 classes (low, medium, high)
├── val/            # 215 images
├── test/           # 214 images
├── masks_train/    # Precomputed DINO ViT-B/8 soft masks (.npy)
├── masks_val/
└── masks_test/
```

The AV-FFIA public benchmark dataset is available from its [original source](https://github.com/cuilimeng/AV-FFIA). Place extracted frames in `./avffia_images/`.

### 3. Generate DINO masks (if not using precomputed masks)

```bash
python generate_masks.py
```

### 4. Reproduce experiments

| Paper table | Command |
|-------------|---------|
| Table 1 (AV-FFIA) | `python experiments/avffia_benchmark.py` |
| Table 2–4 (Main comparison + attention) | `python experiments/main_comparison.py` |
| Table 5 (Mask-source ablation) | `python experiments/mask_source_ablation.py` |
| Table 6 (Fusion-strategy ablation) | `python experiments/fusion_ablation.py` |
| Table 7 (Co-augmentation control) | `python experiments/coaug_control.py` |

All results are saved to `results_prical/` as JSON files.

### 5. Generate figures

```bash
python figures/plot_metrics.py
```

## Citation

```bibtex
@inproceedings{huo2026attention,
  title={Attention Calibration with Frozen DINO Masks: A Bioinformatics Perspective on Interpretable Fish Feeding Intensity Classification},
  author={Huo, YuChen and Huang, Weibin and Dong, Xiaohui and Zheng, Jiangmao},
  booktitle={Proceedings of the Pacific Rim International Conference on Artificial Intelligence (PRICAI)},
  year={2026}
}
```

## License

This project is provided for research purposes. The dataset is available from the corresponding author (Xiaohui Dong, dongxiaohui2003@163.com) upon reasonable request.
