# FAST: Fast Arbitrary-Shaped Text Detector

This repository provides a faithful, clean implementation of the FAST model from [arXiv:2111.02394](https://arxiv.org/pdf/2111.02394). The code is designed for clarity, reproducibility, and speed, following the official config and all core tricks from the paper.Neural Networks for Bermudan Option Pricing

## File Guide
- `models/` — Model code (backbone, neck, head)
- `losses/` — Dice loss, OHEM
- `datasets/` — Simple dataset loader (polygon .txt format)
- `utils/` — Logger, metrics, post-processing
- `train.py` — Training script
- `test.py` — Evaluation script
- `inference.py` — Inference script

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare your data:**
   - Put images in `images/` and matching polygon annotation `.txt` files in `annotations/`.
   - Each annotation should be a single line of comma-separated coordinates (see `colab_fast_test.py` for an example).
3. **Train:**
   ```bash
   python train.py --img_dir images/ --ann_dir annotations/ --save_dir checkpoints/
   ```
4. **Test:**
   ```bash
   python test.py --img_dir images/ --ann_dir annotations/ --ckpt checkpoints/fast_epoch10.pth
   ```
5. **Inference:**
   ```bash
   python inference.py --img_dir images/ --ann_dir annotations/ --ckpt checkpoints/fast_epoch10.pth --out_dir inference_results/
   ```

## Notes
- Mixed precision is always on if you use a CUDA GPU.
