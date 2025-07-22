# FAST: Fast Arbitrary-Shaped Text Detector

This repository implements the FAST text detection model from [arXiv:2111.02394](https://arxiv.org/pdf/2111.02394)

## File Structure
```
models/         # Model architecture (TextNet, MKR head)
losses/         # Loss functions (Dice, BCE)
datasets/       # Dataset loader (ICDAR/TotalText/CTW1500 style)
utils/          # Logger, metrics, post-processing
train.py        # Training script
test.py         # Evaluation script
inference.py    # Inference script
requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Data Preparation
- Place images in `images/` and annotations (polygon txt files) in `annotations/`.
- Each annotation file should be named as the image, with `.txt` extension, and contain comma-separated polygon coordinates per line.

## Training
### Train from scratch
```bash
python train.py --img_dir images/ --ann_dir annotations/ --save_dir checkpoints/
```

### Fine-tune or Resume from a Checkpoint
```bash
python train.py --img_dir images/ --ann_dir annotations/ --save_dir checkpoints/ --resume path/to/your_checkpoint.pth
```
- Use the `--resume` argument to load model weights from a previous checkpoint for fine-tuning or resuming training.

## Testing
```bash
python test.py --img_dir images/ --ann_dir annotations/ --ckpt checkpoints/fast_epoch10.pth
```

## Inference
```bash
python inference.py --img_dir images/ --ann_dir annotations/ --ckpt checkpoints/fast_epoch10.pth --out_dir inference_results/
```
