import os
import glob
import numpy as np
from PIL import Image, ImageDraw
import subprocess
import torch

# =========================
# 1. SYNTHETIC DATA CREATION
# =========================

def create_synthetic_data():
    os.makedirs('images', exist_ok=True)
    os.makedirs('annotations', exist_ok=True)
    for i in range(3):
        img = Image.new('RGB', (256, 256), color=(255,255,255))
        draw = ImageDraw.Draw(img)
        x0, y0 = 50 + 20*i, 60 + 10*i
        x1, y1 = 180 + 10*i, 120 + 20*i
        draw.rectangle([x0, y0, x1, y1], outline='black', width=3)
        img.save(f'images/img{i}.jpg')
        poly = [x0, y0, x1, y0, x1, y1, x0, y1]
        with open(f'annotations/img{i}.txt', 'w') as f:
            f.write(','.join(map(str, poly)) + '\n')
    print("Sample image and annotation created:")
    img = Image.open('images/img0.jpg')
    img.show()
    with open('annotations/img0.txt') as f:
        print(f.read())

# =========================
# 2. TRAINING
# =========================

def run_training():
    print("=== TRAINING ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cmd = [
        'python', 'train.py',
        '--img_dir', 'images',
        '--ann_dir', 'annotations',
        '--save_dir', 'checkpoints',
        '--epochs', '2',  # quick test
        '--batch_size', '2',
        '--img_size', '128',
        '--num_kernels', '3',
        '--device', device,
        '--dilation_size', '9'
    ]
    subprocess.run(cmd, check=True)

# =========================
# 3. TESTING
# =========================

def run_testing():
    print("=== TESTING ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpts = sorted(glob.glob('checkpoints/*.pth'))
    ckpt = ckpts[-1] if ckpts else None
    print("Using checkpoint:", ckpt)
    if ckpt:
        cmd = [
            'python', 'test.py',
            '--img_dir', 'images',
            '--ann_dir', 'annotations',
            '--ckpt', ckpt,
            '--batch_size', '2',
            '--img_size', '128',
            '--num_kernels', '3',
            '--device', device,
            '--dilation_size', '9'
        ]
        subprocess.run(cmd, check=True)

# =========================
# 4. INFERENCE
# =========================

def run_inference():
    print("=== INFERENCE ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpts = sorted(glob.glob('checkpoints/*.pth'))
    ckpt = ckpts[-1] if ckpts else None
    if ckpt:
        cmd = [
            'python', 'inference.py',
            '--img_dir', 'images',
            '--ann_dir', 'annotations',
            '--ckpt', ckpt,
            '--out_dir', 'inference_results',
            '--batch_size', '2',
            '--img_size', '128',
            '--num_kernels', '3',
            '--device', device,
            '--dilation_size', '9'
        ]
        subprocess.run(cmd, check=True)

# =========================
# 5. CHECK OUTPUTS
# =========================

def check_outputs():
    npy_files = glob.glob('inference_results/*.npy')
    print("Inference output files:", npy_files)
    if npy_files:
        polys = np.load(npy_files[0], allow_pickle=True)
        print("Polygons for first image:", polys)

# =========================
# MAIN
# =========================

def main():
    create_synthetic_data()
    run_training()
    run_testing()
    run_inference()
    check_outputs()
    print("\n[INFO] Mixed precision is automatically enabled if CUDA is available.")

if __name__ == '__main__':
    main() 