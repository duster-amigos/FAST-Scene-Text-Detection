import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class FASTDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None, num_kernels=6, img_size=640):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.num_kernels = num_kernels
        self.img_size = img_size
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('jpg','png','jpeg'))]
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, os.path.splitext(img_name)[0]+'.txt')
        # Try loading image
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size))
        except Exception as e:
            print(f"[ERROR] Could not load image {img_path}: {e}")
            raise
        # Try reading annotation
        polygons = []
        try:
            with open(ann_path, 'r') as f:
                for line in f:
                    try:
                        pts = np.array(line.strip().split(',')).astype(np.float32).reshape(-1,2)
                        polygons.append(pts * (self.img_size / max(img.size)))
                    except Exception as e:
                        print(f"[ERROR] Could not parse polygon in {ann_path}: {e}")
        except Exception as e:
            print(f"[ERROR] Could not read annotation file {ann_path}: {e}")
            raise
        # Generate masks
        text_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        kernel_masks = []
        for i in range(self.num_kernels-1):
            kernel_masks.append(np.zeros((self.img_size, self.img_size), dtype=np.uint8))
        for poly in polygons:
            try:
                cv2.fillPoly(text_mask, [poly.astype(np.int32)], 1)
                for i in range(self.num_kernels-1):
                    rate = 1.0 - 0.5 * (self.num_kernels-2-i)/(self.num_kernels-2)  # shrinkage for kernels
                    shrinked = poly * rate + poly.mean(axis=0) * (1-rate)
                    cv2.fillPoly(kernel_masks[i], [shrinked.astype(np.int32)], 1)
            except Exception as e:
                print(f"[ERROR] Could not fill polygon for {img_name}: {e}")
        kernel_masks = np.stack(kernel_masks, axis=0)
        # Training mask (all ones, can be refined for ignore regions)
        training_mask = np.ones((self.img_size, self.img_size), dtype=np.uint8)
        # To tensor
        try:
            img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.
            text_mask = torch.from_numpy(text_mask).unsqueeze(0).float()
            kernel_masks = torch.from_numpy(kernel_masks).float()
            training_mask = torch.from_numpy(training_mask).unsqueeze(0).float()
        except Exception as e:
            print(f"[ERROR] Could not convert data to tensor for {img_name}: {e}")
            raise
        sample = {
            'image': img,
            'gt_text': text_mask,
            'gt_kernels': kernel_masks,
            'training_mask': training_mask,
            'img_name': img_name
        }
        if self.transform:
            try:
                sample = self.transform(sample)
            except Exception as e:
                print(f"[ERROR] Transform failed for {img_name}: {e}")
        return sample 