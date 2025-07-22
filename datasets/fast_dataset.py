import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class FASTDataset(Dataset):
    """Dataset for FAST: loads images and polygon annotations, generates text and kernel masks."""
    def __init__(self, img_dir, ann_dir, transform=None, num_kernels=6, img_size=640):
        # Store dataset paths and config
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.num_kernels = num_kernels
        self.img_size = img_size
        # List all image files
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('jpg','png','jpeg'))]
    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.img_names)
    def __getitem__(self, idx):
        """Loads image and annotation, generates text and kernel masks, returns as tensors."""
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, os.path.splitext(img_name)[0]+'.txt')
        # Load and resize image
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size))
        except Exception as e:
            print(f"[ERROR] Could not load image {img_path}: {e}")
            raise
        # Read polygon annotation file
        polygons = []
        try:
            with open(ann_path, 'r') as f:
                for line in f:
                    try:
                        # Parse comma-separated coordinates
                        pts = np.array(line.strip().split(',')).astype(np.float32).reshape(-1,2)
                        # Scale polygon to image size
                        polygons.append(pts * (self.img_size / max(img.size)))
                    except Exception as e:
                        print(f"[ERROR] Could not parse polygon in {ann_path}: {e}")
        except Exception as e:
            print(f"[ERROR] Could not read annotation file {ann_path}: {e}")
            raise
        # Generate text and kernel masks (all zeros initially)
        text_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        kernel_masks = []
        for i in range(self.num_kernels-1):
            kernel_masks.append(np.zeros((self.img_size, self.img_size), dtype=np.uint8))
        # Fill polygons for text and each kernel (shrunk)
        for poly in polygons:
            try:
                cv2.fillPoly(text_mask, [poly.astype(np.int32)], 1)
                for i in range(self.num_kernels-1):
                    # Shrink polygon for kernel mask
                    rate = 1.0 - 0.5 * (self.num_kernels-2-i)/(self.num_kernels-2)
                    shrinked = poly * rate + poly.mean(axis=0) * (1-rate)
                    cv2.fillPoly(kernel_masks[i], [shrinked.astype(np.int32)], 1)
            except Exception as e:
                print(f"[ERROR] Could not fill polygon for {img_name}: {e}")
        kernel_masks = np.stack(kernel_masks, axis=0)
        # Training mask (all ones by default, can be used for ignore regions)
        training_mask = np.ones((self.img_size, self.img_size), dtype=np.uint8)
        # Convert all to torch tensors
        try:
            img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.
            text_mask = torch.from_numpy(text_mask).unsqueeze(0).float()
            kernel_masks = torch.from_numpy(kernel_masks).float()
            training_mask = torch.from_numpy(training_mask).unsqueeze(0).float()
        except Exception as e:
            print(f"[ERROR] Could not convert data to tensor for {img_name}: {e}")
            raise
        # Return all tensors in a dict
        sample = {
            'image': img,
            'gt_text': text_mask,
            'gt_kernels': kernel_masks,
            'training_mask': training_mask,
            'img_name': img_name
        }
        # Optionally apply transform
        if self.transform:
            try:
                sample = self.transform(sample)
            except Exception as e:
                print(f"[ERROR] Transform failed for {img_name}: {e}")
        return sample 