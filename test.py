import os
import torch
from torch.utils.data import DataLoader
from models import FAST
from datasets import FASTDataset
from utils.logger import SimpleLogger
from utils.metrics import compute_f1
from tqdm import tqdm
import numpy as np

def test(
    img_dir, ann_dir, ckpt_path, batch_size=8, num_kernels=6, img_size=640, device='cuda'):
    logger = SimpleLogger()
    dataset = FASTDataset(img_dir, ann_dir, num_kernels=num_kernels, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    model = FAST(num_kernels=num_kernels).to(device)
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    except Exception as e:
        logger.info(f"[ERROR] Failed to load checkpoint {ckpt_path}: {e}")
        raise
    model.eval()
    all_prec, all_rec, all_f1 = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            imgs = batch['image'].to(device)
            gt_text = batch['gt_text'].cpu().numpy()
            pred = torch.sigmoid(model(imgs))
            # Upsample prediction to match ground truth size
            pred = torch.nn.functional.interpolate(pred, size=gt_text.shape[2:], mode='bilinear', align_corners=False)
            pred = pred.cpu().numpy()
            for i in range(imgs.size(0)):
                pred_bin = (pred[i,0] > 0.5).astype(np.uint8)
                gt_bin = (gt_text[i,0] > 0.5).astype(np.uint8)
                prec, rec, f1 = compute_f1(pred_bin, gt_bin)
                all_prec.append(prec)
                all_rec.append(rec)
                all_f1.append(f1)
    logger.info(f'Precision: {np.mean(all_prec):.4f}, Recall: {np.mean(all_rec):.4f}, F1: {np.mean(all_f1):.4f}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--ann_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    test(
        args.img_dir, args.ann_dir, args.ckpt, args.batch_size,
        args.num_kernels, args.img_size, args.device
    ) 