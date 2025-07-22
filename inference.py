import os
import torch
from torch.utils.data import DataLoader
from models import FAST
from datasets import FASTDataset
from utils.logger import SimpleLogger
from utils.postprocess import fast_postprocess
from tqdm import tqdm
import numpy as np

def inference(
    img_dir, ann_dir, ckpt_path, out_dir, batch_size=8, num_kernels=6, img_size=640, device='cuda'):
    os.makedirs(out_dir, exist_ok=True)
    logger = SimpleLogger(os.path.join(out_dir, 'inference.log'))
    dataset = FASTDataset(img_dir, ann_dir, num_kernels=num_kernels, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    model = FAST(num_kernels=num_kernels).to(device)
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    except Exception as e:
        logger.info(f"[ERROR] Failed to load checkpoint {ckpt_path}: {e}")
        raise
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Inference'):
            imgs = batch['image'].to(device)
            img_names = batch['img_name']
            pred = torch.sigmoid(model(imgs))
            # Upsample prediction to match input image size
            pred = torch.nn.functional.interpolate(pred, size=imgs.shape[2:], mode='bilinear', align_corners=False)
            polygons_batch = fast_postprocess(pred)
            for img_name, polygons in zip(img_names, polygons_batch):
                out_path = os.path.join(out_dir, img_name + '.npy')
                try:
                    np.save(out_path, polygons)
                except Exception as e:
                    logger.info(f"[ERROR] Failed to save polygons for {img_name}: {e}")
            logger.info(f'Processed batch: {[n for n in img_names]}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--ann_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='inference_results')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    inference(
        args.img_dir, args.ann_dir, args.ckpt, args.out_dir, args.batch_size,
        args.num_kernels, args.img_size, args.device
    ) 