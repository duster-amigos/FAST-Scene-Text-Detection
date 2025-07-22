import os
import torch
from torch.utils.data import DataLoader
from models import FAST
from losses import FASTLoss
from datasets import FASTDataset
from utils.logger import SimpleLogger
from tqdm import tqdm

def train(
    img_dir, ann_dir, save_dir, num_epochs=300, batch_size=8, lr=1e-3, num_kernels=6, img_size=640, device='cuda', resume=None):
    os.makedirs(save_dir, exist_ok=True)
    logger = SimpleLogger(os.path.join(save_dir, 'train.log'))
    dataset = FASTDataset(img_dir, ann_dir, num_kernels=num_kernels, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    model = FAST(num_kernels=num_kernels).to(device)
    if resume is not None:
        try:
            model.load_state_dict(torch.load(resume, map_location=device))
            logger.info(f"Loaded weights from {resume}")
        except Exception as e:
            logger.info(f"[ERROR] Failed to load checkpoint {resume}: {e}")
            raise
    criterion = FASTLoss(num_kernels=num_kernels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            imgs = batch['image'].to(device)
            gt_text = batch['gt_text'].to(device)
            gt_kernels = batch['gt_kernels'].to(device)
            training_mask = batch['training_mask'].to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss, loss_dict = criterion(pred, gt_text, gt_kernels, training_mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({k: f'{v:.4f}' for k, v in loss_dict.items()})
        logger.info(f'Epoch {epoch+1}: Loss={epoch_loss/len(dataloader):.4f}')
        # Save checkpoint
        if (epoch+1) % 10 == 0:
            try:
                torch.save(model.state_dict(), os.path.join(save_dir, f'fast_epoch{epoch+1}.pth'))
            except Exception as e:
                logger.info(f"[ERROR] Failed to save checkpoint at epoch {epoch+1}: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--ann_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume/fine-tune from')
    args = parser.parse_args()
    train(
        args.img_dir, args.ann_dir, args.save_dir, args.epochs, args.batch_size,
        args.lr, args.num_kernels, args.img_size, args.device, args.resume
    ) 