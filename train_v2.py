"""Train UAV RetinaNet v2.

Usage:
  python train_v2.py --data dataset --epochs 30
"""
import argparse, random, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from src.uav_detector.dataset import VOCDataset, collate_fn
from src.uav_detector.model import build_model

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='dataset'); p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=4); p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--output', default='checkpoints'); args = p.parse_args()
    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = VOCDataset(args.data, train=True)
    n_val = max(1, int(0.15 * len(ds))); n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    Path(args.output).mkdir(exist_ok=True)
    best = float('inf')
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for images, targets in train_loader:
            images = [x.to(device) for x in images]
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                losses = model(images, targets); loss = sum(losses.values())
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer); scaler.update(); running += loss.item()
        scheduler.step(); avg = running / max(1, len(train_loader))
        print(f'Epoch {epoch+1:03d}/{args.epochs} | loss={avg:.4f} | lr={scheduler.get_last_lr()[0]:.2e}')
        if avg < best:
            best = avg; torch.save({'model': model.state_dict(), 'epoch': epoch, 'loss': avg}, Path(args.output)/'best.pt')
    print(f'Finished on {device}. Best training loss: {best:.4f}')

if __name__ == '__main__': main()
