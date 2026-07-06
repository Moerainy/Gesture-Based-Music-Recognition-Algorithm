import os
import argparse
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from dataloader import GestureDataset, get_dataloader
from train_3dcnn import compute_class_weights, validate, train_one_epoch

class LogisticsRegression(nn.Module):
    """Linear classifier. Accepts either (B, 3, 16, 112, 112) raw video
    or pre-flattened (B, 602112) from shards."""
    def __init__(self, in_features=3 * 16 * 112 * 112, num_classes=10, lbd=1):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.lbd = torch.tensor(lbd)

    def forward(self, x):
        if x.ndim == 5:                     # raw video: (B, C, T, H, W)
            x = x.flatten(start_dim=1)       # (B, 602112)
        # else: already flat (B, 602112) from shards
        return self.linear(x)

def parse_args():
    """Parse command-line arguments for training.

    Returns:
        argparse.Namespace with training hyperparameters.
    """
    p = argparse.ArgumentParser(description='Train R2Plus1D gesture classifier')
    p.add_argument('--data', type=str, default='dataset',
                   help='path to dataset directory')
    p.add_argument('--epochs', type=int, default=50,
                   help='total training epochs')
    p.add_argument('--batch_size', type=int, default=32,
                   help='mini-batch size')
    p.add_argument('--lr', type=float, default=1e-3,
                   help='initial learning rate')
    p.add_argument('--weight_decay', type=float, default=1,
                   help='AdamW weight decay')
    p.add_argument('--num_frames', type=int, default=16,
                   help='number of frames sampled per clip')
    p.add_argument('--num_workers', type=int, default=2,
                   help='DataLoader worker processes')
    p.add_argument('--seed', type=int, default=42,
                   help='random seed for reproducible splits')
    p.add_argument('--resume', type=str, default=None,
                   help='resume training from a checkpoint file')
    p.add_argument('--log_dir', type=str, default='logs/logistics_runs',
                   help='TensorBoard log directory')
    p.add_argument('--patience', type=int, default=8,
                   help='early stopping patience (epochs)')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    model = LogisticsRegression().to(device)

    # Data
    train_loader = get_dataloader(
        args.data, batch_size=args.batch_size,
        num_frames=args.num_frames, num_workers=args.num_workers,
        train=True, seed=args.seed,
    )
    val_loader = get_dataloader(
        args.data, batch_size=args.batch_size,
        num_frames=args.num_frames, num_workers=args.num_workers,
        train=False, seed=args.seed,
    )

    num_classes = train_loader.dataset.dataset.num_classes
    class_weights = compute_class_weights(train_loader.dataset, num_classes)

    USE_WEIGHTED_LOSS = True
    if USE_WEIGHTED_LOSS:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer, scheduler, AMP
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')

    # Resume from checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f'Resumed from {args.resume}, starting at epoch {start_epoch}')

    # TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(args.log_dir, timestamp))

    print(f'Device: {device}')
    print(f'Num classes: {num_classes}')
    print(f'Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}')
    print(f'Class weights: {class_weights.tolist()}')

    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch {epoch:3d}/{args.epochs}  '
              f'train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  '
              f'val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  '
              f'lr={optimizer.param_groups[0]["lr"]:.2e}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
            }, 'models/best_logistics.pt')
            print(f'  -> Saved best model (val_acc={val_acc:.4f})')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping: val_loss not improved for {args.patience} epochs')
                break

        # Save latest checkpoint for resuming
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
        }, 'models/last_logistics.pt')

    writer.close()
    print(f'Training finished, best val_acc={best_val_acc:.4f}')

    