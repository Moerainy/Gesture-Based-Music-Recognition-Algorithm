import os
import argparse
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from datetime import datetime
from tqdm import tqdm

from dataloader import GestureDataset, get_dataloader

def build_model(num_classes=None):
    """Build an R2Plus1D-18 video classification model with pretrained weights.

    Args:
        num_classes: Number of output classes. If None, inferred from the
            number of subdirectories under dataset/.

    Returns:
        Tuple of (model, weights) where weights is the transform config
        from the pretrained checkpoint.
    """
    if num_classes is None:
        root_dir = os.path.join(os.path.dirname(__file__), 'dataset')
        num_classes = len([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

    weights = R2Plus1D_18_Weights.DEFAULT
    model = r2plus1d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model, weights


def compute_class_weights(dataset, num_classes):
    """Compute inverse-frequency class weights for weighted CrossEntropyLoss.

    Args:
        dataset: A Subset wrapping a GestureDataset.
        num_classes: Total number of classes.

    Returns:
        Tensor of shape (num_classes,) with per-class weights.
    """
    base = dataset.dataset  # Subset -> GestureDataset
    labels = [base.samples[i][1] for i in dataset.indices]
    counts = Counter(labels)
    total = sum(counts.values())
    weight = torch.zeros(num_classes)
    for c in range(num_classes):
        weight[c] = total / (num_classes * counts.get(c, 1))
    return weight


def train_one_epoch(model: nn.Module,
                    loader: torch.utils.data.DataLoader, 
                    criterion: nn.modules.loss._WeightedLoss, 
                    optimizer: torch.optim.Optimizer, 
                    scaler: torch.amp.GradScaler, 
                    device
                    ):
    """Run a single training epoch with mixed precision.

    Args:
        model: The R2Plus1D model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: GradScaler for AMP.
        device: torch device.

    Returns:
        Tuple of (average loss, accuracy) over the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for videos, labels in tqdm(loader, desc='Training', unit='batch', total=len(loader)):
        videos, labels = videos.to(device), labels.to(device)

        with torch.amp.autocast('cuda'):
            output: torch.Tensor = model(videos)
            loss: torch.Tensor = criterion(output, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item() * videos.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += videos.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluate model on the validation set.

    Args:
        model: The R2Plus1D model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: torch device.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for videos, labels in tqdm(loader, desc='Validating', unit='batch', total=len(loader)):
        videos, labels = videos.to(device), labels.to(device)

        with torch.amp.autocast('cuda'):
            output = model(videos)
            loss = criterion(output, labels)

        total_loss += loss.item() * videos.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += videos.size(0)

    return total_loss / total, correct / total


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
    p.add_argument('--batch_size', type=int, default=8,
                   help='mini-batch size')
    p.add_argument('--lr', type=float, default=1e-4,
                   help='initial learning rate')
    p.add_argument('--weight_decay', type=float, default=1e-4,
                   help='AdamW weight decay')
    p.add_argument('--num_frames', type=int, default=16,
                   help='number of frames sampled per clip')
    p.add_argument('--num_workers', type=int, default=2,
                   help='DataLoader worker processes')
    p.add_argument('--seed', type=int, default=42,
                   help='random seed for reproducible splits')
    p.add_argument('--resume', type=str, default=None,
                   help='resume training from a checkpoint file')
    p.add_argument('--log_dir', type=str, default='logs/runs',
                   help='TensorBoard log directory')
    p.add_argument('--patience', type=int, default=8,
                   help='early stopping patience (epochs)')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    # Model
    model, weights = build_model()
    model = model.to(device)

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

    # Class weights
    num_classes = train_loader.dataset.dataset.num_classes
    class_weights = compute_class_weights(train_loader.dataset, num_classes)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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
            }, 'models/best.pt')
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
        }, 'models/last.pt')

    writer.close()
    print(f'Training finished, best val_acc={best_val_acc:.4f}')


if __name__ == '__main__':
    main()
