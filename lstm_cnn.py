import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision.models import resnet18, ResNet18_Weights

from dataloader import get_dataloader
from train_3dcnn import compute_class_weights, train_one_epoch, validate


class CNN_LSTM_Jester(nn.Module):
    def __init__(self, num_classes=10, hidden_size=256, num_lstm_layers=2):
        super().__init__()

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size,
                            num_layers=num_lstm_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.size()

        x = x.transpose(1, 2).contiguous().view(B * T, C, H, W)

        spatial_features = self.feature_extractor(x)

        spatial_features = spatial_features.view(B, T, -1)

        r_out, _ = self.lstm(spatial_features)

        out = self.fc(r_out[:, -1, :])

        return out


def build_model(num_classes=None, hidden_size=256, num_lstm_layers=2, frozen_resnet=False):
    if num_classes is None:
        root_dir = os.path.join(os.path.dirname(__file__), 'dataset')
        num_classes = len([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

    model = CNN_LSTM_Jester(num_classes=num_classes,
                            hidden_size=hidden_size,
                            num_lstm_layers=num_lstm_layers)

    if frozen_resnet:
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

    return model


def parse_args():
    p = argparse.ArgumentParser(description='Train CNN-LSTM gesture classifier')
    p.add_argument('--data', type=str, default='dataset',
                   help='path to dataset directory')
    p.add_argument('--epochs', type=int, default=50,
                   help='total training epochs')
    p.add_argument('--batch_size', type=int, default=8,
                   help='mini-batch size')
    p.add_argument('--lr', type=float, default=1e-3,
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
    p.add_argument('--log_dir', type=str, default='logs/lstm_cnn_runs',
                   help='TensorBoard log directory')
    p.add_argument('--patience', type=int, default=8,
                   help='early stopping patience (epochs)')
    p.add_argument('--hidden_size', type=int, default=256,
                   help='LSTM hidden state size')
    p.add_argument('--num_lstm_layers', type=int, default=2,
                   help='number of LSTM layers')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    model = build_model(num_classes=None,
                        hidden_size=args.hidden_size,
                        num_lstm_layers=args.num_lstm_layers)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'Trainable params: {trainable:,}  Frozen params: {frozen:,}')

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
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f'Resumed from {args.resume}, starting at epoch {start_epoch}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(args.log_dir, f'lstm_{timestamp}'))

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
            }, 'models/best_lstm.pt')
            print(f'  -> Saved best model (val_acc={val_acc:.4f})')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping: val_loss not improved for {args.patience} epochs')
                break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
        }, 'models/last_lstm.pt')

    writer.close()
    print(f'Training finished, best val_acc={best_val_acc:.4f}')


if __name__ == '__main__':
    main()