"""Evaluate a trained model on the gesture classification task.

Reports loss, overall accuracy, confusion matrix, and per-class
precision / recall / F1-score on both training and validation sets.

Supports:
    --model_type 3dcnn     R2Plus1D-18 (checkpoint: models/best.pt)
    --model_type lstm      CNN-LSTM (checkpoint: models/best_lstm.pt)
    --model_type logistics Logistic regression (checkpoint: models/best_logistics.pt)
    --model_type all       Evaluate all three models (default)
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
)
from dataloader import get_dataloader
from train_3dcnn import build_model as build_3dcnn_model
from lstm_cnn import build_model as build_lstm_model
from logistics import LogisticsRegression


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run inference on the entire loader and collect predictions.

    Args:
        model: Trained model.
        loader: DataLoader over the evaluation split.
        criterion: Loss function.
        device: torch device.

    Returns:
        Tuple of (average_loss, all_labels, all_predictions).
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    total = 0

    for videos, labels in tqdm(loader, desc='Validating', unit='batch', total=len(loader)):
        videos, labels = videos.to(device), labels.to(device)

        with torch.amp.autocast('cuda'):
            output = model(videos)
            loss = criterion(output, labels)

        total_loss += loss.item() * videos.size(0)
        preds = output.argmax(dim=1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        total += videos.size(0)

    return total_loss / total, np.array(all_labels), np.array(all_preds)


def print_metrics(loss, labels, preds, class_names, title):
    """Print a full metrics report for one data split.

    Args:
        loss: Scalar loss value.
        labels, preds: numpy arrays of ground-truth and predicted labels.
        class_names: List of class name strings.
        title: Header string for this report block.
    """
    num_classes = len(class_names)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=range(num_classes), zero_division=0,
    )

    print(f'\n{"="*60}')
    print(f'  {title}')
    print(f'{"="*60}')
    print(f'  Loss:      {loss:.4f}')
    print(f'  Accuracy:  {acc:.4f}')
    print(f'{"="*60}\n')

    # Per-class table
    print(f'{"Class":35s} {"Prec":>6s}  {"Rec":>6s}  {"F1":>6s}  {"Support":>7s}')
    print('-' * 67)
    for i, name in enumerate(class_names):
        print(f'{name:35s} {precision[i]:6.4f}  {recall[i]:6.4f}  {f1[i]:6.4f}  {support[i]:7d}')

    # Macro / weighted averages
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0,
    )
    weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0,
    )
    print('-' * 67)
    print(f'{"Macro avg":35s} {macro_p:6.4f}  {macro_r:6.4f}  {macro_f:6.4f}')
    print(f'{"Weighted avg":35s} {weighted_p:6.4f}  {weighted_r:6.4f}  {weighted_f:6.4f}')

    # Confusion matrix
    print(f'\n{" Confusion Matrix ":=^67s}')
    header = ' ' * 7 + ''.join(f'{i:>5d}' for i in range(num_classes))
    print(header)
    for i, row in enumerate(cm):
        print(f'{i:3d} {class_names[i][:30]:30s} ' + ''.join(f'{v:5d}' for v in row))


def build_model_from_args(args, num_classes, device, model_type):
    """Build the appropriate model based on model_type.

    Args:
        args: Parsed command-line arguments.
        num_classes: Number of output classes.
        device: torch device.
        model_type: One of '3dcnn', 'lstm', 'logistics'.

    Returns:
        Instantiated model (weights not loaded).
    """
    if model_type == '3dcnn':
        model, _ = build_3dcnn_model(num_classes=num_classes)
    elif model_type == 'lstm':
        model = build_lstm_model(
            num_classes=num_classes,
            hidden_size=args.hidden_size,
            num_lstm_layers=args.num_lstm_layers,
        )
    elif model_type == 'logistics':
        in_features = 3 * args.num_frames * 112 * 112
        model = LogisticsRegression(in_features=in_features, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown model_type: {model_type}')
    return model.to(device)


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate a gesture classifier')
    p.add_argument('--model_type', type=str, default='all',
                   choices=['3dcnn', 'lstm', 'logistics', 'all'],
                   help='model architecture (default: evaluate all)')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='path to model checkpoint (required unless --model_type all)')
    p.add_argument('--data', type=str, default='dataset',
                   help='path to dataset directory')
    p.add_argument('--batch_size', type=int, default=16,
                   help='mini-batch size')
    p.add_argument('--num_frames', type=int, default=16,
                   help='number of frames sampled per clip')
    p.add_argument('--num_workers', type=int, default=2,
                   help='DataLoader worker processes')
    p.add_argument('--seed', type=int, default=42,
                   help='random seed for reproducible splits')
    p.add_argument('--splits', type=str, default='both',
                   choices=['train', 'val', 'both'],
                   help='which data splits to evaluate')
    p.add_argument('--hidden_size', type=int, default=256,
                   help='LSTM hidden state size (only for --model_type lstm)')
    p.add_argument('--num_lstm_layers', type=int, default=2,
                   help='number of LSTM layers (only for --model_type lstm)')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    criterion = nn.CrossEntropyLoss()

    # Load one loader first to get class metadata
    temp_loader = get_dataloader(
        args.data, batch_size=args.batch_size,
        num_frames=args.num_frames, num_workers=args.num_workers,
        train=False, seed=args.seed,
    )
    num_classes = temp_loader.dataset.dataset.num_classes
    class_names = temp_loader.dataset.dataset.classes

    # Determine which model types and checkpoints to evaluate
    default_checkpoints = {
        '3dcnn': 'models/best.pt',
        'lstm': 'models/best_lstm.pt',
        'logistics': 'models/best_logistics.pt',
    }

    if args.model_type == 'all':
        to_evaluate = [
            (mt, args.checkpoint or ckpt)
            for mt, ckpt in default_checkpoints.items()
        ]
    else:
        if args.checkpoint is None:
            raise ValueError('--checkpoint is required when --model_type is specified')
        to_evaluate = [(args.model_type, args.checkpoint)]

    for model_type, ckpt_path in to_evaluate:
        print(f'\n{"#"*60}')
        print(f'#  Model: {model_type}  |  Checkpoint: {ckpt_path}')
        print(f'{"#"*60}')

        model = build_model_from_args(args, num_classes, device, model_type)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'Loaded checkpoint: {ckpt_path} (epoch {ckpt["epoch"]})')
        print(f'Model type: {model_type}')
        print(f'Num classes: {num_classes}')

        if args.splits in ('train', 'both'):
            train_loader = get_dataloader(
                args.data, batch_size=args.batch_size,
                num_frames=args.num_frames, num_workers=args.num_workers,
                train=True, seed=args.seed,
            )
            train_loss, train_labels, train_preds = evaluate(
                model, train_loader, criterion, device)
            print_metrics(train_loss, train_labels, train_preds, class_names,
                          title=f'Training Set [{model_type}]')

        if args.splits in ('val', 'both'):
            val_loader = get_dataloader(
                args.data, batch_size=args.batch_size,
                num_frames=args.num_frames, num_workers=args.num_workers,
                train=False, seed=args.seed,
            )
            val_loss, val_labels, val_preds = evaluate(
                model, val_loader, criterion, device)
            print_metrics(val_loss, val_labels, val_preds, class_names,
                          title=f'Validation Set [{model_type}]')

    # Column label key
    print(f'\nColumn indices:')
    for i, name in enumerate(class_names):
        print(f'  {i}: {name}')


if __name__ == '__main__':
    main()
