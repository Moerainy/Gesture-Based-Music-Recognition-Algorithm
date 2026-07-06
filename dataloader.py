import os
import torch
from torch.utils.data import Dataset, DataLoader

class GestureDataset(Dataset):
    """Load .pt gesture video tensors from dataset/.

    Each .pt file has shape (37, 3, 112, 112), dtype=uint8.
    Preprocessing: uint8→float32, uniformly sample T frames, normalize.
    """

    def __init__(self, root_dir, num_frames=16,
                 mean=(0.43216, 0.39467, 0.37645),
                 std=(0.22803, 0.22145, 0.21699),
                 train=True):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.mean = mean
        self.std = std
        self.train = train

        self.classes = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.endswith('.pt'):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def _load_and_preprocess(self, path):
        tensor = torch.load(path, weights_only=True)  # (37, 3, 112, 112), uint8

        # uint8 -> float32, normalize to [0, 1]
        tensor = tensor.to(torch.float32) / 255.0

        # Uniformly sample num_frames frames
        total_frames = tensor.shape[0]
        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        tensor = tensor[indices]  # (num_frames, 3, 112, 112)

        # Apply Kinetics normalization
        tensor = (tensor - tensor.new_tensor(self.mean).view(1, 3, 1, 1)) \
                 / tensor.new_tensor(self.std).view(1, 3, 1, 1)

        # Rearrange dimensions: (T, C, H, W) -> (C, T, H, W)
        tensor = tensor.permute(1, 0, 2, 3)

        return tensor

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = self._load_and_preprocess(path)
        return tensor, label
    
def get_dataloader(root_dir, batch_size=4, num_frames=16,
                   num_workers=2, train=True, val_split=0.2, seed=42):
    """Create train and validation DataLoaders (stratified split, 8:2 per class).

    Args:
        root_dir: Path to resized_dataset directory
        batch_size: Batch size (4-8 recommended for 8GB VRAM)
        num_frames: Number of frames to sample (R2Plus1D default 16)
        num_workers: Number of data loading workers
        train: Return training set (True) or validation set (False)
        val_split: Validation set ratio
        seed: Random seed
    """
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split

    dataset = GestureDataset(
        root_dir=root_dir,
        num_frames=num_frames,
        train=train,
    )

    # Stratified split: split train/val by class proportion
    labels = [label for _, label in dataset.samples]
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=labels,
        random_state=seed,
    )

    target_idx = train_idx if train else val_idx

    return DataLoader(
        Subset(dataset, target_idx),
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        prefetch_factor=2,            
        persistent_workers=True      
    )