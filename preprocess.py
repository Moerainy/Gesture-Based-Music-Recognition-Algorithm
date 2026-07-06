import torch
from torchvision.io import read_image
from pathlib import Path
import shutil
import argparse
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count

raw_dataset_dir = Path(__file__).parent / "rawdata"

def folder2pt(folder: Path):
    """Read all images in a folder, stack them into a video tensor, resize to 112x112, and save as .pt file."""
    if not folder.exists():
        return
    frames = []
    for img_path in sorted(folder.iterdir()):
        img = read_image(str(img_path))
        frames.append(img)

    video = torch.stack(frames).to(torch.uint8)  # [T, 3, H, W]
    video = video.to(torch.float32)
    video = torch.nn.functional.interpolate(video, size=(112, 112), mode='bilinear', align_corners=False)
    video = video.to(torch.uint8)

    pt_path = Path("dataset") / f"{folder.name}.pt"
    torch.save(video, pt_path)

def parse_args():
    p = argparse.ArgumentParser(description='Save the videos as .pt files that can be directly loaded by PyTorch')
    p.add_argument('--all', 
                   action='store_true', 
                   help='Transfer all 27 classes')
    
    p.add_argument('--classes', 
                   nargs='+',
                   default=['Doing other things', 
                            'No gesture',
                            'Sliding Two Fingers Left', 
                            'Sliding Two Fingers Right',
                            'Stop Sign', 
                            'Swiping Down',
                            'Swiping Left', 
                            'Swiping Right',
                            'Zooming In With Two Fingers', 
                            'Zooming Out With Two Fingers'
                            ], 
                   help='Declear the classes that you want to use.\n When --all is True, this argument will be ignored. If it is not given, default classes will be empolyed.')
    return p.parse_args()

if __name__ == "__main__":
    # Create output directory for .pt files
    dataset_dir = Path(__file__).parent / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    # Load and sort labels by video ID
    dataset = pd.read_csv('label.csv', index_col='video_id')
    dataset.sort_index(inplace=True)
    dataset.to_csv("label.csv")

    args = parse_args()
    if args.all:
        used_categories = dataset['label'].drop_duplicates().tolist()
    else:
        used_categories = args.classes

    # Collect raw frame folders that have a matching label in used_categories
    folders = []
    for vid in dataset.index:
        folder = raw_dataset_dir / str(vid)
        if not folder.is_dir():
            continue
        if not folder.exists():
            continue
        if not dataset.at[vid, 'label'] in used_categories:
            continue
        folders.append(folder)

    # Convert each folder of frames to a .pt video tensor in parallel
    print(f"Converting images to .pt checkpoints and resizing videos ({len(folders)} folders)...")
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(folder2pt, folders), total=len(folders)))

    # Move .pt files into subdirectories named by gesture category
    print("Copying checkpoints to category folders...")
    failure = 0
    for id, c in tqdm(zip(dataset.index, dataset['label']), total=len(dataset)):
        try:
            origin = dataset_dir / (str(id) + ".pt")
            if not origin.exists():
                continue
            if not c in used_categories:
                continue
            dst = dataset_dir / str(c)
            if not dst.exists():
                dst.mkdir()
            shutil.move(dataset_dir / (str(id) + ".pt"), dst / (str(id) + ".pt"))
        except Exception:
            failure += 1
    print(f"Copy complete, {failure} failures")

    # Clean up raw frame images to save disk space
    print("Removing rawdata directory...")
    shutil.rmtree(raw_dataset_dir)
    print("Removal complete")