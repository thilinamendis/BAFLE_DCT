import glob
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

class dataset_(Dataset):
    def __init__(self, cover_img_dir, stego_img_dir, transform):
        self.cover_img_dir = cover_img_dir
        self.stego_img_dir = stego_img_dir
        self.transforms = transform
        self.cover_img_filenames = list(sorted(os.listdir(cover_img_dir)))
        self.stego_img_filenames = list(sorted(os.listdir(stego_img_dir)))
    
    def __len__(self):
        return len(self.cover_img_filenames)
    
    def __getitem__(self, index):
        cover_img_paths = os.path.join(self.cover_img_dir, self.cover_img_filenames[index])
        stego_img_paths = os.path.join(self.stego_img_dir, self.stego_img_filenames[index])
        cover_img = Image.open(cover_img_paths).convert("RGB")
        stego_img = Image.open(stego_img_paths).convert("RGB")
        if self.transforms:
            cover_img = self.transforms(cover_img)
            stego_img = self.transforms(stego_img)
        label1 = torch.tensor(0, dtype=torch.long)
        label2 = torch.tensor(1, dtype=torch.long)
        sample = {"cover": cover_img, "stego": stego_img}
        sample["label"] = [label1, label2]
        return sample
    
class BalancedDataset(Dataset):
    def __init__(self, cover_img_dir, stego_img_dir, transform):
        self.cover_img_dir = cover_img_dir
        self.stego_img_dir = stego_img_dir
        self.transform = transform
        self.cover_filenames = list(sorted(os.listdir(cover_img_dir)))
        self.stego_filenames = list(sorted(os.listdir(stego_img_dir)))
        self.all_filenames = []
        self.all_classes = []
        for filename in self.cover_filenames:
            self.all_filenames.append(('cover', filename))
            self.all_classes.append(0)
        for filename in self.stego_filenames:
            self.all_filenames.append(('stego', filename))
            self.all_classes.append(1)
        self.class_counts = np.bincount(self.all_classes)
        self.class_weights = 1.0 / self.class_counts
        self.weights = [self.class_weights[class_id] for class_id in self.all_classes]
    def __len__(self):
        return len(self.all_filenames)
    def __getitem__(self, index):
        img_type, filename = self.all_filenames[index]
        if img_type == 'cover':
            img_path = os.path.join(self.cover_img_dir, filename)
            label = torch.tensor(0, dtype=torch.long)
        else:
            img_path = os.path.join(self.stego_img_dir, filename)
            label = torch.tensor(1, dtype=torch.long)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": label}

class PairedBalancedDataset(Dataset):
    def __init__(self, cover_img_dir, stego_img_dir, transform):
        self.cover_img_dir = cover_img_dir
        self.stego_img_dir = stego_img_dir
        self.transforms = transform
        self.cover_filenames = list(sorted(os.listdir(cover_img_dir)))
        self.stego_filenames = list(sorted(os.listdir(stego_img_dir)))
        assert len(self.cover_filenames) == len(self.stego_filenames), \
            "Cover and stego directories must contain same number of images"
        self.indices = list(range(len(self.cover_filenames)))
        self.weights = np.ones(len(self.indices))
    def __len__(self):
        return len(self.cover_filenames)
    def __getitem__(self, index):
        cover_img_path = os.path.join(self.cover_img_dir, self.cover_filenames[index])
        stego_img_path = os.path.join(self.stego_img_dir, self.stego_filenames[index])
        cover_img = Image.open(cover_img_path).convert("RGB")
        stego_img = Image.open(stego_img_path).convert("RGB")
        if self.transforms:
            cover_img = self.transforms(cover_img)
            stego_img = self.transforms(stego_img)
        label1 = torch.tensor(0, dtype=torch.long)
        label2 = torch.tensor(1, dtype=torch.long)
        sample = {"cover": cover_img, "stego": stego_img}
        sample["label"] = [label1, label2]
        return sample

transform_train = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(),  # Data augmentation
    T.RandomVerticalFlip(),    # Data augmentation
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1] range
])

transform_val = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1] range
])

def load_train_data(cover_data_dir, stego_data_dir, batchsize=32, balanced=False, paired_balanced=False, transform=None):
    if paired_balanced:
        dataset = PairedBalancedDataset(cover_data_dir, stego_data_dir, transform_train)
        sampler = WeightedRandomSampler(
            weights=dataset.weights,
            num_samples=len(dataset),
            replacement=True
        )
        train_loader = DataLoader(
            dataset,
            batch_size=batchsize,
            sampler=sampler,
            pin_memory=True,
            drop_last=True
        )
        plot_pair_distribution(dataset)
        return train_loader
    elif not balanced:
        train_loader = DataLoader(
            dataset_(cover_data_dir, stego_data_dir, transform_train),
            batch_size=batchsize,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        return train_loader
    else:
        dataset = BalancedDataset(cover_data_dir, stego_data_dir, transform_train)
        sampler = WeightedRandomSampler(
            weights=dataset.weights,
            num_samples=len(dataset),
            replacement=True
        )
        train_loader = DataLoader(
            dataset,
            batch_size=batchsize,
            sampler=sampler,
            pin_memory=True,
            drop_last=True
        )
        plot_balanced_distribution(dataset)
        return train_loader

def load_test_data(cover_data_dir, stego_data_dir, batchsize=32, balanced=False, paired_balanced=False, transform=None):
    if paired_balanced:
        dataset = PairedBalancedDataset(cover_data_dir, stego_data_dir, transform_val)
        test_loader = DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=False,
            pin_memory=False,
            drop_last=False
        )
        return test_loader
    elif not balanced:
        test_loader = DataLoader(
            dataset_(cover_data_dir, stego_data_dir, transform_val),
            batch_size=batchsize,
            shuffle=True,
            pin_memory=False,
            drop_last=False
        )
        return test_loader
    else:
        dataset = BalancedDataset(cover_data_dir, stego_data_dir, transform_val)
        sampler = WeightedRandomSampler(
            weights=dataset.weights,
            num_samples=len(dataset),
            replacement=True
        )
        test_loader = DataLoader(
            dataset,
            batch_size=batchsize,
            sampler=sampler,
            pin_memory=False,
            drop_last=False
        )
        return test_loader

def plot_pair_distribution(dataset, save_path="./plots/paired_sampling_distribution.png"):
    plt.figure(figsize=(10, 6))
    plt.bar(['Image Pairs'], [len(dataset)], color='#3498db', width=0.4)
    plt.title('Paired Balanced Sampling Strategy', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Image Pairs', fontsize=12)
    plt.ylim(0, len(dataset) * 1.2)
    plt.text(0, len(dataset) + len(dataset) * 0.05, 
             f"Total Pairs: {len(dataset)}", 
             ha='center', fontsize=12, fontweight='bold')
    plt.text(0, len(dataset) * 0.9,
             "Each pair contains:\n1 Cover Image (label 0)\n1 Stego Image (label 1)",
             ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0, len(dataset) * 0.5,
             "Paired sampling preserves\ncover-stego relationships\nwhile applying weights",
             ha='center', fontsize=10, bbox=dict(facecolor='lightyellow', alpha=0.8))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_dir = Path(os.path.dirname(save_path))
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(save_path)
    print(f"Paired sampling distribution plot saved to {save_path}")
    plt.close()

def plot_balanced_distribution(dataset, save_path="./plots/balanced_dataset_distribution.png"):
    class_counts = dataset.class_counts
    total_samples = sum(class_counts)
    effective_class_counts = np.ones_like(class_counts) * (total_samples / len(class_counts))
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(['Cover (0)', 'Stego (1)'], class_counts, color=['#3498db', '#e74c3c'])
    plt.title('Original Class Distribution')
    plt.ylabel('Number of Images')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, count in enumerate(class_counts):
        plt.text(i, count + 5, str(count), ha='center', fontweight='bold')
        percentage = (count / total_samples) * 100
        plt.text(i, count / 2, f"{percentage:.1f}%", ha='center', color='white', fontweight='bold')
    plt.subplot(1, 2, 2)
    plt.bar(['Cover (0)', 'Stego (1)'], effective_class_counts, color=['#2ecc71', '#9b59b6'])
    plt.title('Balanced Distribution (After Weighted Sampling)')
    plt.ylabel('Effective Number of Images')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, count in enumerate(effective_class_counts):
        plt.text(i, count + 5, f"{count:.0f}", ha='center', fontweight='bold')
        percentage = 50.0
        plt.text(i, count / 2, f"{percentage:.1f}%", ha='center', color='white', fontweight='bold')
    plt.tight_layout()
    plot_dir = Path(os.path.dirname(save_path))
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(save_path)
    print(f"Balanced distribution plot saved to {save_path}")
    plt.close()



