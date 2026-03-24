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
        # print(cover_img_paths)
        stego_img_paths = os.path.join(self.stego_img_dir, self.stego_img_filenames[index])
        # print(stego_img_paths)
 
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
        """
        Dataset that enables weighted sampling for balanced training
        
        Args:
            cover_img_dir (str): Directory containing cover images
            stego_img_dir (str): Directory containing stego images
            transform: Image transforms to apply
        """
        self.cover_img_dir = cover_img_dir
        self.stego_img_dir = stego_img_dir
        self.transform = transform
        
        # Load all filenames
        self.cover_filenames = list(sorted(os.listdir(cover_img_dir)))
        self.stego_filenames = list(sorted(os.listdir(stego_img_dir)))
        
        # Combine both classes in one dataset for weighted sampling
        self.all_filenames = []
        self.all_classes = []
        
        # Add cover images (class 0)
        for filename in self.cover_filenames:
            self.all_filenames.append(('cover', filename))
            self.all_classes.append(0)
            
        # Add stego images (class 1)
        for filename in self.stego_filenames:
            self.all_filenames.append(('stego', filename))
            self.all_classes.append(1)
            
        # Calculate class weights
        self.class_counts = np.bincount(self.all_classes)
        self.class_weights = 1.0 / self.class_counts
        self.weights = [self.class_weights[class_id] for class_id in self.all_classes]
            
    def __len__(self):
        return len(self.all_filenames)
    
    def __getitem__(self, index):
        img_type, filename = self.all_filenames[index]
        
        # Load either cover or stego based on the type
        if img_type == 'cover':
            img_path = os.path.join(self.cover_img_dir, filename)
            label = torch.tensor(0, dtype=torch.long)
        else:
            img_path = os.path.join(self.stego_img_dir, filename)
            label = torch.tensor(1, dtype=torch.long)
            
        # Load and transform image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
            
        return {"image": img, "label": label}

# New class that preserves pairs while performing balanced sampling
class PairedBalancedDataset(Dataset):
    def __init__(self, cover_img_dir, stego_img_dir, transform):
        """
        Dataset that maintains cover-stego pairs while enabling balanced sampling
        
        Args:
            cover_img_dir (str): Directory containing cover images
            stego_img_dir (str): Directory containing stego images
            transform: Image transforms to apply
        """
        self.cover_img_dir = cover_img_dir
        self.stego_img_dir = stego_img_dir
        self.transforms = transform
        
        # Get filenames (assuming cover and stego have matching filenames)
        self.cover_filenames = list(sorted(os.listdir(cover_img_dir)))
        self.stego_filenames = list(sorted(os.listdir(stego_img_dir)))
        
        # Verify we have matching pairs
        assert len(self.cover_filenames) == len(self.stego_filenames), \
            "Cover and stego directories must contain same number of images"
        
        # Set up indices and weights
        # For steganalysis, often we want to ensure variety in image content
        # rather than balancing classes (which are already balanced)
        self.indices = list(range(len(self.cover_filenames)))
        
        # Instead of class weights, we can use image complexity or other factors
        # as weights to bias sampling toward more informative examples
        # For now, using uniform weights (equal probability for all pairs)
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
    # T.RandomHorizontalFlip(),
    # T.RandomRotation(degrees=90),
    T.ToTensor()
])

transform_val = T.Compose([
    T.ToTensor(),
])

transform_train = transform_val

def load_train_data(cover_data_dir, stego_data_dir, batchsize=32, balanced=False, paired_balanced=False):
    """
    Load training data with various balancing options
    
    Args:
        cover_data_dir (str): Directory containing cover images
        stego_data_dir (str): Directory containing stego images
        batchsize (int): Batch size for training
        balanced (bool): Whether to use weighted sampling for balanced classes (breaks pairs)
        paired_balanced (bool): Whether to use weighted sampling while preserving pairs
    
    Returns:
        DataLoader: Training data loader
    """
    if paired_balanced:
        # Use the new paired balanced approach
        dataset = PairedBalancedDataset(cover_data_dir, stego_data_dir, transform_train)
        
        # Create weighted sampler that preserves pairs
        sampler = WeightedRandomSampler(
            weights=dataset.weights,
            num_samples=len(dataset),
            replacement=True  # Can be set to False if dataset is large enough
        )
        
        train_loader = DataLoader(
            dataset,
            batch_size=batchsize,
            sampler=sampler,
            pin_memory=True,
            drop_last=True
        )
        
        # Plot distribution of sampled images
        plot_pair_distribution(dataset)
        
        return train_loader
    elif not balanced:
        # Original implementation - paired cover/stego samples
        train_loader = DataLoader(
            dataset_(cover_data_dir, stego_data_dir, transform_train),
            batch_size=batchsize,
            shuffle=True,
            pin_memory=True,
            # num_workers=8,
            drop_last=True
        )
        return train_loader
    else:
        # Balanced implementation using weighted sampling (breaks pairs)
        dataset = BalancedDataset(cover_data_dir, stego_data_dir, transform_train)
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=dataset.weights,
            num_samples=len(dataset),
            replacement=True
        )
        
        train_loader = DataLoader(
            dataset,
            batch_size=batchsize,
            sampler=sampler,  # Using sampler instead of shuffle
            pin_memory=True,
            # num_workers=8,
            drop_last=True
        )
        
        # Plot class distribution after balancing
        plot_balanced_distribution(dataset)
        
        return train_loader

def load_test_data(cover_data_dir, stego_data_dir, batchsize=32, balanced=False, paired_balanced=False):
    """
    Load test data with various balancing options
    
    Args:
        cover_data_dir (str): Directory containing cover images
        stego_data_dir (str): Directory containing stego images
        batchsize (int): Batch size for testing
        balanced (bool): Whether to use weighted sampling for balanced classes
        paired_balanced (bool): Whether to use weighted sampling while preserving pairs
    
    Returns:
        DataLoader: Test data loader
    """
    if paired_balanced:
        # Use the new paired balanced approach
        dataset = PairedBalancedDataset(cover_data_dir, stego_data_dir, transform_val)
        
        # For testing, we typically want deterministic evaluation, so no sampling
        test_loader = DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=False,  # No shuffling for more deterministic evaluation
            pin_memory=False,
            drop_last=False
        )
        
        return test_loader
    elif not balanced:
        # Original implementation
        test_loader = DataLoader(
            dataset_(cover_data_dir, stego_data_dir, transform_val),
            batch_size=batchsize,
            shuffle=True,
            pin_memory=False,
            # num_workers=8,
            drop_last=False
        )
        return test_loader
    else:
        # Balanced implementation using weighted sampling
        dataset = BalancedDataset(cover_data_dir, stego_data_dir, transform_val)
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=dataset.weights,
            num_samples=len(dataset),
            replacement=True
        )
        
        test_loader = DataLoader(
            dataset,
            batch_size=batchsize,
            sampler=sampler,  # Using sampler instead of shuffle
            pin_memory=False,
            # num_workers=8,
            drop_last=False
        )
        
        return test_loader

def plot_pair_distribution(dataset, save_path="./plots/paired_sampling_distribution.png"):
    """
    Plot the distribution of image pairs that will be sampled
    
    Args:
        dataset: The PairedBalancedDataset instance
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create bar chart showing paired sampling strategy
    plt.bar(['Image Pairs'], [len(dataset)], color='#3498db', width=0.4)
    
    plt.title('Paired Balanced Sampling Strategy', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Image Pairs', fontsize=12)
    plt.ylim(0, len(dataset) * 1.2)  # Add some headroom
    
    # Add annotations
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
    
    # Create directory if it doesn't exist
    plot_dir = Path(os.path.dirname(save_path))
    plot_dir.mkdir(exist_ok=True)
    
    plt.savefig(save_path)
    print(f"Paired sampling distribution plot saved to {save_path}")
    plt.close()
    
def plot_balanced_distribution(dataset, save_path="./plots/balanced_dataset_distribution.png"):
    """
    Plot the distribution of classes after applying weighted sampling
    
    Args:
        dataset: The BalancedDataset instance
        save_path: Path to save the plot
    """
    # Calculate effective class weights after sampling
    class_counts = dataset.class_counts
    total_samples = sum(class_counts)
    
    # Create equal distribution based on weights
    effective_class_counts = np.ones_like(class_counts) * (total_samples / len(class_counts))
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot original distribution
    plt.subplot(1, 2, 1)
    plt.bar(['Cover (0)', 'Stego (1)'], class_counts, color=['#3498db', '#e74c3c'])
    plt.title('Original Class Distribution')
    plt.ylabel('Number of Images')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels
    for i, count in enumerate(class_counts):
        plt.text(i, count + 5, str(count), ha='center', fontweight='bold')
        percentage = (count / total_samples) * 100
        plt.text(i, count / 2, f"{percentage:.1f}%", ha='center', color='white', fontweight='bold')
    
    # Plot effective distribution after weighted sampling
    plt.subplot(1, 2, 2)
    plt.bar(['Cover (0)', 'Stego (1)'], effective_class_counts, color=['#2ecc71', '#9b59b6'])
    plt.title('Balanced Distribution (After Weighted Sampling)')
    plt.ylabel('Effective Number of Images')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels
    for i, count in enumerate(effective_class_counts):
        plt.text(i, count + 5, f"{count:.0f}", ha='center', fontweight='bold')
        percentage = 50.0  # Will always be 50% for each class after balancing
        plt.text(i, count / 2, f"{percentage:.1f}%", ha='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    plot_dir = Path(os.path.dirname(save_path))
    plot_dir.mkdir(exist_ok=True)
    
    plt.savefig(save_path)
    print(f"Balanced distribution plot saved to {save_path}")
    plt.close()

# Original commented code below
# transform_train = A.Compose(
#     [
#         A.RandomCrop(128, 128),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         ToTensorV2(),
#     ]
# )

# transform_val = A.Compose([
#     A.CenterCrop(256, 256),
#     ToTensorV2(),
# ])


# class dataset_(Dataset):
#     def __init__(self, img_dir, sigma, transform):
#         self.img_dir = img_dir
#         self.img_filenames = list(sorted(os.listdir(img_dir)))
#         self.sigma = sigma
#         self.transform = transform

#     def __len__(self):
#         return len(self.img_filenames)

#     def __getitem__(self, idx):
#         img_filename = self.img_filenames[idx]
#         img = cv2.imread(os.path.join(self.img_dir, img_filename))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.float32(img/255)
        
#         if self.transform:
#             img = self.transform(image=img)["image"]
            
#         noised_img = img + torch.randn(img.shape).mul_(self.sigma/255)

#         return img, noised_img


# def load_dataset(train_data_dir, test_data_dir, batch_size, sigma=None):

#     train_loader = DataLoader(
#         dataset_(train_data_dir, sigma, transform_train),
#         batch_size=batch_size,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=8,
#         drop_last=True
#     )

#     test_loader = DataLoader(
#         dataset_(test_data_dir, sigma, transform_val),
#         batch_size=2,
#         shuffle=False,
#         pin_memory=True,
#         num_workers=1,
#         drop_last=True
#     )

#     return train_loader, test_loader



