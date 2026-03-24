"""This module provides data loading functionality for the J-UNIWARD steganalysis model."""

import os
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import defaultdict

class SteganoDataset(Dataset):
    """Dataset for steganalysis containing cover and stego images."""
    
    def __init__(self, cover_path, stego_path, transform=None, balanced=False, paired_balanced=False):
        """Initialize the dataset.
        
        Args:
            cover_path: Path to cover images
            stego_path: Path to stego images
            transform: Transformations to apply to images
            balanced: Whether to use balanced sampling
            paired_balanced: Whether to use paired balanced sampling
        """
        self.cover_path = Path(cover_path)
        self.stego_path = Path(stego_path)
        self.transform = transform
        self.balanced = balanced
        self.paired_balanced = paired_balanced
        
        # Helper function to get all image files from a directory and its subdirectories
        def get_image_files(directory):
            image_files = []
            if not os.path.exists(directory):
                print(f"Warning: Directory {directory} does not exist")
                return []
            
            # Walk through directory and all subdirectories
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(('.jpg', '.png', '.bmp', '.jpeg', '.tif', '.tiff')):
                        # Get path relative to the base directory
                        rel_dir = os.path.relpath(root, directory)
                        if rel_dir == '.':
                            image_files.append(file)
                        else:
                            image_files.append(os.path.join(rel_dir, file))
            return sorted(image_files)
        
        # Get all cover and stego images
        self.cover_files = get_image_files(self.cover_path)
        self.stego_files = get_image_files(self.stego_path)
        
        print(f"Found {len(self.cover_files)} cover files and {len(self.stego_files)} stego files")
        
        # Check if cover and stego files match for paired balanced
        if self.paired_balanced:
            # Extract the base part of the filename (without _original or _stego suffix)
            def get_base_id(filename):
                basename = os.path.basename(filename)
                # Remove file extension
                basename_no_ext = os.path.splitext(basename)[0]
                # Remove _original or _stego suffix
                if basename_no_ext.endswith('_original'):
                    return basename_no_ext[:-9]  # Remove '_original'
                elif basename_no_ext.endswith('_stego'):
                    return basename_no_ext[:-6]  # Remove '_stego'
                return basename_no_ext
            
            # Create dictionaries to map base IDs to full paths
            cover_id_dict = {}
            stego_id_dict = {}
            
            for f in self.cover_files:
                base_id = get_base_id(f)
                cover_id_dict[base_id] = f
                
            for f in self.stego_files:
                base_id = get_base_id(f)
                stego_id_dict[base_id] = f
            
            # Find common base IDs
            common_base_ids = set(cover_id_dict.keys()).intersection(set(stego_id_dict.keys()))
            print(f"Found {len(common_base_ids)} matching files for paired balanced dataset")
            
            if len(common_base_ids) == 0:
                print("WARNING: No matching files found! This will cause problems.")
                print(f"Cover files sample: {[os.path.basename(self.cover_files[i]) for i in range(min(5, len(self.cover_files)))]}")
                print(f"Stego files sample: {[os.path.basename(self.stego_files[i]) for i in range(min(5, len(self.stego_files)))]}")
                print(f"Cover base IDs sample: {list(cover_id_dict.keys())[:5]}")
                print(f"Stego base IDs sample: {list(stego_id_dict.keys())[:5]}")
            
            # Update file lists to only include files with matching pairs
            self.cover_files = [cover_id_dict[base_id] for base_id in common_base_ids]
            self.stego_files = [stego_id_dict[base_id] for base_id in common_base_ids]
            
            # Sort to ensure same order
            self.cover_files = sorted(self.cover_files, key=lambda x: get_base_id(x))
            self.stego_files = sorted(self.stego_files, key=lambda x: get_base_id(x))
        
        # Prepare image paths and labels
        if paired_balanced:
            # For paired balanced, we've already matched cover-stego files by basename
            # We don't need to prepare self.images or self.labels in this mode
            # The __getitem__ method will handle pairing directly from self.cover_files and self.stego_files
            print(f"Paired balanced mode: {len(self.cover_files)} matched pairs")
            # No further preparation needed
            
        elif balanced:
            # For regular balanced (non-paired)
            self.images = []
            self.labels = []
            
            # Add cover images
            for cover_file in self.cover_files:
                self.images.append(str(self.cover_path / cover_file))
                self.labels.append(0)  # Cover label is 0
            
            # Add stego images
            for stego_file in self.stego_files:
                self.images.append(str(self.stego_path / stego_file))
                self.labels.append(1)  # Stego label is 1
                
            print(f"Balanced mode: {len(self.images)} total images")
        else:
            # Original approach without balancing
            self.cover_images = [str(self.cover_path / f) for f in self.cover_files]
            self.stego_images = [str(self.stego_path / f) for f in self.stego_files]
            print(f"Standard mode: {len(self.cover_images)} cover, {len(self.stego_images)} stego images")
    
    def __len__(self):
        """Return the length of the dataset."""
        if self.paired_balanced:
            # In paired balanced mode, each item represents a cover-stego pair
            return len(self.cover_files)
        elif self.balanced:
            # In balanced mode, we return the length of the combined array
            return len(self.images)
        else:
            # Standard mode - we use max of cover and stego counts
            # Ensure we have cover and stego images
            if not hasattr(self, 'cover_images') or not self.cover_images:
                self.cover_images = [str(self.cover_path / f) for f in self.cover_files]
            if not hasattr(self, 'stego_images') or not self.stego_images:
                self.stego_images = [str(self.stego_path / f) for f in self.stego_files]
            
            # Handle empty lists
            if len(self.cover_images) == 0 or len(self.stego_images) == 0:
                print("Warning: Empty dataset detected!")
                return 0
                
            return max(len(self.cover_images), len(self.stego_images))
    
    def __getitem__(self, idx):
        """Get item from dataset.
        
        Args:
            idx: Index of item
            
        Returns:
            Dictionary containing image and label or cover/stego images and labels
        """
        if self.balanced:
            # Return single image and label for balanced dataset
            image_path = self.images[idx]
            label = self.labels[idx]
            
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            return {"image": image, "label": label}
        
        elif self.paired_balanced:
            # In paired balanced mode, we ensure cover-stego pairs with same base name
            # Get the index modulo the number of pairs
            pair_idx = idx % len(self.cover_files)
            
            # Get matching cover and stego file paths
            cover_path = str(self.cover_path / self.cover_files[pair_idx])
            stego_path = str(self.stego_path / self.stego_files[pair_idx])
            
            try:
                # Load images
                cover_img = Image.open(cover_path).convert('RGB')
                stego_img = Image.open(stego_path).convert('RGB')
                
                if self.transform:
                    cover_img = self.transform(cover_img)
                    stego_img = self.transform(stego_img)
                
                return {
                    "cover": cover_img, 
                    "stego": stego_img,
                    "label": (torch.tensor(0), torch.tensor(1))
                }
            except Exception as e:
                print(f"Error loading paired images: {e}")
                print(f"Cover path: {cover_path}")
                print(f"Stego path: {stego_path}")
                
                # Fallback to first pair if there's an error
                cover_path = str(self.cover_path / self.cover_files[0])
                stego_path = str(self.stego_path / self.stego_files[0])
                
                cover_img = Image.open(cover_path).convert('RGB')
                stego_img = Image.open(stego_path).convert('RGB')
                
                if self.transform:
                    cover_img = self.transform(cover_img)
                    stego_img = self.transform(stego_img)
                
                return {
                    "cover": cover_img, 
                    "stego": stego_img,
                    "label": (torch.tensor(0), torch.tensor(1))
                }
        
        else:
            # Original approach
            cover_idx = idx % len(self.cover_images)
            stego_idx = idx % len(self.stego_images)
            
            # Safely handle file paths
            try:
                cover_img_path = self.cover_images[cover_idx]
                stego_img_path = self.stego_images[stego_idx]
                
                cover_img = Image.open(cover_img_path).convert('RGB')
                stego_img = Image.open(stego_img_path).convert('RGB')
                
                if self.transform:
                    cover_img = self.transform(cover_img)
                    stego_img = self.transform(stego_img)
                
                return {
                    "cover": cover_img, 
                    "stego": stego_img,
                    "label": (torch.tensor(0), torch.tensor(1))
                }
            except Exception as e:
                print(f"Error loading images: {e}")
                print(f"Cover path: {self.cover_images[cover_idx]}")
                print(f"Stego path: {self.stego_images[stego_idx]}")
                # Fallback to any valid images
                valid_cover_idx = 0
                valid_stego_idx = 0
                cover_img = Image.open(self.cover_images[valid_cover_idx]).convert('RGB')
                stego_img = Image.open(self.stego_images[valid_stego_idx]).convert('RGB')
                
                if self.transform:
                    cover_img = self.transform(cover_img)
                    stego_img = self.transform(stego_img)
                
                return {
                    "cover": cover_img, 
                    "stego": stego_img,
                    "label": (torch.tensor(0), torch.tensor(1))
                }


def load_train_data(cover_path, stego_path, batch_size, transform=None, balanced=False, paired_balanced=False):
    """Load training data.
    
    Args:
        cover_path: Path to cover images
        stego_path: Path to stego images
        batch_size: Batch size
        transform: Transformations to apply to images
        balanced: Whether to use balanced sampling
        paired_balanced: Whether to use paired balanced sampling
        
    Returns:
        DataLoader for training data
    """
    dataset = SteganoDataset(
        cover_path=cover_path,
        stego_path=stego_path,
        transform=transform,
        balanced=balanced,
        paired_balanced=paired_balanced
    )
    
    shuffle = True if balanced else False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )


def load_test_data(cover_path, stego_path, batch_size, transform=None, balanced=False, paired_balanced=False):
    """Load test data.
    
    Args:
        cover_path: Path to cover images
        stego_path: Path to stego images
        batch_size: Batch size
        transform: Transformations to apply to images
        balanced: Whether to use balanced sampling
        paired_balanced: Whether to use paired balanced sampling
        
    Returns:
        DataLoader for test data
    """
    dataset = SteganoDataset(
        cover_path=cover_path,
        stego_path=stego_path,
        transform=transform,
        balanced=balanced,
        paired_balanced=paired_balanced
    )
    
    shuffle = False  # Don't shuffle test data
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
