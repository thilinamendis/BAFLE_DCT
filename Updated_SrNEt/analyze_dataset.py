import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from load_data import dataset_
import torchvision.transforms as T

def analyze_dataset_distribution(cover_dir, stego_dir):
    """
    Analyze the distribution of images in cover and stego directories
    
    Args:
        cover_dir: Directory containing cover images
        stego_dir: Directory containing stego images
    
    Returns:
        tuple: (cover_count, stego_count, cover_files, stego_files)
    """
    # Count files
    cover_files = list(sorted(os.listdir(cover_dir)))
    stego_files = list(sorted(os.listdir(stego_dir)))
    
    cover_count = len(cover_files)
    stego_count = len(stego_files)
    
    print(f"Cover images: {cover_count}")
    print(f"Stego images: {stego_count}")
    
    return cover_count, stego_count, cover_files, stego_files

def plot_dataset_distribution(cover_dir, stego_dir, save_path=None):
    """
    Plot the distribution of cover vs stego images
    
    Args:
        cover_dir: Directory containing cover images
        stego_dir: Directory containing stego images
        save_path: Optional path to save the plot
    """
    cover_count, stego_count, _, _ = analyze_dataset_distribution(cover_dir, stego_dir)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.bar(['Cover (0)', 'Stego (1)'], [cover_count, stego_count], color=['#3498db', '#e74c3c'])
    plt.title('Dataset Class Distribution')
    plt.ylabel('Number of Images')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add actual count labels on bars
    for i, count in enumerate([cover_count, stego_count]):
        plt.text(i, count + 10, str(count), ha='center', fontweight='bold')
    
    # Add percentage labels
    total = cover_count + stego_count
    for i, count in enumerate([cover_count, stego_count]):
        percentage = (count / total) * 100
        plt.text(i, count / 2, f"{percentage:.1f}%", ha='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plot_dir = Path(os.path.dirname(save_path))
        plot_dir.mkdir(exist_ok=True)
        plt.savefig(save_path)
        print(f"Distribution plot saved to {save_path}")
    
    plt.close()

if __name__ == "__main__":
    # Example usage
    cover_dir = "/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/100bit/100bitdataset/train/originals"
    stego_dir = "/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/100bit/100bitdataset/train/stegos"
    
    plot_dataset_distribution(
        cover_dir, 
        stego_dir, 
        save_path="./plots/original_dataset_distribution.png"
    )
