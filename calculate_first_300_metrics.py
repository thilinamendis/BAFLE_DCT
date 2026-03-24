#!/usr/bin/env python3
"""
A simple script to calculate average metrics (PSNR, SSIM, MSE) for the first 300 images
in each ablation test file.
"""

import sys
import os
import re
import numpy as np
from pathlib import Path

def process_file(file_path, limit=300, sample_method="random"):
    """
    Process an ablation test file and calculate metrics for N images.
    
    Args:
        file_path: Path to the ablation test file
        limit: Maximum number of images to process
        sample_method: Method to select images ("first" or "random")
        
    Returns:
        dict: Metrics including PSNR, SSIM, MSE, and count of processed images
    """
    filename = os.path.basename(file_path)
    print(f"Processing {filename}...")
    
    # Read the file content
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract test name from the first line
    test_name = "unknown"
    if lines and "Ablation Study Results:" in lines[0]:
        test_name = lines[0].split("Ablation Study Results:")[1].strip()
    
    # Find all image data lines (those containing JPEG)
    all_data_lines = [line for line in lines if 'JPEG' in line]
    
    # Select images based on specified method
    if sample_method == "first":
        # Take the first 'limit' images
        data_lines = all_data_lines[:limit]
    else:  # random
        # Randomly select 'limit' images if there are more than 'limit' images
        if len(all_data_lines) > limit:
            import random
            random.seed(42)  # Set seed for reproducibility
            data_lines = random.sample(all_data_lines, limit)
        else:
            data_lines = all_data_lines
    
    # Extract metrics
    psnr_values = []
    ssim_values = []
    mse_values = []
    
    # Regular expression to extract values
    pattern = r'ILSVRC.*\.JPEG\s+(\S+)\s+(\S+)\s+(\S+)'
    
    image_count = 0
    for line in data_lines:
        match = re.search(pattern, line)
        if match:
            psnr_str = match.group(1)
            ssim = float(match.group(2))
            mse = float(match.group(3))
            
            # Handle 'inf' PSNR values
            if psnr_str.lower() == 'inf':
                # Skip inf values when calculating average
                continue
            else:
                psnr = float(psnr_str)
                psnr_values.append(psnr)
                
            ssim_values.append(ssim)
            mse_values.append(mse)
            
            image_count += 1
    
    # Calculate averages
    avg_psnr = np.mean(psnr_values) if psnr_values else float('inf')
    avg_ssim = np.mean(ssim_values)
    avg_mse = np.mean(mse_values)
    
    return {
        'file_name': filename,
        'test_name': test_name,
        'image_count': image_count,
        'psnr_count': len(psnr_values),  # Number of non-inf PSNR values
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'avg_mse': avg_mse
    }

def main():
    # Set the limit of images to process
    limit = 300
    sample_method = "random"  # Options: "first" or "random"
    
    # Process command line arguments
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            print(f"Invalid limit: {sys.argv[1]}. Using default: 300")
            
    if len(sys.argv) > 2:
        if sys.argv[2].lower() in ["first", "random"]:
            sample_method = sys.argv[2].lower()
        else:
            print(f"Invalid sample method: {sys.argv[2]}. Using default: random")
    
    print(f"Using sample method: {sample_method}, with limit: {limit} images")
    
    # Define the directory containing ablation test files
    ablation_dir = Path("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/ablation_test")
    
    # Get all ablation test files
    test_files = list(ablation_dir.glob("*bit_ablation_test_*.txt"))
    
    if not test_files:
        print(f"No ablation test files found in {ablation_dir}")
        return
    
    # Group files by bit rate and test type for better presentation
    results_by_bit = {'100bit': [], '500bit': []}
    
    # Process each file
    for file_path in sorted(test_files):
        file_name = file_path.name
        
        # Determine the bit rate
        if '100bit' in file_name:
            bit_rate = '100bit'
        elif '500bit' in file_name:
            bit_rate = '500bit'
        else:
            continue  # Skip files with unknown bit rate
        
        # Process the file
        result = process_file(file_path, limit, sample_method)
        results_by_bit[bit_rate].append(result)
    
    # Display results
    print("\n" + "="*80)
    sampling_description = f"First {limit}" if sample_method == "first" else f"Random {limit}"
    print(f"ABLATION STUDY METRICS ({sampling_description} images)")
    print("="*80)
    
    for bit_rate in ['100bit', '500bit']:
        print(f"\n{bit_rate.upper()} RESULTS:\n")
        print(f"{'Test Type':<25} {'Images':<10} {'PSNR (dB)':<15} {'SSIM':<10} {'MSE':<10}")
        print("-"*70)
        
        # Sort by test type for consistent display
        sorted_results = sorted(results_by_bit[bit_rate], key=lambda x: x['test_name'])
        
        for result in sorted_results:
            psnr_display = f"{result['avg_psnr']:.2f}" if result['psnr_count'] > 0 else "N/A"
            img_count = f"{result['image_count']} ({result['psnr_count']})"
            
            print(f"{result['test_name']:<25} {img_count:<10} {psnr_display:<15} {result['avg_ssim']:.6f} {result['avg_mse']:.6f}")

if __name__ == "__main__":
    main()
