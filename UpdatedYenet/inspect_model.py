#!/usr/bin/env python
"""
Script to inspect YeNet outputs and inspect model behavior on individual images
This script will:
1. Load trained model if available
2. Visualize filters and model outputs
3. Print detailed classification results for individual samples
4. Compare cover and stego image pairs to see how the model behaves
"""

import os
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from model.model_yenet import YeNet
from opts.options import arguments

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("inspect_model.log")
    ]
)
logger = logging.getLogger(__name__)

def load_image_pair(cover_path, stego_path):
    """Load a pair of cover and stego images"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    cover_img = Image.open(cover_path).convert('RGB')
    stego_img = Image.open(stego_path).convert('RGB')
    
    cover_tensor = transform(cover_img).unsqueeze(0)
    stego_tensor = transform(stego_img).unsqueeze(0)
    
    return cover_tensor, stego_tensor, cover_img, stego_img

def visualize_srm_filters(model):
    """Visualize the SRM filters in the YeNet model"""
    # Extract SRM filters from the model
    try:
        srm_filters = model.srm_conv.weight.detach().cpu()
        
        # Plot the filters
        plt.figure(figsize=(15, 5))
        for i in range(min(srm_filters.size(0), 30)):  # Show up to 30 filters
            plt.subplot(3, 10, i+1)
            plt.imshow(srm_filters[i, 0].numpy(), cmap='gray')
            plt.axis('off')
            
        plt.suptitle("SRM Filters")
        plt.tight_layout()
        plt.savefig('srm_filters.png')
        logger.info("Saved SRM filters visualization to srm_filters.png")
    except Exception as e:
        logger.error(f"Error visualizing SRM filters: {e}")

def visualize_feature_maps(model, image_tensor, layer_name="tlu", save_path="feature_maps.png"):
    """Visualize feature maps from a specific layer"""
    # Dictionary to store outputs
    activations = {}
    
    # Define hook function
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if layer_name in name:
            module.register_forward_hook(get_activation(name))
    
    # Forward pass
    with torch.no_grad():
        model(image_tensor)
    
    # If no activations were captured
    if not activations:
        logger.warning(f"No layers with name '{layer_name}' found")
        return
    
    # Get the first layer's activations
    first_layer_name = list(activations.keys())[0]
    feature_maps = activations[first_layer_name][0]  # First batch item
    
    # Determine grid dimensions
    n_features = min(feature_maps.size(0), 64)  # Display up to 64 feature maps
    grid_size = int(np.ceil(np.sqrt(n_features)))
    
    # Create plot
    plt.figure(figsize=(15, 15))
    for i in range(n_features):
        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(feature_maps[i].numpy(), cmap='viridis')
        plt.axis('off')
    
    plt.suptitle(f"Feature Maps - {first_layer_name}")
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Saved feature maps visualization to {save_path}")

def inspect_model_decision(model, cover_tensor, stego_tensor):
    """Inspect model's decision process on a cover/stego pair"""
    model.eval()
    
    with torch.no_grad():
        # Get predictions
        cover_output = model(cover_tensor)
        stego_output = model(stego_tensor)
        
        # Convert to probabilities
        cover_probs = torch.nn.functional.softmax(cover_output, dim=1)
        stego_probs = torch.nn.functional.softmax(stego_output, dim=1)
        
        # Get predicted classes
        _, cover_pred = torch.max(cover_output, 1)
        _, stego_pred = torch.max(stego_output, 1)
        
        # Print detailed results
        logger.info("Cover Image Classification:")
        logger.info(f"  Raw Output: {cover_output.squeeze().tolist()}")
        logger.info(f"  Probabilities: Cover={cover_probs[0,0].item():.4f}, Stego={cover_probs[0,1].item():.4f}")
        logger.info(f"  Prediction: {cover_pred.item()} ({'Cover' if cover_pred.item() == 0 else 'Stego'})")
        
        logger.info("Stego Image Classification:")
        logger.info(f"  Raw Output: {stego_output.squeeze().tolist()}")
        logger.info(f"  Probabilities: Cover={stego_probs[0,0].item():.4f}, Stego={stego_probs[0,1].item():.4f}")
        logger.info(f"  Prediction: {stego_pred.item()} ({'Cover' if stego_pred.item() == 0 else 'Stego'})")
        
        # Calculate difference between predictions
        diff = (stego_output - cover_output).squeeze().tolist()
        logger.info(f"Difference (Stego - Cover): {diff}")
        
        # Visualize the difference
        plt.figure(figsize=(10, 5))
        labels = ['Cover Prob', 'Stego Prob']
        cover_values = cover_probs.squeeze().tolist()
        stego_values = stego_probs.squeeze().tolist()
        
        x = range(len(labels))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], cover_values, width, label='Cover Image')
        plt.bar([i + width/2 for i in x], stego_values, width, label='Stego Image')
        
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Model Classification Probabilities')
        plt.xticks(x, labels)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('classification_comparison.png')
        logger.info("Saved classification comparison to classification_comparison.png")
        
        return cover_pred.item(), stego_pred.item()

def find_challenging_samples(model, data_loader, device, num_samples=5):
    """Find samples that are particularly challenging for the model"""
    model.eval()
    
    correct_cover_imgs = []
    incorrect_cover_imgs = []
    correct_stego_imgs = []
    incorrect_stego_imgs = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Separate correct and incorrect predictions
            for i in range(images.size(0)):
                img = images[i].cpu()
                pred = preds[i].item()
                label = labels[i].item()
                
                if label == 0:  # Cover
                    if pred == label:
                        if len(correct_cover_imgs) < num_samples:
                            correct_cover_imgs.append((img, outputs[i].cpu()))
                    else:
                        if len(incorrect_cover_imgs) < num_samples:
                            incorrect_cover_imgs.append((img, outputs[i].cpu()))
                else:  # Stego
                    if pred == label:
                        if len(correct_stego_imgs) < num_samples:
                            correct_stego_imgs.append((img, outputs[i].cpu()))
                    else:
                        if len(incorrect_stego_imgs) < num_samples:
                            incorrect_stego_imgs.append((img, outputs[i].cpu()))
            
            # Check if we've found enough samples
            if (len(correct_cover_imgs) >= num_samples and 
                len(incorrect_cover_imgs) >= num_samples and
                len(correct_stego_imgs) >= num_samples and
                len(incorrect_stego_imgs) >= num_samples):
                break
    
    # Visualize challenging samples
    def plot_samples(samples, title, filename):
        if not samples:
            return
            
        plt.figure(figsize=(15, 5 * min(len(samples), num_samples)))
        
        for i, (img, output) in enumerate(samples[:num_samples]):
            probs = torch.nn.functional.softmax(output, dim=0)
            
            plt.subplot(min(len(samples), num_samples), 2, i*2+1)
            plt.imshow(img.permute(1, 2, 0))
            plt.axis('off')
            plt.title(f"Image {i+1}")
            
            plt.subplot(min(len(samples), num_samples), 2, i*2+2)
            plt.bar(['Cover', 'Stego'], probs.tolist())
            plt.ylim(0, 1)
            plt.title(f"Probabilities")
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(filename)
        logger.info(f"Saved {title} samples to {filename}")
    
    plot_samples(correct_cover_imgs, "Correctly Classified Cover Images", "correct_cover.png")
    plot_samples(incorrect_cover_imgs, "Incorrectly Classified Cover Images", "incorrect_cover.png")
    plot_samples(correct_stego_imgs, "Correctly Classified Stego Images", "correct_stego.png")
    plot_samples(incorrect_stego_imgs, "Incorrectly Classified Stego Images", "incorrect_stego.png")

def main():
    logger.info("Starting YeNet model inspection")
    
    # Load arguments
    opt = arguments()
    
    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = YeNet(opt.in_channels, opt.num_cls).to(device)
    
    # Load pretrained weights if available
    checkpoint_path = os.path.join(opt.ckpt_dir, 'model_best.pth')
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} with accuracy {checkpoint['best_acc']}%")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}, using untrained model")
    
    model.eval()
    
    # Visualize SRM filters
    logger.info("Visualizing SRM filters...")
    visualize_srm_filters(model)
    
    # Choose a few sample images
    cover_dir = opt.test_cover_dir
    stego_dir = opt.test_stego_dir
    
    cover_files = sorted([f for f in os.listdir(cover_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if cover_files:
        # Take first 5 images for inspection
        for i, filename in enumerate(cover_files[:5]):
            cover_path = os.path.join(cover_dir, filename)
            stego_path = os.path.join(stego_dir, filename)
            
            if not os.path.exists(stego_path):
                logger.warning(f"Stego counterpart for {filename} not found, skipping")
                continue
                
            logger.info(f"Inspecting image pair: {filename}")
            
            # Load image pair
            cover_tensor, stego_tensor, cover_img, stego_img = load_image_pair(cover_path, stego_path)
            cover_tensor = cover_tensor.to(device)
            stego_tensor = stego_tensor.to(device)
            
            # Visualize feature maps
            logger.info("Visualizing feature maps...")
            visualize_feature_maps(model, cover_tensor, "tlu", f"cover_feature_maps_{i}.png")
            visualize_feature_maps(model, stego_tensor, "tlu", f"stego_feature_maps_{i}.png")
            
            # Inspect model decision
            logger.info("Inspecting model decision...")
            cover_pred, stego_pred = inspect_model_decision(model, cover_tensor, stego_tensor)
            
            # Check if correct classification
            cover_correct = cover_pred == 0
            stego_correct = stego_pred == 1
            
            logger.info(f"Cover classification: {'CORRECT' if cover_correct else 'INCORRECT'}")
            logger.info(f"Stego classification: {'CORRECT' if stego_correct else 'INCORRECT'}")
    else:
        logger.warning(f"No images found in {cover_dir}")
    
    logger.info("Inspection complete!")

if __name__ == "__main__":
    main()
