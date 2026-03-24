#!/usr/bin/env python
"""
Script to monitor YeNet's gradient flow and analyze model stability
This script will:
1. Train the model for a few iterations
2. Monitor and visualize gradient flow through different layers
3. Check weight distributions
4. Test different learning rates
"""

import os
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from model.model_yenet import YeNet
from opts.options import arguments
from load_data import load_train_data, load_test_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gradient_analysis.log")
    ]
)
logger = logging.getLogger(__name__)

def setup_gradient_hooks(model):
    """Setup hooks to track gradient flow through the model"""
    gradients = {}
    
    def save_grad(name):
        def hook(grad):
            gradients[name] = grad.detach().cpu().clone()
        return hook
    
    # Register hooks for all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(save_grad(name))
    
    return gradients

def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for vanishing/exploding gradients.
    
    Usage: Plug this function in after loss.backward() to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    
    plt.figure(figsize=(15, 10))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # Zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('gradient_flow.png')
    logger.info("Saved gradient flow plot to gradient_flow.png")

def plot_weight_distributions(model):
    """Plot the distribution of weights in different layers"""
    plt.figure(figsize=(15, 10))
    
    # Get all weight parameters
    weights = [param.data.flatten().cpu().numpy() for name, param in model.named_parameters() 
               if 'weight' in name and param.dim() > 1]
    
    # Plot histograms
    for i, w in enumerate(weights):
        plt.subplot(len(weights), 1, i+1)
        plt.hist(w, bins=100, alpha=0.7)
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('weight_distributions.png')
    logger.info("Saved weight distributions plot to weight_distributions.png")

def test_different_learning_rates(train_loader, device, opt):
    """Test different learning rates and monitor loss behavior"""
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    num_iterations = 20
    
    plt.figure(figsize=(10, 6))
    
    for lr in learning_rates:
        losses = []
        
        # Create model and optimizer with this learning rate
        model = YeNet(opt.in_channels, opt.num_cls).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Train for a few iterations
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            if i >= num_iterations:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Plot losses for this learning rate
        plt.plot(range(1, len(losses)+1), losses, label=f'LR: {lr}')
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs. Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_rate_test.png')
    logger.info("Saved learning rate test plot to learning_rate_test.png")

def analyze_activations(model, train_loader, device):
    """Analyze activations in different layers"""
    # Get a batch of data
    images, _ = next(iter(train_loader))
    images = images.to(device)
    
    # Dictionary to store activations
    activations = {}
    
    # Define hook function
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook
    
    # Register hooks for each layer
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            module.register_forward_hook(get_activation(name))
    
    # Forward pass
    _ = model(images)
    
    # Plot activations
    plt.figure(figsize=(15, 10))
    
    # Select a few key layers to visualize
    layers_to_plot = list(activations.keys())[:min(8, len(activations))]
    
    for i, layer_name in enumerate(layers_to_plot):
        plt.subplot(len(layers_to_plot), 1, i+1)
        act = activations[layer_name].numpy().reshape(-1)
        plt.hist(act, bins=50)
        plt.title(f'Layer: {layer_name}')
        plt.xlabel('Activation Value')
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('activation_distributions.png')
    logger.info("Saved activation distributions plot to activation_distributions.png")

def main():
    logger.info("Starting YeNet gradient analysis")
    
    # Load arguments
    opt = arguments()
    
    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    train_loader, _ = load_train_data(opt)
    
    # Create model, loss function, and optimizer
    model = YeNet(opt.in_channels, opt.num_cls).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=opt.lr,
        weight_decay=opt.decay
    )
    
    logger.info("Analyzing model architecture...")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Plot initial weight distributions
    logger.info("Plotting initial weight distributions...")
    plot_weight_distributions(model)
    
    # Test different learning rates
    logger.info("Testing different learning rates...")
    test_different_learning_rates(train_loader, device, opt)
    
    # Setup gradient hooks
    logger.info("Setting up gradient tracking...")
    gradients = setup_gradient_hooks(model)
    
    # Train for a few iterations and track gradients
    logger.info("Training and tracking gradients for a few iterations...")
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        if i >= 5:  # Only run a few iterations
            break
            
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Plot gradient flow
        if i == 4:  # Plot on the last iteration
            plot_grad_flow(model.named_parameters())
        
        optimizer.step()
        
        logger.info(f"Iteration {i+1}, Loss: {loss.item()}")
    
    # Analyze activations
    logger.info("Analyzing activations...")
    analyze_activations(model, train_loader, device)
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
