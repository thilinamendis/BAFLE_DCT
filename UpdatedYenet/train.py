"""This module is use to train the yeNet model."""

import logging
import os
import sys
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import sys
sys.path.append('./UpdatedYenet')

# from dataset import dataset
from opts.options import arguments
from model.model_yenet import YeNet
print("✅ YeNet imported successfully")

from utils.utils import (
    latest_checkpoint,
    adjust_learning_rate,
    weights_init,
    saver,
)
from load_data import load_train_data, load_test_data

opt = arguments()

# Add balanced dataset options to arguments (default values can be adjusted as needed)
opt.use_balanced_dataset = False  # Default to False
opt.use_paired_balanced = True   # Default to True for paired balanced

# Setup directories for logs and plots
logs_dir = Path('./logs')
plots_dir = Path('./plots')
logs_dir.mkdir(exist_ok=True)
plots_dir.mkdir(exist_ok=True)

# Configure timestamp for this training run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"training_{timestamp}.log"

# Add logging to file and console
logging.basicConfig(
    filename=str(logs_dir / log_filename),
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Initialize history tracking for plotting
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'lr': []
}

def plot_training_graphs(history, epoch, timestamp):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./plots/loss_acc_epoch_{epoch}_{timestamp}.png')
    plt.close()
    plt.figure(figsize=(10, 4))
    plt.plot(history['lr'], marker='o')
    plt.title('Learning Rate Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f'./plots/learning_rate_epoch_{epoch}_{timestamp}.png')
    plt.close()
    logging.info(f"Training plots saved at epoch {epoch}")

def plot_accuracy_vs_epoch(history, timestamp):
    plt.figure(figsize=(10, 6))
    epochs = list(range(2, 2*len(history['train_acc'])+1, 2))
    plt.plot(epochs, history['train_acc'], marker='o', linestyle='-', linewidth=2, label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], marker='s', linestyle='-', linewidth=2, label='Validation Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Accuracy over Training Epochs', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    min_acc = min(min(history['train_acc']), min(history['val_acc']))
    plt.ylim([max(0, min_acc-10), 100])
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=75, color='g', linestyle='--', alpha=0.3)
    plt.axhline(y=90, color='b', linestyle='--', alpha=0.3)
    for i, (train_acc, val_acc) in enumerate(zip(history['train_acc'], history['val_acc'])):
        epoch_num = epochs[i]
        if i % 2 == 0:
            plt.annotate(f'{train_acc:.1f}', (epoch_num, train_acc), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            plt.annotate(f'{val_acc:.1f}', (epoch_num, val_acc), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'./plots/accuracy_vs_epoch_{timestamp}.png', dpi=300)
    plt.savefig(f'./plots/accuracy_vs_epoch_{timestamp}.pdf')
    plt.close()
    logging.info(f"Accuracy vs Epoch plot saved as accuracy_vs_epoch_{timestamp}.png/pdf")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    logging.info(f"Starting YeNet training with the following configuration:")
    logging.info(f"Device: {device}")
    logging.info(f"Cover path: {opt.cover_path}")
    logging.info(f"Stego path: {opt.stego_path}")
    logging.info(f"Validation cover path: {opt.valid_cover_path}")
    logging.info(f"Validation stego path: {opt.valid_stego_path}")
    logging.info(f"Learning rate: {opt.lr}")
    logging.info(f"Number of epochs: {opt.num_epochs}")
    logging.info(f"Train batch size: {opt.train_size}")
    logging.info(f"Validation batch size: {opt.val_size}")
    logging.info(f"Using standard balanced dataset: {opt.use_balanced_dataset}")
    logging.info(f"Using paired balanced dataset: {opt.use_paired_balanced}")

    # Define transform to resize images to 512×512 and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # Pass balanced/paired options to data loaders
    train_loader = load_train_data(
        opt.cover_path, 
        opt.stego_path, 
        opt.train_size,
        balanced=opt.use_balanced_dataset,
        paired_balanced=opt.use_paired_balanced,
        transform=transform
    )
    valid_loader = load_test_data(
        opt.valid_cover_path, 
        opt.valid_stego_path, 
        opt.val_size,
        balanced=opt.use_balanced_dataset,
        paired_balanced=opt.use_paired_balanced,
        transform=transform
    )

    logging.info(f"Train dataset size: {len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 'Unknown'}")
    logging.info(f"Validation dataset size: {len(valid_loader.dataset) if hasattr(valid_loader, 'dataset') else 'Unknown'}")

    # model creation and initialization.
    model = YeNet()
    model.to(device)
    model = model.apply(weights_init)
    logging.info(f"Model initialized on {device}")

    # Loss function and Optimizer
    # loss_fn = nn.NLLLoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=opt.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )

    check_point = latest_checkpoint()
    if not check_point:
        START_EPOCH = 1
        if not os.path.exists(opt.checkpoints_dir):
            os.makedirs(opt.checkpoints_dir)
        logging.info("No checkpoints found! Retraining started...")
    else:
        pth = opt.checkpoints_dir + "net_" + str(check_point) + ".pt"
        ckpt = torch.load(pth)
        START_EPOCH = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "train_loss_history" in ckpt:
            history['train_loss'] = ckpt["train_loss_history"]
            history['train_acc'] = ckpt["train_acc_history"]
            history['val_loss'] = ckpt["val_loss_history"]
            history['val_acc'] = ckpt["val_acc_history"]
            history['lr'] = ckpt["lr_history"]
        logging.info(f"Model loaded from checkpoint, resuming from epoch {START_EPOCH}")

    for epoch in range(START_EPOCH, opt.num_epochs + 1):
        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []
        test_accuracy = []

        # Training
        model.train()
        st_time = time.time()
        adjust_learning_rate(optimizer, epoch)
        stream = tqdm(train_loader)
        for i, batch in enumerate(stream):
            # Handle different batch formats based on balancing method
            if opt.use_balanced_dataset and not opt.use_paired_balanced:
                # For regular balanced dataset (unpaired)
                images = batch["image"].to(device, dtype=torch.float)
                labels = batch["label"].to(device, dtype=torch.long)
            else:
                # For original implementation or paired balanced (both use same format)
                images = torch.cat((batch["cover"], batch["stego"]), 0)
                labels = torch.cat((batch["label"][0], batch["label"][1]), 0)
                images = images.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
            prediction = outputs.data.max(1)[1]
            accuracy = (prediction.eq(labels.data).sum() * 100.0 / (labels.size()[0]))
            training_accuracy.append(accuracy.item())
            sys.stdout.write(
                f"\r Epoch:{epoch}/{opt.num_epochs}"
                f" Batch:{i+1}/{len(train_loader)}"
                f" Loss:{training_loss[-1]:.4f}"
                f" Acc:{training_accuracy[-1]:.2f}"
                f" LR:{optimizer.param_groups[0]['lr']:.4f}"
            )
        end_time = time.time()
        avg_epoch_train_loss = sum(training_loss) / len(training_loss)
        avg_epoch_train_acc = sum(training_accuracy) / len(training_accuracy)
        epoch_time = end_time - st_time
        logging.info(f"Epoch {epoch}/{opt.num_epochs} - Time: {epoch_time:.2f}s - "
                     f"Training Loss: {avg_epoch_train_loss:.5f} - "
                     f"Training Accuracy: {avg_epoch_train_acc:.2f}% - "
                     f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        # Validation
        if epoch % 2 == 0 or epoch == opt.num_epochs:
            model.eval()
            with torch.no_grad():
                stream = tqdm(valid_loader)
                for i, batch in enumerate(stream):
                    if opt.use_balanced_dataset and not opt.use_paired_balanced:
                        images = batch["image"].to(device, dtype=torch.float)
                        labels = batch["label"].to(device, dtype=torch.long)
                    else:
                        images = torch.cat((batch["cover"], batch["stego"]), 0)
                        labels = torch.cat((batch["label"][0], batch["label"][1]), 0)
                        images = images.to(device, dtype=torch.float)
                        labels = labels.to(device, dtype=torch.long)
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    validation_loss.append(loss.item())
                    prediction = outputs.data.max(1)[1]
                    accuracy = (prediction.eq(labels.data).sum() * 100.0 / (labels.size()[0]))
                    validation_accuracy.append(accuracy.item())
            avg_train_loss = sum(training_loss) / len(training_loss)
            avg_valid_loss = sum(validation_loss) / len(validation_loss)
            avg_train_acc = sum(training_accuracy) / len(training_accuracy)
            avg_valid_acc = sum(validation_accuracy) / len(validation_accuracy)
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(avg_train_acc)
            history['val_loss'].append(avg_valid_loss)
            history['val_acc'].append(avg_valid_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            message = (
                f"Epoch: {epoch}. "
                f"Train Loss:{avg_train_loss:.5f}. "
                f"Valid Loss:{avg_valid_loss:.5f}. "
                f"Train Acc:{avg_train_acc:.2f}% "
                f"Valid Acc:{avg_valid_acc:.2f}% "
                f"LR:{optimizer.param_groups[0]['lr']:.6f}"
            )
            print("\n", message)
            logging.info(message)
            
            # Record the current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save plots at the final epoch
            if epoch == opt.num_epochs:
                plot_training_graphs(history, epoch, timestamp)
                plot_accuracy_vs_epoch(history, timestamp)
            
            # Save model state
            state = {
                "epoch": epoch,
                "opt": opt,
                "train_loss": avg_train_loss,
                "valid_loss": avg_valid_loss,
                "train_accuracy": avg_train_acc,
                "valid_accuracy": avg_valid_acc,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr": optimizer.param_groups[0]['lr'],
                "train_loss_history": history['train_loss'],
                "train_acc_history": history['train_acc'],
                "val_loss_history": history['val_loss'],
                "val_acc_history": history['val_acc'],
                "lr_history": history['lr']
            }
            
            # Save the model
            saver(state, opt.checkpoints_dir, epoch)
            logging.info(f"Model saved at epoch {epoch}")
            
            # Save the model
            logging.info(f"Model saved at epoch {epoch}")
