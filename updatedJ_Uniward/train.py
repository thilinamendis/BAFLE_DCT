"""This module is used to train the J-UNIWARD model."""

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
sys.path.append('./updatedJ_Uniward')

from opts.options import arguments
from model.model_juniwarden import JUniwarden
print("✅ J-UNIWARD model imported successfully")

from utils.utils import (
    latest_checkpoint,
    adjust_learning_rate,
    weights_init,
    saver,
    calculate_accuracy,
    calculate_confusion_matrix
)
from load_data import load_train_data, load_test_data

opt = arguments()

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
    'lr': [],
    'train_precision': [],
    'train_recall': [],
    'train_f1': [],
    'val_precision': [],
    'val_recall': [],
    'val_f1': []
}

def plot_training_graphs(history, epoch, timestamp):
    """Plot training metrics and save to disk.
    
    Args:
        history: Dictionary containing training history
        epoch: Current epoch number
        timestamp: Timestamp for unique filenames
    """
    # Loss and accuracy plot
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
    
    # Learning rate plot
    plt.figure(figsize=(10, 4))
    plt.plot(history['lr'], marker='o')
    plt.title('Learning Rate Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f'./plots/learning_rate_epoch_{epoch}_{timestamp}.png')
    plt.close()
    
    # Precision, Recall, F1 plot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_precision'], label='Training Precision')
    plt.plot(history['val_precision'], label='Validation Precision')
    plt.title('Precision Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_recall'], label='Training Recall')
    plt.plot(history['val_recall'], label='Validation Recall')
    plt.title('Recall Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='Training F1')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.title('F1 Score Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'./plots/metrics_epoch_{epoch}_{timestamp}.png')
    plt.close()
    
    logging.info(f"Training plots saved at epoch {epoch}")

def plot_accuracy_vs_epoch(history, timestamp):
    """Create a dedicated accuracy vs epoch plot.
    
    Args:
        history: Dictionary containing training history
        timestamp: Timestamp for unique filenames
    """
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
    logging.info(f"Starting J-UNIWARD training with the following configuration:")
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
    logging.info(f"JPEG quality factor: {opt.quality_factor}")
    logging.info(f"Direct DCT analysis: {opt.dct_direct_analysis}")

    # Define transform to resize images and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # Check if paths exist and have files
    from glob import glob
    cover_files = glob(os.path.join(opt.cover_path, "*.*"))
    stego_files = glob(os.path.join(opt.stego_path, "*.*"))
    valid_cover_files = glob(os.path.join(opt.valid_cover_path, "*.*"))
    valid_stego_files = glob(os.path.join(opt.valid_stego_path, "*.*"))
    
    # Log the number of files found
    logging.info(f"Found {len(cover_files)} cover training files")
    logging.info(f"Found {len(stego_files)} stego training files")
    logging.info(f"Found {len(valid_cover_files)} cover validation files")
    logging.info(f"Found {len(valid_stego_files)} stego validation files")
    
    if len(cover_files) == 0 or len(stego_files) == 0:
        logging.error(f"No training files found in cover path: {opt.cover_path} or stego path: {opt.stego_path}")
        logging.error("Please check that the paths are correct and contain image files")
        sys.exit(1)
    
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

    # Model creation and initialization
    model = JUniwarden(quality_factor=opt.quality_factor, direct_dct=opt.dct_direct_analysis)
    model.to(device)
    model = model.apply(weights_init)
    logging.info(f"Model initialized on {device}")

    # Loss function and Optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=opt.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )

    # Load checkpoint if exists
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
            
            if "train_precision_history" in ckpt:
                history['train_precision'] = ckpt["train_precision_history"]
                history['train_recall'] = ckpt["train_recall_history"]
                history['train_f1'] = ckpt["train_f1_history"]
                history['val_precision'] = ckpt["val_precision_history"]
                history['val_recall'] = ckpt["val_recall_history"]
                history['val_f1'] = ckpt["val_f1_history"]
            
        logging.info(f"Model loaded from checkpoint, resuming from epoch {START_EPOCH}")

        # Ensure we have data before starting training
    if len(train_loader) == 0:
        logging.error("Empty training dataset! Please check the data paths and file formats.")
        sys.exit(1)
        
    if len(valid_loader) == 0:
        logging.error("Empty validation dataset! Please check the data paths and file formats.")
        sys.exit(1)

    # Start training loop
    for epoch in range(START_EPOCH, opt.num_epochs + 1):
        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []
        
        # Metrics
        train_metrics = {
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'precision': 0, 'recall': 0, 'f1_score': 0
        }
        val_metrics = {
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'precision': 0, 'recall': 0, 'f1_score': 0
        }

        # Training
        model.train()
        st_time = time.time()
        adjust_learning_rate(optimizer, epoch)
        # Configure tqdm to avoid interference with logging
        stream = tqdm(
            train_loader, 
            desc=f"Epoch {epoch}/{opt.num_epochs}",
            leave=False,
            ncols=100,
            position=0
        )
        
        for i, batch in enumerate(stream):
            # Handle different batch formats based on balancing method
            try:
                if opt.use_balanced_dataset and not opt.use_paired_balanced:
                    # For regular balanced dataset (unpaired)
                    images = batch["image"].to(device, dtype=torch.float)
                    labels = batch["label"].to(device, dtype=torch.long)
                elif opt.use_paired_balanced:
                    # For paired balanced (cover-stego pairs)
                    cover_images = batch["cover"].to(device, dtype=torch.float)
                    stego_images = batch["stego"].to(device, dtype=torch.float)
                    images = torch.cat((cover_images, stego_images), 0)
                    
                    # Create corresponding labels (0 for cover, 1 for stego)
                    batch_size = cover_images.size(0)
                    cover_labels = torch.zeros(batch_size, dtype=torch.long).to(device)
                    stego_labels = torch.ones(batch_size, dtype=torch.long).to(device)
                    labels = torch.cat((cover_labels, stego_labels), 0)
                    
                    logging.debug(f"Paired balanced batch: {images.shape}, labels: {labels.shape}")
                else:
                    # For original implementation
                    images = torch.cat((batch["cover"], batch["stego"]), 0)
                    labels = torch.cat((batch["label"][0], batch["label"][1]), 0)
                    images = images.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)
            except Exception as e:
                logging.error(f"Error processing batch: {e}")
                logging.error(f"Batch keys: {batch.keys()}")
                for k, v in batch.items():
                    if hasattr(v, 'shape'):
                        logging.error(f"Batch[{k}].shape = {v.shape}")
                    else:
                        logging.error(f"Batch[{k}] = {v}")
                continue  # Skip this batch
                
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Track metrics
            training_loss.append(loss.item())
            batch_accuracy = calculate_accuracy(outputs, labels)
            training_accuracy.append(batch_accuracy)
            
            # Calculate batch metrics
            batch_metrics = calculate_confusion_matrix(outputs, labels)
            for k, v in batch_metrics.items():
                if k in train_metrics:
                    train_metrics[k] += v
            
            # Update the tqdm progress bar with the current metrics
            stream.set_postfix({
                'loss': f"{training_loss[-1]:.4f}",
                'acc': f"{training_accuracy[-1]:.2f}%",
                'LR': f"{optimizer.param_groups[0]['lr']:.4f}"
            })
            
        end_time = time.time()
        
        # Calculate epoch metrics
        avg_epoch_train_loss = sum(training_loss) / len(training_loss)
        avg_epoch_train_acc = sum(training_accuracy) / len(training_accuracy)
        epoch_time = end_time - st_time
        
        # Calculate precision, recall, F1
        n_samples = sum([train_metrics['tp'], train_metrics['fp'], train_metrics['tn'], train_metrics['fn']])
        if n_samples > 0:
            train_metrics['precision'] = train_metrics['tp'] / (train_metrics['tp'] + train_metrics['fp']) if (train_metrics['tp'] + train_metrics['fp']) > 0 else 0
            train_metrics['recall'] = train_metrics['tp'] / (train_metrics['tp'] + train_metrics['fn']) if (train_metrics['tp'] + train_metrics['fn']) > 0 else 0
            train_metrics['f1_score'] = 2 * train_metrics['precision'] * train_metrics['recall'] / (train_metrics['precision'] + train_metrics['recall']) if (train_metrics['precision'] + train_metrics['recall']) > 0 else 0
        
        # Clear the progress bar line and print a newline to avoid interference
        print()
        
        # Log the epoch results
        logging.info(f"Epoch {epoch}/{opt.num_epochs} - Time: {epoch_time:.2f}s - "
                     f"Training Loss: {avg_epoch_train_loss:.5f} - "
                     f"Training Accuracy: {avg_epoch_train_acc:.2f}% - "
                     f"Precision: {train_metrics['precision']:.4f} - "
                     f"Recall: {train_metrics['recall']:.4f} - "
                     f"F1: {train_metrics['f1_score']:.4f} - "
                     f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                     
        # Validation
        if epoch % 2 == 0 or epoch == opt.num_epochs:
            model.eval()
            with torch.no_grad():
                stream = tqdm(
                    valid_loader, 
                    desc=f"Validation {epoch}/{opt.num_epochs}",
                    leave=False,
                    ncols=100,
                    position=0
                )
                for i, batch in enumerate(stream):
                    # Handle different batch formats
                    try:
                        if opt.use_balanced_dataset and not opt.use_paired_balanced:
                            # For regular balanced dataset (unpaired)
                            images = batch["image"].to(device, dtype=torch.float)
                            labels = batch["label"].to(device, dtype=torch.long)
                        elif opt.use_paired_balanced:
                            # For paired balanced (cover-stego pairs)
                            cover_images = batch["cover"].to(device, dtype=torch.float)
                            stego_images = batch["stego"].to(device, dtype=torch.float)
                            images = torch.cat((cover_images, stego_images), 0)
                            
                            # Create corresponding labels (0 for cover, 1 for stego)
                            batch_size = cover_images.size(0)
                            cover_labels = torch.zeros(batch_size, dtype=torch.long).to(device)
                            stego_labels = torch.ones(batch_size, dtype=torch.long).to(device)
                            labels = torch.cat((cover_labels, stego_labels), 0)
                        else:
                            # For original implementation
                            images = torch.cat((batch["cover"], batch["stego"]), 0)
                            labels = torch.cat((batch["label"][0], batch["label"][1]), 0)
                            images = images.to(device, dtype=torch.float)
                            labels = labels.to(device, dtype=torch.long)
                    except Exception as e:
                        logging.error(f"Error processing validation batch: {e}")
                        logging.error(f"Batch keys: {batch.keys()}")
                        continue  # Skip this batch
                        
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    validation_loss.append(loss.item())
                    
                    # Calculate accuracy
                    batch_accuracy = calculate_accuracy(outputs, labels)
                    validation_accuracy.append(batch_accuracy)
                    
                    # Calculate batch metrics
                    batch_metrics = calculate_confusion_matrix(outputs, labels)
                    for k, v in batch_metrics.items():
                        if k in val_metrics:
                            val_metrics[k] += v
            
            # Calculate validation metrics
            avg_train_loss = sum(training_loss) / len(training_loss)
            avg_valid_loss = sum(validation_loss) / len(validation_loss)
            avg_train_acc = sum(training_accuracy) / len(training_accuracy)
            avg_valid_acc = sum(validation_accuracy) / len(validation_accuracy)
            
            # Calculate precision, recall, F1 for validation
            n_samples_val = sum([val_metrics['tp'], val_metrics['fp'], val_metrics['tn'], val_metrics['fn']])
            if n_samples_val > 0:
                val_metrics['precision'] = val_metrics['tp'] / (val_metrics['tp'] + val_metrics['fp']) if (val_metrics['tp'] + val_metrics['fp']) > 0 else 0
                val_metrics['recall'] = val_metrics['tp'] / (val_metrics['tp'] + val_metrics['fn']) if (val_metrics['tp'] + val_metrics['fn']) > 0 else 0
                val_metrics['f1_score'] = 2 * val_metrics['precision'] * val_metrics['recall'] / (val_metrics['precision'] + val_metrics['recall']) if (val_metrics['precision'] + val_metrics['recall']) > 0 else 0
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(avg_train_acc)
            history['val_loss'].append(avg_valid_loss)
            history['val_acc'].append(avg_valid_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['train_precision'].append(train_metrics['precision'])
            history['train_recall'].append(train_metrics['recall'])
            history['train_f1'].append(train_metrics['f1_score'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            history['val_f1'].append(val_metrics['f1_score'])
            
            # Log validation results
            message = (
                f"Epoch: {epoch}. "
                f"Train Loss:{avg_train_loss:.5f}. "
                f"Valid Loss:{avg_valid_loss:.5f}. "
                f"Train Acc:{avg_train_acc:.2f}% "
                f"Valid Acc:{avg_valid_acc:.2f}% "
                f"Train F1:{train_metrics['f1_score']:.4f} "
                f"Valid F1:{val_metrics['f1_score']:.4f} "
                f"LR:{optimizer.param_groups[0]['lr']:.6f}"
            )
            # Ensure we're not interfering with progress bars
            print("\n" + message)
            logging.info(message)
            
            # Print confusion matrix
            logging.info(f"Confusion Matrix - TP: {val_metrics['tp']}, FP: {val_metrics['fp']}, "
                         f"TN: {val_metrics['tn']}, FN: {val_metrics['fn']}")
            
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
                "lr_history": history['lr'],
                "train_precision_history": history['train_precision'],
                "train_recall_history": history['train_recall'],
                "train_f1_history": history['train_f1'],
                "val_precision_history": history['val_precision'],
                "val_recall_history": history['val_recall'],
                "val_f1_history": history['val_f1'],
                "confusion_matrix": {
                    "tp": val_metrics['tp'],
                    "fp": val_metrics['fp'],
                    "tn": val_metrics['tn'],
                    "fn": val_metrics['fn']
                }
            }
            
            # Save the model
            saver(state, opt.checkpoints_dir, epoch)
            logging.info(f"Model saved at epoch {epoch}")
