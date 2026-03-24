"""
YeNet Model Evaluation Script

This script evaluates the YeNet steganalysis model on test data,
producing detailed metrics, visualizations, and logs.

Usage:
    python test.py

Author: Research AI Stegno Analysis Team
"""

import datetime
import logging
from pathlib import Path
from glob import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    average_precision_score, confusion_matrix,
    precision_recall_fscore_support
)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    print("Seaborn not installed. Confusion matrix will use matplotlib instead.")
    HAS_SEABORN = False

from model.model_yenet import YeNet

# Setup directories for logs and plots
logs_dir = Path('./logs')
plots_dir = Path('./plots')
logs_dir.mkdir(exist_ok=True)
plots_dir.mkdir(exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"test_{timestamp}.log"

# Set up logging configuration with both file and console output
log_file_path = logs_dir / log_filename
print(f"Setting up logging to file: {log_file_path}")

# Reset root logger to avoid duplicate handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

# Test logging
logging.info("Logging initialized successfully")

# Helper functions for plotting and metrics calculation
def plot_roc_curve(fpr, tpr, roc_auc, output_path):
    """
    Plot ROC curve and save to file.
    
    Args:
        fpr (ndarray): False positive rates
        tpr (ndarray): True positive rates
        roc_auc (float): Area under ROC curve
        output_path (Path): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_path, bbox_inches='tight')
    logging.info(f"ROC curve saved to {output_path}")
    plt.close()

def plot_pr_curve(precision, recall, average_precision, output_path):
    """
    Plot Precision-Recall curve and save to file.
    
    Args:
        precision (ndarray): Precision values
        recall (ndarray): Recall values
        average_precision (float): Average precision score
        output_path (Path): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(output_path, bbox_inches='tight')
    logging.info(f"Precision-Recall curve saved to {output_path}")
    plt.close()

def plot_det_curve(fpr, fnr, output_path):
    """
    Plot Detection Error Tradeoff (DET) curve and save to file.
    
    Args:
        fpr (ndarray): False positive rates
        fnr (ndarray): False negative rates (1 - tpr)
        output_path (Path): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Convert to percentage
    fpr_percent = 100 * fpr
    fnr_percent = 100 * fnr
    
    # Apply probit transform for DET curve (maps to normal distribution)
    probit_fpr = norm.ppf(fpr)
    probit_fnr = norm.ppf(fnr)
    
    # Filter out infinite values
    mask = np.isfinite(probit_fpr) & np.isfinite(probit_fnr)
    probit_fpr = probit_fpr[mask]
    probit_fnr = probit_fnr[mask]
    
    if len(probit_fpr) > 0 and len(probit_fnr) > 0:
        plt.plot(probit_fpr, probit_fnr, 'b-', lw=2)
        
        # Set custom tick locations and labels for x and y axes
        tick_locs = norm.ppf([0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999])
        tick_labels = [0.1, 1, 5, 20, 50, 80, 95, 99, 99.9]
        
        plt.xticks(tick_locs, tick_labels)
        plt.yticks(tick_locs, tick_labels)
        
        plt.xlim(norm.ppf(0.001), norm.ppf(0.999))
        plt.ylim(norm.ppf(0.001), norm.ppf(0.999))
        
        plt.grid(True)
        plt.xlabel('False Positive Rate (%)')
        plt.ylabel('False Negative Rate (%)')
        plt.title('Detection Error Tradeoff (DET) Curve')
        
        plt.savefig(output_path, bbox_inches='tight')
        logging.info(f"DET curve saved to {output_path}")
    else:
        logging.warning("Could not plot DET curve due to insufficient valid data points")
    
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_path):
    """
    Plot confusion matrix and save to file.
    
    Args:
        y_true (ndarray): True labels
        y_pred (ndarray): Predicted labels
        output_path (Path): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=['Cover', 'Stego'],
                    yticklabels=['Cover', 'Stego'])
    else:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = [0, 1]
        plt.xticks(tick_marks, ['Cover', 'Stego'], rotation=45)
        plt.yticks(tick_marks, ['Cover', 'Stego'])
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig(output_path, bbox_inches='tight')
    logging.info(f"Confusion matrix saved to {output_path}")
    plt.close()

def calculate_metrics(y_true, y_pred_probs):
    """
    Calculate various performance metrics.
    
    Args:
        y_true (ndarray): True labels
        y_pred_probs (ndarray): Predicted probabilities
    
    Returns:
        dict: Dictionary containing various metrics
    """
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Precision-Recall curve and Average Precision
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    ap = average_precision_score(y_true, y_pred_probs)
    
    # Calculate FNR for DET curve
    fnr = 1 - tpr
    
    # Get binary predictions using 0.5 as threshold
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    # Calculate precision, recall, F1-score
    precision_val, recall_val, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    
    # Calculate accuracy
    accuracy = (y_true == y_pred).mean()
    
    # Calculate confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate Equal Error Rate (EER)
    abs_diff = np.abs(fpr - fnr)
    idx = np.argmin(abs_diff)
    eer = (fpr[idx] + fnr[idx]) / 2
    
    # Find points at common operating thresholds
    fnr_at_fpr01 = np.interp(0.01, fpr, fnr) if 0.01 <= max(fpr) else None
    fnr_at_fpr05 = np.interp(0.05, fpr, fnr) if 0.05 <= max(fpr) else None
    fnr_at_fpr10 = np.interp(0.1, fpr, fnr) if 0.1 <= max(fpr) else None
    
    return {
        'accuracy': accuracy,
        'precision': precision_val,
        'recall': recall_val,
        'f1_score': f1_score,
        'roc_auc': roc_auc,
        'average_precision': ap,
        'fpr': fpr,
        'tpr': tpr,
        'fnr': fnr,
        'precision_curve': precision,
        'recall_curve': recall,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'y_pred': y_pred,
        'eer': eer,
        'fnr_at_fpr01': fnr_at_fpr01,
        'fnr_at_fpr05': fnr_at_fpr05,
        'fnr_at_fpr10': fnr_at_fpr10
    }

# This function has been unified with the other plot_confusion_matrix function defined above

def plot_accuracy_by_class(cover_acc, stego_acc, overall_acc, timestamp, balanced_acc=None):
    """
    Create a bar chart showing accuracy by class
    
    Args:
        cover_acc (float): Accuracy for cover images
        stego_acc (float): Accuracy for stego images
        overall_acc (float): Overall accuracy
        timestamp (str): Timestamp for filename
        balanced_acc (float): Balanced accuracy
    """
    plt.figure(figsize=(12, 6))
    
    # Create bar chart
    classes = ['Cover (Not-Stego)', 'Stego', 'Overall', 'Balanced']
    accuracies = [cover_acc, stego_acc, overall_acc, balanced_acc]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    bars = plt.bar(classes, accuracies, color=colors, width=0.5)
    
    # Add percentage numbers on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.2f}%', ha='center', fontsize=12)
    
    # Customize plot
    plt.ylim(0, 105)  # 0-100% + some padding
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title('Accuracy Metrics', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12)
    
    # Add explanation text
    explanation = "Overall: Accuracy across all samples\nBalanced: Average of class-wise accuracies"
    plt.figtext(0.75, 0.01, explanation, wrap=True, horizontalalignment='center', fontsize=10)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(plots_dir / f"test_accuracy_by_class_{timestamp}.png", dpi=300)
    plt.savefig(plots_dir / f"test_accuracy_by_class_{timestamp}.pdf")  # PDF for publication quality
    plt.close()
    
    logging.info(f"Accuracy by class plot saved as test_accuracy_by_class_{timestamp}.png")

def plot_precision_recall_f1(precision, recall, f1_score, timestamp):
    """
    Create a bar chart showing precision, recall, and F1 scores
    
    Args:
        precision (dict): Precision values for each class
        recall (dict): Recall values for each class
        f1_score (dict): F1 scores for each class
        timestamp (str): Timestamp for filename
    """
    plt.figure(figsize=(12, 8))
    
    # Set up data for plotting
    metrics_labels = ['Precision', 'Recall', 'F1 Score']
    cover_metrics = [precision['cover'], recall['cover'], f1_score['cover']]
    stego_metrics = [precision['stego'], recall['stego'], f1_score['stego']]
    macro_metrics = [precision['macro'], recall['macro'], f1_score['macro']]
    weighted_metrics = [precision['weighted'], recall['weighted'], f1_score['weighted']]
    
    # Set width and positions
    barWidth = 0.2
    r1 = np.arange(len(metrics_labels))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    
    # Create bars
    plt.bar(r1, cover_metrics, width=barWidth, label='Cover (Not-Stego)', color='#3498db')
    plt.bar(r2, stego_metrics, width=barWidth, label='Stego', color='#2ecc71')
    plt.bar(r3, macro_metrics, width=barWidth, label='Macro Avg', color='#f39c12')
    plt.bar(r4, weighted_metrics, width=barWidth, label='Weighted Avg', color='#9b59b6')
    
    # Add labels and title
    plt.xlabel('Metrics', fontweight='bold', fontsize=12)
    plt.ylabel('Score', fontweight='bold', fontsize=12)
    plt.title('Precision, Recall, and F1 Scores', fontweight='bold', fontsize=16)
    plt.xticks([r + barWidth*1.5 for r in range(len(metrics_labels))], metrics_labels)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=4)
    
    # Add value labels on bars
    def add_labels(positions, values, color):
        for i, (pos, val) in enumerate(zip(positions, values)):
            plt.text(pos, val + 0.02, f'{val:.4f}', ha='center', va='bottom', fontsize=9, color=color)
    
    add_labels(r1, cover_metrics, '#3498db')
    add_labels(r2, stego_metrics, '#2ecc71')
    add_labels(r3, macro_metrics, '#f39c12')
    add_labels(r4, weighted_metrics, '#9b59b6')
    
    # Save plots
    plt.tight_layout(pad=3)
    plt.savefig(plots_dir / f"test_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / f"test_metrics_{timestamp}.pdf", bbox_inches='tight')  # PDF for publication quality
    plt.close()
    
    logging.info(f"Precision, Recall, F1 plot saved as test_metrics_{timestamp}.png")

def load_model(model_path='checkpoints/net_50.pt'):
    """
    Load the YeNet model from checkpoint.
    
    Args:
        model_path (str): Path to model checkpoint
        
    Returns:
        YeNet: Loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model from {model_path} on {device}")
    
    model = YeNet().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()  # Set the model to evaluation mode
    
    return model, device


def get_image_paths(cover_dir, stego_dir):
    """
    Get lists of cover and stego image paths.
    
    Args:
        cover_dir (str): Path pattern for cover images
        stego_dir (str): Path pattern for stego images
        
    Returns:
        tuple: Lists of cover and stego image paths
    """
    logging.info(f"Searching for test images in {cover_dir} and {stego_dir}")
    
    # Get all image paths
    cover_images = sorted(glob(cover_dir))
    stego_images = sorted(glob(stego_dir))
    
    if not cover_images or not stego_images:
        logging.error("No test images found. Check your data paths.")
        return [], []
    
    logging.info(f"Found {len(cover_images)} cover images and {len(stego_images)} stego images")
    return cover_images, stego_images


def load_and_resize(image_paths):
    """
    Load and preprocess images.
    
    Args:
        image_paths (list): List of image paths
        
    Returns:
        list: List of (path, preprocessed_image) tuples
    """
    resized = []
    for p in image_paths:
        # Open the image in RGB mode
        img = Image.open(p).convert("RGB")
        # Use 256x256 to match the training dimensions in load_data.py
        img = img.resize((256, 256), Image.LANCZOS)
        img_array = np.array(img)
        # Normalize pixel values to [0, 1] range, which is what ToTensor() does in training
        img_array = img_array / 255.0
        resized.append((p, img_array))
    return resized


def process_batch(model, device, cover_batch, stego_batch):
    """
    Process a batch of cover and stego images.
    
    Args:
        model (YeNet): Model to use for prediction
        device (torch.device): Device to run model on
        cover_batch (list): List of cover (path, image) tuples
        stego_batch (list): List of stego (path, image) tuples
        
    Returns:
        tuple: Lists of predictions, ground truth, scores, and batch accuracy
    """
    batch = []
    batch_labels = []
    batch_filenames = []
    
    # Interleave cover and stego images
    for i in range(len(cover_batch) + len(stego_batch)):
        if i % 2 == 0 and len(stego_batch) > i//2:
            filename, img = stego_batch[i//2]
            batch.append(img)
            batch_labels.append(1)  # Stego class
            batch_filenames.append(filename)
        elif i % 2 == 1 and len(cover_batch) > i//2:
            filename, img = cover_batch[i//2]
            batch.append(img)
            batch_labels.append(0)  # Cover class
            batch_filenames.append(filename)
    
    # Convert to PyTorch tensors
    images = torch.empty((len(batch), 3, 256, 256), dtype=torch.float)
    for i in range(len(batch)):
        images[i] = torch.tensor(batch[i], dtype=torch.float).permute(2, 0, 1)
    
    # Move to device and run model
    image_tensor = images.to(device)
    batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        
    # Get predictions and probabilities
    prediction = outputs.data.max(1)[1]
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    stego_probabilities = probabilities[:, 1].detach().cpu().numpy()
    
    # Calculate accuracy
    accuracy = (
        prediction.eq(batch_labels_tensor.data).sum() * 100.0 / batch_labels_tensor.size(0)
    ).item()
    
    # Convert to numpy arrays
    batch_gt = batch_labels_tensor.detach().cpu().numpy()
    batch_pred = prediction.detach().cpu().numpy()
    
    # Log results
    batch_results = []
    for i in range(len(prediction)):
        label = "Stego" if prediction[i].item() == 1 else "Not-Stego"
        batch_results.append(f"{batch_filenames[i]} -> {label} | {stego_probabilities[i]:.4f}")
    
    logging.info("\n".join(batch_results))
    
    return batch_pred, batch_gt, stego_probabilities, accuracy, batch_filenames

def main():
    """Main function to execute YeNet testing."""
    
    # Define configuration
    TEST_BATCH_SIZE = 32
    COVER_PATH = str(Path("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/500bit/500bitdataset/test/original/*.*").expanduser())
    STEGO_PATH = str(Path("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/500bit/500bitdataset/test/stego/*.*").expanduser())
    CHKPT = str(Path("./checkpoints/net_50.pt").expanduser())
    
    # Load the model
    model, device = load_model(CHKPT)
    
    # Log configuration
    logging.info(f"Starting model testing with the following configuration:")
    logging.info(f"Device: {device}")
    logging.info(f"Cover test path: {COVER_PATH}")
    logging.info(f"Stego test path: {STEGO_PATH}")
    logging.info(f"Checkpoint file: {CHKPT}")
    logging.info(f"Test batch size: {TEST_BATCH_SIZE}")
    
    # Get image paths
    cover_image_names, stego_image_names = get_image_paths(COVER_PATH, STEGO_PATH)
    if not cover_image_names or not stego_image_names:
        return
    
    # Create labels
    cover_labels = np.zeros((len(cover_image_names)))
    stego_labels = np.ones((len(stego_image_names)))
    
    # Load and preprocess images
    cover_data = load_and_resize(cover_image_names)
    stego_data = load_and_resize(stego_image_names)
    
    # Initialize tracking variables
    all_predictions = []
    all_ground_truth = []
    all_stego_probabilities = []
    all_filenames = []
    test_accuracy = []
    
    # Process batches
    for idx in range(0, len(cover_data), TEST_BATCH_SIZE // 2):
        cover_batch = cover_data[idx : idx + TEST_BATCH_SIZE // 2]
        stego_batch = stego_data[idx : idx + TEST_BATCH_SIZE // 2]
        
        batch_pred, batch_gt, stego_probs, accuracy, batch_filenames = process_batch(
            model, device, cover_batch, stego_batch
        )
        
        # Accumulate results
        test_accuracy.append(accuracy)
        all_predictions.extend(batch_pred)
        all_ground_truth.extend(batch_gt)
        all_stego_probabilities.extend(stego_probs)
        all_filenames.extend(batch_filenames)
    
    # Convert to numpy arrays for calculations
    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)
    all_stego_probabilities = np.array(all_stego_probabilities)
    
    # Calculate per-class metrics
    correct_cover = 0
    correct_stego = 0
    total_cover = 0
    total_stego = 0
    
    for i, gt in enumerate(all_ground_truth):
        if gt == 0:
            total_cover += 1
            if all_predictions[i] == 0:
                correct_cover += 1
        else:
            total_stego += 1
            if all_predictions[i] == 1:
                correct_stego += 1
    
    # Calculate overall accuracy
    overall_accuracy = sum(test_accuracy) / len(test_accuracy) if test_accuracy else 0
    
    # Calculate accuracies and recalls
    cover_accuracy = (correct_cover / total_cover * 100) if total_cover > 0 else 0
    stego_accuracy = (correct_stego / total_stego * 100) if total_stego > 0 else 0
    cover_recall = correct_cover / total_cover if total_cover > 0 else 0
    stego_recall = correct_stego / total_stego if total_stego > 0 else 0
    balanced_accuracy = ((cover_recall + stego_recall) / 2) * 100
    
    # Calculate precision, recall, and F1 score
    # For Cover class (class 0)
    tp_cover = correct_cover
    fp_cover = total_stego - correct_stego
    fn_cover = total_cover - correct_cover
    tn_cover = correct_stego
    
    precision_cover = tp_cover / (tp_cover + fp_cover) if (tp_cover + fp_cover) > 0 else 0
    recall_cover = tp_cover / (tp_cover + fn_cover) if (tp_cover + fn_cover) > 0 else 0
    f1_cover = 2 * (precision_cover * recall_cover) / (precision_cover + recall_cover) if (precision_cover + recall_cover) > 0 else 0
    
    # For Stego class (class 1)
    tp_stego = correct_stego
    fp_stego = total_cover - correct_cover
    fn_stego = total_stego - correct_stego
    tn_stego = correct_cover
    
    precision_stego = tp_stego / (tp_stego + fp_stego) if (tp_stego + fp_stego) > 0 else 0
    recall_stego = tp_stego / (tp_stego + fn_stego) if (tp_stego + fn_stego) > 0 else 0
    f1_stego = 2 * (precision_stego * recall_stego) / (precision_stego + recall_stego) if (precision_stego + recall_stego) > 0 else 0
    
    # Calculate macro and weighted averages
    # Macro = simple average of metrics across classes
    precision_macro = (precision_cover + precision_stego) / 2
    recall_macro = (recall_cover + recall_stego) / 2
    f1_macro = (f1_cover + f1_stego) / 2
    
    # Weighted = average weighted by class support
    total_samples = total_cover + total_stego
    weight_cover = total_cover / total_samples
    weight_stego = total_stego / total_samples
    
    precision_weighted = (precision_cover * weight_cover + precision_stego * weight_stego)
    recall_weighted = (recall_cover * weight_cover + recall_stego * weight_stego)
    f1_weighted = (f1_cover * weight_cover + f1_stego * weight_stego)
    
    # Create dictionaries for easier plotting
    precision = {
        'cover': precision_cover, 
        'stego': precision_stego, 
        'macro': precision_macro, 
        'weighted': precision_weighted
    }
    
    recall = {
        'cover': recall_cover, 
        'stego': recall_stego, 
        'macro': recall_macro, 
        'weighted': recall_weighted
    }
    
    f1_score = {
        'cover': f1_cover, 
        'stego': f1_stego, 
        'macro': f1_macro, 
        'weighted': f1_weighted
    }
    
    # Also calculate using sklearn for verification
    from sklearn.metrics import precision_recall_fscore_support
    precision_vals, recall_vals, f1_vals, _ = precision_recall_fscore_support(
        all_ground_truth, all_predictions, average=None, labels=[0, 1]
    )
    precision_macro_sk, recall_macro_sk, f1_macro_sk, _ = precision_recall_fscore_support(
        all_ground_truth, all_predictions, average="macro"
    )
    precision_weighted_sk, recall_weighted_sk, f1_weighted_sk, _ = precision_recall_fscore_support(
        all_ground_truth, all_predictions, average="weighted"
    )
    
    # Calculate full metrics using our helper function
    metrics = calculate_metrics(all_ground_truth, all_stego_probabilities)
    
    return {
        'all_predictions': all_predictions,
        'all_ground_truth': all_ground_truth,
        'all_stego_probabilities': all_stego_probabilities,
        'test_accuracy': test_accuracy,
        'overall_accuracy': overall_accuracy,
        'cover_accuracy': cover_accuracy,
        'stego_accuracy': stego_accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'metrics': metrics,
        'cm': np.array([
            [correct_cover, total_cover - correct_cover],
            [total_stego - correct_stego, correct_stego]
        ])
    }
def log_results(results, config):
    """
    Log test results to console and log file.
    
    Args:
        results (dict): Dictionary containing test results
        config (dict): Dictionary containing test configuration
    """
    # Extract variables for easier access
    test_accuracy = results['test_accuracy']
    overall_accuracy = results['overall_accuracy']
    cover_accuracy = results['cover_accuracy']
    stego_accuracy = results['stego_accuracy']
    balanced_accuracy = results['balanced_accuracy']
    precision = results['precision']
    recall = results['recall']
    f1_score = results['f1_score']
    cm = results['cm']
    metrics = results['metrics']
    
    # Log summary information
    logging.info(f"\nTest Summary:")
    logging.info(f"Processed {len(test_accuracy)} batches")
    logging.info(f"From Cover Path: {config['cover_path']}")
    logging.info(f"From Stego Path: {config['stego_path']}")
    logging.info(f"Device used: {config['device']}")
    
    # Log overall results
    logging.info(f"Overall Accuracy: {overall_accuracy:.2f}%")
    logging.info(f"Cover Class Accuracy: {cover_accuracy:.2f}% ({cm[0][0]}/{cm[0][0] + cm[0][1]})")
    logging.info(f"Stego Class Accuracy: {stego_accuracy:.2f}% ({cm[1][1]}/{cm[1][0] + cm[1][1]})")
    logging.info(f"Balanced Accuracy: {balanced_accuracy:.2f}%")
    
    # Log precision, recall, and F1 score
    logging.info("\nPrecision:")
    logging.info(f"  Cover Class: {precision['cover']:.4f}")
    logging.info(f"  Stego Class: {precision['stego']:.4f}")
    logging.info(f"  Macro Average: {precision['macro']:.4f}")
    logging.info(f"  Weighted Average: {precision['weighted']:.4f}")
    
    logging.info("\nRecall:")
    logging.info(f"  Cover Class: {recall['cover']:.4f}")
    logging.info(f"  Stego Class: {recall['stego']:.4f}")
    logging.info(f"  Macro Average: {recall['macro']:.4f}")
    logging.info(f"  Weighted Average: {recall['weighted']:.4f}")
    
    logging.info("\nF1 Score:")
    logging.info(f"  Cover Class: {f1_score['cover']:.4f}")
    logging.info(f"  Stego Class: {f1_score['stego']:.4f}")
    logging.info(f"  Macro Average: {f1_score['macro']:.4f}")
    logging.info(f"  Weighted Average: {f1_score['weighted']:.4f}")
    
    # Print to console for convenience
    print(f"\nLoaded {len(test_accuracy)} batches")
    print(f"From Cover Path: {config['cover_path']}")
    print(f"From Stego Path: {config['stego_path']}")
    print(f"\nFinal Accuracy Summary:")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Cover Class Accuracy: {cover_accuracy:.2f}%")
    print(f"Stego Class Accuracy: {stego_accuracy:.2f}%")
    print(f"Balanced Accuracy: {balanced_accuracy:.2f}%")
    
    print("\nPrecision:")
    print(f"  Cover Class: {precision['cover']:.4f}")
    print(f"  Stego Class: {precision['stego']:.4f}")
    print(f"  Macro Average: {precision['macro']:.4f}")
    print(f"  Weighted Average: {precision['weighted']:.4f}")
    
    print("\nRecall:")
    print(f"  Cover Class: {recall['cover']:.4f}")
    print(f"  Stego Class: {recall['stego']:.4f}")
    print(f"  Macro Average: {recall['macro']:.4f}")
    print(f"  Weighted Average: {recall['weighted']:.4f}")
    
    print("\nF1 Score:")
    print(f"  Cover Class: {f1_score['cover']:.4f}")
    print(f"  Stego Class: {f1_score['stego']:.4f}")
    print(f"  Macro Average: {f1_score['macro']:.4f}")
    print(f"  Weighted Average: {f1_score['weighted']:.4f}")
    
    # Log additional metrics
    logging.info("\nAUC and DET Metrics:")
    logging.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    logging.info(f"  Equal Error Rate (EER): {metrics['eer']:.4f}")
    if metrics['fnr_at_fpr01'] is not None:
        logging.info(f"  FNR at FPR=1%: {metrics['fnr_at_fpr01']:.4f}")
    if metrics['fnr_at_fpr05'] is not None:
        logging.info(f"  FNR at FPR=5%: {metrics['fnr_at_fpr05']:.4f}")
    if metrics['fnr_at_fpr10'] is not None:
        logging.info(f"  FNR at FPR=10%: {metrics['fnr_at_fpr10']:.4f}")
    logging.info(f"  Average Precision: {metrics['average_precision']:.4f}")
    
    # Print additional metrics to console
    print("\nAUC and DET Metrics:")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  Equal Error Rate (EER): {metrics['eer']:.4f}")
    if metrics['fnr_at_fpr01'] is not None:
        print(f"  FNR at FPR=1%: {metrics['fnr_at_fpr01']:.4f}")
    if metrics['fnr_at_fpr05'] is not None:
        print(f"  FNR at FPR=5%: {metrics['fnr_at_fpr05']:.4f}")
    if metrics['fnr_at_fpr10'] is not None:
        print(f"  FNR at FPR=10%: {metrics['fnr_at_fpr10']:.4f}")
    print(f"  Average Precision: {metrics['average_precision']:.4f}")
    
    # Also log the confusion matrix values
    logging.info(f"\nConfusion Matrix:\n{cm}")
    logging.info(f"True Cover, Predicted Cover: {cm[0][0]}")
    logging.info(f"True Cover, Predicted Stego: {cm[0][1]}")
    logging.info(f"True Stego, Predicted Cover: {cm[1][0]}")
    logging.info(f"True Stego, Predicted Stego: {cm[1][1]}")
    
    # Log all parameters used for this test run
    logging.info("\nTest Configuration:")
    logging.info(f"  Model: YeNet")
    logging.info(f"  Checkpoint: {config['checkpoint']}")
    logging.info(f"  Batch Size: {config['batch_size']}")
    logging.info(f"  Device: {config['device']}")
    logging.info(f"  Cover Images Path: {config['cover_path']}")
    logging.info(f"  Stego Images Path: {config['stego_path']}")


def generate_plots(results, timestamp):
    """
    Generate and save visualization plots.
    
    Args:
        results (dict): Dictionary containing test results
        timestamp (str): Timestamp for filename
    """
    # Extract variables for easier access
    all_ground_truth = results['all_ground_truth']
    all_stego_probabilities = results['all_stego_probabilities']
    metrics = results['metrics']
    cm = results['cm']
    
    # Create ROC curve plot
    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'], 
                   plots_dir / f"roc_curve_{timestamp}.png")
    
    # Create DET curve plot
    plot_det_curve(metrics['fpr'], metrics['fnr'], 
                   plots_dir / f"det_curve_{timestamp}.png")
    
    # Create precision-recall curve plot
    plot_pr_curve(metrics['precision_curve'], metrics['recall_curve'], metrics['average_precision'], 
                 plots_dir / f"pr_curve_{timestamp}.png")
    
    # Create confusion matrix plot
    plot_confusion_matrix(all_ground_truth, metrics['y_pred'], 
                         plots_dir / f"confusion_matrix_{timestamp}.png")
    
    # Create accuracy by class plot
    plot_accuracy_by_class(results['cover_accuracy'], results['stego_accuracy'], 
                          results['overall_accuracy'], timestamp, 
                          results['balanced_accuracy'])
    
    # Create precision, recall, F1 plot
    plot_precision_recall_f1(results['precision'], results['recall'], 
                            results['f1_score'], timestamp)
    
    # Save metrics for future reference
    np.savez_compressed(logs_dir / f"metrics_{timestamp}.npz",
                       fpr=metrics['fpr'],
                       tpr=metrics['tpr'],
                       fnr=metrics['fnr'],
                       precision=metrics['precision_curve'],
                       recall=metrics['recall_curve'],
                       roc_auc=metrics['roc_auc'],
                       average_precision=metrics['average_precision'],
                       accuracy=metrics['accuracy'],
                       precision_val=metrics['precision'],
                       recall_val=metrics['recall'],
                       f1_score=metrics['f1_score'],
                       confusion_matrix=cm,
                       eer=metrics['eer'])
    
    # Save predictions for future analysis
    np.savez_compressed(logs_dir / f"predictions_{timestamp}.npz", 
                       predictions=results['all_predictions'], 
                       ground_truth=all_ground_truth, 
                       stego_probabilities=all_stego_probabilities)


if __name__ == '__main__':
    # Define configuration
    config = {
        'batch_size': 32,
        'cover_path': str(Path("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/100bit/100bitdataset/test/original/*.*").expanduser()),
        'stego_path': str(Path("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/100bit/100bitdataset/test/stego/*.*").expanduser()),
        'checkpoint': str(Path("./checkpoints/net_50.pt").expanduser())
    }
    
    # Run test
    results = main()
    
    if results:
        # Get device from config
        model, device = load_model(config['checkpoint'])
        config['device'] = str(device)
        
        # Log results
        log_results(results, config)
        
        # Generate plots
        generate_plots(results, timestamp)
        
        logging.info("Testing completed successfully")
    else:
        logging.error("Testing failed. Check logs for details.")
        print("Testing failed. Check logs for details.")
