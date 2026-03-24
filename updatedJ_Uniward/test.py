"""This module is used to test the trained J-UNIWARD model with labeled output per image."""

import os
import sys
import time
import logging
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from PIL import Image
import datetime

import torch
from torch import nn
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

sys.path.append('./updatedJ_Uniward')

from model.model_juniwarden import JUniwarden
from utils.utils import latest_checkpoint, calculate_confusion_matrix

# Setup directories for logs and plots
logs_dir = Path('./logs')
plots_dir = Path('./plots')
results_dir = Path('./test_results')
logs_dir.mkdir(exist_ok=True)
plots_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

# Configure timestamp for this test run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"test_{timestamp}.log"

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test configuration
TEST_BATCH_SIZE = 32
COVER_PATH = str(Path("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/100bit/100bitdataset/test/original/*.*").expanduser())
STEGO_PATH = str(Path("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/100bit/100bitdataset/test/stego/*.*").expanduser())
CHKPT = str(Path("./checkpoints/net_50.pt").expanduser())
QUALITY_FACTOR = 75
DCT_DIRECT_ANALYSIS = True

# Command-line arguments for manual configuration
def parse_args():
    parser = argparse.ArgumentParser(description='Test J-UNIWARD steganalysis model')
    parser.add_argument('--cover_path', type=str, default=COVER_PATH,
                        help='Path to cover test images (glob pattern)')
    parser.add_argument('--stego_path', type=str, default=STEGO_PATH,
                        help='Path to stego test images (glob pattern)')
    parser.add_argument('--checkpoint', type=str, default=CHKPT,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=TEST_BATCH_SIZE,
                        help='Test batch size')
    parser.add_argument('--quality_factor', type=int, default=QUALITY_FACTOR,
                        help='JPEG quality factor')
    parser.add_argument('--dct_direct_analysis', action='store_true', default=DCT_DIRECT_ANALYSIS,
                        help='Use direct DCT coefficient analysis')
    return parser.parse_args()
    """Evaluate the model and generate comprehensive metrics and visualizations.
    
    Args:
        model: The trained model
        test_loader: Test data loader
        device: Device to run inference on
        output_dir: Directory to save outputs
        
    Returns:
        metrics: Dictionary of computed metrics
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get predictions
    y_true, y_pred, y_scores = get_prediction_scores(model, test_loader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Generate plots
    roc_auc = plot_roc_curve(y_true, y_scores, output_dir)
    avg_precision = plot_precision_recall_curve(y_true, y_scores, output_dir)
    cm = plot_confusion_matrix(y_true, y_pred, output_dir)
    
    # Compute TP, FP, TN, FN from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    
    # Compile metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
    }
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Average Precision: {avg_precision:.4f}\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"  True Negative: {tn}\n")
        f.write(f"  False Positive: {fp}\n")
        f.write(f"  False Negative: {fn}\n")
        f.write(f"  True Positive: {tp}\n")
    
    return metrics

def load_and_resize(image_paths):
    """Load and resize images from paths.
    
    Args:
        image_paths: List of image paths
        
    Returns:
        List of tuples (path, image_array)
    """
    resized = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((512, 512), Image.LANCZOS)
        img_array = np.array(img)
        resized.append((p, img_array))
    return resized

def main():
    # Log test configuration
    logging.info(f"Starting model testing with the following configuration:")
    logging.info(f"Device: {device}")
    logging.info(f"Cover test path: {COVER_PATH}")
    logging.info(f"Stego test path: {STEGO_PATH}")
    logging.info(f"Checkpoint file: {CHKPT}")
    logging.info(f"Test batch size: {TEST_BATCH_SIZE}")
    logging.info(f"JPEG Quality Factor: {QUALITY_FACTOR}")
    logging.info(f"Direct DCT analysis: {DCT_DIRECT_ANALYSIS}")
    
    # Track metrics
    all_predictions = []
    all_ground_truth = []
    all_stego_probabilities = []
    correct_cover = 0
    correct_stego = 0
    total_cover = 0
    total_stego = 0
    
    # Get image paths
    cover_image_names = glob(COVER_PATH)
    stego_image_names = glob(STEGO_PATH)
    
    # Set up labels
    cover_labels = np.zeros((len(cover_image_names)))
    stego_labels = np.ones((len(stego_image_names)))
    
    # Initialize model
    model = JUniwarden(quality_factor=QUALITY_FACTOR, direct_dct=DCT_DIRECT_ANALYSIS)
    model.to(device)
    
    # Load checkpoint
    ckpt = torch.load(CHKPT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logging.info(f"Model loaded from checkpoint")
    
    # Preprocess images
    logging.info(f"Found {len(cover_image_names)} cover images and {len(stego_image_names)} stego images")
    cover_data = load_and_resize(cover_image_names)
    stego_data = load_and_resize(stego_image_names)
    
    # Track per-image results
    image_results = []
    test_accuracy = []
    
    # Process batches
    model.eval()
    with torch.no_grad():
        for idx in range(0, min(len(cover_data), len(stego_data)), TEST_BATCH_SIZE // 2):
            cover_batch = cover_data[idx : idx + TEST_BATCH_SIZE // 2]
            stego_batch = stego_data[idx : idx + TEST_BATCH_SIZE // 2]
            batch = []
            batch_labels = []
            batch_filenames = []
            
            xi = 0
            yi = 0
            for i in range(2 * len(cover_batch)):
                if i % 2 == 0:
                    filename, img = stego_batch[xi]
                    batch.append(img)
                    batch_labels.append(1)
                    batch_filenames.append(filename)
                    xi += 1
                else:
                    filename, img = cover_batch[yi]
                    batch.append(img)
                    batch_labels.append(0)
                    batch_filenames.append(filename)
                    yi += 1
            
            # Convert to tensors
            batch = np.array(batch).transpose(0, 3, 1, 2) / 255.0
            batch = torch.from_numpy(batch).to(device, dtype=torch.float)
            batch_labels = torch.tensor(batch_labels).to(device, dtype=torch.long)
            
            # Forward pass
            outputs = model(batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate accuracy for this batch
            correct = (predictions == batch_labels).sum().item()
            accuracy = 100 * correct / len(batch_labels)
            test_accuracy.append(accuracy)
            
            # Record predictions and calculate per-class accuracy
            for i, (pred, label, prob, filename) in enumerate(zip(predictions, batch_labels, probabilities, batch_filenames)):
                pred = pred.item()
                label = label.item()
                stego_prob = prob[1].item()
                
                all_predictions.append(pred)
                all_ground_truth.append(label)
                all_stego_probabilities.append(stego_prob)
                
                # Track per-class accuracy
                if label == 0:  # Cover
                    total_cover += 1
                    if pred == 0:
                        correct_cover += 1
                else:  # Stego
                    total_stego += 1
                    if pred == 1:
                        correct_stego += 1
                
                # Store per-image results
                image_results.append({
                    'filename': os.path.basename(filename),
                    'true_label': 'stego' if label == 1 else 'cover',
                    'predicted': 'stego' if pred == 1 else 'cover',
                    'stego_probability': stego_prob
                })
    
    # Calculate overall metrics
    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)
    all_stego_probabilities = np.array(all_stego_probabilities)
    
    # Overall accuracy
    accuracy = accuracy_score(all_ground_truth, all_predictions) * 100
    
    # Per-class accuracy
    cover_accuracy = 100 * correct_cover / total_cover if total_cover > 0 else 0
    stego_accuracy = 100 * correct_stego / total_stego if total_stego > 0 else 0
    
    # Other metrics
    precision = precision_score(all_ground_truth, all_predictions)
    recall = recall_score(all_ground_truth, all_predictions)
    f1 = f1_score(all_ground_truth, all_predictions)
    
    # Confusion matrix
    cm = confusion_matrix(all_ground_truth, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_ground_truth, all_stego_probabilities)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(all_ground_truth, all_stego_probabilities)
    avg_precision = average_precision_score(all_ground_truth, all_stego_probabilities)
    
    # Save detailed per-image results
    with open(str(results_dir / 'per_image_results.txt'), 'w') as f:
        f.write("Filename,True Label,Predicted,Stego Probability\n")
        for result in image_results:
            f.write(f"{result['filename']},{result['true_label']},{result['predicted']},{result['stego_probability']:.6f}\n")
    
    # Log summary results
    logging.info(f"\n===== Test Results Summary =====")
    logging.info(f"Overall Accuracy: {accuracy:.2f}%")
    logging.info(f"Cover Accuracy: {cover_accuracy:.2f}% ({correct_cover}/{total_cover})")
    logging.info(f"Stego Accuracy: {stego_accuracy:.2f}% ({correct_stego}/{total_stego})")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"ROC AUC: {roc_auc:.4f}")
    logging.info(f"Average Precision: {avg_precision:.4f}")
    logging.info(f"Confusion Matrix: \n  TN: {tn}, FP: {fp}\n  FN: {fn}, TP: {tp}")
    
    # Generate and save plots
    output_dir = str(results_dir)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'roc_curve.pdf'))
    plt.close()
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.pdf'))
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar()
    classes = ['Cover', 'Stego']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12)
    plt.tight_layout()
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.pdf'))
    plt.close()
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"Cover Accuracy: {cover_accuracy:.2f}% ({correct_cover}/{total_cover})\n")
        f.write(f"Stego Accuracy: {stego_accuracy:.2f}% ({correct_stego}/{total_stego})\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Average Precision: {avg_precision:.4f}\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"  True Negative: {tn}\n")
        f.write(f"  False Positive: {fp}\n")
        f.write(f"  False Negative: {fn}\n")
        f.write(f"  True Positive: {tp}\n")
    
    logging.info(f"Test results saved to {results_dir}")
    
if __name__ == "__main__":
    args = parse_args()
    
    # Override defaults with command-line arguments if provided
    if args.cover_path != COVER_PATH:
        COVER_PATH = args.cover_path
    if args.stego_path != STEGO_PATH:
        STEGO_PATH = args.stego_path
    if args.checkpoint != CHKPT:
        CHKPT = args.checkpoint
    if args.batch_size != TEST_BATCH_SIZE:
        TEST_BATCH_SIZE = args.batch_size
    if args.quality_factor != QUALITY_FACTOR:
        QUALITY_FACTOR = args.quality_factor
    if args.dct_direct_analysis != DCT_DIRECT_ANALYSIS:
        DCT_DIRECT_ANALYSIS = args.dct_direct_analysis
    
    main()
