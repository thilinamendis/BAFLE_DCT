# filepath: /home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/Updated_SrNEt/test.py
"""This module is used to test the Srnet model with labeled output per image."""
from glob import glob
import torch
import numpy as np
from model.model import Srnet
from pathlib import Path
from PIL import Image
import logging
import datetime
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os

# Setup directories for logs and plots
logs_dir = Path('./logs')
plots_dir = Path('./plots')
logs_dir.mkdir(exist_ok=True)
plots_dir.mkdir(exist_ok=True)

# Configure timestamp for this test run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"test_{timestamp}.log"

def plot_confusion_matrix(cm, class_names, timestamp):
    """
    Generate a confusion matrix visualization
    
    Args:
        cm (array): Confusion matrix array [TP, FP, FN, TN]
        class_names (list): List of class names
        timestamp (str): Timestamp for filename
    """
    plt.figure(figsize=(10, 8))
    
    # Create the confusion matrix plot with a darker colormap
    plt.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=np.max(cm))
    plt.title('Confusion Matrix', fontsize=18, fontweight='bold')
    plt.colorbar()
    
    # Add tick marks and class names
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)
    
    # Add the values in each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'./plots/test_confusion_matrix_{timestamp}.png', dpi=300)
    plt.savefig(f'./plots/test_confusion_matrix_{timestamp}.pdf')  # PDF for publication quality
    plt.close()
    
    logging.info(f"Confusion matrix saved as test_confusion_matrix_{timestamp}.png")

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
    plt.savefig(f'./plots/test_accuracy_by_class_{timestamp}.png', dpi=300)
    plt.savefig(f'./plots/test_accuracy_by_class_{timestamp}.pdf')  # PDF for publication quality
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
            plt.text(pos, val + 0.02, f'{val:.3f}', ha='center', va='bottom', 
                     fontsize=9, color=color, rotation=0, fontweight='bold')
    
    add_labels(r1, cover_metrics, '#3498db')
    add_labels(r2, stego_metrics, '#2ecc71')
    add_labels(r3, macro_metrics, '#f39c12')
    add_labels(r4, weighted_metrics, '#9b59b6')
    
    # Save plots
    plt.tight_layout(pad=3)
    plt.savefig(f'./plots/test_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'./plots/test_metrics_{timestamp}.pdf', bbox_inches='tight')  # PDF for publication quality
    plt.close()
    
    logging.info(f"Precision, Recall, F1 plot saved as test_metrics_{timestamp}.png")

def plot_roc_curve(y_true, y_score, timestamp):
    """
    Create a ROC curve plot
    
    Args:
        y_true (array): Ground truth labels
        y_score (array): Predicted probabilities for the positive class
        timestamp (str): Timestamp for filename
    
    Returns:
        float: Area Under Curve (AUC) score
    """
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    # Add labels and title
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    
    # Add grid
    plt.grid(linestyle='--', alpha=0.6)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'./plots/roc_curve_{timestamp}.png', dpi=300)
    plt.savefig(f'./plots/roc_curve_{timestamp}.pdf')  # PDF for publication quality
    plt.close()
    
    logging.info(f"ROC curve saved as roc_curve_{timestamp}.png")
    return roc_auc

def plot_det_curve(y_true, y_score, timestamp):
    """
    Create a Detection Error Tradeoff (DET) curve plot
    
    Args:
        y_true (array): Ground truth labels
        y_score (array): Predicted probabilities for the positive class
        timestamp (str): Timestamp for filename
    
    Returns:
        tuple: (FPR at specific operating points, FNR at specific operating points)
    """
    # Calculate FPR and TPR for ROC curve
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # Calculate False Negative Rate (1 - TPR)
        fnr = 1 - tpr
        
        # Apply inverse of normal CDF to get DET coordinates
        ppndf = lambda p: stats.norm.ppf(p)
        
        # Add small epsilon to prevent infinity issues
        epsilon = 1e-8
        fpr_safe = np.clip(fpr, epsilon, 1.0 - epsilon)
        fnr_safe = np.clip(fnr, epsilon, 1.0 - epsilon)
        
        # Convert to DET scale
        x_det = ppndf(fpr_safe)
        y_det = ppndf(fnr_safe)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.plot(x_det, y_det, color='crimson', lw=2)
        
        # Create custom x and y ticks for DET curve
        det_ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99]
        tick_labels = [f"{tick*100:.1f}" if tick < 0.01 else f"{tick*100:.0f}" for tick in det_ticks]
        
        # Create tick locations but handle extreme values
        tick_locations = []
        for tick in det_ticks:
            if epsilon < tick < 1.0 - epsilon:
                tick_locations.append(ppndf(tick))
        
        # Only set ticks if we have valid locations
        if tick_locations:
            plt.xticks(tick_locations, tick_labels[:len(tick_locations)])
            plt.yticks(tick_locations, tick_labels[:len(tick_locations)])
        
        # Create grid at the tick positions
        plt.grid(True, alpha=0.3)
        
        # Add labels and title
        plt.xlabel('False Positive Rate (%)', fontsize=12)
        plt.ylabel('False Negative Rate (%)', fontsize=12)
        plt.title('Detection Error Tradeoff (DET) Curve', fontsize=16, fontweight='bold')
        
        # Find Equal Error Rate (EER)
        # This is where FPR = FNR
        abs_diff = np.abs(fpr - fnr)
        idx = np.argmin(abs_diff)
        eer = (fpr[idx] + fnr[idx]) / 2
        
        # Plot EER point if it's within our limits
        if epsilon < eer < 1.0 - epsilon:
            eer_x = ppndf(eer)
            eer_y = eer_x  # On DET curve, EER is on the diagonal
            plt.plot(eer_x, eer_y, 'ro', markersize=8, label=f'EER = {eer:.4f}')
            plt.legend(loc='upper right', fontsize=10)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f'./plots/det_curve_{timestamp}.png', dpi=300)
        plt.savefig(f'./plots/det_curve_{timestamp}.pdf')  # PDF for publication quality
        plt.close()
        
        logging.info(f"DET curve saved as det_curve_{timestamp}.png")
        
        # Return operating points for logging
        # Find points at common operating thresholds
        fnr_at_fpr01 = np.interp(0.01, fpr, fnr) if 0.01 <= max(fpr) else None
        fnr_at_fpr05 = np.interp(0.05, fpr, fnr) if 0.05 <= max(fpr) else None
        fnr_at_fpr10 = np.interp(0.1, fpr, fnr) if 0.1 <= max(fpr) else None
        
        return eer, fnr_at_fpr01, fnr_at_fpr05, fnr_at_fpr10
    except Exception as e:
        logging.error(f"Error generating DET curve: {str(e)}")
        return None, None, None, None
    
    # Find Equal Error Rate (EER)
    # This is where FPR = FNR
    abs_diff = np.abs(fpr - fnr)
    idx = np.argmin(abs_diff)
    eer = (fpr[idx] + fnr[idx]) / 2
    
    plt.plot(ppndf(eer), ppndf(eer), 'ro', label=f'EER = {eer:.4f}')
    plt.legend(loc='upper right')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'./plots/det_curve_{timestamp}.png', dpi=300)
    plt.savefig(f'./plots/det_curve_{timestamp}.pdf')  # PDF for publication quality
    plt.close()
    
    logging.info(f"DET curve saved as det_curve_{timestamp}.png")
    
    # Return operating points for logging
    # Find points at common operating thresholds
    fnr_at_fpr01 = np.interp(0.01, fpr, fnr) if 0.01 <= max(fpr) else None
    fnr_at_fpr05 = np.interp(0.05, fpr, fnr) if 0.05 <= max(fpr) else None
    fnr_at_fpr10 = np.interp(0.1, fpr, fnr) if 0.1 <= max(fpr) else None
    
    return eer, fnr_at_fpr01, fnr_at_fpr05, fnr_at_fpr10

def plot_precision_recall_curve(y_true, y_score, timestamp):
    """
    Create a Precision-Recall curve plot
    
    Args:
        y_true (array): Ground truth labels
        y_score (array): Predicted probabilities for the positive class
        timestamp (str): Timestamp for filename
        
    Returns:
        float: Average precision score
    """
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, label=f'AP = {avg_precision:.4f}')
    
    # Add baseline for random classifier
    baseline = sum(y_true) / len(y_true)
    plt.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--', label=f'Random (AP = {baseline:.4f})')
    
    # Add labels and title
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    
    # Add grid and set axis limits
    plt.grid(linestyle='--', alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'./plots/pr_curve_{timestamp}.png', dpi=300)
    plt.savefig(f'./plots/pr_curve_{timestamp}.pdf')  # PDF for publication quality
    plt.close()
    
    logging.info(f"Precision-Recall curve saved as pr_curve_{timestamp}.png")
    return avg_precision

# Setup logging to both file and console
logging.basicConfig(
    filename=str(logs_dir / log_filename),
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Add console handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_BATCH_SIZE = 32  # Changed from 40 to match train/validation batch size
COVER_PATH = str(Path("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/400bit/400bitdataset/test/original/*.*").expanduser())
STEGO_PATH = str(Path("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/400bit/400bitdataset/test/stego/*.*").expanduser())
CHKPT = str(Path("./checkpoints/net_50.pt").expanduser())

# Log test setup information
logging.info(f"Starting model testing with the following configuration:")
logging.info(f"Device: {device}")
logging.info(f"Cover test path: {COVER_PATH}")
logging.info(f"Stego test path: {STEGO_PATH}")
logging.info(f"Checkpoint file: {CHKPT}")
logging.info(f"Test batch size: {TEST_BATCH_SIZE}")

# Initialize tracking variables for results
all_predictions = []
all_ground_truth = []
all_stego_probabilities = []  # For ROC, DET curves
correct_cover = 0
correct_stego = 0
total_cover = 0
total_stego = 0

cover_image_names = glob(COVER_PATH)
stego_image_names = glob(STEGO_PATH)

cover_labels = np.zeros((len(cover_image_names)))
stego_labels = np.ones((len(stego_image_names)))

model = Srnet().to(device)

ckpt = torch.load(CHKPT, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])

test_accuracy = []

# Load and resize RGB images to 256x256, return (filename, image array)
def load_and_resize(image_paths):
    resized = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((256, 256), Image.LANCZOS)
        img_array = np.array(img)
        resized.append((p, img_array))  # store (filename, image array)
    return resized

logging.info(f"Found {len(cover_image_names)} cover images and {len(stego_image_names)} stego images")

# Load and process data
cover_data = load_and_resize(cover_image_names)
stego_data = load_and_resize(stego_image_names)

logging.info("Starting evaluation...")

# Setup batch processing with detailed logging
for idx in range(0, len(cover_data), TEST_BATCH_SIZE // 2):
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

    # Allocate image tensor
    images = torch.empty((len(batch), 3, 256, 256), dtype=torch.float)
    for i in range(len(batch)):
        images[i] = torch.tensor(batch[i]).permute(2, 0, 1).to(device)

    image_tensor = images.to(device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

    outputs = model(image_tensor)
    
    # Get predicted class (argmax)
    prediction = outputs.data.max(1)[1]
    
    # Get probabilities (confidence scores) using softmax
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    stego_probabilities = probabilities[:, 1].detach().cpu().numpy()  # probability of class 1 (stego)

    # Calculate batch accuracy
    accuracy = (
        prediction.eq(batch_labels.data).sum()
        * 100.0
        / batch_labels.size(0)
    )
    test_accuracy.append(accuracy.item())

    # Log batch results
    logging.info(f"Batch {len(test_accuracy)}: Accuracy = {accuracy.item():.2f}%")

    # Track class-specific metrics
    batch_gt = batch_labels.detach().cpu().numpy()
    batch_pred = prediction.detach().cpu().numpy()
    
    # Add to overall tracking
    all_predictions.extend(batch_pred)
    all_ground_truth.extend(batch_gt)
    all_stego_probabilities = [] if 'all_stego_probabilities' not in locals() else all_stego_probabilities
    all_stego_probabilities.extend(stego_probabilities)

    # Print detailed results for each image
    batch_results = []
    for i in range(len(prediction)):
        pred_label = prediction[i].item()
        true_label = batch_labels[i].item()
        correct = pred_label == true_label
        
        # Update class-specific metrics
        if true_label == 0:  # Cover image
            total_cover += 1
            if correct:
                correct_cover += 1
        else:  # Stego image
            total_stego += 1
            if correct:
                correct_stego += 1
        
        # Format prediction result
        result_str = "Stego" if pred_label == 1 else "Not-Stego"
        status = "✓" if correct else "✗"
        file_basename = os.path.basename(batch_filenames[i])
        
        result_log = f"{file_basename} -> {result_str} {status}"
        print(result_log)
        batch_results.append(result_log)
    
    # Log detailed results to file
    logging.info("\n".join(batch_results))

# Calculate overall metrics
overall_accuracy = sum(test_accuracy) / len(test_accuracy) if test_accuracy else 0
cover_accuracy = (correct_cover / total_cover * 100) if total_cover > 0 else 0
stego_accuracy = (correct_stego / total_stego * 100) if total_stego > 0 else 0

# Calculate balanced accuracy (average of recall for each class)
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

# Log summary information
logging.info(f"\nTest Summary:")
logging.info(f"Processed {len(test_accuracy)} batches")
logging.info(f"From Cover Path: {COVER_PATH}")
logging.info(f"From Stego Path: {STEGO_PATH}")
logging.info(f"Device used: {device}")

if test_accuracy:
    # Log overall results
    logging.info(f"Overall Accuracy: {overall_accuracy:.2f}%")
    logging.info(f"Cover Class Accuracy: {cover_accuracy:.2f}% ({correct_cover}/{total_cover})")
    logging.info(f"Stego Class Accuracy: {stego_accuracy:.2f}% ({correct_stego}/{total_stego})")
    logging.info(f"Balanced Accuracy: {balanced_accuracy:.2f}%")
    
    # Log precision, recall, and F1 score
    logging.info("\nPrecision:")
    logging.info(f"  Cover Class: {precision_cover:.4f}")
    logging.info(f"  Stego Class: {precision_stego:.4f}")
    logging.info(f"  Macro Average: {precision_macro:.4f}")
    logging.info(f"  Weighted Average: {precision_weighted:.4f}")
    
    logging.info("\nRecall:")
    logging.info(f"  Cover Class: {recall_cover:.4f}")
    logging.info(f"  Stego Class: {recall_stego:.4f}")
    logging.info(f"  Macro Average: {recall_macro:.4f}")
    logging.info(f"  Weighted Average: {recall_weighted:.4f}")
    
    logging.info("\nF1 Score:")
    logging.info(f"  Cover Class: {f1_cover:.4f}")
    logging.info(f"  Stego Class: {f1_stego:.4f}")
    logging.info(f"  Macro Average: {f1_macro:.4f}")
    logging.info(f"  Weighted Average: {f1_weighted:.4f}")
    
    # Print to console
    print(f"\nLoaded {len(test_accuracy)} batches")
    print(f"From Cover Path: {COVER_PATH}")
    print(f"From Stego Path: {STEGO_PATH}")
    print(f"\nFinal Accuracy Summary:")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Cover Class Accuracy: {cover_accuracy:.2f}%")
    print(f"Stego Class Accuracy: {stego_accuracy:.2f}%")
    print(f"Balanced Accuracy: {balanced_accuracy:.2f}%")
    
    print("\nPrecision:")
    print(f"  Cover Class: {precision_cover:.4f}")
    print(f"  Stego Class: {precision_stego:.4f}")
    print(f"  Macro Average: {precision_macro:.4f}")
    print(f"  Weighted Average: {precision_weighted:.4f}")
    
    print("\nRecall:")
    print(f"  Cover Class: {recall_cover:.4f}")
    print(f"  Stego Class: {recall_stego:.4f}")
    print(f"  Macro Average: {recall_macro:.4f}")
    print(f"  Weighted Average: {recall_weighted:.4f}")
    
    print("\nF1 Score:")
    print(f"  Cover Class: {f1_cover:.4f}")
    print(f"  Stego Class: {f1_stego:.4f}")
    print(f"  Macro Average: {f1_macro:.4f}")
    print(f"  Weighted Average: {f1_weighted:.4f}")
    
    # Create confusion matrix
    import numpy as np
    cm = np.array([
        [correct_cover, total_cover - correct_cover],
        [total_stego - correct_stego, correct_stego]
    ])
    
    # Calculate ROC AUC and plot ROC curve
    roc_auc = plot_roc_curve(all_ground_truth, all_stego_probabilities, timestamp)
    
    # Calculate and plot DET curve
    eer, fnr_at_fpr01, fnr_at_fpr05, fnr_at_fpr10 = plot_det_curve(all_ground_truth, all_stego_probabilities, timestamp)
    
    # Calculate and plot Precision-Recall curve
    avg_precision = plot_precision_recall_curve(all_ground_truth, all_stego_probabilities, timestamp)
    
    # Log ROC AUC and DET metrics
    logging.info("\nAUC and DET Metrics:")
    logging.info(f"  ROC AUC: {roc_auc:.4f}")
    logging.info(f"  Equal Error Rate (EER): {eer:.4f}")
    logging.info(f"  FNR at FPR=1%: {fnr_at_fpr01:.4f}")
    logging.info(f"  FNR at FPR=5%: {fnr_at_fpr05:.4f}")
    logging.info(f"  FNR at FPR=10%: {fnr_at_fpr10:.4f}")
    logging.info(f"  Average Precision: {avg_precision:.4f}")
    
    # Print ROC AUC and DET metrics to console
    print("\nAUC and DET Metrics:")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Equal Error Rate (EER): {eer:.4f}")
    print(f"  FNR at FPR=1%: {fnr_at_fpr01:.4f}")
    print(f"  FNR at FPR=5%: {fnr_at_fpr05:.4f}")
    print(f"  FNR at FPR=10%: {fnr_at_fpr10:.4f}")
    print(f"  Average Precision: {avg_precision:.4f}")
    
    # Generate visualizations
    plot_confusion_matrix(cm, ["Not-Stego", "Stego"], timestamp)
    plot_accuracy_by_class(cover_accuracy, stego_accuracy, overall_accuracy, timestamp, balanced_accuracy)
    plot_precision_recall_f1(precision, recall, f1_score, timestamp)
    plot_roc_curve(np.array(all_ground_truth), np.array(all_predictions), timestamp)
    eer, fnr01, fnr05, fnr10 = plot_det_curve(np.array(all_ground_truth), np.array(all_predictions), timestamp)
    avg_precision = plot_precision_recall_curve(np.array(all_ground_truth), np.array(all_predictions), timestamp)
else:
    logging.error("No batches processed. Check your image paths.")
    print("No batches processed. Check your image paths.")
