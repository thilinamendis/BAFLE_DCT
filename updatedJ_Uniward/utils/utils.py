"""This module provides utility function for training."""
import os
import re
from typing import Any, Dict
import torch
from torch import nn

from opts.options import arguments

opt = arguments()


def saver(state: Dict[str, float], save_dir: str, epoch: int) -> None:
    """Saves model state to disk.
    
    Args:
        state: State dictionary containing model parameters and training info
        save_dir: Directory to save the checkpoint
        epoch: Current epoch number
    """
    torch.save(state, save_dir + "net_" + str(epoch) + ".pt")


def latest_checkpoint() -> int:
    """Returns latest checkpoint.
    
    Returns:
        The number of the latest checkpoint or None if no checkpoints exist
    """
    if os.path.exists(opt.checkpoints_dir):
        all_chkpts = "".join(os.listdir(opt.checkpoints_dir))
        if len(all_chkpts) > 0:
            latest = max(map(int, re.findall("\d+", all_chkpts)))
        else:
            latest = None
    else:
        latest = None
    return latest


def adjust_learning_rate(optimizer: Any, epoch: int) -> None:
    """Sets the learning rate to the initial learning_rate and decays by 10
    every 30 epochs.
    
    Args:
        optimizer: The optimizer to update
        epoch: Current epoch number
    """
    learning_rate = opt.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def weights_init(param: Any) -> None:
    """Initializes weights of Conv and fully connected layers.
    
    Args:
        param: Network parameter to initialize
    """
    if isinstance(param, nn.Conv2d):
        nn.init.kaiming_normal_(param.weight.data, mode='fan_out', nonlinearity='relu')
        if param.bias is not None:
            nn.init.constant_(param.bias.data, 0.2)
    elif isinstance(param, nn.BatchNorm2d):
        nn.init.constant_(param.weight.data, 1)
        nn.init.constant_(param.bias.data, 0)
    elif isinstance(param, nn.Linear):
        nn.init.normal_(param.weight.data, mean=0.0, std=0.01)
        nn.init.constant_(param.bias.data, 0.0)


def calculate_accuracy(outputs, labels):
    """Calculate accuracy from model outputs and true labels.
    
    Args:
        outputs: Model outputs with shape [B, 2]
        labels: True labels with shape [B]
        
    Returns:
        Accuracy as a percentage
    """
    _, predicted = torch.max(outputs, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def calculate_confusion_matrix(outputs, labels):
    """Calculate confusion matrix metrics.
    
    Args:
        outputs: Model outputs with shape [B, 2]
        labels: True labels with shape [B]
        
    Returns:
        Dictionary containing TP, FP, TN, FN counts and derived metrics
    """
    _, predicted = torch.max(outputs, 1)
    
    # Calculate TP, FP, TN, FN
    tp = ((predicted == 1) & (labels == 1)).sum().item()
    fp = ((predicted == 1) & (labels == 0)).sum().item()
    tn = ((predicted == 0) & (labels == 0)).sum().item()
    fn = ((predicted == 0) & (labels == 1)).sum().item()
    
    # Calculate metrics
    accuracy = 100 * (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
