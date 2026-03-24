"""This module provides method to enter various input to the model training."""
import argparse
import os


def arguments():
    """This function returns arguments."""
    parser = argparse.ArgumentParser()
    # Default dataset paths (can be changed as needed)
    parser.add_argument("--cover_path", default=os.path.expanduser("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/500bit/500bitdataset/train/original"))
    parser.add_argument("--stego_path", default=os.path.expanduser("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/500bit/500bitdataset/train/stego"))
    parser.add_argument("--valid_cover_path", default=os.path.expanduser("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/500bit/500bitdataset/val/original"))
    parser.add_argument("--valid_stego_path", default=os.path.expanduser("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/500bit/500bitdataset/val/stego"))
    parser.add_argument("--checkpoints_dir", default="./checkpoints/")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--train_size", type=int, default=32)
    parser.add_argument("--val_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--use_balanced_dataset", action="store_true", help="Use weighted sampling for balanced training")
    parser.add_argument("--use_paired_balanced", action="store_true", help="Use weighted sampling while preserving cover-stego pairs")
    opt = parser.parse_args()
    return opt