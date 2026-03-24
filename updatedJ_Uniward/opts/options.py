"""This module provides method to enter various input to the model training."""
import argparse
import os


def arguments():
    """This function returns arguments."""
    parser = argparse.ArgumentParser()
    # Default dataset paths (can be changed as needed)
    parser.add_argument("--cover_path", default=os.path.expanduser("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/100bit/100bitdataset/train/original"))
    parser.add_argument("--stego_path", default=os.path.expanduser("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/100bit/100bitdataset/train/stego"))
    parser.add_argument("--valid_cover_path", default=os.path.expanduser("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/100bit/100bitdataset/val/original"))
    parser.add_argument("--valid_stego_path", default=os.path.expanduser("/home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/mount/100bit/100bitdataset/val/stego"))
    parser.add_argument("--checkpoints_dir", default="./checkpoints/")
    parser.add_argument("--num_epochs", type=int, default=32)
    parser.add_argument("--train_size", type=int, default=32)
    parser.add_argument("--val_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--use_balanced_dataset", action="store_true", help="Use weighted sampling for balanced training")
    parser.add_argument("--use_paired_balanced", action="store_true", help="Use weighted sampling while preserving cover-stego pairs")
    parser.add_argument("--quality_factor", type=int, default=75, help="JPEG quality factor for DCT analysis")
    parser.add_argument("--dct_direct_analysis", action="store_true", help="Analyze DCT coefficients directly rather than spatial domain")
    opt = parser.parse_args()
    return opt
