import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from pathlib import Path

# Paths
orig_dir = Path("./RoSteals_Images/originals")
stego_dir = Path("./RoSteals_Images/stegos")
output_csv = "rosteals_metrics.csv"

# Results list
metrics = []

# Loop through all original images
for orig_file in orig_dir.glob("*_original.JPEG"):
    base_name = orig_file.stem.replace("_original", "")
    stego_file = stego_dir / f"{base_name}_stego.JPEG"

    if not stego_file.exists():
        print(f"[!] Missing stego image for: {base_name}")
        continue

    # Read images
    orig = cv2.imread(str(orig_file))
    stego = cv2.imread(str(stego_file))

    if orig is None or stego is None:
        print(f"[!] Failed to read one or both images: {base_name}")
        continue

    if orig.shape != stego.shape:
        orig = cv2.resize(orig, (stego.shape[1], stego.shape[0]))
    # Compute metrics without grayscale conversion
    psnr_val = psnr(orig, stego, data_range=255)
    ssim_val = ssim(orig, stego, channel_axis=-1, data_range=255)
    mse_val = mse(orig, stego)

    metrics.append({
        "ImageName": f"{base_name}.JPEG",
        "PSNR (dB)": psnr_val,
        "SSIM": ssim_val,
        "MSE": mse_val
    })

# Save to CSV
df = pd.DataFrame(metrics)
df.to_csv(output_csv, index=False)
print(f"[✓] Metrics saved to {output_csv}")
