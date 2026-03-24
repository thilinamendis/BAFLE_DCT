"""This module is used to test the Srnet model with labeled output per image."""
from glob import glob
import torch
import numpy as np
from model.model import Srnet
from pathlib import Path
from PIL import Image

device = torch.device("cpu")

TEST_BATCH_SIZE = 40
COVER_PATH = str(Path("~/NewModelsForCV/image_outputs_new/originals/*.*").expanduser())
STEGO_PATH = str(Path("~/NewModelsForCV/image_outputs_new/stegos/*.*").expanduser())
CHKPT = str(Path("./checkpoints/net_12.pt").expanduser())

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

cover_data = load_and_resize(cover_image_names)
stego_data = load_and_resize(stego_image_names)

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
    prediction = outputs.data.max(1)[1]

    accuracy = (
        prediction.eq(batch_labels.data).sum()
        * 100.0
        / batch_labels.size(0)
    )
    test_accuracy.append(accuracy.item())

    # Print file name and classification result
    for i in range(len(prediction)):
        label = "Stego" if prediction[i].item() == 1 else "Not-Stego"
        print(f"{batch_filenames[i]} -> {label}")

print(f"\nLoaded {len(test_accuracy)} batches")
print(f"From Cover Path: {COVER_PATH}")
print(f"From Stego Path: {STEGO_PATH}")

if test_accuracy:
    print(f"\nFinal test_accuracy = {sum(test_accuracy)/len(test_accuracy):.2f}")
else:
    print("No batches processed. Check your image paths.")

