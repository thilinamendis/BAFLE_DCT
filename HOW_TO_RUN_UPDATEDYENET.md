# How to Run UpdatedYenet with Your Data

This guide explains how to run the UpdatedYenet model with your own data containing stego and original images.

## Dataset Structure

Your dataset should be organized as follows:

```
/path/to/your/data/
├── originals/   # Contains original/cover images
└── stegos/      # Contains stego images
```

Make sure both directories contain the same number of images with matching filenames.

## Training the Model

To train the UpdatedYenet model with your own data, use the following command:

```bash
cd /home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/UpdatedYenet

python train.py \
  --custom_cover_path /path/to/your/data/originals \
  --custom_stego_path /path/to/your/data/stegos \
  --custom_valid_cover_path /path/to/your/validation/originals \
  --custom_valid_stego_path /path/to/your/validation/stegos \
  --img_size 224 \
  --num_epochs 50
```

If you don't have separate validation data, you can use the same paths for both training and validation:

```bash
python train.py \
  --custom_cover_path /path/to/your/data/originals \
  --custom_stego_path /path/to/your/data/stegos \
  --img_size 224 \
  --num_epochs 50
```

## Testing the Model

After training, you can test the model on your data using:

```bash
cd /home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/UpdatedYenet

python test.py \
  --cover_dir /path/to/your/data/originals \
  --stego_dir /path/to/your/data/stegos \
  --checkpoint ./checkpoints/net_50.pt \
  --img_size 224 \
  --batch_size 32
```

Replace `net_50.pt` with your actual trained model file (you can find the latest one in the `checkpoints` directory).

## Recommended Image Size

The model works best with square images resized to either 224×224 or 512×512 pixels. Use the `--img_size` parameter to specify your preferred size.

## Using Your Data from Image Outputs Directory

Based on your workspace structure, you can use the original and stego directories from your image_outputs folder:

```bash
# For training
python train.py \
  --custom_cover_path /home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/image_outputs/originals \
  --custom_stego_path /home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/image_outputs/stegos \
  --img_size 224 \
  --num_epochs 50

# For testing
python test.py \
  --cover_dir /home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/image_outputs/originals \
  --stego_dir /home/btm0050/Research_AI_Stegno_Analysis/SteganoGan/image_outputs/stegos \
  --checkpoint ./checkpoints/net_50.pt \
  --img_size 224
```

## Troubleshooting

If you encounter errors related to image loading or dimensions, try:

1. Ensure all your images are valid and can be opened
2. Make sure both original and stego directories have the same number of images
3. Use a smaller batch size if you run into memory issues
4. Check that the image files have the same extensions in both directories
