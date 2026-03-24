# J-UNIWARD Steganalysis Model

This implementation provides a steganalysis model designed to detect steganographic content in JPEG images, particularly those using DCT-domain embedding methods like LSB modification of DCT coefficients.

## Background

The J-UNIWARD model is specifically designed for analyzing steganography in the DCT domain (JPEG images). It is particularly effective at detecting methods that manipulate the LSB of DCT coefficients, which makes it ideal for evaluating the security of DCT-based steganographic techniques.

## Key Features

- DCT coefficient extraction for direct frequency domain analysis
- Phase-aware processing to detect subtle statistical anomalies
- Advanced metrics tracking including precision, recall, and F1 score
- Support for both RGB and grayscale images
- Customizable JPEG quality factor for analysis

## Usage

### Training

```bash
python train.py --cover_path /path/to/cover/images --stego_path /path/to/stego/images --quality_factor 75
```

### Optional Arguments

- `--quality_factor`: JPEG quality factor (default: 75)
- `--dct_direct_analysis`: Analyze DCT coefficients directly rather than spatial domain
- `--use_balanced_dataset`: Use balanced sampling for training
- `--use_paired_balanced`: Use paired balanced sampling (preserves cover-stego pairs)
- `--num_epochs`: Number of training epochs (default: 32)
- `--train_size`: Training batch size (default: 32)
- `--val_size`: Validation batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)

## Model Architecture

The model consists of:

1. **DCT Extraction Layer**: Extracts and quantizes DCT coefficients from input images
2. **Phase-Aware Processing Blocks**: Specialized convolutional blocks for analyzing frequency domain patterns
3. **Feature Aggregation**: Global average pooling to combine features
4. **Classification Head**: Final layers for binary classification (cover vs. stego)

## Evaluation

The model tracks several metrics during training:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix (TP, FP, TN, FN)

These metrics provide a comprehensive view of the model's ability to detect steganographic content.

## DCT Domain Analysis

For DCT-LSB embedding methods, this model offers specialized analysis by:
1. Directly analyzing the DCT coefficients
2. Focusing on the statistical artifacts that typically arise from LSB modifications in the frequency domain
3. Using phase-aware processing to detect subtle changes that might be missed by spatial domain analysis

## References

- Holub, V., Fridrich, J., & Denemark, T. (2014). Universal distortion function for steganography in an arbitrary domain. *EURASIP Journal on Information Security*, 2014(1), 1-13. [https://doi.org/10.1186/1687-417X-2014-1](https://doi.org/10.1186/1687-417X-2014-1)
  
- Xu, G., Wu, H. Z., & Shi, Y. Q. (2016). Structural design of convolutional neural networks for steganalysis. *IEEE Signal Processing Letters*, 23(5), 708-712. [https://doi.org/10.1109/LSP.2016.2548421](https://doi.org/10.1109/LSP.2016.2548421)
  
- Boroumand, M., Chen, M., & Fridrich, J. (2018). Deep residual network for steganalysis of digital images. *IEEE Transactions on Information Forensics and Security*, 14(5), 1181-1193. [https://doi.org/10.1109/TIFS.2018.2871749](https://doi.org/10.1109/TIFS.2018.2871749)
  
- Zeng, J., Tan, S., & Li, B. (2018). Pre-processing for JPEG steganalysis by selecting suspicious DCT coefficients. *IEEE Transactions on Information Forensics and Security*, 14(3), 801-815. [https://doi.org/10.1109/TIFS.2018.2870589](https://doi.org/10.1109/TIFS.2018.2870589)
  
- Chen, M., Sedighi, V., Boroumand, M., & Fridrich, J. (2017). JPEG-phase-aware convolutional neural network for steganalysis of JPEG images. *Proceedings of the 5th ACM Workshop on Information Hiding and Multimedia Security*, 75-84. [https://doi.org/10.1145/3082031.3083248](https://doi.org/10.1145/3082031.3083248)

