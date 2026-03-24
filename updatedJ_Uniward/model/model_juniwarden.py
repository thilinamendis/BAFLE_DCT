"""This is implementation of JPEG domain steganalysis model based on the J-UNIWARD detection principles."""

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2
import math

class DCTExtractor(nn.Module):
    """This class extracts DCT coefficients from images for JPEG-domain steganalysis."""
    
    def __init__(self, quality_factor=75):
        """Constructor.
        
        Args:
            quality_factor: JPEG quality factor (default: 75)
        """
        super().__init__()
        self.quality_factor = quality_factor
        # Standard JPEG quantization tables
        self.std_luminance_quant_table = torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float32)
        
        # Initialize quantization table based on quality factor
        self.quant_table = self._create_quantization_table(self.quality_factor)
        
        # Register quantization table as buffer to move it to the correct device
        self.register_buffer('quantization_table', self.quant_table)
        
        # Create DCT kernels for fast DCT computation
        self.dct_kernels = self._create_dct_kernels()
        
        # Define DCT normalization weights
        self.norm_weights = torch.ones((8, 8), dtype=torch.float32)
        self.norm_weights[0, :] = 1.0 / math.sqrt(2.0)
        self.norm_weights[:, 0] = 1.0 / math.sqrt(2.0)
        self.norm_weights[0, 0] = 0.5
        self.register_buffer('normalization_weights', self.norm_weights)
        
        print(f"DCT Extractor initialized with quality factor: {quality_factor}")
    
    def _create_quantization_table(self, quality):
        """Create quantization table based on quality factor."""
        if quality < 50:
            quality = 5000 / quality
        else:
            quality = 200 - quality * 2
        
        quant_table = torch.floor((self.std_luminance_quant_table * quality + 50) / 100)
        quant_table = torch.clamp(quant_table, 1, 255)
        return quant_table
    
    def _create_dct_kernels(self):
        """Create DCT kernels for efficient DCT computation."""
        kernels = []
        for u in range(8):
            for v in range(8):
                kernel = torch.zeros((8, 8), dtype=torch.float32)
                for x in range(8):
                    for y in range(8):
                        kernel[x, y] = math.cos((2 * x + 1) * u * math.pi / 16) * math.cos((2 * y + 1) * v * math.pi / 16)
                kernel = kernel / 4.0
                if u == 0:
                    kernel = kernel / math.sqrt(2.0)
                if v == 0:
                    kernel = kernel / math.sqrt(2.0)
                kernels.append(kernel)
        
        return torch.stack(kernels).reshape(64, 1, 8, 8)
    
    def forward(self, x):
        """Extract DCT coefficients from images.
        
        Args:
            x: Input image tensor of shape [B, C, H, W]
            
        Returns:
            DCT coefficients tensor of shape [B, 64, H//8, W//8]
        """
        batch_size, channels, height, width = x.shape
        
        # Ensure dimensions are divisible by 8
        pad_h = 8 - (height % 8) if height % 8 != 0 else 0
        pad_w = 8 - (width % 8) if width % 8 != 0 else 0
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            
        # Move kernels to device if needed
        device = x.device
        dct_kernels = self.dct_kernels.to(device)
        
        # Process each color channel separately
        dct_coeffs = []
        for c in range(channels):
            # Extract single channel
            channel = x[:, c:c+1, :, :]
            
            # Subtract 128 from pixel values (JPEG preprocessing)
            channel = channel - 128.0
            
            # Apply DCT using convolution
            coeffs = F.conv2d(channel, dct_kernels, stride=8, padding=0)
            
            # Reshape to [B, 8, 8, H//8, W//8]
            h_blocks = (height + pad_h) // 8
            w_blocks = (width + pad_w) // 8
            coeffs = coeffs.reshape(batch_size, 8, 8, h_blocks, w_blocks)
            
            # Quantize coefficients
            coeffs = coeffs / self.quantization_table.view(1, 8, 8, 1, 1).to(device)
            
            # Round to simulate JPEG quantization
            coeffs = torch.round(coeffs)
            
            # Reshape to [B, 64, H//8, W//8]
            coeffs = coeffs.reshape(batch_size, 64, h_blocks, w_blocks)
            
            dct_coeffs.append(coeffs)
        
        # Average over channels for grayscale-like processing
        dct_coeffs = torch.stack(dct_coeffs, dim=1)  # [B, C, 64, H//8, W//8]
        
        return dct_coeffs


class PhaseAwareBlock(nn.Module):
    """Phase-aware processing block for DCT coefficient analysis."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool(x)
        return x


class JUniwarden(nn.Module):
    """JPEG domain steganalysis model based on J-UNIWARD principles."""
    
    def __init__(self, quality_factor=75, direct_dct=False):
        """Constructor.
        
        Args:
            quality_factor: JPEG quality factor for DCT analysis
            direct_dct: Whether to analyze DCT coefficients directly or extracted from images
        """
        super().__init__()
        self.direct_dct = direct_dct
        self.dct_extractor = DCTExtractor(quality_factor=quality_factor)
        
        # First layer processes raw DCT coefficients
        self.initial_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # Phase-aware processing blocks
        self.phase_block1 = PhaseAwareBlock(32, 64)
        self.phase_block2 = PhaseAwareBlock(64, 128)
        self.phase_block3 = PhaseAwareBlock(128, 256)
        self.phase_block4 = PhaseAwareBlock(256, 512)
        
        # Global average pooling and classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Binary classification: cover vs stego
        )
        
        print(f"J-UNIWARD model initialized with quality factor {quality_factor}, direct_dct={direct_dct}")
        
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Either raw images [B, C, H, W] or DCT coefficients [B, C, 64, H//8, W//8]
            
        Returns:
            Classification logits [B, 2]
        """
        if not self.direct_dct:
            # Extract DCT coefficients from images
            batch_size, channels, height, width = x.shape
            dct_coeffs = self.dct_extractor(x)  # [B, C, 64, H//8, W//8]
            
            # Average over color channels for consistent processing
            dct_coeffs = torch.mean(dct_coeffs, dim=1)  # [B, 64, H//8, W//8]
        else:
            # Assume input is already DCT coefficients
            dct_coeffs = x
        
        # Initial convolution
        x = self.initial_conv(dct_coeffs)
        
        # Phase-aware blocks
        x = self.phase_block1(x)
        x = self.phase_block2(x)
        x = self.phase_block3(x)
        x = self.phase_block4(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x


if __name__ == "__main__":
    # Simple test
    model = JUniwarden()
    print(model)
    
    # Test with random input
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    print(f"Output shape: {output.shape}")
