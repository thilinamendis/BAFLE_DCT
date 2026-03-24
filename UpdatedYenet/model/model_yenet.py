"""This is unofficial implementation of YeNet:
Deep Learning Hierarchical Representation for Image Steganalysis.
"""

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import sys
from pathlib import Path
import os

import numpy as np

# Define the correct path for srm.npy
srm_path = Path(__file__).parent / "srm.npy"
print(f"Looking for SRM filters at: {srm_path.absolute()}")

# Fallback if file doesn't exist in the primary location
if not srm_path.exists():
    # Try to find it in other possible locations
    alternative_paths = [
        Path('./UpdatedYenet/model/srm.npy'),
        Path('./model/srm.npy'),
        Path('./srm.npy')
    ]
    
    for alt_path in alternative_paths:
        if alt_path.exists():
            srm_path = alt_path
            print(f"Found SRM filters at alternative location: {srm_path.absolute()}")
            break
    
    if not srm_path.exists():
        raise FileNotFoundError(f"Could not find srm.npy in any expected locations")

class SRMConv(nn.Module):
    """This class computes convolution of input tensor with 30 SRM filters"""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Load SRM filters and ensure they're properly formatted
            self.srm = torch.from_numpy(np.load(str(srm_path))).to(
                self.device, dtype=torch.float
            )
            
            # Handle both grayscale and RGB inputs
            if self.srm.dim() == 3:
                # Add channel dimension if not present
                self.srm = self.srm.unsqueeze(1)
            
            # Print shape only once during initialization
            print(f"SRMConv initialized with filter shape: {self.srm.shape}")
            
            # Set truncation limits for the truncated linear unit
            self.tlu = nn.Hardtanh(min_val=-3.0, max_val=3.0)
        except Exception as e:
            print(f"Error initializing SRM filters: {e}")
            # Fallback to simple edge detection kernels if SRM fails to load
            print("Using fallback simple edge detection kernels")
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
            
            # Create 3 basic filters
            filters = torch.stack([sobel_x, sobel_y, laplacian])
            # Expand to 30 filters by rotation and scaling
            expanded_filters = []
            for f in filters:
                expanded_filters.append(f)
                expanded_filters.append(f.t())  # Transpose
                expanded_filters.append(f * 0.5)  # Scaled versions
                expanded_filters.append(f.t() * 0.5)
            
            # Fill to 30 filters by repeating if needed
            while len(expanded_filters) < 30:
                expanded_filters.append(expanded_filters[0])
                
            # Convert to tensor and reshape to proper kernel format
            self.srm = torch.stack(expanded_filters[:30]).unsqueeze(1).to(self.device)
            print(f"Created fallback filters with shape: {self.srm.shape}")
            self.tlu = nn.Hardtanh(min_val=-3.0, max_val=3.0)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns output tensor after convolution with 30 SRM filters
        followed by TLU activation."""
        # Handle RGB or grayscale input
        if inp.shape[1] == 3:  # RGB input
            # Apply filters to each channel and average
            results = []
            for i in range(3):
                channel = inp[:, i:i+1]  # Extract single channel
                result = F.conv2d(channel, self.srm)
                results.append(result)
            result = torch.mean(torch.stack(results), dim=0)
        else:  # Grayscale input
            result = F.conv2d(inp, self.srm)
        
        # Apply truncated linear unit
        output = self.tlu(result)
        
        # Check for NaN values without printing every time
        if torch.isnan(output).any() and not hasattr(self, '_nan_warning_shown'):
            print(f"WARNING: NaN values in SRM output")
            self._nan_warning_shown = True
            
        return output


class ConvBlock(nn.Module):
    """This class returns building block for YeNet class."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        use_pool: bool = False,
        pool_size: int = 3,
        pool_padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.activation = nn.ReLU()
        self.pool = nn.AvgPool2d(
            kernel_size=pool_size, stride=2, padding=pool_padding
        )
        self.use_pool = use_pool

    def forward(self, inp: Tensor) -> Tensor:
        """Returns conv->gaussian->average pooling."""
        if self.use_pool:
            return self.pool(self.activation(self.conv(inp)))
        return self.activation(self.conv(inp))


class YeNet(nn.Module):
    """This class returns YeNet model."""

    def __init__(self) -> None:
        super().__init__()
        # Initialize SRM layer once during model creation
        self.srm_layer = SRMConv()
        self.layer1 = ConvBlock(30, 30, kernel_size=3)
        self.layer2 = ConvBlock(30, 30, kernel_size=3)
        self.layer3 = ConvBlock(
            30, 30, kernel_size=3, use_pool=True, pool_size=2, pool_padding=0
        )
        self.layer4 = ConvBlock(
            30,
            32,
            kernel_size=5,
            padding=0,
            use_pool=True,
            pool_size=3,
            pool_padding=0,
        )
        self.layer5 = ConvBlock(
            32, 32, kernel_size=5, use_pool=True, pool_padding=0
        )
        self.layer6 = ConvBlock(32, 32, kernel_size=5, use_pool=True)
        self.layer7 = ConvBlock(32, 16, kernel_size=3)
        self.layer8 = ConvBlock(16, 16, kernel_size=3, stride=3)
        # Calculate the expected output size after all convolutions and pooling
        # Original implementation had 16 features with 3x3 spatial dimensions
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=16 * 3 * 3, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=2)
        )


    def forward(self, image: Tensor) -> Tensor:
        """Returns logit for the given tensor."""
        # Convert RGB to grayscale if needed (original YeNet expects grayscale)
        if image.shape[1] == 3:  # If RGB (3 channels)
            # Use the standard RGB to grayscale conversion weights
            gray_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(image.device)
            image = (image * gray_weights).sum(dim=1, keepdim=True)
        
        # Print the input shape for debugging (only once)
        if not hasattr(self, '_shape_warning_shown'):
            print(f"Input image shape: {image.shape}")
            self._shape_warning_shown = True
            
        # Apply SRM convolution using the pre-initialized layer
        out = self.srm_layer(image)
        
        # Apply sequential layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        
        # Flatten for fully connected layer
        out = out.view(out.size(0), -1)
        
        # Check for NaNs without printing every time
        if torch.isnan(out).any() and not hasattr(self, '_nan_fc_warning_shown'):
            print(f"WARNING: NaN values detected in tensor before FC layer")
            self._nan_fc_warning_shown = True
        
        # Apply fully connected layer
        out = self.fully_connected(out)
        return out


if __name__ == "__main__":
    net = YeNet()
    print(net)
    inp_image = torch.randn((1, 1, 256, 256))
    print(net(inp_image))
