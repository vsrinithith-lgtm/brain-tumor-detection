import os
import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage

class UNet3D(nn.Module):
    """
    3D U-Net Architecture for Volumetric Medical Image Segmentation.
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super(UNet3D, self).__init__()
        features = init_features
        
        # Encoder
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = UNet3D._block(features * 2, features * 4, name="bottleneck")
        
        # Decoder
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")
        
        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        
        dec2 = self.upconv2(bottleneck)
        if dec2.shape != enc2.shape:
            dec2 = nn.functional.interpolate(dec2, size=enc2.shape[2:])
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        if dec1.shape != enc1.shape:
            dec1 = nn.functional.interpolate(dec1, size=enc1.shape[2:])
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.final_conv(dec1))


class BaselineSegmentationModel:
    """
    Algorithmic segmentation baseline for MRI scans when pretrained network weights 
    are absent. Delineates hyper-intense tumor regions using adaptive thresholding, 
    brain background mask extraction, and spatial morphological operations.
    """
    def __init__(self):
        self.name = "Baseline Segmentation Engine (Morphological/Intensity-based)"

    def predict_mask(self, volume: np.ndarray, sensitivity_level: float = 0.5) -> np.ndarray:
        """
        Generates a 3D binary mask array (1 = tumor, 0 = background).
        """
        non_zero = volume > 0.01
        if not np.any(non_zero):
            return np.zeros_like(volume, dtype=np.uint8)

        v_max = np.max(volume)
        v_mean = np.mean(volume[non_zero])
        v_std = np.std(volume[non_zero])
        
        # Adaptive thresholding for hyper-intense MRI lesion enhancement
        thresh = min(v_max * 0.85, v_mean + sensitivity_level * v_std)
        raw_mask = (volume >= thresh) & non_zero
        
        if not np.any(raw_mask):
            # Fallback threshold for distinct upper percentile signal
            thresh = np.percentile(volume[non_zero], 92)
            raw_mask = (volume >= thresh) & non_zero

        # Morphological opening/closing to clean up isolated noise
        struct = ndimage.generate_binary_structure(3, 1)
        cleaned_mask = ndimage.binary_opening(raw_mask, structure=struct, iterations=1)
        cleaned_mask = ndimage.binary_closing(cleaned_mask, structure=struct, iterations=1)
        
        # Fallback to raw_mask if binary opening erased tiny lesion
        if not np.any(cleaned_mask):
            cleaned_mask = raw_mask
            
        # Keep significant connected components (tumors)
        labeled, num_features = ndimage.label(cleaned_mask)
        if num_features == 0:
            return np.zeros_like(volume, dtype=np.uint8)
            
        component_sizes = ndimage.sum(1, labeled, range(1, num_features + 1))
        min_size = 10  # minimum voxel component size
        
        mask = np.zeros_like(volume, dtype=np.uint8)
        for idx, size in enumerate(component_sizes):
            if size >= min_size:
                mask[labeled == (idx + 1)] = 1
                
        return mask
