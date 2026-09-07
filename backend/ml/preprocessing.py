import io
import os
import tempfile
import numpy as np
import nibabel as nib
from PIL import Image

def load_mri_volume(file_bytes: bytes, filename: str):
    """
    Loads single MRI volume data from bytes. Supports .nii, .nii.gz, and PNG/JPG fallback.
    """
    ext = filename.lower()
    
    if ext.endswith('.nii') or ext.endswith('.nii.gz'):
        suffix = '.nii.gz' if ext.endswith('.nii.gz') else '.nii'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            nifti_img = nib.load(tmp_path)
            data = nifti_img.get_fdata().astype(np.float32)
            header = nifti_img.header
            zooms = header.get_zooms()
            spacing = (
                float(zooms[0]) if len(zooms) > 0 and zooms[0] > 0 else 1.0,
                float(zooms[1]) if len(zooms) > 1 and zooms[1] > 0 else 1.0,
                float(zooms[2]) if len(zooms) > 2 and zooms[2] > 0 else 1.0,
            )
            affine = nifti_img.affine.tolist() if hasattr(nifti_img, 'affine') else None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        if data.ndim == 4 and data.shape[3] == 1:
            data = data[:, :, :, 0]
        elif data.ndim == 4:
            data = data[:, :, :, 0]

        if data.ndim != 3:
            raise ValueError(f"Expected 3D MRI volume, got array with shape {data.shape}")
            
        is_nifti = True
    else:
        try:
            pil_img = Image.open(io.BytesIO(file_bytes)).convert('L')
            img_arr = np.array(pil_img, dtype=np.float32)
            data = np.stack([img_arr] * 5, axis=-1)
            spacing = (1.0, 1.0, 1.0)
            affine = None
            is_nifti = False
        except Exception as e:
            raise ValueError(f"Could not decode image file: {str(e)}")

    if np.isnan(data).any() or np.isinf(data).any():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    metadata = {
        "filename": filename,
        "is_nifti": is_nifti,
        "shape": list(data.shape),
        "spacing": list(spacing),
        "voxel_volume_mm3": float(spacing[0] * spacing[1] * spacing[2]),
        "min_intensity": float(np.min(data)),
        "max_intensity": float(np.max(data)),
        "mean_intensity": float(np.mean(data)),
        "std_intensity": float(np.std(data)),
        "affine": affine
    }
    
    return data, spacing, metadata

def load_multimodal_mri_volumes(modality_bytes_dict: dict):
    """
    Loads 4 multimodal MRI sequences (T1c, T1, T2, FLAIR) for MONAI BraTS inference.
    Channel 0: T1c, Channel 1: T1, Channel 2: T2, Channel 3: FLAIR
    Returns:
        tensor_4d (np.ndarray): Shape (4, D, H, W) normalized float32
        spacing (tuple): (dx, dy, dz)
        metadata (dict): composite metadata
    """
    modalities = ['t1c', 't1', 't2', 'flair']
    volumes = []
    base_shape = None
    base_spacing = None
    base_affine = None

    for mod in modalities:
        if mod not in modality_bytes_dict or not modality_bytes_dict[mod]:
            raise ValueError(f"Missing required multimodal scan sequence: '{mod.upper()}'.")
            
        data, spacing, meta = load_mri_volume(modality_bytes_dict[mod]["bytes"], modality_bytes_dict[mod]["filename"])
        
        if base_shape is None:
            base_shape = data.shape
            base_spacing = spacing
            base_affine = meta["affine"]
        elif data.shape != base_shape:
            raise ValueError(f"Spatial dimension mismatch for {mod.upper()}: got {data.shape}, expected {base_shape}.")

        # Channel-wise intensity normalization
        norm_data = normalize_intensity(data)
        volumes.append(norm_data)

    stacked_4d = np.stack(volumes, axis=0) # Shape: (4, D, H, W)
    
    metadata = {
        "multimodal": True,
        "modalities": modalities,
        "shape": list(base_shape),
        "spacing": list(base_spacing),
        "voxel_volume_mm3": float(base_spacing[0] * base_spacing[1] * base_spacing[2]),
        "affine": base_affine
    }

    return stacked_4d, base_spacing, metadata

def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    """
    Min-Max normalization clipped between 1st and 99th percentiles for robust intensity contrast.
    """
    non_zero_mask = volume > 0
    if not np.any(non_zero_mask):
        return np.zeros_like(volume)
        
    p1 = np.percentile(volume[non_zero_mask], 1)
    p99 = np.percentile(volume[non_zero_mask], 99)
    
    clipped = np.clip(volume, p1, p99)
    if p99 > p1:
        normalized = (clipped - p1) / (p99 - p1)
    else:
        normalized = np.zeros_like(volume)
        
    return normalized.astype(np.float32)
