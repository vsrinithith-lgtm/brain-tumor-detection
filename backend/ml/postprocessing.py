import numpy as np
from scipy import ndimage

def postprocess_mask(mask: np.ndarray, min_cluster_size: int = 10) -> np.ndarray:
    """
    Cleans up binary tumor mask by removing small disconnected noise fragments.
    Supports 3D array (D, H, W) or 4D multi-channel array (C, D, H, W).
    """
    if mask.ndim == 4:
        cleaned_channels = [postprocess_mask(mask[c], min_cluster_size) for c in range(mask.shape[0])]
        return np.stack(cleaned_channels, axis=0)

    if not np.any(mask):
        return mask.astype(np.uint8)

    labeled, num_features = ndimage.label(mask > 0)
    if num_features == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    sizes = ndimage.sum(1, labeled, range(1, num_features + 1))
    clean_mask = np.zeros_like(mask, dtype=np.uint8)

    for i, size in enumerate(sizes):
        if size >= min_cluster_size:
            clean_mask[labeled == (i + 1)] = 1

    return clean_mask

def quantify_single_subregion(mask: np.ndarray, spacing: tuple) -> dict:
    """
    Calculates sub-region tumor metrics based on voxel spacing (dx, dy, dz) in mm.
    """
    tumor_voxels = int(np.sum(mask > 0))
    dx, dy, dz = spacing
    voxel_volume_mm3 = float(dx * dy * dz)
    tumor_volume_mm3 = float(tumor_voxels * voxel_volume_mm3)
    tumor_volume_cm3 = float(tumor_volume_mm3 / 1000.0)

    if tumor_voxels == 0:
        return {
            "tumor_detected": False,
            "tumor_voxels": 0,
            "tumor_volume_cm3": 0.0,
            "tumor_volume_mm3": 0.0,
            "bounding_box": None,
            "affected_slices": {"axial": [], "coronal": [], "sagittal": []},
            "primary_slice_index": 0
        }

    nonzero_coords = np.argwhere(mask > 0)
    min_coords = nonzero_coords.min(axis=0).tolist()
    max_coords = nonzero_coords.max(axis=0).tolist()
    
    bounding_box = {
        "x": [int(min_coords[0]), int(max_coords[0])],
        "y": [int(min_coords[1]), int(max_coords[1])],
        "z": [int(min_coords[2]), int(max_coords[2])]
    }

    axial_counts = np.sum(mask > 0, axis=(0, 1))
    coronal_counts = np.sum(mask > 0, axis=(0, 2))
    sagittal_counts = np.sum(mask > 0, axis=(1, 2))

    affected_axial = np.where(axial_counts > 0)[0].tolist()
    affected_coronal = np.where(coronal_counts > 0)[0].tolist()
    affected_sagittal = np.where(sagittal_counts > 0)[0].tolist()

    primary_slice_index = int(np.argmax(axial_counts))

    return {
        "tumor_detected": True,
        "tumor_voxels": tumor_voxels,
        "tumor_volume_cm3": round(tumor_volume_cm3, 3),
        "tumor_volume_mm3": round(tumor_volume_mm3, 2),
        "bounding_box": bounding_box,
        "affected_slices": {
            "axial": affected_axial,
            "coronal": affected_coronal,
            "sagittal": affected_sagittal
        },
        "primary_slice_index": primary_slice_index
    }

def quantify_tumor(mask: np.ndarray, spacing: tuple) -> dict:
    """
    Quantifies tumor mask(s). Handles 3D single binary mask OR 4D (3, D, H, W) BraTS multiclass masks:
    Channel 0: TC (Tumor Core), Channel 1: WT (Whole Tumor), Channel 2: ET (Enhancing Tumor)
    """
    dx, dy, dz = spacing

    if mask.ndim == 4 and mask.shape[0] == 3:
        # Multiclass BraTS masks
        tc_mask = mask[0]
        wt_mask = mask[1]
        et_mask = mask[2]

        quant_tc = quantify_single_subregion(tc_mask, spacing)
        quant_wt = quantify_single_subregion(wt_mask, spacing)
        quant_et = quantify_single_subregion(et_mask, spacing)

        primary_slice = quant_wt.get("primary_slice_index", 0)

        return {
            "multiclass": True,
            "tumor_detected": quant_wt["tumor_detected"],
            "tumor_volume_cm3": quant_wt["tumor_volume_cm3"], # Whole Tumor Volume
            "tumor_volume_mm3": quant_wt["tumor_volume_mm3"],
            "tumor_voxels": quant_wt["tumor_voxels"],
            "voxel_spacing_mm": [round(float(dx), 3), round(float(dy), 3), round(float(dz), 3)],
            "bounding_box": quant_wt["bounding_box"],
            "affected_slices": quant_wt["affected_slices"],
            "primary_slice_index": primary_slice,
            "subregions": {
                "whole_tumor": quant_wt,
                "tumor_core": quant_tc,
                "enhancing_tumor": quant_et
            }
        }
    else:
        # Single 3D binary mask
        quant = quantify_single_subregion(mask, spacing)
        quant["multiclass"] = False
        quant["voxel_spacing_mm"] = [round(float(dx), 3), round(float(dy), 3), round(float(dz), 3)]
        return quant

def calculate_evaluation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Computes Dice, IoU, Sensitivity, and Specificity when Ground Truth mask is available.
    """
    if pred_mask.ndim == 4:
        # Flatten to binary tumor vs background
        pred_b = np.any(pred_mask > 0, axis=0)
    else:
        pred_b = (pred_mask > 0)

    if gt_mask.ndim == 4:
        gt_b = np.any(gt_mask > 0, axis=0)
    else:
        gt_b = (gt_mask > 0)

    tp = np.sum(pred_b & gt_b)
    fp = np.sum(pred_b & ~gt_b)
    fn = np.sum(~pred_b & gt_b)
    tn = np.sum(~pred_b & ~gt_b)

    intersection = tp
    dice = (2.0 * intersection) / (np.sum(pred_b) + np.sum(gt_b) + 1e-8)
    union = tp + fp + fn
    iou = intersection / (union + 1e-8)

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        "dice_score": round(float(dice), 4),
        "iou_score": round(float(iou), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn)
    }
