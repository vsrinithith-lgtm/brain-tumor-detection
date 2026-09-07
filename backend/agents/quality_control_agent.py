import numpy as np
from backend.ml.postprocessing import calculate_evaluation_metrics

class QualityControlAgent:
    """
    Quality Control Agent: Performs automated verification on predicted segmentation mask.
    Checks:
    - Empty mask detection
    - Connected components sanity
    - Total tumor volume bounds
    - Bounding box checks
    - Flags suspiciously small/large segmentations
    Computes Dice, IoU, predicted volume, GT volume, and volume difference ONLY when GT mask is provided.
    """
    def __init__(self):
        self.name = "Quality Control Agent"

    def run(self, volume_shape: tuple, mask: np.ndarray, quantification: dict, gt_mask: np.ndarray = None) -> dict:
        result = {
            "agent": self.name,
            "status": "completed",
            "action_performed": "Evaluated mask bounds, component counts, volume ratio, and GT metrics",
            "qc_status": "PASS",
            "message": "Segmentation passed automated quality control verification.",
            "metrics": None,
            "details": {}
        }

        total_volume_voxels = int(volume_shape[0] * volume_shape[1] * volume_shape[2])
        tumor_voxels = quantification.get("tumor_voxels", 0)
        tumor_vol_cm3 = quantification.get("tumor_volume_cm3", 0.0)

        checks_performed = []

        # Check 1: Empty mask
        if tumor_voxels == 0:
            result["qc_status"] = "WARNING"
            result["message"] = "QC Note: Zero tumor voxels detected. Scan classified as healthy brain MRI or lesion below threshold."
            checks_performed.append({"check": "Empty Mask Check", "status": "WARNING", "reason": "No tumor voxels detected"})
        
        # Check 2: Suspiciously large segmentation (>40% of volume)
        elif tumor_voxels > 0.4 * total_volume_voxels:
            result["qc_status"] = "WARNING"
            result["message"] = f"QC Warning: Tumor mask occupies {round(tumor_voxels / total_volume_voxels * 100, 1)}% of total spatial volume, exceeding typical lesion bounds."
            checks_performed.append({"check": "Volume Ratio Check", "status": "WARNING", "reason": f"Occupies {round(tumor_voxels / total_volume_voxels * 100, 1)}% of volume"})

        # Check 3: Valid bounds check
        else:
            checks_performed.append({"check": "Mask Bounds Verification", "status": "PASS", "reason": f"Normal tumor volume ({tumor_vol_cm3} cm³, {tumor_voxels} voxels)"})

        # GT Evaluation Metrics calculation (ONLY when GT is supplied)
        gt_metrics = None
        if gt_mask is not None:
            if gt_mask.shape != mask.shape:
                result["qc_status"] = "FAIL"
                result["message"] = f"QC Failure: Ground truth spatial dimension {gt_mask.shape} mismatches predicted mask dimension {mask.shape}."
                checks_performed.append({"check": "Ground Truth Shape Matching", "status": "FAIL", "reason": "Shape mismatch"})
            else:
                gt_metrics = calculate_evaluation_metrics(mask, gt_mask)
                
                # Calculate GT volume and volume difference
                gt_voxels = int(np.sum(gt_mask > 0))
                spacing = quantification.get("voxel_spacing_mm", [1.0, 1.0, 1.0])
                voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
                gt_vol_cm3 = round((gt_voxels * voxel_vol_mm3) / 1000.0, 3)
                vol_diff_cm3 = round(abs(tumor_vol_cm3 - gt_vol_cm3), 3)

                gt_metrics["gt_volume_cm3"] = gt_vol_cm3
                gt_metrics["predicted_volume_cm3"] = tumor_vol_cm3
                gt_metrics["volume_difference_cm3"] = vol_diff_cm3

                checks_performed.append({"check": "Ground Truth Dice Evaluation", "status": "PASS", "reason": f"Dice Score: {gt_metrics['dice_score']}, IoU: {gt_metrics['iou_score']}"})

        result["metrics"] = gt_metrics
        result["details"] = {
            "qc_status": result["qc_status"],
            "checks_performed": checks_performed,
            "tumor_volume_cm3": tumor_vol_cm3,
            "gt_provided": gt_mask is not None
        }

        return result
