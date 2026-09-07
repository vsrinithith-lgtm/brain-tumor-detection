import io
import base64
import numpy as np
from PIL import Image, ImageDraw

class ReportingAgent:
    """
    Reporting Agent: Compiles a clinical-style RESEARCH PROTOTYPE report containing:
    - Input modalities
    - Preprocessing status
    - Model name & pretrained status
    - Segmentation status
    - Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) volumes
    - Tumor voxel count & spacing
    - Bounding box
    - QC result & Research disclaimer
    - Visual overlay slice previews
    """
    def __init__(self):
        self.name = "Result & Reporting Agent"

    @staticmethod
    def _slice_to_overlay_base64(mri_slice: np.ndarray, mask_slice: np.ndarray) -> str:
        s_min, s_max = mri_slice.min(), mri_slice.max()
        if s_max > s_min:
            gray = ((mri_slice - s_min) / (s_max - s_min) * 255.0).astype(np.uint8)
        else:
            gray = np.zeros_like(mri_slice, dtype=np.uint8)

        base_img = Image.fromarray(gray).convert("RGBA")
        overlay_arr = np.zeros((*mri_slice.shape, 4), dtype=np.uint8)

        if mask_slice.ndim == 3 and mask_slice.shape[0] == 3:
            # Multiclass: Channel 0: TC, Channel 1: WT, Channel 2: ET
            tc_pixels = mask_slice[0] > 0
            wt_pixels = mask_slice[1] > 0
            et_pixels = mask_slice[2] > 0

            # Whole Tumor = Red
            overlay_arr[wt_pixels] = [255, 60, 60, 140]
            # Tumor Core = Cyan
            overlay_arr[tc_pixels] = [0, 220, 255, 170]
            # Enhancing Tumor = Yellow
            overlay_arr[et_pixels] = [255, 220, 0, 200]
        else:
            tumor_pixels = mask_slice > 0
            overlay_arr[tumor_pixels] = [255, 60, 60, 160]

        overlay_img = Image.fromarray(overlay_arr, mode="RGBA")
        blended = Image.alpha_composite(base_img, overlay_img)

        buf = io.BytesIO()
        blended.convert("RGB").save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"

    def run(self, volume_norm: np.ndarray, mask: np.ndarray, quantification: dict) -> dict:
        result = {
            "agent": self.name,
            "status": "pending",
            "message": "",
            "action_performed": "Generated clinical-style research prototype report and slice overlays",
            "visualizations": {},
            "report_summary": {}
        }

        try:
            if volume_norm.ndim == 4:
                display_volume = volume_norm[3] if volume_norm.shape[0] > 3 else volume_norm[0]
            else:
                display_volume = volume_norm

            slice_idx = quantification.get("primary_slice_index", display_volume.shape[2] // 2)
            slice_idx = max(0, min(slice_idx, display_volume.shape[2] - 1))

            # Axial slice preview
            mri_axial = display_volume[:, :, slice_idx]
            mask_axial = mask[:, :, slice_idx] if mask.ndim == 3 else mask[:, :, :, slice_idx]
            axial_b64 = self._slice_to_overlay_base64(mri_axial, mask_axial)

            # Coronal slice preview
            coronal_mid = display_volume.shape[1] // 2
            mri_coronal = display_volume[:, coronal_mid, :]
            mask_coronal = mask[:, coronal_mid, :] if mask.ndim == 3 else mask[:, :, coronal_mid, :]
            coronal_b64 = self._slice_to_overlay_base64(mri_coronal, mask_coronal)

            subregions = quantification.get("subregions", {})

            result["status"] = "completed"
            result["message"] = f"Rendered clinical-style quantitative research report & slice overlays."
            result["visualizations"] = {
                "axial_overlay": axial_b64,
                "coronal_overlay": coronal_b64,
                "primary_slice_idx": slice_idx
            }
            result["report_summary"] = {
                "disclaimer": "This prototype is intended for research and educational demonstration only and is not a substitute for clinical diagnosis.",
                "whole_tumor_volume_cm3": quantification.get("tumor_volume_cm3", 0.0),
                "tumor_core_volume_cm3": subregions.get("tumor_core", {}).get("tumor_volume_cm3", 0.0) if subregions else 0.0,
                "enhancing_tumor_volume_cm3": subregions.get("enhancing_tumor", {}).get("tumor_volume_cm3", 0.0) if subregions else 0.0,
                "tumor_voxels": quantification.get("tumor_voxels", 0),
                "voxel_spacing_mm": quantification.get("voxel_spacing_mm"),
                "bounding_box": quantification.get("bounding_box")
            }
            return result
        except Exception as e:
            result["status"] = "failed"
            result["message"] = f"Reporting Agent failed: {str(e)}"
            return result
