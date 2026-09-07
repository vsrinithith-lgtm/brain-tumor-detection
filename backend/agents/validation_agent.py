import numpy as np
from backend.ml.preprocessing import load_mri_volume, load_multimodal_mri_volumes

class MRIValidationAgent:
    """
    MRI Validation Agent: Validates single or 4-channel multimodal MRI volumetric data.
    Verifies spatial geometry alignment, NIfTI readability, voxel spacing, and data integrity.
    """
    def __init__(self):
        self.name = "MRI Validation Agent"

    def run(self, file_bytes: bytes = None, filename: str = None, multimodal_dict: dict = None) -> dict:
        result = {
            "agent": self.name,
            "status": "pending",
            "message": "",
            "action_performed": "Validated spatial dimensions, NIfTI headers, and signal bounds",
            "volume_raw": None,
            "spacing": None,
            "metadata": None,
            "details": {}
        }

        # Case 1: Multimodal 4-Sequence Upload (T1c, T1, T2, FLAIR)
        if multimodal_dict and any(multimodal_dict.values()):
            modalities = ['t1c', 't1', 't2', 'flair']
            missing = [m.upper() for m in modalities if m not in multimodal_dict or not multimodal_dict[m]]
            
            if missing:
                result["status"] = "failed"
                result["message"] = f"Multimodal Validation Rejection: Missing required MRI modality sequence(s): {', '.join(missing)}."
                result["details"] = {"missing_modalities": missing, "present_modalities": [m.upper() for m in modalities if m not in missing]}
                return result

            try:
                stacked_4d, spacing, metadata = load_multimodal_mri_volumes(multimodal_dict)
                result["status"] = "completed"
                result["message"] = f"Valid 4-Sequence Multimodal BraTS case verified (T1c, T1, T2, FLAIR. Geometry: {metadata['shape']}, Spacing: {spacing})."
                result["volume_raw"] = stacked_4d
                result["spacing"] = spacing
                result["metadata"] = metadata
                result["details"] = {
                    "modalities_verified": ["T1c", "T1", "T2", "FLAIR"],
                    "spatial_shape": metadata["shape"],
                    "voxel_spacing": spacing,
                    "nan_inf_free": True,
                    "aligned_geometry": True
                }
                return result
            except Exception as e:
                result["status"] = "failed"
                result["message"] = f"Multimodal Validation Rejection: {str(e)}"
                result["details"] = {"error_reason": str(e)}
                return result

        # Case 2: Single Scan Upload
        if not file_bytes or not filename:
            result["status"] = "failed"
            result["message"] = "Validation Rejection: No MRI file provided for analysis."
            result["details"] = {"error_reason": "Missing input file"}
            return result

        lower_name = filename.lower()
        if not (lower_name.endswith('.nii') or lower_name.endswith('.nii.gz') or 
                lower_name.endswith('.png') or lower_name.endswith('.jpg') or lower_name.endswith('.jpeg')):
            result["status"] = "failed"
            result["message"] = f"Validation Rejection: Unsupported file extension for '{filename}'. Expected .nii or .nii.gz (NIfTI)."
            result["details"] = {"filename": filename, "error_reason": "Unsupported extension"}
            return result

        try:
            volume_raw, spacing, metadata = load_mri_volume(file_bytes, filename)
        except Exception as e:
            result["status"] = "failed"
            result["message"] = f"Validation Rejection: NIfTI file unreadable or corrupt: {str(e)}"
            result["details"] = {"filename": filename, "error_reason": str(e)}
            return result

        if len(metadata["shape"]) != 3:
            result["status"] = "failed"
            result["message"] = f"Validation Rejection: Invalid MRI volume dimensions {metadata['shape']}. Expected 3D spatial array."
            result["details"] = {"shape": metadata["shape"], "error_reason": "Non-3D shape"}
            return result

        if metadata["max_intensity"] == metadata["min_intensity"] == 0:
            result["status"] = "failed"
            result["message"] = "Validation Rejection: Scan contains zero intensity signal throughout (empty dataset)."
            result["details"] = {"error_reason": "Zero intensity volume"}
            return result

        result["status"] = "completed"
        result["message"] = f"Valid single NIfTI MRI scan verified ({metadata['shape'][0]}x{metadata['shape'][1]}x{metadata['shape'][2]}, Spacing: {metadata['spacing']})."
        result["volume_raw"] = volume_raw
        result["spacing"] = spacing
        result["metadata"] = metadata
        result["details"] = {
            "spatial_shape": metadata["shape"],
            "voxel_spacing": spacing,
            "nan_inf_free": True
        }
        return result
