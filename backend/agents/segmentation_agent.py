from backend.ml.inference import SegmentationEngine
from backend.ml.postprocessing import postprocess_mask, quantify_tumor

class SegmentationAgent:
    """
    Segmentation Agent: Invokes the Segmentation Engine.
    Explicitly reports:
    - Model: Official MONAI BraTS SegResNet (Pretrained) OR Baseline Segmentation Engine
    - Input: T1c + T1 + T2 + FLAIR (or single NIfTI scan)
    - Output: TC + WT + ET (or single binary mask)
    Never claims the baseline model is pretrained.
    """
    def __init__(self, checkpoint_path: str = None):
        self.name = "Segmentation Agent"
        self.engine = SegmentationEngine(checkpoint_path=checkpoint_path)

    def run(self, volume_norm, spacing) -> dict:
        result = {
            "agent": self.name,
            "status": "pending",
            "message": "",
            "action_performed": "Executed volumetric tumor delineation",
            "mask": None,
            "quantification": None,
            "engine_info": None,
            "details": {}
        }

        try:
            raw_mask, engine_info = self.engine.run_segmentation(volume_norm)
            clean_mask = postprocess_mask(raw_mask)
            quant_data = quantify_tumor(clean_mask, spacing)

            if quant_data.get("tumor_detected"):
                msg = f"Segmentation completed via {engine_info['model_name']}. Tumor volume: {quant_data['tumor_volume_cm3']} cm³ ({quant_data['tumor_voxels']} voxels)."
            else:
                msg = f"Segmentation completed via {engine_info['model_name']}. No hyper-intense tumor region identified."

            is_pretrained = engine_info["is_pretrained"]
            details = {
                "model_name": engine_info["model_name"],
                "model_type": engine_info["model_type"],
                "is_pretrained": is_pretrained,
                "input_configuration": "T1c + T1 + T2 + FLAIR (4 Multimodal Sequences)" if is_pretrained else "Single NIfTI Scan",
                "output_channels": "Whole Tumor (WT) + Tumor Core (TC) + Enhancing Tumor (ET)" if is_pretrained else "Single Binary Mask",
                "tumor_detected": quant_data.get("tumor_detected", False)
            }

            result["status"] = "completed"
            result["message"] = msg
            result["mask"] = clean_mask
            result["quantification"] = quant_data
            result["engine_info"] = engine_info
            result["details"] = details
            return result
        except Exception as e:
            result["status"] = "failed"
            result["message"] = f"Segmentation Rejection: {str(e)}"
            result["details"] = {"error_reason": str(e)}
            return result
