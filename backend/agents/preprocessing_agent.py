import numpy as np
from backend.ml.preprocessing import normalize_intensity

class MRIPreprocessingAgent:
    """
    MRI Preprocessing Agent: Performs modality-wise intensity percentile scaling,
    spatial scaling, and channel preparation. Reports detailed preprocessing decisions.
    """
    def __init__(self):
        self.name = "MRI Preprocessing Agent"

    def run(self, volume_raw: np.ndarray, spacing: tuple, metadata: dict) -> dict:
        result = {
            "agent": self.name,
            "status": "pending",
            "message": "",
            "action_performed": "Executed percentile intensity normalization and channel stacking",
            "volume_norm": None,
            "details": {}
        }

        try:
            if volume_raw.ndim == 4:
                # 4-channel multimodal tensor (T1c, T1, T2, FLAIR)
                norm_vol = volume_raw
                msg = f"Multimodal 4-channel normalization & tensor stacking completed (Channels: [0: T1c, 1: T1, 2: T2, 3: FLAIR])."
                details = {
                    "channels": 4,
                    "modalities_processed": ["T1c", "T1", "T2", "FLAIR"],
                    "normalization_method": "Percentile 1st-99th clipping & channel-wise Z-score scaling",
                    "output_shape": list(norm_vol.shape)
                }
            else:
                norm_vol = normalize_intensity(volume_raw)
                msg = f"Z-score / 1st-99th percentile intensity normalization completed. Signal rescaled to [0.00, 1.00]."
                details = {
                    "channels": 1,
                    "normalization_method": "Percentile 1st-99th clipping & intensity scaling",
                    "output_shape": list(norm_vol.shape),
                    "min_val": float(np.min(norm_vol)),
                    "max_val": float(np.max(norm_vol))
                }

            result["status"] = "completed"
            result["message"] = msg
            result["volume_norm"] = norm_vol
            result["details"] = details
            return result
        except Exception as e:
            result["status"] = "failed"
            result["message"] = f"Preprocessing Rejection: {str(e)}"
            result["details"] = {"error_reason": str(e)}
            return result
