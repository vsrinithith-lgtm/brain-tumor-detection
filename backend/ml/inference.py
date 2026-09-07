import os
import torch
import numpy as np
from backend.ml.model import UNet3D, BaselineSegmentationModel

class SegmentationEngine:
    """
    Unified Inference Engine for Brain Tumor Segmentation.
    Loads official MONAI Model Zoo SegResNet BraTS pretrained bundle if present.
    Executes pretrained MONAI model ONLY when genuine 4-channel multimodal input (T1c, T1, T2, FLAIR) is provided.
    For single NIfTI scans or missing modalities, executes Baseline Segmentation Engine.
    """
    def __init__(self, checkpoint_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_pretrained = False
        self.model_type = "baseline_segmentation"
        self.model_name = "Baseline Segmentation Engine (Morphological/Intensity-based)"
        self.monai_model = None
        self.baseline_model = BaselineSegmentationModel()

        if checkpoint_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            default_ckpt = os.path.join(project_root, "models", "brats_mri_segmentation", "models", "model.pt")
            if os.path.exists(default_ckpt):
                checkpoint_path = default_ckpt

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                from monai.networks.nets import SegResNet
                model = SegResNet(
                    blocks_down=[1, 2, 2, 4],
                    blocks_up=[1, 1, 1],
                    init_filters=16,
                    in_channels=4,
                    out_channels=3,
                    dropout_prob=0.2
                )
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                elif "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"])
                else:
                    model.load_state_dict(checkpoint)
                    
                model.to(self.device)
                model.eval()
                self.monai_model = model
                self.is_pretrained = True
                print(f"SegmentationEngine: Successfully loaded MONAI BraTS pretrained weights from {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Failed to load MONAI BraTS checkpoint ({e}). Baseline engine active.")

    def run_segmentation(self, volume: np.ndarray) -> tuple:
        """
        Executes segmentation.
        Pretrained MONAI model is ONLY executed when input volume is a genuine 4-channel tensor (4, D, H, W).
        Single 3D NIfTI scans (D, H, W) execute Baseline Segmentation Engine WITHOUT modality replication.

        Returns:
            mask (np.ndarray): 3D binary mask or 4D multiclass mask (3, D, H, W)
            engine_info (dict): dict with keys 'model_type', 'is_pretrained', 'model_name'
        """
        if self.is_pretrained and self.monai_model is not None and volume.ndim == 4 and volume.shape[0] == 4:
            try:
                from monai.inferers import SlidingWindowInferer
                
                # Input shape: (4, D, H, W) -> (1, 4, D, H, W)
                input_tensor = torch.from_numpy(volume).unsqueeze(0).to(self.device)
                
                spatial_shape = volume.shape[1:]
                roi_d = min(128, spatial_shape[0])
                roi_h = min(128, spatial_shape[1])
                roi_w = min(128, spatial_shape[2])
                
                inferer = SlidingWindowInferer(
                    roi_size=(roi_d, roi_h, roi_w),
                    sw_batch_size=1,
                    overlap=0.5
                )
                
                with torch.no_grad():
                    logits = inferer(input_tensor, self.monai_model)
                    probs = torch.sigmoid(logits)
                    mask_3ch = (probs > 0.5).squeeze(0).cpu().numpy().astype(np.uint8)
                    
                    engine_info = {
                        "model_type": "pretrained_monai_brats",
                        "is_pretrained": True,
                        "model_name": "MONAI SegResNet (BraTS MRI Pretrained Bundle)"
                    }
                    return mask_3ch, engine_info
            except Exception as e:
                print(f"Warning: Pretrained MONAI inference failed ({e}). Falling back to baseline engine.")

        # Single scan or fallback -> execute Baseline Segmentation Engine
        if volume.ndim == 4:
            volume = volume[0]
            
        mask_single = self.baseline_model.predict_mask(volume)
        engine_info = {
            "model_type": "baseline_segmentation",
            "is_pretrained": False,
            "model_name": "Baseline Segmentation Engine (Morphological/Intensity-based)"
        }
        return mask_single, engine_info
