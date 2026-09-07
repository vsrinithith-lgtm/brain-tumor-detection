from backend.agents.validation_agent import MRIValidationAgent
from backend.agents.preprocessing_agent import MRIPreprocessingAgent
from backend.agents.segmentation_agent import SegmentationAgent
from backend.agents.quality_control_agent import QualityControlAgent
from backend.agents.reporting_agent import ReportingAgent

class AgentOrchestrator:
    """
    Modular Agentic Workflow Pipeline Orchestrator for Brain Tumor MRI Segmentation.
    Coordinates sequential execution of validation, preprocessing, segmentation,
    quality control, quantification, and reporting agents.
    Supports Pretrained MONAI SegResNet BraTS model for 4-sequence cases and Baseline Engine fallback.
    """
    def __init__(self, checkpoint_path: str = None):
        self.validator = MRIValidationAgent()
        self.preprocessor = MRIPreprocessingAgent()
        self.segmenter = SegmentationAgent(checkpoint_path=checkpoint_path)
        self.qc = QualityControlAgent()
        self.reporter = ReportingAgent()

    def process_mri(self, file_bytes: bytes = None, filename: str = None, gt_bytes: bytes = None, multimodal_dict: dict = None) -> dict:
        workflow_log = []

        # 1. Validation Stage
        val_res = self.validator.run(file_bytes=file_bytes, filename=filename, multimodal_dict=multimodal_dict)
        workflow_log.append({
            "agent": val_res["agent"],
            "status": val_res["status"],
            "message": val_res["message"],
            "action_performed": val_res.get("action_performed", ""),
            "details": val_res.get("details", {})
        })

        if val_res["status"] == "failed":
            return {
                "success": False,
                "error": val_res["message"],
                "prediction_type": "segmentation",
                "segmentation_status": "failed",
                "model_type": self.segmenter.engine.model_type,
                "is_pretrained": self.segmenter.engine.is_pretrained,
                "workflow": workflow_log
            }

        volume_raw = val_res["volume_raw"]
        spacing = val_res["spacing"]
        metadata = val_res["metadata"]

        # Parse Ground Truth File if provided
        gt_volume = None
        if gt_bytes:
            gt_val = self.validator.run(file_bytes=gt_bytes, filename="ground_truth.nii")
            if gt_val["status"] == "completed":
                gt_volume = gt_val["volume_raw"]

        # 2. Preprocessing Stage
        prep_res = self.preprocessor.run(volume_raw, spacing, metadata)
        workflow_log.append({
            "agent": prep_res["agent"],
            "status": prep_res["status"],
            "message": prep_res["message"],
            "action_performed": prep_res.get("action_performed", ""),
            "details": prep_res.get("details", {})
        })

        if prep_res["status"] == "failed":
            return {
                "success": False,
                "error": prep_res["message"],
                "prediction_type": "segmentation",
                "segmentation_status": "failed",
                "model_type": self.segmenter.engine.model_type,
                "is_pretrained": self.segmenter.engine.is_pretrained,
                "workflow": workflow_log
            }

        volume_norm = prep_res["volume_norm"]

        # 3. Segmentation Stage
        seg_res = self.segmenter.run(volume_norm, spacing)
        workflow_log.append({
            "agent": seg_res["agent"],
            "status": seg_res["status"],
            "message": seg_res["message"],
            "action_performed": seg_res.get("action_performed", ""),
            "details": seg_res.get("details", {})
        })

        if seg_res["status"] == "failed":
            return {
                "success": False,
                "error": seg_res["message"],
                "prediction_type": "segmentation",
                "segmentation_status": "failed",
                "model_type": self.segmenter.engine.model_type,
                "is_pretrained": self.segmenter.engine.is_pretrained,
                "workflow": workflow_log
            }

        mask = seg_res["mask"]
        quant = seg_res["quantification"]

        # 4. Quality Control Stage
        qc_shape = volume_norm.shape[1:] if volume_norm.ndim == 4 else volume_norm.shape
        qc_res = self.qc.run(qc_shape, mask, quant, gt_mask=gt_volume)
        workflow_log.append({
            "agent": qc_res["agent"],
            "status": qc_res["status"],
            "message": qc_res["message"],
            "action_performed": qc_res.get("action_performed", ""),
            "details": qc_res.get("details", {})
        })

        # 5. Reporting Stage
        rep_res = self.reporter.run(volume_norm, mask, quant)
        workflow_log.append({
            "agent": rep_res["agent"],
            "status": rep_res["status"],
            "message": rep_res["message"],
            "action_performed": rep_res.get("action_performed", ""),
            "details": rep_res.get("details", {})
        })

        return {
            "success": True,
            "prediction_type": "segmentation",
            "segmentation_status": "completed",
            "model_type": seg_res["engine_info"]["model_type"],
            "is_pretrained": seg_res["engine_info"]["is_pretrained"],
            "model_info": seg_res["engine_info"],
            "tumor_detected": quant.get("tumor_detected", False),
            "tumor_volume_cm3": quant.get("tumor_volume_cm3", 0.0),
            "tumor_volume_mm3": quant.get("tumor_volume_mm3", 0.0),
            "tumor_voxels": quant.get("tumor_voxels", 0),
            "bounding_box": quant.get("bounding_box"),
            "voxel_spacing_mm": quant.get("voxel_spacing_mm", list(spacing)),
            "affected_slices": quant.get("affected_slices", {}),
            "subregions": quant.get("subregions"),
            "quality_control": {
                "status": qc_res["qc_status"],
                "message": qc_res["message"],
                "evaluation_metrics": qc_res["metrics"]
            },
            "visualizations": rep_res["visualizations"],
            "report_summary": rep_res["report_summary"],
            "metadata": metadata,
            "workflow": workflow_log
        }
