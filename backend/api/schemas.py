from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    service: str = Field(..., example="NeuroScan AI Segmentation Backend")
    version: str = Field(..., example="1.0.0")

class WorkflowStep(BaseModel):
    agent: str
    status: str
    message: str
    action_performed: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class QualityControlReport(BaseModel):
    status: str
    message: str
    evaluation_metrics: Optional[Dict[str, Any]] = None

class ModelInfo(BaseModel):
    model_type: str
    is_pretrained: bool
    model_name: str

class SegmentResponse(BaseModel):
    success: bool
    prediction_type: str = "segmentation"
    segmentation_status: str
    model_type: str
    is_pretrained: bool
    model_info: ModelInfo
    tumor_detected: bool
    tumor_volume_cm3: float
    tumor_volume_mm3: float
    tumor_voxels: int
    bounding_box: Optional[Dict[str, List[int]]] = None
    voxel_spacing_mm: List[float]
    affected_slices: Dict[str, List[int]]
    subregions: Optional[Dict[str, Any]] = None
    quality_control: QualityControlReport
    visualizations: Dict[str, Any]
    report_summary: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]
    workflow: List[WorkflowStep]
    error: Optional[str] = None
