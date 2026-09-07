from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Optional
from backend.api.schemas import HealthResponse
from backend.agents.orchestrator import AgentOrchestrator

router = APIRouter()
orchestrator = AgentOrchestrator()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="NeuroScan AI Segmentation Backend",
        version="1.0.0"
    )

@router.post("/segment")
async def segment_mri(
    file: Optional[UploadFile] = File(None),
    gt_file: Optional[UploadFile] = File(None),
    file_t1c: Optional[UploadFile] = File(None),
    file_t1: Optional[UploadFile] = File(None),
    file_t2: Optional[UploadFile] = File(None),
    file_flair: Optional[UploadFile] = File(None)
):
    gt_bytes = await gt_file.read() if gt_file else None

    # Check if 4-channel multimodal upload is provided
    if file_t1c and file_t1 and file_t2 and file_flair:
        t1c_bytes = await file_t1c.read()
        t1_bytes = await file_t1.read()
        t2_bytes = await file_t2.read()
        flair_bytes = await file_flair.read()

        multimodal_dict = {
            "t1c": {"bytes": t1c_bytes, "filename": file_t1c.filename},
            "t1": {"bytes": t1_bytes, "filename": file_t1.filename},
            "t2": {"bytes": t2_bytes, "filename": file_t2.filename},
            "flair": {"bytes": flair_bytes, "filename": file_flair.filename}
        }

        result = orchestrator.process_mri(gt_bytes=gt_bytes, multimodal_dict=multimodal_dict)
    elif file and file.filename:
        file_bytes = await file.read()
        result = orchestrator.process_mri(file_bytes=file_bytes, filename=file.filename, gt_bytes=gt_bytes)
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing file upload. Please provide a single MRI file or all 4 multimodal sequences (file_t1c, file_t1, file_t2, file_flair)."
        )

    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Segmentation failed."))

    return result

@router.post("/predict")
async def predict_legacy_alias(
    file: Optional[UploadFile] = File(None),
    gt_file: Optional[UploadFile] = File(None),
    file_t1c: Optional[UploadFile] = File(None),
    file_t1: Optional[UploadFile] = File(None),
    file_t2: Optional[UploadFile] = File(None),
    file_flair: Optional[UploadFile] = File(None)
):
    return await segment_mri(file=file, gt_file=gt_file, file_t1c=file_t1c, file_t1=file_t1, file_t2=file_t2, file_flair=file_flair)

@router.post("/evaluate")
async def evaluate_masks(
    pred_file: UploadFile = File(...),
    gt_file: UploadFile = File(...)
):
    pred_bytes = await pred_file.read()
    gt_bytes = await gt_file.read()

    result = orchestrator.process_mri(file_bytes=pred_bytes, filename=pred_file.filename, gt_bytes=gt_bytes)
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Evaluation failed."))

    return {
        "success": True,
        "evaluation_metrics": result.get("quality_control", {}).get("evaluation_metrics")
    }
