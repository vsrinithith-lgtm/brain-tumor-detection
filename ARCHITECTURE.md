# NeuroScan AI System Architecture: Automated Brain Tumor MRI Segmentation

## Overview

NeuroScan AI is a modular medical image analysis platform designed for automated 3D brain tumor segmentation, quantitative volume measurement, and interactive visual overlay rendering from MRI NIfTI scans (`.nii` / `.nii.gz`).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           React + Vite Frontend                             │
│       (Dual Upload Modes · Mask Opacity Slider · Quantitative Dashboard)    │
└──────────────────────────────────────────────────┬──────────────────────────┘
                                                   │ POST /segment (multipart)
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FastAPI Backend Server                           │
│                          (CORS · Router · Schemas)                          │
└──────────────────────────────────────────────────┬──────────────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Agent Orchestrator                             │
│                  (Sequential 5-Agent Pipeline Execution)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. MRI Validation Agent     ──> Check modalities, spatial alignment, NaNs  │
│ 2. Preprocessing Agent      ──> Channel-wise percentile & Z-score scaling   │
│ 3. Segmentation Agent       ──> Execute MONAI SegResNet or Baseline Engine  │
│ 4. Quality Control Agent    ──> Mask bounds check & GT evaluation (Dice/IoU)│
│ 5. Result & Reporting Agent ──> Render PNG slice overlays (base64) & JSON   │
└──────────────────────────────────────────────────┬──────────────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Postprocessing & Volume Quantification                   │
│  - Whole Tumor, Tumor Core, Enhancing Tumor Volume Breakdown                │
│  - Tumor Volume: V = N_voxels * (dx * dy * dz) / 1000.0 (cm³)                │
│  - 3D Bounding Box: [min_x, max_x, min_y, max_y, min_z, max_z]              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Modular Component Breakdown

### 1. Frontend Layer (`src/`)
- **Dual Upload Workspace**:
  - Option 1: Multimodal 4-Sequence BraTS Case (`file_t1c`, `file_t1`, `file_t2`, `file_flair`).
  - Option 2: Single Scan Baseline (`file`).
- **Interactive Visualization**: Rendered Axial and Coronal slice previews with semi-transparent tumor overlays, boundary contours, and an opacity control slider ($10\% - 100\%$).
- **1-Click Demo Mode**: Generates synthetic 4-sequence BraTS data for instant live pipeline demonstration.

### 2. FastAPI Backend Layer (`backend/api/`)
- `GET /health`: Health diagnostic route.
- `POST /segment`: Multipart route accepting single or 4-channel NIfTI files and optional GT mask.
- `POST /predict`: Legacy alias route mapping to `/segment`.
- `POST /evaluate`: Evaluation route for GT vs prediction masks.

### 3. Agentic Orchestration Layer (`backend/agents/`)
- `MRIValidationAgent`: Validates format integrity, spatial alignment across modalities, and checks for corrupted NaNs/Infs or zero-signal scans.
- `MRIPreprocessingAgent`: Executes 1st-99th percentile clipping and Z-score intensity scaling.
- `SegmentationAgent`: Executes official MONAI SegResNet BraTS model (or Baseline Engine fallback).
- `QualityControlAgent`: Evaluates mask bounds, component counts, volume ratios, and computes Dice / IoU / Sensitivity / Specificity when Ground Truth mask is supplied.
- `ReportingAgent`: Renders base64 PNG slice preview overlays and compiles a clinical-style quantitative report.

### 4. ML Engine Layer (`backend/ml/`)
- `SegmentationEngine`: Unified engine managing official MONAI Model Zoo SegResNet model weights (`models/brats_mri_segmentation/models/model.pt`) and `BaselineSegmentationModel` fallback.
- `UNet3D`: PyTorch 3D Convolutional Neural Network architecture.
- `BaselineSegmentationModel`: Algorithmic baseline model utilizing hyper-intense MRI lesion thresholding and 3D binary morphology.
