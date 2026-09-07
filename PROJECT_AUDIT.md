# Technical Assessment & Project Audit: Brain Tumor MRI Segmentation

**Project Name:** NeuroScan AI (Brain Tumor MRI Segmentation)  
**Audit Date:** September 7, 2026  
**Auditor:** Senior ML Engineer + Full-Stack Engineer + Medical AI Hackathon Mentor  

---

## Executive Summary

The existing repository was inspected and found to be a **React/Vite frontend prototype** built for 2D brain tumor **classification** (PNG/JPG input, binary confidence score display). The Python backend (`backend/main.py` and `backend/requirements.txt`) was **completely empty (0 bytes)**. 

To convert this project into a robust, end-to-end **Automated Brain Tumor Segmentation MVP**, the core workflow must shift from 2D classification to **3D NIfTI volumetric segmentation**, including voxel-level tumor mask generation, volume quantification ($cm^3$), slice visualization overlays, automated Quality Control (QC), and an Agentic Orchestration backend built with Python + FastAPI.

---

## Audit Findings

### 1. What Currently Works?
* **Frontend UI Shell:** Clean Vite + React UI layout with navigation tabs (Dashboard, Analysis, History, About, System Status).
* **UI Design System:** Modern CSS styles in `src/index.css` with responsive dashboard panels and stateful UI components.
* **Local Storage & State:** Browser-based history persistence and system status health check polling logic.

### 2. What Is Broken / Non-Functional?
* **Backend:** `backend/main.py` and `backend/requirements.txt` are completely empty. No API server or ML code existed.
* **Pipeline:** No medical image reader (`nibabel`), preprocessor, segmentation model, or volume quantification logic.
* **API Integration:** Frontend was wired for a classification endpoint (`/predict`), expecting simple `prediction` and `confidence` fields.

### 3. What Can Be Reused?
* **Frontend Shell:** Navbar, Footer, Medical Disclaimer, Local Storage History, System Status UI structure.
* **Vite Config & Styling:** `vite.config.js`, Tailwind/CSS classes, icons from `lucide-react`.

### 4. What Must Be Replaced?
* **Classification Language & Contracts:** Replace "AI classification", "Tumor/No Tumor", and "Confidence Score" with "MRI Segmentation", "Tumor Mask", "Tumor Volume ($cm^3$)", "Voxel Count", and "QC Status".
* **Input Specifications:** Replace 2D PNG/JPG restriction with native support for NIfTI volumetric data (`.nii` and `.nii.gz`).
* **API Endpoints:** Replace `/predict` with `/segment` returning structured segmentation payload, slice preview images (base64 PNG overlays), tumor volume metadata, and workflow agent status.

### 5. What Is Missing for MRI Segmentation?
1. **Medical Data Preprocessing & Validation:** 3D NIfTI loading (`nibabel`), dimension checks, orientation, voxel spacing extraction, intensity normalization, NaN/Inf sanitization.
2. **Segmentation Engine:** Clean ML model interface supporting PyTorch / 3D U-Net / MONAI architectures, with a robust baseline segmentation engine (morphological/intensity feature segmentation fallback when pretrained weights are not present).
3. **Postprocessing & Quantification:** Connected component analysis, noise removal, tumor bounding box, affected slice index range, and exact physical volume calculation ($V = N_{\text{voxels}} \times v_{\text{voxel\_size}}$).
4. **Quality Control Agent:** Validation of generated masks (detecting empty masks, abnormal volumes, preprocessing failures) and ground-truth Dice / IoU evaluation when GT mask is supplied.
5. **Agentic Orchestrator Layer:** Modular python backend agents (`MRIValidationAgent`, `MRIPreprocessingAgent`, `SegmentationAgent`, `QualityControlAgent`, `ReportingAgent`) coordinated via `AgentOrchestrator`.
6. **Slice Overlay Visualization:** Backend rendering of representative Axial, Coronal, and Sagittal slice previews with color-coded semi-transparent tumor mask overlays.

---

## Architectural Transition Plan

```
[ Frontend: React + Vite ]
        │
        ▼ POST /segment (.nii / .nii.gz)
[ FastAPI Backend Engine ]
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                   Agent Orchestrator                      │
│                                                           │
│ 1. MRIValidationAgent   ──> Inspect dimensions, NaNs, affine│
│ 2. MRIPreprocessingAgent──> Rescale, normalize intensity │
│ 3. SegmentationAgent    ──> Execute UNet/Baseline engine │
│ 4. PostprocessingEngine ──> Quantify volume, compute BBox │
│ 5. QualityControlAgent  ──> Check mask validity & metrics │
│ 6. ReportingAgent       ──> Render slice previews & JSON  │
└───────────────────────────────────────────────────────────┘
```

---

## Ground Truth & Honesty Guarantee

* **No Faked Models:** The system will explicitly identify whether a deep neural network checkpoint (e.g., 3D U-Net) or a feature-based prototype segmentation baseline is active.
* **No Invented Metrics:** Dice, IoU, sensitivity, and specificity will only be computed when a ground-truth mask is provided for validation (`POST /evaluate` or GT upload field).
