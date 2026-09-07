# NeuroScan AI: Automated Brain Tumor MRI Segmentation Engine

An end-to-end medical AI application for **automated 3D brain tumor segmentation, quantitative volume measurement, and interactive visual overlay delineations** from MRI NIfTI scans (`.nii` / `.nii.gz`).

Built with **React + Vite** on the frontend and a **Python + FastAPI + PyTorch / MONAI** 5-Agent orchestrator backend.

---

## 🌟 Key Features

- 🧠 **Dual Segmentation Modes**:
  - **Option 1: Multimodal 4-Sequence Case (Pretrained MONAI SegResNet)**: Processes T1c, T1, T2, and FLAIR NIfTI scans to segment Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) subregions.
  - **Option 2: Single Scan MRI (Baseline Engine)**: Delineates hyper-intense MRI lesion regions using adaptive intensity thresholding and 3D binary morphology when single scans are uploaded.
- 🤖 **5-Agent Pipeline Orchestration**:
  1. `MRI Validation Agent`: Validates format integrity, spatial geometry alignment, and checks for corrupted NaNs/Infs or zero-signal scans.
  2. `Multimodal Preprocessing Agent`: Performs 1st-99th percentile clipping and Z-score intensity scaling.
  3. `Segmentation Agent`: Executes official MONAI SegResNet BraTS model (or Baseline Engine fallback).
  4. `Quality Control Agent`: Evaluates mask bounds, component counts, volume ratios, and computes Dice / IoU / Sensitivity / Specificity when Ground Truth mask is supplied.
  5. `Result & Reporting Agent`: Renders base64 PNG slice preview overlays and compiles a clinical-style quantitative report.
- 📐 **Exact Voxel-Level Quantification**: Calculates physical tumor volumes ($cm^3$), voxel counts, 3D Bounding Boxes $[X, Y, Z]$, and affected slice indices using real NIfTI header voxel spacing ($V = N_{\text{voxels}} \times \frac{dx \cdot dy \cdot dz}{1000}$).
- 👁️ **Interactive Visual Overlays**: Rendered Axial and Coronal slice previews with semi-transparent tumor mask overlays, boundary contour lines, and an **Interactive Opacity Slider** ($10\% - 100\%$).
- 🧪 **Ground Truth Evaluation**: Calculates Dice, IoU, predicted volume, GT volume, and volume difference ONLY when a ground-truth mask is provided (never fabricates metrics).
- 🚀 **1-Click Hackathon Demo Mode**: Includes a "Load Demo Case" button to load a pre-configured sample 4-sequence case for instant live pipeline demonstration.

---

## 🏗️ System Architecture

```
React + Vite UI  ──> POST /segment ──> FastAPI Backend ──> AgentOrchestrator
                                                               │
        ┌──────────────────────────────────────────────────────┴──────────────────────────────────────┐
        │                                                                                             │
 1. Validation Agent ──> 2. Preprocessing Agent ──> 3. Segmentation Agent ──> 4. Quality Control Agent ──> 5. Reporting Agent
```

---

## 🚀 Quickstart Guide

### Prerequisites
- Node.js (v18+)
- Python 3.11+

### 1. Start FastAPI Backend

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Start backend server from project root
python -m uvicorn backend.main:app --reload --port 8000
```
Backend will run at `http://localhost:8000`. Interactive API Docs are available at `http://localhost:8000/docs`.

### 2. Start React Frontend

```bash
# Install Node dependencies
npm install

# Start Vite dev server
npm run dev
```
Frontend will run at `http://localhost:5173`.

---

## 🧪 Testing & Verification

Run the full backend test suite (unit tests, integration tests, and failure demo scenario tests):

```bash
python -m unittest discover -s backend/tests -p "test_*.py"
```

To run end-to-end API contract tests:

```bash
python scratch/verify_final_strict_contract.py
```

To verify frontend production compilation:

```bash
npm run build
```

---

## 📡 API Reference

### `GET /health`
Returns backend operational status.

### `POST /segment`
Multipart form endpoint accepting:
- **Multimodal Upload**: `file_t1c`, `file_t1`, `file_t2`, `file_flair` (4 `.nii` / `.nii.gz` files) $\rightarrow$ Triggers Pretrained MONAI SegResNet.
- **Single Scan Upload**: `file` (single `.nii` / `.nii.gz` file) $\rightarrow$ Triggers Baseline Segmentation Engine.
- **Ground Truth Upload**: `gt_file` (optional `.nii` / `.nii.gz` file for Dice evaluation).

---

## 🛡️ Medical & Ethical Disclaimer

*This prototype is intended for research and educational demonstration only and is not a substitute for clinical diagnosis or radiologist evaluation.*
