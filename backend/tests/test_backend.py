import io
import os
import tempfile
import unittest
import numpy as np
import nibabel as nib
from fastapi.testclient import TestClient

from backend.main import app
from backend.ml.preprocessing import load_mri_volume, normalize_intensity
from backend.ml.postprocessing import quantify_tumor, calculate_evaluation_metrics
from backend.agents.orchestrator import AgentOrchestrator

def create_synthetic_nifti_bytes(shape=(32, 32, 16), add_tumor=True):
    data = np.zeros(shape, dtype=np.float32)
    cz, cy, cx = shape[2] // 2, shape[1] // 2, shape[0] // 2
    z, y, x = np.ogrid[:shape[2], :shape[1], :shape[0]]
    brain_mask = ((x - cx)**2 + (y - cy)**2 + (z - cz)**2) <= 12**2
    data[brain_mask.transpose(2, 1, 0)] = 100.0

    if add_tumor:
        tumor_mask = ((x - (cx + 4))**2 + (y - (cy + 4))**2 + (z - cz)**2) <= 4**2
        data[tumor_mask.transpose(2, 1, 0)] = 350.0

    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nifti_img = nib.Nifti1Image(data, affine)
    
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        nib.save(nifti_img, tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        file_bytes = f.read()

    os.remove(tmp_path)
    return file_bytes

class TestBackendSegmentation(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("NeuroScan", data["service"])

    def test_nifti_preprocessing(self):
        nii_bytes = create_synthetic_nifti_bytes()
        volume, spacing, metadata = load_mri_volume(nii_bytes, "test_scan.nii")
        self.assertEqual(volume.shape, (32, 32, 16))
        self.assertEqual(spacing, (1.0, 1.0, 1.0))
        self.assertEqual(metadata["voxel_volume_mm3"], 1.0)

        norm_vol = normalize_intensity(volume)
        self.assertLessEqual(np.max(norm_vol), 1.0)
        self.assertGreaterEqual(np.min(norm_vol), 0.0)

    def test_tumor_quantification(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1 # 1000 voxels
        spacing = (1.0, 1.0, 1.0)
        
        quant = quantify_tumor(mask, spacing)
        self.assertTrue(quant["tumor_detected"])
        self.assertEqual(quant["tumor_voxels"], 1000)
        self.assertEqual(quant["tumor_volume_mm3"], 1000.0)
        self.assertEqual(quant["tumor_volume_cm3"], 1.0)

    def test_evaluation_metrics(self):
        pred = np.zeros((10, 10, 10), dtype=np.uint8)
        gt = np.zeros((10, 10, 10), dtype=np.uint8)
        pred[2:8, 2:8, 2:8] = 1
        gt[2:8, 2:8, 2:8] = 1
        
        metrics = calculate_evaluation_metrics(pred, gt)
        self.assertEqual(metrics["dice_score"], 1.0)
        self.assertEqual(metrics["iou_score"], 1.0)

    def test_agent_orchestrator(self):
        orchestrator = AgentOrchestrator()
        nii_bytes = create_synthetic_nifti_bytes(add_tumor=True)
        res = orchestrator.process_mri(nii_bytes, "synthetic_brain.nii")
        
        self.assertTrue(res["success"])
        self.assertEqual(res["prediction_type"], "segmentation")
        self.assertEqual(res["segmentation_status"], "completed")
        self.assertEqual(len(res["workflow"]), 5)
        self.assertIn("visualizations", res)
        self.assertIn("axial_overlay", res["visualizations"])
        self.assertIn(res["quality_control"]["status"], ["PASS", "WARNING"])

    def test_api_segment_endpoint(self):
        nii_bytes = create_synthetic_nifti_bytes()
        files = {"file": ("scan.nii", io.BytesIO(nii_bytes), "application/octet-stream")}
        response = self.client.post("/segment", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("tumor_volume_cm3", data)
        self.assertIn("workflow", data)

if __name__ == '__main__':
    unittest.main()
