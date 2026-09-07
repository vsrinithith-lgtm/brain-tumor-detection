import io
import os
import tempfile
import unittest
import numpy as np
import nibabel as nib
from fastapi.testclient import TestClient
from backend.main import app

def create_nifti_bytes(shape=(32, 32, 16), zero_signal=False):
    data = np.zeros(shape, dtype=np.float32)
    if not zero_signal:
        cz, cy, cx = shape[2] // 2, shape[1] // 2, shape[0] // 2
        z, y, x = np.ogrid[:shape[2], :shape[1], :shape[0]]
        brain_mask = ((x - cx)**2 + (y - cy)**2 + (z - cz)**2) <= 10**2
        data[brain_mask.transpose(2, 1, 0)] = 150.0

    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nifti_img = nib.Nifti1Image(data, affine)
    
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        nib.save(nifti_img, tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        file_bytes = f.read()

    os.remove(tmp_path)
    return file_bytes

class TestFailureScenarios(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_missing_multimodal_sequence(self):
        t1c = create_nifti_bytes()
        t1 = create_nifti_bytes()
        # Missing T2 and FLAIR
        files = {
            "file_t1c": ("t1c.nii", io.BytesIO(t1c), "application/octet-stream"),
            "file_t1": ("t1.nii", io.BytesIO(t1), "application/octet-stream"),
        }
        response = self.client.post("/segment", files=files)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("multimodal", data["detail"].lower())

    def test_mismatched_dimensions(self):
        t1c = create_nifti_bytes(shape=(32, 32, 16))
        t1 = create_nifti_bytes(shape=(32, 32, 16))
        t2 = create_nifti_bytes(shape=(32, 32, 16))
        flair_mismatched = create_nifti_bytes(shape=(64, 64, 16))

        files = {
            "file_t1c": ("t1c.nii", io.BytesIO(t1c), "application/octet-stream"),
            "file_t1": ("t1.nii", io.BytesIO(t1), "application/octet-stream"),
            "file_t2": ("t2.nii", io.BytesIO(t2), "application/octet-stream"),
            "file_flair": ("flair.nii", io.BytesIO(flair_mismatched), "application/octet-stream"),
        }
        response = self.client.post("/segment", files=files)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("mismatch", data["detail"].lower())

    def test_invalid_file_extension(self):
        files = {"file": ("document.txt", io.BytesIO(b"Not an MRI scan"), "text/plain")}
        response = self.client.post("/segment", files=files)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("unsupported file extension", data["detail"].lower())

    def test_zero_intensity_volume(self):
        empty_nii = create_nifti_bytes(zero_signal=True)
        files = {"file": ("empty.nii", io.BytesIO(empty_nii), "application/octet-stream")}
        response = self.client.post("/segment", files=files)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("zero intensity", data["detail"].lower())

if __name__ == '__main__':
    unittest.main()
