"""
Verification tests to compare new implementation against original code.

These tests ensure that the refactored code produces identical outputs
to the original TUS-REC2025-Challenge_baseline implementation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import pytest


class TestConstants:
    """Test that constants match expected values."""

    def test_image_dimensions(self):
        from usrec_zoo.constants import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_PIXELS

        assert IMAGE_HEIGHT == 480
        assert IMAGE_WIDTH == 640
        assert NUM_PIXELS == 480 * 640 == 307200

    def test_euler_convention(self):
        from usrec_zoo.constants import EULER_CONVENTION

        assert EULER_CONVENTION == "ZYX"


class TestCalibration:
    """Test calibration loading matches original."""

    @pytest.fixture
    def calib_path(self):
        """Get path to calibration file."""
        paths = [
            Path("TUS-REC2025-Challenge_baseline/data/calib_matrix.csv"),
            Path("data/calib_matrix.csv"),
        ]
        for p in paths:
            if p.exists():
                return p
        pytest.skip("Calibration file not found")

    def test_calibration_loading(self, calib_path):
        """Test that calibration loads correctly."""
        from usrec_zoo.calibration import CalibrationData

        calib = CalibrationData.from_csv(calib_path)

        assert calib.tform_pixel_to_mm.shape == (4, 4)
        assert calib.tform_mm_to_tool.shape == (4, 4)
        assert calib.tform_pixel_to_tool.shape == (4, 4)

    def test_calibration_matches_original(self, calib_path):
        """Test that calibration values match original implementation."""
        from usrec_zoo.calibration import CalibrationData

        # Load with new code
        calib = CalibrationData.from_csv(calib_path)

        # Load with original code pattern
        tform_calib = np.empty((8, 4), np.float32)
        with open(calib_path, "r") as csv_file:
            txt = [i.strip("\n").split(",") for i in csv_file.readlines()]
            tform_calib[0:4, :] = np.array(txt[1:5]).astype(np.float32)
            tform_calib[4:8, :] = np.array(txt[6:10]).astype(np.float32)

        original_pixel_to_mm = tform_calib[0:4, :]
        original_mm_to_tool = tform_calib[4:8, :]

        np.testing.assert_array_almost_equal(
            calib.tform_pixel_to_mm, original_pixel_to_mm
        )
        np.testing.assert_array_almost_equal(
            calib.tform_mm_to_tool, original_mm_to_tool
        )


class TestTransforms:
    """Test transform utilities match original."""

    def test_params_to_matrix_identity(self):
        """Test that zero params produce identity transform."""
        from usrec_zoo.transforms import params_to_matrix

        params = torch.zeros(6)
        matrix = params_to_matrix(params)

        expected = torch.eye(4)
        torch.testing.assert_close(matrix, expected, atol=1e-6, rtol=1e-6)

    def test_params_to_matrix_batched(self):
        """Test batched params to matrix conversion."""
        from usrec_zoo.transforms import params_to_matrix

        params = torch.zeros(10, 6)
        matrices = params_to_matrix(params)

        assert matrices.shape == (10, 4, 4)

        # All should be identity
        for i in range(10):
            torch.testing.assert_close(
                matrices[i], torch.eye(4), atol=1e-6, rtol=1e-6
            )

    def test_matrix_to_params_roundtrip(self):
        """Test matrix to params roundtrip."""
        from usrec_zoo.transforms import params_to_matrix, matrix_to_params

        # Random small parameters
        params_orig = torch.randn(6) * 0.1
        matrix = params_to_matrix(params_orig)
        params_recovered = matrix_to_params(matrix)

        torch.testing.assert_close(params_recovered, params_orig, atol=1e-5, rtol=1e-5)

    def test_params_to_matrix_matches_pytorch3d(self):
        """Test that conversion matches pytorch3d directly."""
        import pytorch3d.transforms
        from usrec_zoo.transforms import params_to_matrix

        params = torch.randn(6) * 0.1
        euler = params[:3]
        translation = params[3:]

        # Our implementation
        matrix_ours = params_to_matrix(params)

        # Direct pytorch3d
        rotation = pytorch3d.transforms.euler_angles_to_matrix(euler, "ZYX")
        matrix_direct = torch.eye(4)
        matrix_direct[:3, :3] = rotation
        matrix_direct[:3, 3] = translation

        torch.testing.assert_close(matrix_ours, matrix_direct, atol=1e-6, rtol=1e-6)


class TestTransformAccumulation:
    """Test transform accumulation matches original."""

    def test_accumulation_identity(self):
        """Test accumulation of identity transforms."""
        from usrec_zoo.transforms.accumulation import TransformAccumulator

        # 5 identity transforms
        local = torch.eye(4).unsqueeze(0).repeat(5, 1, 1)
        global_transforms = TransformAccumulator.accumulate_local_to_global(local)

        # Global should also be identity
        for i in range(5):
            torch.testing.assert_close(
                global_transforms[i], torch.eye(4), atol=1e-6, rtol=1e-6
            )

    def test_accumulation_composition(self):
        """Test that accumulation correctly composes transforms."""
        from usrec_zoo.transforms.accumulation import TransformAccumulator
        from usrec_zoo.transforms import params_to_matrix

        # Create simple transforms
        params1 = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # translate x by 1
        params2 = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # translate x by 1

        local = torch.stack([
            params_to_matrix(params1),
            params_to_matrix(params2),
        ])

        global_transforms = TransformAccumulator.accumulate_local_to_global(local)

        # First global = first local
        torch.testing.assert_close(
            global_transforms[0], local[0], atol=1e-6, rtol=1e-6
        )

        # Second global should translate by 2
        expected_translation = torch.tensor([2.0, 0.0, 0.0])
        torch.testing.assert_close(
            global_transforms[1, :3, 3], expected_translation, atol=1e-6, rtol=1e-6
        )


class TestReferenceImagePoints:
    """Test reference image point generation."""

    def test_reference_points_shape(self):
        """Test shape of reference image points."""
        from usrec_zoo.evaluation.ddf import reference_image_points

        points = reference_image_points(480, 640)

        assert points.shape == (4, 307200)

    def test_reference_points_homogeneous(self):
        """Test that points are homogeneous (last row is 1)."""
        from usrec_zoo.evaluation.ddf import reference_image_points

        points = reference_image_points(480, 640)

        # Last row should be all 1s
        torch.testing.assert_close(
            points[3, :],
            torch.ones(307200),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_reference_points_z_zero(self):
        """Test that z coordinates are 0."""
        from usrec_zoo.evaluation.ddf import reference_image_points

        points = reference_image_points(480, 640)

        # z (row 2) should be all 0s
        torch.testing.assert_close(
            points[2, :],
            torch.zeros(307200),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_reference_points_order(self):
        """Test that points are in expected order with x varying fastest.

        The output format is [x, y, z, 1] where x (column) varies fastest.
        This matches the original TUS-REC implementation's cartesian_prod order.
        """
        from usrec_zoo.evaluation.ddf import reference_image_points

        # Use small size for testing
        points = reference_image_points(3, 4)  # 3 rows, 4 cols = 12 points

        # First point should be (x=1, y=1)
        assert points[0, 0].item() == 1.0  # x
        assert points[1, 0].item() == 1.0  # y

        # Second point should be (x=2, y=1) - x varies fastest
        assert points[0, 1].item() == 2.0  # x
        assert points[1, 1].item() == 1.0  # y

        # Fifth point should be (x=1, y=2) - first of second row
        assert points[0, 4].item() == 1.0  # x
        assert points[1, 4].item() == 2.0  # y


class TestTypes:
    """Test type dataclasses."""

    def test_scan_data_validation(self):
        """Test ScanData validates inputs."""
        from usrec_zoo.types import ScanData

        frames = np.zeros((10, 480, 640), dtype=np.uint8)
        transforms = np.zeros((10, 4, 4), dtype=np.float32)

        scan = ScanData(
            frames=frames,
            transforms=transforms,
            subject_id="000",
            scan_name="test_scan",
        )

        assert scan.num_frames == 10

    def test_scan_data_invalid_shape(self):
        """Test ScanData rejects invalid shapes."""
        from usrec_zoo.types import ScanData

        frames = np.zeros((10, 100, 100), dtype=np.uint8)  # Wrong size

        with pytest.raises(ValueError):
            ScanData(
                frames=frames,
                transforms=None,
                subject_id="000",
                scan_name="test",
            )

    def test_transform_prediction_validation(self):
        """Test TransformPrediction validates inputs."""
        from usrec_zoo.types import TransformPrediction

        local = torch.zeros(9, 4, 4)
        global_t = torch.zeros(9, 4, 4)

        pred = TransformPrediction(
            local_transforms=local,
            global_transforms=global_t,
        )

        assert pred.num_frame_pairs == 9


class TestAlgorithmRegistry:
    """Test algorithm registry."""

    def test_baseline_registered(self):
        """Test that baseline algorithm is registered."""
        from usrec_zoo.algorithms import list_algorithms

        assert "tusrec_baseline" in list_algorithms()

    def test_get_algorithm_class(self):
        """Test getting algorithm class."""
        from usrec_zoo.algorithms import get_algorithm_class

        cls = get_algorithm_class("tusrec_baseline")
        assert cls.get_name() == "tusrec_baseline"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
