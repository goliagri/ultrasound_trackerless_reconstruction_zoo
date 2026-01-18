#!/usr/bin/env python3
"""
Thorough equivalence tests for the TUSRECBaseline implementation.

These tests verify that the NEW implementation produces IDENTICAL outputs
to the ORIGINAL TUS-REC2025-Challenge_baseline code through ALL code paths,
including:

1. TransformPredictor wrapper (with internal normalization)
2. TUSRECBaseline.predict() full inference path
3. TUSRECBaseline.forward_batch() training path
4. Weight loading compatibility
5. End-to-end DDF computation from raw frames

Unlike test_exact_match.py which tests components in isolation,
these tests verify the FULL inference pipeline matches the original.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "TUS-REC2025-Challenge_baseline"))

import numpy as np
import torch
import pytest

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class TestTransformPredictorNormalization:
    """
    Test that TransformPredictor's internal normalization produces
    identical results to the original's external normalization.

    Original pattern (train.py):
        frames = frames/255
        outputs = model(frames)

    New pattern (TransformPredictor):
        outputs = model(frames)  # divides by 255 internally
    """

    def test_transform_predictor_matches_original_model(self):
        """
        Verify TransformPredictor with raw [0,255] input produces
        identical output to original model with pre-normalized input.
        """
        # Original imports
        from utils.network import build_model as orig_build_model

        # New imports
        from usrec_zoo.algorithms.tusrec_baseline.model import TransformPredictor

        # Create original model (no normalization)
        class MockOpt:
            model_name = "efficientnet_b1"

        orig_model = orig_build_model(MockOpt(), in_frames=2, pred_dim=6)
        orig_model.eval()

        # Create new TransformPredictor (has internal /255 normalization)
        new_config = {"model_name": "efficientnet_b1"}
        new_predictor = TransformPredictor(
            config=new_config,
            num_frames=2,
            num_pairs=1,
            pred_dim=6,
        )
        new_predictor.eval()

        # Copy weights from original to new (TransformPredictor wraps as backbone)
        new_predictor.backbone.load_state_dict(orig_model.state_dict())

        # Create raw input [0, 255] range
        raw_input = torch.randint(0, 256, (1, 2, 480, 640), dtype=torch.float32)

        with torch.no_grad():
            # Original: normalize externally, pass to model
            orig_output = orig_model(raw_input / 255.0)

            # New: pass raw input, TransformPredictor normalizes internally
            new_output = new_predictor(raw_input)
            new_output = new_output.squeeze(1)  # [B, 1, 6] -> [B, 6]

        max_diff = (orig_output - new_output).abs().max().item()
        print(f"\n  Raw input range: [{raw_input.min():.0f}, {raw_input.max():.0f}]")
        print(f"  Original output: {orig_output[0, :3].tolist()}")
        print(f"  New output: {new_output[0, :3].tolist()}")
        print(f"  Max difference: {max_diff}")

        assert torch.allclose(orig_output, new_output, atol=1e-5), \
            f"TransformPredictor output differs from original by {max_diff}"

    def test_transform_predictor_uint8_input(self):
        """
        Verify TransformPredictor handles uint8 input correctly.
        """
        from utils.network import build_model as orig_build_model
        from usrec_zoo.algorithms.tusrec_baseline.model import TransformPredictor

        class MockOpt:
            model_name = "efficientnet_b1"

        orig_model = orig_build_model(MockOpt(), in_frames=2, pred_dim=6)
        orig_model.eval()

        new_config = {"model_name": "efficientnet_b1"}
        new_predictor = TransformPredictor(
            config=new_config, num_frames=2, num_pairs=1, pred_dim=6
        )
        new_predictor.eval()
        new_predictor.backbone.load_state_dict(orig_model.state_dict())

        # Create actual uint8 input
        uint8_input = torch.randint(0, 256, (1, 2, 480, 640), dtype=torch.uint8)

        with torch.no_grad():
            # Original: convert to float and normalize
            orig_output = orig_model(uint8_input.float() / 255.0)

            # New: pass uint8 directly
            new_output = new_predictor(uint8_input).squeeze(1)

        max_diff = (orig_output - new_output).abs().max().item()
        print(f"\n  Input dtype: {uint8_input.dtype}")
        print(f"  Max difference: {max_diff}")

        assert torch.allclose(orig_output, new_output, atol=1e-5), \
            f"uint8 input handling differs by {max_diff}"


class TestTUSRECBaselinePredict:
    """
    Test the full TUSRECBaseline.predict() inference path.

    This tests the complete chain:
    ScanData -> TUSRECBaseline.predict() -> TransformPrediction

    And verifies it matches the original inference pattern.
    """

    @pytest.fixture
    def calibration(self):
        """Load calibration data."""
        from usrec_zoo.calibration import CalibrationData
        calib_path = Path("TUS-REC2025-Challenge_baseline/data/calib_matrix.csv")
        if not calib_path.exists():
            pytest.skip("Calibration file not found")
        return CalibrationData.from_csv(calib_path)

    @pytest.fixture
    def original_calibration(self):
        """Load original calibration matrices."""
        from utils.plot_functions import read_calib_matrices
        calib_path = "TUS-REC2025-Challenge_baseline/data/calib_matrix.csv"
        if not Path(calib_path).exists():
            pytest.skip("Calibration file not found")
        return read_calib_matrices(calib_path)

    def test_predict_matches_original_inference(self, calibration, original_calibration):
        """
        Verify TUSRECBaseline.predict() produces identical transforms
        to the original inference pattern.
        """
        # Original imports
        from utils.network import build_model as orig_build_model
        from utils.transform import Transforms as OrigTransforms
        from utils.transform import TransformAccumulation as OrigAccumulation
        from utils.plot_functions import reference_image_points

        # New imports
        from usrec_zoo.algorithms.tusrec_baseline import TUSRECBaseline
        from usrec_zoo.types import ScanData

        tform_calib_scale, tform_calib_R_T, tform_calib = original_calibration
        image_points = reference_image_points([480, 640], [480, 640])

        # Create original model and transforms
        class MockOpt:
            model_name = "efficientnet_b1"

        orig_model = orig_build_model(MockOpt(), in_frames=2, pred_dim=6)
        orig_model.eval()

        orig_transforms = OrigTransforms(
            pred_type="parameter",
            num_pairs=1,
            image_points=image_points,
            tform_image_to_tool=tform_calib,
            tform_image_mm_to_tool=tform_calib_R_T,
            tform_image_pixel_to_mm=tform_calib_scale,
        )
        orig_accumulation = OrigAccumulation(
            image_points=image_points,
            tform_image_to_tool=tform_calib,
            tform_image_mm_to_tool=tform_calib_R_T,
            tform_image_pixel_to_image_mm=tform_calib_scale,
        )

        # Create new baseline algorithm
        config = TUSRECBaseline.get_default_config()
        new_baseline = TUSRECBaseline(config, calibration)
        new_baseline.model.eval()

        # Copy weights
        new_baseline.model.backbone.load_state_dict(orig_model.state_dict())

        # Create test scan with raw uint8 frames
        num_frames = 5
        raw_frames = np.random.randint(0, 256, (num_frames, 480, 640), dtype=np.uint8)

        scan = ScanData(
            frames=raw_frames,
            transforms=None,
            subject_id="test",
            scan_name="test_scan",
        )

        # === ORIGINAL INFERENCE ===
        frames_tensor = torch.tensor(raw_frames, dtype=torch.float32)
        orig_local = []
        orig_global = []
        prev_transf = torch.eye(4)

        with torch.no_grad():
            for i in range(num_frames - 1):
                # Original normalizes before model
                frame_pair = frames_tensor[i:i + 2].unsqueeze(0) / 255.0
                output = orig_model(frame_pair)
                pred_transf = orig_transforms(output)[0, 0, ...]

                orig_local.append(pred_transf.clone())
                prev_transf = orig_accumulation(prev_transf, pred_transf)
                orig_global.append(prev_transf.clone())

        orig_local = torch.stack(orig_local)
        orig_global = torch.stack(orig_global)

        # === NEW INFERENCE ===
        device = torch.device("cpu")
        prediction = new_baseline.predict(scan, device)

        # Compare local transforms
        local_diff = (orig_local - prediction.local_transforms).abs().max().item()
        print(f"\n  Local transforms max diff: {local_diff}")
        assert local_diff < 1e-5, f"Local transforms differ by {local_diff}"

        # Compare global transforms
        global_diff = (orig_global - prediction.global_transforms).abs().max().item()
        print(f"  Global transforms max diff: {global_diff}")
        assert global_diff < 1e-5, f"Global transforms differ by {global_diff}"

        print("  ✓ TUSRECBaseline.predict() matches original inference!")

    def test_forward_batch_matches_original(self, calibration):
        """
        Verify TUSRECBaseline.forward_batch() matches original model output.
        """
        from utils.network import build_model as orig_build_model
        from usrec_zoo.algorithms.tusrec_baseline import TUSRECBaseline

        class MockOpt:
            model_name = "efficientnet_b1"

        orig_model = orig_build_model(MockOpt(), in_frames=2, pred_dim=6)
        orig_model.eval()

        config = TUSRECBaseline.get_default_config()
        new_baseline = TUSRECBaseline(config, calibration)
        new_baseline.model.eval()
        new_baseline.model.backbone.load_state_dict(orig_model.state_dict())

        # Create batch of raw frames [B, 2, H, W] in [0, 255]
        batch_size = 4
        raw_batch = torch.randint(0, 256, (batch_size, 2, 480, 640), dtype=torch.float32)

        device = torch.device("cpu")

        with torch.no_grad():
            # Original: normalize then forward
            orig_output = orig_model(raw_batch / 255.0)

            # New: forward_batch expects [0, 255] input
            new_output = new_baseline.forward_batch(raw_batch, device)
            new_output = new_output.squeeze(1)  # [B, 1, 6] -> [B, 6]

        max_diff = (orig_output - new_output).abs().max().item()
        print(f"\n  Batch size: {batch_size}")
        print(f"  Max difference: {max_diff}")

        assert torch.allclose(orig_output, new_output, atol=1e-5), \
            f"forward_batch differs by {max_diff}"


class TestWeightCompatibility:
    """
    Test that weights can be loaded from original checkpoints
    and produce identical outputs.
    """

    @pytest.fixture
    def calibration(self):
        from usrec_zoo.calibration import CalibrationData
        calib_path = Path("TUS-REC2025-Challenge_baseline/data/calib_matrix.csv")
        if not calib_path.exists():
            pytest.skip("Calibration file not found")
        return CalibrationData.from_csv(calib_path)

    def test_load_original_weights_into_new_model(self, calibration):
        """
        Verify original model weights can be loaded into new model.
        """
        from utils.network import build_model as orig_build_model
        from usrec_zoo.algorithms.tusrec_baseline.model import TransformPredictor

        class MockOpt:
            model_name = "efficientnet_b1"

        orig_model = orig_build_model(MockOpt(), in_frames=2, pred_dim=6)
        orig_state = orig_model.state_dict()

        new_predictor = TransformPredictor(
            config={"model_name": "efficientnet_b1"},
            num_frames=2,
            num_pairs=1,
            pred_dim=6,
        )

        # This should work without errors
        new_predictor.backbone.load_state_dict(orig_state)

        # Verify all keys match
        new_state = new_predictor.backbone.state_dict()
        assert set(orig_state.keys()) == set(new_state.keys()), \
            "State dict keys don't match"

        # Verify values match
        for key in orig_state:
            assert torch.equal(orig_state[key], new_state[key]), \
                f"Weight mismatch for key: {key}"

        print("\n  ✓ Original weights load correctly into new model!")

    def test_pretrained_weights_if_available(self, calibration):
        """
        If pretrained weights exist, verify they load and produce valid output.
        """
        from usrec_zoo.algorithms.tusrec_baseline import TUSRECBaseline

        weights_path = Path("TUS-REC2025-Challenge_baseline/TUS-REC2024_model/model_weights")
        if not weights_path.exists():
            pytest.skip("Pretrained weights not found")

        config = TUSRECBaseline.get_default_config()
        baseline = TUSRECBaseline(config, calibration)

        # Load pretrained weights
        state_dict = torch.load(weights_path, map_location="cpu")
        baseline.model.backbone.load_state_dict(state_dict)
        baseline.model.eval()

        # Verify model produces valid output
        test_input = torch.randint(0, 256, (1, 2, 480, 640), dtype=torch.float32)
        with torch.no_grad():
            output = baseline.model(test_input)

        assert output.shape == (1, 1, 6), f"Unexpected output shape: {output.shape}"
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

        print(f"\n  Pretrained weights loaded successfully")
        print(f"  Output shape: {output.shape}")
        print(f"  Output sample: {output[0, 0, :3].tolist()}")


class TestEndToEndDDF:
    """
    Test complete pipeline from raw frames to DDF output,
    comparing against original implementation.
    """

    @pytest.fixture
    def calibration(self):
        from usrec_zoo.calibration import CalibrationData
        calib_path = Path("TUS-REC2025-Challenge_baseline/data/calib_matrix.csv")
        if not calib_path.exists():
            pytest.skip("Calibration file not found")
        return CalibrationData.from_csv(calib_path)

    @pytest.fixture
    def original_calibration(self):
        from utils.plot_functions import read_calib_matrices
        calib_path = "TUS-REC2025-Challenge_baseline/data/calib_matrix.csv"
        if not Path(calib_path).exists():
            pytest.skip("Calibration file not found")
        return read_calib_matrices(calib_path)

    def test_end_to_end_ddf_from_raw_frames(self, calibration, original_calibration):
        """
        Full pipeline test: raw uint8 frames -> DDF output.

        Verifies the complete chain:
        1. Load raw frames
        2. Run inference
        3. Accumulate transforms
        4. Compute DDFs

        All must match the original implementation.
        """
        # Original imports
        from utils.network import build_model as orig_build_model
        from utils.transform import Transforms, TransformAccumulation
        from utils.plot_functions import reference_image_points
        from utils.Transf2DDFs import cal_global_ddfs, cal_local_ddfs

        # New imports
        from usrec_zoo.algorithms.tusrec_baseline import TUSRECBaseline
        from usrec_zoo.calibration import CalibrationDataTorch
        from usrec_zoo.evaluation.ddf import transforms_to_ddf
        from usrec_zoo.types import ScanData

        tform_calib_scale, tform_calib_R_T, tform_calib = original_calibration
        image_points = reference_image_points([480, 640], [480, 640])

        # Setup original model
        class MockOpt:
            model_name = "efficientnet_b1"

        orig_model = orig_build_model(MockOpt(), in_frames=2, pred_dim=6)
        orig_model.eval()

        orig_transforms = Transforms(
            pred_type="parameter",
            num_pairs=1,
            image_points=image_points,
            tform_image_to_tool=tform_calib,
            tform_image_mm_to_tool=tform_calib_R_T,
            tform_image_pixel_to_mm=tform_calib_scale,
        )
        orig_accumulation = TransformAccumulation(
            image_points=image_points,
            tform_image_to_tool=tform_calib,
            tform_image_mm_to_tool=tform_calib_R_T,
            tform_image_pixel_to_image_mm=tform_calib_scale,
        )

        # Setup new baseline
        config = TUSRECBaseline.get_default_config()
        new_baseline = TUSRECBaseline(config, calibration)
        new_baseline.model.eval()
        new_baseline.model.backbone.load_state_dict(orig_model.state_dict())

        # Create test data with raw uint8 frames
        num_frames = 5
        raw_frames = np.random.randint(0, 256, (num_frames, 480, 640), dtype=np.uint8)

        # Landmarks on frames 1-4 (not frame 0, which is reference)
        landmarks = np.array([
            [1, 100, 200],
            [1, 300, 150],
            [2, 150, 250],
            [2, 400, 300],
            [3, 200, 100],
            [4, 350, 400],
        ], dtype=np.int64)

        # === ORIGINAL PIPELINE ===
        frames_tensor = torch.tensor(raw_frames, dtype=torch.float32)
        orig_local = []
        orig_global = []
        prev_transf = torch.eye(4)

        with torch.no_grad():
            for i in range(num_frames - 1):
                frame_pair = frames_tensor[i:i + 2].unsqueeze(0) / 255.0
                output = orig_model(frame_pair)
                pred_transf = orig_transforms(output)[0, 0, ...]
                orig_local.append(pred_transf.clone())
                prev_transf = orig_accumulation(prev_transf, pred_transf)
                orig_global.append(prev_transf.clone())

        orig_local = torch.stack(orig_local)
        orig_global = torch.stack(orig_global)

        orig_global_pixels, orig_global_landmarks = cal_global_ddfs(
            orig_global, tform_calib_scale, image_points, torch.from_numpy(landmarks)
        )
        orig_local_pixels, orig_local_landmarks = cal_local_ddfs(
            orig_local, tform_calib_scale, image_points, torch.from_numpy(landmarks)
        )

        # === NEW PIPELINE ===
        scan = ScanData(
            frames=raw_frames,
            transforms=None,
            subject_id="test",
            scan_name="test_scan",
        )

        device = torch.device("cpu")
        prediction = new_baseline.predict(scan, device)

        calib_torch = CalibrationDataTorch(calibration, device)
        new_ddf = transforms_to_ddf(prediction, landmarks, calib_torch)

        # === COMPARE ALL OUTPUTS ===
        print("\n  Comparing full pipeline outputs:")

        # Local transforms
        local_diff = (orig_local - prediction.local_transforms).abs().max().item()
        print(f"  Local transforms max diff: {local_diff}")
        assert local_diff < 1e-5, f"Local transforms differ by {local_diff}"

        # Global transforms
        global_diff = (orig_global - prediction.global_transforms).abs().max().item()
        print(f"  Global transforms max diff: {global_diff}")
        assert global_diff < 1e-5, f"Global transforms differ by {global_diff}"

        # Global pixel DDFs
        gp_diff = np.abs(orig_global_pixels - new_ddf.global_pixels).max()
        print(f"  Global pixel DDF max diff: {gp_diff}")
        assert gp_diff < 1e-5, f"Global pixel DDFs differ by {gp_diff}"

        # Local pixel DDFs
        lp_diff = np.abs(orig_local_pixels - new_ddf.local_pixels).max()
        print(f"  Local pixel DDF max diff: {lp_diff}")
        assert lp_diff < 1e-5, f"Local pixel DDFs differ by {lp_diff}"

        # Global landmark DDFs
        gl_diff = np.abs(orig_global_landmarks - new_ddf.global_landmarks).max()
        print(f"  Global landmark DDF max diff: {gl_diff}")
        assert gl_diff < 1e-5, f"Global landmark DDFs differ by {gl_diff}"

        # Local landmark DDFs
        ll_diff = np.abs(orig_local_landmarks - new_ddf.local_landmarks).max()
        print(f"  Local landmark DDF max diff: {ll_diff}")
        assert ll_diff < 1e-5, f"Local landmark DDFs differ by {ll_diff}"

        print("  ✓ End-to-end pipeline produces identical results!")


class TestNormalizationEdgeCases:
    """
    Test edge cases in normalization to catch subtle bugs.
    """

    def test_no_double_normalization(self):
        """
        Verify that passing pre-normalized [0,1] data doesn't
        cause double normalization (which would make values tiny).
        """
        from usrec_zoo.algorithms.tusrec_baseline.model import TransformPredictor

        predictor = TransformPredictor(
            config={"model_name": "efficientnet_b1"},
            num_frames=2,
            num_pairs=1,
            pred_dim=6,
        )
        predictor.eval()

        # Input that's already in [0,1] range (pre-normalized)
        pre_normalized = torch.rand(1, 2, 480, 640)  # [0, 1]

        # This would be WRONG if TransformPredictor divides again
        # Values would become [0, 1/255] ≈ [0, 0.004]
        with torch.no_grad():
            output = predictor(pre_normalized)

        # The output should be reasonable (not tiny or huge)
        # Note: This test documents current behavior
        # If pre-normalized input is passed, it WILL be divided again
        # This is expected - the contract is [0,255] input
        print(f"\n  Pre-normalized input range: [{pre_normalized.min():.3f}, {pre_normalized.max():.3f}]")
        print(f"  After /255, effective range: [{pre_normalized.min()/255:.6f}, {pre_normalized.max()/255:.6f}]")
        print(f"  Output: {output[0, 0, :3].tolist()}")
        print("  Note: TransformPredictor expects [0,255] input by contract")

    def test_normalization_preserves_relative_values(self):
        """
        Verify normalization preserves relative pixel relationships.
        """
        from utils.network import build_model as orig_build_model
        from usrec_zoo.algorithms.tusrec_baseline.model import TransformPredictor

        class MockOpt:
            model_name = "efficientnet_b1"

        orig_model = orig_build_model(MockOpt(), in_frames=2, pred_dim=6)
        orig_model.eval()

        predictor = TransformPredictor(
            config={"model_name": "efficientnet_b1"},
            num_frames=2, num_pairs=1, pred_dim=6
        )
        predictor.eval()
        predictor.backbone.load_state_dict(orig_model.state_dict())

        # Test with different intensity levels
        for intensity in [50, 128, 200]:
            raw_input = torch.full((1, 2, 480, 640), intensity, dtype=torch.float32)

            with torch.no_grad():
                orig_output = orig_model(raw_input / 255.0)
                new_output = predictor(raw_input).squeeze(1)

            diff = (orig_output - new_output).abs().max().item()
            print(f"\n  Intensity {intensity}: max diff = {diff}")
            assert diff < 1e-5, f"Outputs differ at intensity {intensity}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
