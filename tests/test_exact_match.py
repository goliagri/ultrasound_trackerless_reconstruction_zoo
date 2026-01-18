#!/usr/bin/env python3
"""
Comprehensive verification that the new implementation exactly matches
the original TUS-REC2025-Challenge_baseline implementation.

This test verifies:
1. Model architecture produces identical outputs
2. Transform conversions (6DOF <-> 4x4) are identical
3. Transform accumulation produces identical results
4. DDF computation produces identical results
5. Ground truth DDF computation matches
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "TUS-REC2025-Challenge_baseline"))

import numpy as np
import torch
import pytorch3d.transforms

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def test_model_architecture_match():
    """Test that model architectures produce identical outputs with same weights."""
    print("\n" + "=" * 60)
    print("TEST: Model Architecture Match")
    print("=" * 60)

    # Original imports
    from utils.network import build_model as orig_build_model

    # New imports
    from usrec_zoo.algorithms.tusrec_baseline.model import build_model as new_build_model

    # Create original model
    class MockOpt:
        model_name = "efficientnet_b1"

    orig_model = orig_build_model(MockOpt(), in_frames=2, pred_dim=6)
    orig_model.eval()

    # Create new model
    new_config = {"model_name": "efficientnet_b1"}
    new_model = new_build_model(new_config, in_frames=2, pred_dim=6)
    new_model.eval()

    # Copy weights from original to new
    new_model.load_state_dict(orig_model.state_dict())

    # Test with random input (normalized as original code does: /255)
    test_input = torch.randint(0, 256, (1, 2, 480, 640), dtype=torch.float32) / 255.0

    with torch.no_grad():
        orig_output = orig_model(test_input)
        new_output = new_model(test_input)

    print(f"  Original output shape: {orig_output.shape}")
    print(f"  New output shape: {new_output.shape}")
    print(f"  Original output: {orig_output[0, :3].tolist()}")
    print(f"  New output: {new_output[0, :3].tolist()}")

    max_diff = (orig_output - new_output).abs().max().item()
    print(f"  Max difference: {max_diff}")

    assert torch.allclose(orig_output, new_output, atol=1e-6), f"Outputs differ by {max_diff}"
    print("  ✓ Model outputs match exactly!")


def test_params_to_matrix_match():
    """Test that 6DOF to 4x4 matrix conversion matches original."""
    print("\n" + "=" * 60)
    print("TEST: 6DOF to Matrix Conversion Match")
    print("=" * 60)

    from usrec_zoo.transforms import params_to_matrix

    # Test various parameter values
    test_cases = [
        torch.zeros(6),
        torch.tensor([0.1, 0.2, 0.3, 1.0, 2.0, 3.0]),
        torch.randn(6) * 0.5,
        torch.randn(10, 6) * 0.5,  # Batched
    ]

    for i, params in enumerate(test_cases):
        # Original implementation (from transform.py)
        if params.ndim == 1:
            euler = params[:3]
            trans = params[3:]
            rotation = pytorch3d.transforms.euler_angles_to_matrix(euler, 'ZYX')
            orig_matrix = torch.eye(4)
            orig_matrix[:3, :3] = rotation
            orig_matrix[:3, 3] = trans
        else:
            # Batched
            euler = params[..., :3]
            trans = params[..., 3:]
            rotation = pytorch3d.transforms.euler_angles_to_matrix(euler, 'ZYX')
            orig_matrix = torch.zeros(*params.shape[:-1], 4, 4)
            orig_matrix[..., :3, :3] = rotation
            orig_matrix[..., :3, 3] = trans
            orig_matrix[..., 3, 3] = 1.0

        # New implementation
        new_matrix = params_to_matrix(params)

        max_diff = (orig_matrix - new_matrix).abs().max().item()
        print(f"  Case {i + 1}: params shape {params.shape}, max diff = {max_diff}")

        assert torch.allclose(orig_matrix, new_matrix, atol=1e-6), f"Case {i + 1} differs by {max_diff}"

    print("  ✓ All param to matrix conversions match!")


def test_transform_accumulation_match():
    """Test that transform accumulation matches original."""
    print("\n" + "=" * 60)
    print("TEST: Transform Accumulation Match")
    print("=" * 60)

    # Original imports
    from utils.transform import TransformAccumulation as OrigAccumulation
    from utils.plot_functions import reference_image_points as orig_ref_points, read_calib_matrices

    # New imports
    from usrec_zoo.transforms.accumulation import TransformAccumulator
    from usrec_zoo.transforms import params_to_matrix

    # Load calibration
    calib_path = Path("TUS-REC2025-Challenge_baseline/data/calib_matrix.csv")
    if not calib_path.exists():
        print("  ⚠ Calibration file not found, skipping")
        return

    tform_calib_scale, tform_calib_R_T, tform_calib = read_calib_matrices(str(calib_path))
    image_points = orig_ref_points([480, 640], [480, 640])

    # Create original accumulator
    orig_accum = OrigAccumulation(
        image_points=image_points,
        tform_image_to_tool=tform_calib,
        tform_image_mm_to_tool=tform_calib_R_T,
        tform_image_pixel_to_image_mm=tform_calib_scale,
    )

    # Create test local transforms
    test_params = [
        torch.tensor([0.01, 0.02, 0.03, 0.1, 0.2, 0.3]),
        torch.tensor([0.02, 0.01, 0.01, 0.2, 0.1, 0.1]),
        torch.tensor([0.03, 0.02, 0.02, 0.3, 0.2, 0.2]),
        torch.tensor([-0.01, 0.01, -0.02, -0.1, 0.15, 0.05]),
    ]

    local_transforms = torch.stack([params_to_matrix(p) for p in test_params])

    # Original accumulation (manual loop as in Prediction.cal_pred_transformations)
    orig_global = torch.zeros_like(local_transforms)
    prev_transf = torch.eye(4)
    for i in range(len(local_transforms)):
        prev_transf = orig_accum(prev_transf, local_transforms[i])
        orig_global[i] = prev_transf

    # New accumulation
    new_global = TransformAccumulator.accumulate_local_to_global(local_transforms)

    for i in range(len(local_transforms)):
        max_diff = (orig_global[i] - new_global[i]).abs().max().item()
        print(f"  Transform {i}: max diff = {max_diff}")
        assert torch.allclose(orig_global[i], new_global[i], atol=1e-6), f"Transform {i} differs"

    print("  ✓ Transform accumulation matches!")


def test_reference_image_points_match():
    """Test that reference image points generation matches."""
    print("\n" + "=" * 60)
    print("TEST: Reference Image Points Match")
    print("=" * 60)

    # Original
    from utils.plot_functions import reference_image_points as orig_ref_points

    # New
    from usrec_zoo.evaluation.ddf import reference_image_points as new_ref_points

    # Generate full resolution
    orig_points = orig_ref_points([480, 640], [480, 640])
    new_points = new_ref_points(480, 640)

    print(f"  Original shape: {orig_points.shape}")
    print(f"  New shape: {new_points.shape}")
    print(f"  Original first 5 points (y,x,z,1):")
    for i in range(5):
        print(f"    [{i}]: {orig_points[:, i].tolist()}")
    print(f"  New first 5 points (y,x,z,1):")
    for i in range(5):
        print(f"    [{i}]: {new_points[:, i].tolist()}")

    # Check shapes match
    assert orig_points.shape == new_points.shape, f"Shapes differ: {orig_points.shape} vs {new_points.shape}"

    # Check values match
    max_diff = (orig_points - new_points).abs().max().item()
    print(f"  Max difference: {max_diff}")

    assert torch.allclose(orig_points, new_points, atol=1e-6), f"Points differ by {max_diff}"
    print("  ✓ Reference image points match!")


def test_ddf_computation_match():
    """Test that DDF computation matches original."""
    print("\n" + "=" * 60)
    print("TEST: DDF Computation Match")
    print("=" * 60)

    # Original imports
    from utils.Transf2DDFs import cal_global_ddfs, cal_local_ddfs
    from utils.plot_functions import reference_image_points as orig_ref_points, read_calib_matrices

    # New imports
    from usrec_zoo.evaluation.ddf import reference_image_points as new_ref_points
    from usrec_zoo.calibration import CalibrationData, CalibrationDataTorch
    from usrec_zoo.types import TransformPrediction
    from usrec_zoo.evaluation.ddf import transforms_to_ddf
    from usrec_zoo.transforms import params_to_matrix

    # Load calibration
    calib_path = Path("TUS-REC2025-Challenge_baseline/data/calib_matrix.csv")
    if not calib_path.exists():
        print("  ⚠ Calibration file not found, skipping")
        return

    # Original calibration
    tform_calib_scale, tform_calib_R_T, tform_calib = read_calib_matrices(str(calib_path))
    orig_image_points = orig_ref_points([480, 640], [480, 640])

    # New calibration
    new_calib = CalibrationData.from_csv(calib_path)
    new_calib_torch = CalibrationDataTorch(new_calib)

    # Create test transforms (5 frames = 4 transform pairs)
    test_params = [
        torch.tensor([0.01, 0.02, 0.03, 0.1, 0.2, 0.3]),
        torch.tensor([0.02, 0.01, 0.01, 0.2, 0.1, 0.1]),
        torch.tensor([0.03, 0.02, 0.02, 0.3, 0.2, 0.2]),
        torch.tensor([-0.01, 0.01, -0.02, -0.1, 0.15, 0.05]),
    ]

    local_transforms = torch.stack([params_to_matrix(p) for p in test_params])

    # Compute global transforms
    from usrec_zoo.transforms.accumulation import TransformAccumulator
    global_transforms = TransformAccumulator.accumulate_local_to_global(local_transforms)

    # Create test landmarks (spread across frames 1-4, not frame 0)
    # Note: DDF only covers frames 1 to N-1, frame 0 is reference
    # Original code uses landmark[:,0]-1 for indexing, so frame indices should be 1-based
    # i.e., landmark frame 1 -> DDF index 0, landmark frame 2 -> DDF index 1, etc.
    landmarks = np.array([
        [1, 100, 200],  # frame 1 -> DDF index 0
        [1, 300, 150],
        [2, 150, 250],  # frame 2 -> DDF index 1
        [2, 400, 300],
        [3, 200, 100],  # frame 3 -> DDF index 2
        [3, 350, 200],
        [4, 250, 350],  # frame 4 -> DDF index 3
        [4, 450, 400],
    ], dtype=np.int64)

    # === Original DDF computation ===
    orig_global_pixels, orig_global_landmarks = cal_global_ddfs(
        global_transforms, tform_calib_scale, orig_image_points, torch.from_numpy(landmarks)
    )
    orig_local_pixels, orig_local_landmarks = cal_local_ddfs(
        local_transforms, tform_calib_scale, orig_image_points, torch.from_numpy(landmarks)
    )

    # === New DDF computation ===
    prediction = TransformPrediction(
        local_transforms=local_transforms,
        global_transforms=global_transforms,
    )
    new_ddf = transforms_to_ddf(prediction, landmarks, new_calib_torch)

    # Compare global pixel DDFs
    print(f"  Global pixels shape - orig: {orig_global_pixels.shape}, new: {new_ddf.global_pixels.shape}")
    global_pixel_diff = np.abs(orig_global_pixels - new_ddf.global_pixels).max()
    print(f"  Global pixels max diff: {global_pixel_diff}")

    # Compare local pixel DDFs
    print(f"  Local pixels shape - orig: {orig_local_pixels.shape}, new: {new_ddf.local_pixels.shape}")
    local_pixel_diff = np.abs(orig_local_pixels - new_ddf.local_pixels).max()
    print(f"  Local pixels max diff: {local_pixel_diff}")

    # Compare global landmark DDFs
    print(f"  Global landmarks shape - orig: {orig_global_landmarks.shape}, new: {new_ddf.global_landmarks.shape}")
    global_landmark_diff = np.abs(orig_global_landmarks - new_ddf.global_landmarks).max()
    print(f"  Global landmarks max diff: {global_landmark_diff}")

    # Compare local landmark DDFs
    print(f"  Local landmarks shape - orig: {orig_local_landmarks.shape}, new: {new_ddf.local_landmarks.shape}")
    local_landmark_diff = np.abs(orig_local_landmarks - new_ddf.local_landmarks).max()
    print(f"  Local landmarks max diff: {local_landmark_diff}")

    # Assert all match
    assert global_pixel_diff < 1e-5, f"Global pixels differ by {global_pixel_diff}"
    assert local_pixel_diff < 1e-5, f"Local pixels differ by {local_pixel_diff}"
    assert global_landmark_diff < 1e-5, f"Global landmarks differ by {global_landmark_diff}"
    assert local_landmark_diff < 1e-5, f"Local landmarks differ by {local_landmark_diff}"

    print("  ✓ DDF computation matches!")


def test_label_transform_match():
    """Test that ground truth transform computation matches."""
    print("\n" + "=" * 60)
    print("TEST: Label Transform (Ground Truth) Match")
    print("=" * 60)

    # Original imports
    from utils.transform import LabelTransform
    from utils.plot_functions import reference_image_points, read_calib_matrices, data_pairs_global

    # New imports
    from usrec_zoo.calibration import CalibrationData, CalibrationDataTorch

    calib_path = Path("TUS-REC2025-Challenge_baseline/data/calib_matrix.csv")
    if not calib_path.exists():
        print("  ⚠ Calibration file not found, skipping")
        return

    # Load calibration
    tform_calib_scale, tform_calib_R_T, tform_calib = read_calib_matrices(str(calib_path))
    image_points = reference_image_points([480, 640], 2)

    # Create synthetic tracker transforms (5 frames)
    num_frames = 5
    tforms = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, num_frames, 1, 1)
    # Add small variations
    for i in range(num_frames):
        tforms[0, i, 0, 3] = i * 0.5  # Translation in x
        tforms[0, i, 1, 3] = i * 0.3  # Translation in y
    tforms_inv = torch.linalg.inv(tforms)

    # Original label transform (global)
    data_pairs = data_pairs_global(num_frames)[1:, :]  # Skip first (0->0)
    orig_label_transform = LabelTransform(
        "transform",
        pairs=data_pairs,
        image_points=image_points,
        tform_image_to_tool=tform_calib,
        tform_image_mm_to_tool=tform_calib_R_T,
        tform_image_pixel_to_mm=tform_calib_scale,
    )
    orig_global_transforms = torch.squeeze(orig_label_transform(tforms, tforms_inv))

    # New implementation: compute the same transforms
    new_calib = CalibrationData.from_csv(calib_path)
    new_calib_torch = CalibrationDataTorch(new_calib)

    # Compute transforms using new code's approach
    # T_{img1_mm -> img0_mm} = T_{tool -> img_mm} @ T_{tool0 -> world}^{-1} @ T_{tool1 -> world} @ T_{img_mm -> tool}
    new_global_transforms = []
    for i in range(1, num_frames):
        t_tool_global = torch.matmul(tforms_inv[0, 0], tforms[0, i])
        t_img = torch.matmul(
            new_calib_torch.tform_tool_to_mm,
            torch.matmul(t_tool_global, new_calib_torch.tform_mm_to_tool)
        )
        new_global_transforms.append(t_img)
    new_global_transforms = torch.stack(new_global_transforms)

    print(f"  Original global transforms shape: {orig_global_transforms.shape}")
    print(f"  New global transforms shape: {new_global_transforms.shape}")

    max_diff = (orig_global_transforms - new_global_transforms).abs().max().item()
    print(f"  Max difference: {max_diff}")

    assert torch.allclose(orig_global_transforms, new_global_transforms, atol=1e-5), f"Differs by {max_diff}"
    print("  ✓ Label transform computation matches!")


def test_full_pipeline_integration():
    """Test full pipeline: model forward -> transforms -> DDF."""
    print("\n" + "=" * 60)
    print("TEST: Full Pipeline Integration")
    print("=" * 60)

    # Original imports
    from utils.network import build_model as orig_build_model
    from utils.transform import Transforms as OrigTransforms, TransformAccumulation as OrigAccumulation
    from utils.plot_functions import reference_image_points, read_calib_matrices
    from utils.Transf2DDFs import cal_global_ddfs, cal_local_ddfs

    # New imports
    from usrec_zoo.algorithms.tusrec_baseline.model import build_model as new_build_model
    from usrec_zoo.transforms import params_to_matrix
    from usrec_zoo.transforms.accumulation import TransformAccumulator
    from usrec_zoo.calibration import CalibrationData, CalibrationDataTorch
    from usrec_zoo.evaluation.ddf import transforms_to_ddf
    from usrec_zoo.types import TransformPrediction

    calib_path = Path("TUS-REC2025-Challenge_baseline/data/calib_matrix.csv")
    if not calib_path.exists():
        print("  ⚠ Calibration file not found, skipping")
        return

    # Load calibration
    tform_calib_scale, tform_calib_R_T, tform_calib = read_calib_matrices(str(calib_path))
    image_points = reference_image_points([480, 640], [480, 640])

    new_calib = CalibrationData.from_csv(calib_path)
    new_calib_torch = CalibrationDataTorch(new_calib)

    # Create models with same weights
    class MockOpt:
        model_name = "efficientnet_b1"

    orig_model = orig_build_model(MockOpt(), in_frames=2, pred_dim=6)
    new_config = {"model_name": "efficientnet_b1"}
    new_model = new_build_model(new_config, in_frames=2, pred_dim=6)

    # Copy weights
    new_model.load_state_dict(orig_model.state_dict())
    orig_model.eval()
    new_model.eval()

    # Original transform handler
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

    # Simulate a scan with 5 frames
    num_frames = 5
    frames = torch.randint(0, 256, (1, num_frames, 480, 640), dtype=torch.float32) / 255.0

    # Test landmarks (frame indices are 1-based, matching original TUS-REC format)
    # Frame 1-4 map to DDF indices 0-3
    landmarks = np.array([
        [1, 100, 200],  # frame 1 -> DDF index 0
        [2, 150, 250],  # frame 2 -> DDF index 1
        [3, 200, 100],  # frame 3 -> DDF index 2
        [4, 250, 350],  # frame 4 -> DDF index 3
    ], dtype=np.int64)

    # === ORIGINAL PIPELINE ===
    orig_local = []
    orig_global = []
    prev_transf = torch.eye(4)

    with torch.no_grad():
        for i in range(num_frames - 1):
            frame_pair = frames[:, i:i + 2, ...]
            output = orig_model(frame_pair)
            pred_transf = orig_transforms(output)[0, 0, ...]  # Get single transform

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
    new_local = []

    with torch.no_grad():
        for i in range(num_frames - 1):
            frame_pair = frames[:, i:i + 2, ...]
            output = new_model(frame_pair)  # [1, 6]
            params = output.squeeze()  # [6]
            transform = params_to_matrix(params)  # [4, 4]
            new_local.append(transform)

    new_local = torch.stack(new_local)
    new_global = TransformAccumulator.accumulate_local_to_global(new_local)

    prediction = TransformPrediction(
        local_transforms=new_local,
        global_transforms=new_global,
    )
    new_ddf = transforms_to_ddf(prediction, landmarks, new_calib_torch)

    # === COMPARE ===
    print("  Comparing local transforms...")
    local_diff = (orig_local - new_local).abs().max().item()
    print(f"    Max diff: {local_diff}")
    assert local_diff < 1e-5, f"Local transforms differ by {local_diff}"

    print("  Comparing global transforms...")
    global_diff = (orig_global - new_global).abs().max().item()
    print(f"    Max diff: {global_diff}")
    assert global_diff < 1e-5, f"Global transforms differ by {global_diff}"

    print("  Comparing global pixel DDFs...")
    gp_diff = np.abs(orig_global_pixels - new_ddf.global_pixels).max()
    print(f"    Max diff: {gp_diff}")
    assert gp_diff < 1e-5, f"Global pixel DDFs differ by {gp_diff}"

    print("  Comparing local pixel DDFs...")
    lp_diff = np.abs(orig_local_pixels - new_ddf.local_pixels).max()
    print(f"    Max diff: {lp_diff}")
    assert lp_diff < 1e-5, f"Local pixel DDFs differ by {lp_diff}"

    print("  Comparing global landmark DDFs...")
    gl_diff = np.abs(orig_global_landmarks - new_ddf.global_landmarks).max()
    print(f"    Max diff: {gl_diff}")
    assert gl_diff < 1e-5, f"Global landmark DDFs differ by {gl_diff}"

    print("  Comparing local landmark DDFs...")
    ll_diff = np.abs(orig_local_landmarks - new_ddf.local_landmarks).max()
    print(f"    Max diff: {ll_diff}")
    assert ll_diff < 1e-5, f"Local landmark DDFs differ by {ll_diff}"

    print("  ✓ Full pipeline produces identical results!")


def main():
    print("=" * 60)
    print("VERIFICATION: New Implementation vs Original TUS-REC Baseline")
    print("=" * 60)

    tests = [
        test_model_architecture_match,
        test_params_to_matrix_match,
        test_reference_image_points_match,
        test_transform_accumulation_match,
        test_label_transform_match,
        test_ddf_computation_match,
        test_full_pipeline_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
