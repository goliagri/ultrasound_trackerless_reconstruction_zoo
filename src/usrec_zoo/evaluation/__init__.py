"""Evaluation utilities including DDF conversion and metrics."""

from usrec_zoo.evaluation.ddf import (
    transforms_to_ddf,
    reference_image_points,
    compute_ddf_from_tracker_transforms,
    data_pairs_global,
    data_pairs_local,
)
from usrec_zoo.evaluation.metrics import (
    compute_landmark_error,
    compute_per_frame_landmark_error,
    compute_pixel_error,
    ddf_to_landmark_positions,
)

__all__ = [
    # DDF conversion
    "transforms_to_ddf",
    "reference_image_points",
    "compute_ddf_from_tracker_transforms",
    "data_pairs_global",
    "data_pairs_local",
    # Metrics
    "compute_landmark_error",
    "compute_per_frame_landmark_error",
    "compute_pixel_error",
    "ddf_to_landmark_positions",
]
