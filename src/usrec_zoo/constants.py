"""
Constants used throughout the usrec_zoo package.

These values are derived from the TUS-REC2025 Challenge dataset specifications
and should not be modified unless working with different data formats.
"""

# Image dimensions (in pixels)
IMAGE_HEIGHT: int = 480
IMAGE_WIDTH: int = 640
NUM_PIXELS: int = IMAGE_HEIGHT * IMAGE_WIDTH  # 307200

# Landmark specifications
DEFAULT_LANDMARKS_PER_SCAN: int = 20

# Number of scans per subject in the training data
SCANS_PER_SUBJECT: int = 24

# Total number of subjects in training data
NUM_TRAINING_SUBJECTS: int = 50

# Euler angle convention used for 6DOF parameters
# ZYX means: first rotate around Z, then Y, then X
EULER_CONVENTION: str = "ZYX"

# Coordinate indexing note:
# Landmarks use 1-based indexing for x,y coordinates (columns 1,2)
# and 0-based indexing for frame index (column 0).
# This is consistent with the calibration process.
LANDMARK_COORD_OFFSET: int = 1
