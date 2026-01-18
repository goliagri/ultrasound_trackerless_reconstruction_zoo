# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements and compares methods for trackerless 3D freehand ultrasound reconstruction. The goal is to estimate transformations between pairs of 2D ultrasound frames to reconstruct them into 3D volumes without external tracking devices. Built around the TUS-REC2025 Challenge baseline from UCL.

## Environment

**Conda environment**: `US_v1` (Python 3.9.13)

**Key packages**:
| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.8.0+cu128 | Required for RTX 50-series (sm_120/Blackwell) |
| torchvision | 0.23.0+cu128 | |
| pytorch3d | 0.7.5 | Installed via conda: `conda install pytorch3d -c pytorch3d --no-deps` |
| h5py | 3.14.0 | |
| numpy | 2.0.2 | |
| matplotlib | 3.9.4 | |

**GPU**: RTX 5070 Ti Laptop (12GB VRAM, compute capability 12.0)
- Default batch size 16 causes OOM; use `--MINIBATCH_SIZE 4` or `8`

### Environment Setup (from scratch)
```bash
conda create -n US_v1 python=3.9.13
conda activate US_v1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install h5py matplotlib tensorboard
conda install pytorch3d -c pytorch3d --no-deps
```

## Data

### Download Training Data
```bash
python scripts/download_tus_rec_2024.py
# Options:
#   --delete-zips    Delete zip files after extraction (saves ~84GB)
#   --verify-only    Only verify existing downloads
#   --parts X Y      Download specific parts (train_part1, train_part2, landmarks, calib, validation)
```

### Data Location
Data is stored at `../tus-rec-data/tus-rec-2024/` with symlink at `TUS-REC2025-Challenge_baseline/data`

### Data Structure
```
tus-rec-data/tus-rec-2024/
├── frames_transfs/           # Training data (174GB)
│   ├── 000/ ... 049/         # 50 subjects
│   │   └── *.h5              # 24 scans per subject (1200 total)
├── landmarks/                # 50 landmark files
├── validation/               # Validation data (4.6GB)
│   ├── frames/050,051,052/   # 3 subjects, 24 scans each
│   ├── transfs/050,051,052/  # Ground truth transforms
│   └── landmark/             # 3 landmark files
└── calib_matrix.csv          # Calibration matrix
```

### H5 File Format
Training files contain:
- `frames`: [N, 480, 640] uint8 ultrasound images
- `tforms`: [N, 4, 4] float32 transformation matrices

Landmark files contain:
- `{scan_name}`: [20, 3] int64 with (frame_index, x, y) coordinates
  - Note: 20 landmarks per scan (not 100 as originally documented)

## Development Commands

### Training
```bash
conda activate US_v1
cd TUS-REC2025-Challenge_baseline
python train.py --MINIBATCH_SIZE 4  # Use smaller batch for 12GB GPU
```

### Generate Displacement Fields (Evaluation)
```bash
cd TUS-REC2025-Challenge_baseline
python generate_DDF.py
```

## Architecture

### Core Pipeline
1. **Input**: Sequential 2D ultrasound frames (480×640) with ground-truth 4×4 transforms from optical tracker
2. **Model**: Modified EfficientNet-B1 that takes N consecutive frames and predicts 6DOF transformations between them
3. **Output**: Four Displacement Vector Fields (DDFs):
   - GP: Global displacement for all pixels (to reference frame)
   - GL: Global displacement for landmarks
   - LP: Local displacement for all pixels (to previous frame)
   - LL: Local displacement for landmarks

### Coordinate Systems
Three coordinate systems are used throughout:
- **Image coordinate (pixels)**: 480×640 frame space
- **Image coordinate (mm)**: Scaled from pixels using calibration
- **Tracker tool coordinate**: From optical tracker

Key transformation equation: `T_{j←i} = T_rotation^{-1} · T_{j←i}^tool · T_rotation`

### Key Modules (in `TUS-REC2025-Challenge_baseline/utils/`)
- `loader.py`: `Dataset` class for H5 file loading, subject-level partitioning, frame sampling
- `network.py`: `build_model()` creates EfficientNet-B1 with modified input/output layers
- `transform.py`: `LabelTransform`, `PredictionTransform`, `PointTransform`, `TransformAccumulation` - convert between 4×4 transforms, 6DOF parameters, and point coordinates
- `loss.py`: `PointDistance` loss metric
- `Transf2DDFs.py`: `cal_global_ddfs()`, `cal_local_ddfs()` - transform to DDF conversion

### Configuration
Training options in `options/train_options.py`:
- `PRED_TYPE`: 'parameter' (6DOF), 'transform' (4×4), 'point', 'quaternion'
- `LABEL_TYPE`: 'point', 'parameter', 'transform'
- `NUM_SAMPLES`: Number of input frames (default: 2)
- `MINIBATCH_SIZE`: Batch size (default: 16, use 4-8 for 12GB GPU)

## Code Style Requirements

From `style.md`:
- Use type-checking in function inputs and outputs
- Raise informative errors liberally - practically every complex function should verify inputs and outputs
- Design for unit testability
- Document input/output requirements in non-trivial functions
- For complex loops, add `#LOOP INVARIANT:` comments describing expected state of variables each iteration
