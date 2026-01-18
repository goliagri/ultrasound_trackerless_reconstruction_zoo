#!/usr/bin/env python3
"""
Download script for TUS-REC 2024 Challenge data from Zenodo.

Downloads training, validation, landmark, and calibration files with:
- Resumable downloads via wget
- MD5 verification
- Automatic extraction and organization
- State tracking for interrupted downloads

Usage:
    python download_tus_rec_2024.py                    # Download everything
    python download_tus_rec_2024.py --parts landmarks calib  # Specific parts
    python download_tus_rec_2024.py --verify-only      # Only verify existing files
    python download_tus_rec_2024.py --data-dir /path   # Custom data directory
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional


# File definitions with URLs, sizes, and MD5 checksums
FILES: Dict[str, Dict] = {
    "train_part1": {
        "url": "https://zenodo.org/api/records/11178509/files/train_part1.zip/content",
        "filename": "train_part1.zip",
        "md5": "7226991c6b04ef85e3cfb6f4e0ea7ad2",
        "size_gb": 43.4,
        "extract_to": "frames_transfs",
        "description": "Training data part 1 (43.4 GB)",
    },
    "train_part2": {
        "url": "https://zenodo.org/api/records/11180795/files/train_part2.zip/content",
        "filename": "train_part2.zip",
        "md5": "c09bbd73d62ebdb25535b793668adf46",
        "size_gb": 40.4,
        "extract_to": "frames_transfs",
        "description": "Training data part 2 (40.4 GB)",
    },
    "landmarks": {
        "url": "https://zenodo.org/records/11355500/files/landmark.zip?download=1",
        "filename": "landmark.zip",
        "md5": "f437d60627b5670dfd72f11dbee2cd2d",
        "size_gb": 0.000185,
        "extract_to": "landmarks",
        "description": "Landmark annotations (185 KB)",
    },
    "calib": {
        "url": "https://zenodo.org/api/records/11178509/files/calib_matrix.csv/content",
        "filename": "calib_matrix.csv",
        "md5": "97b04023f70cfdf9566138e370e19ff2",
        "size_gb": 0.000000413,
        "extract_to": None,  # No extraction needed
        "description": "Calibration matrix (413 B)",
    },
    "validation": {
        "url": "https://zenodo.org/records/12979481/files/Freehand_US_data_val.zip?download=1",
        "filename": "Freehand_US_data_val.zip",
        "md5": "487ebe3241678569296e47efeb2ea325",
        "size_gb": 4.8,
        "extract_to": "validation",
        "description": "Validation data (4.8 GB)",
    },
}

DEFAULT_PARTS = ["train_part1", "train_part2", "landmarks", "calib", "validation"]


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent.resolve()


def get_default_data_dir() -> Path:
    """Get the default data directory (sibling to repo)."""
    repo_root = get_repo_root()
    return repo_root.parent / "tus-rec-data" / "tus-rec-2024"


def load_state(state_file: Path) -> Dict:
    """Load download state from JSON file."""
    if state_file.exists():
        with open(state_file, "r") as f:
            return json.load(f)
    return {"downloaded": [], "verified": [], "extracted": []}


def save_state(state_file: Path, state: Dict) -> None:
    """Save download state to JSON file."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def verify_md5(filepath: Path, expected_md5: str, chunk_size: int = 8192 * 1024) -> bool:
    """
    Verify MD5 checksum of a file using chunked reading.

    Args:
        filepath: Path to file to verify
        expected_md5: Expected MD5 hex digest
        chunk_size: Size of chunks to read (default 8MB for large files)

    Returns:
        True if MD5 matches, False otherwise
    """
    if not filepath.exists():
        return False

    md5_hash = hashlib.md5()
    file_size = filepath.stat().st_size
    bytes_read = 0

    print(f"  Verifying MD5 of {filepath.name}...", end="", flush=True)

    with open(filepath, "rb") as f:
        # LOOP INVARIANT: bytes_read contains total bytes processed so far
        while chunk := f.read(chunk_size):
            md5_hash.update(chunk)
            bytes_read += len(chunk)
            # Print progress for large files (>100MB)
            if file_size > 100_000_000:
                pct = (bytes_read / file_size) * 100
                print(f"\r  Verifying MD5 of {filepath.name}... {pct:.1f}%", end="", flush=True)

    computed_md5 = md5_hash.hexdigest()
    matches = computed_md5 == expected_md5

    if matches:
        print(f"\r  Verifying MD5 of {filepath.name}... OK")
    else:
        print(f"\r  Verifying MD5 of {filepath.name}... FAILED")
        print(f"    Expected: {expected_md5}")
        print(f"    Got:      {computed_md5}")

    return matches


def download_with_wget(
    url: str,
    output_path: Path,
    max_retries: int = 3,
    timeout: int = 30
) -> bool:
    """
    Download a file using wget with resume support.

    Args:
        url: URL to download from
        output_path: Path to save file to
        max_retries: Maximum number of retry attempts
        timeout: Connection timeout in seconds

    Returns:
        True if download succeeded, False otherwise

    Raises:
        FileNotFoundError: If wget is not installed
    """
    # Check if wget is available
    if shutil.which("wget") is None:
        raise FileNotFoundError(
            "wget is not installed. Please install it:\n"
            "  Ubuntu/Debian: sudo apt install wget\n"
            "  macOS: brew install wget\n"
            "  Windows: Use WSL or install wget for Windows"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # LOOP INVARIANT: attempt is current retry count (0-indexed)
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"  Retry attempt {attempt + 1}/{max_retries}...")

        cmd = [
            "wget",
            "-c",  # Continue/resume partial downloads
            "--progress=bar:force:noscroll",
            f"--timeout={timeout}",
            "--tries=3",  # wget internal retries per attempt
            "-O", str(output_path),
            url
        ]

        try:
            result = subprocess.run(
                cmd,
                check=False,
                stderr=subprocess.STDOUT
            )

            if result.returncode == 0:
                return True

            print(f"  wget returned code {result.returncode}")

        except subprocess.SubprocessError as e:
            print(f"  Download error: {e}")

    return False


def extract_zip(
    zip_path: Path,
    extract_to: Path,
    merge_contents: bool = False
) -> bool:
    """
    Extract a zip file to specified directory.

    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
        merge_contents: If True, merge contents into existing directory

    Returns:
        True if extraction succeeded, False otherwise
    """
    if not zip_path.exists():
        print(f"  Error: {zip_path} does not exist")
        return False

    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting {zip_path.name} to {extract_to}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get total size for progress
            total_size = sum(info.file_size for info in zf.infolist())
            extracted_size = 0

            # LOOP INVARIANT: extracted_size is total bytes extracted so far
            for member in zf.infolist():
                zf.extract(member, extract_to)
                extracted_size += member.file_size

                if total_size > 100_000_000:  # Show progress for >100MB
                    pct = (extracted_size / total_size) * 100
                    print(f"\r  Extracting {zip_path.name}... {pct:.1f}%", end="", flush=True)

            print(f"\r  Extracting {zip_path.name}... Done")

        return True

    except zipfile.BadZipFile as e:
        print(f"  Error: {zip_path} is corrupted: {e}")
        return False
    except Exception as e:
        print(f"  Error extracting {zip_path}: {e}")
        return False


def organize_extracted_files(data_dir: Path) -> None:
    """
    Organize extracted files into the expected structure.

    Moves files from nested zip structure to flat structure expected by
    the baseline code.

    Args:
        data_dir: Root data directory
    """
    frames_transfs = data_dir / "frames_transfs"

    # train_part1.zip extracts to train_part1/ containing h5 files
    # Move contents up one level
    for part_dir in ["train_part1", "train_part2"]:
        nested_dir = frames_transfs / part_dir
        if nested_dir.exists() and nested_dir.is_dir():
            print(f"  Moving files from {part_dir}/ to frames_transfs/...")
            for item in nested_dir.iterdir():
                target = frames_transfs / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
            # Remove empty directory
            if not any(nested_dir.iterdir()):
                nested_dir.rmdir()

    # landmarks.zip extracts to landmark/ - rename to landmarks/
    landmark_dir = data_dir / "landmarks" / "landmark"
    landmarks_dir = data_dir / "landmarks"
    if landmark_dir.exists() and landmark_dir.is_dir():
        print("  Reorganizing landmark files...")
        for item in landmark_dir.iterdir():
            target = landmarks_dir / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
        if not any(landmark_dir.iterdir()):
            landmark_dir.rmdir()

    # validation.zip may extract with nested structure
    validation_dir = data_dir / "validation"
    if validation_dir.exists():
        # Check for nested Freehand_US_data_val directory
        nested = validation_dir / "Freehand_US_data_val"
        if nested.exists() and nested.is_dir():
            print("  Reorganizing validation files...")
            for item in nested.iterdir():
                target = validation_dir / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
            if not any(nested.iterdir()):
                nested.rmdir()


def setup_symlink(data_dir: Path, repo_root: Path) -> bool:
    """
    Create symlink from baseline code data directory to downloaded data.

    Args:
        data_dir: Path to downloaded data
        repo_root: Path to repository root

    Returns:
        True if symlink created/exists, False on error
    """
    baseline_data = repo_root / "TUS-REC2025-Challenge_baseline" / "data"

    # Calculate relative path from symlink location to data directory
    try:
        rel_path = os.path.relpath(data_dir, baseline_data.parent)
    except ValueError:
        # On Windows, relpath fails across drives
        rel_path = str(data_dir)

    print(f"\nSetting up symlink: {baseline_data} -> {rel_path}")

    if baseline_data.exists():
        if baseline_data.is_symlink():
            existing_target = os.readlink(baseline_data)
            if existing_target == rel_path or baseline_data.resolve() == data_dir.resolve():
                print("  Symlink already exists and points to correct location")
                return True
            else:
                print(f"  Removing existing symlink (pointed to {existing_target})")
                baseline_data.unlink()
        else:
            print(f"  Warning: {baseline_data} exists and is not a symlink")
            print("  Please remove it manually if you want to use the downloaded data")
            return False

    try:
        baseline_data.symlink_to(rel_path)
        print("  Symlink created successfully")
        return True
    except OSError as e:
        print(f"  Error creating symlink: {e}")
        print("  On Windows, you may need to run as Administrator or enable Developer Mode")
        return False


def download_part(
    part_name: str,
    data_dir: Path,
    state: Dict,
    state_file: Path,
    verify_only: bool = False,
    keep_zips: bool = True
) -> bool:
    """
    Download, verify, and extract a single data part.

    Args:
        part_name: Name of part to download (key in FILES dict)
        data_dir: Directory to download to
        state: Current download state dict
        state_file: Path to state file
        verify_only: If True, only verify existing files
        keep_zips: If True, keep zip files after extraction

    Returns:
        True if part is ready (downloaded, verified, extracted), False otherwise
    """
    if part_name not in FILES:
        print(f"Error: Unknown part '{part_name}'")
        print(f"Available parts: {', '.join(FILES.keys())}")
        return False

    file_info = FILES[part_name]
    filename = file_info["filename"]
    filepath = data_dir / filename
    url = file_info["url"]
    expected_md5 = file_info["md5"]
    extract_to = file_info["extract_to"]

    print(f"\n{'='*60}")
    print(f"Processing: {file_info['description']}")
    print(f"{'='*60}")

    # Check if already verified and extracted
    if part_name in state.get("verified", []):
        print(f"  Already verified: {filename}")
        if extract_to and part_name not in state.get("extracted", []):
            # Need to extract - continue processing
            pass
        elif extract_to is None or part_name in state.get("extracted", []):
            # Already complete - but check if we should delete the zip
            if not keep_zips and filepath.exists() and extract_to is not None:
                print(f"  Deleting {filename} to save space...")
                filepath.unlink()
            return True

    # Download if needed (and not verify_only mode)
    if not filepath.exists():
        if verify_only:
            print(f"  File not found: {filepath}")
            return False

        print(f"  Downloading {filename}...")
        if not download_with_wget(url, filepath):
            print(f"  Failed to download {filename}")
            return False

        if part_name not in state["downloaded"]:
            state["downloaded"].append(part_name)
            save_state(state_file, state)
    else:
        print(f"  File exists: {filepath}")

    # Verify MD5
    if part_name not in state.get("verified", []):
        if not verify_md5(filepath, expected_md5):
            if not verify_only:
                print(f"  MD5 verification failed. Deleting and re-downloading...")
                filepath.unlink()
                return download_part(part_name, data_dir, state, state_file, verify_only, keep_zips)
            return False

        if "verified" not in state:
            state["verified"] = []
        state["verified"].append(part_name)
        save_state(state_file, state)

    # Extract if needed
    if extract_to and part_name not in state.get("extracted", []):
        if verify_only:
            print(f"  Would extract to: {data_dir / extract_to}")
        else:
            target_dir = data_dir / extract_to
            if not extract_zip(filepath, target_dir):
                return False

            if "extracted" not in state:
                state["extracted"] = []
            state["extracted"].append(part_name)
            save_state(state_file, state)

            # Delete zip after extraction if not keeping
            if not keep_zips:
                print(f"  Deleting {filename} to save space...")
                filepath.unlink()
    elif extract_to is None:
        # No extraction needed (e.g., calib_matrix.csv)
        if part_name not in state.get("extracted", []):
            if "extracted" not in state:
                state["extracted"] = []
            state["extracted"].append(part_name)
            save_state(state_file, state)

    return True


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download TUS-REC 2024 Challenge data from Zenodo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           Download all data
  %(prog)s --parts landmarks calib   Download only landmarks and calibration
  %(prog)s --verify-only             Verify existing downloads
  %(prog)s --data-dir /path/to/data  Use custom data directory
  %(prog)s --no-symlink              Don't create symlink to baseline code
  %(prog)s --delete-zips             Delete zip files after extraction
        """
    )

    parser.add_argument(
        "--parts",
        nargs="+",
        choices=list(FILES.keys()),
        default=DEFAULT_PARTS,
        help="Which parts to download (default: all)"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory to store data (default: ../tus-rec-data/tus-rec-2024)"
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloads, don't download"
    )

    parser.add_argument(
        "--no-symlink",
        action="store_true",
        help="Don't create symlink to baseline code data directory"
    )

    parser.add_argument(
        "--delete-zips",
        action="store_true",
        help="Delete zip files after extraction to save space"
    )

    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Reset download state and start fresh"
    )

    args = parser.parse_args()

    # Setup paths
    repo_root = get_repo_root()
    data_dir = args.data_dir if args.data_dir else get_default_data_dir()
    state_file = data_dir / ".download_state.json"

    print(f"TUS-REC 2024 Data Downloader")
    print(f"{'='*60}")
    print(f"Repository root: {repo_root}")
    print(f"Data directory:  {data_dir}")
    print(f"Parts to process: {', '.join(args.parts)}")

    # Calculate total size
    total_size = sum(FILES[p]["size_gb"] for p in args.parts)
    print(f"Total size: ~{total_size:.1f} GB")

    if args.verify_only:
        print("\nMode: Verify only (no downloads)")

    # Load or reset state
    if args.reset_state:
        state = {"downloaded": [], "verified": [], "extracted": []}
        print("\nState reset.")
    else:
        state = load_state(state_file)
        if state["downloaded"] or state["verified"] or state["extracted"]:
            print(f"\nResuming from previous state:")
            print(f"  Downloaded: {len(state.get('downloaded', []))} parts")
            print(f"  Verified: {len(state.get('verified', []))} parts")
            print(f"  Extracted: {len(state.get('extracted', []))} parts")

    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)

    # Process each part
    success = True
    for part in args.parts:
        if not download_part(
            part,
            data_dir,
            state,
            state_file,
            verify_only=args.verify_only,
            keep_zips=not args.delete_zips
        ):
            success = False
            if not args.verify_only:
                print(f"\nFailed to process {part}. You can resume later.")

    # Organize files after extraction
    if not args.verify_only and success:
        print(f"\n{'='*60}")
        print("Organizing extracted files...")
        organize_extracted_files(data_dir)

    # Setup symlink
    if not args.no_symlink and not args.verify_only and success:
        setup_symlink(data_dir, repo_root)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    if success:
        print("All requested parts processed successfully!")
        print(f"\nData location: {data_dir}")

        if not args.no_symlink:
            baseline_data = repo_root / "TUS-REC2025-Challenge_baseline" / "data"
            print(f"Symlink: {baseline_data}")

        print("\nTo test the data with baseline code:")
        print("  cd TUS-REC2025-Challenge_baseline")
        print('  python -c "from utils.loader import Dataset; d = Dataset(\'data/frames_transfs\', 2, 2); print(f\'Loaded {len(d)} samples\')"')
    else:
        print("Some parts failed. Re-run the script to retry.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
