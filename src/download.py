"""Phase 1: Dataset download and validation."""
import os
import sys
import numpy as np
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def download_dataset():
    """Download all .npz files from KernelCo/robot_control on HuggingFace."""
    from huggingface_hub import hf_hub_download, list_repo_tree

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Listing files in KernelCo/robot_control...")
    files = [
        f.rfilename
        for f in list_repo_tree(
            "KernelCo/robot_control", repo_type="dataset", path_in_repo="data"
        )
        if f.rfilename.endswith(".npz")
    ]
    print(f"Found {len(files)} .npz files to download.")

    for i, fname in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {fname}...", end="", flush=True)
        hf_hub_download(
            repo_id="KernelCo/robot_control",
            filename=fname,
            repo_type="dataset",
            local_dir=str(PROJECT_ROOT / "data_raw"),
        )
        print(" done")

    # Copy/symlink from data_raw/data/ to data/
    raw_data_dir = PROJECT_ROOT / "data_raw" / "data"
    if raw_data_dir.exists():
        import shutil
        for f in raw_data_dir.glob("*.npz"):
            dest = DATA_DIR / f.name
            if not dest.exists():
                shutil.copy2(str(f), str(dest))
        print(f"Copied files to {DATA_DIR}")

    return list(DATA_DIR.glob("*.npz"))


def validate_dataset():
    """Validate all downloaded .npz files and report statistics."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"Total files: {len(files)}")

    label_counts = Counter()
    subject_ids = set()
    session_ids = set()
    bad_files = []

    for fpath in files:
        try:
            arr = np.load(str(fpath), allow_pickle=True)
            label_info = arr["label"].item()

            assert "feature_eeg" in arr, f"Missing feature_eeg in {fpath.name}"
            assert "feature_moments" in arr, f"Missing feature_moments in {fpath.name}"
            assert "label" in arr, f"Missing label in {fpath.name}"

            eeg = arr["feature_eeg"]
            moments = arr["feature_moments"]
            assert eeg.shape == (7499, 6), f"Bad EEG shape {eeg.shape} in {fpath.name}"
            assert moments.shape[0] == 72, f"Bad moments shape {moments.shape} in {fpath.name}"

            assert "label" in label_info, f"Missing label key in {fpath.name}"
            assert "subject_id" in label_info, f"Missing subject_id in {fpath.name}"
            assert "session_id" in label_info, f"Missing session_id in {fpath.name}"
            assert "duration" in label_info, f"Missing duration in {fpath.name}"

            label_counts[label_info["label"]] += 1
            subject_ids.add(label_info["subject_id"])
            session_ids.add(label_info["session_id"])

        except Exception as e:
            bad_files.append((fpath.name, str(e)))

    print(f"\nLabel distribution: {dict(label_counts)}")
    print(f"Unique subjects: {len(subject_ids)} -> {sorted(subject_ids)}")
    print(f"Unique sessions: {len(session_ids)}")
    print(f"Bad files: {len(bad_files)}")
    for bf in bad_files:
        print(f"  {bf}")

    # Save results
    with open(str(RESULTS_DIR / "label_distribution.txt"), "w") as f:
        f.write(f"Total files: {len(files)}\n\n")
        f.write("Label distribution:\n")
        for label, count in sorted(label_counts.items()):
            f.write(f"  {label}: {count}\n")
        f.write(f"\nBad files: {len(bad_files)}\n")
        for bf in bad_files:
            f.write(f"  {bf}\n")

    with open(str(RESULTS_DIR / "subject_ids.txt"), "w") as f:
        f.write(f"Unique subjects: {len(subject_ids)}\n\n")
        for sid in sorted(subject_ids):
            f.write(f"{sid}\n")

    # Return the exact label strings for downstream use
    return {
        "label_counts": dict(label_counts),
        "subject_ids": sorted(subject_ids),
        "session_count": len(session_ids),
        "bad_files": bad_files,
        "exact_labels": sorted(label_counts.keys()),
    }


if __name__ == "__main__":
    if not list(DATA_DIR.glob("*.npz")):
        print("No .npz files found in data/. Downloading...")
        download_dataset()
    else:
        print(f"Found {len(list(DATA_DIR.glob('*.npz')))} files in data/. Skipping download.")

    stats = validate_dataset()
    print(f"\nExact label strings: {stats['exact_labels']}")
    print("Validation complete.")
