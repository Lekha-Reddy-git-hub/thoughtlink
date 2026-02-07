"""
ThoughtLink Setup Script
========================
Downloads the dataset from HuggingFace and trains both classifier models.
Run once after cloning:  uv run python setup_project.py

Steps:
  1. Download 900 EEG recordings from KernelCo/robot_control (~536 MB)
  2. Preprocess: bandpass filter, segment, window
  3. Extract features (69 per window)
  4. Train Stage 1 classifier (Rest vs Active)
  5. Train Stage 2 classifier (Direction, 3-class)
  6. Clone brain-robot-interface for MuJoCo demos (optional)
"""
import sys
import os
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

sys.path.insert(0, str(SRC_DIR))


def banner(text, char="=", width=70):
    print()
    print(char * width)
    print(f"  {text}")
    print(char * width)
    print()


def run_step(description, func):
    """Run a setup step, printing status and timing."""
    banner(description)
    sys.stdout.flush()
    t0 = time.time()
    result = func()
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s", flush=True)
    return result


def step_download():
    from download import download_dataset, validate_dataset

    if len(list(DATA_DIR.glob("*.npz"))) >= 900:
        print("  Dataset already downloaded (900 files found). Skipping.")
        stats = validate_dataset()
    else:
        download_dataset()
        stats = validate_dataset()
    return stats


def step_preprocess():
    from preprocess import preprocess_all
    X, y, subjects = preprocess_all()
    print(f"  Windows: {len(y)}  Subjects: {len(set(subjects))}")
    return X, y, subjects


def step_features():
    from features import build_feature_matrix
    X_feat, y, subjects = build_feature_matrix()
    print(f"  Feature matrix: {X_feat.shape}")
    return X_feat, y, subjects


def step_train_stage1():
    from stage1_binary import train_stage1
    train_stage1()


def step_train_stage2():
    from stage2_direction import train_stage2
    train_stage2()


def step_clone_bri():
    bri_dir = PROJECT_ROOT / "brain-robot-interface"
    if bri_dir.exists():
        print("  brain-robot-interface/ already exists. Skipping clone.")
        return True

    print("  Cloning Nabla7/brain-robot-interface for MuJoCo demos...")
    try:
        subprocess.run(
            ["git", "clone", "https://github.com/Nabla7/brain-robot-interface.git",
             str(bri_dir)],
            check=True, capture_output=True, text=True,
        )
        print("  Cloned successfully.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  Could not clone: {e}")
        print("  MuJoCo demos will not be available, but headless mode works.")
        return False


def step_verify():
    """Run pipeline on one file to verify everything works."""
    from pipeline import ThoughtLinkPipeline
    from collections import Counter

    pipeline = ThoughtLinkPipeline()
    pipeline.load_models(
        str(MODELS_DIR / "stage1_binary.pkl"),
        str(MODELS_DIR / "stage2_direction.pkl"),
    )

    test_file = sorted(DATA_DIR.glob("*.npz"))[0]
    actions = []
    for action, conf, lat, phase in pipeline.process_file(str(test_file)):
        actions.append(action)

    dist = Counter(actions)
    dominant = dist.most_common(1)[0][0]
    metrics = pipeline.get_metrics()

    print(f"  Test file: {test_file.name}")
    print(f"  Decoded: {dominant} ({dict(dist)})")
    print(f"  Avg latency: {metrics['avg_latency_ms']:.1f}ms")
    print(f"  Pipeline OK.")


def main():
    banner("ThoughtLink Setup", "=", 70)
    print("  This script downloads the dataset and trains all models.")
    print("  Total time: ~5-10 minutes depending on network speed.")

    t_total = time.time()

    run_step("Step 1/6: Download dataset from HuggingFace", step_download)
    run_step("Step 2/6: Preprocess EEG recordings", step_preprocess)
    run_step("Step 3/6: Extract features (69 per window)", step_features)
    run_step("Step 4/6: Train Stage 1 classifier (Rest vs Active)", step_train_stage1)
    run_step("Step 5/6: Train Stage 2 classifier (Direction)", step_train_stage2)
    run_step("Step 6/6: Clone brain-robot-interface (MuJoCo)", step_clone_bri)

    banner("Verification", "-")
    step_verify()

    total_elapsed = time.time() - t_total

    banner("Setup Complete!")
    print(f"  Total time: {total_elapsed:.0f}s")
    print()
    print("  Files created:")
    print(f"    data/            {len(list(DATA_DIR.glob('*.npz')))} EEG recordings")
    print(f"    models/          stage1_binary.pkl, stage2_direction.pkl")
    print()
    print("  Next steps:")
    print("    uv run python demo/run_all.py          # All 4 actions with MuJoCo")
    print("    uv run python demo/override_demo.py    # Autonomous override scenario")
    print("    uv run python demo/scalability_demo.py  # Fleet scalability test")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
