"""
ThoughtLink Demo -- Brain-to-Robot Control
Usage: python demo/run_demo.py <path_to_npz_file>
       python demo/run_demo.py data/0b2dbd41-10.npz
"""
import sys
import os

# Add project root and src/ to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo/run_demo.py <path_to_npz_file>")
        print("Example: python demo/run_demo.py data/0b2dbd41-10.npz")
        sys.exit(1)

    npz_file = sys.argv[1]
    if not os.path.exists(npz_file):
        npz_file = os.path.join(project_root, sys.argv[1])
        if not os.path.exists(npz_file):
            print(f"Error: File not found: {sys.argv[1]}")
            sys.exit(1)

    print("=" * 60)
    print("  ThoughtLink: Brain-to-Robot Control System")
    print("  Hack Nation 2026 -- Challenge 9")
    print("=" * 60)

    import numpy as np
    arr = np.load(npz_file, allow_pickle=True)
    label_info = arr["label"].item()
    print(f"\n  EEG File: {os.path.basename(npz_file)}")
    print(f"  Ground Truth Label: {label_info['label']}")
    print(f"  Subject: {label_info['subject_id']}")
    print(f"  Duration: {label_info['duration']:.1f}s")
    print(f"  EEG Shape: {arr['feature_eeg'].shape}")
    print()

    # Try MuJoCo sim first
    try:
        from integration import run_bci_sim
        print("  [MuJoCo mode] Starting G1 humanoid simulation...")
        metrics = run_bci_sim(npz_file)
    except Exception as e:
        print(f"  [MuJoCo unavailable: {e}]")
        print("  [Fallback mode] Using matplotlib visualizer...")
        try:
            from integration import run_fallback_demo
            metrics = run_fallback_demo(npz_file)
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            print("  Running headless pipeline only...")
            from pipeline import ThoughtLinkPipeline
            pipeline = ThoughtLinkPipeline()
            pipeline.load_models(
                os.path.join(project_root, "models", "stage1_binary.pkl"),
                os.path.join(project_root, "models", "stage2_direction.pkl"),
            )
            for action, conf, lat, phase in pipeline.process_file(npz_file):
                print(f"    Action: {action:8s} [{phase:10s}]  Confidence: {conf:.2f}  Latency: {lat:.1f}ms")
            metrics = pipeline.get_metrics()

    print("\n" + "=" * 60)
    print("  Pipeline Metrics")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        else:
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
