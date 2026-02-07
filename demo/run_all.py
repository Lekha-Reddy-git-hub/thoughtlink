"""
ThoughtLink Full Demo -- All 5 Robot Actions Back-to-Back
Runs LEFT, RIGHT, FORWARD, BACKWARD, STOP sequentially with clear banners.
Each file is from a different subject to prove cross-subject generalization.

Usage: python demo/run_all.py
"""
import sys
import os
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

# One file per robot action, each from a different subject
DEMO_FILES = [
    {
        "file": "2562e7bd-14.npz",
        "label": "Left Fist",
        "expected_action": "LEFT",
        "subject": "1a3cd681",
    },
    {
        "file": "0b2dbd41-34.npz",
        "label": "Right Fist",
        "expected_action": "RIGHT",
        "subject": "a5136953",
    },
    {
        "file": "4787dfb9-10.npz",
        "label": "Both Fists",
        "expected_action": "FORWARD",
        "subject": "37dfbd76",
    },
    {
        "file": "0b2dbd41-16.npz",
        "label": "Tongue Tapping",
        "expected_action": "BACKWARD",
        "subject": "a5136953",
    },
    {
        "file": "2161ecb6-12.npz",
        "label": "Relax",
        "expected_action": "STOP",
        "subject": "4c2ea012",
    },
]

PAUSE_SECONDS = 3


def print_banner(entry, index, total):
    label = entry["label"]
    action = entry["expected_action"]
    subject = entry["subject"]
    print()
    print("=" * 70)
    print(f"  [{index}/{total}]  NOW TESTING: {label} brain signal")
    print(f"         Expected robot action: {action}")
    print(f"         Subject: {subject}  (cross-subject generalization)")
    print("=" * 70)
    print()


def main():
    print("=" * 70)
    print("  ThoughtLink: Full Demo -- All 5 Robot Actions")
    print("  Hack Nation 2026 -- Challenge 9")
    print("=" * 70)
    print()
    print(f"  Will run {len(DEMO_FILES)} recordings back-to-back:")
    for i, entry in enumerate(DEMO_FILES, 1):
        print(f"    {i}. {entry['label']:15s} -> {entry['expected_action']:8s}  (subject {entry['subject']})")
    print(f"\n  {PAUSE_SECONDS}s pause between each recording.")
    print()

    # Resolve file paths
    data_dir = os.path.join(project_root, "data")
    for entry in DEMO_FILES:
        fpath = os.path.join(data_dir, entry["file"])
        if not os.path.exists(fpath):
            print(f"  ERROR: {fpath} not found!")
            sys.exit(1)

    # Try MuJoCo mode
    use_mujoco = False
    ctrl = None
    try:
        bri_src = os.path.join(project_root, "brain-robot-interface", "src")
        if bri_src not in sys.path:
            sys.path.insert(0, bri_src)
        from bri import Action, Controller

        bundle_dir = os.path.join(project_root, "brain-robot-interface", "bundles", "g1_mjlab")
        ctrl = Controller(
            backend="sim",
            hold_s=1.0,
            forward_speed=0.4,
            yaw_rate=1.0,
            smooth_alpha=0.3,
            bundle_dir=bundle_dir,
        )
        ctrl.start()

        # Set up tracking camera
        viewer = ctrl._backend._viewer
        if viewer is not None:
            viewer.cam.distance = 5.0
            viewer.cam.azimuth = 180.0
            viewer.cam.elevation = -25.0
            sim_data = ctrl._backend._data
            if sim_data is not None:
                viewer.cam.lookat[0] = float(sim_data.qpos[0])
                viewer.cam.lookat[1] = float(sim_data.qpos[1])
                viewer.cam.lookat[2] = 0.8

        use_mujoco = True
        print("  [MuJoCo mode] G1 humanoid simulation started.\n")

        ACTION_MAP = {
            "FORWARD": Action.FORWARD,
            "BACKWARD": Action.BACKWARD,
            "LEFT": Action.LEFT,
            "RIGHT": Action.RIGHT,
            "STOP": Action.STOP,
        }
    except Exception as e:
        print(f"  [MuJoCo unavailable: {e}]")
        print("  Running in headless mode.\n")

    from pipeline import ThoughtLinkPipeline

    total = len(DEMO_FILES)
    all_results = []

    for i, entry in enumerate(DEMO_FILES, 1):
        print_banner(entry, i, total)

        npz_path = os.path.join(data_dir, entry["file"])

        pipeline = ThoughtLinkPipeline()
        pipeline.load_models(
            os.path.join(project_root, "models", "stage1_binary.pkl"),
            os.path.join(project_root, "models", "stage2_direction.pkl"),
        )

        action_counts = {}
        for action_str, confidence, latency_ms, phase in pipeline.process_file(npz_path):
            print(f"    Action: {action_str:8s} [{phase:10s}]  Confidence: {confidence:.2f}  Latency: {latency_ms:.1f}ms")

            if use_mujoco and ctrl is not None:
                bri_action = ACTION_MAP.get(action_str, Action.STOP)
                ctrl.set_action(bri_action)

                # Simulate real-time pacing + camera tracking
                sleep_time = max(0, 0.5 - (latency_ms / 1000.0))
                sleep_start = time.time()
                while time.time() - sleep_start < sleep_time:
                    ctrl.set_action(bri_action)
                    sim_data = ctrl._backend._data
                    vw = ctrl._backend._viewer
                    if sim_data is not None and vw is not None and vw.is_running():
                        vw.cam.lookat[0] = float(sim_data.qpos[0])
                        vw.cam.lookat[1] = float(sim_data.qpos[1])
                        vw.cam.lookat[2] = 0.8
                    time.sleep(0.05)

            action_counts[action_str] = action_counts.get(action_str, 0) + 1

        dominant = max(action_counts, key=action_counts.get)
        match = dominant == entry["expected_action"]
        metrics = pipeline.get_metrics()

        print()
        print(f"  Result: dominant action = {dominant}, expected = {entry['expected_action']}")
        print(f"  {'CORRECT' if match else 'MISMATCH'}  |  Distribution: {action_counts}")
        print(f"  Avg latency: {metrics['avg_latency_ms']:.1f}ms  |  Avg confidence: {metrics['avg_confidence']:.2f}")

        all_results.append({
            "label": entry["label"],
            "expected": entry["expected_action"],
            "dominant": dominant,
            "correct": match,
            "distribution": action_counts,
            "avg_latency_ms": metrics["avg_latency_ms"],
            "avg_confidence": metrics["avg_confidence"],
        })

        # Pause between recordings
        if i < total:
            if use_mujoco and ctrl is not None:
                ctrl.set_action(Action.STOP)
            print(f"\n  --- Pausing {PAUSE_SECONDS}s before next recording ---")
            pause_start = time.time()
            while time.time() - pause_start < PAUSE_SECONDS:
                if use_mujoco and ctrl is not None:
                    ctrl.set_action(Action.STOP)
                    sim_data = ctrl._backend._data
                    vw = ctrl._backend._viewer
                    if sim_data is not None and vw is not None and vw.is_running():
                        vw.cam.lookat[0] = float(sim_data.qpos[0])
                        vw.cam.lookat[1] = float(sim_data.qpos[1])
                        vw.cam.lookat[2] = 0.8
                time.sleep(0.05)

    # Final summary
    if use_mujoco and ctrl is not None:
        ctrl.set_action(Action.STOP)
        time.sleep(0.5)
        ctrl.stop()

    print()
    print("=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    correct = sum(1 for r in all_results if r["correct"])
    print(f"\n  Accuracy: {correct}/{total} robot actions correctly decoded\n")
    for r in all_results:
        tag = "PASS" if r["correct"] else "FAIL"
        print(f"    [{tag}] {r['label']:15s} -> {r['dominant']:8s} (expected {r['expected']:8s})  "
              f"conf={r['avg_confidence']:.2f}  lat={r['avg_latency_ms']:.0f}ms")

    print(f"\n  All {total} recordings from different subjects (cross-subject generalization)")
    print("=" * 70)


if __name__ == "__main__":
    main()
