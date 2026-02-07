"""Phase 8: Brain-Robot Interface integration."""
import sys
import os
import time
import threading
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add BRI to path
BRI_SRC = str(PROJECT_ROOT / "brain-robot-interface" / "src")
if BRI_SRC not in sys.path:
    sys.path.insert(0, BRI_SRC)

# Import our pipeline (src/ is already on path when run from src/)
from pipeline import ThoughtLinkPipeline


def _import_bri():
    """Try to import BRI modules, return (Action, Controller) or raise."""
    from bri import Action, Controller
    return Action, Controller


ACTION_STR_MAP = None  # populated lazily


def _get_action_map():
    global ACTION_STR_MAP
    if ACTION_STR_MAP is not None:
        return ACTION_STR_MAP
    Action, _ = _import_bri()
    ACTION_STR_MAP = {
        "FORWARD": Action.FORWARD,
        "LEFT": Action.LEFT,
        "RIGHT": Action.RIGHT,
        "STOP": Action.STOP,
    }
    return ACTION_STR_MAP


def bci_policy_loop(ctrl, pipeline, npz_file, stop_event, set_action_fn):
    """
    Run BCI pipeline on an .npz file and send decoded actions to the robot.
    Paces output to simulate real-time (0.5s per window step).
    """
    action_map = _get_action_map()
    Action, _ = _import_bri()
    current_action = Action.STOP

    for action_str, confidence, latency_ms in pipeline.process_file(npz_file):
        if stop_event.is_set():
            break

        action = action_map.get(action_str, Action.STOP)
        set_action_fn(action)
        current_action = action

        print(f"  Action: {action_str:8s}  Confidence: {confidence:.2f}  Latency: {latency_ms:.1f}ms")

        # Simulate real-time: each window step = 0.5s
        sleep_time = max(0, 0.5 - (latency_ms / 1000.0))
        sleep_start = time.time()
        while time.time() - sleep_start < sleep_time:
            if stop_event.is_set():
                break
            set_action_fn(current_action)  # re-send to prevent hold_s timeout
            time.sleep(min(0.1, max(0, sleep_time - (time.time() - sleep_start))))

    set_action_fn(action_map.get("STOP", current_action))
    print("\n  Recording complete. Robot stopped.")


def run_bci_sim(npz_file, model_dir=None):
    """
    Main entry point: load models, start MuJoCo sim, run BCI loop.
    Returns pipeline metrics dict.
    """
    if model_dir is None:
        model_dir = str(PROJECT_ROOT / "models")

    Action, Controller = _import_bri()

    pipeline = ThoughtLinkPipeline()
    pipeline.load_models(
        os.path.join(model_dir, "stage1_binary.pkl"),
        os.path.join(model_dir, "stage2_direction.pkl"),
    )

    # Bundle dir: must point to the BRI repo's bundles
    bundle_dir = str(PROJECT_ROOT / "brain-robot-interface" / "bundles" / "g1_mjlab")

    ctrl = Controller(
        backend="sim",
        hold_s=1.0,
        forward_speed=0.4,
        yaw_rate=1.0,
        smooth_alpha=0.3,
        bundle_dir=bundle_dir,
    )

    print("  Starting MuJoCo simulation (G1 humanoid)...")
    ctrl.start()
    print("  MuJoCo viewer launched. Decoding brain signals...")

    stop_event = threading.Event()
    thread = threading.Thread(
        target=bci_policy_loop,
        args=(ctrl, pipeline, npz_file, stop_event, ctrl.set_action),
        daemon=True,
    )
    thread.start()

    try:
        while thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
        stop_event.set()
    finally:
        thread.join(timeout=2.0)
        ctrl.stop()

    metrics = pipeline.get_metrics()

    os.makedirs(os.path.join(str(PROJECT_ROOT), "results"), exist_ok=True)
    with open(os.path.join(str(PROJECT_ROOT), "results", "demo_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def run_fallback_demo(npz_file, model_dir=None):
    """
    Fallback if MuJoCo fails: matplotlib top-down grid visualization.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    if model_dir is None:
        model_dir = str(PROJECT_ROOT / "models")

    pipeline = ThoughtLinkPipeline()
    pipeline.load_models(
        os.path.join(model_dir, "stage1_binary.pkl"),
        os.path.join(model_dir, "stage2_direction.pkl"),
    )

    # Robot state
    x, y = 5.0, 5.0
    heading = 90  # degrees
    trail = [(x, y)]
    speed = 0.3
    turn_rate = 20

    fig, (ax_map, ax_conf) = plt.subplots(1, 2, figsize=(14, 6))
    plt.ion()

    actions_log = []
    confidences_log = []

    arr = np.load(npz_file, allow_pickle=True)
    label_info = arr["label"].item()
    gt_label = label_info["label"]

    for action_str, confidence, latency_ms in pipeline.process_file(npz_file):
        if action_str == "FORWARD":
            x += speed * np.cos(np.radians(heading))
            y += speed * np.sin(np.radians(heading))
        elif action_str == "LEFT":
            heading += turn_rate
        elif action_str == "RIGHT":
            heading -= turn_rate

        trail.append((x, y))
        actions_log.append(action_str)
        confidences_log.append(confidence)

        ax_map.clear()
        ax_map.set_xlim(0, 10)
        ax_map.set_ylim(0, 10)
        ax_map.set_aspect("equal")
        ax_map.set_title(f"Robot Position | GT: {gt_label} | Action: {action_str} | Conf: {confidence:.2f}")

        trail_arr = np.array(trail)
        ax_map.plot(trail_arr[:, 0], trail_arr[:, 1], "b-", alpha=0.3, linewidth=1)
        ax_map.plot(x, y, "ro", markersize=12)
        dx = 0.5 * np.cos(np.radians(heading))
        dy = 0.5 * np.sin(np.radians(heading))
        ax_map.arrow(x, y, dx, dy, head_width=0.15, head_length=0.1, fc="red", ec="red")

        ax_conf.clear()
        ax_conf.plot(confidences_log, "g-")
        ax_conf.set_title("Confidence Over Time")
        ax_conf.set_xlabel("Window")
        ax_conf.set_ylabel("Confidence")
        ax_conf.set_ylim(0, 1)

        plt.tight_layout()
        plt.pause(0.3)

    plt.ioff()
    results_dir = os.path.join(str(PROJECT_ROOT), "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "fallback_demo.png"), dpi=150)
    print(f"  Fallback demo saved to results/fallback_demo.png")
    plt.show()

    return pipeline.get_metrics()


def run_headless_demo(npz_file, model_dir=None):
    """Headless mode: just print actions to console."""
    if model_dir is None:
        model_dir = str(PROJECT_ROOT / "models")

    pipeline = ThoughtLinkPipeline()
    pipeline.load_models(
        os.path.join(model_dir, "stage1_binary.pkl"),
        os.path.join(model_dir, "stage2_direction.pkl"),
    )

    arr = np.load(npz_file, allow_pickle=True)
    label_info = arr["label"].item()
    print(f"  Ground Truth: {label_info['label']}")
    print(f"  Subject: {label_info['subject_id']}")
    print(f"  Duration: {label_info['duration']:.1f}s\n")

    for action, conf, lat in pipeline.process_file(npz_file):
        print(f"    Action: {action:8s}  Confidence: {conf:.2f}  Latency: {lat:.1f}ms")

    metrics = pipeline.get_metrics()
    return metrics


if __name__ == "__main__":
    import sys as _sys
    npz = _sys.argv[1] if len(_sys.argv) > 1 else str(PROJECT_ROOT / "data" / "0b2dbd41-10.npz")

    print("Testing headless mode...")
    metrics = run_headless_demo(npz)
    print(f"\nMetrics: {metrics}")
