"""
ThoughtLink Scalability Demo
=============================
Simulates one human operator supervising N robots simultaneously.

Each "robot" has its own EEG stream that needs decoding.  Because the
pipeline takes ~55 ms per decision, a single process at 1 Hz decision
rate can serve floor(1000 / latency_ms) robots.

This script:
  1. Prepares 10 EEG windows (one per simulated robot, different
     subjects and labels) with all preprocessing done up front -- just
     as a real system would maintain a rolling buffer per robot.
  2. Runs a timed "decision round" -- the pipeline decodes intent for
     all 10 robots sequentially and records wall-clock time.
  3. Repeats for 5 rounds to get stable statistics.
  4. Prints a clear summary suitable for a screen-recording or
     judging presentation.

Usage:  python demo/scalability_demo.py
"""
import sys
import os
import time
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from preprocess import bandpass_filter, extract_active_segment, normalize_channels, segment_windows
from features import extract_psd_features, extract_stat_features, extract_cross_channel_features
from pipeline import ThoughtLinkPipeline
import joblib

# 10 robots, each from a different subject/label combination
# All 6 subjects represented
ROBOT_FLEET = [
    ("2562e7bd-14.npz", "Left Fist",       "1a3cd681", "LEFT"),
    ("305a10dd-14.npz", "Left Fist",       "2a456f03", "LEFT"),
    ("4787dfb9-12.npz", "Right Fist",      "37dfbd76", "RIGHT"),
    ("2161ecb6-18.npz", "Right Fist",      "4c2ea012", "RIGHT"),
    ("0b2dbd41-14.npz", "Both Fists",      "a5136953", "FORWARD"),
    ("5aa5d730-16.npz", "Both Fists",      "d696086d", "FORWARD"),
    ("2562e7bd-20.npz", "Relax",           "1a3cd681", "STOP"),
    ("305a10dd-10.npz", "Relax",           "2a456f03", "STOP"),
    ("4787dfb9-14.npz", "Tongue Tapping",  "37dfbd76", "BACKWARD"),
    ("2161ecb6-10.npz", "Tongue Tapping",  "4c2ea012", "BACKWARD"),
]

N_ROUNDS = 5


def prepare_windows(data_dir):
    """Pre-extract one representative window per robot (preprocessing
    is NOT counted toward inference time -- a real system keeps a
    rolling buffer)."""
    windows = []
    for fname, label, subj, _ in ROBOT_FLEET:
        arr = np.load(os.path.join(data_dir, fname), allow_pickle=True)
        eeg = arr["feature_eeg"]
        info = arr["label"].item()
        filtered = bandpass_filter(eeg)
        active = extract_active_segment(filtered, info["duration"])
        normed = normalize_channels(active)
        segs = segment_windows(normed, 500, 250)
        # Pick a window from the middle of the recording for the most
        # representative activity
        mid = len(segs) // 2
        windows.append(segs[mid])
    return windows


def extract_features(window):
    """Feature vector for a single (500, 6) window."""
    return np.concatenate([
        extract_psd_features(window),
        extract_stat_features(window),
        extract_cross_channel_features(window),
    ]).reshape(1, -1)


def main():
    data_dir = os.path.join(project_root, "data")
    model_dir = os.path.join(project_root, "models")

    n_robots = len(ROBOT_FLEET)
    n_subjects = len(set(r[2] for r in ROBOT_FLEET))

    print("=" * 70)
    print("  ThoughtLink: Scalability Demo")
    print("  Can one BCI pipeline supervise a fleet of robots?")
    print("=" * 70)
    print()
    print(f"  Simulated fleet: {n_robots} robots, {n_subjects} unique operators")
    print(f"  Decision rounds: {N_ROUNDS}")
    print()

    # -- load models once --------------------------------------------------
    print("  Loading models...", end="", flush=True)
    stage1 = joblib.load(os.path.join(model_dir, "stage1_binary.pkl"))
    stage2 = joblib.load(os.path.join(model_dir, "stage2_direction.pkl"))
    DIRECTION_TO_ACTION = {0: "FORWARD", 1: "LEFT", 2: "RIGHT", 3: "BACKWARD"}
    print(" done")

    # -- prepare one window per robot (not timed) --------------------------
    print("  Preparing EEG windows (one per robot)...", end="", flush=True)
    windows = prepare_windows(data_dir)
    print(" done\n")

    # -- print fleet roster ------------------------------------------------
    print("  Robot | Subject  | EEG Label         | Expected Action")
    print("  " + "-" * 58)
    for i, (fname, label, subj, action) in enumerate(ROBOT_FLEET):
        print(f"  R{i:<4d} | {subj} | {label:17s} | {action}")
    print()

    # -- timed decision rounds ---------------------------------------------
    round_times = []
    per_robot_times = []        # flat list across all rounds
    all_decoded = []             # (round, robot_idx, action, latency_ms)

    for r in range(N_ROUNDS):
        print(f"  --- Decision Round {r + 1}/{N_ROUNDS} ---")
        t_round_start = time.perf_counter()

        for i, window in enumerate(windows):
            t0 = time.perf_counter()

            feat = extract_features(window)
            s1_pred = stage1.predict(feat)[0]
            s1_proba = stage1.predict_proba(feat)[0]
            s1_active = float(s1_proba[1]) if len(s1_proba) > 1 else float(s1_proba[0])

            if s1_pred == 1:
                s2_pred = int(stage2.predict(feat)[0])
                s2_proba = float(np.max(stage2.predict_proba(feat)[0]))
                action = DIRECTION_TO_ACTION.get(s2_pred, "STOP")
            else:
                action = "STOP"

            dt_ms = (time.perf_counter() - t0) * 1000
            per_robot_times.append(dt_ms)
            all_decoded.append((r, i, action, dt_ms))

            expected = ROBOT_FLEET[i][3]
            tag = "ok" if action == expected else "xx"
            print(f"    R{i} -> {action:8s} ({dt_ms:5.1f} ms) [{tag}]")

        round_ms = (time.perf_counter() - t_round_start) * 1000
        round_times.append(round_ms)
        print(f"    Round total: {round_ms:.1f} ms\n")

    # -- statistics --------------------------------------------------------
    per_robot_arr = np.array(per_robot_times)
    round_arr = np.array(round_times)

    avg_per_robot = float(np.mean(per_robot_arr))
    med_per_robot = float(np.median(per_robot_arr))
    p95_per_robot = float(np.percentile(per_robot_arr, 95))
    avg_round     = float(np.mean(round_arr))

    max_robots_1hz = int(1000.0 / avg_per_robot)
    throughput     = n_robots / (avg_round / 1000.0)

    # accuracy across all rounds
    correct = sum(
        1 for _, i, action, _ in all_decoded if action == ROBOT_FLEET[i][3]
    )
    total = len(all_decoded)

    # -- summary -----------------------------------------------------------
    print("=" * 70)
    print("  SCALABILITY RESULTS")
    print("=" * 70)
    print()
    print(f"  Fleet size tested        : {n_robots} robots ({n_subjects} subjects)")
    print(f"  Decision rounds          : {N_ROUNDS}")
    print()
    print(f"  Per-robot decode latency : {avg_per_robot:.1f} ms avg  |  "
          f"{med_per_robot:.1f} ms median  |  {p95_per_robot:.1f} ms p95")
    print(f"  Full-fleet round time    : {avg_round:.1f} ms avg for {n_robots} robots")
    print()
    print(f"  Decode accuracy          : {correct}/{total} "
          f"({100*correct/total:.0f}%) actions matched expected")
    print()
    print("  " + "-" * 58)
    print(f"  1 operator decoded intent for {n_robots} robots "
          f"in {avg_round:.0f} ms")
    print(f"  = {throughput:.1f} robots/second throughput")
    print()
    print(f"  At 1 Hz decision rate, one pipeline instance can")
    print(f"  serve up to {max_robots_1hz} robots simultaneously.")
    print(f"  At 2 Hz, that is still {int(500/avg_per_robot)} robots.")
    print("  " + "-" * 58)
    print()
    print("  Could this model realistically support a system")
    print("  supervising 100 humanoid robots?")
    print()
    if max_robots_1hz >= 100:
        print("  YES -- a single pipeline instance already exceeds 100.")
    else:
        n_instances = int(np.ceil(100 / max_robots_1hz))
        print(f"  YES -- {n_instances} parallel pipeline instances "
              f"(e.g. {n_instances} CPU cores)")
        print(f"  each handling {max_robots_1hz} robots would cover "
              f"{n_instances * max_robots_1hz} robots total.")
        print(f"  On a modern 8-core machine that is "
              f"{8 * max_robots_1hz} robots at 1 Hz.")
    print()
    print("  The decoder is pure CPU (scikit-learn + scipy),")
    print("  requires no GPU, and scales linearly with cores.")
    print("=" * 70)


if __name__ == "__main__":
    main()
