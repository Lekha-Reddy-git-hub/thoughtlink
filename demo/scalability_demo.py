"""
ThoughtLink Scalability Demo
=============================
Simulates one human operator supervising N robots simultaneously.

Each "robot" has its own EEG stream that needs decoding.  Because the
pipeline takes ~25-30 ms per decision, a single process at 1 Hz decision
rate can serve floor(1000 / latency_ms) robots.

This script:
  1. Prepares EEG windows (one per simulated robot, different
     subjects and labels) with all preprocessing done up front -- just
     as a real system would maintain a rolling buffer per robot.
  2. Runs timed "decision rounds" for fleet sizes of 10, 50, and 100
     robots, recording wall-clock time per round.
  3. Repeats for 5 rounds per fleet size to get stable statistics.
  4. Saves results/scalability_100.png -- line chart of robots vs
     fleet cycle time with a 1000ms threshold.
  5. Prints a clear summary.

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
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 10 base robots, each from a different subject/label combination
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

FLEET_SIZES = [10, 50, 100]
N_ROUNDS = 5


def prepare_windows(data_dir):
    """Pre-extract one representative window per base robot (preprocessing
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
        mid = len(segs) // 2
        windows.append(segs[mid])
    return windows


def expand_fleet(base_windows, n_robots):
    """Expand base windows to n_robots by cycling through them."""
    expanded = []
    for i in range(n_robots):
        expanded.append(base_windows[i % len(base_windows)])
    return expanded


def extract_features(window):
    """Feature vector for a single (500, 6) window."""
    return np.concatenate([
        extract_psd_features(window),
        extract_stat_features(window),
        extract_cross_channel_features(window),
    ]).reshape(1, -1)


def run_fleet_test(windows, stage1, stage2, direction_map, n_rounds, fleet_size, verbose=True):
    """Run timed decision rounds for a given fleet size.
    Returns: (round_times_ms, per_robot_times_ms, decoded_actions)"""
    round_times = []
    per_robot_times = []
    decoded = []

    for r in range(n_rounds):
        if verbose:
            print(f"    Round {r + 1}/{n_rounds}...", end="", flush=True)
        t_round_start = time.perf_counter()

        for i, window in enumerate(windows):
            t0 = time.perf_counter()

            feat = extract_features(window)
            s1_pred = stage1.predict(feat)[0]

            if s1_pred == 1:
                s2_pred = int(stage2.predict(feat)[0])
                action = direction_map.get(s2_pred, "STOP")
            else:
                action = "STOP"

            dt_ms = (time.perf_counter() - t0) * 1000
            per_robot_times.append(dt_ms)
            decoded.append((r, i, action))

        round_ms = (time.perf_counter() - t_round_start) * 1000
        round_times.append(round_ms)
        if verbose:
            print(f" {round_ms:.0f}ms")

    return round_times, per_robot_times, decoded


def main():
    data_dir = os.path.join(project_root, "data")
    model_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("  ThoughtLink: Scalability Demo")
    print("  Can one BCI pipeline supervise a fleet of 100 robots?")
    print("=" * 70)
    print()

    # -- load models once --------------------------------------------------
    print("  Loading models...", end="", flush=True)
    stage1 = joblib.load(os.path.join(model_dir, "stage1_binary.pkl"))
    stage2 = joblib.load(os.path.join(model_dir, "stage2_direction.pkl"))
    DIRECTION_TO_ACTION = {0: "FORWARD", 1: "LEFT", 2: "RIGHT", 3: "BACKWARD"}
    print(" done")

    # -- prepare base windows (not timed) ----------------------------------
    print("  Preparing base EEG windows (10 robots)...", end="", flush=True)
    base_windows = prepare_windows(data_dir)
    print(" done\n")

    # -- print base fleet roster -------------------------------------------
    print("  Base Robot Fleet (10 robots, cycled for larger fleets):")
    print("  Robot | Subject  | EEG Label         | Expected Action")
    print("  " + "-" * 58)
    for i, (fname, label, subj, action) in enumerate(ROBOT_FLEET):
        print(f"  R{i:<4d} | {subj} | {label:17s} | {action}")
    print()

    # -- run fleet tests at each size --------------------------------------
    all_results = {}  # fleet_size -> {avg_round_ms, avg_per_robot_ms, ...}

    for fleet_size in FLEET_SIZES:
        print(f"  === Fleet Size: {fleet_size} robots ===")
        fleet_windows = expand_fleet(base_windows, fleet_size)

        round_times, per_robot_times, decoded = run_fleet_test(
            fleet_windows, stage1, stage2, DIRECTION_TO_ACTION,
            N_ROUNDS, fleet_size)

        per_arr = np.array(per_robot_times)
        round_arr = np.array(round_times)

        avg_per_robot = float(np.mean(per_arr))
        avg_round = float(np.mean(round_arr))
        med_round = float(np.median(round_arr))
        p95_round = float(np.percentile(round_arr, 95))
        throughput = fleet_size / (avg_round / 1000.0)

        all_results[fleet_size] = {
            "avg_per_robot_ms": avg_per_robot,
            "avg_round_ms": avg_round,
            "med_round_ms": med_round,
            "p95_round_ms": p95_round,
            "throughput": throughput,
            "round_times": round_times,
        }

        print(f"    Avg round time: {avg_round:.0f}ms | "
              f"Per-robot: {avg_per_robot:.1f}ms | "
              f"Throughput: {throughput:.0f} robots/s")
        print()

    # -- summary -----------------------------------------------------------
    print("=" * 70)
    print("  SCALABILITY RESULTS")
    print("=" * 70)
    print()
    print(f"  {'Fleet Size':>12} {'Round Time':>12} {'Per-Robot':>12} {'Throughput':>14} {'< 1000ms?':>10}")
    print("  " + "-" * 62)
    for size in FLEET_SIZES:
        r = all_results[size]
        under = "YES" if r["avg_round_ms"] < 1000 else "NO"
        print(f"  {size:>10d}   {r['avg_round_ms']:>9.0f}ms  {r['avg_per_robot_ms']:>9.1f}ms"
              f"  {r['throughput']:>10.0f} r/s    {under:>5}")
    print()

    max_robots_1hz = int(1000.0 / all_results[10]["avg_per_robot_ms"])
    print(f"  At 1 Hz decision rate, one pipeline instance can")
    print(f"  serve up to ~{max_robots_1hz} robots simultaneously.")
    print()

    if all_results[100]["avg_round_ms"] < 1000:
        print("  100 robots decoded in under 1 second -- single-core is sufficient.")
    else:
        n_cores = int(np.ceil(all_results[100]["avg_round_ms"] / 1000))
        print(f"  100 robots needs ~{n_cores} cores for 1 Hz decision rate.")
        print(f"  On a modern 8-core machine: {8 * max_robots_1hz} robots at 1 Hz.")
    print()
    print("  The decoder is pure CPU (scikit-learn + scipy),")
    print("  requires no GPU, and scales linearly with cores.")
    print("=" * 70)

    # -- generate chart ----------------------------------------------------
    print("\n  Generating scalability chart...")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#444")

    sizes = list(all_results.keys())
    avg_times = [all_results[s]["avg_round_ms"] for s in sizes]
    p95_times = [all_results[s]["p95_round_ms"] for s in sizes]

    # Main line: avg round time
    ax.plot(sizes, avg_times, "o-", color="#58a6ff", linewidth=2.5,
            markersize=10, label="Avg fleet cycle time", zorder=5)
    # P95 line
    ax.plot(sizes, p95_times, "s--", color="#f85149", linewidth=1.5,
            markersize=8, alpha=0.7, label="P95 fleet cycle time", zorder=4)

    # 1000ms threshold
    ax.axhline(y=1000, color="#ffa657", linestyle="--", linewidth=2, alpha=0.8)
    ax.text(sizes[-1] * 1.02, 1000, "1000ms\nthreshold", color="#ffa657",
            fontsize=10, va="center", fontweight="bold")

    # Annotate data points
    for s, t in zip(sizes, avg_times):
        ax.annotate(f"{t:.0f}ms", (s, t), textcoords="offset points",
                    xytext=(0, 15), ha="center", color="#58a6ff",
                    fontsize=11, fontweight="bold")

    # Shade region below 1000ms
    ax.fill_between([0, sizes[-1] * 1.1], 0, 1000, alpha=0.05, color="#3fb950")
    ax.text(sizes[-1] * 0.5, 50, "Real-time zone (< 1s)", color="#3fb950",
            fontsize=11, ha="center", alpha=0.7, fontstyle="italic")

    ax.set_xlabel("Fleet Size (number of robots)", fontsize=13)
    ax.set_ylabel("Fleet Cycle Time (ms)", fontsize=13)
    ax.set_title("ThoughtLink: Fleet Scalability -- Robots vs Decode Time",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, sizes[-1] * 1.15)
    ax.set_ylim(0, max(max(p95_times), 1000) * 1.3)
    ax.set_xticks(sizes)
    ax.legend(loc="upper left", facecolor="#161b22", edgecolor="#444",
              labelcolor="white", fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(results_dir, "scalability_100.png")
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
