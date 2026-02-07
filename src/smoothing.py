"""Phase 6: Temporal smoothing, confidence gating, and hysteresis."""
import numpy as np
from collections import deque, Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


class MajorityVoteSmoother:
    """Sliding window majority vote over recent predictions."""

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, prediction):
        self.buffer.append(prediction)
        if len(self.buffer) < self.window_size:
            return prediction
        counts = Counter(self.buffer)
        return counts.most_common(1)[0][0]

    def reset(self):
        self.buffer.clear()


class ConfidenceGate:
    """Two-stage confidence gating for BCI decisions."""

    def __init__(self, stage1_threshold=0.6, stage2_threshold=0.4):
        self.stage1_threshold = stage1_threshold
        self.stage2_threshold = stage2_threshold
        self.last_confident_action = "STOP"

    def decide(self, stage1_proba, stage1_pred, stage2_proba, stage2_pred, direction_to_action):
        """
        Returns action string: "FORWARD", "LEFT", "RIGHT", or "STOP".
        """
        if stage1_pred == 0 or stage1_proba < self.stage1_threshold:
            self.last_confident_action = "STOP"
            return "STOP"

        if stage2_proba >= self.stage2_threshold:
            action = direction_to_action.get(stage2_pred, "STOP")
            self.last_confident_action = action
            return action

        return self.last_confident_action


class HysteresisFilter:
    """Require min_consecutive identical predictions before switching action."""

    def __init__(self, min_consecutive=3):
        self.min_consecutive = min_consecutive
        self.current_action = "STOP"
        self.candidate = None
        self.candidate_count = 0

    def update(self, action):
        if action == self.current_action:
            self.candidate = None
            self.candidate_count = 0
            return self.current_action

        if action == self.candidate:
            self.candidate_count += 1
            if self.candidate_count >= self.min_consecutive:
                self.current_action = action
                self.candidate = None
                self.candidate_count = 0
                return self.current_action
        else:
            self.candidate = action
            self.candidate_count = 1

        return self.current_action

    def reset(self):
        self.current_action = "STOP"
        self.candidate = None
        self.candidate_count = 0


def test_smoothing_effectiveness():
    """Test smoothing on synthetic noisy predictions and measure flickering reduction."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate synthetic noisy predictions (simulating raw classifier output)
    np.random.seed(42)
    n_steps = 100
    # Ground truth: 30 STOP, 40 FORWARD, 30 LEFT
    ground_truth = (["STOP"] * 30 + ["FORWARD"] * 40 + ["LEFT"] * 30)
    actions = ["STOP", "FORWARD", "LEFT", "RIGHT"]

    # Add noise: 40% chance of wrong prediction
    raw_predictions = []
    for gt in ground_truth:
        if np.random.random() < 0.4:
            raw_predictions.append(np.random.choice(actions))
        else:
            raw_predictions.append(gt)

    # Count switches (flickering) in raw predictions
    raw_switches = sum(1 for i in range(1, len(raw_predictions)) if raw_predictions[i] != raw_predictions[i-1])

    # Apply smoothing pipeline
    smoother = MajorityVoteSmoother(window_size=5)
    hysteresis = HysteresisFilter(min_consecutive=3)

    smoothed_predictions = []
    for pred in raw_predictions:
        s1 = smoother.update(pred)
        s2 = hysteresis.update(s1)
        smoothed_predictions.append(s2)

    smoothed_switches = sum(1 for i in range(1, len(smoothed_predictions)) if smoothed_predictions[i] != smoothed_predictions[i-1])

    # Calculate metrics
    raw_accuracy = sum(1 for r, g in zip(raw_predictions, ground_truth) if r == g) / len(ground_truth)
    smoothed_accuracy = sum(1 for s, g in zip(smoothed_predictions, ground_truth) if s == g) / len(ground_truth)
    flicker_reduction = (1 - smoothed_switches / max(raw_switches, 1)) * 100

    results = {
        "raw_switches": raw_switches,
        "smoothed_switches": smoothed_switches,
        "flicker_reduction_pct": flicker_reduction,
        "raw_accuracy": raw_accuracy,
        "smoothed_accuracy": smoothed_accuracy,
        "raw_switches_per_sec": raw_switches / (n_steps * 0.5),  # 0.5s per step
        "smoothed_switches_per_sec": smoothed_switches / (n_steps * 0.5),
    }

    print("Smoothing Effectiveness Test:")
    print(f"  Raw switches: {raw_switches} ({results['raw_switches_per_sec']:.1f}/s)")
    print(f"  Smoothed switches: {smoothed_switches} ({results['smoothed_switches_per_sec']:.1f}/s)")
    print(f"  Flicker reduction: {flicker_reduction:.1f}%")
    print(f"  Raw accuracy: {raw_accuracy:.3f}")
    print(f"  Smoothed accuracy: {smoothed_accuracy:.3f}")

    with open(str(RESULTS_DIR / "smoothing_comparison.txt"), "w") as f:
        f.write("Smoothing Effectiveness Comparison\n")
        f.write("=" * 40 + "\n\n")
        for k, v in results.items():
            if isinstance(v, float):
                f.write(f"  {k}: {v:.3f}\n")
            else:
                f.write(f"  {k}: {v}\n")
        f.write(f"\nParameters:\n")
        f.write(f"  MajorityVote window_size: 5\n")
        f.write(f"  Hysteresis min_consecutive: 3\n")
        f.write(f"  Test: 100 steps, 40% noise rate\n")

    return results


if __name__ == "__main__":
    test_smoothing_effectiveness()
