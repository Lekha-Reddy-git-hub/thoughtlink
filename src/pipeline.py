"""Phase 7: End-to-end ThoughtLink pipeline."""
import os
import json
import time
import numpy as np
import joblib
from pathlib import Path
from collections import Counter

from preprocess import bandpass_filter, extract_active_segment, normalize_channels, segment_windows
from features import extract_psd_features, extract_stat_features, extract_cross_channel_features
from smoothing import MajorityVoteSmoother, ConfidenceGate, HysteresisFilter


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ThoughtLinkPipeline:
    """
    Complete pipeline: raw .npz file -> stream of Action commands.

    Usage:
        pipeline = ThoughtLinkPipeline()
        pipeline.load_models("models/stage1_binary.pkl", "models/stage2_direction.pkl")

        for action, confidence, latency_ms, phase in pipeline.process_file("data/0b2dbd41-10.npz"):
            print(f"Action: {action} [{phase}], Confidence: {confidence:.2f}, Latency: {latency_ms:.1f}ms")
    """

    # Direction classifier output index -> Robot Action string
    # 4-class (Both Fists=0, Left Fist=1, Right Fist=2, Tongue Tapping=3):
    DIRECTION_TO_ACTION = {
        0: "FORWARD",    # "Both Fists"      -> go forward
        1: "LEFT",       # "Left Fist"       -> turn left
        2: "RIGHT",      # "Right Fist"      -> turn right
        3: "BACKWARD",   # "Tongue Tapping"  -> go backward
    }

    def __init__(
        self,
        window_size=500,
        window_overlap=250,
        filter_low=8.0,
        filter_high=30.0,
        fs=500.0,
        majority_vote_n=5,
        confidence_stage1=0.6,
        confidence_stage2=0.4,
        hysteresis_n=3,
    ):
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.fs = fs

        self.smoother = MajorityVoteSmoother(majority_vote_n)
        self.gate = ConfidenceGate(confidence_stage1, confidence_stage2)
        self.hysteresis = HysteresisFilter(hysteresis_n)

        self.stage1_model = None
        self.stage2_model = None

        # Metrics tracking
        self.latencies = []
        self.predictions = []
        self.confidences = []
        self.phases = []
        self._prev_action = "STOP"

    def load_models(self, stage1_path, stage2_path):
        self.stage1_model = joblib.load(str(stage1_path))
        self.stage2_model = joblib.load(str(stage2_path))

        # Try to load direction label map to verify mapping
        label_map_path = Path(stage1_path).parent.parent / "direction_label_map.npz"
        if label_map_path.exists():
            lm = np.load(str(label_map_path), allow_pickle=True)
            labels = list(lm["labels"])
            print(f"  Direction classes: {labels}")
            # Update DIRECTION_TO_ACTION based on actual label order
            action_map = {}
            for idx, label in enumerate(labels):
                if "Both" in label:
                    action_map[idx] = "FORWARD"
                elif "Left" in label:
                    action_map[idx] = "LEFT"
                elif "Right" in label:
                    action_map[idx] = "RIGHT"
                elif "Tongue" in label:
                    action_map[idx] = "BACKWARD"
                else:
                    action_map[idx] = "STOP"
            self.DIRECTION_TO_ACTION = action_map
            print(f"  Action mapping: {self.DIRECTION_TO_ACTION}")

    def process_file(self, npz_path):
        """
        Generator: yields (action_str, confidence, latency_ms, phase) for each window.
        Phase is one of "INITIATION", "SUSTAINED", or "RELEASE".
        """
        arr = np.load(str(npz_path), allow_pickle=True)
        eeg_raw = arr["feature_eeg"]
        label_info = arr["label"].item()

        # Preprocess
        eeg_filtered = bandpass_filter(eeg_raw, self.filter_low, self.filter_high, self.fs)

        if np.any(np.isnan(eeg_filtered)) or np.any(np.isinf(eeg_filtered)):
            yield "STOP", 0.0, 0.0
            return

        duration = label_info["duration"]
        eeg_active = extract_active_segment(eeg_filtered, duration)
        eeg_norm = normalize_channels(eeg_active)

        windows = segment_windows(eeg_norm, self.window_size, self.window_overlap)

        if len(windows) == 0:
            if eeg_norm.shape[0] > 0:
                padded = np.zeros((self.window_size, eeg_norm.shape[1]))
                padded[:eeg_norm.shape[0]] = eeg_norm
                windows = [padded]
            else:
                yield "STOP", 0.0, 0.0
                return

        # Reset smoothing state for each file
        self.smoother.reset()
        self.hysteresis.reset()
        self._prev_action = "STOP"

        for window in windows:
            t_start = time.perf_counter()

            # Extract features
            features = np.concatenate([
                extract_psd_features(window),
                extract_stat_features(window),
                extract_cross_channel_features(window),
            ]).reshape(1, -1)

            # Stage 1: rest vs active
            s1_pred = self.stage1_model.predict(features)[0]
            s1_proba = self.stage1_model.predict_proba(features)[0]
            s1_active_prob = float(s1_proba[1]) if len(s1_proba) > 1 else float(s1_proba[0])

            # Stage 2: direction (only if active)
            s2_pred = 0
            s2_proba = 0.0
            if s1_pred == 1:
                s2_pred = int(self.stage2_model.predict(features)[0])
                s2_proba_all = self.stage2_model.predict_proba(features)[0]
                s2_proba = float(np.max(s2_proba_all))

            # Confidence gating
            raw_action = self.gate.decide(
                s1_active_prob, s1_pred, s2_proba, s2_pred, self.DIRECTION_TO_ACTION
            )

            # Majority vote smoothing
            smoothed_action = self.smoother.update(raw_action)

            # Hysteresis
            final_action = self.hysteresis.update(smoothed_action)

            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000

            confidence = s2_proba if s1_pred == 1 else (1.0 - s1_active_prob)

            # Phase detection
            if self._prev_action == "STOP" and final_action != "STOP":
                phase = "INITIATION"
            elif self._prev_action != "STOP" and final_action == "STOP":
                phase = "RELEASE"
            else:
                phase = "SUSTAINED"
            self._prev_action = final_action

            self.latencies.append(latency_ms)
            self.predictions.append(final_action)
            self.confidences.append(confidence)
            self.phases.append(phase)

            yield final_action, confidence, latency_ms, phase

    def get_metrics(self):
        return {
            "avg_latency_ms": float(np.mean(self.latencies)) if self.latencies else 0,
            "max_latency_ms": float(np.max(self.latencies)) if self.latencies else 0,
            "predictions_count": len(self.predictions),
            "action_distribution": dict(Counter(self.predictions)),
            "avg_confidence": float(np.mean(self.confidences)) if self.confidences else 0,
            "phase_distribution": dict(Counter(self.phases)) if self.phases else {},
        }


def run_pipeline_test():
    """Test the pipeline on a few files and report metrics."""
    import os

    pipeline = ThoughtLinkPipeline()
    pipeline.load_models(
        str(PROJECT_ROOT / "models" / "stage1_binary.pkl"),
        str(PROJECT_ROOT / "models" / "stage2_direction.pkl"),
    )

    data_dir = PROJECT_ROOT / "data"
    test_files = sorted(data_dir.glob("*.npz"))[:10]  # Test on first 10 files

    print(f"\nTesting pipeline on {len(test_files)} files...")
    for fpath in test_files:
        arr = np.load(str(fpath), allow_pickle=True)
        label_info = arr["label"].item()
        gt_label = label_info["label"]

        actions = []
        for action, conf, lat, phase in pipeline.process_file(str(fpath)):
            actions.append(action)

        action_dist = Counter(actions)
        most_common = action_dist.most_common(1)[0][0] if actions else "NONE"
        print(f"  {fpath.name}: GT={gt_label:15s} -> Predicted={most_common:8s} ({dict(action_dist)})")

    metrics = pipeline.get_metrics()
    print(f"\nPipeline Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Save metrics
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(str(results_dir / "pipeline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to results/pipeline_metrics.json")

    return metrics


if __name__ == "__main__":
    run_pipeline_test()
