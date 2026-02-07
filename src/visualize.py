"""Phase 9: Visualizations and interpretability."""
import os
import numpy as np
import joblib
from pathlib import Path
from collections import Counter
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

CHANNEL_NAMES = ["AFF6", "AFp2", "AFp1", "AFF5", "FCz", "CPz"]
FS = 500.0

# Actual label strings from the dataset
LABELS_5CLASS = ["Both Fists", "Left Fist", "Relax", "Right Fist", "Tongue Tapping"]


def plot_psd_per_class():
    """Plot average PSD per class for all 6 channels."""
    print("Generating PSD per class plots...")
    data = np.load(str(PROJECT_ROOT / "preprocessed_data.npz"), allow_pickle=True)
    X = data["X"]  # (n_windows, 500, 6)
    y = data["y"]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    fig.suptitle("Average PSD per Class (All 6 Channels)", fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    for idx, label in enumerate(LABELS_5CLASS):
        ax = axes[idx]
        mask = y == label
        windows = X[mask]

        # Subsample if too many
        if len(windows) > 500:
            rng = np.random.RandomState(42)
            sel = rng.choice(len(windows), 500, replace=False)
            windows = windows[sel]

        for ch in range(6):
            all_psd = []
            for w in windows:
                freqs, psd = welch(w[:, ch], fs=FS, nperseg=256)
                all_psd.append(psd)
            avg_psd = np.mean(all_psd, axis=0)
            ax.semilogy(freqs, avg_psd, color=colors[ch], label=CHANNEL_NAMES[ch], alpha=0.8)

        ax.set_title(label)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(0, 50)
        ax.axvline(8, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(30, color="gray", linestyle="--", alpha=0.3)
        if idx == 0:
            ax.set_ylabel("PSD (log scale)")
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "psd_per_class.png"), dpi=150)
    plt.close()
    print("  Saved psd_per_class.png")


def plot_tsne():
    """t-SNE visualization of feature space colored by label."""
    print("Generating t-SNE plot...")
    from sklearn.manifold import TSNE

    data = np.load(str(PROJECT_ROOT / "features.npz"), allow_pickle=True)
    X = data["X"]
    y = data["y"]

    # Subsample for speed
    n_max = 3000
    if len(X) > n_max:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), n_max, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
    else:
        X_sub = X
        y_sub = y

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"Both Fists": "C0", "Left Fist": "C1", "Relax": "C2", "Right Fist": "C3", "Tongue Tapping": "C4"}

    for label in LABELS_5CLASS:
        mask = y_sub == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label, alpha=0.5, s=10,
                   color=colors.get(label, "gray"))

    ax.set_title("t-SNE of EEG Feature Space (5 classes)")
    ax.legend(markerscale=3)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "feature_tsne.png"), dpi=150)
    plt.close()
    print("  Saved feature_tsne.png")


def plot_temporal_timeline():
    """Plot confidence and action timeline for a single test file."""
    print("Generating temporal timeline...")
    from preprocess import bandpass_filter, extract_active_segment, normalize_channels, segment_windows
    from features import extract_psd_features, extract_stat_features, extract_cross_channel_features
    from smoothing import MajorityVoteSmoother, ConfidenceGate, HysteresisFilter

    # Pick a test file
    test_files = sorted(DATA_DIR.glob("*.npz"))
    sample_file = test_files[0]
    arr = np.load(str(sample_file), allow_pickle=True)
    label_info = arr["label"].item()
    gt_label = label_info["label"]
    eeg_raw = arr["feature_eeg"]

    # Load models
    stage1 = joblib.load(str(MODELS_DIR / "stage1_binary.pkl"))
    stage2 = joblib.load(str(MODELS_DIR / "stage2_direction.pkl"))

    # Direction map
    DIRECTION_TO_ACTION = {0: "FORWARD", 1: "LEFT", 2: "RIGHT"}

    # Preprocess
    eeg_filtered = bandpass_filter(eeg_raw)
    duration = label_info["duration"]
    eeg_active = extract_active_segment(eeg_filtered, duration)
    eeg_norm = normalize_channels(eeg_active)
    windows = segment_windows(eeg_norm, 500, 250)

    # Raw predictions
    raw_actions = []
    raw_confidences = []
    smoother = MajorityVoteSmoother(5)
    hysteresis = HysteresisFilter(3)
    gate = ConfidenceGate(0.6, 0.4)
    smoothed_actions = []

    for w in windows:
        features = np.concatenate([
            extract_psd_features(w),
            extract_stat_features(w),
            extract_cross_channel_features(w),
        ]).reshape(1, -1)

        s1_pred = stage1.predict(features)[0]
        s1_proba = stage1.predict_proba(features)[0]
        s1_active = float(s1_proba[1]) if len(s1_proba) > 1 else float(s1_proba[0])

        s2_pred = 0
        s2_proba = 0.0
        if s1_pred == 1:
            s2_pred = int(stage2.predict(features)[0])
            s2_proba = float(np.max(stage2.predict_proba(features)[0]))

        raw_action = gate.decide(s1_active, s1_pred, s2_proba, s2_pred, DIRECTION_TO_ACTION)
        raw_actions.append(raw_action)

        conf = s2_proba if s1_pred == 1 else (1.0 - s1_active)
        raw_confidences.append(conf)

        s1 = smoother.update(raw_action)
        s2 = hysteresis.update(s1)
        smoothed_actions.append(s2)

    # Plot
    action_to_num = {"STOP": 0, "LEFT": 1, "FORWARD": 2, "RIGHT": 3}
    action_labels = ["STOP", "LEFT", "FORWARD", "RIGHT"]
    time_axis = np.arange(len(raw_actions)) * 0.5  # 0.5s per step

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Temporal Timeline: {sample_file.name} (GT: {gt_label})", fontsize=13)

    # Confidence
    ax1.plot(time_axis, raw_confidences, "g-", linewidth=1.5)
    ax1.set_ylabel("Confidence")
    ax1.set_ylim(0, 1)
    ax1.set_title("Confidence Score")
    ax1.axhline(0.6, color="red", linestyle="--", alpha=0.5, label="Stage1 threshold")
    ax1.axhline(0.4, color="orange", linestyle="--", alpha=0.5, label="Stage2 threshold")
    ax1.legend(fontsize=8)

    # Raw actions
    raw_nums = [action_to_num.get(a, 0) for a in raw_actions]
    ax2.step(time_axis, raw_nums, "b-", where="mid", linewidth=1.5)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(action_labels)
    ax2.set_title("Raw Predictions")

    # Smoothed actions
    smooth_nums = [action_to_num.get(a, 0) for a in smoothed_actions]
    ax3.step(time_axis, smooth_nums, "r-", where="mid", linewidth=2)
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(action_labels)
    ax3.set_title("Smoothed Predictions (MajorityVote + Hysteresis)")
    ax3.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "temporal_timeline.png"), dpi=150)
    plt.close()
    print("  Saved temporal_timeline.png")


def plot_channel_importance():
    """Feature importance from RandomForest model."""
    print("Generating channel importance plot...")

    # Load Stage 1 model (RandomForest)
    model = joblib.load(str(MODELS_DIR / "stage1_binary.pkl"))
    # Get the classifier from pipeline
    clf = model.named_steps["clf"]

    if not hasattr(clf, "feature_importances_"):
        print("  Model doesn't have feature_importances_, skipping.")
        return

    importances = clf.feature_importances_

    # Feature layout: 24 PSD + 42 Stat + 3 Cross = 69
    # PSD: 4 features per channel x 6 channels = 24
    # Stat: 7 features per channel x 6 channels = 42
    # Cross: 3 asymmetry features
    psd_names = ["theta", "alpha", "beta", "a/b"]
    stat_names = ["var", "MAV", "RMS", "peak", "kurt", "skew", "ZC"]

    feature_names = []
    for ch in range(6):
        for pn in psd_names:
            feature_names.append(f"{CHANNEL_NAMES[ch]}_{pn}")
    for ch in range(6):
        for sn in stat_names:
            feature_names.append(f"{CHANNEL_NAMES[ch]}_{sn}")
    feature_names.extend(["Asym_AFF5-AFF6", "Asym_AFp1-AFp2", "Diff_FCz-CPz"])

    # Group by channel
    channel_importance = {}
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        # PSD features: indices ch_idx*4 to ch_idx*4+3
        psd_imp = importances[ch_idx*4:ch_idx*4+4].sum()
        # Stat features: indices 24+ch_idx*7 to 24+ch_idx*7+6
        stat_imp = importances[24+ch_idx*7:24+ch_idx*7+7].sum()
        channel_importance[ch_name] = psd_imp + stat_imp

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Per-channel total importance
    channels = list(channel_importance.keys())
    values = list(channel_importance.values())
    colors = ["#ff7f7f" if ch in ["AFF6", "AFp2"] else "#7f7fff" if ch in ["AFp1", "AFF5"] else "#7fff7f" for ch in channels]
    ax1.barh(channels, values, color=colors)
    ax1.set_title("Feature Importance by Channel (Stage 1)")
    ax1.set_xlabel("Total Importance")

    # Top 15 individual features
    sorted_idx = np.argsort(importances)[::-1][:15]
    top_names = [feature_names[i] for i in sorted_idx]
    top_vals = importances[sorted_idx]
    ax2.barh(range(len(top_names)), top_vals)
    ax2.set_yticks(range(len(top_names)))
    ax2.set_yticklabels(top_names)
    ax2.set_title("Top 15 Individual Features")
    ax2.set_xlabel("Importance")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "channel_importance.png"), dpi=150)
    plt.close()
    print("  Saved channel_importance.png")


def plot_smoothing_comparison():
    """Side-by-side raw vs smoothed predictions for a test file."""
    print("Generating smoothing comparison plot...")
    from preprocess import bandpass_filter, extract_active_segment, normalize_channels, segment_windows
    from features import extract_psd_features, extract_stat_features, extract_cross_channel_features
    from smoothing import MajorityVoteSmoother, HysteresisFilter, ConfidenceGate

    test_files = sorted(DATA_DIR.glob("*.npz"))
    # Use a couple different files for variety
    sample_files = test_files[:3]

    fig, axes = plt.subplots(len(sample_files), 2, figsize=(16, 4 * len(sample_files)))
    fig.suptitle("Smoothing Effect: Raw vs Smoothed Predictions", fontsize=14)

    stage1 = joblib.load(str(MODELS_DIR / "stage1_binary.pkl"))
    stage2 = joblib.load(str(MODELS_DIR / "stage2_direction.pkl"))
    DIRECTION_TO_ACTION = {0: "FORWARD", 1: "LEFT", 2: "RIGHT"}
    action_to_num = {"STOP": 0, "LEFT": 1, "FORWARD": 2, "RIGHT": 3}

    for file_idx, fpath in enumerate(sample_files):
        arr = np.load(str(fpath), allow_pickle=True)
        label_info = arr["label"].item()
        gt = label_info["label"]
        eeg = arr["feature_eeg"]

        eeg_filt = bandpass_filter(eeg)
        if np.any(np.isnan(eeg_filt)):
            continue
        eeg_active = extract_active_segment(eeg_filt, label_info["duration"])
        eeg_norm = normalize_channels(eeg_active)
        windows = segment_windows(eeg_norm, 500, 250)

        raw_actions = []
        smoothed_actions = []
        smoother = MajorityVoteSmoother(5)
        hysteresis = HysteresisFilter(3)
        gate = ConfidenceGate(0.6, 0.4)

        for w in windows:
            features = np.concatenate([
                extract_psd_features(w),
                extract_stat_features(w),
                extract_cross_channel_features(w),
            ]).reshape(1, -1)

            s1_pred = stage1.predict(features)[0]
            s1_proba = stage1.predict_proba(features)[0]
            s1_active = float(s1_proba[1]) if len(s1_proba) > 1 else float(s1_proba[0])
            s2_pred = 0
            s2_proba = 0.0
            if s1_pred == 1:
                s2_pred = int(stage2.predict(features)[0])
                s2_proba = float(np.max(stage2.predict_proba(features)[0]))

            raw = gate.decide(s1_active, s1_pred, s2_proba, s2_pred, DIRECTION_TO_ACTION)
            raw_actions.append(raw)
            s = smoother.update(raw)
            smoothed_actions.append(hysteresis.update(s))

        t = np.arange(len(raw_actions)) * 0.5

        ax_raw = axes[file_idx, 0] if len(sample_files) > 1 else axes[0]
        ax_smooth = axes[file_idx, 1] if len(sample_files) > 1 else axes[1]

        raw_nums = [action_to_num.get(a, 0) for a in raw_actions]
        smooth_nums = [action_to_num.get(a, 0) for a in smoothed_actions]

        ax_raw.step(t, raw_nums, "b-", where="mid")
        ax_raw.set_yticks([0, 1, 2, 3])
        ax_raw.set_yticklabels(["STOP", "LEFT", "FWD", "RIGHT"])
        ax_raw.set_title(f"Raw ({fpath.name}, GT: {gt})")
        raw_switches = sum(1 for i in range(1, len(raw_actions)) if raw_actions[i] != raw_actions[i-1])
        ax_raw.text(0.02, 0.98, f"Switches: {raw_switches}", transform=ax_raw.transAxes,
                   va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat"))

        ax_smooth.step(t, smooth_nums, "r-", where="mid", linewidth=2)
        ax_smooth.set_yticks([0, 1, 2, 3])
        ax_smooth.set_yticklabels(["STOP", "LEFT", "FWD", "RIGHT"])
        ax_smooth.set_title(f"Smoothed ({fpath.name}, GT: {gt})")
        smooth_switches = sum(1 for i in range(1, len(smoothed_actions)) if smoothed_actions[i] != smoothed_actions[i-1])
        ax_smooth.text(0.02, 0.98, f"Switches: {smooth_switches}", transform=ax_smooth.transAxes,
                      va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat"))

    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "smoothing_comparison.png"), dpi=150)
    plt.close()
    print("  Saved smoothing_comparison.png")


def plot_channel_layout():
    """Simple head diagram showing 6 channel positions."""
    print("Generating channel layout diagram...")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw head outline
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2)

    # Nose
    ax.plot([0, 0.1, 0], [1, 1.15, 1], "k-", linewidth=2)

    # Ears
    ax.plot([-1, -1.1, -1], [0.1, 0, -0.1], "k-", linewidth=2)
    ax.plot([1, 1.1, 1], [0.1, 0, -0.1], "k-", linewidth=2)

    # Channel positions (approximate on 10-20 system)
    channels = {
        "AFF6": (0.35, 0.75, "red", "Right anterior frontal"),
        "AFp2": (0.15, 0.85, "red", "Right anterior frontopolar"),
        "AFp1": (-0.15, 0.85, "blue", "Left anterior frontopolar"),
        "AFF5": (-0.35, 0.75, "blue", "Left anterior frontal"),
        "FCz": (0.0, 0.3, "green", "Midline frontocentral"),
        "CPz": (0.0, -0.1, "green", "Midline centroparietal"),
    }

    for name, (x, y, color, desc) in channels.items():
        ax.plot(x, y, "o", markersize=20, color=color, zorder=5)
        ax.text(x, y, name, ha="center", va="center", fontsize=8, fontweight="bold", zorder=6)
        ax.text(x, y - 0.12, desc, ha="center", va="top", fontsize=6, color="gray")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_title("EEG Channel Layout (6 channels)\nRed=Right, Blue=Left, Green=Midline", fontsize=12)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "channel_layout.png"), dpi=150)
    plt.close()
    print("  Saved channel_layout.png")


def plot_cross_subject_accuracy():
    """Leave-one-subject-out accuracy for Stage 1."""
    print("Generating cross-subject accuracy plot...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score

    data = np.load(str(PROJECT_ROOT / "features.npz"), allow_pickle=True)
    X = data["X"]
    y_str = data["y"]
    subjects = data["subjects"]

    y_binary = np.array([0 if s == "Relax" else 1 for s in y_str])
    unique_subjects = sorted(set(subjects))

    subject_accuracies = {}
    for test_subj in unique_subjects:
        train_mask = subjects != test_subj
        test_mask = subjects == test_subj

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))
        ])
        model.fit(X[train_mask], y_binary[train_mask])
        y_pred = model.predict(X[test_mask])
        acc = accuracy_score(y_binary[test_mask], y_pred)
        subject_accuracies[test_subj] = acc

    fig, ax = plt.subplots(figsize=(10, 5))
    subjects_list = list(subject_accuracies.keys())
    accs = list(subject_accuracies.values())

    bars = ax.bar(range(len(subjects_list)), accs, color="steelblue")
    ax.set_xticks(range(len(subjects_list)))
    ax.set_xticklabels([s[:8] for s in subjects_list], rotation=45)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Leave-One-Subject-Out Accuracy (Stage 1 Binary)\nMean: {np.mean(accs):.3f}")
    ax.axhline(np.mean(accs), color="red", linestyle="--", label=f"Mean: {np.mean(accs):.3f}")
    ax.axhline(0.5, color="gray", linestyle=":", label="Random baseline")
    ax.legend()
    ax.set_ylim(0, 1)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f"{acc:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "cross_subject_accuracy.png"), dpi=150)
    plt.close()
    print("  Saved cross_subject_accuracy.png")


def generate_all():
    """Generate all visualization plots."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_psd_per_class()
    plot_tsne()
    plot_temporal_timeline()
    plot_channel_importance()
    plot_smoothing_comparison()
    plot_channel_layout()
    plot_cross_subject_accuracy()

    print("\nAll visualizations generated in results/")


if __name__ == "__main__":
    generate_all()
