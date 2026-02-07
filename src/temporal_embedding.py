"""Temporal Embedding: Visualize how brain activity evolves within a recording.
One file per intent type, compute features per window, PCA to 2D, draw trajectory
with arrows colored by phase (rest->initiation->sustained).
Saves results/intent_evolution.png."""
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from preprocess import bandpass_filter, extract_active_segment, normalize_channels, segment_windows
from features import extract_psd_features, extract_stat_features, extract_cross_channel_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# One representative file per intent type
INTENT_FILES = {
    "Left Fist":       "2562e7bd-14.npz",
    "Right Fist":      "0b2dbd41-34.npz",
    "Both Fists":      "4787dfb9-10.npz",
    "Tongue Tapping":  "0b2dbd41-16.npz",
    "Relax":           "2161ecb6-12.npz",
}

INTENT_COLORS = {
    "Left Fist":       "#4285f4",
    "Right Fist":      "#ea4335",
    "Both Fists":      "#34a853",
    "Tongue Tapping":  "#fbbc04",
    "Relax":           "#9aa0a6",
}

INTENT_ACTIONS = {
    "Left Fist":       "LEFT",
    "Right Fist":      "RIGHT",
    "Both Fists":      "FORWARD",
    "Tongue Tapping":  "BACKWARD",
    "Relax":           "STOP",
}

PHASE_MARKERS = {
    "rest":        "o",
    "initiation":  "^",
    "sustained":   "s",
}


def extract_windows_and_features(npz_path):
    """Extract per-window features from a single file, return features + window indices."""
    arr = np.load(str(npz_path), allow_pickle=True)
    eeg = arr["feature_eeg"]
    info = arr["label"].item()

    filtered = bandpass_filter(eeg)
    if np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
        return None, None
    active = extract_active_segment(filtered, info["duration"])
    normed = normalize_channels(active)
    windows = segment_windows(normed, 500, 250)

    feats = []
    for w in windows:
        feat = np.concatenate([
            extract_psd_features(w),
            extract_stat_features(w),
            extract_cross_channel_features(w),
        ])
        feats.append(feat)

    return np.array(feats), info


def assign_phases(n_windows, label):
    """Assign phase labels to each window based on position and label."""
    phases = []
    for i in range(n_windows):
        if label == "Relax":
            phases.append("rest")
        elif i < 2:
            phases.append("rest")
        elif i == 2:
            phases.append("initiation")
        else:
            phases.append("sustained")
    return phases


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / "data"

    print("=" * 70)
    print("  Temporal Embedding: Intent Evolution Visualization")
    print("=" * 70)

    # Collect all features for PCA fitting
    all_features = []
    intent_data = {}

    for label, fname in INTENT_FILES.items():
        fpath = data_dir / fname
        feats, info = extract_windows_and_features(fpath)
        if feats is None:
            print(f"  Skipping {fname} (filter error)")
            continue

        phases = assign_phases(len(feats), label)
        intent_data[label] = {"features": feats, "phases": phases, "info": info}
        all_features.append(feats)
        print(f"  {label:17s}: {len(feats)} windows from {fname}")

    # Fit PCA on all windows jointly
    X_all = np.vstack(all_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    print(f"\n  PCA explained variance: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%}")

    # Split back into per-intent
    idx = 0
    for label in intent_data:
        n = len(intent_data[label]["features"])
        intent_data[label]["pca"] = X_2d[idx:idx + n]
        idx += n

    # Plot
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    phase_colors = {
        "rest":        "#666666",
        "initiation":  "#ffffff",
        "sustained":   None,  # use intent color
    }

    for label, data in intent_data.items():
        pts = data["pca"]
        phases = data["phases"]
        color = INTENT_COLORS[label]
        action = INTENT_ACTIONS[label]

        # Draw trajectory line
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, alpha=0.3, linewidth=1.5)

        # Draw arrows along trajectory
        for i in range(len(pts) - 1):
            dx = pts[i+1, 0] - pts[i, 0]
            dy = pts[i+1, 1] - pts[i, 1]
            ax.annotate("", xy=(pts[i+1, 0], pts[i+1, 1]),
                        xytext=(pts[i, 0], pts[i, 1]),
                        arrowprops=dict(arrowstyle="->", color=color, alpha=0.5, lw=1.2))

        # Draw points colored by phase
        for i, (pt, phase) in enumerate(zip(pts, phases)):
            marker = PHASE_MARKERS[phase]
            if phase == "sustained":
                fc = color
            elif phase == "initiation":
                fc = "white"
            else:
                fc = "#666666"
            ec = color
            size = 80 if phase == "initiation" else 40
            ax.scatter(pt[0], pt[1], marker=marker, s=size, c=fc,
                      edgecolors=ec, linewidths=1.5, zorder=5)

        # Label start and end
        ax.annotate(f"{action}\nstart", xy=(pts[0, 0], pts[0, 1]),
                   fontsize=8, color=color, fontweight="bold",
                   ha="center", va="bottom",
                   xytext=(0, 8), textcoords="offset points")
        ax.annotate(f"end", xy=(pts[-1, 0], pts[-1, 1]),
                   fontsize=8, color=color,
                   ha="center", va="top",
                   xytext=(0, -8), textcoords="offset points")

    # Legend for intents
    for label, color in INTENT_COLORS.items():
        action = INTENT_ACTIONS[label]
        ax.plot([], [], "o-", color=color, markersize=6, label=f"{label} -> {action}")

    # Legend for phases
    ax.scatter([], [], marker="o", s=40, c="#666666", edgecolors="white",
              label="Rest phase")
    ax.scatter([], [], marker="^", s=80, c="white", edgecolors="white",
              label="Initiation")
    ax.scatter([], [], marker="s", s=40, c="#aaaaaa", edgecolors="white",
              label="Sustained")

    legend = ax.legend(loc="upper left", fontsize=10, facecolor="#16213e",
                      edgecolor="#444", labelcolor="white", ncol=2)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
                 fontsize=12, color="white")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
                 fontsize=12, color="white")
    ax.set_title("Intent Evolution: How Brain Activity Changes Within a Recording",
                fontsize=14, fontweight="bold", color="white", pad=15)

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")

    # Subtitle
    fig.text(0.5, 0.01,
             "Each trajectory shows one EEG recording projected to feature space. "
             "Arrows show temporal progression. Phases: rest (circle) -> initiation (triangle) -> sustained (square).",
             ha="center", fontsize=9, color="#aaaaaa")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = str(RESULTS_DIR / "intent_evolution.png")
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
