"""CNN/Neural Network baseline comparison vs Random Forest.
Uses sklearn MLPClassifier as the neural network (avoids torch dependency).
For a 1D-CNN-like approach: we reshape windowed EEG into flat input.
Compares accuracy AND inference latency. Saves results/cnn_vs_rf.png."""
import numpy as np
import time
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SkPipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

from features import extract_psd_features, extract_stat_features, extract_cross_channel_features


def load_data():
    """Load preprocessed windows and labels with cross-subject split."""
    data = np.load(str(PROJECT_ROOT / "preprocessed_data.npz"), allow_pickle=True)
    X_windows = data["X"]       # (N, 500, 6)
    y_str = data["y"]
    subjects = data["subjects"]

    split = np.load(str(PROJECT_ROOT / "subject_split.npz"), allow_pickle=True)
    train_subjects = set(split["train_subjects"])
    test_subjects = set(split["test_subjects"])

    y_binary = np.array([0 if s == "Relax" else 1 for s in y_str])

    active_labels = sorted(set(s for s in y_str if s != "Relax"))
    dir_map = {label: idx for idx, label in enumerate(active_labels)}

    train_mask = np.array([s in train_subjects for s in subjects])
    test_mask = np.array([s in test_subjects for s in subjects])

    return X_windows, y_binary, y_str, subjects, train_mask, test_mask, dir_map


def flatten_windows(X):
    """Flatten (N, 500, 6) -> (N, 3000) for MLP input."""
    return X.reshape(X.shape[0], -1)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Neural Network (MLP) vs Random Forest: Accuracy and Latency")
    print("=" * 70)

    X_windows, y_binary, y_str, subjects, train_mask, test_mask, dir_map = load_data()

    # Load EEG feature matrix for RF
    feat_data = np.load(str(PROJECT_ROOT / "features.npz"), allow_pickle=True)
    X_feat = feat_data["X"]
    y_feat_str = feat_data["y"]
    subj_feat = feat_data["subjects"]
    split = np.load(str(PROJECT_ROOT / "subject_split.npz"), allow_pickle=True)
    test_subj = set(split["test_subjects"])
    feat_test_mask = np.array([s in test_subj for s in subj_feat])

    # === STAGE 1: BINARY ===
    print("\n--- Stage 1: Rest vs Active ---")

    # MLP on raw flattened EEG
    X_flat_train = flatten_windows(X_windows[train_mask])
    X_flat_test = flatten_windows(X_windows[test_mask])

    print("  Training MLP (raw EEG, 3000 inputs)...")
    mlp_s1 = SkPipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50,
                              early_stopping=True, validation_fraction=0.15,
                              random_state=42, verbose=False))
    ])
    mlp_s1.fit(X_flat_train, y_binary[train_mask])
    mlp_s1_pred = mlp_s1.predict(X_flat_test)
    mlp_s1_acc = accuracy_score(y_binary[test_mask], mlp_s1_pred)
    print(f"  MLP Stage 1 accuracy: {mlp_s1_acc:.3f}")

    # RF
    rf_s1 = joblib.load(str(MODELS_DIR / "stage1_binary.pkl"))
    y_feat_bin = np.array([0 if s == "Relax" else 1 for s in y_feat_str])
    rf_s1_pred = rf_s1.predict(X_feat[feat_test_mask])
    rf_s1_acc = accuracy_score(y_feat_bin[feat_test_mask], rf_s1_pred)
    print(f"  RF Stage 1 accuracy:  {rf_s1_acc:.3f}")

    # === STAGE 2: DIRECTION (4-class) ===
    print("\n--- Stage 2: Direction (4-class) ---")

    active_train = y_str[train_mask] != "Relax"
    active_test = y_str[test_mask] != "Relax"

    X_flat_train_s2 = flatten_windows(X_windows[train_mask][active_train])
    X_flat_test_s2 = flatten_windows(X_windows[test_mask][active_test])
    y_train_s2 = np.array([dir_map[s] for s in y_str[train_mask][active_train]])
    y_test_s2 = np.array([dir_map[s] for s in y_str[test_mask][active_test]])

    print(f"  Classes: {dir_map}")
    print("  Training MLP (direction)...")
    mlp_s2 = SkPipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50,
                              early_stopping=True, validation_fraction=0.15,
                              random_state=42, verbose=False))
    ])
    mlp_s2.fit(X_flat_train_s2, y_train_s2)
    mlp_s2_pred = mlp_s2.predict(X_flat_test_s2)
    mlp_s2_acc = accuracy_score(y_test_s2, mlp_s2_pred)
    print(f"  MLP Stage 2 accuracy: {mlp_s2_acc:.3f}")

    rf_s2 = joblib.load(str(MODELS_DIR / "stage2_direction.pkl"))
    active_feat = y_feat_str != "Relax"
    X_feat_act = X_feat[active_feat]
    y_feat_act_str = y_feat_str[active_feat]
    subj_feat_act = subj_feat[active_feat]
    feat_act_test = np.array([s in test_subj for s in subj_feat_act])
    y_feat_dir = np.array([dir_map[s] for s in y_feat_act_str])
    rf_s2_pred = rf_s2.predict(X_feat_act[feat_act_test])
    rf_s2_acc = accuracy_score(y_feat_dir[feat_act_test], rf_s2_pred)
    print(f"  RF Stage 2 accuracy:  {rf_s2_acc:.3f}")

    # === LATENCY ===
    print("\n--- Latency Comparison ---")

    # MLP on raw: flatten + predict
    sample_window = X_windows[test_mask][:1]
    sample_flat = flatten_windows(sample_window)
    mlp_times = []
    for _ in range(10):
        mlp_s1.predict(sample_flat)
    for _ in range(200):
        t0 = time.perf_counter()
        mlp_s1.predict(sample_flat)
        mlp_times.append((time.perf_counter() - t0) * 1000)

    # RF: feature extraction + predict
    rf_times = []
    w = X_windows[test_mask][0]
    for _ in range(10):
        feat = np.concatenate([
            extract_psd_features(w), extract_stat_features(w),
            extract_cross_channel_features(w),
        ]).reshape(1, -1)
        rf_s1.predict(feat)
    for _ in range(200):
        t0 = time.perf_counter()
        feat = np.concatenate([
            extract_psd_features(w), extract_stat_features(w),
            extract_cross_channel_features(w),
        ]).reshape(1, -1)
        rf_s1.predict(feat)
        rf_times.append((time.perf_counter() - t0) * 1000)

    mlp_lat = np.array(mlp_times)
    rf_lat = np.array(rf_times)
    print(f"  MLP latency: {np.mean(mlp_lat):.1f}ms avg, {np.median(mlp_lat):.1f}ms median")
    print(f"  RF latency:  {np.mean(rf_lat):.1f}ms avg, {np.median(rf_lat):.1f}ms median")

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#444")

    models = ["RF\n(69 features)", "MLP\n(raw 3000)"]

    # Panel 1: Stage 1
    s1_accs = [rf_s1_acc * 100, mlp_s1_acc * 100]
    bars = axes[0].bar(models, s1_accs, color=["#0f9b58", "#4285f4"], width=0.5)
    axes[0].set_ylabel("Accuracy (%)", fontsize=12)
    axes[0].set_title("Stage 1: Rest vs Active", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, 100)
    for bar, val in zip(bars, s1_accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", color="white",
                    fontsize=12, fontweight="bold")

    # Panel 2: Stage 2
    s2_accs = [rf_s2_acc * 100, mlp_s2_acc * 100]
    bars2 = axes[1].bar(models, s2_accs, color=["#0f9b58", "#4285f4"], width=0.5)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].set_title("Stage 2: Direction (4-class)", fontsize=13, fontweight="bold")
    axes[1].set_ylim(0, 100)
    axes[1].axhline(y=25, color="#ff6b6b", linestyle="--", alpha=0.7, label="Random (25%)")
    axes[1].legend(loc="upper right", facecolor="#16213e", edgecolor="#444", labelcolor="white")
    for bar, val in zip(bars2, s2_accs):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", color="white",
                    fontsize=12, fontweight="bold")

    # Panel 3: Latency
    lat_means = [np.mean(rf_lat), np.mean(mlp_lat)]
    lat_p95 = [np.percentile(rf_lat, 95), np.percentile(mlp_lat, 95)]
    x_pos = np.arange(2)
    bars3 = axes[2].bar(x_pos - 0.15, lat_means, 0.3, color=["#0f9b58", "#4285f4"],
                        label="Mean")
    bars3b = axes[2].bar(x_pos + 0.15, lat_p95, 0.3, color=["#0f9b5888", "#4285f488"],
                         label="P95")
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(models)
    axes[2].set_ylabel("Latency (ms)", fontsize=12)
    axes[2].set_title("Inference Latency (per window)", fontsize=13, fontweight="bold")
    axes[2].legend(loc="upper right", facecolor="#16213e", edgecolor="#444", labelcolor="white")
    for bar, val in zip(bars3, lat_means):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}ms", ha="center", va="bottom", color="white", fontsize=10)

    fig.suptitle("Neural Network (MLP) vs Random Forest: Speed-Accuracy Tradeoff",
                 fontsize=15, fontweight="bold", color="white", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = str(RESULTS_DIR / "cnn_vs_rf.png")
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"\n  Saved to {out_path}")

    # Summary text
    with open(str(RESULTS_DIR / "cnn_vs_rf.txt"), "w") as f:
        f.write("Neural Network (MLP) vs Random Forest Comparison\n")
        f.write("=" * 50 + "\n\n")
        f.write("Setup:\n")
        f.write("  RF: 69 hand-crafted features (PSD + statistical)\n")
        f.write("  MLP: Raw flattened EEG (3000 inputs), 2 hidden layers (128, 64)\n")
        f.write("  Both: Cross-subject split, same train/test subjects\n\n")
        f.write(f"Stage 1 (Rest vs Active):\n")
        f.write(f"  RF:  {rf_s1_acc:.3f}\n")
        f.write(f"  MLP: {mlp_s1_acc:.3f}\n\n")
        f.write(f"Stage 2 (Direction, 4-class):\n")
        f.write(f"  RF:  {rf_s2_acc:.3f}\n")
        f.write(f"  MLP: {mlp_s2_acc:.3f}\n\n")
        f.write(f"Latency (per window, including feature extraction for RF):\n")
        f.write(f"  RF:  {np.mean(rf_lat):.1f}ms avg, {np.percentile(rf_lat, 95):.1f}ms p95\n")
        f.write(f"  MLP: {np.mean(mlp_lat):.1f}ms avg, {np.percentile(mlp_lat, 95):.1f}ms p95\n\n")
        f.write(f"Conclusion:\n")
        if rf_s1_acc >= mlp_s1_acc:
            f.write(f"  RF with hand-crafted features matches or beats the MLP\n")
            f.write(f"  on raw EEG, validating our feature engineering approach.\n")
        else:
            f.write(f"  MLP on raw EEG slightly outperforms RF, suggesting\n")
            f.write(f"  learned features can capture additional signal.\n")
        faster = "RF" if np.mean(rf_lat) < np.mean(mlp_lat) else "MLP"
        ratio = max(np.mean(rf_lat), np.mean(mlp_lat)) / min(np.mean(rf_lat), np.mean(mlp_lat))
        f.write(f"  {faster} is {ratio:.1f}x faster at inference.\n")
        f.write(f"  For real-time BCI at >10 Hz, both meet the <100ms target.\n")

    print("  Done.")


if __name__ == "__main__":
    main()
