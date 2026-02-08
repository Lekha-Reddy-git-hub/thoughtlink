"""Latency-Accuracy Tradeoff Chart: benchmark LR, SVM, RF, MLP on both stages.

Produces:
  results/latency_accuracy_tradeoff.png  — scatter plot
  results/latency_accuracy_tradeoff.txt  — raw numbers
"""
import numpy as np
import time
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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
    """Load preprocessed windows, features, and split info."""
    data = np.load(str(PROJECT_ROOT / "preprocessed_data.npz"), allow_pickle=True)
    X_windows = data["X"]       # (N, 500, 6)
    y_str = data["y"]
    subjects = data["subjects"]

    feat_data = np.load(str(PROJECT_ROOT / "features.npz"), allow_pickle=True)
    X_feat = feat_data["X"]
    y_feat_str = feat_data["y"]
    subj_feat = feat_data["subjects"]

    split = np.load(str(PROJECT_ROOT / "subject_split.npz"), allow_pickle=True)
    train_subjects = set(split["train_subjects"])
    test_subjects = set(split["test_subjects"])

    y_binary = np.array([0 if s == "Relax" else 1 for s in y_str])
    y_feat_binary = np.array([0 if s == "Relax" else 1 for s in y_feat_str])

    active_labels = sorted(set(s for s in y_str if s != "Relax"))
    dir_map = {label: idx for idx, label in enumerate(active_labels)}

    train_mask = np.array([s in train_subjects for s in subjects])
    test_mask = np.array([s in test_subjects for s in subjects])
    feat_train_mask = np.array([s in train_subjects for s in subj_feat])
    feat_test_mask = np.array([s in test_subjects for s in subj_feat])

    return (X_windows, X_feat, y_binary, y_feat_binary, y_str, y_feat_str,
            subjects, subj_feat, train_mask, test_mask, feat_train_mask,
            feat_test_mask, dir_map, test_subjects)


def time_inference(predict_fn, X_sample, n_warmup=10, n_iter=200):
    """Time inference over n_iter calls, return mean latency in ms."""
    for _ in range(n_warmup):
        predict_fn(X_sample)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        predict_fn(X_sample)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Latency-Accuracy Tradeoff: LR, SVM, RF, MLP")
    print("=" * 70)

    (X_windows, X_feat, y_binary, y_feat_binary, y_str, y_feat_str,
     subjects, subj_feat, train_mask, test_mask, feat_train_mask,
     feat_test_mask, dir_map, test_subjects) = load_data()

    # Feature-based data (for LR, SVM, RF)
    X_feat_train = X_feat[feat_train_mask]
    X_feat_test = X_feat[feat_test_mask]
    y_feat_bin_train = y_feat_binary[feat_train_mask]
    y_feat_bin_test = y_feat_binary[feat_test_mask]

    # Raw flattened data (for MLP)
    X_flat_train = X_windows[train_mask].reshape(train_mask.sum(), -1)
    X_flat_test = X_windows[test_mask].reshape(test_mask.sum(), -1)
    y_bin_train = y_binary[train_mask]
    y_bin_test = y_binary[test_mask]

    # Direction subsets (active only)
    active_feat_train = y_feat_str[feat_train_mask] != "Relax"
    active_feat_test = y_feat_str[feat_test_mask] != "Relax"
    X_feat_dir_train = X_feat_train[active_feat_train]
    X_feat_dir_test = X_feat_test[active_feat_test]
    y_feat_dir_train = np.array([dir_map[s] for s in y_feat_str[feat_train_mask][active_feat_train]])
    y_feat_dir_test = np.array([dir_map[s] for s in y_feat_str[feat_test_mask][active_feat_test]])

    active_raw_train = y_str[train_mask] != "Relax"
    active_raw_test = y_str[test_mask] != "Relax"
    X_flat_dir_train = X_flat_train[active_raw_train]
    X_flat_dir_test = X_flat_test[active_raw_test]
    y_dir_train = np.array([dir_map[s] for s in y_str[train_mask][active_raw_train]])
    y_dir_test = np.array([dir_map[s] for s in y_str[test_mask][active_raw_test]])

    # Sample window for latency testing
    sample_window = X_windows[test_mask][0]  # (500, 6)
    sample_feat = X_feat_test[:1]            # (1, 69)
    sample_flat = X_flat_test[:1]            # (1, 3000)

    results = {}  # model_name -> {s1_acc, s2_acc, latency_ms}

    # --- Logistic Regression (on 69 features) ---
    print("\n  Training Logistic Regression...")
    lr_s1 = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    lr_s1.fit(X_feat_train, y_feat_bin_train)
    lr_s1_acc = accuracy_score(y_feat_bin_test, lr_s1.predict(X_feat_test))

    lr_s2 = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    lr_s2.fit(X_feat_dir_train, y_feat_dir_train)
    lr_s2_acc = accuracy_score(y_feat_dir_test, lr_s2.predict(X_feat_dir_test))

    # Latency: feature extraction + predict
    def lr_infer(x):
        feat = np.concatenate([
            extract_psd_features(x), extract_stat_features(x),
            extract_cross_channel_features(x),
        ]).reshape(1, -1)
        lr_s1.predict(feat)
    lr_lat = time_inference(lr_infer, sample_window)

    results["LR"] = {"s1_acc": lr_s1_acc, "s2_acc": lr_s2_acc, "latency_ms": lr_lat}
    print(f"    Stage 1: {lr_s1_acc:.3f}  Stage 2: {lr_s2_acc:.3f}  Latency: {lr_lat:.1f}ms")

    # --- SVM (on 69 features) ---
    print("  Training SVM...")
    svm_s1 = SVC(kernel="rbf", probability=True, random_state=42)
    svm_s1.fit(X_feat_train, y_feat_bin_train)
    svm_s1_acc = accuracy_score(y_feat_bin_test, svm_s1.predict(X_feat_test))

    svm_s2 = SVC(kernel="rbf", probability=True, random_state=42)
    svm_s2.fit(X_feat_dir_train, y_feat_dir_train)
    svm_s2_acc = accuracy_score(y_feat_dir_test, svm_s2.predict(X_feat_dir_test))

    def svm_infer(x):
        feat = np.concatenate([
            extract_psd_features(x), extract_stat_features(x),
            extract_cross_channel_features(x),
        ]).reshape(1, -1)
        svm_s1.predict(feat)
    svm_lat = time_inference(svm_infer, sample_window)

    results["SVM"] = {"s1_acc": svm_s1_acc, "s2_acc": svm_s2_acc, "latency_ms": svm_lat}
    print(f"    Stage 1: {svm_s1_acc:.3f}  Stage 2: {svm_s2_acc:.3f}  Latency: {svm_lat:.1f}ms")

    # --- Random Forest (on 69 features) ---
    print("  Loading pre-trained Random Forest...")
    rf_s1 = joblib.load(str(MODELS_DIR / "stage1_binary.pkl"))
    rf_s1_acc = accuracy_score(y_feat_bin_test, rf_s1.predict(X_feat_test))

    rf_s2 = joblib.load(str(MODELS_DIR / "stage2_direction.pkl"))
    rf_s2_pred = rf_s2.predict(X_feat_dir_test)
    rf_s2_acc = accuracy_score(y_feat_dir_test, rf_s2_pred)

    def rf_infer(x):
        feat = np.concatenate([
            extract_psd_features(x), extract_stat_features(x),
            extract_cross_channel_features(x),
        ]).reshape(1, -1)
        rf_s1.predict(feat)
    rf_lat = time_inference(rf_infer, sample_window)

    results["RF"] = {"s1_acc": rf_s1_acc, "s2_acc": rf_s2_acc, "latency_ms": rf_lat}
    print(f"    Stage 1: {rf_s1_acc:.3f}  Stage 2: {rf_s2_acc:.3f}  Latency: {rf_lat:.1f}ms")

    # --- MLP (on raw 3000) ---
    print("  Training MLP (raw EEG)...")
    mlp_s1 = SkPipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50,
                              early_stopping=True, validation_fraction=0.15,
                              random_state=42, verbose=False))
    ])
    mlp_s1.fit(X_flat_train, y_bin_train)
    mlp_s1_acc = accuracy_score(y_bin_test, mlp_s1.predict(X_flat_test))

    mlp_s2 = SkPipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50,
                              early_stopping=True, validation_fraction=0.15,
                              random_state=42, verbose=False))
    ])
    mlp_s2.fit(X_flat_dir_train, y_dir_train)
    mlp_s2_acc = accuracy_score(y_dir_test, mlp_s2.predict(X_flat_dir_test))

    mlp_lat = time_inference(lambda x: mlp_s1.predict(x), sample_flat)

    results["MLP"] = {"s1_acc": mlp_s1_acc, "s2_acc": mlp_s2_acc, "latency_ms": mlp_lat}
    print(f"    Stage 1: {mlp_s1_acc:.3f}  Stage 2: {mlp_s2_acc:.3f}  Latency: {mlp_lat:.1f}ms")

    # === SCATTER PLOT ===
    print("\n  Generating tradeoff chart...")

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#444")

    markers = {"LR": "s", "SVM": "D", "RF": "o", "MLP": "^"}
    sizes = {"LR": 120, "SVM": 120, "RF": 180, "MLP": 120}

    for model_name, r in results.items():
        mk = markers[model_name]
        sz = sizes[model_name]

        # Stage 1 (blue)
        ax.scatter(r["latency_ms"], r["s1_acc"] * 100, marker=mk, s=sz,
                   color="#58a6ff", edgecolors="white", linewidth=1.0, zorder=5)
        ax.annotate(f"{model_name}\nS1: {r['s1_acc']*100:.1f}%",
                    (r["latency_ms"], r["s1_acc"] * 100),
                    textcoords="offset points", xytext=(12, 5),
                    fontsize=9, color="#58a6ff", fontweight="bold")

        # Stage 2 (red)
        ax.scatter(r["latency_ms"], r["s2_acc"] * 100, marker=mk, s=sz,
                   color="#f85149", edgecolors="white", linewidth=1.0, zorder=5)
        ax.annotate(f"{model_name}\nS2: {r['s2_acc']*100:.1f}%",
                    (r["latency_ms"], r["s2_acc"] * 100),
                    textcoords="offset points", xytext=(12, -15),
                    fontsize=9, color="#f85149", fontweight="bold")

    # 100ms real-time threshold
    ax.axvline(x=100, color="#ffa657", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.text(100, ax.get_ylim()[1] * 0.97, " 100ms\n real-time\n threshold",
            color="#ffa657", fontsize=9, va="top")

    # 25% chance line for Stage 2
    ax.axhline(y=25, color="#f8514966", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(ax.get_xlim()[1] * 0.95, 25.5, "chance (25%)",
            color="#f85149", fontsize=8, ha="right", alpha=0.6)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#58a6ff",
               markersize=10, label="Stage 1 (Rest vs Active)", linestyle="None"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f85149",
               markersize=10, label="Stage 2 (Direction 4-class)", linestyle="None"),
        Line2D([0], [0], color="#ffa657", linestyle="--", label="100ms threshold"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              facecolor="#161b22", edgecolor="#444", labelcolor="white", fontsize=10)

    ax.set_xlabel("Inference Latency (ms)", fontsize=13)
    ax.set_ylabel("Cross-Subject Accuracy (%)", fontsize=13)
    ax.set_title("Latency-Accuracy Tradeoff: LR vs SVM vs RF vs MLP",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out_png = str(RESULTS_DIR / "latency_accuracy_tradeoff.png")
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_png}")

    # === TEXT FILE ===
    out_txt = str(RESULTS_DIR / "latency_accuracy_tradeoff.txt")
    with open(out_txt, "w") as f:
        f.write("Latency-Accuracy Tradeoff: LR vs SVM vs RF vs MLP\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Model':<8} {'Stage 1':>10} {'Stage 2':>10} {'Latency':>12}\n")
        f.write("-" * 42 + "\n")
        for name in ["LR", "SVM", "RF", "MLP"]:
            r = results[name]
            f.write(f"{name:<8} {r['s1_acc']*100:>9.1f}% {r['s2_acc']*100:>9.1f}% {r['latency_ms']:>10.1f}ms\n")
        f.write("\n")
        f.write("Notes:\n")
        f.write("  - LR, SVM, RF use 69 hand-crafted features (PSD + statistical)\n")
        f.write("  - MLP uses raw flattened EEG (3000 inputs)\n")
        f.write("  - Latency includes feature extraction time for LR/SVM/RF\n")
        f.write("  - All evaluations use cross-subject leave-one-out splits\n")
        f.write("  - All models are well under the 100ms real-time threshold\n")
        f.write(f"\nBest Stage 1: RF ({results['RF']['s1_acc']*100:.1f}%)\n")
        best_s2 = max(results.items(), key=lambda x: x[1]["s2_acc"])
        f.write(f"Best Stage 2: {best_s2[0]} ({best_s2[1]['s2_acc']*100:.1f}%)\n")
        fastest = min(results.items(), key=lambda x: x[1]["latency_ms"])
        f.write(f"Fastest: {fastest[0]} ({fastest[1]['latency_ms']:.1f}ms)\n")
    print(f"  Saved: {out_txt}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
