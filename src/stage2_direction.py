"""Phase 5: Stage 2 direction classifier -- 4 active classes."""
import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def train_stage2():
    """Train and evaluate Stage 2 direction classifier (active classes only)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load features
    data = np.load(str(PROJECT_ROOT / "features.npz"), allow_pickle=True)
    X = data["X"]
    y_str = data["y"]
    subjects = data["subjects"]

    # Load subject split from Stage 1 for consistency
    split = np.load(str(PROJECT_ROOT / "subject_split.npz"), allow_pickle=True)
    train_subjects = set(split["train_subjects"])
    test_subjects = set(split["test_subjects"])

    # Filter to active-only (exclude Relax)
    active_mask = y_str != "Relax"
    X_active = X[active_mask]
    y_active_str = y_str[active_mask]
    subjects_active = subjects[active_mask]

    # Build direction label map dynamically from actual labels
    unique_active_labels = sorted(set(y_active_str))
    LABEL_MAP_DIRECTION = {label: idx for idx, label in enumerate(unique_active_labels)}
    print(f"Direction label map: {LABEL_MAP_DIRECTION}")

    y_direction = np.array([LABEL_MAP_DIRECTION[s] for s in y_active_str])
    class_names = list(LABEL_MAP_DIRECTION.keys())

    # Save label map for pipeline use
    np.savez(
        str(PROJECT_ROOT / "direction_label_map.npz"),
        labels=np.array(class_names),
        indices=np.array(list(range(len(class_names)))),
    )

    # Cross-subject split
    train_mask = np.array([s in train_subjects for s in subjects_active])
    test_mask = np.array([s in test_subjects for s in subjects_active])

    X_train, y_train = X_active[train_mask], y_direction[train_mask]
    X_test, y_test = X_active[test_mask], y_direction[test_mask]
    subjects_train = subjects_active[train_mask]

    print(f"\nActive samples only:")
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
    for name, idx in LABEL_MAP_DIRECTION.items():
        print(f"    {name} ({idx}): train={np.sum(y_train==idx)}, test={np.sum(y_test==idx)}")

    # Models
    models = {
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42))
        ]),
    }

    # Cross-validation
    print("\n--- Cross-Validation (GroupKFold) ---")
    n_unique_train = len(set(subjects_train))
    gkf = GroupKFold(n_splits=min(5, n_unique_train))
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train, cv=gkf, groups=subjects_train, scoring="accuracy"
        )
        cv_results[name] = scores
        print(f"  {name}: {scores.mean():.3f} +/- {scores.std():.3f}")

    # Test set evaluation
    print("\n--- Test Set Evaluation (4-class) ---")
    test_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        test_results[name] = (acc, model, y_pred)
        print(f"  {name}: {acc:.3f}")

    best_name = max(test_results, key=lambda k: test_results[k][0])
    best_acc, best_model, best_pred = test_results[best_name]
    y_test_final = y_test

    print(f"\nBest model: {best_name} (accuracy: {best_acc:.3f})")
    print(f"  4-class: Both Fists->FORWARD, Left Fist->LEFT, Right Fist->RIGHT, Tongue Tapping->BACKWARD")

    # Save model
    model_path = MODELS_DIR / "stage2_direction.pkl"
    joblib.dump(best_model, str(model_path))
    print(f"Saved to {model_path}")

    # Classification report
    report = classification_report(y_test_final, best_pred, target_names=class_names)
    print(f"\nClassification Report:\n{report}")

    with open(str(RESULTS_DIR / "stage2_report.txt"), "w") as f:
        f.write(f"Stage 2 Direction Classifier: {best_name}\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"4-class: Both Fists->FORWARD, Left Fist->LEFT, Right Fist->RIGHT, Tongue Tapping->BACKWARD\n")
        f.write(f"Test Accuracy: {best_acc:.3f}\n\n")
        f.write(f"Cross-validation (4-class):\n")
        for name, scores in cv_results.items():
            f.write(f"  {name}: {scores.mean():.3f} +/- {scores.std():.3f}\n")
        f.write(f"\nClassification Report:\n{report}\n")

    # Confusion matrix
    cm = confusion_matrix(y_test_final, best_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Stage 2: {best_name} (4-class, Acc: {best_acc:.3f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "stage2_confusion.png"), dpi=150)
    plt.close()
    print(f"Confusion matrix saved to results/stage2_confusion.png")

    return best_model, best_acc, class_names


if __name__ == "__main__":
    train_stage2()
