"""Phase 4: Stage 1 binary classifier -- Rest vs Active."""
import os
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


def to_binary(label_str):
    """Map label string to binary: 0=Rest, 1=Active."""
    return 0 if label_str == "Relax" else 1


def train_stage1():
    """Train and evaluate Stage 1 binary classifier."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load features
    data = np.load(str(PROJECT_ROOT / "features.npz"), allow_pickle=True)
    X = data["X"]
    y_str = data["y"]
    subjects = data["subjects"]

    # Binary labels
    y_binary = np.array([to_binary(s) for s in y_str])
    print(f"Total samples: {len(y_binary)}")
    print(f"  Rest (0): {np.sum(y_binary == 0)}")
    print(f"  Active (1): {np.sum(y_binary == 1)}")

    # Cross-subject split: hold out ~20% of subjects
    unique_subjects = sorted(set(subjects))
    n_test = max(1, len(unique_subjects) // 5)
    test_subjects = set(unique_subjects[-n_test:])
    train_subjects = set(unique_subjects[:-n_test])

    print(f"\nSubject split:")
    print(f"  Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"  Test subjects ({len(test_subjects)}): {sorted(test_subjects)}")

    train_mask = np.array([s in train_subjects for s in subjects])
    test_mask = np.array([s in test_subjects for s in subjects])

    X_train, y_train = X[train_mask], y_binary[train_mask]
    X_test, y_test = X[test_mask], y_binary[test_mask]
    subjects_train = subjects[train_mask]

    print(f"  Train samples: {len(y_train)} (Rest: {np.sum(y_train==0)}, Active: {np.sum(y_train==1)})")
    print(f"  Test samples: {len(y_test)} (Rest: {np.sum(y_test==0)}, Active: {np.sum(y_test==1)})")

    # Verify no subject leakage
    assert len(set(subjects[train_mask]) & set(subjects[test_mask])) == 0, "Subject leakage!"

    # Models to try
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

    # 5-fold GroupKFold cross-validation on training set
    print("\n--- Cross-Validation (5-fold GroupKFold) ---")
    gkf = GroupKFold(n_splits=min(5, len(train_subjects)))
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train, cv=gkf, groups=subjects_train, scoring="accuracy"
        )
        cv_results[name] = scores
        print(f"  {name}: {scores.mean():.3f} +/- {scores.std():.3f}")

    # Train on full training set and evaluate on test set
    print("\n--- Test Set Evaluation ---")
    test_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        test_results[name] = (acc, model, y_pred)
        print(f"  {name}: {acc:.3f}")

    # Pick best model
    best_name = max(test_results, key=lambda k: test_results[k][0])
    best_acc, best_model, best_pred = test_results[best_name]
    print(f"\nBest model: {best_name} (accuracy: {best_acc:.3f})")

    # Save best model
    model_path = MODELS_DIR / "stage1_binary.pkl"
    joblib.dump(best_model, str(model_path))
    print(f"Saved to {model_path}")

    # Classification report
    report = classification_report(y_test, best_pred, target_names=["Rest", "Active"])
    print(f"\nClassification Report:\n{report}")

    with open(str(RESULTS_DIR / "stage1_report.txt"), "w") as f:
        f.write(f"Stage 1 Binary Classifier: {best_name}\n")
        f.write(f"Test Accuracy: {best_acc:.3f}\n\n")
        f.write(f"Train subjects: {sorted(train_subjects)}\n")
        f.write(f"Test subjects: {sorted(test_subjects)}\n\n")
        f.write(f"Cross-validation results:\n")
        for name, scores in cv_results.items():
            f.write(f"  {name}: {scores.mean():.3f} +/- {scores.std():.3f}\n")
        f.write(f"\nTest set results:\n")
        for name, (acc, _, _) in test_results.items():
            f.write(f"  {name}: {acc:.3f}\n")
        f.write(f"\nBest: {best_name}\n\n")
        f.write(f"Classification Report:\n{report}\n")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rest", "Active"],
                yticklabels=["Rest", "Active"], ax=ax)
    ax.set_title(f"Stage 1: {best_name} (Acc: {best_acc:.3f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "stage1_confusion.png"), dpi=150)
    plt.close()
    print(f"Confusion matrix saved to results/stage1_confusion.png")

    # Save the train/test subject split for consistency with Stage 2
    np.savez(
        str(PROJECT_ROOT / "subject_split.npz"),
        train_subjects=np.array(sorted(train_subjects)),
        test_subjects=np.array(sorted(test_subjects)),
    )

    return best_model, best_acc


if __name__ == "__main__":
    train_stage1()
