"""Phase 3: Feature extraction from preprocessed EEG windows."""
import numpy as np
from pathlib import Path
from scipy.signal import welch
from scipy import stats as scipy_stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FS = 500.0


def extract_psd_features(window, fs=500.0):
    """
    Extract PSD band-power features per channel.
    window: (n_samples, 6)
    Returns: (24,) feature vector
    """
    features = []
    for ch in range(window.shape[1]):
        freqs, psd = welch(window[:, ch], fs=fs, nperseg=min(256, window.shape[0]))

        theta = np.mean(psd[(freqs >= 4) & (freqs < 8)])
        alpha = np.mean(psd[(freqs >= 8) & (freqs < 13)])
        beta = np.mean(psd[(freqs >= 13) & (freqs < 30)])
        alpha_beta = alpha / (beta + 1e-10)

        features.extend([theta, alpha, beta, alpha_beta])

    return np.array(features, dtype=np.float64)


def extract_stat_features(window):
    """
    Extract statistical features per channel.
    window: (n_samples, 6)
    Returns: (42,) feature vector
    """
    features = []
    for ch in range(window.shape[1]):
        signal = window[:, ch]
        features.extend([
            np.var(signal),
            np.mean(np.abs(signal)),
            np.sqrt(np.mean(signal ** 2)),
            float(np.max(np.abs(signal))),
            float(scipy_stats.kurtosis(signal)),
            float(scipy_stats.skew(signal)),
            np.sum(np.diff(np.sign(signal)) != 0),
        ])
    return np.array(features, dtype=np.float64)


def extract_cross_channel_features(window):
    """
    Extract inter-channel asymmetry features.
    window: (n_samples, 6)
    Channels: 0=AFF6(R), 1=AFp2(R), 2=AFp1(L), 3=AFF5(L), 4=FCz, 5=CPz
    Returns: (3,) feature vector
    """
    features = [
        np.var(window[:, 3]) - np.var(window[:, 0]),  # AFF5(L) - AFF6(R)
        np.var(window[:, 2]) - np.var(window[:, 1]),  # AFp1(L) - AFp2(R)
        np.var(window[:, 4]) - np.var(window[:, 5]),  # FCz - CPz
    ]
    return np.array(features, dtype=np.float64)


def extract_all_features(window):
    """Extract all features from a single window. Returns (69,) vector."""
    return np.concatenate([
        extract_psd_features(window),
        extract_stat_features(window),
        extract_cross_channel_features(window),
    ])


def build_feature_matrix():
    """Load preprocessed data, extract features for all windows, save."""
    data_path = PROJECT_ROOT / "preprocessed_data.npz"
    print(f"Loading preprocessed data from {data_path}...")
    data = np.load(str(data_path), allow_pickle=True)
    X_windows = data["X"]  # (n_windows, 500, 6)
    y = data["y"]           # (n_windows,) string labels
    subjects = data["subjects"]  # (n_windows,) subject IDs

    n_windows = X_windows.shape[0]
    print(f"Extracting features from {n_windows} windows...")

    # Pre-allocate
    sample_feat = extract_all_features(X_windows[0])
    n_features = len(sample_feat)
    print(f"  Features per window: {n_features}")

    X_features = np.zeros((n_windows, n_features), dtype=np.float64)

    for i in range(n_windows):
        if (i + 1) % 2000 == 0:
            print(f"  [{i+1}/{n_windows}]...")
        X_features[i] = extract_all_features(X_windows[i])

    # Check for NaN/Inf
    nan_count = np.sum(np.isnan(X_features))
    inf_count = np.sum(np.isinf(X_features))
    if nan_count > 0 or inf_count > 0:
        print(f"  WARNING: {nan_count} NaN, {inf_count} Inf values found. Replacing with 0.")
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

    # Feature stats
    print(f"\nFeature matrix shape: {X_features.shape}")
    print(f"  Mean per feature: min={X_features.mean(axis=0).min():.4f}, max={X_features.mean(axis=0).max():.4f}")
    print(f"  Std per feature: min={X_features.std(axis=0).min():.4f}, max={X_features.std(axis=0).max():.4f}")

    # Save
    out_path = PROJECT_ROOT / "features.npz"
    np.savez_compressed(str(out_path), X=X_features, y=y, subjects=subjects)
    print(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    return X_features, y, subjects


if __name__ == "__main__":
    X_features, y, subjects = build_feature_matrix()
