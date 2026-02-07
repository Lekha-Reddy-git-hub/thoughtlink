"""Phase 2: Preprocessing pipeline -- filtering, segmentation, windowing."""
import os
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Channel names in order
CHANNEL_NAMES = ["AFF6", "AFp2", "AFp1", "AFF5", "FCz", "CPz"]
FS = 500.0  # Sampling rate


def bandpass_filter(data, low=8.0, high=30.0, fs=500.0, order=4):
    """Bandpass filter EEG data. data: (n_samples, n_channels)."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    filtered = filtfilt(b, a, data, axis=0)
    return filtered


def extract_active_segment(eeg, duration, fs=500.0, stim_onset_s=3.0):
    """Extract the stimulus-active portion of EEG."""
    start_sample = int(stim_onset_s * fs)
    end_sample = start_sample + int(duration * fs)
    end_sample = min(end_sample, eeg.shape[0])
    return eeg[start_sample:end_sample]


def normalize_channels(data):
    """Zero-mean, unit-variance per channel."""
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (data - mean) / std


def segment_windows(data, window_size=500, overlap=250):
    """Segment data into overlapping windows."""
    step = window_size - overlap
    windows = []
    for start in range(0, data.shape[0] - window_size + 1, step):
        windows.append(data[start:start + window_size])
    return windows


def preprocess_file(fpath, window_size=500, overlap=250):
    """
    Full preprocessing for one .npz file.
    Returns: (windows_list, label_str, subject_id) or None if file is bad.
    """
    arr = np.load(str(fpath), allow_pickle=True)
    eeg_raw = arr["feature_eeg"]  # (7499, 6)
    label_info = arr["label"].item()
    label_str = label_info["label"]
    subject_id = label_info["subject_id"]
    duration = label_info["duration"]

    # Step 1: Bandpass filter
    eeg_filtered = bandpass_filter(eeg_raw, low=8.0, high=30.0, fs=FS)

    # Check for NaN/Inf
    if np.any(np.isnan(eeg_filtered)) or np.any(np.isinf(eeg_filtered)):
        print(f"  WARNING: NaN/Inf in {fpath.name}, skipping.")
        return None

    # Step 2: Extract active segment
    eeg_active = extract_active_segment(eeg_filtered, duration, fs=FS)

    # Step 3: Normalize
    eeg_norm = normalize_channels(eeg_active)

    # Step 4: Segment into windows
    windows = segment_windows(eeg_norm, window_size, overlap)

    # Edge case: very short recordings
    if len(windows) == 0:
        if eeg_norm.shape[0] > 0:
            # Pad to window_size
            padded = np.zeros((window_size, eeg_norm.shape[1]))
            padded[:eeg_norm.shape[0]] = eeg_norm
            windows = [padded]
            print(f"  WARNING: Short recording in {fpath.name}, zero-padded.")
        else:
            print(f"  WARNING: Empty active segment in {fpath.name}, skipping.")
            return None

    return windows, label_str, subject_id


def preprocess_all(window_size=500, overlap=250):
    """Process all .npz files and save preprocessed data."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"Processing {len(files)} files...")

    all_windows = []
    all_labels = []
    all_subjects = []
    skipped = 0

    for i, fpath in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(files)}]...")

        result = preprocess_file(fpath, window_size, overlap)
        if result is None:
            skipped += 1
            continue

        windows, label_str, subject_id = result
        for w in windows:
            all_windows.append(w)
            all_labels.append(label_str)
            all_subjects.append(subject_id)

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels)
    subjects = np.array(all_subjects)

    print(f"\nPreprocessing complete:")
    print(f"  Total windows: {X.shape[0]}")
    print(f"  Window shape: {X.shape[1:]}")
    print(f"  Skipped files: {skipped}")
    print(f"  Unique labels: {np.unique(y)}")
    print(f"  Unique subjects: {np.unique(subjects)}")

    # Save
    out_path = PROJECT_ROOT / "preprocessed_data.npz"
    np.savez_compressed(str(out_path), X=X, y=y, subjects=subjects)
    print(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    return X, y, subjects


def verify_psd(sample_file=None):
    """Generate PSD verification plot: raw vs filtered."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if sample_file is None:
        files = sorted(DATA_DIR.glob("*.npz"))
        sample_file = files[0]

    arr = np.load(str(sample_file), allow_pickle=True)
    eeg_raw = arr["feature_eeg"]
    eeg_filtered = bandpass_filter(eeg_raw, low=8.0, high=30.0, fs=FS)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"PSD Verification: {sample_file.name}\nRaw (blue) vs Filtered 8-30Hz (orange)")

    for ch in range(6):
        ax = axes[ch // 3, ch % 3]

        freqs_raw, psd_raw = welch(eeg_raw[:, ch], fs=FS, nperseg=1024)
        freqs_filt, psd_filt = welch(eeg_filtered[:, ch], fs=FS, nperseg=1024)

        ax.semilogy(freqs_raw, psd_raw, label="Raw", alpha=0.7)
        ax.semilogy(freqs_filt, psd_filt, label="Filtered 8-30Hz", alpha=0.7)
        ax.axvline(8, color="gray", linestyle="--", alpha=0.5, label="8 Hz")
        ax.axvline(30, color="gray", linestyle="--", alpha=0.5, label="30 Hz")
        ax.set_title(f"Ch {ch}: {CHANNEL_NAMES[ch]}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (uV^2/Hz)")
        ax.set_xlim(0, 60)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = RESULTS_DIR / "psd_verification.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"PSD verification saved to {out_path}")


if __name__ == "__main__":
    verify_psd()
    X, y, subjects = preprocess_all()
